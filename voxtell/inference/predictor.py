"""
VoxTell Predictor — optimized inference pipeline.

Optimization history
--------------------
v0  Original         : 145.25s  (FP32 text model on CPU, tile_step=0.5)
v1  FP16 + VRAM mgmt : 8.06s    (FP16 on GPU, VRAM swap, tile_step=0.75)
v2  GPU preprocess
    + disk cache     : 5.93s    (CUDA crop/norm, persistent embed cache)
v3  Numba preprocess
    + INT4 quant
    + batched window : TBD      (LLVM native code, 4-bit text model, batch=N patches)
"""

import hashlib
import pydoc
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import torch
from torch._dynamo import OptimizedModule
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from acvl_utils.cropping_and_padding.bounding_boxes import insert_crop_into_image
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from batchgenerators.utilities.file_and_folder_operations import join, load_json

from nnunetv2.inference.sliding_window_prediction import compute_gaussian, compute_steps_for_sliding_window
from nnunetv2.utilities.helpers import dummy_context, empty_cache

from voxtell.model.voxtell_model import VoxTellModel
from voxtell.utils.text_embedding import last_token_pool, wrap_with_instruction
from voxtell.utils.fast_preprocess import numba_crop_to_nonzero, numpy_zscore_normalize


# ── Persistent disk embedding cache ──────────────────────────────────────────
_EMBED_CACHE_DIR = Path.home() / ".cache" / "voxtell" / "embeddings"
_EMBED_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _prompt_cache_path(prompt: str, model_name: str) -> Path:
    key = hashlib.sha256(f"{model_name}::{prompt}".encode()).hexdigest()
    return _EMBED_CACHE_DIR / f"{key}.pt"


def _load_disk_cache(prompt: str, model_name: str) -> Union[torch.Tensor, None]:
    path = _prompt_cache_path(prompt, model_name)
    if path.exists():
        return torch.load(path, map_location="cpu", weights_only=True)
    return None


def _save_disk_cache(prompt: str, model_name: str, embedding: torch.Tensor) -> None:
    torch.save(embedding.cpu(), _prompt_cache_path(prompt, model_name))


# ── INT4 quantization loader ──────────────────────────────────────────────────

def _load_text_backbone(model_name: str, device: torch.device):
    """
    Load text backbone with INT4 quantization (bitsandbytes NF4) when available,
    falling back to FP16 on CPU.

    INT4 (NF4):  ~2GB VRAM, stays on GPU, faster cache-miss embedding.
    FP16 (CPU):  ~8GB when on GPU, must be offloaded after use.

    Returns (model, is_quantized: bool)
    """
    try:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,   # double quant saves ~0.4 bits extra
            bnb_4bit_quant_type="nf4",        # NF4: optimal for normally-distributed weights
        )
        model = AutoModel.from_pretrained(
            model_name, quantization_config=bnb_config
        ).eval()
        print(f"  [Text backbone] INT4 (NF4) quantization — ~2GB VRAM")
        return model, True
    except Exception as e:
        print(f"  [Text backbone] INT4 unavailable ({type(e).__name__}), using FP16 on CPU")
        model = AutoModel.from_pretrained(model_name, dtype=torch.float16).eval()
        return model, False


class VoxTellPredictor:
    """
    Optimized VoxTell predictor.

    Three-level optimizations applied
    ----------------------------------
    1. Numba JIT preprocessing   — LLVM-compiled native bbox search + NumPy C normalize
    2. INT4 text backbone        — bitsandbytes NF4 (2GB vs 8GB FP16); GPU-resident
    3. Batched sliding window    — N patches per forward pass instead of 1
    4. Two-level embedding cache — in-memory dict + persistent disk (SHA256-keyed)
    5. tile_step_size = 0.75     — 25% overlap, fewer patches
    """

    def __init__(
        self,
        model_dir: str,
        device: torch.device = torch.device("cuda"),
        text_encoding_model: str = "Qwen/Qwen3-Embedding-4B",
        sliding_window_batch_size: int = 1,
    ) -> None:
        self.device = device
        self.text_encoding_model = text_encoding_model
        self.sliding_window_batch_size = sliding_window_batch_size

        if device.type == "cuda":
            torch.backends.cudnn.benchmark = True

        self.tile_step_size = 0.75
        self.perform_everything_on_device = True

        # ── Text backbone ─────────────────────────────────────────────────────
        self.tokenizer = AutoTokenizer.from_pretrained(
            text_encoding_model, padding_side="left"
        )
        self.text_backbone, self._backbone_quantized = _load_text_backbone(
            text_encoding_model, device
        )
        self.max_text_length = 8192
        self._embed_cache: dict = {}

        # ── Segmentation network ──────────────────────────────────────────────
        plans = load_json(join(model_dir, "plans.json"))
        arch_kwargs = plans["configurations"]["3d_fullres"]["architecture"]["arch_kwargs"]
        self.patch_size = plans["configurations"]["3d_fullres"]["patch_size"]

        arch_kwargs = dict(**arch_kwargs)
        for key in plans["configurations"]["3d_fullres"]["architecture"]["_kw_requires_import"]:
            if arch_kwargs[key] is not None:
                arch_kwargs[key] = pydoc.locate(arch_kwargs[key])

        network = VoxTellModel(
            input_channels=1,
            **arch_kwargs,
            decoder_layer=4,
            text_embedding_dim=2560,
            num_maskformer_stages=5,
            num_heads=32,
            query_dim=2048,
            project_to_decoder_hidden_dim=2048,
            deep_supervision=False,
        )

        checkpoint = torch.load(
            join(model_dir, "fold_0", "checkpoint_final.pth"),
            map_location=torch.device("cpu"),
            weights_only=False,
        )
        if not isinstance(network, OptimizedModule):
            network.load_state_dict(checkpoint["network_weights"])
        else:
            network._orig_mod.load_state_dict(checkpoint["network_weights"])

        network.eval()
        self.network = network

    # ── Preprocessing — Numba JIT (LLVM native code) ──────────────────────────

    def preprocess(self, data: np.ndarray) -> Tuple[torch.Tensor, List, Tuple[int, ...]]:
        """
        Preprocess image using Numba JIT (LLVM) for bbox search and NumPy C for normalize.

        Numba compiles _find_nonzero_bbox to native machine code via LLVM on
        first call (warm-up done at import). All subsequent calls run at C speed.

        Args:
            data: Image array in RAS orientation, shape (H,W,D) or (C,H,W,D).

        Returns:
            (preprocessed_tensor, bbox, original_shape)
        """
        if data.ndim == 3:
            data = data[None]
        original_shape = data.shape[1:]

        data = data.astype(np.float32)

        # Numba JIT — LLVM native bounding-box search
        data, bbox = numba_crop_to_nonzero(data)

        # NumPy C backend — z-score normalization
        data = numpy_zscore_normalize(data)

        return torch.from_numpy(data), bbox, original_shape

    # ── Sliding window helpers ────────────────────────────────────────────────

    def _internal_get_sliding_window_slicers(self, image_size: Tuple[int, ...]) -> List[Tuple]:
        slicers = []
        if len(self.patch_size) < len(image_size):
            assert len(self.patch_size) == len(image_size) - 1
            steps = compute_steps_for_sliding_window(
                image_size[1:], self.patch_size, self.tile_step_size
            )
            for d in range(image_size[0]):
                for sx in steps[0]:
                    for sy in steps[1]:
                        slicers.append(
                            tuple([slice(None), d, *[slice(si, si + ti)
                                   for si, ti in zip((sx, sy), self.patch_size)]])
                        )
        else:
            steps = compute_steps_for_sliding_window(
                image_size, self.patch_size, self.tile_step_size
            )
            for sx in steps[0]:
                for sy in steps[1]:
                    for sz in steps[2]:
                        slicers.append(
                            tuple([slice(None), *[slice(si, si + ti)
                                   for si, ti in zip((sx, sy, sz), self.patch_size)]])
                        )
        return slicers

    # ── Text embedding — INT4/FP16 + two-level cache ──────────────────────────

    @torch.inference_mode()
    def embed_text_prompts(self, text_prompts: Union[List[str], str]) -> torch.Tensor:
        """
        Embed text prompts with two-level cache and INT4/FP16 backbone.

        Cache hierarchy
        ---------------
        1. In-memory dict  (O(1), per-session)
        2. Disk cache      (~/.cache/voxtell/embeddings/, persistent across sessions)
        3. INT4 GPU model  (only runs for genuinely new prompts)

        INT4 backbone stays on GPU at ~2GB (never offloaded).
        FP16 backbone starts on CPU, moves to GPU on cache-miss, then back.
        """
        if isinstance(text_prompts, str):
            text_prompts = [text_prompts]
        n_prompts = len(text_prompts)

        # Level 1 → Level 2: fill memory cache from disk
        for p in text_prompts:
            if p not in self._embed_cache:
                cached = _load_disk_cache(p, self.text_encoding_model)
                if cached is not None:
                    self._embed_cache[p] = cached

        # Level 3: run model for truly new prompts
        uncached = [p for p in text_prompts if p not in self._embed_cache]
        if uncached:
            if not self._backbone_quantized:
                # FP16 path: move from CPU to GPU temporarily
                self.text_backbone = self.text_backbone.to(self.device)

            wrapped = wrap_with_instruction(uncached)
            text_tokens = self.tokenizer(
                wrapped, padding=True, truncation=True,
                max_length=self.max_text_length, return_tensors="pt",
            )
            text_tokens = {k: v.to(self.device) for k, v in text_tokens.items()}
            text_embed = self.text_backbone(**text_tokens)
            new_embeddings = last_token_pool(
                text_embed.last_hidden_state, text_tokens["attention_mask"]
            )
            for prompt, emb in zip(uncached, new_embeddings):
                emb_cpu = emb.cpu()
                self._embed_cache[prompt] = emb_cpu
                _save_disk_cache(prompt, self.text_encoding_model, emb_cpu)

            if not self._backbone_quantized:
                # FP16 path: offload to CPU to free VRAM for segmentation network
                self.text_backbone = self.text_backbone.to("cpu")
                empty_cache(self.device)

        embeddings = torch.stack([self._embed_cache[p] for p in text_prompts], dim=0)
        return embeddings.view(1, n_prompts, -1).to(self.device)

    # ── Sliding window inference — batched ────────────────────────────────────

    @torch.inference_mode()
    def predict_sliding_window_return_logits(
        self,
        input_image: torch.Tensor,
        text_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        if not isinstance(input_image, torch.Tensor):
            raise ValueError(f"input_image must be a torch.Tensor, got {type(input_image)}")
        if input_image.ndim != 4:
            raise ValueError(f"input_image must be 4D (C,X,Y,Z), got {input_image.shape}")

        self.network = self.network.to(self.device)
        empty_cache(self.device)

        with (
            torch.autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else dummy_context()
        ):
            data, slicer_revert_padding = pad_nd_image(
                input_image, self.patch_size, "constant", {"value": 0}, True, None
            )
            slicers = self._internal_get_sliding_window_slicers(data.shape[1:])
            predicted_logits = self._internal_predict_sliding_window_return_logits(
                data, text_embeddings, slicers, self.perform_everything_on_device
            )
            empty_cache(self.device)
            predicted_logits = predicted_logits[(slice(None), *slicer_revert_padding[1:])]

        return predicted_logits

    @torch.inference_mode()
    def _internal_predict_sliding_window_return_logits(
        self,
        data: torch.Tensor,
        text_embeddings: torch.Tensor,
        slicers: List[Tuple],
        do_on_device: bool = True,
    ) -> torch.Tensor:
        """
        Batched sliding window inference.

        Instead of one patch per forward pass, collects `sliding_window_batch_size`
        patches and runs them through the network together. This improves GPU
        utilization significantly — especially when individual patches are small
        relative to GPU compute capacity.

        For a 4-patch volume with batch_size=4: 4 forward passes → 1 forward pass.
        """
        results_device = self.device if do_on_device else torch.device("cpu")
        empty_cache(self.device)

        data = data.to(results_device)

        predicted_logits = torch.zeros(
            (text_embeddings.shape[1], *data.shape[1:]),
            dtype=torch.half,
            device=results_device,
        )
        n_predictions = torch.zeros(data.shape[1:], dtype=torch.half, device=results_device)
        gaussian = compute_gaussian(
            tuple(self.patch_size), sigma_scale=1.0 / 8,
            value_scaling_factor=10, device=results_device,
        )

        batch_size = self.sliding_window_batch_size
        patch_buffer: List[torch.Tensor] = []
        slicer_buffer: List[Tuple] = []

        def _flush_batch():
            """Run one forward pass on the accumulated patch batch."""
            if not patch_buffer:
                return
            B = len(patch_buffer)
            batch = torch.cat(patch_buffer, dim=0)          # (B, C, H, W, D)
            # Expand text_embeddings from (1, N, D) to (B, N, D)
            batch_text = text_embeddings.expand(B, -1, -1)
            batch_pred = self.network(batch, batch_text).to(results_device)  # (B, N, H, W, D)

            for b_idx, slicer in enumerate(slicer_buffer):
                pred = batch_pred[b_idx]   # (N, H, W, D)
                pred = pred * gaussian
                predicted_logits[slicer] += pred
                n_predictions[slicer[1:]] += gaussian

            patch_buffer.clear()
            slicer_buffer.clear()

        with tqdm(desc=None, total=len(slicers)) as pbar:
            for slicer in slicers:
                patch = torch.clone(
                    data[slicer][None], memory_format=torch.contiguous_format
                ).to(self.device)
                patch_buffer.append(patch)
                slicer_buffer.append(slicer)

                if len(patch_buffer) == batch_size:
                    _flush_batch()
                    pbar.update(batch_size)

            # Flush remaining patches (when total patches % batch_size != 0)
            if patch_buffer:
                remaining = len(patch_buffer)
                _flush_batch()
                pbar.update(remaining)

        torch.div(predicted_logits, n_predictions, out=predicted_logits)

        if torch.any(torch.isinf(predicted_logits)):
            raise RuntimeError(
                "Encountered inf in predicted array. Reduce value_scaling_factor in "
                "compute_gaussian or increase dtype of predicted_logits to fp32."
            )
        return predicted_logits

    # ── Top-level prediction ──────────────────────────────────────────────────

    def predict_single_image(
        self,
        data: np.ndarray,
        text_prompts: Union[str, List[str]],
    ) -> np.ndarray:
        data, bbox, orig_shape = self.preprocess(data)
        embeddings = self.embed_text_prompts(text_prompts)
        prediction = self.predict_sliding_window_return_logits(data, embeddings).to("cpu")

        with torch.no_grad():
            prediction = torch.sigmoid(prediction.float()) > 0.5

        seg = np.zeros([prediction.shape[0], *orig_shape], dtype=np.uint8)
        seg = insert_crop_into_image(seg, prediction, bbox)
        return seg
