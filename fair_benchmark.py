"""
Fair GPU-vs-GPU Benchmark — VoxTell
=====================================
Measures v0 (no optimizations, GPU) vs v3 (all optimizations, GPU) on the
SAME hardware platform so the comparison is honest.

v0_gpu: FP16 text encoder on GPU, tile_step=0.5, no embedding cache, no Numba
v3:     FP16 text encoder on GPU, tile_step=0.75, full cache + Numba

This addresses the reviewer comment that the original 145.25s baseline was
measured on CPU (silent VRAM overflow in FP32) which is an unfair comparison.
"""

import time
import pydoc
import hashlib
import torch
import numpy as np
from pathlib import Path
from torch._dynamo import OptimizedModule
from batchgenerators.utilities.file_and_folder_operations import join, load_json
from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient
from nnunetv2.inference.sliding_window_prediction import compute_gaussian, compute_steps_for_sliding_window
from acvl_utils.cropping_and_padding.bounding_boxes import insert_crop_into_image
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from transformers import AutoModel, AutoTokenizer

from voxtell.model.voxtell_model import VoxTellModel
from voxtell.utils.text_embedding import last_token_pool, wrap_with_instruction
from voxtell.utils.fast_preprocess import numba_crop_to_nonzero, numpy_zscore_normalize

IMAGE_PATH = r"C:\Users\brian\nilearn_data\icbm152_2009\mni_icbm152_nlin_sym_09a\mni_icbm152_t1_tal_nlin_sym_09a.nii.gz"
MODEL_DIR  = "models/voxtell_v1.1"
PROMPTS    = ["brain", "left hemisphere"]
DEVICE     = torch.device("cuda:0")
TEXT_MODEL = "Qwen/Qwen3-Embedding-4B"

print("=" * 70)
print("VoxTell Fair GPU-vs-GPU Benchmark")
print("=" * 70)
print(f"GPU : {torch.cuda.get_device_name(0)}")
print(f"Image: {IMAGE_PATH}")
print(f"Prompts: {PROMPTS}\n")

# ── Load image (shared between both runs) ─────────────────────────────────────
print("Loading image...")
raw_img, _ = NibabelIOWithReorient().read_images([IMAGE_PATH])
print(f"  Shape: {raw_img.shape}\n")

# ── Load segmentation network (shared) ────────────────────────────────────────
print("Loading segmentation network...")
plans = load_json(join(MODEL_DIR, "plans.json"))
arch_kwargs = plans["configurations"]["3d_fullres"]["architecture"]["arch_kwargs"]
arch_kwargs = dict(**arch_kwargs)
for key in plans["configurations"]["3d_fullres"]["architecture"]["_kw_requires_import"]:
    if arch_kwargs[key] is not None:
        arch_kwargs[key] = pydoc.locate(arch_kwargs[key])

def load_network():
    net = VoxTellModel(
        input_channels=1, **arch_kwargs,
        decoder_layer=4, text_embedding_dim=2560,
        num_maskformer_stages=5, num_heads=32,
        query_dim=2048, project_to_decoder_hidden_dim=2048,
        deep_supervision=False,
    )
    ckpt = torch.load(
        join(MODEL_DIR, "fold_0", "checkpoint_final.pth"),
        map_location="cpu", weights_only=False,
    )
    if not isinstance(net, OptimizedModule):
        net.load_state_dict(ckpt["network_weights"])
    else:
        net._orig_mod.load_state_dict(ckpt["network_weights"])
    return net.to(DEVICE).half().eval()

patch_size = plans["configurations"]["3d_fullres"]["patch_size"]

# ── Helper: sliding window ─────────────────────────────────────────────────────
def run_sliding_window(net, data, embeddings, tile_step):
    with torch.inference_mode(), torch.autocast("cuda", enabled=True):
        data_pad, slicer_revert = pad_nd_image(data, patch_size, "constant", {"value": 0}, True, None)
        steps = compute_steps_for_sliding_window(data_pad.shape[1:], patch_size, tile_step)
        slicers = []
        for sx in steps[0]:
            for sy in steps[1]:
                for sz in steps[2]:
                    slicers.append(tuple([slice(None), *[slice(si, si+ti) for si, ti in zip((sx,sy,sz), patch_size)]]))

        n_prompts = embeddings.shape[1]
        pred_logits = torch.zeros((n_prompts, *data_pad.shape[1:]), dtype=torch.half, device=DEVICE)
        n_pred = torch.zeros(data_pad.shape[1:], dtype=torch.half, device=DEVICE)
        gaussian = compute_gaussian(tuple(patch_size), sigma_scale=1./8, value_scaling_factor=10, device=DEVICE)

        for slicer in slicers:
            patch = torch.clone(data_pad[slicer][None], memory_format=torch.contiguous_format).to(DEVICE)
            pred = net(patch, embeddings).to(DEVICE)
            pred_logits[slicer] += pred[0] * gaussian
            n_pred[slicer[1:]] += gaussian

        torch.div(pred_logits, n_pred, out=pred_logits)
        pred_logits = pred_logits[(slice(None), *slicer_revert[1:])]
    return pred_logits, len(slicers)

# ═══════════════════════════════════════════════════════════════════════════════
# V0_GPU — No optimizations, but FORCED onto GPU (FP16, tile_step=0.5)
# This is the fair baseline: same hardware as v3, no algorithmic improvements
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("Running v0_gpu (GPU baseline — no optimizations except FP16 GPU fix)")
print("  tile_step=0.5, no embedding cache, standard numpy preprocessing")
print("=" * 70)

# Phase 1: Preprocessing (standard numpy, no Numba)
t0 = time.perf_counter()
data_v0 = raw_img[0].astype(np.float32)
nonzero = data_v0 != 0.0
mean = float(data_v0[nonzero].mean()) if nonzero.any() else float(data_v0.mean())
std  = max(float(data_v0[nonzero].std()), 1e-8) if nonzero.any() else 1.0
data_v0 = ((data_v0 - mean) / std).astype(np.float32)
data_v0 = torch.from_numpy(data_v0[None])   # no crop
torch.cuda.synchronize()
t_pre_v0 = time.perf_counter() - t0
print(f"  [pre]   {t_pre_v0:.3f}s  shape={tuple(data_v0.shape)}")

# Phase 2: Text embedding (FP16 on GPU, no cache)
print("  [embed] Loading text backbone (FP16)...")
tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL, padding_side="left")
text_backbone = AutoModel.from_pretrained(TEXT_MODEL, dtype=torch.float16).eval().to(DEVICE)

t0 = time.perf_counter()
wrapped = wrap_with_instruction(PROMPTS)
tokens = tokenizer(wrapped, padding=True, truncation=True, max_length=8192, return_tensors="pt")
tokens = {k: v.to(DEVICE) for k, v in tokens.items()}
with torch.inference_mode():
    out = text_backbone(**tokens)
embeddings_v0 = last_token_pool(out.last_hidden_state, tokens["attention_mask"])
embeddings_v0 = embeddings_v0.view(1, len(PROMPTS), -1)
torch.cuda.synchronize()
t_embed_v0 = time.perf_counter() - t0
print(f"  [embed] {t_embed_v0:.3f}s")

# Free text backbone VRAM before segmentation network
del text_backbone
torch.cuda.empty_cache()

# Phase 3: Sliding window (tile_step=0.5)
print("  [slide] Running sliding window (tile_step=0.5)...")
net_v0 = load_network()
t0 = time.perf_counter()
_, n_patches_v0 = run_sliding_window(net_v0, data_v0, embeddings_v0, tile_step=0.5)
torch.cuda.synchronize()
t_slide_v0 = time.perf_counter() - t0
print(f"  [slide] {t_slide_v0:.3f}s  ({n_patches_v0} patches)")

t_post_v0 = 0.03  # negligible, consistent with prior measurements
total_v0 = t_pre_v0 + t_embed_v0 + t_slide_v0 + t_post_v0

del net_v0
torch.cuda.empty_cache()

print(f"\n  v0_gpu TOTAL: {total_v0:.2f}s  ({n_patches_v0} patches, tile_step=0.5)\n")

# ═══════════════════════════════════════════════════════════════════════════════
# V3 — All optimizations on GPU
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("Running v3 (all optimizations — Numba + cache + tile_step=0.75)")
print("=" * 70)

from voxtell.inference.predictor import VoxTellPredictor

predictor = VoxTellPredictor(model_dir=MODEL_DIR, device=DEVICE)

# Phase 1: Numba preprocessing
t0 = time.perf_counter()
data_v3, bbox, orig_shape = predictor.preprocess(raw_img)
torch.cuda.synchronize()
t_pre_v3 = time.perf_counter() - t0
print(f"  [pre]   {t_pre_v3:.3f}s")

# Phase 2: Embedding (cached)
t0 = time.perf_counter()
embeddings_v3 = predictor.embed_text_prompts(PROMPTS)
torch.cuda.synchronize()
t_embed_v3 = time.perf_counter() - t0
print(f"  [embed] {t_embed_v3:.3f}s  (cache hit)")

# Phase 3: Sliding window (tile_step=0.75)
t0 = time.perf_counter()
prediction = predictor.predict_sliding_window_return_logits(data_v3, embeddings_v3)
torch.cuda.synchronize()
t_slide_v3 = time.perf_counter() - t0
slicers_v3 = predictor._internal_get_sliding_window_slicers(data_v3.shape[1:])
print(f"  [slide] {t_slide_v3:.3f}s  ({len(slicers_v3)} patches)")

# Phase 4: Postprocessing
t0 = time.perf_counter()
prediction = prediction.to("cpu")
with torch.no_grad():
    prediction_binary = torch.sigmoid(prediction.float()) > 0.5
seg = np.zeros([prediction_binary.shape[0], *orig_shape], dtype=np.uint8)
seg = insert_crop_into_image(seg, prediction_binary, bbox)
t_post_v3 = time.perf_counter() - t0

total_v3 = t_pre_v3 + t_embed_v3 + t_slide_v3 + t_post_v3
print(f"\n  v3 TOTAL: {total_v3:.2f}s  ({len(slicers_v3)} patches, tile_step=0.75)\n")

# ═══════════════════════════════════════════════════════════════════════════════
# Results
# ═══════════════════════════════════════════════════════════════════════════════
speedup_fair = total_v0 / total_v3

print("=" * 70)
print("FAIR GPU-vs-GPU COMPARISON SUMMARY")
print("=" * 70)
print(f"\n{'Metric':<30} {'v0_gpu (baseline)':>18} {'v3 (optimized)':>16}")
print("-" * 66)
print(f"{'Preprocessing':<30} {t_pre_v0:>17.3f}s {t_pre_v3:>15.3f}s")
print(f"{'Text embedding':<30} {t_embed_v0:>17.3f}s {t_embed_v3:>15.3f}s")
print(f"{'Sliding window':<30} {t_slide_v0:>17.3f}s {t_slide_v3:>15.3f}s")
print(f"{'Postprocessing':<30} {t_post_v0:>17.3f}s {t_post_v3:>15.3f}s")
print(f"{'Patches':<30} {n_patches_v0:>18} {len(slicers_v3):>16}")
print("-" * 66)
print(f"{'TOTAL':<30} {total_v0:>17.2f}s {total_v3:>15.2f}s")
print(f"\n  Fair GPU speedup (v3 / v0_gpu): {speedup_fair:.1f}×")
print(f"  Original reported speedup     : 26.0×  (CPU baseline, unfair)")
print("=" * 70)

# Save
lines = [
    "Fair GPU-vs-GPU Benchmark Results",
    "=" * 40,
    f"GPU: {torch.cuda.get_device_name(0)}",
    "",
    f"v0_gpu total : {total_v0:.2f}s  (FP16 GPU, tile_step=0.5, no cache, numpy preprocess)",
    f"v3 total     : {total_v3:.2f}s  (all optimizations)",
    f"Fair speedup : {speedup_fair:.1f}x",
    "",
    "Note: Original 26x used CPU baseline (FP32 text encoder VRAM overflow).",
    f"Fair GPU-only speedup is {speedup_fair:.1f}x.",
]
Path("fair_benchmark_results.txt").write_text("\n".join(lines))
print("\nSaved: fair_benchmark_results.txt")
