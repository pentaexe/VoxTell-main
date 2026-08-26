"""
INT4 vs FP16 DSC Agreement — VoxTell
======================================
Measures whether INT4 (NF4) text quantization changes the segmentation output.

Design:
  - Both arms in one process, one job, same image, same segmentation network
  - FP16 arm first, INT4 arm second
  - No VoxTellPredictor — raw AutoModel only, so the embedding cache is never
    consulted and cannot leak FP16 embeddings into the INT4 arm
  - Network weights are identical (same checkpoint); only the text backbone differs
  - Agreement DSC: DSC(prediction_fp16, prediction_int4)
    A score near 1.0 means quantization introduces no segmentation change.
    A score below 0.95 warrants reporting.
"""

import time
import pydoc
import torch
import numpy as np
from pathlib import Path
from torch._dynamo import OptimizedModule
from batchgenerators.utilities.file_and_folder_operations import join, load_json
from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient
from nnunetv2.inference.sliding_window_prediction import compute_gaussian, compute_steps_for_sliding_window
from acvl_utils.cropping_and_padding.bounding_boxes import insert_crop_into_image
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

from voxtell.model.voxtell_model import VoxTellModel
from voxtell.utils.text_embedding import last_token_pool, wrap_with_instruction
from voxtell.utils.fast_preprocess import numba_crop_to_nonzero, numpy_zscore_normalize

import os
_CLUSTER_IMAGE = "/scratch/brianx7/mni_icbm152_t1_tal_nlin_sym_09a.nii.gz"
_LOCAL_IMAGE   = r"C:\Users\brian\Downloads\mni_icbm152_t1_tal_nlin_sym_09a.nii.gz"
IMAGE_PATH = _CLUSTER_IMAGE if os.path.exists(_CLUSTER_IMAGE) else _LOCAL_IMAGE

_CLUSTER_MODEL = "/scratch/brianx7/VoxTell-main/models/voxtell_v1.1"
_LOCAL_MODEL   = r"C:\Users\brian\OneDrive\Desktop\Code\VoxTell-main\models\voxtell_v1.1"
MODEL_DIR  = _CLUSTER_MODEL if os.path.exists(_CLUSTER_MODEL) else _LOCAL_MODEL

PROMPTS    = ["brain"]
DEVICE     = torch.device("cuda:0")
TEXT_MODEL = "Qwen/Qwen3-Embedding-4B"
TILE_STEP  = 0.75  # v3 setting for both arms

print("=" * 70)
print("VoxTell INT4 vs FP16 DSC Agreement")
print("=" * 70)
print(f"GPU : {torch.cuda.get_device_name(0)}")
print(f"Image: {IMAGE_PATH}")
print(f"Prompts: {PROMPTS}")
print(f"tile_step: {TILE_STEP}  (v3 setting, same for both arms)\n")

# ── Load image (shared) ───────────────────────────────────────────────────────
print("Loading and preprocessing image...")
raw_img, _ = NibabelIOWithReorient().read_images([IMAGE_PATH])
# Numba crop-to-nonzero + z-score (v3 preprocessing — same for both arms)
data_cropped, bbox, orig_shape = numba_crop_to_nonzero(raw_img[0])
data_norm = numpy_zscore_normalize(data_cropped.astype(np.float32))
data_t = torch.from_numpy(data_norm[None])
print(f"  Shape after crop: {tuple(data_t.shape)}\n")

# ── Load segmentation network (shared between both arms) ──────────────────────
plans = load_json(join(MODEL_DIR, "plans.json"))
arch_kwargs = plans["configurations"]["3d_fullres"]["architecture"]["arch_kwargs"]
arch_kwargs = dict(**arch_kwargs)
for key in plans["configurations"]["3d_fullres"]["architecture"]["_kw_requires_import"]:
    if arch_kwargs[key] is not None:
        arch_kwargs[key] = pydoc.locate(arch_kwargs[key])
patch_size = plans["configurations"]["3d_fullres"]["patch_size"]

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

def run_sliding_window(net, data, embeddings):
    with torch.inference_mode(), torch.autocast("cuda", enabled=True):
        data_pad, slicer_revert = pad_nd_image(data, patch_size, "constant", {"value": 0}, True, None)
        steps = compute_steps_for_sliding_window(data_pad.shape[1:], patch_size, TILE_STEP)
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

def embed(backbone, prompts, tokenizer):
    # tokenizer pre-loaded outside timer — measures forward pass only, not model/tokenizer load
    wrapped = wrap_with_instruction(prompts)
    tokens = tokenizer(wrapped, padding=True, truncation=True, max_length=8192, return_tensors="pt")
    tokens = {k: v.to(DEVICE) for k, v in tokens.items()}
    with torch.inference_mode():
        out = backbone(**tokens)
    emb = last_token_pool(out.last_hidden_state, tokens["attention_mask"])
    return emb.view(1, len(prompts), -1)

# ── GPU warmup ────────────────────────────────────────────────────────────────
print("Warming up GPU...")
_net_warm = load_network()
_dummy = torch.zeros((1, 1, 4, 4, 4), dtype=torch.float16, device=DEVICE)
with torch.inference_mode(), torch.autocast("cuda", enabled=True):
    _ = _net_warm(_dummy, torch.zeros((1, 1, 2560), dtype=torch.float16, device=DEVICE))
torch.cuda.synchronize()
del _net_warm, _dummy
torch.cuda.empty_cache()
print("GPU warm.\n")

# ── Tokenizer (shared, loaded once outside any timed section) ─────────────────
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL, padding_side="left")
print("Tokenizer ready.\n")

# ═══════════════════════════════════════════════════════════════════════════════
# ARM 1 — FP16 text backbone
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("ARM 1: FP16 text backbone")
print("=" * 70)
print("  Loading FP16 backbone...")
backbone_fp16 = AutoModel.from_pretrained(TEXT_MODEL, dtype=torch.float16).eval().to(DEVICE)

# Timer measures forward pass only — backbone and tokenizer already loaded above
t0 = time.perf_counter()
embeddings_fp16 = embed(backbone_fp16, PROMPTS, tokenizer)
torch.cuda.synchronize()
t_embed_fp16 = time.perf_counter() - t0
print(f"  [embed FP16 forward]  {t_embed_fp16:.3f}s  (forward pass only, backbone pre-loaded)")

del backbone_fp16
torch.cuda.empty_cache()

print("  Running sliding window...")
net = load_network()
t0 = time.perf_counter()
logits_fp16, n_patches = run_sliding_window(net, data_t, embeddings_fp16)
torch.cuda.synchronize()
t_slide_fp16 = time.perf_counter() - t0
print(f"  [slide FP16]  {t_slide_fp16:.3f}s  ({n_patches} patches)")

pred_fp16 = (torch.sigmoid(logits_fp16.float().cpu()) > 0.5).numpy()

del net, logits_fp16, embeddings_fp16
torch.cuda.empty_cache()
print(f"  ARM 1 done. Prediction volume shape: {pred_fp16.shape}\n")

# ═══════════════════════════════════════════════════════════════════════════════
# ARM 2 — INT4 (NF4) text backbone  — loaded fresh, no shared cache
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("ARM 2: INT4 (NF4) text backbone  [fresh load, no cached embeddings]")
print("=" * 70)
_bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)
print("  Loading INT4 backbone...")
backbone_int4 = AutoModel.from_pretrained(TEXT_MODEL, quantization_config=_bnb_config).eval()

# Timer measures forward pass only — consistent with FP16 arm above
t0 = time.perf_counter()
embeddings_int4 = embed(backbone_int4, PROMPTS, tokenizer)
torch.cuda.synchronize()
t_embed_int4 = time.perf_counter() - t0
print(f"  [embed INT4 forward]  {t_embed_int4:.3f}s  (forward pass only, backbone pre-loaded)")

del backbone_int4
torch.cuda.empty_cache()

print("  Running sliding window...")
net = load_network()
t0 = time.perf_counter()
logits_int4, _ = run_sliding_window(net, data_t, embeddings_int4)
torch.cuda.synchronize()
t_slide_int4 = time.perf_counter() - t0
print(f"  [slide INT4]  {t_slide_int4:.3f}s  ({n_patches} patches)")

pred_int4 = (torch.sigmoid(logits_int4.float().cpu()) > 0.5).numpy()

del net, logits_int4, embeddings_int4
torch.cuda.empty_cache()
print(f"  ARM 2 done. Prediction volume shape: {pred_int4.shape}\n")

# ═══════════════════════════════════════════════════════════════════════════════
# Agreement DSC
# ═══════════════════════════════════════════════════════════════════════════════
def dsc_agreement(a, b):
    inter = (a & b).sum()
    union = a.sum() + b.sum()
    if union == 0:
        return float('nan')
    return float(2 * inter / union)

agreement = dsc_agreement(pred_fp16, pred_int4)
fp16_vox  = int(pred_fp16.sum())
int4_vox  = int(pred_int4.sum())

print("=" * 70)
print("INT4 vs FP16 AGREEMENT")
print("=" * 70)
print()
print(f"  FP16 voxels segmented:  {fp16_vox:,}")
print(f"  INT4 voxels segmented:  {int4_vox:,}")
print(f"  Voxel delta:            {int4_vox - fp16_vox:+,}  ({100*(int4_vox-fp16_vox)/max(fp16_vox,1):+.2f}%)")
print()
print(f"  Output agreement DSC (FP16 vs INT4): {agreement:.4f}")
print()
print("  Note: Agreement DSC measures whether quantization changes the model output.")
print("        It does NOT measure accuracy vs ground truth — both arms could drift")
print("        from GT in the same direction and still agree perfectly.")
print("        This is n=1 (one MNI brain, one prompt); interpret accordingly.")
print()
print(f"  Embed forward pass: FP16 {t_embed_fp16:.3f}s  →  INT4 {t_embed_int4:.3f}s  ({t_embed_fp16/t_embed_int4:.1f}× faster)")
print("  Timer covers forward pass only (backbone and tokenizer pre-loaded).")
print("  This is the INT4 speed figure absent from fair_benchmark (both arms INT4 there).")
print("=" * 70)

# Save results
gpu_name = torch.cuda.get_device_name(0)
tag = "h100" if "H100" in gpu_name else gpu_name.replace(" ", "_")
out = f"int4_dsc_results_{tag}.txt"
lines = [
    "INT4 vs FP16 Output Agreement — VoxTell",
    "=" * 40,
    f"GPU: {gpu_name}",
    f"Image: {IMAGE_PATH}  (n=1)",
    f"Prompt: {PROMPTS}",
    f"tile_step: {TILE_STEP}  (v3, same both arms)",
    "",
    f"FP16 voxels: {fp16_vox:,}",
    f"INT4 voxels: {int4_vox:,}",
    f"Voxel delta: {int4_vox - fp16_vox:+,}",
    "",
    f"Output agreement DSC (FP16 vs INT4): {agreement:.4f}",
    "Note: measures output change, NOT accuracy vs ground truth.",
    "Note: n=1, one prompt — weak evidence about quantization generally.",
    "",
    f"Embed FP16: {t_embed_fp16:.3f}s",
    f"Embed INT4: {t_embed_int4:.3f}s",
    f"Embed forward-pass speedup (FP16->INT4, backbone pre-loaded): {t_embed_fp16/t_embed_int4:.1f}x",
]
Path(out).write_text("\n".join(lines))
print(f"\nSaved: {out}")
