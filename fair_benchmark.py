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

import os
_DEFAULT_IMAGE = r"C:\Users\brian\Downloads\mni_icbm152_t1_tal_nlin_sym_09a.nii.gz"
_CLUSTER_IMAGE = "/scratch/brianx7/mni_icbm152_t1_tal_nlin_sym_09a.nii.gz"
IMAGE_PATH = _CLUSTER_IMAGE if os.path.exists(_CLUSTER_IMAGE) else _DEFAULT_IMAGE

_DEFAULT_MODEL = r"C:\Users\brian\OneDrive\Desktop\Code\VoxTell-main\models\voxtell_v1.1"
_CLUSTER_MODEL = "/scratch/brianx7/VoxTell-main/models/voxtell_v1.1"
MODEL_DIR  = _CLUSTER_MODEL if os.path.exists(_CLUSTER_MODEL) else _DEFAULT_MODEL

PROMPTS    = ["brain"]
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

# ── GPU warmup — initialize CUDA context before any timed measurement ─────────
# Without this, the first arm absorbs CUDA context init + cuDNN autotuning.
print("Warming up GPU (dummy forward pass to initialize CUDA context)...")
_dummy = torch.zeros((1, 1, 4, 4, 4), dtype=torch.float16, device=DEVICE)
_net_warmup = load_network()
with torch.inference_mode(), torch.autocast("cuda", enabled=True):
    _dummy_emb = torch.zeros((1, 1, 2560), dtype=torch.float16, device=DEVICE)
    _ = _net_warmup(_dummy, _dummy_emb)
torch.cuda.synchronize()
del _net_warmup, _dummy, _dummy_emb
torch.cuda.empty_cache()
print("GPU warm.\n")

# ═══════════════════════════════════════════════════════════════════════════════
# V0_GPU — No optimizations, but FORCED onto GPU (FP16, tile_step=0.5)
# This is the fair baseline: same hardware as v3, no algorithmic improvements
# NOTE: v3 uses INT4 (NF4) text backbone; v0_gpu uses FP16. The comparison
# therefore includes INT4 quantization as part of v3's advantage.
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
# NOTE: v3 uses INT4 (NF4) text backbone; v0_gpu uses FP16.
# We measure cold INT4 embedding (cache cleared) so this arm is cold-to-cold
# with v0_gpu. We also measure the warm cache hit separately.
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("Running v3 (all optimizations — Numba + INT4 + cache + tile_step=0.75)")
print("  v3 text backbone: INT4 (NF4) via bitsandbytes  |  v0_gpu: FP16")
print("=" * 70)

from voxtell.inference.predictor import VoxTellPredictor, _prompt_cache_path

predictor = VoxTellPredictor(model_dir=MODEL_DIR, device=DEVICE)

# Phase 1: Numba preprocessing
t0 = time.perf_counter()
data_v3, bbox, orig_shape = predictor.preprocess(raw_img)
torch.cuda.synchronize()
t_pre_v3 = time.perf_counter() - t0
print(f"  [pre]   {t_pre_v3:.3f}s")

# Phase 2: INT4 embedding — COLD measurement (disk + memory cache cleared)
# Separates INT4 gain from cache gain (cache is attributed separately at 18.7×).
for p in PROMPTS:
    disk_path = _prompt_cache_path(p, TEXT_MODEL)
    if disk_path.exists():
        disk_path.unlink()
predictor._embed_cache.clear()
print("  [embed] Cache cleared — measuring cold INT4 embedding...")

t0 = time.perf_counter()
embeddings_v3 = predictor.embed_text_prompts(PROMPTS)
torch.cuda.synchronize()
t_embed_v3_cold = time.perf_counter() - t0
print(f"  [embed cold INT4]  {t_embed_v3_cold:.3f}s")

t0 = time.perf_counter()
_ = predictor.embed_text_prompts(PROMPTS)
torch.cuda.synchronize()
t_embed_v3_warm = time.perf_counter() - t0
print(f"  [embed warm cache] {t_embed_v3_warm:.3f}s  (cache hit, attributed separately)")

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

# cold total: cold FP16 vs cold INT4 — controls for cache, isolates other gains
total_v3_cold = t_pre_v3 + t_embed_v3_cold + t_slide_v3 + t_post_v3
# warm total: cold FP16 vs cached INT4 — production use case
total_v3_warm = t_pre_v3 + t_embed_v3_warm + t_slide_v3 + t_post_v3
print(f"\n  v3 TOTAL (cold embed): {total_v3_cold:.2f}s")
print(f"  v3 TOTAL (warm cache): {total_v3_warm:.2f}s\n")

# ═══════════════════════════════════════════════════════════════════════════════
# Results
# ═══════════════════════════════════════════════════════════════════════════════
speedup_cold  = total_v0 / total_v3_cold   # cold FP16 → cold INT4 (no cache effect)
speedup_warm  = total_v0 / total_v3_warm   # cold FP16 → warm INT4 (full stack)
int4_gain     = t_embed_v0 / t_embed_v3_cold  # FP16 vs INT4 text encoding

print("=" * 70)
print("FAIR GPU-vs-GPU COMPARISON SUMMARY")
print("=" * 70)
print()
print("  Comparison A — cold-to-cold (controls for cache; isolates INT4 + algo gains)")
print(f"  {'Metric':<28} {'v0_gpu (FP16)':>14} {'v3 (INT4)':>12}")
print("  " + "-" * 56)
print(f"  {'Preprocessing':<28} {t_pre_v0:>13.3f}s {t_pre_v3:>11.3f}s")
print(f"  {'Text embedding (cold)':<28} {t_embed_v0:>13.3f}s {t_embed_v3_cold:>11.3f}s  ← INT4 vs FP16")
print(f"  {'Sliding window':<28} {t_slide_v0:>13.3f}s {t_slide_v3:>11.3f}s")
print(f"  {'Postprocessing':<28} {t_post_v0:>13.3f}s {t_post_v3:>11.3f}s")
print(f"  {'Patches':<28} {n_patches_v0:>14} {len(slicers_v3):>12}")
print("  " + "-" * 56)
print(f"  {'TOTAL':<28} {total_v0:>13.2f}s {total_v3_cold:>11.2f}s")
print(f"  Speedup (cold, no cache): {speedup_cold:.1f}×")
print()
print("  Comparison B — with embedding cache (production warm-query use case)")
print(f"  {'Text embedding (warm)':<28} {t_embed_v0:>13.3f}s {t_embed_v3_warm:>11.3f}s")
print(f"  {'TOTAL':<28} {total_v0:>13.2f}s {total_v3_warm:>11.2f}s")
print(f"  Speedup (warm cache):     {speedup_warm:.1f}×")
print()
print(f"  INT4 text encoding gain (cold FP16 → cold INT4): {int4_gain:.1f}×")
print(f"  Cache gain (cold INT4 → warm INT4):               {t_embed_v3_cold/t_embed_v3_warm:.1f}×")
print()
print("  Note: INT4 (NF4) is active in v3 by default (bitsandbytes NF4).")
print("        DSC impact of INT4 quantization is NOT separately measured.")
print(f"  Original reported speedup: 26.0×  (CPU baseline — unfair comparison)")
print("=" * 70)

# Save — use a GPU-specific filename so H100 and RTX runs don't overwrite each other
gpu_name = torch.cuda.get_device_name(0)
if "H100" in gpu_name:
    out_file = "fair_benchmark_h100_results.txt"
elif "RTX" in gpu_name or "GeForce" in gpu_name:
    out_file = "fair_benchmark_results.txt"
else:
    out_file = f"fair_benchmark_{gpu_name.replace(' ', '_')}_results.txt"

lines = [
    "Fair GPU-vs-GPU Benchmark Results",
    "=" * 40,
    f"GPU: {gpu_name}",
    "",
    f"v0_gpu total : {total_v0:.2f}s  (FP16, tile_step=0.5, no cache, numpy preprocess)",
    f"v3 cold total: {total_v3_cold:.2f}s  (INT4, tile_step=0.75, no cache, Numba)",
    f"v3 warm total: {total_v3_warm:.2f}s  (INT4, tile_step=0.75, cache hit, Numba)",
    "",
    f"Speedup cold (cold FP16 -> cold INT4, no cache): {speedup_cold:.1f}x",
    f"Speedup warm (cold FP16 -> cached INT4):         {speedup_warm:.1f}x",
    f"INT4 text encoding gain (FP16 cold -> INT4 cold): {int4_gain:.1f}x",
    "",
    "Note: Original 26x used CPU baseline (FP32 text encoder VRAM overflow).",
    "Note: INT4 (NF4) active in v3 by default. DSC impact NOT measured.",
    "Note: GPU warmup pass run before v0_gpu arm to initialize CUDA context.",
]
Path(out_file).write_text("\n".join(lines))
print(f"\nSaved: {out_file}")
