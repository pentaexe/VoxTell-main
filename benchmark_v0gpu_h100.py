"""
VoxTell v0_gpu Baseline Benchmark — H100
Runs the unoptimized pipeline (tile_step=0.5, no cache, cold embed)
to establish a fair H100 baseline for algorithmic speedup comparison.
"""
import time
import torch
import numpy as np
from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient
from acvl_utils.cropping_and_padding.bounding_boxes import insert_crop_into_image
from voxtell.inference.predictor import VoxTellPredictor
import voxtell.inference.predictor as _pred_module

import os
IMAGE_PATH = "/scratch/brianx7/mni_icbm152_t1_tal_nlin_sym_09a.nii.gz"
MODEL_DIR  = "/scratch/brianx7/VoxTell-main/models/voxtell_v1.1"
PROMPTS    = ["brain", "left hemisphere"]
DEVICE     = torch.device("cuda:0")

# Disable disk cache so embed is always cold
_pred_module._load_disk_cache = lambda prompt, model_name: None
_pred_module._save_disk_cache = lambda prompt, model_name, embedding: None

print(f"Device : {DEVICE}")
print(f"GPU    : {torch.cuda.get_device_name(0)}")
print(f"Prompts: {PROMPTS}")
print("Config : tile_step=0.5, NO cache (v0_gpu baseline)\n")

# ── Load image ────────────────────────────────────────────────────────────────
t0 = time.perf_counter()
img, props = NibabelIOWithReorient().read_images([IMAGE_PATH])
print(f"Image loaded: {img.shape}  ({time.perf_counter() - t0:.2f}s)\n")

# ── Load model ────────────────────────────────────────────────────────────────
print("Loading model...")
t0 = time.perf_counter()
predictor = VoxTellPredictor(model_dir=MODEL_DIR, device=DEVICE)
predictor.tile_step_size = 0.5          # override to unoptimized setting
predictor._embed_cache.clear()          # clear in-memory cache
print(f"Model loaded: {time.perf_counter() - t0:.2f}s")
print(f"tile_step_size = {predictor.tile_step_size}\n")

# ── Phase 1: Preprocessing ────────────────────────────────────────────────────
print("[Phase 1] Preprocessing...")
t0 = time.perf_counter()
data, bbox, orig_shape = predictor.preprocess(img)
torch.cuda.synchronize()
t_preprocess = time.perf_counter() - t0
print(f"  Shape : {data.shape}  |  Time: {t_preprocess:.3f}s")

# ── Phase 2: Text embedding (cold — cache disabled) ───────────────────────────
print("\n[Phase 2] Text embedding (cold, no cache)...")
t0 = time.perf_counter()
torch.cuda.synchronize()
embeddings = predictor.embed_text_prompts(PROMPTS)
torch.cuda.synchronize()
t_embed = time.perf_counter() - t0
print(f"  Shape : {embeddings.shape}  |  Time: {t_embed:.3f}s")

# ── Phase 3: Sliding window (tile_step=0.5) ───────────────────────────────────
print("\n[Phase 3] Sliding window (tile_step=0.5)...")
slicers = predictor._internal_get_sliding_window_slicers(data.shape[1:])
print(f"  Patches: {len(slicers)}")

print("  Warming up (2 passes)...")
for _ in range(2):
    _ = predictor.predict_sliding_window_return_logits(data, embeddings)
    torch.cuda.synchronize()

print("  Timing (3 passes, reporting mean)...")
times = []
for _ in range(3):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    prediction = predictor.predict_sliding_window_return_logits(data, embeddings)
    torch.cuda.synchronize()
    times.append(time.perf_counter() - t0)
t_inference = float(np.mean(times))
print(f"  Time: {t_inference:.3f}s  (runs: {[f'{t:.3f}s' for t in times]})")

# ── Phase 4: Postprocessing ───────────────────────────────────────────────────
print("\n[Phase 4] Postprocessing...")
t0 = time.perf_counter()
prediction = prediction.to("cpu")
with torch.no_grad():
    prediction_binary = torch.sigmoid(prediction.float()) > 0.5
seg = np.zeros([prediction_binary.shape[0], *orig_shape], dtype=np.uint8)
seg = insert_crop_into_image(seg, prediction_binary, bbox)
t_postprocess = time.perf_counter() - t0
print(f"  Shape : {seg.shape}  |  Time: {t_postprocess:.3f}s")

# ── Summary ───────────────────────────────────────────────────────────────────
t_total = t_preprocess + t_embed + t_inference + t_postprocess
print("\n" + "=" * 60)
print("v0_gpu BASELINE — H100 MIG 3g.40gb")
print("=" * 60)
print(f"  Preprocessing : {t_preprocess:.3f}s")
print(f"  Text embedding: {t_embed:.3f}s  (cold)")
print(f"  Sliding window: {t_inference:.3f}s  ({len(slicers)} patches, tile_step=0.5)")
print(f"  Postprocessing: {t_postprocess:.3f}s")
print(f"  TOTAL         : {t_total:.3f}s")
print("=" * 60)