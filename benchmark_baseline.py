"""
VoxTell Inference Benchmark
Measures per-phase timing and compares against recorded baselines.
"""
import time
import torch
import numpy as np
from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient
from acvl_utils.cropping_and_padding.bounding_boxes import insert_crop_into_image
from voxtell.inference.predictor import VoxTellPredictor

import os
_DEFAULT_IMAGE = r"C:\Users\brian\nilearn_data\icbm152_2009\mni_icbm152_nlin_sym_09a\mni_icbm152_t1_tal_nlin_sym_09a.nii.gz"
_CLUSTER_IMAGE = "/scratch/brianx7/mni_icbm152_t1_tal_nlin_sym_09a.nii.gz"
IMAGE_PATH = _CLUSTER_IMAGE if os.path.exists(_CLUSTER_IMAGE) else _DEFAULT_IMAGE

_DEFAULT_MODEL = r"C:\Users\brian\OneDrive\Desktop\Code\VoxTell-main\models\voxtell_v1.1"
_CLUSTER_MODEL = "/scratch/brianx7/VoxTell-main/models/voxtell_v1.1"
MODEL_DIR = _CLUSTER_MODEL if os.path.exists(_CLUSTER_MODEL) else _DEFAULT_MODEL
PROMPTS    = ["brain", "left hemisphere"]
DEVICE     = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ── Recorded baselines for comparison ────────────────────────────────────────
CHANGELOG = [
    {
        "version": "v0_gpu — GPU baseline",
        "changes": "FP16 text model on GPU, tile_step=0.5, no cache (RTX 4070 SUPER)",
        "preprocess": 0.13, "text_embed": 0.51, "sliding": 2.44, "postprocess": 0.03,
    },
    {
        "version": "v1 — tile_step=0.75",
        "changes": "Reduce overlap: tile_step_size 0.5→0.75, fewer patches",
        "preprocess": 0.13, "text_embed": 0.51, "sliding": 2.22, "postprocess": 0.03,
    },
    {
        "version": "v2 — + embedding cache",
        "changes": "2-level embed cache (memory LRU + disk SHA-256)",
        "preprocess": 0.13, "text_embed": 0.02, "sliding": 2.22, "postprocess": 0.03,
    },
]

print(f"Device : {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU    : {torch.cuda.get_device_name(0)}")
print(f"Prompts: {PROMPTS}\n")

# ── Load image ────────────────────────────────────────────────────────────────
t0 = time.perf_counter()
img, props = NibabelIOWithReorient().read_images([IMAGE_PATH])
t_load = time.perf_counter() - t0
print(f"Image loaded: {img.shape}  ({t_load:.2f}s)\n")

# ── Load model ────────────────────────────────────────────────────────────────
print("Loading model (torch.compile adds ~30s on first load)...")
t0 = time.perf_counter()
predictor = VoxTellPredictor(model_dir=MODEL_DIR, device=DEVICE)
t_model = time.perf_counter() - t0
print(f"Model loaded: {t_model:.2f}s\n")

# ── Phase 1: Preprocessing (GPU/C++) ─────────────────────────────────────────
print("[Phase 1] GPU preprocessing (C++/CUDA ops)...")
t0 = time.perf_counter()
data, bbox, orig_shape = predictor.preprocess(img)
if DEVICE.type == "cuda":
    torch.cuda.synchronize()
t_preprocess = time.perf_counter() - t0
print(f"  Shape : {data.shape}  |  Time: {t_preprocess:.3f}s")

# ── Phase 2: Text embedding (disk cache + FP16) ───────────────────────────────
print("\n[Phase 2] Text embedding (disk cache + FP16 GPU)...")
t0 = time.perf_counter()
embeddings = predictor.embed_text_prompts(PROMPTS)
if DEVICE.type == "cuda":
    torch.cuda.synchronize()
t_embed = time.perf_counter() - t0
print(f"  Shape : {embeddings.shape}  |  Time: {t_embed:.3f}s")

# ── Phase 3: Sliding window (torch.compile) ───────────────────────────────────
print("\n[Phase 3] Sliding window inference (torch.compile)...")
slicers = predictor._internal_get_sliding_window_slicers(data.shape[1:])
print(f"  Patches: {len(slicers)}")

# Warmup: 2 passes to trigger cuDNN autotuning before timing
print("  Warming up (2 passes)...")
for _ in range(2):
    _ = predictor.predict_sliding_window_return_logits(data, embeddings)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

print("  Timing (3 passes, reporting mean)...")
times = []
for _ in range(3):
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    prediction = predictor.predict_sliding_window_return_logits(data, embeddings)
    if DEVICE.type == "cuda":
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

# ── Current run totals ────────────────────────────────────────────────────────
t_current = t_preprocess + t_embed + t_inference + t_postprocess
CHANGELOG.append({
    "version": "CURRENT RUN",
    "changes": "Numba JIT preprocess + embed cache + tile_step=0.75",
    "preprocess": t_preprocess,
    "text_embed": t_embed,
    "sliding": t_inference,
    "postprocess": t_postprocess,
})

# ── Changelog table ───────────────────────────────────────────────────────────
baseline_total = sum([
    CHANGELOG[0]["preprocess"],
    CHANGELOG[0]["text_embed"],
    CHANGELOG[0]["sliding"],
    CHANGELOG[0]["postprocess"],
])

print("\n" + "=" * 80)
print("OPTIMIZATION CHANGELOG")
print("=" * 80)
header = f"{'Version':<35} {'Pre':>6} {'Embed':>7} {'Slide':>7} {'Post':>6} {'TOTAL':>7} {'Speedup':>8}"
print(header)
print("-" * 80)
for entry in CHANGELOG:
    total = entry["preprocess"] + entry["text_embed"] + entry["sliding"] + entry["postprocess"]
    speedup = baseline_total / total
    marker = " ◄" if entry["version"] == "CURRENT RUN" else ""
    print(
        f"{entry['version']:<35} "
        f"{entry['preprocess']:>6.2f} "
        f"{entry['text_embed']:>7.2f} "
        f"{entry['sliding']:>7.2f} "
        f"{entry['postprocess']:>6.2f} "
        f"{total:>7.2f}s "
        f"{speedup:>7.1f}x"
        f"{marker}"
    )
print("=" * 80)
print(f"\nBaseline           : {baseline_total:.2f}s")
print(f"Current            : {t_current:.2f}s")
print(f"Speedup achieved   : {baseline_total / t_current:.1f}x")
print(f"Target (5x)        : {baseline_total / 5:.2f}s")
print(f"Target met         : {'YES ✓' if baseline_total / t_current >= 5 else 'NO'}")
print()
print("Optimizations applied this run:")
for entry in CHANGELOG[1:]:
    print(f"  • {entry['version']}: {entry['changes']}")
