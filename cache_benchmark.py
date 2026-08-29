"""
Embedding cache benchmark — VoxTell
====================================
Isolates what the embedding cache is worth, on a CVPR validation CT volume.

The cache has two layers and they are worth separating:
  cold        — no cache at all: tokenize + full Qwen3-4B forward pass
  disk hit    — SHA-256 keyed .pt on scratch, deserialized (memory dict cleared)
  memory hit  — in-process dict lookup

Everything except the cache state is held constant: same predictor instance,
same INT4 (NF4) backbone, same image, same tile_step. So the difference is the
cache and nothing else.

Design notes, per the project's measurement rules:
  - precision held constant (both arms INT4; asserted, not assumed)
  - GPU, text backbone and sliding-window path all warmed before any timing
  - cache asserted empty before the cold measurement and written after it
  - CT volume, not the MNI brain
  - the cache only affects the embedding phase, so the end-to-end figure is
    reported separately from the embedding-only figure rather than blended

Usage:
    sbatch cache_benchmark.sh
"""

import os
import time
import numpy as np
import torch
from pathlib import Path

from voxtell.inference.predictor import VoxTellPredictor, _prompt_cache_path

MODEL_DIR  = "/scratch/brianx7/VoxTell-main/models/voxtell_v1.1"
VAL_DIR    = "/scratch/brianx7/cvpr_val/3D_val_npz"
DEVICE     = torch.device("cuda:0")
TEXT_MODEL = "Qwen/Qwen3-Embedding-4B"
N_WARM     = 20          # repeat hits, to get a stable read on a sub-ms operation

print("=" * 70)
print("VoxTell — embedding cache benchmark")
print("=" * 70)
print(f"GPU: {torch.cuda.get_device_name(0)}")

# ── Image: a CT volume, not the brain ─────────────────────────────────────────
case = os.environ.get("BENCH_IMAGE") or sorted(Path(VAL_DIR).glob("CT_*.npz"))[0]
npz = np.load(case, allow_pickle=True)
raw_img = npz["imgs"][None].astype(np.float32)

# These cases ship their own prompts; read rather than guess the anatomy.
prompts = npz["text_prompts"].tolist() if "text_prompts" in npz.files else None
if isinstance(prompts, dict):
    prompts = [prompts[k] for k in sorted(prompts)]
PROMPTS = [str(prompts[0])] if prompts else ["liver"]

print(f"Case: {Path(case).name}  {raw_img.shape[1:]}  ({int(np.prod(raw_img.shape[1:])):,} voxels)")
print(f"Prompt: {PROMPTS}\n")

# ── Predictor, and proof it is actually INT4 ──────────────────────────────────
predictor = VoxTellPredictor(model_dir=MODEL_DIR, device=DEVICE)

if not getattr(predictor, "_backbone_quantized", False):
    raise RuntimeError(
        "Text backbone did NOT load as INT4 — it fell back to FP16.\n"
        "_load_text_backbone swallows any exception, so the label lies.\n"
        "Usual cause: accelerate missing. Fix the env; do not report this run."
    )
print("[check] backbone confirmed INT4 (NF4)\n")

assert predictor.text_encoding_model == TEXT_MODEL, (
    f"cache key mismatch: predictor uses {predictor.text_encoding_model!r}, "
    f"this script clears {TEXT_MODEL!r} — the 'cold' timing would be a cache hit"
)


def clear_all_caches():
    for p in PROMPTS:
        d = _prompt_cache_path(p, TEXT_MODEL)
        if d.exists():
            d.unlink()
    predictor._embed_cache.clear()


def assert_cold():
    for p in PROMPTS:
        assert not _prompt_cache_path(p, TEXT_MODEL).exists(), f"disk cache survived for {p!r}"
    assert not predictor._embed_cache, "memory cache survived clear()"


def assert_written():
    for p in PROMPTS:
        assert _prompt_cache_path(p, TEXT_MODEL).exists(), (
            f"no disk cache written for {p!r} — no encode actually ran"
        )


# ── Warm everything before timing anything ────────────────────────────────────
# Whichever phase runs first otherwise absorbs CUDA context init and cuDNN
# autotuning, which is what inflated an earlier result from 1.0x to 7.1x.
print("Warming GPU, backbone and sliding-window path (untimed)...")
_t = time.perf_counter()
_ = predictor.embed_text_prompts(["__warmup__"])           # backbone + kernels
_data, _bbox, _shape = predictor.preprocess(raw_img)        # Numba JIT
_emb = predictor.embed_text_prompts(["__warmup__"])
_ = predictor.predict_sliding_window_return_logits(_data, _emb)   # cuDNN autotune
torch.cuda.synchronize()
print(f"Warm. ({time.perf_counter() - _t:.1f}s absorbed, not counted)\n")

# Drop the warmup prompt so it cannot pollute the measured cache state.
for p in ["__warmup__"]:
    d = _prompt_cache_path(p, TEXT_MODEL)
    if d.exists():
        d.unlink()
predictor._embed_cache.pop("__warmup__", None)

data, bbox, orig_shape = predictor.preprocess(raw_img)

# ══ A — cold: no cache, full Qwen3-4B forward ════════════════════════════════
clear_all_caches()
assert_cold()
t0 = time.perf_counter()
emb = predictor.embed_text_prompts(PROMPTS)
torch.cuda.synchronize()
t_cold = time.perf_counter() - t0
assert_written()
print(f"[A] cold encode          {t_cold*1000:>9.2f} ms   (verified cold)")

# ══ B — disk hit: memory dict cleared, .pt on scratch still present ═══════════
disk_times = []
for _ in range(N_WARM):
    predictor._embed_cache.clear()
    t0 = time.perf_counter()
    _ = predictor.embed_text_prompts(PROMPTS)
    torch.cuda.synchronize()
    disk_times.append(time.perf_counter() - t0)
t_disk = float(np.median(disk_times))
print(f"[B] disk-cache hit       {t_disk*1000:>9.2f} ms   (median of {N_WARM})")

# ══ C — memory hit: in-process dict ══════════════════════════════════════════
mem_times = []
for _ in range(N_WARM):
    t0 = time.perf_counter()
    _ = predictor.embed_text_prompts(PROMPTS)
    torch.cuda.synchronize()
    mem_times.append(time.perf_counter() - t0)
t_mem = float(np.median(mem_times))
print(f"[C] memory-cache hit     {t_mem*1000:>9.2f} ms   (median of {N_WARM})\n")

# ══ End-to-end context: how much of a full prediction is the embedding ═══════
t0 = time.perf_counter()
_ = predictor.predict_sliding_window_return_logits(data, emb)
torch.cuda.synchronize()
t_slide = time.perf_counter() - t0

total_cold = t_cold + t_slide
total_warm = t_mem + t_slide

print("=" * 70)
print("RESULTS")
print("=" * 70)
print(f"  Embedding phase only")
print(f"    cold             {t_cold*1000:>9.2f} ms")
print(f"    disk hit         {t_disk*1000:>9.2f} ms      {t_cold/max(t_disk,1e-9):>7.1f}x vs cold")
print(f"    memory hit       {t_mem*1000:>9.2f} ms      {t_cold/max(t_mem,1e-9):>7.1f}x vs cold")
print()
print(f"  In context (embedding + sliding window)")
print(f"    sliding window   {t_slide*1000:>9.2f} ms   (identical in both arms)")
print(f"    total, cold      {total_cold*1000:>9.2f} ms")
print(f"    total, cached    {total_warm*1000:>9.2f} ms      {total_cold/total_warm:>7.2f}x end-to-end")
print()
print(f"  Embedding is {100*t_cold/total_cold:.1f}% of a cold prediction, "
      f"{100*t_mem/total_warm:.1f}% of a cached one.")
print()
print("  Scope: the cache does nothing on a first-ever prompt. The end-to-end")
print("  figure applies only to repeat queries for the same text, and the")
print("  sliding window is unchanged, which bounds what the cache can ever buy.")
print("  Single run — repeat to n>=4 and quote the range before citing.")
print("=" * 70)

out = Path(f"cache_benchmark_{Path(case).stem}_results.txt")
out.write_text("\n".join([
    "VoxTell embedding cache benchmark",
    "=" * 40,
    f"GPU: {torch.cuda.get_device_name(0)}",
    f"Case: {Path(case).name}  shape={tuple(raw_img.shape[1:])}",
    f"Prompt: {PROMPTS}",
    "Both arms INT4 (NF4), asserted. All paths warmed before timing.",
    "",
    f"cold encode      : {t_cold*1000:.2f} ms  (cache verified empty before, written after)",
    f"disk-cache hit   : {t_disk*1000:.2f} ms  (median of {N_WARM})",
    f"memory-cache hit : {t_mem*1000:.2f} ms  (median of {N_WARM})",
    f"sliding window   : {t_slide*1000:.2f} ms  (identical both arms)",
    "",
    f"embedding speedup: {t_cold/max(t_mem,1e-9):.1f}x   end-to-end: {total_cold/total_warm:.2f}x",
    "",
    "Applies to repeat prompts only; a first-ever prompt gets nothing.",
    "Single run - repeat to n>=4 and quote the range before citing.",
]))
print(f"\nSaved: {out}")
