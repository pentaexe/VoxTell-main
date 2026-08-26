# nnInteractive Optimization Report

## Baseline Profiling — H100 MIG 3g.40gb

**Model**: nnInteractive v1.0 checkpoint (nnUNet 3D, patch size 192³)  
**Settings**: fold=0, autozoom=off, bbox prompt, FP16 autocast  
**Hardware**: NVIDIA H100 80GB HBM3 MIG 3g.40gb

| Phase | Time | Notes |
|-------|------|-------|
| set_image | 0.345s | Once per case — preprocessing + image encoding |
| _predict (cold) | ~2.4s | First call only — CUDA kernel compilation |
| _predict (warm) | **0.108s** | Per object, mean of 3 runs |
| Total per case (15 obj, warm) | **1.96s** | 0.345 + 15 × 0.108 |

**Bottleneck**: `_predict` — specifically the first-call cold-start (~2.4s).  
Warm calls at 0.108s are already fast; the CUDA kernel compilation overhead on session start is the main user-visible latency.

---

## Comparison with VoxTell

| Model | Per-prompt latency (H100, warm) |
|-------|---------------------------------|
| VoxTell v0_gpu (baseline) | 2.27s |
| VoxTell v3 (optimized) | 0.55s |
| nnInteractive (fold=0, warm) | 0.107s |

nnInteractive is already ~5× faster per prompt than optimized VoxTell because it uses bbox prompts (no 4B text encoder). The challenge is the cold-start.

---

## Optimization Plan

### O1 — torch.compile (Triton, H100 native) ✅ COMPLETE

**Method**: Wrap `session.network` with `torch.compile(mode='reduce-overhead')`

```python
session.network = torch.compile(session.network, mode='reduce-overhead')
```

**Measured results (job 46108850, H100 MIG 3g.40gb, N_WARMUP=2):**

| Setting | _predict (warm) | Speedup |
|---------|----------------|---------|
| Baseline (no compile) | 0.108s | 1.0× |
| **torch.compile** | **0.069s** | **1.58×** |

- Triton kernel compilation cold-start: **71.4s** (one-time; kernels cached to scratch after first run)
- Run 1 after 1 warmup: 0.388s (residual compilation) — requires 2nd warmup call to stabilize
- Runs 2–3 after 2 warmups: **0.066–0.069s** (fully warm)
- 15-object case with compile (warm Triton cache): 0.345 + 15 × 0.069 = **1.38s** (vs 1.96s baseline, **1.42× case-level speedup**)
- **Cold Triton cache**: first case = ~73s (71.4s compile + 1.38s inference). Break-even vs baseline: ~1,830 objects (~122 cases). After that, all subsequent cases benefit from the 1.42× speedup.
- Practical deployment: Triton cache persists on scratch across jobs. Cache is populated on first run and reused thereafter — cold-start cost is one-time per cluster environment.

**Note**: `N_WARMUP=2` required (not 1) for compiled sessions. First warmup triggers Triton compilation (71.4s); second warmup stabilizes kernel dispatch (0.388s → 0.069s).

**DSC accuracy check (job 55034701, 20 CT cases, 294 objects, fold='all', H100 MIG):**

| Setting | Mean DSC | Objects |
|---------|----------|---------|
| Baseline (fold='all', no compile) | 0.7913 | 294 |
| **torch.compile (fold='all')** | **0.7907** | 294 |
| Difference | **−0.0006** | — |

**Verdict: accuracy maintained** (< 0.005 DSC change). torch.compile produces numerically near-identical results using the official fold='all' checkpoint.

---

### O2 — fold='all' vs fold=0 (SUPERSEDED)

**Correction**: `fold='all'` in nnU-Net is a single model trained on the full training set — it is NOT a 5-fold ensemble. The previous estimate of "~0.54s = 5×" was incorrect and has been removed. Since it is the same architecture and patch count as fold=0, per-object latency is identical (~0.108s baseline). The 0.33 DSC seen with fold=0 vs 0.79 with fold='all' likely reflects undertrained or mismatched weights in the fold_0 directory, not a fold-quality difference. **Do not present fold=0 vs fold='all' as a speed/quality tradeoff.**

---

### O3 — Kernel cache pre-warming

**Target**: Eliminate cold-start overhead on first session  
**Method**: Submit a pre-warm job that runs one dummy `_predict` call to populate the CUDA kernel cache at `/scratch/brianx7/torch_home`, then reuse across benchmark runs.  
**Status**: Partially addressed — CUDA kernel cache persists across jobs on scratch. Triton cache (71.4s) also persists via `XDG_CACHE_HOME=/scratch/brianx7/cache`.

---

### O4 — autozoom analysis

**Target**: Quantify autozoom overhead (currently disabled in benchmark)  
**Method**: Run with `do_autozoom=True` and compare per-object latency.  
**Expected**: +0–3 extra forward passes per object when anatomy spans bounding box boundary.  
**Status**: Pending.

---

## Summary of Results

⚠️ **Config note**: Latency numbers (0.108s/0.069s) were measured with fold=0 + autozoom=OFF. DSC numbers were measured with fold='all' + autozoom=ON. Since fold='all' is the same architecture as fold=0, latency is expected to be identical — but a combined single job measuring both latency and DSC under the same config (fold='all', autozoom=ON) is needed to confirm this before presenting as final results.

| Setting | _predict (warm) | 15-obj case (warm cache) | Cold-cache first case | vs Baseline |
|---------|----------------|--------------------------|----------------------|-------------|
| Baseline (fold='all', no compile) | 0.108s | 1.96s | 1.96s | 1.0× |
| **torch.compile (fold='all')** | **0.069s** | **1.38s** | **~73s** | **1.42× (warm)** |

**Break-even** (cold Triton cache): ~1,830 objects (~122 cases). After break-even, all subsequent cases run at 1.42×.

**DSC (fold='all', 20 cases, 294 objects, job 55034701):** Baseline 0.7913 → torch.compile 0.7907 (−0.0006, accuracy maintained)