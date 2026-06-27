# nnInteractive Optimization Report

## Baseline Profiling — H100 MIG 3g.40gb

**Model**: nnInteractive v1.0 checkpoint (nnUNet 3D, patch size 192³)  
**Settings**: fold=0, autozoom=off, bbox prompt, FP16 autocast  
**Hardware**: NVIDIA H100 80GB HBM3 MIG 3g.40gb

| Phase | Time | Notes |
|-------|------|-------|
| set_image | 0.345s | Once per case — preprocessing + image encoding |
| _predict (cold) | ~4.1s | First call only — CUDA kernel compilation |
| _predict (warm) | **0.107s** | Per object, mean of 3 runs |
| Total per case (15 obj, warm) | **1.95s** | 0.345 + 15 × 0.107 |

**Bottleneck**: `_predict` — specifically the first-call cold-start (4.1s).  
Warm calls at 0.107s are already fast; the CUDA kernel compilation overhead on session start is the main user-visible latency.

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

### O1 — torch.compile (Triton, H100 native)

**Target**: Warm `_predict` 0.107s → ~0.05s (estimated 2× speedup)  
**Method**: Wrap `session.network` with `torch.compile(mode='reduce-overhead')`  
**Benefit on H100**: Triton kernels replace cuDNN kernels; also eliminates CUDA compilation overhead on subsequent calls within a compiled session.

```python
session.network = torch.compile(session.network, mode='reduce-overhead')
```

**Risk**: First compiled call still slow (~10-30s), but cached across objects in the same session.

---

### O2 — Reduce ensemble (fold=0 only vs fold='all')

**Target**: Understand quality/speed tradeoff of single fold vs 5-fold ensemble  
**Method**: Profile fold=0 (0.107s) vs fold='all' (~0.535s estimated = 5×)  
**Benefit**: If fold=0 quality is acceptable for the benchmark metric, 5× speedup for free.

---

### O3 — Kernel cache pre-warming

**Target**: Eliminate 4.1s cold-start overhead  
**Method**: Submit a pre-warm job that runs one dummy `_predict` call to populate the CUDA kernel cache at `/scratch/brianx7/torch_home`, then reuse across benchmark runs.

This is analogous to the nnUNet epoch-0 slowdown in training.

---

### O4 — autozoom analysis

**Target**: Quantify autozoom overhead (currently disabled in benchmark)  
**Method**: Run with `do_autozoom=True` and compare per-object latency.  
**Expected**: +0-3 extra forward passes per object when anatomy spans bounding box boundary.

---

## Recommended Next Step

Implement **O1 (torch.compile)** first — highest impact, proven on H100 Triton backend.

```python
# In load_session():
session.initialize_from_trained_model_folder(...)
session.network = torch.compile(session.network, mode='reduce-overhead')
```

Expected results table after O1:

| Setting | _predict (warm) | Speedup |
|---------|----------------|---------|
| Baseline (no compile) | 0.107s | 1.0× |
| torch.compile | ~0.05s | ~2× |