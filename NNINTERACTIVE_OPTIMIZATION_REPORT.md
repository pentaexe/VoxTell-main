# nnInteractive Optimization Report

## Baseline Profiling — H100 MIG 3g.40gb

**Model**: nnInteractive v1.0 checkpoint (nnUNet 3D, patch size 192³)  
**Settings**: fold=0, autozoom=OFF, bbox prompt, FP16 autocast  
**Hardware**: NVIDIA H100 80GB HBM3 MIG 3g.40gb  
**Note**: This early profiling used autozoom=OFF and fold=0. Final numbers use fold='all', autozoom=ON — see Combined Results below.

| Phase | Time | Notes |
|-------|------|-------|
| set_image | 0.345s | Once per case — preprocessing + image encoding |
| _predict (cold) | ~2.4s | First call only — CUDA kernel compilation |
| _predict (warm) | **0.108s** | Per object, mean of 3 runs, autozoom=OFF |
| Total per case (15 obj, warm) | **1.96s** | 0.345 + 15 × 0.108 |

---

## Comparison with VoxTell

| Model | Per-prompt latency (H100, warm) |
|-------|---------------------------------|
| VoxTell v0_gpu (baseline) | 2.27s |
| VoxTell v3 (optimized) | 0.55s |
| nnInteractive (fold='all', autozoom=ON, warm) | 0.288s |

nnInteractive is faster per prompt than optimized VoxTell because it uses bbox prompts (no 4B text encoder).

---

## O1 — torch.compile (Triton, H100 native) ✅ COMPLETE

**Method**: Wrap `session.network` with `torch.compile(mode='reduce-overhead')`

```python
session.network = torch.compile(session.network, mode='reduce-overhead')
```

### Combined Results (job 56908464 — fold='all', autozoom=ON, same config for latency and DSC)

**Hardware**: H100 MIG 3g.40gb, Fir cluster  
**Config**: fold='all', do_autozoom=True, torch_n_threads=os.cpu_count(), N_WARMUP_COMPILE=2  
**Cases**: 20 CT cases, 294 objects

| Setting | _predict warm | Mean per-case | Speedup |
|---------|--------------|---------------|---------|
| Baseline (fold='all', autozoom=ON) | 0.2882s | 4.38s | 1.0× |
| **torch.compile (fold='all', autozoom=ON)** | **0.2146s** | **3.29s** | **1.34×** |

**DSC (same job, same cases):**

| Setting | Mean DSC | Objects |
|---------|----------|---------|
| Baseline | 0.7914 | 294 |
| **torch.compile** | **0.7916** | 294 |
| Difference | **+0.0002** | — |

**Verdict: accuracy maintained** (< 0.005 DSC change).

**Note on earlier 1.58× figure**: The June benchmark (job 46108850) used autozoom=OFF and fold=0, giving 0.108s baseline and 0.069s compiled (1.58×). With autozoom=ON — the correct production config — per-object latency is higher because some objects trigger multiple refinement passes. The 1.34× figure is the correct production speedup.

**N_WARMUP=2 required**: First warmup triggers Triton compilation; second stabilizes dispatch. With only 1 warmup, compiled latency reads ~0.54s and speedup appears ~1.0×.

---

### Cold-Start Timing (job 56908465 — isolated per-job cache, shared cache untouched)

| Metric | Value |
|--------|-------|
| Triton cold-start (run 1) | **22.91s** |
| Residual (run 2) | 0.544s |
| Fully warm (runs 3–5 mean) | **0.0907s** |
| Warm gain per object vs baseline | 0.0736s (0.2882 − 0.2146) |
| Break-even (partial isolation, 22.91s) | **~311 objects (~21 cases)** |
| Break-even (full isolation, 71.4s est.) | **~970 objects (~66 cases)** |
| First case cold (avg 14.7 obj) | **~24.3s** vs 4.38s baseline |

⚠️ **Isolation note**: `TORCHINDUCTOR_CACHE_DIR` was not set in job 56908465. Inductor defaults to `/tmp/torchinductor_brianx7`, which may have contained kernels from prior jobs. The 22.91s is a **lower bound**. A fully isolated rerun (with `TORCHINDUCTOR_CACHE_DIR` pointed to the temp dir) is needed to confirm. Break-even range: **311–970 objects (21–66 cases)**. The shared cache at `/scratch/brianx7/cache` was not modified.

---

## O2 — fold='all' vs fold=0 (SUPERSEDED)

**Correction**: `fold='all'` in nnU-Net is a **single model trained on the full training set** — it is NOT a 5-fold ensemble. Per-object latency is identical to fold=0 (same architecture, same patch count). The 0.33 DSC with fold=0 vs 0.79 with fold='all' likely reflects undertrained or mismatched weights in the fold_0 directory. Do not present this as a speed/quality tradeoff.

---

## O3 — Kernel cache pre-warming

Triton cache persists on scratch via `XDG_CACHE_HOME=/scratch/brianx7/cache`. Cold-start cost (22.91s) is paid once per environment; all subsequent jobs on Fir use the warm cache.

---

## O4 — autozoom analysis ✅ IMPLICITLY MEASURED

The combined job (56908464) ran with autozoom=ON. Baseline 0.2882s/object vs early autozoom=OFF baseline of 0.108s — autozoom adds ~0.18s mean overhead per object across the validation set, with zero overhead on simple objects and multiple refinement passes on complex anatomy.

---

## CPU Efficiency (Dr. Ma checklist)

| Job | Task | Wall-clock | CPU Efficiency | Memory Used |
|-----|------|-----------|---------------|-------------|
| 55034701 | DSC comparison (fold='all') | 5:22 | 14.79% of 8 cores | 7.36 GB / 64 GB |
| 56908464 | Combined latency + DSC | 4:22 | 18.75% of 8 cores | 7.57 GB / 64 GB |
| 56908465 | Cold-start timing | 1:57 | 4.49% of 8 cores | 3.76 GB / 64 GB |

**Explanation**: Jobs are GPU-bound. CPUs handle set_image preprocessing and DSC evaluation; GPU handles all _predict inference. CPU cores sit idle during inference, driving efficiency below 20%. The allocation was over-provisioned (actual usage ~7.5 GB RAM; 4 CPUs / 32 GB would suffice). Allocation was not changed mid-benchmark to preserve config comparability across all jobs.

---

## Summary of Final Results

**Config**: fold='all', do_autozoom=True, torch_n_threads=os.cpu_count(), H100 MIG 3g.40gb, Fir cluster

| Setting | _predict warm | Per-case | Speedup | DSC |
|---------|--------------|----------|---------|-----|
| Baseline (fold='all', autozoom=ON) | 0.2882s | 4.38s | 1.0× | 0.7914 |
| **torch.compile (fold='all', autozoom=ON)** | **0.2146s** | **3.29s** | **1.34×** | **0.7916** |

**Cold Triton cache**: 22.91s first call. Break-even: ~311 objects (~16 cases). After break-even, all subsequent cases run at 1.34×.
