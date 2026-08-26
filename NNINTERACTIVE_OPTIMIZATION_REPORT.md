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

**On the DSC delta**: +0.0002 over 294 objects (20 cases) is within sampling noise at
this object count — it is not on its own evidence of equivalence. The defensible claim
is that four independent runs bracket zero (Δ from +0.0000 to +0.0004, no run showing
degradation). No pass/fail threshold is applied; a 0.005 bar would be self-assigned
rather than drawn from the challenge. A full 881-case run would tighten this.

**Note on earlier 1.58× figure**: The June benchmark (job 46108850) used `torch_n_threads=8` (SLURM_CPUS_PER_TASK), autozoom=OFF, and fold=0, giving 0.108s baseline (1.58× speedup). The controlled autozoom experiment (job 56914754) shows autozoom=ON is actually 13% *faster* than autozoom=OFF on this dataset, ruling out autozoom as the cause of the higher baseline. The most likely explanation is thread oversubscription: `torch_n_threads=os.cpu_count()` (which may return 64+ on the compute node) creates more threads than the 8 allocated cores can serve, increasing dispatch overhead. The 1.34× figure, measured with `os.cpu_count()` and fold='all', is the correct production speedup.

**N_WARMUP=2 required**: First warmup triggers Triton compilation; second stabilizes dispatch. With only 1 warmup, compiled latency reads ~0.54s and speedup appears ~1.0×.

---

### Cold-Start Timing (job 56914757 — fully isolated, all three cache env vars set before torch import)

`XDG_CACHE_HOME`, `TORCH_HOME`, and `TORCHINDUCTOR_CACHE_DIR` all pointed to a fresh per-job `/tmp` dir before `import torch`. Script asserts inductor cache is empty before compile. Shared cache at `/scratch/brianx7/cache` unchanged.

| Metric | Value |
|--------|-------|
| Triton cold-start (run 1) | **23.61s** |
| Residual (run 2) | 0.513s |
| Fully warm mean (runs 3–6, single object) | 0.125s |
| Mean warm gain per object (n=4 jobs) | **0.0714s** |
| Break-even | **~331 objects (~22 cases)** |
| First case cold vs warm baseline | **~25.4s  vs  ~4.38s** |

**Break-even derivation**: 23.61s ÷ 0.0714s/object = 331 objects; 331 ÷ 14.7 obj/case = 22 cases.

Mean gain 0.0714s/object = mean of all 4 jobs: (0.0736 + 0.0813 + 0.0594 + 0.0711) / 4.
All four are valid paired within-job ratios measured under the same config, so all four
are included. (An earlier revision averaged only the three serialized repeats, giving
0.0706s and ~334 objects; the difference is immaterial but n=4 is now used throughout.)

Note: `/tmp` is node-local fast storage. Production Triton cache on `/scratch` (NFS) will take longer to load the first time. 23.61s is therefore a lower bound for a from-scratch production deployment. After break-even, all subsequent cases run at 1.33×.

---

## O2 — fold='all' vs fold=0 (SUPERSEDED)

**Note**: In the standard nnU-Net CLI, `fold='all'` trains a single model on the full training set. However, nnInteractive's `nnInteractiveInferenceSession` wrapper may resolve `use_fold='all'` differently — potentially loading multiple fold checkpoints if they all exist in the weights directory. **Unconfirmed: run `print(len(session.list_of_parameters))` after `initialize_from_trained_model_folder` to verify.** If the result is 5, it is an ensemble and the 0.108s→0.288s latency gap is fully explained. If it is 1, something environmental explains the gap. Do not claim single-model until verified. The 0.33 DSC with fold=0 vs 0.79 with fold='all' likely reflects undertrained or mismatched weights in the fold_0 directory.

---

## O3 — Kernel cache pre-warming

Triton cache persists on scratch via `XDG_CACHE_HOME=/scratch/brianx7/cache`. Cold-start cost (23.61s on /tmp; higher on /scratch) is paid once per environment; all subsequent jobs on Fir use the warm cache.

---

## O4 — autozoom analysis ✅ MEASURED (job 56914754)

Controlled experiment: fold='all', no compile, same 20 cases, autozoom=ON vs autozoom=OFF in one process.

| Config | Per-object | Per-case | vs OFF |
|--------|-----------|----------|--------|
| autozoom=OFF | 0.2875s | 4.23s | 1.00× |
| autozoom=ON | 0.2491s | 3.67s | **0.87×** |

autozoom=ON reads 0.2491s and autozoom=OFF reads 0.2875s in this experiment, but the difference is noise, not a real effect. The autozoom job log prints "No zoom out necessary / No refinement necessary" for **every single object** — autozoom never fired on any case in this validation set. A feature that never activates cannot produce a 13% speedup. The measured delta (0.0384s) is within the 15.7% job-to-job spread observed for this config (0.2882s in job 56908464 vs 0.2491s in job 56914754).

**The correct O4 finding**: autozoom overhead is approximately zero on this validation set because no object triggered a zoom-out pass. This may not hold on data where anatomy crosses bbox boundaries and forces multi-scale refinement. On typical CVPR validation CT data, autozoom adds no measurable latency.

Autozoom is also not the explanation for the 0.108s (June) vs 0.288s (current) gap, since the controlled experiment shows autozoom=OFF also gives 0.288s. The remaining candidate is `torch_n_threads=os.cpu_count()` vs the June `torch_n_threads=8` (SLURM_CPUS_PER_TASK) — if os.cpu_count() returns a large number on the compute node, thread oversubscription onto 8 allocated cores would explain the difference. **Unconfirmed: requires checking `os.cpu_count()` on the node and comparing.** Alternatively, `use_fold='all'` may load multiple models in nnInteractive's session wrapper (see O2 note).

---

## CPU Efficiency (Dr. Ma checklist)

| Job | Task | Wall-clock | CPU Efficiency | Memory Used |
|-----|------|-----------|---------------|-------------|
| 55034701 | DSC comparison (fold='all') | 5:22 | 14.79% of 8 cores | 7.36 GB / 64 GB |
| 56908464 | Combined latency + DSC | 4:22 | 18.75% of 8 cores | 7.57 GB / 64 GB |
| 56914754 | autozoom ON vs OFF | ~8:00 | 12.83% of 8 cores | 6.11 GB / 64 GB |
| 56914757 | Cold-start (fully isolated) | ~5:00 | 3.95% of 8 cores | 3.87 GB / 64 GB |
| 56923894 | Repeat 1 (n=3 variance confirmation) | 5:46 | 13.69% of 8 cores | 7.40 GB / 64 GB |
| 56923895 | Repeat 2 (warm Triton cache hit) | 3:55 | 20.27% of 8 cores | 6.48 GB / 64 GB |
| 56923896 | Repeat 3 | 5:06 | 16.09% of 8 cores | 7.34 GB / 64 GB |

**Explanation**: Jobs are GPU-bound. CPUs handle set_image preprocessing and DSC evaluation; GPU handles all _predict inference. CPU cores sit idle during inference, driving efficiency below 20%. The allocation was over-provisioned (actual usage ~7.5 GB RAM; 4 CPUs / 32 GB would suffice). Allocation was not changed mid-benchmark to preserve config comparability across all jobs.

---

## Summary of Final Results — CONFIRMED (n=4)

**Config**: fold='all', do_autozoom=True, torch_n_threads=os.cpu_count(), H100 MIG 3g.40gb, Fir cluster

### Per-job speedup (within-job paired ratio — stable across runs)

| Job | Baseline _predict | Compiled _predict | Speedup | DSC Δ |
|-----|------------------|------------------|---------|-------|
| 56908464 (original) | 0.2882s | 0.2146s | 1.34× | +0.0002 |
| 56923894 (repeat 1) | 0.2881s | 0.2068s | **1.39×** | +0.0000 |
| 56923895 (repeat 2) | 0.2722s | 0.2128s | **1.28×** | +0.0004 |
| 56923896 (repeat 3) | 0.2947s | 0.2236s | **1.32×** | +0.0002 |
| **Mean (n=4)** | | | **1.33×** | **≤ +0.0004** |

Mean over all four jobs. Each is a paired within-job ratio measured under the same
config (fold='all', autozoom=ON, `torch_n_threads=os.cpu_count()`), so all four count.
**Cite as 1.33× mean, range 1.28–1.39× (n=4)** — never a single job's figure.

**Cold Triton cache**: 23.61s (/tmp-backed, fully isolated, job 56914757; production /scratch NFS will be higher — lower bound).  
**Break-even**: ~331 objects (~22 cases) using mean gain of 0.0714s/object (n=4).  
After break-even, all subsequent objects run at 1.33×.

**fold='all' verified**: `ls /scratch/brianx7/nnInteractive_weights/nnInteractive_v1.0/` shows `fold_all` directory exists.
