---
name: nninteractive-inference
description: Load context for running nnInteractive benchmarking, DSC evaluation, or torch.compile jobs on the Fir cluster. Invoke with /nninteractive-inference before asking Claude to write, submit, or check any nnInteractive experiment.
---

You are helping with nnInteractive inference experiments on the Fir HPC cluster. Load the following context and use it to answer questions or write/submit jobs correctly.

## Cluster access
```
ssh brianx7@fir.alliancecan.ca
# Password → type 1 → Duo push on iPhone
```

## Key paths
| Resource | Path |
|----------|------|
| Checkpoint | `/scratch/brianx7/nnInteractive_weights/nnInteractive_v1.0` |
| Validation data | `/scratch/brianx7/cvpr_val/3D_val_npz` |
| Ground truth | `/scratch/brianx7/cvpr_val/3D_val_gt/3D_val_gt_interactive` |
| Conda env | `source /scratch/brianx7/envs/nninteractive/bin/activate` |
| Logs | `/scratch/brianx7/logs/` |
| Project | `/scratch/brianx7/VoxTell-main` |

## Checkpoint rules (critical)
- Always use `use_fold='all'` — the official CVPR 2025 checkpoint, DSC ~0.79
- `use_fold=0` gives ~0.33 DSC — likely undertrained/mismatched weights; never use for comparisons
- `torch_n_threads=os.cpu_count()` is required — using `SLURM_CPUS_PER_TASK` hangs fold='all'

## Session pattern (copy exactly)
```python
import os, torch
from nnInteractive.inference.inference_session import nnInteractiveInferenceSession

session = nnInteractiveInferenceSession(
    device=torch.device('cuda', 0),
    use_torch_compile=False,
    verbose=False,
    torch_n_threads=os.cpu_count(),   # must be os.cpu_count()
    do_autozoom=True,
    use_pinned_memory=True,
)
session.initialize_from_trained_model_folder(
    '/scratch/brianx7/nnInteractive_weights/nnInteractive_v1.0',
    use_fold='all',
)
# To enable torch.compile:
# session.network = torch.compile(session.network, mode='reduce-overhead')
```

## Per-object inference
```python
session.reset_interactions()
session.add_bbox_interaction(bbox, include_interaction=True, run_prediction=False)
session.new_interaction_centers          = [session.new_interaction_centers[-1]]
session.new_interaction_zoom_out_factors = [session.new_interaction_zoom_out_factors[-1]]
session._predict()
```

## torch.compile requirements
- `N_WARMUP=2` for the compiled session — warmup 1 triggers Triton compilation, warmup 2 stabilizes dispatch
- Set `TORCHINDUCTOR_CACHE_DIR` **before** `import torch` (inductor reads it at import time)
- Warm cache lives at `XDG_CACHE_HOME=/scratch/brianx7/cache`

## Key verified results (fold='all', autozoom=ON, H100 MIG 3g.40gb, n=4 jobs)

Per-job paired speedup — cite the mean and the range, never a single job:

| Job | Baseline _predict | Compiled _predict | Gain/obj | Speedup | DSC Δ |
|-----|------------------|-------------------|----------|---------|-------|
| 56908464 | 0.2882s | 0.2146s | 0.0736s | 1.34× | +0.0002 |
| 56923894 | 0.2881s | 0.2068s | 0.0813s | 1.39× | +0.0000 |
| 56923895 | 0.2722s | 0.2128s | 0.0594s | 1.28× | +0.0004 |
| 56923896 | 0.2947s | 0.2236s | 0.0711s | 1.32× | +0.0002 |
| **Mean (n=4)** | | | **0.0714s** | **1.33×** | **≤ +0.0004** |

- Speedup to cite: **1.33× mean, range 1.28–1.39× (n=4)** — all four are valid paired
  within-job ratios; cite the mean and range, never a single job
- DSC: 0.7914 → 0.7916 in job 56908464; all four runs bracket zero (Δ ≤ +0.0004)
- Triton cold-start: **23.61s** (fully isolated, /tmp-backed, job 56914757; /scratch NFS will be higher — lower bound)
- Break-even: **~331 objects (~22 cases)** — 23.61s ÷ 0.0714s/object mean gain (n=4), 14.7 obj/case
- autozoom adds zero measurable overhead on this validation set (no zoom-out passes triggered)

**Caveat to state when reporting DSC**: at `N_CASES = 20` (294 objects) a delta of
+0.0002 sits inside sampling noise. The honest claim is that four runs bracket zero,
not that any single run proves equivalence. A full 881-case run is the stronger answer.
Do not invoke a 0.005 pass/fail threshold — it is self-assigned, not from the challenge.

## SLURM template
```bash
#!/bin/bash
#SBATCH --job-name=nni_job
#SBATCH --output=/scratch/brianx7/logs/nni_job_%j.out
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --account=rrg-jma

source /scratch/brianx7/envs/nninteractive/bin/activate
export TORCH_HOME=/scratch/brianx7/torch_home
export XDG_CACHE_HOME=/scratch/brianx7/cache
cd /scratch/brianx7/VoxTell-main
python -u <script>.py
```

## Submit workflow
```bash
# Always write scripts locally, then:
git add <file> && git commit -m "..." && git push
# On cluster:
cd /scratch/brianx7/VoxTell-main && git pull
sbatch <script>.sh
tail -f /scratch/brianx7/logs/<jobname>_<jobid>.out
```

## Common issues
- `TritonMissing`: `pip install --no-index triton` in the nninteractive env
- fold='all' hangs: check `torch_n_threads=os.cpu_count()`
- Stale inductor cache: set all three env vars before `import torch` (`XDG_CACHE_HOME`, `TORCH_HOME`, `TORCHINDUCTOR_CACHE_DIR`)
- Job queues slow: keep `--time` at 1:00:00 for higher priority
- Never paste large scripts in the terminal — git push / git pull instead (heredocs corrupt)
