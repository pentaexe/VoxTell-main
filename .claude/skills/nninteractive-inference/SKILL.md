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

## Key verified results (job 56908464, fold='all', autozoom=ON, H100 MIG 3g.40gb)
| Setting | Per object | Per case | Speedup | DSC |
|---------|-----------|----------|---------|-----|
| Baseline | 0.2882s | 4.38s | 1.0× | 0.7914 |
| torch.compile | 0.2146s | 3.29s | **1.34×** | 0.7916 |

- Triton cold-start: **23.61s** (fully isolated, /tmp-backed, job 56914757)
- Break-even: **~321 objects (~22 cases)** — 23.61s ÷ 0.0736s/object gain
- autozoom adds zero measurable overhead on this validation set (no zoom-out passes triggered)

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
