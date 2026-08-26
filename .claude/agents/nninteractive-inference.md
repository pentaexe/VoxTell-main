---
name: nninteractive-inference
description: Run nnInteractive inference on the Fir cluster — benchmarking, DSC evaluation, or torch.compile comparison. Use when the user wants to run, check, or submit any nnInteractive job.
---

# nnInteractive Inference on Fir

## Cluster access
```
ssh brianx7@fir.alliancecan.ca
# Password → then type 1 for Duo push → approve on iPhone
```

## Key paths
| Resource | Path |
|----------|------|
| Checkpoint | `/scratch/brianx7/nnInteractive_weights/nnInteractive_v1.0` |
| Validation input | `/scratch/brianx7/cvpr_val/3D_val_npz` |
| Ground truth | `/scratch/brianx7/cvpr_val/3D_val_gt/3D_val_gt_interactive` |
| Conda env | `/scratch/brianx7/envs/nninteractive` (Python 3.12) |
| Logs | `/scratch/brianx7/logs/` |
| Project | `/scratch/brianx7/VoxTell-main` |

## Official checkpoint
Always use `use_fold='all'` — this is the official CVPR 2025 checkpoint (single model trained on full training set, NOT a 5-fold ensemble).  
`use_fold=0` gives ~0.33 DSC — likely undertrained/mismatched weights, not a fold-quality difference. Never use fold=0 for accuracy comparisons.

## Session pattern (copy exactly — fold='all' hangs with different configs)
```python
import os, torch
from nnInteractive.inference.inference_session import nnInteractiveInferenceSession

session = nnInteractiveInferenceSession(
    device=torch.device('cuda', 0),
    use_torch_compile=False,
    verbose=False,
    torch_n_threads=os.cpu_count(),   # must be os.cpu_count(), not SLURM_CPUS_PER_TASK
    do_autozoom=True,
    use_pinned_memory=True,
)
session.initialize_from_trained_model_folder(
    '/scratch/brianx7/nnInteractive_weights/nnInteractive_v1.0',
    use_fold='all',
)
# Optionally add torch.compile:
# session.network = torch.compile(session.network, mode='reduce-overhead')
```

## Per-object inference pattern
```python
session.reset_interactions()
session.add_bbox_interaction(bbox, include_interaction=True, run_prediction=False)
session.new_interaction_centers          = [session.new_interaction_centers[-1]]
session.new_interaction_zoom_out_factors = [session.new_interaction_zoom_out_factors[-1]]
session._predict()
```

## Submitting jobs
```bash
cd /scratch/brianx7/VoxTell-main && git pull
sbatch nni_dsc_foldall.sh     # DSC comparison: baseline vs torch.compile
```

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

## Key results (job 56908464, fold='all', autozoom=ON, H100 MIG)
- Baseline warm: **0.2882s** per object, **4.38s** per case
- torch.compile warm: **0.2146s** per object, **3.29s** per case → **1.34× speedup**
- DSC: Baseline 0.7914 → Compiled 0.7916 (**+0.0002**, accuracy maintained)
- Triton cold-start: 22.91s (partial isolation) to 71.4s (full isolation) — range pending clean run
- Break-even: 311–970 objects (21–66 cases)
- Note: earlier 1.58× figure used autozoom=OFF/fold=0 — not the production config

## Common issues
- `TritonMissing`: run `source /scratch/brianx7/envs/nninteractive/bin/activate && pip install --no-index triton`
- fold='all' hangs: ensure `torch_n_threads=os.cpu_count()` (not capped to SLURM_CPUS_PER_TASK)
- Job queues too long: reduce `--time` to 1:00:00 for higher priority
- Always push scripts via GitHub then `git pull` on cluster — never paste heredocs in terminal
