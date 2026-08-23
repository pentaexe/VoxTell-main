---
name: voxtell-inference
description: Run VoxTell model inference, training, or evaluation. Use when the user asks about VoxTell benchmarking, timing, or comparison against nnInteractive.
---

# VoxTell Inference

## Project location
- Local: `c:\Users\brian\OneDrive\Desktop\Code\VoxTell-main`
- Cluster: `/scratch/brianx7/VoxTell-main`
- GitHub: `https://github.com/pentaexe/VoxTell-main`

## Cluster access
```
ssh brianx7@fir.alliancecan.ca
# Password → type 1 for Duo push → approve on iPhone
```

## Key benchmarked latencies (H100 MIG 3g.40gb)
| Model | Per-prompt latency |
|-------|-------------------|
| VoxTell v0_gpu (baseline) | 2.27s |
| VoxTell v3 (optimized) | 0.55s |
| nnInteractive (fold='all', warm) | 0.108s |

nnInteractive is ~5× faster per prompt than optimized VoxTell because it uses bbox prompts (no 4B text encoder).

## Context
VoxTell is the lab's own model submitted to the CVPR 2025 medical image segmentation challenge. The nnInteractive work is a separate optimization study on the challenge baseline model, requested by Dr. Jun Ma.

## Conda environment (local / Fir)
```bash
source /scratch/brianx7/envs/nninteractive/bin/activate  # cluster
# or
conda activate voxtell  # local Windows
```

## Workflow for any new experiment
1. Write/edit the script locally in `VoxTell-main/`
2. `git add <file> && git commit -m "..." && git push`
3. On cluster: `cd /scratch/brianx7/VoxTell-main && git pull`
4. `sbatch <script>.sh`
5. Check output: `tail -f /scratch/brianx7/logs/<jobname>_<jobid>.out`

## SLURM requirements (always include)
```
#SBATCH --account=rrg-jma
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --time=1:00:00   # keep short for faster queue priority
```

## Dr. Ma's review checklist
- CPU efficiency must be reported
- GPU comparison must be fair (same checkpoint, same fold)
- DSC accuracy must be verified for any speed optimization
- Always use fold='all' (official checkpoint) for nnInteractive comparisons
