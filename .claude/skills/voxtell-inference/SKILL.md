---
name: voxtell-inference
description: Load context for running VoxTell benchmarking, training, or evaluation on the Fir cluster. Invoke with /voxtell-inference before asking Claude to write, submit, or check any VoxTell experiment.
---

You are helping with VoxTell model inference and optimization on the Fir HPC cluster. Load the following context and use it to answer questions or write/submit jobs correctly.

## Project
VoxTell is the lab's model submitted to the CVPR 2025 medical image segmentation challenge.  
nnInteractive is a separate optimization study on the challenge baseline — see `/nninteractive-inference` for that context.

## Locations
| Resource | Path |
|----------|------|
| Local project | `c:\Users\brian\OneDrive\Desktop\Code\VoxTell-main` |
| Cluster project | `/scratch/brianx7/VoxTell-main` |
| GitHub | `https://github.com/pentaexe/VoxTell-main` |

## Cluster access
```
ssh brianx7@fir.alliancecan.ca
# Password → type 1 → Duo push on iPhone
```

## Environments

**Cluster — VoxTell jobs use the venv in /home, NOT the nnInteractive env:**
```bash
source /home/brianx7/envs/voxtell/bin/activate
```
It is a venv, not conda, so there is no `conda activate` step. It has
`transformers`, `positional_encodings`, `bitsandbytes` and `accelerate` (1.13.0).

`/scratch/brianx7/envs/nninteractive/` is for nnInteractive only. Pointing a
VoxTell job at it fails partway in with `ModuleNotFoundError: transformers` or
`positional_encodings` — this cost two jobs on 2026-08-26.

`accelerate` matters specifically: without it, `_load_text_backbone` catches the
ImportError and silently falls back to FP16, so a run labelled INT4 is not INT4.

**Local Windows:**
```
C:\Users\brian\miniconda3\envs\voxtell\python.exe
```
Has CUDA torch 2.8.0+cu126 on the RTX 4070 SUPER, plus bitsandbytes and
accelerate. The base miniconda env is CPU-only torch and will not run benchmarks.

## Benchmarked latencies
| Model / Config | Hardware | Per-prompt | Source |
|----------------|----------|-----------|--------|
| VoxTell v0_gpu (Qwen3 silently on CPU, FP32 VRAM overflow) | RTX 4070 SUPER | 3.10s | fair_benchmark_results.txt |
| VoxTell v3 (all opts, GPU-vs-GPU fair baseline) | RTX 4070 SUPER | 2.38s | fair_benchmark_results.txt |
| nnInteractive baseline (fold='all', autozoom=ON) | H100 MIG 3g.40gb | 0.2882s | job 56908464 |
| nnInteractive torch.compile | H100 MIG 3g.40gb | 0.2146s | job 56908464 |

VoxTell H100 numbers are unverified — benchmark_v0gpu_h100.py exists but may never have been run. Verify first: `sacct -u brianx7 --starttime=2026-03-01 --format=JobID,JobName%30,State,Start | grep -i vox`. If no VoxTell H100 job appears, submit benchmark_v0gpu_h100.sh (15 min, rrg-jma). RTX vs H100 is not directly comparable — do not state a ratio without same-hardware numbers.

## VoxTell changes (RTX 4070 SUPER, all from experiment_log.md)
| Change | Method | Speedup | DSC change |
|--------|--------|---------|-----------|
| **BUG FIX**: FP16 GPU placement | `dtype=torch.float16` on Qwen3 — was silently on CPU | 46.7× text encoding | < 0.001 |
| Sliding window overlap | tile_step 0.5 → 0.75 (343→125 patches) | 3.6× sliding window | +0.0006 |
| Embedding cache | LRU memory + SHA256 disk cache | 18.7× warm re-query | Identical |
| Numba preprocessing | `@njit(parallel=True)` crop + normalize | 1.4× preprocessing | Unchanged |

Fair GPU-vs-GPU algorithmic gain (bug fix excluded): **1.3×** (3.10s → 2.38s)

## Submit workflow
```bash
# Write/edit locally, then:
git add <file> && git commit -m "..." && git push
# On cluster:
cd /scratch/brianx7/VoxTell-main && git pull
sbatch <script>.sh
tail -f /scratch/brianx7/logs/<jobname>_<jobid>.out
```

## SLURM template (copy this for VoxTell jobs)
```bash
#!/bin/bash
#SBATCH --job-name=vox_<name>
#SBATCH --output=/scratch/brianx7/logs/vox_<name>_%j.out
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --mem=16G                 # 32G for CT volumes
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2         # jobs are GPU-bound; more looks bad in seff
#SBATCH --time=1:00:00
#SBATCH --account=rrg-jma

source /home/brianx7/envs/voxtell/bin/activate

export TORCH_HOME=/scratch/brianx7/torch_home
export XDG_CACHE_HOME=/scratch/brianx7/cache
export HF_HOME=/scratch/brianx7/hf_cache   # compute nodes have no internet;
                                           # without this from_pretrained fails

cd /scratch/brianx7/VoxTell-main
python -u <script>.py
```

## Dr. Ma's review checklist
- Report CPU efficiency (`seff <jobid>`) for every submitted job
- GPU comparison must use the same checkpoint and fold (no fold=0 vs fold='all' mixing)
- DSC accuracy must be verified for any speed optimization
- Always use `use_fold='all'` for nnInteractive; never fold=0
