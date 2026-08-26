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

## Conda environments
```bash
source /scratch/brianx7/envs/nninteractive/bin/activate   # cluster (nnInteractive + VoxTell deps)
conda activate voxtell                                      # local Windows
```

## Key benchmarked latencies (H100 MIG 3g.40gb, Fir cluster)
| Model / Config | Per-prompt latency |
|----------------|-------------------|
| VoxTell v0_gpu (baseline, CPU inference) | 2.27s |
| VoxTell v3 (FP16 GPU + sliding window + embedding cache + Numba) | 0.55s |
| nnInteractive baseline (fold='all', autozoom=ON) | 0.2882s |
| nnInteractive torch.compile (fold='all', autozoom=ON) | 0.2146s |

nnInteractive is ~5× faster per prompt than optimized VoxTell because it uses bbox prompts and has no large text encoder.

## VoxTell optimizations (all verified, DSC maintained)
| Optimization | Method | Speedup | DSC change |
|-------------|--------|---------|-----------|
| FP16 GPU placement | `dtype=torch.float16` on Qwen3 | 46.7× text encoding | < 0.001 |
| Sliding window overlap | tile_step 0.5 → 0.75 (343→125 patches) | 3.6× | +0.0006 |
| Embedding cache | LRU memory + SHA256 disk cache | 18.7× warm | Identical |
| Numba preprocessing | `@njit(parallel=True)` crop + normalize | 1.4× | Unchanged |

## Submit workflow
```bash
# Write/edit locally, then:
git add <file> && git commit -m "..." && git push
# On cluster:
cd /scratch/brianx7/VoxTell-main && git pull
sbatch <script>.sh
tail -f /scratch/brianx7/logs/<jobname>_<jobid>.out
```

## SLURM requirements (always include)
```bash
#SBATCH --account=rrg-jma
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --time=1:00:00
source /scratch/brianx7/envs/nninteractive/bin/activate
export TORCH_HOME=/scratch/brianx7/torch_home
export XDG_CACHE_HOME=/scratch/brianx7/cache
```

## Dr. Ma's review checklist
- Report CPU efficiency (`seff <jobid>`) for every submitted job
- GPU comparison must use the same checkpoint and fold (no fold=0 vs fold='all' mixing)
- DSC accuracy must be verified for any speed optimization
- Always use `use_fold='all'` for nnInteractive; never fold=0
