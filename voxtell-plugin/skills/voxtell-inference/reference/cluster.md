# Fir cluster reference

## Access

```bash
ssh brianx7@fir.alliancecan.ca
# password, then type 1, then approve the Duo push
```

## Paths

| Resource | Path |
|---|---|
| Cluster project | `/scratch/brianx7/VoxTell-main` |
| Local project | `c:\Users\brian\OneDrive\Desktop\Code\VoxTell-main` |
| Logs | `/scratch/brianx7/logs/` |
| CVPR validation images | `/scratch/brianx7/cvpr_val/3D_val_npz` |
| CVPR ground truth | `/scratch/brianx7/cvpr_val/3D_val_gt/3D_val_gt_interactive` |
| nnInteractive weights | `/scratch/brianx7/nnInteractive_weights/nnInteractive_v1.0` |

## Environments

These are venvs, not conda. There is no `conda activate` on Fir.

```bash
source /home/brianx7/envs/voxtell/bin/activate            # VoxTell jobs
source /scratch/brianx7/envs/nninteractive/bin/activate   # nnInteractive jobs
```

**Getting this wrong does not fail at startup.** A VoxTell job pointed at the
nnInteractive env dies about a minute in with `ModuleNotFoundError: transformers`
or `positional_encodings`. That cost two jobs on 2026-08-26.

The voxtell env also carries `accelerate` 1.13.0, which matters more than it
looks: without it `_load_text_backbone` swallows the ImportError and quietly
serves FP16 under an INT4 label.

## Git flows one direction

Commit and push from the local Windows machine, then `git pull` on Fir. Never
commit on the cluster: the login node has no push credentials and no configured
git identity, so a commit made there is stamped with the wrong author and then
stalls on a credential prompt.

If one lands there anyway:

```bash
git status
git reset --hard HEAD~1
```

Redo it locally, push, and pull on the cluster.

Keep local and cluster command blocks visually separate. A Windows `cd` in a
cluster block fails silently and everything after it runs in the wrong directory.

## Allocation

`rrg-jma` is the group name and what `--account=` wants. `axc-572-ac` is the same
allocation's RAPI, which is what CCDB displays. They are not two options.

Do **not** use `def-jma-ab` (RAPI `axc-572-ab`): the no-RAC default, low priority,
and it does not draw on the lab's 10 RGU-years.

## GPU request

| Request | RGU | Use |
|---|---|---|
| `h100:1` (80 GB) | 12.2 | only if the model needs 80 GB |
| `nvidia_h100_80gb_hbm3_3g.40gb:1` | 6.1 | default for these jobs |
| `nvidia_h100_80gb_hbm3_2g.20gb:1` | 3.5 | debugging |

## SLURM template

```bash
#!/bin/bash
#SBATCH --job-name=vox_<name>
#SBATCH --output=/scratch/brianx7/logs/vox_<name>_%j.out
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --mem=16G                 # 32G for CT volumes
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2         # GPU-bound; more looks wasteful in seff
#SBATCH --time=1:00:00
#SBATCH --account=rrg-jma

source /home/brianx7/envs/voxtell/bin/activate

export TORCH_HOME=/scratch/brianx7/torch_home
export XDG_CACHE_HOME=/scratch/brianx7/cache
export HF_HOME=/scratch/brianx7/hf_cache   # compute nodes have no internet

cd /scratch/brianx7/VoxTell-main
python -u <script>.py
```

Validate it before submitting:

```bash
python scripts/validate_job.py <script>.sh
```

## Submitting

```bash
cd /scratch/brianx7/VoxTell-main && git pull
sbatch <script>.sh
squeue -u brianx7
tail -f /scratch/brianx7/logs/vox_<name>_<jobid>.out
seff <jobid>                       # report this with every result
```

## Measured CPU efficiency

These jobs are GPU-bound. Across seven jobs CPU efficiency ran 3 to 20% of the
cores requested, and CPU time held near constant (6:18 to 6:33) while walltime
varied. Request 2 cores, not 8, and report `seff` with the result.
