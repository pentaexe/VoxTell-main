#!/bin/bash
#SBATCH --job-name=nni_coldstart
#SBATCH --output=/scratch/brianx7/logs/nni_coldstart_%j.out
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0:30:00
#SBATCH --account=rrg-jma

source /scratch/brianx7/envs/nninteractive/bin/activate

# Do NOT set XDG_CACHE_HOME or TORCH_HOME here.
# The Python script sets them to a per-job temp dir so the shared
# cache at /scratch/brianx7/cache is not touched.

cd /scratch/brianx7/VoxTell-main

python -u nni_coldstart.py
