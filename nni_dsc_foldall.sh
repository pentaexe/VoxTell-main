#!/bin/bash
#SBATCH --job-name=nni_dsc_foldall
#SBATCH --output=/scratch/brianx7/logs/nni_dsc_foldall_%j.out
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=3:00:00
#SBATCH --account=rrg-jma

source /scratch/brianx7/envs/nninteractive/bin/activate

export TORCH_HOME=/scratch/brianx7/torch_home
export XDG_CACHE_HOME=/scratch/brianx7/cache

cd /scratch/brianx7/VoxTell-main

python -u nni_dsc_foldall.py