#!/bin/bash
#SBATCH --job-name=vox_int4_dsc
#SBATCH --output=/scratch/brianx7/logs/vox_int4_dsc_%j.out
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=1:00:00
#SBATCH --account=rrg-jma

source /home/brianx7/envs/voxtell/bin/activate

export TORCH_HOME=/scratch/brianx7/torch_home
export XDG_CACHE_HOME=/scratch/brianx7/cache
export HF_HOME=/scratch/brianx7/hf_cache

cd /scratch/brianx7/VoxTell-main

python -u int4_dsc_comparison.py
