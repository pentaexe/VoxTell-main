#!/bin/bash
#SBATCH --job-name=vox_fair_h100
#SBATCH --output=/scratch/brianx7/logs/vox_fair_h100_%j.out
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0:15:00
#SBATCH --account=rrg-jma

# Use voxtell env — nninteractive env does not have transformers
# Check: ls /scratch/brianx7/envs/ to verify env name, then update this line
source /scratch/brianx7/envs/voxtell/bin/activate

export TORCH_HOME=/scratch/brianx7/torch_home
export XDG_CACHE_HOME=/scratch/brianx7/cache

cd /scratch/brianx7/VoxTell-main

python -u fair_benchmark.py
