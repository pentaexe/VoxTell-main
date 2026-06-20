#!/bin/bash
#SBATCH --job-name=nni_bench
#SBATCH --output=/scratch/brianx7/logs/nni_bench_%j.out
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=0:30:00

cd /scratch/brianx7/VoxTell-main

python -u nninteractive_benchmark.py \
    --input_dir /scratch/brianx7/cvpr_val/3D_val_CT