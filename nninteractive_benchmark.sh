#!/bin/bash
#SBATCH --job-name=nni_bench
#SBATCH --output=/scratch/brianx7/logs/nni_bench_%j.out
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --time=3:00:00

source ~/envs/nninteractive/bin/activate

cd /scratch/brianx7/VoxTell-main

python -u nninteractive_benchmark.py \
    --input_dir /scratch/brianx7/cvpr_val/3D_val_npz \
    --skip_compile