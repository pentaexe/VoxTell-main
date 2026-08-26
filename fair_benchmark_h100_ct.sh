#!/bin/bash
#SBATCH --job-name=vox_fair_ct
#SBATCH --output=/scratch/brianx7/logs/vox_fair_ct_%j.out
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --account=rrg-jma

# Runs the fair benchmark against a CVPR validation CT volume instead of the MNI
# brain. The brain is 189x233x197 against a 192^3 patch: 4 patches at any
# tile_step, and arrays too small for Numba to pay off. On CT both optimizations
# have room to act, and the result is measured on the same data type as every
# nnInteractive number in the deck.

source /home/brianx7/envs/voxtell/bin/activate

export TORCH_HOME=/scratch/brianx7/torch_home
export XDG_CACHE_HOME=/scratch/brianx7/cache
export HF_HOME=/scratch/brianx7/hf_cache

cd /scratch/brianx7/VoxTell-main

# First CT case, matching the ordering nni_combined.py uses (sorted CT_*.npz).
CT_CASE=$(ls -1 /scratch/brianx7/cvpr_val/3D_val_npz/CT_*.npz | sort | head -1)
echo "CT case: $CT_CASE"

export BENCH_IMAGE="$CT_CASE"
# BENCH_PROMPTS deliberately unset: these npz cases carry a 'text_prompts' key,
# so the script reads the correct anatomy from the file rather than guessing it.

python -u fair_benchmark.py
