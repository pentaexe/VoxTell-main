#!/bin/bash
#SBATCH --job-name=nni_mcp_verify
#SBATCH --account=rrg-jma
#SBATCH --gpus=h100_3g.40gb:1
#SBATCH --cpus-per-task=2
#SBATCH --ntasks=1
#SBATCH --mem=48G
#SBATCH --time=1:00:00
#SBATCH --output=/scratch/$USER/logs/nni_mcp_verify_%j.out

# Verify nninteractive_segment end to end through the MCP server, against the
# real fold='all' weights, and score the result against ground truth.
#
# The nnInteractive venv, not the VoxTell one: this path needs nnInteractive and
# does not touch transformers. Pointing it at the voxtell env dies about a
# minute in on ModuleNotFoundError, after the queue wait.
source /scratch/$USER/envs/nninteractive/bin/activate

export TORCH_HOME=/scratch/$USER/torch_home
export XDG_CACHE_HOME=/scratch/$USER/cache

cd /scratch/$USER/VoxTell-main

# -u so the log fills as the job runs. Without it a hang and steady progress
# look identical until the job exits.
python -u nni_mcp_verify.py
