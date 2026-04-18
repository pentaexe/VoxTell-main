# ComputeCanada Workflow — Brian Xiao
**Lab:** Peter Munk Cardiac Centre AI Team (Dr. Jun Ma)  
**Username:** brianx7  
**Account:** axc-572-ac (RRG #5772 — Foundation Models for Medical Image Segmentation)  
**Clusters:** Trillium (short jobs <1 day) · Fir (long jobs >1 day, lab has 10 RGU-years allocation)

---

## 1. Initial Setup (one-time)

### 1.1 Generate and install SSH key
```bash
# On local machine
ssh-keygen -b 4096 -t rsa -f ~/.ssh/ccdb

# Upload public key to CCDB
cat ~/.ssh/ccdb.pub
# Paste into: https://ccdb.computecanada.ca/ssh_authorized_keys
```

### 1.2 SSH config (`~/.ssh/config`)
```
Host *
    ServerAliveInterval 300

Host trillium-gpu.scinet.utoronto.ca
    IdentityFile ~/.ssh/ccdb
    User brianx7

Host fir.alliancecan.ca
    IdentityFile ~/.ssh/ccdb
    User brianx7

# Fir compute nodes (for VS Code direct connection)
Host fc?????
    ProxyJump fir.alliancecan.ca
    IdentityFile ~/.ssh/ccdb
    User brianx7
```

### 1.3 Log in
```bash
# Trillium (short jobs)
ssh -i ~/.ssh/ccdb -Y brianx7@trillium-gpu.scinet.utoronto.ca

# Fir (long jobs, lab allocation)
ssh -i ~/.ssh/ccdb -Y brianx7@fir.alliancecan.ca
```

### 1.4 Create virtual environment (on login node only, quickly)
```bash
module load python/3.10
virtualenv --no-download ~/envs/voxtell
source ~/envs/voxtell/bin/activate
pip install --no-index --upgrade pip

# Install CC-optimised wheels first (faster, no dependency conflicts)
pip install numpy --no-index
pip install torch --no-index

# Packages not in CC wheels — install from PyPI directly
pip install nnunetv2 batchgenerators
```

### 1.5 Set Hugging Face cache to scratch (large models)
```bash
export HF_HOME=/scratch/$USER/hf_cache
# Add to ~/.bashrc so it persists
echo 'export HF_HOME=/scratch/$USER/hf_cache' >> ~/.bashrc
```

---

## 2. Starting a Job

### 2.1 GPU resources — what to request
The lab has **10 RGU-years** exclusive allocation on **Fir**. Use it.

| Request | RGUs used | Use case |
|---------|-----------|----------|
| `h100:1` (80GB) | 12.2 | Full H100 — only for very large models |
| `nvidia_h100_80gb_hbm3_3g.40gb:1` | 6.1 | Half H100 — recommended for most jobs |
| `nvidia_h100_80gb_hbm3_2g.20gb:1` | 3.5 | Quarter H100 — inference/debugging |

**Rule:** Request 20/40G H100 (MIG instance) rather than the full 80G unless model requires it.

### 2.2 Example batch script (`train.sh`)
```bash
#!/bin/bash
#SBATCH --account=axc-572-ac
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --mem=140G
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --time=0-12:00
#SBATCH --mail-user=YOUR_EMAIL
#SBATCH --mail-type=ALL
#SBATCH --output=/scratch/$USER/slurm_logs/%x_%j.log

module load python/3.10
source ~/envs/voxtell/bin/activate

python train.py
```

Submit with:
```bash
sbatch train.sh
```

### 2.3 Interactive session (for debugging)
```bash
salloc --time=1:0:0 --mem=32G --ntasks=1 \
       --gpus=nvidia_h100_80gb_hbm3_2g.20gb:1 \
       --account=axc-572-ac
```

**CRITICAL: Never run jobs >1 minute on the login node.**

---

## 3. Monitoring Computing Usage

This is the most important section — if GPU is underutilised, the **entire lab's job priority drops**.

### 3.1 Check if your job is running
```bash
squeue -u $USER   # or: sq -u $USER
```

### 3.2 Monitor GPU utilisation (live) — SSH to running node
```bash
# Watch nvidia-smi every 30 seconds on the compute node
srun --jobid <JOBID> --pty watch -n 30 nvidia-smi

# Side-by-side CPU (htop) + GPU (nvidia-smi) in tmux
srun --jobid <JOBID> --pty tmux new-session -d 'htop -u $USER' \; \
     split-window -h 'watch nvidia-smi' \; attach
```

**Target:** GPU utilisation close to 100%, GPU memory close to fully allocated.

### 3.3 Check completed job efficiency
```bash
seff <JOBID>
```
Example output:
```
CPU Efficiency: 99.72% of 02:49:26 core-walltime   ← want this high
Memory Utilized: 213.85 MB out of 125.00 GB         ← reduce memory request if low
GPU Utilization: 94%                                 ← want >80%
```

### 3.4 Check group and individual usage (RGU-years)
- Log in to https://ccdb.alliancecan.ca/
- Go to **My Projects → View Group Usage**
- Check that usage is on track — lab has 10 RGU-years, don't waste it on idle jobs

---

## 4. Recipe: Starting a New Model Training

Following the lab's standard recipe:

### Step 1 — Small trial run (5–10 samples)
```bash
# Request interactive H100 40G session
salloc --time=2:0:0 --mem=140G --ntasks=6 \
       --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1 \
       --account=axc-572-ac

# In another terminal: monitor GPU
srun --jobid <JOBID> --pty watch -n 10 nvidia-smi
```

Check:
- [ ] GPU memory close to fully used
- [ ] GPU utilisation close to 100%
- [ ] Loss curves decreasing in wandb
- [ ] Best + latest checkpoints saved each epoch
- [ ] Early stopping (patience) implemented

### Step 2 — Verify metrics on training set
Run inference on the 5–10 training samples. Expect ~100% DSC on training data to confirm the pipeline is correct.

### Step 3 — Full dataset job
```bash
sbatch train_full.sh  # one-day job with full H100 if needed
```

---

## 5. Transferring Data

### 5.1 scp (small files)
```bash
scp local_file.tar.gz brianx7@fir.alliancecan.ca:/scratch/$USER/data/
```

### 5.2 Globus (large datasets — recommended)
1. Go to https://app.globus.org/
2. Search collection: `computecanada#fir` or `alliancecan#fir`
3. Drag and drop between local and cluster

### 5.3 wget (downloading from web on cluster)
```bash
wget -O dataset.zip "https://download_link_here"
```

---

## 6. VS Code on Compute Node (recommended workflow)

1. Submit a job with `sleep infinity` to keep node alive
2. Get the node name from `squeue -u $USER`
3. In VS Code: **Remote SSH** → connect to `fc12345` (node name)
4. Work as if it's your own machine
5. For long training: use `tmux` so job survives disconnection

```bash
# Batch script for interactive VS Code session
#!/bin/bash
#SBATCH --account=axc-572-ac
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --mem=140G
#SBATCH --ntasks=6
#SBATCH --time=0-08:00
sleep infinity
```

---

## 7. Key Rules (do not break these)

| Rule | Consequence if broken |
|------|-----------------------|
| Never run jobs >1 min on login node | Account warning / suspension |
| Never run VS Code on login node | Same |
| Always monitor GPU utilisation | Entire lab's priority lowered |
| Use `seff` after every job | Required to optimise future requests |
| Request 20/40G H100 not 80G | Wastes RGU budget unnecessarily |
| Submit only 1-day jobs if requesting ≥ H100-80G | Lab policy |

---

## 8. Useful Commands Cheatsheet

```bash
squeue -u $USER                          # check my jobs
sq -u $USER                              # shorter version
seff <JOBID>                             # job efficiency summary
sbatch script.sh                         # submit job
scancel <JOBID>                          # cancel job
srun --jobid <JOBID> --pty bash          # attach to running job
watch -n 10 nvidia-smi                   # live GPU monitor
module avail python                      # list available Python versions
module load python/3.10                  # load Python
source ~/envs/voxtell/bin/activate       # activate venv
```

---

## References
- Running jobs: https://docs.alliancecan.ca/wiki/Running_jobs
- Python virtualenv: https://docs.alliancecan.ca/wiki/Python
- Fir cluster wiki: https://docs.alliancecan.ca/wiki/Fir
- Globus transfer: https://docs.alliancecan.ca/wiki/Globus
- Cluster status: https://status.alliancecan.ca/
- Technical support: support@tech.alliancecan.ca
