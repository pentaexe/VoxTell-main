---
name: voxtell-inference
description: Run VoxTell benchmarking, training, or evaluation on the Fir cluster. Covers writing and validating SLURM jobs, the measurement rules that make a benchmark trustworthy, and the failure modes that have already cost jobs on this project.
---

You are running VoxTell experiments on the Fir HPC cluster. Follow the procedure
below rather than improvising — most of it exists because a specific mistake was
made once and should not be made again.

---

# PROCEDURE — writing and submitting a job

Work through these in order. Do not skip to the end.

### 1. Establish what is being measured
Ask, or determine from context:
- Which model — VoxTell or nnInteractive? They use **different venvs**.
- Is this a comparison? If so, what is held constant between the two arms?
- Which image? VoxTell timings on the MNI brain are near-useless (see Measurement
  rules, item 5).

### 2. Write the `.py`, then the `.sh`
A `.py` alone cannot run. Always produce both. Use the template below verbatim,
changing only job name, output path, `--mem`, and the python target.

### 3. Run the preflight checklist (below) against the `.sh` you just wrote
This is not optional. Every item on it corresponds to a job that already failed.

### 4. Commit and push locally. Never write files on the cluster.
```bash
git add <script>.py <script>.sh && git commit -m "..." && git push
```
Git flows one direction. The Fir login node has no push credentials and no git
identity, so a commit made there is stamped with the wrong author and then stalls
on a credential prompt. If one lands there anyway: `git status`, then
`git reset --hard HEAD~1`, redo it locally, push, and pull on the cluster.

### 5. Hand over the cluster block, with no Windows paths in it
A failed `cd` does not stop the commands after it, so a Windows path in a cluster
block silently runs everything in the wrong directory.
```bash
cd /scratch/brianx7/VoxTell-main && git pull
sbatch <script>.sh
squeue -u brianx7
```

### 6. When it finishes, report `seff <jobid>` alongside the result
Not as an afterthought — it is part of the result.

---

# PREFLIGHT CHECKLIST — apply to every `.sh` before saying "submit"

State explicitly which of these you checked.

1. **Right venv for the model.** VoxTell → `/home/brianx7/envs/voxtell`.
   nnInteractive → `/scratch/brianx7/envs/nninteractive`. Wrong one does not fail
   at startup; it dies a minute in on `ModuleNotFoundError`.
2. **`HF_HOME` exported.** Compute nodes have no internet. Without it
   `from_pretrained` fails after the job has already queued and started.
3. **`--account=rrg-jma`.** Not `def-jma-ab`, which is the no-RAC default and runs
   at low priority without drawing on the lab allocation.
4. **`--cpus-per-task=2`, `--ntasks=1`.** These jobs are GPU-bound; measured CPU
   efficiency is around 3%. Larger requests are visible in `seff` and lower the
   group's queue priority.
5. **MIG slice, not a full H100.** `3g.40gb` costs 6.1 RGU against 12.2 for the
   whole card.
6. **`python -u`.** Without it the log stays empty until the job ends, so a job
   that hangs looks identical to one that is working.
7. **`--time` is realistic but not padded.** Short walltime queues sooner; a
   timeout costs a whole cycle. 1:00:00 suits current jobs.

---

# MEASUREMENT RULES — these are the point of this skill

The headline speedup on this project fell **26× → 17.6× → 7.1× → 2.7×**. Not one
of those corrections changed the code being measured. Every one was a measurement
artifact. Any new benchmark must satisfy all of the following or it will produce
another wrong number.

1. **Hold precision constant.** Both arms INT4, or both FP16. VoxTell's predictor
   uses INT4 (NF4) by default, so a hand-built FP16 baseline silently makes
   quantization look like an algorithmic gain.

2. **Verify INT4 actually loaded.** `_load_text_backbone` (predictor.py) catches
   *any* exception and falls back to FP16 while the log still says INT4. Assert
   `predictor._backbone_quantized` is True. A missing `accelerate` is the usual
   cause.

3. **Warm everything before timing** — GPU, text backbone, *and* the
   sliding-window path. Whichever arm runs first otherwise absorbs CUDA context
   init and cuDNN autotuning. This alone inflated one result from 1.0× to 7.1×.
   Warm at the real `patch_size`; a small dummy tensor downsamples to 1×1×1 and
   `InstanceNorm3d` raises.

4. **Prove the embedding cache is cold.** Delete the disk entry and clear
   `_embed_cache`, then assert both are empty before timing and that the disk
   entry was written after. Otherwise a cache hit gets reported as a cold encode.

5. **Use a CT volume, not the MNI brain.** At 189×233×197 against a 192³ patch the
   brain yields 4 patches at *any* `tile_step`, and its arrays are too small for
   Numba to beat numpy. Neither optimization can act. Use
   `/scratch/brianx7/cvpr_val/3D_val_npz/CT_*.npz`; those carry a `text_prompts`
   key, so read the prompt from the file rather than guessing the anatomy.

6. **Repeat to n≥4 and quote the range.** Two runs of the same script have
   disagreed by 33%. Serialize repeats with `--dependency=afterany:` so they do
   not share thermal state. Report mean and range, never a single run.

7. **Cross-check on a second GPU when a number looks surprising.** Running the
   same script on the RTX 4070 SUPER and the H100 is what exposed the ordering
   artifact — identical phases behaved differently at matched precision.

---

# ENVIRONMENTS

**Cluster, VoxTell** — a venv, not conda, so there is no `conda activate`:
```bash
source /home/brianx7/envs/voxtell/bin/activate
```
Has `transformers`, `positional_encodings`, `bitsandbytes`, `accelerate` 1.13.0.

**Cluster, nnInteractive** — `/scratch/brianx7/envs/nninteractive/bin/activate`.
Lacks `transformers` and `positional_encodings`; pointing a VoxTell job here cost
two jobs on 2026-08-26.

**Local Windows** — `C:\Users\brian\miniconda3\envs\voxtell\python.exe`.
CUDA torch 2.8.0+cu126 on an RTX 4070 SUPER, plus bitsandbytes and accelerate.
Base miniconda is CPU-only torch and will not run benchmarks. Useful for a quick
cross-check without queueing.

---

# SLURM TEMPLATE

```bash
#!/bin/bash
#SBATCH --job-name=vox_<name>
#SBATCH --output=/scratch/brianx7/logs/vox_<name>_%j.out
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --mem=16G                 # 32G for CT volumes
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=1:00:00
#SBATCH --account=rrg-jma

source /home/brianx7/envs/voxtell/bin/activate

export TORCH_HOME=/scratch/brianx7/torch_home
export XDG_CACHE_HOME=/scratch/brianx7/cache
export HF_HOME=/scratch/brianx7/hf_cache

cd /scratch/brianx7/VoxTell-main
python -u <script>.py
```

---

# CURRENT RESULTS — cite these, not the older figures in the repo

Several files in this repo still contain superseded numbers. These are the ones
that survive scrutiny:

| Result | Value | Evidence |
|--------|-------|----------|
| VoxTell v0→v3, abdominal CT, H100 MIG | **2.7×** (2.6–2.8×) | n=4: jobs 56964411, 56966901–903 |
| VoxTell DSC | 0.8090 → 0.8093 (+0.0003) | accuracy_results.csv, 65 objects / 5 cases |
| INT4 vs FP16 output agreement | DSC 0.9716, INT4 −5.5% voxels | job 56964412, n=1 |
| nnInteractive torch.compile | **1.33×** (1.28–1.39×) | n=4: 56908464, 56923894–896 |
| nnInteractive DSC | 0.7914 → 0.7916 | job 56908464, 294 objects |
| Triton cold start / break-even | 23.6s → ~331 objects (~22 cases) | job 56914757 |

**Do not cite** 26×, 17.6×, 7.1×, 46.7×, `3.10s → 2.38s`, `343 → 125 patches`, or
`1.4× Numba preprocessing`. All are superseded or were measurement artifacts.
VoxTell on the H100 *brain* volume measures 1.0×, which is a property of the
volume, not the optimizations.

---

# SUBMISSION STANDARDS

Build these in from the start rather than retrofitting:
- Report `seff` for every job; request only the cores actually used
- Both sides of a comparison use the same checkpoint, fold and precision
- No CPU baselines; a GPU speedup is measured against a GPU baseline
- Any speed optimization needs a DSC number showing accuracy held
- Explain regressions rather than omitting them
- Always `use_fold='all'` for nnInteractive; never fold=0 (which gives ~0.33 DSC)
- Quote n and range, never a bare point estimate
