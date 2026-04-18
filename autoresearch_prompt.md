# AutoResearch Framework — General Inference Acceleration for Medical Segmentation Models

**Goal:** Automatically discover, implement, and validate inference optimizations for any 3D medical image segmentation model.  
**Primary targets:** VoxTell v1.1 (text-promptable segmentation) · nnInteractive (interactive segmentation)  
**Hardware:** NVIDIA H100 80GB (Fir cluster, ComputeCanada, account: axc-572-ac)  
**Local validation:** RTX 4070 SUPER 12GB  

---

## 1. Problem Statement

3D medical segmentation models share a common inference bottleneck structure regardless of architecture:

```
Input Volume (CT/MRI)
    │
    ▼
[Preprocessing]      ← crop, normalize, resample
    │
    ▼
[Prompt Encoding]    ← text (VoxTell) / clicks/scribbles (nnInteractive) / none
    │
    ▼
[Sliding Window]     ← repeated forward passes over 3D patches  ← MAIN BOTTLENECK
    │
    ▼
[Postprocessing]     ← threshold, insert back, optional CRF
    │
    ▼
Output Mask
```

The sliding window stage dominates runtime in all models because 3D volumes are too large to process in a single forward pass. The goal of this framework is to reduce end-to-end latency while preserving segmentation accuracy above defined thresholds.

---

## 2. Supported Models

The framework is designed to work with any model that follows the sliding-window inference pattern. Two reference implementations are provided.

### 2.1 VoxTell v1.1

**Type:** Free-text promptable 3D segmentation (CVPR 2026)  
**Architecture:** 3D ResidualEncoder + Qwen3-Embedding-4B text encoder + MaskFormer decoder  
**Patch size:** 192×192×192  
**Input:** NIfTI volume (CT/MRI) + free-text anatomical prompts  
**Output:** Binary mask per prompt  
**Inference entry point:**
```python
from voxtell.inference.predictor import VoxTellPredictor
predictor = VoxTellPredictor(model_dir="models/voxtell_v1.1", device=device)
data, bbox, orig_shape = predictor.preprocess(image)
embeddings = predictor.embed_text_prompts(["liver", "spleen", ...])
logits = predictor.predict_sliding_window_return_logits(data, embeddings)
```
**Current H100 estimate (warm cache):** ~0.8s sliding window, ~1.0s total

### 2.2 nnInteractive

**Type:** Interactive segmentation — user provides clicks or scribbles to guide segmentation  
**Architecture:** nnU-Net backbone with interaction encoder  
**Patch size:** model-dependent (typically 128×128×128 or 192×192×192)  
**Input:** NIfTI volume + interaction map (click coordinates or scribble mask)  
**Output:** Binary segmentation mask  
**Inference entry point:** nnInteractive predictor API (same sliding-window pattern as nnU-Net)  
**Baseline latency:** To be profiled on H100 — run `python profile_nninteractive.py` (see §5)

### 2.3 Adding a New Model

To apply this framework to a new model:
1. Identify the sliding-window inference function
2. Profile each phase using the template in §5.1
3. Record baseline in `experiment_log.md`
4. Apply optimizations from §3 in priority order

---

## 3. Optimization Search Space

Optimizations are listed in priority order based on expected impact. Each applies to **any** sliding-window segmentation model.

### 3.1 TensorRT Engine (highest priority — all models)

Convert the segmentation network to TensorRT FP16. Provides kernel fusion and graph-level optimization beyond what PyTorch autocast offers.

```python
import torch_tensorrt
trt_model = torch_tensorrt.compile(
    net,
    inputs=[torch_tensorrt.Input(shape, dtype=torch.float16) for shape in input_shapes],
    enabled_precisions={torch.float16},
)
```

- **VoxTell inputs:** `[(1,1,192,192,192), (1,N,2560)]`
- **nnInteractive inputs:** profile model to determine input shapes
- Expected speedup: 1.5–3.0× over PyTorch eager on H100
- Requires Linux (available on ComputeCanada Fir)

### 3.2 Batched Sliding Window (all models)

Process multiple patches per forward pass instead of one at a time. H100 has 80GB — batch_size=2 or 4 is feasible for most 3D models.

```python
# Accumulate patches, forward as batch, un-stack
batch = torch.stack([patch1, patch2, patch3, patch4])   # (B, C, X, Y, Z)
preds = net(batch, ...)                                   # (B, ...)
```

- Expected speedup: 1.3–1.8× for batch_size=4
- Works for any model without architecture changes

### 3.3 Flash Attention (transformer-based models)

Replace standard multi-head attention with `flash_attn_func` for models that use attention in their decoder (VoxTell MaskFormer, nnInteractive interaction encoder).

```python
from flash_attn import flash_attn_func
out = flash_attn_func(q, k, v, causal=False)
```

- Install: `pip install flash-attn --no-build-isolation` (CUDA 11.6+, available on H100)
- Expected speedup: 1.2–2.0× for attention-heavy decoders

### 3.4 CUDA Graphs (all models)

Capture the per-patch forward pass as a CUDA graph to eliminate CPU kernel-launch overhead between patches.

```python
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    out = net(static_input)
# Per patch: copy input, replay graph, copy output
static_input.copy_(patch)
g.replay()
result = out.clone()
```

- Requires fixed patch shape — satisfied for all sliding-window models
- On H100 with Triton: `torch.compile(net, backend="inductor")` is equivalent

### 3.5 INT8 Post-Training Quantization (all models)

Quantize conv layers to INT8 using NVIDIA's `pytorch-quantization` toolkit. Calibrate on 5–10 representative volumes.

```python
from pytorch_quantization import quant_modules
quant_modules.initialize()
# load model, run calibration forward passes, export
```

- Risk: accuracy drop — must pass accuracy gates in §4.2
- Keep attention layers in FP16; quantize conv layers only

### 3.6 Prompt/Interaction Encoding Cache (prompt-based models)

For models that accept repeated prompts (VoxTell) or fixed interaction maps, cache the encoded representation to avoid re-encoding on repeated queries.

- VoxTell: already implemented (disk + memory cache) — 18.7× speedup on warm cache
- nnInteractive: interaction encoding is query-specific, caching not applicable
- Generalizable to any model with a separable prompt encoder

### 3.7 Preprocessing Parallelization (all models)

Use Numba JIT or CuPy to parallelize crop-to-nonzero and z-score normalization over the 3D volume.

```python
import numba
@numba.njit(parallel=True)
def crop_and_normalize(volume): ...
```

- Expected speedup: 1.3–1.5× for preprocessing phase
- VoxTell: already implemented — serves as reference for other models

---

## 4. Evaluation Metrics

Every experiment must report all three metric categories. A result is only accepted if **all gates pass**.

### 4.1 Latency (per model)

Measure with `time.perf_counter()` + `torch.cuda.synchronize()`. Report per-phase and total.

| Phase | VoxTell H100 target | nnInteractive H100 target |
|-------|--------------------|-----------------------------|
| Preprocessing | < 0.05s | < 0.05s |
| Prompt/interaction encoding (cold) | < 0.5s | < 0.1s |
| Prompt/interaction encoding (warm) | < 0.01s | N/A |
| Sliding window | **< 0.5s** | **< 1.0s** |
| Postprocessing | < 0.05s | < 0.05s |
| **End-to-end (warm)** | **< 0.6s** | **< 1.2s** |

### 4.2 Accuracy Gates (mandatory)

| Model | Dataset | Metric | Baseline | Minimum |
|-------|---------|--------|---------|---------|
| VoxTell | FLARE 2022 AbdomenCT, 5 cases, seed=42 | Mean DSC (13 organs) | 0.8873 | ≥ 0.880 |
| VoxTell | FLARE 2022 AbdomenCT, 5 cases, seed=42 | Mean NSD (2mm) | 0.9052 | ≥ 0.898 |
| nnInteractive | TBD — profile baseline first | DSC | TBD | ≤ 1% drop |

**Critical:** Use `NibabelIOWithReorient` (not SimpleITK) to load NIfTI files for VoxTell evaluation. SimpleITK returns raw voxel order causing DSC ≈ 0.04.

### 4.3 Reproducibility

- Report GPU model, CUDA version, PyTorch version
- Seed: `torch.manual_seed(42)`, `np.random.seed(42)`
- Warm-up: 3 passes; average over 5 measured passes
- Provide exact command to reproduce

---

## 5. Experiment Protocol

### 5.1 Profiling a New Model (baseline)

Before optimizing any model, profile it with this template to establish the baseline:

```python
"""Profile script — run once per model to establish baseline."""
import time, torch
from <model> import <Predictor>

predictor = <Predictor>(model_dir=MODEL_DIR, device=torch.device("cuda:0"))
image = load_test_volume(IMAGE_PATH)   # use NibabelIOWithReorient

phases = {}

t0 = time.perf_counter(); torch.cuda.synchronize()
data, bbox, orig_shape = predictor.preprocess(image)
torch.cuda.synchronize(); phases["preprocess"] = time.perf_counter() - t0

t0 = time.perf_counter(); torch.cuda.synchronize()
prompt = predictor.encode_prompt(...)   # text / clicks / none
torch.cuda.synchronize(); phases["prompt"] = time.perf_counter() - t0

t0 = time.perf_counter(); torch.cuda.synchronize()
logits = predictor.sliding_window_inference(data, prompt)
torch.cuda.synchronize(); phases["sliding_window"] = time.perf_counter() - t0

t0 = time.perf_counter()
mask = predictor.postprocess(logits, bbox, orig_shape)
phases["postprocess"] = time.perf_counter() - t0

for k, v in phases.items():
    print(f"{k:20s}: {v:.3f}s")
print(f"{'TOTAL':20s}: {sum(phases.values()):.3f}s")
```

### 5.2 Experiment Script Template

One script per optimization attempt: `exp_<model>_<optimization>.py`

```python
"""
Model: <VoxTell | nnInteractive | ...>
Experiment: <optimization name>
Hypothesis: <expected improvement>
Risk: <accuracy risk if any>
"""
# 1. Load model with optimization applied
# 2. Warm up (3 passes)
# 3. Benchmark (5 passes, report mean ± std)
# 4. Run accuracy_eval.py and record DSC/NSD
# 5. Log result in experiment_log.md
```

### 5.3 Decision Criteria

| Outcome | Action |
|---------|--------|
| Speedup ≥ 1.2× AND accuracy gates pass | Merge, record in experiment_log.md |
| Speedup ≥ 1.2× BUT accuracy degrades | Investigate mixed-precision; try calibration |
| Speedup < 1.2× | Document as "not effective"; do not merge |
| OOM or crash | Document failure mode and move on |

---

## 6. Results Summary

### 6.1 VoxTell — Optimization History (RTX 4070 SUPER)

| Optimization | Pre | Embed | Slide | Post | Total | Speedup | Status |
|-------------|-----|-------|-------|------|-------|---------|--------|
| v0 — CPU baseline (bug) | 0.38s | 126.02s | 18.66s | 0.19s | 145.25s | 1.0× | — |
| v0_gpu — GPU baseline (fair) | 0.13s | 0.51s | 2.44s | 0.03s | 3.10s | 1.0× | — |
| v1 — FP16 + tile_step=0.75 | 0.10s | 2.70s | 5.22s | 0.04s | 8.06s | 18.0× | ✓ |
| v2 — cache + Numba | 0.09s | 0.02s | 2.22s | 0.03s | 2.36s | 61.5× | ✓ |
| ONNX + ORT CUDA EP | — | — | 19.1s | — | — | 0.07× | ✗ Rejected |
| torch.compile cudagraphs | — | — | 1116ms/patch | — | — | 1.00× | ✗ Rejected |
| TensorRT | — | — | TBD | — | — | TBD | ⬜ H100 |
| Flash Attention | — | — | TBD | — | — | TBD | ⬜ H100 |

### 6.2 nnInteractive — Optimization History

| Optimization | Total | Speedup | Status |
|-------------|-------|---------|--------|
| Baseline (H100) | TBD | 1.0× | ⬜ Profile first |

---

## 7. ComputeCanada Job Templates

Account: `axc-572-ac` · Username: `brianx7` · Cluster: Fir (H100)

**Benchmark job (< 30 min):**
```bash
#!/bin/bash
#SBATCH --account=axc-572-ac
#SBATCH --gpus=nvidia_h100_80gb_hbm3_2g.20gb:1
#SBATCH --mem=40G
#SBATCH --ntasks=4
#SBATCH --time=0-00:30
#SBATCH --output=/scratch/$USER/slurm_logs/%x_%j.log

module load python/3.10
source ~/envs/voxtell/bin/activate
python exp_<model>_<optimization>.py 2>&1 | tee results/$(date +%Y%m%d_%H%M)_$SLURM_JOB_NAME.txt
```

**Accuracy eval job (< 2 hours):**
```bash
#!/bin/bash
#SBATCH --account=axc-572-ac
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --mem=80G
#SBATCH --ntasks=6
#SBATCH --time=0-02:00
#SBATCH --output=/scratch/$USER/slurm_logs/%x_%j.log

module load python/3.10
source ~/envs/voxtell/bin/activate
python accuracy_eval.py --config <candidate>
```

---

## 8. Success Criteria

The framework succeeds when **all** of the following hold across **both** models:

| Goal | VoxTell | nnInteractive |
|------|---------|--------------|
| End-to-end latency (warm) | ≤ 0.6s on H100 | ≤ 1.2s on H100 |
| Accuracy drop vs baseline | DSC ≤ 1% | DSC ≤ 1% |
| Optimization generalizes | Same technique applies to both | ✓ |

Document final results for each model in `experiment_log.md` and summarize in a compact report (≤ 10 pages).
