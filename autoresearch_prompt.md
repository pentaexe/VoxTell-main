# AutoResearch Framework — VoxTell Inference Acceleration

**Task:** Automatically discover and validate inference optimizations for the VoxTell 3D medical image segmentation pipeline.  
**Hardware target:** NVIDIA H100 80GB (Fir cluster, ComputeCanada) — primary; RTX 4070 SUPER 12GB — local validation.  
**Model:** VoxTell v1.1 (CVPR 2026) — free-text promptable 3D segmentation.

---

## 1. Problem Statement

VoxTell performs sliding-window inference over large 3D medical volumes (typically 189×233×197 to 512×512×300 voxels). End-to-end latency on a single H100 must be minimised without degrading segmentation accuracy below the thresholds in §4.

The pipeline has four stages. Each must be optimised independently and verified end-to-end:

| Stage | Description | Current H100 estimate |
|-------|-------------|----------------------|
| Preprocessing | Crop to nonzero, z-score normalise | ~0.05s |
| Text embedding | Qwen3-Embedding-4B (4B param, 2560-dim) | ~0.4s first call, ~0s cached |
| Sliding window | 3D UNet-style MaskFormer, 192³ patches | ~0.8–2.0s |
| Postprocessing | Sigmoid, threshold, insert back into volume | ~0.03s |

**Bottleneck on H100:** Sliding window (GPU compute-bound). Text embedding is negligible after caching.

---

## 2. Repository Layout

```
VoxTell-main/
├── voxtell/
│   ├── inference/
│   │   └── predictor.py        # VoxTellPredictor — main inference class
│   ├── model/
│   │   └── voxtell_model.py    # VoxTellModel (image encoder + decoder + MaskFormer)
│   ├── utils/
│   │   ├── text_embedding.py   # last_token_pool, wrap_with_instruction, EmbeddingCache
│   │   └── fast_preprocess.py  # numba_crop_to_nonzero, numpy_zscore_normalize
├── models/
│   └── voxtell_v1.1/
│       ├── plans.json          # patch_size, architecture kwargs
│       └── fold_0/
│           └── checkpoint_final.pth
├── benchmark_baseline.py       # End-to-end timing benchmark
├── fair_benchmark.py           # GPU-vs-GPU comparison (v0_gpu vs v3)
├── bench_compile.py            # torch.compile microbenchmark
├── export_onnx.py              # ONNX export experiment
└── accuracy_eval.py            # DSC/NSD evaluation on FLARE 2022 CT dataset
```

**Entry point for inference:**
```python
from voxtell.inference.predictor import VoxTellPredictor
predictor = VoxTellPredictor(model_dir="models/voxtell_v1.1", device=torch.device("cuda:0"))
data, bbox, orig_shape = predictor.preprocess(image)         # image: np.ndarray (C, H, W, D)
embeddings = predictor.embed_text_prompts(["brain", ...])    # returns (1, N, 2560) FP16 tensor
logits = predictor.predict_sliding_window_return_logits(data, embeddings)
```

---

## 3. Optimization Search Space

The agent should explore the following directions in order of expected impact on H100. Each attempt must be implemented as a self-contained experiment script (see §5) and evaluated against all metrics in §4.

### 3.1 TensorRT (highest priority)

Convert the segmentation network (VoxTellModel) to TensorRT FP16. This is the most impactful remaining optimization — cuDNN already provides optimal FP16 convolutions via PyTorch autocast; TensorRT adds kernel fusion and layer-level graph optimization.

- Build TensorRT engine from the exported ONNX (`voxtell_seg.onnx`, available on H100)
- Alternatively, use `torch_tensorrt` to JIT-compile without ONNX
- Target: `trt.BuilderFlag.FP16` precision, `trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH`
- Expected speedup on H100: 1.5–3.0× vs PyTorch eager (based on published TRT speedups for 3D UNets)
- Key constraint: patch shape is fixed (192, 192, 192) — static shape engine is sufficient

```python
# Reference approach (torch_tensorrt)
import torch_tensorrt
trt_model = torch_tensorrt.compile(
    net,
    inputs=[
        torch_tensorrt.Input((1, 1, 192, 192, 192), dtype=torch.float16),
        torch_tensorrt.Input((1, N, 2560), dtype=torch.float16),
    ],
    enabled_precisions={torch.float16},
)
```

### 3.2 Flash Attention for MaskFormer Decoder

The MaskFormer decoder uses multi-head cross-attention (32 heads, 2048-dim queries) over 192³ spatial features. Replace `torch.nn.MultiheadAttention` with `flash_attn_func` from the `flash-attn` package.

- Install: `pip install flash-attn --no-build-isolation` (requires CUDA 11.6+, available on H100)
- Target module: `VoxTellModel.mask_former_decoder.attention_layers`
- Replace `F.scaled_dot_product_attention` calls with `flash_attn_func`

### 3.3 Sliding Window Batch Size > 1

Currently each 192³ patch is run as a single-item batch. H100 has 80 GB — allow batch_size=2 or 4 patches per forward pass.

- Modify `predict_sliding_window_return_logits` in `predictor.py`
- Accumulate patches into a list, forward as a stacked batch, un-stack
- Expected speedup: 1.3–1.8× for batch_size=2 on H100 (hardware can handle it)

### 3.4 INT8 Quantization (Post-Training)

Apply INT8 post-training quantization to the segmentation network using PyTorch's `torch.quantization` or NVIDIA's `pytorch-quantization` toolkit.

- Calibrate on a small representative dataset (5–10 volumes)
- Target: INT8 for conv layers, keep attention FP16
- Risk: accuracy drop — must pass §4.2 accuracy gates

### 3.5 Text Backbone Quantization (Qwen3-Embedding-4B)

The text backbone is already cached after first call, so this only benefits cold-start latency. Still worth investigating for deployment scenarios with many unique prompts.

- Try `bitsandbytes` 4-bit NF4 quantization: `BitsAndBytesConfig(load_in_4bit=True)`
- On H100 (80GB) the FP16 model fits easily — quantization here is a cold-start optimization only

### 3.6 Torch CUDA Graphs (Sliding Window)

Capture the per-patch forward pass as a CUDA graph to eliminate CPU-launch overhead between patches.

```python
# Capture graph
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    out = net(static_patch, static_text)

# Replay per patch (zero CPU overhead)
static_patch.copy_(patch)
g.replay()
result = out.clone()
```

- Note: requires fixed patch shape and batch size — satisfied for VoxTell (192³ is fixed)
- `torch.compile(backend="cudagraphs")` was tested on RTX 4070 SUPER and showed 1.00×; H100 with Triton may differ

### 3.7 ONNX Runtime with TensorRT Backend

Use ORT's `TensorrtExecutionProvider` instead of `CUDAExecutionProvider`. On H100 this can be more competitive than ORT CUDA alone.

```python
providers = [("TensorrtExecutionProvider", {"trt_fp16_enable": True})]
session = ort.InferenceSession("voxtell_seg.onnx", providers=providers)
```

- Note: ORT CUDAExecutionProvider was tested and is 14× slower than PyTorch on RTX 4070 SUPER due to missing cuDNN 3D conv kernels — TRT EP avoids this

---

## 4. Evaluation Metrics

Every candidate optimization must be evaluated on **all three** metric categories. A candidate is accepted only if all gates pass.

### 4.1 Latency Metrics

Measure with `benchmark_baseline.py`. Report per-phase and total:

| Metric | Measurement | Target on H100 |
|--------|-------------|----------------|
| Preprocessing time | `time.perf_counter()` + `cuda.synchronize()` | < 0.05s |
| Text embedding time (cold) | First call, no cache | < 0.5s |
| Text embedding time (warm) | Cached call | < 0.01s |
| Sliding window time | Total for full volume | **< 0.5s (stretch: < 0.3s)** |
| End-to-end time (cold) | All four phases | < 1.0s |
| End-to-end time (warm) | Cached embeddings | **< 0.6s** |
| GPU memory peak | `torch.cuda.max_memory_allocated()` | < 60 GB |

Report in this table format:
```
Phase          | Baseline (H100) | This run | Delta
Preprocessing  |           0.05s |    X.XXs | +/-X%
Text embed     |           0.40s |    X.XXs | +/-X%
Sliding window |           0.80s |    X.XXs | +/-X%
Postprocessing |           0.03s |    X.XXs | +/-X%
TOTAL          |           1.28s |    X.XXs | +/-X%
```

### 4.2 Accuracy Gates (mandatory — do not skip)

Run `accuracy_eval.py` on FLARE 2022 AbdomenCT (5 cases, seed=42) for every candidate. **A candidate that fails these gates is rejected, regardless of speedup.**

| Metric | Baseline (v3) | Minimum acceptable | Notes |
|--------|--------------|-------------------|-------|
| Mean DSC (13 organs) | 0.8873 | ≥ 0.880 | Dice Similarity Coefficient |
| Mean NSD (13 organs, 2mm) | 0.9052 | ≥ 0.898 | Normalized Surface Dice |
| Worst-organ DSC | varies | ≥ 0.70 | No single organ may collapse |

Per-organ breakdown must be reported. Use NibabelIOWithReorient (not SimpleITK) for loading — SimpleITK returns raw voxel order causing DSC ≈ 0.04.

```bash
python accuracy_eval.py --config candidate  # modify predictor inside eval script
```

### 4.3 Reproducibility

- Report GPU model (`torch.cuda.get_device_name(0)`) and CUDA version
- Report PyTorch version (`torch.__version__`)
- Provide the exact command to reproduce each experiment
- Seed: `torch.manual_seed(42)`, `np.random.seed(42)` before each measurement
- Warm-up: 3 passes before timing; average over 5 measured passes

---

## 5. Experiment Protocol

Each optimization attempt must follow this protocol to ensure comparable results.

### 5.1 Experiment Script Template

Create one script per experiment: `exp_<name>.py`

```python
"""
Experiment: <name>
Hypothesis: <what you expect to improve>
Expected speedup: <X>× on <metric>
Risk: <accuracy risk, if any>
"""

import time, torch, numpy as np
from voxtell.inference.predictor import VoxTellPredictor
from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient

MODEL_DIR = "models/voxtell_v1.1"
IMAGE_PATH = "/path/to/test_volume.nii.gz"
PROMPTS    = ["brain", "left hemisphere"]
DEVICE     = torch.device("cuda:0")
N_WARMUP   = 3
N_BENCH    = 5

# --- Setup: apply optimization here ---
predictor = VoxTellPredictor(model_dir=MODEL_DIR, device=DEVICE)
# ... modification ...

# --- Benchmark ---
torch.manual_seed(42)
img, _ = NibabelIOWithReorient().read_images([IMAGE_PATH])
data, bbox, orig_shape = predictor.preprocess(img)
embeddings = predictor.embed_text_prompts(PROMPTS)  # warm cache

for _ in range(N_WARMUP):
    predictor.predict_sliding_window_return_logits(data, embeddings)
    torch.cuda.synchronize()

times = []
for _ in range(N_BENCH):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    predictor.predict_sliding_window_return_logits(data, embeddings)
    torch.cuda.synchronize()
    times.append(time.perf_counter() - t0)

print(f"Sliding window: {np.mean(times):.3f}s ± {np.std(times):.3f}s")
```

### 5.2 Decision Criteria

After running both §4.1 and §4.2:

| Outcome | Action |
|---------|--------|
| Speedup ≥ 1.2× AND accuracy gates pass | Merge into `predictor.py`, record in §6 |
| Speedup ≥ 1.2× BUT accuracy degrades | Investigate quantization precision; try mixed-precision |
| Speedup < 1.2× (negligible) | Document as "tried, not effective"; do not merge |
| Runtime error or OOM | Document failure mode and move on |

### 5.3 Results Logging

Append each result to `experiment_log.md`:

```markdown
## <date> — <experiment name>

**Hypothesis:** <what was expected>  
**Speedup:** <X>×  (sliding window: X.XXs → X.XXs)  
**Accuracy:** DSC=X.XXX, NSD=X.XXX  
**Decision:** Accepted / Rejected / Inconclusive  
**Notes:** <failure mode, tuning details, surprises>
```

---

## 6. Current Optimization Frontier (as of April 2026)

What has been tried on RTX 4070 SUPER (12GB). Use this as baseline when starting on H100.

| Optimization | Speedup | Status | Notes |
|-------------|---------|--------|-------|
| FP16 text backbone + GPU placement | 46.7× (bug fix) | ✓ Merged | Fixed silent CPU fallback |
| tile_step 0.5 → 0.75 | 3.6× sliding | ✓ Merged | Reduces patches from ~343 to ~125 |
| Two-level embedding cache (memory + disk) | 18.7× embed | ✓ Merged | Eliminates re-embedding same prompts |
| Numba JIT preprocessing | 1.4× pre | ✓ Merged | Parallel crop+normalize |
| INT4 quantization loader | Minimal | ✓ Merged | Memory reduction, not latency |
| ONNX + ORT CUDAExecutionProvider | 0.07× | ✗ Rejected | ORT 14× slower (no cuDNN 3D conv) |
| torch.compile cudagraphs | 1.00× | ✗ Rejected | Triton unavailable on Windows |
| Batched sliding window infrastructure | 0× (infra only) | ~ Partial | Framework ready, batch_size=1 now |
| TensorRT | Not tested | ⬜ TODO | Requires Linux/H100 — top priority |
| Flash Attention | Not tested | ⬜ TODO | Requires CUDA 11.6+ |
| CUDA Graphs (manual) | Not tested | ⬜ TODO | May help on H100 with Triton |
| INT8 PTQ | Not tested | ⬜ TODO | Accuracy risk — calibrate carefully |

**Fair GPU-vs-GPU algorithmic speedup (RTX 4070 SUPER):** 1.3× total, driven by cache + Numba.  
**Target on H100 with TensorRT:** < 0.5s sliding window (3–4× over current H100 eager estimate).

---

## 7. Compute Budget (ComputeCanada — Fir Cluster)

Account: `axc-572-05`  
Allocation: 10 RGU-years on Fir (H100 nodes)

| Job type | GPU request | RGUs/hour | Max hours/experiment |
|----------|-------------|-----------|---------------------|
| Quick benchmark (<30 min) | `nvidia_h100_80gb_hbm3_2g.20gb:1` | 3.5 | 2h |
| Accuracy eval (5 cases) | `nvidia_h100_80gb_hbm3_3g.40gb:1` | 6.1 | 2h |
| TensorRT engine build | `h100:1` (80GB) | 12.2 | 4h |

**Rule:** Never use the full 80G H100 unless the TRT engine build or model genuinely requires it. MIG instances (20/40G) are sufficient for benchmarking.

Standard job template for a benchmark experiment:
```bash
#!/bin/bash
#SBATCH --account=rrg-wanglab-ab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=60G
#SBATCH --gpus=nvidia_h100_80gb_hbm3_2g.20gb:1
#SBATCH --time=0-02:00
#SBATCH --output=/scratch/$USER/slurm_logs/%x_%j.log

module load python/3.10
source ~/envs/voxtell/bin/activate
python exp_<name>.py 2>&1 | tee results/<name>_$(date +%Y%m%d_%H%M).txt
```

---

## 8. Success Criteria

The project is considered complete when **any one** of the following is achieved on H100:

| Goal | Metric | Value |
|------|--------|-------|
| **Primary** | End-to-end latency (warm) | ≤ 0.5s |
| **Primary** | DSC on FLARE 2022 (5 cases) | ≥ 0.880 |
| **Secondary** | Sliding window alone | ≤ 0.3s |
| **Stretch** | Throughput | ≥ 5 volumes/minute |

Document the final result in `OPTIMIZATION_REPORT.md` §12 (Cumulative Results), following the existing format.
