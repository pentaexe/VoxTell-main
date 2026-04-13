# VoxTell Inference Optimization — Experiment Log

Track every optimization attempt here. One entry per experiment, newest first.

---

## 2026-04-01 — Fair GPU-vs-GPU Benchmark

**Hypothesis:** The reported 26× speedup included fixing a silent CPU fallback bug; measure honest algorithmic improvement on equal hardware.  
**Speedup:** 1.3× total  (v0_gpu: 3.10s → v3: 2.38s)  
**Accuracy:** Not measured (latency experiment only)  
**Decision:** Documented — not a new optimization, a correction to reported framing  
**Notes:** v0_gpu uses FP16 on GPU (same as v3), tile_step=0.5, no cache, numpy preprocessing. v3 adds Numba preprocess + disk cache + tile_step=0.75. The cache contributes negligibly here because only 2 unique prompts are used (short benchmark). The 1.3× is from Numba (0.4×) + fewer patches (0.9×). Original 26× included the CPU→GPU fix which is the dominant gain.

---

## 2026-03-15 — torch.compile (cudagraphs backend)

**Hypothesis:** CUDA graph capture eliminates CPU-launch overhead for the per-patch forward pass.  
**Speedup:** 1.00× (1116.5ms → 1115.6ms per patch)  
**Accuracy:** Not measured (microbenchmark only)  
**Decision:** Rejected — no improvement  
**Notes:** Triton is unavailable on Windows, so only `cudagraphs` backend was available. Model is GPU compute-bound (192³ 3D convolutions), not CPU-launch-bound. Expect different result on H100 with Triton available — re-test with `backend="inductor"`.

---

## 2026-03-14 — ONNX Export + ORT CUDAExecutionProvider

**Hypothesis:** ONNX Runtime with CUDA EP can match or beat PyTorch for static-shape inference.  
**Speedup:** 0.07× (ORT: 19,102ms vs PyTorch: 1,350ms — 14× slower)  
**Accuracy:** ONNX forward pass passed numerical validation  
**Decision:** Rejected — ORT is 14× slower  
**Notes:** ORT CUDAExecutionProvider does not use cuDNN's optimized 3D convolution kernels (GroupNorm + Conv3d fusion). PyTorch FP16 + torch.autocast already leverages cuDNN. ONNX FP16 model is 688MB. Export required dynamo=True (symbolic tracing) to avoid materializing 192³ intermediate tensors. Try ORT TensorrtExecutionProvider on H100 — may close this gap.

---

## 2026-03-10 — Numba JIT Preprocessing (v3 → current)

**Hypothesis:** Numba-compiled crop-to-nonzero and z-score normalize is faster than NumPy.  
**Speedup:** 1.4× preprocessing (0.13s → 0.09s)  
**Accuracy:** DSC=0.8873, NSD=0.9052 (unchanged)  
**Decision:** Accepted — merged  
**Notes:** Numba parallelizes nonzero mask scan and vectorized normalize over the 3D volume. First call includes JIT compile overhead (~2s) but subsequent calls benefit fully.

---

## 2026-03-08 — Two-Level Embedding Cache (v2 → v3)

**Hypothesis:** Text embeddings for repeated prompts can be cached in memory and on disk.  
**Speedup:** 18.7× text embedding (0.51s → 0.02–0.03s on warm cache)  
**Accuracy:** DSC=0.8873, NSD=0.9052 (unchanged — same embeddings)  
**Decision:** Accepted — merged  
**Notes:** Two-level: LRU memory cache keyed by sorted prompt tuple, disk cache in `~/.voxtell_cache/`. Cache hit on disk loads a .pt file. Cache key is SHA256 of wrapped prompt strings. Embedding is deterministic (frozen backbone, no dropout in eval mode).

---

## 2026-03-05 — tile_step 0.5 → 0.75 (v1 → v2)

**Hypothesis:** Reducing sliding window overlap from 50% to 25% reduces patches without meaningful accuracy loss.  
**Speedup:** 3.6× sliding window (18.66s → 5.22s)  
**Accuracy:** DSC: 0.8867→0.8873 (+0.0006), NSD: 0.9040→0.9052 (+0.0012)  
**Decision:** Accepted — merged. Accuracy improved slightly (larger step = less over-smoothing at boundaries)  
**Notes:** Patches reduced from ~343 to ~125. The 3D Gaussian weighting already handles boundary stitching gracefully.

---

## 2026-03-01 — FP16 Text Backbone + GPU Placement (v0 → v1)

**Hypothesis:** Text encoder was silently falling back to CPU due to VRAM overflow in FP32.  
**Speedup:** 46.7× text embedding (126.02s → 2.70s)  
**Accuracy:** DSC within 0.001 of FP32 (FP16 is numerically sufficient for embedding extraction)  
**Decision:** Accepted — merged (was a bug fix, not a true optimization)  
**Notes:** Qwen3-Embedding-4B in FP32 requires ~16GB VRAM; RTX 4070 SUPER has 12GB. PyTorch silently falls back to CPU when VRAM is insufficient. Solution: `AutoModel.from_pretrained(..., dtype=torch.float16)` followed by `.to(device)`. Free text backbone VRAM before loading segmentation network.
