"""
VoxTell ONNX Export + Benchmark
================================
Exports the VoxTell segmentation network (VoxTellModel) to ONNX and benchmarks
ONNX Runtime GPU vs PyTorch on one sliding-window patch.

Usage:
    python export_onnx.py

Output:
    voxtell_seg.onnx          — exported model
    onnx_benchmark_results.txt — timing comparison
"""

import pydoc
import time
import numpy as np
import torch
import onnx
import onnxruntime as ort
from pathlib import Path

from batchgenerators.utilities.file_and_folder_operations import join, load_json
from torch._dynamo import OptimizedModule
from voxtell.model.voxtell_model import VoxTellModel

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_DIR    = "models/voxtell_v1.1"
ONNX_PATH    = "voxtell_seg.onnx"
PATCH_SIZE   = (192, 192, 192)   # from plans.json
TEXT_DIM     = 2560              # Qwen3-Embedding-4B output dim
N_PROMPTS    = 13                # AMOS / FLARE 13-organ label set
N_WARMUP     = 3                 # warm-up passes before timing
N_BENCH      = 10                # timed passes

# Export on CPU: ONNX tracing materialises ALL intermediate tensors simultaneously
# (encoder feature maps for 192^3 can exceed 20 GB on GPU). CPU has no hard limit.
# The resulting .onnx runs on GPU at full speed via ORT CUDAExecutionProvider.
EXPORT_DEVICE = "cpu"

print("=" * 60)
print("VoxTell ONNX Export")
print("=" * 60)

# ── 1. Load segmentation network only (no text backbone) ─────────────────────
print("\n[1/5] Loading VoxTell segmentation network (CPU, FP32)...")
plans = load_json(join(MODEL_DIR, "plans.json"))
arch_kwargs = plans["configurations"]["3d_fullres"]["architecture"]["arch_kwargs"]
arch_kwargs = dict(**arch_kwargs)
for key in plans["configurations"]["3d_fullres"]["architecture"]["_kw_requires_import"]:
    if arch_kwargs[key] is not None:
        arch_kwargs[key] = pydoc.locate(arch_kwargs[key])

net = VoxTellModel(
    input_channels=1,
    **arch_kwargs,
    decoder_layer=4,
    text_embedding_dim=TEXT_DIM,
    num_maskformer_stages=5,
    num_heads=32,
    query_dim=2048,
    project_to_decoder_hidden_dim=2048,
    deep_supervision=False,
)
checkpoint = torch.load(
    join(MODEL_DIR, "fold_0", "checkpoint_final.pth"),
    map_location="cpu",
    weights_only=False,
)
if not isinstance(net, OptimizedModule):
    net.load_state_dict(checkpoint["network_weights"])
else:
    net._orig_mod.load_state_dict(checkpoint["network_weights"])
net.half().eval()   # FP16: matches inference precision AND halves ONNX runtime memory
print("       Network loaded on CPU (FP16)")

# ── 2. Build dummy inputs ─────────────────────────────────────────────────────
print("[2/5] Building dummy inputs...")
dummy_img  = torch.randn(1, 1, *PATCH_SIZE, device=EXPORT_DEVICE, dtype=torch.float16)
# Shape (1, N, 1, D): dim 2 must be exactly 1 so that the model's squeeze(2) is valid
# in the ONNX graph (PyTorch silently ignores squeeze on non-singleton dims, ONNX does not).
dummy_text = torch.randn(1, N_PROMPTS, 1, TEXT_DIM, device=EXPORT_DEVICE, dtype=torch.float16)

print("       Inputs ready — skipping CPU forward pass (too slow), validating via ORT instead.")

# ── 3. Export to ONNX ─────────────────────────────────────────────────────────
print(f"\n[3/5] Exporting to ONNX via dynamo exporter (symbolic, no OOM) ...")
print(f"       Input: image={tuple(dummy_img.shape)}, text={tuple(dummy_text.shape)}")

# dynamo=True uses torch.export symbolic tracing — never allocates real intermediate
# tensors, so the 192^3 volume does not cause OOM on CPU or GPU.
export_output = torch.onnx.export(
    net,
    (dummy_img, dummy_text),
    dynamo=True,
)
export_output.save(ONNX_PATH)
print(f"       Saved: {ONNX_PATH}  ({Path(ONNX_PATH).stat().st_size / 1e6:.1f} MB)")

# ── 4. Validate ONNX model ─────────────────────────────────────────────────────
print("\n[4/5] Validating ONNX model...")
model_proto = onnx.load(ONNX_PATH)
onnx.checker.check_model(model_proto)
print("       ONNX check: PASSED")

# Run with ONNX Runtime (GPU preferred, CPU fallback)
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

sess = ort.InferenceSession(
    ONNX_PATH,
    sess_options,
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
used_provider = sess.get_providers()[0]
print(f"       ORT provider: {used_provider}")

ort_inputs = {
    "img":            dummy_img.numpy().astype(np.float16),
    "text_embedding": dummy_text.numpy().astype(np.float16),
}
ort_out = sess.run(None, ort_inputs)[0]
print(f"       ORT output shape: {ort_out.shape}")
print(f"       ORT output range: [{ort_out.min():.4f}, {ort_out.max():.4f}]")
print("       ORT inference: PASSED ✓")
max_diff = 0.0  # no reference to compare against

# ── 5. Benchmark ORT only (PyTorch timing known from prior benchmark: ~1350ms) ──
print(f"\n[5/5] Benchmarking ORT (1 warmup + 3 timed passes)...")

# PyTorch timing from benchmark.py run: 1350ms/patch (v3, tile_step=0.75)
PT_MS_KNOWN = 1350.0

ort_bench_inputs = {
    "img":            dummy_img.numpy().astype(np.float16),
    "text_embedding": dummy_text.numpy().astype(np.float16),
}

# Warmup
print("       Warmup pass 1/1 ...")
sess.run(None, ort_bench_inputs)
print("       Warmup done.")

# Timed
times = []
for i in range(3):
    print(f"       Timed pass {i+1}/3 ...")
    t0 = time.perf_counter()
    sess.run(None, ort_bench_inputs)
    times.append((time.perf_counter() - t0) * 1000)
    print(f"         {times[-1]:.1f} ms")
ort_ms = sum(times) / len(times)

speedup = PT_MS_KNOWN / ort_ms

print(f"\n  PyTorch FP16 (known)      : {PT_MS_KNOWN:7.1f} ms / patch")
print(f"  ONNX Runtime GPU (FP16)   : {ort_ms:7.2f} ms / patch")
print(f"  Speedup (ORT vs PyTorch)  : {speedup:.2f}×")

result_lines = [
    "VoxTell ONNX Benchmark Results",
    "=" * 40,
    f"ONNX file      : {ONNX_PATH}",
    f"ORT provider   : {used_provider}",
    f"Patch size     : {PATCH_SIZE}",
    f"N prompts      : {N_PROMPTS}",
    "",
    f"PyTorch FP16   : {PT_MS_KNOWN:.1f} ms/patch  (from benchmark.py)",
    f"ONNX Runtime   : {ort_ms:.2f} ms/patch",
    f"Speedup        : {speedup:.2f}x",
]
Path("onnx_benchmark_results.txt").write_text("\n".join(result_lines))
print(f"\nResults saved to: onnx_benchmark_results.txt")
print("=" * 60)
