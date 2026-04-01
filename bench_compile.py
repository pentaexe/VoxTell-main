"""
torch.compile benchmark — VoxTell segmentation network
=======================================================
Compares PyTorch eager (current v3) vs torch.compile on one sliding-window patch.

Usage:
    python bench_compile.py
"""

import time
import pydoc
import torch
from torch._dynamo import OptimizedModule
from batchgenerators.utilities.file_and_folder_operations import join, load_json
from voxtell.model.voxtell_model import VoxTellModel

MODEL_DIR  = "models/voxtell_v1.1"
PATCH_SIZE = (192, 192, 192)
TEXT_DIM   = 2560
N_PROMPTS  = 13
N_WARMUP   = 3
N_BENCH    = 10

print("=" * 60)
print("torch.compile Benchmark — VoxTell")
print("=" * 60)

# ── Load network ──────────────────────────────────────────────────────────────
print("\nLoading network...")
plans = load_json(join(MODEL_DIR, "plans.json"))
arch_kwargs = plans["configurations"]["3d_fullres"]["architecture"]["arch_kwargs"]
arch_kwargs = dict(**arch_kwargs)
for key in plans["configurations"]["3d_fullres"]["architecture"]["_kw_requires_import"]:
    if arch_kwargs[key] is not None:
        arch_kwargs[key] = pydoc.locate(arch_kwargs[key])

net = VoxTellModel(
    input_channels=1, **arch_kwargs,
    decoder_layer=4, text_embedding_dim=TEXT_DIM,
    num_maskformer_stages=5, num_heads=32,
    query_dim=2048, project_to_decoder_hidden_dim=2048,
    deep_supervision=False,
)
checkpoint = torch.load(
    join(MODEL_DIR, "fold_0", "checkpoint_final.pth"),
    map_location="cpu", weights_only=False,
)
if not isinstance(net, OptimizedModule):
    net.load_state_dict(checkpoint["network_weights"])
else:
    net._orig_mod.load_state_dict(checkpoint["network_weights"])

net = net.to("cuda").half().eval()

# ── Dummy inputs ──────────────────────────────────────────────────────────────
img  = torch.randn(1, 1, *PATCH_SIZE, device="cuda", dtype=torch.float16)
text = torch.randn(1, N_PROMPTS, TEXT_DIM, device="cuda", dtype=torch.float16)

# ── Benchmark helper ──────────────────────────────────────────────────────────
def benchmark(model, label):
    print(f"\n  [{label}] warming up ({N_WARMUP} passes)...")
    with torch.inference_mode(), torch.autocast("cuda", enabled=True):
        for i in range(N_WARMUP):
            model(img, text)
            torch.cuda.synchronize()
            print(f"    warmup {i+1}/{N_WARMUP} done")
    print(f"  [{label}] timing {N_BENCH} passes...")
    with torch.inference_mode(), torch.autocast("cuda", enabled=True):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(N_BENCH):
            model(img, text)
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / N_BENCH * 1000
    print(f"  [{label}] {ms:.1f} ms / patch")
    return ms

# ── Eager baseline ────────────────────────────────────────────────────────────
eager_ms = benchmark(net, "Eager (v3 baseline)")

# ── torch.compile ─────────────────────────────────────────────────────────────
print("\n  Compiling with torch.compile(backend='cudagraphs')...")
print("  (records CUDA graph on first pass, no Triton needed)")
net_compiled = torch.compile(net, backend="cudagraphs")
compiled_ms = benchmark(net_compiled, "torch.compile (cudagraphs)")

# ── Results ───────────────────────────────────────────────────────────────────
speedup = eager_ms / compiled_ms
print(f"\n{'='*60}")
print(f"  Eager baseline   : {eager_ms:.1f} ms / patch")
print(f"  torch.compile    : {compiled_ms:.1f} ms / patch")
print(f"  Speedup          : {speedup:.2f}×")
print(f"{'='*60}")
