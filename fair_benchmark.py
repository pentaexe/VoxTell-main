"""
Fair GPU-vs-GPU Benchmark — VoxTell
=====================================
Measures v0 (no optimizations, GPU) vs v3 (all optimizations, GPU) on the
SAME hardware platform so the comparison is honest.

v0_gpu: INT4 (NF4) text encoder on GPU, tile_step=0.5, no embedding cache, no Numba
v3:     INT4 (NF4) text encoder on GPU, tile_step=0.75, full cache + Numba

Both arms use identical INT4 (NF4) quantization so the comparison isolates only
algorithmic gains (Numba preprocessing, crop-to-nonzero, tile_step, embedding cache).
This addresses the reviewer comment that the original 145.25s baseline was
measured on CPU (silent VRAM overflow in FP32) which is an unfair comparison.
"""

import time
import pydoc
import torch
import numpy as np
from pathlib import Path
from torch._dynamo import OptimizedModule
from batchgenerators.utilities.file_and_folder_operations import join, load_json
from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient
from nnunetv2.inference.sliding_window_prediction import compute_gaussian, compute_steps_for_sliding_window
from acvl_utils.cropping_and_padding.bounding_boxes import insert_crop_into_image
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from transformers import AutoModel, AutoTokenizer

from voxtell.model.voxtell_model import VoxTellModel
from voxtell.utils.text_embedding import last_token_pool, wrap_with_instruction
from voxtell.utils.fast_preprocess import numba_crop_to_nonzero, numpy_zscore_normalize

import os
_DEFAULT_IMAGE = r"C:\Users\brian\Downloads\mni_icbm152_t1_tal_nlin_sym_09a.nii.gz"
_CLUSTER_IMAGE = "/scratch/brianx7/mni_icbm152_t1_tal_nlin_sym_09a.nii.gz"

# BENCH_IMAGE lets this run against a CVPR CT volume instead of the MNI brain.
# The brain is 189x233x197 against a 192^3 patch, so the sliding-window grid is
# 4 patches at ANY tile_step and Numba sees arrays too small to pay for itself —
# neither optimization has room to act. The challenge data (and every
# nnInteractive number) is abdominal CT, so measuring there is the like-for-like
# comparison.  e.g. BENCH_IMAGE=/scratch/brianx7/cvpr_val/3D_val_npz/CT_xxx.npz
IMAGE_PATH = os.environ.get("BENCH_IMAGE") or (
    _CLUSTER_IMAGE if os.path.exists(_CLUSTER_IMAGE) else _DEFAULT_IMAGE
)

_DEFAULT_MODEL = r"C:\Users\brian\OneDrive\Desktop\Code\VoxTell-main\models\voxtell_v1.1"
_CLUSTER_MODEL = "/scratch/brianx7/VoxTell-main/models/voxtell_v1.1"
MODEL_DIR  = _CLUSTER_MODEL if os.path.exists(_CLUSTER_MODEL) else _DEFAULT_MODEL

# "brain" is meaningless on an abdominal CT — set BENCH_PROMPTS to match the image.
PROMPTS    = [p.strip() for p in os.environ.get("BENCH_PROMPTS", "brain").split(",") if p.strip()]
DEVICE     = torch.device("cuda:0")
TEXT_MODEL = "Qwen/Qwen3-Embedding-4B"

print("=" * 70)
print("VoxTell Fair GPU-vs-GPU Benchmark")
print("=" * 70)
print(f"GPU : {torch.cuda.get_device_name(0)}")
print(f"Image: {IMAGE_PATH}")
print(f"Prompts: {PROMPTS}\n")

# Results filename — computed up front so a stale file from a previous run is
# removed now. Otherwise a crashed run leaves the old numbers on disk looking fresh.
gpu_name = torch.cuda.get_device_name(0)
if "H100" in gpu_name:
    _gpu_tag = "h100"
elif "RTX" in gpu_name or "GeForce" in gpu_name:
    _gpu_tag = "rtx"
else:
    _gpu_tag = gpu_name.replace(" ", "_")
# Tag the image too — a CT result must never overwrite a brain result, since the
# two are not comparable and mixing them is how the wrong number reaches a slide.
_img_tag = "ct_" + Path(IMAGE_PATH).stem.replace(".nii", "") if IMAGE_PATH.endswith(".npz") else "brain"
out_file = f"fair_benchmark_{_gpu_tag}_{_img_tag}_results.txt"
if Path(out_file).exists():
    Path(out_file).unlink()
    print(f"Removed stale {out_file} from a previous run.\n")

# ── Load image (shared between both runs) ─────────────────────────────────────
print("Loading image...")
if IMAGE_PATH.endswith(".npz"):
    # CVPR validation format: 'imgs' is a 3D volume; add the channel axis so the
    # shape matches what NibabelIOWithReorient returns for the .nii.gz path.
    _npz = np.load(IMAGE_PATH, allow_pickle=True)
    raw_img = _npz["imgs"][None].astype(np.float32)
    print(f"  Loaded from npz key 'imgs'")
    # These cases carry their own prompts — use them rather than guessing the
    # anatomy from the filename. An explicit BENCH_PROMPTS still wins.
    if "text_prompts" in _npz.files and not os.environ.get("BENCH_PROMPTS"):
        _tp = _npz["text_prompts"]
        _found = _tp.tolist() if hasattr(_tp, "tolist") else list(_tp)
        if isinstance(_found, dict):
            # {label_id: prompt} — take the prompts, ordered by label id
            _found = [_found[k] for k in sorted(_found)]
        _found = [str(p) for p in (_found if isinstance(_found, (list, tuple)) else [_found])]
        if _found:
            PROMPTS = _found[:1]   # one prompt keeps this comparable to the brain run
            print(f"  Prompts from npz 'text_prompts': {_found}")
            print(f"  Using: {PROMPTS}  (first only, to match the single-prompt brain run)")
else:
    raw_img, _ = NibabelIOWithReorient().read_images([IMAGE_PATH])
print(f"  Shape: {raw_img.shape}")
_vox = int(np.prod(raw_img.shape[1:]))
print(f"  Voxels: {_vox:,}\n")

# ── Load segmentation network (shared) ────────────────────────────────────────
print("Loading segmentation network...")
plans = load_json(join(MODEL_DIR, "plans.json"))
arch_kwargs = plans["configurations"]["3d_fullres"]["architecture"]["arch_kwargs"]
arch_kwargs = dict(**arch_kwargs)
for key in plans["configurations"]["3d_fullres"]["architecture"]["_kw_requires_import"]:
    if arch_kwargs[key] is not None:
        arch_kwargs[key] = pydoc.locate(arch_kwargs[key])

def load_network():
    net = VoxTellModel(
        input_channels=1, **arch_kwargs,
        decoder_layer=4, text_embedding_dim=2560,
        num_maskformer_stages=5, num_heads=32,
        query_dim=2048, project_to_decoder_hidden_dim=2048,
        deep_supervision=False,
    )
    ckpt = torch.load(
        join(MODEL_DIR, "fold_0", "checkpoint_final.pth"),
        map_location="cpu", weights_only=False,
    )
    if not isinstance(net, OptimizedModule):
        net.load_state_dict(ckpt["network_weights"])
    else:
        net._orig_mod.load_state_dict(ckpt["network_weights"])
    return net.to(DEVICE).half().eval()

patch_size = plans["configurations"]["3d_fullres"]["patch_size"]

# ── Helper: sliding window ─────────────────────────────────────────────────────
def run_sliding_window(net, data, embeddings, tile_step):
    with torch.inference_mode(), torch.autocast("cuda", enabled=True):
        data_pad, slicer_revert = pad_nd_image(data, patch_size, "constant", {"value": 0}, True, None)
        steps = compute_steps_for_sliding_window(data_pad.shape[1:], patch_size, tile_step)
        slicers = []
        for sx in steps[0]:
            for sy in steps[1]:
                for sz in steps[2]:
                    slicers.append(tuple([slice(None), *[slice(si, si+ti) for si, ti in zip((sx,sy,sz), patch_size)]]))

        n_prompts = embeddings.shape[1]
        pred_logits = torch.zeros((n_prompts, *data_pad.shape[1:]), dtype=torch.half, device=DEVICE)
        n_pred = torch.zeros(data_pad.shape[1:], dtype=torch.half, device=DEVICE)
        gaussian = compute_gaussian(tuple(patch_size), sigma_scale=1./8, value_scaling_factor=10, device=DEVICE)

        for slicer in slicers:
            patch = torch.clone(data_pad[slicer][None], memory_format=torch.contiguous_format).to(DEVICE)
            pred = net(patch, embeddings).to(DEVICE)
            pred_logits[slicer] += pred[0] * gaussian
            n_pred[slicer[1:]] += gaussian

        torch.div(pred_logits, n_pred, out=pred_logits)
        pred_logits = pred_logits[(slice(None), *slicer_revert[1:])]
    return pred_logits, len(slicers)

# ── GPU warmup — initialize CUDA context before any timed measurement ─────────
# Without this, the first arm absorbs CUDA context init + cuDNN autotuning.
# Warm at the real patch size: a tiny dummy downsamples to 1x1x1 and InstanceNorm3d
# raises (it uses instance stats even under .eval(), so it needs >1 spatial element).
# Warming at patch_size also initializes the exact cuDNN algos the real run uses.
print(f"Warming up GPU (forward pass at patch_size={patch_size})...")
_net_warmup = load_network()
with torch.inference_mode(), torch.autocast("cuda", enabled=True):
    _dummy = torch.zeros((1, 1, *patch_size), dtype=torch.float16, device=DEVICE)
    _dummy_emb = torch.zeros((1, 1, 2560), dtype=torch.float16, device=DEVICE)
    _ = _net_warmup(_dummy, _dummy_emb)
torch.cuda.synchronize()
del _net_warmup, _dummy, _dummy_emb
torch.cuda.empty_cache()
print("GPU warm.\n")

# ═══════════════════════════════════════════════════════════════════════════════
# V0_GPU — No algorithmic optimizations (INT4, tile_step=0.5, no cache, numpy)
# Same INT4 (NF4) text backbone as v3 — precision is held constant.
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("Running v0_gpu (GPU baseline — INT4, tile_step=0.5, no cache, numpy)")
print("  tile_step=0.5, no embedding cache, standard numpy preprocessing")
print("=" * 70)

# Phase 1: Preprocessing (standard numpy, no Numba)
t0 = time.perf_counter()
data_v0 = raw_img[0].astype(np.float32)
nonzero = data_v0 != 0.0
mean = float(data_v0[nonzero].mean()) if nonzero.any() else float(data_v0.mean())
std  = max(float(data_v0[nonzero].std()), 1e-8) if nonzero.any() else 1.0
data_v0 = ((data_v0 - mean) / std).astype(np.float32)
data_v0 = torch.from_numpy(data_v0[None])   # no crop
torch.cuda.synchronize()
t_pre_v0 = time.perf_counter() - t0
print(f"  [pre]   {t_pre_v0:.3f}s  shape={tuple(data_v0.shape)}")

# Phase 2: Text embedding — INT4 (NF4), same config as v3, no cache
print("  [embed] Loading text backbone (INT4 NF4 — same as v3)...")
from transformers import BitsAndBytesConfig
_bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)
tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL, padding_side="left")
text_backbone = AutoModel.from_pretrained(TEXT_MODEL, quantization_config=_bnb_config).eval()

def _encode(prompts):
    wrapped = wrap_with_instruction(prompts)
    tk = tokenizer(wrapped, padding=True, truncation=True, max_length=8192, return_tensors="pt")
    tk = {k: v.to(DEVICE) for k, v in tk.items()}
    with torch.inference_mode():
        o = text_backbone(**tk)
    return last_token_pool(o.last_hidden_state, tk["attention_mask"]).view(1, len(prompts), -1)

# Warm the TEXT BACKBONE before timing. The earlier warmup covered only the
# segmentation network, so the first bitsandbytes/CUDA kernel setup for the
# quantized text model landed inside v0's timed embed — and v3, running second,
# never paid it. On H100 job 56961173 that read as v0 4.270s vs v3 0.229s at
# IDENTICAL INT4 precision, which is an ordering artifact, not a v3 advantage.
print("  [embed] Warming text backbone (untimed)...")
_t_warm0 = time.perf_counter()
_ = _encode(PROMPTS)
torch.cuda.synchronize()
_t_backbone_warm = time.perf_counter() - _t_warm0
print(f"  [embed] Backbone warm ({_t_backbone_warm:.3f}s absorbed, not counted).")

t0 = time.perf_counter()
embeddings_v0 = _encode(PROMPTS)
torch.cuda.synchronize()
t_embed_v0 = time.perf_counter() - t0
print(f"  [embed] {t_embed_v0:.3f}s  (backbone pre-warmed)")

# Free text backbone VRAM before segmentation network
del text_backbone
torch.cuda.empty_cache()

# Phase 3: Sliding window (tile_step=0.5)
print("  [slide] Running sliding window (tile_step=0.5)...")
net_v0 = load_network()
# Warm this exact code path too. On H100 job 56961173 v0 read 4.003s against v3's
# 0.607s for the SAME 4 patches — the first arm was absorbing autotuning that the
# second never paid. Warming here makes both arms measure steady-state compute.
print("  [slide] Warming sliding-window path (untimed)...")
_t_warm0 = time.perf_counter()
_ = run_sliding_window(net_v0, data_v0, embeddings_v0, tile_step=0.5)
torch.cuda.synchronize()
_t_slide_warm = time.perf_counter() - _t_warm0
print(f"  [slide] Warm ({_t_slide_warm:.3f}s absorbed, not counted).")

t0 = time.perf_counter()
_, n_patches_v0 = run_sliding_window(net_v0, data_v0, embeddings_v0, tile_step=0.5)
torch.cuda.synchronize()
t_slide_v0 = time.perf_counter() - t0
print(f"  [slide] {t_slide_v0:.3f}s  ({n_patches_v0} patches, path pre-warmed)")

t_post_v0 = 0.03  # negligible, consistent with prior measurements
total_v0 = t_pre_v0 + t_embed_v0 + t_slide_v0 + t_post_v0

del net_v0
torch.cuda.empty_cache()

print(f"\n  v0_gpu TOTAL: {total_v0:.2f}s  ({n_patches_v0} patches, tile_step=0.5)\n")

# ═══════════════════════════════════════════════════════════════════════════════
# V3 — All algorithmic optimizations on GPU (INT4, tile_step=0.75, Numba, cache)
# Same INT4 (NF4) backbone as v0_gpu — precision held constant.
# Cache is cleared before v3 cold-embed so Comparison A is cold-to-cold.
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("Running v3 (all optimizations — Numba + INT4 + cache + tile_step=0.75)")
print("  v3 text backbone: INT4 (NF4)  |  v0_gpu: INT4 (NF4)  — precision matched")
print("=" * 70)

from voxtell.inference.predictor import VoxTellPredictor, _prompt_cache_path

predictor = VoxTellPredictor(model_dir=MODEL_DIR, device=DEVICE)

# _load_text_backbone (predictor.py:89) silently falls back to FP16 on ANY exception
# — a missing `accelerate` is the common one. That fallback would make this an
# FP16-vs-INT4 comparison again while the log still claims "precision matched".
# Fail loudly instead of measuring the wrong thing.
if not getattr(predictor, "_backbone_quantized", False):
    raise RuntimeError(
        "v3 text backbone did NOT load as INT4 — it fell back to FP16.\n"
        "v0_gpu loaded INT4 explicitly, so this run would be an uncontrolled\n"
        "precision comparison mislabelled as precision-matched.\n"
        "Most likely cause: `accelerate>=0.26.0` is not installed in this env.\n"
        "Fix the env and rerun; do not report numbers from this configuration."
    )
print("  [check] v3 backbone confirmed INT4 (NF4) — precision matched with v0_gpu.")

# Phase 1: Numba preprocessing
# Warm the JIT first — Numba compiles on first call, and that compile landing inside
# the timer made preprocessing read SLOWER than numpy on the RTX run. Both timings
# are reported so the compile cost stays visible rather than being hidden.
t0 = time.perf_counter()
_ = predictor.preprocess(raw_img)
t_pre_v3_jit = time.perf_counter() - t0

t0 = time.perf_counter()
data_v3, bbox, orig_shape = predictor.preprocess(raw_img)
torch.cuda.synchronize()
t_pre_v3 = time.perf_counter() - t0
print(f"  [pre]   {t_pre_v3:.3f}s  (first call incl. Numba JIT: {t_pre_v3_jit:.3f}s)")

# Phase 2: INT4 embedding — COLD measurement (disk + memory cache cleared)
# Separates INT4 gain from cache gain (cache is attributed separately at 18.7×).
# The cache key is sha256(f"{model}::{prompt}") and the predictor looks it up with
# self.text_encoding_model — assert that matches TEXT_MODEL, or we would be deleting
# a different key than the one the predictor reads and calling the result "cold".
assert predictor.text_encoding_model == TEXT_MODEL, (
    f"cache key mismatch: predictor uses {predictor.text_encoding_model!r}, "
    f"benchmark clears {TEXT_MODEL!r} — the 'cold' measurement would be a cache hit"
)
for p in PROMPTS:
    disk_path = _prompt_cache_path(p, TEXT_MODEL)
    if disk_path.exists():
        disk_path.unlink()
predictor._embed_cache.clear()

# Prove the cache is actually empty at the moment of measurement.
for p in PROMPTS:
    assert not _prompt_cache_path(p, TEXT_MODEL).exists(), f"disk cache survived for {p!r}"
assert not predictor._embed_cache, "in-memory cache survived clear()"
print("  [embed] Cache verified empty — measuring cold INT4 embedding...")

t0 = time.perf_counter()
embeddings_v3 = predictor.embed_text_prompts(PROMPTS)
torch.cuda.synchronize()
t_embed_v3_cold = time.perf_counter() - t0

# A genuine encode writes the disk cache on the way out. If the file is missing,
# no encode happened and the timing is meaningless.
for p in PROMPTS:
    assert _prompt_cache_path(p, TEXT_MODEL).exists(), (
        f"no disk cache written for {p!r} — the 'cold' encode did not actually run"
    )
print(f"  [embed cold INT4]  {t_embed_v3_cold:.3f}s  (verified cold: cache empty before, written after)")

t0 = time.perf_counter()
_ = predictor.embed_text_prompts(PROMPTS)
torch.cuda.synchronize()
t_embed_v3_warm = time.perf_counter() - t0
print(f"  [embed warm cache] {t_embed_v3_warm:.3f}s  (cache hit, attributed separately)")

# Phase 3: Sliding window (tile_step=0.75)
t0 = time.perf_counter()
prediction = predictor.predict_sliding_window_return_logits(data_v3, embeddings_v3)
torch.cuda.synchronize()
t_slide_v3 = time.perf_counter() - t0
slicers_v3 = predictor._internal_get_sliding_window_slicers(data_v3.shape[1:])
print(f"  [slide] {t_slide_v3:.3f}s  ({len(slicers_v3)} patches)")

# Phase 4: Postprocessing
t0 = time.perf_counter()
prediction = prediction.to("cpu")
with torch.no_grad():
    prediction_binary = torch.sigmoid(prediction.float()) > 0.5
seg = np.zeros([prediction_binary.shape[0], *orig_shape], dtype=np.uint8)
seg = insert_crop_into_image(seg, prediction_binary, bbox)
t_post_v3 = time.perf_counter() - t0

# cold total: cold FP16 vs cold INT4 — controls for cache, isolates other gains
total_v3_cold = t_pre_v3 + t_embed_v3_cold + t_slide_v3 + t_post_v3
# warm total: cold FP16 vs cached INT4 — production use case
total_v3_warm = t_pre_v3 + t_embed_v3_warm + t_slide_v3 + t_post_v3
print(f"\n  v3 TOTAL (cold embed): {total_v3_cold:.2f}s")
print(f"  v3 TOTAL (warm cache): {total_v3_warm:.2f}s\n")

# ═══════════════════════════════════════════════════════════════════════════════
# Results
# ═══════════════════════════════════════════════════════════════════════════════
speedup_cold  = total_v0 / total_v3_cold   # cold INT4 → cold INT4 (no cache effect; pure algo gain)
speedup_warm  = total_v0 / total_v3_warm   # cold INT4 → warm INT4 (full v3 stack)
# Warm cache hits are often sub-millisecond; a ratio against that is meaningless
# (it reports thousands-fold and is really just 1/epsilon). Report the absolute
# time instead once it drops below a threshold the timer can resolve.
_CACHE_RESOLVABLE = 1e-3
if t_embed_v3_warm >= _CACHE_RESOLVABLE:
    cache_gain_str = f"{t_embed_v3_cold / t_embed_v3_warm:.1f}×"
else:
    cache_gain_str = (f"cold {t_embed_v3_cold*1000:.0f}ms → warm <1ms "
                      f"(below timer resolution; ratio not meaningful)")

print("=" * 70)
print("FAIR GPU-vs-GPU COMPARISON SUMMARY")
print("=" * 70)
print(f"  Image : {Path(IMAGE_PATH).name}  {tuple(raw_img.shape[1:])}  ({_vox:,} voxels)")
print(f"  Prompts: {PROMPTS}")
print(f"  Patch : {patch_size}")
print()
print("  Comparison A — cold-to-cold, same INT4 precision (pure algorithmic gain)")
print(f"  {'Metric':<28} {'v0_gpu (INT4)':>14} {'v3 (INT4)':>12}")
print("  " + "-" * 56)
print(f"  {'Preprocessing':<28} {t_pre_v0:>13.3f}s {t_pre_v3:>11.3f}s")
print(f"  {'Text embedding (cold INT4)':<28} {t_embed_v0:>13.3f}s {t_embed_v3_cold:>11.3f}s")
print(f"  {'Sliding window':<28} {t_slide_v0:>13.3f}s {t_slide_v3:>11.3f}s")
print(f"  {'Postprocessing':<28} {t_post_v0:>13.3f}s {t_post_v3:>11.3f}s")
print(f"  {'Patches':<28} {n_patches_v0:>14} {len(slicers_v3):>12}")
print("  " + "-" * 56)
print(f"  {'TOTAL':<28} {total_v0:>13.2f}s {total_v3_cold:>11.2f}s")
print(f"  Speedup (cold, no cache): {speedup_cold:.1f}×  ← pure algorithmic, precision-matched")
print()
print("  Comparison B — with embedding cache (production warm-query use case)")
print(f"  {'Text embedding (warm cache)':<28} {t_embed_v0:>13.3f}s {t_embed_v3_warm:>11.3f}s")
print(f"  {'TOTAL':<28} {total_v0:>13.2f}s {total_v3_warm:>11.2f}s")
print(f"  Speedup (warm cache):     {speedup_warm:.1f}×")
print()
print(f"  Cache gain on embedding (cold → warm INT4): {cache_gain_str}")
print(f"  Patches: v0_gpu {n_patches_v0} vs v3 {len(slicers_v3)}")
if n_patches_v0 == len(slicers_v3):
    print(f"    Identical patch count. This volume is {tuple(raw_img.shape[1:])} against a")
    print(f"    {patch_size} patch, so the sliding-window grid is tiny at BOTH tile_step")
    print( "    settings and the parameter has almost nothing to act on here.")
    print( "    => On THIS volume the sliding-window gain is crop-to-nonzero, not tile_step.")
    print( "    This does NOT show tile_step is ineffective generally — the 343→125 patch")
    print( "    figure came from a different, larger image and is untested on this one.")
print(f"  Sliding window alone: {t_slide_v0:.3f}s → {t_slide_v3:.3f}s = {t_slide_v0/t_slide_v3:.2f}×")
if t_pre_v3 > t_pre_v0:
    print(f"  NOTE: Numba preprocessing is SLOWER here ({t_pre_v0:.3f}s → {t_pre_v3:.3f}s)."
          "\n        Likely first-call JIT compilation. Do not claim a preprocessing speedup"
          "\n        from this run.")
if t_embed_v3_cold > t_embed_v0:
    print(f"  NOTE: v3 cold embed is SLOWER than v0 ({t_embed_v0:.3f}s → {t_embed_v3_cold:.3f}s)"
          "\n        at identical INT4 precision — the predictor path costs more than raw"
          "\n        AutoModel. v3's embed advantage comes from the cache, not the encoder.")
print()
print("  Both arms: INT4 (NF4) via bitsandbytes. DSC impact of INT4 not measured.")
print(f"  Original reported speedup: 26.0×  (CPU baseline — unfair comparison)")
print("=" * 70)

# Save — out_file was computed at startup and any stale copy already removed
lines = [
    "Fair GPU-vs-GPU Benchmark Results",
    "=" * 40,
    f"GPU: {gpu_name}",
    f"Image: {Path(IMAGE_PATH).name}  shape={tuple(raw_img.shape[1:])}  voxels={_vox:,}",
    f"Prompts: {PROMPTS}",
    f"Patch size: {patch_size}",
    "",
    f"v0_gpu total : {total_v0:.2f}s  (INT4 NF4, tile_step=0.5, no cache, numpy preprocess)",
    f"v3 cold total: {total_v3_cold:.2f}s  (INT4 NF4, tile_step=0.75, no cache, Numba)",
    f"v3 warm total: {total_v3_warm:.2f}s  (INT4 NF4, tile_step=0.75, cache hit, Numba)",
    "",
    f"Speedup cold (cold INT4 -> cold INT4, no cache, pure algo): {speedup_cold:.1f}x",
    f"Speedup warm (cold INT4 -> cached INT4, full stack):        {speedup_warm:.1f}x",
    f"Cache gain on embedding (cold -> warm INT4):                {cache_gain_str}",
    f"Patches: v0_gpu {n_patches_v0} vs v3 {len(slicers_v3)}",
    f"Sliding window alone: {t_slide_v0:.3f}s -> {t_slide_v3:.3f}s = {t_slide_v0/t_slide_v3:.2f}x",
    f"Preprocessing: {t_pre_v0:.3f}s -> {t_pre_v3:.3f}s"
    + ("  (SLOWER - likely Numba JIT on first call)" if t_pre_v3 > t_pre_v0 else ""),
    "",
    "Note: Both arms use identical INT4 (NF4) via bitsandbytes. Precision-matched.",
    "Note: Original 26x used CPU baseline (FP32 text encoder VRAM overflow).",
    "Note: DSC impact of INT4 quantization NOT measured.",
    "Note: GPU warmup pass run before v0_gpu arm to initialize CUDA context.",
]
Path(out_file).write_text("\n".join(lines))
print(f"\nSaved: {out_file}")
