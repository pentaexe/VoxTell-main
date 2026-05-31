"""
Quick RTX 4070 SUPER baseline benchmark — 1 prompt, no torch.compile.
Matches H100 benchmark conditions exactly.
"""
import time
import torch
import numpy as np
from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient
from acvl_utils.cropping_and_padding.bounding_boxes import insert_crop_into_image
from voxtell.inference.predictor import VoxTellPredictor
import voxtell.inference.predictor as _pred_module

IMAGE_PATH = r"C:\Users\brian\Downloads\mni_icbm152_t1_tal_nlin_sym_09a.nii.gz"
MODEL_DIR  = r"C:\Users\brian\OneDrive\Desktop\Code\VoxTell-main\models\voxtell_v1.1"
PROMPTS    = ["brain"]
DEVICE     = torch.device("cuda:0")

# Disable disk cache — cold embed (forward pass only, model already loaded)
_pred_module._load_disk_cache = lambda prompt, model_name: None
_pred_module._save_disk_cache = lambda prompt, model_name, embedding: None

print(f"Device : {DEVICE}")
print(f"GPU    : {torch.cuda.get_device_name(0)}")
print(f"Prompts: {PROMPTS}")

img, props = NibabelIOWithReorient().read_images([IMAGE_PATH])
print(f"Image loaded: {img.shape}\n")

print("Loading model...")
predictor = VoxTellPredictor(model_dir=MODEL_DIR, device=DEVICE)
predictor.tile_step_size = 0.5
predictor._embed_cache.clear()
print("Model loaded\n")

# Phase 1
t0 = time.perf_counter()
data, bbox, orig_shape = predictor.preprocess(img)
torch.cuda.synchronize()
t_pre = time.perf_counter() - t0
print(f"[Phase 1] Preprocessing: {t_pre:.3f}s  shape={data.shape}")

# Phase 2a — warm up model (first call loads backbone)
embeddings = predictor.embed_text_prompts(PROMPTS)
predictor._embed_cache.clear()

# Phase 2b — warm forward pass only
torch.cuda.synchronize()
t0 = time.perf_counter()
embeddings = predictor.embed_text_prompts(PROMPTS)
torch.cuda.synchronize()
t_embed = time.perf_counter() - t0
print(f"[Phase 2] Text embedding (warm): {t_embed:.3f}s  shape={embeddings.shape}")

# Phase 3 — sliding window
slicers = predictor._internal_get_sliding_window_slicers(data.shape[1:])
print(f"[Phase 3] Sliding window ({len(slicers)} patches)...")
for _ in range(2):
    _ = predictor.predict_sliding_window_return_logits(data, embeddings)
    torch.cuda.synchronize()
times = []
for _ in range(3):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    prediction = predictor.predict_sliding_window_return_logits(data, embeddings)
    torch.cuda.synchronize()
    times.append(time.perf_counter() - t0)
t_slide = float(np.mean(times))
print(f"  Time: {t_slide:.3f}s  (runs: {[f'{t:.3f}s' for t in times]})")

# Phase 4
t0 = time.perf_counter()
prediction = prediction.to("cpu")
with torch.no_grad():
    prediction_binary = torch.sigmoid(prediction.float()) > 0.5
seg = np.zeros([prediction_binary.shape[0], *orig_shape], dtype=np.uint8)
seg = insert_crop_into_image(seg, prediction_binary, bbox)
t_post = time.perf_counter() - t0
print(f"[Phase 4] Postprocessing: {t_post:.3f}s  shape={seg.shape}")

t_total = t_pre + t_embed + t_slide + t_post
print(f"\n{'='*50}")
print(f"RTX 4070 SUPER — v0_gpu baseline (1 prompt, FP16)")
print(f"{'='*50}")
print(f"  Preprocessing : {t_pre:.3f}s")
print(f"  Text embedding: {t_embed:.3f}s")
print(f"  Sliding window: {t_slide:.3f}s")
print(f"  Postprocessing: {t_post:.3f}s")
print(f"  TOTAL         : {t_total:.3f}s")
print(f"{'='*50}")