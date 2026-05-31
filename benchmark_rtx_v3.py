"""RTX 4070 SUPER v3 benchmark — 1 prompt, tile_step=0.75, no compile."""
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

# Disable disk cache to avoid triggering torch.compile path
_pred_module._load_disk_cache = lambda prompt, model_name: None
_pred_module._save_disk_cache = lambda prompt, model_name, embedding: None

img, _ = NibabelIOWithReorient().read_images([IMAGE_PATH])
predictor = VoxTellPredictor(model_dir=MODEL_DIR, device=DEVICE)
predictor.tile_step_size = 0.75
predictor._embed_cache.clear()

# Warm up model
predictor.embed_text_prompts(PROMPTS)
predictor._embed_cache.clear()

# Phase 1 — preprocess (run twice, time second)
predictor.preprocess(img)
torch.cuda.synchronize(); t0 = time.perf_counter()
data, bbox, orig_shape = predictor.preprocess(img)
torch.cuda.synchronize(); t_pre = time.perf_counter() - t0

# Phase 2 — embed (cache hit)
torch.cuda.synchronize(); t0 = time.perf_counter()
embeddings = predictor.embed_text_prompts(PROMPTS)
torch.cuda.synchronize(); t_embed = time.perf_counter() - t0

# Phase 3 — sliding window (2 warmup, 3 timed)
slicers = predictor._internal_get_sliding_window_slicers(data.shape[1:])
for _ in range(2):
    predictor.predict_sliding_window_return_logits(data, embeddings)
    torch.cuda.synchronize()
times = []
for _ in range(3):
    torch.cuda.synchronize(); t0 = time.perf_counter()
    prediction = predictor.predict_sliding_window_return_logits(data, embeddings)
    torch.cuda.synchronize(); times.append(time.perf_counter() - t0)
t_slide = float(np.mean(times))

# Phase 4 — postprocess
prediction = prediction.to("cpu")
t0 = time.perf_counter()
with torch.no_grad():
    pb = torch.sigmoid(prediction.float()) > 0.5
seg = np.zeros([pb.shape[0], *orig_shape], dtype=np.uint8)
seg = insert_crop_into_image(seg, pb, bbox)
t_post = time.perf_counter() - t0

t_total = t_pre + t_embed + t_slide + t_post
print(f"\n{'='*50}")
print(f"RTX 4070 SUPER — v3 (1 prompt, tile=0.75, cache)")
print(f"{'='*50}")
print(f"  Preprocessing : {t_pre:.3f}s  (patches: {len(slicers)})")
print(f"  Text embedding: {t_embed:.3f}s  (cache hit)")
print(f"  Sliding window: {t_slide:.3f}s  runs={[f'{t:.3f}' for t in times]}")
print(f"  Postprocessing: {t_post:.3f}s")
print(f"  TOTAL         : {t_total:.3f}s")
print(f"{'='*50}")