"""Measure in-memory cache hit time (CPU RAM, no disk involved)."""
import time
import torch
from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient
from voxtell.inference.predictor import VoxTellPredictor

IMAGE_PATH = r"C:\Users\brian\Downloads\mni_icbm152_t1_tal_nlin_sym_09a.nii.gz"
MODEL_DIR  = r"C:\Users\brian\OneDrive\Desktop\Code\VoxTell-main\models\voxtell_v1.1"
PROMPTS    = ["brain"]
DEVICE     = torch.device("cuda:0")

img, _ = NibabelIOWithReorient().read_images([IMAGE_PATH])
predictor = VoxTellPredictor(model_dir=MODEL_DIR, device=DEVICE)

# Fill in-memory cache (forward pass)
print("Warming up (fills in-memory cache)...")
predictor.embed_text_prompts(PROMPTS)

# Time in-memory cache hit — no disk access, embedding already in CPU RAM
times = []
for i in range(5):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    embeddings = predictor.embed_text_prompts(PROMPTS)
    torch.cuda.synchronize()
    times.append(time.perf_counter() - t0)

import numpy as np
print(f"\nIn-memory cache hit times: {[f'{t:.4f}s' for t in times]}")
print(f"Mean: {np.mean(times):.4f}s")
print(f"Shape: {embeddings.shape}")