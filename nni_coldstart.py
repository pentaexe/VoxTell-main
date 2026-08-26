"""
Cold-start Triton timing: measures true first-call compiled latency.
Uses a per-job temp cache dir — does NOT touch the shared Triton cache
at /scratch/brianx7/cache. Temp dir is removed at exit.

Usage:
    sbatch nni_coldstart.sh
"""
import os, time, shutil, atexit, tempfile
import numpy as np
import torch
from pathlib import Path
from nnInteractive.inference.inference_session import nnInteractiveInferenceSession

CHECKPOINT_DIR = '/scratch/brianx7/nnInteractive_weights/nnInteractive_v1.0'
INPUT_DIR      = '/scratch/brianx7/cvpr_val/3D_val_npz'

# Per-job isolated cache — shared cache at /scratch/brianx7/cache is untouched
_JOB_ID   = os.environ.get('SLURM_JOB_ID', 'local')
_TEMP_DIR = f'/scratch/brianx7/tmp_coldstart_{_JOB_ID}'
os.makedirs(_TEMP_DIR, exist_ok=True)
os.environ['XDG_CACHE_HOME'] = _TEMP_DIR
os.environ['TORCH_HOME']     = _TEMP_DIR

def _cleanup():
    if os.path.exists(_TEMP_DIR):
        shutil.rmtree(_TEMP_DIR, ignore_errors=True)
        print(f"\nCleaned up temp cache: {_TEMP_DIR}")

atexit.register(_cleanup)


def make_bbox(b):
    return [
        [int(b['z_min']),       int(b['z_max']) + 1],
        [int(b['z_mid_y_min']), int(b['z_mid_y_max']) + 1],
        [int(b['z_mid_x_min']), int(b['z_mid_x_max']) + 1],
    ]


def run_predict(session, bbox):
    session.reset_interactions()
    session.add_bbox_interaction(bbox, include_interaction=True, run_prediction=False)
    session.new_interaction_centers          = [session.new_interaction_centers[-1]]
    session.new_interaction_zoom_out_factors = [session.new_interaction_zoom_out_factors[-1]]
    session._predict()


print(f"Hardware:    {torch.cuda.get_device_name(0)}")
print(f"Cache dir:   {_TEMP_DIR}  (isolated, shared cache untouched)")
print(f"Shared cache stays at: /scratch/brianx7/cache  (not modified)")

# Load first case
case_path = sorted(Path(INPUT_DIR).glob('CT_*.npz'))[0]
data      = np.load(case_path, allow_pickle=True)
image     = data['imgs']
bboxes    = data.get('boxes')
bbox      = make_bbox(bboxes[0])
print(f"Case:        {case_path.name}  shape={image.shape}")

# Load session
print("\nLoading session (fold='all', autozoom=True)...")
t0 = time.perf_counter()
session = nnInteractiveInferenceSession(
    device=torch.device('cuda', 0),
    use_torch_compile=False,
    verbose=False,
    torch_n_threads=os.cpu_count(),
    do_autozoom=True,
    use_pinned_memory=True,
)
session.initialize_from_trained_model_folder(CHECKPOINT_DIR, use_fold='all')
print(f"Session loaded in {time.perf_counter() - t0:.1f}s")

target_buf = torch.zeros(image.shape, dtype=torch.uint8, device='cpu')
session.set_image(image[None].astype(np.float32))
session.set_target_buffer(target_buf)

# Baseline warm _predict (no compile)
print("\nBaseline (no compile) — 3 warm runs:")
for i in range(4):
    target_buf.zero_()
    t = time.perf_counter()
    run_predict(session, bbox)
    elapsed = time.perf_counter() - t
    label = "warmup" if i == 0 else f"run {i}"
    print(f"  {label}: {elapsed:.4f}s")

del session
torch.cuda.empty_cache()

# Compiled session — fresh cache, measures true Triton cold start
print("\nApplying torch.compile (reduce-overhead) with EMPTY Triton cache...")
session2 = nnInteractiveInferenceSession(
    device=torch.device('cuda', 0),
    use_torch_compile=False,
    verbose=False,
    torch_n_threads=os.cpu_count(),
    do_autozoom=True,
    use_pinned_memory=True,
)
session2.initialize_from_trained_model_folder(CHECKPOINT_DIR, use_fold='all')
session2.network = torch.compile(session2.network, mode='reduce-overhead')

target_buf2 = torch.zeros(image.shape, dtype=torch.uint8, device='cpu')
session2.set_image(image[None].astype(np.float32))
session2.set_target_buffer(target_buf2)

print("Compiled runs (no prior cache):")
times_compile = []
for i in range(5):
    target_buf2.zero_()
    t = time.perf_counter()
    run_predict(session2, bbox)
    elapsed = time.perf_counter() - t
    times_compile.append(elapsed)
    label = f"run {i+1} {'← Triton compile' if i == 0 else ('← residual' if i == 1 else '← fully warm')}"
    print(f"  {label}: {elapsed:.4f}s")

print(f"\n{'='*60}")
print("COLD-START RESULTS — fold='all', autozoom=ON, H100 MIG")
print(f"{'='*60}")
print(f"  Triton cold-start (run 1):  {times_compile[0]:.2f}s")
print(f"  Residual (run 2):           {times_compile[1]:.4f}s")
print(f"  Fully warm (runs 3-5 mean): {np.mean(times_compile[2:]):.4f}s")
print(f"  Baseline warm:              ~0.108s  (from nni_combined job)")
print()
warm_gain = 0.108 - np.mean(times_compile[2:])
if warm_gain > 0:
    breakeven = int(times_compile[0] / warm_gain)
    print(f"  Warm gain per object:       {warm_gain:.4f}s")
    print(f"  Break-even (cold cache):    ~{breakeven} objects")
    print(f"  First case (15 obj) cold:   {times_compile[0] + 15*np.mean(times_compile[2:]):.1f}s  vs baseline 1.96s")
print()
print("  Shared cache at /scratch/brianx7/cache: UNCHANGED")
