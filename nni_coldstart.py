"""
Cold-start Triton timing: measures true first-call compiled latency.
Uses a per-job temp dir on /tmp (node-local, fast I/O) so the shared
cache at /scratch/brianx7/cache is untouched. Temp dir is removed at exit.

Env vars are set BEFORE any torch import — TORCHINDUCTOR_CACHE_DIR is
read at import time, so ordering matters.

Usage:
    sbatch nni_coldstart.sh
"""
import os
import atexit
import shutil

# ── Cache isolation — must happen before any torch import ──
_JOB_ID   = os.environ.get('SLURM_JOB_ID', 'local')
_TEMP_DIR = f'/tmp/nni_coldstart_{_JOB_ID}'
os.makedirs(os.path.join(_TEMP_DIR, 'inductor'), exist_ok=True)

os.environ['XDG_CACHE_HOME']          = _TEMP_DIR
os.environ['TORCH_HOME']              = _TEMP_DIR
os.environ['TORCHINDUCTOR_CACHE_DIR'] = os.path.join(_TEMP_DIR, 'inductor')

# Prove isolation at startup
print(f"Cache isolation:")
print(f"  XDG_CACHE_HOME          = {os.environ['XDG_CACHE_HOME']}")
print(f"  TORCH_HOME              = {os.environ['TORCH_HOME']}")
print(f"  TORCHINDUCTOR_CACHE_DIR = {os.environ['TORCHINDUCTOR_CACHE_DIR']}")
inductor_files = list(os.scandir(os.environ['TORCHINDUCTOR_CACHE_DIR']))
assert len(inductor_files) == 0, f"Inductor cache not empty: {inductor_files}"
print(f"  Inductor cache confirmed empty before compile.")
print(f"  Shared cache at /scratch/brianx7/cache: NOT modified.")

# ── Now safe to import torch ──
import time
import numpy as np
import torch
from pathlib import Path
from nnInteractive.inference.inference_session import nnInteractiveInferenceSession

def _cleanup():
    if os.path.exists(_TEMP_DIR):
        shutil.rmtree(_TEMP_DIR, ignore_errors=True)
        print(f"\nCleaned up temp cache: {_TEMP_DIR}")

atexit.register(_cleanup)

CHECKPOINT_DIR = '/scratch/brianx7/nnInteractive_weights/nnInteractive_v1.0'
INPUT_DIR      = '/scratch/brianx7/cvpr_val/3D_val_npz'


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


print(f"\nHardware: {torch.cuda.get_device_name(0)}")
print(f"torch_n_threads: os.cpu_count() = {os.cpu_count()}")

# Load first case (same as job 56908464)
case_path = sorted(Path(INPUT_DIR).glob('CT_*.npz'))[0]
data      = np.load(case_path, allow_pickle=True)
image     = data['imgs']
bboxes    = data.get('boxes')
bbox      = make_bbox(bboxes[0])
print(f"Case: {case_path.name}  shape={image.shape}")

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

# Baseline warm (no compile) — 4 runs, first is warmup
print("\nBaseline (no compile) — 1 warmup + 3 timed:")
base_times = []
for i in range(4):
    target_buf.zero_()
    t = time.perf_counter()
    run_predict(session, bbox)
    elapsed = time.perf_counter() - t
    label = "warmup" if i == 0 else f"run {i}"
    print(f"  {label}: {elapsed:.4f}s")
    if i > 0:
        base_times.append(elapsed)

del session
torch.cuda.empty_cache()

# Compiled session — fresh /tmp-backed inductor cache
print("\nApplying torch.compile — inductor cache is empty (/tmp-backed)...")
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

print("Compiled runs (empty /tmp inductor cache):")
times_compile = []
for i in range(6):
    target_buf2.zero_()
    t = time.perf_counter()
    run_predict(session2, bbox)
    elapsed = time.perf_counter() - t
    times_compile.append(elapsed)
    if i == 0:   label = "← Triton compile (cold)"
    elif i == 1: label = "← residual"
    else:        label = "← fully warm"
    print(f"  run {i+1} {label}: {elapsed:.4f}s")

warm_baseline = float(np.mean(base_times))
warm_compiled = float(np.mean(times_compile[2:]))
warm_gain     = warm_baseline - warm_compiled

print(f"\n{'='*62}")
print("COLD-START RESULTS — fold='all', autozoom=ON, /tmp inductor cache")
print(f"{'='*62}")
print(f"  Triton cold-start (run 1):     {times_compile[0]:.2f}s")
print(f"  Residual (run 2):              {times_compile[1]:.4f}s")
print(f"  Fully warm mean (runs 3-6):    {warm_compiled:.4f}s")
print(f"  Baseline warm mean (runs 2-4): {warm_baseline:.4f}s")
print(f"  Warm gain per object:          {warm_gain:.4f}s")
if warm_gain > 0:
    breakeven_obj  = int(times_compile[0] / warm_gain)
    obj_per_case   = 294 / 20   # from job 56908464
    breakeven_case = int(breakeven_obj / obj_per_case)
    print(f"  Break-even:                    ~{breakeven_obj} objects (~{breakeven_case} cases)")
    print(f"  First case cold (~14.7 obj):   {times_compile[0] + 14.7*warm_compiled:.1f}s vs {14.7*warm_baseline:.1f}s baseline")
print()
print("  Note: /tmp is node-local. Production cache is on /scratch (network FS),")
print("  so production cold-start will read higher than this figure.")
print(f"  Shared cache at /scratch/brianx7/cache: UNCHANGED")
