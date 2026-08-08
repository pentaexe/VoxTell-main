"""
nnInteractive Inference Benchmark — H100 MIG 3g.40gb
=====================================================
Profiles per-phase latency to identify optimization targets.

Phases timed
------------
  Phase 1: set_image  — preprocessing + image encoding (once per case)
  Phase 2: _predict   — network inference per object (one bbox prompt)

Baseline results (fold=0, no autozoom, warm model, H100 MIG 3g.40gb)
----------------------------------------------------------------------
  set_image : 0.345s   (once per case)
  _predict  : 0.108s   (per object, mean of 3 warm runs)
  cold start: ~2.4s    (first _predict call only — CUDA kernel compilation)

  For a 15-object case: 0.345 + 15 × 0.108 ≈ 1.96s

Results are used to decide which optimizations to pursue:
  - set_image (0.345s) is NOT the bottleneck
  - _predict (0.108s warm, 2.4s cold) is the target:
      → torch.compile to reduce cold-start and speed up warm calls
      → Investigate use_fold='all' (5× ensemble) performance
      → Autozoom impact measurement

Usage (Fir cluster):
    sbatch --account=rrg-jma nninteractive_benchmark.sh
"""

import time
from pathlib import Path

import numpy as np
import torch

CHECKPOINT_DIR = '/scratch/brianx7/nnInteractive_weights/nnInteractive_v1.0'
INPUT_DIR      = '/scratch/brianx7/cvpr_val/3D_val_npz'
N_WARMUP = 1
N_TIMED  = 3


def load_session(use_fold=0, do_autozoom=False):
    from nnInteractive.inference.inference_session import nnInteractiveInferenceSession
    session = nnInteractiveInferenceSession(
        device=torch.device('cuda', 0),
        use_torch_compile=False,
        verbose=False,
        torch_n_threads=int(__import__('os').environ.get('SLURM_CPUS_PER_TASK', '8')),
        do_autozoom=do_autozoom,
        use_pinned_memory=True,
    )
    session.initialize_from_trained_model_folder(
        model_training_output_dir=CHECKPOINT_DIR,
        use_fold=use_fold,
    )
    return session


def make_bbox(b):
    return [
        [b['z_min'],       b['z_max'] + 1],
        [b['z_mid_y_min'], b['z_mid_y_max'] + 1],
        [b['z_mid_x_min'], b['z_mid_x_max'] + 1],
    ]


def run_predict(session, bbox):
    session.reset_interactions()
    session.add_bbox_interaction(bbox, include_interaction=True, run_prediction=False)
    session.new_interaction_centers = [session.new_interaction_centers[-1]]
    session.new_interaction_zoom_out_factors = [session.new_interaction_zoom_out_factors[-1]]
    session._predict()


def main():
    data = np.load(next(Path(INPUT_DIR).glob('CT_*.npz')), allow_pickle=True)
    image, bbox_list = data['imgs'], data['boxes']
    bbox = make_bbox(bbox_list[0])

    print(f"Case shape : {image.shape}")
    print(f"GPU        : {torch.cuda.get_device_name(0)}")

    print("\nLoading session (fold=0, no autozoom)...")
    t0 = time.perf_counter()
    session = load_session(use_fold=0, do_autozoom=False)
    print(f"Model loaded in {time.perf_counter() - t0:.1f}s")

    # set_image (warm — called once, this IS the production pattern)
    t0 = time.perf_counter()
    session.set_image(image[None].astype(np.float32))
    session.set_target_buffer(torch.zeros(image.shape, dtype=torch.uint8))
    t_set_image = time.perf_counter() - t0
    print(f"\nset_image  : {t_set_image:.3f}s")

    # Warmup _predict (first call triggers CUDA kernel compilation)
    print(f"Warmup _predict ({N_WARMUP} call{'s' if N_WARMUP != 1 else ''})...")
    for _ in range(N_WARMUP):
        run_predict(session, bbox)

    # Timed _predict
    times = []
    for i in range(N_TIMED):
        t0 = time.perf_counter()
        run_predict(session, bbox)
        times.append(time.perf_counter() - t0)
        print(f"  _predict run {i+1}: {times[-1]:.3f}s")

    t_predict = float(np.mean(times))

    print(f"\n{'='*55}")
    print("BASELINE RESULTS (fold=0, no autozoom, H100 MIG 3g.40gb)")
    print(f"{'='*55}")
    print(f"  set_image : {t_set_image:.3f}s  (once per case)")
    print(f"  _predict  : {t_predict:.3f}s  (per object, mean of {N_TIMED} warm runs)")
    print(f"  total/obj : {t_set_image + t_predict:.3f}s")
    print(f"\nBottleneck: {'set_image' if t_set_image > t_predict else '_predict'}")


if __name__ == '__main__':
    main()