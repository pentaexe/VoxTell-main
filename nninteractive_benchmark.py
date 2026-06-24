"""
nnInteractive Inference Benchmark — H100 MIG 3g.40gb
=====================================================
Profiles per-phase latency to identify optimization targets.
Mirrors the VoxTell benchmark structure for direct comparison.

Phases timed
------------
  Phase 1: set_image  — preprocessing + image encoding (once per case)
  Phase 2: _predict   — network inference per object (one bbox prompt)

Results are used to decide which optimizations to pursue:
  - If set_image dominates → optimize preprocessing / image encoder
  - If _predict dominates  → optimize sliding window (torch.compile, TensorRT, batch)

Usage (Fir cluster):
    python nninteractive_benchmark.py --input_dir /scratch/brianx7/cvpr_val/3D_val_CT
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch

CHECKPOINT_DIR = '/scratch/brianx7/nnInteractive_weights/nnInteractive_v1.0'
N_WARMUP = 2
N_TIMED  = 3


def load_session(use_torch_compile: bool = False):
    from nnInteractive.inference.inference_session import nnInteractiveInferenceSession
    session = nnInteractiveInferenceSession(
        device=torch.device('cuda', 0),
        use_torch_compile=use_torch_compile,
        verbose=False,
        torch_n_threads=os.cpu_count(),
        do_autozoom=True,
        use_pinned_memory=True,
    )
    session.initialize_from_trained_model_folder(
        model_training_output_dir=CHECKPOINT_DIR,
        use_fold='all',
    )
    return session


def make_bbox(b):
    return [
        [b['z_min'],       b['z_max'] + 1],
        [b['z_mid_y_min'], b['z_mid_y_max'] + 1],
        [b['z_mid_x_min'], b['z_mid_x_max'] + 1],
    ]


def run_benchmark(session, image, bbox_list, label: str):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    target_buffer = torch.zeros(image.shape, dtype=torch.uint8, device='cpu')

    # ── Phase 1: set_image ────────────────────────────────────────────────────
    # Warmup
    session.set_image(image[None].astype(np.float32))
    session.set_target_buffer(target_buffer)

    # Timed
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    session.set_image(image[None].astype(np.float32))
    session.set_target_buffer(target_buffer)
    torch.cuda.synchronize()
    t_set_image = time.perf_counter() - t0
    print(f"  Phase 1  set_image   : {t_set_image:.3f}s  (image shape: {image.shape})")

    # ── Phase 2: _predict (first object with bbox) ────────────────────────────
    bbox_here = make_bbox(bbox_list[0])

    # Warmup passes
    for _ in range(N_WARMUP):
        session.reset_interactions()
        session.add_bbox_interaction(bbox_here, include_interaction=True, run_prediction=False)
        session.new_interaction_centers = [session.new_interaction_centers[-1]]
        session.new_interaction_zoom_out_factors = [session.new_interaction_zoom_out_factors[-1]]
        torch.cuda.synchronize()
        session._predict()
        torch.cuda.synchronize()

    # Timed passes
    times = []
    for _ in range(N_TIMED):
        session.reset_interactions()
        session.add_bbox_interaction(bbox_here, include_interaction=True, run_prediction=False)
        session.new_interaction_centers = [session.new_interaction_centers[-1]]
        session.new_interaction_zoom_out_factors = [session.new_interaction_zoom_out_factors[-1]]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        session._predict()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    t_predict = float(np.mean(times))
    print(f"  Phase 2  _predict    : {t_predict:.3f}s  "
          f"(runs: {[f'{t:.3f}' for t in times]})")

    t_total = t_set_image + t_predict
    print(f"  {'─'*54}")
    print(f"  Total (1 object)     : {t_total:.3f}s")
    print(f"    set_image  {t_set_image/t_total*100:.0f}%  |  _predict  {t_predict/t_total*100:.0f}%")

    return t_set_image, t_predict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True,
                        help='Directory containing CT_*.npz validation cases')
    parser.add_argument('--case', default=None,
                        help='Specific .npz filename to benchmark (default: first found)')
    parser.add_argument('--skip_compile', action='store_true',
                        help='Skip the torch.compile comparison (saves ~1h of kernel compilation)')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    cases = sorted(input_dir.glob('CT_*.npz'))
    if not cases:
        raise FileNotFoundError(f"No CT_*.npz files found in {input_dir}")

    case_path = input_dir / args.case if args.case else cases[0]
    print(f"Benchmark case: {case_path.name}")

    data = np.load(case_path, allow_pickle=True)
    image = data['imgs']
    bbox_list = data.get('boxes')
    if bbox_list is None:
        raise ValueError(f"Case {case_path.name} has no bounding-box prompts")

    print(f"Image shape : {image.shape}")
    print(f"Objects     : {len(bbox_list)}")
    print(f"GPU         : {torch.cuda.get_device_name(0)}")

    # ── Baseline: no torch.compile ────────────────────────────────────────────
    print("\nLoading session (no torch.compile)...")
    t0 = time.perf_counter()
    session = load_session(use_torch_compile=False)
    print(f"Model loaded in {time.perf_counter() - t0:.1f}s")

    t_set_v0, t_pred_v0 = run_benchmark(session, image, bbox_list,
                                         "Baseline  (use_torch_compile=False)")
    del session
    torch.cuda.empty_cache()

    # ── torch.compile (Triton available on H100) ──────────────────────────────
    if not args.skip_compile:
        print("\nLoading session (torch.compile=True)...")
        t0 = time.perf_counter()
        session_compiled = load_session(use_torch_compile=True)
        print(f"Model loaded in {time.perf_counter() - t0:.1f}s  "
              f"(includes compile warmup)")

        t_set_v1, t_pred_v1 = run_benchmark(session_compiled, image, bbox_list,
                                              "torch.compile (use_torch_compile=True)")
        del session_compiled
        torch.cuda.empty_cache()

        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"{'':30s} {'Baseline':>10} {'torch.compile':>14} {'Speedup':>8}")
        print(f"{'─'*60}")
        print(f"{'set_image':30s} {t_set_v0:>9.3f}s {t_set_v1:>13.3f}s "
              f"{t_set_v0/t_set_v1:>7.2f}×")
        print(f"{'_predict (1 object)':30s} {t_pred_v0:>9.3f}s {t_pred_v1:>13.3f}s "
              f"{t_pred_v0/t_pred_v1:>7.2f}×")
        print(f"{'Total (1 object)':30s} {t_set_v0+t_pred_v0:>9.3f}s "
              f"{t_set_v1+t_pred_v1:>13.3f}s "
              f"{(t_set_v0+t_pred_v0)/(t_set_v1+t_pred_v1):>7.2f}×")
        print(f"{'='*60}")
    else:
        print(f"\n{'='*60}")
        print("BASELINE RESULTS")
        print(f"{'='*60}")

    print(f"\n  set_image  : {t_set_v0:.3f}s")
    print(f"  _predict   : {t_pred_v0:.3f}s  (1 object, mean of {N_TIMED} runs)")
    print(f"  Total      : {t_set_v0+t_pred_v0:.3f}s")
    print(f"\nBottleneck: "
          f"{'set_image' if t_set_v0 > t_pred_v0 else '_predict'} dominates "
          f"({'%.0f' % (max(t_set_v0,t_pred_v0)/(t_set_v0+t_pred_v0)*100)}% of total)")


if __name__ == '__main__':
    main()