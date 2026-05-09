"""
Batch nnInteractive inference on CVPR BiomedSegFM CT validation set.
Loads the model once, processes all cases in a loop.
Saves predictions to output_dir as .npz files (key: segs, dtype: uint8).
"""
import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch

from nnInteractive.inference.inference_session import nnInteractiveInferenceSession
from nnunetv2.utilities.helpers import empty_cache

CHECKPOINT_DIR = '/scratch/brianx7/nnInteractive_weights/nnInteractive_v1.0'


def predict_case(session, image, bbox):
    session.set_image(image[None].astype(np.float32))
    target_buffer = torch.zeros(image.shape, dtype=torch.uint8, device='cpu')
    session.set_target_buffer(target_buffer)
    result = torch.zeros(image.shape, dtype=torch.uint8)

    for oid in range(1, len(bbox) + 1):
        session.reset_interactions()
        b = bbox[oid - 1]
        bbox_here = [
            [b['z_min'], b['z_max'] + 1],
            [b['z_mid_y_min'], b['z_mid_y_max'] + 1],
            [b['z_mid_x_min'], b['z_mid_x_max'] + 1],
        ]
        session.add_bbox_interaction(bbox_here, include_interaction=True, run_prediction=False)
        session.new_interaction_centers = [session.new_interaction_centers[-1]]
        session.new_interaction_zoom_out_factors = [session.new_interaction_zoom_out_factors[-1]]
        session._predict()
        result[session.target_buffer > 0] = oid

    return result.cpu().numpy().astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--max_cases', type=int, default=None)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = sorted(input_dir.glob('CT_*.npz'))
    if args.max_cases:
        cases = cases[:args.max_cases]
    print(f"Found {len(cases)} CT cases")

    print("Loading nnInteractive session...")
    t0 = time.perf_counter()
    session = nnInteractiveInferenceSession(
        device=torch.device('cuda', 0),
        use_torch_compile=False,
        verbose=False,
        torch_n_threads=os.cpu_count(),
        do_autozoom=True,
        use_pinned_memory=True,
    )
    session.initialize_from_trained_model_folder(
        model_training_output_dir=CHECKPOINT_DIR,
        use_fold='all',
    )
    print(f"Model loaded in {time.perf_counter() - t0:.1f}s")

    times = []
    for i, case_path in enumerate(cases):
        out_path = output_dir / case_path.name
        if out_path.exists():
            print(f"[{i+1}/{len(cases)}] Skip (exists): {case_path.name}")
            continue

        data = np.load(case_path, allow_pickle=True)
        image = data['imgs']
        bbox = data['boxes']

        t0 = time.perf_counter()
        seg = predict_case(session, image, bbox)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)

        np.savez_compressed(out_path, segs=seg)
        print(f"[{i+1}/{len(cases)}] {case_path.name}  shape={image.shape}  "
              f"objs={len(bbox)}  {elapsed:.2f}s")

    print(f"\nDone. Mean per-case: {np.mean(times):.2f}s  Total: {sum(times):.1f}s")


if __name__ == '__main__':
    main()
