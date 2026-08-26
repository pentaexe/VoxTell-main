"""
Controlled autozoom measurement: fold='all', baseline only, same cases,
autozoom=ON vs autozoom=OFF in one process. Closes O4 as a measured result.

Usage:
    sbatch nni_autozoom.sh
"""
import os, time
import numpy as np
import torch
from pathlib import Path
from nnInteractive.inference.inference_session import nnInteractiveInferenceSession

CHECKPOINT_DIR = '/scratch/brianx7/nnInteractive_weights/nnInteractive_v1.0'
INPUT_DIR      = '/scratch/brianx7/cvpr_val/3D_val_npz'
N_CASES        = 20
N_WARMUP       = 1


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


def run_batch(cases, do_autozoom):
    label = f"Baseline — fold='all', autozoom={'ON' if do_autozoom else 'OFF'}"
    print(f"\n{'='*60}\n{label}\n{'='*60}")

    session = nnInteractiveInferenceSession(
        device=torch.device('cuda', 0),
        use_torch_compile=False,
        verbose=False,
        torch_n_threads=os.cpu_count(),
        do_autozoom=do_autozoom,
        use_pinned_memory=True,
    )
    session.initialize_from_trained_model_folder(CHECKPOINT_DIR, use_fold='all')

    # Warmup
    wdata = np.load(cases[0], allow_pickle=True)
    wbboxes = wdata.get('boxes')
    if wbboxes is not None:
        wbuf = torch.zeros(wdata['imgs'].shape, dtype=torch.uint8, device='cpu')
        session.set_image(wdata['imgs'][None].astype(np.float32))
        session.set_target_buffer(wbuf)
        for _ in range(N_WARMUP):
            run_predict(session, make_bbox(wbboxes[0]))

    obj_times, case_times = [], []
    for i, case_path in enumerate(cases):
        data   = np.load(case_path, allow_pickle=True)
        image  = data['imgs']
        bboxes = data.get('boxes')
        if bboxes is None:
            continue

        target_buf = torch.zeros(image.shape, dtype=torch.uint8, device='cpu')
        session.set_image(image[None].astype(np.float32))
        session.set_target_buffer(target_buf)

        t_case = time.perf_counter()
        for b in bboxes:
            target_buf.zero_()
            t_obj = time.perf_counter()
            run_predict(session, make_bbox(b))
            obj_times.append(time.perf_counter() - t_obj)
        case_times.append(time.perf_counter() - t_case)

        if i < 3 or (i + 1) % 5 == 0:
            print(f"  [{i+1}/{len(cases)}] {case_path.name}  {case_times[-1]:.2f}s  ({len(bboxes)} obj)")

    print(f"\n  Mean per-object: {np.mean(obj_times):.4f}s")
    print(f"  Mean per-case:   {np.mean(case_times):.2f}s")
    return float(np.mean(obj_times)), float(np.mean(case_times))


cases = sorted(Path(INPUT_DIR).glob('CT_*.npz'))[:N_CASES]
print(f"Hardware: {torch.cuda.get_device_name(0)}")
print(f"torch_n_threads: os.cpu_count() = {os.cpu_count()}")
print(f"Cases: {len(cases)}  |  fold='all'  |  no compile  |  N_WARMUP={N_WARMUP}")

obj_off, case_off = run_batch(cases, do_autozoom=False)
obj_on,  case_on  = run_batch(cases, do_autozoom=True)

print(f"\n{'='*60}")
print("AUTOZOOM OVERHEAD — fold='all', baseline only, H100 MIG")
print(f"{'='*60}")
print(f"  {'':25s}  {'autozoom=OFF':>14}  {'autozoom=ON':>14}  {'Overhead':>10}")
print(f"  {'Mean per-object':25s}  {obj_off:>14.4f}s  {obj_on:>14.4f}s  {obj_on/obj_off:>9.2f}×")
print(f"  {'Mean per-case':25s}  {case_off:>14.2f}s  {case_on:>14.2f}s  {case_on/case_off:>9.2f}×")
print(f"\n  Autozoom adds {obj_on - obj_off:.4f}s per object on average ({(obj_on/obj_off - 1)*100:.0f}% overhead).")
