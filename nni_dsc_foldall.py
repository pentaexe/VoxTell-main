"""
DSC comparison: fold='all' baseline vs fold='all' + torch.compile.
Uses the official checkpoint (fold='all') — comparable to the 0.7794 CodaBench submission.

Usage:
    sbatch nni_dsc_foldall.sh
"""
import os, time, shutil
import numpy as np
import torch
from pathlib import Path
from nnInteractive.inference.inference_session import nnInteractiveInferenceSession

CHECKPOINT_DIR = '/scratch/brianx7/nnInteractive_weights/nnInteractive_v1.0'
INPUT_DIR      = '/scratch/brianx7/cvpr_val/3D_val_npz'
GT_DIR         = '/scratch/brianx7/cvpr_val/3D_val_gt/3D_val_gt_interactive'
PRED_BASE_DIR  = '/scratch/brianx7/nninteractive_preds_foldall_base'
PRED_COMP_DIR  = '/scratch/brianx7/nninteractive_preds_foldall_compile'
N_CASES        = 20   # set to None to run all 881


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


def dice(pred, gt, label):
    p = pred == label
    g = gt == label
    inter = (p & g).sum()
    union = p.sum() + g.sum()
    if union == 0:
        return float('nan')
    return 2 * inter / union


def eval_dsc(pred_dir, gt_dir, case_list):
    pred_dir, gt_dir = Path(pred_dir), Path(gt_dir)
    all_dsc = []
    skipped = 0
    for case_path in case_list:
        pred_path = pred_dir / case_path.name
        gt_path   = gt_dir   / case_path.name
        if not pred_path.exists() or not gt_path.exists():
            skipped += 1
            continue
        pred = np.load(pred_path, allow_pickle=True)['segs']
        gt   = np.load(gt_path,   allow_pickle=True)['gts']
        labels = np.unique(gt)
        labels = labels[labels > 0]
        for lbl in labels:
            d = dice(pred, gt, lbl)
            if not np.isnan(d):
                all_dsc.append(d)
    if skipped:
        print(f"  (skipped {skipped} cases — no pred or GT)")
    return float(np.mean(all_dsc)), len(all_dsc)


def run_batch(output_dir, use_compile):
    output_dir = Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    label = "torch.compile (fold='all')" if use_compile else "Baseline     (fold='all')"
    print(f"\n{'='*60}\n{label}\n{'='*60}")

    cases = sorted(Path(INPUT_DIR).glob('CT_*.npz'))
    if N_CASES is not None:
        cases = cases[:N_CASES]
    print(f"Running {len(cases)} CT cases")

    print("Loading session (fold='all', do_autozoom=True)...")
    t_load = time.perf_counter()
    session = nnInteractiveInferenceSession(
        device=torch.device('cuda', 0),
        use_torch_compile=False,
        verbose=False,
        torch_n_threads=os.cpu_count(),
        do_autozoom=True,
        use_pinned_memory=True,
    )
    session.initialize_from_trained_model_folder(CHECKPOINT_DIR, use_fold='all')
    print(f"Session loaded in {time.perf_counter() - t_load:.1f}s")

    if use_compile:
        print("Applying torch.compile (reduce-overhead)...")
        session.network = torch.compile(session.network, mode='reduce-overhead')

    times = []
    for i, case_path in enumerate(cases):
        out_path = output_dir / case_path.name

        data   = np.load(case_path, allow_pickle=True)
        image  = data['imgs']
        bboxes = data.get('boxes')
        if bboxes is None:
            print(f"  [{i+1}/{len(cases)}] {case_path.name}  SKIP (no bbox)")
            continue

        target_buf = torch.zeros(image.shape, dtype=torch.uint8, device='cpu')
        session.set_image(image[None].astype(np.float32))
        session.set_target_buffer(target_buf)

        t0 = time.perf_counter()
        result = torch.zeros(image.shape, dtype=torch.uint8)

        for oid, b in enumerate(bboxes):
            target_buf.zero_()
            run_predict(session, make_bbox(b))
            result[target_buf > 0] = oid + 1

        elapsed = time.perf_counter() - t0
        times.append(elapsed)

        np.savez_compressed(out_path, segs=result.numpy().astype(np.uint8))

        if i == 0 and use_compile:
            print(f"  [{i+1}/{len(cases)}] {case_path.name}  {elapsed:.1f}s  "
                  f"(includes Triton compilation)")
        else:
            print(f"  [{i+1}/{len(cases)}] {case_path.name}  {elapsed:.2f}s")

    if times:
        skip_first = 1 if use_compile else 0
        warm_times = times[skip_first:]
        print(f"Inference done. Mean/case (warm): {np.mean(warm_times):.2f}s  "
              f"Total: {sum(times)/60:.1f} min")

    print("Computing DSC...")
    dsc, n = eval_dsc(output_dir, GT_DIR, cases)
    print(f"Mean DSC: {dsc:.4f}  ({n} objects)")
    return dsc, n


dsc_base, n_base = run_batch(PRED_BASE_DIR, use_compile=False)
dsc_comp, n_comp = run_batch(PRED_COMP_DIR, use_compile=True)

print(f"\n{'='*60}")
print("DSC COMPARISON — fold='all', do_autozoom=True, H100 MIG")
print(f"{'='*60}")
print(f"  Baseline      (fold='all'): {dsc_base:.4f}  ({n_base} objects)")
print(f"  torch.compile (fold='all'): {dsc_comp:.4f}  ({n_comp} objects)")
print(f"  Difference:                 {dsc_comp - dsc_base:+.4f}")
print(f"  Objects scored: {n_base} (baseline) / {n_comp} (compiled).")
print("  Note: no pass/fail threshold applied. At this object count a delta of this")
print("        magnitude is within sampling noise — report the number and the n.")