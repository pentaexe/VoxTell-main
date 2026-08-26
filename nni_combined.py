"""
Combined fold='all' benchmark: latency + DSC in one run, same config.
Runs baseline then torch.compile, both with fold='all', autozoom=True.
Produces speedup and DSC from identical cases/config — no mismatch.

Usage:
    sbatch nni_combined.sh
"""
import os, time, shutil
import numpy as np
import torch
from pathlib import Path
from nnInteractive.inference.inference_session import nnInteractiveInferenceSession

CHECKPOINT_DIR = '/scratch/brianx7/nnInteractive_weights/nnInteractive_v1.0'
INPUT_DIR      = '/scratch/brianx7/cvpr_val/3D_val_npz'
GT_DIR         = '/scratch/brianx7/cvpr_val/3D_val_gt/3D_val_gt_interactive'
PRED_BASE_DIR  = '/scratch/brianx7/nninteractive_preds_combined_base'
PRED_COMP_DIR  = '/scratch/brianx7/nninteractive_preds_combined_compile'
N_CASES        = 20
N_WARMUP       = 1        # baseline
N_WARMUP_COMP  = 2        # compiled: 1st=Triton compile, 2nd=stabilize


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
    all_dsc, skipped = [], 0
    for case_path in case_list:
        pred_path = pred_dir / case_path.name
        gt_path   = gt_dir   / case_path.name
        if not pred_path.exists() or not gt_path.exists():
            skipped += 1
            continue
        pred   = np.load(pred_path, allow_pickle=True)['segs']
        gt     = np.load(gt_path,   allow_pickle=True)['gts']
        labels = np.unique(gt)
        labels = labels[labels > 0]
        for lbl in labels:
            d = dice(pred, gt, lbl)
            if not np.isnan(d):
                all_dsc.append(d)
    if skipped:
        print(f"  (skipped {skipped} cases — no pred or GT)")
    return float(np.mean(all_dsc)), len(all_dsc)


def make_session():
    session = nnInteractiveInferenceSession(
        device=torch.device('cuda', 0),
        use_torch_compile=False,
        verbose=False,
        torch_n_threads=os.cpu_count(),   # must be os.cpu_count() — SLURM_CPUS_PER_TASK causes hang with fold='all'
        do_autozoom=True,
        use_pinned_memory=True,
    )
    session.initialize_from_trained_model_folder(CHECKPOINT_DIR, use_fold='all')
    return session


def run_batch(output_dir, use_compile, cases, warmup_case):
    output_dir = Path(output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    label = "torch.compile (fold='all', autozoom=ON)" if use_compile else "Baseline     (fold='all', autozoom=ON)"
    print(f"\n{'='*65}\n{label}\n{'='*65}")

    print("Loading session...")
    t_load = time.perf_counter()
    session = make_session()
    print(f"Session loaded in {time.perf_counter() - t_load:.1f}s")

    if use_compile:
        print("Applying torch.compile (reduce-overhead)...")
        session.network = torch.compile(session.network, mode='reduce-overhead')

    # Warmup on first case (not counted in latency or DSC)
    n_warmup = N_WARMUP_COMP if use_compile else N_WARMUP
    print(f"Warming up ({n_warmup} pass{'es' if n_warmup > 1 else ''} on first case)...")
    wdata  = np.load(warmup_case, allow_pickle=True)
    wimage = wdata['imgs']
    wbboxes = wdata.get('boxes')
    if wbboxes is not None:
        wbuf = torch.zeros(wimage.shape, dtype=torch.uint8, device='cpu')
        session.set_image(wimage[None].astype(np.float32))
        session.set_target_buffer(wbuf)
        for w in range(n_warmup):
            t_w = time.perf_counter()
            wbuf.zero_()
            run_predict(session, make_bbox(wbboxes[0]))
            print(f"  warmup {w+1}: {time.perf_counter() - t_w:.3f}s")

    # Timed runs
    times, obj_times = [], []
    for i, case_path in enumerate(cases):
        out_path = output_dir / case_path.name
        data    = np.load(case_path, allow_pickle=True)
        image   = data['imgs']
        bboxes  = data.get('boxes')
        if bboxes is None:
            print(f"  [{i+1}/{len(cases)}] {case_path.name}  SKIP (no bbox)")
            continue

        target_buf = torch.zeros(image.shape, dtype=torch.uint8, device='cpu')
        session.set_image(image[None].astype(np.float32))
        session.set_target_buffer(target_buf)

        t0     = time.perf_counter()
        result = torch.zeros(image.shape, dtype=torch.uint8)
        for oid, b in enumerate(bboxes):
            target_buf.zero_()
            t_obj = time.perf_counter()
            run_predict(session, make_bbox(b))
            obj_times.append(time.perf_counter() - t_obj)
            result[target_buf > 0] = oid + 1

        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        np.savez_compressed(out_path, segs=result.numpy().astype(np.uint8))
        print(f"  [{i+1}/{len(cases)}] {case_path.name}  {elapsed:.2f}s  ({len(bboxes)} objects)")

    print(f"\nLatency summary:")
    print(f"  Mean per-object (_predict warm): {np.mean(obj_times):.4f}s")
    print(f"  Mean per-case:                   {np.mean(times):.2f}s")
    print(f"  Total:                           {sum(times)/60:.1f} min")

    print("\nComputing DSC...")
    dsc, n = eval_dsc(output_dir, GT_DIR, cases)
    print(f"  Mean DSC: {dsc:.4f}  ({n} objects)")
    return dsc, n, float(np.mean(obj_times)), float(np.mean(times))


cases = sorted(Path(INPUT_DIR).glob('CT_*.npz'))[:N_CASES]
warmup_case = cases[0]
eval_cases  = cases   # warmup uses same first case; pred is not saved for warmup

print(f"Hardware: {torch.cuda.get_device_name(0)}")
print(f"torch_n_threads: os.cpu_count() = {os.cpu_count()}")
print(f"Cases: {len(cases)}  |  fold='all'  |  autozoom=ON  |  N_WARMUP_COMP={N_WARMUP_COMP}")

dsc_b, n_b, obj_b, case_b = run_batch(PRED_BASE_DIR, False, eval_cases, warmup_case)
dsc_c, n_c, obj_c, case_c = run_batch(PRED_COMP_DIR, True,  eval_cases, warmup_case)

print(f"\n{'='*65}")
print("COMBINED RESULTS — fold='all', autozoom=ON, H100 MIG")
print(f"{'='*65}")
print(f"  Config: fold='all', do_autozoom=True, torch_n_threads=os.cpu_count()")
print(f"  Cases: {len(eval_cases)}  |  N_WARMUP_COMPILE={N_WARMUP_COMP}")
print()
print(f"  {'':30s}  {'Baseline':>10}  {'Compiled':>10}  {'Speedup':>8}")
print(f"  {'_predict (warm, per-object)':30s}  {obj_b:>10.4f}s  {obj_c:>10.4f}s  {obj_b/obj_c:>7.2f}×")
print(f"  {'Mean per-case':30s}  {case_b:>10.2f}s  {case_c:>10.2f}s  {case_b/case_c:>7.2f}×")
print()
print(f"  DSC:  Baseline {dsc_b:.4f}  →  Compiled {dsc_c:.4f}  (Δ {dsc_c-dsc_b:+.4f})")
if abs(dsc_c - dsc_b) < 0.005:
    print("  Verdict: accuracy maintained (< 0.005 DSC change)")
else:
    print("  Verdict: accuracy change exceeds 0.005 — review results")
print()
cold_s = 23.61  # measured in job 56914757, /tmp-backed fully isolated cache
obj_per_case = n_b / len(eval_cases)
be_obj = int(cold_s / (obj_b - obj_c))
be_case = int(be_obj / obj_per_case)
print(f"  Cold-cache note: Triton cold-start ~{cold_s:.2f}s (job 56914757, /tmp; /scratch NFS will be higher).")
print(f"  Break-even vs baseline: ~{be_obj} objects (~{be_case} cases, {obj_per_case:.1f} obj/case).")
print("  After break-even, all subsequent objects run at the compiled speed.")
