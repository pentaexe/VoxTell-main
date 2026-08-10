"""
DSC accuracy comparison: baseline (fold=0) vs torch.compile (fold=0)
Verifies torch.compile does not degrade segmentation accuracy.
"""
import torch, numpy as np
from pathlib import Path
from nnInteractive.inference.inference_session import nnInteractiveInferenceSession

CHECKPOINT_DIR = '/scratch/brianx7/nnInteractive_weights/nnInteractive_v1.0'
INPUT_DIR      = '/scratch/brianx7/cvpr_val/3D_val_npz'
N_CASES        = 10


def dice(pred, gt):
    pred, gt = pred.astype(bool), gt.astype(bool)
    inter = (pred & gt).sum()
    denom = pred.sum() + gt.sum()
    return 1.0 if denom == 0 else 2 * inter / denom


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


def evaluate(cases, use_compile):
    label = "torch.compile (fold=0)" if use_compile else "Baseline     (fold=0)"
    print(f"\n{'='*60}\n{label}\n{'='*60}")
    all_dscs = []

    for i, path in enumerate(cases):
        data   = np.load(path, allow_pickle=True)
        image  = data['imgs']
        bboxes = data['boxes']
        gts    = data['gts']

        session = nnInteractiveInferenceSession(
            device=torch.device('cuda', 0), use_torch_compile=False,
            verbose=False, torch_n_threads=8, do_autozoom=False, use_pinned_memory=True)
        session.initialize_from_trained_model_folder(CHECKPOINT_DIR, use_fold=0)

        if use_compile:
            session.network = torch.compile(session.network, mode='reduce-overhead')

        target_buf = torch.zeros(image.shape, dtype=torch.uint8)
        session.set_image(image[None].astype(np.float32))
        session.set_target_buffer(target_buf)

        n_warmup = 2 if use_compile else 1
        for _ in range(n_warmup):
            run_predict(session, make_bbox(bboxes[0]))

        case_dscs = []
        for j, b in enumerate(bboxes):
            target_buf.zero_()
            run_predict(session, make_bbox(b))
            dsc = dice(target_buf.cpu().numpy(), gts[j])
            case_dscs.append(dsc)

        all_dscs.extend(case_dscs)
        print(f"  [{i+1:2d}/{len(cases)}] {path.name}: {len(bboxes)} objects, mean DSC={np.mean(case_dscs):.4f}")
        del session
        torch.cuda.empty_cache()

    print(f"\n  Overall ({len(all_dscs)} objects): mean DSC = {np.mean(all_dscs):.4f}")
    return float(np.mean(all_dscs))


# Inspect data format first
sample = np.load(next(Path(INPUT_DIR).glob('CT_*.npz')), allow_pickle=True)
print("Keys:", list(sample.keys()))
print("imgs:", sample['imgs'].shape, "  boxes:", len(sample['boxes']))
if 'gts' not in sample:
    print("ERROR: 'gts' key not found. Available:", list(sample.keys()))
    exit(1)
print("gts:", np.array(sample['gts']).shape)

cases = sorted(Path(INPUT_DIR).glob('CT_*.npz'))[:N_CASES]
print(f"\nEvaluating {len(cases)} CT cases")

dsc_base = evaluate(cases, use_compile=False)
dsc_comp = evaluate(cases, use_compile=True)

print(f"\n{'='*60}")
print("ACCURACY COMPARISON")
print(f"{'='*60}")
print(f"  Baseline      (fold=0): {dsc_base:.4f}")
print(f"  torch.compile (fold=0): {dsc_comp:.4f}")
print(f"  Difference:             {abs(dsc_base - dsc_comp):.6f}")
if abs(dsc_base - dsc_comp) < 0.001:
    print("  Accuracy maintained:    YES")
else:
    print("  Accuracy maintained:    CHECK RESULTS")