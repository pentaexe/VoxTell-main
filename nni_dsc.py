"""
Output consistency check: baseline (fold=0) vs torch.compile (fold=0)
Verifies torch.compile produces identical predictions — no ground truth needed.
The validation set has no local ground truth (scores come from CodaBench).
"""
import torch, numpy as np
from pathlib import Path
from nnInteractive.inference.inference_session import nnInteractiveInferenceSession

CHECKPOINT_DIR = '/scratch/brianx7/nnInteractive_weights/nnInteractive_v1.0'
INPUT_DIR      = '/scratch/brianx7/cvpr_val/3D_val_npz'
N_CASES        = 10


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


def get_predictions(path, use_compile):
    data   = np.load(path, allow_pickle=True)
    image  = data['imgs']
    bboxes = data['boxes']

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

    preds = []
    for b in bboxes:
        target_buf.zero_()
        run_predict(session, make_bbox(b))
        preds.append(target_buf.cpu().numpy().copy())

    del session
    torch.cuda.empty_cache()
    return preds


cases = sorted(Path(INPUT_DIR).glob('CT_*.npz'))[:N_CASES]
print(f"Comparing baseline vs torch.compile predictions on {len(cases)} CT cases\n")

all_identical = True
all_diffs = []

for i, path in enumerate(cases):
    preds_base = get_predictions(path, use_compile=False)
    preds_comp = get_predictions(path, use_compile=True)

    case_identical = True
    for j, (pb, pc) in enumerate(zip(preds_base, preds_comp)):
        identical = np.array_equal(pb, pc)
        diff_voxels = int((pb != pc).sum())
        all_diffs.append(diff_voxels)
        if not identical:
            case_identical = False
            all_identical = False
        print(f"  [{i+1:2d}/{len(cases)}] {path.name} obj {j}: "
              f"{'IDENTICAL' if identical else f'DIFFERS ({diff_voxels} voxels)'}")

    if case_identical:
        print(f"  [{i+1:2d}/{len(cases)}] {path.name}: all objects IDENTICAL")

print(f"\n{'='*60}")
print("OUTPUT CONSISTENCY SUMMARY")
print(f"{'='*60}")
print(f"  Cases evaluated:          {len(cases)}")
print(f"  Predictions identical:    {'YES — torch.compile output matches baseline exactly' if all_identical else 'NO — see differences above'}")
if not all_identical:
    print(f"  Max voxel diff:           {max(all_diffs)}")
    print(f"  Mean voxel diff:          {np.mean(all_diffs):.1f}")