"""
Compute mean DSC between nnInteractive predictions and ground truth.
Usage:
    python nninteractive_eval_dsc.py \
        --pred_dir /scratch/brianx7/nninteractive_preds \
        --gt_dir   /scratch/brianx7/cvpr_val/3D_val_gt/3D_val_gt_interactive
"""
import argparse
import numpy as np
from pathlib import Path


def dice(pred, gt, label):
    p = pred == label
    g = gt == label
    inter = (p & g).sum()
    union = p.sum() + g.sum()
    if union == 0:
        return float('nan')
    return 2 * inter / union


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_dir', required=True)
    parser.add_argument('--gt_dir',   required=True)
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    gt_dir   = Path(args.gt_dir)

    cases = sorted(pred_dir.glob('CT_*.npz'))
    print(f"Evaluating {len(cases)} cases...")

    all_dsc = []
    for case_path in cases:
        gt_path = gt_dir / case_path.name
        if not gt_path.exists():
            print(f"  SKIP (no GT): {case_path.name}")
            continue

        pred = np.load(case_path,  allow_pickle=True)['segs']
        gt   = np.load(gt_path,    allow_pickle=True)['segs']

        labels = np.unique(gt)
        labels = labels[labels > 0]
        case_dsc = [dice(pred, gt, l) for l in labels]
        case_dsc = [d for d in case_dsc if not np.isnan(d)]
        if case_dsc:
            all_dsc.extend(case_dsc)

    print(f"\nResults over {len(cases)} CT cases:")
    print(f"  Mean DSC : {np.mean(all_dsc):.4f}")
    print(f"  Median   : {np.median(all_dsc):.4f}")
    print(f"  Std      : {np.std(all_dsc):.4f}")
    print(f"  N objects: {len(all_dsc)}")


if __name__ == '__main__':
    main()