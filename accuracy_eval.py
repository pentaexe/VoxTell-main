"""
VoxTell Accuracy Evaluation — AMOS Abdomen Dataset
====================================================
Compares segmentation accuracy (DSC, NSD) between:
  - v0-style : tile_step_size=0.5  (original baseline overlap)
  - v3-style : tile_step_size=0.75 (optimized, 25% overlap)

Dataset format: NIfTI (.nii.gz) — AMOS abdominal multi-organ dataset
  imagesTs/  — amos_XXXX.nii.gz  (MRI/CT volumes)
  labelsTs/  — amos_XXXX.nii.gz  (multi-class ground truth, label IDs 1–15)

AMOS label mapping:
  1=spleen, 2=right kidney, 3=left kidney, 4=gallbladder, 5=esophagus,
  6=liver, 7=stomach, 8=aorta, 9=inferior vena cava, 10=pancreas,
  11=right adrenal gland, 12=left adrenal gland, 13=duodenum,
  14=bladder, 15=prostate/uterus

Usage:
  python accuracy_eval.py \
      --imgs   C:\\path\\to\\imagesTs \
      --gts    C:\\path\\to\\labelsTs \
      --model  C:\\path\\to\\voxtell_v1.1 \
      --cases  5 \
      --seed   42

Dependencies:
  pip install surface-distance SimpleITK
"""

import argparse
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import pandas as pd
from scipy.ndimage import zoom

from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient
from voxtell.inference.predictor import VoxTellPredictor

# Target spacing for resampling — matches VoxTell training distribution (~1mm MRI)
TARGET_SPACING_MM = 1.5

# ── AMOS organ label map ──────────────────────────────────────────────────────
AMOS_LABELS: Dict[int, str] = {
    1:  "liver",
    2:  "right kidney",
    3:  "spleen",
    4:  "pancreas",
    5:  "aorta",
    6:  "inferior vena cava",
    7:  "right adrenal gland",
    8:  "left adrenal gland",
    9:  "gallbladder",
    10: "esophagus",
    11: "stomach",
    12: "duodenum",
    13: "left kidney",
}

# ── Surface distance library ──────────────────────────────────────────────────
try:
    from surface_distance import (
        compute_surface_distances,
        compute_surface_dice_at_tolerance,
        compute_dice_coefficient,
    )
    _SURFACE_DIST_AVAILABLE = True
except ImportError:
    try:
        from SurfaceDice import (
            compute_surface_distances,
            compute_surface_dice_at_tolerance,
            compute_dice_coefficient,
        )
        _SURFACE_DIST_AVAILABLE = True
    except ImportError:
        _SURFACE_DIST_AVAILABLE = False
        print("[WARNING] surface-distance library not found — NSD uses scipy fallback.")
        print("          Install: pip install surface-distance\n")


# ── Metric helpers ─────────────────────────────────────────────────────────────

def dice_coefficient(gt: np.ndarray, pred: np.ndarray) -> float:
    if _SURFACE_DIST_AVAILABLE:
        return float(compute_dice_coefficient(gt.astype(bool), pred.astype(bool)))
    intersection = np.logical_and(gt, pred).sum()
    denom = gt.sum() + pred.sum()
    if denom == 0:
        return 1.0
    return float(2.0 * intersection / denom)


def normalized_surface_dice(
    gt: np.ndarray, pred: np.ndarray, spacing: np.ndarray, tolerance_mm: float = 2.0
) -> float:
    gt_bool   = gt.astype(bool)
    pred_bool = pred.astype(bool)
    if not gt_bool.any() and not pred_bool.any():
        return 1.0
    if not gt_bool.any() or not pred_bool.any():
        return 0.0
    if _SURFACE_DIST_AVAILABLE:
        surf = compute_surface_distances(gt_bool, pred_bool, spacing_mm=spacing)
        return float(compute_surface_dice_at_tolerance(surf, tolerance_mm))
    # scipy fallback
    from scipy.ndimage import binary_erosion
    gt_surf   = gt_bool   & ~binary_erosion(gt_bool)
    pred_surf = pred_bool & ~binary_erosion(pred_bool)
    denom = gt_surf.sum() + pred_surf.sum()
    if denom == 0:
        return 1.0
    return float((np.logical_and(gt_surf, pred_bool).sum() +
                  np.logical_and(pred_surf, gt_bool).sum()) / denom)


def compute_case_metrics(
    gt: np.ndarray,
    pred: np.ndarray,
    spacing: np.ndarray,
    class_ids: List[int],
    tolerance_mm: float = 2.0,
) -> Dict[int, Dict[str, float]]:
    results = {}
    for cid in class_ids:
        gt_bin   = (gt   == cid)
        pred_bin = (pred == cid)
        if not gt_bin.any() and not pred_bin.any():
            continue
        results[cid] = {
            "dsc": dice_coefficient(gt_bin, pred_bin),
            "nsd": normalized_surface_dice(gt_bin, pred_bin, spacing, tolerance_mm),
        }
    return results


# ── NIfTI loading + resampling ────────────────────────────────────────────────

def load_nifti(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load NIfTI using NibabelIOWithReorient — same reader used during VoxTell
    training. This reorients every image to canonical (RAS/LPS) orientation,
    which is critical for correct model input. SimpleITK returns raw voxel
    order without reorientation, causing orientation mismatch and near-zero DSC.

    Returns (array of shape (H, W, D), spacing_mm as float64 [x, y, z]).
    """
    reader = NibabelIOWithReorient()
    images, props = reader.read_images([str(path)])   # → list of (C, H, W, D)
    arr = images[0]                                    # (H, W, D)
    spacing = np.array(props["spacing"], dtype=np.float64)
    return arr, spacing


def resample_to_target(
    image: np.ndarray,
    spacing: np.ndarray,
    target_spacing: float,
    is_label: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resample image/label to isotropic target_spacing mm.

    Uses spline interpolation (order=3) for images and nearest-neighbour
    (order=0) for label maps to preserve integer class IDs.

    Returns (resampled_array, new_spacing).
    """
    zoom_factors = spacing / target_spacing
    # Skip resampling if already close to target (within 20%)
    if np.all(np.abs(zoom_factors - 1.0) < 0.2):
        return image, spacing
    order = 0 if is_label else 3
    resampled = zoom(image.astype(np.float32), zoom_factors, order=order)
    if is_label:
        resampled = np.round(resampled).astype(np.uint8)
    new_spacing = np.full(3, target_spacing, dtype=np.float64)
    return resampled, new_spacing


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(
    predictor: VoxTellPredictor,
    image: np.ndarray,
    class_ids: List[int],
) -> np.ndarray:
    """
    Run VoxTell on a single image and return a multi-class label map.

    Returns uint8 array of shape (H, W, D) with voxel values = class_ids.
    """
    prompts     = [AMOS_LABELS[cid] for cid in class_ids]
    image_input = image[None].astype(np.float32)   # add channel dim → (1, H, W, D)

    # predict_single_image returns (N_prompts, H, W, D) binary uint8
    seg = predictor.predict_single_image(image_input, prompts)

    pred_map = np.zeros(seg.shape[1:], dtype=np.uint8)
    for i, cid in enumerate(class_ids):
        pred_map[seg[i] > 0] = cid
    return pred_map


# ── Main evaluation ────────────────────────────────────────────────────────────

def run_evaluation(
    imgs_dir: Path,
    gts_dir:  Path,
    model_dir: str,
    n_cases: int,
    seed: int,
    tolerance_mm: float,
) -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    if device.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")

    # ── Match image files to ground truth ─────────────────────────────────────
    all_imgs = sorted(imgs_dir.glob("*.nii.gz"))
    if not all_imgs:
        raise FileNotFoundError(f"No .nii.gz files found in: {imgs_dir}")

    # Keep only cases that have a matching ground truth
    # Images use nnunetv2 naming: amos_XXXX_0000.nii.gz → label: amos_XXXX.nii.gz
    def gt_name(img_path: Path) -> str:
        return img_path.name.replace("_0000.nii.gz", ".nii.gz")

    paired = [(img, gts_dir / gt_name(img)) for img in all_imgs if (gts_dir / gt_name(img)).exists()]
    if not paired:
        raise FileNotFoundError(
            f"No matching ground truth files found.\n"
            f"  Images:  {imgs_dir}\n"
            f"  Labels:  {gts_dir}\n"
            f"  Example image name: {all_imgs[0].name}\n"
            f"  Make sure image and label filenames match exactly."
        )

    random.seed(seed)
    selected = random.sample(paired, min(n_cases, len(paired)))
    print(f"\nEvaluating {len(selected)} randomly selected cases (seed={seed}):")
    for img_p, _ in selected:
        print(f"  {img_p.name}")

    # ── Load one config at a time to avoid VRAM pressure ─────────────────────
    # Pre-load all images and ground truths into RAM
    cases_data = []
    for img_path, gt_path in selected:
        image,   spacing = load_nifti(img_path)
        gt_full, _       = load_nifti(gt_path)
        gt_full = gt_full.astype(np.uint8)

        # Resample to TARGET_SPACING_MM so patch count stays manageable
        orig_spacing = spacing.copy()
        image,   spacing = resample_to_target(image,   orig_spacing, TARGET_SPACING_MM, is_label=False)
        gt_full, _       = resample_to_target(gt_full, orig_spacing, TARGET_SPACING_MM, is_label=True)

        present_ids = [cid for cid in AMOS_LABELS if (gt_full == cid).any()]
        cases_data.append((img_path, image, gt_full, spacing, present_ids))
        print(f"  Loaded: {img_path.name}  shape={image.shape}  "
              f"spacing={np.round(spacing,2)} mm")

    records = []

    for config_name, tile_step in [("v0 (step=0.5)", 0.5), ("v3 (step=0.75)", 0.75)]:
        print(f"\n{'='*70}")
        print(f"Running {config_name} ...")
        predictor = VoxTellPredictor(model_dir=model_dir, device=device)
        predictor.tile_step_size = tile_step

        for img_path, image, gt_full, spacing, present_ids in cases_data:
            print(f"\n{'─'*70}")
            print(f"  Case   : {img_path.name}")
            print(f"  Organs : {[AMOS_LABELS[c] for c in present_ids]}")

            t0      = time.perf_counter()
            pred_map = run_inference(predictor, image, present_ids)
            elapsed  = time.perf_counter() - t0

            metrics  = compute_case_metrics(gt_full, pred_map, spacing, present_ids, tolerance_mm)
            dsc_vals = [m["dsc"] for m in metrics.values()]
            nsd_vals = [m["nsd"] for m in metrics.values()]
            mean_dsc = float(np.mean(dsc_vals)) if dsc_vals else float("nan")
            mean_nsd = float(np.mean(nsd_vals)) if nsd_vals else float("nan")

            print(f"  [{config_name}]  Mean DSC={mean_dsc:.4f}  Mean NSD={mean_nsd:.4f}  ({elapsed:.1f}s)")

            for cid, m in metrics.items():
                records.append({
                    "case":     img_path.stem.replace(".nii", ""),
                    "config":   config_name,
                    "class_id": cid,
                    "organ":    AMOS_LABELS[cid],
                    "dsc":      round(m["dsc"], 4),
                    "nsd":      round(m["nsd"], 4),
                    "time_s":   round(elapsed, 2),
                })

        # Unload model and free VRAM before loading next config
        del predictor
        torch.cuda.empty_cache()

    # ── Summary ────────────────────────────────────────────────────────────────
    if not records:
        print("\nNo results to report.")
        return

    df = pd.DataFrame(records)

    print("\n" + "=" * 70)
    print("ACCURACY COMPARISON SUMMARY")
    print("=" * 70)

    # Per-organ mean across cases
    pivot = (
        df.groupby(["organ", "config"])[["dsc", "nsd"]]
        .mean()
        .round(4)
        .reset_index()
    )
    print("\nPer-organ mean DSC / NSD (averaged across all evaluated cases):\n")
    print(pivot.to_string(index=False))

    # Overall mean per config
    print("\nOverall mean across all organs and cases:")
    overall = df.groupby("config")[["dsc", "nsd"]].mean().round(4)
    print(overall.to_string())

    # Delta (v3 − v0)
    print("\nDifference (v3 − v0)  [positive = v3 better, negative = v3 worse]:")
    v0_vals = df[df["config"] == "v0 (step=0.5)"].groupby("organ")[["dsc", "nsd"]].mean()
    v3_vals = df[df["config"] == "v3 (step=0.75)"].groupby("organ")[["dsc", "nsd"]].mean()
    delta   = (v3_vals - v0_vals).round(4)
    print(delta.to_string())

    overall_dsc_v0 = float(overall.loc["v0 (step=0.5)",  "dsc"])
    overall_dsc_v3 = float(overall.loc["v3 (step=0.75)", "dsc"])
    overall_nsd_v0 = float(overall.loc["v0 (step=0.5)",  "nsd"])
    overall_nsd_v3 = float(overall.loc["v3 (step=0.75)", "nsd"])
    delta_dsc = overall_dsc_v3 - overall_dsc_v0
    delta_nsd = overall_nsd_v3 - overall_nsd_v0

    print(f"\n  Overall ΔDSC = {delta_dsc:+.4f}")
    print(f"  Overall ΔNSD = {delta_nsd:+.4f}")

    threshold = 0.02
    dsc_ok = abs(delta_dsc) < threshold
    nsd_ok = abs(delta_nsd) < threshold
    print(f"\n  Accuracy preserved (|Δ| < {threshold:.0%}):")
    print(f"    DSC : {'YES ✓' if dsc_ok else 'NO — consider reverting tile_step_size'}")
    print(f"    NSD : {'YES ✓' if nsd_ok else 'NO — consider reverting tile_step_size'}")

    out_csv = Path("accuracy_results.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nFull per-organ results saved to: {out_csv.resolve()}")
    print("=" * 70)


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate VoxTell DSC/NSD accuracy before and after optimization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--imgs",   required=True, type=Path,
                   help="Folder containing image .nii.gz files (e.g. imagesTs/)")
    p.add_argument("--gts",    required=True, type=Path,
                   help="Folder containing ground truth .nii.gz files (e.g. labelsTs/)")
    p.add_argument("--model",  required=True, type=str,
                   help="Path to VoxTell model directory (contains plans.json and fold_0/)")
    p.add_argument("--cases",  default=5,     type=int,
                   help="Number of cases to randomly sample")
    p.add_argument("--seed",   default=42,    type=int,
                   help="Random seed for reproducible case selection")
    p.add_argument("--tol",    default=2.0,   type=float,
                   help="Surface distance tolerance in mm for NSD (2 mm = abdominal standard)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_evaluation(
        imgs_dir    = args.imgs,
        gts_dir     = args.gts,
        model_dir   = args.model,
        n_cases     = args.cases,
        seed        = args.seed,
        tolerance_mm= args.tol,
    )
