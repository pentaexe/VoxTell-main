"""
Request validation for medical image segmentation.

The problem this solves: VoxTell is text-prompted, so it will accept "brain
tumor" against an abdominal CT and return *something* — a plausible-looking
mask of nothing. Nothing in the model refuses. The cost is a wasted GPU job and,
worse, a result that looks like an answer.

This module inspects the image and the prompt independently, then reports
whether they are compatible. It refuses only on a clear mismatch, and says what
it inferred and how confident it is, so a wrong refusal is visible rather than
mysterious.

Everything here is deliberately simple and inspectable. Intensity statistics and
geometry, not a classifier — a validator nobody can audit is a validator nobody
should trust.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Optional


# ── Anatomy vocabulary ────────────────────────────────────────────────────────
# Maps prompt terms to the body region they live in. Deliberately incomplete:
# an unrecognised term yields "unknown", which never blocks a request.

REGION_TERMS = {
    "brain": [
        "brain", "cerebr", "cerebell", "hippocamp", "ventricle", "white matter",
        "grey matter", "gray matter", "cortex", "thalamus", "putamen", "glioma",
        "glioblastoma", "meningioma", "brain tumor", "brain tumour", "skull",
        "corpus callosum", "brainstem", "amygdala", "pituitary",
    ],
    "abdomen": [
        "liver", "spleen", "kidney", "renal", "pancreas", "gallbladder",
        "stomach", "duodenum", "colon", "bowel", "intestine", "adrenal",
        "aorta", "inferior vena cava", "portal vein", "bladder", "prostate",
        "uterus", "hepatic", "splenic",
    ],
    "thorax": [
        "lung", "pulmonary", "heart", "cardiac", "myocard", "atrium",
        "ventricle of the heart", "trachea", "bronch", "esophagus",
        "oesophagus", "rib", "sternum", "pleural", "mediastin",
    ],
    "pelvis": [
        "pelvis", "pelvic", "femur", "hip", "sacrum", "rectum", "ovary",
    ],
}

# Regions that can plausibly co-occur in one field of view.
# "torso" is a scan spanning several of these at once, so it is adjacent to all
# of them — a prompt for any abdominal or thoracic structure is fine in one.
ADJACENT = {
    ("thorax", "abdomen"), ("abdomen", "thorax"),
    ("abdomen", "pelvis"), ("pelvis", "abdomen"),
    ("thorax", "torso"), ("abdomen", "torso"), ("pelvis", "torso"),
    ("torso", "thorax"), ("torso", "abdomen"), ("torso", "pelvis"),
}


@dataclass
class ImageFacts:
    modality: str = "unknown"          # "CT" | "MR" | "unknown"
    region: str = "unknown"            # brain | abdomen | thorax | pelvis | unknown
    shape: tuple = ()
    spacing: Optional[tuple] = None
    hu_min: Optional[float] = None
    hu_max: Optional[float] = None
    air_fraction: Optional[float] = None
    notes: list = field(default_factory=list)


def describe_image(arr, spacing=None, filename: str = "") -> ImageFacts:
    """Infer modality and body region from intensities and geometry."""
    import numpy as np

    f = ImageFacts(shape=tuple(int(s) for s in arr.shape), spacing=spacing)

    # Sample rather than scan the whole volume; these are 10^7-voxel arrays.
    flat = arr.reshape(-1)
    if flat.size > 2_000_000:
        flat = flat[:: max(1, flat.size // 2_000_000)]
    flat = flat.astype("float32")

    f.hu_min = float(np.percentile(flat, 0.5))
    f.hu_max = float(np.percentile(flat, 99.5))

    # ── Modality ─────────────────────────────────────────────────────────────
    # CT is calibrated in Hounsfield units: air is about -1000, water 0, bone
    # several hundred positive. MR intensities are arbitrary and rarely negative.
    if f.hu_min < -300:
        f.modality = "CT"
        f.notes.append(f"intensities reach {f.hu_min:.0f}, consistent with Hounsfield units (air ~ -1000)")
    elif f.hu_min >= -20 and f.hu_max > 0:
        f.modality = "MR"
        f.notes.append(f"no strongly negative intensities (min ~ {f.hu_min:.0f}); not calibrated like CT")
    else:
        f.notes.append(f"intensity range [{f.hu_min:.0f}, {f.hu_max:.0f}] is not a clear CT or MR signature")

    # Air fraction separates a head (no internal air to speak of) from a torso
    # (lungs, or bowel gas).
    if f.modality == "CT":
        f.air_fraction = float((flat < -700).mean())
        f.notes.append(f"{f.air_fraction*100:.1f}% of voxels below -700 HU (air)")

    # ── Region ───────────────────────────────────────────────────────────────
    # Geometry first. Brain studies are near-isotropic and modest in-plane;
    # torso CT is conventionally 512x512 with thicker slices.
    inplane = max(f.shape[-2:]) if len(f.shape) >= 2 else 0

    # z-extent separates a targeted study from a scan covering several regions.
    # A chest-only CT is roughly 300 mm; anything much past that is spanning
    # thorax and abdomen together, and calling it either one alone is wrong.
    z_mm = None
    if spacing and len(f.shape) >= 3:
        try:
            z_mm = max(s * z for s, z in zip(f.shape[-3:], spacing[-3:]))
        except Exception:
            z_mm = None

    if f.modality == "CT" and f.air_fraction is not None:
        if f.air_fraction > 0.08:
            if z_mm and z_mm > 400:
                f.region = "torso"
                f.notes.append(f"{z_mm:.0f} mm of coverage spans thorax and abdomen together")
            else:
                f.region = "thorax" if f.air_fraction > 0.20 else "abdomen"
                f.notes.append("substantial internal air implies a torso field of view")
        elif f.air_fraction < 0.02 and inplane <= 320:
            f.region = "brain"
            f.notes.append("almost no internal air and a compact field of view implies a head")
        else:
            f.region = "abdomen"
            f.notes.append("low air fraction with a large field of view; abdomen is the most likely torso region")
    elif f.modality == "MR":
        if inplane <= 320:
            f.region = "brain"
            f.notes.append("MR with a compact field of view; brain is the most common such study")

    # Filename is a weak hint, used only to break a tie the pixels left open.
    low = filename.lower()
    if f.region == "unknown":
        for region, terms in REGION_TERMS.items():
            if any(t.replace(" ", "") in low.replace("_", "").replace("-", "") for t in terms[:4]):
                f.region = region
                f.notes.append(f"region taken from the filename, not the image data")
                break

    return f


def prompt_region(prompt: str) -> tuple[str, list[str]]:
    """Which body region does this prompt refer to? Returns (region, matched terms)."""
    low = prompt.lower()
    hits = {}
    for region, terms in REGION_TERMS.items():
        matched = [t for t in terms if re.search(r"\b" + re.escape(t), low)]
        if matched:
            hits[region] = matched
    if not hits:
        return "unknown", []
    # Prefer the region with the most specific match.
    best = max(hits, key=lambda r: max(len(t) for t in hits[r]))
    return best, hits[best]


@dataclass
class Verdict:
    allowed: bool
    reason: str
    image: ImageFacts
    prompt_region: str
    matched_terms: list
    caveats: list = field(default_factory=list)


# Slice thickness at which the measurements below were taken. VoxTell v1.1 is a
# no-resampling model — its plans.json says
# "nnUNetResEncUNetLPlans_noResampling_3d_fullres", and the inference path does
# not resample either — so the network sees voxels at whatever spacing the scan
# was acquired at, through a fixed 192^3 patch. At 1 mm that patch covers about
# 192 mm of anatomy; at 5 mm it covers 960 mm, longer than a torso. A 15 mm
# lesion is 15 slices at 1 mm and 3 at 5 mm, so boundary partial-volume dominates.
COARSE_SPACING_MM = 3.0


def spacing_caveat(spacing) -> str:
    """Warn about acquisition geometry the model is known to handle poorly.

    Not grounds to refuse: the mask is still worth having, and on a large target
    it may be fine. It is grounds to say so before the number is quoted.
    """
    if not spacing:
        return ""
    try:
        coarsest = max(float(s) for s in spacing)
    except (TypeError, ValueError):
        return ""
    if coarsest < COARSE_SPACING_MM:
        return ""
    return (
        f"Coarsest voxel spacing is {coarsest:.1f} mm. VoxTell v1.1 does not resample, "
        "so accuracy falls off on thick-slice scans: measured DSC 0.48 and 0.52 on two "
        "5 mm liver-tumour cases against 0.86-0.87 on three at 0.8-1.0 mm (n=5). "
        "Where the falloff begins between 1 mm and 5 mm has not been measured. "
        "Expect a usable mask on a large target and an unreliable one on a small lesion."
    )


def check(arr, prompt: str, spacing=None, filename: str = "") -> Verdict:
    img = describe_image(arr, spacing, filename)
    p_region, terms = prompt_region(prompt)
    caveats = [c for c in (spacing_caveat(spacing),) if c]

    def verdict(allowed: bool, reason: str) -> Verdict:
        return Verdict(allowed, reason, img, p_region, terms, caveats)

    # Unknown on either side is not grounds to refuse — it is grounds to proceed
    # with a note. A validator that blocks what it cannot classify is useless.
    if p_region == "unknown":
        return verdict(True, "No recognised anatomy in the prompt; proceeding without a region check.")
    if img.region == "unknown":
        return verdict(True, "Could not determine the body region from the image; proceeding without a region check.")

    if p_region == img.region:
        return verdict(True, f"Prompt targets the {p_region}, and the image looks like a {img.region} study.")

    if (p_region, img.region) in ADJACENT:
        return verdict(True,
                       f"Prompt targets the {p_region}; the image reads as {img.region}. "
                       f"These regions often share a field of view, so proceeding — check the output covers the target.")

    return verdict(False,
                   f"The prompt asks for a structure in the {p_region} "
                   f"({', '.join(terms[:3])}), but the image is a {img.modality} study of the {img.region}. "
                   f"That structure is not in this field of view, so any mask returned would be meaningless.")
