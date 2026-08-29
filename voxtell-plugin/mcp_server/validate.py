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
ADJACENT = {
    ("thorax", "abdomen"), ("abdomen", "thorax"),
    ("abdomen", "pelvis"), ("pelvis", "abdomen"),
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

    if f.modality == "CT" and f.air_fraction is not None:
        if f.air_fraction > 0.08:
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


def check(arr, prompt: str, spacing=None, filename: str = "") -> Verdict:
    img = describe_image(arr, spacing, filename)
    p_region, terms = prompt_region(prompt)

    # Unknown on either side is not grounds to refuse — it is grounds to proceed
    # with a note. A validator that blocks what it cannot classify is useless.
    if p_region == "unknown":
        return Verdict(True, f"No recognised anatomy in the prompt; proceeding without a region check.",
                       img, p_region, terms)
    if img.region == "unknown":
        return Verdict(True, f"Could not determine the body region from the image; proceeding without a region check.",
                       img, p_region, terms)

    if p_region == img.region:
        return Verdict(True, f"Prompt targets the {p_region}, and the image looks like a {img.region} study.",
                       img, p_region, terms)

    if (p_region, img.region) in ADJACENT:
        return Verdict(True,
                       f"Prompt targets the {p_region}; the image reads as {img.region}. "
                       f"These regions often share a field of view, so proceeding — check the output covers the target.",
                       img, p_region, terms)

    return Verdict(False,
                   f"The prompt asks for a structure in the {p_region} "
                   f"({', '.join(terms[:3])}), but the image is a {img.modality} study of the {img.region}. "
                   f"That structure is not in this field of view, so any mask returned would be meaningless.",
                   img, p_region, terms)
