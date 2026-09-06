#!/usr/bin/env python3
"""
MCP server exposing VoxTell and nnInteractive as callable segmentation tools.

Speaks JSON-RPC 2.0 over stdio — the MCP wire protocol — with no SDK dependency,
so it runs under any Python that can import nibabel and numpy. Keeping the
protocol visible also makes this readable as a worked example of what MCP is.

Tools:
  check_request        validate an image/prompt pair without spending any compute
  voxtell_segment      text-prompted segmentation; validates first, refuses on mismatch
  nninteractive_segment bbox-prompted segmentation
  list_models          what is available and what each model can and cannot do

The point of check_request, and of the validation gate inside voxtell_segment:
VoxTell will accept "brain tumor" against an abdominal CT and return a mask of
nothing. The model does not refuse. This server does.
"""

from __future__ import annotations
import json
import os
import sys
import traceback
from pathlib import Path

# stdout IS the JSON-RPC channel on stdio, so anything else written there
# corrupts the stream. VoxTell's predictor announces its quantization mode with
# a plain print(); that alone is enough to break a strict client mid-request.
# Take a private handle for protocol writes before anything else can touch the
# stream, then point sys.stdout at stderr, so library output still reaches the
# logs but can never land on the wire. Do this before any heavy import, since an
# import-time banner corrupts the stream just as thoroughly as one from inference.
_PROTOCOL_OUT = sys.stdout
sys.stdout = sys.stderr

sys.path.insert(0, str(Path(__file__).parent))
import validate as V  # noqa: E402

# A Windows console defaults to cp1252, which cannot encode the em dashes and
# arrows below. Without this the very first print raises UnicodeEncodeError and
# the script produces no output at all — a confusing failure for something whose
# whole job is to explain what is wrong. Degrade the characters, not the run.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


# ── Use the optimized voxtell, not the stock one from PyPI ────────────────────
# This plugin lives inside the VoxTell fork whose predictor carries the INT4
# backbone, the embedding cache, Numba preprocessing and tile_step=0.75. None of
# those are in the DKFZ package on PyPI. If `import voxtell` resolves to
# site-packages, the tools run and return plausible masks with none of the
# optimizations — a silent downgrade, which is exactly the failure this plugin
# exists to prevent elsewhere. So put the repo first on sys.path when we can
# find it, and report which copy is live either way.

def _find_repo_voxtell() -> Path | None:
    for parent in Path(__file__).resolve().parents[:4]:
        if (parent / "voxtell" / "__init__.py").exists():
            return parent
    return None


REPO_ROOT = _find_repo_voxtell()
if REPO_ROOT is not None:
    sys.path.insert(0, str(REPO_ROOT))


# Markers of the optimized fork. These are additions to this repo's predictor,
# absent from the DKFZ package on PyPI. Detect them directly rather than by file
# path: an editable install can live anywhere, and a marketplace install may
# place the plugin outside the checkout, in which case path matching says
# "unrecognised" about a copy that is perfectly good.
OPT_MARKERS = [
    ("voxtell.inference.predictor", "_prompt_cache_path", "embedding cache"),
    ("voxtell.inference.predictor", "_load_text_backbone", "INT4 text backbone"),
    ("voxtell.utils.fast_preprocess", "numba_crop_to_nonzero", "Numba preprocessing"),
]


def voxtell_provenance() -> tuple[str, str]:
    """Return (path, kind) for the voxtell that will actually import."""
    try:
        import voxtell
    except Exception as e:
        return "", f"not importable: {type(e).__name__}"

    path = str(Path(voxtell.__file__).resolve().parent)
    # A marker can be unavailable for two very different reasons, and conflating
    # them sends people to the wrong fix. If the module imports but lacks the
    # attribute, this really is the stock build and reinstalling helps. If the
    # import itself fails, the code is very likely right and a dependency is
    # missing — reinstalling the same package changes nothing. That case is real:
    # predictor.py imports fast_preprocess, which needs numba, which nothing
    # declared for a long time, so a clean install reported itself as stock.
    absent, broken = [], {}
    for mod, attr, label in OPT_MARKERS:
        try:
            m = __import__(mod, fromlist=[attr])
            if not hasattr(m, attr):
                absent.append(label)
        except ModuleNotFoundError as e:
            broken[e.name or mod] = label
        except Exception as e:
            broken[f"{mod} ({type(e).__name__})"] = label

    if broken and not absent:
        return path, ("optimizations unreadable — cannot import: "
                      + ", ".join(sorted(broken)) + ". Install the missing package; "
                      "reinstalling voxtell will not help")
    missing = absent + [f"{k}?" for k in broken]
    if not missing:
        return path, "optimized fork"
    if len(missing) == len(OPT_MARKERS):
        return path, "STOCK build — no optimizations present"
    return path, f"PARTIAL — missing: {', '.join(missing)}"

def int4_availability() -> str:
    """Return "" if INT4 will really be used, else why it will not be.

    Feature detection on the predictor cannot catch this. The INT4 code is part
    of the fork, so a build missing only the runtime packages still looks like
    the optimized one — and _load_text_backbone catches the ImportError and
    serves FP16 anyway. The result is a correct mask produced by a slower path
    than the one the measurements describe, with nothing to say so.
    """
    missing = []
    for mod in ("bitsandbytes", "accelerate"):
        try:
            __import__(mod)
        except Exception:
            missing.append(mod)
    if not missing:
        return ""
    return (f"INT4 unavailable: {', '.join(missing)} not installed. The text backbone "
            "will silently run FP16, so it needs ~8 GB rather than ~2 GB and the "
            "measured INT4 timings do not apply.")


PROTOCOL_VERSION = "2024-11-05"


def _plugin_version() -> str:
    """Read the version from the manifest rather than repeating it here.

    Two copies of a version number drift, and the moment they do, the one a user
    reads when reporting a problem is the wrong one.
    """
    try:
        manifest = Path(__file__).resolve().parent.parent / ".claude-plugin" / "plugin.json"
        return json.loads(manifest.read_text(encoding="utf-8")).get("version", "unknown")
    except Exception:
        return "unknown"


VERSION = _plugin_version()
NNI_WEIGHTS = os.environ.get("NNINTERACTIVE_WEIGHTS", "")
OUT_DIR     = Path(os.environ.get("VOXTELL_OUTPUT_DIR",
                                  Path.home() / ".voxtell_mcp" / "segmentations"))
HF_REPO     = "mrokuss/VoxTell"
MODEL_NAME  = "voxtell_v1.1"


def resolve_model_dir(download: bool = False) -> tuple[str, str]:
    """
    Find the VoxTell checkpoint without requiring the user to configure a path.

    Order: an explicit VOXTELL_MODEL_DIR, then anywhere we have downloaded it
    before, then (only when asked) a fresh download from Hugging Face. The
    checkpoint is ~1.7 GB and is not in the git repository, so a first run on a
    new machine has to fetch it.

    Returns (path, status) where path is "" if nothing usable was found.
    """
    env = os.environ.get("VOXTELL_MODEL_DIR", "").strip()
    if env and (Path(env) / "plans.json").exists():
        return env, "configured"

    cache = Path.home() / ".cache" / "voxtell_models" / MODEL_NAME
    if (cache / "plans.json").exists():
        return str(cache), "cached"

    if not download:
        return "", "missing"

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return "", "no-hf-hub"

    cache.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=HF_REPO, allow_patterns=[f"{MODEL_NAME}/*"],
                      local_dir=str(cache.parent))
    return (str(cache), "downloaded") if (cache / "plans.json").exists() else ("", "download-failed")


MODEL_DIR, MODEL_STATUS = resolve_model_dir()


# ── Protocol plumbing ────────────────────────────────────────────────────────

def send(msg: dict) -> None:
    _PROTOCOL_OUT.write(json.dumps(msg) + "\n")
    _PROTOCOL_OUT.flush()


def reply(rid, result):
    send({"jsonrpc": "2.0", "id": rid, "result": result})


def error(rid, code, message):
    send({"jsonrpc": "2.0", "id": rid, "error": {"code": code, "message": message}})


def text_result(s: str, is_error: bool = False):
    return {"content": [{"type": "text", "text": s}], "isError": is_error}


# ── Image loading ─────────────────────────────────────────────────────────────

def load_image(path: str):
    """
    Return (array of shape (C,H,W,D), spacing).

    NIfTI goes through NibabelIOWithReorient — the same reader VoxTell was
    trained and evaluated with. Reading with plain nib.load().get_fdata()
    instead leaves the volume in its stored orientation, and the model then
    segments a fragment: measured DSC 0.18 against 0.94 for the same organ
    read correctly. It fails quietly, with a mask that looks plausible.
    """
    import numpy as np
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"image not found: {path}")

    if p.suffix == ".npz":
        d = np.load(p, allow_pickle=True)
        if "imgs" not in d.files:
            raise ValueError(f"{p.name} has no 'imgs' key (found: {list(d.files)})")
        arr = d["imgs"]
        if arr.ndim == 3:
            arr = arr[None]                       # (C,H,W,D), matching the reader
        spacing = tuple(d["spacing"].tolist()) if "spacing" in d.files else None
        return arr.astype(np.float32), spacing, None

    from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient
    arr, props = NibabelIOWithReorient().read_images([str(p)])
    spacing = tuple(float(z) for z in props.get("spacing", ())[:3]) or None
    return np.asarray(arr, dtype=np.float32), spacing, props


# ── Tools ─────────────────────────────────────────────────────────────────────

TOOLS = [
    {
        "name": "check_request",
        "description": (
            "Check whether a segmentation prompt is anatomically possible for a given image, "
            "without running any model. Returns the inferred modality and body region, what the "
            "prompt asks for, and whether they are compatible. Use this before spending GPU time, "
            "or to explain why a request was refused."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "image_path": {"type": "string", "description": "Path to a .nii/.nii.gz or CVPR .npz volume"},
                "prompt": {"type": "string", "description": "The text prompt, e.g. 'the spleen'"},
            },
            "required": ["image_path", "prompt"],
        },
    },
    {
        "name": "voxtell_segment",
        "description": (
            "Segment a structure from a 3-D medical image using a free-text prompt (VoxTell). "
            "Validates the request first and REFUSES anatomically impossible pairings — asking for "
            "a brain structure in an abdominal CT returns an error, not a meaningless mask. "
            "Set force=true only to override a refusal deliberately."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "image_path": {"type": "string"},
                "prompt": {"type": "string", "description": "What to segment, e.g. 'the liver'"},
                "force": {"type": "boolean", "description": "Run even if validation refuses. Default false.", "default": False},
                "output_dir": {"type": "string", "description": "Where to write the mask. Defaults to VOXTELL_OUTPUT_DIR."},
            },
            "required": ["image_path", "prompt"],
        },
    },
    {
        "name": "nninteractive_segment",
        "description": (
            "Segment using a 3-D bounding box prompt (nnInteractive). No text encoder, so there is "
            "no anatomy to validate — the box defines the target. Faster per object than VoxTell."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "image_path": {"type": "string"},
                "bbox": {
                    "type": "array",
                    "description": ("[[z_min,z_max],[y_min,y_max],[x_min,x_max]], max exclusive, in voxels. "
                                    "Coordinates are in the reoriented volume, whose axis order may differ "
                                    "from the file as your viewer displays it. Run check_request first to "
                                    "see the shape these coordinates are relative to."),
                    "items": {"type": "array", "items": {"type": "integer"}},
                },
                "output_dir": {"type": "string", "description": "Where to write the mask. Defaults to VOXTELL_OUTPUT_DIR."},
            },
            "required": ["image_path", "bbox"],
        },
    },
    {
        "name": "list_models",
        "description": "List available segmentation models, how each is prompted, and their known limits.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "setup",
        "description": (
            "Check this machine is ready to run VoxTell and report what is missing. "
            "Pass download=true to fetch the ~1.7 GB checkpoint from Hugging Face if it "
            "is not already present. Run this first on a new install."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "download": {
                    "type": "boolean",
                    "description": "Download the checkpoint if missing. Default false (report only).",
                    "default": False,
                }
            },
        },
    },
]


def tool_setup(args):
    global MODEL_DIR, MODEL_STATUS
    # Lead with the version and the interpreter. When someone reports that the
    # plugin misbehaves, these are the first two things worth knowing, and
    # "which copy did you actually install" is the question that has cost the
    # most time so far.
    lines, ready = [f"  plugin  {'version':<22} {VERSION}",
                    f"  python  {'interpreter':<22} {sys.executable}"], True

    def probe(mod, label, hint):
        nonlocal ready
        try:
            m = __import__(mod)
            lines.append(f"  ok      {label:<22} {getattr(m, '__version__', 'installed')}")
        except Exception:
            ready = False
            lines.append(f"  MISSING {label:<22} {hint}")

    probe("torch", "torch", "pip install torch --index-url https://download.pytorch.org/whl/cu126")

    vpath, vkind = voxtell_provenance()
    if vkind == "optimized fork":
        lines.append(f"  ok      {'voxtell build':<22} {vkind}")
    elif not vpath:
        ready = False
        lines.append(f"  MISSING {'voxtell':<22} pip install -e . from the VoxTell-main checkout")
    else:
        ready = False
        lines.append(f"  WRONG   {'voxtell build':<22} {vkind}")
        lines.append(f"          {'':<22} loaded from {vpath}")
        lines.append(f"          {'':<22} the speedups measured here are not in that copy.")
        lines.append(f"          {'':<22} Fix: pip install -e /path/to/VoxTell-main")
    probe("nibabel", "nibabel", "pip install nibabel")
    probe("nnunetv2", "nnunetv2", "pip install nnunetv2  (needed to read NIfTI in the trained orientation)")
    probe("transformers", "transformers", "pip install transformers")

    try:
        import torch
        lines.append(f"  {'ok     ' if torch.cuda.is_available() else 'warn   '} "
                     f"{'CUDA':<22} "
                     f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'not available; CPU will be very slow'}")
    except Exception:
        pass

    # Both packages, not just accelerate: BitsAndBytesConfig needs bitsandbytes
    # too, and pyproject did not declare either for a long time, so an install
    # that followed the README to the letter still served FP16.
    int4 = int4_availability()
    if int4:
        lines.append(f"  warn    {'INT4 backbone':<22} {int4.split(':', 1)[1].strip()}")
        lines.append(f"          {'':<22} pip install bitsandbytes accelerate")
    else:
        lines.append(f"  ok      {'INT4 backbone':<22} bitsandbytes and accelerate present")

    if args.get("download") and MODEL_STATUS == "missing":
        lines.append("\n  Downloading checkpoint from Hugging Face (~1.7 GB, this takes a while)...")
        MODEL_DIR, MODEL_STATUS = resolve_model_dir(download=True)

    note = {
        "configured": f"using VOXTELL_MODEL_DIR: {MODEL_DIR}",
        "cached":     f"found at {MODEL_DIR}",
        "downloaded": f"downloaded to {MODEL_DIR}",
        "missing":    f"not found. Call setup with download=true, or set VOXTELL_MODEL_DIR to a folder containing plans.json",
        "no-hf-hub":  "huggingface_hub not installed: pip install huggingface_hub",
        "download-failed": "download completed but plans.json is not where expected",
    }[MODEL_STATUS]
    ok_model = MODEL_STATUS in ("configured", "cached", "downloaded")
    lines.append(f"\n  {'ok     ' if ok_model else 'MISSING'} checkpoint             {note}")

    ready = ready and ok_model
    lines.append("")
    lines.append("READY — try: check_request on an image, then voxtell_segment"
                 if ready else "NOT READY — resolve the items above first")
    return text_result("VoxTell setup check\n\n" + "\n".join(lines), is_error=not ready)


def tool_check_request(args):
    arr, spacing, _props = load_image(args["image_path"])
    v = V.check(arr, args["prompt"], spacing, Path(args["image_path"]).name)
    lines = [
        f"Request: {'ALLOWED' if v.allowed else 'REFUSED'}",
        "",
        f"  {v.reason}",
        "",
        "Image",
        f"  modality : {v.image.modality}",
        f"  region   : {v.image.region}",
        # Report the volume only: the leading channel axis is an artifact of how
        # the reader hands the array over, not something the caller chose. And
        # spacing rounds to two decimals — 0.8203120231628418 mm is float noise
        # dressed up as precision.
        f"  shape    : {tuple(v.image.shape[-3:])}"
        + (f"   spacing: {tuple(round(float(x), 2) for x in v.image.spacing)}"
           if v.image.spacing else ""),
    ]
    for n in v.image.notes:
        lines.append(f"  - {n}")
    lines += ["", "Prompt", f"  region  : {v.prompt_region}"]
    if v.matched_terms:
        lines.append(f"  matched : {', '.join(v.matched_terms[:5])}")
    return text_result("\n".join(lines), is_error=not v.allowed)


def tool_voxtell_segment(args):
    image_path = args["image_path"]
    prompt = args["prompt"]
    force = bool(args.get("force", False))

    arr, spacing, props = load_image(image_path)
    v = V.check(arr, prompt, spacing, Path(image_path).name)

    if not v.allowed and not force:
        return text_result(
            "REFUSED — no compute was spent.\n\n"
            f"{v.reason}\n\n"
            f"Image reads as: {v.image.modality} of the {v.image.region}.\n"
            f"Prompt targets: {v.prompt_region}.\n\n"
            "If this is deliberate — testing behaviour on out-of-distribution input, say — "
            "call again with force=true. The mask will not be meaningful.",
            is_error=True,
        )

    # Refuse to run on a build without the optimizations. It would segment and
    # return a plausible mask, so nothing downstream would reveal that the thing
    # this plugin exists to deliver is absent. Someone who skips the setup tool
    # should still not get a silent downgrade.
    vpath, vkind = voxtell_provenance()
    if vkind != "optimized fork":
        return text_result(
            f"REFUSED — the loaded VoxTell is not the optimized build.\n\n"
            f"  build : {vkind}\n"
            f"  from  : {vpath}\n\n"
            "This copy would segment and return a plausible mask with none of the\n"
            "measured optimizations, which is the kind of silent downgrade this\n"
            "plugin exists to prevent.\n\n"
            "Fix: pip install -e /path/to/VoxTell-main into the interpreter named by\n"
            "voxtell_python, then run the setup tool to confirm.",
            is_error=True,
        )

    if not MODEL_DIR or not Path(MODEL_DIR).exists():
        return text_result(
            f"Validation passed ({v.reason})\n\n"
            "Cannot run inference: the VoxTell checkpoint was not found.\n"
            "Run the setup tool with download=true to fetch it (~1.7 GB), or point "
            "VOXTELL_MODEL_DIR at a folder containing plans.json.",
            is_error=True,
        )

    import numpy as np
    import torch
    from voxtell.inference.predictor import VoxTellPredictor

    out_dir = Path(args["output_dir"]) if args.get("output_dir") else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    predictor = VoxTellPredictor(model_dir=MODEL_DIR, device=device)
    raw = arr[None].astype(np.float32) if arr.ndim == 3 else arr.astype(np.float32)
    data, bbox, orig_shape = predictor.preprocess(raw)
    emb = predictor.embed_text_prompts([prompt])
    logits = predictor.predict_sliding_window_return_logits(data, emb)

    # preprocess() crops to the non-zero bounding box, so the logits are in
    # cropped space. Put them back into the full volume before saving, or the
    # mask silently comes out a different shape than the input.
    from acvl_utils.cropping_and_padding.bounding_boxes import insert_crop_into_image
    binary = (torch.sigmoid(logits.float().cpu()) > 0.5)
    seg = np.zeros([binary.shape[0], *orig_shape], dtype=np.uint8)
    seg = insert_crop_into_image(seg, binary, bbox)
    n_vox = int(seg.sum())

    stem = f"{Path(image_path).name.split('.')[0]}__{prompt.replace(' ', '_')[:40]}"
    if props is not None:
        # Write through the same reader, which undoes the reorientation. The mask
        # then overlays on the file the user passed in, rather than sitting in
        # the model's internal axis order.
        from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient
        out = out_dir / f"{stem}.nii.gz"
        NibabelIOWithReorient().write_seg(seg[0].astype(np.uint8), str(out), props)
        note = "aligned to the input image"
    else:
        out = out_dir / f"{stem}.npz"
        np.savez_compressed(out, mask=seg.astype(np.uint8))
        note = "npz input: mask is in the array's own axis order"

    warn = ""
    if force and not v.allowed:
        warn = "\n\nNOTE: validation was overridden with force=true. Treat this mask as meaningless.\n"
    if n_vox == 0:
        warn += "\nThe mask is empty — the model found nothing matching this prompt."
    # The mask is still correct on FP16; what is wrong is any speed claim made
    # about the run. Say so here rather than only in setup, which a caller who
    # went straight to segmenting will never have seen.
    int4 = int4_availability()
    if int4:
        warn += f"\n\nNOTE: {int4}"

    return text_result(
        f"Segmented '{prompt}'\n"
        f"  device      : {device}\n"
        f"  image       : {Path(image_path).name}  {tuple(arr.shape[-3:])}\n"
        f"  validation  : {v.reason}\n"
        f"  voxels      : {n_vox:,}\n"
        f"  saved       : {out}  ({note})"
        + warn
    )


def validate_bbox(bbox, volume) -> str:
    """Return an error message for an unusable bounding box, or "" if it is fine.

    Checked here rather than left to nnInteractive, which fails deep inside a
    zoom-out step with an error that says nothing about the box. The common
    mistakes are a box in the wrong axis order and a box read off a viewer that
    shows the file's original orientation rather than the reoriented volume.
    """
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 3:
        return (f"bbox must be three [min, max] pairs, one per axis, got {bbox!r}.\n"
                f"This image is {tuple(volume)} after reorientation.")
    for i, pair in enumerate(bbox):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            return f"bbox axis {i} must be a [min, max] pair, got {pair!r}."
        try:
            lo, hi = int(pair[0]), int(pair[1])
        except (TypeError, ValueError):
            return f"bbox axis {i} must be two integers, got {pair!r}."
        if lo >= hi:
            return (f"bbox axis {i} is empty: min {lo} is not below max {hi} "
                    "(max is exclusive).")
        if lo < 0 or hi > volume[i]:
            return (f"bbox axis {i} spans [{lo}, {hi}) but the image is only "
                    f"{volume[i]} voxels on that axis.\n"
                    f"Full extent is {tuple(volume)}. Note this is the volume after "
                    "reorientation, which may not match the axis order your viewer shows.")
    return ""


def tool_nninteractive_segment(args):
    if not NNI_WEIGHTS or not Path(NNI_WEIGHTS).exists():
        return text_result(
            f"nnInteractive weights not configured or missing ({NNI_WEIGHTS!r}). "
            "Set nninteractive_weights in the plugin's user config.",
            is_error=True,
        )
    import numpy as np
    import torch

    arr, _spacing, props = load_image(args["image_path"])
    # load_image always returns (C,H,W,D). nnInteractive wants the image with
    # that channel axis and the target buffer WITHOUT it — see the verified
    # cluster script, which passes image[None] because its npz array is bare 3-D.
    # Feeding arr[None] here instead produced a 5-D image and a 4-D buffer, and
    # this tool could never have run.
    volume = arr.shape[-3:]

    # Check the box before the hardware. Validation costs nothing and needs no
    # device, and a caller with a malformed box should hear about the box rather
    # than about a GPU they may not need to care about yet.
    bad = validate_bbox(args.get("bbox"), volume)
    if bad:
        return text_result(bad, is_error=True)
    bbox = [[int(a), int(b)] for a, b in args["bbox"]]

    if not torch.cuda.is_available():
        return text_result(
            "nnInteractive needs a CUDA GPU and none is visible to this interpreter.\n"
            "Unlike VoxTell there is no usable CPU path here: the session allocates "
            "pinned memory and expects a device.",
            is_error=True,
        )

    from nnInteractive.inference.inference_session import nnInteractiveInferenceSession

    session = nnInteractiveInferenceSession(
        device=torch.device("cuda", 0),
        use_torch_compile=False,
        verbose=False,
        torch_n_threads=os.cpu_count(),   # SLURM_CPUS_PER_TASK hangs with fold='all'
        do_autozoom=True,
        use_pinned_memory=True,
    )
    # fold='all' is not a default worth changing: fold=0 scores about 0.33 DSC.
    session.initialize_from_trained_model_folder(NNI_WEIGHTS, use_fold="all")

    buf = torch.zeros(volume, dtype=torch.uint8, device="cpu")
    session.set_image(arr.astype(np.float32))
    session.set_target_buffer(buf)
    session.reset_interactions()
    session.add_bbox_interaction(bbox, include_interaction=True, run_prediction=False)
    session.new_interaction_centers = [session.new_interaction_centers[-1]]
    session.new_interaction_zoom_out_factors = [session.new_interaction_zoom_out_factors[-1]]
    session._predict()

    seg = buf.numpy().astype(np.uint8)
    out_dir = Path(args["output_dir"]) if args.get("output_dir") else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{Path(args['image_path']).name.split('.')[0]}__bbox"

    # Write back through the same reader, as the VoxTell path does. Saving the
    # raw buffer instead leaves the mask in the reader's internal axis order, so
    # it will not overlay on the file the caller passed in.
    if props is not None:
        from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient
        out = out_dir / f"{stem}.nii.gz"
        NibabelIOWithReorient().write_seg(seg, str(out), props)
        note = "aligned to the input image"
    else:
        out = out_dir / f"{stem}.npz"
        np.savez_compressed(out, mask=seg)
        note = "npz input: mask is in the array's own axis order"

    return text_result(
        f"Segmented from bounding box {bbox}\n"
        f"  image  : {Path(args['image_path']).name}  {tuple(volume)}\n"
        f"  voxels : {int((seg > 0).sum()):,}\n"
        f"  saved  : {out}  ({note})"
    )


def tool_list_models(args):
    return text_result(
        f"voxtell build : {voxtell_provenance()[1]}\n\n"
        "VoxTell — text-prompted 3-D segmentation\n"
        "  prompt      : free text, e.g. 'the spleen'\n"
        "  encoder     : Qwen3-Embedding-4B, INT4 (NF4) by default\n"
        "  limits      : segments what the prompt names IF it is in the field of view. It does not\n"
        "                verify that itself, which is why check_request exists.\n"
        "  measured    : 2.7x faster after optimization on abdominal CT (n=4), DSC +0.0003\n"
        f"  checkpoint  : {MODEL_DIR or 'NOT CONFIGURED'}\n"
        "\n"
        "nnInteractive — bounding-box prompted 3-D segmentation\n"
        "  prompt      : [[z0,z1],[y0,y1],[x0,x1]]\n"
        "  limits      : must use fold='all'; fold=0 gives ~0.33 DSC and is not a valid baseline\n"
        "  measured    : 1.33x with torch.compile (n=4), DSC +0.0002\n"
        f"  weights     : {NNI_WEIGHTS or 'NOT CONFIGURED'}\n"
    )


HANDLERS = {
    "check_request": tool_check_request,
    "voxtell_segment": tool_voxtell_segment,
    "nninteractive_segment": tool_nninteractive_segment,
    "list_models": tool_list_models,
    "setup": tool_setup,
}


# ── Dispatch, shared by both transports ───────────────────────────────────────
# MCP's protocol semantics are identical on every transport; a transport only
# decides how messages are framed and delivered. So the routing below is written
# once and both stdio and HTTP call into it.

def handle(req: dict):
    """Return a response dict, or None for a notification (which gets no reply)."""
    method, rid = req.get("method"), req.get("id")

    def ok(result):
        return {"jsonrpc": "2.0", "id": rid, "result": result}

    if method == "initialize":
        return ok({
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "voxtell-seg", "version": VERSION},
        })
    if method == "notifications/initialized":
        return None
    if method == "tools/list":
        return ok({"tools": TOOLS})
    if method == "ping":
        return ok({})
    if method == "tools/call":
        params = req.get("params", {})
        name = params.get("name")
        handler = HANDLERS.get(name)
        if handler is None:
            return {"jsonrpc": "2.0", "id": rid,
                    "error": {"code": -32601, "message": f"unknown tool: {name}"}}
        try:
            return ok(handler(params.get("arguments", {})))
        except Exception as e:
            return ok(text_result(
                f"{type(e).__name__}: {e}\n\n{traceback.format_exc(limit=3)}", is_error=True))
    if rid is not None:
        return {"jsonrpc": "2.0", "id": rid,
                "error": {"code": -32601, "message": f"unknown method: {method}"}}
    return None


# ── Transport: stdio ──────────────────────────────────────────────────────────
# Newline-delimited JSON-RPC over the standard streams of a client-launched
# subprocess. This is what Claude Code and Claude Desktop use.

def serve_stdio():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue
        resp = handle(req)
        if resp is not None:
            send(resp)


# ── Transport: Streamable HTTP ────────────────────────────────────────────────
# Each message is an HTTP POST to a single endpoint. A web client such as
# ChatGPT cannot launch a subprocess on your machine, so stdio is unavailable to
# it and this is the binding it needs.

def serve_http(host: str, port: int):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):        # keep stdout clean
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

        def _send(self, code: int, body: bytes = b"", ctype="application/json"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if body:
                self.wfile.write(body)

        def do_GET(self):
            # Convenience for humans checking the server is up. The spec's GET
            # is for opening an SSE stream, which these tools do not need.
            if self.path.rstrip("/") in ("", "/mcp", "/health"):
                self._send(200, json.dumps({
                    "server": "voxtell-seg",
                    "transport": "streamable-http",
                    "endpoint": "/mcp",
                    "tools": [t["name"] for t in TOOLS],
                }).encode())
            else:
                self._send(404, b'{"error":"not found"}')

        def do_POST(self):
            if self.path.rstrip("/") not in ("", "/mcp"):
                self._send(404, b'{"error":"not found"}')
                return
            n = int(self.headers.get("Content-Length", 0))
            try:
                req = json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError:
                self._send(400, json.dumps({
                    "jsonrpc": "2.0", "id": None,
                    "error": {"code": -32700, "message": "parse error"}}).encode())
                return

            # A batch is a JSON array; a single message is an object.
            if isinstance(req, list):
                out = [r for r in (handle(m) for m in req) if r is not None]
                self._send(200 if out else 202, json.dumps(out).encode() if out else b"")
                return

            resp = handle(req)
            if resp is None:
                self._send(202)                    # notification: accepted, no body
            else:
                self._send(200, json.dumps(resp).encode())

    srv = ThreadingHTTPServer((host, port), Handler)
    sys.stderr.write(
        f"voxtell-seg MCP server on http://{host}:{port}/mcp\n"
        f"  tools: {', '.join(t['name'] for t in TOOLS)}\n"
        f"  bound to {host}; to reach it from a hosted client you need a tunnel\n"
        f"  (e.g. cloudflared / ngrok) and an auth layer in front. Do not expose\n"
        f"  this directly: it reads local files and runs GPU jobs, with no auth.\n"
    )
    srv.serve_forever()


def main():
    args = sys.argv[1:]
    if "--http" in args:
        host = "127.0.0.1"
        port = 8765
        if "--port" in args:
            port = int(args[args.index("--port") + 1])
        if "--host" in args:
            host = args[args.index("--host") + 1]
        serve_http(host, port)
    else:
        serve_stdio()


if __name__ == "__main__":
    main()
