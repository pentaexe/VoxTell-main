#!/usr/bin/env python3
"""
Check a laptop is ready to run VoxTell locally, and fetch the checkpoint.

    python setup_local.py            # report only
    python setup_local.py --download # also fetch the ~1.7 GB checkpoint

The checkpoint is not in the git repository (it is 1.7 GB and .pth is
gitignored), so a fresh clone always needs this step. It comes from Hugging Face
at mrokuss/VoxTell and lands in ~/.cache/voxtell_models, which is where the MCP
server looks by default.

Exit code 0 means ready to segment. 1 means something is missing.
"""
import argparse
import sys
from pathlib import Path

HF_REPO    = "mrokuss/VoxTell"
MODEL_NAME = "voxtell_v1.1"
CACHE      = Path.home() / ".cache" / "voxtell_models" / MODEL_NAME


def probe(mod, hint):
    try:
        m = __import__(mod)
        return True, getattr(m, "__version__", "installed")
    except Exception:
        return False, hint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true",
                    help="fetch the checkpoint if it is not already present")
    args = ap.parse_args()

    print("VoxTell local setup check")
    print("=" * 62)
    ok = True

    for mod, hint in [
        ("torch",        "pip install torch --index-url https://download.pytorch.org/whl/cu126"),
        ("voxtell",      "pip install voxtell"),
        ("nibabel",      "pip install nibabel"),
        ("transformers", "pip install transformers"),
        ("nnunetv2",     "pip install nnunetv2"),
    ]:
        good, detail = probe(mod, hint)
        ok = ok and good
        print(f"  {'ok     ' if good else 'MISSING'}  {mod:<16} {detail}")

    # accelerate is not optional in practice: without it the INT4 loader
    # silently falls back to FP16 and the run is mislabelled.
    good, detail = probe("accelerate", "pip install accelerate")
    print(f"  {'ok     ' if good else 'warn   '}  {'accelerate':<16} "
          + (detail if good else "missing: INT4 will silently serve FP16"))

    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ok       {'CUDA':<16} {torch.cuda.get_device_name(0)}")
        else:
            print(f"  warn     {'CUDA':<16} not available; CPU inference is impractically slow")
    except Exception:
        pass

    print()
    if (CACHE / "plans.json").exists():
        print(f"  ok       checkpoint       {CACHE}")
    elif args.download:
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            print("  MISSING  huggingface_hub  pip install huggingface_hub")
            return 1
        print(f"  fetching {HF_REPO} ({MODEL_NAME}, about 1.7 GB) ...")
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        snapshot_download(repo_id=HF_REPO, allow_patterns=[f"{MODEL_NAME}/*"],
                          local_dir=str(CACHE.parent))
        good = (CACHE / "plans.json").exists()
        ok = ok and good
        print(f"  {'ok     ' if good else 'FAILED '}  checkpoint       {CACHE}")
    else:
        ok = False
        print("  MISSING  checkpoint       rerun with --download to fetch it")

    print()
    print("READY" if ok else "NOT READY: resolve the items above")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
