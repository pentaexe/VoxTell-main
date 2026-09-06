#!/usr/bin/env python3
"""
Work out which interpreter the plugin should use, and print the exact value.

Run this if the plugin installs and enables but no tools appear:

    python3 doctor.py          (or py doctor.py on Windows)

Why this script exists. The MCP server is launched as a subprocess by whatever
`voxtell_python` names. If that name does not resolve to a real interpreter with
voxtell installed, the process never starts, so there is no server to write a
log — the plugin looks installed and enabled with no tools and nothing to read.

There is no bare name that works everywhere. On Ubuntu `python` usually does not
exist. On Windows `python`, `python3` and `py` all commonly resolve to the
Microsoft Store stub, which exits immediately without running anything. The
value has to be a full path to the environment where you installed voxtell.

This script has no dependencies and runs under any Python 3.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

# A Windows console defaults to cp1252, which cannot encode the em dashes and
# arrows below. Without this the very first print raises UnicodeEncodeError and
# the script produces no output at all — a confusing failure for something whose
# whole job is to explain what is wrong. Degrade the characters, not the run.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Ask the interpreter what it has, rather than judging by where the files sit.
# Path matching looked reasonable and was wrong: an editable install can live
# anywhere, and running this script from one checkout while the interpreter is
# installed from another reported a perfectly good optimized build as "stock".
# The markers below are additions in this fork, absent from the PyPI package.
PROBE = (
    "import json,sys\n"
    "out={'exe':sys.executable,'ver':sys.version.split()[0]}\n"
    "try:\n"
    "    import voxtell,pathlib\n"
    "    out['voxtell']=str(pathlib.Path(voxtell.__file__).resolve())\n"
    "    missing=[]\n"
    "    for mod,attr,label in ((  'voxtell.inference.predictor','_prompt_cache_path','embedding cache'),\n"
    "                           ('voxtell.inference.predictor','_load_text_backbone','INT4 text backbone'),\n"
    "                           ('voxtell.utils.fast_preprocess','numba_crop_to_nonzero','Numba preprocessing')):\n"
    "        try:\n"
    "            m=__import__(mod,fromlist=[attr])\n"
    "            if not hasattr(m,attr): missing.append(label)\n"
    "        except Exception: missing.append(label)\n"
    "    out['missing']=missing\n"
    "except Exception as e:\n"
    "    out['voxtell']=None; out['err']=type(e).__name__\n"
    "try:\n"
    "    import torch; out['torch']=torch.__version__; out['cuda']=torch.cuda.is_available()\n"
    "except Exception:\n"
    "    out['torch']=None\n"
    # The INT4 backbone needs these at run time. Without them the fork still
    # passes every feature check above and quietly serves FP16 instead.
    "int4=[]\n"
    "for m in ('bitsandbytes','accelerate'):\n"
    "    try: __import__(m)\n"
    "    except Exception: int4.append(m)\n"
    "out['int4_missing']=int4\n"
    "print('PROBE'+json.dumps(out))\n"
)


def candidates():
    """Interpreters worth testing, most specific first."""
    seen, out = set(), []

    def add(p):
        if p and str(p) not in seen and Path(p).exists():
            seen.add(str(p))
            out.append(str(p))

    add(sys.executable)                      # the one running this script

    for name in ("python3", "python", "py"):
        add(shutil.which(name))

    # Common environment layouts, so a user who installed into a venv or conda
    # env does not have to remember where it landed.
    home = Path.home()
    for base in (home / "miniconda3" / "envs", home / "anaconda3" / "envs",
                 home / ".conda" / "envs", home / "envs", home / ".virtualenvs"):
        if base.is_dir():
            for env in sorted(base.iterdir()):
                add(env / "bin" / "python")          # posix
                add(env / "python.exe")              # windows conda
    for venv in (".venv", "venv"):
        add(Path.cwd() / venv / "bin" / "python")
        add(Path.cwd() / venv / "Scripts" / "python.exe")
    return out


def probe(exe):
    # Run from a neutral directory. Probing inside the repo makes every
    # interpreter look like it has voxtell, because the current directory is on
    # sys.path and ./voxtell/ imports whether or not anything is installed.
    import tempfile
    try:
        r = subprocess.run([exe, "-c", PROBE], capture_output=True, text=True,
                           timeout=60, cwd=tempfile.gettempdir())
    except Exception as e:
        return {"exe": exe, "dead": f"{type(e).__name__}"}
    for line in (r.stdout or "").splitlines():
        if line.startswith("PROBE"):
            return json.loads(line[5:])
    # The Microsoft Store stub exits without output. It returns 49 when launched
    # directly and 9009 through a shell that resolves it as a missing command;
    # both mean the same thing, so name it either way.
    return {"exe": exe, "dead": f"no output, exit {r.returncode}"
            + (" (Microsoft Store stub)" if r.returncode in (49, 9009) else "")}


def main():
    repo = None
    for parent in Path(__file__).resolve().parents[:6]:
        if (parent / "voxtell" / "__init__.py").exists():
            repo = parent
            break

    print("Looking for an interpreter that can run VoxTell")
    print("=" * 66)
    if repo:
        print(f"repo checkout: {repo}\n")
    else:
        print("repo checkout: not found near this script\n")

    good = []
    for exe in candidates():
        info = probe(exe)
        if info.get("dead"):
            print(f"  unusable  {exe}\n            {info['dead']}")
            continue
        v = info.get("voxtell")
        if not v:
            print(f"  no voxtell {exe}  (python {info['ver']})")
            continue
        missing = info.get("missing", [])
        optimized = not missing
        if optimized:
            kind = "OPTIMIZED"
        elif len(missing) == 3:
            kind = "stock PyPI build"
        else:
            kind = "PARTIAL, missing " + ", ".join(missing)
        cuda = "CUDA" if info.get("cuda") else "no CUDA"
        print(f"  {'BEST    ' if optimized else 'works   '} {exe}")
        print(f"            python {info['ver']}, {cuda}, voxtell: {kind}")
        print(f"            {v}")
        int4_missing = info.get("int4_missing", [])
        if optimized and int4_missing:
            print(f"            INT4 unavailable: no {', '.join(int4_missing)}; "
                  "the backbone will quietly run FP16")
        good.append((optimized, bool(info.get("cuda")), exe))

    print()
    if not good:
        print("No interpreter found with voxtell installed.")
        print()
        print("Install it from the repo checkout so you get the optimized build:")
        print(f"    pip install -e {repo or '/path/to/VoxTell-main'}")
        print("    pip install huggingface_hub")
        return 1

    good.sort(key=lambda t: (not t[0], not t[1]))
    optimized, has_cuda, best = good[0]
    print("Set voxtell_python to:")
    print(f"    {best}")
    if not optimized:
        print()
        print("  WARNING: that interpreter has the stock PyPI voxtell, which does not")
        print("  contain the INT4 backbone, embedding cache, Numba preprocessing or")
        print("  tile_step=0.75. It will segment, but none of the measured speedups")
        print("  apply. Install the repo copy instead:")
        print(f"      {best} -m pip install -e {repo or '/path/to/VoxTell-main'}")
        print()
        print("  Both distributions are called 'voxtell' and share a base version, so")
        print("  check with: pip show voxtell   ->   0.1.0+optimized is this fork.")
    if not has_cuda:
        print()
        print("  WARNING: no CUDA on that interpreter. Inference will run on CPU and")
        print("  a single 3-D volume takes minutes rather than seconds.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
