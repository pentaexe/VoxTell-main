#!/usr/bin/env python3
"""
Regression tests for the MCP server, driven over real stdio like a client.

    python tests/test_server.py

No data files needed. The volumes are synthesized to land on either side of the
validator's thresholds, so this runs anywhere the server's own dependencies are
installed. The one test that needs the real checkpoint and the optimized build
skips itself with a note when they are absent.

What this is guarding, in order of how quietly each one fails:

1. stdout discipline. In stdio mode stdout IS the JSON-RPC channel, so every
   line the server writes there must parse as JSON. VoxTell's predictor
   announces its quantization mode with a bare print(); that one line corrupts
   the stream. Claude Code happened to tolerate it, which is exactly why it
   needs a test — a stricter client would not, and the failure would surface as
   an unrelated-looking protocol error mid-conversation.

2. The refusal gate. An abdominal CT with a brain prompt must come back as an
   error before any model is loaded, not as a plausible mask of nothing.

3. Build provenance. A stock PyPI voxtell shadowing the fork must be refused.
   It segments perfectly well and returns a believable mask with none of the
   measured optimizations, so nothing downstream would reveal the downgrade.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE   = Path(__file__).resolve().parent
PLUGIN = HERE.parent
SERVER = PLUGIN / "mcp_server" / "server.py"

failures = []
skipped  = []


# ── Synthetic volumes ─────────────────────────────────────────────────────────

def write_volumes(dest: Path):
    """Two volumes chosen to sit firmly on either side of the region thresholds."""
    import nibabel as nib
    import numpy as np

    rng = np.random.default_rng(0)

    # Torso CT: air background at -1000, a body cylinder around soft-tissue HU,
    # and 510 mm of coverage, which is past the 400 mm the validator treats as
    # spanning thorax and abdomen together.
    shape = (340, 256, 256)
    ct = np.full(shape, -1000.0, dtype=np.float32)
    zz, yy, xx = np.ogrid[: shape[0], : shape[1], : shape[2]]
    body = ((yy - 128) ** 2 + (xx - 128) ** 2) < 100 ** 2
    ct[np.broadcast_to(body, shape)] = 40.0
    lungs = ((yy - 128) ** 2 + (xx - 110) ** 2) < 45 ** 2
    ct[np.broadcast_to(lungs, shape) & (zz < 150)] = -850.0
    ct += rng.normal(0, 8, shape).astype(np.float32)
    ct_path = dest / "synthetic_torso_ct.nii.gz"
    nib.save(nib.Nifti1Image(ct, np.diag([1.5, 1.5, 1.5, 1.0])), str(ct_path))

    # Brain MR: no negative intensities, so it reads as MR rather than CT, and a
    # compact field of view, which is what the validator uses for a head.
    shape = (180, 220, 200)
    mr = np.zeros(shape, dtype=np.float32)
    zz, yy, xx = np.ogrid[: shape[0], : shape[1], : shape[2]]
    head = (((zz - 90) / 80.0) ** 2 + ((yy - 110) / 95.0) ** 2
            + ((xx - 100) / 85.0) ** 2) < 1.0
    mr[head] = 600.0
    mr += rng.uniform(0, 30, shape).astype(np.float32)
    mr_path = dest / "synthetic_brain_mr.nii.gz"
    nib.save(nib.Nifti1Image(mr, np.diag([1.0, 1.0, 1.0, 1.0])), str(mr_path))

    return ct_path, mr_path


def write_stock_voxtell(dest: Path) -> Path:
    """A voxtell package with the right module names and none of the optimizations.

    This is what `pip install voxtell` gives you: it imports, it segments, and
    it carries no embedding cache, no INT4 backbone and no Numba preprocessing.
    """
    root = dest / "stock_site_packages" / "voxtell"
    (root / "inference").mkdir(parents=True, exist_ok=True)
    (root / "utils").mkdir(parents=True, exist_ok=True)
    (root / "__init__.py").write_text("__version__ = '1.0.0-stock'\n")
    (root / "inference" / "__init__.py").write_text("")
    (root / "inference" / "predictor.py").write_text(
        "class VoxTellPredictor:\n"
        "    def __init__(self, *a, **k):\n"
        "        raise AssertionError('stock predictor must never be constructed')\n"
    )
    (root / "utils" / "__init__.py").write_text("")
    (root / "utils" / "fast_preprocess.py").write_text("def crop_to_nonzero(*a, **k):\n    pass\n")
    return root.parent


# ── Driving the server ────────────────────────────────────────────────────────

def call(label, calls, server=SERVER, extra_env=None, timeout=1800):
    """Send initialize, tools/list and the given tool calls over stdio."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["PYTHONIOENCODING"] = "utf-8"
    if extra_env:
        env.update(extra_env)

    msgs = [{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}]
    for i, (name, args) in enumerate(calls, start=3):
        msgs.append({"jsonrpc": "2.0", "id": i, "method": "tools/call",
                     "params": {"name": name, "arguments": args}})

    p = subprocess.run(
        [sys.executable, str(server)],
        input="".join(json.dumps(m) + "\n" for m in msgs),
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=timeout, env=env, cwd=tempfile.gettempdir(),
    )

    print(f"\n{label}\n" + "-" * len(label))

    parsed, dirty = [], []
    for line in p.stdout.splitlines():
        if line.strip():
            try:
                parsed.append(json.loads(line))
            except json.JSONDecodeError:
                dirty.append(line)

    # Guard 1, and the reason this harness exists.
    check("stdout carries only JSON", not dirty,
          f"{len(dirty)} stray line(s), first: {dirty[0][:90]!r}" if dirty else "")
    if p.returncode != 0:
        check(f"server exited cleanly", False, f"exit {p.returncode}: "
              + " | ".join(p.stderr.splitlines()[-3:]))

    return {m.get("id"): m for m in parsed}, p.stderr


def check(label, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'}  {label}" + (f"   [{detail}]" if detail and not ok else ""))
    if not ok:
        failures.append(label)


def body(res, rid):
    r = res.get(rid, {}).get("result", {})
    return "\n".join(c.get("text", "") for c in r.get("content", [])), r.get("isError", False)


# ── Tests ─────────────────────────────────────────────────────────────────────

def main():
    try:
        import nibabel, numpy  # noqa: F401
    except ImportError as e:
        print(f"cannot run: {e}. pip install nibabel numpy")
        return 2

    tmp = Path(tempfile.mkdtemp(prefix="voxtell_tests_"))
    try:
        ct, mr = write_volumes(tmp)
        out = tmp / "out"

        # 1. Anatomically impossible request is refused, with no model loaded.
        res, _ = call("Refuses a brain prompt on a torso CT",
                      [("check_request", {"image_path": str(ct), "prompt": "brain tumor"}),
                       ("voxtell_segment", {"image_path": str(ct), "prompt": "brain tumor"})],
                      timeout=300)
        t, _ = body(res, 3)
        check("check_request says REFUSED", "REFUSED" in t, t[:100])
        check("it names the region it inferred", "torso" in t or "thorax" in t, t[:100])
        t, err = body(res, 4)
        check("voxtell_segment returns isError", err is True)
        check("it refuses before spending compute", "REFUSED" in t and "no compute" in t, t[:100])

        # 2. Compatible requests are allowed. A validator that blocks whatever it
        #    cannot classify would be useless, so this is the half that matters.
        res, _ = call("Allows requests that make anatomical sense",
                      [("check_request", {"image_path": str(mr), "prompt": "brain"}),
                       ("check_request", {"image_path": str(ct), "prompt": "liver"})],
                      timeout=300)
        t, _ = body(res, 3)
        check("brain prompt on a brain MR is allowed", "REFUSED" not in t, t[:100])
        t, _ = body(res, 4)
        check("liver prompt on a torso CT is allowed", "REFUSED" not in t, t[:100])

        # 3. A stock voxtell shadowing the fork is caught. The server is copied
        #    outside the repo first: in place it puts the repo checkout at the
        #    front of sys.path, which would defeat the shadowing this tests.
        iso = tmp / "isolated" / "mcp_server"
        iso.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(SERVER.parent, iso, ignore=shutil.ignore_patterns("__pycache__"))
        res, _ = call("Refuses a stock build that would silently drop the optimizations",
                      [("voxtell_segment", {"image_path": str(mr), "prompt": "brain"})],
                      server=iso / "server.py",
                      extra_env={"PYTHONPATH": str(write_stock_voxtell(tmp))},
                      timeout=300)
        t, err = body(res, 3)
        check("stock build is refused", err is True and "REFUSED" in t, t[:150])
        check("the refusal says which build it found", "STOCK" in t, t[:150])

        # 4. nnInteractive. The weights are cluster-only, so the session itself is
        #    stubbed — but the stub enforces the real API's shapes, which is
        #    exactly what was wrong: the server passed a 5-D image and a 4-D
        #    buffer, so this tool had never run to completion.
        weights = tmp / "fake_weights" / "nnInteractive_v1.0"
        (weights / "fold_all").mkdir(parents=True)
        nni_env = {"NNINTERACTIVE_WEIGHTS": str(weights),
                   "PYTHONPATH": str(HERE / "stub_nninteractive")}

        res, _ = call("Rejects a malformed bounding box before loading anything",
                      [("nninteractive_segment", {"image_path": str(mr), "bbox": [[0, 10], [0, 10]]}),
                       ("nninteractive_segment", {"image_path": str(mr), "bbox": [[0, 10], [10, 5], [0, 10]]}),
                       ("nninteractive_segment", {"image_path": str(mr), "bbox": [[0, 99999], [0, 10], [0, 10]]})],
                      extra_env=nni_env, timeout=300)
        t, err = body(res, 3)
        check("wrong number of axes is caught", err is True and "three" in t, t[:120])
        t, err = body(res, 4)
        check("an empty axis is caught", err is True and "not below" in t, t[:120])
        t, err = body(res, 5)
        check("an out-of-bounds axis is caught", err is True and "only" in t, t[:120])

        res, _ = call("Runs the bbox path with the API shapes the real session requires",
                      [("nninteractive_segment", {"image_path": str(mr),
                                                  "bbox": [[40, 120], [60, 160], [50, 150]],
                                                  "output_dir": str(out)})],
                      extra_env=nni_env, timeout=600)
        t, err = body(res, 3)
        if "needs a CUDA GPU" in t:
            # The stub replaces the model, not the device: the session still
            # allocates pinned memory, so this half needs a real GPU.
            skipped.append("nnInteractive bbox path: no CUDA on this interpreter")
            print("  skip  bbox segmentation (no CUDA here)")
        else:
            check("bbox segmentation completed", err is not True, t[:300])
            if err is not True:
                print("\n" + "\n".join("        " + l for l in t.splitlines()))
                check("mask is written aligned, not as a raw buffer", ".nii.gz" in t and "aligned" in t)

        # Missing weights must be a clean message, not a traceback.
        res, _ = call("Says so plainly when the weights are not configured",
                      [("nninteractive_segment", {"image_path": str(mr), "bbox": [[0, 10], [0, 10], [0, 10]]})],
                      extra_env={"NNINTERACTIVE_WEIGHTS": ""}, timeout=300)
        t, err = body(res, 3)
        check("missing weights reported cleanly", err is True and "weights" in t.lower(), t[:120])

        # 5. The FP16 downgrade that feature detection cannot see. The INT4 code
        #    is part of the fork, so a build missing only bitsandbytes still
        #    looks optimized and still reports itself as such — while serving
        #    FP16. Simulate that by shadowing the package with a module that
        #    raises on import.
        blocker = tmp / "no_bnb"
        blocker.mkdir()
        (blocker / "bitsandbytes.py").write_text("raise ImportError('simulated: not installed')\n")
        res, _ = call("Notices when INT4 will silently fall back to FP16",
                      [("setup", {})], extra_env={"PYTHONPATH": str(blocker)}, timeout=300)
        t, _ = body(res, 3)
        check("setup reports the INT4 fallback", "bitsandbytes" in t and "FP16" in t, t[:200])

        # 6. The real thing, if this machine has it.
        res, _ = call("Reports what is actually installed", [("list_models", {})], timeout=300)
        tools = [x["name"] for x in res.get(2, {}).get("result", {}).get("tools", [])]
        check("all five tools are exposed", len(tools) == 5, str(tools))
        t, _ = body(res, 3)
        # Read the checkpoint line specifically. Matching "NOT CONFIGURED" across
        # the whole report also caught the nnInteractive weights line, which has
        # nothing to do with whether VoxTell can run.
        ckpt = next((l.split(":", 1)[1].strip() for l in t.splitlines()
                     if l.strip().startswith("checkpoint")), "")
        ready = "optimized fork" in t and ckpt and "NOT CONFIGURED" not in ckpt
        if not ready:
            skipped.append("end-to-end inference: needs the optimized build and the checkpoint")
            print("\n  skip  end-to-end inference (no optimized build or no checkpoint here)")
        else:
            res, errtxt = call("Segments a brain MR end to end",
                               [("voxtell_segment", {"image_path": str(mr), "prompt": "brain",
                                                     "output_dir": str(out)})])
            t, err = body(res, 3)
            check("segmentation succeeded", err is not True, t[:200])
            if err is not True:
                print("\n" + "\n".join("        " + l for l in t.splitlines()))
                check("a mask was written", any(out.glob("*.nii.gz")) if out.exists() else False)
                check("the INT4 banner went to stderr", "[Text backbone]" in errtxt)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n" + "=" * 60)
    for s in skipped:
        print(f"skipped: {s}")
    print("ALL PASS" if not failures else
          f"{len(failures)} FAILURE(S):\n  " + "\n  ".join(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
