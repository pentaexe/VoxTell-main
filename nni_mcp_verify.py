#!/usr/bin/env python3
"""
Verify nninteractive_segment through the MCP server, against real weights.

Everything else about that tool is covered by tests/test_server.py, but the
model itself has never run through it: the weights are cluster-only, so the
suite exercises the code path against a stand-in session that enforces the
API's shapes. That stand-in caught a real bug — the tool passed a 5-D image and
a 4-D buffer and had never once completed — but it cannot tell us whether the
mask that comes back is any good.

This closes that. It drives the real server over stdio the way a client does,
segments boxes that come with the CVPR cases, and scores each against ground
truth. The number to compare with is the ~0.79 DSC the direct script gets with
fold='all'; anything near 0.33 means fold=0 weights crept in somewhere.

    sbatch nni_mcp_verify.sh

Reads nothing it does not need and writes only into its own output directory.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

USER      = os.environ.get("USER", "")
PROJECT   = Path(f"/scratch/{USER}/VoxTell-main")
SERVER    = PROJECT / "voxtell-plugin" / "mcp_server" / "server.py"
WEIGHTS   = f"/scratch/{USER}/nnInteractive_weights/nnInteractive_v1.0"
INPUT_DIR = Path(f"/scratch/{USER}/cvpr_val/3D_val_npz")
GT_DIR    = Path(f"/scratch/{USER}/cvpr_val/3D_val_gt/3D_val_gt_interactive")
OUT_DIR   = Path(f"/scratch/{USER}/nni_mcp_verify_out")

# The server builds a fresh session per call, so every box costs a model load.
# Keep this small: the point is correctness, not throughput.
N_CASES   = 2
N_BOXES   = 2


def make_bbox(b):
    """Same conversion the verified direct script uses (max is exclusive)."""
    return [
        [int(b["z_min"]),       int(b["z_max"]) + 1],
        [int(b["z_mid_y_min"]), int(b["z_mid_y_max"]) + 1],
        [int(b["z_mid_x_min"]), int(b["z_mid_x_max"]) + 1],
    ]


def dice(pred_mask, gt, label):
    p = pred_mask > 0
    g = gt == label
    denom = p.sum() + g.sum()
    if denom == 0:
        return float("nan")
    return float(2 * (p & g).sum() / denom)


def preflight():
    """Fail loudly here rather than a minute into a session load."""
    print("Preflight")
    ok = True
    for label, path in [("server", SERVER), ("weights", Path(WEIGHTS)),
                        ("inputs", INPUT_DIR), ("ground truth", GT_DIR)]:
        exists = path.exists()
        ok = ok and exists
        print(f"  {'ok  ' if exists else 'MISS'}  {label:<14} {path}")
    fold_all = (Path(WEIGHTS) / "fold_all").exists()
    print(f"  {'ok  ' if fold_all else 'MISS'}  {'fold_all':<14} "
          + ("present" if fold_all else "absent — fold=0 scores ~0.33 and is not a valid baseline"))
    ok = ok and fold_all

    # This is the venv trap: a job pointed at the wrong environment dies a minute
    # in, after the queue wait, on an import that could have been checked here.
    for mod in ("torch", "numpy", "nnInteractive"):
        try:
            __import__(mod)
            print(f"  ok    {mod:<14} importable")
        except Exception as e:
            ok = False
            print(f"  MISS  {mod:<14} {type(e).__name__}: {e}")
    try:
        import torch
        print(f"  {'ok  ' if torch.cuda.is_available() else 'MISS'}  {'CUDA':<14} "
              + (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no GPU visible"))
        ok = ok and torch.cuda.is_available()
    except Exception:
        ok = False
    print()
    return ok


def pick_jobs():
    """Cases that have both boxes and ground truth, with their boxes."""
    jobs = []
    for case in sorted(INPUT_DIR.glob("CT_*.npz"))[:40]:
        if not (GT_DIR / case.name).exists():
            continue
        d = np.load(case, allow_pickle=True)
        boxes = d.get("boxes")
        if boxes is None or len(boxes) == 0:
            continue
        jobs.append((case, list(boxes)[:N_BOXES], d["imgs"].shape))
        if len(jobs) >= N_CASES:
            break
    return jobs


def main():
    if not preflight():
        print("NOT READY — resolve the items above before submitting again.")
        return 1

    jobs = pick_jobs()
    if not jobs:
        print(f"No usable cases found in {INPUT_DIR} with boxes and matching ground truth.")
        return 1
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build every call up front and send them down one stdio session, the way a
    # client would hold the connection open.
    msgs = [{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}]
    plan = []
    rid = 2
    for case, boxes, shape in jobs:
        for oid, b in enumerate(boxes):
            bbox = make_bbox(b)
            clamped = [[max(0, lo), min(hi, shape[i])] for i, (lo, hi) in enumerate(bbox)]
            if clamped != bbox:
                print(f"note: {case.name} box {oid} clipped to the volume {shape}: "
                      f"{bbox} -> {clamped}")
            # One directory per box. The server names the file after the input,
            # so every box of a case would otherwise write to the same path —
            # and since all calls run before any result is read, each box would
            # be scored against the last box's mask.
            box_dir = OUT_DIR / f"{case.name.split('.')[0]}_box{oid}"
            msgs.append({"jsonrpc": "2.0", "id": rid, "method": "tools/call",
                         "params": {"name": "nninteractive_segment",
                                    "arguments": {"image_path": str(case),
                                                  "bbox": clamped,
                                                  "output_dir": str(box_dir)}}})
            plan.append((rid, case, oid, clamped, box_dir))
            rid += 1

    print(f"Running {len(plan)} box(es) across {len(jobs)} case(s) through the MCP server.")
    print(f"Each call builds its own session, so expect a model load per box.\n")

    env = dict(os.environ)
    env["NNINTERACTIVE_WEIGHTS"] = WEIGHTS
    env["PYTHONUNBUFFERED"] = "1"

    # Deliberately untimed. This job answers "is the mask correct", and timing
    # code here would be quotable as a benchmark without any of the warmup and
    # synchronisation that would make it mean anything. SLURM's own accounting
    # has the wall clock if anyone wants it.
    proc = subprocess.run(
        [sys.executable, str(SERVER)],
        input="".join(json.dumps(m) + "\n" for m in msgs),
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        env=env, cwd=str(PROJECT), timeout=3600,
    )
    print(f"server exited {proc.returncode}\n")

    # stdout is the JSON-RPC channel; anything else on it is a protocol bug.
    replies, dirty = {}, []
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        try:
            m = json.loads(line)
            replies[m.get("id")] = m
        except json.JSONDecodeError:
            dirty.append(line)
    if dirty:
        print(f"PROTOCOL PROBLEM: {len(dirty)} non-JSON line(s) on stdout, first:")
        print(f"  {dirty[0][:160]!r}\n")

    print(f"{'case':<28} {'box':>3} {'label':>5} {'voxels':>10} {'DSC':>7}")
    print("-" * 60)
    scores, failures = [], []
    for rid, case, oid, bbox, box_dir in plan:
        r = replies.get(rid, {}).get("result", {})
        text = "\n".join(c.get("text", "") for c in r.get("content", []))
        if r.get("isError") or not r:
            failures.append((case.name, oid, text.strip().splitlines()[:2]))
            print(f"{case.name:<28} {oid:>3} {'-':>5} {'-':>10} {'ERROR':>7}")
            continue
        pred_path = box_dir / f"{case.name.split('.')[0]}__bbox.npz"
        if not pred_path.exists():
            failures.append((case.name, oid, [f"no mask written at {pred_path}"]))
            print(f"{case.name:<28} {oid:>3} {'-':>5} {'-':>10} {'NO FILE':>7}")
            continue
        pred = np.load(pred_path, allow_pickle=True)["mask"]
        gt = np.load(GT_DIR / case.name, allow_pickle=True)["gts"]
        if pred.shape != gt.shape:
            failures.append((case.name, oid, [f"shape {pred.shape} != gt {gt.shape}"]))
            print(f"{case.name:<28} {oid:>3} {'-':>5} {'-':>10} {'SHAPE':>7}")
            continue
        d = dice(pred, gt, oid + 1)
        scores.append(d)
        print(f"{case.name:<28} {oid:>3} {oid+1:>5} {int((pred>0).sum()):>10,} {d:>7.4f}")

    print("-" * 60)
    for name, oid, why in failures:
        print(f"FAILED {name} box {oid}: {' | '.join(why)}")

    if scores:
        valid = [s for s in scores if s == s]  # drop NaN
        mean = sum(valid) / len(valid) if valid else float("nan")
        print(f"\nmean DSC over {len(valid)} box(es): {mean:.4f}")
        print("reference: fold='all' scores about 0.79 on this data; "
              "about 0.33 means fold=0 weights, which is not a valid baseline.")
        verdict = 0.55 <= mean if valid else False
        print("\n" + ("VERIFIED — the MCP path produces masks of the expected quality."
                      if verdict else
                      "PROBLEM — DSC is not in the range fold='all' should give. Do not ship on this."))
        return 0 if verdict and not failures and not dirty else 1

    print("\nNo scores produced.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
