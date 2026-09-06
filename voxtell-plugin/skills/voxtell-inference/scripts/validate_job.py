"""
Preflight validator for Fir cluster jobs.

Every check here corresponds to a job that already failed, a number that turned
out to be a measurement artifact, or a resource request that draws a complaint.
Run it before submitting anything:

    python validate_job.py <script>.sh

Exit code 0 = safe to submit. 1 = do not submit.

Checks are deliberately textual (grep-level) rather than clever. A validator that
is hard to read does not get trusted, and one that is hard to trust does not get
run.
"""

import re
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

OK, WARN, FAIL = "PASS", "WARN", "FAIL"


class Report:
    def __init__(self):
        self.rows = []

    def add(self, status, name, detail=""):
        self.rows.append((status, name, detail))

    def render(self, title):
        w = max(len(n) for _, n, _ in self.rows) + 2
        print("=" * 74)
        print(title)
        print("=" * 74)
        for status, name, detail in self.rows:
            mark = {OK: "  ok  ", WARN: " warn ", FAIL: " FAIL "}[status]
            print(f"[{mark}] {name:<{w}} {detail}")
        n_fail = sum(1 for s, _, _ in self.rows if s == FAIL)
        n_warn = sum(1 for s, _, _ in self.rows if s == WARN)
        print("-" * 74)
        print(f"{len(self.rows)} checks · {n_fail} failed · {n_warn} warnings")
        return n_fail


def check_sh(path: Path, rep: Report):
    t = path.read_text(encoding="utf-8", errors="replace")

    # 1 — venv must match the model being run
    py_target = re.search(r"python\s+-u\s+(\S+\.py)", t)
    target = py_target.group(1) if py_target else ""
    is_nni = "nni" in target.lower() or "nninteractive" in target.lower()
    # Match on the environment name, not the full path. Everyone on the cluster
    # has their own home and scratch, so pinning one account's paths here made
    # this report a FAIL for every other user regardless of what the job said.
    activations = re.findall(r"source\s+(\S+)/bin/activate", t)
    has_vox_env = any(re.search(r"envs/voxtell\b", a) for a in activations)
    has_nni_env = any(re.search(r"envs/nninteractive\b", a) for a in activations)
    if is_nni and has_nni_env:
        rep.add(OK, "venv matches model", "nnInteractive env")
    elif not is_nni and has_vox_env:
        rep.add(OK, "venv matches model", "voxtell env")
    elif not is_nni and has_nni_env:
        rep.add(FAIL, "venv matches model",
                f"{target} looks like VoxTell but activates the nnInteractive env. "
                "Dies ~1 min in on ModuleNotFoundError: transformers.")
    elif is_nni and has_vox_env:
        rep.add(FAIL, "venv matches model",
                f"{target} looks like nnInteractive but activates the voxtell env.")
    elif "source " not in t:
        rep.add(FAIL, "venv matches model", "no environment activated at all")
    else:
        rep.add(WARN, "venv matches model", "could not classify the target script")

    # 2 — HF_HOME: compute nodes are offline
    if "HF_HOME" in t:
        rep.add(OK, "HF_HOME exported")
    elif is_nni:
        rep.add(OK, "HF_HOME exported", "not needed for nnInteractive")
    else:
        rep.add(FAIL, "HF_HOME exported",
                "compute nodes have no internet; from_pretrained fails after the job starts")

    # 3 — allocation
    acct = re.search(r"--account=(\S+)", t)
    if not acct:
        rep.add(FAIL, "allocation", "--account missing; job will be rejected")
    elif acct.group(1) == "rrg-jma":
        rep.add(OK, "allocation", "rrg-jma")
    elif acct.group(1) in ("def-jma-ab", "axc-572-ab"):
        rep.add(FAIL, "allocation",
                f"{acct.group(1)} is the no-RAC default: low priority, does not draw on the RRG")
    else:
        rep.add(WARN, "allocation", f"{acct.group(1)} — expected rrg-jma")

    # 4 — CPU request; these jobs measure ~3% CPU efficiency
    cpus = re.search(r"--cpus-per-task=(\d+)", t)
    ntasks = re.search(r"--ntasks=(\d+)", t)
    if cpus and int(cpus.group(1)) > 4:
        rep.add(FAIL, "cpu request",
                f"{cpus.group(1)} cores; jobs are GPU-bound (~3% CPU eff) and this shows in seff")
    elif cpus and int(cpus.group(1)) > 2:
        rep.add(WARN, "cpu request", f"{cpus.group(1)} cores; 2 has been sufficient")
    elif cpus:
        rep.add(OK, "cpu request", f"{cpus.group(1)} cores")
    else:
        rep.add(WARN, "cpu request", "--cpus-per-task not set")
    if ntasks and ntasks.group(1) != "1":
        rep.add(WARN, "ntasks", f"{ntasks.group(1)}; inference jobs want 1")

    # 5 — MIG slice rather than a whole card
    if re.search(r"--gpus=.*3g\.40gb|--gpus=.*2g\.20gb", t):
        rep.add(OK, "gpu slice", "MIG")
    elif re.search(r"--gpus=h100:1|--gpus=.*80gb_hbm3:1", t):
        rep.add(FAIL, "gpu slice", "full H100 costs 12.2 RGU vs 6.1 for the 3g.40gb slice")
    else:
        rep.add(WARN, "gpu slice", "could not identify the GPU request")

    # 6 — unbuffered output, or a hung job looks like a working one
    if re.search(r"python\s+-u\b", t):
        rep.add(OK, "python -u")
    else:
        rep.add(FAIL, "python -u", "log stays empty until exit; a hang is indistinguishable from progress")

    # 7 — walltime
    tm = re.search(r"--time=(\S+)", t)
    if not tm:
        rep.add(FAIL, "walltime", "--time missing")
    else:
        rep.add(OK, "walltime", tm.group(1))

    # 8 — log destination. Match the shape of the path, not one user's account:
    # what matters is that logs land on scratch and carry the job id.
    on_scratch = re.search(r"--output=\S*/scratch/\S+", t)
    if on_scratch and "%j" in t:
        rep.add(OK, "log path", "scratch, job-id stamped")
    elif on_scratch:
        rep.add(WARN, "log path", "no %j; concurrent jobs will overwrite each other")
    else:
        rep.add(FAIL, "log path", "logs should go to a /scratch/<user>/logs/ path")

    # 9 — working directory. Home is small and not meant for job I/O; the repo
    # should be run from scratch.
    if re.search(r"cd\s+/scratch/\S+/VoxTell-main", t):
        rep.add(OK, "working dir")
    elif re.search(r"^\s*cd\s+", t, re.M):
        rep.add(WARN, "working dir", "cd is not to a /scratch/<user>/VoxTell-main path")
    else:
        rep.add(FAIL, "working dir", "no cd; the job starts wherever sbatch was run")

    return target


def check_py(path: Path, rep: Report):
    """Measurement rules. Only meaningful for scripts that time something."""
    t = path.read_text(encoding="utf-8", errors="replace")
    if not re.search(r"perf_counter|time\.time", t):
        rep.add(OK, "timing code", "none found; measurement rules not applicable")
        return

    if re.search(r"[Ww]arm", t):
        rep.add(OK, "warmup present")
    else:
        rep.add(FAIL, "warmup present",
                "whichever arm runs first absorbs CUDA/cuDNN init; this inflated one result 1.0x -> 7.1x")

    if "synchronize" in t:
        n_sync = t.count("torch.cuda.synchronize")
        n_timer = len(re.findall(r"perf_counter\(\)\s*-\s*t0", t))
        if n_sync >= n_timer:
            rep.add(OK, "cuda sync before timer stop", f"{n_sync} syncs / {n_timer} timers")
        else:
            rep.add(WARN, "cuda sync before timer stop",
                    f"{n_sync} syncs for {n_timer} timers; CUDA is async, you may be timing the launch")
    else:
        rep.add(FAIL, "cuda sync before timer stop", "no torch.cuda.synchronize; timings measure kernel launch")

    if re.search(r"_backbone_quantized", t):
        rep.add(OK, "precision asserted")
    elif re.search(r"BitsAndBytesConfig|load_in_4bit|VoxTellPredictor", t):
        rep.add(FAIL, "precision asserted",
                "_load_text_backbone falls back to FP16 on any exception while still logging INT4")
    else:
        rep.add(OK, "precision asserted", "no quantized backbone in play")

    if re.search(r"_embed_cache|_prompt_cache_path", t):
        if re.search(r"assert.*(cache|exists)", t):
            rep.add(OK, "cache state asserted")
        else:
            rep.add(FAIL, "cache state asserted",
                    "clearing is not proof; assert empty before the cold read and written after")
    else:
        rep.add(OK, "cache state asserted", "cache not touched")

    if re.search(r"mni_icbm|_t1_tal_nlin", t) and not re.search(r"BENCH_IMAGE|cvpr_val", t):
        rep.add(FAIL, "representative volume",
                "MNI brain gives 4 patches at any tile_step; use a cvpr_val CT case")
    else:
        rep.add(OK, "representative volume")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    sh = Path(sys.argv[1])
    if not sh.exists():
        print(f"not found: {sh}")
        return 2

    rep = Report()
    target = check_sh(sh, rep)

    py = sh.parent / target if target else None
    if py and py.exists():
        check_py(py, rep)
    elif target:
        rep.add(FAIL, "python target exists", f"{target} not found next to the .sh")

    n_fail = rep.render(f"PREFLIGHT — {sh.name}" + (f" → {target}" if target else ""))
    print()
    if n_fail:
        print(f"BLOCKED: {n_fail} check(s) failed. Fix these before submitting.")
        return 1
    print("CLEAR: safe to submit.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
