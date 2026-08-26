# Slide fixes — do before finalizing the deck

Deferred deliberately until all numbers were confirmed. Each item below is a
specific line in `make_slides.py`, not a general "review the slides" note.

## 1. Self-assigned 0.005 DSC threshold — 3 places

- `make_slides.py:296` — "Below 0.005 threshold"
- `make_slides.py:299` — "Threshold: any DSC change < 0.005 is within measurement
  noise for FP16 non-associativity."
- `make_slides.py:488` — "Well below 0.005 threshold"

**Why this matters, especially line 299**: it attaches a *mechanism story*
(FP16 non-associativity) to a threshold that was self-assigned rather than drawn
from the challenge. That is the same failure pattern as explaining the NF4 gain
with a mechanism instead of measuring it. Two separate problems compounding:
the bar is invented, and the justification is post-hoc.

**Replace with**: the object count and the across-run range. "Four runs, Δ from
+0.0000 to +0.0004, none showing degradation (294 objects, 20 cases)." No bar.

## 2. INT4 row — both cells are stale

`make_slides.py:245-248`, the optimization table INT4 row currently reads
`'~4× text (H100)'` and `'Not measured'`.

The `~4×` came from the FP16-vs-INT4 benchmark that was invalidated when both
arms were matched to INT4. **Nothing in `fair_benchmark.py` measures it any more.**
Replace with the forward-pass ratio from `int4_dsc_comparison.py` (job 56955602),
labeled: text encoding forward pass only, backbone pre-loaded, H100 MIG.

Do not carry the old `~4×` forward. Both cells need replacing, not softening.

## 3. VoxTell H100 speedup — replace the invalid 17.6×

Job 56948503's 17.6× had three defects: no GPU warmup, a disk-cache hit compared
against an uncached FP16 embed, and uncontrolled INT4-vs-FP16 precision.
Superseded by job 56955600 (precision-matched, warmed, cold-to-cold).

Report both numbers that job prints:
- Comparison A — cold-to-cold, both INT4: **pure algorithmic gain**
- Comparison B — with embedding cache: production warm-query case

Label A as algorithmic. Do not call it "fair GPU-vs-GPU" without saying what is
held constant.

## 4. nnInteractive speedup — n=4, not a single job

Cite **1.33× mean, range 1.28–1.39× (n=4)**. Never 1.34× alone (that is job
56908464 only). Break-even ~331 objects / ~22 cases at 0.0714s/object mean gain.

## 5. Two different DSC measures — do not present them as one

- nnInteractive: **true DSC vs ground truth** at
  `/scratch/brianx7/cvpr_val/3D_val_gt/3D_val_gt_interactive`, 294 objects.
- VoxTell INT4: **output agreement** DSC(pred_fp16, pred_int4), n=1, no ground truth.

A reviewer will notice these are different quantities. Label them differently on
the slide. Agreement does not establish accuracy — both arms can drift from GT
together and still agree perfectly.
