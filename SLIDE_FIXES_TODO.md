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

## 4. RTX "1.3× algorithmic gain" — measured with the defective methodology

`make_slides.py:260` and `make_slides.py:564` cite
`v0_gpu 3.10s → v3 2.38s = 1.3× algorithmic gain` from `fair_benchmark_results.txt`.

That file is dated **2026-04-10** and was produced by the pre-fix
`fair_benchmark.py`. Two of the three defects that invalidated the H100 run apply:

- **No GPU warmup** — v0_gpu absorbed CUDA context init. Applies.
- **v3 embed was likely a cache hit** — the old script never cleared the cache. Applies.
- **FP16-vs-INT4 precision asymmetry** — probably does *NOT* apply here. The local
  `voxtell` env has no `accelerate`, so `_load_text_backbone` (predictor.py:89) hits
  an ImportError and silently falls back to FP16. The April v3 arm was therefore
  most likely FP16, matching v0_gpu by accident. **Unconfirmed** — no April log was
  kept; if one surfaces, look for `[Text backbone] INT4 unavailable (ImportError)`.

So the RTX number is less wrong than the H100 one was, but "1.3× algorithmic gain"
still rests on an unwarmed baseline and a cached v3 embed.

**Fix**: rerun locally with the corrected script. The local `voxtell` conda env has
CUDA on the RTX 4070 SUPER, bitsandbytes 0.49.2, and Qwen3-4B cached — but needs
`accelerate>=0.26.0` installed before INT4 will load. Then cite RTX and H100 from the
same script version. If the rerun is not used, the slide must say the RTX figure came
from an earlier methodology and is not comparable to the H100 number beside it.

Note that installing `accelerate` changes local behaviour: v3 will genuinely run INT4
where it previously fell back to FP16. That is the intended state (it matches the
cluster), but it means local results before and after that install are not comparable.

## 5. nnInteractive speedup — n=4, not a single job

Cite **1.33× mean, range 1.28–1.39× (n=4)**. Never 1.34× alone (that is job
56908464 only). Break-even ~331 objects / ~22 cases at 0.0714s/object mean gain.

## 6. The VoxTell benchmark image may be unrepresentative of the challenge data

Every VoxTell timing number comes from ONE MNI T1 brain volume
(`mni_icbm152_t1_tal_nlin_sym_09a.nii.gz`, shape 189×233×197) with the single
prompt "brain". The CVPR challenge validation data — and all the nnInteractive
numbers — are abdominal CT.

The RTX rerun showed why this matters: at 189×233×197 against a 192³ patch, the
sliding-window grid is 4 patches at **both** tile_step 0.5 and 0.75, so tile_step
has essentially nothing to act on. The March claim of `343 → 125 patches, 3.6×`
came from a larger image. Neither figure has been tested on the other's volume.

**Do not state** "tile_step gives no benefit" — that overreaches from one small
volume. **Do state** which volume each number was measured on. If time allows
before the interview, run `fair_benchmark.py` against one CT volume from
`/scratch/brianx7/cvpr_val/3D_val_npz` so the VoxTell and nnInteractive numbers
describe the same kind of data.

## 7. Two different DSC measures — do not present them as one

- nnInteractive: **true DSC vs ground truth** at
  `/scratch/brianx7/cvpr_val/3D_val_gt/3D_val_gt_interactive`, 294 objects.
- VoxTell INT4: **output agreement** DSC(pred_fp16, pred_int4), n=1, no ground truth.

A reviewer will notice these are different quantities. Label them differently on
the slide. Agreement does not establish accuracy — both arms can drift from GT
together and still agree perfectly.
