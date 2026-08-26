# Slide fixes — do before finalizing the deck

## FINAL NUMBERS (jobs 56964410 / 56964411 / 56964412, all completed)

**Cite this one: VoxTell v3 vs v0 on CT = 2.6× (cold), 2.7× (warm cache).**
Job 56964411, `CT_AMOS_amos_0018.npz`, 63×512×512, prompt read from the file
(`"CT imaging of the spleen within the abdomen"`). Both arms INT4 (NF4), GPU and
text backbone and sliding-window path all pre-warmed, embed cache verified empty
before the cold measurement. Patches 25 → 9. Sliding window 3.083s → 1.059s = 2.91×.

⚠️ **This is n=1 and it is the headline.** The brain benchmark was run nine times
*because two runs disagreed by 33%*. The same standard has to apply here before it
goes in front of Dr. Ma. Submit 2–3 serialized repeats of `fair_benchmark_h100_ct.sh`
and cite mean with range. Until then this number has had exactly one run on one case.

**The headline speedup fell four times as methodology tightened:**

| Reported | Why it was wrong |
|---|---|
| 26.0× | CPU baseline — FP32 text encoder silently overflowed VRAM |
| 17.6× | No GPU warmup; v3 embed was a cache hit; FP16 vs INT4 |
| 7.1× | v0 ran first and absorbed text-backbone + cuDNN first-use cost |
| **1.0× (brain) / 2.6× (CT)** | current — warmed, precision-matched, verified cold |

**On the MNI brain the H100 shows 1.0× — no gain at all** (job 56964410), and v3's
sliding window is *slower* (0.91×). That is not a failure of the optimizations; it
is a 189×233×197 volume against a 192³ patch, where tile_step has 4 patches at any
setting and the GPU is fast enough that v3's extra bookkeeping dominates. It is,
however, proof that the brain volume cannot support any speedup claim.

**INT4 vs FP16: DSC 0.9716 agreement, INT4 segments 5.52% fewer voxels**
(1,828,296 vs 1,935,162). INT4 text encoding is 1.5× faster on the forward pass
(0.088s → 0.058s, backbone pre-loaded).

Exact slide wording: *"On one CT case, INT4 agrees with FP16 at DSC 0.97 and
systematically under-segments by 5.5%. Not measured across the validation set."*

Do not soften the direction. A one-sided 5.5% voxel deficit is a bias signature,
not noise — noise would scatter both ways. The n=1 caveat limits how far it
generalizes; it does not make the direction uncertain.

**RTX 4070 SUPER, brain, n=5 verified cold: 1.7× (range 1.6–1.8×)** vs H100 brain
at 1.0×.

⚠️ The obvious explanation — "the H100 is fast enough that fixed overhead dominates
on a small volume" — is a HYPOTHESIS. Nothing measured it. Say it in conversation
if asked; do not put it on a slide as a finding. What is measured is only that the
same image and script give 1.7× on one GPU and 1.0× on the other.


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

## 4. RTX number — RERUN DONE, cite n=9 mean with range

**Superseded**: the April `3.10s → 2.38s = 1.3×` at `make_slides.py:260` and `:564`.

Reran with the corrected script (warmed GPU, verified-cold embed, both arms INT4).
Nine comparable runs on the RTX 4070 SUPER:

**Cite the five verified-cold runs only:**

| | value |
|---|---|
| Comparison A (cold-to-cold, algorithmic) | **1.7× mean, range 1.6–1.8× (n=5)** |
| Comparison B (warm cache) | 1.9× mean, range 1.9–2.0× (n=5) |
| Sliding window alone | 1.91× mean, range 1.83–2.01× (n=5) |

Per-run Comparison A: 1.7, 1.7, 1.6, 1.8, 1.6. An 11% spread — comparable to
nnInteractive's 8.6% and tight enough to be a selling point rather than a caveat.

**Do NOT pool these with the four earlier runs** (1.5, 1.5, 2.0, 1.5). Pooling
gives "1.7× range 1.5–2.0×", which buries the tight spread and takes its upper
bound from an unverified outlier. Only these five assert that the embed cache was
empty before the cold measurement and written after it; the earlier set includes a
run whose "cold" embed read 0.085s — faster than the FP16-matched v0 arm, which a
genuine cold encode should not do. That run produced the 2.0×, and its cause was
never established.

(The assertions are pure checks and change no timing behaviour, so this is not a
script-version difference — it is a measurement-verification difference.)

**Cite the mean with the range, never a point estimate.**

Two caveats that must travel with this number:

1. **Not comparable to April's 2.38s.** `accelerate` was installed partway through,
   so v3 now genuinely runs INT4 where it previously fell back to FP16 silently.
   Different quantity, not a correction of the old one.
2. **Measured on the MNI brain**, where tile_step and Numba both have no room to
   act (see items 6 and 7). The CT run is the number that belongs beside the
   nnInteractive results.

### Superseded analysis (kept so the reasoning is auditable)

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

## 6. "1.4× Numba preprocessing" is a REGRESSION on this volume, not a gain

`make_slides.py:245` lists `Numba preprocessing | @njit(parallel=True) crop +
z-score normalize | 1.4× preprocessing | Unchanged`.

Measured on the RTX with the JIT warmed separately:

| | time |
|---|---|
| numpy (v0_gpu) | 0.075s |
| Numba, first call incl. JIT | 0.121s |
| Numba, warmed | 0.105s |

Numba is ~1.4× **slower**, and only ~16ms of that is JIT compilation — so warming
the JIT does not rescue it. On a 189×233×197 volume the arrays are small enough
that Numba's dispatch overhead exceeds what parallelism buys.

The March 1.4× gain was presumably measured on a larger volume. Either re-measure
on the volume you present, or state the volume size the gain applies to. Do not
show it as an unqualified speedup.

## 7. The VoxTell benchmark image may be unrepresentative of the challenge data

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

## 8. Two different DSC measures — do not present them as one

- nnInteractive: **true DSC vs ground truth** at
  `/scratch/brianx7/cvpr_val/3D_val_gt/3D_val_gt_interactive`, 294 objects.
- VoxTell INT4: **output agreement** DSC(pred_fp16, pred_int4), n=1, no ground truth.

A reviewer will notice these are different quantities. Label them differently on
the slide. Agreement does not establish accuracy — both arms can drift from GT
together and still agree perfectly.
