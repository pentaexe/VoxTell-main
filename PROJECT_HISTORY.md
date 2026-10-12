# Project History

VoxTell and nnInteractive inference optimization, packaging and deployment.

**Author:** Brian Xiao
**Lab:** Peter Munk Cardiac Centre AI Team, under Dr. Jun Ma
**Span:** 11 March 2026 to 9 October 2026, 147 commits
**Repository:** github.com/pentaexe/VoxTell-main

This is the full record: what was asked, what was done, what was found to be
wrong, and what the numbers actually are. Dates and commit hashes come from git
history. Figures come from the reports and result files named against each one.

---

## 1. What the project set out to do

Make two 3D medical image segmentation models run faster on GPU without losing
accuracy, then make them usable by other people.

| Model | Prompt | Role |
|---|---|---|
| VoxTell v1.1 | Free text, e.g. "the spleen" | The lab's CVPR 2026 submission |
| nnInteractive v1.0 | Point, scribble, box, lasso | CVPR 2025 challenge baseline |

Constraint throughout: **inference only.** No retraining, no fine tuning, no
architecture changes. The checkpoint is fixed, so any accuracy change is a bug
in the work rather than a property of the model.

---

## 2. Dr. Ma's requirements, in order

Each one is quoted as given, with what it changed.

### 2.1 Submission standards (standing requirement)

Every result must meet these before it leaves the project:

- A **GPU baseline**, not a CPU one
- **Right sized** cluster allocation
- **Matched precision** between compared arms
- **Explained regressions**, never unexplained ones

**Effect:** this is the origin of the entire measurement audit in section 4. It
is also recorded as a standing rule rather than a one time fix.

### 2.2 CPU efficiency check

Dr. Ma checks `seff` on submitted jobs.

**Effect:** SLURM requests were right sized (commit `628b4e1`, 27 August).
These jobs run at 3 to 20% CPU efficiency because they are GPU bound, so
requesting 8 cores looked wasteful on the report. Settled at 2 cores. The
preflight validator now fails any job requesting more than 4.

### 2.3 On what skills are actually for

> "The goal is not to compare whether the outcome is the same between claude
> code with or without skills. Skills define the workflow, protocol, and when to
> use it, and how to run it... e.g. send an abdomen CT image but ask voxtell to
> segment a brain tumor. Default voxtell may still run it but your skill can
> help generate an answer 'the model cannot segment brain tumor from abdomen ct
> images'."

**Effect:** this redirected the work. The original plan was an A/B comparison of
output quality with and without a skill, which was the wrong question. An
honest finding had already emerged from that attempt: on this repository the
quality gap does not exist, because six test tasks came out correct without the
skill, the documentation carrying the knowledge instead. The real differences
were cost (4 tool calls against 6 to 21, up to 78k tokens) and determinism.

Dr. Ma's abdomen/brain example became the project's central example and is now
the validation gate in section 6.

### 2.4 Build a plugin

> "Please create a plugin that allows to directly use voxtell and nninteractive
> in chatgpt/claude code"

Stated purpose: to understand "the key features of Skills, MCP, and Plugin."

**Effect:** the `voxtell-seg` plugin, section 6. The three mechanisms are
deliberately all present in one package so the boundaries between them are
visible. ChatGPT cannot launch a local subprocess, which is why a second
transport exists.

### 2.5 Publish it and make it installable

> "Could u please push the code to github? Please have a test and make sure
> anyone can use single command/prompt to install voxtell plugin on claude code
> and use it on local laptop. Looks only skill.md is in your skill folder. In
> general, for such complex pipeline, it also has a reference/script folder."

**Effect:** three separate things.

1. A marketplace manifest, so install is two commands
2. `reference/` and `scripts/` directories added to both skills, which had held
   only a `SKILL.md`
3. A clean machine test, which failed twice and is section 7

### 2.6 Test it properly before sending it out

> "No rush, please do more testing before sending it out. I hope there will be
> no issue in the next test."

**Effect:** the regression suite, the build provenance check, and the full
clean clone verification. This instruction directly produced the most serious
find in the project: a clean install was completely broken (section 7.3).

### 2.7 The slide deck

> "How is your presentation slide deck going? We'd expect a comprehensive slide
> deck with 30-50 slides to introduce your work in detail."

**Effect:** the deck went from 8 slides to 51. Section 8.

### 2.8 Cover the architectures

> "please also mention the architectures in voxtell and nninteractive. You'll be
> asked some technical questions during the interview."

**Effect:** twelve architecture slides, read from the checkpoint weights rather
than from the source. That distinction mattered: it caught three wrong figures
(section 8.2).

---

## 3. Timeline

### Phase 1: Optimization (11 March to 9 May)

| Date | Commit | Event |
|---|---|---|
| 11 Mar | `47bb1b3` | First optimized pipeline, reported as 26x |
| 17 Mar | `7752f97` | v0 against v3 comparison |
| 30 Mar | `4a286c0` | CT accuracy validation, with an orientation fix |
| 31 Mar | `0ed39e1` | ONNX export experiment |
| 31 Mar | `4efee71` | torch.compile benchmark |
| 10 Apr | `dc6fadb` | **Fair GPU to GPU benchmark, speedup framing corrected** |
| 19 Apr | `438ed87` | FP16 text backbone fallback fixed to use GPU, not CPU |
| 26 Apr | `e3a4403` | Warmup passes added for accurate H100 timing |
| 27 Apr | `f4290a3` | Speedup reference changed to the GPU baseline |
| 2 May | `12391d9` | Real AMOS accuracy data: DSC 0.8090 / 0.8093 |
| 8 May | `0662a93` | FP16 forced in the H100 baseline to match RTX conditions |

Four optimizations were applied: sliding window overlap (`tile_step` 0.5 to
0.75), a two level embedding cache, an INT4 NF4 text backbone, and Numba
preprocessing. The fourth produced no measurable gain and is reported as a
negative result.

### Phase 2: Cluster work (30 May to 2 June)

Cache behaviour corrected (`a0091e7`), both GPUs measured under identical
conditions (`c8b3473`: one prompt, warm model, no cache).

### Phase 3: nnInteractive (20 June to 26 August)

| Date | Commit | Event |
|---|---|---|
| 20 Jun | `fb15226` | nnInteractive benchmark script |
| 25 Jun | `08534ae` | **Wrong venv found: nninteractive, not miniconda** |
| 26 Jun | `f0ae34b` | PyTorch cache redirected to scratch, home filesystem hangs |
| 27 Jun | `d29fea4` | Confirmed 0.107s per object warm |
| 28 Jun | `8018494` | torch.compile, first reported as 1.58x |
| 14 Aug | `0a3367a` | **fold='all' comparison added, the official checkpoint** |
| 26 Aug | `53bd1f3` | fold='all' is a single model, not an ensemble |
| 26 Aug | `674d35f` | **Settled at 1.33x, n=3, later n=4** |
| 26 Aug | `0e0cdb2` | Break even corrected: 23.61s verified cold start |

The `fold='all'` requirement was the key finding here. Loading `fold=0` gives
about 0.33 DSC and is not a valid baseline, which invalidated the earlier
profiling.

### Phase 4: Skills (23 to 29 August)

| Date | Commit | Event |
|---|---|---|
| 23 Aug | `bac4cab` | First Claude Code skills for both models |
| 29 Aug | `556af09` | **Preflight validator for Fir jobs** |
| 29 Aug | `be27819` | Skill rewritten as a procedure rather than a reference card |
| 26 Aug | `06bd6ff` | Agents removed; skills were the mechanism that worked |

### Phase 5: Plugin (29 August to 8 September)

| Date | Commit | Event |
|---|---|---|
| 29 Aug | `d019317` | **voxtell-seg plugin: skills plus an MCP server with validation** |
| 29 Aug | `005fbd7` | Both branches verified end to end on GPU |
| 30 Aug | `07d4768` | Streamable HTTP transport, for hosted clients |
| 30 Aug | `e10326a` | **Silent MCP failure fixed: userConfig keys need defaults** |
| 30 Aug | `2a5d4ec` | **Two bugs that produced plausible wrong masks** |
| 2 Sep | `81f17a0` | Installable in one command; reference and scripts folders added |
| 5 Sep | `c74cd12` | Two install bugs from a clean Ubuntu machine |
| 5 Sep | `0eb90c2` | **Six bugs found by a new regression suite** |
| 6 Sep | `3b14ff2` | **A clean install was broken, and nnInteractive had never worked** |
| 8 Sep | `500a518` | Two more install bugs, both silent |

### Phase 6: Verification (6 to 15 September)

| Date | Commit | Event |
|---|---|---|
| 6 Sep | `0d90ce8` | Cluster job verifying nnInteractive against real weights |
| 6 Sep | `8fb2f5d` | SLURM could not expand `$USER`; validator now catches it |
| 15 Sep | `a525cdf` | **Thick slice warning, from an independent tester's result** |

### Phase 7: Deck (26 August to 9 October)

| Date | Commit | Event |
|---|---|---|
| 26 Aug | `6ab699c` | First `slides.pptx` and generator |
| 26 Aug | `706c220` | Rebuilt with measured numbers; fabricated figures removed |
| 27 Aug | `a27f41e` | **Misattributed DSC: slide 5 showed nnInteractive's as VoxTell's** |
| 27 Aug | `ce398ba` | Redesigned, 12 slides to 9, plus a speaker script |
| 24 Sep | `ee5f1e8` | **Expanded to 51 slides with architecture sections** |
| 9 Oct | `a87234f` | **query_dim corrected on three slides** |

---

## 4. The measurement audit

This is the most important part of the project.

### 4.1 The arc

| Reported | Why it was wrong |
|---|---|
| **26x** | The baseline ran on CPU |
| **17.6x** | No warm up, and cache state differed between arms |
| **7.1x** | Whichever arm ran first absorbed CUDA start up cost |
| **2.7x** | Measured correctly |

**Not one of these corrections changed the code being measured.** The
optimizations always did exactly what they do. Every drop came from the
benchmark flattering them. A 10x error in the reported result, entirely in the
measurement.

### 4.2 The four artifacts

**Artifact 1: the baseline ran on CPU.** The baseline loaded the text encoder in
FP32. A 4B parameter model at FP32 needs about 16 GB; the card has 12 GB. The
allocation failed and execution fell back to CPU. Nothing errored, the run
completed, and it returned a plausible time. Fixed in `438ed87`.

**Artifact 2: cache state differed.** The optimized arm ran second, after the
baseline had populated the embedding cache, so it read the embedding from memory
while the baseline computed it. That is a real gain on that phase but it is the
cache, not the algorithm. The two claims are now reported separately: 1.8x cold
and 2.0x warm.

**Artifact 3: warm up ordering.** CUDA context creation, cuDNN algorithm
selection and kernel autotuning all happen on first use. The baseline always ran
first and absorbed all of it. On the H100 this alone moved the result from 1.0x
to 7.1x. Fixed in `e3a4403` and `d5b07a9`.

A bug inside the fix is worth recording: the first warm up used a 4x4x4 dummy
volume, which crashed `InstanceNorm3d`, since a single spatial element has no
variance to normalise over. Warm up has to run at the real patch size, 192
cubed, or it is not warming the kernels that actually run (`40589f4`).

**Artifact 4: precision was not held constant.** One arm ran INT4 while the
other ran FP16, so quantization was being counted as algorithmic gain. Both arms
now run INT4 (`bee9e9f`), and the INT4 effect is measured separately.

### 4.3 The rules this produced

1. Hold precision constant, so quantization cannot pose as algorithmic gain
2. Warm the GPU, the text backbone and the sliding window path before timing
3. Assert the embedding cache is empty before every cold measurement
4. Assert the device of every module, rather than assuming a flag took effect
5. Repeat to n >= 4 and quote the range, never a single run
6. Compare an effect against run to run spread before calling it real
7. Never compare across GPUs without saying so

These are enforced by `voxtell-plugin/skills/voxtell-inference/scripts/validate_job.py`,
eleven checks, which refuses to pass a job that breaks them.

---

## 5. Results

### 5.1 VoxTell speed

Phase profile before optimization, one prompt, warm model, no cache:

| Phase | RTX 4070 SUPER | H100 MIG 3g.40gb |
|---|---|---|
| Preprocessing | 0.09s | 0.14s |
| Text embedding | 2.17s (67.3%) | 1.63s |
| Sliding window | 0.94s (29.2%) | 0.39s |
| Postprocessing | 0.02s (0.6%) | 0.11s |
| **Total** | **3.22s** | **2.27s** |

Precision matched benchmark, both arms INT4:

| Case | v0_gpu | v3 cold | v3 warm | Cold | Warm |
|---|---|---|---|---|---|
| Abdominal CT | 2.27s | 1.28s | 1.12s | 1.8x | 2.0x |
| Brain MR | 2.17s | 1.36s | 1.12s | 1.6x | 1.9x |

Sliding window alone: 2.007s to 1.036s (1.94x) abdominal, 1.912s to 1.024s
(1.87x) brain.

**Headline: 2.7x mean, range 2.6 to 2.8x over 4 runs**, abdominal CT on H100
MIG. One run 3.27s to 1.28s. Case `CT_AMOS_amos_0018`, 63x512x512.

Figures including the cache hit, reported separately because they are a
different claim: 3.5x on RTX (3.22s to 0.93s), 4.1x on H100 (2.27s to 0.55s).

### 5.2 VoxTell accuracy

65 objects, 5 AMOS cases, 13 abdominal organs, seed 42:

| Config | Mean DSC | Median | Min | Mean NSD |
|---|---|---|---|---|
| v0, tile_step 0.5 | 0.8090 | 0.8775 | 0.1840 | 0.8129 |
| v3, tile_step 0.75 | 0.8093 | 0.8775 | 0.1840 | 0.8135 |
| Difference | **+0.0003** | 0 | 0 | +0.0006 |

Per organ, v0 to v3: liver 0.9667 to 0.9671, left kidney 0.9431 to 0.9431,
right kidney 0.9393 to 0.9397, spleen 0.9356 to 0.9355, aorta 0.9207 to 0.9225,
stomach 0.8934 to 0.8924, gallbladder 0.8264 to 0.8264, pancreas 0.7763 to
0.7699, duodenum 0.7240 to 0.7254, inferior vena cava 0.7174 to 0.7190, left
adrenal 0.6670 to 0.6646, esophagus 0.6207 to 0.6208, right adrenal 0.5866 to
0.5948.

**INT4 against FP16:** 0.97 output agreement, segments 5.5% fewer voxels, one
sided so bias rather than noise. n=1, and it measures agreement rather than
accuracy against ground truth.

### 5.3 nnInteractive

Job `56908464`, fold='all', autozoom on, 20 CT cases, 294 objects:

| Setting | `_predict` warm | Per case | Speedup | Mean DSC |
|---|---|---|---|---|
| Baseline | 0.2882s | 4.38s | 1.0x | 0.7914 |
| torch.compile | 0.2146s | 3.29s | 1.34x | 0.7916 |
| Difference | | | | **+0.0002** |

**Headline: 1.33x, range 1.28 to 1.39x over 4 runs.** The 1.34x above is this
single job; 1.33x is the mean of four.

Break even: 23.6s one time compile, 0.071s saved per object, about 22 cases.
This is a batch optimization and should not be sold as anything else. A batch of
881 validation cases clearly earns it back; a radiologist with three scans never
does.

CPU time held at 6:18 to 6:33 across runs while walltime fell, which is what a
GPU bound workload looks like. CPU efficiency 13 to 20%.

### 5.4 Negative results

| Approach | Outcome | Reason |
|---|---|---|
| ONNX plus ORT CUDA | 14x **slower** | ORT lacks cuDNN 3D convolution kernels |
| torch.compile on VoxTell | 1.00x, no change | Compute bound; Triton unavailable on Windows |
| Numba preprocessing | 0.09s to 0.09s | The phase is under 3% of runtime |
| Batched sliding window | Built, no payoff yet | batch_size=1; needs 80 GB to matter |

torch.compile working on nnInteractive at 1.33x and not at all on VoxTell is
worth more than either number alone. VoxTell is bound by 3D convolution
throughput, which compilation does not change. nnInteractive makes many small
calls per object, where dispatch overhead is the cost.

### 5.5 Results through the plugin

**VoxTell, FLARE abdominal CT, prompt "the liver":** DSC **0.9757**, recall
0.9851, precision 0.9665. 5,117,747 predicted against 5,021,229 ground truth
voxels.

**VoxTell, MNI T1, prompt "brain":** 1,829,613 voxels against the 1,828,296
reference from the INT4 comparison, 0.07% apart.

**nnInteractive, job `58439379`**, fold='all' on H100 MIG, driven through the
MCP server:

| Case | Box | DSC |
|---|---|---|
| CT_AMOS_amos_0018 | 0 | 0.9617 |
| CT_AMOS_amos_0018 | 1 | 0.9045 |
| CT_AMOS_amos_0021 | 0 | 0.9563 |
| CT_AMOS_amos_0021 | 1 | 0.9339 |
| | **Mean** | **0.9391** |

A smoke test on large abdominal organs, **not** comparable with the 0.79 full
set score, which spans small and difficult structures this sample avoids.

### 5.6 Independent accuracy check

Run by Keishi Suzuki on five MSD-Liver cases from `Dataset004_LiverCancer`,
prompt `liver tumor`:

| Case | z-spacing | Slices | DSC | Recall | Precision |
|---|---|---|---|---|---|
| MSD-Liver_000 | 5.0 mm | 75 | 0.479 | 0.371 | 0.676 |
| MSD-Liver_001 | 5.0 mm | 123 | 0.520 | 0.579 | 0.472 |
| MSD-Liver_003 | 1.0 mm | 534 | 0.861 | 0.815 | 0.914 |
| MSD-Liver_004 | 0.8 mm | 841 | 0.868 | 0.939 | 0.807 |
| MSD-Liver_002 | 1.0 mm | 517 | 0.872 | 0.858 | 0.886 |

Mean 0.720, median 0.861. The split is by slice thickness and nothing overlaps.
See section 7.5.

---

## 6. The plugin

`voxtell-plugin/`, installable from a GitHub marketplace in two commands.

### 6.1 Why it exists

VoxTell is text prompted, so it accepts any string. Send an abdominal CT and ask
for a brain tumour and it runs, returns a mask, and nothing says the mask is of
nothing. This is Dr. Ma's example from section 2.3.

### 6.2 The three mechanisms

| Mechanism | What it is | What it can do |
|---|---|---|
| **Skill** | Instructions loaded into the model's context | Shapes workflow, protocol, standards. Advisory |
| **MCP server** | A process exposing callable tools over JSON-RPC | Real code, real return values. It can refuse |
| **Plugin** | A package bundling skills, MCP servers, commands, hooks | One installable, versioned unit |

The distinction that matters: a skill can **tell** the model that a brain prompt
on an abdominal CT is wrong. Only a tool can make the call **fail**. One is
guidance the model may or may not follow, the other is a gate it cannot walk
past.

### 6.3 The five tools

| Tool | Purpose |
|---|---|
| `check_request` | Validate an image and prompt pair. No compute |
| `voxtell_segment` | Text prompted segmentation, validates first |
| `nninteractive_segment` | Box prompted, with box validation before the GPU check |
| `list_models` | What is available and what each cannot do |
| `setup` | Check the machine, report the live build, fetch the checkpoint |

Written against the MCP wire protocol directly with no SDK dependency. Two
transports from one routing path: stdio for Claude Code, Streamable HTTP for
hosted clients that cannot launch a local subprocess.

### 6.4 How validation decides

Intensity statistics and geometry, not a classifier, because a validator nobody
can audit is a validator nobody should trust.

- **Modality** from the intensity distribution. CT is calibrated in Hounsfield
  units so air sits near -1000; MR intensities are arbitrary and rarely negative
- **Region** from air fraction and field of view. A head has almost no internal
  air; a torso has lungs or bowel gas
- **Prompt** against an anatomy vocabulary mapping terms to regions

It refuses only on a clear mismatch. Unknown on either side proceeds with a
note, because a validator that blocks whatever it cannot classify is useless.
Adjacent regions pass, since thorax and abdomen routinely share a field of view.

---

## 7. Bugs found

Grouped by what made each one hard to see. The recurring pattern is a run that
completes, looks right, and is wrong.

### 7.1 Silent wrong answers

| Bug | Effect | Why it hid |
|---|---|---|
| Reader orientation | DSC 0.2978 instead of 0.9864 | Plain `nib.load().get_fdata()` leaves the stored orientation. The model segments a plausible fragment |
| Crop not reinserted | Mask in cropped space | `preprocess()` crops to the non-zero box; logits came back in that space |
| FP32 CPU fallback | 26x speedup claim | Allocation failed, PyTorch fell back silently |
| INT4 serving FP16 | Mislabelled precision | `_load_text_backbone` catches every exception and falls back |
| `fold=0` instead of `fold='all'` | About 0.33 DSC | Loads and runs normally |

### 7.2 Plugin infrastructure

| Bug | Effect |
|---|---|
| `userConfig` key without a `default` | MCP server entry invalid, **skipped silently**. Plugin validates, loads, has no tools, nothing in `--debug` |
| `print()` to stdout in the predictor | Corrupted the JSON-RPC stream. One client tolerated it |
| 5D array to `set_image` | `nninteractive_segment` had **never once run**. `load_image` returns `(C,H,W,D)` so `arr[None]` made it 5D |
| `output_dir` accepted and ignored | Masks written somewhere else than asked |
| Machine specific `.mcp.json` committed | Would fire with another machine's paths on any clone |
| Raw buffer saved for bbox masks | Mask would not overlay on the input |

### 7.3 Packaging, found by a clean install

The most serious group. Found only because Dr. Ma asked for more testing.

| Bug | Effect |
|---|---|
| **`numba` undeclared** | `import voxtell.inference.predictor` raised on any clean machine. **Segmentation could not run at all** |
| `accelerate`, `bitsandbytes` undeclared | Install passed every check, reported itself optimized, served FP16 |
| `batchgenerators` undeclared | Worked only by arriving with nnunetv2 |
| Same name and version as PyPI | `pip show voxtell` could not tell fork from stock. Now `0.1.0+optimized` |
| Unquoted `torch` constraint | `pip install torch --index-url .../cu126` took 2.14.0, violating `torch<2.9`, so `pip install -e .` re-resolved from PyPI and dropped `+cu126` |
| No `requires-python` upper bound | On Python 3.14, pip failed inside torch resolution without naming the interpreter |

`numba` was the one that mattered. It worked locally only because that machine
had numba installed for something unrelated.

### 7.4 Diagnosis quality

Two fixes were about error messages rather than behaviour, because a wrong
diagnosis costs as much as a bug.

- A missing dependency made the build report itself as **stock**, pointing at a
  reinstall that would have changed nothing. Provenance now separates "imports
  but lacks the marker" from "will not import"
- `doctor.py` classified by path prefix, so it called a working install "stock
  PyPI build" when run from a different checkout. Switched to feature detection

### 7.5 Tooling that passed broken work

| Bug | Effect |
|---|---|
| `validate_job.py` crashed on Windows | Printed an arrow to a cp1252 console, raised `UnicodeEncodeError` on the first line, produced **no output at all** |
| `validate_job.py` pinned one account | Reported FAIL for every other user regardless of their job |
| `$USER` in an `#SBATCH` directive | SLURM does not expand it. Job `58438222` died at submission with no log. The validator passed the script, checking the path's shape but never whether it could resolve. It now fails any shell variable in a directive |

### 7.6 The thick slice finding

Not a bug, a characterisation. Keishi's five case result in section 5.6 splits
perfectly by slice thickness, and there is a mechanism.

VoxTell v1.1 does not resample. Its plans file says
`nnUNetResEncUNetLPlans_noResampling_3d_fullres`, and the inference path has no
resampling step, so the network sees voxels at the acquired spacing through a
fixed 192 cubed patch. At 1 mm that patch covers about 192 mm of anatomy; at
5 mm it covers 960 mm, longer than a torso. A 15 mm lesion is 15 slices at 1 mm
and 3 at 5 mm, so boundary partial volume dominates.

`check_request` and `voxtell_segment` now attach a caveat at 3 mm or coarser,
before any GPU time is spent. A caveat and not a refusal: the mask is still
useful on a large target. 3 mm is conservative rather than measured, since
nothing in the sample sits between 1 mm and 5 mm.

One hypothesis of mine was wrong here and is recorded for the sake of accuracy:
I suggested the low scores came from an empty ground truth mask giving 0/0.
Keishi checked. No case was empty and the DSC formula was correct. The real
explanation came out of the per case table.

---

## 8. The deck

`slides.pptx`, generated by `make_slides.py` so edits survive regeneration.

### 8.1 History

8 slides for a short hand in, then 51 after section 2.7. Seven parts: context,
architecture, optimization, measurement, results, deployment, limits.

Earlier deck fixes worth recording: a misattributed DSC showed nnInteractive's
score as VoxTell's (`a27f41e`); fabricated and superseded figures were removed
in a rebuild (`706c220`); a "46.7x bug fix" row was deleted because it was never
an optimization (`e0eca80`).

### 8.2 Architecture figures, and three that were wrong

The architecture sections were read from `models/voxtell_v1.1/plans.json` and
`voxtell/model/*.py`. Three figures were then found wrong because they had been
taken from **constructor defaults rather than the checkpoint**:

| Figure | Slides said | Checkpoint says |
|---|---|---|
| `query_dim` / `d_model` | 864 | **2048** |
| MaskFormer `num_heads` | 1 | **32** |
| Text embedding width | 2560 | 2560, correct |

Evidence from `fold_0/checkpoint_final.pth`:

```
project_text_embed.0.weight             (2048, 2560)     text embedding is 2560
project_text_embed.2.weight             (2048, 2048)     query_dim is 2048
pos_embed                               (1728, 1, 2048)  1728 = 12 cubed, stage 4
project_to_decoder_channels.4.2.weight  (10240, 2048)    320 x 32 heads
```

Verified correct: 6 transformer layers, 2048 feedforward, stage 4 at 320
channels and 12 cubed, 5 mask-former stages, and the full encoder table (6
stages, channels 32/64/128/256/320/320, blocks 1/3/4/6/6/6, 32x downsampling).

**Constructor defaults are not the configuration. The weights are.** This is the
class of error Dr. Ma's note in section 2.8 was warning about.

### 8.3 Verification

Checked after generation rather than assumed: 51 slides, every printed page
number matches its index, no shape extends past the slide edge, and nothing
estimated to overflow or collide by more than 0.05 inches.

---

## 9. What exists now

| Deliverable | Location |
|---|---|
| Optimized inference pipeline | `voxtell/inference/predictor.py` |
| Optimization reports | `OPTIMIZATION_REPORT.md`, `OPTIMIZATION_REPORT_COMPACT.md`, `NNINTERACTIVE_OPTIMIZATION_REPORT.md` |
| Benchmark and accuracy scripts | `fair_benchmark.py`, `accuracy_eval.py`, `int4_dsc_comparison.py`, `nni_*.py` |
| Preflight validator | `voxtell-plugin/skills/voxtell-inference/scripts/validate_job.py` |
| Two Claude Code skills | `voxtell-plugin/skills/`, each with `reference/` and `scripts/` |
| MCP server, five tools, two transports | `voxtell-plugin/mcp_server/` |
| Marketplace manifest | `.claude-plugin/marketplace.json` |
| Regression suite | `voxtell-plugin/tests/test_server.py`, 32 checks, synthesises its own volumes |
| Cluster verification job | `nni_mcp_verify.py`, `nni_mcp_verify.sh` |
| Slide deck and generator | `slides.pptx`, `make_slides.py` |
| This record | `PROJECT_HISTORY.md` |

Verified across both platforms: clean clone into an empty venv, marketplace
install, configuration, a connected server with five tools, the 1.7 GB
checkpoint download, and real segmentations scored against ground truth. Windows
by me, Ubuntu by Keishi.

---

## 10. Still open

- VoxTell speedup is measured on a single CT volume at n=4 runs. It is not a
  distribution
- The INT4 agreement figure is n=1 and measures output agreement, not accuracy
  against ground truth
- The thick slice finding is n=5 with a plausible mechanism, which makes it a
  strong hypothesis rather than a result. Where the falloff begins between 1 mm
  and 5 mm is unmeasured
- DSC scales with target size, and the five liver cases ranged over 543x in
  ground truth volume, so their mean is not a clean summary
- The region heuristic is thresholds on one CT convention, untested on MR beyond
  the modality check
- The batched sliding window is built but has not paid off; it needs 80 GB
- TensorRT FP16 on Fir is queued and not yet run
- The HTTP transport has no authentication and must not be exposed without an
  auth layer in front
- `SPEAKER_NOTES.md` is still written for the earlier 9 slide flow
- The deck is 51 slides against a stated range of 30 to 50

---

## 11. The thing worth taking from it

Both models got faster without losing accuracy, 2.7x and 1.33x, each with a DSC
measurement attached. But the reported speedup fell by 10x under scrutiny, and
not one correction changed the code.

The same failure shape kept recurring: a run that completes, looks right, and is
wrong. A CPU baseline posing as a GPU one. A cache hit posing as an algorithm. A
reader orientation that segments the wrong thing plausibly. An INT4 label on an
FP16 run. A plugin that installs and has no tools. A 5D array to a function
expecting 4D. A clean install that cannot import the model at all.

So the durable output is not the speedup. It is the checks that make those
visible: a preflight validator, a regression suite that synthesises its own
data, a build provenance check, and two gates that refuse work before it is
wasted. Those outlast any particular number.
