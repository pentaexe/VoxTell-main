# Measurement rules

Why these exist: the headline speedup on this project fell **26× → 17.6× → 7.1× →
2.7×**. Not one of those corrections changed the code being measured. Every drop
removed a way the benchmark was flattering the result.

Each rule below prevents a specific artifact that has already produced a wrong
number here.

---

## 1. Hold precision constant

VoxTell's predictor uses INT4 (NF4) by default. A hand-built FP16 baseline makes
quantization look like an algorithmic gain.

**Symptom when violated:** a speedup that shrinks when you match precision.
**Cost when it happened:** 17.6× reported, real figure far lower.

## 2. Verify the quantization actually loaded

`_load_text_backbone` in `voxtell/inference/predictor.py` catches *any* exception
and falls back to FP16 while the log still prints INT4.

```python
assert predictor._backbone_quantized, "backbone fell back to FP16"
```

A missing `accelerate` is the usual cause, and it fails silently.

## 3. Warm everything before timing

GPU, text backbone, **and** the sliding-window path. Whichever arm runs first
otherwise absorbs CUDA context creation and cuDNN autotuning.

**Cost when it happened:** an H100 result read 7.1×. With warmup it read 1.0×.
The entire gap was start-up cost landing in one arm.

Warm at the real `patch_size`. A small dummy tensor downsamples to 1×1×1 and
`InstanceNorm3d` raises, because it computes instance statistics even under
`.eval()`.

## 4. Prove the embedding cache is cold

Deleting the entry is not proof. Assert it.

```python
for p in PROMPTS:
    d = _prompt_cache_path(p, TEXT_MODEL)
    if d.exists(): d.unlink()
predictor._embed_cache.clear()
for p in PROMPTS:
    assert not _prompt_cache_path(p, TEXT_MODEL).exists()
assert not predictor._embed_cache
# ... time the encode ...
for p in PROMPTS:
    assert _prompt_cache_path(p, TEXT_MODEL).exists(), "no encode actually ran"
```

Also check the key: the cache is `sha256(f"{model}::{prompt}")` and the predictor
reads it with `self.text_encoding_model`. Clearing a different key means the
"cold" measurement is a cache hit.

## 5. Use a representative volume

The MNI brain is 189×233×197 against a 192³ patch: **4 patches at any
`tile_step`**, and arrays too small for Numba to beat numpy. Neither optimization
can act, so the measurement says nothing about either.

Use `/scratch/brianx7/cvpr_val/3D_val_npz/CT_*.npz`. Those carry a
`text_prompts` key, so read the prompt from the file rather than guessing the
anatomy.

**Measured contrast:** on CT, 25 patches → 9. On the brain, 4 → 4.

## 6. Repeat to n≥4 and quote the range

Two runs of the same script have disagreed by 33%. Serialize repeats so they do
not share thermal state:

```bash
J1=$(sbatch --parsable job.sh)
J2=$(sbatch --parsable --dependency=afterany:$J1 job.sh)
J3=$(sbatch --parsable --dependency=afterany:$J2 job.sh)
```

Report mean and range. Never a bare point estimate.

## 7. Cross-check on a second GPU when a number surprises you

Running the same script on an RTX 4070 SUPER and an H100 is what exposed the
ordering artifact: identical phases behaved differently at matched precision and
identical patch counts. That discrepancy is what unravelled 7.1×.

---

## Reading the result

**Attribute precisely.** `accuracy_results.csv` has two arms, `v0 (step=0.5)` and
`v3 (step=0.75)`, differing by `tile_step` alone. So its +0.0003 DSC is the
tile_step effect, not the whole optimized stack. INT4's effect is measured
separately and was never folded in.

**Do not cite** 26×, 17.6×, 7.1×, 46.7×, `3.10s → 2.38s`, `343 → 125 patches`, or
`1.4× Numba preprocessing`. All are superseded or were artifacts. Several still
appear in `experiment_log.md` and `OPTIMIZATION_REPORT.md` with nothing marking
them as retracted.
