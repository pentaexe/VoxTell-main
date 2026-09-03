# nnInteractive checkpoint and session rules

## Always use fold='all'

`use_fold='all'` is the official CVPR 2025 checkpoint, DSC ~0.79.
`use_fold=0` gives ~0.33 and is **not a valid comparison baseline** — the fold_0
directory holds undertrained or mismatched weights. Numbers from `nni_dsc.py`
and `nni_batch_dsc.py` were produced with fold=0; do not cite them.

## torch_n_threads must be os.cpu_count()

Using `SLURM_CPUS_PER_TASK` instead hangs with `fold='all'`. This is not
documented anywhere upstream; it was found by a job that never returned.

## Session pattern

```python
session = nnInteractiveInferenceSession(
    device=torch.device('cuda', 0),
    use_torch_compile=False,
    verbose=False,
    torch_n_threads=os.cpu_count(),   # not SLURM_CPUS_PER_TASK
    do_autozoom=True,
    use_pinned_memory=True,
)
session.initialize_from_trained_model_folder(CHECKPOINT_DIR, use_fold='all')
# to enable compilation:
# session.network = torch.compile(session.network, mode='reduce-overhead')
```

Per object:

```python
session.reset_interactions()
session.add_bbox_interaction(bbox, include_interaction=True, run_prediction=False)
session.new_interaction_centers          = [session.new_interaction_centers[-1]]
session.new_interaction_zoom_out_factors = [session.new_interaction_zoom_out_factors[-1]]
session._predict()
```

## torch.compile

- `N_WARMUP=2`. The first warmup triggers Triton compilation, the second
  stabilises dispatch. With only one, compiled latency reads ~0.54s and the
  speedup appears to be 1.0×.
- Set `TORCHINDUCTOR_CACHE_DIR` **before** `import torch`; inductor reads it at
  import time, so setting it in the `.sh` is too late.

## Measured results (n=4, fold='all', autozoom=ON, H100 MIG 3g.40gb)

| Job | Baseline | Compiled | Gain/obj | Speedup | DSC Δ |
|---|---|---|---|---|---|
| 56908464 | 0.2882s | 0.2146s | 0.0736s | 1.34× | +0.0002 |
| 56923894 | 0.2881s | 0.2068s | 0.0813s | 1.39× | +0.0000 |
| 56923895 | 0.2722s | 0.2128s | 0.0594s | 1.28× | +0.0004 |
| 56923896 | 0.2947s | 0.2236s | 0.0711s | 1.32× | +0.0002 |
| **Mean** | | | **0.0714s** | **1.33×** | **≤ +0.0004** |

Cite **1.33× mean, range 1.28–1.39× (n=4)**. Never a single job.

The gain is 33% against an 8% run-to-run spread, roughly a 4× ratio, which is
why this result held up under repetition.

## Cold start and break-even

Triton compilation costs **23.61s** (job 56914757, `/tmp`-backed and fully
isolated; the shared filesystem will be slower, so treat it as a lower bound).

Break-even: 23.61 ÷ 0.0714 = **~331 objects, about 22 cases** at 14.7 objects per
case. Worth it for a batch of 881 validation cases; never recovered by a
clinician segmenting three scans.

## Reporting DSC

At `N_CASES = 20` (294 objects) a delta of +0.0002 sits inside sampling noise.
The defensible claim is that four runs bracket zero, not that any single run
proves equivalence. Do not invoke a 0.005 pass/fail threshold — it is
self-assigned, not from the challenge.

## autozoom

Adds no measurable overhead on this validation set, because no object triggered
a zoom-out pass. That may not hold on data where anatomy crosses bbox boundaries.
