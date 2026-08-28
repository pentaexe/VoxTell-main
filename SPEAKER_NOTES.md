# Speaker script — VoxTell & nnInteractive optimization

9 slides, ~9–11 minutes at a normal pace. Written to be spoken, not read aloud
verbatim — the phrasing is there so you have a line ready if you blank.

Every figure quoted here traces to a source; the "if asked" blocks are the
follow-ups most likely to come.

---

## 1 · Title  (15 sec)

> "This is work on making two medical image segmentation models run faster on
> the H100 nodes — our lab's model VoxTell, and nnInteractive, which is the
> CVPR challenge baseline."

Don't linger. Move.

---

## 2 · Two models, two prompting styles  (60 sec)

> "The two models solve the same problem but take prompts differently, and that
> difference drives everything about their cost.
>
> VoxTell takes free text — you type 'the spleen' and it segments the spleen.
> That's the useful part, but it means a four-billion-parameter language model
> has to encode your text before segmentation even starts.
>
> nnInteractive takes a 3-D bounding box instead. No text encoder at all, so it
> starts from a much cheaper place per prompt.
>
> Both were already accurate. My question was whether they could be made faster
> without giving that up."

**If asked what DSC is:** overlap between the predicted mask and the expert
annotation, 0 to 1. Above roughly 0.7 is usually considered clinically usable.

---

## 3 · The first number I reported was wrong  (90 sec) — *the important slide*

Slow down here. This is the slide that earns credibility.

> "I want to start with the thing I got wrong, because it's the most useful part
> of this work.
>
> The first speedup I reported for VoxTell was 26×. That was measured against a
> baseline accidentally running on the CPU — the text encoder was loaded in
> FP32, overflowed the card's memory, and PyTorch silently fell back to CPU. So
> I was comparing a GPU run against a CPU run and calling it a speedup.
>
> I fixed that and got 17.6×. That was still wrong: the optimized side was
> reading from a warm cache while the baseline wasn't, and neither side was
> warmed up.
>
> Fixed that, got 7.1×. Still wrong — whichever arm ran first absorbed all the
> CUDA and kernel start-up cost, and the baseline always ran first.
>
> Once all of that was controlled, the real number is 2.7×.
>
> The part I'd emphasize: not one of those corrections changed the code being
> measured. The optimizations always did exactly what they do. Every drop came
> from the benchmark flattering them."

**If asked how you found them:** mostly by running the same script on two
different GPUs and noticing phases that should behave the same didn't. Covered
on slide 8.

**If asked "how do you know 2.7 is right":** I don't know it's final — I know
it survives every check I've been able to think of. Four repeats, precision held
constant, everything warmed, cache asserted empty.

---

## 4 · What actually made it faster  (75 sec)

> "Four changes. The one that matters is the sliding window: raising the tile
> step and cropping to the non-zero region takes a CT volume from 25 patches to
> 9. That's most of the 2.7×.
>
> The embedding cache is a large win on repeated prompts and does nothing on a
> first query — I report it separately rather than folding it into the headline.
>
> INT4 quantization on the text backbone: 1.5× faster encoding, and it drops the
> model from 8 gigabytes of VRAM to 2.
>
> And Numba preprocessing, which I'll be honest about — it showed no measurable
> gain at this volume size. It's in the table because I measured it, not because
> it worked."

**If asked why keep Numba in:** because removing a row I measured and disliked
is how you end up with a table nobody can check. It may pay off on larger
volumes; I haven't tested that.

**Key point if pushed on fairness:** both arms run INT4. Precision is held
constant, so this is algorithmic gain only — no quantization credit hidden in it.

---

## 5 · VoxTell — accuracy held  (60 sec)

> "The speed optimizations don't cost accuracy — DSC moves by three ten-thousandths
> across 65 objects in 5 CT cases.
>
> The caveat is on the right, and I want to state it plainly. INT4 quantization
> is on by default. On one CT case it agreed with full precision at 0.97 DSC, and
> it segmented five and a half percent fewer voxels. That's a consistent
> under-segmentation in one direction, which is a bias signature — noise would
> scatter both ways.
>
> I have not measured that across the validation set. So I'm reporting the
> direction, not a bound."

**If asked whether that's acceptable:** unknown, and that's the honest answer.
For a 5% voxel deficit on organ boundaries, whether it matters depends on the
downstream use. It needs the full-set measurement before anyone relies on it.

---

## 6 · nnInteractive — compiling the network  (50 sec)

> "nnInteractive was a much smaller change — one line, wrapping the network in
> torch.compile with reduce-overhead mode. It fuses kernels and cuts per-call
> dispatch cost.
>
> That gives 1.33× per object, ranging 1.28 to 1.39 across four runs. Accuracy
> moves by two ten-thousandths, and no run showed degradation."

**If asked why report a range:** because a single run of this landed at 1.39×
and another at 1.28×. Quoting either alone would be picking a number.

---

## 7 · The speedup is not free  (60 sec)

> "There's a catch worth being upfront about. Compiling costs 23.6 seconds before
> the first prediction, and you save 0.071 seconds per object. So you need about
> 331 objects — roughly 22 cases — before compilation pays for itself.
>
> For a batch job over 900 validation cases, clearly worth it. For a radiologist
> segmenting three scans in a session, you never get it back. It's a batch
> optimization, and it shouldn't be sold as anything else."

**If asked about the 23.6s:** measured on node-local storage. On the shared
filesystem it'll be slower, so treat it as a lower bound.

---

## 8 · How I checked the numbers  (75 sec)

> "Four things, and they're all reactions to a specific way I'd been fooled.
>
> Hold precision constant — both arms run INT4, so a quantization gain can't
> masquerade as an algorithmic one.
>
> Warm everything before timing — GPU, text backbone, sliding window path.
> Otherwise whichever arm runs first eats the start-up cost, which is exactly the
> bug behind the 7.1×.
>
> Prove the cache is empty. The script asserts the embedding cache is cleared
> before each cold measurement and written afterward, so a cache hit can't be
> reported as a cold encode.
>
> And repeat everything, quoting the range.
>
> The line at the bottom is the one I'd actually recommend to someone else:
> running the same benchmark on two different GPUs is what exposed the largest
> error. Phases that should have behaved identically didn't, and that discrepancy
> is what unravelled it."

---

## 9 · Where both models landed  (45 sec)

> "VoxTell 2.7×, nnInteractive 1.33×, both with accuracy held and both backed by
> four runs quoted with their spread.
>
> Two things are still open. INT4's accuracy effect is measured on one case, not
> the validation set. And nnInteractive's compile gain is close to the
> run-to-run spread on small batches, so on short jobs I wouldn't claim it.
>
> The next thing I'd run is the INT4 comparison across all 881 validation cases —
> that turns a directional finding into something you could actually act on.
>
> If I had to keep one thing from this project, it wouldn't be either speedup.
> It's that I ended up with a benchmark that kept catching itself, and I'd rather
> report 2.7× that survives scrutiny than 26× that doesn't."

Stop there. Don't add anything after that line.

---

## Numbers, and where each comes from

| Figure | Source |
|---|---|
| 2.7× (2.6–2.8×), n=4 | jobs 56964411, 56966901–903 |
| 3.27s → 1.28s | job 56964411, `CT_AMOS_amos_0018.npz` |
| 25 → 9 patches | same job |
| VoxTell 0.8090 → 0.8093 | `accuracy_results.csv`, 65 objects / 5 cases |
| INT4 0.9716, −5.5% voxels | job 56964412, n=1 |
| 1.33× (1.28–1.39×), n=4 | jobs 56908464, 56923894–896 |
| nnInteractive +0.0002 DSC | job 56908464, 294 objects |
| 23.6s compile, 0.0714s/object | job 56914757 + n=4 mean |
| ~331 objects / ~22 cases | 23.61 ÷ 0.0714, at 14.7 obj/case |

## Questions you should expect

**"Why is VoxTell measured on 5 CT cases and nnInteractive on 20?"**
Different experiments run at different times. The VoxTell accuracy set is
smaller than I'd like — that's a real limitation, not a defence.

**"Is 2.7× worth the complexity?"**
The sliding-window change is most of it and is a two-line configuration change.
The cache matters for repeated prompts. INT4 needs the accuracy question settled
before I'd argue for it.

**"What would you do differently?"**
Warm the GPU and assert the cache state from the very first benchmark. Three of
my four wrong numbers came from not doing that, and each took a day to find.
