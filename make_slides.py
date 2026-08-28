"""
Generate slides.pptx — VoxTell & nnInteractive optimization deck.
Run: python make_slides.py

Design: minimal. White ground, ink text, one amber accent used sparingly,
hairline rules instead of filled boxes. One idea per slide.
Every figure on these slides traces to a measured source — see SPEAKER_NOTES.md.
"""
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

# ── Palette ────────────────────────────────────────────────────────────────
INK    = RGBColor(0x1A, 0x1F, 0x26)   # body text
ACCENT = RGBColor(0xB5, 0x7E, 0x1F)   # amber, used sparingly
TEAL   = RGBColor(0x1D, 0x6F, 0x69)   # second series only
MUTED  = RGBColor(0x8A, 0x91, 0x9B)   # captions, sources
RULE   = RGBColor(0xDD, 0xDA, 0xD3)   # hairlines
BG     = RGBColor(0xFC, 0xFC, 0xFA)
WHITE  = RGBColor(0xFF, 0xFF, 0xFF)

L      = 0.9      # left margin
CW     = 11.5     # content width

prs = Presentation()
prs.slide_width  = Inches(13.33)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]


def slide(bg=BG):
    s = prs.slides.add_slide(BLANK)
    f = s.background.fill
    f.solid()
    f.fore_color.rgb = bg
    return s


def tx(s, text, l, t, w, h, size=13, bold=False, color=INK,
       align=PP_ALIGN.LEFT, space=None):
    box = s.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
    box.word_wrap = True
    tf = box.text_frame
    tf.word_wrap = True
    for i, line in enumerate(text.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        if space:
            p.space_after = Pt(space)
        r = p.add_run()
        r.text = line
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.name = "Calibri"
        r.font.color.rgb = color
    return box


def rule(s, l, t, w, color=RULE, thick=1):
    sh = s.shapes.add_shape(1, Inches(l), Inches(t), Inches(w), Inches(thick / 72))
    sh.fill.solid()
    sh.fill.fore_color.rgb = color
    sh.line.fill.background()
    return sh


def head(s, title, num):
    tx(s, title, L, 0.55, 9.8, 0.6, size=26, bold=True)
    tx(s, num, 11.0, 0.68, 1.4, 0.4, size=10, color=MUTED, align=PP_ALIGN.RIGHT)
    rule(s, L, 1.25, CW)


def stat(s, value, label, l, t, w, vsize=52, color=INK):
    """A large number with a small label beneath it."""
    tx(s, value, l, t, w, 0.9, size=vsize, bold=True, color=color)
    tx(s, label, l, t + 0.95, w, 0.5, size=10, color=MUTED)


def table(s, headers, rows, l, t, w, h, widths=None):
    n = len(rows) + 1
    tbl = s.shapes.add_table(n, len(headers), Inches(l), Inches(t),
                             Inches(w), Inches(h)).table
    if widths:
        for i, ww in enumerate(widths):
            tbl.columns[i].width = Inches(ww)
    for c, htxt in enumerate(headers):
        cell = tbl.cell(0, c)
        cell.fill.solid()
        cell.fill.fore_color.rgb = BG
        p = cell.text_frame.paragraphs[0]
        r = p.add_run(); r.text = htxt
        r.font.size = Pt(10); r.font.bold = True
        r.font.name = "Calibri"; r.font.color.rgb = MUTED
    for ri, row in enumerate(rows, start=1):
        for ci, val in enumerate(row):
            cell = tbl.cell(ri, ci)
            cell.fill.solid()
            cell.fill.fore_color.rgb = WHITE if ri % 2 else BG
            p = cell.text_frame.paragraphs[0]
            r = p.add_run(); r.text = str(val)
            r.font.size = Pt(11)
            r.font.name = "Calibri"
            r.font.color.rgb = INK
            r.font.bold = (ci == 0)
    return tbl


def source(s, text):
    tx(s, text, L, 6.85, CW, 0.4, size=9, color=MUTED)


# ═══ 1 — Title ═════════════════════════════════════════════════════════════
s = slide(RGBColor(0x14, 0x1A, 0x22))
tx(s, "CVPR 2025 · Medical Image Segmentation", L, 2.15, 9, 0.4,
   size=11, color=RGBColor(0xB5, 0x7E, 0x1F))
tx(s, "Making Two Segmentation\nModels Faster", L, 2.65, 11, 1.8,
   size=42, bold=True, color=RGBColor(0xF4, 0xF2, 0xEE))
rule(s, L, 4.65, 1.2, ACCENT, thick=3)
tx(s, "VoxTell  ·  nnInteractive  ·  NVIDIA H100 MIG", L, 4.95, 10, 0.4,
   size=14, color=RGBColor(0x9A, 0xA3, 0xAF))
tx(s, "Brian Xiao   ·   Fir cluster, Alliance Canada   ·   August 2026",
   L, 6.6, 10, 0.4, size=10, color=RGBColor(0x6E, 0x77, 0x84))

# ═══ 2 — The two models ════════════════════════════════════════════════════
s = slide()
head(s, "Two models, two prompting styles", "02")

tx(s, "VoxTell", L, 1.75, 5, 0.5, size=20, bold=True, color=ACCENT)
tx(s, "Our lab's CVPR submission. Segments from a free-text prompt\n"
      "(\"the spleen\"), which requires a 4-billion-parameter language\n"
      "model to encode the text before segmentation begins.",
   L, 2.3, 5.1, 1.6, size=13, space=4)

tx(s, "nnInteractive", 6.9, 1.75, 5, 0.5, size=20, bold=True, color=TEAL)
tx(s, "The challenge baseline. Segments from a 3-D bounding box.\n"
      "No text encoder at all, so it starts from a much lower\n"
      "per-prompt cost.",
   6.9, 2.3, 5.1, 1.6, size=13, space=4)

rule(s, L, 4.3, CW)
tx(s, "Both were already accurate. The question was whether they could be made\nfaster without giving that up.",
   L, 4.6, 10, 1.0, size=17, space=6)
source(s, "Accuracy measured as Dice Similarity Coefficient — overlap between prediction and expert annotation, 0 to 1.")

# ═══ 3 — The audit ═════════════════════════════════════════════════════════
s = slide()
head(s, "The first number I reported was wrong", "03")

tx(s, "VoxTell's speedup, as measurement error was removed:", L, 1.7, 8, 0.4,
   size=13, color=MUTED)

for i, (val, why, col) in enumerate([
    ("26×",   "baseline ran on CPU",              MUTED),
    ("17.6×", "no GPU warm-up; cached vs uncached", MUTED),
    ("7.1×",  "first arm absorbed start-up cost", MUTED),
    ("2.7×",  "measured correctly",               ACCENT),
]):
    x = L + i * 2.95
    tx(s, val, x, 2.3, 2.7, 0.95, size=44, bold=True, color=col)
    tx(s, why, x, 3.3, 2.7, 0.8, size=10, color=MUTED)

rule(s, L, 4.5, CW)
tx(s, "Every drop came from correcting how I measured — never from changing the code.\n"
      "The optimizations always did what they do; the benchmark was flattering them.",
   L, 4.85, 11, 1.1, size=17, space=6)
source(s, "Final figure: n=4 repeats, both arms INT4, GPU and text backbone pre-warmed, embedding cache verified empty.")

# ═══ 4 — What made it faster ═══════════════════════════════════════════════
s = slide()
head(s, "VoxTell — what actually made it faster", "04")

table(s, ["Change", "Effect", "DSC"], [
    ("Sliding window", "tile_step 0.75 + crop to non-zero: 25 → 9 patches", "+0.0003"),
    ("Embedding cache", "repeat prompts return a stored tensor", "identical"),
    ("INT4 text backbone", "1.5× faster encode, 2 GB VRAM instead of 8 GB", "0.97 agreement"),
    ("Numba preprocessing", "no measurable gain at this volume size", "unchanged"),
], L, 1.7, CW, 2.3, widths=[2.9, 6.4, 2.2])

rule(s, L, 4.35, CW)
stat(s, "2.7×", "end-to-end, abdominal CT, H100 MIG  ·  range 2.6–2.8× over 4 runs", L, 4.7, 6)
tx(s, "3.27s  →  1.28s", 7.3, 4.85, 5, 0.7, size=24, color=MUTED)
source(s, "CVPR validation case CT_AMOS_amos_0018 (63×512×512). Both arms INT4 (NF4) — precision held constant, so this is algorithmic gain only.")

# ═══ 5 — VoxTell accuracy ══════════════════════════════════════════════════
s = slide()
head(s, "VoxTell — accuracy held", "05")

stat(s, "+0.0003", "change in mean DSC after optimization", L, 1.9, 5, vsize=56, color=TEAL)

tx(s, "0.8090  →  0.8093", L, 3.5, 6, 0.6, size=22, color=MUTED)
tx(s, "65 objects across 5 abdominal CT cases.", L, 4.15, 6, 0.4, size=12, color=MUTED)

rule(s, 7.2, 1.9, 0.04, ACCENT, thick=90)
tx(s, "One caveat worth stating", 7.5, 1.9, 4.6, 0.4, size=13, bold=True)
tx(s, "INT4 quantization is on by default. On one CT case it agreed with "
      "full precision at 0.97 DSC and segmented 5.5% fewer voxels — a "
      "consistent under-segmentation, not noise.\n\n"
      "I have not measured that across the validation set, so I am reporting "
      "the direction, not a bound.",
   7.5, 2.4, 4.6, 2.4, size=12, space=8)
source(s, "VoxTell DSC from accuracy_results.csv. INT4 comparison is n=1 and measures output agreement, not accuracy against ground truth.")

# ═══ 6 — nnInteractive result ══════════════════════════════════════════════
s = slide()
head(s, "nnInteractive — compiling the network", "06")

tx(s, "torch.compile(network, mode='reduce-overhead')", L, 1.7, 8, 0.4,
   size=14, color=MUTED)
tx(s, "One line. It fuses kernels and cuts per-call dispatch overhead.",
   L, 2.15, 8, 0.4, size=13)

rule(s, L, 2.85, CW)

stat(s, "1.33×", "per object  ·  range 1.28–1.39× over 4 runs", L, 3.3, 5, color=ACCENT)
stat(s, "+0.0002", "mean DSC change  ·  294 objects", 6.9, 3.3, 5, color=TEAL)

tx(s, "0.288s → 0.215s per object", L, 5.4, 6, 0.4, size=14, color=MUTED)
tx(s, "No run showed degradation.", 6.9, 5.4, 5, 0.4, size=14, color=MUTED)
source(s, "20 CT cases from the CVPR validation set, fold='all' checkpoint, H100 MIG 3g.40gb. Speed and accuracy from the same jobs.")

# ═══ 7 — Break-even ════════════════════════════════════════════════════════
s = slide()
head(s, "The speedup is not free at the start", "07")

tx(s, "Compiling costs 23.6 seconds before the first prediction. "
      "That has to be earned back.", L, 1.75, 10.5, 0.5, size=17)

rule(s, L, 2.6, CW)

stat(s, "23.6s", "one-time compilation", L, 3.0, 3.4)
stat(s, "0.071s", "saved per object", 4.5, 3.0, 3.4)
stat(s, "~22 cases", "before it pays for itself", 8.0, 3.0, 4, color=ACCENT)

tx(s, "Worth it for a batch of 900 validation cases. Not worth it for a radiologist\n"
      "segmenting three scans — there the compile cost is never recovered.",
   L, 4.9, 11, 1.0, size=15, space=6)
source(s, "23.6s measured on node-local /tmp; the shared filesystem will be slower, so this is a lower bound. ~331 objects at 14.7 objects per case.")

# ═══ 8 — How it was validated ══════════════════════════════════════════════
s = slide()
head(s, "How I checked the numbers", "08")

for i, (t1, t2) in enumerate([
    ("Hold precision constant",
     "Both arms run INT4. A speedup that comes from quantization is not an algorithmic speedup."),
    ("Warm everything before timing",
     "GPU, text backbone, and sliding-window path. Whichever arm runs first otherwise absorbs start-up cost."),
    ("Prove the cache is empty",
     "The script asserts the embedding cache is cleared before each cold measurement, and written after."),
    ("Repeat, and report the range",
     "Every headline figure is n≥4 and quoted with its spread, never as a single number."),
]):
    y = 1.7 + i * 1.02
    rule(s, L, y + 0.12, 0.035, ACCENT, thick=26)
    tx(s, t1, L + 0.35, y, 2.9, 0.4, size=14, bold=True)
    tx(s, t2, 4.3, y + 0.02, 8.1, 0.8, size=12, color=MUTED)

rule(s, L, 5.9, CW)
tx(s, "Running the same script on two different GPUs is what exposed the largest error.",
   L, 6.2, 11, 0.5, size=15)
source(s, "Identical phases behaving differently on an RTX 4070 SUPER and an H100 revealed the ordering artifact behind the 7.1× figure.")

# ═══ 9 — Summary ═══════════════════════════════════════════════════════════
s = slide()
head(s, "Where both models landed", "09")

table(s, ["", "Speedup", "Accuracy", "Evidence"], [
    ("VoxTell", "2.7×  (2.6–2.8×)", "+0.0003 DSC", "4 runs, abdominal CT"),
    ("nnInteractive", "1.33×  (1.28–1.39×)", "+0.0002 DSC", "4 runs, 294 objects"),
], L, 1.8, CW, 1.5, widths=[2.6, 3.0, 2.6, 3.3])

rule(s, L, 3.7, CW)

tx(s, "Still open", L, 4.05, 5, 0.4, size=13, bold=True, color=ACCENT)
tx(s, "INT4's effect on accuracy is measured on one case, not the validation set.\n"
      "nnInteractive's compile gain is below the run-to-run spread on small batches.",
   L, 4.5, 6.0, 1.2, size=12, color=MUTED, space=6)

tx(s, "What I'd do next", 7.3, 4.05, 5, 0.4, size=13, bold=True, color=ACCENT)
tx(s, "Run the INT4 comparison across all 881 validation cases, which turns a\n"
      "directional finding into a bound worth acting on.",
   7.3, 4.5, 5.0, 1.2, size=12, color=MUTED, space=6)

rule(s, L, 6.1, CW)
tx(s, "The most useful thing I built was a benchmark that kept catching itself.",
   L, 6.4, 11, 0.5, size=16, bold=True)

prs.save("slides.pptx")
print(f"Saved: slides.pptx  ({len(prs.slides.__iter__.__self__._sldIdLst)} slides)")
