"""
Generate slides.pptx — VoxTell & nnInteractive optimization deck.
Run: python make_slides.py

Design: minimal. White ground, ink text, one amber accent used sparingly,
hairline rules instead of filled boxes. Bullets and fragments, not prose.
Every figure traces to a measured source — see SPEAKER_NOTES.md.
"""
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

# ── Palette ────────────────────────────────────────────────────────────────
INK    = RGBColor(0x1A, 0x1F, 0x26)
ACCENT = RGBColor(0xB5, 0x7E, 0x1F)
TEAL   = RGBColor(0x1D, 0x6F, 0x69)
MUTED  = RGBColor(0x8A, 0x91, 0x9B)
RULE   = RGBColor(0xDD, 0xDA, 0xD3)
BG     = RGBColor(0xFC, 0xFC, 0xFA)
WHITE  = RGBColor(0xFF, 0xFF, 0xFF)

L, CW = 0.9, 11.5

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
    tx(s, value, l, t, w, 0.9, size=vsize, bold=True, color=color)
    tx(s, label, l, t + 0.95, w, 0.5, size=10, color=MUTED)


def bullets(s, items, l, t, w, size=13, gap=0.42, color=INK):
    """Fragments, not sentences."""
    for i, it in enumerate(items):
        tx(s, "—  " + it, l, t + i * gap, w, gap, size=size, color=color)


def table(s, headers, rows, l, t, w, h, widths=None):
    tbl = s.shapes.add_table(len(rows) + 1, len(headers), Inches(l), Inches(t),
                             Inches(w), Inches(h)).table
    if widths:
        for i, ww in enumerate(widths):
            tbl.columns[i].width = Inches(ww)
    for c, htxt in enumerate(headers):
        cell = tbl.cell(0, c)
        cell.fill.solid()
        cell.fill.fore_color.rgb = BG
        r = cell.text_frame.paragraphs[0].add_run()
        r.text = htxt
        r.font.size = Pt(10)
        r.font.bold = True
        r.font.name = "Calibri"
        r.font.color.rgb = MUTED
    for ri, row in enumerate(rows, start=1):
        for ci, val in enumerate(row):
            cell = tbl.cell(ri, ci)
            cell.fill.solid()
            cell.fill.fore_color.rgb = WHITE if ri % 2 else BG
            r = cell.text_frame.paragraphs[0].add_run()
            r.text = str(val)
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
   size=11, color=ACCENT)
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

tx(s, "VoxTell", L, 1.7, 5, 0.45, size=20, bold=True, color=ACCENT)
bullets(s, [
    "Our lab's CVPR submission",
    "Prompt is free text — “the spleen”",
    "Needs a 4B-parameter LLM to encode it first",
], L, 2.25, 5.2)

tx(s, "nnInteractive", 6.9, 1.7, 5, 0.45, size=20, bold=True, color=TEAL)
bullets(s, [
    "The CVPR challenge baseline",
    "Prompt is a 3-D bounding box",
    "No text encoder — far cheaper per prompt",
], 6.9, 2.25, 5.2)

rule(s, L, 4.1, CW)
tx(s, "Both already accurate.", L, 4.45, 10, 0.5, size=20, bold=True)
tx(s, "Can they be made faster without giving that up?",
   L, 5.0, 10, 0.5, size=17, color=MUTED)
source(s, "DSC = Dice Similarity Coefficient. Overlap between prediction and expert annotation, 0 to 1. Above ~0.7 is clinically usable.")

# ═══ 3 — The audit ═════════════════════════════════════════════════════════
s = slide()
head(s, "The first number I reported was wrong", "03")

tx(s, "VoxTell speedup, as measurement error was removed", L, 1.65, 8, 0.4,
   size=12, color=MUTED)

for i, (val, why, col) in enumerate([
    ("26×",   "baseline ran on CPU",         MUTED),
    ("17.6×", "no warm-up; cache mismatch",  MUTED),
    ("7.1×",  "first arm ate start-up cost", MUTED),
    ("2.7×",  "measured correctly",          ACCENT),
]):
    x = L + i * 2.95
    tx(s, val, x, 2.2, 2.7, 0.9, size=44, bold=True, color=col)
    tx(s, why, x, 3.2, 2.7, 0.7, size=10, color=MUTED)

rule(s, L, 4.25, CW)
bullets(s, [
    "No correction changed the code — only how it was measured",
    "The optimizations always worked; the benchmark flattered them",
    "Caught by running the same script on two different GPUs",
], L, 4.6, 11, size=16, gap=0.52)
source(s, "Final figure: n=4 repeats · both arms INT4 · GPU and text backbone pre-warmed · embedding cache asserted empty.")

# ═══ 4 — What made it faster ═══════════════════════════════════════════════
s = slide()
head(s, "VoxTell — what made it faster", "04")

table(s, ["Change", "Effect", "DSC"], [
    ("Sliding window", "tile_step 0.75 + crop to non-zero  ·  25 → 9 patches", "+0.0003"),
    ("Embedding cache", "repeat prompts return a stored tensor", "identical"),
    ("INT4 backbone", "1.5× faster encode  ·  2 GB VRAM, not 8 GB", "0.97 agreement"),
    ("Numba preprocess", "no measurable gain at this volume size", "unchanged"),
], L, 1.65, CW, 2.3, widths=[2.7, 6.6, 2.2])

rule(s, L, 4.3, CW)
stat(s, "2.7×", "abdominal CT, H100 MIG  ·  2.6–2.8× over 4 runs", L, 4.65, 6)
tx(s, "3.27s → 1.28s", 7.4, 4.8, 5, 0.7, size=26, color=MUTED)
source(s, "Case CT_AMOS_amos_0018 (63×512×512). Both arms INT4 — precision held constant, so this is algorithmic gain only.")

# ═══ 5 — VoxTell accuracy ══════════════════════════════════════════════════
s = slide()
head(s, "VoxTell — accuracy held", "05")

stat(s, "+0.0003", "mean DSC change  ·  0.8090 → 0.8093", L, 1.85, 5.5,
     vsize=54, color=TEAL)
tx(s, "65 objects, 5 abdominal CT cases", L, 3.5, 5.5, 0.4, size=13, color=MUTED)

rule(s, 6.9, 1.85, 0.035, ACCENT, thick=78)
tx(s, "One caveat, stated plainly", 7.25, 1.85, 4.8, 0.4, size=15, bold=True)
bullets(s, [
    "INT4 quantization is on by default",
    "0.97 agreement with full precision",
    "Segments 5.5% fewer voxels",
    "One-sided → bias, not noise",
    "Not measured across the validation set",
], 7.25, 2.4, 5.0, size=12, gap=0.4)
source(s, "VoxTell DSC from accuracy_results.csv. INT4 comparison is n=1 and measures output agreement, not accuracy vs ground truth.")

# ═══ 6 — nnInteractive result ══════════════════════════════════════════════
s = slide()
head(s, "nnInteractive — compiling the network", "06")

tx(s, "torch.compile(network, mode='reduce-overhead')", L, 1.65, 9, 0.4,
   size=15, color=ACCENT)
bullets(s, [
    "One line changed",
    "Fuses kernels, cuts per-call dispatch overhead",
], L, 2.15, 9, size=13)

rule(s, L, 3.1, CW)
stat(s, "1.33×", "per object  ·  1.28–1.39× over 4 runs", L, 3.5, 5, color=ACCENT)
stat(s, "+0.0002", "mean DSC  ·  294 objects", 6.9, 3.5, 5, color=TEAL)

bullets(s, [
    "0.288s → 0.215s per object",
    "No run showed degradation",
], L, 5.5, 8, size=14, gap=0.4)
source(s, "20 CT cases, CVPR validation set, fold='all' checkpoint, H100 MIG 3g.40gb. Speed and accuracy from the same jobs.")

# ═══ 7 — Break-even ════════════════════════════════════════════════════════
s = slide()
head(s, "The speedup is not free at the start", "07")

tx(s, "Compiling costs time before the first prediction.", L, 1.7, 10, 0.5, size=18)
rule(s, L, 2.5, CW)

stat(s, "23.6s", "one-time compile", L, 2.9, 3.2)
stat(s, "0.071s", "saved per object", 4.4, 2.9, 3.2)
stat(s, "~22 cases", "to break even", 7.9, 2.9, 4, color=ACCENT)

rule(s, L, 4.75, CW)
bullets(s, [
    "Batch of 900 validation cases — clearly worth it",
    "Radiologist segmenting 3 scans — never recovered",
    "A batch optimization. Shouldn't be sold as anything else.",
], L, 5.1, 11, size=15, gap=0.48)
source(s, "23.6s measured on node-local /tmp; the shared filesystem is slower, so treat it as a lower bound. ~331 objects at 14.7 obj/case.")

# ═══ 8 — How it was validated ══════════════════════════════════════════════
s = slide()
head(s, "How I checked the numbers", "08")

for i, (t1, t2) in enumerate([
    ("Hold precision constant",
     "Both arms INT4 — a quantization gain can't pose as an algorithmic one"),
    ("Warm everything first",
     "GPU, text backbone, sliding-window path — or arm one eats the start-up cost"),
    ("Prove the cache is empty",
     "Asserted cleared before each cold measurement, written after"),
    ("Repeat, quote the range",
     "Every headline figure n≥4, never a single run"),
]):
    y = 1.7 + i * 1.0
    rule(s, L, y + 0.1, 0.03, ACCENT, thick=24)
    tx(s, t1, L + 0.32, y, 3.0, 0.4, size=14, bold=True)
    tx(s, t2, 4.25, y + 0.02, 8.1, 0.8, size=12, color=MUTED)

rule(s, L, 5.85, CW)
tx(s, "Two GPUs exposed the largest error.", L, 6.15, 11, 0.45, size=17, bold=True)
source(s, "Identical phases behaving differently on an RTX 4070 SUPER and an H100 revealed the ordering artifact behind the 7.1×.")

# ═══ 9 — Summary ═══════════════════════════════════════════════════════════
s = slide()
head(s, "Where both models landed", "09")

table(s, ["", "Speedup", "Accuracy", "Evidence"], [
    ("VoxTell", "2.7×  (2.6–2.8×)", "+0.0003 DSC", "4 runs, abdominal CT"),
    ("nnInteractive", "1.33×  (1.28–1.39×)", "+0.0002 DSC", "4 runs, 294 objects"),
], L, 1.75, CW, 1.5, widths=[2.6, 3.0, 2.6, 3.3])

rule(s, L, 3.6, CW)

tx(s, "Still open", L, 3.95, 5, 0.4, size=14, bold=True, color=ACCENT)
bullets(s, [
    "INT4 accuracy: one case, not the full set",
    "Compile gain sits near run-to-run spread",
], L, 4.45, 5.9, size=12, gap=0.42, color=MUTED)

tx(s, "Next", 7.3, 3.95, 5, 0.4, size=14, bold=True, color=ACCENT)
bullets(s, [
    "Run INT4 across all 881 validation cases",
    "Turns a direction into a bound worth acting on",
], 7.3, 4.45, 5.0, size=12, gap=0.42, color=MUTED)

rule(s, L, 5.85, CW)
tx(s, "The most useful thing I built was a benchmark that kept catching itself.",
   L, 6.15, 11.5, 0.5, size=17, bold=True)

prs.save("slides.pptx")
print("Saved: slides.pptx  (9 slides)")
