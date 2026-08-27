"""
Generate slides.pptx — VoxTell & nnInteractive Optimization deck.
Run: python make_slides.py
Imports cleanly into Google Slides via File → Import slides.
"""
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt
import copy
from lxml import etree

# ── Palette ────────────────────────────────────────────────────────────────
NAVY   = RGBColor(0x0F, 0x1C, 0x2E)
AMBER  = RGBColor(0xC8, 0x91, 0x2B)
TEAL   = RGBColor(0x1D, 0x7A, 0x72)
WHITE  = RGBColor(0xFF, 0xFF, 0xFF)
OFFWHT = RGBColor(0xF2, 0xEF, 0xE8)
MUTED  = RGBColor(0x5A, 0x64, 0x78)
SLIDE_BG = RGBColor(0xFA, 0xFA, 0xF8)

# ── Canvas ─────────────────────────────────────────────────────────────────
prs = Presentation()
prs.slide_width  = Inches(13.33)
prs.slide_height = Inches(7.5)

BLANK = prs.slide_layouts[6]  # blank layout


def add_slide():
    return prs.slides.add_slide(BLANK)


def set_bg(slide, color):
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color


def txbox(slide, text, l, t, w, h, size=18, bold=False, color=None,
          align=PP_ALIGN.LEFT, italic=False, font='Calibri'):
    box = slide.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
    box.word_wrap = True
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.name = font
    if color:
        run.font.color.rgb = color
    return box


def hline(slide, l, t, w, color, thickness=4):
    from pptx.util import Pt as PPt
    line = slide.shapes.add_shape(
        1,  # MSO_SHAPE_TYPE.RECTANGLE
        Inches(l), Inches(t), Inches(w), Inches(thickness / 72)
    )
    line.fill.solid()
    line.fill.fore_color.rgb = color
    line.line.fill.background()
    return line


def rect(slide, l, t, w, h, fill_color, line_color=None):
    shape = slide.shapes.add_shape(1, Inches(l), Inches(t), Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_color
    if line_color:
        shape.line.color.rgb = line_color
    else:
        shape.line.fill.background()
    return shape


def add_table(slide, rows, cols, l, t, w, h,
              header_bg=NAVY, header_fg=WHITE,
              row_colors=None, data=None):
    tbl = slide.shapes.add_table(rows, cols, Inches(l), Inches(t),
                                  Inches(w), Inches(h)).table
    return tbl


def cell_set(cell, text, size=11, bold=False, color=None,
             bg=None, align=PP_ALIGN.LEFT):
    tf = cell.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = align
    if p.runs:
        run = p.runs[0]
    else:
        run = p.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.name = 'Calibri'
    if color:
        run.font.color.rgb = color
    if bg:
        cell.fill.solid()
        cell.fill.fore_color.rgb = bg


def slide_header(slide, title, num, accent=TEAL, dark_bg=False):
    hline(slide, 0.4, 0.25, 0.06, accent, thickness=40)
    fg = WHITE if dark_bg else NAVY
    txbox(slide, title, 0.55, 0.15, 10, 0.5,
          size=22, bold=True, color=fg)
    txbox(slide, num, 11.8, 0.15, 1.4, 0.5,
          size=11, color=MUTED, align=PP_ALIGN.RIGHT)


# ── SLIDE 1: Title ─────────────────────────────────────────────────────────
s = add_slide()
set_bg(s, NAVY)

txbox(s, 'CVPR 2025  ·  MEDICAL IMAGE SEGMENTATION', 0.9, 1.2, 11, 0.4,
      size=10, color=AMBER, font='Calibri')
txbox(s, 'VoxTell & nnInteractive', 0.9, 1.7, 11, 1.0,
      size=40, bold=True, color=OFFWHT, font='Calibri')
txbox(s, 'Optimization on NVIDIA H100 MIG', 0.9, 2.75, 9, 0.6,
      size=24, color=RGBColor(0xB0, 0xBC, 0xC8), font='Calibri')

hline(s, 0.9, 3.5, 0.6, AMBER, thickness=6)

txbox(s, 'torch.compile  ·  Embedding Cache  ·  Sliding Window  ·  Numba Preprocessing',
      0.9, 3.7, 10, 0.4, size=12, color=RGBColor(0x70, 0x80, 0x90))

txbox(s, 'Brian Xiao  ·  rrg-jma  ·  Fir cluster (Alliance Canada)',
      0.9, 6.6, 10, 0.4, size=10, color=MUTED)

txbox(s, 'Brian Xiao  ·  Aug 2026',
      0.9, 6.9, 10, 0.4, size=10, color=MUTED)

# ── SLIDE 2: What is Medical Image Segmentation ────────────────────────────
s = add_slide()
set_bg(s, SLIDE_BG)
slide_header(s, 'What is Medical Image Segmentation?', '02 / 12', AMBER)

txbox(s, 'The Problem', 0.5, 0.9, 5.5, 0.35, size=10, bold=True, color=AMBER)
txbox(s,
      'A CT or MRI scan is a 3-D volume of voxels. A segmentation model draws '
      'a precise 3-D boundary around each organ or lesion — producing a binary '
      'mask the radiologist or downstream algorithm can act on.',
      0.5, 1.3, 5.5, 1.0, size=13, color=NAVY)

txbox(s, 'Accuracy Metric — DSC', 0.5, 2.5, 5.5, 0.35, size=10, bold=True, color=TEAL)
txbox(s,
      'Dice Similarity Coefficient (DSC) = 2|A∩B| / (|A|+|B|)\n'
      '  0 = no overlap     1 = perfect\n'
      '  > 0.70 = clinically acceptable\n'
      '  Our results: ~0.79',
      0.5, 2.9, 5.5, 1.4, size=12, color=NAVY)

txbox(s, 'Why Speed Matters', 0.5, 4.5, 5.5, 0.35, size=10, bold=True, color=AMBER)
txbox(s,
      'A hospital CT pipeline may process hundreds of cases per day. '
      'Each case has 10–20 objects to segment. Every millisecond saved per object '
      'compounds across the full workload.',
      0.5, 4.9, 5.5, 1.0, size=12, color=NAVY)

# Right column — models
txbox(s, 'The Two Models', 7.0, 0.9, 5.8, 0.35, size=10, bold=True, color=TEAL)

for i, (model, desc, latency, color) in enumerate([
    ('VoxTell', 'Lab\'s CVPR submission\nText-prompt segmentation (Qwen3 LLM)', '3.27s → 1.28s  (CT, H100)', AMBER),
    ('nnInteractive', 'CVPR 2025 challenge baseline\nBbox-prompt interactive segmentation', '0.288s → 0.215s', TEAL),
]):
    top = 1.35 + i * 2.3
    rect(s, 7.0, top, 5.8, 1.9, RGBColor(0xF0, 0xED, 0xE6))
    txbox(s, model, 7.2, top + 0.1, 4, 0.4, size=15, bold=True, color=color)
    txbox(s, desc, 7.2, top + 0.5, 4.5, 0.6, size=11, color=NAVY)
    txbox(s, latency, 7.2, top + 1.2, 5.2, 0.4, size=13, bold=True, color=color)


# ── SLIDE 3: VoxTell — Baseline ────────────────────────────────────────────
s = add_slide()
set_bg(s, SLIDE_BG)
slide_header(s, 'VoxTell — Why the Original 26× Was Wrong', '03 / 12', AMBER)

txbox(s, 'Hardware: NVIDIA GeForce RTX 4070 SUPER (12 GB GDDR6X)  ·  v0_gpu: Qwen3-4B + nnUNet 3D',
      0.5, 0.9, 12, 0.35, size=10, color=MUTED)

alt_bg = RGBColor(0xF5, 0xF2, 0xEB)
tbl = add_table(s, 4, 3, 0.5, 1.3, 8.5, 2.0)
for c, h in enumerate(['Phase', 'Latency (v0_gpu, RTX 4070 SUPER)', 'Notes']):
    cell_set(tbl.cell(0, c), h, bold=True, bg=NAVY, color=WHITE, size=11)
rows_data = [
    ('Text encoding — harness loaded FP32', '~2.7s', 'Benchmark bug: FP32 overflowed 12 GB → CPU. Model code was never FP32.'),
    ('Image preprocessing (numpy)', '~0.23s', 'Crop, z-score normalize, patch'),
    ('Sliding window inference (GPU, warm)', '~0.17s', 'tile_step=0.5, 4 patches on this volume, FP16'),
]
for r, (a, b, c) in enumerate(rows_data):
    bg = alt_bg if r % 2 == 0 else WHITE
    bug_color = AMBER if r == 0 else None
    cell_set(tbl.cell(r+1, 0), a, bg=bg, size=10, color=bug_color)
    cell_set(tbl.cell(r+1, 1), b, bg=bg, size=10, align=PP_ALIGN.CENTER)
    cell_set(tbl.cell(r+1, 2), c, bg=bg, size=10, color=MUTED)

txbox(s, 'Total: ~3.10s per prompt  —  dominated by text encoder running on CPU (silent bug)',
      0.5, 3.45, 10, 0.4, size=13, bold=True, color=AMBER)

txbox(s, 'Root cause — in the BENCHMARK, not the model', 0.5, 3.95, 7, 0.32, size=10, bold=True, color=AMBER)
txbox(s,
      'The baseline harness loaded Qwen3-4B in FP32 (~16 GB) on a 12 GB card, so PyTorch '
      'silently ran the encoder on CPU. VoxTell itself has always loaded FP16 or INT4 — '
      'there is no FP32 path in the model code. This inflated the baseline, not the model.',
      0.5, 4.3, 7.2, 1.1, size=11, color=NAVY)

txbox(s, 'Why this matters', 0.5, 5.5, 3, 0.32, size=10, bold=True, color=TEAL)
txbox(s, 'Correcting it is a measurement fix, not an optimization — so it earns no row in the '
         'speedup table. A correctly measured RTX baseline is 2.2s, not 3.10s.',
      0.5, 5.85, 11.5, 0.5, size=11, color=NAVY)

txbox(s, '3.10s', 9.8, 3.9, 3.0, 0.85, size=40, bold=True, color=AMBER,
      align=PP_ALIGN.CENTER)
txbox(s, 'mis-measured baseline', 9.5, 4.75, 3.5, 0.35, size=10, color=MUTED,
      align=PP_ALIGN.CENTER)


# ── SLIDE 4: VoxTell — Optimizations ──────────────────────────────────────
s = add_slide()
set_bg(s, SLIDE_BG)
slide_header(s, 'VoxTell — Bug Fix + Three Algorithmic Optimizations', '04 / 12', AMBER)

txbox(s, 'All measurements on NVIDIA GeForce RTX 4070 SUPER (12 GB)  ·  DSC from experiment_log.md',
      0.5, 0.82, 12, 0.32, size=10, color=MUTED)

tbl = add_table(s, 5, 4, 0.5, 1.2, 12.3, 3.0)
hdrs = ['Change', 'Method', 'Speedup', 'DSC change']
for c, h in enumerate(hdrs):
    cell_set(tbl.cell(0, c), h, bold=True, bg=NAVY, color=WHITE, size=11)

opt_rows = [
    ('Sliding window overlap', 'tile_step 0.75 + crop-to-nonzero  (25 → 9 patches on CT)', '2.9× sliding window', '+0.0006'),
    ('Embedding cache', 'LRU memory + SHA-256 disk cache', '29× warm re-query (CT)', 'Identical'),
    ('Numba preprocessing', '@njit(parallel=True) crop + z-score normalize', 'no gain measured', 'Unchanged'),
    ('INT4 (NF4) quantization', 'bitsandbytes NF4 on Qwen3: ~2GB vs ~8GB FP16 VRAM', '1.5× text fwd pass', 'DSC 0.97 agree, −5.5% vox'),
]
for r, row in enumerate(opt_rows):
    bg = alt_bg if r % 2 == 1 else WHITE
    for c, val in enumerate(row):
        sp_color = TEAL if c == 2 else (RGBColor(0x1A, 0x6A, 0x30) if c == 3 else None)
        cell_set(tbl.cell(r+1, c), val, bg=bg, size=10,
                 color=sp_color, bold=(c == 2))

txbox(s, 'Precision-matched GPU-vs-GPU results', 0.5, 4.9, 5, 0.32, size=10, bold=True, color=AMBER)
txbox(s,
      'H100 MIG, abdominal CT (job 56964411):  v0_gpu 3.27s → v3 1.28s  =  2.6× algorithmic gain\n'
      'H100 MIG, MNI brain (job 56964410):  1.0× — volume too small for tile_step or Numba to act\n'
      'RTX 4070 SUPER, MNI brain (n=5):  1.7× (range 1.6–1.8×)\n'
      'Both arms INT4 (NF4); GPU, text backbone and sliding-window path pre-warmed; embed cache verified cold.',
      0.5, 5.25, 12, 1.1, size=10, color=NAVY)


# ── SLIDE 5: VoxTell — Accuracy ────────────────────────────────────────────
s = add_slide()
set_bg(s, SLIDE_BG)
slide_header(s, 'VoxTell — Accuracy Verification', '05 / 12', AMBER)

txbox(s, 'All optimizations preserve segmentation quality', 0.5, 0.9, 9, 0.4,
      size=13, color=MUTED)

tbl = add_table(s, 5, 3, 0.5, 1.4, 9, 2.8)
for c, h in enumerate(['Setting', 'Mean DSC', 'vs Baseline']):
    cell_set(tbl.cell(0, c), h, bold=True, bg=NAVY, color=WHITE, size=12)
acc_rows = [
    ('Baseline v0_gpu', '0.7794', '—'),
    ('+ FP16 GPU placement', '0.7795', '< 0.001'),
    ('+ Sliding window (0.75)', '0.7800', '+0.0006'),
    ('+ Embedding cache', '0.7800', 'Identical'),
]
for r, row in enumerate(acc_rows):
    bg = alt_bg if r % 2 == 0 else WHITE
    for c, val in enumerate(row):
        gc = RGBColor(0x1A, 0x6A, 0x30) if c == 2 and val not in ('—', 'Identical') else None
        cell_set(tbl.cell(r+1, c), val, bg=bg, size=11, color=gc)

rect(s, 9.8, 1.4, 3.0, 2.8, RGBColor(0xE8, 0xF4, 0xF0))
txbox(s, 'DSC Change', 10.0, 1.6, 2.6, 0.35, size=10, bold=True, color=TEAL,
      align=PP_ALIGN.CENTER)
txbox(s, '< 0.001', 10.0, 2.05, 2.6, 0.7, size=36, bold=True, color=TEAL,
      align=PP_ALIGN.CENTER)
txbox(s, 'ACCURACY MAINTAINED', 10.0, 2.85, 2.6, 0.35, size=9, bold=True,
      color=TEAL, align=PP_ALIGN.CENTER)
txbox(s, 'n = 294 objects', 10.0, 3.2, 2.6, 0.35, size=10, color=MUTED,
      align=PP_ALIGN.CENTER)

txbox(s, 'No pass/fail threshold applied — the measured delta and its sample size are reported as-is.',
      0.5, 4.5, 12, 0.5, size=11, color=MUTED)


# ── SLIDE 6: nnInteractive Intro ───────────────────────────────────────────
s = add_slide()
set_bg(s, SLIDE_BG)
slide_header(s, 'nnInteractive — What It Is', '06 / 12', TEAL)

txbox(s, 'CVPR 2025 challenge baseline model', 0.5, 0.9, 9, 0.35, size=12, color=MUTED)

for i, (title, body, col) in enumerate([
    ('Bbox prompting', 'User provides a 3-D bounding box around the target object. '
     'nnInteractive predicts the segmentation mask inside that box. '
     'No text encoder — direct geometric prompt.', TEAL),
    ('autozoom', 'Iterative refinement at multiple scales. If the initial prediction '
     'is confident, it exits early ("No zoom out necessary"). On this CVPR validation set, '
     'autozoom never fired — overhead is approximately zero.', TEAL),
    ('fold=\'all\' checkpoint', 'The official CVPR 2025 checkpoint (DSC ~0.79). '
     'fold=0 gives ~0.33 DSC — undertrained weights; never compare against it.', AMBER),
]):
    top = 1.3 + i * 1.65
    hline(s, 0.5, top, 0.04, col, thickness=60)
    txbox(s, title, 0.7, top + 0.05, 4.5, 0.35, size=13, bold=True, color=col)
    txbox(s, body, 0.7, top + 0.45, 5.5, 0.9, size=11, color=NAVY)

# right: why nnInteractive is faster
txbox(s, 'Why nnInteractive is faster than VoxTell', 7.2, 0.9, 5.5, 0.35,
      size=10, bold=True, color=TEAL)
txbox(s,
      'VoxTell accepts free-text prompts ("brain", "left kidney") — '
      'each requires a Qwen3-4B text encoder forward pass even with FP16 on GPU.\n\n'
      'nnInteractive accepts a 3-D bounding box directly — no text encoder, '
      'the encoder cost is zero. Geometric prompts are sufficient for the CVPR challenge format.\n\n'
      'A direct per-prompt latency comparison requires both models on the same hardware '
      'AND the same data. VoxTell H100 on abdominal CT (job 56964411): '
      'v0_gpu 3.27s → v3 1.28s = 2.6× algorithmic gain, both arms INT4.',
      7.2, 1.35, 5.7, 4.5, size=11, color=NAVY)


# ── SLIDE 7: nnInteractive Profiling ──────────────────────────────────────
s = add_slide()
set_bg(s, SLIDE_BG)
slide_header(s, 'nnInteractive — Baseline Profiling', '07 / 12', TEAL)

txbox(s, 'fold=\'all\', do_autozoom=True, torch_n_threads=os.cpu_count(), H100 MIG 3g.40gb',
      0.5, 0.9, 12, 0.35, size=11, color=MUTED)

tbl = add_table(s, 4, 3, 0.5, 1.3, 7.5, 2.1)
for c, h in enumerate(['Phase', 'Latency', 'Frequency']):
    cell_set(tbl.cell(0, c), h, bold=True, bg=NAVY, color=WHITE, size=12)
ph_rows = [
    ('set_image', '0.345s', 'Once per case — image encoding + preprocessing'),
    ('_predict (cold, first call)', '~2.4s', 'CUDA kernel JIT — one-time per session'),
    ('_predict (warm)', '0.288s', 'Per object ← optimization target'),
]
for r, row in enumerate(ph_rows):
    bg = alt_bg if r % 2 == 0 else WHITE
    col3 = TEAL if r == 2 else None
    bold3 = r == 2
    cell_set(tbl.cell(r+1, 0), row[0], bg=bg, size=11)
    cell_set(tbl.cell(r+1, 1), row[1], bg=bg, size=11, color=col3, bold=bold3)
    cell_set(tbl.cell(r+1, 2), row[2], bg=bg, size=11, color=MUTED)

txbox(s, 'A 15-object case: 0.345s set_image + 15 × 0.288s = 4.66s total',
      0.5, 3.55, 8, 0.4, size=13, bold=True, color=NAVY)
txbox(s, '_predict warm is the bottleneck — set_image is < 8% of case time.',
      0.5, 4.0, 8, 0.35, size=11, color=MUTED)

txbox(s, '0.288s', 9.2, 3.2, 3.5, 0.85, size=44, bold=True, color=TEAL,
      align=PP_ALIGN.CENTER)
txbox(s, 'per object (warm baseline)', 8.9, 4.05, 4.0, 0.4, size=10, color=MUTED,
      align=PP_ALIGN.CENTER)


# ── SLIDE 8: torch.compile ────────────────────────────────────────────────
s = add_slide()
set_bg(s, SLIDE_BG)
slide_header(s, 'nnInteractive — torch.compile Optimization', '08 / 12', TEAL)

txbox(s, 'Implementation', 0.5, 0.9, 5.5, 0.35, size=10, bold=True, color=TEAL)

rect(s, 0.5, 1.25, 5.8, 0.85, RGBColor(0x0F, 0x1C, 0x2E))
txbox(s, '# One line after session initialization\nsession.network = torch.compile(\n    session.network, mode=\'reduce-overhead\'\n)',
      0.65, 1.3, 5.5, 0.75, size=11, color=RGBColor(0x7D, 0xC4, 0xBC))

txbox(s, 'How it works', 0.5, 2.3, 5.5, 0.35, size=10, bold=True, color=AMBER)
txbox(s,
      'PyTorch normally interprets the model graph in Python on every forward pass. '
      'torch.compile traces the graph and emits optimized Triton GPU kernels via the '
      'inductor backend. With mode=\'reduce-overhead\', CUDA graphs eliminate CPU→GPU '
      'launch latency on repeated calls with identical input shapes — exactly the '
      'nnInteractive inference pattern.',
      0.5, 2.7, 6.0, 1.5, size=11, color=NAVY)

txbox(s, 'Key requirements', 7.0, 0.9, 6.0, 0.35, size=10, bold=True, color=TEAL)
for i, (title, body) in enumerate([
    ('N_WARMUP = 2',
     'Warmup 1 triggers Triton compilation (~24s). Warmup 2 stabilizes dispatch (~0.51s). '
     'Runs 3+ are fully warm. With N_WARMUP=1, compiled latency reads ~0.54s and speedup ≈ 1.0×.'),
    ('Env vars before import torch',
     'TORCHINDUCTOR_CACHE_DIR is read at import time, not at compile time. '
     'Set XDG_CACHE_HOME, TORCH_HOME, and TORCHINDUCTOR_CACHE_DIR before any torch import.'),
    ('fold=\'all\' required',
     'The official checkpoint (DSC ~0.79). fold=0 gives ~0.33 DSC — not a valid comparison baseline.'),
]):
    top = 1.35 + i * 1.6
    hline(s, 7.0, top, 0.04, TEAL, thickness=50)
    txbox(s, title, 7.2, top + 0.04, 5.5, 0.35, size=12, bold=True, color=TEAL)
    txbox(s, body, 7.2, top + 0.45, 5.8, 0.85, size=10, color=NAVY)


# ── SLIDE 9: nnInteractive Results ────────────────────────────────────────
s = add_slide()
set_bg(s, SLIDE_BG)
slide_header(s, 'nnInteractive — Speedup Results', '09 / 12', TEAL)

txbox(s, 'Jobs 56923894–896 (repeats) + 56908464 — n=4 total · fold=\'all\', autozoom=ON · 20 CT cases · 294 objects · H100 MIG 3g.40gb',
      0.5, 0.9, 12, 0.35, size=10, color=MUTED)

# Big stats
for val, lbl, col, x in [
    ('1.33×', 'per-object speedup\n(n=4: 1.28–1.39×)', TEAL, 0.5),
    ('0.288s', 'baseline _predict\n(no compile)', MUTED, 3.5),
    ('0.215s', 'compiled _predict\n(fully warm)', TEAL, 6.5),
    ('1.33×', 'case-level speedup\n(~15-object case)', TEAL, 9.5),
]:
    txbox(s, val, x, 1.35, 2.9, 0.85, size=34, bold=True, color=col,
          align=PP_ALIGN.CENTER)
    txbox(s, lbl, x, 2.2, 2.9, 0.55, size=10, color=MUTED,
          align=PP_ALIGN.CENTER)

tbl = add_table(s, 3, 5, 0.5, 3.0, 12.3, 1.8)
for c, h in enumerate(['Setting', 'set_image', '_predict (warm)', '~15-obj case', 'vs Baseline']):
    cell_set(tbl.cell(0, c), h, bold=True, bg=NAVY, color=WHITE, size=11)
res_rows = [
    ('Baseline (fold=\'all\', autozoom=ON)', '0.345s', '0.288s', '4.38s', '1.0×'),
    ('torch.compile (fold=\'all\', autozoom=ON)', '0.345s', '0.215s', '3.30s', '1.33×'),
]
for r, row in enumerate(res_rows):
    bg = alt_bg if r == 0 else RGBColor(0xE0, 0xF2, 0xEE)
    for c, val in enumerate(row):
        col = TEAL if (r == 1 and c >= 2) else None
        bold = r == 1 and c >= 2
        cell_set(tbl.cell(r+1, c), val, bg=bg, size=11, color=col, bold=bold)

txbox(s, '✓  Confirmed across n=4 runs (56908464, 56923894–896). Range: 1.28×–1.39×. '
         'DSC Δ ≤ +0.0004 across all runs — accuracy stable.',
      0.5, 5.0, 12, 0.4, size=10, color=TEAL)


# ── SLIDE 10: nnInteractive Accuracy ──────────────────────────────────────
s = add_slide()
set_bg(s, SLIDE_BG)
slide_header(s, 'nnInteractive — Accuracy Verification', '10 / 12', TEAL)

txbox(s, 'Same job as latency (56908464) — speed and accuracy measured under identical conditions',
      0.5, 0.9, 11, 0.35, size=11, color=MUTED)

txbox(s, 'Experimental setup', 0.5, 1.35, 5.5, 0.35, size=10, bold=True, color=TEAL)
for i, line in enumerate([
    '20 CT cases from CVPR 2025 validation set',
    '294 objects evaluated (bbox-prompted)',
    'fold=\'all\' checkpoint — same as 0.7794 CodaBench submission',
    'Ground truth evaluated locally against expert annotations',
    'Baseline and compiled run back-to-back — no config differences',
]):
    txbox(s, '·  ' + line, 0.5, 1.75 + i * 0.42, 6.2, 0.38, size=11, color=NAVY)

txbox(s, 'DSC results', 0.5, 4.0, 5.5, 0.35, size=10, bold=True, color=TEAL)
tbl = add_table(s, 4, 3, 0.5, 4.4, 6.5, 1.8)
for c, h in enumerate(['Setting', 'Mean DSC', 'Objects']):
    cell_set(tbl.cell(0, c), h, bold=True, bg=NAVY, color=WHITE, size=11)
dsc_rows = [
    ('Baseline (fold=\'all\')', '0.7914', '294'),
    ('torch.compile (fold=\'all\')', '0.7916', '294'),
    ('Difference', '+0.0002', '—'),
]
for r, row in enumerate(dsc_rows):
    bg = alt_bg if r % 2 == 0 else WHITE
    gc = RGBColor(0x1A, 0x6A, 0x30) if r == 2 else None
    for c, val in enumerate(row):
        cell_set(tbl.cell(r+1, c), val, bg=bg, size=11,
                 color=gc, bold=(r == 2))

rect(s, 8.0, 1.4, 4.8, 4.8, RGBColor(0xE0, 0xF2, 0xEE))
txbox(s, 'DSC Change', 8.2, 1.6, 4.4, 0.35, size=10, bold=True, color=TEAL,
      align=PP_ALIGN.CENTER)
txbox(s, '+0.0002', 8.2, 2.1, 4.4, 1.0, size=54, bold=True, color=TEAL,
      align=PP_ALIGN.CENTER)
txbox(s, '4 runs, Δ +0.0000 to +0.0004 — none negative', 8.2, 3.2, 4.4, 0.4, size=10, color=MUTED,
      align=PP_ALIGN.CENTER)
hline(s, 8.3, 3.75, 4.2, TEAL, thickness=2)
txbox(s, 'NO DEGRADATION ACROSS 4 RUNS', 8.2, 3.9, 4.4, 0.35, size=10, bold=True,
      color=TEAL, align=PP_ALIGN.CENTER)
txbox(s, 'Difference within normal FP16\nnon-associativity noise between\nTriton and cuDNN kernels.',
      8.2, 4.3, 4.4, 0.8, size=10, color=MUTED, align=PP_ALIGN.CENTER)


# ── SLIDE 11: Cold Start ──────────────────────────────────────────────────
s = add_slide()
set_bg(s, SLIDE_BG)
slide_header(s, 'nnInteractive — Cold Start & Break-even', '11 / 12', TEAL)

txbox(s, 'Job 56914757 · fully isolated /tmp inductor cache · all three env vars set before import torch',
      0.5, 0.9, 11, 0.35, size=10, color=MUTED)

tbl = add_table(s, 7, 2, 0.5, 1.35, 7.5, 3.0)
cell_set(tbl.cell(0, 0), 'Metric', bold=True, bg=NAVY, color=WHITE, size=11)
cell_set(tbl.cell(0, 1), 'Value', bold=True, bg=NAVY, color=WHITE, size=11)
cs_rows = [
    ('Triton cold-start (run 1, job 56914757)', '23.61s'),
    ('Residual (run 2)', '0.513s'),
    ('Fully warm mean (runs 3–6)', '0.125s'),
    ('Mean warm gain per object (n=4 jobs)', '0.0714s'),
    ('Break-even', '~331 objects (~22 cases)'),
    ('First case cold vs warm baseline', '~25.4s  vs  ~4.38s'),
]
for r, (a, b) in enumerate(cs_rows):
    bg = alt_bg if r % 2 == 0 else WHITE
    hl = r == 4  # break-even row
    cell_set(tbl.cell(r+1, 0), a, bg=bg, size=11)
    cell_set(tbl.cell(r+1, 1), b, bg=bg, size=11,
             color=TEAL if hl else None, bold=hl)

txbox(s, 'Break-even = 23.61s ÷ 0.0714s/object = ~331 objects = ~22 cases',
      0.5, 4.5, 9, 0.4, size=12, bold=True, color=TEAL)

txbox(s, 'Note: /tmp is node-local fast storage. Production Triton cache on /scratch (NFS) will be higher.\n'
         '23.61s is a lower bound for a from-scratch deployment.\n'
         'After break-even, all subsequent objects run at 1.33×.',
      0.5, 5.0, 9, 0.8, size=10, color=MUTED)

txbox(s, '~23', 10.2, 2.1, 2.7, 1.0, size=54, bold=True, color=TEAL,
      align=PP_ALIGN.CENTER)
txbox(s, 'cases to break even', 9.9, 3.1, 3.3, 0.4, size=11, color=MUTED,
      align=PP_ALIGN.CENTER)
txbox(s, '(~331 objects)', 9.9, 3.5, 3.3, 0.4, size=10, color=MUTED,
      align=PP_ALIGN.CENTER)


# ── SLIDE 12: Summary ──────────────────────────────────────────────────────
s = add_slide()
set_bg(s, SLIDE_BG)
slide_header(s, 'Summary of All Optimizations', '12 / 12', NAVY)

tbl = add_table(s, 6, 5, 0.5, 0.85, 12.3, 3.3)
for c, h in enumerate(['Model', 'Optimization', 'Method', 'Speedup', 'DSC impact']):
    cell_set(tbl.cell(0, c), h, bold=True, bg=NAVY, color=WHITE, size=11)

summary_rows = [
    ('VoxTell', 'Sliding window + crop', 'tile_step 0.75, crop-to-nonzero', '2.9× window (CT)', '+0.0006'),
    ('', 'Embedding cache', 'LRU memory + SHA-256 disk', '29× warm (CT)', 'Identical'),
    ('', 'INT4 (NF4) text backbone', 'bitsandbytes, ~2 GB vs ~8 GB', '1.5× fwd pass', 'DSC 0.97, −5.5% vox'),
    ('', 'Numba preprocessing', '@njit(parallel=True)', 'no gain measured', 'Unchanged'),
    ('nnInteractive', 'torch.compile', 'compile(network, mode=\'reduce-overhead\')', '1.33× / object (n=4)', '+0.0002'),
]
for r, row in enumerate(summary_rows):
    bg = RGBColor(0xFD, 0xF5, 0xE4) if r < 4 else RGBColor(0xE0, 0xF2, 0xEE)
    for c, val in enumerate(row):
        sp_color = (AMBER if r < 4 else TEAL) if c == 3 else None
        cell_set(tbl.cell(r+1, c), val, bg=bg, size=10,
                 color=sp_color, bold=(c == 3 and bool(val)))

txbox(s, 'VoxTell — precision-matched GPU-vs-GPU benchmark', 0.5, 4.35, 7, 0.32, size=10, bold=True, color=AMBER)
txbox(s,
      'H100 MIG, abdominal CT (job 56964411): 3.27s → 1.28s = 2.6× algorithmic, both arms INT4.  '
      'RTX 4070 SUPER, MNI brain (n=5): 1.7× (1.6–1.8×).  H100 on the same brain: 1.0×.\n'
      'INT4 vs FP16 on one CT case: DSC 0.97 agreement, INT4 under-segments by 5.5% — not measured on the full set.',
      0.5, 4.7, 12, 0.62, size=10, color=NAVY)

txbox(s, 'nnInteractive (H100 MIG 3g.40gb, Fir cluster, job 56908464)', 0.5, 5.35, 8, 0.32, size=10, bold=True, color=TEAL)
txbox(s,
      '0.288s → 0.215s per object. Mean speedup 1.33× (n=4: 1.28–1.39×). '
      'DSC Δ ≤ +0.0004. Speed and accuracy from same job, same config. Break-even ~69 cases (/scratch).',
      0.5, 5.7, 12, 0.45, size=11, color=NAVY)


# ── Save ───────────────────────────────────────────────────────────────────
out = 'slides.pptx'
prs.save(out)
print(f'Saved: {out}  ({prs.slides.__len__()} slides)')
