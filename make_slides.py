"""
Generate slides.pptx — VoxTell & nnInteractive: architecture, optimization,
measurement, and deployment.

Run: python make_slides.py     (needs python-pptx; base miniconda has it)

Design: one accent colour on a near-white ground. Whitespace separates sections
rather than full-width rules, which read as generated. No em dashes.
Every figure traces to a measured source — see SPEAKER_NOTES.md.

Architecture figures come from the checkpoint and the source, not from memory:
VoxTell from models/voxtell_v1.1/plans.json and voxtell/model/*.py,
nnInteractive from the paper (arXiv 2503.08373) and this project's own runs.
"""
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

# ── Palette: ground, ink, one accent ──────────────────────────────────────────
BG     = RGBColor(0xFD, 0xFD, 0xFD)   # near-white, neutral
INK    = RGBColor(0x1C, 0x20, 0x26)
ACCENT = RGBColor(0xA8, 0x6D, 0x18)
MUTED  = RGBColor(0x8C, 0x8F, 0x96)
HAIR   = RGBColor(0xE8, 0xE8, 0xE6)
PANEL  = RGBColor(0xF4, 0xF3, 0xF0)

L, CW = 0.95, 11.45

prs = Presentation()
prs.slide_width  = Inches(13.33)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]

_n = [0]   # running slide number, so inserting a slide never renumbers by hand


def slide():
    _n[0] += 1
    s = prs.slides.add_slide(BLANK)
    f = s.background.fill
    f.solid()
    f.fore_color.rgb = BG
    return s


def tx(s, text, l, t, w, h, size=13, bold=False, color=INK,
       align=PP_ALIGN.LEFT, space=None, mono=False):
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
        r.font.name = "Consolas" if mono else "Calibri"
        r.font.color.rgb = color
    return box


def rule(s, l, t, w, color=HAIR, thick=1):
    sh = s.shapes.add_shape(1, Inches(l), Inches(t), Inches(w), Inches(thick / 72))
    sh.fill.solid()
    sh.fill.fore_color.rgb = color
    sh.line.fill.background()
    return sh


def head(s, title):
    """Title, page number, and a short accent mark. No full-width bar."""
    tx(s, title, L, 0.6, 9.8, 0.6, size=27, bold=True)
    tx(s, f"{_n[0]:02d}", 11.0, 0.77, 1.4, 0.4, size=10, color=MUTED,
       align=PP_ALIGN.RIGHT)
    rule(s, L, 1.32, 0.9, ACCENT, thick=2.5)


def divider(title, kicker):
    """Section break. Counts as a slide but carries no page number."""
    s = slide()
    rule(s, L, 3.0, 1.1, ACCENT, thick=3)
    tx(s, kicker, L, 3.3, 9, 0.4, size=12, color=ACCENT)
    tx(s, title, L, 3.8, 11, 1.2, size=38, bold=True)
    return s


def stat(s, value, label, l, t, w, vsize=46, color=INK):
    tx(s, value, l, t, w, 0.8, size=vsize, bold=True, color=color)
    tx(s, label, l, t + 0.82, w, 0.5, size=10, color=MUTED)


def bullets(s, items, l, t, w, size=13, gap=0.44, color=INK):
    for i, it in enumerate(items):
        y = t + i * gap
        sq = s.shapes.add_shape(1, Inches(l), Inches(y + 0.085),
                                Inches(0.07), Inches(0.07))
        sq.fill.solid()
        sq.fill.fore_color.rgb = ACCENT
        sq.line.fill.background()
        tx(s, it, l + 0.24, y, w - 0.24, gap, size=size, color=color)


def table(s, headers, rows, l, t, w, h, widths=None, fsize=11):
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
        r.font.size = Pt(10); r.font.bold = True
        r.font.name = "Calibri"; r.font.color.rgb = MUTED
    for ri, row in enumerate(rows, start=1):
        for ci, val in enumerate(row):
            cell = tbl.cell(ri, ci)
            cell.fill.solid()
            cell.fill.fore_color.rgb = BG
            r = cell.text_frame.paragraphs[0].add_run()
            r.text = str(val)
            r.font.size = Pt(fsize)
            r.font.name = "Calibri"; r.font.color.rgb = INK
            r.font.bold = (ci == 0)
    return tbl


def panel(s, l, t, w, h, color=PANEL):
    sh = s.shapes.add_shape(1, Inches(l), Inches(t), Inches(w), Inches(h))
    sh.fill.solid()
    sh.fill.fore_color.rgb = color
    sh.line.fill.background()
    sh.shadow.inherit = False
    return sh


def box(s, label, sub, l, t, w, h=0.72, fill=PANEL, lab_color=INK, size=12):
    """One node in a pipeline diagram."""
    panel(s, l, t, w, h, fill)
    tx(s, label, l + 0.12, t + 0.09, w - 0.24, 0.3, size=size, bold=True,
       color=lab_color)
    if sub:
        tx(s, sub, l + 0.12, t + 0.36, w - 0.24, 0.3, size=9, color=MUTED)


def arrow(s, l, t, w=0.28):
    tx(s, "→", l, t, w, 0.3, size=16, color=MUTED, align=PP_ALIGN.CENTER)


def code(s, lines, l, t, w, h, size=11):
    panel(s, l, t, w, h)
    tx(s, "\n".join(lines), l + 0.18, t + 0.14, w - 0.36, h - 0.28,
       size=size, mono=True, space=3)


def source(s, text):
    tx(s, text, L, 6.9, CW, 0.4, size=9, color=MUTED)


# ═══════════════════════════════════════════════════════════════════════════
# 1 — Title
# ═══════════════════════════════════════════════════════════════════════════
s = slide()
rule(s, L, 1.55, 1.1, ACCENT, thick=3)
tx(s, "Peter Munk Cardiac Centre  ·  AI Team", L, 1.85, 9, 0.4,
   size=11, color=ACCENT)
tx(s, "Text and Box Prompted\n3D Medical Segmentation", L, 2.35, 11, 1.9,
   size=44, bold=True, color=INK)
tx(s, "Architecture  ·  Optimization  ·  Measurement  ·  Deployment",
   L, 4.35, 10, 0.4, size=15, color=MUTED)
tx(s, "Brian Xiao   ·   VoxTell and nnInteractive   ·   Fir cluster, Alliance Canada",
   L, 6.6, 11, 0.4, size=10, color=MUTED)

# ═══ 2 — Roadmap ═══════════════════════════════════════════════════════════
s = slide()
head(s, "What This Covers")
table(s, ["Part", "Content", "Slides"], [
    ("1. Context", "The task, and two ways of prompting a segmentation model", "3 to 5"),
    ("2. Architecture", "How VoxTell and nnInteractive are actually built", "6 to 17"),
    ("3. Optimization", "Four changes, what each one bought", "18 to 24"),
    ("4. Measurement", "Four artifacts that inflated the result, and the fixes", "25 to 32"),
    ("5. Results", "Speed and accuracy, with the caveats attached", "33 to 39"),
    ("6. Deployment", "Skills, MCP and plugins, and the validation gate", "40 to 46"),
    ("7. Limits", "Where the models fail, and what is still open", "47 to 51"),
], L, 1.8, CW, 3.6, widths=[2.4, 7.2, 1.85])
tx(s, "The measurement section is the part I would most want questions on.",
   L, 5.9, 11, 0.5, size=15, bold=True)

# ═══ 3 — The task ══════════════════════════════════════════════════════════
s = slide()
head(s, "The Task")
tx(s, "Given a 3D scan, outline a structure.", L, 1.8, 11, 0.5, size=19)
bullets(s, [
    "Input is a volume: a CT or MR study, typically 200 to 900 slices",
    "Output is a binary mask, one voxel label per position",
    "Scored with Dice Similarity Coefficient: overlap with expert annotation, 0 to 1",
    "Manual annotation of one organ across a volume takes a trained reader tens of minutes",
], L, 2.5, 11.3, size=14, gap=0.5)

panel(s, L, 4.7, CW, 1.5)
tx(s, "Why it is hard in 3D", L + 0.25, 4.85, 5, 0.3, size=12, bold=True, color=ACCENT)
tx(s, "A 512 x 512 x 400 volume is 105 million voxels. It does not fit in GPU memory as one\n"
      "forward pass, so the volume is tiled into overlapping patches and stitched back together.\n"
      "That tiling is where most of the inference time goes, and where most of the speedups live.",
   L + 0.25, 5.2, 10.9, 1.0, size=13)
source(s, "DSC above roughly 0.7 is generally considered clinically usable. Lesions score well below organs for every model in this class.")

# ═══ 4 — Two prompting paradigms ═══════════════════════════════════════════
s = slide()
head(s, "Two Ways to Say What You Want")

tx(s, "Text prompt", L, 1.8, 5, 0.4, size=18, bold=True, color=ACCENT)
code(s, ['segment("the spleen")'], L, 2.3, 5.2, 0.6)
bullets(s, [
    "Any structure the language model can name",
    "No interaction needed, so it batches",
    "Requires a language model in the loop",
    "Accepts any string, including nonsense",
], L, 3.15, 5.2, size=13, gap=0.46)

tx(s, "Spatial prompt", 6.95, 1.8, 5, 0.4, size=18, bold=True, color=INK)
code(s, ["segment(bbox=[[z0,z1],", "              [y0,y1],", "              [x0,x1]])"],
     6.95, 2.3, 5.2, 1.1)
bullets(s, [
    "The box defines the target, so no vocabulary limit",
    "Cheap: no text encoder at all",
    "Needs a human to place the box",
    "Cannot be wrong about what you meant",
], 6.95, 3.65, 5.2, size=13, gap=0.46)

tx(s, "The tradeoff drives everything downstream: cost, failure modes, and what can be automated.",
   L, 5.85, 11.4, 0.5, size=15, bold=True)

# ═══ 5 — The two models ════════════════════════════════════════════════════
s = slide()
head(s, "The Two Models")
table(s, ["", "VoxTell v1.1", "nnInteractive v1.0"], [
    ("Prompt", "Free text", "Point, scribble, box, lasso"),
    ("Origin", "DKFZ, CVPR 2026 submission", "DKFZ, CVPR 2025 challenge baseline"),
    ("Backbone", "Residual Encoder U-Net", "Residual Encoder U-Net (ResEnc-L)"),
    ("Patch size", "192 x 192 x 192", "192 x 192 x 192"),
    ("Extra cost", "Qwen3-Embedding-4B text encoder", "None"),
    ("Training scale", "Not published in the checkpoint", "64,518 volumes, 717,148 objects"),
], L, 1.8, CW, 3.1, widths=[2.2, 4.6, 4.65])
tx(s, "Same backbone family, same patch size, completely different front end.",
   L, 5.35, 11, 0.5, size=17, bold=True)
source(s, "VoxTell rows read from models/voxtell_v1.1/plans.json. nnInteractive rows from arXiv 2503.08373 and this project's own runs.")

# ═══════════════════════════════════════════════════════════════════════════
divider("Architecture", "PART 2")
# ═══════════════════════════════════════════════════════════════════════════

# ═══ 7 — VoxTell pipeline ══════════════════════════════════════════════════
s = slide()
head(s, "VoxTell: Inference Pipeline")
for i, (lab, sub) in enumerate([
    ("Preprocess", "crop to non-zero, z-score"),
    ("Text encode", "Qwen3-Embedding-4B"),
    ("Sliding window", "192 cubed patches"),
    ("Postprocess", "sigmoid, insert crop"),
]):
    x = L + i * 2.95
    box(s, lab, sub, x, 1.9, 2.6, 0.85)
    if i < 3:
        arrow(s, x + 2.62, 2.12)

tx(s, "Where the time went, before optimization", L, 3.2, 8, 0.35,
   size=12, bold=True, color=ACCENT)
table(s, ["Phase", "Time", "Share"], [
    ("Preprocessing", "0.09s", "2.8%"),
    ("Text embedding", "2.17s", "67.3%"),
    ("Sliding window", "0.94s", "29.2%"),
    ("Postprocessing", "0.02s", "0.6%"),
], L, 3.65, 6.1, 2.1, widths=[2.5, 1.8, 1.8])

bullets(s, [
    "Text embedding dominated, so caching it was the first target",
    "Sliding window is the floor: it is the actual segmentation",
    "Preprocessing and postprocessing are noise at this volume size",
], 7.3, 3.9, 5.0, size=12, gap=0.55)
source(s, "RTX 4070 SUPER, 1 prompt, warm model, no cache. OPTIMIZATION_REPORT_COMPACT.md section 2.")

# ═══ 8 — VoxTell backbone ══════════════════════════════════════════════════
s = slide()
head(s, "VoxTell: Segmentation Backbone")
tx(s, "dynamic_network_architectures.architectures.unet.ResidualEncoderUNet",
   L, 1.75, 11, 0.35, size=12, mono=True, color=ACCENT)

table(s, ["Stage", "Channels", "Blocks", "Stride", "Resolution"], [
    ("0", "32", "1", "1, 1, 1", "192 cubed"),
    ("1", "64", "3", "2, 2, 2", "96 cubed"),
    ("2", "128", "4", "2, 2, 2", "48 cubed"),
    ("3", "256", "6", "2, 2, 2", "24 cubed"),
    ("4", "320", "6", "2, 2, 2", "12 cubed"),
    ("5", "320", "6", "2, 2, 2", "6 cubed"),
], L, 2.2, 6.6, 3.0, widths=[1.0, 1.5, 1.2, 1.5, 1.4], fsize=10)

bullets(s, [
    "Six stages, 32x total downsampling",
    "Residual blocks weighted to the deep stages, where context matters",
    "Conv3d with InstanceNorm3d and LeakyReLU",
    "Decoder is light: one conv per stage, transposed conv upsampling",
    "Deep supervision off at inference",
], 7.9, 2.3, 4.5, size=12, gap=0.55)
source(s, "Read from models/voxtell_v1.1/plans.json, configurations.3d_fullres.architecture.arch_kwargs.")

# ═══ 9 — VoxTell text encoder ══════════════════════════════════════════════
s = slide()
head(s, "VoxTell: The Text Side")
box(s, "Prompt string", '"the spleen"', L, 1.85, 2.6, 0.8)
arrow(s, L + 2.7, 2.08)
box(s, "Qwen3-Embedding-4B", "4B parameter encoder", L + 3.05, 1.85, 3.1, 0.8)
arrow(s, L + 6.25, 2.08)
box(s, "Embedding", "2560 dimensions", L + 6.6, 1.85, 2.4, 0.8)
arrow(s, L + 9.1, 2.08)
box(s, "Query", "864 dimensions", L + 9.45, 1.85, 2.0, 0.8, fill=PANEL,
    lab_color=ACCENT)

tx(s, "The projection", L, 3.15, 6, 0.35, size=13, bold=True, color=ACCENT)
code(s, [
    "self.project_text_embed = nn.Sequential(",
    "    nn.Linear(text_embedding_dim, 2048),",
    "    nn.GELU(),",
    "    nn.Linear(2048, query_dim),   # query_dim = 864",
    ")",
], L, 3.6, 6.4, 1.5)

bullets(s, [
    "A 4B parameter language model runs before any segmentation starts",
    "That is the entire reason text prompting is expensive",
    "It is also fully cacheable: the same prompt always gives the same vector",
    "This observation is optimization 2",
], 7.6, 3.65, 4.8, size=12, gap=0.55)
source(s, "Projection from voxtell/model/voxtell_model.py. Embedding width from OPTIMIZATION_REPORT_COMPACT.md section 1.")

# ═══ 10 — How text becomes a mask ══════════════════════════════════════════
s = slide()
head(s, "VoxTell: How Text Becomes a Mask")
tx(s, "A MaskFormer style head. The text embedding is used as a query, not as a class label.",
   L, 1.75, 11.3, 0.4, size=14)

for i, (lab, sub) in enumerate([
    ("Text query", "864-d, from the prompt"),
    ("Transformer decoder", "6 layers, 8 heads, pre-norm"),
    ("Per-stage projection", "864 to stage channels"),
    ("Dot with features", "mask logits per scale"),
]):
    x = L + i * 2.95
    box(s, lab, sub, x, 2.3, 2.6, 0.85, lab_color=ACCENT if i == 0 else INK)
    if i < 3:
        arrow(s, x + 2.62, 2.52)

bullets(s, [
    "The decoder attends over image features at stage 4 (320 channels, 12 cubed), with a 3D positional encoding",
    "The refined query is projected separately for each of 5 mask-former stages",
    "A dot product between query and decoder feature map gives the mask at that scale",
    "So the prompt does not select from a fixed label set. It parameterises the mask directly, which is why the vocabulary is open",
], L, 3.55, 11.4, size=13, gap=0.58)
source(s, "voxtell/model/voxtell_model.py: DECODER_CONFIGS, num_maskformer_stages=5, decoder_layer=4, TRANSFORMER_NUM_LAYERS=6, TRANSFORMER_NUM_HEADS=8.")

# ═══ 11 — Transformer decoder detail ═══════════════════════════════════════
s = slide()
head(s, "VoxTell: Prompt Decoder Internals")
table(s, ["Parameter", "Value", "Note"], [
    ("d_model", "864", "query dimension throughout"),
    ("Layers", "6", "TRANSFORMER_NUM_LAYERS"),
    ("Heads", "8", "TRANSFORMER_NUM_HEADS"),
    ("Feedforward", "2048", "inner width of each layer"),
    ("Norm placement", "Pre-norm", "normalize_before=True"),
    ("Positional encoding", "3D sinusoidal", "flattened over h, w, d"),
    ("Attention", "Self then cross", "query attends to image features"),
], L, 1.8, 6.9, 3.3, widths=[2.2, 1.6, 3.1], fsize=10)

tx(s, "Each layer", 8.2, 1.9, 4, 0.35, size=13, bold=True, color=ACCENT)
code(s, [
    "self_attn  (query to query)",
    "multihead_attn  (query to image)",
    "linear1 -> GELU -> linear2",
    "three LayerNorms, residual adds",
], 8.2, 2.35, 4.2, 1.5, size=10)
bullets(s, [
    "Standard transformer decoder, used for prompt fusion rather than generation",
    "Cheap next to the 4B text encoder: it runs on one 864-d token",
], 8.2, 4.05, 4.2, size=12, gap=0.62)
source(s, "voxtell/model/transformer.py, TransformerDecoderLayer.")

# ═══ 12 — No resampling ════════════════════════════════════════════════════
s = slide()
head(s, "VoxTell: One Design Choice That Matters Later")
code(s, ['"data_identifier": "nnUNetResEncUNetLPlans_noResampling_3d_fullres"'],
     L, 1.8, 11.4, 0.6)
tx(s, "VoxTell v1.1 does not resample to a target spacing.", L, 2.65, 11, 0.45,
   size=19, bold=True)

bullets(s, [
    "Most nnU-Net configurations resample every scan to a fixed voxel size first",
    "This one does not, and the inference path has no resampling step either",
    "So the network sees voxels at whatever spacing the scan was acquired at",
    "Through a fixed 192 cubed patch, that changes the physical field of view",
], L, 3.3, 11.3, size=14, gap=0.5)

table(s, ["Acquired spacing", "What one patch covers"], [
    ("1.0 mm", "192 mm of anatomy"),
    ("5.0 mm", "960 mm, longer than a torso"),
], L, 5.35, 6.0, 1.0, widths=[2.8, 3.2])
tx(s, "This turns into a measured accuracy\nfalloff on slide 48.",
   7.4, 5.5, 4.8, 0.8, size=14, bold=True, color=ACCENT)
source(s, "plans.json, configurations.3d_fullres.data_identifier. Confirmed by grep over voxtell/inference/predictor.py: no resampling call.")

# ═══ 13 — nnInteractive pipeline ═══════════════════════════════════════════
s = slide()
head(s, "nnInteractive: Inference Pipeline")
for i, (lab, sub) in enumerate([
    ("Set image", "volume, no text"),
    ("Add interaction", "box, point, scribble, lasso"),
    ("AutoZoom", "expand ROI until captured"),
    ("Refine", "sliding window at full res"),
]):
    x = L + i * 2.95
    box(s, lab, sub, x, 1.95, 2.6, 0.85)
    if i < 3:
        arrow(s, x + 2.62, 2.17)

bullets(s, [
    "No text encoder, so cost per prompt starts far lower than VoxTell",
    "Interactive by design: the model expects to be corrected and re-run",
    "The session holds the image, so repeated prompts on one volume are cheap",
    "The checkpoint must be loaded with fold='all'. Using fold 0 scores about 0.33 DSC and is not a valid baseline",
], L, 3.25, 11.4, size=14, gap=0.55)
source(s, "API from nnInteractive v1.0. The fold='all' requirement was found the hard way on this project and is documented in the plugin skill.")

# ═══ 14 — nnInteractive prompting ══════════════════════════════════════════
s = slide()
head(s, "nnInteractive: Early Prompting")
tx(s, "Prompts are not a separate branch. They are extra input channels.",
   L, 1.75, 11.3, 0.4, size=17, bold=True)

table(s, ["Channel group", "Count", "Carries"], [
    ("Image", "1", "the scan itself"),
    ("Point", "2", "positive and negative clicks"),
    ("Scribble", "2", "positive and negative strokes"),
    ("Lasso and box", "2", "the two share one pair"),
    ("Previous prediction", "1", "the model's own last output"),
], L, 2.35, 6.6, 2.6, widths=[2.4, 1.0, 3.2], fsize=10)

bullets(s, [
    "The network is a U-Net over a stack of channels, not an encoder plus a prompt module",
    "Feeding the previous prediction back is what makes refinement iterative",
    "A correction is just another channel edit and another forward pass",
    "Backbone is the nnU-Net Residual Encoder L configuration, the same family as VoxTell",
], 7.9, 2.45, 4.5, size=12, gap=0.62)
source(s, "arXiv 2503.08373, early prompting. The paper describes the channel groups; the exact total is not quoted here because the paper does not state it in one place.")

# ═══ 15 — AutoZoom ═════════════════════════════════════════════════════════
s = slide()
head(s, "nnInteractive: AutoZoom")
tx(s, "The model decides its own field of view.", L, 1.8, 11, 0.45, size=18, bold=True)

for i, (lab, sub) in enumerate([
    ("Predict in ROI", "start tight around the prompt"),
    ("Check borders", "does the mask touch the edge?"),
    ("Zoom out 1.5x", "repeat, up to 4x total"),
    ("Resize and refine", "sliding window at full res"),
]):
    x = L + i * 2.95
    box(s, lab, sub, x, 2.5, 2.6, 0.85)
    if i < 3:
        arrow(s, x + 2.62, 2.72)

bullets(s, [
    "Small objects never pay for a whole-volume pass",
    "Large objects still get captured, because the ROI grows until the mask stops touching the border",
    "The cost is data dependent, which makes per object timing noisier than VoxTell's",
    "do_autozoom=True is the default and was left on for every measurement here",
], L, 3.8, 11.4, size=13, gap=0.55)
source(s, "arXiv 2503.08373: ROI expands by a factor of 1.5 iteratively, up to 4x zoom out, then the low resolution mask is resized and refined with a sliding window.")

# ═══ 16 — nnInteractive training scale ═════════════════════════════════════
s = slide()
head(s, "nnInteractive: Training Scale")
for i, (v, lab) in enumerate([
    ("120+", "public 3D datasets"),
    ("64,518", "volumes"),
    ("717,148", "annotated objects"),
]):
    stat(s, v, lab, L + i * 3.8, 1.95, 3.5, vsize=42, color=ACCENT if i == 2 else INK)

bullets(s, [
    "Spans CT, MR, PET, ultrasound and microscopy",
    "That breadth is the reason a bounding box generalises to structures it was never given a name for",
    "It is also why the vocabulary question does not arise: there is no label set to be outside of",
    "VoxTell trades that for the ability to be asked in words, which is a different kind of generality",
], L, 3.6, 11.4, size=14, gap=0.55)
source(s, "arXiv 2503.08373. Figures are the paper's, not measured here.")

# ═══ 17 — Architectures compared ═══════════════════════════════════════════
s = slide()
head(s, "Architectures Side by Side")
table(s, ["", "VoxTell v1.1", "nnInteractive v1.0"], [
    ("Backbone", "ResidualEncoderUNet, 6 stages", "nnU-Net ResEnc-L"),
    ("Channels", "32, 64, 128, 256, 320, 320", "Same family"),
    ("Patch", "192 cubed", "192 cubed"),
    ("Prompt enters", "As a transformer query, late", "As input channels, early"),
    ("Prompt encoder", "Qwen3-Embedding-4B, 4B params", "None"),
    ("Head", "MaskFormer style, 5 scales", "Standard segmentation head"),
    ("Resampling", "None", "Handled by AutoZoom instead"),
    ("Iterative", "No", "Yes, previous prediction fed back"),
], L, 1.8, CW, 3.9, widths=[2.3, 4.55, 4.6], fsize=10)
tx(s, "Late fusion buys an open vocabulary. Early fusion buys iteration and a much cheaper prompt.",
   L, 6.0, 11.4, 0.5, size=16, bold=True)

# ═══════════════════════════════════════════════════════════════════════════
divider("Optimization", "PART 3")
# ═══════════════════════════════════════════════════════════════════════════

# ═══ 19 — Objective ════════════════════════════════════════════════════════
s = slide()
head(s, "What I Was Trying to Do")
tx(s, "Minimise end to end GPU inference latency without losing accuracy.",
   L, 1.8, 11.3, 0.5, size=19, bold=True)
bullets(s, [
    "Accuracy is the constraint, not the objective. A faster model that segments worse is not a result",
    "So every speed change is paired with a DSC measurement on the same checkpoint",
    "Target hardware is the H100 MIG 3g.40gb partition on Fir, with an RTX 4070 SUPER for local iteration",
    "Both models were already accurate. The question was only whether they could be made cheaper",
], L, 2.6, 11.4, size=14, gap=0.55)
panel(s, L, 5.0, CW, 1.25)
tx(s, "Constraint that shaped everything", L + 0.25, 5.15, 6, 0.3,
   size=12, bold=True, color=ACCENT)
tx(s, "Inference only. No retraining, no fine tuning, no architecture changes. The checkpoint is\n"
      "fixed, so any accuracy change is a bug in what I did rather than a property of the model.",
   L + 0.25, 5.5, 10.9, 0.7, size=13)

# ═══ 20 — Optimization 1 ═══════════════════════════════════════════════════
s = slide()
head(s, "Optimization 1: Sliding Window Overlap")
tx(s, "tile_step_size 0.5 to 0.75", L, 1.75, 8, 0.4, size=17, mono=True, color=ACCENT)
bullets(s, [
    "Patches overlap so that stitching artifacts do not appear at patch boundaries",
    "At step 0.5 each patch overlaps its neighbour by half, which is generous",
    "At 0.75 the overlap drops and the patch count falls with it",
    "Combined with cropping to the non-zero bounding box, 25 patches became 9 on the test case",
], L, 2.35, 11.4, size=14, gap=0.52)
stat(s, "0.94s to 0.82s", "sliding window, 1 prompt, RTX 4070 SUPER", L, 4.6, 6.5, vsize=34)
stat(s, "+0.0003", "DSC, so no accuracy cost", 7.6, 4.6, 4.5, vsize=34, color=ACCENT)
source(s, "13% reduction on this phase. The DSC figure is the full 5 case AMOS evaluation, not the single timing case.")

# ═══ 21 — Optimization 2 ═══════════════════════════════════════════════════
s = slide()
head(s, "Optimization 2: Two Level Embedding Cache")
tx(s, "The same prompt always produces the same vector. There is no reason to compute it twice.",
   L, 1.75, 11.3, 0.4, size=15)

for i, (lab, sub) in enumerate([
    ("In memory", "LRU dict, about 5 KB per entry"),
    ("On disk", "SHA-256 keyed .pt files"),
    ("Miss", "run Qwen3 and store"),
]):
    x = L + i * 3.9
    box(s, lab, sub, x, 2.4, 3.5, 0.85, lab_color=ACCENT if i == 0 else INK)

stat(s, "2.17s to under 0.001s", "in memory cache hit", L, 3.6, 7.5, vsize=32, color=ACCENT)
bullets(s, [
    "Lookup is a dict get plus a GPU memory copy",
    "Disk layer survives across sessions: 0.02s on NVMe, 0.04s on Lustre",
    "This is the single largest win, because embedding was 67% of runtime",
    "It matters clinically: the same anatomical queries repeat across every volume in a study",
], L, 4.85, 11.4, size=13, gap=0.5)
source(s, "Cache state is asserted empty before every cold measurement. A warm cache on one arm and a cold cache on the other was one of the four artifacts.")

# ═══ 22 — Optimization 3 ═══════════════════════════════════════════════════
s = slide()
head(s, "Optimization 3: INT4 Text Backbone")
tx(s, "bitsandbytes NF4, double quantized, FP16 compute", L, 1.75, 9, 0.4,
   size=14, mono=True, color=ACCENT)
code(s, [
    "BitsAndBytesConfig(",
    "    load_in_4bit=True,",
    "    bnb_4bit_quant_type='nf4',",
    "    bnb_4bit_use_double_quant=True,",
    "    bnb_4bit_compute_dtype=torch.float16,",
    ")",
], L, 2.3, 5.6, 1.8, size=10)

stat(s, "8 GB to 2 GB", "text backbone VRAM", 6.9, 2.35, 5.2, vsize=34, color=ACCENT)
bullets(s, [
    "NF4 is designed for normally distributed weights, which transformer weights approximately are",
    "Double quantization compresses the quantization constants too, worth about 0.4 bits",
    "The point is fitting on a 12 GB card, not speed",
], 6.9, 3.6, 5.3, size=12, gap=0.6)

panel(s, L, 4.45, CW, 1.55)
tx(s, "The failure mode worth knowing", L + 0.25, 4.6, 6, 0.3, size=12, bold=True, color=ACCENT)
tx(s, "_load_text_backbone builds the config inside a try block and catches every exception, falling\n"
      "back to FP16. If bitsandbytes or accelerate is missing, the model loads, reports INT4 in the log,\n"
      "and runs FP16. Nothing downstream would tell you. The plugin checks for this separately.",
   L + 0.25, 4.95, 10.9, 1.0, size=13)
source(s, "voxtell/inference/predictor.py, _load_text_backbone. Both packages were undeclared dependencies until this project added them.")

# ═══ 23 — Optimization 4 ═══════════════════════════════════════════════════
s = slide()
head(s, "Optimization 4: Numba Preprocessing")
tx(s, "@numba.njit(parallel=True) over crop to non-zero and z-score normalisation",
   L, 1.75, 11, 0.4, size=14, mono=True, color=ACCENT)
bullets(s, [
    "Replaces the NumPy implementation with a compiled, parallel one",
    "Compiles once per process, then runs at native speed",
], L, 2.35, 11.3, size=14, gap=0.5)

panel(s, L, 3.4, CW, 1.6)
tx(s, "Result: no measurable gain", L + 0.25, 3.58, 6, 0.35, size=15, bold=True, color=ACCENT)
tx(s, "0.09s before, 0.09s after on the RTX. 0.14s both ways on the cluster CPU. On the first call it is\n"
      "actually slower, because that call pays the JIT compile. Preprocessing is under 3% of runtime, so\n"
      "there was never much to win here.",
   L + 0.25, 4.0, 10.9, 0.9, size=13)

tx(s, "Reported because a negative result is still a result. Optimising a 3% phase was the wrong "
      "place to spend effort, and measuring it is how that became visible.",
   L, 5.3, 11.4, 0.8, size=15, bold=True)
source(s, "The one place it may still pay is multi-prompt or much larger volumes, which has not been tested.")

# ═══ 24 — Negative results ═════════════════════════════════════════════════
s = slide()
head(s, "Things That Did Not Work")
table(s, ["Approach", "Outcome", "Why"], [
    ("ONNX plus ORT CUDA", "14x slower than PyTorch", "ORT lacks cuDNN 3D convolution kernel support"),
    ("torch.compile on VoxTell", "1.00x, no change", "Triton unavailable on Windows; model is compute bound"),
    ("Numba preprocessing", "No measurable gain", "Phase is under 3% of total runtime"),
    ("Batched sliding window", "Built, not yet paid off", "batch_size=1 today; needs H100 80 GB to matter"),
], L, 1.85, CW, 2.5, widths=[2.9, 3.1, 5.45], fsize=11)

tx(s, "torch.compile did work on nnInteractive, at 1.33x. Same technique, different model, "
      "opposite result. That is worth more than either number alone.",
   L, 4.7, 11.4, 0.8, size=15, bold=True)
bullets(s, [
    "VoxTell is bound by 3D convolution throughput, which compilation does not change",
    "nnInteractive makes many small calls per object, where dispatch overhead is the cost",
], L, 5.6, 11.4, size=13, gap=0.45)

# ═══════════════════════════════════════════════════════════════════════════
divider("Measurement", "PART 4")
# ═══════════════════════════════════════════════════════════════════════════

# ═══ 26 — The arc ══════════════════════════════════════════════════════════
s = slide()
head(s, "The First Number I Reported Was Wrong")
tx(s, "VoxTell speedup, as measurement error was removed", L, 1.7, 8, 0.35,
   size=11, color=MUTED)

for i, (val, why, col) in enumerate([
    ("26x",   "baseline ran on CPU",         MUTED),
    ("17.6x", "no warm-up, cache mismatch",  MUTED),
    ("7.1x",  "first arm ate start-up cost", MUTED),
    ("2.7x",  "measured correctly",          ACCENT),
]):
    x = L + i * 2.9
    tx(s, val, x, 2.15, 2.7, 0.85, size=42, bold=True, color=col)
    tx(s, why, x, 3.08, 2.7, 0.6, size=10, color=MUTED)

tx(s, "Not one of those corrections changed the code being measured.",
   L, 4.0, 11.4, 0.5, size=19, bold=True)
tx(s, "The optimizations always did exactly what they do. Every drop came from the benchmark "
      "flattering them. A 10x error in the reported result, entirely in the measurement.",
   L, 4.65, 11.4, 0.9, size=15, color=MUTED)
tx(s, "The next four slides are each one of those errors.", L, 5.8, 11, 0.4,
   size=14, bold=True, color=ACCENT)
source(s, "The largest error surfaced by running the same script on two GPUs and noticing that phases which should behave the same did not.")

# ═══ 27 — Artifact 1 ═══════════════════════════════════════════════════════
s = slide()
head(s, "Artifact 1: The Baseline Ran on CPU")
stat(s, "26x", "reported", L, 1.85, 2.6, vsize=48, color=MUTED)
tx(s, "What actually happened", 4.0, 1.9, 8, 0.4, size=15, bold=True, color=ACCENT)
bullets(s, [
    "The baseline loaded the text encoder in FP32",
    "FP32 Qwen3-4B needs about 16 GB; the card has 12 GB",
    "The allocation failed and the model fell back to CPU",
    "So a GPU run was being compared against a CPU run",
], 4.0, 2.4, 8.2, size=13, gap=0.46)

panel(s, L, 4.4, CW, 1.5)
tx(s, "Why it was not obvious", L + 0.25, 4.55, 6, 0.3, size=12, bold=True, color=ACCENT)
tx(s, "Nothing errored. The run completed, produced a correct mask, and returned a plausible time.\n"
      "The only signal was that the baseline was slower than it had any right to be, and that only\n"
      "looks wrong if you have an expectation of how fast it should be.",
   L + 0.25, 4.9, 10.9, 1.0, size=13)
source(s, "Fix: assert the device of every module before timing, and hold precision identical across arms.")

# ═══ 28 — Artifact 2 ═══════════════════════════════════════════════════════
s = slide()
head(s, "Artifact 2: Cache State Differed Between Arms")
stat(s, "17.6x", "reported", L, 1.85, 2.8, vsize=48, color=MUTED)
tx(s, "What actually happened", 4.0, 1.9, 8, 0.4, size=15, bold=True, color=ACCENT)
bullets(s, [
    "The optimized arm ran second, after the baseline had populated the cache",
    "So it read the embedding from memory while the baseline computed it",
    "That is a real 2000x gain on that phase, but it is the cache, not the algorithm",
    "Reporting it as one number conflates two different claims",
], 4.0, 2.4, 8.2, size=13, gap=0.46)

tx(s, "The fix is to report both, separately.", L, 4.45, 11, 0.4, size=16, bold=True)
table(s, ["Claim", "Number", "What it measures"], [
    ("Algorithmic", "1.8x cold", "tile_step and Numba, both arms cold"),
    ("Full stack", "2.0x warm", "the same plus a cache hit"),
], L, 4.95, 11.0, 1.2, widths=[2.4, 2.4, 6.2])
source(s, "fair_benchmark_results.txt, RTX 4070 SUPER, abdominal CT. Cache is now asserted empty before every cold arm.")

# ═══ 29 — Artifact 3 ═══════════════════════════════════════════════════════
s = slide()
head(s, "Artifact 3: Whichever Arm Ran First Paid Start-Up")
stat(s, "7.1x", "reported", L, 1.85, 2.6, vsize=48, color=MUTED)
tx(s, "What actually happened", 4.0, 1.9, 8, 0.4, size=15, bold=True, color=ACCENT)
bullets(s, [
    "CUDA context creation, cuDNN algorithm selection and kernel autotuning all happen on first use",
    "The baseline always ran first, so it absorbed all of it",
    "On the H100 this alone moved the result from 1.0x to 7.1x",
], 4.0, 2.4, 8.2, size=13, gap=0.5)

panel(s, L, 4.25, CW, 1.7)
tx(s, "The fix, and a bug it caused", L + 0.25, 4.4, 6, 0.3, size=12, bold=True, color=ACCENT)
tx(s, "Warm the GPU, the text backbone and the sliding window path before timing anything. The first\n"
      "attempt warmed with a 4x4x4 dummy volume, which crashed InstanceNorm3d: a single spatial\n"
      "element has no variance to normalise over. The warm-up has to run at the real patch size, 192\n"
      "cubed, or it is not warming the kernels that actually run.",
   L + 0.25, 4.75, 10.9, 1.2, size=13)
source(s, "Expected more than 1 spatial element. This is why warm-up code has to be as carefully written as the thing it is warming.")

# ═══ 30 — Artifact 4 ═══════════════════════════════════════════════════════
s = slide()
head(s, "Artifact 4: Precision Was Not Held Constant")
tx(s, "If one arm runs INT4 and the other runs FP16, quantization is being counted as an algorithmic gain.",
   L, 1.8, 11.3, 0.5, size=16, bold=True)
bullets(s, [
    "INT4 is a legitimate optimization, but it is a different claim from tile_step or caching",
    "Mixing them means the headline number cannot be attributed to anything in particular",
    "Both arms in the final benchmark run INT4, so what is left is algorithmic only",
    "The INT4 gain is then measured separately, against its own baseline",
], L, 2.55, 11.4, size=14, gap=0.52)

panel(s, L, 4.5, CW, 1.6)
tx(s, "The general rule this produced", L + 0.25, 4.65, 6, 0.3, size=12, bold=True, color=ACCENT)
tx(s, "Change one thing per comparison. If two things differ between arms, the result is a fact about\n"
      "the pair and cannot be attributed to either. This sounds obvious written down and was violated\n"
      "three separate ways before it was written down.",
   L + 0.25, 5.0, 10.9, 1.0, size=13)

# ═══ 31 — The rules ════════════════════════════════════════════════════════
s = slide()
head(s, "The Rules That Came Out of It")
bullets(s, [
    "Hold precision constant, so quantization cannot pose as algorithmic gain",
    "Warm the GPU, the text backbone and the sliding window path before timing",
    "Assert the embedding cache is empty before every cold measurement",
    "Assert the device of every module, rather than assuming the flag took effect",
    "Repeat to n >= 4 and quote the range, never a single run",
    "Compare an effect against run to run spread before calling it real",
    "Never compare across GPUs without saying so",
], L, 1.85, 11.4, size=15, gap=0.52)

panel(s, L, 5.55, CW, 1.15)
tx(s, "Every one of these exists because it was violated first. They are written into a preflight "
      "script that refuses to submit a job that breaks them.",
   L + 0.25, 5.75, 10.9, 0.8, size=14, bold=True)

# ═══ 32 — The validator ════════════════════════════════════════════════════
s = slide()
head(s, "Turning Rules Into a Preflight Check")
tx(s, "python validate_job.py my_benchmark.sh", L, 1.75, 8, 0.4,
   size=14, mono=True, color=ACCENT)
code(s, [
    "[  ok  ] venv matches model      voxtell env",
    "[  ok  ] HF_HOME exported",
    "[  ok  ] allocation              rrg-jma",
    "[  ok  ] gpu slice               MIG",
    "[ FAIL ] warmup present          whichever arm runs first absorbs",
    "                                 CUDA init; this inflated 1.0x to 7.1x",
    "[ FAIL ] cuda sync before timer  timings measure kernel launch only",
    "",
    "BLOCKED: 2 checks failed. Fix these before submitting.",
], L, 2.3, 11.4, 2.5, size=11)
bullets(s, [
    "Eleven checks, each one traceable to a job that already failed or a number that was already wrong",
    "Deliberately textual and simple: a validator nobody can read is a validator nobody runs",
    "Catches allocation, CPU over-request, MIG slice, unbuffered output and log paths as well",
], L, 5.1, 11.4, size=13, gap=0.5)
source(s, "voxtell-plugin/skills/voxtell-inference/scripts/validate_job.py. It also refuses shell variables in SBATCH directives, which SLURM does not expand.")

# ═══════════════════════════════════════════════════════════════════════════
divider("Results", "PART 5")
# ═══════════════════════════════════════════════════════════════════════════

# ═══ 34 — VoxTell speed ════════════════════════════════════════════════════
s = slide()
head(s, "VoxTell: Speed")
table(s, ["Change", "Effect", "DSC"], [
    ("Sliding window", "tile_step 0.75 plus crop to non-zero, 25 to 9 patches", "+0.0003"),
    ("Embedding cache", "repeat prompts return a stored tensor", "identical"),
    ("INT4 backbone", "8 GB to 2 GB VRAM, enables the 12 GB card", "0.97 agreement"),
    ("Numba preprocess", "no measurable gain at this volume size", "unchanged"),
], L, 1.75, CW, 2.3, widths=[2.7, 6.55, 2.2])

stat(s, "2.7x mean", "abdominal CT, H100 MIG  ·  2.6 to 2.8x across 4 runs",
     L, 4.55, 6, color=ACCENT)
tx(s, "One run: 3.27s to 1.28s", 7.6, 4.7, 5, 0.6, size=20, color=MUTED)
source(s, "Case CT_AMOS_amos_0018 (63x512x512). Both arms run INT4, so precision is held constant and this is algorithmic gain only.")

# ═══ 35 — VoxTell accuracy ═════════════════════════════════════════════════
s = slide()
head(s, "VoxTell: Accuracy")
stat(s, "+0.0003", "tile_step 0.5 to 0.75  ·  0.8090 to 0.8093", L, 1.9, 5.5, vsize=50)
tx(s, "65 objects across 5 abdominal CT cases", L, 3.35, 5.5, 0.4,
   size=13, color=MUTED)
bullets(s, [
    "Measured against expert annotation",
    "Same checkpoint on both arms",
    "13 abdominal organs, AMOS, seed 42",
    "No pass/fail threshold applied",
], L, 4.0, 5.4, size=12, gap=0.42)

tx(s, "One caveat, stated plainly", 7.0, 1.9, 4.8, 0.4, size=15, bold=True, color=ACCENT)
bullets(s, [
    "INT4 quantization is on by default",
    "0.97 agreement with full precision",
    "Segments 5.5% fewer voxels",
    "One sided, so bias rather than noise",
    "Measured on one case, not the full set",
], 7.0, 2.45, 5.2, size=12, gap=0.42)
source(s, "VoxTell DSC from accuracy_results.csv. The INT4 comparison is n=1 and measures output agreement, not accuracy against ground truth.")

# ═══ 36 — Reading orientation ══════════════════════════════════════════════
s = slide()
head(s, "An Accuracy Bug Worth Showing")
tx(s, "The same volume, the same prompt, the same checkpoint.", L, 1.8, 11, 0.45,
   size=17, bold=True)
for i, (v, lab, col) in enumerate([
    ("0.2978", "read with nib.load().get_fdata()", MUTED),
    ("0.9864", "read with NibabelIOWithReorient", ACCENT),
]):
    stat(s, v, lab, L + i * 5.9, 2.45, 5.6, vsize=46, color=col)

bullets(s, [
    "VoxTell was trained through a reader that reorients the volume to a canonical axis order",
    "Reading it any other way leaves the array in its stored orientation",
    "The model then segments a plausible looking fragment of the wrong thing",
    "It does not error. The only way to catch it is to score against ground truth",
], L, 4.15, 11.4, size=14, gap=0.52)
tx(s, "This is the same shape of failure as the CPU baseline: a wrong answer that looks like a right one.",
   L, 6.3, 11.4, 0.5, size=15, bold=True, color=ACCENT)
source(s, "Found while building the plugin, by scoring a demo run against ground truth rather than eyeballing the mask.")

# ═══ 37 — nnInteractive speed ══════════════════════════════════════════════
s = slide()
head(s, "nnInteractive: Compiling the Network")
tx(s, "torch.compile(session.network, mode='reduce-overhead')", L, 1.75, 9, 0.4,
   size=15, mono=True, color=ACCENT)
bullets(s, [
    "One line changed",
    "Fuses kernels and cuts per-call dispatch overhead",
], L, 2.3, 9, size=13)

stat(s, "1.33x", "per object, mean of 4 runs", L, 3.4, 5, color=ACCENT)
bullets(s, [
    "0.288s to 0.215s per object",
    "Range 1.28 to 1.39x, so a 33% gain against an 8% spread",
], L, 4.85, 5.6, size=12, gap=0.42)

stat(s, "+0.0002", "mean DSC change  ·  294 objects", 7.0, 3.4, 5)
bullets(s, [
    "No run showed degradation",
    "Speed and accuracy from the same jobs",
], 7.0, 4.85, 5.2, size=12, gap=0.42)

tx(s, "CPU time held at 6:18 to 6:33 across runs while walltime fell, "
      "which is what a GPU bound workload looks like.",
   L, 5.95, 11.4, 0.5, size=13)
source(s, "20 CT cases from the CVPR validation set, fold='all' checkpoint, H100 MIG 3g.40gb. CPU efficiency 13 to 20% of the 8 cores requested.")

# ═══ 38 — Break-even ═══════════════════════════════════════════════════════
s = slide()
head(s, "The Speedup Is Not Free at the Start")
tx(s, "Compiling costs time before the first prediction.", L, 1.8, 10, 0.5, size=18)

for i, (v, lab) in enumerate([
    ("23.6s", "one-time compile"),
    ("0.071s", "saved per object"),
    ("~22 cases", "to break even"),
]):
    stat(s, v, lab, L + i * 3.7, 2.6, 3.4,
         color=ACCENT if i == 2 else INK)

bullets(s, [
    "Batch of 881 validation cases: clearly worth it",
    "Radiologist with 3 scans: never recovered",
    "A batch optimization, and it should not be sold as anything else",
], L, 4.5, 11.3, size=15, gap=0.48)
source(s, "23.6s measured on node-local /tmp. The shared filesystem is slower, so treat it as a lower bound. About 331 objects at 14.7 objects per case.")

# ═══ 39 — Results summary ══════════════════════════════════════════════════
s = slide()
head(s, "Results Summary")
table(s, ["", "Speedup", "Accuracy", "Evidence"], [
    ("VoxTell", "2.7x  (2.6 to 2.8x)", "+0.0003 DSC", "4 speed runs, 65 objects for DSC"),
    ("nnInteractive", "1.33x  (1.28 to 1.39x)", "+0.0002 DSC", "4 runs, 294 objects"),
], L, 1.85, CW, 1.5, widths=[2.6, 3.0, 2.6, 3.25])

tx(s, "Still Open", L, 3.9, 5, 0.4, size=14, bold=True, color=ACCENT)
bullets(s, [
    "Speedup measured on a single CT volume",
    "INT4 under-segments 5.5%, on one case only",
    "Batched sliding window built but not yet paid off",
], L, 4.4, 5.9, size=12, gap=0.42, color=MUTED)

tx(s, "Next", 7.0, 3.9, 5, 0.4, size=14, bold=True, color=ACCENT)
bullets(s, [
    "Repeat across more CT volumes and prompt types",
    "Run INT4 against all 881 validation cases",
    "TensorRT FP16 engine on Fir",
], 7.0, 4.4, 5.2, size=12, gap=0.42, color=MUTED)

tx(s, "The most useful thing I built was a benchmark that kept catching itself.",
   L, 5.9, 11.5, 0.5, size=17, bold=True)

# ═══════════════════════════════════════════════════════════════════════════
divider("Deployment", "PART 6")
# ═══════════════════════════════════════════════════════════════════════════

# ═══ 41 — The gap ══════════════════════════════════════════════════════════
s = slide()
head(s, "A Model That Cannot Say No")
tx(s, "VoxTell is text prompted, so it accepts any string.", L, 1.8, 11, 0.45,
   size=18, bold=True)
bullets(s, [
    "Send an abdominal CT and ask for a brain tumour",
    "It runs. It returns a mask. The mask is of nothing",
    "Nothing in the pipeline says so, so you get a plausible result and no warning",
    "This is the same failure shape as the CPU baseline and the orientation bug",
], L, 2.5, 11.4, size=14, gap=0.52)

panel(s, L, 4.5, CW, 1.6)
tx(s, "The question Dr. Ma posed", L + 0.25, 4.68, 6, 0.3, size=12, bold=True, color=ACCENT)
tx(s, "Skills define workflow, protocol, when to use something and how to run it. Default VoxTell may\n"
      "still run a brain tumour prompt on an abdomen CT. A skill can help generate the answer that the\n"
      "model cannot segment a brain tumour from an abdominal CT.",
   L + 0.25, 5.05, 10.9, 1.0, size=13)

# ═══ 42 — Three mechanisms ═════════════════════════════════════════════════
s = slide()
head(s, "Skills, MCP Servers and Plugins")
table(s, ["Mechanism", "What it is", "What it can do"], [
    ("Skill", "Instructions loaded into the model's context",
     "Shapes how the model works: workflow, protocol, standards. Advisory"),
    ("MCP server", "A process exposing callable tools over JSON-RPC",
     "Real code with real return values. Deterministic. It can refuse"),
    ("Plugin", "A package bundling skills, MCP servers, commands and hooks",
     "One installable unit, versioned and distributable"),
], L, 1.85, CW, 2.6, widths=[2.0, 3.6, 5.85], fsize=11)

tx(s, "The distinction that matters", L, 4.7, 8, 0.4, size=16, bold=True, color=ACCENT)
tx(s, "A skill can tell the model that a brain prompt on an abdominal CT is wrong.\n"
      "Only a tool can make the call fail.",
   L, 5.2, 11.4, 0.8, size=17, bold=True)
tx(s, "One is guidance the model may or may not follow. The other is a gate it cannot walk past.",
   L, 6.1, 11.4, 0.5, size=14, color=MUTED)

# ═══ 43 — The gate ═════════════════════════════════════════════════════════
s = slide()
head(s, "The Validation Gate")
code(s, [
    "Request: REFUSED",
    "",
    "  The prompt asks for a structure in the brain (brain, brain tumor), but the",
    "  image is a CT study of the torso. That structure is not in this field of",
    "  view, so any mask returned would be meaningless.",
    "",
    "Image",
    "  modality : CT",
    "  region   : torso",
    "  shape    : (406, 512, 512)   spacing: (1.25, 0.82, 0.82)",
    "  - intensities reach -2048, consistent with Hounsfield units (air ~ -1000)",
    "  - 25.0% of voxels below -700 HU (air)",
    "  - 508 mm of coverage spans thorax and abdomen together",
], L, 1.8, CW, 3.5, size=11)
bullets(s, [
    "Returns an error, not a mask. No model is loaded and no GPU is touched",
    "The refusal states what it inferred and why, so a wrong refusal is visible rather than mysterious",
], L, 5.5, 11.4, size=13, gap=0.5)
source(s, "Real output from the installed plugin on a FLARE abdominal CT with the prompt 'brain tumor'.")

# ═══ 44 — How validation works ═════════════════════════════════════════════
s = slide()
head(s, "How the Gate Decides")
tx(s, "Intensity statistics and geometry, not a classifier. A validator nobody can audit is a validator nobody should trust.",
   L, 1.75, 11.3, 0.45, size=14)
table(s, ["Signal", "Rule", "Why"], [
    ("Modality", "CT is calibrated in Hounsfield units, so air sits near -1000",
     "MR intensities are arbitrary and rarely negative"),
    ("Region", "Air fraction plus field of view",
     "A head has almost no internal air; a torso has lungs or bowel gas"),
    ("Prompt", "Anatomy vocabulary mapping terms to regions",
     "Deliberately incomplete: an unknown term never blocks"),
], L, 2.35, CW, 2.2, widths=[1.7, 4.5, 5.25], fsize=11)

bullets(s, [
    "Refuses only on a clear mismatch. Unknown on either side proceeds with a note",
    "Adjacent regions pass, since thorax and abdomen routinely share a field of view",
    "A validator that blocks whatever it cannot classify is useless",
], L, 4.8, 11.4, size=13, gap=0.5)
source(s, "voxtell-plugin/mcp_server/validate.py. Thresholds are stated in the source and printed in every response.")

# ═══ 45 — The tools ════════════════════════════════════════════════════════
s = slide()
head(s, "What the Plugin Exposes")
table(s, ["Tool", "Purpose"], [
    ("check_request", "Validate an image and prompt pair. No compute. Returns modality, region and the reasoning"),
    ("voxtell_segment", "Text prompted segmentation. Validates first, refuses on mismatch unless forced"),
    ("nninteractive_segment", "Bounding box prompted segmentation, with box validation ahead of the GPU check"),
    ("list_models", "What is available and what each model can and cannot do"),
    ("setup", "Check the machine, report which build is live, fetch the checkpoint"),
], L, 1.85, CW, 2.9, widths=[2.9, 8.55], fsize=11)

bullets(s, [
    "Written against the MCP wire protocol directly, with no SDK dependency, so the protocol stays readable",
    "Two transports from the same routing code: stdio for Claude Code, streamable HTTP for hosted clients",
    "Installs in two commands from a GitHub marketplace, on Windows and on Ubuntu",
], L, 5.0, 11.4, size=13, gap=0.5)
source(s, "github.com/pentaexe/VoxTell-main. Verified end to end on Windows and on a clean Ubuntu machine by a second tester.")

# ═══ 46 — Bugs the work surfaced ═══════════════════════════════════════════
s = slide()
head(s, "What Packaging It Exposed")
tx(s, "Building an installer found bugs that running it locally never would.",
   L, 1.75, 11.3, 0.45, size=16, bold=True)
table(s, ["Bug", "Symptom", "Why it hid"], [
    ("numba undeclared", "Clean install could not import the predictor at all",
     "My machine had numba from something else"),
    ("bitsandbytes undeclared", "INT4 silently served FP16",
     "The loader catches every exception"),
    ("5D array to the session", "nninteractive_segment had never once run",
     "It had never been exercised"),
    ("print() to stdout", "Corrupted the JSON-RPC stream",
     "One client tolerated it"),
    ("Same name and version as PyPI", "pip could not tell fork from stock",
     "Both reported voxtell 0.1.0"),
], L, 2.4, CW, 3.1, widths=[2.7, 4.5, 4.25], fsize=10)
tx(s, "Every one of these is the same pattern as the benchmark artifacts: it worked, it looked right, "
      "and it was wrong.",
   L, 5.75, 11.4, 0.6, size=15, bold=True)

# ═══════════════════════════════════════════════════════════════════════════
divider("Limits and Next Steps", "PART 7")
# ═══════════════════════════════════════════════════════════════════════════

# ═══ 48 — Slice thickness ══════════════════════════════════════════════════
s = slide()
head(s, "Where VoxTell Falls Off")
tx(s, "Five MSD-Liver cases, prompt 'liver tumor'. Mean DSC 0.720, median 0.861.",
   L, 1.75, 11.3, 0.4, size=15)
tx(s, "The gap disappears once you sort by slice thickness.", L, 2.2, 11, 0.4,
   size=15, bold=True, color=ACCENT)

table(s, ["Case", "z-spacing", "Slices", "DSC"], [
    ("MSD-Liver_000", "5.0 mm", "75", "0.479"),
    ("MSD-Liver_001", "5.0 mm", "123", "0.520"),
    ("MSD-Liver_003", "1.0 mm", "534", "0.861"),
    ("MSD-Liver_004", "0.8 mm", "841", "0.868"),
    ("MSD-Liver_002", "1.0 mm", "517", "0.872"),
], L, 2.75, 6.4, 2.4, widths=[2.2, 1.5, 1.3, 1.4], fsize=10)

bullets(s, [
    "Both failures are the thick slice scans. Nothing overlaps",
    "The mechanism is slide 12: no resampling, fixed 192 cubed patch",
    "At 5 mm that patch spans 960 mm of anatomy",
    "A 15 mm lesion is 15 slices at 1 mm and 3 at 5 mm, so the boundary is smeared",
    "Out of distribution acquisition geometry, not a model defect",
], 7.7, 2.85, 4.7, size=12, gap=0.5)
source(s, "Independent run by a second tester on our Fir dataset. n=5, and nothing in the sample sits between 1 mm and 5 mm.")

# ═══ 49 — The gate for it ══════════════════════════════════════════════════
s = slide()
head(s, "Turning That Into a Warning")
code(s, [
    "Caveat",
    "  Coarsest voxel spacing is 5.0 mm. VoxTell v1.1 does not resample, so",
    "  accuracy falls off on thick-slice scans: measured DSC 0.48 and 0.52 on two",
    "  5 mm liver-tumour cases against 0.86-0.87 on three at 0.8-1.0 mm (n=5).",
    "  Where the falloff begins between 1 mm and 5 mm has not been measured.",
    "  Expect a usable mask on a large target and an unreliable one on a small lesion.",
], L, 1.8, CW, 1.9, size=11)

bullets(s, [
    "Fires at 3 mm or coarser, before any GPU time is spent",
    "A caveat and not a refusal: the request is legitimate and the mask is still useful on a large target",
    "The point is that it is said before a DSC is written down, rather than after",
    "3 mm is conservative, not measured. The deck and the code both say so",
], L, 4.0, 11.4, size=14, gap=0.52)

tx(s, "Same shape as the anatomy gate, applied to acquisition geometry instead.",
   L, 6.2, 11.4, 0.5, size=16, bold=True, color=ACCENT)

# ═══ 50 — Limitations ══════════════════════════════════════════════════════
s = slide()
head(s, "Limitations, Stated Plainly")
bullets(s, [
    "VoxTell speedup is measured on a single CT volume, n=4 runs. It is not a distribution",
    "The INT4 agreement figure is n=1 and measures output agreement, not accuracy against ground truth",
    "The slice thickness finding is n=5 with a plausible mechanism, which makes it a strong hypothesis rather than a result",
    "DSC scales with target size. The five liver cases ranged over 543x in ground truth volume, so their mean is not a clean summary",
    "The region heuristic is thresholds on one CT convention, and is untested on MR beyond the modality check",
    "nnInteractive numbers come from the CVPR validation set only",
    "The HTTP transport has no authentication and must not be exposed without an auth layer in front",
], L, 1.85, 11.4, size=14, gap=0.56)
tx(s, "Every number in this deck has a source line. If a figure has no provenance, it should not be here.",
   L, 6.2, 11.4, 0.5, size=15, bold=True, color=ACCENT)

# ═══ 51 — Close ════════════════════════════════════════════════════════════
s = slide()
head(s, "What I Would Take From This")
bullets(s, [
    "Both models got faster without losing accuracy: 2.7x and 1.33x, each with a DSC measurement attached",
    "The reported speedup fell by 10x under scrutiny, and not one correction changed the code",
    "The same failure shape kept recurring: a run that completes, looks right, and is wrong",
    "CPU baseline, cache mismatch, warm-up ordering, reader orientation, silent FP16 fallback, a 5D array",
    "So the durable output is not the speedup. It is the checks that make those visible",
], L, 1.9, 11.4, size=15, gap=0.6)

panel(s, L, 5.0, CW, 1.4)
tx(s, "A preflight validator, a regression suite that synthesises its own data, a build provenance check,\n"
      "and two gates that refuse work before it is wasted. Those outlast any particular number.",
   L + 0.25, 5.35, 10.9, 0.9, size=15, bold=True)

prs.save("slides.pptx")
print(f"Saved: slides.pptx  ({len(prs.slides.__iter__.__self__._sldIdLst)} slides)")
