"""Editable PowerPoint version of Figure 2 (same layout as fig_overview.tex). usage: python fig2.py OUT.pptx"""
import sys
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.dml import MSO_LINE_DASH_STYLE
from pptx.oxml.ns import qn
from lxml import etree

K = 0.65  # inches per TikZ unit
X = lambda x: 0.1 + x * K
Y = lambda y: 0.1 + (8.95 - y) * K
F = "Arial"
C = dict(rk="2F6FD0", rkl="E6EFFC", bad="D7332A", badl="FDECEB", dupl="F7C9C6", ok="2E8B3E", okd="1F6A2C", okl="EEF8EC",
         okm="BFE6B5", okn="4CCB6A", A="C9E8BE", B="CFE0F7", Cc="FBE3B5", D="E2CCF4", grd="8A8A8A", blk="222222", w="FFFFFF")
rgb = lambda h: RGBColor.from_string(h)

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.5), Inches(6.25)
s = prs.slides.add_slide(prs.slide_layouts[6])
sh = s.shapes


def shape(kind, x1, y1, x2, y2, fill=None, line=None, lw=1.0, dash=False, radius=None, name=None):
    p = sh.add_shape(kind, Inches(X(x1)), Inches(Y(y2)), Inches((x2 - x1) * K), Inches((y2 - y1) * K))
    p.shadow.inherit = False
    if fill: p.fill.solid(); p.fill.fore_color.rgb = rgb(fill)
    else: p.fill.background()
    if line:
        p.line.color.rgb = rgb(line); p.line.width = Pt(lw)
        if dash: p.line.dash_style = MSO_LINE_DASH_STYLE.DASH
    else: p.line.fill.background()
    if radius is not None and kind == MSO_SHAPE.ROUNDED_RECTANGLE: p.adjustments[0] = radius
    if name: p.name = name
    return p


def put_text(p, runs, size=14, bold=True, align=PP_ALIGN.CENTER, margin=0.0):
    tf = p.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = tf.margin_right = Inches(margin); tf.margin_top = tf.margin_bottom = 0
    if isinstance(runs, str): runs = [(runs, C["blk"])]
    lines = [[]]
    for t, c in runs:  # split runs on newlines into paragraphs
        parts = t.split("\n")
        for i, part in enumerate(parts):
            if i: lines.append([])
            if part: lines[-1].append((part, c))
    for i, ln in enumerate(lines):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.alignment = align
        for t, c in ln:
            r = para.add_run(); r.text = t
            r.font.name = F; r.font.size = Pt(size); r.font.bold = bold; r.font.color.rgb = rgb(c)


def txt(t, cx, cy, w, h, size=14, color=None, fill=None, bold=True, name=None):
    p = shape(MSO_SHAPE.RECTANGLE, cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, fill=fill, name=name)
    put_text(p, [(t, color or C["blk"])] if isinstance(t, str) else t, size=size, bold=bold)
    return p


def tag(t, cx, cy, w, h, color, line=None, fill="FFFFFF", size=12, name=None):
    p = shape(MSO_SHAPE.ROUNDED_RECTANGLE, cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, fill=fill, line=line, radius=0.25, name=name)
    put_text(p, [(t, color)] if isinstance(t, str) else t, size=size)
    return p


def arrow(x1, y1, x2, y2, color=None, lw=1.25, head=True, dash=False):
    c = sh.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(X(x1)), Inches(Y(y1)), Inches(X(x2)), Inches(Y(y2)))
    c.line.color.rgb = rgb(color or C["blk"]); c.line.width = Pt(lw)
    if dash: c.line.dash_style = MSO_LINE_DASH_STYLE.DASH
    if head:
        ln = c.line._get_or_add_ln()
        tail = etree.SubElement(ln, qn("a:tailEnd")); tail.set("type", "triangle"); tail.set("w", "med"); tail.set("len", "med")
    return c


def seg(t, x1, x2, yc, fill, line="555555", lw=0.75, dash=False, size=15):
    p = shape(MSO_SHAPE.RECTANGLE, x1, yc - 0.3, x2, yc + 0.3, fill=fill, line=line, lw=lw, dash=dash, name="segment " + t)
    put_text(p, t, size=size)


# headers and split
txt("Before the LLM call (context construction)", 9.6, 8.55, 11, 0.5, size=20)
txt("Inside the LLM", 17.9, 8.55, 4, 0.5, size=20)
arrow(15.35, -0.3, 15.35, 8.8, head=False, dash=True, lw=1)

# input document and question
for d in (0.3, 0.15, 0):
    shape(MSO_SHAPE.RECTANGLE, 0.2 + d, 2.9 + d, 1.5 + d, 4.5 + d, fill=C["w"], line=C["blk"], lw=0.75)
for y in (3.25, 3.55, 3.85, 4.15):
    arrow(0.4, y, 1.3, y, color=C["grd"], head=False)
txt("Long\ndocument\n(~12.5K\ntokens)", 0.9, 1.85, 1.9, 1.5, size=14)
tag("Question", 0.95, 6.3, 1.6, 0.55, C["blk"], line=C["blk"], size=14)

# ranker
shape(MSO_SHAPE.ROUNDED_RECTANGLE, 2.2, 2.35, 4.4, 5.1, fill=C["rkl"], line=C["rk"], lw=2, radius=0.06, name="Ranker box")
shape(MSO_SHAPE.ROUNDED_RECTANGLE, 2.2, 4.55, 4.4, 5.1, fill=C["rk"], radius=0.2, name="Ranker title bar")
txt("Any ranker", 3.3, 4.82, 2.2, 0.5, size=15, color=C["w"])
for t, y in (("BM25 / Hybrid", 4.15), ("BGE", 3.6), ("Reranker", 3.05)):
    tag(t, 3.3, y, 1.95, 0.45, C["blk"], line=C["rk"], fill="F3F7FD", size=11)
txt("swappable", 3.3, 2.57, 1.5, 0.25, size=10, color=C["rk"])
arrow(1.85, 3.75, 2.2, 3.75)
arrow(4.4, 3.75, 4.75, 3.75, head=False)
arrow(4.75, 2.0, 4.75, 5.75, head=False)
arrow(4.75, 5.75, 5.0, 5.75); arrow(4.75, 2.0, 5.0, 2.0)

# full-context path
arrow(1.1, 6.62, 1.1, 7.75, color=C["grd"], lw=2, head=False)
arrow(1.1, 7.75, 16.9, 7.75, color=C["grd"], lw=2, head=False)
arrow(16.9, 7.75, 16.9, 6.45, color=C["grd"], lw=2)
tag("Full context (12.5K tokens)", 9.6, 7.75, 4.6, 0.5, C["blk"], line=C["grd"], fill="EDEDED", size=14)

# standard chunk packing
shape(MSO_SHAPE.ROUNDED_RECTANGLE, 5.0, 4.55, 15.0, 7.2, fill=C["badl"], line=C["bad"], lw=1.25, radius=0.05, name="Chunk packing lane")
txt("Standard chunk packing", 7.6, 6.95, 5, 0.45, size=17)
seg("A", 5.4, 6.6, 5.6, C["A"])
seg("A (dup.)", 6.6, 8.1, 5.6, C["dupl"], line=C["bad"], dash=True, size=12)
seg("B", 8.1, 9.4, 5.6, C["B"])
seg("B (dup.)", 9.4, 11.0, 5.6, C["dupl"], line=C["bad"], dash=True, size=12)
seg("C", 11.0, 12.15, 5.6, C["Cc"])
seg("D", 12.15, 13.3, 5.6, C["D"])
seg("unused", 13.3, 14.55, 5.6, "E8E8E8", size=12)
tag("duplicate (23%)", 9.6, 6.45, 2.3, 0.4, C["bad"], line=C["bad"])
arrow(9.0, 6.25, 7.35, 5.92, color=C["bad"]); arrow(10.0, 6.25, 10.2, 5.92, color=C["bad"])
tag("cut mid-sentence", 12.6, 6.45, 2.4, 0.4, C["bad"], line=C["bad"])
arrow(12.3, 6.25, 12.15, 5.92, color=C["bad"]); arrow(12.9, 6.25, 13.3, 5.92, color=C["bad"])
arrow(5.4, 4.95, 14.55, 4.95, head=False, lw=0.75)
txt("1,024 tokens (fixed budget)", 9.97, 4.95, 3.8, 0.35, size=12, fill=C["badl"])

# DedupPack lane
shape(MSO_SHAPE.ROUNDED_RECTANGLE, 5.0, -0.25, 15.0, 4.3, fill=C["okl"], line=C["ok"], lw=1.25, radius=0.04, name="DedupPack lane")
txt("DedupPack (ours)", 6.9, 4.0, 3.6, 0.45, size=17)
tag("training-free  |  CPU, 0.4 s  |  plug-and-play", 11.6, 4.0, 5.4, 0.42, C["w"], fill=C["ok"])
steps = (("1. pad to\nsentence boundaries", 6.6), ("2. merge\noverlapping ranges", 9.0), ("3. exact-budget\nfill + backfill", 11.4), ("4. emit in\ndocument order", 13.8))
for t, x in steps:
    p = shape(MSO_SHAPE.ROUNDED_RECTANGLE, x - 1.12, 2.48, x + 1.13, 3.52, fill=C["w"], line=C["ok"], radius=0.12, name="Step " + t[0])
    put_text(p, t, size=12, align=PP_ALIGN.LEFT, margin=0.06)
for i in range(3):
    arrow(steps[i][1] + 1.13, 3.0, steps[i + 1][1] - 1.12, 3.0, color=C["ok"])
arrow(9.97, 2.42, 9.97, 2.0, color=C["ok"], lw=1.5)
for t, x in (("A", 5.4), ("B", 6.65), ("C", 7.9), ("D", 9.15)):
    seg(t, x, x + 1.25, 0.95, C["okm"])
seg("E", 10.4, 12.475, 0.95, C["okn"], line=C["ok"], lw=2)
seg("F", 12.475, 14.55, 0.95, C["okn"], line=C["ok"], lw=2)
tag("recovered budget → new evidence", 12.2, 1.75, 4.9, 0.4, C["ok"], line=C["ok"])
arrow(11.9, 1.55, 11.44, 1.27, color=C["ok"]); arrow(13.0, 1.55, 13.5, 1.27, color=C["ok"])
arrow(5.4, 0.3, 14.55, 0.3, head=False, lw=0.75)
txt("1,024 tokens, no duplicates, more evidence", 9.97, 0.3, 5.4, 0.35, size=12, color=C["okd"], fill=C["okl"])

# LLM
shape(MSO_SHAPE.ROUNDED_RECTANGLE, 15.9, 1.3, 17.9, 6.45, fill="E9E9E9", line=C["grd"], lw=1.5, radius=0.08, name="LLM box")
txt("LLM", 16.9, 5.8, 1.8, 0.6, size=24)
txt("Qwen / Llama\n7B–32B", 16.9, 5.05, 1.9, 0.6, size=11)
tag("prefill", 16.9, 3.9, 1.6, 0.45, C["blk"], line=C["grd"], size=14)
tag("KV cache", 16.9, 2.6, 1.6, 0.45, C["blk"], line=C["grd"], fill="C4C4C4", size=14)
arrow(16.9, 3.67, 16.9, 2.83)
arrow(15.0, 5.75, 15.9, 5.75); arrow(15.0, 2.0, 15.9, 2.0)
tag("6× smaller KV cache\nthan full context", 16.9, 0.55, 2.5, 0.65, C["okd"], line=C["ok"], fill=C["okl"], size=11)

# answers
tag([("Answer ", C["blk"]), ("?", C["grd"])], 19.2, 5.75, 1.7, 0.7, None, line=C["blk"], size=14)
tag([("Answer ", C["blk"]), ("✓", C["ok"])], 19.2, 2.0, 1.7, 0.7, None, line=C["blk"], size=14)
arrow(17.9, 5.75, 18.35, 5.75); arrow(17.9, 2.0, 18.35, 2.0)

prs.save(sys.argv[1])
print("saved", sys.argv[1])
