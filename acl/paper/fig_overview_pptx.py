"""Figure 2 (master copy): editable PowerPoint, exported to PDF for the paper.
Colour-blind-safe palette: orange = waste in standard packing, blue = DedupPack, grey = shared parts.
usage: python fig_overview_pptx.py OUT.pptx"""
import sys
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.dml import MSO_LINE_DASH_STYLE
from pptx.oxml.ns import qn
from lxml import etree

HERE = Path(__file__).parent
F = "Arial"
SCALE = 1.15  # global text scale
C = dict(ink="222222", mute="6B6B6B", line="8A8A8A", panel="F3F4F6", panel2="E6E8EC",
         org="D55E00", orgl="FDF0E7", orgm="F6C7A6",              # waste (orange)
         blu="0072B2", blul="EAF3FA", blum="9CC7E4", blud="005A8C",  # DedupPack (blue)
         seg="DCE1E8", w="FFFFFF")
rgb = lambda h: RGBColor.from_string(h)
prs = Presentation()
prs.slide_width, prs.slide_height = Inches(14.0), Inches(6.2)
s = prs.slides.add_slide(prs.slide_layouts[6])
sh = s.shapes
RECT, RRECT = MSO_SHAPE.RECTANGLE, MSO_SHAPE.ROUNDED_RECTANGLE


def box(x, y, w, h, fill=None, line=None, lw=1.0, dash=False, kind=RECT, radius=0.08, name=None):
    p = sh.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h))
    p.shadow.inherit = False
    if fill: p.fill.solid(); p.fill.fore_color.rgb = rgb(fill)
    else: p.fill.background()
    if line:
        p.line.color.rgb = rgb(line); p.line.width = Pt(lw)
        if dash: p.line.dash_style = MSO_LINE_DASH_STYLE.DASH
    else: p.line.fill.background()
    if kind == RRECT: p.adjustments[0] = radius
    if name: p.name = name
    return p


def write(p, runs, size=14, bold=True, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, ml=0.0):
    tf = p.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    tf.margin_left = Inches(ml); tf.margin_right = 0; tf.margin_top = tf.margin_bottom = 0
    if isinstance(runs, str): runs = [(runs, C["ink"])]
    lines = [[]]
    for t, c in runs:
        for i, part in enumerate(t.split("\n")):
            if i: lines.append([])
            if part: lines[-1].append((part, c))
    for i, ln in enumerate(lines):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.alignment = align
        for t, c in ln:
            r = para.add_run(); r.text = t
            r.font.name = F; r.font.size = Pt(round(size * SCALE)); r.font.bold = bold; r.font.color.rgb = rgb(c)
    return p


def text(t, x, y, w, h, size=14, color=None, bold=True, align=PP_ALIGN.CENTER, fill=None):
    return write(box(x, y, w, h, fill=fill), [(t, color or C["ink"])] if isinstance(t, str) else t, size, bold, align)


def pill(t, x, y, w, h, color, line=None, fill="FFFFFF", size=12, radius=0.3):
    return write(box(x, y, w, h, fill=fill, line=line, kind=RRECT, radius=radius), [(t, color)] if isinstance(t, str) else t, size)


def line(x1, y1, x2, y2, color=None, lw=1.25, head=True, dash=False):
    c = sh.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = rgb(color or C["ink"]); c.line.width = Pt(lw)
    if dash: c.line.dash_style = MSO_LINE_DASH_STYLE.DASH
    if head:
        e = etree.SubElement(c.line._get_or_add_ln(), qn("a:tailEnd")); e.set("type", "triangle"); e.set("w", "med"); e.set("len", "med")
    return c


def bar(x, y, segs, h=0.42):
    """segs: (label, width, fill, border, dashed, size)"""
    for lab, w, fill, bd, dash, size in segs:
        write(box(x, y, w, h, fill=fill, line=bd, lw=1.75 if dash else 0.75, dash=dash, name="segment " + lab), lab, size)
        x += w


def budget(x1, x2, y, label, color, bg):
    line(x1, y, x2, y, color=C["line"], head=False, lw=0.75)
    for xx in (x1, x2): line(xx, y - 0.07, xx, y + 0.07, color=C["line"], head=False, lw=0.75)
    text(label, (x1 + x2) / 2 - 2.5, y - 0.14, 5.0, 0.28, size=12, color=color, fill=bg)


# ------------------------------------------------------------------ grid
LX0, LX1 = 3.95, 10.95          # lanes
BX0, BW = 4.25, 6.4             # bars (1,024 tokens)
LLX, LLW = 11.45, 1.45          # LLM column
AX = 13.0                      # answers
YA, YB = 2.45, 5.25             # bar centres of the two lanes

# headers and divider
text("Before the LLM call (context construction)", 0.3, 0.12, 10.7, 0.42, size=20)
text("Inside the LLM", 11.3, 0.12, 2.6, 0.42, size=20)
line(11.25, 0.15, 11.25, 6.1, color=C["line"], head=False, dash=True, lw=1)

# input column
pill("Question", 0.35, 2.05, 1.3, 0.42, C["ink"], line=C["ink"], size=14)
for d in (0.16, 0.08, 0):
    box(0.45 + d, 2.85 - d, 0.95, 1.2, fill=C["w"], line=C["ink"], lw=0.75)
for i in range(4):
    line(0.6, 3.07 + 0.22 * i, 1.25, 3.07 + 0.22 * i, color=C["line"], head=False)
text("Long document\n(~12.5K tokens)", 0.15, 4.2, 1.7, 0.6, size=13)
line(1.6, 3.45, 1.95, 3.45)

# ranker
box(1.95, 2.6, 1.7, 1.75, fill=C["panel"], line=C["mute"], lw=1.25, kind=RRECT, radius=0.08, name="Ranker")
text("Any ranker", 1.97, 2.65, 1.3, 0.36, size=14)
for i, t in enumerate(("BM25 / Hybrid", "BGE", "Reranker")):
    pill(t, 2.08, 3.05 + 0.4 * i, 1.44, 0.32, C["ink"], line=C["line"], size=11)
line(3.65, 3.45, 3.8, 3.45, head=False)
# plug icon: the ranker is swappable
px, py = 3.36, 2.73
box(px, py, 0.16, 0.2, fill=C["mute"], kind=RRECT, radius=0.25, name="plug body")
for dy in (0.04, 0.12): box(px + 0.16, py + dy, 0.09, 0.04, fill=C["mute"])
line(px - 0.08, py + 0.1, px, py + 0.1, color=C["mute"], head=False, lw=2)
line(3.8, YA, 3.8, YB, head=False)
line(3.8, YA, LX0, YA); line(3.8, YB, LX0, YB)

# full-context path (question -> LLM)
line(1.0, 2.05, 1.0, 0.85, color=C["line"], lw=2, head=False)
line(1.0, 0.85, LLX + LLW / 2, 0.85, color=C["line"], lw=2, head=False)
line(LLX + LLW / 2, 0.85, LLX + LLW / 2, 1.3, color=C["line"], lw=2)
pill("Full context (12.5K tokens)", 5.15, 0.66, 3.8, 0.38, C["ink"], line=C["line"], fill=C["panel2"], size=13)

# lane 1: standard chunk packing
box(LX0, 1.25, LX1 - LX0, 1.95, fill=C["orgl"], line=C["org"], lw=1.25, kind=RRECT, radius=0.05, name="Chunk packing lane")
text("Standard chunk packing", LX0 + 0.15, 1.3, 3.6, 0.36, size=16, align=PP_ALIGN.LEFT)
bar(BX0, YA - 0.21, [("A", 0.9, C["seg"], C["line"], False, 15), ("A (dup.)", 1.0, C["orgm"], C["org"], True, 12),
                     ("B", 0.9, C["seg"], C["line"], False, 15), ("B (dup.)", 1.0, C["orgm"], C["org"], True, 12),
                     ("C", 0.9, C["seg"], C["line"], False, 15), ("D", 0.9, C["seg"], C["line"], False, 15),
                     ("unused", 0.8, C["w"], C["line"], True, 10.5)])
pill("duplicate (23%)", 6.0, 1.72, 1.9, 0.3, C["org"], line=C["org"], size=12)
line(6.6, 2.02, BX0 + 1.4, YA - 0.21, color=C["org"]); line(7.3, 2.02, BX0 + 3.3, YA - 0.21, color=C["org"])
pill("cut mid-sentence", 8.35, 1.72, 2.0, 0.3, C["org"], line=C["org"], size=12)
line(8.95, 2.02, BX0 + 4.7, YA - 0.21, color=C["org"]); line(9.6, 2.02, BX0 + 5.6, YA - 0.21, color=C["org"])
budget(BX0, BX0 + BW, 2.92, "1,024 tokens (fixed budget)", C["ink"], C["orgl"])
# jagged edges where pieces are cut mid-sentence
for cx in (BX0 + 4.7, BX0 + 5.6):
    zz = sh.build_freeform(Inches(cx), Inches(YA - 0.21))
    pts, yy = [], YA - 0.21
    for k in range(6):
        yy += 0.07; pts.append((Inches(cx + (0.06 if k % 2 == 0 else 0)), Inches(yy)))
    zz.add_line_segments(pts, close=False)
    z = zz.convert_to_shape(); z.fill.background(); z.line.color.rgb = rgb(C["org"]); z.line.width = Pt(2); z.shadow.inherit = False

# lane 2: DedupPack
box(LX0, 3.4, LX1 - LX0, 2.7, fill=C["blul"], line=C["blu"], lw=1.25, kind=RRECT, radius=0.04, name="DedupPack lane")
text("DedupPack (ours)", LX0 + 0.15, 3.45, 3.6, 0.36, size=16, align=PP_ALIGN.LEFT)
SW, SG, SY, SH = (BW - 3 * 0.12) / 4, 0.12, 3.88, 0.82
caps = ("sentence padding", "merge overlaps", "exact fill + backfill", "document order")
xs = [BX0 + i * (SW + SG) for i in range(4)]
for x, cap in zip(xs, caps):
    box(x, SY, SW, SH, fill=C["w"], line=C["blu"], kind=RRECT, radius=0.1, name="Step: " + cap)
    text(cap, x - 0.03, SY + 0.53, SW + 0.06, 0.24, size=9.5, color=C["blud"])
for i in range(3):
    line(xs[i] + SW, SY + 0.3, xs[i + 1], SY + 0.3, color=C["blu"], lw=1)
iy = SY + 0.27   # icon centre line
# (1) a span is padded out to the sentence boundaries
x = xs[0]
line(x + 0.15, iy, x + SW - 0.15, iy, color=C["line"], head=False, lw=0.75)
for t in (x + 0.15, x + SW - 0.15): line(t, iy - 0.14, t, iy + 0.14, head=False, lw=2)
box(x + SW / 2 - 0.2, iy - 0.07, 0.4, 0.14, fill=C["blu"])
line(x + SW / 2 - 0.22, iy, x + 0.2, iy, color=C["blu"], dash=True); line(x + SW / 2 + 0.22, iy, x + SW - 0.2, iy, color=C["blu"], dash=True)
# (2) two overlapping ranges become one
x = xs[1]
box(x + 0.12, iy - 0.15, 0.55, 0.12, fill=C["seg"], line=C["mute"], lw=0.75)
box(x + 0.42, iy + 0.03, 0.5, 0.12, fill=C["seg"], line=C["mute"], lw=0.75)
box(x + 0.42, iy - 0.03, 0.25, 0.06, fill=C["orgm"])
line(x + 0.98, iy, x + 1.13, iy, color=C["blu"])
box(x + 1.17, iy - 0.07, 0.2, 0.14, fill=C["blu"])
# (3) exact fill: a piece that does not fit is skipped, a later one backfills
x = xs[2]
box(x + 0.12, iy + 0.02, 0.75, 0.14, fill=C["seg"])
box(x + 0.87, iy + 0.02, 0.38, 0.14, fill=C["blu"])
box(x + 0.12, iy + 0.02, 1.13, 0.14, line=C["ink"], lw=0.75)
line(x + 1.25, iy - 0.05, x + 1.25, iy + 0.23, head=False, lw=2)
box(x + 0.5, iy - 0.17, 0.6, 0.11, fill=C["w"], line=C["mute"], lw=0.75)
line(x + 0.72, iy - 0.2, x + 0.88, iy - 0.03, color=C["org"], head=False, lw=1.5)
line(x + 0.72, iy - 0.03, x + 0.88, iy - 0.2, color=C["org"], head=False, lw=1.5)
# (4) rank order -> document order
x = xs[3]
for i, (l, fl) in enumerate((("3", "seg"), ("1", "seg"), ("2", "seg"))):
    write(box(x + 0.08 + 0.2 * i, iy - 0.09, 0.18, 0.18, fill=C[fl], line=C["mute"], lw=0.5), l, 9)
line(x + 0.7, iy, x + 0.84, iy, color=C["blu"])
for i, l in enumerate("123"):
    write(box(x + 0.9 + 0.2 * i, iy - 0.09, 0.18, 0.18, fill=C["blum"], line=C["blu"], lw=0.5), l, 9)

text("recovered budget → new evidence", BX0 + 2.9, 4.74, 3.5, 0.28, size=11, color=C["blud"])
bar(BX0, YB - 0.21, [("A", 0.9, C["seg"], C["line"], False, 15), ("B", 0.9, C["seg"], C["line"], False, 15),
                     ("C", 0.9, C["seg"], C["line"], False, 15), ("D", 0.9, C["seg"], C["line"], False, 15),
                     ("E", 1.4, C["blum"], C["blu"], False, 15), ("F", 1.4, C["blum"], C["blu"], False, 15)])
budget(BX0, BX0 + BW, 5.78, "1,024 tokens, no duplicates, more evidence", C["blud"], C["blul"])

# LLM column
box(LLX, 1.3, LLW, 4.8, fill=C["panel"], line=C["mute"], lw=1.25, kind=RRECT, radius=0.08, name="LLM")
lg = 0.36
for i, f in enumerate(("qwen_t.png", "meta_t.png", "mistral_t.png")):
    sh.add_picture(str(HERE / "logos" / f), Inches(LLX + 0.1 + i * (lg + 0.065)), Inches(1.45), Inches(lg), Inches(lg))
text("LLM", LLX, 1.9, LLW, 0.45, size=22)
text("Qwen2.5 / Llama-3.1\n/ Mistral, 7B–32B", LLX, 2.35, LLW, 0.45, size=8, color=C["mute"])
pill("prefill", LLX + 0.15, 3.15, LLW - 0.3, 0.38, C["ink"], line=C["line"], size=13)
line(LLX + LLW / 2, 3.53, LLX + LLW / 2, 3.9)
pill("KV cache", LLX + 0.15, 3.9, LLW - 0.3, 0.38, C["ink"], line=C["line"], fill=C["panel2"], size=13)
pill("6× smaller KV cache\nthan full context", LLX + 0.06, 4.55, LLW - 0.12, 0.5, C["blud"], line=C["blu"], fill=C["blul"], size=8, radius=0.2)
line(LX1, YA, LLX, YA); line(LX1, YB, LLX, YB)

# answers
pill([("Answer ", C["ink"]), ("?", C["mute"])], AX, YA - 0.22, 0.97, 0.44, None, line=C["ink"], size=11)
pill([("Answer ", C["ink"]), ("✓", C["blu"])], AX, YB - 0.22, 0.97, 0.44, None, line=C["ink"], size=11)
line(LLX + LLW, YA, AX, YA); line(LLX + LLW, YB, AX, YB)

prs.save(sys.argv[1])
print("saved", sys.argv[1])
