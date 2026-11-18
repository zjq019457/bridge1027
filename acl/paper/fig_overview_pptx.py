"""Figure 2 (master copy): editable PowerPoint, exported to PDF for the paper.
Colour-blind-safe palette: orange = waste in standard packing, blue = DedupPack, grey = shared parts.
Canvas 12.4 x 5.5 in, printed at \\textwidth (~6.3 in, scale ~0.51): every label is >= 12.5 pt here (>= 6.3 pt in print).
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
F = "Segoe UI"
C = dict(ink="1F2328", mute="5F6670", line="7D848D", panel="F4F5F7", panel2="E4E7EB",
         org="D55E00", orgl="FDF1E8", orgm="F5C29E",              # waste (orange)
         blu="0072B2", blul="EAF3FA", blum="94C3E3", blud="00507D",  # DedupPack (blue)
         seg="D9DEE5", w="FFFFFF")
rgb = lambda h: RGBColor.from_string(h)
DY = 0.4   # everything below the headers moves up (full-context path removed)
W, H = 12.6, 5.5 - DY
prs = Presentation()
prs.slide_width, prs.slide_height = Inches(W), Inches(H)
s = prs.slides.add_slide(prs.slide_layouts[6])
sh = s.shapes
RECT, RRECT, OVAL = MSO_SHAPE.RECTANGLE, MSO_SHAPE.ROUNDED_RECTANGLE, MSO_SHAPE.OVAL
LW = 1.75  # default stroke


sy = lambda y: y - DY if y > 0.45 else y  # header row (y <= 0.45) stays put


def box(x, y, w, h, fill=None, line=None, lw=LW, dash=False, kind=RECT, radius=0.08, name=None):
    y = sy(y)
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


def write(p, runs, size=14, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE):
    tf = p.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    if isinstance(runs, str): runs = [(runs, C["ink"])]
    lines = [[]]
    for t, c in runs:
        for i, part in enumerate(t.split("\n")):
            if i: lines.append([])
            if part: lines[-1].append((part, c))
    for i, ln in enumerate(lines):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.alignment = align; para.line_spacing = 0.95
        for t, c in ln:
            r = para.add_run(); r.text = t
            r.font.name = F; r.font.size = Pt(size); r.font.bold = True; r.font.color.rgb = rgb(c)
    return p


def text(t, x, y, w, h, size=14, color=None, align=PP_ALIGN.CENTER, fill=None):
    return write(box(x, y, w, h, fill=fill), [(t, color or C["ink"])] if isinstance(t, str) else t, size, align)


def pill(t, x, y, w, h, color, line=None, fill="FFFFFF", size=13, radius=0.5, lw=LW):
    return write(box(x, y, w, h, fill=fill, line=line, lw=lw, kind=RRECT, radius=radius), [(t, color)] if isinstance(t, str) else t, size)


def line(x1, y1, x2, y2, color=None, lw=LW, head=True, dash=False):
    y1, y2 = sy(y1), sy(y2)
    c = sh.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = rgb(color or C["ink"]); c.line.width = Pt(lw)
    if dash: c.line.dash_style = MSO_LINE_DASH_STYLE.DASH
    if head:
        e = etree.SubElement(c.line._get_or_add_ln(), qn("a:tailEnd")); e.set("type", "triangle"); e.set("w", "med"); e.set("len", "med")
    return c


def poly(pts, color=None, lw=LW, head=True):
    """elbow connector through the given points; arrow head on the last segment"""
    for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
        line(x1, y1, x2, y2, color=color, lw=lw, head=head and (x2, y2) == pts[-1])


def bar(x, y, segs, h=0.44):
    """segs: (label, width, fill, border, dashed, size, colour)"""
    for lab, w, fill, bd, dash, size, col in segs:
        write(box(x, y, w, h, fill=fill, line=bd, lw=2.25 if dash else 1.0, dash=dash, name="segment " + lab), [(lab, col)], size)
        x += w


def budget(x1, x2, y, label, color, bg):
    line(x1, y, x2, y, color=C["line"], head=False, lw=1.25)
    for xx in (x1, x2): line(xx, y - 0.08, xx, y + 0.08, color=C["line"], head=False, lw=1.25)
    text(label, (x1 + x2) / 2 - 2.0, y - 0.15, 4.0, 0.3, size=13, color=color, fill=bg)


def plug(px, py, col):
    """small plug icon: DedupPack plugs in after any ranker"""
    box(px, py, 0.2, 0.24, fill=col, kind=RRECT, radius=0.3, name="plug body")
    for dy in (0.05, 0.15): box(px + 0.2, py + dy, 0.1, 0.045, fill=col)
    line(px - 0.12, py + 0.12, px, py + 0.12, color=col, head=False, lw=2.5)


# ------------------------------------------------------------------ grid
LX0, LX1 = 3.95, 9.45           # lanes
BX0, BW = 4.2, 5.0              # bars (1,024 tokens)
LLX, LLW = 9.8, 1.5             # LLM column
AX, AW = 11.5, 1.02            # answers
YA, YB = 2.2, 4.84            # bar centres of the two lanes
TOP, BOT = 1.02, 5.42          # lane / LLM vertical extent

# headers and divider
text("Before the LLM call: context construction", 0.2, 0.05, 9.3, 0.4, size=19)
text("Inside the LLM", 9.65, 0.05, 2.7, 0.4, size=19)
line(9.62, 0.12, 9.62, BOT, color=C["line"], head=False, dash=True, lw=1.25)

# input column
pill("Question", 0.3, 1.1, 1.35, 0.44, C["ink"], line=C["ink"], size=15)
for d in (0.16, 0.08, 0):
    box(0.5 + d, 2.25 - d, 0.95, 1.15, fill=C["w"], line=C["ink"], lw=1.25)
for i in range(4):
    line(0.65, 2.47 + 0.21 * i, 1.3, 2.47 + 0.21 * i, color=C["line"], head=False, lw=1.5)
text("Long document\n~12.5K tokens", 0.1, 3.5, 1.85, 0.55, size=14)

# ranker (question and document both go in)
RX, RY, RWd, RH = 2.05, 1.95, 1.65, 2.1
box(RX, RY, RWd, RH, fill=C["panel"], line=C["mute"], kind=RRECT, radius=0.08, name="Ranker")
text("Ranker", RX, RY + 0.08, RWd, 0.36, size=15)
for i, t in enumerate(("Lexical / Hybrid", "BGE", "Reranker")):
    pill(t, RX + 0.1, RY + 0.55 + 0.47 * i, RWd - 0.2, 0.36, C["ink"], line=C["line"], size=12.5, lw=1.25)
poly([(1.65, 1.32), (RX + RWd / 2, 1.32), (RX + RWd / 2, RY)])
line(1.62, 2.82, RX, 2.82)
# ranker -> both lanes
JX = 3.8
line(RX + RWd, 3.0, JX, 3.0, head=False)
line(JX, YA, JX, YB, head=False)
line(JX, YA, LX0, YA); line(JX, YB, LX0, YB)

# lane 1: standard chunk packing
box(LX0, TOP, LX1 - LX0, 1.75, fill=C["orgl"], line=C["org"], kind=RRECT, radius=0.05, name="Chunk packing lane")
text("Standard chunk packing", LX0 + 0.15, TOP + 0.06, 3.0, 0.36, size=16, align=PP_ALIGN.LEFT)
O, G, K = C["org"], C["seg"], C["ink"]
bar(BX0, YA - 0.22, [("A", 0.65, G, C["line"], False, 15, K), ("A dup.", 0.8, C["orgm"], O, True, 13, K),
                     ("B", 0.65, G, C["line"], False, 15, K), ("B dup.", 0.8, C["orgm"], O, True, 13, K),
                     ("C…", 0.65, G, C["line"], False, 15, K), ("…D…", 0.65, G, C["line"], False, 15, K),
                     ("", 0.8, C["w"], C["line"], True, 12.5, C["mute"])])
# three waste callouts in the same style: duplicated, cut mid-sentence, unused
PY = TOP + 0.44
pill("23% duplicated", 4.75, PY, 1.6, 0.32, O, line=O, size=12.5)
line(5.15, PY + 0.32, BX0 + 1.05, YA - 0.22, color=O); line(6.0, PY + 0.32, BX0 + 2.5, YA - 0.22, color=O)
pill("cut mid-sentence", 6.5, PY, 1.65, 0.32, O, line=O, size=12.5)
line(7.35, PY + 0.32, BX0 + 3.55, YA - 0.22, color=O); line(7.95, PY + 0.32, BX0 + 4.2, YA - 0.22, color=O)
pill("unused", 8.3, PY, 1.0, 0.32, O, line=O, size=12.5)
line(8.8, PY + 0.32, BX0 + 4.6, YA - 0.22, color=O)
budget(BX0, BX0 + BW, 2.6, "1,024 tokens", C["ink"], C["orgl"])
for cx in (BX0 + 3.55, BX0 + 4.2):  # jagged edges where pieces are cut mid-sentence
    zz = sh.build_freeform(Inches(cx), Inches(sy(YA - 0.22)))
    pts, yy = [], sy(YA - 0.22)
    for k in range(6):
        yy += 0.0733; pts.append((Inches(cx + (0.06 if k % 2 == 0 else 0)), Inches(yy)))
    zz.add_line_segments(pts, close=False)
    z = zz.convert_to_shape(); z.fill.background(); z.line.color.rgb = rgb(O); z.line.width = Pt(2.5); z.shadow.inherit = False

# lane 2: DedupPack
L2 = 2.95
box(LX0, L2, LX1 - LX0, BOT - L2, fill=C["blul"], line=C["blu"], lw=2.25, kind=RRECT, radius=0.04, name="DedupPack lane")
text("DedupPack (ours)", LX0 + 0.15, L2 + 0.06, 2.2, 0.36, size=16, align=PP_ALIGN.LEFT)
plug(LX0 + 2.45, L2 + 0.12, C["blu"])
text("plugs in after any ranker", LX0 + 2.85, L2 + 0.08, 2.4, 0.32, size=12.5, color=C["blud"], align=PP_ALIGN.LEFT)
SG = 0.1; SW = (BW - 3 * SG) / 4; SY, SH = L2 + 0.48, 0.9
caps = ("sentence\npadding", "merge\noverlaps", "exact fill\n+ backfill", "document\norder")
xs = [BX0 + i * (SW + SG) for i in range(4)]
for x, cap in zip(xs, caps):
    box(x, SY, SW, SH, fill=C["w"], line=C["blu"], lw=1.5, kind=RRECT, radius=0.1, name="Step: " + cap)
    text(cap, x, SY + 0.37, SW, 0.5, size=12.5, color=C["blud"])
for i in range(3):
    line(xs[i] + SW, SY + 0.22, xs[i + 1], SY + 0.22, color=C["blu"], lw=1.5)
iy = SY + 0.22   # icon centre line
# (1) a span is padded out to the sentence boundaries
x = xs[0]
line(x + 0.15, iy, x + SW - 0.15, iy, color=C["line"], head=False, lw=1)
for t in (x + 0.15, x + SW - 0.15): line(t, iy - 0.12, t, iy + 0.12, head=False, lw=2.5)
box(x + SW / 2 - 0.17, iy - 0.06, 0.34, 0.12, fill=C["blu"])
line(x + SW / 2 - 0.19, iy, x + 0.21, iy, color=C["blu"], dash=True, lw=1.5); line(x + SW / 2 + 0.19, iy, x + SW - 0.21, iy, color=C["blu"], dash=True, lw=1.5)
# (2) two overlapping ranges become one
x = xs[1]
box(x + 0.1, iy - 0.14, 0.45, 0.11, fill=C["seg"], line=C["mute"], lw=1)
box(x + 0.33, iy + 0.03, 0.42, 0.11, fill=C["seg"], line=C["mute"], lw=1)
box(x + 0.33, iy - 0.03, 0.22, 0.06, fill=C["orgm"])
line(x + 0.8, iy, x + 0.92, iy, color=C["blu"], lw=1.5)
box(x + 0.95, iy - 0.06, 0.13, 0.12, fill=C["blu"])
# (3) exact fill: a piece that does not fit is skipped, a later one backfills
x = xs[2]
box(x + 0.1, iy + 0.0, 0.62, 0.13, fill=C["seg"])
box(x + 0.72, iy + 0.0, 0.3, 0.13, fill=C["blu"])
box(x + 0.1, iy + 0.0, 0.92, 0.13, line=C["ink"], lw=1)
line(x + 1.02, iy - 0.06, x + 1.02, iy + 0.19, head=False, lw=2.5)
box(x + 0.4, iy - 0.17, 0.5, 0.1, fill=C["w"], line=C["mute"], lw=1)
line(x + 0.57, iy - 0.2, x + 0.73, iy - 0.04, color=O, head=False, lw=2)
line(x + 0.57, iy - 0.04, x + 0.73, iy - 0.2, color=O, head=False, lw=2)
# (4) rank order -> document order
x = xs[3]
for i, l in enumerate("312"):
    write(box(x + 0.06 + 0.16 * i, iy - 0.08, 0.15, 0.16, fill=C["seg"], line=C["mute"], lw=0.75), l, 8)
line(x + 0.55, iy, x + 0.63, iy, color=C["blu"], lw=1.5)
for i, l in enumerate("123"):
    write(box(x + 0.66 + 0.16 * i, iy - 0.08, 0.15, 0.16, fill=C["blum"], line=C["blu"], lw=0.75), l, 8)

text("recovered budget → new evidence", BX0 + 1.5, SY + SH + 0.03, 3.5, 0.26, size=13, color=C["blud"], align=PP_ALIGN.RIGHT)
bar(BX0, YB - 0.22, [("A", 0.7, G, C["line"], False, 15, K), ("B", 0.7, G, C["line"], False, 15, K),
                     ("C", 0.7, G, C["line"], False, 15, K), ("D", 0.7, G, C["line"], False, 15, K),
                     ("E", 1.1, C["blum"], C["blu"], False, 15, K), ("F", 1.1, C["blum"], C["blu"], False, 15, K)])
budget(BX0, BX0 + BW, 5.21, "1,024 tokens: whole sentences, no duplicates", C["blud"], C["blul"])

# LLM column
box(LLX, TOP, LLW, BOT - TOP, fill=C["panel"], line=C["mute"], kind=RRECT, radius=0.07, name="LLM")
text("LLM", LLX, TOP + 0.1, LLW, 0.42, size=22)
lg = 0.36
for i, (f, nm) in enumerate((("qwen_t.png", "Qwen2.5"), ("meta_t.png", "Llama-3.1"))):
    ly = TOP + 0.58 + i * 0.72
    sh.add_picture(str(HERE / "logos" / f), Inches(LLX + (LLW - lg) / 2), Inches(sy(ly)), Inches(lg), Inches(lg))
    text(nm, LLX, ly + lg + 0.0, LLW, 0.28, size=12.5, color=C["mute"])
pill("prefill", LLX + 0.17, 3.0, LLW - 0.34, 0.38, C["ink"], line=C["line"], size=14)
line(LLX + LLW / 2, 3.38, LLX + LLW / 2, 3.6)
pill("KV cache", LLX + 0.17, 3.6, LLW - 0.34, 0.38, C["ink"], line=C["line"], fill=C["panel2"], size=14)
line(LX1, YA, LLX, YA); line(LX1, YB, LLX, YB)

# answers
pill([("Answer ", C["ink"]), ("?", C["mute"])], AX, YA - 0.23, AW, 0.46, None, line=C["ink"], size=13)
pill([("Answer ", C["ink"]), ("✓", C["blu"])], AX, YB - 0.23, AW, 0.46, None, line=C["blu"], size=13, lw=2.25)
line(LLX + LLW, YA, AX, YA); line(LLX + LLW, YB, AX, YB)

prs.save(sys.argv[1])
print("saved", sys.argv[1])
