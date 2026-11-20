"""Figure 2, v4: every element explained in the picture itself.
- the long document shows the candidate spans A-F; E holds the answer (green star)
- the ranker box shows the ranking 1-6 that both packers receive
- standard packing: the overlap between A's and B's padded windows is paid twice, C and D are cut,
  budget is left over, and E does not get in -> wrong answer
- DedupPack: A, overlap, B merged into one passage; the recovered budget brings in E and F -> correct answer
- LLM column: the packed context is 12x shorter than the document (replaces prefill / KV cache)
Colour-blind-safe palette: orange = waste, blue = DedupPack, grey = shared parts, green = answer evidence.
Canvas 12.6 x 5.3 in, printed at \\textwidth (scale ~0.5): every label is >= 12.5 pt here.
usage: python fig_overview_pptx_v4.py OUT.pptx"""
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
         grn="009E73",                                               # answer evidence
         seg="D9DEE5", w="FFFFFF")
rgb = lambda h: RGBColor.from_string(h)
W, H = 12.6, 5.3
prs = Presentation()
prs.slide_width, prs.slide_height = Inches(W), Inches(H)
s = prs.slides.add_slide(prs.slide_layouts[6])
sh = s.shapes
RECT, RRECT = MSO_SHAPE.RECTANGLE, MSO_SHAPE.ROUNDED_RECTANGLE
LW = 1.75


def box(x, y, w, h, fill=None, line=None, lw=LW, dash=False, kind=RECT, radius=0.08, name=None):
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
    c = sh.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = rgb(color or C["ink"]); c.line.width = Pt(lw)
    if dash: c.line.dash_style = MSO_LINE_DASH_STYLE.DASH
    if head:
        e = etree.SubElement(c.line._get_or_add_ln(), qn("a:tailEnd")); e.set("type", "triangle"); e.set("w", "med"); e.set("len", "med")
    return c


def poly(pts, color=None, lw=LW, head=True):
    for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
        line(x1, y1, x2, y2, color=color, lw=lw, head=head and (x2, y2) == pts[-1])


def star(cx, cy, r=0.085):
    box(cx - r, cy - r, 2 * r, 2 * r, fill=C["grn"], kind=MSO_SHAPE.STAR_5_POINT, name="answer star")


def chip(lab, x, y, w=0.28, h=0.22, fill=None, bd=None, ans=False):
    write(box(x, y, w, h, fill=fill or C["seg"], line=bd or C["line"], lw=1.0, name="span " + lab), lab, 12.5)
    if ans: star(x + w, y, 0.08)


def bar(x, y, segs, h=0.44):
    """segs: (label, width, fill, border, dashed, size, colour); returns the x of every segment"""
    xs = []
    for lab, w, fill, bd, dash, size, col in segs:
        xs.append(x)
        write(box(x, y, w, h, fill=fill, line=bd, lw=2.25 if dash else 1.0, dash=dash, name="segment " + lab), [(lab, col)], size)
        x += w
    return xs


def budget(x1, x2, y, label, color, bg):
    line(x1, y, x2, y, color=C["line"], head=False, lw=1.25)
    for xx in (x1, x2): line(xx, y - 0.08, xx, y + 0.08, color=C["line"], head=False, lw=1.25)
    text(label, (x1 + x2) / 2 - 2.0, y - 0.15, 4.0, 0.3, size=13, color=color, fill=bg)


def plug(px, cy, sx, col, dark):
    """plug (body left of px + prongs) inserted into a socket that starts at sx; cy = centre line"""
    box(px - 0.24, cy - 0.16, 0.24, 0.32, fill=col, kind=RRECT, radius=0.25, name="plug body")
    for dy in (-0.09, 0.05): box(px, cy + dy, sx - px, 0.045, fill=col, name="plug prong")
    box(sx, cy - 0.21, 0.22, 0.42, fill=C["w"], line=col, lw=1.75, kind=RRECT, radius=0.25, name="socket")
    for dy in (-0.09, 0.05): box(sx + 0.03, cy + dy, 0.08, 0.045, fill=dark, name="socket slot")


# ------------------------------------------------------------------ grid
LX0, LX1 = 3.95, 9.45           # packing lanes
BX0, BW = 4.2, 5.0              # bars (1,024 tokens)
LLX, LLW = 9.8, 1.5             # LLM column
AX, AW = 11.5, 1.02             # answers
YA, YB = 1.92, 4.56             # bar centres of the two lanes
TOP, BOT = 0.74, 5.14           # lane / LLM vertical extent
G, K, O = C["seg"], C["ink"], C["org"]

text("Before the LLM call: context construction", 0.2, 0.05, 9.3, 0.4, size=19)

# ---------------------------------------------------------------- inputs
pill("Question", 0.3, 0.82, 1.35, 0.44, C["ink"], line=C["ink"], size=15)
PX0, PY0, PW, PH = 0.2, 1.5, 1.3, 2.5
for d in (0.12, 0.06, 0):
    box(PX0 + d, PY0 - d, PW, PH, fill=C["w"], line=C["ink"], lw=1.25, name="page")
for k in range(12):
    yy = 1.66 + 0.19 * k
    line(PX0 + 0.1, yy, PX0 + PW - 0.1, yy, color=C["line"], head=False, lw=1.25)
# candidate spans in the document (A and B are neighbours)
for lab, x, k in (("A", 0.3, 1), ("B", 0.68, 1), ("C", 0.95, 4), ("D", 0.45, 6), ("E", 1.02, 8), ("F", 0.6, 10)):
    chip(lab, x, 1.66 + 0.19 * k - 0.11, ans=(lab == "E"))
text("Long document\n~12.5K tokens", 0.0, 4.07, 1.7, 0.5, size=14)
star(0.3, 4.79)
text("answer evidence", 0.43, 4.66, 1.45, 0.26, size=12.5, color=C["grn"], align=PP_ALIGN.LEFT)

# ---------------------------------------------------------------- ranker
RX, RY, RWd, RH = 2.0, 1.45, 1.7, 2.3
box(RX, RY, RWd, RH, fill=C["panel"], line=C["mute"], kind=RRECT, radius=0.08, name="Ranker")
text("Ranker", RX, RY + 0.07, RWd, 0.36, size=15)
text("lexical, BGE,\nreranker, …", RX, RY + 0.45, RWd, 0.42, size=12.5, color=C["mute"])
for i, lab in enumerate("ABCDEF"):   # the ranking both packers receive
    nx, y = RX + 0.25 + (i // 3) * 0.72, RY + 0.97 + (i % 3) * 0.32
    text(str(i + 1), nx, y, 0.18, 0.22, size=12.5, color=C["mute"])
    chip(lab, nx + 0.2, y, ans=(lab == "E"))
text("unchanged", RX, RY + 1.93, RWd, 0.26, size=12.5, color=C["mute"])
poly([(1.65, 1.04), (RX + RWd / 2, 1.04), (RX + RWd / 2, RY)])
line(PX0 + PW + 0.13, 2.75, RX, 2.75)

# ---------------------------------------------------------------- packing step slot
FX0, FX1 = LX0 - 0.08, LX1 + 0.07
box(FX0, TOP - 0.12, FX1 - FX0, BOT - TOP + 0.18, line=C["line"], lw=1.25, dash=True, kind=RRECT, radius=0.03, name="Packing step slot")
text("packing step", LX0 + 0.1, TOP - 0.23, 1.24, 0.22, size=12.5, color=C["mute"], fill=C["w"])

# ranker -> standard packing: fixed wire; ranker -> DedupPack: cable ending in a plug
line(RX + RWd, YA, LX0, YA)
CX, PX = RX + 0.35, LX0 - 0.21
line(CX, RY + RH, CX, YB, color=C["blu"], head=False, lw=2.5)
line(CX, YB, PX - 0.24, YB, color=C["blu"], head=False, lw=2.5)
text("plugs in after\nany ranker", CX - 0.1, YB + 0.14, 1.3, 0.44, size=12.5, color=C["blud"])

# ---------------------------------------------------------------- lane 1: standard chunk packing
box(LX0, TOP, LX1 - LX0, 1.75, fill=C["orgl"], line=C["org"], kind=RRECT, radius=0.05, name="Chunk packing lane")
text("Standard chunk packing", LX0 + 0.15, TOP + 0.06, 3.0, 0.36, size=16, align=PP_ALIGN.LEFT)
xa = bar(BX0, YA - 0.22, [("A", 0.55, G, C["line"], False, 15, K),
                          ("overlap", 0.78, C["orgm"], O, True, 12.5, K), ("overlap", 0.78, C["orgm"], O, True, 12.5, K),
                          ("B", 0.55, G, C["line"], False, 15, K),
                          ("C…", 0.8, G, C["line"], False, 15, K), ("…D…", 0.8, G, C["line"], False, 15, K),
                          ("", 0.74, C["w"], C["line"], True, 12.5, C["mute"])])
PY, BT = TOP + 0.44, YA - 0.22
pill("23% duplicated", 4.7, PY, 1.6, 0.32, O, line=O, size=12.5)
line(5.2, PY + 0.32, xa[1] + 0.39, BT, color=O); line(5.8, PY + 0.32, xa[2] + 0.39, BT, color=O)
pill("cut mid-sentence", 6.55, PY, 1.65, 0.32, O, line=O, size=12.5)
line(7.1, PY + 0.32, xa[5], BT, color=O); line(7.75, PY + 0.32, xa[6], BT, color=O)
pill("unused", 8.32, PY, 1.0, 0.32, O, line=O, size=12.5)
line(8.85, PY + 0.32, 8.85, BT, color=O)
budget(BX0, BX0 + BW, YA + 0.4, "1,024 tokens", C["ink"], C["orgl"])
for cx in (xa[5], xa[6]):  # jagged edges where pieces are cut mid-sentence
    zz = sh.build_freeform(Inches(cx), Inches(BT))
    pts, yy = [], BT
    for k in range(6):
        yy += 0.0733; pts.append((Inches(cx + (0.06 if k % 2 == 0 else 0)), Inches(yy)))
    zz.add_line_segments(pts, close=False)
    z = zz.convert_to_shape(); z.fill.background(); z.line.color.rgb = rgb(O); z.line.width = Pt(2.5); z.shadow.inherit = False

# ---------------------------------------------------------------- lane 2: DedupPack
L2 = 2.67
box(LX0, L2, LX1 - LX0, BOT - L2, fill=C["blul"], line=C["blu"], lw=2.25, kind=RRECT, radius=0.04, name="DedupPack lane")
text("DedupPack (ours)", LX0 + 0.15, L2 + 0.06, 2.2, 0.36, size=16, align=PP_ALIGN.LEFT)
text("training-free · no model calls", LX0 + 2.5, L2 + 0.08, 2.9, 0.32, size=12.5, color=C["blud"], align=PP_ALIGN.LEFT)
SG = 0.1; SW = (BW - 3 * SG) / 4; SY, SH = L2 + 0.48, 0.9
caps = ("sentence\npadding", "merge\noverlaps", "exact fill\n+ backfill", "document\norder")
xs = [BX0 + i * (SW + SG) for i in range(4)]
for x, cap in zip(xs, caps):
    box(x, SY, SW, SH, fill=C["w"], line=C["blu"], lw=1.5, kind=RRECT, radius=0.1, name="Step: " + cap)
    text(cap, x, SY + 0.37, SW, 0.5, size=12.5, color=C["blud"])
for i in range(3):
    line(xs[i] + SW, SY + 0.22, xs[i + 1], SY + 0.22, color=C["blu"], lw=1.5)
iy = SY + 0.22
x = xs[0]   # (1) a span is padded out to the sentence boundaries
line(x + 0.15, iy, x + SW - 0.15, iy, color=C["line"], head=False, lw=1)
for t in (x + 0.15, x + SW - 0.15): line(t, iy - 0.12, t, iy + 0.12, head=False, lw=2.5)
box(x + SW / 2 - 0.17, iy - 0.06, 0.34, 0.12, fill=C["blu"])
line(x + SW / 2 - 0.19, iy, x + 0.21, iy, color=C["blu"], dash=True, lw=1.5); line(x + SW / 2 + 0.19, iy, x + SW - 0.21, iy, color=C["blu"], dash=True, lw=1.5)
x = xs[1]   # (2) two overlapping ranges become one
box(x + 0.1, iy - 0.14, 0.45, 0.11, fill=C["seg"], line=C["mute"], lw=1)
box(x + 0.33, iy + 0.03, 0.42, 0.11, fill=C["seg"], line=C["mute"], lw=1)
box(x + 0.33, iy - 0.03, 0.22, 0.06, fill=C["orgm"])
line(x + 0.8, iy, x + 0.92, iy, color=C["blu"], lw=1.5)
box(x + 0.95, iy - 0.06, 0.13, 0.12, fill=C["blu"])
x = xs[2]   # (3) exact fill: a piece that does not fit is skipped, a later one backfills
box(x + 0.1, iy + 0.0, 0.62, 0.13, fill=C["seg"])
box(x + 0.72, iy + 0.0, 0.3, 0.13, fill=C["blu"])
box(x + 0.1, iy + 0.0, 0.92, 0.13, line=C["ink"], lw=1)
line(x + 1.02, iy - 0.06, x + 1.02, iy + 0.19, head=False, lw=2.5)
box(x + 0.4, iy - 0.17, 0.5, 0.1, fill=C["w"], line=C["mute"], lw=1)
line(x + 0.57, iy - 0.2, x + 0.73, iy - 0.04, color=O, head=False, lw=2)
line(x + 0.57, iy - 0.04, x + 0.73, iy - 0.2, color=O, head=False, lw=2)
x = xs[3]   # (4) rank order -> document order
for i, l in enumerate("312"):
    write(box(x + 0.06 + 0.16 * i, iy - 0.08, 0.15, 0.16, fill=C["seg"], line=C["mute"], lw=0.75), l, 8)
line(x + 0.55, iy, x + 0.63, iy, color=C["blu"], lw=1.5)
for i, l in enumerate("123"):
    write(box(x + 0.66 + 0.16 * i, iy - 0.08, 0.15, 0.16, fill=C["blum"], line=C["blu"], lw=0.75), l, 8)

text("recovered budget → new evidence", BX0 + 1.5, SY + SH + 0.03, 3.5, 0.26, size=13, color=C["blud"], align=PP_ALIGN.RIGHT)
xb = bar(BX0, YB - 0.22, [("A", 0.55, G, C["line"], False, 15, K), ("overlap", 0.78, G, C["line"], False, 12.5, C["mute"]),
                          ("B", 0.55, G, C["line"], False, 15, K),
                          ("C", 0.75, G, C["line"], False, 15, K), ("D", 0.75, G, C["line"], False, 15, K),
                          ("E", 0.82, C["blum"], C["blu"], False, 15, K), ("F", 0.8, C["blum"], C["blu"], False, 15, K)])
box(BX0, YB - 0.22, xb[3] - BX0, 0.44, line=C["blu"], lw=2.25, name="merged passage A+B")
star(xb[6] - 0.13, YB - 0.22 + 0.11)
budget(BX0, BX0 + BW, YB + 0.37, "1,024 tokens: whole sentences, no duplicates", C["blud"], C["blul"])
plug(PX, YB, LX0 - 0.1, C["blu"], C["blud"])

# ---------------------------------------------------------------- LLM
box(LLX, TOP, LLW, BOT - TOP, fill=C["panel"], line=C["mute"], kind=RRECT, radius=0.07, name="LLM")
text("LLM", LLX, TOP + 0.1, LLW, 0.42, size=22)
lg = 0.36
for i, (f, nm) in enumerate((("qwen_t.png", "Qwen2.5"), ("meta_t.png", "Llama-3.1"))):
    ly = TOP + 0.58 + i * 0.72
    sh.add_picture(str(HERE / "logos" / f), Inches(LLX + (LLW - lg) / 2), Inches(ly), Inches(lg), Inches(lg))
    text(nm, LLX, ly + lg, LLW, 0.28, size=12.5, color=C["mute"])
# input length: the whole document vs the packed context
text("input tokens", LLX, 2.92, LLW, 0.26, size=12.5, color=C["mute"])
box(LLX + 0.1, 3.24, 0.82, 0.22, fill=C["seg"], line=C["line"], lw=1.0, name="document length")
text("12.5K", LLX + 0.96, 3.22, 0.5, 0.26, size=12.5, color=C["ink"], align=PP_ALIGN.LEFT)
box(LLX + 0.1, 3.54, 0.07, 0.22, fill=C["blu"], name="packed length")
text("1,024", LLX + 0.22, 3.52, 0.6, 0.26, size=12.5, color=C["blud"], align=PP_ALIGN.LEFT)
text("12× less input", LLX, 3.86, LLW, 0.28, size=12.5, color=C["blud"])
text("unchanged", LLX, BOT - 0.42, LLW, 0.28, size=12.5, color=C["mute"])
line(LX1, YA, LLX, YA); line(LX1, YB, LLX, YB)

# ---------------------------------------------------------------- answers
pill([("Answer ", C["ink"]), ("✗", C["org"])], AX, YA - 0.23, AW, 0.46, None, line=C["ink"], size=13)
text([("E", C["ink"]), ("★", C["grn"]), (" missing", C["mute"])], AX - 0.05, YA + 0.3, AW + 0.1, 0.26, size=12.5)
pill([("Answer ", C["ink"]), ("✓", C["blu"])], AX, YB - 0.23, AW, 0.46, None, line=C["blu"], size=13, lw=2.25)
text([("E", C["ink"]), ("★", C["grn"]), (" included", C["blud"])], AX - 0.05, YB + 0.3, AW + 0.1, 0.26, size=12.5)
line(LLX + LLW, YA, AX, YA); line(LLX + LLW, YB, AX, YB)

prs.save(sys.argv[1])
print("saved", sys.argv[1])
