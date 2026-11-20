"""Figure 2, v5: the v4 content in the flat card style of the user's reference picture
(header pills with token badges, passages drawn as text cards, hatched duplicates, tags under the bar, legend row).
Colour-blind-safe accents: orange = waste, blue = DedupPack, green = answer evidence, grey = shared parts.
Canvas 12.6 x 5.52 in, printed at \\textwidth (scale ~0.5): every label is >= 12.5 pt here.
usage: python fig_overview_pptx_v5.py OUT.pptx"""
import sys
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.dml import MSO_LINE_DASH_STYLE, MSO_PATTERN
from pptx.oxml.ns import qn
from lxml import etree

HERE = Path(__file__).parent
F = "Segoe UI"
C = dict(ink="1F2328", mute="5F6670", frame="9AA1AA",
         seg="E6E9ED", segb="9AA1AA", ph="C5CAD1", card="F1F3F5",
         org="D55E00", peach="FBD9BC", peachb="E39A63", hatchb="FDEBDD", tag="FCE3D0", pph="EBAE80",
         blu="0072B2", bluf="DCEBF8", blum="B5D4EE", blud="00507D", bpill="CFE3F7", bph="9FC3E6",
         grn="009E73", grnf="DCF2E7", gph="93D2B5",
         lav="DDE3F6", lavb="5A6AA0", w="FFFFFF")
rgb = lambda h: RGBColor.from_string(h)
W, H = 12.6, 5.52
prs = Presentation()
prs.slide_width, prs.slide_height = Inches(W), Inches(H)
s = prs.slides.add_slide(prs.slide_layouts[6])
sh = s.shapes
RECT, RRECT, OVAL = MSO_SHAPE.RECTANGLE, MSO_SHAPE.ROUNDED_RECTANGLE, MSO_SHAPE.OVAL
L, CTR = PP_ALIGN.LEFT, PP_ALIGN.CENTER


def box(x, y, w, h, fill=None, line=None, lw=1.75, dash=False, kind=RECT, radius=0.08, name=None, hatch=None):
    p = sh.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h))
    p.shadow.inherit = False
    if hatch:
        p.fill.patterned(); p.fill.pattern = MSO_PATTERN.WIDE_UPWARD_DIAGONAL
        p.fill.fore_color.rgb = rgb(hatch[0]); p.fill.back_color.rgb = rgb(hatch[1])
    elif fill: p.fill.solid(); p.fill.fore_color.rgb = rgb(fill)
    else: p.fill.background()
    if line:
        p.line.color.rgb = rgb(line); p.line.width = Pt(lw)
        if dash: p.line.dash_style = MSO_LINE_DASH_STYLE.DASH
    else: p.line.fill.background()
    if kind == RRECT: p.adjustments[0] = radius
    if name: p.name = name
    return p


def write(p, runs, size=14, align=CTR, bold=True, anchor=MSO_ANCHOR.MIDDLE):
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
            r.font.name = F; r.font.size = Pt(size); r.font.bold = bold; r.font.color.rgb = rgb(c)
    return p


def text(t, x, y, w, h, size=14, color=None, align=CTR, fill=None, bold=True):
    return write(box(x, y, w, h, fill=fill), [(t, color or C["ink"])] if isinstance(t, str) else t, size, align, bold)


def pill(t, x, y, w, h, color, fill, line=None, size=13, radius=0.3, lw=1.5, bold=True):
    return write(box(x, y, w, h, fill=fill, line=line, lw=lw, kind=RRECT, radius=radius), [(t, color)] if isinstance(t, str) else t, size, bold=bold)


def line(x1, y1, x2, y2, color=None, lw=1.75, head=True, dash=False):
    c = sh.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = rgb(color or C["ink"]); c.line.width = Pt(lw)
    if dash: c.line.dash_style = MSO_LINE_DASH_STYLE.DASH
    if head:
        e = etree.SubElement(c.line._get_or_add_ln(), qn("a:tailEnd")); e.set("type", "triangle"); e.set("w", "med"); e.set("len", "med")
    return c


def poly(pts, color=None, lw=1.75, head=True):
    for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
        line(x1, y1, x2, y2, color=color, lw=lw, head=head and (x2, y2) == pts[-1])


def star(cx, cy, r=0.085, hollow=False):
    if hollow: box(cx - r, cy - r, 2 * r, 2 * r, line=C["mute"], lw=1.5, kind=MSO_SHAPE.STAR_5_POINT, name="empty star")
    else: box(cx - r, cy - r, 2 * r, 2 * r, fill=C["grn"], kind=MSO_SHAPE.STAR_5_POINT, name="answer star")


def ph_lines(x, y, w, col, n=2):
    """grey placeholder lines that stand for running text"""
    for i in range(n):
        box(x, y + 0.115 * i, w * (1.0 if i == 0 else 0.62), 0.045, fill=col, kind=RRECT, radius=0.5, name="text line")


def zigzag(cx, y, h, color, lw=2.25):
    zz = sh.build_freeform(Inches(cx), Inches(y))
    pts, yy, n = [], y, 7
    for k in range(n):
        yy += h / n; pts.append((Inches(cx + (0.06 if k % 2 == 0 else 0)), Inches(yy)))
    zz.add_line_segments(pts, close=False)
    z = zz.convert_to_shape(); z.fill.background(); z.line.color.rgb = rgb(color); z.line.width = Pt(lw); z.shadow.inherit = False


KIND = {  # fill, border, placeholder colour, letter colour
    "grey": (C["seg"], C["segb"], C["ph"], C["ink"]),
    "cut": (C["peach"], C["peachb"], C["pph"], C["ink"]),
    "blue": (C["bluf"], C["blu"], C["bph"], C["blud"]),
    "new": (C["blum"], C["blu"], C["bph"], C["blud"]),
    "ans": (C["grnf"], C["grn"], C["gph"], C["ink"]),
}


def passage(x, y, w, h, lab, kind):
    """one packed piece drawn as a text card: label top-left, placeholder lines below"""
    if kind == "dup":
        box(x, y, w, h, line=C["org"], lw=1.25, hatch=(C["org"], C["hatchb"]), name="duplicated " + lab)
        pill(lab, x + 0.05, y + 0.06, w - 0.1, 0.25, C["ink"], C["w"], size=12.5, radius=0.4)
        return
    if kind == "unused":
        box(x, y, w, h, fill=C["w"], line=C["segb"], lw=1.5, dash=True, name="unused budget")
        for i in range(2): line(x + 0.12, y + 0.34 + 0.115 * i, x + w - 0.12 - 0.2 * i, y + 0.34 + 0.115 * i, color=C["ph"], head=False, lw=1.0, dash=True)
        return
    fill, bd, phc, lc = KIND[kind]
    box(x, y, w, h, fill=fill, line=bd, lw=1.0, name="passage " + lab)
    small = len(lab) > 3
    text(lab, x + 0.08, y + 0.05, w - 0.12, 0.26, size=12.5 if small else 14, color=C["mute"] if small else lc, align=L, bold=not small)
    if kind == "ans": star(x + 0.38, y + 0.18, 0.08)
    ph_lines(x + 0.1, y + 0.36, w - 0.2, phc)


def plug(px, cy, sx, col, dark):
    box(px - 0.24, cy - 0.16, 0.24, 0.32, fill=col, kind=RRECT, radius=0.25, name="plug body")
    for dy in (-0.09, 0.05): box(px, cy + dy, sx - px, 0.045, fill=col, name="plug prong")
    box(sx, cy - 0.21, 0.22, 0.42, fill=C["w"], line=col, lw=1.75, kind=RRECT, radius=0.25, name="socket")
    for dy in (-0.09, 0.05): box(sx + 0.03, cy + dy, 0.08, 0.045, fill=dark, name="socket slot")


def tag(t, x, y, w, fill, color=None):
    return pill(t, x, y, w, 0.28, color or C["ink"], fill, size=12.5, radius=0.25)


# ------------------------------------------------------------------ grid
BX0, BW, BH = 4.15, 5.15, 0.6           # bars (1,024 tokens)
YA, YB = 1.25, 4.2                      # bar centres
FX0, FX1, FY0, FY1 = 3.97, 9.48, 0.28, 4.85   # packing-step frame
LLX, LLW = 9.68, 1.32                   # LLM column
AX, AW = 11.2, 1.32                     # outcomes

# ---------------------------------------------------------------- inputs
box(0.12, 0.28, 1.78, 0.6, fill=C["w"], line=C["ink"], kind=RRECT, radius=0.2, name="Question")
write(box(0.24, 0.43, 0.3, 0.3, fill=C["ink"], kind=OVAL), [("Q", C["w"])], 14)
text("Question", 0.62, 0.43, 1.2, 0.3, size=15, align=L)

box(0.12, 1.08, 1.78, 3.77, fill=C["w"], line=C["ink"], kind=RRECT, radius=0.06, name="Long document")
box(0.24, 1.2, 0.2, 0.26, fill=C["w"], line=C["ink"], lw=1.25, kind=MSO_SHAPE.FOLDED_CORNER, name="doc icon")
for i in range(2): line(0.28, 1.28 + 0.07 * i, 0.39, 1.28 + 0.07 * i, color=C["ink"], head=False, lw=1.0)
text("Long document", 0.5, 1.14, 1.38, 0.26, size=13, align=L)
text("~12.5K tokens", 0.5, 1.4, 1.38, 0.24, size=12.5, color=C["mute"], align=L, bold=False)
for i, lab in enumerate("ABCDEF"):
    ry = 1.75 + 0.5 * i; ans = lab == "E"
    box(0.22, ry, 1.58, 0.42, fill=C["grnf"] if ans else C["seg"], line=C["grn"] if ans else None, lw=1.25, kind=RRECT, radius=0.18, name="span " + lab)
    text(lab, 0.3, ry + 0.08, 0.22, 0.26, size=14)
    if ans: star(0.64, ry + 0.21)
    ph_lines(0.8 if ans else 0.6, ry + 0.15, 0.9 if ans else 1.1, C["gph"] if ans else C["ph"])

# ---------------------------------------------------------------- ranker and ranking
RX, RW = 2.15, 1.6
box(RX, 1.0, RW, 1.12, fill=C["lav"], line=C["lavb"], lw=1.5, kind=RRECT, radius=0.12, name="Ranker")
text("Ranker", RX, 1.06, RW, 0.3, size=15)
text("lexical · BGE ·\nreranker, …", RX, 1.38, RW, 0.42, size=12.5, color=C["mute"])
text("unchanged", RX, 1.82, RW, 0.24, size=12.5, color=C["mute"])
poly([(1.9, 0.58), (RX + RW / 2, 0.58), (RX + RW / 2, 1.0)])
line(1.9, 1.5, RX, 1.5)
line(RX + RW / 2, 2.12, RX + RW / 2, 2.4)

box(RX, 2.4, RW, 0.97, fill=C["w"], line=C["ink"], lw=1.5, kind=RRECT, radius=0.1, name="Ranked spans")
text("Ranked spans", RX, 2.45, RW, 0.28, size=13.5)
for i, lab in enumerate("ABCDEF"):
    cx, cy = RX + 0.18 + (i % 3) * 0.44, 2.8 + (i // 3) * 0.29
    ans = lab == "E"
    write(box(cx, cy, 0.26, 0.22, fill=C["grnf"] if ans else C["seg"], line=C["grn"] if ans else C["segb"], lw=1.0, kind=RRECT, radius=0.2), lab, 12.5)
    if ans: star(cx + 0.22, cy - 0.03, 0.07)
    if i < 5: text("→", cx + 0.26, cy - 0.02, 0.18, 0.24, size=12.5, color=C["mute"])

# ranking -> standard packing (fixed wire); ranking -> DedupPack (cable and plug)
poly([(RX + RW, 2.62), (3.86, 2.62), (3.86, YA), (BX0, YA)])
CX, PX = 2.45, FX0 - 0.23
line(CX, 3.37, CX, YB, color=C["blu"], head=False, lw=2.5)
line(CX, YB, PX - 0.24, YB, color=C["blu"], head=False, lw=2.5)
text("plugs in after\nany ranker", 2.2, 4.33, 1.4, 0.44, size=12.5, color=C["blud"])

# ---------------------------------------------------------------- packing step frame
box(FX0, FY0, FX1 - FX0, FY1 - FY0, line=C["frame"], lw=1.25, dash=True, kind=RRECT, radius=0.03, name="Packing step")
text("packing step · before the LLM call", FX0 + 0.15, FY0 - 0.11, 2.85, 0.22, size=12.5, color=C["mute"], fill=C["w"])

# lane 1: standard chunk packing
pill("Standard chunk packing", 4.1, 0.42, 2.8, 0.38, C["ink"], C["seg"], size=16)
pill("1,024 tokens", 8.12, 0.44, 1.18, 0.34, C["ink"], C["seg"], size=13)
line(4.1, 0.86, 9.3, 0.86, head=False, lw=1.0)
segs = [("A", 0.57, "grey"), ("overlap", 0.8, "dup"), ("overlap", 0.8, "dup"), ("B", 0.57, "grey"),
        ("C…", 0.82, "cut"), ("…D…", 0.82, "cut"), ("", 0.77, "unused")]
xa, x = [], BX0
for lab, w, k in segs:
    xa.append(x); passage(x, YA - BH / 2, w, BH, lab, k); x += w
xa.append(x)
for cx in (xa[5], xa[6]): zigzag(cx, YA - BH / 2, BH, C["org"])
TY = 1.75
tag("23% duplicated", 4.79, TY, 1.46, C["tag"])
line(5.3, TY, xa[1] + 0.4, YA + BH / 2, color=C["org"], lw=1.5); line(5.74, TY, xa[2] + 0.4, YA + BH / 2, color=C["org"], lw=1.5)
tag("cut mid-sentence", 6.92, TY, 1.61, C["tag"])
line(7.55, TY, xa[5], YA + BH / 2, color=C["org"], lw=1.5); line(8.25, TY, xa[6], YA + BH / 2, color=C["org"], lw=1.5)
tag("unused", 8.6, TY, 0.72, C["seg"])
line(8.96, TY, 8.94, YA + BH / 2, color=C["mute"], lw=1.5)

# lane 2: DedupPack
pill("DedupPack (ours)", 4.1, 2.2, 2.15, 0.38, C["blud"], C["bpill"], size=16)
text("training-free ·\nno model calls", 6.4, 2.17, 1.55, 0.44, size=12.5, color=C["blud"])
pill("1,024 tokens", 8.12, 2.22, 1.18, 0.34, C["blud"], C["bpill"], size=13)
line(4.1, 2.64, 9.3, 2.64, head=False, lw=1.0, color=C["blud"])
SG = 0.1; SW = (BW - 3 * SG) / 4; SY, SH = 2.74, 0.72
caps = ("sentence\npadding", "merge\noverlaps", "exact fill\n+ backfill", "document\norder")
xs = [BX0 + i * (SW + SG) for i in range(4)]
for x, cap in zip(xs, caps):
    box(x, SY, SW, SH, fill=C["w"], line=C["blu"], lw=1.25, kind=RRECT, radius=0.12, name="Step: " + cap)
    text(cap, x, SY + 0.3, SW, 0.4, size=12.5, color=C["blud"])
for i in range(3):
    line(xs[i] + SW, SY + 0.2, xs[i + 1], SY + 0.2, color=C["blu"], lw=1.5)
iy = SY + 0.18
O = C["org"]
x = xs[0]   # (1) a span is padded out to the sentence boundaries
line(x + 0.15, iy, x + SW - 0.15, iy, color=C["frame"], head=False, lw=1)
for t in (x + 0.15, x + SW - 0.15): line(t, iy - 0.1, t, iy + 0.1, head=False, lw=2.25)
box(x + SW / 2 - 0.17, iy - 0.05, 0.34, 0.1, fill=C["blu"])
line(x + SW / 2 - 0.19, iy, x + 0.21, iy, color=C["blu"], dash=True, lw=1.5); line(x + SW / 2 + 0.19, iy, x + SW - 0.21, iy, color=C["blu"], dash=True, lw=1.5)
x = xs[1]   # (2) two overlapping ranges become one
box(x + 0.12, iy - 0.13, 0.45, 0.1, fill=C["seg"], line=C["mute"], lw=1)
box(x + 0.35, iy + 0.03, 0.42, 0.1, fill=C["seg"], line=C["mute"], lw=1)
box(x + 0.35, iy - 0.03, 0.22, 0.06, fill=C["peach"])
line(x + 0.82, iy, x + 0.94, iy, color=C["blu"], lw=1.5)
box(x + 0.97, iy - 0.05, 0.13, 0.1, fill=C["blu"])
x = xs[2]   # (3) exact fill: a piece that does not fit is skipped, a later one backfills
box(x + 0.12, iy, 0.62, 0.12, fill=C["seg"])
box(x + 0.74, iy, 0.3, 0.12, fill=C["blu"])
box(x + 0.12, iy, 0.92, 0.12, line=C["ink"], lw=1)
line(x + 1.04, iy - 0.05, x + 1.04, iy + 0.17, head=False, lw=2.25)
box(x + 0.42, iy - 0.15, 0.5, 0.09, fill=C["w"], line=C["mute"], lw=1)
line(x + 0.59, iy - 0.18, x + 0.75, iy - 0.03, color=O, head=False, lw=2)
line(x + 0.59, iy - 0.03, x + 0.75, iy - 0.18, color=O, head=False, lw=2)
x = xs[3]   # (4) rank order -> document order
for i, l in enumerate("312"):
    write(box(x + 0.08 + 0.16 * i, iy - 0.08, 0.15, 0.16, fill=C["seg"], line=C["mute"], lw=0.75), l, 8)
line(x + 0.57, iy, x + 0.65, iy, color=C["blu"], lw=1.5)
for i, l in enumerate("123"):
    write(box(x + 0.68 + 0.16 * i, iy - 0.08, 0.15, 0.16, fill=C["blum"], line=C["blu"], lw=0.75), l, 8)

segs = [("A", 0.57, "blue"), ("overlap", 0.8, "blue"), ("B", 0.57, "blue"), ("C", 0.78, "blue"), ("D", 0.78, "blue"),
        ("E", 0.85, "ans"), ("F", 0.8, "new")]
xb, x = [], BX0
for lab, w, k in segs:
    xb.append(x); passage(x, YB - BH / 2, w, BH, lab, k); x += w
xb.append(x)
box(BX0, YB - BH / 2, xb[3] - BX0, BH, line=C["blu"], lw=2.5, name="merged passage A+B")
text("recovered budget → new evidence", 6.3, 3.5, 3.0, 0.26, size=13, color=C["blud"], align=PP_ALIGN.RIGHT)
line(xb[5], 3.82, xb[7], 3.82, color=C["blu"], head=False, lw=1.5)
for xx in (xb[5], xb[7]): line(xx, 3.82, xx, 3.88, color=C["blu"], head=False, lw=1.5)
text("whole sentences, no duplicates", BX0, 4.54, BW, 0.24, size=12.5, color=C["blud"])
plug(PX, YB, FX0 - 0.12, C["blu"], C["blud"])

# ---------------------------------------------------------------- LLM
box(LLX, 0.42, LLW, 4.43, fill=C["w"], line=C["ink"], kind=RRECT, radius=0.1, name="LLM")
mx, my = LLX + LLW / 2, 0.8
layers = [(-0.22, (-0.12, 0.12)), (0.0, (-0.2, 0.0, 0.2)), (0.22, (-0.12, 0.12))]
nodes = [[(mx + dx, my + dy) for dy in ys] for dx, ys in layers]
for a, b in zip(nodes, nodes[1:]):
    for p in a:
        for q in b: line(p[0], p[1], q[0], q[1], color=C["blud"], head=False, lw=1.0)
for layer in nodes:
    for (nx, ny) in layer: box(nx - 0.05, ny - 0.05, 0.1, 0.1, fill=C["bluf"], line=C["blud"], lw=1.25, kind=OVAL)
text("LLM", LLX, 1.08, LLW, 0.34, size=20)
lg = 0.3
for i, (f, nm) in enumerate((("qwen_t.png", "Qwen2.5"), ("meta_t.png", "Llama-3.1"))):
    ly = 1.5 + i * 0.62
    sh.add_picture(str(HERE / "logos" / f), Inches(LLX + (LLW - lg) / 2), Inches(ly), Inches(lg), Inches(lg))
    text(nm, LLX, ly + lg, LLW, 0.24, size=12.5, color=C["mute"])
text("input tokens", LLX, 2.85, LLW, 0.24, size=12.5, color=C["mute"])
box(LLX + 0.1, 3.14, 0.66, 0.18, fill=C["seg"], line=C["segb"], lw=1.0, name="document length")
text("12.5K", LLX + 0.8, 3.11, 0.5, 0.24, size=12.5, align=L)
box(LLX + 0.1, 3.4, 0.055, 0.18, fill=C["blu"], name="packed length")
text("1,024", LLX + 0.22, 3.37, 0.6, 0.24, size=12.5, color=C["blud"], align=L)
text("12× less input", LLX, 3.66, LLW, 0.26, size=12.5, color=C["blud"])
text("unchanged", LLX, 4.5, LLW, 0.24, size=12.5, color=C["mute"])
line(BX0 + BW, YA, LLX, YA); line(BX0 + BW, YB, LLX, YB)

# ---------------------------------------------------------------- outcomes
box(AX, YA - 0.32, AW, 0.64, fill=C["card"], line=C["ink"], lw=1.5, kind=RRECT, radius=0.18, name="outcome standard")
star(AX + 0.2, YA - 0.14, 0.1, hollow=True)
text("E missing", AX + 0.36, YA - 0.28, AW - 0.4, 0.28, size=13.5, align=L)
text([("Answer ", C["ink"]), ("✗", C["org"])], AX, YA + 0.02, AW, 0.26, size=12.5)
box(AX, YB - 0.32, AW, 0.64, fill=C["bluf"], line=C["blu"], lw=2.0, kind=RRECT, radius=0.18, name="outcome DedupPack")
star(AX + 0.2, YB - 0.14, 0.1)
text("E included", AX + 0.36, YB - 0.28, AW - 0.4, 0.28, size=13.5, align=L)
text([("Answer ", C["ink"]), ("✓", C["blu"])], AX, YB + 0.02, AW, 0.26, size=12.5)
line(LLX + LLW, YA, AX, YA); line(LLX + LLW, YB, AX, YB)

# ---------------------------------------------------------------- legend
box(0.12, 4.98, 12.38, 0.44, fill=C["w"], line=C["frame"], lw=1.0, dash=True, kind=RRECT, radius=0.3, name="Legend")
lx = 0.75
text("Legend:", lx, 5.07, 0.7, 0.26, size=13, align=L); lx += 0.78
items = [("grey", "candidate span", 1.16), ("dup", "duplicated text (overlap)", 1.86), ("cut", "cut mid-sentence", 1.33),
         ("unused", "unused budget", 1.16), ("ans", "answer evidence", 1.26)]
for k, t, tw in items:
    sx, sy, sw, shh = lx, 5.06, 0.42, 0.26
    if k == "dup": box(sx, sy, sw, shh, line=C["org"], lw=1.0, hatch=(C["org"], C["hatchb"]))
    elif k == "unused": box(sx, sy, sw, shh, fill=C["w"], line=C["segb"], lw=1.25, dash=True)
    else:
        fill, bd, phc, _ = KIND[k]
        box(sx, sy, sw, shh, fill=fill, line=bd, lw=1.0)
        ph_lines(sx + 0.07, sy + 0.08, sw - 0.14 - (0.08 if k == "cut" else 0), phc)
        if k == "cut": zigzag(sx + sw - 0.04, sy, shh, C["org"], 1.75)
        if k == "ans": star(sx + sw - 0.03, sy + 0.02, 0.07)
    text(t, lx + 0.5, 5.06, tw + 0.05, 0.26, size=12.5, align=L, bold=False)
    lx += 0.5 + tw + 0.3

prs.save(sys.argv[1])
print("saved", sys.argv[1])
