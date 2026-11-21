"""Figure 2, v6: a one-to-one PowerPoint redraw of the user's reference picture (images/3.webp, 1886 x 834 px).
All coordinates are the reference's pixel coordinates; P() converts them to inches on a 12.6 in wide slide.
Font sizes and weights were fitted to the reference's ink boxes (Segoe UI / Semibold / Bold).
usage: python fig_overview_pptx_v6.py OUT.pptx"""
import sys
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.dml import MSO_LINE_DASH_STYLE, MSO_PATTERN
from pptx.oxml.ns import qn
from lxml import etree

HERE = Path(__file__).parent
PPI = 1886 / 12.6
P = lambda v: Inches(v / PPI)
W, H = 1886, 834
prs = Presentation()
prs.slide_width, prs.slide_height = P(W), P(H)
s = prs.slides.add_slide(prs.slide_layouts[6])
sh = s.shapes
RECT, RRECT, OVAL = MSO_SHAPE.RECTANGLE, MSO_SHAPE.ROUNDED_RECTANGLE, MSO_SHAPE.OVAL
L, CTR = PP_ALIGN.LEFT, PP_ALIGN.CENTER
FONT = {"R": "Segoe UI", "SB": "Segoe UI Semibold", "B": "Segoe UI"}
C = dict(ink="0E1A33", line="1A1F2B", banner="CEE5FB", head="0B1838",
         panel="F0F4F7", panelb="6B7787", llmb="44505F",
         page="2A2F3A", pline="C5CAD2", chip="D9E1EA", chipb="A0A9B6",
         gfill="CEF3D2", gbd="8ECB9C", star="15A04A", gtext="0C903F",
         seg="DAE2E9", segb="949FAB",
         olane="FEF5EC", olaneb="EC8A43", hb="FEEDDA", hs="F8BE88", hbd="EDB68A",
         dfill="F9CFA9", dbd="E8A877", zig="E66916", otext="E06A1F", obr="D98048", unused="8A949E",
         blane1="E7F6FE", blane2="F4FBFD", blaneb="2B86C8", card="F7FBFE", cardb="3E97DA",
         ibar="CEE9F9", ibarb="3A8CC7", odash="E29A5E", yellow="F5EE1A", barrow="1E8BF0",
         packed="D1DFEC", free="F1F7FC", pfb="8D969C", exact="C9E0F1", backfill="319FE1", xorg="F3620D",
         c312="E0E7F3", c123="DCEBF8", c123b="2F7FC0", cdark="3A4250",
         rtext="1D5A96", rbr="3A6EA5", efill="D0F5D6", ffill="AFDBF3", dsegb="87939C",
         icon="3A98F0", mute="5F6B7A", ppill="EDF0F6", ppillb="7B8794", ptext="3A4456", frame="9BA3AE",
         sep="B9BFC6", absf="F6FAFD", absb="3F495C", hstar="62676F", presf="EEF8FE", presb="3583BD", pstar="02983C",
         w="FFFFFF")
rgb = lambda h: RGBColor.from_string(h)


def box(x0, y0, x1, y1, fill=None, line=None, lw=1.5, dash=False, kind=RECT, r=None, name=None, hatch=None, grad=None):
    p = sh.add_shape(kind, P(x0), P(y0), P(x1 - x0), P(y1 - y0))
    p.shadow.inherit = False
    if hatch:
        p.fill.patterned(); p.fill.pattern = MSO_PATTERN.WIDE_UPWARD_DIAGONAL
        p.fill.fore_color.rgb = rgb(hatch[0]); p.fill.back_color.rgb = rgb(hatch[1])
    elif grad:
        p.fill.gradient(); p.fill.gradient_angle = 90
        st = p.fill.gradient_stops; st[0].color.rgb = rgb(grad[0]); st[1].color.rgb = rgb(grad[1])
    elif fill: p.fill.solid(); p.fill.fore_color.rgb = rgb(fill)
    else: p.fill.background()
    if line:
        p.line.color.rgb = rgb(line); p.line.width = Pt(lw)
        if dash: p.line.dash_style = MSO_LINE_DASH_STYLE.SQUARE_DOT if dash == "sys" else MSO_LINE_DASH_STYLE.DASH
    else: p.line.fill.background()
    if kind == RRECT and r is not None:   # r = corner radius in px
        p.adjustments[0] = min(0.5, r / min(x1 - x0, y1 - y0))
    if name: p.name = name
    return p


def write(p, runs, size, weight="R", align=CTR):
    tf = p.text_frame; tf.word_wrap = False; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    if isinstance(runs, str): runs = [(runs, C["ink"])]
    para = tf.paragraphs[0]; para.alignment = align
    for t, c in runs:
        r = para.add_run(); r.text = t
        r.font.name = FONT[weight]; r.font.size = Pt(size); r.font.bold = weight == "B"; r.font.color.rgb = rgb(c)
    return p


def text(t, x0, y0, x1, y1, size, weight="R", color=None, align=CTR):
    return write(box(x0, y0, x1, y1), [(t, color or C["ink"])] if isinstance(t, str) else t, size, weight, align)


def line(x1, y1, x2, y2, color=None, lw=1.5, head=False, dash=False):
    c = sh.add_connector(MSO_CONNECTOR.STRAIGHT, P(x1), P(y1), P(x2), P(y2))
    c.line.color.rgb = rgb(color or C["line"]); c.line.width = Pt(lw)
    if dash: c.line.dash_style = MSO_LINE_DASH_STYLE.DASH
    if head:
        e = etree.SubElement(c.line._get_or_add_ln(), qn("a:tailEnd")); e.set("type", "triangle"); e.set("w", "med"); e.set("len", "med")
    return c


def arrow(pts, color=None, lw=1.5):
    for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
        line(x1, y1, x2, y2, color=color, lw=lw, head=(x2, y2) == pts[-1])


def polyline(pts, color, lw=1.25, fill=None):
    fb = sh.build_freeform(P(pts[0][0]), P(pts[0][1]))
    fb.add_line_segments([(P(x), P(y)) for x, y in pts[1:]], close=fill is not None)
    z = fb.convert_to_shape(); z.shadow.inherit = False
    if fill: z.fill.solid(); z.fill.fore_color.rgb = rgb(fill)
    else: z.fill.background()
    if color: z.line.color.rgb = rgb(color); z.line.width = Pt(lw)
    else: z.line.fill.background()
    return z


def star(cx, cy, r, fill=None, outline=None):
    box(cx - r, cy - r, cx + r, cy + r, fill=fill, line=outline, lw=1.0, kind=MSO_SHAPE.STAR_5_POINT, name="star")


def brace(x0, x1, y, mid, color, up=True):
    """flat curly brace: ends bend down, a notch points up at mid"""
    d = 6
    polyline([(x0, y + d), (x0 + 4, y), (mid - 5, y), (mid, y - d), (mid + 5, y), (x1 - 4, y), (x1, y + d)], color, 1.25)


def chip(lab, x0, y0, x1, y1, fill, bd, size=14.5, weight="B", r=6):
    write(box(x0, y0, x1, y1, fill=fill, line=bd, lw=1.0, kind=RRECT, r=r, name="chip " + lab), lab, size, weight)


def segment(lab, x0, x1, y0, y1, fill, bd, size=15, lw=1.0):
    write(box(x0, y0, x1, y1, fill=fill, line=bd, lw=lw, name="segment " + lab), lab, size, "B")


def tick_line(x0, x1, y, label, lx0, lx1, size):
    """budget line |------ label ------|"""
    line(x0, y, lx0, y, color=C["line"], lw=1.0); line(lx1, y, x1, y, color=C["line"], lw=1.0)
    for xx in (x0, x1): line(xx, y - 9, xx, y + 9, color=C["line"], lw=1.0)
    text(label, lx0, y - 15, lx1, y + 15, size, "B")


# ---------------------------------------------------------------- header banner
box(12, 8, 1876, 78, fill=C["banner"], kind=RRECT, r=18, name="Header")
text("Before the LLM call: context construction", 12, 10, 1816, 76, 18.8, "B", C["head"])

# ---------------------------------------------------------------- question
box(22, 125, 258, 194, fill=C["w"], line=C["line"], lw=1.5, kind=RRECT, r=35, name="Question")
b = box(50, 141, 95, 175, fill=C["w"], line=C["line"], lw=1.25, kind=MSO_SHAPE.OVAL_CALLOUT, name="speech bubble")
b.adjustments[0] = -0.28; b.adjustments[1] = 0.68
for dx in (62, 72, 82): box(dx - 2.6, 155.4, dx + 2.6, 160.6, fill=C["line"], kind=OVAL)
text("Question", 108, 140, 240, 182, 13.6, "B", align=L)
arrow([(258, 160), (328, 160), (328, 320)])

# ---------------------------------------------------------------- long document
for x0, y0 in ((46, 231), (32, 245), (18, 259)):
    box(x0, y0, x0 + 205, y0 + 363, fill=C["w"], line=C["page"], lw=1.5, name="page")
for k in range(10):
    yy = 284 + 34.66 * k
    line(35, yy, 207, yy, color=C["pline"], lw=1.25)
for lab, x0, y0, x1, y1 in (("A", 37, 298, 92, 335), ("B", 103, 298, 157, 335), ("C", 125, 371, 180, 409),
                            ("D", 52, 437, 108, 474), ("F", 76, 544, 131, 582)):
    chip(lab, x0, y0, x1, y1, C["chip"], C["chipb"])
chip("E", 140, 477, 197, 516, C["gfill"], C["gbd"])
star(199, 480, 12, fill=C["star"])
text("Long document", 20, 636, 258, 670, 14.0, "SB")
text("~12.5K tokens", 20, 671, 258, 701, 14.2, "SB")
star(40, 735, 13, fill=C["star"])
text("answer evidence", 61, 720, 275, 750, 12.8, "SB", C["gtext"], align=L)
arrow([(249, 428), (279, 428)])

# ---------------------------------------------------------------- ranker
box(279, 322, 412, 546, fill=C["panel"], line=C["panelb"], lw=1.5, kind=RRECT, r=14, name="Ranker")
for x0, y0, x1 in ((322, 378, 334), (338, 365, 350), (354, 352, 368)):
    box(x0, y0, x1, 394, fill=C["icon"], name="bar icon")
text("Ranker", 282, 408, 410, 442, 14.4, "B")
text("relevance", 282, 455, 410, 481, 10.7, "SB", color=C["mute"])
text("ranking", 282, 481, 410, 511, 10.7, "SB", color=C["mute"])
arrow([(412, 428), (442, 428)])

# ---------------------------------------------------------------- ranked evidence
box(442, 243, 596, 630, fill=C["panel"], line=C["panelb"], lw=1.5, kind=RRECT, r=14, name="Ranked evidence")
text("Ranked", 445, 254, 593, 286, 13.1, "SB")
text("Evidence", 445, 287, 593, 318, 13.1, "SB")
for i, lab in enumerate("ABCDEF"):
    y0 = 328 + 49 * i
    text(str(i + 1), 470, y0, 498, y0 + 40, 13.5, "SB")
    ans = lab == "E"
    chip(lab, 510, y0, 566, y0 + 40, C["gfill"] if ans else C["chip"], C["gbd"] if ans else C["chipb"])
star(567, 527, 11, fill=C["star"])
line(596, 430, 630, 430, lw=1.5); line(630, 270, 630, 537, lw=1.5)
arrow([(630, 270), (672, 270)]); arrow([(630, 537), (672, 537)])

# ---------------------------------------------------------------- packing step frame
box(642, 115, 1517, 797, line=C["frame"], lw=1.25, dash=True, kind=RRECT, r=25, name="Packing step")
box(675, 98, 855, 137, fill=C["ppill"], line=C["ppillb"], lw=1.25, kind=RRECT, r=19, name="packing step label")
text("packing step", 675, 99, 855, 136, 11.9, "SB", color=C["ptext"])

# ---------------------------------------------------------------- standard chunk packing
box(674, 143, 1488, 356, fill=C["olane"], line=C["olaneb"], lw=1.75, kind=RRECT, r=16, name="Standard lane")
text("Standard chunk packing", 696, 156, 1045, 197, 13.9, "B", align=L)
Y0, Y1 = 249, 303
segment("A", 704, 810, Y0, Y1, C["seg"], C["segb"])
box(810, Y0, 947, Y1, line=C["hbd"], lw=1.0, hatch=(C["hs"], C["hb"]), name="duplicate")
segment("B", 947, 1060, Y0, Y1, C["seg"], C["segb"])
segment("C", 1060, 1180, Y0, Y1, C["seg"], C["segb"])
segment("D", 1180, 1330, Y0, Y1, C["dfill"], C["dbd"])
zz, yy = [], Y0
for k in range(8):
    zz.append((1330 + (7 if k % 2 == 0 else 0), yy)); yy += (Y1 - Y0) / 7
polyline([(1330, Y0)] + [(x, min(y, Y1)) for x, y in zz[1:]] + [(1330, Y1)], C["zig"], 1.75)
box(1336, Y0, 1460, Y1, fill=C["w"], line=C["unused"], lw=1.25, dash=True, name="unused")
text("Duplicate", 820, 205, 944, 235, 10.0, "B", C["otext"])
brace(818, 940, 241, 879, C["obr"])
text("Truncated", 1190, 196, 1314, 224, 10.4, "B", C["otext"])
brace(1195, 1315, 241, 1255, C["obr"])
text("Unused", 1352, 196, 1450, 224, 10.2, "B", C["otext"])
tick_line(704, 1460, 327, "1,024 tokens", 965, 1167, 12.6)

# ---------------------------------------------------------------- DedupPack
box(673, 385, 1497, 779, line=C["blaneb"], lw=2.0, kind=RRECT, r=16, grad=(C["blane1"], C["blane2"]), name="DedupPack lane")
text("DedupPack (ours)", 691, 397, 985, 443, 15.4, "B", C["head"], align=L)
for x0, y0, x1, y1 in ((690, 449, 853, 612), (878, 449, 1036, 612), (1058, 446, 1328, 608), (1349, 449, 1486, 606)):
    box(x0, y0, x1, y1, fill=C["card"], line=C["cardb"], lw=1.5, kind=RRECT, r=12, name="step card")
for x0, x1 in ((853, 876), (1036, 1056), (1328, 1347)):
    arrow([(x0, 530), (x1, 530)], color=C["barrow"], lw=1.75)
# (1) sentence padding
box(712, 474, 790, 489, fill=C["ibar"], line=C["ibarb"], lw=1.0, kind=RRECT, r=4)
box(790, 474, 828, 489, line=C["odash"], lw=1.25, dash="sys", kind=RRECT, r=3)
arrow([(772, 496), (772, 517)], lw=1.25)
box(706, 524, 838, 541, fill=C["ibar"], line=C["ibarb"], lw=1.0, kind=RRECT, r=4)
line(752, 532.5, 792, 532.5, lw=1.0)
text("sentence", 692, 548, 851, 576, 11.3, "SB")
text("padding", 692, 575, 851, 603, 11.3, "SB")
# (2) merge overlaps
box(900, 470, 985, 487, fill=C["ibar"], line=C["ibarb"], lw=1.0, kind=RRECT, r=4)
box(930, 480, 1015, 497, fill=C["ibar"], line=C["ibarb"], lw=1.0, kind=RRECT, r=4)
box(925, 474, 985, 490, fill=C["yellow"], name="overlap highlight")
arrow([(957, 500), (957, 519)], lw=1.25)
box(893, 524, 1022, 541, fill=C["ibar"], line=C["ibarb"], lw=1.0, kind=RRECT, r=4)
line(930, 532.5, 985, 532.5, lw=1.0)
text("merge", 880, 548, 1034, 576, 11.3, "SB")
text("overlaps", 880, 575, 1034, 603, 11.3, "SB")
# (3) exact fill + backfill
text("Candidate too large", 1071, 459, 1195, 480, 6.0, "SB", align=L)
box(1197, 458, 1273, 477, line=C["odash"], lw=1.25, dash="sys", kind=RRECT, r=3)
line(1283, 461, 1297, 475, color=C["xorg"], lw=1.75); line(1297, 461, 1283, 475, color=C["xorg"], lw=1.75)
write(box(1073, 489, 1228, 521, fill=C["packed"], line=C["pfb"], lw=1.0), "Packed", 6.7, "SB")
write(box(1228, 489, 1317, 521, fill=C["free"], line=C["pfb"], lw=1.0), "Free", 6.7, "SB")
arrow([(1172, 527), (1172, 547)], color=C["barrow"], lw=1.5)
write(box(1073, 550, 1210, 588, fill=C["exact"], line=C["pfb"], lw=1.0), "Exact fill", 7.0, "SB")
write(box(1210, 550, 1317, 588, fill=C["backfill"], line=C["backfill"], lw=1.0), [("Backfill", C["w"])], 9.3, "SB")
# (4) document order
for i, d in enumerate("312"):
    write(box(1365 + 25 * i, 468, 1388 + 25 * i, 492, fill=C["c312"], line=C["cdark"], lw=0.75, kind=RRECT, r=3), d, 10.5, "SB")
text("→", 1438, 466, 1472, 494, 12, "SB")
for i, d in enumerate("123"):
    write(box(1394 + 25 * i, 500, 1417 + 25 * i, 524, fill=C["c123"], line=C["c123b"], lw=0.75, kind=RRECT, r=3), d, 10.5, "SB")
text("document", 1351, 536, 1484, 564, 11.2, "SB")
text("order", 1351, 564, 1484, 592, 11.2, "SB")
# recovered space and the packed context
text("Recovered space → new evidence", 1100, 623, 1448, 655, 9.3, "B", C["rtext"])
brace(1165, 1430, 661, 1297, C["rbr"])
Z0, Z1 = 682, 731
for lab, x0, x1 in (("A", 729, 835), ("B", 835, 939), ("C", 939, 1047), ("D", 1047, 1167)):
    segment(lab, x0, x1, Z0, Z1, C["seg"], C["dsegb"])
segment("E", 1167, 1304, Z0, Z1, C["efill"], C["dsegb"])
segment("F", 1304, 1438, Z0, Z1, C["ffill"], C["dsegb"])
star(1262, 690, 12, fill=C["star"])
tick_line(730, 1437, 752, "1,024-token budget", 930, 1180, 11.1)

# ---------------------------------------------------------------- to the LLM
line(1488, 270, 1533, 270, lw=1.5); line(1497, 532, 1533, 532, lw=1.5); line(1533, 270, 1533, 532, lw=1.5)
arrow([(1533, 435), (1562, 435)])
box(1562, 241, 1708, 601, fill=C["panel"], line=C["llmb"], lw=1.75, kind=RRECT, r=14, name="LLM")
text("LLM", 1565, 274, 1705, 318, 18.0, "B")
sh.add_picture(str(HERE / "logos" / "qwen_t.png"), P(1612), P(349), P(51), P(52))
text("Qwen2.5", 1565, 402, 1705, 429, 10.2, "SB")
line(1585, 447, 1685, 447, color=C["sep"], lw=1.0)
sh.add_picture(str(HERE / "logos" / "meta_t.png"), P(1609), P(476), P(58), P(39))
text("Llama-3.1", 1565, 522, 1705, 550, 10.2, "SB")

# ---------------------------------------------------------------- outcomes
arrow([(1708, 349), (1730, 349)]); arrow([(1708, 484), (1730, 484)])
box(1731, 316, 1880, 381, fill=C["absf"], line=C["absb"], lw=1.5, kind=RRECT, r=32, name="E absent")
star(1755, 348, 11, outline=C["hstar"])
text("E absent", 1776, 331, 1879, 366, 10.7, "B", align=L)
box(1731, 449, 1881, 518, fill=C["presf"], line=C["presb"], lw=1.75, kind=RRECT, r=34, name="E present")
star(1756, 484, 11, fill=C["pstar"])
text("E present", 1776, 467, 1880, 501, 10.0, "B", align=L)

prs.save(sys.argv[1])
print("saved", sys.argv[1])
