"""Worked example (real LongBench instance, 2WikiMultihopQA b3845fca...): same BGE ranking, same 1,024-token budget.
Text excerpts are taken verbatim from the two packed contexts (run7_q32 bge_chunk / run8_q32 lex_rep seed_mode=1);
"[...]" marks omitted text. Answers are the actual predictions of Qwen2.5-14B and Llama-3.1-8B.
usage: python fig_example_pptx.py OUT.pptx"""
import sys
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

F = "Segoe UI"
C = dict(ink="1F2328", mute="6B7280", line="7D848D", panel="F4F5F7",
         org="C24E00", orgl="FDF1E8", orgm="F8D9C2", blu="0072B2", blul="EAF3FA", blum="CFE4F4", blud="00507D", w="FFFFFF")
rgb = lambda h: RGBColor.from_string(h)
W, H = 11.0, 6.68
prs = Presentation(); prs.slide_width, prs.slide_height = Inches(W), Inches(H)
s = prs.slides.add_slide(prs.slide_layouts[6]); sh = s.shapes


def box(x, y, w, h, fill=None, line=None, lw=1.5, radius=0.06, kind=MSO_SHAPE.ROUNDED_RECTANGLE):
    p = sh.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h)); p.shadow.inherit = False
    if fill: p.fill.solid(); p.fill.fore_color.rgb = rgb(fill)
    else: p.fill.background()
    if line: p.line.color.rgb = rgb(line); p.line.width = Pt(lw)
    else: p.line.fill.background()
    if kind == MSO_SHAPE.ROUNDED_RECTANGLE: p.adjustments[0] = radius
    return p


def rich(p, runs, size=11, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.MIDDLE, margin=0.07):
    """runs: list of (text, style) with style in {n, dup, cut, key, new, mute, b}"""
    tf = p.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(margin); tf.margin_top = tf.margin_bottom = Inches(0.03)
    para = tf.paragraphs[0]; para.alignment = align; para.line_spacing = 1.0
    for t, st in runs:
        if t == "\n":
            para = tf.add_paragraph(); para.alignment = align; para.line_spacing = 1.0; continue
        r = para.add_run(); r.text = t; f = r.font; f.name = F; f.size = Pt(size)
        f.bold = st in ("dup", "key", "b", "new"); f.italic = st == "cut"
        f.color.rgb = rgb({"n": C["ink"], "dup": C["org"], "cut": C["mute"], "key": C["blud"], "new": C["blu"],
                           "mute": C["mute"], "b": C["ink"]}[st])
    return p


def text(runs, x, y, w, h, size=11, align=PP_ALIGN.LEFT, **kw):
    return rich(box(x, y, w, h, **kw), runs if isinstance(runs, list) else [(runs, "b")], size, align)


def card(runs, x, y, w, h, fill, line, tag=None, tagcol=None):
    c = rich(box(x, y, w, h, fill=fill, line=line, lw=1.25), runs, 10.5, anchor=MSO_ANCHOR.TOP)
    if tag:
        c.text_frame.margin_right = Inches(0.62)
        t = box(x + w - 0.56, y + 0.08, 0.48, 0.22, fill=tagcol, radius=0.5)
        rich(t, [(tag, "b")], 10, PP_ALIGN.CENTER, margin=0)
        t.text_frame.paragraphs[0].runs[0].font.color.rgb = rgb(C["w"])


# question
q = box(0.2, 0.12, W - 0.4, 0.5, fill=C["panel"], line=C["line"], lw=1.25, radius=0.3)
rich(q, [("Question: ", "b"), ("Where was the director of film Kanneshwara Rama born?", "n"),
         ("      gold answer: ", "mute"), ("Mysore", "b")], 13.5, PP_ALIGN.CENTER)
text([("Same BGE ranking, same exact budget of 1,024 tokens (real LongBench instance, 2WikiMultihopQA).", "mute")],
     0.2, 0.66, W - 0.4, 0.28, size=11, align=PP_ALIGN.CENTER)

PW, PX1, PX2, PY, PH = 5.2, 0.2, 5.6, 1.0, 4.82
# ---------------- standard chunk packing
box(PX1, PY, PW, PH, fill=C["orgl"], line=C["org"], lw=2)
text("Standard chunk packing", PX1 + 0.12, PY + 0.06, 3.5, 0.32, size=14.5)
cx, cw, y = PX1 + 0.12, PW - 0.24, PY + 0.46
CH = 0.58
L = [  # (runs, tag): excerpts of the six packed chunks
    ([("…N. Kulkarni. ", "cut"), ("[…] ", "mute"), ("Kanneshwara Rama premiered at the International Film Festival of India. ", "dup"), ("[…]", "mute")], "2×"),
    ([("…Anant Nag, Shabana Azmi, […] ", "cut"), ("The film is based on the novel Kannayya Rama written by S. K. Nadig. ", "dup"), ("[…]", "mute")], "5×"),
    ([("…premiered at the International Film Festival of India. The film was theatrically released on 30 March 1989 ", "dup"), ("[…]", "mute")], "2×"),
    ([("…theatres. ", "cut"), ("It was screened in many national and international film festivals ", "dup"), ("[…]", "mute")], "3×"),
    ([("[…] political film directed by ", "n"), ("M. S. Sathyu", "key"), (". ", "n"), ("The film is based on the novel Kannayya Rama ", "dup"), ("[…]", "mute")], None),
    ([("…. Nadig. ", "cut"), ("The film is set in the 1920s during which a rebellious youth, Kanneshwara Rama ", "dup"), ("[…]", "mute")], "3×"),
]
for runs, tag in L:
    card(runs, cx, y, cw, CH, C["w"], C["line"], tag, C["org"]); y += CH + 0.06
text([("Budget full (993 tokens). The same sentences fill it up to five times; the director's biography never fits.", "dup")],
     cx, y + 0.0, cw, 0.42, size=11)

# ---------------- DedupPack
box(PX2, PY, PW, PH, fill=C["blul"], line=C["blu"], lw=2.25)
text("DedupPack", PX2 + 0.12, PY + 0.06, 1.3, 0.32, size=14.5)
text([("each range once, whole sentences, document order", "mute")], PX2 + 1.4, PY + 0.1, PW - 1.5, 0.28, size=10.5)
cx, y = PX2 + 0.12, PY + 0.46
card([("M. S. Sathyu: ", "new"), ("Mysore Shrinivas Sathyu (born 6 July 1930) is a film director, stage designer and art director from India. "
      "[…] Born into a Kannada Brahmin family, Sathyu grew up in ", "n"), ("Mysore", "key"), (".", "n")],
     cx, y, cw, 0.8, C["blum"], C["blu"], "new", C["blu"]); y += 0.86
card([("[…] Films: […] Chithegu Chinthe 1978 – Screened at 7th IFFI. ", "n"), ("Kanneshwara Rama", "key"),
      (" (The Legendary Outlaw) […]", "n")], cx, y, cw, 0.58, C["blum"], C["blu"], "new", C["blu"]); y += 0.64
card([("Kanneshwara Rama ", "b"), ("[…] is a 1977 Kannada-language political film directed by ", "n"), ("M. S. Sathyu", "key"),
      (". The film is based on the novel Kannayya Rama written by S. K. Nadig. […] Kanneshwara Rama premiered at the "
       "International Film Festival of India. The film was theatrically released on 30 March 1989 […] It was screened in many "
       "national and international film festivals […] The film starts with Kanneshwara Rama, a long-sought-after fugitive […]", "n")],
     cx, y, cw, 1.62, C["w"], C["line"]); y += 1.7
text([("1,017 tokens. The film passage appears once; the recovered budget brings in the director's biography (second hop).", "key")],
     cx, y, cw, 0.42, size=11)

# ---------------- answers
AY = PY + PH + 0.1
for x, runs, col in [
    (PX1, [("Qwen2.5-14B: ", "b"), ("“The given passages do not contain information about the birthplace…”", "n"), ("  ✗", "dup"), ("\n", ""),
           ("Llama-3.1-8B: ", "b"), ("“Not mentioned in the passages.”", "n"), ("  ✗", "dup")], C["org"]),
    (PX2, [("Qwen2.5-14B: ", "b"), ("“Mysore”", "key"), ("  ✓", "key"), ("\n", ""),
           ("Llama-3.1-8B: ", "b"), ("“Mysore”", "key"), ("  ✓", "key")], C["blu"])]:
    rich(box(x, AY, PW, 0.62, fill=C["w"], line=col, lw=2), runs, 11.5)
prs.save(sys.argv[1]); print("saved", sys.argv[1])
