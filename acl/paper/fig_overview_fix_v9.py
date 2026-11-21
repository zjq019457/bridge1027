"""Figure 2 v9 (from the user's v8): make backfill concrete with the A-F example.
- ranking becomes A, B, C, D, E, G, F: a long span G is ranked 6th and F moves to 7th;
- the document shows G as a long span;
- card 3: "Candidate G / too large" is skipped, the packed part is "Packed A-E", and the slot is filled by "Backfill F";
- the DedupPack bar marks F as "backfill".
Shapes are found by name / text / position, so the user's manual edits stay as they are."""
import copy
from pptx import Presentation
from pptx.util import Emu, Inches
from pptx.dml.color import RGBColor

SRC, DST = "fig_overview_v8.pptx", "fig_overview_v9.pptx"
PPI = 1886 / 12.6
P = lambda v: Emu(int(Inches(v / PPI)))
px = lambda e: Emu(e).inches * PPI
prs = Presentation(SRC)
sl = prs.slides[0]


def shapes():
    return list(sl.shapes)


def box_of(s):
    return px(s.left), px(s.top), px(s.left + s.width), px(s.top + s.height)


def txt(s):
    return s.text_frame.text if s.has_text_frame else ""


def one(pred, what):
    hits = [s for s in shapes() if pred(s)]
    assert len(hits) == 1, f"{what}: {len(hits)} hits"
    return hits[0]


def clone(s):
    el = copy.deepcopy(s._element); s._element.addnext(el)
    return [t for t in shapes() if t._element is el][0]


def set_text(s, t):
    runs = [r for p in s.text_frame.paragraphs for r in p.runs]
    runs[0].text = t
    for r in runs[1:]: r._r.getparent().remove(r._r)


def place(s, x0, y0, x1, y1):
    s.left, s.top, s.width, s.height = P(x0), P(y0), P(x1 - x0), P(y1 - y0)


in_rank = lambda s: 500 <= box_of(s)[0] <= 520 and 320 <= box_of(s)[1] <= 620
in_nums = lambda s: 465 <= box_of(s)[0] <= 475 and 320 <= box_of(s)[1] <= 620
in_doc = lambda s: box_of(s)[0] < 240 and 280 <= box_of(s)[1] <= 600

# ---------------------------------------------------------------- ranking: A B C D E G F
chip = {txt(s): s for s in shapes() if s.name.startswith("chip ") and in_rank(s)}
num = {txt(s): s for s in shapes() if in_nums(s) and txt(s).isdigit()}
star_r = one(lambda s: s.name == "star" and 540 <= box_of(s)[0] <= 570 and 500 <= box_of(s)[1] <= 530, "ranked star")
g_chip = clone(chip["F"]); set_text(g_chip, "G"); g_chip.name = "chip G"
n7 = clone(num["6"]); set_text(n7, "7")
order = ["A", "B", "C", "D", "E", "G", "F"]
chip["G"] = g_chip; num["7"] = n7
PITCH, CH, Y0 = 42, 36, 325
for r, lab in enumerate(order):
    y = Y0 + PITCH * r
    place(chip[lab], 510, y, 566, y + CH)
    place(num[str(r + 1)], 470, y, 498, y + CH)
    if lab == "E":
        place(star_r, 556, y - 8, 578, y + 14)

# ---------------------------------------------------------------- document: a long span G
doc_f = one(lambda s: s.name == "chip F" and in_doc(s), "doc chip F")
doc_g = clone(doc_f); set_text(doc_g, "G"); doc_g.name = "chip G"
place(doc_g, 42, 338, 152, 368)

# ---------------------------------------------------------------- card 3: G is skipped, F backfills
set_text(one(lambda s: txt(s) == "Candidate", "Candidate label"), "Candidate G")
for s in [s for s in shapes() if txt(s) == "Packed" and 1060 <= box_of(s)[0] <= 1080]:
    set_text(s, "Packed A–E")
set_text(one(lambda s: txt(s) == "Backfill" and 1180 <= box_of(s)[0] <= 1200, "Backfill block"), "Backfill F")

# ---------------------------------------------------------------- DedupPack bar: F is the backfilled span
seg_f = one(lambda s: s.name == "segment F" and 670 <= box_of(s)[1] <= 700, "DP segment F")
fx0, _, fx1, fy1 = box_of(seg_f)
lab = clone(one(lambda s: s.name == "budget label", "budget label"))
set_text(lab, "backfill"); lab.name = "backfill label"
for r in lab.text_frame.paragraphs[0].runs: r.font.color.rgb = RGBColor.from_string("1D5A96")
cx = (fx0 + fx1) / 2
place(lab, cx - 40, fy1 + 1, cx + 40, fy1 + 19)

prs.save(DST)
print("saved", DST)
