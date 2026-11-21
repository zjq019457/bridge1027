"""Fix three points on the user's hand-edited fig_overview_v6.pptx and save it as fig_overview_v7.pptx:
1. card 3: the skipped candidate is wider than the free slot (1.3x), starts where the slot starts,
   and the budget-end line now runs through the candidate row so the overhang is visible;
2. card 3 caption: "exact fill + backfill" (missing space);
3. the DedupPack bar, its brace, star, label and budget ruler are stretched to the same x-range as the
   standard bar (704-1460 px), so both 1,024-token budgets are drawn with the same length.
Shapes are found by their text / name and position, so every other manual edit is kept."""
from pptx import Presentation
from pptx.util import Emu, Inches

SRC, DST = "fig_overview_v6.pptx", "fig_overview_v7.pptx"
PPI = 1886 / 12.6
P = lambda v: Emu(int(Inches(v / PPI)))
px = lambda e: Emu(e).inches * PPI
prs = Presentation(SRC)
shapes = list(prs.slides[0].shapes)


def box_of(sh):
    return px(sh.left), px(sh.top), px(sh.left + sh.width), px(sh.top + sh.height)


def near(a, b, tol=3):
    return all(abs(x - y) <= tol for x, y in zip(a, b))


def find(pred, what):
    hits = [sh for sh in shapes if pred(sh)]
    assert len(hits) >= 1, f"not found: {what}"
    return hits


def text_of(sh):
    return sh.text_frame.text if sh.has_text_frame else ""


# ---------------------------------------------------------------- 1. card 3 candidate
free = find(lambda s: text_of(s) == "Free", "Free")[0]
fx0, fy0, fx1, fy1 = box_of(free)
cand = find(lambda s: s.name == "candidate too large", "candidate box")[0]
cx0, cy0, cx1, cy1 = box_of(cand)
new_w = 1.3 * (fx1 - fx0)
cand.left, cand.width = P(fx0), P(new_w)
shift = (fx0 + new_w) - cx1                       # move the red x with the box's right end
for s in find(lambda s: s.shape_type == 9 and near(box_of(s), (1310, 459, 1322, 471), 4), "x mark"):
    s.begin_x, s.end_x = s.begin_x + P(shift), s.end_x + P(shift)
budget_end = find(lambda s: s.shape_type == 9 and near(box_of(s), (fx1, 479, fx1, 543), 4), "budget end line")[0]
if budget_end.begin_y < budget_end.end_y:
    budget_end.begin_y = P(cy0 - 3)
else:
    budget_end.end_y = P(cy0 - 3)

# ---------------------------------------------------------------- 2. caption spacing
cap = find(lambda s: text_of(s).replace(" ", "") == "exactfill+backfill", "caption")[0]
runs = [r for p in cap.text_frame.paragraphs for r in p.runs]
runs[0].text = runs[0].text.rstrip() + " "
card = find(lambda s: s.name == "step card" and near(box_of(s), (1058, 446, 1328, 608), 4), "card 3")[0]
k0, _, k1, _ = box_of(card)
cap.left = P((k0 + k1) / 2 - px(cap.width) / 2)      # centre the caption on the card

# ---------------------------------------------------------------- 3. stretch the DedupPack bar to 704-1460
STD0, STD1 = 704, 1460
segs = sorted(find(lambda s: s.name.startswith("segment ") and 675 <= box_of(s)[1] <= 690, "DP segments"), key=lambda s: s.left)
D0, D1 = box_of(segs[0])[0], box_of(segs[-1])[2]
T = lambda x: STD0 + (x - D0) * (STD1 - STD0) / (D1 - D0)
for s in segs:
    x0, _, x1, _ = box_of(s)
    s.left, s.width = P(T(x0)), P(T(x1) - T(x0))
st = find(lambda s: s.name == "star" and 670 <= box_of(s)[1] <= 700, "E star")[0]
x0, _, x1, _ = box_of(st)
st.left = P(T((x0 + x1) / 2) - (x1 - x0) / 2)
br = find(lambda s: s.shape_type == 5 and near(box_of(s)[1::2], (655, 667), 4) and box_of(s)[0] > 1100, "brace")[0]
x0, _, x1, _ = box_of(br)
br.left, br.width = P(T(x0)), P(T(x1) - T(x0))
lab = find(lambda s: text_of(s).startswith("Recovered space"), "recovered label")[0]
x0, _, x1, _ = box_of(lab)
lab.left = P(T((x0 + x1) / 2) - (x1 - x0) / 2)
blab = find(lambda s: text_of(s) == "1,024-token budget", "budget label")[0]
x0, _, x1, _ = box_of(blab)
mid = T((x0 + x1) / 2); half = (x1 - x0) / 2
blab.left = P(mid - half)
for s in find(lambda s: s.shape_type == 9 and 740 <= box_of(s)[1] <= 755, "budget ruler"):
    a, b = px(s.begin_x), px(s.end_x)
    if abs(a - b) < 1:                                   # end tick
        xx = STD0 if a < 1000 else STD1
        s.begin_x = s.end_x = P(xx)
    elif min(a, b) < 1000:                               # left half of the ruler
        s.begin_x, s.end_x = (P(STD0), P(mid - half)) if a < b else (P(mid - half), P(STD0))
    else:                                                # right half
        s.begin_x, s.end_x = (P(mid + half), P(STD1)) if a < b else (P(STD1), P(mid + half))

prs.save(DST)
print("saved", DST, "| DP bar", round(D0), round(D1), "->", STD0, STD1, "| candidate width", round(new_w), "vs free", round(fx1 - fx0))
