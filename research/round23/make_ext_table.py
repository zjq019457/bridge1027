"""Standalone 1,024-token table with EXIT, CPC and CompAct added (preview only, not part of the paper)."""
t = open("recomp_table.tex", encoding="utf-8").read()
anchor = r"RECOMP (extractive)$^\dagger$ & 40.46 & 41.56 & 45.50 & 42.51 \\"
assert anchor in t
new = [(r"EXIT (Gemma-2B)$^\dagger$", [40.84, 42.24, 44.48]),
       (r"CPC (Mistral-7B)$^\dagger$", [42.03, 42.97, 45.62]),
       (r"CompAct (7B, abstractive)$^\dagger$", [35.70, 34.70, 37.53])]
rows = "".join("\n" + r"\rowcolor[gray]{0.90}" + n + " & " + " & ".join(f"{x:.2f}" for x in v + [round(sum(v) / 3 + 1e-9, 2)]) + r" \\" for n, v in new)
t = t.replace(anchor, anchor + rows)
old = r"BGE-large + \spk{} & 42.28 & 43.98 & 46.23 & 44.16 \\"
assert old in t
t = t.replace(old, r"BGE-large + \spk{} & \underline{42.28} & \underline{43.98} & \underline{46.23} & \underline{44.16} \\")
t = t.replace(r"\usepackage{graphicx}", "\\usepackage{graphicx}\n\\usepackage{colortbl}")
t = t.replace("paperheight=14cm", "paperheight=15.6cm")
t = t.replace(r"Bold: best compressed result in each column.",
              r"Shaded: EXIT, CPC and CompAct (released checkpoints, official prompts). Bold: best compressed result in each column; underlined: second best.")
open("ext_table.tex", "w", encoding="utf-8").write(t)
print("ok")
