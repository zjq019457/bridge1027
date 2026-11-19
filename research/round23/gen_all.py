"""Standalone all-budget table (Table 3 + RECOMP), bold = best compressed, underline = second; not part of the paper."""
BS = chr(92)  # backslash
R = [  # name, group, [q7 512,1024,2048, llama..., q14..., q32...]
 ("LLMLingua-2", "Prompt compression", [20.06, 24.60, 32.76, 22.84, 28.43, 34.28, 21.12, 27.10, 35.97, 22.38, 29.39, 36.31]),
 ("LongLLMLingua", "Prompt compression", [30.85, 37.46, 40.80, 32.06, 39.58, 43.21, 34.68, 40.77, 45.51, 35.70, 41.88, 46.03]),
 ("LlamaIndex sentence window", "Published context construction", [31.98, 37.02, 41.48, 32.65, 36.96, 41.35, 35.86, 39.99, 45.50, 36.08, 41.49, 46.02]),
 ("LlamaIndex auto-merging", "Published context construction", [31.84, 39.30, 40.92, 33.23, 39.13, 43.52, 34.65, 41.50, 45.93, 35.11, 42.52, 47.12]),
 (r"Provence (BGE + pruning)$^\dagger$", "Published context construction", [40.66, 41.90, 41.78, 39.62, 40.69, 41.36, 42.70, 44.27, 44.36, 43.66, 45.66, 45.68]),
 (r"RECOMP (extractive)$^\dagger$", "Published context construction", [38.22, 40.46, 41.74, 38.08, 41.56, 42.40, 42.88, 45.50, 47.30, 42.63, 45.93, 47.39]),
 ("Sentence chunks + BGE", "Published context construction", [35.74, 38.55, 42.46, 35.73, 39.98, 43.11, 40.04, 44.24, 47.79, 39.89, 44.34, 48.62]),
 ("Lexical pair packing", "Published context construction", [33.88, 39.11, 40.68, 34.05, 37.79, 39.60, 36.73, 41.69, 44.23, 37.51, 42.57, 44.77]),
 ("BGE-small chunks", "Chunk packing", [33.05, 38.85, 40.98, 33.27, 38.99, 40.85, 36.27, 42.34, 45.29, 37.52, 42.40, 46.08]),
 (r"\quad + LongContextReorder", "Chunk packing", [33.06, 39.21, 41.24, 33.32, 39.12, 41.26, 35.90, 41.48, 44.97, 37.42, 42.34, 45.96]),
 (r"\quad + EmbeddingsRedundantFilter", "Chunk packing", [33.53, 38.53, 41.66, 33.94, 39.28, 42.16, 36.50, 42.41, 45.96, 37.94, 42.66, 45.72]),
 ("BGE-large chunks", "Chunk packing", [34.34, 38.74, 42.55, 35.05, 40.14, 42.49, 38.69, 42.48, 45.38, 39.93, 43.04, 47.14]),
 ("Reranker chunks", "Chunk packing", [37.19, 41.59, 43.48, 38.29, 42.22, 43.72, 41.18, 44.37, 47.30, 41.74, 45.09, 48.15]),
 ("Hybrid chunks", "Chunk packing", [34.27, 38.60, 41.55, 34.20, 38.85, 40.61, 36.75, 42.42, 45.45, 37.96, 42.02, 45.25]),
 (r"BGE-small + \spk{}", r"\spk{}", [35.64, 40.76, 43.14, 36.16, 41.67, 43.58, 39.21, 45.18, 48.10, 40.34, 46.28, 49.01]),
 (r"BGE-large + \spk{}", r"\spk{}", [38.24, 42.28, 44.32, 38.81, 43.98, 44.65, 41.54, 46.23, 48.61, 41.05, 46.52, 49.25]),
 (r"Reranker + \spk{}", r"\spk{}", [39.56, 42.79, 44.96, 40.49, 44.13, 45.61, 42.85, 46.43, 49.54, 44.05, 47.32, 50.47]),
 (r"Hybrid + \spk{}", r"\spk{}", [36.74, 40.64, 43.60, 36.44, 40.98, 42.78, 40.44, 44.32, 47.50, 40.88, 45.07, 49.15]),
]
FULL = [43.04, 43.13, 47.59, 47.79]
rows = []
for n, g, v in R:
    avg = [round(sum(v[m * 3 + b] for m in range(4)) / 4 + 1e-9, 2) for b in range(3)]
    rows.append((n, g, v + avg))
ncol = 15
fmt = [[f"{r[2][c]:.2f}" for c in range(ncol)] for r in rows]
for c in range(ncol):
    vals = sorted({r[2][c] for r in rows}, reverse=True)
    for i, r in enumerate(rows):
        if r[2][c] == vals[0]:
            fmt[i][c] = BS + "textbf{" + fmt[i][c] + "}"
        elif r[2][c] == vals[1]:
            fmt[i][c] = BS + "underline{" + fmt[i][c] + "}"
L = [r"""\documentclass[11pt]{article}
% Standalone preview in ACL style; not part of the paper.
\usepackage[T1]{fontenc}
\usepackage{times}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{caption}
\usepackage{colortbl}
\usepackage[paperwidth=19cm,paperheight=12.3cm,margin=0.4cm]{geometry}
\pagestyle{empty}
\newcommand{\spk}{\textsc{DedupPack}}
\begin{document}
\begin{table}[h]
\centering
\small
\setlength{\tabcolsep}{3pt}
\resizebox{18.2cm}{!}{%
\begin{tabular}{l ccc ccc ccc ccc ccc}
\toprule
& \multicolumn{3}{c}{Qwen2.5-7B} & \multicolumn{3}{c}{Llama-3.1-8B} & \multicolumn{3}{c}{Qwen2.5-14B} & \multicolumn{3}{c}{Qwen2.5-32B} & \multicolumn{3}{c}{Average} \\""",
     "Full context (30K tokens) & " + " & ".join(BS + "multicolumn{3}{c}{" + f"{x:.2f}" + "}" for x in FULL + [round(sum(FULL) / 4, 2)]) + r" \\",
     r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}\cmidrule(lr){11-13}\cmidrule(lr){14-16}",
     "Budget (tokens) & " + " & ".join(["512 & 1,024 & 2,048"] * 5) + r" \\"]
g0 = None
for (n, g, _), f in zip(rows, fmt):
    if g != g0:
        L += [r"\midrule", BS + "multicolumn{16}{l}{" + BS + "textit{" + g + "}} " + BS + BS]
        g0 = g
    pre = r"\rowcolor[gray]{0.90}" if n.startswith("RECOMP") else ""
    L.append(pre + n + " & " + " & ".join(f) + r" \\")
L += [r"\bottomrule", r"\end{tabular}}",
      r"\caption*{\footnotesize Macro F1 on the 1,150 LongBench QA instances at each exact budget (shaded: RECOMP with its HotpotQA extractive checkpoint). Bold: best compressed result in each column; underlined: second best. $^\dagger$Trained compressor; \spk{} needs no training.}",
      r"\end{table}", r"\end{document}"]
open("recomp_all.tex", "w", encoding="utf-8").write("\n".join(L) + "\n")
rc = next(r for r in rows if r[0].startswith("RECOMP"))
print("RECOMP rank per column (of", len(rows), "):", [1 + sum(r[2][c] > rc[2][c] for r in rows) for c in range(ncol)])
for r in rows:
    if "spk" in r[0] or "Provence" in r[0] or "RECOMP" in r[0]:
        print(r[0], "avg", r[2][12:])
for c in range(ncol):
    best = max(rows, key=lambda r: r[2][c])
    print(c, best[0], best[2][c])
