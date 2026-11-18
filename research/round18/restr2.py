"""Restructure step 2: drop Table 2 (its numbers are all in Table 3; its significance column moves to Table 4,
now per ranker and pooled over four models), and put the full-context reference in Table 3's header."""
import re
P = r"\pend{}"


def cell(v, ci):
    return (f"\\textbf{{{v}}}", f"{{\\scriptsize [{ci}]}}") if v else (P, "")


ROWS = [  # name_en, name_zh, [(value, interval) x 3]
    ("BGE-small", "BGE-small", [("+2.81", "1.69, 3.95"), ("+2.83", "1.68, 3.98"), ("+2.66", "1.53, 3.81")]),
    ("BGE-large", "BGE-large", [("+2.91", "1.83, 4.04"), (None, None), ("+2.32", "1.23, 3.42")]),
    ("Reranker", "重排序器", [("+2.14", "1.13, 3.18"), (None, None), ("+1.98", "0.95, 3.03")]),
    ("Hybrid", "混合", [(None, None)] * 3),
]
PAIR = [("+2.30", "0.69, 3.93"), ("+3.19", "1.72, 4.64"), ("+3.63", "2.26, 5.00")]


def sig_table(zh):
    head = (r"\spk{} 减去同一排序器的切块打包 & 512 & 1,024 & 2,048 \\" if zh else
            r"\spk{} minus chunk packing, same ranker & 512 & 1,024 & 2,048 \\")
    L = [r"\begin{tabular}{lccc}", r"\toprule", head, r"\midrule"]
    for en, z, vals in ROWS:
        c = [cell(*v) for v in vals]
        L.append(f"{z if zh else en} & " + " & ".join(x[0] for x in c) + r" \\")
        if any(x[1] for x in c):
            L.append(" & " + " & ".join(x[1] for x in c) + r" \\")
    L.append(r"\midrule")
    c = [cell(*v) for v in PAIR]
    L.append((r"\spk{}（BGE-small）减去配对打包" if zh else r"\spk{} (BGE-small) minus pair packing") + " & " + " & ".join(x[0] for x in c) + r" \\")
    L.append(" & " + " & ".join(x[1] for x in c) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}}"]
    cap = (r"\caption{四个模型合并后的配对 F1 差值（95\% bootstrap 区间）。所有区间都不包含 0。各模型单独的检验见附录~\ref{app:permodel}。}" if zh else
           r"\caption{Paired F1 differences pooled over the four models (95\% bootstrap intervals). All intervals exclude zero. Per-model tests are in Appendix~\ref{app:permodel}.}")
    return "\n".join(L) + "\n" + cap


for fn, zh in [("main.tex", False), ("main_zh.tex", True)]:
    t = open(fn, encoding="utf-8").read()
    # 1. drop Table 2
    i = t.index(r"\label{tab:external}")
    a = t.rindex(r"\begin{table}[t]", 0, i); b = t.index(r"\end{table}", i) + len(r"\end{table}")
    t = t[:a].rstrip("\n") + "\n" + t[b:].lstrip("\n")
    t = t.replace(r"(Tables~\ref{tab:external} and~\ref{tab:main})", r"(Table~\ref{tab:main})")
    t = t.replace(r"（表~\ref{tab:external} 和表~\ref{tab:main}）", r"（表~\ref{tab:main}）")
    t = t.replace(r"\ref{tab:external}", r"\ref{tab:main}")
    # 2. Table 3: full context in the header
    if zh:
        old = ("方法 & 512 & 1,024 & 2,048 & 512 & 1,024 & 2,048 & 512 & 1,024 & 2,048 & 512 & 1,024 & 2,048 \\\\\n\\midrule\n"
               "完整上下文（参考） & \\multicolumn{3}{c}{43.04} & \\multicolumn{3}{c}{43.13} & \\multicolumn{3}{c}{47.59} & \\multicolumn{3}{c}{47.79} \\\\\n\\midrule\n")
        new = ("预算（token） & 512 & 1,024 & 2,048 & 512 & 1,024 & 2,048 & 512 & 1,024 & 2,048 & 512 & 1,024 & 2,048 \\\\\n\\midrule\n")
        full = "完整上下文（30K token） & \\multicolumn{3}{c}{43.04} & \\multicolumn{3}{c}{43.13} & \\multicolumn{3}{c}{47.59} & \\multicolumn{3}{c}{47.79} \\\\\n"
    else:
        old = ("Method & 512 & 1,024 & 2,048 & 512 & 1,024 & 2,048 & 512 & 1,024 & 2,048 & 512 & 1,024 & 2,048 \\\\\n\\midrule\n"
               "Full context (reference) & \\multicolumn{3}{c}{43.04} & \\multicolumn{3}{c}{43.13} & \\multicolumn{3}{c}{47.59} & \\multicolumn{3}{c}{47.79} \\\\\n\\midrule\n")
        new = ("Budget (tokens) & 512 & 1,024 & 2,048 & 512 & 1,024 & 2,048 & 512 & 1,024 & 2,048 & 512 & 1,024 & 2,048 \\\\\n\\midrule\n")
        full = "Full context (30K tokens) & \\multicolumn{3}{c}{43.04} & \\multicolumn{3}{c}{43.13} & \\multicolumn{3}{c}{47.59} & \\multicolumn{3}{c}{47.79} \\\\\n"
    assert t.count(old) == 1, fn
    t = t.replace(old, new)
    cm = "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}\\cmidrule(lr){11-13}\n" + new
    assert t.count(cm) == 1, fn
    t = t.replace(cm, full + cm)
    # 3. Table 4
    i = t.index(r"\label{tab:sig}")
    a = t.rindex(r"\begin{tabular}{lccc}", 0, i)
    t = t[:a] + sig_table(zh) + "\n" + t[i:]
    open(fn, "w", encoding="utf-8", newline="").write(t)
    print(fn, "ok", t.count("tab:external"))
