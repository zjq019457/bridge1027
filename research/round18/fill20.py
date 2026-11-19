"""Round 18/19 final fill: re-generated predictions for Llama/14B ranker cells (earlier predictions were lost;
table values now match the predictions used in the paired tests), significance rows for BGE-large/reranker at
1,024 tokens and for the hybrid ranker (pooled over four models), and the two paragraphs quoting these numbers."""
import re


def ci(v, lo, hi):
    return (f"\\textbf{{{v}}}", f"{{\\scriptsize [{lo}, {hi}]}}")


SIG = {  # name -> three (value, lo, hi)
    "BGE-large": [("+2.91", "1.83", "4.04"), ("+3.66", "2.51", "4.80"), ("+2.32", "1.23", "3.42")],
    "Reranker": [("+2.14", "1.13", "3.18"), ("+1.85", "0.77", "2.97"), ("+1.98", "0.95", "3.03")],
    "Hybrid": [("+2.83", "1.68", "3.98"), ("+2.28", "1.12", "3.49"), ("+2.54", "1.42", "3.68")],
}
ZHN = {"BGE-large": "BGE-large", "Reranker": "重排序器", "Hybrid": "混合"}

for fn, zh in [("main.tex", False), ("main_zh.tex", True)]:
    t = open(fn, encoding="utf-8").read()
    reps = [
        ("& 35.05 & 40.12 & 42.49 &", "& 35.05 & 40.14 & 42.49 &"),
        ("& 38.29 & 42.21 & 43.72 &", "& 38.29 & 42.22 & 43.72 &"),
        (r"\textbf{44.12}", r"\textbf{44.13}"),
        (r"\textbf{46.45}", r"\textbf{46.43}"),
        ("& 42.79 & 44.12 & 46.45 & 47.32", "& 42.79 & 44.13 & 46.43 & 47.32"),
    ]
    if zh:
        reps += [
            ("表~\\ref{tab:main} 在 Qwen2.5-14B 和 Llama-3.1-8B 上用两种更强的排序器重复了“只换打包器”的实验。使用 BGE-large 时，\\spk{} 比切块打包高 3.77 F1；使用交叉编码重排序器（在 BGE 前 100 个片段上运行 \\texttt{bge-reranker-v2-m3}）时高 2.00。",
             "表~\\ref{tab:main} 在全部四个模型上用两种更强的排序器重复了“只换打包器”的实验。在 1,024 token 下合并四个模型，使用 BGE-large 时 \\spk{} 比切块打包高 3.66 F1，使用交叉编码重排序器（在 BGE 前 100 个片段上运行 \\texttt{bge-reranker-v2-m3}）时高 1.85（表~\\ref{tab:sig}）。"),
            ("配合重排序器的 \\spk{} 在两个模型上都是最好的压缩配置，在 Llama-3.1-8B 上甚至超过了 3 万 token 的完整上下文（44.12 对 43.13）。",
             "配合重排序器的 \\spk{} 在 12 个模型--预算组合中的 11 个上是最好的压缩结果，在 Llama-3.1-8B 上 1,024 token 时甚至超过了 3 万 token 的完整上下文（44.13 对 43.13）。"),
            ("更换打包器在三档预算下分别提升 +2.81、+2.48 和 +2.57", "更换打包器在三档预算下分别提升 +2.81、+2.83 和 +2.66"),
        ]
    else:
        reps += [
            ("Table~\\ref{tab:main} repeats the packer swap with two stronger rankers on Qwen2.5-14B and Llama-3.1-8B. With BGE-large, \\spk{} improves on chunk packing by 3.77 F1; with a cross-encoder reranker (\\texttt{bge-reranker-v2-m3} over the top 100 BGE spans) by 2.00.",
             "Table~\\ref{tab:main} repeats the packer swap with two stronger rankers on all four models. Pooled over the four models at 1,024 tokens, \\spk{} improves on chunk packing by 3.66 F1 with BGE-large and by 1.85 with a cross-encoder reranker (\\texttt{bge-reranker-v2-m3} over the top 100 BGE spans; Table~\\ref{tab:sig})."),
            ("\\spk{} with the reranker is the best compressed configuration on both models, and on Llama-3.1-8B it scores above the 30K-token full context (44.12 vs.\\ 43.13).",
             "\\spk{} with the reranker is the best compressed result in 11 of the 12 model--budget columns, and on Llama-3.1-8B at 1,024 tokens it scores above the 30K-token full context (44.13 vs.\\ 43.13)."),
            ("Changing the packer improves F1 by +2.81, +2.48 and +2.57", "Changing the packer improves F1 by +2.81, +2.83 and +2.66"),
        ]
    for a, b in reps:
        assert t.count(a) == 1, (fn, a[:60], t.count(a))
        t = t.replace(a, b)
    # significance rows (value line + interval line)
    for name, vals in SIG.items():
        lab = ZHN[name] if zh else name
        c = [ci(*v) for v in vals]
        new = f"{lab} & " + " & ".join(x[0] for x in c) + " \\\\\n & " + " & ".join(x[1] for x in c) + " \\\\\n"
        pat = re.compile("^" + re.escape(lab) + r" & .*\\\\\n( & .*\\\\\n)?", re.M)
        i = t.index(r"\label{tab:sig}")
        a = t.rindex(r"\begin{tabular}{lccc}", 0, i)
        seg, n = pat.subn(lambda _: new, t[a:i], count=1)
        assert n == 1, (fn, name)
        t = t[:a] + seg + t[i:]
    open(fn, "w", encoding="utf-8", newline="").write(t)
    print(fn, "ok")
