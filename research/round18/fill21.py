"""Point the baseline references at the new Table base18 and keep only the BM25/TF-IDF part of the
earlier-environment paragraph (truncation and Selective Context are now run in the current environment;
LLMLingua-2 and LongLLMLingua are in the main grid)."""
EN = [
    ("Appendix~\\ref{app:extra} compares the pair packer with truncation, Selective Context and BM25/TF-IDF chunk packing in an earlier environment.",
     "Appendix~\\ref{app:extra} adds head and middle truncation, Selective Context and RECOMP under the same budget."),
    ("Appendix~\\ref{app:extra} shows the same ordering for Selective Context and LongLLMLingua.",
     "Selective Context, which also deletes tokens, reaches only 22.3--25.5 F1 (Appendix~\\ref{app:extra})."),
    ("the pair packer of Table~\\ref{tab:main} outperformed head and middle truncation by 13.1 and 13.6 F1, Selective Context by 18.2, LLMLingua-2 by 11.3, LongLLMLingua by 6.6, and BM25 and TF-IDF chunk packing by 2.1 and 1.9 F1 (two-model averages, all $p<0.001$). Since \\spk{} outperforms the pair packer in every setting of Table~\\ref{tab:main}, these baselines are dominated by \\spk{} as well; we report them here rather than in the main table because they were run in a different environment.",
     "the pair packer of Table~\\ref{tab:main} also outperformed BM25 and TF-IDF chunk packing by 2.1 and 1.9 F1 (two-model averages, both $p<0.001$). Since \\spk{} outperforms the pair packer in every setting of Table~\\ref{tab:main}, it outperforms these two baselines as well."),
]
ZH = [
    ("附录~\\ref{app:extra} 给出了早期环境中配对打包器与截断、Selective Context 以及 BM25/TF-IDF 切块打包的比较。",
     "附录~\\ref{app:extra} 在相同预算下补充了开头截断、中间截断、Selective Context 和 RECOMP。"),
    ("附录~\\ref{app:extra} 显示 Selective Context 和 LongLLMLingua 也呈现同样的排序。",
     "同样删除 token 的 Selective Context 也只有 22.3--25.5 F1（附录~\\ref{app:extra}）。"),
    ("表~\\ref{tab:main} 中的配对打包器比开头截断和中间截断分别高 13.1 和 13.6 F1，比 Selective Context 高 18.2，比 LLMLingua-2 高 11.3，比 LongLLMLingua 高 6.6，比 BM25 和 TF-IDF 切块打包分别高 2.1 和 1.9 F1（两模型平均，均 $p<0.001$）。由于 \\spk{} 在表~\\ref{tab:main} 的每个组合上都优于配对打包器，这些基线同样被 \\spk{} 超过；由于它们是在不同环境中运行的，我们把它们放在这里而不是主表中。",
     "表~\\ref{tab:main} 中的配对打包器还比 BM25 和 TF-IDF 切块打包分别高 2.1 和 1.9 F1（两模型平均，均 $p<0.001$）。由于 \\spk{} 在表~\\ref{tab:main} 的每个组合上都优于配对打包器，这两个基线同样被 \\spk{} 超过。"),
]
for fn, reps in [("main.tex", EN), ("main_zh.tex", ZH)]:
    t = open(fn, encoding="utf-8").read()
    for a, b in reps:
        assert t.count(a) == 1, (fn, a[:60], t.count(a))
        t = t.replace(a, b)
    open(fn, "w", encoding="utf-8", newline="").write(t)
    print(fn, "ok")
