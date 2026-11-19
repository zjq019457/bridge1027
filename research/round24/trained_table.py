"""Move EXIT and CPC from Table 2 into a separate 1,024-token table of trained compressors (main.tex, main_zh.tex)."""
import shutil, sys
D = r"C:\Users\dell\Desktop\BridgePack_ACL\paper"
ROWS = ("EXIT$^\\dagger$ & -- & 40.84 & -- & -- & 42.24 & -- & -- & 44.48 & -- & -- & -- & -- \\\\\n"
        "CPC$^\\dagger$ & -- & 42.03 & -- & -- & 42.97 & -- & -- & 45.62 & -- & -- & -- & -- \\\\\n")


def table(head, g1, g2, cap):
    return (r"""\begin{table}[t]
\centering
\small
\setlength{\tabcolsep}{3pt}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lccccc}
\toprule
""" + head + r""" \\
\midrule
\multicolumn{6}{l}{\textit{""" + g1 + r"""}} \\
Provence & 435M & 41.90 & 40.69 & 44.27 & 42.29 \\
RECOMP & 110M & 40.46 & 41.56 & 45.50 & 42.51 \\
EXIT & 2.5B & 40.84 & 42.24 & 44.48 & 42.52 \\
CPC & 7.2B & 42.03 & 42.97 & 45.62 & 43.54 \\
\midrule
\multicolumn{6}{l}{\textit{""" + g2 + r"""}} \\
BGE-small + \spk{} & 33M & 40.76 & 41.67 & 45.18 & 42.54 \\
BGE-large + \spk{} & 335M & \underline{42.28} & \underline{43.98} & \underline{46.23} & \underline{44.16} \\
Reranker + \spk{} & 33M + 568M & \textbf{42.79} & \textbf{44.13} & \textbf{46.43} & \textbf{44.45} \\
\bottomrule
\end{tabular}}
\caption{""" + cap + r"""}
\label{tab:trained}
\end{table}

""")


EN_TAB = table(r"1,024 tokens & Model & Qwen-7B & Llama-8B & Qwen-14B & Avg.", "Trained compressors (released checkpoints)", r"\spk{} (no training)",
               r"Trained compressors and \spk{} under the same exact 1,024-token budget (macro F1, 1,150 instances). Model: parameters of the model run at inference besides the reader; for \spk{} this is the ranker, and the packer itself has none. Each compressor is trained with supervision for context selection. EXIT keeps the sentences above a fixed threshold (893 tokens on average) and fills a 2,048-token budget in fewer than 7\% of instances, so all four are compared at 1,024 tokens. Bold: best; underlined: second best.")
ZH_TAB = table(r"1,024 token & 模型 & Qwen-7B & Llama-8B & Qwen-14B & 平均", "需要训练的压缩器（发布的检查点）", r"\spk{}（无需训练）",
               r"相同 1,024 token 精确预算下需要训练的压缩器与 \spk{} 的对比（宏平均 F1，1,150 个样本）。模型：除阅读模型外推理时运行的模型参数量；对 \spk{} 而言是排序器，打包器本身没有参数。每个压缩器都为上下文选择做过有监督训练。EXIT 保留高于固定阈值的句子（平均 893 token），在 2,048 token 预算下只有不到 7\% 的样本能填满预算，因此四者都在 1,024 token 下比较。加粗：最好；下划线：第二好。")

EN = [
    (ROWS, ""),
    (r"EXIT and CPC were run at 1,024 tokens on the first three models (--: not run). ", ""),
    (r"(Provence: context pruner; RECOMP: extractive sentence compressor; EXIT: sentence classifier; CPC: sentence encoder);",
     r"(Provence: context pruner; RECOMP: extractive sentence compressor);"),
    (r"on every model tested (Table~\ref{tab:main}).", r"on every model tested (Tables~\ref{tab:main} and~\ref{tab:trained})."),
    (r"and trained compressor at 1,024 tokens on every model; Section~\ref{sec:external} discusses these comparisons.",
     r"and trained compressor at 1,024 tokens on every model (Tables~\ref{tab:main} and~\ref{tab:trained}); Section~\ref{sec:external} discusses these comparisons."),
    (r"EXIT \citep{hwang2025exit} and CPC \citep{liskavets2025cpc}, which we run at 1,024 tokens on Qwen2.5-7B, Llama-3.1-8B and Qwen2.5-14B, are also below \spk{} with the reranker on every model.",
     r"Table~\ref{tab:trained} collects the trained compressors at 1,024 tokens, including EXIT \citep{hwang2025exit} and CPC \citep{liskavets2025cpc}, which we run on Qwen2.5-7B, Llama-3.1-8B and Qwen2.5-14B; all four are below \spk{} with BGE-large or the reranker on every model."),
    ("\\subsection{Beyond question answering}\n", EN_TAB + "\\subsection{Beyond question answering}\n"),
]
ZH = [
    (ROWS, ""),
    (r"EXIT 和 CPC 只在前三个模型上以 1,024 token 运行（--：未运行）。", ""),
    (r"（Provence 为上下文剪枝模型，RECOMP 为抽取式句子压缩器，EXIT 为句子分类器，CPC 为句子编码器）；",
     r"（Provence 为上下文剪枝模型，RECOMP 为抽取式句子压缩器）；"),
    (r"（Provence、RECOMP、EXIT、CPC）（表~\ref{tab:main}）。", r"（Provence、RECOMP、EXIT、CPC）（表~\ref{tab:main} 和表~\ref{tab:trained}）。"),
    (r"已发表的上下文构建系统和需要训练的压缩器；第~\ref{sec:external} 节讨论这些比较。",
     r"已发表的上下文构建系统和需要训练的压缩器（表~\ref{tab:main} 和表~\ref{tab:trained}）；第~\ref{sec:external} 节讨论这些比较。"),
    (r"我们在 Qwen2.5-7B、Llama-3.1-8B 和 Qwen2.5-14B 上以 1,024 token 运行了 EXIT \citep{hwang2025exit} 和 CPC \citep{liskavets2025cpc}，它们在每个模型上也都低于配合重排序器的 \spk{}。",
     r"表~\ref{tab:trained} 汇总了 1,024 token 下需要训练的压缩器，其中 EXIT \citep{hwang2025exit} 和 CPC \citep{liskavets2025cpc} 是我们在 Qwen2.5-7B、Llama-3.1-8B 和 Qwen2.5-14B 上运行的；四者在每个模型上都低于配合 BGE-large 或重排序器的 \spk{}。"),
    ("\\subsection{问答之外的任务}\n", ZH_TAB + "\\subsection{问答之外的任务}\n"),
]
for name, edits in (("main.tex", EN), ("main_zh.tex", ZH)):
    p = D + "\\" + name
    t = open(p, encoding="utf-8").read()
    for i, (a, b) in enumerate(edits):
        if t.count(a) != 1:
            sys.exit(f"{name} edit {i}: found {t.count(a)}x: {a[:70]}")
        t = t.replace(a, b)
    shutil.copy(p, p + ".bak_r26")
    open(p, "w", encoding="utf-8", newline="").write(t)
    print(name, "ok")
