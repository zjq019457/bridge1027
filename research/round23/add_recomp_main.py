"""Move RECOMP into Table 3 (all budgets, four readers) in main.tex and main_zh.tex, and update the claims it affects."""
import shutil, sys
D = r"C:\Users\dell\Desktop\BridgePack_ACL\paper"

NUMS = r"38.22 & 40.46 & 41.74 & 38.08 & 41.56 & 42.40 & \textbf{42.88} & 45.50 & 47.30 & 42.63 & 45.93 & 47.39 \\"
PROV_OLD = r"& \textbf{40.66} & 41.90 & 41.78 & \underline{39.62} & 40.69 & 41.36 & \underline{42.70} & 44.27 & 44.36 & \underline{43.66} & 45.66 & 45.68 \\"
PROV_NEW = r"& \textbf{40.66} & 41.90 & 41.78 & \underline{39.62} & 40.69 & 41.36 & 42.70 & 44.27 & 44.36 & \underline{43.66} & 45.66 & 45.68 \\"
RR_OLD = r"& \underline{39.56} & \textbf{42.79} & \textbf{44.96} & \textbf{40.49} & \textbf{44.13} & \textbf{45.61} & \textbf{42.85} & \textbf{46.43}"
RR_NEW = r"& \underline{39.56} & \textbf{42.79} & \textbf{44.96} & \textbf{40.49} & \textbf{44.13} & \textbf{45.61} & \underline{42.85} & \textbf{46.43}"

EN = [
    # Table 3: RECOMP row after Provence, bold/underline on Qwen-14B at 512
    ("Provence (BGE + pruning)$^\\dagger$ " + PROV_OLD,
     "Provence (BGE + pruning)$^\\dagger$ " + PROV_NEW + "\nRECOMP (extractive)$^\\dagger$ & " + NUMS),
    (RR_OLD, RR_NEW),
    (r"$^\dagger$Uses a context pruner trained with supervision for this purpose; \spk{} needs no training. }",
     r"$^\dagger$Uses a compressor trained with supervision for this purpose (a context pruner for Provence, an extractive sentence compressor for RECOMP); \spk{} needs no training. }"),
    # setup
    (r"Provence \citep{chirkova2025provence} applied to the top 60 BGE chunks, and LongLLMLingua",
     r"Provence \citep{chirkova2025provence} applied to the top 60 BGE chunks, the extractive compressor of RECOMP \citep{xu2023recomp} (HotpotQA checkpoint), which ranks all sentences of the context by their similarity to the question and adds them in that order while they fit, and LongLLMLingua"),
    (r"Appendix~\ref{app:extra} adds head and middle truncation, Selective Context and RECOMP under the same budget.",
     r"Appendix~\ref{app:extra} adds head and middle truncation and Selective Context under the same budget."),
    # contribution bullet
    (r"above LlamaIndex's sentence-window and auto-merging retrievers, Provence, LongLLMLingua and LLMLingua-2 (Table~\ref{tab:main})",
     r"above LlamaIndex's sentence-window and auto-merging retrievers, Provence, RECOMP, LongLLMLingua and LLMLingua-2 (Table~\ref{tab:main})"),
    # 6.1
    (r"Table~\ref{tab:main} compares all methods at 1,024 tokens: \spk{} is above every prompt compressor and published context-construction system,",
     r"Table~\ref{tab:main} compares all methods at 1,024 tokens: with BGE-large or the reranker, \spk{} is above every prompt compressor and published context-construction system on all four models,"),
    # 6.3
    (r"\spk{} with the reranker is the best compressed result in 11 of the 12 model--budget columns,",
     r"\spk{} with the reranker is the best compressed result in 10 of the 12 model--budget columns and second in the other two,"),
    (r"at 512 tokens it is the best system on Qwen2.5-7B and second on the other three models (Table~\ref{tab:main}).",
     r"at 512 tokens it is the best system on Qwen2.5-7B and second on Llama-3.1-8B and Qwen2.5-32B (Table~\ref{tab:main})."),
    (r"given a reranker, \spk{} is ahead of it in 11 of the 12 model--budget settings.",
     r"given a reranker, \spk{} is ahead of it in 11 of the 12 model--budget settings. RECOMP \citep{xu2023recomp}, whose extractive compressor is trained on HotpotQA, is likewise strongest at small budgets, where it is the best system on Qwen2.5-14B at 512 tokens (by 0.03 F1), but it gains less from a larger budget. Pooled over the four models, \spk{} with the reranker is ahead of it by 1.28 F1 at 512 tokens, 1.80 at 1,024 and 2.94 at 2,048 (the last two intervals exclude zero), and at 2,048 tokens all four \spk{} variants are above it on every model."),
    # appendix table
    ("RECOMP (extractive) & 40.46 & 41.56 & 45.50 \\\\\n", ""),
    (r"Selective Context \citep{li2023selectivecontext} removes the phrases with the lowest GPT-2 self-information; RECOMP \citep{xu2023recomp} adds sentences in the order given by its extractive compressor (HotpotQA checkpoint) until the budget is full.}",
     r"Selective Context \citep{li2023selectivecontext} removes the phrases with the lowest GPT-2 self-information. RECOMP is compared at every budget in Table~\ref{tab:main}.}"),
    (r"Table~\ref{tab:base18} adds four further baselines in the current environment, under the same 1,024-token budget. Head and middle truncation and Selective Context stay 15--20 F1 below BGE-small + \spk{}. RECOMP is the strongest of these baselines; \spk{} with BGE-large or the reranker scores above it on all three models.",
     r"Table~\ref{tab:base18} adds three further baselines in the current environment, under the same 1,024-token budget. Head and middle truncation and Selective Context stay 15--20 F1 below BGE-small + \spk{}."),
]

ZH = [
    ("Provence（BGE + 剪枝）$^\\dagger$ " + PROV_OLD,
     "Provence（BGE + 剪枝）$^\\dagger$ " + PROV_NEW + "\nRECOMP（抽取式）$^\\dagger$ & " + NUMS),
    (RR_OLD, RR_NEW),
    (r"$^\dagger$使用专门为此训练的上下文剪枝模型；\spk{} 无需训练。}",
     r"$^\dagger$使用专门为此有监督训练的压缩模型（Provence 为上下文剪枝模型，RECOMP 为抽取式句子压缩器）；\spk{} 无需训练。}"),
    (r"应用于 BGE 前 60 个块的 Provence \citep{chirkova2025provence}，以及使用其推荐的问题感知设置的 LongLLMLingua",
     r"应用于 BGE 前 60 个块的 Provence \citep{chirkova2025provence}、RECOMP 的抽取式压缩器 \citep{xu2023recomp}（HotpotQA 检查点；按与问题的相似度给上下文中所有句子排序，并按此顺序在放得下时依次加入），以及使用其推荐的问题感知设置的 LongLLMLingua"),
    (r"附录~\ref{app:extra} 在相同预算下补充了开头截断、中间截断、Selective Context 和 RECOMP。",
     r"附录~\ref{app:extra} 在相同预算下补充了开头截断、中间截断和 Selective Context。"),
    (r"高于 LlamaIndex 的句子窗口与自动合并检索器、Provence、LongLLMLingua 和 LLMLingua-2（表~\ref{tab:main}）",
     r"高于 LlamaIndex 的句子窗口与自动合并检索器、Provence、RECOMP、LongLLMLingua 和 LLMLingua-2（表~\ref{tab:main}）"),
    (r"表~\ref{tab:main} 比较了 1,024 token 下的全部方法：\spk{} 高于所有提示压缩方法和已发表的上下文构建系统，",
     r"表~\ref{tab:main} 比较了 1,024 token 下的全部方法：使用 BGE-large 或重排序器时，\spk{} 在全部四个模型上都高于所有提示压缩方法和已发表的上下文构建系统，"),
    (r"配合重排序器的 \spk{} 在 12 个模型--预算组合中的 11 个上是最好的压缩结果，",
     r"配合重排序器的 \spk{} 在 12 个模型--预算组合中的 10 个上是最好的压缩结果，在其余 2 个上排第二，"),
    (r"512 token 时，它在 Qwen2.5-7B 上是最好的系统，在另外三个模型上排第二（表~\ref{tab:main}）。",
     r"512 token 时，它在 Qwen2.5-7B 上是最好的系统，在 Llama-3.1-8B 和 Qwen2.5-32B 上排第二（表~\ref{tab:main}）。"),
    (r"配合重排序器时，\spk{} 在 12 个模型--预算组合中的 11 个上领先于它。",
     r"配合重排序器时，\spk{} 在 12 个模型--预算组合中的 11 个上领先于它。RECOMP \citep{xu2023recomp} 的抽取式压缩器在 HotpotQA 上训练，同样在小预算下最强：512 token 时它在 Qwen2.5-14B 上是最好的系统（高 0.03 F1），但预算增大时它的提升较小。合并四个模型后，配合重排序器的 \spk{} 在 512、1,024 和 2,048 token 时分别比它高 1.28、1.80 和 2.94 F1（后两个区间不包含 0）；在 2,048 token 时，全部四个 \spk{} 变体在每个模型上都高于它。"),
    ("RECOMP（抽取式） & 40.46 & 41.56 & 45.50 \\\\\n", ""),
    (r"Selective Context \citep{li2023selectivecontext} 删除 GPT-2 自信息最低的短语；RECOMP \citep{xu2023recomp} 按其抽取式压缩器（HotpotQA 检查点）的句子排序依次加入句子，直到填满预算。}",
     r"Selective Context \citep{li2023selectivecontext} 删除 GPT-2 自信息最低的短语。RECOMP 在每档预算下的结果见表~\ref{tab:main}。}"),
    (r"表~\ref{tab:base18} 在当前环境和相同的 1,024 token 预算下补充了四个基线。开头截断、中间截断和 Selective Context 比 BGE-small + \spk{} 低 15--20 F1。RECOMP 是其中最强的基线；使用 BGE-large 或重排序器的 \spk{} 在三个模型上都高于它。",
     r"表~\ref{tab:base18} 在当前环境和相同的 1,024 token 预算下补充了三个基线。开头截断、中间截断和 Selective Context 比 BGE-small + \spk{} 低 15--20 F1。"),
]

for name, edits in (("main.tex", EN), ("main_zh.tex", ZH)):
    p = D + "\\" + name
    t = open(p, encoding="utf-8").read()
    for i, (a, b) in enumerate(edits):
        n = t.count(a)
        if n != 1:
            sys.exit(f"{name} edit {i}: found {n} times: {a[:80]}")
        t = t.replace(a, b)
    shutil.copy(p, p + ".bak_r22")
    open(p, "w", encoding="utf-8", newline="").write(t)
    print("updated", name, len(edits), "edits")
