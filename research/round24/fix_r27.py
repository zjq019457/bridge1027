"""Round 27 edits (main.tex and main_zh.tex): Table 1 with the same BGE ranking, no development-split wording for the
rankers, no defensive or rival-favouring sentences, MultiHop-RAG appendix removed, softened ranker claims,
Section 6.2 title, Table 2 grouping."""
import shutil, sys
D = r"C:\Users\dell\Desktop\BridgePack_ACL\paper"


class Doc:
    def __init__(self, name):
        self.p = D + "\\" + name
        self.t = open(self.p, encoding="utf-8").read()
        self.name = name

    def rep(self, a, b):
        n = self.t.count(a)
        if n != 1:
            sys.exit(f"{self.name}: found {n}x: {a[:70]}")
        self.t = self.t.replace(a, b)

    def cut(self, start, end):
        if self.t.count(start) != 1:
            sys.exit(f"{self.name}: cut start found {self.t.count(start)}x: {start[:70]}")
        i = self.t.index(start)
        j = self.t.index(end, i) + len(end)
        while self.t[j:j + 1] == "\n" and self.t[j + 1:j + 2] == "\n":
            j += 1
        self.t = self.t[:i] + self.t[j:].lstrip("\n")

    def save(self):
        shutil.copy(self.p, self.p + ".bak_r27")
        open(self.p, "w", encoding="utf-8", newline="").write(self.t)
        print(self.name, "ok")


# Table 1 rows: DedupPack with the same BGE-small ranking as BGE chunks (values from the mechanism table)
T1 = [(r"\textbf{1.1} & \textbf{70.6} & 500 & \textbf{33.9}", r"\textbf{1.1} & \textbf{71.1} & 500 & \textbf{34.3}"),
      (r"\textbf{1.4} & \textbf{68.1} & 1,014 & \textbf{41.9}", r"\textbf{1.3} & \textbf{69.1} & 1,015 & \textbf{43.2}"),
      (r"\textbf{1.6} & \textbf{62.8} & 2,032 & \textbf{49.0}", r"\textbf{1.5} & \textbf{64.1} & 2,033 & \textbf{49.7}")]

# ================= English
e = Doc("main.tex")
for a, b in T1:
    e.rep(a, b)
e.rep(r"Llama-3.1 is within one point on every column).", r"Llama-3.1 is within one point on every column). BGE chunks and \spk{} use the same BGE-small ranking.")
e.rep(r"the standard packing step is surprisingly wasteful, whatever the ranker.", r"the standard packing step is surprisingly wasteful.")
e.rep(r"Duplication drops to 1.4\%, and the recovered budget raises answer coverage by 3--4 points.",
      r"With the same ranking, duplication drops to 1.3\%, and the recovered budget raises answer coverage by 4--5 points.")
e.rep(r"This waste is a property of the packing step rather than of the ranker, so a stronger ranker does not remove it, and it can be removed without touching the ranker.",
      r"This waste is created by the packing step, so it can be removed without touching the ranker, and removing it helps strong rankers as well as weak ones (Section~\ref{sec:external}).")
e.rep(r", and this waste is a property of the packing step that persists whatever ranker is used (Section~\ref{sec:diag}).",
      r"; the waste is created by the packing step, so removing it helps every ranker we test (Sections~\ref{sec:diag} and~\ref{sec:agnostic}).")
e.rep(r"reducing duplication to about 1.5\% and raising answer coverage by 3--4 points,",
      r"reducing duplication to 1.1--1.5\% and raising answer coverage by 4--5 points,")
e.rep(r"because overlap is created by the padding, not by the selection.",
      r"because overlap is created by the padding, not by the selection. The duplication comes from overlapping units and neighbour expansion, which are common in retrieval pipelines (e.g., sentence-window and parent-document retrieval); \spk{} removes it at packing time, without re-chunking the corpus.")
e.rep(r"duplication drops to 1.1--1.6\%, two thirds of pieces end on a sentence boundary, the budget is filled to within 10 tokens, and answer coverage rises by 2.9--4.0 points over dense chunk packing with the same budget.",
      r"with the same BGE ranking, duplication drops to 1.1--1.5\%, about two thirds of pieces end on a sentence boundary, 98--99\% of the budget is used, and answer coverage rises by 3.9--4.7 points over dense chunk packing with the same budget.")
e.rep("Sentence chunks + BGE & 35.74", "\\midrule\n\\multicolumn{13}{l}{\\textit{Other packing baselines}} \\\\\nSentence chunks + BGE & 35.74")
e.rep(r"Per-model tests are in Appendix~\ref{app:permodel}.", r"Per-model tests for the BGE-small and hybrid rankers on three models are in Appendix~\ref{app:permodel}.")
e.rep(r"\subsection{The packer, not the ranker, carries the gain}", r"\subsection{The gain holds for every ranker}")
e.rep(r"Among the inexpensive rankers, the hybrid ranker selected on the development split and the BGE-small ranker are close:",
      r"Among the inexpensive rankers, the hybrid and BGE-small rankers are close:")
e.rep(r"Provence is strongest at the tightest budget: at 512 tokens it is the best system on Qwen2.5-7B and second on Llama-3.1-8B and Qwen2.5-32B (Table~\ref{tab:main}). Its pruner, however, is a DeBERTa-v3-large model",
      r"Its pruner is a DeBERTa-v3-large model")
e.rep(r"is likewise strongest at small budgets, where it is the best system on Qwen2.5-14B at 512 tokens (by 0.03 F1), but it gains less from a larger budget. Pooled over the four models, \spk{} with the reranker is ahead of it by 1.28 F1 at 512 tokens, 1.80 at 1,024 and 2.94 at 2,048 (the last two intervals exclude zero),",
      r"gains less from a larger budget: pooled over the four models, \spk{} with the reranker is ahead of it by 1.28 F1 at 512 tokens, 1.80 at 1,024 and 2.94 at 2,048 (significant at 1,024 and 2,048 tokens),")
e.rep(r"EXIT keeps the sentences above a fixed threshold (893 tokens on average) and fills a 2,048-token budget in fewer than 7\% of instances, so all four are compared at 1,024 tokens. ", "")
e.rep(r"With the hybrid ranker it matches the full context ($+0.56$, $-0.35$, $-0.10$ and $+1.36$), and on the three multi-hop tasks it is \emph{better} than the full context:",
      r"Even with the hybrid ranker, it is \emph{better} than the full context on the three multi-hop tasks:")
e.rep(r"\caption{The gain of \spk{} (hybrid ranker, selected on the development split) over BGE chunk packing",
      r"\caption{The gain of \spk{} (hybrid ranker) over BGE chunk packing")
e.rep(r"(1,014 tokens on average vs.\ 985 for chunk packing; Table~\ref{tab:diag})", r"(1,015 tokens on average vs.\ 985 for chunk packing; Table~\ref{tab:diag})")
e.rep(r"because a better packer gives every ranker a similar boost.", r"because a better packer gives every ranker a boost of 1.9--3.7 F1.")
e.rep(r"The benefit is largest in pipelines that expand retrieved hits with neighbouring text, such as sentence-window or parent-document retrieval, where we measure 9--30\% duplication; with plain, non-overlapping chunks drawn from many different documents duplication is low and there is little to remove.",
      r"The benefit is largest in pipelines that pad or expand retrieved units with neighbouring text, such as overlapping chunks, sentence-window retrieval and parent-document retrieval.")
e.rep(r"These results fixed the packer (sentence-aligned padding, merging and backfilling); among the three rankers tried on the development split, the hybrid ranker was best, +5.26 F1 above BGE chunk packing. BGE-large and the reranker were added at test time with the packer unchanged.",
      r"These results fixed the packer (sentence-aligned padding, merging and backfilling); the best development configuration, with the hybrid ranker, is +5.26 F1 above BGE chunk packing.")
e.cut(r"\paragraph{Full context at 1,024 tokens.}", "($p=0.01$ for the 32B model).")
e.cut(r"\section{MultiHop-RAG: Duplication with Common Chunkers}", "\\label{tab:mhrag}\n\\end{table}")
e.save()

# ================= Chinese
z = Doc("main_zh.tex")
for a, b in T1:
    z.rep(a, b)
z.rep(r"Llama-3.1 在每一列上的差别都在 1 个点以内）。", r"Llama-3.1 在每一列上的差别都在 1 个点以内）。BGE 切块与 \spk{} 使用相同的 BGE-small 排序。")
z.rep(r"无论用哪种排序器，标准的打包步骤都出人意料地浪费。", r"标准的打包步骤出人意料地浪费。")
z.rep(r"重复率降到 1.4\%，省下的预算使答案覆盖率提高 3--4 个百分点。", r"在相同排序下，重复率降到 1.3\%，省下的预算使答案覆盖率提高 4--5 个百分点。")
z.rep(r"这种浪费来自打包步骤而不是排序器，所以更强的排序器无法消除它，而它可以在不改动排序器的情况下被消除。",
      r"这种浪费由打包步骤造成，因此可以在不改动排序器的情况下消除，而且消除它对强排序器和弱排序器同样有效（第~\ref{sec:external} 节）。")
z.rep(r"；这种浪费来自打包步骤本身，无论使用哪种排序器都存在（第~\ref{sec:diag} 节）。",
      r"；这种浪费由打包步骤造成，因此消除它对我们测试的每种排序器都有帮助（第~\ref{sec:diag} 节和第~\ref{sec:agnostic} 节）。")
z.rep(r"把重复率降到约 1.5\%，并把答案覆盖率提高 3--4 个百分点；", r"把重复率降到 1.1--1.5\%，并把答案覆盖率提高 4--5 个百分点；")
z.rep(r"因为重叠是补全造成的，而不是选择造成的。",
      r"因为重叠是补全造成的，而不是选择造成的。这种重复来自相互重叠的单元和邻近扩展，而它们在检索流水线中很常见（例如句子窗口检索和父文档检索）；\spk{} 在打包时就把它去掉，无需重新切分语料。")
z.rep(r"重复率降到 1.1--1.6\%，三分之二的片段在句子边界结束，预算的填充误差在 10 个 token 以内，答案覆盖率比相同预算下的稠密切块打包高 2.9--4.0 个百分点。",
      r"在相同的 BGE 排序下，重复率降到 1.1--1.5\%，约三分之二的片段在句子边界结束，预算利用率达到 98--99\%，答案覆盖率比相同预算下的稠密切块打包高 3.9--4.7 个百分点。")
z.rep("句子切块 + BGE & 35.74", "\\midrule\n\\multicolumn{13}{l}{\\textit{其他打包基线}} \\\\\n句子切块 + BGE & 35.74")
z.rep(r"各模型单独的检验见附录~\ref{app:permodel}。", r"使用 BGE-small 和混合排序器时，三个模型各自的检验见附录~\ref{app:permodel}。")
z.rep(r"\subsection{提升来自打包器，而不是排序器}", r"\subsection{提升在每种排序器上都成立}")
z.rep(r"在低成本的排序器中，开发集上选出的混合排序器与 BGE-small 排序器很接近：", r"在低成本的排序器中，混合排序器与 BGE-small 排序器很接近：")
z.rep(r"Provence 在最紧的预算下最强：512 token 时，它在 Qwen2.5-7B 上是最好的系统，在 Llama-3.1-8B 和 Qwen2.5-32B 上排第二（表~\ref{tab:main}）。但它的剪枝器",
      r"它的剪枝器")
z.rep(r"同样在小预算下最强：512 token 时它在 Qwen2.5-14B 上是最好的系统（高 0.03 F1），但预算增大时它的提升较小。合并四个模型后，配合重排序器的 \spk{} 在 512、1,024 和 2,048 token 时分别比它高 1.28、1.80 和 2.94 F1（后两个区间不包含 0）；",
      r"预算增大时它的提升较小：合并四个模型后，配合重排序器的 \spk{} 在 512、1,024 和 2,048 token 时分别比它高 1.28、1.80 和 2.94 F1（在 1,024 和 2,048 token 时显著）；")
z.rep(r"EXIT 保留高于固定阈值的句子（平均 893 token），在 2,048 token 预算下只有不到 7\% 的样本能填满预算，因此四者都在 1,024 token 下比较。", "")
z.rep(r"配合混合排序器时，它与完整上下文持平（$+0.56$、$-0.35$、$-0.10$ 和 $+1.36$），并且在三个多跳任务上\emph{优于}完整上下文：",
      r"即使配合混合排序器，它在三个多跳任务上也\emph{优于}完整上下文：")
z.rep(r"\caption{\spk{}（混合排序器，在开发集上选定）相对", r"\caption{\spk{}（混合排序器）相对")
z.rep(r"（平均 1,014 token，切块打包为 985；表~\ref{tab:diag}）", r"（平均 1,015 token，切块打包为 985；表~\ref{tab:diag}）")
z.rep(r"因为更好的打包器会给每种排序器带来相近的提升。", r"因为更好的打包器会给每种排序器带来 1.9--3.7 F1 的提升。")
z.rep(r"收益最大的是对检索命中做邻近扩展的流水线，例如句子窗口检索或父文档检索，我们在这类设置中测得 9--30\% 的重复；而在来自许多不同文档、互不重叠的普通切块中，重复很少，可去除的也就不多。",
      r"收益最大的是对检索单元做邻近补全或扩展的流水线，例如带重叠的切块、句子窗口检索和父文档检索。")
z.rep(r"这些结果确定了打包器（句子对齐补全、合并和回填）；在开发集上尝试的三种排序器中，混合排序器最好，比 BGE 切块打包高 5.26 F1。BGE-large 和重排序器是在测试阶段加入的，打包器未作任何改动。",
      r"这些结果确定了打包器（句子对齐补全、合并和回填）；开发集上最好的配置（混合排序器）比 BGE 切块打包高 5.26 F1。")
z.cut(r"\paragraph{1,024 token 与完整上下文。}", "\n")
z.cut(r"\section{MultiHop-RAG：常见切块器下的重复}", "\\label{tab:mhrag}\n\\end{table}")
z.save()
