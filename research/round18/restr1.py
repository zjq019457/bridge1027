"""Restructure step 1: no 'main' configuration; the hybrid ranker is one of the rankers, with its own
chunk-packing row (results pending). Tables, captions and short method text only."""
P = r"\pend{}"
PEND12 = " & ".join([P] * 12)
EN = [
    # Rankers / development protocol
    (r" Our main configuration uses the hybrid ranker, chosen on the development split; Section~\ref{sec:agnostic} shows the gains do not depend on this choice.",
     r" We report \spk{} with every ranker; Section~\ref{sec:external} adds BGE-large and a cross-encoder reranker. The packer is the same in all cases."),
    (r"All choices---the ranker, sentence rather than fixed repair,", r"All packer choices---sentence rather than fixed repair,"),
    (r" The configuration was then frozen and the test set was run once with every model and budget reported here; no test result was used to change the method.",
     r" The packer was then frozen and the test set was run once with every ranker, model and budget reported here; no test result was used to change the packer."),
    # Table 2
    ("\\midrule\n\\multicolumn{2}{l}{\\spk{} (frozen hybrid configuration)} & 44.32 & 40.98 & 45.07 & 42.65 & \\\\\n", ""),
    (" & \\spk{} & \\textbf{46.45} & \\textbf{44.12} & \\textbf{47.32} & \\textbf{45.29} & \\textbf{+2.00}$^{***}$ \\\\\n",
     " & \\spk{} & \\textbf{46.45} & \\textbf{44.12} & \\textbf{47.32} & \\textbf{45.29} & \\textbf{+2.00}$^{***}$ \\\\\n"
     "\\multirow{2}{*}{\\shortstack[l]{Hybrid (lexical\\\\+ BGE-small)}} & chunks & " + P + " & " + P + " & " + P + " & " + P + " & \\\\\n"
     " & \\spk{} & 44.32 & 40.98 & 45.07 & 42.65 & " + P + " \\\\\n"),
    (r"\spk{} is a packer and can follow any ranker: the frozen hybrid configuration was fixed on the development split before testing; the other rows pair it with the ranker of the chunk-packing row above.",
     r"\spk{} is a packer and can follow any ranker: each \spk{} row uses the ranker of the chunk-packing row above it, and the packer is the same in all four."),
    # Table 3
    ("\nHybrid + \\spk{} (frozen, main) &", "\nHybrid + \\spk{} &"),
    (r"\spk{} is a packer and can follow any ranker: the hybrid configuration was fixed on the development split before testing, and the other \spk{} rows pair it with the rankers of the chunk-packing rows.",
     r"\spk{} is a packer and can follow any ranker: each \spk{} row pairs the same packer, fixed on the development split before testing, with the ranker of the corresponding chunk-packing row."),
    # other tables
    (r"\spk{} (hybrid, frozen) & \textbf{62.56}", r"\spk{} (hybrid) & \textbf{62.56}"),
    (r"\spk{} (hybrid, frozen) & \textbf{38.03}", r"\spk{} (hybrid) & \textbf{38.03}"),
    (r"\spk{} denotes the frozen hybrid configuration.", r"\spk{} denotes \spk{} with the hybrid ranker."),
    (r"At 1,024 tokens, the frozen configuration stays below", r"At 1,024 tokens, \spk{} with the hybrid ranker stays below"),
    (r"\caption{Leave-one-out ablation of the frozen hybrid configuration.", r"\caption{Leave-one-out ablation of \spk{} with the hybrid ranker."),
    (r"\paragraph{Leave-one-out on the hybrid configuration.}", r"\paragraph{Leave-one-out with the hybrid ranker.}"),
    (r"(Qwen tokenizer; ablation rows: hybrid configuration)", r"(Qwen tokenizer; ablation rows: hybrid ranker)"),
    ("Hybrid (frozen) & \\spk{} & 40.64", "Hybrid & \\spk{} & 40.64"),
]
ZH = [
    ("我们的主配置使用在开发集上选出的混合排序器；第~\\ref{sec:agnostic} 节表明提升并不依赖这一选择。",
     "我们报告 \\spk{} 与每一种排序器的组合；第~\\ref{sec:external} 节另外加入 BGE-large 和交叉编码器重排序器。所有情况下打包器完全相同。"),
    ("所有选择——排序器、句子补全还是固定窗口补全、", "打包器的所有选择——句子补全还是固定窗口补全、"),
    ("随后配置被冻结，测试集以文中报告的每个模型和预算只运行一次；没有任何测试结果被用于修改方法。",
     "随后打包器被冻结，测试集以文中报告的每种排序器、每个模型和每个预算只运行一次；没有任何测试结果被用于修改打包器。"),
    ("\\midrule\n\\multicolumn{2}{l}{\\spk{}（冻结的混合配置）} & 44.32 & 40.98 & 45.07 & 42.65 & \\\\\n", ""),
    (" & \\spk{} & \\textbf{46.45} & \\textbf{44.12} & \\textbf{47.32} & \\textbf{45.29} & \\textbf{+2.00}$^{***}$ \\\\\n",
     " & \\spk{} & \\textbf{46.45} & \\textbf{44.12} & \\textbf{47.32} & \\textbf{45.29} & \\textbf{+2.00}$^{***}$ \\\\\n"
     "\\multirow{2}{*}{\\shortstack[l]{混合（词法\\\\+ BGE-small）}} & 切块 & " + P + " & " + P + " & " + P + " & " + P + " & \\\\\n"
     " & \\spk{} & 44.32 & 40.98 & 45.07 & 42.65 & " + P + " \\\\\n"),
    ("\\spk{} 是打包器，可接在任何排序器之后：冻结的混合配置在测试前于开发集上固定，其余各行把它与上方切块打包行所用的排序器配对。",
     "\\spk{} 是打包器，可接在任何排序器之后：每个 \\spk{} 行使用其上方切块打包行的排序器，四组中的打包器完全相同。"),
    ("\n混合 + \\spk{}（冻结，主配置） &", "\n混合 + \\spk{} &"),
    ("\\spk{} 是打包器，可接在任何排序器之后：混合配置在测试前于开发集上固定，其余 \\spk{} 各行把它与切块打包各行所用的排序器配对。",
     "\\spk{} 是打包器，可接在任何排序器之后：每个 \\spk{} 行都是同一个打包器（测试前在开发集上固定）与对应切块打包行的排序器的组合。"),
    ("\\spk{}（混合，冻结配置） & \\textbf{62.56}", "\\spk{}（混合） & \\textbf{62.56}"),
    ("\\spk{}（混合，冻结配置） & \\textbf{38.03}", "\\spk{}（混合） & \\textbf{38.03}"),
    ("\\spk{} 指冻结的混合配置。", "\\spk{} 指使用混合排序器的 \\spk{}。"),
    ("在 1,024 token 时，冻结配置在每个模型上都低于完整上下文", "在 1,024 token 时，使用混合排序器的 \\spk{} 在每个模型上都低于完整上下文"),
    ("\\caption{对冻结的混合配置逐项去除的消融。", "\\caption{对使用混合排序器的 \\spk{} 逐项去除的消融。"),
    ("\\paragraph{混合配置上的逐项去除。}", "\\paragraph{混合排序器下的逐项去除。}"),
    ("（Qwen 分词器；消融行为混合配置）", "（Qwen 分词器；消融行使用混合排序器）"),
    ("混合（冻结） & \\spk{} & 40.64", "混合 & \\spk{} & 40.64"),
]
for fn, reps in [("main.tex", EN), ("main_zh.tex", ZH)]:
    t = open(fn, encoding="utf-8").read()
    for a, b in reps:
        assert t.count(a) == 1, (fn, a[:70], t.count(a))
        t = t.replace(a, b)
    # Table 3: hybrid chunk-packing row after the reranker chunk row
    L = t.split("\n")
    i = next(k for k, l in enumerate(L) if l.startswith(("Reranker chunks &", "重排序切块 &")))
    name = "Hybrid chunks" if fn == "main.tex" else "混合排序切块"
    L.insert(i + 1, f"{name} & {PEND12} \\\\")
    t = "\n".join(L)
    open(fn, "w", encoding="utf-8", newline="").write(t); print(fn, "ok")
