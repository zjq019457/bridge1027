"""Abstract tail and Figure 1 caption for the 32B-only Figure 1 (main.tex, main_zh.tex)."""
import shutil, sys
D = r"C:\Users\dell\Desktop\BridgePack_ACL\paper"
EDITS = {
    "main.tex": [
        ("With the reranker, it is the best compressed method in 10 of the 12 model--budget settings, above LlamaIndex, LongLLMLingua, LLMLingua-2 and four trained compressors at 1,024 tokens, and with one sixth of the input it beats the full context on all four models.",
         "With the reranker, it is the best compressed method in 10 of the 12 model--budget settings and outperforms four trained compressors at 1,024 tokens; with one sixth of the input it beats the full context on all four models, by 2.7 F1 on Qwen2.5-32B."),
        (r"\caption{Macro F1 on LongBench QA (1,150 instances) under exact token budgets, with two rankers. With either ranker, \spk{} outperforms chunk packing with the same ranking at every budget on all four models; with the reranker, at 2,048 tokens (one sixth of the average 12.5K-token document) it scores above the full context on all four models.}",
         r"\caption{Macro F1 of Qwen2.5-32B on LongBench QA (1,150 instances) under exact token budgets, against representative methods from every family in Table~\ref{tab:main}. Reranker + \spk{} is the best compressed method at every budget, 2.2--2.3 F1 above chunk packing with the same reranker; at 2,048 tokens, one sixth of the average 12.5K-token document, it scores 2.7 F1 above the full context. Table~\ref{tab:main} gives all four models.}"),
    ],
    "main_zh.tex": [
        ("配合重排序器时，它在 12 个模型--预算组合中的 10 个上是最好的压缩方法；在 1,024 token 下，它高于 LlamaIndex、LongLLMLingua、LLMLingua-2 和四种需要训练的压缩器；只用六分之一的输入，它在全部四个模型上都超过完整上下文。",
         "配合重排序器时，它在 12 个模型--预算组合中的 10 个上是最好的压缩方法，在 1,024 token 下高于四种需要训练的压缩器；只用六分之一的输入，它在全部四个模型上都超过完整上下文，在 Qwen2.5-32B 上高出 2.7 F1。"),
        (r"\caption{LongBench 问答（1,150 个样本）在精确 token 预算下的宏平均 F1，使用两种排序器。无论用哪种排序器，\spk{} 在全部四个模型、每档预算下都优于使用同一排序的切块打包；配合重排序器时，在 2,048 token（约为平均 12.5K token 文档的六分之一）下，它在全部四个模型上都高于完整上下文。}",
         r"\caption{Qwen2.5-32B 在 LongBench 问答（1,150 个样本）上、精确 token 预算下的宏平均 F1，对比表~\ref{tab:main} 中每一类方法的代表。重排序器 + \spk{} 在每档预算下都是最好的压缩方法，比使用同一重排序器的切块打包高 2.2--2.3 F1；在 2,048 token（约为平均 12.5K token 文档的六分之一）下，它比完整上下文高 2.7 F1。全部四个模型的结果见表~\ref{tab:main}。}"),
    ],
}
for name, pairs in EDITS.items():
    p = D + "\\" + name
    t = open(p, encoding="utf-8").read()
    for old, new in pairs:
        if t.count(old) != 1:
            sys.exit(f"{name}: found {t.count(old)}x: {old[:50]}")
        t = t.replace(old, new)
    shutil.copy(p, p + ".bak_r36")
    open(p, "w", encoding="utf-8", newline="").write(t)
    if name == "main.tex":
        a = t[t.index("\\begin{abstract}") + len("\\begin{abstract}"):t.index("\\end{abstract}")]
        print(name, "ok, abstract words:", len(a.replace("\\spk{}", "DedupPack").split()))
    else:
        print(name, "ok")
