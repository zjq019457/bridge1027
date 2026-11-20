"""Figure 2 caption for the v4 figure (A-F in rank order, E holds the answer), main.tex and main_zh.tex."""
import shutil, sys
D = r"C:\Users\dell\Desktop\BridgePack_ACL\paper"
NEW = {
    "main.tex": ("\\caption{Where \\spk{} runs.",
        r"\caption{Where \spk{} runs (illustrative example: A--F are candidate spans in rank order, and E holds the answer). Before the LLM call, any ranker scores spans of the long document; standard chunk packing then pays twice for the overlap between the padded windows of neighbouring spans (23\% of a 1,024-token budget on LongBench), cuts pieces mid-sentence and leaves budget unused, so E does not fit. \spk{} replaces only the packing step: it pads spans to sentence boundaries, merges overlapping ranges, fills the exact token budget with backfilling and emits the result in document order, so the same budget also carries E and F. It runs on the CPU, needs no training and leaves the ranker and the model unchanged; compared with the full context, the LLM prefills about one twelfth of the tokens at 1,024 tokens (one sixth at 2,048).}"),
    "main_zh.tex": ("\\caption{\\spk{} 的运行位置。",
        r"\caption{\spk{} 的运行位置（示意图：A--F 为按排名排列的候选片段，E 含有答案）。在调用 LLM 之前，任意排序器为长文档的片段打分；标准切块打包会为相邻片段补全窗口之间的重叠重复付费（LongBench 上 1,024 token 预算的 23\%），在句子中间截断片段，并留下未用完的预算，因此 E 放不进去。\spk{} 只替换打包这一步：它把片段补全到句子边界、合并重叠区间、以回填方式精确填满 token 预算，并按原文顺序输出，因此同样的预算还能装下 E 和 F。它在 CPU 上运行，无需训练，排序器和模型都保持不变；在 1,024 token 预算下，LLM 预填充的 token 约为完整上下文的十二分之一（2,048 token 时约为六分之一）。}"),
}
for name, (start, new) in NEW.items():
    p = D + "\\" + name
    t = open(p, encoding="utf-8").read()
    if t.count(start) != 1:
        sys.exit(f"{name}: start found {t.count(start)}x")
    i = t.index(start); j = t.index("\n", i)
    old = t[i:j]
    if not old.endswith("}"):
        sys.exit(f"{name}: caption is not on one line")
    t = t[:i] + new + t[j:]
    shutil.copy(p, p + ".bak_r33")
    open(p, "w", encoding="utf-8", newline="").write(t)
    print(name, "ok")
