"""Figure 2 caption for v9: ranking A, B, C, D, E, G, F; G is skipped and F backfills (main.tex, main_zh.tex)."""
import shutil, sys
D = r"C:\Users\dell\Desktop\BridgePack_ACL\paper"
EDITS = {
    "main.tex": [
        ("(illustrative example: A--F are candidate spans in rank order, and E holds the answer)",
         "(illustrative example: the ranker orders the candidate spans A, B, C, D, E, G, F, and E holds the answer)"),
        (r"\spk{} replaces only the packing step: it pads spans to sentence boundaries, merges overlapping ranges, fills the exact token budget with backfilling and emits the result in document order, so the same budget also carries E and F.",
         r"\spk{} replaces only the packing step: it pads spans to sentence boundaries, merges overlapping ranges and admits spans in rank order under the exact token budget. The recovered budget brings in E; the long span G does not fit and is skipped, and the shorter F backfills the remaining tokens. The result is emitted in document order."),
    ],
    "main_zh.tex": [
        ("（示意图：A--F 为按排名排列的候选片段，E 含有答案）",
         "（示意图：排序器给出的候选片段顺序为 A、B、C、D、E、G、F，E 含有答案）"),
        (r"\spk{} 只替换打包这一步：它把片段补全到句子边界、合并重叠区间、以回填方式精确填满 token 预算，并按原文顺序输出，因此同样的预算还能装下 E 和 F。",
         r"\spk{} 只替换打包这一步：它把片段补全到句子边界、合并重叠区间，并在精确的 token 预算下按排名依次接纳片段。省下的预算装进了 E；较长的 G 放不下，被跳过；较短的 F 补满剩余预算（回填）。结果按原文顺序输出。"),
    ],
}
for name, pairs in EDITS.items():
    p = D + "\\" + name
    t = open(p, encoding="utf-8").read()
    for old, new in pairs:
        if t.count(old) != 1:
            sys.exit(f"{name}: found {t.count(old)}x: {old[:50]}")
        t = t.replace(old, new)
    shutil.copy(p, p + ".bak_r34")
    open(p, "w", encoding="utf-8", newline="").write(t)
    print(name, "ok")
