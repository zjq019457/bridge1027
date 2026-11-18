"""(1) BGE-large chunks, Qwen2.5-14B, 1,024 tokens: re-generated predictions (the earlier ones were lost)
give 42.48 instead of 42.55; the table must match the predictions used for the paired test.
(2) Ranker table: fill the Qwen2.5-7B cells for BGE-large and reranker + DedupPack (already in the grid)."""
for fn, zh in [("main.tex", False), ("main_zh.tex", True)]:
    t = open(fn, encoding="utf-8").read()
    reps = [("& 38.69 & 42.55 & 45.38 &", "& 38.69 & 42.48 & 45.38 &"),
            (r"BGE-large & \spk{} & -- & 43.98", r"BGE-large & \spk{} & 42.28 & 43.98"),
            ((r"BGE + 重排序器 & \spk{} & -- & 44.12" if zh else r"BGE + reranker & \spk{} & -- & 44.12"),
             (r"BGE + 重排序器 & \spk{} & 42.79 & 44.12" if zh else r"BGE + reranker & \spk{} & 42.79 & 44.12"))]
    for a, b in reps:
        assert t.count(a) == 1, (fn, a)
        t = t.replace(a, b)
    open(fn, "w", encoding="utf-8", newline="").write(t)
    print(fn, "ok")
