import json, numpy as np
from collections import defaultdict
exec(open("paper_tables2.py").read().split("M = {")[0])  # load(), macro(), boot()
LB = "_2wikimqa-hotpotqa-multifieldqa_en-musique-narrativeqa-qasper"
def F(m, cfg, b=1024): return f"test_preds2/run4_{m}/final_abl_seed_mode=1_{cfg}_budget={b}{LB}.preds.jsonl"
full = {"qwen": "test_preds/qwen14/lex_rep_seed_mode=1_sent=1_budget=1024" + LB + ".preds.jsonl",
        "llama": "test_preds/llama/lex_rep_seed_mode=1_sent=1_budget=1024" + LB + ".preds.jsonl"}
steps = [("BGE chunks (re-impl.)", "sent=0_merge=0_docorder=0_backfill=1"), ("+ merge", "sent=0_merge=1_docorder=0_backfill=1"),
         ("+ document order", "sent=0_merge=1_docorder=1_backfill=1"), ("+ sentence repair (= full)", None),
         ("full - backfill", "sent=1_merge=1_docorder=1_backfill=0"), ("full - merge", "sent=1_merge=0_docorder=1_backfill=1")]
D = {m: {n: load(full[m] if c is None else F(m, c)) for n, c in steps} for m in ("qwen", "llama")}
for n, _ in steps:
    print(f"{n:28s}", " ".join(f"{m}={macro(D[m][n])[0]:.2f}" for m in D))
names = [n for n, _ in steps]
for a, b in [(1, 0), (2, 1), (3, 2), (3, 0), (3, 4), (3, 5), (1, 3)]:
    print(f"{names[a]} - {names[b]}:", " | ".join(boot([(D[m][names[a]], D[m][names[b]])]) for m in D), "|| pooled", boot([(D[m][names[a]], D[m][names[b]]) for m in D]))
print("== merge only at other budgets")
for b in (512, 2048):
    X = {m: (load(F(m, "sent=0_merge=1_docorder=0_backfill=1", b)), load(F(m, "sent=0_merge=0_docorder=0_backfill=1", b))) for m in D}
    print(b, " ".join(f"{m}: base={macro(X[m][1])[0]:.2f} merge={macro(X[m][0])[0]:.2f}" for m in X), "|| pooled", boot([X[m] for m in X]))
