exec(open("paper_tables2.py").read().split("M = {")[0])
X = "_passage_retrieval_en-qmsum-samsum-trec-triviaqa"
LB = "_2wikimqa-hotpotqa-multifieldqa_en-musique-narrativeqa-qasper"
P = lambda m, c: f"test_preds2/run5_{m}/{c}_budget=1024{X}.preds.jsonl"
C = {"BGE chunks": "bge_chunk", "Pair": "bp_main", "DP-bge": "lex_rep_seed_mode=1_sent=1", "DP-hyb": "lex_rep_seed_mode=2_sent=1"}
M = ("qwen", "llama")
D = {m: {k: load(P(m, v)) for k, v in C.items()} for m in M}
for m in M:
    try: D[m]["Full"] = load(f"test_preds2/run5_{m}/fullextra.preds.jsonl")
    except FileNotFoundError: pass
for k in ["BGE chunks", "Pair", "DP-bge", "DP-hyb", "Full"]:
    print(f"{k:12s}", " ".join(f"{m}={macro(D[m][k])[0]:.2f}" for m in M if k in D[m]))
for a in ("DP-hyb", "DP-bge"):
    for b in ("BGE chunks", "Pair"):
        print(a, "-", b, " | ".join(boot([(D[m][a], D[m][b])]) for m in M), "|| pooled", boot([(D[m][a], D[m][b]) for m in M]))
# per task excluding passage retrieval
def sub(d, drop): return {k: v for k, v in d.items() if k[0] != drop}
for a, b in (("DP-hyb", "BGE chunks"),):
    print("w/o passage_retrieval:", a, "-", b, "|| pooled", boot([(sub(D[m][a], "passage_retrieval_en"), sub(D[m][b], "passage_retrieval_en")) for m in M]))
# 11-task combined (LongBench QA 6 from round1 + 5 extra)
R1 = {"qwen": "test_preds/qwen14", "llama": "test_preds/llama"}
F1 = {"BGE chunks": "bge_chunk_budget=1024", "Pair": "bp_main_budget=1024", "DP-bge": "lex_rep_seed_mode=1_sent=1_budget=1024", "DP-hyb": "lex_rep_seed_mode=2_sent=1_budget=1024"}
A = {m: {k: {**load(f"{R1[m]}/{v}{LB}.preds.jsonl"), **D[m][k]} for k, v in F1.items()} for m in M}
for m in M:
    if "Full" in D[m]: A[m]["Full"] = {**load(f"{R1[m]}/fullctx.preds.jsonl"), **D[m]["Full"]}
print("== 11 tasks")
for k in ["BGE chunks", "Pair", "DP-bge", "DP-hyb", "Full"]:
    print(f"{k:12s}", " ".join(f"{m}={macro(A[m][k])[0]:.2f}" for m in M if k in A[m]))
for a in ("DP-hyb", "DP-bge"):
    for b in ("BGE chunks", "Pair"):
        print(a, "-", b, " | ".join(boot([(A[m][a], A[m][b])]) for m in M), "|| pooled", boot([(A[m][a], A[m][b]) for m in M]))
