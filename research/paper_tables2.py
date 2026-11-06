import json, numpy as np
from collections import defaultdict
LB = "_2wikimqa-hotpotqa-multifieldqa_en-musique-narrativeqa-qasper"
def load(p): return {(r["task"], r["sample_id"]): r["score"] for r in map(json.loads, open(p, encoding="utf-8"))}
def macro(d):
    by = defaultdict(list)
    for (t, _), v in d.items(): by[t].append(v)
    return np.mean([np.mean(v) for v in by.values()]), {t: np.mean(v) for t, v in by.items()}
def boot(pairs):
    strata = []
    for A, B in pairs:
        by = defaultdict(list)
        for k in A:
            if k in B: by[k[0]].append(A[k] - B[k])
        strata += [np.asarray(v) for v in by.values()]
    rng = np.random.default_rng(20260701)
    bs = np.mean([v[rng.integers(0, len(v), (10000, len(v)))].mean(1) for v in strata], 0)
    pt = np.mean([v.mean() for v in strata]); lo, hi = np.percentile(bs, [2.5, 97.5])
    return "%+.2f [%.2f, %.2f] p=%.4f" % (pt, lo, hi, min(1.0, 2*min((bs <= 0).mean(), (bs >= 0).mean())))
M = {"qwen14": ("run2_qwen", "run3_qwen", "test_preds/qwen14"), "llama": ("run2_llama", "run3_llama", "test_preds/llama")}
F = {}
for m, (r2, r3, r1) in M.items():
    F[m] = {
     "SP-hyb": f"{r1}/lex_rep_seed_mode=2_sent=1_budget=1024{LB}.preds.jsonl",
     "SP-bge": f"{r1}/lex_rep_seed_mode=1_sent=1_budget=1024{LB}.preds.jsonl",
     "BGE chunks": f"{r1}/bge_chunk_budget=1024{LB}.preds.jsonl",
     "BGE-large chunks": f"test_preds2/{r2}/bge_chunk_budget=1024{LB}.preds.jsonl",
     "SP-bgelarge": f"test_preds2/{r2}/lex_rep_seed_mode=1_sent=1_budget=1024{LB}.preds.jsonl",
     "Sentence chunks": f"test_preds2/{r2}/sentchunk_budget=1024.preds.jsonl",
     "Rerank chunks": f"test_preds2/{r3}/bge_chunk_budget=1024{LB}.preds.jsonl",
     "SP-rerank": f"test_preds2/{r3}/lex_rep_seed_mode=1_sent=1_budget=1024{LB}.preds.jsonl",
     "LI window": f"test_preds2/{r3}/li_window_budget=1024.preds.jsonl",
     "LI automerge": f"test_preds2/{r3}/li_automerge_budget=1024.preds.jsonl",
     "Provence": f"test_preds2/{r3}/provence_budget=1024.preds.jsonl",
     "LongLLMLingua": f"test_preds2/{r3}/longlingua_budget=1024.preds.jsonl",
     "LLMLingua-2": f"{r1}/lingua2_budget=1024.preds.jsonl",
     "Full": f"{r1}/fullctx.preds.jsonl"}
D = {m: {k: load(v) for k, v in F[m].items()} for m in M}
print("== LongBench 1024 macro (n)")
for k in F["llama"]:
    print(f"{k:18s}", " ".join(f"{m}={macro(D[m][k])[0]:.2f}(n={len(D[m][k])})" for m in M))
print("== SP-hyb vs baselines: per model | pooled")
for b in ["BGE chunks", "BGE-large chunks", "Sentence chunks", "Rerank chunks", "LI window", "LI automerge", "Provence", "LongLLMLingua"]:
    print(f"SP-hyb - {b:16s}", " | ".join(boot([(D[m]["SP-hyb"], D[m][b])]) for m in M), " || pooled", boot([(D[m]["SP-hyb"], D[m][b]) for m in M]))
print("== same ranker, packer swap")
for a, b in [("SP-bge", "BGE chunks"), ("SP-bgelarge", "BGE-large chunks"), ("SP-rerank", "Rerank chunks"), ("SP-bge", "Sentence chunks"), ("SP-rerank", "Provence")]:
    print(f"{a} - {b}", " | ".join(boot([(D[m][a], D[m][b])]) for m in M), " || pooled", boot([(D[m][a], D[m][b]) for m in M]))
print("== InfiniteBench")
I = lambda r, c: f"test_preds2/{r}/{c}.preds.jsonl"
for m, (r2, _, _) in M.items():
    for bud in (1024, 2048):
        C = {"Pair": f"bp_main_budget={bud}_en_mc-en_qa", "BGE chunks": f"bge_chunk_budget={bud}_en_mc-en_qa", "Sentence chunks": f"sentchunk_inf_budget={bud}",
             "SP-hyb": f"lex_rep_seed_mode=2_sent=1_budget={bud}_en_mc-en_qa", "SP-bge": f"lex_rep_seed_mode=1_sent=1_budget={bud}_en_mc-en_qa"}
        L_ = {k: load(I(r2, v)) for k, v in C.items()}
        print(m, bud, " ".join(f"{k}={macro(v)[0]:.2f}({' '.join(f'{t}:{x:.1f}' for t,x in sorted(macro(v)[1].items()))})" for k, v in L_.items()))
        for b in ("BGE chunks", "Sentence chunks", "Pair"):
            print("   SP-hyb -", b, boot([(L_["SP-hyb"], L_[b])]), "| SP-bge -", b, boot([(L_["SP-bge"], L_[b])]))
    try: print(m, "fullinf", f"{macro(load(I(r2, 'fullinf')))[0]:.2f}")
    except FileNotFoundError: print(m, "fullinf missing")
print("== pooled InfiniteBench (2 models x 2 budgets)")
pairs = lambda a, b: [(load(I(M[m][0], a.format(bud=bud))), load(I(M[m][0], b.format(bud=bud)))) for m in M for bud in (1024, 2048)]
for b in ("bge_chunk_budget={bud}_en_mc-en_qa", "sentchunk_inf_budget={bud}", "bp_main_budget={bud}_en_mc-en_qa"):
    print("SP-hyb -", b.split("_")[0], boot(pairs("lex_rep_seed_mode=2_sent=1_budget={bud}_en_mc-en_qa", b)), "| SP-bge -", b.split("_")[0], boot(pairs("lex_rep_seed_mode=1_sent=1_budget={bud}_en_mc-en_qa", b)))
