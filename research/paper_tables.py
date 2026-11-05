"""All numbers for the rewritten paper, from test_preds/{qwen,llama,qwen14}."""
import json, numpy as np
from collections import defaultdict
S = "_2wikimqa-hotpotqa-multifieldqa_en-musique-narrativeqa-qasper.preds.jsonl"
M = ["qwen", "llama", "qwen14"]
TASKS = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", "narrativeqa"]
def f(m, c): return f"test_preds/{m}/{c}{S}" if not c.startswith(("lingua2", "fullctx")) else f"test_preds/{m}/{c}.preds.jsonl"
def load(m, c): return {(r["task"], r["sample_id"]): r["score"] for r in map(json.loads, open(f(m, c), encoding="utf-8"))}
def macro(d):
    by = defaultdict(list)
    for (t, _), v in d.items(): by[t].append(v)
    return {t: 100*np.mean(by[t]) if max(max(x) for x in by.values()) <= 1.0 else np.mean(by[t]) for t in TASKS}
def mac(m, c):
    r = macro(load(m, c)); return np.mean(list(r.values())), r
def boot(pairs):  # list of (A dict, B dict) -> pooled stratified bootstrap over model-task strata
    strata = []
    for A, B in pairs:
        by = defaultdict(list)
        for k in A:
            if k in B: by[k[0]].append(A[k] - B[k])
        strata += [np.asarray(v) for v in by.values()]
    rng = np.random.default_rng(20260701)
    bs = np.mean([v[rng.integers(0, len(v), (10000, len(v)))].mean(1) for v in strata], 0)
    pt = np.mean([v.mean() for v in strata]); lo, hi = np.percentile(bs, [2.5, 97.5])
    p = min(1.0, 2*min((bs <= 0).mean(), (bs >= 0).mean()))
    return pt, lo, hi, p
H = lambda b: f"lex_rep_seed_mode=2_sent=1_budget={b}"
G = lambda b: f"lex_rep_seed_mode=1_sent=1_budget={b}"
BG = lambda b: f"bge_chunk_budget={b}"
BP = lambda b: f"bp_main_budget={b}"
scale = 1.0
print("== macro")
for m in M:
    for c in [BP(b) for b in (512,1024,2048)]+[BG(b) for b in (512,1024,2048)]+[H(b) for b in (512,1024,2048)]+[G(b) for b in (512,1024,2048)]+["lingua2_budget=1024","fullctx","lex_rep_seed_mode=0_sent=1_budget=1024"]+[f"final_abl_{a}_budget=1024" for a in ("sent=0","merge=0","docorder=0","backfill=0")]:
        v, r = mac(m, c); print(m, c, f"{v:.2f}", " ".join(f"{r[t]:.2f}" for t in TASKS), "MH", f"{np.mean([r['hotpotqa'],r['2wikimqa'],r['musique']]):.2f}")
print("== pooled over 3 models")
for b in (512,1024,2048):
    for name, X in (("hyb", H), ("bge", G)):
        for bn, Y in (("BGEchunk", BG), ("BP", BP)):
            print(b, name, "-", bn, "%+.2f [%.2f, %.2f] p=%.4f" % boot([(load(m, X(b)), load(m, Y(b))) for m in M]))
print("== per model")
for m in M:
    for b in (512,1024,2048):
        for name, X in (("hyb", H), ("bge", G)):
            for bn, Y in (("BGEchunk", BG), ("BP", BP)):
                print(m, b, name, "-", bn, "%+.2f [%.2f, %.2f] p=%.4f" % boot([(load(m, X(b)), load(m, Y(b)))]))
    for c, n in (("lingua2_budget=1024", "lingua"), ("fullctx", "full")):
        print(m, "hyb1024 -", n, "%+.2f [%.2f, %.2f] p=%.4f" % boot([(load(m, H(1024)), load(m, c))]))
        print(m, "hyb2048 -", n, "%+.2f [%.2f, %.2f] p=%.4f" % boot([(load(m, H(2048)), load(m, c))]))
    print(m, "lex-seed1024 - BGEchunk", "%+.2f [%.2f, %.2f] p=%.4f" % boot([(load(m, "lex_rep_seed_mode=0_sent=1_budget=1024"), load(m, BG(1024)))]))
    print(m, "lex-seed1024 - BP", "%+.2f [%.2f, %.2f] p=%.4f" % boot([(load(m, "lex_rep_seed_mode=0_sent=1_budget=1024"), load(m, BP(1024)))]))
print("== pooled lexical seeds / ablated variants vs BGE chunks (1024)")
print("lex", "%+.2f [%.2f, %.2f] p=%.4f" % boot([(load(m, "lex_rep_seed_mode=0_sent=1_budget=1024"), load(m, BG(1024))) for m in M]))
print("lex-BP", "%+.2f [%.2f, %.2f] p=%.4f" % boot([(load(m, "lex_rep_seed_mode=0_sent=1_budget=1024"), load(m, BP(1024))) for m in M]))
for a in ("sent=0","merge=0","docorder=0","backfill=0"):
    print(a, "vs BGEchunk", "%+.2f [%.2f, %.2f] p=%.4f" % boot([(load(m, f"final_abl_{a}_budget=1024"), load(m, BG(1024))) for m in M]),
          "| full - abl", "%+.2f [%.2f, %.2f] p=%.4f" % boot([(load(m, H(1024)), load(m, f"final_abl_{a}_budget=1024")) for m in M]))
print("hyb-lingua pooled", "%+.2f [%.2f, %.2f] p=%.4f" % boot([(load(m, H(1024)), load(m, "lingua2_budget=1024")) for m in M]))
print("hyb-bge seeds pooled 1024", "%+.2f [%.2f, %.2f] p=%.4f" % boot([(load(m, H(1024)), load(m, G(1024))) for m in M]))
