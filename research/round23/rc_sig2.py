"""RECOMP minus same-budget alternatives at 512, 1,024 and 2,048 tokens on four readers: task-stratified paired
bootstrap (10,000 resamples, seed 20260701), per model and pooled. Comparison files are located by macro F1 (Table 3)."""
import json, glob, re
from collections import defaultdict
import numpy as np
def load(p): return {(r["task"], r["sample_id"]): r["score"] for r in map(json.loads, open(p))}
def macro(d):
    by = defaultdict(list)
    for (t, _), v in d.items(): by[t].append(v)
    return sum(sum(v)/len(v) for v in by.values())/len(by)
files = [f for f in glob.glob("**/*.preds.jsonl", recursive=True) if "t07s" not in f]
cache = {}
def find(stem, b, wants):
    pat = re.compile(r"(^|/)" + re.escape(stem) + rf"_budget={b}(_2wikimqa\S*)?\.preds\.jsonl$")
    for f in files:
        if pat.search(f):
            if f not in cache:
                try: cache[f] = load(f)
                except Exception: cache[f] = {}
            if len(cache[f]) == 1150 and any(abs(macro(cache[f]) - w) < 0.006 for w in wants): return cache[f]
M, B = ["q7", "llama", "q14", "q32"], [512, 1024, 2048]
RCD = {("q7", 1024): "p18q7", ("llama", 1024): "p18l", ("q14", 1024): "p18q", ("q32", 1024): "p22q32"}
for m, d in zip(M, ["p22q7", "p22l", "p22q14", "p22q32"]):
    RCD[m, 512] = d; RCD[m, 2048] = d
RCD.update({("q7", 1024): "p18q7", ("llama", 1024): "p18l", ("q14", 1024): "p18q", ("q32", 1024): "p22q32"})
RC = {(m, b): load(glob.glob(f"{RCD[m, b]}/preds/recomp_ext_budget={b}_*.preds.jsonl")[0]) for m in M for b in B}
# Table 3 values (rows x [model][budget]); alternatives with rerun values listed together
T = {
 "BGE-small+DP": ("lex_rep_seed_mode=1_sent=1", [[35.64, 40.76, 43.14], [36.16, 41.67, 43.58], [39.21, 45.18, 48.10], [40.34, 46.28, 49.01]]),
 "Hybrid+DP": ("lex_rep_seed_mode=2_sent=1", [[36.74, 40.64, 43.60], [36.44, 40.98, 42.78], [40.44, 44.32, 47.50], [40.88, 45.07, 49.15]]),
 "BGE-large+DP": ("bgel_lex_rep_seed_mode=1_sent=1", [[38.24, 42.28, 44.32], [38.81, 43.98, 44.65], [41.54, 46.23, 48.61], [41.05, 46.52, 49.25]]),
 "Reranker+DP": ("rr_lex_rep_seed_mode=1_sent=1", [[39.56, 42.79, 44.96], [40.49, (44.13, 44.12), 45.61], [42.85, (46.43, 46.45), 49.54], [44.05, 47.32, 50.47]]),
 "Provence": ("provence", [[40.66, 41.90, 41.78], [39.62, 40.69, 41.36], [42.70, 44.27, 44.36], [43.66, 45.66, 45.68]]),
 "Reranker chunks": ("rr_bge_chunk", [[37.19, 41.59, 43.48], [38.29, (42.22, 42.21), 43.72], [41.18, 44.37, 47.30], [41.74, 45.09, 48.15]]),
}
keys = sorted(RC["q7", 1024]); tasks = sorted({k[0] for k in keys})
ix = {t: np.array([i for i, k in enumerate(keys) if k[0] == t]) for t in tasks}
rng = np.random.default_rng(20260701); dr = {t: rng.integers(0, len(v), size=(10000, len(v))) for t, v in ix.items()}
def stat(D):
    pt = np.mean([D[v].mean() for v in ix.values()]); bs = np.mean([D[v][dr[t]].mean(1) for t, v in ix.items()], axis=0)
    lo, hi = np.percentile(bs, [2.5, 97.5]); return [round(pt, 2), round(lo, 2), round(hi, 2)]
out = {"recomp": {f"{m}/{b}": round(macro(RC[m, b]), 2) for m in M for b in B}}
for name, (stem, vals) in T.items():
    for b_i, b in enumerate(B):
        diffs = {}
        for m_i, m in enumerate(M):
            v = vals[m_i][b_i]; d = find(stem, b, v if isinstance(v, tuple) else (v,))
            if d is None: print("MISSING", name, m, b, v); continue
            diffs[m] = np.array([RC[m, b][k] - d[k] for k in keys])
        r = {m: stat(x) for m, x in diffs.items()}
        if len(diffs) == 4: r["pooled"] = stat(np.mean(list(diffs.values()), axis=0))
        out[f"{name}/{b}"] = r; print(f"RECOMP - {name} @{b}:", r, flush=True)
json.dump(out, open("logs/rc_sig2.json", "w"), indent=1)
