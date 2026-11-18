"""Same-ranker significance for every ranker: DedupPack minus chunk packing, pooled over four models and per
model (task-stratified paired bootstrap, 10,000 resamples, seed 20260701). Prediction files are located by
matching their macro F1 to the values in Table 3. usage: python sig19.py [extra.json]"""
import json, glob, re, sys
from collections import defaultdict
import numpy as np
M = ["q7", "llama", "q14", "q32"]
B = [512, 1024, 2048]
EXP = {  # (stem prefix) -> {model: [512, 1024, 2048]}
    "bge_chunk": dict(q7=[33.05, 38.85, 40.98], llama=[33.27, 38.99, 40.85], q14=[36.27, 42.34, 45.29], q32=[37.52, 42.40, 46.08]),
    "lex_rep_seed_mode=1_sent=1": dict(q7=[35.64, 40.76, 43.14], llama=[36.16, 41.67, 43.58], q14=[39.21, 45.18, 48.10], q32=[40.34, 46.28, 49.01]),
    "bgel_bge_chunk": dict(q7=[34.34, 38.74, 42.55], llama=[35.05, 40.12, 42.49], q14=[38.69, 42.48, 45.38], q32=[39.93, 43.04, 47.14]),
    "bgel_lex_rep_seed_mode=1_sent=1": dict(q7=[38.24, 42.28, 44.32], llama=[38.81, 43.98, 44.65], q14=[41.54, 46.23, 48.61], q32=[41.05, 46.52, 49.25]),
    "rr_bge_chunk": dict(q7=[37.19, 41.59, 43.48], llama=[38.29, 42.21, 43.72], q14=[41.18, 44.37, 47.30], q32=[41.74, 45.09, 48.15]),
    "rr_lex_rep_seed_mode=1_sent=1": dict(q7=[39.56, 42.79, 44.96], llama=[40.49, 44.12, 45.61], q14=[42.85, 46.45, 49.54], q32=[44.05, 47.32, 50.47]),
    "bp_main": dict(q7=[33.88, 39.11, 40.68], llama=[34.05, 37.79, 39.60], q14=[36.73, 41.69, 44.23], q32=[37.51, 42.57, 44.77]),
    "lex_rep_seed_mode=2_sent=1": dict(q7=[36.74, 40.64, 43.60], llama=[36.44, 40.98, 42.78], q14=[40.44, 44.32, 47.50], q32=[40.88, 45.07, 49.15]),
}
if len(sys.argv) > 1:
    EXP.update(json.load(open(sys.argv[1])))
CMP = [("BGE-small", "lex_rep_seed_mode=1_sent=1", "bge_chunk"), ("BGE-large", "bgel_lex_rep_seed_mode=1_sent=1", "bgel_bge_chunk"),
       ("Reranker", "rr_lex_rep_seed_mode=1_sent=1", "rr_bge_chunk"), ("BGE-small vs pair", "lex_rep_seed_mode=1_sent=1", "bp_main"),
       ("Hybrid", "lex_rep_seed_mode=2_sent=1", "hyb_chunk")]


def load(p):
    return {(r["task"], r["sample_id"]): r["score"] for r in map(json.loads, open(p, encoding="utf-8"))}


def macro(d):
    by = defaultdict(list)
    for (t, _), v in d.items(): by[t].append(v)
    return sum(sum(v) / len(v) for v in by.values()) / len(by)


files = [f for f in glob.glob("**/*.preds.jsonl", recursive=True) if "t07s" not in f]
cache = {}
def find(stem, b, want):
    pat = re.compile(r"(^|/)" + re.escape(stem) + rf"_budget={b}(_2wikimqa\S*)?\.preds\.jsonl$")
    for f in files:
        if pat.search(f):
            if f not in cache:
                try: cache[f] = load(f)
                except Exception: cache[f] = {}
            d = cache[f]
            if len(d) == 1150 and abs(macro(d) - want) < 0.006:
                return d
    return None


P = {}
for stem, per in EXP.items():
    for m, vals in per.items():
        for b, v in zip(B, vals):
            d = find(stem, b, v)
            P[stem, m, b] = d
            if d is None: print("MISSING", stem, m, b, v)
rng = np.random.default_rng(20260701)
keys = sorted(next(d for d in P.values() if d))
tasks = sorted({k[0] for k in keys})
idx_by_task = {t: np.array([i for i, k in enumerate(keys) if k[0] == t]) for t in tasks}
R = 10000
draws = {t: rng.integers(0, len(ix), size=(R, len(ix))) for t, ix in idx_by_task.items()}


def stat(D):  # D: per-instance diff vector (already averaged over models)
    point = np.mean([D[ix].mean() for ix in idx_by_task.values()])
    bs = np.mean([D[ix][draws[t]].mean(1) for t, ix in idx_by_task.items()], axis=0)
    lo, hi = np.percentile(bs, [2.5, 97.5]); p = 2 * min((bs <= 0).mean(), (bs >= 0).mean())
    return round(float(point), 2), round(float(lo), 2), round(float(hi), 2), float(p)


out = {}
for name, a, c in CMP:
    for b in B:
        ms = [m for m in M if P.get((a, m, b)) and P.get((c, m, b))]
        if not ms: continue
        diffs = {m: np.array([P[a, m, b][k] - P[c, m, b][k] for k in keys]) for m in ms}
        pooled = stat(np.mean([diffs[m] for m in ms], axis=0))
        per = {m: stat(diffs[m]) for m in ms}
        out[f"{name}|{b}"] = {"models": ms, "pooled": pooled, "per": per}
        print(name, b, ms, "pooled", pooled, {m: v[:3] for m, v in per.items()})
json.dump(out, open("sig19.json", "w"), indent=1)
