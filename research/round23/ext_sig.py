"""EXIT, CPC and CompAct at 1,024 tokens on Qwen-7B, Llama-8B and Qwen-14B: macro F1, and DedupPack variants minus
each method (task-stratified paired bootstrap, 10,000 resamples, seed 20260701), per model and pooled."""
import json, glob, re
from collections import defaultdict
import numpy as np
def load(p): return {(r["task"], r["sample_id"]): r["score"] for r in map(json.loads, open(p))}
def macro(d):
    by = defaultdict(list)
    for (t, _), v in d.items(): by[t].append(v)
    return sum(sum(v)/len(v) for v in by.values())/len(by)
M = ["q7", "llama", "q14"]
DIRS = {"q7": ["p23q7", "p23c7"], "llama": ["p23l"], "q14": ["p23q14", "p23c14"]}
EXT = {}
for m in M:
    for meth in ["exit_ext", "cpc_ext", "compact_ext"]:
        fs = [f for d in DIRS[m] for f in glob.glob(f"{d}/preds/{meth}_budget=1024_*.preds.jsonl")]
        if fs:
            EXT[meth, m] = load(fs[0])
files = [f for f in glob.glob("**/*budget=1024*.preds.jsonl", recursive=True) if "t07s" not in f]
cache = {}
def find(stem, wants):
    pat = re.compile(r"(^|/)" + re.escape(stem) + r"_budget=1024(_2wikimqa\S*)?\.preds\.jsonl$")
    for f in files:
        if pat.search(f):
            if f not in cache:
                try: cache[f] = load(f)
                except Exception: cache[f] = {}
            if len(cache[f]) == 1150 and any(abs(macro(cache[f]) - w) < 0.006 for w in wants): return cache[f]
DP = {"BGE-small+DP": ("lex_rep_seed_mode=1_sent=1", [(40.76,), (41.67,), (45.18,)]),
      "Hybrid+DP": ("lex_rep_seed_mode=2_sent=1", [(40.64,), (40.98,), (44.32,)]),
      "BGE-large+DP": ("bgel_lex_rep_seed_mode=1_sent=1", [(42.28,), (43.98,), (46.23,)]),
      "Reranker+DP": ("rr_lex_rep_seed_mode=1_sent=1", [(42.79,), (44.13, 44.12), (46.43, 46.45)])}
keys = sorted(next(iter(EXT.values()))); tasks = sorted({k[0] for k in keys})
ix = {t: np.array([i for i, k in enumerate(keys) if k[0] == t]) for t in tasks}
rng = np.random.default_rng(20260701); dr = {t: rng.integers(0, len(v), size=(10000, len(v))) for t, v in ix.items()}
def stat(D):
    pt = np.mean([D[v].mean() for v in ix.values()]); bs = np.mean([D[v][dr[t]].mean(1) for t, v in ix.items()], axis=0)
    lo, hi = np.percentile(bs, [2.5, 97.5]); p = 2 * min((bs <= 0).mean(), (bs >= 0).mean())
    return [round(pt, 2), round(lo, 2), round(hi, 2), round(max(p, 1e-4), 4)]
out = {"f1": {f"{a}/{m}": round(macro(d), 2) for (a, m), d in EXT.items()}}
print("macro F1:", out["f1"])
for meth in ["exit_ext", "cpc_ext", "compact_ext"]:
    for name, (stem, vals) in DP.items():
        diffs = {}
        for m, v in zip(M, vals):
            if (meth, m) not in EXT: continue
            d = find(stem, v)
            if d is None: print("MISSING", name, m); continue
            diffs[m] = np.array([d[k] - EXT[meth, m][k] for k in keys])
        r = {m: stat(x) for m, x in diffs.items()}
        if len(diffs) == 3: r["pooled"] = stat(np.mean(list(diffs.values()), axis=0))
        out[f"{name} - {meth}"] = r; print(f"{name} - {meth}:", r, flush=True)
json.dump(out, open("logs/ext_sig.json", "w"), indent=1)
