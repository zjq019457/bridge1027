"""Paired task-stratified bootstrap (10k, seed 20260701) for the run7 results -> sig7.txt"""
import json, glob, numpy as np
from pathlib import Path
H = Path(__file__).parent / "test_preds2"


def load(pat):
    f = glob.glob(str(H / pat)); assert len(f) == 1, (pat, f)
    by = {}
    for l in open(f[0], encoding="utf-8"):
        r = json.loads(l); by.setdefault(r["task"], {})[r["sample_id"]] = r["score"]
    return by


def boot(a, b, n=10000):
    rng = np.random.default_rng(20260701)
    tasks = sorted(a); D = [np.array([a[t][k] - b[t][k] for k in sorted(a[t])]) for t in tasks]
    obs = np.mean([d.mean() for d in D])
    bs = np.mean([d[rng.integers(0, len(d), (n, len(d)))].mean(1) for d in D], axis=0)
    p = 2 * min((bs <= 0).mean(), (bs >= 0).mean())
    return f"{obs:+.2f} [{np.percentile(bs, 2.5):.2f}, {np.percentile(bs, 97.5):.2f}] p={max(p, 1/n):.4f}"


def macro(a): return np.mean([np.mean(list(v.values())) for v in a.values()])


out = []
for b in (512, 1024, 2048):
    dp = load(f"run7_q32/lex_rep_seed_mode=2_sent=1_budget={b}_*"); ch = load(f"run7_q32/bge_chunk_budget={b}_*")
    out.append(f"q32 {b}: dedup={macro(dp):.2f} chunk={macro(ch):.2f}  diff {boot(dp, ch)}")
full = glob.glob(str(H / "run7_q32/fullctx*"))
if full:
    fc = load("run7_q32/fullctx*")
    for b in (1024, 2048):
        dp = load(f"run7_q32/lex_rep_seed_mode=2_sent=1_budget={b}_*")
        out.append(f"q32 {b} dedup - full({macro(fc):.2f}): {boot(dp, fc)}  retain {100*macro(dp)/macro(fc):.1f}%")
base = load("run4_llama/final_abl_seed_mode=1_sent=0_merge=0_docorder=0_backfill=1_budget=1024_*")
merge = load("run4_llama/final_abl_seed_mode=1_sent=0_merge=1_docorder=0_backfill=1_budget=1024_*")
for name in ("dedup_skip", "mmr_pack"):
    x = load(f"run7_llama/{name}_*")
    out.append(f"llama {name}={macro(x):.2f} vs chunk({macro(base):.2f}): {boot(x, base)} | merge({macro(merge):.2f}) - {name}: {boot(merge, x)}")
open(Path(__file__).parent / "sig7.txt", "w").write("\n".join(out) + "\n"); print("\n".join(out))
