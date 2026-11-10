"""Multi-hop subset (hotpotqa, 2wikimqa, musique): DedupPack @2048 vs full context, per model and pooled."""
import json, glob, numpy as np
MH = ("hotpotqa", "2wikimqa", "musique")
M = {"qwen7": "test_preds/qwen", "llama8": "test_preds/llama", "qwen14": "test_preds/qwen14", "qwen32": "test_preds2/run7_q32"}
def load(d, pat):
    f = glob.glob(f"{d}/{pat}"); assert len(f) == 1, (d, pat, f); by = {}
    for l in open(f[0], encoding="utf-8"):
        r = json.loads(l)
        if r["task"] in MH: by.setdefault(r["task"], {})[str(r["sample_id"])] = r["score"]
    return by
def diffs(a, b): return [np.array([a[t][k] - b[t][k] for k in sorted(a[t])]) for t in MH]
def boot(D, n=10000):
    rng = np.random.default_rng(20260701)
    obs = np.mean([d.mean() for d in D]); bs = np.mean([d[rng.integers(0, len(d), (n, len(d)))].mean(1) for d in D], axis=0)
    p = 2 * min((bs <= 0).mean(), (bs >= 0).mean())
    return obs, np.percentile(bs, 2.5), np.percentile(bs, 97.5), max(p, 1 / n)
mac = lambda a: np.mean([np.mean(list(a[t].values())) for t in MH])
allD = []; out = []
for name, d in M.items():
    dp = load(d, "lex_rep_seed_mode=2_sent=1_budget=2048_2wikimqa*"); fc = load(d, "fullctx.preds.jsonl")
    D = diffs(dp, fc); allD.append(D); o, lo, hi, p = boot(D)
    out.append(f"{name}: dedup={mac(dp):.2f} full={mac(fc):.2f} diff {o:+.2f} [{lo:.2f}, {hi:.2f}] p={p:.4f}")
for tag, sel in (("pooled 3 (7B/8B/14B)", allD[:3]), ("pooled 4", allD)):
    D = [np.mean([x[i] for x in sel], axis=0) for i in range(3)]
    o, lo, hi, p = boot(D); out.append(f"{tag}: diff {o:+.2f} [{lo:.2f}, {hi:.2f}] p={p:.4f}")
open("sig_mh.txt", "w").write("\n".join(out) + "\n"); print("\n".join(out))
