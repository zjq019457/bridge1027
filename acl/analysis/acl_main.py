"""Paired tests with the main BridgePack configuration (lexical bridge selection, model-tokenizer
admission, neighbor repair, document-order emission) as A. Same bootstrap as acl_tables.py.
usage: python acl_main.py RUNS_DIR
"""
import sys
from pathlib import Path

import numpy as np

import acl_tables as T  # noqa: E402  (reuses paths/loaders; its __main__ part is skipped below)

R = Path(sys.argv[1])
MAIN = {"qwen": "qwen/default", "llama": "llama/default/qa"}
EXTRA = {"nn": {"qwen": "qwen/neighbor0", "llama": "llama/neighbor0/qa"},
         "csup0": {"qwen": "qwen/c_sup0", "llama": "llama/c_sup0/qa"}}


def load(m, k):
    if k == "main":
        p = R / MAIN[m]
    elif k in EXTRA:
        p = R / EXTRA[k][m]
    else:
        return T.load(m, k)
    import json
    p = p / "raw_predictions.jsonl"
    return {(r["task"], str(r["sample_id"])): (100 * r["score"] if r["score"] <= 1 else r["score"], None)
            for r in map(json.loads, p.open(encoding="utf-8"))}


def boot(strata, B=10000):
    pt = np.mean([np.mean(v) for v in strata])
    rng = np.random.default_rng(20260701)
    bs = np.mean([np.asarray(v)[rng.integers(0, len(v), (B, len(v)))].mean(1) for v in strata], 0)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    p = min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean()))
    return pt, lo, hi, p


def diff(a, b, models, tasks):
    strata = []
    for m in models:
        da, db = load(m, a), load(m, b)
        for t in tasks:
            strata.append([da[k][0] - db[k][0] for k in da if k[0] == t and k in db])
    return boot(strata)


if __name__ == "__main__":
    for m in ("qwen", "llama"):
        tm = T.task_means(load(m, "main"))
        print(f"main {m} macro={sum(tm.values())/6:.2f} MH={sum(tm[t] for t in T.MH)/3:.2f} "
              + " ".join(f"{t[:6]}={tm[t]:.2f}" for t in T.TASKS))
    CMP = [("main", x) for x in ("span", "bm25", "tfidf", "bge", "head", "middle", "sc", "lingua2", "longlingua", "full",
                                  "nn", "csup0", "lbp", "bp")] + [("bge_recall", "main"), ("bge_recall", "bge")]
    for a, b in CMP:
        for ms, tag in ((("qwen",), "qwen"), (("llama",), "llama"), (("qwen", "llama"), "2M")):
            for tasks, tt in ((T.TASKS, "all"), (T.MH, "MH")):
                r = diff(a, b, ms, tasks)
                print(f"{a:10s} - {b:10s} {tag:5s} {tt:3s} {r[0]:+.2f} [{r[1]:+.2f}, {r[2]:+.2f}] p={r[3]:.4f} {'*' if r[1] > 0 else ''}")
