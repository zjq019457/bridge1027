"""Paired task-stratified bootstrap between two prediction files from colab_gen.py
(10,000 resamples, seed 20260701). usage: python sig_dev.py A.preds.jsonl B.preds.jsonl [...pairs]
"""
import json
import sys
from collections import defaultdict

import numpy as np


def load(p):
    return {(r["task"], r["sample_id"]): r["score"] for r in map(json.loads, open(p, encoding="utf-8"))}


args = sys.argv[1:]
for a, b in zip(args[::2], args[1::2]):
    A, B = load(a), load(b)
    by = defaultdict(list)
    for k in A:
        if k in B:
            by[k[0]].append(A[k] - B[k])
    strata = [np.asarray(v) for v in by.values()]
    rng = np.random.default_rng(20260701)
    bs = np.mean([v[rng.integers(0, len(v), (10000, len(v)))].mean(1) for v in strata], 0)
    pt = np.mean([v.mean() for v in strata])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    p = min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean()))
    short = lambda s: s.split("/")[-1].replace("_2wikimqa_e-hotpotqa_e.preds.jsonl", "")  # noqa: E731
    print(f"{short(a)} - {short(b)}: {pt:+.2f} [{lo:+.2f}, {hi:+.2f}] p={p:.4f}")
