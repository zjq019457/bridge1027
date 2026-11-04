"""Sampling robustness (nucleus T=0.7, p=0.9, 3 seeds) and BABILong, BridgePack vs BM25 (Qwen).
usage: python acl_robust.py RUNS_DIR
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

from acl_main import boot

R = Path(sys.argv[1])


def load(d):
    out = {}
    for line in (R / d / "raw_predictions.jsonl").open(encoding="utf-8"):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:  # partially copied line in the backup
            continue
        s = r["score"]
        out[(r["task"], str(r["sample_id"]))] = (100 * s if s <= 1 else s, r.get("input_token_count"))
    return out


def paired(a, b):
    by = defaultdict(list)
    for k in a:
        if k in b:
            by[k[0]].append(a[k][0] - b[k][0])
    return boot(list(by.values()))


if __name__ == "__main__":
    seeds = [(load(f"sampling/qwen_bp_s{i}"), load(f"sampling/qwen_bm25_s{i}")) for i in range(3)]
    avg = lambda ds: {k: (sum(d[k][0] for d in ds) / len(ds), None) for k in ds[0]}  # noqa: E731
    print("sampling seed-avg BP-BM25", paired(avg([s[0] for s in seeds]), avg([s[1] for s in seeds])))
    for i, (a, b) in enumerate(seeds):
        print(f"sampling s{i}", paired(a, b))
    a, b = load("babilong/qwen_bridgepack"), load("babilong/qwen_bm25")
    print("babilong BP-BM25", paired(a, b))
    for n, d in (("bp", a), ("bm25", b)):
        toks = [v[1] for v in d.values() if v[1]]
        by = defaultdict(list)
        for k, v in d.items():
            by[k[0]].append(v[0])
        print(n, "macro", sum(sum(v) / len(v) for v in by.values()) / len(by), "mean_in_tok", sum(toks) / max(1, len(toks)))
