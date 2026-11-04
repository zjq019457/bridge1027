"""One-at-a-time hyperparameter sensitivity of the main configuration (paired vs. the frozen default).
usage: python acl_sens.py RUNS_DIR
"""
import sys
from collections import defaultdict
from pathlib import Path

from acl_robust import load
from acl_main import boot

R = Path(sys.argv[1])
DEF = {"qwen": "qwen/default", "llama": "llama/default/qa"}
CFG = ["alpha025", "alpha075", "csup025", "csup100", "lam030", "lam070", "c_div0", "c_jac0", "neighbor16", "neighbor64"]
for m in ("qwen", "llama"):
    d = load(DEF[m])
    for c in CFG:
        p = R / m / c / ("qa" if m == "llama" else "") / "raw_predictions.jsonl"
        if not p.exists():
            continue
        x = load(str(p.parent.relative_to(R)))
        by = defaultdict(list)
        for k in d:
            if k in x:
                by[k[0]].append(x[k][0] - d[k][0])
        mac = lambda z: sum(sum(v[0] for k, v in z.items() if k[0] == t) / sum(1 for k in z if k[0] == t) for t in by) / len(by)  # noqa: E731
        pt, lo, hi, pv = boot(list(by.values()))
        print(f"{m:5s} {c:10s} n={sum(map(len, by.values()))} macro={mac(x):.2f} delta={pt:+.2f} [{lo:+.2f}, {hi:+.2f}] p={pv:.3f}")
