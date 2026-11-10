"""Answer-coverage proxy for a packer on a set of instances (multiprocess, CPU only).
usage: python evalcov.py PACKER [TASKS] [LIMIT]   e.g. python evalcov.py bp_main hotpotqa,2wikimqa,musique 200
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import localpack as L
from localpack import norm

TOK = L.TOKDIR
_tok = None


def _init():
    global _tok
    _tok = L.HFLike(TOK)


def _one(args):
    name, s, kw = args
    import packers
    text = getattr(packers, name)(s, _tok, **{k: float(v) if "." in v else int(v) for k, v in kw.items()})
    cov = any(norm(a) and norm(a) in norm(text) for a in s["answers"])
    return s["task"], s["sample_id"], cov, len(_tok(text).input_ids), text, s["question"], s["answers"], s.get("all_classes")


if __name__ == "__main__":
    name = sys.argv[1]
    kw = dict(a.split("=") for a in sys.argv[4:])
    tasks = sys.argv[2].split(",") if len(sys.argv) > 2 else ["hotpotqa", "2wikimqa", "musique"]
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 10**9
    samples = L.load_dev() if tasks == ["dev"] else L.load_inf() if set(tasks) <= {"en_qa", "en_mc"} else L.load_extra(tasks) if set(tasks) <= set(L.EXTRA) else [s for s in L.load_samples() if s["task"] in tasks]
    tasks = sorted({s["task"] for s in samples})
    per = {}
    samples = [s for s in samples if per.setdefault(s["task"], []).append(1) or len(per[s["task"]]) <= limit]
    with Pool(8, initializer=_init) as p:
        res = p.map(_one, [(name, s, kw) for s in samples], chunksize=4)
    out = Path("out"); out.mkdir(exist_ok=True)
    with (out / f"{name}{'_' + '_'.join(sys.argv[4:]) if len(sys.argv) > 4 else ''}_{'-'.join(tasks)}.jsonl").open("w", encoding="utf-8") as f:
        for t, sid, c, n, text, q, ans, ac in res:
            f.write(json.dumps({"task": t, "sample_id": sid, "cov": c, "ntok": n, "packed_text": text,
                                "question": q, "answers": ans}) + "\n")
    for t in tasks:
        r = [x for x in res if x[0] == t]
        print(f"{name} {t}: cov={100*sum(x[2] for x in r)/len(r):.1f} maxtok={max(x[3] for x in r)} n={len(r)}")
