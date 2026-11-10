"""Run BridgePack packers locally on CPU (no transformers needed): a thin adapter makes a
`tokenizers.Tokenizer` look like the HF tokenizer the repo code expects.
"""
import json
import sys
from pathlib import Path

import os

# BP_ROOT holds the bridge1027 checkout (local backup on D:, /content/bp on Colab)
BK = Path(os.environ.get("BP_ROOT", "D:/BridgePack_backup_20261027"))
REPO = BK / "bridge1027"
SRC = REPO / "server/source/SlimInfer_snapshot"
DATA = REPO / "research/data/longbench"
GOLD = REPO / "server/golden"
TOKDIR = Path(os.environ["BP_TOK"]) if os.environ.get("BP_TOK") else REPO / "research/data/tok/qwen"
sys.path.insert(0, str(SRC))

from tokenizers import Tokenizer  # noqa: E402

TASKS = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", "narrativeqa"]


class _Enc:
    def __init__(self, ids):
        self.input_ids = ids


class HFLike:
    def __init__(self, path):
        self.tok = Tokenizer.from_file(str(Path(path) / "tokenizer.json"))

    def __call__(self, text, add_special_tokens=False, **_):
        return _Enc(self.tok.encode(text, add_special_tokens=False).ids)

    def encode(self, text, add_special_tokens=False, **_):
        return self.tok.encode(text, add_special_tokens=False).ids

    def decode(self, ids, skip_special_tokens=True, **_):
        return self.tok.decode(ids, skip_special_tokens=skip_special_tokens)


def load_samples():
    """The 1,150 test instances, in the order and with the ids of the golden reference."""
    import gzip
    gold = [json.loads(x) for x in gzip.open(GOLD / "GOLDEN_REFERENCE_V1.jsonl.gz", "rt", encoding="utf-8") if x.strip()]
    src = {t: {str(r["_id"]): r for r in map(json.loads, (DATA / f"{t}.jsonl").open(encoding="utf-8"))} for t in TASKS}
    out = []
    for g in gold:
        d = src[g["task"]][str(g["sample_id"])]
        out.append({"task": g["task"], "sample_id": str(g["sample_id"]), "question": d["input"],
                    "context": d["context"], "answers": d.get("answers", [])})
    return out


def norm(s):
    """LongBench answer normalisation (lowercase, strip punctuation and articles)."""
    import re
    import string
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


DEV = REPO / "research/data/dev"


def load_dev(tasks=("hotpotqa_e", "2wikimqa_e")):
    """Development instances: LongBench-E with every test question removed (no test tuning)."""
    out = []
    for t in tasks:
        for i, r in enumerate(map(json.loads, (DEV / f"dev_{t}.jsonl").open(encoding="utf-8"))):
            out.append({"task": t, "sample_id": str(r.get("_id", i)), "question": r["input"],
                        "context": r["context"], "answers": r.get("answers", [])})
    return out


INF = REPO / "research/data/infbench"


def load_inf(tasks=("en_qa", "en_mc")):
    """InfiniteBench En.QA / En.MC (English novels, ~150K+ tokens). For En.MC the options are
    part of the question text and the gold answer is the option letter."""
    out = []
    for t in tasks:
        for i, r in enumerate(map(json.loads, (INF / f"{t.replace('_', '.')}.jsonl").open(encoding="utf-8"))):
            q, ans = r["input"], r["answer"] if isinstance(r["answer"], list) else [r["answer"]]
            if t == "en_mc":
                opts = r["options"]
                q = (f"Question: {q}\n\nOnly one of the following options is correct, tell me the answer using one "
                     "single letter (A, B, C, or D). Don't say anything else.\n"
                     + "\n".join(f"{'ABCD'[k]}. {o}" for k, o in enumerate(opts)))
                ans = ["ABCD"[opts.index(ans[0])]] if ans[0] in opts else ans
            out.append({"task": t, "sample_id": str(r.get("id", i)), "question": q, "context": r["context"], "answers": ans})
    return out


EXTRA = ("qmsum", "triviaqa", "trec", "samsum", "passage_retrieval_en")


def load_extra(tasks=EXTRA):
    """LongBench query-bearing tasks beyond QA (query-based summarisation, few-shot, retrieval)."""
    out = []
    for t in tasks:
        for i, r in enumerate(map(json.loads, (DATA / f"{t}.jsonl").open(encoding="utf-8"))):
            out.append({"task": t, "sample_id": str(r.get("_id", i)), "question": r["input"], "context": r["context"],
                        "answers": r.get("answers", []), "all_classes": r.get("all_classes")})
    return out
