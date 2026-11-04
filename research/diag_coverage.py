"""Diagnostic: is a method's packed context missing the answer (recall) or holding it
without the model using it? Answer coverage = some gold answer, normalised as in the
LongBench scorer, appears verbatim in the packed context.
usage: python diag_coverage.py TRACES_DIR RUNS_DIR
"""
import json
import re
import string
import sys
from collections import defaultdict
from pathlib import Path

TR, RU = Path(sys.argv[1]), Path(sys.argv[2])
TASKS = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", "narrativeqa"]
MH = {"hotpotqa", "2wikimqa", "musique"}


def norm(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def jl(p):
    out = []
    for line in p.open(encoding="utf-8"):
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def scores(qa):
    return {(r["task"], str(r["sample_id"])): (100 * r["score"] if r["score"] <= 1 else r["score"])
            for r in jl(qa / "raw_predictions.jsonl")}


# (trace file or None, prediction dir); traces without packed text fall back to packed_context in predictions
METHODS = {
    "bp_main": (None, "qwen/default"),
    "bp_main_norepair": (None, "qwen/neighbor0"),
    "bge_chunk": ("rv5/qwen_bge.jsonl", "rv5/qwen_bge_qa"),
    "tfidf": ("rv5/qwen_tfidf.jsonl", "rv5/qwen_tfidf_qa"),
    "lbp_path": ("a100/qwen_lexical_bridgepack_path_order_1024.jsonl", "a100/qwen_lexical_bridgepack_path_order_1024_qa"),
    "bge_recall_bp": ("a100/qwen_bge_recall_bp_1024.jsonl", "a100/qwen_bge_recall_bp_1024_qa"),
    "lingua2": ("rv5/qwen_lingua2.jsonl", "rv5/qwen_lingua2_qa"),
    "full": ("rv5/qwen_full.jsonl", "rv5/qwen_full_qa"),
}
rows = {}
for name, (tr, qa) in METHODS.items():
    sc = scores(RU / qa)
    if tr:
        recs = jl(TR / tr)
        text = {(r["task"], str(r["sample_id"])): r.get("packed_text") or r.get("generation_prompt", "") for r in recs}
        ans = {(r["task"], str(r["sample_id"])): r.get("answers") or [] for r in recs}
    else:
        recs = jl(RU / qa / "raw_predictions.jsonl")
        text = {(r["task"], str(r["sample_id"])): r.get("packed_context", "") for r in recs}
        ans = {(r["task"], str(r["sample_id"])): r.get("answers") or [] for r in recs}
    cov = {k: any(norm(a) and norm(a) in norm(text[k]) for a in ans[k]) for k in text if ans.get(k)}
    rows[name] = (cov, sc)

print("Qwen-1024. cov = gold answer appears verbatim in packed context (short-answer tasks only)")
SHORT = ["multifieldqa_en", "hotpotqa", "2wikimqa", "musique"]
print(f"{'method':18s} " + " ".join(f"{t[:8]:>9s}" for t in SHORT) + "   F1|cov   F1|miss")
for name, (cov, sc) in rows.items():
    by = defaultdict(list)
    f_cov, f_miss = [], []
    for k, c in cov.items():
        if k[0] in SHORT and k in sc:
            by[k[0]].append(c)
            (f_cov if c else f_miss).append(sc[k])
    print(f"{name:18s} " + " ".join(f"{100*sum(by[t])/len(by[t]):9.1f}" for t in SHORT)
          + f"   {sum(f_cov)/max(1,len(f_cov)):6.1f}   {sum(f_miss)/max(1,len(f_miss)):6.1f}")

# head-to-head decomposition: BP main vs BGE chunks
a, b = rows["bp_main"], rows["bge_chunk"]
cells = defaultdict(list)
for k in a[0]:
    if k[0] in SHORT and k in b[0] and k in a[1] and k in b[1]:
        cells[(a[0][k], b[0][k])].append(a[1][k] - b[1][k])
print("\nBP_main vs BGE (short-answer tasks): (BP covers, BGE covers) -> n, mean F1 diff, sum")
for c, v in sorted(cells.items()):
    print(c, len(v), round(sum(v) / len(v), 2), round(sum(v), 1))
