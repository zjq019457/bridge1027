"""Paired comparisons for Qwen sampling runs and Llama runs (same bootstrap as summarize_sens.py)."""
import json
import os
import random

W = "/diskE/jiaqizhang/BRIDGEPACK_SENSITIVITY_V1_20261025"
TASKS = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", "narrativeqa"]


def load(p):
    return {(r["task"], str(r["sample_id"])): r["score"] for r in map(json.loads, open(p))}


def summ(d):
    return json.load(open(d + "/macro_summary.json"))


def paired(a, b, B=10000):
    d = {t: [a[k] - b[k] for k in a if k[0] == t and k in b] for t in TASKS}
    pt = sum(sum(v) / len(v) for v in d.values()) / 6
    rng = random.Random(20260701)
    bs = sorted(sum(sum(v[rng.randrange(len(v))] for _ in v) / len(v) for v in d.values()) / 6 for _ in range(B))
    p = 2 * min(sum(x <= 0 for x in bs), sum(x >= 0 for x in bs)) / B
    return f"{pt:+.3f} [{bs[int(.025 * B)]:+.3f}, {bs[int(.975 * B) - 1]:+.3f}] p={min(1, p):.4f}"


print("== Qwen nucleus sampling (top_p 0.9, T 0.7), BP minus BM25")
A, Bm = {}, {}
for s in range(3):
    a, b = f"{W}/runs/sampling/qwen_bp_s{s}", f"{W}/runs/sampling/qwen_bm25_s{s}"
    pa, pb = load(a + "/raw_predictions.jsonl"), load(b + "/raw_predictions.jsonl")
    print(f"seed{s}: BP {summ(a)['macro']:.3f} BM25 {summ(b)['macro']:.3f} paired {paired(pa, pb)}")
    for k, v in pa.items():
        A[k] = A.get(k, 0) + v / 3
    for k, v in pb.items():
        Bm[k] = Bm.get(k, 0) + v / 3
print("seed-averaged paired:", paired(A, Bm))

print("== Llama (this environment): default minus config")
L = f"{W}/runs/llama"
base = load(L + "/default/qa/raw_predictions.jsonl")
print("default", round(summ(L + "/default/qa")["macro"], 3), summ(L + "/default/qa")["task_scores"])
for n in sorted(os.listdir(L)):
    q = f"{L}/{n}/qa"
    if n == "default" or not os.path.exists(q + "/macro_summary.json"):
        continue
    print(f"{n:12s} {summ(q)['macro']:.3f}  default-minus-this {paired(base, load(q + '/raw_predictions.jsonl'))}")
print("lingua tasks", summ(L + "/lingua/qa")["task_scores"])
