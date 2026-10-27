import json, random, sys
from pathlib import Path
sys.path.insert(0, "source/SlimInfer_snapshot")
from longbench_eval import scorer
TASKS = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", "narrativeqa"]
R = Path("/diskE/jiaqizhang/BRIDGEPACK_V2_REPRODUCTION_QWEN_INSTRUCT_1024_V1_20260926/full_11_variants/predictions")
def load11(v):
    d = {}
    for t in TASKS:
        for l in open(R / v / f"{t}.jsonl"):
            r = json.loads(l); d[(t, str(r["sample_id"]))] = float(scorer(t, [r["pred"]], [r["answers"]], [r.get("length", 0)], r.get("all_classes")))
    return d
def loadrun(p): return {(r["task"], str(r["sample_id"])): r["score"] for r in map(json.loads, open(p))}
def paired(a, b, B=10000):
    d = {t: [a[k] - b[k] for k in a if k[0] == t and k in b] for t in TASKS}
    pt = sum(sum(v) / len(v) for v in d.values()) / 6
    rng = random.Random(20260701)
    bs = sorted(sum(sum(v[rng.randrange(len(v))] for _ in v) / len(v) for v in d.values()) / 6 for _ in range(B))
    p = 2 * min(sum(x <= 0 for x in bs), sum(x >= 0 for x in bs)) / B
    return f"{pt:+.3f} [{bs[int(.025*B)]:+.3f}, {bs[int(.975*B)-1]:+.3f}] p={min(1,p):.4f} n={sum(len(v) for v in d.values())}"
mac = lambda d: sum(sum(v for k, v in d.items() if k[0] == t) / sum(1 for k in d if k[0] == t) for t in TASKS) / 6
V = {v: load11(v) for v in ["lexical_bridgepack_source_order", "bridge_same_spans_source_order", "bm25_chunk_source_order", "tfidf_chunk_source_order"]}
V["rerun_default"] = loadrun("runs/qwen/default/raw_predictions.jsonl")
for k, d in V.items(): print(k, round(mac(d), 3), len(d))
for bp in ["lexical_bridgepack_source_order", "bridge_same_spans_source_order", "rerun_default"]:
    for base in ["bm25_chunk_source_order", "tfidf_chunk_source_order"]:
        print(bp, "vs", base, paired(V[bp], V[base]))
