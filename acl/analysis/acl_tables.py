"""All numbers for the ACL draft, from A100 per-sample predictions.
Paired task-stratified bootstrap: 10,000 resamples, seed 20260701 (two-model: model x task strata).
usage: python acl_tables.py RUNS_DIR > acl_numbers.txt
"""
import json
import random
import sys
from pathlib import Path

R = Path(sys.argv[1])
TASKS = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", "narrativeqa"]
MH = ["hotpotqa", "2wikimqa", "musique"]
M = {
    "span": "a100/{m}_span_only_source_order_1024_qa", "tfidf": "rv5/{m}_tfidf_qa", "bm25": "rv5/{m}_bm25_qa",
    "bge": "rv5/{m}_bge_qa", "bp": "a100/{m}_bridge_neighbor32_path_order_1024_qa",
    "lbp": "a100/{m}_lexical_bridgepack_path_order_1024_qa", "bge_recall": "a100/{m}_bge_recall_bp_1024_qa",
    "bge_union": "a100/{m}_bge_union_bp_1024_qa", "bge_rerank": "a100/{m}_bge_rerank_bp_1024_qa",
    "head": "rv5/{m}_head_qa", "middle": "rv5/{m}_middle_qa", "sc": "rv5/{m}_sc_qa", "lingua2": "rv5/{m}_lingua2_qa",
    "full": "rv5/{m}_full_qa", "lbp_nn": "a100/{m}_lbp_no_neighbor_1024_qa", "bp_nn": "a100/{m}_bp_no_neighbor_1024_qa",
    "lsrc": "a100/{m}_lexical_bridgepack_source_order_1024_qa", "bsrc": "a100/{m}_bridge_same_spans_source_order_1024_qa",
    "tags": "a100/{m}_bridge_neighbor32_path_order_tags_1024_qa",
}
LINGUA = {"qwen": "lingua/full", "llama": "llama/lingua/qa"}
for b in (512, 2048):
    for k, v in {"span": "span_only_source_order", "tfidf": "tfidf_chunk_source_order", "bm25": "bm25_chunk_source_order",
                 "bge": "bge", "bp": "bridge_neighbor32_path_order", "lbp": "lexical_bridgepack_path_order",
                 "bge_recall": "bge_recall_bp", "bge_union": "bge_union_bp", "bge_rerank": "bge_rerank_bp"}.items():
        M[f"{k}@{b}"] = f"a100/{{m}}_{v}_{b}_qa"


def load(m, k):
    p = R / (LINGUA[m] if k == "longlingua" else M[k].format(m=m)) / "raw_predictions.jsonl"
    if not p.exists():
        return None
    return {(r["task"], str(r["sample_id"])): (100 * r["score"] if r["score"] <= 1 else r["score"], r.get("input_token_count"))
            for r in map(json.loads, p.open(encoding="utf-8"))}


def task_means(d, tasks=TASKS):
    return {t: sum(v[0] for k, v in d.items() if k[0] == t) / max(1, sum(1 for k in d if k[0] == t)) for t in tasks}


def boot(pairs_by_stratum, B=10000):
    # pairs_by_stratum: list of lists of differences; statistic = mean over strata of stratum means
    pt = sum(sum(v) / len(v) for v in pairs_by_stratum) / len(pairs_by_stratum)
    rng = random.Random(20260701)
    bs = sorted(sum(sum(v[rng.randrange(len(v))] for _ in v) / len(v) for v in pairs_by_stratum) / len(pairs_by_stratum)
                for _ in range(B))
    p = min(1.0, 2 * min(sum(x <= 0 for x in bs), sum(x >= 0 for x in bs)) / B)
    return pt, bs[int(.025 * B)], bs[int(.975 * B) - 1], p


def diff(a, b, models, tasks=TASKS):
    strata = []
    for m in models:
        da, db = load(m, a), load(m, b)
        if da is None or db is None:
            return None
        for t in tasks:
            strata.append([da[k][0] - db[k][0] for k in da if k[0] == t and k in db])
    return boot(strata)


if __name__ == "__main__":
    ALL = list(M) + ["longlingua"]
    print("## per-task / macro / MH")
    for k in ALL:
        for m in ("qwen", "llama"):
            d = load(m, k)
            if d is None:
                continue
            tm = task_means(d)
            toks = [v[1] for v in d.values() if v[1]]
            print(f"{k:12s} {m:5s} n={len(d)} macro={sum(tm.values())/6:.2f} MH={sum(tm[t] for t in MH)/3:.2f} "
                  + " ".join(f"{t[:6]}={tm[t]:.2f}" for t in TASKS)
                  + (f" in_tok_mean={sum(toks)/len(toks):.0f} max={max(toks)}" if toks else ""))

    print("\n## paired A-B (per model and two-model)")
    CMP = [("lbp", x) for x in ("span", "bm25", "tfidf", "bge", "bp", "head", "middle", "sc", "lingua2", "longlingua", "full")] + \
          [("bp", x) for x in ("span", "bm25", "tfidf")] + \
          [(x, y) for x in ("bge_recall", "bge_union", "bge_rerank") for y in ("lbp", "bge", "bm25", "tfidf", "span")] + \
          [("lbp", "lbp_nn"), ("bp", "bp_nn"), ("lbp", "lsrc"), ("bp", "bsrc"), ("bp", "tags")] + \
          [(f"{x}@512", f"{y}@512") for x in ("lbp", "bge_rerank", "bge_union", "bge_recall") for y in ("bm25", "tfidf", "bge", "span")]
    for a, b in CMP:
        for ms, tag in ((("qwen",), "qwen"), (("llama",), "llama"), (("qwen", "llama"), "2M")):
            for tasks, tt in ((TASKS, "all"), (MH, "MH")):
                r = diff(a, b, ms, tasks)
                if r:
                    star = "*" if r[1] > 0 else ""
                    print(f"{a:12s} - {b:12s} {tag:5s} {tt:3s} {r[0]:+.2f} [{r[1]:+.2f}, {r[2]:+.2f}] p={r[3]:.4f} {star}")
