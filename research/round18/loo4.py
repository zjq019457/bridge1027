"""Pooled leave-one-out deltas with stratified (per-task) paired bootstrap; reports 3-model check and 4-model result."""
import json, random
from collections import defaultdict
S = "_budget=1024_2wikimqa-hotpotqa-multifieldqa_en-musique-narrativeqa-qasper.preds.jsonl"
FULL = {"qwen": "test_preds/qwen/lex_rep_seed_mode=2_sent=1" + S, "llama": "test_preds/llama/lex_rep_seed_mode=2_sent=1" + S,
        "qwen14": "test_preds/qwen14/lex_rep_seed_mode=2_sent=1" + S, "q32": "run7_q32/preds/lex_rep_seed_mode=2_sent=1" + S}
ABL = {m: (f"run15_q32/preds/final_abl_{{}}=0" + S) if m == "q32" else (f"test_preds/{m}/final_abl_{{}}=0" + S) for m in FULL}
def load(p):
    return {(r["task"], r["sample_id"]): r["score"] for r in map(json.loads, open(p))}
random.seed(0)
for a in ["merge", "backfill", "sent", "docorder"]:
    diff = {m: {} for m in FULL}
    for m in FULL:
        f, g = load(FULL[m]), load(ABL[m].format(a))
        for k in f: diff[m][k] = g[k] - f[k]
    keys = list(diff["qwen"]); bytask = defaultdict(list)
    for k in keys: bytask[k[0]].append(k)
    def stat(ms, sample):
        return sum(sum(sum(diff[m][k] for k in sample[t]) / len(sample[t]) for t in sample) / len(sample) for m in ms) / len(ms)
    for ms in (["qwen", "llama", "qwen14"], list(FULL)):
        pt = stat(ms, bytask); bs = []
        for _ in range(2000):
            bs.append(stat(ms, {t: random.choices(v, k=len(v)) for t, v in bytask.items()}))
        bs.sort()
        print(a, len(ms), "per-model", [round(stat([m], bytask), 2) for m in ms], "pooled %.2f [%.2f, %.2f]" % (pt, bs[50], bs[1949]))
