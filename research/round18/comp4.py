"""Build-up ablation (BGE ranking, 1,024) on four models: macro per model and pooled step deltas with
task-stratified paired bootstrap (2,000 resamples)."""
import json, random, glob
from collections import defaultdict
T = "_budget=1024_2wikimqa-hotpotqa-multifieldqa_en-musique-narrativeqa-qasper.preds.jsonl"
A = "final_abl_seed_mode=1_"
ROWS = {"chunk": A + "sent=0_merge=0_docorder=0_backfill=1", "merge": A + "sent=0_merge=1_docorder=0_backfill=1",
        "doc": A + "sent=0_merge=1_docorder=1_backfill=1", "dp": "lex_rep_seed_mode=1_sent=1",
        "nobf": A + "sent=1_merge=1_docorder=1_backfill=0", "nomerge": A + "sent=1_merge=0_docorder=1_backfill=1",
        "mmr": "mmr_pack_seed_mode=1", "skip": "dedup_skip_seed_mode=1"}
DIRS = {"q7": ["run16_q7/preds", "test_preds/qwen"], "llama": ["test_preds2/run4_llama", "test_preds/llama", "run7_llama/preds"],
        "q14": ["test_preds2/run4_qwen", "test_preds/qwen14", "run7_qwen/preds", "test_preds2/*qwen*", "run*_qwen/preds"],
        "q32": ["run15_q32/preds", "run8_q32/preds", "run11_q32/preds"]}
def find(m, stem):
    for d in DIRS[m]:
        g = glob.glob(f"{d}/{stem}{T}")
        if g: return g[0]
P = {}
for m in DIRS:
    for r, stem in ROWS.items():
        p = find(m, stem); P[m, r] = p
        print(m, r, p)
load = lambda p: {(x["task"], x["sample_id"]): x["score"] for x in map(json.loads, open(p))}
S = {k: load(p) for k, p in P.items() if p}
def macro(d, keys_by_task): return sum(sum(d[k] for k in ks) / len(ks) for ks in keys_by_task.values()) / len(keys_by_task)
bt = defaultdict(list)
for k in S["q7", "chunk"]: bt[k[0]].append(k)
for m in DIRS: print(m, {r: round(macro(S[m, r], bt), 2) for r in ROWS if (m, r) in S})
random.seed(0); B = [{t: random.choices(v, k=len(v)) for t, v in bt.items()} for _ in range(2000)]
def pooled(a, b, ms):
    ms = [m for m in ms if (m, a) in S and (m, b) in S]
    f = lambda kb: sum(macro(S[m, a], kb) - macro(S[m, b], kb) for m in ms) / len(ms)
    bs = sorted(f(kb) for kb in B)
    return len(ms), round(f(bt), 2), round(bs[50], 2), round(bs[1949], 2)
for a, b in [("merge", "chunk"), ("doc", "merge"), ("dp", "doc"), ("nobf", "dp"), ("nomerge", "dp"), ("mmr", "chunk"), ("skip", "chunk"), ("dp", "chunk")]:
    print(a, "-", b, "2 models", pooled(a, b, ["q14", "llama"]), "4 models", pooled(a, b, list(DIRS)))
