"""Find a real example: same BGE ranking, standard chunk packing wrong and DedupPack right (Qwen-14B and Llama-8B),
standard context contains duplicated sentences, the answer string appears only in the DedupPack context."""
import json, re
from collections import Counter
TS = "_budget=1024_2wikimqa-hotpotqa-multifieldqa_en-musique-narrativeqa-qasper"
STD, DP = "bge_chunk" + TS, "lex_rep_seed_mode=1_sent=1" + TS
def load(p):
    return {(r["task"], r["sample_id"]): r for r in map(json.loads, open(p, encoding="utf-8"))}
ctx_s = load(f"run7_q32/out/{STD}.jsonl"); ctx_d = load(f"run8_q32/out/{DP}.jsonl")
pr = {m: (load(f"test_preds/{m}/{STD}.preds.jsonl"), load(f"test_preds/{m}/{DP}.preds.jsonl")) for m in ("qwen14", "llama")}
sent = lambda t: [x.strip() for x in re.split(r"(?<=[.!?])\s+", t) if len(x.strip()) > 30]
out = []
for k in ctx_s:
    if k[0] not in ("hotpotqa", "2wikimqa"): continue
    if not all(pr[m][0][k]["score"] == 0 and pr[m][1][k]["score"] == 100 for m in pr): continue
    s, d = ctx_s[k]["packed_text"], ctx_d[k]["packed_text"]
    ans = ctx_s[k]["answers"]
    if any(a.lower() in s.lower() for a in ans) or not any(a.lower() in d.lower() for a in ans): continue
    c = Counter(sent(s)); dup = sum(n - 1 for n in c.values() if n > 1)
    out.append((dup, k, ctx_s[k]["question"], ans, pr["qwen14"][0][k]["prediction"]))
out.sort(reverse=True)
print(len(out))
for o in out[:12]: print(o)
