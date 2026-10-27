import json,random,sys
from pathlib import Path
W=Path("/diskE/jiaqizhang/BRIDGEPACK_SENSITIVITY_V1_20261025")
T=["qasper","multifieldqa_en","hotpotqa","2wikimqa","musique","narrativeqa"]
def ld(p):
    return {(r["task"],str(r["sample_id"])):r["score"] for r in map(json.loads,open(p))}
base=ld(W/"runs/llama/default/qa/raw_predictions.jsonl")
def paired(o):
    d={t:[] for t in T}
    for k,v in base.items():
        if k in o: d[k[0]].append(o[k]-v)
    rng=random.Random(20260701); pt=sum(sum(x)/len(x) for x in d.values())/6; bs=[]
    for _ in range(10000):
        bs.append(sum(sum(rng.choice(x) for _ in x)/len(x) for x in d.values())/6)
    bs.sort(); return pt,bs[249],bs[9749],sum(len(x) for x in d.values())
print("default", json.load(open(W/"runs/llama/default/qa/macro_summary.json"))["macro"])
for n in sys.argv[1:]:
    p=W/"runs/llama"/n/"qa"
    if not (p/"raw_predictions.jsonl").exists(): continue
    m=json.load(open(p/"macro_summary.json")); r=paired(ld(p/"raw_predictions.jsonl"))
    print(n, round(m["macro"],3), "delta(other-default) %.3f [%.3f,%.3f] n=%d"%r, {k:round(v,2) for k,v in m["task_scores"].items()})
