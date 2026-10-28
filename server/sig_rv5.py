import json, random, sys
TASKS = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", "narrativeqa"]
def load(p): return {(r["task"], str(r["sample_id"])): r["score"] for r in map(json.loads, open(p))}
def paired(a, b, B=10000):
    d = {t: [a[k] - b[k] for k in a if k[0] == t and k in b] for t in TASKS}
    pt = sum(sum(v) / len(v) for v in d.values()) / 6
    rng = random.Random(20260701)
    bs = sorted(sum(sum(v[rng.randrange(len(v))] for _ in v) / len(v) for v in d.values()) / 6 for _ in range(B))
    p = 2 * min(sum(x <= 0 for x in bs), sum(x >= 0 for x in bs)) / B
    return f"{pt:+.3f} [{bs[int(.025*B)]:+.3f}, {bs[int(.975*B)-1]:+.3f}] p={min(1,p):.4f} n={sum(len(v) for v in d.values())}"

if __name__ == "__main__":
    BP = {"qwen": load("runs/qwen/default/raw_predictions.jsonl"), "llama": load("runs/llama/default/qa/raw_predictions.jsonl")}
    for name in sys.argv[1:]:
        m = name.split("_")[0]
        print(f"BP({m}) minus {name}:", paired(BP[m], load(f"runs/rv5/{name}_qa/raw_predictions.jsonl")))
