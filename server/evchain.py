"""Evidence-chain retention (answer-bearing / supporting-fact / complete-chain) for packing traces.

Reuses the matching functions of tools/bridgepack_p0_evidence_chain.py unchanged; only the
annotation files (pre-downloaded) and the LongBench root are redirected to local paths.
usage: evchain.py name=trace.jsonl [name=trace.jsonl ...]
"""
import json
import sys
from pathlib import Path

W = Path("/diskE/jiaqizhang/BRIDGEPACK_SENSITIVITY_V1_20261025")
sys.path.insert(0, str(W / "source" / "SlimInfer_snapshot"))
import huggingface_hub  # noqa: E402

CACHE = W / "sf_cache"
LOCAL = {
    "hotpotqa/hotpot_qa": CACHE / "hotpot" / "distractor" / "validation-00000-of-00001.parquet",
    "voidful/2WikiMultihopQA": CACHE / "2wiki" / "dev.json",
    "dgslibisey/MuSiQue": CACHE / "musique" / "musique_ans_v1.0_dev.jsonl",
}
huggingface_hub.hf_hub_download = lambda repo_id, **kw: str(LOCAL[repo_id])

from tools import bridgepack_p0_evidence_chain as ec  # noqa: E402

ec.LONGBENCH_ROOT = W / "assets" / "longbench_g1_data"
bundle = ec.download_supporting_facts(CACHE)
aligned, stats = ec.align_longbench(bundle["by_task_q"])
print("align", json.dumps(stats))


def metrics(packed_text, gold, answers):
    ans = 1.0 if ec._answer_in_text(answers, packed_text) else 0.0
    facts = gold["supporting_facts"]
    hits = []
    for fact in facts:
        title = str(fact.get("title") or "")
        if fact.get("paragraph_text"):
            ok = ec._norm(fact["paragraph_text"])[:80] in ec._norm(packed_text) or ec._fact_in_packed(
                packed_text, fact.get("paragraph_text"), title)
        else:
            ctx = gold.get("orig_context")
            if ctx is None:
                ctx = gold.get("context") or ""
            ok = ec._fact_in_packed(packed_text, ec._fact_sentence(ctx, title, int(fact.get("sent_id", 0))), title)
        hits.append(bool(ok))
    return ans, sum(hits) / max(1, len(hits)), 1.0 if hits and all(hits) else 0.0


out = {}
for arg in sys.argv[1:]:
    name, path = arg.split("=", 1)
    rows = [json.loads(l) for l in open(path)]
    per = {t: [] for t in ec.MULTIHOP}
    for r in rows:
        key = (r["task"], str(r["sample_id"]))
        if r["task"] not in per or key not in aligned or not aligned[key].get("supporting_facts"):
            continue
        per[r["task"]].append(metrics(str(r["packed_text"]), aligned[key], aligned[key].get("answers") or r.get("answers") or []))
    res = {}
    for i, m in enumerate(["answer_bearing", "supporting_fact", "complete_chain"]):
        task_means = {t: sum(v[i] for v in per[t]) / len(per[t]) for t in per if per[t]}
        res[m] = sum(task_means.values()) / len(task_means)
        res[m + "_by_task"] = task_means
    res["n"] = {t: len(v) for t, v in per.items()}
    out[name] = res
    print(f"{name:28s} answer={res['answer_bearing']:.4f} support={res['supporting_fact']:.4f} chain={res['complete_chain']:.4f} n={res['n']}")
(W / "analysis" / "evchain_summary.json").write_text(json.dumps(out, indent=2))
