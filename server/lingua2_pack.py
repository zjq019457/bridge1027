"""LLMLingua-2 (official PromptCompressor, xlm-roberta-large-meetingbank) packed under the
exact 1,024-token budget of each downstream tokenizer.

Same protocol as lingua_pack.py / sc_pack.py: compress to the target size, add the
"Question/Selected evidence" prefix, prefix-truncate in model tokens only on overflow.
One compression feeds both the Qwen and the Llama trace.
usage: lingua2_pack.py SHARD NSHARD
"""
import hashlib
import json
import sys
import time
from pathlib import Path

W = Path("/diskE/jiaqizhang/BRIDGEPACK_SENSITIVITY_V1_20261025")
SHARD, NSHARD = int(sys.argv[1]), int(sys.argv[2])
BUDGET = 1024
TARGET = 1000
CHUNK_WORDS = 300
TOKS = {"qwen": "/diskF/jiaqizhang/baselines/RepairKV/models/Qwen2.5-7B-Instruct",
        "llama": "/diskD/jiaqizhang/models/Meta-Llama-3-8B-Instruct"}
OUT = W / "runs" / "rv5" / "lingua2"
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(W / "source" / "SlimInfer_snapshot"))
sys.path.insert(0, str(W / "scripts"))
import portable_runner as pr  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from llmlingua import PromptCompressor  # noqa: E402

templates = json.loads((W / "source" / "SlimInfer_snapshot" / "configs" / "dataset2prompt.json").read_text())


def sha(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def chunks(context):
    paras = [p.strip() for p in context.split("\n") if p.strip()]
    out, cur, n = [], [], 0
    for p in paras:
        w = len(p.split())
        if cur and n + w > CHUNK_WORDS:
            out.append("\n".join(cur))
            cur, n = [], 0
        cur.append(p)
        n += w
    if cur:
        out.append("\n".join(cur))
    return out or [context]


ids, samples, _ = pr.load_selection_inputs(W / "assets" / "longbench_g1_data", W / "golden", None)
flat = [s for t in pr.TASKS for s in samples[t]]
mine = [s for i, s in enumerate(flat) if i % NSHARD == SHARD]
toks = {k: AutoTokenizer.from_pretrained(v, local_files_only=True, use_fast=True) for k, v in TOKS.items()}
comp = PromptCompressor(model_name=str(W / "llmlingua-2-xlm-roberta-large-meetingbank"), use_llmlingua2=True, device_map="cuda")

files = {k: (OUT / f"{k}_shard{SHARD}.jsonl") for k in TOKS}
done = set()
if files["qwen"].exists():
    done = {(json.loads(l)["task"], json.loads(l)["sample_id"]) for l in files["qwen"].open()}
fh = {k: p.open("a") for k, p in files.items()}
for s in mine:
    if (s["task"], s["sample_id"]) in done:
        continue
    t0 = time.time()
    q = s["question"]
    res = comp.compress_prompt(chunks(s["context"]), target_token=TARGET, force_tokens=["\n", "?"],
                               drop_consecutive=True)
    compressed = res["compressed_prompt"].strip()
    secs = round(time.time() - t0, 2)
    prefix = f"Question: {q}\n\nSelected evidence:"
    for k, tok in toks.items():
        packed = "\n\n".join([prefix, compressed]).strip()
        pid = tok(packed, add_special_tokens=False).input_ids
        pre_len = len(pid)
        full_ids, keep = pid, BUDGET
        while len(pid) > BUDGET:  # prefix truncation; shrink again on decode/re-encode drift
            packed = tok.decode(full_ids[:keep], skip_special_tokens=True)
            pid = tok(packed, add_special_tokens=False).input_ids
            keep -= max(1, len(pid) - BUDGET)
        gp = templates.get(s["task"], "").format(context=packed, input=q, question=q)
        fh[k].write(json.dumps({
            "task": s["task"], "sample_id": s["sample_id"], "variant": "llmlingua2_exact_budget",
            "budget": BUDGET, "packed_text": packed, "generation_prompt": gp, "answers": s["answers"],
            "all_classes": s["all_classes"], "question": q, "model_token_count": len(pid),
            "pre_truncation_tokens": pre_len, "truncated": pre_len > BUDGET,
            "packed_token_ids_sha256": sha(json.dumps(pid, separators=(",", ":"))), "compress_seconds": secs,
        }) + "\n")
        fh[k].flush()
print("SHARD_DONE", SHARD)
