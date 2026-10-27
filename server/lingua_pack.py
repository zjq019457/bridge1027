"""LongLLMLingua (official PromptCompressor, Llama-2-7b-chat) packed under the exact Qwen 1,024-token budget.

Writes packing_trace.jsonl rows in the portable runner's format, so QA uses the identical
Qwen model, prompt template, decoding and scorer as BridgePack (runner --qa-only).
Protocol: official LongLLMLingua question-aware compression to ~target tokens, then the
same "Question/Selected evidence" prefix as BridgePack, then an explicit Qwen-token prefix
truncation if (and only if) the compressed text still exceeds 1,024 tokens.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

W = Path(sys.argv[1])
QWEN = sys.argv[2]
COMPRESSOR = sys.argv[3]
OUT = Path(sys.argv[4])
SHARD, NSHARD = int(sys.argv[5]), int(sys.argv[6])
BUDGET = 1024
TARGET = 1000  # compressor-side target (Llama-2 tokens); exact Qwen-token truncation applies only on overflow
CHUNK_WORDS = 300

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
tok = AutoTokenizer.from_pretrained(QWEN, local_files_only=True, trust_remote_code=True, use_fast=True)
comp = PromptCompressor(model_name=COMPRESSOR, device_map="cuda")

done = set()
if OUT.exists():
    done = {(json.loads(l)["task"], json.loads(l)["sample_id"]) for l in OUT.open()}
with OUT.open("a") as f:
    for s in mine:
        if (s["task"], s["sample_id"]) in done:
            continue
        t0 = time.time()
        q = s["question"]
        res = comp.compress_prompt(
            chunks(s["context"]), question=q, instruction="", target_token=TARGET,
            rank_method="longllmlingua", condition_in_question="after_condition",
            reorder_context="sort", dynamic_context_compression_ratio=0.3,
            condition_compare=True, context_budget="+100", concate_question=False,
        )
        compressed = res["compressed_prompt"].strip()
        prefix = f"Question: {q}\n\nSelected evidence:"
        packed = "\n\n".join([prefix, compressed]).strip()
        pid = tok(packed, add_special_tokens=False).input_ids
        pre_len = len(pid)
        truncated = pre_len > BUDGET
        if truncated:
            packed = tok.decode(pid[:BUDGET], skip_special_tokens=True)
            pid = tok(packed, add_special_tokens=False).input_ids
            if len(pid) > BUDGET:  # decode/re-encode drift; trim again
                packed = tok.decode(pid[:BUDGET], skip_special_tokens=True)
                pid = tok(packed, add_special_tokens=False).input_ids
        gp = templates.get(s["task"], "").format(context=packed, input=q, question=q)
        f.write(json.dumps({
            "task": s["task"], "sample_id": s["sample_id"], "variant": "longllmlingua_exact_budget",
            "budget": BUDGET, "packed_text": packed, "generation_prompt": gp,
            "answers": s["answers"], "all_classes": s["all_classes"], "question": q,
            "model_token_count": len(pid), "pre_truncation_tokens": pre_len, "truncated": truncated,
            "exact_model_token_budget": len(pid) <= BUDGET,
            "packed_token_ids": pid, "packed_token_ids_sha256": sha(json.dumps(pid, separators=(",", ":"))),
            "compressor": COMPRESSOR, "compress_seconds": round(time.time() - t0, 2),
        }) + "\n")
        f.flush()
print("SHARD_DONE", SHARD)
