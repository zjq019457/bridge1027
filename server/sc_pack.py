"""Selective Context (Li et al. 2023, official package, GPT-2 self-information, phrase level)
packed under the exact 1,024-token budget of each downstream tokenizer.

Same protocol as lingua_pack.py: compress the context to the target size, add the
"Question/Selected evidence" prefix, and apply an explicit prefix truncation in model
tokens only if the result still exceeds the budget. One compression feeds both the
Qwen and the Llama trace.
usage: sc_pack.py SHARD NSHARD
"""
import hashlib
import json
import re
import sys
import time
from pathlib import Path

W = Path("/diskE/jiaqizhang/BRIDGEPACK_SENSITIVITY_V1_20261025")
SHARD, NSHARD = int(sys.argv[1]), int(sys.argv[2])
BUDGET = 1024
GPT2 = str(W / "models_small_gpt2")
TOKS = {"qwen": "/diskF/jiaqizhang/baselines/RepairKV/models/Qwen2.5-7B-Instruct",
        "llama": "/diskD/jiaqizhang/models/Meta-Llama-3-8B-Instruct"}
OUT = W / "runs" / "rv5" / "sc"
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(W / "source" / "SlimInfer_snapshot"))
sys.path.insert(0, str(W / "scripts"))
import portable_runner as pr  # noqa: E402
import selective_context as scm  # noqa: E402
from transformers import AutoTokenizer, GPT2LMHeadModel, GPT2Tokenizer  # noqa: E402


class LocalSC(scm.SelectiveContext):
    def _prepare_model(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained(GPT2)
        self.model = GPT2LMHeadModel.from_pretrained(GPT2).to(self.device).eval()
        self.max_token_length = self.model.config.n_positions
        self.get_self_information = self._get_self_info_via_gpt2


templates = json.loads((W / "source" / "SlimInfer_snapshot" / "configs" / "dataset2prompt.json").read_text())
ids, samples, _ = pr.load_selection_inputs(W / "assets" / "longbench_g1_data", W / "golden", None)
flat = [s for t in pr.TASKS for s in samples[t]]
mine = [s for i, s in enumerate(flat) if i % NSHARD == SHARD]
toks = {k: AutoTokenizer.from_pretrained(v, local_files_only=True, use_fast=True) for k, v in TOKS.items()}
sc = LocalSC(model_type="gpt2", lang="en")


def split_long(text, max_words=400):
    # GPT-2 has 1,024 positions; give the package's sentence splitter pieces it can score
    out = []
    for sent in re.split(sc.sent_tokenize_pattern, sc.beautify_context(text)):
        w = sent.split()
        out += [" ".join(w[i:i + max_words]) + "." for i in range(0, len(w), max_words)] or []
    return " ".join(out)


def sha(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


files = {k: (OUT / f"{k}_shard{SHARD}.jsonl") for k in TOKS}
done = set()
if files["qwen"].exists():
    done = {(json.loads(l)["task"], json.loads(l)["sample_id"]) for l in files["qwen"].open()}
fh = {k: p.open("a") for k, p in files.items()}
for s in mine:
    if (s["task"], s["sample_id"]) in done:
        continue
    t0 = time.time()
    q, ctx = s["question"], split_long(s["context"])
    prefix = f"Question: {q}\n\nSelected evidence:"
    n_ctx = len(toks["qwen"](ctx, add_special_tokens=False).input_ids)
    room = BUDGET - len(toks["qwen"](prefix + "\n\n", add_special_tokens=False).input_ids)
    ratio = min(0.99, max(0.0, 1 - room / max(1, n_ctx)))
    compressed = sc(ctx, reduce_ratio=ratio, reduce_level="phrase")[0].strip() if ratio > 0 else ctx
    secs = round(time.time() - t0, 2)
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
            "task": s["task"], "sample_id": s["sample_id"], "variant": "selective_context_exact_budget",
            "budget": BUDGET, "packed_text": packed, "generation_prompt": gp, "answers": s["answers"],
            "all_classes": s["all_classes"], "question": q, "model_token_count": len(pid),
            "pre_truncation_tokens": pre_len, "truncated": pre_len > BUDGET, "reduce_ratio": ratio,
            "packed_token_ids_sha256": sha(json.dumps(pid, separators=(",", ":"))), "compress_seconds": secs,
        }) + "\n")
        fh[k].flush()
print("SHARD_DONE", SHARD)
