"""Provence (Chirkova et al. 2025, official naver/provence-reranker-debertav3-v1) under the
exact 1,024 downstream-token budget, in the same trace format as the other baselines.

The context is cut into ~300-word chunks (Provence encodes question + passage in 512 tokens);
each chunk is pruned at the official threshold 0.1 and gets Provence's reranking score.
Pruned chunks are admitted by score while they fit the exact budget, then serialized in
source order after the shared "Question / Selected evidence" prefix.
usage: provence_pack.py TOKENIZER DEFAULT_TRACE OUT [BUDGET]
"""
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

W = Path("/diskE/jiaqizhang/BRIDGEPACK_SENSITIVITY_V1_20261025")
TOKP, TRACE, OUT = sys.argv[1], sys.argv[2], Path(sys.argv[3])
BUDGET = int(sys.argv[4]) if len(sys.argv) > 4 else 1024
CHUNK_WORDS = 300

tok = AutoTokenizer.from_pretrained(TOKP, use_fast=True)
ntok = lambda s: len(tok(s, add_special_tokens=False).input_ids)  # noqa: E731
pv = AutoModel.from_pretrained(W / "models_provence", trust_remote_code=True).to("cuda").eval()

ctx = {}
for t in ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", "narrativeqa"]:
    for l in open(W / "assets/longbench_g1_data" / f"{t}.jsonl"):
        r = json.loads(l); ctx[(t, r["_id"])] = (r["input"], r["context"])

trunc = 0
with open(TRACE) as fin, OUT.open("w") as f:
    for l in fin:
        x = json.loads(l)
        q, c = ctx[(x["task"], x["sample_id"])]
        words = c.split()
        chunks = [" ".join(words[i:i + CHUNK_WORDS]) for i in range(0, len(words), CHUNK_WORDS)] or [""]
        with torch.no_grad():
            res = pv.process(q, [chunks], title=None, threshold=0.1, enable_warnings=False, batch_size=32)
        pruned, scores = res["pruned_context"][0], res["reranking_score"][0]
        prefix = x["packed_text"].split("\n\n")[0] + "\n\nSelected evidence:"
        sel = []
        for i in sorted(range(len(chunks)), key=lambda i: -float(scores[i])):
            if not pruned[i].strip():
                continue
            trial = sorted(sel + [i])
            if ntok("\n\n".join([prefix] + [pruned[j] for j in trial])) <= BUDGET:
                sel = trial
        if not sel:  # first chunk alone overflows: exact prefix truncation, as for the other baselines
            best = max(range(len(chunks)), key=lambda i: float(scores[i]))
            ids = tok(pruned[best] or chunks[best], add_special_tokens=False).input_ids
            room = BUDGET - ntok(prefix + "\n\n")
            while ntok(prefix + "\n\n" + tok.decode(ids[:room])) > BUDGET:
                room -= 4
            packed = prefix + "\n\n" + tok.decode(ids[:room]); trunc += 1
        else:
            packed = "\n\n".join([prefix] + [pruned[j] for j in sel])
        assert x["generation_prompt"].count(x["packed_text"]) == 1
        y = dict(x); y["variant"] = "provence"; y["packed_text"] = packed
        y["generation_prompt"] = x["generation_prompt"].replace(x["packed_text"], packed)
        y["model_token_count"] = ntok(packed)
        f.write(json.dumps(y) + "\n"); f.flush()
print("PROVENCE_DONE truncated", trunc)
