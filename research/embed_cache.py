"""Cache bge-small-en-v1.5 similarities (question vs. every candidate span) for an instance set,
so that dense-seed variants can be evaluated without re-encoding.
usage: python embed_cache.py dev|test
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

import localpack as L
from bridgekv import evidence_text_packing as etp

BGE = os.environ.get("BGE_PATH", "BAAI/bge-small-en-v1.5")
INSTR = "Represent this sentence for searching relevant passages: "
OUT = Path("cache"); OUT.mkdir(exist_ok=True)

torch.set_num_threads(8)
tok = AutoTokenizer.from_pretrained(BGE)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModel.from_pretrained(BGE).to(DEV).eval()


@torch.no_grad()
def enc(texts):
    out = []
    for i in range(0, len(texts), 256):
        b = tok(texts[i:i + 256], padding=True, truncation=True, max_length=512, return_tensors="pt").to(DEV)
        e = model(**b).last_hidden_state[:, 0]
        out.append(torch.nn.functional.normalize(e, dim=-1))
    return torch.cat(out).float().cpu().numpy()


which = sys.argv[1]
samples = L.load_dev() if which == "dev" else L.load_inf() if which == "inf" else L.load_samples()
path = OUT / f"{os.environ.get('BP_ENC', 'bge')}_{which}.jsonl"
done = set()
if path.exists():
    done = {json.loads(l)["key"] for l in path.open(encoding="utf-8")}
with path.open("a", encoding="utf-8") as f:
    for n, s in enumerate(samples):
        key = f"{s['task']}/{s['sample_id']}"
        if key in done:
            continue
        cands, _ = etp.build_candidate_spans(s["sample_id"], s["task"], s["question"], s["context"])
        if not cands:
            f.write(json.dumps({"key": key, "sim": {}}) + "\n"); continue
        q = enc([INSTR + s["question"]])[0]
        sims = enc([c.text for c in cands]) @ q
        f.write(json.dumps({"key": key, "sim": {c.span_id: round(float(x), 5) for c, x in zip(cands, sims)}}) + "\n")
        if n % 50 == 0:
            print(n, flush=True)
print("EMBED_DONE", which)
