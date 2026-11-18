"""RECOMP-extractive sentence order for every test instance, computed once on the GPU (same checkpoint, same
max_length and mean pooling as packers._recomp_embed). recomp_ext reads it for every tokenizer.
usage: CUDA_VISIBLE_DEVICES=0 python rc_scores.py   (resumable; writes cache/recomp_order.jsonl)"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import localpack as L
import packers as P
from transformers import AutoModel, AutoTokenizer

torch.cuda.set_per_process_memory_fraction(0.06)  # ~2.4 GB: the card is shared with other jobs
OUT = os.path.join(P._HERE, "cache", "recomp_order.jsonl")
done = set()
if os.path.exists(OUT):
    done = {json.loads(l)["key"] for l in open(OUT, encoding="utf-8")}
S = [s for s in L.load_samples() if f"{s['task']}/{s['sample_id']}" not in done]
print("todo", len(S), flush=True)
p = os.path.join(P._AUX, "hotpotqa_extractive_compressor")
tok = AutoTokenizer.from_pretrained(p)
m = AutoModel.from_pretrained(p).cuda().eval()


def embed(texts):
    idx = sorted(range(len(texts)), key=lambda i: len(texts[i]))  # length-sorted batches, less padding
    E = torch.empty(len(texts), m.config.hidden_size, device="cuda")
    for a in range(0, len(idx), 128):
        ii = idx[a:a + 128]
        b = tok([texts[i] for i in ii], padding=True, truncation=True, max_length=256, return_tensors="pt").to("cuda")
        h = m(**b).last_hidden_state
        mk = b["attention_mask"].unsqueeze(-1).float()
        E[ii] = (h * mk).sum(1) / mk.sum(1).clamp(min=1e-9)
    return E


t0 = time.time()
with torch.no_grad(), open(OUT, "a", encoding="utf-8") as f:
    for k, s in enumerate(S):
        sents = P._sentences(s["context"])
        order = []
        if sents:
            E = embed([s["question"]] + sents)
            sc = (E[1:] @ E[0]).tolist()
            order = sorted(range(len(sents)), key=lambda i: -sc[i])
        f.write(json.dumps({"key": f"{s['task']}/{s['sample_id']}", "n": len(sents), "order": order}) + "\n")
        f.flush()
        if k % 50 == 0:
            print(k, round(time.time() - t0), flush=True)
print("RC_DONE", round(time.time() - t0), flush=True)
