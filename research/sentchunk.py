"""Baseline: sentence-aligned chunks + dense ranking (a strong conventional packer).
The context is split into sentences, consecutive sentences are grouped into chunks of at most
64 words, chunks are ranked by BGE cosine similarity to the question and admitted in similarity
order while they fit the exact budget (later chunks may still fit), emitted in similarity order.
usage: python sentchunk.py embed test|inf      (GPU; writes cache/{BP_ENC}sent_{which}.jsonl)
       python sentchunk.py pack test|inf TOKDIR BUDGET OUT.jsonl
"""
import json, os, re, sys
from pathlib import Path
import localpack as L

ENC = os.environ.get("BP_ENC", "bge")
which = sys.argv[2]
samples = L.load_inf() if which == "inf" else L.load_samples()
cache = Path("cache") / f"{ENC}sent_{which}.jsonl"
SPLIT = re.compile(r"(?<=[.?!])\s+")


def chunks(ctx, maxw=64):
    out, cur, n = [], [], 0
    for s in SPLIT.split(ctx):
        w = len(s.split())
        if cur and n + w > maxw:
            out.append(" ".join(cur)); cur, n = [], 0
        cur.append(s); n += w
    if cur:
        out.append(" ".join(cur))
    return out


if sys.argv[1] == "embed":
    import torch
    from transformers import AutoModel, AutoTokenizer
    path = os.environ["BGE_PATH"]
    tok = AutoTokenizer.from_pretrained(path); model = AutoModel.from_pretrained(path).cuda().half().eval()
    INSTR = "Represent this sentence for searching relevant passages: "

    @torch.no_grad()
    def enc(texts):
        out = []
        for i in range(0, len(texts), 256):
            b = tok(texts[i:i + 256], padding=True, truncation=True, max_length=512, return_tensors="pt").to("cuda")
            out.append(torch.nn.functional.normalize(model(**b).last_hidden_state[:, 0].float(), dim=-1))
        return torch.cat(out)
    with cache.open("w", encoding="utf-8") as f:
        for n, s in enumerate(samples):
            ch = chunks(s["context"])
            sims = (enc(ch) @ enc([INSTR + s["question"]])[0]).tolist() if ch else []
            f.write(json.dumps({"key": f"{s['task']}/{s['sample_id']}", "chunks": ch, "sim": [round(x, 5) for x in sims]}) + "\n")
            if n % 100 == 0: print(n, flush=True)
    print("SENT_EMBED_DONE")
else:
    import packers as P
    from bridgekv import evidence_text_packing as etp
    tok = L.HFLike(sys.argv[3]); B = int(sys.argv[4])
    C = {r["key"]: r for r in map(json.loads, cache.open(encoding="utf-8"))}
    with open(sys.argv[5], "w", encoding="utf-8") as f:
        for s in samples:
            r = C[f"{s['task']}/{s['sample_id']}"]
            order = sorted(range(len(r["chunks"])), key=lambda i: -r["sim"][i])[:64]
            prefix = etp._prefix(s["question"], False); pieces = []
            for i in order:
                if len(tok("\n\n".join([prefix, *pieces, r["chunks"][i]]).strip()).input_ids) <= B:
                    pieces.append(r["chunks"][i])
            f.write(json.dumps({"task": s["task"], "sample_id": s["sample_id"], "question": s["question"],
                                "answers": s["answers"], "packed_text": P._emit(tok, s["question"], pieces, B)}) + "\n")
