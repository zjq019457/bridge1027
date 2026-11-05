"""Full-context reference traces: the whole LongBench context, middle-truncated to MAXTOK model
tokens (LongBench practice). usage: python fullctx.py TOKENIZER_DIR OUT.jsonl [MAXTOK]"""
import json, sys
import localpack as L
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(sys.argv[1])
m = int(sys.argv[3]) if len(sys.argv) > 3 else 30000
with open(sys.argv[2], "w", encoding="utf-8") as f:
    for s in (L.load_inf() if __import__("os").environ.get("BP_SET") == "inf" else L.load_samples()):
        ids = tok(s["context"], add_special_tokens=False).input_ids
        ctx = s["context"] if len(ids) <= m else tok.decode(ids[:m // 2]) + tok.decode(ids[-m // 2:])
        f.write(json.dumps({"task": s["task"], "sample_id": s["sample_id"], "packed_text": ctx,
                            "question": s["question"], "answers": s["answers"]}) + "\n")
