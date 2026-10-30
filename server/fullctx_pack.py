"""Full-context (LongBench middle truncation to the model window) and 1,024-token
middle-truncation baseline traces, reusing each default trace's prompt wrapper.
usage: fullctx_pack.py MODEL_TOK DEFAULT_TRACE OUT_DIR WINDOW"""
import json, sys
from pathlib import Path
from transformers import AutoTokenizer

W = Path("/diskE/jiaqizhang/BRIDGEPACK_SENSITIVITY_V1_20261025")
tok = AutoTokenizer.from_pretrained(sys.argv[1])
out = Path(sys.argv[3]); out.mkdir(parents=True, exist_ok=True)
WIN = int(sys.argv[4])
ctx = {}
for t in ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", "narrativeqa"]:
    for l in open(W / "assets/longbench_g1_data" / f"{t}.jsonl"):
        r = json.loads(l); ctx[(t, r["_id"])] = r["context"]


def mid(ids, n):
    return ids if len(ids) <= n else ids[:n // 2] + ids[-(n - n // 2):]


def ntok(s):
    return len(tok(s, add_special_tokens=False).input_ids)


f_full = open(out / "fullctx.jsonl", "w"); f_tr = open(out / "trunc1024.jsonl", "w")
over = 0; lens = []
for l in open(sys.argv[2]):
    x = json.loads(l); c = ctx[(x["task"], x["sample_id"])]
    pre = x["packed_text"].split("\n\n")[0] + "\n\nSelected evidence:"
    assert x["generation_prompt"].count(x["packed_text"]) == 1
    wrap = x["generation_prompt"].replace(x["packed_text"], "\x00")
    cids = tok(c, add_special_tokens=False).input_ids
    full_n = ntok(wrap.replace("\x00", pre + "\n\n" + c)); lens.append(full_n); over += full_n > WIN
    t_full = pre + "\n\n" + tok.decode(mid(cids, WIN - 160 - (full_n - len(cids))))
    n = 1024 - ntok(pre + "\n\n")
    while True:
        t_tr = pre + "\n\n" + tok.decode(mid(cids, n))
        if ntok(t_tr) <= 1024: break
        n -= 8
    for f, t in ((f_full, t_full), (f_tr, t_tr)):
        y = dict(x); y["packed_text"] = t; y["generation_prompt"] = wrap.replace("\x00", t)
        f.write(json.dumps(y) + "\n")
lens.sort(); print("n", len(lens), "over_window", over, "median_full_prompt", lens[len(lens) // 2])
