"""Selective Context, expensive half: phrase-level lexical units and their GPT-2 self-information for every
test instance (official _lexical_unit, sentence-level self-information). The cheap half (masking at a given
ratio) is sel_ctx2 in packers.py, so one cache serves every tokenizer and budget.
usage: python sc_units.py NPROC OUT.jsonl   (resumable)"""
import json, os, re, sys
from multiprocessing import Pool
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import localpack as L
import packers


def one(s):
    sc = packers._sc()
    ctx = sc.beautify_context(s["context"])
    sents = [x.strip() for x in re.split(sc.sent_tokenize_pattern, ctx) if x.strip()]
    _, ph, _ = sc._lexical_unit(sents)
    return json.dumps({"key": f"{s['task']}/{s['sample_id']}", "text": ph.text, "info": ph.self_info})


if __name__ == "__main__":
    n, out = int(sys.argv[1]), sys.argv[2]
    done = set()
    if os.path.exists(out):
        done = {json.loads(l)["key"] for l in open(out, encoding="utf-8")}
    todo = [s for s in L.load_samples() if f"{s['task']}/{s['sample_id']}" not in done]
    print("todo", len(todo), flush=True)
    with Pool(n) as p, open(out, "a", encoding="utf-8") as f:
        for i, line in enumerate(p.imap_unordered(one, todo, chunksize=1)):
            f.write(line + "\n"); f.flush()
            if i % 50 == 0:
                print(i, flush=True)
    print("SC_UNITS_DONE", flush=True)
