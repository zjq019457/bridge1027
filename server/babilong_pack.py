"""Pack BABILong (qa1-qa3, 16k/32k) under the exact 1,024 downstream-token budget.

Methods: BridgePack (frozen default selector + source-order exact admission, as in the
main runs) and BM25 chunk packing (same candidate spans, BM25 ranking, exact admission,
source order). Writes one trace per method in the gen_runner format.
usage: babilong_pack.py TOKENIZER_PATH OUT_DIR N_PER_TASK
"""
import hashlib
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

W = Path("/diskE/jiaqizhang/BRIDGEPACK_SENSITIVITY_V1_20261025")
sys.path.insert(0, str(W / "source" / "SlimInfer_snapshot"))
TOKP, OUT, N = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3])
BUDGET = 1024
LENGTHS, TASKS = ["16k", "32k"], ["qa1", "qa2", "qa3"]
PROMPT = ("I will give you context with the facts about positions of different persons hidden in some "
          "random text and a question. You need to answer the question based only on the information "
          "from the facts.\n\n<context>\n{context}\n</context>\n\nQuestion: {question}\n"
          "Answer with a single word naming the location.")

_tok = None


def _init():
    global _tok
    from transformers import AutoTokenizer
    _tok = AutoTokenizer.from_pretrained(TOKP, use_fast=True)


def _n(text):
    return len(_tok(text, add_special_tokens=False).input_ids)


def _row(s, method, packed_text, n_spans):
    ids = _tok(packed_text, add_special_tokens=False).input_ids
    return {"task": s["task"], "sample_id": s["sample_id"], "variant": method, "budget": BUDGET,
            "packed_text": packed_text, "model_token_count": len(ids), "selected_spans": n_spans,
            "generation_prompt": PROMPT.format(context=packed_text, question=s["question"]),
            "answers": [s["target"]], "all_classes": None, "question": s["question"]}


def bridgepack(s):
    from bridgekv.bridgepack_cpu_extensions import freeze_lexical_bridgepack_selection, reorder_frozen
    from tools.bridgepack_g1 import run_g1 as g1
    fr = freeze_lexical_bridgepack_selection(s["sample_id"], s["task"], s["question"], s["context"], BUDGET, tokenizer=_tok)
    packed_obj, _ = reorder_frozen(fr, "source_order", seed=20260727, tokenizer=_tok)
    ordered = [(next(c for c in fr["candidates"] if c.span_id == i), "support") for i in packed_obj.selected_span_ids]
    p = g1._exact_pack_from_ordered(_tok, s["sample_id"], s["task"], s["question"], s["context"],
                                    "same_spans_source_order", ordered, fr["page_map"], 32, True, "source_order")
    return _row(s, "bridgepack", p["packed_text"], len(p["selected_span_ids"]))


def bm25(s):
    from bridgekv import evidence_text_packing as etp
    cands, page_map = etp.build_candidate_spans(s["sample_id"], s["task"], s["question"], s["context"])
    sc = etp.bm25_scores(s["question"], cands)
    ranked = sorted(cands, key=lambda c: (-sc[c.span_id], c.source_order_index))
    prefix = etp._prefix(s["question"], False)
    sel = []
    for c in ranked:
        if any(etp._overlaps(c, p, max_fraction=0.55) for p in sel):
            continue
        trial = sorted(sel + [c], key=lambda x: x.source_order_index)
        if _n("\n\n".join([prefix] + [x.text for x in trial])) <= BUDGET:
            sel.append(c)
            if _n("\n\n".join([prefix] + [x.text for x in sel])) >= 0.95 * BUDGET:
                break
    sel.sort(key=lambda x: x.source_order_index)
    return _row(s, "bm25", "\n\n".join([prefix] + [x.text for x in sel]).strip(), len(sel))


def one(s):
    return [bridgepack(s), bm25(s)]


if __name__ == "__main__":
    import pandas as pd
    samples = []
    for L in LENGTHS:
        for t in TASKS:
            d = pd.read_parquet(W / "babilong" / L / f"{t}.parquet").head(N)
            for i, r in d.iterrows():
                samples.append({"task": f"babilong_{t}_{L}", "sample_id": f"{t}_{L}_{i}",
                                "context": r["input"], "question": r["question"].strip(), "target": r["target"]})
    OUT.mkdir(parents=True, exist_ok=True)
    with Pool(int(os.environ.get("NPROC", "16")), initializer=_init) as pool:
        res = pool.map(one, samples, chunksize=2)
    for k, m in enumerate(["bridgepack", "bm25"]):
        with (OUT / f"{m}.jsonl").open("w") as f:
            for rs in res:
                f.write(json.dumps(rs[k]) + "\n")
        lens = [rs[k]["model_token_count"] for rs in res]
        print(m, "n", len(lens), "mean_len", sum(lens) / len(lens), "max", max(lens))
