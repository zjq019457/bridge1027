"""BridgePack with Provence as the span scorer: the only change from the default is that each
candidate's final_score (lexical z(BM25)/z(TF-IDF) blend) is replaced by
  MODE=replace : z(Provence reranking score of the span)
  MODE=fuse    : 0.5 * lexical + 0.5 * z(Provence)
Pairing, neighbor repair, source order and exact admission are the frozen defaults.
usage: provbp_pack.py MODE TOKENIZER DEFAULT_TRACE OUT
"""
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

W = Path("/diskE/jiaqizhang/BRIDGEPACK_SENSITIVITY_V1_20261025")
sys.path.insert(0, str(W / "source" / "SlimInfer_snapshot"))
from bridgekv import bridgepack_cpu_extensions as ext  # noqa: E402
from bridgekv import evidence_text_packing as etp  # noqa: E402
from tools.bridgepack_g1 import run_g1 as g1  # noqa: E402

MODE, TOKP, TRACE, OUT = sys.argv[1:5]
BUDGET = 1024
tok = AutoTokenizer.from_pretrained(TOKP, use_fast=True)
pv = AutoModel.from_pretrained(W / "models_provence", trust_remote_code=True).to("cuda").eval()
lexical = etp.lexical_scored_candidates


def provence_scored(query, candidates):
    with torch.no_grad():
        res = pv.process(query, [[c.text for c in candidates]], title=None, threshold=0.1,
                         enable_warnings=False, batch_size=64)
    z = etp.normalize_scores([float(s) for s in res["reranking_score"][0]])
    lex = lexical(query, candidates)
    scores = {c.span_id: (z[i] if MODE == "replace" else 0.5 * lex[i].final_score + 0.5 * z[i])
              for i, c in enumerate(candidates)}
    return etp._replace_scores(candidates, scores)


ext.lexical_scored_candidates = provence_scored

ctx = {}
for t in ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", "narrativeqa"]:
    for l in open(W / "assets/longbench_g1_data" / f"{t}.jsonl"):
        r = json.loads(l); ctx[(t, r["_id"])] = (r["input"], r["context"])

with open(TRACE) as fin, open(OUT, "w") as f:
    for l in fin:
        x = json.loads(l)
        q, c = ctx[(x["task"], x["sample_id"])]
        fr = ext.freeze_lexical_bridgepack_selection(x["sample_id"], x["task"], q, c, BUDGET, tokenizer=tok)
        packed_obj, _ = ext.reorder_frozen(fr, "source_order", seed=20260727, tokenizer=tok)
        ordered = [(next(s for s in fr["candidates"] if s.span_id == i), "support") for i in packed_obj.selected_span_ids]
        p = g1._exact_pack_from_ordered(tok, x["sample_id"], x["task"], q, c, "same_spans_source_order",
                                        ordered, fr["page_map"], 32, True, "source_order")
        y = dict(x); y["variant"] = f"provence_bp_{MODE}"; y["packed_text"] = p["packed_text"]
        y["generation_prompt"] = x["generation_prompt"].replace(x["packed_text"], p["packed_text"])
        y["model_token_count"] = len(tok(p["packed_text"], add_special_tokens=False).input_ids)
        f.write(json.dumps(y) + "\n"); f.flush()
print("PROVBP_DONE", MODE)
