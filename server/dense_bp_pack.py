"""Dense-enhanced Lexical-BridgePack variants (our reimplementation; no original code survives).
The BridgePack pipeline (bridge/path selection, neighbor-32 repair, path order, exact-budget
admission) is unchanged; only the span scores that seed it are swapped:
  recall   : BGE query-span cosine
  union    : max(lexical, BGE) after per-sample normalization (union of the two candidate pools)
  rerank   : 0.5 * lexical + 0.5 * BGE (lexical pool rescored by BGE)
usage: dense_bp_pack.py MODE TOKENIZER OUT [BUDGET]
"""
import json
import sys
from pathlib import Path

W = Path("/diskE/jiaqizhang/BRIDGEPACK_SENSITIVITY_V1_20261025")
MODE, TOKP, OUT = sys.argv[1], sys.argv[2], Path(sys.argv[3])
BUDGET = int(sys.argv[4]) if len(sys.argv) > 4 else 1024
sys.path.insert(0, str(W / "source" / "SlimInfer_snapshot"))
sys.path.insert(0, str(W / "scripts"))
sys.path.insert(0, str(W))
import portable_runner as pr  # noqa: E402
from bridgekv import evidence_text_packing as etp  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from tools.run_bridgepack_exact_budget_repack import _exact_repack  # noqa: E402
from bge_pack import BGE_Embedder, QUERY_INSTRUCTION  # noqa: E402

emb = BGE_Embedder()
_lexical = etp.lexical_scores


def dense_scores(query, candidates):
    q = emb.encode([QUERY_INSTRUCTION + query])[0]
    vecs = emb.encode([c.text for c in candidates])
    bge = etp.normalize_scores([sum(a * b for a, b in zip(q, v)) for v in vecs])
    if MODE == "recall":
        return {c.span_id: bge[i] for i, c in enumerate(candidates)}
    lex = _lexical(query, candidates)
    lexn = etp.normalize_scores([lex[c.span_id] for c in candidates])
    if MODE == "union":
        return {c.span_id: max(lexn[i], bge[i]) for i, c in enumerate(candidates)}
    return {c.span_id: 0.5 * lexn[i] + 0.5 * bge[i] for i, c in enumerate(candidates)}


etp.lexical_scores = dense_scores
ids, samples, _ = pr.load_selection_inputs(W / "assets" / "longbench_g1_data", W / "golden", None)
tok = AutoTokenizer.from_pretrained(TOKP, local_files_only=True, use_fast=True)
with OUT.open("w") as f:
    for t in pr.TASKS:
        for s in samples[t]:
            r = _exact_repack(tok, s, "lexical_bridgepack_path_order", BUDGET, TOKP)
            r["variant"] = f"bge_{MODE}_bridgepack"
            f.write(json.dumps(r) + "\n")
print("DENSE_BP_PACK_DONE", OUT)
