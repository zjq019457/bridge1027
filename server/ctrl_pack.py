"""Ablation controls under the exact model-token budget, via the repo s control functions.
KIND:
  sel_bridge_path | sel_independent_lexical_top | sel_mmr_relevance_redundancy
      selection_only_control: candidates, lexical init, neighbor-32 repair, source-order packing
      and budget fixed; only the selection rule changes (joint pairing vs. per-span ranking).
  lbp_no_neighbor : score_component_control("no_neighbor"), Lexical-BridgePack w/o neighbor repair
  bp_no_neighbor  : BridgePack w/o neighbor repair (same, without lexical initialization)
usage: ctrl_pack.py KIND TOKENIZER OUT [BUDGET]
"""
import json
import sys
from pathlib import Path

W = Path("/diskE/jiaqizhang/BRIDGEPACK_SENSITIVITY_V1_20261025")
KIND, TOKP, OUT = sys.argv[1], sys.argv[2], Path(sys.argv[3])
BUDGET = int(sys.argv[4]) if len(sys.argv) > 4 else 1024
sys.path.insert(0, str(W / "source" / "SlimInfer_snapshot"))
sys.path.insert(0, str(W / "scripts"))
import portable_runner as pr  # noqa: E402
from bridgekv import bridgepack_cpu_extensions as ext  # noqa: E402
from bridgekv import evidence_text_packing as etp  # noqa: E402
from bridgekv.evidence_text_packing import packed_to_json  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from tools.run_bridgepack_exact_budget_repack import _generation_prompt  # noqa: E402


def pack(s):
    args = (s["sample_id"], s["task"], s["question"], s["context"], BUDGET)
    if KIND.startswith("sel_"):
        return ext.selection_only_control(*args, KIND[4:], tokenizer=tok)
    if KIND == "lbp_no_neighbor":
        return ext.score_component_control(*args, "no_neighbor", tokenizer=tok)
    candidates, page_map = etp.build_candidate_spans(*args[:4])
    ordered, _ = etp._select_bridge_paths(s["question"], candidates, page_map, BUDGET, use_tags=False, neighbor_size=0)
    return ext._pack_ordered(*args[:3], "bp_no_neighbor", BUDGET, ordered, page_map, neighbor_size=0,
                             ordering="bridge_path_order", used_bridge_path=True, tokenizer=tok)


ids, samples, _ = pr.load_selection_inputs(W / "assets" / "longbench_g1_data", W / "golden", None)
tok = AutoTokenizer.from_pretrained(TOKP, local_files_only=True, use_fast=True)
with OUT.open("w") as f:
    for t in pr.TASKS:
        for s in samples[t]:
            packed, _ = pack(s)
            r = packed_to_json(packed)
            n = len(tok(r["packed_text"], add_special_tokens=False).input_ids)
            assert n <= BUDGET, (s["sample_id"], n)
            r.update({"variant": KIND, "question": s["question"], "answers": s["answers"],
                      "all_classes": s["all_classes"], "model_token_count": n,
                      "generation_prompt": _generation_prompt(s["task"], r["packed_text"], s["question"])})
            f.write(json.dumps(r) + "\n")
print("CTRL_PACK_DONE", OUT)
