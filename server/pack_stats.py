"""CPU-only packing statistics for the default Lexical-BridgePack selector.

For each sample and budget: spans chosen by the bridge selector, spans kept after exact
downstream-token packing, whether the first span was truncated, packed length.
"""
import json
import os
import statistics
import sys
from multiprocessing import Pool
from pathlib import Path

W = Path(sys.argv[1])
MODEL = sys.argv[2]
OUT = Path(sys.argv[3])
BUDGETS = [256, 512, 1024]
sys.path.insert(0, str(W / "source" / "SlimInfer_snapshot"))
sys.path.insert(0, str(W / "scripts"))

_tok = None


def _init():
    global _tok
    from transformers import AutoTokenizer
    _tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=True, use_fast=True)


def one(sample):
    from bridgekv.bridgepack_cpu_extensions import freeze_lexical_bridgepack_selection
    rows = []
    for b in BUDGETS:
        fr = freeze_lexical_bridgepack_selection(sample["sample_id"], sample["task"], sample["question"],
                                                 sample["context"], b, tokenizer=_tok)
        n_sel = len(fr["ordered_spans"])
        n_kept = len(fr["span_ids"])
        packed = fr["packed"]
        rows.append({
            "task": sample["task"], "sample_id": sample["sample_id"], "budget": b,
            "selector_spans": n_sel, "kept_spans": n_kept, "dropped": max(0, n_sel - n_kept),
            "truncated": bool(packed.truncated),
            "packed_tokens": len(_tok(packed.packed_text, add_special_tokens=False).input_ids),
            "pairs": fr["path_audit"].get("path_count"),
        })
    return rows


if __name__ == "__main__":
    import portable_runner as pr
    ids, samples, _ = pr.load_selection_inputs(W / "assets" / "longbench_g1_data", W / "golden", None)
    flat = [s for t in pr.TASKS for s in samples[t]]
    with Pool(int(os.environ.get("NPROC", "16")), initializer=_init) as pool:
        rows = [r for rs in pool.imap(one, flat, chunksize=4) for r in rs]
    OUT.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    summary = {}
    for b in BUDGETS:
        rs = [r for r in rows if r["budget"] == b]
        lens = sorted(r["packed_tokens"] for r in rs)
        n = len(rs)
        summary[b] = {
            "n": n,
            "selector_spans_mean": statistics.mean(r["selector_spans"] for r in rs),
            "kept_spans_mean": statistics.mean(r["kept_spans"] for r in rs),
            "pairs_mean": statistics.mean(r["pairs"] or 0 for r in rs),
            "share_with_at_most_one_pair": sum((r["pairs"] or 0) <= 1 for r in rs) / n,
            "samples_with_any_drop": sum(r["dropped"] > 0 for r in rs) / n,
            "dropped_spans_share": sum(r["dropped"] for r in rs) / max(1, sum(r["selector_spans"] for r in rs)),
            "truncated_rate": sum(r["truncated"] for r in rs) / n,
            "len_mean": statistics.mean(lens), "len_p50": lens[n // 2], "len_p95": lens[int(0.95 * n) - 1],
            "len_max": lens[-1], "over_budget": sum(l > b for l in lens),
        }
    print(json.dumps(summary, indent=2))
    OUT.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
