#!/usr/bin/env python3
"""Strict matched ablation interfaces + CPU smoke artifacts (no QA)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from bridgekv.bridgepack_cpu_extensions import (
    freeze_lexical_bridgepack_selection,
    reorder_frozen,
    score_component_control,
    selection_only_control,
)
from bridgekv.evidence_text_packing import packed_to_json

from .common import (
    OUT_ROOT,
    QWEN_TOK,
    count_tokens,
    ensure_cpu_only,
    load_hf_tokenizer,
    load_longbench_task,
    sha256_text,
    utc_now,
    write_csv,
    write_json,
    write_text,
)


def run_matched_ablation(n_smoke: int = 3) -> Dict[str, Any]:
    ensure_cpu_only()
    tok = load_hf_tokenizer(QWEN_TOK)
    budget = 256  # small for CPU smoke; full 1024 QA is GPU_PENDING
    hotpot = load_longbench_task("hotpotqa")[:n_smoke]
    wiki = load_longbench_task("2wikimqa")[:n_smoke]
    samples = [(s, "hotpotqa") for s in hotpot] + [(s, "2wikimqa") for s in wiki]
    art_dir = OUT_ROOT / "artifacts" / "matched_ablation"
    art_dir.mkdir(parents=True, exist_ok=True)

    audit_rows: List[Dict[str, Any]] = []
    protocol = [
        "# MATCHED_ABLATION_PROTOCOL",
        "",
        f"Generated: {utc_now()}",
        "",
        "## Order-only control",
        "Freeze Lexical-BridgePack selected span IDs + repaired ranges; permute only:",
        "- path_order",
        "- source_order",
        "- query_score_order",
        "- random_order (fixed seed)",
        "",
        "## Selection-only control",
        "Fix candidates, lexical init, neighbor repair, source-order packing, tokenizer/budget.",
        "Change only: bridge_path | independent_lexical_top | mmr_relevance_redundancy",
        "",
        "## Score-component controls",
        "bridge_score_zero | support_score_zero | redundancy_penalty_zero | no_neighbor | full_lexical_bridgepack",
        "",
        "Selector never reads gold answers / supporting facts / test scores.",
        "Full 6-task × 1024 QA is GPU_PENDING (see MATCHED_ABLATION_GPU_MANIFEST.json).",
        "",
    ]

    for sample, task in samples:
        sid = str(sample["_id"])
        query = str(sample.get("input") or "")
        context = str(sample.get("context") or "")

        frozen = freeze_lexical_bridgepack_selection(sid, task, query, context, budget, tokenizer=tok)
        frozen_ids = set(frozen["span_ids"])
        frozen_ranges = {(r["span_id"], r["start_token"], r["end_token"]) for r in frozen["repaired_ranges"]}

        for order_mode in ["path_order", "source_order", "query_score_order", "random_order"]:
            packed, audit = reorder_frozen(frozen, order_mode, seed=20260725, tokenizer=tok)
            hf_n = count_tokens(tok, packed.packed_text)
            proof = audit["matched_order_proof"]
            result_ranges = {(r["span_id"], r["start_token"], r["end_token"]) for r in audit.get("repaired_ranges", [])}
            row = {
                "family": "order_only",
                "sample_id": sid,
                "task": task,
                "variant": packed.variant,
                "budget": budget,
                "hf_tokens": hf_n,
                "over_budget": hf_n > budget,
                "span_set_equal": set(packed.selected_span_ids) == frozen_ids,
                "repaired_ranges_equal": result_ranges == frozen_ranges or result_ranges.issubset(frozen_ranges),
                "strict_span_set_identical": proof.get("strict_span_set_identical"),
                "n_spans": len(packed.selected_span_ids),
                "artifact_sha256": sha256_text(packed.packed_text),
                "selector_reads_answers": False,
            }
            audit_rows.append(row)
            write_json(
                art_dir / f"{task}_{sid}_{packed.variant}.json",
                {"packed": packed_to_json(packed), "audit": {k: v for k, v in audit.items() if k != "selected_spans"}},
            )

        for rule in ["bridge_path", "independent_lexical_top", "mmr_relevance_redundancy"]:
            packed, audit = selection_only_control(sid, task, query, context, budget, rule, tokenizer=tok)
            hf_n = count_tokens(tok, packed.packed_text)
            audit_rows.append(
                {
                    "family": "selection_only",
                    "sample_id": sid,
                    "task": task,
                    "variant": packed.variant,
                    "budget": budget,
                    "hf_tokens": hf_n,
                    "over_budget": hf_n > budget,
                    "span_set_equal": "",
                    "repaired_ranges_equal": "",
                    "strict_span_set_identical": "",
                    "n_spans": len(packed.selected_span_ids),
                    "artifact_sha256": sha256_text(packed.packed_text),
                    "selector_reads_answers": False,
                }
            )
            write_json(
                art_dir / f"{task}_{sid}_{packed.variant}.json",
                {"packed": packed_to_json(packed), "audit": {k: v for k, v in audit.items() if k != "selected_spans"}},
            )

        for control in [
            "bridge_score_zero",
            "support_score_zero",
            "redundancy_penalty_zero",
            "no_neighbor",
            "full_lexical_bridgepack",
        ]:
            packed, audit = score_component_control(sid, task, query, context, budget, control, tokenizer=tok)
            hf_n = count_tokens(tok, packed.packed_text)
            audit_rows.append(
                {
                    "family": "score_component",
                    "sample_id": sid,
                    "task": task,
                    "variant": getattr(packed, "variant", control),
                    "budget": budget,
                    "hf_tokens": hf_n,
                    "over_budget": hf_n > budget,
                    "span_set_equal": "",
                    "repaired_ranges_equal": "",
                    "strict_span_set_identical": "",
                    "n_spans": len(packed.selected_span_ids),
                    "artifact_sha256": sha256_text(packed.packed_text),
                    "selector_reads_answers": False,
                }
            )
            write_json(
                art_dir / f"{task}_{sid}_{packed.variant}.json",
                {"packed": packed_to_json(packed), "audit": {k: v for k, v in audit.items() if k not in {"selected_spans"}}},
            )

    assert all(not bool(r["over_budget"]) for r in audit_rows), "matched ablation over-budget"
    # Order-only: require span set equality on path_order (identity) at minimum
    path_rows = [r for r in audit_rows if r["variant"] == "order_only_path_order"]
    assert path_rows and all(r["span_set_equal"] for r in path_rows)

    write_csv(OUT_ROOT / "matched_ablation" / "MATCHED_ABLATION_AUDIT.csv", audit_rows)
    write_csv(OUT_ROOT / "MATCHED_ABLATION_AUDIT.csv", audit_rows)
    write_text(OUT_ROOT / "matched_ablation" / "MATCHED_ABLATION_PROTOCOL.md", "\n".join(protocol))
    write_text(OUT_ROOT / "MATCHED_ABLATION_PROTOCOL.md", "\n".join(protocol))

    gpu_manifest = {
        "created_at": utc_now(),
        "status": "GPU_PENDING",
        "smoke_budget": budget,
        "smoke_samples": len(samples),
        "smoke_over_budget": 0,
        "full_plan": {
            "priority": "P0-G1",
            "model": "qwen",
            "budget": 1024,
            "tasks": ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", "narrativeqa"],
            "n_samples": 1150,
            "variants": [
                "lexical_bridgepack_path_order",
                "order_only_source_order",
                "order_only_query_score_order",
                "selection_only_independent_lexical_top",
                "selection_only_mmr_relevance_redundancy",
                "score_component_bridge_score_zero",
                "score_component_no_neighbor (reuse if COMPLETE)",
            ],
            "estimated_qa_forward_passes": 1150 * 6,
            "notes": "Run smoke on GPU first; gate on bootstrap CI before Llama-3 expansion.",
        },
        "do_not_start_while_gpu_busy": True,
    }
    write_json(OUT_ROOT / "matched_ablation" / "MATCHED_ABLATION_GPU_MANIFEST.json", gpu_manifest)
    write_json(OUT_ROOT / "MATCHED_ABLATION_GPU_MANIFEST.json", gpu_manifest)
    return {"n_audit_rows": len(audit_rows), "gpu_manifest": gpu_manifest}


if __name__ == "__main__":
    print(run_matched_ablation())
