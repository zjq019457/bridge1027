#!/usr/bin/env python3
"""MCU sensitivity audit on held-out HotpotQA/MuSiQue pool (prefill probes only).

Pool: first 64 indices per task, excluding the original 12-sample debug set.
No generation, no scorer, no gold answer usage.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.audit_mcu_sliminfer_counterfactual_utility import (
    TASKS,
    build_runner_prompt_bundle,
    layer_scores_from_record,
    load_dataset2prompt,
    load_final_layer_records,
    load_probe_model,
    load_samples,
    run_gate_audit,
)
from tools.mcu_sliminfer_selection import (
    MCU_DEBUG_EXCLUDE,
    MCU_HELDOUT_POOL_SIZE,
    MCU_SENSITIVE_TOP_K_PER_TASK,
    annotate_mcu_sensitivity_scores,
    heldout_pool_indices,
    heldout_pool_keys,
    run_mcu_counterfactual_selection,
    select_top_sensitive_samples,
    write_frozen_selections_jsonl,
)

DEFAULT_LONGBENCH_DIR = Path("/home/yu-bai/Desktop/jiaqizhang/HeadKV/data/LongBench")
DEFAULT_MODEL_PATH = Path(
    "/home/yu-bai/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/"
    "snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
)

Key = Tuple[str, int]

SENSITIVITY_FIELDS = [
    "task",
    "sample_index",
    "sample_id",
    "candidate_pool_size",
    "num_counterfactual_probes",
    "selection_changed",
    "selected_jaccard_vs_original",
    "utility_margin",
    "hidden_change_mean",
    "logit_kl_mean",
    "entropy_change_mean",
    "top_logit_margin_change_mean",
    "added_block_utility",
    "removed_block_utility",
    "added_minus_removed_utility",
    "query_overlap_delta",
    "redundancy_delta",
    "probe_runtime_ms",
    "budget_preserved",
    "fallback_used",
    "mcu_sensitivity_score",
]


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                field: json.dumps(row.get(field), ensure_ascii=False)
                if isinstance(row.get(field), (list, dict))
                else row.get(field, "")
                for field in fields
            })


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_sample_spec(expected_indices: Mapping[str, Sequence[int]]) -> str:
    return ";".join(
        f"{task}:" + ",".join(str(i) for i in expected_indices[task])
        for task in TASKS
    )


def expected_indices_map() -> Dict[str, List[int]]:
    return {task: heldout_pool_indices(task, MCU_HELDOUT_POOL_SIZE) for task in TASKS}


def audit_row_from_probe(
    task: str,
    sample_index: int,
    probe_result: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "task": task,
        "sample_index": sample_index,
        "sample_id": probe_result.get("sample_id", ""),
        "candidate_pool_size": probe_result.get("candidate_pool_size", 0),
        "num_counterfactual_probes": probe_result.get("num_counterfactual_probes", 0),
        "selection_changed": bool(probe_result.get("selection_changed")),
        "selected_jaccard_vs_original": float(probe_result.get("selected_jaccard_vs_original", 1.0) or 1.0),
        "utility_margin": float(probe_result.get("utility_margin", 0.0) or 0.0),
        "hidden_change_mean": float(probe_result.get("hidden_change_mean", 0.0) or 0.0),
        "logit_kl_mean": float(probe_result.get("logit_kl_mean", 0.0) or 0.0),
        "entropy_change_mean": float(probe_result.get("entropy_change_mean", 0.0) or 0.0),
        "top_logit_margin_change_mean": float(probe_result.get("top_logit_margin_change_mean", 0.0) or 0.0),
        "added_block_utility": probe_result.get("added_block_utility", []),
        "removed_block_utility": probe_result.get("removed_block_utility", []),
        "query_overlap_delta": float(probe_result.get("query_overlap_delta", 0.0) or 0.0),
        "redundancy_delta": float(probe_result.get("redundancy_delta", 0.0) or 0.0),
        "probe_runtime_ms": float(probe_result.get("probe_runtime_ms", 0.0) or 0.0),
        "budget_preserved": bool(probe_result.get("budget_preserved", True)),
        "fallback_used": bool(probe_result.get("fallback_used")),
        "fallback_reason": probe_result.get("fallback_reason", ""),
        "original_selected_block_ids": probe_result.get("original_selected_block_ids", []),
        "mcu_selected_block_ids": probe_result.get("mcu_selected_block_ids", []),
        "removed_block_ids": probe_result.get("removed_block_ids", []),
        "added_block_ids": probe_result.get("added_block_ids", []),
    }


def decide_audit(rows: Sequence[Mapping[str, Any]], stop_reason: Optional[str]) -> Tuple[str, List[str]]:
    if stop_reason:
        return "BUG_STOP_FIX_REQUIRED", [stop_reason]
    required = {
        "utility_margin",
        "logit_kl_mean",
        "hidden_change_mean",
        "top_logit_margin_change_mean",
        "mcu_sensitivity_score",
    }
    for row in rows:
        if any(field not in row for field in required):
            return "BUG_STOP_FIX_REQUIRED", ["sensitivity_audit_field_missing"]
    for task in TASKS:
        task_rows = [row for row in rows if row.get("task") == task]
        if len(task_rows) < MCU_SENSITIVE_TOP_K_PER_TASK:
            return "BUG_STOP_FIX_REQUIRED", [f"insufficient_{task}_samples={len(task_rows)}"]
    return "MCU_SENSITIVITY_AUDIT_PASS", ["heldout_pool_scored"]


def build_report(
    output_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    selected: Mapping[str, Sequence[Mapping[str, Any]]],
    decision: str,
    reasons: Sequence[str],
) -> str:
    lines = [
        "# MCU Sensitivity Held-out Pool Audit",
        "",
        f"- Output dir: `{output_dir}`",
        f"- Decision: **{decision}**",
        f"- Pool size per task: {MCU_HELDOUT_POOL_SIZE}",
        f"- Excluded debug indices: `{MCU_DEBUG_EXCLUDE}`",
        f"- Samples audited: {len(rows)}",
        "",
        "## Selection rule",
        "- Rank by fixed `mcu_sensitivity_score` within held-out pool.",
        "- Top 10 HotpotQA + top 10 MuSiQue.",
        "- No gold answer, no score, no prediction, no task-label gate.",
        "",
        "## Sensitivity formula",
        "```",
        "mcu_sensitivity_score =",
        "  z(utility_margin)",
        "  + z(logit_kl_mean)",
        "  + z(hidden_change_mean)",
        "  + 0.5*z(abs(top_logit_margin_change_mean))",
        "  + 0.5*z(added_minus_removed_utility)",
        "  - 0.5*z(fallback_used)",
        "```",
        "",
        "## Selected 20 samples",
    ]
    for task in TASKS:
        lines.append(f"### {task}")
        for row in selected[task]:
            lines.append(
                f"- {row['sample_index']} | score={float(row['mcu_sensitivity_score']):.4f} "
                f"| id=`{row.get('sample_id', '')}` | margin={float(row.get('utility_margin', 0.0)):.3f}"
            )
        lines.append("")
    lines.extend(["## Decision reasons"])
    for reason in reasons:
        lines.append(f"- {reason}")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit MCU sensitivity on held-out pool.")
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--longbench_dir", type=Path, default=DEFAULT_LONGBENCH_DIR)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--skip_gate_run", action="store_true")
    parser.add_argument("--score_dump_path", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    expected_indices = expected_indices_map()
    keys = heldout_pool_keys()
    sample_spec = build_sample_spec(expected_indices)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (REPO_ROOT / f"results/mcu_sensitivity_heldout_audit_{timestamp}")
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    gate_dir = output_dir / "run_sliminfer_original_gate_audit"
    score_dump_path = args.score_dump_path or gate_dir / "uasr_score_dump.jsonl"
    stop_reason: Optional[str] = None
    audit_rows: List[Dict[str, Any]] = []

    try:
        if args.skip_gate_run:
            if not score_dump_path.exists():
                raise RuntimeError(f"--skip_gate_run but score dump missing: {score_dump_path}")
        else:
            paths = run_gate_audit(args, output_dir, sample_spec)
            score_dump_path = paths["score_dump_path"]

        final_records = load_final_layer_records(score_dump_path, keys)
        missing = [key for key in keys if key not in final_records]
        if missing:
            raise RuntimeError(f"missing final layer records: {missing[:8]} ... total={len(missing)}")

        tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), trust_remote_code=True)
        prompt_map = load_dataset2prompt()
        samples_cache = {task: load_samples(args.longbench_dir, task) for task in TASKS}

        model = load_probe_model(args.model_path, args.device)
        for task, sample_index in keys:
            key = (task, sample_index)
            final_record = final_records[key]
            sample = samples_cache[task][sample_index]
            input_ids, query_text, block_texts, token_lengths = build_runner_prompt_bundle(
                tokenizer, prompt_map[task], sample, task,
            )
            original_selected = {int(x) for x in (final_record.get("selected_block_ids") or [])}
            all_scores = layer_scores_from_record(final_record)
            sample_id = final_record.get("sample_id", sample.get("_id", sample_index))

            try:
                probe_result = run_mcu_counterfactual_selection(
                    model,
                    tokenizer,
                    input_ids,
                    query_text,
                    block_texts,
                    token_lengths,
                    original_selected,
                    all_scores,
                    sample_id=sample_id,
                )
            except torch.cuda.OutOfMemoryError as exc:
                stop_reason = f"CUDA OOM at {task}:{sample_index}: {exc}"
                break

            row = audit_row_from_probe(task, sample_index, probe_result)
            audit_rows.append(row)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except RuntimeError as exc:
        stop_reason = str(exc)
        if "OOM" not in stop_reason:
            raise

    annotate_mcu_sensitivity_scores(audit_rows)
    decision, reasons = decide_audit(audit_rows, stop_reason)

    selected: Dict[str, List[Dict[str, Any]]] = {}
    if decision == "MCU_SENSITIVITY_AUDIT_PASS":
        selected = select_top_sensitive_samples(audit_rows)

    write_csv(output_dir / "mcu_sensitivity_heldout_audit.csv", audit_rows, SENSITIVITY_FIELDS)

    selected_indices = {task: [int(row["sample_index"]) for row in selected.get(task, [])] for task in TASKS}
    selected_payload = {
        "selection_rule": (
            "top 10 per task by mcu_sensitivity_score from held-out pool; "
            "no gold/score/prediction/task-label gate"
        ),
        "excluded_debug_indices": {task: sorted(MCU_DEBUG_EXCLUDE[task]) for task in TASKS},
        "heldout_pool_size": MCU_HELDOUT_POOL_SIZE,
        "sensitivity_formula": (
            "z(utility_margin)+z(logit_kl_mean)+z(hidden_change_mean)"
            "+0.5*z(abs(top_logit_margin_change_mean))"
            "+0.5*z(added_minus_removed_utility)-0.5*z(fallback_used)"
        ),
        "expected_indices": selected_indices,
        "selected_sample_ids": {
            task: [str(row.get("sample_id", "")) for row in selected.get(task, [])]
            for task in TASKS
        },
        "selected_rows": {
            task: selected.get(task, [])
            for task in TASKS
        },
    }
    write_json(output_dir / "mcu_sensitive_sample_selection.json", selected_payload)

    if selected:
        frozen_map: Dict[str, Dict[str, Any]] = {}
        for task in TASKS:
            for row in selected[task]:
                key = f"{task}:{int(row['sample_index'])}"
                frozen_map[key] = {
                    **row,
                    "selected": row.get("mcu_selected_block_ids", []),
                    "budget_preserved": row.get("budget_preserved", True),
                    "selection_changed": row.get("selection_changed", False),
                }
        write_frozen_selections_jsonl(output_dir / "mcu_sensitive_frozen_selections.jsonl", frozen_map)

    report = build_report(output_dir, audit_rows, selected, decision, reasons)
    (output_dir / "mcu_sensitivity_audit_report.md").write_text(report, encoding="utf-8")
    (output_dir / "decision.md").write_text(
        f"# Decision\n\n**{decision}**\n\n" + "\n".join(f"- {r}" for r in reasons) + "\n",
        encoding="utf-8",
    )
    write_json(output_dir / "run_config.json", {
        "sample_spec": sample_spec,
        "expected_indices": expected_indices,
        "pool_keys_count": len(keys),
        "score_dump_path": str(score_dump_path),
        "quality_generation_performed": False,
        "scorer_invoked": False,
        "stop_reason": stop_reason,
    })

    print(report)
    print(f"Wrote {output_dir / 'decision.md'}")


if __name__ == "__main__":
    main()
