#!/usr/bin/env python3
"""CPU-only audit for Uncertainty-Aware SlimInfer Repair (UASR).

UASR asks whether SlimInfer's hard top-K cutoff is unstable near the boundary.
This script only reads existing artifacts.  It does not run a model, does not
generate predictions, and does not modify SlimInfer's frozen base selector.

If selected/pruned per-block score details are unavailable, the audit fail-closes
and records the missing fields instead of fabricating cutoff margins.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


DEFAULT_HISTORICAL_DIR = Path("results/drm_budget_neutral_swap_multihop200_20260706_042717")
METHOD_ORIGINAL = "sliminfer_original"
METHOD_DRM_SWAP = "budget_neutral_drm_swap"
METHOD_RANDOM_SWAP = "random_budget_neutral_swap"
METHOD_LARGER = "sliminfer_larger"
BASELINE_METHODS = [METHOD_ORIGINAL, METHOD_RANDOM_SWAP, METHOD_DRM_SWAP, METHOD_LARGER]

TINY_SAMPLE_INDICES = {
    "hotpotqa": [0, 1, 2, 3, 4, 5, 41, 56, 57, 117],
    "musique": [0, 1, 2, 3, 4, 20, 50, 54, 56, 134],
}

TAU_MARGIN = 0.05
EPS_NEAR_CUTOFF = 0.05
QUERY_OVERLAP_MIN = 0.10
REDUNDANCY_MIN = 0.50


JsonRow = Dict[str, Any]
Key = Tuple[str, int]


def read_jsonl(path: Path) -> List[JsonRow]:
    if not path.exists():
        raise RuntimeError(f"missing JSONL: {path}")
    rows: List[JsonRow] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: serialize_csv(row.get(field, "")) for field in fields})


def serialize_csv(value: Any) -> Any:
    if isinstance(value, (list, dict, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return value


def key_of(row: Mapping[str, Any]) -> Key:
    return str(row["task"]), int(row["sample_index"])


def as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null", "nan", "unavailable"}:
        return True
    if isinstance(value, float) and not math.isfinite(value):
        return True
    return False


def load_historical(historical_dir: Path) -> Dict[str, Dict[Key, JsonRow]]:
    by_method: Dict[str, Dict[Key, JsonRow]] = defaultdict(dict)
    for row in read_jsonl(historical_dir / "audit" / "per_sample_audit.jsonl"):
        method = str(row.get("method_key") or "")
        if method:
            by_method[method][key_of(row)] = row
    missing = [method for method in BASELINE_METHODS if method not in by_method]
    if missing:
        raise RuntimeError(f"historical audit missing methods: {missing}")
    return dict(by_method)


def load_score_dump(score_dump_dir: Path | None) -> Dict[Key, JsonRow]:
    if score_dump_dir is None:
        return {}
    path = score_dump_dir / "score_dump_audit.jsonl"
    if not path.exists():
        return {}
    grouped: Dict[Key, List[JsonRow]] = defaultdict(list)
    for row in read_jsonl(path):
        try:
            grouped[key_of(row)].append(row)
        except Exception:
            continue
    selected: Dict[Key, JsonRow] = {}
    for key, rows in grouped.items():
        # Use the last/highest prune layer as the final SlimInfer active set.
        selected[key] = sorted(
            rows,
            key=lambda row: (
                int(row.get("layer_id", -1)),
                0 if str(row.get("phase", "")) == "prefill" else 1,
            ),
        )[-1]
    return selected


def load_longbench_queries(historical_dir: Path, tasks: Iterable[str]) -> Dict[Key, JsonRow]:
    config_path = historical_dir / "config.json"
    longbench_dir: Path | None = None
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        candidate = config.get("protocol", {}).get("longbench_dir")
        if candidate:
            longbench_dir = Path(candidate)
    if longbench_dir is None:
        return {}
    rows: Dict[Key, JsonRow] = {}
    for task in tasks:
        path = longbench_dir / f"{task}.jsonl"
        if not path.exists():
            continue
        for index, row in enumerate(read_jsonl(path)):
            rows[(task, index)] = row
    return rows


def tiny_keys() -> List[Key]:
    return [(task, idx) for task, indices in TINY_SAMPLE_INDICES.items() for idx in indices]


def available_score_fields(row: Mapping[str, Any]) -> Dict[str, bool]:
    return {
        "selected_block_ids": bool(row.get("used_blocks")),
        "pruned_block_ids": bool(row.get("prompt_block_count") and row.get("used_blocks") is not None),
        "selected_block_scores": bool(row.get("selected_block_scores") or row.get("selected_tail_scores")),
        "pruned_block_scores": bool(row.get("pruned_block_scores") or row.get("pruned_head_scores")),
        "per_layer_block_schedule": bool(row.get("per_layer_block_schedule")),
        "used_blocks": bool(row.get("used_blocks")),
        "used_tokens": bool(row.get("used_tokens")),
    }


def derived_pruned_ids(original: Mapping[str, Any]) -> List[int]:
    selected = {int(x) for x in as_list(original.get("used_blocks"))}
    prompt_block_count = original.get("prompt_block_count")
    if is_missing(prompt_block_count):
        return []
    try:
        total = int(prompt_block_count)
    except Exception:
        return []
    return [idx for idx in range(total) if idx not in selected]


def score_delta(
    historical: Mapping[str, Mapping[Key, Mapping[str, Any]]],
    method: str,
    key: Key,
    base_method: str = METHOD_ORIGINAL,
) -> float | None:
    if key not in historical.get(method, {}) or key not in historical.get(base_method, {}):
        return None
    try:
        return float(historical[method][key].get("prediction_score")) - float(historical[base_method][key].get("prediction_score"))
    except Exception:
        return None


def case_type(delta: float | None) -> str:
    if delta is None:
        return "unavailable"
    if delta > 1e-9:
        return "positive"
    if delta < -1e-9:
        return "negative"
    return "zero"


def finite_or_none(value: Any) -> float | None:
    try:
        result = float(value)
    except Exception:
        return None
    if not math.isfinite(result):
        return None
    return result


def mean_or_none(values: Iterable[Any]) -> float | None:
    finite = [float(v) for v in values if finite_or_none(v) is not None]
    return sum(finite) / len(finite) if finite else None


def as_int_list(value: Any) -> List[int]:
    result: List[int] = []
    for item in as_list(value):
        try:
            result.append(int(item))
        except Exception:
            continue
    return result


def score_dump_available_fields(score_row: Mapping[str, Any]) -> Dict[str, bool]:
    return {
        "selected_block_ids": bool(score_row.get("selected_block_ids")),
        "pruned_block_ids": bool(score_row.get("pruned_block_ids")),
        "selected_block_scores": bool(score_row.get("selected_block_scores")),
        "pruned_block_scores": bool(score_row.get("pruned_block_scores")),
        "selected_tail_scores": bool(score_row.get("selected_tail_scores")),
        "pruned_head_scores": bool(score_row.get("pruned_head_scores")),
        "cutoff_margin": finite_or_none(score_row.get("cutoff_margin")) is not None,
        "near_cutoff_block_metadata": bool(score_row.get("near_cutoff_block_metadata")),
    }


def derive_score_dump_gate(
    score_row: Mapping[str, Any],
    original: Mapping[str, Any],
) -> Dict[str, Any]:
    metadata = [
        row for row in as_list(score_row.get("near_cutoff_block_metadata"))
        if isinstance(row, Mapping)
    ]
    pruned_meta = [row for row in metadata if not bool(row.get("is_selected"))]
    selected_meta = [row for row in metadata if bool(row.get("is_selected"))]
    removable_meta = [
        row for row in selected_meta
        if not bool(row.get("protected_flag")) and not bool(row.get("query_critical_flag"))
    ]
    repair_candidates = [
        int(row.get("block_id"))
        for row in pruned_meta
        if finite_or_none(row.get("query_overlap_proxy")) is not None
        and float(row.get("query_overlap_proxy")) >= QUERY_OVERLAP_MIN
    ]
    removable_ids = [int(row.get("block_id")) for row in removable_meta]
    pruned_overlap_values = [
        float(row.get("query_overlap_proxy"))
        for row in pruned_meta
        if finite_or_none(row.get("query_overlap_proxy")) is not None
    ]
    selected_overlap_values = [
        float(row.get("query_overlap_proxy"))
        for row in selected_meta
        if finite_or_none(row.get("query_overlap_proxy")) is not None
    ]
    selected_redundancy = (
        mean_or_none([1.0 - value for value in selected_overlap_values])
        if selected_overlap_values
        else None
    )
    pruned_query_overlap = mean_or_none(pruned_overlap_values)
    cutoff_margin = finite_or_none(score_row.get("cutoff_margin"))
    budget_preserved = bool(original.get("budget_equal_to_sliminfer", True)) and not bool(original.get("over_budget", False))
    protected_violation = bool(original.get("protected_block_violation", False))
    fallback_reason = "" if not bool(original.get("fallback_used", False)) else str(original.get("fallback_reason") or "fallback_used")

    reasons: List[str] = []
    if cutoff_margin is None:
        reasons.append("missing_cutoff_margin")
    elif cutoff_margin >= TAU_MARGIN:
        reasons.append("cutoff_margin_not_low")
    if not repair_candidates:
        reasons.append("no_pruned_head_query_overlap_candidate")
    if selected_redundancy is None:
        reasons.append("selected_redundancy_unavailable")
    elif selected_redundancy < REDUNDANCY_MIN:
        reasons.append("selected_tail_not_redundant")
    if not removable_ids:
        reasons.append("no_safe_removable_low_conf_block")
    if not budget_preserved:
        reasons.append("budget_not_preserved")
    if protected_violation:
        reasons.append("protected_violation")
    if fallback_reason:
        reasons.append("fallback_used")
    gate_triggered = not reasons
    return {
        "selected_block_ids": as_int_list(score_row.get("selected_block_ids")),
        "derived_pruned_block_ids": as_int_list(score_row.get("pruned_block_ids")),
        "top_pruned_block_ids": as_int_list(score_row.get("pruned_head_block_ids")),
        "selected_tail_scores": score_row.get("selected_tail_scores"),
        "pruned_head_scores": score_row.get("pruned_head_scores"),
        "selected_tail_block_ids": as_int_list(score_row.get("selected_tail_block_ids")),
        "cutoff_margin": cutoff_margin,
        "cutoff_margin_ratio": finite_or_none(score_row.get("cutoff_margin_ratio")),
        "score_entropy_near_cutoff": finite_or_none(score_row.get("score_entropy_near_cutoff")),
        "score_std_near_cutoff": finite_or_none(score_row.get("score_std_near_cutoff")),
        "score_iqr_near_cutoff": finite_or_none(score_row.get("score_iqr_near_cutoff")),
        "num_blocks_within_margin_eps": int(score_row.get("num_blocks_within_margin_eps") or score_row.get("num_blocks_within_eps_0_02") or 0),
        "num_blocks_within_eps_0_01": int(score_row.get("num_blocks_within_eps_0_01") or 0),
        "num_blocks_within_eps_0_02": int(score_row.get("num_blocks_within_eps_0_02") or 0),
        "selected_redundancy_near_cutoff": selected_redundancy,
        "pruned_query_overlap_near_cutoff": pruned_query_overlap,
        "repair_candidate_block_ids": repair_candidates,
        "removable_low_conf_block_ids": removable_ids,
        "uasr_gate_triggered": gate_triggered,
        "uasr_gate_reason": "trigger_low_margin_repair" if gate_triggered else ";".join(reasons),
        "budget_preserved": budget_preserved,
        "protected_violation": protected_violation,
        "fallback_reason": fallback_reason,
        "score_cutoff_fields_available": True,
        "available_fields": score_dump_available_fields(score_row),
        "missing_core_fields": [],
        "score_dump_source_layer": score_row.get("layer_id"),
        "score_dump_source_phase": score_row.get("phase"),
        "near_cutoff_block_metadata": score_row.get("near_cutoff_block_metadata", []),
    }


def audit_one(
    key: Key,
    historical: Mapping[str, Mapping[Key, Mapping[str, Any]]],
    longbench_rows: Mapping[Key, Mapping[str, Any]],
    score_dump_rows: Mapping[Key, Mapping[str, Any]],
) -> JsonRow:
    task, sample_index = key
    original = historical[METHOD_ORIGINAL].get(key)
    if original is None:
        return {
            "task": task,
            "sample_index": sample_index,
            "sample_id": "",
            "audit_status": "blocked",
            "blocked_reason": "missing_sliminfer_original_audit",
            "uasr_gate_triggered": False,
            "uasr_gate_reason": "missing_sliminfer_original_audit",
        }
    query_row = longbench_rows.get(key, {})
    score_row = score_dump_rows.get(key)
    if score_row is not None:
        derived = derive_score_dump_gate(score_row, original)
        selected_ids = derived["selected_block_ids"]
        pruned_ids = derived["derived_pruned_block_ids"]
        fields = derived["available_fields"]
        missing_core: List[str] = []
        score_fields_available = True
        blocked_reason = ""
        budget_preserved = bool(derived["budget_preserved"])
        protected_violation = bool(derived["protected_violation"])
        fallback_reason = str(derived["fallback_reason"])
        gate_triggered = bool(derived["uasr_gate_triggered"])
        gate_reason = str(derived["uasr_gate_reason"])
    else:
        selected_ids = [int(x) for x in as_list(original.get("used_blocks"))]
        pruned_ids = derived_pruned_ids(original)
        fields = available_score_fields(original)
        missing_core = [
            name
            for name, ok in fields.items()
            if name in {"selected_block_scores", "pruned_block_scores"} and not ok
        ]
        score_fields_available = not missing_core
        blocked_reason = ""
        if not score_fields_available:
            blocked_reason = (
                "selected/pruned per-block score details unavailable in existing artifact; "
                "cannot compute SlimInfer hard-cutoff margin"
            )
        budget_preserved = bool(original.get("budget_equal_to_sliminfer", True)) and not bool(original.get("over_budget", False))
        protected_violation = bool(original.get("protected_block_violation", False))
        fallback_reason = "" if not bool(original.get("fallback_used", False)) else str(original.get("fallback_reason") or "fallback_used")
        # Fail-closed when cutoff score data are unavailable.
        gate_triggered = False
        gate_reason = "missing_cutoff_score_fields" if not score_fields_available else "gate_not_evaluated"
        derived = {}

    return {
        "task": task,
        "sample_index": sample_index,
        "sample_id": str(original.get("sample_id") or ""),
        "query": str(query_row.get("input") or query_row.get("question") or ""),
        "context_metadata_available": bool(query_row.get("context")),
        "selected_block_ids": selected_ids,
        "derived_pruned_block_ids": pruned_ids,
        "top_pruned_block_ids": derived.get("top_pruned_block_ids", []),
        "selected_tail_block_ids": derived.get("selected_tail_block_ids", []),
        "selected_tail_scores": derived.get("selected_tail_scores"),
        "pruned_head_scores": derived.get("pruned_head_scores"),
        "cutoff_margin": derived.get("cutoff_margin"),
        "cutoff_margin_percentile": None,
        "cutoff_margin_ratio": derived.get("cutoff_margin_ratio"),
        "score_entropy_near_cutoff": derived.get("score_entropy_near_cutoff"),
        "score_std_near_cutoff": derived.get("score_std_near_cutoff"),
        "score_iqr_near_cutoff": derived.get("score_iqr_near_cutoff"),
        "num_blocks_within_margin_eps": derived.get("num_blocks_within_margin_eps"),
        "num_blocks_within_eps_0_01": derived.get("num_blocks_within_eps_0_01"),
        "num_blocks_within_eps_0_02": derived.get("num_blocks_within_eps_0_02"),
        "selected_redundancy_near_cutoff": derived.get("selected_redundancy_near_cutoff"),
        "pruned_query_overlap_near_cutoff": derived.get("pruned_query_overlap_near_cutoff"),
        "repair_candidate_block_ids": derived.get("repair_candidate_block_ids", []),
        "removable_low_conf_block_ids": derived.get("removable_low_conf_block_ids", []),
        "uasr_gate_triggered": gate_triggered,
        "uasr_gate_reason": gate_reason,
        "budget_preserved": budget_preserved,
        "protected_violation": protected_violation,
        "fallback_reason": fallback_reason,
        "score_cutoff_fields_available": score_fields_available,
        "available_fields": fields,
        "missing_core_fields": missing_core,
        "score_dump_source_layer": derived.get("score_dump_source_layer"),
        "score_dump_source_phase": derived.get("score_dump_source_phase"),
        "near_cutoff_block_metadata": derived.get("near_cutoff_block_metadata", []),
        "per_layer_block_schedule": original.get("per_layer_block_schedule", []),
        "used_blocks": selected_ids,
        "used_tokens": original.get("used_tokens", []),
        "prompt_block_count": original.get("prompt_block_count"),
        "selector_candidate_block_count": original.get("selector_candidate_block_count"),
        "drm_swap_delta_if_available": score_delta(historical, METHOD_DRM_SWAP, key),
        "random_swap_delta_if_available": score_delta(historical, METHOD_RANDOM_SWAP, key),
        "case_type_positive_negative_zero_if_available": case_type(score_delta(historical, METHOD_DRM_SWAP, key)),
        "audit_status": "blocked_cutoff_scores_unavailable" if not score_fields_available else "pass",
        "blocked_reason": blocked_reason,
    }


def attach_cutoff_margin_percentiles(rows: Sequence[JsonRow]) -> None:
    margins = sorted(
        float(row["cutoff_margin"])
        for row in rows
        if finite_or_none(row.get("cutoff_margin")) is not None
    )
    if not margins:
        return
    for row in rows:
        margin = finite_or_none(row.get("cutoff_margin"))
        if margin is None:
            continue
        leq = sum(1 for value in margins if value <= margin)
        row["cutoff_margin_percentile"] = leq / len(margins)


def summarize(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    score_available = sum(bool(row.get("score_cutoff_fields_available")) for row in rows)
    trigger_count = sum(bool(row.get("uasr_gate_triggered")) for row in rows)
    low_margin_values = [
        float(row["cutoff_margin"])
        for row in rows
        if not is_missing(row.get("cutoff_margin"))
    ]
    reason_counts = Counter(str(row.get("uasr_gate_reason")) for row in rows)
    by_case = defaultdict(list)
    by_task = defaultdict(list)
    for row in rows:
        by_case[str(row.get("case_type_positive_negative_zero_if_available"))].append(row)
        by_task[str(row.get("task"))].append(row)

    def mean(values: Sequence[float]) -> float | None:
        vals = [float(v) for v in values if not is_missing(v)]
        return sum(vals) / len(vals) if vals else None

    go_reasons: List[str] = []
    if score_available < len(rows):
        go_reasons.append("score_cutoff_fields_not_fully_available")
    if not low_margin_values:
        go_reasons.append("cutoff_margin_unavailable")
    if trigger_count == 0:
        go_reasons.append("uasr_gate_all_reject_fail_closed")
    if trigger_count == len(rows) and rows:
        go_reasons.append("uasr_gate_all_accept_not_conservative")
    protected_violations = sum(bool(row.get("protected_violation")) for row in rows)
    if protected_violations:
        go_reasons.append("protected_violation_present")
    return {
        "samples": len(rows),
        "score_cutoff_fields_available_count": score_available,
        "score_cutoff_fields_available_rate": score_available / max(1, len(rows)),
        "cutoff_margin_distribution": {
            "available_count": len(low_margin_values),
            "mean": mean(low_margin_values),
            "min": min(low_margin_values) if low_margin_values else None,
            "max": max(low_margin_values) if low_margin_values else None,
        },
        "low_margin_sample_count": sum(
            float(row["cutoff_margin"]) < TAU_MARGIN
            for row in rows
            if not is_missing(row.get("cutoff_margin"))
        ),
        "uasr_trigger_count": trigger_count,
        "uasr_trigger_rate": trigger_count / max(1, len(rows)),
        "gate_reason_counts": dict(reason_counts),
        "by_task": {
            task: {
                "samples": len(task_rows),
                "score_cutoff_fields_available": sum(bool(r.get("score_cutoff_fields_available")) for r in task_rows),
                "trigger_count": sum(bool(r.get("uasr_gate_triggered")) for r in task_rows),
            }
            for task, task_rows in sorted(by_task.items())
        },
        "by_case_type": {
            label: {
                "samples": len(case_rows),
                "score_cutoff_fields_available": sum(bool(r.get("score_cutoff_fields_available")) for r in case_rows),
                "trigger_count": sum(bool(r.get("uasr_gate_triggered")) for r in case_rows),
            }
            for label, case_rows in sorted(by_case.items())
        },
        "protected_violation_count": protected_violations,
        "instrumentation_plan": [
            "In SlimInfer original selector path, persist sorted block ids and base scores at each prune layer before top-K.",
            "Record selected_tail_scores and pruned_head_scores for the final used_block_audit_layer.",
            "Record top_pruned_block_ids, selected tail block ids, and score normalization metadata for token_unit_qk/delta_o.",
            "Record query/context block text or token offsets for near-cutoff selected/pruned blocks.",
            "Record protected/query-critical flags for tail selected blocks and head pruned blocks.",
            "Keep this as audit-only first; do not alter SlimInfer selection or generation behavior.",
        ],
        "go_no_go": {
            "decision": "GO" if not go_reasons else "NO-GO",
            "reasons": go_reasons,
        },
    }


def write_report(output_dir: Path, args: argparse.Namespace, summary: Mapping[str, Any]) -> None:
    lines = [
        "# UASR Uncertainty Audit",
        "",
        f"- Result directory: `{output_dir.resolve()}`",
        f"- Input diagnostic: `{args.historical_dir}`",
        f"- Score dump input: `{args.score_dump_dir or 'not provided'}`",
        "- Model/GPU used: **No**",
        "- Generation performed: **No**",
        "- SlimInfer selector modified: **No**",
        "- SlimInfer original frozen: **True**",
        "",
        "## Fixed gate thresholds",
        "",
        f"- `cutoff_margin < {TAU_MARGIN}`",
        f"- `num_blocks_within_margin_eps` uses eps `{EPS_NEAR_CUTOFF}`",
        f"- `pruned_query_overlap_near_cutoff >= {QUERY_OVERLAP_MIN}`",
        f"- `selected_redundancy_near_cutoff >= {REDUNDANCY_MIN}`",
        "",
        "## Score/cutoff field availability",
        "",
        f"- Score cutoff fields available: {summary['score_cutoff_fields_available_count']}/{summary['samples']} ({summary['score_cutoff_fields_available_rate']:.2%})",
        "- If score dump input is provided, selected/pruned scores come from the explicit UASR selector instrumentation.",
        "- If score dump input is absent, the audit fail-closes because historical artifacts lack selected/pruned score details.",
        "",
        "## Cutoff margin / trigger",
        "",
        f"- cutoff_margin available count: {summary['cutoff_margin_distribution']['available_count']}",
        f"- low-margin sample count: {summary['low_margin_sample_count']}",
        f"- UASR trigger count: {summary['uasr_trigger_count']}/{summary['samples']} ({summary['uasr_trigger_rate']:.2%})",
        f"- Gate reasons: `{json.dumps(summary['gate_reason_counts'], ensure_ascii=False)}`",
        "",
        "## By task",
        "",
        "| task | samples | score fields available | triggers |",
        "|---|---:|---:|---:|",
    ]
    for task, item in summary["by_task"].items():
        lines.append(f"| {task} | {item['samples']} | {item['score_cutoff_fields_available']} | {item['trigger_count']} |")
    lines.extend([
        "",
        "## By DRM-Swap case type",
        "",
        "| case type | samples | score fields available | triggers |",
        "|---|---:|---:|---:|",
    ])
    for label, item in summary["by_case_type"].items():
        lines.append(f"| {label} | {item['samples']} | {item['score_cutoff_fields_available']} | {item['trigger_count']} |")
    lines.extend([
        "",
        "## Minimal instrumentation needed",
        "",
    ])
    for item in summary["instrumentation_plan"]:
        lines.append(f"- {item}")
    lines.extend([
        "",
        "## GO / NO-GO",
        "",
        f"- Decision: **{summary['go_no_go']['decision']}**",
    ])
    for reason in summary["go_no_go"]["reasons"]:
        lines.append(f"- Reason: `{reason}`")
    if not summary["go_no_go"]["reasons"]:
        lines.append("- Reason: score/cutoff observability is populated and the fixed UASR gate has non-trivial trigger behavior.")
    lines.append("")
    lines.append("This is still an offline/audit-only UASR analysis; it is not a new generation result.")
    (output_dir / "uasr_uncertainty_audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Path:
    historical_dir = Path(args.historical_dir)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("results") / f"uasr_uncertainty_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    historical = load_historical(historical_dir)
    longbench_rows = load_longbench_queries(historical_dir, TINY_SAMPLE_INDICES.keys())
    score_dump_rows = load_score_dump(Path(args.score_dump_dir) if args.score_dump_dir else None)
    rows = [audit_one(key, historical, longbench_rows, score_dump_rows) for key in tiny_keys()]
    attach_cutoff_margin_percentiles(rows)
    summary = summarize(rows)
    summary.update({
        "input_artifacts": {
            "historical_dir": str(historical_dir),
            "score_dump_dir": str(args.score_dump_dir or ""),
        },
        "model_or_gpu_used": False,
        "generation_performed": False,
        "sliminfer_selector_modified": False,
        "gold_answer_used_for_gate": False,
        "sample_indices_by_task": TINY_SAMPLE_INDICES,
    })
    fields = [
        "task",
        "sample_index",
        "sample_id",
        "selected_block_ids",
        "top_pruned_block_ids",
        "selected_tail_scores",
        "pruned_head_scores",
        "cutoff_margin",
        "cutoff_margin_percentile",
        "cutoff_margin_ratio",
        "score_entropy_near_cutoff",
        "score_std_near_cutoff",
        "score_iqr_near_cutoff",
        "num_blocks_within_margin_eps",
        "num_blocks_within_eps_0_01",
        "num_blocks_within_eps_0_02",
        "selected_redundancy_near_cutoff",
        "pruned_query_overlap_near_cutoff",
        "repair_candidate_block_ids",
        "removable_low_conf_block_ids",
        "uasr_gate_triggered",
        "uasr_gate_reason",
        "budget_preserved",
        "protected_violation",
        "fallback_reason",
        "score_cutoff_fields_available",
        "missing_core_fields",
        "score_dump_source_layer",
        "score_dump_source_phase",
        "drm_swap_delta_if_available",
        "random_swap_delta_if_available",
        "case_type_positive_negative_zero_if_available",
        "audit_status",
        "blocked_reason",
    ]
    write_jsonl(output_dir / "uasr_uncertainty_audit.jsonl", rows)
    write_csv(output_dir / "uasr_uncertainty_audit.csv", rows, fields)
    write_json(output_dir / "uasr_uncertainty_audit_summary.json", summary)
    write_report(output_dir, args, summary)
    print(output_dir)
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical_dir", default=str(DEFAULT_HISTORICAL_DIR))
    parser.add_argument("--score_dump_dir", default="")
    parser.add_argument("--output_dir", default="")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
