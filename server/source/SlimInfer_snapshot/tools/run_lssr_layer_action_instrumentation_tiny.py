#!/usr/bin/env python3
"""Offline layer-action instrumentation for LSSR.

This script joins existing selector score dumps with existing DRM-Swap rescue
policy audits for the fixed 20-sample HotpotQA/MuSiQue set.  It is deliberately
read-only with respect to SlimInfer behavior: no model run, no decoding, no
scorer, and no selector changes.
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


DEFAULT_DRM_DIR = Path("results/drm_budget_neutral_swap_multihop200_20260706_042717")
DEFAULT_SCORE_DIR = Path("results/uasr_score_instrumentation_tiny_20260707_075243")
METHOD_ORIGINAL = "sliminfer_original"
METHOD_DRM_SWAP = "budget_neutral_drm_swap"
METHOD_RANDOM_SWAP = "random_budget_neutral_swap"

FIXED_SAMPLE_INDICES = {
    "hotpotqa": [0, 1, 2, 3, 4, 5, 41, 56, 57, 117],
    "musique": [0, 1, 2, 3, 4, 20, 50, 54, 56, 134],
}
LAYER_GROUPS = ["early", "middle", "late"]

PER_LAYER_FIELDS = [
    "task",
    "sample_index",
    "sample_id",
    "num_layers",
    "layer_group_definition",
    "selection_behavior_unchanged",
    "layer_id",
    "layer_group",
    "selected_block_ids_original",
    "pruned_block_ids_original",
    "selected_scores",
    "pruned_scores",
    "selected_tail_block_ids",
    "pruned_head_block_ids",
    "cutoff_margin",
    "candidate_donor_block_ids",
    "candidate_donor_scores",
    "candidate_receiver_block_ids",
    "candidate_receiver_scores",
    "candidate_donor_risk_proxy",
    "candidate_receiver_gain_proxy",
    "would_delete_under_middle_donor_policy",
    "would_add_under_late_receiver_policy",
    "action_available",
    "action_unavailable_reason",
    "actual_removed_block_ids_drm_swap",
    "actual_rescued_block_ids_drm_swap",
    "actual_removed_score_mean_drm_swap",
    "actual_rescue_score_mean_drm_swap",
    "actual_swap_count_drm_swap",
    "actual_action_layer_source",
]

PER_SAMPLE_FIELDS = [
    "task",
    "sample_index",
    "sample_id",
    "num_layers",
    "layer_group_definition",
    "selection_behavior_unchanged",
    "middle_donor_capacity",
    "early_protected_capacity",
    "late_receiver_need",
    "late_query_or_evidence_candidate_count",
    "layerwise_repair_possible",
    "layerwise_repair_reason",
    "actual_action_group_counts",
    "actual_final_layer_only_artifact_reason",
    "case_type_vs_original",
    "case_type_vs_random_swap",
]

JsonRow = Dict[str, Any]
Key = Tuple[str, int]


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def serialize_csv(value: Any) -> Any:
    if isinstance(value, (list, dict, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return value


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: serialize_csv(row.get(field, "")) for field in fields})


def key_of(row: Mapping[str, Any]) -> Key:
    return str(row["task"]), int(row["sample_index"])


def fixed_keys() -> List[Key]:
    return [(task, int(index)) for task, indices in FIXED_SAMPLE_INDICES.items() for index in indices]


def as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def as_int_list(value: Any) -> List[int]:
    result: List[int] = []
    for item in as_list(value):
        try:
            result.append(int(item))
        except Exception:
            continue
    return result


def finite_or_none(value: Any) -> float | None:
    try:
        result = float(value)
    except Exception:
        return None
    return result if math.isfinite(result) else None


def mean_or_none(values: Iterable[Any]) -> float | None:
    finite = [float(value) for value in values if finite_or_none(value) is not None]
    return sum(finite) / len(finite) if finite else None


def score(row: Mapping[str, Any] | None) -> float | None:
    return finite_or_none(row.get("prediction_score")) if row is not None else None


def case_type(delta: float | None) -> str:
    if delta is None:
        return "unavailable"
    if delta > 1e-9:
        return "positive"
    if delta < -1e-9:
        return "negative"
    return "zero"


def layer_group(layer_id: int, num_layers: int) -> str:
    if num_layers <= 0:
        return "unavailable"
    first = num_layers // 3
    second = (2 * num_layers) // 3
    if layer_id < first:
        return "early"
    if layer_id < second:
        return "middle"
    return "late"


def layer_group_definition(num_layers: int) -> Dict[str, List[int]]:
    if num_layers <= 0:
        return {group: [] for group in LAYER_GROUPS}
    first = num_layers // 3
    second = (2 * num_layers) // 3
    return {
        "early": list(range(0, first)),
        "middle": list(range(first, second)),
        "late": list(range(second, num_layers)),
    }


def model_layer_count(drm_dir: Path) -> Tuple[int, str]:
    config_path = drm_dir / "config.json"
    if config_path.exists():
        config = read_json(config_path)
        model_name = config.get("protocol", {}).get("model_name")
        if model_name:
            model_config_path = Path(model_name) / "config.json"
            if model_config_path.exists():
                model_config = read_json(model_config_path)
                for field in ["num_hidden_layers", "n_layer", "num_layers", "n_layers"]:
                    value = finite_or_none(model_config.get(field))
                    if value is not None:
                        return int(value), "model_config"
        max_layer = -1
        protocol = config.get("protocol", {})
        for layer in as_int_list(protocol.get("prune_layers")):
            max_layer = max(max_layer, layer)
        value = finite_or_none(protocol.get("final_used_block_audit_layer"))
        if value is not None:
            max_layer = max(max_layer, int(value))
        if max_layer >= 0:
            return max_layer + 1, "drm_config_max_layer_plus_one"
    return 0, "unavailable"


def discover_score_dump_path(explicit: str, score_dir: Path) -> Path:
    if explicit:
        path = Path(explicit)
        if path.is_file():
            return path
        candidate = path / "score_dump_audit.jsonl"
        if candidate.exists():
            return candidate
        raise RuntimeError(f"score dump path not found: {explicit}")
    candidate = score_dir / "score_dump_audit.jsonl"
    if candidate.exists():
        return candidate
    discovered = sorted(
        Path("results").glob("uasr_score_instrumentation_tiny*/score_dump_audit.jsonl"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not discovered:
        raise RuntimeError("could not find score_dump_audit.jsonl")
    return discovered[0]


def load_historical(drm_dir: Path) -> Dict[str, Dict[Key, JsonRow]]:
    by_method: Dict[str, Dict[Key, JsonRow]] = defaultdict(dict)
    for row in read_jsonl(drm_dir / "audit" / "per_sample_audit.jsonl"):
        method = str(row.get("method_key") or "")
        if method:
            by_method[method][key_of(row)] = row
    for method in [METHOD_ORIGINAL, METHOD_DRM_SWAP, METHOD_RANDOM_SWAP]:
        if method not in by_method:
            raise RuntimeError(f"missing historical method: {method}")
    return dict(by_method)


def load_score_rows(path: Path) -> Dict[Key, List[JsonRow]]:
    rows_by_key: Dict[Key, List[JsonRow]] = defaultdict(list)
    for row in read_jsonl(path):
        key = key_of(row)
        if key in set(fixed_keys()):
            rows_by_key[key].append(row)
    for rows in rows_by_key.values():
        rows.sort(key=lambda row: int(row.get("layer_id", -1)))
    return dict(rows_by_key)


def load_drm_action_rows(drm_dir: Path) -> Dict[Key, Dict[int, JsonRow]]:
    by_key_layer: Dict[Key, Dict[int, JsonRow]] = defaultdict(dict)
    for task in FIXED_SAMPLE_INDICES:
        path = drm_dir / "runs" / METHOD_DRM_SWAP / task / "audit" / "rescue_policy_audit.jsonl"
        if not path.exists():
            continue
        for row in read_jsonl(path):
            key = key_of(row)
            if key not in set(fixed_keys()):
                continue
            layer = finite_or_none(row.get("layer_id"))
            if layer is None:
                continue
            by_key_layer[key][int(layer)] = row
    return dict(by_key_layer)


def normalize_lookup(ids: Sequence[int], scores: Sequence[Any]) -> Dict[int, float]:
    pairs: List[Tuple[int, float]] = []
    for block_id, score_value in zip(ids, scores):
        score_float = finite_or_none(score_value)
        if score_float is not None:
            pairs.append((int(block_id), score_float))
    if not pairs:
        return {}
    values = [value for _, value in pairs]
    min_value = min(values)
    max_value = max(values)
    denom = max(max_value - min_value, 1e-12)
    return {block_id: (value - min_value) / denom for block_id, value in pairs}


def score_lookup(ids: Sequence[int], scores: Sequence[Any]) -> Dict[int, float]:
    lookup: Dict[int, float] = {}
    for block_id, score_value in zip(ids, scores):
        score_float = finite_or_none(score_value)
        if score_float is not None:
            lookup[int(block_id)] = score_float
    return lookup


def metadata_lookup(row: Mapping[str, Any]) -> Dict[int, Mapping[str, Any]]:
    out: Dict[int, Mapping[str, Any]] = {}
    for item in row.get("near_cutoff_block_metadata", []) or []:
        if isinstance(item, Mapping) and finite_or_none(item.get("block_id")) is not None:
            out[int(item["block_id"])] = item
    return out


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def query_overlap(meta: Mapping[str, Any] | None) -> float:
    if not meta:
        return 0.0
    value = finite_or_none(meta.get("query_overlap_proxy"))
    return max(0.0, min(1.0, float(value))) if value is not None else 0.0


def candidate_donor_ids(row: Mapping[str, Any], meta_by_block: Mapping[int, Mapping[str, Any]]) -> List[int]:
    candidates: List[int] = []
    for block_id in as_int_list(row.get("selected_tail_block_ids")):
        meta = meta_by_block.get(block_id, {})
        if boolish(meta.get("protected_flag")) or boolish(meta.get("query_critical_flag")):
            continue
        candidates.append(block_id)
    return candidates


def donor_risk(block_id: int, score_norm: Mapping[int, float], meta_by_block: Mapping[int, Mapping[str, Any]]) -> float:
    meta = meta_by_block.get(int(block_id), {})
    if boolish(meta.get("protected_flag")):
        return 1.0
    risk = 0.70 * float(score_norm.get(int(block_id), 0.0)) + 0.30 * query_overlap(meta)
    if boolish(meta.get("query_critical_flag")):
        risk = max(risk, 0.75)
    return max(0.0, min(1.0, risk))


def receiver_gain(block_id: int, score_norm: Mapping[int, float], meta_by_block: Mapping[int, Mapping[str, Any]]) -> float:
    meta = meta_by_block.get(int(block_id), {})
    gain = 0.70 * float(score_norm.get(int(block_id), 0.0)) + 0.30 * query_overlap(meta)
    return max(0.0, min(1.0, gain))


def build_layer_record(
    *,
    key: Key,
    score_row: Mapping[str, Any],
    action_row: Mapping[str, Any] | None,
    num_layers: int,
    group_def: Mapping[str, Sequence[int]],
) -> JsonRow:
    selected_ids = as_int_list(score_row.get("selected_block_ids"))
    pruned_ids = as_int_list(score_row.get("pruned_block_ids"))
    selected_scores = [finite_or_none(value) for value in as_list(score_row.get("selected_block_scores"))]
    pruned_scores = [finite_or_none(value) for value in as_list(score_row.get("pruned_block_scores"))]
    selected_lookup = score_lookup(selected_ids, selected_scores)
    pruned_lookup = score_lookup(pruned_ids, pruned_scores)
    all_ids = as_int_list(score_row.get("all_candidate_block_ids")) or selected_ids + pruned_ids
    all_scores = as_list(score_row.get("all_candidate_scores")) or selected_scores + pruned_scores
    norm = normalize_lookup(all_ids, all_scores)
    meta = metadata_lookup(score_row)
    layer_id = int(score_row.get("layer_id", -1))
    group = layer_group(layer_id, num_layers)

    donors = candidate_donor_ids(score_row, meta)
    receivers = as_int_list(score_row.get("pruned_head_block_ids"))
    donor_scores = [selected_lookup.get(block_id) for block_id in donors]
    receiver_scores = [pruned_lookup.get(block_id) for block_id in receivers]
    donor_risks = [donor_risk(block_id, norm, meta) for block_id in donors]
    receiver_gains = [receiver_gain(block_id, norm, meta) for block_id in receivers]

    removed = as_int_list(action_row.get("removed_block_ids")) if action_row else []
    rescued = as_int_list(action_row.get("rescued_block_ids")) if action_row else []
    action_available = action_row is not None
    if action_row is None:
        if pruned_ids:
            reason = "no_drm_swap_rescue_policy_record_for_layer"
        else:
            reason = "no_pruned_candidates_or_no_drm_swap_record_for_layer"
    elif not removed and not rescued:
        reason = str(action_row.get("fallback_reason") or "drm_swap_record_present_without_action")
    else:
        reason = ""

    return {
        "task": key[0],
        "sample_index": key[1],
        "sample_id": str(score_row.get("sample_id") or (action_row or {}).get("sample_id") or ""),
        "num_layers": num_layers,
        "layer_group_definition": dict(group_def),
        "selection_behavior_unchanged": True,
        "layer_id": layer_id,
        "layer_group": group,
        "selected_block_ids_original": selected_ids,
        "pruned_block_ids_original": pruned_ids,
        "selected_scores": selected_scores,
        "pruned_scores": pruned_scores,
        "selected_tail_block_ids": as_int_list(score_row.get("selected_tail_block_ids")),
        "pruned_head_block_ids": as_int_list(score_row.get("pruned_head_block_ids")),
        "cutoff_margin": finite_or_none(score_row.get("cutoff_margin")),
        "candidate_donor_block_ids": donors,
        "candidate_donor_scores": donor_scores,
        "candidate_receiver_block_ids": receivers,
        "candidate_receiver_scores": receiver_scores,
        "candidate_donor_risk_proxy": donor_risks,
        "candidate_receiver_gain_proxy": receiver_gains,
        "would_delete_under_middle_donor_policy": group == "middle" and bool(donors),
        "would_add_under_late_receiver_policy": group == "late" and bool(receivers),
        "action_available": action_available,
        "action_unavailable_reason": reason,
        "actual_removed_block_ids_drm_swap": removed,
        "actual_rescued_block_ids_drm_swap": rescued,
        "actual_removed_score_mean_drm_swap": finite_or_none((action_row or {}).get("removed_score_mean")),
        "actual_rescue_score_mean_drm_swap": finite_or_none((action_row or {}).get("rescue_score_mean")),
        "actual_swap_count_drm_swap": int((action_row or {}).get("swap_count") or 0),
        "actual_action_layer_source": (
            "drm_rescue_policy_audit" if action_row is not None else "unavailable"
        ),
        "late_query_or_evidence_candidate_count": (
            sum(
                1
                for block_id in receivers
                if query_overlap(meta.get(block_id)) > 0.0 or block_id in set(rescued)
            )
            if group == "late"
            else 0
        ),
    }


def case_labels(historical: Mapping[str, Mapping[Key, Mapping[str, Any]]], key: Key) -> Tuple[str, str]:
    original = historical[METHOD_ORIGINAL].get(key)
    drm = historical[METHOD_DRM_SWAP].get(key)
    random = historical[METHOD_RANDOM_SWAP].get(key)
    original_score = score(original)
    drm_score = score(drm)
    random_score = score(random)
    delta_vs_original = None if original_score is None or drm_score is None else drm_score - original_score
    delta_vs_random = None if random_score is None or drm_score is None else drm_score - random_score
    return case_type(delta_vs_original), case_type(delta_vs_random)


def summarize_sample(
    key: Key,
    layer_rows: Sequence[Mapping[str, Any]],
    historical: Mapping[str, Mapping[Key, Mapping[str, Any]]],
    num_layers: int,
    group_def: Mapping[str, Sequence[int]],
) -> JsonRow:
    middle_donor_capacity = sum(
        len(row.get("candidate_donor_block_ids", []) or [])
        for row in layer_rows
        if row.get("layer_group") == "middle"
    )
    early_protected_capacity = sum(
        len(row.get("selected_tail_block_ids", []) or []) - len(row.get("candidate_donor_block_ids", []) or [])
        for row in layer_rows
        if row.get("layer_group") == "early"
    )
    late_receiver_need = sum(
        len(row.get("candidate_receiver_block_ids", []) or [])
        for row in layer_rows
        if row.get("layer_group") == "late"
    )
    late_query_or_evidence_count = sum(
        int(row.get("late_query_or_evidence_candidate_count") or 0)
        for row in layer_rows
    )
    action_counts = {group: 0 for group in LAYER_GROUPS}
    for row in layer_rows:
        group = str(row.get("layer_group"))
        if group in action_counts:
            action_counts[group] += int(row.get("actual_swap_count_drm_swap") or 0)
    repair_possible = middle_donor_capacity > 0 and late_receiver_need > 0
    if repair_possible:
        reason = "middle_donor_capacity_and_late_receiver_need_nonzero"
    elif middle_donor_capacity <= 0:
        reason = "middle_donor_capacity_zero"
    else:
        reason = "late_receiver_need_zero"
    case_original, case_random = case_labels(historical, key)
    return {
        "task": key[0],
        "sample_index": key[1],
        "sample_id": str(layer_rows[0].get("sample_id") if layer_rows else ""),
        "num_layers": num_layers,
        "layer_group_definition": dict(group_def),
        "selection_behavior_unchanged": True,
        "per_layer": list(layer_rows),
        "middle_donor_capacity": middle_donor_capacity,
        "early_protected_capacity": early_protected_capacity,
        "late_receiver_need": late_receiver_need,
        "late_query_or_evidence_candidate_count": late_query_or_evidence_count,
        "layerwise_repair_possible": repair_possible,
        "layerwise_repair_reason": reason,
        "actual_action_group_counts": action_counts,
        "actual_final_layer_only_artifact_reason": (
            "per_sample_audit stores only final used/rescue audit layers; "
            "rescue_policy_audit exposes per-prune-layer actions"
        ),
        "case_type_vs_original": case_original,
        "case_type_vs_random_swap": case_random,
    }


def aggregate_proxy_by_group(layer_rows: Sequence[Mapping[str, Any]], field: str) -> Dict[str, float | None]:
    grouped: Dict[str, List[float]] = defaultdict(list)
    for row in layer_rows:
        group = str(row.get("layer_group"))
        for value in row.get(field, []) or []:
            numeric = finite_or_none(value)
            if group in LAYER_GROUPS and numeric is not None:
                grouped[group].append(numeric)
    return {group: mean_or_none(grouped[group]) for group in LAYER_GROUPS}


def aggregate_summary(sample_rows: Sequence[Mapping[str, Any]], layer_rows: Sequence[Mapping[str, Any]], num_layers: int, layer_source: str, score_path: Path, drm_dir: Path) -> JsonRow:
    action_group_counts = {group: 0 for group in LAYER_GROUPS}
    for row in sample_rows:
        counts = row.get("actual_action_group_counts", {}) or {}
        for group in LAYER_GROUPS:
            action_group_counts[group] += int(counts.get(group, 0) or 0)
    middle_capacity = sum(int(row.get("middle_donor_capacity") or 0) for row in sample_rows)
    late_need = sum(int(row.get("late_receiver_need") or 0) for row in sample_rows)
    late_query_evidence = sum(int(row.get("late_query_or_evidence_candidate_count") or 0) for row in sample_rows)
    repair_possible_count = sum(bool(row.get("layerwise_repair_possible")) for row in sample_rows)
    margin_by_group: Dict[str, List[float]] = defaultdict(list)
    for row in layer_rows:
        margin = finite_or_none(row.get("cutoff_margin"))
        group = str(row.get("layer_group"))
        if margin is not None and group in LAYER_GROUPS:
            margin_by_group[group].append(margin)
    donor_risk = aggregate_proxy_by_group(layer_rows, "candidate_donor_risk_proxy")
    receiver_gain = aggregate_proxy_by_group(layer_rows, "candidate_receiver_gain_proxy")
    action_groups_observed = [group for group in LAYER_GROUPS if action_group_counts[group] > 0]
    layerwise_signal_not_final_only = len(action_groups_observed) > 1 or (action_groups_observed and action_groups_observed != ["late"])
    return {
        "samples": len(sample_rows),
        "per_layer_rows": len(layer_rows),
        "sample_indices_by_task": FIXED_SAMPLE_INDICES,
        "num_layers": num_layers,
        "num_layers_source": layer_source,
        "layer_group_definition": layer_group_definition(num_layers),
        "input_artifacts": {
            "drm_dir": str(drm_dir),
            "score_dump_path": str(score_path),
            "drm_swap_rescue_policy_audit_glob": str(drm_dir / "runs" / METHOD_DRM_SWAP / "*" / "audit" / "rescue_policy_audit.jsonl"),
        },
        "constraints": {
            "model_or_gpu_used": False,
            "prefill_rerun_performed": False,
            "generation_performed": False,
            "full_run_performed": False,
            "scorer_invoked": False,
            "sliminfer_selector_modified": False,
            "selection_behavior_unchanged": True,
            "gold_answer_used_for_decision": False,
            "statistical_significance_claimed": False,
        },
        "score_availability": {
            "selected_pruned_scores_available_layers": sum(bool(row.get("selected_scores") or row.get("pruned_scores")) for row in layer_rows),
            "cutoff_margin_available_layers": sum(finite_or_none(row.get("cutoff_margin")) is not None for row in layer_rows),
            "action_available_layers": sum(bool(row.get("action_available")) for row in layer_rows),
        },
        "case_counts_vs_original": dict(Counter(str(row.get("case_type_vs_original")) for row in sample_rows)),
        "middle_donor_capacity_total": middle_capacity,
        "late_receiver_need_total": late_need,
        "late_query_or_evidence_candidate_count_total": late_query_evidence,
        "layerwise_repair_possible_samples": repair_possible_count,
        "actual_action_group_counts": action_group_counts,
        "action_groups_observed": action_groups_observed,
        "layerwise_signal_not_final_only": layerwise_signal_not_final_only,
        "cutoff_margin_mean_by_group": {group: mean_or_none(margin_by_group[group]) for group in LAYER_GROUPS},
        "candidate_donor_risk_mean_by_group": donor_risk,
        "candidate_receiver_gain_mean_by_group": receiver_gain,
        "location_findings": {
            "sliminfer_topk_path": "utils/cache_utils.py:_calculate_pruning_with_virtual_blocks fallback/top-k block selection and _write_uasr_score_dump",
            "drm_swap_action_path": "utils/cache_utils.py:_apply_optional_rescue_policy with rescue_policy=budget_neutral_drm_swap",
            "final_late_mapping_reason": "aggregate per_sample_audit copied final-layer removed/rescued fields and used_block_audit_layer/rescue_audit_layer only",
        },
    }


def write_report(output_dir: Path, summary: Mapping[str, Any]) -> None:
    lines = [
        "# LSSR Layer Action Instrumentation",
        "",
        f"- Result directory: `{output_dir.resolve()}`",
        f"- Score dump: `{summary['input_artifacts']['score_dump_path']}`",
        f"- DRM input: `{summary['input_artifacts']['drm_dir']}`",
        "- Model/GPU used: **No**",
        "- Prefill rerun performed: **No**",
        "- Generation performed: **No**",
        "- Scorer invoked: **No**",
        "- SlimInfer selector modified: **No**",
        "- Selection behavior unchanged: **True**",
        "",
        "## Locator Findings",
        "",
        f"- SlimInfer top-k path: `{summary['location_findings']['sliminfer_topk_path']}`",
        f"- DRM-Swap action path: `{summary['location_findings']['drm_swap_action_path']}`",
        f"- Previous all-late reason: {summary['location_findings']['final_late_mapping_reason']}",
        "",
        "## Layer Observability",
        "",
        f"- Layers: {summary['num_layers']} (`{summary['num_layers_source']}`)",
        f"- Per-layer rows: {summary['per_layer_rows']}",
        f"- Selected/pruned score layers: {summary['score_availability']['selected_pruned_scores_available_layers']}",
        f"- Cutoff margin layers: {summary['score_availability']['cutoff_margin_available_layers']}",
        f"- DRM action layers available: {summary['score_availability']['action_available_layers']}",
        "",
        "## Cross-Layer Signals",
        "",
        f"- Middle donor capacity: {summary['middle_donor_capacity_total']}",
        f"- Late receiver need: {summary['late_receiver_need_total']}",
        f"- Late query/evidence proxy candidates: {summary['late_query_or_evidence_candidate_count_total']}",
        f"- Layerwise repair possible samples: {summary['layerwise_repair_possible_samples']}/{summary['samples']}",
        f"- Actual action group counts: `{json.dumps(summary['actual_action_group_counts'], ensure_ascii=False)}`",
        f"- Layerwise signal not final-only: {summary['layerwise_signal_not_final_only']}",
        f"- Cutoff margin mean by group: `{json.dumps(summary['cutoff_margin_mean_by_group'], ensure_ascii=False)}`",
        f"- Donor risk mean by group: `{json.dumps(summary['candidate_donor_risk_mean_by_group'], ensure_ascii=False)}`",
        f"- Receiver gain mean by group: `{json.dumps(summary['candidate_receiver_gain_mean_by_group'], ensure_ascii=False)}`",
        "",
        "This is instrumentation only, not a quality result.",
    ]
    (output_dir / "lssr_layer_action_instrumentation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Path:
    drm_dir = Path(args.drm_dir)
    score_path = discover_score_dump_path(args.score_dump_path, Path(args.score_dump_dir))
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("results") / f"lssr_layer_action_instrumentation_tiny_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    historical = load_historical(drm_dir)
    score_rows = load_score_rows(score_path)
    action_rows = load_drm_action_rows(drm_dir)
    num_layers, layer_source = model_layer_count(drm_dir)
    group_def = layer_group_definition(num_layers)

    sample_rows: List[JsonRow] = []
    all_layer_rows: List[JsonRow] = []
    missing_score_keys = [key for key in fixed_keys() if key not in score_rows]
    if missing_score_keys:
        write_json(output_dir / "STOPPED_MISSING_SCORE_ROWS.json", {"missing_keys": missing_score_keys})
        raise RuntimeError(f"missing score rows for fixed keys: {missing_score_keys}")

    for key in fixed_keys():
        layer_rows: List[JsonRow] = []
        action_by_layer = action_rows.get(key, {})
        for score_row in score_rows[key]:
            layer_id = int(score_row.get("layer_id", -1))
            layer_row = build_layer_record(
                key=key,
                score_row=score_row,
                action_row=action_by_layer.get(layer_id),
                num_layers=num_layers,
                group_def=group_def,
            )
            layer_rows.append(layer_row)
            all_layer_rows.append(layer_row)
        sample_rows.append(summarize_sample(key, layer_rows, historical, num_layers, group_def))

    summary = aggregate_summary(sample_rows, all_layer_rows, num_layers, layer_source, score_path, drm_dir)
    write_jsonl(output_dir / "lssr_layer_action_instrumentation.jsonl", sample_rows)
    write_jsonl(output_dir / "lssr_layer_action_instrumentation_layers.jsonl", all_layer_rows)
    write_csv(output_dir / "lssr_layer_action_instrumentation.csv", sample_rows, PER_SAMPLE_FIELDS)
    write_csv(output_dir / "lssr_layer_action_instrumentation_layers.csv", all_layer_rows, PER_LAYER_FIELDS)
    write_json(output_dir / "lssr_layer_action_instrumentation_summary.json", summary)
    write_report(output_dir, summary)
    print(output_dir)
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drm_dir", default=str(DEFAULT_DRM_DIR))
    parser.add_argument("--score_dump_dir", default=str(DEFAULT_SCORE_DIR))
    parser.add_argument("--score_dump_path", default="")
    parser.add_argument("--output_dir", default="")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
