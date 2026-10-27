#!/usr/bin/env python3
"""Layer-Stratified SlimInfer Repair (LSSR) audit.

This is an offline audit only.  It reads existing SlimInfer/DRM-Swap and UASR
artifacts, checks whether swap outcomes have an observable layer-wise pattern,
and writes a GO/NO-GO recommendation for a later LSSR controller audit.

It does not run a model, does not generate predictions, does not invoke a
scorer, and does not modify the frozen SlimInfer selector.
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
DEFAULT_UASR_DIR = Path("results/uasr_uncertainty_audit_20260707_053411")

METHOD_ORIGINAL = "sliminfer_original"
METHOD_DRM_SWAP = "budget_neutral_drm_swap"
METHOD_RANDOM_SWAP = "random_budget_neutral_swap"
REQUIRED_METHODS = [METHOD_ORIGINAL, METHOD_DRM_SWAP, METHOD_RANDOM_SWAP]

FIXED_SAMPLE_INDICES = {
    "hotpotqa": [0, 1, 2, 3, 4, 5, 41, 56, 57, 117],
    "musique": [0, 1, 2, 3, 4, 20, 50, 54, 56, 134],
}

LAYER_GROUPS = ["early", "middle", "late"]

CSV_FIELDS = [
    "task",
    "sample_index",
    "sample_id",
    "case_type_vs_original",
    "case_type_vs_random_swap",
    "drm_swap_delta_vs_original",
    "drm_swap_delta_vs_random_swap",
    "removed_block_ids",
    "rescued_block_ids",
    "removed_block_layers",
    "rescued_block_layers",
    "removed_layer_group_counts",
    "rescued_layer_group_counts",
    "early_removed_count",
    "middle_removed_count",
    "late_removed_count",
    "early_rescued_count",
    "middle_rescued_count",
    "late_rescued_count",
    "removed_score_by_layer_group",
    "rescued_score_by_layer_group",
    "delete_risk_by_layer_group",
    "rescue_gain_by_layer_group",
    "cutoff_margin_by_layer_if_available",
    "layer_group_safety_signal",
    "layer_group_repair_recommendation",
    "layer_observability_status",
    "score_instrumentation_available",
    "lssr_action_instrumentation_available",
    "middle_donor_capacity",
    "early_protected_capacity",
    "late_receiver_need",
    "late_query_or_evidence_candidate_count",
    "layerwise_repair_possible",
    "layerwise_repair_reason",
    "actual_action_group_counts",
    "schedule_pruned_count_by_layer_group",
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
    if isinstance(value, (dict, list, tuple)):
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
    keys: List[Key] = []
    for task, indices in FIXED_SAMPLE_INDICES.items():
        keys.extend((task, int(index)) for index in indices)
    return keys


def as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def as_int_list(value: Any) -> List[int]:
    out: List[int] = []
    for item in as_list(value):
        try:
            out.append(int(item))
        except Exception:
            continue
    return out


def finite_or_none(value: Any) -> float | None:
    try:
        result = float(value)
    except Exception:
        return None
    if not math.isfinite(result):
        return None
    return result


def mean_or_none(values: Iterable[Any]) -> float | None:
    finite = [float(value) for value in values if finite_or_none(value) is not None]
    return sum(finite) / len(finite) if finite else None


def score(row: Mapping[str, Any] | None) -> float | None:
    if row is None:
        return None
    return finite_or_none(row.get("prediction_score"))


def case_type(delta: float | None) -> str:
    if delta is None:
        return "unavailable"
    if delta > 1e-9:
        return "positive"
    if delta < -1e-9:
        return "negative"
    return "zero"


def load_historical(drm_dir: Path) -> Dict[str, Dict[Key, JsonRow]]:
    by_method: Dict[str, Dict[Key, JsonRow]] = defaultdict(dict)
    for row in read_jsonl(drm_dir / "audit" / "per_sample_audit.jsonl"):
        method = str(row.get("method_key") or "")
        if method:
            by_method[method][key_of(row)] = row
    missing = [method for method in REQUIRED_METHODS if method not in by_method]
    if missing:
        raise RuntimeError(f"historical audit missing methods: {missing}")
    return dict(by_method)


def load_uasr(uasr_dir: Path) -> Dict[Key, JsonRow]:
    path = uasr_dir / "uasr_uncertainty_audit.jsonl"
    if not path.exists():
        return {}
    return {key_of(row): row for row in read_jsonl(path)}


def discover_lssr_instrumentation_dir(explicit: str) -> Path | None:
    if explicit:
        path = Path(explicit)
        if path.is_file():
            return path.parent
        if (path / "lssr_layer_action_instrumentation.jsonl").exists():
            return path
        raise RuntimeError(f"LSSR instrumentation path not found: {explicit}")
    discovered = sorted(
        Path("results").glob("lssr_layer_action_instrumentation_tiny_*/lssr_layer_action_instrumentation.jsonl"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return discovered[0].parent if discovered else None


def load_lssr_instrumentation(lssr_dir: Path | None) -> Tuple[Dict[Key, JsonRow], Dict[str, Any]]:
    if lssr_dir is None:
        return {}, {}
    path = lssr_dir / "lssr_layer_action_instrumentation.jsonl"
    if not path.exists():
        return {}, {}
    rows = {key_of(row): row for row in read_jsonl(path)}
    summary_path = lssr_dir / "lssr_layer_action_instrumentation_summary.json"
    summary = read_json(summary_path) if summary_path.exists() else {}
    summary["lssr_instrumentation_dir"] = str(lssr_dir)
    return rows, summary


def discover_score_dump_path(explicit: str, uasr_dir: Path) -> Path | None:
    if explicit:
        path = Path(explicit)
        if path.is_file():
            return path
        candidate = path / "score_dump_audit.jsonl"
        if candidate.exists():
            return candidate
        raise RuntimeError(f"score dump path not found: {explicit}")

    direct_candidates = [
        uasr_dir / "score_dump_audit.jsonl",
        uasr_dir / "score_dump" / "score_dump_audit.jsonl",
        uasr_dir / "raw" / "score_dump_audit.jsonl",
    ]
    for candidate in direct_candidates:
        if candidate.exists():
            return candidate

    discovered = list(Path("results").glob("uasr_score_instrumentation_tiny*/score_dump_audit.jsonl"))
    if not discovered:
        return None
    return sorted(discovered, key=lambda path: path.stat().st_mtime, reverse=True)[0]


def load_score_dumps(path: Path | None) -> Dict[Key, List[JsonRow]]:
    if path is None:
        return {}
    by_key: Dict[Key, List[JsonRow]] = defaultdict(list)
    for row in read_jsonl(path):
        try:
            by_key[key_of(row)].append(row)
        except Exception:
            continue
    for rows in by_key.values():
        rows.sort(key=lambda row: int(row.get("layer_id", -1)))
    return dict(by_key)


def model_layer_count_from_config(drm_dir: Path) -> int | None:
    config_path = drm_dir / "config.json"
    if not config_path.exists():
        return None
    config = read_json(config_path)
    model_name = config.get("protocol", {}).get("model_name")
    if not model_name:
        return None
    model_config_path = Path(model_name) / "config.json"
    if not model_config_path.exists():
        return None
    model_config = read_json(model_config_path)
    for field in ["num_hidden_layers", "n_layer", "num_layers", "n_layers"]:
        value = model_config.get(field)
        try:
            if value is not None:
                return int(value)
        except Exception:
            continue
    return None


def infer_layer_count(
    drm_dir: Path,
    historical: Mapping[str, Mapping[Key, Mapping[str, Any]]],
    score_dump_rows: Mapping[Key, Sequence[Mapping[str, Any]]],
) -> Tuple[int, str]:
    from_model = model_layer_count_from_config(drm_dir)
    if from_model is not None and from_model > 0:
        return from_model, "model_config"

    max_layer = -1
    config_path = drm_dir / "config.json"
    if config_path.exists():
        config = read_json(config_path)
        protocol = config.get("protocol", {})
        for field in ["prune_layers"]:
            for layer in as_int_list(protocol.get(field)):
                max_layer = max(max_layer, layer)
        for field in ["final_used_block_audit_layer"]:
            value = finite_or_none(protocol.get(field))
            if value is not None:
                max_layer = max(max_layer, int(value))

    for method_rows in historical.values():
        for row in method_rows.values():
            for field in ["used_block_audit_layer", "rescue_audit_layer"]:
                value = finite_or_none(row.get(field))
                if value is not None:
                    max_layer = max(max_layer, int(value))
            for item in row.get("per_layer_block_schedule", []) or []:
                if not isinstance(item, Mapping):
                    continue
                for field in ["prune_layer", "used_layer"]:
                    value = finite_or_none(item.get(field))
                    if value is not None:
                        max_layer = max(max_layer, int(value))

    for rows in score_dump_rows.values():
        for row in rows:
            value = finite_or_none(row.get("layer_id"))
            if value is not None:
                max_layer = max(max_layer, int(value))

    if max_layer >= 0:
        return max_layer + 1, "audit_max_layer_plus_one"
    return 0, "unavailable"


def layer_group_ranges(num_layers: int) -> Dict[str, List[int]]:
    if num_layers <= 0:
        return {group: [] for group in LAYER_GROUPS}
    first = num_layers // 3
    second = (2 * num_layers) // 3
    return {
        "early": list(range(0, first)),
        "middle": list(range(first, second)),
        "late": list(range(second, num_layers)),
    }


def layer_group(layer: int | None, num_layers: int) -> str:
    if layer is None or num_layers <= 0:
        return "unavailable"
    first = num_layers // 3
    second = (2 * num_layers) // 3
    if int(layer) < first:
        return "early"
    if int(layer) < second:
        return "middle"
    return "late"


def group_count_from_layers(layers: Sequence[int], num_layers: int) -> Dict[str, int]:
    counts = {group: 0 for group in LAYER_GROUPS}
    for layer in layers:
        group = layer_group(layer, num_layers)
        if group in counts:
            counts[group] += 1
    return counts


def repeated_layer(row: Mapping[str, Any], primary: str, fallback: str) -> int | None:
    value = finite_or_none(row.get(primary))
    if value is not None:
        return int(value)
    value = finite_or_none(row.get(fallback))
    if value is not None:
        return int(value)
    return None


def score_dump_margins(score_rows: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    margins: Dict[str, float] = {}
    for row in score_rows:
        layer = finite_or_none(row.get("layer_id"))
        margin = finite_or_none(row.get("cutoff_margin"))
        if layer is None or margin is None:
            continue
        margins[str(int(layer))] = margin
    return margins


def score_lookup_by_layer(score_rows: Sequence[Mapping[str, Any]]) -> Dict[int, Dict[int, float]]:
    lookup: Dict[int, Dict[int, float]] = {}
    for row in score_rows:
        layer = finite_or_none(row.get("layer_id"))
        if layer is None:
            continue
        layer_id = int(layer)
        lookup.setdefault(layer_id, {})
        for ids_field, scores_field in [
            ("selected_block_ids", "selected_block_scores"),
            ("pruned_block_ids", "pruned_block_scores"),
        ]:
            block_ids = as_int_list(row.get(ids_field))
            scores = as_list(row.get(scores_field))
            for block_id, value in zip(block_ids, scores):
                numeric = finite_or_none(value)
                if numeric is not None:
                    lookup[layer_id][int(block_id)] = numeric
    return lookup


def scored_mean_by_group(
    block_ids: Sequence[int],
    layers: Sequence[int],
    score_lookup: Mapping[int, Mapping[int, float]],
    num_layers: int,
) -> Tuple[Dict[str, float], Dict[str, str]]:
    grouped: Dict[str, List[float]] = defaultdict(list)
    source: Dict[str, str] = {}
    for block_id, layer in zip(block_ids, layers):
        score_map = score_lookup.get(int(layer), {})
        value = score_map.get(int(block_id))
        if value is None:
            continue
        group = layer_group(layer, num_layers)
        grouped[group].append(float(value))
        source[group] = "score_dump_block_score"
    return (
        {group: mean for group, mean in ((group, mean_or_none(values)) for group, values in grouped.items()) if mean is not None},
        source,
    )


def row_mean_assigned_by_group(
    value: Any,
    layers: Sequence[int],
    num_layers: int,
    source_name: str,
) -> Tuple[Dict[str, float], Dict[str, str]]:
    numeric = finite_or_none(value)
    if numeric is None or not layers:
        return {}, {}
    groups = sorted({layer_group(layer, num_layers) for layer in layers if layer_group(layer, num_layers) in LAYER_GROUPS})
    return ({group: numeric for group in groups}, {group: source_name for group in groups})


def merge_score_maps(
    primary: Tuple[Dict[str, float], Dict[str, str]],
    fallback: Tuple[Dict[str, float], Dict[str, str]],
) -> Tuple[Dict[str, float], Dict[str, str]]:
    values = dict(fallback[0])
    sources = dict(fallback[1])
    values.update(primary[0])
    sources.update(primary[1])
    return values, sources


RISK_FIELDS = [
    "deletion_risk",
    "deletion_evidence_risk_v3",
    "deletion_risk_v3_1",
    "deletion_risk_v3_raw",
    "deletion_query_overlap_risk",
    "deletion_entity_risk",
    "deletion_date_number_risk",
    "deletion_title_risk",
    "deletion_answer_type_risk",
    "deletion_base_importance_risk",
    "deletion_protected_proximity_risk",
    "removed_drop_utility_mean",
]

GAIN_FIELDS = [
    "rescued_vs_removed_entity_gain",
    "rescued_vs_active_entity_gain",
    "sentence_novelty_gain",
    "bridge_entity_gain",
    "answer_type_support_gain",
    "rescue_coverage_gap",
    "coverage_gap_v3_1",
]


def row_feature_by_group(
    row: Mapping[str, Any],
    fields: Sequence[str],
    layers: Sequence[int],
    num_layers: int,
) -> Dict[str, float]:
    value = mean_or_none(row.get(field) for field in fields)
    if value is None or not layers:
        return {}
    groups = sorted({layer_group(layer, num_layers) for layer in layers if layer_group(layer, num_layers) in LAYER_GROUPS})
    return {group: value for group in groups}


def schedule_pruned_counts(row: Mapping[str, Any], num_layers: int) -> Dict[str, int]:
    counts = {group: 0 for group in LAYER_GROUPS}
    schedule = [item for item in row.get("per_layer_block_schedule", []) or [] if isinstance(item, Mapping)]
    schedule.sort(key=lambda item: int(item.get("prune_layer", item.get("used_layer", 0)) or 0))
    previous = finite_or_none(row.get("prompt_block_count"))
    if previous is None and schedule:
        previous = finite_or_none(schedule[0].get("selected_block_count"))
    if previous is None:
        return counts
    prev_count = int(previous)
    for item in schedule:
        selected = finite_or_none(item.get("selected_block_count"))
        if selected is None:
            continue
        layer = finite_or_none(item.get("prune_layer"))
        group = layer_group(int(layer), num_layers) if layer is not None else "unavailable"
        drop = max(0, prev_count - int(selected))
        if group in counts:
            counts[group] += drop
        prev_count = int(selected)
    return counts


def safety_signal(row: Mapping[str, Any]) -> str:
    case = str(row.get("case_type_vs_original"))
    removed_counts = row.get("removed_layer_group_counts", {})
    rescued_counts = row.get("rescued_layer_group_counts", {})
    affected_groups = {
        group
        for group in LAYER_GROUPS
        if int(removed_counts.get(group, 0) or 0) > 0 or int(rescued_counts.get(group, 0) or 0) > 0
    }
    if not affected_groups:
        return "no_swap_observed"
    parts: List[str] = []
    if len(affected_groups) == 1:
        parts.append(f"single_group_{next(iter(affected_groups))}_only")
    if case == "negative" and (
        int(removed_counts.get("early", 0) or 0) > 0 or int(removed_counts.get("late", 0) or 0) > 0
    ):
        parts.append("negative_with_early_or_late_deletion")
    if case == "positive" and int(rescued_counts.get("late", 0) or 0) > 0:
        parts.append("positive_with_late_rescue")
    if case != "negative" and int(removed_counts.get("middle", 0) or 0) > 0:
        parts.append("nonnegative_with_middle_deletion")
    if not parts:
        parts.append(f"{case}_without_layer_specific_signal")
    return ";".join(parts)


def repair_recommendation(row: Mapping[str, Any]) -> str:
    status = str(row.get("layer_observability_status", ""))
    case = str(row.get("case_type_vs_original"))
    removed_counts = row.get("removed_layer_group_counts", {})
    rescued_counts = row.get("rescued_layer_group_counts", {})
    has_middle_removed = int(removed_counts.get("middle", 0) or 0) > 0
    has_protected_removed = (
        int(removed_counts.get("early", 0) or 0) > 0 or int(removed_counts.get("late", 0) or 0) > 0
    )
    has_late_rescued = int(rescued_counts.get("late", 0) or 0) > 0
    if bool(row.get("layerwise_repair_possible")):
        return "candidate_middle_donor_late_receiver"
    if "degenerate" in status:
        return "instrument_layer_specific_selection_before_controller"
    if case == "negative" and has_protected_removed:
        return "candidate_protect_early_late_delete_less"
    if case == "positive" and has_late_rescued and has_middle_removed:
        return "candidate_middle_donor_late_receiver"
    if case == "positive" and has_late_rescued:
        return "candidate_late_receiver_needs_donor_audit"
    return "no_layer_stratified_action_recommended"


def mean_from_layer_rows(per_layer: Sequence[Mapping[str, Any]], field: str) -> Dict[str, float]:
    grouped: Dict[str, List[float]] = defaultdict(list)
    for layer_row in per_layer:
        group = str(layer_row.get("layer_group"))
        if group not in LAYER_GROUPS:
            continue
        value = finite_or_none(layer_row.get(field))
        if value is not None:
            grouped[group].append(value)
    return {
        group: mean
        for group, mean in (
            (group, mean_or_none(values)) for group, values in grouped.items()
        )
        if mean is not None
    }


def proxy_mean_from_layer_rows(per_layer: Sequence[Mapping[str, Any]], field: str) -> Dict[str, float]:
    grouped: Dict[str, List[float]] = defaultdict(list)
    for layer_row in per_layer:
        group = str(layer_row.get("layer_group"))
        if group not in LAYER_GROUPS:
            continue
        for value in layer_row.get(field, []) or []:
            numeric = finite_or_none(value)
            if numeric is not None:
                grouped[group].append(numeric)
    return {
        group: mean
        for group, mean in (
            (group, mean_or_none(values)) for group, values in grouped.items()
        )
        if mean is not None
    }


def lssr_action_view(lssr_row: Mapping[str, Any], num_layers: int) -> Dict[str, Any]:
    per_layer = [
        item for item in lssr_row.get("per_layer", []) or []
        if isinstance(item, Mapping)
    ]
    removed_ids: List[int] = []
    rescued_ids: List[int] = []
    removed_layers: List[int] = []
    rescued_layers: List[int] = []
    for layer_row in per_layer:
        layer = finite_or_none(layer_row.get("layer_id"))
        if layer is None:
            continue
        layer_id = int(layer)
        for block_id in as_int_list(layer_row.get("actual_removed_block_ids_drm_swap")):
            removed_ids.append(block_id)
            removed_layers.append(layer_id)
        for block_id in as_int_list(layer_row.get("actual_rescued_block_ids_drm_swap")):
            rescued_ids.append(block_id)
            rescued_layers.append(layer_id)
    cutoff_margins = {
        str(int(layer_row["layer_id"])): float(layer_row["cutoff_margin"])
        for layer_row in per_layer
        if finite_or_none(layer_row.get("layer_id")) is not None
        and finite_or_none(layer_row.get("cutoff_margin")) is not None
    }
    return {
        "removed_ids": removed_ids,
        "rescued_ids": rescued_ids,
        "removed_layers": removed_layers,
        "rescued_layers": rescued_layers,
        "removed_counts": group_count_from_layers(removed_layers, num_layers),
        "rescued_counts": group_count_from_layers(rescued_layers, num_layers),
        "removed_scores": mean_from_layer_rows(per_layer, "actual_removed_score_mean_drm_swap"),
        "rescued_scores": mean_from_layer_rows(per_layer, "actual_rescue_score_mean_drm_swap"),
        "delete_risk_by_group": proxy_mean_from_layer_rows(per_layer, "candidate_donor_risk_proxy"),
        "rescue_gain_by_group": proxy_mean_from_layer_rows(per_layer, "candidate_receiver_gain_proxy"),
        "cutoff_margins": cutoff_margins,
        "actual_action_group_counts": lssr_row.get("actual_action_group_counts", {}),
        "middle_donor_capacity": int(lssr_row.get("middle_donor_capacity") or 0),
        "early_protected_capacity": int(lssr_row.get("early_protected_capacity") or 0),
        "late_receiver_need": int(lssr_row.get("late_receiver_need") or 0),
        "late_query_or_evidence_candidate_count": int(lssr_row.get("late_query_or_evidence_candidate_count") or 0),
        "layerwise_repair_possible": bool(lssr_row.get("layerwise_repair_possible")),
        "layerwise_repair_reason": str(lssr_row.get("layerwise_repair_reason") or ""),
    }


def audit_one(
    key: Key,
    historical: Mapping[str, Mapping[Key, Mapping[str, Any]]],
    uasr_rows: Mapping[Key, Mapping[str, Any]],
    score_dump_rows: Mapping[Key, Sequence[Mapping[str, Any]]],
    lssr_rows: Mapping[Key, Mapping[str, Any]],
    num_layers: int,
) -> JsonRow:
    task, sample_index = key
    original = historical[METHOD_ORIGINAL].get(key)
    drm = historical[METHOD_DRM_SWAP].get(key)
    random = historical[METHOD_RANDOM_SWAP].get(key)
    if original is None or drm is None or random is None:
        return {
            "task": task,
            "sample_index": sample_index,
            "sample_id": "",
            "case_type_vs_original": "unavailable",
            "case_type_vs_random_swap": "unavailable",
            "layer_observability_status": "missing_required_historical_row",
            "score_instrumentation_available": False,
            "layer_group_safety_signal": "missing_required_historical_row",
            "layer_group_repair_recommendation": "no_layer_stratified_action_recommended",
        }

    original_score = score(original)
    drm_score = score(drm)
    random_score = score(random)
    delta_vs_original = None if original_score is None or drm_score is None else drm_score - original_score
    delta_vs_random = None if random_score is None or drm_score is None else drm_score - random_score

    key_score_rows = list(score_dump_rows.get(key, []))
    lssr_row = lssr_rows.get(key)
    if lssr_row is not None:
        action_view = lssr_action_view(lssr_row, num_layers)
        removed_ids = action_view["removed_ids"]
        rescued_ids = action_view["rescued_ids"]
        removed_layers = action_view["removed_layers"]
        rescued_layers = action_view["rescued_layers"]
        removed_counts = action_view["removed_counts"]
        rescued_counts = action_view["rescued_counts"]
        removed_scores = action_view["removed_scores"]
        rescued_scores = action_view["rescued_scores"]
        removed_score_sources = {
            group: "lssr_per_layer_drm_action_removed_score_mean"
            for group in removed_scores
        }
        rescued_score_sources = {
            group: "lssr_per_layer_drm_action_rescue_score_mean"
            for group in rescued_scores
        }
        cutoff_margins = action_view["cutoff_margins"]
        delete_risk_by_group = action_view["delete_risk_by_group"]
        rescue_gain_by_group = action_view["rescue_gain_by_group"]
        score_available = bool(key_score_rows) or bool(cutoff_margins)
    else:
        removed_ids = as_int_list(drm.get("removed_block_ids") or drm.get("candidate_removed_block_ids"))
        rescued_ids = as_int_list(drm.get("rescued_block_ids") or drm.get("candidate_rescued_block_ids"))

        removed_layer = repeated_layer(drm, "used_block_audit_layer", "rescue_audit_layer")
        rescued_layer = repeated_layer(drm, "rescue_audit_layer", "used_block_audit_layer")
        removed_layers = [removed_layer for _ in removed_ids if removed_layer is not None]
        rescued_layers = [rescued_layer for _ in rescued_ids if rescued_layer is not None]

        removed_counts = group_count_from_layers([int(layer) for layer in removed_layers], num_layers)
        rescued_counts = group_count_from_layers([int(layer) for layer in rescued_layers], num_layers)

        score_lookup = score_lookup_by_layer(key_score_rows)
        removed_score_from_dump = scored_mean_by_group(removed_ids, [int(layer) for layer in removed_layers], score_lookup, num_layers)
        removed_score_fallback = row_mean_assigned_by_group(
            drm.get("removed_score_mean"), [int(layer) for layer in removed_layers], num_layers, "row_removed_score_mean"
        )
        removed_scores, removed_score_sources = merge_score_maps(removed_score_fallback, removed_score_from_dump)

        rescued_score_from_dump = scored_mean_by_group(rescued_ids, [int(layer) for layer in rescued_layers], score_lookup, num_layers)
        rescued_score_fallback = row_mean_assigned_by_group(
            drm.get("rescue_score_mean"), [int(layer) for layer in rescued_layers], num_layers, "row_rescue_score_mean"
        )
        rescued_scores, rescued_score_sources = merge_score_maps(rescued_score_fallback, rescued_score_from_dump)

        cutoff_margins = score_dump_margins(key_score_rows)
        if not cutoff_margins and key in uasr_rows:
            uasr = uasr_rows[key]
            source_layer = finite_or_none(uasr.get("score_dump_source_layer"))
            margin = finite_or_none(uasr.get("cutoff_margin"))
            if source_layer is not None and margin is not None:
                cutoff_margins[str(int(source_layer))] = margin
        delete_risk_by_group = row_feature_by_group(
            drm, RISK_FIELDS, [int(layer) for layer in removed_layers], num_layers
        )
        rescue_gain_by_group = row_feature_by_group(
            drm, GAIN_FIELDS, [int(layer) for layer in rescued_layers], num_layers
        )
        score_available = bool(key_score_rows) or bool(cutoff_margins)
    affected_groups = {
        group
        for group in LAYER_GROUPS
        if removed_counts.get(group, 0) > 0 or rescued_counts.get(group, 0) > 0
    }
    if lssr_row is not None:
        observability = "lssr_layer_action_instrumentation_available"
    elif not removed_layers and not rescued_layers:
        observability = "no_swap_action_observed"
    elif len(affected_groups) <= 1:
        observability = "degenerate_single_action_layer_group"
    else:
        observability = "multi_group_action_observed"
    if not score_available:
        observability += ";score_instrumentation_unavailable"

    row: JsonRow = {
        "task": task,
        "sample_index": sample_index,
        "sample_id": str(original.get("sample_id") or drm.get("sample_id") or ""),
        "case_type_vs_original": case_type(delta_vs_original),
        "case_type_vs_random_swap": case_type(delta_vs_random),
        "drm_swap_delta_vs_original": delta_vs_original,
        "drm_swap_delta_vs_random_swap": delta_vs_random,
        "original_score": original_score,
        "drm_swap_score": drm_score,
        "random_swap_score": random_score,
        "removed_block_ids": removed_ids,
        "rescued_block_ids": rescued_ids,
        "removed_block_layers": removed_layers,
        "rescued_block_layers": rescued_layers,
        "removed_layer_group_counts": removed_counts,
        "rescued_layer_group_counts": rescued_counts,
        "early_removed_count": removed_counts["early"],
        "middle_removed_count": removed_counts["middle"],
        "late_removed_count": removed_counts["late"],
        "early_rescued_count": rescued_counts["early"],
        "middle_rescued_count": rescued_counts["middle"],
        "late_rescued_count": rescued_counts["late"],
        "removed_score_by_layer_group": removed_scores,
        "rescued_score_by_layer_group": rescued_scores,
        "removed_score_source_by_layer_group": removed_score_sources,
        "rescued_score_source_by_layer_group": rescued_score_sources,
        "delete_risk_by_layer_group": delete_risk_by_group,
        "rescue_gain_by_layer_group": rescue_gain_by_group,
        "cutoff_margin_by_layer_if_available": cutoff_margins,
        "layer_observability_status": observability,
        "score_instrumentation_available": score_available,
        "lssr_action_instrumentation_available": lssr_row is not None,
        "middle_donor_capacity": action_view["middle_donor_capacity"] if lssr_row is not None else 0,
        "early_protected_capacity": action_view["early_protected_capacity"] if lssr_row is not None else 0,
        "late_receiver_need": action_view["late_receiver_need"] if lssr_row is not None else 0,
        "late_query_or_evidence_candidate_count": (
            action_view["late_query_or_evidence_candidate_count"] if lssr_row is not None else 0
        ),
        "layerwise_repair_possible": action_view["layerwise_repair_possible"] if lssr_row is not None else False,
        "layerwise_repair_reason": action_view["layerwise_repair_reason"] if lssr_row is not None else "",
        "actual_action_group_counts": action_view["actual_action_group_counts"] if lssr_row is not None else {},
        "schedule_pruned_count_by_layer_group": schedule_pruned_counts(original, num_layers),
        "used_block_audit_layer": drm.get("used_block_audit_layer"),
        "rescue_audit_layer": drm.get("rescue_audit_layer"),
        "per_layer_block_schedule": original.get("per_layer_block_schedule", []),
        "uasr_cutoff_status": uasr_rows.get(key, {}).get("audit_status", "unavailable"),
        "uasr_blocked_reason": uasr_rows.get(key, {}).get("blocked_reason", ""),
    }
    row["layer_group_safety_signal"] = safety_signal(row)
    row["layer_group_repair_recommendation"] = repair_recommendation(row)
    return row


def sum_group_counts(rows: Sequence[Mapping[str, Any]], field: str) -> Dict[str, int]:
    counts = {group: 0 for group in LAYER_GROUPS}
    for row in rows:
        value = row.get(field, {})
        if not isinstance(value, Mapping):
            continue
        for group in LAYER_GROUPS:
            counts[group] += int(value.get(group, 0) or 0)
    return counts


def group_counts_by_case(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("case_type_vs_original"))].append(row)
    out: Dict[str, Dict[str, Any]] = {}
    for label, case_rows in sorted(grouped.items()):
        out[label] = {
            "samples": len(case_rows),
            "removed": sum_group_counts(case_rows, "removed_layer_group_counts"),
            "rescued": sum_group_counts(case_rows, "rescued_layer_group_counts"),
            "score_instrumentation_available": sum(bool(row.get("score_instrumentation_available")) for row in case_rows),
        }
    return out


def group_counts_by_task_case(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for task in sorted({str(row.get("task")) for row in rows}):
        task_rows = [row for row in rows if str(row.get("task")) == task]
        out[task] = {
            "samples": len(task_rows),
            "case_counts": dict(Counter(str(row.get("case_type_vs_original")) for row in task_rows)),
            "by_case": group_counts_by_case(task_rows),
            "removed": sum_group_counts(task_rows, "removed_layer_group_counts"),
            "rescued": sum_group_counts(task_rows, "rescued_layer_group_counts"),
        }
    return out


def compare_layer_signal(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    case_rows: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        case_rows[str(row.get("case_type_vs_original"))].append(row)

    def average_count(label: str, field: str, group: str) -> float | None:
        selected = case_rows.get(label, [])
        if not selected:
            return None
        return sum(int((row.get(field, {}) or {}).get(group, 0) or 0) for row in selected) / len(selected)

    groups_observed = {
        group
        for row in rows
        for field in ["removed_layer_group_counts", "rescued_layer_group_counts"]
        for group, count in (row.get(field, {}) or {}).items()
        if group in LAYER_GROUPS and int(count or 0) > 0
    }
    return {
        "groups_observed_in_actions": sorted(groups_observed),
        "is_degenerate": len(groups_observed) <= 1,
        "average_removed_by_case": {
            label: {group: average_count(label, "removed_layer_group_counts", group) for group in LAYER_GROUPS}
            for label in sorted(case_rows)
        },
        "average_rescued_by_case": {
            label: {group: average_count(label, "rescued_layer_group_counts", group) for group in LAYER_GROUPS}
            for label in sorted(case_rows)
        },
    }


def mean_metric_by_group(rows: Sequence[Mapping[str, Any]], field: str) -> Dict[str, float | None]:
    grouped: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        value = row.get(field, {})
        if not isinstance(value, Mapping):
            continue
        for group in LAYER_GROUPS:
            numeric = finite_or_none(value.get(group))
            if numeric is not None:
                grouped[group].append(numeric)
    return {group: mean_or_none(grouped[group]) for group in LAYER_GROUPS}


def sum_numeric_field(rows: Sequence[Mapping[str, Any]], field: str) -> int:
    return sum(int(row.get(field) or 0) for row in rows)


def summarize(
    rows: Sequence[Mapping[str, Any]],
    num_layers: int,
    layer_source: str,
    score_dump_path: Path | None,
    lssr_summary: Mapping[str, Any],
) -> Dict[str, Any]:
    hotpot_negative = [
        row for row in rows
        if row.get("task") == "hotpotqa" and row.get("case_type_vs_original") == "negative"
    ]
    musique_positive = [
        row for row in rows
        if row.get("task") == "musique" and row.get("case_type_vs_original") == "positive"
    ]
    middle_removed_rows = [
        row for row in rows
        if int((row.get("removed_layer_group_counts", {}) or {}).get("middle", 0) or 0) > 0
    ]
    layer_signal = compare_layer_signal(rows)
    score_available_count = sum(bool(row.get("score_instrumentation_available")) for row in rows)
    cutoff_available_count = sum(bool(row.get("cutoff_margin_by_layer_if_available")) for row in rows)
    lssr_available_count = sum(bool(row.get("lssr_action_instrumentation_available")) for row in rows)
    middle_donor_capacity_total = sum_numeric_field(rows, "middle_donor_capacity")
    late_receiver_need_total = sum_numeric_field(rows, "late_receiver_need")
    late_query_or_evidence_total = sum_numeric_field(rows, "late_query_or_evidence_candidate_count")
    layerwise_repair_possible_count = sum(bool(row.get("layerwise_repair_possible")) for row in rows)
    donor_risk_by_group = mean_metric_by_group(rows, "delete_risk_by_layer_group")
    receiver_gain_by_group = mean_metric_by_group(rows, "rescue_gain_by_layer_group")
    cutoff_margin_values: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        for layer, margin in (row.get("cutoff_margin_by_layer_if_available", {}) or {}).items():
            numeric = finite_or_none(margin)
            if numeric is None:
                continue
            try:
                group = layer_group(int(layer), num_layers)
            except Exception:
                continue
            if group in LAYER_GROUPS:
                cutoff_margin_values[group].append(numeric)
    cutoff_margin_mean_by_group = {
        group: mean_or_none(cutoff_margin_values[group]) for group in LAYER_GROUPS
    }
    risk_gain_has_difference = (
        donor_risk_by_group.get("middle") is not None
        and donor_risk_by_group.get("late") is not None
        and abs(float(donor_risk_by_group["middle"]) - float(donor_risk_by_group["late"])) > 1e-12
    ) or (
        receiver_gain_by_group.get("middle") is not None
        and receiver_gain_by_group.get("late") is not None
        and abs(float(receiver_gain_by_group["middle"]) - float(receiver_gain_by_group["late"])) > 1e-6
    ) or (
        cutoff_margin_mean_by_group.get("middle") is not None
        and cutoff_margin_mean_by_group.get("late") is not None
        and abs(float(cutoff_margin_mean_by_group["middle"]) - float(cutoff_margin_mean_by_group["late"])) > 1e-12
    )
    using_lssr = lssr_available_count > 0

    go_failures: List[str] = []
    if using_lssr:
        if middle_donor_capacity_total <= 0:
            go_failures.append("middle_donor_capacity_zero")
        if late_receiver_need_total <= 0:
            go_failures.append("late_receiver_need_zero")
        if layerwise_repair_possible_count <= 0:
            go_failures.append("layerwise_repair_plan_not_possible")
        if layer_signal["is_degenerate"]:
            go_failures.append("layer_group_signal_degenerate_single_action_group")
        if not risk_gain_has_difference:
            go_failures.append("layerwise_score_risk_gain_no_group_difference")
        if not hotpot_negative:
            go_failures.append("hotpotqa_negative_cases_absent_on_fixed_set")
        elif sum_group_counts(hotpot_negative, "removed_layer_group_counts")["late"] <= 0:
            go_failures.append("hotpotqa_negative_late_deletion_risk_not_observed")
        if not musique_positive:
            go_failures.append("musique_positive_cases_absent_on_fixed_set")
        elif late_receiver_need_total <= 0 or receiver_gain_by_group.get("late") is None:
            go_failures.append("musique_positive_late_receiver_gain_not_observed")
    else:
        if not hotpot_negative:
            go_failures.append("hotpotqa_negative_cases_absent_on_fixed_set")
        elif layer_signal["is_degenerate"]:
            go_failures.append("hotpotqa_negative_layer_pattern_degenerate")
        else:
            hotpot_removed = sum_group_counts(hotpot_negative, "removed_layer_group_counts")
            if hotpot_removed["early"] + hotpot_removed["late"] <= hotpot_removed["middle"]:
                go_failures.append("hotpotqa_negative_not_more_early_late_deletion")

        if not musique_positive:
            go_failures.append("musique_positive_cases_absent_on_fixed_set")
        else:
            musique_rescued = sum_group_counts(musique_positive, "rescued_layer_group_counts")
            if musique_rescued["late"] == 0 and sum_group_counts(musique_positive, "removed_layer_group_counts")["middle"] == 0:
                go_failures.append("musique_positive_not_late_rescue_or_middle_donor")
            if layer_signal["is_degenerate"]:
                go_failures.append("musique_late_rescue_not_discriminative")

        if not middle_removed_rows:
            go_failures.append("middle_donor_not_observed_in_swap_actions")
        if layer_signal["is_degenerate"]:
            go_failures.append("layer_group_signal_degenerate_single_action_group")
    if score_available_count == 0:
        go_failures.append("score_instrumentation_unavailable")
    if cutoff_available_count == 0:
        go_failures.append("cutoff_margin_by_layer_unavailable")

    return {
        "samples": len(rows),
        "sample_indices_by_task": FIXED_SAMPLE_INDICES,
        "num_layers": num_layers,
        "num_layers_source": layer_source,
        "layer_group_ranges": layer_group_ranges(num_layers),
        "score_dump_path": str(score_dump_path or ""),
        "lssr_instrumentation_dir": str(lssr_summary.get("lssr_instrumentation_dir", "")),
        "score_instrumentation_available_count": score_available_count,
        "cutoff_margin_available_count": cutoff_available_count,
        "lssr_action_instrumentation_available_count": lssr_available_count,
        "case_counts_vs_original": dict(Counter(str(row.get("case_type_vs_original")) for row in rows)),
        "case_counts_vs_random_swap": dict(Counter(str(row.get("case_type_vs_random_swap")) for row in rows)),
        "by_case_type_vs_original": group_counts_by_case(rows),
        "by_task_case": group_counts_by_task_case(rows),
        "hotpotqa_negative": {
            "samples": len(hotpot_negative),
            "sample_indices": [int(row["sample_index"]) for row in hotpot_negative],
            "removed": sum_group_counts(hotpot_negative, "removed_layer_group_counts"),
            "rescued": sum_group_counts(hotpot_negative, "rescued_layer_group_counts"),
        },
        "musique_positive": {
            "samples": len(musique_positive),
            "sample_indices": [int(row["sample_index"]) for row in musique_positive],
            "removed": sum_group_counts(musique_positive, "removed_layer_group_counts"),
            "rescued": sum_group_counts(musique_positive, "rescued_layer_group_counts"),
        },
        "middle_donor": {
            "observed_samples": len(middle_removed_rows),
            "negative_samples": sum(row.get("case_type_vs_original") == "negative" for row in middle_removed_rows),
            "case_counts": dict(Counter(str(row.get("case_type_vs_original")) for row in middle_removed_rows)),
            "capacity_total": middle_donor_capacity_total,
        },
        "cross_layer_repair": {
            "middle_donor_capacity_total": middle_donor_capacity_total,
            "late_receiver_need_total": late_receiver_need_total,
            "late_query_or_evidence_candidate_count_total": late_query_or_evidence_total,
            "layerwise_repair_possible_samples": layerwise_repair_possible_count,
            "budget_neutral_layer_aware_plan": (
                "delete_low_risk_middle_tail_donors_and_add_late_pruned_head_receivers"
                if middle_donor_capacity_total > 0 and late_receiver_need_total > 0
                else "unavailable"
            ),
        },
        "layer_group_signal": layer_signal,
        "layerwise_score_risk_gain": {
            "cutoff_margin_mean_by_group": cutoff_margin_mean_by_group,
            "candidate_donor_risk_mean_by_group": donor_risk_by_group,
            "candidate_receiver_gain_mean_by_group": receiver_gain_by_group,
            "has_group_difference": risk_gain_has_difference,
        },
        "hotpotqa_negative_late_deletion_risk_explanation": {
            "late_removed_count": sum_group_counts(hotpot_negative, "removed_layer_group_counts")["late"],
            "late_donor_risk_proxy_mean": donor_risk_by_group.get("late"),
            "interpretation": (
                "late_deletion_observed_in_negative_cases"
                if hotpot_negative and sum_group_counts(hotpot_negative, "removed_layer_group_counts")["late"] > 0
                else "not_observed"
            ),
        },
        "musique_positive_late_receiver_gain_explanation": {
            "late_rescued_count": sum_group_counts(musique_positive, "rescued_layer_group_counts")["late"],
            "late_receiver_gain_proxy_mean": receiver_gain_by_group.get("late"),
            "interpretation": (
                "late_receiver_gain_available_for_positive_cases"
                if musique_positive and receiver_gain_by_group.get("late") is not None
                else "not_observed"
            ),
        },
        "layerwise_signal_independence": {
            "independent_of_final_late_action": using_lssr and not layer_signal["is_degenerate"],
            "independent_of_uasr_cutoff_only": using_lssr,
            "protected_anchor_overlap_checked": False,
            "interpretation": (
                "per_layer_action_and_candidate_provenance_not_just_final_late_action"
                if using_lssr and not layer_signal["is_degenerate"]
                else "insufficient_independence_evidence"
            ),
        },
        "layer_vs_global_score_signal": {
            "layer_signal_discriminative": not layer_signal["is_degenerate"],
            "global_row_scores_available": any(
                bool(row.get("removed_score_by_layer_group")) or bool(row.get("rescued_score_by_layer_group"))
                for row in rows
            ),
            "conclusion": (
                "layer_group_degenerate_no_evidence_over_global_score"
                if layer_signal["is_degenerate"]
                else "layer_group_non_degenerate_compare_in_controller_audit"
            ),
        },
        "go_no_go": {
            "decision": "GO" if not go_failures else "NO-GO",
            "reasons": go_failures,
        },
        "constraints": {
            "model_or_gpu_used": False,
            "generation_performed": False,
            "full_run_performed": False,
            "scorer_invoked": False,
            "sliminfer_selector_modified": False,
            "gold_answer_used_for_decision": False,
            "statistical_significance_claimed": False,
        },
    }


def format_group_ranges(ranges: Mapping[str, Sequence[int]]) -> str:
    parts: List[str] = []
    for group in LAYER_GROUPS:
        values = list(ranges.get(group, []))
        if values:
            parts.append(f"{group}={values[0]}..{values[-1]}")
        else:
            parts.append(f"{group}=unavailable")
    return ", ".join(parts)


def table_rows_by_case(summary: Mapping[str, Any]) -> List[str]:
    lines = [
        "| case | samples | removed early/middle/late | rescued early/middle/late | score fields |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, item in summary["by_case_type_vs_original"].items():
        removed = item["removed"]
        rescued = item["rescued"]
        lines.append(
            f"| {label} | {item['samples']} | "
            f"{removed['early']}/{removed['middle']}/{removed['late']} | "
            f"{rescued['early']}/{rescued['middle']}/{rescued['late']} | "
            f"{item['score_instrumentation_available']} |"
        )
    return lines


def write_report(output_dir: Path, args: argparse.Namespace, summary: Mapping[str, Any]) -> None:
    hotpot = summary["hotpotqa_negative"]
    musique = summary["musique_positive"]
    middle = summary["middle_donor"]
    signal = summary["layer_group_signal"]
    layer_vs_score = summary["layer_vs_global_score_signal"]
    cross = summary["cross_layer_repair"]
    risk_gain = summary["layerwise_score_risk_gain"]
    ranges = format_group_ranges(summary["layer_group_ranges"])
    lines = [
        "# Layer-Stratified SlimInfer Repair Audit",
        "",
        f"- Result directory: `{output_dir.resolve()}`",
        f"- DRM input: `{args.drm_dir}`",
        f"- UASR input: `{args.uasr_dir}`",
        f"- Score instrumentation: `{summary['score_dump_path'] or 'unavailable'}`",
        f"- LSSR action instrumentation: `{summary['lssr_instrumentation_dir'] or 'unavailable'}`",
        "- Model/GPU used: **No**",
        "- Generation performed: **No**",
        "- Full run performed: **No**",
        "- Scorer invoked: **No**",
        "- SlimInfer selector modified: **No**",
        "- Gold answer used for controller decision: **No**",
        "- Statistical significance claimed: **No**",
        "",
        "## 1. Layer group definition",
        "",
        f"- Number of layers: {summary['num_layers']} (`{summary['num_layers_source']}`)",
        f"- Groups: {ranges}",
        "- Layer groups are computed as lower/middle/upper thirds over layer ids.",
        "- With LSSR instrumentation, per-prune-layer DRM-Swap action provenance is used when available.",
        "",
        "## 2. Positive/negative/zero layer distribution",
        "",
        *table_rows_by_case(summary),
        "",
        "## 3. HotpotQA negative layer pattern",
        "",
        f"- Negative samples: {hotpot['samples']} indices `{hotpot['sample_indices']}`",
        f"- Removed early/middle/late: {hotpot['removed']['early']}/{hotpot['removed']['middle']}/{hotpot['removed']['late']}",
        f"- Rescued early/middle/late: {hotpot['rescued']['early']}/{hotpot['rescued']['middle']}/{hotpot['rescued']['late']}",
        "",
        "## 4. MuSiQue positive layer pattern",
        "",
        f"- Positive samples: {musique['samples']} indices `{musique['sample_indices']}`",
        f"- Removed early/middle/late: {musique['removed']['early']}/{musique['removed']['middle']}/{musique['removed']['late']}",
        f"- Rescued early/middle/late: {musique['rescued']['early']}/{musique['rescued']['middle']}/{musique['rescued']['late']}",
        "",
        "## 5. Middle-layer donor safety",
        "",
        f"- Middle donor observed samples: {middle['observed_samples']}",
        f"- Middle donor negative samples: {middle['negative_samples']}",
        f"- Middle donor capacity: {middle['capacity_total']}",
        f"- Middle donor case counts: `{json.dumps(middle['case_counts'], ensure_ascii=False)}`",
        "",
        "## 6. Early/late protection need",
        "",
        f"- Action groups observed: `{signal['groups_observed_in_actions']}`",
        f"- Degenerate single-group signal: {signal['is_degenerate']}",
        f"- Late receiver need: {cross['late_receiver_need_total']}",
        f"- Late query/evidence proxy candidates: {cross['late_query_or_evidence_candidate_count_total']}",
        "",
        "## 7. Layer group vs global score",
        "",
        f"- Layer signal discriminative: {layer_vs_score['layer_signal_discriminative']}",
        f"- Global row scores available: {layer_vs_score['global_row_scores_available']}",
        f"- Conclusion: `{layer_vs_score['conclusion']}`",
        "",
        "## 8. Layer-Specific Answers",
        "",
        f"- Selected/pruned scores available samples: {summary['score_instrumentation_available_count']}/{summary['samples']}",
        f"- Cutoff margin available samples: {summary['cutoff_margin_available_count']}/{summary['samples']}",
        f"- Cutoff margin mean by group: `{json.dumps(risk_gain['cutoff_margin_mean_by_group'], ensure_ascii=False)}`",
        f"- Donor risk mean by group: `{json.dumps(risk_gain['candidate_donor_risk_mean_by_group'], ensure_ascii=False)}`",
        f"- Receiver gain mean by group: `{json.dumps(risk_gain['candidate_receiver_gain_mean_by_group'], ensure_ascii=False)}`",
        f"- Middle donor capacity total: {cross['middle_donor_capacity_total']}",
        f"- Late receiver need total: {cross['late_receiver_need_total']}",
        f"- HotpotQA negative late deletion explanation: `{summary['hotpotqa_negative_late_deletion_risk_explanation']['interpretation']}`",
        f"- MuSiQue positive late receiver explanation: `{summary['musique_positive_late_receiver_gain_explanation']['interpretation']}`",
        f"- Layer-wise signal independence: `{summary['layerwise_signal_independence']['interpretation']}`",
        f"- Budget-neutral layer-aware plan: `{cross['budget_neutral_layer_aware_plan']}`",
        "",
        "## 9. LSSR controller recommendation",
        "",
    ]
    if summary["go_no_go"]["decision"] == "GO":
        lines.append("- Recommendation: implement a tiny LSSR controller audit.")
        followup_lines = [
            "- Next step: use the recorded middle donors and late receivers for an audit-only budget-neutral controller proposal.",
            "- Keep the next step audit-only before any generation run.",
        ]
    else:
        lines.append("- Recommendation: do not implement the LSSR controller yet; first add layer-specific selection/action instrumentation.")
        followup_lines = [
            "- Minimal next instrumentation: persist selected/pruned block ids and scores at each SlimInfer prune layer, plus action-layer provenance for removed/rescued blocks.",
            "- Keep the next step audit-only before any generation run.",
        ]
    lines.extend(followup_lines + [
        "",
        "## 10. GO / NO-GO",
        "",
        f"- Decision: **{summary['go_no_go']['decision']}**",
    ])
    if summary["go_no_go"]["reasons"]:
        for reason in summary["go_no_go"]["reasons"]:
            lines.append(f"- Reason: `{reason}`")
    else:
        lines.append("- Reason: fixed-set artifacts show a non-degenerate layer signal matching the LSSR assumptions.")
    lines.extend([
        "",
        "## Notes",
        "",
        "- This audit may use observed historical prediction scores only to label positive/negative/zero cases after the fact.",
        "- It does not use gold answers to choose blocks or make a controller decision.",
        "- No statistical significance is claimed on this tiny fixed set.",
    ])
    (output_dir / "layer_stratified_sliminfer_audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Path:
    drm_dir = Path(args.drm_dir)
    uasr_dir = Path(args.uasr_dir)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("results") / f"layer_stratified_sliminfer_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    historical = load_historical(drm_dir)
    uasr_rows = load_uasr(uasr_dir)
    score_dump_path = discover_score_dump_path(args.score_dump_path, uasr_dir)
    score_dump_rows = load_score_dumps(score_dump_path)
    lssr_dir = discover_lssr_instrumentation_dir(args.lssr_instrumentation_dir)
    lssr_rows, lssr_summary = load_lssr_instrumentation(lssr_dir)
    num_layers, layer_source = infer_layer_count(drm_dir, historical, score_dump_rows)

    rows = [
        audit_one(key, historical, uasr_rows, score_dump_rows, lssr_rows, num_layers)
        for key in fixed_keys()
    ]
    summary = summarize(rows, num_layers, layer_source, score_dump_path, lssr_summary)
    summary["input_artifacts"] = {
        "drm_dir": str(drm_dir),
        "uasr_dir": str(uasr_dir),
        "score_dump_path": str(score_dump_path or ""),
        "lssr_instrumentation_dir": str(lssr_dir or ""),
    }

    write_jsonl(output_dir / "layer_stratified_sliminfer_audit.jsonl", rows)
    write_csv(output_dir / "layer_stratified_sliminfer_audit.csv", rows, CSV_FIELDS)
    write_json(output_dir / "layer_stratified_sliminfer_audit_summary.json", summary)
    write_report(output_dir, args, summary)
    print(output_dir)
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drm_dir", default=str(DEFAULT_DRM_DIR))
    parser.add_argument("--uasr_dir", default=str(DEFAULT_UASR_DIR))
    parser.add_argument("--score_dump_path", default="")
    parser.add_argument("--lssr_instrumentation_dir", default="")
    parser.add_argument("--output_dir", default="")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
