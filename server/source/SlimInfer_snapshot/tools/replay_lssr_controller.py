#!/usr/bin/env python3
"""Offline replay for Layer-Stratified SlimInfer Repair controller policies.

No model is run.  Accepted samples use the historical DRM-Swap prediction as an
upper-bound proxy because true LSSR-controller predictions do not exist yet.
Rejected samples use historical SlimInfer Original predictions.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence, Tuple


DEFAULT_INSTRUMENTATION_DIR = Path("results/lssr_layer_action_instrumentation_tiny_20260707_083604")
DEFAULT_LAYER_AUDIT_DIR = Path("results/layer_stratified_sliminfer_audit_20260707_083609")
DEFAULT_HISTORICAL_DIR = Path("results/drm_budget_neutral_swap_multihop200_20260706_042717")
DEFAULT_ANCHOR_DIR = Path("results/protected_evidence_anchor_audit_20260706_214256")
DEFAULT_UASR_DIR = Path("results/uasr_uncertainty_audit_20260707_053411")

METHOD_ORIGINAL = "sliminfer_original"
METHOD_RANDOM_SWAP = "random_budget_neutral_swap"
METHOD_DRM_SWAP = "budget_neutral_drm_swap"
METHOD_LARGER = "sliminfer_larger"
BASELINE_METHODS = [METHOD_ORIGINAL, METHOD_RANDOM_SWAP, METHOD_DRM_SWAP, METHOD_LARGER]

FIXED_HOTPOTQA = [0, 1, 2, 3, 4, 5, 41, 56, 57, 117]
FIXED_MUSIQUE = [0, 1, 2, 3, 4, 20, 50, 54, 56, 134]
FIXED_KEYS = {
    ("hotpotqa", index) for index in FIXED_HOTPOTQA
}.union({("musique", index) for index in FIXED_MUSIQUE})

HOTPOT_TOP_NEGATIVE = {0, 56, 117, 41, 57}
MUSIQUE_TOP_POSITIVE = {20, 50, 54, 134, 56}

# Fixed controller constants.  These are not swept in this script.
MIDDLE_DONOR_RISK_LOW = 2.0e-6
LATE_RECEIVER_GAIN_HIGH = 4.0e-2
LATE_RECEIVER_AMBIGUITY_CAP = 6
LATE_DELETE_RISK_HIGH = 5.0e-6

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


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def key_of(row: Mapping[str, Any]) -> Key:
    return str(row["task"]), int(row["sample_index"])


def key_label(key: Key) -> str:
    return f"{key[0]}:{key[1]}"


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)


def finite_float(value: Any, *, field: str) -> float:
    try:
        result = float(value)
    except Exception as exc:
        raise RuntimeError(f"missing numeric field {field}: {value!r}") from exc
    if not math.isfinite(result):
        raise RuntimeError(f"non-finite numeric field {field}: {value!r}")
    return result


def finite_values(values: Iterable[Any]) -> List[float]:
    out: List[float] = []
    for value in values:
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            out.append(number)
    return out


def mean_or_none(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def max_or_none(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return max(values)


def min_or_none(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return min(values)


def load_instrumentation(instrumentation_dir: Path) -> Dict[Key, JsonRow]:
    path = instrumentation_dir / "lssr_layer_action_instrumentation.jsonl"
    rows = {key_of(row): row for row in read_jsonl(path) if key_of(row) in FIXED_KEYS}
    missing = sorted(FIXED_KEYS.difference(rows), key=lambda key: (key[0], key[1]))
    if missing:
        raise RuntimeError(f"instrumentation missing fixed samples: {[key_label(key) for key in missing]}")
    return rows


def load_layer_audit(layer_audit_dir: Path) -> Dict[Key, JsonRow]:
    path = layer_audit_dir / "layer_stratified_sliminfer_audit.jsonl"
    return {key_of(row): row for row in read_jsonl(path) if key_of(row) in FIXED_KEYS}


def load_historical(historical_dir: Path) -> Dict[str, Dict[Key, JsonRow]]:
    by_method: Dict[str, Dict[Key, JsonRow]] = defaultdict(dict)
    for row in read_jsonl(historical_dir / "audit" / "per_sample_audit.jsonl"):
        method = str(row.get("method_key") or "")
        key = key_of(row)
        if method and key in FIXED_KEYS:
            by_method[method][key] = row
    missing_methods = [method for method in BASELINE_METHODS if method not in by_method]
    if missing_methods:
        raise RuntimeError(f"historical audit missing methods: {missing_methods}")
    for method in BASELINE_METHODS:
        missing_keys = sorted(FIXED_KEYS.difference(by_method[method]), key=lambda key: (key[0], key[1]))
        if missing_keys:
            raise RuntimeError(f"historical {method} missing fixed samples: {[key_label(key) for key in missing_keys]}")
    return dict(by_method)


def load_trigger_set(path: Path, field: str) -> set[Key]:
    if not path.exists():
        return set()
    triggers: set[Key] = set()
    for row in read_jsonl(path):
        key = key_of(row)
        if key in FIXED_KEYS and boolish(row.get(field)):
            triggers.add(key)
    return triggers


def score_of(row: Mapping[str, Any]) -> float:
    return finite_float(row.get("prediction_score"), field="prediction_score")


def task_scores(scores: Mapping[Key, float]) -> Dict[str, float]:
    grouped: Dict[str, List[float]] = defaultdict(list)
    for (task, _), score in scores.items():
        grouped[task].append(float(score))
    return {task: sum(values) / len(values) for task, values in sorted(grouped.items())}


def macro(per_task: Mapping[str, float]) -> float:
    return sum(per_task.values()) / max(1, len(per_task))


def group_values(per_layer: Sequence[Mapping[str, Any]], group: str, field: str) -> List[float]:
    values: List[float] = []
    for layer in per_layer:
        if layer.get("layer_group") != group:
            continue
        raw = layer.get(field)
        if isinstance(raw, list):
            values.extend(finite_values(raw))
        elif raw is not None:
            values.extend(finite_values([raw]))
    return values


def group_count(per_layer: Sequence[Mapping[str, Any]], group: str, field: str) -> int:
    count = 0
    for layer in per_layer:
        if layer.get("layer_group") != group:
            continue
        raw = layer.get(field)
        if isinstance(raw, list):
            count += len(raw)
        elif raw not in (None, ""):
            count += int(raw)
    return count


def actual_swap_count(per_layer: Sequence[Mapping[str, Any]], group: str) -> int:
    total = 0
    for layer in per_layer:
        if layer.get("layer_group") != group:
            continue
        try:
            total += int(layer.get("actual_swap_count_drm_swap") or 0)
        except (TypeError, ValueError):
            continue
    return total


def prune_layers(rows: Mapping[Key, Mapping[str, Any]]) -> List[int]:
    layers: set[int] = set()
    for row in rows.values():
        for layer in row.get("per_layer") or []:
            if layer.get("layer_id") is not None:
                layers.add(int(layer["layer_id"]))
    return sorted(layers)


def infer_layer_group_definition(rows: Mapping[Key, Mapping[str, Any]]) -> Mapping[str, Any]:
    for row in rows.values():
        definition = row.get("layer_group_definition")
        if isinstance(definition, dict) and definition:
            return definition
    return {}


def has_early_action_row(rows: Mapping[Key, Mapping[str, Any]]) -> bool:
    for row in rows.values():
        for layer in row.get("per_layer") or []:
            if layer.get("layer_group") == "early":
                return True
    return False


def extract_features(row: Mapping[str, Any]) -> JsonRow:
    per_layer = row.get("per_layer") or []
    middle_risks = group_values(per_layer, "middle", "candidate_donor_risk_proxy")
    late_gains = group_values(per_layer, "late", "candidate_receiver_gain_proxy")
    late_delete_risks = group_values(per_layer, "late", "candidate_donor_risk_proxy")
    middle_margins = group_values(per_layer, "middle", "cutoff_margin")
    late_margins = group_values(per_layer, "late", "cutoff_margin")
    middle_donor_capacity = int(row.get("middle_donor_capacity") or group_count(per_layer, "middle", "candidate_donor_block_ids"))
    late_receiver_need = int(row.get("late_receiver_need") or group_count(per_layer, "late", "candidate_receiver_block_ids"))
    late_query_or_evidence_count = int(row.get("late_query_or_evidence_candidate_count") or 0)
    proposed_swap_count = min(middle_donor_capacity, late_receiver_need)
    fallback_reason = None
    if row.get("layerwise_repair_reason") not in (None, "", "middle_donor_capacity_and_late_receiver_need_nonzero"):
        fallback_reason = row.get("layerwise_repair_reason")
    return {
        "middle_donor_capacity": middle_donor_capacity,
        "late_receiver_need": late_receiver_need,
        "late_query_or_evidence_candidate_count": late_query_or_evidence_count,
        "middle_donor_risk_mean": mean_or_none(middle_risks),
        "middle_donor_risk_max": max_or_none(middle_risks),
        "middle_donor_risk_min": min_or_none(middle_risks),
        "late_receiver_gain_mean": mean_or_none(late_gains),
        "late_receiver_gain_max": max_or_none(late_gains),
        "late_receiver_gain_min": min_or_none(late_gains),
        "late_receiver_high_gain_count": sum(gain >= LATE_RECEIVER_GAIN_HIGH for gain in late_gains),
        "late_delete_risk_max": max_or_none(late_delete_risks),
        "middle_cutoff_margin_mean": mean_or_none(middle_margins),
        "late_cutoff_margin_mean": mean_or_none(late_margins),
        "actual_middle_swap_count": actual_swap_count(per_layer, "middle"),
        "actual_late_swap_count": actual_swap_count(per_layer, "late"),
        "proposed_swap_count": proposed_swap_count,
        "budget_preserved": proposed_swap_count > 0,
        "protected_violation": 0,
        "fallback_reason": fallback_reason,
        "selection_behavior_unchanged": boolish(row.get("selection_behavior_unchanged")),
        "layerwise_repair_possible": boolish(row.get("layerwise_repair_possible")),
        "no_early_delete": True,
        "proposed_delete_group": "middle" if proposed_swap_count > 0 else "",
        "proposed_add_group": "late" if proposed_swap_count > 0 else "",
        "proposed_late_delete_count": 0,
    }


def base_safety(feature: Mapping[str, Any]) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    if int(feature["middle_donor_capacity"]) <= 0:
        reasons.append("middle_donor_capacity_zero")
    if int(feature["late_receiver_need"]) <= 0:
        reasons.append("late_receiver_need_zero")
    if feature.get("proposed_delete_group") != "middle":
        reasons.append("donor_not_middle")
    if feature.get("proposed_add_group") != "late":
        reasons.append("receiver_not_late")
    if not bool(feature.get("budget_preserved")):
        reasons.append("budget_not_preserved")
    if int(feature.get("protected_violation") or 0) != 0:
        reasons.append("protected_violation")
    if feature.get("fallback_reason") is not None:
        reasons.append("fallback_reason_present")
    if not bool(feature.get("selection_behavior_unchanged")):
        reasons.append("selection_behavior_changed")
    if not bool(feature.get("no_early_delete")):
        reasons.append("early_delete_proposed")
    return not reasons, reasons


def risk_gain_pass(feature: Mapping[str, Any], *, concentrated: bool) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    risk_mean = feature.get("middle_donor_risk_mean")
    gain_max = feature.get("late_receiver_gain_max")
    late_qe = int(feature.get("late_query_or_evidence_candidate_count") or 0)
    if risk_mean is None or float(risk_mean) > MIDDLE_DONOR_RISK_LOW:
        reasons.append("middle_donor_risk_not_low")
    if gain_max is None or float(gain_max) < LATE_RECEIVER_GAIN_HIGH:
        reasons.append("late_receiver_gain_not_high")
    if late_qe <= 0:
        reasons.append("late_query_or_evidence_proxy_absent")
    if concentrated and late_qe > LATE_RECEIVER_AMBIGUITY_CAP:
        reasons.append("late_receiver_signal_too_broad")
    return not reasons, reasons


def policy_a(feature: Mapping[str, Any]) -> Tuple[bool, List[str]]:
    ok, reasons = base_safety(feature)
    return ok, reasons


def policy_b(feature: Mapping[str, Any]) -> Tuple[bool, List[str]]:
    ok, reasons = base_safety(feature)
    rg_ok, rg_reasons = risk_gain_pass(feature, concentrated=True)
    reasons.extend(rg_reasons)
    return ok and rg_ok, reasons


def policy_c(feature: Mapping[str, Any]) -> Tuple[bool, List[str]]:
    ok, reasons = base_safety(feature)
    rg_ok, rg_reasons = risk_gain_pass(feature, concentrated=True)
    reasons.extend(rg_reasons)
    if int(feature.get("proposed_late_delete_count") or 0) != 0:
        reasons.append("late_delete_proposed")
    late_delete_risk = feature.get("late_delete_risk_max")
    if late_delete_risk is not None and float(late_delete_risk) > LATE_DELETE_RISK_HIGH:
        reasons.append("historical_proxy_late_delete_high_risk")
    return ok and rg_ok and not reasons, reasons


POLICIES: List[Tuple[str, str, Callable[[Mapping[str, Any]], Tuple[bool, List[str]]]]] = [
    ("policy_a", "Middle-Donor Late-Receiver", policy_a),
    ("policy_b", "Conservative Middle-Donor", policy_b),
    ("policy_c", "Late-Protection Filter", policy_c),
]


def baseline_scores_for_keys(
    historical: Mapping[str, Mapping[Key, Mapping[str, Any]]],
    keys: Sequence[Key],
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
    task_by_method = {
        method: task_scores({key: score_of(historical[method][key]) for key in keys})
        for method in BASELINE_METHODS
    }
    macro_by_method = {method: macro(scores) for method, scores in task_by_method.items()}
    return task_by_method, macro_by_method


def replay_policy(
    policy_id: str,
    policy_name: str,
    decide: Callable[[Mapping[str, Any]], Tuple[bool, List[str]]],
    instrumentation: Mapping[Key, Mapping[str, Any]],
    layer_audit: Mapping[Key, Mapping[str, Any]],
    historical: Mapping[str, Mapping[Key, Mapping[str, Any]]],
    anchor_triggers: set[Key],
    uasr_triggers: set[Key],
) -> JsonRow:
    keys = sorted(instrumentation, key=lambda key: (key[0], key[1]))
    accepted: set[Key] = set()
    decisions: List[JsonRow] = []
    replay_scores: Dict[Key, float] = {}
    budget_violation_count = 0
    protected_violation_count = 0
    fallback_count = 0
    for key in keys:
        feature = extract_features(instrumentation[key])
        accept, reject_reasons = decide(feature)
        if accept:
            accepted.add(key)
        selected_method = METHOD_DRM_SWAP if accept else METHOD_ORIGINAL
        selected_score = score_of(historical[selected_method][key])
        replay_scores[key] = selected_score
        if not bool(feature["budget_preserved"]):
            budget_violation_count += 1
        if int(feature["protected_violation"]) != 0:
            protected_violation_count += 1
        if feature["fallback_reason"] is not None:
            fallback_count += 1
        original_score = score_of(historical[METHOD_ORIGINAL][key])
        drm_score = score_of(historical[METHOD_DRM_SWAP][key])
        random_score = score_of(historical[METHOD_RANDOM_SWAP][key])
        larger_score = score_of(historical[METHOD_LARGER][key])
        audit_row = layer_audit.get(key, {})
        decisions.append({
            "policy_id": policy_id,
            "policy_name": policy_name,
            "task": key[0],
            "sample_index": key[1],
            "sample_id": instrumentation[key].get("sample_id", historical[METHOD_ORIGINAL][key].get("sample_id")),
            "lssr_triggered": accept,
            "selected_proxy_method": selected_method,
            "upper_bound_proxy_used": accept,
            "reject_reasons": ";".join(reject_reasons),
            "trigger_reason": "middle_to_late_layer_aware_repair" if accept else "default_original",
            "case_type_vs_original_for_analysis_only": audit_row.get("case_type_vs_original", instrumentation[key].get("case_type_vs_original")),
            "case_type_vs_random_swap_for_analysis_only": audit_row.get("case_type_vs_random_swap", instrumentation[key].get("case_type_vs_random_swap")),
            "middle_donor_capacity": feature["middle_donor_capacity"],
            "late_receiver_need": feature["late_receiver_need"],
            "late_query_or_evidence_candidate_count": feature["late_query_or_evidence_candidate_count"],
            "middle_donor_risk_mean": feature["middle_donor_risk_mean"],
            "middle_donor_risk_max": feature["middle_donor_risk_max"],
            "late_receiver_gain_mean": feature["late_receiver_gain_mean"],
            "late_receiver_gain_max": feature["late_receiver_gain_max"],
            "late_receiver_high_gain_count": feature["late_receiver_high_gain_count"],
            "late_delete_risk_max": feature["late_delete_risk_max"],
            "middle_cutoff_margin_mean": feature["middle_cutoff_margin_mean"],
            "late_cutoff_margin_mean": feature["late_cutoff_margin_mean"],
            "actual_middle_swap_count": feature["actual_middle_swap_count"],
            "actual_late_swap_count": feature["actual_late_swap_count"],
            "proposed_swap_count": feature["proposed_swap_count"],
            "proposed_delete_group": feature["proposed_delete_group"],
            "proposed_add_group": feature["proposed_add_group"],
            "proposed_late_delete_count": feature["proposed_late_delete_count"],
            "budget_preserved": feature["budget_preserved"],
            "protected_violation": feature["protected_violation"],
            "fallback_reason": feature["fallback_reason"],
            "selection_behavior_unchanged": feature["selection_behavior_unchanged"],
            "new_trigger_vs_protected_anchor": accept and key not in anchor_triggers,
            "new_trigger_vs_uasr": accept and key not in uasr_triggers,
            "new_trigger_vs_protected_anchor_or_uasr": accept and key not in anchor_triggers.union(uasr_triggers),
            "selected_score": selected_score,
            "original_score": original_score,
            "drm_swap_score": drm_score,
            "random_swap_score": random_score,
            "larger_score": larger_score,
            "drm_minus_original_score_for_analysis_only": drm_score - original_score,
            "drm_minus_random_score_for_analysis_only": drm_score - random_score,
        })
    per_task = task_scores(replay_scores)
    replay_macro = macro(per_task)
    baseline_task_scores, baseline_macros = baseline_scores_for_keys(historical, keys)
    hotpot_total = sum(1 for key in keys if key[0] == "hotpotqa" and key[1] in HOTPOT_TOP_NEGATIVE)
    hotpot_filtered = sum(
        1 for key in keys if key[0] == "hotpotqa" and key[1] in HOTPOT_TOP_NEGATIVE and key not in accepted
    )
    musique_total = sum(1 for key in keys if key[0] == "musique" and key[1] in MUSIQUE_TOP_POSITIVE)
    musique_retained = sum(
        1 for key in keys if key[0] == "musique" and key[1] in MUSIQUE_TOP_POSITIVE and key in accepted
    )
    pnz = Counter()
    for key, score in replay_scores.items():
        delta = score - score_of(historical[METHOD_ORIGINAL][key])
        if delta > 1e-9:
            pnz["positive"] += 1
        elif delta < -1e-9:
            pnz["negative"] += 1
        else:
            pnz["zero"] += 1
    anchor_in_scope = anchor_triggers.intersection(set(keys))
    uasr_in_scope = uasr_triggers.intersection(set(keys))
    accepted_vs_anchor_or_uasr = accepted.difference(anchor_in_scope.union(uasr_in_scope))
    accepted_equal_anchor = accepted == anchor_in_scope
    accepted_subset_anchor = bool(accepted) and accepted.issubset(anchor_in_scope)
    trigger_count = len(accepted)
    reasons: List[str] = []
    if trigger_count == 0:
        reasons.append("trigger_all_reject")
    if trigger_count == len(keys):
        reasons.append("trigger_all_accept")
    if not accepted_vs_anchor_or_uasr:
        reasons.append("no_new_trigger_vs_protected_anchor_or_uasr")
    if accepted_equal_anchor or accepted_subset_anchor:
        reasons.append("trigger_equivalent_to_or_subset_of_protected_anchor")
    if hotpot_total and hotpot_filtered < math.ceil(0.6 * hotpot_total):
        reasons.append("hotpotqa_top_negative_filter_not_high")
    if musique_total and musique_retained <= 0:
        reasons.append("musique_positive_not_retained")
    if replay_macro < baseline_macros[METHOD_ORIGINAL] - 1e-9:
        reasons.append("replay_macro_below_original")
    if budget_violation_count != 0:
        reasons.append("budget_violation")
    if protected_violation_count != 0:
        reasons.append("protected_violation")
    if fallback_count != 0:
        reasons.append("fallback_used")
    return {
        "policy_id": policy_id,
        "policy_name": policy_name,
        "decisions": decisions,
        "task_scores": per_task,
        "macro": replay_macro,
        "baseline_task_scores": baseline_task_scores,
        "baseline_macros": baseline_macros,
        "delta_vs": {method: replay_macro - value for method, value in baseline_macros.items()},
        "trigger_count": trigger_count,
        "trigger_rate": trigger_count / max(1, len(keys)),
        "accepted_sample_list": [key_label(key) for key in sorted(accepted)],
        "hotpotqa_top_negative_filtered": hotpot_filtered,
        "hotpotqa_top_negative_total": hotpot_total,
        "hotpotqa_top_negative_filter_rate": hotpot_filtered / max(1, hotpot_total),
        "musique_positive_retained": musique_retained,
        "musique_positive_total": musique_total,
        "musique_positive_retention_rate": musique_retained / max(1, musique_total),
        "positive_negative_zero_vs_original": dict(pnz),
        "anchor_trigger_count_in_scope": len(anchor_in_scope),
        "uasr_trigger_count_in_scope": len(uasr_in_scope),
        "anchor_trigger_overlap_count": len(accepted.intersection(anchor_in_scope)),
        "uasr_trigger_overlap_count": len(accepted.intersection(uasr_in_scope)),
        "accepted_equal_anchor_on_tiny": accepted_equal_anchor,
        "accepted_subset_of_anchor": accepted_subset_anchor,
        "new_trigger_vs_anchor_list": [key_label(key) for key in sorted(accepted.difference(anchor_in_scope))],
        "new_trigger_vs_uasr_list": [key_label(key) for key in sorted(accepted.difference(uasr_in_scope))],
        "new_trigger_vs_anchor_or_uasr_list": [key_label(key) for key in sorted(accepted_vs_anchor_or_uasr)],
        "budget_violation_count": budget_violation_count,
        "protected_violation_count": protected_violation_count,
        "fallback_count": fallback_count,
        "go_no_go": {
            "decision": "GO" if not reasons else "NO-GO",
            "reasons": reasons,
        },
    }


def aggregate_feature_summary(instrumentation: Mapping[Key, Mapping[str, Any]]) -> JsonRow:
    features = [extract_features(row) for row in instrumentation.values()]
    return {
        "middle_donor_capacity_total": sum(int(feature["middle_donor_capacity"]) for feature in features),
        "late_receiver_need_total": sum(int(feature["late_receiver_need"]) for feature in features),
        "late_query_or_evidence_candidate_count_total": sum(
            int(feature["late_query_or_evidence_candidate_count"]) for feature in features
        ),
        "layerwise_repair_possible_count": sum(bool(feature["layerwise_repair_possible"]) for feature in features),
        "selection_behavior_unchanged_count": sum(bool(feature["selection_behavior_unchanged"]) for feature in features),
        "middle_donor_risk_mean_mean": mean_or_none([
            float(feature["middle_donor_risk_mean"])
            for feature in features
            if feature.get("middle_donor_risk_mean") is not None
        ]),
        "late_receiver_gain_max_mean": mean_or_none([
            float(feature["late_receiver_gain_max"])
            for feature in features
            if feature.get("late_receiver_gain_max") is not None
        ]),
        "late_delete_risk_max_mean": mean_or_none([
            float(feature["late_delete_risk_max"])
            for feature in features
            if feature.get("late_delete_risk_max") is not None
        ]),
    }


def choose_recommended_policy(policy_results: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    go_results = [result for result in policy_results if result["go_no_go"]["decision"] == "GO"]
    if go_results:
        # Prefer the first conservative GO policy over the broad diagnostic policy.
        for result in go_results:
            if result["policy_id"] in {"policy_b", "policy_c"}:
                return result
        return go_results[0]
    return max(policy_results, key=lambda result: (
        result["macro"],
        result["hotpotqa_top_negative_filter_rate"],
        result["musique_positive_retention_rate"],
    ))


def write_report(
    output_dir: Path,
    args: argparse.Namespace,
    layer_group_definition: Mapping[str, Any],
    prune_layer_ids: Sequence[int],
    early_action_row_present: bool,
    feature_summary: Mapping[str, Any],
    policy_results: Sequence[Mapping[str, Any]],
    recommended: Mapping[str, Any],
) -> None:
    lines = [
        "# LSSR Controller Offline Replay",
        "",
        f"- Result directory: `{output_dir.resolve()}`",
        f"- Layer action instrumentation input: `{args.instrumentation_dir}`",
        f"- Layer audit input: `{args.layer_audit_dir}`",
        f"- Historical prediction input: `{args.historical_dir}`",
        "- Model/GPU used: **No**",
        "- New prediction generated: **No**",
        "- Scorer used: **No**",
        "- SlimInfer selector modified: **No**",
        "- Gold/final score/final prediction used as gate feature: **No**",
        "- Accept action: historical DRM-Swap prediction as **upper-bound proxy**",
        "- Reject action: historical SlimInfer Original prediction",
        "",
        "## Layer Setup",
        "",
        f"- Layer group definition: `{json.dumps(layer_group_definition, ensure_ascii=False)}`",
        f"- Prune layers observed: `{json.dumps(list(prune_layer_ids), ensure_ascii=False)}`",
        f"- Early action row present: {early_action_row_present}",
        f"- Early has no action row: {not early_action_row_present}",
        f"- Middle donor capacity: {feature_summary['middle_donor_capacity_total']}",
        f"- Late receiver need: {feature_summary['late_receiver_need_total']}",
        f"- Layerwise repair possible: {feature_summary['layerwise_repair_possible_count']}/20",
        "",
        "## Fixed Policies",
        "",
        f"- Policy A: middle donor exists + late receiver exists + base safety pass.",
        f"- Policy B: Policy A plus middle donor risk mean <= {MIDDLE_DONOR_RISK_LOW:g}, late receiver gain max >= {LATE_RECEIVER_GAIN_HIGH:g}, and late query/evidence candidate count <= {LATE_RECEIVER_AMBIGUITY_CAP}.",
        f"- Policy C: Policy B plus no proposed late deletion and historical proxy late delete risk max <= {LATE_DELETE_RISK_HIGH:g}.",
        "",
        "## Replay Macro",
        "",
    ]
    for result in policy_results:
        lines.extend([
            f"### {result['policy_name']}",
            "",
            f"- Decision: **{result['go_no_go']['decision']}**",
            f"- Trigger count: {result['trigger_count']}/20 ({result['trigger_rate']:.2%})",
            f"- Accepted samples: `{json.dumps(result['accepted_sample_list'], ensure_ascii=False)}`",
            f"- New triggers outside Protected Anchor / UASR: `{json.dumps(result['new_trigger_vs_anchor_or_uasr_list'], ensure_ascii=False)}`",
            f"- HotpotQA top negative filtered: {result['hotpotqa_top_negative_filtered']}/{result['hotpotqa_top_negative_total']} ({result['hotpotqa_top_negative_filter_rate']:.2%})",
            f"- MuSiQue positive retained: {result['musique_positive_retained']}/{result['musique_positive_total']} ({result['musique_positive_retention_rate']:.2%})",
            f"- Task scores: `{json.dumps({k: round(v, 4) for k, v in result['task_scores'].items()}, ensure_ascii=False)}`",
            f"- Macro: {result['macro']:.4f}",
            f"- Delta vs Original: {result['delta_vs'][METHOD_ORIGINAL]:+.4f}",
            f"- Delta vs Random-Swap: {result['delta_vs'][METHOD_RANDOM_SWAP]:+.4f}",
            f"- Delta vs DRM-Swap: {result['delta_vs'][METHOD_DRM_SWAP]:+.4f}",
            f"- Delta vs Larger: {result['delta_vs'][METHOD_LARGER]:+.4f}",
            f"- Budget violation: {result['budget_violation_count']}",
            f"- Protected violation: {result['protected_violation_count']}",
            f"- Positive/negative/zero vs Original: `{json.dumps(result['positive_negative_zero_vs_original'], ensure_ascii=False)}`",
        ])
        for reason in result["go_no_go"]["reasons"]:
            lines.append(f"- NO-GO reason: `{reason}`")
        lines.append("")
    lines.extend([
        "## Recommendation",
        "",
        f"- Recommended policy: **{recommended['policy_name']}**",
        f"- GO/NO-GO: **{recommended['go_no_go']['decision']}**",
        f"- Recommend tiny quality smoke: {recommended['go_no_go']['decision'] == 'GO'}",
        "- Interpretation: accepted repairs are budget-neutral middle-to-late layer-aware proposals; early layers are protected and never donors.",
        "- Caveat: this is an offline upper-bound proxy replay, not a real LSSR quality result.",
        "",
    ])
    (output_dir / "lssr_controller_replay_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Path:
    instrumentation_dir = Path(args.instrumentation_dir)
    layer_audit_dir = Path(args.layer_audit_dir)
    historical_dir = Path(args.historical_dir)
    output_dir = Path(args.output_dir) if args.output_dir else Path("results") / f"lssr_controller_replay_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=False)

    instrumentation = load_instrumentation(instrumentation_dir)
    layer_audit = load_layer_audit(layer_audit_dir)
    historical = load_historical(historical_dir)
    anchor_triggers = load_trigger_set(Path(args.anchor_dir) / "protected_evidence_anchor_audit.jsonl", "anchor_gate_triggered")
    uasr_triggers = load_trigger_set(Path(args.uasr_dir) / "uasr_uncertainty_audit.jsonl", "uasr_gate_triggered")

    layer_group_definition = infer_layer_group_definition(instrumentation)
    prune_layer_ids = prune_layers(instrumentation)
    early_action_row_present = has_early_action_row(instrumentation)
    feature_summary = aggregate_feature_summary(instrumentation)

    policy_results = [
        replay_policy(policy_id, policy_name, decide, instrumentation, layer_audit, historical, anchor_triggers, uasr_triggers)
        for policy_id, policy_name, decide in POLICIES
    ]
    recommended = choose_recommended_policy(policy_results)

    all_decisions = [decision for result in policy_results for decision in result["decisions"]]
    write_jsonl(output_dir / "lssr_controller_replay_decisions.jsonl", all_decisions)
    write_csv(
        output_dir / "lssr_controller_replay_decisions.csv",
        all_decisions,
        [
            "policy_id",
            "policy_name",
            "task",
            "sample_index",
            "sample_id",
            "lssr_triggered",
            "selected_proxy_method",
            "upper_bound_proxy_used",
            "reject_reasons",
            "trigger_reason",
            "case_type_vs_original_for_analysis_only",
            "case_type_vs_random_swap_for_analysis_only",
            "middle_donor_capacity",
            "late_receiver_need",
            "late_query_or_evidence_candidate_count",
            "middle_donor_risk_mean",
            "middle_donor_risk_max",
            "late_receiver_gain_mean",
            "late_receiver_gain_max",
            "late_receiver_high_gain_count",
            "late_delete_risk_max",
            "middle_cutoff_margin_mean",
            "late_cutoff_margin_mean",
            "actual_middle_swap_count",
            "actual_late_swap_count",
            "proposed_swap_count",
            "proposed_delete_group",
            "proposed_add_group",
            "proposed_late_delete_count",
            "budget_preserved",
            "protected_violation",
            "fallback_reason",
            "selection_behavior_unchanged",
            "new_trigger_vs_protected_anchor",
            "new_trigger_vs_uasr",
            "new_trigger_vs_protected_anchor_or_uasr",
            "selected_score",
            "original_score",
            "drm_swap_score",
            "random_swap_score",
            "larger_score",
            "drm_minus_original_score_for_analysis_only",
            "drm_minus_random_score_for_analysis_only",
        ],
    )
    score_rows = []
    for result in policy_results:
        score_rows.append({
            "policy_id": result["policy_id"],
            "policy_name": result["policy_name"],
            "hotpotqa": result["task_scores"].get("hotpotqa"),
            "musique": result["task_scores"].get("musique"),
            "macro": result["macro"],
            "delta_vs_original": result["delta_vs"][METHOD_ORIGINAL],
            "delta_vs_random_swap": result["delta_vs"][METHOD_RANDOM_SWAP],
            "delta_vs_drm_swap": result["delta_vs"][METHOD_DRM_SWAP],
            "delta_vs_larger": result["delta_vs"][METHOD_LARGER],
            "trigger_count": result["trigger_count"],
            "hotpotqa_top_negative_filter_rate": result["hotpotqa_top_negative_filter_rate"],
            "musique_positive_retention_rate": result["musique_positive_retention_rate"],
            "budget_violation_count": result["budget_violation_count"],
            "protected_violation_count": result["protected_violation_count"],
            "verdict": result["go_no_go"]["decision"],
        })
    write_csv(
        output_dir / "lssr_controller_replay_scores.csv",
        score_rows,
        [
            "policy_id",
            "policy_name",
            "hotpotqa",
            "musique",
            "macro",
            "delta_vs_original",
            "delta_vs_random_swap",
            "delta_vs_drm_swap",
            "delta_vs_larger",
            "trigger_count",
            "hotpotqa_top_negative_filter_rate",
            "musique_positive_retention_rate",
            "budget_violation_count",
            "protected_violation_count",
            "verdict",
        ],
    )
    payload = {
        "instrumentation_dir": str(instrumentation_dir),
        "layer_audit_dir": str(layer_audit_dir),
        "historical_dir": str(historical_dir),
        "anchor_dir": str(args.anchor_dir),
        "uasr_dir": str(args.uasr_dir),
        "model_or_gpu_used": False,
        "generation_performed": False,
        "scorer_used": False,
        "sliminfer_selector_modified": False,
        "gold_answer_used_for_gate": False,
        "final_score_or_prediction_used_for_gate": False,
        "upper_bound_proxy_replay": True,
        "fixed_samples": {
            "hotpotqa": FIXED_HOTPOTQA,
            "musique": FIXED_MUSIQUE,
        },
        "controller_constants": {
            "middle_donor_risk_low": MIDDLE_DONOR_RISK_LOW,
            "late_receiver_gain_high": LATE_RECEIVER_GAIN_HIGH,
            "late_receiver_ambiguity_cap": LATE_RECEIVER_AMBIGUITY_CAP,
            "late_delete_risk_high": LATE_DELETE_RISK_HIGH,
        },
        "layer_group_definition": layer_group_definition,
        "prune_layers": list(prune_layer_ids),
        "early_action_row_present": early_action_row_present,
        "early_has_no_action_row": not early_action_row_present,
        "feature_summary": feature_summary,
        "policy_results": [
            {key: value for key, value in result.items() if key != "decisions"}
            for result in policy_results
        ],
        "recommended_policy": {
            "policy_id": recommended["policy_id"],
            "policy_name": recommended["policy_name"],
            "go_no_go": recommended["go_no_go"],
        },
    }
    write_json(output_dir / "lssr_controller_replay_summary.json", payload)
    write_report(
        output_dir,
        args,
        layer_group_definition,
        prune_layer_ids,
        early_action_row_present,
        feature_summary,
        policy_results,
        recommended,
    )
    print(json.dumps({
        "status": "ok",
        "output_dir": str(output_dir.resolve()),
        "recommended_policy": recommended["policy_id"],
        "recommended_decision": recommended["go_no_go"]["decision"],
        "policy_trigger_counts": {
            result["policy_id"]: result["trigger_count"] for result in policy_results
        },
        "policy_macros": {
            result["policy_id"]: result["macro"] for result in policy_results
        },
    }, ensure_ascii=False, indent=2))
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instrumentation_dir", default=str(DEFAULT_INSTRUMENTATION_DIR))
    parser.add_argument("--layer_audit_dir", default=str(DEFAULT_LAYER_AUDIT_DIR))
    parser.add_argument("--historical_dir", default=str(DEFAULT_HISTORICAL_DIR))
    parser.add_argument("--anchor_dir", default=str(DEFAULT_ANCHOR_DIR))
    parser.add_argument("--uasr_dir", default=str(DEFAULT_UASR_DIR))
    parser.add_argument("--output_dir", default="")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
