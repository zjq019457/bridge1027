#!/usr/bin/env python3
"""Offline replay for deterministic Risk-Aware Fallback policies.

This script does not run a model and does not generate predictions.  It merges
existing tiny audit artifacts and replays fixed deterministic policies:

  reject -> historical SlimInfer Original prediction
  accept -> historical Budget-neutral DRM-Swap prediction as an upper-bound proxy

Gold answers, final predictions, and final scores are not used by the policy.
Historical per-sample scores are used only after the policy decision is made to
evaluate the replay.
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


DEFAULT_HISTORICAL_DIR = Path("results/drm_budget_neutral_swap_multihop200_20260706_042717")
DEFAULT_PROXY_DIR = Path("results/residual_aware_proxy_audit_tiny_20260706_204145")
DEFAULT_ANCHOR_DIR = Path("results/protected_evidence_anchor_audit_20260706_214256")
DEFAULT_COVERAGE_DIR = Path("results/evidence_coverage_controller_audit_20260706_220014")

METHOD_ORIGINAL = "sliminfer_original"
METHOD_RANDOM_SWAP = "random_budget_neutral_swap"
METHOD_DRM_SWAP = "budget_neutral_drm_swap"
METHOD_LARGER = "sliminfer_larger"
BASELINE_METHODS = [METHOD_ORIGINAL, METHOD_RANDOM_SWAP, METHOD_DRM_SWAP, METHOD_LARGER]

HOTPOT_TOP_NEGATIVE = {0, 56, 117, 41, 57}
MUSIQUE_TOP_POSITIVE = {20, 50, 54, 134, 56}
TINY_SAMPLE_INDICES = {
    "hotpotqa": [0, 1, 2, 3, 4, 5, 41, 56, 57, 117],
    "musique": [0, 1, 2, 3, 4, 20, 50, 54, 56, 134],
}

DELETE_RISK_MAX = 0.20
RESIDUAL_BENEFIT_MIN = -1.0


Key = Tuple[str, int]
JsonRow = Dict[str, Any]


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


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def key_of(row: Mapping[str, Any]) -> Key:
    return str(row["task"]), int(row["sample_index"])


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null", "nan", "unavailable"}:
        return True
    if isinstance(value, float) and not math.isfinite(value):
        return True
    return False


def finite_float(value: Any, *, field: str) -> float:
    if is_missing(value):
        raise RuntimeError(f"missing numeric field {field}: {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"non-finite numeric field {field}: {value!r}")
    return result


def list_value(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def load_historical(historical_dir: Path) -> Dict[str, Dict[Key, JsonRow]]:
    path = historical_dir / "audit" / "per_sample_audit.jsonl"
    by_method: Dict[str, Dict[Key, JsonRow]] = defaultdict(dict)
    for row in read_jsonl(path):
        method = str(row.get("method_key") or "")
        if method:
            by_method[method][key_of(row)] = row
    missing = [method for method in BASELINE_METHODS if method not in by_method]
    if missing:
        raise RuntimeError(f"historical audit missing methods: {missing}")
    return dict(by_method)


def load_proxy(proxy_dir: Path) -> Dict[Key, JsonRow]:
    path = proxy_dir / "audit" / "gate_feature_audit.jsonl"
    return {key_of(row): row for row in read_jsonl(path)}


def load_anchor(anchor_dir: Path) -> Dict[Key, JsonRow]:
    path = anchor_dir / "protected_evidence_anchor_audit.jsonl"
    return {key_of(row): row for row in read_jsonl(path)}


def load_coverage(coverage_dir: Path) -> Dict[Key, JsonRow]:
    path = coverage_dir / "evidence_coverage_audit.jsonl"
    return {key_of(row): row for row in read_jsonl(path)}


def tiny_keys() -> List[Key]:
    return [(task, idx) for task, indices in TINY_SAMPLE_INDICES.items() for idx in indices]


def sample_id_for(key: Key, historical: Mapping[str, Mapping[Key, Mapping[str, Any]]], proxy: Mapping[Key, Mapping[str, Any]]) -> str:
    if key in proxy and not is_missing(proxy[key].get("sample_id")):
        return str(proxy[key]["sample_id"])
    return str(historical[METHOD_ORIGINAL][key].get("sample_id", ""))


def score_of(row: Mapping[str, Any]) -> float:
    return finite_float(row.get("prediction_score"), field="prediction_score")


def task_scores(scores: Mapping[Key, float]) -> Dict[str, float]:
    grouped: Dict[str, List[float]] = defaultdict(list)
    for (task, _), score in scores.items():
        grouped[task].append(float(score))
    return {task: sum(values) / len(values) for task, values in sorted(grouped.items())}


def macro(per_task: Mapping[str, float]) -> float:
    return sum(per_task.values()) / max(1, len(per_task))


def coverage_not_opposed(row: Mapping[str, Any]) -> bool:
    """Coverage may support or be neutral; high delete risk is treated as opposition."""
    if boolish(row.get("coverage_gate_triggered")):
        return True
    reason = str(row.get("coverage_gate_reason") or "")
    return reason in {"coverage_gap_below_threshold", ""}


def safety_state(
    key: Key,
    historical: Mapping[str, Mapping[Key, Mapping[str, Any]]],
    proxy: Mapping[Key, Mapping[str, Any]],
    anchor: Mapping[Key, Mapping[str, Any]],
    coverage: Mapping[Key, Mapping[str, Any]],
) -> Dict[str, Any]:
    hist_drm = historical[METHOD_DRM_SWAP][key]
    proxy_row = proxy[key]
    anchor_row = anchor[key]
    coverage_row = coverage[key]
    removed = list_value(proxy_row.get("candidate_removed_block_ids")) or list_value(hist_drm.get("removed_block_ids"))
    rescued = list_value(proxy_row.get("candidate_rescued_block_ids")) or list_value(hist_drm.get("rescued_block_ids"))
    method_fallback_reason = (
        proxy_row.get("method_fallback_reason")
        or hist_drm.get("fallback_reason")
        or proxy_row.get("fallback_used_reason")
        or ""
    )
    method_fallback_used = boolish(proxy_row.get("fallback_used")) or boolish(hist_drm.get("fallback_used"))
    protected_violation = (
        boolish(anchor_row.get("protected_violation"))
        or boolish(coverage_row.get("protected_violation"))
        or boolish(proxy_row.get("protected_block_violation"))
        or boolish(hist_drm.get("protected_block_violation"))
    )
    over_budget = boolish(hist_drm.get("over_budget")) or boolish(proxy_row.get("over_budget"))
    budget_preserved = (
        boolish(anchor_row.get("budget_preserved"))
        and boolish(coverage_row.get("budget_preserved"))
        and boolish(hist_drm.get("budget_equal_to_sliminfer"))
        and not over_budget
    )
    candidate_exists = bool(removed) and bool(rescued)
    safety_pass = (
        budget_preserved
        and not protected_violation
        and not over_budget
        and not method_fallback_used
        and is_missing(method_fallback_reason)
        and candidate_exists
    )
    return {
        "budget_preserved": budget_preserved,
        "protected_violation": protected_violation,
        "over_budget": over_budget,
        "method_fallback_used": method_fallback_used,
        "method_fallback_reason": "" if is_missing(method_fallback_reason) else method_fallback_reason,
        "candidate_exists": candidate_exists,
        "safety_pass": safety_pass,
        "removed_block_ids": removed,
        "rescued_block_ids": rescued,
    }


def explain_reject(
    *,
    anchor_triggered: bool,
    coverage_triggered: bool,
    coverage_neutral_or_support: bool,
    delete_risk: Any,
    residual_benefit: Any,
    safety: Mapping[str, Any],
    require_coverage: bool,
    require_residual: bool,
) -> str:
    if not anchor_triggered:
        return "anchor_not_triggered"
    if require_coverage and not coverage_triggered:
        return "coverage_not_triggered"
    if not require_coverage and not coverage_neutral_or_support:
        return "coverage_opposes_intervention"
    if is_missing(delete_risk):
        return "delete_risk_missing"
    if float(delete_risk) >= DELETE_RISK_MAX:
        return "delete_risk_at_or_above_threshold"
    if require_residual:
        if is_missing(residual_benefit):
            return "residual_benefit_missing"
        if float(residual_benefit) < RESIDUAL_BENEFIT_MIN:
            return "residual_benefit_extreme_negative"
    if not safety.get("candidate_exists"):
        return "missing_removed_or_rescued_candidates"
    if not safety.get("budget_preserved"):
        return "budget_not_preserved"
    if safety.get("protected_violation"):
        return "protected_violation"
    if safety.get("over_budget"):
        return "over_budget"
    if safety.get("method_fallback_used") or not is_missing(safety.get("method_fallback_reason")):
        return "method_fallback"
    return "risk_aware_fallback_pass"


def decide_policy(
    policy: str,
    key: Key,
    historical: Mapping[str, Mapping[Key, Mapping[str, Any]]],
    proxy: Mapping[Key, Mapping[str, Any]],
    anchor: Mapping[Key, Mapping[str, Any]],
    coverage: Mapping[Key, Mapping[str, Any]],
) -> Dict[str, Any]:
    proxy_row = proxy[key]
    anchor_row = anchor[key]
    coverage_row = coverage[key]
    safety = safety_state(key, historical, proxy, anchor, coverage)
    anchor_triggered = boolish(anchor_row.get("anchor_gate_triggered"))
    coverage_triggered = boolish(coverage_row.get("coverage_gate_triggered"))
    coverage_neutral_or_support = coverage_not_opposed(coverage_row)
    delete_risk = proxy_row.get("delete_risk_proxy_max")
    residual_benefit = proxy_row.get("residual_benefit_proxy")

    if policy == "policy1_anchor_only_conservative":
        require_coverage = False
        require_residual = False
        accept = (
            anchor_triggered
            and coverage_neutral_or_support
            and not is_missing(delete_risk)
            and float(delete_risk) < DELETE_RISK_MAX
            and safety["safety_pass"]
        )
    elif policy == "policy2_anchor_coverage_agreement":
        require_coverage = True
        require_residual = False
        accept = anchor_triggered and coverage_triggered and safety["safety_pass"]
    elif policy == "policy3_anchor_residual_safety":
        require_coverage = False
        require_residual = True
        accept = (
            anchor_triggered
            and coverage_neutral_or_support
            and not is_missing(delete_risk)
            and float(delete_risk) < DELETE_RISK_MAX
            and not is_missing(residual_benefit)
            and float(residual_benefit) >= RESIDUAL_BENEFIT_MIN
            and safety["safety_pass"]
        )
    else:
        raise RuntimeError(f"unknown policy: {policy}")

    reason = (
        "risk_aware_fallback_pass"
        if accept
        else explain_reject(
            anchor_triggered=anchor_triggered,
            coverage_triggered=coverage_triggered,
            coverage_neutral_or_support=coverage_neutral_or_support,
            delete_risk=delete_risk,
            residual_benefit=residual_benefit,
            safety=safety,
            require_coverage=require_coverage,
            require_residual=require_residual,
        )
    )
    return {
        "accept": accept,
        "reason": reason,
        "anchor_triggered": anchor_triggered,
        "coverage_triggered": coverage_triggered,
        "coverage_not_opposed": coverage_neutral_or_support,
        "delete_risk_proxy_max": delete_risk,
        "rescue_gain_proxy_mean": proxy_row.get("rescue_gain_proxy_mean"),
        "residual_benefit_proxy": residual_benefit,
        "rescue_score_margin": proxy_row.get("rescue_score_margin"),
        "anchor_coverage_before": anchor_row.get("anchor_coverage_before"),
        "anchor_coverage_after": anchor_row.get("anchor_coverage_after"),
        "missing_anchor_count": len(list_value(anchor_row.get("missing_anchor_block_ids"))),
        "anchor_gate_reason": anchor_row.get("anchor_gate_reason"),
        "coverage_gap_score": coverage_row.get("coverage_gap_score"),
        "missing_coverage_types": coverage_row.get("missing_coverage_types"),
        "coverage_gate_reason": coverage_row.get("coverage_gate_reason"),
        "budget_preserved": safety["budget_preserved"],
        "protected_violation": safety["protected_violation"],
        "over_budget": safety["over_budget"],
        "method_fallback_reason": safety["method_fallback_reason"],
        "candidate_exists": safety["candidate_exists"],
        "removed_block_ids": safety["removed_block_ids"],
        "rescued_block_ids": safety["rescued_block_ids"],
        "rescue_score_mean": historical[METHOD_DRM_SWAP][key].get("rescue_score_mean"),
        "removed_score_mean": historical[METHOD_DRM_SWAP][key].get("removed_score_mean"),
        "active_redundancy_mean": historical[METHOD_DRM_SWAP][key].get("active_redundancy_mean"),
    }


def baseline_task_and_macro(
    keys: Sequence[Key],
    historical: Mapping[str, Mapping[Key, Mapping[str, Any]]],
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
    task_by_method = {
        method: task_scores({key: score_of(historical[method][key]) for key in keys})
        for method in BASELINE_METHODS
    }
    macro_by_method = {method: macro(per_task) for method, per_task in task_by_method.items()}
    return task_by_method, macro_by_method


def replay_policy(
    policy: str,
    keys: Sequence[Key],
    historical: Mapping[str, Mapping[Key, Mapping[str, Any]]],
    proxy: Mapping[Key, Mapping[str, Any]],
    anchor: Mapping[Key, Mapping[str, Any]],
    coverage: Mapping[Key, Mapping[str, Any]],
) -> Dict[str, Any]:
    direction1_set = {key for key in keys if boolish(anchor[key].get("anchor_gate_triggered"))}
    direction2_set = {key for key in keys if boolish(coverage[key].get("coverage_gate_triggered"))}
    decisions: List[Dict[str, Any]] = []
    replay_scores: Dict[Key, float] = {}
    accepted_set = set()
    for key in keys:
        decision = decide_policy(policy, key, historical, proxy, anchor, coverage)
        accept = bool(decision["accept"])
        if accept:
            accepted_set.add(key)
        selected_method = METHOD_DRM_SWAP if accept else METHOD_ORIGINAL
        selected_score = score_of(historical[selected_method][key])
        replay_scores[key] = selected_score
        original_score = score_of(historical[METHOD_ORIGINAL][key])
        decisions.append({
            "policy": policy,
            "task": key[0],
            "sample_index": key[1],
            "sample_id": sample_id_for(key, historical, proxy),
            "accept_drm_swap_upper_bound": accept,
            "selected_proxy_method": selected_method,
            "decision_reason": decision["reason"],
            "selected_score": selected_score,
            "original_score": original_score,
            "selected_minus_original_score": selected_score - original_score,
            "drm_minus_random_score": score_of(historical[METHOD_DRM_SWAP][key]) - score_of(historical[METHOD_RANDOM_SWAP][key]),
            "proxy_upper_bound": True,
            **{field: value for field, value in decision.items() if field not in {"accept", "reason"}},
        })

    per_task = task_scores(replay_scores)
    replay_macro = macro(per_task)
    baseline_task_scores, baseline_macros = baseline_task_and_macro(keys, historical)
    hotpot_total = sum(1 for key in keys if key[0] == "hotpotqa" and key[1] in HOTPOT_TOP_NEGATIVE)
    hotpot_filtered = sum(
        1 for key in keys
        if key[0] == "hotpotqa" and key[1] in HOTPOT_TOP_NEGATIVE and key not in accepted_set
    )
    musique_total = sum(1 for key in keys if key[0] == "musique" and key[1] in MUSIQUE_TOP_POSITIVE)
    musique_retained = sum(
        1 for key in keys
        if key[0] == "musique" and key[1] in MUSIQUE_TOP_POSITIVE and key in accepted_set
    )
    pnz = Counter()
    for key, selected_score in replay_scores.items():
        delta = selected_score - score_of(historical[METHOD_ORIGINAL][key])
        if delta > 1e-9:
            pnz["positive"] += 1
        elif delta < -1e-9:
            pnz["negative"] += 1
        else:
            pnz["zero"] += 1

    accepted_only_direction1 = bool(accepted_set) and accepted_set.issubset(direction1_set)
    accepted_only_direction2 = bool(accepted_set) and accepted_set.issubset(direction2_set)
    accepted_equal_direction1 = accepted_set == direction1_set
    accepted_equal_direction2 = accepted_set == direction2_set

    return {
        "policy": policy,
        "task_scores": per_task,
        "macro": replay_macro,
        "baseline_task_scores": baseline_task_scores,
        "baseline_macros": baseline_macros,
        "delta_vs": {method: replay_macro - value for method, value in baseline_macros.items()},
        "trigger_count": len(accepted_set),
        "trigger_rate": len(accepted_set) / max(1, len(keys)),
        "accepted_sample_list": [f"{task}:{idx}" for task, idx in sorted(accepted_set)],
        "direction1_trigger_count": len(direction1_set),
        "direction2_trigger_count": len(direction2_set),
        "overlap_with_direction1": len(accepted_set & direction1_set),
        "overlap_with_direction2": len(accepted_set & direction2_set),
        "accepted_subset_of_direction1": accepted_only_direction1,
        "accepted_subset_of_direction2": accepted_only_direction2,
        "accepted_equal_direction1": accepted_equal_direction1,
        "accepted_equal_direction2": accepted_equal_direction2,
        "unique_triggers_not_in_direction1": [f"{task}:{idx}" for task, idx in sorted(accepted_set - direction1_set)],
        "unique_triggers_not_in_direction2": [f"{task}:{idx}" for task, idx in sorted(accepted_set - direction2_set)],
        "hotpotqa_negative_filtered": hotpot_filtered,
        "hotpotqa_negative_total": hotpot_total,
        "hotpotqa_negative_filter_rate": hotpot_filtered / max(1, hotpot_total),
        "musique_positive_retained": musique_retained,
        "musique_positive_total": musique_total,
        "musique_positive_retention_rate": musique_retained / max(1, musique_total),
        "positive_negative_zero_vs_original": dict(pnz),
        "decisions": decisions,
    }


def feature_coverage(
    keys: Sequence[Key],
    historical: Mapping[str, Mapping[Key, Mapping[str, Any]]],
    proxy: Mapping[Key, Mapping[str, Any]],
    anchor: Mapping[Key, Mapping[str, Any]],
    coverage: Mapping[Key, Mapping[str, Any]],
) -> Dict[str, int]:
    extractors: Dict[str, Callable[[Key], Any]] = {
        "residual.delete_risk_proxy_max": lambda key: proxy[key].get("delete_risk_proxy_max"),
        "residual.rescue_gain_proxy_mean": lambda key: proxy[key].get("rescue_gain_proxy_mean"),
        "residual.residual_benefit_proxy": lambda key: proxy[key].get("residual_benefit_proxy"),
        "residual.rescue_score_margin": lambda key: proxy[key].get("rescue_score_margin"),
        "anchor.anchor_gate_triggered": lambda key: anchor[key].get("anchor_gate_triggered"),
        "anchor.anchor_coverage_before": lambda key: anchor[key].get("anchor_coverage_before"),
        "anchor.anchor_coverage_after": lambda key: anchor[key].get("anchor_coverage_after"),
        "anchor.missing_anchor_count": lambda key: len(list_value(anchor[key].get("missing_anchor_block_ids"))),
        "anchor.anchor_gate_reason": lambda key: anchor[key].get("anchor_gate_reason"),
        "coverage.coverage_gap_score": lambda key: coverage[key].get("coverage_gap_score"),
        "coverage.coverage_gate_triggered": lambda key: coverage[key].get("coverage_gate_triggered"),
        "coverage.missing_coverage_types": lambda key: coverage[key].get("missing_coverage_types"),
        "safety.budget_preserved": lambda key: safety_state(key, historical, proxy, anchor, coverage).get("budget_preserved"),
        "safety.protected_violation": lambda key: safety_state(key, historical, proxy, anchor, coverage).get("protected_violation"),
        "safety.fallback_reason": lambda key: safety_state(key, historical, proxy, anchor, coverage).get("method_fallback_reason"),
        "safety.over_budget": lambda key: safety_state(key, historical, proxy, anchor, coverage).get("over_budget"),
        "audit.removed_block_ids": lambda key: historical[METHOD_DRM_SWAP][key].get("removed_block_ids"),
        "audit.rescued_block_ids": lambda key: historical[METHOD_DRM_SWAP][key].get("rescued_block_ids"),
        "audit.rescue_score_mean": lambda key: historical[METHOD_DRM_SWAP][key].get("rescue_score_mean"),
        "audit.removed_score_mean": lambda key: historical[METHOD_DRM_SWAP][key].get("removed_score_mean"),
        "audit.active_redundancy_mean": lambda key: historical[METHOD_DRM_SWAP][key].get("active_redundancy_mean"),
    }
    counts: Dict[str, int] = {}
    for name, getter in extractors.items():
        covered = 0
        for key in keys:
            try:
                value = getter(key)
            except KeyError:
                value = None
            if name.endswith("fallback_reason"):
                # Empty fallback reason is a valid observed "no fallback" value.
                covered += 1
            elif not is_missing(value):
                covered += 1
        counts[name] = covered
    return counts


def historical_200_status(
    historical: Mapping[str, Mapping[Key, Mapping[str, Any]]],
    proxy: Mapping[Key, Mapping[str, Any]],
    anchor: Mapping[Key, Mapping[str, Any]],
    coverage: Mapping[Key, Mapping[str, Any]],
) -> Dict[str, Any]:
    hist_keys = set(historical[METHOD_DRM_SWAP])
    external_keys = set(proxy) & set(anchor) & set(coverage)
    residual_fields = [
        "delete_risk_proxy_max",
        "rescue_gain_proxy_mean",
        "residual_benefit_proxy",
        "rescue_score_margin",
    ]
    hist_residual_coverage = {
        field: sum(
            1
            for key in hist_keys
            if not is_missing(historical[METHOD_DRM_SWAP][key].get(field))
        )
        for field in residual_fields
    }
    return {
        "historical_drm_swap_rows": len(hist_keys),
        "external_proxy_anchor_coverage_intersection_rows": len(external_keys),
        "historical_residual_field_coverage": hist_residual_coverage,
        "offline_replay_performed_on_200": False,
        "reason": (
            "Direction1/Direction2/residual proxy artifacts cover only the fixed 20-sample set; "
            "the old 200-sample historical artifact does not contain the new residual or "
            "controller fields, so no 200-sample fallback replay was performed."
        ),
    }


def verdict_for_policy(policy_result: Mapping[str, Any], direction1_macro: float | None) -> Dict[str, Any]:
    reasons: List[str] = []
    if policy_result["trigger_count"] == 0:
        reasons.append("policy_all_reject")
    if policy_result["accepted_subset_of_direction1"]:
        reasons.append("triggers_subset_of_direction1")
    if policy_result["accepted_equal_direction1"]:
        reasons.append("exactly_reproduces_direction1")
    if policy_result["musique_positive_retention_rate"] < 0.4:
        reasons.append("musique_positive_retention_below_direction1_or_too_low")
    if direction1_macro is not None and policy_result["macro"] < direction1_macro - 1e-9:
        reasons.append("macro_below_direction1")
    if not policy_result["unique_triggers_not_in_direction1"]:
        reasons.append("no_new_triggers_outside_direction1")
    if policy_result["hotpotqa_negative_filter_rate"] < 0.8:
        reasons.append("hotpotqa_negative_filter_not_high")
    if policy_result["trigger_count"] == len(policy_result["decisions"]):
        reasons.append("policy_all_accept")
    return {
        "decision": "GO" if not reasons else "NO-GO",
        "reasons": reasons,
    }


def overall_verdict(policy_results: Mapping[str, Mapping[str, Any]], direction1_macro: float | None) -> Dict[str, Any]:
    go_policies = [name for name, result in policy_results.items() if result["verdict"]["decision"] == "GO"]
    reasons: List[str] = []
    if not go_policies:
        reasons.append("no_policy_met_go_criteria")
    if all(result["accepted_subset_of_direction1"] for result in policy_results.values() if result["trigger_count"] > 0):
        reasons.append("all_nonzero_policies_are_direction1_subsets")
    if direction1_macro is not None and all(result["macro"] < direction1_macro - 1e-9 for result in policy_results.values()):
        reasons.append("all_policy_macros_below_direction1")
    if all(not result["unique_triggers_not_in_direction1"] for result in policy_results.values()):
        reasons.append("no_policy_has_new_trigger_outside_direction1")
    return {
        "decision": "GO" if go_policies else "NO-GO",
        "go_policies": go_policies,
        "reasons": reasons,
    }


def load_direction_replay_macro(path: Path) -> float | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    result = payload.get("result", payload)
    value = result.get("macro")
    return None if is_missing(value) else float(value)


def write_report(
    output_dir: Path,
    args: argparse.Namespace,
    coverage_counts: Mapping[str, int],
    policy_results: Mapping[str, Mapping[str, Any]],
    status_200: Mapping[str, Any],
    direction1_macro: float | None,
    direction2_macro: float | None,
    overall: Mapping[str, Any],
) -> None:
    lines = [
        "# Risk-Aware Fallback Controller Offline Replay",
        "",
        f"- Result directory: `{output_dir.resolve()}`",
        "- Model/GPU used: **No**",
        "- New prediction generated: **No**",
        "- SlimInfer original selector modified: **No**",
        "- Policy feature uses gold/final prediction/final score: **No**",
        "- Accept action: historical Budget-neutral DRM-Swap prediction as **upper-bound proxy**",
        "- Reject action: historical SlimInfer Original prediction",
        "",
        "## Input artifacts",
        "",
        f"- Historical 200-sample diagnostic: `{args.historical_dir}`",
        f"- Residual proxy tiny audit: `{args.proxy_dir}`",
        f"- Direction1 Protected Evidence Anchor audit: `{args.anchor_dir}`",
        f"- Direction2 Evidence Coverage audit: `{args.coverage_dir}`",
        "",
        "## Fixed thresholds",
        "",
        f"- `delete_risk_proxy_max < {DELETE_RISK_MAX}`",
        f"- `residual_benefit_proxy >= {RESIDUAL_BENEFIT_MIN}` for Policy 3",
        "",
        "## Feature coverage on fixed 20-sample set",
        "",
        "| Feature | coverage |",
        "|---|---:|",
    ]
    for name, count in coverage_counts.items():
        lines.append(f"| `{name}` | {count}/20 |")
    lines.extend([
        "",
        "## 200-sample status",
        "",
        f"- Historical DRM-Swap rows: {status_200['historical_drm_swap_rows']}",
        f"- External proxy/anchor/coverage intersection rows: {status_200['external_proxy_anchor_coverage_intersection_rows']}",
        f"- 200-sample replay performed: **{status_200['offline_replay_performed_on_200']}**",
        f"- Reason: {status_200['reason']}",
        "",
        "## Policy replay summary",
        "",
        "| Policy | triggers | accepted | macro | Δ Original | Δ Random-Swap | Δ DRM-Swap | Δ Larger | Hotpot neg filtered | MuSiQue pos retained | Direction1 overlap | Direction2 overlap | Verdict |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ])
    for name, result in policy_results.items():
        accepted = ", ".join(result["accepted_sample_list"]) if result["accepted_sample_list"] else "-"
        lines.append(
            "| {name} | {trig}/20 | {accepted} | {macro:.4f} | {d0:+.4f} | {dr:+.4f} | {dd:+.4f} | {dl:+.4f} | {hf}/{ht} | {mr}/{mt} | {o1}/{trig} | {o2}/{trig} | {verdict} |".format(
                name=name,
                trig=result["trigger_count"],
                accepted=accepted,
                macro=result["macro"],
                d0=result["delta_vs"][METHOD_ORIGINAL],
                dr=result["delta_vs"][METHOD_RANDOM_SWAP],
                dd=result["delta_vs"][METHOD_DRM_SWAP],
                dl=result["delta_vs"][METHOD_LARGER],
                hf=result["hotpotqa_negative_filtered"],
                ht=result["hotpotqa_negative_total"],
                mr=result["musique_positive_retained"],
                mt=result["musique_positive_total"],
                o1=result["overlap_with_direction1"],
                o2=result["overlap_with_direction2"],
                verdict=result["verdict"]["decision"],
            )
        )
    lines.extend([
        "",
        "## Baseline/reference replay macros",
        "",
        f"- Direction1 Protected Evidence Anchor proxy macro: {direction1_macro if direction1_macro is not None else 'unavailable'}",
        f"- Direction2 Evidence Coverage proxy macro: {direction2_macro if direction2_macro is not None else 'unavailable'}",
        "",
        "## GO / NO-GO",
        "",
        f"- Overall decision: **{overall['decision']}**",
    ])
    for reason in overall.get("reasons", []):
        lines.append(f"- Reason: `{reason}`")
    lines.extend([
        "",
        "This is an offline upper-bound proxy replay, not a true generation result.",
    ])
    (output_dir / "risk_aware_fallback_replay_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Path:
    historical_dir = Path(args.historical_dir)
    proxy_dir = Path(args.proxy_dir)
    anchor_dir = Path(args.anchor_dir)
    coverage_dir = Path(args.coverage_dir)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path("results") / f"risk_aware_fallback_replay_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=False)

    historical = load_historical(historical_dir)
    proxy = load_proxy(proxy_dir)
    anchor = load_anchor(anchor_dir)
    coverage = load_coverage(coverage_dir)
    keys = tiny_keys()
    missing_by_source = {
        "historical_original": [key for key in keys if key not in historical[METHOD_ORIGINAL]],
        "historical_drm_swap": [key for key in keys if key not in historical[METHOD_DRM_SWAP]],
        "proxy": [key for key in keys if key not in proxy],
        "anchor": [key for key in keys if key not in anchor],
        "coverage": [key for key in keys if key not in coverage],
    }
    missing_any = {name: value for name, value in missing_by_source.items() if value}
    if missing_any:
        raise RuntimeError(f"missing fixed tiny samples: {missing_any}")

    coverage_counts = feature_coverage(keys, historical, proxy, anchor, coverage)
    policies = [
        "policy1_anchor_only_conservative",
        "policy2_anchor_coverage_agreement",
        "policy3_anchor_residual_safety",
    ]
    policy_results: Dict[str, Dict[str, Any]] = {}
    all_decisions: List[Dict[str, Any]] = []
    for policy in policies:
        result = replay_policy(policy, keys, historical, proxy, anchor, coverage)
        policy_results[policy] = result
        all_decisions.extend(result["decisions"])

    direction1_macro = load_direction_replay_macro(
        anchor_dir / "offline_replay" / "protected_evidence_anchor_replay_results.json"
    )
    direction2_macro = load_direction_replay_macro(
        coverage_dir / "offline_replay" / "evidence_coverage_replay_results.json"
    )
    for result in policy_results.values():
        result["verdict"] = verdict_for_policy(result, direction1_macro)
    overall = overall_verdict(policy_results, direction1_macro)
    status_200 = historical_200_status(historical, proxy, anchor, coverage)

    score_rows = []
    for name, result in policy_results.items():
        score_rows.append({
            "policy": name,
            "hotpotqa": result["task_scores"].get("hotpotqa"),
            "musique": result["task_scores"].get("musique"),
            "macro": result["macro"],
            "delta_vs_original": result["delta_vs"][METHOD_ORIGINAL],
            "delta_vs_random_swap": result["delta_vs"][METHOD_RANDOM_SWAP],
            "delta_vs_drm_swap": result["delta_vs"][METHOD_DRM_SWAP],
            "delta_vs_larger": result["delta_vs"][METHOD_LARGER],
            "trigger_count": result["trigger_count"],
            "hotpotqa_negative_filtered": result["hotpotqa_negative_filtered"],
            "musique_positive_retained": result["musique_positive_retained"],
            "verdict": result["verdict"]["decision"],
        })
    decision_fields = [
        "policy",
        "task",
        "sample_index",
        "sample_id",
        "accept_drm_swap_upper_bound",
        "selected_proxy_method",
        "decision_reason",
        "selected_score",
        "original_score",
        "selected_minus_original_score",
        "drm_minus_random_score",
        "anchor_triggered",
        "coverage_triggered",
        "coverage_not_opposed",
        "delete_risk_proxy_max",
        "rescue_gain_proxy_mean",
        "residual_benefit_proxy",
        "rescue_score_margin",
        "anchor_coverage_before",
        "anchor_coverage_after",
        "missing_anchor_count",
        "anchor_gate_reason",
        "coverage_gap_score",
        "missing_coverage_types",
        "coverage_gate_reason",
        "budget_preserved",
        "protected_violation",
        "over_budget",
        "method_fallback_reason",
        "candidate_exists",
        "removed_block_ids",
        "rescued_block_ids",
        "rescue_score_mean",
        "removed_score_mean",
        "active_redundancy_mean",
        "proxy_upper_bound",
    ]
    write_csv(output_dir / "risk_aware_fallback_decisions.csv", all_decisions, decision_fields)
    write_csv(
        output_dir / "risk_aware_fallback_scores.csv",
        score_rows,
        [
            "policy",
            "hotpotqa",
            "musique",
            "macro",
            "delta_vs_original",
            "delta_vs_random_swap",
            "delta_vs_drm_swap",
            "delta_vs_larger",
            "trigger_count",
            "hotpotqa_negative_filtered",
            "musique_positive_retained",
            "verdict",
        ],
    )
    payload = {
        "input_artifacts": {
            "historical_dir": str(historical_dir),
            "proxy_dir": str(proxy_dir),
            "anchor_dir": str(anchor_dir),
            "coverage_dir": str(coverage_dir),
        },
        "output_dir": str(output_dir),
        "model_or_gpu_used": False,
        "generation_performed": False,
        "gold_answer_used_for_policy": False,
        "final_prediction_or_score_used_for_policy": False,
        "upper_bound_proxy_replay": True,
        "thresholds": {
            "delete_risk_proxy_max": DELETE_RISK_MAX,
            "residual_benefit_min": RESIDUAL_BENEFIT_MIN,
        },
        "sample_indices_by_task": TINY_SAMPLE_INDICES,
        "feature_coverage": coverage_counts,
        "policy_results": {
            name: {key: value for key, value in result.items() if key != "decisions"}
            for name, result in policy_results.items()
        },
        "historical_200_status": status_200,
        "direction1_proxy_macro": direction1_macro,
        "direction2_proxy_macro": direction2_macro,
        "overall_verdict": overall,
    }
    write_json(output_dir / "risk_aware_fallback_replay_results.json", payload)
    write_report(output_dir, args, coverage_counts, policy_results, status_200, direction1_macro, direction2_macro, overall)
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical_dir", default=str(DEFAULT_HISTORICAL_DIR))
    parser.add_argument("--proxy_dir", default=str(DEFAULT_PROXY_DIR))
    parser.add_argument("--anchor_dir", default=str(DEFAULT_ANCHOR_DIR))
    parser.add_argument("--coverage_dir", default=str(DEFAULT_COVERAGE_DIR))
    parser.add_argument("--output_dir", default="")
    return parser.parse_args()


def main() -> None:
    output_dir = run(parse_args())
    print(output_dir)


if __name__ == "__main__":
    main()
