#!/usr/bin/env python3
"""Offline replay for Residual-Aware DRM-Swap gate candidates.

This script never runs a model. It replays a deterministic gate over existing
Original/DRM-Swap predictions:

* gate accept -> use the existing DRM-Swap prediction/score
* gate reject -> use the existing SlimInfer Original prediction/score

Gate features are read only from audit rows. Gold answers, final predictions,
and final scores are used only after the gate decision for reporting.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


METHOD_ORIGINAL = "sliminfer_original"
METHOD_RANDOM_SWAP = "random_budget_neutral_swap"
METHOD_DRM_SWAP = "budget_neutral_drm_swap"
METHOD_LARGER = "sliminfer_larger"

TASKS = ["hotpotqa", "2wikimqa", "musique"]

RESIDUAL_PROXY_FIELDS = [
    "removed_value_norm_mean",
    "removed_value_norm_max",
    "rescued_value_norm_mean",
    "rescued_value_norm_max",
    "removed_qk_or_delta_score_mean",
    "removed_qk_or_delta_score_max",
    "rescued_qk_or_delta_score_mean",
    "rescued_qk_or_delta_score_max",
    "delete_risk_proxy_mean",
    "delete_risk_proxy_max",
    "rescue_gain_proxy_mean",
    "rescue_gain_proxy_max",
    "residual_benefit_proxy",
    "rescue_score_top1",
    "rescue_score_top2",
    "rescue_score_margin",
    "candidate_score_std",
    "candidate_score_iqr",
    "candidate_rescued_block_ids",
    "candidate_removed_block_ids",
    "removed_block_text_preview",
    "rescued_block_text_preview",
    "gate_reason",
    "gate_triggered",
]


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
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


def finite_float(value: Any) -> Optional[float]:
    if value is None or value == "" or value == "unavailable":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def score_of(row: Mapping[str, Any]) -> float:
    value = finite_float(row.get("prediction_score"))
    if value is None:
        raise RuntimeError(f"missing prediction_score for {row.get('method_key')}/{row.get('task')}/{row.get('sample_id')}")
    return value


def macro(scores_by_task: Mapping[str, float]) -> float:
    return sum(float(scores_by_task[task]) for task in sorted(scores_by_task)) / max(1, len(scores_by_task))


def load_audit(input_dir: Path) -> Dict[str, Dict[Tuple[str, str], Dict[str, Any]]]:
    path = input_dir / "audit" / "per_sample_audit.jsonl"
    if not path.exists():
        raise RuntimeError(f"missing per-sample audit: {path}")
    by_method: Dict[str, Dict[Tuple[str, str], Dict[str, Any]]] = defaultdict(dict)
    for row in read_jsonl(path):
        method = str(row.get("method_key", ""))
        task = str(row.get("task", ""))
        sid = str(row.get("sample_id", ""))
        if method and task and sid:
            by_method[method][(task, sid)] = row
    required = [METHOD_ORIGINAL, METHOD_RANDOM_SWAP, METHOD_DRM_SWAP, METHOD_LARGER]
    missing = [method for method in required if method not in by_method]
    if missing:
        raise RuntimeError(f"missing required methods in audit: {missing}")
    return by_method


def load_proxy_audit(proxy_audit_dir: Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
    candidates = [
        proxy_audit_dir / "audit" / "gate_feature_audit.jsonl",
        proxy_audit_dir / "gate_feature_audit.jsonl",
        proxy_audit_dir / "audit" / "per_sample_audit.jsonl",
    ]
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        raise RuntimeError(
            f"missing proxy audit rows under {proxy_audit_dir}; looked for "
            + ", ".join(str(candidate) for candidate in candidates)
        )
    rows: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in read_jsonl(path):
        task = str(row.get("task", ""))
        sid = str(row.get("sample_id", ""))
        if not task or not sid:
            continue
        rows[(task, sid)] = row
    if not rows:
        raise RuntimeError(f"proxy audit file is empty: {path}")
    return rows


def overlay_proxy_features(
    by_method: Dict[str, Dict[Tuple[str, str], Dict[str, Any]]],
    proxy_rows: Mapping[Tuple[str, str], Mapping[str, Any]],
) -> List[Tuple[str, str]]:
    missing: List[Tuple[str, str]] = []
    for key, proxy in proxy_rows.items():
        target = by_method[METHOD_DRM_SWAP].get(key)
        if target is None:
            missing.append(key)
            continue
        merged = dict(target)
        for field in RESIDUAL_PROXY_FIELDS + [
            "fallback_used",
            "fallback_reason",
            "protected_block_violation",
            "over_budget",
            "selected_block_count",
            "K",
            "used_blocks",
            "used_tokens",
            "pruned_candidate_count",
            "swap_count",
            "rescued_block_ids",
            "removed_block_ids",
            "rescue_score_mean",
            "removed_score_mean",
            "active_redundancy_mean",
        ]:
            if field in proxy:
                merged[field] = proxy.get(field)
        merged["proxy_audit_sample_index"] = proxy.get("sample_index")
        merged["proxy_audit_source"] = "gate_feature_audit"
        by_method[METHOD_DRM_SWAP][key] = merged
    if missing:
        preview = ", ".join(f"{task}:{sid}" for task, sid in missing[:5])
        raise RuntimeError(
            f"proxy audit samples absent from historical DRM-Swap audit: "
            f"{len(missing)} missing; first={preview}"
        )
    return sorted(
        proxy_rows,
        key=lambda key: (
            key[0],
            int(proxy_rows[key].get("sample_index", -1)),
            key[1],
        ),
    )


def aligned_keys(by_method: Mapping[str, Mapping[Tuple[str, str], Mapping[str, Any]]]) -> List[Tuple[str, str]]:
    key_sets = [set(by_method[method]) for method in [METHOD_ORIGINAL, METHOD_RANDOM_SWAP, METHOD_DRM_SWAP, METHOD_LARGER]]
    common = set.intersection(*key_sets)
    all_keys = set.union(*key_sets)
    if common != all_keys:
        raise RuntimeError(f"sample_id mismatch across required methods: common={len(common)} union={len(all_keys)}")
    return sorted(common, key=lambda key: (key[0], int(by_method[METHOD_ORIGINAL][key].get("sample_index", -1)), key[1]))


def field_available(value: Any) -> bool:
    if value is None or value == "" or value == "unavailable":
        return False
    if isinstance(value, float) and not math.isfinite(value):
        return False
    if isinstance(value, (list, dict)) and len(value) == 0:
        return False
    return True


def audit_field_coverage(drm_rows: Iterable[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    rows = list(drm_rows)
    coverage: Dict[str, Dict[str, Any]] = {}
    for field in RESIDUAL_PROXY_FIELDS:
        available = sum(1 for row in rows if field_available(row.get(field)))
        coverage[field] = {
            "available": available,
            "total": len(rows),
            "coverage": available / max(1, len(rows)),
        }
    return coverage


def gate_a(row: Mapping[str, Any], tau_delete: float) -> Tuple[bool, str]:
    delete_risk = finite_float(row.get("delete_risk_proxy_max"))
    if delete_risk is None:
        return False, "missing_delete_risk_proxy_max"
    return delete_risk < tau_delete, (
        "delete_risk_below_threshold" if delete_risk < tau_delete else "delete_risk_at_or_above_threshold"
    )


def gate_b(row: Mapping[str, Any], tau_margin: float, tau_gain: float) -> Tuple[bool, str]:
    margin = finite_float(row.get("rescue_score_margin"))
    gain = finite_float(row.get("rescue_gain_proxy_mean"))
    missing = []
    if margin is None:
        missing.append("rescue_score_margin")
    if gain is None:
        missing.append("rescue_gain_proxy_mean")
    if missing:
        return False, "missing_" + ",".join(missing)
    accept = margin > tau_margin and gain > tau_gain
    return accept, "margin_and_gain_pass" if accept else "margin_or_gain_below_threshold"


def gate_c(row: Mapping[str, Any], tau_delete: float, tau_margin: float, tau_benefit: float) -> Tuple[bool, str]:
    if bool(row.get("fallback_used", False)):
        return False, "fallback_used"
    if bool(row.get("protected_block_violation", False)):
        return False, "protected_block_violation"
    delete_risk = finite_float(row.get("delete_risk_proxy_max"))
    margin = finite_float(row.get("rescue_score_margin"))
    benefit = finite_float(row.get("residual_benefit_proxy"))
    missing = []
    if delete_risk is None:
        missing.append("delete_risk_proxy_max")
    if margin is None:
        missing.append("rescue_score_margin")
    if benefit is None:
        missing.append("residual_benefit_proxy")
    if missing:
        return False, "missing_" + ",".join(missing)
    accept = delete_risk < tau_delete and margin > tau_margin and benefit > tau_benefit
    return accept, "residual_benefit_gate_pass" if accept else "residual_benefit_gate_reject"


def choose_representative_keys(
    keys: Sequence[Tuple[str, str]],
    by_method: Mapping[str, Mapping[Tuple[str, str], Mapping[str, Any]]],
) -> List[Tuple[str, str]]:
    selected: List[Tuple[str, str]] = []
    wanted = {
        "hotpotqa": {0, 56, 117, 41, 57},
        "musique": {20, 50, 54, 134, 56},
    }
    for key in keys:
        task, _ = key
        index = int(by_method[METHOD_DRM_SWAP][key].get("sample_index", -1))
        if index in wanted.get(task, set()):
            selected.append(key)
    for task in ["hotpotqa", "musique"]:
        zero_added = 0
        for key in keys:
            if key[0] != task:
                continue
            drm = score_of(by_method[METHOD_DRM_SWAP][key])
            random_swap = score_of(by_method[METHOD_RANDOM_SWAP][key])
            if abs(drm - random_swap) <= 1e-9 and key not in selected:
                selected.append(key)
                zero_added += 1
                if zero_added >= 5:
                    break
    return sorted(selected, key=lambda key: (key[0], int(by_method[METHOD_ORIGINAL][key].get("sample_index", -1)), key[1]))


def task_scores(method_or_gate_rows: Mapping[Tuple[str, str], float]) -> Dict[str, float]:
    grouped: Dict[str, List[float]] = defaultdict(list)
    for (task, _), score in method_or_gate_rows.items():
        grouped[task].append(float(score))
    return {task: sum(values) / len(values) for task, values in sorted(grouped.items()) if values}


def concentration(rows: Mapping[Tuple[str, str], float], original_rows: Mapping[Tuple[str, str], Mapping[str, Any]], task: str = "musique") -> Dict[str, Any]:
    deltas = []
    for key, score in rows.items():
        if key[0] != task:
            continue
        deltas.append(float(score) - score_of(original_rows[key]))
    positives = sorted([delta for delta in deltas if delta > 1e-9], reverse=True)
    negatives = [delta for delta in deltas if delta < -1e-9]
    net = sum(deltas)
    payload: Dict[str, Any] = {
        "net_delta_sum": net,
        "mean_delta": net / max(1, len(deltas)),
        "positive_count": len(positives),
        "negative_count": len(negatives),
        "zero_count": len(deltas) - len(positives) - len(negatives),
    }
    for topn in [1, 3, 5]:
        top_sum = sum(positives[:topn])
        payload[f"top{topn}_positive_share_of_net"] = (
            top_sum / net if abs(net) > 1e-12 else None
        )
    return payload


def replay_gate(
    gate_name: str,
    keys: Sequence[Tuple[str, str]],
    by_method: Mapping[str, Mapping[Tuple[str, str], Mapping[str, Any]]],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    gate_scores: Dict[Tuple[str, str], float] = {}
    decisions: List[Dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    for key in keys:
        drm_row = by_method[METHOD_DRM_SWAP][key]
        if gate_name == "gate_a_removal_risk":
            accept, reason = gate_a(drm_row, args.tau_delete)
        elif gate_name == "gate_b_rescue_margin":
            accept, reason = gate_b(drm_row, args.tau_margin, args.tau_gain)
        elif gate_name == "gate_c_residual_benefit":
            accept, reason = gate_c(drm_row, args.tau_delete, args.tau_margin, args.tau_benefit)
        else:
            raise RuntimeError(f"unknown gate: {gate_name}")
        selected_method = METHOD_DRM_SWAP if accept else METHOD_ORIGINAL
        selected_row = by_method[selected_method][key]
        gate_scores[key] = score_of(selected_row)
        reason_counts[reason] += 1
        decisions.append({
            "gate": gate_name,
            "task": key[0],
            "sample_id": key[1],
            "sample_index": int(drm_row.get("sample_index", -1)),
            "accept": bool(accept),
            "selected_method": selected_method,
            "reason": reason,
            "delete_risk_proxy_max": drm_row.get("delete_risk_proxy_max"),
            "rescue_score_margin": drm_row.get("rescue_score_margin"),
            "rescue_gain_proxy_mean": drm_row.get("rescue_gain_proxy_mean"),
            "residual_benefit_proxy": drm_row.get("residual_benefit_proxy"),
            "fallback_used": bool(drm_row.get("fallback_used", False)),
            "protected_block_violation": bool(drm_row.get("protected_block_violation", False)),
        })
    per_task = task_scores(gate_scores)
    gate_macro = macro(per_task)
    baseline_scores = {
        method: task_scores({key: score_of(by_method[method][key]) for key in keys})
        for method in [METHOD_ORIGINAL, METHOD_RANDOM_SWAP, METHOD_DRM_SWAP, METHOD_LARGER]
    }
    baseline_macros = {method: macro(scores) for method, scores in baseline_scores.items()}
    delta_vs = {method: gate_macro - value for method, value in baseline_macros.items()}
    pos_neg_zero = {}
    for method in [METHOD_ORIGINAL, METHOD_RANDOM_SWAP, METHOD_DRM_SWAP, METHOD_LARGER]:
        counts = Counter()
        for key, score in gate_scores.items():
            delta = float(score) - score_of(by_method[method][key])
            if delta > 1e-9:
                counts["positive"] += 1
            elif delta < -1e-9:
                counts["negative"] += 1
            else:
                counts["zero"] += 1
        pos_neg_zero[method] = dict(counts)

    hotpot_neg_total = 0
    hotpot_neg_rejected = 0
    musique_pos_total = 0
    musique_pos_accepted = 0
    for decision in decisions:
        key = (decision["task"], decision["sample_id"])
        drm_vs_random = score_of(by_method[METHOD_DRM_SWAP][key]) - score_of(by_method[METHOD_RANDOM_SWAP][key])
        if decision["task"] == "hotpotqa" and drm_vs_random < -1e-9:
            hotpot_neg_total += 1
            hotpot_neg_rejected += int(not decision["accept"])
        if decision["task"] == "musique" and drm_vs_random > 1e-9:
            musique_pos_total += 1
            musique_pos_accepted += int(decision["accept"])
    hotpot_score = per_task.get("hotpotqa")
    random_hotpot = baseline_scores[METHOD_RANDOM_SWAP].get("hotpotqa")
    return {
        "gate": gate_name,
        "n": len(keys),
        "trigger_count": sum(1 for decision in decisions if decision["accept"]),
        "trigger_rate": sum(1 for decision in decisions if decision["accept"]) / max(1, len(decisions)),
        "reason_counts": dict(reason_counts),
        "task_scores": per_task,
        "macro": gate_macro,
        "baseline_macros": baseline_macros,
        "delta_vs": delta_vs,
        "positive_negative_zero_vs": pos_neg_zero,
        "hotpotqa_negative_swaps_total": hotpot_neg_total,
        "hotpotqa_negative_swaps_filtered": hotpot_neg_rejected,
        "hotpotqa_negative_swaps_filter_rate": hotpot_neg_rejected / max(1, hotpot_neg_total),
        "musique_positive_swaps_total": musique_pos_total,
        "musique_positive_swaps_retained": musique_pos_accepted,
        "musique_positive_swaps_retention_rate": musique_pos_accepted / max(1, musique_pos_total),
        "musique_concentration_vs_original": concentration(gate_scores, by_method[METHOD_ORIGINAL], "musique"),
        "hotpotqa_minus_random_swap": (
            float(hotpot_score) - float(random_hotpot)
            if hotpot_score is not None and random_hotpot is not None
            else None
        ),
        "decisions": decisions,
    }


def go_no_go(full_results: Mapping[str, Mapping[str, Any]], coverage: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    unavailable_proxy = [
        field for field in [
            "delete_risk_proxy_max",
            "rescue_gain_proxy_mean",
            "residual_benefit_proxy",
            "rescue_score_margin",
        ]
        if coverage.get(field, {}).get("available", 0) == 0
    ]
    gate_c = full_results["gate_c_residual_benefit"]
    conc = gate_c["musique_concentration_vs_original"]
    top3_share = conc.get("top3_positive_share_of_net")
    reasons: List[str] = []
    if unavailable_proxy:
        reasons.append("proxy_fields_unavailable:" + ",".join(unavailable_proxy))
    if not (
        gate_c["macro"] > gate_c["baseline_macros"][METHOD_ORIGINAL]
        and gate_c["macro"] > gate_c["baseline_macros"][METHOD_RANDOM_SWAP]
        and gate_c["macro"] > gate_c["baseline_macros"][METHOD_LARGER]
    ):
        reasons.append("gated_macro_not_above_original_random_and_larger")
    if gate_c["hotpotqa_minus_random_swap"] is not None and gate_c["hotpotqa_minus_random_swap"] < -0.5:
        reasons.append("hotpotqa_still_below_random_swap")
    if top3_share is not None and top3_share > 0.7:
        reasons.append("musique_gain_still_top3_concentrated")
    if gate_c["trigger_count"] == 0:
        reasons.append("gate_only_reduces_to_original_no_drm_accepts")
    return {
        "decision": "GO" if not reasons else "NO-GO",
        "reasons": reasons,
        "recommend_tiny_smoke": False if reasons else True,
        "recommend_stop_drm_swap_direction": bool(reasons and "proxy_fields_unavailable:" in reasons[0]),
    }


def format_float(value: Any) -> str:
    numeric = finite_float(value)
    return "unavailable" if numeric is None else f"{numeric:.4f}"


def write_report(
    output_dir: Path,
    input_dir: Path,
    proxy_audit_dir: Optional[Path],
    coverage: Mapping[str, Mapping[str, Any]],
    full_results: Mapping[str, Mapping[str, Any]],
    representative_results: Mapping[str, Mapping[str, Any]],
    decision: Mapping[str, Any],
    args: argparse.Namespace,
) -> None:
    unavailable = [field for field, row in coverage.items() if int(row["available"]) == 0]
    lines: List[str] = []
    lines.append("# Residual-Aware DRM-Swap v0 Offline Gate Replay\n")
    lines.append(f"- Result directory: `{output_dir}`")
    lines.append(f"- Input result directory: `{input_dir}`")
    if proxy_audit_dir is not None:
        lines.append(f"- Proxy audit directory: `{proxy_audit_dir}`")
    lines.append("- Modified generation or old results: **No**")
    lines.append("- Model/GPU used: **No**")
    lines.append("- Replay semantics: gate accept uses existing DRM-Swap prediction; gate reject uses existing Original prediction.")
    lines.append("- Gate features use only audit/proxy fields; gold/prediction/score are used only after gate decisions for reporting.\n")
    lines.append("## Thresholds\n")
    lines.append(f"- Gate A `tau_delete`: {args.tau_delete}")
    lines.append(f"- Gate B `tau_margin`: {args.tau_margin}; `tau_gain`: {args.tau_gain}")
    lines.append(f"- Gate C `tau_delete`: {args.tau_delete}; `tau_margin`: {args.tau_margin}; `tau_benefit`: {args.tau_benefit}\n")
    lines.append("## Audit field coverage on existing DRM-Swap rows\n")
    for field, row in coverage.items():
        lines.append(f"- `{field}`: {row['available']}/{row['total']} ({row['coverage']:.1%})")
    lines.append("")
    if unavailable:
        lines.append("Unavailable fields in this existing 200-sample result:")
        for field in unavailable:
            lines.append(f"- `{field}`")
        lines.append("")
    lines.append("## Full offline replay results\n")
    for gate, result in full_results.items():
        lines.append(f"### {gate}\n")
        lines.append(f"- N: {result['n']}")
        lines.append(f"- Trigger rate: {result['trigger_count']}/{result['n']} ({result['trigger_rate']:.2%})")
        lines.append(f"- Task scores: {json.dumps({k: round(v, 4) for k, v in result['task_scores'].items()}, ensure_ascii=False)}")
        lines.append(f"- Macro: {result['macro']:.4f}")
        lines.append(f"- Delta vs Original: {result['delta_vs'][METHOD_ORIGINAL]:+.4f}")
        lines.append(f"- Delta vs Random-Swap: {result['delta_vs'][METHOD_RANDOM_SWAP]:+.4f}")
        lines.append(f"- Delta vs Larger: {result['delta_vs'][METHOD_LARGER]:+.4f}")
        lines.append(f"- Delta vs DRM-Swap: {result['delta_vs'][METHOD_DRM_SWAP]:+.4f}")
        lines.append(f"- HotpotQA negative swap filter rate: {result['hotpotqa_negative_swaps_filtered']}/{result['hotpotqa_negative_swaps_total']} ({result['hotpotqa_negative_swaps_filter_rate']:.2%})")
        lines.append(f"- MuSiQue positive swap retention rate: {result['musique_positive_swaps_retained']}/{result['musique_positive_swaps_total']} ({result['musique_positive_swaps_retention_rate']:.2%})")
        lines.append(f"- HotpotQA minus Random-Swap: {format_float(result['hotpotqa_minus_random_swap'])}")
        lines.append(f"- Positive/negative/zero vs Original: {json.dumps(result['positive_negative_zero_vs'][METHOD_ORIGINAL], ensure_ascii=False)}")
        lines.append(f"- MuSiQue concentration vs Original: {json.dumps(result['musique_concentration_vs_original'], ensure_ascii=False)}")
        lines.append(f"- Reject/accept reasons: {json.dumps(result['reason_counts'], ensure_ascii=False)}\n")
    lines.append("## Representative subset replay\n")
    for gate, result in representative_results.items():
        lines.append(f"- `{gate}`: N={result['n']}, triggers={result['trigger_count']}, macro={result['macro']:.4f}, reasons={json.dumps(result['reason_counts'], ensure_ascii=False)}")
    lines.append("")
    lines.append("## GO / NO-GO\n")
    lines.append(f"- Decision: **{decision['decision']}**")
    for reason in decision.get("reasons", []):
        lines.append(f"- Reason: {reason}")
    lines.append(f"- Recommend tiny smoke: {bool(decision.get('recommend_tiny_smoke'))}")
    lines.append(f"- Recommend stop DRM-Swap direction now: {bool(decision.get('recommend_stop_drm_swap_direction'))}")
    lines.append("")
    lines.append("## Negative diagnostic wording if NO-GO\n")
    lines.append(
        "Residual-Aware DRM-Swap v0 was implemented as audit-only proxy instrumentation and replayed offline on the 200-sample multi-hop diagnostic. "
        "The existing run did not contain the new value/residual proxy fields, so all conservative gates failed closed to Original. "
        "This is a negative diagnostic for the current artifacts, not evidence against future runs with the new proxy instrumentation populated."
    )
    (output_dir / "residual_aware_gate_replay_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument(
        "--proxy_audit_dir",
        type=Path,
        default=None,
        help="optional tiny gate/proxy audit whose fields are overlaid onto historical DRM-Swap rows",
    )
    parser.add_argument("--tau_delete", type=float, default=0.20)
    parser.add_argument("--tau_margin", type=float, default=0.25)
    parser.add_argument("--tau_gain", type=float, default=0.0)
    parser.add_argument("--tau_benefit", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    if not input_dir.exists():
        raise RuntimeError(f"input result directory does not exist: {input_dir}")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else Path("results") / f"residual_aware_drm_swap_offline_replay_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    by_method = load_audit(input_dir)
    keys = aligned_keys(by_method)
    proxy_audit_dir = args.proxy_audit_dir.resolve() if args.proxy_audit_dir else None
    if proxy_audit_dir is not None:
        proxy_rows = load_proxy_audit(proxy_audit_dir)
        proxy_keys = overlay_proxy_features(by_method, proxy_rows)
        key_set = set(keys)
        keys = [key for key in proxy_keys if key in key_set]
        if len(keys) != len(proxy_rows):
            raise RuntimeError(
                f"proxy/historical aligned keys={len(keys)} proxy rows={len(proxy_rows)}"
            )
    drm_rows = [by_method[METHOD_DRM_SWAP][key] for key in keys]
    coverage = audit_field_coverage(drm_rows)
    representative_keys = keys if proxy_audit_dir is not None else choose_representative_keys(keys, by_method)

    gates = [
        "gate_a_removal_risk",
        "gate_b_rescue_margin",
        "gate_c_residual_benefit",
    ]
    full_results = {gate: replay_gate(gate, keys, by_method, args) for gate in gates}
    representative_results = {
        gate: replay_gate(gate, representative_keys, by_method, args) for gate in gates
    }
    decision = go_no_go(full_results, coverage)

    summary_rows = []
    for scope, results in [("full", full_results), ("representative", representative_results)]:
        for gate, result in results.items():
            row = {
                "scope": scope,
                "gate": gate,
                "n": result["n"],
                "trigger_count": result["trigger_count"],
                "trigger_rate": result["trigger_rate"],
                "macro": result["macro"],
                "delta_vs_original": result["delta_vs"][METHOD_ORIGINAL],
                "delta_vs_random_swap": result["delta_vs"][METHOD_RANDOM_SWAP],
                "delta_vs_larger": result["delta_vs"][METHOD_LARGER],
                "delta_vs_drm_swap": result["delta_vs"][METHOD_DRM_SWAP],
                "hotpotqa": result["task_scores"].get("hotpotqa"),
                "2wikimqa": result["task_scores"].get("2wikimqa"),
                "musique": result["task_scores"].get("musique"),
                "hotpotqa_negative_filter_rate": result["hotpotqa_negative_swaps_filter_rate"],
                "musique_positive_retention_rate": result["musique_positive_swaps_retention_rate"],
                "hotpotqa_minus_random_swap": result["hotpotqa_minus_random_swap"],
            }
            summary_rows.append(row)
    write_csv(
        output_dir / "gate_replay_summary.csv",
        summary_rows,
        [
            "scope",
            "gate",
            "n",
            "trigger_count",
            "trigger_rate",
            "macro",
            "delta_vs_original",
            "delta_vs_random_swap",
            "delta_vs_larger",
            "delta_vs_drm_swap",
            "hotpotqa",
            "2wikimqa",
            "musique",
            "hotpotqa_negative_filter_rate",
            "musique_positive_retention_rate",
            "hotpotqa_minus_random_swap",
        ],
    )
    for gate, result in full_results.items():
        write_csv(
            output_dir / f"{gate}_decisions_full.csv",
            result["decisions"],
            [
                "gate",
                "task",
                "sample_id",
                "sample_index",
                "accept",
                "selected_method",
                "reason",
                "delete_risk_proxy_max",
                "rescue_score_margin",
                "rescue_gain_proxy_mean",
                "residual_benefit_proxy",
                "fallback_used",
                "protected_block_violation",
            ],
        )
    write_json(
        output_dir / "gate_replay_results.json",
        {
            "input_dir": str(input_dir),
            "proxy_audit_dir": str(proxy_audit_dir) if proxy_audit_dir is not None else None,
            "output_dir": str(output_dir.resolve()),
            "thresholds": {
                "tau_delete": args.tau_delete,
                "tau_margin": args.tau_margin,
                "tau_gain": args.tau_gain,
                "tau_benefit": args.tau_benefit,
            },
            "field_coverage": coverage,
            "full_results": full_results,
            "representative_results": representative_results,
            "decision": decision,
            "representative_sample_count": len(representative_keys),
            "generation_modified": False,
            "old_results_modified": False,
            "model_or_gpu_used": False,
        },
    )
    write_report(output_dir, input_dir, proxy_audit_dir, coverage, full_results, representative_results, decision, args)
    print(json.dumps({
        "status": "ok",
        "output_dir": str(output_dir.resolve()),
        "decision": decision,
        "summary_csv": str((output_dir / "gate_replay_summary.csv").resolve()),
        "report": str((output_dir / "residual_aware_gate_replay_report.md").resolve()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
