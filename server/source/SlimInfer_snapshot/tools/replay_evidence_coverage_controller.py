#!/usr/bin/env python3
"""Offline replay for Evidence Coverage Controller.

No model is run.  Accept uses historical DRM-Swap prediction as an upper-bound
proxy; reject uses historical SlimInfer Original prediction.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple


DEFAULT_HISTORICAL_DIR = Path("results/drm_budget_neutral_swap_multihop200_20260706_042717")
DEFAULT_ANCHOR_DIR = Path("results/protected_evidence_anchor_audit_20260706_214256")
METHOD_ORIGINAL = "sliminfer_original"
METHOD_RANDOM_SWAP = "random_budget_neutral_swap"
METHOD_DRM_SWAP = "budget_neutral_drm_swap"
METHOD_LARGER = "sliminfer_larger"
BASELINES = [METHOD_ORIGINAL, METHOD_RANDOM_SWAP, METHOD_DRM_SWAP, METHOD_LARGER]
HOTPOT_TOP_NEGATIVE = {0, 56, 117, 41, 57}
MUSIQUE_TOP_POSITIVE = {20, 50, 54, 134, 56}


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


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


def finite_float(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"non-finite value: {value!r}")
    return result


def load_coverage_audit(audit_dir: Path) -> Dict[Tuple[str, int], Dict[str, Any]]:
    path = audit_dir / "evidence_coverage_audit.jsonl"
    if not path.exists():
        raise RuntimeError(f"missing evidence coverage audit: {path}")
    return {(str(row["task"]), int(row["sample_index"])): row for row in read_jsonl(path)}


def load_anchor_audit(anchor_dir: Path) -> Dict[Tuple[str, int], Dict[str, Any]]:
    path = anchor_dir / "protected_evidence_anchor_audit.jsonl"
    if not path.exists():
        raise RuntimeError(f"missing protected anchor audit: {path}")
    return {(str(row["task"]), int(row["sample_index"])): row for row in read_jsonl(path)}


def load_historical(historical_dir: Path) -> Dict[str, Dict[Tuple[str, int], Dict[str, Any]]]:
    path = historical_dir / "audit" / "per_sample_audit.jsonl"
    if not path.exists():
        raise RuntimeError(f"missing historical audit: {path}")
    by_method: Dict[str, Dict[Tuple[str, int], Dict[str, Any]]] = defaultdict(dict)
    for row in read_jsonl(path):
        method = str(row.get("method_key", ""))
        if method:
            by_method[method][(str(row["task"]), int(row["sample_index"]))] = row
    missing = [method for method in BASELINES if method not in by_method]
    if missing:
        raise RuntimeError(f"missing baseline methods: {missing}")
    return by_method


def score_of(row: Mapping[str, Any]) -> float:
    return finite_float(row.get("prediction_score"))


def task_scores(scores: Mapping[Tuple[str, int], float]) -> Dict[str, float]:
    grouped: Dict[str, List[float]] = defaultdict(list)
    for (task, _), score in scores.items():
        grouped[task].append(score)
    return {task: sum(values) / len(values) for task, values in sorted(grouped.items())}


def macro(per_task: Mapping[str, float]) -> float:
    return sum(per_task.values()) / max(1, len(per_task))


def replay(
    coverage_rows: Mapping[Tuple[str, int], Mapping[str, Any]],
    anchor_rows: Mapping[Tuple[str, int], Mapping[str, Any]],
    historical: Mapping[str, Mapping[Tuple[str, int], Mapping[str, Any]]],
) -> Dict[str, Any]:
    keys = sorted(coverage_rows, key=lambda key: (key[0], key[1]))
    decisions: List[Dict[str, Any]] = []
    replay_scores: Dict[Tuple[str, int], float] = {}
    anchor_trigger_set = {
        key for key, row in anchor_rows.items() if bool(row.get("anchor_gate_triggered"))
    }
    coverage_trigger_set = {
        key for key, row in coverage_rows.items() if bool(row.get("coverage_gate_triggered"))
    }
    for key in keys:
        for method in BASELINES:
            if key not in historical[method]:
                raise RuntimeError(f"missing historical {method}: {key}")
        row = coverage_rows[key]
        accept = bool(row.get("coverage_gate_triggered"))
        selected_method = METHOD_DRM_SWAP if accept else METHOD_ORIGINAL
        selected_score = score_of(historical[selected_method][key])
        replay_scores[key] = selected_score
        drm_minus_random = score_of(historical[METHOD_DRM_SWAP][key]) - score_of(historical[METHOD_RANDOM_SWAP][key])
        decisions.append({
            "task": key[0],
            "sample_index": key[1],
            "sample_id": row.get("sample_id"),
            "coverage_gate_triggered": accept,
            "selected_proxy_method": selected_method,
            "coverage_gate_reason": row.get("coverage_gate_reason"),
            "coverage_gap_score": row.get("coverage_gap_score"),
            "delete_risk_proxy_max": row.get("delete_risk_proxy_max"),
            "budget_preserved": row.get("budget_preserved"),
            "protected_violation": row.get("protected_violation"),
            "direction1_anchor_triggered": key in anchor_trigger_set,
            "drm_minus_random_score": drm_minus_random,
            "selected_score": selected_score,
            "proxy_upper_bound": True,
        })
    per_task = task_scores(replay_scores)
    replay_macro = macro(per_task)
    baseline_scores = {
        method: task_scores({key: score_of(historical[method][key]) for key in keys})
        for method in BASELINES
    }
    baseline_macros = {method: macro(scores) for method, scores in baseline_scores.items()}
    hotpot_total = sum(1 for key in keys if key[0] == "hotpotqa" and key[1] in HOTPOT_TOP_NEGATIVE)
    hotpot_filtered = sum(
        1
        for key in keys
        if key[0] == "hotpotqa"
        and key[1] in HOTPOT_TOP_NEGATIVE
        and not bool(coverage_rows[key].get("coverage_gate_triggered"))
    )
    musique_total = sum(1 for key in keys if key[0] == "musique" and key[1] in MUSIQUE_TOP_POSITIVE)
    musique_retained = sum(
        1
        for key in keys
        if key[0] == "musique"
        and key[1] in MUSIQUE_TOP_POSITIVE
        and bool(coverage_rows[key].get("coverage_gate_triggered"))
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
    overlap = len(coverage_trigger_set & anchor_trigger_set)
    return {
        "task_scores": per_task,
        "macro": replay_macro,
        "baseline_task_scores": baseline_scores,
        "baseline_macros": baseline_macros,
        "delta_vs": {method: replay_macro - value for method, value in baseline_macros.items()},
        "trigger_count": len(coverage_trigger_set),
        "trigger_rate": len(coverage_trigger_set) / max(1, len(keys)),
        "hotpotqa_negative_filtered": hotpot_filtered,
        "hotpotqa_negative_total": hotpot_total,
        "hotpotqa_negative_filter_rate": hotpot_filtered / max(1, hotpot_total),
        "musique_positive_retained": musique_retained,
        "musique_positive_total": musique_total,
        "musique_positive_retention_rate": musique_retained / max(1, musique_total),
        "positive_negative_zero_vs_original": dict(pnz),
        "direction1_anchor_trigger_count": len(anchor_trigger_set & set(keys)),
        "trigger_overlap_with_direction1": overlap,
        "coverage_trigger_subset_of_direction1": bool(coverage_trigger_set) and coverage_trigger_set.issubset(anchor_trigger_set),
        "unique_coverage_triggers_not_in_direction1": sorted(
            [f"{task}:{idx}" for task, idx in coverage_trigger_set - anchor_trigger_set]
        ),
        "decisions": decisions,
    }


def verdict(result: Mapping[str, Any], coverage_rows: Mapping[Tuple[str, int], Mapping[str, Any]]) -> Dict[str, Any]:
    reasons: List[str] = []
    if result["trigger_count"] == 0:
        reasons.append("gate_all_reject")
    if result["trigger_count"] == len(coverage_rows):
        reasons.append("gate_all_accept")
    if any(not bool(row.get("budget_preserved")) for row in coverage_rows.values()):
        reasons.append("budget_violation")
    if any(bool(row.get("protected_violation")) for row in coverage_rows.values()):
        reasons.append("protected_violation")
    if result["hotpotqa_negative_filter_rate"] < 0.5:
        reasons.append("hotpotqa_negative_not_filtered")
    if result["musique_positive_retention_rate"] <= 0.0:
        reasons.append("musique_positive_not_retained")
    if result["macro"] < result["baseline_macros"][METHOD_ORIGINAL]:
        reasons.append("proxy_macro_below_original")
    if all(not bool(row.get("block_text_available")) for row in coverage_rows.values()):
        reasons.append("true_block_text_unavailable_proxy_only")
    if result.get("coverage_trigger_subset_of_direction1"):
        reasons.append("coverage_triggers_subset_of_direction1_no_unique_signal")
    return {
        "decision": "GO_FOR_TINY_QUALITY_SMOKE" if not reasons else "NO-GO",
        "reasons": reasons,
        "proxy_upper_bound_only": True,
    }


def write_report(output_dir: Path, audit_dir: Path, historical_dir: Path, anchor_dir: Path, result: Mapping[str, Any], decision: Mapping[str, Any]) -> None:
    lines = [
        "# Evidence Coverage Controller Offline Replay",
        "",
        f"- Result directory: `{output_dir.resolve()}`",
        f"- Audit input: `{audit_dir}`",
        f"- Historical prediction input: `{historical_dir}`",
        f"- Direction1 anchor input: `{anchor_dir}`",
        "- Model/GPU used: **No**",
        "- New prediction generated: **No**",
        "- Accept action: historical DRM-Swap prediction as **upper-bound proxy**",
        "- Reject action: historical SlimInfer Original prediction",
        "",
        "## Scores",
        "",
        f"- Task scores: `{json.dumps({k: round(v, 4) for k, v in result['task_scores'].items()}, ensure_ascii=False)}`",
        f"- Macro: {result['macro']:.4f}",
        f"- Delta vs Original: {result['delta_vs'][METHOD_ORIGINAL]:+.4f}",
        f"- Delta vs Random-Swap: {result['delta_vs'][METHOD_RANDOM_SWAP]:+.4f}",
        f"- Delta vs DRM-Swap: {result['delta_vs'][METHOD_DRM_SWAP]:+.4f}",
        f"- Delta vs Larger: {result['delta_vs'][METHOD_LARGER]:+.4f}",
        "",
        "## Gate behavior",
        "",
        f"- Trigger rate: {result['trigger_count']}/20 ({result['trigger_rate']:.2%})",
        f"- HotpotQA top negative filtered: {result['hotpotqa_negative_filtered']}/{result['hotpotqa_negative_total']} ({result['hotpotqa_negative_filter_rate']:.2%})",
        f"- MuSiQue top positive retained: {result['musique_positive_retained']}/{result['musique_positive_total']} ({result['musique_positive_retention_rate']:.2%})",
        f"- Trigger overlap with direction1: {result['trigger_overlap_with_direction1']}/{result['trigger_count']}",
        f"- Unique coverage triggers not in direction1: `{json.dumps(result['unique_coverage_triggers_not_in_direction1'], ensure_ascii=False)}`",
        "",
        "## GO / NO-GO",
        "",
        f"- Decision: **{decision['decision']}**",
    ]
    for reason in decision.get("reasons", []):
        lines.append(f"- Reason: `{reason}`")
    lines.append("")
    lines.append("This is an upper-bound proxy replay, not a true generation result.")
    (output_dir / "evidence_coverage_replay_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit_dir", type=Path, required=True)
    parser.add_argument("--historical_dir", type=Path, default=DEFAULT_HISTORICAL_DIR)
    parser.add_argument("--anchor_dir", type=Path, default=DEFAULT_ANCHOR_DIR)
    parser.add_argument("--output_dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.audit_dir / "offline_replay"
    output_dir.mkdir(parents=True, exist_ok=False)
    coverage_rows = load_coverage_audit(args.audit_dir)
    anchor_rows = load_anchor_audit(args.anchor_dir)
    historical = load_historical(args.historical_dir)
    result = replay(coverage_rows, anchor_rows, historical)
    decision = verdict(result, coverage_rows)
    write_csv(
        output_dir / "evidence_coverage_replay_decisions.csv",
        result["decisions"],
        [
            "task",
            "sample_index",
            "sample_id",
            "coverage_gate_triggered",
            "selected_proxy_method",
            "coverage_gate_reason",
            "coverage_gap_score",
            "delete_risk_proxy_max",
            "budget_preserved",
            "protected_violation",
            "direction1_anchor_triggered",
            "drm_minus_random_score",
            "selected_score",
            "proxy_upper_bound",
        ],
    )
    write_csv(
        output_dir / "evidence_coverage_replay_scores.csv",
        [
            {
                "scope": "proxy_replay",
                "hotpotqa": result["task_scores"].get("hotpotqa"),
                "musique": result["task_scores"].get("musique"),
                "macro": result["macro"],
                "delta_vs_original": result["delta_vs"][METHOD_ORIGINAL],
                "delta_vs_random_swap": result["delta_vs"][METHOD_RANDOM_SWAP],
                "delta_vs_drm_swap": result["delta_vs"][METHOD_DRM_SWAP],
                "delta_vs_larger": result["delta_vs"][METHOD_LARGER],
            }
        ],
        [
            "scope",
            "hotpotqa",
            "musique",
            "macro",
            "delta_vs_original",
            "delta_vs_random_swap",
            "delta_vs_drm_swap",
            "delta_vs_larger",
        ],
    )
    write_json(
        output_dir / "evidence_coverage_replay_results.json",
        {
            "audit_dir": str(args.audit_dir),
            "historical_dir": str(args.historical_dir),
            "anchor_dir": str(args.anchor_dir),
            "result": {k: v for k, v in result.items() if k != "decisions"},
            "decisions": result["decisions"],
            "go_no_go": decision,
            "model_or_gpu_used": False,
            "new_prediction_generated": False,
            "proxy_upper_bound_only": True,
        },
    )
    write_report(output_dir, args.audit_dir, args.historical_dir, args.anchor_dir, result, decision)
    print(json.dumps({
        "status": "ok",
        "output_dir": str(output_dir.resolve()),
        "trigger_count": result["trigger_count"],
        "macro": result["macro"],
        "go_no_go": decision,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
