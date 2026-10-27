#!/usr/bin/env python3
"""Offline replay for Protected Evidence Anchor proposals.

No model is run.  If the anchor controller accepts, the script uses the
historical DRM-Swap prediction as an upper-bound proxy because no real
anchor-controller prediction exists yet.  If it rejects, it uses the historical
SlimInfer Original prediction.
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


DEFAULT_HISTORICAL_DIR = Path(
    "results/drm_budget_neutral_swap_multihop200_20260706_042717"
)
METHOD_ORIGINAL = "sliminfer_original"
METHOD_RANDOM_SWAP = "random_budget_neutral_swap"
METHOD_DRM_SWAP = "budget_neutral_drm_swap"
METHOD_LARGER = "sliminfer_larger"
BASELINE_METHODS = [
    METHOD_ORIGINAL,
    METHOD_RANDOM_SWAP,
    METHOD_DRM_SWAP,
    METHOD_LARGER,
]
HOTPOT_TOP_NEGATIVE = {0, 56, 117, 41, 57}
MUSIQUE_TOP_POSITIVE = {20, 50, 54, 134, 56}


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


def finite_float(value: Any) -> float:
    if value in (None, "", "unavailable"):
        raise RuntimeError(f"missing numeric value: {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"non-finite numeric value: {value!r}")
    return result


def load_anchor_audit(audit_dir: Path) -> Dict[Tuple[str, int], Dict[str, Any]]:
    candidates = [
        audit_dir / "protected_evidence_anchor_audit.jsonl",
        audit_dir / "audit" / "protected_evidence_anchor_audit.jsonl",
    ]
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        raise RuntimeError(f"missing protected anchor audit under {audit_dir}")
    rows: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for row in read_jsonl(path):
        rows[(str(row["task"]), int(row["sample_index"]))] = row
    return rows


def load_historical(historical_dir: Path) -> Dict[str, Dict[Tuple[str, int], Dict[str, Any]]]:
    path = historical_dir / "audit" / "per_sample_audit.jsonl"
    if not path.exists():
        raise RuntimeError(f"missing historical audit: {path}")
    by_method: Dict[str, Dict[Tuple[str, int], Dict[str, Any]]] = defaultdict(dict)
    for row in read_jsonl(path):
        method = str(row.get("method_key", ""))
        if method:
            by_method[method][(str(row["task"]), int(row["sample_index"]))] = row
    missing = [method for method in BASELINE_METHODS if method not in by_method]
    if missing:
        raise RuntimeError(f"historical audit missing methods: {missing}")
    return by_method


def score_of(row: Mapping[str, Any]) -> float:
    return finite_float(row.get("prediction_score"))


def task_scores(scores: Mapping[Tuple[str, int], float]) -> Dict[str, float]:
    grouped: Dict[str, List[float]] = defaultdict(list)
    for (task, _), score in scores.items():
        grouped[task].append(float(score))
    return {task: sum(values) / len(values) for task, values in sorted(grouped.items())}


def macro(per_task: Mapping[str, float]) -> float:
    return sum(per_task.values()) / max(1, len(per_task))


def replay(
    audit_rows: Mapping[Tuple[str, int], Mapping[str, Any]],
    historical: Mapping[str, Mapping[Tuple[str, int], Mapping[str, Any]]],
) -> Dict[str, Any]:
    keys = sorted(audit_rows, key=lambda key: (key[0], key[1]))
    decisions: List[Dict[str, Any]] = []
    replay_scores: Dict[Tuple[str, int], float] = {}
    for key in keys:
        task, index = key
        audit = audit_rows[key]
        for method in BASELINE_METHODS:
            if key not in historical[method]:
                raise RuntimeError(f"missing historical {method} for {task}:{index}")
        accept = bool(audit.get("anchor_gate_triggered"))
        selected_method = METHOD_DRM_SWAP if accept else METHOD_ORIGINAL
        selected_row = historical[selected_method][key]
        replay_scores[key] = score_of(selected_row)
        drm_minus_random = score_of(historical[METHOD_DRM_SWAP][key]) - score_of(historical[METHOD_RANDOM_SWAP][key])
        decisions.append({
            "task": task,
            "sample_index": index,
            "sample_id": audit.get("sample_id", historical[METHOD_ORIGINAL][key].get("sample_id")),
            "anchor_gate_triggered": accept,
            "selected_proxy_method": selected_method,
            "anchor_gate_reason": audit.get("anchor_gate_reason"),
            "anchor_score_mean": audit.get("anchor_score_mean"),
            "removal_risk_max": audit.get("removal_risk_max"),
            "budget_preserved": audit.get("budget_preserved"),
            "protected_violation": audit.get("protected_violation"),
            "drm_minus_random_score": drm_minus_random,
            "selected_score": replay_scores[key],
            "proxy_upper_bound": True,
        })
    per_task = task_scores(replay_scores)
    replay_macro = macro(per_task)
    baseline_scores = {
        method: task_scores({key: score_of(historical[method][key]) for key in keys})
        for method in BASELINE_METHODS
    }
    baseline_macros = {method: macro(scores) for method, scores in baseline_scores.items()}
    hotpot_total = sum(1 for row in decisions if row["task"] == "hotpotqa" and int(row["sample_index"]) in HOTPOT_TOP_NEGATIVE)
    hotpot_filtered = sum(
        1
        for row in decisions
        if row["task"] == "hotpotqa"
        and int(row["sample_index"]) in HOTPOT_TOP_NEGATIVE
        and not bool(row["anchor_gate_triggered"])
    )
    musique_total = sum(1 for row in decisions if row["task"] == "musique" and int(row["sample_index"]) in MUSIQUE_TOP_POSITIVE)
    musique_retained = sum(
        1
        for row in decisions
        if row["task"] == "musique"
        and int(row["sample_index"]) in MUSIQUE_TOP_POSITIVE
        and bool(row["anchor_gate_triggered"])
    )
    counts_vs_original = Counter()
    for key, score in replay_scores.items():
        delta = score - score_of(historical[METHOD_ORIGINAL][key])
        if delta > 1e-9:
            counts_vs_original["positive"] += 1
        elif delta < -1e-9:
            counts_vs_original["negative"] += 1
        else:
            counts_vs_original["zero"] += 1
    return {
        "keys": keys,
        "decisions": decisions,
        "task_scores": per_task,
        "macro": replay_macro,
        "baseline_task_scores": baseline_scores,
        "baseline_macros": baseline_macros,
        "delta_vs": {method: replay_macro - value for method, value in baseline_macros.items()},
        "trigger_count": sum(bool(row["anchor_gate_triggered"]) for row in decisions),
        "trigger_rate": sum(bool(row["anchor_gate_triggered"]) for row in decisions) / max(1, len(decisions)),
        "hotpotqa_negative_filtered": hotpot_filtered,
        "hotpotqa_negative_total": hotpot_total,
        "hotpotqa_negative_filter_rate": hotpot_filtered / max(1, hotpot_total),
        "musique_positive_retained": musique_retained,
        "musique_positive_total": musique_total,
        "musique_positive_retention_rate": musique_retained / max(1, musique_total),
        "positive_negative_zero_vs_original": dict(counts_vs_original),
    }


def decision(result: Mapping[str, Any], audit_rows: Mapping[Tuple[str, int], Mapping[str, Any]]) -> Dict[str, Any]:
    reasons: List[str] = []
    trigger_count = int(result["trigger_count"])
    if trigger_count == 0:
        reasons.append("gate_all_reject")
    if trigger_count == len(audit_rows):
        reasons.append("gate_all_accept")
    if any(not bool(row.get("budget_preserved")) for row in audit_rows.values()):
        reasons.append("budget_violation")
    if any(bool(row.get("protected_violation")) for row in audit_rows.values()):
        reasons.append("protected_violation")
    if result["hotpotqa_negative_filter_rate"] < 0.5:
        reasons.append("hotpotqa_negative_not_filtered")
    if result["musique_positive_retention_rate"] <= 0.0:
        reasons.append("musique_positive_not_retained")
    if result["macro"] < result["baseline_macros"][METHOD_ORIGINAL]:
        reasons.append("proxy_macro_below_original")
    return {
        "decision": "GO_FOR_TINY_QUALITY_SMOKE" if not reasons else "NO-GO",
        "reasons": reasons,
        "proxy_upper_bound_only": True,
    }


def write_report(output_dir: Path, audit_dir: Path, historical_dir: Path, result: Mapping[str, Any], verdict: Mapping[str, Any]) -> None:
    lines = [
        "# Protected Evidence Anchor Offline Replay",
        "",
        f"- Result directory: `{output_dir.resolve()}`",
        f"- Audit input: `{audit_dir}`",
        f"- Historical prediction input: `{historical_dir}`",
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
        f"- Delta vs Larger: {result['delta_vs'][METHOD_LARGER]:+.4f}",
        f"- Delta vs DRM-Swap: {result['delta_vs'][METHOD_DRM_SWAP]:+.4f}",
        "",
        "## Gate behavior",
        "",
        f"- Trigger rate: {result['trigger_count']}/{len(result['decisions'])} ({result['trigger_rate']:.2%})",
        f"- HotpotQA top negative filtered: {result['hotpotqa_negative_filtered']}/{result['hotpotqa_negative_total']} ({result['hotpotqa_negative_filter_rate']:.2%})",
        f"- MuSiQue top positive retained: {result['musique_positive_retained']}/{result['musique_positive_total']} ({result['musique_positive_retention_rate']:.2%})",
        f"- Positive/negative/zero vs Original: `{json.dumps(result['positive_negative_zero_vs_original'], ensure_ascii=False)}`",
        "",
        "## GO / NO-GO",
        "",
        f"- Decision: **{verdict['decision']}**",
    ]
    for reason in verdict.get("reasons", []):
        lines.append(f"- Reason: `{reason}`")
    lines.append("")
    lines.append("This replay is not a real protected-anchor quality run because no anchor-controller predictions exist yet.")
    (output_dir / "protected_evidence_anchor_replay_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit_dir", type=Path, required=True)
    parser.add_argument("--historical_dir", type=Path, default=DEFAULT_HISTORICAL_DIR)
    parser.add_argument("--output_dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or Path("results") / f"protected_evidence_anchor_replay_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=False)
    audit_rows = load_anchor_audit(args.audit_dir)
    historical = load_historical(args.historical_dir)
    result = replay(audit_rows, historical)
    verdict = decision(result, audit_rows)
    write_csv(
        output_dir / "protected_evidence_anchor_replay_decisions.csv",
        result["decisions"],
        [
            "task",
            "sample_index",
            "sample_id",
            "anchor_gate_triggered",
            "selected_proxy_method",
            "anchor_gate_reason",
            "anchor_score_mean",
            "removal_risk_max",
            "budget_preserved",
            "protected_violation",
            "drm_minus_random_score",
            "selected_score",
            "proxy_upper_bound",
        ],
    )
    write_csv(
        output_dir / "protected_evidence_anchor_replay_scores.csv",
        [
            {
                "scope": "proxy_replay",
                "hotpotqa": result["task_scores"].get("hotpotqa"),
                "musique": result["task_scores"].get("musique"),
                "macro": result["macro"],
                "delta_vs_original": result["delta_vs"][METHOD_ORIGINAL],
                "delta_vs_random_swap": result["delta_vs"][METHOD_RANDOM_SWAP],
                "delta_vs_larger": result["delta_vs"][METHOD_LARGER],
                "delta_vs_drm_swap": result["delta_vs"][METHOD_DRM_SWAP],
            }
        ],
        [
            "scope",
            "hotpotqa",
            "musique",
            "macro",
            "delta_vs_original",
            "delta_vs_random_swap",
            "delta_vs_larger",
            "delta_vs_drm_swap",
        ],
    )
    write_json(
        output_dir / "protected_evidence_anchor_replay_results.json",
        {
            "audit_dir": str(args.audit_dir),
            "historical_dir": str(args.historical_dir),
            "result": {
                key: value
                for key, value in result.items()
                if key not in {"keys", "decisions"}
            },
            "decisions": result["decisions"],
            "go_no_go": verdict,
            "model_or_gpu_used": False,
            "new_prediction_generated": False,
            "proxy_upper_bound_only": True,
        },
    )
    write_report(output_dir, args.audit_dir, args.historical_dir, result, verdict)
    print(json.dumps({
        "status": "ok",
        "output_dir": str(output_dir.resolve()),
        "trigger_count": result["trigger_count"],
        "macro": result["macro"],
        "go_no_go": verdict,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
