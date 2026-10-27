#!/usr/bin/env python3
"""Offline replay for UASR gate decisions.

No model is run.  If UASR accepts, this uses the historical Budget-neutral
DRM-Swap prediction as an upper-bound proxy because no true UASR generation
exists.  If UASR rejects, it uses SlimInfer Original.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple


DEFAULT_HISTORICAL_DIR = Path("results/drm_budget_neutral_swap_multihop200_20260706_042717")
DEFAULT_ANCHOR_DIR = Path("results/protected_evidence_anchor_audit_20260706_214256")
METHOD_ORIGINAL = "sliminfer_original"
METHOD_RANDOM_SWAP = "random_budget_neutral_swap"
METHOD_DRM_SWAP = "budget_neutral_drm_swap"
METHOD_LARGER = "sliminfer_larger"
BASELINE_METHODS = [METHOD_ORIGINAL, METHOD_RANDOM_SWAP, METHOD_DRM_SWAP, METHOD_LARGER]
HOTPOT_TOP_NEGATIVE = {0, 56, 117, 41, 57}
MUSIQUE_TOP_POSITIVE = {20, 50, 54, 134, 56}


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


def score_of(row: Mapping[str, Any]) -> float:
    return finite_float(row.get("prediction_score"), field="prediction_score")


def task_scores(scores: Mapping[Key, float]) -> Dict[str, float]:
    grouped: Dict[str, List[float]] = defaultdict(list)
    for (task, _), score in scores.items():
        grouped[task].append(float(score))
    return {task: sum(values) / len(values) for task, values in sorted(grouped.items())}


def macro(per_task: Mapping[str, float]) -> float:
    return sum(per_task.values()) / max(1, len(per_task))


def load_audit(audit_dir: Path) -> Dict[Key, JsonRow]:
    path = audit_dir / "uasr_uncertainty_audit.jsonl"
    return {key_of(row): row for row in read_jsonl(path)}


def load_anchor_triggers(anchor_dir: Path | None) -> set[Key]:
    if anchor_dir is None:
        return set()
    path = anchor_dir / "protected_evidence_anchor_audit.jsonl"
    if not path.exists():
        return set()
    triggers: set[Key] = set()
    for row in read_jsonl(path):
        if boolish(row.get("anchor_gate_triggered")):
            triggers.add(key_of(row))
    return triggers


def replay(
    audit_rows: Mapping[Key, Mapping[str, Any]],
    historical: Mapping[str, Mapping[Key, Mapping[str, Any]]],
    anchor_triggers: set[Key],
) -> Dict[str, Any]:
    keys = sorted(audit_rows, key=lambda k: (k[0], k[1]))
    decisions: List[JsonRow] = []
    replay_scores: Dict[Key, float] = {}
    accepted = set()
    for key in keys:
        for method in BASELINE_METHODS:
            if key not in historical[method]:
                raise RuntimeError(f"missing historical {method}: {key}")
        audit = audit_rows[key]
        accept = boolish(audit.get("uasr_gate_triggered"))
        if accept:
            accepted.add(key)
        selected_method = METHOD_DRM_SWAP if accept else METHOD_ORIGINAL
        selected_score = score_of(historical[selected_method][key])
        replay_scores[key] = selected_score
        decisions.append({
            "task": key[0],
            "sample_index": key[1],
            "sample_id": audit.get("sample_id", historical[METHOD_ORIGINAL][key].get("sample_id")),
            "uasr_gate_triggered": accept,
            "selected_proxy_method": selected_method,
            "uasr_gate_reason": audit.get("uasr_gate_reason"),
            "cutoff_margin": audit.get("cutoff_margin"),
            "score_cutoff_fields_available": audit.get("score_cutoff_fields_available"),
            "selected_score": selected_score,
            "original_score": score_of(historical[METHOD_ORIGINAL][key]),
            "drm_swap_score": score_of(historical[METHOD_DRM_SWAP][key]),
            "random_swap_score": score_of(historical[METHOD_RANDOM_SWAP][key]),
            "proxy_upper_bound": True,
        })
    per_task = task_scores(replay_scores)
    replay_macro = macro(per_task)
    baseline_task_scores = {
        method: task_scores({key: score_of(historical[method][key]) for key in keys})
        for method in BASELINE_METHODS
    }
    baseline_macros = {method: macro(scores) for method, scores in baseline_task_scores.items()}
    pnz = Counter()
    for key, score in replay_scores.items():
        delta = score - score_of(historical[METHOD_ORIGINAL][key])
        if delta > 1e-9:
            pnz["positive"] += 1
        elif delta < -1e-9:
            pnz["negative"] += 1
        else:
            pnz["zero"] += 1
    hotpot_total = sum(1 for key in keys if key[0] == "hotpotqa" and key[1] in HOTPOT_TOP_NEGATIVE)
    hotpot_filtered = sum(
        1 for key in keys
        if key[0] == "hotpotqa" and key[1] in HOTPOT_TOP_NEGATIVE and key not in accepted
    )
    musique_total = sum(1 for key in keys if key[0] == "musique" and key[1] in MUSIQUE_TOP_POSITIVE)
    musique_retained = sum(
        1 for key in keys
        if key[0] == "musique" and key[1] in MUSIQUE_TOP_POSITIVE and key in accepted
    )
    score_field_available = sum(boolish(row.get("score_cutoff_fields_available")) for row in audit_rows.values())
    trigger_count = len(accepted)
    anchor_overlap = accepted.intersection(anchor_triggers)
    new_vs_anchor = accepted.difference(anchor_triggers)
    accepted_subset_of_anchor = bool(accepted) and accepted.issubset(anchor_triggers)
    accepted_equal_anchor_on_tiny = bool(accepted) and accepted == anchor_triggers.intersection(set(keys))
    reasons: List[str] = []
    if score_field_available == 0:
        reasons.append("score_cutoff_fields_unavailable")
    if trigger_count == 0:
        reasons.append("uasr_gate_all_reject")
    if replay_macro < baseline_macros[METHOD_ORIGINAL] - 1e-9:
        reasons.append("replay_macro_below_original")
    if trigger_count == 0:
        reasons.append("cannot_compare_to_protected_anchor_no_signal")
    if hotpot_total and hotpot_filtered < math.ceil(hotpot_total / 2):
        reasons.append("hotpotqa_top_negative_not_majority_filtered")
    if musique_total and musique_retained == 0:
        reasons.append("musique_top_positive_not_retained")
    if accepted_subset_of_anchor:
        reasons.append("uasr_trigger_subset_of_protected_anchor")
    verdict = "GO" if not reasons else "NO-GO"
    return {
        "task_scores": per_task,
        "macro": replay_macro,
        "baseline_task_scores": baseline_task_scores,
        "baseline_macros": baseline_macros,
        "delta_vs": {method: replay_macro - value for method, value in baseline_macros.items()},
        "trigger_count": trigger_count,
        "trigger_rate": trigger_count / max(1, len(keys)),
        "accepted_sample_list": [f"{task}:{idx}" for task, idx in sorted(accepted)],
        "score_cutoff_fields_available_count": score_field_available,
        "hotpotqa_negative_filtered": hotpot_filtered,
        "hotpotqa_negative_total": hotpot_total,
        "hotpotqa_negative_filter_rate": hotpot_filtered / max(1, hotpot_total),
        "musique_positive_retained": musique_retained,
        "musique_positive_total": musique_total,
        "musique_positive_retention_rate": musique_retained / max(1, musique_total),
        "positive_negative_zero_vs_original": dict(pnz),
        "anchor_trigger_overlap_count": len(anchor_overlap),
        "anchor_trigger_count_in_scope": len(anchor_triggers.intersection(set(keys))),
        "accepted_subset_of_anchor": accepted_subset_of_anchor,
        "accepted_equal_anchor_on_tiny": accepted_equal_anchor_on_tiny,
        "new_trigger_vs_anchor_list": [f"{task}:{idx}" for task, idx in sorted(new_vs_anchor)],
        "decisions": decisions,
        "go_no_go": {
            "decision": verdict,
            "reasons": reasons,
        },
    }


def write_report(output_dir: Path, args: argparse.Namespace, result: Mapping[str, Any]) -> None:
    lines = [
        "# UASR Offline Gate Replay",
        "",
        f"- Result directory: `{output_dir.resolve()}`",
        f"- Audit input: `{args.audit_dir}`",
        f"- Historical prediction input: `{args.historical_dir}`",
        "- Model/GPU used: **No**",
        "- New prediction generated: **No**",
        "- Accept action: historical DRM-Swap prediction as **upper-bound proxy**",
        "- Reject action: historical SlimInfer Original prediction",
        "- Policy uses gold/final score/final prediction: **No**",
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
        f"- Score cutoff fields available: {result['score_cutoff_fields_available_count']}/{len(result['decisions'])}",
        f"- Trigger rate: {result['trigger_count']}/{len(result['decisions'])} ({result['trigger_rate']:.2%})",
        f"- Accepted samples: `{json.dumps(result['accepted_sample_list'], ensure_ascii=False)}`",
        f"- HotpotQA top negative filtered: {result['hotpotqa_negative_filtered']}/{result['hotpotqa_negative_total']} ({result['hotpotqa_negative_filter_rate']:.2%})",
        f"- MuSiQue top positive retained: {result['musique_positive_retained']}/{result['musique_positive_total']} ({result['musique_positive_retention_rate']:.2%})",
        f"- Positive/negative/zero vs Original: `{json.dumps(result['positive_negative_zero_vs_original'], ensure_ascii=False)}`",
        f"- Protected Anchor trigger overlap: {result['anchor_trigger_overlap_count']}/{result['trigger_count']}",
        f"- UASR trigger subset of Protected Anchor: {result['accepted_subset_of_anchor']}",
        f"- New UASR triggers vs Protected Anchor: `{json.dumps(result['new_trigger_vs_anchor_list'], ensure_ascii=False)}`",
        "",
        "## GO / NO-GO",
        "",
        f"- Decision: **{result['go_no_go']['decision']}**",
    ]
    for reason in result["go_no_go"]["reasons"]:
        lines.append(f"- Reason: `{reason}`")
    lines.append("")
    lines.append("This is an upper-bound proxy replay, not a true UASR generation result.")
    (output_dir / "uasr_gate_replay_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Path:
    audit_dir = Path(args.audit_dir)
    output_dir = Path(args.output_dir) if args.output_dir else audit_dir / "offline_replay"
    output_dir.mkdir(parents=True, exist_ok=False)
    historical = load_historical(Path(args.historical_dir))
    audit_rows = load_audit(audit_dir)
    anchor_triggers = load_anchor_triggers(Path(args.anchor_dir) if args.anchor_dir else None)
    result = replay(audit_rows, historical, anchor_triggers)
    payload = {
        "audit_dir": str(audit_dir),
        "historical_dir": str(args.historical_dir),
        "anchor_dir": str(args.anchor_dir or ""),
        "model_or_gpu_used": False,
        "generation_performed": False,
        "upper_bound_proxy_replay": True,
        "gold_answer_used_for_gate": False,
        "result": {k: v for k, v in result.items() if k != "decisions"},
    }
    write_json(output_dir / "uasr_gate_replay_results.json", payload)
    write_csv(
        output_dir / "uasr_gate_replay_decisions.csv",
        result["decisions"],
        [
            "task",
            "sample_index",
            "sample_id",
            "uasr_gate_triggered",
            "selected_proxy_method",
            "uasr_gate_reason",
            "cutoff_margin",
            "score_cutoff_fields_available",
            "selected_score",
            "original_score",
            "drm_swap_score",
            "random_swap_score",
            "proxy_upper_bound",
        ],
    )
    write_csv(
        output_dir / "uasr_gate_replay_scores.csv",
        [{
            "scope": "proxy_replay",
            **result["task_scores"],
            "macro": result["macro"],
            "delta_vs_original": result["delta_vs"][METHOD_ORIGINAL],
            "delta_vs_random_swap": result["delta_vs"][METHOD_RANDOM_SWAP],
            "delta_vs_drm_swap": result["delta_vs"][METHOD_DRM_SWAP],
            "delta_vs_larger": result["delta_vs"][METHOD_LARGER],
            "trigger_count": result["trigger_count"],
            "anchor_trigger_overlap_count": result["anchor_trigger_overlap_count"],
            "accepted_subset_of_anchor": result["accepted_subset_of_anchor"],
            "verdict": result["go_no_go"]["decision"],
        }],
        [
            "scope",
            "hotpotqa",
            "musique",
            "macro",
            "delta_vs_original",
            "delta_vs_random_swap",
            "delta_vs_drm_swap",
            "delta_vs_larger",
            "trigger_count",
            "anchor_trigger_overlap_count",
            "accepted_subset_of_anchor",
            "verdict",
        ],
    )
    write_report(output_dir, args, result)
    print(output_dir)
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit_dir", required=True)
    parser.add_argument("--historical_dir", default=str(DEFAULT_HISTORICAL_DIR))
    parser.add_argument("--anchor_dir", default=str(DEFAULT_ANCHOR_DIR))
    parser.add_argument("--output_dir", default="")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
