#!/usr/bin/env python3
"""Audit-only Protected Evidence Anchor controller.

This script does not run a model and does not modify SlimInfer selection.  It
constructs an external, frozen-base controller proposal from existing audit
artifacts:

    A = SlimInfer Original selected K blocks
    E = protected evidence-anchor candidates outside A
    R = low-risk removable blocks inside A
    proposal = A - R + E

The output is an audit/proposal only.  A later offline replay may use existing
predictions as a proxy upper bound; no new quality generation happens here.
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


DEFAULT_HISTORICAL_DIR = Path(
    "results/drm_budget_neutral_swap_multihop200_20260706_042717"
)
DEFAULT_PROXY_DIR = Path("results/residual_aware_proxy_audit_tiny_20260706_204145")
DEFAULT_SAMPLE_INDICES = {
    "hotpotqa": [0, 1, 2, 3, 4, 5, 41, 56, 57, 117],
    "musique": [0, 1, 2, 3, 4, 20, 50, 54, 56, 134],
}
METHOD_ORIGINAL = "sliminfer_original"
METHOD_DRM_SWAP = "budget_neutral_drm_swap"


AUDIT_FIELDS = [
    "sample_id",
    "sample_index",
    "task",
    "original_selected_block_ids",
    "anchor_candidate_block_ids",
    "missing_anchor_block_ids",
    "removable_block_ids",
    "proposed_removed_block_ids",
    "proposed_added_anchor_block_ids",
    "anchor_score_mean",
    "anchor_score_max",
    "removal_risk_mean",
    "removal_risk_max",
    "anchor_coverage_before",
    "anchor_coverage_after",
    "anchor_gate_triggered",
    "anchor_gate_reason",
    "budget_preserved",
    "protected_violation",
    "fallback_reason",
    "proxy_only_no_generation",
    "anchor_feature_source",
    "anchor_feature_unavailable",
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


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def finite_float(value: Any, default: float = 0.0) -> float:
    if value in (None, "", "unavailable"):
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def parse_sample_indices(value: str) -> Dict[str, List[int]]:
    value = str(value or "").strip()
    if not value:
        return {task: list(indices) for task, indices in DEFAULT_SAMPLE_INDICES.items()}
    parsed: Dict[str, List[int]] = {}
    for chunk in value.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise RuntimeError("--sample_indices_by_task must use task:i,j;task:k")
        task, raw_indices = chunk.split(":", 1)
        parsed[task.strip()] = [int(part.strip()) for part in raw_indices.split(",") if part.strip()]
    return parsed


def sample_keys(sample_indices: Mapping[str, Sequence[int]]) -> List[Tuple[str, int]]:
    keys: List[Tuple[str, int]] = []
    for task in sorted(sample_indices):
        keys.extend((task, int(index)) for index in sample_indices[task])
    return keys


def load_proxy_rows(proxy_dir: Path) -> Dict[Tuple[str, int], Dict[str, Any]]:
    path = proxy_dir / "audit" / "gate_feature_audit.jsonl"
    if not path.exists():
        raise RuntimeError(f"missing proxy audit: {path}")
    rows: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for row in read_jsonl(path):
        rows[(str(row["task"]), int(row["sample_index"]))] = row
    return rows


def load_historical_rows(historical_dir: Path) -> Dict[str, Dict[Tuple[str, int], Dict[str, Any]]]:
    path = historical_dir / "audit" / "per_sample_audit.jsonl"
    if not path.exists():
        raise RuntimeError(f"missing historical audit: {path}")
    by_method: Dict[str, Dict[Tuple[str, int], Dict[str, Any]]] = defaultdict(dict)
    for row in read_jsonl(path):
        method = str(row.get("method_key", ""))
        if not method:
            continue
        by_method[method][(str(row["task"]), int(row["sample_index"]))] = row
    for method in [METHOD_ORIGINAL, METHOD_DRM_SWAP]:
        if method not in by_method:
            raise RuntimeError(f"historical audit missing method: {method}")
    return by_method


def block_ids_from_candidates(candidates: Any) -> List[int]:
    if not isinstance(candidates, list):
        return []
    ids: List[int] = []
    for item in candidates:
        if isinstance(item, Mapping) and "block_id" in item:
            ids.append(int(item["block_id"]))
        elif isinstance(item, int):
            ids.append(int(item))
    return ids


def protected_ids(row: Mapping[str, Any]) -> set[int]:
    ids: set[int] = set()
    for field in ["protected_block_ids", "query_critical_block_ids", "forced_keep_block_ids"]:
        for block_id in row.get(field, []) or []:
            ids.add(int(block_id))
    for item in row.get("protected_blocks_with_reasons", []) or []:
        if isinstance(item, Mapping) and "block_id" in item:
            ids.add(int(item["block_id"]))
    return ids


def anchor_evidence_signal(row: Mapping[str, Any]) -> float:
    feature_names = [
        "query_overlap_rescued",
        "entity_overlap_rescued",
        "date_number_overlap_rescued",
        "rescued_vs_active_entity_gain",
        "rescued_vs_removed_entity_gain",
        "bridge_entity_gain",
        "sentence_novelty_gain",
        "coverage_gap_v3_1",
        "semantic_novelty_rank_mean",
    ]
    return max(finite_float(row.get(name), 0.0) for name in feature_names)


def anchor_score(row: Mapping[str, Any]) -> float:
    # DRM/rescue rank is deliberately auxiliary.  The primary signal is lexical /
    # entity / coverage novelty already extracted in the selector-native audit.
    evidence = anchor_evidence_signal(row)
    rescue_percentile = finite_float(row.get("rescue_selected_mean_percentile"), 0.0)
    return max(0.0, min(1.0, 0.70 * evidence + 0.30 * rescue_percentile))


def removal_risk_for_block(row: Mapping[str, Any], block_id: int) -> float:
    # The residual/value proxy is row-level in the current artifact.  Add a small
    # candidate-local term from active drop metadata when available.
    row_level = finite_float(row.get("delete_risk_proxy_max"), 1.0)
    for item in row.get("all_active_drop_candidates", []) or []:
        if not isinstance(item, Mapping) or int(item.get("block_id", -1)) != int(block_id):
            continue
        base = finite_float(item.get("base_score"), 0.0)
        # base_score is tiny but nonnegative; squash to [0, 1) without tuning.
        base_proxy = base / (base + 1.0)
        return max(row_level, base_proxy)
    return row_level


def coverage_before_after(
    original_ids: Sequence[int],
    anchor_ids: Sequence[int],
    proposed_added: Sequence[int],
) -> Tuple[float, float]:
    anchors = set(int(x) for x in anchor_ids)
    if not anchors:
        return 1.0, 1.0
    before = len(anchors.intersection(int(x) for x in original_ids)) / len(anchors)
    after_ids = set(int(x) for x in original_ids) | set(int(x) for x in proposed_added)
    after = len(anchors.intersection(after_ids)) / len(anchors)
    return before, after


def build_audit_row(
    task: str,
    sample_index: int,
    proxy: Mapping[str, Any],
    original: Mapping[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    original_selected = [int(x) for x in original.get("used_blocks", [])]
    original_set = set(original_selected)
    protected = protected_ids(proxy)
    anchor_candidates = block_ids_from_candidates(proxy.get("selected_rescued_candidates"))
    if not anchor_candidates:
        anchor_candidates = block_ids_from_candidates(proxy.get("candidate_rescued_block_ids"))
    missing_anchors = [block_id for block_id in anchor_candidates if block_id not in original_set]

    active_drop = block_ids_from_candidates(proxy.get("selected_removed_candidates"))
    if not active_drop:
        active_drop = block_ids_from_candidates(proxy.get("candidate_removed_block_ids"))
    removable = [
        block_id
        for block_id in active_drop
        if block_id in original_set and block_id not in protected
    ]
    removable = sorted(removable, key=lambda block_id: removal_risk_for_block(proxy, block_id))

    add_count = min(len(missing_anchors), len(removable))
    proposed_added = missing_anchors[:add_count]
    proposed_removed = removable[:add_count]
    anchor_scores = [anchor_score(proxy) for _ in proposed_added] or [anchor_score(proxy)]
    removal_risks = [removal_risk_for_block(proxy, block_id) for block_id in proposed_removed]
    removal_risk_mean = sum(removal_risks) / len(removal_risks) if removal_risks else None
    removal_risk_max = max(removal_risks) if removal_risks else None
    before, after = coverage_before_after(original_selected, anchor_candidates, proposed_added)
    budget_preserved = len(proposed_added) == len(proposed_removed)
    protected_violation = bool(set(proposed_removed).intersection(protected))

    fallback_reason = ""
    if not anchor_candidates:
        fallback_reason = "no_anchor_candidates"
    elif not missing_anchors:
        fallback_reason = "all_anchor_candidates_already_selected"
    elif not removable:
        fallback_reason = "no_safe_removable_blocks"
    elif add_count <= 0:
        fallback_reason = "no_budget_neutral_anchor_swap"
    elif removal_risk_max is None:
        fallback_reason = "removal_risk_unavailable"
    elif removal_risk_max >= float(args.removal_risk_max):
        fallback_reason = "removal_risk_at_or_above_threshold"
    elif (sum(anchor_scores) / max(1, len(anchor_scores))) < float(args.anchor_score_min):
        fallback_reason = "anchor_score_below_threshold"
    elif not budget_preserved:
        fallback_reason = "budget_not_preserved"
    elif protected_violation:
        fallback_reason = "protected_violation"

    triggered = not bool(fallback_reason)
    gate_reason = "protected_anchor_swap_pass" if triggered else fallback_reason
    unavailable = []
    if proxy.get("block_texts") in (None, "", []):
        unavailable.append("full_block_texts")
    if not proxy.get("removed_block_text_preview"):
        unavailable.append("removed_block_text_preview")
    if not proxy.get("rescued_block_text_preview"):
        unavailable.append("rescued_block_text_preview")

    return {
        "sample_id": str(proxy.get("sample_id", original.get("sample_id", ""))),
        "sample_index": int(sample_index),
        "task": task,
        "original_selected_block_ids": original_selected,
        "anchor_candidate_block_ids": anchor_candidates,
        "missing_anchor_block_ids": missing_anchors,
        "removable_block_ids": removable,
        "proposed_removed_block_ids": proposed_removed,
        "proposed_added_anchor_block_ids": proposed_added,
        "anchor_score_mean": sum(anchor_scores) / max(1, len(anchor_scores)),
        "anchor_score_max": max(anchor_scores) if anchor_scores else None,
        "removal_risk_mean": removal_risk_mean,
        "removal_risk_max": removal_risk_max,
        "anchor_coverage_before": before,
        "anchor_coverage_after": after,
        "anchor_gate_triggered": bool(triggered),
        "anchor_gate_reason": gate_reason,
        "budget_preserved": bool(budget_preserved),
        "protected_violation": bool(protected_violation),
        "fallback_reason": fallback_reason,
        "proxy_only_no_generation": True,
        "anchor_feature_source": "selector_native_audit_proxy_plus_rescue_candidates",
        "anchor_feature_unavailable": unavailable,
        "protected_block_ids": sorted(protected),
        "anchor_evidence_signal": anchor_evidence_signal(proxy),
        "drm_rescue_auxiliary_used": True,
    }


def summarize(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    trigger_count = sum(bool(row.get("anchor_gate_triggered")) for row in rows)
    budget_violations = sum(not bool(row.get("budget_preserved")) for row in rows)
    protected_violations = sum(bool(row.get("protected_violation")) for row in rows)
    missing_counts = [len(row.get("missing_anchor_block_ids", [])) for row in rows]
    candidate_counts = [len(row.get("anchor_candidate_block_ids", [])) for row in rows]
    return {
        "samples": len(rows),
        "tasks": dict(Counter(str(row.get("task")) for row in rows)),
        "anchor_candidate_coverage": sum(1 for count in candidate_counts if count > 0) / max(1, len(rows)),
        "missing_anchor_count_total": sum(missing_counts),
        "missing_anchor_count_mean": sum(missing_counts) / max(1, len(missing_counts)),
        "trigger_count": trigger_count,
        "trigger_rate": trigger_count / max(1, len(rows)),
        "budget_violation": budget_violations,
        "protected_violation": protected_violations,
        "gate_reason_counts": dict(Counter(str(row.get("anchor_gate_reason", "")) for row in rows)),
        "field_coverage": {
            field: sum(row.get(field) not in (None, "", [], "unavailable") for row in rows)
            for field in AUDIT_FIELDS
        },
    }


def write_report(output_dir: Path, rows: Sequence[Mapping[str, Any]], summary: Mapping[str, Any], args: argparse.Namespace) -> None:
    lines = [
        "# Protected Evidence Anchor Audit",
        "",
        f"- Result directory: `{output_dir.resolve()}`",
        f"- Historical input: `{args.historical_dir}`",
        f"- Proxy/audit input: `{args.proxy_dir}`",
        "- Model/GPU used: **No**",
        "- Generation performed: **No**",
        "- SlimInfer Original selector modified: **No**",
        "- Replay semantics: this audit only proposes an external controller; it does not create predictions.",
        "",
        "## Thresholds",
        "",
        f"- `removal_risk_max < {args.removal_risk_max}`",
        f"- `anchor_score_mean >= {args.anchor_score_min}`",
        "",
        "## Summary",
        "",
        f"- Samples: {summary['samples']}",
        f"- Anchor candidate coverage: {summary['anchor_candidate_coverage']:.2%}",
        f"- Missing anchor total / mean: {summary['missing_anchor_count_total']} / {summary['missing_anchor_count_mean']:.3f}",
        f"- Gate trigger rate: {summary['trigger_count']}/{summary['samples']} ({summary['trigger_rate']:.2%})",
        f"- Budget violation: {summary['budget_violation']}",
        f"- Protected violation: {summary['protected_violation']}",
        f"- Gate reasons: `{json.dumps(summary['gate_reason_counts'], ensure_ascii=False)}`",
        "",
        "## Limitations",
        "",
        "- Full block text is unavailable in the existing tiny proxy artifact, so anchor text matching is based on selector-native entity/coverage proxies and rescued/removed previews.",
        "- DRM/rescue score is used only as an auxiliary rank signal, not as the sole anchor definition.",
    ]
    (output_dir / "protected_evidence_anchor_audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical_dir", type=Path, default=DEFAULT_HISTORICAL_DIR)
    parser.add_argument("--proxy_dir", type=Path, default=DEFAULT_PROXY_DIR)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--sample_indices_by_task", type=str, default="")
    parser.add_argument("--removal_risk_max", type=float, default=0.20)
    parser.add_argument("--anchor_score_min", type=float, default=0.50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or Path("results") / f"protected_evidence_anchor_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=False)
    sample_indices = parse_sample_indices(args.sample_indices_by_task)
    keys = sample_keys(sample_indices)
    proxy_rows = load_proxy_rows(args.proxy_dir)
    historical = load_historical_rows(args.historical_dir)

    audit_rows: List[Dict[str, Any]] = []
    missing: List[str] = []
    for task, index in keys:
        key = (task, int(index))
        proxy = proxy_rows.get(key)
        original = historical[METHOD_ORIGINAL].get(key)
        if proxy is None:
            missing.append(f"{task}:{index}:proxy")
            continue
        if original is None:
            missing.append(f"{task}:{index}:original")
            continue
        audit_rows.append(build_audit_row(task, int(index), proxy, original, args))
    if missing:
        raise RuntimeError(f"missing required rows: {missing}")

    summary = summarize(audit_rows)
    write_jsonl(output_dir / "protected_evidence_anchor_audit.jsonl", audit_rows)
    write_csv(output_dir / "protected_evidence_anchor_audit.csv", audit_rows, AUDIT_FIELDS)
    write_json(
        output_dir / "protected_evidence_anchor_audit_summary.json",
        {
            "summary": summary,
            "sample_indices_by_task": sample_indices,
            "thresholds": {
                "removal_risk_max": args.removal_risk_max,
                "anchor_score_min": args.anchor_score_min,
            },
            "historical_dir": str(args.historical_dir),
            "proxy_dir": str(args.proxy_dir),
            "model_or_gpu_used": False,
            "generation_performed": False,
            "sliminfer_original_modified": False,
        },
    )
    write_report(output_dir, audit_rows, summary, args)
    print(json.dumps({
        "status": "ok",
        "output_dir": str(output_dir.resolve()),
        "samples": len(audit_rows),
        "trigger_count": summary["trigger_count"],
        "trigger_rate": summary["trigger_rate"],
        "budget_violation": summary["budget_violation"],
        "protected_violation": summary["protected_violation"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
