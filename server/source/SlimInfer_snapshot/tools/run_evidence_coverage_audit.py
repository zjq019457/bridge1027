#!/usr/bin/env python3
"""Audit-only Evidence Coverage Controller.

SlimInfer Original is treated as a frozen base selector.  This script reads
existing artifacts and computes an external coverage-controller proposal; it
does not run a model, generate predictions, or change SlimInfer selection.

When true selected-block text/offsets are unavailable, the script records that
fact and falls back to selector-native entity/coverage proxies.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


DEFAULT_HISTORICAL_DIR = Path("results/drm_budget_neutral_swap_multihop200_20260706_042717")
DEFAULT_PROXY_DIR = Path("results/residual_aware_proxy_audit_tiny_20260706_204145")
DEFAULT_ANCHOR_DIR = Path("results/protected_evidence_anchor_audit_20260706_214256")
DEFAULT_LONGBENCH_DIR = Path("../HeadKV/data/LongBench")
DEFAULT_SAMPLE_INDICES = {
    "hotpotqa": [0, 1, 2, 3, 4, 5, 41, 56, 57, 117],
    "musique": [0, 1, 2, 3, 4, 20, 50, 54, 56, 134],
}
METHOD_ORIGINAL = "sliminfer_original"

STOPWORDS = {
    "which", "what", "when", "where", "who", "whom", "whose", "how",
    "many", "much", "was", "were", "is", "are", "did", "does", "do",
    "the", "a", "an", "or", "and", "to", "of", "in", "on", "for",
    "with", "from", "by", "as", "that", "this", "than", "then",
    "first", "case", "one", "had", "has", "have",
}

AUDIT_FIELDS = [
    "sample_id",
    "sample_index",
    "task",
    "query_terms",
    "query_entities_or_proxy",
    "query_numbers",
    "selected_block_ids_original",
    "candidate_block_ids",
    "block_text_available",
    "block_offset_available",
    "title_or_doc_marker_available",
    "query_term_coverage_before",
    "query_term_coverage_after",
    "query_number_coverage_before",
    "query_number_coverage_after",
    "entity_overlap_coverage_before",
    "entity_overlap_coverage_after",
    "bridge_proxy_coverage_before",
    "bridge_proxy_coverage_after",
    "support_doc_diversity_before",
    "support_doc_diversity_after",
    "coverage_gap_score",
    "missing_coverage_types",
    "coverage_candidate_block_ids",
    "proposed_added_block_ids",
    "proposed_removed_block_ids",
    "delete_risk_proxy_max",
    "coverage_gate_triggered",
    "coverage_gate_reason",
    "budget_preserved",
    "protected_violation",
    "fallback_reason",
]


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


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
    if not str(value or "").strip():
        return {task: list(indices) for task, indices in DEFAULT_SAMPLE_INDICES.items()}
    parsed: Dict[str, List[int]] = {}
    for chunk in value.split(";"):
        if not chunk.strip():
            continue
        task, raw = chunk.split(":", 1)
        parsed[task.strip()] = [int(part.strip()) for part in raw.split(",") if part.strip()]
    return parsed


def load_proxy_rows(proxy_dir: Path) -> Dict[Tuple[str, int], Dict[str, Any]]:
    path = proxy_dir / "audit" / "gate_feature_audit.jsonl"
    if not path.exists():
        raise RuntimeError(f"missing tiny proxy audit: {path}")
    return {(str(row["task"]), int(row["sample_index"])): row for row in read_jsonl(path)}


def load_anchor_rows(anchor_dir: Path) -> Dict[Tuple[str, int], Dict[str, Any]]:
    path = anchor_dir / "protected_evidence_anchor_audit.jsonl"
    if not path.exists():
        raise RuntimeError(f"missing protected anchor audit: {path}")
    return {(str(row["task"]), int(row["sample_index"])): row for row in read_jsonl(path)}


def load_historical_original(historical_dir: Path) -> Dict[Tuple[str, int], Dict[str, Any]]:
    path = historical_dir / "audit" / "per_sample_audit.jsonl"
    if not path.exists():
        raise RuntimeError(f"missing historical per-sample audit: {path}")
    rows: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for row in read_jsonl(path):
        if row.get("method_key") == METHOD_ORIGINAL:
            rows[(str(row["task"]), int(row["sample_index"]))] = row
    return rows


def load_longbench(longbench_dir: Path, task: str) -> List[Dict[str, Any]]:
    candidates = [
        longbench_dir / f"{task}.jsonl",
        longbench_dir / task / "test.jsonl",
        longbench_dir / task / f"{task}.jsonl",
    ]
    for path in candidates:
        if path.exists():
            return read_jsonl(path)
    raise RuntimeError(f"LongBench task not found for {task} under {longbench_dir}")


def query_terms(question: str) -> List[str]:
    terms: List[str] = []
    for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9'-]*", question.lower()):
        if len(token) <= 2 or token in STOPWORDS:
            continue
        if token not in terms:
            terms.append(token)
    return terms


def query_entities(question: str, proxy: Mapping[str, Any]) -> List[str]:
    entities: List[str] = []
    for item in proxy.get("question_entity_units", []) or []:
        text = str(item).strip().lower()
        if text and text not in entities:
            entities.append(text)
    for match in re.findall(r"\b[A-Z][A-Za-z0-9&.'-]*(?:\s+[A-Z][A-Za-z0-9&.'-]*)*", question):
        text = match.strip().lower()
        if len(text) > 1 and text not in entities:
            entities.append(text)
    return entities


def query_numbers(question: str) -> List[str]:
    numbers: List[str] = []
    for match in re.findall(r"\b\d+(?:,\d{3})*(?:\.\d+)?\b", question):
        text = match.replace(",", "")
        if text not in numbers:
            numbers.append(text)
    return numbers


def coverage(items: Sequence[str], units: Sequence[Any]) -> float:
    if not items:
        return 1.0
    haystack = " ".join(str(unit).lower() for unit in units)
    return sum(1 for item in items if str(item).lower() in haystack) / len(items)


def block_ids(candidates: Any) -> List[int]:
    if not isinstance(candidates, list):
        return []
    result: List[int] = []
    for item in candidates:
        if isinstance(item, Mapping) and "block_id" in item:
            result.append(int(item["block_id"]))
        elif isinstance(item, int):
            result.append(int(item))
    return result


def build_row(
    task: str,
    index: int,
    proxy: Mapping[str, Any],
    anchor: Mapping[str, Any],
    original: Mapping[str, Any],
    raw_sample: Mapping[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    question = str(raw_sample.get("input", ""))
    context = str(raw_sample.get("context", ""))
    q_terms = query_terms(question)
    q_entities = query_entities(question, proxy)
    q_numbers = query_numbers(question)
    active_units = list(proxy.get("active_entity_units", []) or [])
    rescued_units = list(proxy.get("rescued_entity_units", []) or [])
    after_units = sorted(set(str(unit).lower() for unit in active_units + rescued_units))

    qterm_before = coverage(q_terms, active_units)
    qterm_after = coverage(q_terms, after_units)
    qnum_before = coverage(q_numbers, active_units)
    qnum_after = coverage(q_numbers, after_units)
    ent_before = coverage(q_entities, active_units)
    ent_after = coverage(q_entities, after_units)

    bridge_gap = max(
        finite_float(proxy.get("bridge_entity_gain")),
        finite_float(proxy.get("coverage_gap_v3_1")),
        finite_float(proxy.get("sentence_novelty_gain")),
    )
    bridge_before = max(0.0, 1.0 - bridge_gap)
    bridge_after = min(1.0, bridge_before + bridge_gap)
    support_before: Optional[float] = None
    support_after: Optional[float] = None

    qterm_gain = max(0.0, qterm_after - qterm_before)
    qnum_gain = max(0.0, qnum_after - qnum_before)
    ent_gain = max(0.0, ent_after - ent_before)
    doc_gap_proxy = max(
        finite_float(proxy.get("title_or_heading_novelty")),
        0.25 * finite_float(proxy.get("context_entity_novelty")),
    )
    coverage_gap_score = (
        0.25 * qterm_gain
        + 0.20 * qnum_gain
        + 0.20 * ent_gain
        + 0.25 * bridge_gap
        + 0.10 * doc_gap_proxy
    )

    missing: List[str] = []
    if qterm_before < 1.0 and qterm_after > qterm_before:
        missing.append("query_terms")
    if qnum_before < 1.0 and qnum_after > qnum_before:
        missing.append("query_numbers")
    if ent_before < 1.0 and ent_after > ent_before:
        missing.append("entities")
    if bridge_gap > 0.0:
        missing.append("bridge_proxy")
    if support_before is None:
        missing.append("support_doc_diversity_unavailable")

    selected_ids = [int(x) for x in original.get("used_blocks", [])]
    candidate_ids = block_ids(proxy.get("selected_rescued_candidates")) or block_ids(proxy.get("candidate_rescued_block_ids"))
    coverage_candidates = [block_id for block_id in candidate_ids if block_id not in set(selected_ids)]
    removable = [int(x) for x in anchor.get("removable_block_ids", [])]
    add_count = min(len(coverage_candidates), len(removable))
    proposed_added = coverage_candidates[:add_count]
    proposed_removed = removable[:add_count]
    delete_risk = finite_float(proxy.get("delete_risk_proxy_max"), 1.0)
    budget_preserved = len(proposed_added) == len(proposed_removed)
    protected_violation = bool(anchor.get("protected_violation", False))

    fallback_reason = ""
    if not coverage_candidates:
        fallback_reason = "no_coverage_candidate_blocks"
    elif add_count <= 0:
        fallback_reason = "no_budget_neutral_removal"
    elif delete_risk >= float(args.delete_risk_max):
        fallback_reason = "delete_risk_at_or_above_threshold"
    elif coverage_gap_score < float(args.coverage_gap_min):
        fallback_reason = "coverage_gap_below_threshold"
    elif not budget_preserved:
        fallback_reason = "budget_not_preserved"
    elif protected_violation:
        fallback_reason = "protected_violation"
    triggered = not bool(fallback_reason)

    block_text_available = bool(proxy.get("block_texts"))
    return {
        "sample_id": str(proxy.get("sample_id", original.get("sample_id", ""))),
        "sample_index": int(index),
        "task": task,
        "query_terms": q_terms,
        "query_entities_or_proxy": q_entities,
        "query_numbers": q_numbers,
        "selected_block_ids_original": selected_ids,
        "candidate_block_ids": candidate_ids,
        "block_text_available": block_text_available,
        "block_offset_available": False,
        "title_or_doc_marker_available": bool(re.search(r"\bPassage\s+\d+:", context)) or bool(proxy.get("title_or_heading_available")),
        "query_term_coverage_before": qterm_before,
        "query_term_coverage_after": qterm_after,
        "query_number_coverage_before": qnum_before,
        "query_number_coverage_after": qnum_after,
        "entity_overlap_coverage_before": ent_before,
        "entity_overlap_coverage_after": ent_after,
        "bridge_proxy_coverage_before": bridge_before,
        "bridge_proxy_coverage_after": bridge_after,
        "support_doc_diversity_before": support_before,
        "support_doc_diversity_after": support_after,
        "coverage_gap_score": coverage_gap_score,
        "missing_coverage_types": sorted(set(missing)),
        "coverage_candidate_block_ids": coverage_candidates,
        "proposed_added_block_ids": proposed_added,
        "proposed_removed_block_ids": proposed_removed,
        "delete_risk_proxy_max": delete_risk,
        "coverage_gate_triggered": bool(triggered),
        "coverage_gate_reason": "coverage_controller_pass" if triggered else fallback_reason,
        "budget_preserved": bool(budget_preserved),
        "protected_violation": bool(protected_violation),
        "fallback_reason": fallback_reason,
        "coverage_feature_source": "selector_native_proxy" if not block_text_available else "block_text",
        "context_available": bool(context),
        "gold_answer_used_for_gate": False,
    }


def summarize(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    values = [finite_float(row.get("coverage_gap_score")) for row in rows]
    sorted_values = sorted(values)
    def pct(p: float) -> float:
        if not sorted_values:
            return 0.0
        pos = min(len(sorted_values) - 1, max(0, int(round((len(sorted_values) - 1) * p))))
        return sorted_values[pos]
    return {
        "samples": len(rows),
        "tasks": dict(Counter(str(row["task"]) for row in rows)),
        "block_text_available": sum(bool(row.get("block_text_available")) for row in rows),
        "block_offset_available": sum(bool(row.get("block_offset_available")) for row in rows),
        "title_or_doc_marker_available": sum(bool(row.get("title_or_doc_marker_available")) for row in rows),
        "coverage_gap_score": {
            "mean": sum(values) / max(1, len(values)),
            "min": min(values) if values else 0.0,
            "p50": pct(0.50),
            "p75": pct(0.75),
            "max": max(values) if values else 0.0,
        },
        "trigger_count": sum(bool(row.get("coverage_gate_triggered")) for row in rows),
        "trigger_rate": sum(bool(row.get("coverage_gate_triggered")) for row in rows) / max(1, len(rows)),
        "budget_violation": sum(not bool(row.get("budget_preserved")) for row in rows),
        "protected_violation": sum(bool(row.get("protected_violation")) for row in rows),
        "gate_reason_counts": dict(Counter(str(row.get("coverage_gate_reason")) for row in rows)),
        "field_coverage": {
            field: sum(row.get(field) not in (None, "", [], "unavailable") for row in rows)
            for field in AUDIT_FIELDS
        },
    }


def write_report(output_dir: Path, summary: Mapping[str, Any], args: argparse.Namespace) -> None:
    lines = [
        "# Evidence Coverage Controller Audit",
        "",
        f"- Result directory: `{output_dir.resolve()}`",
        "## Input artifacts",
        "",
        f"- Historical 200-sample audit: `{args.historical_dir}`",
        f"- Tiny proxy audit: `{args.proxy_dir}`",
        f"- Protected anchor audit: `{args.anchor_dir}`",
        f"- LongBench raw data: `{args.longbench_dir}`",
        "",
        "## Availability",
        "",
        f"- Full selected block text available: {summary['block_text_available']}/{summary['samples']}",
        f"- Block offsets available: {summary['block_offset_available']}/{summary['samples']}",
        f"- Context/title markers available: {summary['title_or_doc_marker_available']}/{summary['samples']}",
        "- Because selected-block text/offsets are unavailable, coverage is proxy coverage, not true text-level coverage.",
        "",
        "## Gate",
        "",
        f"- Threshold: `coverage_gap_score >= {args.coverage_gap_min}`",
        f"- Threshold: `delete_risk_proxy_max < {args.delete_risk_max}`",
        f"- Trigger rate: {summary['trigger_count']}/{summary['samples']} ({summary['trigger_rate']:.2%})",
        f"- Budget violation: {summary['budget_violation']}",
        f"- Protected violation: {summary['protected_violation']}",
        f"- Coverage gap distribution: `{json.dumps(summary['coverage_gap_score'], ensure_ascii=False)}`",
        f"- Gate reasons: `{json.dumps(summary['gate_reason_counts'], ensure_ascii=False)}`",
    ]
    (output_dir / "evidence_coverage_audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical_dir", type=Path, default=DEFAULT_HISTORICAL_DIR)
    parser.add_argument("--proxy_dir", type=Path, default=DEFAULT_PROXY_DIR)
    parser.add_argument("--anchor_dir", type=Path, default=DEFAULT_ANCHOR_DIR)
    parser.add_argument("--longbench_dir", type=Path, default=DEFAULT_LONGBENCH_DIR)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--sample_indices_by_task", type=str, default="")
    parser.add_argument("--coverage_gap_min", type=float, default=0.08)
    parser.add_argument("--delete_risk_max", type=float, default=0.20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or Path("results") / f"evidence_coverage_controller_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=False)
    sample_indices = parse_sample_indices(args.sample_indices_by_task)
    proxy_rows = load_proxy_rows(args.proxy_dir)
    anchor_rows = load_anchor_rows(args.anchor_dir)
    original_rows = load_historical_original(args.historical_dir)
    raw_by_task = {task: load_longbench(args.longbench_dir, task) for task in sample_indices}

    rows: List[Dict[str, Any]] = []
    missing: List[str] = []
    for task, indices in sample_indices.items():
        for index in indices:
            key = (task, int(index))
            proxy = proxy_rows.get(key)
            anchor = anchor_rows.get(key)
            original = original_rows.get(key)
            raw = raw_by_task[task][int(index)] if int(index) < len(raw_by_task[task]) else None
            if proxy is None or anchor is None or original is None or raw is None:
                missing.append(f"{task}:{index}:proxy={proxy is not None}:anchor={anchor is not None}:original={original is not None}:raw={raw is not None}")
                continue
            rows.append(build_row(task, int(index), proxy, anchor, original, raw, args))
    if missing:
        raise RuntimeError(f"missing required rows: {missing}")
    summary = summarize(rows)
    write_jsonl(output_dir / "evidence_coverage_audit.jsonl", rows)
    write_csv(output_dir / "evidence_coverage_audit.csv", rows, AUDIT_FIELDS)
    write_json(
        output_dir / "evidence_coverage_audit_summary.json",
        {
            "summary": summary,
            "input_artifacts": {
                "historical_dir": str(args.historical_dir),
                "proxy_dir": str(args.proxy_dir),
                "anchor_dir": str(args.anchor_dir),
                "longbench_dir": str(args.longbench_dir),
            },
            "thresholds": {
                "coverage_gap_min": args.coverage_gap_min,
                "delete_risk_max": args.delete_risk_max,
            },
            "sample_indices_by_task": sample_indices,
            "model_or_gpu_used": False,
            "generation_performed": False,
            "gold_answer_used_for_gate": False,
            "sliminfer_original_modified": False,
        },
    )
    write_report(output_dir, summary, args)
    print(json.dumps({
        "status": "ok",
        "output_dir": str(output_dir.resolve()),
        "samples": len(rows),
        "trigger_count": summary["trigger_count"],
        "trigger_rate": summary["trigger_rate"],
        "block_text_available": summary["block_text_available"],
        "block_offset_available": summary["block_offset_available"],
        "budget_violation": summary["budget_violation"],
        "protected_violation": summary["protected_violation"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
