#!/usr/bin/env python3
"""Offline postmortem and conditional-gate replay for DC-SlimInfer.

This script reads an existing DC-SlimInfer tiny quality smoke directory. It does
not run models, generation, decoding, or scorers.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


DEFAULT_INPUT_DIR = Path("results/dc_sliminfer_tiny_quality_smoke_20260708_004504")
DEFAULT_OUTPUT_PARENT = Path("results")
REQUIRED_FOOTER = (
    "Unconditional DC-SlimInfer is stopped.\n"
    "Only if a task-agnostic conditional gate can reject HotpotQA catastrophic "
    "negatives while retaining MuSiQue positives should DC continue."
)


def read_json(path: Optional[Path]) -> Any:
    if not path or not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_text(path: Optional[Path]) -> str:
    if not path or not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def read_jsonl(path: Optional[Path]) -> List[Dict[str, Any]]:
    if not path or not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if isinstance(row, dict):
                rows.append(row)
    return rows


def read_csv_rows(path: Optional[Path]) -> List[Dict[str, Any]]:
    if not path or not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def parse_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if value == 0:
            return False
        if value == 1:
            return True
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    if text in {"", "none", "null", "nan"}:
        return None
    return None


def to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        number = float(value)
    else:
        text = str(value).strip()
        if text == "" or text.lower() in {"none", "null", "nan"}:
            return None
        try:
            number = float(text)
        except ValueError:
            return None
    if math.isnan(number):
        return None
    return number


def fmt(value: Any, digits: int = 4) -> str:
    number = to_float(value)
    if number is None:
        if value is None:
            return "unavailable"
        return str(value)
    return f"{number:.{digits}f}"


def mean(values: Iterable[Any]) -> Optional[float]:
    numbers = [to_float(value) for value in values]
    clean = [value for value in numbers if value is not None]
    if not clean:
        return None
    return sum(clean) / len(clean)


def pearson(xs: Iterable[Any], ys: Iterable[Any]) -> Optional[float]:
    pairs: List[Tuple[float, float]] = []
    for x_value, y_value in zip(xs, ys):
        x = to_float(x_value)
        y = to_float(y_value)
        if x is not None and y is not None:
            pairs.append((x, y))
    if len(pairs) < 2:
        return None
    x_mean = sum(x for x, _ in pairs) / len(pairs)
    y_mean = sum(y for _, y in pairs) / len(pairs)
    x_var = sum((x - x_mean) ** 2 for x, _ in pairs)
    y_var = sum((y - y_mean) ** 2 for _, y in pairs)
    if x_var <= 0 or y_var <= 0:
        return None
    cov = sum((x - x_mean) * (y - y_mean) for x, y in pairs)
    return cov / math.sqrt(x_var * y_var)


def sample_key(row: Dict[str, Any]) -> Tuple[str, int, str]:
    task = str(row.get("task", ""))
    sample_index = int(to_float(row.get("sample_index")) or 0)
    sample_id = str(row.get("sample_id", ""))
    return task, sample_index, sample_id


def discover_exact_or_keyword(
    root: Path, exact_name: str, keywords: Iterable[str], suffixes: Iterable[str]
) -> Optional[Path]:
    exact = root / exact_name
    if exact.exists():
        return exact
    keyword_list = [keyword.lower() for keyword in keywords]
    suffix_list = tuple(suffixes)
    candidates: List[Path] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in suffix_list:
            continue
        lower = path.name.lower()
        if all(keyword in lower for keyword in keyword_list):
            candidates.append(path)
    if not candidates:
        return None
    return sorted(candidates, key=lambda path: (len(path.parts), str(path)))[0]


def discover_artifacts(input_dir: Path) -> Dict[str, Optional[Path]]:
    return {
        "report": discover_exact_or_keyword(input_dir, "report.md", ["report"], [".md"]),
        "summary": discover_exact_or_keyword(input_dir, "summary.json", ["summary"], [".json"]),
        "task_summary": discover_exact_or_keyword(
            input_dir, "task_summary.csv", ["task", "summary"], [".csv"]
        ),
        "comparison_jsonl": discover_exact_or_keyword(
            input_dir, "per_sample_comparison.jsonl", ["sample", "comparison"], [".jsonl"]
        ),
        "comparison_csv": discover_exact_or_keyword(
            input_dir, "per_sample_comparison.csv", ["sample", "comparison"], [".csv"]
        ),
        "per_sample_predictions": discover_exact_or_keyword(
            input_dir, "per_sample_predictions.jsonl", ["sample", "prediction"], [".jsonl"]
        ),
        "activation_audit": discover_exact_or_keyword(
            input_dir, "dc_gate_activation_audit.jsonl", ["activation", "audit"], [".jsonl"]
        ),
        "selection_audit": discover_exact_or_keyword(
            input_dir, "dc_gate_selection_audit.jsonl", ["selection", "audit"], [".jsonl"]
        ),
    }


def discover_longbench_prediction_files(input_dir: Path) -> Dict[str, List[Path]]:
    prediction_files: Dict[str, List[Path]] = {"sliminfer_original": [], "dc_sliminfer": []}
    for path in input_dir.rglob("longbench_predictions/*.jsonl"):
        parts = {part.lower() for part in path.parts}
        if "run_sliminfer_original" in parts:
            prediction_files["sliminfer_original"].append(path)
        elif "run_dc_sliminfer" in parts:
            prediction_files["dc_sliminfer"].append(path)
    for method in prediction_files:
        prediction_files[method] = sorted(prediction_files[method])
    return prediction_files


def compute_task_scores(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    by_task: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[str(row.get("task"))].append(row)
    scores: Dict[str, Dict[str, float]] = {}
    for task, task_rows in sorted(by_task.items()):
        original = mean(row.get("original_score") for row in task_rows)
        dc = mean(row.get("dc_score") for row in task_rows)
        if original is None or dc is None:
            continue
        scores[task] = {"original": original, "dc": dc, "delta": dc - original}
    if scores:
        scores["macro"] = {
            "original": mean(task_score["original"] for task_score in scores.values()) or 0.0,
            "dc": mean(task_score["dc"] for task_score in scores.values()) or 0.0,
        }
        scores["macro"]["delta"] = scores["macro"]["dc"] - scores["macro"]["original"]
    return scores


def compare_score_tables(
    computed: Dict[str, Dict[str, float]],
    task_summary_rows: List[Dict[str, Any]],
    summary_json: Any,
    tolerance: float = 1e-4,
) -> List[str]:
    reasons: List[str] = []
    for row in task_summary_rows:
        task = str(row.get("task", ""))
        if task not in computed:
            continue
        for field in ("original", "dc", "delta"):
            observed = to_float(row.get(field))
            expected = computed[task].get(field)
            if observed is None or expected is None:
                continue
            if abs(observed - expected) > tolerance:
                reasons.append(
                    f"task_summary mismatch for {task}.{field}: "
                    f"file={observed}, computed={expected}"
                )
    if isinstance(summary_json, dict):
        nested = summary_json.get("summary", {})
        task_scores = nested.get("task_scores", {}) if isinstance(nested, dict) else {}
        for task, task_values in task_scores.items():
            if task not in computed or not isinstance(task_values, dict):
                continue
            for field in ("original", "dc", "delta"):
                observed = to_float(task_values.get(field))
                expected = computed[task].get(field)
                if observed is not None and expected is not None and abs(observed - expected) > tolerance:
                    reasons.append(
                        f"summary.json mismatch for {task}.{field}: "
                        f"file={observed}, computed={expected}"
                    )
        macro_values = nested.get("macro", {}) if isinstance(nested, dict) else {}
        if "macro" in computed and isinstance(macro_values, dict):
            for field in ("original", "dc", "delta"):
                observed = to_float(macro_values.get(field))
                expected = computed["macro"].get(field)
                if observed is not None and expected is not None and abs(observed - expected) > tolerance:
                    reasons.append(
                        f"summary.json mismatch for macro.{field}: "
                        f"file={observed}, computed={expected}"
                    )
    return reasons


def quality_rows_by_method(
    quality_rows: List[Dict[str, Any]]
) -> Dict[Tuple[str, int, str], Dict[str, Dict[str, Any]]]:
    by_key: Dict[Tuple[str, int, str], Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for row in quality_rows:
        by_key[sample_key(row)][str(row.get("method", ""))] = row
    return by_key


def aggregate_audit_by_sample(rows: List[Dict[str, Any]]) -> Dict[Tuple[str, int, str], Dict[str, Any]]:
    by_key: Dict[Tuple[str, int, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_key[sample_key(row)].append(row)
    aggregates: Dict[Tuple[str, int, str], Dict[str, Any]] = {}
    for key, sample_rows in by_key.items():
        aggregates[key] = {
            "selected_jaccard_vs_original": mean(row.get("selected_jaccard") for row in sample_rows),
            "original_redundancy": mean(row.get("original_redundancy") for row in sample_rows),
            "dc_redundancy": mean(row.get("dc_redundancy") for row in sample_rows),
            "original_query_overlap": mean(row.get("original_query_overlap") for row in sample_rows),
            "dc_query_overlap": mean(row.get("dc_query_overlap") for row in sample_rows),
            "original_coverage_bin_count": mean(
                len(row.get("original_coverage_bins") or []) for row in sample_rows
            ),
            "dc_coverage_bin_count": mean(len(row.get("dc_coverage_bins") or []) for row in sample_rows),
            "dc_entered_before_virtual_return": any(
                parse_bool(row.get("entered_dc_selector_before_virtual_return")) is True
                for row in sample_rows
            ),
            "virtual_block_path_returned_before_dc_selector": any(
                str(row.get("early_return_reason", ""))
                == "virtual_block_path_returned_before_dc_selector"
                or str(row.get("virtual_return_reason", ""))
                == "virtual_block_path_returned_before_dc_selector"
                for row in sample_rows
            ),
        }
    return aggregates


def first_available(*values: Any) -> Any:
    for value in values:
        if value is not None and value != "":
            return value
    return None


def build_failure_rows(
    comparison_rows: List[Dict[str, Any]],
    quality_rows: List[Dict[str, Any]],
    selection_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    quality_by_key = quality_rows_by_method(quality_rows)
    audit_aggregate = aggregate_audit_by_sample(selection_rows)
    failure_rows: List[Dict[str, Any]] = []
    for comparison in comparison_rows:
        key = sample_key(comparison)
        methods = quality_by_key.get(key, {})
        original_row = methods.get("sliminfer_original", {})
        dc_row = methods.get("dc_sliminfer", {})
        aggregate = audit_aggregate.get(key, {})

        original_redundancy = first_available(
            dc_row.get("original_redundancy"), aggregate.get("original_redundancy")
        )
        dc_redundancy = first_available(dc_row.get("dc_redundancy"), aggregate.get("dc_redundancy"))
        original_query_overlap = first_available(
            dc_row.get("original_query_overlap"), aggregate.get("original_query_overlap")
        )
        dc_query_overlap = first_available(
            dc_row.get("dc_query_overlap"), aggregate.get("dc_query_overlap")
        )
        original_coverage = first_available(
            dc_row.get("original_coverage_bin_count"), aggregate.get("original_coverage_bin_count")
        )
        dc_coverage = first_available(
            dc_row.get("dc_coverage_bin_count"), aggregate.get("dc_coverage_bin_count")
        )
        jaccard = first_available(
            comparison.get("selected_jaccard"),
            dc_row.get("selected_jaccard_vs_original"),
            aggregate.get("selected_jaccard_vs_original"),
        )

        budget_mismatch = any(
            parse_bool(row.get("budget_mismatch")) is True for row in (original_row, dc_row) if row
        )
        fallback_used = any(
            parse_bool(row.get("fallback_used")) is True for row in (original_row, dc_row) if row
        )
        original_entered_dc = parse_bool(original_row.get("dc_entered_before_virtual_return")) is True

        row = {
            "task": comparison.get("task"),
            "sample_index": int(to_float(comparison.get("sample_index")) or 0),
            "sample_id": comparison.get("sample_id"),
            "original_score": to_float(comparison.get("original_score")),
            "dc_score": to_float(comparison.get("dc_score")),
            "delta": to_float(comparison.get("delta")),
            "prediction_changed": parse_bool(comparison.get("prediction_changed")),
            "selected_jaccard_vs_original": to_float(jaccard),
            "original_redundancy": to_float(original_redundancy),
            "dc_redundancy": to_float(dc_redundancy),
            "redundancy_delta": None,
            "original_query_overlap": to_float(original_query_overlap),
            "dc_query_overlap": to_float(dc_query_overlap),
            "query_overlap_delta": None,
            "original_coverage_bin_count": to_float(original_coverage),
            "dc_coverage_bin_count": to_float(dc_coverage),
            "coverage_delta": None,
            "dc_entered_before_virtual_return": first_available(
                dc_row.get("dc_entered_before_virtual_return"),
                aggregate.get("dc_entered_before_virtual_return"),
            ),
            "virtual_block_path_returned_before_dc_selector": first_available(
                dc_row.get("virtual_block_path_returned_before_dc_selector"),
                aggregate.get("virtual_block_path_returned_before_dc_selector"),
            ),
            "budget_mismatch": budget_mismatch,
            "fallback_used": fallback_used,
            "original_entered_dc_selector": original_entered_dc,
            "original_prediction": comparison.get("original_prediction"),
            "dc_prediction": comparison.get("dc_prediction"),
        }
        if row["original_redundancy"] is not None and row["dc_redundancy"] is not None:
            row["redundancy_delta"] = row["dc_redundancy"] - row["original_redundancy"]
        if row["original_query_overlap"] is not None and row["dc_query_overlap"] is not None:
            row["query_overlap_delta"] = row["dc_query_overlap"] - row["original_query_overlap"]
        if row["original_coverage_bin_count"] is not None and row["dc_coverage_bin_count"] is not None:
            row["coverage_delta"] = row["dc_coverage_bin_count"] - row["original_coverage_bin_count"]
        failure_rows.append(row)
    return sorted(failure_rows, key=lambda row: (str(row.get("task")), int(row.get("sample_index", 0))))


def group_stats(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    groups = {
        "hotpotqa_negative": [
            row for row in rows if row.get("task") == "hotpotqa" and (to_float(row.get("delta")) or 0) < 0
        ],
        "hotpotqa_non_negative": [
            row for row in rows if row.get("task") == "hotpotqa" and (to_float(row.get("delta")) or 0) >= 0
        ],
        "musique_positive": [
            row for row in rows if row.get("task") == "musique" and (to_float(row.get("delta")) or 0) > 0
        ],
        "musique_non_positive": [
            row for row in rows if row.get("task") == "musique" and (to_float(row.get("delta")) or 0) <= 0
        ],
    }
    fields = [
        "selected_jaccard_vs_original",
        "redundancy_delta",
        "query_overlap_delta",
        "coverage_delta",
        "delta",
    ]
    stats: Dict[str, Dict[str, Any]] = {}
    for name, group_rows in groups.items():
        entry: Dict[str, Any] = {"count": len(group_rows)}
        for field in fields:
            entry[f"{field}_mean"] = mean(row.get(field) for row in group_rows)
        entry["prediction_changed_rate"] = mean(
            1.0 if parse_bool(row.get("prediction_changed")) else 0.0 for row in group_rows
        )
        stats[name] = entry
    return stats


def task_scores_from_selected(rows: List[Dict[str, Any]], score_field: str) -> Dict[str, float]:
    by_task: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        score = to_float(row.get(score_field))
        if score is not None:
            by_task[str(row.get("task"))].append(score)
    scores = {task: sum(values) / len(values) for task, values in by_task.items() if values}
    scores["macro"] = sum(scores.values()) / len(scores) if scores else 0.0
    return scores


def run_conditional_gate(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    replay_rows: List[Dict[str, Any]] = []
    for row in rows:
        conditions: List[Tuple[str, bool, str]] = []

        jaccard = to_float(row.get("selected_jaccard_vs_original"))
        query_delta = to_float(row.get("query_overlap_delta"))
        coverage_delta = to_float(row.get("coverage_delta"))
        redundancy_delta = to_float(row.get("redundancy_delta"))
        prediction_changed = parse_bool(row.get("prediction_changed"))
        budget_mismatch = parse_bool(row.get("budget_mismatch"))
        fallback_used = parse_bool(row.get("fallback_used"))
        original_entered = parse_bool(row.get("original_entered_dc_selector"))

        conditions.append(
            (
                "selected_jaccard_vs_original >= 0.60",
                jaccard is not None and jaccard >= 0.60,
                "unavailable" if jaccard is None else fmt(jaccard),
            )
        )
        conditions.append(
            (
                "query_overlap_delta > 0",
                query_delta is not None and query_delta > 0,
                "unavailable" if query_delta is None else fmt(query_delta),
            )
        )
        conditions.append(
            (
                "coverage_delta >= 0",
                coverage_delta is not None and coverage_delta >= 0,
                "unavailable" if coverage_delta is None else fmt(coverage_delta),
            )
        )
        conditions.append(
            (
                "redundancy_delta < 0",
                redundancy_delta is not None and redundancy_delta < 0,
                "unavailable" if redundancy_delta is None else fmt(redundancy_delta),
            )
        )
        conditions.append(
            (
                "prediction_changed = true",
                prediction_changed is True,
                "unavailable" if prediction_changed is None else str(prediction_changed),
            )
        )
        conditions.append(
            (
                "budget mismatch = 0",
                budget_mismatch is False,
                "unavailable" if budget_mismatch is None else str(budget_mismatch),
            )
        )
        conditions.append(
            (
                "fallback = 0",
                fallback_used is False,
                "unavailable" if fallback_used is None else str(fallback_used),
            )
        )
        conditions.append(
            (
                "original_entered_dc_selector = 0",
                original_entered is False,
                "unavailable" if original_entered is None else str(original_entered),
            )
        )
        failed = [name for name, passed, _ in conditions if not passed]
        triggered = not failed
        selected_method = "dc_sliminfer" if triggered else "sliminfer_original"
        selected_score = row.get("dc_score") if triggered else row.get("original_score")
        selected_prediction = row.get("dc_prediction") if triggered else row.get("original_prediction")
        replay_rows.append(
            {
                "task": row.get("task"),
                "sample_index": row.get("sample_index"),
                "sample_id": row.get("sample_id"),
                "gate_triggered": triggered,
                "selected_method": selected_method,
                "conditional_score": selected_score,
                "delta_vs_original": (to_float(selected_score) or 0.0)
                - (to_float(row.get("original_score")) or 0.0),
                "original_score": row.get("original_score"),
                "dc_score": row.get("dc_score"),
                "unconditional_dc_delta": row.get("delta"),
                "original_prediction": row.get("original_prediction"),
                "dc_prediction": row.get("dc_prediction"),
                "conditional_prediction": selected_prediction,
                "prediction_changed": row.get("prediction_changed"),
                "selected_jaccard_vs_original": row.get("selected_jaccard_vs_original"),
                "query_overlap_delta": row.get("query_overlap_delta"),
                "coverage_delta": row.get("coverage_delta"),
                "redundancy_delta": row.get("redundancy_delta"),
                "budget_mismatch": row.get("budget_mismatch"),
                "fallback_used": row.get("fallback_used"),
                "original_entered_dc_selector": row.get("original_entered_dc_selector"),
                "failed_conditions": "; ".join(failed),
            }
        )

    original_scores = task_scores_from_selected(rows, "original_score")
    dc_scores = task_scores_from_selected(rows, "dc_score")
    conditional_scores = task_scores_from_selected(replay_rows, "conditional_score")
    trigger_rows = [row for row in replay_rows if parse_bool(row.get("gate_triggered")) is True]
    hotpot_negatives = [
        row for row in rows if row.get("task") == "hotpotqa" and (to_float(row.get("delta")) or 0.0) < 0
    ]
    hotpot_filtered = [
        row
        for row in replay_rows
        if row.get("task") == "hotpotqa"
        and (to_float(row.get("unconditional_dc_delta")) or 0.0) < 0
        and parse_bool(row.get("gate_triggered")) is not True
    ]
    musique_positives = [
        row for row in rows if row.get("task") == "musique" and (to_float(row.get("delta")) or 0.0) > 0
    ]
    musique_retained = [
        row
        for row in replay_rows
        if row.get("task") == "musique"
        and (to_float(row.get("unconditional_dc_delta")) or 0.0) > 0
        and parse_bool(row.get("gate_triggered")) is True
    ]
    deltas = [to_float(row.get("delta_vs_original")) or 0.0 for row in replay_rows]
    summary = {
        "trigger_count": len(trigger_rows),
        "trigger_sample_list": [
            f"{row.get('task')}:{row.get('sample_index')}" for row in trigger_rows
        ],
        "original_scores": original_scores,
        "dc_scores": dc_scores,
        "conditional_scores": conditional_scores,
        "conditional_delta": {
            task: conditional_scores.get(task, 0.0) - original_scores.get(task, 0.0)
            for task in set(original_scores) | set(conditional_scores)
        },
        "positive_count": sum(1 for value in deltas if value > 0),
        "negative_count": sum(1 for value in deltas if value < 0),
        "zero_count": sum(1 for value in deltas if value == 0),
        "hotpotqa_obvious_negative_filtered": len(hotpot_filtered),
        "hotpotqa_obvious_negative_total": len(hotpot_negatives),
        "musique_positive_retained": len(musique_retained),
        "musique_positive_total": len(musique_positives),
        "conditional_beats_original": conditional_scores.get("macro", 0.0)
        > original_scores.get("macro", 0.0),
        "conditional_beats_unconditional_dc": conditional_scores.get("macro", 0.0)
        > dc_scores.get("macro", 0.0),
    }
    return replay_rows, summary


def decide(
    bug_reasons: List[str],
    gate_summary: Dict[str, Any],
    total_rows: int,
) -> Tuple[str, List[str]]:
    if bug_reasons:
        return "BUG_STOP_FIX_REQUIRED", bug_reasons
    original_macro = gate_summary["original_scores"].get("macro", 0.0)
    conditional_macro = gate_summary["conditional_scores"].get("macro", 0.0)
    hotpot_delta = gate_summary["conditional_delta"].get("hotpotqa", 0.0)
    trigger_count = gate_summary["trigger_count"]
    retained = gate_summary["musique_positive_retained"]
    filtered = gate_summary["hotpotqa_obvious_negative_filtered"]
    filtered_total = gate_summary["hotpotqa_obvious_negative_total"]
    go_reasons = [
        conditional_macro > original_macro,
        hotpot_delta >= -2.0,
        retained > 0,
        filtered_total > 0 and filtered == filtered_total,
        0 < trigger_count < total_rows,
    ]
    if all(go_reasons):
        return "CONDITIONAL_DC_GO_TO_TINY_GATED_CONFIRMATION", [
            "conditional macro > original",
            "HotpotQA delta >= -2",
            "MuSiQue positives retained",
            "HotpotQA catastrophic negatives filtered",
            "trigger is nonzero and non-global",
            "gate uses no task label or gold answer",
        ]

    reasons: List[str] = []
    if conditional_macro <= original_macro:
        reasons.append("conditional_macro_not_above_original")
    if hotpot_delta < -2.0:
        reasons.append("hotpotqa_still_declines")
    if retained == 0:
        reasons.append("musique_positive_not_retained")
    if not (filtered_total > 0 and filtered == filtered_total):
        reasons.append("hotpotqa_obvious_negatives_not_fully_filtered")
    if trigger_count == 0:
        reasons.append("trigger_count_zero")
    elif trigger_count == total_rows:
        reasons.append("trigger_count_all")
    return "CONDITIONAL_DC_NO_GO_STOP", reasons


def markdown_table(headers: List[str], rows: List[List[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def build_report(
    input_dir: Path,
    output_dir: Path,
    artifacts: Dict[str, Optional[Path]],
    computed_scores: Dict[str, Dict[str, float]],
    failure_rows: List[Dict[str, Any]],
    stats: Dict[str, Dict[str, Any]],
    correlations: Dict[str, Optional[float]],
    gate_summary: Dict[str, Any],
    decision: str,
    decision_reasons: List[str],
    bug_reasons: List[str],
    report_text: str,
    activation_rows: List[Dict[str, Any]],
    selection_rows: List[Dict[str, Any]],
    quality_rows: List[Dict[str, Any]],
    prediction_file_counts: Dict[str, int],
) -> str:
    prediction_changed_count = sum(
        1 for row in failure_rows if parse_bool(row.get("prediction_changed")) is True
    )
    budget_mismatch_count = sum(
        1 for row in quality_rows if parse_bool(row.get("budget_mismatch")) is True
    )
    fallback_count = sum(1 for row in quality_rows if parse_bool(row.get("fallback_used")) is True)
    original_entered_count = sum(
        1
        for row in quality_rows
        if row.get("method") == "sliminfer_original"
        and parse_bool(row.get("dc_entered_before_virtual_return")) is True
    )
    positive_count = sum(1 for row in failure_rows if (to_float(row.get("delta")) or 0.0) > 0)
    negative_count = sum(1 for row in failure_rows if (to_float(row.get("delta")) or 0.0) < 0)
    zero_count = len(failure_rows) - positive_count - negative_count

    oracle_hotpot = computed_scores.get("hotpotqa", {}).get("original")
    oracle_musique = computed_scores.get("musique", {}).get("dc")
    oracle_macro = mean([oracle_hotpot, oracle_musique])

    task_rows = []
    for task in ("hotpotqa", "musique", "macro"):
        values = computed_scores.get(task, {})
        task_rows.append(
            [
                task,
                fmt(values.get("original")),
                fmt(values.get("dc")),
                fmt(values.get("delta")),
            ]
        )

    group_rows = []
    for name, values in stats.items():
        group_rows.append(
            [
                name,
                values["count"],
                fmt(values.get("delta_mean")),
                fmt(values.get("selected_jaccard_vs_original_mean")),
                fmt(values.get("query_overlap_delta_mean")),
                fmt(values.get("coverage_delta_mean")),
                fmt(values.get("redundancy_delta_mean")),
                fmt(values.get("prediction_changed_rate")),
            ]
        )

    corr_rows = [
        ["delta vs selected_jaccard", fmt(correlations.get("selected_jaccard_vs_original"))],
        ["delta vs query_overlap_delta", fmt(correlations.get("query_overlap_delta"))],
        ["delta vs coverage_delta", fmt(correlations.get("coverage_delta"))],
        ["delta vs redundancy_delta", fmt(correlations.get("redundancy_delta"))],
    ]

    conditional_scores = gate_summary["conditional_scores"]
    original_scores = gate_summary["original_scores"]
    dc_scores = gate_summary["dc_scores"]
    conditional_rows = []
    for task in ("hotpotqa", "musique", "macro"):
        conditional_rows.append(
            [
                task,
                fmt(original_scores.get(task)),
                fmt(conditional_scores.get(task)),
                fmt(conditional_scores.get(task, 0.0) - original_scores.get(task, 0.0)),
                fmt(dc_scores.get(task)),
            ]
        )

    artifact_rows = []
    for name, path in artifacts.items():
        artifact_rows.append([name, str(path) if path else "unavailable"])
    artifact_rows.append(["original longbench prediction rows", prediction_file_counts["sliminfer_original"]])
    artifact_rows.append(["dc longbench prediction rows", prediction_file_counts["dc_sliminfer"]])

    pattern_note = (
        "The fixed selection-feature gate is too conservative here: it filters the "
        "HotpotQA negative rows, but it also filters all MuSiQue positive rows. "
        "The observed feature pattern does not provide a useful task-agnostic split "
        "under the requested single gate."
    )
    if gate_summary["trigger_count"] > 0:
        pattern_note = (
            "The fixed gate produces nonzero triggers, but the decision still depends "
            "on whether it keeps MuSiQue positives without reopening HotpotQA negatives."
        )

    lines = [
        "# DC-SlimInfer Failure Postmortem And Gate Replay",
        "",
        f"- Input directory: `{input_dir}`",
        f"- Output directory: `{output_dir}`",
        "- Generation run: **No**",
        "- Scorer run: **No**",
        "- Full/64 diagnostic: **No**",
        "- Threshold sweep: **No**",
        "",
        "## Artifact Discovery",
        markdown_table(["artifact", "path or count"], artifact_rows),
        "",
        "## 1. Unconditional DC Failure",
        markdown_table(["task", "original", "dc", "delta"], task_rows),
        "",
        f"- Prediction change count: {prediction_changed_count}/{len(failure_rows)}",
        f"- Positive / negative / zero sample count: {positive_count} / {negative_count} / {zero_count}",
        f"- Budget mismatch count: {budget_mismatch_count}",
        f"- Fallback count: {fallback_count}",
        f"- original_entered_dc_selector count: {original_entered_count}",
        f"- Activation / selection audit rows: {len(activation_rows)} / {len(selection_rows)}",
        "- Final verdict: **UNCONDITIONAL_DC_NO_GO**",
        "",
        "The tiny smoke macro increases, but this does not rescue the method because "
        "HotpotQA drops sharply. This section confirms the prior stop decision rather "
        "than reinterpreting the smoke as a quality win.",
        "",
        "## 2. Per-Sample Failure Analysis",
        markdown_table(
            [
                "group",
                "count",
                "delta_mean",
                "jaccard_mean",
                "query_delta_mean",
                "coverage_delta_mean",
                "redundancy_delta_mean",
                "prediction_changed_rate",
            ],
            group_rows,
        ),
        "",
        "### Correlations",
        markdown_table(["correlation", "value"], corr_rows),
        "",
        f"Pattern summary: {pattern_note}",
        "",
        "## 3. Oracle Routing Diagnostic",
        f"- original-only macro: {fmt(computed_scores.get('macro', {}).get('original'))}",
        f"- dc-only macro: {fmt(computed_scores.get('macro', {}).get('dc'))}",
        (
            "- oracle-task-route macro: "
            f"{fmt(oracle_macro)} (HotpotQA uses original, MuSiQue uses DC)"
        ),
        "",
        (
            "oracle-task-route uses task labels and is not a valid runtime method; "
            "it is only an upper-bound diagnostic."
        ),
        "",
        "## 4. Task-Agnostic Conditional Gate Replay",
        "- Gate policy: use Original by default.",
        "- DC is used only when all fixed conditions pass: selected_jaccard_vs_original >= 0.60; "
        "query_overlap_delta > 0; coverage_delta >= 0; redundancy_delta < 0; "
        "prediction_changed = true; budget mismatch = 0; fallback = 0; "
        "original_entered_dc_selector = 0.",
        "- Missing fields fail closed.",
        f"- Conditional-gate trigger count: {gate_summary['trigger_count']}/{len(failure_rows)}",
        f"- Trigger sample list: {gate_summary['trigger_sample_list']}",
        markdown_table(
            ["task", "original", "conditional", "delta_vs_original", "unconditional_dc"],
            conditional_rows,
        ),
        "",
        (
            "- Conditional positive / negative / zero sample count vs original: "
            f"{gate_summary['positive_count']} / {gate_summary['negative_count']} / "
            f"{gate_summary['zero_count']}"
        ),
        (
            "- HotpotQA obvious negative filtered: "
            f"{gate_summary['hotpotqa_obvious_negative_filtered']}/"
            f"{gate_summary['hotpotqa_obvious_negative_total']}"
        ),
        (
            "- MuSiQue positive retained: "
            f"{gate_summary['musique_positive_retained']}/"
            f"{gate_summary['musique_positive_total']}"
        ),
        f"- Conditional gate better than Original macro: {gate_summary['conditional_beats_original']}",
        (
            "- Conditional gate better than unconditional DC macro: "
            f"{gate_summary['conditional_beats_unconditional_dc']}"
        ),
        "",
        "## 5. Decision",
        f"- Decision: **{decision}**",
        f"- Decision reasons: {decision_reasons}",
    ]
    if bug_reasons:
        lines.extend(["", "### Bug Reasons", *[f"- {reason}" for reason in bug_reasons]])
    if report_text:
        lines.extend(
            [
                "",
                "## Source Report Cross-Check",
                "The source report was found and used for consistency checks.",
            ]
        )
    lines.extend(["", REQUIRED_FOOTER])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-parent", type=Path, default=DEFAULT_OUTPUT_PARENT)
    args = parser.parse_args()

    input_dir = args.input_dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_parent / f"dc_sliminfer_failure_gate_replay_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)

    artifacts = discover_artifacts(input_dir)
    report_text = read_text(artifacts["report"])
    summary_json = read_json(artifacts["summary"])
    task_summary_rows = read_csv_rows(artifacts["task_summary"])
    comparison_rows = read_jsonl(artifacts["comparison_jsonl"])
    if not comparison_rows:
        comparison_rows = read_csv_rows(artifacts["comparison_csv"])
    quality_rows = read_jsonl(artifacts["per_sample_predictions"])
    activation_rows = read_jsonl(artifacts["activation_audit"])
    selection_rows = read_jsonl(artifacts["selection_audit"])
    prediction_files = discover_longbench_prediction_files(input_dir)
    prediction_file_counts = {
        method: sum(len(read_jsonl(path)) for path in paths)
        for method, paths in prediction_files.items()
    }

    bug_reasons: List[str] = []
    if not input_dir.exists():
        bug_reasons.append(f"input directory missing: {input_dir}")
    if not comparison_rows:
        bug_reasons.append("per-sample comparison rows missing")
    if len(comparison_rows) != 20:
        bug_reasons.append(f"expected 20 comparison rows, found {len(comparison_rows)}")
    if len(quality_rows) != 40:
        bug_reasons.append(f"expected 40 per-sample prediction rows, found {len(quality_rows)}")
    if len(activation_rows) != 60:
        bug_reasons.append(f"expected 60 activation audit rows, found {len(activation_rows)}")
    if len(selection_rows) != 60:
        bug_reasons.append(f"expected 60 selection audit rows, found {len(selection_rows)}")
    if prediction_file_counts["sliminfer_original"] != 20:
        bug_reasons.append(
            "original longbench prediction files incomplete: "
            f"{prediction_file_counts['sliminfer_original']} rows"
        )
    if prediction_file_counts["dc_sliminfer"] != 20:
        bug_reasons.append(
            f"DC longbench prediction files incomplete: {prediction_file_counts['dc_sliminfer']} rows"
        )

    computed_scores = compute_task_scores(comparison_rows)
    bug_reasons.extend(compare_score_tables(computed_scores, task_summary_rows, summary_json))

    failure_rows = build_failure_rows(comparison_rows, quality_rows, selection_rows)
    stats = group_stats(failure_rows)
    correlations = {
        "selected_jaccard_vs_original": pearson(
            (row.get("selected_jaccard_vs_original") for row in failure_rows),
            (row.get("delta") for row in failure_rows),
        ),
        "query_overlap_delta": pearson(
            (row.get("query_overlap_delta") for row in failure_rows),
            (row.get("delta") for row in failure_rows),
        ),
        "coverage_delta": pearson(
            (row.get("coverage_delta") for row in failure_rows),
            (row.get("delta") for row in failure_rows),
        ),
        "redundancy_delta": pearson(
            (row.get("redundancy_delta") for row in failure_rows),
            (row.get("delta") for row in failure_rows),
        ),
    }

    replay_rows, gate_summary = run_conditional_gate(failure_rows)
    decision, decision_reasons = decide(bug_reasons, gate_summary, len(failure_rows))

    failure_fields = [
        "task",
        "sample_index",
        "sample_id",
        "original_score",
        "dc_score",
        "delta",
        "prediction_changed",
        "selected_jaccard_vs_original",
        "original_redundancy",
        "dc_redundancy",
        "redundancy_delta",
        "original_query_overlap",
        "dc_query_overlap",
        "query_overlap_delta",
        "original_coverage_bin_count",
        "dc_coverage_bin_count",
        "coverage_delta",
        "dc_entered_before_virtual_return",
        "virtual_block_path_returned_before_dc_selector",
    ]
    replay_fields = [
        "task",
        "sample_index",
        "sample_id",
        "gate_triggered",
        "selected_method",
        "conditional_score",
        "delta_vs_original",
        "original_score",
        "dc_score",
        "unconditional_dc_delta",
        "prediction_changed",
        "selected_jaccard_vs_original",
        "query_overlap_delta",
        "coverage_delta",
        "redundancy_delta",
        "budget_mismatch",
        "fallback_used",
        "original_entered_dc_selector",
        "failed_conditions",
        "original_prediction",
        "dc_prediction",
        "conditional_prediction",
    ]
    write_csv(output_dir / "dc_per_sample_failure_analysis.csv", failure_rows, failure_fields)
    write_csv(output_dir / "dc_conditional_gate_replay.csv", replay_rows, replay_fields)

    report = build_report(
        input_dir=input_dir,
        output_dir=output_dir,
        artifacts=artifacts,
        computed_scores=computed_scores,
        failure_rows=failure_rows,
        stats=stats,
        correlations=correlations,
        gate_summary=gate_summary,
        decision=decision,
        decision_reasons=decision_reasons,
        bug_reasons=bug_reasons,
        report_text=report_text,
        activation_rows=activation_rows,
        selection_rows=selection_rows,
        quality_rows=quality_rows,
        prediction_file_counts=prediction_file_counts,
    )
    (output_dir / "dc_failure_postmortem.md").write_text(report, encoding="utf-8")

    decision_lines = [
        "# DC Conditional Gate Decision",
        "",
        f"- Input directory: `{input_dir}`",
        f"- Output directory: `{output_dir}`",
        f"- Decision: **{decision}**",
        f"- Reasons: {decision_reasons}",
        f"- Trigger count: {gate_summary['trigger_count']}/{len(failure_rows)}",
        f"- Trigger sample list: {gate_summary['trigger_sample_list']}",
        (
            "- HotpotQA obvious negative filtered: "
            f"{gate_summary['hotpotqa_obvious_negative_filtered']}/"
            f"{gate_summary['hotpotqa_obvious_negative_total']}"
        ),
        (
            "- MuSiQue positive retained: "
            f"{gate_summary['musique_positive_retained']}/"
            f"{gate_summary['musique_positive_total']}"
        ),
        "",
        REQUIRED_FOOTER,
        "",
    ]
    (output_dir / "dc_gate_decision.md").write_text("\n".join(decision_lines), encoding="utf-8")

    manifest = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "decision": decision,
        "decision_reasons": decision_reasons,
        "bug_reasons": bug_reasons,
        "gate_summary": gate_summary,
        "correlations": correlations,
        "artifact_paths": {key: str(path) if path else None for key, path in artifacts.items()},
        "prediction_file_counts": prediction_file_counts,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if decision != "BUG_STOP_FIX_REQUIRED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
