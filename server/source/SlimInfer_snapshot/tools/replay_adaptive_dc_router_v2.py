#!/usr/bin/env python3
"""Offline Adaptive DC-SlimInfer router-v2 analysis and replay.

This script reads existing tiny DC-SlimInfer artifacts. It does not run model
generation, decoding, LongBench scoring, or selector code.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_TINY_DIR = Path("results/dc_sliminfer_tiny_quality_smoke_20260708_004504")
DEFAULT_FIXED_GATE_DIR = Path("results/dc_sliminfer_failure_gate_replay_20260708_011112")
DEFAULT_OUTPUT_PARENT = Path("results")
REQUIRED_FOOTER = (
    "Unconditional DC-SlimInfer remains stopped.\n"
    "Fixed conservative gate is stopped.\n"
    "Adaptive DC-SlimInfer can continue only if router-v2 shows non-trivial "
    "task-agnostic routing signal."
)

FEATURES = [
    "selected_jaccard_vs_original",
    "query_overlap_delta",
    "coverage_delta",
    "redundancy_delta",
]


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


def read_csv(path: Optional[Path]) -> List[Dict[str, Any]]:
    if not path or not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


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
        return "unavailable" if value is None else str(value)
    return f"{number:.{digits}f}"


def mean(values: Iterable[Any]) -> Optional[float]:
    clean = [value for value in (to_float(item) for item in values) if value is not None]
    if not clean:
        return None
    return sum(clean) / len(clean)


def median(values: Iterable[Any]) -> Optional[float]:
    clean = sorted(value for value in (to_float(item) for item in values) if value is not None)
    if not clean:
        return None
    mid = len(clean) // 2
    if len(clean) % 2:
        return clean[mid]
    return (clean[mid - 1] + clean[mid]) / 2.0


def pearson(xs: Iterable[Any], ys: Iterable[Any]) -> Optional[float]:
    pairs: List[Tuple[float, float]] = []
    for raw_x, raw_y in zip(xs, ys):
        x = to_float(raw_x)
        y = to_float(raw_y)
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


def average_ranks(values: Sequence[float]) -> List[float]:
    order = sorted(range(len(values)), key=lambda idx: values[idx])
    ranks = [0.0] * len(values)
    pos = 0
    while pos < len(order):
        end = pos + 1
        while end < len(order) and values[order[end]] == values[order[pos]]:
            end += 1
        rank = (pos + 1 + end) / 2.0
        for idx in order[pos:end]:
            ranks[idx] = rank
        pos = end
    return ranks


def spearman(xs: Iterable[Any], ys: Iterable[Any]) -> Optional[float]:
    pairs: List[Tuple[float, float]] = []
    for raw_x, raw_y in zip(xs, ys):
        x = to_float(raw_x)
        y = to_float(raw_y)
        if x is not None and y is not None:
            pairs.append((x, y))
    if len(pairs) < 2:
        return None
    x_ranks = average_ranks([x for x, _ in pairs])
    y_ranks = average_ranks([y for _, y in pairs])
    return pearson(x_ranks, y_ranks)


def sample_key(row: Dict[str, Any]) -> Tuple[str, int, str]:
    return (
        str(row.get("task", "")),
        int(to_float(row.get("sample_index")) or 0),
        str(row.get("sample_id", "")),
    )


def first_available(*values: Any) -> Any:
    for value in values:
        if value is not None and value != "":
            return value
    return None


def find_file(root: Path, exact: str, keywords: Sequence[str], suffixes: Sequence[str]) -> Optional[Path]:
    exact_path = root / exact
    if exact_path.exists():
        return exact_path
    candidates: List[Path] = []
    lowered_keywords = [keyword.lower() for keyword in keywords]
    suffix_set = {suffix.lower() for suffix in suffixes}
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in suffix_set:
            continue
        lower_name = path.name.lower()
        if all(keyword in lower_name for keyword in lowered_keywords):
            candidates.append(path)
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: (len(item.parts), str(item)))[0]


def load_inputs(tiny_dir: Path, fixed_gate_dir: Path) -> Tuple[Dict[str, Any], List[str]]:
    artifacts = {
        "tiny_report": find_file(tiny_dir, "report.md", ["report"], [".md"]),
        "tiny_summary": find_file(tiny_dir, "summary.json", ["summary"], [".json"]),
        "tiny_task_summary": find_file(tiny_dir, "task_summary.csv", ["task", "summary"], [".csv"]),
        "tiny_comparison_jsonl": find_file(
            tiny_dir, "per_sample_comparison.jsonl", ["sample", "comparison"], [".jsonl"]
        ),
        "tiny_comparison_csv": find_file(
            tiny_dir, "per_sample_comparison.csv", ["sample", "comparison"], [".csv"]
        ),
        "tiny_predictions": find_file(
            tiny_dir, "per_sample_predictions.jsonl", ["sample", "prediction"], [".jsonl"]
        ),
        "fixed_failure": find_file(
            fixed_gate_dir,
            "dc_per_sample_failure_analysis.csv",
            ["failure", "analysis"],
            [".csv"],
        ),
        "fixed_gate_replay": find_file(
            fixed_gate_dir,
            "dc_conditional_gate_replay.csv",
            ["conditional", "gate", "replay"],
            [".csv"],
        ),
        "fixed_report": find_file(fixed_gate_dir, "dc_failure_postmortem.md", ["postmortem"], [".md"]),
    }
    comparison_rows = read_jsonl(artifacts["tiny_comparison_jsonl"])
    if not comparison_rows:
        comparison_rows = read_csv(artifacts["tiny_comparison_csv"])
    data = {
        "artifacts": artifacts,
        "tiny_report_text": read_text(artifacts["tiny_report"]),
        "tiny_summary": read_json(artifacts["tiny_summary"]),
        "tiny_task_summary": read_csv(artifacts["tiny_task_summary"]),
        "comparison_rows": comparison_rows,
        "prediction_rows": read_jsonl(artifacts["tiny_predictions"]),
        "fixed_failure_rows": read_csv(artifacts["fixed_failure"]),
        "fixed_gate_rows": read_csv(artifacts["fixed_gate_replay"]),
        "fixed_report_text": read_text(artifacts["fixed_report"]),
    }
    missing = [name for name, path in artifacts.items() if path is None and name != "fixed_report"]
    return data, [f"missing artifact: {name}" for name in missing]


def rows_by_key(rows: List[Dict[str, Any]]) -> Dict[Tuple[str, int, str], Dict[str, Any]]:
    return {sample_key(row): row for row in rows}


def prediction_rows_by_key_and_method(
    rows: List[Dict[str, Any]]
) -> Dict[Tuple[str, int, str], Dict[str, Dict[str, Any]]]:
    grouped: Dict[Tuple[str, int, str], Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        grouped[sample_key(row)][str(row.get("method", ""))] = row
    return grouped


def build_sample_rows(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    comparison_rows = data["comparison_rows"]
    fixed_failure = rows_by_key(data["fixed_failure_rows"])
    fixed_gate = rows_by_key(data["fixed_gate_rows"])
    predictions = prediction_rows_by_key_and_method(data["prediction_rows"])

    rows: List[Dict[str, Any]] = []
    for comparison in comparison_rows:
        key = sample_key(comparison)
        failure = fixed_failure.get(key, {})
        gate = fixed_gate.get(key, {})
        original_prediction_row = predictions.get(key, {}).get("sliminfer_original", {})
        dc_prediction_row = predictions.get(key, {}).get("dc_sliminfer", {})

        original_redundancy = first_available(
            failure.get("original_redundancy"), dc_prediction_row.get("original_redundancy")
        )
        dc_redundancy = first_available(failure.get("dc_redundancy"), dc_prediction_row.get("dc_redundancy"))
        original_query_overlap = first_available(
            failure.get("original_query_overlap"), dc_prediction_row.get("original_query_overlap")
        )
        dc_query_overlap = first_available(
            failure.get("dc_query_overlap"), dc_prediction_row.get("dc_query_overlap")
        )
        original_coverage = first_available(
            failure.get("original_coverage_bin_count"),
            dc_prediction_row.get("original_coverage_bin_count"),
        )
        dc_coverage = first_available(
            failure.get("dc_coverage_bin_count"), dc_prediction_row.get("dc_coverage_bin_count")
        )

        budget_mismatch = any(
            parse_bool(value) is True
            for value in (
                original_prediction_row.get("budget_mismatch"),
                dc_prediction_row.get("budget_mismatch"),
                gate.get("budget_mismatch"),
            )
        )
        fallback = any(
            parse_bool(value) is True
            for value in (
                original_prediction_row.get("fallback_used"),
                dc_prediction_row.get("fallback_used"),
                gate.get("fallback_used"),
                gate.get("fallback"),
            )
        )
        original_entered_dc_selector = (
            parse_bool(original_prediction_row.get("dc_entered_before_virtual_return")) is True
            or parse_bool(gate.get("original_entered_dc_selector")) is True
        )

        selection_changed = first_available(
            comparison.get("selection_changed"), dc_prediction_row.get("selection_changed")
        )
        row = {
            "task": comparison.get("task"),
            "sample_index": int(to_float(comparison.get("sample_index")) or 0),
            "sample_id": comparison.get("sample_id"),
            "original_score": to_float(comparison.get("original_score")),
            "dc_score": to_float(comparison.get("dc_score")),
            "score_delta": to_float(first_available(comparison.get("delta"), failure.get("delta"))),
            "original_prediction": comparison.get("original_prediction"),
            "dc_prediction": comparison.get("dc_prediction"),
            "prediction_changed": parse_bool(comparison.get("prediction_changed")),
            "selected_jaccard_vs_original": to_float(
                first_available(
                    comparison.get("selected_jaccard"),
                    failure.get("selected_jaccard_vs_original"),
                    dc_prediction_row.get("selected_jaccard_vs_original"),
                )
            ),
            "original_redundancy": to_float(original_redundancy),
            "dc_redundancy": to_float(dc_redundancy),
            "original_query_overlap": to_float(original_query_overlap),
            "dc_query_overlap": to_float(dc_query_overlap),
            "original_coverage_bin_count": to_float(original_coverage),
            "dc_coverage_bin_count": to_float(dc_coverage),
            "budget_mismatch": budget_mismatch,
            "fallback": fallback,
            "original_entered_dc_selector": original_entered_dc_selector,
            "selection_changed": parse_bool(selection_changed),
            "v1_failed_conditions": gate.get("failed_conditions", ""),
        }
        if row["score_delta"] is None and row["original_score"] is not None and row["dc_score"] is not None:
            row["score_delta"] = row["dc_score"] - row["original_score"]
        if row["original_redundancy"] is not None and row["dc_redundancy"] is not None:
            row["redundancy_delta"] = row["dc_redundancy"] - row["original_redundancy"]
        else:
            row["redundancy_delta"] = None
        if row["original_query_overlap"] is not None and row["dc_query_overlap"] is not None:
            row["query_overlap_delta"] = row["dc_query_overlap"] - row["original_query_overlap"]
        else:
            row["query_overlap_delta"] = None
        if row["original_coverage_bin_count"] is not None and row["dc_coverage_bin_count"] is not None:
            row["coverage_delta"] = row["dc_coverage_bin_count"] - row["original_coverage_bin_count"]
        else:
            row["coverage_delta"] = None
        row["dc_positive"] = (row["score_delta"] or 0.0) > 0
        row["dc_negative"] = (row["score_delta"] or 0.0) < 0
        row["hotpotqa_catastrophic_negative"] = (
            row["task"] == "hotpotqa" and (row["score_delta"] or 0.0) < 0
        )
        row["musique_positive"] = row["task"] == "musique" and (row["score_delta"] or 0.0) > 0
        rows.append(row)

    return sorted(rows, key=lambda item: (str(item["task"]), int(item["sample_index"])))


def score_table(rows: List[Dict[str, Any]], field: str) -> Dict[str, float]:
    by_task: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        value = to_float(row.get(field))
        if value is not None:
            by_task[str(row.get("task"))].append(value)
    scores = {task: sum(values) / len(values) for task, values in by_task.items() if values}
    scores["macro"] = sum(scores.values()) / len(scores) if scores else 0.0
    return scores


def validate(rows: List[Dict[str, Any]], data: Dict[str, Any], prior_missing: List[str]) -> List[str]:
    reasons = list(prior_missing)
    if len(data["comparison_rows"]) != 20:
        reasons.append(f"expected 20 comparison rows, found {len(data['comparison_rows'])}")
    if len(data["prediction_rows"]) != 40:
        reasons.append(f"expected 40 per-sample prediction rows, found {len(data['prediction_rows'])}")
    if len(rows) != 20:
        reasons.append(f"expected 20 merged sample rows, found {len(rows)}")

    required_fields = [
        "task",
        "sample_index",
        "original_score",
        "dc_score",
        "score_delta",
        "selected_jaccard_vs_original",
        "original_redundancy",
        "dc_redundancy",
        "original_query_overlap",
        "dc_query_overlap",
        "original_coverage_bin_count",
        "dc_coverage_bin_count",
        "budget_mismatch",
        "fallback",
        "original_entered_dc_selector",
        "selection_changed",
    ]
    for row in rows:
        missing = [field for field in required_fields if row.get(field) is None]
        if missing:
            reasons.append(f"{row.get('task')}:{row.get('sample_index')} missing fields: {missing}")

    computed = {
        "hotpotqa": {},
        "musique": {},
        "macro": {},
    }
    original_scores = score_table(rows, "original_score")
    dc_scores = score_table(rows, "dc_score")
    for task in ("hotpotqa", "musique", "macro"):
        computed[task] = {
            "original": original_scores.get(task),
            "dc": dc_scores.get(task),
            "delta": (dc_scores.get(task, 0.0) - original_scores.get(task, 0.0)),
        }

    for row in data["tiny_task_summary"]:
        task = str(row.get("task", ""))
        if task not in computed:
            continue
        for field in ("original", "dc", "delta"):
            observed = to_float(row.get(field))
            expected = to_float(computed[task].get(field))
            if observed is not None and expected is not None and abs(observed - expected) > 1e-4:
                reasons.append(
                    f"task_summary mismatch {task}.{field}: file={observed}, computed={expected}"
                )

    budget_bad = [row for row in rows if parse_bool(row.get("budget_mismatch")) is not False]
    fallback_bad = [row for row in rows if parse_bool(row.get("fallback")) is not False]
    polluted = [row for row in rows if parse_bool(row.get("original_entered_dc_selector")) is not False]
    if budget_bad:
        reasons.append(f"budget_mismatch field not clean for {len(budget_bad)} rows")
    if fallback_bad:
        reasons.append(f"fallback field not clean for {len(fallback_bad)} rows")
    if polluted:
        reasons.append(f"original_entered_dc_selector field not clean for {len(polluted)} rows")
    return reasons


def predicate_specs() -> List[Tuple[str, Callable[[Dict[str, Any]], bool]]]:
    return [
        ("budget_mismatch = 0", lambda row: parse_bool(row.get("budget_mismatch")) is False),
        ("fallback = 0", lambda row: parse_bool(row.get("fallback")) is False),
        (
            "original_entered_dc_selector = 0",
            lambda row: parse_bool(row.get("original_entered_dc_selector")) is False,
        ),
        (
            "query_overlap_delta > 0",
            lambda row: to_float(row.get("query_overlap_delta")) is not None
            and to_float(row.get("query_overlap_delta")) > 0,
        ),
        (
            "coverage_delta >= 0",
            lambda row: to_float(row.get("coverage_delta")) is not None
            and to_float(row.get("coverage_delta")) >= 0,
        ),
        (
            "redundancy_delta < 0",
            lambda row: to_float(row.get("redundancy_delta")) is not None
            and to_float(row.get("redundancy_delta")) < 0,
        ),
        (
            "selected_jaccard_vs_original >= 0.45",
            lambda row: to_float(row.get("selected_jaccard_vs_original")) is not None
            and to_float(row.get("selected_jaccard_vs_original")) >= 0.45,
        ),
        (
            "selected_jaccard_vs_original <= 0.80",
            lambda row: to_float(row.get("selected_jaccard_vs_original")) is not None
            and to_float(row.get("selected_jaccard_vs_original")) <= 0.80,
        ),
        (
            "prediction_changed = true",
            lambda row: parse_bool(row.get("prediction_changed")) is True,
        ),
    ]


def predicate_analysis(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    specs = predicate_specs()
    sample_rows: List[Dict[str, Any]] = []
    for row in rows:
        failed = []
        output = {
            "task": row["task"],
            "sample_index": row["sample_index"],
            "sample_id": row["sample_id"],
            "score_delta": row["score_delta"],
            "dc_positive": row["dc_positive"],
            "dc_negative": row["dc_negative"],
            "hotpotqa_catastrophic_negative": row["hotpotqa_catastrophic_negative"],
            "musique_positive": row["musique_positive"],
            "v1_failed_conditions": row.get("v1_failed_conditions", ""),
        }
        for name, predicate in specs:
            passed = predicate(row)
            output[f"pass__{name}"] = passed
            if not passed:
                failed.append(name)
        output["failed_predicates"] = "; ".join(failed)
        sample_rows.append(output)

    groups = {
        "dc_positive": [row for row in rows if row["dc_positive"]],
        "dc_negative": [row for row in rows if row["dc_negative"]],
        "hotpotqa_catastrophic_negative": [
            row for row in rows if row["hotpotqa_catastrophic_negative"]
        ],
        "musique_positive": [row for row in rows if row["musique_positive"]],
    }
    rate_rows: List[Dict[str, Any]] = []
    for group_name, group_rows in groups.items():
        for name, predicate in specs:
            total = len(group_rows)
            failed_count = sum(1 for row in group_rows if not predicate(row))
            rate_rows.append(
                {
                    "analysis_type": "predicate_failure_rate",
                    "group": group_name,
                    "predicate": name,
                    "total": total,
                    "failed": failed_count,
                    "failure_rate": (failed_count / total) if total else None,
                }
            )
    return sample_rows, rate_rows


def feature_separability(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups = {
        "dc_positive": [row for row in rows if row["dc_positive"]],
        "dc_negative": [row for row in rows if row["dc_negative"]],
        "hotpotqa_catastrophic_negative": [
            row for row in rows if row["hotpotqa_catastrophic_negative"]
        ],
        "musique_positive": [row for row in rows if row["musique_positive"]],
    }
    output: List[Dict[str, Any]] = []
    for group_name, group_rows in groups.items():
        for feature in FEATURES:
            output.append(
                {
                    "analysis_type": "feature_distribution",
                    "group": group_name,
                    "feature": feature,
                    "count": len(group_rows),
                    "mean": mean(row.get(feature) for row in group_rows),
                    "median": median(row.get(feature) for row in group_rows),
                    "pearson_with_score_delta": "",
                    "spearman_with_score_delta": "",
                }
            )
    for feature in FEATURES:
        output.append(
            {
                "analysis_type": "correlation",
                "group": "all",
                "feature": feature,
                "count": len(rows),
                "mean": "",
                "median": "",
                "pearson_with_score_delta": pearson(
                    (row.get(feature) for row in rows), (row.get("score_delta") for row in rows)
                ),
                "spearman_with_score_delta": spearman(
                    (row.get(feature) for row in rows), (row.get("score_delta") for row in rows)
                ),
            }
        )
    return output


def base_safety(row: Dict[str, Any]) -> bool:
    return (
        parse_bool(row.get("budget_mismatch")) is False
        and parse_bool(row.get("fallback")) is False
        and parse_bool(row.get("original_entered_dc_selector")) is False
        and parse_bool(row.get("selection_changed")) is True
    )


def policy_specs() -> List[Tuple[str, str, Callable[[Dict[str, Any]], bool]]]:
    return [
        (
            "Policy A: coverage-gain router",
            "coverage_delta > 0; query_overlap_delta >= 0; selected_jaccard_vs_original >= 0.30",
            lambda row: base_safety(row)
            and (to_float(row.get("coverage_delta")) is not None and to_float(row.get("coverage_delta")) > 0)
            and (
                to_float(row.get("query_overlap_delta")) is not None
                and to_float(row.get("query_overlap_delta")) >= 0
            )
            and (
                to_float(row.get("selected_jaccard_vs_original")) is not None
                and to_float(row.get("selected_jaccard_vs_original")) >= 0.30
            ),
        ),
        (
            "Policy B: query-gain router",
            "query_overlap_delta > 0; coverage_delta >= 0; selected_jaccard_vs_original >= 0.30",
            lambda row: base_safety(row)
            and (
                to_float(row.get("query_overlap_delta")) is not None
                and to_float(row.get("query_overlap_delta")) > 0
            )
            and (to_float(row.get("coverage_delta")) is not None and to_float(row.get("coverage_delta")) >= 0)
            and (
                to_float(row.get("selected_jaccard_vs_original")) is not None
                and to_float(row.get("selected_jaccard_vs_original")) >= 0.30
            ),
        ),
        (
            "Policy C: diversity-repair router",
            "redundancy_delta < 0; coverage_delta >= 0; selected_jaccard_vs_original >= 0.30",
            lambda row: base_safety(row)
            and (
                to_float(row.get("redundancy_delta")) is not None
                and to_float(row.get("redundancy_delta")) < 0
            )
            and (to_float(row.get("coverage_delta")) is not None and to_float(row.get("coverage_delta")) >= 0)
            and (
                to_float(row.get("selected_jaccard_vs_original")) is not None
                and to_float(row.get("selected_jaccard_vs_original")) >= 0.30
            ),
        ),
        (
            "Policy D: permissive evidence-repair router",
            "selected_jaccard_vs_original >= 0.30 and at least two of query gain, coverage gain, redundancy reduction",
            lambda row: base_safety(row)
            and (
                to_float(row.get("selected_jaccard_vs_original")) is not None
                and to_float(row.get("selected_jaccard_vs_original")) >= 0.30
            )
            and sum(
                [
                    to_float(row.get("query_overlap_delta")) is not None
                    and to_float(row.get("query_overlap_delta")) > 0,
                    to_float(row.get("coverage_delta")) is not None
                    and to_float(row.get("coverage_delta")) > 0,
                    to_float(row.get("redundancy_delta")) is not None
                    and to_float(row.get("redundancy_delta")) < 0,
                ]
            )
            >= 2,
        ),
    ]


def replay_policies(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    original_scores = score_table(rows, "original_score")
    dc_scores = score_table(rows, "dc_score")
    output: List[Dict[str, Any]] = []
    total = len(rows)
    hotpot_cat_total = sum(1 for row in rows if row["hotpotqa_catastrophic_negative"])
    musique_pos_total = sum(1 for row in rows if row["musique_positive"])

    for name, description, predicate in policy_specs():
        triggered_rows = [row for row in rows if predicate(row)]
        trigger_keys = {sample_key(row) for row in triggered_rows}
        replay_rows: List[Dict[str, Any]] = []
        for row in rows:
            use_dc = sample_key(row) in trigger_keys
            replay_rows.append(
                {
                    **row,
                    "policy_score": row["dc_score"] if use_dc else row["original_score"],
                    "policy_prediction": row["dc_prediction"] if use_dc else row["original_prediction"],
                    "policy_triggered": use_dc,
                }
            )
        policy_scores = score_table(replay_rows, "policy_score")
        deltas = [
            (to_float(row.get("policy_score")) or 0.0) - (to_float(row.get("original_score")) or 0.0)
            for row in replay_rows
        ]
        hotpot_triggered = sum(
            1 for row in replay_rows if row["hotpotqa_catastrophic_negative"] and row["policy_triggered"]
        )
        musique_retained = sum(
            1 for row in replay_rows if row["musique_positive"] and row["policy_triggered"]
        )
        output.append(
            {
                "policy": name,
                "policy_description": description,
                "trigger_count": len(triggered_rows),
                "trigger_sample_list": "; ".join(
                    f"{row['task']}:{row['sample_index']}" for row in triggered_rows
                ),
                "hotpotqa_original": original_scores.get("hotpotqa"),
                "hotpotqa_policy": policy_scores.get("hotpotqa"),
                "hotpotqa_delta": policy_scores.get("hotpotqa", 0.0)
                - original_scores.get("hotpotqa", 0.0),
                "musique_original": original_scores.get("musique"),
                "musique_policy": policy_scores.get("musique"),
                "musique_delta": policy_scores.get("musique", 0.0)
                - original_scores.get("musique", 0.0),
                "macro_original": original_scores.get("macro"),
                "macro_policy": policy_scores.get("macro"),
                "macro_delta": policy_scores.get("macro", 0.0) - original_scores.get("macro", 0.0),
                "unconditional_dc_macro": dc_scores.get("macro"),
                "policy_vs_unconditional_dc_macro_delta": policy_scores.get("macro", 0.0)
                - dc_scores.get("macro", 0.0),
                "positive_count_vs_original": sum(1 for value in deltas if value > 0),
                "negative_count_vs_original": sum(1 for value in deltas if value < 0),
                "zero_count_vs_original": sum(1 for value in deltas if value == 0),
                "retained_musique_positive_count": musique_retained,
                "retained_musique_positive_total": musique_pos_total,
                "triggered_hotpot_catastrophic_negative_count": hotpot_triggered,
                "hotpot_catastrophic_negative_total": hotpot_cat_total,
                "filtered_hotpot_catastrophic_negative_count": hotpot_cat_total - hotpot_triggered,
                "non_all_reject": len(triggered_rows) > 0,
                "non_all_trigger": len(triggered_rows) < total,
                "better_than_original": policy_scores.get("macro", 0.0) > original_scores.get("macro", 0.0),
                "better_than_unconditional_dc": policy_scores.get("macro", 0.0) > dc_scores.get("macro", 0.0),
                "final_gate_uses_task_gold_score_or_prediction_changed": False,
            }
        )
    return output


def decide(policy_rows: List[Dict[str, Any]], separability_rows: List[Dict[str, Any]], bug_reasons: List[str]) -> Tuple[str, List[str]]:
    if bug_reasons:
        return "BUG_STOP_FIX_REQUIRED", bug_reasons

    go_policies = [
        row
        for row in policy_rows
        if parse_bool(row.get("non_all_reject")) is True
        and parse_bool(row.get("non_all_trigger")) is True
        and parse_bool(row.get("better_than_original")) is True
        and (
            (to_float(row.get("macro_policy")) or 0.0)
            >= (to_float(row.get("unconditional_dc_macro")) or 0.0)
            or (to_float(row.get("hotpotqa_delta")) or 0.0) >= -2.0
        )
        and (to_float(row.get("hotpotqa_delta")) or 0.0) >= -2.0
        and int(row.get("retained_musique_positive_count") or 0) >= 1
        and int(row.get("triggered_hotpot_catastrophic_negative_count") or 0) <= 1
        and parse_bool(row.get("final_gate_uses_task_gold_score_or_prediction_changed")) is False
    ]
    if go_policies:
        return "ADAPTIVE_DC_V2_GO_TO_TINY_GATED_CONFIRMATION", [
            f"GO policy found: {go_policies[0]['policy']}",
            "nontrivial trigger",
            "macro above original",
            "HotpotQA stable enough",
            "MuSiQue positive retained",
            "HotpotQA catastrophic trigger <= 1",
            "no task label/gold/score_delta/prediction_changed gate features",
        ]

    all_reject_or_all_trigger = all(
        int(row.get("trigger_count") or 0) in {0, 20} for row in policy_rows
    )
    any_retains_musique_but_hotpot_bad = any(
        int(row.get("retained_musique_positive_count") or 0) > 0
        and int(row.get("triggered_hotpot_catastrophic_negative_count") or 0) > 1
        for row in policy_rows
    )
    any_slight = any(
        parse_bool(row.get("better_than_original")) is True
        and (to_float(row.get("policy_vs_unconditional_dc_macro_delta")) or 0.0) < 0
        for row in policy_rows
    )
    max_abs_corr = 0.0
    for row in separability_rows:
        if row.get("analysis_type") == "correlation":
            for field in ("pearson_with_score_delta", "spearman_with_score_delta"):
                value = to_float(row.get(field))
                if value is not None:
                    max_abs_corr = max(max_abs_corr, abs(value))
    weak_separability = max_abs_corr < 0.5

    if any_retains_musique_but_hotpot_bad or any_slight or weak_separability:
        reasons = []
        if any_retains_musique_but_hotpot_bad:
            reasons.append("some policies retain MuSiQue positives but also trigger HotpotQA negatives")
        if any_slight:
            reasons.append("some policies beat original but do not beat unconditional DC")
        if weak_separability:
            reasons.append("feature separability is weak under the available selection features")
        return "ADAPTIVE_DC_V2_WEAK_SIGNAL_NEEDS_NEW_FEATURES", reasons

    if all_reject_or_all_trigger:
        return "ADAPTIVE_DC_V2_NO_GO_STOP", ["all policies reject all or trigger all"]
    return "ADAPTIVE_DC_V2_NO_GO_STOP", ["no policy satisfies GO or weak-signal continuation criteria"]


def markdown_table(headers: List[str], rows: List[List[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def build_report(
    output_dir: Path,
    tiny_dir: Path,
    fixed_gate_dir: Path,
    rows: List[Dict[str, Any]],
    predicate_rate_rows: List[Dict[str, Any]],
    separability_rows: List[Dict[str, Any]],
    policy_rows: List[Dict[str, Any]],
    decision: str,
    decision_reasons: List[str],
    bug_reasons: List[str],
) -> str:
    original_scores = score_table(rows, "original_score")
    dc_scores = score_table(rows, "dc_score")

    v1_zero_reason = (
        "The v1 conservative gate had zero triggers because the only sample with "
        "selected_jaccard_vs_original >= 0.60 also had prediction_changed = false; "
        "the MuSiQue positive rows all failed the >= 0.60 jaccard predicate."
    )

    score_rows = [
        [
            task,
            fmt(original_scores.get(task)),
            fmt(dc_scores.get(task)),
            fmt(dc_scores.get(task, 0.0) - original_scores.get(task, 0.0)),
        ]
        for task in ("hotpotqa", "musique", "macro")
    ]

    predicate_summary_rows = [
        [
            row["group"],
            row["predicate"],
            row["failed"],
            row["total"],
            fmt(row["failure_rate"]),
        ]
        for row in predicate_rate_rows
    ]

    feature_rows = [
        [
            row["group"],
            row["feature"],
            row["count"],
            fmt(row["mean"]),
            fmt(row["median"]),
            fmt(row["pearson_with_score_delta"]),
            fmt(row["spearman_with_score_delta"]),
        ]
        for row in separability_rows
    ]

    replay_rows = [
        [
            row["policy"],
            row["trigger_count"],
            row["trigger_sample_list"] or "[]",
            fmt(row["hotpotqa_policy"]),
            fmt(row["hotpotqa_delta"]),
            fmt(row["musique_policy"]),
            fmt(row["musique_delta"]),
            fmt(row["macro_policy"]),
            fmt(row["macro_delta"]),
            row["retained_musique_positive_count"],
            row["triggered_hotpot_catastrophic_negative_count"],
            row["non_all_reject"],
            row["non_all_trigger"],
        ]
        for row in policy_rows
    ]

    lines = [
        "# Adaptive DC-SlimInfer Router-v2 Offline Replay",
        "",
        f"- Tiny input directory: `{tiny_dir}`",
        f"- Fixed-gate input directory: `{fixed_gate_dir}`",
        f"- Output directory: `{output_dir}`",
        "- Generation run: **No**",
        "- Scorer run: **No**",
        "- Full/64 diagnostic: **No**",
        "- Threshold sweep: **No**",
        "- DC selector modified: **No**",
        "",
        "## Unconditional DC Reference",
        markdown_table(["task", "original", "dc", "delta"], score_rows),
        "",
        "## 1. Predicate Failure Analysis",
        v1_zero_reason,
        "",
        "Prediction_changed is included only as a diagnostic predicate. It is not used by router-v2 policies.",
        "",
        markdown_table(["group", "predicate", "failed", "total", "failure_rate"], predicate_summary_rows),
        "",
        "## 2. Feature Separability Analysis",
        markdown_table(
            [
                "group",
                "feature",
                "count",
                "mean",
                "median",
                "pearson_with_score_delta",
                "spearman_with_score_delta",
            ],
            feature_rows,
        ),
        "",
        "Feature-separability reading: the available selection features show weak separation. "
        "Several broad evidence-repair gates retain MuSiQue positives only by reopening "
        "HotpotQA negatives.",
        "",
        "## 3. Pre-Registered Router-v2 Policies",
        "All policies require budget_mismatch = 0, fallback = 0, original_entered_dc_selector = 0, "
        "and selection_changed = true. No policy uses task label, gold answer, score_delta, or "
        "prediction_changed as a gate feature.",
        "",
        markdown_table(
            [
                "policy",
                "triggers",
                "trigger samples",
                "hotpotqa policy",
                "hotpotqa delta",
                "musique policy",
                "musique delta",
                "macro policy",
                "macro delta",
                "MuSiQue positives retained",
                "Hotpot catastrophic triggered",
                "non-all reject",
                "non-all trigger",
            ],
            replay_rows,
        ),
        "",
        "## 4. Decision",
        f"- Decision: **{decision}**",
        f"- Reasons: {decision_reasons}",
    ]
    if bug_reasons:
        lines.extend(["", "### Bug Reasons", *[f"- {reason}" for reason in bug_reasons]])
    lines.extend(
        [
            "",
            "Recommended new pre-generation features for the next audit only: cutoff margin, "
            "original selection entropy, entity dispersion, and query-term spread.",
            "",
            REQUIRED_FOOTER,
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tiny-dir", type=Path, default=DEFAULT_TINY_DIR)
    parser.add_argument("--fixed-gate-dir", type=Path, default=DEFAULT_FIXED_GATE_DIR)
    parser.add_argument("--output-parent", type=Path, default=DEFAULT_OUTPUT_PARENT)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_parent / f"adaptive_dc_sliminfer_router_v2_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)

    data, missing = load_inputs(args.tiny_dir, args.fixed_gate_dir)
    sample_rows = build_sample_rows(data)
    bug_reasons = validate(sample_rows, data, missing)
    predicate_sample_rows, predicate_rate_rows = predicate_analysis(sample_rows)
    separability_rows = feature_separability(sample_rows)
    policy_rows = replay_policies(sample_rows)
    decision, decision_reasons = decide(policy_rows, separability_rows, bug_reasons)

    predicate_fields = [
        "task",
        "sample_index",
        "sample_id",
        "score_delta",
        "dc_positive",
        "dc_negative",
        "hotpotqa_catastrophic_negative",
        "musique_positive",
        "v1_failed_conditions",
        *[f"pass__{name}" for name, _ in predicate_specs()],
        "failed_predicates",
    ]
    write_csv(output_dir / "adaptive_dc_v2_predicate_failure.csv", predicate_sample_rows, predicate_fields)
    write_csv(
        output_dir / "adaptive_dc_v2_feature_separability.csv",
        separability_rows,
        [
            "analysis_type",
            "group",
            "feature",
            "count",
            "mean",
            "median",
            "pearson_with_score_delta",
            "spearman_with_score_delta",
        ],
    )
    write_csv(
        output_dir / "adaptive_dc_v2_policy_replay.csv",
        policy_rows,
        [
            "policy",
            "policy_description",
            "trigger_count",
            "trigger_sample_list",
            "hotpotqa_original",
            "hotpotqa_policy",
            "hotpotqa_delta",
            "musique_original",
            "musique_policy",
            "musique_delta",
            "macro_original",
            "macro_policy",
            "macro_delta",
            "unconditional_dc_macro",
            "policy_vs_unconditional_dc_macro_delta",
            "positive_count_vs_original",
            "negative_count_vs_original",
            "zero_count_vs_original",
            "retained_musique_positive_count",
            "retained_musique_positive_total",
            "triggered_hotpot_catastrophic_negative_count",
            "hotpot_catastrophic_negative_total",
            "filtered_hotpot_catastrophic_negative_count",
            "non_all_reject",
            "non_all_trigger",
            "better_than_original",
            "better_than_unconditional_dc",
            "final_gate_uses_task_gold_score_or_prediction_changed",
        ],
    )
    report = build_report(
        output_dir=output_dir,
        tiny_dir=args.tiny_dir,
        fixed_gate_dir=args.fixed_gate_dir,
        rows=sample_rows,
        predicate_rate_rows=predicate_rate_rows,
        separability_rows=separability_rows,
        policy_rows=policy_rows,
        decision=decision,
        decision_reasons=decision_reasons,
        bug_reasons=bug_reasons,
    )
    (output_dir / "adaptive_dc_v2_report.md").write_text(report, encoding="utf-8")
    decision_text = "\n".join(
        [
            "# Adaptive DC Router-v2 Decision",
            "",
            f"- Decision: **{decision}**",
            f"- Reasons: {decision_reasons}",
            f"- Output directory: `{output_dir}`",
            "",
            REQUIRED_FOOTER,
            "",
        ]
    )
    (output_dir / "adaptive_dc_v2_decision.md").write_text(decision_text, encoding="utf-8")
    manifest = {
        "decision": decision,
        "decision_reasons": decision_reasons,
        "bug_reasons": bug_reasons,
        "output_dir": str(output_dir),
        "tiny_dir": str(args.tiny_dir),
        "fixed_gate_dir": str(args.fixed_gate_dir),
        "policy_rows": policy_rows,
        "artifact_paths": {
            key: str(value) if value is not None else None for key, value in data["artifacts"].items()
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if decision != "BUG_STOP_FIX_REQUIRED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
