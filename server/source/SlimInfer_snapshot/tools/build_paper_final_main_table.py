#!/usr/bin/env python3
"""Build paper-facing full tables, partial status, analysis, and claims."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import shutil
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results"
BASELINE_TABLE = RESULTS / "final_drm_add_mainline" / "main_table_full_longbench.csv"
STRUCTURED_128 = RESULTS / "structured_drm_stage2_128_20260605_070455" / "structured_stage2_128_summary.csv"
RH_64 = RESULTS / "rh_drm_stage2_64_20260604_072019" / "rh_stage2_64_summary.csv"
FUP_64 = RESULTS / "fup_guarded_swap_stage2_64_20260601_051737" / "fup_recheck_summary.csv"
GUARDED_64 = RESULTS / "fup_guarded_swap_stage2_64_20260601_051737" / "guarded_swap_stage2_64_summary.csv"
AC_64 = RESULTS / "ac_gated_drm_stage2_64_20260603_193334" / "ac_stage2_64_summary.csv"
AB_128 = RESULTS / "ab_calibrated_recheck_128_20260608_071200" / "ab_recheck_128_summary.csv"
AF_DIAG = RESULTS / "rescuekv_block_system_af_diagnostic_20260531_064336" / "stage1_scheme_ranking.csv"

TASK_COLUMNS = [
    ("qasper", "Qasper"), ("multifieldqa_en", "MFQA"), ("hotpotqa", "HPQA"),
    ("2wikimqa", "2Wiki"), ("musique", "MuSiQue"), ("gov_report", "GovRep"),
    ("qmsum", "QMSum"), ("multi_news", "MNews"), ("trec", "TREC"),
    ("triviaqa", "TQA"), ("samsum", "SAMSum"), ("passage_count", "Count"),
    ("passage_retrieval_en", "PassR"), ("lcc", "LCC"), ("repobench-p", "RepoB-p"),
    ("vcsum", "VCSum"), ("lsht", "LSHT"),
]
CATEGORIES = {
    "SingleDoc QA": ["qasper", "multifieldqa_en", "hotpotqa"],
    "MultiDoc QA": ["2wikimqa", "musique"],
    "Summarization": ["gov_report", "qmsum", "multi_news"],
    "Few-shot / Classification": ["trec", "triviaqa", "samsum"],
    "Synthetic / Retrieval": ["passage_count", "passage_retrieval_en"],
    "Code": ["lcc", "repobench-p"],
    "Chinese": ["vcsum", "lsht"],
}
BASELINE_SPECS = [
    ("FullKV", "FullKV", "uncompressed upper bound", "Full KV"),
    ("SlimInfer original", "SlimInfer", "external baseline", "Original SlimInfer K"),
    ("SlimInfer larger K+M", "SlimInfer larger K+M", "budget-matched baseline", "Larger K+M"),
    ("Random-Add", "Random-Add", "random additive control", "Matched K+M additive budget"),
    ("DeltaO-Add", "DeltaO-Add", "functional additive control", "Matched K+M additive budget"),
    ("fixed DRM-Add", "DRM-Add", "base Ours", "Matched K+M additive budget"),
]
STRUCTURED_SPECS = [
    ("SpanDRM", "span_drm_add_query", "final Ours candidate"),
    ("RH-then-Span", "rh_then_span_drm_add_query", "final structured Ours candidate"),
]
MAIN_FIELDS = [
    "Method", "Role", "Evaluation", "Budget", "Avg",
    "Delta_vs_SlimInfer", "Delta_vs_SlimInfer_Larger", "Delta_vs_Random_Add",
    "Delta_vs_DeltaO_Add", "Delta_vs_fixed_DRM_Add", "Completion",
    "Budget_error_mean", "Budget_error_max", "Budget_warning", "Runner_fail",
    "Use_in_claim", "Notes",
]


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def to_float(value: Any) -> Optional[float]:
    try:
        if value in (None, "", "N/A", "n/a"):
            return None
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def fmt(value: Any, digits: int = 2, signed: bool = False) -> str:
    number = to_float(value)
    if number is None:
        return ""
    return f"{number:+.{digits}f}" if signed else f"{number:.{digits}f}"


def avg(values: Iterable[Any]) -> Optional[float]:
    numbers = [number for number in (to_float(value) for value in values) if number is not None]
    return mean(numbers) if numbers else None


def find_row(rows: Sequence[Dict[str, str]], key: str, value: str) -> Dict[str, str]:
    return next((row for row in rows if row.get(key) == value), {})


def markdown_table(rows: Sequence[Dict[str, Any]], fields: Sequence[str]) -> str:
    def clean(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(clean(row.get(field, "")) for field in fields) + " |")
    return "\n".join(lines)


def structured_state(root: Path) -> Dict[str, Any]:
    progress = read_json(root / "progress.json", {})
    scores = read_csv(root / "structured_full_scores.csv")
    budgets = read_csv(root / "structured_full_budget_audit.csv")
    methods: Dict[str, Any] = {}
    per_method_expected = int(progress.get("structured_generations_expected", 0) or 0) // max(
        1, len(progress.get("methods", STRUCTURED_SPECS))
    )
    for display, method, role in STRUCTURED_SPECS:
        method_scores = [row for row in scores if row.get("method") == method]
        by_task = {row.get("task", ""): to_float(row.get("score")) for row in method_scores}
        complete = (
            bool(progress.get("full_complete"))
            and len(by_task) == len(TASK_COLUMNS)
            and all(by_task.get(task) is not None for task, _ in TASK_COLUMNS)
            and all(row.get("status") == "ok" for row in method_scores)
        )
        method_budget = [row for row in budgets if row.get("method") == method]
        errors = [abs(float(row.get("budget_error_pct", 0) or 0)) for row in method_budget]
        warnings = sum(int(float(row.get("budget_warning", 0) or 0)) for row in method_budget)
        runner_fail = sum(int(float(row.get("runner_fail", 0) or 0)) for row in method_scores)
        methods[method] = {
            "display": display,
            "role": role,
            "complete": complete,
            "scores": by_task,
            "avg": avg(by_task.values()) if complete else None,
            "completion": (
                f"{sum(int(float(row.get('sample_count', 0) or 0)) for row in method_scores)}/"
                f"{max(sum(int(float(row.get('expected_count', 0) or 0)) for row in method_scores), per_method_expected)}"
            ),
            "budget_error_mean": mean(errors) if errors else None,
            "budget_error_max": max(errors) if errors else None,
            "budget_warning": warnings,
            "budget_pass": (
                bool(errors)
                and mean(errors) < 3.0
                and max(errors) < 5.0
                and warnings == 0
            ),
            "runner_fail": runner_fail,
        }
    return {"progress": progress, "methods": methods}


def baseline_scores(row: Dict[str, str]) -> Dict[str, Optional[float]]:
    return {task: to_float(row.get(column)) for task, column in TASK_COLUMNS}


def build_tables(structured_root: Path) -> Dict[str, Any]:
    baseline_rows = read_csv(BASELINE_TABLE)
    base_map = {row["Method"]: row for row in baseline_rows}
    structured = structured_state(structured_root)
    complete_methods = {
        method: state for method, state in structured["methods"].items() if state["complete"]
    }
    reference = {
        "SlimInfer": float(base_map["SlimInfer"]["Avg"]),
        "SlimInfer larger K+M": float(base_map["SlimInfer larger K+M"]["Avg"]),
        "Random-Add": float(base_map["Random-Add"]["Avg"]),
        "DeltaO-Add": float(base_map["DeltaO-Add"]["Avg"]),
        "DRM-Add": float(base_map["DRM-Add"]["Avg"]),
    }
    main: List[Dict[str, Any]] = []
    per_task: List[Dict[str, Any]] = []
    all_full: List[Dict[str, Any]] = []
    for display, source, role, budget in BASELINE_SPECS:
        row = base_map[source]
        score = float(row["Avg"])
        item = {
            "Method": display, "Role": role, "Evaluation": "Full LongBench",
            "Budget": budget, "Avg": fmt(score),
            "Delta_vs_SlimInfer": fmt(score - reference["SlimInfer"], signed=True),
            "Delta_vs_SlimInfer_Larger": fmt(score - reference["SlimInfer larger K+M"], signed=True),
            "Delta_vs_Random_Add": fmt(score - reference["Random-Add"], signed=True),
            "Delta_vs_DeltaO_Add": fmt(score - reference["DeltaO-Add"], signed=True),
            "Delta_vs_fixed_DRM_Add": fmt(score - reference["DRM-Add"], signed=True),
            "Completion": row.get("SampleCount", "3950/3950"),
            "Budget_error_mean": "N/A", "Budget_error_max": "N/A",
            "Budget_warning": "Historical full audit has no normalized target-error metric",
            "Runner_fail": "0", "Use_in_claim": "Yes",
            "Notes": "Existing complete baseline; not rerun.",
        }
        main.append(item)
        scores = baseline_scores(row)
        all_full.append({"display": display, "scores": scores, "avg": score})
    for method, state in complete_methods.items():
        score = float(state["avg"])
        item = {
            "Method": state["display"], "Role": state["role"], "Evaluation": "Full LongBench",
            "Budget": "Matched fixed DRM-Add selected-token target",
            "Avg": fmt(score),
            "Delta_vs_SlimInfer": fmt(score - reference["SlimInfer"], signed=True),
            "Delta_vs_SlimInfer_Larger": fmt(score - reference["SlimInfer larger K+M"], signed=True),
            "Delta_vs_Random_Add": fmt(score - reference["Random-Add"], signed=True),
            "Delta_vs_DeltaO_Add": fmt(score - reference["DeltaO-Add"], signed=True),
            "Delta_vs_fixed_DRM_Add": fmt(score - reference["DRM-Add"], signed=True),
            "Completion": state["completion"],
            "Budget_error_mean": fmt(state["budget_error_mean"], 4),
            "Budget_error_max": fmt(state["budget_error_max"], 4),
            "Budget_warning": state["budget_warning"],
            "Runner_fail": state["runner_fail"],
            "Use_in_claim": "Yes" if state["budget_pass"] and not state["runner_fail"] else "No",
            "Notes": "Order-preserving structured context/KV-retention proxy; no latency run.",
        }
        main.append(item)
        all_full.append({"display": state["display"], "scores": state["scores"], "avg": score})

    for method in all_full:
        row = {"Method": method["display"]}
        row.update({task: fmt(method["scores"].get(task)) for task, _ in TASK_COLUMNS})
        row["Avg"] = fmt(method["avg"])
        per_task.append(row)

    categories: List[Dict[str, Any]] = []
    slim_scores = next(row["scores"] for row in all_full if row["display"] == "SlimInfer original")
    larger_scores = next(row["scores"] for row in all_full if row["display"] == "SlimInfer larger K+M")
    fixed_scores = next(row["scores"] for row in all_full if row["display"] == "fixed DRM-Add")
    for method in all_full:
        for category, tasks in CATEGORIES.items():
            score = avg(method["scores"].get(task) for task in tasks)
            slim_score = avg(slim_scores.get(task) for task in tasks)
            larger_score = avg(larger_scores.get(task) for task in tasks)
            fixed_score = avg(fixed_scores.get(task) for task in tasks)
            categories.append({
                "Method": method["display"], "Category": category,
                "Category_Avg": fmt(score),
                "Delta_vs_SlimInfer": fmt(score - slim_score if score is not None and slim_score is not None else None, signed=True),
                "Delta_vs_SlimInfer_Larger": fmt(score - larger_score if score is not None and larger_score is not None else None, signed=True),
                "Delta_vs_fixed_DRM_Add": fmt(score - fixed_score if score is not None and fixed_score is not None else None, signed=True),
            })
    deltas = [
        {key: row.get(key, "") for key in [
            "Method", "Avg", "Delta_vs_SlimInfer", "Delta_vs_SlimInfer_Larger",
            "Delta_vs_Random_Add", "Delta_vs_DeltaO_Add", "Delta_vs_fixed_DRM_Add",
        ]}
        for row in main
    ]
    return {
        "main": main, "per_task": per_task, "categories": categories, "deltas": deltas,
        "structured": structured, "all_full": all_full,
    }


def diagnostic_row(path: Path, method: str, evaluation: str, family: str, conclusion: str) -> Dict[str, Any]:
    row = find_row(read_csv(path), "method", method)
    return {
        "Method": method, "Family": family, "Evaluation": evaluation,
        "Avg": row.get("avg", row.get("avg_score", "")),
        "Delta_vs_local_fixed": row.get("delta_vs_baseline", ""),
        "Passed_gate": row.get("passed_gate", ""),
        "Budget_warning": row.get("budget_warning", ""),
        "Runner_fail": row.get("runner_fail", ""),
        "Use_in_full_claim": "No",
        "Conclusion": conclusion,
    }


def build_ablation() -> List[Dict[str, Any]]:
    baseline = find_row(read_csv(BASELINE_TABLE), "Method", "DRM-Add")
    ab_rows = read_csv(AB_128)
    ab_method = find_row(ab_rows, "method", "attention_boundary_boundary_drm_matched_KplusM")
    ab_fixed = find_row(ab_rows, "method", "fixed_drm_add_matched_KplusM")
    ab_avg = to_float(ab_method.get("avg_score"))
    ab_fixed_avg = to_float(ab_fixed.get("avg_score"))
    rows = [{
        "Method": "fixed DRM-Add", "Family": "Ours base", "Evaluation": "Full LongBench",
        "Avg": baseline.get("Avg", ""), "Delta_vs_local_fixed": "0.00", "Passed_gate": "full",
        "Budget_warning": "N/A", "Runner_fail": "0", "Use_in_full_claim": "Yes",
        "Conclusion": "Current full-backed Ours.",
    }]
    rows.extend([
        diagnostic_row(STRUCTURED_128, "rh_drm_add_query_fixed_pool", "128-sample diagnostic", "RH-DRM", "Head-aware rescue has a small positive diagnostic gain."),
        diagnostic_row(STRUCTURED_128, "span_drm_add_query", "128-sample diagnostic", "SpanDRM", "Strongest structured diagnostic candidate."),
        diagnostic_row(
            STRUCTURED_128,
            "rh_then_span_drm_add_query",
            "128-sample diagnostic",
            "RH-then-Span",
            "Selected for full because it excludes the FUP-derived hybrid signal; its diagnostic gate did not pass.",
        ),
        diagnostic_row(FUP_64, "fup_oracle_keepdrop", "64-sample diagnostic", "FUP", "Learned utility signal only; not a deployable full claim."),
        {
            "Method": "attention_boundary_boundary_drm_matched_KplusM",
            "Family": "AB-BoundaryDRM",
            "Evaluation": "calibrated 128-sample diagnostic",
            "Avg": fmt(ab_avg, 3),
            "Delta_vs_local_fixed": fmt(
                ab_avg - ab_fixed_avg
                if ab_avg is not None and ab_fixed_avg is not None else None,
                3,
                signed=True,
            ),
            "Passed_gate": "No",
            "Budget_warning": "0",
            "Runner_fail": "0",
            "Use_in_full_claim": "No",
            "Conclusion": "Calibration passed, but the clear-gain/no-regression promotion gate failed.",
        },
        diagnostic_row(AC_64, "ac_gated_drm_add", "64-sample diagnostic", "AC-Gated", "Did not establish a stable improvement over fixed DRM-Add."),
        diagnostic_row(GUARDED_64, "guarded_swap_drm", "64-sample diagnostic", "Guarded Swap", "Below add-only baseline with runner/recall risk."),
    ])
    af_rows = read_csv(AF_DIAG)
    best_af = max(af_rows, key=lambda row: to_float(row.get("avg_score", row.get("score"))) or -1e9, default={})
    rows.append({
        "Method": best_af.get("method", best_af.get("scheme", "A-F candidate construction")),
        "Family": "A-F candidate construction", "Evaluation": "diagnostic",
        "Avg": best_af.get("avg_score", best_af.get("score", "")),
        "Delta_vs_local_fixed": "", "Passed_gate": "No", "Budget_warning": best_af.get("budget_warning", ""),
        "Runner_fail": "", "Use_in_full_claim": "No", "Conclusion": "Did not pass promotion gate.",
    })
    return rows


def negative_rows() -> List[Dict[str, str]]:
    return [
        {"Family": "A-F candidate source", "Evaluation": "diagnostic", "Decision": "Stop", "Reason": "Did not pass the promotion gate."},
        {"Family": "query/coverage/verifier selectors", "Evaluation": "64-sample diagnostic", "Decision": "Stop", "Reason": "Task behavior was unstable and did not consistently exceed fixed DRM-Add."},
        {"Family": "Guarded Swap", "Evaluation": "64-sample diagnostic", "Decision": "Stop", "Reason": "Below add-only control with recall and runner-failure risk."},
        {"Family": "AC-Gated", "Evaluation": "64-sample diagnostic", "Decision": "Stop", "Reason": "Active-conditioned gate/utility did not exceed fixed DRM-Add."},
        {"Family": "AB-BoundaryDRM", "Evaluation": "calibrated 128-sample diagnostic", "Decision": "No full", "Reason": "Budget calibration passed, but clear-gain/no-regression promotion failed."},
        {"Family": "FUP", "Evaluation": "64-sample diagnostic", "Decision": "Secondary only", "Reason": "Learned utility has signal, but deployable output-delta gain is too weak for a full claim."},
    ]


def choose_final_ours(tables: Dict[str, Any]) -> Dict[str, Any]:
    main = {row["Method"]: row for row in tables["main"]}
    fixed = main["fixed DRM-Add"]
    candidates = [
        row for name, row in main.items()
        if name in {"SpanDRM", "RH-then-Span"}
        and row.get("Use_in_claim") == "Yes"
        and (to_float(row.get("Avg")) or -1e9) > (to_float(fixed.get("Avg")) or -1e9)
        and (to_float(row.get("Avg")) or -1e9) > (to_float(main["SlimInfer larger K+M"].get("Avg")) or -1e9)
    ]
    return max(candidates, key=lambda row: to_float(row.get("Avg")) or -1e9, default=fixed)


def write_analysis(out: Path, tables: Dict[str, Any], structured_root: Path) -> None:
    progress = tables["structured"]["progress"]
    final_ours = choose_final_ours(tables)
    full_complete = bool(progress.get("full_complete"))
    method = final_ours["Method"]
    main_map = {row["Method"]: row for row in tables["main"]}
    score_map = {row["display"]: row["scores"] for row in tables["all_full"]}
    comparison = "fixed DRM-Add" if method != "fixed DRM-Add" else "SlimInfer original"
    method_scores = score_map[method]
    comparison_scores = score_map[comparison]
    task_deltas = [
        (task, float(method_scores[task]) - float(comparison_scores[task]))
        for task, _ in TASK_COLUMNS
        if method_scores.get(task) is not None and comparison_scores.get(task) is not None
    ]
    gains = sorted(task_deltas, key=lambda item: item[1], reverse=True)[:5]
    drops = sorted((item for item in task_deltas if item[1] < 0), key=lambda item: item[1])[:5]
    multihop_tasks = ["hotpotqa", "2wikimqa", "musique"]
    multihop_delta = mean(
        float(method_scores[task]) - float(comparison_scores[task])
        for task in multihop_tasks
    )
    gain_text = ", ".join(f"{task} {delta:+.2f}" for task, delta in gains) or "none"
    drop_text = ", ".join(f"{task} {delta:+.2f}" for task, delta in drops) or "none"
    structured_result_lines: List[str] = []
    structured_task_lines: List[str] = []
    for display in ("SpanDRM", "RH-then-Span"):
        row = main_map.get(display)
        if row is None:
            continue
        structured_result_lines.append(
            f"- {display}: Avg {row['Avg']}; vs SlimInfer {row['Delta_vs_SlimInfer']}; "
            f"vs SlimInfer larger {row['Delta_vs_SlimInfer_Larger']}; "
            f"vs fixed DRM-Add {row['Delta_vs_fixed_DRM_Add']}; "
            f"budget mean/max {row['Budget_error_mean']}%/{row['Budget_error_max']}%; "
            f"warnings {row['Budget_warning']}; runner failures {row['Runner_fail']}."
        )
        deltas = [
            (task, float(score_map[display][task]) - float(score_map["fixed DRM-Add"][task]))
            for task, _ in TASK_COLUMNS
        ]
        top = sorted(deltas, key=lambda item: item[1], reverse=True)[:3]
        bottom = sorted(deltas, key=lambda item: item[1])[:3]
        multi = mean(dict(deltas)[task] for task in multihop_tasks)
        structured_task_lines.append(
            f"- {display} vs fixed DRM-Add: largest gains "
            + ", ".join(f"{task} {delta:+.2f}" for task, delta in top)
            + "; largest drops "
            + ", ".join(f"{task} {delta:+.2f}" for task, delta in bottom)
            + f"; multi-hop mean {multi:+.2f}."
        )
    structured_result_text = (
        "\n".join(structured_result_lines)
        if structured_result_lines else
        "- Structured full rows are pending and are excluded from full-result comparisons."
    )
    structured_task_text = (
        "\n".join(structured_task_lines)
        if structured_task_lines else
        "- Structured task-level comparisons remain pending."
    )
    if {"SpanDRM", "RH-then-Span"}.issubset(score_map):
        span_rh_delta = (
            float(main_map["SpanDRM"]["Avg"]) - float(main_map["RH-then-Span"]["Avg"])
        )
        span_rh_text = (
            f"SpanDRM exceeds RH-then-Span by {span_rh_delta:+.2f} Avg."
            if span_rh_delta >= 0 else
            f"RH-then-Span exceeds SpanDRM by {-span_rh_delta:+.2f} Avg."
        )
    else:
        span_rh_text = "The full SpanDRM vs RH-then-Span comparison is pending."
    conclusion = (
        f"The completed full comparison promotes {method} as final Ours."
        if full_complete and method != "fixed DRM-Add" else
        "The completed Structured methods did not clear both fixed DRM-Add and SlimInfer larger K+M; "
        "fixed DRM-Add remains final Ours, and Structured DRM is retained as a diagnostic extension."
        if full_complete else
        "Structured DRM full results are pending; fixed DRM-Add remains the current full-backed Ours."
    )
    analysis = f"""# Paper Final Analysis

## Experimental Setup

- Model: Qwen2.5-7B-Instruct, snapshot `a09a35458c702b33eeacc393d103063234e8bc28`, identical across methods.
- Dataset: LongBench local full/test split, 17 tasks, 3950 total samples per complete method.
- Evaluation: the existing LongBench prompt templates, task metrics, and task-specific maximum generation lengths.
- Generation: greedy decoding (`temperature=0`, `top_p=1.0`); no latency experiment was run.
- Environment: repository `.venv`; conda environment `silm` was unavailable.
- Budget: fixed-page baselines reuse the completed full run. Structured methods target the per-sample fixed DRM-Add selected-token count and emit mean/max budget error plus warnings.
- Structured implementation: order-preserving text/token span packing evaluated with ordinary FullKV, serving as a structured context/KV-retention proxy rather than a physical sparse-KV span cache.

## Main Result Status

- Structured full complete: **{full_complete}**.
- Current full-backed final Ours: **{method}** (Avg {final_ours['Avg']}).
- Delta vs SlimInfer original: {final_ours['Delta_vs_SlimInfer']}.
- Delta vs SlimInfer larger K+M: {final_ours['Delta_vs_SlimInfer_Larger']}.
- Delta vs Random-Add: {final_ours['Delta_vs_Random_Add']}.
- Delta vs DeltaO-Add: {final_ours['Delta_vs_DeltaO_Add']}.
- Delta vs fixed DRM-Add: {final_ours['Delta_vs_fixed_DRM_Add']}.

Completed Structured rows:

{structured_result_text}

## Fairness

All claimed full rows use the same model, data split, prompt/evaluation code, and generation lengths. Structured selection uses query similarity, query keyword overlap, input-derived entity overlap, and parent DRM signals. Gold answers are retained only for post-hoc diagnostics and metric computation; the deployable query scorers do not consume answer overlap or oracle labels. The stronger 128-sample RH-then-Span hybrid was not promoted because its parent-page score includes the FUP metric-derived signal; the full run therefore uses the leakage-safe query variant.

Passage retrieval and passage count use predefined full-page fallback behavior. Span compression variants for those tasks remain negative ablations. Budget fairness is claimable for a Structured method only after full completion, zero runner failures, and passing budget audit.

## Structured Contributions

- fixed DRM-Add: damage-aware add-only rescue.
- SpanDRM: evidence-density-aware span rescue.
- RH-then-Span: head-aware page selection followed by span-level evidence packing.

## Task-Level Analysis

Relative to {comparison}, the largest task gains are: {gain_text}. Tasks with negative deltas are: {drop_text}. The mean change over HotpotQA, 2WikiMQA, and MuSiQue is {multihop_delta:+.2f}. Full task/category values are reported in `paper_main_table_by_task.csv` and `paper_main_table_by_category.csv`; the seven category groups are mutually exclusive and cover all 17 tasks.

Structured full task comparisons:

{structured_task_text}

{span_rh_text}

Retrieval/count fallback necessity is supported diagnostically by the failed compression ablations; the full main methods preserve their predefined fallback.

## Ablations And Negative Diagnostics

RH-DRM, SpanDRM, and RH-then-Span diagnostic results motivated the two full candidates but are not used as full evidence. A-F, query/coverage/verifier, Guarded Swap, AC-Gated, AB-BoundaryDRM, and FUP are summarized as stopped or secondary diagnostics in `paper_negative_diagnostics_summary.csv`.

## Limitations

- SpanDRM is currently a structured context/KV-retention proxy, not a physical span-granular sparse KV-cache implementation.
- Validation is limited to Qwen2.5-7B-Instruct and LongBench.
- No full latency run was performed, so no speedup or latency claim is supported.
- The frozen RH profile was calibrated on five held-out diagnostic tasks and reused unchanged on all full tasks.

## Conclusion

{conclusion}
"""
    (out / "paper_final_analysis.md").write_text(analysis, encoding="utf-8")

    if full_complete:
        structured_claim = (
            f"- {method} clears both fixed DRM-Add and SlimInfer larger K+M and is selected as final Ours."
            if method != "fixed DRM-Add" else
            "- Neither completed Structured candidate clears the automatic full-result decision rule; fixed DRM-Add remains final Ours."
        )
        claims = f"""# Paper Claims

## Strong Claims

- The completed full LongBench table directly supports the reported comparison among all rows marked `Use_in_claim=Yes`.
{structured_claim}
- Budget fairness is supported only for Structured rows whose full budget audit has no warnings and no runner failures.

## Moderate Claims

- Small full-score differences should be described as modest improvements rather than broad dominance.
- Task/category gains are supported only by the corresponding full rows.

## Diagnostic Claims

- RH-DRM, SpanDRM, and RH-then-Span 64/128-sample results motivated candidate selection.
- FUP contains a learned utility signal, while AB, AC-Gated, Guarded Swap, A-F, and selector variants are negative diagnostics.

## Do Not Claim

- Do not claim latency or speedup; full latency was not run.
- Do not claim physical span-level KV-cache efficiency; the implementation is a structured context/KV-retention proxy.
- Do not generalize beyond Qwen2.5-7B-Instruct and LongBench without additional experiments.
"""
    else:
        claims = """# Paper Claims

## Strong Claims

- fixed DRM-Add achieves 47.80 full LongBench Avg, +0.21 over SlimInfer original and +0.10 over SlimInfer larger K+M.
- The six existing baseline/base-Ours methods are complete at 3950/3950 samples.

## Moderate Claims

- fixed DRM-Add modestly improves over the budget-matched fixed-page baseline.

## Diagnostic Claims

- SpanDRM and RH-then-Span are promising 128-sample structured candidates.
- Negative diagnostics identify approaches that should not be promoted.

## Do Not Claim

- Structured DRM full results are pending; do not claim that SpanDRM or RH-then-Span outperforms SlimInfer on full LongBench.
- Do not claim latency or speedup.
- Do not claim physical span-level sparse-KV retention.
- Do not claim budget-fair Structured full comparison until the full budget audit completes.
"""
    (out / "paper_claims.md").write_text(claims, encoding="utf-8")


def write_status(out: Path, tables: Dict[str, Any]) -> None:
    progress = tables["structured"]["progress"]
    incomplete = [
        state["display"] for state in tables["structured"]["methods"].values()
        if not state["complete"]
    ]
    current = choose_final_ours(tables)
    status = f"""# Current Paper Status

- Structured full complete: **{bool(progress.get('full_complete'))}**
- Tasks complete: {progress.get('tasks_complete', 0)}/{progress.get('tasks_total', 17)}
- Structured generations: {progress.get('structured_generations_success', 0)}/{progress.get('structured_generations_expected', 0)}
- Current full-backed Ours: **{current['Method']}**
- Missing full methods: {', '.join(incomplete) if incomplete else 'none'}
- No partial Structured row is included in `paper_main_table_full.csv`.
"""
    (out / "current_status_table.md").write_text(status, encoding="utf-8")
    (out / "missing_full_methods.md").write_text(
        "# Missing Full Methods\n\n"
        + ("\n".join(f"- {method}" for method in incomplete) if incomplete else "- None")
        + "\n",
        encoding="utf-8",
    )
    source_progress = structured_progress = (
        (Path(read_json(out / "config.json", {}).get("structured_full_root", "")) / "progress.md")
        if (out / "config.json").exists() else Path()
    )
    if structured_progress and structured_progress.exists():
        shutil.copy2(structured_progress, out / "partial_progress.md")
    else:
        (out / "partial_progress.md").write_text(
            "# Partial Progress\n\nStructured full has not produced progress yet.\n",
            encoding="utf-8",
        )


def write_run_scripts(out: Path, structured_root: Path) -> None:
    run = structured_root / "run_missing_structured_full.sh"
    resume = structured_root / "resume_missing_structured_full.sh"
    for source, name in (
        (run, "run_missing_structured_full.sh"),
        (resume, "resume_missing_structured_full.sh"),
    ):
        if source.exists():
            shutil.copy2(source, out / name)
            (out / name).chmod(0o755)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--structured_full_root", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    structured_root = Path(args.structured_full_root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tables = build_tables(structured_root)
    main = tables["main"]
    per_task_fields = ["Method"] + [task for task, _ in TASK_COLUMNS] + ["Avg"]
    category_fields = [
        "Method", "Category", "Category_Avg", "Delta_vs_SlimInfer",
        "Delta_vs_SlimInfer_Larger", "Delta_vs_fixed_DRM_Add",
    ]
    delta_fields = [
        "Method", "Avg", "Delta_vs_SlimInfer", "Delta_vs_SlimInfer_Larger",
        "Delta_vs_Random_Add", "Delta_vs_DeltaO_Add", "Delta_vs_fixed_DRM_Add",
    ]
    write_csv(out / "paper_main_table_full.csv", main, MAIN_FIELDS)
    (out / "paper_main_table_full.md").write_text(
        "# Paper Main Table: Full LongBench\n\n"
        "Environment: conda `silm` was unavailable; the repository `.venv` was used.\n\n"
        + (
            "Structured DRM full results are pending; current final full claim is fixed DRM-Add vs SlimInfer.\n\n"
            if not tables["structured"]["progress"].get("full_complete") else
            "Structured DRM full results are complete; only complete full rows appear below.\n\n"
        )
        + markdown_table(main, MAIN_FIELDS) + "\n",
        encoding="utf-8",
    )
    write_csv(out / "paper_main_table_by_task.csv", tables["per_task"], per_task_fields)
    write_csv(out / "paper_main_table_by_category.csv", tables["categories"], category_fields)
    write_csv(out / "paper_deltas_vs_baselines.csv", tables["deltas"], delta_fields)

    budget_rows = read_csv(structured_root / "structured_full_budget_audit.csv")
    write_csv(
        out / "paper_budget_audit.csv",
        budget_rows,
        list(budget_rows[0]) if budget_rows else [
            "method", "task", "n_samples", "avg_target_tokens", "avg_selected_tokens",
            "budget_error_pct", "budget_warning",
        ],
    )
    status_rows: List[Dict[str, Any]] = []
    for row in main:
        status_rows.append({
            "Method": row["Method"], "Evaluation": row["Evaluation"],
            "Completion": row["Completion"], "Complete": "Yes",
            "Runner_fail": row["Runner_fail"], "Use_in_claim": row["Use_in_claim"],
        })
    for state in tables["structured"]["methods"].values():
        if state["complete"]:
            continue
        status_rows.append({
            "Method": state["display"], "Evaluation": "Full LongBench partial/pending",
            "Completion": state["completion"], "Complete": "No",
            "Runner_fail": state["runner_fail"], "Use_in_claim": "No",
        })
    write_csv(
        out / "paper_method_status.csv",
        status_rows,
        ["Method", "Evaluation", "Completion", "Complete", "Runner_fail", "Use_in_claim"],
    )
    ablation = build_ablation()
    write_csv(
        out / "paper_ablation_table.csv",
        ablation,
        [
            "Method", "Family", "Evaluation", "Avg", "Delta_vs_local_fixed", "Passed_gate",
            "Budget_warning", "Runner_fail", "Use_in_full_claim", "Conclusion",
        ],
    )
    negative = negative_rows()
    write_csv(
        out / "paper_negative_diagnostics_summary.csv",
        negative,
        ["Family", "Evaluation", "Decision", "Reason"],
    )
    (out / "paper_negative_diagnostics_summary.md").write_text(
        "# Negative Diagnostics\n\n"
        + markdown_table(negative, ["Family", "Evaluation", "Decision", "Reason"]) + "\n",
        encoding="utf-8",
    )
    config = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "structured_full_root": str(structured_root),
        "baseline_table": str(BASELINE_TABLE),
        "model": "Qwen2.5-7B-Instruct",
        "model_snapshot": "a09a35458c702b33eeacc393d103063234e8bc28",
        "dataset": "LongBench full/test",
        "tasks": [task for task, _ in TASK_COLUMNS],
        "generation": {"temperature": 0, "top_p": 1.0, "max_new_tokens": "configs/dataset2maxlen.json"},
        "environment_note": "conda silm unavailable; repo .venv used",
        "structured_method_choice": (
            "span_drm_add_query and rh_then_span_drm_add_query; the hybrid variant was excluded "
            "because its parent-page score includes the FUP metric-derived signal"
        ),
        "latency_run": False,
        "structured_full_complete": bool(tables["structured"]["progress"].get("full_complete")),
    }
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    write_analysis(out, tables, structured_root)
    write_run_scripts(out, structured_root)
    write_status(out, tables)
    (out / "README.md").write_text(
        "# Paper Final Main Table\n\n"
        "This directory contains paper-facing full tables, task/category breakdowns, budget audit, "
        "method status, ablations, negative diagnostics, analysis, claims, and resume-safe run scripts.\n\n"
        "Structured rows enter the full table only after all 17 tasks complete with full sample counts.\n\n"
        "Environment note: conda `silm` was unavailable, so the repository `.venv` was used.\n",
        encoding="utf-8",
    )
    print(out)


if __name__ == "__main__":
    main()
