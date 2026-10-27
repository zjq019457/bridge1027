#!/usr/bin/env python3
"""Run and audit the 32-sample Qwen Evidence-density diagnostic gate."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
RUNNER = REPO_ROOT / "experiments" / "run_qwen25_sliminfer_variants.py"
TASKS = ["qasper", "hotpotqa", "musique", "passage_count", "passage_retrieval_en"]
METHODS = [
    "fixed_drm_add",
    "evidence_density_srm",
    "rh_pyramid_srm",
    "bridge_graph_srm",
]
STRUCTURED = set(METHODS) - {"fixed_drm_add"}
DEFAULT_MODEL = (
    "/home/yu-bai/.cache/huggingface/hub/"
    "models--Qwen--Qwen2.5-7B-Instruct/snapshots/"
    "a09a35458c702b33eeacc393d103063234e8bc28"
)
DEFAULT_LONGBENCH = "/home/yu-bai/Desktop/jiaqizhang/HeadKV/data/LongBench"


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _write_csv(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
    fields: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(fields),
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def _jaccard(left: Set[int], right: Set[int]) -> float:
    if not left and not right:
        return 1.0
    return len(left & right) / max(1, len(left | right))


def _method_spec(method: str) -> Dict[str, Any]:
    if method == "fixed_drm_add":
        return {"rescue_score": "drm", "structured": False}
    if method == "rh_pyramid_srm":
        return {
            "rescue_score": method,
            "structured": True,
            "rh_weight": 0.50,
            "pyramid_weight": 0.20,
        }
    if method == "bridge_graph_srm":
        return {
            "rescue_score": method,
            "structured": True,
            "bridge_weight": 0.45,
        }
    if method == "evidence_density_srm":
        return {
            "rescue_score": method,
            "structured": True,
            "density_alpha_max": 1.0,
            "density_alpha_count": 0.6,
            "density_alpha_continuity": 0.8,
            "density_alpha_coverage": 0.5,
            "density_lambda_red": 0.3,
            "density_threshold_mode": "mean_std",
            "density_top_quantile": 0.30,
        }
    raise ValueError(f"Unknown method: {method}")


def _command(root: Path, method: str, args: argparse.Namespace) -> List[str]:
    spec = _method_spec(method)
    command = [
        str(PYTHON),
        str(RUNNER),
        "--model_path", args.model_path,
        "--tokenizer_path", args.model_path,
        "--model_family", "qwen2",
        "--dtype", "fp16",
        "--device", "cuda",
        "--backend", "flash_attention_2",
        "--method", "sliminfer_variant",
        "--scoring", "delta_o",
        "--decode_policy", "reuse_prefill",
        "--eval", "longbench",
        "--longbench_dir", args.longbench_dir,
        "--longbench_tasks", ",".join(TASKS),
        "--max_samples_per_task", str(args.max_samples),
        "--generation_max_new_tokens", str(args.max_new_tokens),
        "--smoke_mode", "1",
        "--max_smoke_samples", str(args.max_samples),
        "--output_dir", str(root / "runs" / method),
        "--isolated_run_id", method,
        "--block_strategy", "fixed",
        "--block_size", "64",
        "--prune_layers", "9,18,26",
        "--retain_tokens", "12288,6144,4096",
        "--preserve_first_block",
        "--temperature", "0",
        "--top_p", "1.0",
        "--enable_cpu_offload",
        "--enable_async_kv",
        "--save_outputs",
        "--rescue_policy", "add",
        "--rescue_score", str(spec["rescue_score"]),
        "--rescue_ratio", str(args.rescue_ratio),
        "--rescue_audit_path", str(root / "audit" / f"{method}.jsonl"),
        "--seed", "0",
        "--fd_alpha_o", "1.0",
        "--fd_beta_h", "0.0",
        "--fd_gamma_s", "0.0",
    ]
    if spec["structured"]:
        command.extend([
            "--rescue_granularity", "microblock",
            "--microblock_size", "32",
            "--strict_structured_path_check", "true",
            "--dump_candidate_scores", "1",
            "--dump_final_selection", "1",
        ])
    for key, value in spec.items():
        if key not in {"rescue_score", "structured"}:
            command.extend([f"--{key}", str(value)])
    return command


def _score_table(root: Path) -> Dict[str, Dict[str, Dict[str, Any]]]:
    scores: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for method in METHODS:
        path = root / "runs" / method / "longbench_scores.csv"
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                task = str(row.get("dataset", ""))
                scores[method][task] = {
                    "score": _as_float(row.get("score")),
                    "num_samples": int(_as_float(row.get("num_samples"))),
                    "status": row.get("status", ""),
                    "error": row.get("error", ""),
                }
    return scores


def _build_summary(root: Path) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
    summary_dir = root / "summary"
    scores = _score_table(root)
    task_scores: Dict[str, Dict[str, float]] = defaultdict(dict)
    breakdown: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    macros: Dict[str, float] = {}

    for method in METHODS:
        for task in TASKS:
            row = scores.get(method, {}).get(task, {})
            score = _as_float(row.get("score"))
            task_scores[method][task] = score
            breakdown.append({
                "method": method,
                "task": task,
                "score": round(score, 4),
                "num_samples": row.get("num_samples", 0),
                "status": row.get("status", "missing"),
                "error": row.get("error", ""),
            })
        ok_values = [
            task_scores[method][task]
            for task in TASKS
            if scores.get(method, {}).get(task, {}).get("status") == "ok"
        ]
        macro = statistics.mean(ok_values) if ok_values else 0.0
        macros[method] = macro
        run_status_path = root / "runs" / method / "run_status.json"
        run_status = (
            json.loads(run_status_path.read_text(encoding="utf-8"))
            if run_status_path.exists()
            else {}
        )
        summary_rows.append({
            "method": method,
            "task_macro_avg": round(macro, 4),
            "tasks_completed": len(ok_values),
            "samples_per_task": 32,
            "runner_fail": int(run_status.get("exit_code", 1) != 0),
            "delta_vs_fixed_drm_add": "",
        })

    fixed_macro = macros.get("fixed_drm_add", 0.0)
    for row in summary_rows:
        row["delta_vs_fixed_drm_add"] = round(
            _as_float(row["task_macro_avg"]) - fixed_macro,
            4,
        )

    delta_rows: List[Dict[str, Any]] = []
    for method in METHODS:
        delta_rows.append({
            "method": method,
            "task": "_macro",
            "score": round(macros.get(method, 0.0), 4),
            "fixed_drm_add_score": round(fixed_macro, 4),
            "delta_vs_fixed_drm_add": round(macros.get(method, 0.0) - fixed_macro, 4),
        })
        for task in TASKS:
            delta_rows.append({
                "method": method,
                "task": task,
                "score": round(task_scores[method][task], 4),
                "fixed_drm_add_score": round(task_scores["fixed_drm_add"][task], 4),
                "delta_vs_fixed_drm_add": round(
                    task_scores[method][task] - task_scores["fixed_drm_add"][task],
                    4,
                ),
            })

    budget_groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for method in METHODS:
        for row in _read_jsonl(root / "audit" / f"{method}.jsonl"):
            budget_groups[(method, str(row.get("task", "")))].append(row)
    budget_rows: List[Dict[str, Any]] = []
    for (method, task), rows in sorted(budget_groups.items()):
        errors = [
            abs(_as_float(row.get("budget_error_pct", row.get("selected_token_error_pct"))))
            for row in rows
        ]
        layer26_rows = [row for row in rows if int(row.get("layer_id", -1)) == 26]
        physical_superset_failures = sum(
            not (
                set(row.get("active_fixed_page_ids", []))
                <= set(row.get("final_block_ids_first100", []))
            )
            for row in layer26_rows
        )
        budget_rows.append({
            "method": method,
            "task": task,
            "audit_rows": len(rows),
            "mean_abs_budget_error_pct": round(statistics.mean(errors), 4) if errors else "",
            "max_abs_budget_error_pct": round(max(errors), 4) if errors else "",
            "all_rows_lt_5pct": bool(errors) and max(errors) < 5.0,
            "budget_warning_rows": sum(_as_bool(row.get("budget_warning")) for row in rows),
            "reported_add_only_contract_fail_rows": sum(
                not _as_bool(row.get("add_only_contract_ok", True)) for row in rows
            ),
            "layer26_physical_superset_checked_rows": len(layer26_rows),
            "layer26_physical_superset_fail_rows": physical_superset_failures,
        })

    _write_csv(
        summary_dir / "summary.csv",
        summary_rows,
        [
            "method", "task_macro_avg", "tasks_completed", "samples_per_task",
            "runner_fail", "delta_vs_fixed_drm_add",
        ],
    )
    _write_csv(
        summary_dir / "method_task_breakdown.csv",
        breakdown,
        ["method", "task", "score", "num_samples", "status", "error"],
    )
    _write_csv(
        summary_dir / "delta_vs_fixed_drm_add.csv",
        delta_rows,
        ["method", "task", "score", "fixed_drm_add_score", "delta_vs_fixed_drm_add"],
    )
    _write_csv(
        summary_dir / "budget_audit.csv",
        budget_rows,
        [
            "method", "task", "audit_rows", "mean_abs_budget_error_pct",
            "max_abs_budget_error_pct", "all_rows_lt_5pct",
            "budget_warning_rows", "reported_add_only_contract_fail_rows",
            "layer26_physical_superset_checked_rows",
            "layer26_physical_superset_fail_rows",
        ],
    )
    return task_scores, macros


def _layer26_rows(root: Path) -> Dict[str, Dict[str, Dict[str, Any]]]:
    rows: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for method in METHODS:
        for row in _read_jsonl(root / "audit" / f"{method}.jsonl"):
            if int(row.get("layer_id", -1)) != 26:
                continue
            key = f"{row.get('task')}:{row.get('sample_id')}"
            rows[method][key] = row
    return rows


def _page_ids(row: Mapping[str, Any]) -> Set[int]:
    values = (
        row.get("selected_parent_page_ids")
        or row.get("rescued_fixed_page_ids")
        or row.get("physical_units_after_expansion")
        or row.get("rescued_ids")
        or row.get("rescued_virtual_ids_first50")
        or []
    )
    return {int(value) for value in values}


def _overlap_rows(
    layer26: Mapping[str, Mapping[str, Mapping[str, Any]]],
    left_method: str,
    right_method: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    keys = sorted(set(layer26.get(left_method, {})) | set(layer26.get(right_method, {})))
    for key in keys:
        left = _page_ids(layer26.get(left_method, {}).get(key, {}))
        right = _page_ids(layer26.get(right_method, {}).get(key, {}))
        task, sample_id = key.split(":", 1)
        rows.append({
            "task": task,
            "sample_id": sample_id,
            "method_a": left_method,
            "method_b": right_method,
            "jaccard": round(_jaccard(left, right), 4),
            "selection_differs": left != right,
            "method_a_page_ids": json.dumps(sorted(left)),
            "method_b_page_ids": json.dumps(sorted(right)),
        })
    return rows


def _prediction_rows(root: Path, method: str, task: str) -> Dict[str, Dict[str, Any]]:
    path = root / "runs" / method / "longbench_predictions" / f"{task}.jsonl"
    return {
        str(row.get("sample_id", "")): row
        for row in _read_jsonl(path)
        if row.get("sample_id")
    }


def _build_validation(
    root: Path,
    task_scores: Mapping[str, Mapping[str, float]],
    macros: Mapping[str, float],
) -> Dict[str, Any]:
    out = root / "real_path_validation"
    layer26 = _layer26_rows(root)

    marker_rows: List[Dict[str, Any]] = []
    for method in sorted(STRUCTURED):
        marker_path = root / "runs" / method / "path_marker.jsonl"
        for row in _read_jsonl(marker_path):
            if int(row.get("layer_id", -1)) == 26:
                marker_rows.append({**row, "source_file": str(marker_path)})
    _write_csv(
        out / "01_path_marker_audit.csv",
        marker_rows,
        [
            "sample_id", "task", "layer_id", "method",
            "entered_structured_selector", "entered_real_fixed_block_path",
            "entered_cache_construction_path", "used_microblock_first_selector",
            "used_candidate_score_dump", "used_final_selection_dump",
            "used_evidence_density_selector", "source_file",
        ],
    )

    micro_rows = [
        {
            "sample_id": row.get("sample_id", ""),
            "task": row.get("task", ""),
            "layer_id": row.get("layer_id", ""),
            "method": method,
            "selected_microblock_count": row.get("selected_microblock_count", 0),
            "selected_parent_page_count": row.get(
                "selected_parent_page_count",
                len(row.get("selected_parent_page_ids", [])),
            ),
            "final_cache_granularity": row.get("final_cache_granularity", ""),
            "expansion_ratio": row.get("expansion_ratio", ""),
        }
        for method in sorted(STRUCTURED)
        for row in layer26.get(method, {}).values()
    ]
    _write_csv(
        out / "02_microblock_count_audit.csv",
        micro_rows,
        [
            "sample_id", "task", "layer_id", "method",
            "selected_microblock_count", "selected_parent_page_count",
            "final_cache_granularity", "expansion_ratio",
        ],
    )

    overlap_fixed = _overlap_rows(
        layer26,
        "evidence_density_srm",
        "fixed_drm_add",
    )
    _write_csv(
        out / "03_selection_overlap_vs_fixed.csv",
        overlap_fixed,
        [
            "task", "sample_id", "method_a", "method_b", "jaccard",
            "selection_differs", "method_a_page_ids", "method_b_page_ids",
        ],
    )

    overlap_structured: List[Dict[str, Any]] = []
    for pair in (
        ("evidence_density_srm", "rh_pyramid_srm"),
        ("evidence_density_srm", "bridge_graph_srm"),
        ("rh_pyramid_srm", "bridge_graph_srm"),
    ):
        overlap_structured.extend(_overlap_rows(layer26, *pair))
    _write_csv(
        out / "04_selection_overlap_vs_structured.csv",
        overlap_structured,
        [
            "task", "sample_id", "method_a", "method_b", "jaccard",
            "selection_differs", "method_a_page_ids", "method_b_page_ids",
        ],
    )

    density_candidates = [
        row
        for row in _read_jsonl(
            root / "runs" / "evidence_density_srm" / "candidate_score_dump.jsonl"
        )
        if int(row.get("layer_id", -1)) == 26
    ]
    candidates_by_sample: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in density_candidates:
        candidates_by_sample[f"{row.get('task')}:{row.get('sample_id')}"].append(row)

    component_rows: List[Dict[str, Any]] = []
    for key, rows in sorted(candidates_by_sample.items()):
        by_page: Dict[int, Dict[str, Any]] = {}
        for row in rows:
            by_page.setdefault(int(row["page_id"]), row)
        if not by_page:
            continue
        density_top = max(
            by_page,
            key=lambda page: (_as_float(by_page[page].get("density_parent_score")), -page),
        )
        max_only_top = max(
            by_page,
            key=lambda page: (_as_float(by_page[page].get("density_max_score")), -page),
        )
        top = by_page[density_top]
        task, sample_id = key.split(":", 1)
        component_rows.append({
            "task": task,
            "sample_id": sample_id,
            "density_top_page": density_top,
            "max_only_top_page": max_only_top,
            "ranking_changed_vs_max_only": density_top != max_only_top,
            "density_parent_score": top.get("density_parent_score", ""),
            "density_max_score": top.get("density_max_score", ""),
            "density_high_count": top.get("density_high_count", ""),
            "density_continuity_bonus": top.get("density_continuity_bonus", ""),
            "density_span_coverage": top.get("density_span_coverage", ""),
            "density_threshold": top.get("density_threshold", ""),
            "density_threshold_mode": top.get("density_threshold_mode", ""),
            "density_count_contribution": top.get("density_count_contribution", ""),
            "density_continuity_contribution": top.get(
                "density_continuity_contribution", ""
            ),
            "density_coverage_contribution": top.get("density_coverage_contribution", ""),
            "density_redundancy_contribution": top.get(
                "density_redundancy_contribution", ""
            ),
            "redundancy_fallback_used": top.get("redundancy_fallback_used", ""),
        })
    _write_csv(
        out / "05_density_component_audit.csv",
        component_rows,
        [
            "task", "sample_id", "density_top_page", "max_only_top_page",
            "ranking_changed_vs_max_only", "density_parent_score",
            "density_max_score", "density_high_count",
            "density_continuity_bonus", "density_span_coverage",
            "density_threshold", "density_threshold_mode",
            "density_count_contribution", "density_continuity_contribution",
            "density_coverage_contribution", "density_redundancy_contribution",
            "redundancy_fallback_used",
        ],
    )

    score_table = _score_table(root)
    failure_rows = [
        {
            "method": method,
            "task": task,
            "status": score_table.get(method, {}).get(task, {}).get("status", "missing"),
            "score": score_table.get(method, {}).get(task, {}).get("score", ""),
            "num_samples": score_table.get(method, {}).get(task, {}).get("num_samples", 0),
            "error": score_table.get(method, {}).get(task, {}).get("error", ""),
        }
        for method in METHODS
        for task in TASKS
    ]
    _write_csv(
        out / "06_task_failure_breakdown.csv",
        failure_rows,
        ["method", "task", "status", "score", "num_samples", "error"],
    )

    musique_predictions = {
        method: _prediction_rows(root, method, "musique")
        for method in METHODS
    }
    musique_ids = sorted(set().union(*(set(rows) for rows in musique_predictions.values())))
    musique_cases: List[Dict[str, Any]] = []
    for sample_id in musique_ids:
        key = f"musique:{sample_id}"
        density_pages = _page_ids(layer26["evidence_density_srm"].get(key, {}))
        fixed_pages = _page_ids(layer26["fixed_drm_add"].get(key, {}))
        rh_pages = _page_ids(layer26["rh_pyramid_srm"].get(key, {}))
        bridge_pages = _page_ids(layer26["bridge_graph_srm"].get(key, {}))
        fixed_pred = musique_predictions["fixed_drm_add"].get(sample_id, {})
        density_pred = musique_predictions["evidence_density_srm"].get(sample_id, {})
        musique_cases.append({
            "sample_id": sample_id,
            "answers": json.dumps(fixed_pred.get("answers", []), ensure_ascii=False),
            "fixed_drm_add_pred": fixed_pred.get("pred", ""),
            "evidence_density_srm_pred": density_pred.get("pred", ""),
            "rh_pyramid_srm_pred": musique_predictions["rh_pyramid_srm"].get(
                sample_id, {}
            ).get("pred", ""),
            "bridge_graph_srm_pred": musique_predictions["bridge_graph_srm"].get(
                sample_id, {}
            ).get("pred", ""),
            "density_pred_differs_from_fixed": density_pred.get("pred", "")
            != fixed_pred.get("pred", ""),
            "density_vs_fixed_page_jaccard": round(
                _jaccard(density_pages, fixed_pages), 4
            ),
            "density_vs_rh_page_jaccard": round(_jaccard(density_pages, rh_pages), 4),
            "density_vs_bridge_page_jaccard": round(
                _jaccard(density_pages, bridge_pages), 4
            ),
            "density_page_ids": json.dumps(sorted(density_pages)),
            "fixed_page_ids": json.dumps(sorted(fixed_pages)),
        })
    _write_csv(
        out / "07_case_examples_musique.csv",
        musique_cases,
        [
            "sample_id", "answers", "fixed_drm_add_pred",
            "evidence_density_srm_pred", "rh_pyramid_srm_pred",
            "bridge_graph_srm_pred", "density_pred_differs_from_fixed",
            "density_vs_fixed_page_jaccard", "density_vs_rh_page_jaccard",
            "density_vs_bridge_page_jaccard", "density_page_ids", "fixed_page_ids",
        ],
    )

    density_markers = [
        row for row in marker_rows if row.get("method") == "evidence_density_srm"
    ]
    real_path_ok = bool(density_markers) and all(
        _as_bool(row.get("entered_structured_selector"))
        and _as_bool(row.get("entered_real_fixed_block_path"))
        and _as_bool(row.get("entered_cache_construction_path"))
        and _as_bool(row.get("used_microblock_first_selector"))
        and _as_bool(row.get("used_evidence_density_selector"))
        for row in density_markers
    )
    density_micro_counts = [
        int(_as_float(row.get("selected_microblock_count")))
        for row in micro_rows
        if row.get("method") == "evidence_density_srm"
    ]
    micro_ok = bool(density_micro_counts) and min(density_micro_counts) > 0
    dump_status: Dict[str, Dict[str, bool]] = {}
    for method in sorted(STRUCTURED):
        run_dir = root / "runs" / method
        dump_status[method] = {
            filename: (run_dir / filename).exists()
            and (run_dir / filename).stat().st_size > 0
            for filename in (
                "candidate_score_dump.jsonl",
                "final_selection_dump.jsonl",
                "path_marker.jsonl",
            )
        }
    dumps_ok = all(all(status.values()) for status in dump_status.values())

    fixed_js = [_as_float(row["jaccard"], 1.0) for row in overlap_fixed]
    density_rh = [
        row for row in overlap_structured
        if row["method_a"] == "evidence_density_srm"
        and row["method_b"] == "rh_pyramid_srm"
    ]
    density_bridge = [
        row for row in overlap_structured
        if row["method_a"] == "evidence_density_srm"
        and row["method_b"] == "bridge_graph_srm"
    ]
    rh_bridge = [
        row for row in overlap_structured
        if row["method_a"] == "rh_pyramid_srm"
        and row["method_b"] == "bridge_graph_srm"
    ]
    structured_js = [
        _as_float(row["jaccard"], 1.0)
        for row in density_rh + density_bridge
    ]
    rh_bridge_js = [_as_float(row["jaccard"], 1.0) for row in rh_bridge]
    differs_fixed = any(value < 1.0 for value in fixed_js)
    differs_structured = any(value < 1.0 for value in structured_js)
    avoids_collapse = differs_structured

    component_contributions_ok = bool(component_rows) and all(
        any(
            abs(_as_float(row.get(field))) > 1e-9
            for row in component_rows
        )
        for field in (
            "density_count_contribution",
            "density_continuity_contribution",
            "density_coverage_contribution",
        )
    )
    ranking_changed_count = sum(
        _as_bool(row.get("ranking_changed_vs_max_only"))
        for row in component_rows
    )

    fixed_macro = macros.get("fixed_drm_add", 0.0)
    density_macro = macros.get("evidence_density_srm", 0.0)
    macro_delta = density_macro - fixed_macro
    task_deltas = {
        task: task_scores["evidence_density_srm"][task]
        - task_scores["fixed_drm_add"][task]
        for task in TASKS
    }
    improved = [task for task, delta in task_deltas.items() if delta > 0]
    degraded = [task for task, delta in task_deltas.items() if delta < 0]
    musique_delta = task_deltas["musique"]
    # Keep the established structured-rescue 32->64 diagnostic tolerance.
    macro_not_clearly_worse = macro_delta >= -0.30
    quality_gate = macro_not_clearly_worse or musique_delta > 0.0
    justify_64 = (
        real_path_ok
        and micro_ok
        and dumps_ok
        and differs_fixed
        and differs_structured
        and component_contributions_ok
        and quality_gate
    )

    recommendation = [
        "# Evidence-density 32-sample Qwen diagnostic recommendation",
        "",
        "This is a 32-sample-per-task diagnostic gate, not a final quality claim.",
        "",
        "## Required Answers",
        "",
        f"1. Evidence-density remained wired into the real fixed-block path: **{real_path_ok}** "
        f"({len(density_markers)} layer-26 markers).",
        f"2. `selected_microblock_count > 0` across real layer-26 samples: **{micro_ok}** "
        f"(min={min(density_micro_counts) if density_micro_counts else 0}, "
        f"mean={round(statistics.mean(density_micro_counts), 3) if density_micro_counts else 0}, "
        f"n={len(density_micro_counts)}).",
        f"3. Candidate/final/path dumps are complete for all structured methods: "
        f"**{dumps_ok}** (`{json.dumps(dump_status, sort_keys=True)}`).",
        f"4. Parent-page diversity vs `fixed_drm_add`: **{differs_fixed}** "
        f"(min/mean Jaccard={round(min(fixed_js), 4) if fixed_js else 1.0}/"
        f"{round(statistics.mean(fixed_js), 4) if fixed_js else 1.0}).",
        f"5. Parent-page diversity vs RH/Bridge: **{differs_structured}** "
        f"(min/mean Jaccard={round(min(structured_js), 4) if structured_js else 1.0}/"
        f"{round(statistics.mean(structured_js), 4) if structured_js else 1.0}).",
        f"6. Density scoring avoids complete RH/Bridge parent-page collapse: "
        f"**{avoids_collapse}**. RH-vs-Bridge mean Jaccard is "
        f"{round(statistics.mean(rh_bridge_js), 4) if rh_bridge_js else 1.0}; "
        f"density ranking differs from max-only on {ranking_changed_count}/"
        f"{len(component_rows)} audited samples.",
        f"7. Macro comparison: Evidence-density={density_macro:.4f}, "
        f"fixed DRM-Add={fixed_macro:.4f}, delta={macro_delta:+.4f}.",
        f"8. Improved tasks: `{','.join(improved) or 'none'}`. "
        f"Degraded tasks: `{','.join(degraded) or 'none'}`. "
        f"Musique delta={musique_delta:+.4f}.",
        f"9. A 64-sample rerun is justified: **{justify_64}**.",
        "",
        "## Gate Detail",
        "",
        f"- Density count/continuity/coverage contributions are nonzero: "
        f"**{component_contributions_ok}**.",
        f"- Macro is treated as not clearly worse using the repository's "
        f"existing 32->64 tolerance (delta >= -0.30): "
        f"**{macro_not_clearly_worse}**.",
        f"- Quality continuation gate (macro not clearly worse OR musique improves): "
        f"**{quality_gate}**.",
        "",
        "## Decision",
        "",
        (
            "The 32-sample diagnostic passes the continuation gate. A separate "
            "64-sample diagnostic is justified, but was not started."
            if justify_64
            else
            "The 32-sample diagnostic does not pass the continuation gate. "
            "Do not start a 64-sample rerun."
        ),
        "",
        "Do not claim final success from this diagnostic. Do not interpret these "
        "results as an algorithmic failure of RH-Pyramid or BridgeGraph.",
    ]
    (out / "08_recommendation.md").write_text(
        "\n".join(recommendation) + "\n",
        encoding="utf-8",
    )
    return {
        "real_path_ok": real_path_ok,
        "micro_ok": micro_ok,
        "dumps_ok": dumps_ok,
        "differs_fixed": differs_fixed,
        "differs_structured": differs_structured,
        "component_contributions_ok": component_contributions_ok,
        "density_macro": round(density_macro, 4),
        "fixed_macro": round(fixed_macro, 4),
        "macro_delta": round(macro_delta, 4),
        "musique_delta": round(musique_delta, 4),
        "justify_64": justify_64,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--model_path", default=DEFAULT_MODEL)
    parser.add_argument("--longbench_dir", default=DEFAULT_LONGBENCH)
    parser.add_argument("--max_samples", type=int, default=32)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--rescue_ratio", type=float, default=0.05)
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--audit_only", action="store_true")
    args = parser.parse_args()

    if args.max_samples != 32:
        raise ValueError("This diagnostic is intentionally fixed at 32 samples per task.")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = (
        Path(args.output_dir)
        if args.output_dir
        else REPO_ROOT / "results" / f"evidence_density_qwen32_{stamp}"
    )
    (root / "runs").mkdir(parents=True, exist_ok=True)
    (root / "audit").mkdir(exist_ok=True)
    requested = [method.strip() for method in args.methods.split(",") if method.strip()]
    unknown = sorted(set(requested) - set(METHODS))
    if unknown:
        raise ValueError(f"Unsupported methods: {unknown}")
    if set(requested) != set(METHODS):
        raise ValueError(f"This gate requires all methods: {METHODS}")

    manifest: Dict[str, Any] = {
        "output_root": str(root),
        "model_path": args.model_path,
        "longbench_dir": args.longbench_dir,
        "tasks": TASKS,
        "sample_limit": args.max_samples,
        "methods": requested,
        "max_new_tokens": args.max_new_tokens,
        "rescue_ratio": args.rescue_ratio,
        "density_defaults": {
            key: value
            for key, value in _method_spec("evidence_density_srm").items()
            if key not in {"rescue_score", "structured"}
        },
        "runs": [],
    }
    existing_config = root / "config.json"
    if args.audit_only and existing_config.exists():
        existing_manifest = json.loads(existing_config.read_text(encoding="utf-8"))
        manifest["runs"] = existing_manifest.get("runs", [])
    if not args.audit_only:
        for method in requested:
            status_path = root / "runs" / method / "run_status.json"
            if args.skip_existing and status_path.exists():
                status = json.loads(status_path.read_text(encoding="utf-8"))
                manifest["runs"].append(status)
                print(f"===== SKIP {method} =====", flush=True)
                continue
            print(f"===== RUN {method} =====", flush=True)
            started = time.time()
            log_path = root / "runs" / method / "run.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("w", encoding="utf-8") as log:
                process = subprocess.run(
                    _command(root, method, args),
                    cwd=str(REPO_ROOT),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            status = {
                "method": method,
                "exit_code": process.returncode,
                "elapsed_s": round(time.time() - started, 2),
                "log_path": str(log_path),
            }
            status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
            manifest["runs"].append(status)
            print(json.dumps(status), flush=True)
            if process.returncode != 0:
                manifest["status"] = "failed"
                (root / "config.json").write_text(
                    json.dumps(manifest, indent=2),
                    encoding="utf-8",
                )
                raise RuntimeError(f"{method} failed; inspect {log_path}")

    task_scores, macros = _build_summary(root)
    validation = _build_validation(root, task_scores, macros)
    manifest["validation"] = validation
    manifest["status"] = "complete"
    (root / "config.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(f"OUT_ROOT={root}", flush=True)


if __name__ == "__main__":
    main()
