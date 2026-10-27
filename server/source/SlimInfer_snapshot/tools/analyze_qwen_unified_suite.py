#!/usr/bin/env python3
"""Build the CPU-only six-method Qwen analysis package after Stage 2."""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

_REPO_BOOTSTRAP = Path(__file__).resolve().parents[1]
if str(_REPO_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(_REPO_BOOTSTRAP))

from tools.consolidate_official_repro_results import per_sample_score
from tools.qwen_pipeline_common import (
    COUNTS,
    PROTOCOL_ID,
    QWEN_DRM_ROOT,
    REPO_ROOT,
    RUN_DIR_BY_METHOD,
    TASKS,
    atomic_write_csv,
    atomic_write_json,
    atomic_write_text,
    canonical_sample_identity,
    iter_jsonl,
    load_json,
    load_task_samples,
    now_iso,
    per_task_score,
    prediction_status,
    sha256_file,
    stage_decision_path,
    task_macro,
)


METHODS = (
    "sliminfer_original", "cc_sliminfer_v1", "drm_swap_original_k",
    "sliminfer_larger_km", "sliminfer_random_add_km", "sliminfer_drm_add_km",
)
BUDGET = {
    method: ("K+M" if method in {
        "sliminfer_larger_km", "sliminfer_random_add_km", "sliminfer_drm_add_km"
    } else "K")
    for method in METHODS
}
COMPARISONS = (
    ("cc_sliminfer_v1", "sliminfer_original"),
    ("drm_swap_original_k", "sliminfer_original"),
    ("sliminfer_drm_add_km", "sliminfer_larger_km"),
    ("sliminfer_drm_add_km", "sliminfer_random_add_km"),
)
CATEGORIES = {
    "single_document_qa": ("qasper", "multifieldqa_en"),
    "multi_document_qa": ("hotpotqa", "2wikimqa", "musique"),
    "summarization": ("gov_report", "qmsum", "multi_news", "vcsum"),
    "few_shot_learning": ("trec", "triviaqa", "samsum", "lsht"),
    "synthetic_tasks": ("passage_count", "passage_retrieval_en"),
    "code_completion": ("lcc", "repobench-p"),
}
K_BY_LAYER = {9: 192, 18: 96, 26: 64}
M_BY_LAYER = {9: 10, 18: 5, 26: 3}
SELECTION_AUDIT_REVISION = "effective-budget-v2"


def _cc_prediction_path(run_root: Path, task: str) -> Path:
    return run_root / "tasks" / task / "run/longbench_predictions" / f"{task}.jsonl"


def _source_path(method: str, cc_root: Path, task: str) -> Path:
    if method == "cc_sliminfer_v1":
        return _cc_prediction_path(cc_root, task)
    return QWEN_DRM_ROOT / RUN_DIR_BY_METHOD[method] / "longbench_predictions" / f"{task}.jsonl"


def load_method_rows(method: str, cc_root: Path) -> Dict[str, List[Dict[str, Any]]]:
    result: Dict[str, List[Dict[str, Any]]] = {}
    for task in TASKS:
        path = _source_path(method, cc_root, task)
        rows = list(iter_jsonl(path)) if path.exists() else []
        samples = load_task_samples(task)
        normalized: List[Dict[str, Any]] = []
        for position, row in enumerate(rows):
            sample_index = int(row.get("sample_index", position))
            sample_id, content_hash = canonical_sample_identity(task, sample_index, samples[sample_index])
            normalized.append({
                **row,
                "task": task,
                "sample_index": sample_index,
                "sample_id": sample_id,
                "content_sha256": content_hash,
                "status": prediction_status(row),
                "prediction": str(row.get("prediction", row.get("pred", ""))),
                "pred": str(row.get("pred", row.get("prediction", ""))),
            })
        result[task] = normalized
    return result


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _metric_from_sample_scores(values: Sequence[float]) -> float:
    """Exactly matches LongBench's rounded mean after elementary metrics are computed."""
    return round(sum(float(value) for value in values) / len(values), 2)


def paired_bootstrap(
    variant: str,
    baseline: str,
    scores: Mapping[Tuple[str, str, str], float],
    valid_ids: Mapping[Tuple[str, str], Sequence[str]],
    *,
    seed: int = 0,
    replicates: int = 10000,
) -> Dict[str, Any]:
    rng = random.Random(seed)
    pair_name = f"{variant}__{baseline}"
    paired: Dict[str, List[Tuple[float, float]]] = {}
    for task in TASKS:
        paired[task] = [
            (scores[(variant, task, sample_id)], scores[(baseline, task, sample_id)])
            for sample_id in valid_ids[(task, pair_name)]
        ]
    observed_by_task = {
        task: _metric_from_sample_scores([left for left, _ in values])
        - _metric_from_sample_scores([right for _, right in values])
        for task, values in paired.items()
    }
    observed = sum(observed_by_task.values()) / len(TASKS)
    boot: List[float] = []
    for _ in range(replicates):
        deltas = []
        for task in TASKS:
            values = paired[task]
            sampled = [values[rng.randrange(len(values))] for _ in values]
            variant_score = _metric_from_sample_scores([left for left, _ in sampled])
            baseline_score = _metric_from_sample_scores([right for _, right in sampled])
            deltas.append(variant_score - baseline_score)
        boot.append(sum(deltas) / len(TASKS))
    return {
        "variant": variant,
        "baseline": baseline,
        "seed": seed,
        "replicates": replicates,
        "resampling_unit": "paired samples within each task",
        "task_metric_recomputed_each_replicate": True,
        "macro": "simple arithmetic mean of 17 task metrics",
        "paired_samples": sum(len(values) for values in paired.values()),
        "per_task_n": {task: len(values) for task, values in paired.items()},
        "observed_per_task_delta": observed_by_task,
        "observed_delta": observed,
        "bootstrap_mean_delta": sum(boot) / len(boot),
        "ci95": [_percentile(boot, 0.025), _percentile(boot, 0.975)],
        "p_delta_gt_zero": sum(value > 0 for value in boot) / len(boot),
    }


def _score_rows(task: str, rows: Sequence[Mapping[str, Any]]) -> float:
    if not rows:
        raise RuntimeError(f"No rows available for {task}")
    return per_task_score(task, rows)


def _historical_selection_audit(
    method: str,
    selection_path: Path,
    rescue_path: Path,
) -> Dict[str, Any]:
    rescue_by_key: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    for row in iter_jsonl(rescue_path):
        if row.get("phase") != "prefill":
            continue
        key = (str(row.get("task")), str(row.get("sample_id")), int(row.get("layer_id", -1)))
        if key in rescue_by_key:
            raise RuntimeError(f"Duplicate prefill rescue row for {method}: {key}")
        rescue_by_key[key] = row

    result = {
        "method": method, "source": str(selection_path), "events": 0,
        "budget_bad": 0, "sink_bad": 0, "duplicate_bad": 0, "invalid_bad": 0,
        "schedule_clamped": 0, "candidate_exhausted_clamps": 0,
        "clamp_validation_bad": 0, "add_preservation_bad": 0,
        "swap_parity_bad": 0, "rescue_metadata_missing": 0,
        "samples_with_events": None,
        "budget_rule": "min(scheduled_budget, physical_block_count)",
        "audit_revision": SELECTION_AUDIT_REVISION,
    }
    seen_samples = set()
    for row in iter_jsonl(selection_path):
        result["events"] += 1
        seen_samples.add((row.get("task"), row.get("sample_index")))
        selected = list(row.get("selected_block_ids", []))
        baseline = list(row.get("silm_baseline_ids", []))
        selected_set = set(selected)
        baseline_set = set(baseline)
        layer_value = row.get("abs_layer") if row.get("abs_layer") is not None else row.get("layer_idx")
        layer = int(layer_value) if layer_value is not None else -1
        key = (str(row.get("task")), str(row.get("sample_id")), layer)
        rescue = rescue_by_key.get(key)
        k = K_BY_LAYER.get(layer)
        scheduled_m = M_BY_LAYER.get(layer, 0)

        if rescue is None or k is None:
            result["rescue_metadata_missing"] += 1
            result["budget_bad"] += 1
        else:
            total_blocks = int(rescue.get("block_count_total", rescue.get("num_original_blocks", 0)) or 0)
            scheduled_budget = k + scheduled_m if BUDGET[method] == "K+M" else k
            expected = min(scheduled_budget, total_blocks)
            if len(selected) != expected:
                result["budget_bad"] += 1

            if method in {"sliminfer_random_add_km", "sliminfer_drm_add_km"}:
                effective_k = min(k, total_blocks)
                effective_m = min(scheduled_m, max(0, total_blocks - effective_k))
                added = selected_set - baseline_set
                runtime_budget_ok = (
                    int(rescue.get("K", -1)) == effective_k
                    and int(rescue.get("M", -1)) == effective_m
                    and int(rescue.get("rescued_count", -1)) == effective_m
                    and len(baseline) == effective_k
                    and len(added) == effective_m
                )
                if not runtime_budget_ok:
                    result["budget_bad"] += 1
                if not baseline_set.issubset(selected_set):
                    result["add_preservation_bad"] += 1
                    result["budget_bad"] += 1
                if effective_m < scheduled_m:
                    result["schedule_clamped"] += 1
                    clamp_ok = expected == total_blocks and selected_set == set(range(total_blocks))
                    if clamp_ok:
                        result["candidate_exhausted_clamps"] += 1
                    else:
                        result["clamp_validation_bad"] += 1
                        result["budget_bad"] += 1

        if int(row.get("sink_violation", 0)):
            result["sink_bad"] += 1
        if len(selected) != len(selected_set):
            result["duplicate_bad"] += 1
        if any(not isinstance(value, int) or value < 0 for value in selected):
            result["invalid_bad"] += 1
        if method == "drm_swap_original_k":
            if len(selected_set - baseline_set) != len(baseline_set - selected_set):
                result["swap_parity_bad"] += 1
                result["budget_bad"] += 1
    result["samples_with_events"] = len(seen_samples)
    return result


def _selection_audit(method: str, cc_root: Path) -> Dict[str, Any]:
    if method == "cc_sliminfer_v1":
        task_reports = [load_json(cc_root / "tasks" / task / "TASK_DECISION.json") for task in TASKS]
        return {
            "method": method,
            "source": "Stage 2 feature-complete runtime trace",
            "events": sum(int(row.get("trace", {}).get("events", 0)) for row in task_reports),
            "budget_bad": sum(int(row.get("trace", {}).get("budget_failures", 0)) for row in task_reports),
            "sink_bad": sum(int(row.get("trace", {}).get("sink_failures", 0)) for row in task_reports),
            "duplicate_bad": sum(int(row.get("trace", {}).get("duplicate_failures", 0)) for row in task_reports),
            "invalid_bad": sum(int(row.get("trace", {}).get("invalid_id_failures", 0)) for row in task_reports),
            "samples_with_events": sum(int(row.get("trace", {}).get("samples_with_events", 0)) for row in task_reports),
            "schedule_clamped": 0, "candidate_exhausted_clamps": 0,
            "clamp_validation_bad": 0, "add_preservation_bad": 0,
            "swap_parity_bad": 0, "rescue_metadata_missing": 0,
            "budget_rule": "runtime trace exact selected-count contract",
            "audit_revision": SELECTION_AUDIT_REVISION,
        }
    path = QWEN_DRM_ROOT / "audits" / f"{method}_selection_audit.jsonl"
    rescue_path = QWEN_DRM_ROOT / "audits" / f"{method}_rescue_policy_audit.jsonl"
    return _historical_selection_audit(method, path, rescue_path)


def _write_checksums(root: Path) -> None:
    paths = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS.txt"
    )
    atomic_write_text(
        root / "SHA256SUMS.txt",
        "".join(f"{sha256_file(path)}  {path.relative_to(root)}\n" for path in paths),
    )


def analyze(pipeline_dir: Path, *, dry_run: bool = False) -> Dict[str, Any]:
    stage2_path = stage_decision_path(pipeline_dir, 2)
    if not stage2_path.exists() or load_json(stage2_path).get("decision") != "PASS":
        raise RuntimeError("Stage 02 is not PASS; refusing six-method analysis")
    cc_root = Path(load_json(stage2_path)["run_root"])
    existing_plan_path = pipeline_dir / "STAGE_03_ANALYSIS_PLAN.json"
    if existing_plan_path.exists():
        prior_root = Path(load_json(existing_plan_path)["export_root"])
        prior_complete = prior_root / "ANALYSIS_COMPLETE.json"
        if prior_complete.exists() and load_json(prior_complete).get("selection_audit_revision") != SELECTION_AUDIT_REVISION:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_root = REPO_ROOT / "results_exports" / f"qwen_unified_six_method_{stamp}_budget_v2"
        else:
            export_root = prior_root
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_root = REPO_ROOT / "results_exports" / f"qwen_unified_six_method_{stamp}"
    plan = {
        "created_at": now_iso(), "protocol_id": PROTOCOL_ID,
        "methods": list(METHODS), "comparisons": [list(item) for item in COMPARISONS],
        "cc_root": str(cc_root), "historical_drm_root": str(QWEN_DRM_ROOT),
        "export_root": str(export_root), "bootstrap_replicates": 10000, "seed": 0,
    }
    atomic_write_json(existing_plan_path, plan)
    if dry_run:
        return {"stage": 3, "decision": "NOT_RUN", **plan}
    if export_root.exists() and (export_root / "ANALYSIS_COMPLETE.json").exists():
        complete = load_json(export_root / "ANALYSIS_COMPLETE.json")
        decision = {
            "stage": 3, "name": "six_method_analysis", "created_at": now_iso(),
            "decision": "PASS", "export_root": str(export_root),
            "global_common_valid": complete["global_common_valid"], "next_stage": 4,
            "resumed": True,
        }
        atomic_write_json(stage_decision_path(pipeline_dir, 3), decision)
        return decision
    if export_root.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        incomplete_root = export_root.with_name(f"{export_root.name}.incomplete_{stamp}")
        export_root.rename(incomplete_root)
        plan["recovered_incomplete_root"] = str(incomplete_root)
        atomic_write_json(existing_plan_path, plan)
    all_rows = {method: load_method_rows(method, cc_root) for method in METHODS}
    completeness = {
        method: all(len(all_rows[method][task]) == COUNTS[task] for task in TASKS)
        for method in METHODS
    }
    if not all(completeness.values()):
        raise RuntimeError(f"Incomplete six-method inputs: {completeness}")
    export_root.mkdir(parents=True, exist_ok=False)

    score_rows: List[Dict[str, Any]] = []
    score_map: Dict[Tuple[str, str, str], float] = {}
    status_map: Dict[Tuple[str, str, str], str] = {}
    row_map: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    raw_task_rows: List[Dict[str, Any]] = []
    raw_scores: Dict[Tuple[str, str], float] = {}
    for method in METHODS:
        method_scores: Dict[str, float] = {}
        for task in TASKS:
            rows = all_rows[method][task]
            task_score = _score_rows(task, rows)
            method_scores[task] = task_score
            raw_scores[(method, task)] = task_score
            raw_task_rows.append({
                "analysis_set": "raw_full", "method": method, "budget": BUDGET[method],
                "task": task, "score": task_score, "samples": len(rows),
                "non_ok": sum(row["status"] != "ok" for row in rows),
            })
            for row in rows:
                sample_score = per_sample_score(
                    task, row["pred"], row.get("answers", []), row.get("all_classes", [])
                )
                key = (method, task, row["sample_id"])
                score_map[key] = sample_score
                status_map[key] = row["status"]
                row_map[key] = row
                score_rows.append({
                    "analysis_set": "raw_full", "method": method, "budget": BUDGET[method],
                    "task": task, "sample_index": row["sample_index"],
                    "sample_id": row["sample_id"], "status": row["status"],
                    "score": sample_score, "input_tokens": row.get("input_tokens", row.get("length")),
                    "prediction": row["pred"],
                })
        raw_task_rows.append({
            "analysis_set": "raw_full", "method": method, "budget": BUDGET[method],
            "task": "17_task_macro", "score": task_macro(method_scores), "samples": 3950,
            "non_ok": sum(row["status"] != "ok" for task in TASKS for row in all_rows[method][task]),
        })

    global_valid: Dict[str, List[str]] = {}
    pair_valid: Dict[Tuple[str, str], Sequence[str]] = {}
    for task in TASKS:
        ids = [row["sample_id"] for row in all_rows[METHODS[0]][task]]
        global_valid[task] = [
            sample_id for sample_id in ids
            if all(status_map[(method, task, sample_id)] == "ok" for method in METHODS)
        ]
        for variant, baseline in COMPARISONS:
            pair_valid[(task, f"{variant}__{baseline}")] = [
                sample_id for sample_id in ids
                if status_map[(variant, task, sample_id)] == "ok"
                and status_map[(baseline, task, sample_id)] == "ok"
            ]

    subset_rows: List[Dict[str, Any]] = []
    for method in METHODS:
        global_scores: Dict[str, float] = {}
        for task in TASKS:
            rows = [row_map[(method, task, sample_id)] for sample_id in global_valid[task]]
            value = _score_rows(task, rows)
            global_scores[task] = value
            subset_rows.append({
                "analysis_set": "global_common_valid", "method": method,
                "baseline": "", "task": task, "score": value, "samples": len(rows),
            })
        subset_rows.append({
            "analysis_set": "global_common_valid", "method": method,
            "baseline": "", "task": "17_task_macro",
            "score": task_macro(global_scores),
            "samples": sum(len(ids) for ids in global_valid.values()),
        })

    comparison_task_rows: List[Dict[str, Any]] = []
    wlt_rows: List[Dict[str, Any]] = []
    sensitivity_rows: List[Dict[str, Any]] = []
    for variant, baseline in COMPARISONS:
        task_deltas: Dict[str, float] = {}
        total_wins = total_losses = total_ties = total_changed = 0
        for task in TASKS:
            ids = pair_valid[(task, f"{variant}__{baseline}")]
            left_rows = [row_map[(variant, task, sample_id)] for sample_id in ids]
            right_rows = [row_map[(baseline, task, sample_id)] for sample_id in ids]
            left_score = _score_rows(task, left_rows)
            right_score = _score_rows(task, right_rows)
            delta = left_score - right_score
            task_deltas[task] = delta
            wins = sum(score_map[(variant, task, sid)] > score_map[(baseline, task, sid)] for sid in ids)
            losses = sum(score_map[(variant, task, sid)] < score_map[(baseline, task, sid)] for sid in ids)
            ties = len(ids) - wins - losses
            changed = sum(
                str(row_map[(variant, task, sid)]["pred"]) != str(row_map[(baseline, task, sid)]["pred"])
                for sid in ids
            )
            total_wins += wins
            total_losses += losses
            total_ties += ties
            total_changed += changed
            comparison_task_rows.append({
                "variant": variant, "baseline": baseline, "task": task,
                "samples": len(ids), "variant_score": left_score,
                "baseline_score": right_score, "delta": delta,
            })
            wlt_rows.append({
                "variant": variant, "baseline": baseline, "task": task,
                "samples": len(ids), "wins": wins, "losses": losses,
                "ties": ties, "changed_predictions": changed,
            })
        macro_delta = sum(task_deltas.values()) / len(TASKS)
        comparison_task_rows.append({
            "variant": variant, "baseline": baseline, "task": "17_task_macro",
            "samples": sum(len(pair_valid[(task, f"{variant}__{baseline}")]) for task in TASKS),
            "variant_score": "", "baseline_score": "", "delta": macro_delta,
        })
        wlt_rows.append({
            "variant": variant, "baseline": baseline, "task": "all_tasks",
            "samples": total_wins + total_losses + total_ties,
            "wins": total_wins, "losses": total_losses, "ties": total_ties,
            "changed_predictions": total_changed,
        })
        for removed in TASKS:
            remaining = [value for task, value in task_deltas.items() if task != removed]
            sensitivity_rows.append({
                "variant": variant, "baseline": baseline,
                "analysis": "leave_one_task_out", "removed": removed,
                "macro_delta": sum(remaining) / len(remaining),
            })
        best_task = max(task_deltas, key=task_deltas.get)
        for label, removed_tasks in (
            ("remove_musique", {"musique"}),
            ("remove_best_task", {best_task}),
        ):
            remaining = [value for task, value in task_deltas.items() if task not in removed_tasks]
            sensitivity_rows.append({
                "variant": variant, "baseline": baseline, "analysis": label,
                "removed": ",".join(sorted(removed_tasks)),
                "macro_delta": sum(remaining) / len(remaining),
            })

    bootstrap = {
        f"{variant}_vs_{baseline}": paired_bootstrap(
            variant, baseline, score_map, pair_valid, seed=0, replicates=10000
        )
        for variant, baseline in COMPARISONS
    }
    category_rows = []
    for method in METHODS:
        for category, tasks in CATEGORIES.items():
            category_rows.append({
                "method": method, "budget": BUDGET[method], "category": category,
                "score": sum(raw_scores[(method, task)] for task in tasks) / len(tasks),
                "tasks": ",".join(tasks),
            })
    oom_rows = []
    for method in METHODS:
        for task in TASKS:
            for row in all_rows[method][task]:
                if row["status"] != "ok":
                    oom_rows.append({
                        "method": method, "task": task, "sample_index": row["sample_index"],
                        "sample_id": row["sample_id"], "status": row["status"],
                        "error": row.get("error", ""),
                    })
    selection_rows = [_selection_audit(method, cc_root) for method in METHODS]
    source_rows = [
        {
            "method": method, "task": task, "path": str(_source_path(method, cc_root, task)),
            "sha256": sha256_file(_source_path(method, cc_root, task)),
            "provenance": "fresh_stage_2" if method == "cc_sliminfer_v1" else "canonical_read_only",
        }
        for method in METHODS for task in TASKS
    ]
    fields = (
        "sample_index", "sample_id", "content_sha256", "prompt_sha256",
        "input_token_ids_sha256", "generation_config_sha256", "status",
        "input_tokens", "output_tokens", "source_config_hash", "implementation_revision",
    )
    field_rows = []
    for method in METHODS:
        method_rows = [row for task in TASKS for row in all_rows[method][task]]
        for field in fields:
            covered = sum(row.get(field) is not None for row in method_rows)
            field_rows.append({
                "method": method, "field": field, "covered": covered,
                "total": len(method_rows), "coverage": covered / len(method_rows),
                "source": "runtime" if covered else "unavailable_historical",
            })

    atomic_write_csv(export_root / "scores_by_sample.csv", score_rows, [
        "analysis_set", "method", "budget", "task", "sample_index", "sample_id",
        "status", "score", "input_tokens", "prediction",
    ])
    atomic_write_csv(export_root / "scores_by_task.csv", raw_task_rows, [
        "analysis_set", "method", "budget", "task", "score", "samples", "non_ok",
    ])
    atomic_write_csv(export_root / "common_valid_scores.csv", subset_rows, [
        "analysis_set", "method", "baseline", "task", "score", "samples",
    ])
    atomic_write_csv(export_root / "pairwise_common_valid.csv", comparison_task_rows, [
        "variant", "baseline", "task", "samples", "variant_score", "baseline_score", "delta",
    ])
    atomic_write_csv(export_root / "win_loss_tie.csv", wlt_rows, [
        "variant", "baseline", "task", "samples", "wins", "losses", "ties", "changed_predictions",
    ])
    atomic_write_csv(export_root / "outlier_sensitivity.csv", sensitivity_rows, [
        "variant", "baseline", "analysis", "removed", "macro_delta",
    ])
    atomic_write_csv(export_root / "category_summary.csv", category_rows, [
        "method", "budget", "category", "score", "tasks",
    ])
    atomic_write_csv(export_root / "OOM_MANIFEST.csv", oom_rows, [
        "method", "task", "sample_index", "sample_id", "status", "error",
    ])
    atomic_write_csv(export_root / "BUDGET_SELECTION_AUDIT.csv", selection_rows, [
        "method", "source", "events", "samples_with_events", "budget_bad",
        "sink_bad", "duplicate_bad", "invalid_bad", "schedule_clamped",
        "candidate_exhausted_clamps", "clamp_validation_bad", "add_preservation_bad",
        "swap_parity_bad", "rescue_metadata_missing", "budget_rule", "audit_revision",
    ])
    atomic_write_csv(export_root / "ACTIVE_TRACE_COVERAGE.csv", selection_rows, [
        "method", "source", "events", "samples_with_events",
    ])
    atomic_write_csv(export_root / "FIELD_COVERAGE.csv", field_rows, [
        "method", "field", "covered", "total", "coverage", "source",
    ])
    atomic_write_csv(export_root / "SOURCE_MAP.csv", source_rows, [
        "method", "task", "path", "sha256", "provenance",
    ])
    atomic_write_json(export_root / "paired_bootstrap.json", bootstrap)
    atomic_write_json(export_root / "BOOTSTRAP_CONFIG.json", {
        "replicates": 10000, "seed": 0, "paired": True,
        "pairwise_common_valid": True, "within_task_resampling": True,
        "task_metric_recomputed": True, "macro": "simple_mean_17_tasks",
    })
    atomic_write_json(export_root / "GLOBAL_COMMON_VALID.json", {
        "total": sum(len(ids) for ids in global_valid.values()),
        "per_task": {task: len(ids) for task, ids in global_valid.items()},
        "excluded_union": 3950 - sum(len(ids) for ids in global_valid.values()),
    })
    atomic_write_json(export_root / "PAIRWISE_COMMON_VALID.json", {
        f"{variant}_vs_{baseline}": {
            "total": sum(len(pair_valid[(task, f"{variant}__{baseline}")]) for task in TASKS),
            "per_task": {
                task: len(pair_valid[(task, f"{variant}__{baseline}")]) for task in TASKS
            },
        }
        for variant, baseline in COMPARISONS
    })
    atomic_write_json(export_root / "PROTOCOL_ATTESTATION.json", load_json(pipeline_dir / "PROTOCOL_ATTESTATION.json"))
    atomic_write_json(export_root / "METHODS.json", {"methods": list(METHODS), "budgets": BUDGET})
    atomic_write_text(
        export_root / "README.md",
        "# Qwen Unified Six-Method Analysis\n\n"
        "CPU-recomputed under `ULB-v1-052cfbbeaa2cc42f`. Raw full-set, global common-valid, "
        "and pairwise common-valid results are distinct. K and K+M remain separate budget groups.\n",
    )
    atomic_write_text(
        export_root / "KNOWN_LIMITATIONS.md",
        "# Known Limitations\n\nHistorical five-method predictions predate canonical per-row prompt/token hashes; "
        "those fields are marked unavailable rather than reconstructed as runtime facts. Official LongBench "
        "does not provide gold supporting-fact spans, so no gold-evidence recall is reported.\n",
    )
    complete = {
        "created_at": now_iso(), "protocol_id": PROTOCOL_ID,
        "methods": list(METHODS), "samples_per_method": 3950,
        "global_common_valid": sum(len(ids) for ids in global_valid.values()),
        "oom_rows": len(oom_rows),
        "selection_audit_revision": SELECTION_AUDIT_REVISION,
    }
    atomic_write_json(export_root / "ANALYSIS_COMPLETE.json", complete)
    _write_checksums(export_root)
    decision = {
        "stage": 3, "name": "six_method_analysis", "created_at": now_iso(),
        "decision": "PASS", "export_root": str(export_root),
        "global_common_valid": complete["global_common_valid"], "next_stage": 4,
    }
    atomic_write_json(stage_decision_path(pipeline_dir, 3), decision)
    return decision


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = analyze(args.pipeline_dir.resolve(), dry_run=args.dry_run)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
