#!/usr/bin/env python3
"""CPU mechanism analyses and preregistered ablation availability audits."""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import statistics
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

_REPO_BOOTSTRAP = Path(__file__).resolve().parents[1]
if str(_REPO_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(_REPO_BOOTSTRAP))

from tools.qwen_pipeline_common import (
    PROTOCOL_ID,
    QWEN_DRM_ROOT,
    REPO_ROOT,
    TASKS,
    atomic_write_csv,
    atomic_write_json,
    atomic_write_text,
    canonical_sample_identity,
    iter_jsonl,
    load_json,
    load_task_samples,
    now_iso,
    stage_decision_path,
)
from tools.qwen_protocol_gate import run_protocol_gate


REPRESENTATIVE_TASKS = ("qasper", "hotpotqa", "2wikimqa", "musique", "gov_report", "repobench-p")


def _runs(values: Sequence[int]) -> List[List[int]]:
    ordered = sorted(set(int(value) for value in values))
    if not ordered:
        return []
    result = [[ordered[0]]]
    for value in ordered[1:]:
        if value == result[-1][-1] + 1:
            result[-1].append(value)
        else:
            result.append([value])
    return result


def _structure(prefix: str, values: Sequence[int]) -> Dict[str, Any]:
    ordered = sorted(set(int(value) for value in values))
    runs = _runs(ordered)
    gaps = [right - left for left, right in zip(ordered, ordered[1:])]
    span = ordered[-1] - ordered[0] + 1 if ordered else 0
    return {
        f"{prefix}_selected_count": len(ordered),
        f"{prefix}_contiguous_segment_count": len(runs),
        f"{prefix}_run_count": len(runs),
        f"{prefix}_fragmentation": len(runs) / max(1, len(ordered)),
        f"{prefix}_mean_gap": statistics.mean(gaps) if gaps else 0.0,
        f"{prefix}_median_gap": statistics.median(gaps) if gaps else 0.0,
        f"{prefix}_longest_contiguous_run": max((len(run) for run in runs), default=0),
        f"{prefix}_spatial_concentration": len(ordered) / max(1, span),
    }


def _nearest_distance(values: Sequence[int], reference: Sequence[int]) -> float:
    if not values or not reference:
        return 0.0
    return statistics.mean(min(abs(int(value) - int(other)) for other in reference) for value in values)


def _score_delta_map(pipeline_dir: Path, variant: str, baseline: str) -> Dict[Tuple[str, str], float]:
    stage3 = load_json(stage_decision_path(pipeline_dir, 3))
    path = Path(stage3["export_root"]) / "scores_by_sample.csv"
    values: Dict[Tuple[str, str, str], float] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["method"] in {variant, baseline}:
                values[(row["method"], row["task"], row["sample_id"])] = float(row["score"])
    return {
        (task, sample_id): value - values[(baseline, task, sample_id)]
        for (method, task, sample_id), value in values.items() if method == variant
    }


def _correlation(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) < 2 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return None
    mean_x, mean_y = statistics.mean(xs), statistics.mean(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denominator = math.sqrt(
        sum((x - mean_x) ** 2 for x in xs) * sum((y - mean_y) ** 2 for y in ys)
    )
    return numerator / denominator if denominator else None


def _cc_analysis(pipeline_dir: Path, root: Path) -> Dict[str, Any]:
    stage2 = load_json(stage_decision_path(pipeline_dir, 2))
    cc_root = Path(stage2["run_root"])
    deltas = _score_delta_map(pipeline_dir, "cc_sliminfer_v1", "sliminfer_original")
    aggregate: Dict[Tuple[str, str], Dict[str, Any]] = {}
    events = 0
    for task in TASKS:
        path = cc_root / "tasks" / task / "active_block_trace" / f"{task}.jsonl.gz"
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                event = json.loads(line)
                events += 1
                key = (task, str(event.get("sample_id")))
                item = aggregate.setdefault(key, {
                    "task": task, "sample_id": key[1], "sample_index": event.get("sample_index"),
                    "events": 0, "changed_events": 0, "jaccard_sum": 0.0,
                    "neighbor_coverage_sum": 0.0, "added_distance_sum": 0.0,
                    "removed_distance_sum": 0.0,
                })
                baseline = list(event.get("base_original_active_ids", []))
                selected = list(event.get("cc_active_ids", []))
                added = list(event.get("added_ids", []))
                removed = list(event.get("removed_ids", []))
                item["events"] += 1
                item["changed_events"] += int(bool(added or removed))
                item["jaccard_sum"] += float(event.get("jaccard_vs_silm", 1.0) or 0.0)
                item["neighbor_coverage_sum"] += (
                    sum(min((abs(value - base) for base in baseline), default=999) <= 2 for value in added)
                    / max(1, len(added))
                )
                item["added_distance_sum"] += _nearest_distance(added, baseline)
                item["removed_distance_sum"] += _nearest_distance(removed, selected)
                base_structure = _structure("base", baseline)
                final_structure = _structure("final", selected)
                for field, value in {**base_structure, **final_structure}.items():
                    item[field + "_sum"] = item.get(field + "_sum", 0.0) + float(value)
    rows: List[Dict[str, Any]] = []
    structure_fields = [
        f"{prefix}_{metric}" for prefix in ("base", "final") for metric in (
            "selected_count", "contiguous_segment_count", "run_count", "fragmentation",
            "mean_gap", "median_gap", "longest_contiguous_run", "spatial_concentration",
        )
    ]
    for key, item in aggregate.items():
        count = int(item["events"])
        row = {
            "task": item["task"], "sample_id": item["sample_id"],
            "sample_index": item["sample_index"], "events": count,
            "selection_changed_ratio": item["changed_events"] / count,
            "mean_jaccard": item["jaccard_sum"] / count,
            "neighbor_coverage": item["neighbor_coverage_sum"] / count,
            "mean_added_distance": item["added_distance_sum"] / count,
            "mean_removed_distance": item["removed_distance_sum"] / count,
            "sample_score_delta": deltas.get(key),
        }
        for field in structure_fields:
            row[field] = item.get(field + "_sum", 0.0) / count
            if field.startswith("final_"):
                base_field = "base_" + field[len("final_"):]
                row[field + "_delta"] = row[field] - item.get(base_field + "_sum", 0.0) / count
        rows.append(row)
    output_fields = [
        "task", "sample_id", "sample_index", "events", "selection_changed_ratio",
        "mean_jaccard", "neighbor_coverage", "mean_added_distance", "mean_removed_distance",
        "sample_score_delta", *structure_fields,
        *[field + "_delta" for field in structure_fields if field.startswith("final_")],
    ]
    atomic_write_csv(root / "cc_structure_by_sample.csv", rows, output_fields)
    valid = [row for row in rows if row["sample_score_delta"] is not None]
    correlations = {}
    for metric in (
        "selection_changed_ratio", "mean_jaccard", "neighbor_coverage",
        "mean_added_distance", "final_fragmentation_delta", "final_longest_contiguous_run_delta",
    ):
        correlations[metric] = _correlation(
            [float(row[metric]) for row in valid],
            [float(row["sample_score_delta"]) for row in valid],
        )
    group_rows = []
    for label, predicate in (
        ("wins", lambda value: value > 0), ("losses", lambda value: value < 0),
        ("ties", lambda value: value == 0),
    ):
        group = [row for row in valid if predicate(float(row["sample_score_delta"]))]
        group_rows.append({
            "group": label, "samples": len(group),
            "selection_changed_ratio": statistics.mean(float(row["selection_changed_ratio"]) for row in group) if group else None,
            "mean_jaccard": statistics.mean(float(row["mean_jaccard"]) for row in group) if group else None,
            "neighbor_coverage": statistics.mean(float(row["neighbor_coverage"]) for row in group) if group else None,
            "final_fragmentation_delta": statistics.mean(float(row["final_fragmentation_delta"]) for row in group) if group else None,
        })
    atomic_write_csv(root / "cc_win_loss_structure.csv", group_rows, [
        "group", "samples", "selection_changed_ratio", "mean_jaccard",
        "neighbor_coverage", "final_fragmentation_delta",
    ])
    atomic_write_json(root / "CC_STRUCTURE_CORRELATIONS.json", correlations)
    return {"trace_events": events, "samples": len(rows), "correlations": correlations}


def _drm_analysis(pipeline_dir: Path, root: Path) -> Dict[str, Any]:
    comparisons = {
        "sliminfer_drm_add_km": "sliminfer_larger_km",
        "drm_swap_original_k": "sliminfer_original",
    }
    delta_maps = {
        method: _score_delta_map(pipeline_dir, method, baseline)
        for method, baseline in comparisons.items()
    }
    canonical_ids = {
        task: [
            canonical_sample_identity(task, index, sample)[0]
            for index, sample in enumerate(load_task_samples(task))
        ]
        for task in TASKS
    }
    rows: List[Dict[str, Any]] = []
    for method in ("sliminfer_drm_add_km", "sliminfer_random_add_km", "drm_swap_original_k"):
        path = QWEN_DRM_ROOT / "audits" / f"{method}_rescue_policy_audit.jsonl"
        for event in iter_jsonl(path):
            if str(event.get("phase")) != "prefill":
                continue
            task = str(event.get("task"))
            historical_sample_id = str(event.get("sample_id"))
            sample_index = int(event.get("sample_index"))
            sample_id = canonical_ids[task][sample_index]
            rescued = list(event.get("rescued_block_ids_first50", event.get("rescued_ids_sampled", [])))
            dropped = list(event.get("dropped_ids_sampled", []))
            rows.append({
                "method": method, "task": task, "sample_id": sample_id,
                "historical_sample_id": historical_sample_id,
                "sample_index": sample_index, "layer": event.get("layer_id"),
                "context_length": event.get("context_length"), "K": event.get("K"), "M": event.get("M"),
                "rescued_count": event.get("rescued_count", len(rescued)),
                "swapped_count": event.get("swapped_count", len(dropped)),
                "rescued_mean_pair_distance": (
                    statistics.mean(abs(left - right) for index, left in enumerate(rescued) for right in rescued[index + 1:])
                    if len(rescued) > 1 else 0.0
                ),
                "rescued_span": max(rescued) - min(rescued) if rescued else 0,
                "dropped_count": len(dropped),
                "duplicate": event.get("duplicate_block_ids"),
                "protected_violation": event.get("protected_block_violation"),
                "fallback": event.get("fallback_used"),
                "d_base_mean": (event.get("D_base_stats") or {}).get("mean"),
                "gain_mean": (event.get("G_diffusion_stats") or {}).get("mean"),
                "active_redundancy_mean": (event.get("R_active_stats") or {}).get("mean"),
                "mmr_diversity_mean": (event.get("MMR_diversity_stats") or {}).get("mean"),
                "sample_score_delta": delta_maps.get(method, {}).get((task, sample_id)),
            })
    atomic_write_csv(root / "drm_selection_mechanism.csv", rows, [
        "method", "task", "sample_id", "historical_sample_id", "sample_index", "layer", "context_length", "K", "M",
        "rescued_count", "swapped_count", "rescued_mean_pair_distance", "rescued_span",
        "dropped_count", "duplicate", "protected_violation", "fallback", "d_base_mean",
        "gain_mean", "active_redundancy_mean", "mmr_diversity_mean", "sample_score_delta",
    ])
    bucket_rows = []
    for method in sorted({row["method"] for row in rows}):
        method_rows = [row for row in rows if row["method"] == method]
        for task in sorted({row["task"] for row in method_rows}):
            group = [row for row in method_rows if row["task"] == task]
            bucket_rows.append({
                "method": method, "bucket_type": "task", "bucket": task, "events": len(group),
                "mean_rescued_distance": statistics.mean(float(row["rescued_mean_pair_distance"]) for row in group),
                "mean_rescued_span": statistics.mean(float(row["rescued_span"]) for row in group),
                "mean_mmr_diversity": statistics.mean(
                    float(row["mmr_diversity_mean"]) for row in group if row["mmr_diversity_mean"] is not None
                ) if any(row["mmr_diversity_mean"] is not None for row in group) else None,
            })
        for label, low, high in (("<8k", 0, 8192), ("8-16k", 8192, 16384), ("16-32k", 16384, 32768), ("32k+", 32768, 10**9)):
            group = [row for row in method_rows if low <= int(row["context_length"] or 0) < high]
            if group:
                bucket_rows.append({
                    "method": method, "bucket_type": "input_length", "bucket": label,
                    "events": len(group),
                    "mean_rescued_distance": statistics.mean(float(row["rescued_mean_pair_distance"]) for row in group),
                    "mean_rescued_span": statistics.mean(float(row["rescued_span"]) for row in group),
                    "mean_mmr_diversity": statistics.mean(
                        float(row["mmr_diversity_mean"]) for row in group if row["mmr_diversity_mean"] is not None
                    ) if any(row["mmr_diversity_mean"] is not None for row in group) else None,
                })
    atomic_write_csv(root / "drm_task_length_buckets.csv", bucket_rows, [
        "method", "bucket_type", "bucket", "events", "mean_rescued_distance",
        "mean_rescued_span", "mean_mmr_diversity",
    ])
    return {"events": len(rows), "buckets": len(bucket_rows)}


def _ablation_manifest(stage: int) -> Dict[str, Any]:
    if stage == 9:
        return {
            "representative_tasks": list(REPRESENTATIVE_TASKS),
            "frozen_main": {"alpha": 0.25, "min_cluster_len": 2, "max_neighbors": 2, "replacement_cap": 0.05},
            "requested": [
                "continuity_strength", "neighbor_radius", "replacement_ratio", "neighbor_only",
                "cluster_only", "random_swap_fixed_k", "original", "cc_full",
            ],
            "availability": "SOFT_BLOCKED",
            "reason": (
                "The frozen runner hard-codes CC v1 parameters and exposes no audited parameter injection. "
                "Existing topology wrappers are different methods and are not relabeled as component ablations."
            ),
        }
    return {
        "representative_tasks": list(REPRESENTATIVE_TASKS),
        "requested": [
            "rescue_ratio_2_5_10", "random_seed_0_1_2", "full_drm", "without_gain",
            "without_active_core_redundancy", "without_rescue_mmr", "base_utility_only",
            "add_vs_swap", "protected_core_fixed_k",
        ],
        "availability": "SOFT_BLOCKED",
        "reason": (
            "The quality-critical five-method full results remain frozen. A new ablation method registry and "
            "Stage-1-style equivalence gate are absent, so the pipeline records the preregistration without "
            "silently assigning the main protocol identity to new method configurations."
        ),
    }


def run_stage(pipeline_dir: Path, stage: int, *, dry_run: bool) -> Dict[str, Any]:
    gate = run_protocol_gate(pipeline_dir, stage)
    if gate["decision"] != "PASS":
        raise RuntimeError(f"Stage {stage} protocol gate failed")
    prefix = "qwen_cc_ablation_mechanism" if stage == 9 else "qwen_drm_ablation_mechanism"
    pointer = pipeline_dir / f"STAGE_{stage:02d}_MECHANISM_ROOT.json"
    if pointer.exists():
        root = Path(load_json(pointer)["path"])
    else:
        root = REPO_ROOT / "results" / f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        atomic_write_json(pointer, {"created_at": now_iso(), "path": str(root)})
    manifest = _ablation_manifest(stage)
    plan = {
        "created_at": now_iso(), "protocol_id": PROTOCOL_ID, "stage": stage,
        "root": str(root), **manifest,
    }
    atomic_write_json(pipeline_dir / f"STAGE_{stage:02d}_PLAN.json", plan)
    if dry_run:
        decision = {"stage": stage, "decision": "NOT_RUN", **plan}
        atomic_write_json(stage_decision_path(pipeline_dir, stage), decision)
        return decision
    root.mkdir(parents=True, exist_ok=True)
    analysis = _cc_analysis(pipeline_dir, root) if stage == 9 else _drm_analysis(pipeline_dir, root)
    atomic_write_json(root / "ABLATION_AVAILABILITY.json", manifest)
    atomic_write_json(root / "MECHANISM_ANALYSIS_SUMMARY.json", analysis)
    atomic_write_text(
        root / "GOLD_EVIDENCE_ANALYSIS_BLOCKED.md",
        "# Gold Evidence Analysis Blocked\n\nOfficial LongBench rows do not provide a common, verified "
        "supporting-fact span/title annotation for these 17 tasks. No answer-derived pseudo-gold recall is computed.\n",
    )
    decision = {
        "stage": stage, "name": "cc_ablation_mechanism" if stage == 9 else "drm_ablation_mechanism",
        "created_at": now_iso(), "decision": "SOFT_BLOCKED", "run_root": str(root),
        "reason": "non_core_gpu_ablation_registry_not_equivalence_gated",
        "mechanism_analysis": "PASS", "analysis": analysis, "next_stage": stage + 1,
    }
    atomic_write_json(stage_decision_path(pipeline_dir, stage), decision)
    atomic_write_text(
        pipeline_dir / f"STAGE_{stage:02d}_SOFT_BLOCKER.md",
        f"# Stage {stage:02d} Soft Blocker\n\nCPU mechanism analysis completed from real runtime traces. "
        "The requested non-core GPU ablation grid was not launched because its new method identities have not "
        "passed an equivalence gate. Existing methods were not relabeled as ablations. The pipeline continues.\n",
    )
    return decision


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-dir", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--stage-9", action="store_true")
    group.add_argument("--stage-10", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    stage = 9 if args.stage_9 else 10 if args.stage_10 else None
    if stage is None:
        raise SystemExit("Select --stage-9 or --stage-10")
    result = run_stage(args.pipeline_dir.resolve(), stage, dry_run=args.dry_run)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
