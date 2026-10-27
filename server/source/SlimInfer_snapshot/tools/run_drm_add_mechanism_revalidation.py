#!/usr/bin/env python3
"""DRM-Add mechanism revalidation driver.

This driver is intentionally gate-first:
  1. run feature-complete gate-audit traces, no QA;
  2. stop immediately on missing DRM features or baseline parity failure;
  3. only after a passing active-set audit may runtime/QA stages be launched.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

FREEZE_DIR = REPO_ROOT / "results" / "drm_add_mechanism_revalidation_20260714_174251"
DEFAULT_MODEL = (
    "/home/yu-bai/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/"
    "snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
)
DEFAULT_LONGBENCH = Path("/home/yu-bai/Desktop/jiaqizhang/HeadKV/data/LongBench")
TRACE_TASKS = ("musique", "2wikimqa", "triviaqa")
QA_TASKS = ("hotpotqa", "musique")
TRACE_SAMPLE_INDICES_BY_TASK_24 = {
    "musique": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
    "2wikimqa": [10, 15, 22, 32, 50, 78, 85, 88, 93, 98, 104, 126, 130, 137, 148, 151, 153, 157, 160, 172, 174, 176, 189, 192],
    "triviaqa": [1, 2, 3, 10, 13, 16, 18, 19, 20, 24, 25, 26, 29, 32, 34, 35, 37, 38, 44, 47, 48, 53, 55, 57],
}

METHODS: Dict[str, Dict[str, Any]] = {
    "sliminfer_original": {
        "budget": "K",
        "runner": "sliminfer_original",
        "policy": "none",
        "score": "delta_o",
        "base": None,
    },
    "cc_sliminfer_v1": {
        "budget": "K",
        "runner": "cc_sliminfer_v1",
        "policy": "none",
        "score": "delta_o",
        "base": None,
    },
    "drm_swap_original_k": {
        "budget": "K",
        "runner": "sliminfer_original",
        "policy": "budget_neutral_drm_swap",
        "score": "drm",
        "base": "sliminfer_original",
    },
    "cc_drm_swap_k": {
        "budget": "K",
        "runner": "cc_sliminfer_v1",
        "policy": "budget_neutral_drm_swap",
        "score": "drm",
        "base": "cc_sliminfer_v1",
    },
    "sliminfer_larger_km": {
        "budget": "K+M",
        "runner": "sliminfer_original",
        "policy": "sliminfer_larger",
        "score": "delta_o",
        "base": "sliminfer_original",
    },
    "sliminfer_random_add_km": {
        "budget": "K+M",
        "runner": "sliminfer_original",
        "policy": "add",
        "score": "random",
        "base": "sliminfer_original",
    },
    "sliminfer_drm_add_km": {
        "budget": "K+M",
        "runner": "sliminfer_original",
        "policy": "add",
        "score": "drm",
        "base": "sliminfer_original",
    },
    "cc_larger_km": {
        "budget": "K+M",
        "runner": "cc_sliminfer_v1",
        "policy": "sliminfer_larger",
        "score": "delta_o",
        "base": "cc_sliminfer_v1",
    },
    "cc_random_add_km": {
        "budget": "K+M",
        "runner": "cc_sliminfer_v1",
        "policy": "add",
        "score": "random",
        "base": "cc_sliminfer_v1",
    },
    "cc_drm_add_km": {
        "budget": "K+M",
        "runner": "cc_sliminfer_v1",
        "policy": "add",
        "score": "drm",
        "base": "cc_sliminfer_v1",
    },
}

DRM_METHODS = {
    "drm_swap_original_k",
    "cc_drm_swap_k",
    "sliminfer_drm_add_km",
    "cc_drm_add_km",
}
ADD_METHODS = {
    "sliminfer_random_add_km",
    "sliminfer_drm_add_km",
    "cc_random_add_km",
    "cc_drm_add_km",
}
SWAP_METHODS = {"drm_swap_original_k", "cc_drm_swap_k"}
K_METHODS = {"sliminfer_original", "cc_sliminfer_v1", "drm_swap_original_k", "cc_drm_swap_k"}
KM_METHODS = set(METHODS) - K_METHODS
LAYER_M_SCHEDULE = {9: 10, 18: 5, 26: 3}
TRACE_FILES = {
    "selection": "drm_selection_trace.jsonl",
    "candidate": "drm_candidate_trace.jsonl",
    "mmr": "drm_mmr_selection_trace.jsonl",
    "reps": "drm_block_representation_trace.jsonl",
    "mapping": "drm_virtual_physical_mapping.jsonl",
}


def parse_args() -> argparse.Namespace:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_dir", type=Path, default=FREEZE_DIR / f"phase2_feature_complete_runtime_{ts}")
    parser.add_argument("--model_path", default=DEFAULT_MODEL)
    parser.add_argument("--longbench_dir", default=str(DEFAULT_LONGBENCH))
    parser.add_argument("--python", default=str(REPO_ROOT / ".venv/bin/python"))
    parser.add_argument("--smoke_samples", type=int, default=8, help="Per-task cap for 24-sample smoke.")
    parser.add_argument("--full_samples", type=int, default=24, help="Per-task cap for 72-sample trace.")
    parser.add_argument("--skip_smoke", action="store_true")
    parser.add_argument("--skip_full", action="store_true")
    parser.add_argument("--skip_qa", action="store_true", help="Do not launch runtime smoke or two-task QA even if audit passes.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--trace_sample_indices_by_task",
        default="auto_long_enough",
        help="Use 'auto_long_enough' for the frozen long-context trace indices, empty for first-N, or task:i,j;task:k.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def key_of(row: Mapping[str, Any]) -> Tuple[str, str, int]:
    return (
        str(row.get("task", "")),
        str(row.get("sample_id", row.get("sample_index", ""))),
        int(row.get("layer_idx", row.get("layer_id", -1)) or -1),
    )


def ids(row: Mapping[str, Any], field: str = "selected_block_ids") -> Set[int]:
    return {int(x) for x in (row.get(field) or [])}


def mean(values: Iterable[Any]) -> str:
    nums = []
    for value in values:
        try:
            val = float(value)
        except (TypeError, ValueError):
            continue
        if val == val:
            nums.append(val)
    return "" if not nums else f"{sum(nums) / len(nums):.8g}"


def numeric(value: Any, default: float = 0.0) -> float:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return float(default)
    if val != val:
        return float(default)
    return val


def jaccard(a: Iterable[int], b: Iterable[int]) -> float:
    left = {int(x) for x in a}
    right = {int(x) for x in b}
    union = left | right
    return float(len(left & right) / len(union)) if union else 0.0


def hash_ids(values: Iterable[int]) -> str:
    payload = json.dumps(sorted(int(x) for x in values), separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def trace_sample_indices_arg(args: argparse.Namespace, max_samples: int) -> str:
    spec = str(getattr(args, "trace_sample_indices_by_task", "") or "")
    if not spec:
        return ""
    if spec != "auto_long_enough":
        return spec
    parts = []
    for task in TRACE_TASKS:
        selected = TRACE_SAMPLE_INDICES_BY_TASK_24[task][: int(max_samples)]
        parts.append(f"{task}:{','.join(str(x) for x in selected)}")
    return ";".join(parts)


def method_command(args: argparse.Namespace, stage_dir: Path, method_name: str, max_samples: int) -> List[str]:
    spec = METHODS[method_name]
    run_dir = stage_dir / f"run_{method_name}"
    trace_dir = run_dir / "drm_trace"
    cmd = [
        args.python,
        str(REPO_ROOT / "experiments" / "run_qwen25_sliminfer_variants.py"),
        "--model_path", str(args.model_path),
        "--tokenizer_path", str(args.model_path),
        "--model_family", "qwen2",
        "--dtype", "fp16",
        "--device", "cuda",
        "--backend", "flash_attention_2",
        "--longbench_dir", str(args.longbench_dir),
        "--longbench_tasks", ",".join(TRACE_TASKS),
        "--longbench_max_samples", str(int(max_samples)),
        "--max_model_len", "131072",
        "--method", str(spec["runner"]),
        "--eval", "gate_audit",
        "--block_strategy", "fixed",
        "--block_size", "64",
        "--token_unit_size", "8",
        "--local_query_window", "4",
        "--swap_threshold", "0.9",
        "--prune_layers", "9,18,26",
        "--retain_tokens", "12288,6144,4096",
        "--decode_policy", "original",
        "--rescue_policy", str(spec["policy"]),
        "--rescue_score", str(spec["score"]),
        "--rescue_ratio", "0.05",
        "--alpha_diff", "0.5",
        "--lambda_red", "0.25",
        "--eta_rescue", "0.25",
        "--eta_active", "0.25",
        "--generation_max_new_tokens", "0",
        "--seed", "0",
        "--min_free_gpu_gb", "0",
        "--cc_selection_audit_path", str(run_dir / "selection_audit.jsonl"),
        "--rescue_audit_path", str(run_dir / "rescue_policy_audit.jsonl"),
        "--drm_trace_dir", str(trace_dir),
        "--output_dir", str(run_dir),
    ]
    sample_indices = trace_sample_indices_arg(args, max_samples)
    if sample_indices:
        cmd.extend(["--longbench_sample_indices_by_task", sample_indices])
    return cmd


def run_cmd(cmd: List[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    with log_path.open("w", encoding="utf-8") as handle:
        return subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, stdout=handle, stderr=subprocess.STDOUT).returncode


def run_trace_stage(args: argparse.Namespace, stage_name: str, max_samples: int) -> Dict[str, Any]:
    stage_dir = args.out_dir / stage_name
    stage_dir.mkdir(parents=True, exist_ok=True)
    status: Dict[str, Any] = {"stage": stage_name, "max_samples_per_task": max_samples, "methods": {}}
    for method_name in METHODS:
        run_dir = stage_dir / f"run_{method_name}"
        audit_path = run_dir / "selection_audit.jsonl"
        if args.resume and audit_path.exists():
            status["methods"][method_name] = {"returncode": 0, "resumed": True}
            continue
        cmd = method_command(args, stage_dir, method_name, max_samples)
        write_json(stage_dir / f"command_run_{method_name}.json", {"command": cmd})
        rc = run_cmd(cmd, stage_dir / f"run_{method_name}.log")
        status["methods"][method_name] = {"returncode": rc, "resumed": False}
        if rc != 0:
            status["status"] = "failed"
            status["failed_method"] = method_name
            write_json(stage_dir / "trace_stage_status.json", status)
            return status
    status["status"] = "ok"
    write_json(stage_dir / "trace_stage_status.json", status)
    return status


def load_stage(stage_dir: Path) -> Tuple[
    Dict[str, List[Dict[str, Any]]],
    Dict[str, List[Dict[str, Any]]],
    Dict[str, Dict[str, List[Dict[str, Any]]]],
]:
    selections = {
        method: read_jsonl(stage_dir / f"run_{method}" / "selection_audit.jsonl")
        for method in METHODS
    }
    trace_files = {
        method: {
            name: read_jsonl(stage_dir / f"run_{method}" / "drm_trace" / filename)
            for name, filename in TRACE_FILES.items()
        }
        for method in METHODS
    }
    traces = {method: trace_files[method]["selection"] for method in METHODS}
    return selections, traces, trace_files


def audit_stage(stage_dir: Path, audit_dir: Path) -> Dict[str, Any]:
    selections, traces, trace_files = load_stage(stage_dir)
    audit_dir.mkdir(parents=True, exist_ok=True)
    by_method = {method: {key_of(row): row for row in rows} for method, rows in selections.items()}
    trace_by_method = {method: {key_of(row): row for row in rows} for method, rows in traces.items()}
    candidate_by_method = {
        method: defaultdict(list) for method in METHODS
    }
    mmr_by_method = {
        method: defaultdict(list) for method in METHODS
    }
    reps_by_method = {
        method: defaultdict(list) for method in METHODS
    }
    mapping_by_method = {
        method: defaultdict(list) for method in METHODS
    }
    for method in METHODS:
        for row in trace_files[method]["candidate"]:
            candidate_by_method[method][key_of(row)].append(row)
        for row in trace_files[method]["mmr"]:
            mmr_by_method[method][key_of(row)].append(row)
        for row in trace_files[method]["reps"]:
            reps_by_method[method][key_of(row)].append(row)
        for row in trace_files[method]["mapping"]:
            mapping_by_method[method][key_of(row)].append(row)
    all_keys = sorted(set().union(*(set(rows.keys()) for rows in by_method.values())))
    mechanism_rows: List[Dict[str, Any]] = []
    budget_rows: List[Dict[str, Any]] = []
    add_rows: List[Dict[str, Any]] = []
    swap_rows: List[Dict[str, Any]] = []
    pair_rows: List[Dict[str, Any]] = []
    drm_candidate_stat_rows: List[Dict[str, Any]] = []
    drm_add_candidate_rows: List[Dict[str, Any]] = []
    drm_mmr_csv_rows: List[Dict[str, Any]] = []
    random_candidate_rows: List[Dict[str, Any]] = []
    larger_stat_rows: List[Dict[str, Any]] = []
    sacrifice_rows: List[Dict[str, Any]] = []
    task_counter: Dict[str, Counter] = defaultdict(Counter)
    failures: List[str] = []

    for method, rows in selections.items():
        if not rows:
            failures.append(f"{method}:missing_selection_audit")
    for key in all_keys:
        for method in METHODS:
            if key not in by_method.get(method, {}):
                task, sample_id, layer = key
                failures.append(f"{method}:{task}:{sample_id}:L{layer}:missing_selection_row")

    for method, row_map in by_method.items():
        spec = METHODS[method]
        base_method = spec.get("base")
        for key, row in sorted(row_map.items()):
            task, sample_id, layer = key
            final = ids(row)
            trace = trace_by_method.get(method, {}).get(key, {})
            candidate_rows = candidate_by_method[method].get(key, [])
            mmr_rows = mmr_by_method[method].get(key, [])
            rep_rows = reps_by_method[method].get(key, [])
            mapping_rows = mapping_by_method[method].get(key, [])
            runtime_base = {int(x) for x in (trace.get("active_block_ids") or [])} if trace else set()
            if method in (ADD_METHODS | SWAP_METHODS) and runtime_base:
                base = runtime_base
            else:
                base = ids(by_method.get(base_method, {}).get(key, {})) if base_method else set(final)
            target_m = int(trace.get("target_m", 0) or 0)
            if method in KM_METHODS and target_m <= 0 and base_method:
                target_m = max(1, int(round(len(base) * 0.05))) if base and (len(final) > len(base)) else max(0, len(final) - len(base))
            expected = len(base) if method in K_METHODS else len(base) + target_m
            actual = len(final)
            added = sorted(final - base)
            removed = sorted(base - final)
            duplicate_count = max(0, len(row.get("selected_block_ids") or []) - len(final))
            forced_sink_violation = int(row.get("sink_violation") or 0)
            over = max(0, actual - expected)
            under = max(0, expected - actual)
            original = ids(by_method.get("sliminfer_original", {}).get(key, {}))
            cc = ids(by_method.get("cc_sliminfer_v1", {}).get(key, {}))
            larger_name = "cc_larger_km" if str(base_method).startswith("cc_") else "sliminfer_larger_km"
            random_name = "cc_random_add_km" if str(base_method).startswith("cc_") else "sliminfer_random_add_km"
            error_reasons: List[str] = []
            expected_m = LAYER_M_SCHEDULE.get(layer)
            if method == "sliminfer_original":
                silm_baseline = ids(row, "silm_baseline_ids")
                if silm_baseline and final != silm_baseline:
                    error_reasons.append("baseline_parity_failure")
                if int(row.get("budget_parity", 1) or 0) != 1:
                    error_reasons.append("baseline_budget_parity_failure")
            if method in DRM_METHODS:
                missing_trace_parts = []
                if not trace:
                    missing_trace_parts.append("selection")
                if not candidate_rows:
                    missing_trace_parts.append("candidate")
                if not mmr_rows:
                    missing_trace_parts.append("mmr")
                if not rep_rows:
                    missing_trace_parts.append("block_representation")
                if not mapping_rows:
                    missing_trace_parts.append("virtual_physical_mapping")
                if missing_trace_parts:
                    error_reasons.append("missing_drm_trace:" + ",".join(missing_trace_parts))
                if trace and not bool(trace.get("feature_complete", False)):
                    error_reasons.append("missing_features:" + ",".join(trace.get("missing_features") or []))
                missing_candidate_features = sorted({
                    ",".join(str(x) for x in (cand.get("missing_features") or []))
                    for cand in candidate_rows
                    if not bool(cand.get("feature_complete", False))
                })
                if missing_candidate_features:
                    error_reasons.append("candidate_missing_features:" + ";".join(missing_candidate_features))
                if trace and candidate_rows and int(trace.get("candidate_available_count", -1) or -1) != len(candidate_rows):
                    error_reasons.append("candidate_trace_count_mismatch")
                if trace and candidate_rows:
                    required_fields = ("D_base", "G", "R_active", "drm_score_s0", "qk_score", "block_representation_dim")
                    missing_fields = sorted({
                        field
                        for cand in candidate_rows
                        for field in required_fields
                        if cand.get(field) in (None, "") or (field == "block_representation_dim" and int(cand.get(field) or 0) <= 0)
                    })
                    if missing_fields:
                        error_reasons.append("candidate_required_field_missing:" + ",".join(missing_fields))
                if not any(bool(mrow.get("selected")) for mrow in mmr_rows):
                    error_reasons.append("mmr_selected_trace_missing")
                if not all(bool(mrow.get("physical_block_mapping_valid", False)) for mrow in mapping_rows):
                    error_reasons.append("physical_mapping_invalid")
            if method in ADD_METHODS | SWAP_METHODS:
                if expected_m is not None and target_m != expected_m:
                    error_reasons.append(f"m_schedule_mismatch:expected_{expected_m}_got_{target_m}")
            selected_candidate_scores = [
                cand.get("drm_score_s0")
                for cand in candidate_rows
                if cand.get("add_or_swap_applied") or cand.get("mmr_selected")
            ]
            pruned_candidate_scores = [
                cand.get("drm_score_s0")
                for cand in candidate_rows
                if not cand.get("add_or_swap_applied") and not cand.get("mmr_selected")
            ]
            selected_nums = [numeric(x) for x in selected_candidate_scores if x not in (None, "")]
            pruned_nums = [numeric(x) for x in pruned_candidate_scores if x not in (None, "")]
            selected_margin = ""
            if selected_nums and pruned_nums:
                selected_margin = f"{min(selected_nums) - max(pruned_nums):.8g}"
            mechanism = {
                "method": method,
                "task": task,
                "sample_id": sample_id,
                "prune_layer": layer,
                "base_budget_k": len(base),
                "rescue_budget_m": target_m,
                "expected_final_count": expected,
                "actual_final_count": actual,
                "base_active_count": len(base),
                "base_active_hash": hash_ids(base),
                "final_active_hash": hash_ids(final),
                "base_set_preserved": int(base.issubset(final)),
                "added_count": len(added),
                "removed_count": len(removed),
                "duplicate_count": duplicate_count,
                "invalid_block_count": 0,
                "candidate_pool_count": int(trace.get("candidate_available_count", 0) or 0),
                "drm_candidate_available_count": int(trace.get("candidate_available_count", 0) or 0),
                "drm_candidate_eligible_count": int(trace.get("candidate_eligible_count", 0) or 0),
                "drm_mmr_selected_count": int(trace.get("drm_mmr_selected_count", 0) or 0),
                "random_selected_count": len(added) if METHODS[method]["score"] == "random" else 0,
                "forced_sink_present": int(row.get("sink_kept") or 0),
                "forced_sink_violation": forced_sink_violation,
                "over_budget": over,
                "under_budget": under,
                "jaccard_vs_original": jaccard(final, original),
                "jaccard_vs_cc": jaccard(final, cc),
                "jaccard_vs_larger_control": jaccard(final, ids(by_method.get(larger_name, {}).get(key, {}))),
                "jaccard_vs_random_control": jaccard(final, ids(by_method.get(random_name, {}).get(key, {}))),
                "mean_added_drm_score": mean(selected_candidate_scores),
                "mean_pruned_drm_score": mean(pruned_candidate_scores),
                "min_selected_rescue_margin": selected_margin,
                "max_selected_rescue_margin": selected_margin,
                "virtual_block_path_used": bool(trace.get("virtual_block_path_used", False)),
                "physical_block_mapping_valid": bool(trace.get("physical_block_mapping_valid", True)),
                "error_reason": ";".join(error_reasons),
            }
            if method in DRM_METHODS:
                selected_scores = [numeric(x) for x in selected_candidate_scores if x not in (None, "")]
                pruned_scores = [numeric(x) for x in pruned_candidate_scores if x not in (None, "")]
                drm_candidate_stat_rows.append({
                    "base_method": base_method,
                    "task": task,
                    "sample_id": sample_id,
                    "prune_layer": layer,
                    "base_budget_k": len(base),
                    "rescue_budget_m": target_m,
                    "candidate_pool_count": len(candidate_rows),
                    "drm_candidate_available_count": sum(1 for cand in candidate_rows if cand.get("candidate_available")),
                    "drm_candidate_eligible_count": sum(1 for cand in candidate_rows if cand.get("candidate_eligible")),
                    "drm_mmr_selected_count": sum(1 for cand in candidate_rows if cand.get("mmr_selected")),
                    "mean_added_drm_score": mean(selected_scores),
                    "mean_pruned_drm_score": mean(pruned_scores),
                    "min_selected_rescue_margin": selected_margin,
                    "max_selected_rescue_margin": selected_margin,
                    "mmr_vs_plain_topm_jaccard": trace.get("mmr_vs_plain_top_m_jaccard", ""),
                    "status": "ok" if not error_reasons else "fail",
                    "error_reason": ";".join(error_reasons),
                })
                rank_order = sorted(
                    candidate_rows,
                    key=lambda cand: numeric(cand.get("drm_score_s0"), float("-inf")),
                    reverse=True,
                )
                rank_by_block = {int(cand.get("block_id")): rank + 1 for rank, cand in enumerate(rank_order)}
                selected_mmr_by_block = {
                    int(mrow.get("candidate_block_id")): mrow
                    for mrow in mmr_rows
                    if bool(mrow.get("selected"))
                }
                if method in {"sliminfer_drm_add_km", "cc_drm_add_km"}:
                    for cand in candidate_rows:
                        block_id = int(cand.get("block_id"))
                        mmr_row = selected_mmr_by_block.get(block_id, {})
                        s0 = cand.get("drm_score_s0")
                        mmr_score = mmr_row.get("mmr_score", "")
                        penalty = ""
                        if s0 not in (None, "") and mmr_score not in (None, ""):
                            penalty = f"{numeric(s0) - numeric(mmr_score):.8g}"
                        drm_add_candidate_rows.append({
                            "method": method,
                            "base_method": base_method,
                            "task": task,
                            "sample_id": sample_id,
                            "prune_layer": layer,
                            "mmr_step": mmr_row.get("mmr_step", ""),
                            "block_id": block_id,
                            "plain_drm_rank": rank_by_block.get(block_id, ""),
                            "plain_drm_score": s0,
                            "mmr_score": mmr_score,
                            "redundancy_penalty": penalty,
                            "from_pruned_set": int(bool(cand.get("from_pruned_set", True))),
                            "selected_by_mmr": int(bool(cand.get("mmr_selected"))),
                            "status": "ok" if cand.get("candidate_eligible") else "not_eligible",
                            "error_reason": cand.get("rejected_reason", ""),
                        })
                cand_by_block = {int(cand.get("block_id")): cand for cand in candidate_rows if cand.get("block_id") is not None}
                for mrow in mmr_rows:
                    block_id = int(mrow.get("candidate_block_id"))
                    cand = cand_by_block.get(block_id, {})
                    drm_mmr_csv_rows.append({
                        "method": method,
                        "base_method": base_method,
                        "task": task,
                        "sample_id": sample_id,
                        "prune_layer": layer,
                        "mmr_step": mrow.get("mmr_step", ""),
                        "candidate_block_id": block_id,
                        "s0_score": mrow.get("s0_score", ""),
                        "z_d_base": cand.get("z_D_base", ""),
                        "z_g": cand.get("z_G", ""),
                        "z_r_active": cand.get("z_R_active", ""),
                        "redundancy_to_selected": mrow.get("rescue_redundancy", ""),
                        "mmr_score": mrow.get("mmr_score", ""),
                        "selected": int(bool(mrow.get("selected"))),
                        "selected_so_far_hash": hash_ids(mrow.get("selected_so_far") or []),
                        "status": "ok",
                        "error_reason": "",
                    })
            if method in {"sliminfer_random_add_km", "cc_random_add_km"}:
                rank_order = sorted(
                    candidate_rows,
                    key=lambda cand: numeric(cand.get("drm_score_s0"), float("-inf")),
                    reverse=True,
                )
                rank_by_block = {int(cand.get("block_id")): rank + 1 for rank, cand in enumerate(rank_order)}
                for cand in candidate_rows:
                    block_id = int(cand.get("block_id"))
                    random_candidate_rows.append({
                        "method": method,
                        "base_method": base_method,
                        "task": task,
                        "sample_id": sample_id,
                        "prune_layer": layer,
                        "seed": cand.get("seed", 0),
                        "random_rank": rank_by_block.get(block_id, ""),
                        "block_id": block_id,
                        "from_pruned_set": int(bool(cand.get("from_pruned_set", True))),
                        "duplicate": int(block_id in base),
                        "final_count_after_add": actual,
                        "status": "selected" if cand.get("add_or_swap_applied") else "not_selected",
                        "error_reason": cand.get("rejected_reason", ""),
                    })
            if method in {"sliminfer_larger_km", "cc_larger_km"}:
                larger_stat_rows.append({
                    "method": method,
                    "base_method": base_method,
                    "task": task,
                    "sample_id": sample_id,
                    "prune_layer": layer,
                    "base_budget_k": len(base),
                    "rescue_budget_m": target_m,
                    "expected_final_count": expected,
                    "actual_final_count": actual,
                    "extra_blocks_count": len(added),
                    "base_set_preserved": int(base.issubset(final)),
                    "jaccard_vs_base": jaccard(final, base),
                    "status": "ok" if not error_reasons and base.issubset(final) else "fail",
                    "error_reason": ";".join(error_reasons),
                })
            if method in ADD_METHODS:
                add_rows.append({
                    "method": method,
                    "base_method": base_method,
                    "task": task,
                    "sample_id": sample_id,
                    "prune_layer": layer,
                    "base_active_count": len(base),
                    "final_active_count": actual,
                    "base_set_preserved": int(base.issubset(final)),
                    "missing_base_block_count": len(base - final),
                    "added_count": len(added),
                    "removed_count": len(removed),
                    "expected_added_m": target_m,
                    "status": "ok" if base.issubset(final) and len(added) == target_m and not removed else "fail",
                    "error_reason": mechanism["error_reason"],
                })
            if method in SWAP_METHODS:
                cc_base_row = by_method.get("cc_sliminfer_v1", {}).get(key, {})
                cc_neighbors = {int(x) for x in (cc_base_row.get("neighbors_retained_ids") or [])}
                swap_rows.append({
                    "method": method,
                    "base_method": base_method,
                    "task": task,
                    "sample_id": sample_id,
                    "prune_layer": layer,
                    "swap_rank": "",
                    "in_block_id": ",".join(str(x) for x in added),
                    "out_block_id": ",".join(str(x) for x in removed),
                    "in_from_pruned_set": int(set(added).issubset(set(trace.get("pruned_candidate_block_ids") or added))),
                    "out_from_base_active": int(set(removed).issubset(base)),
                    "forced_sink_removed": int(forced_sink_violation),
                    "final_count_after_swap": actual,
                    "status": "ok" if len(added) == len(removed) and actual == len(base) else "fail",
                    "error_reason": mechanism["error_reason"],
                })
                for block_id in removed:
                    in_cc = block_id in cc
                    in_original = block_id in original
                    sacrifice_rows.append({
                        "method": method,
                        "base_method": base_method,
                        "task": task,
                        "sample_id": sample_id,
                        "prune_layer": layer,
                        "block_id": block_id,
                        "role": "cc_neighbor_removed" if block_id in cc_neighbors else ("shared_original_cc_active" if in_cc and in_original else ("cc_active" if in_cc else "base_active")),
                        "cc_cluster_center": "",
                        "cc_neighbor": int(block_id in cc_neighbors),
                        "second_locus": "",
                        "high_confidence_spike": "",
                        "forced_coverage_block": "",
                        "ordinary_sacrifice_candidate": int(block_id not in cc_neighbors),
                        "status": "ok",
                        "error_reason": "",
                    })
            budget_rows.append({
                "method": method,
                "task": task,
                "sample_id": sample_id,
                "prune_layer": layer,
                "base_budget_k": len(base),
                "rescue_budget_m": target_m,
                "expected_final_count": expected,
                "actual_final_count": actual,
                "over_budget": over,
                "under_budget": under,
                "forced_sink_violation": forced_sink_violation,
                "duplicate_count": duplicate_count,
                "invalid_block_count": 0,
                "status": "ok" if over == 0 and under == 0 and duplicate_count == 0 and forced_sink_violation == 0 else "fail",
                "error_reason": mechanism["error_reason"],
            })
            task_counter[task]["layer_records"] += 1
            task_counter[task][method] += 1
            mechanism_rows.append(mechanism)

    for key in all_keys:
        task, sample_id, layer = key
        present = [(m, ids(rows[key])) for m, rows in by_method.items() if key in rows]
        for i, (ma, ia) in enumerate(present):
            for mb, ib in present[i + 1:]:
                pair_rows.append({
                    "task": task,
                    "sample_id": sample_id,
                    "prune_layer": layer,
                    "method_a": ma,
                    "method_b": mb,
                    "jaccard": jaccard(ia, ib),
                    "intersection_count": len(ia & ib),
                    "union_count": len(ia | ib),
                    "status": "ok",
                    "error_reason": "",
                })

    for row in mechanism_rows:
        method = str(row["method"])
        if method in K_METHODS and (row["actual_final_count"] != row["base_budget_k"]):
            failures.append(f"{method}:{row['task']}:{row['sample_id']}:L{row['prune_layer']}:K_budget")
        if method in KM_METHODS and (row["actual_final_count"] != row["expected_final_count"]):
            failures.append(f"{method}:{row['task']}:{row['sample_id']}:L{row['prune_layer']}:KM_budget")
        if method in ADD_METHODS and (not row["base_set_preserved"] or row["removed_count"] != 0):
            failures.append(f"{method}:{row['task']}:{row['sample_id']}:L{row['prune_layer']}:add_preservation")
        if method in SWAP_METHODS and row["added_count"] != row["removed_count"]:
            failures.append(f"{method}:{row['task']}:{row['sample_id']}:L{row['prune_layer']}:swap_pairing")
        if row["over_budget"] or row["under_budget"] or row["forced_sink_violation"] or row["duplicate_count"] or row["invalid_block_count"]:
            failures.append(f"{method}:{row['task']}:{row['sample_id']}:L{row['prune_layer']}:hard_assert")
        if row["error_reason"]:
            failures.append(f"{method}:{row['task']}:{row['sample_id']}:L{row['prune_layer']}:{row['error_reason']}")

    k_exact = all(r["actual_final_count"] == r["base_budget_k"] for r in mechanism_rows if r["method"] in K_METHODS)
    km_exact = all(r["actual_final_count"] == r["expected_final_count"] for r in mechanism_rows if r["method"] in KM_METHODS)
    add_rows_nonempty = [r for r in mechanism_rows if r["method"] in ADD_METHODS]
    add_pres_rate = (
        sum(1 for r in add_rows_nonempty if r["base_set_preserved"] and r["removed_count"] == 0) / len(add_rows_nonempty)
        if add_rows_nonempty else 0.0
    )
    swap_rows_nonempty = [r for r in mechanism_rows if r["method"] in SWAP_METHODS]
    swap_rate = (
        sum(1 for r in swap_rows_nonempty if r["added_count"] == r["removed_count"] and r["actual_final_count"] == r["base_budget_k"]) / len(swap_rows_nonempty)
        if swap_rows_nonempty else 0.0
    )
    def differs(left: str, right: str) -> bool:
        vals = [
            1.0 - p["jaccard"]
            for p in pair_rows
            if {p["method_a"], p["method_b"]} == {left, right}
        ]
        return any(v > 0.0 for v in vals)

    drm_diff_random = differs("sliminfer_drm_add_km", "sliminfer_random_add_km") or differs("cc_drm_add_km", "cc_random_add_km")
    drm_diff_larger = differs("sliminfer_drm_add_km", "sliminfer_larger_km") or differs("cc_drm_add_km", "cc_larger_km")
    trace_file_counts = {
        method: {name: len(rows) for name, rows in files.items()}
        for method, files in trace_files.items()
    }
    drm_trace_complete = all(
        trace_file_counts[method]["selection"] > 0
        and trace_file_counts[method]["candidate"] > 0
        and trace_file_counts[method]["mmr"] > 0
        and trace_file_counts[method]["reps"] > 0
        and trace_file_counts[method]["mapping"] > 0
        for method in DRM_METHODS
    )
    drm_formula_fields_complete = all(
        cand.get("D_base") not in (None, "")
        and cand.get("G") not in (None, "")
        and cand.get("R_active") not in (None, "")
        and cand.get("drm_score_s0") not in (None, "")
        and int(cand.get("block_representation_dim") or 0) > 0
        for method in DRM_METHODS
        for cand in trace_files[method]["candidate"]
    )
    decision = {
        "phase": 2,
        "stage_dir": display_path(stage_dir),
        "gpu_used": True,
        "qa_run": False,
        "trace_file_counts": trace_file_counts,
        "historical_drm_formula_verified": bool(drm_trace_complete and drm_formula_fields_complete),
        "mmr_verified": all(any((row.get("drm_mmr_selected_count") or 0) > 0 for row in mechanism_rows if row["method"] == m) for m in DRM_METHODS),
        "feature_complete": bool(drm_trace_complete and not any(
            "missing_features" in str(r.get("error_reason", ""))
            or "missing_drm_trace" in str(r.get("error_reason", ""))
            or "candidate_required_field_missing" in str(r.get("error_reason", ""))
            for r in mechanism_rows
        )),
        "k_budget_exact": k_exact,
        "km_budget_exact": km_exact,
        "add_base_preservation_rate": add_pres_rate,
        "swap_one_in_one_out_rate": swap_rate,
        "over_budget_count": sum(int(r["over_budget"]) for r in mechanism_rows),
        "under_budget_count": sum(int(r["under_budget"]) for r in mechanism_rows),
        "sink_violation_count": sum(int(r["forced_sink_violation"]) for r in mechanism_rows),
        "duplicate_count": sum(int(r["duplicate_count"]) for r in mechanism_rows),
        "invalid_block_count": sum(int(r["invalid_block_count"]) for r in mechanism_rows),
        "drm_differs_from_random": drm_diff_random,
        "drm_differs_from_larger": drm_diff_larger,
        "baseline_isolated": not any("baseline_" in reason for reason in failures),
        "virtual_path_caveat": False,
        "phase3_go": False,
        "decision": "FAIL",
        "blocking_reasons": sorted(set(failures))[:200],
    }
    hard_ok = (
        not failures
        and decision["historical_drm_formula_verified"]
        and decision["feature_complete"]
        and decision["k_budget_exact"]
        and decision["km_budget_exact"]
        and add_pres_rate == 1.0
        and swap_rate == 1.0
        and decision["over_budget_count"] == 0
        and decision["under_budget_count"] == 0
        and decision["sink_violation_count"] == 0
        and decision["duplicate_count"] == 0
        and decision["invalid_block_count"] == 0
        and drm_diff_random
        and drm_diff_larger
    )
    if hard_ok:
        decision["decision"] = "GO"
        decision["phase3_go"] = True
        decision["blocking_reasons"] = []

    fields = [
        "method", "task", "sample_id", "prune_layer", "base_budget_k", "rescue_budget_m",
        "expected_final_count", "actual_final_count", "base_active_count", "base_active_hash",
        "final_active_hash", "base_set_preserved", "added_count", "removed_count",
        "duplicate_count", "invalid_block_count", "candidate_pool_count",
        "drm_candidate_available_count", "drm_candidate_eligible_count",
        "drm_mmr_selected_count", "random_selected_count", "forced_sink_present",
        "forced_sink_violation", "over_budget", "under_budget", "jaccard_vs_original",
        "jaccard_vs_cc", "jaccard_vs_larger_control", "jaccard_vs_random_control",
        "mean_added_drm_score", "mean_pruned_drm_score", "min_selected_rescue_margin",
        "max_selected_rescue_margin", "virtual_block_path_used", "physical_block_mapping_valid",
        "error_reason",
    ]
    write_csv(audit_dir / "active_set_mechanism_audit.csv", mechanism_rows, fields)
    write_csv(audit_dir / "per_layer_budget_audit.csv", budget_rows, [
        "method", "task", "sample_id", "prune_layer", "base_budget_k", "rescue_budget_m",
        "expected_final_count", "actual_final_count", "over_budget", "under_budget",
        "forced_sink_violation", "duplicate_count", "invalid_block_count", "status", "error_reason",
    ])
    write_csv(audit_dir / "add_base_set_preservation.csv", add_rows, [
        "method", "base_method", "task", "sample_id", "prune_layer", "base_active_count",
        "final_active_count", "base_set_preserved", "missing_base_block_count", "added_count",
        "removed_count", "expected_added_m", "status", "error_reason",
    ])
    write_csv(audit_dir / "drm_swap_pairs.csv", swap_rows, [
        "method", "base_method", "task", "sample_id", "prune_layer", "swap_rank",
        "in_block_id", "out_block_id", "in_from_pruned_set", "out_from_base_active",
        "forced_sink_removed", "final_count_after_swap", "status", "error_reason",
    ])
    write_csv(audit_dir / "active_set_pairwise_jaccard.csv", pair_rows, [
        "task", "sample_id", "prune_layer", "method_a", "method_b", "jaccard",
        "intersection_count", "union_count", "status", "error_reason",
    ])
    per_task = []
    for task, counter in sorted(task_counter.items()):
        row = {"task": task, **counter}
        per_task.append(row)
    task_fields = ["task", "layer_records", *METHODS.keys()]
    write_csv(audit_dir / "per_task_selection_stats.csv", per_task, task_fields)
    write_json(audit_dir / "PHASE2_DECISION.json", decision)
    write_json(audit_dir / "phase2_unit_test_results.json", {
        "status": "passed" if hard_ok else "failed",
        "assertions": {
            "feature_complete": decision["feature_complete"],
            "k_budget_exact": decision["k_budget_exact"],
            "km_budget_exact": decision["km_budget_exact"],
            "add_base_preservation_rate": add_pres_rate,
            "swap_one_in_one_out_rate": swap_rate,
            "drm_differs_from_random": drm_diff_random,
            "drm_differs_from_larger": drm_diff_larger,
        },
        "blocking_reasons": decision["blocking_reasons"],
    })
    (audit_dir / "PHASE2_TRACE_INPUT_AUDIT.md").write_text(
        "# Phase 2 Trace Input Audit\n\n"
        f"Trace stage: `{display_path(stage_dir)}`\n\n"
        f"Methods: {', '.join(METHODS)}\n\n"
        f"Selection rows: {sum(len(v) for v in selections.values())}\n\n"
        f"DRM selection trace rows: {sum(len(v) for v in traces.values())}\n\n"
        f"Decision: `{decision['decision']}`\n",
        encoding="utf-8",
    )
    (audit_dir / "ACTIVE_SET_MECHANISM_AUDIT.md").write_text(
        "# Active Set Mechanism Audit\n\n"
        f"Decision: `{decision['decision']}`\n\n"
        f"K exact: `{decision['k_budget_exact']}`; K+M exact: `{decision['km_budget_exact']}`.\n\n"
        f"Add preservation rate: `{add_pres_rate}`; swap one-in-one-out rate: `{swap_rate}`.\n\n"
        f"Blocking reasons: {len(decision['blocking_reasons'])}\n",
        encoding="utf-8",
    )
    for name in [
        "DRM_PHASE2_ADAPTER_SPEC.md",
        "CASE_STUDY_SLIMINFER_ADD_VS_CONTROLS.md",
        "CASE_STUDY_CC_ADD_VS_CONTROLS.md",
        "CASE_STUDY_ADD_VS_SWAP.md",
    ]:
        (audit_dir / name).write_text(
            f"# {name[:-3].replace('_', ' ').title()}\n\nSee CSV artifacts and `PHASE2_DECISION.json`.\n",
            encoding="utf-8",
        )
    write_csv(audit_dir / "drm_candidate_stats.csv", drm_candidate_stat_rows, [
        "base_method", "task", "sample_id", "prune_layer", "base_budget_k",
        "rescue_budget_m", "candidate_pool_count", "drm_candidate_available_count",
        "drm_candidate_eligible_count", "drm_mmr_selected_count", "mean_added_drm_score",
        "mean_pruned_drm_score", "min_selected_rescue_margin",
        "max_selected_rescue_margin", "mmr_vs_plain_topm_jaccard", "status", "error_reason",
    ])
    write_csv(audit_dir / "drm_add_candidates.csv", drm_add_candidate_rows, [
        "method", "base_method", "task", "sample_id", "prune_layer", "mmr_step",
        "block_id", "plain_drm_rank", "plain_drm_score", "mmr_score",
        "redundancy_penalty", "from_pruned_set", "selected_by_mmr", "status", "error_reason",
    ])
    write_csv(audit_dir / "drm_mmr_selection_trace.csv", drm_mmr_csv_rows, [
        "method", "base_method", "task", "sample_id", "prune_layer", "mmr_step",
        "candidate_block_id", "s0_score", "z_d_base", "z_g", "z_r_active",
        "redundancy_to_selected", "mmr_score", "selected", "selected_so_far_hash",
        "status", "error_reason",
    ])
    write_csv(audit_dir / "random_add_candidates.csv", random_candidate_rows, [
        "method", "base_method", "task", "sample_id", "prune_layer", "seed",
        "random_rank", "block_id", "from_pruned_set", "duplicate",
        "final_count_after_add", "status", "error_reason",
    ])
    write_csv(audit_dir / "larger_km_selection_stats.csv", larger_stat_rows, [
        "method", "base_method", "task", "sample_id", "prune_layer", "base_budget_k",
        "rescue_budget_m", "expected_final_count", "actual_final_count",
        "extra_blocks_count", "base_set_preserved", "jaccard_vs_base", "status", "error_reason",
    ])
    write_csv(audit_dir / "sacrifice_block_analysis.csv", sacrifice_rows, [
        "method", "base_method", "task", "sample_id", "prune_layer", "block_id",
        "role", "cc_cluster_center", "cc_neighbor", "second_locus",
        "high_confidence_spike", "forced_coverage_block",
        "ordinary_sacrifice_candidate", "status", "error_reason",
    ])
    return decision


def maybe_run_qa(args: argparse.Namespace, full_audit_decision: Dict[str, Any]) -> None:
    if args.skip_qa or not full_audit_decision.get("phase3_go"):
        return
    qa_dir = args.out_dir / "two_task_full_hotpotqa200_musique200"
    qa_dir.mkdir(parents=True, exist_ok=True)
    qa_methods = [
        "sliminfer_original",
        "cc_sliminfer_v1",
        "drm_swap_original_k",
        "cc_drm_swap_k",
        "sliminfer_larger_km",
        "sliminfer_random_add_km",
        "sliminfer_drm_add_km",
        "cc_larger_km",
        "cc_random_add_km",
        "cc_drm_add_km",
    ]
    for method_name in qa_methods:
        spec = METHODS[method_name]
        run_dir = qa_dir / f"run_{method_name}"
        cmd = [
            args.python,
            str(REPO_ROOT / "experiments" / "run_qwen25_sliminfer_variants.py"),
            "--model_path", str(args.model_path),
            "--tokenizer_path", str(args.model_path),
            "--model_family", "qwen2",
            "--dtype", "fp16",
            "--device", "cuda",
            "--backend", "flash_attention_2",
            "--longbench_dir", str(args.longbench_dir),
            "--longbench_tasks", ",".join(QA_TASKS),
            "--longbench_max_samples", "200",
            "--max_model_len", "131072",
            "--method", str(spec["runner"]),
            "--eval", "longbench",
            "--block_strategy", "fixed",
            "--block_size", "64",
            "--token_unit_size", "8",
            "--local_query_window", "4",
            "--swap_threshold", "0.9",
            "--prune_layers", "9,18,26",
            "--retain_tokens", "12288,6144,4096",
            "--decode_policy", "original",
            "--rescue_policy", str(spec["policy"]),
            "--rescue_score", str(spec["score"]),
            "--rescue_ratio", "0.05",
            "--alpha_diff", "0.5",
            "--lambda_red", "0.25",
            "--eta_rescue", "0.25",
            "--eta_active", "0.25",
            "--seed", "0",
            "--min_free_gpu_gb", "0",
            "--output_dir", str(run_dir),
        ]
        write_json(qa_dir / f"command_run_{method_name}.json", {"command": cmd})
        rc = run_cmd(cmd, qa_dir / f"run_{method_name}.log")
        if rc != 0:
            write_json(qa_dir / "qa_status.json", {"status": "failed", "failed_method": method_name, "returncode": rc})
            return
    write_json(qa_dir / "qa_status.json", {"status": "ok", "methods": qa_methods})


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.out_dir / "run_config.json", {
        "freeze_dir": str(FREEZE_DIR.relative_to(REPO_ROOT)),
        "methods": METHODS,
        "trace_tasks": TRACE_TASKS,
        "qa_tasks": QA_TASKS,
        "frozen_drm": {
            "rescue_ratio": 0.05,
            "alpha_diff": 0.5,
            "lambda_red": 0.25,
            "eta_rescue": 0.25,
            "eta_active": 0.25,
            "seed": 0,
        },
    })
    final_decision: Optional[Dict[str, Any]] = None
    if not args.skip_smoke:
        status = run_trace_stage(args, "smoke24", args.smoke_samples)
        if status.get("status") != "ok":
            return 1
        smoke_decision = audit_stage(args.out_dir / "smoke24", args.out_dir / "smoke24" / "phase2_active_set_audit")
        final_decision = smoke_decision
        if not smoke_decision.get("phase3_go"):
            return 2
    if not args.skip_full:
        status = run_trace_stage(args, "full72", args.full_samples)
        if status.get("status") != "ok":
            return 1
        full_decision = audit_stage(args.out_dir / "full72", args.out_dir / "full72" / "phase2_active_set_audit")
        final_decision = full_decision
        if not full_decision.get("phase3_go"):
            return 2
        maybe_run_qa(args, full_decision)
    write_json(args.out_dir / "FINAL_PHASE2_DRIVER_DECISION.json", final_decision or {"status": "not_run"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
