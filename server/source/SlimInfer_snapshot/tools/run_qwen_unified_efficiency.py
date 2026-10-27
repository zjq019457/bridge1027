#!/usr/bin/env python3
"""Run isolated, repeatable Qwen efficiency cases under the unified protocol."""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

_REPO_BOOTSTRAP = Path(__file__).resolve().parents[1]
if str(_REPO_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(_REPO_BOOTSTRAP))

from tools.qwen_pipeline_common import (
    PROTOCOL_DIR,
    PROTOCOL_ID,
    PYTHON,
    QWEN_MODEL_ROOT,
    REPO_ROOT,
    atomic_write_csv,
    atomic_write_json,
    atomic_write_text,
    implementation_revision,
    load_json,
    now_iso,
    sha256_file,
    stage_decision_path,
)
from tools.qwen_protocol_gate import RUNTIME_FILES, run_protocol_gate
from tools.qwen_resource_guard import monitor_resources, write_blocker
from tools.qwen_runtime_watchdog import RuntimeWatchdog


CONTEXT_LENGTHS = (8192, 16384, 24576, 28672, 32768)
REPETITIONS = 5
INTERNAL_METHODS: Dict[str, Dict[str, str]] = {
    "full_kv_fa2": {"runner": "fullkv", "policy": "none", "score": "delta_o"},
    "sliminfer_original": {"runner": "sliminfer_original", "policy": "none", "score": "delta_o"},
    "cc_sliminfer_v1": {"runner": "cc_sliminfer_v1", "policy": "none", "score": "delta_o"},
    "drm_swap_original_k": {
        "runner": "sliminfer_original", "policy": "budget_neutral_drm_swap", "score": "drm",
    },
    "sliminfer_larger_km": {
        "runner": "sliminfer_original", "policy": "sliminfer_larger", "score": "delta_o",
    },
    "sliminfer_random_add_km": {
        "runner": "sliminfer_original", "policy": "add", "score": "random",
    },
    "sliminfer_drm_add_km": {
        "runner": "sliminfer_original", "policy": "add", "score": "drm",
    },
}


def _method_hash(method: str) -> str:
    return str(load_json(PROTOCOL_DIR / "method_configs.json")["methods"][method]["method_config_sha256"])


def _command(method: str, length: int, output: Path, revision: str) -> List[str]:
    spec = INTERNAL_METHODS[method]
    return [
        str(PYTHON), str(REPO_ROOT / "tools/qwen_efficiency_direct_gpu_loader.py"),
        "--model_path", str(QWEN_MODEL_ROOT), "--tokenizer_path", str(QWEN_MODEL_ROOT),
        "--model_family", "qwen2", "--dtype", "fp16", "--device", "cuda",
        "--backend", "flash_attention_2", "--method", spec["runner"],
        "--eval", "latency", "--context_lengths", str(length), "--decode_tokens", "16",
        "--block_strategy", "fixed", "--selection_path", "canonical_fixed_page",
        "--block_size", "64", "--token_unit_size", "8", "--local_query_window", "4",
        "--swap_threshold", "0.9", "--prune_layers", "9,18,26",
        "--retain_tokens", "12288,6144,4096", "--decode_policy", "original",
        "--rescue_policy", spec["policy"], "--rescue_score", spec["score"],
        "--rescue_ratio", "0.05", "--alpha_diff", "0.5", "--lambda_red", "0.25",
        "--eta_rescue", "0.25", "--eta_active", "0.25", "--temperature", "0.0",
        "--top_p", "1.0", "--seed", "0", "--profile_io", "--profile_memory",
        "--profile_cuda_events", "--profile_active_blocks", "--profile_score_overhead",
        "--protocol_id", PROTOCOL_ID, "--implementation_revision", revision,
        "--source_config_hash", _method_hash(method), "--output_dir", str(output),
    ]


def _available_methods(pipeline_dir: Path) -> List[str]:
    methods = [
        "sliminfer_original", "cc_sliminfer_v1", "drm_swap_original_k",
        "sliminfer_larger_km", "sliminfer_drm_add_km", "sliminfer_random_add_km",
    ]
    stage4 = load_json(stage_decision_path(pipeline_dir, 4))
    if stage4.get("decision") == "PASS":
        methods.insert(0, "full_kv_fa2")
    return methods


def _read_latency(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise RuntimeError(f"Expected one latency row in {path}, got {len(rows)}")
    return rows[0]


def _deterministic_cuda_oom(log_path: Path) -> bool:
    if not log_path.exists():
        return False
    text = log_path.read_text(encoding="utf-8", errors="replace")
    fatal_markers = (
        "Xid", "fallen off bus", "Failed to initialize NVML",
        "CUDA driver error", "driver communication error",
    )
    return "torch.OutOfMemoryError: CUDA out of memory" in text and not any(
        marker in text for marker in fatal_markers
    )


def _oom_row(method: str, length: int, repetition: int, log_path: Path) -> Dict[str, Any]:
    return {
        "method": method, "context_length": length, "actual_prompt_len": length,
        "decode_tokens": 16, "repetition": repetition, "status": "oom",
        "resumed": True, "error": "deterministic_cuda_oom", "log": str(log_path),
    }


def _summary(rows: Sequence[Mapping[str, Any]], metric: str) -> Dict[str, Any]:
    values = [float(row[metric]) for row in rows]
    ordered = sorted(values)
    return {
        "median": statistics.median(values), "min": min(values), "max": max(values),
        "std": statistics.pstdev(values),
        "iqr": ordered[(3 * (len(ordered) - 1)) // 4] - ordered[(len(ordered) - 1) // 4],
    }


def run_efficiency(pipeline_dir: Path, *, execute: bool, dry_run: bool = False) -> Dict[str, Any]:
    gate = run_protocol_gate(pipeline_dir, 8)
    if gate["decision"] != "PASS":
        raise RuntimeError("Stage 8 protocol gate failed")
    loader_gate_path = pipeline_dir / "STAGE_08_LOADER_EQUIVALENCE.json"
    loader_gate = load_json(loader_gate_path)
    loader_path = REPO_ROOT / "tools/qwen_efficiency_direct_gpu_loader.py"
    if (
        loader_gate.get("decision") != "PASS"
        or loader_gate.get("loader_sha256") != sha256_file(loader_path)
    ):
        raise RuntimeError("Stage 8 direct-GPU loader equivalence gate is absent or stale")
    pointer = pipeline_dir / "QWEN_EFFICIENCY_ROOT.json"
    if pointer.exists():
        root = Path(load_json(pointer)["path"])
    else:
        root = REPO_ROOT / "results" / f"qwen_unified_efficiency_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        atomic_write_json(pointer, {"created_at": now_iso(), "path": str(root)})
    methods = _available_methods(pipeline_dir)
    cases = [
        (method, length, repetition)
        for method in methods
        for length in (CONTEXT_LENGTHS if method != "sliminfer_random_add_km" else (32768,))
        for repetition in range(1, REPETITIONS + 1)
    ]
    plan = {
        "created_at": now_iso(), "protocol_id": PROTOCOL_ID, "root": str(root),
        "methods": methods, "context_lengths": list(CONTEXT_LENGTHS),
        "random_add_lengths": [32768], "decode_tokens": 16, "repetitions": REPETITIONS,
        "independent_process_per_case": True, "warmup_in_runner": True,
        "cuda_synchronize_in_runner": True, "offload_fill_used": False,
        "cross_machine_absolute_comparison_forbidden": True,
        "model_loader": "direct-gpu-low-cpu-memory-v1",
        "model_loader_sha256": sha256_file(loader_path),
        "loader_equivalence_gate": str(loader_gate_path),
        "loader_only_affects_pre_measurement_model_startup": True,
    }
    atomic_write_json(pipeline_dir / "STAGE_08_EFFICIENCY_PLAN.json", plan)
    if dry_run or not execute:
        result = {"stage": 8, "name": "qwen_efficiency", "decision": "NOT_RUN", **plan}
        atomic_write_json(stage_decision_path(pipeline_dir, 8), result)
        return result
    root.mkdir(parents=True, exist_ok=True)
    policy = load_json(pipeline_dir / "SAFETY_POLICY.json")
    resource_gate = monitor_resources(
        policy, int(policy["thresholds"]["resource_gate_duration_seconds"]),
        int(policy["thresholds"]["resource_gate_interval_seconds"]),
        pipeline_dir=pipeline_dir, allowed_pids=(os.getpid(),),
    )
    atomic_write_json(root / "RESOURCE_GATE.json", resource_gate)
    if not resource_gate["pass"]:
        write_blocker(pipeline_dir, resource_gate)
        decision = {
            "stage": 8, "name": "qwen_efficiency", "decision": "HARD_BLOCKED",
            "created_at": now_iso(), "reason": "resource_gate_failed", "details": resource_gate,
        }
        atomic_write_json(stage_decision_path(pipeline_dir, 8), decision)
        return decision
    stage_swap_baseline = int(
        resource_gate["snapshots"][-1]["host_memory"]["swap_used_bytes"]
    )
    atomic_write_json(root / "STAGE_SWAP_BASELINE.json", {
        "created_at": now_iso(),
        "swap_used_bytes": stage_swap_baseline,
        "max_growth_bytes": int(policy["thresholds"]["max_swap_growth_bytes_during_gate"]),
        "scope": "all resumed Stage 8 cases",
    })
    revision = implementation_revision(RUNTIME_FILES)
    rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    oom_cases: List[Dict[str, Any]] = []
    for case_index, (method, length, repetition) in enumerate(cases, start=1):
        case_dir = root / "cases" / method / str(length) / f"rep_{repetition:02d}"
        latency_path = case_dir / "latency.csv"
        log_path = root / "logs" / f"{method}_{length}_rep{repetition:02d}_direct_gpu.log"
        if latency_path.exists():
            try:
                row = _read_latency(latency_path)
            except RuntimeError:
                if _deterministic_cuda_oom(log_path):
                    outcome = _oom_row(method, length, repetition, log_path)
                    rows.append(outcome)
                    oom_cases.append(outcome)
                    atomic_write_json(case_dir / "CASE_OUTCOME.json", {
                        "created_at": now_iso(), **outcome,
                        "classification": "valid_oom_outcome_not_worker_failure",
                    })
                    continue
            else:
                rows.append({"method": method, "repetition": repetition, **row, "status": "ok", "resumed": True})
                continue
        if case_dir.exists():
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            incomplete = case_dir.with_name(f"{case_dir.name}.incomplete_{stamp}")
            case_dir.rename(incomplete)
        command = _command(method, length, case_dir, revision)
        command_path = case_dir.parent / f"rep_{repetition:02d}_command.json"
        if command_path.exists():
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            command_path.rename(command_path.with_name(
                f"{command_path.stem}.pre_direct_loader_{stamp}{command_path.suffix}"
            ))
        atomic_write_json(command_path, {"command": command})
        log_path.parent.mkdir(parents=True, exist_ok=True)
        success = False
        for attempt in (1, 2):
            with log_path.open("a", encoding="utf-8") as log:
                proc = subprocess.Popen(command, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT)
                pid = load_json(pipeline_dir / "PID.json")
                pid.update({
                    "updated_at": now_iso(), "task_worker_pid": os.getpid(),
                    "gpu_worker_pid": proc.pid, "current_stage": 8,
                    "current_method": method, "current_task": f"{length}/rep{repetition}",
                })
                atomic_write_json(pipeline_dir / "PID.json", pid)
                watchdog = RuntimeWatchdog(
                    pipeline_dir=pipeline_dir, run_root=root, policy=policy, stage=8,
                    method=method, task=f"{length}/rep{repetition}", worker_pid=proc.pid,
                    prediction_path=None, trace_path=None, log_path=log_path,
                    resume_command=(pipeline_dir / "RESUME_COMMAND.txt").read_text(encoding="utf-8").strip(),
                    interval_seconds=5,
                    swap_baseline_bytes=stage_swap_baseline,
                )
                watchdog.start()
                returncode = proc.wait()
                watchdog_failure = watchdog.stop()
            if watchdog_failure:
                decision = {
                    "stage": 8, "name": "qwen_efficiency", "decision": "HARD_BLOCKED",
                    "created_at": now_iso(), "reason": "runtime_watchdog_failure",
                    "details": watchdog_failure,
                }
                atomic_write_json(stage_decision_path(pipeline_dir, 8), decision)
                return decision
            if returncode == 0 and latency_path.exists():
                success = True
                break
        if success:
            row = _read_latency(latency_path)
            rows.append({"method": method, "repetition": repetition, **row, "status": "ok", "resumed": False})
        elif _deterministic_cuda_oom(log_path):
            outcome = _oom_row(method, length, repetition, log_path)
            rows.append(outcome)
            oom_cases.append(outcome)
            atomic_write_json(case_dir / "CASE_OUTCOME.json", {
                "created_at": now_iso(), **outcome,
                "classification": "valid_oom_outcome_not_worker_failure",
            })
        else:
            failures.append({
                "method": method, "context_length": length, "repetition": repetition,
                "returncode": returncode, "log": str(log_path), "attempts": 2,
            })
        atomic_write_json(pipeline_dir / "PROGRESS.json", {
            "updated_at": now_iso(), "stage": 8, "method": method,
            "task": f"{length}/rep{repetition}", "case_index": case_index,
            "case_total": len(cases), "completed_cases": case_index,
        })
    atomic_write_csv(root / "efficiency_raw.csv", rows, [
        "method", "context_length", "actual_prompt_len", "decode_tokens", "repetition",
        "TTFT_s", "E2E_s", "peak_gpu_mem_GB", "prompt_kv_mem_GB", "h2d_GB", "d2h_GB",
        "async_kv", "swap_in_blocks", "swap_out_blocks", "swap_count", "score_overhead_ms",
        "score_calls", "active_set_churn", "jaccard_vs_original", "status", "resumed",
        "error", "log",
    ])
    summary_rows: List[Dict[str, Any]] = []
    for method in methods:
        lengths = CONTEXT_LENGTHS if method != "sliminfer_random_add_km" else (32768,)
        for length in lengths:
            group = [
                row for row in rows
                if row["method"] == method and int(row["context_length"]) == length
                and row.get("status") == "ok"
            ]
            if not group:
                continue
            ttft = _summary(group, "TTFT_s")
            e2e = _summary(group, "E2E_s")
            vram = _summary(group, "peak_gpu_mem_GB")
            summary_rows.append({
                "method": method, "context_length": length, "repetitions": len(group),
                "ttft_median_s": ttft["median"], "ttft_min_s": ttft["min"],
                "ttft_max_s": ttft["max"], "ttft_std_s": ttft["std"], "ttft_iqr_s": ttft["iqr"],
                "e2e_median_s": e2e["median"], "e2e_min_s": e2e["min"],
                "e2e_max_s": e2e["max"], "e2e_std_s": e2e["std"], "e2e_iqr_s": e2e["iqr"],
                "peak_allocated_vram_median_gb": vram["median"],
                "peak_reserved_vram_gb": "unavailable_in_frozen_runner",
                "selector_overhead_median_ms": statistics.median(float(row["score_overhead_ms"]) for row in group),
                "h2d_median_gb": statistics.median(float(row["h2d_GB"]) for row in group),
                "d2h_median_gb": statistics.median(float(row["d2h_GB"]) for row in group),
            })
    atomic_write_csv(root / "efficiency_summary.csv", summary_rows, [
        "method", "context_length", "repetitions", "ttft_median_s", "ttft_min_s", "ttft_max_s",
        "ttft_std_s", "ttft_iqr_s", "e2e_median_s", "e2e_min_s", "e2e_max_s", "e2e_std_s",
        "e2e_iqr_s", "peak_allocated_vram_median_gb", "peak_reserved_vram_gb",
        "selector_overhead_median_ms", "h2d_median_gb", "d2h_median_gb",
    ])
    atomic_write_json(root / "FAILED_CASES.json", failures)
    atomic_write_json(root / "OOM_CASES.json", oom_cases)
    atomic_write_text(
        root / "KNOWN_LIMITATIONS.md",
        "# Known Limitations\n\nThe frozen latency runner records peak allocated VRAM but does not expose "
        "`torch.cuda.max_memory_reserved`; that field is explicitly unavailable. External methods are included "
        "only when their official quality stage passes. Deterministic CUDA OOM cases are retained as explicit "
        f"outcomes ({len(oom_cases)} cases) and are not imputed. No LLaMA-machine timings are merged.\n",
    )
    decision = {
        "stage": 8, "name": "qwen_efficiency", "created_at": now_iso(),
        "decision": "SOFT_BLOCKED" if failures else "PASS", "run_root": str(root),
        "completed_cases": len(rows),
        "measured_cases": sum(row.get("status") == "ok" for row in rows),
        "oom_cases": len(oom_cases), "failed_cases": len(failures), "next_stage": 9,
        "reason": "one_or_more_efficiency_cases_failed" if failures else None,
    }
    atomic_write_json(stage_decision_path(pipeline_dir, 8), decision)
    if failures:
        atomic_write_text(
            pipeline_dir / "STAGE_08_SOFT_BLOCKER.md",
            "# Stage 08 Soft Blocker\n\nOne or more isolated efficiency cases failed after one retry. "
            "See `FAILED_CASES.json`; quality artifacts are unaffected and Stage 9 may continue.\n",
        )
    else:
        stale = pipeline_dir / "STAGE_08_SOFT_BLOCKER.md"
        if stale.exists():
            stale.rename(pipeline_dir / "STAGE_08_SOFT_BLOCKER.resolved.md")
    return decision


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = run_efficiency(args.pipeline_dir.resolve(), execute=args.execute, dry_run=args.dry_run)
    print(json.dumps(result, indent=2))
    return 0 if result.get("decision") in {"PASS", "SOFT_BLOCKED", "NOT_RUN"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
