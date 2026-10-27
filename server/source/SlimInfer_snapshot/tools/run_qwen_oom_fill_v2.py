#!/usr/bin/env python3
"""Equivalence-gate and run final, immutable Qwen OOM offload fills.

This Stage-11 implementation is deliberately independent of the raw runners.
It runs one sample per process, compares direct-GPU and offloaded executions on
20 method-sample cases, and only then attempts the recorded OOM rows.  Raw
prediction files are hash-locked and are never edited.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

_REPO_BOOTSTRAP = Path(__file__).resolve().parents[1]
if str(_REPO_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(_REPO_BOOTSTRAP))

from tools.consolidate_official_repro_results import per_sample_score
from tools.qwen_pipeline_common import (
    DATA_ROOT,
    PROTOCOL_ID,
    PYTHON,
    QWEN_MODEL_ROOT,
    REPO_ROOT,
    TASKS,
    atomic_write_csv,
    atomic_write_json,
    atomic_write_text,
    canonical_bytes,
    load_json,
    now_iso,
    prediction_status,
    sha256_bytes,
    sha256_file,
    stage_decision_path,
)
from tools.qwen_protocol_gate import FROZEN_IMPLEMENTATION_REVISION, run_protocol_gate
from tools.qwen_resource_guard import collect_snapshot, monitor_resources, write_blocker


ACCEPTED = {"PASS", "SOFT_BLOCKED"}
ADAPTER = REPO_ROOT / "tools/qwen_whole_model_offload_adapter.py"
PRIMARY_PROFILE = "input_embedding_cpu"
FALLBACK_PROFILE = "input_embedding_lm_head_cpu"
PROFILES = (PRIMARY_PROFILE, FALLBACK_PROFILE)
DRIVER_MARKERS = (
    "xid",
    "nvrm",
    "gpu has fallen off the bus",
    "cuda driver error",
    "unspecified launch failure",
    "illegal memory access",
)
METHOD_SPECS: Dict[str, Dict[str, str]] = {
    "sliminfer_original": {
        "runner": "sliminfer_original", "policy": "none", "score": "delta_o",
    },
    "cc_sliminfer_v1": {
        "runner": "cc_sliminfer_v1", "policy": "none", "score": "delta_o",
    },
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
    "full_kv_fa2": {
        "runner": "fullkv", "policy": "none", "score": "delta_o",
    },
}

# Every compressed case is above the first 12,288-token budget in the canonical
# CC records, so the gate tests selection as well as generation.  The 20 cases
# cover all seven runtime configurations and six LongBench task shapes.
CANARY_CASES: Tuple[Tuple[str, str, int], ...] = (
    ("sliminfer_original", "hotpotqa", 37),
    ("sliminfer_original", "2wikimqa", 130),
    ("sliminfer_original", "repobench-p", 346),
    ("cc_sliminfer_v1", "hotpotqa", 70),
    ("cc_sliminfer_v1", "musique", 76),
    ("cc_sliminfer_v1", "lcc", 54),
    ("drm_swap_original_k", "qasper", 16),
    ("drm_swap_original_k", "2wikimqa", 104),
    ("drm_swap_original_k", "repobench-p", 218),
    ("sliminfer_larger_km", "hotpotqa", 130),
    ("sliminfer_larger_km", "musique", 56),
    ("sliminfer_larger_km", "lcc", 200),
    ("sliminfer_random_add_km", "qasper", 15),
    ("sliminfer_random_add_km", "2wikimqa", 126),
    ("sliminfer_random_add_km", "repobench-p", 422),
    ("sliminfer_drm_add_km", "hotpotqa", 132),
    ("sliminfer_drm_add_km", "musique", 170),
    ("sliminfer_drm_add_km", "lcc", 299),
    ("full_kv_fa2", "hotpotqa", 63),
    ("full_kv_fa2", "repobench-p", 53),
)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _method_hashes() -> Dict[str, str]:
    hashes: Dict[str, str] = {}
    protocol = load_json(REPO_ROOT / "protocol/unified_longbench_v1/method_configs.json")
    for method in ("cc_sliminfer_v1", "full_kv_fa2"):
        hashes[method] = str(protocol["methods"][method]["method_config_sha256"])
    historical = _read_csv(
        REPO_ROOT
        / "results/drm_add_official_17task_full_20260715_084131/FULL17_METHOD_SPEC.csv"
    )
    for row in historical:
        if row.get("method") in METHOD_SPECS:
            hashes[str(row["method"])] = str(row["method_hash"])
    missing = sorted(set(METHOD_SPECS) - set(hashes))
    if missing:
        raise RuntimeError(f"Missing frozen method hashes: {missing}")
    return hashes


def _collect_sources(pipeline_dir: Path) -> Dict[Tuple[str, str], Path]:
    stage3 = load_json(stage_decision_path(pipeline_dir, 3))
    sources: Dict[Tuple[str, str], Path] = {}
    for row in _read_csv(Path(stage3["export_root"]) / "SOURCE_MAP.csv"):
        sources[(str(row["method"]), str(row["task"]))] = Path(row["path"])
    stage4 = load_json(stage_decision_path(pipeline_dir, 4))
    full_root = Path(stage4["run_root"]) / "full/tasks"
    for task in TASKS:
        sources[("full_kv_fa2", task)] = (
            full_root / task / "run/longbench_predictions" / f"{task}.jsonl"
        )
    missing = [key for key, path in sources.items() if not path.exists()]
    if missing:
        raise RuntimeError(f"Prediction sources are missing: {missing[:10]}")
    return sources


def _collect_oom_rows(pipeline_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    stage3 = load_json(stage_decision_path(pipeline_dir, 3))
    rows.extend(_read_csv(Path(stage3["export_root"]) / "OOM_MANIFEST.csv"))
    stage4 = load_json(stage_decision_path(pipeline_dir, 4))
    rows.extend(_read_csv(Path(stage4["run_root"]) / "full/OOM_MANIFEST.csv"))
    normalized: List[Dict[str, Any]] = []
    seen = set()
    for row in rows:
        key = (str(row["method"]), str(row["task"]), int(row["sample_index"]))
        if key in seen:
            raise RuntimeError(f"Duplicate raw OOM row: {key}")
        seen.add(key)
        normalized.append({
            "method": key[0], "task": key[1], "sample_index": key[2],
            "sample_id": row.get("sample_id"), "raw_status": row.get("status"),
            "raw_error": row.get("error"),
        })
    normalized.sort(key=lambda row: (row["method"], row["task"], row["sample_index"]))
    return normalized


def _source_row(path: Path, sample_index: int) -> Dict[str, Any]:
    matches = [
        row for row in _read_jsonl(path)
        if int(row.get("sample_index", -1)) == int(sample_index)
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one source row for index {sample_index} in {path}, got {len(matches)}")
    return matches[0]


def _raw_hash_lock(sources: Mapping[Tuple[str, str], Path]) -> List[Dict[str, Any]]:
    return [
        {"method": method, "task": task, "path": str(path), "sha256": sha256_file(path)}
        for (method, task), path in sorted(sources.items())
    ]


def _verify_raw_hash_lock(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, str]]:
    failures = []
    for row in rows:
        path = Path(str(row["path"]))
        actual = sha256_file(path) if path.exists() else "MISSING"
        if actual != row["sha256"]:
            failures.append({"path": str(path), "expected": str(row["sha256"]), "actual": actual})
    return failures


def _case_id(method: str, task: str, sample_index: int) -> str:
    return f"{method}__{task}__{sample_index:04d}"


def _build_command(
    *,
    method: str,
    task: str,
    sample_index: int,
    output_root: Path,
    loader_mode: str,
    profile: str,
    method_hashes: Mapping[str, str],
) -> List[str]:
    spec = METHOD_SPECS[method]
    trace = spec["runner"] != "fullkv"
    command = [
        str(PYTHON), str(ADAPTER),
        "--loader-mode", loader_mode,
        "--offload-profile", profile,
        "--offload-loader-metadata-path", str(output_root / "LOADER_METADATA.json"),
        "--model_path", str(QWEN_MODEL_ROOT),
        "--tokenizer_path", str(QWEN_MODEL_ROOT),
        "--model_family", "qwen2",
        "--dtype", "fp16",
        "--device", "cuda",
        "--backend", "flash_attention_2",
        "--longbench_dir", str(DATA_ROOT),
        "--longbench_tasks", task,
        "--longbench_max_samples", "0",
        "--longbench_sample_indices_by_task", f"{task}:{sample_index}",
        "--max_model_len", "131072",
        "--method", spec["runner"],
        "--eval", "longbench",
        "--block_strategy", "fixed",
        "--selection_path", "canonical_fixed_page",
        "--block_size", "64",
        "--token_unit_size", "8",
        "--local_query_window", "4",
        "--swap_threshold", "0.9",
        "--prune_layers", "9,18,26",
        "--retain_tokens", "12288,6144,4096",
        "--decode_policy", "original",
        "--rescue_policy", spec["policy"],
        "--rescue_score", spec["score"],
        "--rescue_ratio", "0.05",
        "--alpha_diff", "0.5",
        "--lambda_red", "0.25",
        "--eta_rescue", "0.25",
        "--eta_active", "0.25",
        "--temperature", "0.0",
        "--top_p", "1.0",
        "--generation_max_new_tokens", "0",
        "--seed", "0",
        "--min_free_gpu_gb", "0",
        "--protocol_id", PROTOCOL_ID,
        "--implementation_revision", FROZEN_IMPLEMENTATION_REVISION,
        "--source_config_hash", method_hashes[method],
        "--active_block_trace", "full" if trace else "none",
        "--rescue_audit_path", str(output_root / "audits/rescue_policy.jsonl"),
        "--cc_selection_audit_path", str(output_root / "audits/cc_selection.jsonl"),
        "--debug_used_block_audit_path", str(output_root / "audits/used_blocks.jsonl"),
        "--output_dir", str(output_root / "run"),
    ]
    if trace:
        command.extend(["--active_block_trace_dir", str(output_root / "active_block_trace")])
    return command


def _prediction_path(root: Path, task: str) -> Path:
    return root / "run/longbench_predictions" / f"{task}.jsonl"


def _trace_path(root: Path, task: str) -> Path:
    return root / "active_block_trace" / f"{task}.jsonl.gz"


def _driver_failure(log_text: str) -> Optional[str]:
    lowered = log_text.lower()
    return next((marker for marker in DRIVER_MARKERS if marker in lowered), None)


def _case_resource_gate(
    policy: Mapping[str, Any],
    *,
    output_path: Path,
    wait_seconds: int = 90,
) -> Dict[str, Any]:
    thresholds = policy["thresholds"]
    deadline = time.monotonic() + wait_seconds
    snapshots = []
    while True:
        snapshot = collect_snapshot(REPO_ROOT)
        gpu = snapshot["gpu"]
        host = snapshot["host_memory"]
        disk = snapshot["disk"]
        non_desktop = [
            row for row in snapshot["gpu_compute_processes"]
            if not row.get("desktop_allowlisted")
        ]
        failures = []
        if not gpu.get("available"):
            failures.append("gpu_unavailable")
        else:
            if int(gpu["memory_free_mib"]) < int(thresholds["min_gpu_free_mib_before_model"]):
                failures.append("gpu_memory_not_released")
            if int(gpu["utilization_gpu_percent"]) > int(thresholds["max_idle_gpu_utilization_percent"]):
                failures.append("gpu_not_idle")
            if int(gpu["temperature_gpu_c"]) > int(thresholds["max_gpu_temperature_c"]):
                failures.append("gpu_temperature")
        if non_desktop:
            failures.append("non_desktop_gpu_process")
        if snapshot["experiment_processes"]:
            failures.append("other_experiment_process")
        if int(host["ram_available_bytes"]) < int(thresholds["min_host_available_bytes"]):
            failures.append("host_memory")
        if int(disk["free_bytes"]) < int(thresholds["min_disk_free_bytes"]):
            failures.append("disk_space")
        if snapshot["recent_driver_errors"].get("matches"):
            failures.append("recent_driver_error")
        snapshots.append({
            "captured_at": snapshot["captured_at"], "failures": failures,
            "gpu": gpu, "gpu_compute_processes": snapshot["gpu_compute_processes"],
            "experiment_processes": snapshot["experiment_processes"],
            "host_memory": host, "disk": disk,
            "recent_driver_errors": snapshot["recent_driver_errors"],
        })
        if not failures:
            report = {
                "created_at": now_iso(), "pass": True,
                "attempts": len(snapshots), "snapshots": snapshots,
            }
            atomic_write_json(output_path, report)
            return report
        transient = set(failures).issubset({"gpu_memory_not_released", "gpu_not_idle"})
        if not transient or non_desktop or snapshot["experiment_processes"] or time.monotonic() >= deadline:
            report = {
                "created_at": now_iso(), "pass": False,
                "attempts": len(snapshots), "failures": failures,
                "snapshots": snapshots,
            }
            atomic_write_json(output_path, report)
            return report
        time.sleep(5)


def _run_isolated_case(
    *,
    pipeline_dir: Path,
    fill_root: Path,
    method: str,
    task: str,
    sample_index: int,
    loader_mode: str,
    profile: str,
    output_root: Path,
    method_hashes: Mapping[str, str],
    timeout_seconds: int,
    environment: Mapping[str, Any],
    safety_policy: Mapping[str, Any],
) -> Dict[str, Any]:
    decision_path = output_root / "CASE_DECISION.json"
    if decision_path.exists():
        previous = load_json(decision_path)
        prediction = _prediction_path(output_root, task)
        if previous.get("decision") == "PASS" and prediction.exists():
            prediction_ok = sha256_file(prediction) == previous.get("prediction_sha256")
            recorded_trace = previous.get("trace_sha256")
            trace = _trace_path(output_root, task)
            trace_ok = (
                recorded_trace is None
                or (trace.exists() and sha256_file(trace) == recorded_trace)
            )
            if not prediction_ok or not trace_ok:
                raise RuntimeError(
                    f"Completed case hash mismatch: {output_root}; "
                    f"prediction_ok={prediction_ok}, trace_ok={trace_ok}"
                )
            return {**previous, "resumed": True}

    output_root.mkdir(parents=True, exist_ok=True)
    case_gate = _case_resource_gate(
        safety_policy, output_path=output_root / "RESOURCE_GATE.json"
    )
    if not case_gate["pass"]:
        raise RuntimeError(
            f"Per-case resource gate failed for {method}/{task}/{sample_index}: "
            f"{case_gate.get('failures')}"
        )
    command = _build_command(
        method=method, task=task, sample_index=sample_index,
        output_root=output_root, loader_mode=loader_mode, profile=profile,
        method_hashes=method_hashes,
    )
    atomic_write_json(output_root / "COMMAND.json", {
        "created_at": now_iso(), "command": command, "cwd": str(REPO_ROOT),
        "one_sample_per_process": True, "timeout_seconds": timeout_seconds,
    })
    log_path = output_root / "worker.log"
    env = os.environ.copy()
    env.update({key: str(value) for key, value in environment.items() if str(key).isupper()})
    started = time.monotonic()
    timed_out = False
    with log_path.open("a", encoding="utf-8") as log:
        proc = subprocess.Popen(
            command, cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        atomic_write_json(pipeline_dir / "PID.json", {
            "updated_at": now_iso(), "pipeline_pid": os.getpid(),
            "gpu_worker_pid": proc.pid, "current_stage": 11,
            "current_method": method, "current_task": task,
            "current_sample_index": sample_index, "loader_mode": loader_mode,
            "offload_profile": profile,
        })
        try:
            returncode = proc.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                returncode = proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                returncode = proc.wait(timeout=30)
    elapsed = time.monotonic() - started
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    driver_marker = _driver_failure(log_text)
    predictions = _read_jsonl(_prediction_path(output_root, task))
    row = predictions[0] if len(predictions) == 1 else None
    identity_ok = bool(row) and int(row.get("sample_index", -1)) == int(sample_index)
    status = prediction_status(row) if row else "missing"
    decision = "PASS" if returncode == 0 and not timed_out and not driver_marker and identity_ok else "FAIL"
    report = {
        "created_at": now_iso(), "decision": decision,
        "method": method, "task": task, "sample_index": sample_index,
        "loader_mode": loader_mode, "offload_profile": profile,
        "worker_pid": proc.pid, "returncode": returncode,
        "timed_out": timed_out, "driver_failure_marker": driver_marker,
        "elapsed_seconds": round(elapsed, 6), "prediction_rows": len(predictions),
        "prediction_identity_ok": identity_ok, "prediction_status": status,
        "prediction_path": str(_prediction_path(output_root, task)),
        "prediction_sha256": sha256_file(_prediction_path(output_root, task)) if predictions else None,
        "trace_path": str(_trace_path(output_root, task)),
        "trace_sha256": sha256_file(_trace_path(output_root, task)) if _trace_path(output_root, task).exists() else None,
        "log": str(log_path),
    }
    atomic_write_json(decision_path, report)
    atomic_write_json(fill_root / "PROGRESS.json", report)
    if decision != "PASS":
        raise RuntimeError(f"Isolated worker failed: {method}/{task}/{sample_index}: {report}")
    return report


def _prediction_semantics(row: Mapping[str, Any]) -> Dict[str, Any]:
    fields = (
        "sample_index", "sample_id", "content_sha256", "prompt_sha256",
        "input_token_ids_sha256", "input_tokens", "max_new_tokens",
        "generation_config", "generation_config_sha256", "method",
        "source_config_hash", "protocol_id", "implementation_revision",
        "runtime_dtype", "attention_implementation", "status", "pred",
        "prediction", "output_tokens", "answers", "all_classes", "length",
    )
    return {field: row.get(field) for field in fields}


def _selection_value(key: str, value: Any) -> bool:
    if key.endswith("_id") or key.endswith("_ids"):
        return True
    tokens = (
        "selected", "candidate", "budget", "sink", "duplicate", "invalid",
        "preservation", "added", "removed", "swap", "over_budget", "under_budget",
        "rescue_policy", "rescue_score", "mmr", "eligible", "applied", "rejected",
    )
    return any(token in key for token in tokens) or key in {
        "task", "sample_index", "decode_step", "selection_event_index",
        "layer_idx", "abs_layer", "phase", "method", "mode", "triggered",
        "num_keep_target", "trace_level", "trace_source",
    }


def _selection_projection(value: Any) -> Any:
    if isinstance(value, list):
        return [_selection_projection(item) for item in value]
    if isinstance(value, dict):
        return {
            key: _selection_projection(item)
            for key, item in sorted(value.items())
            if _selection_value(str(key), item)
        }
    return value


def _selection_payload(root: Path, task: str) -> Dict[str, Any]:
    active_path = _trace_path(root, task)
    active = [_selection_projection(row) for row in _read_jsonl(active_path)]
    audits: Dict[str, Any] = {}
    audit_root = root / "audits"
    if audit_root.exists():
        for path in sorted(audit_root.glob("*.jsonl")):
            audits[path.name] = [_selection_projection(row) for row in _read_jsonl(path)]
    payload = {"active_trace": active, "audits": audits}
    payload["sha256"] = sha256_bytes(canonical_bytes(payload))
    payload["active_events"] = len(active)
    payload["audit_events"] = sum(len(rows) for rows in audits.values())
    return payload


def _compare_canary_case(
    method: str,
    task: str,
    sample_index: int,
    direct_root: Path,
    offload_root: Path,
) -> Dict[str, Any]:
    direct_rows = _read_jsonl(_prediction_path(direct_root, task))
    offload_rows = _read_jsonl(_prediction_path(offload_root, task))
    if len(direct_rows) != 1 or len(offload_rows) != 1:
        return {"decision": "FAIL", "reason": "prediction_row_count"}
    direct_pred = _prediction_semantics(direct_rows[0])
    offload_pred = _prediction_semantics(offload_rows[0])
    direct_selection = _selection_payload(direct_root, task)
    offload_selection = _selection_payload(offload_root, task)
    prediction_equal = direct_pred == offload_pred
    selection_required = METHOD_SPECS[method]["runner"] != "fullkv"
    selection_equal = direct_selection["sha256"] == offload_selection["sha256"]
    selection_present = direct_selection["active_events"] > 0 and offload_selection["active_events"] > 0
    passed = (
        prediction_equal
        and prediction_status(direct_rows[0]) == "ok"
        and prediction_status(offload_rows[0]) == "ok"
        and (not selection_required or (selection_present and selection_equal))
    )
    return {
        "decision": "PASS" if passed else "FAIL",
        "method": method, "task": task, "sample_index": sample_index,
        "prediction_equal": prediction_equal,
        "direct_status": prediction_status(direct_rows[0]),
        "offload_status": prediction_status(offload_rows[0]),
        "selection_required": selection_required,
        "selection_present": selection_present,
        "selection_equal": selection_equal,
        "direct_selection_sha256": direct_selection["sha256"],
        "offload_selection_sha256": offload_selection["sha256"],
        "direct_active_events": direct_selection["active_events"],
        "offload_active_events": offload_selection["active_events"],
        "direct_prediction": direct_pred,
        "offload_prediction": offload_pred,
    }


def _run_equivalence_gate(
    *,
    pipeline_dir: Path,
    fill_root: Path,
    profile: str,
    method_hashes: Mapping[str, str],
    timeout_seconds: int,
    environment: Mapping[str, Any],
    safety_policy: Mapping[str, Any],
) -> Dict[str, Any]:
    gate_root = fill_root / "equivalence_gate"
    cases = []
    for position, (method, task, sample_index) in enumerate(CANARY_CASES, start=1):
        case_id = _case_id(method, task, sample_index)
        direct_root = gate_root / "direct" / case_id
        offload_root = gate_root / profile / case_id
        _run_isolated_case(
            pipeline_dir=pipeline_dir, fill_root=fill_root, method=method, task=task,
            sample_index=sample_index, loader_mode="direct", profile=profile,
            output_root=direct_root, method_hashes=method_hashes,
            timeout_seconds=timeout_seconds, environment=environment,
            safety_policy=safety_policy,
        )
        _run_isolated_case(
            pipeline_dir=pipeline_dir, fill_root=fill_root, method=method, task=task,
            sample_index=sample_index, loader_mode="offload", profile=profile,
            output_root=offload_root, method_hashes=method_hashes,
            timeout_seconds=timeout_seconds, environment=environment,
            safety_policy=safety_policy,
        )
        comparison = _compare_canary_case(
            method, task, sample_index, direct_root, offload_root
        )
        comparison.update({"case": position, "case_id": case_id, "profile": profile})
        atomic_write_json(offload_root / "EQUIVALENCE_DECISION.json", comparison)
        cases.append(comparison)
        atomic_write_json(fill_root / "PROGRESS.json", {
            "created_at": now_iso(), "stage": 11, "phase": "equivalence_gate",
            "profile": profile, "completed_cases": position,
            "total_cases": len(CANARY_CASES), "latest": comparison,
        })
        if comparison["decision"] != "PASS":
            break
    report = {
        "created_at": now_iso(), "profile": profile,
        "decision": "PASS" if len(cases) == len(CANARY_CASES) and all(
            row["decision"] == "PASS" for row in cases
        ) else "FAIL",
        "required_cases": len(CANARY_CASES), "completed_cases": len(cases),
        "prediction_matches": sum(bool(row.get("prediction_equal")) for row in cases),
        "selection_matches": sum(bool(row.get("selection_equal")) for row in cases),
        "cases": cases,
    }
    atomic_write_json(gate_root / f"{profile}_EQUIVALENCE_DECISION.json", report)
    if report["decision"] != "PASS":
        raise RuntimeError(f"Offload equivalence gate failed for {profile}")
    return report


def _score_offload_completed(
    *,
    fill_root: Path,
    sources: Mapping[Tuple[str, str], Path],
    fills: Mapping[Tuple[str, str, int], Mapping[str, Any]],
) -> Dict[str, Any]:
    sample_scores: List[Dict[str, Any]] = []
    task_scores: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []
    merged_rows: List[Dict[str, Any]] = []
    for method in METHOD_SPECS:
        method_task_scores = []
        method_fills = 0
        method_remaining_oom = 0
        for task in TASKS:
            raw_rows = _read_jsonl(sources[(method, task)])
            task_values = []
            for raw in raw_rows:
                sample_index = int(raw.get("sample_index", -1))
                fill = fills.get((method, task, sample_index))
                if fill and fill.get("prediction_status") == "ok":
                    prediction = dict(fill["prediction"])
                    prediction_source = "offload_fill"
                    method_fills += 1
                else:
                    prediction = dict(raw)
                    prediction_source = "raw"
                status = prediction_status(prediction)
                if status == "oom":
                    method_remaining_oom += 1
                pred = str(prediction.get("pred", prediction.get("prediction", "")) or "")
                value = per_sample_score(
                    task, pred, list(prediction.get("answers", [])), prediction.get("all_classes", [])
                )
                task_values.append(value)
                sample_scores.append({
                    "method": method, "task": task, "sample_index": sample_index,
                    "sample_id": prediction.get("sample_id"), "status": status,
                    "prediction_source": prediction_source, "score": value,
                })
                merged_rows.append({
                    "method_alias": method, "task": task,
                    "prediction_source": prediction_source, **prediction,
                })
            score = round(sum(task_values) / max(1, len(task_values)), 4)
            method_task_scores.append(score)
            task_scores.append({
                "method": method, "task": task, "samples": len(task_values), "score": score,
            })
        summaries.append({
            "method": method, "task_macro": round(sum(method_task_scores) / len(method_task_scores), 4),
            "filled_oom_rows": method_fills, "remaining_oom_rows": method_remaining_oom,
        })
    atomic_write_csv(
        fill_root / "offload_completed_scores_by_sample.csv", sample_scores,
        ["method", "task", "sample_index", "sample_id", "status", "prediction_source", "score"],
    )
    atomic_write_csv(
        fill_root / "offload_completed_scores_by_task.csv", task_scores,
        ["method", "task", "samples", "score"],
    )
    atomic_write_csv(
        fill_root / "offload_completed_scores_summary.csv", summaries,
        ["method", "task_macro", "filled_oom_rows", "remaining_oom_rows"],
    )
    _write_jsonl(fill_root / "offload_completed_predictions.jsonl", merged_rows)
    return {"summaries": summaries, "merged_prediction_rows": len(merged_rows)}


def build_and_run(
    pipeline_dir: Path,
    safety_policy_path: Path,
    *,
    execute: bool,
    timeout_seconds: int,
) -> Dict[str, Any]:
    gate = run_protocol_gate(pipeline_dir, 11)
    if gate["decision"] != "PASS":
        raise RuntimeError("Stage 11 protocol gate failed")
    prerequisite_failures = []
    for stage in range(11):
        decision_path = stage_decision_path(pipeline_dir, stage)
        decision = load_json(decision_path).get("decision") if decision_path.exists() else None
        if decision not in ACCEPTED:
            prerequisite_failures.append({"stage": stage, "decision": decision})
    if prerequisite_failures:
        raise RuntimeError(f"OOM fill is final-only; prerequisites not accepted: {prerequisite_failures}")
    if not ADAPTER.exists():
        raise RuntimeError(f"Offload adapter is absent: {ADAPTER}")

    pointer = pipeline_dir / "QWEN_OOM_FILL_ROOT_V2.json"
    if pointer.exists():
        fill_root = Path(load_json(pointer)["path"])
    else:
        fill_root = REPO_ROOT / "results" / f"qwen_oom_offload_fill_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        atomic_write_json(pointer, {"created_at": now_iso(), "path": str(fill_root)})
    sources = _collect_sources(pipeline_dir)
    oom_rows = _collect_oom_rows(pipeline_dir)
    method_hashes = _method_hashes()
    raw_lock = _raw_hash_lock(sources)
    plan = {
        "created_at": now_iso(), "stage": 11, "protocol_id": PROTOCOL_ID,
        "decision": "NOT_RUN", "execute_requested": execute,
        "fill_root": str(fill_root), "raw_oom_rows": len(oom_rows),
        "unique_oom_samples": len({(row["task"], row["sample_index"]) for row in oom_rows}),
        "equivalence_cases_per_used_profile": len(CANARY_CASES),
        "profiles": list(PROFILES), "one_sample_per_process": True,
        "timeout_seconds": timeout_seconds, "raw_predictions_immutable": True,
        "adapter": str(ADAPTER), "adapter_sha256": sha256_file(ADAPTER),
        "method_hashes": method_hashes,
    }
    atomic_write_json(pipeline_dir / "STAGE_11_OOM_FILL_PLAN_V2.json", plan)
    if not execute:
        return plan

    fill_root.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(
        fill_root / "raw_oom_failed_manifest.csv", oom_rows,
        ["method", "task", "sample_index", "sample_id", "raw_status", "raw_error"],
    )
    atomic_write_json(fill_root / "RAW_UNCHANGED_HASH_LOCK.json", {
        "created_at": now_iso(), "raw_predictions_immutable": True, "files": raw_lock,
    })
    for row in oom_rows:
        source = sources[(row["method"], row["task"])]
        raw = _source_row(source, row["sample_index"])
        if prediction_status(raw) != "oom":
            raise RuntimeError(f"OOM manifest/source mismatch: {row}")

    policy = load_json(safety_policy_path)
    resource_gate = monitor_resources(
        policy,
        int(policy["thresholds"]["resource_gate_duration_seconds"]),
        int(policy["thresholds"]["resource_gate_interval_seconds"]),
        pipeline_dir=pipeline_dir,
        allowed_pids=(os.getpid(),),
    )
    atomic_write_json(fill_root / "RESOURCE_GATE.json", resource_gate)
    if not resource_gate["pass"]:
        write_blocker(pipeline_dir, resource_gate)
        raise RuntimeError("Stage 11 resource gate failed")

    fills: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    gate_reports: Dict[str, Dict[str, Any]] = {}
    unresolved = list(oom_rows)
    for profile in PROFILES:
        if not unresolved:
            break
        gate_reports[profile] = _run_equivalence_gate(
            pipeline_dir=pipeline_dir, fill_root=fill_root, profile=profile,
            method_hashes=method_hashes, timeout_seconds=timeout_seconds,
            environment=policy["environment"], safety_policy=policy,
        )
        next_unresolved = []
        for position, row in enumerate(unresolved, start=1):
            key = (row["method"], row["task"], int(row["sample_index"]))
            output_root = fill_root / "fills" / _case_id(*key) / profile
            result = _run_isolated_case(
                pipeline_dir=pipeline_dir, fill_root=fill_root,
                method=key[0], task=key[1], sample_index=key[2],
                loader_mode="offload", profile=profile, output_root=output_root,
                method_hashes=method_hashes, timeout_seconds=timeout_seconds,
                environment=policy["environment"], safety_policy=policy,
            )
            prediction = _read_jsonl(_prediction_path(output_root, key[1]))[0]
            fill = {
                **result, "prediction": prediction,
                "raw_source_path": str(sources[(key[0], key[1])]),
                "raw_source_sha256": next(
                    item["sha256"] for item in raw_lock
                    if item["method"] == key[0] and item["task"] == key[1]
                ),
            }
            if result["prediction_status"] == "ok":
                fills[key] = fill
            elif result["prediction_status"] == "oom":
                next_unresolved.append(row)
            else:
                raise RuntimeError(f"Invalid fill status for {key}: {result['prediction_status']}")
            atomic_write_json(fill_root / "PROGRESS.json", {
                "created_at": now_iso(), "stage": 11, "phase": "oom_fill",
                "profile": profile, "completed_in_profile": position,
                "attempted_in_profile": len(unresolved), "filled_total": len(fills),
                "remaining_after_current": len(next_unresolved) + len(unresolved) - position,
                "latest": {key_name: fill.get(key_name) for key_name in (
                    "method", "task", "sample_index", "prediction_status", "elapsed_seconds",
                )},
            })
            failures = _verify_raw_hash_lock(raw_lock)
            if failures:
                raise RuntimeError(f"Raw prediction hash changed during fill: {failures[:3]}")
        unresolved = next_unresolved

    fill_manifest = []
    for row in oom_rows:
        key = (row["method"], row["task"], int(row["sample_index"]))
        fill = fills.get(key)
        fill_manifest.append({
            **row,
            "fill_status": fill.get("prediction_status") if fill else "unfilled_oom",
            "offload_profile": fill.get("offload_profile") if fill else None,
            "fill_prediction_path": fill.get("prediction_path") if fill else None,
            "fill_prediction_sha256": fill.get("prediction_sha256") if fill else None,
            "elapsed_seconds": fill.get("elapsed_seconds") if fill else None,
        })
    atomic_write_csv(
        fill_root / "OOM_FILL_MANIFEST.csv", fill_manifest,
        [
            "method", "task", "sample_index", "sample_id", "raw_status", "raw_error",
            "fill_status", "offload_profile", "fill_prediction_path",
            "fill_prediction_sha256", "elapsed_seconds",
        ],
    )
    _write_jsonl(
        fill_root / "offload_fill_predictions.jsonl",
        (
            {"method_alias": key[0], "task": key[1], "offload_profile": value["offload_profile"],
             **value["prediction"]}
            for key, value in sorted(fills.items())
        ),
    )
    scoring = _score_offload_completed(fill_root=fill_root, sources=sources, fills=fills)
    raw_failures = _verify_raw_hash_lock(raw_lock)
    decision_name = "PASS" if len(fills) == len(oom_rows) and not raw_failures else "SOFT_BLOCKED"
    decision = {
        "created_at": now_iso(), "stage": 11, "name": "oom_offload_fill_v2",
        "decision": decision_name, "run_root": str(fill_root),
        "raw_oom_rows": len(oom_rows), "filled": len(fills),
        "remaining_oom": len(oom_rows) - len(fills),
        "equivalence_gates": {
            profile: {
                "decision": report["decision"], "completed_cases": report["completed_cases"],
                "prediction_matches": report["prediction_matches"],
                "selection_matches": report["selection_matches"],
            }
            for profile, report in gate_reports.items()
        },
        "raw_predictions_modified": bool(raw_failures),
        "raw_hash_failures": raw_failures,
        "scoring": scoring,
        "next_stage": 12,
    }
    atomic_write_json(stage_decision_path(pipeline_dir, 11), decision)
    atomic_write_json(fill_root / "FINAL_OOM_FILL_DECISION.json", decision)
    atomic_write_text(
        pipeline_dir / "STAGE_11_RESUME_COMMAND.txt",
        f"{PYTHON} {Path(__file__).resolve()} --pipeline-dir {pipeline_dir} "
        f"--safety-policy {safety_policy_path} --execute\n",
    )
    atomic_write_json(pipeline_dir / "PID.json", {
        "updated_at": now_iso(), "pipeline_pid": None, "gpu_worker_pid": None,
        "current_stage": None, "current_method": None, "current_task": None,
    })
    return decision


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-dir", type=Path, required=True)
    parser.add_argument("--safety-policy", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    args = parser.parse_args()
    try:
        result = build_and_run(
            args.pipeline_dir.resolve(), args.safety_policy.resolve(),
            execute=args.execute, timeout_seconds=args.timeout_seconds,
        )
    except Exception as exc:
        report = {
            "created_at": now_iso(), "stage": 11, "name": "oom_offload_fill_v2",
            "decision": "HARD_BLOCKED", "reason": "stage_exception",
            "error": repr(exc), "next_stage": None,
        }
        pipeline_dir = args.pipeline_dir.resolve()
        atomic_write_json(pipeline_dir / "STAGE_11_V2_FAILURE.json", report)
        atomic_write_json(stage_decision_path(pipeline_dir, 11), report)
        print(json.dumps(report, indent=2))
        return 2
    print(json.dumps(result, indent=2))
    return 0 if result.get("decision") in {"PASS", "SOFT_BLOCKED", "NOT_RUN"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
