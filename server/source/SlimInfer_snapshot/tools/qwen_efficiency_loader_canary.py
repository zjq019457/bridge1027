#!/usr/bin/env python3
"""Prove the direct-GPU efficiency loader matches the frozen Stage-1 run."""
from __future__ import annotations

import argparse
import copy
import gzip
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

_REPO_BOOTSTRAP = Path(__file__).resolve().parents[1]
if str(_REPO_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(_REPO_BOOTSTRAP))

from tools.qwen_pipeline_common import (
    REPO_ROOT,
    atomic_write_json,
    iter_jsonl,
    load_json,
    now_iso,
    sha256_file,
)
from tools.qwen_protocol_gate import run_protocol_gate
from tools.qwen_resource_guard import monitor_resources
from tools.qwen_runtime_watchdog import RuntimeWatchdog


TASKS = ("qasper", "hotpotqa", "2wikimqa", "musique", "gov_report", "repobench-p")
PREDICTION_FIELDS = (
    "sample_id", "content_sha256", "prompt_sha256", "input_token_ids_sha256",
    "status", "pred", "input_tokens", "output_tokens", "source_config_hash",
    "max_new_tokens", "generation_config", "generation_config_sha256",
    "runtime_dtype", "attention_implementation", "tf32_matmul_allowed",
    "tf32_cudnn_allowed", "tokenizer_id",
)


def _set_arg(command: list[str], flag: str, value: str) -> None:
    command[command.index(flag) + 1] = value


def _event_key(row: Mapping[str, Any]) -> Tuple[Any, ...]:
    layer = row.get("abs_layer")
    if layer is None:
        layer = row.get("layer_idx", -1)
    return (
        str(row.get("task")), int(row.get("sample_index", -1)), str(row.get("phase")),
        int(row.get("decode_step", 0)), int(row.get("selection_event_index", -1)),
        int(layer),
    )


def _gzip_rows(path: Path) -> Iterable[Dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _prediction_map(root: Path) -> Dict[Tuple[str, int], Dict[str, Any]]:
    result: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for task in TASKS:
        for row in iter_jsonl(root / "longbench_predictions" / f"{task}.jsonl"):
            key = (task, int(row["sample_index"]))
            if key in result:
                raise RuntimeError(f"Duplicate prediction: {key}")
            result[key] = row
    return result


def _event_map(root: Path, subdir: str) -> Dict[Tuple[Any, ...], Dict[str, Any]]:
    result: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for task in TASKS:
        for row in _gzip_rows(root / subdir / f"{task}.jsonl.gz"):
            key = _event_key(row)
            if key in result:
                raise RuntimeError(f"Duplicate {subdir} event: {key}")
            result[key] = row
    return result


def _compare(standard: Path, direct: Path) -> Dict[str, Any]:
    old_predictions = _prediction_map(standard)
    new_predictions = _prediction_map(direct)
    prediction_mismatches = []
    for key in sorted(set(old_predictions) | set(new_predictions)):
        old = old_predictions.get(key)
        new = new_predictions.get(key)
        fields = [
            field for field in PREDICTION_FIELDS
            if old is None or new is None or old.get(field) != new.get(field)
        ]
        if fields:
            prediction_mismatches.append({"sample": list(key), "fields": fields})

    old_digest = _event_map(standard, "selection_digests")
    new_digest = _event_map(direct, "selection_digests")
    digest_mismatches = [
        list(key) for key in sorted(set(old_digest) | set(new_digest))
        if old_digest.get(key) != new_digest.get(key)
    ]
    old_trace = _event_map(standard, "active_block_trace")
    new_trace = _event_map(direct, "active_block_trace")
    trace_fields: Sequence[str] = (
        "selected_block_ids", "base_original_active_ids", "selected_count",
        "sink_ids", "budget_parity", "duplicate_violation", "invalid_id_violation",
    )
    trace_mismatches = []
    for key in sorted(set(old_trace) | set(new_trace)):
        old = old_trace.get(key)
        new = new_trace.get(key)
        fields = [
            field for field in trace_fields
            if old is None or new is None or old.get(field) != new.get(field)
        ]
        if fields:
            trace_mismatches.append({"event": list(key), "fields": fields})
    direct_trace_failures = [
        list(key) for key, row in new_trace.items()
        if int(row.get("budget_parity", 0)) != 1
        or int(row.get("sink_violation", 0)) != 0
        or int(row.get("duplicate_violation", 0)) != 0
        or int(row.get("invalid_id_violation", 0)) != 0
    ]
    checks = {
        "prediction_keys_12": set(old_predictions) == set(new_predictions) and len(new_predictions) == 12,
        "prediction_fields_exact": not prediction_mismatches,
        "selection_digest_keys_exact": set(old_digest) == set(new_digest),
        "selection_digest_rows_exact": not digest_mismatches,
        "active_trace_keys_exact": set(old_trace) == set(new_trace),
        "active_ids_and_counts_exact": not trace_mismatches,
        "direct_budget_sink_duplicate_invalid_zero": not direct_trace_failures,
    }
    return {
        "checks": checks,
        "prediction_count": len(new_predictions),
        "selection_digest_events": len(new_digest),
        "active_trace_events": len(new_trace),
        "prediction_mismatches": prediction_mismatches,
        "digest_mismatches": digest_mismatches,
        "trace_mismatches": trace_mismatches,
        "direct_trace_failures": direct_trace_failures,
    }


def run(pipeline_dir: Path) -> Dict[str, Any]:
    gate = run_protocol_gate(pipeline_dir, 8)
    if gate["decision"] != "PASS":
        raise RuntimeError("Stage 8 protocol gate failed")
    policy = load_json(pipeline_dir / "SAFETY_POLICY.json")
    resource_gate = monitor_resources(
        policy,
        int(policy["thresholds"]["resource_gate_duration_seconds"]),
        int(policy["thresholds"]["resource_gate_interval_seconds"]),
        pipeline_dir=pipeline_dir,
        allowed_pids=(os.getpid(),),
    )
    if not resource_gate["pass"]:
        raise RuntimeError(f"Resource gate failed: {resource_gate['checks'][-1]['failures']}")

    root = REPO_ROOT / "results" / f"qwen_efficiency_loader_canary_{datetime.now():%Y%m%d_%H%M%S}"
    direct = root / "direct_trace_full"
    direct.mkdir(parents=True)
    commands = load_json(pipeline_dir / "stage_01_canary/COMMANDS.json")
    command = copy.deepcopy(commands["trace_full"])
    command[1] = str(REPO_ROOT / "tools/qwen_efficiency_direct_gpu_loader.py")
    _set_arg(command, "--selection_digest_dir", str(direct / "selection_digests"))
    _set_arg(command, "--output_dir", str(direct))
    _set_arg(command, "--active_block_trace_dir", str(direct / "active_block_trace"))
    atomic_write_json(root / "COMMAND.json", {
        "command": command,
        "loader_sha256": sha256_file(REPO_ROOT / "tools/qwen_efficiency_direct_gpu_loader.py"),
    })
    atomic_write_json(root / "RESOURCE_GATE.json", resource_gate)

    log_path = root / "worker.log"
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(command, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT)
        watchdog = RuntimeWatchdog(
            pipeline_dir=pipeline_dir, run_root=root, policy=policy, stage=8,
            method="sliminfer_original", task="direct_loader_12_canary",
            worker_pid=proc.pid, prediction_path=None, trace_path=None,
            log_path=log_path,
            resume_command="rerun tools/qwen_efficiency_loader_canary.py",
        )
        watchdog.start()
        returncode = proc.wait()
        watchdog_failure = watchdog.stop()

    comparison: Dict[str, Any] = {}
    if returncode == 0 and watchdog_failure is None:
        comparison = _compare(pipeline_dir / "stage_01_canary/trace_full", direct)
    checks = comparison.get("checks", {})
    decision = "PASS" if returncode == 0 and watchdog_failure is None and checks and all(checks.values()) else "FAIL"
    result = {
        "created_at": now_iso(), "decision": decision,
        "loader": "direct-gpu-low-cpu-memory-v1",
        "loader_sha256": sha256_file(REPO_ROOT / "tools/qwen_efficiency_direct_gpu_loader.py"),
        "returncode": returncode, "watchdog_failure": watchdog_failure,
        "standard_root": str(pipeline_dir / "stage_01_canary/trace_full"),
        "direct_root": str(direct), "root": str(root),
        **comparison,
    }
    atomic_write_json(root / "LOADER_EQUIVALENCE.json", result)
    atomic_write_json(pipeline_dir / "STAGE_08_LOADER_EQUIVALENCE.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.pipeline_dir.resolve())
    print(json.dumps(result, indent=2))
    return 0 if result["decision"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
