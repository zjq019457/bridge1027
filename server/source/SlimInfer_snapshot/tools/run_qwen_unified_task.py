#!/usr/bin/env python3
"""Run one isolated Qwen LongBench task and publish it atomically."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

_REPO_BOOTSTRAP = Path(__file__).resolve().parents[1]
if str(_REPO_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(_REPO_BOOTSTRAP))

from tools.qwen_pipeline_common import (
    COUNTS,
    DATA_ROOT,
    PROTOCOL_DIR,
    PROTOCOL_ID,
    PYTHON,
    QWEN_MODEL_ROOT,
    REPO_ROOT,
    TASKS,
    atomic_write_csv,
    atomic_write_json,
    canonical_bytes,
    canonical_sample_identity,
    implementation_revision,
    iter_jsonl,
    load_json,
    load_task_samples,
    now_iso,
    per_task_score,
    prediction_status,
    sha256_bytes,
    validate_unique_ids,
)
from tools.qwen_protocol_gate import (
    FROZEN_IMPLEMENTATION_REVISION,
    RUNTIME_FILES,
    run_protocol_gate,
)
from tools.qwen_resource_guard import monitor_resources, write_blocker
from tools.qwen_runtime_watchdog import RuntimeWatchdog


METHOD_SPECS: Dict[str, Dict[str, Any]] = {
    "cc_sliminfer_v1": {
        "runtime_method": "cc_sliminfer_v1",
        "stage": 2,
        "trace": True,
    },
    "full_kv_fa2": {
        "runtime_method": "fullkv",
        "stage": 4,
        "trace": False,
    },
}
NO_CHAT_TASKS = {"trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"}
MAX_MODEL_LEN = 131072
EXPECTED_K = {9: 192, 18: 96, 26: 64}
CONTINUITY_FIELDS = (
    "mode", "triggered", "alpha", "min_cluster_len", "max_neighbors",
    "replacement_cap", "clusters_found", "swaps_applied", "neighbors_proposed",
    "neighbors_added", "neighbors_retained", "neighbor_retention_rate",
    "isolated_replacements", "isolated_replacement_rate", "max_swap_cap",
)


class TaskExecutionError(RuntimeError):
    def __init__(self, kind: str, message: str, details: Optional[Mapping[str, Any]] = None):
        super().__init__(message)
        self.kind = kind
        self.details = dict(details or {})


def _read_jsonl_raw(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                break
    return rows


def _read_trace_events(path: Path) -> Optional[List[Dict[str, Any]]]:
    if not path.exists():
        return None
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    except (OSError, EOFError, json.JSONDecodeError):
        return None


def _recoverable_partial(
    run_root: Path,
    task: str,
    method: str,
    requested_indices: Sequence[int],
) -> Optional[Dict[str, Any]]:
    """Recover a clean prediction prefix and matching closed trace after worker failure."""
    attempts = sorted((run_root / ".incomplete").glob(f"{task}.attempt-*"), reverse=True)
    for attempt_dir in attempts:
        prediction_path = attempt_dir / "run/longbench_predictions" / f"{task}.jsonl"
        rows = _read_jsonl_raw(prediction_path)
        if not rows or len(rows) > len(requested_indices):
            continue
        expected_prefix = list(requested_indices[: len(rows)])
        observed = [int(row.get("sample_index", -1)) for row in rows]
        if observed != expected_prefix:
            continue
        if any(
            row.get("protocol_id") != PROTOCOL_ID
            or row.get("implementation_revision") != FROZEN_IMPLEMENTATION_REVISION
            or row.get("method") != METHOD_SPECS[method]["runtime_method"]
            for row in rows
        ):
            continue
        trace_events: List[Dict[str, Any]] = []
        if METHOD_SPECS[method]["trace"]:
            trace_dir = attempt_dir / "active_block_trace"
            trace_path = trace_dir / f"{task}.jsonl.gz"
            if not trace_path.exists():
                trace_path = trace_dir / f"{task}.jsonl.gz.tmp"
            raw_events = _read_trace_events(trace_path)
            if raw_events is None:
                continue
            completed = set(expected_prefix)
            trace_events = [
                event for event in raw_events
                if int(event.get("sample_index", -1)) in completed
            ]
        return {
            "source_dir": attempt_dir,
            "rows": rows,
            "completed_indices": expected_prefix,
            "trace_events": trace_events,
        }
    return None


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _merge_partial_and_shard(
    *,
    run_root: Path,
    task: str,
    method: str,
    partial: Mapping[str, Any],
    shard_dir: Optional[Path],
) -> Path:
    merge_dir = run_root / ".incomplete" / f"{task}.merged-{os.getpid()}"
    if merge_dir.exists():
        shutil.rmtree(merge_dir)
    source = shard_dir or Path(partial["source_dir"])
    shutil.copytree(source, merge_dir)
    shard_rows = (
        _read_jsonl_raw(shard_dir / "run/longbench_predictions" / f"{task}.jsonl")
        if shard_dir is not None else []
    )
    combined_rows = list(partial["rows"]) + shard_rows
    combined_rows.sort(key=lambda row: int(row["sample_index"]))
    _write_jsonl(
        merge_dir / "run/longbench_predictions" / f"{task}.jsonl",
        combined_rows,
    )
    if METHOD_SPECS[method]["trace"]:
        shard_events: List[Dict[str, Any]] = []
        if shard_dir is not None:
            shard_trace = shard_dir / "active_block_trace" / f"{task}.jsonl.gz"
            shard_events = _read_trace_events(shard_trace) or []
        trace_path = merge_dir / "active_block_trace" / f"{task}.jsonl.gz"
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = trace_path.with_name(trace_path.name + ".merge")
        with gzip.open(tmp, "wt", encoding="utf-8", compresslevel=6) as handle:
            for event in [*partial["trace_events"], *shard_events]:
                handle.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")
        os.replace(tmp, trace_path)
    atomic_write_json(merge_dir / "RESUME_MERGE_PROVENANCE.json", {
        "created_at": now_iso(),
        "source_partial_dir": str(partial["source_dir"]),
        "source_shard_dir": str(shard_dir) if shard_dir is not None else None,
        "skipped_completed_sample_indices": list(partial["completed_indices"]),
        "skipped_completed_sample_ids": [row.get("sample_id") for row in partial["rows"]],
    })
    return merge_dir


def _method_hash(method: str) -> str:
    return str(load_json(PROTOCOL_DIR / "method_configs.json")["methods"][method]["method_config_sha256"])


def _sample_spec(task: str, sample_indices: Optional[Sequence[int]]) -> str:
    if not sample_indices:
        return ""
    return f"{task}:{','.join(str(int(index)) for index in sample_indices)}"


def build_command(
    method: str,
    task: str,
    work_dir: Path,
    revision: str,
    sample_indices: Optional[Sequence[int]] = None,
) -> List[str]:
    spec = METHOD_SPECS[method]
    command = [
        str(PYTHON),
        str(REPO_ROOT / "experiments/run_qwen25_sliminfer_variants.py"),
        "--model_path", str(QWEN_MODEL_ROOT),
        "--tokenizer_path", str(QWEN_MODEL_ROOT),
        "--model_family", "qwen2",
        "--dtype", "fp16",
        "--device", "cuda",
        "--backend", "flash_attention_2",
        "--longbench_dir", str(DATA_ROOT),
        "--longbench_tasks", task,
        "--longbench_max_samples", "0",
        "--max_model_len", str(MAX_MODEL_LEN),
        "--method", spec["runtime_method"],
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
        "--rescue_policy", "none",
        "--rescue_score", "delta_o",
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
        "--implementation_revision", revision,
        "--source_config_hash", _method_hash(method),
        "--active_block_trace", "full" if spec["trace"] else "none",
        "--output_dir", str(work_dir / "run"),
    ]
    selection = _sample_spec(task, sample_indices)
    if selection:
        command.extend(["--longbench_sample_indices_by_task", selection])
    if spec["trace"]:
        command.extend(["--active_block_trace_dir", str(work_dir / "active_block_trace")])
    return command


def build_cc_command(task: str, work_dir: Path, revision: str) -> List[str]:
    """Compatibility alias retained for existing tests and task plans."""
    return build_command("cc_sliminfer_v1", task, work_dir, revision)


def _expected_rows(task: str, indices: Sequence[int]) -> Dict[int, Dict[str, Any]]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_ROOT, trust_remote_code=True)
    templates = load_json(REPO_ROOT / "configs/dataset2prompt.json")
    max_new = load_json(REPO_ROOT / "configs/dataset2maxlen.json")
    samples = load_task_samples(task)
    expected: Dict[int, Dict[str, Any]] = {}
    for index in indices:
        sample = samples[index]
        sample_id, content_hash = canonical_sample_identity(task, index, sample)
        prompt = templates[task].format(**sample)
        raw_ids = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        if int(raw_ids.shape[0]) > MAX_MODEL_LEN:
            half = MAX_MODEL_LEN // 2
            prompt = (
                tokenizer.decode(raw_ids[:half], skip_special_tokens=True)
                + tokenizer.decode(raw_ids[-half:], skip_special_tokens=True)
            )
        if task not in NO_CHAT_TASKS:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True,
            )
        input_ids = [
            int(value) for value in tokenizer(
                prompt, truncation=False, return_tensors="pt"
            ).input_ids[0].tolist()
        ]
        eos: Any = tokenizer.eos_token_id
        if task == "samsum":
            newline_ids = tokenizer.encode("\n", add_special_tokens=False)
            eos = [tokenizer.eos_token_id, newline_ids[-1]] if newline_ids else tokenizer.eos_token_id
        generation = {
            "max_new_tokens": int(max_new[task]),
            "num_beams": 1,
            "do_sample": False,
            "use_cache": True,
            "eos_token_id": eos,
            "temperature_effective": None,
            "top_p_effective": None,
        }
        expected[index] = {
            "sample_id": sample_id,
            "content_sha256": content_hash,
            "prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
            "input_token_ids_sha256": sha256_bytes(
                ",".join(str(value) for value in input_ids).encode("ascii")
            ),
            "input_tokens": len(input_ids),
            "max_new_tokens": int(max_new[task]),
            "generation_config": generation,
            "generation_config_sha256": sha256_bytes(canonical_bytes(generation)),
        }
    del tokenizer
    return expected


def _validate_trace(task: str, path: Path, expected_indices: Sequence[int]) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "path": str(path), "exists": path.exists(), "events": 0,
        "samples_with_events": 0, "field_failures": 0, "continuity_field_failures": 0,
        "budget_failures": 0, "sink_failures": 0, "duplicate_failures": 0,
        "invalid_id_failures": 0, "set_accounting_failures": 0,
        "identity_failures": 0, "event_key_duplicates": 0,
    }
    if not path.exists():
        return report
    seen_keys = set()
    sample_indices = set()
    expected_set = {int(index) for index in expected_indices}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            event = json.loads(line)
            report["events"] += 1
            sample_index = int(event.get("sample_index", -1))
            sample_indices.add(sample_index)
            layer = event.get("abs_layer")
            if layer is None:
                layer = event.get("layer_idx")
            key = (
                event.get("sample_id"), event.get("phase"),
                int(event.get("decode_step", 0)),
                int(event.get("selection_event_index", -1)), layer,
            )
            if key in seen_keys:
                report["event_key_duplicates"] += 1
            seen_keys.add(key)
            required = (
                "base_original_active_ids", "cc_active_ids", "added_ids", "removed_ids",
                "selected_count", "sink_ids", "jaccard_vs_silm", "selection_time_ms",
                "selection_event_index", "decode_step", "phase", "layer_idx",
                "candidate_ids", "budget_parity", "sink_violation",
            )
            if any(field not in event for field in required):
                report["field_failures"] += 1
            if any(field not in event for field in CONTINUITY_FIELDS):
                report["continuity_field_failures"] += 1
            if (
                event.get("protocol_id") != PROTOCOL_ID
                or event.get("implementation_revision") != FROZEN_IMPLEMENTATION_REVISION
                or event.get("task") != task
                or sample_index not in expected_set
            ):
                report["identity_failures"] += 1
            selected = list(event.get("cc_active_ids", []))
            baseline = list(event.get("base_original_active_ids", []))
            candidates = set(event.get("candidate_ids", []))
            added = set(event.get("added_ids", []))
            removed = set(event.get("removed_ids", []))
            selected_set = set(selected)
            baseline_set = set(baseline)
            # Duplicate validity is derived from the complete active-ID list. The
            # frozen Stage 1 trace schema does not emit a duplicate_violation flag.
            if len(selected) != len(selected_set) or int(event.get("duplicate_violation", 0)):
                report["duplicate_failures"] += 1
            invalid = [
                value for value in selected
                if not isinstance(value, int) or value < 0 or value not in candidates
            ]
            if invalid:
                report["invalid_id_failures"] += 1
            if added != selected_set - baseline_set or removed != baseline_set - selected_set:
                report["set_accounting_failures"] += 1
            if len(added) != len(removed):
                report["set_accounting_failures"] += 1
            expected_k = EXPECTED_K.get(int(layer)) if layer is not None else None
            if (
                int(event.get("budget_parity", 0)) != 1
                or int(event.get("over_budget", 0)) != 0
                or int(event.get("selected_count", -1)) != len(selected)
                or (expected_k is not None and len(selected) != expected_k)
            ):
                report["budget_failures"] += 1
            sink_ids = set(event.get("sink_ids", []))
            if int(event.get("sink_violation", 0)) or not sink_ids.issubset(selected_set):
                report["sink_failures"] += 1
    report["samples_with_events"] = len(sample_indices & expected_set)
    report["sample_coverage"] = len(sample_indices & expected_set) / max(1, len(expected_set))
    return report


def validate_task(
    task: str,
    work_dir: Path,
    *,
    method: str = "cc_sliminfer_v1",
    sample_indices: Optional[Sequence[int]] = None,
    verify_tokenization: bool = True,
) -> Dict[str, Any]:
    spec = METHOD_SPECS[method]
    runtime_method = str(spec["runtime_method"])
    indices = list(sample_indices) if sample_indices is not None else list(range(COUNTS[task]))
    prediction_path = work_dir / "run/longbench_predictions" / f"{task}.jsonl"
    trace_path = work_dir / "active_block_trace" / f"{task}.jsonl.gz"
    predictions = list(iter_jsonl(prediction_path)) if prediction_path.exists() else []
    expected_inputs = _expected_rows(task, indices) if verify_tokenization and predictions else {}
    identity_failures: List[Dict[str, Any]] = []
    protocol_field_failures: List[Dict[str, Any]] = []
    statuses: Dict[str, int] = {}
    method_hash = _method_hash(method)
    for position, row in enumerate(predictions):
        status = prediction_status(row)
        statuses[status] = statuses.get(status, 0) + 1
        if position >= len(indices):
            identity_failures.append({"position": position, "reason": "out_of_range"})
            continue
        sample_index = int(indices[position])
        expected = expected_inputs.get(sample_index)
        if expected is None:
            samples = load_task_samples(task)
            sample_id, content_hash = canonical_sample_identity(task, sample_index, samples[sample_index])
            expected = {"sample_id": sample_id, "content_sha256": content_hash}
        for field, value in (
            ("sample_index", sample_index),
            ("sample_id", expected["sample_id"]),
            ("content_sha256", expected["content_sha256"]),
            ("protocol_id", PROTOCOL_ID),
            ("method", runtime_method),
        ):
            if row.get(field) != value:
                identity_failures.append({
                    "position": position, "field": field,
                    "expected": value, "actual": row.get(field),
                })
        expected_protocol = {
            "implementation_revision": FROZEN_IMPLEMENTATION_REVISION,
            "source_config_hash": method_hash,
            "runtime_dtype": "torch.float16",
            "attention_implementation": "flash_attention_2",
            "tokenizer_id": str(QWEN_MODEL_ROOT),
        }
        if verify_tokenization:
            expected_protocol.update({
                field: expected[field] for field in (
                    "prompt_sha256", "input_token_ids_sha256", "input_tokens",
                    "max_new_tokens", "generation_config", "generation_config_sha256",
                )
            })
        for field, value in expected_protocol.items():
            if row.get(field) != value:
                protocol_field_failures.append({
                    "position": position, "field": field,
                    "expected": value, "actual": row.get(field),
                })
    uniqueness = validate_unique_ids(predictions, task)
    trace = (
        _validate_trace(task, trace_path, indices)
        if spec["trace"] else {"path": str(trace_path), "required": False, "events": 0}
    )
    trace_failures = sum(
        int(trace.get(field, 0)) for field in (
            "field_failures", "continuity_field_failures", "budget_failures",
            "sink_failures", "duplicate_failures", "invalid_id_failures",
            "set_accounting_failures", "identity_failures", "event_key_duplicates",
        )
    )
    invalid_statuses = {
        status: count for status, count in statuses.items()
        if status not in {"ok", "oom"}
    }
    report: Dict[str, Any] = {
        "task": task,
        "method": method,
        "runtime_method": runtime_method,
        "created_at": now_iso(),
        "expected_predictions": len(indices),
        "predictions": len(predictions),
        "sample_indices": indices,
        "statuses": statuses,
        "invalid_statuses": invalid_statuses,
        "unique_ids": uniqueness["pass"],
        "identity_failures": identity_failures,
        "protocol_field_failures": protocol_field_failures,
        "trace": trace,
        "prediction_path": str(prediction_path),
    }
    complete = (
        len(predictions) == len(indices)
        and uniqueness["pass"]
        and not identity_failures
        and not protocol_field_failures
        and not invalid_statuses
        and (not spec["trace"] or (trace_path.exists() and trace_failures == 0))
    )
    report["score"] = per_task_score(task, predictions) if complete else None
    report["decision"] = "PASS" if complete else "FAIL"
    return report


def _clear_pid(pipeline_dir: Path) -> None:
    pid_path = pipeline_dir / "PID.json"
    state = load_json(pid_path) if pid_path.exists() else {}
    state.update({
        "updated_at": now_iso(), "task_worker_pid": None, "gpu_worker_pid": None,
        "current_task": None, "current_method": None,
    })
    atomic_write_json(pid_path, state)


def run_task(
    pipeline_dir: Path,
    run_root: Path,
    safety_policy_path: Path,
    task: str,
    *,
    method: str = "cc_sliminfer_v1",
    stage: Optional[int] = None,
    sample_indices: Optional[Sequence[int]] = None,
    execute: bool,
) -> Dict[str, Any]:
    if task not in TASKS:
        raise ValueError(f"Unknown task: {task}")
    if method not in METHOD_SPECS:
        raise ValueError(f"Unsupported unified task method: {method}")
    stage = int(stage if stage is not None else METHOD_SPECS[method]["stage"])
    gate = run_protocol_gate(pipeline_dir, stage)
    if gate["decision"] != "PASS":
        raise TaskExecutionError("protocol", "Protocol gate failed", gate)
    revision = implementation_revision(RUNTIME_FILES)
    if revision != FROZEN_IMPLEMENTATION_REVISION:
        raise TaskExecutionError(
            "protocol", "Frozen implementation revision changed",
            {"expected": FROZEN_IMPLEMENTATION_REVISION, "actual": revision},
        )
    final_dir = run_root / "tasks" / task
    if final_dir.exists():
        report_path = final_dir / "TASK_DECISION.json"
        report = load_json(report_path) if report_path.exists() else validate_task(
            task, final_dir, method=method, sample_indices=sample_indices,
        )
        if report.get("decision") == "PASS":
            return {**report, "resumed": True}
        raise TaskExecutionError(
            "audit", f"Existing task directory is not a verified PASS: {final_dir}", report,
        )
    requested_indices = (
        [int(value) for value in sample_indices]
        if sample_indices is not None else list(range(COUNTS[task]))
    )
    partial = _recoverable_partial(run_root, task, method, requested_indices) if execute else None
    completed_indices = list(partial["completed_indices"]) if partial else []
    effective_indices = [value for value in requested_indices if value not in set(completed_indices)]
    attempt = 1 + len(list((run_root / ".incomplete").glob(f"{task}.attempt-*")))
    work_dir = run_root / ".incomplete" / f"{task}.attempt-{attempt:02d}"
    command_indices: Optional[Sequence[int]] = (
        effective_indices if partial or sample_indices is not None else None
    )
    command = build_command(method, task, work_dir, revision, command_indices)
    plan = {
        "created_at": now_iso(), "stage": stage, "task": task, "method": method,
        "runtime_method": METHOD_SPECS[method]["runtime_method"],
        "protocol_id": PROTOCOL_ID, "implementation_revision": revision,
        "source_config_hash": _method_hash(method),
        "requested_sample_indices": sample_indices,
        "effective_sample_indices": effective_indices,
        "resume_skipped_sample_indices": completed_indices,
        "resume_source": str(partial["source_dir"]) if partial else None,
        "attempt": attempt, "command": command, "final_dir": str(final_dir),
    }
    atomic_write_json(run_root / "commands" / f"{task}.attempt-{attempt:02d}.json", plan)
    if not execute:
        return {"decision": "NOT_RUN", **plan}
    if partial and not effective_indices:
        publish_dir = _merge_partial_and_shard(
            run_root=run_root, task=task, method=method, partial=partial, shard_dir=None,
        )
        report = validate_task(
            task, publish_dir, method=method, sample_indices=requested_indices,
            verify_tokenization=True,
        )
        report.update({
            "attempt": attempt, "resumed": True,
            "resume_skipped_sample_indices": completed_indices,
            "resume_skipped_sample_ids": [row.get("sample_id") for row in partial["rows"]],
            "log": None,
        })
        if report["decision"] != "PASS":
            raise TaskExecutionError("audit", f"Merged task {method}/{task} failed validation", report)
        work_dir = publish_dir
    else:
        report = None
    policy = load_json(safety_policy_path)
    if report is not None:
        resource_gate = None
    else:
        resource_gate = monitor_resources(
            policy,
            int(policy["thresholds"]["resource_gate_duration_seconds"]),
            int(policy["thresholds"]["resource_gate_interval_seconds"]),
            pipeline_dir=pipeline_dir,
            allowed_pids=(os.getpid(),),
        )
        atomic_write_json(run_root / "resource_gates" / f"{task}.attempt-{attempt:02d}.json", resource_gate)
        if not resource_gate["pass"]:
            write_blocker(pipeline_dir, resource_gate)
            raise TaskExecutionError("safety", f"Resource gate blocked {method}/{task}", resource_gate)
    if report is None:
        work_dir.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.update({key: str(value) for key, value in policy["environment"].items() if key.isupper()})
    log_path = run_root / "logs" / f"{task}.attempt-{attempt:02d}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    resume_command = (pipeline_dir / "RESUME_COMMAND.txt").read_text(encoding="utf-8").strip()
    watchdog: Optional[RuntimeWatchdog] = None
    returncode = -1
    if report is None:
        try:
            with log_path.open("a", encoding="utf-8") as log:
                proc = subprocess.Popen(
                    command, cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                )
                pid_path = pipeline_dir / "PID.json"
                pid_state = load_json(pid_path) if pid_path.exists() else {}
                pid_state.update({
                    "updated_at": now_iso(), "task_worker_pid": os.getpid(),
                    "gpu_worker_pid": proc.pid, "current_task": task,
                    "current_method": method, "current_stage": stage,
                })
                atomic_write_json(pid_path, pid_state)
                prediction_path = work_dir / "run/longbench_predictions" / f"{task}.jsonl"
                trace_path = work_dir / "active_block_trace" / f"{task}.jsonl.gz.tmp"
                watchdog = RuntimeWatchdog(
                    pipeline_dir=pipeline_dir, run_root=run_root, policy=policy,
                    stage=stage, method=method, task=task, worker_pid=proc.pid,
                    prediction_path=prediction_path,
                    trace_path=trace_path if METHOD_SPECS[method]["trace"] else None,
                    log_path=log_path, resume_command=resume_command,
                )
                watchdog.start()
                returncode = proc.wait()
        finally:
            watchdog_failure = watchdog.stop() if watchdog is not None else None
            _clear_pid(pipeline_dir)
        if watchdog_failure:
            raise TaskExecutionError("safety", "Runtime watchdog stopped the GPU worker", watchdog_failure)
        if returncode != 0:
            failure = {
                "task": task, "method": method, "decision": "FAILED",
                "returncode": returncode, "created_at": now_iso(),
                "log": str(log_path), "attempt": attempt,
                "completed_predictions": len(_read_jsonl_raw(
                    work_dir / "run/longbench_predictions" / f"{task}.jsonl"
                )),
            }
            atomic_write_json(work_dir / "TASK_DECISION.json", failure)
            raise TaskExecutionError("recoverable", f"Task worker exited {returncode}", failure)
        shard_report = validate_task(
            task, work_dir, method=method, sample_indices=effective_indices,
            verify_tokenization=True,
        )
        shard_report.update({"attempt": attempt, "log": str(log_path), "shard": bool(partial)})
        atomic_write_json(work_dir / "TASK_DECISION.json", shard_report)
        if shard_report["decision"] != "PASS":
            raise TaskExecutionError("audit", f"Task {method}/{task} failed atomic validation", shard_report)
        if partial:
            publish_dir = _merge_partial_and_shard(
                run_root=run_root, task=task, method=method,
                partial=partial, shard_dir=work_dir,
            )
            report = validate_task(
                task, publish_dir, method=method, sample_indices=requested_indices,
                verify_tokenization=True,
            )
            report.update({
                "attempt": attempt, "log": str(log_path), "resumed": True,
                "resume_skipped_sample_indices": completed_indices,
                "resume_skipped_sample_ids": [row.get("sample_id") for row in partial["rows"]],
            })
            atomic_write_json(publish_dir / "TASK_DECISION.json", report)
            if report["decision"] != "PASS":
                raise TaskExecutionError("audit", f"Merged task {method}/{task} failed validation", report)
            work_dir = publish_dir
        else:
            report = shard_report
    oom_rows = []
    prediction_path = Path(report["prediction_path"])
    for row in iter_jsonl(prediction_path):
        if prediction_status(row) == "oom":
            oom_rows.append({
                "method": method, "task": task, "sample_index": row.get("sample_index"),
                "sample_id": row.get("sample_id"), "status": "oom", "error": row.get("error"),
            })
    atomic_write_csv(
        work_dir / "OOM_MANIFEST.csv", oom_rows,
        ["method", "task", "sample_index", "sample_id", "status", "error"],
    )
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    os.replace(work_dir, final_dir)
    report["prediction_path"] = str(
        final_dir / "run/longbench_predictions" / f"{task}.jsonl"
    )
    if METHOD_SPECS[method]["trace"]:
        report["trace"]["path"] = str(
            final_dir / "active_block_trace" / f"{task}.jsonl.gz"
        )
    atomic_write_json(final_dir / "TASK_DECISION.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-dir", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--safety-policy", type=Path, required=True)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--method", choices=tuple(METHOD_SPECS), default="cc_sliminfer_v1")
    parser.add_argument("--stage", type=int)
    parser.add_argument("--sample-indices", default="")
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sample_indices = (
        [int(value) for value in args.sample_indices.split(",") if value.strip()]
        if args.sample_indices else None
    )
    try:
        report = run_task(
            args.pipeline_dir.resolve(), args.run_root.resolve(), args.safety_policy.resolve(),
            args.task, method=args.method, stage=args.stage,
            sample_indices=sample_indices, execute=args.execute,
        )
    except TaskExecutionError as exc:
        report = {
            "decision": "FAILED", "failure_kind": exc.kind,
            "error": str(exc), "details": exc.details,
            "traceback": traceback.format_exc(), "created_at": now_iso(),
        }
        failure_path = args.run_root / "failures" / f"{args.task}.{os.getpid()}.json"
        atomic_write_json(failure_path, report)
        print(json.dumps(report, indent=2))
        return {"protocol": 20, "safety": 21, "audit": 22, "recoverable": 30}.get(exc.kind, 31)
    print(json.dumps(report, indent=2))
    return 0 if report.get("decision") in {"PASS", "NOT_RUN"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
