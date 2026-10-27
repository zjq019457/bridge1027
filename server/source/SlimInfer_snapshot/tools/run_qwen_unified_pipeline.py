#!/usr/bin/env python3
"""Sequential, resumable state machine for the Qwen unified LongBench suite."""
from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import tempfile
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

_REPO_BOOTSTRAP = Path(__file__).resolve().parents[1]
if str(_REPO_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(_REPO_BOOTSTRAP))

from tools.qwen_equivalence_canary import CANARY_INDICES, run_canary
from tools.qwen_pipeline_common import (
    COUNTS,
    PROTOCOL_ID,
    PYTHON,
    QWEN_DRM_ROOT,
    REPO_ROOT,
    STAGES,
    TASKS,
    PipelineLock,
    atomic_write_csv,
    atomic_write_json,
    atomic_write_text,
    iter_jsonl,
    load_json,
    now_iso,
    per_task_score,
    prediction_status,
    read_stage_decision,
    stage_decision_path,
    task_macro,
)
from tools.qwen_protocol_gate import run_protocol_gate
from tools.qwen_stage_preflight import run_preflight


DEFAULT_SAFETY_POLICY = REPO_ROOT / "configs/qwen_machine_safety_policy.json"
DEFAULT_AUDIT_DIR = REPO_ROOT / "results/unified_protocol_v1_audit_20260720_202245"
ACCEPTED_PREVIOUS = {"PASS", "SOFT_BLOCKED"}
HARD_EXIT_CODES = {20: "protocol", 21: "safety", 22: "audit"}


def new_pipeline_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return REPO_ROOT / "results" / f"qwen_unified_pipeline_{stamp}"


def _command_text(args: Sequence[str]) -> str:
    return shlex.join([str(value) for value in args])


def initialize_pipeline(pipeline_dir: Path, safety_policy: Path) -> None:
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    for name in ("logs", "commands", "checkpoints"):
        (pipeline_dir / name).mkdir(exist_ok=True)
    orchestrator_log = pipeline_dir / "logs/orchestrator.log"
    if not orchestrator_log.exists():
        atomic_write_text(orchestrator_log, f"{now_iso()} pipeline initialized\n")
    if not (pipeline_dir / "PIPELINE_MANIFEST.json").exists():
        import subprocess as _subprocess

        commit = _subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
        manifest = {
            "created_at": now_iso(), "pipeline_dir": str(pipeline_dir),
            "repo_root": str(REPO_ROOT), "git_commit": commit,
            "protocol_id": PROTOCOL_ID, "canonical_original_root": str(QWEN_DRM_ROOT),
            "machine_role": "qwen_experiment_machine", "llama_input_required": False,
            "stage_order": list(STAGES),
        }
        atomic_write_json(pipeline_dir / "PIPELINE_MANIFEST.json", manifest)
    if not (pipeline_dir / "PIPELINE_STATE.json").exists():
        atomic_write_json(pipeline_dir / "PIPELINE_STATE.json", {
            "updated_at": now_iso(), "state": "NOT_STARTED", "current_stage": None,
            "last_passed_stage": None, "next_stage": 0,
        })
    if not (pipeline_dir / "PID.json").exists():
        atomic_write_json(pipeline_dir / "PID.json", {
            "updated_at": now_iso(), "orchestrator_pid": None, "task_worker_pid": None,
            "gpu_worker_pid": None, "current_method": None, "current_task": None,
        })
    atomic_write_csv(pipeline_dir / "STAGE_REGISTRY.csv", [
        {
            "stage": row["number"], "name": row["name"], "gpu": row["gpu"],
            "decision_path": stage_decision_path(pipeline_dir, row["number"]),
        }
        for row in STAGES
    ], ["stage", "name", "gpu", "decision_path"])
    if not (pipeline_dir / "RUN_REGISTRY.csv").exists():
        atomic_write_csv(pipeline_dir / "RUN_REGISTRY.csv", [], [
            "stage", "method", "task", "status", "path", "updated_at",
        ])
    if not safety_policy.exists():
        raise RuntimeError(f"Safety policy does not exist: {safety_policy}")
    atomic_write_json(pipeline_dir / "SAFETY_POLICY.json", load_json(safety_policy))
    base = [str(PYTHON), str(REPO_ROOT / "tools/run_qwen_unified_pipeline.py")]
    common = ["--pipeline-dir", str(pipeline_dir), "--safety-policy", str(safety_policy)]
    launch = ["setsid", "-f", *base, *common, "--resume", "--auto-advance", "--start-stage", "2"]
    atomic_write_text(
        pipeline_dir / "LAUNCH_COMMAND.txt",
        _command_text(launch) + f" > {orchestrator_log} 2>&1 < /dev/null &\n",
    )
    atomic_write_text(
        pipeline_dir / "RESUME_COMMAND.txt",
        _command_text([*base, *common, "--resume", "--auto-advance"]) + "\n",
    )
    atomic_write_text(
        pipeline_dir / "STATUS_COMMAND.txt",
        _command_text([*base, *common, "--status"]) + "\n",
    )
    atomic_write_text(pipeline_dir / "STOP_COMMAND.txt", f"touch {pipeline_dir / 'STOP_REQUESTED'}\n")


def update_state(pipeline_dir: Path, **updates: Any) -> Dict[str, Any]:
    path = pipeline_dir / "PIPELINE_STATE.json"
    state = load_json(path) if path.exists() else {}
    state.update(updates)
    state["updated_at"] = now_iso()
    atomic_write_json(path, state)
    return state


def _stage_row(stage: int) -> Mapping[str, Any]:
    return next(row for row in STAGES if row["number"] == stage)


def _write_blocker(
    pipeline_dir: Path,
    stage: int,
    kind: str,
    reason: str,
    details: Optional[Mapping[str, Any]] = None,
    attempted: Optional[Sequence[str]] = None,
    resume_command: Optional[str] = None,
) -> Dict[str, Any]:
    if kind not in {"HARD_BLOCKED", "SOFT_BLOCKED"}:
        raise ValueError(kind)
    row = _stage_row(stage)
    decision = {
        "stage": stage, "name": row["name"], "created_at": now_iso(),
        "decision": kind, "reason": reason, "details": dict(details or {}),
        "attempted_steps": list(attempted or []),
        "resume_command": resume_command or (pipeline_dir / "RESUME_COMMAND.txt").read_text(encoding="utf-8").strip(),
        "next_stage": stage + 1 if kind == "SOFT_BLOCKED" and stage < 12 else None,
    }
    atomic_write_json(stage_decision_path(pipeline_dir, stage), decision)
    label = "Soft Blocker" if kind == "SOFT_BLOCKED" else "Hard Blocker"
    lines = [
        f"# Stage {stage:02d} {label}", "", f"- Reason: `{reason}`",
        f"- Protocol: `{PROTOCOL_ID}`", f"- Created: `{decision['created_at']}`", "",
        "## Attempted Steps", "",
    ]
    lines.extend(f"- {value}" for value in decision["attempted_steps"] or ["No unsafe workaround was attempted."])
    lines.extend([
        "", "## Official Path Status", "",
        f"```json\n{json.dumps(decision['details'], indent=2, ensure_ascii=False)}\n```",
        "", "## Resume", "", f"`{decision['resume_command']}`", "",
        "## Downstream Impact", "",
        "Independent later stages continue." if kind == "SOFT_BLOCKED" else "The pipeline stops before later stages.",
    ])
    atomic_write_text(
        pipeline_dir / f"STAGE_{stage:02d}_{'SOFT' if kind == 'SOFT_BLOCKED' else 'HARD'}_BLOCKER.md",
        "\n".join(lines) + "\n",
    )
    return decision


def _append_registry(pipeline_dir: Path, row: Mapping[str, Any]) -> None:
    path = pipeline_dir / "RUN_REGISTRY.csv"
    fields = ["stage", "method", "task", "status", "path", "updated_at"]
    rows: List[Dict[str, Any]] = []
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows.extend(csv.DictReader(handle))
    key = (str(row.get("stage")), str(row.get("method")), str(row.get("task")))
    rows = [
        existing for existing in rows
        if (str(existing.get("stage")), str(existing.get("method")), str(existing.get("task"))) != key
    ]
    rows.append({**row, "updated_at": now_iso()})
    atomic_write_csv(path, rows, fields)


def _run_task_wrapper(
    pipeline_dir: Path,
    run_root: Path,
    safety_policy: Path,
    *,
    stage: int,
    method: str,
    task: str,
    sample_indices: Optional[Sequence[int]] = None,
    dry_run: bool,
) -> Tuple[str, Dict[str, Any]]:
    command = [
        str(PYTHON), str(REPO_ROOT / "tools/run_qwen_unified_task.py"),
        "--pipeline-dir", str(pipeline_dir), "--run-root", str(run_root),
        "--safety-policy", str(safety_policy), "--stage", str(stage),
        "--method", method, "--task", task,
    ]
    if sample_indices:
        command.extend(["--sample-indices", ",".join(str(value) for value in sample_indices)])
    if not dry_run:
        command.append("--execute")
    if dry_run:
        return "NOT_RUN", {"command": command}
    max_attempts = 2
    for wrapper_attempt in range(1, max_attempts + 1):
        log_path = pipeline_dir / "logs" / f"stage_{stage:02d}_{method}_{task}.wrapper.log"
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n{now_iso()} wrapper attempt {wrapper_attempt}: {_command_text(command)}\n")
            log.flush()
            proc = subprocess.Popen(command, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT)
            pid = load_json(pipeline_dir / "PID.json")
            pid.update({
                "updated_at": now_iso(), "task_worker_pid": proc.pid,
                "current_stage": stage, "current_method": method, "current_task": task,
            })
            atomic_write_json(pipeline_dir / "PID.json", pid)
            returncode = proc.wait()
        if returncode == 0:
            decision_path = run_root / "tasks" / task / "TASK_DECISION.json"
            return "PASS", load_json(decision_path)
        if returncode in HARD_EXIT_CODES:
            return "HARD_BLOCKED", {
                "failure_kind": HARD_EXIT_CODES[returncode], "returncode": returncode,
                "log": str(log_path), "wrapper_attempt": wrapper_attempt,
            }
        if returncode == 30 and wrapper_attempt < max_attempts:
            continue
        return "FAILED", {
            "failure_kind": "worker_failure", "returncode": returncode,
            "log": str(log_path), "wrapper_attempt": wrapper_attempt,
        }
    raise AssertionError("unreachable")


def _atomic_concatenate_jsonl(paths: Sequence[Path], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    try:
        with os.fdopen(fd, "wb") as target:
            for path in paths:
                with path.open("rb") as source:
                    while chunk := source.read(8 * 1024 * 1024):
                        target.write(chunk)
            target.flush()
            os.fsync(target.fileno())
        os.replace(tmp_name, destination)
    finally:
        if Path(tmp_name).exists():
            Path(tmp_name).unlink()


def _finalize_quality_run(run_root: Path, method: str, *, trace_required: bool) -> Dict[str, Any]:
    prediction_paths: List[Path] = []
    task_rows: List[Dict[str, Any]] = []
    oom_rows: List[Dict[str, Any]] = []
    trace_rows: List[Dict[str, Any]] = []
    task_scores: Dict[str, float] = {}
    total = 0
    for task in TASKS:
        task_dir = run_root / "tasks" / task
        report = load_json(task_dir / "TASK_DECISION.json")
        if report.get("decision") != "PASS":
            raise RuntimeError(f"{method}/{task} is not PASS")
        path = task_dir / "run/longbench_predictions" / f"{task}.jsonl"
        rows = list(iter_jsonl(path))
        if len(rows) != COUNTS[task]:
            raise RuntimeError(f"{method}/{task}: expected {COUNTS[task]}, got {len(rows)}")
        prediction_paths.append(path)
        total += len(rows)
        score = per_task_score(task, rows)
        task_scores[task] = score
        counts: Dict[str, int] = {}
        for row in rows:
            status = prediction_status(row)
            counts[status] = counts.get(status, 0) + 1
            if status != "ok":
                oom_rows.append({
                    "method": method, "task": task, "sample_index": row.get("sample_index"),
                    "sample_id": row.get("sample_id"), "status": status,
                    "error": row.get("error", ""),
                })
        task_rows.append({
            "method": method, "task": task, "samples": len(rows), "score": score,
            "ok": counts.get("ok", 0), "oom": counts.get("oom", 0),
            "failed": counts.get("failed", 0), "missing": counts.get("missing", 0),
        })
        trace = report.get("trace", {})
        trace_rows.append({
            "method": method, "task": task, "events": trace.get("events", 0),
            "samples_with_events": trace.get("samples_with_events", 0),
            "sample_coverage": trace.get("sample_coverage", 0),
            "budget_bad": trace.get("budget_failures", 0),
            "sink_bad": trace.get("sink_failures", 0),
            "duplicate_bad": trace.get("duplicate_failures", 0),
            "invalid_bad": trace.get("invalid_id_failures", 0),
            "field_bad": trace.get("field_failures", 0),
            "continuity_field_bad": trace.get("continuity_field_failures", 0),
        })
    if total != 3950:
        raise RuntimeError(f"{method}: expected 3950 predictions, got {total}")
    _atomic_concatenate_jsonl(prediction_paths, run_root / "predictions.jsonl")
    macro = task_macro(task_scores)
    task_rows.append({
        "method": method, "task": "17_task_macro", "samples": total, "score": macro,
        "ok": total - len(oom_rows), "oom": sum(row["status"] == "oom" for row in oom_rows),
        "failed": sum(row["status"] == "failed" for row in oom_rows), "missing": 0,
    })
    atomic_write_csv(run_root / "scores_by_task.csv", task_rows, [
        "method", "task", "samples", "score", "ok", "oom", "failed", "missing",
    ])
    atomic_write_csv(run_root / "OOM_MANIFEST.csv", oom_rows, [
        "method", "task", "sample_index", "sample_id", "status", "error",
    ])
    atomic_write_csv(run_root / "ACTIVE_TRACE_COVERAGE.csv", trace_rows, [
        "method", "task", "events", "samples_with_events", "sample_coverage",
        "budget_bad", "sink_bad", "duplicate_bad", "invalid_bad", "field_bad",
        "continuity_field_bad",
    ])
    trace_failures = sum(
        int(row[field]) for row in trace_rows
        for field in ("budget_bad", "sink_bad", "duplicate_bad", "invalid_bad", "field_bad", "continuity_field_bad")
    )
    result = {
        "created_at": now_iso(), "protocol_id": PROTOCOL_ID, "method": method,
        "tasks": len(TASKS), "predictions": total, "raw_macro": macro,
        "oom": sum(row["status"] == "oom" for row in oom_rows),
        "failed": sum(row["status"] == "failed" for row in oom_rows),
        "missing": 0, "trace_required": trace_required,
        "trace_events": sum(int(row["events"]) for row in trace_rows),
        "trace_audit_failures": trace_failures,
        "decision": "PASS" if (not trace_required or trace_failures == 0) else "FAIL",
    }
    atomic_write_json(run_root / "RUN_DECISION.json", result)
    return result


def _quality_root(pipeline_dir: Path, pointer_name: str, prefix: str) -> Path:
    pointer = pipeline_dir / pointer_name
    if pointer.exists():
        return Path(load_json(pointer)["path"])
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = REPO_ROOT / "results" / f"{prefix}_{stamp}"
    atomic_write_json(pointer, {"created_at": now_iso(), "path": str(root)})
    return root


def _run_task_set(
    pipeline_dir: Path,
    run_root: Path,
    safety_policy: Path,
    *,
    stage: int,
    method: str,
    task_indices: Sequence[Tuple[str, Optional[Sequence[int]]]],
    dry_run: bool,
    failure_is_soft: bool,
) -> Dict[str, Any]:
    statuses = []
    for position, (task, indices) in enumerate(task_indices, start=1):
        if (pipeline_dir / "STOP_REQUESTED").exists():
            return _write_blocker(
                pipeline_dir, stage, "HARD_BLOCKED", "operator_stop_requested",
                {"completed_units": position - 1},
            )
        atomic_write_json(pipeline_dir / "PROGRESS.json", {
            "updated_at": now_iso(), "stage": stage, "method": method, "task": task,
            "task_index": position, "task_total": len(task_indices),
            "completed_tasks": position - 1,
        })
        status, details = _run_task_wrapper(
            pipeline_dir, run_root, safety_policy, stage=stage, method=method,
            task=task, sample_indices=indices, dry_run=dry_run,
        )
        statuses.append({"task": task, "sample_indices": indices, "status": status, "details": details})
        _append_registry(pipeline_dir, {
            "stage": stage, "method": method, "task": task, "status": status,
            "path": str(run_root / "tasks" / task),
        })
        if status == "HARD_BLOCKED":
            return _write_blocker(
                pipeline_dir, stage, "HARD_BLOCKED", f"{method}_task_hard_failure", details,
            )
        if status == "FAILED":
            return _write_blocker(
                pipeline_dir, stage, "SOFT_BLOCKED" if failure_is_soft else "HARD_BLOCKED",
                f"{method}_worker_failed_after_one_retry", details,
                attempted=["Initial isolated worker", "One automatic recoverable worker retry"],
            )
    return {"decision": "NOT_RUN" if dry_run else "PASS", "units": statuses}


def run_stage_02(pipeline_dir: Path, safety_policy: Path, *, dry_run: bool) -> Dict[str, Any]:
    run_root = _quality_root(
        pipeline_dir, "QWEN_CC_RUN_ROOT.json", "qwen_cc_unified_17task_full"
    )
    atomic_write_json(run_root / "RUN_MANIFEST.json", {
        "created_at": now_iso(), "protocol_id": PROTOCOL_ID, "stage": 2,
        "method": "cc_sliminfer_v1", "tasks": list(TASKS), "sample_count": 3950,
        "batch_size": 1, "greedy": True, "active_trace": "full",
        "historical_predictions_overwritten": False,
    })
    result = _run_task_set(
        pipeline_dir, run_root, safety_policy, stage=2, method="cc_sliminfer_v1",
        task_indices=[(task, None) for task in TASKS], dry_run=dry_run,
        failure_is_soft=False,
    )
    if result["decision"] != "PASS":
        if result["decision"] == "NOT_RUN":
            return {"stage": 2, "name": "qwen_cc_full", "decision": "NOT_RUN", "run_root": str(run_root)}
        return result
    final = _finalize_quality_run(run_root, "cc_sliminfer_v1", trace_required=True)
    if final["decision"] != "PASS":
        return _write_blocker(pipeline_dir, 2, "HARD_BLOCKED", "cc_final_audit_failed", final)
    decision = {
        "stage": 2, "name": "qwen_cc_full", "created_at": now_iso(),
        "decision": "PASS", "run_root": str(run_root), "summary": final, "next_stage": 3,
    }
    atomic_write_json(stage_decision_path(pipeline_dir, 2), decision)
    return decision


def run_stage_04(pipeline_dir: Path, safety_policy: Path, *, dry_run: bool) -> Dict[str, Any]:
    root = _quality_root(
        pipeline_dir, "QWEN_FULLKV_RUN_ROOT.json", "qwen_fullkv_unified_17task"
    )
    atomic_write_json(root / "OFFICIAL_PATH_AUDIT.json", {
        "created_at": now_iso(), "decision": "PASS", "method": "full_kv_fa2",
        "model": str(REPO_ROOT / ".venv"), "runtime": "Hugging Face AutoModelForCausalLM",
        "attention": "flash_attention_2", "sliminfer_pruning": False,
        "cpu_offload": False, "silent_attention_fallback_allowed": False,
    })
    atomic_write_json(root / "CPU_CONFIG_AUDIT.json", {
        "created_at": now_iso(), "decision": "PASS", "protocol_id": PROTOCOL_ID,
        "same_model_tokenizer_prompt_decode_scorer": True, "batch_size": 1,
    })
    phases: Sequence[Tuple[str, Sequence[Tuple[str, Optional[Sequence[int]]]]]] = (
        ("smoke", (("qasper", (0, 1)),)),
        ("gate", tuple((task, tuple(indices)) for task, indices in CANARY_INDICES.items())),
        ("full", tuple((task, None) for task in TASKS)),
    )
    for phase, units in phases:
        phase_root = root / phase
        phase_result = _run_task_set(
            pipeline_dir, phase_root, safety_policy, stage=4, method="full_kv_fa2",
            task_indices=units, dry_run=dry_run, failure_is_soft=True,
        )
        if phase_result["decision"] != "PASS":
            return phase_result
        atomic_write_json(root / f"{phase.upper()}_DECISION.json", {
            "created_at": now_iso(), "decision": "PASS", "units": len(units),
        })
    final = _finalize_quality_run(root / "full", "full_kv_fa2", trace_required=False)
    decision = {
        "stage": 4, "name": "fullkv_fa2", "created_at": now_iso(),
        "decision": "PASS", "run_root": str(root), "summary": final, "next_stage": 5,
    }
    atomic_write_json(stage_decision_path(pipeline_dir, 4), decision)
    return decision


EXTERNAL_SPECS = {
    5: {
        "method": "minference_official", "directory": REPO_ROOT / "MInference",
        "remote": "https://github.com/microsoft/MInference.git",
    },
    6: {
        "method": "flexprefill_official", "directory": REPO_ROOT / "FlexPrefill",
        "remote": "https://github.com/ByteDance-Seed/FlexPrefill.git",
    },
    7: {
        "method": "lazyllm_official", "directory": REPO_ROOT / "LazyLLM",
        "remote": None,
    },
}


def _git_repo_identity(path: Path) -> Dict[str, Any]:
    if not (path / ".git").exists():
        return {"present": False, "path": str(path)}
    def query(*args: str) -> str:
        proc = subprocess.run(
            ["git", "-C", str(path), *args], text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
        )
        return proc.stdout.strip() if proc.returncode == 0 else ""
    return {
        "present": True, "path": str(path), "remote": query("remote", "get-url", "origin"),
        "commit": query("rev-parse", "HEAD"), "dirty": bool(query("status", "--short")),
    }


def run_external_stage(pipeline_dir: Path, stage: int, *, dry_run: bool) -> Dict[str, Any]:
    spec = EXTERNAL_SPECS[stage]
    identity = _git_repo_identity(spec["directory"])
    expected_remote = spec["remote"]
    remote_match = bool(expected_remote and identity.get("remote") == expected_remote)
    adapter = REPO_ROOT / "tools/external_baselines" / f"{spec['method']}_adapter.py"
    isolated_env = spec["directory"] / ".venv-qwen-unified"
    audit = {
        "created_at": now_iso(), "stage": stage, "method": spec["method"],
        "identity": identity, "expected_remote": expected_remote,
        "official_remote_match": remote_match, "adapter": str(adapter),
        "adapter_present": adapter.exists(), "isolated_env": str(isolated_env),
        "isolated_env_present": isolated_env.exists(),
        "base_environment_modified": False, "approximation_used": False,
    }
    atomic_write_json(pipeline_dir / f"STAGE_{stage:02d}_OFFICIAL_PATH_AUDIT.json", audit)
    if dry_run:
        return {"stage": stage, "decision": "NOT_RUN", "audit": audit}
    if stage == 7 and expected_remote is None:
        reason = "lazyllm_official_repository_identity_unconfirmed"
    elif not remote_match:
        reason = f"{spec['method']}_official_repository_unverified"
    elif not adapter.exists():
        reason = f"{spec['method']}_thin_adapter_not_implemented"
    elif not isolated_env.exists():
        reason = f"{spec['method']}_isolated_environment_not_available"
    else:
        reason = f"{spec['method']}_official_gate_requires_explicit_adapter_validation"
    return _write_blocker(
        pipeline_dir, stage, "SOFT_BLOCKED", reason, audit,
        attempted=[
            "Inspected local Git remote and commit without network mutation",
            "Checked for a method-specific isolated environment",
            "Checked for an official thin schema adapter",
            "Declined to install into the base Qwen environment or substitute a local approximation",
        ],
    )


def _run_control_tool(
    pipeline_dir: Path, stage: int, tool: str, *, dry_run: bool,
    execute_flag: Optional[str | Sequence[str]] = None,
) -> Dict[str, Any]:
    command = [str(PYTHON), str(REPO_ROOT / "tools" / tool), "--pipeline-dir", str(pipeline_dir)]
    if dry_run:
        command.append("--dry-run")
    elif execute_flag:
        if isinstance(execute_flag, str):
            command.append(execute_flag)
        else:
            command.extend(execute_flag)
    proc = subprocess.run(command, cwd=REPO_ROOT, check=False)
    if proc.returncode != 0:
        existing = read_stage_decision(pipeline_dir, stage)
        if existing and existing.get("decision") in {"SOFT_BLOCKED", "HARD_BLOCKED"}:
            return existing
        return _write_blocker(
            pipeline_dir, stage, "SOFT_BLOCKED", f"stage_{stage:02d}_control_tool_failed",
            {"command": command, "returncode": proc.returncode},
        )
    return load_json(stage_decision_path(pipeline_dir, stage))


def execute_stage(
    stage: int,
    pipeline_dir: Path,
    safety_policy: Path,
    audit_dir: Path,
    *,
    dry_run: bool,
) -> Dict[str, Any]:
    if stage == 0:
        return {"stage": 0, "decision": "NOT_RUN", "reason": "dry_run"} if dry_run else run_preflight(
            pipeline_dir, safety_policy, audit_dir, verify_history=True
        )
    previous = read_stage_decision(pipeline_dir, stage - 1)
    if not previous or previous.get("decision") not in ACCEPTED_PREVIOUS:
        return _write_blocker(
            pipeline_dir, stage, "HARD_BLOCKED", "previous_stage_not_accepted",
            {"previous_stage": stage - 1, "previous_decision": previous},
        )
    gate = run_protocol_gate(pipeline_dir, stage)
    if gate["decision"] != "PASS":
        return _write_blocker(pipeline_dir, stage, "HARD_BLOCKED", "protocol_gate_failed", gate)
    if stage == 1:
        return run_canary(pipeline_dir, safety_policy, execute=not dry_run)
    if stage == 2:
        return run_stage_02(pipeline_dir, safety_policy, dry_run=dry_run)
    if stage == 3:
        return _run_control_tool(pipeline_dir, 3, "analyze_qwen_unified_suite.py", dry_run=dry_run)
    if stage == 4:
        return run_stage_04(pipeline_dir, safety_policy, dry_run=dry_run)
    if stage in EXTERNAL_SPECS:
        return run_external_stage(pipeline_dir, stage, dry_run=dry_run)
    if stage == 8:
        return _run_control_tool(
            pipeline_dir, 8, "run_qwen_unified_efficiency.py", dry_run=dry_run, execute_flag="--execute"
        )
    if stage in {9, 10}:
        return _run_control_tool(
            pipeline_dir, stage, "run_qwen_ablation_suite.py", dry_run=dry_run,
            execute_flag=(f"--stage-{stage}", "--execute"),
        )
    if stage == 11:
        return _run_control_tool(
            pipeline_dir, 11, "run_qwen_oom_fill.py", dry_run=dry_run, execute_flag="--execute"
        )
    if stage == 12:
        return _run_control_tool(
            pipeline_dir, 12, "export_qwen_unified_suite.py", dry_run=dry_run
        )
    raise ValueError(stage)


def status(pipeline_dir: Path) -> Dict[str, Any]:
    return {
        "pipeline_dir": str(pipeline_dir),
        "state": load_json(pipeline_dir / "PIPELINE_STATE.json"),
        "pid": load_json(pipeline_dir / "PID.json"),
        "progress": load_json(pipeline_dir / "PROGRESS.json") if (pipeline_dir / "PROGRESS.json").exists() else None,
        "watchdog": load_json(pipeline_dir / "WATCHDOG_STATUS.json") if (pipeline_dir / "WATCHDOG_STATUS.json").exists() else None,
        "stage_decisions": {
            str(row["number"]): read_stage_decision(pipeline_dir, row["number"])
            for row in STAGES if stage_decision_path(pipeline_dir, row["number"]).exists()
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-dir", type=Path)
    parser.add_argument("--safety-policy", type=Path, default=DEFAULT_SAFETY_POLICY)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--auto-advance", action="store_true")
    parser.add_argument("--start-stage", type=int)
    parser.add_argument("--stop-after-stage", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--no-auto-next", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pipeline_dir = (args.pipeline_dir or new_pipeline_dir()).resolve()
    args.safety_policy = args.safety_policy.resolve()
    args.audit_dir = args.audit_dir.resolve()
    initialize_pipeline(pipeline_dir, args.safety_policy)
    if args.status:
        print(json.dumps(status(pipeline_dir), indent=2))
        return 0
    with PipelineLock(pipeline_dir):
        pid = load_json(pipeline_dir / "PID.json")
        pid.update({"updated_at": now_iso(), "orchestrator_pid": os.getpid()})
        atomic_write_json(pipeline_dir / "PID.json", pid)
        state = load_json(pipeline_dir / "PIPELINE_STATE.json")
        if args.start_stage is not None:
            start = args.start_stage
        elif args.resume:
            start = int(state.get("next_stage", 0) or 0)
        else:
            start = 0
        auto = bool(args.auto_advance or not args.no_auto_next)
        stop = args.stop_after_stage if args.stop_after_stage is not None else (12 if auto else start)
        if start < 0 or stop > 12 or stop < start:
            raise RuntimeError(f"Invalid stage range {start}..{stop}")
        last_result: Dict[str, Any] = {}
        for stage in range(start, stop + 1):
            stage_name = _stage_row(stage)["name"]
            update_state(
                pipeline_dir, state="RUNNING", current_stage=stage,
                current_stage_name=stage_name, next_stage=stage,
            )
            try:
                last_result = execute_stage(
                    stage, pipeline_dir, args.safety_policy, args.audit_dir, dry_run=args.dry_run
                )
            except Exception as exc:
                last_result = {
                    "stage": stage, "name": stage_name, "created_at": now_iso(),
                    "decision": "HARD_BLOCKED", "reason": "stage_exception",
                    "error": repr(exc), "traceback": traceback.format_exc(), "next_stage": None,
                }
                atomic_write_json(stage_decision_path(pipeline_dir, stage), last_result)
            decision = last_result.get("decision")
            if decision == "PASS":
                update_state(
                    pipeline_dir, state="COMPLETE" if stage == 12 else "PASS",
                    last_passed_stage=stage, next_stage=stage + 1 if stage < 12 else None,
                )
            elif decision == "SOFT_BLOCKED":
                update_state(
                    pipeline_dir, state="SOFT_BLOCKED", last_soft_blocked_stage=stage,
                    next_stage=stage + 1 if stage < 12 else None,
                )
                if stage == 12:
                    break
                continue
            elif decision == "NOT_RUN" and args.dry_run:
                update_state(pipeline_dir, state="PAUSED", next_stage=stage)
                break
            else:
                update_state(pipeline_dir, state="HARD_BLOCKED", next_stage=stage)
                break
        pid = load_json(pipeline_dir / "PID.json")
        pid.update({
            "updated_at": now_iso(), "orchestrator_pid": None,
            "task_worker_pid": None, "gpu_worker_pid": None,
        })
        atomic_write_json(pipeline_dir / "PID.json", pid)
    print(json.dumps({"pipeline_dir": str(pipeline_dir), "result": last_result}, indent=2))
    return 0 if last_result.get("decision") in {"PASS", "SOFT_BLOCKED", "NOT_RUN"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
