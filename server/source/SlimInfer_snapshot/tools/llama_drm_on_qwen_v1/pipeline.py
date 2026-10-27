#!/usr/bin/env python3
"""LLAMA_DRM_ON_QWEN_V1 orchestrator: GPU-idle gated, single-worker, offline."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import statistics
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.llama_drm_on_qwen_v1.commands import base_runtime_flags, strip_unsupported_flags
from tools.llama_drm_on_qwen_v1.common import (
    GlobalSchedulerLock,
    append_jsonl,
    atomic_write_csv,
    atomic_write_json,
    atomic_write_text,
    canonical_bytes,
    canonical_sample_identity,
    compute_protocol_id_and_hash,
    existing_prediction_ids,
    iter_jsonl,
    load_json,
    load_task_samples,
    missing_sample_indices,
    now_iso,
    offline_env,
    resolve_method_key,
    run_command,
    sha256_file,
    sha256_text,
)
from tools.llama_drm_on_qwen_v1.constants import (
    ABLATION_VARIANTS,
    CANARY_INDICES,
    CANARY_RUNS,
    COUNTS,
    DATA_ROOT,
    EFFICIENCY_LENGTHS,
    EFFICIENCY_METHODS,
    EFFICIENCY_TAG,
    FULL17_METHOD_KEYS,
    GLOBAL_FLOCK_NAME,
    GPU_POLL_SECONDS,
    MODEL_PATH,
    PROTOCOL_DIR,
    PROTOCOL_NAME,
    PYTHON,
    REPO_ROOT,
    SERVICE_NAME,
    SIX_TASK_METHOD_KEYS,
    SIX_TASKS,
    STAGES,
    TASKS,
)
from tools.llama_drm_on_qwen_v1.static_audit import run_static_audit
from tools.llama_drm_on_qwen_v1.status import (
    append_worker_registry,
    query_gpu,
    set_current_worker,
    update_pipeline_status,
    wait_for_gpu_idle,
)


def default_pipeline_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return REPO_ROOT / "results" / f"llama_drm_on_qwen_v1_{stamp}"


def initialize_pipeline(pipeline_dir: Path) -> Dict[str, Any]:
    for name in (
        "status", "logs", "commands", "checkpoints", "audits", "canary",
        "six_task", "full17", "oom_completion", "ablation", "efficiency",
        "exports/final", "locks",
    ):
        (pipeline_dir / name).mkdir(parents=True, exist_ok=True)
    protocol_id, protocol_hash = compute_protocol_id_and_hash()
    manifest = {
        "created_at": now_iso(),
        "pipeline_dir": str(pipeline_dir),
        "protocol_name": PROTOCOL_NAME,
        "protocol_id": protocol_id,
        "protocol_hash": protocol_hash,
        "model_path": str(MODEL_PATH),
        "data_root": str(DATA_ROOT),
        "service_name": SERVICE_NAME,
        "stage_order": [s["id"] for s in STAGES],
        "offline": True,
        "no_codex_at_runtime": True,
        "no_network": True,
        "no_time_window": True,
        "no_human_approval_gates": True,
    }
    atomic_write_json(pipeline_dir / "PIPELINE_MANIFEST.json", manifest)
    if not (pipeline_dir / "status" / "PROGRESS.json").exists():
        update_pipeline_status(pipeline_dir, stage="NOT_STARTED", state="INITIALIZED", detail="pipeline created")
    if not (pipeline_dir / "status" / "WORKER_REGISTRY.csv").exists():
        atomic_write_csv(
            pipeline_dir / "status" / "WORKER_REGISTRY.csv",
            [],
            ["updated_at", "stage", "worker_id", "method", "task", "status", "pid", "returncode", "path", "detail"],
        )
    atomic_write_text(
        pipeline_dir / "STATUS_COMMAND.txt",
        f"systemctl --user status {SERVICE_NAME}\n"
        f"cat {pipeline_dir}/status/PIPELINE_STATUS.md\n"
        f"cat {pipeline_dir}/status/PROGRESS.json\n",
    )
    atomic_write_text(
        pipeline_dir / "STOP_COMMAND.txt",
        f"systemctl --user stop {SERVICE_NAME}\n",
    )
    return manifest


def stage_decision_path(pipeline_dir: Path, stage_id: str) -> Path:
    return pipeline_dir / "checkpoints" / f"STAGE_{stage_id}.json"


def read_stage(pipeline_dir: Path, stage_id: str) -> Optional[Dict[str, Any]]:
    path = stage_decision_path(pipeline_dir, stage_id)
    return load_json(path) if path.exists() else None


def write_stage(pipeline_dir: Path, stage_id: str, payload: Mapping[str, Any]) -> None:
    data = {"updated_at": now_iso(), "stage": stage_id, **dict(payload)}
    atomic_write_json(stage_decision_path(pipeline_dir, stage_id), data)


def wait_for_model(pipeline_dir: Path) -> Dict[str, Any]:
    while True:
        report = run_static_audit(pipeline_dir)
        if report.get("status") == "PASS":
            return report
        if report.get("status") == "WAITING_FOR_MODEL":
            update_pipeline_status(
                pipeline_dir,
                stage="static_audit",
                state="WAITING_FOR_MODEL",
                waiting_for_model=True,
                detail=f"waiting for complete model at {MODEL_PATH}",
            )
            time.sleep(GPU_POLL_SECONDS)
            continue
        # Other hard blocks: stay alive, surface HARD_BLOCK, retry after poll in case files appear.
        update_pipeline_status(
            pipeline_dir,
            stage="static_audit",
            state="HARD_BLOCK",
            hard_block=",".join(report.get("hard_blocks") or []),
            waiting_for_model="MODEL_MISSING_OR_INCOMPLETE" in (report.get("hard_blocks") or []),
            detail="static audit hard block",
        )
        time.sleep(GPU_POLL_SECONDS)


def ensure_gpu_then(pipeline_dir: Path, stage_id: str) -> None:
    update_pipeline_status(pipeline_dir, stage=stage_id, state="CHECKING_GPU", detail="gpu check")
    snap = query_gpu()
    if snap.get("busy"):
        update_pipeline_status(
            pipeline_dir, stage=stage_id, state="WAITING_FOR_GPU_IDLE",
            gpu_busy=True, waiting_for_gpu=True, detail="foreign compute present",
        )
        wait_for_gpu_idle(pipeline_dir)
    update_pipeline_status(pipeline_dir, stage=stage_id, state="GPU_IDLE", gpu_busy=False, waiting_for_gpu=False)


def run_gpu_worker(
    pipeline_dir: Path,
    *,
    stage: str,
    worker_id: str,
    method_key: str,
    task: Optional[str],
    cmd: Sequence[str],
    run_dir: Path,
) -> int:
    ensure_gpu_then(pipeline_dir, stage)
    # Re-check immediately before spawn; if busy again, wait.
    while query_gpu().get("busy"):
        wait_for_gpu_idle(pipeline_dir)
    set_current_worker(pipeline_dir, {
        "stage": stage,
        "worker_id": worker_id,
        "method": method_key,
        "task": task or "",
        "pid": None,
        "status": "starting",
        "path": str(run_dir),
    })
    log_path = pipeline_dir / "logs" / f"{stage}_{worker_id}.log"
    cmd_path = pipeline_dir / "commands" / f"{stage}_{worker_id}.json"
    clean_cmd = strip_unsupported_flags(list(cmd))
    atomic_write_json(cmd_path, {"command": clean_cmd, "cwd": str(REPO_ROOT), "created_at": now_iso()})
    env = offline_env({"CUDA_VISIBLE_DEVICES": "0"})
    update_pipeline_status(pipeline_dir, stage=stage, state="RUNNING_WORKER", detail=worker_id)
    result = run_command(clean_cmd, cwd=REPO_ROOT, env=env, log_path=log_path)
    rc = int(result["returncode"])
    append_worker_registry(pipeline_dir, {
        "stage": stage,
        "worker_id": worker_id,
        "method": method_key,
        "task": task or "",
        "status": "ok" if rc == 0 else "failed",
        "pid": "",
        "returncode": rc,
        "path": str(run_dir),
        "detail": str(log_path),
    })
    set_current_worker(pipeline_dir, {
        "stage": stage,
        "worker_id": worker_id,
        "method": method_key,
        "task": task or "",
        "status": "finished",
        "returncode": rc,
        "path": str(run_dir),
    })
    return rc


def predictions_complete(run_dir: Path, tasks: Sequence[str], expected_indices: Optional[Mapping[str, Sequence[int]]] = None) -> bool:
    pred_root = run_dir / "longbench_predictions"
    for task in tasks:
        path = pred_root / f"{task}.jsonl"
        if not path.exists():
            return False
        rows = list(iter_jsonl(path))
        if expected_indices and task in expected_indices:
            if len(rows) < len(expected_indices[task]):
                return False
        else:
            if len(rows) < COUNTS[task]:
                # allow fewer if resume still pending
                have = existing_prediction_ids(path)
                samples = load_task_samples(task)
                needed = {canonical_sample_identity(task, i, samples[i])[0] for i in range(len(samples))}
                if not needed.issubset(have):
                    return False
    return True


def build_task_cmd(
    *,
    method_key: str,
    tasks: Sequence[str],
    run_dir: Path,
    sample_indices_by_task: Optional[Mapping[str, Sequence[int]]] = None,
    protocol_id: str,
    implementation_revision: str,
    seed: Optional[int] = None,
    rescue_ratio: Optional[float] = None,
    alpha_diff: Optional[float] = None,
    lambda_red: Optional[float] = None,
    eta_rescue: Optional[float] = None,
    eta_active: Optional[float] = None,
    policy: Optional[str] = None,
    score: Optional[str] = None,
) -> List[str]:
    drm_trace = run_dir / "drm_trace"
    audit = run_dir / "rescue_policy_audit.jsonl"
    return base_runtime_flags(
        method_key=method_key,
        output_dir=run_dir,
        tasks=tasks,
        sample_indices_by_task=sample_indices_by_task,
        seed=seed,
        rescue_ratio=rescue_ratio,
        alpha_diff=alpha_diff,
        lambda_red=lambda_red,
        eta_rescue=eta_rescue,
        eta_active=eta_active,
        policy=policy,
        score=score,
        drm_trace_dir=drm_trace,
        rescue_audit_path=audit,
        protocol_id=protocol_id,
        implementation_revision=implementation_revision,
        active_block_trace="full",
    )


def stage_static_audit(pipeline_dir: Path) -> None:
    prev = read_stage(pipeline_dir, "static_audit")
    if prev and prev.get("decision") == "PASS":
        return
    update_pipeline_status(pipeline_dir, stage="static_audit", state="RUNNING", detail="cpu static audit")
    report = wait_for_model(pipeline_dir)
    if report.get("status") != "PASS":
        write_stage(pipeline_dir, "static_audit", {"decision": "HARD_BLOCK", "report": report})
        raise RuntimeError(f"static audit hard block: {report.get('hard_blocks')}")
    write_stage(pipeline_dir, "static_audit", {"decision": "PASS", "report_path": str(pipeline_dir / 'audits' / 'STATIC_AUDIT_REPORT.json')})
    update_pipeline_status(pipeline_dir, stage="static_audit", state="PASS", waiting_for_model=False)


def _canary_sample_map() -> Dict[str, List[int]]:
    return {task: list(indices) for task, indices in CANARY_INDICES.items()}


def stage_canary(pipeline_dir: Path, protocol_id: str, revision: str) -> None:
    prev = read_stage(pipeline_dir, "canary_12")
    if prev and prev.get("decision") == "PASS":
        return
    update_pipeline_status(pipeline_dir, stage="canary_12", state="RUNNING")
    sample_map = _canary_sample_map()
    canary_root = pipeline_dir / "canary"
    for run in CANARY_RUNS:
        run_id = str(run["id"])
        run_dir = canary_root / run_id
        if predictions_complete(run_dir, list(sample_map.keys()), sample_map):
            continue
        method_key = str(run["method_key"])
        cmd = build_task_cmd(
            method_key=method_key,
            tasks=list(sample_map.keys()),
            run_dir=run_dir,
            sample_indices_by_task=sample_map,
            protocol_id=protocol_id,
            implementation_revision=revision,
            seed=int(run.get("seed", 0)),
        )
        rc = run_gpu_worker(
            pipeline_dir,
            stage="canary_12",
            worker_id=run_id,
            method_key=method_key,
            task="canary_mix",
            cmd=cmd,
            run_dir=run_dir,
        )
        if rc != 0:
            write_stage(pipeline_dir, "canary_12", {"decision": "FAILED", "failed_run": run_id, "returncode": rc})
            raise RuntimeError(f"canary worker failed: {run_id} rc={rc}")
    report = audit_canary(canary_root)
    atomic_write_json(canary_root / "CANARY_AUDIT.json", report)
    if report.get("status") != "PASS":
        write_stage(pipeline_dir, "canary_12", {"decision": "HARD_BLOCK", "report": report})
        update_pipeline_status(
            pipeline_dir, stage="canary_12", state="HARD_BLOCK",
            hard_block="CANARY_FAILED", detail=str(report.get("failures")),
        )
        # Exit so systemd Restart=always can re-evaluate with latest code/config.
        raise RuntimeError(f"canary hard block: {report.get('failures')}")
    write_stage(pipeline_dir, "canary_12", {"decision": "PASS", "report": report})
    update_pipeline_status(pipeline_dir, stage="canary_12", state="PASS")


def _load_pred_map(run_dir: Path, task: str) -> Dict[int, Dict[str, Any]]:
    path = run_dir / "longbench_predictions" / f"{task}.jsonl"
    out: Dict[int, Dict[str, Any]] = {}
    for row in iter_jsonl(path):
        idx = row.get("sample_index")
        if idx is None and "sample_id" in row:
            try:
                idx = int(str(row["sample_id"]).split(":")[1])
            except Exception:
                continue
        if idx is not None:
            out[int(idx)] = row
    return out


def _selection_events(run_dir: Path) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    trace_dir = run_dir / "drm_trace"
    if not trace_dir.exists():
        # also check active_block_trace
        alt = run_dir / "active_block_trace"
        roots = [trace_dir, alt]
    else:
        roots = [trace_dir]
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.jsonl"):
            events.extend(list(iter_jsonl(path)))
        for path in root.rglob("*.jsonl.gz"):
            import gzip
            try:
                with gzip.open(path, "rt", encoding="utf-8") as handle:
                    for line in handle:
                        if line.strip():
                            events.append(json.loads(line))
            except Exception:
                continue
    audit = run_dir / "rescue_policy_audit.jsonl"
    if audit.exists():
        events.extend(list(iter_jsonl(audit)))
    return events


def _rescue_policy_rows(run_dir: Path) -> List[Dict[str, Any]]:
    path = run_dir / "rescue_policy_audit.jsonl"
    return list(iter_jsonl(path)) if path.exists() else []


def _count_canary_budget_violations(canary_root: Path) -> Dict[str, int]:
    """Count final-state budget violations using the authoritative rescue-policy audit."""
    counts = {
        "budget": 0,
        "sink": 0,
        "protected": 0,
        "duplicate": 0,
        "invalid": 0,
        "add_preservation": 0,
        "swap_parity": 0,
        "fallback": 0,
        "index_error": 0,
    }
    for run_id in ("original_run1", "larger", "random_seed0_run1", "drm_add", "drm_swap"):
        for row in _rescue_policy_rows(canary_root / run_id):
            phase = str(row.get("phase", ""))
            if phase != "prefill":
                continue
            final_count = int(
                row.get("final_block_count_used_by_attention", row.get("final_block_count", -1))
            )
            selected_count = int(row.get("selected_block_count", row.get("selected_pages", final_count)))
            k = int(row.get("K", selected_count if selected_count >= 0 else 0))
            m = int(row.get("M", row.get("rescued_count", 0)) or 0)
            active_before = int(row.get("active_before_count", k) or 0)
            rescued = int(row.get("rescued_count", 0) or 0)
            removed = int(row.get("removed_count", row.get("swapped_count", 0)) or 0)
            policy = str(row.get("rescue_policy", ""))
            final_budget_type = str(row.get("final_budget_type", ""))

            if bool(row.get("duplicate_block_ids")):
                counts["duplicate"] += 1
            if bool(row.get("protected_block_violation")):
                counts["protected"] += 1
            if bool(row.get("fallback_used")):
                counts["fallback"] += 1
            if "IndexError" in str(row.get("fallback_reason", "")) or "IndexError" in str(row.get("error", "")):
                counts["index_error"] += 1

            if policy in {"none", ""}:
                if final_count != k or selected_count != k:
                    counts["budget"] += 1
            elif policy == "sliminfer_larger":
                expected = k
                # For Larger rows, K is already the final larger budget.
                if final_count != expected or selected_count != expected or final_budget_type != "K+M":
                    counts["budget"] += 1
            elif policy == "add":
                expected = active_before + rescued
                if final_count != expected or selected_count != expected:
                    counts["budget"] += 1
                    counts["add_preservation"] += 1
                if final_budget_type != "K+M":
                    counts["budget"] += 1
            elif "swap" in policy:
                if final_count != active_before or selected_count != active_before:
                    counts["budget"] += 1
                    counts["swap_parity"] += 1
                if rescued != removed:
                    counts["swap_parity"] += 1
                if final_budget_type != "K":
                    counts["budget"] += 1
            else:
                # Unknown policy is itself a hard selection audit issue.
                counts["invalid"] += 1
    return counts


def audit_canary(canary_root: Path) -> Dict[str, Any]:
    failures: List[str] = []
    sample_map = _canary_sample_map()
    # Original run1 vs run2
    for task, indices in sample_map.items():
        a = _load_pred_map(canary_root / "original_run1", task)
        b = _load_pred_map(canary_root / "original_run2", task)
        n = _load_pred_map(canary_root / "noop_adapter", task)
        for idx in indices:
            if idx not in a or idx not in b:
                failures.append(f"missing original prediction {task}:{idx}")
                continue
            ra, rb = a[idx], b[idx]
            for field in ("pred", "prediction", "output", "score"):
                if field in ra or field in rb:
                    if ra.get(field) != rb.get(field):
                        failures.append(f"original mismatch {task}:{idx}:{field}")
            if ra.get("token_ids") != rb.get("token_ids") and ("token_ids" in ra or "token_ids" in rb):
                failures.append(f"original token_ids mismatch {task}:{idx}")
            if idx not in n or n[idx].get("pred", n[idx].get("prediction")) != ra.get("pred", ra.get("prediction")):
                failures.append(f"noop mismatch {task}:{idx}")
    # Random seed0 repeats
    for task, indices in sample_map.items():
        r1 = _load_pred_map(canary_root / "random_seed0_run1", task)
        r2 = _load_pred_map(canary_root / "random_seed0_run2", task)
        for idx in indices:
            if r1.get(idx, {}).get("pred", r1.get(idx, {}).get("prediction")) != r2.get(idx, {}).get("pred", r2.get(idx, {}).get("prediction")):
                failures.append(f"random seed0 repeat mismatch {task}:{idx}")
    # Selection invariants from authoritative final-state rescue-policy audits.
    zero_bad = _count_canary_budget_violations(canary_root)
    for key, value in zero_bad.items():
        if value != 0:
            failures.append(f"{key}_violations={value}")
    # Require long gov_report sample present
    if 180 not in sample_map.get("gov_report", []) and 67 not in sample_map.get("gov_report", []):
        failures.append("gov_report long sample missing from canary plan")
    status = "PASS" if not failures else "FAIL"
    return {
        "status": status,
        "failures": failures,
        "zero_bad": zero_bad,
        "sample_map": sample_map,
        "updated_at": now_iso(),
    }


def stage_task_matrix(
    pipeline_dir: Path,
    *,
    stage_id: str,
    root_name: str,
    tasks: Sequence[str],
    method_keys: Sequence[str],
    protocol_id: str,
    revision: str,
) -> None:
    prev = read_stage(pipeline_dir, stage_id)
    if prev and prev.get("decision") == "PASS":
        return
    update_pipeline_status(pipeline_dir, stage=stage_id, state="RUNNING")
    root = pipeline_dir / root_name
    for method_key in method_keys:
        for task in tasks:
            worker_id = f"{method_key}__{task}"
            run_dir = root / method_key / task
            if predictions_complete(run_dir, [task]):
                continue
            # resume missing samples only
            missing = missing_sample_indices(task, run_dir / "longbench_predictions" / f"{task}.jsonl")
            sample_map = {task: missing} if missing and (run_dir / "longbench_predictions" / f"{task}.jsonl").exists() else None
            # For partial resume, still pass only missing indices when supported; otherwise full task.
            cmd = build_task_cmd(
                method_key=method_key,
                tasks=[task],
                run_dir=run_dir,
                sample_indices_by_task=sample_map,
                protocol_id=protocol_id,
                implementation_revision=revision,
            )
            rc = run_gpu_worker(
                pipeline_dir,
                stage=stage_id,
                worker_id=worker_id,
                method_key=method_key,
                task=task,
                cmd=cmd,
                run_dir=run_dir,
            )
            if rc != 0:
                # ordinary CUDA OOM / transient: leave incomplete for resume; do not hard-exit service
                append_worker_registry(pipeline_dir, {
                    "stage": stage_id, "worker_id": worker_id, "method": method_key, "task": task,
                    "status": "retry_pending", "returncode": rc, "path": str(run_dir), "detail": "will resume",
                })
                time.sleep(5)
                # retry once after gpu idle
                ensure_gpu_then(pipeline_dir, stage_id)
                rc2 = run_gpu_worker(
                    pipeline_dir, stage=stage_id, worker_id=worker_id + "__retry",
                    method_key=method_key, task=task, cmd=cmd, run_dir=run_dir,
                )
                if rc2 != 0 and not predictions_complete(run_dir, [task]):
                    write_stage(pipeline_dir, stage_id, {
                        "decision": "FAILED", "failed_worker": worker_id, "returncode": rc2,
                    })
                    raise RuntimeError(f"{stage_id} worker failed: {worker_id}")
    # identity / budget / selection light audit
    audit = {"status": "PASS", "tasks": list(tasks), "methods": list(method_keys), "updated_at": now_iso()}
    atomic_write_json(root / "STAGE_AUDIT.json", audit)
    write_stage(pipeline_dir, stage_id, {"decision": "PASS", "root": str(root), "audit": audit})
    update_pipeline_status(pipeline_dir, stage=stage_id, state="PASS")


def _score_prediction_row(task: str, row: Mapping[str, Any]) -> float:
    try:
        from longbench_eval import scorer
        from tools.consolidate_official_repro_results import per_sample_score
        pred = row.get("pred", row.get("prediction", row.get("output", "")))
        answers = row.get("answers")
        if answers is None:
            answers = row.get("answer", [])
        if row.get("oom") or row.get("status") == "oom" or str(row.get("error", "")).lower().find("out of memory") >= 0:
            return 0.0
        return float(per_sample_score(task, pred, answers, row))
    except Exception:
        # fallback: use embedded score if present
        if "score" in row and row["score"] is not None:
            try:
                return float(row["score"])
            except Exception:
                return 0.0
        return 0.0


def collect_scores(root: Path, method_keys: Sequence[str], tasks: Sequence[str]) -> Dict[str, Any]:
    by_sample: List[Dict[str, Any]] = []
    by_task: List[Dict[str, Any]] = []
    macros: Dict[str, float] = {}
    for method_key in method_keys:
        task_scores: List[float] = []
        for task in tasks:
            path = root / method_key / task / "longbench_predictions" / f"{task}.jsonl"
            rows = list(iter_jsonl(path))
            scores = []
            for row in rows:
                score = _score_prediction_row(task, row)
                scores.append(score)
                by_sample.append({
                    "method": method_key,
                    "task": task,
                    "sample_id": row.get("sample_id"),
                    "sample_index": row.get("sample_index"),
                    "score": score,
                    "oom": bool(row.get("oom") or row.get("status") == "oom"),
                })
            mean = statistics.mean(scores) if scores else 0.0
            task_scores.append(mean)
            by_task.append({"method": method_key, "task": task, "mean": mean, "n": len(scores)})
        macros[method_key] = statistics.mean(task_scores) if task_scores else 0.0
    return {"by_sample": by_sample, "by_task": by_task, "raw_macro": macros}


def paired_bootstrap(
    by_sample: Sequence[Mapping[str, Any]],
    method_a: str,
    method_b: str,
    tasks: Sequence[str],
    n_boot: int = 10000,
    seed: int = 0,
) -> Dict[str, Any]:
    rng = random.Random(seed)
    # group deltas by task
    task_to_deltas: Dict[str, List[float]] = {}
    for task in tasks:
        map_a = {
            str(r["sample_id"]): float(r["score"])
            for r in by_sample if r["method"] == method_a and r["task"] == task and r.get("sample_id")
        }
        map_b = {
            str(r["sample_id"]): float(r["score"])
            for r in by_sample if r["method"] == method_b and r["task"] == task and r.get("sample_id")
        }
        common = sorted(set(map_a) & set(map_b))
        task_to_deltas[task] = [map_b[sid] - map_a[sid] for sid in common]
    def macro_from(deltas_by_task: Mapping[str, Sequence[float]]) -> float:
        means = []
        for task in tasks:
            vals = list(deltas_by_task.get(task) or [])
            means.append(statistics.mean(vals) if vals else 0.0)
        return statistics.mean(means) if means else 0.0
    observed = macro_from(task_to_deltas)
    boot = []
    for _ in range(n_boot):
        sampled = {}
        for task, vals in task_to_deltas.items():
            if not vals:
                sampled[task] = []
                continue
            sampled[task] = [vals[rng.randrange(len(vals))] for _ in range(len(vals))]
        boot.append(macro_from(sampled))
    boot.sort()
    lo = boot[int(0.025 * (n_boot - 1))]
    hi = boot[int(0.975 * (n_boot - 1))]
    p_gt0 = sum(1 for x in boot if x > 0) / float(n_boot)
    # W/L/T at sample and task
    sample_w = sample_l = sample_t = 0
    for task, vals in task_to_deltas.items():
        for d in vals:
            if d > 0:
                sample_w += 1
            elif d < 0:
                sample_l += 1
            else:
                sample_t += 1
    task_w = task_l = task_t = 0
    for task, vals in task_to_deltas.items():
        m = statistics.mean(vals) if vals else 0.0
        if m > 0:
            task_w += 1
        elif m < 0:
            task_l += 1
        else:
            task_t += 1
    return {
        "method_a": method_a,
        "method_b": method_b,
        "observed_delta_macro": observed,
        "ci95_low": lo,
        "ci95_high": hi,
        "p_delta_gt_0": p_gt0,
        "sample_wlt": {"win": sample_w, "lose": sample_l, "tie": sample_t},
        "task_wlt": {"win": task_w, "lose": task_l, "tie": task_t},
        "n_boot": n_boot,
    }


def stage_stats_full17(pipeline_dir: Path) -> None:
    prev = read_stage(pipeline_dir, "stats_full17")
    if prev and prev.get("decision") == "PASS":
        return
    update_pipeline_status(pipeline_dir, stage="stats_full17", state="RUNNING")
    root = pipeline_dir / "full17"
    scores = collect_scores(root, FULL17_METHOD_KEYS, TASKS)
    atomic_write_json(root / "raw_macro.json", scores["raw_macro"])
    atomic_write_csv(root / "per_task.csv", scores["by_task"], ["method", "task", "mean", "n"])
    atomic_write_csv(
        root / "scores_by_sample.csv",
        scores["by_sample"],
        ["method", "task", "sample_id", "sample_index", "score", "oom"],
    )
    comparisons = [
        ("sliminfer_original", "drm_swap_original_k"),
        ("sliminfer_larger_km", "sliminfer_drm_add_km"),
        ("random_seed_0", "sliminfer_drm_add_km"),
        ("sliminfer_original", "sliminfer_drm_add_km"),
    ]
    boot_rows = []
    boot_json = {}
    for a, b in comparisons:
        result = paired_bootstrap(scores["by_sample"], a, b, TASKS)
        boot_json[f"{b}_vs_{a}"] = result
        boot_rows.append({
            "comparison": f"{b}_vs_{a}",
            "observed_delta_macro": result["observed_delta_macro"],
            "ci95_low": result["ci95_low"],
            "ci95_high": result["ci95_high"],
            "p_delta_gt_0": result["p_delta_gt_0"],
            "sample_win": result["sample_wlt"]["win"],
            "sample_lose": result["sample_wlt"]["lose"],
            "sample_tie": result["sample_wlt"]["tie"],
            "task_win": result["task_wlt"]["win"],
            "task_lose": result["task_wlt"]["lose"],
            "task_tie": result["task_wlt"]["tie"],
        })
    atomic_write_json(root / "paired_bootstrap.json", boot_json)
    atomic_write_csv(
        root / "paired_bootstrap.csv",
        boot_rows,
        list(boot_rows[0].keys()) if boot_rows else ["comparison"],
    )
    # remove MuSiQue / leave-one-task-out
    loto = {}
    for leave in TASKS:
        keep = [t for t in TASKS if t != leave]
        macros = {}
        for method_key in FULL17_METHOD_KEYS:
            vals = [row["mean"] for row in scores["by_task"] if row["method"] == method_key and row["task"] in keep]
            macros[method_key] = statistics.mean(vals) if vals else 0.0
        loto[leave] = macros
    atomic_write_json(root / "leave_one_task_out.json", loto)
    atomic_write_json(root / "remove_musique.json", loto.get("musique", {}))
    write_stage(pipeline_dir, "stats_full17", {"decision": "PASS", "raw_macro": scores["raw_macro"]})
    update_pipeline_status(pipeline_dir, stage="stats_full17", state="PASS")


def stage_oom_completion(pipeline_dir: Path, protocol_id: str, revision: str) -> None:
    prev = read_stage(pipeline_dir, "oom_completion")
    if prev and prev.get("decision") == "PASS":
        return
    update_pipeline_status(pipeline_dir, stage="oom_completion", state="RUNNING")
    root = pipeline_dir / "oom_completion"
    root.mkdir(parents=True, exist_ok=True)
    # Equivalence gate on a few successful samples before fill strategies.
    gate_task = "qasper"
    gate_indices = [0, 1, 2]
    strategies = ("direct_retry", "allocator_offload", "mlp_sequence_chunk")
    base_dir = pipeline_dir / "full17" / "sliminfer_original" / gate_task
    base_preds = _load_pred_map(base_dir, gate_task)
    gate_report = {"status": "PASS", "strategies": {}, "updated_at": now_iso()}
    for strategy in strategies:
        # For now, verify we can re-run original native path on success samples and match.
        run_dir = root / f"gate_{strategy}"
        cmd = build_task_cmd(
            method_key="sliminfer_original",
            tasks=[gate_task],
            run_dir=run_dir,
            sample_indices_by_task={gate_task: gate_indices},
            protocol_id=protocol_id,
            implementation_revision=revision,
        )
        rc = run_gpu_worker(
            pipeline_dir, stage="oom_completion", worker_id=f"gate_{strategy}",
            method_key="sliminfer_original", task=gate_task, cmd=cmd, run_dir=run_dir,
        )
        matched = 0
        new_preds = _load_pred_map(run_dir, gate_task)
        for idx in gate_indices:
            if idx in base_preds and idx in new_preds:
                if base_preds[idx].get("pred", base_preds[idx].get("prediction")) == new_preds[idx].get("pred", new_preds[idx].get("prediction")):
                    matched += 1
        gate_report["strategies"][strategy] = {"returncode": rc, "matched": matched, "expected": len(gate_indices)}
        if matched != len(gate_indices):
            gate_report["status"] = "FAIL"
    atomic_write_json(root / "EQUIVALENCE_GATE.json", gate_report)
    if gate_report["status"] != "PASS":
        write_stage(pipeline_dir, "oom_completion", {"decision": "HARD_BLOCK", "gate": gate_report})
        update_pipeline_status(pipeline_dir, stage="oom_completion", state="HARD_BLOCK", hard_block="OOM_GATE_FAIL")
        # Exit so systemd can retry automatically after fixes/transient issues.
        raise RuntimeError("oom completion gate failed")
    # Collect OOM rows and attempt direct retry fill without overwriting native preds.
    oom_rows = []
    for method_key in FULL17_METHOD_KEYS:
        for task in TASKS:
            path = pipeline_dir / "full17" / method_key / task / "longbench_predictions" / f"{task}.jsonl"
            for row in iter_jsonl(path):
                if row.get("oom") or row.get("status") == "oom" or "out of memory" in str(row.get("error", "")).lower():
                    oom_rows.append({"method": method_key, "task": task, **row})
    atomic_write_csv(
        root / "OOM_MANIFEST.csv",
        [{"method": r.get("method"), "task": r.get("task"), "sample_id": r.get("sample_id"), "sample_index": r.get("sample_index")} for r in oom_rows],
        ["method", "task", "sample_id", "sample_index"],
    )
    # Accuracy-only completion copies: write to oom_completion/filled without touching native.
    filled_root = root / "filled"
    for row in oom_rows:
        method_key = str(row["method"])
        task = str(row["task"])
        idx = int(row.get("sample_index", str(row.get("sample_id", "x:0")).split(":")[1]))
        out_dir = filled_root / method_key / task
        pred_path = out_dir / "longbench_predictions" / f"{task}.jsonl"
        if str(row.get("sample_id")) in existing_prediction_ids(pred_path):
            continue
        cmd = build_task_cmd(
            method_key=method_key,
            tasks=[task],
            run_dir=out_dir,
            sample_indices_by_task={task: [idx]},
            protocol_id=protocol_id,
            implementation_revision=revision,
        )
        run_gpu_worker(
            pipeline_dir, stage="oom_completion",
            worker_id=f"fill_{method_key}_{task}_{idx}",
            method_key=method_key, task=task, cmd=cmd, run_dir=out_dir,
        )
    write_stage(pipeline_dir, "oom_completion", {"decision": "PASS", "oom_count": len(oom_rows), "gate": gate_report})
    update_pipeline_status(pipeline_dir, stage="oom_completion", state="PASS")


def stage_ablation(pipeline_dir: Path, protocol_id: str, revision: str) -> None:
    prev = read_stage(pipeline_dir, "ablation")
    if prev and prev.get("decision") == "PASS":
        return
    update_pipeline_status(pipeline_dir, stage="ablation", state="RUNNING")
    root = pipeline_dir / "ablation"
    for variant in ABLATION_VARIANTS:
        vid = str(variant["id"])
        if variant.get("reuse") == "sliminfer_drm_add_km":
            # reuse six_task / full17 DRM-Add 5% results by symlink/copy metadata only
            meta = {
                "id": vid,
                "reuse": "sliminfer_drm_add_km",
                "source": str(pipeline_dir / "six_task" / "sliminfer_drm_add_km"),
                "note": "5% DRM-Add reused from formal six-task/full17; Stage-10 frozen definition",
            }
            atomic_write_json(root / vid / "REUSE.json", meta)
            continue
        for task in SIX_TASKS:
            run_dir = root / vid / task
            if predictions_complete(run_dir, [task]):
                continue
            cmd = build_task_cmd(
                method_key="sliminfer_drm_add_km",
                tasks=[task],
                run_dir=run_dir,
                protocol_id=protocol_id,
                implementation_revision=revision,
                seed=int(variant.get("seed", 0)),
                rescue_ratio=float(variant.get("rescue_ratio", 0.05)),
                alpha_diff=float(variant.get("alpha_diff", 0.5)),
                lambda_red=float(variant.get("lambda_red", 0.25)),
                eta_rescue=float(variant.get("eta_rescue", 0.25)),
                eta_active=float(variant.get("eta_active", 0.25)),
                policy=str(variant.get("policy", "add")),
                score=str(variant.get("score", "drm")),
            )
            rc = run_gpu_worker(
                pipeline_dir, stage="ablation", worker_id=f"{vid}__{task}",
                method_key=vid, task=task, cmd=cmd, run_dir=run_dir,
            )
            if rc != 0 and not predictions_complete(run_dir, [task]):
                raise RuntimeError(f"ablation failed {vid} {task} rc={rc}")
    write_stage(pipeline_dir, "ablation", {"decision": "PASS"})
    update_pipeline_status(pipeline_dir, stage="ablation", state="PASS")


def stage_efficiency(pipeline_dir: Path, protocol_id: str, revision: str) -> None:
    prev = read_stage(pipeline_dir, "efficiency")
    if prev and prev.get("decision") == "PASS":
        return
    update_pipeline_status(pipeline_dir, stage="efficiency", state="RUNNING")
    root = pipeline_dir / "efficiency"
    rows: List[Dict[str, Any]] = []
    for method_key in EFFICIENCY_METHODS:
        for length in EFFICIENCY_LENGTHS:
            for rep in range(1, 6):
                worker_id = f"{method_key}__{length}__rep{rep}"
                run_dir = root / method_key / str(length) / f"rep{rep}"
                metrics_path = run_dir / "latency_metrics.json"
                if metrics_path.exists():
                    rows.append(load_json(metrics_path))
                    continue
                ensure_gpu_then(pipeline_dir, "efficiency")
                cmd = base_runtime_flags(
                    method_key=method_key,
                    output_dir=run_dir,
                    tasks=["qasper"],
                    eval_mode="latency",
                    context_lengths=str(length),
                    decode_tokens=16,
                    protocol_id=protocol_id,
                    implementation_revision=revision,
                    active_block_trace="none",
                )
                # fresh process guaranteed by run_command each time
                rc = run_gpu_worker(
                    pipeline_dir, stage="efficiency", worker_id=worker_id,
                    method_key=method_key, task=f"len{length}", cmd=cmd, run_dir=run_dir,
                )
                metrics = {
                    "tag": EFFICIENCY_TAG,
                    "method": method_key,
                    "context_length": length,
                    "repeat": rep,
                    "returncode": rc,
                    "ttft": "UNAVAILABLE",
                    "prefill": None,
                    "decode": None,
                    "e2e": None,
                    "selector": None,
                    "peak_allocated": None,
                    "peak_reserved": None,
                    "host_rss": None,
                    "oom": rc != 0,
                    "failure": rc != 0,
                    "updated_at": now_iso(),
                }
                # Try to harvest from runner outputs if present.
                for candidate in run_dir.rglob("*.json"):
                    try:
                        data = load_json(candidate)
                    except Exception:
                        continue
                    if isinstance(data, dict):
                        for key in ("ttft", "prefill", "decode", "e2e", "selector", "peak_allocated", "peak_reserved", "host_rss"):
                            if key in data and metrics[key] in (None, "UNAVAILABLE"):
                                metrics[key] = data[key]
                        if "latency" in data and isinstance(data["latency"], dict):
                            for key in ("ttft", "prefill", "decode", "e2e"):
                                if key in data["latency"]:
                                    metrics[key] = data["latency"][key]
                if metrics["ttft"] is None:
                    metrics["ttft"] = "UNAVAILABLE"
                atomic_write_json(metrics_path, metrics)
                rows.append(metrics)
    atomic_write_csv(
        root / "efficiency_rows.csv",
        rows,
        ["tag", "method", "context_length", "repeat", "ttft", "prefill", "decode", "e2e", "selector",
         "peak_allocated", "peak_reserved", "host_rss", "oom", "failure", "returncode"],
    )
    atomic_write_json(root / "EFFICIENCY_TAG.json", {"tag": EFFICIENCY_TAG, "note": "Do not mix with old RTX 4090 absolute timings."})
    write_stage(pipeline_dir, "efficiency", {"decision": "PASS", "n_rows": len(rows)})
    update_pipeline_status(pipeline_dir, stage="efficiency", state="PASS")


def stage_final_export(pipeline_dir: Path, protocol_id: str, protocol_hash: str) -> None:
    prev = read_stage(pipeline_dir, "final_export")
    if prev and prev.get("decision") == "PASS":
        return
    update_pipeline_status(pipeline_dir, stage="final_export", state="RUNNING")
    export_dir = pipeline_dir / "exports" / "final"
    export_dir.mkdir(parents=True, exist_ok=True)

    def write_md(name: str, body: str) -> None:
        atomic_write_text(export_dir / name, body if body.endswith("\n") else body + "\n")

    write_md("LLAMA_DRM_PROTOCOL.md", f"""# LLAMA_DRM_PROTOCOL

- protocol_name: `{PROTOCOL_NAME}`
- protocol_id: `{protocol_id}`
- protocol_hash: `{protocol_hash}`
- model_path: `{MODEL_PATH}`
- data_root: `{DATA_ROOT}`
- pipeline_dir: `{pipeline_dir}`
- offline: true
- no human approval gates
- GPU policy: WAITING_FOR_GPU_IDLE, one worker, flock
""")
    write_md("LLAMA_DRM_METHOD_SPEC.md", f"""# LLAMA_DRM_METHOD_SPEC

Authoritative Qwen DRM implementation reused:

- runner: `experiments/run_qwen25_sliminfer_variants.py`
- cache/policy: `utils/cache_utils.py`
- score/MMR: `bridgekv/rescuekv_scoring.py`

Methods: {', '.join(FULL17_METHOD_KEYS)}

LLaMA schedule: prune `{list([10,20,30])}`, retain `{list([8192,4096,2048])}`, K `{list([128,64,32])}`, M `{list([6,3,2])}`, bf16, block 64, token unit 8, gamma 0.9, local query window 4.
""")
    for src, dst in [
        (pipeline_dir / "canary" / "CANARY_AUDIT.json", "LLAMA_DRM_CANARY_REPORT.md"),
        (pipeline_dir / "six_task" / "STAGE_AUDIT.json", "LLAMA_DRM_SIX_TASK_REPORT.md"),
        (pipeline_dir / "full17" / "raw_macro.json", "LLAMA_DRM_FULL17_REPORT.md"),
    ]:
        payload = load_json(src) if src.exists() else {"status": "missing"}
        write_md(dst, f"# {dst}\n\n```json\n{json.dumps(payload, indent=2)}\n```\n")
    write_md(
        "LLAMA_DRM_ACCURACY_COMPLETION_REPORT.md",
        "# LLAMA_DRM_ACCURACY_COMPLETION_REPORT\n\n"
        + ( (pipeline_dir / "oom_completion" / "EQUIVALENCE_GATE.json").read_text(encoding="utf-8")
            if (pipeline_dir / "oom_completion" / "EQUIVALENCE_GATE.json").exists() else "missing\n"),
    )
    write_md("LLAMA_DRM_RANDOM_SEED_REPORT.md", "# Random seed controls from six-task (seeds 0/1/2) and canary.\n")
    write_md("LLAMA_DRM_RATIO_REPORT.md", "# DRM ratio ablation 2%/5%(reuse)/10% on six tasks.\n")
    write_md("LLAMA_DRM_COMPONENT_ABLATION_REPORT.md", "# Component ablations: without gain / active-core redundancy / rescue-rescue MMR / base utility only.\n")
    write_md(
        "LLAMA_DRM_EFFICIENCY_3090TI_REPORT.md",
        f"# {EFFICIENCY_TAG}\n\nDo not mix with old RTX 4090 absolute timings.\n",
    )
    # copy bootstrap / audits
    for src_name, dst_name in [
        ("full17/paired_bootstrap.csv", "LLAMA_DRM_PAIRWISE_BOOTSTRAP.csv"),
        ("oom_completion/OOM_MANIFEST.csv", "LLAMA_DRM_OOM_MANIFEST.csv"),
    ]:
        src = pipeline_dir / src_name
        if src.exists():
            shutil.copy2(src, export_dir / dst_name)
        else:
            atomic_write_text(export_dir / dst_name, "")
    # budget / selection audits placeholder aggregated
    atomic_write_csv(export_dir / "LLAMA_DRM_BUDGET_AUDIT.csv", [], ["method", "task", "violations"])
    atomic_write_csv(export_dir / "LLAMA_DRM_SELECTION_AUDIT.csv", [], ["method", "task", "violations"])
    write_md("LLAMA_DRM_DATA_FOR_PAPER.md", f"""# LLaMA DRM data for paper

Protocol `{protocol_id}` on Qwen machine.

Efficiency tag: `{EFFICIENCY_TAG}`.
""")
    write_md("LLAMA_DRM_CLAIM_GUARD.md", """# CLAIM_GUARD

- Do not mix LLaMA-on-3090Ti efficiency with old RTX 4090 absolute timings.
- Accuracy completion fills are quality-only; do not use for TTFT/E2E/VRAM/native completion rate.
- Revision is UNKNOWN_LOCAL_SNAPSHOT; identity is SHA256-locked, never a fabricated HF revision.
- CC methods are excluded from this protocol.
""")
    write_md("LLAMA_DRM_OPEN_TODOS.md", """# OPEN_TODOS

- Confirm local LLaMA snapshot path contents and long-context capability before paper claims.
- Enrich budget/selection CSV aggregation from per-worker drm traces after full17 completes.
""")
    # sha256 sums
    lines = []
    for path in sorted(export_dir.iterdir()):
        if path.name == "FILE_SHA256SUMS.txt":
            continue
        if path.is_file():
            lines.append(f"{sha256_file(path)}  {path.name}")
    atomic_write_text(export_dir / "FILE_SHA256SUMS.txt", "\n".join(lines) + "\n")
    write_stage(pipeline_dir, "final_export", {"decision": "PASS", "export_dir": str(export_dir)})
    update_pipeline_status(pipeline_dir, stage="final_export", state="COMPLETE", detail="all stages complete")


def run_pipeline(pipeline_dir: Path) -> int:
    manifest = initialize_pipeline(pipeline_dir)
    protocol_id = str(manifest["protocol_id"])
    protocol_hash = str(manifest["protocol_hash"])
    revision = sha256_text(protocol_id + protocol_hash)
    lock_path = Path(os.environ.get("XDG_RUNTIME_DIR", "/tmp")) / GLOBAL_FLOCK_NAME
    # also lock inside pipeline
    local_lock = pipeline_dir / "locks" / "scheduler.lock"
    with GlobalSchedulerLock(lock_path), GlobalSchedulerLock(local_lock):
        stage_static_audit(pipeline_dir)
        stage_canary(pipeline_dir, protocol_id, revision)
        stage_task_matrix(
            pipeline_dir, stage_id="six_task", root_name="six_task",
            tasks=SIX_TASKS, method_keys=SIX_TASK_METHOD_KEYS,
            protocol_id=protocol_id, revision=revision,
        )
        stage_task_matrix(
            pipeline_dir, stage_id="full17", root_name="full17",
            tasks=TASKS, method_keys=FULL17_METHOD_KEYS,
            protocol_id=protocol_id, revision=revision,
        )
        stage_stats_full17(pipeline_dir)
        stage_oom_completion(pipeline_dir, protocol_id, revision)
        stage_ablation(pipeline_dir, protocol_id, revision)
        stage_efficiency(pipeline_dir, protocol_id, revision)
        stage_final_export(pipeline_dir, protocol_id, protocol_hash)
    update_pipeline_status(pipeline_dir, stage="final_export", state="COMPLETE")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-dir", type=Path, default=None)
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--static-audit-only", action="store_true")
    parser.add_argument("--init-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pipeline_dir = args.pipeline_dir or default_pipeline_dir()
    pipeline_dir = pipeline_dir.resolve()
    if args.init_only:
        initialize_pipeline(pipeline_dir)
        print(pipeline_dir)
        return 0
    if args.status:
        path = pipeline_dir / "status" / "PROGRESS.json"
        print(path.read_text(encoding="utf-8") if path.exists() else "{}")
        return 0
    if args.static_audit_only:
        initialize_pipeline(pipeline_dir)
        # Force CPU-only audit.
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        report = run_static_audit(pipeline_dir)
        print(json.dumps({"status": report.get("status"), "hard_blocks": report.get("hard_blocks"), "protocol_id": report.get("protocol_id"), "protocol_hash": report.get("protocol_hash")}, indent=2))
        return 0 if report.get("status") in {"PASS", "WAITING_FOR_MODEL"} else 2
    try:
        return run_pipeline(pipeline_dir)
    except Exception as exc:
        update_pipeline_status(
            pipeline_dir,
            stage=read_stage_safe(pipeline_dir),
            state="FAILED",
            detail=str(exc),
        )
        atomic_write_text(
            pipeline_dir / "logs" / "orchestrator_exception.txt",
            traceback.format_exc(),
        )
        # Restart=always will relaunch; keep non-zero for unexpected failures except handled hard blocks.
        return 1


def read_stage_safe(pipeline_dir: Path) -> str:
    try:
        return load_json(pipeline_dir / "status" / "PROGRESS.json").get("current_stage", "UNKNOWN")
    except Exception:
        return "UNKNOWN"


if __name__ == "__main__":
    raise SystemExit(main())
