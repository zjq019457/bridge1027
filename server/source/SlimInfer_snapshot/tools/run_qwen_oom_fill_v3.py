#!/usr/bin/env python3
"""Retry only Stage-11 residual OOM rows with allocator-safe exact offload.

V3 is independent of the canonical Stage-11 decision. It hash-locks the raw
predictions, reuses the immutable V2 direct canaries as references, and writes
all candidate predictions and scores to a new result directory.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

_REPO_BOOTSTRAP = Path(__file__).resolve().parents[1]
if str(_REPO_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(_REPO_BOOTSTRAP))

from tools import run_qwen_oom_fill_v2 as v2
from tools.qwen_pipeline_common import (
    PYTHON,
    REPO_ROOT,
    atomic_write_csv,
    atomic_write_json,
    atomic_write_text,
    load_json,
    now_iso,
    prediction_status,
    sha256_file,
)
from tools.qwen_protocol_gate import run_protocol_gate
from tools.qwen_resource_guard import monitor_resources, write_blocker


ALLOCATOR_PROFILE_ID = "expandable_segments_v1"
OFFLOAD_PROFILE = v2.FALLBACK_PROFILE
ALLOCATOR_ENV = {
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
}


def _remaining_rows(v2_root: Path) -> List[Dict[str, Any]]:
    rows = v2._read_csv(v2_root / "OOM_FILL_MANIFEST.csv")
    remaining = [
        {
            "method": str(row["method"]),
            "task": str(row["task"]),
            "sample_index": int(row["sample_index"]),
            "sample_id": str(row["sample_id"]),
            "raw_status": str(row["raw_status"]),
            "raw_error": str(row["raw_error"]),
        }
        for row in rows
        if row.get("fill_status") == "unfilled_oom"
    ]
    remaining.sort(key=lambda row: (
        row["method"], row["task"], int(row["sample_index"])
    ))
    return remaining


def _load_v2_fills(v2_root: Path) -> Dict[Tuple[str, str, int], Dict[str, Any]]:
    fills: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    for row in v2._read_csv(v2_root / "OOM_FILL_MANIFEST.csv"):
        if row.get("fill_status") != "ok":
            continue
        key = (str(row["method"]), str(row["task"]), int(row["sample_index"]))
        path = Path(str(row["fill_prediction_path"]))
        predictions = v2._read_jsonl(path)
        if len(predictions) != 1 or prediction_status(predictions[0]) != "ok":
            raise RuntimeError(f"Invalid V2 fill row for {key}: {path}")
        if sha256_file(path) != row["fill_prediction_sha256"]:
            raise RuntimeError(f"V2 fill hash mismatch for {key}: {path}")
        fills[key] = {
            "method": key[0],
            "task": key[1],
            "sample_index": key[2],
            "prediction_status": "ok",
            "prediction": predictions[0],
            "prediction_path": str(path),
            "prediction_sha256": row["fill_prediction_sha256"],
            "offload_profile": row["offload_profile"],
            "fill_source": "v2",
        }
    return fills


def _verify_reference_case(root: Path, task: str) -> Dict[str, Any]:
    decision_path = root / "CASE_DECISION.json"
    if not decision_path.exists():
        raise RuntimeError(f"V2 direct canary decision is absent: {decision_path}")
    decision = load_json(decision_path)
    prediction = v2._prediction_path(root, task)
    trace = v2._trace_path(root, task)
    prediction_ok = (
        decision.get("decision") == "PASS"
        and prediction.exists()
        and sha256_file(prediction) == decision.get("prediction_sha256")
    )
    recorded_trace = decision.get("trace_sha256")
    trace_ok = recorded_trace is None or (
        trace.exists() and sha256_file(trace) == recorded_trace
    )
    if not prediction_ok or not trace_ok:
        raise RuntimeError(
            f"V2 direct canary integrity failure: {root}; "
            f"prediction_ok={prediction_ok}, trace_ok={trace_ok}"
        )
    return {
        "root": str(root),
        "prediction_sha256": decision.get("prediction_sha256"),
        "trace_sha256": recorded_trace,
    }


def _run_equivalence_gate(
    *,
    pipeline_dir: Path,
    fill_root: Path,
    v2_root: Path,
    method_hashes: Mapping[str, str],
    timeout_seconds: int,
    environment: Mapping[str, Any],
    safety_policy: Mapping[str, Any],
) -> Dict[str, Any]:
    cases = []
    reference_manifest = []
    for position, (method, task, sample_index) in enumerate(
        v2.CANARY_CASES, start=1
    ):
        case_id = v2._case_id(method, task, sample_index)
        reference_root = v2_root / "equivalence_gate/direct" / case_id
        reference_manifest.append(_verify_reference_case(reference_root, task))
        candidate_root = (
            fill_root / "equivalence_gate" / ALLOCATOR_PROFILE_ID / case_id
        )
        v2._run_isolated_case(
            pipeline_dir=pipeline_dir,
            fill_root=fill_root,
            method=method,
            task=task,
            sample_index=sample_index,
            loader_mode="offload",
            profile=OFFLOAD_PROFILE,
            output_root=candidate_root,
            method_hashes=method_hashes,
            timeout_seconds=timeout_seconds,
            environment=environment,
            safety_policy=safety_policy,
        )
        comparison = v2._compare_canary_case(
            method, task, sample_index, reference_root, candidate_root
        )
        comparison.update({
            "case": position,
            "case_id": case_id,
            "allocator_profile": ALLOCATOR_PROFILE_ID,
            "offload_profile": OFFLOAD_PROFILE,
            "reference_source": "immutable_v2_direct_canary",
        })
        atomic_write_json(
            candidate_root / "EQUIVALENCE_DECISION.json", comparison
        )
        cases.append(comparison)
        atomic_write_json(fill_root / "PROGRESS.json", {
            "created_at": now_iso(),
            "phase": "allocator_equivalence_gate",
            "completed_cases": position,
            "total_cases": len(v2.CANARY_CASES),
            "latest": comparison,
        })
        if comparison["decision"] != "PASS":
            break
    report = {
        "created_at": now_iso(),
        "decision": (
            "PASS"
            if len(cases) == len(v2.CANARY_CASES)
            and all(row["decision"] == "PASS" for row in cases)
            else "FAIL"
        ),
        "allocator_profile": ALLOCATOR_PROFILE_ID,
        "allocator_environment": dict(ALLOCATOR_ENV),
        "offload_profile": OFFLOAD_PROFILE,
        "required_cases": len(v2.CANARY_CASES),
        "completed_cases": len(cases),
        "prediction_matches": sum(
            bool(row.get("prediction_equal")) for row in cases
        ),
        "selection_matches": sum(
            bool(row.get("selection_equal")) for row in cases
        ),
        "reference_manifest": reference_manifest,
        "cases": cases,
    }
    atomic_write_json(
        fill_root / "equivalence_gate/ALLOCATOR_EQUIVALENCE_DECISION.json",
        report,
    )
    if report["decision"] != "PASS":
        raise RuntimeError("Allocator equivalence gate failed")
    return report


def _write_manifest(
    fill_root: Path,
    remaining: Sequence[Mapping[str, Any]],
    fills: Mapping[Tuple[str, str, int], Mapping[str, Any]],
) -> None:
    rows = []
    for row in remaining:
        key = (
            str(row["method"]),
            str(row["task"]),
            int(row["sample_index"]),
        )
        fill = fills.get(key)
        rows.append({
            **row,
            "fill_status": (
                fill.get("prediction_status") if fill else "unfilled_oom"
            ),
            "allocator_profile": (
                fill.get("allocator_profile") if fill else ALLOCATOR_PROFILE_ID
            ),
            "offload_profile": (
                fill.get("offload_profile") if fill else OFFLOAD_PROFILE
            ),
            "fill_prediction_path": (
                fill.get("prediction_path") if fill else None
            ),
            "fill_prediction_sha256": (
                fill.get("prediction_sha256") if fill else None
            ),
            "elapsed_seconds": fill.get("elapsed_seconds") if fill else None,
        })
    atomic_write_csv(
        fill_root / "OOM_FILL_V3_MANIFEST.csv",
        rows,
        [
            "method", "task", "sample_index", "sample_id", "raw_status",
            "raw_error", "fill_status", "allocator_profile",
            "offload_profile", "fill_prediction_path",
            "fill_prediction_sha256", "elapsed_seconds",
        ],
    )


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
    v2_pointer = pipeline_dir / "QWEN_OOM_FILL_ROOT_V2.json"
    if not v2_pointer.exists():
        raise RuntimeError("V2 OOM fill pointer is absent")
    v2_root = Path(load_json(v2_pointer)["path"])
    v2_decision = load_json(v2_root / "FINAL_OOM_FILL_DECISION.json")
    if v2_decision.get("decision") not in {"PASS", "SOFT_BLOCKED"}:
        raise RuntimeError(f"V2 result is not accepted: {v2_decision}")

    pointer = pipeline_dir / "QWEN_OOM_FILL_ROOT_V3.json"
    if pointer.exists():
        fill_root = Path(load_json(pointer)["path"])
    else:
        fill_root = (
            REPO_ROOT / "results"
            / f"qwen_oom_offload_fill_v3_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        atomic_write_json(pointer, {"created_at": now_iso(), "path": str(fill_root)})

    sources = v2._collect_sources(pipeline_dir)
    raw_lock = v2._raw_hash_lock(sources)
    v2_raw_lock = load_json(v2_root / "RAW_UNCHANGED_HASH_LOCK.json")["files"]
    if raw_lock != v2_raw_lock:
        raise RuntimeError("Current raw hash lock differs from V2")
    raw_failures = v2._verify_raw_hash_lock(raw_lock)
    if raw_failures:
        raise RuntimeError(f"Raw prediction hash mismatch: {raw_failures[:3]}")
    remaining = _remaining_rows(v2_root)
    method_hashes = v2._method_hashes()
    policy = load_json(safety_policy_path)
    candidate_environment = {
        **policy["environment"],
        **ALLOCATOR_ENV,
    }
    plan = {
        "created_at": now_iso(),
        "decision": "NOT_RUN",
        "execute_requested": execute,
        "fill_root": str(fill_root),
        "v2_root": str(v2_root),
        "remaining_oom_rows": len(remaining),
        "unique_samples": len({
            (row["task"], int(row["sample_index"])) for row in remaining
        }),
        "remaining_rows": remaining,
        "allocator_profile": ALLOCATOR_PROFILE_ID,
        "allocator_environment": dict(ALLOCATOR_ENV),
        "offload_profile": OFFLOAD_PROFILE,
        "reference_canaries": "immutable V2 direct canaries",
        "equivalence_cases": len(v2.CANARY_CASES),
        "raw_predictions_immutable": True,
        "adapter": str(v2.ADAPTER),
        "adapter_sha256": sha256_file(v2.ADAPTER),
        "runner": str(Path(__file__).resolve()),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "timeout_seconds": timeout_seconds,
    }
    atomic_write_json(pipeline_dir / "STAGE_11_OOM_FILL_PLAN_V3.json", plan)
    if not execute:
        return plan

    fill_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(fill_root / "PLAN.json", plan)
    atomic_write_json(fill_root / "RAW_UNCHANGED_HASH_LOCK.json", {
        "created_at": now_iso(),
        "raw_predictions_immutable": True,
        "files": raw_lock,
    })
    atomic_write_csv(
        fill_root / "V2_REMAINING_OOM_MANIFEST.csv",
        remaining,
        [
            "method", "task", "sample_index", "sample_id",
            "raw_status", "raw_error",
        ],
    )
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
        raise RuntimeError("V3 resource gate failed")

    equivalence = _run_equivalence_gate(
        pipeline_dir=pipeline_dir,
        fill_root=fill_root,
        v2_root=v2_root,
        method_hashes=method_hashes,
        timeout_seconds=timeout_seconds,
        environment=candidate_environment,
        safety_policy=policy,
    )
    new_fills: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    unresolved = []
    for position, row in enumerate(remaining, start=1):
        key = (
            str(row["method"]),
            str(row["task"]),
            int(row["sample_index"]),
        )
        output_root = (
            fill_root / "fills" / v2._case_id(*key) / ALLOCATOR_PROFILE_ID
        )
        result = v2._run_isolated_case(
            pipeline_dir=pipeline_dir,
            fill_root=fill_root,
            method=key[0],
            task=key[1],
            sample_index=key[2],
            loader_mode="offload",
            profile=OFFLOAD_PROFILE,
            output_root=output_root,
            method_hashes=method_hashes,
            timeout_seconds=timeout_seconds,
            environment=candidate_environment,
            safety_policy=policy,
        )
        prediction = v2._read_jsonl(
            v2._prediction_path(output_root, key[1])
        )[0]
        fill = {
            **result,
            "prediction": prediction,
            "allocator_profile": ALLOCATOR_PROFILE_ID,
            "fill_source": "v3",
        }
        if result["prediction_status"] == "ok":
            new_fills[key] = fill
        elif result["prediction_status"] == "oom":
            unresolved.append(row)
        else:
            raise RuntimeError(
                f"Invalid V3 status for {key}: {result['prediction_status']}"
            )
        _write_manifest(fill_root, remaining, new_fills)
        atomic_write_json(fill_root / "PROGRESS.json", {
            "created_at": now_iso(),
            "phase": "allocator_oom_fill",
            "completed_cases": position,
            "total_cases": len(remaining),
            "new_fills": len(new_fills),
            "remaining": len(unresolved) + len(remaining) - position,
            "latest": {
                name: fill.get(name)
                for name in (
                    "method", "task", "sample_index",
                    "prediction_status", "elapsed_seconds",
                )
            },
        })
        raw_failures = v2._verify_raw_hash_lock(raw_lock)
        if raw_failures:
            raise RuntimeError(
                f"Raw prediction changed during V3: {raw_failures[:3]}"
            )

    combined_fills = _load_v2_fills(v2_root)
    combined_fills.update(new_fills)
    scoring = v2._score_offload_completed(
        fill_root=fill_root,
        sources=sources,
        fills=combined_fills,
    )
    _write_manifest(fill_root, remaining, new_fills)
    raw_failures = v2._verify_raw_hash_lock(raw_lock)
    decision = {
        "created_at": now_iso(),
        "name": "oom_offload_fill_v3",
        "decision": (
            "PASS"
            if len(new_fills) == len(remaining) and not raw_failures
            else "SOFT_BLOCKED"
        ),
        "run_root": str(fill_root),
        "v2_remaining_oom": len(remaining),
        "new_fills": len(new_fills),
        "remaining_oom": len(remaining) - len(new_fills),
        "remaining_rows": unresolved,
        "allocator_equivalence": {
            "decision": equivalence["decision"],
            "completed_cases": equivalence["completed_cases"],
            "prediction_matches": equivalence["prediction_matches"],
            "selection_matches": equivalence["selection_matches"],
        },
        "raw_predictions_modified": bool(raw_failures),
        "raw_hash_failures": raw_failures,
        "scoring": scoring,
    }
    atomic_write_json(fill_root / "FINAL_OOM_FILL_V3_DECISION.json", decision)
    atomic_write_json(pipeline_dir / "QWEN_OOM_FILL_V3_DECISION.json", decision)
    atomic_write_text(
        pipeline_dir / "STAGE_11_V3_RESUME_COMMAND.txt",
        f"{PYTHON} {Path(__file__).resolve()} --pipeline-dir {pipeline_dir} "
        f"--safety-policy {safety_policy_path} --execute "
        f"--timeout-seconds {timeout_seconds}\n",
    )
    atomic_write_json(pipeline_dir / "PID.json", {
        "updated_at": now_iso(),
        "pipeline_pid": None,
        "gpu_worker_pid": None,
        "current_stage": None,
        "current_method": None,
        "current_task": None,
    })
    return decision


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-dir", type=Path, required=True)
    parser.add_argument("--safety-policy", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    args = parser.parse_args()
    pipeline_dir = args.pipeline_dir.resolve()
    try:
        result = build_and_run(
            pipeline_dir,
            args.safety_policy.resolve(),
            execute=args.execute,
            timeout_seconds=args.timeout_seconds,
        )
    except Exception as exc:
        failure = {
            "created_at": now_iso(),
            "name": "oom_offload_fill_v3",
            "decision": "HARD_BLOCKED",
            "reason": "stage_exception",
            "error": repr(exc),
        }
        atomic_write_json(pipeline_dir / "QWEN_OOM_FILL_V3_FAILURE.json", failure)
        print(json.dumps(failure, indent=2))
        return 2
    print(json.dumps(result, indent=2))
    return 0 if result.get("decision") in {
        "PASS", "SOFT_BLOCKED", "NOT_RUN"
    } else 2


if __name__ == "__main__":
    raise SystemExit(main())
