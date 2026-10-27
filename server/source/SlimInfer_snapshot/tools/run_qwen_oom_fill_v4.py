#!/usr/bin/env python3
"""Retry V3 residual full-KV OOM rows with gated MLP sequence chunking."""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

_REPO_BOOTSTRAP = Path(__file__).resolve().parents[1]
if str(_REPO_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(_REPO_BOOTSTRAP))

from tools import run_qwen_oom_fill_v2 as v2
from tools import run_qwen_oom_fill_v3 as v3
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


ADAPTER = REPO_ROOT / "tools/qwen_whole_model_offload_chunked_mlp_adapter.py"
PROFILE_ID = "fullkv_mlp_chunk_8192_v1"
OFFLOAD_PROFILE = v2.FALLBACK_PROFILE
ALLOCATOR_ENV = {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
REFERENCE_CASES = (
    ("full_kv_fa2", "hotpotqa", 63, "v2"),
    ("full_kv_fa2", "repobench-p", 53, "v2"),
    ("full_kv_fa2", "gov_report", 46, "fresh"),
)
IMPLEMENTATION_FILES = (
    REPO_ROOT / "tools/run_qwen_oom_fill_v4.py",
    REPO_ROOT / "tools/qwen_whole_model_offload_chunked_mlp_adapter.py",
    REPO_ROOT / "tools/audit_qwen_oom_fill_v3.py",
    REPO_ROOT / "tests/test_qwen_oom_fill_v4.py",
    REPO_ROOT / "tests/test_qwen_chunked_mlp_adapter.py",
    REPO_ROOT / "tests/test_audit_qwen_oom_fill_v3.py",
)


def _remaining_rows(v3_root: Path) -> List[Dict[str, Any]]:
    rows = v2._read_csv(v3_root / "OOM_FILL_V3_MANIFEST.csv")
    remaining = [
        {
            "method": str(row["method"]),
            "task": str(row["task"]),
            "sample_index": int(row["sample_index"]),
            "sample_id": str(row["sample_id"]),
            "v3_status": str(row["fill_status"]),
        }
        for row in rows
        if row.get("fill_status") != "ok"
    ]
    remaining.sort(key=lambda row: (
        row["method"], row["task"], row["sample_index"]
    ))
    return remaining


def _load_manifest_fills(root: Path, manifest_name: str) -> Dict[Tuple[str, str, int], Dict[str, Any]]:
    fills: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    for row in v2._read_csv(root / manifest_name):
        if row.get("fill_status") != "ok":
            continue
        key = (str(row["method"]), str(row["task"]), int(row["sample_index"]))
        path = Path(row["fill_prediction_path"])
        predictions = v2._read_jsonl(path)
        expected_hash = row["fill_prediction_sha256"]
        if (
            len(predictions) != 1
            or prediction_status(predictions[0]) != "ok"
            or sha256_file(path) != expected_hash
        ):
            raise RuntimeError(f"Invalid prior fill for {key}: {path}")
        fills[key] = {
            "method": key[0],
            "task": key[1],
            "sample_index": key[2],
            "prediction_status": "ok",
            "prediction": predictions[0],
            "prediction_path": str(path),
            "prediction_sha256": expected_hash,
            "fill_source": manifest_name,
        }
    return fills


def _write_manifest(
    root: Path,
    remaining: Sequence[Mapping[str, Any]],
    fills: Mapping[Tuple[str, str, int], Mapping[str, Any]],
) -> None:
    rows = []
    for row in remaining:
        key = (str(row["method"]), str(row["task"]), int(row["sample_index"]))
        fill = fills.get(key)
        rows.append({
            **row,
            "fill_status": fill.get("prediction_status") if fill else "unfilled_oom",
            "profile": PROFILE_ID,
            "offload_profile": OFFLOAD_PROFILE,
            "fill_prediction_path": fill.get("prediction_path") if fill else None,
            "fill_prediction_sha256": fill.get("prediction_sha256") if fill else None,
            "elapsed_seconds": fill.get("elapsed_seconds") if fill else None,
        })
    atomic_write_csv(
        root / "OOM_FILL_V4_MANIFEST.csv",
        rows,
        [
            "method", "task", "sample_index", "sample_id", "v3_status",
            "fill_status", "profile", "offload_profile",
            "fill_prediction_path", "fill_prediction_sha256",
            "elapsed_seconds",
        ],
    )


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def _freeze_implementation(root: Path) -> Dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    status = _git("status", "--short")
    if status.returncode != 0:
        raise RuntimeError(f"git status failed: {status.stderr}")
    patch_parts = [_git("diff", "--binary").stdout]
    for path in IMPLEMENTATION_FILES:
        tracked = _git("ls-files", "--error-unmatch", str(path.relative_to(REPO_ROOT)))
        if tracked.returncode == 0:
            continue
        addition = _git(
            "diff", "--no-index", "--binary", "--", "/dev/null",
            str(path.relative_to(REPO_ROOT)),
        )
        if addition.returncode not in {0, 1}:
            raise RuntimeError(f"git no-index diff failed for {path}: {addition.stderr}")
        patch_parts.append(addition.stdout)
    patch = "".join(patch_parts)
    atomic_write_text(root / "IMPLEMENTATION_PATCH.diff", patch)
    atomic_write_text(root / "GIT_STATUS_SHORT.txt", status.stdout)
    snapshot_root = root / "source_snapshot"
    snapshot_root.mkdir(parents=True, exist_ok=True)
    file_hashes = {}
    for path in IMPLEMENTATION_FILES:
        file_hashes[str(path.relative_to(REPO_ROOT))] = sha256_file(path)
        shutil.copy2(path, snapshot_root / path.name)
    commit = _git("rev-parse", "HEAD")
    if commit.returncode != 0:
        raise RuntimeError(f"git rev-parse failed: {commit.stderr}")
    revision = {
        "created_at": now_iso(),
        "base_commit": commit.stdout.strip(),
        "dirty": bool(status.stdout.strip()),
        "git_status_short": status.stdout.splitlines(),
        "implementation_patch": str(root / "IMPLEMENTATION_PATCH.diff"),
        "implementation_patch_sha256": sha256_file(
            root / "IMPLEMENTATION_PATCH.diff"
        ),
        "source_snapshot": str(snapshot_root),
        "file_sha256": file_hashes,
    }
    atomic_write_json(root / "IMPLEMENTATION_REVISION.json", revision)
    return revision


def _run_gate(
    *,
    pipeline_dir: Path,
    root: Path,
    v2_root: Path,
    method_hashes: Mapping[str, str],
    timeout_seconds: int,
    environment: Mapping[str, Any],
    safety_policy: Mapping[str, Any],
) -> Dict[str, Any]:
    original_adapter = v2.ADAPTER
    v2.ADAPTER = ADAPTER
    cases = []
    try:
        for position, (method, task, sample_index, source) in enumerate(
            REFERENCE_CASES, start=1
        ):
            case_id = v2._case_id(method, task, sample_index)
            if source == "v2":
                direct_root = v2_root / "equivalence_gate/direct" / case_id
                v3._verify_reference_case(direct_root, task)
            else:
                direct_root = root / "equivalence_gate/direct" / case_id
                v2._run_isolated_case(
                    pipeline_dir=pipeline_dir,
                    fill_root=root,
                    method=method,
                    task=task,
                    sample_index=sample_index,
                    loader_mode="direct",
                    profile=OFFLOAD_PROFILE,
                    output_root=direct_root,
                    method_hashes=method_hashes,
                    timeout_seconds=timeout_seconds,
                    environment=environment,
                    safety_policy=safety_policy,
                )
            candidate_root = root / "equivalence_gate/chunked" / case_id
            v2._run_isolated_case(
                pipeline_dir=pipeline_dir,
                fill_root=root,
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
                method, task, sample_index, direct_root, candidate_root
            )
            comparison.update({
                "case": position,
                "case_id": case_id,
                "reference_source": source,
                "profile": PROFILE_ID,
            })
            atomic_write_json(
                candidate_root / "EQUIVALENCE_DECISION.json", comparison
            )
            cases.append(comparison)
            atomic_write_json(root / "PROGRESS.json", {
                "created_at": now_iso(),
                "phase": "fullkv_mlp_chunk_equivalence_gate",
                "completed_cases": position,
                "total_cases": len(REFERENCE_CASES),
                "latest": comparison,
            })
            if comparison["decision"] != "PASS":
                break
    finally:
        v2.ADAPTER = original_adapter
    report = {
        "created_at": now_iso(),
        "decision": (
            "PASS"
            if len(cases) == len(REFERENCE_CASES)
            and all(case["decision"] == "PASS" for case in cases)
            else "FAIL"
        ),
        "profile": PROFILE_ID,
        "adapter": str(ADAPTER),
        "adapter_sha256": sha256_file(ADAPTER),
        "required_cases": len(REFERENCE_CASES),
        "completed_cases": len(cases),
        "prediction_matches": sum(
            bool(case.get("prediction_equal")) for case in cases
        ),
        "selection_matches": sum(
            bool(case.get("selection_equal")) for case in cases
        ),
        "cases": cases,
    }
    atomic_write_json(
        root / "equivalence_gate/MLP_CHUNK_EQUIVALENCE_DECISION.json",
        report,
    )
    if report["decision"] != "PASS":
        raise RuntimeError("Full-KV MLP chunk equivalence gate failed")
    return report


def build_and_run(
    pipeline_dir: Path,
    safety_policy_path: Path,
    *,
    execute: bool,
    timeout_seconds: int,
) -> Dict[str, Any]:
    protocol = run_protocol_gate(pipeline_dir, 11)
    if protocol["decision"] != "PASS":
        raise RuntimeError("Stage 11 protocol gate failed")
    v2_root = Path(load_json(pipeline_dir / "QWEN_OOM_FILL_ROOT_V2.json")["path"])
    v3_root = Path(load_json(pipeline_dir / "QWEN_OOM_FILL_ROOT_V3.json")["path"])
    v3_final_path = v3_root / "FINAL_OOM_FILL_V3_DECISION.json"
    if execute and not v3_final_path.exists():
        raise RuntimeError("V3 must finish before V4 execution")
    if v3_final_path.exists():
        v3_decision = load_json(v3_final_path)
        if v3_decision.get("decision") not in {"PASS", "SOFT_BLOCKED"}:
            raise RuntimeError(f"V3 result is not accepted: {v3_decision}")

    pointer = pipeline_dir / "QWEN_OOM_FILL_ROOT_V4.json"
    if pointer.exists():
        root = Path(load_json(pointer)["path"])
    else:
        root = (
            REPO_ROOT / "results"
            / f"qwen_oom_offload_fill_v4_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        atomic_write_json(pointer, {"created_at": now_iso(), "path": str(root)})

    sources = v2._collect_sources(pipeline_dir)
    raw_lock = v2._raw_hash_lock(sources)
    v3_raw_lock = load_json(v3_root / "RAW_UNCHANGED_HASH_LOCK.json")["files"]
    if raw_lock != v3_raw_lock:
        raise RuntimeError("Current raw hash lock differs from V3")
    raw_failures = v2._verify_raw_hash_lock(raw_lock)
    if raw_failures:
        raise RuntimeError(f"Raw prediction hash mismatch: {raw_failures[:3]}")
    remaining = _remaining_rows(v3_root)
    disallowed = [row for row in remaining if row["method"] != "full_kv_fa2"]
    if disallowed:
        raise RuntimeError(
            f"V4 only supports residual full-KV rows: {disallowed[:3]}"
        )
    policy = load_json(safety_policy_path)
    environment = {**policy["environment"], **ALLOCATOR_ENV}
    plan = {
        "created_at": now_iso(),
        "decision": "NOT_RUN",
        "execute_requested": execute,
        "root": str(root),
        "v3_root": str(v3_root),
        "remaining_rows": remaining,
        "profile": PROFILE_ID,
        "adapter": str(ADAPTER),
        "adapter_sha256": sha256_file(ADAPTER),
        "runner": str(Path(__file__).resolve()),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "allocator_environment": ALLOCATOR_ENV,
        "mlp_chunk_tokens": 8192,
        "raw_predictions_immutable": True,
        "equivalence_cases": list(REFERENCE_CASES),
        "timeout_seconds": timeout_seconds,
    }
    atomic_write_json(pipeline_dir / "STAGE_11_OOM_FILL_PLAN_V4.json", plan)
    if not execute:
        return plan

    root.mkdir(parents=True, exist_ok=True)
    implementation = _freeze_implementation(root)
    atomic_write_json(root / "PLAN.json", plan)
    atomic_write_json(root / "FROZEN_IMPLEMENTATION.json", implementation)
    atomic_write_json(root / "RAW_UNCHANGED_HASH_LOCK.json", {
        "created_at": now_iso(),
        "raw_predictions_immutable": True,
        "files": raw_lock,
    })
    resource_gate = monitor_resources(
        policy,
        int(policy["thresholds"]["resource_gate_duration_seconds"]),
        int(policy["thresholds"]["resource_gate_interval_seconds"]),
        pipeline_dir=pipeline_dir,
        allowed_pids=(os.getpid(),),
    )
    atomic_write_json(root / "RESOURCE_GATE.json", resource_gate)
    if not resource_gate["pass"]:
        write_blocker(pipeline_dir, resource_gate)
        raise RuntimeError("V4 resource gate failed")

    method_hashes = v2._method_hashes()
    gate = _run_gate(
        pipeline_dir=pipeline_dir,
        root=root,
        v2_root=v2_root,
        method_hashes=method_hashes,
        timeout_seconds=timeout_seconds,
        environment=environment,
        safety_policy=policy,
    )
    new_fills: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    unresolved = []
    original_adapter = v2.ADAPTER
    v2.ADAPTER = ADAPTER
    try:
        for position, row in enumerate(remaining, start=1):
            key = (row["method"], row["task"], int(row["sample_index"]))
            output_root = root / "fills" / v2._case_id(*key) / PROFILE_ID
            result = v2._run_isolated_case(
                pipeline_dir=pipeline_dir,
                fill_root=root,
                method=key[0],
                task=key[1],
                sample_index=key[2],
                loader_mode="offload",
                profile=OFFLOAD_PROFILE,
                output_root=output_root,
                method_hashes=method_hashes,
                timeout_seconds=timeout_seconds,
                environment=environment,
                safety_policy=policy,
            )
            prediction = v2._read_jsonl(
                v2._prediction_path(output_root, key[1])
            )[0]
            fill = {
                **result,
                "prediction": prediction,
                "profile": PROFILE_ID,
                "fill_source": "v4",
            }
            if result["prediction_status"] == "ok":
                new_fills[key] = fill
            elif result["prediction_status"] == "oom":
                unresolved.append(row)
            else:
                raise RuntimeError(
                    f"Invalid V4 status for {key}: {result['prediction_status']}"
                )
            _write_manifest(root, remaining, new_fills)
            atomic_write_json(root / "PROGRESS.json", {
                "created_at": now_iso(),
                "phase": "fullkv_mlp_chunk_fill",
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
    finally:
        v2.ADAPTER = original_adapter

    combined = v3._load_v2_fills(v2_root)
    combined.update(_load_manifest_fills(v3_root, "OOM_FILL_V3_MANIFEST.csv"))
    combined.update(new_fills)
    scoring = v2._score_offload_completed(
        fill_root=root,
        sources=sources,
        fills=combined,
    )
    _write_manifest(root, remaining, new_fills)
    raw_failures = v2._verify_raw_hash_lock(raw_lock)
    decision = {
        "created_at": now_iso(),
        "name": "oom_offload_fill_v4",
        "decision": (
            "PASS"
            if len(new_fills) == len(remaining) and not raw_failures
            else "SOFT_BLOCKED"
        ),
        "run_root": str(root),
        "v3_remaining_oom": len(remaining),
        "new_fills": len(new_fills),
        "remaining_oom": len(remaining) - len(new_fills),
        "remaining_rows": unresolved,
        "equivalence_gate": gate,
        "raw_predictions_modified": bool(raw_failures),
        "raw_hash_failures": raw_failures,
        "scoring": scoring,
    }
    atomic_write_json(root / "FINAL_OOM_FILL_V4_DECISION.json", decision)
    atomic_write_json(pipeline_dir / "QWEN_OOM_FILL_V4_DECISION.json", decision)
    atomic_write_text(
        pipeline_dir / "STAGE_11_V4_RESUME_COMMAND.txt",
        f"{PYTHON} {Path(__file__).resolve()} --pipeline-dir {pipeline_dir} "
        f"--safety-policy {safety_policy_path} --execute "
        f"--timeout-seconds {timeout_seconds}\n",
    )
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
            args.pipeline_dir.resolve(),
            args.safety_policy.resolve(),
            execute=args.execute,
            timeout_seconds=args.timeout_seconds,
        )
    except Exception as exc:
        failure = {
            "created_at": now_iso(),
            "name": "oom_offload_fill_v4",
            "decision": "HARD_BLOCKED",
            "reason": "stage_exception",
            "error": repr(exc),
        }
        atomic_write_json(
            args.pipeline_dir.resolve() / "QWEN_OOM_FILL_V4_FAILURE.json",
            failure,
        )
        print(json.dumps(failure, indent=2))
        return 2
    print(json.dumps(result, indent=2))
    return 0 if result.get("decision") in {
        "PASS", "SOFT_BLOCKED", "NOT_RUN"
    } else 2


if __name__ == "__main__":
    raise SystemExit(main())
