#!/usr/bin/env python3
"""Stage 0: read-only Qwen protocol, baseline, result, and resource preflight."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

_REPO_BOOTSTRAP = Path(__file__).resolve().parents[1]
if str(_REPO_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(_REPO_BOOTSTRAP))

from tools.qwen_pipeline_common import (
    COUNTS,
    DRM_METHODS,
    EXPECTED_OOM,
    EXPECTED_RAW_MACRO,
    PROTOCOL_DIR,
    PROTOCOL_ID,
    QWEN_DRM_ROOT,
    QWEN_MODEL_ROOT,
    REPO_ROOT,
    RUN_DIR_BY_METHOD,
    TASKS,
    atomic_write_csv,
    atomic_write_json,
    atomic_write_text,
    canonical_sample_identity,
    git_status,
    iter_jsonl,
    load_json,
    load_task_samples,
    now_iso,
    per_task_score,
    prediction_status,
    raw_sample_id,
    run_command,
    task_macro,
    validate_unique_ids,
)
from tools.qwen_protocol_attest import attest
from tools.qwen_resource_guard import collect_snapshot, evaluate_snapshot


def _read_identity_matrix() -> Dict[str, Any]:
    path = PROTOCOL_DIR / "PROTOCOL_IDENTITY_MATRIX.csv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    drm_rows = [
        row for row in rows
        if row["result_group"] == "qwen25_drm_official_3950" and row["method"] in DRM_METHODS
    ]
    by_method = {row["method"]: row["overall_status"] for row in drm_rows}
    return {
        "pass": len(drm_rows) == 5 and all(by_method.get(method) == "PASS" for method in DRM_METHODS),
        "status_by_method": by_method,
        "rows": len(drm_rows),
    }


def audit_method(method: str) -> Dict[str, Any]:
    run_dir = QWEN_DRM_ROOT / RUN_DIR_BY_METHOD[method]
    per_task: Dict[str, float] = {}
    task_audits: List[Dict[str, Any]] = []
    oom_rows: List[Dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    total_rows = 0
    identity_failures: List[Dict[str, Any]] = []
    duplicate_failures: List[Dict[str, Any]] = []
    native_status_rows = 0
    for task in TASKS:
        samples = load_task_samples(task)
        pred_path = run_dir / "longbench_predictions" / f"{task}.jsonl"
        predictions = list(iter_jsonl(pred_path)) if pred_path.exists() else []
        total_rows += len(predictions)
        uniqueness = validate_unique_ids(predictions, task)
        if not uniqueness["pass"]:
            duplicate_failures.extend(uniqueness["duplicates"])
        task_identity_failures: List[Dict[str, Any]] = []
        for index, row in enumerate(predictions):
            if row.get("status") in {"ok", "oom", "failed", "missing"}:
                native_status_rows += 1
            status = prediction_status(row)
            status_counts[status] += 1
            if status != "ok":
                oom_rows.append({
                    "method": method,
                    "task": task,
                    "sample_index": index,
                    "sample_id": row.get("sample_id"),
                    "status": status,
                    "prediction": row.get("pred", row.get("prediction", "")),
                })
            if index >= len(samples):
                task_identity_failures.append({"sample_index": index, "reason": "prediction_out_of_range"})
                continue
            sample = samples[index]
            expected_raw = raw_sample_id(sample, index)
            expected_canonical, content_hash = canonical_sample_identity(task, index, sample)
            actual_index = int(row.get("sample_index", index))
            actual_raw = str(row.get("sample_id", actual_index))
            if actual_index != index or actual_raw != expected_raw:
                task_identity_failures.append({
                    "sample_index": index,
                    "expected_raw_sample_id": expected_raw,
                    "actual_raw_sample_id": actual_raw,
                    "actual_sample_index": actual_index,
                    "canonical_sample_id": expected_canonical,
                    "content_sha256": content_hash,
                })
        identity_failures.extend({"task": task, **row} for row in task_identity_failures)
        score = per_task_score(task, predictions) if len(predictions) == COUNTS[task] else None
        if score is not None:
            per_task[task] = score
        task_audits.append({
            "task": task,
            "expected": COUNTS[task],
            "predictions": len(predictions),
            "score": score,
            "identity_failures": len(task_identity_failures),
            "duplicates": len(uniqueness["duplicates"]),
            "pass": (
                len(predictions) == COUNTS[task]
                and not task_identity_failures
                and uniqueness["pass"]
            ),
        })
    macro = task_macro(per_task) if len(per_task) == 17 else None
    expected_macro = EXPECTED_RAW_MACRO[method]
    expected_oom = EXPECTED_OOM[method]
    non_ok = sum(value for key, value in status_counts.items() if key != "ok")
    return {
        "method": method,
        "run_dir": str(run_dir),
        "prediction_count": total_rows,
        "expected_prediction_count": 3950,
        "per_task_scores": per_task,
        "macro": macro,
        "expected_macro": expected_macro,
        "macro_delta_from_crosscheck": (macro - expected_macro) if macro is not None else None,
        "status_counts": dict(status_counts),
        "native_status_rows": native_status_rows,
        "posthoc_status_rows": total_rows - native_status_rows,
        "canonical_status_coverage": total_rows,
        "oom_or_non_ok_count": non_ok,
        "expected_oom": expected_oom,
        "oom_manifest": oom_rows,
        "identity_failures": identity_failures,
        "duplicate_failures": duplicate_failures,
        "task_audits": task_audits,
        "pass": bool(
            total_rows == 3950
            and macro is not None
            and abs(macro - expected_macro) <= 0.0001
            and non_ok == expected_oom
            and not identity_failures
            and not duplicate_failures
            and all(row["pass"] for row in task_audits)
        ),
    }


def external_baseline_availability() -> Dict[str, Any]:
    search = run_command([
        "find", "/home/yu-bai", "-maxdepth", "5", "-type", "d", "(",
        "-iname", "*minference*", "-o", "-iname", "*flexprefill*", "-o", "-iname", "*lazyllm*", ")",
        "-print",
    ])
    candidates = [line for line in search["output"].splitlines() if line.strip()]
    conda = run_command(["conda", "env", "list", "--json"])
    try:
        environments = json.loads(conda["output"]).get("envs", []) if conda["returncode"] == 0 else []
    except Exception:
        environments = []
    result: Dict[str, Any] = {
        "created_at": now_iso(),
        "local_candidates": candidates,
        "conda_environments": environments,
        "methods": {},
    }
    for method, token in (
        ("minference_official", "minference"),
        ("flexprefill_official", "flexprefill"),
        ("lazyllm_official", "lazyllm"),
    ):
        matching = [path for path in candidates if token in path.lower()]
        result["methods"][method] = {
            "candidate_paths": matching,
            "status": "CANDIDATE_FOUND_IDENTITY_UNVERIFIED" if matching else "NOT_FOUND_LOCAL",
            "official_identity_pass": False,
            "full_launch_allowed": False,
        }
    result["note"] = "Absence is not a Stage 0 blocker; every external method requires a later official-path audit and isolated environment."
    return result


def run_preflight(
    pipeline_dir: Path,
    safety_policy_path: Path,
    audit_dir: Path,
    *,
    verify_history: bool = True,
) -> Dict[str, Any]:
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    attestation = attest(
        pipeline_dir,
        audit_dir,
        verify_historical_files=verify_history,
    )
    method_audits = [audit_method(method) for method in DRM_METHODS]
    identity = _read_identity_matrix()
    resources = collect_snapshot()
    policy = load_json(safety_policy_path)
    resource_check = evaluate_snapshot(resources, policy, pipeline_dir=pipeline_dir)
    external = external_baseline_availability()
    canonical = next(row for row in method_audits if row["method"] == "sliminfer_original")
    canonical_decision = {
        "created_at": now_iso(),
        "decision": "PASS" if canonical["pass"] else "FAIL",
        "protocol_id": PROTOCOL_ID,
        "method": "sliminfer_original",
        "source_root": str(QWEN_DRM_ROOT),
        "run_dir": canonical["run_dir"],
        "prediction_count": canonical["prediction_count"],
        "raw_macro": canonical["macro"],
        "oom_count": canonical["oom_or_non_ok_count"],
        "reason": "Canonical shared Qwen baseline is the DRM official full Original run; historical CC Original is development-only.",
    }
    atomic_write_json(pipeline_dir / "CANONICAL_BASELINE_DECISION.json", canonical_decision)
    atomic_write_json(pipeline_dir / "EXTERNAL_BASELINE_AVAILABILITY.json", external)
    atomic_write_json(pipeline_dir / "GPU_BASELINE.json", resources["gpu"])
    atomic_write_json(pipeline_dir / "ENVIRONMENT.json", resources)
    atomic_write_json(pipeline_dir / "DISK_USAGE.json", resources["disk"])
    oom_rows = [item for method in method_audits for item in method["oom_manifest"]]
    atomic_write_csv(
        pipeline_dir / "STAGE_00_OOM_MANIFEST.csv",
        oom_rows,
        ["method", "task", "sample_index", "sample_id", "status", "prediction"],
    )
    existing = {
        "created_at": now_iso(),
        "qwen_drm_root": str(QWEN_DRM_ROOT),
        "methods": method_audits,
        "identity_matrix": identity,
    }
    atomic_write_json(pipeline_dir / "EXISTING_RESULTS_AUDIT.json", existing)
    registry_rows = [
        {
            "stage": 0,
            "method": row["method"],
            "task": "official_17_task_full",
            "status": "AUDITED_PASS" if row["pass"] else "AUDITED_FAIL",
            "path": row["run_dir"],
            "updated_at": now_iso(),
        }
        for row in method_audits
    ]
    registry_rows.append({
        "stage": 2,
        "method": "cc_sliminfer_v1",
        "task": "official_17_task_full",
        "status": "PLANNED_WAITING_FOR_STAGE_01",
        "path": "",
        "updated_at": now_iso(),
    })
    atomic_write_csv(
        pipeline_dir / "RUN_REGISTRY.csv",
        registry_rows,
        ["stage", "method", "task", "status", "path", "updated_at"],
    )
    md = [
        "# Existing Qwen Results Audit",
        "",
        "All formal values below were recomputed from the five original prediction trees.",
        "",
        "| method | predictions | raw macro | non-ok/OOM | identity | decision |",
        "|---|---:|---:|---:|---|---|",
    ]
    for row in method_audits:
        md.append(
            f"| `{row['method']}` | {row['prediction_count']} | {row['macro']:.6f} | "
            f"{row['oom_or_non_ok_count']} | {'PASS' if not row['identity_failures'] else 'FAIL'} | "
            f"{'PASS' if row['pass'] else 'FAIL'} |"
        )
    md.extend([
        "",
        "Canonical statuses are fully reconstructable. Historical rows without a native status are labeled posthoc; `[OOM]` remains status `oom`.",
        "LLaMA is intentionally out of scope and its absence is not a Qwen blocker.",
    ])
    atomic_write_text(pipeline_dir / "EXISTING_RESULTS_AUDIT.md", "\n".join(md) + "\n")
    pass_conditions = {
        "protocol_attestation": attestation["decision"] == "PASS",
        "canonical_original": canonical_decision["decision"] == "PASS",
        "five_method_results": all(row["pass"] for row in method_audits),
        "five_method_identity": identity["pass"],
        "resource_snapshot": resource_check["pass"],
        "model_path_present": QWEN_MODEL_ROOT.exists(),
    }
    decision = "PASS" if all(pass_conditions.values()) else "BLOCKED"
    report = {
        "stage": 0,
        "name": "preflight_and_baseline",
        "created_at": now_iso(),
        "decision": decision,
        "pass_conditions": pass_conditions,
        "protocol_attestation": attestation["decision"],
        "implementation_equivalence": attestation["implementation_equivalence"],
        "canonical_baseline": canonical_decision,
        "resources": resource_check,
        "next_stage": 1 if decision == "PASS" else None,
        "next_stage_name": "original_equivalence_canary" if decision == "PASS" else None,
        "gpu_model_loaded": False,
    }
    atomic_write_json(pipeline_dir / "STAGE_00_DECISION.json", report)
    atomic_write_text(
        pipeline_dir / "logs/stage_00.log",
        json.dumps({
            "created_at": report["created_at"],
            "decision": decision,
            "pass_conditions": pass_conditions,
            "canonical_macro": canonical_decision["raw_macro"],
            "canonical_oom": canonical_decision["oom_count"],
            "gpu_model_loaded": False,
        }, sort_keys=True) + "\n",
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-dir", type=Path, required=True)
    parser.add_argument("--safety-policy", type=Path, required=True)
    parser.add_argument(
        "--audit-dir",
        type=Path,
        default=REPO_ROOT / "results/unified_protocol_v1_audit_20260720_202245",
    )
    parser.add_argument("--skip-history-hash-check", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run_preflight(
        args.pipeline_dir,
        args.safety_policy,
        args.audit_dir,
        verify_history=not args.skip_history_hash_check,
    )
    print(json.dumps(report, indent=2))
    return 0 if report["decision"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
