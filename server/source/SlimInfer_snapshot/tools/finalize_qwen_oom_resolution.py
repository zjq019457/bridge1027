#!/usr/bin/env python3
"""Build the CPU-only closure audit for Qwen OOM Fill V2/V3/V4."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping

_REPO_BOOTSTRAP = Path(__file__).resolve().parents[1]
if str(_REPO_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(_REPO_BOOTSTRAP))

from tools.audit_qwen_oom_fill_v3 import audit as audit_v3
from tools.qwen_pipeline_common import (
    COUNTS,
    TASKS,
    atomic_write_csv,
    atomic_write_json,
    atomic_write_text,
    load_json,
    now_iso,
    prediction_status,
    sha256_file,
)


METHODS = (
    "sliminfer_original",
    "cc_sliminfer_v1",
    "drm_swap_original_k",
    "sliminfer_larger_km",
    "sliminfer_random_add_km",
    "sliminfer_drm_add_km",
    "full_kv_fa2",
)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _raw_hash_failures(lock_path: Path) -> List[str]:
    rows = load_json(lock_path)["files"]
    items = (
        [(Path(row["path"]), row["sha256"]) for row in rows]
        if isinstance(rows, list)
        else [(Path(path), digest) for path, digest in rows.items()]
    )
    return [
        str(path)
        for path, expected in items
        if not path.exists() or sha256_file(path) != expected
    ]


def finalize(pipeline_dir: Path) -> Dict[str, Any]:
    v3_root = Path(load_json(pipeline_dir / "QWEN_OOM_FILL_ROOT_V3.json")["path"])
    v4_root = Path(load_json(pipeline_dir / "QWEN_OOM_FILL_ROOT_V4.json")["path"])
    v3_decision = load_json(v3_root / "FINAL_OOM_FILL_V3_DECISION.json")
    v4_decision = load_json(v4_root / "FINAL_OOM_FILL_V4_DECISION.json")
    v3_audit = audit_v3(v3_root)
    v4_manifest = _read_csv(v4_root / "OOM_FILL_V4_MANIFEST.csv")
    v4_fill_ok = (
        len(v4_manifest) == 1
        and v4_manifest[0]["fill_status"] == "ok"
        and Path(v4_manifest[0]["fill_prediction_path"]).exists()
        and sha256_file(Path(v4_manifest[0]["fill_prediction_path"]))
        == v4_manifest[0]["fill_prediction_sha256"]
    )
    residual_identity_ok = (
        len(v3_decision.get("remaining_rows", [])) == 1
        and len(v4_manifest) == 1
        and all(
            str(v3_decision["remaining_rows"][0][key])
            == str(v4_manifest[0][key])
            for key in ("method", "task", "sample_index", "sample_id")
        )
    )

    predictions_path = v4_root / "offload_completed_predictions.jsonl"
    method_task = Counter()
    method_status = Counter()
    method_sample = Counter()
    prediction_rows = 0
    with predictions_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            method = row.get("method_alias") or row.get("method")
            task = row.get("task")
            sample_id = row.get("sample_id")
            prediction_rows += 1
            method_task[(method, task)] += 1
            method_status[(method, prediction_status(row))] += 1
            method_sample[(method, sample_id)] += 1

    count_rows = []
    count_failures = []
    for method in METHODS:
        for task in TASKS:
            actual = method_task[(method, task)]
            expected = COUNTS[task]
            status = "PASS" if actual == expected else "FAIL"
            count_rows.append({
                "method": method,
                "task": task,
                "expected": expected,
                "actual": actual,
                "status": status,
            })
            if status != "PASS":
                count_failures.append(f"{method}/{task}:{actual}!={expected}")
    non_ok = {
        f"{method}/{status}": count
        for (method, status), count in method_status.items()
        if status != "ok"
    }
    duplicate_ids = sum(count != 1 for count in method_sample.values())
    score_rows = _read_csv(v4_root / "offload_completed_scores_by_sample.csv")
    summaries = {
        row["method"]: row
        for row in v4_decision["scoring"]["summaries"]
    }
    raw_failures = _raw_hash_failures(
        v4_root / "RAW_UNCHANGED_HASH_LOCK.json"
    )
    v3_totals_zero = all(
        int(value) == 0 for value in v3_audit["totals"].values()
    )
    checks = {
        "v3_decision_soft_blocked_only_for_one_fullkv": (
            v3_decision.get("decision") == "SOFT_BLOCKED"
            and v3_decision.get("remaining_oom") == 1
        ),
        "v3_completed_structure_rows_pass": (
            v3_audit["completed_rows"] == 12
            and v3_audit["failed_rows"] == 0
            and v3_audit["pending_rows"] == 1
            and v3_totals_zero
        ),
        "v4_residual_identity_matches": residual_identity_ok,
        "v4_equivalence_gate_pass": (
            v4_decision["equivalence_gate"]["decision"] == "PASS"
            and v4_decision["equivalence_gate"]["completed_cases"] == 3
            and v4_decision["equivalence_gate"]["prediction_matches"] == 3
        ),
        "v4_fill_pass": v4_fill_ok,
        "v4_final_decision_pass": (
            v4_decision.get("decision") == "PASS"
            and v4_decision.get("remaining_oom") == 0
        ),
        "combined_prediction_rows_27650": prediction_rows == 27650,
        "combined_score_rows_27650": len(score_rows) == 27650,
        "all_method_task_counts_exact": not count_failures,
        "all_combined_predictions_valid": not non_ok,
        "method_sample_ids_unique": duplicate_ids == 0,
        "all_method_summaries_complete": (
            set(summaries) == set(METHODS)
            and all(int(row["remaining_oom_rows"]) == 0 for row in summaries.values())
        ),
        "raw_predictions_unchanged": not raw_failures,
    }
    decision = "PASS" if all(checks.values()) else "FAIL"
    report = {
        "created_at": now_iso(),
        "decision": decision,
        "pipeline_dir": str(pipeline_dir),
        "v3_root": str(v3_root),
        "v4_root": str(v4_root),
        "checks": checks,
        "prediction_rows": prediction_rows,
        "score_rows": len(score_rows),
        "methods": len(METHODS),
        "tasks": len(TASKS),
        "samples_per_method": sum(COUNTS.values()),
        "remaining_oom": sum(
            int(row["remaining_oom_rows"]) for row in summaries.values()
        ),
        "non_ok_predictions": non_ok,
        "duplicate_method_sample_ids": duplicate_ids,
        "count_failures": count_failures,
        "raw_hash_failures": raw_failures,
        "v3_structure_totals": v3_audit["totals"],
        "summaries": [summaries[method] for method in METHODS],
    }
    atomic_write_csv(
        v4_root / "FINAL_COMPLETE_COUNTS.csv",
        count_rows,
        ["method", "task", "expected", "actual", "status"],
    )
    atomic_write_json(v4_root / "FINAL_OOM_RESOLUTION_AUDIT.json", report)
    lines = [
        "# Final Qwen OOM Resolution Audit",
        "",
        f"- Decision: **{decision}**",
        f"- Combined predictions: **{prediction_rows} / 27650**",
        f"- Combined scores: **{len(score_rows)} / 27650**",
        f"- Remaining OOM: **{report['remaining_oom']}**",
        f"- Raw prediction hash failures: **{len(raw_failures)}**",
        f"- Method/sample duplicate IDs: **{duplicate_ids}**",
        "",
        "## Method Summary",
        "",
        "| Method | 17-task macro | Filled OOM | Remaining OOM |",
        "|---|---:|---:|---:|",
    ]
    for method in METHODS:
        row = summaries[method]
        lines.append(
            f"| `{method}` | {row['task_macro']} | "
            f"{row['filled_oom_rows']} | {row['remaining_oom_rows']} |"
        )
    lines.extend([
        "",
        "## Closure Checks",
        "",
        *[
            f"- {'PASS' if value else 'FAIL'}: `{name}`"
            for name, value in checks.items()
        ],
        "",
        "Raw experiment outputs were hash-locked and not overwritten. "
        "The completed-accuracy files are separate V4 artifacts.",
    ])
    atomic_write_text(
        v4_root / "FINAL_OOM_RESOLUTION_REPORT.md",
        "\n".join(lines) + "\n",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-dir", type=Path, required=True)
    args = parser.parse_args()
    report = finalize(args.pipeline_dir.resolve())
    print(json.dumps(report, indent=2))
    return 0 if report["decision"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
