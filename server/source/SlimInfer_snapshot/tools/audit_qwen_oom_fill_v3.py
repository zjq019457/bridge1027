#!/usr/bin/env python3
"""CPU-only integrity and active-budget audit for Qwen OOM Fill V3."""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


K_BY_LAYER = {9: 192, 18: 96, 26: 64}
M_BY_LAYER = {9: 10, 18: 5, 26: 3}
ADD_METHODS = {"sliminfer_drm_add_km", "sliminfer_random_add_km"}
K_METHODS = {"sliminfer_original", "cc_sliminfer_v1"}
SWAP_METHODS = {"drm_swap_original_k"}
LARGER_METHODS = {"sliminfer_larger_km"}
FULL_KV_METHODS = {"full_kv_fa2"}
KNOWN_METHODS = ADD_METHODS | K_METHODS | SWAP_METHODS | LARGER_METHODS | FULL_KV_METHODS


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    if path.suffix == ".gz":
        handle = gzip.open(path, "rt", encoding="utf-8")
    else:
        handle = path.open("r", encoding="utf-8")
    with handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = [
        "method", "task", "sample_index", "sample_id", "fill_status",
        "prediction_ok", "case_ok", "trace_events", "budget_violations",
        "sink_violations", "duplicate_violations", "invalid_id_violations",
        "add_preservation_violations", "swap_one_in_one_out_violations",
        "event_count_violations", "audit_decision", "errors",
    ]
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def expected_budget(method: str, layer: int) -> int:
    if method in ADD_METHODS | LARGER_METHODS:
        return K_BY_LAYER[layer] + M_BY_LAYER[layer]
    return K_BY_LAYER[layer]


def audit_trace_event(method: str, event: Mapping[str, Any]) -> List[str]:
    errors: List[str] = []
    try:
        layer = int(event["layer_idx"])
    except (KeyError, TypeError, ValueError):
        return ["invalid_layer"]
    if layer not in K_BY_LAYER:
        return [f"unexpected_layer:{layer}"]

    selected_raw = event.get("selected_block_ids")
    base_raw = event.get("base_original_active_ids")
    candidate_raw = event.get("candidate_ids")
    sink_raw = event.get("sink_ids")
    if not isinstance(selected_raw, list):
        return ["selected_ids_missing"]
    selected = set(selected_raw)
    base = set(base_raw or [])
    candidates = set(candidate_raw or [])
    sinks = set(sink_raw or [])
    if len(selected) != len(selected_raw):
        errors.append("duplicate_selected_ids")
    if any(not isinstance(value, int) or value < 0 for value in selected_raw):
        errors.append("invalid_selected_id")
    if candidates and not selected.issubset(candidates):
        errors.append("selected_outside_candidate_pool")
    if not sinks.issubset(selected):
        errors.append("sink_not_selected")

    target = expected_budget(method, layer)
    recorded_counts = {
        int(value)
        for value in (
            event.get("selected_count"),
            event.get("num_selected"),
            len(selected_raw),
        )
        if value is not None
    }
    if recorded_counts != {target}:
        errors.append(f"budget_mismatch:{sorted(recorded_counts)}!={target}")

    k = K_BY_LAYER[layer]
    m = M_BY_LAYER[layer]
    if method in ADD_METHODS:
        if len(base) != k or not base.issubset(selected):
            errors.append("add_base_not_preserved")
        if len(selected - base) != m or selected & base != base:
            errors.append("add_count_mismatch")
    elif method in SWAP_METHODS:
        if len(base) != k:
            errors.append("swap_base_count_mismatch")
        if len(selected - base) != m or len(base - selected) != m:
            errors.append("swap_not_one_in_one_out")
    elif method in K_METHODS:
        if len(selected) != k:
            errors.append("k_budget_mismatch")
    elif method in LARGER_METHODS:
        if len(selected) != k + m:
            errors.append("larger_budget_mismatch")
    return errors


def _classify_errors(errors: Iterable[str]) -> Dict[str, int]:
    errors = list(errors)
    return {
        "budget_violations": sum(
            error.startswith(("budget_", "k_budget", "larger_budget"))
            for error in errors
        ),
        "sink_violations": sum(error == "sink_not_selected" for error in errors),
        "duplicate_violations": sum(
            error == "duplicate_selected_ids" for error in errors
        ),
        "invalid_id_violations": sum(
            error in {"invalid_selected_id", "selected_outside_candidate_pool"}
            for error in errors
        ),
        "add_preservation_violations": sum(
            error.startswith("add_") for error in errors
        ),
        "swap_one_in_one_out_violations": sum(
            error.startswith("swap_") for error in errors
        ),
    }


def audit_case(fill_root: Path, manifest_row: Mapping[str, str]) -> Dict[str, Any]:
    method = manifest_row["method"]
    task = manifest_row["task"]
    sample_index = int(manifest_row["sample_index"])
    sample_id = manifest_row["sample_id"]
    fill_status = manifest_row["fill_status"]
    result: Dict[str, Any] = {
        "method": method,
        "task": task,
        "sample_index": sample_index,
        "sample_id": sample_id,
        "fill_status": fill_status,
        "prediction_ok": False,
        "case_ok": False,
        "trace_events": 0,
        "budget_violations": 0,
        "sink_violations": 0,
        "duplicate_violations": 0,
        "invalid_id_violations": 0,
        "add_preservation_violations": 0,
        "swap_one_in_one_out_violations": 0,
        "event_count_violations": 0,
        "audit_decision": "PENDING",
        "errors": "",
    }
    if fill_status != "ok":
        return result
    errors: List[str] = []
    prediction_path = Path(manifest_row["fill_prediction_path"])
    predictions = read_jsonl(prediction_path)
    prediction = predictions[0] if len(predictions) == 1 else {}
    prediction_ok = (
        len(predictions) == 1
        and prediction.get("status") == "ok"
        and int(prediction.get("sample_index", -1)) == sample_index
        and prediction.get("sample_id") == sample_id
        and sha256_file(prediction_path)
        == manifest_row.get("fill_prediction_sha256")
    )
    result["prediction_ok"] = prediction_ok
    if not prediction_ok:
        errors.append("prediction_integrity")

    case_root = prediction_path.parents[2]
    case_path = case_root / "CASE_DECISION.json"
    case = json.loads(case_path.read_text(encoding="utf-8")) if case_path.exists() else {}
    case_ok = (
        case.get("decision") == "PASS"
        and case.get("returncode") == 0
        and not case.get("timed_out")
        and not case.get("driver_failure_marker")
        and case.get("prediction_status") == "ok"
        and case.get("prediction_identity_ok") is True
    )
    result["case_ok"] = case_ok
    if not case_ok:
        errors.append("case_decision")

    if method not in KNOWN_METHODS:
        errors.append(f"unknown_method:{method}")
    if method not in FULL_KV_METHODS:
        trace_path = case_root / "active_block_trace" / f"{task}.jsonl.gz"
        trace_rows = read_jsonl(trace_path)
        result["trace_events"] = len(trace_rows)
        if not trace_rows:
            errors.append("active_trace_missing")
        event_errors: List[str] = []
        counts: Counter[int] = Counter()
        for event in trace_rows:
            if (
                event.get("sample_id") != sample_id
                or int(event.get("sample_index", -1)) != sample_index
            ):
                event_errors.append("trace_identity")
            try:
                counts[int(event["layer_idx"])] += 1
            except (KeyError, TypeError, ValueError):
                pass
            event_errors.extend(audit_trace_event(method, event))
        output_tokens = int(prediction.get("output_tokens", 0) or 0)
        expected_counts = {
            layer: output_tokens for layer in sorted(K_BY_LAYER)
        }
        if dict(sorted(counts.items())) != expected_counts:
            result["event_count_violations"] = 1
            event_errors.append(
                f"event_counts:{dict(sorted(counts.items()))}!={expected_counts}"
            )
        result.update(_classify_errors(event_errors))
        errors.extend(event_errors)

    result["audit_decision"] = "PASS" if not errors else "FAIL"
    result["errors"] = ";".join(sorted(set(errors)))
    return result


def audit(fill_root: Path) -> Dict[str, Any]:
    manifest_path = fill_root / "OOM_FILL_V3_MANIFEST.csv"
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        manifest = list(csv.DictReader(handle))
    rows = [audit_case(fill_root, row) for row in manifest]
    completed = [row for row in rows if row["fill_status"] == "ok"]
    pending = len(rows) - len(completed)
    failed = [row for row in completed if row["audit_decision"] != "PASS"]
    raw_lock_path = fill_root / "RAW_UNCHANGED_HASH_LOCK.json"
    raw_lock = json.loads(raw_lock_path.read_text(encoding="utf-8"))
    locked_files = raw_lock["files"]
    if isinstance(locked_files, list):
        lock_items = [
            (Path(row["path"]), row["sha256"]) for row in locked_files
        ]
    else:
        lock_items = [
            (Path(path), expected) for path, expected in locked_files.items()
        ]
    raw_failures = [
        str(path)
        for path, expected in lock_items
        if not path.exists() or sha256_file(path) != expected
    ]
    decision = (
        "PASS"
        if not pending and not failed and not raw_failures
        else "FAIL" if failed or raw_failures else "IN_PROGRESS"
    )
    report = {
        "decision": decision,
        "fill_root": str(fill_root),
        "manifest_rows": len(rows),
        "completed_rows": len(completed),
        "pending_rows": pending,
        "failed_rows": len(failed),
        "raw_predictions_modified": bool(raw_failures),
        "raw_hash_failures": raw_failures,
        "totals": {
            key: sum(int(row[key]) for row in completed)
            for key in (
                "budget_violations", "sink_violations",
                "duplicate_violations", "invalid_id_violations",
                "add_preservation_violations",
                "swap_one_in_one_out_violations",
                "event_count_violations",
            )
        },
        "rows": rows,
    }
    atomic_csv(fill_root / "OOM_FILL_V3_BUDGET_AUDIT.csv", rows)
    atomic_json(fill_root / "OOM_FILL_V3_BUDGET_AUDIT.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fill-root", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.fill_root.resolve())
    print(json.dumps(report, indent=2))
    return 0 if report["decision"] in {"PASS", "IN_PROGRESS"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
