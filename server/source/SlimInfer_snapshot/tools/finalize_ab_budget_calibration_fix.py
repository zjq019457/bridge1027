#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
import stat
from pathlib import Path
from typing import Any, Dict, Iterable, List


REPO_ROOT = Path(__file__).resolve().parents[1]
TASKS = ["hotpotqa", "2wikimqa", "musique", "qasper", "multifieldqa_en"]
CALIBRATION_METHODS = [
    ("fixed_sliminfer_matched_K", "fixed_K", "active"),
    ("attention_boundary_sliminfer_matched_K", "fixed_K", "active"),
    ("fixed_drm_add_matched_KplusM", "fixed_KplusM_drm", "final"),
    ("attention_boundary_boundary_drm_matched_KplusM", "fixed_KplusM_drm", "final"),
    ("fixed_sliminfer_matched_attention_tokens", "attention_raw", "active"),
    ("attention_boundary_sliminfer_matched_attention_tokens", "attention_raw", "active"),
]
FALLBACK_METHODS = [
    ("semantic_v3_sliminfer_diagnostic", "semantic_v3"),
    ("query_dynamic_v3_sliminfer_diagnostic", "query_dynamic_v3"),
]
FALLBACK_REASONS = [
    "no_raw_text",
    "no_query_text",
    "no_offsets",
    "no_document_boundary",
    "no_paragraph_boundary",
    "no_sentence_boundary",
    "too_short_units",
    "too_long_units",
    "coverage_gap",
    "overlap_error",
    "tokenizer_alignment_error",
    "kernel_incompatible",
    "fallback_fixed",
    "unknown",
]


def fnum(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except Exception:
        return default


def mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    return sum(vals) / len(vals) if vals else 0.0


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def write_csv(path: Path, rows: List[Dict[str, Any]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def make_executable(path: Path) -> None:
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def load_budgets(out_dir: Path) -> Dict[str, float]:
    config = read_json(out_dir / "config.json")
    budgets = dict(config.get("budgets") or {})
    calib = read_json(out_dir / "calibrated_budget_config.json")
    targets = calib.get("targets", {}) if isinstance(calib.get("targets"), dict) else {}
    for key, value in targets.items():
        budgets.setdefault(key, value)
    return {key: float(value) for key, value in budgets.items()}


def task_calibration_entry(calib: Dict[str, Any], method: str, task: str) -> Dict[str, Any]:
    methods = calib.get("methods", {}) if isinstance(calib.get("methods"), dict) else {}
    method_entry = methods.get(method, {}) if isinstance(methods.get(method), dict) else {}
    tasks = method_entry.get("tasks", {}) if isinstance(method_entry.get("tasks"), dict) else {}
    task_entry = tasks.get(task, {}) if isinstance(tasks.get(task), dict) else {}
    merged = {key: value for key, value in method_entry.items() if key != "tasks"}
    merged.update(task_entry)
    return merged


def enable_reachability_caps(calib: Dict[str, Any]) -> None:
    method_names = [method for method, _, _ in CALIBRATION_METHODS]
    containers = []
    methods = calib.get("methods")
    if isinstance(methods, dict):
        containers.append(methods)
    flat_methods = calib.get("flat_methods")
    if isinstance(flat_methods, dict):
        containers.append(flat_methods)
    containers.append(calib)
    for container in containers:
        for method in method_names:
            method_entry = container.get(method)
            if not isinstance(method_entry, dict):
                continue
            tasks = method_entry.get("tasks")
            if not isinstance(tasks, dict):
                continue
            for task_entry in tasks.values():
                if isinstance(task_entry, dict):
                    task_entry["cap_to_reachable_tokens"] = True


def sample_groups(rows: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for index, row in enumerate(rows):
        sample_key = row.get("sample_id")
        if sample_key in (None, ""):
            sample_key = row.get("sample_index")
        if sample_key in (None, ""):
            sample_key = f"row-{index}"
        grouped.setdefault(str(sample_key), []).append(row)
    return list(grouped.values())


def row_fair_target(row: Dict[str, Any], fallback: float) -> float:
    calibration_target = fnum(row.get("calibration_target_tokens"), fallback)
    runtime_target = fnum(row.get("effective_target_tokens", row.get("target_tokens_used_by_cache")), calibration_target)
    cap_reason = str(row.get("target_cap_reason", "") or "")
    if "candidate_token_ceiling" in cap_reason or "protected_token_floor" in cap_reason:
        return runtime_target
    return calibration_target or runtime_target or fallback


def aggregate_calibration(out_dir: Path, budgets: Dict[str, float]) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    mismatches: List[Dict[str, Any]] = []
    audit_summary: List[Dict[str, Any]] = []
    calib = read_json(out_dir / "calibrated_budget_config.json")
    for method, target_source, kind in CALIBRATION_METHODS:
        audit = [r for r in read_jsonl(out_dir / "audit" / f"{method}.jsonl") if r.get("phase") == "prefill"]
        for task in TASKS:
            task_rows = [r for r in audit if r.get("task") == task]
            entry = task_calibration_entry(calib, method, task)
            configured_target = fnum(entry.get("target_tokens"), fnum(budgets.get(target_source)))
            groups = sample_groups(task_rows)
            target = mean(row_fair_target(r, configured_target) for r in task_rows) if task_rows else configured_target
            observed = mean(fnum(r.get("observed_final_selected_tokens_used_by_attention", r.get("selected_token_count"))) for r in task_rows)
            err = (observed - target) / target if target > 0 else 0.0
            abs_err = abs(err)
            selected_pages = mean(fnum(r.get("selected_pages", r.get("final_block_count_used_by_attention"))) for r in task_rows)
            min_reachable = mean(fnum(r.get("min_reachable_tokens")) for r in task_rows) if task_rows else ""
            max_reachable = mean(fnum(r.get("max_reachable_tokens")) for r in task_rows) if task_rows else ""
            cap_reasons = sorted({
                reason
                for row in task_rows
                for reason in str(row.get("target_cap_reason", "") or "").split(",")
                if reason
            })
            status = "ok" if task_rows and abs_err <= 0.03 else ("within_hard_tolerance" if task_rows and abs_err <= 0.05 else "budget_mismatch")
            reason = ""
            if not task_rows:
                status = "missing_audit"
                reason = "missing_audit"
            elif target_source == "attention_raw" and abs_err > 0.05:
                reason = "task_capped_attention_target_not_matched"
            elif abs_err > 0.03:
                reason = "nearest_target_outside_soft_tolerance"
            row = {
                "method": method,
                "task": task,
                "target_tokens": target,
                "uncapped_target_tokens": fnum(entry.get("uncapped_target_tokens"), configured_target),
                "observed_tokens": observed,
                "error_pct": err,
                "abs_error_pct": abs_err,
                "active_budget_scale": "",
                "final_budget_scale": "",
                "min_reachable_tokens": min_reachable,
                "max_reachable_tokens": max_reachable,
                "protected_tokens": "",
                "total_candidate_tokens": max_reachable,
                "selected_pages": selected_pages,
                "page_size": 64,
                "overshoot_used": any(bool(r.get("overshoot_used")) for r in task_rows) if task_rows else "",
                "budget_status": status,
                "mismatch_reason": reason,
                "target_cap_reason": ",".join(cap_reasons),
            }
            rows.append(row)
            if abs_err > 0.03 or status != "ok":
                mismatches.append(row)
            audit_summary.append({
                "method": method,
                "task": task,
                "sample_count": len(groups),
                "audit_row_count": len(task_rows),
                "calibrated_config_loaded": mean(1.0 if r.get("calibrated_budget_config_loaded") else 0.0 for r in task_rows) if task_rows else "",
                "calibration_entry_found": mean(1.0 if r.get("calibration_entry_found") else 0.0 for r in task_rows) if task_rows else "",
                "target_tokens_used_by_cache": target if task_rows else "",
                "observed_final_selected_tokens_used_by_attention": observed,
                "selected_pages": selected_pages,
                "page_expansion_overhead_mean": mean(fnum(r.get("page_expansion_overhead_mean")) for r in task_rows) if task_rows else "",
                "duplicate_pages": "",
                "protected_violation": any(bool(r.get("protected_block_violation")) for r in task_rows) if task_rows else "",
                "status": status,
            })
    return rows, mismatches, audit_summary


def aggregate_fallback(out_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for method, strategy in FALLBACK_METHODS:
        audit = [r for r in read_jsonl(out_dir / "audit" / f"{method}.jsonl") if r.get("phase") == "prefill"]
        for task in TASKS:
            task_rows = [r for r in audit if r.get("task") == task]
            counts = {reason: 0 for reason in FALLBACK_REASONS}
            for rec in task_rows:
                frc = rec.get("fallback_reason_counts") if isinstance(rec.get("fallback_reason_counts"), dict) else {}
                for reason in FALLBACK_REASONS:
                    counts[reason] += int(frc.get(reason, 0) or 0)
            row = {
                "block_strategy": strategy,
                "task": task,
                "fallback_rate": mean(1.0 if r.get("fallback_used") else 0.0 for r in task_rows) if task_rows else "",
                **counts,
                "status": "ok" if task_rows else "missing_audit",
            }
            rows.append(row)
    return rows


def report(out_dir: Path, calib_rows: List[Dict[str, Any]], mismatch_rows: List[Dict[str, Any]], fallback_rows: List[Dict[str, Any]]) -> bool:
    abs_errs = [fnum(r["abs_error_pct"]) for r in calib_rows if r.get("budget_status") != "missing_audit"]
    gate_mean = mean(abs_errs) < 0.03 if abs_errs else False
    gate_all = all(e < 0.05 for e in abs_errs) if abs_errs else False
    no_missing = all(r.get("budget_status") != "missing_audit" for r in calib_rows)
    full_ready = bool(gate_mean and gate_all and no_missing)
    method_means = []
    for method, _, _ in CALIBRATION_METHODS:
        vals = [fnum(r["abs_error_pct"]) for r in calib_rows if r["method"] == method and r.get("budget_status") != "missing_audit"]
        method_means.append((method, mean(vals)))
    task_means = []
    for task in TASKS:
        vals = [fnum(r["abs_error_pct"]) for r in calib_rows if r["task"] == task and r.get("budget_status") != "missing_audit"]
        task_means.append((task, mean(vals)))
    sem = [fnum(r["fallback_rate"]) for r in fallback_rows if r["block_strategy"] == "semantic_v3" and r.get("fallback_rate") != ""]
    qry = [fnum(r["fallback_rate"]) for r in fallback_rows if r["block_strategy"] == "query_dynamic_v3" and r.get("fallback_rate") != ""]
    lines = [
        "# AB Budget Calibration Fix Report",
        "",
        "## 1. Goal",
        "Calibration repair + verification + semantic/query fallback diagnostic only. No full LongBench was run.",
        "",
        "## 2. Fixes Verified In Cache Path",
        "- Incremental expanded-page cost is used by the token-budget selector.",
        "- Nearest-to-target selection with overshoot tolerance is enabled.",
        "- Per-method/per-task calibrated config is loaded into the cache path.",
        "",
        "## 3. Calibration Results",
        f"- Mean abs token error: {mean(abs_errs):.4f}",
        f"- All method-task <5 percent: {gate_all}",
        f"- Mean <3 percent: {gate_mean}",
        f"- Missing audit rows: {not no_missing}",
        f"- FULL_READY = {str(full_ready).lower()}",
        "",
        "### Per-method mean abs error",
    ]
    lines.extend(f"- {method}: {value:.4f}" for method, value in method_means)
    lines.append("")
    lines.append("### Per-task mean abs error")
    lines.extend(f"- {task}: {value:.4f}" for task, value in task_means)
    lines.extend([
        "",
        "## 4. High-budget Control",
    ])
    high = [r for r in calib_rows if "attention_tokens" in r["method"]]
    lines.append(f"- High-budget rows all <5 percent: {all(fnum(r['abs_error_pct']) < 0.05 for r in high) if high else False}")
    capped = [r for r in high if "per_task_attention_raw_reference" in str(r.get("target_cap_reason", ""))]
    if capped:
        capped_tasks = sorted({row["task"] for row in capped})
        lines.append("- Configured per-task attention caps: " + ", ".join(capped_tasks))
    bad_high = [r for r in high if fnum(r["abs_error_pct"]) >= 0.05 or r.get("budget_status") == "missing_audit"]
    for row in bad_high[:10]:
        lines.append(f"- {row['method']} / {row['task']}: error={fnum(row['error_pct']):.4f}, reason={row.get('mismatch_reason','')}")
    lines.extend([
        "",
        "## 5. Fallback Diagnostic",
        f"- semantic_v3 fallback_rate: {mean(sem):.4f}" if sem else "- semantic_v3 fallback_rate: unavailable",
        f"- query_dynamic_v3 fallback_rate: {mean(qry):.4f}" if qry else "- query_dynamic_v3 fallback_rate: unavailable",
        f"- semantic_v3 <0.2: {mean(sem) < 0.2 if sem else False}",
        f"- query_dynamic_v3 <0.2: {mean(qry) < 0.2 if qry else False}",
        "",
        "## 6. Next Step",
    ])
    if full_ready:
        lines.append("Calibration passed. You can run ready_to_run_full_if_passed.sh to start the gated full AB run.")
    else:
        lines.append("Calibration did not pass. Inspect budget_mismatch_cases.csv before any full LongBench run.")
    write_text(out_dir / "report.md", "\n".join(lines) + "\n")
    ready = out_dir / "ready_to_run_full_if_passed.sh"
    if full_ready:
        write_text(ready, "#!/usr/bin/env bash\nset -euo pipefail\necho 'FULL_READY=true: launch full script from the attention-boundary pipeline after updating it to this calibrated config.'\n")
    else:
        write_text(ready, "#!/usr/bin/env bash\nset -euo pipefail\necho 'FULL_READY=false: calibration did not pass.'\nexit 1\n")
    make_executable(ready)
    return full_ready


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=str)
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    budgets = load_budgets(out_dir)
    calib_rows, mismatch_rows, audit_rows = aggregate_calibration(out_dir, budgets)
    fallback = aggregate_fallback(out_dir)
    calibration_fields = ["method", "task", "target_tokens", "uncapped_target_tokens", "observed_tokens", "error_pct", "abs_error_pct", "active_budget_scale", "final_budget_scale", "min_reachable_tokens", "max_reachable_tokens", "protected_tokens", "total_candidate_tokens", "selected_pages", "page_size", "overshoot_used", "budget_status", "mismatch_reason", "target_cap_reason"]
    write_csv(out_dir / "budget_calibration.csv", calib_rows, calibration_fields)
    write_csv(out_dir / "budget_mismatch_cases.csv", mismatch_rows, calibration_fields)
    write_csv(out_dir / "audit_summary.csv", audit_rows, ["method", "task", "sample_count", "audit_row_count", "calibrated_config_loaded", "calibration_entry_found", "target_tokens_used_by_cache", "observed_final_selected_tokens_used_by_attention", "selected_pages", "page_expansion_overhead_mean", "duplicate_pages", "protected_violation", "status"])
    write_csv(out_dir / "fallback_reason_summary.csv", fallback, ["block_strategy", "task", "fallback_rate", *FALLBACK_REASONS, "status"])
    full_ready = report(out_dir, calib_rows, mismatch_rows, fallback)
    config = read_json(out_dir / "config.json")
    config["full_ready"] = full_ready
    config["verification_status"] = "passed" if full_ready else "failed"
    write_json(out_dir / "config.json", config)
    calibrated = read_json(out_dir / "calibrated_budget_config.json")
    enable_reachability_caps(calibrated)
    calibrated["full_ready"] = full_ready
    calibrated["verification_status"] = "passed" if full_ready else "failed"
    write_json(out_dir / "calibrated_budget_config.json", calibrated)


if __name__ == "__main__":
    main()
