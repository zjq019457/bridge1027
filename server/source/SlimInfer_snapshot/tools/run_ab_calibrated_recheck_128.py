#!/usr/bin/env python3
"""Prepare and finalize the calibrated six-method AB 128-sample recheck."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import shlex
import shutil
import stat
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results"
DEFAULT_MODEL = (
    "/home/yu-bai/.cache/huggingface/hub/"
    "models--Qwen--Qwen2.5-7B-Instruct/snapshots/"
    "a09a35458c702b33eeacc393d103063234e8bc28"
)
DEFAULT_LONGBENCH = "/home/yu-bai/Desktop/jiaqizhang/HeadKV/data/LongBench"
DEFAULT_CALIBRATION_DIR = RESULTS / "rescuekv_ab_budget_calibration_attention_caps_20260607_181713"
TASKS = ["hotpotqa", "2wikimqa", "musique", "qasper", "multifieldqa_en"]
METHODS = [
    {
        "method": "fixed_sliminfer_matched_K",
        "label": "Fixed SlimInfer matched K",
        "family": "sliminfer_matched_K",
        "paired_baseline": "fixed_sliminfer_matched_K",
        "target_source": "fixed_K",
        "budget_kind": "active",
        "args": [
            "--block_strategy", "fixed",
            "--rescue_policy", "none",
            "--budget_mode", "token_budget",
            "--target_token_budget_source", "fixed_K",
            "--target_token_budget", "6040",
            "--target_active_tokens", "6040",
        ],
    },
    {
        "method": "attention_boundary_sliminfer_matched_K",
        "label": "AB-SlimInfer matched K",
        "family": "sliminfer_matched_K",
        "paired_baseline": "fixed_sliminfer_matched_K",
        "target_source": "fixed_K",
        "budget_kind": "active",
        "args": [
            "--block_strategy", "attention_boundary",
            "--rescue_policy", "none",
            "--budget_mode", "token_budget",
            "--target_token_budget_source", "fixed_K",
            "--target_token_budget", "6040",
            "--target_active_tokens", "6040",
        ],
    },
    {
        "method": "fixed_drm_add_matched_KplusM",
        "label": "Fixed DRM-Add matched K+M",
        "family": "drm_matched_KplusM",
        "paired_baseline": "fixed_drm_add_matched_KplusM",
        "target_source": "fixed_KplusM_drm",
        "budget_kind": "final",
        "args": [
            "--block_strategy", "fixed",
            "--rescue_policy", "add",
            "--rescue_score", "drm",
            "--budget_mode", "token_budget",
            "--target_token_budget", "6040",
            "--active_budget_mode", "token_budget",
            "--target_active_tokens", "6040",
            "--rescue_budget_mode", "token_budget",
            "--target_final_tokens_source", "fixed_KplusM_drm",
            "--target_final_tokens", "6342",
        ],
    },
    {
        "method": "attention_boundary_boundary_drm_matched_KplusM",
        "label": "AB-BoundaryDRM matched K+M",
        "family": "drm_matched_KplusM",
        "paired_baseline": "fixed_drm_add_matched_KplusM",
        "target_source": "fixed_KplusM_drm",
        "budget_kind": "final",
        "args": [
            "--block_strategy", "attention_boundary",
            "--rescue_policy", "add",
            "--rescue_score", "boundary_drm",
            "--budget_mode", "token_budget",
            "--target_token_budget", "6040",
            "--active_budget_mode", "token_budget",
            "--target_active_tokens", "6040",
            "--rescue_budget_mode", "token_budget",
            "--target_final_tokens_source", "fixed_KplusM_drm",
            "--target_final_tokens", "6342",
        ],
    },
    {
        "method": "fixed_sliminfer_matched_attention_tokens",
        "label": "Fixed SlimInfer matched attention tokens",
        "family": "sliminfer_matched_attention_tokens",
        "paired_baseline": "fixed_sliminfer_matched_attention_tokens",
        "target_source": "attention_raw",
        "budget_kind": "active",
        "args": [
            "--block_strategy", "fixed",
            "--rescue_policy", "none",
            "--budget_mode", "token_budget",
            "--target_token_budget_source", "attention_raw",
            "--target_token_budget", "9222",
            "--target_active_tokens", "9222",
        ],
    },
    {
        "method": "attention_boundary_sliminfer_matched_attention_tokens",
        "label": "AB-SlimInfer matched attention tokens",
        "family": "sliminfer_matched_attention_tokens",
        "paired_baseline": "fixed_sliminfer_matched_attention_tokens",
        "target_source": "attention_raw",
        "budget_kind": "active",
        "args": [
            "--block_strategy", "attention_boundary",
            "--rescue_policy", "none",
            "--budget_mode", "token_budget",
            "--target_token_budget_source", "attention_raw",
            "--target_token_budget", "9222",
            "--target_active_tokens", "9222",
        ],
    },
]
METHOD_BY_NAME = {spec["method"]: spec for spec in METHODS}
FIXED_DRM = "fixed_drm_add_matched_KplusM"
AB_DRM = "attention_boundary_boundary_drm_matched_KplusM"
FIXED_SLIM = "fixed_sliminfer_matched_K"
AB_SLIM = "attention_boundary_sliminfer_matched_K"
FIXED_ATTN = "fixed_sliminfer_matched_attention_tokens"
AB_ATTN = "attention_boundary_sliminfer_matched_attention_tokens"
SUMMARY_FIELDS = [
    "rank",
    "method",
    "label",
    "family",
    "avg_score",
    *TASKS,
    "total_quality_samples",
    "budget_mean_abs_error",
    "budget_max_abs_error",
    "budget_all_lt_5pct",
    "budget_mean_lt_3pct",
    "calibrated_config_loaded",
    "calibration_entry_found",
    "status",
]
BUDGET_FIELDS = [
    "method",
    "task",
    "audit_rows",
    "audit_samples",
    "configured_target_tokens",
    "effective_target_tokens",
    "observed_tokens",
    "error_pct",
    "abs_error_pct",
    "min_reachable_tokens",
    "max_reachable_tokens",
    "overshoot_rate",
    "calibrated_config_loaded",
    "calibration_entry_found",
    "target_cap_reason",
    "status",
]


def fnum(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except Exception:
        return default


def mean(values: Iterable[float]) -> float:
    vals = [float(value) for value in values if value is not None and not math.isnan(float(value))]
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
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, Any]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def command(parts: List[Any]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def make_executable(path: Path) -> None:
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def task_entry(calibration: Dict[str, Any], method: str, task: str) -> Dict[str, Any]:
    methods = calibration.get("methods", {}) if isinstance(calibration.get("methods"), dict) else {}
    method_entry = methods.get(method, {}) if isinstance(methods.get(method), dict) else {}
    tasks = method_entry.get("tasks", {}) if isinstance(method_entry.get("tasks"), dict) else {}
    entry = {key: value for key, value in method_entry.items() if key != "tasks"}
    if isinstance(tasks.get(task), dict):
        entry.update(tasks[task])
    return entry


def dedupe_audit(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
    for index, row in enumerate(rows):
        key = (
            str(row.get("task", "")),
            str(row.get("sample_id", row.get("sample_index", index))),
            str(row.get("layer_id", "")),
            str(row.get("phase", "")),
        )
        deduped[key] = row
    return list(deduped.values())


def fair_target(row: Dict[str, Any], configured_target: float) -> float:
    calibration_target = fnum(row.get("calibration_target_tokens"), configured_target)
    runtime_target = fnum(
        row.get("effective_target_tokens", row.get("target_tokens_used_by_cache")),
        calibration_target,
    )
    cap_reason = str(row.get("target_cap_reason", "") or "")
    if "candidate_token_ceiling" in cap_reason or "protected_token_floor" in cap_reason:
        return runtime_target
    return calibration_target or runtime_target or configured_target


def prepare(args: argparse.Namespace) -> Path:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ["runs", "audit"]:
        (out_dir / name).mkdir(exist_ok=True)

    source_calibration = Path(args.calibrated_config)
    if not source_calibration.exists():
        raise FileNotFoundError(f"Missing calibrated config: {source_calibration}")
    calibration = read_json(source_calibration)
    if not calibration.get("full_ready"):
        raise RuntimeError("Calibrated config is not marked FULL_READY.")
    for spec in METHODS:
        for task in TASKS:
            entry = task_entry(calibration, spec["method"], task)
            if not entry:
                raise RuntimeError(f"Missing calibrated entry: {spec['method']} / {task}")
            if not entry.get("cap_to_reachable_tokens"):
                raise RuntimeError(
                    f"Reachability cap is disabled: {spec['method']} / {task}"
                )

    local_calibration = out_dir / "calibrated_budget_config.json"
    shutil.copy2(source_calibration, local_calibration)
    config = {
        "stage": "ab_calibrated_recheck_128",
        "created_at": dt.datetime.now().astimezone().isoformat(),
        "tasks": TASKS,
        "samples_per_task": 128,
        "methods": [spec["method"] for spec in METHODS],
        "generation_max_new_tokens": args.max_new_tokens,
        "model_path": args.model_path,
        "longbench_dir": args.longbench_dir,
        "source_calibrated_config": str(source_calibration.resolve()),
        "local_calibrated_config": str(local_calibration.resolve()),
        "calibrated_config_sha256": sha256(local_calibration),
        "calibration_full_ready": True,
        "quality_gate": {
            "clear_avg_gain": args.clear_avg_gain,
            "no_task_regression": True,
            "budget_all_method_task_lt": 0.05,
            "budget_mean_lt": 0.03,
        },
        "run_full_longbench": False,
        "run_latency": False,
        "status": "prepared",
    }
    write_json(out_dir / "config.json", config)

    common = [
        sys.executable,
        str(REPO_ROOT / "experiments" / "run_qwen25_sliminfer_variants.py"),
        "--model_family", "qwen2",
        "--dtype", "fp16",
        "--device", "cuda",
        "--backend", "flash_attention_2",
        "--method", "sliminfer_variant",
        "--scoring", "token_unit_qk",
        "--eval", "longbench",
        "--model_path", args.model_path,
        "--tokenizer_path", args.model_path,
        "--longbench_dir", args.longbench_dir,
        "--longbench_tasks", ",".join(TASKS),
        "--max_samples_per_task", "128",
        "--generation_max_new_tokens", str(args.max_new_tokens),
        "--smoke_mode", "1",
        "--max_smoke_samples", "128",
        "--rescue_ratio", "0.05",
        "--alpha_diff", "0.5",
        "--lambda_red", "0.25",
        "--eta_rescue", "0.25",
        "--eta_active", "0.25",
        "--gamma_boundary", "0.25",
        "--block_size", "64",
        "--token_unit_size", "8",
        "--prune_layers", "9,18,26",
        "--retain_tokens", "12288,6144,4096",
        "--temperature", "0",
        "--top_p", "1.0",
        "--max_model_len", "32768",
        "--selection_cost", "expanded_page_tokens",
        "--score_density_mode", "sqrt_len",
        "--allow_budget_overshoot", "1",
        "--soft_budget_tolerance", "0.03",
        "--hard_budget_tolerance", "0.05",
        "--calibrated_budget_config", str(local_calibration.resolve()),
        "--save_metrics_json",
        "--save_outputs",
    ]
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {shlex.quote(str(REPO_ROOT))}",
        "export TRANSFORMERS_VERBOSITY=error",
        "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
        f"mkdir -p {shlex.quote(str(out_dir / 'runs'))} {shlex.quote(str(out_dir / 'audit'))}",
        "",
    ]
    for spec in METHODS:
        method = spec["method"]
        cmd = common + [
            "--output_dir", str((out_dir / "runs" / method).resolve()),
            "--isolated_run_id", method,
            "--rescue_audit_path", str((out_dir / "audit" / f"{method}.jsonl").resolve()),
        ] + spec["args"]
        lines.append(f"echo '[ab-recheck-128] starting {method}'")
        lines.append(command(cmd))
    lines.append(command([
        sys.executable,
        str(Path(__file__).resolve()),
        "--finalize",
        "--output_dir", str(out_dir.resolve()),
        "--clear_avg_gain", str(args.clear_avg_gain),
    ]))
    run_script = out_dir / "run_ab_recheck_128.sh"
    run_script.write_text("\n".join(lines) + "\n", encoding="utf-8")
    make_executable(run_script)
    resume = out_dir / "resume.sh"
    resume.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        f"bash {shlex.quote(str(run_script.resolve()))}\n",
        encoding="utf-8",
    )
    make_executable(resume)

    write_csv(out_dir / "ab_recheck_128_summary.csv", [], SUMMARY_FIELDS)
    write_csv(out_dir / "ab_recheck_128_ranking.csv", [], SUMMARY_FIELDS)
    write_csv(
        out_dir / "ab_recheck_128_task_deltas.csv",
        [],
        ["method", "task", "score", "paired_baseline", "paired_baseline_score", "delta_vs_paired", "fixed_drm_score", "delta_vs_fixed_drm"],
    )
    write_csv(out_dir / "ab_recheck_128_budget_audit.csv", [], BUDGET_FIELDS)
    (out_dir / "report_ab_recheck_128.md").write_text(
        "# Calibrated AB 128-sample Recheck\n\n"
        "Prepared but not yet run. Full LongBench and latency are disabled.\n",
        encoding="utf-8",
    )
    print(out_dir)
    return out_dir


def aggregate_budget(out_dir: Path, calibration: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for spec in METHODS:
        method = spec["method"]
        audit = dedupe_audit([
            row
            for row in read_jsonl(out_dir / "audit" / f"{method}.jsonl")
            if row.get("phase") == "prefill"
        ])
        for task in TASKS:
            task_rows = [row for row in audit if row.get("task") == task]
            entry = task_entry(calibration, method, task)
            configured_target = fnum(entry.get("target_tokens"))
            effective_target = mean(fair_target(row, configured_target) for row in task_rows) if task_rows else configured_target
            observed = mean(
                fnum(row.get("observed_final_selected_tokens_used_by_attention", row.get("selected_token_count")))
                for row in task_rows
            )
            error = (observed - effective_target) / effective_target if effective_target > 0 else 0.0
            cap_reasons = sorted({
                reason
                for row in task_rows
                for reason in str(row.get("target_cap_reason", "") or "").split(",")
                if reason
            })
            loaded = mean(1.0 if row.get("calibrated_budget_config_loaded") else 0.0 for row in task_rows) if task_rows else 0.0
            found = mean(1.0 if row.get("calibration_entry_found") else 0.0 for row in task_rows) if task_rows else 0.0
            status = "ok"
            if not task_rows:
                status = "missing_audit"
            elif loaded < 1.0 or found < 1.0:
                status = "calibration_not_loaded"
            elif abs(error) >= 0.05:
                status = "budget_mismatch"
            rows.append({
                "method": method,
                "task": task,
                "audit_rows": len(task_rows),
                "audit_samples": len({
                    str(row.get("sample_id", row.get("sample_index", "")))
                    for row in task_rows
                }),
                "configured_target_tokens": configured_target,
                "effective_target_tokens": effective_target,
                "observed_tokens": observed,
                "error_pct": error,
                "abs_error_pct": abs(error),
                "min_reachable_tokens": mean(fnum(row.get("min_reachable_tokens")) for row in task_rows) if task_rows else "",
                "max_reachable_tokens": mean(fnum(row.get("max_reachable_tokens")) for row in task_rows) if task_rows else "",
                "overshoot_rate": mean(1.0 if row.get("overshoot_used") else 0.0 for row in task_rows) if task_rows else "",
                "calibrated_config_loaded": loaded,
                "calibration_entry_found": found,
                "target_cap_reason": ",".join(cap_reasons),
                "status": status,
            })
    return rows


def aggregate_quality(out_dir: Path, budget_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for spec in METHODS:
        method = spec["method"]
        score_rows = read_csv(out_dir / "runs" / method / "longbench_scores.csv")
        scores = {
            row.get("dataset", ""): fnum(row.get("score"))
            for row in score_rows
            if row.get("status") == "ok"
        }
        sample_counts = {
            row.get("dataset", ""): int(fnum(row.get("num_samples")))
            for row in score_rows
            if row.get("status") == "ok"
        }
        method_budget = [row for row in budget_rows if row["method"] == method]
        budget_errors = [fnum(row["abs_error_pct"]) for row in method_budget if row["status"] != "missing_audit"]
        complete_quality = all(task in scores and sample_counts.get(task) == 128 for task in TASKS)
        complete_budget = len(method_budget) == len(TASKS) and all(row["status"] == "ok" for row in method_budget)
        status = "ok" if complete_quality and complete_budget else "incomplete_or_failed"
        row = {
            "rank": "",
            "method": method,
            "label": spec["label"],
            "family": spec["family"],
            "avg_score": mean(scores.get(task, 0.0) for task in TASKS) if complete_quality else "",
            **{task: scores.get(task, "") for task in TASKS},
            "total_quality_samples": sum(sample_counts.get(task, 0) for task in TASKS),
            "budget_mean_abs_error": mean(budget_errors) if budget_errors else "",
            "budget_max_abs_error": max(budget_errors) if budget_errors else "",
            "budget_all_lt_5pct": bool(budget_errors) and all(error < 0.05 for error in budget_errors),
            "budget_mean_lt_3pct": bool(budget_errors) and mean(budget_errors) < 0.03,
            "calibrated_config_loaded": min(
                (fnum(item["calibrated_config_loaded"]) for item in method_budget),
                default=0.0,
            ),
            "calibration_entry_found": min(
                (fnum(item["calibration_entry_found"]) for item in method_budget),
                default=0.0,
            ),
            "status": status,
        }
        rows.append(row)
    ranked = sorted(
        rows,
        key=lambda row: fnum(row["avg_score"], -1e9),
        reverse=True,
    )
    for rank, row in enumerate(ranked, 1):
        row["rank"] = rank
    return rows


def task_deltas(summary: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_method = {row["method"]: row for row in summary}
    fixed_drm = by_method.get(FIXED_DRM, {})
    rows: List[Dict[str, Any]] = []
    for row in summary:
        method = row["method"]
        paired = METHOD_BY_NAME[method]["paired_baseline"]
        paired_row = by_method.get(paired, {})
        for task in TASKS:
            score = fnum(row.get(task))
            paired_score = fnum(paired_row.get(task))
            fixed_score = fnum(fixed_drm.get(task))
            rows.append({
                "method": method,
                "task": task,
                "score": score,
                "paired_baseline": paired,
                "paired_baseline_score": paired_score,
                "delta_vs_paired": score - paired_score,
                "fixed_drm_score": fixed_score,
                "delta_vs_fixed_drm": score - fixed_score,
            })
    return rows


def comparison(summary: Dict[str, Dict[str, Any]], candidate: str, baseline: str) -> Dict[str, Any]:
    cand = summary.get(candidate, {})
    base = summary.get(baseline, {})
    task_delta = {
        task: fnum(cand.get(task)) - fnum(base.get(task))
        for task in TASKS
    }
    return {
        "candidate": candidate,
        "baseline": baseline,
        "avg_delta": fnum(cand.get("avg_score")) - fnum(base.get("avg_score")),
        "task_delta": task_delta,
        "no_task_regression": all(delta >= 0.0 for delta in task_delta.values()),
    }


def write_full_script(
    out_dir: Path,
    config: Dict[str, Any],
    calibration_path: Path,
    enabled: bool,
) -> None:
    script = out_dir / "run_ab_full_longbench.sh"
    if not enabled:
        if script.exists():
            script.unlink()
        return
    args = METHOD_BY_NAME[AB_DRM]["args"]
    cmd = [
        sys.executable,
        str(REPO_ROOT / "experiments" / "run_qwen25_sliminfer_variants.py"),
        "--model_family", "qwen2",
        "--dtype", "fp16",
        "--device", "cuda",
        "--backend", "flash_attention_2",
        "--method", "sliminfer_variant",
        "--scoring", "token_unit_qk",
        "--eval", "longbench",
        "--model_path", config["model_path"],
        "--tokenizer_path", config["model_path"],
        "--longbench_dir", config["longbench_dir"],
        "--longbench_tasks",
        "qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,gov_report,qmsum,multi_news,vcsum,trec,triviaqa,samsum,lsht,passage_count,passage_retrieval_en,lcc,repobench-p",
        "--generation_max_new_tokens", "16",
        "--rescue_ratio", "0.05",
        "--alpha_diff", "0.5",
        "--lambda_red", "0.25",
        "--eta_rescue", "0.25",
        "--eta_active", "0.25",
        "--gamma_boundary", "0.25",
        "--block_size", "64",
        "--token_unit_size", "8",
        "--prune_layers", "9,18,26",
        "--retain_tokens", "12288,6144,4096",
        "--temperature", "0",
        "--top_p", "1.0",
        "--max_model_len", "32768",
        "--selection_cost", "expanded_page_tokens",
        "--score_density_mode", "sqrt_len",
        "--allow_budget_overshoot", "1",
        "--soft_budget_tolerance", "0.03",
        "--hard_budget_tolerance", "0.05",
        "--calibrated_budget_config", str(calibration_path.resolve()),
        "--output_dir", str((out_dir / "full_longbench" / AB_DRM).resolve()),
        "--isolated_run_id", AB_DRM,
        "--rescue_audit_path", str((out_dir / "full_longbench" / "audit" / f"{AB_DRM}.jsonl").resolve()),
        "--save_metrics_json",
        "--save_outputs",
    ] + args
    script.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        f"cd {shlex.quote(str(REPO_ROOT))}\n"
        "# Prepared by the 128-sample gate. This file is not launched automatically.\n"
        + command(cmd)
        + "\n",
        encoding="utf-8",
    )
    make_executable(script)


def write_report(
    out_dir: Path,
    summary_rows: List[Dict[str, Any]],
    budget_rows: List[Dict[str, Any]],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    by_method = {row["method"]: row for row in summary_rows}
    slim = comparison(by_method, AB_SLIM, FIXED_SLIM)
    drm = comparison(by_method, AB_DRM, FIXED_DRM)
    attention = comparison(by_method, AB_ATTN, FIXED_ATTN)
    budget_errors = [fnum(row["abs_error_pct"]) for row in budget_rows]
    budget_pass = (
        len(budget_rows) == len(METHODS) * len(TASKS)
        and all(row["status"] == "ok" for row in budget_rows)
        and mean(budget_errors) < 0.03
        and all(error < 0.05 for error in budget_errors)
    )
    quality_complete = all(row["status"] == "ok" for row in summary_rows)
    clear_gain = drm["avg_delta"] >= fnum(config.get("quality_gate", {}).get("clear_avg_gain"), 0.5)
    full_recommended = bool(
        quality_complete
        and budget_pass
        and drm["avg_delta"] > 0.0
        and clear_gain
        and drm["no_task_regression"]
    )
    budget_only = attention["avg_delta"] <= 0.0 and slim["avg_delta"] <= 0.0
    abnormal_tasks = []
    for task in ["qasper", "2wikimqa", "multifieldqa_en"]:
        task_budget = [row for row in budget_rows if row["task"] == task]
        task_quality_delta = drm["task_delta"][task]
        if any(fnum(row["abs_error_pct"]) >= 0.05 for row in task_budget) or task_quality_delta < 0.0:
            abnormal_tasks.append(task)

    ranking = sorted(summary_rows, key=lambda row: fnum(row["avg_score"], -1e9), reverse=True)
    lines = [
        "# Calibrated AB 128-sample Recheck",
        "",
        "## Setup",
        "",
        f"- Tasks: {', '.join(TASKS)}; 128 samples per task.",
        "- Methods: six calibrated fixed/attention-boundary fairness controls only.",
        f"- Generation cap: {config.get('generation_max_new_tokens', 16)} tokens, matching the earlier AB diagnostic protocol.",
        f"- Calibrated config: `{config.get('source_calibrated_config', '')}`.",
        "- No full LongBench and no latency run were started.",
        "",
        "## Quality Ranking",
        "",
    ]
    for row in ranking:
        lines.append(
            f"- {row['rank']}. `{row['method']}`: Avg {fnum(row['avg_score']):.3f}; "
            f"budget mean abs {fnum(row['budget_mean_abs_error']):.2%}; "
            f"budget max {fnum(row['budget_max_abs_error']):.2%}."
        )
    lines += [
        "",
        "## Required Answers",
        "",
        f"1. Calibration still passes at 128 samples: **{budget_pass}**. "
        f"Mean abs error {mean(budget_errors):.2%}; max method-task error {max(budget_errors, default=0.0):.2%}.",
        f"2. AB-SlimInfer exceeds fixed SlimInfer: **{slim['avg_delta'] > 0.0}** "
        f"(Avg delta {slim['avg_delta']:+.3f}).",
        f"3. AB-BoundaryDRM exceeds fixed DRM-Add: **{drm['avg_delta'] > 0.0}** "
        f"(Avg delta {drm['avg_delta']:+.3f}; no task regression: {drm['no_task_regression']}).",
        f"4. matched_attention_tokens indicates a budget-only AB effect: **{budget_only}**. "
        f"At the same high token target, AB delta is {attention['avg_delta']:+.3f}; "
        f"at matched K, AB delta is {slim['avg_delta']:+.3f}.",
        f"5. qasper / 2wikimqa / multifieldqa_en still abnormal: **{bool(abnormal_tasks)}** "
        f"({', '.join(abnormal_tasks) if abnormal_tasks else 'none'}).",
        f"6. Recommend AB full LongBench: **{full_recommended}**. "
        f"Gate requires budget pass, Avg gain >= {fnum(config.get('quality_gate', {}).get('clear_avg_gain'), 0.5):.2f}, "
        "and non-negative AB-BoundaryDRM delta on every task.",
        f"7. Keep AB as a negative diagnostic if full is not recommended: **{not full_recommended}**.",
        "",
        "## AB-BoundaryDRM Task Deltas",
        "",
    ]
    lines.extend(f"- {task}: {delta:+.3f}" for task, delta in drm["task_delta"].items())
    lines += [
        "",
        "## Decision",
        "",
    ]
    if full_recommended:
        lines.append(
            "`run_ab_full_longbench.sh` was generated as a prepared command only. It was not launched."
        )
    elif drm["avg_delta"] <= 0.0:
        lines.append(
            "AB-BoundaryDRM did not exceed fixed DRM-Add at 128 samples. Do not continue to AB full; retain AB as a negative diagnostic."
        )
    else:
        lines.append(
            "AB-BoundaryDRM did not satisfy the clear-gain/no-regression promotion gate. Do not continue to AB full; retain AB as a negative diagnostic."
        )
    lines += [
        "",
        "## Artifacts",
        "",
        "- `ab_recheck_128_summary.csv`: quality and budget summary.",
        "- `ab_recheck_128_ranking.csv`: quality ranking.",
        "- `ab_recheck_128_task_deltas.csv`: per-task paired and fixed-DRM deltas.",
        "- `ab_recheck_128_budget_audit.csv`: method-task cache budget audit.",
    ]
    (out_dir / "report_ab_recheck_128.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "budget_pass": budget_pass,
        "quality_complete": quality_complete,
        "ab_sliminfer_delta": slim["avg_delta"],
        "ab_boundary_drm_delta": drm["avg_delta"],
        "ab_boundary_drm_no_task_regression": drm["no_task_regression"],
        "matched_attention_delta": attention["avg_delta"],
        "matched_attention_budget_only": budget_only,
        "abnormal_tasks": abnormal_tasks,
        "full_recommended": full_recommended,
    }


def finalize(args: argparse.Namespace) -> Path:
    out_dir = Path(args.output_dir)
    config_path = out_dir / "config.json"
    config = read_json(config_path)
    if not config:
        raise FileNotFoundError(f"Missing config: {config_path}")
    calibration_path = out_dir / "calibrated_budget_config.json"
    calibration = read_json(calibration_path)
    budget_rows = aggregate_budget(out_dir, calibration)
    summary_rows = aggregate_quality(out_dir, budget_rows)
    ranking = sorted(summary_rows, key=lambda row: fnum(row["avg_score"], -1e9), reverse=True)
    deltas = task_deltas(summary_rows)
    write_csv(out_dir / "ab_recheck_128_summary.csv", summary_rows, SUMMARY_FIELDS)
    write_csv(out_dir / "ab_recheck_128_ranking.csv", ranking, SUMMARY_FIELDS)
    write_csv(
        out_dir / "ab_recheck_128_task_deltas.csv",
        deltas,
        ["method", "task", "score", "paired_baseline", "paired_baseline_score", "delta_vs_paired", "fixed_drm_score", "delta_vs_fixed_drm"],
    )
    write_csv(out_dir / "ab_recheck_128_budget_audit.csv", budget_rows, BUDGET_FIELDS)
    decision = write_report(out_dir, summary_rows, budget_rows, config)
    config["status"] = "complete" if decision["quality_complete"] else "incomplete"
    config["decision"] = decision
    config["run_full_longbench"] = False
    config["run_latency"] = False
    write_json(config_path, config)
    write_full_script(out_dir, config, calibration_path, decision["full_recommended"])
    print(out_dir)
    return out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--model_path", default=DEFAULT_MODEL)
    parser.add_argument("--longbench_dir", default=DEFAULT_LONGBENCH)
    parser.add_argument(
        "--calibrated_config",
        default=str(DEFAULT_CALIBRATION_DIR / "calibrated_budget_config.json"),
    )
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--clear_avg_gain", type=float, default=0.5)
    args = parser.parse_args()
    if not args.output_dir:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = str(RESULTS / f"ab_calibrated_recheck_128_{stamp}")
    if not args.prepare and not args.finalize:
        args.prepare = True
    return args


def main() -> None:
    args = parse_args()
    if args.prepare:
        prepare(args)
    if args.finalize:
        finalize(args)


if __name__ == "__main__":
    main()
