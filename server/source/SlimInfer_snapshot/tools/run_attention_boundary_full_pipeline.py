#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
import stat
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
DIAG_TASKS = ["hotpotqa", "2wikimqa", "musique", "qasper", "multifieldqa_en"]
FULL_TASKS = [
    "qasper",
    "multifieldqa_en",
    "hotpotqa",
    "2wikimqa",
    "musique",
    "gov_report",
    "qmsum",
    "multi_news",
    "vcsum",
    "trec",
    "triviaqa",
    "samsum",
    "lsht",
    "passage_count",
    "passage_retrieval_en",
    "lcc",
    "repobench-p",
]
FULL_METHODS_TO_REUSE = [
    "FullKV",
    "SlimInfer",
    "SlimInfer larger K+M",
    "Random-Add",
    "DeltaO-Add",
    "DRM-Add",
]
AB_FULL_METHODS = [
    "AB-SlimInfer",
    "AB-SlimInfer larger K+M",
    "AB-DRM",
    "AB-BoundaryDRM (ours)",
]
CALIBRATION_METHODS = [
    ("fixed_sliminfer_matched_K", "fixed_K", "active"),
    ("attention_boundary_sliminfer_matched_K", "fixed_K", "active"),
    ("fixed_drm_add_matched_KplusM", "fixed_KplusM_drm", "final"),
    ("attention_boundary_boundary_drm_matched_KplusM", "fixed_KplusM_drm", "final"),
    ("fixed_sliminfer_matched_attention_tokens", "attention_raw", "active"),
    ("attention_boundary_sliminfer_matched_attention_tokens", "attention_raw", "active"),
]


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, Any]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def make_executable(path: Path) -> None:
    mode = path.stat().st_mode
    path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def latest_dir(pattern: str) -> Optional[Path]:
    dirs = sorted((p for p in RESULTS_DIR.glob(pattern) if p.is_dir()), key=lambda p: p.name, reverse=True)
    return dirs[0] if dirs else None


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except Exception:
        return default


def as_int(value: Any, default: int = 0) -> int:
    try:
        if value in (None, ""):
            return default
        return int(round(float(value)))
    except Exception:
        return default


def mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    return sum(vals) / len(vals) if vals else 0.0


def index_by(rows: List[Dict[str, str]], key: str) -> Dict[str, Dict[str, str]]:
    return {str(row.get(key, "")): row for row in rows}


def extract_previous_context(fair_dir: Optional[Path], abcd_dir: Optional[Path], full_dir: Path) -> Dict[str, Any]:
    fair_summary = read_csv((fair_dir or Path("")) / "fair_summary.csv") if fair_dir else []
    summary = read_csv((fair_dir or Path("")) / "summary.csv") if fair_dir else []
    fallback = read_csv((fair_dir or Path("")) / "fallback_reason_summary.csv") if fair_dir else []
    abcd_layout = read_csv((abcd_dir or Path("")) / "block_layout_summary.csv") if abcd_dir else []
    main_table = read_csv(full_dir / "main_table.csv")
    category_table = read_csv(full_dir / "category_table.csv")

    fair_by_method = index_by(fair_summary, "method")
    budgets = {
        "fixed_K": as_float(fair_by_method.get("fixed_sliminfer_original", {}).get("selected_token_count_mean"), 6040.0),
        "fixed_KplusM_drm": as_float(fair_by_method.get("fixed_drm_add", {}).get("selected_token_count_mean"), 6342.0),
        "fixed_KplusM_larger": as_float(fair_by_method.get("fixed_sliminfer_larger", {}).get("selected_token_count_mean"), 6323.0),
        "attention_raw": as_float(fair_by_method.get("attention_boundary_sliminfer_raw", {}).get("selected_token_count_mean"), 9222.0),
        "attention_boundary_boundary_drm": as_float(fair_by_method.get("attention_boundary_boundary_drm_matched_KplusM", {}).get("selected_token_count_mean"), 0.0),
    }
    if budgets["fixed_KplusM_larger"] <= 0:
        # The fair-hybrid run did not include fixed larger. The ABCD layout summary
        # did, so use it as a fallback reference only.
        for row in abcd_layout:
            if row.get("method") == "fixed_sliminfer_larger":
                budgets["fixed_KplusM_larger"] = as_float(row.get("selected_token_count_mean"), budgets["fixed_KplusM_drm"])

    fallback_rates: Dict[str, float] = {}
    for strategy in ["semantic_v2", "query_dynamic_v2"]:
        vals = [as_float(row.get("fallback_rate")) for row in fallback if row.get("block_strategy") == strategy]
        fallback_rates[strategy] = mean(vals)

    return {
        "fair_dir": str(fair_dir) if fair_dir else "",
        "abcd_dir": str(abcd_dir) if abcd_dir else "",
        "full_dir": str(full_dir),
        "fair_summary": fair_summary,
        "summary": summary,
        "fallback": fallback,
        "main_table": main_table,
        "category_table": category_table,
        "budgets": budgets,
        "fallback_rates": fallback_rates,
    }


def derive_budget_calibration(ctx: Dict[str, Any], out_dir: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]], bool]:
    fair_rows = index_by(ctx.get("fair_summary", []), "method")
    summary_rows = ctx.get("summary", [])
    budgets = ctx["budgets"]
    calibration_rows: List[Dict[str, Any]] = []
    mismatch_rows: List[Dict[str, Any]] = []
    methods_cfg: Dict[str, Any] = {}

    for method, target_source, budget_kind in CALIBRATION_METHODS:
        target = float(budgets.get(target_source, 0.0) or 0.0)
        fair_row = fair_rows.get(method, {})
        global_observed = as_float(fair_row.get("selected_token_count_mean"), 0.0)
        global_error = (global_observed - target) / target if target > 0 else 0.0
        global_scale = target / global_observed if global_observed > 0 else 1.0
        method_entry: Dict[str, Any] = {
            "target_budget_source": target_source,
            "target_token_budget": int(round(target)),
            "budget_kind": budget_kind,
            "budget_scale": global_scale,
            "active_budget_scale": global_scale if budget_kind == "active" else 1.0,
            "final_budget_scale": global_scale if budget_kind == "final" else 1.0,
            "previous_observed_selected_tokens": global_observed,
            "previous_selected_token_error_pct": global_error,
            "calibration_source": "derived_from_previous_64_sample_diagnostic",
            "verification_status": "not_verified",
            "tasks": {},
        }
        for task in DIAG_TASKS:
            rows = [r for r in summary_rows if r.get("method") == method and r.get("task") == task]
            observed = as_float(rows[0].get("selected_token_count_mean"), global_observed) if rows else global_observed
            error = (observed - target) / target if target > 0 else 0.0
            scale = target / observed if observed > 0 else global_scale
            task_entry: Dict[str, Any] = {
                "target_token_budget": int(round(target)),
                "previous_observed_selected_tokens": observed,
                "previous_selected_token_error_pct": error,
                "budget_scale": scale,
                "active_budget_scale": scale if budget_kind == "active" else 1.0,
                "final_budget_scale": scale if budget_kind == "final" else 1.0,
                "controller_target_active_tokens": int(round(target * scale)) if budget_kind == "active" else "",
                "controller_target_final_tokens": int(round(target * scale)) if budget_kind == "final" else "",
                "max_reachable_tokens": "",
                "min_reachable_tokens": "",
                "discrete_step_size": 64,
                "budget_mismatch_reason": "previous_observation_unverified_scale" if abs(error) > 0.03 else "",
            }
            method_entry["tasks"][task] = task_entry
            calibration_rows.append({
                "method": method,
                "task": task,
                "target_budget_source": target_source,
                "target_token_budget": int(round(target)),
                "previous_observed_selected_tokens": observed,
                "previous_selected_token_error_pct": error,
                "proposed_budget_scale": scale,
                "active_budget_scale": task_entry["active_budget_scale"],
                "final_budget_scale": task_entry["final_budget_scale"],
                "controller_target_active_tokens": task_entry["controller_target_active_tokens"],
                "controller_target_final_tokens": task_entry["controller_target_final_tokens"],
                "max_reachable_tokens": "",
                "min_reachable_tokens": "",
                "discrete_step_size": 64,
                "budget_mismatch_reason": task_entry["budget_mismatch_reason"],
                "status": "proposed_unverified",
            })
            if abs(error) > 0.03:
                mismatch_rows.append(calibration_rows[-1])
        methods_cfg[method] = method_entry

    abs_errors = [abs(as_float(row["previous_selected_token_error_pct"])) for row in calibration_rows]
    previous_mean_error = mean(abs_errors)
    previous_per_task_pass = all(abs(as_float(row["previous_selected_token_error_pct"])) < 0.05 for row in calibration_rows)
    # Derived scales are not a verification run. Keep the full gate closed until
    # the same 64-sample calibration is re-run with calibrated_budget_config.
    calibration_pass = False
    config = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "mode": "derived_unverified",
        "selection_cost": "expanded_page_tokens",
        "score_density_mode": "sqrt_len",
        "targets": {k: int(round(v)) for k, v in budgets.items()},
        "previous_mean_abs_selected_token_error_pct": previous_mean_error,
        "previous_per_task_under_5pct": previous_per_task_pass,
        "calibration_pass": calibration_pass,
        "full_run_gate": "blocked_until_verified_under_3pct_mean_and_5pct_per_task",
        "methods": methods_cfg,
    }

    calib_dir = out_dir / "budget_calibration"
    write_json(calib_dir / "calibrated_budget_config.json", config)
    write_csv(
        calib_dir / "budget_calibration.csv",
        calibration_rows,
        [
            "method",
            "task",
            "target_budget_source",
            "target_token_budget",
            "previous_observed_selected_tokens",
            "previous_selected_token_error_pct",
            "proposed_budget_scale",
            "active_budget_scale",
            "final_budget_scale",
            "controller_target_active_tokens",
            "controller_target_final_tokens",
            "max_reachable_tokens",
            "min_reachable_tokens",
            "discrete_step_size",
            "budget_mismatch_reason",
            "status",
        ],
    )
    write_csv(
        calib_dir / "budget_mismatch_cases.csv",
        mismatch_rows,
        [
            "method",
            "task",
            "target_budget_source",
            "target_token_budget",
            "previous_observed_selected_tokens",
            "previous_selected_token_error_pct",
            "proposed_budget_scale",
            "budget_mismatch_reason",
            "status",
        ],
    )
    write_json(
        calib_dir / "budget_calibration.json",
        {
            "summary": {
                "previous_mean_abs_selected_token_error_pct": previous_mean_error,
                "previous_per_task_under_5pct": previous_per_task_pass,
                "calibration_pass": calibration_pass,
                "mismatch_case_count": len(mismatch_rows),
            },
            "rows": calibration_rows,
        },
    )
    report = [
        "# Budget Calibration Report",
        "",
        "This file derives controller scales from the previous 64-sample diagnostic.",
        "It is not a verification run, so the full LongBench gate remains closed.",
        "",
        f"- Previous mean absolute selected-token error: {previous_mean_error:.4f}",
        f"- Previous per-task under 5 percent: {previous_per_task_pass}",
        f"- Mismatch cases over 3 percent: {len(mismatch_rows)}",
        f"- High-budget target attention_raw: {budgets.get('attention_raw', 0):.2f}",
        "",
        "A verification run should re-run the six calibration methods with calibrated_budget_config.json.",
        "Full LongBench should start only after mean error is under 3 percent and per-task error is under 5 percent.",
    ]
    write_text(calib_dir / "budget_calibration_report.md", "\n".join(report) + "\n")
    return config, calibration_rows, mismatch_rows, calibration_pass


def add_placeholder_ab_rows(main_rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    if not main_rows:
        fields = [
            "Method",
            "Qasper",
            "MFQA",
            "HPQA",
            "2Wiki",
            "MuSiQue",
            "GovRep",
            "QMSum",
            "MNews",
            "VCSum",
            "TREC",
            "TQA",
            "SAMSum",
            "LSHT",
            "Count",
            "PassR",
            "LCC",
            "RepoB-p",
            "Avg",
            "SampleMode",
            "SampleCount",
            "Status",
            "E2E_32k_s_if_available",
            "TTFT_32k_s_if_available",
        ]
        main_rows = [{field: "" for field in fields}]
        main_rows = []
    fields = list(main_rows[0].keys()) if main_rows else [
        "Method",
        "Qasper",
        "MFQA",
        "HPQA",
        "2Wiki",
        "MuSiQue",
        "GovRep",
        "QMSum",
        "MNews",
        "VCSum",
        "TREC",
        "TQA",
        "SAMSum",
        "LSHT",
        "Count",
        "PassR",
        "LCC",
        "RepoB-p",
        "Avg",
        "SampleMode",
        "SampleCount",
        "Status",
        "E2E_32k_s_if_available",
        "TTFT_32k_s_if_available",
    ]
    extra = [
        "SelectedTokens_if_available",
        "SelectedPages_if_available",
        "PageExpansionOverhead_if_available",
    ]
    for field in extra:
        if field not in fields:
            fields.append(field)
    out: List[Dict[str, Any]] = []
    for row in main_rows:
        if row.get("Method") in FULL_METHODS_TO_REUSE:
            item = {field: row.get(field, "") for field in fields}
            item["Status"] = row.get("Status", "reused")
            out.append(item)
    for method in AB_FULL_METHODS:
        item = {field: "" for field in fields}
        item.update({
            "Method": method,
            "SampleMode": "full_not_run",
            "SampleCount": "0/3950",
            "Status": "not_run_calibration_unverified",
        })
        out.append(item)
    return out


def category_placeholders(category_rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    fields = ["Method", "SingleDocQA", "MultiDocQA", "Summarization", "FewShot", "Synthetic", "Code", "Avg"]
    out = [{field: row.get(field, "") for field in fields} for row in category_rows if row.get("Method") in FULL_METHODS_TO_REUSE]
    for method in AB_FULL_METHODS:
        row = {field: "" for field in fields}
        row["Method"] = method
        out.append(row)
    return out


def markdown_table(rows: List[Dict[str, Any]], fields: List[str], max_rows: Optional[int] = None) -> str:
    rows = rows[:max_rows] if max_rows else rows
    if not rows:
        return "_No rows._"
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    return "\n".join(lines)


def write_latex(path: Path, rows: List[Dict[str, Any]]) -> None:
    metric_fields = ["Method", "Qasper", "MFQA", "HPQA", "2Wiki", "MuSiQue", "GovRep", "QMSum", "MNews", "VCSum", "TREC", "TQA", "SAMSum", "LSHT", "Count", "PassR", "LCC", "RepoB-p", "Avg"]
    complete = [r for r in rows if str(r.get("Status", "")).lower() in {"ok", "reused"} and r.get("Avg")]
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{lrrrrrrrrrrrrrrrrrr}",
        "\\toprule",
        " & ".join(metric_fields) + " \\\\",
        "\\midrule",
    ]
    for row in complete:
        vals = [str(row.get("Method", ""))]
        for field in metric_fields[1:]:
            value = row.get(field, "")
            try:
                vals.append(f"{float(value):.2f}")
            except Exception:
                vals.append(str(value))
        lines.append(" & ".join(vals) + " \\\\")
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Full LongBench results. New AB methods are excluded until calibration is verified and full runs complete.}",
        "\\end{table*}",
    ])
    write_text(path, "\n".join(lines) + "\n")


def command_line(parts: List[str]) -> str:
    return " ".join(shlex.quote(str(x)) for x in parts)


def write_scripts(out_dir: Path, args: argparse.Namespace, ctx: Dict[str, Any]) -> None:
    calib = out_dir / "budget_calibration" / "calibrated_budget_config.json"
    full_dir = out_dir / "full_longbench"
    full_dir.mkdir(parents=True, exist_ok=True)
    budgets = ctx.get("budgets", {})
    fixed_k = str(int(round(float(budgets.get("fixed_K", 6040)))))
    fixed_km = str(int(round(float(budgets.get("fixed_KplusM_drm", 6342)))))
    attention_raw = str(int(round(float(budgets.get("attention_raw", 9222)))))
    common = [
        "python",
        str(REPO_ROOT / "experiments" / "run_qwen25_sliminfer_variants.py"),
        "--model_family",
        "qwen2",
        "--dtype",
        "fp16",
        "--device",
        "cuda",
        "--backend",
        "flash_attention_2",
        "--method",
        "sliminfer_variant",
        "--scoring",
        "token_unit_qk",
        "--eval",
        "longbench",
        "--model_path",
        args.model_path,
        "--tokenizer_path",
        args.model_path,
        "--longbench_dir",
        args.longbench_dir,
        "--longbench_tasks",
        ",".join(FULL_TASKS),
        "--decode_tokens",
        str(args.max_new_tokens),
        "--generation_max_new_tokens",
        str(args.max_new_tokens),
        "--max_smoke_samples",
        "0",
        "--rescue_ratio",
        "0.05",
        "--alpha_diff",
        "0.5",
        "--lambda_red",
        "0.25",
        "--eta_rescue",
        "0.25",
        "--eta_active",
        "0.25",
        "--gamma_boundary",
        "0.25",
        "--block_size",
        "64",
        "--token_unit_size",
        "8",
        "--prune_layers",
        "9,18,26",
        "--retain_tokens",
        "12288,6144,4096",
        "--temperature",
        "0",
        "--top_p",
        "1.0",
        "--max_model_len",
        "32768",
        "--selection_cost",
        "expanded_page_tokens",
        "--score_density_mode",
        "sqrt_len",
        "--calibrated_budget_config",
        str(calib),
    ]
    specs = [
        ("AB-SlimInfer", ["--block_strategy", "attention_boundary", "--rescue_policy", "none", "--budget_mode", "token_budget", "--target_token_budget_source", "fixed_K", "--target_token_budget", fixed_k, "--target_active_tokens", fixed_k]),
        ("AB-SlimInfer_larger_KplusM", ["--block_strategy", "attention_boundary", "--rescue_policy", "none", "--budget_mode", "token_budget", "--target_token_budget_source", "fixed_KplusM_drm", "--target_token_budget", fixed_km, "--target_active_tokens", fixed_km]),
        ("AB-DRM", ["--block_strategy", "attention_boundary", "--rescue_policy", "add", "--rescue_score", "drm", "--budget_mode", "token_budget", "--target_token_budget", fixed_k, "--active_budget_mode", "token_budget", "--target_active_tokens", fixed_k, "--rescue_budget_mode", "token_budget", "--target_final_tokens_source", "fixed_KplusM_drm", "--target_final_tokens", fixed_km]),
        ("AB-BoundaryDRM", ["--block_strategy", "attention_boundary", "--rescue_policy", "add", "--rescue_score", "boundary_drm", "--budget_mode", "token_budget", "--target_token_budget", fixed_k, "--active_budget_mode", "token_budget", "--target_active_tokens", fixed_k, "--rescue_budget_mode", "token_budget", "--target_final_tokens_source", "fixed_KplusM_drm", "--target_final_tokens", fixed_km]),
    ]
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {shlex.quote(str(REPO_ROOT))}",
        f"mkdir -p {shlex.quote(str(full_dir / 'audit'))} {shlex.quote(str(full_dir / 'block_layouts'))} {shlex.quote(str(full_dir / 'generations'))} {shlex.quote(str(full_dir / 'logits'))} {shlex.quote(str(full_dir / 'used_blocks'))}",
        "",
    ]
    for name, extra in specs:
        cmd = common + [
            "--output_dir", str(full_dir / name),
            "--isolated_run_id", name,
            "--rescue_audit_path", str(full_dir / "audit" / f"{name}.jsonl"),
            "--block_layout_audit_path", str(full_dir / "block_layouts" / f"{name}.jsonl"),
            "--debug_used_block_audit_path", str(full_dir / "used_blocks" / f"{name}.jsonl"),
            "--debug_generation_jsonl", str(full_dir / "generations" / f"{name}.jsonl"),
            "--debug_first_token_logits_path", str(full_dir / "logits" / f"{name}.jsonl"),
            "--save_layer_stats",
            "--save_block_scores", "aggregate_only",
            "--save_metrics_json",
            "--save_outputs",
        ] + extra
        lines.append(command_line(cmd))
    full_script = full_dir / "run_full_longbench.sh"
    write_text(full_script, "\n".join(lines) + "\n")
    make_executable(full_script)
    resume_full = full_dir / "resume_full_longbench.sh"
    write_text(resume_full, f"#!/usr/bin/env bash\nset -euo pipefail\nbash {shlex.quote(str(full_script))}\n")
    make_executable(resume_full)

    latency_dir = out_dir / "latency"
    latency_dir.mkdir(parents=True, exist_ok=True)
    latency_script = latency_dir / "run_latency_32k.sh"
    write_text(
        latency_script,
        "\n".join([
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            f"cd {shlex.quote(str(REPO_ROOT))}",
            "# Ready-to-run latency hook. Fill in the repo-specific latency entrypoint if it changes.",
            "# Methods to test: FullKV, SlimInfer original, SlimInfer larger K+M, fixed DRM-Add, AB-SlimInfer, AB-SlimInfer larger K+M, AB-DRM, AB-BoundaryDRM.",
            f"# Output CSV should be written to {latency_dir / 'latency_summary.csv'}",
        ]) + "\n",
    )
    make_executable(latency_script)
    write_csv(
        latency_dir / "latency_summary.csv",
        [
            {
                "method": method,
                "E2E_32k_s_mean": "",
                "E2E_32k_s_std": "",
                "TTFT_32k_s_mean": "",
                "TTFT_32k_s_std": "",
                "selected_tokens": "",
                "selected_pages": "",
                "selected_token_ratio": "",
                "selected_page_ratio": "",
                "page_expansion_overhead": "",
                "peak_gpu_memory": "",
                "runs": 0,
                "status": "not_run_calibration_unverified",
            }
            for method in ["FullKV", "SlimInfer original", "SlimInfer larger K+M", "fixed DRM-Add", "AB-SlimInfer", "AB-SlimInfer larger K+M", "AB-DRM", "AB-BoundaryDRM"]
        ],
        ["method", "E2E_32k_s_mean", "E2E_32k_s_std", "TTFT_32k_s_mean", "TTFT_32k_s_std", "selected_tokens", "selected_pages", "selected_token_ratio", "selected_page_ratio", "page_expansion_overhead", "peak_gpu_memory", "runs", "status"],
    )

    calib_dir = out_dir / "budget_calibration"
    calib_verify = calib_dir / "run_budget_calibration_verify.sh"
    calib_common = [
        "python",
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
        "--longbench_tasks", ",".join(DIAG_TASKS),
        "--max_samples_per_task", "64",
        "--generation_max_new_tokens", str(args.max_new_tokens),
        "--smoke_mode", "1",
        "--max_smoke_samples", "64",
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
        "--calibrated_budget_config", str(calib),
        "--save_layer_stats",
        "--save_block_scores", "aggregate_only",
        "--save_metrics_json",
        "--save_outputs",
    ]
    calib_specs = [
        ("fixed_sliminfer_matched_K", ["--block_strategy", "fixed", "--rescue_policy", "none", "--budget_mode", "token_budget", "--target_token_budget_source", "fixed_K", "--target_token_budget", fixed_k, "--target_active_tokens", fixed_k]),
        ("attention_boundary_sliminfer_matched_K", ["--block_strategy", "attention_boundary", "--rescue_policy", "none", "--budget_mode", "token_budget", "--target_token_budget_source", "fixed_K", "--target_token_budget", fixed_k, "--target_active_tokens", fixed_k]),
        ("fixed_drm_add_matched_KplusM", ["--block_strategy", "fixed", "--rescue_policy", "add", "--rescue_score", "drm", "--budget_mode", "token_budget", "--target_token_budget", fixed_k, "--active_budget_mode", "token_budget", "--target_active_tokens", fixed_k, "--rescue_budget_mode", "token_budget", "--target_final_tokens_source", "fixed_KplusM_drm", "--target_final_tokens", fixed_km]),
        ("attention_boundary_boundary_drm_matched_KplusM", ["--block_strategy", "attention_boundary", "--rescue_policy", "add", "--rescue_score", "boundary_drm", "--budget_mode", "token_budget", "--target_token_budget", fixed_k, "--active_budget_mode", "token_budget", "--target_active_tokens", fixed_k, "--rescue_budget_mode", "token_budget", "--target_final_tokens_source", "fixed_KplusM_drm", "--target_final_tokens", fixed_km]),
        ("fixed_sliminfer_matched_attention_tokens", ["--block_strategy", "fixed", "--rescue_policy", "none", "--budget_mode", "token_budget", "--target_token_budget_source", "attention_raw", "--target_token_budget", attention_raw, "--target_active_tokens", attention_raw]),
        ("attention_boundary_sliminfer_matched_attention_tokens", ["--block_strategy", "attention_boundary", "--rescue_policy", "none", "--budget_mode", "token_budget", "--target_token_budget_source", "attention_raw", "--target_token_budget", attention_raw, "--target_active_tokens", attention_raw]),
    ]
    verify_lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {shlex.quote(str(REPO_ROOT))}",
        f"mkdir -p {shlex.quote(str(calib_dir / 'verify_runs'))} {shlex.quote(str(calib_dir / 'verify_audit'))} {shlex.quote(str(calib_dir / 'verify_layouts'))} {shlex.quote(str(calib_dir / 'verify_generations'))}",
        "",
    ]
    for name, extra in calib_specs:
        cmd = calib_common + [
            "--output_dir", str(calib_dir / "verify_runs" / name),
            "--isolated_run_id", name,
            "--rescue_audit_path", str(calib_dir / "verify_audit" / f"{name}.jsonl"),
            "--block_layout_audit_path", str(calib_dir / "verify_layouts" / f"{name}.jsonl"),
            "--debug_generation_jsonl", str(calib_dir / "verify_generations" / f"{name}.jsonl"),
        ] + extra
        verify_lines.append(command_line(cmd))
    write_text(calib_verify, "\n".join(verify_lines) + "\n")
    make_executable(calib_verify)

    pipeline = out_dir / "run_full_pipeline.sh"
    write_text(
        pipeline,
        "\n".join([
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            f"cd {shlex.quote(str(REPO_ROOT))}",
            f"python {shlex.quote(str(REPO_ROOT / 'tools' / 'run_attention_boundary_full_pipeline.py'))} --output_dir {shlex.quote(str(out_dir))} --model_path {shlex.quote(args.model_path)} --longbench_dir {shlex.quote(args.longbench_dir)}",
        ]) + "\n",
    )
    make_executable(pipeline)
    resume = out_dir / "resume.sh"
    write_text(resume, f"#!/usr/bin/env bash\nset -euo pipefail\nbash {shlex.quote(str(pipeline))}\n")
    make_executable(resume)


def write_report(
    out_dir: Path,
    ctx: Dict[str, Any],
    calibration_config: Dict[str, Any],
    mismatch_rows: List[Dict[str, Any]],
    main_rows: List[Dict[str, Any]],
    category_rows: List[Dict[str, Any]],
    full_ran: bool,
    latency_ran: bool,
) -> None:
    budgets = ctx["budgets"]
    fb = ctx.get("fallback_rates", {})
    fields = ["Method", "Qasper", "MFQA", "HPQA", "2Wiki", "MuSiQue", "GovRep", "QMSum", "MNews", "VCSum", "TREC", "TQA", "SAMSum", "LSHT", "Count", "PassR", "LCC", "RepoB-p", "Avg", "SampleMode", "SampleCount", "Status"]
    cat_fields = ["Method", "SingleDocQA", "MultiDocQA", "Summarization", "FewShot", "Synthetic", "Code", "Avg"]
    report = [
        "# Attention-Boundary Full Pipeline Report",
        "",
        "## 1. Goal",
        "This run is the Attention-Boundary KV Pruning + Boundary-Confidence DRM pipeline.",
        "It first gates on budget calibration, then full LongBench, then latency. The calibration gate is not yet verified, so full LongBench was not started.",
        "",
        "## 2. Existing Full Main Table Background",
        "- FullKV Avg = 47.97",
        "- SlimInfer original Avg = 47.59",
        "- SlimInfer larger K+M Avg = 47.70",
        "- Random-Add Avg = 47.65",
        "- DeltaO-Add Avg = 47.67",
        "- fixed DRM-Add Avg = 47.80",
        "",
        "## 3. Method Definition",
        "- AB-SlimInfer: attention-boundary virtual blocks with token-budget SlimInfer selection.",
        "- AB-SlimInfer larger K+M: attention-boundary virtual blocks with matched K+M token budget and no rescue.",
        "- AB-DRM: attention-boundary active and token-budget ordinary DRM additive rescue.",
        "- AB-BoundaryDRM (ours): attention-boundary active and token-budget boundary-confidence weighted DRM additive rescue.",
        "",
        "## 4. Attention-Boundary Formula",
        "d_i = 1 - cosine(r_i, r_{i+1}), where r_i is the pooled page K/V representation.",
        "C_bd(B) = mean(boundary_strength_left, boundary_strength_right).",
        "",
        "## 5. Boundary-DRM Formula",
        "S = z(D_base) + alpha*z(G) - lambda*z(R) + gamma*z(C_bd).",
        "MMR: argmax[S - eta_rescue max cos(B,R) - eta_active max cos(B,A)].",
        "",
        "## 6. Budget Calibration",
        f"- T_fixed_K = {budgets.get('fixed_K', 0):.2f}",
        f"- T_fixed_KplusM_drm = {budgets.get('fixed_KplusM_drm', 0):.2f}",
        f"- T_attention_raw = {budgets.get('attention_raw', 0):.2f}",
        f"- Previous mean absolute token error = {calibration_config.get('previous_mean_abs_selected_token_error_pct', 0):.4f}",
        f"- Previous mismatch cases >3 percent = {len(mismatch_rows)}",
        "- Calibration status: proposed_unverified. Full run gate remains blocked.",
        "- High-budget control was not verified because the previous matched-attention runs were about 11.5 percent below target.",
        "",
        "## 7. Fallback Optimization",
        f"- semantic_v2 fallback from previous fair-hybrid run: {fb.get('semantic_v2', 0):.3f}. semantic_v3 is implemented but not verified in this run.",
        f"- query_dynamic_v2 fallback from previous fair-hybrid run: {fb.get('query_dynamic_v2', 0):.3f}. query_dynamic_v3 is implemented but not verified in this run.",
        "- Neither semantic_v3 nor query_dynamic_v3 enters the full table until fallback diagnostic is re-run and fallback_rate < 0.2.",
        "",
        "## 8. Main Table",
        markdown_table(main_rows, fields),
        "",
        "## 9. Category Average",
        markdown_table(category_rows, cat_fields),
        "",
        "## 10. Key Comparisons",
        "- AB-SlimInfer vs SlimInfer original: unavailable because full AB run was blocked by calibration.",
        "- AB-SlimInfer larger K+M vs SlimInfer larger K+M: unavailable.",
        "- AB-DRM vs fixed DRM-Add: unavailable.",
        "- AB-BoundaryDRM vs SlimInfer original: unavailable.",
        "- AB-BoundaryDRM vs SlimInfer larger K+M: unavailable.",
        "- AB-BoundaryDRM vs fixed DRM-Add: unavailable.",
        "- AB-BoundaryDRM vs AB-DRM: unavailable.",
        "- The 64-sample diagnostic remains encouraging: attention_boundary_boundary_drm_matched_KplusM beat fixed_drm_add_matched_KplusM by about +1.02 Avg, but full validation is pending.",
        "",
        "## 11. Fairness Notes",
        "- Token-budget uses expanded fixed pages because the real attention path consumes fixed KV pages.",
        "- Selected tokens, selected pages, and page expansion overhead must be reported before claiming selector superiority.",
        "- Methods with budget error >5 percent should not be used for core fairness conclusions.",
        "",
        "## 12. Failure And Limits",
        "- Full LongBench was not run because calibration has not been verified under the requested thresholds.",
        "- Latency was not run; a ready-to-run script was generated.",
        "- semantic_v3/query_dynamic_v3 were wired but not evaluated here.",
        "- High-budget control remains the main budget mismatch risk.",
        "",
        "## 13. Next Steps",
        "1. Run the six-method calibration verification with calibrated_budget_config.json.",
        "2. If mean token error is <3 percent and per-task error is <5 percent, run full LongBench for the AB methods.",
        "3. Then run the 32k latency script and case studies.",
        "",
        "## Artifacts",
        f"- budget_calibration_report: {out_dir / 'budget_calibration' / 'budget_calibration_report.md'}",
        f"- calibrated_budget_config: {out_dir / 'budget_calibration' / 'calibrated_budget_config.json'}",
        f"- full resume: {out_dir / 'full_longbench' / 'resume_full_longbench.sh'}",
        f"- latency script: {out_dir / 'latency' / 'run_latency_32k.sh'}",
        f"- full_ran: {full_ran}",
        f"- latency_ran: {latency_ran}",
    ]
    write_text(out_dir / "report.md", "\n".join(report) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate and prepare the attention-boundary full pipeline.")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--model_path", type=str, default="/diskE/models/Qwen2.5-7B-Instruct")
    parser.add_argument("--longbench_dir", type=str, default="/diskE/datasets/LongBench")
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--force_full", action="store_true", help="Write full commands but do not bypass this script's calibration warning.")
    parser.add_argument("--calibration_only", action="store_true")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) if args.output_dir else RESULTS_DIR / f"rescuekv_attention_boundary_full_pipeline_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    full_dir = RESULTS_DIR / "rescuekv_drm_add_full_longbench_20260523_233633"
    fair_dir = latest_dir("rescuekv_block_boundary_fair_hybrid_64_*")
    abcd_dir = latest_dir("rescuekv_block_boundary_abcd_64_*")
    ctx = extract_previous_context(fair_dir, abcd_dir, full_dir)
    calibration_config, calibration_rows, mismatch_rows, calibration_pass = derive_budget_calibration(ctx, out_dir)

    main_rows = add_placeholder_ab_rows(ctx.get("main_table", []))
    main_fields = list(main_rows[0].keys()) if main_rows else []
    write_csv(out_dir / "main_table.csv", main_rows, main_fields)
    category_rows = category_placeholders(ctx.get("category_table", []))
    write_csv(out_dir / "category_table.csv", category_rows, ["Method", "SingleDocQA", "MultiDocQA", "Summarization", "FewShot", "Synthetic", "Code", "Avg"])
    write_latex(out_dir / "main_table_latex.tex", main_rows)
    write_csv(out_dir / "summary.csv", main_rows, main_fields)
    write_csv(
        out_dir / "audit_summary.csv",
        [
            {
                "method": row["method"],
                "task": row["task"],
                "target_budget_source": row["target_budget_source"],
                "target_token_budget": row["target_token_budget"],
                "selected_token_count_mean": row["previous_observed_selected_tokens"],
                "selected_token_error_pct": row["previous_selected_token_error_pct"],
                "status": row["status"],
            }
            for row in calibration_rows
        ],
        ["method", "task", "target_budget_source", "target_token_budget", "selected_token_count_mean", "selected_token_error_pct", "status"],
    )
    write_scripts(out_dir, args, ctx)

    config = {
        "created_at": timestamp,
        "repo": str(REPO_ROOT),
        "output_dir": str(out_dir),
        "full_main_table_dir": str(full_dir),
        "fair_hybrid_dir": str(fair_dir) if fair_dir else "",
        "abcd_dir": str(abcd_dir) if abcd_dir else "",
        "budgets": ctx["budgets"],
        "calibration_pass": calibration_pass,
        "full_longbench_run": False,
        "latency_run": False,
        "reason_full_not_run": "calibration_unverified_or_failed",
    }
    write_json(out_dir / "config.json", config)

    write_report(
        out_dir=out_dir,
        ctx=ctx,
        calibration_config=calibration_config,
        mismatch_rows=mismatch_rows,
        main_rows=main_rows,
        category_rows=category_rows,
        full_ran=False,
        latency_ran=False,
    )
    print(out_dir)


if __name__ == "__main__":
    main()
