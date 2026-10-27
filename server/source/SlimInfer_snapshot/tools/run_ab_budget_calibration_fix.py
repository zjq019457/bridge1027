#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
import stat
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results"
SOURCE_DIR = RESULTS / "rescuekv_attention_boundary_full_pipeline_20260526_072144"
PREV_DIAG_DIR = RESULTS / "rescuekv_block_boundary_fair_hybrid_64_20260525_175447"
TASKS = ["hotpotqa", "2wikimqa", "musique", "qasper", "multifieldqa_en"]
CALIBRATION_METHODS = [
    ("fixed_sliminfer_matched_K", "fixed", "none", "delta_o", "fixed_K", "active"),
    ("attention_boundary_sliminfer_matched_K", "attention_boundary", "none", "delta_o", "fixed_K", "active"),
    ("fixed_drm_add_matched_KplusM", "fixed", "add", "drm", "fixed_KplusM_drm", "final"),
    ("attention_boundary_boundary_drm_matched_KplusM", "attention_boundary", "add", "boundary_drm", "fixed_KplusM_drm", "final"),
    ("fixed_sliminfer_matched_attention_tokens", "fixed", "none", "delta_o", "attention_raw", "active"),
    ("attention_boundary_sliminfer_matched_attention_tokens", "attention_boundary", "none", "delta_o", "attention_raw", "active"),
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


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def make_executable(path: Path) -> None:
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


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


def command(parts: List[Any]) -> str:
    return " ".join(shlex.quote(str(x)) for x in parts)


def torch_available() -> Tuple[bool, str]:
    proc = subprocess.run(
        [sys.executable, "-c", "import torch; print(torch.__version__); print(torch.cuda.is_available())"],
        cwd=str(REPO_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        return False, (proc.stderr or proc.stdout).strip()
    return True, proc.stdout.strip()


def load_source_artifacts() -> Dict[str, Any]:
    missing = []
    required = [
        SOURCE_DIR / "report.md",
        SOURCE_DIR / "budget_calibration" / "budget_calibration_report.md",
        SOURCE_DIR / "budget_calibration" / "calibrated_budget_config.json",
        SOURCE_DIR / "budget_calibration" / "budget_calibration.csv",
        SOURCE_DIR / "budget_calibration" / "budget_mismatch_cases.csv",
        SOURCE_DIR / "config.json",
    ]
    for path in required:
        if not path.exists():
            missing.append(str(path))
    source_config = read_json(SOURCE_DIR / "config.json")
    prev_calib = read_csv(SOURCE_DIR / "budget_calibration" / "budget_calibration.csv")
    prev_mismatch = read_csv(SOURCE_DIR / "budget_calibration" / "budget_mismatch_cases.csv")
    prev_summary = read_csv(PREV_DIAG_DIR / "summary.csv")
    prev_fallback = read_csv(PREV_DIAG_DIR / "fallback_reason_summary.csv")
    return {
        "missing": missing,
        "source_config": source_config,
        "prev_calib": prev_calib,
        "prev_mismatch": prev_mismatch,
        "prev_summary": prev_summary,
        "prev_fallback": prev_fallback,
    }


def infer_budgets(source_config: Dict[str, Any], prev_summary: List[Dict[str, str]]) -> Dict[str, float]:
    budgets = dict(source_config.get("budgets") or {})
    if "fixed_K" not in budgets:
        vals = [fnum(r.get("selected_token_count_mean")) for r in prev_summary if r.get("method") == "fixed_sliminfer_original"]
        budgets["fixed_K"] = mean(vals)
    if "fixed_KplusM_drm" not in budgets:
        vals = [fnum(r.get("selected_token_count_mean")) for r in prev_summary if r.get("method") == "fixed_drm_add"]
        budgets["fixed_KplusM_drm"] = mean(vals)
    if "attention_raw" not in budgets:
        vals = [fnum(r.get("selected_token_count_mean")) for r in prev_summary if r.get("method") == "attention_boundary_sliminfer_raw"]
        budgets["attention_raw"] = mean(vals)
    return {k: float(v) for k, v in budgets.items()}


def previous_observed(prev_calib: List[Dict[str, str]], method: str, task: str) -> float:
    for row in prev_calib:
        if row.get("method") == method and row.get("task") == task:
            return fnum(row.get("previous_observed_selected_tokens"))
    return 0.0


def raw_task_reference(prev_summary: List[Dict[str, str]], method: str, task: str) -> float:
    for row in prev_summary:
        if row.get("method") == method and row.get("task") == task:
            return fnum(row.get("selected_token_count_mean"))
    return 0.0


def build_calibration_rows(ctx: Dict[str, Any], budgets: Dict[str, float]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    mismatch: List[Dict[str, Any]] = []
    config: Dict[str, Any] = {
        "version": "ab_budget_calibration_fix_v2",
        "controller": {
            "incremental_expanded_page_cost": True,
            "nearest_to_target": True,
            "allow_budget_overshoot": True,
            "soft_budget_tolerance": 0.03,
            "hard_budget_tolerance": 0.05,
            "candidate_expansion": True,
            "local_budget_refinement": True,
            "cap_to_reachable_tokens": True,
            "selection_cost": "expanded_page_tokens",
            "score_density_mode": "sqrt_len",
        },
        "targets": {k: int(round(v)) for k, v in budgets.items()},
        "methods": {},
    }
    flat_methods: Dict[str, Any] = {}
    prev_summary = ctx["prev_summary"]
    for method, strategy, rescue_policy, rescue_score, target_source, kind in CALIBRATION_METHODS:
        global_target = float(budgets.get(target_source, 0.0) or 0.0)
        method_entry = {
            "block_strategy": strategy,
            "rescue_policy": rescue_policy,
            "rescue_score": rescue_score,
            "target_budget_source": target_source,
            "target_tokens": int(round(global_target)),
            "budget_kind": kind,
            "tasks": {},
        }
        flat_methods[method] = {"tasks": method_entry["tasks"]}
        for task in TASKS:
            raw_ref = raw_task_reference(prev_summary, "attention_boundary_sliminfer_raw", task) if target_source == "attention_raw" else 0.0
            target = min(global_target, raw_ref) if raw_ref > 0 else global_target
            target = float(int(round(target)))
            observed = previous_observed(ctx["prev_calib"], method, task)
            error = (observed - target) / target if target > 0 else 0.0
            abs_error = abs(error)
            scale = target / observed if observed > 0 and target_source != "attention_raw" else 1.0
            mismatch_reason = ""
            target_cap_reason = ""
            if target_source == "attention_raw" and raw_ref and target < global_target:
                target_cap_reason = "per_task_attention_raw_reference"
            elif abs_error > 0.03:
                mismatch_reason = "previous_controller_under_target"
            budget_status = "needs_verification"
            row = {
                "method": method,
                "task": task,
                "target_tokens": target,
                "uncapped_target_tokens": global_target,
                "observed_tokens": observed,
                "error_pct": error,
                "abs_error_pct": abs_error,
                "active_budget_scale": scale if kind == "active" else 1.0,
                "final_budget_scale": scale if kind == "final" else 1.0,
                "min_reachable_tokens": "",
                "max_reachable_tokens": raw_ref if raw_ref else "",
                "protected_tokens": "",
                "total_candidate_tokens": "",
                "selected_pages": int(round(observed / 64.0)) if observed else "",
                "page_size": 64,
                "overshoot_used": "",
                "budget_status": budget_status,
                "mismatch_reason": mismatch_reason,
                "target_cap_reason": target_cap_reason,
            }
            rows.append(row)
            if abs_error > 0.03 or budget_status != "ok":
                mismatch.append(row)
            task_entry = {
                "target_tokens": int(round(target)),
                "active_budget_scale": row["active_budget_scale"],
                "final_budget_scale": row["final_budget_scale"],
                "active_threshold": "",
                "rescue_threshold": "",
                "observed_tokens": observed,
                "error_pct": error,
                "uncapped_target_tokens": int(round(global_target)),
                "task_reference_tokens": int(round(raw_ref)) if raw_ref else "",
                "target_cap_reason": target_cap_reason,
                "cap_to_reachable_tokens": True,
                "candidate_expansion": True,
                "local_budget_refinement": True,
                "controller_target_active_tokens": int(round(target if target_source == "attention_raw" else target * scale)) if kind == "active" else "",
                "controller_target_final_tokens": int(round(target if target_source == "attention_raw" else target * scale)) if kind == "final" else "",
                "discrete_limit": False,
                "best_possible_error_pct": "",
                "mismatch_reason": mismatch_reason,
            }
            method_entry["tasks"][task] = task_entry
        config["methods"][method] = method_entry
        config[method] = {"tasks": method_entry["tasks"]}
    config["flat_methods"] = flat_methods
    return rows, mismatch, config


def fallback_rows(ctx: Dict[str, Any], env_ready: bool) -> List[Dict[str, Any]]:
    previous = ctx["prev_fallback"]
    rows: List[Dict[str, Any]] = []
    for strategy in ["semantic_v3", "query_dynamic_v3"]:
        for task in TASKS:
            row = {"block_strategy": strategy, "task": task, "fallback_rate": "", "status": "not_run_env_missing_torch" if not env_ready else "ready_to_run"}
            for reason in FALLBACK_REASONS:
                row[reason] = ""
            rows.append(row)
    for row in previous:
        if row.get("block_strategy") in {"semantic_v2", "query_dynamic_v2"}:
            rows.append(row)
    return rows


def write_scripts(out_dir: Path, budgets: Dict[str, float], args: argparse.Namespace, full_ready: bool) -> None:
    calib_config = out_dir / "calibrated_budget_config.json"
    fixed_k = str(int(round(budgets.get("fixed_K", 6040.0))))
    fixed_km = str(int(round(budgets.get("fixed_KplusM_drm", 6342.0))))
    attention_raw = str(int(round(budgets.get("attention_raw", 9222.0))))
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
        "--allow_budget_overshoot", "1",
        "--soft_budget_tolerance", "0.03",
        "--hard_budget_tolerance", "0.05",
        "--calibrated_budget_config", str(calib_config),
        "--save_layer_stats",
        "--save_block_scores", "aggregate_only",
        "--save_metrics_json",
        "--save_outputs",
    ]
    specs = [
        ("fixed_sliminfer_matched_K", ["--block_strategy", "fixed", "--rescue_policy", "none", "--budget_mode", "token_budget", "--target_token_budget_source", "fixed_K", "--target_token_budget", fixed_k, "--target_active_tokens", fixed_k]),
        ("attention_boundary_sliminfer_matched_K", ["--block_strategy", "attention_boundary", "--rescue_policy", "none", "--budget_mode", "token_budget", "--target_token_budget_source", "fixed_K", "--target_token_budget", fixed_k, "--target_active_tokens", fixed_k]),
        ("fixed_drm_add_matched_KplusM", ["--block_strategy", "fixed", "--rescue_policy", "add", "--rescue_score", "drm", "--budget_mode", "token_budget", "--target_token_budget", fixed_k, "--active_budget_mode", "token_budget", "--target_active_tokens", fixed_k, "--rescue_budget_mode", "token_budget", "--target_final_tokens_source", "fixed_KplusM_drm", "--target_final_tokens", fixed_km]),
        ("attention_boundary_boundary_drm_matched_KplusM", ["--block_strategy", "attention_boundary", "--rescue_policy", "add", "--rescue_score", "boundary_drm", "--budget_mode", "token_budget", "--target_token_budget", fixed_k, "--active_budget_mode", "token_budget", "--target_active_tokens", fixed_k, "--rescue_budget_mode", "token_budget", "--target_final_tokens_source", "fixed_KplusM_drm", "--target_final_tokens", fixed_km]),
        ("fixed_sliminfer_matched_attention_tokens", ["--block_strategy", "fixed", "--rescue_policy", "none", "--budget_mode", "token_budget", "--target_token_budget_source", "attention_raw", "--target_token_budget", attention_raw, "--target_active_tokens", attention_raw]),
        ("attention_boundary_sliminfer_matched_attention_tokens", ["--block_strategy", "attention_boundary", "--rescue_policy", "none", "--budget_mode", "token_budget", "--target_token_budget_source", "attention_raw", "--target_token_budget", attention_raw, "--target_active_tokens", attention_raw]),
        ("semantic_v3_sliminfer_diagnostic", ["--block_strategy", "semantic_v3", "--rescue_policy", "none", "--budget_mode", "token_budget", "--target_token_budget_source", "fixed_K", "--target_token_budget", fixed_k, "--target_active_tokens", fixed_k]),
        ("query_dynamic_v3_sliminfer_diagnostic", ["--block_strategy", "query_dynamic_v3", "--rescue_policy", "none", "--budget_mode", "token_budget", "--target_token_budget_source", "fixed_K", "--target_token_budget", fixed_k, "--target_active_tokens", fixed_k]),
    ]
    def script_lines(selected_specs: List[Tuple[str, List[str]]]) -> List[str]:
        lines = ["#!/usr/bin/env bash", "set -euo pipefail", f"cd {shlex.quote(str(REPO_ROOT))}", f"mkdir -p {shlex.quote(str(out_dir / 'runs'))} {shlex.quote(str(out_dir / 'audit'))} {shlex.quote(str(out_dir / 'block_layouts'))} {shlex.quote(str(out_dir / 'generations'))}", ""]
        for name, extra in selected_specs:
            cmd = common + [
                "--output_dir", str(out_dir / "runs" / name),
                "--isolated_run_id", name,
                "--rescue_audit_path", str(out_dir / "audit" / f"{name}.jsonl"),
                "--block_layout_audit_path", str(out_dir / "block_layouts" / f"{name}.jsonl"),
                "--debug_generation_jsonl", str(out_dir / "generations" / f"{name}.jsonl"),
            ] + extra
            lines.append(command(cmd))
        lines.append(command([sys.executable, str(REPO_ROOT / "tools" / "finalize_ab_budget_calibration_fix.py"), str(out_dir)]))
        return lines

    run_script = out_dir / "run_calibration_fix.sh"
    write_text(run_script, "\n".join(script_lines(specs)) + "\n")
    make_executable(run_script)
    calibration_only = out_dir / "run_budget_calibration_only.sh"
    write_text(calibration_only, "\n".join(script_lines(specs[:6])) + "\n")
    make_executable(calibration_only)
    matched_attention = out_dir / "run_matched_attention_calibration.sh"
    write_text(matched_attention, "\n".join(script_lines(specs[4:6])) + "\n")
    make_executable(matched_attention)
    resume = out_dir / "resume.sh"
    write_text(resume, f"#!/usr/bin/env bash\nset -euo pipefail\nbash {shlex.quote(str(run_script))}\n")
    make_executable(resume)
    ready = out_dir / "ready_to_run_full_if_passed.sh"
    if full_ready:
        write_text(ready, f"#!/usr/bin/env bash\nset -euo pipefail\nbash {shlex.quote(str(SOURCE_DIR / 'full_longbench' / 'run_full_longbench.sh'))}\n")
    else:
        write_text(ready, "#!/usr/bin/env bash\nset -euo pipefail\necho 'FULL_READY=false: run calibration verification and inspect report.md first.'\nexit 1\n")
    make_executable(ready)


def write_report(out_dir: Path, ctx: Dict[str, Any], rows: List[Dict[str, Any]], mismatch: List[Dict[str, Any]], env_ready: bool, env_note: str, full_ready: bool) -> None:
    abs_errors = [fnum(r["abs_error_pct"]) for r in rows]
    method_means = {}
    task_means = {}
    for method, *_ in CALIBRATION_METHODS:
        method_means[method] = mean(fnum(r["abs_error_pct"]) for r in rows if r["method"] == method)
    for task in TASKS:
        task_means[task] = mean(fnum(r["abs_error_pct"]) for r in rows if r["task"] == task)
    high_budget_rows = [r for r in rows if "attention_tokens" in r["method"]]
    high_budget_ok = all(fnum(r["abs_error_pct"]) <= 0.05 and not str(r.get("mismatch_reason", "")).startswith("global_attention") for r in high_budget_rows)
    lines = [
        "# AB Budget Calibration Fix Report",
        "",
        "## 1. Goal",
        "This run only repairs and prepares token-budget calibration verification plus semantic/query fallback diagnostic. It does not run full LongBench.",
        "",
        "## 2. Previous Problem",
        "- Previous mean absolute token error: 5.83%.",
        "- Previous mismatch cases over 3 percent: 11.",
        "- High-budget control was systematically below the global attention_raw target on short/medium tasks.",
        f"- Missing source artifacts: {ctx['missing'] if ctx['missing'] else 'none'}",
        "",
        "## 3. Fixes Implemented",
        "- Incremental expanded-page cost: delta_cost(B|S) counts only newly introduced fixed pages.",
        "- Nearest-to-target selection: the controller keeps the selected set with the smallest absolute token error.",
        "- Overshoot tolerance: CLI supports allow_budget_overshoot, soft_budget_tolerance=0.03, hard_budget_tolerance=0.05.",
        "- Per-method/per-task calibrated budget config is emitted and supported by the cache path.",
        "- matched_attention_tokens uses min(global attention_raw target, per-task attention_raw reference), with qasper, 2wikimqa, and multifieldqa_en capped before selection.",
        "- Per-sample reachability caps and local candidate refinement keep the controller target inside the selectable page range.",
        "- Cache audit now records calibrated_budget_config_loaded, calibration_entry_found, target_tokens_used_by_cache, and observed_final_selected_tokens_used_by_attention.",
        "- Protected/recent/sink pages remain counted in final selected pages because selected_token_count is derived from final attention pages.",
        "",
        "## 4. Verification Status",
        f"- Torch environment available: {env_ready}",
        f"- Environment note: {env_note or 'ok'}",
        "- Model-level verification was not run in this shell because torch is unavailable." if not env_ready else "- Model-level verification script is ready to run.",
        "",
        "## 5. Calibration Snapshot From Previous Run",
        f"- Previous overall mean abs error: {mean(abs_errors):.4f}",
        f"- All method-task below 5 percent: {all(e < 0.05 for e in abs_errors)}",
        f"- Mean below 3 percent: {mean(abs_errors) < 0.03}",
        f"- Gate passed: {full_ready}",
        "",
        "### Per-method mean abs error",
    ]
    lines.extend(f"- {k}: {v:.4f}" for k, v in method_means.items())
    lines.append("")
    lines.append("### Per-task mean abs error")
    lines.extend(f"- {k}: {v:.4f}" for k, v in task_means.items())
    lines.extend([
        "",
        "## 6. High-budget Control",
        f"- fixed_sliminfer_matched_attention_tokens / attention_boundary_sliminfer_matched_attention_tokens aligned: {high_budget_ok}",
        "- High-budget controls use per-task attention_raw caps when the global target is above the task reference.",
        "",
        "## 7. Fallback Diagnostic",
        "- semantic_v3 fallback_rate: not verified in model path.",
        "- query_dynamic_v3 fallback_rate: not verified in model path.",
        "- Current shell lacks torch, so fallback diagnostic commands were generated but not executed.",
        "",
        "## 8. FULL_READY",
        f"FULL_READY = {str(full_ready).lower()}",
        "",
        "## 9. Next Step",
        "Run run_budget_calibration_only.sh in the proper torch/GPU environment. If the resulting audit summary shows mean abs error <3 percent and every method-task <5 percent, then enable the full LongBench script.",
    ])
    write_text(out_dir / "report.md", "\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair AB budget calibration artifacts and scripts.")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--model_path", type=str, default="/diskE/models/Qwen2.5-7B-Instruct")
    parser.add_argument("--longbench_dir", type=str, default="/diskE/datasets/LongBench")
    parser.add_argument("--max_new_tokens", type=int, default=16)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) if args.output_dir else RESULTS / f"rescuekv_ab_budget_calibration_fix_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ctx = load_source_artifacts()
    budgets = infer_budgets(ctx["source_config"], ctx["prev_summary"])
    rows, mismatch, calib_config = build_calibration_rows(ctx, budgets)
    env_ready, env_note = torch_available()
    abs_errors = [fnum(r["abs_error_pct"]) for r in rows]
    full_ready = env_ready and bool(abs_errors) and mean(abs_errors) < 0.03 and all(e < 0.05 for e in abs_errors)
    calib_config["generated_at"] = timestamp
    calib_config["full_ready"] = full_ready
    calib_config["verification_status"] = "not_run_env_missing_torch" if not env_ready else "ready_to_verify"
    write_json(out_dir / "calibrated_budget_config.json", calib_config)
    calibration_fields = ["method", "task", "target_tokens", "uncapped_target_tokens", "observed_tokens", "error_pct", "abs_error_pct", "active_budget_scale", "final_budget_scale", "min_reachable_tokens", "max_reachable_tokens", "protected_tokens", "total_candidate_tokens", "selected_pages", "page_size", "overshoot_used", "budget_status", "mismatch_reason", "target_cap_reason"]
    write_csv(out_dir / "budget_calibration.csv", rows, calibration_fields)
    write_csv(out_dir / "budget_mismatch_cases.csv", mismatch, calibration_fields)
    fb_rows = fallback_rows(ctx, env_ready)
    write_csv(out_dir / "fallback_reason_summary.csv", fb_rows, ["block_strategy", "task", "fallback_rate", *FALLBACK_REASONS, "status"])
    write_csv(out_dir / "audit_summary.csv", [
        {
            "method": r["method"],
            "task": r["task"],
            "calibrated_config_loaded": "",
            "calibration_entry_found": "",
            "target_tokens_used_by_cache": "",
            "observed_final_selected_tokens_used_by_attention": r["observed_tokens"],
            "selected_pages": r["selected_pages"],
            "page_expansion_overhead_mean": "",
            "duplicate_pages": "",
            "protected_violation": "",
            "status": "not_run_env_missing_torch" if not env_ready else "ready_to_verify",
        }
        for r in rows
    ], ["method", "task", "calibrated_config_loaded", "calibration_entry_found", "target_tokens_used_by_cache", "observed_final_selected_tokens_used_by_attention", "selected_pages", "page_expansion_overhead_mean", "duplicate_pages", "protected_violation", "status"])
    write_json(out_dir / "config.json", {
        "source_dir": str(SOURCE_DIR),
        "previous_diagnostic_dir": str(PREV_DIAG_DIR),
        "output_dir": str(out_dir),
        "budgets": budgets,
        "torch_available": env_ready,
        "torch_note": env_note,
        "full_ready": full_ready,
        "full_longbench_run": False,
    })
    write_scripts(out_dir, budgets, args, full_ready)
    write_report(out_dir, ctx, rows, mismatch, env_ready, env_note, full_ready)
    print(out_dir)


if __name__ == "__main__":
    main()
