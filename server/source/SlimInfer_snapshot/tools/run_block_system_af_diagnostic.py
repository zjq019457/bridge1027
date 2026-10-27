#!/usr/bin/env python3
"""Staged A-F block-system diagnostic.

Round contract (see report.md):
- The fixed SlimInfer active set is NEVER replaced. Only the rescue candidate
  pool varies across schemes A-F.
- Every rescue scheme is compared against fixed DRM-Add on the SAME matched
  token budget (K+M), on the SAME sample subset.
- This tool does NOT run full LongBench and does NOT run latency.

It runs in two stages:
  Stage 1: 5 tasks x 16 samples sanity, broad method set.
  Stage 2: 7 tasks x 64 samples, only promoted top methods (>= criteria).
Promotion / success criteria follow the task spec.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
RUNNER = REPO_ROOT / "experiments" / "run_qwen25_sliminfer_variants.py"

DEFAULT_MODEL = "/home/yu-bai/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
DEFAULT_LONGBENCH = "/home/yu-bai/Desktop/jiaqizhang/HeadKV/data/LongBench"

STAGE1_TASKS = ["musique", "passage_count", "passage_retrieval_en", "hotpotqa", "qasper"]
STAGE2_TASKS = ["musique", "passage_count", "passage_retrieval_en", "hotpotqa", "qasper", "2wikimqa", "multifieldqa_en"]

CATEGORY_MAP = {
    "qasper": "SingleDocQA", "multifieldqa_en": "SingleDocQA",
    "hotpotqa": "MultiDocQA", "2wikimqa": "MultiDocQA", "musique": "MultiDocQA",
    "passage_count": "Synthetic", "passage_retrieval_en": "Synthetic",
}

BASELINE_METHOD = "fixed_drm_add_subset"

# Each spec drives experiments/run_qwen25_sliminfer_variants.py via subprocess.
METHOD_SPECS: Dict[str, Dict[str, Any]] = {
    # --- baselines (fixed active set; existing validated paths) ---
    "fixed_sliminfer_original_subset": {
        "runner_method": "sliminfer_original", "scoring": "token_unit_qk", "decode_policy": "original",
        "block_strategy": "fixed", "rescue_policy": "none", "rescue_score": "delta_o",
        "rescue_candidate_strategy": "", "scheme": "baseline"},
    "fixed_sliminfer_larger_subset": {
        "runner_method": "sliminfer_original", "scoring": "token_unit_qk", "decode_policy": "original",
        "block_strategy": "fixed", "rescue_policy": "sliminfer_larger", "rescue_score": "delta_o",
        "rescue_candidate_strategy": "", "scheme": "baseline"},
    "fixed_drm_add_subset": {
        "runner_method": "sliminfer_variant", "scoring": "delta_o", "decode_policy": "reuse_prefill",
        "block_strategy": "fixed", "rescue_policy": "add", "rescue_score": "drm",
        "rescue_candidate_strategy": "", "scheme": "baseline"},
    "random_add_subset": {
        "runner_method": "sliminfer_variant", "scoring": "delta_o", "decode_policy": "reuse_prefill",
        "block_strategy": "fixed", "rescue_policy": "add", "rescue_score": "random",
        "rescue_candidate_strategy": "", "scheme": "baseline"},
    # --- A: overlap rescue ---
    "overlap_drm_rescue": {
        "runner_method": "sliminfer_variant", "scoring": "delta_o", "decode_policy": "reuse_prefill",
        "block_strategy": "fixed_active_rescue", "rescue_policy": "add", "rescue_score": "drm",
        "rescue_candidate_strategy": "overlap", "scheme": "A"},
    "overlap_deltao_rescue": {
        "runner_method": "sliminfer_variant", "scoring": "delta_o", "decode_policy": "reuse_prefill",
        "block_strategy": "fixed_active_rescue", "rescue_policy": "add", "rescue_score": "delta_o",
        "rescue_candidate_strategy": "overlap", "scheme": "A"},
    # --- B: neighbor expansion ---
    "neighbor_expansion_score": {
        "runner_method": "sliminfer_variant", "scoring": "delta_o", "decode_policy": "reuse_prefill",
        "block_strategy": "fixed_active_rescue", "rescue_policy": "add", "rescue_score": "delta_o",
        "rescue_candidate_strategy": "neighbor", "scheme": "B"},
    "neighbor_expansion_drm": {
        "runner_method": "sliminfer_variant", "scoring": "delta_o", "decode_policy": "reuse_prefill",
        "block_strategy": "fixed_active_rescue", "rescue_policy": "add", "rescue_score": "drm",
        "rescue_candidate_strategy": "neighbor", "scheme": "B"},
    # --- C: two-level hierarchy ---
    "hierarchy_rescue": {
        "runner_method": "sliminfer_variant", "scoring": "delta_o", "decode_policy": "reuse_prefill",
        "block_strategy": "fixed_active_rescue", "rescue_policy": "add", "rescue_score": "drm",
        "rescue_candidate_strategy": "hierarchy", "scheme": "C"},
    # --- D: page-aligned semantic ---
    "semantic_page_aligned_rescue": {
        "runner_method": "sliminfer_variant", "scoring": "delta_o", "decode_policy": "reuse_prefill",
        "block_strategy": "fixed_active_rescue", "rescue_policy": "add", "rescue_score": "drm",
        "rescue_candidate_strategy": "page_aligned_semantic", "scheme": "D"},
    # --- E: evidence-centered windows ---
    "evidence_centered_rescue": {
        "runner_method": "sliminfer_variant", "scoring": "delta_o", "decode_policy": "reuse_prefill",
        "block_strategy": "fixed_active_rescue", "rescue_policy": "add", "rescue_score": "drm",
        "rescue_candidate_strategy": "evidence_centered", "scheme": "E"},
    # --- F: fixed active + fine rescue fusion ---
    "fusion_fine_rescue": {
        "runner_method": "sliminfer_variant", "scoring": "delta_o", "decode_policy": "reuse_prefill",
        "block_strategy": "fixed_active_rescue", "rescue_policy": "add", "rescue_score": "drm",
        "rescue_candidate_strategy": "fusion_fine", "scheme": "F", "beta_query": 0.2},
    "fusion_fine_no_query": {
        "runner_method": "sliminfer_variant", "scoring": "delta_o", "decode_policy": "reuse_prefill",
        "block_strategy": "fixed_active_rescue", "rescue_policy": "add", "rescue_score": "drm",
        "rescue_candidate_strategy": "fusion_fine_no_query", "scheme": "F", "beta_query": 0.0},
}

STAGE1_METHODS = [
    "fixed_sliminfer_original_subset", "fixed_drm_add_subset",
    "overlap_drm_rescue", "neighbor_expansion_drm", "hierarchy_rescue",
    "semantic_page_aligned_rescue", "evidence_centered_rescue", "fusion_fine_rescue",
    # optional but cheap, run by default:
    "overlap_deltao_rescue", "fusion_fine_no_query", "random_add_subset",
]
# A-F candidate schemes eligible for promotion (exclude pure baselines)
AF_METHODS = [
    "overlap_drm_rescue", "overlap_deltao_rescue", "neighbor_expansion_score",
    "neighbor_expansion_drm", "hierarchy_rescue", "semantic_page_aligned_rescue",
    "evidence_centered_rescue", "fusion_fine_rescue", "fusion_fine_no_query",
]


# --------------------------------------------------------------------------
# io helpers
# --------------------------------------------------------------------------
def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            pass
    return out


def _mean(values: Iterable[Any]) -> Optional[float]:
    vals = []
    for v in values:
        try:
            if v not in ("", None):
                vals.append(float(v))
        except Exception:
            pass
    return statistics.mean(vals) if vals else None


def _score_rows(method_dir: Path) -> List[Dict[str, Any]]:
    path = method_dir / "longbench_scores.csv"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# --------------------------------------------------------------------------
# runner command
# --------------------------------------------------------------------------
def _runner_cmd(out_root: Path, method: str, spec: Dict[str, Any], config: Dict[str, Any], tasks: List[str], max_samples: int) -> List[str]:
    cmd = [
        str(PYTHON), str(RUNNER),
        "--model_path", config["model_path"],
        "--tokenizer_path", config["model_path"],
        "--model_family", "qwen2", "--dtype", "fp16", "--device", "cuda",
        "--backend", "flash_attention_2",
        "--method", spec["runner_method"],
        "--scoring", spec["scoring"],
        "--decode_policy", spec["decode_policy"],
        "--eval", "longbench",
        "--longbench_dir", config["longbench_dir"],
        "--longbench_tasks", ",".join(tasks),
        "--max_samples_per_task", str(max_samples),
        "--generation_max_new_tokens", str(config["max_new_tokens"]),
        "--smoke_mode", "1", "--max_smoke_samples", str(max_samples),
        "--output_dir", str(out_root / "runs" / method),
        "--isolated_run_id", method,
        "--block_strategy", spec["block_strategy"],
        "--target_block_tokens", "64", "--min_block_tokens", "32", "--max_block_tokens", "128",
        "--block_overlap_tokens", "0", "--block_size", "64",
        "--coarse_block_tokens", "256", "--fine_block_tokens", "64", "--medium_block_tokens", "128",
        "--prune_layers", "9,18,26", "--retain_tokens", "12288,6144,4096", "--preserve_first_block",
        "--temperature", "0", "--top_p", "1.0",
        "--enable_cpu_offload", "--enable_async_kv", "--max_model_len", "32768",
        "--rescue_policy", spec["rescue_policy"],
        "--rescue_score", spec["rescue_score"],
        "--rescue_ratio", str(config["rescue_ratio"]),
        "--seed", "0",
        "--alpha_diff", "0.5", "--lambda_red", "0.25", "--eta_rescue", "0.25", "--eta_active", "0.25",
        "--lambda_len", str(config["lambda_len"]),
        "--beta_query", str(spec.get("beta_query", config["beta_query"])),
        "--beta_semantic", str(config["beta_semantic"]),
        "--rescue_overlap_window_tokens", "64", "--rescue_overlap_stride_tokens", "32",
        "--rescue_evidence_half_window_tokens", "64", "--rescue_evidence_top_ratio", "0.2",
        "--rescue_semantic_page_tolerance", "16",
        "--rescue_audit_path", str(out_root / "audit" / f"{method}.jsonl"),
        "--block_layout_audit_path", str(out_root / "block_layouts" / f"{method}.jsonl"),
        "--debug_generation_jsonl", str(out_root / "generations" / f"{method}.jsonl"),
        "--save_outputs",
    ]
    if spec.get("rescue_candidate_strategy"):
        cmd.extend(["--rescue_candidate_strategy", spec["rescue_candidate_strategy"]])
    if spec["runner_method"] == "sliminfer_variant":
        cmd.extend(["--fd_alpha_o", "1.0", "--fd_beta_h", "0.0", "--fd_gamma_s", "0.0"])
    return cmd


def _method_complete(out_root: Path, method: str, tasks: List[str], max_samples: int) -> bool:
    rows = _score_rows(out_root / "runs" / method)
    by_task = {r.get("dataset"): r for r in rows}
    for task in tasks:
        row = by_task.get(task)
        if not row or row.get("status") != "ok":
            return False
        try:
            if int(float(row.get("num_samples", 0))) < min(max_samples, 1):
                return False
        except Exception:
            return False
    return True


def run_method(out_root: Path, method: str, config: Dict[str, Any], tasks: List[str], max_samples: int,
               *, dry_run: bool, skip_existing: bool) -> Dict[str, Any]:
    spec = METHOD_SPECS[method]
    if skip_existing and _method_complete(out_root, method, tasks, max_samples):
        return {"method": method, "status": "skipped_existing", "returncode": 0}
    for sub in ["audit", "block_layouts", "generations", "logs", "runs"]:
        (out_root / sub).mkdir(parents=True, exist_ok=True)
    cmd = _runner_cmd(out_root, method, spec, config, tasks, max_samples)
    if dry_run:
        return {"method": method, "status": "dry_run", "returncode": 0, "cmd": cmd}
    log_path = out_root / "logs" / f"{method}.log"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    with log_path.open("w", encoding="utf-8") as logf:
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, stdout=logf, stderr=subprocess.STDOUT)
    return {"method": method, "status": "ok" if proc.returncode == 0 else "failed", "returncode": proc.returncode}


# --------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------
def _audit_prefill_by_task(out_root: Path, method: str) -> Dict[str, List[Dict[str, Any]]]:
    by_task: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in _load_jsonl(out_root / "audit" / f"{method}.jsonl"):
        if row.get("phase") == "decode":
            continue
        by_task[str(row.get("task", ""))].append(row)
    return by_task


def build_summary(out_root: Path, methods: List[str], tasks: List[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for method in methods:
        spec = METHOD_SPECS[method]
        audit = _audit_prefill_by_task(out_root, method)
        score_by_task = {r.get("dataset"): r for r in _score_rows(out_root / "runs" / method)}
        for task in tasks:
            sr = score_by_task.get(task, {})
            ta = audit.get(task, [])
            err_vals = [r.get("selected_token_error_pct") for r in ta if r.get("selected_token_error_pct") not in ("", None)]
            abs_err = [abs(float(x)) for x in err_vals] if err_vals else []
            rows.append({
                "method": method, "scheme": spec.get("scheme", ""),
                "rescue_candidate_strategy": spec.get("rescue_candidate_strategy", ""),
                "rescue_score": spec.get("rescue_score", ""),
                "task": task, "category": CATEGORY_MAP.get(task, ""),
                "sample_count": sr.get("num_samples", ""),
                "score": sr.get("score", ""), "status": sr.get("status", "missing"),
                "selected_tokens_mean": _mean(r.get("selected_token_count") for r in ta) or "",
                "selected_pages_mean": _mean(r.get("selected_pages") for r in ta) or "",
                "selected_token_error_pct_mean": (statistics.mean(abs_err) if abs_err else ""),
                "page_expansion_overhead_mean": _mean(r.get("page_expansion_overhead_mean") for r in ta) or "",
                "budget_warning": (max(abs_err) > 0.05) if abs_err else False,
                "fallback_rate": _mean(1.0 if r.get("fallback_used") else 0.0 for r in ta) or 0.0,
                "notes": sr.get("error", ""),
            })
    return rows


def macro_avg(summary: List[Dict[str, Any]], method: str, tasks: Optional[set] = None) -> Optional[float]:
    vals = []
    for r in summary:
        if r.get("method") != method:
            continue
        if tasks and r.get("task") not in tasks:
            continue
        try:
            vals.append(float(r.get("score")))
        except Exception:
            pass
    return _mean(vals)


def task_score(summary: List[Dict[str, Any]], method: str, task: str) -> Optional[float]:
    for r in summary:
        if r.get("method") == method and r.get("task") == task:
            try:
                return float(r.get("score"))
            except Exception:
                return None
    return None


def build_budget_audit(out_root: Path, methods: List[str], tasks: List[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for method in methods:
        spec = METHOD_SPECS[method]
        audit = _audit_prefill_by_task(out_root, method)
        for task in tasks:
            ta = audit.get(task, [])
            if not ta:
                continue
            err = [abs(float(r.get("selected_token_error_pct"))) for r in ta if r.get("selected_token_error_pct") not in ("", None)]
            rows.append({
                "method": method, "scheme": spec.get("scheme", ""), "task": task,
                "selected_tokens_mean": _mean(r.get("selected_token_count") for r in ta) or "",
                "selected_pages_mean": _mean(r.get("selected_pages") for r in ta) or "",
                "target_final_tokens_mean": _mean(r.get("target_final_tokens") for r in ta) or "",
                "selected_token_error_pct_mean": (statistics.mean(err) if err else ""),
                "selected_token_error_pct_max": (max(err) if err else ""),
                "page_expansion_overhead_mean": _mean(r.get("page_expansion_overhead_mean") for r in ta) or "",
                "page_expansion_overhead_max": _mean(r.get("page_expansion_overhead_max") for r in ta) or "",
                "K_mean": _mean(r.get("K") for r in ta) or "",
                "M_mean": _mean(r.get("M") for r in ta) or "",
                "fallback_rate": _mean(1.0 if r.get("fallback_used") else 0.0 for r in ta) or 0.0,
                "budget_warning": (max(err) > 0.05) if err else False,
            })
    return rows


def build_scheme_ranking(summary: List[Dict[str, Any]], methods: List[str], tasks: List[str]) -> List[Dict[str, Any]]:
    base_avg = macro_avg(summary, BASELINE_METHOD)
    rows: List[Dict[str, Any]] = []
    for method in methods:
        spec = METHOD_SPECS[method]
        avg = macro_avg(summary, method)
        delta = (avg - base_avg) if avg is not None and base_avg is not None else None
        wins = 0
        worst_drop = 0.0
        pr_delta = None
        for task in tasks:
            ms = task_score(summary, method, task)
            bs = task_score(summary, BASELINE_METHOD, task)
            if ms is None or bs is None:
                continue
            d = ms - bs
            if d > 0.01:
                wins += 1
            worst_drop = min(worst_drop, d)
            if task == "passage_retrieval_en":
                pr_delta = d
        err_means = [r.get("selected_token_error_pct_mean") for r in summary if r.get("method") == method and r.get("selected_token_error_pct_mean") not in ("", None)]
        err_mean = statistics.mean([float(x) for x in err_means]) if err_means else None
        over_means = [r.get("page_expansion_overhead_mean") for r in summary if r.get("method") == method and r.get("page_expansion_overhead_mean") not in ("", None)]
        over_mean = statistics.mean([float(x) for x in over_means]) if over_means else None
        warn_rate = _mean(1.0 if r.get("budget_warning") else 0.0 for r in summary if r.get("method") == method) or 0.0
        rows.append({
            "method": method, "scheme": spec.get("scheme", ""),
            "avg_score": (round(avg, 4) if avg is not None else ""),
            "delta_vs_fixed_drm_add": (round(delta, 4) if delta is not None else ""),
            "tasks_beating_baseline": wins,
            "worst_task_drop_vs_baseline": round(worst_drop, 4),
            "passage_retrieval_delta": (round(pr_delta, 4) if pr_delta is not None else ""),
            "selected_token_error_pct_mean": (round(err_mean, 4) if err_mean is not None else ""),
            "page_expansion_overhead_mean": (round(over_mean, 4) if over_mean is not None else ""),
            "budget_warning_rate": round(warn_rate, 4),
        })
    rows.sort(key=lambda r: (r["avg_score"] if isinstance(r["avg_score"], float) else -1e9), reverse=True)
    return rows


def stage1_promotion(ranking: List[Dict[str, Any]]) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
    """Apply Stage 1 -> Stage 2 promotion criteria. Returns (promoted_top3, diag_by_method)."""
    diag: Dict[str, Dict[str, Any]] = {}
    eligible: List[Dict[str, Any]] = []
    for r in ranking:
        if r["method"] not in AF_METHODS:
            continue
        avg = r["avg_score"] if isinstance(r["avg_score"], float) else None
        delta = r["delta_vs_fixed_drm_add"] if isinstance(r["delta_vs_fixed_drm_add"], float) else None
        wins = r["tasks_beating_baseline"]
        worst = r["worst_task_drop_vs_baseline"]
        pr = r["passage_retrieval_delta"] if isinstance(r["passage_retrieval_delta"], float) else 0.0
        err = r["selected_token_error_pct_mean"] if isinstance(r["selected_token_error_pct_mean"], float) else 1.0
        warn = r["budget_warning_rate"]
        reasons = []
        c_quality = (delta is not None and delta >= -0.1) or (wins >= 2)
        if not c_quality:
            reasons.append("quality<baseline-0.1 and <2 task wins")
        c_drop = worst > -3.0
        if not c_drop:
            reasons.append(f"a task dropped {worst:.2f}>3.0")
        c_pr = pr > -2.0
        if not c_pr:
            reasons.append(f"passage_retrieval drop {pr:.2f}>2.0")
        c_err = err < 0.05
        if not c_err:
            reasons.append(f"token_error {err:.3f}>=5%")
        c_warn = warn <= 0.34
        if not c_warn:
            reasons.append(f"budget_warning_rate {warn:.2f} high")
        passed = c_quality and c_drop and c_pr and c_err and c_warn
        diag[r["method"]] = {"passed": passed, "reasons": reasons, "avg": avg, "delta": delta}
        if passed:
            eligible.append(r)
    eligible.sort(key=lambda r: (r["avg_score"] if isinstance(r["avg_score"], float) else -1e9), reverse=True)
    promoted = [r["method"] for r in eligible[:3]]
    return promoted, diag


# --------------------------------------------------------------------------
# scheme prior assessment
# --------------------------------------------------------------------------
def write_scheme_prior_assessment(out_root: Path) -> None:
    rows = [
        {"scheme_id": "A", "scheme_name": "Overlap-Rescue Blocks",
         "changes_active_set": "false", "changes_rescue_candidates": "true",
         "uses_query": "false", "uses_semantic_parser": "false",
         "expected_quality_gain": "low_medium", "expected_budget_risk": "low",
         "expected_task_specificity": "low", "implementation_risk": "low",
         "recommended_stage": "stage1",
         "reason": "Sliding windows only fix page-boundary cuts; cheap and budget-safe, but gain likely marginal vs fixed DRM."},
        {"scheme_id": "B", "scheme_name": "Neighbor-Expansion Rescue",
         "changes_active_set": "false", "changes_rescue_candidates": "true",
         "uses_query": "false", "uses_semantic_parser": "false",
         "expected_quality_gain": "low_medium", "expected_budget_risk": "medium",
         "expected_task_specificity": "low", "implementation_risk": "low",
         "recommended_stage": "stage1",
         "reason": "3-page windows help cross-boundary evidence but inflate cost; length penalty must keep budget matched."},
        {"scheme_id": "C", "scheme_name": "Two-Level Fixed Hierarchy",
         "changes_active_set": "false", "changes_rescue_candidates": "true",
         "uses_query": "false", "uses_semantic_parser": "false",
         "expected_quality_gain": "low", "expected_budget_risk": "medium",
         "expected_task_specificity": "medium", "implementation_risk": "medium",
         "recommended_stage": "stage1",
         "reason": "Coarse-to-fine localisation reduces noise but coarse regions may mislocate for retrieval tasks."},
        {"scheme_id": "D", "scheme_name": "Page-Aligned Semantic Rescue",
         "changes_active_set": "false", "changes_rescue_candidates": "true",
         "uses_query": "false", "uses_semantic_parser": "true",
         "expected_quality_gain": "medium", "expected_budget_risk": "medium",
         "expected_task_specificity": "medium", "implementation_risk": "medium",
         "recommended_stage": "stage1",
         "reason": "Semantic units snapped to pages may keep evidence intact for QA; risk of page-expansion overhead and parser noise."},
        {"scheme_id": "E", "scheme_name": "Evidence-Centered Local Windows",
         "changes_active_set": "false", "changes_rescue_candidates": "true",
         "uses_query": "true", "uses_semantic_parser": "false",
         "expected_quality_gain": "medium_high", "expected_budget_risk": "low_medium",
         "expected_task_specificity": "high", "implementation_risk": "medium",
         "recommended_stage": "stage1",
         "reason": "Directly targets QA/retrieval evidence; strong task dependence and lexical query may miss bridge evidence; needs task gate."},
        {"scheme_id": "F", "scheme_name": "Fixed Active + Fine Rescue Fusion",
         "changes_active_set": "false", "changes_rescue_candidates": "true",
         "uses_query": "true", "uses_semantic_parser": "true",
         "expected_quality_gain": "medium_high", "expected_budget_risk": "medium",
         "expected_task_specificity": "medium", "implementation_risk": "high",
         "recommended_stage": "stage1",
         "reason": "Fuses overlap+semantic+evidence pools; highest ceiling but complex scoring and overfitting risk; ablate query source via fusion_fine_no_query."},
    ]
    _write_csv(out_root / "scheme_prior_assessment.csv", rows, [
        "scheme_id", "scheme_name", "changes_active_set", "changes_rescue_candidates",
        "uses_query", "uses_semantic_parser", "expected_quality_gain", "expected_budget_risk",
        "expected_task_specificity", "implementation_risk", "recommended_stage", "reason",
    ])


# --------------------------------------------------------------------------
# case candidates (per-sample, fixed_drm_add vs best A-F method)
# --------------------------------------------------------------------------
def build_case_candidates(out_root: Path, methods: List[str], tasks: List[str], best_method: Optional[str]) -> List[Dict[str, Any]]:
    if not best_method:
        return []
    base_gen = {(r.get("task"), r.get("sample_id")): r for r in _load_jsonl(out_root / "generations" / f"{BASELINE_METHOD}.jsonl")}
    rows: List[Dict[str, Any]] = []
    for rec in _load_jsonl(out_root / "generations" / f"{best_method}.jsonl"):
        key = (rec.get("task"), rec.get("sample_id"))
        base = base_gen.get(key, {})
        try:
            ms = float(rec.get("metric_score", "nan"))
            bs = float(base.get("metric_score", "nan"))
            delta = ms - bs
        except Exception:
            ms, bs, delta = "", "", ""
        rows.append({
            "task": rec.get("task", ""), "sample_id": rec.get("sample_id", ""),
            "method": best_method, "method_score": rec.get("metric_score", ""),
            "fixed_drm_add_score": base.get("metric_score", ""),
            "delta": (round(delta, 4) if isinstance(delta, float) else ""),
            "method_output_preview": str(rec.get("output_text", ""))[:160],
            "fixed_drm_add_output_preview": str(base.get("output_text", ""))[:160],
        })
    rows.sort(key=lambda r: (r["delta"] if isinstance(r["delta"], float) else 0.0))
    return rows


# --------------------------------------------------------------------------
# reports
# --------------------------------------------------------------------------
def _fmt(v: Any) -> str:
    try:
        return f"{float(v):.2f}"
    except Exception:
        return "n/a" if v in ("", None) else str(v)


def write_gated_plan(out_root: Path, ranking1: List[Dict[str, Any]], promoted: List[str], best_method: Optional[str]) -> None:
    lines = [
        "# Gated Strategy Plan (do NOT auto-run)",
        "",
        "This plan is generated for schemes that may only help on a subset of tasks.",
        "It mirrors the AB post-mortem gating lessons: never apply a rescue scheme that",
        "degrades passage_retrieval_en / hotpotqa / 2wikimqa, and always fall back to",
        "fixed DRM-Add when a gate is not satisfied.",
        "",
        "## Candidate gates",
        "- task gate: only enable the scheme on tasks where it shows Stage-1/2 positive delta.",
        "- prompt length gate: short prompts (< target K+M tokens) fall back to fixed/full selected pages.",
        "- page expansion overhead gate: require page_expansion_overhead < 1.08, else fixed DRM-Add.",
        "- budget error risk gate: require predicted abs token error < 5%, else fixed DRM-Add.",
        "- query fallback gate: if query evidence is unavailable (lexical fallback), set beta_query=0 or fall back.",
        "- candidate source confidence gate: if fusion sources disagree strongly, use fixed DRM rescue.",
        "",
        "## Example gate",
        "```",
    ]
    if best_method:
        lines += [
            f"Use {best_method} only if:",
            "  task in {tasks where Stage-2 delta_vs_fixed_drm_add > 0}",
            "  and page_expansion_overhead < 1.08",
            "  and predicted_abs_budget_error < 0.05",
            "  and (scheme uses query => query_evidence_available)",
            "else use fixed_drm_add.",
        ]
    else:
        lines += [
            "No A-F scheme passed Stage-1 promotion; no gate is recommended.",
            "Fall back to fixed_drm_add as the main line.",
        ]
    lines += ["```", ""]
    (out_root / "gated_strategy_plan.md").write_text("\n".join(lines), encoding="utf-8")


def write_report(out_root: Path, config: Dict[str, Any], summary1: List[Dict[str, Any]], ranking1: List[Dict[str, Any]],
                 promoted: List[str], diag1: Dict[str, Dict[str, Any]],
                 summary2: Optional[List[Dict[str, Any]]], ranking2: Optional[List[Dict[str, Any]]],
                 stage2_success: Optional[Dict[str, bool]], run_status: List[Dict[str, Any]],
                 best_method: Optional[str], recommendation: str) -> None:
    base1 = macro_avg(summary1, BASELINE_METHOD)
    L: List[str] = []
    L += [
        "# A-F Block-System Staged Diagnostic Report",
        "",
        "## 1. 本轮目标",
        "围绕 A-F 六种 block-system rescue 方案做 staged diagnostic，判断哪种最值得继续。",
        "**这不是 full LongBench，也不是 latency 实验**；每方法只在小样本子集上与 fixed DRM-Add 做 matched-token 对比。",
        f"Stage 1 = {config['stage1_samples']} samples/task × {len(config['stage1_tasks'])} tasks；"
        f"Stage 2 = {config['stage2_samples']} samples/task × {len(config['stage2_tasks'])} tasks（仅 promoted 方法）。",
        "",
        "## 2. 为什么不再做 always-on attention-boundary (AB)",
        "- AB-BoundaryDRM full Avg = 47.08，低于 fixed DRM-Add full Avg = 47.80（delta = -0.72）。",
        "- AB-SlimInfer 本身已经低于 fixed SlimInfer original，说明退化主要来自 attention-boundary block construction 替换了 active block，而不只是 rescue。",
        "- 剔除 budget warning 的 clean subset 后 AB 仍低于 fixed DRM-Add，失败不是单纯预算错配。",
        "- boundary confidence 对 full 没有贡献（AB-BoundaryDRM vs AB-DRM ≈ -0.01）。",
        "- 因此本轮固定 SlimInfer active set，绝不替换 active blocks；block construction 只用于 rescue candidates / neighbor / fine evidence / gated fallback。",
        "",
        "## 3. A-F 方法解释",
        "- **A Overlap-Rescue**: active 不变；rescue 候选为 64-token 窗口、stride 32 的 overlap windows，命中后展开为 fixed pages。缓解 page 边界切断 evidence。",
        "- **B Neighbor-Expansion**: active 不变；对候选 page 允许 B_{i-1}∪B_i∪B_{i+1} 邻居扩展，由 DRM/length penalty 决定是否值得。",
        "- **C Two-Level Hierarchy**: coarse 256 / fine 64 的 coarse-to-fine 候选池，先定位区域再细粒度救回。",
        "- **D Page-Aligned Semantic**: 语义单元（句/段/文档）贴合 fixed page 边界作为候选；真实 attention 单位仍是 fixed page。",
        "- **E Evidence-Centered Windows**: 用 query lexical relevance 找 evidence center，构造 [c-L, c+L] 局部窗口；query 不可用时自动回退。",
        "- **F Fixed Active + Fine Rescue Fusion**: 合并 overlap + page-aligned semantic + evidence-centered 候选池，统一 DRM + length penalty (+query/semantic) 选择；并用 fusion_fine_no_query 做 query source ablation。",
        "",
        "所有 A-F: active = fixed SlimInfer top-K（与 fixed DRM-Add 完全一致），rescue_policy=add，final = A_fixed ∪ R_extra，无 swap、无删除 active；selection 用 incremental expanded fixed-page cost，token budget 对齐 fixed DRM-Add K+M。",
        "",
        "## 4. 先验评估",
        "见 `scheme_prior_assessment.csv`。先验上更稳的是 A(overlap，预算安全) 与 E(evidence，直接面向 QA 但任务依赖强)；高风险的是 F(fusion，scoring 复杂、过拟合风险) 与 C(coarse 误定位)。",
        "",
        "## 5. Stage 1 结果 (16-sample sanity, 不是结论)",
        "",
    ]
    methods1 = [r["method"] for r in ranking1]
    L.append("| Method | scheme | " + " | ".join(config["stage1_tasks"]) + " | Avg | dVS_fixedDRM | tokErr% | overhead | warnRate |")
    L.append("|---|---|" + "---|" * (len(config["stage1_tasks"]) + 5))
    for r in ranking1:
        m = r["method"]
        vals = [_fmt(task_score(summary1, m, t)) for t in config["stage1_tasks"]]
        L.append(
            f"| {m} | {r['scheme']} | " + " | ".join(vals) +
            f" | {_fmt(r['avg_score'])} | {_fmt(r['delta_vs_fixed_drm_add'])} | "
            f"{_fmt((r['selected_token_error_pct_mean'] or 0)*100 if isinstance(r['selected_token_error_pct_mean'], float) else r['selected_token_error_pct_mean'])} | "
            f"{_fmt(r['page_expansion_overhead_mean'])} | {_fmt(r['budget_warning_rate'])} |"
        )
    L += ["", f"- fixed_drm_add_subset Stage-1 Avg = {_fmt(base1)}（matched-token baseline）。", ""]
    L.append("### Stage 1 淘汰判定")
    for m in AF_METHODS:
        d = diag1.get(m)
        if not d:
            continue
        if d["passed"]:
            L.append(f"- **{m}**: 进入候选 (Avg={_fmt(d['avg'])}, delta={_fmt(d['delta'])})。")
        else:
            L.append(f"- {m}: 淘汰 — {('; '.join(d['reasons']) or 'below criteria')}。")
    L += ["", f"**Stage 1 promoted (top≤3)**: {', '.join(promoted) if promoted else '无（无方案达标）'}", ""]

    L += ["## 6. Stage 2 结果 (64-sample diagnostic)", ""]
    if summary2 and ranking2:
        base2 = macro_avg(summary2, BASELINE_METHOD)
        L.append("| Method | scheme | " + " | ".join(config["stage2_tasks"]) + " | Avg | dVS_fixedDRM | tokErr% | overhead |")
        L.append("|---|---|" + "---|" * (len(config["stage2_tasks"]) + 4))
        for r in ranking2:
            m = r["method"]
            vals = [_fmt(task_score(summary2, m, t)) for t in config["stage2_tasks"]]
            L.append(
                f"| {m} | {r['scheme']} | " + " | ".join(vals) +
                f" | {_fmt(r['avg_score'])} | {_fmt(r['delta_vs_fixed_drm_add'])} | "
                f"{_fmt((r['selected_token_error_pct_mean'] or 0)*100 if isinstance(r['selected_token_error_pct_mean'], float) else r['selected_token_error_pct_mean'])} | "
                f"{_fmt(r['page_expansion_overhead_mean'])} |"
            )
        L += ["", f"- fixed_drm_add_subset Stage-2 Avg = {_fmt(base2)}。", ""]
        if stage2_success:
            L.append("### Stage 2 success 判定")
            for m, ok in stage2_success.items():
                L.append(f"- {m}: {'满足 full 候选标准' if ok else '未满足'}。")
            L.append("")
    else:
        L += ["- Stage 2 未运行（Stage 1 没有方案达标，或 --stage 限制）。", ""]

    L += [
        "## 7. Budget Fairness",
        "见 `stage1_budget_audit.csv`" + (" 与 `stage2_budget_audit.csv`" if summary2 else "") + "。",
        "所有 A-F 用 incremental expanded fixed-page cost 选择，token budget 对齐 fixed DRM-Add K+M。",
        "selected_token_error_pct mean 普遍 < 5% 时方可作公平比较；超 5% 的 method/task 标 budget_warning，不作强结论。",
        "page_expansion_overhead 若明显高于 fixed DRM (≈1.0)，说明候选展开带入额外 page，需在解读时扣除。",
        "",
        "## 8. 任务退化分析 (AB full 的退化任务重点检查)",
    ]
    watch = ["passage_retrieval_en", "hotpotqa", "2wikimqa"]
    src = summary2 if summary2 else summary1
    for m in (promoted or [r["method"] for r in ranking1 if r["method"] in AF_METHODS][:3]):
        deltas = []
        for t in watch:
            ms = task_score(src, m, t)
            bs = task_score(src, BASELINE_METHOD, t)
            if ms is not None and bs is not None:
                deltas.append(f"{t}={ms - bs:+.2f}")
        if deltas:
            L.append(f"- {m}: " + ", ".join(deltas) + " (delta vs fixed DRM-Add)")
    L += ["", "## 9. 是否值得继续", f"- **结论: {recommendation}**", ""]
    if best_method:
        L.append(f"- 当前最佳 A-F 方法: **{best_method}**。")
    L += [
        "",
        "## 10. 下一步建议",
    ]
    if recommendation == "continue full candidate":
        L.append(f"- {best_method} 在 64-sample diagnostic 上稳定优于 fixed DRM-Add 且无关键任务退化：建议扩到 128-sample 再到 full，先补 matched larger baseline 与 latency。")
    elif recommendation == "gated only":
        L.append(f"- {best_method} 仅部分任务有效：按 `gated_strategy_plan.md` 做 gated diagnostic（不要 always-on，不要直接 full）。")
    else:
        L.append("- 没有 A-F 方案稳定超过 fixed DRM-Add：主线回到 fixed DRM-Add，做 ablation / latency / case study。")
    L += [
        "",
        "## 11. 失败与限制",
    ]
    failed = [r for r in run_status if r.get("status") == "failed"]
    if failed:
        L += [f"- {r['method']}: runner failed (returncode={r.get('returncode')})" for r in failed]
    else:
        L.append("- 无 runner 级失败。")
    L += [
        "- 16-sample / 64-sample 结果不是 full 结论；正负 delta 都如实报告。",
        "- 预算不公平 (token error > 5%) 的行不作方法有效结论。",
        "",
        "## Resume",
        "```bash",
        f"bash {out_root / 'resume.sh'}",
        "```",
    ]
    (out_root / "report.md").write_text("\n".join(L), encoding="utf-8")


def write_scripts(out_root: Path, config: Dict[str, Any]) -> None:
    common = (
        f'{PYTHON} tools/run_block_system_af_diagnostic.py '
        f'--out_root "{out_root}" --model_path "{config["model_path"]}" '
        f'--longbench_dir "{config["longbench_dir"]}"'
    )
    (out_root / "run_stage1.sh").write_text(
        "\n".join(["#!/usr/bin/env bash", "set -euo pipefail", f'cd "{REPO_ROOT}"', common + " --stage 1", ""]),
        encoding="utf-8")
    (out_root / "run_stage2.sh").write_text(
        "\n".join(["#!/usr/bin/env bash", "set -euo pipefail", f'cd "{REPO_ROOT}"', common + " --stage 2 --skip_existing", ""]),
        encoding="utf-8")
    (out_root / "resume.sh").write_text(
        "\n".join(["#!/usr/bin/env bash", "set -euo pipefail", f'cd "{REPO_ROOT}"', common + " --stage all --skip_existing", ""]),
        encoding="utf-8")
    for s in ["run_stage1.sh", "run_stage2.sh", "resume.sh"]:
        (out_root / s).chmod(0o755)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_root", type=str, default="")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--longbench_dir", type=str, default=DEFAULT_LONGBENCH)
    parser.add_argument("--stage", type=str, default="all", choices=["1", "2", "all"])
    parser.add_argument("--stage1_samples", type=int, default=16)
    parser.add_argument("--stage2_samples", type=int, default=64)
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--rescue_ratio", type=float, default=0.05)
    parser.add_argument("--lambda_len", type=float, default=0.1)
    parser.add_argument("--beta_query", type=float, default=0.2)
    parser.add_argument("--beta_semantic", type=float, default=0.1)
    parser.add_argument("--stage1_methods", type=str, default=",".join(STAGE1_METHODS))
    parser.add_argument("--stage2_extra_methods", type=str, default="")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.out_root) if args.out_root else REPO_ROOT / "results" / f"rescuekv_block_system_af_diagnostic_{timestamp}"
    out_root.mkdir(parents=True, exist_ok=True)

    config = {
        "timestamp": timestamp, "model_path": args.model_path, "longbench_dir": args.longbench_dir,
        "stage1_tasks": STAGE1_TASKS, "stage2_tasks": STAGE2_TASKS,
        "stage1_samples": args.stage1_samples, "stage2_samples": args.stage2_samples,
        "max_new_tokens": args.max_new_tokens, "rescue_ratio": args.rescue_ratio,
        "lambda_len": args.lambda_len, "beta_query": args.beta_query, "beta_semantic": args.beta_semantic,
        "full_longbench": False, "latency": False,
        "principle": "fixed active set unchanged; only rescue candidates vary; matched K+M token budget vs fixed DRM-Add",
    }
    _write_json(out_root / "config.json", config)
    write_scheme_prior_assessment(out_root)
    write_scripts(out_root, config)

    run_status: List[Dict[str, Any]] = []
    summary1: List[Dict[str, Any]] = []
    ranking1: List[Dict[str, Any]] = []
    promoted: List[str] = []
    diag1: Dict[str, Dict[str, Any]] = {}
    summary2 = ranking2 = None
    stage2_success = None
    best_method: Optional[str] = None
    recommendation = "stop A-F"

    # -------- Stage 1 --------
    if args.stage in ("1", "all"):
        stage1_methods = [m.strip() for m in args.stage1_methods.split(",") if m.strip() and m.strip() in METHOD_SPECS]
        if BASELINE_METHOD not in stage1_methods:
            stage1_methods.insert(0, BASELINE_METHOD)
        for m in stage1_methods:
            print(f"[Stage1] running {m} ...", flush=True)
            run_status.append(run_method(out_root, m, config, STAGE1_TASKS, args.stage1_samples,
                                         dry_run=args.dry_run, skip_existing=args.skip_existing))
        summary1 = build_summary(out_root, stage1_methods, STAGE1_TASKS)
        ranking1 = build_scheme_ranking(summary1, stage1_methods, STAGE1_TASKS)
        budget1 = build_budget_audit(out_root, stage1_methods, STAGE1_TASKS)
        _write_csv(out_root / "stage1_summary.csv", summary1, [
            "method", "scheme", "rescue_candidate_strategy", "rescue_score", "task", "category",
            "sample_count", "score", "status", "selected_tokens_mean", "selected_pages_mean",
            "selected_token_error_pct_mean", "page_expansion_overhead_mean", "budget_warning", "fallback_rate", "notes"])
        _write_csv(out_root / "stage1_budget_audit.csv", budget1, [
            "method", "scheme", "task", "selected_tokens_mean", "selected_pages_mean", "target_final_tokens_mean",
            "selected_token_error_pct_mean", "selected_token_error_pct_max", "page_expansion_overhead_mean",
            "page_expansion_overhead_max", "K_mean", "M_mean", "fallback_rate", "budget_warning"])
        _write_csv(out_root / "stage1_scheme_ranking.csv", ranking1, [
            "method", "scheme", "avg_score", "delta_vs_fixed_drm_add", "tasks_beating_baseline",
            "worst_task_drop_vs_baseline", "passage_retrieval_delta", "selected_token_error_pct_mean",
            "page_expansion_overhead_mean", "budget_warning_rate"])
        if not args.dry_run:
            promoted, diag1 = stage1_promotion(ranking1)

    # -------- Stage 2 --------
    if args.stage in ("2", "all") and not args.dry_run:
        if args.stage == "2":
            # re-derive promoted from existing stage1 ranking if present
            prev = out_root / "stage1_scheme_ranking.csv"
            if prev.exists() and not ranking1:
                with prev.open() as f:
                    ranking1 = [{k: (float(v) if k in {"avg_score", "delta_vs_fixed_drm_add", "worst_task_drop_vs_baseline",
                                                        "passage_retrieval_delta", "selected_token_error_pct_mean",
                                                        "page_expansion_overhead_mean", "budget_warning_rate"} and v not in ("", None) else v)
                                     for k, v in row.items()} for row in csv.DictReader(f)]
                for r in ranking1:
                    r["tasks_beating_baseline"] = int(float(r.get("tasks_beating_baseline", 0) or 0))
                promoted, diag1 = stage1_promotion(ranking1)
                summary1 = build_summary(out_root, [r["method"] for r in ranking1], STAGE1_TASKS)
        extra = [m.strip() for m in args.stage2_extra_methods.split(",") if m.strip() in METHOD_SPECS]
        stage2_methods = ["fixed_sliminfer_original_subset", "fixed_sliminfer_larger_subset",
                          BASELINE_METHOD, "random_add_subset"] + promoted + extra
        seen = set()
        stage2_methods = [m for m in stage2_methods if not (m in seen or seen.add(m))]
        if promoted:
            for m in stage2_methods:
                print(f"[Stage2] running {m} ...", flush=True)
                run_status.append(run_method(out_root, m, config, STAGE2_TASKS, args.stage2_samples,
                                             dry_run=args.dry_run, skip_existing=args.skip_existing))
            summary2 = build_summary(out_root, stage2_methods, STAGE2_TASKS)
            ranking2 = build_scheme_ranking(summary2, stage2_methods, STAGE2_TASKS)
            budget2 = build_budget_audit(out_root, stage2_methods, STAGE2_TASKS)
            # category summary
            cat_rows = []
            for m in stage2_methods:
                row = {"method": m, "scheme": METHOD_SPECS[m].get("scheme", "")}
                for cat in ["SingleDocQA", "MultiDocQA", "Synthetic"]:
                    tasks_in = {t for t, c in CATEGORY_MAP.items() if c == cat and t in STAGE2_TASKS}
                    row[cat] = _fmt(macro_avg(summary2, m, tasks_in)) if tasks_in else ""
                row["Avg"] = _fmt(macro_avg(summary2, m))
                cat_rows.append(row)
            task_delta_rows = []
            for m in promoted:
                for t in STAGE2_TASKS:
                    ms = task_score(summary2, m, t)
                    bs = task_score(summary2, BASELINE_METHOD, t)
                    task_delta_rows.append({"method": m, "task": t, "method_score": ms,
                                            "fixed_drm_add_score": bs,
                                            "delta": (round(ms - bs, 4) if ms is not None and bs is not None else "")})
            _write_csv(out_root / "stage2_summary.csv", summary2, [
                "method", "scheme", "rescue_candidate_strategy", "rescue_score", "task", "category",
                "sample_count", "score", "status", "selected_tokens_mean", "selected_pages_mean",
                "selected_token_error_pct_mean", "page_expansion_overhead_mean", "budget_warning", "fallback_rate", "notes"])
            _write_csv(out_root / "stage2_category_summary.csv", cat_rows,
                       ["method", "scheme", "SingleDocQA", "MultiDocQA", "Synthetic", "Avg"])
            _write_csv(out_root / "stage2_budget_audit.csv", budget2, [
                "method", "scheme", "task", "selected_tokens_mean", "selected_pages_mean", "target_final_tokens_mean",
                "selected_token_error_pct_mean", "selected_token_error_pct_max", "page_expansion_overhead_mean",
                "page_expansion_overhead_max", "K_mean", "M_mean", "fallback_rate", "budget_warning"])
            _write_csv(out_root / "stage2_scheme_ranking.csv", ranking2, [
                "method", "scheme", "avg_score", "delta_vs_fixed_drm_add", "tasks_beating_baseline",
                "worst_task_drop_vs_baseline", "passage_retrieval_delta", "selected_token_error_pct_mean",
                "page_expansion_overhead_mean", "budget_warning_rate"])
            _write_csv(out_root / "stage2_task_delta.csv", task_delta_rows,
                       ["method", "task", "method_score", "fixed_drm_add_score", "delta"])
            # Stage 2 success criteria
            base2 = macro_avg(summary2, BASELINE_METHOD)
            stage2_success = {}
            for m in promoted:
                avg = macro_avg(summary2, m)
                ok = avg is not None and base2 is not None and (avg > base2 + 0.2)
                pr = task_score(summary2, m, "passage_retrieval_en"); pr_b = task_score(summary2, BASELINE_METHOD, "passage_retrieval_en")
                hp = task_score(summary2, m, "hotpotqa"); hp_b = task_score(summary2, BASELINE_METHOD, "hotpotqa")
                wk = task_score(summary2, m, "2wikimqa"); wk_b = task_score(summary2, BASELINE_METHOD, "2wikimqa")
                if pr is not None and pr_b is not None:
                    ok = ok and (pr - pr_b > -1.0)
                if hp is not None and hp_b is not None:
                    ok = ok and (hp - hp_b > -1.0)
                if wk is not None and wk_b is not None:
                    ok = ok and (wk - wk_b > -1.0)
                pos = 0
                for t in ["musique", "passage_count", "qasper"]:
                    ms = task_score(summary2, m, t); bs = task_score(summary2, BASELINE_METHOD, t)
                    if ms is not None and bs is not None and ms - bs > 0.01:
                        pos += 1
                ok = ok and pos >= 2
                err = [r.get("selected_token_error_pct_mean") for r in summary2 if r.get("method") == m and r.get("selected_token_error_pct_mean") not in ("", None)]
                if err:
                    ok = ok and (statistics.mean([float(x) for x in err]) < 0.05)
                stage2_success[m] = bool(ok)
            best_method = max(promoted, key=lambda m: (macro_avg(summary2, m) or -1e9)) if promoted else None
            if best_method and stage2_success.get(best_method):
                recommendation = "continue full candidate"
            elif best_method and any(stage2_success.values()):
                recommendation = "continue full candidate"
            elif best_method:
                # partial signal -> gated
                b_avg = macro_avg(summary2, best_method)
                recommendation = "gated only" if (b_avg is not None and base2 is not None and b_avg >= base2 - 0.2) else "stop A-F"
        else:
            recommendation = "stop A-F"

    # If only stage 1 ran, derive a tentative best/recommendation from Stage 1
    if summary2 is None and ranking1:
        af_ranked = [r for r in ranking1 if r["method"] in AF_METHODS]
        best_method = promoted[0] if promoted else (af_ranked[0]["method"] if af_ranked else None)
        if promoted:
            recommendation = "gated only"  # needs Stage 2 to confirm full
        else:
            recommendation = "stop A-F"

    case_rows = build_case_candidates(out_root,
                                      [r["method"] for r in (ranking2 or ranking1)],
                                      (STAGE2_TASKS if summary2 else STAGE1_TASKS),
                                      best_method)
    _write_csv(out_root / "case_candidates.csv", case_rows,
               ["task", "sample_id", "method", "method_score", "fixed_drm_add_score", "delta",
                "method_output_preview", "fixed_drm_add_output_preview"])
    write_gated_plan(out_root, ranking1, promoted, best_method)
    write_report(out_root, config, summary1, ranking1, promoted, diag1, summary2, ranking2,
                 stage2_success, run_status, best_method, recommendation)

    print(f"OUT_ROOT={out_root}")
    print(f"PROMOTED={promoted}")
    print(f"BEST={best_method}")
    print(f"RECOMMENDATION={recommendation}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
