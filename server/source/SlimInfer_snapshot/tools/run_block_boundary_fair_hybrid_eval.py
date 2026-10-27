#!/usr/bin/env python3
"""Fallback-fix and matched-token fair diagnostic for block boundary methods."""
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
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
if not PYTHON.exists():
    PYTHON = Path(sys.executable)
RUNNER = REPO_ROOT / "experiments" / "run_qwen25_sliminfer_variants.py"

DEFAULT_MODEL = "/home/yu-bai/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
DEFAULT_LONGBENCH = "/home/yu-bai/Desktop/jiaqizhang/HeadKV/data/LongBench"
DEFAULT_TASKS = ["hotpotqa", "2wikimqa", "musique", "qasper", "multifieldqa_en"]

FALLBACK_FIELDS = [
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


def spec(
    method: str,
    block_strategy: str,
    rescue_policy: str = "none",
    rescue_score: str = "delta_o",
    *,
    group: str,
    priority: str = "P3",
    budget_mode: str = "block_count",
    target_source: str = "",
    active_budget_mode: str = "block_count",
    rescue_budget_mode: str = "block_count",
    target_final_source: str = "",
    runner_method: Optional[str] = None,
    scoring: Optional[str] = None,
    decode_policy: Optional[str] = None,
    active_block_strategy: str = "",
    rescue_candidate_strategy: str = "",
    boundary_weight_active: int = 0,
) -> Dict[str, Any]:
    if runner_method is None:
        runner_method = "sliminfer_variant" if rescue_policy == "add" else "sliminfer_original"
    if scoring is None:
        scoring = "delta_o" if runner_method == "sliminfer_variant" else "token_unit_qk"
    if decode_policy is None:
        decode_policy = "reuse_prefill" if runner_method == "sliminfer_variant" else "original"
    return {
        "method": method,
        "block_strategy": block_strategy,
        "rescue_policy": rescue_policy,
        "rescue_score": rescue_score,
        "group": group,
        "priority": priority,
        "budget_mode": budget_mode,
        "target_budget_source": target_source,
        "active_budget_mode": active_budget_mode,
        "rescue_budget_mode": rescue_budget_mode,
        "target_final_source": target_final_source,
        "runner_method": runner_method,
        "scoring": scoring,
        "decode_policy": decode_policy,
        "active_block_strategy": active_block_strategy,
        "rescue_candidate_strategy": rescue_candidate_strategy,
        "boundary_weight_active": boundary_weight_active,
    }


METHOD_SPECS: List[Dict[str, Any]] = [
    spec("fixed_sliminfer_original", "fixed", group="raw", priority="P0"),
    spec("fixed_drm_add", "fixed", "add", "drm", group="raw", priority="P0"),
    spec("semantic_v2_sliminfer_raw", "semantic_v2", group="raw", priority="P1"),
    spec("semantic_v2_drm_add_raw", "semantic_v2", "add", "drm", group="raw", priority="P1"),
    spec("attention_boundary_sliminfer_raw", "attention_boundary", group="raw", priority="P0"),
    spec("attention_boundary_drm_add_raw", "attention_boundary", "add", "drm", group="raw", priority="P0"),
    spec("query_dynamic_v2_sliminfer_raw", "query_dynamic_v2", group="raw", priority="P3"),
    spec("query_dynamic_v2_drm_add_raw", "query_dynamic_v2", "add", "drm", group="raw", priority="P3"),
    spec("hierarchical_sliminfer_raw", "hierarchical", group="raw", priority="P3"),
    spec("hierarchical_drm_add_raw", "hierarchical", "add", "drm", group="raw", priority="P3"),
    spec("fixed_sliminfer_matched_K", "fixed", group="matched_K", priority="P0", budget_mode="token_budget", target_source="fixed_K"),
    spec("semantic_v2_sliminfer_matched_K", "semantic_v2", group="matched_K", priority="P1", budget_mode="token_budget", target_source="fixed_K"),
    spec("attention_boundary_sliminfer_matched_K", "attention_boundary", group="matched_K", priority="P0", budget_mode="token_budget", target_source="fixed_K", boundary_weight_active=0),
    spec("query_dynamic_v2_sliminfer_matched_K", "query_dynamic_v2", group="matched_K", priority="P3", budget_mode="token_budget", target_source="fixed_K"),
    spec("hierarchical_sliminfer_matched_K", "hierarchical", group="matched_K", priority="P3", budget_mode="token_budget", target_source="fixed_K"),
    spec("fixed_drm_add_matched_KplusM", "fixed", "add", "drm", group="matched_drm", priority="P0", budget_mode="token_budget",
         active_budget_mode="token_budget", rescue_budget_mode="token_budget", target_source="fixed_K", target_final_source="fixed_KplusM_drm"),
    spec("semantic_v2_drm_add_matched_KplusM", "semantic_v2", "add", "drm", group="matched_drm", priority="P1", budget_mode="token_budget",
         active_budget_mode="token_budget", rescue_budget_mode="token_budget", target_source="fixed_K", target_final_source="fixed_KplusM_drm"),
    spec("attention_boundary_drm_add_matched_KplusM", "attention_boundary", "add", "drm", group="matched_drm", priority="P0", budget_mode="token_budget",
         active_budget_mode="token_budget", rescue_budget_mode="token_budget", target_source="fixed_K", target_final_source="fixed_KplusM_drm"),
    spec("query_dynamic_v2_drm_add_matched_KplusM", "query_dynamic_v2", "add", "drm", group="matched_drm", priority="P3", budget_mode="token_budget",
         active_budget_mode="token_budget", rescue_budget_mode="token_budget", target_source="fixed_K", target_final_source="fixed_KplusM_drm"),
    spec("hierarchical_drm_add_matched_KplusM", "hierarchical", "add", "drm", group="matched_drm", priority="P3", budget_mode="token_budget",
         active_budget_mode="token_budget", rescue_budget_mode="token_budget", target_source="fixed_K", target_final_source="fixed_KplusM_drm"),
    spec("fixed_sliminfer_matched_attention_tokens", "fixed", group="high_budget", priority="P0", budget_mode="token_budget", target_source="attention_raw"),
    spec("attention_boundary_sliminfer_matched_attention_tokens", "attention_boundary", group="high_budget", priority="P0", budget_mode="token_budget", target_source="attention_raw"),
    spec("hybrid_attention_fixed_drm_matched_KplusM", "hybrid_attention_fixed", "add", "drm", group="hybrid", priority="P2", budget_mode="token_budget",
         active_budget_mode="token_budget", rescue_budget_mode="token_budget", target_source="fixed_K", target_final_source="fixed_KplusM_drm",
         active_block_strategy="attention_boundary", rescue_candidate_strategy="fixed"),
    spec("hybrid_attention_semantic_v2_drm_matched_KplusM", "hybrid_attention_semantic", "add", "drm", group="hybrid", priority="P2", budget_mode="token_budget",
         active_budget_mode="token_budget", rescue_budget_mode="token_budget", target_source="fixed_K", target_final_source="fixed_KplusM_drm",
         active_block_strategy="attention_boundary", rescue_candidate_strategy="semantic_v2"),
    spec("attention_boundary_boundary_drm_matched_KplusM", "attention_boundary", "add", "boundary_drm", group="boundary_drm", priority="P2", budget_mode="token_budget",
         active_budget_mode="token_budget", rescue_budget_mode="token_budget", target_source="fixed_K", target_final_source="fixed_KplusM_drm",
         boundary_weight_active=1),
    spec("hybrid_attention_fixed_boundary_drm_matched_KplusM", "hybrid_attention_fixed", "add", "boundary_drm", group="boundary_drm", priority="P2", budget_mode="token_budget",
         active_budget_mode="token_budget", rescue_budget_mode="token_budget", target_source="fixed_K", target_final_source="fixed_KplusM_drm",
         active_block_strategy="attention_boundary", rescue_candidate_strategy="fixed", boundary_weight_active=1),
]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
    return rows


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _mean(values: Iterable[Any]) -> Optional[float]:
    vals: List[float] = []
    for value in values:
        try:
            if value not in ("", None):
                vals.append(float(value))
        except Exception:
            pass
    return statistics.mean(vals) if vals else None


def _std(values: Iterable[Any]) -> Optional[float]:
    vals: List[float] = []
    for value in values:
        try:
            if value not in ("", None):
                vals.append(float(value))
        except Exception:
            pass
    if not vals:
        return None
    return statistics.pstdev(vals) if len(vals) > 1 else 0.0


def _fmt(value: Any) -> str:
    try:
        return f"{float(value):.2f}"
    except Exception:
        return "n/a" if value in ("", None) else str(value)


def find_previous_dir() -> Optional[Path]:
    dirs = [p for p in (REPO_ROOT / "results").glob("rescuekv_block_boundary_abcd_64_*") if p.is_dir()]
    if not dirs:
        return None
    return max(dirs, key=lambda p: p.stat().st_mtime)


def extract_previous_reference(prev_dir: Optional[Path]) -> Dict[str, Any]:
    ref = {
        "previous_dir": str(prev_dir or ""),
        "budgets": {
            "fixed_K": 6040,
            "fixed_KplusM_larger": 6323,
            "fixed_KplusM_drm": 6342,
            "attention_raw": 9222,
            "hierarchical_raw": 8168,
        },
        "fallback_rates": {"semantic": 0.95, "query_dynamic": 0.51},
    }
    if not prev_dir:
        ref["fallback_note"] = "previous_abcd_dir_missing_defaults_used"
        return ref
    block_rows = _read_csv(prev_dir / "block_layout_summary.csv")
    audit_rows = _read_csv(prev_dir / "audit_summary.csv")

    def selected_for(method: str) -> Optional[float]:
        vals = [r.get("selected_token_count_mean") for r in block_rows if r.get("method") == method]
        out = _mean(vals)
        if out is not None:
            return out
        vals = [r.get("selected_token_count_mean") for r in audit_rows if r.get("method") == method]
        return _mean(vals)

    mapping = {
        "fixed_K": "fixed_sliminfer_original",
        "fixed_KplusM_larger": "fixed_sliminfer_larger",
        "fixed_KplusM_drm": "fixed_drm_add",
        "attention_raw": "attention_boundary_sliminfer",
        "hierarchical_raw": "hierarchical_sliminfer",
    }
    for key, method in mapping.items():
        value = selected_for(method)
        if value is not None:
            ref["budgets"][key] = int(round(value))

    def fallback_for(prefix: str) -> Optional[float]:
        vals = []
        for row in audit_rows:
            if str(row.get("method", "")).startswith(prefix):
                vals.append(row.get("fallback_rate"))
        if vals:
            return _mean(vals)
        vals = []
        for row in block_rows:
            if str(row.get("method", "")).startswith(prefix):
                vals.append(row.get("fallback_rate"))
        return _mean(vals)

    sem = fallback_for("semantic")
    query = fallback_for("query_dynamic")
    if sem is not None:
        ref["fallback_rates"]["semantic"] = float(sem)
    if query is not None:
        ref["fallback_rates"]["query_dynamic"] = float(query)
    for name in ["report.md", "summary.csv", "block_layout_summary.csv", "audit_summary.csv", "config.json", "resume.sh"]:
        ref[f"previous_{name.replace('.', '_')}"] = str(prev_dir / name)
    return ref


def _score_rows(method_dir: Path) -> List[Dict[str, Any]]:
    return _read_csv(method_dir / "longbench_scores.csv")


def _method_complete(out_root: Path, spec_row: Dict[str, Any], tasks: List[str], max_samples: int) -> bool:
    rows = _score_rows(out_root / "runs" / spec_row["method"])
    by_task = {r.get("dataset"): r for r in rows}
    for task in tasks:
        row = by_task.get(task)
        if not row or row.get("status") != "ok":
            return False
        try:
            if int(float(row.get("num_samples", 0))) < max_samples:
                return False
        except Exception:
            return False
    return True


def _budget_value(reference: Dict[str, Any], key: str) -> int:
    if not key:
        return 0
    return int(reference.get("budgets", {}).get(key, 0) or 0)


def _runner_cmd(
    out_root: Path,
    spec_row: Dict[str, Any],
    config: Dict[str, Any],
) -> List[str]:
    method = spec_row["method"]
    target_token_budget = _budget_value(config["previous_reference"], spec_row.get("target_budget_source", ""))
    target_final_tokens = _budget_value(config["previous_reference"], spec_row.get("target_final_source", ""))
    cmd = [
        str(PYTHON), str(RUNNER),
        "--model_path", config["model_path"],
        "--tokenizer_path", config["model_path"],
        "--model_family", "qwen2",
        "--dtype", "fp16",
        "--device", "cuda",
        "--backend", "flash_attention_2",
        "--method", spec_row["runner_method"],
        "--scoring", spec_row["scoring"],
        "--decode_policy", spec_row["decode_policy"],
        "--eval", "longbench",
        "--longbench_dir", config["longbench_dir"],
        "--longbench_tasks", ",".join(config["tasks"]),
        "--max_samples_per_task", str(config["max_samples"]),
        "--generation_max_new_tokens", str(config["max_new_tokens"]),
        "--smoke_mode", "1",
        "--max_smoke_samples", str(config["max_samples"]),
        "--output_dir", str(out_root / "runs" / method),
        "--isolated_run_id", method,
        "--block_strategy", spec_row["block_strategy"],
        "--target_block_tokens", "64",
        "--min_block_tokens", "32",
        "--max_block_tokens", "128",
        "--block_overlap_tokens", "0",
        "--semantic_parser", "task_aware",
        "--semantic_target_block_tokens", "64",
        "--semantic_min_block_tokens", "32",
        "--semantic_max_block_tokens", "128",
        "--semantic_page_align", "1",
        "--max_page_expansion_overhead", "2.0",
        "--attention_boundary_layer", "9",
        "--boundary_percentile", "85",
        "--query_top_ratio", "0.2",
        "--query_neighbor_sentences", "1",
        "--query_high_max_block_tokens", "64",
        "--query_low_max_block_tokens", "128",
        "--query_score_mode", "lexical_bm25",
        "--query_parser", "task_aware",
        "--coarse_block_tokens", "256",
        "--fine_block_tokens", "64",
        "--medium_block_tokens", "128",
        "--top_coarse_ratio", "0.3",
        "--budget_mode", spec_row["budget_mode"],
        "--target_token_budget_source", spec_row.get("target_budget_source") or "custom",
        "--target_token_budget", str(target_token_budget),
        "--token_budget_tolerance", str(config["token_budget_tolerance"]),
        "--selection_cost", "expanded_page_tokens",
        "--score_density_mode", "sqrt_len" if spec_row["budget_mode"] == "token_budget" else "none",
        "--active_budget_mode", spec_row["active_budget_mode"],
        "--rescue_budget_mode", spec_row["rescue_budget_mode"],
        "--target_active_tokens", str(target_token_budget if spec_row["active_budget_mode"] == "token_budget" else 0),
        "--target_final_tokens", str(target_final_tokens),
        "--target_final_tokens_source", spec_row.get("target_final_source") or "custom",
        "--active_block_strategy", spec_row.get("active_block_strategy") or "attention_boundary",
        "--rescue_candidate_strategy", spec_row.get("rescue_candidate_strategy") or "fixed",
        "--hybrid_exclude_active_overlap", "1",
        "--hybrid_overlap_threshold", "0.5",
        "--gamma_boundary", "0.25",
        "--boundary_weight_active_score", str(spec_row.get("boundary_weight_active", 0)),
        "--gamma_active_boundary", "0.1",
        "--block_size", "64",
        "--token_unit_size", "8",
        "--local_query_window", "4",
        "--swap_threshold", "0.9",
        "--prune_layers", "9,18,26",
        "--retain_tokens", "12288,6144,4096",
        "--preserve_first_block",
        "--temperature", "0",
        "--top_p", "1.0",
        "--enable_cpu_offload",
        "--enable_async_kv",
        "--save_layer_stats",
        "--save_block_scores", "aggregate_only",
        "--save_metrics_json",
        "--max_model_len", "32768",
        "--rescue_policy", spec_row["rescue_policy"],
        "--rescue_score", spec_row["rescue_score"],
        "--rescue_ratio", str(config["rescue_ratio"]),
        "--seed", "0",
        "--alpha_diff", "0.5",
        "--lambda_red", "0.25",
        "--eta_rescue", "0.25",
        "--eta_active", "0.25",
        "--rescue_audit_path", str(out_root / "audit" / f"{method}.jsonl"),
        "--block_layout_audit_path", str(out_root / "block_layouts" / f"{method}.jsonl"),
        "--debug_used_block_audit_path", str(out_root / "used_blocks" / f"{method}.jsonl"),
        "--debug_generation_jsonl", str(out_root / "generations" / f"{method}.jsonl"),
        "--debug_first_token_logits_path", str(out_root / "logits" / f"{method}.jsonl"),
        "--save_outputs",
    ]
    if spec_row["runner_method"] == "sliminfer_variant":
        cmd.extend(["--fd_alpha_o", "1.0", "--fd_beta_h", "0.0", "--fd_gamma_s", "0.0"])
    return cmd


def run_method(out_root: Path, spec_row: Dict[str, Any], config: Dict[str, Any], *, dry_run: bool, skip_existing: bool) -> Dict[str, Any]:
    method = spec_row["method"]
    if skip_existing and _method_complete(out_root, spec_row, config["tasks"], int(config["max_samples"])):
        return {"method": method, "status": "skipped_existing", "returncode": 0}
    for sub in ["audit", "block_layouts", "generations", "logits", "used_blocks", "logs"]:
        (out_root / sub).mkdir(parents=True, exist_ok=True)
    for path in [
        out_root / "audit" / f"{method}.jsonl",
        out_root / "block_layouts" / f"{method}.jsonl",
        out_root / "generations" / f"{method}.jsonl",
        out_root / "logits" / f"{method}.jsonl",
        out_root / "used_blocks" / f"{method}.jsonl",
    ]:
        if not (skip_existing and path.exists()):
            path.write_text("", encoding="utf-8")
    cmd = _runner_cmd(out_root, spec_row, config)
    if dry_run:
        return {"method": method, "status": "dry_run", "returncode": 0, "cmd": cmd}
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    log_path = out_root / "logs" / f"{method}.log"
    with log_path.open("w", encoding="utf-8") as logf:
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, stdout=logf, stderr=subprocess.STDOUT)
    return {"method": method, "status": "ok" if proc.returncode == 0 else "failed", "returncode": proc.returncode, "cmd": cmd}


def _audit_rows(out_root: Path, method: str) -> List[Dict[str, Any]]:
    return [r for r in _load_jsonl(out_root / "audit" / f"{method}.jsonl") if r.get("phase") != "decode"]


def _audit_by_task(out_root: Path, method: str) -> Dict[str, List[Dict[str, Any]]]:
    by_task: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in _audit_rows(out_root, method):
        by_task[str(row.get("task", ""))].append(row)
    return by_task


def build_summary(out_root: Path, specs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for sp in specs:
        method = sp["method"]
        audit = _audit_by_task(out_root, method)
        for score_row in _score_rows(out_root / "runs" / method):
            task = score_row.get("dataset", "")
            arows = audit.get(task, [])
            fallback = _mean(1.0 if r.get("fallback_used") else 0.0 for r in arows)
            status = score_row.get("status", "missing")
            if fallback is not None and fallback > 0.2 and sp["block_strategy"] in {"semantic_v2", "query_dynamic_v2", "hybrid_attention_semantic"}:
                status = f"{status};high_fallback"
            err = _mean(abs(float(r.get("selected_token_error_pct") or 0.0)) for r in arows if r.get("selected_token_error_pct") not in ("", None))
            if sp["budget_mode"] == "token_budget" and err is not None and err > 0.05:
                status = f"{status};budget_mismatch"
            rows.append({
                "method": method,
                "block_strategy": sp["block_strategy"],
                "active_block_strategy": sp.get("active_block_strategy", ""),
                "rescue_candidate_strategy": sp.get("rescue_candidate_strategy", ""),
                "budget_mode": sp["budget_mode"],
                "rescue_policy": sp["rescue_policy"],
                "rescue_score": sp["rescue_score"],
                "task": task,
                "sample_count": score_row.get("num_samples", ""),
                "score": score_row.get("score", ""),
                "status": status,
                "selected_token_count_mean": _mean(r.get("selected_token_count") for r in arows) or "",
                "selected_token_ratio_mean": _mean(r.get("selected_token_ratio") for r in arows) or "",
                "fallback_rate": fallback if fallback is not None else "",
                "notes": score_row.get("error", ""),
            })
    return rows


def build_block_layout_summary(out_root: Path, specs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for sp in specs:
        method = sp["method"]
        layout_rows = _load_jsonl(out_root / "block_layouts" / f"{method}.jsonl")
        audit = _audit_by_task(out_root, method)
        tasks = sorted({r.get("task", "") for r in layout_rows} | set(audit.keys()))
        for task in tasks:
            lrows = [r for r in layout_rows if r.get("task") == task]
            lengths: List[Any] = []
            overheads: List[Any] = []
            boundaries: List[Any] = []
            for row in lrows:
                lengths.extend(row.get("block_lengths", []) or [])
                overheads.extend(row.get("page_expansion_overhead", []) or [])
                boundaries.extend(row.get("boundary_confidence", []) or [])
            arows = audit.get(task, [])
            rows.append({
                "method": method,
                "block_strategy": sp["block_strategy"],
                "budget_mode": sp["budget_mode"],
                "task": task,
                "num_blocks_mean": _mean(r.get("num_blocks") for r in lrows) or "",
                "block_len_mean": _mean(lengths) or "",
                "block_len_std": _std(lengths) or "",
                "block_len_min": min([int(x) for x in lengths]) if lengths else "",
                "block_len_max": max([int(x) for x in lengths]) if lengths else "",
                "selected_token_count_mean": _mean(r.get("selected_token_count") for r in arows) or "",
                "page_expansion_overhead_mean": _mean(overheads) or "",
                "fallback_rate": _mean(1.0 if r.get("fallback_used") else 0.0 for r in lrows) or 0.0,
                "boundary_confidence_mean": _mean(boundaries) or "",
            })
    return rows


def build_audit_summary(out_root: Path, specs: List[Dict[str, Any]], reference: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for sp in specs:
        method = sp["method"]
        audit = _audit_rows(out_root, method)
        expected = "K+M" if sp["rescue_policy"] in {"add", "sliminfer_larger"} else "K"
        if sp["budget_mode"] == "token_budget":
            expected = str(_budget_value(reference, sp.get("target_final_source") or sp.get("target_budget_source")))
        rows.append({
            "method": method,
            "block_strategy": sp["block_strategy"],
            "budget_mode": sp["budget_mode"],
            "rescue_policy": sp["rescue_policy"],
            "rescue_score": sp["rescue_score"],
            "expected_budget": expected,
            "observed_final_block_count_mean": _mean(r.get("final_block_count_policy") for r in audit) or "",
            "selected_token_count_mean": _mean(r.get("selected_token_count") for r in audit) or "",
            "duplicate_block_ids": any(bool(r.get("duplicate_block_ids")) for r in audit),
            "protected_block_violation": any(bool(r.get("protected_block_violation")) for r in audit),
            "fallback_rate": _mean(1.0 if r.get("fallback_used") else 0.0 for r in audit) or 0.0,
            "coverage_ok_rate": _mean(1.0 if r.get("coverage_ok", True) else 0.0 for r in audit) or "",
            "D_base_std_mean": _mean((r.get("D_base_stats") or {}).get("std", r.get("D_base_std_mean")) for r in audit) or "",
            "G_std_mean": _mean((r.get("G_diffusion_stats") or {}).get("std", r.get("G_std_mean")) for r in audit) or "",
            "R_std_mean": _mean((r.get("R_active_stats") or {}).get("std", r.get("R_std_mean")) for r in audit) or "",
            "boundary_confidence_mean": _mean(r.get("boundary_confidence_mean") for r in audit) or "",
        })
    return rows


def build_fallback_reason_summary(out_root: Path, specs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen = {(sp["block_strategy"], task) for sp in specs for task in DEFAULT_TASKS}
    for strategy, task in sorted(seen):
        task_rows: List[Dict[str, Any]] = []
        for sp in specs:
            if sp["block_strategy"] != strategy:
                continue
            task_rows.extend([r for r in _audit_rows(out_root, sp["method"]) if r.get("task") == task])
        if not task_rows:
            continue
        counts = Counter()
        for row in task_rows:
            frc = row.get("fallback_reason_counts") or {}
            if isinstance(frc, dict):
                for key, value in frc.items():
                    if key in FALLBACK_FIELDS:
                        try:
                            counts[key] += int(value)
                        except Exception:
                            pass
        out = {
            "block_strategy": strategy,
            "task": task,
            "fallback_rate": _mean(1.0 if r.get("fallback_used") else 0.0 for r in task_rows) or 0.0,
        }
        for key in FALLBACK_FIELDS:
            out[key] = counts.get(key, 0)
        rows.append(out)
    return rows


def _macro(summary_rows: List[Dict[str, Any]], method: str, tasks: Optional[set] = None) -> Optional[float]:
    return _mean(r.get("score") for r in summary_rows if r.get("method") == method and (not tasks or r.get("task") in tasks))


def build_fair_summary(summary_rows: List[Dict[str, Any]], specs: List[Dict[str, Any]], reference: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    fixed_by_group = {
        "matched_K": "fixed_sliminfer_matched_K",
        "matched_drm": "fixed_drm_add_matched_KplusM",
        "hybrid": "fixed_drm_add_matched_KplusM",
        "boundary_drm": "fixed_drm_add_matched_KplusM",
        "high_budget": "fixed_sliminfer_matched_attention_tokens",
    }
    raw_map = {
        "semantic_v2_sliminfer_matched_K": "semantic_v2_sliminfer_raw",
        "attention_boundary_sliminfer_matched_K": "attention_boundary_sliminfer_raw",
        "query_dynamic_v2_sliminfer_matched_K": "query_dynamic_v2_sliminfer_raw",
        "hierarchical_sliminfer_matched_K": "hierarchical_sliminfer_raw",
        "semantic_v2_drm_add_matched_KplusM": "semantic_v2_drm_add_raw",
        "attention_boundary_drm_add_matched_KplusM": "attention_boundary_drm_add_raw",
        "query_dynamic_v2_drm_add_matched_KplusM": "query_dynamic_v2_drm_add_raw",
        "hierarchical_drm_add_matched_KplusM": "hierarchical_drm_add_raw",
    }
    for sp in specs:
        method = sp["method"]
        score = _macro(summary_rows, method)
        selected = _mean(r.get("selected_token_count_mean") for r in summary_rows if r.get("method") == method)
        target_key = sp.get("target_final_source") or sp.get("target_budget_source") or ""
        target = _budget_value(reference, target_key)
        err = ((selected - target) / max(1, target)) if selected is not None and target else ""
        fixed = fixed_by_group.get(sp["group"], "")
        fixed_score = _macro(summary_rows, fixed) if fixed else None
        raw_score = _macro(summary_rows, raw_map.get(method, "")) if raw_map.get(method) else None
        status_vals = {r.get("status") for r in summary_rows if r.get("method") == method}
        rows.append({
            "method": method,
            "block_strategy": sp["block_strategy"],
            "budget_mode": sp["budget_mode"],
            "target_budget_source": target_key,
            "target_token_budget": target,
            "selected_token_count_mean": selected if selected is not None else "",
            "selected_token_error_pct": err,
            "score": score if score is not None else "",
            "delta_vs_fixed_same_budget": (score - fixed_score) if score is not None and fixed_score is not None else "",
            "delta_vs_raw_same_strategy": (score - raw_score) if score is not None and raw_score is not None else "",
            "status": ",".join(sorted(str(x) for x in status_vals if x)),
        })
    return rows


def build_generation_diff(out_root: Path, specs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    fixed_base = {(r.get("task"), r.get("sample_id")): r for r in _load_jsonl(out_root / "generations" / "fixed_sliminfer_original.jsonl")}
    strategy_base: Dict[str, Dict[Tuple[Any, Any], Dict[str, Any]]] = {}
    for sp in specs:
        if "sliminfer_raw" in sp["method"] or sp["method"] in {"fixed_sliminfer_original"}:
            strategy_base[sp["block_strategy"]] = {
                (r.get("task"), r.get("sample_id")): r
                for r in _load_jsonl(out_root / "generations" / f"{sp['method']}.jsonl")
            }
    rows: List[Dict[str, Any]] = []
    for sp in specs:
        for rec in _load_jsonl(out_root / "generations" / f"{sp['method']}.jsonl"):
            key = (rec.get("task"), rec.get("sample_id"))
            fixed = fixed_base.get(key, {})
            same_strategy = strategy_base.get(sp["block_strategy"], {}).get(key, {})
            rows.append({
                "task": rec.get("task", ""),
                "sample_id": rec.get("sample_id", ""),
                "method": sp["method"],
                "block_strategy": sp["block_strategy"],
                "budget_mode": sp["budget_mode"],
                "output_hash": rec.get("output_hash", ""),
                "metric_score": rec.get("metric_score", ""),
                "output_same_as_fixed_sliminfer": rec.get("output_hash") == fixed.get("output_hash") if fixed else "",
                "output_same_as_same_strategy_raw": rec.get("output_hash") == same_strategy.get("output_hash") if same_strategy else "",
                "output_text_preview": str(rec.get("output_text", ""))[:160],
            })
    return rows


def split_block_layouts_by_task(out_root: Path, specs: List[Dict[str, Any]]) -> None:
    for sp in specs:
        method = sp["method"]
        rows = _load_jsonl(out_root / "block_layouts" / f"{method}.jsonl")
        by_task: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_task[str(row.get("task", ""))].append(row)
        for task, task_rows in by_task.items():
            path = out_root / "block_layouts" / f"{method}_{task}.jsonl"
            with path.open("w", encoding="utf-8") as f:
                for row in task_rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_scripts(out_root: Path, config: Dict[str, Any]) -> None:
    args = [
        f'--out_root "{out_root}"',
        f'--model_path "{config["model_path"]}"',
        f'--longbench_dir "{config["longbench_dir"]}"',
        f'--tasks "{",".join(config["tasks"])}"',
        f'--max_samples {config["max_samples"]}',
        f'--max_new_tokens {config["max_new_tokens"]}',
        f'--rescue_ratio {config["rescue_ratio"]}',
    ]
    if config.get("method_filter"):
        args.append(f'--methods "{",".join(config["method_filter"])}"')
    if config.get("priority_filter"):
        args.append(f'--priorities "{",".join(config["priority_filter"])}"')
    base = f'{PYTHON} tools/run_block_boundary_fair_hybrid_eval.py ' + " ".join(args)
    run = "\n".join(["#!/usr/bin/env bash", "set -euo pipefail", f'cd "{REPO_ROOT}"', base, ""])
    resume = "\n".join(["#!/usr/bin/env bash", "set -euo pipefail", f'cd "{REPO_ROOT}"', base + " --skip_existing", ""])
    (out_root / "run_fair_hybrid_abcd_64.sh").write_text(run, encoding="utf-8")
    (out_root / "resume.sh").write_text(resume, encoding="utf-8")
    (out_root / "run_fair_hybrid_abcd_64.sh").chmod(0o755)
    (out_root / "resume.sh").chmod(0o755)


def write_report(
    out_root: Path,
    specs: List[Dict[str, Any]],
    summary_rows: List[Dict[str, Any]],
    fair_rows: List[Dict[str, Any]],
    fallback_rows: List[Dict[str, Any]],
    block_rows: List[Dict[str, Any]],
    audit_rows: List[Dict[str, Any]],
    run_status: List[Dict[str, Any]],
    config: Dict[str, Any],
) -> None:
    prev = config["previous_reference"]
    avg = {sp["method"]: _macro(summary_rows, sp["method"]) for sp in specs}
    fair = {r["method"]: r for r in fair_rows}
    matched_methods = [
        "semantic_v2_sliminfer_matched_K",
        "attention_boundary_sliminfer_matched_K",
        "query_dynamic_v2_sliminfer_matched_K",
        "hierarchical_sliminfer_matched_K",
    ]
    drm_methods = [
        "semantic_v2_drm_add_matched_KplusM",
        "attention_boundary_drm_add_matched_KplusM",
        "query_dynamic_v2_drm_add_matched_KplusM",
        "hierarchical_drm_add_matched_KplusM",
        "hybrid_attention_fixed_drm_matched_KplusM",
        "hybrid_attention_semantic_v2_drm_matched_KplusM",
        "attention_boundary_boundary_drm_matched_KplusM",
        "hybrid_attention_fixed_boundary_drm_matched_KplusM",
    ]
    best_matched = max((m for m in matched_methods if avg.get(m) is not None), key=lambda m: avg.get(m) or -1e9, default="")
    best_drm = max((m for m in drm_methods if avg.get(m) is not None), key=lambda m: avg.get(m) or -1e9, default="")
    sem_v2_fallback = _mean(r.get("fallback_rate") for r in audit_rows if r.get("block_strategy") == "semantic_v2")
    query_v2_fallback = _mean(r.get("fallback_rate") for r in audit_rows if r.get("block_strategy") == "query_dynamic_v2")
    failed = [r for r in run_status if r.get("status") == "failed"]
    missing = [sp["method"] for sp in specs if avg.get(sp["method"]) is None]

    def answer_delta(method: str, baseline: str) -> str:
        mv, bv = avg.get(method), avg.get(baseline)
        if mv is None or bv is None:
            return "未完成"
        return f"{_fmt(mv)} vs {_fmt(bv)}, delta={_fmt(mv - bv)}"

    lines = [
        f"# Block Boundary Fair Hybrid {config['max_samples']}-Sample Diagnostic Report",
        "",
        "## 1. 本轮目标",
        f"这是 fallback fix + matched-token fair comparison + hybrid/boundary-confidence diagnostic；每个任务最多 {config['max_samples']} samples，不是 full LongBench。",
        "",
        "## 2. 上一轮问题复盘",
        f"- 上一轮目录: `{prev.get('previous_dir', '')}`",
        f"- fixed_K token budget reference: {prev['budgets'].get('fixed_K')}",
        f"- fixed_KplusM_drm token budget reference: {prev['budgets'].get('fixed_KplusM_drm')}",
        f"- attention_boundary raw token reference: {prev['budgets'].get('attention_raw')}",
        f"- semantic fallback previous ~= {_fmt(prev['fallback_rates'].get('semantic'))}; query_dynamic fallback previous ~= {_fmt(prev['fallback_rates'].get('query_dynamic'))}",
        "- attention/hierarchical raw selected_tokens 更多，因此 raw gain 标为 not fair，核心看 matched-token。",
        "",
        "## 3. fallback 修复说明",
        "- semantic_v2 使用 prompt-covering task-aware parser、regex sentence/soft boundary、overlong split、greedy unit merge，并记录 fallback_reason_counts。",
        "- query_dynamic_v2 复用 semantic_v2 units，再用 lexical/BM25/number/entity/title overlap 做 query-aware granularity。",
        "- BlockLayout 记录 token spans、char spans、expanded fixed pages、semantic/expanded token budget、page expansion overhead、coverage flags。",
        "",
        "## 4. 公平比较设计",
        "- block_count mode 用于复现 raw 逻辑。",
        "- matched-token mode 使用 expanded_page_tokens 作为预算，因为真实 attention 仍消耗 fixed KV pages。",
        "- token-budget DRM 不再固定救 M 个 variable blocks，而是控制 final selected tokens 接近 fixed_drm_add budget。",
        "",
        "## 5. 五种组合思路的结果",
        "",
        "### A. BlockLayout + SlimInfer-only",
        f"- matched K 下当前最好: {best_matched or '未完成'} (Avg={_fmt(avg.get(best_matched))})",
        f"- semantic_v2_sliminfer_matched_K: {answer_delta('semantic_v2_sliminfer_matched_K', 'fixed_sliminfer_matched_K')}",
        f"- attention_boundary_sliminfer_matched_K: {answer_delta('attention_boundary_sliminfer_matched_K', 'fixed_sliminfer_matched_K')}",
        f"- query_dynamic_v2_sliminfer_matched_K: {answer_delta('query_dynamic_v2_sliminfer_matched_K', 'fixed_sliminfer_matched_K')}",
        f"- hierarchical_sliminfer_matched_K: {answer_delta('hierarchical_sliminfer_matched_K', 'fixed_sliminfer_matched_K')}",
        "",
        "### B. BlockLayout + SlimInfer larger / matched-token",
        f"- high-budget fixed vs attention: {answer_delta('attention_boundary_sliminfer_matched_attention_tokens', 'fixed_sliminfer_matched_attention_tokens')}",
        "- 若 attention raw 提升在 high-budget fixed 对照下消失，则上一轮提升主要来自 token 更多。",
        "",
        "### C. BlockLayout + Token-budget DRM-Add",
        f"- matched K+M 下当前最好: {best_drm or '未完成'} (Avg={_fmt(avg.get(best_drm))})",
        f"- semantic_v2_drm_add_matched_KplusM: {answer_delta('semantic_v2_drm_add_matched_KplusM', 'fixed_drm_add_matched_KplusM')}",
        f"- attention_boundary_drm_add_matched_KplusM: {answer_delta('attention_boundary_drm_add_matched_KplusM', 'fixed_drm_add_matched_KplusM')}",
        f"- query_dynamic_v2_drm_add_matched_KplusM: {answer_delta('query_dynamic_v2_drm_add_matched_KplusM', 'fixed_drm_add_matched_KplusM')}",
        f"- hierarchical_drm_add_matched_KplusM: {answer_delta('hierarchical_drm_add_matched_KplusM', 'fixed_drm_add_matched_KplusM')}",
        "",
        "### D. Hybrid selection",
        f"- hybrid_attention_fixed_drm_matched_KplusM: {answer_delta('hybrid_attention_fixed_drm_matched_KplusM', 'fixed_drm_add_matched_KplusM')}",
        f"- hybrid_attention_semantic_v2_drm_matched_KplusM: {answer_delta('hybrid_attention_semantic_v2_drm_matched_KplusM', 'fixed_drm_add_matched_KplusM')}",
        "",
        "### E. Boundary-confidence weighted DRM",
        f"- attention boundary_drm vs ordinary drm: {answer_delta('attention_boundary_boundary_drm_matched_KplusM', 'attention_boundary_drm_add_matched_KplusM')}",
        f"- hybrid boundary_drm vs ordinary hybrid: {answer_delta('hybrid_attention_fixed_boundary_drm_matched_KplusM', 'hybrid_attention_fixed_drm_matched_KplusM')}",
        "",
        "## 6. 核心问题回答",
        f"- semantic_v2 fallback 是否显著降低: previous={_fmt(prev['fallback_rates'].get('semantic'))}, current={_fmt(sem_v2_fallback)}。",
        f"- query_dynamic_v2 fallback 是否显著降低: previous={_fmt(prev['fallback_rates'].get('query_dynamic'))}, current={_fmt(query_v2_fallback)}。",
        f"- matched-token 下 A/B/C/D 谁最好: {best_matched or '未完成'}。",
        f"- attention_boundary raw gain matched-token 后是否仍存在: {answer_delta('attention_boundary_sliminfer_matched_K', 'fixed_sliminfer_matched_K')}。",
        f"- fixed high-budget 是否追上 attention raw: {answer_delta('attention_boundary_sliminfer_matched_attention_tokens', 'fixed_sliminfer_matched_attention_tokens')}。",
        f"- token-budget DRM 是否更公平/更好: 预算误差见 fair_summary；分数看 {best_drm or '未完成'}。",
        f"- hybrid 是否解决 attention_boundary_drm_add 不稳定: {answer_delta('hybrid_attention_fixed_drm_matched_KplusM', 'attention_boundary_drm_add_matched_KplusM')}。",
        f"- boundary-confidence 是否有帮助: {answer_delta('attention_boundary_boundary_drm_matched_KplusM', 'attention_boundary_drm_add_matched_KplusM')}。",
        f"- 最值得 full LongBench: {best_drm or best_matched or '暂无'}。",
        "",
        "## 7. 结果表",
        "",
        "| Method | " + " | ".join(config["tasks"]) + " | Avg | Status |",
        "|---|" + "---|" * (len(config["tasks"]) + 2),
    ]
    for sp in specs:
        method = sp["method"]
        vals = [_fmt(_macro(summary_rows, method, {task})) for task in config["tasks"]]
        statuses = {r.get("status") for r in summary_rows if r.get("method") == method}
        lines.append(f"| {method} | " + " | ".join(vals) + f" | {_fmt(avg.get(method))} | {','.join(sorted(str(s) for s in statuses if s))} |")

    lines.extend(["", "## 8. Block layout / fallback summary", ""])
    lines.append("| Strategy | fallback_rate |")
    lines.append("|---|---:|")
    for strategy in sorted({r.get("block_strategy") for r in audit_rows}):
        lines.append(f"| {strategy} | {_fmt(_mean(r.get('fallback_rate') for r in audit_rows if r.get('block_strategy') == strategy))} |")

    lines.extend([
        "",
        "## 9. 公平性结论",
        "- raw block-count gain 一律不能直接解释为 boundary 真实有效，尤其 attention/hierarchical 会选中更多 expanded_page_tokens。",
        "- 只有 matched-token 仍超过 fixed same budget，才可作为 boundary construction 的正证据。",
        "",
        "## 10. 与 SlimInfer / DRM-Add 的关系",
        "- 若 SlimInfer-only 新 block 有效，可转为 Boundary-Aware SlimInfer 主线。",
        "- 若 token-budget DRM 有效，可转为 Boundary-Aware DRM-Add。",
        "- 若 hybrid 有效，可转为 Attention-active + Fine-grained-rescue。",
        "- 若都无效，保留 fixed DRM-Add，block construction 放 appendix negative result。",
        "",
        "## 11. 失败与限制",
    ])
    if failed:
        lines.extend([f"- {r['method']}: failed, returncode={r.get('returncode')}" for r in failed])
    else:
        lines.append("- 无 runner 级失败。")
    if missing:
        lines.append(f"- partial/missing methods: {', '.join(missing)}")
    high_fallback = [r for r in audit_rows if float(r.get("fallback_rate") or 0.0) > 0.2 and r.get("block_strategy") in {"semantic_v2", "query_dynamic_v2", "hybrid_attention_semantic"}]
    for row in high_fallback:
        lines.append(f"- {row['method']}: high_fallback={_fmt(row.get('fallback_rate'))}")
    lines.extend([
        "- attention boundary 仍使用 page-level pooled representations，不是 token-level attention profile。",
        "- variable block 映射 fixed KV pages 会产生 page expansion overhead。",
        "- 本轮未做 latency breakdown。",
        "",
        "## 12. 下一步建议",
    ])
    best_delta = fair.get(best_drm or best_matched or "", {}).get("delta_vs_fixed_same_budget", "")
    try:
        worthwhile = float(best_delta) > 0
    except Exception:
        worthwhile = False
    if worthwhile:
        lines.append("- 有 matched-token 正增益，建议下一步跑 full LongBench，并加入 fixed SlimInfer / fixed DRM-Add / SlimInfer larger / Random-Add / latency / case study。")
    else:
        lines.append("- 若 matched-token 无正增益，不建议转主线；保留 fixed DRM-Add，把 block construction 作为 appendix/negative diagnostic。")
    lines.append("")
    lines.append("## Resume")
    lines.append("```bash")
    lines.append(f"bash {out_root / 'resume.sh'}")
    lines.append("```")
    (out_root / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_root", type=str, default="")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--longbench_dir", type=str, default=DEFAULT_LONGBENCH)
    parser.add_argument("--tasks", type=str, default=",".join(DEFAULT_TASKS))
    parser.add_argument("--max_samples", type=int, default=64)
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--rescue_ratio", type=float, default=0.05)
    parser.add_argument("--token_budget_tolerance", type=float, default=0.05)
    parser.add_argument("--methods", type=str, default="")
    parser.add_argument("--priorities", type=str, default="")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.out_root) if args.out_root else REPO_ROOT / "results" / f"rescuekv_block_boundary_fair_hybrid_64_{timestamp}"
    out_root.mkdir(parents=True, exist_ok=True)
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    specs = list(METHOD_SPECS)
    method_filter: List[str] = []
    if args.methods:
        wanted = {m.strip() for m in args.methods.split(",") if m.strip()}
        method_filter = [sp["method"] for sp in specs if sp["method"] in wanted]
        specs = [sp for sp in specs if sp["method"] in wanted]
    priority_filter: List[str] = []
    if args.priorities:
        wanted_p = {p.strip() for p in args.priorities.split(",") if p.strip()}
        priority_filter = sorted(wanted_p)
        specs = [sp for sp in specs if sp["priority"] in wanted_p]

    prev_dir = find_previous_dir()
    reference = extract_previous_reference(prev_dir)
    config = {
        "timestamp": timestamp,
        "model_path": args.model_path,
        "longbench_dir": args.longbench_dir,
        "tasks": tasks,
        "max_samples": args.max_samples,
        "max_new_tokens": args.max_new_tokens,
        "rescue_ratio": args.rescue_ratio,
        "token_budget_tolerance": args.token_budget_tolerance,
        "sample_mode": f"{args.max_samples}-sample diagnostic",
        "full_longbench": False,
        "methods": [sp["method"] for sp in specs],
        "method_filter": method_filter,
        "priority_filter": priority_filter,
        "previous_reference": reference,
    }
    _write_json(out_root / "config.json", config)
    write_scripts(out_root, config)

    run_status: List[Dict[str, Any]] = []
    for sp in specs:
        print(f"Running {sp['method']} ({sp['group']}, {sp['block_strategy']}) ...", flush=True)
        run_status.append(run_method(out_root, sp, config, dry_run=args.dry_run, skip_existing=args.skip_existing))

    split_block_layouts_by_task(out_root, specs)
    summary_rows = build_summary(out_root, specs)
    block_rows = build_block_layout_summary(out_root, specs)
    audit_rows = build_audit_summary(out_root, specs, reference)
    fair_rows = build_fair_summary(summary_rows, specs, reference)
    fallback_rows = build_fallback_reason_summary(out_root, specs)
    generation_rows = build_generation_diff(out_root, specs)

    _write_csv(out_root / "summary.csv", summary_rows, [
        "method", "block_strategy", "active_block_strategy", "rescue_candidate_strategy", "budget_mode",
        "rescue_policy", "rescue_score", "task", "sample_count", "score", "status",
        "selected_token_count_mean", "selected_token_ratio_mean", "fallback_rate", "notes",
    ])
    _write_csv(out_root / "fair_summary.csv", fair_rows, [
        "method", "block_strategy", "budget_mode", "target_budget_source", "target_token_budget",
        "selected_token_count_mean", "selected_token_error_pct", "score",
        "delta_vs_fixed_same_budget", "delta_vs_raw_same_strategy", "status",
    ])
    _write_csv(out_root / "fallback_reason_summary.csv", fallback_rows, [
        "block_strategy", "task", "fallback_rate", *FALLBACK_FIELDS,
    ])
    _write_csv(out_root / "block_layout_summary.csv", block_rows, [
        "method", "block_strategy", "budget_mode", "task", "num_blocks_mean", "block_len_mean",
        "block_len_std", "block_len_min", "block_len_max", "selected_token_count_mean",
        "page_expansion_overhead_mean", "fallback_rate", "boundary_confidence_mean",
    ])
    _write_csv(out_root / "audit_summary.csv", audit_rows, [
        "method", "block_strategy", "budget_mode", "rescue_policy", "rescue_score", "expected_budget",
        "observed_final_block_count_mean", "selected_token_count_mean", "duplicate_block_ids",
        "protected_block_violation", "fallback_rate", "coverage_ok_rate", "D_base_std_mean",
        "G_std_mean", "R_std_mean", "boundary_confidence_mean",
    ])
    _write_csv(out_root / "generation_diff.csv", generation_rows, [
        "task", "sample_id", "method", "block_strategy", "budget_mode", "output_hash", "metric_score",
        "output_same_as_fixed_sliminfer", "output_same_as_same_strategy_raw", "output_text_preview",
    ])
    write_report(out_root, specs, summary_rows, fair_rows, fallback_rows, block_rows, audit_rows, run_status, config)
    print(out_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
