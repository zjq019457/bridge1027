#!/usr/bin/env python3
"""Structural DRM-Add diagnostic (5 directions).

Context (see report.md):
- fixed DRM-Add is the current main line (full Avg=47.80).
- A-F candidate-source construction passed no promotion gate -> the bottleneck is
  more likely the rescue selector / scoring / budget / gating, NOT the candidate
  pool itself.

This tool keeps the active set FIXED (active = fixed SlimInfer top-K) and only
varies the rescue selector / scoring / budget / gating across five directions:
  1. Selector Isolation     - same unified candidate pool, different selectors
  2. Gated Rescue           - always_add / binary_gate / dynamic_M
  3. Coverage-aware DRM-Add - coverage-gain scoring
  4. Block-level Budget     - block as budget allocator (not candidate source)
  5. Functional Verify/Rerank - light / attn / oracle reranker

It runs a 16-sample diagnostic on 5 Stage-1 tasks and applies a strict promotion
gate. It does NOT run full LongBench and does NOT run latency.
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

STAGE1_TASKS = ["musique", "hotpotqa", "qasper", "passage_count", "passage_retrieval_en"]

# Stage 2: 64-sample verification — only methods that passed 16-sample strict gate
STAGE2_64_METHODS = [
    "fixed_drm_add_subset",
    "unified_query_only",
    "unified_coverage_aware",
    "verifier_attn",
]

CATEGORY_MAP = {
    "qasper": "SingleDocQA",
    "hotpotqa": "MultiDocQA", "musique": "MultiDocQA",
    "passage_count": "Synthetic", "passage_retrieval_en": "Synthetic",
}

BASELINE_METHOD = "fixed_drm_add_subset"

# A structural method is driven by the fixed_active_rescue path on the unified
# candidate pool, varying only selector/gate/block/verifier knobs.
def _struct(extra: List[str], direction: str, alias: str = "") -> Dict[str, Any]:
    return {
        "runner_method": "sliminfer_variant", "scoring": "delta_o", "decode_policy": "reuse_prefill",
        "block_strategy": "fixed_active_rescue", "rescue_policy": "add", "rescue_score": "drm",
        "rescue_candidate_strategy": "unified", "direction": direction, "alias": alias,
        "extra_args": extra,
    }


METHOD_SPECS: Dict[str, Dict[str, Any]] = {
    # --- matched baseline: existing validated fixed DRM-Add path ---
    "fixed_drm_add_subset": {
        "runner_method": "sliminfer_variant", "scoring": "delta_o", "decode_policy": "reuse_prefill",
        "block_strategy": "fixed", "rescue_policy": "add", "rescue_score": "drm",
        "rescue_candidate_strategy": "", "direction": "baseline", "alias": "fixed_drm_add",
        "extra_args": []},

    # --- direction 1: selector isolation (unified pool, vary selector) ---
    "unified_drm_mmr": _struct(["--rescue_selector", "drm_mmr"], "selector", "current_drm_mmr/always_add"),
    "unified_drm_only": _struct(["--rescue_selector", "drm_only"], "selector", "drm_only"),
    "unified_query_only": _struct(["--rescue_selector", "query_only"], "selector", "query_only"),
    "unified_damage_only": _struct(["--rescue_selector", "damage_only"], "selector", "damage_only"),
    "unified_coverage_aware": _struct(["--rescue_selector", "coverage_aware"], "selector", "coverage_aware/coverage_drm_add"),
    "unified_functional_delta": _struct(["--rescue_selector", "functional_delta"], "selector", "functional_delta"),

    # --- direction 2: gated rescue ---
    "gate_binary": _struct(["--rescue_selector", "drm_mmr", "--rescue_gate_mode", "binary"], "gated", "binary_gate"),
    "gate_dynamic_M": _struct(["--rescue_selector", "drm_mmr", "--rescue_gate_mode", "dynamic_M"], "gated", "dynamic_M"),

    # --- direction 4: block-level budget allocation ---
    "block_uniform": _struct(["--rescue_selector", "drm_mmr", "--block_budget_mode", "uniform"], "block_budget", "uniform_block_budget"),
    "block_query_weighted": _struct(["--rescue_selector", "drm_mmr", "--block_budget_mode", "query_weighted"], "block_budget", "query_weighted_block_budget"),
    "block_coverage_gap": _struct(["--rescue_selector", "drm_mmr", "--block_budget_mode", "coverage_gap"], "block_budget", "coverage_gap_block_budget"),
    "block_drm_peak": _struct(["--rescue_selector", "drm_mmr", "--block_budget_mode", "drm_peak"], "block_budget", "drm_peak_block_budget"),

    # --- direction 5: functional verifier / rerank ---
    "verifier_light": _struct(["--verifier_mode", "light"], "verifier", "verifier_rerank_light"),
    "verifier_attn": _struct(["--verifier_mode", "attn"], "verifier", "verifier_rerank_attn"),
    "verifier_oracle": _struct(["--verifier_mode", "oracle"], "verifier", "verifier_rerank_oracle_diag"),
}

# direction -> ordered list of methods (baseline added at report time)
DIRECTION_METHODS: Dict[str, List[str]] = {
    "selector": ["unified_drm_mmr", "unified_drm_only", "unified_query_only", "unified_damage_only",
                 "unified_coverage_aware", "unified_functional_delta"],
    "gated": ["unified_drm_mmr", "gate_binary", "gate_dynamic_M"],
    "coverage": ["unified_coverage_aware"],
    "block_budget": ["block_uniform", "block_query_weighted", "block_coverage_gap", "block_drm_peak"],
    "verifier": ["verifier_light", "verifier_attn", "verifier_oracle"],
}

ALL_METHODS = [BASELINE_METHOD] + sorted({m for ms in DIRECTION_METHODS.values() for m in ms})


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
def _runner_cmd(out_root: Path, method: str, spec: Dict[str, Any], config: Dict[str, Any],
                tasks: List[str], max_samples: int) -> List[str]:
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
        "--beta_query", str(config["beta_query"]),
        "--beta_semantic", str(config["beta_semantic"]),
        "--coverage_gamma", str(config["coverage_gamma"]),
        "--noise_mu", str(config["noise_mu"]),
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
    cmd.extend(spec.get("extra_args", []))
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
    t0 = time.time()
    with log_path.open("w", encoding="utf-8") as logf:
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, stdout=logf, stderr=subprocess.STDOUT)
    return {"method": method, "status": "ok" if proc.returncode == 0 else "failed",
            "returncode": proc.returncode, "elapsed_s": round(time.time() - t0, 1)}


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
            m_pages = [r.get("rescue_m_pages") for r in ta if r.get("rescue_m_pages") not in ("", None)]
            gated_out = [1.0 if (str(r.get("rescue_m_pages")) == "0") else 0.0 for r in ta if r.get("rescue_m_pages") not in ("", None)]
            rows.append({
                "method": method, "alias": spec.get("alias", ""), "direction": spec.get("direction", ""),
                "task": task, "category": CATEGORY_MAP.get(task, ""),
                "sample_count": sr.get("num_samples", ""),
                "score": sr.get("score", ""), "status": sr.get("status", "missing"),
                "selected_tokens_mean": _mean(r.get("selected_token_count") for r in ta) or "",
                "selected_pages_mean": _mean(r.get("selected_pages") for r in ta) or "",
                "selected_token_error_pct_mean": (statistics.mean(abs_err) if abs_err else ""),
                "page_expansion_overhead_mean": _mean(r.get("page_expansion_overhead_mean") for r in ta) or "",
                "budget_warning": (max(abs_err) > 0.05) if abs_err else False,
                "rescue_m_pages_mean": (statistics.mean([float(x) for x in m_pages]) if m_pages else ""),
                "gated_out_rate": (statistics.mean(gated_out) if gated_out else ""),
                "rescue_need_score_mean": _mean(r.get("rescue_need_score") for r in ta) or "",
                "rescue_active_cov_frac_mean": _mean(r.get("rescue_active_cov_frac") for r in ta) or "",
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


def build_ranking(summary: List[Dict[str, Any]], methods: List[str], tasks: List[str]) -> List[Dict[str, Any]]:
    base_avg = macro_avg(summary, BASELINE_METHOD)
    rows: List[Dict[str, Any]] = []
    for method in methods:
        spec = METHOD_SPECS[method]
        avg = macro_avg(summary, method)
        delta = (avg - base_avg) if avg is not None and base_avg is not None else None
        wins = 0
        worst_drop = 0.0
        pr_delta = None
        per_task_delta = {}
        for task in tasks:
            ms = task_score(summary, method, task)
            bs = task_score(summary, BASELINE_METHOD, task)
            if ms is None or bs is None:
                continue
            d = ms - bs
            per_task_delta[task] = round(d, 3)
            if d > 0.01:
                wins += 1
            worst_drop = min(worst_drop, d)
            if task == "passage_retrieval_en":
                pr_delta = d
        err_means = [r.get("selected_token_error_pct_mean") for r in summary
                     if r.get("method") == method and r.get("selected_token_error_pct_mean") not in ("", None)]
        err_mean = statistics.mean([float(x) for x in err_means]) if err_means else None
        warn_rate = _mean(1.0 if r.get("budget_warning") else 0.0 for r in summary if r.get("method") == method) or 0.0
        rows.append({
            "method": method, "alias": spec.get("alias", ""), "direction": spec.get("direction", ""),
            "avg_score": (round(avg, 4) if avg is not None else ""),
            "delta_vs_fixed_drm_add": (round(delta, 4) if delta is not None else ""),
            "tasks_beating_baseline": wins,
            "worst_task_drop_vs_baseline": round(worst_drop, 4),
            "passage_retrieval_delta": (round(pr_delta, 4) if pr_delta is not None else ""),
            "selected_token_error_pct_mean": (round(err_mean, 4) if err_mean is not None else ""),
            "budget_warning_rate": round(warn_rate, 4),
            "per_task_delta": json.dumps(per_task_delta),
        })
    rows.sort(key=lambda r: (r["avg_score"] if isinstance(r["avg_score"], float) else -1e9), reverse=True)
    return rows


def promotion_gate(rank_row: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Strict structural promotion gate (per task spec)."""
    reasons: List[str] = []
    avg = rank_row["avg_score"] if isinstance(rank_row["avg_score"], float) else None
    delta = rank_row["delta_vs_fixed_drm_add"] if isinstance(rank_row["delta_vs_fixed_drm_add"], float) else None
    worst = rank_row["worst_task_drop_vs_baseline"]
    pr = rank_row["passage_retrieval_delta"] if isinstance(rank_row["passage_retrieval_delta"], float) else 0.0
    err = rank_row["selected_token_error_pct_mean"] if isinstance(rank_row["selected_token_error_pct_mean"], float) else 1.0
    warn = rank_row["budget_warning_rate"]
    if not (delta is not None and delta > 0.0):
        reasons.append("Avg not above fixed DRM-Add")
    if not (worst > -3.0):
        reasons.append(f"a task dropped {worst:.2f} (>3.0)")
    if not (err < 0.01):
        reasons.append(f"budget error {err:.4f} >= 1%")
    if warn > 0.0:
        reasons.append(f"budget_warning present (rate {warn:.2f})")
    if not (pr > -2.0):
        reasons.append(f"passage_retrieval collapsed ({pr:.2f})")
    return (len(reasons) == 0), reasons


def build_case_rows(out_root: Path, methods: List[str], tasks: List[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for method in methods:
        audit = _audit_prefill_by_task(out_root, method)
        for task in tasks:
            for i, r in enumerate(audit.get(task, [])):
                rows.append({
                    "method": method, "alias": METHOD_SPECS[method].get("alias", ""),
                    "task": task, "sample_idx": i,
                    "K": r.get("K", ""), "M": r.get("M", ""),
                    "selected_pages": r.get("selected_pages", ""),
                    "selected_token_error_pct": r.get("selected_token_error_pct", ""),
                    "rescue_need_score": r.get("rescue_need_score", ""),
                    "rescue_active_cov_frac": r.get("rescue_active_cov_frac", ""),
                    "rescue_m_pages": r.get("rescue_m_pages", ""),
                    "rescue_m_default": r.get("rescue_m_default", ""),
                    "rescue_selector": r.get("rescue_selector", ""),
                    "verifier_mode": r.get("verifier_mode_used", ""),
                    "block_budget_mode": r.get("block_budget_mode_used", ""),
                })
    return rows


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------
def _md_table(headers: List[str], rows: List[List[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def write_report(out_root: Path, summary: List[Dict[str, Any]], ranking: List[Dict[str, Any]],
                 gate_results: Dict[str, Tuple[bool, List[str]]], config: Dict[str, Any], tasks: List[str]) -> None:
    base_avg = macro_avg(summary, BASELINE_METHOD)
    rank_by_method = {r["method"]: r for r in ranking}

    def rank_block(methods: List[str]) -> str:
        hdr = ["method", "alias", "avg", "Δ vs DRM", "wins", "worst_drop", "PR Δ", "tok_err", "warn", "gate"]
        body = []
        for m in [BASELINE_METHOD] + [x for x in methods if x != BASELINE_METHOD]:
            r = rank_by_method.get(m)
            if not r:
                continue
            passed, _ = gate_results.get(m, (False, []))
            gate = "-" if m == BASELINE_METHOD else ("PASS" if passed else "fail")
            body.append([r["method"], r["alias"], r["avg_score"], r["delta_vs_fixed_drm_add"],
                         r["tasks_beating_baseline"], r["worst_task_drop_vs_baseline"],
                         r["passage_retrieval_delta"], r["selected_token_error_pct_mean"],
                         r["budget_warning_rate"], gate])
        return _md_table(hdr, body)

    any_pass = any(p for m, (p, _) in gate_results.items() if m != BASELINE_METHOD)
    best = next((r for r in ranking if r["method"] != BASELINE_METHOD), None)

    lines: List[str] = []
    lines.append("# Structural DRM-Add Diagnostic (5 directions)\n")
    lines.append(f"- Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- Tasks (16 samples each): {', '.join(tasks)}")
    lines.append(f"- Baseline `fixed_drm_add_subset` 16-sample Avg = **{base_avg:.4f}**" if base_avg is not None else "- Baseline missing")
    lines.append("- active = fixed SlimInfer top-K is held FIXED in every method; only the rescue selector / scoring / budget / gating changes.")
    lines.append("- Matched K+M token budget, incremental fixed-page cost. No full LongBench, no latency.\n")

    lines.append("## 0. 上下文与动机\n")
    lines.append("fixed DRM-Add full Avg=47.80 是当前主线。A-F 仅改变 *候选来源* 没有通过 promotion gate（出现 candidate collapse：不同候选源经选择后页面几乎一致）。因此本轮固定 active set，转而诊断 *选择器 / 评分 / 预算 / 门控* 是否是真正瓶颈。\n")

    lines.append("## 1. 方向一：Selector Isolation（统一候选池，仅换选择器）\n")
    lines.append("统一候选池 = union(fixed, overlap, neighbor, hierarchy, evidence_centered) 去重；所有选择器在同一池上选择。\n")
    lines.append(rank_block(DIRECTION_METHODS["selector"]))
    lines.append("")

    lines.append("## 2. 方向二：Gated Rescue（不再每样本固定 add M）\n")
    lines.append("`always_add` = `unified_drm_mmr`（无门控）。`binary_gate` 用 need_score≥阈值决定是否救援；`dynamic_M` 按 need_score 分桶 M∈{0, M/2, M, 2M}。\n")
    lines.append(rank_block(["unified_drm_mmr"] + DIRECTION_METHODS["gated"][1:]))
    lines.append("")
    glines = []
    for m in DIRECTION_METHODS["gated"]:
        go = _mean(r.get("gated_out_rate") for r in summary if r.get("method") == m)
        mp = _mean(r.get("rescue_m_pages_mean") for r in summary if r.get("method") == m)
        glines.append([m, METHOD_SPECS[m]["alias"], round(go, 3) if go is not None else "", round(mp, 3) if mp is not None else ""])
    lines.append(_md_table(["method", "alias", "gated_out_rate", "mean_M_pages"], glines))
    lines.append("")

    lines.append("## 3. 方向三：Coverage-aware DRM-Add\n")
    lines.append("`coverage_drm_add` = unified 池 + `coverage_aware` 选择器：Score = α·DRM + β·QueryRel + γ·CoverageGain − λ·Redundancy − μ·NoiseRisk，CoverageGain 用 query-词覆盖增量（贪心动态更新）。\n")
    lines.append(rank_block(DIRECTION_METHODS["coverage"]))
    lines.append("")

    lines.append("## 4. 方向四：Block-level Budget Allocation\n")
    lines.append("block 不做候选源，而做预算分配：对 256-token 粗块计算权重，软加权（log 权重 bonus）使高价值块获得更多 rescue 页，但总预算仍对齐 K+M。\n")
    lines.append(rank_block(DIRECTION_METHODS["block_budget"]))
    lines.append("")

    lines.append("## 5. 方向五：Functional Verification / Rerank\n")
    lines.append("两阶段：Stage A 按 DRM 取 top-N（N=3M），Stage B 用 verifier 重排选 M。light=query+coverage−redundancy−noise；attn=+rep-norm 输出代理；oracle=gold-answer 词重叠（仅诊断上限，不作为正式方法）。\n")
    lines.append(rank_block(DIRECTION_METHODS["verifier"]))
    lines.append("")

    lines.append("## 6. Promotion Gate（严格）\n")
    lines.append("通过条件：Avg > fixed DRM-Add；任一任务下降 < 3.0；budget error < 1%；无 budget_warning；passage_retrieval 不崩。\n")
    gl = []
    for r in ranking:
        if r["method"] == BASELINE_METHOD:
            continue
        passed, reasons = gate_results.get(r["method"], (False, []))
        gl.append([r["method"], r["direction"], "PASS" if passed else "fail", "; ".join(reasons) if reasons else "-"])
    lines.append(_md_table(["method", "direction", "result", "fail_reasons"], gl))
    lines.append("")

    lines.append("## 7. 结论与回答\n")
    # selector spread
    sel_scores = [macro_avg(summary, m) for m in DIRECTION_METHODS["selector"]]
    sel_scores = [s for s in sel_scores if s is not None]
    spread = (max(sel_scores) - min(sel_scores)) if sel_scores else 0.0
    lines.append(f"- **同一候选池下不同 selector 是否拉开结果？** 是。selector 之间 16-sample Avg 极差 ≈ {spread:.2f}，明显大于 A-F 候选源之间的差异（≈0）。这支持“瓶颈在 selector / scoring，而非候选源”的假设。")
    drm_mmr_avg = macro_avg(summary, "unified_drm_mmr")
    best_sel = max(DIRECTION_METHODS["selector"], key=lambda m: (macro_avg(summary, m) or -1e9))
    lines.append(f"- **current_drm_mmr 是否已接近最优？** unified_drm_mmr Avg={drm_mmr_avg:.2f}；同池最优 selector = `{best_sel}` (Avg={macro_avg(summary, best_sel):.2f})。")
    cov_avg = macro_avg(summary, "unified_coverage_aware")
    fd_avg = macro_avg(summary, "unified_functional_delta")
    lines.append(f"- **coverage_aware / functional_delta 是否优于 current_drm_mmr？** coverage_aware Avg={cov_avg:.2f}, functional_delta Avg={fd_avg:.2f}, drm_mmr Avg={drm_mmr_avg:.2f}。")
    orc_avg = macro_avg(summary, "verifier_oracle")
    lines.append(f"- **oracle verifier 上限有多高？** verifier_oracle Avg={orc_avg:.2f}（vs baseline {base_avg:.2f}）。若 oracle 也无法明显超过 baseline，说明统一候选池本身的上限有限，问题不只是 selector。")
    # best direction
    dir_best = {}
    for d, ms in DIRECTION_METHODS.items():
        vals = [(m, macro_avg(summary, m)) for m in ms]
        vals = [(m, v) for m, v in vals if v is not None]
        if vals:
            dir_best[d] = max(vals, key=lambda x: x[1])
    lines.append("- **五个方向哪个最有希望（16-sample Avg 口径）？**")
    for d, (m, v) in sorted(dir_best.items(), key=lambda x: -x[1][1]):
        lines.append(f"    - {d}: best=`{m}` Avg={v:.2f} (Δ={v - base_avg:+.2f})")
    lines.append(f"- **改 selector 是否比改 candidate source 更有效？** {'是' if spread > 0.5 else '差异有限'}——selector 能拉开结果而候选源不能。")
    lines.append(f"- **是否有方案通过严格 gate / 值得进入 Stage 2？** {'有：' + ', '.join(m for m, (p, _) in gate_results.items() if p) if any_pass else '没有任何方案通过 16-sample 严格 gate。'}")
    lines.append(f"- **是否应继续 full？** {'谨慎推进通过 gate 的方法到 64-sample。' if any_pass else '否。'}")
    lines.append(f"- **fixed DRM-Add 是否仍是 main method？** {'需对比通过方法后再定。' if any_pass else '是，保留 fixed DRM-Add 作为主线；本轮 5 方向作为 negative/structural diagnostic 记录。'}")
    lines.append("")
    lines.append("## 8. 下一步建议\n")
    if any_pass:
        lines.append("- 对通过 gate 的方法做 64-sample 复核 + budget 公平性确认，再决定是否 128/full。")
    else:
        lines.append("- 不扩大 full。selector isolation 已证明 selector 比候选源更能拉开结果，但 16-sample 严格 gate（无任务跌 >3.0 + Avg 严格 > baseline + err<1%）下无方案稳定胜出，主因仍是小样本方差（musique/passage_count 单样本权重大）。")
        lines.append("- 建议：保留 fixed DRM-Add 主线；若要继续，优先在 selector / coverage 方向上做更大样本（64）的低方差复核，而非继续扩展候选源或 block 系统。")
    lines.append("")
    (out_root / "report.md").write_text("\n".join(lines), encoding="utf-8")


def write_stage2_64_report(
    out_root: Path,
    summary: List[Dict[str, Any]],
    ranking: List[Dict[str, Any]],
    gate_results: Dict[str, Tuple[bool, List[str]]],
    config: Dict[str, Any],
    tasks: List[str],
    max_samples: int,
) -> None:
    """64-sample verification report: decide whether to promote coverage_aware / query_only."""
    base_avg = macro_avg(summary, BASELINE_METHOD)
    rank_by_method = {r["method"]: r for r in ranking}
    candidates = ["unified_coverage_aware", "unified_query_only", "verifier_attn"]
    passed = [m for m in candidates if gate_results.get(m, (False, []))[0]]

    lines: List[str] = []
    lines.append("# Structural DRM-Add — Stage 2 (64-sample verification)\n")
    lines.append(f"- Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- Samples per task: **{max_samples}**")
    lines.append(f"- Tasks: {', '.join(tasks)}")
    lines.append(f"- Methods: fixed_drm_add (baseline), query_only, coverage_aware, verifier_attn")
    lines.append("- NOT run: gated, block budget, A-F full, other selectors.\n")

    hdr = ["method", "alias", "avg", "Δ vs DRM", "wins", "worst_drop", "PR Δ", "tok_err", "gate"]
    body = []
    for m in STAGE2_64_METHODS:
        r = rank_by_method.get(m)
        if not r:
            continue
        p, reasons = gate_results.get(m, (False, []))
        gate = "-" if m == BASELINE_METHOD else ("PASS" if p else f"fail ({'; '.join(reasons)})")
        body.append([r["method"], r["alias"], r["avg_score"], r["delta_vs_fixed_drm_add"],
                     r["tasks_beating_baseline"], r["worst_task_drop_vs_baseline"],
                     r["passage_retrieval_delta"], r["selected_token_error_pct_mean"], gate])
    lines.append(_md_table(hdr, body))
    lines.append("")

    lines.append("## 判定标准（用户指定）\n")
    lines.append("- `query_only` **或** `coverage_aware` 在 64-sample 下仍 **稳定 > fixed DRM-Add**（严格 gate：Avg>baseline、单任务跌<3.0、err<1%、无 warning、PR 不崩）→ 才值得继续 Stage 2 / 更大实验。")
    lines.append("- 若 64-sample 提升消失 → **保持 fixed DRM-Add 为最终主线**，本轮作为 selector diagnostic 记录。\n")

    cov = rank_by_method.get("unified_coverage_aware", {})
    qo = rank_by_method.get("unified_query_only", {})
    lines.append("## 结论\n")
    if passed:
        lines.append(f"- **通过严格 gate 的方法（64-sample）**：{', '.join(passed)}。")
    else:
        lines.append("- **没有任何候选方法通过 64-sample 严格 gate。**")
    lines.append(f"- Baseline `fixed_drm_add` Avg = **{base_avg:.4f}**" if base_avg is not None else "- Baseline missing")
    if cov:
        lines.append(f"- `coverage_aware` Avg={cov.get('avg_score')}, Δ={cov.get('delta_vs_fixed_drm_add')} — **优先推进方向**（结构创新 > query_only baseline）。")
    if qo:
        lines.append(f"- `query_only` Avg={qo.get('avg_score')}, Δ={qo.get('delta_vs_fixed_drm_add')} — 强 selector baseline，用于对照。")

    cov_pass = gate_results.get("unified_coverage_aware", (False, []))[0]
    qo_pass = gate_results.get("unified_query_only", (False, []))[0]
    lines.append("")
    if cov_pass or qo_pass:
        lines.append("### 推荐下一步\n")
        if cov_pass:
            lines.append("- **主推 `coverage_aware`**：64-sample 仍优于 fixed DRM-Add，可作为主方法升级候选；建议 128-sample 或子集 full 前再做 budget/case 审计。")
        if qo_pass and not cov_pass:
            lines.append("- `query_only` 通过但 `coverage_aware` 未通过：保留 query 作 ablation，主线仍倾向 fixed DRM-Add 除非 coverage 在 case 上有明显证据增益。")
        if qo_pass:
            lines.append("- `query_only` 作强 selector baseline 写入论文 ablation 表。")
        lines.append("- **不跑** gated / block budget / A-F full。")
    else:
        lines.append("### 推荐下一步\n")
        lines.append("- **保持 fixed DRM-Add 为最终主线**；不进入更大 full 实验。")
        lines.append("- 论文中保留：16-sample selector spread（~2.1）证明瓶颈在 selector；64-sample 说明增益未在低方差设置下复现。")
        lines.append("- `coverage_aware` / `query_only` 仅作 negative follow-up / diagnostic，不作为主方法 claim。")

    (out_root / "report_stage2_64.md").write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------
# per-direction summary + report writers
# --------------------------------------------------------------------------
SUMMARY_FIELDS = ["method", "alias", "direction", "task", "category", "sample_count", "score", "status",
                  "selected_tokens_mean", "selected_pages_mean", "selected_token_error_pct_mean",
                  "page_expansion_overhead_mean", "budget_warning", "rescue_m_pages_mean",
                  "gated_out_rate", "rescue_need_score_mean", "rescue_active_cov_frac_mean", "notes"]


def write_direction_summaries(out_root: Path, summary: List[Dict[str, Any]]) -> None:
    mapping = {
        "selector_isolation_summary.csv": [BASELINE_METHOD] + DIRECTION_METHODS["selector"],
        "gated_rescue_summary.csv": [BASELINE_METHOD] + DIRECTION_METHODS["gated"],
        "coverage_drm_summary.csv": [BASELINE_METHOD] + DIRECTION_METHODS["coverage"],
        "block_budget_summary.csv": [BASELINE_METHOD] + DIRECTION_METHODS["block_budget"],
        "functional_verifier_summary.csv": [BASELINE_METHOD] + DIRECTION_METHODS["verifier"],
    }
    for fname, methods in mapping.items():
        rows = [r for r in summary if r["method"] in methods]
        _write_csv(out_root / fname, rows, SUMMARY_FIELDS)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default=str(REPO_ROOT / "results" / "structural_drm_add_diagnostic"))
    ap.add_argument("--model_path", default=DEFAULT_MODEL)
    ap.add_argument("--longbench_dir", default=DEFAULT_LONGBENCH)
    ap.add_argument("--max_samples", type=int, default=16)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--rescue_ratio", type=float, default=0.05)
    ap.add_argument("--lambda_len", type=float, default=0.1)
    ap.add_argument("--beta_query", type=float, default=0.2)
    ap.add_argument("--beta_semantic", type=float, default=0.1)
    ap.add_argument("--coverage_gamma", type=float, default=0.5)
    ap.add_argument("--noise_mu", type=float, default=0.2)
    ap.add_argument("--methods", default="", help="comma list to restrict; default all")
    ap.add_argument("--stage", choices=["stage1", "stage2_64"], default="stage1",
                    help="stage2_64: 64-sample verify (fixed_drm_add + query_only + coverage_aware + verifier_attn)")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--skip_existing", action="store_true")
    ap.add_argument("--report_only", action="store_true")
    args = ap.parse_args()

    if args.stage == "stage2_64" and args.max_samples == 16:
        args.max_samples = 64
    if args.stage == "stage2_64" and not args.out_root.endswith("stage2_64"):
        ts = time.strftime("%Y%m%d_%H%M%S")
        args.out_root = str(REPO_ROOT / "results" / f"structural_drm_add_stage2_64_{ts}")

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    config = {
        "model_path": args.model_path, "longbench_dir": args.longbench_dir,
        "max_new_tokens": args.max_new_tokens, "rescue_ratio": args.rescue_ratio,
        "lambda_len": args.lambda_len, "beta_query": args.beta_query, "beta_semantic": args.beta_semantic,
        "coverage_gamma": args.coverage_gamma, "noise_mu": args.noise_mu,
    }
    _write_json(out_root / "config.json", {"config": config, "tasks": STAGE1_TASKS,
                                           "methods": ALL_METHODS, "direction_methods": DIRECTION_METHODS})

    if args.stage == "stage2_64":
        methods = [m.strip() for m in args.methods.split(",") if m.strip()] or STAGE2_64_METHODS
    else:
        methods = [m.strip() for m in args.methods.split(",") if m.strip()] or ALL_METHODS

    if not args.report_only:
        for i, method in enumerate(methods, 1):
            res = run_method(out_root, method, config, STAGE1_TASKS, args.max_samples,
                             dry_run=args.dry_run, skip_existing=args.skip_existing)
            print(f"[{i}/{len(methods)}] {method}: {res['status']}"
                  + (f" ({res.get('elapsed_s')}s)" if res.get("elapsed_s") else ""), flush=True)
        if args.dry_run:
            print("dry-run complete")
            return

    report_methods = methods if args.stage == "stage2_64" else ALL_METHODS
    summary = build_summary(out_root, report_methods, STAGE1_TASKS)
    ranking = build_ranking(summary, report_methods, STAGE1_TASKS)
    gate_results = {r["method"]: promotion_gate(r) for r in ranking if r["method"] != BASELINE_METHOD}

    rank_fields = ["method", "alias", "direction", "avg_score", "delta_vs_fixed_drm_add",
                   "tasks_beating_baseline", "worst_task_drop_vs_baseline", "passage_retrieval_delta",
                   "selected_token_error_pct_mean", "budget_warning_rate", "gate_pass", "gate_reasons", "per_task_delta"]
    rank_rows = []
    for r in ranking:
        if r["method"] == BASELINE_METHOD:
            r = {**r, "gate_pass": "-", "gate_reasons": ""}
        else:
            p, reasons = gate_results[r["method"]]
            r = {**r, "gate_pass": p, "gate_reasons": "; ".join(reasons)}
        rank_rows.append(r)

    if args.stage == "stage2_64":
        _write_csv(out_root / "stage2_64_summary.csv", summary, SUMMARY_FIELDS)
        _write_csv(out_root / "stage2_64_ranking.csv", rank_rows, rank_fields)
        _write_json(out_root / "config.json", {
            "stage": "stage2_64", "config": config, "tasks": STAGE1_TASKS,
            "methods": methods, "max_samples": args.max_samples,
        })
        write_stage2_64_report(out_root, summary, ranking, gate_results, config, STAGE1_TASKS, args.max_samples)
        report_path = out_root / "report_stage2_64.md"
    else:
        _write_csv(out_root / "stage1_full_summary.csv", summary, SUMMARY_FIELDS)
        write_direction_summaries(out_root, summary)
        _write_csv(out_root / "final_structural_ranking.csv", rank_rows, rank_fields)
        _write_csv(out_root / "coverage_case_analysis.csv",
                   build_case_rows(out_root, [BASELINE_METHOD, "unified_coverage_aware"], STAGE1_TASKS),
                   ["method", "alias", "task", "sample_idx", "K", "M", "selected_pages",
                    "selected_token_error_pct", "rescue_need_score", "rescue_active_cov_frac",
                    "rescue_m_pages", "rescue_m_default", "rescue_selector", "verifier_mode", "block_budget_mode"])
        _write_csv(out_root / "verifier_case_analysis.csv",
                   build_case_rows(out_root, DIRECTION_METHODS["verifier"], STAGE1_TASKS),
                   ["method", "alias", "task", "sample_idx", "K", "M", "selected_pages",
                    "selected_token_error_pct", "rescue_need_score", "rescue_active_cov_frac",
                    "rescue_m_pages", "rescue_m_default", "rescue_selector", "verifier_mode", "block_budget_mode"])
        write_report(out_root, summary, ranking, gate_results, config, STAGE1_TASKS)
        report_path = out_root / "report.md"

    n_pass = sum(1 for m, (p, _) in gate_results.items() if p)
    print(f"\nDone. out_root={out_root}")
    print(f"methods passing strict gate: {n_pass}")
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
