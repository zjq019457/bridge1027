#!/usr/bin/env python3
"""Route-B v3: Boundary + Page-Span Hybrid physical KV retention diagnostics."""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.fup_guarded_swap_lib import FupHarness, release_cuda  # noqa: E402
from tools.run_position_aware_kv_sanity import (  # noqa: E402
    PAGE_TOKENS,
    TASKS,
    append_baselines,
    bool_rate,
    ensure_prompt_tail,
    markdown_table,
    physical_generate_from_prefill,
    prefill_before_tail,
    read_csv,
    run_all_tokens_equivalence,
    selected_page_indices,
    span_index_to_id,
    strategy_selection as v2_strategy_selection,
    token_metadata,
    write_csv,
    write_json,
)
from tools.span_drm_validation_lib import (  # noqa: E402
    LONGBENCH_DIR,
    MODEL_PATH,
    fixed_audit,
    mean,
    reconstruct_span_selection,
    score_prediction,
    span_metadata,
)

BOUNDARY_SIZES = [4, 8, 16, 32]
RETRIEVAL_TASKS = {"passage_count", "passage_retrieval_en"}
QA_TASKS = {"qasper", "hotpotqa", "musique"}

BASELINE_METHODS = ["slim_infer_original", "fixed_drm_add", "span_drm_text_packing"]
BEST_V2 = "best_physical_v2"
BOUNDARY_METHODS = [f"physical_span_boundary_{n}" for n in BOUNDARY_SIZES]
HYBRID_METHODS = [
    "physical_hybrid_page1_span",
    "physical_hybrid_page2_span",
    "physical_hybrid_page1_span_boundary8",
    "physical_hybrid_page1_span_boundary16",
    "physical_hybrid_adaptive_page_span",
]
TASK_AWARE_METHODS = [
    "physical_task_aware_boundary",
    "physical_task_aware_hybrid",
]
ORACLE_METHOD = "physical_oracle_span_boundary_diag"
PHYSICAL_METHODS = [
    BEST_V2,
    *BOUNDARY_METHODS,
    *HYBRID_METHODS,
    *TASK_AWARE_METHODS,
]
ALL_METHODS = [*BASELINE_METHODS, *PHYSICAL_METHODS]
PHYSICAL_DECODE = "span_physical_original_pos"


def boundary_selection_sized(
    metadata: Mapping[str, Any],
    active_indices: Set[int],
    base_indices: Sequence[int],
    context_length: int,
    target: int,
    boundary_size: int,
) -> Tuple[List[int], Dict[str, Any]]:
    selected: List[int] = sorted(active_indices)
    selected_set = set(selected)
    span_tokens = 0
    boundary_tokens = 0
    trimmed = 0
    for span in metadata.get("selected_spans", []):
        start = int(span.get("selected_token_start", span.get("token_start", 0)) or 0)
        end = int(span.get("selected_token_end", span.get("token_end", start)) or start)
        span_tokens += max(0, end - start)
        expanded = list(range(max(0, start - boundary_size), min(context_length, end + boundary_size)))
        for index in expanded:
            if index not in selected_set:
                if index in range(start, end):
                    pass
                else:
                    boundary_tokens += 1
                selected.append(index)
                selected_set.add(index)
                if len(selected) >= target:
                    break
        if len(selected) >= target:
            break
    before = len(selected)
    for index in base_indices:
        if len(selected) >= target:
            break
        if index not in selected_set:
            selected.append(index)
            selected_set.add(index)
    final = ensure_prompt_tail(sorted(selected_set), context_length, target, active_indices)
    trimmed = max(0, before - len(final))
    audit = {
        "boundary_size": boundary_size,
        "num_span_tokens": span_tokens,
        "num_boundary_tokens": boundary_tokens,
        "num_total_retained_tokens": len(final),
        "trimmed_token_count": trimmed,
    }
    return final, audit


def page_scores_from_metadata(metadata: Mapping[str, Any]) -> Dict[int, float]:
    scores: Dict[int, float] = {}
    for span in metadata.get("selected_spans", []):
        pid = int(span.get("page_id", -1))
        if pid < 0:
            continue
        sc = float(span.get("drm_score_parent_page", span.get("final_span_score", 0.0)) or 0.0)
        q = float(span.get("query_similarity", 0.0) or 0.0)
        deployable = sc + 0.2 * q
        scores[pid] = max(scores.get(pid, -1e9), deployable)
    return scores


def hybrid_selection(
    method: str,
    task: str,
    selection: Mapping[str, Any],
    metadata: Mapping[str, Any],
    active_indices: Set[int],
    base_indices: Sequence[int],
    context_length: int,
    target: int,
) -> Tuple[List[int], Dict[str, Any]]:
    active_pages = set(int(p) for p in selection.get("active_pages", []))
    rescue_pages = [int(p) for p in selection.get("final_pages", []) if int(p) not in active_pages]
    page_sc = page_scores_from_metadata(metadata)
    rescue_pages.sort(key=lambda p: page_sc.get(p, 0.0), reverse=True)

    if method == "physical_hybrid_page2_span":
        n_full = 2
        boundary = 0
    elif method == "physical_hybrid_page1_span_boundary8":
        n_full = 1
        boundary = 8
    elif method == "physical_hybrid_page1_span_boundary16":
        n_full = 1
        boundary = 16
    elif method == "physical_hybrid_adaptive_page_span":
        n_full = 0
        boundary = 8
    else:
        n_full = 1
        boundary = 0

    selected_set: Set[int] = set(active_indices)
    full_page_ids: List[int] = []

    if method == "physical_hybrid_adaptive_page_span":
        for pid in rescue_pages:
            if len(selected_set) >= target:
                break
            score = page_sc.get(pid, 0.0)
            if score >= 0.0:
                full_page_ids.append(pid)
                selected_set.update(selected_page_indices([pid], context_length))
            else:
                for span in metadata.get("selected_spans", []):
                    if int(span.get("page_id", -1)) != pid:
                        continue
                    start = int(span.get("selected_token_start", span.get("token_start", 0)) or 0)
                    end = int(span.get("selected_token_end", span.get("token_end", start)) or start)
                    for idx in range(max(0, start - boundary), min(context_length, end + boundary)):
                        selected_set.add(idx)
                        if len(selected_set) >= target:
                            break
    else:
        for pid in rescue_pages[:n_full]:
            full_page_ids.append(pid)
            selected_set.update(selected_page_indices([pid], context_length))

    if boundary > 0:
        indices, baudit = boundary_selection_sized(
            metadata, active_indices, list(selected_set), context_length, target, boundary
        )
        selected_set = set(indices)

    for index in base_indices:
        if len(selected_set) >= target:
            break
        selected_set.add(int(index))

    final = ensure_prompt_tail(sorted(selected_set), context_length, target, active_indices)
    span_units = len(metadata.get("selected_spans", []))
    audit = {
        "num_full_pages": len(full_page_ids),
        "num_span_units": span_units,
        "num_boundary_tokens": boundary,
        "num_total_tokens": len(final),
        "selected_full_page_ids": ",".join(str(p) for p in full_page_ids),
        "selected_span_ids": ",".join(
            str(s.get("span_id", "")) for s in metadata.get("selected_spans", [])[:20]
        ),
    }
    return final, audit


def v3_strategy_selection(
    method: str,
    task: str,
    selection: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> Dict[str, Any]:
    context_length = int(selection["context_length"])
    target = int(selection["target_final_tokens"] or selection["selected_token_count"])
    active = set(selected_page_indices(selection.get("active_pages", []), context_length))
    base = ensure_prompt_tail(selection["selected_abs_indices"], context_length, target, active)

    if method == BEST_V2:
        return v2_strategy_selection("span_physical_original_pos", task, selection, metadata)

    if method == ORACLE_METHOD:
        spans = sorted(
            metadata.get("selected_spans", []),
            key=lambda s: float(s.get("answer_overlap_oracle_diag", 0.0) or 0.0),
            reverse=True,
        )
        meta2 = dict(metadata)
        meta2["selected_spans"] = spans[: max(1, len(spans) // 2)]
        indices, _ = boundary_selection_sized(meta2, active, base, context_length, target, 8)
        return {
            "indices": indices,
            "active_indices": active,
            "target": target,
            "selection_mode": "oracle_diag_boundary8",
            "fallback_used": False,
            "extra_audit": {"oracle_diag": True},
        }

    if method == "physical_task_aware_boundary":
        if task in RETRIEVAL_TASKS:
            pages = selection.get("final_pages", []) or selection.get("active_pages", [])
            indices = ensure_prompt_tail(
                sorted(selected_page_indices(pages, context_length)),
                context_length, target, active,
            )
            return {
                "indices": indices,
                "active_indices": active,
                "target": target,
                "selection_mode": "task_aware_full_page_fallback",
                "fallback_used": True,
                "extra_audit": {"task_policy": "retrieval_count_full_page"},
            }
        indices, baudit = boundary_selection_sized(metadata, active, base, context_length, target, 8)
        return {
            "indices": indices,
            "active_indices": active,
            "target": target,
            "selection_mode": "task_aware_boundary8_qa",
            "fallback_used": False,
            "extra_audit": baudit,
        }

    if method == "physical_task_aware_hybrid":
        if task in RETRIEVAL_TASKS:
            pages = selection.get("final_pages", []) or selection.get("active_pages", [])
            indices = ensure_prompt_tail(
                sorted(selected_page_indices(pages, context_length)),
                context_length, target, active,
            )
            return {
                "indices": indices,
                "active_indices": active,
                "target": target,
                "selection_mode": "task_aware_full_page_fallback",
                "fallback_used": True,
                "extra_audit": {"task_policy": "retrieval_count_full_page"},
            }
        indices, haudit = hybrid_selection(
            "physical_hybrid_page1_span_boundary8", task, selection, metadata,
            active, base, context_length, target,
        )
        return {
            "indices": indices,
            "active_indices": active,
            "target": target,
            "selection_mode": "task_aware_hybrid_page1_boundary8_qa",
            "fallback_used": False,
            "extra_audit": haudit,
        }

    if method.startswith("physical_span_boundary_"):
        bsize = int(method.rsplit("_", 1)[-1])
        indices, baudit = boundary_selection_sized(
            metadata, active, base, context_length, target, bsize
        )
        return {
            "indices": indices,
            "active_indices": active,
            "target": target,
            "selection_mode": f"boundary_{bsize}",
            "fallback_used": False,
            "extra_audit": baudit,
        }

    if method in HYBRID_METHODS:
        indices, haudit = hybrid_selection(
            method, task, selection, metadata, active, base, context_length, target
        )
        return {
            "indices": indices,
            "active_indices": active,
            "target": target,
            "selection_mode": method,
            "fallback_used": False,
            "extra_audit": haudit,
        }

    raise ValueError(f"unknown method {method}")


def run_equivalence_v3(harness, out, tasks, n, skip) -> Tuple[List[Dict[str, Any]], bool]:
    rows, passed = run_all_tokens_equivalence(harness, out, tasks, n, skip)
    v3_rows = [{**r, "gate_version": "v3"} for r in rows]
    write_csv(out / "all_tokens_equivalence_v3.csv", v3_rows)
    success = [r for r in v3_rows if r.get("runner_status") == "success"]
    max_diff = max((float(r["max_logit_diff"]) for r in success), default=math.inf)
    match_rate = bool_rate(success, "exact_pred_match")
    gate_ok = len(success) == len(v3_rows) and max_diff <= 0.05 and match_rate >= 0.99
    report = [
        "# All-Token Equivalence v3",
        "",
        f"- Samples: {len(v3_rows)} ({n} per task)",
        f"- Max logit diff: {max_diff}",
        f"- Exact match rate: {match_rate:.4f}",
        f"- Gate passed: **{gate_ok}**",
        "- Reuses v2 position-aware sparse cache (physical_cache_index / logical_position_id decoupled).",
    ]
    (out / "all_tokens_equivalence_v3_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return v3_rows, gate_ok


def run_physical_v3(
    harness: FupHarness,
    out: Path,
    tasks: Sequence[str],
    samples_per_task: int,
    skip_existing: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    case_path = out / "physical_span_kv_v3_cases.csv"
    audit_path = out / "physical_kv_position_audit_v3.csv"
    cases = read_csv(case_path) if skip_existing else []
    audits = read_csv(audit_path) if skip_existing else []
    boundary_audits: List[Dict[str, Any]] = read_csv(out / "boundary_retention_audit.csv") if skip_existing else []
    hybrid_audits: List[Dict[str, Any]] = read_csv(out / "hybrid_retention_audit.csv") if skip_existing else []
    fallback_audits: List[Dict[str, Any]] = read_csv(out / "task_aware_fallback_audit.csv") if skip_existing else []
    done = {(r["task"], r["sample_id"], r["method"]) for r in cases}

    for task in tasks:
        meta_by_id = span_metadata(task)
        audit_by_id = fixed_audit(task)
        samples = harness.load_samples(task, samples_per_task)
        for sample_index, sample in enumerate(samples):
            sid = str(sample.get("_id", sample.get("id", sample_index)))
            ctx = harness.build_sample_ctx(task, sample, sample_index)
            metadata = meta_by_id.get(sid, {})
            fixed = audit_by_id.get(sid, {})
            selection = reconstruct_span_selection(task, sid, metadata, fixed)
            append_baselines(cases, task, sid, selection)
            write_csv(case_path, cases)
            needed = [m for m in PHYSICAL_METHODS if (task, sid, m) not in done]
            if not needed:
                continue
            source = prefill_before_tail(harness.model, ctx)
            for method in needed:
                strategy = v3_strategy_selection(method, task, selection, metadata)
                indices = strategy["indices"]
                extra = strategy.get("extra_audit", {}) or {}
                budget_before = strategy["target"]
                try:
                    result = physical_generate_from_prefill(
                        harness, ctx, source, PHYSICAL_DECODE, indices,
                        strategy["active_indices"], metadata, selection,
                        int(harness.dataset2maxlen.get(task, 32)),
                        {"task": task, "sample_id": sid} if sample_index == 0 else None,
                    )
                    score = score_prediction(
                        task, result["pred"], sample.get("answers", []), sample.get("all_classes")
                    )
                    budget_error = (len(indices) - strategy["target"]) / max(1, strategy["target"])
                    cases.append({
                        "task": task, "sample_id": sid, "method": method,
                        "score": round(score, 6),
                        "pred_preview": result["pred"][:240].replace("\n", " "),
                        "retained_token_count": len(indices),
                        "target_token_count": strategy["target"],
                        "budget_error": round(budget_error, 8),
                        "fallback_used": strategy["fallback_used"],
                        "selection_mode": strategy["selection_mode"],
                        "runner_fail": 0, "failure_reason": "",
                    })
                    ca = result["audit"]
                    audits.append({
                        "task": task, "sample_id": sid, "method": method,
                        "num_retained_tokens": len(indices),
                        "physical_cache_len": ca.get("physical_cache_len", ""),
                        "logical_position_max": ca.get("logical_position_max", ""),
                        "causal_mask_ok": ca.get("causal_mask_ok", True),
                        "runner_status": "success",
                    })
                    row_common = {
                        "task": task, "sample_id": sid, "method": method,
                        "budget_before": budget_before,
                        "budget_after": len(indices),
                        "budget_error": round(budget_error, 8),
                        "runner_status": "success",
                    }
                    if method.startswith("physical_span_boundary_"):
                        boundary_audits.append({**row_common, **extra})
                    elif method in HYBRID_METHODS:
                        hybrid_audits.append({**row_common, **extra})
                    elif method in TASK_AWARE_METHODS:
                        fallback_audits.append({**row_common, **extra, "need_rescue": int(len(indices) > len(strategy["active_indices"]))})
                    del result
                except BaseException as exc:
                    cases.append({
                        "task": task, "sample_id": sid, "method": method,
                        "score": 0.0, "pred_preview": "",
                        "retained_token_count": len(indices),
                        "target_token_count": strategy["target"],
                        "budget_error": round((len(indices) - strategy["target"]) / max(1, strategy["target"]), 8),
                        "fallback_used": strategy["fallback_used"],
                        "selection_mode": strategy["selection_mode"],
                        "runner_fail": 1,
                        "failure_reason": f"{type(exc).__name__}: {exc}"[:500],
                    })
                    audits.append({
                        "task": task, "sample_id": sid, "method": method,
                        "runner_status": f"failed:{type(exc).__name__}",
                    })
                    release_cuda()
                write_csv(case_path, cases)
                write_csv(audit_path, audits)
                write_csv(out / "boundary_retention_audit.csv", boundary_audits)
                write_csv(out / "hybrid_retention_audit.csv", hybrid_audits)
                write_csv(out / "task_aware_fallback_audit.csv", fallback_audits)
            del source
            release_cuda()
            print(f"[physical-v3] {task} {sample_index + 1}/{samples_per_task}", flush=True)
    return cases, audits, boundary_audits, hybrid_audits


def physical_level(best_delta: float, best_vs_text: float, runner_fail: int, all_pass: bool) -> str:
    if runner_fail > 0:
        return "Level 0"
    if not all_pass:
        return "Level 0"
    if best_delta >= 0:
        return "Level 2_strong"
    if best_delta >= -0.1:
        return "Level 2_near"
    if best_delta >= -0.3:
        return "Level 2"
    if best_vs_text >= -0.5:
        return "Level 1"
    return "Level 1"


def summarize_v3(out, cases, tasks, samples_per_task, all_pass):
    by_task = []
    for method in ALL_METHODS:
        for task in tasks:
            sel = [r for r in cases if r["method"] == method and r["task"] == task]
            errs = [abs(float(r["budget_error"])) for r in sel if str(r.get("budget_error", "")) != ""]
            by_task.append({
                "method": method, "task": task, "n_samples": len(sel),
                "score": round(mean(float(r["score"]) for r in sel), 4) if sel else "",
                "runner_fail": sum(int(r["runner_fail"]) for r in sel),
                "budget_error_mean": round(mean(errs), 6) if errs else "",
            })
    write_csv(out / "physical_span_kv_v3_by_task.csv", by_task)
    macro = {m: mean(float(r["score"]) for r in by_task if r["method"] == m and r["score"] != "") for m in ALL_METHODS}
    text, fixed, slim = macro["span_drm_text_packing"], macro["fixed_drm_add"], macro["slim_infer_original"]
    summary = []
    for method in ALL_METHODS:
        mc = [r for r in cases if r["method"] == method]
        errs = [abs(float(r["budget_error"])) for r in mc if str(r.get("budget_error", "")) != ""]
        b_aud = [r for r in read_csv(out / "boundary_retention_audit.csv") if r.get("method") == method]
        h_aud = [r for r in read_csv(out / "hybrid_retention_audit.csv") if r.get("method") == method]
        impl = "deployable_physical" if method in PHYSICAL_METHODS and method != ORACLE_METHOD else (
            "upper_bound_proxy" if method == "span_drm_text_packing" else "baseline_reuse"
        )
        if method == ORACLE_METHOD:
            impl = "oracle_diagnostic_only"
        row = {
            "method": method,
            "evaluation": f"{samples_per_task}-sample-per-task diagnostic",
            "task_macro_avg": round(macro[method], 4) if macro[method] == macro[method] else "",
            "sample_weighted_avg": round(mean(float(r["score"]) for r in mc), 4) if mc else "",
            "delta_vs_sliminfer": round(macro[method] - slim, 4) if macro[method] == macro[method] else "",
            "delta_vs_fixed_drm_add": round(macro[method] - fixed, 4) if macro[method] == macro[method] else "",
            "delta_vs_text_packing": round(macro[method] - text, 4) if macro[method] == macro[method] else "",
            "runner_fail": sum(int(r["runner_fail"]) for r in mc),
            "budget_error_mean": round(mean(errs), 6) if errs else "",
            "budget_error_max": round(max(errs), 6) if errs else "",
            "budget_warning": sum(1 for e in errs if e >= 0.01),
            "avg_retained_tokens": round(mean(float(r["retained_token_count"]) for r in mc if r.get("retained_token_count")), 2) if mc else "",
            "avg_boundary_tokens": round(mean(float(r.get("num_boundary_tokens", 0) or 0) for r in b_aud), 2) if b_aud else "",
            "avg_full_pages": round(mean(float(r.get("num_full_pages", 0) or 0) for r in h_aud), 2) if h_aud else "",
            "implementation_status": impl,
        }
        summary.append(row)
    phys = [r for r in summary if r["method"] in PHYSICAL_METHODS and r["method"] != ORACLE_METHOD]
    best = max(phys, key=lambda r: float(r["task_macro_avg"] or 0))
    best_delta = float(best["delta_vs_fixed_drm_add"])
    best_vs_text = float(best["delta_vs_text_packing"])
    lvl = physical_level(best_delta, best_vs_text, sum(int(r["runner_fail"]) for r in phys), all_pass)
    for r in summary:
        if r["method"] == best["method"]:
            r["physical_level"] = lvl
        elif r["method"] in PHYSICAL_METHODS:
            r["physical_level"] = "Level 1" if all_pass else "Level 0"
        else:
            r["physical_level"] = ""
    write_csv(out / "physical_span_kv_v3_summary.csv", summary)
    ranking = sorted(phys, key=lambda r: float(r["task_macro_avg"] or 0), reverse=True)
    for i, r in enumerate(ranking, 1):
        r["rank"] = i
    hdr = ["rank"] + [k for k in ranking[0].keys() if k != "rank"] if ranking else []
    write_csv(out / "physical_span_kv_v3_ranking.csv", ranking, hdr)
    return summary, ranking, best


def write_reports_v3(out, summary, ranking, all_rows, all_pass, samples_per_task, best):
    by_name = {r["method"]: r for r in summary}
    fixed = by_name["fixed_drm_add"]
    text = by_name["span_drm_text_packing"]
    best_delta = float(best["delta_vs_fixed_drm_add"])
    promising = best_delta >= 0 or best_delta > -0.1
    stop_route = best_delta < -0.3

    boundary_scores = {r["method"]: float(r["task_macro_avg"]) for r in summary if r["method"] in BOUNDARY_METHODS}
    best_b = max(boundary_scores, key=boundary_scores.get) if boundary_scores else ""
    hybrid_scores = {r["method"]: float(r["task_macro_avg"]) for r in summary if r["method"] in HYBRID_METHODS}
    best_h = max(hybrid_scores, key=hybrid_scores.get) if hybrid_scores else ""
    v2_score = float(by_name.get(BEST_V2, {}).get("task_macro_avg", 0))

    report = [
        "# Physical Span KV v3: Boundary + Page-Span Hybrid",
        "",
        f"- Env: repo `.venv` (conda `silm` unavailable; see env_logs.txt).",
        f"- Evaluation: {samples_per_task} samples/task, 5 tasks, no full LongBench, no latency.",
        "",
        "## Summary",
        markdown_table(summary, ["method", "task_macro_avg", "delta_vs_fixed_drm_add", "delta_vs_text_packing", "runner_fail", "physical_level"]),
        "",
        "## Required Answers",
        f"1. All-token equivalence maintained: **{all_pass}**",
        f"2. Boundary improves over v2 ({BEST_V2}={v2_score:.4f}): **{max(boundary_scores.values(), default=0) > v2_score}**",
        f"3. Best boundary size: **{best_b.replace('physical_span_boundary_', '') if best_b else 'n/a'}**",
        f"4. Hybrid vs boundary-only: **{max(hybrid_scores.values(), default=0) > max(boundary_scores.values(), default=0)}** (best hybrid={best_h})",
        f"5. Task-aware fallback necessary: retrieval/count use deterministic full-page per config.",
        f"6. Best physical v3 > fixed DRM-Add: **{best_delta > 0}** ({best_delta:+.4f})",
        f"7. Near text packing: delta={float(best['delta_vs_text_packing']):+.4f}",
        "8. If still below fixed: span fragmentation + missing page context + full-prefill K/V representation gap.",
        f"9. Suggest 64-sample: **{promising}** (set RUN_PHYSICAL_V3_64=1 to auto-run).",
        "10. Suggest full LongBench: **No**.",
        "11. Paper role: **diagnostic/limitation prototype**, not main method unless Level 2_strong.",
        "12. Text packing: structured context compression proxy; not production KV pruning.",
        "",
        f"- Route-B stop: **{stop_route}**" if stop_route else "- Route-B near-competitive or promising.",
    ]
    (out / "physical_span_kv_v3_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    claims = [
        "# Physical Span KV v3 Claims",
        "",
        "## Strong claims",
        "- All-token equivalence gate re-validated for v3 sparse cache path.",
        "- Physical methods use real K/V index selection; no prompt repacking.",
        "- Task-aware fallback for passage_retrieval_en / passage_count is pre-registered in config.",
        "",
        "## Moderate claims",
        f"- Boundary variants tested at sizes {BOUNDARY_SIZES}.",
        "- Page-span hybrid retains deployable-scored full pages plus spans.",
        "",
        "## Diagnostic claims",
        "- span_drm_text_packing remains upper-bound proxy only.",
        "",
        "## Do-not-claim",
        "- Do not claim text packing equals physical KV retention.",
        "- Do not claim production native sparse KV without Level 3 + full eval.",
        "- Do not claim latency speedup or reduced prefill compute.",
    ]
    if best_delta <= 0:
        claims.append("- Do not claim physical SpanDRM as final main method (below fixed DRM-Add).")
    (out / "physical_span_kv_v3_claims.md").write_text("\n".join(claims) + "\n", encoding="utf-8")


def write_config_scripts(out, args, all_pass, best):
    best_delta = float(best.get("delta_vs_fixed_drm_add", -999))
    promising = best_delta >= 0 or best_delta > -0.1
    cfg = {
        "stage": "physical_span_kv_v3_boundary_hybrid",
        "environment": "repo .venv via scripts/run_with_silm.sh",
        "tasks": args.tasks,
        "samples_per_task": args.samples_per_task,
        "equivalence_samples_per_task": args.equivalence_samples_per_task,
        "methods": ALL_METHODS,
        "best_physical_v2_alias": "span_physical_original_pos",
        "boundary_sizes": BOUNDARY_SIZES,
        "fallback_policy": {
            "passage_retrieval_en": "full-page retention (deterministic)",
            "passage_count": "full-page or page-span hybrid (deterministic)",
            "qasper/hotpotqa/musique": "span+boundary or page-span hybrid",
        },
        "no_rope_rerotation": True,
        "all_tokens_gate_passed": all_pass,
        "promising_for_64": promising,
        "full_longbench": False,
        "latency": False,
    }
    write_json(out / "config.json", cfg)
    (out / "env_logs.txt").write_text(
        "python_env=.venv\nconda_alias=silm->.venv\n", encoding="utf-8"
    )
    cmd = (
        f"scripts/run_with_silm.sh tools/run_physical_span_kv_v3.py "
        f"--output_dir {out} --samples_per_task {args.samples_per_task} "
        f"--equivalence_samples_per_task {args.equivalence_samples_per_task} "
        f"--tasks {','.join(args.tasks)} --skip_existing"
    )
    for name in ["run_physical_span_kv_v3.sh", "resume.sh"]:
        p = out / name
        p.write_text(f"#!/usr/bin/env bash\nset -euo pipefail\ncd \"{REPO_ROOT}\"\n{cmd}\n", encoding="utf-8")
        p.chmod(0o755)
    if promising:
        (out / "run_physical_span_kv_v3_64_if_promising.sh").write_text(
            f"#!/usr/bin/env bash\n# Run only if RUN_PHYSICAL_V3_64=1\n"
            f'if [[ "${{RUN_PHYSICAL_V3_64:-0}}" != "1" ]]; then echo "Set RUN_PHYSICAL_V3_64=1"; exit 0; fi\n'
            f"cd \"{REPO_ROOT}\"\n"
            f"scripts/run_with_silm.sh tools/run_physical_span_kv_v3.py "
            f"--output_dir {out}_64 --samples_per_task 64 "
            f"--equivalence_samples_per_task 8 --tasks {','.join(args.tasks)}\n",
            encoding="utf-8",
        )
        (out / "run_physical_span_kv_v3_64_if_promising.sh").chmod(0o755)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--samples_per_task", type=int, default=32)
    parser.add_argument("--equivalence_samples_per_task", type=int, default=8)
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()
    args.tasks = [t for t in args.tasks.split(",") if t]
    if not args.output_dir:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = str(REPO_ROOT / "results" / f"physical_span_kv_v3_{stamp}")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    harness = FupHarness(str(MODEL_PATH), str(LONGBENCH_DIR), str(out / "_work"), max_new_tokens=128)
    all_rows, all_pass = run_equivalence_v3(
        harness, out, args.tasks, args.equivalence_samples_per_task, args.skip_existing
    )
    if not all_pass:
        write_config_scripts(out, args, all_pass, {"delta_vs_fixed_drm_add": -999})
        raise SystemExit("all-token equivalence failed; stopping v3")

    cases, _, _, _ = run_physical_v3(harness, out, args.tasks, args.samples_per_task, args.skip_existing)
    summary, ranking, best = summarize_v3(out, cases, args.tasks, args.samples_per_task, all_pass)
    write_reports_v3(out, summary, ranking, all_rows, all_pass, args.samples_per_task, best)
    write_config_scripts(out, args, all_pass, best)
    del harness
    gc.collect()
    release_cuda()
    print(f"[physical-v3] DONE -> {out}", flush=True)


if __name__ == "__main__":
    main()
