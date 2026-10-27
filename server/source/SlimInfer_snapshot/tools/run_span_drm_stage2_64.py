#!/usr/bin/env python3
"""SpanDRM-Add 64-sample diagnostic using packed active pages plus evidence spans."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.build_span_candidates import PAGE_TOKENS, STOPWORDS, TOKEN_RE, build as build_candidates
from tools.fup_guarded_swap_lib import FupHarness, SampleCtx, release_cuda

DEFAULT_MODEL = "/home/yu-bai/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
DEFAULT_LB = "/home/yu-bai/Desktop/jiaqizhang/HeadKV/data/LongBench"
DEFAULT_FUP = REPO_ROOT / "results" / "fup_guarded_swap_stage2_64_20260601_051737"
TASKS = ["musique", "hotpotqa", "qasper", "passage_count", "passage_retrieval_en"]
QA_TASKS = {"musique", "hotpotqa", "qasper"}

METHOD_SPECS: Dict[str, Dict[str, Any]] = {
    "span_drm_add_query": {"scorer": "query", "k": 2, "packing": "budget_refill_spans", "count_fallback": True, "retrieval_fallback": True},
    "span_drm_add_drm": {"scorer": "drm", "k": 2, "packing": "budget_refill_spans", "count_fallback": True, "retrieval_fallback": True},
    "span_drm_add_fup": {"scorer": "fup", "k": 2, "packing": "budget_refill_spans", "count_fallback": True, "retrieval_fallback": True},
    "span_drm_add_hybrid": {"scorer": "hybrid", "k": 2, "packing": "budget_refill_spans", "count_fallback": True, "retrieval_fallback": True},
    "span_drm_add_oracle_diag": {"scorer": "oracle", "k": 2, "packing": "budget_refill_spans", "count_fallback": True, "retrieval_fallback": True},
    "span_drm_add_hybrid_k1": {"scorer": "hybrid", "k": 1, "packing": "budget_refill_spans", "count_fallback": True, "retrieval_fallback": True, "inherit": "span_drm_add_hybrid", "target_tasks": QA_TASKS},
    "span_drm_add_hybrid_k3": {"scorer": "hybrid", "k": 3, "packing": "budget_refill_spans", "count_fallback": True, "retrieval_fallback": True, "inherit": "span_drm_add_hybrid", "target_tasks": QA_TASKS},
    "span_drm_add_hybrid_same_num_pages": {"scorer": "hybrid", "k": 2, "packing": "same_num_pages_compressed", "count_fallback": True, "retrieval_fallback": True, "inherit": "span_drm_add_hybrid", "target_tasks": QA_TASKS},
    "span_drm_add_hybrid_count_compress": {"scorer": "hybrid", "k": 2, "packing": "budget_refill_spans", "count_fallback": False, "retrieval_fallback": True, "inherit": "span_drm_add_hybrid", "target_tasks": {"passage_count"}},
    "span_drm_add_hybrid_retrieval_compress": {"scorer": "hybrid", "k": 2, "packing": "budget_refill_spans", "count_fallback": True, "retrieval_fallback": False, "inherit": "span_drm_add_hybrid", "target_tasks": {"passage_retrieval_en"}},
}

ATTR = {
    "fixed_drm_add": "deployable baseline",
    "span_drm_add_query": "deployable query span scorer",
    "span_drm_add_drm": "deployable DRM span scorer",
    "span_drm_add_fup": "deployable FUP span scorer",
    "span_drm_add_hybrid": "deployable hybrid span scorer",
    "span_drm_add_oracle_diag": "diagnostic_only",
    "span_drm_add_hybrid_k1": "k ablation",
    "span_drm_add_hybrid_k3": "k ablation",
    "span_drm_add_hybrid_same_num_pages": "packing ablation",
    "span_drm_add_hybrid_count_compress": "passage_count compression ablation",
    "span_drm_add_hybrid_retrieval_compress": "passage_retrieval compression ablation",
}
FEATURE_EXPORT_METHOD = "span_drm_add_hybrid"


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def read_jsonl_index(path: Path) -> Dict[str, Dict[str, Any]]:
    return {str(row.get("sample_id", "")): row for row in read_jsonl(path)}


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: List[Dict[str, Any]], header: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    header = header or list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in header})


def mean(values: Iterable[float]) -> float:
    vals = [float(value) for value in values]
    return sum(vals) / len(vals) if vals else 0.0


def page_indices(pages: Sequence[int], context_length: int) -> Set[int]:
    out: Set[int] = set()
    for page in pages:
        start = int(page) * PAGE_TOKENS
        out.update(range(max(0, start), min(context_length, start + PAGE_TOKENS)))
    return out


def text_words(text: str) -> Set[str]:
    return {word.lower() for word in TOKEN_RE.findall(text or "") if len(word) > 1 and word.lower() not in STOPWORDS}


def zscore_map(candidates: List[Dict[str, Any]], key: str) -> Dict[str, float]:
    if not candidates:
        return {}
    vals = [float(row.get(key, 0.0) or 0.0) for row in candidates]
    avg = mean(vals)
    std = math.sqrt(mean((value - avg) ** 2 for value in vals))
    if std <= 1e-9:
        return {str(row["span_id"]): 0.0 for row in candidates}
    return {str(row["span_id"]): (float(row.get(key, 0.0) or 0.0) - avg) / std for row in candidates}


def base_scores(candidates: List[Dict[str, Any]], scorer: str) -> Dict[str, float]:
    z_query = zscore_map(candidates, "query_similarity")
    z_keyword = zscore_map(candidates, "query_keyword_overlap")
    z_entity = zscore_map(candidates, "entity_overlap")
    z_drm = zscore_map(candidates, "sentence_drm_proxy")
    z_fup = zscore_map(candidates, "fup_score_parent_page")
    z_oracle = zscore_map(candidates, "answer_overlap_oracle_diag")
    scores: Dict[str, float] = {}
    for row in candidates:
        sid = str(row["span_id"])
        query = z_query.get(sid, 0.0) + 0.5 * z_keyword.get(sid, 0.0) + 0.25 * z_entity.get(sid, 0.0)
        if scorer == "query":
            score = query
        elif scorer == "drm":
            score = z_drm.get(sid, 0.0) + 0.25 * query
        elif scorer == "fup":
            score = z_fup.get(sid, 0.0) + 0.25 * query
        elif scorer == "oracle":
            score = 2.0 * z_oracle.get(sid, 0.0) + 0.25 * z_drm.get(sid, 0.0)
        else:
            score = 0.6 * query + 0.8 * z_drm.get(sid, 0.0) + 0.5 * z_fup.get(sid, 0.0) + 0.2 * z_entity.get(sid, 0.0)
        scores[sid] = float(score)
    return scores


def max_redundancy(candidate: Dict[str, Any], selected: List[Dict[str, Any]]) -> float:
    current = text_words(str(candidate.get("span_text_preview", "")))
    best = 0.0
    for row in selected:
        other = text_words(str(row.get("span_text_preview", "")))
        if current or other:
            best = max(best, len(current & other) / max(1, len(current | other)))
    return best


def select_context(meta: Dict[str, Any], spec: Dict[str, Any], method: str) -> Dict[str, Any]:
    task = str(meta["task"])
    context_length = int(meta["context_length"])
    target = int(meta.get("target_final_tokens", context_length) or context_length)
    active = page_indices(meta.get("active_pages", []), context_length)
    final_full = page_indices(meta.get("final_pages", []), context_length)
    rescue_full = page_indices(meta.get("rescue_pages", []), context_length)
    fallback_reason = str(meta.get("fallback_reason", "") or "")
    if task == "passage_retrieval_en" and bool(spec.get("retrieval_fallback", True)):
        fallback_reason = fallback_reason or "passage_retrieval_en_full_page_fallback"
    if task == "passage_count" and bool(spec.get("count_fallback", True)):
        fallback_reason = fallback_reason or "passage_count_full_page_fallback"
    if fallback_reason:
        selected_abs = sorted(final_full if final_full else set(range(context_length)))
        return {
            "task": task, "sample_id": str(meta["sample_id"]), "method": method,
            "selected_abs_indices": selected_abs, "selected_spans": [],
            "target_final_tokens": target, "selected_token_count": len(selected_abs),
            "fallback_used": True, "fallback_reason": fallback_reason,
            "avg_spans_per_page": 0.0, "avg_tokens_per_rescue": 0.0,
            "unique_source_pages": 0, "compression_ratio": 1.0,
            "evidence_density": 0.0, "budget_warning": 0,
        }

    candidates = [dict(row) for row in meta.get("candidates", [])]
    scores = base_scores(candidates, str(spec["scorer"]))
    selected: List[Dict[str, Any]] = []
    selected_abs: Set[int] = set(active)
    used_span_ids: Set[str] = set()

    def add(row: Dict[str, Any], reason: str) -> bool:
        span_id = str(row["span_id"])
        if span_id in used_span_ids:
            return False
        indices = [
            index for index in range(int(row["token_start"]), int(row["token_end"]))
            if index not in selected_abs and 0 <= index < context_length
        ]
        if not indices:
            used_span_ids.add(span_id)
            return False
        remaining = max(0, target - len(selected_abs))
        if spec["packing"] == "budget_refill_spans" and remaining <= 0:
            return False
        if spec["packing"] == "budget_refill_spans" and len(indices) > remaining:
            indices = indices[:remaining]
        if not indices:
            return False
        red = max_redundancy(row, selected)
        item = dict(row)
        item.update({
            "method": method,
            "selected": 1,
            "redundancy_with_selected_spans": round(red, 7),
            "final_span_score": round(scores.get(span_id, 0.0) - 0.25 * red, 7),
            "selected_token_start": min(indices),
            "selected_token_end": max(indices) + 1,
            "selected_token_len": len(indices),
            "packing_reason": reason,
        })
        selected.append(item)
        selected_abs.update(indices)
        used_span_ids.add(span_id)
        return True

    rescue_pages = {int(page) for page in meta.get("rescue_pages", [])}
    for page in sorted(rescue_pages):
        page_candidates = [row for row in candidates if int(row["page_id"]) == page]
        page_candidates.sort(key=lambda row: scores.get(str(row["span_id"]), 0.0), reverse=True)
        for row in page_candidates[: int(spec["k"])]:
            add(row, "selected_rescue_page_topk")

    if spec["packing"] == "budget_refill_spans":
        while len(selected_abs) < target:
            remaining = [row for row in candidates if str(row["span_id"]) not in used_span_ids]
            if not remaining:
                break
            best = max(
                remaining,
                key=lambda row: scores.get(str(row["span_id"]), 0.0) - 0.25 * max_redundancy(row, selected),
            )
            if not add(best, "budget_refill"):
                used_span_ids.add(str(best["span_id"]))

    selected_abs_sorted = sorted(selected_abs)
    source_pages = {int(row["page_id"]) for row in selected}
    span_tokens = sum(int(row.get("selected_token_len", 0) or 0) for row in selected)
    rescue_tokens = len(rescue_full)
    warning = int(spec["packing"] == "budget_refill_spans" and abs(len(selected_abs_sorted) - target) / max(1, target) >= 0.01)
    return {
        "task": task, "sample_id": str(meta["sample_id"]), "method": method,
        "selected_abs_indices": selected_abs_sorted, "selected_spans": selected,
        "target_final_tokens": target, "selected_token_count": len(selected_abs_sorted),
        "fallback_used": False, "fallback_reason": "",
        "avg_spans_per_page": round(len(selected) / max(1, len(source_pages)), 4),
        "avg_tokens_per_rescue": round(span_tokens / max(1, len(source_pages)), 4),
        "unique_source_pages": len(source_pages),
        "compression_ratio": round(span_tokens / max(1, rescue_tokens), 7),
        "evidence_density": round(mean(float(row.get("answer_overlap_oracle_diag", 0.0) or 0.0) for row in selected), 7),
        "budget_warning": warning,
    }


class SpanDriver:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.out = Path(args.output_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "preds").mkdir(exist_ok=True)
        self.harness: Optional[FupHarness] = None
        self.ctx: Dict[Tuple[str, str], SampleCtx] = {}
        self.sample_order: Dict[str, List[str]] = {}
        self.answers: Dict[Tuple[str, str], Any] = {}
        self.lengths: Dict[Tuple[str, str], int] = {}
        self.all_classes: Dict[str, Any] = {}
        self.meta: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self.selections: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        self.results: Dict[str, Dict[str, Any]] = {}

    def log(self, *args: Any) -> None:
        print("[span-drm]", *args, flush=True)

    def ensure_harness(self) -> FupHarness:
        if self.harness is None:
            self.log("loading model ...")
            self.harness = FupHarness(self.args.model_path, self.args.longbench_dir, str(self.out / "_hwork"))
        return self.harness

    def load_contexts(self) -> None:
        harness = self.ensure_harness()
        for task in self.args.tasks:
            ids: List[str] = []
            for index, sample in enumerate(harness.load_samples(task, self.args.samples_per_task)):
                ctx = harness.build_sample_ctx(task, sample, index)
                sid = str(ctx.sample_id)
                self.ctx[(task, sid)] = ctx
                self.answers[(task, sid)] = ctx.answers
                self.lengths[(task, sid)] = ctx.context_length
                self.all_classes[task] = ctx.all_classes
                ids.append(sid)
            self.sample_order[task] = ids
            self.log(f"contexts {task}: {len(ids)}")

    def load_candidates(self) -> None:
        for row in read_jsonl(self.out / "_span_candidates.jsonl"):
            self.meta[(str(row["task"]), str(row["sample_id"]))] = row

    def baseline_predictions(self, task: str) -> Dict[str, Dict[str, Any]]:
        return read_jsonl_index(Path(self.args.baseline_dir) / "preds" / f"fixed_drm_add__{task}.jsonl")

    def prepare_selections(self) -> None:
        feature_rows: List[Dict[str, Any]] = []
        context_rows: List[Dict[str, Any]] = []
        fallback_rows: List[Dict[str, Any]] = []
        for method, spec in METHOD_SPECS.items():
            for task in self.args.tasks:
                task_selections = []
                for sid in self.sample_order[task]:
                    selection = select_context(self.meta[(task, sid)], spec, method)
                    self.selections[(method, task, sid)] = selection
                    task_selections.append(selection)
                    candidates = self.meta[(task, sid)].get("candidates", [])
                    scores = base_scores(candidates, str(spec["scorer"]))
                    selected_by_id = {
                        str(row["span_id"]): row for row in selection["selected_spans"]
                    }
                    if method == FEATURE_EXPORT_METHOD:
                        for row in candidates:
                            item = dict(row)
                            item["method"] = method
                            item["selected"] = 0
                            item["final_span_score"] = scores.get(str(row["span_id"]), 0.0)
                            item.update(selected_by_id.get(str(row["span_id"]), {}))
                            feature_rows.append(item)
                    context_rows.append({
                        key: value for key, value in selection.items() if key != "selected_abs_indices"
                    })
                fallback_rows.append({
                    "method": method, "task": task, "n_samples": len(task_selections),
                    "fallback_count": sum(1 for row in task_selections if row["fallback_used"]),
                    "fallback_rate_by_task": round(mean(1.0 if row["fallback_used"] else 0.0 for row in task_selections), 7),
                    "avg_spans_per_page": round(mean(row["avg_spans_per_page"] for row in task_selections), 7),
                    "avg_tokens_per_rescue": round(mean(row["avg_tokens_per_rescue"] for row in task_selections), 7),
                    "unique_source_pages": round(mean(row["unique_source_pages"] for row in task_selections), 7),
                    "compression_ratio": round(mean(row["compression_ratio"] for row in task_selections), 7),
                })
        feature_header = [
            "task", "sample_id", "method", "page_id", "span_id", "span_text_len", "span_token_len",
            "sentence_position", "page_position", "query_similarity", "query_keyword_overlap",
            "entity_overlap", "drm_score_parent_page", "sentence_drm_proxy", "fup_score_parent_page",
            "answer_overlap_oracle_diag", "redundancy_with_selected_spans", "source_page_selected_by",
            "final_span_score", "selected",
        ]
        write_csv(self.out / "span_candidate_features.csv", feature_rows, feature_header)
        write_jsonl(self.out / "span_selected_contexts.jsonl", context_rows)
        write_csv(self.out / "span_fallback_audit.csv", fallback_rows)

    def run_baseline(self) -> Dict[str, Any]:
        from longbench_eval import scorer

        result: Dict[str, Any] = {"method": "fixed_drm_add", "per_task": {}, "runner_fail": 0}
        for task in self.args.tasks:
            existing = self.baseline_predictions(task)
            preds = [str(existing.get(sid, {}).get("pred", "") or "") for sid in self.sample_order[task]]
            answers = [self.answers[(task, sid)] for sid in self.sample_order[task]]
            lengths = [self.lengths[(task, sid)] for sid in self.sample_order[task]]
            result["runner_fail"] += sum(1 for pred in preds if not pred.strip())
            result["per_task"][task] = float(scorer(task, preds, answers, lengths, self.all_classes.get(task, [])))
        result["avg"] = mean(result["per_task"].values())
        result["budget_error"] = 0.0
        result["budget_warning"] = 0
        return result

    def run_method(self, method: str, spec: Dict[str, Any]) -> Dict[str, Any]:
        from longbench_eval import scorer

        harness = self.ensure_harness()
        result: Dict[str, Any] = {"method": method, "per_task": {}, "runner_fail": 0}
        inherit = str(spec.get("inherit", "") or "")
        target_tasks = set(spec.get("target_tasks", self.args.tasks))
        for task in self.args.tasks:
            pred_path = self.out / "preds" / f"{method}__{task}.jsonl"
            existing = read_jsonl_index(pred_path) if self.args.skip_existing else {}
            baseline = self.baseline_predictions(task)
            inherited = read_jsonl_index(self.out / "preds" / f"{inherit}__{task}.jsonl") if inherit else {}
            preds, answers, lengths = [], [], []
            for sid in self.sample_order[task]:
                prev = existing.get(sid)
                ok = prev is not None and str(prev.get("status", "success")) == "success" and str(prev.get("pred", "")).strip()
                selection = self.selections[(method, task, sid)]
                if not ok or not self.args.skip_existing or self.args.overwrite_failed:
                    if inherit and task not in target_tasks and sid in inherited:
                        row = inherited[sid]
                        pred, status = str(row.get("pred", "") or ""), str(row.get("status", "success"))
                    elif selection["fallback_used"]:
                        row = baseline.get(sid, {})
                        pred, status = str(row.get("pred", "") or ""), str(row.get("status", "success"))
                    else:
                        release_cuda()
                        try:
                            pred = harness.eval_compact_context(
                                self.ctx[(task, sid)],
                                selection["selected_abs_indices"],
                            )["pred"]
                            status = "success" if str(pred).strip() else "failed"
                        except Exception as exc:
                            pred, status = "", "failed"
                            self.log(f"FAIL {method}/{task}/{sid}: {type(exc).__name__}: {exc}")
                    existing[sid] = {
                        "sample_id": sid, "pred": pred, "status": status,
                        "selected_token_count": selection["selected_token_count"],
                    }
                    write_jsonl(
                        pred_path,
                        [existing[item] for item in self.sample_order[task] if item in existing],
                    )
                row = existing.get(sid, {"pred": "", "status": "failed"})
                preds.append(str(row.get("pred", "") or ""))
                answers.append(self.answers[(task, sid)])
                lengths.append(self.lengths[(task, sid)])
            write_jsonl(pred_path, [existing.get(sid, {"sample_id": sid, "pred": "", "status": "failed"}) for sid in self.sample_order[task]])
            result["runner_fail"] += sum(1 for pred in preds if not pred.strip())
            result["per_task"][task] = float(scorer(task, preds, answers, lengths, self.all_classes.get(task, [])))
            self.log(f"{method}/{task}: {result['per_task'][task]:.3f}")
        result["avg"] = mean(result["per_task"].values())
        selections = [self.selections[(method, task, sid)] for task in self.args.tasks for sid in self.sample_order[task]]
        result["budget_error"] = mean(
            abs(float(row["selected_token_count"]) - float(row["target_final_tokens"])) / max(1.0, float(row["target_final_tokens"]))
            for row in selections
        )
        result["budget_warning"] = sum(int(row.get("budget_warning", 0) or 0) for row in selections)
        return result


def gate(result: Dict[str, Any], baseline: Dict[str, Any], method: str) -> Tuple[bool, List[str]]:
    if "oracle_diag" in method:
        return False, ["diagnostic_only"]
    reasons: List[str] = []
    if float(result["avg"]) <= float(baseline["avg"]):
        reasons.append("avg<=baseline")
    for task, score in result["per_task"].items():
        if float(score) < float(baseline["per_task"].get(task, 0.0)) - 3.0:
            reasons.append(f"{task} drop>3.0")
    if float(result.get("budget_error", 0.0)) >= 0.01:
        reasons.append("budget_error>=1%")
    if int(result.get("budget_warning", 0)) != 0:
        reasons.append("budget_warning")
    if int(result.get("runner_fail", 0)) != 0:
        reasons.append("runner_fail")
    return not reasons, reasons


def write_outputs(drv: SpanDriver) -> None:
    baseline = drv.results["fixed_drm_add"]
    summary: List[Dict[str, Any]] = []
    budget_rows: List[Dict[str, Any]] = []
    for method, result in drv.results.items():
        passed, reasons = gate(result, baseline, method)
        row: Dict[str, Any] = {
            "method": method, "attribute": ATTR.get(method, ""),
            "avg": round(float(result["avg"]), 4),
            "delta_vs_baseline": round(float(result["avg"]) - float(baseline["avg"]), 4),
        }
        for task in drv.args.tasks:
            row[task] = round(float(result["per_task"].get(task, 0.0)), 3)
        row.update({
            "budget_error_pct": round(float(result.get("budget_error", 0.0)) * 100, 4),
            "budget_warning": int(result.get("budget_warning", 0)),
            "runner_fail": int(result.get("runner_fail", 0)),
            "passed_gate": "yes" if passed else "no",
            "gate_reasons": "; ".join(reasons),
        })
        summary.append(row)
        if method != "fixed_drm_add":
            for task in drv.args.tasks:
                sels = [drv.selections[(method, task, sid)] for sid in drv.sample_order[task]]
                budget_rows.append({
                    "method": method, "task": task, "n_samples": len(sels),
                    "avg_target_tokens": round(mean(row["target_final_tokens"] for row in sels), 3),
                    "avg_selected_tokens": round(mean(row["selected_token_count"] for row in sels), 3),
                    "budget_error_pct": round(mean(
                        abs(float(row["selected_token_count"]) - float(row["target_final_tokens"])) / max(1.0, float(row["target_final_tokens"]))
                        for row in sels
                    ) * 100, 4),
                    "budget_warning": sum(int(row.get("budget_warning", 0) or 0) for row in sels),
                })
    header = ["method", "attribute", "avg", "delta_vs_baseline"] + drv.args.tasks + [
        "budget_error_pct", "budget_warning", "runner_fail", "passed_gate", "gate_reasons",
    ]
    write_csv(drv.out / "span_stage2_64_summary.csv", summary, header)
    ranking = [row for row in summary if row["method"] != "fixed_drm_add" and "diagnostic_only" not in row["attribute"]]
    ranking.sort(key=lambda row: float(row["avg"]), reverse=True)
    write_csv(drv.out / "span_stage2_64_ranking.csv", ranking, header)
    write_csv(drv.out / "span_budget_audit.csv", budget_rows)
    write_case_analysis(drv)
    write_report(drv, summary)


def write_case_analysis(drv: SpanDriver) -> None:
    from longbench_eval import scorer

    rows: List[Dict[str, Any]] = []
    for method in drv.results:
        if method == "fixed_drm_add":
            continue
        for task in drv.args.tasks:
            baseline = drv.baseline_predictions(task)
            preds = read_jsonl_index(drv.out / "preds" / f"{method}__{task}.jsonl")
            for sid in drv.sample_order[task]:
                answer = [drv.answers[(task, sid)]]
                length = [drv.lengths[(task, sid)]]
                classes = drv.all_classes.get(task, [])
                base_score = float(scorer(task, [str(baseline.get(sid, {}).get("pred", "") or "")], answer, length, classes))
                method_score = float(scorer(task, [str(preds.get(sid, {}).get("pred", "") or "")], answer, length, classes))
                selection = drv.selections[(method, task, sid)]
                rows.append({
                    "task": task, "sample_id": sid, "method": method,
                    "method_score": round(method_score, 4), "fixed_drm_add_score": round(base_score, 4),
                    "delta": round(method_score - base_score, 4),
                    "fallback_used": selection["fallback_used"],
                    "selected_token_count": selection["selected_token_count"],
                    "target_final_tokens": selection["target_final_tokens"],
                    "compression_ratio": selection["compression_ratio"],
                    "evidence_density": selection["evidence_density"],
                })
    write_csv(drv.out / "span_case_analysis.csv", rows)


def write_report(drv: SpanDriver, summary: List[Dict[str, Any]]) -> None:
    by_name = {row["method"]: row for row in summary}
    baseline = by_name["fixed_drm_add"]
    main = [name for name in ["span_drm_add_query", "span_drm_add_drm", "span_drm_add_fup", "span_drm_add_hybrid"] if name in by_name]
    best = max(main, key=lambda name: float(by_name[name]["avg"])) if main else ""
    k_variants = [name for name in ["span_drm_add_hybrid_k1", "span_drm_add_hybrid", "span_drm_add_hybrid_k3"] if name in by_name]
    best_k = max(k_variants, key=lambda name: float(by_name[name]["avg"])) if k_variants else ""
    hybrid = by_name.get("span_drm_add_hybrid", {})
    same = by_name.get("span_drm_add_hybrid_same_num_pages", {})
    retrieval_compress = by_name.get("span_drm_add_hybrid_retrieval_compress", {})
    count_compress = by_name.get("span_drm_add_hybrid_count_compress", {})
    oracle = by_name.get("span_drm_add_oracle_diag", {})
    passed = [row["method"] for row in summary if row.get("passed_gate") == "yes"]
    case_rows = []
    case_path = drv.out / "span_case_analysis.csv"
    if case_path.exists():
        with case_path.open("r", encoding="utf-8") as f:
            case_rows = list(csv.DictReader(f))
    hybrid_cases = [row for row in case_rows if row.get("method") == "span_drm_add_hybrid" and row.get("fallback_used") in {"False", "false", "0"}]
    density = mean(float(row.get("evidence_density", 0.0) or 0.0) for row in hybrid_cases)
    if float(oracle.get("avg", 0.0) or 0.0) <= float(baseline["avg"]) + 0.1:
        failure = "span compression loses context or page-level candidates are already strong"
    elif float(hybrid.get("avg", 0.0) or 0.0) <= float(baseline["avg"]):
        failure = "span scorer noise or budget refill introduces distractors"
    else:
        failure = "see per-task drops and budget audit"
    lines = [
        "# SpanDRM-Add Stage2 64-sample Diagnostic",
        "",
        "- Active set: fixed SlimInfer top-K pages, preserved in full.",
        "- Rescue granularity: fixed 24-token intra-page windows packed in original token order.",
        "- Passage retrieval uses full-page fallback in main methods; passage count is tested with fallback and compression.",
        "- Packed span contexts are evaluated with ordinary FullKV. This is a granularity diagnostic, not a sparse physical-KV cache implementation.",
        "- Environment: requested conda `silm` is unavailable; the repo `.venv` was used.",
        "",
        "## Results",
        "",
    ]
    for row in summary:
        lines.append(
            f"- `{row['method']}`: Avg {row['avg']}, delta {row['delta_vs_baseline']}, "
            f"gate {row['passed_gate']}, budget error {row['budget_error_pct']}%"
        )
    lines += [
        "",
        "## Questions",
        "",
        f"1. SpanDRM exceeds fixed DRM-Add: {bool(best) and float(by_name[best]['avg']) > float(baseline['avg'])}.",
        f"2. Best span scorer: {best or 'n/a'}.",
        f"3. Best k variant: {best_k or 'n/a'}.",
        f"4. Budget refill beats same-number-page compression: {float(hybrid.get('avg', 0.0) or 0.0) > float(same.get('avg', 0.0) or 0.0)}.",
        f"5. Passage retrieval fallback appears necessary: {float(retrieval_compress.get('passage_retrieval_en', 0.0) or 0.0) < float(hybrid.get('passage_retrieval_en', 0.0) or 0.0)}.",
        f"6. Passage count compression hurts: {float(count_compress.get('passage_count', 0.0) or 0.0) < float(hybrid.get('passage_count', 0.0) or 0.0)}.",
        f"7. Hybrid selected-span oracle-overlap density: {density:.6f}.",
        f"8. Failure diagnosis if no method passes: {failure}.",
        "",
        "## Promotion",
        "",
        f"- Methods passing the strict gate: {passed if passed else 'none'}.",
        "- Oracle diagnostics are excluded from deployable ranking.",
        "- No full LongBench or latency run was performed.",
    ]
    (drv.out / "report_span_stage2_64.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Path:
    start = time.time()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    candidate_path = out / "_span_candidates.jsonl"
    if not candidate_path.exists() or not args.skip_existing:
        build_args = argparse.Namespace(
            tasks=args.tasks,
            samples_per_task=args.samples_per_task,
            span_tokens=args.span_tokens,
            max_model_len=32768,
            model_path=args.model_path,
            longbench_dir=args.longbench_dir,
            baseline_dir=args.baseline_dir,
            page_features=args.page_features,
            fup_scores=args.fup_scores,
            output_dir=str(out),
        )
        build_candidates(build_args)

    drv = SpanDriver(args)
    config = {
        "stage": "stage2_64",
        "tasks": args.tasks,
        "samples_per_task": args.samples_per_task,
        "span_unit": f"fixed_token_window_{args.span_tokens}",
        "main_k": 2,
        "main_packing": "budget_refill_spans",
        "candidate_feature_export_method": FEATURE_EXPORT_METHOD,
        "active_construction": "fixed SlimInfer top-K pages from final available baseline audit layer",
        "packed_context_backend": "ordinary FullKV diagnostic",
        "baseline_dir": args.baseline_dir,
        "python_env": str(REPO_ROOT / ".venv"),
        "env_note": "conda env silm unavailable; repo .venv used",
    }
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    run_sh = (
        "#!/usr/bin/env bash\nset -e\n"
        f"cd {REPO_ROOT}\n"
        f"scripts/run_with_silm.sh tools/run_span_drm_stage2_64.py --output_dir {out} --skip_existing\n"
    )
    (out / "run_span_stage2_64.sh").write_text(run_sh, encoding="utf-8")
    (out / "resume.sh").write_text(run_sh.replace("--skip_existing", "--skip_existing --overwrite_failed"), encoding="utf-8")
    for name in ["run_span_stage2_64.sh", "resume.sh"]:
        (out / name).chmod(0o755)

    drv.load_contexts()
    drv.load_candidates()
    drv.prepare_selections()
    drv.results["fixed_drm_add"] = drv.run_baseline()
    requested = {item for item in args.methods.split(",") if item}
    for method, spec in METHOD_SPECS.items():
        if requested and method not in requested:
            continue
        drv.log(f"eval {method}")
        drv.results[method] = drv.run_method(method, spec)
    write_outputs(drv)
    drv.log(f"DONE {(time.time() - start) / 60:.1f} min -> {out}")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument("--samples_per_task", type=int, default=64)
    parser.add_argument("--span_tokens", type=int, default=24)
    parser.add_argument("--methods", default="")
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--model_path", default=DEFAULT_MODEL)
    parser.add_argument("--longbench_dir", default=DEFAULT_LB)
    parser.add_argument("--baseline_dir", default=str(DEFAULT_FUP))
    parser.add_argument("--page_features", default=str(DEFAULT_FUP / "features_fup_candidates.csv"))
    parser.add_argument("--fup_scores", default=str(DEFAULT_FUP / "fup_output_delta_scores.csv"))
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--overwrite_failed", action="store_true")
    args = parser.parse_args()
    args.tasks = [task for task in args.tasks.split(",") if task]
    if not args.output_dir:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = str(REPO_ROOT / "results" / f"span_drm_stage2_64_{stamp}")
    return args


if __name__ == "__main__":
    run(parse_args())
