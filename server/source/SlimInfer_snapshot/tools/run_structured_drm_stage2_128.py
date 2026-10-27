#!/usr/bin/env python3
"""Structured DRM-Add 128-sample recheck with RH-then-Span hybrids."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import pickle
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from longbench_eval import scorer
from tools.build_span_candidates import (  # noqa: E402
    ENTITY_RE,
    PAGE_TOKENS,
    STOPWORDS,
    TOKEN_RE,
    entities,
    fixed_page_sets_from_audit,
    jaccard,
    load_fup_scores,
    load_page_features,
    overlap,
    words,
)
from tools.fup_guarded_swap_lib import FupHarness, SampleCtx, release_cuda  # noqa: E402
from tools.run_fup_guarded_swap_stage2_64 import FEATURE_COLS as FUP_FEATURE_COLS  # noqa: E402
from tools.run_rh_drm_stage2_64 import RhDriver  # noqa: E402
from tools.run_span_drm_stage2_64 import base_scores, max_redundancy  # noqa: E402

DEFAULT_MODEL = "/home/yu-bai/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
DEFAULT_LB = "/home/yu-bai/Desktop/jiaqizhang/HeadKV/data/LongBench"
DEFAULT_PROFILE = REPO_ROOT / "profiles" / "retrieval_head_profiles"
DEFAULT_FUP = REPO_ROOT / "results" / "fup_guarded_swap_stage2_64_20260601_051737"
DEFAULT_AUDIT = REPO_ROOT / "results" / "structured_drm_recheck_audit"
DEFAULT_FULL_BASELINE = REPO_ROOT / "results" / "rescuekv_drm_add_full_longbench_20260523_233633"
TASKS = ["musique", "hotpotqa", "qasper", "passage_count", "passage_retrieval_en"]
QA_TASKS = {"musique", "hotpotqa", "qasper"}

PAGE_METHOD_ATTR = {
    "fixed_drm_add": "deployable baseline",
    "fup_oracle_keepdrop": "learned_pseudo_label_reference_not_main",
    "rh_drm_add_static_fixed_pool": "deployable RH fixed-pool static head profile",
    "rh_drm_add_query_fixed_pool": "deployable RH fixed-pool query-conditioned head profile",
}

SPAN_METHOD_SPECS: Dict[str, Dict[str, Any]] = {
    "span_drm_add_query": {
        "source": "fixed_drm_add", "scorer": "query", "k": 2,
        "packing": "budget_refill_spans", "count_fallback": True, "retrieval_fallback": True,
    },
    "span_drm_add_hybrid": {
        "source": "fixed_drm_add", "scorer": "hybrid", "k": 2,
        "packing": "budget_refill_spans", "count_fallback": True, "retrieval_fallback": True,
    },
    "rh_then_span_drm_add_query": {
        "source": "rh_drm_add_query_fixed_pool", "scorer": "query", "k": 2,
        "packing": "budget_refill_spans", "count_fallback": True, "retrieval_fallback": True,
    },
    "rh_then_span_drm_add_hybrid": {
        "source": "rh_drm_add_query_fixed_pool", "scorer": "hybrid", "k": 2,
        "packing": "budget_refill_spans", "count_fallback": True, "retrieval_fallback": True,
    },
    "span_drm_add_query_k1": {
        "source": "fixed_drm_add", "scorer": "query", "k": 1,
        "packing": "budget_refill_spans", "count_fallback": True, "retrieval_fallback": True,
        "inherit": "span_drm_add_query", "target_tasks": QA_TASKS,
    },
    "span_drm_add_query_k3": {
        "source": "fixed_drm_add", "scorer": "query", "k": 3,
        "packing": "budget_refill_spans", "count_fallback": True, "retrieval_fallback": True,
        "inherit": "span_drm_add_query", "target_tasks": QA_TASKS,
    },
    "span_drm_add_query_same_num_pages": {
        "source": "fixed_drm_add", "scorer": "query", "k": 2,
        "packing": "same_num_pages_compressed", "count_fallback": True, "retrieval_fallback": True,
        "inherit": "span_drm_add_query", "target_tasks": QA_TASKS,
    },
    "span_drm_add_query_count_compress": {
        "source": "fixed_drm_add", "scorer": "query", "k": 2,
        "packing": "budget_refill_spans", "count_fallback": False, "retrieval_fallback": True,
        "inherit": "span_drm_add_query", "target_tasks": {"passage_count"},
    },
    "span_drm_add_query_retrieval_compress": {
        "source": "fixed_drm_add", "scorer": "query", "k": 2,
        "packing": "budget_refill_spans", "count_fallback": True, "retrieval_fallback": False,
        "inherit": "span_drm_add_query", "target_tasks": {"passage_retrieval_en"},
    },
    "span_drm_add_oracle_diag": {
        "source": "fixed_drm_add", "scorer": "oracle", "k": 2,
        "packing": "budget_refill_spans", "count_fallback": True, "retrieval_fallback": True,
        "diagnostic_only": True,
    },
    "rh_then_span_drm_add_oracle_diag": {
        "source": "rh_drm_add_query_fixed_pool", "scorer": "oracle", "k": 2,
        "packing": "budget_refill_spans", "count_fallback": True, "retrieval_fallback": True,
        "diagnostic_only": True,
    },
}

METHOD_ATTR = {
    **PAGE_METHOD_ATTR,
    "span_drm_add_query": "deployable SpanDRM query k=2 budget-refill",
    "span_drm_add_hybrid": "deployable SpanDRM hybrid k=2 budget-refill",
    "rh_then_span_drm_add_query": "deployable RH query fixed-pool pages + Span query k=2",
    "rh_then_span_drm_add_hybrid": "deployable RH query fixed-pool pages + Span hybrid k=2",
    "span_drm_add_query_k1": "span k ablation",
    "span_drm_add_query_k3": "span k ablation",
    "span_drm_add_query_same_num_pages": "packing ablation",
    "span_drm_add_query_count_compress": "passage_count negative compression ablation",
    "span_drm_add_query_retrieval_compress": "passage_retrieval negative compression ablation",
    "span_drm_add_oracle_diag": "diagnostic_only",
    "rh_then_span_drm_add_oracle_diag": "diagnostic_only",
}


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
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


def latest_layer_records(path: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in read_jsonl(path):
        sid = str(row.get("sample_id", ""))
        if not sid:
            continue
        old = out.get(sid)
        if old is None or int(row.get("layer_id", -1) or -1) >= int(old.get("layer_id", -1) or -1):
            out[sid] = row
    return out


def zscore(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    avg = mean(values)
    std = math.sqrt(mean((float(value) - avg) ** 2 for value in values))
    if std <= 1e-9:
        return [0.0 for _ in values]
    return [(float(value) - avg) / std for value in values]


def page_indices(pages: Sequence[int], context_length: int) -> Set[int]:
    out: Set[int] = set()
    for page in pages:
        start = int(page) * PAGE_TOKENS
        out.update(range(max(0, start), min(context_length, start + PAGE_TOKENS)))
    return out


def audit_path(out: Path, method: str, task: str) -> Path:
    return out / "preds" / f"{method}__{task}__audit.jsonl"


def pred_path(out: Path, method: str, task: str) -> Path:
    return out / "preds" / f"{method}__{task}.jsonl"


def source_status_path(out: Path, method: str, task: str) -> Path:
    return out / "preds" / f"{method}__{task}__source_status.jsonl"


def fup_feature_vector(row: Dict[str, Any], payload: Dict[str, Any], task: str) -> List[float]:
    sources = list(payload.get("sources", []))
    tasks = list(payload.get("tasks", TASKS))
    vec = [float(row.get(key, 0.0) or 0.0) for key in payload.get("feature_cols", FUP_FEATURE_COLS)]
    vec.extend(1.0 if str(row.get("candidate_source", "")) == source else 0.0 for source in sources)
    vec.extend(1.0 if task == item else 0.0 for item in tasks)
    return vec


class StructuredDriver:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.out = Path(args.output_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "preds").mkdir(exist_ok=True)
        (self.out / "_features").mkdir(exist_ok=True)
        self.rh_args = argparse.Namespace(
            output_dir=str(self.out),
            model_path=args.model_path,
            longbench_dir=args.longbench_dir,
            profile_dir=args.profile_dir,
            reuse_baseline_dir=str(self.out / "_no_reuse_baseline"),
            fup_labels=str(Path(args.fup_dir) / "_labels.jsonl"),
            tasks=args.tasks,
            samples_per_task=args.samples_per_task,
            methods="",
            rh_alpha_drm=args.rh_alpha_drm,
            rh_beta_retrieval=args.rh_beta_retrieval,
            rh_beta_query=args.rh_beta_query,
            rh_beta_reasoning=args.rh_beta_reasoning,
            rh_lambda_head=args.rh_lambda_head,
            rh_lambda_page=args.rh_lambda_page,
            skip_existing=args.skip_existing,
            overwrite_failed=args.overwrite_failed,
        )
        self.rh = RhDriver(self.rh_args)
        self.results: Dict[str, Dict[str, Any]] = {}
        self.selections: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        self.meta: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        self.page_features = load_page_features(Path(args.page_features))
        self.fup_scores = load_fup_scores(Path(args.fup_scores))

    def log(self, *args: Any) -> None:
        print("[structured-128]", *args, flush=True)

    @property
    def harness(self) -> FupHarness:
        return self.rh.ensure_harness()

    def load_contexts(self) -> None:
        self.rh.load_contexts()

    def page_override(self, method: str) -> Callable[[str, str], Dict[str, Any]]:
        if method == "fixed_drm_add":
            return self.rh.baseline_override
        if method == "rh_drm_add_static_fixed_pool":
            return self.rh.override("rh_static", "fixed")
        if method == "rh_drm_add_query_fixed_pool":
            return self.rh.override("rh_query", "fixed")
        if method == "fup_oracle_keepdrop":
            maps = self.build_fup_keepdrop_maps()

            def fn(task: str, sid: str) -> Dict[str, Any]:
                return {
                    "block_strategy": "fixed_active_rescue",
                    "rescue_candidate_strategy": "fixed",
                    "rescue_selector": "fup",
                    "page_utility_map": maps.get((task, sid), {}),
                }

            return fn
        raise KeyError(method)

    def run_page_method(self, method: str) -> Dict[str, Any]:
        self.log(f"eval page {method}")
        result = self.rh.run_eval_method(method, self.page_override(method))
        self.results[method] = result
        return result

    def ensure_page_methods(self) -> None:
        if self.args.evaluation_mode == "full" and self.args.source_audit_only:
            self.load_reused_fixed_baseline()
            self.prepare_source_audits("fixed_drm_add", self.rh.baseline_override)
            requested = set(self.args.methods)
            needs_rh = any(
                SPAN_METHOD_SPECS.get(method, {}).get("source") == "rh_drm_add_query_fixed_pool"
                for method in requested
            )
            if needs_rh:
                self.prepare_source_audits(
                    "rh_drm_add_query_fixed_pool",
                    self.rh.override("rh_query", "fixed"),
                )
            return
        methods = ["fixed_drm_add", "rh_drm_add_static_fixed_pool", "rh_drm_add_query_fixed_pool"]
        if self.args.include_fup_reference:
            methods.insert(1, "fup_oracle_keepdrop")
        requested = set(self.args.methods)
        for method in methods:
            if requested and method not in requested and not any(
                spec.get("source") == method for name, spec in SPAN_METHOD_SPECS.items() if name in requested
            ):
                continue
            self.run_page_method(method)

    def load_reused_fixed_baseline(self) -> None:
        from longbench_eval import scorer

        source_root = Path(self.args.reuse_full_baseline_dir) / "raw_predictions" / "drm_add"
        result: Dict[str, Any] = {"method": "fixed_drm_add", "per_task": {}, "runner_fail": 0}
        for task in self.args.tasks:
            source = read_jsonl_index(source_root / f"{task}.jsonl")
            rows: List[Dict[str, Any]] = []
            predictions: List[str] = []
            answers: List[Any] = []
            lengths: List[int] = []
            for sid in self.rh.sample_order[task]:
                row = source.get(sid, {})
                pred = str(row.get("pred", "") or "")
                status = "success" if pred.strip() else "failed"
                rows.append({"sample_id": sid, "pred": pred, "status": status, "source": "reused_full_fixed_drm_add"})
                predictions.append(pred)
                answers.append(self.rh.answers[(task, sid)])
                lengths.append(self.rh.lengths[(task, sid)])
                if status != "success":
                    result["runner_fail"] += 1
            write_jsonl(pred_path(self.out, "fixed_drm_add", task), rows)
            result["per_task"][task] = float(
                scorer(task, predictions, answers, lengths, self.rh.all_classes.get(task, []))
            )
            self.log(
                f"reused fixed_drm_add/{task}: {len(rows) - sum(r['status'] != 'success' for r in rows)}"
                f"/{len(rows)} score={result['per_task'][task]:.3f}"
            )
        result["avg"] = mean(result["per_task"].values())
        self.results["fixed_drm_add"] = result

    def prepare_source_audits(
        self,
        method: str,
        override_fn: Callable[[str, str], Dict[str, Any]],
    ) -> None:
        failures: List[Dict[str, Any]] = []
        for task in self.args.tasks:
            path = audit_path(self.out, method, task)
            status_path = source_status_path(self.out, method, task)
            if path.exists() and not self.args.skip_existing:
                path.unlink()
            if status_path.exists() and not self.args.skip_existing:
                status_path.unlink()
            records = latest_layer_records(path)
            statuses = read_jsonl_index(status_path)
            for sid in records:
                statuses.setdefault(sid, {
                    "sample_id": sid,
                    "status": "success",
                    "audit_record_found": True,
                    "source": "recovered_from_existing_audit",
                })
            if statuses:
                write_jsonl(
                    status_path,
                    [statuses[sid] for sid in self.rh.sample_order[task] if sid in statuses],
                )
            for index, sid in enumerate(self.rh.sample_order[task]):
                previous = statuses.get(sid)
                if previous is not None and str(previous.get("status", "")) == "success":
                    continue
                if (
                    previous is not None
                    and "empty_generation_output" in str(previous.get("error", ""))
                ):
                    statuses[sid] = {
                        "sample_id": sid,
                        "status": "success",
                        "audit_record_found": False,
                        "source": "one_token_eos_no_audit_fallback",
                    }
                    continue
                if previous is not None and not self.args.overwrite_failed:
                    continue
                release_cuda()
                override = override_fn(task, sid)
                override["_method"] = method
                override["rescue_audit_path"] = str(path)
                try:
                    self.harness.eval_pageset(
                        self.rh.ctx[(task, sid)],
                        override,
                        mode="gen",
                        gen_tokens=1,
                    )
                    statuses[sid] = {
                        "sample_id": sid,
                        "status": "success",
                        "audit_record_found": "",
                        "source": "one_token_source_prefill",
                    }
                except Exception as exc:
                    if isinstance(exc, RuntimeError) and str(exc) == "empty_generation_output":
                        statuses[sid] = {
                            "sample_id": sid,
                            "status": "success",
                            "audit_record_found": False,
                            "source": "one_token_eos_no_audit_fallback",
                        }
                        write_jsonl(
                            status_path,
                            [statuses[item] for item in self.rh.sample_order[task] if item in statuses],
                        )
                        continue
                    failure = {
                        "method": method,
                        "task": task,
                        "sample_id": sid,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    failures.append(failure)
                    statuses[sid] = {
                        "sample_id": sid,
                        "status": "failed",
                        "audit_record_found": False,
                        "error": failure["error"],
                    }
                    self.log(f"SOURCE FAIL {method}/{task}/{sid}: {type(exc).__name__}: {exc}")
                write_jsonl(
                    status_path,
                    [statuses[item] for item in self.rh.sample_order[task] if item in statuses],
                )
                if (index + 1) % 10 == 0 or index + 1 == len(self.rh.sample_order[task]):
                    self.log(f"source audit {method}/{task}: {index + 1}/{len(self.rh.sample_order[task])}")
            write_jsonl(
                status_path,
                [statuses[item] for item in self.rh.sample_order[task] if item in statuses],
            )
            records = latest_layer_records(path)
            success = sum(1 for row in statuses.values() if str(row.get("status", "")) == "success")
            self.log(
                f"source prep complete {method}/{task}: {success}/{len(self.rh.sample_order[task])}; "
                f"cache audits={len(records)}"
            )
        failure_path = self.out / "source_audit_failures.jsonl"
        if failures:
            write_jsonl(failure_path, failures)
        elif failure_path.exists():
            failure_path.unlink()

    def build_fup_keepdrop_maps(self) -> Dict[Tuple[str, str], Dict[int, float]]:
        feature_path = self.out / "_features" / "fup_reference_features.jsonl"
        if not feature_path.exists() or not self.args.skip_existing:
            if feature_path.exists():
                feature_path.unlink()
            self.log("extracting FUP reference features")
            for task in self.args.tasks:
                for sid in self.rh.sample_order.get(task, []):
                    ov = {
                        "block_strategy": "fixed_active_rescue",
                        "rescue_candidate_strategy": "fixed",
                        "feature_dump_path": str(feature_path),
                        "_method": "fup_reference_feature_extract",
                    }
                    release_cuda()
                    try:
                        self.harness.eval_pageset(self.rh.ctx[(task, sid)], ov, mode="gen", gen_tokens=1)
                    except Exception as exc:
                        self.log(f"FUP feature extract fail {task}/{sid}: {type(exc).__name__}: {exc}")
        model_path = Path(self.args.fup_dir) / "models" / "fup_oracle_keepdrop_model.pkl"
        if not model_path.exists():
            self.log("FUP reference model missing; using zero utility maps")
            return {}
        with model_path.open("rb") as f:
            payload = pickle.load(f)
        model = payload.get("model")
        if model is None:
            return {}
        rows = read_jsonl(feature_path)
        by_sample: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
        for row in rows:
            if int(row.get("is_candidate", 0) or 0) == 1:
                by_sample[(str(row.get("task", "")), str(row.get("sample_id", "")))].append(row)
        maps: Dict[Tuple[str, str], Dict[int, float]] = defaultdict(dict)
        try:
            import numpy as np
        except Exception:
            return {}
        for (task, sid), group in by_sample.items():
            if not group:
                continue
            X = np.array([fup_feature_vector(row, payload, task) for row in group], dtype=float)
            try:
                probs = model.predict_proba(X)[:, 1]
            except Exception:
                probs = model.predict(X)
            for row, prob in zip(group, probs.tolist()):
                page = int(row.get("page_id", -1))
                if page >= 0:
                    maps[(task, sid)][page] = max(float(maps[(task, sid)].get(page, -1e9)), float(prob))
        return maps

    def page_budget_result(self, method: str) -> Dict[str, Any]:
        baseline_records: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for task in self.args.tasks:
            for sid, row in latest_layer_records(audit_path(self.out, "fixed_drm_add", task)).items():
                baseline_records[(task, sid)] = row
        errors: List[float] = []
        warnings = 0
        for task in self.args.tasks:
            records = latest_layer_records(audit_path(self.out, method, task))
            for sid, row in records.items():
                cur = float(row.get("selected_token_count", 0.0) or 0.0)
                ref = float(baseline_records.get((task, sid), {}).get("selected_token_count", cur) or cur)
                if ref > 0:
                    errors.append(abs(cur - ref) / ref)
                warnings += int(bool(row.get("fallback_used", False)))
        return {"budget_error": mean(errors), "budget_warning": warnings}

    def source_meta(self, source: str, task: str, sid: str) -> Dict[str, Any]:
        key = (source, task, sid)
        if key in self.meta:
            return self.meta[key]
        ctx = self.rh.ctx[(task, sid)]
        records = latest_layer_records(audit_path(self.out, source, task))
        audit = records.get(sid)
        meta = self.build_span_meta_from_audit(source, ctx, audit)
        self.meta[key] = meta
        return meta

    def build_span_meta_from_audit(self, source: str, ctx: SampleCtx, audit: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        input_ids = [int(x) for x in ctx.input_ids[0].detach().cpu().tolist()]
        context_length = len(input_ids)
        num_pages = int(math.ceil(context_length / PAGE_TOKENS))
        if audit:
            final_pages, active_pages, rescue_pages = fixed_page_sets_from_audit(audit)
            target_final_tokens = int(audit.get("selected_token_count", 0) or 0)
            source_layer = int(audit.get("layer_id", -1) or -1)
            fallback_reason = ""
        else:
            final_pages = set(range(num_pages))
            active_pages = set(final_pages)
            rescue_pages = set()
            target_final_tokens = context_length
            source_layer = -1
            fallback_reason = f"missing_{source}_audit_or_no_pruning"
        final_pages = {page for page in final_pages if 0 <= int(page) < num_pages}
        active_pages = {page for page in active_pages if 0 <= int(page) < num_pages}
        rescue_pages = {page for page in rescue_pages if 0 <= int(page) < num_pages}
        if not final_pages:
            final_pages = set(range(num_pages))
            active_pages = set(final_pages)
            rescue_pages = set()
            target_final_tokens = context_length
            fallback_reason = f"empty_{source}_final_pages"
        query_words = words(ctx.query_text)
        query_entities = entities(ctx.query_text)
        answer_text = " ".join(str(item) for item in ctx.answers)
        answer_words = words(answer_text)
        candidates: List[Dict[str, Any]] = []
        tokenizer = self.harness.tokenizer
        for page in range(num_pages):
            if page in active_pages:
                continue
            start = page * PAGE_TOKENS
            end = min(context_length, start + PAGE_TOKENS)
            page_text = tokenizer.decode(input_ids[start:end], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            title = next((line.strip() for line in page_text.splitlines() if line.strip()), f"page_{page}")
            for local_index, span_start in enumerate(range(start, end, self.args.span_tokens)):
                span_end = min(end, span_start + self.args.span_tokens)
                if span_end <= span_start:
                    continue
                text = tokenizer.decode(
                    input_ids[span_start:span_end],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                span_words = words(text)
                parent = self.page_features.get((ctx.task, str(ctx.sample_id), page), {})
                drm = float(parent.get("drm", 0.0))
                query_similarity = jaccard(query_words, span_words)
                keyword_overlap = overlap(query_words, span_words)
                entity_overlap = overlap(query_entities, entities(text))
                answer_overlap = overlap(answer_words, span_words)
                candidates.append({
                    "task": ctx.task,
                    "sample_id": str(ctx.sample_id),
                    "page_id": page,
                    "span_id": f"{source}:p{page}_w{local_index}",
                    "span_text_len": len(text),
                    "span_token_len": span_end - span_start,
                    "sentence_position": local_index,
                    "page_position": round(page / max(1, num_pages - 1), 7),
                    "query_similarity": round(query_similarity, 7),
                    "query_keyword_overlap": round(keyword_overlap, 7),
                    "entity_overlap": round(entity_overlap, 7),
                    "drm_score_parent_page": round(drm, 7),
                    "sentence_drm_proxy": round(drm * (0.5 + query_similarity + 0.5 * keyword_overlap), 7),
                    "fup_score_parent_page": round(float(self.fup_scores.get((ctx.task, str(ctx.sample_id), page), 0.0)), 7),
                    "answer_overlap_oracle_diag": round(answer_overlap, 7),
                    "redundancy_with_selected_spans": 0.0,
                    "source_page_selected_by": source if page in rescue_pages else f"{source}_budget_refill_pool",
                    "final_span_score": 0.0,
                    "selected": 0,
                    "token_start": span_start,
                    "token_end": span_end,
                    "page_title": title[:160],
                    "span_text_preview": text[:240],
                })
        return {
            "task": ctx.task,
            "sample_id": str(ctx.sample_id),
            "sample_index": ctx.sample_index,
            "context_length": context_length,
            "num_pages": num_pages,
            "target_final_tokens": int(target_final_tokens),
            "source_layer": source_layer,
            "active_pages": sorted(active_pages),
            "rescue_pages": sorted(rescue_pages),
            "final_pages": sorted(final_pages),
            "fallback_reason": fallback_reason,
            "candidates": candidates,
            "source_method": source,
            "query_text": ctx.query_text,
        }

    def select_context(self, meta: Dict[str, Any], spec: Dict[str, Any], method: str) -> Dict[str, Any]:
        task = str(meta["task"])
        context_length = int(meta["context_length"])
        target = int(meta.get("target_final_tokens", context_length) or context_length)
        active = page_indices(meta.get("active_pages", []), context_length)
        final_full = page_indices(meta.get("final_pages", []), context_length)
        rescue_full = page_indices(meta.get("rescue_pages", []), context_length)
        fallback_reason = str(meta.get("fallback_reason", "") or "")
        if task == "passage_retrieval_en" and bool(spec.get("retrieval_fallback", True)):
            fallback_reason = fallback_reason or "passage_retrieval_en_full_page_fallback_predefined"
        if task == "passage_count" and bool(spec.get("count_fallback", True)):
            fallback_reason = fallback_reason or "passage_count_full_page_fallback_predefined"
        if fallback_reason:
            selected_abs = sorted(final_full if final_full else set(range(context_length)))
            return {
                "task": task, "sample_id": str(meta["sample_id"]), "method": method,
                "source_method": str(meta.get("source_method", spec.get("source", ""))),
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
            red = max_redundancy(row, selected)
            item = dict(row)
            page_title = str(row.get("page_title", f"page_{row.get('page_id', '')}"))
            page_id = row.get("page_id", "")
            span_pos = row.get("sentence_position", "")
            item.update({
                "method": method,
                "selected": 1,
                "redundancy_with_selected_spans": round(red, 7),
                "final_span_score": round(scores.get(span_id, 0.0) - 0.25 * red, 7),
                "selected_token_start": min(indices),
                "selected_token_end": max(indices) + 1,
                "selected_token_len": len(indices),
                "packing_reason": reason,
                "source_metadata": f"[{page_title} | span {span_pos} | page_id {page_id}]",
            })
            selected.append(item)
            selected_abs.update(indices)
            used_span_ids.add(span_id)
            return True

        for page in sorted({int(page) for page in meta.get("rescue_pages", [])}):
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
        warning = int(
            spec["packing"] == "budget_refill_spans"
            and abs(len(selected_abs_sorted) - target) / max(1, target) >= 0.01
        )
        return {
            "task": task, "sample_id": str(meta["sample_id"]), "method": method,
            "source_method": str(meta.get("source_method", spec.get("source", ""))),
            "selected_abs_indices": selected_abs_sorted,
            "selected_spans": selected,
            "target_final_tokens": target,
            "selected_token_count": len(selected_abs_sorted),
            "fallback_used": False,
            "fallback_reason": "",
            "avg_spans_per_page": round(len(selected) / max(1, len(source_pages)), 4),
            "avg_tokens_per_rescue": round(span_tokens / max(1, len(source_pages)), 4),
            "unique_source_pages": len(source_pages),
            "compression_ratio": round(span_tokens / max(1, rescue_tokens), 7),
            "evidence_density": round(mean(float(row.get("answer_overlap_oracle_diag", 0.0) or 0.0) for row in selected), 7),
            "budget_warning": warning,
        }

    def prepare_span_selections(self, methods: Sequence[str]) -> None:
        context_rows: List[Dict[str, Any]] = []
        fallback_rows: List[Dict[str, Any]] = []
        for method in methods:
            spec = SPAN_METHOD_SPECS[method]
            for task in self.args.tasks:
                task_rows = []
                for sid in self.rh.sample_order[task]:
                    meta = self.source_meta(str(spec["source"]), task, sid)
                    selection = self.select_context(meta, spec, method)
                    self.selections[(method, task, sid)] = selection
                    task_rows.append(selection)
                    context_rows.append({key: value for key, value in selection.items() if key != "selected_abs_indices"})
                fallback_rows.append({
                    "method": method,
                    "task": task,
                    "n_samples": len(task_rows),
                    "fallback_count": sum(1 for row in task_rows if row["fallback_used"]),
                    "fallback_rate_by_task": round(mean(1.0 if row["fallback_used"] else 0.0 for row in task_rows), 7),
                    "avg_spans_per_page": round(mean(row["avg_spans_per_page"] for row in task_rows), 7),
                    "avg_tokens_per_rescue": round(mean(row["avg_tokens_per_rescue"] for row in task_rows), 7),
                    "unique_source_pages": round(mean(row["unique_source_pages"] for row in task_rows), 7),
                    "compression_ratio": round(mean(row["compression_ratio"] for row in task_rows), 7),
                })
        write_jsonl(self.out / "_span_selected_contexts_structured.jsonl", context_rows)
        write_csv(self.out / "structured_stage2_128_fallback_audit.csv", fallback_rows)

    def run_span_method(self, method: str) -> Dict[str, Any]:
        self.log(f"eval span {method}")
        spec = SPAN_METHOD_SPECS[method]
        inherit = str(spec.get("inherit", "") or "")
        target_tasks = set(spec.get("target_tasks", self.args.tasks))
        result: Dict[str, Any] = {"method": method, "per_task": {}, "runner_fail": 0}
        for task in self.args.tasks:
            path = pred_path(self.out, method, task)
            existing = read_jsonl_index(path) if self.args.skip_existing else {}
            inherited = read_jsonl_index(pred_path(self.out, inherit, task)) if inherit else {}
            source = str(spec["source"])
            source_preds = read_jsonl_index(pred_path(self.out, source, task))
            preds, answers, lengths = [], [], []
            for sid in self.rh.sample_order[task]:
                prev = existing.get(sid)
                ok = (
                    prev is not None and str(prev.get("status", "success")) == "success"
                    and str(prev.get("pred", "")).strip()
                )
                selection = self.selections[(method, task, sid)]
                should_run = (
                    not ok
                    and (
                        prev is None
                        or not self.args.skip_existing
                        or self.args.overwrite_failed
                    )
                )
                if should_run:
                    if inherit and task not in target_tasks and sid in inherited:
                        row = inherited[sid]
                        pred, status = str(row.get("pred", "") or ""), str(row.get("status", "success"))
                    elif selection["fallback_used"] and str(source_preds.get(sid, {}).get("pred", "")).strip():
                        row = source_preds[sid]
                        pred, status = str(row.get("pred", "") or ""), str(row.get("status", "success"))
                    else:
                        release_cuda()
                        pred, status = "", "failed"
                        last_exc: Optional[BaseException] = None
                        for attempt in range(3):
                            try:
                                pred = self.harness.eval_compact_context(
                                    self.rh.ctx[(task, sid)],
                                    selection["selected_abs_indices"],
                                )["pred"]
                                status = "success" if str(pred).strip() else "failed"
                                if status == "success":
                                    break
                            except BaseException as exc:
                                last_exc = exc
                                release_cuda()
                        if status != "success":
                            self.log(
                                f"FAIL {method}/{task}/{sid}: "
                                f"{type(last_exc).__name__ if last_exc else 'RuntimeError'}: {last_exc}"
                            )
                    existing[sid] = {
                        "sample_id": sid,
                        "pred": pred,
                        "status": status,
                        "selected_token_count": selection["selected_token_count"],
                    }
                    write_jsonl(path, [existing[item] for item in self.rh.sample_order[task] if item in existing])
                if (len(preds) + 1) % 10 == 0 or len(preds) + 1 == len(self.rh.sample_order[task]):
                    self.log(f"generation {method}/{task}: {len(preds) + 1}/{len(self.rh.sample_order[task])}")
                row = existing.get(sid, {"pred": "", "status": "failed"})
                preds.append(str(row.get("pred", "") or ""))
                answers.append(self.rh.answers[(task, sid)])
                lengths.append(self.rh.lengths[(task, sid)])
            write_jsonl(path, [existing.get(sid, {"sample_id": sid, "pred": "", "status": "failed"}) for sid in self.rh.sample_order[task]])
            result["runner_fail"] += sum(1 for pred in preds if not pred.strip())
            result["per_task"][task] = float(scorer(task, preds, answers, lengths, self.rh.all_classes.get(task, [])))
            self.log(f"{method}/{task}: {result['per_task'][task]:.3f}")
        result["avg"] = mean(result["per_task"].values())
        rows = [
            self.selections[(method, task, sid)]
            for task in self.args.tasks
            for sid in self.rh.sample_order[task]
        ]
        result["budget_error"] = mean(
            abs(float(row["selected_token_count"]) - float(row["target_final_tokens"])) / max(1.0, float(row["target_final_tokens"]))
            for row in rows
        )
        result["budget_warning"] = sum(int(row.get("budget_warning", 0) or 0) for row in rows)
        self.results[method] = result
        return result

    def run_span_methods(self) -> None:
        methods = self.requested_span_methods()
        self.prepare_span_selections(methods)
        for method in methods:
            self.run_span_method(method)

    def requested_span_methods(self) -> List[str]:
        default = [
            "span_drm_add_query",
            "span_drm_add_hybrid",
            "rh_then_span_drm_add_query",
            "rh_then_span_drm_add_hybrid",
            "span_drm_add_query_k1",
            "span_drm_add_query_k3",
            "span_drm_add_query_same_num_pages",
            "span_drm_add_query_count_compress",
            "span_drm_add_query_retrieval_compress",
        ]
        if self.args.run_oracle_diag:
            default.extend(["span_drm_add_oracle_diag", "rh_then_span_drm_add_oracle_diag"])
        requested = set(self.args.methods)
        if not requested:
            return default
        return [method for method in default if method in requested]

    def gate(self, method: str, result: Dict[str, Any]) -> Tuple[bool, List[str]]:
        if "oracle_diag" in method:
            return False, ["diagnostic_only"]
        if method == "fixed_drm_add":
            return False, ["baseline"]
        if "learned_pseudo_label" in METHOD_ATTR.get(method, ""):
            return False, ["learned_pseudo_label_reference_not_main"]
        reasons: List[str] = []
        baseline = self.results["fixed_drm_add"]
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
        if method == "span_drm_add_query_count_compress":
            reasons.append("count_compression_ablation_not_main")
        return not reasons, reasons

    def augment_page_budgets(self) -> None:
        for method, result in list(self.results.items()):
            if method in PAGE_METHOD_ATTR:
                result.update(self.page_budget_result(method))

    def write_outputs(self) -> None:
        self.augment_page_budgets()
        baseline = self.results["fixed_drm_add"]
        summary: List[Dict[str, Any]] = []
        gate_rows: List[Dict[str, Any]] = []
        for method, result in self.results.items():
            passed, reasons = self.gate(method, result)
            row: Dict[str, Any] = {
                "method": method,
                "attribute": METHOD_ATTR.get(method, ""),
                "avg": round(float(result["avg"]), 4),
                "delta_vs_baseline": round(float(result["avg"]) - float(baseline["avg"]), 4),
            }
            for task in self.args.tasks:
                row[task] = round(float(result["per_task"].get(task, 0.0)), 3)
            row.update({
                "budget_error_pct": round(float(result.get("budget_error", 0.0)) * 100, 4),
                "budget_warning": int(result.get("budget_warning", 0)),
                "runner_fail": int(result.get("runner_fail", 0)),
                "passed_gate": "yes" if passed else "no",
                "gate_reasons": "; ".join(reasons),
            })
            summary.append(row)
            gate_rows.append({
                "method": method,
                "deployable": "no" if (
                    "oracle_diag" in method
                    or method == "fixed_drm_add"
                    or "learned_pseudo_label" in row["attribute"]
                ) else "yes",
                "passed_gate": row["passed_gate"],
                "gate_reasons": row["gate_reasons"],
            })
        header = ["method", "attribute", "avg", "delta_vs_baseline"] + self.args.tasks + [
            "budget_error_pct", "budget_warning", "runner_fail", "passed_gate", "gate_reasons",
        ]
        write_csv(self.out / "structured_stage2_128_summary.csv", summary, header)
        ranking = [
            row for row in summary
            if row["method"] != "fixed_drm_add" and "oracle" not in row["method"] and "diagnostic_only" not in row["attribute"]
        ]
        ranking.sort(key=lambda row: float(row["avg"]), reverse=True)
        write_csv(self.out / "structured_stage2_128_ranking.csv", ranking, header)
        write_csv(self.out / "structured_stage2_128_method_gate.csv", gate_rows)
        self.write_budget_audit()
        self.write_task_deltas(summary)
        self.write_case_analysis()
        self.write_report(summary, ranking)
        self.write_full_prep(ranking)

    def write_budget_audit(self) -> None:
        rows: List[Dict[str, Any]] = []
        for method in self.results:
            if method in PAGE_METHOD_ATTR:
                for task in self.args.tasks:
                    records = latest_layer_records(audit_path(self.out, method, task))
                    baseline = latest_layer_records(audit_path(self.out, "fixed_drm_add", task))
                    errors = []
                    targets = []
                    for sid, row in records.items():
                        ref = float(baseline.get(sid, {}).get("selected_token_count", row.get("selected_token_count", 0)) or 0)
                        cur = float(row.get("selected_token_count", 0) or 0)
                        if ref > 0:
                            targets.append(ref)
                            errors.append(abs(cur - ref) / ref)
                    rows.append({
                        "method": method, "task": task, "n_samples": len(records),
                        "avg_target_tokens": round(mean(targets), 3),
                        "avg_selected_tokens": round(mean(float(row.get("selected_token_count", 0) or 0) for row in records.values()), 3),
                        "budget_error_pct": round(mean(errors) * 100, 4),
                        "budget_warning": 0,
                    })
            else:
                for task in self.args.tasks:
                    sels = [self.selections[(method, task, sid)] for sid in self.rh.sample_order[task]]
                    rows.append({
                        "method": method, "task": task, "n_samples": len(sels),
                        "avg_target_tokens": round(mean(row["target_final_tokens"] for row in sels), 3),
                        "avg_selected_tokens": round(mean(row["selected_token_count"] for row in sels), 3),
                        "budget_error_pct": round(mean(
                            abs(float(row["selected_token_count"]) - float(row["target_final_tokens"])) / max(1.0, float(row["target_final_tokens"]))
                            for row in sels
                        ) * 100, 4),
                        "budget_warning": sum(int(row.get("budget_warning", 0) or 0) for row in sels),
                    })
        write_csv(
            self.out / "structured_stage2_128_budget_audit.csv",
            rows,
            [
                "method", "task", "n_samples", "avg_target_tokens",
                "avg_selected_tokens", "budget_error_pct", "budget_warning",
            ],
        )

    def write_task_deltas(self, summary: List[Dict[str, Any]]) -> None:
        base = next(row for row in summary if row["method"] == "fixed_drm_add")
        rows: List[Dict[str, Any]] = []
        for row in summary:
            for task in self.args.tasks:
                rows.append({
                    "method": row["method"],
                    "task": task,
                    "score": row[task],
                    "fixed_drm_add": base[task],
                    "delta_vs_fixed": round(float(row[task]) - float(base[task]), 4),
                })
        write_csv(self.out / "structured_stage2_128_task_deltas.csv", rows)

    def write_case_analysis(self) -> None:
        rows: List[Dict[str, Any]] = []
        baseline_preds = {
            task: read_jsonl_index(pred_path(self.out, "fixed_drm_add", task))
            for task in self.args.tasks
        }
        for method in self.results:
            if method == "fixed_drm_add":
                continue
            for task in self.args.tasks:
                preds = read_jsonl_index(pred_path(self.out, method, task))
                for sid in self.rh.sample_order[task]:
                    answer = [self.rh.answers[(task, sid)]]
                    length = [self.rh.lengths[(task, sid)]]
                    classes = self.rh.all_classes.get(task, [])
                    base_score = float(scorer(task, [str(baseline_preds[task].get(sid, {}).get("pred", "") or "")], answer, length, classes))
                    method_score = float(scorer(task, [str(preds.get(sid, {}).get("pred", "") or "")], answer, length, classes))
                    sel = self.selections.get((method, task, sid), {})
                    rows.append({
                        "task": task, "sample_id": sid, "method": method,
                        "method_score": round(method_score, 4),
                        "fixed_drm_add_score": round(base_score, 4),
                        "delta": round(method_score - base_score, 4),
                        "fallback_used": sel.get("fallback_used", ""),
                        "source_method": sel.get("source_method", method if method in PAGE_METHOD_ATTR else ""),
                        "selected_token_count": sel.get("selected_token_count", ""),
                        "target_final_tokens": sel.get("target_final_tokens", ""),
                        "compression_ratio": sel.get("compression_ratio", ""),
                        "evidence_density": sel.get("evidence_density", ""),
                    })
        write_csv(self.out / "structured_stage2_128_case_analysis.csv", rows)

    def load_64_rows(self) -> Dict[str, Dict[str, Any]]:
        paths = [
            REPO_ROOT / "results" / "rh_drm_stage2_64_20260604_072019" / "rh_stage2_64_summary.csv",
            REPO_ROOT / "results" / "span_drm_stage2_64_20260604_181036" / "span_stage2_64_summary.csv",
            REPO_ROOT / "results" / "structured_drm_extensions_summary" / "structured_extensions_ranking.csv",
        ]
        aliases = {
            "rh_drm_add_static": "rh_drm_add_static_fixed_pool",
            "rh_drm_add_query": "rh_drm_add_query_fixed_pool",
        }
        rows: Dict[str, Dict[str, Any]] = {}
        for path in paths:
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    name = aliases.get(str(row.get("method", "")), str(row.get("method", "")))
                    rows[name] = row
        return rows

    def write_report(self, summary: List[Dict[str, Any]], ranking: List[Dict[str, Any]]) -> None:
        by_name = {row["method"]: row for row in summary}
        rows64 = self.load_64_rows()
        baseline = by_name["fixed_drm_add"]
        best = ranking[0] if ranking else baseline
        passed = [row["method"] for row in ranking if row.get("passed_gate") == "yes"]
        span_best = max(
            [row for row in ranking if row["method"] in {"span_drm_add_query", "span_drm_add_hybrid"}],
            key=lambda row: float(row["avg"]),
            default=None,
        )
        rh_then_best = max(
            [row for row in ranking if row["method"].startswith("rh_then_span")],
            key=lambda row: float(row["avg"]),
            default=None,
        )
        rh_best = max(
            [row for row in ranking if row["method"].startswith("rh_drm")],
            key=lambda row: float(row["avg"]),
            default=None,
        )
        is_full = self.args.evaluation_mode == "full"
        full_complete = is_full and all(int(row.get("runner_fail", 0) or 0) == 0 for row in summary)
        evaluation_label = "Full LongBench" if is_full else "Stage2 128-sample Recheck"
        comparison_label = "full" if is_full else "128"
        lines = [
            f"# Structured DRM-Add {evaluation_label}",
            "",
            "## Experiment Setup",
            "",
            f"- Tasks: {', '.join(self.args.tasks)}; samples per task: {self.args.samples_per_task}.",
            "- Active set is fixed SlimInfer top-K for every method.",
            "- Rescue is add-only; no swap and no active deletion.",
            "- RH page methods use fixed-pool only; union-pool is intentionally excluded.",
            "- Span methods use fixed 24-token windows packed in original token order with source metadata recorded per selected span.",
            "- Passage retrieval and passage count main methods use deterministic full-page fallback.",
            "- Environment: requested conda `silm` was unavailable; repo `.venv` was used.",
            "",
            f"## 64 vs {comparison_label} Trend",
            "",
        ]
        for method in [
            "rh_drm_add_static_fixed_pool",
            "rh_drm_add_query_fixed_pool",
            "span_drm_add_query",
            "span_drm_add_hybrid",
            "rh_then_span_drm_add_query",
            "rh_then_span_drm_add_hybrid",
        ]:
            if method not in by_name:
                continue
            old = rows64.get(method, {})
            old_avg = old.get("avg", "n/a")
            lines.append(
                f"- `{method}`: 64 Avg {old_avg}; {comparison_label} Avg {by_name[method]['avg']} "
                f"({by_name[method]['delta_vs_baseline']:+.4f} vs fixed)."
            )
        lines += [
            "",
            "## Results",
            "",
        ]
        for row in summary:
            lines.append(
                f"- `{row['method']}`: Avg {row['avg']}, delta {row['delta_vs_baseline']:+.4f}, "
                f"gate {row['passed_gate']}, budget error {row['budget_error_pct']}%"
            )
        k_methods = [name for name in ["span_drm_add_query_k1", "span_drm_add_query", "span_drm_add_query_k3"] if name in by_name]
        best_k = max(k_methods, key=lambda name: float(by_name[name]["avg"])) if k_methods else "n/a"
        refill = by_name.get("span_drm_add_query", {})
        same = by_name.get("span_drm_add_query_same_num_pages", {})
        count_compress = by_name.get("span_drm_add_query_count_compress", {})
        retrieval_compress = by_name.get("span_drm_add_query_retrieval_compress", {})
        lines += [
            "",
            "## Required Answers",
            "",
            f"- Best method: `{best['method']}` Avg {best['avg']}.",
            f"- k=1/2/3 most stable by Avg: `{best_k}`.",
            f"- budget_refill beats same_num_pages_compressed: {float(refill.get('avg', 0) or 0) > float(same.get('avg', 0) or 0)}.",
            f"- retrieval fallback necessary: {float(retrieval_compress.get('passage_retrieval_en', 0) or 0) < float(refill.get('passage_retrieval_en', 0) or 0)}.",
            f"- count fallback necessary: {float(count_compress.get('passage_count', 0) or 0) < float(refill.get('passage_count', 0) or 0)}.",
            "- SpanDRM evidence density is recorded in `structured_stage2_128_case_analysis.csv`; oracle-overlap is diagnostic only and not used by deployable scorers.",
            "- Leakage audit: see `results/structured_drm_recheck_audit/leakage_budget_audit.md`.",
            "",
            "## Promotion Gate",
            "",
            f"- Methods passing gate: {passed if passed else 'none'}.",
        ]
        if rh_then_best and rh_then_best.get("passed_gate") == "yes" and float(rh_then_best["avg"]) >= float(best["avg"]):
            recommendation = "Structured DRM-Add (RH-then-Span) is the new mainline candidate."
        elif span_best and span_best.get("passed_gate") == "yes" and float(span_best["avg"]) >= float(best["avg"]):
            recommendation = "SpanDRM-Add is the new mainline candidate."
        elif rh_best and rh_best.get("passed_gate") == "yes":
            recommendation = "RH-DRM-Add is the structural-upgrade candidate."
        else:
            recommendation = "Keep fixed DRM-Add as the mainline until larger evidence passes."
        lines += [
            f"- Recommendation: {recommendation}",
            f"- Full LongBench prep generated: {'yes' if passed else 'no passing deployable method; prep still records empty candidates'}.",
            "",
            "## Paper Contribution Language",
            "",
            "- fixed DRM-Add: damage-aware add-only rescue.",
            "- RH-DRM: retrieval-head-conditioned page rescue.",
            "- SpanDRM: evidence-density-aware span rescue.",
            "- RH-then-Span: head-aware page selection plus span-level evidence packing.",
            "",
            "## Scope",
            "",
            (
                f"- Full LongBench complete: {full_complete}. Latency was not run."
                if is_full else
                "- No full LongBench or latency run was launched."
            ),
        ]
        (self.out / "report_structured_stage2_128.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def write_full_prep(self, ranking: List[Dict[str, Any]]) -> None:
        out = REPO_ROOT / "results" / "structured_drm_full_prep"
        out.mkdir(parents=True, exist_ok=True)
        passing = [row for row in ranking if row.get("passed_gate") == "yes"]
        span = [
            row for row in passing
            if row["method"] in {"span_drm_add_query", "span_drm_add_hybrid"}
        ]
        rh_or_hybrid = [
            row for row in passing
            if row["method"].startswith("rh_then_span") or row["method"].startswith("rh_drm")
        ]
        candidates: List[str] = []
        if span:
            candidates.append(max(span, key=lambda row: float(row["avg"]))["method"])
        if rh_or_hybrid:
            best = max(rh_or_hybrid, key=lambda row: float(row["avg"]))["method"]
            if best not in candidates:
                candidates.append(best)
        candidates = candidates[:2]
        (out / "full_candidate_methods.txt").write_text("\n".join(candidates) + ("\n" if candidates else ""), encoding="utf-8")
        run_sh = [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            f"cd {REPO_ROOT}",
            "# Prepared only. Do not launch full LongBench without explicit confirmation.",
            f"METHODS={','.join(candidates)}",
            'echo "Candidate methods: ${METHODS}"',
            'echo "Confirm before running full LongBench; this prep script intentionally does not start a full run."',
        ]
        (out / "run_full_structured_drm.sh").write_text("\n".join(run_sh) + "\n", encoding="utf-8")
        (out / "resume_full_structured_drm.sh").write_text("\n".join(run_sh) + "\n", encoding="utf-8")
        for name in ["run_full_structured_drm.sh", "resume_full_structured_drm.sh"]:
            (out / name).chmod(0o755)
        (out / "full_budget_check_plan.md").write_text(
            "# Full Budget Check Plan\n\n"
            "- Verify fixed DRM-Add and each structured candidate share sample order and task set.\n"
            "- Confirm `budget_error < 1%`, `budget_warning = 0`, and `runner_fail = 0` before ranking.\n"
            "- Keep passage_count and passage_retrieval_en full-page fallback for SpanDRM main methods.\n"
            "- Exclude oracle diagnostics and large feature dumps from git.\n",
            encoding="utf-8",
        )
        (out / "full_expected_outputs.md").write_text(
            "# Full Expected Outputs\n\n"
            "- full structured summary/ranking CSVs\n"
            "- per-task budget audit\n"
            "- method gate table\n"
            "- final report comparing FullKV, SlimInfer, fixed DRM-Add, and at most two structured candidates\n",
            encoding="utf-8",
        )


def run(args: argparse.Namespace) -> Path:
    start = time.time()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    config = {
        "stage": "structured_drm_full_longbench" if args.evaluation_mode == "full" else "structured_stage2_128",
        "evaluation_mode": args.evaluation_mode,
        "tasks": args.tasks,
        "samples_per_task": args.samples_per_task,
        "active_construction": "fixed SlimInfer top-K",
        "rescue_policy": "add-only",
        "rh_candidate_pool": "fixed_pool_only",
        "span_tokens": args.span_tokens,
        "main_span_packing": "budget_refill_spans",
        "count_fallback": "predefined full-page fallback for main methods",
        "retrieval_fallback": "predefined full-page fallback for main methods",
        "audit_dir": args.audit_dir,
        "python_env": str(REPO_ROOT / ".venv"),
        "env_note": "conda env silm unavailable; repo .venv used",
    }
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    rerun_args = [
        "--output_dir", str(out),
        "--tasks", ",".join(args.tasks),
        "--samples_per_task", str(args.samples_per_task),
        "--evaluation_mode", args.evaluation_mode,
        "--skip_existing",
    ]
    if args.methods:
        rerun_args.extend(["--methods", ",".join(args.methods)])
    if args.run_oracle_diag:
        rerun_args.append("--run_oracle_diag")
    if args.include_fup_reference:
        rerun_args.append("--include_fup_reference")
    run_sh = (
        "#!/usr/bin/env bash\nset -e\n"
        f"cd {REPO_ROOT}\n"
        "scripts/run_with_silm.sh tools/run_structured_drm_stage2_128.py "
        + " ".join(rerun_args)
        + "\n"
    )
    (out / "run_structured_stage2_128.sh").write_text(run_sh, encoding="utf-8")
    (out / "resume.sh").write_text(run_sh.replace("--skip_existing", "--skip_existing --overwrite_failed"), encoding="utf-8")
    for name in ["run_structured_stage2_128.sh", "resume.sh"]:
        (out / name).chmod(0o755)
    drv = StructuredDriver(args)
    drv.load_contexts()
    drv.ensure_page_methods()
    drv.run_span_methods()
    drv.write_outputs()
    if args.evaluation_mode == "full":
        full_complete = all(
            int(result.get("runner_fail", 0) or 0) == 0
            for result in drv.results.values()
        )
        (out / "full_run_status.json").write_text(
            json.dumps({
                "evaluation_mode": "full",
                "full_complete": full_complete,
                "runner_fail_total": sum(
                    int(result.get("runner_fail", 0) or 0)
                    for result in drv.results.values()
                ),
                "task_sample_counts": {
                    task: len(drv.rh.sample_order.get(task, []))
                    for task in args.tasks
                },
                "methods_evaluated": list(drv.results),
            }, indent=2) + "\n",
            encoding="utf-8",
        )
        aliases = {
            "structured_stage2_128_summary.csv": "structured_full_summary.csv",
            "structured_stage2_128_ranking.csv": "structured_full_ranking.csv",
            "structured_stage2_128_budget_audit.csv": "structured_full_budget_audit.csv",
            "structured_stage2_128_task_deltas.csv": "structured_full_task_deltas.csv",
            "structured_stage2_128_method_gate.csv": "structured_full_method_gate.csv",
            "report_structured_stage2_128.md": "report_structured_full.md",
        }
        for source_name, alias_name in aliases.items():
            source = out / source_name
            if source.exists():
                shutil.copy2(source, out / alias_name)
    drv.log(f"DONE {(time.time() - start) / 60:.1f} min -> {out}")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument("--samples_per_task", type=int, default=128)
    parser.add_argument("--evaluation_mode", choices=["diagnostic", "full"], default="diagnostic")
    parser.add_argument("--methods", default="")
    parser.add_argument("--run_oracle_diag", action="store_true")
    parser.add_argument("--include_fup_reference", action="store_true")
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--model_path", default=DEFAULT_MODEL)
    parser.add_argument("--longbench_dir", default=DEFAULT_LB)
    parser.add_argument("--profile_dir", default=str(DEFAULT_PROFILE))
    parser.add_argument("--fup_dir", default=str(DEFAULT_FUP))
    parser.add_argument("--audit_dir", default=str(DEFAULT_AUDIT))
    parser.add_argument("--reuse_full_baseline_dir", default=str(DEFAULT_FULL_BASELINE))
    parser.add_argument("--source_audit_only", action="store_true")
    parser.add_argument("--page_features", default=str(DEFAULT_FUP / "features_fup_candidates.csv"))
    parser.add_argument("--fup_scores", default=str(DEFAULT_FUP / "fup_output_delta_scores.csv"))
    parser.add_argument("--span_tokens", type=int, default=24)
    parser.add_argument("--rh_alpha_drm", type=float, default=1.0)
    parser.add_argument("--rh_beta_retrieval", type=float, default=0.8)
    parser.add_argument("--rh_beta_query", type=float, default=0.6)
    parser.add_argument("--rh_beta_reasoning", type=float, default=0.4)
    parser.add_argument("--rh_lambda_head", type=float, default=0.25)
    parser.add_argument("--rh_lambda_page", type=float, default=0.25)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--overwrite_failed", action="store_true")
    args = parser.parse_args()
    args.tasks = [task for task in args.tasks.split(",") if task]
    args.methods = [method for method in args.methods.split(",") if method]
    if not args.output_dir:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = str(REPO_ROOT / "results" / f"structured_drm_stage2_128_{stamp}")
    return args


if __name__ == "__main__":
    run(parse_args())
