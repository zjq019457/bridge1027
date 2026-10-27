#!/usr/bin/env python3
"""Shared utilities for paper-facing SpanDRM validation analyses."""
from __future__ import annotations

import csv
import json
import math
import re
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from longbench_eval import dataset2metric  # noqa: E402
from tools.build_span_candidates import PAGE_TOKENS, fixed_page_sets_from_audit  # noqa: E402

TASKS = [
    "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique",
    "gov_report", "qmsum", "multi_news", "trec", "triviaqa", "samsum",
    "passage_count", "passage_retrieval_en", "lcc", "repobench-p", "vcsum", "lsht",
]
SANITY_TASKS = ["musique", "qasper", "hotpotqa", "passage_count", "passage_retrieval_en"]
CATEGORIES = {
    "SingleDoc QA": ["qasper", "multifieldqa_en", "hotpotqa"],
    "MultiDoc QA": ["2wikimqa", "musique"],
    "Summarization": ["gov_report", "qmsum", "multi_news"],
    "Few-shot / Classification": ["trec", "triviaqa", "samsum"],
    "Synthetic / Retrieval": ["passage_count", "passage_retrieval_en"],
    "Code": ["lcc", "repobench-p"],
    "Chinese": ["vcsum", "lsht"],
}
TASK_TO_CATEGORY = {task: category for category, tasks in CATEGORIES.items() for task in tasks}

MODEL_PATH = Path(
    "/home/yu-bai/.cache/huggingface/hub/"
    "models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
)
LONGBENCH_DIR = Path("/home/yu-bai/Desktop/jiaqizhang/HeadKV/data/LongBench")
PAPER_ROOT = REPO_ROOT / "results" / "paper_final_main_table_20260608_194914"
STRUCTURED_ROOT = REPO_ROOT / "results" / "structured_drm_full_longbench_20260608_194914"
BASELINE_ROOT = REPO_ROOT / "results" / "rescuekv_drm_add_full_longbench_20260523_233633"

METHOD_SOURCES = {
    "FullKV": BASELINE_ROOT / "raw_predictions" / "FullKV",
    "SlimInfer original": BASELINE_ROOT / "raw_predictions" / "sliminfer_original",
    "SlimInfer larger K+M": BASELINE_ROOT / "raw_predictions" / "sliminfer_larger",
    "Random-Add": BASELINE_ROOT / "raw_predictions" / "random_add_seed0",
    "DeltaO-Add": BASELINE_ROOT / "raw_predictions" / "delta_o_add",
    "fixed DRM-Add": BASELINE_ROOT / "raw_predictions" / "drm_add",
    "SpanDRM": STRUCTURED_ROOT / "predictions" / "span_drm_add_query",
    "RH-then-Span": STRUCTURED_ROOT / "predictions" / "rh_then_span_drm_add_query",
}

_WORD_RE = re.compile(r"[A-Za-z0-9_\-\u4e00-\u9fff]+")


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fields: Optional[Sequence[str]] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def markdown_table(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> str:
    def clean(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    lines.extend("| " + " | ".join(clean(row.get(field, "")) for field in fields) + " |" for row in rows)
    return "\n".join(lines)


def mean(values: Iterable[float]) -> float:
    vals = [float(value) for value in values]
    return sum(vals) / len(vals) if vals else 0.0


def latest_layer_records(path: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in read_jsonl(path):
        sid = str(row.get("sample_id", ""))
        old = out.get(sid)
        if not sid:
            continue
        if old is None or int(row.get("layer_id", -1) or -1) >= int(old.get("layer_id", -1) or -1):
            out[sid] = row
    return out


def task_run_dir(task: str) -> Path:
    return STRUCTURED_ROOT / "task_runs" / task


@lru_cache(maxsize=None)
def span_metadata(task: str) -> Dict[str, Dict[str, Any]]:
    path = task_run_dir(task) / "_span_selected_contexts_structured.jsonl"
    return {
        str(row.get("sample_id", "")): row
        for row in read_jsonl(path)
        if row.get("method") == "span_drm_add_query"
    }


@lru_cache(maxsize=None)
def fixed_audit(task: str) -> Dict[str, Dict[str, Any]]:
    path = task_run_dir(task) / "preds" / f"fixed_drm_add__{task}__audit.jsonl"
    return latest_layer_records(path)


def page_token_indices(pages: Iterable[int], context_length: int) -> Set[int]:
    selected: Set[int] = set()
    for page in pages:
        start = max(0, int(page) * PAGE_TOKENS)
        selected.update(range(start, min(context_length, start + PAGE_TOKENS)))
    return selected


def reconstruct_span_selection(
    task: str,
    sample_id: str,
    metadata: Optional[Mapping[str, Any]] = None,
    audit: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    metadata = dict(metadata or span_metadata(task).get(str(sample_id), {}))
    audit = dict(audit or fixed_audit(task).get(str(sample_id), {}))
    context_length = int(audit.get("context_length", 0) or 0)
    target = int(metadata.get("selected_token_count", 0) or 0)
    if context_length <= 0:
        context_length = max(
            [int(span.get("selected_token_end", 0) or 0) for span in metadata.get("selected_spans", [])] + [target]
        )

    final_pages, active_pages, _ = fixed_page_sets_from_audit(audit) if audit else (set(), set(), set())
    fallback = bool(metadata.get("fallback_used", False))
    if fallback:
        selected = page_token_indices(final_pages, context_length)
        if not selected:
            selected = set(range(context_length))
    else:
        selected = page_token_indices(active_pages, context_length)
        for span in metadata.get("selected_spans", []):
            start = int(span.get("selected_token_start", span.get("token_start", 0)) or 0)
            end = int(span.get("selected_token_end", span.get("token_end", start)) or start)
            selected.update(range(max(0, start), min(context_length, end)))
    indices = sorted(selected)
    return {
        "task": task,
        "sample_id": str(sample_id),
        "context_length": context_length,
        "target_final_tokens": target,
        "selected_token_count": len(indices),
        "selected_abs_indices": indices,
        "selected_spans": list(metadata.get("selected_spans", [])),
        "fallback_used": fallback,
        "fallback_reason": str(metadata.get("fallback_reason", "") or ""),
        "active_pages": sorted(active_pages),
        "final_pages": sorted(final_pages),
        "budget_consistent": bool(target > 0 and len(indices) == target),
    }


def prediction_rows(method: str, task: str) -> List[Dict[str, Any]]:
    return read_jsonl(METHOD_SOURCES[method] / f"{task}.jsonl")


@lru_cache(maxsize=None)
def prediction_index(method: str, task: str) -> Dict[str, Dict[str, Any]]:
    return {str(row.get("sample_id", "")): row for row in prediction_rows(method, task)}


def score_prediction(task: str, prediction: str, answers: Sequence[Any], all_classes: Any = None) -> float:
    pred = str(prediction or "")
    if task in {"trec", "triviaqa", "samsum", "lsht"}:
        pred = pred.lstrip("\n").split("\n")[0]
    score = 0.0
    if pred:
        for answer in answers or []:
            score = max(score, float(dataset2metric[task](pred, answer, all_classes=all_classes)))
    return 100.0 * score


def load_scored_predictions(methods: Sequence[str]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Return method -> task -> sample_id -> prediction row with per-sample score."""
    result: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for task in TASKS:
        reference = prediction_index("fixed DRM-Add", task)
        for method in methods:
            rows = prediction_index(method, task)
            enriched: Dict[str, Dict[str, Any]] = {}
            for sid, row in rows.items():
                ref = reference.get(sid, {})
                answers = row.get("answers", ref.get("answers", []))
                classes = row.get("all_classes", ref.get("all_classes"))
                item = dict(row)
                item["answers"] = answers
                item["all_classes"] = classes
                item["length"] = row.get("length", ref.get("length", 0))
                item["score"] = score_prediction(task, str(row.get("pred", "")), answers, classes)
                enriched[sid] = item
            result[method][task] = enriched
    return result


@lru_cache(maxsize=None)
def load_longbench_index(task: str) -> Dict[str, Dict[str, Any]]:
    rows = read_jsonl(LONGBENCH_DIR / f"{task}.jsonl")
    out: Dict[str, Dict[str, Any]] = {}
    for index, row in enumerate(rows):
        sid = str(row.get("_id", row.get("id", index)))
        out[sid] = row
    return out


def normalized_words(text: str) -> Set[str]:
    return {word.lower() for word in _WORD_RE.findall(str(text or "")) if len(word) > 1}


def answer_overlap(text: str, answers: Sequence[Any]) -> float:
    source = normalized_words(text)
    answer = normalized_words(" ".join(str(item) for item in answers or []))
    return len(source & answer) / max(1, len(answer)) if answer else 0.0


def percentile_interval(values: Any) -> Tuple[float, float]:
    import numpy as np

    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def significant_positive(low: float) -> bool:
    return bool(math.isfinite(float(low)) and float(low) > 0.0)
