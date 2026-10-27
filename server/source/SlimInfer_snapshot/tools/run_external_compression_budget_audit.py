#!/usr/bin/env python3
"""Audit external prompt-compression baselines under an exact Qwen budget.

The script is packing-only: it never loads a QA model and never generates an
answer.  Official LongLLMLingua is run only when an explicit, local, small
compressor checkpoint passes the CPU safety gate.  Otherwise it is recorded as
unavailable rather than silently replaced.  The two transparent extractive
diagnostics select sentence/token spans without gold answers and hard-repack
with the exact tokenizer used by the official BridgePack Qwen run.
"""

from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import hashlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE = (
    REPO_ROOT / "results" / "bridgepack_exact_qwen_budget_1024_20260619_181059"
)
DEFAULT_TASKS = (
    "qasper",
    "multifieldqa_en",
    "hotpotqa",
    "2wikimqa",
    "musique",
    "narrativeqa",
)
EXPECTED_COUNTS = {
    "qasper": 200,
    "multifieldqa_en": 150,
    "hotpotqa": 200,
    "2wikimqa": 200,
    "musique": 200,
    "narrativeqa": 200,
}
REFERENCE_VARIANT = "bridge_neighbor32_path_order"
STYLE_METHODS = (
    "llmlingua_style_exact_query_aware",
    "llmlingua_style_exact_query_agnostic",
)
OFFICIAL_METHOD = "official_longllmlingua_if_available"
ALL_METHODS = (OFFICIAL_METHOD, *STYLE_METHODS)
WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-']*")
ENTITY_RE = re.compile(
    r"\b(?:[A-Z][A-Za-z0-9_\-']*(?:\s+[A-Z][A-Za-z0-9_\-']*)*|"
    r"\d{1,4}(?:[-/]\d{1,2}){0,2})\b"
)
SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])(?:[\"')\]]*)\s+|\n+")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "by", "did",
    "do", "does", "for", "from", "had", "has", "have", "he", "her", "hers",
    "him", "his", "how", "i", "in", "is", "it", "its", "me", "my", "of",
    "on", "or", "our", "ours", "she", "that", "the", "their", "theirs",
    "them", "they", "this", "those", "to", "us", "was", "we", "were", "what",
    "when", "where", "which", "who", "why", "with", "you", "your",
}


class AuditError(RuntimeError):
    pass


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_csv(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise AuditError(f"invalid JSONL {path}:{line_number}: {exc}") from exc
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def write_csv(
    path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(fields),
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def directory_apparent_size(path: Path) -> int:
    """Sum resolved file sizes once, including files behind snapshot symlinks."""
    seen: Set[Tuple[int, int]] = set()
    total = 0
    for candidate in path.rglob("*"):
        try:
            stat = candidate.stat()
        except OSError:
            continue
        if not candidate.is_file():
            continue
        key = (stat.st_dev, stat.st_ino)
        if key in seen:
            continue
        seen.add(key)
        total += stat.st_size
    return total


def token_count(tokenizer: Any, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False).input_ids)


def exact_prefix(tokenizer: Any, text: str, max_tokens: int) -> str:
    ids = tokenizer(text, add_special_tokens=False).input_ids[:max_tokens]
    return tokenizer.decode(
        ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()


def terms(text: str) -> Set[str]:
    return {
        match.group(0).lower()
        for match in WORD_RE.finditer(text or "")
        if len(match.group(0)) > 1 and match.group(0).lower() not in STOPWORDS
    }


def term_counts(text: str) -> collections.Counter[str]:
    return collections.Counter(
        match.group(0).lower()
        for match in WORD_RE.finditer(text or "")
        if len(match.group(0)) > 1 and match.group(0).lower() not in STOPWORDS
    )


def cosine(
    left: Mapping[str, float],
    right: Mapping[str, float],
) -> float:
    if not left or not right:
        return 0.0
    common = set(left) & set(right)
    numerator = sum(left[key] * right[key] for key in common)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)


def split_spans(tokenizer: Any, context: str, max_span_tokens: int) -> List[Dict[str, Any]]:
    pieces = [piece.strip() for piece in SENTENCE_BOUNDARY_RE.split(context) if piece.strip()]
    spans: List[Dict[str, Any]] = []
    for sentence_index, piece in enumerate(pieces):
        ids = tokenizer(piece, add_special_tokens=False).input_ids
        if not ids:
            continue
        for offset in range(0, len(ids), max_span_tokens):
            chunk_ids = ids[offset : offset + max_span_tokens]
            text = tokenizer.decode(
                chunk_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()
            if not text:
                continue
            spans.append(
                {
                    "span_index": len(spans),
                    "sentence_index": sentence_index,
                    "token_window_index": offset // max_span_tokens,
                    "text": text,
                    "model_tokens": len(chunk_ids),
                    "terms": terms(text),
                    "term_counts": term_counts(text),
                }
            )
    return spans


def tfidf_vectors(
    spans: Sequence[Mapping[str, Any]],
) -> Tuple[List[Dict[str, float]], Dict[str, float]]:
    document_count = max(1, len(spans))
    document_frequency: collections.Counter[str] = collections.Counter()
    for span in spans:
        document_frequency.update(span["terms"])
    idf = {
        term: math.log((document_count + 1.0) / (frequency + 1.0)) + 1.0
        for term, frequency in document_frequency.items()
    }
    vectors: List[Dict[str, float]] = []
    for span in spans:
        counts = span["term_counts"]
        total = max(1, sum(counts.values()))
        vectors.append(
            {term: (count / total) * idf[term] for term, count in counts.items()}
        )
    centroid: collections.defaultdict[str, float] = collections.defaultdict(float)
    for vector in vectors:
        for term, value in vector.items():
            centroid[term] += value / document_count
    return vectors, dict(centroid)


def score_spans(
    spans: Sequence[Mapping[str, Any]],
    question: str,
    method: str,
) -> List[float]:
    vectors, centroid = tfidf_vectors(spans)
    query_terms = terms(question)
    query_entities = {
        item.lower() for item in ENTITY_RE.findall(question or "") if item.strip()
    }
    document_frequency: collections.Counter[str] = collections.Counter()
    for span in spans:
        document_frequency.update(span["terms"])
    n = max(1, len(spans))
    scores: List[float] = []
    for index, (span, vector) in enumerate(zip(spans, vectors)):
        internal_centrality = cosine(vector, centroid)
        position = 1.0 / math.sqrt(index + 1.0)
        if method == "llmlingua_style_exact_query_aware":
            overlap = query_terms & span["terms"]
            query_idf = sum(
                math.log((n + 1.0) / (document_frequency[term] + 1.0)) + 1.0
                for term in overlap
            )
            normalized_overlap = query_idf / math.sqrt(max(1, len(span["terms"])))
            lowered = str(span["text"]).lower()
            entity_hits = sum(entity in lowered for entity in query_entities)
            number_hits = sum(
                term.isdigit() and term in span["terms"] for term in query_terms
            )
            score = (
                3.0 * normalized_overlap
                + 0.8 * entity_hits
                + 0.5 * number_hits
                + 0.25 * internal_centrality
                + 0.05 * position
            )
        elif method == "llmlingua_style_exact_query_agnostic":
            length_quality = min(1.0, len(span["terms"]) / 24.0)
            score = 1.0 * internal_centrality + 0.15 * length_quality + 0.10 * position
        else:
            raise AuditError(f"unsupported style method: {method}")
        scores.append(float(score))
    return scores


def jaccard(left: Set[str], right: Set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def mmr_order(
    spans: Sequence[Mapping[str, Any]],
    base_scores: Sequence[float],
    redundancy_weight: float,
) -> List[int]:
    remaining = set(range(len(spans)))
    selected: List[int] = []
    max_redundancy = [0.0] * len(spans)
    while remaining:
        best_index = min(
            remaining,
            key=lambda index: (
                -(base_scores[index] - redundancy_weight * max_redundancy[index]),
                index,
            ),
        )
        selected.append(best_index)
        remaining.remove(best_index)
        chosen_terms = spans[best_index]["terms"]
        for index in remaining:
            max_redundancy[index] = max(
                max_redundancy[index],
                jaccard(spans[index]["terms"], chosen_terms),
            )
    return selected


def format_packed(question: str, evidence_parts: Sequence[str]) -> str:
    return (
        f"Question: {question}\n\nSelected compressed evidence:\n\n"
        + "\n\n".join(part.strip() for part in evidence_parts if part.strip())
    ).strip()


def exact_style_pack(
    tokenizer: Any,
    context: str,
    question: str,
    method: str,
    budget: int,
    max_span_tokens: int,
    max_mmr_candidates: int,
) -> Dict[str, Any]:
    spans = split_spans(tokenizer, context, max_span_tokens)
    if not spans:
        raise AuditError("source context produced no non-empty sentence/span candidates")
    base_scores = score_spans(spans, question, method)
    pool = sorted(
        range(len(spans)),
        key=lambda index: (-base_scores[index], index),
    )[:max_mmr_candidates]
    pool_order = mmr_order(
        [spans[index] for index in pool],
        [base_scores[index] for index in pool],
        redundancy_weight=0.30,
    )
    order = [pool[index] for index in pool_order]
    selected: List[int] = []
    evidence: List[str] = []
    for index in order:
        candidate_selected = sorted([*selected, index])
        candidate_evidence = [str(spans[item]["text"]) for item in candidate_selected]
        candidate_text = format_packed(question, candidate_evidence)
        if token_count(tokenizer, candidate_text) <= budget:
            selected.append(index)
            evidence = [str(spans[item]["text"]) for item in sorted(selected)]

    # Fill any meaningful residual budget with a partial span.  This remains
    # extractive after tokenizer decode and is explicitly marked as truncated.
    truncated_span_index: Optional[int] = None
    current_text = format_packed(question, evidence)
    current_tokens = token_count(tokenizer, current_text)
    for index in order:
        if index in selected:
            continue
        remaining = budget - current_tokens
        if remaining < 8:
            break
        candidate_ids = tokenizer(
            str(spans[index]["text"]), add_special_tokens=False
        ).input_ids
        low, high = 1, min(len(candidate_ids), remaining)
        best_piece = ""
        while low <= high:
            middle = (low + high) // 2
            piece = tokenizer.decode(
                candidate_ids[:middle],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()
            candidate = format_packed(question, [*evidence, piece])
            if piece and token_count(tokenizer, candidate) <= budget:
                best_piece = piece
                low = middle + 1
            else:
                high = middle - 1
        if best_piece:
            evidence.append(best_piece)
            truncated_span_index = index
            current_text = format_packed(question, evidence)
            current_tokens = token_count(tokenizer, current_text)
        break

    if not evidence:
        # The question/header can consume part of the budget; retain an exact
        # prefix of the highest-ranked source span as a fail-safe.
        header = format_packed(question, [])
        allowance = max(1, budget - token_count(tokenizer, header) - 2)
        piece = exact_prefix(tokenizer, str(spans[order[0]]["text"]), allowance)
        evidence = [piece] if piece else []
        truncated_span_index = order[0]
    packed_text = format_packed(question, evidence)
    while evidence and token_count(tokenizer, packed_text) > budget:
        last_ids = tokenizer(evidence[-1], add_special_tokens=False).input_ids
        if len(last_ids) <= 1:
            evidence.pop()
        else:
            evidence[-1] = tokenizer.decode(
                last_ids[:-1],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()
        packed_text = format_packed(question, evidence)
    compressed_tokens = token_count(tokenizer, packed_text)
    if compressed_tokens > budget:
        raise AuditError(
            f"exact repack failed: compressed={compressed_tokens}, budget={budget}"
        )
    if not any(part.strip() for part in evidence):
        raise AuditError("exact repack produced empty evidence context")

    selected_records = []
    for output_position, index in enumerate(sorted(selected)):
        selected_records.append(
            {
                "output_position": output_position,
                "span_index": int(spans[index]["span_index"]),
                "sentence_index": int(spans[index]["sentence_index"]),
                "token_window_index": int(spans[index]["token_window_index"]),
                "score": base_scores[index],
                "model_tokens": int(spans[index]["model_tokens"]),
                "truncated": False,
                "text": str(spans[index]["text"]),
            }
        )
    if truncated_span_index is not None:
        selected_records.append(
            {
                "output_position": len(selected_records),
                "span_index": int(spans[truncated_span_index]["span_index"]),
                "sentence_index": int(spans[truncated_span_index]["sentence_index"]),
                "token_window_index": int(
                    spans[truncated_span_index]["token_window_index"]
                ),
                "score": base_scores[truncated_span_index],
                "model_tokens": token_count(tokenizer, evidence[-1]),
                "truncated": True,
                "text": evidence[-1],
            }
        )
    return {
        "packed_text": packed_text,
        "compressed_model_tokens": compressed_tokens,
        "selected_sentences": sorted(
            {int(record["sentence_index"]) for record in selected_records}
        ),
        "selected_spans": selected_records,
        "candidate_sentence_count": len({span["sentence_index"] for span in spans}),
        "candidate_span_count": len(spans),
        "ranking_candidate_span_count": len(pool),
        "truncated_final_span": truncated_span_index is not None,
    }


def generation_prompt(task: str, packed_text: str, question: str) -> str:
    templates = read_json(REPO_ROOT / "configs" / "dataset2prompt.json")
    template = str(templates.get(task, ""))
    if not template:
        raise AuditError(f"missing generation prompt template for {task}")
    try:
        return template.format(
            context=packed_text,
            input=question,
            question=question,
        )
    except Exception as exc:
        raise AuditError(f"failed to apply generation prompt for {task}: {exc}") from exc


def resolve_reference_rows(
    reference_dir: Path, task: str, max_samples: int
) -> List[Dict[str, Any]]:
    path = (
        reference_dir
        / "packed_prompts_build"
        / "packed_prompts"
        / REFERENCE_VARIANT
        / "budget_1024"
        / f"{task}.jsonl"
    )
    if not path.exists():
        raise AuditError(f"official BridgePack reference rows missing: {path}")
    rows = read_jsonl(path)
    expected = EXPECTED_COUNTS[task]
    if len(rows) != expected:
        raise AuditError(f"{task}: official rows={len(rows)}, expected={expected}")
    ids = [str(row.get("sample_id", "")) for row in rows]
    if not all(ids) or len(set(ids)) != len(ids):
        raise AuditError(f"{task}: missing or duplicate official sample IDs")
    for index, row in enumerate(rows):
        if str(row.get("task")) != task:
            raise AuditError(f"{task}/{index}: task mismatch in official manifest")
        if int(row.get("budget", -1)) != 1024:
            raise AuditError(f"{task}/{index}: official reference budget is not 1024")
        if not str(row.get("original_context", "")).strip():
            raise AuditError(f"{task}/{index}: official reference lacks original_context")
        if not str(row.get("question", "")).strip():
            raise AuditError(f"{task}/{index}: official reference lacks question")
    return rows[:max_samples] if max_samples > 0 else rows


def find_default_official_model() -> Optional[Path]:
    root = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / "models--NousResearch--Llama-2-7b-hf"
        / "snapshots"
    )
    snapshots = sorted(root.glob("*")) if root.exists() else []
    return snapshots[-1] if snapshots else None


def official_readiness(args: argparse.Namespace) -> Dict[str, Any]:
    source = REPO_ROOT / "LLMLingua" / "llmlingua" / "prompt_compressor.py"
    model_path = args.official_compressor_model or find_default_official_model()
    result: Dict[str, Any] = {
        "official_source_path": str(source.resolve()) if source.exists() else "",
        "official_source_available": source.exists(),
        "official_source_sha256": sha256_file(source) if source.exists() else "",
        "official_model_path": str(model_path.resolve()) if model_path else "",
        "official_model_local": bool(model_path and model_path.exists()),
        "official_model_apparent_gb": None,
        "cpu_model_size_limit_gb": args.max_official_cpu_model_gb,
        "cpu_only": args.cpu_only,
        "eligible": False,
        "status": "unavailable_or_audit_failed",
        "reason": "",
    }
    if not source.exists():
        result["reason"] = "local LLMLingua source is missing"
        return result
    if not model_path or not model_path.exists():
        result["reason"] = (
            "no explicit local official compressor checkpoint; network download is forbidden"
        )
        return result
    size_gb = directory_apparent_size(model_path) / 1024**3
    result["official_model_apparent_gb"] = size_gb
    if args.cpu_only and size_gb > args.max_official_cpu_model_gb:
        result["reason"] = (
            f"local official default compressor is {size_gb:.2f} GiB, above the "
            f"{args.max_official_cpu_model_gb:.2f} GiB CPU safety limit; "
            "the 7B compressor was not loaded"
        )
        return result
    result["eligible"] = True
    result["status"] = "eligible_local_checkpoint"
    return result


def load_official_compressor(readiness: Mapping[str, Any]) -> Any:
    llmlingua_root = REPO_ROOT / "LLMLingua"
    if str(llmlingua_root) not in sys.path:
        sys.path.insert(0, str(llmlingua_root))
    from llmlingua import PromptCompressor

    return PromptCompressor(
        model_name=str(readiness["official_model_path"]),
        device_map="cpu",
    )


def official_compress(
    compressor: Any,
    tokenizer: Any,
    context: str,
    question: str,
    budget: int,
) -> Dict[str, Any]:
    header = format_packed(question, [])
    context_target = max(32, budget - token_count(tokenizer, header) - 4)
    response = compressor.compress_prompt(
        [context],
        question=question,
        target_token=context_target,
        use_sentence_level_filter=True,
        use_context_level_filter=True,
        use_token_level_filter=True,
        condition_in_question="after_condition",
        reorder_context="sort",
        dynamic_context_compression_ratio=0.4,
        condition_compare=True,
        context_budget="+100",
        rank_method="longllmlingua",
        concate_question=False,
    )
    compressed_context = str(response.get("compressed_prompt", "")).strip()
    packed = format_packed(question, [compressed_context] if compressed_context else [])
    return {
        "packed_text": packed,
        "compressed_model_tokens": token_count(tokenizer, packed),
        "official_response_metadata": {
            key: response.get(key)
            for key in ("origin_tokens", "compressed_tokens", "ratio", "rate", "saving")
        },
    }


def summarize_audits(
    rows: Sequence[Mapping[str, Any]],
    methods: Sequence[str],
    tasks: Sequence[str],
    budget: int,
) -> Dict[str, Any]:
    cells: Dict[str, Any] = {}
    for method in methods:
        for task in tasks:
            cell = [
                row
                for row in rows
                if row["method_name"] == method and row["task"] == task
            ]
            successful = [row for row in cell if not row.get("error_message")]
            times = [float(row["compression_time_ms"]) for row in successful]
            token_counts = [int(row["compressed_model_tokens"]) for row in successful]
            ratios = [float(row["compression_ratio"]) for row in successful]
            cells[f"{method}/{task}"] = {
                "requested": len(cell),
                "successful": len(successful),
                "failed": len(cell) - len(successful),
                "over_budget": sum(bool(row.get("over_budget")) for row in cell),
                "empty_context": sum(bool(row.get("empty_context")) for row in cell),
                "compressed_tokens_mean": (
                    sum(token_counts) / len(token_counts) if token_counts else None
                ),
                "compressed_tokens_max": max(token_counts) if token_counts else None,
                "compression_ratio_mean": (
                    sum(ratios) / len(ratios) if ratios else None
                ),
                "compression_time_ms_mean": (
                    sum(times) / len(times) if times else None
                ),
                "compression_time_ms_total": sum(times),
            }
    style_rows = [row for row in rows if row["method_name"] in STYLE_METHODS]
    style_success = [row for row in style_rows if not row.get("error_message")]
    return {
        "status": (
            "PASS"
            if style_rows
            and len(style_success) == len(style_rows)
            and not any(row.get("over_budget") for row in style_rows)
            and not any(row.get("empty_context") for row in style_rows)
            else "FAIL"
        ),
        "budget": budget,
        "methods": list(methods),
        "tasks": list(tasks),
        "audit_rows": len(rows),
        "style_requested": len(style_rows),
        "style_successful": len(style_success),
        "style_over_budget": sum(bool(row.get("over_budget")) for row in style_rows),
        "style_empty_context": sum(bool(row.get("empty_context")) for row in style_rows),
        "style_errors": sum(bool(row.get("error_message")) for row in style_rows),
        "style_compression_time_ms_total": sum(
            float(row["compression_time_ms"]) for row in style_success
        ),
        "cells": cells,
    }


def parse_args() -> argparse.Namespace:
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, default=REPO_ROOT / "results" / f"bridgepack_external_compression_audit_{stamp}")
    parser.add_argument("--reference_dir", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    parser.add_argument("--max_samples_per_task", type=int, default=10)
    parser.add_argument("--budget", type=int, default=1024)
    parser.add_argument("--methods", default=",".join(ALL_METHODS))
    parser.add_argument("--tokenizer", choices=("qwen",), default="qwen")
    parser.add_argument("--cpu_only", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_span_tokens", type=int, default=160)
    parser.add_argument("--max_mmr_candidates", type=int, default=256)
    parser.add_argument("--official_compressor_model", type=Path)
    parser.add_argument("--max_official_cpu_model_gb", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = parse_csv(args.tasks)
    methods = parse_csv(args.methods)
    if tuple(tasks) != DEFAULT_TASKS:
        raise AuditError(f"tasks must exactly match official protocol: {DEFAULT_TASKS}")
    unknown = sorted(set(methods) - set(ALL_METHODS))
    if unknown:
        raise AuditError(f"unknown methods: {unknown}")
    if args.budget != 1024:
        raise AuditError("this external baseline audit is locked to official budget=1024")
    if args.seed != 0:
        raise AuditError("only deterministic seed=0 is supported")
    if not args.cpu_only:
        raise AuditError("--cpu_only is required for packing audit")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise AuditError(f"refusing to overwrite non-empty output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    reference_config = read_json(args.reference_dir / "config.json")
    if int(reference_config.get("budget", -1)) != args.budget:
        raise AuditError("reference result budget does not match requested budget")
    if tuple(reference_config.get("tasks", [])) != DEFAULT_TASKS:
        raise AuditError("reference task list/order does not match required protocol")
    tokenizer_path = Path(str(reference_config.get("model_name_or_path", "")))
    if not tokenizer_path.exists():
        raise AuditError(f"official tokenizer path is unavailable: {tokenizer_path}")
    from transformers import AutoTokenizer

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path),
        trust_remote_code=True,
        local_files_only=True,
    )
    tokenizer_identity = {
        "name_or_path": str(tokenizer_path.resolve()),
        "revision_or_snapshot": tokenizer_path.name,
        "class": tokenizer.__class__.__name__,
        "vocab_size": len(tokenizer),
    }
    readiness = official_readiness(args)
    compressor = None
    if OFFICIAL_METHOD in methods and readiness["eligible"]:
        try:
            compressor = load_official_compressor(readiness)
            readiness["status"] = "loaded_local_cpu"
        except Exception as exc:
            readiness["eligible"] = False
            readiness["status"] = "unavailable_or_audit_failed"
            readiness["reason"] = f"local official compressor load failed: {type(exc).__name__}: {exc}"

    reference_by_task = {
        task: resolve_reference_rows(
            args.reference_dir, task, args.max_samples_per_task
        )
        for task in tasks
    }
    manifest_hash = sha256_text(
        canonical_json(
            {
                task: [str(row["sample_id"]) for row in rows]
                for task, rows in reference_by_task.items()
            }
        )
    )
    audit_rows: List[Dict[str, Any]] = []
    official_failed: List[Dict[str, Any]] = []
    packed_rows: Dict[Tuple[str, str], List[Dict[str, Any]]] = collections.defaultdict(list)
    audit_fields = (
        "method_name",
        "task",
        "sample_id",
        "sample_index",
        "original_model_tokens",
        "compressed_model_tokens",
        "budget",
        "over_budget",
        "empty_context",
        "compressor_name",
        "compressor_config",
        "tokenizer_name",
        "compression_ratio",
        "selected_sentence_count",
        "selected_span_count",
        "candidate_sentence_count",
        "candidate_span_count",
        "ranking_candidate_span_count",
        "compression_time_ms",
        "error_message",
    )

    for task in tasks:
        for sample_index, source in enumerate(reference_by_task[task]):
            sample_id = str(source["sample_id"])
            question = str(source["question"])
            context = str(source["original_context"])
            original_tokens = token_count(tokenizer, context)
            for method in methods:
                started = time.perf_counter()
                error = ""
                result: Dict[str, Any] = {}
                compressor_name = method
                compressor_config: Dict[str, Any]
                if method == OFFICIAL_METHOD:
                    compressor_config = {
                        "rank_method": "longllmlingua",
                        "target_protocol": "Qwen exact packed-context tokens <=1024",
                        "cpu_only": True,
                        "official_model_path": readiness.get("official_model_path", ""),
                    }
                    if compressor is None:
                        error = str(readiness["reason"])
                    else:
                        try:
                            result = official_compress(
                                compressor,
                                tokenizer,
                                context,
                                question,
                                args.budget,
                            )
                        except Exception as exc:
                            error = f"{type(exc).__name__}: {exc}"
                else:
                    compressor_config = {
                        "implementation": "transparent_sentence_span_extractive",
                        "query_aware": method.endswith("query_aware"),
                        "max_span_tokens": args.max_span_tokens,
                        "max_mmr_candidates": args.max_mmr_candidates,
                        "candidate_prefilter": "top_base_salience_then_MMR",
                        "ranking": (
                            "query_IDF_entity_number_plus_internal_salience_MMR"
                            if method.endswith("query_aware")
                            else "context_TFIDF_centroid_position_plus_MMR"
                        ),
                        "redundancy_weight": 0.30,
                        "gold_used": False,
                        "prediction_used": False,
                        "final_repack": "exact_Qwen_tokenizer",
                    }
                    try:
                        result = exact_style_pack(
                            tokenizer,
                            context,
                            question,
                            method,
                            args.budget,
                            args.max_span_tokens,
                            args.max_mmr_candidates,
                        )
                    except Exception as exc:
                        error = f"{type(exc).__name__}: {exc}"
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                packed_text = str(result.get("packed_text", ""))
                compressed_tokens = (
                    int(result["compressed_model_tokens"])
                    if "compressed_model_tokens" in result
                    else None
                )
                empty_context = not packed_text.strip() or (
                    "Selected compressed evidence:" in packed_text
                    and not packed_text.split("Selected compressed evidence:", 1)[1].strip()
                )
                over_budget = bool(
                    compressed_tokens is not None and compressed_tokens > args.budget
                )
                ratio = (
                    compressed_tokens / max(1, original_tokens)
                    if compressed_tokens is not None
                    else None
                )
                row = {
                    "method_name": method,
                    "task": task,
                    "sample_id": sample_id,
                    "sample_index": sample_index,
                    "original_model_tokens": original_tokens,
                    "compressed_model_tokens": (
                        compressed_tokens if compressed_tokens is not None else ""
                    ),
                    "budget": args.budget,
                    "over_budget": over_budget,
                    "empty_context": empty_context if not error else "",
                    "compressor_name": compressor_name,
                    "compressor_config": canonical_json(compressor_config),
                    "tokenizer_name": tokenizer_identity["name_or_path"],
                    "compression_ratio": ratio if ratio is not None else "",
                    "selected_sentence_count": len(
                        result.get("selected_sentences", [])
                    ),
                    "selected_span_count": len(result.get("selected_spans", [])),
                    "candidate_sentence_count": result.get(
                        "candidate_sentence_count", ""
                    ),
                    "candidate_span_count": result.get("candidate_span_count", ""),
                    "ranking_candidate_span_count": result.get(
                        "ranking_candidate_span_count", ""
                    ),
                    "ranking_candidate_span_count": result.get(
                        "ranking_candidate_span_count", ""
                    ),
                    "compression_time_ms": elapsed_ms,
                    "error_message": error,
                }
                audit_rows.append(row)
                if method == OFFICIAL_METHOD and error:
                    official_failed.append(
                        {
                            **row,
                            "official_longllingua_status": readiness["status"],
                        }
                    )
                if error or over_budget or empty_context:
                    continue
                output = {
                    "sample_id": sample_id,
                    "sample_index": sample_index,
                    "task": task,
                    "variant": method,
                    "method_name": method,
                    "budget": args.budget,
                    "packed_text": packed_text,
                    "packed_prompt": packed_text,
                    "generation_prompt": generation_prompt(
                        task, packed_text, question
                    ),
                    "question": question,
                    "original_context": context,
                    "answers": source.get(
                        "answers", source.get("gold_analysis_only_answers", [])
                    ),
                    "gold_analysis_only_answers": source.get(
                        "gold_analysis_only_answers", source.get("answers", [])
                    ),
                    "all_classes": source.get("all_classes"),
                    "length": source.get("length", 0),
                    "source_file": source.get("source_file", ""),
                    "source_token_count": original_tokens,
                    "packed_token_count": compressed_tokens,
                    "model_token_count": compressed_tokens,
                    "compression_ratio": ratio,
                    "selected_sentence_ids": result.get(
                        "selected_sentences", []
                    ),
                    "selected_span_ids": [
                        f"s{item['sentence_index']}_w{item['token_window_index']}"
                        for item in result.get("selected_spans", [])
                    ],
                    "selected_sentences": result.get("selected_sentences", []),
                    "selected_spans": result.get("selected_spans", []),
                    "candidate_sentence_count": result.get(
                        "candidate_sentence_count", ""
                    ),
                    "candidate_span_count": result.get("candidate_span_count", ""),
                    "truncated": bool(result.get("truncated_final_span", False)),
                    "exact_model_token_budget": True,
                    "model_tokenizer": tokenizer_identity["name_or_path"],
                    "compression_time_ms": elapsed_ms,
                    "compression_used_gold": False,
                    "compression_used_prediction": False,
                    "official_longllmlingua": method == OFFICIAL_METHOD,
                }
                packed_rows[(method, task)].append(output)

    for (method, task), rows in packed_rows.items():
        write_jsonl(
            args.output_dir
            / "packed_prompts_build"
            / "packed_prompts"
            / method
            / f"budget_{args.budget}"
            / f"{task}.jsonl",
            rows,
        )

    write_csv(
        args.output_dir / "packing_budget_audit.csv",
        audit_rows,
        audit_fields,
    )
    official_rows = [
        row for row in audit_rows if row["method_name"] == OFFICIAL_METHOD
    ]
    write_csv(
        args.output_dir / "official_llmlingua_budget_audit.csv",
        official_rows,
        audit_fields,
    )
    write_jsonl(
        args.output_dir / "official_llmlingua_failed_cases.jsonl",
        official_failed,
    )
    official_success = [
        row for row in official_rows if not row.get("error_message")
    ]
    official_summary = {
        **readiness,
        "official_longllingua_status": (
            "audit_passed"
            if official_rows
            and len(official_success) == len(official_rows)
            and not any(row["over_budget"] for row in official_rows)
            and not any(row["empty_context"] for row in official_rows)
            else "unavailable_or_audit_failed"
        ),
        "requested_rows": len(official_rows),
        "successful_rows": len(official_success),
        "failed_rows": len(official_rows) - len(official_success),
        "over_budget_rows": sum(bool(row["over_budget"]) for row in official_rows),
        "empty_context_rows": sum(
            bool(row["empty_context"]) for row in official_rows if row["empty_context"] != ""
        ),
        "qa_score_available": False,
        "safe_for_formal_baseline_score": False,
    }
    write_json(
        args.output_dir / "official_llmlingua_budget_summary.json",
        official_summary,
    )
    packing_summary = summarize_audits(
        audit_rows, methods, tasks, args.budget
    )
    packing_summary.update(
        {
            "created_at": dt.datetime.now().astimezone().isoformat(),
            "output_dir": str(args.output_dir.resolve()),
            "reference_dir": str(args.reference_dir.resolve()),
            "reference_variant": REFERENCE_VARIANT,
            "sample_manifest_hash": manifest_hash,
            "sample_id_alignment": True,
            "tokenizer": tokenizer_identity,
            "tokenizer_match_reference": True,
            "prompt_wrapper": (
                "configs/dataset2prompt.json followed by Qwen chat template at QA time"
            ),
            "prompt_config_sha256": sha256_file(
                REPO_ROOT / "configs" / "dataset2prompt.json"
            ),
            "cpu_only": True,
            "qa_model_loaded": False,
            "gpu_used": False,
            "gold_used_for_compression": False,
            "prediction_used_for_compression": False,
            "max_samples_per_task": args.max_samples_per_task,
            "expected_full_sample_count": sum(EXPECTED_COUNTS.values()),
            "official_longllingua_status": official_summary[
                "official_longllingua_status"
            ],
        }
    )
    write_json(args.output_dir / "packing_summary.json", packing_summary)
    write_json(
        args.output_dir / "config.json",
        {
            "reference_dir": str(args.reference_dir.resolve()),
            "tasks": tasks,
            "methods": methods,
            "max_samples_per_task": args.max_samples_per_task,
            "budget": args.budget,
            "tokenizer": tokenizer_identity,
            "cpu_only": True,
            "seed": args.seed,
            "max_span_tokens": args.max_span_tokens,
            "max_mmr_candidates": args.max_mmr_candidates,
            "network_download_allowed": False,
        },
    )
    if packing_summary["status"] != "PASS":
        raise AuditError(
            f"style packing audit failed; inspect {args.output_dir / 'packing_summary.json'}"
        )
    print(str(args.output_dir.resolve()))


if __name__ == "__main__":
    main()
