#!/usr/bin/env python3
"""Build an audit-first verifier dataset from an existing DRM diagnostic.

This script is deliberately CPU-only.  It reconstructs prompt block text with
the exact tokenizer/prompt wrapper recorded by the historical fairness runner,
then verifies both prompt hashes and token counts before emitting any example.
Gold answers, predictions, and historical scores are retained only for labels
and offline analysis; every emitted field carries an inference-allowance flag.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HISTORY = (
    REPO_ROOT
    / "results"
    / "drm_budget_neutral_swap_multihop200_20260704_181544"
)
DEFAULT_LONGBENCH = (
    REPO_ROOT.parent / "HeadKV" / "data" / "LongBench"
)
DEFAULT_TOKENIZER = Path(
    "/home/yu-bai/.cache/huggingface/hub/"
    "models--Qwen--Qwen2.5-7B-Instruct/snapshots/"
    "a09a35458c702b33eeacc393d103063234e8bc28"
)
TASKS = ("hotpotqa", "2wikimqa", "musique")
METHOD_KEYS = (
    "sliminfer_original",
    "random_budget_neutral_swap",
    "budget_neutral_drm_swap",
    "sliminfer_larger",
    "random_add",
    "drm_add",
)
NO_CHAT_TASKS = {"trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"}

WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-']*")
ENTITY_RE = re.compile(
    r"\b(?:[A-Z][A-Za-z0-9_\-']*(?:\s+[A-Z][A-Za-z0-9_\-']*)*|"
    r"\d{1,4}(?:[-/]\d{1,2}){0,2})\b"
)
DATE_NUMBER_RE = re.compile(
    r"\b(?:\d+(?:[.,]\d+)?|(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|"
    r"Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|"
    r"Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)[A-Za-z]*\s+\d{1,2}"
    r"(?:,\s*\d{4})?)\b",
    re.IGNORECASE,
)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does",
    "for", "from", "had", "has", "have", "how", "in", "is", "it", "of",
    "on", "or", "that", "the", "this", "to", "was", "were", "what", "when",
    "where", "which", "who", "why", "with",
}

ANALYSIS_ONLY_FIELDS = {
    "task",
    "sample_id",
    "sample_index",
    "split",
    "gold_answers",
    "original_prediction",
    "random_swap_prediction",
    "drm_swap_prediction",
    "drm_add_prediction",
    "original_score",
    "random_swap_score",
    "drm_swap_score",
    "drm_add_score",
    "delta_drm_vs_original",
    "delta_drm_vs_random",
    "primary_label",
    "label_drm_better_original",
    "label_drm_worse_original",
    "label_drm_better_random",
    "label_drm_worse_random",
    "label_source",
    "historical_prompt_hash",
}

INFERENCE_NUMERIC_FIELDS = (
    "K",
    "selected_block_count",
    "prompt_token_count",
    "candidate_pool_size",
    "swap_count",
    "rescued_block_count",
    "removed_block_count",
    "rescue_score_mean",
    "removed_score_mean",
    "active_redundancy_mean",
    "query_overlap_rescued",
    "query_overlap_removed",
    "entity_overlap_rescued",
    "entity_overlap_removed",
    "date_number_overlap_rescued",
    "date_number_overlap_removed",
    "rescued_query_coverage_gap",
    "rescued_entity_coverage_gap",
    "rescued_vs_removed_query_gap",
    "rescued_vs_removed_entity_gap",
    "rescued_text_char_count",
    "removed_text_char_count",
    "active_summary_char_count",
    "fallback_used",
    "protected_block_violation",
    "over_budget",
)


class DatasetBuildError(RuntimeError):
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


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetBuildError(f"invalid JSONL {path}:{line_number}: {exc}") from exc


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


def finite_float(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise DatasetBuildError(f"{field} is not numeric: {value!r}") from exc
    if not math.isfinite(result):
        raise DatasetBuildError(f"{field} is not finite: {value!r}")
    return result


def load_longbench(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise DatasetBuildError(f"LongBench source not found: {path}")
    return list(iter_jsonl(path))


def split_for(index: int) -> str:
    remainder = index % 4
    if remainder in (0, 1):
        return "train"
    return "dev" if remainder == 2 else "test"


def words(text: str) -> Set[str]:
    return {
        match.group(0).lower()
        for match in WORD_RE.finditer(text or "")
        if match.group(0).lower() not in STOPWORDS and len(match.group(0)) > 1
    }


def entities(text: str) -> Set[str]:
    return {re.sub(r"\s+", " ", x).strip().lower() for x in ENTITY_RE.findall(text or "")}


def dates_numbers(text: str) -> Set[str]:
    return {re.sub(r"\s+", " ", x).strip().lower() for x in DATE_NUMBER_RE.findall(text or "")}


def coverage(needles: Set[str], haystack: str) -> float:
    if not needles:
        return 0.0
    lowered = (haystack or "").lower()
    return len({term for term in needles if term in lowered}) / len(needles)


def uncovered_coverage(needles: Set[str], rescued: str, active: str) -> float:
    if not needles:
        return 0.0
    rescued_lower = rescued.lower()
    active_lower = active.lower()
    novel = {term for term in needles if term in rescued_lower and term not in active_lower}
    return len(novel) / len(needles)


def block_texts_for_ids(block_texts: Sequence[str], ids: Sequence[int]) -> List[str]:
    result: List[str] = []
    for block_id in ids:
        if block_id < 0 or block_id >= len(block_texts):
            raise DatasetBuildError(
                f"block id {block_id} outside reconstructed range [0,{len(block_texts)})"
            )
        result.append(block_texts[block_id])
    return result


def neighbor_texts(
    block_texts: Sequence[str], ids: Sequence[int], window: int
) -> List[str]:
    selected: List[int] = []
    seen: Set[int] = set(int(x) for x in ids)
    for block_id in ids:
        for neighbor in range(max(0, block_id - window), min(len(block_texts), block_id + window + 1)):
            if neighbor not in seen and neighbor not in selected:
                selected.append(neighbor)
    return [block_texts[index] for index in selected]


def active_summary(
    block_texts: Sequence[str],
    active_ids: Sequence[int],
    question: str,
    max_blocks: int,
    max_chars: int,
) -> Tuple[str, List[int]]:
    query_terms = words(question)
    ranked: List[Tuple[float, int, str]] = []
    for block_id in active_ids:
        if block_id < 0 or block_id >= len(block_texts):
            raise DatasetBuildError(f"active block id {block_id} outside reconstructed prompt")
        text = block_texts[block_id]
        text_terms = words(text)
        overlap = len(query_terms & text_terms) / max(1, len(query_terms))
        ranked.append((overlap, int(block_id), text))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    chunks: List[str] = []
    chosen: List[int] = []
    for _, block_id, text in ranked[:max_blocks]:
        chunk = f"[block {block_id}] {text.strip()}"
        remaining = max_chars - sum(len(x) + 1 for x in chunks)
        if remaining <= 0:
            break
        chunks.append(chunk[:remaining])
        chosen.append(block_id)
    return "\n".join(chunks), chosen


def method_rows(audit_path: Path) -> Dict[Tuple[str, str, int], Dict[str, Any]]:
    result: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    for row in iter_jsonl(audit_path):
        method = str(row.get("method_key", ""))
        task = str(row.get("task", ""))
        if method not in METHOD_KEYS or task not in TASKS:
            continue
        index = int(row.get("sample_index", -1))
        key = (method, task, index)
        if key in result:
            raise DatasetBuildError(f"duplicate historical audit key: {key}")
        result[key] = row
    expected = len(METHOD_KEYS) * len(TASKS) * 200
    if len(result) != expected:
        counts = collections.Counter((key[0], key[1]) for key in result)
        raise DatasetBuildError(
            f"expected {expected} aligned audit rows, found {len(result)}; counts={dict(counts)}"
        )
    return result


def reconstruct_blocks(
    tokenizer: Any,
    prompt_template: str,
    sample: Mapping[str, Any],
    task: str,
    max_model_len: int,
    block_size: int,
) -> Tuple[str, List[int], List[str]]:
    prompt = prompt_template.format(**sample)
    raw_ids = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
    if len(raw_ids) > max_model_len:
        half = max_model_len // 2
        prompt = (
            tokenizer.decode(raw_ids[:half], skip_special_tokens=True)
            + tokenizer.decode(raw_ids[-half:], skip_special_tokens=True)
        )
    if task not in NO_CHAT_TASKS:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    ids = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0].tolist()
    texts = [
        tokenizer.decode(
            ids[start : start + block_size],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        for start in range(0, len(ids), block_size)
    ]
    return prompt, ids, texts


def build_example(
    task: str,
    index: int,
    source: Mapping[str, Any],
    aligned: Mapping[Tuple[str, str, int], Mapping[str, Any]],
    tokenizer: Any,
    prompt_template: str,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    rows = {method: aligned[(method, task, index)] for method in METHOD_KEYS}
    sample_ids = {str(row.get("sample_id")) for row in rows.values()}
    if len(sample_ids) != 1:
        raise DatasetBuildError(f"{task}/{index}: method sample IDs differ: {sample_ids}")
    sample_id = sample_ids.pop()
    source_id = str(source.get("_id", source.get("id", index)))
    if source_id != sample_id:
        raise DatasetBuildError(
            f"{task}/{index}: source sample ID {source_id!r} != historical {sample_id!r}"
        )

    prompt, prompt_ids, block_texts = reconstruct_blocks(
        tokenizer,
        prompt_template,
        source,
        task,
        args.max_model_len,
        args.block_size,
    )
    prompt_hash = sha256_text(prompt)
    historical_hashes = {str(row.get("prompt_hash")) for row in rows.values()}
    token_counts = {int(row.get("prompt_token_count", -1)) for row in rows.values()}
    if historical_hashes != {prompt_hash}:
        raise DatasetBuildError(
            f"{task}/{index}: reconstructed prompt hash mismatch "
            f"{prompt_hash} vs {historical_hashes}"
        )
    if token_counts != {len(prompt_ids)}:
        raise DatasetBuildError(
            f"{task}/{index}: reconstructed token count {len(prompt_ids)} vs {token_counts}"
        )

    original = rows["sliminfer_original"]
    random_swap = rows["random_budget_neutral_swap"]
    drm_swap = rows["budget_neutral_drm_swap"]
    drm_add = rows["drm_add"]
    rescued_ids = [int(x) for x in drm_swap.get("rescued_block_ids", [])]
    removed_ids = [int(x) for x in drm_swap.get("removed_block_ids", [])]
    original_active_ids = [int(x) for x in original.get("used_blocks", [])]
    rescued_texts = block_texts_for_ids(block_texts, rescued_ids)
    removed_texts = block_texts_for_ids(block_texts, removed_ids)
    rescued_joined = "\n".join(rescued_texts)
    removed_joined = "\n".join(removed_texts)
    all_active_text = "\n".join(block_texts_for_ids(block_texts, original_active_ids))
    summary, summary_ids = active_summary(
        block_texts,
        original_active_ids,
        str(source.get("input", "")),
        args.active_summary_blocks,
        args.active_summary_max_chars,
    )
    question = str(source.get("input", ""))
    query_terms = words(question)
    query_entities = entities(question)
    query_dates = dates_numbers(question)

    original_score = finite_float(original.get("prediction_score"), "original_score")
    random_score = finite_float(random_swap.get("prediction_score"), "random_swap_score")
    drm_score = finite_float(drm_swap.get("prediction_score"), "drm_swap_score")
    drm_add_score = finite_float(drm_add.get("prediction_score"), "drm_add_score")
    delta_original = drm_score - original_score
    delta_random = drm_score - random_score
    primary_label = (
        1
        if delta_original >= args.delta_threshold and delta_random >= 0.0
        else (
            -1
            if delta_original <= -args.delta_threshold
            or delta_random <= -args.delta_threshold
            else 0
        )
    )

    query_rescued = coverage(query_terms, rescued_joined)
    query_removed = coverage(query_terms, removed_joined)
    entity_rescued = coverage(query_entities, rescued_joined)
    entity_removed = coverage(query_entities, removed_joined)
    date_rescued = coverage(query_dates, rescued_joined)
    date_removed = coverage(query_dates, removed_joined)
    query_gap = uncovered_coverage(query_terms, rescued_joined, all_active_text)
    entity_gap = uncovered_coverage(query_entities, rescued_joined, all_active_text)

    example: Dict[str, Any] = {
        "task": task,
        "sample_id": sample_id,
        "sample_index": index,
        "split": split_for(index),
        "question": question,
        "gold_answers": list(source.get("answers", drm_swap.get("gold_answers", [])) or []),
        "original_prediction": str(original.get("prediction", "")),
        "random_swap_prediction": str(random_swap.get("prediction", "")),
        "drm_swap_prediction": str(drm_swap.get("prediction", "")),
        "drm_add_prediction": str(drm_add.get("prediction", "")),
        "original_score": original_score,
        "random_swap_score": random_score,
        "drm_swap_score": drm_score,
        "drm_add_score": drm_add_score,
        "delta_drm_vs_original": delta_original,
        "delta_drm_vs_random": delta_random,
        "rescued_block_ids": rescued_ids,
        "removed_block_ids": removed_ids,
        "rescued_block_texts": rescued_texts,
        "removed_block_texts": removed_texts,
        "rescued_neighbor_texts": neighbor_texts(
            block_texts, rescued_ids, args.neighbor_window
        ),
        "removed_neighbor_texts": neighbor_texts(
            block_texts, removed_ids, args.neighbor_window
        ),
        "active_context_summary": summary,
        "active_context_summary_block_ids": summary_ids,
        "K": int(drm_swap.get("K", 0)),
        "selected_block_count": int(drm_swap.get("selected_block_count", 0)),
        "prompt_token_count": len(prompt_ids),
        "rescue_score_mean": finite_float(
            drm_swap.get("rescue_score_mean", 0.0), "rescue_score_mean"
        ),
        "rescue_score_top1": None,
        "rescue_score_top2": None,
        "rescue_boundary_margin": None,
        "removed_score_mean": finite_float(
            drm_swap.get("removed_score_mean", 0.0), "removed_score_mean"
        ),
        "removed_score_min": None,
        "removed_score_max": None,
        "active_redundancy_mean": finite_float(
            drm_swap.get("active_redundancy_mean", 0.0), "active_redundancy_mean"
        ),
        "query_overlap_rescued": query_rescued,
        "query_overlap_removed": query_removed,
        "entity_overlap_rescued": entity_rescued,
        "entity_overlap_removed": entity_removed,
        "date_number_overlap_rescued": date_rescued,
        "date_number_overlap_removed": date_removed,
        "rescued_query_coverage_gap": query_gap,
        "rescued_entity_coverage_gap": entity_gap,
        "rescued_vs_removed_query_gap": query_rescued - query_removed,
        "rescued_vs_removed_entity_gap": entity_rescued - entity_removed,
        "candidate_pool_size": int(
            drm_swap.get(
                "pruned_candidate_count",
                drm_swap.get("selector_candidate_block_count", 0),
            )
        ),
        "swap_count": int(drm_swap.get("swap_count", 0)),
        "rescued_block_count": len(rescued_ids),
        "removed_block_count": len(removed_ids),
        "rescued_text_char_count": len(rescued_joined),
        "removed_text_char_count": len(removed_joined),
        "active_summary_char_count": len(summary),
        "fallback_used": bool(drm_swap.get("fallback_used", False)),
        "fallback_reason": str(drm_swap.get("fallback_reason", "")),
        "protected_block_violation": int(
            bool(drm_swap.get("protected_block_violation", False))
        ),
        "over_budget": int(bool(drm_swap.get("over_budget", False))),
        "primary_label": primary_label,
        "label_drm_better_original": delta_original > 0.0,
        "label_drm_worse_original": delta_original < 0.0,
        "label_drm_better_random": delta_random > 0.0,
        "label_drm_worse_random": delta_random < 0.0,
        "label_source": "score_delta_only",
        "historical_prompt_hash": prompt_hash,
        "reconstructed_block_count": len(block_texts),
        "block_text_available": True,
        "candidate_score_detail_available": False,
        "inference_numeric_feature_names": list(INFERENCE_NUMERIC_FIELDS),
    }

    # This text is allowed for a future local cross encoder; it contains no
    # answer, prediction, task name, sample ID, label, or historical score.
    example["verifier_input_text"] = (
        f"[QUERY] {question}\n"
        f"[RESCUED] {rescued_joined}\n"
        f"[REMOVED] {removed_joined}\n"
        f"[ACTIVE] {summary}"
    )
    example["verifier_input_text_hash"] = sha256_text(example["verifier_input_text"])

    allowed_fields = {
        "question",
        "rescued_block_ids",
        "removed_block_ids",
        "rescued_block_texts",
        "removed_block_texts",
        "rescued_neighbor_texts",
        "removed_neighbor_texts",
        "active_context_summary",
        "active_context_summary_block_ids",
        "fallback_reason",
        "verifier_input_text",
        "verifier_input_text_hash",
        "reconstructed_block_count",
        "block_text_available",
        "candidate_score_detail_available",
        "inference_numeric_feature_names",
        *INFERENCE_NUMERIC_FIELDS,
    }
    example["inference_allowed"] = {
        field: bool(field in allowed_fields and field not in ANALYSIS_ONLY_FIELDS)
        for field in example
        if field != "inference_allowed"
    }
    forbidden_allowed = [
        field
        for field in ANALYSIS_ONLY_FIELDS
        if example["inference_allowed"].get(field, False)
    ]
    if forbidden_allowed:
        raise DatasetBuildError(
            f"{task}/{index}: forbidden fields marked inference-allowed: {forbidden_allowed}"
        )
    return example


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history_dir", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--longbench_dir", type=Path, default=DEFAULT_LONGBENCH)
    parser.add_argument("--tokenizer_path", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument("--samples_per_task", type=int, default=200)
    parser.add_argument("--max_model_len", type=int, default=32768)
    parser.add_argument("--block_size", type=int, default=64)
    parser.add_argument("--neighbor_window", type=int, default=1)
    parser.add_argument("--active_summary_blocks", type=int, default=6)
    parser.add_argument("--active_summary_max_chars", type=int, default=3000)
    parser.add_argument("--delta_threshold", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    requested_tasks = tuple(x.strip() for x in args.tasks.split(",") if x.strip())
    if requested_tasks != TASKS:
        raise DatasetBuildError(
            f"this historical builder requires tasks={','.join(TASKS)}; got {requested_tasks}"
        )
    if args.samples_per_task != 200:
        raise DatasetBuildError("historical diagnostic must be consumed as exactly 200 samples/task")
    if not args.tokenizer_path.exists():
        raise DatasetBuildError(f"local tokenizer path does not exist: {args.tokenizer_path}")
    if args.output_dir is None:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = REPO_ROOT / "results" / f"drm_rescue_verifier_dataset_{stamp}"
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise DatasetBuildError(f"refusing to overwrite non-empty output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.tokenizer_path),
        trust_remote_code=True,
        local_files_only=True,
    )
    audit_path = args.history_dir / "audit" / "per_sample_audit.jsonl"
    if not audit_path.exists():
        raise DatasetBuildError(f"historical per-sample audit not found: {audit_path}")
    aligned = method_rows(audit_path)
    prompts_path = REPO_ROOT / "configs" / "dataset2prompt.json"
    prompts = read_json(prompts_path)
    examples: List[Dict[str, Any]] = []
    for task in TASKS:
        source_rows = load_longbench(args.longbench_dir / f"{task}.jsonl")
        if len(source_rows) < args.samples_per_task:
            raise DatasetBuildError(
                f"{task}: source has {len(source_rows)} rows, expected >= {args.samples_per_task}"
            )
        if task not in prompts:
            raise DatasetBuildError(f"missing prompt template for {task}")
        for index in range(args.samples_per_task):
            examples.append(
                build_example(
                    task,
                    index,
                    source_rows[index],
                    aligned,
                    tokenizer,
                    str(prompts[task]),
                    args,
                )
            )

    if len(examples) != 600:
        raise DatasetBuildError(f"expected 600 verifier examples, built {len(examples)}")
    write_jsonl(args.output_dir / "verifier_examples.jsonl", examples)
    for split in ("train", "dev", "test"):
        write_jsonl(
            args.output_dir / f"verifier_examples_{split}.jsonl",
            (row for row in examples if row["split"] == split),
        )

    split_counts = collections.Counter(row["split"] for row in examples)
    task_split_counts = collections.Counter(
        (row["task"], row["split"]) for row in examples
    )
    label_counts = collections.Counter(str(row["primary_label"]) for row in examples)
    label_by_task = {
        task: dict(
            collections.Counter(
                str(row["primary_label"]) for row in examples if row["task"] == task
            )
        )
        for task in TASKS
    }
    label_by_split = {
        split: dict(
            collections.Counter(
                str(row["primary_label"]) for row in examples if row["split"] == split
            )
        )
        for split in ("train", "dev", "test")
    }
    summary = {
        "status": "PASS",
        "created_at": dt.datetime.now().astimezone().isoformat(),
        "output_dir": str(args.output_dir.resolve()),
        "history_dir": str(args.history_dir.resolve()),
        "historical_audit": str(audit_path.resolve()),
        "historical_audit_sha256": sha256_file(audit_path),
        "longbench_dir": str(args.longbench_dir.resolve()),
        "tokenizer_path": str(args.tokenizer_path.resolve()),
        "prompt_config": str(prompts_path.resolve()),
        "prompt_config_sha256": sha256_file(prompts_path),
        "tasks": list(TASKS),
        "examples": len(examples),
        "split_rule": {
            "train": "sample_index % 4 in {0,1}",
            "dev": "sample_index % 4 == 2",
            "test": "sample_index % 4 == 3",
        },
        "split_counts": dict(split_counts),
        "task_split_counts": {
            f"{task}/{split}": task_split_counts[(task, split)]
            for task in TASKS
            for split in ("train", "dev", "test")
        },
        "label_definition": {
            "delta_threshold": args.delta_threshold,
            "positive": "drm-original >= 1.0 and drm-random >= 0",
            "negative": "drm-original <= -1.0 or drm-random <= -1.0",
            "neutral": "otherwise",
            "label_source": "score_delta_only",
        },
        "prompt_hash_matches": len(examples),
        "prompt_token_count_matches": len(examples),
        "sample_id_alignment_matches": len(examples),
        "block_text_reconstructed": len(examples),
        "candidate_score_detail_available": False,
        "candidate_score_detail_reason": (
            "historical consolidated and raw final-layer audits do not retain "
            "per-candidate top1/top2/boundary scores"
        ),
        "class_imbalance_warning": (
            "Only non-neutral examples are eligible for default training; "
            "held-out metrics are high variance when a class has very few examples."
        ),
    }
    write_json(args.output_dir / "dataset_summary.json", summary)
    write_json(
        args.output_dir / "label_distribution.json",
        {
            "overall": dict(label_counts),
            "by_task": label_by_task,
            "by_split": label_by_split,
            "non_neutral_training_default": True,
        },
    )

    all_fields = sorted(
        {field for row in examples for field in row if field != "inference_allowed"}
    )
    coverage: Dict[str, Any] = {}
    for field in all_fields:
        present = sum(
            row.get(field) is not None
            and row.get(field) != ""
            and row.get(field) != []
            for row in examples
        )
        coverage[field] = {
            "available_count": present,
            "total_count": len(examples),
            "coverage": present / len(examples),
            "inference_allowed": all(
                bool(row["inference_allowed"].get(field, False)) for row in examples
            ),
        }
    coverage["rescue_score_top1"]["unavailable_reason"] = (
        "not saved in historical final-layer audit"
    )
    coverage["rescue_score_top2"]["unavailable_reason"] = (
        "not saved in historical final-layer audit"
    )
    coverage["rescue_boundary_margin"]["unavailable_reason"] = (
        "not saved in historical final-layer audit"
    )
    coverage["removed_score_min"]["unavailable_reason"] = (
        "only removed_score_mean was saved historically"
    )
    coverage["removed_score_max"]["unavailable_reason"] = (
        "only removed_score_mean was saved historically"
    )
    write_json(
        args.output_dir / "field_coverage.json",
        {
            "fields": coverage,
            "inference_numeric_feature_names": list(INFERENCE_NUMERIC_FIELDS),
            "analysis_only_fields": sorted(ANALYSIS_ONLY_FIELDS),
            "forbidden_feature_used": False,
        },
    )
    print(str(args.output_dir.resolve()))


if __name__ == "__main__":
    main()
