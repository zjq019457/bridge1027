#!/usr/bin/env python3
"""Router-v3 structural feature audit for Adaptive DC-SlimInfer.

This script is offline-only. It reads existing tiny quality and router-v2
artifacts, reconstructs prompt blocks with the configured Qwen tokenizer, and
replays pre-registered router-v3 diagnostic rules. It does not run generation,
scoring, full LongBench, or the DC selector.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TINY_DIR = Path("results/dc_sliminfer_tiny_quality_smoke_20260708_004504")
DEFAULT_V2_DIR = Path("results/adaptive_dc_sliminfer_router_v2_20260708_042635")
DEFAULT_OUTPUT_PARENT = Path("results")
TOP_PRUNED_M = 16
STOPWORDS = {
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "can",
    "did",
    "do",
    "does",
    "doing",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "has",
    "have",
    "having",
    "he",
    "her",
    "here",
    "hers",
    "him",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "me",
    "more",
    "most",
    "my",
    "no",
    "nor",
    "not",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "our",
    "out",
    "over",
    "own",
    "same",
    "she",
    "so",
    "some",
    "such",
    "than",
    "that",
    "the",
    "their",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "very",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whom",
    "why",
    "will",
    "with",
    "you",
    "your",
}
STRUCTURAL_FEATURES = [
    "cutoff_margin",
    "cutoff_margin_percentile",
    "near_cutoff_candidate_count",
    "original_doc_entropy",
    "original_top_doc_mass",
    "original_neighbor_cluster_ratio",
    "pruned_query_overlap_max",
    "pruned_query_overlap_mean_topM",
    "selected_query_overlap_mean",
    "selected_vs_pruned_query_overlap_gap",
    "query_entity_count",
    "selected_entity_coverage",
    "pruned_entity_coverage_topM",
    "entity_coverage_gap",
    "evidence_dispersion_score",
]
REPORT_FOOTER = (
    "Router-v2 selection-delta features are insufficient.\n"
    "Adaptive DC-SlimInfer can continue only if router-v3 structural under-coverage "
    "features show non-trivial separation."
)


def read_json(path: Optional[Path]) -> Any:
    if not path or not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Optional[Path]) -> List[Dict[str, Any]]:
    if not path or not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if isinstance(row, dict):
                rows.append(row)
    return rows


def read_csv(path: Optional[Path]) -> List[Dict[str, Any]]:
    if not path or not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def parse_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if value == 0:
            return False
        if value == 1:
            return True
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    if text in {"", "none", "null", "nan", "unavailable"}:
        return None
    return None


def to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        number = float(value)
    else:
        text = str(value).strip()
        if text == "" or text.lower() in {"none", "null", "nan", "unavailable"}:
            return None
        try:
            number = float(text)
        except ValueError:
            return None
    if math.isnan(number):
        return None
    return number


def fmt(value: Any, digits: int = 4) -> str:
    number = to_float(value)
    if number is None:
        return "unavailable" if value is None else str(value)
    return f"{number:.{digits}f}"


def mean(values: Iterable[Any]) -> Optional[float]:
    clean = [value for value in (to_float(item) for item in values) if value is not None]
    if not clean:
        return None
    return sum(clean) / len(clean)


def median(values: Iterable[Any]) -> Optional[float]:
    clean = sorted(value for value in (to_float(item) for item in values) if value is not None)
    if not clean:
        return None
    return statistics.median(clean)


def pearson(xs: Iterable[Any], ys: Iterable[Any]) -> Optional[float]:
    pairs: List[Tuple[float, float]] = []
    for raw_x, raw_y in zip(xs, ys):
        x = to_float(raw_x)
        y = to_float(raw_y)
        if x is not None and y is not None:
            pairs.append((x, y))
    if len(pairs) < 2:
        return None
    x_mean = sum(x for x, _ in pairs) / len(pairs)
    y_mean = sum(y for _, y in pairs) / len(pairs)
    x_var = sum((x - x_mean) ** 2 for x, _ in pairs)
    y_var = sum((y - y_mean) ** 2 for _, y in pairs)
    if x_var <= 0 or y_var <= 0:
        return None
    covariance = sum((x - x_mean) * (y - y_mean) for x, y in pairs)
    return covariance / math.sqrt(x_var * y_var)


def average_ranks(values: Sequence[float]) -> List[float]:
    order = sorted(range(len(values)), key=lambda idx: values[idx])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + 1 + end) / 2.0
        for index in order[start:end]:
            ranks[index] = rank
        start = end
    return ranks


def spearman(xs: Iterable[Any], ys: Iterable[Any]) -> Optional[float]:
    pairs: List[Tuple[float, float]] = []
    for raw_x, raw_y in zip(xs, ys):
        x = to_float(raw_x)
        y = to_float(raw_y)
        if x is not None and y is not None:
            pairs.append((x, y))
    if len(pairs) < 2:
        return None
    return pearson(average_ranks([x for x, _ in pairs]), average_ranks([y for _, y in pairs]))


def sample_key(row: Dict[str, Any]) -> Tuple[str, int, str]:
    return (
        str(row.get("task", "")),
        int(to_float(row.get("sample_index")) or 0),
        str(row.get("sample_id", "")),
    )


def find_file(root: Path, exact: str, keywords: Sequence[str], suffixes: Sequence[str]) -> Optional[Path]:
    exact_path = root / exact
    if exact_path.exists():
        return exact_path
    lowered = [keyword.lower() for keyword in keywords]
    suffix_set = {suffix.lower() for suffix in suffixes}
    candidates: List[Path] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in suffix_set:
            continue
        name = path.name.lower()
        if all(keyword in name for keyword in lowered):
            candidates.append(path)
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: (len(item.parts), str(item)))[0]


def load_longbench_samples(longbench_dir: Path, dataset: str) -> List[Dict[str, Any]]:
    candidates = [
        longbench_dir / f"{dataset}.jsonl",
        longbench_dir / f"{dataset}.json",
        longbench_dir / dataset / "test.jsonl",
        longbench_dir / dataset / "test.json",
        longbench_dir / dataset / f"{dataset}.jsonl",
        longbench_dir / dataset / f"{dataset}.json",
    ]
    for path in candidates:
        if not path.exists():
            continue
        if path.suffix == ".jsonl":
            return [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        payload = read_json(path)
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict) and isinstance(payload.get("data"), list):
            return payload["data"]
    raise FileNotFoundError(f"LongBench dataset not found for {dataset} under {longbench_dir}")


def build_chat_prompt(tokenizer: Any, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def build_prompt_blocks(
    tokenizer: Any,
    prompt_format: str,
    sample: Dict[str, Any],
    dataset: str,
    max_model_len: int,
    block_size: int,
) -> Tuple[str, List[str]]:
    prompt = prompt_format.format(**sample)
    input_ids = tokenizer(prompt, truncation=False, return_tensors=None).input_ids
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    if len(input_ids) > max_model_len:
        half = max_model_len // 2
        prompt = tokenizer.decode(input_ids[:half], skip_special_tokens=True) + tokenizer.decode(
            input_ids[-half:], skip_special_tokens=True
        )
    if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
        prompt = build_chat_prompt(tokenizer, prompt)
    chat_ids = tokenizer(prompt, truncation=False, return_tensors=None).input_ids
    if chat_ids and isinstance(chat_ids[0], list):
        chat_ids = chat_ids[0]
    blocks = []
    for start in range(0, len(chat_ids), block_size):
        blocks.append(
            tokenizer.decode(
                chat_ids[start : start + block_size],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        )
    return prompt, blocks


def text_tokens(text: str) -> List[str]:
    return [match.group(0).lower() for match in re.finditer(r"[A-Za-z][A-Za-z0-9_\-']*", text)]


def keyword_set(text: str) -> Set[str]:
    words = {
        token.strip("'").lower()
        for token in text_tokens(text)
        if len(token.strip("'")) >= 3 and token.lower() not in STOPWORDS
    }
    return {word for word in words if word}


def extract_query_keywords(query: str) -> List[str]:
    keywords = sorted(keyword_set(query))
    capitalized_phrases = []
    for match in re.finditer(r"(?:[A-Z][A-Za-z0-9'.-]+(?:\s+|$)){1,5}", query):
        phrase = " ".join(match.group(0).split()).strip()
        if phrase and phrase.lower() not in STOPWORDS:
            capitalized_phrases.extend(text_tokens(phrase))
    combined = sorted(set(keywords) | {token for token in capitalized_phrases if token not in STOPWORDS})
    return combined


def lexical_overlap(text: str, query_keywords: Sequence[str]) -> float:
    if not query_keywords:
        return 0.0
    tokens = keyword_set(text)
    if not tokens:
        return 0.0
    hits = sum(1 for keyword in query_keywords if keyword.lower() in tokens)
    return hits / len(set(query_keywords))


def coverage_ratio(texts: Sequence[str], query_keywords: Sequence[str]) -> float:
    if not query_keywords:
        return 0.0
    joined_tokens: Set[str] = set()
    for text in texts:
        joined_tokens.update(keyword_set(text))
    hits = sum(1 for keyword in query_keywords if keyword.lower() in joined_tokens)
    return hits / len(set(query_keywords))


def block_doc_proxy(block_id: int, block_text: str, num_blocks: int) -> str:
    matches = re.findall(r"\bPassage\s+(\d+)\s*:", block_text, flags=re.IGNORECASE)
    if matches:
        return f"passage:{matches[-1]}"
    bins = 8
    width = max(1, math.ceil(max(1, num_blocks) / bins))
    return f"posbin:{min(bins - 1, int(block_id) // width)}"


def entropy_from_counts(counts: Counter) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    value = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            value -= p * math.log(p)
    return value


def neighbor_cluster_ratio(selected_ids: Sequence[int]) -> float:
    ids = sorted(set(int(item) for item in selected_ids))
    if not ids:
        return 0.0
    selected_set = set(ids)
    clustered = sum(1 for block_id in ids if block_id - 1 in selected_set or block_id + 1 in selected_set)
    return clustered / len(ids)


def minmax_normalize(rows: List[Dict[str, Any]], source: str, target: str) -> None:
    values = [to_float(row.get(source)) for row in rows]
    clean = [value for value in values if value is not None]
    if not clean:
        for row in rows:
            row[target] = None
        return
    lo = min(clean)
    hi = max(clean)
    for row in rows:
        value = to_float(row.get(source))
        if value is None:
            row[target] = None
        elif hi == lo:
            row[target] = 0.5
        else:
            row[target] = (value - lo) / (hi - lo)


def load_inputs(tiny_dir: Path, v2_dir: Path) -> Tuple[Dict[str, Any], List[str]]:
    artifacts = {
        "comparison_jsonl": find_file(
            tiny_dir, "per_sample_comparison.jsonl", ["sample", "comparison"], [".jsonl"]
        ),
        "comparison_csv": find_file(
            tiny_dir, "per_sample_comparison.csv", ["sample", "comparison"], [".csv"]
        ),
        "predictions": find_file(
            tiny_dir, "per_sample_predictions.jsonl", ["sample", "prediction"], [".jsonl"]
        ),
        "selection_audit": find_file(
            tiny_dir, "dc_gate_selection_audit.jsonl", ["selection", "audit"], [".jsonl"]
        ),
        "config": tiny_dir / "run_dc_sliminfer" / "config.json",
        "task_summary": find_file(tiny_dir, "task_summary.csv", ["task", "summary"], [".csv"]),
        "v2_policy_replay": find_file(
            v2_dir, "adaptive_dc_v2_policy_replay.csv", ["policy", "replay"], [".csv"]
        ),
        "v2_report": find_file(v2_dir, "adaptive_dc_v2_report.md", ["report"], [".md"]),
    }
    missing = [name for name, path in artifacts.items() if path is None or not Path(path).exists()]
    comparison_rows = read_jsonl(artifacts["comparison_jsonl"])
    if not comparison_rows:
        comparison_rows = read_csv(artifacts["comparison_csv"])
    data = {
        "artifacts": artifacts,
        "comparison_rows": comparison_rows,
        "prediction_rows": read_jsonl(artifacts["predictions"]),
        "selection_rows": read_jsonl(artifacts["selection_audit"]),
        "config": read_json(artifacts["config"]),
        "task_summary": read_csv(artifacts["task_summary"]),
        "v2_policy_rows": read_csv(artifacts["v2_policy_replay"]),
        "v2_report": Path(artifacts["v2_report"]).read_text(encoding="utf-8")
        if artifacts["v2_report"] and Path(artifacts["v2_report"]).exists()
        else "",
    }
    return data, [f"missing artifact: {name}" for name in missing]


def rows_by_key(rows: List[Dict[str, Any]]) -> Dict[Tuple[str, int, str], Dict[str, Any]]:
    return {sample_key(row): row for row in rows}


def prediction_by_method(
    rows: List[Dict[str, Any]]
) -> Dict[Tuple[str, int, str], Dict[str, Dict[str, Any]]]:
    grouped: Dict[Tuple[str, int, str], Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        grouped[sample_key(row)][str(row.get("method", ""))] = row
    return grouped


def final_selection_rows(rows: List[Dict[str, Any]]) -> Dict[Tuple[str, int, str], Dict[str, Any]]:
    grouped: Dict[Tuple[str, int, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[sample_key(row)].append(row)
    final_rows: Dict[Tuple[str, int, str], Dict[str, Any]] = {}
    for key, sample_rows in grouped.items():
        final_rows[key] = sorted(sample_rows, key=lambda row: int(row.get("layer_id", -1)))[-1]
    return final_rows


def score_table(rows: List[Dict[str, Any]], score_field: str) -> Dict[str, float]:
    by_task: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        score = to_float(row.get(score_field))
        if score is not None:
            by_task[str(row.get("task"))].append(score)
    scores = {task: sum(values) / len(values) for task, values in by_task.items() if values}
    scores["macro"] = sum(scores.values()) / len(scores) if scores else 0.0
    return scores


def compute_cutoff_features(selection_row: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], Optional[int], str]:
    selected_scores = selection_row.get("selected_scores")
    pruned_scores = selection_row.get("pruned_scores")
    if not isinstance(selected_scores, list) or not isinstance(pruned_scores, list):
        return None, None, None, "unavailable:selected_pruned_score_detail_missing"
    selected_clean = [to_float(value) for value in selected_scores]
    pruned_clean = [to_float(value) for value in pruned_scores]
    selected_clean = [value for value in selected_clean if value is not None]
    pruned_clean = [value for value in pruned_clean if value is not None]
    if not selected_clean or not pruned_clean:
        return None, None, None, "unavailable:selected_or_pruned_scores_empty"
    selected_tail = min(selected_clean)
    pruned_head = max(pruned_clean)
    margin = selected_tail - pruned_head
    all_scores = sorted(selected_clean + pruned_clean)
    below_equal = sum(1 for score in all_scores if score <= selected_tail)
    percentile = below_equal / len(all_scores)
    eps = max(1e-12, abs(selected_tail) * 0.05)
    near_count = sum(1 for score in all_scores if abs(score - selected_tail) <= eps)
    return margin, percentile, near_count, ""


def build_base_rows(
    data: Dict[str, Any],
    tokenizer: Any,
    longbench_dir: Path,
    dataset2prompt: Dict[str, str],
    max_model_len: int,
    block_size: int,
) -> List[Dict[str, Any]]:
    comparison_by_key = rows_by_key(data["comparison_rows"])
    prediction_rows = prediction_by_method(data["prediction_rows"])
    final_rows = final_selection_rows(data["selection_rows"])
    samples_cache: Dict[str, List[Dict[str, Any]]] = {}
    feature_rows: List[Dict[str, Any]] = []

    for key, comparison in sorted(comparison_by_key.items(), key=lambda item: (item[0][0], item[0][1])):
        task, sample_index, sample_id = key
        if task not in samples_cache:
            samples_cache[task] = load_longbench_samples(longbench_dir, task)
        sample = samples_cache[task][sample_index]
        query_text = str(sample.get("input", sample.get("question", sample.get("query", ""))))
        prompt_text, block_texts = build_prompt_blocks(
            tokenizer=tokenizer,
            prompt_format=dataset2prompt[task],
            sample=sample,
            dataset=task,
            max_model_len=max_model_len,
            block_size=block_size,
        )
        selection_row = final_rows.get(key, {})
        original_selected = [int(item) for item in selection_row.get("original_selected_block_ids", [])]
        num_blocks = int(selection_row.get("num_blocks", len(block_texts)) or len(block_texts))
        all_block_ids = list(range(min(num_blocks, len(block_texts))))
        selected_set = {block_id for block_id in original_selected if 0 <= block_id < len(block_texts)}
        pruned_ids = [block_id for block_id in all_block_ids if block_id not in selected_set]
        query_keywords = extract_query_keywords(query_text)

        selected_texts = [block_texts[block_id] for block_id in sorted(selected_set)]
        pruned_overlap_pairs = [
            (block_id, lexical_overlap(block_texts[block_id], query_keywords)) for block_id in pruned_ids
        ]
        top_pruned = sorted(pruned_overlap_pairs, key=lambda item: (-item[1], item[0]))[:TOP_PRUNED_M]
        top_pruned_ids = [block_id for block_id, _ in top_pruned]
        top_pruned_texts = [block_texts[block_id] for block_id in top_pruned_ids]
        selected_overlaps = [lexical_overlap(text, query_keywords) for text in selected_texts]

        doc_counts = Counter(
            block_doc_proxy(block_id, block_texts[block_id], max(1, len(block_texts)))
            for block_id in selected_set
        )
        doc_entropy = entropy_from_counts(doc_counts)
        top_doc_mass = (max(doc_counts.values()) / sum(doc_counts.values())) if doc_counts else 0.0
        cutoff_margin, cutoff_percentile, near_cutoff_count, cutoff_reason = compute_cutoff_features(
            selection_row
        )
        original_row = prediction_rows.get(key, {}).get("sliminfer_original", {})
        dc_row = prediction_rows.get(key, {}).get("dc_sliminfer", {})
        budget_mismatch = any(
            parse_bool(value) is True
            for value in (original_row.get("budget_mismatch"), dc_row.get("budget_mismatch"))
        )
        fallback = any(
            parse_bool(value) is True
            for value in (original_row.get("fallback_used"), dc_row.get("fallback_used"))
        )
        original_entered = parse_bool(original_row.get("dc_entered_before_virtual_return")) is True

        original_score = to_float(comparison.get("original_score"))
        dc_score = to_float(comparison.get("dc_score"))
        score_delta = to_float(comparison.get("delta"))
        if score_delta is None and original_score is not None and dc_score is not None:
            score_delta = dc_score - original_score

        selected_query_overlap_mean = mean(selected_overlaps) or 0.0
        pruned_query_overlap_values = [overlap for _, overlap in top_pruned]
        pruned_query_overlap_max = max(pruned_query_overlap_values) if pruned_query_overlap_values else 0.0
        pruned_query_overlap_mean = mean(pruned_query_overlap_values) or 0.0
        selected_entity_coverage = coverage_ratio(selected_texts, query_keywords)
        pruned_entity_coverage = coverage_ratio(top_pruned_texts, query_keywords)

        row = {
            "task": task,
            "sample_index": sample_index,
            "sample_id": sample_id,
            "original_score": original_score,
            "dc_score": dc_score,
            "score_delta": score_delta,
            "selected_jaccard_vs_original": to_float(comparison.get("selected_jaccard")),
            "query_overlap_delta": to_float(dc_row.get("dc_query_overlap"))
            - to_float(dc_row.get("original_query_overlap"))
            if to_float(dc_row.get("dc_query_overlap")) is not None
            and to_float(dc_row.get("original_query_overlap")) is not None
            else None,
            "coverage_delta": to_float(dc_row.get("dc_coverage_bin_count"))
            - to_float(dc_row.get("original_coverage_bin_count"))
            if to_float(dc_row.get("dc_coverage_bin_count")) is not None
            and to_float(dc_row.get("original_coverage_bin_count")) is not None
            else None,
            "redundancy_delta": to_float(dc_row.get("dc_redundancy"))
            - to_float(dc_row.get("original_redundancy"))
            if to_float(dc_row.get("dc_redundancy")) is not None
            and to_float(dc_row.get("original_redundancy")) is not None
            else None,
            "cutoff_margin": cutoff_margin,
            "cutoff_margin_percentile": cutoff_percentile,
            "near_cutoff_candidate_count": near_cutoff_count,
            "cutoff_unavailable_reason": cutoff_reason,
            "original_doc_entropy": doc_entropy,
            "original_top_doc_mass": top_doc_mass,
            "original_neighbor_cluster_ratio": neighbor_cluster_ratio(original_selected),
            "pruned_query_overlap_max": pruned_query_overlap_max,
            "pruned_query_overlap_mean_topM": pruned_query_overlap_mean,
            "selected_query_overlap_mean": selected_query_overlap_mean,
            "selected_vs_pruned_query_overlap_gap": pruned_query_overlap_mean
            - selected_query_overlap_mean,
            "query_entity_count": len(query_keywords),
            "selected_entity_coverage": selected_entity_coverage,
            "pruned_entity_coverage_topM": pruned_entity_coverage,
            "entity_coverage_gap": pruned_entity_coverage - selected_entity_coverage,
            "evidence_dispersion_score": None,
            "selection_changed": parse_bool(comparison.get("selection_changed")),
            "budget_mismatch": budget_mismatch,
            "fallback": fallback,
            "original_entered_dc_selector": original_entered,
            "final_layer_id": selection_row.get("layer_id"),
            "num_blocks": num_blocks,
            "original_selected_count": len(selected_set),
            "pruned_count": len(pruned_ids),
            "top_pruned_M": TOP_PRUNED_M,
            "top_pruned_source": "query_overlap_proxy",
            "query_keywords": " ".join(query_keywords),
            "top_pruned_block_ids": " ".join(str(block_id) for block_id in top_pruned_ids),
            "prompt_token_block_count": len(block_texts),
            "prompt_text_hash_proxy_len": len(prompt_text),
        }
        row["dc_positive"] = (score_delta or 0.0) > 0
        row["dc_negative"] = (score_delta or 0.0) < 0
        row["hotpot_catastrophic_negative"] = task == "hotpotqa" and (score_delta or 0.0) <= -5
        row["musique_positive"] = task == "musique" and (score_delta or 0.0) > 0
        feature_rows.append(row)

    for source in (
        "original_doc_entropy",
        "entity_coverage_gap",
        "pruned_query_overlap_max",
        "near_cutoff_candidate_count",
    ):
        minmax_normalize(feature_rows, source, f"{source}_norm")
    for row in feature_rows:
        components = [
            row.get("original_doc_entropy_norm"),
            row.get("entity_coverage_gap_norm"),
            row.get("pruned_query_overlap_max_norm"),
            row.get("near_cutoff_candidate_count_norm"),
        ]
        available = [value for value in components if to_float(value) is not None]
        row["evidence_dispersion_score"] = sum(available) / len(available) if available else None
        row["evidence_dispersion_components_available"] = len(available)

    return feature_rows


def validate(data: Dict[str, Any], rows: List[Dict[str, Any]], missing: List[str]) -> List[str]:
    reasons = list(missing)
    if len(data["comparison_rows"]) != 20:
        reasons.append(f"expected 20 comparison rows, found {len(data['comparison_rows'])}")
    if len(data["prediction_rows"]) != 40:
        reasons.append(f"expected 40 prediction rows, found {len(data['prediction_rows'])}")
    if len(data["selection_rows"]) != 60:
        reasons.append(f"expected 60 consolidated selection audit rows, found {len(data['selection_rows'])}")
    if len(rows) != 20:
        reasons.append(f"expected 20 feature rows, found {len(rows)}")
    scores_original = score_table(rows, "original_score")
    scores_dc = score_table(rows, "dc_score")
    for summary in data["task_summary"]:
        task = str(summary.get("task", ""))
        expected_original = scores_original.get(task)
        expected_dc = scores_dc.get(task)
        observed_original = to_float(summary.get("original"))
        observed_dc = to_float(summary.get("dc"))
        if expected_original is not None and observed_original is not None and abs(expected_original - observed_original) > 1e-4:
            reasons.append(f"score mismatch {task}.original: {expected_original} vs {observed_original}")
        if expected_dc is not None and observed_dc is not None and abs(expected_dc - observed_dc) > 1e-4:
            reasons.append(f"score mismatch {task}.dc: {expected_dc} vs {observed_dc}")
    return reasons


def coverage_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    output = []
    total = len(rows)
    for feature in STRUCTURAL_FEATURES:
        available = sum(1 for row in rows if to_float(row.get(feature)) is not None)
        output.append(
            {
                "section": "coverage",
                "group": "all",
                "feature": feature,
                "count": total,
                "available": available,
                "mean": "",
                "median": "",
            }
        )
    return output


def group_stats(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups = {
        "DC positive": [row for row in rows if row["dc_positive"]],
        "DC negative": [row for row in rows if row["dc_negative"]],
        "Hotpot catastrophic negative": [row for row in rows if row["hotpot_catastrophic_negative"]],
        "MuSiQue positive": [row for row in rows if row["musique_positive"]],
    }
    output = coverage_rows(rows)
    for group_name, group_rows in groups.items():
        for feature in STRUCTURAL_FEATURES:
            output.append(
                {
                    "section": "group_stats",
                    "group": group_name,
                    "feature": feature,
                    "count": len(group_rows),
                    "available": sum(1 for row in group_rows if to_float(row.get(feature)) is not None),
                    "mean": mean(row.get(feature) for row in group_rows),
                    "median": median(row.get(feature) for row in group_rows),
                }
            )
    return output


def feature_correlations(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    output = []
    for feature in STRUCTURAL_FEATURES:
        output.append(
            {
                "feature": feature,
                "available": sum(1 for row in rows if to_float(row.get(feature)) is not None),
                "pearson_with_score_delta": pearson(
                    (row.get(feature) for row in rows),
                    (row.get("score_delta") for row in rows),
                ),
                "spearman_with_score_delta": spearman(
                    (row.get(feature) for row in rows),
                    (row.get("score_delta") for row in rows),
                ),
                "musique_positive_mean": mean(
                    row.get(feature) for row in rows if row["musique_positive"]
                ),
                "hotpot_catastrophic_negative_mean": mean(
                    row.get(feature) for row in rows if row["hotpot_catastrophic_negative"]
                ),
            }
        )
    return output


def feature_medians(rows: List[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    return {feature: median(row.get(feature) for row in rows) for feature in STRUCTURAL_FEATURES}


def safe_false(value: Any) -> bool:
    return parse_bool(value) is False


def base_safety(row: Dict[str, Any]) -> bool:
    return (
        safe_false(row.get("budget_mismatch"))
        and safe_false(row.get("fallback"))
        and safe_false(row.get("original_entered_dc_selector"))
        and parse_bool(row.get("selection_changed")) is True
    )


def replay_rules(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    med = feature_medians(rows)
    neighbor_median = med.get("original_neighbor_cluster_ratio")
    cutoff_median = med.get("cutoff_margin")
    near_cutoff_median = med.get("near_cutoff_candidate_count")
    dispersion_median = med.get("evidence_dispersion_score")
    original_scores = score_table(rows, "original_score")
    dc_scores = score_table(rows, "dc_score")

    def gt(row: Dict[str, Any], field: str, other: Any) -> bool:
        left = to_float(row.get(field))
        right = to_float(other)
        return left is not None and right is not None and left > right

    def ge(row: Dict[str, Any], field: str, other: Any) -> bool:
        left = to_float(row.get(field))
        right = to_float(other)
        return left is not None and right is not None and left >= right

    def le(row: Dict[str, Any], field: str, other: Any) -> bool:
        left = to_float(row.get(field))
        right = to_float(other)
        return left is not None and right is not None and left <= right

    rule_specs: List[Tuple[str, str, Callable[[Dict[str, Any]], bool]]] = [
        (
            "Rule 1: under-coverage repair",
            "entity_coverage_gap > 0; pruned_query_overlap_max > selected_query_overlap_mean; "
            "original_neighbor_cluster_ratio >= median; selected_jaccard_vs_original >= 0.30",
            lambda row: base_safety(row)
            and gt(row, "entity_coverage_gap", 0.0)
            and gt(row, "pruned_query_overlap_max", row.get("selected_query_overlap_mean"))
            and ge(row, "original_neighbor_cluster_ratio", neighbor_median)
            and ge(row, "selected_jaccard_vs_original", 0.30),
        ),
        (
            "Rule 2: unstable-cutoff repair",
            "cutoff_margin available and <= median; near_cutoff_candidate_count >= median; "
            "pruned_query_overlap_max > selected_query_overlap_mean; selected_jaccard_vs_original >= 0.30",
            lambda row: base_safety(row)
            and cutoff_median is not None
            and near_cutoff_median is not None
            and le(row, "cutoff_margin", cutoff_median)
            and ge(row, "near_cutoff_candidate_count", near_cutoff_median)
            and gt(row, "pruned_query_overlap_max", row.get("selected_query_overlap_mean"))
            and ge(row, "selected_jaccard_vs_original", 0.30),
        ),
        (
            "Rule 3: dispersed-evidence repair",
            "evidence_dispersion_score >= median; entity_coverage_gap >= 0; "
            "pruned_query_overlap_max > selected_query_overlap_mean; selected_jaccard_vs_original >= 0.30",
            lambda row: base_safety(row)
            and dispersion_median is not None
            and ge(row, "evidence_dispersion_score", dispersion_median)
            and ge(row, "entity_coverage_gap", 0.0)
            and gt(row, "pruned_query_overlap_max", row.get("selected_query_overlap_mean"))
            and ge(row, "selected_jaccard_vs_original", 0.30),
        ),
    ]

    output = []
    total = len(rows)
    hotpot_total = sum(1 for row in rows if row["hotpot_catastrophic_negative"])
    musique_positive_total = sum(1 for row in rows if row["musique_positive"])
    for name, description, predicate in rule_specs:
        triggered = [row for row in rows if predicate(row)]
        trigger_keys = {sample_key(row) for row in triggered}
        replayed = []
        for row in rows:
            use_dc = sample_key(row) in trigger_keys
            replayed.append({**row, "rule_score": row["dc_score"] if use_dc else row["original_score"], "trigger": use_dc})
        rule_scores = score_table(replayed, "rule_score")
        deltas = [
            (to_float(row.get("rule_score")) or 0.0) - (to_float(row.get("original_score")) or 0.0)
            for row in replayed
        ]
        hotpot_triggered = sum(
            1 for row in replayed if row["hotpot_catastrophic_negative"] and row["trigger"]
        )
        musique_retained = sum(1 for row in replayed if row["musique_positive"] and row["trigger"])
        output.append(
            {
                "rule": name,
                "rule_description": description,
                "diagnostic_median_original_neighbor_cluster_ratio": neighbor_median,
                "diagnostic_median_cutoff_margin": cutoff_median,
                "diagnostic_median_near_cutoff_candidate_count": near_cutoff_median,
                "diagnostic_median_evidence_dispersion_score": dispersion_median,
                "trigger_count": len(triggered),
                "trigger_sample_list": "; ".join(f"{row['task']}:{row['sample_index']}" for row in triggered),
                "hotpotqa_original": original_scores.get("hotpotqa"),
                "hotpotqa_rule": rule_scores.get("hotpotqa"),
                "hotpotqa_delta": rule_scores.get("hotpotqa", 0.0) - original_scores.get("hotpotqa", 0.0),
                "musique_original": original_scores.get("musique"),
                "musique_rule": rule_scores.get("musique"),
                "musique_delta": rule_scores.get("musique", 0.0) - original_scores.get("musique", 0.0),
                "macro_original": original_scores.get("macro"),
                "macro_rule": rule_scores.get("macro"),
                "macro_delta": rule_scores.get("macro", 0.0) - original_scores.get("macro", 0.0),
                "unconditional_dc_macro": dc_scores.get("macro"),
                "rule_vs_unconditional_dc_macro_delta": rule_scores.get("macro", 0.0) - dc_scores.get("macro", 0.0),
                "positive_count_vs_original": sum(1 for value in deltas if value > 0),
                "negative_count_vs_original": sum(1 for value in deltas if value < 0),
                "zero_count_vs_original": sum(1 for value in deltas if value == 0),
                "retained_musique_positive_count": musique_retained,
                "retained_musique_positive_total": musique_positive_total,
                "triggered_hotpot_catastrophic_negative_count": hotpot_triggered,
                "hotpot_catastrophic_negative_total": hotpot_total,
                "filtered_hotpot_catastrophic_negative_count": hotpot_total - hotpot_triggered,
                "non_all_reject": len(triggered) > 0,
                "non_all_trigger": len(triggered) < total,
                "better_than_original": rule_scores.get("macro", 0.0) > original_scores.get("macro", 0.0),
                "better_than_unconditional_dc": rule_scores.get("macro", 0.0) > dc_scores.get("macro", 0.0),
                "uses_task_gold_score_delta_as_gate": False,
            }
        )
    return output


def decide(
    rows: List[Dict[str, Any]],
    correlation_rows: List[Dict[str, Any]],
    rule_rows: List[Dict[str, Any]],
    bug_reasons: List[str],
) -> Tuple[str, List[str]]:
    if bug_reasons:
        return "BUG_STOP_FIX_REQUIRED", bug_reasons

    feature_signal_reasons = []
    for row in correlation_rows:
        pear = abs(to_float(row.get("pearson_with_score_delta")) or 0.0)
        spear = abs(to_float(row.get("spearman_with_score_delta")) or 0.0)
        if max(pear, spear) >= 0.35:
            feature_signal_reasons.append(f"{row['feature']} corr >= 0.35")
    for feature in ("entity_coverage_gap", "pruned_query_overlap_max", "original_top_doc_mass", "cutoff_margin"):
        mp = mean(row.get(feature) for row in rows if row["musique_positive"])
        hn = mean(row.get(feature) for row in rows if row["hotpot_catastrophic_negative"])
        all_values = [to_float(row.get(feature)) for row in rows]
        clean = [value for value in all_values if value is not None]
        if mp is not None and hn is not None and clean:
            span = max(clean) - min(clean)
            if span > 0 and abs(mp - hn) >= 0.25 * span:
                feature_signal_reasons.append(f"{feature} separates MuSiQue positive vs Hotpot catastrophic by >=25% span")

    go_rules = [
        row
        for row in rule_rows
        if parse_bool(row.get("non_all_reject")) is True
        and parse_bool(row.get("non_all_trigger")) is True
        and parse_bool(row.get("better_than_original")) is True
        and (to_float(row.get("hotpotqa_delta")) or 0.0) >= -2.0
        and int(row.get("retained_musique_positive_count") or 0) >= 1
    ]
    if feature_signal_reasons and go_rules:
        return "ADAPTIVE_DC_V3_FEATURE_GO", [
            f"feature signal: {feature_signal_reasons[0]}",
            f"GO rule: {go_rules[0]['rule']}",
        ]

    any_nontrivial_rule = any(
        parse_bool(row.get("non_all_reject")) is True and parse_bool(row.get("non_all_trigger")) is True
        for row in rule_rows
    )
    any_retains_musique = any(int(row.get("retained_musique_positive_count") or 0) >= 1 for row in rule_rows)
    any_triggers_hotpot = any(
        int(row.get("triggered_hotpot_catastrophic_negative_count") or 0) >= 1 for row in rule_rows
    )
    if feature_signal_reasons or (any_retains_musique and any_triggers_hotpot) or any_nontrivial_rule:
        reasons = []
        if feature_signal_reasons:
            reasons.append("; ".join(feature_signal_reasons[:3]))
        if any_retains_musique and any_triggers_hotpot:
            reasons.append("rules retain MuSiQue positives but still trigger Hotpot catastrophic negatives")
        if any_nontrivial_rule:
            reasons.append("at least one rule is non-all-reject and non-all-trigger")
        return "ADAPTIVE_DC_V3_WEAK_BUT_PROMISING", reasons

    if all(int(row.get("trigger_count") or 0) in {0, len(rows)} for row in rule_rows):
        return "ADAPTIVE_DC_V3_NO_GO_STOP", ["rules are all-reject or all-trigger"]
    return "ADAPTIVE_DC_V3_NO_GO_STOP", ["new structural features show no useful separability"]


def markdown_table(headers: List[str], rows: List[List[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def build_report(
    tiny_dir: Path,
    v2_dir: Path,
    output_dir: Path,
    rows: List[Dict[str, Any]],
    stats_rows: List[Dict[str, Any]],
    corr_rows: List[Dict[str, Any]],
    rule_rows: List[Dict[str, Any]],
    decision: str,
    decision_reasons: List[str],
    bug_reasons: List[str],
) -> str:
    original_scores = score_table(rows, "original_score")
    dc_scores = score_table(rows, "dc_score")
    score_ref = [
        [
            task,
            fmt(original_scores.get(task)),
            fmt(dc_scores.get(task)),
            fmt(dc_scores.get(task, 0.0) - original_scores.get(task, 0.0)),
        ]
        for task in ("hotpotqa", "musique", "macro")
    ]
    coverage = [row for row in stats_rows if row["section"] == "coverage"]
    coverage_table = [[row["feature"], f"{row['available']}/{row['count']}"] for row in coverage]
    group_table = [
        [
            row["group"],
            row["feature"],
            row["count"],
            row["available"],
            fmt(row["mean"]),
            fmt(row["median"]),
        ]
        for row in stats_rows
        if row["section"] == "group_stats"
    ]
    corr_table = [
        [
            row["feature"],
            row["available"],
            fmt(row["pearson_with_score_delta"]),
            fmt(row["spearman_with_score_delta"]),
            fmt(row["musique_positive_mean"]),
            fmt(row["hotpot_catastrophic_negative_mean"]),
        ]
        for row in corr_rows
    ]
    replay_table = [
        [
            row["rule"],
            row["trigger_count"],
            row["trigger_sample_list"] or "[]",
            fmt(row["hotpotqa_rule"]),
            fmt(row["hotpotqa_delta"]),
            fmt(row["musique_rule"]),
            fmt(row["musique_delta"]),
            fmt(row["macro_rule"]),
            fmt(row["macro_delta"]),
            row["retained_musique_positive_count"],
            row["triggered_hotpot_catastrophic_negative_count"],
            row["filtered_hotpot_catastrophic_negative_count"],
            row["better_than_original"],
            row["better_than_unconditional_dc"],
        ]
        for row in rule_rows
    ]
    cutoff_available = sum(1 for row in rows if to_float(row.get("cutoff_margin")) is not None)
    low_jaccard_rows = [
        row
        for row in rows
        if to_float(row.get("selected_jaccard_vs_original")) is not None
        and to_float(row.get("selected_jaccard_vs_original")) < 0.50
    ]
    high_dispersion = median(row.get("evidence_dispersion_score") for row in rows)
    low_jaccard_high_disp_positive = sum(
        1
        for row in low_jaccard_rows
        if high_dispersion is not None
        and (to_float(row.get("evidence_dispersion_score")) or -1) >= high_dispersion
        and (to_float(row.get("score_delta")) or 0) > 0
    )
    low_jaccard_high_disp_total = sum(
        1
        for row in low_jaccard_rows
        if high_dispersion is not None
        and (to_float(row.get("evidence_dispersion_score")) or -1) >= high_dispersion
    )

    lines = [
        "# Adaptive DC-SlimInfer Router-v3 Feature Audit",
        "",
        f"- Tiny input directory: `{tiny_dir}`",
        f"- Router-v2 input directory: `{v2_dir}`",
        f"- Output directory: `{output_dir}`",
        "- Generation run: **No**",
        "- Scorer run: **No**",
        "- 64/full run: **No**",
        "- Threshold sweep: **No**",
        "- DC selector modified: **No**",
        "",
        "## Reference Quality",
        markdown_table(["task", "original", "dc", "delta"], score_ref),
        "",
        "## 1. Field Coverage",
        markdown_table(["feature", "available/total"], coverage_table),
        "",
        (
            "Cutoff features are unavailable because the input artifacts do not contain "
            "complete selected/pruned score details."
            if cutoff_available == 0
            else f"Cutoff features available for {cutoff_available}/{len(rows)} samples."
        ),
        "",
        "Text-derived top-pruned features use fixed Qwen block reconstruction and a query-overlap "
        "proxy to rank pruned candidates; no model generation or scorer is used.",
        "",
        "## 2. Positive/Negative Group Analysis",
        markdown_table(["group", "feature", "count", "available", "mean", "median"], group_table),
        "",
        "## 3. Feature Separability",
        markdown_table(
            [
                "feature",
                "available",
                "pearson",
                "spearman",
                "MuSiQue positive mean",
                "Hotpot catastrophic mean",
            ],
            corr_table,
        ),
        "",
        "- MuSiQue positive vs Hotpot catastrophic checks are diagnostic only; task labels are not used by rules.",
        (
            "- Low selected_jaccard + high evidence_dispersion positive ratio: "
            f"{low_jaccard_high_disp_positive}/{low_jaccard_high_disp_total}"
        ),
        "",
        "## 4. Router-v3 Rule Replay",
        "Rule medians are diagnostic medians computed on this fixed 20-sample set. They are not final method thresholds.",
        "",
        markdown_table(
            [
                "rule",
                "triggers",
                "trigger samples",
                "HotpotQA rule",
                "HotpotQA delta",
                "MuSiQue rule",
                "MuSiQue delta",
                "macro rule",
                "macro delta",
                "MuSiQue positives retained",
                "Hotpot catastrophic triggered",
                "Hotpot catastrophic filtered",
                "better than original",
                "better than unconditional DC",
            ],
            replay_table,
        ),
        "",
        "## 5. Decision",
        f"- Decision: **{decision}**",
        f"- Reasons: {decision_reasons}",
    ]
    if bug_reasons:
        lines.extend(["", "### Bug Reasons", *[f"- {reason}" for reason in bug_reasons]])
    lines.extend(["", REPORT_FOOTER])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tiny-dir", type=Path, default=DEFAULT_TINY_DIR)
    parser.add_argument("--v2-dir", type=Path, default=DEFAULT_V2_DIR)
    parser.add_argument("--output-parent", type=Path, default=DEFAULT_OUTPUT_PARENT)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_parent / f"adaptive_dc_router_v3_feature_audit_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)

    data, missing = load_inputs(args.tiny_dir, args.v2_dir)
    config = data.get("config") or {}
    config_args = config.get("args", {}) if isinstance(config, dict) else {}
    tokenizer_path = config_args.get("tokenizer_path") or config_args.get("model_path")
    longbench_dir = Path(config_args.get("longbench_dir", "/home/yu-bai/Desktop/jiaqizhang/HeadKV/data/LongBench"))
    max_model_len = int(config_args.get("max_model_len", 32768))
    block_size = int(config_args.get("block_size", 64))
    dataset2prompt = read_json(REPO_ROOT / "configs" / "dataset2prompt.json")
    if not isinstance(dataset2prompt, dict):
        raise RuntimeError("configs/dataset2prompt.json missing or invalid")

    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
            local_files_only=True,
        )
    except Exception as exc:  # pragma: no cover - reported in output.
        manifest = {
            "decision": "BUG_STOP_FIX_REQUIRED",
            "decision_reasons": [f"tokenizer load failed: {exc}"],
            "output_dir": str(output_dir),
        }
        (output_dir / "summary.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        (output_dir / "adaptive_dc_v3_decision.md").write_text(
            "# Adaptive DC Router-v3 Decision\n\n"
            "- Decision: **BUG_STOP_FIX_REQUIRED**\n"
            f"- Reasons: tokenizer load failed: {exc}\n\n"
            f"{REPORT_FOOTER}\n",
            encoding="utf-8",
        )
        print(json.dumps(manifest, indent=2))
        return 2

    feature_rows = build_base_rows(
        data=data,
        tokenizer=tokenizer,
        longbench_dir=longbench_dir,
        dataset2prompt=dataset2prompt,
        max_model_len=max_model_len,
        block_size=block_size,
    )
    bug_reasons = validate(data, feature_rows, missing)
    stats_rows = group_stats(feature_rows)
    corr_rows = feature_correlations(feature_rows)
    rule_rows = replay_rules(feature_rows)
    decision, decision_reasons = decide(feature_rows, corr_rows, rule_rows, bug_reasons)

    feature_fields = [
        "task",
        "sample_index",
        "sample_id",
        "original_score",
        "dc_score",
        "score_delta",
        "selected_jaccard_vs_original",
        "query_overlap_delta",
        "coverage_delta",
        "redundancy_delta",
        *STRUCTURAL_FEATURES,
        "cutoff_unavailable_reason",
        "evidence_dispersion_components_available",
        "selection_changed",
        "budget_mismatch",
        "fallback",
        "original_entered_dc_selector",
        "final_layer_id",
        "num_blocks",
        "original_selected_count",
        "pruned_count",
        "top_pruned_M",
        "top_pruned_source",
        "query_keywords",
        "top_pruned_block_ids",
        "dc_positive",
        "dc_negative",
        "hotpot_catastrophic_negative",
        "musique_positive",
    ]
    write_csv(output_dir / "adaptive_dc_v3_feature_table.csv", feature_rows, feature_fields)
    write_csv(
        output_dir / "adaptive_dc_v3_group_stats.csv",
        stats_rows,
        ["section", "group", "feature", "count", "available", "mean", "median"],
    )
    write_csv(
        output_dir / "adaptive_dc_v3_feature_correlation.csv",
        corr_rows,
        [
            "feature",
            "available",
            "pearson_with_score_delta",
            "spearman_with_score_delta",
            "musique_positive_mean",
            "hotpot_catastrophic_negative_mean",
        ],
    )
    write_csv(
        output_dir / "adaptive_dc_v3_rule_replay.csv",
        rule_rows,
        [
            "rule",
            "rule_description",
            "diagnostic_median_original_neighbor_cluster_ratio",
            "diagnostic_median_cutoff_margin",
            "diagnostic_median_near_cutoff_candidate_count",
            "diagnostic_median_evidence_dispersion_score",
            "trigger_count",
            "trigger_sample_list",
            "hotpotqa_original",
            "hotpotqa_rule",
            "hotpotqa_delta",
            "musique_original",
            "musique_rule",
            "musique_delta",
            "macro_original",
            "macro_rule",
            "macro_delta",
            "unconditional_dc_macro",
            "rule_vs_unconditional_dc_macro_delta",
            "positive_count_vs_original",
            "negative_count_vs_original",
            "zero_count_vs_original",
            "retained_musique_positive_count",
            "retained_musique_positive_total",
            "triggered_hotpot_catastrophic_negative_count",
            "hotpot_catastrophic_negative_total",
            "filtered_hotpot_catastrophic_negative_count",
            "non_all_reject",
            "non_all_trigger",
            "better_than_original",
            "better_than_unconditional_dc",
            "uses_task_gold_score_delta_as_gate",
        ],
    )
    report = build_report(
        tiny_dir=args.tiny_dir,
        v2_dir=args.v2_dir,
        output_dir=output_dir,
        rows=feature_rows,
        stats_rows=stats_rows,
        corr_rows=corr_rows,
        rule_rows=rule_rows,
        decision=decision,
        decision_reasons=decision_reasons,
        bug_reasons=bug_reasons,
    )
    (output_dir / "adaptive_dc_v3_report.md").write_text(report, encoding="utf-8")
    decision_text = "\n".join(
        [
            "# Adaptive DC Router-v3 Decision",
            "",
            f"- Decision: **{decision}**",
            f"- Reasons: {decision_reasons}",
            f"- Output directory: `{output_dir}`",
            "",
            REPORT_FOOTER,
            "",
        ]
    )
    (output_dir / "adaptive_dc_v3_decision.md").write_text(decision_text, encoding="utf-8")
    manifest = {
        "decision": decision,
        "decision_reasons": decision_reasons,
        "bug_reasons": bug_reasons,
        "output_dir": str(output_dir),
        "tiny_dir": str(args.tiny_dir),
        "v2_dir": str(args.v2_dir),
        "tokenizer_path": tokenizer_path,
        "longbench_dir": str(longbench_dir),
        "rule_rows": rule_rows,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if decision != "BUG_STOP_FIX_REQUIRED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
