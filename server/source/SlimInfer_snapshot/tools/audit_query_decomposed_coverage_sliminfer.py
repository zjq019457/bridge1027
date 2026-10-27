#!/usr/bin/env python3
"""Offline selection audit for Query-Decomposed Coverage SlimInfer (qdc_sliminfer).

Reads held-out64 sliminfer_original selections, reconstructs prompt blocks, and
proposes budget-neutral coverage repair. No generation, no scorer, no gold.
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
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.cache_utils import _dc_selected_redundancy, _dc_token_jaccard, _gate_content_tokens, _gate_entity_like_units

DEFAULT_HELDOUT_DIR = Path("results/adaptive_dc_v3_heldout64_diagnostic_20260708_073628")
DEFAULT_LONGBENCH_DIR = Path("/home/yu-bai/Desktop/jiaqizhang/HeadKV/data/LongBench")
DEFAULT_MODEL_PATH = Path(
    "/home/yu-bai/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/"
    "snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
)

GATE_LAYER = 26
BLOCK_SIZE = 64
MAX_MODEL_LEN = 32768
TASKS = ["hotpotqa", "musique"]
TINY_EXCLUDE = {
    "hotpotqa": {0, 1, 2, 3, 4, 5, 41, 56, 57, 117},
    "musique": {0, 1, 2, 3, 4, 20, 50, 54, 56, 134},
}

RELATION_WORDS = {
    "where", "when", "who", "whom", "whose", "which", "what", "how",
    "born", "died", "directed", "founded", "located", "nationality", "capital",
    "headquartered", "married", "parent", "child", "author", "written",
    "released", "published", "formed", "member", "leader", "president",
    "governor", "mayor", "country", "city", "state", "province", "county",
}

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of",
    "with", "by", "from", "as", "is", "was", "are", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could", "should",
    "may", "might", "must", "shall", "can", "need", "dare", "ought", "used",
    "that", "this", "these", "those", "there", "their", "they", "them", "then",
    "than", "such", "into", "through", "during", "before", "after", "above",
    "below", "between", "under", "again", "further", "once", "here", "when",
    "where", "why", "how", "all", "each", "few", "more", "most", "other", "some",
    "any", "both", "same", "own", "only", "also", "very", "just", "about",
    "many", "much", "first", "last", "new", "old", "one", "two", "three",
}

Key = Tuple[str, int]


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _serialize(row.get(field, "")) for field in fields})


def _serialize(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, bool):
        return value
    return value


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def heldout_indices(task: str, count: int = 64) -> List[int]:
    selected: List[int] = []
    for idx in range(200):
        if idx in TINY_EXCLUDE[task]:
            continue
        selected.append(idx)
        if len(selected) >= count:
            break
    return selected


def heldout_keys() -> List[Key]:
    return [(task, idx) for task in TASKS for idx in heldout_indices(task)]


def load_tokenizer_and_prompts(model_path: Path) -> Tuple[Any, Dict[str, str]]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    dataset2prompt = json.loads((REPO_ROOT / "configs" / "dataset2prompt.json").read_text(encoding="utf-8"))
    return tokenizer, dataset2prompt


def load_samples(longbench_dir: Path, task: str) -> List[Dict[str, Any]]:
    return [
        json.loads(line)
        for line in (longbench_dir / f"{task}.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_chat_prompt(tokenizer: Any, prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )


def build_prompt_blocks(
    tokenizer: Any, prompt_format: str, sample: Dict[str, Any], dataset: str,
    max_model_len: int, block_size: int,
) -> Tuple[str, List[str], List[int]]:
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
    blocks: List[str] = []
    token_lengths: List[int] = []
    for start in range(0, len(chat_ids), block_size):
        chunk = chat_ids[start : start + block_size]
        blocks.append(
            tokenizer.decode(chunk, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        )
        token_lengths.append(len(chunk))
    query_text = str(sample.get("input", sample.get("question", sample.get("query", ""))))
    return query_text, blocks, token_lengths


def extract_query_components(query: str) -> List[str]:
    components: List[str] = []
    seen: Set[str] = set()

    def add(kind: str, value: str) -> None:
        text = " ".join(str(value).strip().lower().split())
        if not text or len(text) > 120:
            return
        key = f"{kind}:{text}"
        if key not in seen:
            seen.add(key)
            components.append(key)

    for entity in _gate_entity_like_units(query):
        add("entity", entity)

    for match in re.findall(r"\b\d{1,4}(?:,\d{3})*(?:\.\d+)?\b", query):
        add("number", match.replace(",", ""))

    for match in re.findall(r"\b(?:19|20)\d{2}\b", query):
        add("date", match)

    for token in re.findall(r"[A-Za-z][A-Za-z0-9'-]*", query.lower()):
        if len(token) >= 4 and token not in STOPWORDS and not token.isdigit():
            add("noun", token)

    for word in RELATION_WORDS:
        if re.search(rf"\b{re.escape(word)}\b", query.lower()):
            add("constraint", word)

    for match in re.findall(
        r"\b(?:where|when|who|which|what|how)\s+(?:was|is|are|were|did|does|do)\b",
        query.lower(),
    ):
        add("wh", match.strip())

    return components


def component_in_text(component: str, text: str) -> bool:
    kind, _, value = component.partition(":")
    haystack = str(text or "").lower()
    if not value:
        return False
    if kind in ("entity", "noun", "constraint", "wh"):
        if value in haystack:
            return True
        tokens = _gate_content_tokens(value)
        if tokens and tokens.issubset(_gate_content_tokens(haystack)):
            return True
        return False
    if kind in ("number", "date"):
        return value in haystack.replace(",", "")
    return value in haystack


def component_match_score(component: str, text: str) -> float:
    if not component_in_text(component, text):
        return 0.0
    kind, _, value = component.partition(":")
    if kind == "entity":
        return 1.0 + min(0.5, len(value.split()) * 0.1)
    if kind in ("number", "date"):
        return 0.9
    if kind == "constraint":
        return 0.6
    return 0.5


def covered_components(block_ids: Sequence[int], block_texts: Sequence[str], components: Sequence[str]) -> Set[str]:
    covered: Set[str] = set()
    for block_id in block_ids:
        if not (0 <= int(block_id) < len(block_texts)):
            continue
        text = block_texts[int(block_id)]
        for component in components:
            if component_in_text(component, text):
                covered.add(component)
    return covered


def block_token_map(block_texts: Sequence[str]) -> Dict[int, Set[str]]:
    return {idx: _gate_content_tokens(text) for idx, text in enumerate(block_texts)}


def block_query_overlaps(block_texts: Sequence[str], query: str) -> Dict[int, float]:
    query_tokens = _gate_content_tokens(query)
    if not query_tokens:
        return {idx: 0.0 for idx in range(len(block_texts))}
    overlaps: Dict[int, float] = {}
    for idx, text in enumerate(block_texts):
        tokens = _gate_content_tokens(text)
        if not tokens:
            overlaps[idx] = 0.0
            continue
        overlaps[idx] = len(tokens.intersection(query_tokens)) / len(query_tokens)
    return overlaps


def marginal_component_contribution(
    block_id: int, selected: Set[int], block_texts: Sequence[str], components: Sequence[str],
) -> int:
    if block_id not in selected:
        return 0
    before = covered_components(selected, block_texts, components)
    after = covered_components(selected - {block_id}, block_texts, components)
    return len(before - after)


def removal_score(
    block_id: int,
    selected: Set[int],
    token_map: Dict[int, Set[str]],
    query_overlaps: Dict[int, float],
    block_texts: Sequence[str],
    components: Sequence[str],
) -> float:
    contrib = marginal_component_contribution(block_id, selected, block_texts, components)
    redundancies = [
        _dc_token_jaccard(token_map.get(block_id, set()), token_map.get(other, set()))
        for other in selected
        if other != block_id
    ]
    redundancy = max(redundancies) if redundancies else 0.0
    query_overlap = float(query_overlaps.get(block_id, 0.0))
    return redundancy - 0.35 * contrib - 0.25 * query_overlap


def selected_jaccard(left: Set[int], right: Set[int]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def token_count(selected: Set[int], token_lengths: Sequence[int]) -> int:
    return sum(int(token_lengths[block_id]) for block_id in selected if 0 <= block_id < len(token_lengths))


def preview_text(block_texts: Sequence[str], block_ids: Sequence[int], limit: int = 120) -> str:
    parts = []
    for block_id in block_ids[:3]:
        if 0 <= block_id < len(block_texts):
            snippet = " ".join(block_texts[block_id].split())
            parts.append(snippet[:limit])
    return " | ".join(parts)


def apply_qdc_repair(
    original_selected: Set[int],
    block_texts: Sequence[str],
    token_lengths: Sequence[int],
    query_components: Sequence[str],
    query_text: str,
    max_swaps: int = 8,
) -> Dict[str, Any]:
    num_blocks = len(block_texts)
    all_ids = set(range(num_blocks))
    selected = {int(x) for x in original_selected if 0 <= int(x) < num_blocks}
    if not selected:
        return {
            "qdc_selected": set(),
            "removed_block_ids": [],
            "added_block_ids": [],
            "fallback_used": True,
            "qdc_triggered": False,
            "qdc_trigger_reason": "empty_original_selection",
        }

    token_map = block_token_map(block_texts)
    query_overlaps = block_query_overlaps(block_texts, query_text)
    pruned = all_ids - selected
    target_keep = len(selected)
    original_token_count = token_count(selected, token_lengths)

    original_cov = covered_components(selected, block_texts, query_components)
    missing = [comp for comp in query_components if comp not in original_cov]

    if not query_components:
        return {
            "qdc_selected": set(selected),
            "removed_block_ids": [],
            "added_block_ids": [],
            "fallback_used": True,
            "qdc_triggered": False,
            "qdc_trigger_reason": "no_query_components",
        }

    if not missing:
        return {
            "qdc_selected": set(selected),
            "removed_block_ids": [],
            "added_block_ids": [],
            "fallback_used": False,
            "qdc_triggered": False,
            "qdc_trigger_reason": "full_component_coverage",
        }

    removed: List[int] = []
    added: List[int] = []
    swaps = 0
    fallback_reason = ""

    while missing and swaps < max_swaps:
        best_pair: Optional[Tuple[int, int, str, float]] = None
        for comp in missing:
            priority = 0.0
            if comp.startswith("entity:"):
                priority += 1.0
            elif comp.startswith(("number:", "date:")):
                priority += 0.8
            elif comp.startswith("constraint:"):
                priority += 0.5

            candidate_scores: List[Tuple[float, int]] = []
            for block_id in pruned:
                score = component_match_score(comp, block_texts[block_id])
                if score > 0:
                    candidate_scores.append((score + priority, block_id))
            if not candidate_scores:
                continue
            candidate_scores.sort(reverse=True)
            cand_id = candidate_scores[0][1]

            removal_candidates = sorted(
                (
                    removal_score(block_id, selected, token_map, query_overlaps, block_texts, query_components),
                    block_id,
                )
                for block_id in selected
                if marginal_component_contribution(block_id, selected, block_texts, query_components) == 0
                or removal_score(block_id, selected, token_map, query_overlaps, block_texts, query_components) > 0.15
            )
            if not removal_candidates:
                removal_candidates = sorted(
                    (
                        removal_score(block_id, selected, token_map, query_overlaps, block_texts, query_components),
                        block_id,
                    )
                    for block_id in selected
                )
            victim_id = removal_candidates[-1][1]

            if token_lengths[victim_id] != token_lengths[cand_id]:
                continue

            new_selected = (selected - {victim_id}) | {cand_id}
            if len(new_selected) != target_keep:
                continue
            new_cov = covered_components(new_selected, block_texts, query_components)
            gain = len(new_cov) - len(covered_components(selected, block_texts, query_components))
            if comp not in new_cov or gain <= 0:
                continue
            pair_score = gain + candidate_scores[0][0]
            if best_pair is None or pair_score > best_pair[3]:
                best_pair = (victim_id, cand_id, comp, pair_score)

        if best_pair is None:
            fallback_reason = "no_gainful_swap_for_missing_components"
            break

        victim_id, cand_id, _, _ = best_pair
        selected = (selected - {victim_id}) | {cand_id}
        pruned = all_ids - selected
        removed.append(victim_id)
        added.append(cand_id)
        swaps += 1
        missing = [comp for comp in query_components if comp not in covered_components(selected, block_texts, query_components)]

    qdc_cov = covered_components(selected, block_texts, query_components)
    coverage_gain = len(qdc_cov) - len(original_cov)
    changed = set(removed) or set(added)

    if not changed:
        reason = fallback_reason or "no_budget_neutral_repair_found"
        return {
            "qdc_selected": set(original_selected),
            "removed_block_ids": [],
            "added_block_ids": [],
            "fallback_used": True,
            "qdc_triggered": False,
            "qdc_trigger_reason": reason,
            "coverage_gain": 0,
            "budget_preserved": True,
            "qdc_token_count": original_token_count,
            "original_token_count": original_token_count,
        }

    qdc_token_count = token_count(selected, token_lengths)
    budget_preserved = len(selected) == target_keep and qdc_token_count == original_token_count

    if not budget_preserved:
        return {
            "qdc_selected": set(original_selected),
            "removed_block_ids": [],
            "added_block_ids": [],
            "fallback_used": True,
            "qdc_triggered": False,
            "qdc_trigger_reason": "budget_not_preserved_reverted_to_original",
            "coverage_gain": 0,
            "budget_preserved": False,
            "qdc_token_count": original_token_count,
            "original_token_count": original_token_count,
        }

    return {
        "qdc_selected": selected,
        "removed_block_ids": removed,
        "added_block_ids": added,
        "fallback_used": False,
        "qdc_triggered": coverage_gain > 0,
        "qdc_trigger_reason": (
            f"coverage_repair_gain={coverage_gain}; swaps={swaps}"
            if coverage_gain > 0
            else "repair_completed_without_coverage_gain"
        ),
        "coverage_gain": coverage_gain,
        "budget_preserved": True,
        "qdc_token_count": qdc_token_count,
        "original_token_count": original_token_count,
    }


def load_original_selection(heldout_dir: Path, keys: Set[Key]) -> Dict[Key, Dict[str, Any]]:
    selected: Dict[Key, Dict[str, Any]] = {}
    for row in read_jsonl(
        heldout_dir / "run_adaptive_dc_sliminfer_v3" / "audit" / "dc_runtime_selection_audit.jsonl"
    ):
        if str(row.get("phase")) != "prefill" or int(row.get("layer_id", -1)) != GATE_LAYER:
            continue
        key = (str(row.get("task")), int(row.get("sample_index", -1)))
        if key in keys and key not in selected:
            selected[key] = row
    return selected


def percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = (len(ordered) - 1) * pct / 100.0
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return ordered[lo]
    weight = rank - lo
    return ordered[lo] * (1 - weight) + ordered[hi] * weight


def decide_audit(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    if total == 0:
        return {"decision": "NO-GO", "reasons": ["no_rows"]}

    budget_violations = sum(1 for row in rows if not parse_bool(row.get("budget_preserved")))
    fallback_count = sum(1 for row in rows if parse_bool(row.get("fallback_used")))
    trigger_count = sum(1 for row in rows if parse_bool(row.get("qdc_triggered")))
    feature_coverage = sum(
        1 for row in rows
        if int(row.get("num_query_components", 0) or 0) > 0 and parse_bool(row.get("block_text_available", True))
    )

    orig_cov = [float(row["original_component_coverage"]) for row in rows]
    qdc_cov = [float(row["qdc_component_coverage"]) for row in rows]
    cov_delta = [float(row["coverage_delta"]) for row in rows]
    jaccards = [float(row["selected_jaccard_vs_original"]) for row in rows]
    redundancy_delta = [float(row["redundancy_delta"]) for row in rows]

    mean_cov_gain = statistics.mean(cov_delta)
    mean_redundancy_delta = statistics.mean(redundancy_delta)
    median_jaccard = statistics.median(jaccards)

    reasons: List[str] = []
    if budget_violations > 0:
        reasons.append(f"budget_violation_count={budget_violations}")
    if fallback_count == total:
        reasons.append("fallback_all_samples")
    if trigger_count == 0:
        reasons.append("trigger_count=0")
    if trigger_count == total:
        reasons.append("trigger_count=all")
    if mean_cov_gain <= 0.01:
        reasons.append(f"mean_coverage_delta={mean_cov_gain:.4f}")
    if mean_redundancy_delta > 0.08:
        reasons.append(f"mean_redundancy_delta={mean_redundancy_delta:.4f}")
    if median_jaccard < 0.35:
        reasons.append(f"median_jaccard={median_jaccard:.3f}")

    go = (
        budget_violations == 0
        and fallback_count < total
        and 0 < trigger_count < total
        and mean_cov_gain >= 0.03
        and mean_redundancy_delta <= 0.08
        and median_jaccard >= 0.35
        and feature_coverage == total
    )

    if go:
        return {
            "decision": "GO",
            "reasons": [
                f"trigger_count={trigger_count}/{total}",
                f"mean_coverage_delta={mean_cov_gain:+.4f}",
                f"median_jaccard={median_jaccard:.3f}",
                f"budget_violation_count=0",
            ],
            "worth_tiny_quality_smoke": True,
        }

    return {
        "decision": "NO-GO",
        "reasons": reasons or [f"trigger_count={trigger_count}/{total}", f"mean_coverage_delta={mean_cov_gain:+.4f}"],
        "worth_tiny_quality_smoke": trigger_count > 0 and mean_cov_gain > 0 and budget_violations == 0,
    }


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "y")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heldout_dir", type=Path, default=DEFAULT_HELDOUT_DIR)
    parser.add_argument("--longbench_dir", type=Path, default=DEFAULT_LONGBENCH_DIR)
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output_root", type=Path, default=Path("results"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    heldout_dir = (REPO_ROOT / args.heldout_dir).resolve()
    output_dir = (
        REPO_ROOT / args.output_root / f"qdc_sliminfer_selection_audit_{datetime.now():%Y%m%d_%H%M%S}"
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    keys = heldout_keys()
    key_set = set(keys)
    selection_by_key = load_original_selection(heldout_dir, key_set)
    if len(selection_by_key) != 128:
        raise RuntimeError(f"BUG_STOP: selection rows={len(selection_by_key)}/128")

    tokenizer, dataset2prompt = load_tokenizer_and_prompts(args.model_path)
    samples_cache = {task: load_samples(args.longbench_dir, task) for task in TASKS}
    block_cache: Dict[Key, Tuple[str, List[str], List[int]]] = {}

    audit_rows: List[Dict[str, Any]] = []

    for task, sample_index in keys:
        key = (task, sample_index)
        sel = selection_by_key[key]
        if key not in block_cache:
            sample = samples_cache[task][sample_index]
            block_cache[key] = build_prompt_blocks(
                tokenizer, dataset2prompt[task], sample, task, MAX_MODEL_LEN, BLOCK_SIZE
            )
        query_text, block_texts, token_lengths = block_cache[key]
        original_selected = {int(x) for x in sel.get("original_selected_block_ids", [])}
        query_components = extract_query_components(query_text)

        token_map = block_token_map(block_texts)
        query_overlaps = block_query_overlaps(block_texts, query_text)

        original_cov_set = covered_components(original_selected, block_texts, query_components)
        missing_before = [comp for comp in query_components if comp not in original_cov_set]

        repair = apply_qdc_repair(
            original_selected=original_selected,
            block_texts=block_texts,
            token_lengths=token_lengths,
            query_components=query_components,
            query_text=query_text,
        )
        qdc_selected = set(repair["qdc_selected"])
        qdc_cov_set = covered_components(qdc_selected, block_texts, query_components)
        missing_after = [comp for comp in query_components if comp not in qdc_cov_set]

        num_components = len(query_components)
        original_coverage = len(original_cov_set) / num_components if num_components else 1.0
        qdc_coverage = len(qdc_cov_set) / num_components if num_components else 1.0

        original_redundancy = _dc_selected_redundancy(original_selected, token_map)
        qdc_redundancy = _dc_selected_redundancy(qdc_selected, token_map)
        original_query_overlap = float(
            sum(query_overlaps.get(block_id, 0.0) for block_id in original_selected) / max(1, len(original_selected))
        )
        qdc_query_overlap = float(
            sum(query_overlaps.get(block_id, 0.0) for block_id in qdc_selected) / max(1, len(qdc_selected))
        )

        original_token_count = token_count(original_selected, token_lengths)
        qdc_token_count = token_count(qdc_selected, token_lengths)

        row = {
            "task": task,
            "sample_index": sample_index,
            "sample_id": sel.get("sample_id", ""),
            "query_components": query_components,
            "num_query_components": num_components,
            "original_component_coverage": original_coverage,
            "qdc_component_coverage": qdc_coverage,
            "coverage_delta": qdc_coverage - original_coverage,
            "missing_components_before": missing_before,
            "missing_components_after": missing_after,
            "selected_jaccard_vs_original": selected_jaccard(original_selected, qdc_selected),
            "removed_block_ids": repair["removed_block_ids"],
            "added_block_ids": repair["added_block_ids"],
            "removed_block_text_preview": preview_text(block_texts, repair["removed_block_ids"]),
            "added_block_text_preview": preview_text(block_texts, repair["added_block_ids"]),
            "original_redundancy": original_redundancy,
            "qdc_redundancy": qdc_redundancy,
            "redundancy_delta": qdc_redundancy - original_redundancy,
            "original_query_overlap": original_query_overlap,
            "qdc_query_overlap": qdc_query_overlap,
            "query_overlap_delta": qdc_query_overlap - original_query_overlap,
            "budget_preserved": bool(repair.get("budget_preserved", len(qdc_selected) == len(original_selected) and qdc_token_count == original_token_count)),
            "selected_token_count_original": original_token_count,
            "selected_token_count_qdc": qdc_token_count,
            "over_budget": qdc_token_count > original_token_count,
            "fallback_used": bool(repair.get("fallback_used", False)),
            "qdc_triggered": bool(repair.get("qdc_triggered", False)),
            "qdc_trigger_reason": str(repair.get("qdc_trigger_reason", "")),
            "block_text_available": bool(block_texts),
            "gold_answer_used": False,
            "task_label_used": False,
        }
        audit_rows.append(row)

    decision = decide_audit(audit_rows)

    audit_fields = [
        "task", "sample_index", "sample_id",
        "query_components", "num_query_components",
        "original_component_coverage", "qdc_component_coverage", "coverage_delta",
        "missing_components_before", "missing_components_after",
        "selected_jaccard_vs_original",
        "removed_block_ids", "added_block_ids",
        "removed_block_text_preview", "added_block_text_preview",
        "original_redundancy", "qdc_redundancy", "redundancy_delta",
        "original_query_overlap", "qdc_query_overlap", "query_overlap_delta",
        "budget_preserved", "selected_token_count_original", "selected_token_count_qdc",
        "over_budget", "fallback_used", "qdc_triggered", "qdc_trigger_reason",
    ]
    write_csv(output_dir / "qdc_selection_audit.csv", audit_rows, audit_fields)

    group_rows: List[Dict[str, Any]] = []
    for task in TASKS + ["all"]:
        subset = audit_rows if task == "all" else [row for row in audit_rows if row["task"] == task]
        if not subset:
            continue
        group_rows.append({
            "group": task,
            "sample_count": len(subset),
            "feature_coverage": sum(1 for row in subset if int(row["num_query_components"]) > 0),
            "qdc_trigger_count": sum(1 for row in subset if row["qdc_triggered"]),
            "fallback_count": sum(1 for row in subset if row["fallback_used"]),
            "budget_violation_count": sum(1 for row in subset if not row["budget_preserved"]),
            "mean_original_component_coverage": statistics.mean(float(row["original_component_coverage"]) for row in subset),
            "mean_qdc_component_coverage": statistics.mean(float(row["qdc_component_coverage"]) for row in subset),
            "mean_coverage_delta": statistics.mean(float(row["coverage_delta"]) for row in subset),
            "mean_original_redundancy": statistics.mean(float(row["original_redundancy"]) for row in subset),
            "mean_qdc_redundancy": statistics.mean(float(row["qdc_redundancy"]) for row in subset),
            "mean_redundancy_delta": statistics.mean(float(row["redundancy_delta"]) for row in subset),
            "mean_original_query_overlap": statistics.mean(float(row["original_query_overlap"]) for row in subset),
            "mean_qdc_query_overlap": statistics.mean(float(row["qdc_query_overlap"]) for row in subset),
            "mean_query_overlap_delta": statistics.mean(float(row["query_overlap_delta"]) for row in subset),
            "median_selected_jaccard": statistics.median(float(row["selected_jaccard_vs_original"]) for row in subset),
            "p25_selected_jaccard": percentile([float(row["selected_jaccard_vs_original"]) for row in subset], 25),
            "p75_selected_jaccard": percentile([float(row["selected_jaccard_vs_original"]) for row in subset], 75),
        })
    write_csv(output_dir / "qdc_group_summary.csv", group_rows, list(group_rows[0].keys()))

    positive_cases = sorted(
        [row for row in audit_rows if row["qdc_triggered"]],
        key=lambda row: (-float(row["coverage_delta"]), -float(row["selected_jaccard_vs_original"])),
    )[:3]
    risky_cases = sorted(
        audit_rows,
        key=lambda row: (
            -float(row["redundancy_delta"]),
            float(row["selected_jaccard_vs_original"]),
            -len(row.get("removed_block_ids", [])),
        ),
    )[:3]

    case_lines = ["# QDC Case Study", ""]
    case_lines.append("## Positive-looking cases")
    for row in positive_cases:
        case_lines.extend([
            "",
            f"### {row['task']}:{row['sample_index']} ({row['sample_id']})",
            f"- coverage: {float(row['original_component_coverage']):.3f} -> {float(row['qdc_component_coverage']):.3f} (delta {float(row['coverage_delta']):+.3f})",
            f"- missing before: {row['missing_components_before']}",
            f"- missing after: {row['missing_components_after']}",
            f"- added: {row['added_block_ids']} | removed: {row['removed_block_ids']}",
            f"- jaccard: {float(row['selected_jaccard_vs_original']):.3f}",
            f"- added preview: {row['added_block_text_preview']}",
        ])
    case_lines.append("")
    case_lines.append("## Risky cases")
    for row in risky_cases:
        case_lines.extend([
            "",
            f"### {row['task']}:{row['sample_index']} ({row['sample_id']})",
            f"- triggered: {row['qdc_triggered']} | reason: {row['qdc_trigger_reason']}",
            f"- coverage delta: {float(row['coverage_delta']):+.3f} | redundancy delta: {float(row['redundancy_delta']):+.3f}",
            f"- jaccard: {float(row['selected_jaccard_vs_original']):.3f}",
            f"- removed preview: {row['removed_block_text_preview']}",
            f"- added preview: {row['added_block_text_preview']}",
        ])
    case_lines.append("")
    (output_dir / "qdc_case_study.md").write_text("\n".join(case_lines) + "\n", encoding="utf-8")

    all_summary = next(row for row in group_rows if row["group"] == "all")
    report = [
        "# Query-Decomposed Coverage SlimInfer Selection Audit",
        "",
        f"- Output: `{output_dir}`",
        f"- Held-out dir: `{heldout_dir}`",
        "",
        "## Summary",
        "",
        f"1. Sample count: **{all_summary['sample_count']}**",
        f"2. Feature coverage: **{all_summary['feature_coverage']}/{all_summary['sample_count']}**",
        f"3. QDC trigger count: **{all_summary['qdc_trigger_count']}/{all_summary['sample_count']}**",
        f"4. Budget violation count: **{all_summary['budget_violation_count']}**",
        f"5. Fallback count: **{all_summary['fallback_count']}**",
        "",
        "## Coverage / redundancy / overlap",
        "",
        f"- Original component coverage: **{all_summary['mean_original_component_coverage']:.4f}**",
        f"- QDC component coverage: **{all_summary['mean_qdc_component_coverage']:.4f}**",
        f"- Mean coverage delta: **{all_summary['mean_coverage_delta']:+.4f}**",
        f"- Original redundancy: **{all_summary['mean_original_redundancy']:.4f}**",
        f"- QDC redundancy: **{all_summary['mean_qdc_redundancy']:.4f}**",
        f"- Mean redundancy delta: **{all_summary['mean_redundancy_delta']:+.4f}**",
        f"- Original query overlap: **{all_summary['mean_original_query_overlap']:.4f}**",
        f"- QDC query overlap: **{all_summary['mean_qdc_query_overlap']:.4f}**",
        f"- Mean query overlap delta: **{all_summary['mean_query_overlap_delta']:+.4f}**",
        "",
        "## Selected Jaccard distribution (all)",
        "",
        f"- Median: **{all_summary['median_selected_jaccard']:.3f}**",
        f"- P25 / P75: **{all_summary['p25_selected_jaccard']:.3f}** / **{all_summary['p75_selected_jaccard']:.3f}**",
        "",
        "## Per-task",
        "",
        "| task | trigger | mean cov delta | median jaccard | fallback | budget violations |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in group_rows:
        if row["group"] == "all":
            continue
        report.append(
            f"| {row['group']} | {row['qdc_trigger_count']}/{row['sample_count']} | "
            f"{row['mean_coverage_delta']:+.4f} | {row['median_selected_jaccard']:.3f} | "
            f"{row['fallback_count']} | {row['budget_violation_count']} |"
        )

    report.extend([
        "",
        "## Decision",
        "",
        f"- **{decision['decision']}**",
        f"- Worth tiny quality smoke: **{decision.get('worth_tiny_quality_smoke', False)}**",
    ])
    for reason in decision.get("reasons", []):
        report.append(f"- `{reason}`")
    report.append("")
    (output_dir / "qdc_audit_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    decision_md = [
        "# QDC SlimInfer Audit Decision",
        "",
        f"- Decision: **{decision['decision']}**",
        f"- Worth tiny quality smoke: **{decision.get('worth_tiny_quality_smoke', False)}**",
        "",
    ]
    for reason in decision.get("reasons", []):
        decision_md.append(f"- Reason: `{reason}`")
    decision_md.append("")
    (output_dir / "qdc_decision.md").write_text("\n".join(decision_md) + "\n", encoding="utf-8")

    print(json.dumps({
        "status": "ok",
        "output_dir": str(output_dir),
        "decision": decision,
        "summary": all_summary,
    }, indent=2))


if __name__ == "__main__":
    main()
