#!/usr/bin/env python3
"""TCS-SlimInfer teacher-consistency audit on fixed 12 samples.

No QA generation. No scorer. No gold answer usage.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.audit_mcu_sliminfer_counterfactual_utility import (
    build_runner_prompt_bundle,
    layer_scores_from_record,
    load_dataset2prompt,
    load_final_layer_records,
    load_probe_model,
    load_samples,
    run_gate_audit,
)
from tools.audit_query_decomposed_coverage_sliminfer import BLOCK_SIZE, TASKS, parse_bool, preview_text
from tools.mcu_sliminfer_selection import (
    MAX_POOL_SIZE,
    block_query_overlaps,
    block_token_map,
    build_candidate_pool,
    compact_input_ids,
    find_query_span,
    protected_block_ids,
    run_forward_probe,
    selected_jaccard,
    token_count,
)
from utils.cache_utils import _dc_selected_redundancy

DEFAULT_LONGBENCH_DIR = Path("/home/yu-bai/Desktop/jiaqizhang/HeadKV/data/LongBench")
DEFAULT_MODEL_PATH = Path(
    "/home/yu-bai/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/"
    "snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
)
PRUNE_LAYERS = [9, 18, 26]
FINAL_LAYER = PRUNE_LAYERS[-1]
MAX_MODEL_LEN = 32768
MAX_SWAP_PROBES = 24
MAX_REPLACE_FRAC = 0.20
PROBE_SLOW_SAMPLE_SEC = 60.0
PROBE_SLOW_SINGLE_SEC = 30.0
TCS_SAMPLE_INDICES: Dict[str, List[int]] = {
    "hotpotqa": [0, 1, 2, 3, 4, 5],
    "musique": [0, 1, 2, 3, 4, 20],
}

Key = Tuple[str, int]
JsonRow = Dict[str, Any]

AUDIT_FIELDS = [
    "task", "sample_index", "sample_id",
    "target_budget", "teacher_budget", "candidate_pool_size", "num_swap_probes",
    "original_selected_block_ids", "tcs_selected_block_ids",
    "selection_changed", "selected_jaccard_vs_original",
    "removed_block_ids", "added_block_ids",
    "original_teacher_kl", "tcs_teacher_kl", "teacher_kl_delta",
    "original_hidden_distance", "tcs_hidden_distance", "hidden_distance_delta",
    "original_entropy_gap", "tcs_entropy_gap", "entropy_gap_delta",
    "original_top_margin_gap", "tcs_top_margin_gap", "top_margin_gap_delta",
    "utility_margin", "redundancy_delta", "query_overlap_delta",
    "selected_token_count_original", "selected_token_count_tcs",
    "budget_preserved", "over_budget", "fallback_used", "fallback_reason",
    "probe_runtime_ms",
]


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _serialize(row.get(field, "")) for field in fields})


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _serialize(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


def tcs_keys() -> List[Key]:
    return [(task, idx) for task in TASKS for idx in TCS_SAMPLE_INDICES[task]]


def build_sample_spec() -> Tuple[str, Dict[str, List[int]]]:
    spec = ";".join(f"{task}:" + ",".join(str(i) for i in TCS_SAMPLE_INDICES[task]) for task in TASKS)
    return spec, dict(TCS_SAMPLE_INDICES)


def _entropy(logits: torch.Tensor) -> float:
    probs = F.softmax(logits.float(), dim=-1)
    log_probs = F.log_softmax(logits.float(), dim=-1)
    return float(-(probs * log_probs).sum().item())


def _top_margin(logits: torch.Tensor) -> float:
    top2 = torch.topk(logits.float(), k=min(2, logits.shape[-1])).values
    if top2.numel() < 2:
        return 0.0
    return float(top2[0] - top2[1])


def _kl_student_teacher(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> float:
    student_log_probs = F.log_softmax(student_logits.float(), dim=-1)
    teacher_probs = F.softmax(teacher_logits.float(), dim=-1)
    return float(F.kl_div(student_log_probs, teacher_probs, reduction="sum").item())


def teacher_budget_from_target(target_budget: int, candidate_count: int) -> int:
    if target_budget <= 0:
        return 0
    return min(candidate_count, max(target_budget + 1, target_budget * 2))


def build_teacher_selected(
    original_selected: Set[int],
    all_scores: Dict[int, float],
    protected: Set[int],
    teacher_budget: int,
) -> Set[int]:
    all_blocks = set(all_scores.keys()) | set(original_selected) | set(protected)
    ranked = sorted(all_blocks, key=lambda bid: (all_scores.get(bid, float("-inf")), -bid), reverse=True)
    selected: Set[int] = set(protected)
    for bid in ranked:
        selected.add(int(bid))
        if len(selected) >= teacher_budget:
            break
    if len(selected) > teacher_budget:
        removable = sorted(
            [bid for bid in selected if bid not in protected],
            key=lambda bid: all_scores.get(bid, float("-inf")),
        )
        while len(selected) > teacher_budget and removable:
            selected.remove(removable.pop(0))
    return selected


def probe_summary(
    state: Any,
    teacher_state: Any,
) -> Dict[str, float]:
    hidden_distance = float(torch.norm(state.last_hidden - teacher_state.last_hidden, p=2).item())
    hidden_distance += 0.5 * float(torch.norm(state.query_hidden_mean - teacher_state.query_hidden_mean, p=2).item())
    return {
        "teacher_kl": _kl_student_teacher(state.next_logits, teacher_state.next_logits),
        "hidden_distance": hidden_distance,
        "entropy_gap": abs(_entropy(state.next_logits) - _entropy(teacher_state.next_logits)),
        "top_margin_gap": abs(_top_margin(state.next_logits) - _top_margin(teacher_state.next_logits)),
    }


def choose_swap_probe_pairs(
    original_selected: Set[int],
    candidate_pool: Set[int],
    all_scores: Dict[int, float],
    query_overlaps: Dict[int, float],
    protected: Set[int],
    token_lengths: Sequence[int],
) -> List[Tuple[int, int]]:
    victims = [
        bid for bid in sorted(original_selected, key=lambda x: all_scores.get(x, float("-inf")))
        if bid not in protected
    ]
    adds = [bid for bid in candidate_pool if bid not in original_selected]
    adds = sorted(
        adds,
        key=lambda bid: (query_overlaps.get(bid, 0.0), all_scores.get(bid, float("-inf"))),
        reverse=True,
    )
    pairs: List[Tuple[int, int]] = []
    for victim in victims:
        for add in adds:
            if token_lengths[victim] != token_lengths[add]:
                continue
            pairs.append((victim, add))
            if len(pairs) >= MAX_SWAP_PROBES:
                return pairs
    return pairs


def apply_tcs_greedy_swaps(
    original_selected: Set[int],
    swap_candidates: Sequence[Dict[str, Any]],
    token_lengths: Sequence[int],
    max_replace_frac: float = MAX_REPLACE_FRAC,
) -> Tuple[Set[int], List[Dict[str, Any]], float]:
    selected = set(original_selected)
    chosen: List[Dict[str, Any]] = []
    used_victims: Set[int] = set()
    used_adds: Set[int] = set()
    max_swaps = max(0, int(math.floor(max_replace_frac * len(original_selected))))
    ranked = sorted(swap_candidates, key=lambda row: float(row["utility"]), reverse=True)
    for row in ranked:
        victim = int(row["victim"])
        add = int(row["add"])
        utility = float(row["utility"])
        if utility <= 0.0:
            continue
        if victim in used_victims or add in used_adds:
            continue
        if max_swaps and len(chosen) >= max_swaps:
            break
        if victim not in selected:
            continue
        selected.remove(victim)
        selected.add(add)
        if token_count(selected, token_lengths) != token_count(original_selected, token_lengths):
            selected.remove(add)
            selected.add(victim)
            continue
        used_victims.add(victim)
        used_adds.add(add)
        chosen.append(dict(row))
    utility_margin = statistics.mean(float(row["utility"]) for row in chosen) if chosen else 0.0
    return selected, chosen, utility_margin


def decide_tcs_audit(rows: Sequence[Mapping[str, Any]], teacher_probe_ok: bool, stop_reason: Optional[str]) -> Dict[str, Any]:
    if not teacher_probe_ok:
        return {
            "decision": "BUG_STOP_FIX_REQUIRED",
            "reasons": [stop_reason or "teacher_probe_failed"],
            "worth_tiny_quality_smoke": False,
        }
    if not rows:
        return {"decision": "BUG_STOP_FIX_REQUIRED", "reasons": ["no_audit_rows"], "worth_tiny_quality_smoke": False}

    total = len(rows)
    budget_violations = sum(1 for row in rows if not parse_bool(row.get("budget_preserved", True)))
    fallbacks = sum(1 for row in rows if parse_bool(row.get("fallback_used")))
    changed = sum(1 for row in rows if parse_bool(row.get("selection_changed")))
    jaccards = [float(row.get("selected_jaccard_vs_original", 1.0) or 1.0) for row in rows]
    teacher_kl_delta = [float(row.get("teacher_kl_delta", 0.0) or 0.0) for row in rows if not parse_bool(row.get("fallback_used"))]
    hidden_delta = [float(row.get("hidden_distance_delta", 0.0) or 0.0) for row in rows if not parse_bool(row.get("fallback_used"))]
    utility_margin = [float(row.get("utility_margin", 0.0) or 0.0) for row in rows if not parse_bool(row.get("fallback_used"))]
    probe_sec = [float(row.get("probe_runtime_ms", 0.0) or 0.0) / 1000.0 for row in rows]
    mean_probe_sec = statistics.mean(probe_sec) if probe_sec else 0.0
    median_jaccard = statistics.median(jaccards) if jaccards else 1.0
    mean_kl_delta = statistics.mean(teacher_kl_delta) if teacher_kl_delta else 0.0
    mean_hidden_delta = statistics.mean(hidden_delta) if hidden_delta else 0.0
    mean_utility = statistics.mean(utility_margin) if utility_margin else 0.0

    if budget_violations > 0:
        return {"decision": "BUG_STOP_FIX_REQUIRED", "reasons": [f"budget_violation_count={budget_violations}"], "worth_tiny_quality_smoke": False}
    if fallbacks > total // 2:
        return {"decision": "BUG_STOP_FIX_REQUIRED", "reasons": [f"fallback_count={fallbacks}/{total}"], "worth_tiny_quality_smoke": False}

    no_go: List[str] = []
    if changed == 0:
        no_go.append("selection_no_op")
    if median_jaccard < 0.30:
        no_go.append(f"selection_too_chaotic_median_jaccard={median_jaccard:.3f}")
    if mean_probe_sec > 60.0:
        no_go.append(f"probe_cost_too_high_mean_sec={mean_probe_sec:.1f}")
    if mean_kl_delta <= 0.0:
        no_go.append(f"teacher_kl_not_improved_mean_delta={mean_kl_delta:.4f}")
    if mean_hidden_delta <= 0.0:
        no_go.append(f"hidden_distance_not_improved_mean_delta={mean_hidden_delta:.4f}")

    go_ready = (
        changed > 0
        and budget_violations == 0
        and fallbacks <= max(2, total // 4)
        and 0.30 <= median_jaccard <= 0.95
        and mean_probe_sec <= 60.0
        and mean_kl_delta > 0.0
        and mean_hidden_delta > 0.0
        and mean_utility > 0.0
    )
    if go_ready and not no_go:
        return {
            "decision": "TCS_AUDIT_GO",
            "reasons": [
                f"selection_changed={changed}/{total}",
                f"mean_teacher_kl_delta={mean_kl_delta:.4f}",
                f"mean_hidden_distance_delta={mean_hidden_delta:.4f}",
                f"mean_utility_margin={mean_utility:.4f}",
                f"mean_probe_sec={mean_probe_sec:.1f}",
            ],
            "worth_tiny_quality_smoke": True,
        }

    weak_ready = changed > 0 and budget_violations == 0 and mean_kl_delta > 0.0 and mean_hidden_delta >= 0.0 and mean_probe_sec <= 60.0
    if weak_ready:
        return {
            "decision": "TCS_AUDIT_WEAK",
            "reasons": no_go or [
                f"selection_changed={changed}/{total}",
                f"mean_teacher_kl_delta={mean_kl_delta:.4f}",
                f"mean_hidden_distance_delta={mean_hidden_delta:.4f}",
            ],
            "worth_tiny_quality_smoke": True,
        }

    return {"decision": "TCS_AUDIT_NO_GO", "reasons": no_go or ["teacher_alignment_not_improved"], "worth_tiny_quality_smoke": False}


def build_case_study(rows: Sequence[Mapping[str, Any]], block_texts_by_key: Mapping[Key, List[str]]) -> str:
    changed = sorted(
        [row for row in rows if parse_bool(row.get("selection_changed"))],
        key=lambda row: -float(row.get("utility_margin", 0.0) or 0.0),
    )[:5]
    lines = ["# TCS-SlimInfer Case Study", ""]
    for row in changed:
        key = (str(row["task"]), int(row["sample_index"]))
        texts = block_texts_by_key.get(key, [])
        removed = row.get("removed_block_ids") or []
        added = row.get("added_block_ids") or []
        lines.extend([
            f"## {row['task']}:{row['sample_index']} ({row.get('sample_id', '')})",
            f"- teacher KL original -> TCS: {float(row.get('original_teacher_kl', 0.0)):.4f} -> {float(row.get('tcs_teacher_kl', 0.0)):.4f}",
            f"- hidden distance original -> TCS: {float(row.get('original_hidden_distance', 0.0)):.4f} -> {float(row.get('tcs_hidden_distance', 0.0)):.4f}",
            f"- utility margin: {float(row.get('utility_margin', 0.0)):.4f}",
            f"- removed ({removed}): {preview_text(texts, removed if isinstance(removed, list) else json.loads(removed))}",
            f"- added ({added}): {preview_text(texts, added if isinstance(added, list) else json.loads(added))}",
            "",
        ])
    if not changed:
        lines.append("_No TCS selection changes observed._")
        lines.append("")
    return "\n".join(lines)


def audit_sample(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    task: str,
    sample_index: int,
    sample: Dict[str, Any],
    final_record: JsonRow,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], bool]:
    input_ids, query_text, block_texts, token_lengths = build_runner_prompt_bundle(
        tokenizer, load_dataset2prompt()[task], sample, task,
    )
    seq_len = int(input_ids.shape[0])
    query_start, _ = find_query_span(input_ids, tokenizer, query_text)
    protected = protected_block_ids(seq_len, BLOCK_SIZE, query_start)

    original_selected = {int(x) for x in (final_record.get("selected_block_ids") or [])}
    sample_id = final_record.get("sample_id", sample.get("_id", sample_index))
    all_scores = layer_scores_from_record(final_record)
    query_overlaps = block_query_overlaps(block_texts, query_text)
    token_map = block_token_map(block_texts)

    if not original_selected:
        return {
            "task": task,
            "sample_index": sample_index,
            "sample_id": sample_id,
            "fallback_used": True,
            "fallback_reason": "missing_original_selected_block_ids",
        }, [], False

    target_budget = len(original_selected)
    teacher_budget = teacher_budget_from_target(target_budget, len(set(all_scores.keys()) | original_selected | protected))
    teacher_selected = build_teacher_selected(original_selected, all_scores, protected, teacher_budget)
    candidate_pool = build_candidate_pool(original_selected, all_scores, query_overlaps, protected, max_pool=MAX_POOL_SIZE)
    candidate_pool |= protected

    original_active = set(original_selected) | protected
    teacher_active = set(teacher_selected)
    original_compact = compact_input_ids(input_ids, original_active, BLOCK_SIZE)
    teacher_compact = compact_input_ids(input_ids, teacher_active, BLOCK_SIZE)
    if int(original_compact.shape[0]) == 0 or int(teacher_compact.shape[0]) == 0:
        return {
            "task": task,
            "sample_index": sample_index,
            "sample_id": sample_id,
            "target_budget": target_budget,
            "teacher_budget": teacher_budget,
            "fallback_used": True,
            "fallback_reason": "empty_compact_input",
        }, [], False

    q_start_original = max(0, int(original_compact.shape[0]) - (seq_len - query_start))
    q_start_teacher = max(0, int(teacher_compact.shape[0]) - (seq_len - query_start))
    t_start = time.perf_counter()
    try:
        original_state = run_forward_probe(model, original_compact, q_start_original)
        teacher_state = run_forward_probe(model, teacher_compact, q_start_teacher)
    except Exception as exc:
        return {
            "task": task,
            "sample_index": sample_index,
            "sample_id": sample_id,
            "target_budget": target_budget,
            "teacher_budget": teacher_budget,
            "fallback_used": True,
            "fallback_reason": f"teacher_probe_failed:{exc}",
        }, [], False
    base_probe_ms = (time.perf_counter() - t_start) * 1000.0
    original_summary = probe_summary(original_state, teacher_state)
    original_redundancy = _dc_selected_redundancy(original_selected, token_map)

    pairs = choose_swap_probe_pairs(original_selected, candidate_pool, all_scores, query_overlaps, protected, token_lengths)
    swap_details: List[Dict[str, Any]] = []
    stop_reason = ""
    for victim, add in pairs:
        if time.perf_counter() - t_start > PROBE_SLOW_SAMPLE_SEC:
            stop_reason = "probe_runtime_exceeded_sample_budget"
            break
        active = set(original_active)
        if victim in active:
            active.remove(victim)
        active.add(add)
        compact = compact_input_ids(input_ids, active, BLOCK_SIZE)
        q_start = max(0, int(compact.shape[0]) - (seq_len - query_start))
        p0 = time.perf_counter()
        try:
            swapped_state = run_forward_probe(model, compact, q_start)
        except torch.cuda.OutOfMemoryError as exc:
            return {
                "task": task,
                "sample_index": sample_index,
                "sample_id": sample_id,
                "target_budget": target_budget,
                "teacher_budget": teacher_budget,
                "fallback_used": True,
                "fallback_reason": f"swap_probe_oom:{exc}",
            }, swap_details, False
        runtime_ms = (time.perf_counter() - p0) * 1000.0
        if runtime_ms > PROBE_SLOW_SINGLE_SEC * 1000.0:
            stop_reason = f"single_probe_too_slow_ms={runtime_ms:.0f}"
            break

        swapped_selected = (set(original_selected) - {victim}) | {add}
        swapped_summary = probe_summary(swapped_state, teacher_state)
        swapped_redundancy = _dc_selected_redundancy(swapped_selected, token_map)
        redundancy_penalty = max(0.0, swapped_redundancy - original_redundancy)
        utility = (
            (original_summary["teacher_kl"] - swapped_summary["teacher_kl"])
            + 0.5 * (original_summary["hidden_distance"] - swapped_summary["hidden_distance"])
            + 0.5 * (original_summary["entropy_gap"] - swapped_summary["entropy_gap"])
            - 0.25 * redundancy_penalty
        )
        swap_details.append({
            "task": task,
            "sample_index": sample_index,
            "sample_id": sample_id,
            "victim": victim,
            "add": add,
            "utility": utility,
            "teacher_kl": swapped_summary["teacher_kl"],
            "hidden_distance": swapped_summary["hidden_distance"],
            "entropy_gap": swapped_summary["entropy_gap"],
            "top_margin_gap": swapped_summary["top_margin_gap"],
            "redundancy_penalty": redundancy_penalty,
            "runtime_ms": runtime_ms,
        })

    tcs_selected, chosen_swaps, utility_margin = apply_tcs_greedy_swaps(original_selected, swap_details, token_lengths)
    if stop_reason and not chosen_swaps:
        tcs_selected = set(original_selected)

    tcs_active = set(tcs_selected) | protected
    tcs_compact = compact_input_ids(input_ids, tcs_active, BLOCK_SIZE)
    q_start_tcs = max(0, int(tcs_compact.shape[0]) - (seq_len - query_start))
    try:
        tcs_state = run_forward_probe(model, tcs_compact, q_start_tcs)
    except Exception as exc:
        return {
            "task": task,
            "sample_index": sample_index,
            "sample_id": sample_id,
            "target_budget": target_budget,
            "teacher_budget": teacher_budget,
            "fallback_used": True,
            "fallback_reason": f"tcs_probe_failed:{exc}",
        }, swap_details, False

    tcs_summary = probe_summary(tcs_state, teacher_state)
    tcs_redundancy = _dc_selected_redundancy(tcs_selected, token_map)
    original_query_overlap = float(sum(query_overlaps.get(bid, 0.0) for bid in original_selected) / max(1, len(original_selected)))
    tcs_query_overlap = float(sum(query_overlaps.get(bid, 0.0) for bid in tcs_selected) / max(1, len(tcs_selected)))

    removed = sorted(original_selected - tcs_selected)
    added = sorted(tcs_selected - original_selected)
    original_token_count = token_count(original_selected, token_lengths)
    tcs_token_count = token_count(tcs_selected, token_lengths)
    budget_preserved = len(tcs_selected) == len(original_selected) and tcs_token_count == original_token_count
    probe_runtime_ms = base_probe_ms + sum(float(row["runtime_ms"]) for row in swap_details)

    row = {
        "task": task,
        "sample_index": sample_index,
        "sample_id": sample_id,
        "target_budget": target_budget,
        "teacher_budget": teacher_budget,
        "candidate_pool_size": len(candidate_pool),
        "num_swap_probes": len(swap_details),
        "original_selected_block_ids": sorted(original_selected),
        "tcs_selected_block_ids": sorted(tcs_selected),
        "selection_changed": bool(removed or added),
        "selected_jaccard_vs_original": selected_jaccard(original_selected, tcs_selected),
        "removed_block_ids": removed,
        "added_block_ids": added,
        "original_teacher_kl": original_summary["teacher_kl"],
        "tcs_teacher_kl": tcs_summary["teacher_kl"],
        "teacher_kl_delta": original_summary["teacher_kl"] - tcs_summary["teacher_kl"],
        "original_hidden_distance": original_summary["hidden_distance"],
        "tcs_hidden_distance": tcs_summary["hidden_distance"],
        "hidden_distance_delta": original_summary["hidden_distance"] - tcs_summary["hidden_distance"],
        "original_entropy_gap": original_summary["entropy_gap"],
        "tcs_entropy_gap": tcs_summary["entropy_gap"],
        "entropy_gap_delta": original_summary["entropy_gap"] - tcs_summary["entropy_gap"],
        "original_top_margin_gap": original_summary["top_margin_gap"],
        "tcs_top_margin_gap": tcs_summary["top_margin_gap"],
        "top_margin_gap_delta": original_summary["top_margin_gap"] - tcs_summary["top_margin_gap"],
        "utility_margin": utility_margin,
        "redundancy_delta": tcs_redundancy - original_redundancy,
        "query_overlap_delta": tcs_query_overlap - original_query_overlap,
        "selected_token_count_original": original_token_count,
        "selected_token_count_tcs": tcs_token_count,
        "budget_preserved": budget_preserved,
        "over_budget": tcs_token_count > original_token_count,
        "fallback_used": bool(stop_reason and not chosen_swaps),
        "fallback_reason": stop_reason,
        "probe_runtime_ms": probe_runtime_ms,
    }
    return row, swap_details, True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--longbench_dir", type=Path, default=DEFAULT_LONGBENCH_DIR)
    parser.add_argument("--output_root", type=Path, default=Path("results"))
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--skip_gate_run", action="store_true")
    parser.add_argument("--score_dump_path", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    keys = tcs_keys()
    sample_spec, expected_indices = build_sample_spec()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (REPO_ROOT / args.output_root / f"tcs_sliminfer_teacher_consistency_audit_{timestamp}")
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    gate_dir = output_dir / "run_sliminfer_original_gate_audit"
    score_dump_path = args.score_dump_path or gate_dir / "uasr_score_dump.jsonl"
    stop_reason: Optional[str] = None
    teacher_probe_ok = True
    block_texts_by_key: Dict[Key, List[str]] = {}
    audit_rows: List[Dict[str, Any]] = []
    all_probe_details: List[Dict[str, Any]] = []

    try:
        if args.skip_gate_run:
            if not score_dump_path.exists():
                raise RuntimeError(f"--skip_gate_run but score dump missing: {score_dump_path}")
        else:
            paths = run_gate_audit(args, output_dir, sample_spec)
            score_dump_path = paths["score_dump_path"]

        final_records = load_final_layer_records(score_dump_path, keys)
        missing_keys = [key for key in keys if key not in final_records]
        if missing_keys:
            teacher_probe_ok = False
            stop_reason = f"missing_final_layer_records={missing_keys}"

        tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), trust_remote_code=True)
        samples_cache = {task: load_samples(args.longbench_dir, task) for task in TASKS}
        if teacher_probe_ok:
            model = load_probe_model(args.model_path, args.device)
            for task, sample_index in keys:
                key = (task, sample_index)
                final_record = final_records[key]
                sample = samples_cache[task][sample_index]
                _, _, block_texts, _ = build_runner_prompt_bundle(tokenizer, load_dataset2prompt()[task], sample, task)
                block_texts_by_key[key] = block_texts
                row, probe_details, ok = audit_sample(model, tokenizer, task, sample_index, sample, final_record)
                audit_rows.append(row)
                all_probe_details.extend(probe_details)
                if not ok:
                    teacher_probe_ok = False
                    stop_reason = str(row.get("fallback_reason", "teacher_probe_failed"))
                    break
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    except RuntimeError as exc:
        stop_reason = str(exc)
        if "OOM" in stop_reason:
            teacher_probe_ok = False
        else:
            raise

    valid_rows = [row for row in audit_rows if row.get("original_selected_block_ids")]
    decision = decide_tcs_audit(valid_rows, teacher_probe_ok, stop_reason)

    write_csv(output_dir / "tcs_teacher_consistency_audit.csv", audit_rows, AUDIT_FIELDS)
    write_jsonl(output_dir / "tcs_probe_details.jsonl", all_probe_details)

    group_rows: List[Dict[str, Any]] = []
    for group in TASKS + ["all"]:
        subset = audit_rows if group == "all" else [row for row in audit_rows if row.get("task") == group]
        if not subset:
            continue
        valid = [row for row in subset if not parse_bool(row.get("fallback_used"))]
        jaccards = [float(row.get("selected_jaccard_vs_original", 1.0) or 1.0) for row in valid]
        group_rows.append({
            "group": group,
            "sample_count": len(subset),
            "valid_sample_count": len(valid),
            "fallback_count": sum(1 for row in subset if parse_bool(row.get("fallback_used"))),
            "budget_violation_count": sum(1 for row in valid if not parse_bool(row.get("budget_preserved", True))),
            "selection_changed_count": sum(1 for row in valid if parse_bool(row.get("selection_changed"))),
            "mean_probe_runtime_ms": statistics.mean(float(row.get("probe_runtime_ms", 0.0) or 0.0) for row in subset) if subset else 0.0,
            "mean_original_teacher_kl": statistics.mean(float(row.get("original_teacher_kl", 0.0) or 0.0) for row in valid) if valid else 0.0,
            "mean_tcs_teacher_kl": statistics.mean(float(row.get("tcs_teacher_kl", 0.0) or 0.0) for row in valid) if valid else 0.0,
            "mean_teacher_kl_delta": statistics.mean(float(row.get("teacher_kl_delta", 0.0) or 0.0) for row in valid) if valid else 0.0,
            "mean_hidden_distance_delta": statistics.mean(float(row.get("hidden_distance_delta", 0.0) or 0.0) for row in valid) if valid else 0.0,
            "mean_utility_margin": statistics.mean(float(row.get("utility_margin", 0.0) or 0.0) for row in valid) if valid else 0.0,
            "median_selected_jaccard": statistics.median(jaccards) if jaccards else 1.0,
            "mean_target_budget": statistics.mean(float(row.get("target_budget", 0.0) or 0.0) for row in valid) if valid else 0.0,
            "mean_teacher_budget": statistics.mean(float(row.get("teacher_budget", 0.0) or 0.0) for row in valid) if valid else 0.0,
        })
    if group_rows:
        write_csv(output_dir / "tcs_group_summary.csv", group_rows, list(group_rows[0].keys()))

    case_study = build_case_study(valid_rows, block_texts_by_key)
    (output_dir / "tcs_case_study.md").write_text(case_study, encoding="utf-8")
    all_summary = next((row for row in group_rows if row["group"] == "all"), {})

    report_lines = [
        "# TCS-SlimInfer Teacher Consistency Audit",
        "",
        f"- Output: `{output_dir}`",
        f"- Decision: **{decision.get('decision')}**",
        f"- Samples: **{len(keys)}** (completed={len(audit_rows)})",
        "",
        "## Probe signal",
        f"1. Teacher probe success: **{teacher_probe_ok}**",
        f"2. Mean target budget: **{all_summary.get('mean_target_budget', 0.0):.1f}**",
        f"3. Mean teacher budget: **{all_summary.get('mean_teacher_budget', 0.0):.1f}**",
        f"4. Mean probe runtime per sample (ms): **{all_summary.get('mean_probe_runtime_ms', 0.0):.0f}**",
        "",
        "## Teacher alignment",
        f"5. Mean original teacher KL: **{all_summary.get('mean_original_teacher_kl', 0.0):.4f}**",
        f"6. Mean TCS teacher KL: **{all_summary.get('mean_tcs_teacher_kl', 0.0):.4f}**",
        f"7. Mean teacher KL delta (orig-TCS): **{all_summary.get('mean_teacher_kl_delta', 0.0):.4f}**",
        f"8. Mean hidden distance delta (orig-TCS): **{all_summary.get('mean_hidden_distance_delta', 0.0):.4f}**",
        "",
        "## Selection / utility",
        f"9. Selection changed count: **{all_summary.get('selection_changed_count', 0)}**",
        f"10. Median selected Jaccard: **{all_summary.get('median_selected_jaccard', 1.0):.4f}**",
        f"11. Mean utility margin: **{all_summary.get('mean_utility_margin', 0.0):.4f}**",
        f"12. Budget violation count: **{all_summary.get('budget_violation_count', 0)}**",
        f"13. Fallback count: **{all_summary.get('fallback_count', 0)}**",
        "",
        "## Tiny quality smoke",
        f"- Worth tiny quality smoke: **{decision.get('worth_tiny_quality_smoke', False)}**",
        "",
        "## Decision reasons",
    ]
    if stop_reason:
        report_lines.append(f"- Stop reason: {stop_reason}")
    report_lines.extend(f"- {reason}" for reason in decision.get("reasons", []))
    report_lines.append("")

    (output_dir / "tcs_audit_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    (output_dir / "tcs_decision.md").write_text(
        f"# Decision: {decision.get('decision')}\n\n"
        + "\n".join(f"- {reason}" for reason in decision.get("reasons", []))
        + f"\n\n- Worth tiny quality smoke: {decision.get('worth_tiny_quality_smoke', False)}\n",
        encoding="utf-8",
    )
    write_json(output_dir / "run_config.json", {
        "sample_spec": sample_spec,
        "expected_indices": expected_indices,
        "tcs_sample_indices": TCS_SAMPLE_INDICES,
        "target_budget": "sliminfer_original K",
        "teacher_budget": "min(2K, available candidates)",
        "candidate_pool_max": MAX_POOL_SIZE,
        "swap_probes_max": MAX_SWAP_PROBES,
        "max_replace_frac": MAX_REPLACE_FRAC,
        "utility_formula": "KL(orig,teacher)-KL(swapped,teacher)+0.5*hidden_reduction+0.5*entropy_gap_reduction-0.25*redundancy_penalty",
        "score_dump_path": str(score_dump_path),
        "quality_generation_performed": False,
        "scorer_invoked": False,
        "stop_reason": stop_reason,
    })
    write_json(output_dir / "tcs_decision.json", decision)

    print("\n".join(report_lines))
    print(f"Wrote results to {output_dir}")


if __name__ == "__main__":
    main()
