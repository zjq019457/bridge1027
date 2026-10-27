"""Shared MCU-SlimInfer counterfactual utility selection logic."""
from __future__ import annotations

import json
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.cache_utils import _dc_selected_redundancy, _gate_content_tokens

BLOCK_SIZE = 64
MAX_POOL_SIZE = 32
MAX_REMOVE_PROBES = 8
MAX_SWAP_PROBES = 12
MAX_REPLACE_FRAC = 0.20
UTILITY_MARGIN_MIN = 0.15
PROBE_SLOW_SAMPLE_SEC = 600.0
PROBE_SLOW_SINGLE_SEC = 90.0
FINAL_LAYER = 26


@dataclass
class ProbeState:
    last_hidden: torch.Tensor
    query_hidden_mean: torch.Tensor
    next_logits: torch.Tensor
    value_norm: float
    attention_mass: float


@dataclass
class ProbeMetrics:
    last_token_hidden_l2_change: float = 0.0
    query_token_hidden_l2_change: float = 0.0
    next_token_logit_kl: float = 0.0
    next_token_entropy_change: float = 0.0
    top_logit_margin_change: float = 0.0
    value_norm_change: float = 0.0
    attention_mass_change: float = 0.0
    probe_runtime_ms: float = 0.0


def parse_json_field(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return value
    if isinstance(value, str) and value.strip().startswith(("[", "{")):
        return json.loads(value)
    return value


def selected_jaccard(left: Set[int], right: Set[int]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def token_count(selected: Set[int], token_lengths: Sequence[int]) -> int:
    return sum(int(token_lengths[block_id]) for block_id in selected if 0 <= block_id < len(token_lengths))


def block_token_map(block_texts: Sequence[str]) -> Dict[int, Set[str]]:
    return {idx: _gate_content_tokens(text) for idx, text in enumerate(block_texts)}


def block_query_overlaps(block_texts: Sequence[str], query: str) -> Dict[int, float]:
    query_tokens = _gate_content_tokens(query)
    if not query_tokens:
        return {idx: 0.0 for idx in range(len(block_texts))}
    overlaps: Dict[int, float] = {}
    for idx, text in enumerate(block_texts):
        tokens = _gate_content_tokens(text)
        overlaps[idx] = len(tokens.intersection(query_tokens)) / len(query_tokens) if tokens else 0.0
    return overlaps


def layer_scores_from_record(record: Mapping[str, Any]) -> Dict[int, float]:
    block_ids = [int(x) for x in (parse_json_field(record.get("all_candidate_block_ids")) or [])]
    scores = parse_json_field(record.get("all_candidate_scores")) or []
    out: Dict[int, float] = {}
    for block_id, score in zip(block_ids, scores):
        try:
            value = float(score)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            out[int(block_id)] = value
    return out


def find_query_span(input_ids: torch.Tensor, tokenizer: AutoTokenizer, query_text: str) -> Tuple[int, int]:
    query_ids = tokenizer(query_text, add_special_tokens=False).input_ids
    if not query_ids:
        return max(0, int(input_ids.shape[0]) - 128), int(input_ids.shape[0])
    q_tensor = torch.tensor(query_ids, dtype=input_ids.dtype, device=input_ids.device)
    seq_len = int(input_ids.shape[0])
    q_len = int(q_tensor.shape[0])
    for start in range(seq_len - q_len, -1, -1):
        if torch.equal(input_ids[start:start + q_len], q_tensor):
            return start, seq_len
    return max(0, seq_len - min(128, seq_len)), seq_len


def protected_block_ids(seq_len: int, block_size: int, query_start: int) -> Set[int]:
    num_blocks = (seq_len + block_size - 1) // block_size
    protected: Set[int] = set()
    for bid in range(num_blocks):
        bend = min(seq_len, (bid + 1) * block_size)
        if bend > query_start:
            protected.add(bid)
    return protected


def compact_input_ids(input_ids: torch.Tensor, active_blocks: Set[int], block_size: int) -> torch.Tensor:
    indices: List[int] = []
    seq_len = int(input_ids.shape[0])
    num_blocks = (seq_len + block_size - 1) // block_size
    for bid in range(num_blocks):
        if bid not in active_blocks:
            continue
        start = bid * block_size
        end = min(seq_len, (bid + 1) * block_size)
        indices.extend(range(start, end))
    if not indices:
        return input_ids[:0]
    return input_ids[torch.tensor(indices, dtype=torch.long)]


def build_candidate_pool(
    original_selected: Set[int],
    all_scores: Dict[int, float],
    query_overlaps: Dict[int, float],
    protected: Set[int],
    max_pool: int = MAX_POOL_SIZE,
) -> Set[int]:
    all_blocks = set(all_scores.keys()) | original_selected
    pruned = sorted(all_blocks - original_selected)
    selected_low = sorted(original_selected, key=lambda bid: all_scores.get(bid, float("-inf")))[: max(4, max_pool // 4)]
    pruned_near: List[int] = []
    if pruned:
        pruned_by_score = sorted(pruned, key=lambda bid: all_scores.get(bid, float("-inf")), reverse=True)
        pruned_near = pruned_by_score[: max(8, max_pool // 2)]
    overlap_ranked = sorted(pruned, key=lambda bid: query_overlaps.get(bid, 0.0), reverse=True)
    overlap_top = overlap_ranked[: max(6, max_pool // 3)]
    pool: List[int] = []
    seen: Set[int] = set()
    for bid in list(selected_low) + list(pruned_near) + list(overlap_top):
        if bid in seen:
            continue
        seen.add(bid)
        pool.append(bid)
        if len(pool) >= max_pool:
            break
    for bid in sorted(protected):
        if bid not in seen:
            if len(pool) >= max_pool:
                pool.pop(0)
            pool.append(bid)
            seen.add(bid)
    return set(pool[:max_pool])


def _entropy(logits: torch.Tensor) -> float:
    probs = F.softmax(logits.float(), dim=-1)
    log_probs = F.log_softmax(logits.float(), dim=-1)
    return float(-(probs * log_probs).sum().item())


def _top_margin(logits: torch.Tensor) -> float:
    top2 = torch.topk(logits.float(), k=min(2, logits.shape[-1])).values
    if top2.numel() < 2:
        return 0.0
    return float(top2[0] - top2[1])


def run_forward_probe(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    query_start_in_compact: int,
) -> ProbeState:
    device = next(model.parameters()).device
    compact_len = int(input_ids.shape[0])
    if compact_len == 0:
        raise RuntimeError("empty compact input for probe")
    with torch.inference_mode():
        outputs = model(
            input_ids.unsqueeze(0).to(device),
            use_cache=False,
            output_hidden_states=True,
            output_attentions=False,
            return_dict=True,
        )
    if not outputs.hidden_states:
        raise RuntimeError("model forward did not return hidden_states")
    hidden = outputs.hidden_states[-1][0].float()
    logits = outputs.logits[0].float()
    q_start = max(0, min(query_start_in_compact, compact_len - 1))
    return ProbeState(
        last_hidden=hidden[-1].cpu(),
        query_hidden_mean=hidden[q_start:].mean(dim=0).cpu(),
        next_logits=logits[-1].cpu(),
        value_norm=float(hidden.norm(dim=-1).mean().item()),
        attention_mass=0.0,
    )


def compare_probe_states(baseline: ProbeState, probe: ProbeState, runtime_ms: float) -> ProbeMetrics:
    last_l2 = float(torch.norm(probe.last_hidden - baseline.last_hidden, p=2).item())
    query_l2 = float(torch.norm(probe.query_hidden_mean - baseline.query_hidden_mean, p=2).item())
    base_probs = F.softmax(baseline.next_logits.float(), dim=-1)
    probe_probs = F.softmax(probe.next_logits.float(), dim=-1)
    kl = float(F.kl_div(probe_probs.log(), base_probs, reduction="sum").item())
    return ProbeMetrics(
        last_token_hidden_l2_change=last_l2,
        query_token_hidden_l2_change=query_l2,
        next_token_logit_kl=kl,
        next_token_entropy_change=_entropy(probe.next_logits) - _entropy(baseline.next_logits),
        top_logit_margin_change=_top_margin(probe.next_logits) - _top_margin(baseline.next_logits),
        value_norm_change=probe.value_norm - baseline.value_norm,
        attention_mass_change=probe.attention_mass - baseline.attention_mass,
        probe_runtime_ms=runtime_ms,
    )


def zscore_map(values: Dict[int, float]) -> Dict[int, float]:
    finite = [float(v) for v in values.values() if math.isfinite(float(v))]
    if not finite:
        return {k: 0.0 for k in values}
    mean = statistics.mean(finite)
    std = statistics.pstdev(finite) if len(finite) > 1 else 0.0
    if std < 1e-12:
        return {k: 0.0 for k in values}
    return {k: (float(v) - mean) / std for k, v in values.items()}


def compute_block_utility(block_id: int, metrics: ProbeMetrics, sample_z: Dict[str, Dict[int, float]]) -> float:
    hidden_change = metrics.last_token_hidden_l2_change + 0.5 * metrics.query_token_hidden_l2_change
    entropy_reduction = -metrics.next_token_entropy_change
    _ = hidden_change
    return (
        sample_z["hidden"].get(block_id, 0.0)
        + sample_z["logit_kl"].get(block_id, 0.0)
        + 0.5 * sample_z["entropy_reduction"].get(block_id, 0.0)
        + 0.5 * sample_z["margin"].get(block_id, 0.0)
        - 0.5 * sample_z["redundancy"].get(block_id, 0.0)
    )


def mcu_select_blocks(
    original_selected: Set[int],
    block_utilities: Dict[int, float],
    budget_k: int,
    protected: Set[int],
    token_lengths: Sequence[int],
    *,
    max_replace_frac: float = MAX_REPLACE_FRAC,
    utility_margin_min: float = UTILITY_MARGIN_MIN,
) -> Tuple[Set[int], str, List[Tuple[int, int]], float]:
    if budget_k <= 0 or not original_selected:
        return set(original_selected), "empty_budget", [], 0.0
    selected_sorted = sorted(original_selected, key=lambda bid: (block_utilities.get(bid, float("inf")), bid))
    pruned_candidates = sorted(
        ((bid, block_utilities[bid]) for bid in block_utilities if bid not in original_selected),
        key=lambda item: (-item[1], item[0]),
    )
    max_swaps = max(0, int(math.floor(max_replace_frac * budget_k)))
    if max_swaps == 0:
        return set(original_selected), "zero_swap_budget", [], 0.0
    current = set(original_selected)
    swaps: List[Tuple[int, int]] = []
    margins: List[float] = []
    removable = [bid for bid in selected_sorted if bid not in protected]
    for _ in range(max_swaps):
        if not removable or not pruned_candidates:
            break
        victim = removable.pop(0)
        victim_u = block_utilities.get(victim, 0.0)
        best_pruned = None
        best_u = float("-inf")
        for bid, util in pruned_candidates:
            if bid in current:
                continue
            if util > best_u:
                best_u = util
                best_pruned = bid
        if best_pruned is None:
            break
        margin = best_u - victim_u
        if margin < utility_margin_min:
            break
        current.remove(victim)
        current.add(best_pruned)
        swaps.append((victim, best_pruned))
        margins.append(margin)
        pruned_candidates = [(bid, u) for bid, u in pruned_candidates if bid != best_pruned]
    if not swaps:
        return set(original_selected), "no_positive_utility_swaps", [], 0.0
    if len(current) != len(original_selected):
        return set(original_selected), "swap_count_mismatch", [], 0.0
    if token_count(current, token_lengths) != token_count(original_selected, token_lengths):
        return set(original_selected), "token_budget_mismatch", [], 0.0
    return current, "mcu_swap_applied", swaps, float(statistics.mean(margins))


def run_mcu_counterfactual_selection(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    input_ids: torch.Tensor,
    query_text: str,
    block_texts: Sequence[str],
    token_lengths: Sequence[int],
    original_selected: Set[int],
    all_scores: Dict[int, float],
    *,
    block_size: int = BLOCK_SIZE,
    sample_id: Any = "",
) -> Dict[str, Any]:
    seq_len = int(input_ids.shape[0])
    query_start, _ = find_query_span(input_ids, tokenizer, query_text)
    protected = protected_block_ids(seq_len, block_size, query_start)
    query_overlaps = block_query_overlaps(block_texts, query_text)
    token_map = block_token_map(block_texts)
    budget_k = len(original_selected)

    if not original_selected:
        return {
            "sample_id": sample_id,
            "original_selected_block_ids": [],
            "mcu_selected_block_ids": [],
            "selected": set(),
            "fallback_used": True,
            "fallback_reason": "missing_original_selected_block_ids",
            "probe_runtime_ms": 0.0,
        }

    active_with_protected = set(original_selected) | protected
    candidate_pool = build_candidate_pool(original_selected, all_scores, query_overlaps, protected)
    candidate_pool |= protected
    baseline_compact = compact_input_ids(input_ids, active_with_protected, block_size)
    if int(baseline_compact.shape[0]) == 0:
        return {
            "sample_id": sample_id,
            "original_selected_block_ids": sorted(original_selected),
            "mcu_selected_block_ids": sorted(original_selected),
            "selected": set(original_selected),
            "fallback_used": True,
            "fallback_reason": "empty_baseline_compact_input",
            "probe_runtime_ms": 0.0,
        }

    q_start_compact = max(0, int(baseline_compact.shape[0]) - (seq_len - query_start))
    probe_runtime_total = 0.0
    sample_probe_start = time.perf_counter()
    fallback_reason = ""

    try:
        t0 = time.perf_counter()
        baseline_state = run_forward_probe(model, baseline_compact, q_start_compact)
        probe_runtime_total += (time.perf_counter() - t0) * 1000.0
    except Exception as exc:
        return {
            "sample_id": sample_id,
            "original_selected_block_ids": sorted(original_selected),
            "mcu_selected_block_ids": sorted(original_selected),
            "selected": set(original_selected),
            "fallback_used": True,
            "fallback_reason": f"baseline_probe_failed:{exc}",
            "probe_runtime_ms": probe_runtime_total,
        }

    selected_in_pool = [bid for bid in sorted(candidate_pool) if bid in original_selected and bid not in protected][:MAX_REMOVE_PROBES]
    pruned_in_pool = [bid for bid in sorted(candidate_pool) if bid not in original_selected][:MAX_SWAP_PROBES]
    victim_for_swap = min(
        (bid for bid in original_selected if bid not in protected),
        key=lambda bid: all_scores.get(bid, float("inf")),
        default=None,
    )

    raw_signals: Dict[int, Dict[str, float]] = {}
    probe_metrics_by_block: Dict[int, ProbeMetrics] = {}

    def record_probe(block_id: int, metrics: ProbeMetrics) -> None:
        hidden_sig = metrics.last_token_hidden_l2_change + 0.5 * metrics.query_token_hidden_l2_change
        raw_signals[block_id] = {
            "hidden": hidden_sig,
            "logit_kl": metrics.next_token_logit_kl,
            "entropy_reduction": -metrics.next_token_entropy_change,
            "margin": metrics.top_logit_margin_change,
        }
        probe_metrics_by_block[block_id] = metrics

    for block_id in selected_in_pool:
        if time.perf_counter() - sample_probe_start > PROBE_SLOW_SAMPLE_SEC:
            fallback_reason = "probe_runtime_exceeded_sample_budget"
            break
        active = set(active_with_protected)
        active.discard(block_id)
        compact = compact_input_ids(input_ids, active, block_size)
        if int(compact.shape[0]) == 0:
            continue
        q_start = max(0, int(compact.shape[0]) - (seq_len - query_start))
        t0 = time.perf_counter()
        try:
            state = run_forward_probe(model, compact, q_start)
        except Exception as exc:
            return {
                "sample_id": sample_id,
                "original_selected_block_ids": sorted(original_selected),
                "mcu_selected_block_ids": sorted(original_selected),
                "selected": set(original_selected),
                "fallback_used": True,
                "fallback_reason": f"remove_probe_failed:{exc}",
                "probe_runtime_ms": probe_runtime_total,
            }
        runtime_ms = (time.perf_counter() - t0) * 1000.0
        probe_runtime_total += runtime_ms
        record_probe(block_id, compare_probe_states(baseline_state, state, runtime_ms))

    if victim_for_swap is not None:
        for block_id in pruned_in_pool:
            if time.perf_counter() - sample_probe_start > PROBE_SLOW_SAMPLE_SEC:
                fallback_reason = fallback_reason or "probe_runtime_exceeded_sample_budget"
                break
            active = set(active_with_protected)
            active.discard(victim_for_swap)
            active.add(block_id)
            compact = compact_input_ids(input_ids, active, block_size)
            q_start = max(0, int(compact.shape[0]) - (seq_len - query_start))
            t0 = time.perf_counter()
            try:
                state = run_forward_probe(model, compact, q_start)
            except Exception as exc:
                return {
                    "sample_id": sample_id,
                    "original_selected_block_ids": sorted(original_selected),
                    "mcu_selected_block_ids": sorted(original_selected),
                    "selected": set(original_selected),
                    "fallback_used": True,
                    "fallback_reason": f"swap_probe_failed:{exc}",
                    "probe_runtime_ms": probe_runtime_total,
                }
            runtime_ms = (time.perf_counter() - t0) * 1000.0
            probe_runtime_total += runtime_ms
            record_probe(block_id, compare_probe_states(baseline_state, state, runtime_ms))

    if not probe_metrics_by_block:
        return {
            "sample_id": sample_id,
            "original_selected_block_ids": sorted(original_selected),
            "mcu_selected_block_ids": sorted(original_selected),
            "selected": set(original_selected),
            "fallback_used": True,
            "fallback_reason": fallback_reason or "no_successful_probes",
            "probe_runtime_ms": probe_runtime_total,
        }

    redundancy_map = {bid: _dc_selected_redundancy({bid}, token_map) for bid in candidate_pool}
    sample_z = {
        "hidden": zscore_map({bid: raw_signals.get(bid, {}).get("hidden", 0.0) for bid in raw_signals}),
        "logit_kl": zscore_map({bid: raw_signals.get(bid, {}).get("logit_kl", 0.0) for bid in raw_signals}),
        "entropy_reduction": zscore_map({bid: raw_signals.get(bid, {}).get("entropy_reduction", 0.0) for bid in raw_signals}),
        "margin": zscore_map({bid: raw_signals.get(bid, {}).get("margin", 0.0) for bid in raw_signals}),
        "redundancy": zscore_map(redundancy_map),
    }
    block_utilities: Dict[int, float] = {}
    for bid in candidate_pool:
        if bid in probe_metrics_by_block:
            block_utilities[bid] = compute_block_utility(bid, probe_metrics_by_block[bid], sample_z)
        elif bid in all_scores:
            block_utilities[bid] = float(all_scores[bid])

    utility_values = [float(v) for v in block_utilities.values() if math.isfinite(v)]
    utility_spread = (max(utility_values) - min(utility_values)) if len(utility_values) >= 2 else 0.0
    mcu_selected, select_reason, swaps, utility_margin = mcu_select_blocks(
        original_selected, block_utilities, budget_k, protected, token_lengths,
    )

    use_fallback = False
    if fallback_reason:
        use_fallback = True
    elif utility_spread < 0.05:
        use_fallback = True
        fallback_reason = f"utility_spread_too_small={utility_spread:.4f}"
    elif select_reason in {"no_positive_utility_swaps", "zero_swap_budget"}:
        use_fallback = True
        fallback_reason = select_reason

    if use_fallback:
        mcu_selected = set(original_selected)
        swaps = []
        utility_margin = 0.0

    removed = sorted(original_selected - mcu_selected)
    added = sorted(mcu_selected - original_selected)
    hidden_vals = [m.last_token_hidden_l2_change + 0.5 * m.query_token_hidden_l2_change for m in probe_metrics_by_block.values()]
    kl_vals = [m.next_token_logit_kl for m in probe_metrics_by_block.values()]
    entropy_vals = [m.next_token_entropy_change for m in probe_metrics_by_block.values()]
    margin_vals = [m.top_logit_margin_change for m in probe_metrics_by_block.values()]
    original_redundancy = _dc_selected_redundancy(original_selected, token_map)
    mcu_redundancy = _dc_selected_redundancy(mcu_selected, token_map)
    original_query_overlap = float(sum(query_overlaps.get(bid, 0.0) for bid in original_selected) / max(1, len(original_selected)))
    mcu_query_overlap = float(sum(query_overlaps.get(bid, 0.0) for bid in mcu_selected) / max(1, len(mcu_selected)))

    return {
        "sample_id": sample_id,
        "original_selected_block_ids": sorted(original_selected),
        "mcu_selected_block_ids": sorted(mcu_selected),
        "selected": set(mcu_selected),
        "candidate_pool_size": len(candidate_pool),
        "num_counterfactual_probes": len(probe_metrics_by_block),
        "selection_changed": bool(removed or added),
        "removed_block_ids": removed,
        "added_block_ids": added,
        "removed_block_utility": [block_utilities.get(bid, 0.0) for bid in removed],
        "added_block_utility": [block_utilities.get(bid, 0.0) for bid in added],
        "utility_margin": utility_margin,
        "hidden_change_mean": statistics.mean(hidden_vals) if hidden_vals else 0.0,
        "logit_kl_mean": statistics.mean(kl_vals) if kl_vals else 0.0,
        "entropy_change_mean": statistics.mean(entropy_vals) if entropy_vals else 0.0,
        "top_logit_margin_change_mean": statistics.mean(margin_vals) if margin_vals else 0.0,
        "probe_runtime_ms": probe_runtime_total,
        "original_query_overlap": original_query_overlap,
        "mcu_query_overlap": mcu_query_overlap,
        "query_overlap_delta": mcu_query_overlap - original_query_overlap,
        "original_redundancy": original_redundancy,
        "mcu_redundancy": mcu_redundancy,
        "redundancy_delta": mcu_redundancy - original_redundancy,
        "fallback_used": use_fallback,
        "fallback_reason": fallback_reason or select_reason,
        "mcu_select_reason": select_reason,
        "utility_spread": utility_spread,
        "swaps": swaps,
        "selected_jaccard_vs_original": selected_jaccard(original_selected, mcu_selected),
        "budget_preserved": len(mcu_selected) == len(original_selected),
    }


MCU_DEBUG_EXCLUDE: Dict[str, Set[int]] = {
    "hotpotqa": {0, 1, 2, 3, 4, 5},
    "musique": {0, 1, 2, 3, 4, 20},
}
MCU_HELDOUT_POOL_SIZE = 64
MCU_SENSITIVE_TOP_K_PER_TASK = 10


def heldout_pool_indices(task: str, pool_size: int = MCU_HELDOUT_POOL_SIZE) -> List[int]:
    excluded = MCU_DEBUG_EXCLUDE.get(task, set())
    return [idx for idx in range(pool_size) if idx not in excluded]


def heldout_pool_keys(tasks: Sequence[str] = ("hotpotqa", "musique")) -> List[Tuple[str, int]]:
    return [(task, idx) for task in tasks for idx in heldout_pool_indices(task)]


def added_minus_removed_utility(
    removed_util: Any,
    added_util: Any,
) -> float:
    removed = [float(x) for x in (parse_json_field(removed_util) or []) if math.isfinite(float(x))]
    added = [float(x) for x in (parse_json_field(added_util) or []) if math.isfinite(float(x))]
    if not removed and not added:
        return 0.0
    return (statistics.mean(added) if added else 0.0) - (statistics.mean(removed) if removed else 0.0)


def pool_zscore(values: Sequence[float]) -> List[float]:
    nums = [float(v) if math.isfinite(float(v)) else 0.0 for v in values]
    if len(nums) < 2:
        return [0.0] * len(nums)
    mean_val = statistics.mean(nums)
    std_val = statistics.pstdev(nums)
    if std_val < 1e-12:
        return [0.0] * len(nums)
    return [(v - mean_val) / std_val for v in nums]


def annotate_mcu_sensitivity_scores(rows: List[Dict[str, Any]]) -> None:
    """Attach added_minus_removed_utility and mcu_sensitivity_score in place."""
    if not rows:
        return
    for row in rows:
        row["added_minus_removed_utility"] = added_minus_removed_utility(
            row.get("removed_block_utility"),
            row.get("added_block_utility"),
        )
        row["fallback_used_numeric"] = 1.0 if bool(row.get("fallback_used")) else 0.0

    z_margin = pool_zscore([float(r.get("utility_margin", 0.0) or 0.0) for r in rows])
    z_kl = pool_zscore([float(r.get("logit_kl_mean", 0.0) or 0.0) for r in rows])
    z_hidden = pool_zscore([float(r.get("hidden_change_mean", 0.0) or 0.0) for r in rows])
    z_top_margin = pool_zscore([
        abs(float(r.get("top_logit_margin_change_mean", 0.0) or 0.0)) for r in rows
    ])
    z_added_removed = pool_zscore([float(r.get("added_minus_removed_utility", 0.0) or 0.0) for r in rows])
    z_fallback = pool_zscore([float(r.get("fallback_used_numeric", 0.0) or 0.0) for r in rows])

    for idx, row in enumerate(rows):
        row["mcu_sensitivity_score"] = (
            z_margin[idx]
            + z_kl[idx]
            + z_hidden[idx]
            + 0.5 * z_top_margin[idx]
            + 0.5 * z_added_removed[idx]
            - 0.5 * z_fallback[idx]
        )


def select_top_sensitive_samples(
    rows: Sequence[Mapping[str, Any]],
    *,
    top_k: int = MCU_SENSITIVE_TOP_K_PER_TASK,
    tasks: Sequence[str] = ("hotpotqa", "musique"),
) -> Dict[str, List[Dict[str, Any]]]:
    selected: Dict[str, List[Dict[str, Any]]] = {}
    for task in tasks:
        task_rows = [
            dict(row) for row in rows
            if str(row.get("task", "")) == task and "mcu_sensitivity_score" in row
        ]
        if len(task_rows) < top_k:
            raise RuntimeError(
                f"insufficient sensitive candidates for {task}: {len(task_rows)} < {top_k}"
            )
        ranked = sorted(task_rows, key=lambda row: -float(row["mcu_sensitivity_score"]))
        selected[task] = ranked[:top_k]
    return selected


def frozen_row_to_cached_result(row: Mapping[str, Any]) -> Dict[str, Any]:
    original = {int(x) for x in parse_json_field(row.get("original_selected_block_ids")) or []}
    mcu = {int(x) for x in parse_json_field(row.get("mcu_selected_block_ids")) or []}
    selected = mcu if mcu else sorted(original)
    return {
        "task": row.get("task", ""),
        "sample_index": row.get("sample_index", ""),
        "sample_id": row.get("sample_id", ""),
        "original_selected_block_ids": sorted(original),
        "mcu_selected_block_ids": sorted(selected),
        "fallback_used": str(row.get("fallback_used", "")).lower() in ("1", "true", "yes"),
        "fallback_reason": str(row.get("fallback_reason", "") or ""),
        "selection_changed": str(row.get("selection_changed", "")).lower() in ("1", "true", "yes"),
        "selected_jaccard_vs_original": float(row.get("selected_jaccard_vs_original", 1.0) or 1.0),
        "removed_block_ids": parse_json_field(row.get("removed_block_ids")) or [],
        "added_block_ids": parse_json_field(row.get("added_block_ids")) or [],
        "removed_block_utility": parse_json_field(row.get("removed_block_utility")) or [],
        "added_block_utility": parse_json_field(row.get("added_block_utility")) or [],
        "utility_margin": float(row.get("utility_margin", 0.0) or 0.0),
        "hidden_change_mean": float(row.get("hidden_change_mean", 0.0) or 0.0),
        "logit_kl_mean": float(row.get("logit_kl_mean", 0.0) or 0.0),
        "entropy_change_mean": float(row.get("entropy_change_mean", 0.0) or 0.0),
        "top_logit_margin_change_mean": float(row.get("top_logit_margin_change_mean", 0.0) or 0.0),
        "probe_runtime_ms": float(row.get("probe_runtime_ms", 0.0) or 0.0),
        "original_query_overlap": float(row.get("original_query_overlap", 0.0) or 0.0),
        "mcu_query_overlap": float(row.get("mcu_query_overlap", 0.0) or 0.0),
        "query_overlap_delta": float(row.get("query_overlap_delta", 0.0) or 0.0),
        "original_redundancy": float(row.get("original_redundancy", 0.0) or 0.0),
        "mcu_redundancy": float(row.get("mcu_redundancy", 0.0) or 0.0),
        "redundancy_delta": float(row.get("redundancy_delta", 0.0) or 0.0),
        "budget_preserved": str(row.get("budget_preserved", "true")).lower() in ("1", "true", "yes"),
    }


def load_frozen_selections_from_audit_csv(path: Path) -> Dict[str, Dict[str, Any]]:
    import csv

    out: Dict[str, Dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            task = str(row.get("task", ""))
            sample_index = int(row.get("sample_index", -1))
            if not task or sample_index < 0:
                continue
            cached = frozen_row_to_cached_result(row)
            cached["task"] = task
            cached["sample_index"] = sample_index
            out[f"{task}:{sample_index}"] = cached
    return out


def write_frozen_selections_jsonl(path: Path, selections: Mapping[str, Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for key in sorted(selections.keys()):
            row = dict(selections[key])
            task, sample_index = key.split(":", 1)
            row.setdefault("task", task)
            row.setdefault("sample_index", int(sample_index))
            selected = row.get("selected", row.get("mcu_selected_block_ids", []))
            if isinstance(selected, set):
                selected = sorted(selected)
            row["mcu_selected_block_ids"] = sorted(int(x) for x in (row.get("mcu_selected_block_ids") or selected))
            row["original_selected_block_ids"] = sorted(int(x) for x in (row.get("original_selected_block_ids") or []))
            row["selected"] = row["mcu_selected_block_ids"]
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
