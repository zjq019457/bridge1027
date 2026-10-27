"""SlimInfer reliability extensions (methods 2–5): margin-stable, multi-window, adaptive budget, policy gate."""
from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

import torch

from bridgekv.confidence_gated_sliminfer import (
    _trim_to_budget,
    compute_pruning_confidence,
    confidence_gated_active_ids,
)


def _id_score_map(original_block_ids: Sequence[int], block_scores: torch.Tensor) -> Dict[int, float]:
    return {int(bid): float(block_scores[i].item()) for i, bid in enumerate(original_block_ids)}


def margin_stable_active_ids(
    *,
    original_block_ids: Sequence[int],
    block_scores: torch.Tensor,
    top_k_indices: torch.Tensor,
    must_keep_block_ids: Set[int],
    num_blocks_to_keep: int,
    prev_active_block_ids: Optional[Set[int]] = None,
    swap_margin_ratio: float = 0.08,
) -> Tuple[Set[int], Dict[str, Any]]:
    """Only swap blocks when incoming score beats removable block by a clear margin."""
    top_local = [int(i) for i in top_k_indices.detach().cpu().tolist()]
    top_k_block_ids = {int(original_block_ids[i]) for i in top_local}
    id_to_score = _id_score_map(original_block_ids, block_scores)
    baseline = set(top_k_block_ids) | {int(x) for x in must_keep_block_ids}
    baseline = _trim_to_budget(
        baseline,
        original_block_ids=original_block_ids,
        block_scores=block_scores,
        must_keep_block_ids=must_keep_block_ids,
        num_blocks_to_keep=num_blocks_to_keep,
    )

    if not prev_active_block_ids:
        return baseline, {"gated": False, "swaps_applied": 0, "mode": "no_prev_active"}

    scores = block_scores.detach().float().flatten()
    span = max(float(scores.max().item() - scores.min().item()), 1e-6)
    margin_abs = float(swap_margin_ratio) * span

    final = set(prev_active_block_ids) | {int(x) for x in must_keep_block_ids}
    candidate_add = sorted(
        (top_k_block_ids - final),
        key=lambda bid: id_to_score.get(bid, float("-inf")),
        reverse=True,
    )
    swaps = 0
    for bid in candidate_add:
        if len(final) >= num_blocks_to_keep:
            removable = [b for b in final if b not in must_keep_block_ids]
            if not removable:
                break
            worst = min(removable, key=lambda b: id_to_score.get(b, float("-inf")))
            if id_to_score.get(bid, float("-inf")) - id_to_score.get(worst, float("-inf")) >= margin_abs:
                final.remove(worst)
                final.add(bid)
                swaps += 1
            else:
                break
        else:
            final.add(bid)

    final = _trim_to_budget(
        final,
        original_block_ids=original_block_ids,
        block_scores=block_scores,
        must_keep_block_ids=must_keep_block_ids,
        num_blocks_to_keep=num_blocks_to_keep,
    )
    inter = len(final & baseline)
    union = len(final | baseline)
    return final, {
        "gated": swaps > 0 or final != baseline,
        "swaps_applied": swaps,
        "swap_margin_ratio": float(swap_margin_ratio),
        "margin_abs": round(margin_abs, 6),
        "jaccard_vs_silm_topk": round(inter / max(1, union), 6),
        "num_selected": len(final),
    }


from transformers.models.llama.modeling_llama import repeat_kv


def compute_multi_window_block_scores(
    q_for_pruning: torch.Tensor,
    represent_keys: torch.Tensor,
    original_block_ids: Sequence[int],
    block_to_kernel_map: Dict[int, Tuple[int, int]],
    num_key_value_groups: int,
    windows: Sequence[int] = (1, 4, 8),
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Consensus block scores from multiple local query windows (gold-free)."""
    head_dim = q_for_pruning.shape[-1]
    k_kernels_repeated = repeat_kv(represent_keys, num_key_value_groups)
    per_window: List[torch.Tensor] = []
    used_windows: List[int] = []
    q_len = int(q_for_pruning.shape[2])

    for w in windows:
        w = max(1, int(w))
        if w > q_len:
            continue
        q_slice = q_for_pruning if w >= q_len else q_for_pruning[:, :, -w:, :]
        kernel_scores = torch.einsum("bhid,bhkd->bhik", q_slice, k_kernels_repeated) / math.sqrt(head_dim)
        kernel_scores_flat = kernel_scores.mean(dim=[0, 1, 2])
        block_scores = torch.full(
            (len(original_block_ids),),
            -torch.inf,
            device=q_for_pruning.device,
            dtype=kernel_scores_flat.dtype,
        )
        offset = 0
        for i, block_id in enumerate(original_block_ids):
            if block_id in block_to_kernel_map:
                _, num_kernels = block_to_kernel_map[block_id]
                chunk = kernel_scores_flat[offset : offset + num_kernels]
                block_scores[i] = torch.max(chunk) if chunk.numel() else block_scores[i]
                offset += num_kernels
        per_window.append(block_scores)
        used_windows.append(w)

    if not per_window:
        zeros = torch.zeros(len(original_block_ids), device=q_for_pruning.device)
        return zeros, {"windows": [], "mode": "fallback_empty"}

    stacked = torch.stack(per_window, dim=0)
    # Conservative consensus: a block must look good under all windows.
    consensus = stacked.min(dim=0).values
    # Tie-break with mean so flat min ties still separate.
    consensus = 0.7 * consensus + 0.3 * stacked.mean(dim=0)
    return consensus, {"windows": used_windows, "mode": "min_blend_consensus"}


def reliability_policy_from_confidence(confidence: float, *, low: float = 0.35, high: float = 0.65) -> str:
    if confidence >= high:
        return "original"
    if confidence <= low:
        return "conservative"
    return "original"


def adaptive_retain_tokens_k(
    layer_idx: int,
    *,
    difficulty: float,
    easy_schedule: Tuple[int, int, int] = (12, 6, 4),
    hard_schedule: Tuple[int, int, int] = (12, 8, 6),
    prune_layers: Sequence[int] = (9, 18, 26),
) -> Optional[float]:
    """Return fixed_keep_tokens (in k units) for a prune layer given sample difficulty in [0,1]."""
    layers = [int(x) for x in prune_layers]
    if layer_idx not in layers:
        return None
    pos = layers.index(layer_idx)
    sched = hard_schedule if float(difficulty) >= 0.5 else easy_schedule
    return float(sched[pos])


def reliability_gated_active_ids(
    *,
    original_block_ids: Sequence[int],
    block_scores: torch.Tensor,
    top_k_indices: torch.Tensor,
    must_keep_block_ids: Set[int],
    num_blocks_to_keep: int,
    prev_active_block_ids: Optional[Set[int]] = None,
    policy: str = "original",
    confidence_threshold: float = 0.35,
) -> Tuple[Set[int], Dict[str, Any]]:
    if policy == "conservative":
        return confidence_gated_active_ids(
            original_block_ids=original_block_ids,
            block_scores=block_scores,
            top_k_indices=top_k_indices,
            must_keep_block_ids=must_keep_block_ids,
            num_blocks_to_keep=num_blocks_to_keep,
            prev_active_block_ids=prev_active_block_ids,
            confidence_threshold=confidence_threshold,
        )
    top_k_block_ids = {int(original_block_ids[i]) for i in top_k_indices.detach().cpu().tolist()}
    final = set(top_k_block_ids) | {int(x) for x in must_keep_block_ids}
    final = _trim_to_budget(
        final,
        original_block_ids=original_block_ids,
        block_scores=block_scores,
        must_keep_block_ids=must_keep_block_ids,
        num_blocks_to_keep=num_blocks_to_keep,
    )
    return final, {"policy": policy, "gated": False, "num_selected": len(final)}


def _z(values: Sequence[float], x: float) -> float:
    if not values:
        return 0.0
    mu = sum(values) / len(values)
    var = sum((v - mu) ** 2 for v in values) / max(1, len(values) - 1)
    std = math.sqrt(var) if var > 1e-12 else 1.0
    return (float(x) - mu) / std


def rank_stability_scores(
    *,
    block_ids: Sequence[int],
    layer_ranks: Mapping[int, Mapping[int, int]],
    prune_layers: Sequence[int] = (9, 18, 26),
) -> Dict[int, float]:
    """Higher = block stayed near top across prune layers (gold-free, from SILM ranks)."""
    out: Dict[int, float] = {}
    for bid in block_ids:
        ranks = [layer_ranks.get(int(pl), {}).get(int(bid)) for pl in prune_layers]
        ranks = [r for r in ranks if r is not None]
        if not ranks:
            out[int(bid)] = 0.0
            continue
        # stability = mean inverse rank normalized
        out[int(bid)] = float(sum(1.0 / (1.0 + float(r)) for r in ranks) / len(ranks))
    return out


def apply_rank_stability_adjustment(
    block_scores: torch.Tensor,
    original_block_ids: Sequence[int],
    stability: Mapping[int, float],
    *,
    weight: float = 0.12,
) -> torch.Tensor:
    adjusted = block_scores.clone().float()
    vals = [float(stability.get(int(bid), 0.0)) for bid in original_block_ids]
    for i, bid in enumerate(original_block_ids):
        adjusted[i] = adjusted[i] + weight * _z(vals, float(stability.get(int(bid), 0.0)))
    return adjusted


def reliability_full_active_ids(
    *,
    original_block_ids: Sequence[int],
    block_scores: torch.Tensor,
    top_k_indices: torch.Tensor,
    must_keep_block_ids: Set[int],
    num_blocks_to_keep: int,
    prev_active_block_ids: Optional[Set[int]] = None,
    confidence_threshold: float = 0.35,
    swap_margin_ratio: float = 0.08,
) -> Tuple[Set[int], Dict[str, Any]]:
    """Stack confidence gate then margin-safe swap."""
    gated, cgs_meta = confidence_gated_active_ids(
        original_block_ids=original_block_ids,
        block_scores=block_scores,
        top_k_indices=top_k_indices,
        must_keep_block_ids=must_keep_block_ids,
        num_blocks_to_keep=num_blocks_to_keep,
        prev_active_block_ids=prev_active_block_ids,
        confidence_threshold=confidence_threshold,
    )
    # Re-score margin relative to gated set as pseudo-prev
    id_to_local = {int(bid): i for i, bid in enumerate(original_block_ids)}
    gated_local = {id_to_local[b] for b in gated if b in id_to_local}
    if gated_local:
        fake_top = torch.tensor(sorted(gated_local), dtype=torch.long)
    else:
        fake_top = top_k_indices
    final, mss_meta = margin_stable_active_ids(
        original_block_ids=original_block_ids,
        block_scores=block_scores,
        top_k_indices=fake_top,
        must_keep_block_ids=must_keep_block_ids,
        num_blocks_to_keep=num_blocks_to_keep,
        prev_active_block_ids=prev_active_block_ids,
        swap_margin_ratio=swap_margin_ratio,
    )
    return final, {"confidence_gated": cgs_meta, "margin_stable": mss_meta, "mode": "stacked"}


def rank_stability_sliminfer_active_ids(
    *,
    original_block_ids: Sequence[int],
    block_scores: torch.Tensor,
    top_k_indices: torch.Tensor,
    must_keep_block_ids: Set[int],
    num_blocks_to_keep: int,
    cfg: Dict[str, Any],
    layer_idx: int,
    prune_layers: Sequence[int] = (9, 18, 26),
    weight: float = 0.12,
) -> Tuple[Set[int], Dict[str, Any]]:
    layer_ranks: MutableMapping[int, Mapping[int, int]] = cfg.setdefault("_rss_layer_ranks", {})
    scores_list = block_scores.detach().float().cpu().tolist()
    score_map = {int(bid): float(scores_list[i]) for i, bid in enumerate(original_block_ids)}
    ordered = sorted(original_block_ids, key=lambda bid: score_map.get(int(bid), float("-inf")), reverse=True)
    layer_ranks[int(layer_idx)] = {int(bid): rank + 1 for rank, bid in enumerate(ordered)}

    available_layers = [int(pl) for pl in prune_layers if int(pl) in layer_ranks and int(pl) <= int(layer_idx)]
    stab = rank_stability_scores(
        block_ids=original_block_ids,
        layer_ranks=layer_ranks,
        prune_layers=available_layers,
    )
    adjusted = apply_rank_stability_adjustment(
        block_scores,
        original_block_ids,
        stab,
        weight=float(weight),
    )
    k = int(num_blocks_to_keep)
    _, adj_top = torch.topk(adjusted, k=k)
    silm_top = {int(original_block_ids[i]) for i in top_k_indices.detach().cpu().tolist()}
    final = {int(original_block_ids[i]) for i in adj_top.detach().cpu().tolist()} | {int(x) for x in must_keep_block_ids}
    final = _trim_to_budget(
        final,
        original_block_ids=original_block_ids,
        block_scores=adjusted,
        must_keep_block_ids=must_keep_block_ids,
        num_blocks_to_keep=k,
    )
    overlap = len(final & silm_top)
    union = len(final | silm_top)
    churn = 1.0 - overlap / max(1, k)
    return final, {
        "mode": "rank_stability_boost",
        "layers_used": available_layers,
        "weight": float(weight),
        "triggered": 1,
        "jaccard_vs_silm_topk": round(overlap / max(1, union), 6),
        "churn_rate": round(churn, 6),
        "num_selected": len(final),
        "sink_kept": int(bool(original_block_ids and int(original_block_ids[0]) in final)),
    }
