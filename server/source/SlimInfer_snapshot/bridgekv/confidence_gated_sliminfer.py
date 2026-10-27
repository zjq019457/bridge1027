"""Confidence-Gated SlimInfer: stabilize KV selection when block scores are ambiguous.

Uses only SlimInfer block scores (no gold, no task name, no external retrieval).
When pruning confidence is low, prefer overlap with the previous prune-layer active set
instead of aggressively replacing blocks.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import torch


def _safe_softmax(scores: torch.Tensor) -> torch.Tensor:
    if scores.numel() == 0:
        return scores
    shifted = scores - scores.max()
    exp = torch.exp(shifted)
    denom = exp.sum().clamp_min(1e-12)
    return exp / denom


def compute_pruning_confidence(
    block_scores: torch.Tensor,
    *,
    k: int,
    prev_active_local_indices: Optional[Set[int]] = None,
    top_k_local_indices: Optional[Sequence[int]] = None,
) -> Tuple[float, Dict[str, Any]]:
    """Return confidence in [0, 1] and diagnostic metadata (all gold-free)."""
    scores = block_scores.detach().float().flatten()
    n = int(scores.numel())
    k = max(1, min(int(k), n))
    meta: Dict[str, Any] = {"num_blocks": n, "k": k}

    if n <= 1:
        meta["reason"] = "trivial_num_blocks"
        return 1.0, meta

    sorted_scores, sorted_idx = torch.sort(scores, descending=True)
    top_idx = (
        [int(i) for i in top_k_local_indices]
        if top_k_local_indices is not None
        else sorted_idx[:k].tolist()
    )
    meta["top_indices_head"] = top_idx[: min(8, len(top_idx))]

    top1 = float(sorted_scores[0].item())
    topk = float(sorted_scores[k - 1].item())
    tail = float(sorted_scores[-1].item())
    span = max(top1 - tail, 1e-6)

    margin = (top1 - topk) / span
    if k < n:
        next_k = float(sorted_scores[k].item())
        boundary_gap = (topk - next_k) / span
    else:
        boundary_gap = margin
    meta["margin"] = round(margin, 6)
    meta["boundary_gap"] = round(boundary_gap, 6)

    probs = _safe_softmax(scores)
    entropy = float((-(probs * torch.log(probs.clamp_min(1e-12))).sum()).item())
    entropy_norm = entropy / max(math.log(float(n)), 1e-6)
    sharpness = 1.0 - min(1.0, max(0.0, entropy_norm))
    meta["entropy"] = round(entropy, 6)
    meta["entropy_norm"] = round(entropy_norm, 6)
    meta["sharpness"] = round(sharpness, 6)

    prev_overlap = 1.0
    if prev_active_local_indices and top_idx:
        prev = set(int(i) for i in prev_active_local_indices)
        cur = set(int(i) for i in top_idx)
        union = prev | cur
        prev_overlap = len(prev & cur) / max(1, len(union))
    meta["prev_active_jaccard"] = round(prev_overlap, 6)

    # Higher margin / gap / sharpness / prev agreement => more confident.
    confidence = (
        0.35 * min(1.0, margin / 0.25)
        + 0.25 * min(1.0, boundary_gap / 0.15)
        + 0.25 * sharpness
        + 0.15 * prev_overlap
    )
    confidence = float(min(1.0, max(0.0, confidence)))
    meta["confidence"] = round(confidence, 6)
    return confidence, meta


def _trim_to_budget(
    kept: Set[int],
    *,
    original_block_ids: Sequence[int],
    block_scores: torch.Tensor,
    must_keep_block_ids: Set[int],
    num_blocks_to_keep: int,
) -> Set[int]:
    if len(kept) <= num_blocks_to_keep:
        return set(kept)
    id_to_score = {int(bid): float(block_scores[i].item()) for i, bid in enumerate(original_block_ids)}
    removable = [int(bid) for bid in kept if int(bid) not in must_keep_block_ids]
    removable.sort(key=lambda bid: id_to_score.get(bid, float("-inf")))
    out = set(kept)
    for bid in removable:
        if len(out) <= num_blocks_to_keep:
            break
        out.discard(bid)
    return out


def confidence_gated_active_ids(
    *,
    original_block_ids: Sequence[int],
    block_scores: torch.Tensor,
    top_k_indices: torch.Tensor,
    must_keep_block_ids: Set[int],
    num_blocks_to_keep: int,
    prev_active_block_ids: Optional[Set[int]] = None,
    confidence_threshold: float = 0.35,
) -> Tuple[Set[int], Dict[str, Any]]:
    """Select active block IDs; when uncertain, stabilize against previous layer set."""
    k = int(num_blocks_to_keep)
    top_local = [int(i) for i in top_k_indices.detach().cpu().tolist()]
    prev_local: Optional[Set[int]] = None
    if prev_active_block_ids:
        id_to_local = {int(bid): i for i, bid in enumerate(original_block_ids)}
        prev_local = {id_to_local[bid] for bid in prev_active_block_ids if bid in id_to_local}

    confidence, conf_meta = compute_pruning_confidence(
        block_scores,
        k=k,
        prev_active_local_indices=prev_local,
        top_k_local_indices=top_local,
    )
    top_k_block_ids = {int(original_block_ids[i]) for i in top_local}
    baseline = set(top_k_block_ids) | {int(x) for x in must_keep_block_ids}
    baseline = _trim_to_budget(
        baseline,
        original_block_ids=original_block_ids,
        block_scores=block_scores,
        must_keep_block_ids=must_keep_block_ids,
        num_blocks_to_keep=num_blocks_to_keep,
    )

    gated = False
    final_ids = baseline
    if confidence < float(confidence_threshold) and prev_active_block_ids:
        gated = True
        prev = {int(x) for x in prev_active_block_ids}
        stable_core = (top_k_block_ids & prev) | {int(x) for x in must_keep_block_ids}
        pool = top_k_block_ids | prev
        id_to_score = {int(bid): float(block_scores[i].item()) for i, bid in enumerate(original_block_ids)}
        ranked_pool = sorted(pool, key=lambda bid: id_to_score.get(bid, float("-inf")), reverse=True)
        final_ids = set(stable_core)
        for bid in ranked_pool:
            if len(final_ids) >= num_blocks_to_keep:
                break
            final_ids.add(int(bid))
        final_ids = _trim_to_budget(
            final_ids,
            original_block_ids=original_block_ids,
            block_scores=block_scores,
            must_keep_block_ids=must_keep_block_ids,
            num_blocks_to_keep=num_blocks_to_keep,
        )

    jaccard_vs_baseline = 1.0
    if baseline:
        inter = len(final_ids & baseline)
        union = len(final_ids | baseline)
        jaccard_vs_baseline = inter / max(1, union)

    details = {
        "gated": gated,
        "confidence": confidence,
        "confidence_threshold": float(confidence_threshold),
        "num_selected": len(final_ids),
        "num_keep_target": int(num_blocks_to_keep),
        "jaccard_vs_silm_topk": round(jaccard_vs_baseline, 6),
        **conf_meta,
    }
    return final_ids, details


def effective_swap_threshold(
    base_threshold: float,
    confidence: float,
    *,
    min_threshold: float = 0.5,
) -> float:
    """Lower swap threshold when confidence is low => easier to reuse prior active set."""
    conf = min(1.0, max(0.0, float(confidence)))
    base = float(base_threshold)
    floor = min(base, float(min_threshold))
    return floor + conf * (base - floor)
