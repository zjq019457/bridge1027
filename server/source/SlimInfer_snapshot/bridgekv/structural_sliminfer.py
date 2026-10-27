"""Structural SlimInfer variants: Cluster-Continuity, Unit-Density, Diversity-Coverage.

All gold-free; same KV budget as sliminfer_original; official SlimInfer path only.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import torch

from bridgekv.confidence_gated_sliminfer import _trim_to_budget


def _must_keep(original_block_ids: Sequence[int]) -> Set[int]:
    out: Set[int] = set()
    if original_block_ids:
        out.add(int(original_block_ids[0]))
        out.add(int(original_block_ids[-1]))
    return out


def _score_map(original_block_ids: Sequence[int], block_scores: torch.Tensor) -> Dict[int, float]:
    vals = block_scores.detach().float().cpu().tolist()
    return {int(bid): float(vals[i]) for i, bid in enumerate(original_block_ids)}


def _percentile_rank(values: Mapping[int, float]) -> Dict[int, float]:
    if not values:
        return {}
    items = sorted(values.items(), key=lambda x: x[1])
    n = len(items)
    if n == 1:
        return {items[0][0]: 0.5}
    out: Dict[int, float] = {}
    for i, (bid, _) in enumerate(items):
        out[int(bid)] = i / max(1, n - 1)
    return out


def compute_unit_density_block_scores(
    *,
    block_scores: torch.Tensor,
    block_slices: Sequence[Tuple[int, int]],
    kernel_scores_flat: torch.Tensor,
    w_max: float = 0.50,
    w_density: float = 0.35,
    w_spike: float = 0.15,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Replace max-pooling with max + density - spike penalty."""
    device = block_scores.device
    dtype = block_scores.dtype
    n = int(block_scores.numel())
    if n == 0:
        return block_scores, {"applied": False}

    unit_max: List[float] = []
    unit_density: List[float] = []
    unit_conc: List[float] = []
    flat = kernel_scores_flat.detach().float()

    for i, (start, count) in enumerate(block_slices):
        if count <= 0:
            unit_max.append(float("-inf"))
            unit_density.append(0.0)
            unit_conc.append(0.0)
            continue
        units = flat[start : start + count]
        umax = float(torch.max(units).item())
        unit_max.append(umax)
        if count == 1:
            unit_density.append(1.0)
            unit_conc.append(1.0)
            continue
        med = float(torch.median(units).item())
        thresh = med if math.isfinite(med) else umax - 1e-6
        density = float((units >= thresh).sum().item()) / float(count)
        topk = torch.topk(units, k=min(3, int(count))).values
        mean_top = float(topk.mean().item())
        conc = umax / max(mean_top, 1e-8)
        unit_density.append(density)
        unit_conc.append(conc)

    rank_max = _percentile_rank({i: v for i, v in enumerate(unit_max) if math.isfinite(v)})
    rank_den = _percentile_rank({i: v for i, v in enumerate(unit_density)})
    rank_spike = _percentile_rank({i: v for i, v in enumerate(unit_conc)})

    combined = torch.full((n,), float("-inf"), device=device, dtype=dtype)
    for i in range(n):
        if i not in rank_max:
            continue
        val = (
            w_max * rank_max[i]
            + w_density * rank_den.get(i, 0.5)
            - w_spike * rank_spike.get(i, 0.5)
        )
        combined[i] = float(val)

    meta = {
        "applied": True,
        "w_max": w_max,
        "w_density": w_density,
        "w_spike": w_spike,
        "mean_unit_density": round(float(np.mean(unit_density)), 6) if unit_density else 0.0,
        "mean_concentration": round(float(np.mean(unit_conc)), 6) if unit_conc else 0.0,
    }
    return combined, meta


def _silm_baseline(
    original_block_ids: Sequence[int],
    block_scores: torch.Tensor,
    top_k_indices: torch.Tensor,
    must_keep_block_ids: Set[int],
    num_blocks_to_keep: int,
) -> Set[int]:
    top_local = [int(i) for i in top_k_indices.detach().cpu().tolist()]
    silm = {int(original_block_ids[i]) for i in top_local} | {int(x) for x in must_keep_block_ids}
    return _trim_to_budget(
        silm,
        original_block_ids=original_block_ids,
        block_scores=block_scores,
        must_keep_block_ids=must_keep_block_ids,
        num_blocks_to_keep=int(num_blocks_to_keep),
    )


def _churn_meta(baseline: Set[int], final: Set[int], budget: int) -> Dict[str, Any]:
    inter = len(baseline & final)
    union = len(baseline | final)
    return {
        "jaccard_vs_silm": round(inter / max(1, union), 6),
        "churn_rate": round(1.0 - inter / max(1, budget), 6),
        "num_selected": len(final),
    }


def cc_smooth_scores(idx_scores: Sequence[float], alpha: float = 0.25) -> List[float]:
    """Frozen CC local smoother (α default 0.25)."""
    n = len(idx_scores)
    smoothed = list(idx_scores)
    for i in range(n):
        left = idx_scores[i - 1] if i > 0 else idx_scores[i]
        right = idx_scores[i + 1] if i < n - 1 else idx_scores[i]
        if math.isfinite(idx_scores[i]):
            smoothed[i] = idx_scores[i] + alpha * (
                (left if math.isfinite(left) else 0.0) + (right if math.isfinite(right) else 0.0)
            )
    return smoothed


def cc_find_clusters(
    smoothed: Sequence[float],
    *,
    min_cluster_len: int = 2,
    quantile: float = 0.75,
) -> Tuple[List[List[int]], float]:
    """Contiguous runs with smoothed ≥ quantile threshold."""
    finite = [s for s in smoothed if math.isfinite(s)]
    if not finite:
        return [], float("-inf")
    thresh = float(np.quantile(finite, quantile))
    clusters: List[List[int]] = []
    run: List[int] = []
    for i, s in enumerate(smoothed):
        if math.isfinite(s) and s >= thresh:
            run.append(i)
        else:
            if len(run) >= min_cluster_len:
                clusters.append(run)
            run = []
    if len(run) >= min_cluster_len:
        clusters.append(run)
    return clusters, thresh


def cc_prepare_state(
    *,
    original_block_ids: Sequence[int],
    block_scores: torch.Tensor,
    top_k_indices: torch.Tensor,
    must_keep_block_ids: Set[int],
    num_blocks_to_keep: int,
    alpha: float = 0.25,
    min_cluster_len: int = 2,
) -> Dict[str, Any]:
    """Shared SlimInfer baseline + smoothed scores + clusters for CC family policies."""
    k = int(num_blocks_to_keep)
    must_keep = set(must_keep_block_ids) | _must_keep(original_block_ids)
    baseline = _silm_baseline(original_block_ids, block_scores, top_k_indices, must_keep, k)
    scores = _score_map(original_block_ids, block_scores)
    n = len(original_block_ids)
    idx_scores = [scores.get(int(original_block_ids[i]), float("-inf")) for i in range(n)]
    finite = [s for s in idx_scores if math.isfinite(s)]
    if n == 0 or not finite:
        return {
            "ok": False,
            "k": k,
            "n": n,
            "must_keep": must_keep,
            "baseline": baseline,
            "scores": scores,
            "idx_scores": idx_scores,
            "smoothed": list(idx_scores),
            "clusters": [],
            "thresh": float("-inf"),
            "original_block_ids": list(int(x) for x in original_block_ids),
        }
    smoothed = cc_smooth_scores(idx_scores, alpha=alpha)
    clusters, thresh = cc_find_clusters(smoothed, min_cluster_len=min_cluster_len)
    return {
        "ok": True,
        "k": k,
        "n": n,
        "must_keep": must_keep,
        "baseline": baseline,
        "scores": scores,
        "idx_scores": idx_scores,
        "smoothed": smoothed,
        "clusters": clusters,
        "thresh": thresh,
        "original_block_ids": list(int(x) for x in original_block_ids),
        "alpha": alpha,
        "min_cluster_len": min_cluster_len,
    }


def _cc_neighbor_and_cluster_blocks(
    *,
    original_block_ids: Sequence[int],
    clusters: Sequence[Sequence[int]],
    smoothed: Sequence[float],
    baseline: Set[int],
    max_neighbors: int,
) -> Tuple[Set[int], Set[int]]:
    n = len(original_block_ids)
    to_add: Set[int] = set()
    cluster_blocks: Set[int] = set()
    for cl in clusters:
        center_i = max(cl, key=lambda i: smoothed[i])
        for i in cl:
            cluster_blocks.add(int(original_block_ids[i]))
        for delta in range(1, max_neighbors + 1):
            for ni in (center_i - delta, center_i + delta):
                if 0 <= ni < n:
                    bid = int(original_block_ids[ni])
                    if bid not in baseline:
                        to_add.add(bid)
    return to_add, cluster_blocks


def _cc_apply_neighbor_swaps(
    *,
    baseline: Set[int],
    must_keep: Set[int],
    cluster_blocks: Set[int],
    to_add: Set[int],
    scores: Mapping[int, float],
    k: int,
    replacement_cap: float,
    protected_blocks: Optional[Set[int]] = None,
) -> Tuple[Set[int], int]:
    """Swap isolated non-cluster (and non-protected) blocks for neighbor candidates."""
    protected = set(protected_blocks or set())
    final = set(baseline)
    max_swap = max(0, int(math.floor(replacement_cap * max(0, k - len(must_keep)))))
    removable = sorted(
        [
            b
            for b in final
            if b not in must_keep and b not in cluster_blocks and b not in protected
        ],
        key=lambda b: scores.get(b, float("-inf")),
    )
    add_sorted = sorted(to_add - final, key=lambda b: scores.get(b, float("-inf")), reverse=True)
    swaps = 0
    ai = 0
    ri = 0
    while swaps < max_swap and ai < len(add_sorted) and ri < len(removable):
        new_b = add_sorted[ai]
        ai += 1
        if new_b in final:
            continue
        old_b = removable[ri]
        ri += 1
        if old_b in must_keep or old_b in protected:
            continue
        final.discard(old_b)
        final.add(new_b)
        swaps += 1
    return final, swaps


def _cc_finalize_meta(
    *,
    mode: str,
    baseline: Set[int],
    final: Set[int],
    must_keep: Set[int],
    cluster_blocks: Set[int],
    to_add: Set[int],
    original_block_ids: Sequence[int],
    k: int,
    swaps: int,
    max_swap: int,
    alpha: float,
    min_cluster_len: int,
    max_neighbors: int,
    replacement_cap: float,
    clusters_found: int,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    added = sorted(int(x) for x in (final - baseline))
    removed = sorted(int(x) for x in (baseline - final))
    neighbors_retained = sorted(int(x) for x in (to_add & final))
    isolated_removed = sorted(
        int(x) for x in removed if int(x) not in must_keep and int(x) not in cluster_blocks
    )
    sink_id = int(original_block_ids[0]) if original_block_ids else None
    meta: Dict[str, Any] = {
        "mode": mode,
        "triggered": int(swaps > 0 or len(added) > 0),
        "alpha": alpha,
        "min_cluster_len": min_cluster_len,
        "max_neighbors": max_neighbors,
        "replacement_cap": replacement_cap,
        "clusters_found": clusters_found,
        "swaps_applied": swaps,
        "neighbors_proposed": len(to_add),
        "neighbors_added": len(added),
        "neighbors_retained": len(neighbors_retained),
        "neighbor_retention_rate": round(len(neighbors_retained) / max(1, len(to_add)), 6),
        "isolated_replacements": len(isolated_removed),
        "isolated_replacement_rate": round(len(isolated_removed) / max(1, k), 6),
        "max_swap_cap": max_swap,
        "budget_parity": int(len(final) == k),
        "num_selected": len(final),
        "num_keep_target": k,
        "sink_kept": int(sink_id in final) if sink_id is not None else 0,
        "sink_violation": int(sink_id is not None and sink_id not in final),
        "added_block_ids": added,
        "removed_block_ids": removed,
        "neighbors_retained_ids": neighbors_retained,
        "isolated_removed_ids": isolated_removed,
    }
    if extra:
        meta.update(extra)
    meta.update(_churn_meta(baseline, final, k))
    return meta


def cluster_continuity_active_ids(
    *,
    original_block_ids: Sequence[int],
    block_scores: torch.Tensor,
    top_k_indices: torch.Tensor,
    must_keep_block_ids: Set[int],
    num_blocks_to_keep: int,
    alpha: float = 0.25,
    min_cluster_len: int = 2,
    max_neighbors: int = 2,
    replacement_cap: float = 0.05,
) -> Tuple[Set[int], Dict[str, Any]]:
    """Smooth scores, find clusters, swap isolated low blocks for cluster neighbors."""
    state = cc_prepare_state(
        original_block_ids=original_block_ids,
        block_scores=block_scores,
        top_k_indices=top_k_indices,
        must_keep_block_ids=must_keep_block_ids,
        num_blocks_to_keep=num_blocks_to_keep,
        alpha=alpha,
        min_cluster_len=min_cluster_len,
    )
    k = int(state["k"])
    baseline: Set[int] = set(state["baseline"])
    must_keep: Set[int] = set(state["must_keep"])
    if not state["ok"]:
        return baseline, {"mode": "cc_sliminfer", "triggered": 0, "reason": "no_finite_scores"}

    scores: Dict[int, float] = state["scores"]
    smoothed: List[float] = state["smoothed"]
    clusters: List[List[int]] = state["clusters"]
    to_add, cluster_blocks = _cc_neighbor_and_cluster_blocks(
        original_block_ids=original_block_ids,
        clusters=clusters,
        smoothed=smoothed,
        baseline=baseline,
        max_neighbors=max_neighbors,
    )
    final, swaps = _cc_apply_neighbor_swaps(
        baseline=baseline,
        must_keep=must_keep,
        cluster_blocks=cluster_blocks,
        to_add=to_add,
        scores=scores,
        k=k,
        replacement_cap=replacement_cap,
    )
    final = _trim_to_budget(
        final,
        original_block_ids=original_block_ids,
        block_scores=block_scores,
        must_keep_block_ids=must_keep,
        num_blocks_to_keep=k,
    )
    max_swap = max(0, int(math.floor(replacement_cap * max(0, k - len(must_keep)))))
    meta = _cc_finalize_meta(
        mode="cc_sliminfer_v1",
        baseline=baseline,
        final=final,
        must_keep=must_keep,
        cluster_blocks=cluster_blocks,
        to_add=to_add,
        original_block_ids=original_block_ids,
        k=k,
        swaps=swaps,
        max_swap=max_swap,
        alpha=alpha,
        min_cluster_len=min_cluster_len,
        max_neighbors=max_neighbors,
        replacement_cap=replacement_cap,
        clusters_found=len(clusters),
    )
    return final, meta


def _block_key_vector(
    represent_keys: torch.Tensor,
    block_to_kernel_map: Mapping[int, Tuple[int, int]],
    block_id: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    if block_id not in block_to_kernel_map:
        return None
    start, count = block_to_kernel_map[block_id]
    if count <= 0:
        return None
    keys = represent_keys[:, :, start : start + count, :].detach().float()
    vec = keys.mean(dim=(0, 1, 2))
    return vec.to(device)


def _build_normalized_key_matrix(
    original_block_ids: Sequence[int],
    represent_keys: torch.Tensor,
    block_to_kernel_map: Mapping[int, Tuple[int, int]],
    device: torch.device,
) -> Tuple[Dict[int, int], Optional[torch.Tensor]]:
    """Map block_id -> row index in L2-normalized key matrix [N, D]."""
    rows: List[torch.Tensor] = []
    bid_to_row: Dict[int, int] = {}
    for bid in original_block_ids:
        bid = int(bid)
        vec = _block_key_vector(represent_keys, block_to_kernel_map, bid, device)
        if vec is None:
            continue
        bid_to_row[bid] = len(rows)
        rows.append(vec)
    if not rows:
        return {}, None
    mat = torch.stack(rows, dim=0)
    mat = mat / mat.norm(dim=1, keepdim=True).clamp_min(1e-8)
    return bid_to_row, mat


def divcov_active_ids(
    *,
    original_block_ids: Sequence[int],
    block_scores: torch.Tensor,
    top_k_indices: torch.Tensor,
    must_keep_block_ids: Set[int],
    num_blocks_to_keep: int,
    represent_keys: torch.Tensor,
    block_to_kernel_map: Mapping[int, Tuple[int, int]],
    mmr_lambda: float = 0.15,
) -> Tuple[Set[int], Dict[str, Any]]:
    """Greedy MMR selection on SlimInfer relevance with key-rep redundancy penalty."""
    k = int(num_blocks_to_keep)
    must_keep = set(must_keep_block_ids) | _must_keep(original_block_ids)
    baseline = _silm_baseline(original_block_ids, block_scores, top_k_indices, must_keep, k)
    scores = _score_map(original_block_ids, block_scores)
    device = block_scores.device

    bid_to_row, key_mat = _build_normalized_key_matrix(
        original_block_ids, represent_keys, block_to_kernel_map, device
    )

    rel_rank = _percentile_rank(scores)
    selected: Set[int] = set(must_keep)
    for bid in must_keep:
        if bid in original_block_ids:
            selected.add(int(bid))

    candidates = sorted(
        [int(b) for b in original_block_ids if int(b) not in selected],
        key=lambda b: rel_rank.get(b, 0.0),
        reverse=True,
    )
    # MMR only needs high-relevance pool; cap avoids O(n^2) Python/GPU sync on long contexts.
    pool_cap = max(k * 4, k + 32)
    candidates = candidates[:pool_cap]

    while len(selected) < k and candidates:
        best_bid = None
        best_util = float("-inf")
        pending = [bid for bid in candidates if bid not in selected]
        if not pending:
            break

        if key_mat is not None and bid_to_row:
            sel_rows = [bid_to_row[b] for b in selected if b in bid_to_row]
            cand_with_vec = [(bid, bid_to_row[bid]) for bid in pending if bid in bid_to_row]
            cand_no_vec = [bid for bid in pending if bid not in bid_to_row]

            if sel_rows and cand_with_vec:
                sel_mat = key_mat[torch.tensor(sel_rows, device=device, dtype=torch.long)]
                cand_rows = torch.tensor([r for _, r in cand_with_vec], device=device, dtype=torch.long)
                cand_mat = key_mat[cand_rows]
                max_sims = (cand_mat @ sel_mat.T).max(dim=1).values.detach().cpu().tolist()
                for (bid, _), max_sim in zip(cand_with_vec, max_sims):
                    util = rel_rank.get(bid, 0.0) - mmr_lambda * float(max_sim)
                    if util > best_util:
                        best_util = util
                        best_bid = bid
            for bid in cand_no_vec:
                util = rel_rank.get(bid, 0.0)
                if util > best_util:
                    best_util = util
                    best_bid = bid
        else:
            for bid in pending:
                util = rel_rank.get(bid, 0.0)
                if util > best_util:
                    best_util = util
                    best_bid = bid

        if best_bid is None:
            break
        selected.add(best_bid)

    final = _trim_to_budget(
        selected,
        original_block_ids=original_block_ids,
        block_scores=block_scores,
        must_keep_block_ids=must_keep,
        num_blocks_to_keep=k,
    )
    meta = {
        "mode": "divcov_sliminfer",
        "triggered": int(final != baseline),
        "mmr_lambda": mmr_lambda,
        "key_vectors_available": len(bid_to_row),
        "candidate_pool_cap": pool_cap,
        "sink_kept": int(original_block_ids[0] in final) if original_block_ids else 0,
    }
    meta.update(_churn_meta(baseline, final, k))
    return final, meta
