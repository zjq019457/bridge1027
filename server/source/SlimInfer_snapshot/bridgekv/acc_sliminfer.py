"""Advanced Cluster Continuity (ACC) + budget-neutral DRM-ACC / DRM-Swap.

Mainline methods are K-budget. drm_acc_add_sliminfer is K+M tradeoff only.
See ACC_DRM_IDEA.md.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import torch

from bridgekv.confidence_gated_sliminfer import _trim_to_budget
from bridgekv.cc_topology_policies import (
    DEFAULT_CC_GATE_CONFIG,
    is_multi_locus_topology,
    is_spike_topology,
    score_topology_features,
)
from bridgekv.rescuekv_scoring import drm_s0_score, mmr_select_rescue_ids
from bridgekv.structural_sliminfer import (
    _cc_neighbor_and_cluster_blocks,
    _churn_meta,
    _must_keep,
    _silm_baseline,
    cc_prepare_state,
    cluster_continuity_active_ids,
)

# acc_drm_v1: ONLY spike thresholds differ from Phase-1A/v0.
# tau_spike frozen from unlabeled isolated_spike_ratio ~p95 on Phase-1A 72-sample audit.
# See results/acc_drm_v0_audit/ and SPIKE_CALIBRATION_SPEC.md after Phase-1B.
ACC_DRM_VERSION = "acc_drm_v1"
DEFAULT_ACC_CONFIG: Dict[str, float] = {
    **DEFAULT_CC_GATE_CONFIG,
    "tau_nei": 0.18,
    # v0 had tau_spike=0.45 (never fired). v1 joint calibration:
    "tau_spike": 0.28,
    "tau_spike_nei_max": 0.18,
    "tau_cont": 0.10,
    "tau_str": 0.15,
    "tau_second": 0.30,
    "tau_gap": 0.08,
    "tau_margin": 0.20,
    "diversify_top_g": 2.0,
    "cluster_share_cap": 0.35,
    "n_buckets": 8.0,
    "rescue_ratio": 0.05,
    "alpha_diff": 0.5,
    "lambda_red": 0.25,
    "eta_rescue": 0.25,
    "eta_active": 0.25,
}

# Frozen Phase-1A spike params (audit-only reference; not used by default runtime).
ACC_DRM_V0_SPIKE_CONFIG: Dict[str, float] = {
    "tau_spike": 0.45,
    "tau_spike_nei_max": 1.0,
    "tau_margin": 0.20,
}


def _cfg(extra: Optional[Mapping[str, float]] = None) -> Dict[str, float]:
    out = dict(DEFAULT_ACC_CONFIG)
    if extra:
        for k, v in extra.items():
            try:
                out[str(k)] = float(v)
            except (TypeError, ValueError):
                continue
    return out


def _cluster_masses(
    clusters: Sequence[Sequence[int]],
    smoothed: Sequence[float],
) -> List[Tuple[float, int, List[int], int]]:
    """(mass, length, indices, center_i) sorted by mass desc."""
    rows: List[Tuple[float, int, List[int], int]] = []
    for cl in clusters:
        mass = float(sum(max(float(smoothed[i]), 0.0) for i in cl))
        center_i = max(cl, key=lambda i: smoothed[i])
        rows.append((mass, len(cl), list(cl), int(center_i)))
    rows.sort(key=lambda x: x[0], reverse=True)
    return rows


def _bucket_index(i: int, n: int, n_buckets: int) -> int:
    if n <= 0 or n_buckets <= 0:
        return 0
    return min(n_buckets - 1, int(i * n_buckets / n))


def _features_for_state(
    state: Mapping[str, Any],
    *,
    max_neighbors: int,
    cfg: Mapping[str, float],
) -> Dict[str, float]:
    return score_topology_features(
        original_block_ids=state["original_block_ids"],
        idx_scores=state["idx_scores"],
        smoothed=state["smoothed"],
        clusters=state["clusters"],
        baseline=set(state["baseline"]),
        k=int(state["k"]),
        max_neighbors=max_neighbors,
        n_buckets=int(cfg.get("n_buckets", 8)),
        ctx_len_ref_blocks=float(cfg.get("ctx_len_ref_blocks", 512.0)),
        softmax_tau=float(cfg.get("softmax_tau", 1.0)),
    )


def build_safe_sacrifice_pool(
    *,
    active: Set[int],
    original_block_ids: Sequence[int],
    scores: Mapping[int, float],
    smoothed: Sequence[float],
    clusters: Sequence[Sequence[int]],
    must_keep: Set[int],
    n_buckets: int = 8,
    protect_second_locus: bool = True,
    top_g: int = 2,
) -> Tuple[List[int], Dict[str, Any]]:
    """Ranked sacrifice candidates (ascending = drop first). Protected blocks excluded."""
    n = len(original_block_ids)
    bid_to_i = {int(original_block_ids[i]): i for i in range(n)}
    masses = _cluster_masses(clusters, smoothed)
    cluster_blocks: Set[int] = set()
    centers: Set[int] = set()
    second_locus: Set[int] = set()
    for rank, (mass, _length, indices, center_i) in enumerate(masses):
        for i in indices:
            cluster_blocks.add(int(original_block_ids[i]))
        centers.add(int(original_block_ids[center_i]))
        if protect_second_locus and rank < top_g:
            for i in indices:
                second_locus.add(int(original_block_ids[i]))

    # Position-bucket sole coverage (unique active in bucket).
    B = max(1, int(n_buckets))
    bucket_members: List[List[int]] = [[] for _ in range(B)]
    for bid in active:
        i = bid_to_i.get(int(bid))
        if i is None:
            continue
        bucket_members[_bucket_index(i, n, B)].append(int(bid))
    sole_coverage = {bids[0] for bids in bucket_members if len(bids) == 1}

    # Isolated high spikes in active set.
    finite = [float(s) for s in scores.values() if math.isfinite(s)]
    q50 = float(sorted(finite)[len(finite) // 2]) if finite else 0.0
    q90 = float(sorted(finite)[int(0.9 * (len(finite) - 1))]) if finite else 0.0
    protected_spikes: Set[int] = set()
    for bid in active:
        i = bid_to_i.get(int(bid))
        if i is None:
            continue
        s = float(scores.get(bid, float("-inf")))
        if not math.isfinite(s) or s < q90:
            continue
        left_low = i == 0 or not math.isfinite(scores.get(int(original_block_ids[i - 1]), float("nan"))) or float(
            scores.get(int(original_block_ids[i - 1]), 0.0)
        ) < q50
        right_low = i == n - 1 or not math.isfinite(
            scores.get(int(original_block_ids[min(n - 1, i + 1)]), float("nan"))
        ) or float(scores.get(int(original_block_ids[min(n - 1, i + 1)]), 0.0)) < q50
        if left_low and right_low:
            protected_spikes.add(int(bid))

    protected = set(must_keep) | centers | protected_spikes | sole_coverage
    if protect_second_locus and len(masses) >= 2:
        protected |= second_locus

    pool: List[Tuple[float, int]] = []
    for bid in active:
        bid = int(bid)
        if bid in protected:
            continue
        i = bid_to_i.get(bid)
        s = float(scores.get(bid, float("-inf")))
        # Low density: neighbors also low → easier sacrifice.
        dens = 0.0
        if i is not None:
            neigh = []
            for ni in (i - 1, i + 1):
                if 0 <= ni < n:
                    ns = float(scores.get(int(original_block_ids[ni]), float("-inf")))
                    if math.isfinite(ns):
                        neigh.append(ns)
            dens = float(sum(neigh) / len(neigh)) if neigh else 0.0
        # Redundancy proxy: proximity to other active smoothed scores.
        red = 0.0
        if i is not None and math.isfinite(smoothed[i]):
            for bid2 in active:
                j = bid_to_i.get(int(bid2))
                if j is None or j == i:
                    continue
                if abs(j - i) == 1 and math.isfinite(smoothed[j]):
                    red = max(red, 1.0 / max(abs(float(smoothed[i]) - float(smoothed[j])), 1e-3))
        # Prefer low score, low density, high redundancy, non-cluster.
        cluster_pen = 0.0 if bid not in cluster_blocks else 2.0
        rank_key = s + 0.15 * dens - 0.05 * red + cluster_pen
        pool.append((rank_key, bid))
    pool.sort(key=lambda x: x[0])
    ordered = [b for _, b in pool]
    meta = {
        "sacrifice_pool_size": len(ordered),
        "protected_size": len(protected & active),
        "protected_centers": len(centers & active),
        "protected_spikes": len(protected_spikes & active),
        "protected_sole_coverage": len(sole_coverage & active),
        "protected_second_locus": len(second_locus & active) if protect_second_locus else 0,
        "protected_must_keep": len(must_keep & active),
        "protected_spike_ids": sorted(protected_spikes & active),
        "protected_second_locus_ids": sorted(second_locus & active) if protect_second_locus else [],
    }
    return ordered, meta


def enforce_cluster_budget_cap(
    *,
    selected: Set[int],
    original_block_ids: Sequence[int],
    scores: Mapping[int, float],
    clusters: Sequence[Sequence[int]],
    smoothed: Sequence[float],
    must_keep: Set[int],
    k: int,
    max_neighbors: int,
    cluster_share_cap: float,
    n_buckets: int,
) -> Tuple[Set[int], Dict[str, Any]]:
    """Cap per-cluster halo and per-bucket share; refill with global ranking if under budget."""
    n = len(original_block_ids)
    bid_to_i = {int(original_block_ids[i]): i for i in range(n)}
    final = set(selected)
    triggered = 0
    max_per_cluster = 1 + 2 * int(max_neighbors)
    masses = _cluster_masses(clusters, smoothed)

    for mass, _length, indices, center_i in masses:
        halo_idxs = set(indices)
        for delta in range(1, int(max_neighbors) + 1):
            for ni in (center_i - delta, center_i + delta):
                if 0 <= ni < n:
                    halo_idxs.add(ni)
        halo_bids = [int(original_block_ids[i]) for i in halo_idxs if int(original_block_ids[i]) in final]
        if len(halo_bids) <= max_per_cluster:
            continue
        triggered += 1
        ranked = sorted(halo_bids, key=lambda b: scores.get(b, float("-inf")), reverse=True)
        keep = set(ranked[:max_per_cluster]) | (must_keep & set(halo_bids))
        # Keep center always.
        keep.add(int(original_block_ids[center_i]))
        for bid in halo_bids:
            if bid not in keep:
                final.discard(bid)

    # Position bucket share.
    B = max(1, int(n_buckets))
    max_per_bucket = max(1, int(math.floor(float(cluster_share_cap) * k)))
    buckets: List[List[int]] = [[] for _ in range(B)]
    for bid in list(final):
        i = bid_to_i.get(int(bid))
        if i is None:
            continue
        buckets[_bucket_index(i, n, B)].append(int(bid))
    for members in buckets:
        if len(members) <= max_per_bucket:
            continue
        triggered += 1
        ranked = sorted(members, key=lambda b: scores.get(b, float("-inf")), reverse=True)
        keep = set(ranked[:max_per_bucket]) | (must_keep & set(members))
        for bid in members:
            if bid not in keep:
                final.discard(bid)

    # Refill to k.
    ranked_all = sorted(
        [int(b) for b in original_block_ids if int(b) not in final],
        key=lambda b: scores.get(b, float("-inf")),
        reverse=True,
    )
    for bid in ranked_all:
        if len(final) >= k:
            break
        final.add(bid)
    return final, {"cluster_budget_cap_triggered": int(triggered > 0), "cluster_budget_cap_events": triggered}


def _safe_neighbor_swaps(
    *,
    baseline: Set[int],
    to_add: Set[int],
    must_keep: Set[int],
    original_block_ids: Sequence[int],
    scores: Mapping[int, float],
    smoothed: Sequence[float],
    clusters: Sequence[Sequence[int]],
    k: int,
    replacement_cap: float,
    n_buckets: int,
    protect_second_locus: bool,
    top_g: int,
) -> Tuple[Set[int], int, Dict[str, Any]]:
    final = set(baseline)
    max_swap = max(0, int(math.floor(replacement_cap * max(0, k - len(must_keep)))))
    pool, pool_meta = build_safe_sacrifice_pool(
        active=final,
        original_block_ids=original_block_ids,
        scores=scores,
        smoothed=smoothed,
        clusters=clusters,
        must_keep=must_keep,
        n_buckets=n_buckets,
        protect_second_locus=protect_second_locus,
        top_g=top_g,
    )
    add_sorted = sorted(to_add - final, key=lambda b: scores.get(b, float("-inf")), reverse=True)
    swaps = 0
    ai = 0
    ri = 0
    sacrificed: List[int] = []
    prot_spikes = set(int(x) for x in (pool_meta.get("protected_spike_ids") or []))
    prot_second = set(int(x) for x in (pool_meta.get("protected_second_locus_ids") or []))
    spike_repl = 0
    second_repl = 0
    while swaps < max_swap and ai < len(add_sorted) and ri < len(pool):
        new_b = add_sorted[ai]
        ai += 1
        if new_b in final:
            continue
        old_b = pool[ri]
        ri += 1
        if old_b in must_keep or old_b not in final:
            continue
        if old_b in prot_spikes:
            spike_repl += 1
        if old_b in prot_second:
            second_repl += 1
        final.discard(old_b)
        final.add(new_b)
        sacrificed.append(int(old_b))
        swaps += 1
    pool_meta = dict(pool_meta)
    pool_meta["high_confidence_spike_replacement_count"] = spike_repl
    pool_meta["second_locus_replacement_count"] = second_repl
    pool_meta["sacrifice_used"] = sacrificed
    return final, swaps, pool_meta


def acc_sliminfer_active_ids(
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
    acc_cfg: Optional[Mapping[str, float]] = None,
) -> Tuple[Set[int], Dict[str, Any]]:
    """Advanced Cluster Continuity: multi-locus + safe sacrifice + spike guard + caps."""
    cfg = _cfg(acc_cfg)
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
    features = _features_for_state(state, max_neighbors=max_neighbors, cfg=cfg)

    # Reference CC set for churn vs CC.
    cc_set, cc_meta = cluster_continuity_active_ids(
        original_block_ids=original_block_ids,
        block_scores=block_scores,
        top_k_indices=top_k_indices,
        must_keep_block_ids=must_keep_block_ids,
        num_blocks_to_keep=num_blocks_to_keep,
        alpha=alpha,
        min_cluster_len=min_cluster_len,
        max_neighbors=max_neighbors,
        replacement_cap=replacement_cap,
    )

    meta_base: Dict[str, Any] = {
        "mode": "acc_sliminfer",
        "acc_drm_version": ACC_DRM_VERSION,
        "features": features,
        "spike_condition_triggered": 0,
        "spike_protection_triggered": 0,
        "spike_protection_changed_selection": 0,
        "multi_locus_triggered": 0,
        "cluster_budget_cap_triggered": 0,
        "safe_sacrifice_used": 0,
        "high_confidence_spike_replacement_count": 0,
        "second_locus_replacement_count": 0,
        "fallback_policy": "none",
        "spike_thresholds": {
            "tau_spike": float(cfg.get("tau_spike", 0.28)),
            "tau_margin": float(cfg.get("tau_margin", 0.20)),
            "tau_spike_nei_max": float(cfg.get("tau_spike_nei_max", 0.18)),
        },
    }

    if not state["ok"]:
        meta_base.update({"triggered": 0, "reason": "no_finite_scores", "fallback_policy": "original"})
        meta_base.update(_churn_meta(baseline, baseline, k))
        meta_base["jaccard_vs_cc"] = 1.0
        return baseline, meta_base

    scores: Dict[int, float] = state["scores"]
    smoothed: List[float] = state["smoothed"]
    clusters: List[List[int]] = state["clusters"]
    n_buckets = int(cfg.get("n_buckets", 8))
    top_g = max(1, int(cfg.get("diversify_top_g", 2)))

    # 3) Spike protection (joint unlabeled condition).
    if is_spike_topology(features, cfg):
        changed = int(baseline != cc_set)
        meta_base.update(
            {
                "spike_condition_triggered": 1,
                "spike_protection_triggered": 1,
                "spike_protection_changed_selection": changed,
                "fallback_policy": "sliminfer_original",
                "triggered": 0,
                "policy_chosen": "original",
            }
        )
        meta_base.update(_churn_meta(baseline, baseline, k))
        meta_base["jaccard_vs_cc"] = round(len(baseline & cc_set) / max(1, len(baseline | cc_set)), 6)
        meta_base["churn_vs_cc"] = round(1.0 - len(baseline & cc_set) / max(1, k), 6)
        return baseline, meta_base

    multi = is_multi_locus_topology(features, cfg)
    meta_base["multi_locus_triggered"] = int(multi)
    masses = _cluster_masses(clusters, smoothed)

    if multi and len(masses) >= 2:
        meta_base["fallback_policy"] = "multi_locus_cc"
        # Reserve locus seats.
        g = min(top_g, len(masses))
        locus = masses[:g]
        total_mass = sum(m for m, _, _, _ in locus) or 1.0
        free = max(0, k - len(must_keep))
        quotas: List[int] = []
        for idx, (mass, _, _, _) in enumerate(locus):
            if idx == len(locus) - 1:
                quotas.append(max(1, free - sum(quotas)))
            else:
                quotas.append(max(1, int(round(free * (mass / total_mass)))))
        while sum(quotas) > free and any(q > 1 for q in quotas):
            for i in range(len(quotas)):
                if quotas[i] > 1 and sum(quotas) > free:
                    quotas[i] -= 1

        final: Set[int] = set(must_keep)
        locus_blocks: Set[int] = set()
        for (mass, _length, indices, center_i), q in zip(locus, quotas):
            ranked = sorted(
                [int(original_block_ids[i]) for i in indices],
                key=lambda b: scores.get(b, float("-inf")),
                reverse=True,
            )
            take = ranked[: max(1, q)]
            final.update(take)
            locus_blocks.update(ranked)
            # Neighbor candidates around each locus center.
        to_add: Set[int] = set()
        for _mass, _length, indices, center_i in locus:
            for delta in range(1, max_neighbors + 1):
                for ni in (center_i - delta, center_i + delta):
                    if 0 <= ni < len(original_block_ids):
                        bid = int(original_block_ids[ni])
                        if bid not in final:
                            to_add.add(bid)

        # Fill remaining seats with SlimInfer ranking.
        if len(final) > k:
            final = _trim_to_budget(
                final,
                original_block_ids=original_block_ids,
                block_scores=block_scores,
                must_keep_block_ids=must_keep,
                num_blocks_to_keep=k,
            )
        ranked_all = sorted(
            [int(b) for b in original_block_ids if int(b) not in final],
            key=lambda b: scores.get(b, float("-inf")),
            reverse=True,
        )
        for bid in ranked_all:
            if len(final) >= k:
                break
            final.add(bid)

        final, swaps, pool_meta = _safe_neighbor_swaps(
            baseline=final,
            to_add=to_add,
            must_keep=must_keep | locus_blocks,
            original_block_ids=original_block_ids,
            scores=scores,
            smoothed=smoothed,
            clusters=clusters,
            k=k,
            replacement_cap=replacement_cap,
            n_buckets=n_buckets,
            protect_second_locus=True,
            top_g=g,
        )
        meta_base["safe_sacrifice_used"] = int(swaps > 0)
        meta_base["safe_sacrifice"] = pool_meta
        meta_base["swaps_applied"] = swaps
        meta_base["high_confidence_spike_replacement_count"] = int(
            pool_meta.get("high_confidence_spike_replacement_count", 0) or 0
        )
        meta_base["second_locus_replacement_count"] = int(
            pool_meta.get("second_locus_replacement_count", 0) or 0
        )
        meta_base["multi_locus_quotas"] = quotas
    else:
        meta_base["fallback_policy"] = "cc_sliminfer_v1_safe"
        # Frozen CC neighbor set, but swap via Safe Sacrifice Pool.
        to_add, cluster_blocks = _cc_neighbor_and_cluster_blocks(
            original_block_ids=original_block_ids,
            clusters=clusters,
            smoothed=smoothed,
            baseline=baseline,
            max_neighbors=max_neighbors,
        )
        final, swaps, pool_meta = _safe_neighbor_swaps(
            baseline=baseline,
            to_add=to_add,
            must_keep=must_keep | cluster_blocks,
            original_block_ids=original_block_ids,
            scores=scores,
            smoothed=smoothed,
            clusters=clusters,
            k=k,
            replacement_cap=replacement_cap,
            n_buckets=n_buckets,
            protect_second_locus=False,
            top_g=1,
        )
        meta_base["safe_sacrifice_used"] = int(swaps > 0)
        meta_base["safe_sacrifice"] = pool_meta
        meta_base["swaps_applied"] = swaps
        meta_base["high_confidence_spike_replacement_count"] = int(
            pool_meta.get("high_confidence_spike_replacement_count", 0) or 0
        )
        meta_base["second_locus_replacement_count"] = int(
            pool_meta.get("second_locus_replacement_count", 0) or 0
        )

    # 4) Cluster budget cap.
    final, cap_meta = enforce_cluster_budget_cap(
        selected=final,
        original_block_ids=original_block_ids,
        scores=scores,
        clusters=clusters,
        smoothed=smoothed,
        must_keep=must_keep,
        k=k,
        max_neighbors=max_neighbors,
        cluster_share_cap=float(cfg.get("cluster_share_cap", 0.35)),
        n_buckets=n_buckets,
    )
    meta_base.update(cap_meta)

    final = _trim_to_budget(
        final,
        original_block_ids=original_block_ids,
        block_scores=block_scores,
        must_keep_block_ids=must_keep,
        num_blocks_to_keep=k,
    )

    added = sorted(int(x) for x in (final - baseline))
    removed = sorted(int(x) for x in (baseline - final))
    sink_id = int(original_block_ids[0]) if original_block_ids else None
    meta_base.update(
        {
            "triggered": int(len(added) > 0 or meta_base.get("multi_locus_triggered") or meta_base.get("cluster_budget_cap_triggered")),
            "policy_chosen": meta_base["fallback_policy"],
            "clusters_found": len(clusters),
            "budget_parity": int(len(final) == k),
            "num_selected": len(final),
            "num_keep_target": k,
            "over_budget": int(max(0, len(final) - k)),
            "sink_kept": int(sink_id in final) if sink_id is not None else 0,
            "sink_violation": int(sink_id is not None and sink_id not in final),
            "added_block_ids": added,
            "removed_block_ids": removed,
            "alpha": alpha,
            "max_neighbors": max_neighbors,
            "replacement_cap": replacement_cap,
        }
    )
    meta_base.update(_churn_meta(baseline, final, k))
    meta_base["jaccard_vs_cc"] = round(len(final & cc_set) / max(1, len(final | cc_set)), 6)
    meta_base["churn_vs_cc"] = round(1.0 - len(final & cc_set) / max(1, k), 6)
    meta_base["cc_operator_meta"] = {
        "swaps_applied": cc_meta.get("swaps_applied"),
        "clusters_found": cc_meta.get("clusters_found"),
    }
    # Topology stats for audit.
    masses = _cluster_masses(clusters, smoothed)
    meta_base["cluster_topology"] = {
        "n_clusters": len(masses),
        "first_cluster_mass": masses[0][0] if masses else 0.0,
        "second_cluster_mass": masses[1][0] if len(masses) > 1 else 0.0,
        "second_over_first": round(masses[1][0] / max(masses[0][0], 1e-8), 6) if len(masses) > 1 else 0.0,
        "inter_cluster_gap_ratio": features.get("cluster_gap_ratio", 0.0),
        "cluster_length_max": features.get("cluster_length", 0.0),
    }
    return final, meta_base


def _local_to_global_score_tensor(
    original_block_ids: Sequence[int],
    block_scores: torch.Tensor,
) -> torch.Tensor:
    device = block_scores.device
    if not original_block_ids:
        return torch.empty(0, device=device, dtype=torch.float32)
    n = max(int(b) for b in original_block_ids) + 1
    out = torch.full((n,), float("-inf"), device=device, dtype=torch.float32)
    vals = block_scores.detach().float().reshape(-1)
    for i, bid in enumerate(original_block_ids):
        if i < vals.numel():
            out[int(bid)] = vals[i]
    return out


def drm_budget_neutral_swap_from_active(
    *,
    active: Set[int],
    original_block_ids: Sequence[int],
    block_scores: torch.Tensor,
    must_keep_block_ids: Set[int],
    num_blocks_to_keep: int,
    block_reps: Optional[torch.Tensor] = None,
    clusters: Optional[Sequence[Sequence[int]]] = None,
    smoothed: Optional[Sequence[float]] = None,
    scores_map: Optional[Mapping[int, float]] = None,
    replacement_cap: float = 0.05,
    acc_cfg: Optional[Mapping[str, float]] = None,
    mode: str = "drm_swap_sliminfer",
) -> Tuple[Set[int], Dict[str, Any]]:
    """Gold-free DRM rescue: swap inactive high-DRM into Safe Sacrifice Pool."""
    cfg = _cfg(acc_cfg)
    k = int(num_blocks_to_keep)
    must_keep = set(must_keep_block_ids) | _must_keep(original_block_ids)
    final = set(active)
    if len(final) != k:
        final = _trim_to_budget(
            final | must_keep,
            original_block_ids=original_block_ids,
            block_scores=block_scores,
            must_keep_block_ids=must_keep,
            num_blocks_to_keep=k,
        )

    scores = dict(scores_map) if scores_map is not None else {
        int(bid): float(block_scores[i].item()) if i < block_scores.numel() else float("-inf")
        for i, bid in enumerate(original_block_ids)
    }
    smoothed_list = list(smoothed) if smoothed is not None else [
        scores.get(int(bid), float("-inf")) for bid in original_block_ids
    ]
    clusters_list: List[List[int]] = [list(c) for c in (clusters or [])]

    inactive = [int(b) for b in original_block_ids if int(b) not in final]
    active_list = sorted(int(b) for b in final)
    base_global = _local_to_global_score_tensor(original_block_ids, block_scores)
    max_swap = max(0, int(math.floor(float(cfg.get("rescue_ratio", replacement_cap)) * max(0, k - len(must_keep)))))

    meta: Dict[str, Any] = {
        "mode": mode,
        "acc_drm_version": ACC_DRM_VERSION,
        "drm_rescue_triggered": 0,  # legacy alias of drm_swap_accepted>0
        "drm_swaps_applied": 0,
        "drm_candidate_available": 0,
        "drm_candidate_eligible": 0,
        "drm_swap_accepted": 0,
        "drm_swap_rejected": 0,
        "rejection_reason": [],
        "accepted_swaps_per_layer": 0,
        "max_swap_cap": max_swap,
        "drm_candidates": [],
        "sacrifice_used": [],
        "high_confidence_spike_replacement_count": 0,
        "second_locus_replacement_count": 0,
    }
    if max_swap <= 0 or not inactive:
        reasons = []
        if max_swap <= 0:
            reasons.append("max_swap_cap_zero")
        if not inactive:
            reasons.append("no_inactive_blocks")
        meta["rejection_reason"] = reasons
        meta.update(_churn_meta(set(active), final, k))
        meta["budget_parity"] = int(len(final) == k)
        sink_id = int(original_block_ids[0]) if original_block_ids else None
        meta["sink_kept"] = int(sink_id in final) if sink_id is not None else 0
        meta["sink_violation"] = int(sink_id is not None and sink_id not in final)
        meta["over_budget"] = int(max(0, len(final) - k))
        return final, meta

    s0, drm_meta = drm_s0_score(
        base_global,
        block_reps,
        active_list,
        inactive,
        alpha_diff=float(cfg.get("alpha_diff", 0.5)),
        lambda_red=float(cfg.get("lambda_red", 0.25)),
    )
    finite_inactive = [
        int(b)
        for b in inactive
        if int(b) < s0.numel() and math.isfinite(float(s0[int(b)].item()))
    ]
    meta["drm_candidate_available"] = int(len(finite_inactive) > 0)

    # Prefer MMR selection when reps exist; else top scores. Cap unchanged (no strength bump).
    if block_reps is not None:
        rescued, _mmr = mmr_select_rescue_ids(
            inactive_ids=inactive,
            s0_scores=s0,
            block_reps=block_reps,
            active_ids=active_list,
            m=max_swap,
            eta_rescue=float(cfg.get("eta_rescue", 0.25)),
            eta_active=float(cfg.get("eta_active", 0.25)),
        )
        candidates = [int(x) for x in rescued]
    else:
        scored = [(float(s0[int(b)].item()) if int(b) < s0.numel() else float("-inf"), int(b)) for b in inactive]
        scored = [x for x in scored if math.isfinite(x[0])]
        scored.sort(reverse=True)
        candidates = [b for _, b in scored[:max_swap]]

    meta["drm_candidate_eligible"] = int(len(candidates))
    meta["drm_candidates"] = candidates

    pool, pool_meta = build_safe_sacrifice_pool(
        active=final,
        original_block_ids=original_block_ids,
        scores=scores,
        smoothed=smoothed_list,
        clusters=clusters_list,
        must_keep=must_keep,
        n_buckets=int(cfg.get("n_buckets", 8)),
        protect_second_locus=True,
        top_g=max(1, int(cfg.get("diversify_top_g", 2))),
    )
    meta["safe_sacrifice"] = pool_meta
    meta["drm_used_base"] = bool(float(drm_meta.get("used_base", torch.tensor([0.0]))[0].item()) > 0.5)

    swaps = 0
    used_sac: List[int] = []
    used_res: List[int] = []
    reject_reasons: List[str] = []
    prot_spikes = set(int(x) for x in (pool_meta.get("protected_spike_ids") or []))
    prot_second = set(int(x) for x in (pool_meta.get("protected_second_locus_ids") or []))
    spike_repl = 0
    second_repl = 0
    ri = 0
    for new_b in candidates:
        if swaps >= max_swap:
            reject_reasons.append("max_swap_reached")
            break
        if new_b in final:
            reject_reasons.append("already_active")
            continue
        while ri < len(pool) and (pool[ri] in must_keep or pool[ri] not in final):
            ri += 1
        if ri >= len(pool):
            reject_reasons.append("empty_sacrifice_pool")
            break
        old_b = pool[ri]
        ri += 1
        if old_b in prot_spikes:
            spike_repl += 1
        if old_b in prot_second:
            second_repl += 1
        final.discard(old_b)
        final.add(new_b)
        used_sac.append(old_b)
        used_res.append(new_b)
        swaps += 1

    # Candidates not accepted count as rejected (strength unchanged).
    rejected = max(0, len(candidates) - swaps)

    final = _trim_to_budget(
        final,
        original_block_ids=original_block_ids,
        block_scores=block_scores,
        must_keep_block_ids=must_keep,
        num_blocks_to_keep=k,
    )
    sink_id = int(original_block_ids[0]) if original_block_ids else None
    meta.update(
        {
            "drm_rescue_triggered": int(swaps > 0),
            "drm_swaps_applied": swaps,
            "drm_swap_accepted": swaps,
            "drm_swap_rejected": rejected,
            "rejection_reason": reject_reasons,
            "accepted_swaps_per_layer": swaps,
            "sacrifice_used": used_sac,
            "rescued_ids": used_res,
            "high_confidence_spike_replacement_count": spike_repl,
            "second_locus_replacement_count": second_repl,
            "budget_parity": int(len(final) == k),
            "num_selected": len(final),
            "num_keep_target": k,
            "over_budget": int(max(0, len(final) - k)),
            "sink_kept": int(sink_id in final) if sink_id is not None else 0,
            "sink_violation": int(sink_id is not None and sink_id not in final),
            "added_block_ids": sorted(int(x) for x in (final - set(active))),
            "removed_block_ids": sorted(int(x) for x in (set(active) - final)),
        }
    )
    meta.update(_churn_meta(set(active), final, k))
    return final, meta


def drm_swap_sliminfer_active_ids(
    *,
    original_block_ids: Sequence[int],
    block_scores: torch.Tensor,
    top_k_indices: torch.Tensor,
    must_keep_block_ids: Set[int],
    num_blocks_to_keep: int,
    block_reps: Optional[torch.Tensor] = None,
    replacement_cap: float = 0.05,
    acc_cfg: Optional[Mapping[str, float]] = None,
) -> Tuple[Set[int], Dict[str, Any]]:
    must_keep = set(must_keep_block_ids) | _must_keep(original_block_ids)
    baseline = _silm_baseline(original_block_ids, block_scores, top_k_indices, must_keep, int(num_blocks_to_keep))
    state = cc_prepare_state(
        original_block_ids=original_block_ids,
        block_scores=block_scores,
        top_k_indices=top_k_indices,
        must_keep_block_ids=must_keep_block_ids,
        num_blocks_to_keep=num_blocks_to_keep,
    )
    final, meta = drm_budget_neutral_swap_from_active(
        active=baseline,
        original_block_ids=original_block_ids,
        block_scores=block_scores,
        must_keep_block_ids=must_keep,
        num_blocks_to_keep=num_blocks_to_keep,
        block_reps=block_reps,
        clusters=state.get("clusters") or [],
        smoothed=state.get("smoothed"),
        scores_map=state.get("scores"),
        replacement_cap=replacement_cap,
        acc_cfg=acc_cfg,
        mode="drm_swap_sliminfer",
    )
    meta["base_policy"] = "sliminfer_original"
    meta["jaccard_vs_silm"] = round(len(final & baseline) / max(1, len(final | baseline)), 6)
    return final, meta


def drm_acc_swap_sliminfer_active_ids(
    *,
    original_block_ids: Sequence[int],
    block_scores: torch.Tensor,
    top_k_indices: torch.Tensor,
    must_keep_block_ids: Set[int],
    num_blocks_to_keep: int,
    block_reps: Optional[torch.Tensor] = None,
    alpha: float = 0.25,
    min_cluster_len: int = 2,
    max_neighbors: int = 2,
    replacement_cap: float = 0.05,
    acc_cfg: Optional[Mapping[str, float]] = None,
) -> Tuple[Set[int], Dict[str, Any]]:
    acc_set, acc_meta = acc_sliminfer_active_ids(
        original_block_ids=original_block_ids,
        block_scores=block_scores,
        top_k_indices=top_k_indices,
        must_keep_block_ids=must_keep_block_ids,
        num_blocks_to_keep=num_blocks_to_keep,
        alpha=alpha,
        min_cluster_len=min_cluster_len,
        max_neighbors=max_neighbors,
        replacement_cap=replacement_cap,
        acc_cfg=acc_cfg,
    )
    state = cc_prepare_state(
        original_block_ids=original_block_ids,
        block_scores=block_scores,
        top_k_indices=top_k_indices,
        must_keep_block_ids=must_keep_block_ids,
        num_blocks_to_keep=num_blocks_to_keep,
        alpha=alpha,
        min_cluster_len=min_cluster_len,
    )
    final, drm_meta = drm_budget_neutral_swap_from_active(
        active=acc_set,
        original_block_ids=original_block_ids,
        block_scores=block_scores,
        must_keep_block_ids=must_keep_block_ids,
        num_blocks_to_keep=num_blocks_to_keep,
        block_reps=block_reps,
        clusters=state.get("clusters") or [],
        smoothed=state.get("smoothed"),
        scores_map=state.get("scores"),
        replacement_cap=replacement_cap,
        acc_cfg=acc_cfg,
        mode="drm_acc_swap_sliminfer",
    )
    meta = dict(drm_meta)
    meta["mode"] = "drm_acc_swap_sliminfer"
    meta["acc"] = {
        k: acc_meta.get(k)
        for k in (
            "spike_condition_triggered",
            "spike_protection_triggered",
            "spike_protection_changed_selection",
            "multi_locus_triggered",
            "cluster_budget_cap_triggered",
            "high_confidence_spike_replacement_count",
            "second_locus_replacement_count",
            "churn_rate",
            "churn_vs_cc",
            "jaccard_vs_cc",
            "fallback_policy",
            "cluster_topology",
            "features",
            "spike_thresholds",
            "acc_drm_version",
        )
    }
    meta["churn_vs_acc"] = round(1.0 - len(final & acc_set) / max(1, int(num_blocks_to_keep)), 6)
    meta["jaccard_vs_acc"] = round(len(final & acc_set) / max(1, len(final | acc_set)), 6)
    # Forward ACC trigger bits to top-level for easy audit aggregation.
    meta["spike_condition_triggered"] = acc_meta.get("spike_condition_triggered", 0)
    meta["spike_protection_triggered"] = acc_meta.get("spike_protection_triggered", 0)
    meta["spike_protection_changed_selection"] = acc_meta.get("spike_protection_changed_selection", 0)
    meta["multi_locus_triggered"] = acc_meta.get("multi_locus_triggered", 0)
    meta["cluster_budget_cap_triggered"] = acc_meta.get("cluster_budget_cap_triggered", 0)
    # Combine ACC neighbor + DRM replacement safety counters.
    meta["high_confidence_spike_replacement_count"] = int(
        (acc_meta.get("high_confidence_spike_replacement_count") or 0)
        + (drm_meta.get("high_confidence_spike_replacement_count") or 0)
    )
    meta["second_locus_replacement_count"] = int(
        (acc_meta.get("second_locus_replacement_count") or 0)
        + (drm_meta.get("second_locus_replacement_count") or 0)
    )
    return final, meta


def drm_acc_add_sliminfer_active_ids(
    *,
    original_block_ids: Sequence[int],
    block_scores: torch.Tensor,
    top_k_indices: torch.Tensor,
    must_keep_block_ids: Set[int],
    num_blocks_to_keep: int,
    block_reps: Optional[torch.Tensor] = None,
    alpha: float = 0.25,
    min_cluster_len: int = 2,
    max_neighbors: int = 2,
    replacement_cap: float = 0.05,
    acc_cfg: Optional[Mapping[str, float]] = None,
) -> Tuple[Set[int], Dict[str, Any]]:
    """K+M additive DRM on ACC. Tradeoff sideline only — not budget-fair vs K methods."""
    cfg = _cfg(acc_cfg)
    acc_set, acc_meta = acc_sliminfer_active_ids(
        original_block_ids=original_block_ids,
        block_scores=block_scores,
        top_k_indices=top_k_indices,
        must_keep_block_ids=must_keep_block_ids,
        num_blocks_to_keep=num_blocks_to_keep,
        alpha=alpha,
        min_cluster_len=min_cluster_len,
        max_neighbors=max_neighbors,
        replacement_cap=replacement_cap,
        acc_cfg=acc_cfg,
    )
    k = int(num_blocks_to_keep)
    must_keep = set(must_keep_block_ids) | _must_keep(original_block_ids)
    m = max(0, int(math.floor(float(cfg.get("rescue_ratio", replacement_cap)) * max(0, k - len(must_keep)))))
    inactive = [int(b) for b in original_block_ids if int(b) not in acc_set]
    base_global = _local_to_global_score_tensor(original_block_ids, block_scores)
    s0, _ = drm_s0_score(
        base_global,
        block_reps,
        sorted(acc_set),
        inactive,
        alpha_diff=float(cfg.get("alpha_diff", 0.5)),
        lambda_red=float(cfg.get("lambda_red", 0.25)),
    )
    if block_reps is not None and m > 0:
        rescued, _ = mmr_select_rescue_ids(
            inactive_ids=inactive,
            s0_scores=s0,
            block_reps=block_reps,
            active_ids=sorted(acc_set),
            m=m,
            eta_rescue=float(cfg.get("eta_rescue", 0.25)),
            eta_active=float(cfg.get("eta_active", 0.25)),
        )
        add_ids = [int(x) for x in rescued]
    else:
        scored = [(float(s0[int(b)].item()) if int(b) < s0.numel() else float("-inf"), int(b)) for b in inactive]
        scored = [x for x in scored if math.isfinite(x[0])]
        scored.sort(reverse=True)
        add_ids = [b for _, b in scored[:m]]
    final = set(acc_set) | set(add_ids)
    meta = {
        "mode": "drm_acc_add_sliminfer",
        "budget_mode": "K+M",
        "num_keep_target_K": k,
        "num_selected": len(final),
        "added_m": len(add_ids),
        "rescued_ids": add_ids,
        "over_budget": int(max(0, len(final) - k)),
        "budget_parity": int(len(final) == k + len(add_ids)),
        "acc": {
            "spike_protection_triggered": acc_meta.get("spike_protection_triggered"),
            "multi_locus_triggered": acc_meta.get("multi_locus_triggered"),
            "cluster_budget_cap_triggered": acc_meta.get("cluster_budget_cap_triggered"),
        },
        "claim_guard": "K+M tradeoff only; not fair vs K-budget methods",
    }
    return final, meta


ACC_DRM_METHODS = {
    "acc_sliminfer",
    "acc_sliminfer_v1",
    "drm_swap_sliminfer",
    "drm_acc_swap_sliminfer",
    "drm_acc_swap_sliminfer_v1",
    "drm_acc_add_sliminfer",
}

# Map CLI aliases → canonical operators (v1 is default ACC/DRM path).
ACC_DRM_METHOD_ALIASES = {
    "acc_sliminfer": "acc_sliminfer",
    "acc_sliminfer_v1": "acc_sliminfer",
    "drm_swap_sliminfer": "drm_swap_sliminfer",
    "drm_acc_swap_sliminfer": "drm_acc_swap_sliminfer",
    "drm_acc_swap_sliminfer_v1": "drm_acc_swap_sliminfer",
    "drm_acc_add_sliminfer": "drm_acc_add_sliminfer",
}


def dispatch_acc_drm_policy(
    variant_method: str,
    *,
    original_block_ids: Sequence[int],
    block_scores: torch.Tensor,
    top_k_indices: torch.Tensor,
    must_keep_block_ids: Set[int],
    num_blocks_to_keep: int,
    block_reps: Optional[torch.Tensor] = None,
    alpha: float = 0.25,
    min_cluster_len: int = 2,
    max_neighbors: int = 2,
    replacement_cap: float = 0.05,
    acc_cfg: Optional[Mapping[str, float]] = None,
) -> Tuple[Set[int], Dict[str, Any]]:
    method = ACC_DRM_METHOD_ALIASES.get(str(variant_method), str(variant_method))
    common = dict(
        original_block_ids=original_block_ids,
        block_scores=block_scores,
        top_k_indices=top_k_indices,
        must_keep_block_ids=must_keep_block_ids,
        num_blocks_to_keep=num_blocks_to_keep,
        replacement_cap=replacement_cap,
        acc_cfg=acc_cfg,
    )
    if method == "acc_sliminfer":
        return acc_sliminfer_active_ids(
            alpha=alpha,
            min_cluster_len=min_cluster_len,
            max_neighbors=max_neighbors,
            **common,
        )
    if method == "drm_swap_sliminfer":
        return drm_swap_sliminfer_active_ids(block_reps=block_reps, **common)
    if method == "drm_acc_swap_sliminfer":
        return drm_acc_swap_sliminfer_active_ids(
            block_reps=block_reps,
            alpha=alpha,
            min_cluster_len=min_cluster_len,
            max_neighbors=max_neighbors,
            **common,
        )
    if method == "drm_acc_add_sliminfer":
        return drm_acc_add_sliminfer_active_ids(
            block_reps=block_reps,
            alpha=alpha,
            min_cluster_len=min_cluster_len,
            max_neighbors=max_neighbors,
            **common,
        )
    raise ValueError(f"Unknown ACC/DRM method: {variant_method}")
    raise ValueError(f"Unknown ACC/DRM method: {variant_method}")
