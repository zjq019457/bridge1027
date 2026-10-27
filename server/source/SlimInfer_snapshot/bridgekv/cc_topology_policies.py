"""CC topology policies: features, gate, coverage, diversify, router.

Gold-free / task-name-free. Frozen CC operator hyperparameters stay unchanged;
only applicability / coverage / diversity / routing wrappers change.
See CC_TOPOLOGY_POLICY_DESIGN.md.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import torch

from bridgekv.confidence_gated_sliminfer import _trim_to_budget
from bridgekv.structural_sliminfer import (
    _cc_apply_neighbor_swaps,
    _cc_finalize_meta,
    _cc_neighbor_and_cluster_blocks,
    cc_prepare_state,
    cluster_continuity_active_ids,
)

# Default thresholds (offline-tunable; not derived from gold at runtime).
DEFAULT_CC_GATE_CONFIG: Dict[str, float] = {
    "tau_nei": 0.35,
    "tau_spike": 0.45,
    "tau_cont": 0.15,
    "tau_str": 0.25,
    "tau_second": 0.35,
    "tau_gap": 0.08,
    "tau_margin": 0.20,
    # Joint spike: neighbor_score_ratio must be <= this. High default keeps
    # legacy CC-gate spike behavior (iso+margin only); ACC-v1 overrides down.
    "tau_spike_nei_max": 1.0,
    "coverage_buckets": 8.0,
    "coverage_rho": 0.5,
    "diversify_top_g": 2.0,
    "entropy_high": 0.85,
    "ctx_len_ref_blocks": 512.0,
    "softmax_tau": 1.0,
}


def load_cc_gate_config(path: Optional[str] = None) -> Dict[str, float]:
    cfg = dict(DEFAULT_CC_GATE_CONFIG)
    if not path:
        return cfg
    p = Path(path)
    if not p.exists():
        return cfg
    raw = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        for k, v in raw.items():
            if k == "notes":
                continue
            try:
                cfg[str(k)] = float(v)
            except (TypeError, ValueError):
                continue
    return cfg


def _gate_cfg(gate_cfg: Optional[Mapping[str, float]] = None) -> Dict[str, float]:
    return {**DEFAULT_CC_GATE_CONFIG, **(dict(gate_cfg) if gate_cfg else {})}


def is_spike_topology(
    features: Mapping[str, float],
    gate_cfg: Optional[Mapping[str, float]] = None,
) -> bool:
    """Unlabeled joint spike pattern — neighbor expansion hurts.

    Requires ALL of:
      1) isolated_spike_ratio >= tau_spike
      2) topk_margin >= tau_margin
      3) neighbor_score_ratio <= tau_spike_nei_max  (cold neighbors)
    Never uses task name / gold / prediction / QA / scorer.
    """
    cfg = _gate_cfg(gate_cfg)
    return (
        float(features.get("isolated_spike_ratio", 0.0)) >= float(cfg["tau_spike"])
        and float(features.get("topk_margin", 0.0)) >= float(cfg["tau_margin"])
        and float(features.get("neighbor_score_ratio", 0.0)) <= float(cfg.get("tau_spike_nei_max", 1.0))
    )


def is_multi_locus_topology(
    features: Mapping[str, float],
    gate_cfg: Optional[Mapping[str, float]] = None,
) -> bool:
    """Two strong distant clusters — prefer diversify over vanilla CC."""
    cfg = _gate_cfg(gate_cfg)
    return (
        float(features.get("n_clusters", 0.0)) >= 2.0
        and float(features.get("second_cluster_mass_ratio", 0.0)) >= float(cfg["tau_second"])
        and float(features.get("cluster_gap_ratio", 0.0)) >= float(cfg["tau_gap"])
    )


def has_cluster_signal(
    features: Mapping[str, float],
    gate_cfg: Optional[Mapping[str, float]] = None,
) -> bool:
    """Any continuity signal beyond a bare cluster run (OR, not AND)."""
    cfg = _gate_cfg(gate_cfg)
    if float(features.get("cluster_length", 0.0)) < 2.0:
        return False
    nei_ok = float(features.get("neighbor_score_ratio", 0.0)) >= float(cfg["tau_nei"])
    cont_ok = float(features.get("topk_continuity_score", 0.0)) >= float(cfg["tau_cont"])
    str_ok = float(features.get("cluster_strength", 0.0)) >= float(cfg["tau_str"])
    return bool(nei_ok or cont_ok or str_ok)


def _bucket_index(i: int, n: int, n_buckets: int) -> int:
    if n <= 0 or n_buckets <= 0:
        return 0
    return min(n_buckets - 1, int(i * n_buckets / n))


def _cluster_masses(
    clusters: Sequence[Sequence[int]],
    smoothed: Sequence[float],
) -> List[Tuple[int, float, int, List[int]]]:
    """Return list of (rank_idx, mass, length, indices) sorted by mass desc."""
    rows: List[Tuple[float, int, List[int]]] = []
    for cl in clusters:
        mass = float(sum(max(float(smoothed[i]), 0.0) for i in cl))
        rows.append((mass, len(cl), list(cl)))
    rows.sort(key=lambda x: x[0], reverse=True)
    out: List[Tuple[int, float, int, List[int]]] = []
    for rank, (mass, length, indices) in enumerate(rows):
        out.append((rank, mass, length, indices))
    return out


def score_topology_features(
    *,
    original_block_ids: Sequence[int],
    idx_scores: Sequence[float],
    smoothed: Sequence[float],
    clusters: Sequence[Sequence[int]],
    baseline: Set[int],
    k: int,
    max_neighbors: int = 2,
    n_buckets: int = 8,
    ctx_len_ref_blocks: float = 512.0,
    softmax_tau: float = 1.0,
    query_len_tokens: Optional[int] = None,
) -> Dict[str, float]:
    """Gold-free block-score topology features (F1–F15)."""
    n = len(original_block_ids)
    bid_to_i = {int(original_block_ids[i]): i for i in range(n)}
    finite_s = [float(s) for s in idx_scores if math.isfinite(s)]
    finite_sm = [float(s) for s in smoothed if math.isfinite(s)]
    empty = {
        "cluster_strength": 0.0,
        "cluster_length": 0.0,
        "n_clusters": 0.0,
        "neighbor_score_ratio": 0.0,
        "isolated_spike_ratio": 0.0,
        "topk_continuity_score": 0.0,
        "active_set_locality": 0.0,
        "score_entropy": 0.0,
        "topk_margin": 0.0,
        "active_set_redundancy": 0.0,
        "position_coverage": 0.0,
        "ctx_len_norm": float(n) / max(float(ctx_len_ref_blocks), 1.0),
        "query_len_norm": float(query_len_tokens or 4) / 4.0,
        "cluster_gap_ratio": 0.0,
        "second_cluster_mass_ratio": 0.0,
    }
    if n == 0 or not finite_s:
        return empty

    masses = _cluster_masses(clusters, smoothed)
    n_clusters = float(len(masses))
    total_pos_mass = sum(max(float(x), 0.0) for x in finite_sm) or 1e-8
    cluster_strength = float(masses[0][1] / total_pos_mass) if masses else 0.0
    cluster_length = float(masses[0][2]) if masses else 0.0
    second_ratio = float(masses[1][1] / max(masses[0][1], 1e-8)) if len(masses) >= 2 else 0.0

    neighbor_score_ratio = 0.0
    if masses:
        cl = masses[0][3]
        center_i = max(cl, key=lambda i: smoothed[i])
        center_s = float(idx_scores[center_i]) if math.isfinite(idx_scores[center_i]) else 0.0
        neigh_vals: List[float] = []
        for delta in range(1, max_neighbors + 1):
            for ni in (center_i - delta, center_i + delta):
                if 0 <= ni < n and math.isfinite(idx_scores[ni]):
                    neigh_vals.append(float(idx_scores[ni]))
        if neigh_vals and abs(center_s) > 1e-8:
            neighbor_score_ratio = float(sum(neigh_vals) / len(neigh_vals) / abs(center_s))
        elif neigh_vals:
            neighbor_score_ratio = 0.0

    # Isolated spike ratio among baseline members.
    q50 = float(sorted(finite_s)[len(finite_s) // 2])
    cluster_idx: Set[int] = {i for cl in clusters for i in cl}
    spike_flags = 0
    base_count = 0
    for bid in baseline:
        i = bid_to_i.get(int(bid))
        if i is None:
            continue
        base_count += 1
        if i in cluster_idx:
            continue
        left_low = i == 0 or not math.isfinite(idx_scores[i - 1]) or float(idx_scores[i - 1]) < q50
        right_low = i == n - 1 or not math.isfinite(idx_scores[i + 1]) or float(idx_scores[i + 1]) < q50
        if left_low and right_low:
            spike_flags += 1
    isolated_spike_ratio = float(spike_flags / max(1, base_count))

    # Top-k continuity on baseline positions.
    base_pos = sorted(bid_to_i[int(b)] for b in baseline if int(b) in bid_to_i)
    if len(base_pos) >= 2:
        adj = sum(1 for a, b in zip(base_pos, base_pos[1:]) if b - a == 1)
        topk_continuity_score = float(adj / max(1, len(base_pos) - 1))
    else:
        topk_continuity_score = 0.0

    # Position coverage / locality over baseline.
    B = max(1, int(n_buckets))
    bucket_counts = [0] * B
    for bid in baseline:
        i = bid_to_i.get(int(bid))
        if i is None:
            continue
        bucket_counts[_bucket_index(i, n, B)] += 1
    nonempty = sum(1 for c in bucket_counts if c > 0)
    position_coverage = float(nonempty / B)
    # Normalized entropy of bucket occupancy (then locality = 1 - H_norm).
    total_b = sum(bucket_counts) or 1
    probs = [c / total_b for c in bucket_counts if c > 0]
    if probs:
        h = -sum(p * math.log(p + 1e-12) for p in probs)
        h_norm = h / math.log(max(2, len(probs)))
    else:
        h_norm = 0.0
    active_set_locality = float(1.0 - h_norm)

    # Score entropy over finite blocks.
    tau = max(float(softmax_tau), 1e-6)
    # numerical stability
    max_s = max(finite_s)
    exps = [math.exp((s - max_s) / tau) for s in finite_s]
    z = sum(exps) or 1.0
    ps = [e / z for e in exps]
    h_s = -sum(p * math.log(p + 1e-12) for p in ps)
    score_entropy = float(h_s / math.log(max(2, len(ps))))

    sorted_s = sorted(finite_s, reverse=True)
    s1 = sorted_s[0]
    sk = sorted_s[min(len(sorted_s) - 1, max(0, int(k) - 1))]
    topk_margin = float((s1 - sk) / (abs(s1) + 1e-8))

    # Active-set redundancy: adjacent baseline pairs.
    red_vals: List[float] = []
    base_set_pos = set(base_pos)
    for i in base_pos:
        if (i + 1) in base_set_pos and math.isfinite(smoothed[i]) and math.isfinite(smoothed[i + 1]):
            diff = abs(float(smoothed[i]) - float(smoothed[i + 1]))
            red_vals.append(1.0 / max(diff, 1e-3))
    active_set_redundancy = float(sum(red_vals) / len(red_vals)) if red_vals else 0.0

    cluster_gap_ratio = 0.0
    if len(masses) >= 2:
        ends = []
        for _, _, _, indices in masses[:2]:
            ends.append((min(indices), max(indices)))
        (a0, a1), (b0, b1) = ends[0], ends[1]
        if a1 < b0:
            gap = b0 - a1
        elif b1 < a0:
            gap = a0 - b1
        else:
            gap = 0
        cluster_gap_ratio = float(gap / max(1, n))

    return {
        "cluster_strength": round(cluster_strength, 6),
        "cluster_length": round(cluster_length, 6),
        "n_clusters": round(n_clusters, 6),
        "neighbor_score_ratio": round(neighbor_score_ratio, 6),
        "isolated_spike_ratio": round(isolated_spike_ratio, 6),
        "topk_continuity_score": round(topk_continuity_score, 6),
        "active_set_locality": round(active_set_locality, 6),
        "score_entropy": round(score_entropy, 6),
        "topk_margin": round(topk_margin, 6),
        "active_set_redundancy": round(active_set_redundancy, 6),
        "position_coverage": round(position_coverage, 6),
        "ctx_len_norm": round(float(n) / max(float(ctx_len_ref_blocks), 1.0), 6),
        "query_len_norm": round(float(query_len_tokens or 4) / 4.0, 6),
        "cluster_gap_ratio": round(cluster_gap_ratio, 6),
        "second_cluster_mass_ratio": round(second_ratio, 6),
    }


def classify_spike_vs_cluster(
    features: Mapping[str, float],
    gate_cfg: Optional[Mapping[str, float]] = None,
) -> str:
    cfg = _gate_cfg(gate_cfg)
    if is_spike_topology(features, cfg):
        return "spike"
    if has_cluster_signal(features, cfg):
        return "cluster"
    if float(features.get("cluster_length", 0.0)) >= 2.0:
        return "ambiguous"
    return "ambiguous"


def cc_applicable(
    features: Mapping[str, float],
    gate_cfg: Optional[Mapping[str, float]] = None,
) -> Tuple[bool, str]:
    """Return (applicable, reason). Relaxed OR-signal gate; multi_locus → not vanilla CC."""
    cfg = _gate_cfg(gate_cfg)
    if float(features.get("cluster_length", 0.0)) < 2.0:
        return False, "no_cluster"
    if is_spike_topology(features, cfg):
        return False, "spike_pattern"
    if float(features.get("isolated_spike_ratio", 0.0)) > float(cfg["tau_spike"]):
        return False, "spike_dominated"
    if is_multi_locus_topology(features, cfg):
        return False, "multi_locus"
    if not has_cluster_signal(features, cfg):
        return False, "weak_continuity"
    return True, "applicable"


def route_cc_policy(
    features: Mapping[str, float],
    gate_cfg: Optional[Mapping[str, float]] = None,
) -> str:
    """Category-free router → original | cc | coverage_cc | diversify_cc."""
    cfg = _gate_cfg(gate_cfg)
    if is_spike_topology(features, cfg):
        return "original"
    if is_multi_locus_topology(features, cfg):
        return "diversify_cc"
    applicable, _ = cc_applicable(features, cfg)
    if applicable:
        return "cc"
    # Coverage only for diffuse long-context (summ/code tail); not default QA path.
    if (
        float(features.get("score_entropy", 0.0)) >= float(cfg["entropy_high"])
        and float(features.get("ctx_len_norm", 0.0)) >= 1.0
        and float(features.get("position_coverage", 1.0)) >= 0.75
    ):
        return "coverage_cc"
    return "original"


def _features_from_state(
    state: Mapping[str, Any],
    *,
    max_neighbors: int,
    gate_cfg: Mapping[str, float],
    query_len_tokens: Optional[int] = None,
) -> Dict[str, float]:
    return score_topology_features(
        original_block_ids=state["original_block_ids"],
        idx_scores=state["idx_scores"],
        smoothed=state["smoothed"],
        clusters=state["clusters"],
        baseline=set(state["baseline"]),
        k=int(state["k"]),
        max_neighbors=max_neighbors,
        n_buckets=int(gate_cfg.get("coverage_buckets", 8)),
        ctx_len_ref_blocks=float(gate_cfg.get("ctx_len_ref_blocks", 512.0)),
        softmax_tau=float(gate_cfg.get("softmax_tau", 1.0)),
        query_len_tokens=query_len_tokens,
    )


def cc_gated_active_ids(
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
    gate_cfg: Optional[Mapping[str, float]] = None,
    query_len_tokens: Optional[int] = None,
) -> Tuple[Set[int], Dict[str, Any]]:
    """Applicability / spike-hybrid gate around frozen CC."""
    cfg = {**DEFAULT_CC_GATE_CONFIG, **(dict(gate_cfg) if gate_cfg else {})}
    state = cc_prepare_state(
        original_block_ids=original_block_ids,
        block_scores=block_scores,
        top_k_indices=top_k_indices,
        must_keep_block_ids=must_keep_block_ids,
        num_blocks_to_keep=num_blocks_to_keep,
        alpha=alpha,
        min_cluster_len=min_cluster_len,
    )
    baseline: Set[int] = set(state["baseline"])
    features = _features_from_state(
        state, max_neighbors=max_neighbors, gate_cfg=cfg, query_len_tokens=query_len_tokens
    )
    pattern = classify_spike_vs_cluster(features, cfg)
    applicable, reason = cc_applicable(features, cfg)
    enable = bool(applicable)
    if not enable:
        meta = {
            "mode": "cc_gated_v0",
            "policy_chosen": "original",
            "pattern": pattern,
            "applicable": False,
            "gate_reason": reason,
            "triggered": 0,
            "features": features,
            "budget_parity": int(len(baseline) == int(state["k"])),
            "num_selected": len(baseline),
            "num_keep_target": int(state["k"]),
        }
        return baseline, meta

    final, cc_meta = cluster_continuity_active_ids(
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
    meta = dict(cc_meta)
    meta.update(
        {
            "mode": "cc_gated_v0",
            "policy_chosen": "cc",
            "pattern": pattern,
            "applicable": True,
            "gate_reason": reason,
            "features": features,
            "cc_operator_mode": cc_meta.get("mode"),
        }
    )
    return final, meta


def coverage_cc_active_ids(
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
    gate_cfg: Optional[Mapping[str, float]] = None,
    query_len_tokens: Optional[int] = None,
) -> Tuple[Set[int], Dict[str, Any]]:
    """Coverage-preserved CC: position-bucket floors + CC only on free budget."""
    cfg = {**DEFAULT_CC_GATE_CONFIG, **(dict(gate_cfg) if gate_cfg else {})}
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
    n = int(state["n"])
    must_keep: Set[int] = set(state["must_keep"])
    baseline: Set[int] = set(state["baseline"])
    scores: Dict[int, float] = state["scores"]
    features = _features_from_state(
        state, max_neighbors=max_neighbors, gate_cfg=cfg, query_len_tokens=query_len_tokens
    )
    if not state["ok"] or n == 0:
        return baseline, {
            "mode": "cc_coverage_v0",
            "policy_chosen": "original",
            "triggered": 0,
            "features": features,
            "reason": "no_finite_scores",
        }

    B = max(1, int(cfg.get("coverage_buckets", 8)))
    rho = float(cfg.get("coverage_rho", 0.5))
    per_bucket = max(1, int(math.floor(rho * k / B)))
    # Per-bucket candidates.
    buckets: List[List[int]] = [[] for _ in range(B)]
    for i, bid in enumerate(original_block_ids):
        if math.isfinite(state["idx_scores"][i]):
            buckets[_bucket_index(i, n, B)].append(int(bid))
    coverage_lock: Set[int] = set(must_keep)
    for b_ids in buckets:
        if not b_ids:
            continue
        ranked = sorted(b_ids, key=lambda b: scores.get(b, float("-inf")), reverse=True)
        for bid in ranked[:per_bucket]:
            coverage_lock.add(bid)
            if len(coverage_lock) >= k:
                break
        if len(coverage_lock) >= k:
            break

    # If coverage lock exceeds budget, trim lowest non-must_keep.
    if len(coverage_lock) > k:
        coverage_lock = _trim_to_budget(
            coverage_lock,
            original_block_ids=original_block_ids,
            block_scores=block_scores,
            must_keep_block_ids=must_keep,
            num_blocks_to_keep=k,
        )

    # Seed free slots with SlimInfer ranking, then CC swaps that cannot break coverage_lock.
    final = set(coverage_lock)
    ranked_all = sorted(
        [int(b) for b in original_block_ids if int(b) not in final],
        key=lambda b: scores.get(b, float("-inf")),
        reverse=True,
    )
    for bid in ranked_all:
        if len(final) >= k:
            break
        final.add(bid)

    smoothed = state["smoothed"]
    clusters = state["clusters"]
    to_add, cluster_blocks = _cc_neighbor_and_cluster_blocks(
        original_block_ids=original_block_ids,
        clusters=clusters,
        smoothed=smoothed,
        baseline=final,
        max_neighbors=max_neighbors,
    )
    final, swaps = _cc_apply_neighbor_swaps(
        baseline=final,
        must_keep=must_keep,
        cluster_blocks=cluster_blocks,
        to_add=to_add,
        scores=scores,
        k=k,
        replacement_cap=replacement_cap,
        protected_blocks=coverage_lock,
    )
    # Restore any coverage floor accidentally lost, then trim.
    final |= set(coverage_lock) & set(int(x) for x in original_block_ids)
    final = _trim_to_budget(
        final,
        original_block_ids=original_block_ids,
        block_scores=block_scores,
        must_keep_block_ids=must_keep | (coverage_lock if len(must_keep | coverage_lock) <= k else must_keep),
        num_blocks_to_keep=k,
    )
    # Ensure coverage floors as much as possible without violating budget.
    for b_ids in buckets:
        if not b_ids:
            continue
        if any(bid in final for bid in b_ids):
            continue
        # force best of bucket if we can swap out non-locked
        best = max(b_ids, key=lambda b: scores.get(b, float("-inf")))
        removable = [
            b
            for b in final
            if b not in must_keep and b not in coverage_lock
        ]
        if not removable:
            continue
        worst = min(removable, key=lambda b: scores.get(b, float("-inf")))
        final.discard(worst)
        final.add(best)

    final = _trim_to_budget(
        final,
        original_block_ids=original_block_ids,
        block_scores=block_scores,
        must_keep_block_ids=must_keep,
        num_blocks_to_keep=k,
    )
    max_swap = max(0, int(math.floor(replacement_cap * max(0, k - len(must_keep)))))
    meta = _cc_finalize_meta(
        mode="cc_coverage_v0",
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
        extra={
            "policy_chosen": "coverage_cc",
            "features": features,
            "coverage_buckets": B,
            "coverage_rho": rho,
            "coverage_lock_size": len(coverage_lock),
            "per_bucket_floor": per_bucket,
        },
    )
    return final, meta


def diversify_cc_active_ids(
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
    gate_cfg: Optional[Mapping[str, float]] = None,
    query_len_tokens: Optional[int] = None,
) -> Tuple[Set[int], Dict[str, Any]]:
    """Multi-cluster quotas: keep top-g clusters, CC neighbors cannot steal other clusters."""
    cfg = {**DEFAULT_CC_GATE_CONFIG, **(dict(gate_cfg) if gate_cfg else {})}
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
    must_keep: Set[int] = set(state["must_keep"])
    baseline: Set[int] = set(state["baseline"])
    scores: Dict[int, float] = state["scores"]
    features = _features_from_state(
        state, max_neighbors=max_neighbors, gate_cfg=cfg, query_len_tokens=query_len_tokens
    )
    clusters = state["clusters"]
    smoothed = state["smoothed"]
    if not state["ok"] or not clusters:
        return baseline, {
            "mode": "cc_diversify_v0",
            "policy_chosen": "original",
            "triggered": 0,
            "features": features,
            "reason": "no_clusters",
        }

    g = max(1, int(cfg.get("diversify_top_g", 2)))
    masses = _cluster_masses(clusters, smoothed)[:g]
    total_mass = sum(m for _, m, _, _ in masses) or 1.0
    # Proportional quotas (at least 1 per cluster when budget allows).
    quotas: List[int] = []
    remaining = k - len(must_keep)
    for idx, (_, mass, _, _) in enumerate(masses):
        if idx == len(masses) - 1:
            quotas.append(max(1, remaining - sum(quotas)))
        else:
            q = max(1, int(round(remaining * (mass / total_mass))))
            quotas.append(q)
    # Fix over-allocation.
    while sum(quotas) > max(0, remaining) and any(q > 1 for q in quotas):
        for i in range(len(quotas)):
            if quotas[i] > 1 and sum(quotas) > remaining:
                quotas[i] -= 1

    reserved: Set[int] = set(must_keep)
    cluster_protected: Set[int] = set()
    for (_, _, _, indices), q in zip(masses, quotas):
        ranked = sorted(
            [int(original_block_ids[i]) for i in indices],
            key=lambda b: scores.get(b, float("-inf")),
            reverse=True,
        )
        take = ranked[: max(1, q)]
        reserved.update(take)
        cluster_protected.update(ranked)  # all members of top-g clusters protected from eviction

    if len(reserved) > k:
        reserved = _trim_to_budget(
            reserved,
            original_block_ids=original_block_ids,
            block_scores=block_scores,
            must_keep_block_ids=must_keep,
            num_blocks_to_keep=k,
        )

    final = set(reserved)
    ranked_all = sorted(
        [int(b) for b in original_block_ids if int(b) not in final],
        key=lambda b: scores.get(b, float("-inf")),
        reverse=True,
    )
    for bid in ranked_all:
        if len(final) >= k:
            break
        final.add(bid)

    to_add, cluster_blocks = _cc_neighbor_and_cluster_blocks(
        original_block_ids=original_block_ids,
        clusters=[m[3] for m in masses],
        smoothed=smoothed,
        baseline=final,
        max_neighbors=max_neighbors,
    )
    final, swaps = _cc_apply_neighbor_swaps(
        baseline=final,
        must_keep=must_keep,
        cluster_blocks=cluster_blocks,
        to_add=to_add,
        scores=scores,
        k=k,
        replacement_cap=replacement_cap,
        protected_blocks=cluster_protected,
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
        mode="cc_diversify_v0",
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
        extra={
            "policy_chosen": "diversify_cc",
            "features": features,
            "diversify_top_g": g,
            "cluster_quotas": quotas,
            "reserved_size": len(reserved),
            "protected_cluster_blocks": len(cluster_protected),
        },
    )
    return final, meta


def cc_router_active_ids(
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
    gate_cfg: Optional[Mapping[str, float]] = None,
    query_len_tokens: Optional[int] = None,
) -> Tuple[Set[int], Dict[str, Any]]:
    """Route among original / cc / coverage_cc / diversify_cc via topology features."""
    cfg = {**DEFAULT_CC_GATE_CONFIG, **(dict(gate_cfg) if gate_cfg else {})}
    state = cc_prepare_state(
        original_block_ids=original_block_ids,
        block_scores=block_scores,
        top_k_indices=top_k_indices,
        must_keep_block_ids=must_keep_block_ids,
        num_blocks_to_keep=num_blocks_to_keep,
        alpha=alpha,
        min_cluster_len=min_cluster_len,
    )
    features = _features_from_state(
        state, max_neighbors=max_neighbors, gate_cfg=cfg, query_len_tokens=query_len_tokens
    )
    policy = route_cc_policy(features, cfg)
    common = dict(
        original_block_ids=original_block_ids,
        block_scores=block_scores,
        top_k_indices=top_k_indices,
        must_keep_block_ids=must_keep_block_ids,
        num_blocks_to_keep=num_blocks_to_keep,
        alpha=alpha,
        min_cluster_len=min_cluster_len,
        max_neighbors=max_neighbors,
        replacement_cap=replacement_cap,
        gate_cfg=cfg,
        query_len_tokens=query_len_tokens,
    )
    if policy == "cc":
        final, meta = cluster_continuity_active_ids(
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
    elif policy == "coverage_cc":
        final, meta = coverage_cc_active_ids(**common)
    elif policy == "diversify_cc":
        final, meta = diversify_cc_active_ids(**common)
    else:
        final = set(state["baseline"])
        meta = {
            "mode": "cc_router_v0",
            "triggered": 0,
            "budget_parity": int(len(final) == int(state["k"])),
            "num_selected": len(final),
            "num_keep_target": int(state["k"]),
        }
    out = dict(meta)
    out["mode"] = "cc_router_v0"
    out["policy_chosen"] = policy
    out["features"] = features
    out["pattern"] = classify_spike_vs_cluster(features, cfg)
    return final, out


CC_TOPOLOGY_METHODS = {
    "cc_gated_v0",
    "cc_spike_hybrid_v0",  # alias of gated
    "cc_coverage_v0",
    "cc_diversify_v0",
    "cc_router_v0",
}


def dispatch_cc_topology_policy(
    variant_method: str,
    **kwargs: Any,
) -> Tuple[Set[int], Dict[str, Any]]:
    method = str(variant_method)
    if method in ("cc_gated_v0", "cc_spike_hybrid_v0"):
        return cc_gated_active_ids(**kwargs)
    if method == "cc_coverage_v0":
        return coverage_cc_active_ids(**kwargs)
    if method == "cc_diversify_v0":
        return diversify_cc_active_ids(**kwargs)
    if method == "cc_router_v0":
        return cc_router_active_ids(**kwargs)
    raise ValueError(f"Unknown CC topology method: {variant_method}")
