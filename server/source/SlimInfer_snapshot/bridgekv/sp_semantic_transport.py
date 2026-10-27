"""SP-SlimInfer v2: cross-layer semantic transport selection (offline replay + future runtime).

Default off. Used only when ``variant_method == 'sp_sliminfer_v2'`` or explicit replay call.
No gold, no QA-based hyperparameter selection.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

import numpy as np
import torch

from bridgekv.confidence_gated_sliminfer import _trim_to_budget
from bridgekv.sp_sliminfer import active_set_churn
from bridgekv.sp_trajectory import (
    TRAJECTORY_PRUNE_LAYERS,
    UNAVAILABLE,
    _cosine_np,
    block_token_span,
    pool_hidden_block_rep,
    pool_hidden_query_rep,
    pool_key_block_rep,
    pool_local_query_rep,
)

POLICIES = ("transport_only", "transport_residual", "transport_full")

DEFAULT_TAU = 0.10
DEFAULT_TOP_R = 5
DEFAULT_ALPHA_MASS = 0.5
DEFAULT_REPLACEMENT_CAP = 0.05
DEFAULT_NEAR_TOP_M = 8
DEFAULT_INCOMING_POOL_TOP = 12

POLICY_WEIGHTS: Dict[str, Dict[str, float]] = {
    "transport_only": {"silm": 0.60, "incoming": 0.40},
    "transport_residual": {
        "silm": 0.45,
        "incoming": 0.30,
        "uniqueness": 0.15,
        "absorption": 0.05,
        "redundancy": 0.05,
    },
    "transport_full": {
        "silm": 0.40,
        "incoming": 0.25,
        "uniqueness": 0.12,
        "absorption": 0.04,
        "redundancy": 0.04,
        "persistence": 0.08,
        "relay": 0.05,
        "conditional_drift": 0.02,
    },
}


@dataclass
class TransportGraph:
    prev_layer: int
    cur_layer: int
    tau: float
    top_r: int
    num_prev_available: int = 0
    num_cur_available: int = 0
    num_edges: int = 0
    edge_sparsity: float = 0.0
    incoming_mass: Dict[int, float] = field(default_factory=dict)
    persistence_mass: Dict[int, float] = field(default_factory=dict)
    absorption_mass: Dict[int, float] = field(default_factory=dict)
    relay_mass: Dict[int, float] = field(default_factory=dict)
    redundancy_max: Dict[int, float] = field(default_factory=dict)
    conditional_drift: Dict[int, float] = field(default_factory=dict)
    uniqueness: Dict[int, float] = field(default_factory=dict)
    feature_available: Dict[int, bool] = field(default_factory=dict)
    transport_edges: List[Tuple[int, int, float, float]] = field(default_factory=list)


def _prev_prune_layer(layer_idx: int) -> Optional[int]:
    prior = [l for l in TRAJECTORY_PRUNE_LAYERS if l < layer_idx]
    return prior[-1] if prior else None


def _safe_float(val: Any) -> Optional[float]:
    if val is None or val == "" or val == UNAVAILABLE:
        return None
    try:
        f = float(val)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def percentile_rank(values: Mapping[int, float]) -> Dict[int, float]:
    if not values:
        return {}
    items = sorted(values.items(), key=lambda kv: kv[1])
    n = len(items)
    if n == 1:
        return {items[0][0]: 0.5}
    return {int(k): i / (n - 1) for i, (k, _) in enumerate(items)}


def _layer_block_maps(
    history: Mapping[int, Mapping[str, Any]],
    layer_idx: int,
    scalar_by_block: Mapping[int, Mapping[str, Any]],
) -> Tuple[List[int], Dict[int, int], Dict[int, int]]:
    """Return alive block ids, bid->hidden_row, bid unavailable flags."""
    layer = history[layer_idx]
    alive: List[int] = []
    row_map: Dict[int, int] = {}
    for i, bid in enumerate(layer["block_ids"]):
        bid = int(bid)
        if layer["censored"][i]:
            continue
        if not layer["rep_available"][i]:
            continue
        alive.append(bid)
        row_map[bid] = i
    return alive, row_map, {}


def _scalar_score_map(scalar_rows: Sequence[Mapping[str, Any]], layer_idx: int) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for row in scalar_rows:
        if int(row.get("layer_idx", -1)) != layer_idx:
            continue
        if str(row.get("alive_before_pruning", "0")) not in {"1", "True", "true"}:
            continue
        bid = int(row["block_id"])
        sc = _safe_float(row.get("original_sliminfer_score"))
        if sc is not None:
            out[bid] = sc
    return out


def _scalar_query_rel_map(scalar_rows: Sequence[Mapping[str, Any]], layer_idx: int) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for row in scalar_rows:
        if int(row.get("layer_idx", -1)) != layer_idx:
            continue
        if str(row.get("alive_before_pruning", "0")) not in {"1", "True", "true"}:
            continue
        bid = int(row["block_id"])
        qh = _safe_float(row.get("query_hidden_cosine"))
        if qh is not None:
            out[bid] = qh
    return out


def build_transport_graph(
    *,
    history: Mapping[int, Mapping[str, Any]],
    scalar_rows: Sequence[Mapping[str, Any]],
    cur_layer: int,
    prev_layer: Optional[int] = None,
    tau: float = DEFAULT_TAU,
    top_r: int = DEFAULT_TOP_R,
    alpha_mass: float = DEFAULT_ALPHA_MASS,
    candidate_block_ids: Optional[Iterable[int]] = None,
) -> TransportGraph:
    prev_layer = prev_layer if prev_layer is not None else _prev_prune_layer(cur_layer)
    graph = TransportGraph(prev_layer=int(prev_layer or -1), cur_layer=int(cur_layer), tau=tau, top_r=top_r)
    if prev_layer is None or prev_layer not in history or cur_layer not in history:
        return graph

    prev = history[prev_layer]
    cur = history[cur_layer]
    prev_scores = _scalar_score_map(scalar_rows, prev_layer)
    prev_q = _scalar_query_rel_map(scalar_rows, prev_layer)
    cur_q = _scalar_query_rel_map(scalar_rows, cur_layer)

    prev_avail: List[Tuple[int, int]] = []
    for i, bid in enumerate(prev["block_ids"]):
        bid = int(bid)
        if prev["censored"][i] or not prev["rep_available"][i]:
            continue
        prev_avail.append((i, bid))

    cur_avail: List[Tuple[int, int]] = []
    for j, bid in enumerate(cur["block_ids"]):
        bid = int(bid)
        if cur["censored"][j] or not cur["rep_available"][j]:
            continue
        cur_avail.append((j, bid))

    graph.num_prev_available = len(prev_avail)
    graph.num_cur_available = len(cur_avail)
    if not prev_avail or not cur_avail:
        return graph

    score_rank = percentile_rank({bid: prev_scores.get(bid, 0.0) for _, bid in prev_avail})
    q_rank = percentile_rank({bid: prev_q.get(bid, 0.0) for _, bid in prev_avail if bid in prev_q})
    mass: Dict[int, float] = {}
    for _, bid in prev_avail:
        sr = score_rank.get(bid, 0.5)
        qr = q_rank.get(bid, sr)
        mass[bid] = float(alpha_mass * qr + (1.0 - alpha_mass) * sr)

    cur_bid_to_j = {bid: j for j, bid in cur_avail}
    incoming: Dict[int, float] = {bid: 0.0 for _, bid in cur_avail}
    persistence: Dict[int, float] = {}
    absorption: Dict[int, float] = {}

    max_possible_edges = max(1, len(prev_avail) * min(top_r, len(cur_avail)))
    edge_count = 0

    for i, bid_i in prev_avail:
        h_prev = prev["hidden"][i]
        cosines: List[Tuple[int, float]] = []
        for j, bid_j in cur_avail:
            cos_v = _cosine_np(h_prev, cur["hidden"][j])
            if math.isfinite(cos_v):
                cosines.append((j, float(cos_v)))
        if not cosines:
            continue
        cosines.sort(key=lambda x: x[1], reverse=True)
        top_neighbors = cosines[: max(1, min(top_r, len(cosines)))]
        logits = np.array([c / max(tau, 1e-6) for _, c in top_neighbors], dtype=np.float64)
        logits -= logits.max()
        weights = np.exp(logits)
        weights /= weights.sum()
        m_i = mass.get(bid_i, 0.0)
        off_diag = 0.0
        for (j, cos_v), w in zip(top_neighbors, weights.tolist()):
            bid_j = int(cur["block_ids"][j])
            graph.transport_edges.append((bid_i, bid_j, cos_v, float(w)))
            incoming[bid_j] = incoming.get(bid_j, 0.0) + m_i * w
            edge_count += 1
            if bid_i == bid_j:
                persistence[bid_j] = persistence.get(bid_j, 0.0) + m_i * w
            else:
                off_diag += m_i * w
        absorption[bid_i] = off_diag

    graph.incoming_mass = incoming
    graph.persistence_mass = persistence
    graph.absorption_mass = absorption
    graph.num_edges = edge_count
    graph.edge_sparsity = 1.0 - (edge_count / max_possible_edges)

    prev_q_map = prev_q
    for j, bid_j in cur_avail:
        in_m = incoming.get(bid_j, 0.0)
        q_cur = cur_q.get(bid_j)
        q_prev = prev_q_map.get(bid_j)
        if q_cur is not None and q_prev is not None:
            rel_inc = max(0.0, q_cur - q_prev)
            graph.relay_mass[bid_j] = in_m * rel_inc
        if bid_j in cur_bid_to_j and bid_j in {b for _, b in prev_avail}:
            pi = next(i for i, b in prev_avail if b == bid_j)
            cos_self = _cosine_np(prev["hidden"][pi], cur["hidden"][j])
            if math.isfinite(cos_self) and q_cur is not None and q_prev is not None:
                drift = max(0.0, 1.0 - cos_self)
                q_drop = max(0.0, q_prev - q_cur)
                if drift > 0.15 and q_drop > 0.0:
                    graph.conditional_drift[bid_j] = drift * q_drop

    pool = list(candidate_block_ids) if candidate_block_ids is not None else [bid for _, bid in cur_avail]
    vecs: Dict[int, np.ndarray] = {}
    for bid in pool:
        if bid not in cur_bid_to_j:
            continue
        j = cur_bid_to_j[bid]
        vecs[bid] = cur["hidden"][j]
        graph.feature_available[bid] = True

    for bid in pool:
        if bid not in vecs:
            graph.feature_available[bid] = False
            continue
        max_cos = 0.0
        for other, vec in vecs.items():
            if other == bid:
                continue
            cos_v = _cosine_np(vecs[bid], vec)
            if math.isfinite(cos_v):
                max_cos = max(max_cos, cos_v)
        graph.redundancy_max[bid] = max_cos
        graph.uniqueness[bid] = 1.0 - max_cos

    return graph


def _must_keep_block_ids(original_block_ids: Sequence[int]) -> Set[int]:
    ids: Set[int] = set()
    if original_block_ids:
        ids.add(int(original_block_ids[0]))
        ids.add(int(original_block_ids[-1]))
    return ids


def _silm_topk(
    original_block_ids: Sequence[int],
    score_map: Mapping[int, float],
    budget_k: int,
    must_keep: Set[int],
) -> Set[int]:
    ranked = sorted(original_block_ids, key=lambda b: score_map.get(int(b), float("-inf")), reverse=True)
    top = set(int(b) for b in ranked[: max(1, min(budget_k, len(ranked)))]) | must_keep
    scores = torch.tensor([float(score_map.get(int(b), float("-inf"))) for b in original_block_ids], dtype=torch.float32)
    return _trim_to_budget(
        top,
        original_block_ids=original_block_ids,
        block_scores=scores,
        must_keep_block_ids=must_keep,
        num_blocks_to_keep=int(budget_k),
    )


def _compute_utility(
    *,
    policy: str,
    block_ids: Iterable[int],
    silm_scores: Mapping[int, float],
    graph: TransportGraph,
    weights: Optional[Mapping[str, float]] = None,
) -> Dict[int, float]:
    w = dict(POLICY_WEIGHTS[policy])
    if weights:
        w.update({k: float(v) for k, v in weights.items()})

    pool = [int(b) for b in block_ids]
    rank_silm = percentile_rank({b: silm_scores.get(b, float("-inf")) for b in pool})
    rank_in = percentile_rank({b: graph.incoming_mass.get(b, 0.0) for b in pool if graph.feature_available.get(b, False)})
    rank_persist = percentile_rank({b: graph.persistence_mass.get(b, 0.0) for b in pool if b in graph.persistence_mass})
    rank_abs = percentile_rank({b: graph.absorption_mass.get(b, 0.0) for b in pool if b in graph.absorption_mass})
    rank_relay = percentile_rank({b: graph.relay_mass.get(b, 0.0) for b in pool if b in graph.relay_mass})
    rank_red = percentile_rank({b: graph.redundancy_max.get(b, 0.0) for b in pool if b in graph.redundancy_max})
    rank_uniq = percentile_rank({b: graph.uniqueness.get(b, 0.0) for b in pool if b in graph.uniqueness})
    rank_drift = percentile_rank({b: graph.conditional_drift.get(b, 0.0) for b in pool if b in graph.conditional_drift})

    utilities: Dict[int, float] = {}
    for b in pool:
        u = w.get("silm", 0.0) * rank_silm.get(b, 0.5)
        if graph.feature_available.get(b, False) and b in graph.incoming_mass:
            u += w.get("incoming", 0.0) * rank_in.get(b, 0.5)
        elif w.get("incoming", 0.0) > 0:
            u += w.get("incoming", 0.0) * rank_silm.get(b, 0.5)

        if policy in {"transport_residual", "transport_full"}:
            if graph.feature_available.get(b, False):
                u += w.get("uniqueness", 0.0) * rank_uniq.get(b, 0.5)
                u -= w.get("absorption", 0.0) * rank_abs.get(b, 0.5)
                u -= w.get("redundancy", 0.0) * rank_red.get(b, 0.5)
        if policy == "transport_full":
            if b in graph.persistence_mass:
                u += w.get("persistence", 0.0) * rank_persist.get(b, 0.5)
            if b in graph.relay_mass:
                u += w.get("relay", 0.0) * rank_relay.get(b, 0.5)
            if b in graph.conditional_drift:
                u -= w.get("conditional_drift", 0.0) * rank_drift.get(b, 0.5)
        utilities[b] = float(u)
    return utilities


def bounded_transport_select(
    *,
    original_block_ids: Sequence[int],
    silm_scores: Mapping[int, float],
    silm_selected: Set[int],
    budget_k: int,
    graph: TransportGraph,
    policy: str,
    prev_active: Optional[Set[int]] = None,
    replacement_cap: float = DEFAULT_REPLACEMENT_CAP,
    near_top_m: int = DEFAULT_NEAR_TOP_M,
    incoming_pool_top: int = DEFAULT_INCOMING_POOL_TOP,
    weights: Optional[Mapping[str, float]] = None,
) -> Tuple[Set[int], Dict[str, Any]]:
    must_keep = _must_keep_block_ids(original_block_ids)
    k = int(budget_k)
    baseline = set(int(x) for x in silm_selected)
    baseline = _trim_to_budget(
        baseline,
        original_block_ids=original_block_ids,
        block_scores=torch.tensor([float(silm_scores.get(int(b), float("-inf"))) for b in original_block_ids], dtype=torch.float32),
        must_keep_block_ids=must_keep,
        num_blocks_to_keep=k,
    )

    ranked_all = sorted(original_block_ids, key=lambda b: silm_scores.get(int(b), float("-inf")), reverse=True)
    near_top = set(int(b) for b in ranked_all[: min(len(ranked_all), k + near_top_m)])
    high_incoming = sorted(
        [b for b, v in graph.incoming_mass.items() if graph.feature_available.get(b, False)],
        key=lambda b: graph.incoming_mass.get(b, 0.0),
        reverse=True,
    )[:incoming_pool_top]
    pool_ids = set(near_top) | baseline | set(high_incoming)
    if prev_active:
        pool_ids |= {int(x) for x in prev_active if int(x) in silm_scores or int(x) in graph.incoming_mass}

    utilities = _compute_utility(policy=policy, block_ids=pool_ids, silm_scores=silm_scores, graph=graph, weights=weights)

    max_swap = max(0, int(math.floor(replacement_cap * max(0, k - len(must_keep)))))
    removable = sorted(
        [b for b in baseline if b not in must_keep],
        key=lambda b: utilities.get(b, silm_scores.get(b, float("-inf"))),
    )
    addable = sorted(
        [b for b in pool_ids if b not in baseline],
        key=lambda b: utilities.get(b, silm_scores.get(b, float("-inf"))),
        reverse=True,
    )

    final = set(baseline)
    swaps: List[Dict[str, Any]] = []
    ri = 0
    ai = 0
    while len(swaps) < max_swap and ri < len(removable) and ai < len(addable):
        old_b = removable[ri]
        new_b = addable[ai]
        ri += 1
        if new_b in final:
            continue
        u_old = utilities.get(old_b, silm_scores.get(old_b, float("-inf")))
        u_new = utilities.get(new_b, silm_scores.get(new_b, float("-inf")))
        if u_new <= u_old:
            ai += 1
            continue
        final.discard(old_b)
        final.add(new_b)
        swaps.append({"out": old_b, "in": new_b, "u_old": u_old, "u_new": u_new})
        ai += 1

    final = _trim_to_budget(
        final,
        original_block_ids=original_block_ids,
        block_scores=torch.tensor([float(utilities.get(int(b), silm_scores.get(int(b), float("-inf")))) for b in original_block_ids], dtype=torch.float32),
        must_keep_block_ids=must_keep,
        num_blocks_to_keep=k,
    )

    fa_sel = sum(1 for b in final if graph.feature_available.get(int(b), False))
    fb_sel = len(final) - fa_sel
    meta = {
        "policy": policy,
        "budget_k": k,
        "num_selected": len(final),
        "max_swap_cap": max_swap,
        "swaps_applied": len(swaps),
        "swaps": swaps,
        "churn_rate": round(active_set_churn(baseline, final, k), 6),
        "jaccard_vs_silm": round(len(baseline & final) / max(1, len(baseline | final)), 6),
        "sink_kept": int(original_block_ids[0] in final) if original_block_ids else 0,
        "over_budget": int(len(final) > k),
        "feature_available_selected_ratio": round(fa_sel / max(1, len(final)), 6),
        "fallback_selected_count": fb_sel,
        "incoming_mass_mean": round(float(np.mean(list(graph.incoming_mass.values()))) if graph.incoming_mass else 0.0, 6),
        "transport_edges": graph.num_edges,
        "edge_sparsity": round(graph.edge_sparsity, 6),
    }
    return final, meta


def replay_sp_v2_sample_layer(
    *,
    layer_idx: int,
    original_block_ids: Sequence[int],
    silm_scores: Mapping[int, float],
    silm_selected: Set[int],
    budget_k: int,
    history: Mapping[int, Mapping[str, Any]],
    scalar_rows: Sequence[Mapping[str, Any]],
    policy: str,
    prev_active: Optional[Set[int]] = None,
    tau: float = DEFAULT_TAU,
    top_r: int = DEFAULT_TOP_R,
    replacement_cap: float = DEFAULT_REPLACEMENT_CAP,
    weights: Optional[Mapping[str, float]] = None,
) -> Tuple[Set[int], TransportGraph, Dict[str, Any]]:
    if layer_idx == 9:
        selected = set(int(x) for x in silm_selected)
        empty = TransportGraph(prev_layer=-1, cur_layer=9, tau=tau, top_r=top_r)
        return selected, empty, {
            "policy": policy,
            "layer_idx": layer_idx,
            "transport_applied": False,
            "reason": "L9_no_history",
            "churn_rate": 0.0,
            "jaccard_vs_silm": 1.0,
            "num_selected": len(selected),
            "budget_k": budget_k,
            "sink_kept": int(original_block_ids[0] in selected) if original_block_ids else 0,
        }

    graph = build_transport_graph(
        history=history,
        scalar_rows=scalar_rows,
        cur_layer=layer_idx,
        tau=tau,
        top_r=top_r,
        candidate_block_ids=original_block_ids,
    )
    if graph.num_prev_available == 0 or graph.num_cur_available == 0:
        must_keep = _must_keep_block_ids(original_block_ids)
        selected = _trim_to_budget(
            set(silm_selected) | must_keep,
            original_block_ids=original_block_ids,
            block_scores=torch.tensor([float(silm_scores.get(int(b), float("-inf"))) for b in original_block_ids], dtype=torch.float32),
            must_keep_block_ids=must_keep,
            num_blocks_to_keep=int(budget_k),
        )
        return selected, graph, {
            "policy": policy,
            "layer_idx": layer_idx,
            "transport_applied": False,
            "reason": "insufficient_transport_features",
            "churn_rate": round(active_set_churn(set(silm_selected), selected, budget_k), 6),
            "jaccard_vs_silm": round(len(set(silm_selected) & selected) / max(1, len(set(silm_selected) | selected)), 6),
            "num_selected": len(selected),
            "budget_k": budget_k,
            "sink_kept": int(original_block_ids[0] in selected) if original_block_ids else 0,
        }

    selected, meta = bounded_transport_select(
        original_block_ids=original_block_ids,
        silm_scores=silm_scores,
        silm_selected=set(silm_selected),
        budget_k=budget_k,
        graph=graph,
        policy=policy,
        prev_active=prev_active,
        replacement_cap=replacement_cap,
        weights=weights,
    )
    meta["layer_idx"] = layer_idx
    meta["transport_applied"] = True
    return selected, graph, meta


def reset_sp_v2_runtime_state(cfg: Dict[str, Any]) -> None:
    for key in (
        "_sp_v2_layer_history",
        "_sp_v2_scalar_rows",
        "_sp_v2_prev_active",
        "_sp_v2_last",
        "_sp_v2_graph_last",
        "_sp_v2_prune_layer_selected",
    ):
        cfg.pop(key, None)
    cfg["_sp_v2_layer_history"] = {}
    cfg["_sp_v2_scalar_rows"] = []


def capture_sp_v2_runtime_layer_state(
    *,
    cfg: Dict[str, Any],
    original_block_ids: Sequence[int],
    qk_block_scores: torch.Tensor,
    q_for_pruning: torch.Tensor,
    represent_keys: torch.Tensor,
    block_to_kernel_map: Mapping[int, Tuple[int, int]],
) -> None:
    """In-memory causal state for online transport (no disk, no gold)."""
    layer_idx = int(cfg.get("layer_idx", -1))
    if layer_idx not in TRAJECTORY_PRUNE_LAYERS:
        return

    ctx = cfg.get("audit_context", {}) if isinstance(cfg.get("audit_context"), dict) else {}
    task = str(ctx.get("task", cfg.get("task", "")) or "")
    sample_id = str(ctx.get("sample_id", ctx.get("sample_index", "")) or "")
    prompt_len = int(cfg.get("original_prompt_len", ctx.get("context_length", 0)) or 0)
    hs_tokens = cfg.get("_trajectory_hs_tokens")
    abs_to_rel = cfg.get("_trajectory_query_abs_to_relative_map") or {}
    if not isinstance(abs_to_rel, dict):
        abs_to_rel = {}
    local_window = int(cfg.get("local_query_window", 4) or 4)
    local_query = pool_local_query_rep(q_for_pruning)
    hidden_query = pool_hidden_query_rep(hs_tokens, prompt_len, abs_to_rel, local_window)

    block_ids: List[int] = []
    hidden_rows: List[np.ndarray] = []
    key_rows: List[np.ndarray] = []
    censored: List[bool] = []
    rep_avail: List[bool] = []
    qh_list: List[Any] = []
    qk_list: List[Any] = []
    scores_list: List[Any] = []

    id_to_score = {
        int(original_block_ids[i]): float(qk_block_scores[i].item())
        for i in range(min(len(original_block_ids), int(qk_block_scores.numel())))
    }

    for block_id in original_block_ids:
        block_id = int(block_id)
        hidden_rep = pool_hidden_block_rep(hs_tokens, block_id, prompt_len, abs_to_rel)
        key_rep = pool_key_block_rep(represent_keys, block_to_kernel_map, block_id)
        rep_ok = hidden_rep is not None and key_rep is not None
        block_ids.append(block_id)
        hidden_rows.append(hidden_rep if hidden_rep is not None else np.array([np.nan], dtype=np.float32))
        key_rows.append(key_rep if key_rep is not None else np.array([np.nan], dtype=np.float32))
        censored.append(False)
        rep_avail.append(bool(rep_ok))
        if rep_ok and hidden_query is not None:
            qh_list.append(_cosine_np(hidden_query, hidden_rep))
            qk_list.append(_cosine_np(local_query, key_rep))
        else:
            qh_list.append(UNAVAILABLE)
            qk_list.append(UNAVAILABLE)
        scores_list.append(id_to_score.get(block_id, UNAVAILABLE))

        start, end, span_len = block_token_span(block_id, prompt_len)
        cfg.setdefault("_sp_v2_scalar_rows", []).append(
            {
                "sample_id": sample_id,
                "task": task,
                "layer_idx": layer_idx,
                "block_id": block_id,
                "token_start": start,
                "token_end": end,
                "valid_token_count": span_len,
                "alive_before_pruning": 1,
                "original_sliminfer_score": id_to_score.get(block_id, UNAVAILABLE),
                "query_hidden_cosine": qh_list[-1],
                "query_key_cosine": qk_list[-1],
                "representation_available": int(rep_ok),
                "trajectory_censored": 0,
            }
        )

    hidden_dim = hidden_rows[0].shape[0] if hidden_rows and hidden_rows[0].ndim == 1 and hidden_rows[0].size > 1 else 0
    if hidden_dim > 1:
        hidden_stack = np.stack(
            [r if r.size == hidden_dim else np.full((hidden_dim,), np.nan, dtype=np.float32) for r in hidden_rows],
            axis=0,
        )
        key_dim = key_rows[0].shape[0] if key_rows[0].size > 1 else 0
        key_stack = np.stack(
            [r if r.size == key_dim else np.full((key_dim,), np.nan, dtype=np.float32) for r in key_rows],
            axis=0,
        )
    else:
        hidden_stack = np.array([np.nan], dtype=np.float32)
        key_stack = np.array([np.nan], dtype=np.float32)

    history = dict(cfg.get("_sp_v2_layer_history") or {})
    history[int(layer_idx)] = {
        "block_ids": block_ids,
        "hidden": hidden_stack,
        "key": key_stack,
        "local_query": local_query,
        "query_hidden_cosine": qh_list,
        "query_key_cosine": qk_list,
        "scores": scores_list,
        "censored": censored,
        "rep_available": rep_avail,
    }
    cfg["_sp_v2_layer_history"] = history


def enrich_runtime_scalar_with_censored(
    cfg: Dict[str, Any],
    layer_idx: int,
    original_block_ids: Sequence[int],
) -> None:
    """Add censored rows for blocks selected at prior prune layers but not alive now."""
    if layer_idx not in (18, 26):
        return
    prune_hist = cfg.get("_sp_v2_prune_layer_selected") or {}
    prior: Set[int] = set()
    for pl in TRAJECTORY_PRUNE_LAYERS:
        if pl >= layer_idx:
            break
        prior.update(int(x) for x in prune_hist.get(int(pl), set()))
    alive = {int(x) for x in original_block_ids}
    censored = sorted(prior - alive)
    if not censored:
        return
    ctx = cfg.get("audit_context", {}) if isinstance(cfg.get("audit_context"), dict) else {}
    task = str(ctx.get("task", ""))
    sample_id = str(ctx.get("sample_id", ""))
    prompt_len = int(cfg.get("original_prompt_len", 0) or 0)
    rows = list(cfg.get("_sp_v2_scalar_rows") or [])
    existing = {(int(r.get("layer_idx", -1)), int(r.get("block_id", -1))) for r in rows if r.get("task") == task and str(r.get("sample_id")) == sample_id}
    for block_id in censored:
        if (layer_idx, block_id) in existing:
            continue
        start, end, span = block_token_span(block_id, prompt_len)
        rows.append(
            {
                "sample_id": sample_id,
                "task": task,
                "layer_idx": layer_idx,
                "block_id": block_id,
                "token_start": start,
                "token_end": end,
                "valid_token_count": span,
                "alive_before_pruning": 0,
                "original_sliminfer_score": UNAVAILABLE,
                "query_hidden_cosine": UNAVAILABLE,
                "query_key_cosine": UNAVAILABLE,
                "representation_available": 0,
                "trajectory_censored": 1,
            }
        )
    cfg["_sp_v2_scalar_rows"] = rows


def sp_v2_active_ids_runtime(
    *,
    original_block_ids: Sequence[int],
    block_scores: torch.Tensor,
    top_k_indices: torch.Tensor,
    must_keep_block_ids: Set[int],
    num_blocks_to_keep: int,
    cfg: Dict[str, Any],
    layer_idx: int,
) -> Tuple[Set[int], Dict[str, Any]]:
    """Runtime entry for sp_sliminfer_v2."""
    policy = str(cfg.get("sp_v2_policy", "transport_residual"))
    layer_idx = int(layer_idx)
    score_map = {int(bid): float(block_scores[i].item()) for i, bid in enumerate(original_block_ids)}
    top_local = [int(i) for i in top_k_indices.detach().cpu().tolist()]
    silm_selected = {int(original_block_ids[i]) for i in top_local} | {int(x) for x in must_keep_block_ids}
    silm_selected = _trim_to_budget(
        silm_selected,
        original_block_ids=original_block_ids,
        block_scores=block_scores,
        must_keep_block_ids=must_keep_block_ids,
        num_blocks_to_keep=int(num_blocks_to_keep),
    )

    if layer_idx == 9:
        cfg["_sp_v2_prev_active"] = sorted(int(x) for x in silm_selected)
        prune_hist = dict(cfg.get("_sp_v2_prune_layer_selected") or {})
        prune_hist[9] = set(int(x) for x in silm_selected)
        cfg["_sp_v2_prune_layer_selected"] = prune_hist
        meta = {
            "policy": policy,
            "layer_idx": 9,
            "transport_applied": False,
            "reason": "L9_original_sliminfer",
            "churn_rate": 0.0,
            "jaccard_vs_silm": 1.0,
            "num_selected": len(silm_selected),
            "budget_k": int(num_blocks_to_keep),
            "sink_kept": int(original_block_ids[0] in silm_selected) if original_block_ids else 0,
        }
        cfg["_sp_v2_last"] = meta
        return silm_selected, meta

    history = cfg.get("_sp_v2_layer_history", {})
    scalar_rows = cfg.get("_sp_v2_scalar_rows", [])
    enrich_runtime_scalar_with_censored(cfg, layer_idx, original_block_ids)
    scalar_rows = cfg.get("_sp_v2_scalar_rows", [])
    prev_active_raw = cfg.get("_sp_v2_prev_active")
    prev_active = {int(x) for x in prev_active_raw} if isinstance(prev_active_raw, (list, set, tuple)) else None

    selected, graph, meta = replay_sp_v2_sample_layer(
        layer_idx=int(layer_idx),
        original_block_ids=original_block_ids,
        silm_scores=score_map,
        silm_selected=silm_selected,
        budget_k=int(num_blocks_to_keep),
        history=history,
        scalar_rows=scalar_rows,
        policy=policy,
        prev_active=prev_active,
        tau=float(cfg.get("sp_v2_tau", DEFAULT_TAU)),
        top_r=int(cfg.get("sp_v2_top_r", DEFAULT_TOP_R)),
        replacement_cap=float(cfg.get("sp_v2_replacement_cap", DEFAULT_REPLACEMENT_CAP)),
    )
    cfg["_sp_v2_prev_active"] = sorted(int(x) for x in selected)
    prune_hist = dict(cfg.get("_sp_v2_prune_layer_selected") or {})
    prune_hist[int(layer_idx)] = set(int(x) for x in selected)
    cfg["_sp_v2_prune_layer_selected"] = prune_hist
    cfg["_sp_v2_last"] = meta
    cfg.setdefault("_sp_v2_graph_last", {})[int(layer_idx)] = {
        "num_edges": graph.num_edges,
        "edge_sparsity": graph.edge_sparsity,
    }
    return selected, meta
