"""SP-SlimInfer: semantic propagation proxies and CPU replay selection (no gold)."""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

import torch

from bridgekv.confidence_gated_sliminfer import _trim_to_budget

PRUNE_LAYERS = [9, 18, 26]

DEFAULT_SP_WEIGHTS = {
    "persistence": 0.15,
    "absorption": 0.12,
    "drift": 0.10,
    "relay": 0.12,
    "redundancy": 0.08,
}

# Softer weights for runtime score/rank proxies (offline property_features are milder).
RUNTIME_SP_WEIGHTS = {
    "persistence": 0.02,
    "absorption": 0.016,
    "drift": 0.014,
    "relay": 0.016,
    "redundancy": 0.012,
}

FEATURE_SOURCES = {
    "semantic_persistence": "PROXY",
    "semantic_absorption": "PROXY",
    "semantic_drift": "PROXY",
    "semantic_relay": "PROXY",
    "active_set_redundancy": "PROXY",
    "hidden_state_trajectory": "UNAVAILABLE",
    "key_state_representation_drift": "UNAVAILABLE",
}


def _z(values: Sequence[float], x: float) -> float:
    if not values:
        return 0.0
    mu = sum(values) / len(values)
    var = sum((v - mu) ** 2 for v in values) / max(1, len(values) - 1)
    std = math.sqrt(var) if var > 1e-12 else 1.0
    return (float(x) - mu) / std


def _f(row: Mapping[str, Any], key: str) -> Optional[float]:
    v = row.get(key)
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def build_block_layer_index(
    feature_rows: Sequence[Mapping[str, Any]],
) -> Dict[Tuple[str, str, int], Dict[int, Dict[str, Any]]]:
    """sample_id -> layer -> block_id -> row."""
    out: Dict[Tuple[str, str, int], Dict[int, Dict[str, Any]]] = defaultdict(dict)
    for row in feature_rows:
        sid = str(row.get("sample_id", ""))
        task = str(row.get("task", ""))
        layer = int(row.get("layer", -1))
        bid = int(row.get("block_id", -1))
        if sid and bid >= 0 and layer >= 0:
            out[(task, sid, layer)][bid] = dict(row)
    return out


def compute_sp_features_for_block(
    *,
    task: str,
    sample_id: str,
    block_id: int,
    layer: int,
    layer_index: Mapping[Tuple[str, str, int], Dict[int, Dict[str, Any]]],
    prune_layers: Sequence[int] = PRUNE_LAYERS,
) -> Dict[str, Any]:
    rows_by_layer: Dict[int, Dict[str, Any]] = {}
    for pl in prune_layers:
        rows_by_layer[pl] = layer_index.get((task, sample_id, pl), {}).get(block_id, {})

    cur = rows_by_layer.get(layer, {})
    if not cur:
        return {
            "semantic_persistence": "",
            "semantic_absorption": "",
            "semantic_drift": "",
            "semantic_relay": "",
            "active_set_redundancy": "",
            "persistence_status": "UNAVAILABLE",
            "absorption_status": "UNAVAILABLE",
            "drift_status": "UNAVAILABLE",
            "relay_status": "UNAVAILABLE",
            "redundancy_status": "UNAVAILABLE",
        }

    persist_vals = [_f(rows_by_layer[pl], "persistence_ratio") for pl in prune_layers]
    persist_vals = [v for v in persist_vals if v is not None]
    sel_flags = [int(rows_by_layer[pl].get("selected_by_silm", 0) or 0) for pl in prune_layers if rows_by_layer[pl]]
    persistence = (
        (sum(persist_vals) / len(persist_vals)) * (sum(sel_flags) / max(1, len(sel_flags)))
        if persist_vals
        else None
    )

    qg = [_f(rows_by_layer[pl], "query_conditioned_gain") for pl in prune_layers]
    qg_finite = [v for v in qg if v is not None]
    drift = None
    if len(qg_finite) >= 2:
        drift = float(max(qg_finite) - min(qg_finite))
    ent = _f(cur, "propagation_entropy_or_proxy")
    if drift is not None and ent is not None:
        drift = 0.6 * drift + 0.4 * ent
    elif drift is None and ent is not None:
        drift = ent

    early = rows_by_layer.get(prune_layers[0], {})
    late = rows_by_layer.get(prune_layers[-1], {})
    absorption = None
    e_raw = _f(early, "raw_diffusion_score_or_proxy")
    l_red = _f(late, "redundancy_proxy")
    e_sel = int(early.get("selected_by_silm", 0) or 0)
    l_sel = int(late.get("selected_by_silm", 0) or 0)
    if e_raw is not None and l_red is not None and e_sel and not l_sel:
        absorption = max(0.0, float(l_red) * (1.0 + abs(e_raw) * 0.01))

    relay_row = rows_by_layer.get(18, cur)
    bridge = _f(relay_row, "bridge_flow_proxy")
    outflow = _f(relay_row, "directional_outflow_proxy")
    relay = None
    if bridge is not None or outflow is not None:
        relay = max(bridge or 0.0, outflow or 0.0)

    redundancy = _f(cur, "redundancy_proxy")

    def status(val: Optional[float]) -> str:
        return "PROXY" if val is not None else "UNAVAILABLE"

    return {
        "semantic_persistence": round(persistence, 6) if persistence is not None else "",
        "semantic_absorption": round(absorption, 6) if absorption is not None else "",
        "semantic_drift": round(drift, 6) if drift is not None else "",
        "semantic_relay": round(relay, 6) if relay is not None else "",
        "active_set_redundancy": round(redundancy, 6) if redundancy is not None else "",
        "persistence_status": status(persistence),
        "absorption_status": status(absorption),
        "drift_status": status(drift),
        "relay_status": status(relay),
        "redundancy_status": status(redundancy),
        "hidden_state_trajectory": "UNAVAILABLE",
        "key_state_representation_drift": "UNAVAILABLE",
    }


def apply_sp_score_adjustment(
    block_scores: torch.Tensor,
    original_block_ids: Sequence[int],
    sp_features: Mapping[int, Mapping[str, Any]],
    *,
    weights: Optional[Mapping[str, float]] = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    w = dict(DEFAULT_SP_WEIGHTS)
    if weights:
        w.update(weights)

    pools = {k: [] for k in ("persistence", "absorption", "drift", "relay", "redundancy")}
    for bid in original_block_ids:
        feat = sp_features.get(int(bid), {})
        for key, pool_key in [
            ("semantic_persistence", "persistence"),
            ("semantic_absorption", "absorption"),
            ("semantic_drift", "drift"),
            ("semantic_relay", "relay"),
            ("active_set_redundancy", "redundancy"),
        ]:
            v = feat.get(key, "")
            if v != "" and v is not None:
                pools[pool_key].append(float(v))

    adjusted = block_scores.clone().float()
    for i, bid in enumerate(original_block_ids):
        feat = sp_features.get(int(bid), {})
        delta = 0.0
        mapping = [
            ("semantic_persistence", "persistence", w["persistence"]),
            ("semantic_absorption", "absorption", -w["absorption"]),
            ("semantic_drift", "drift", -w["drift"]),
            ("semantic_relay", "relay", w["relay"]),
            ("active_set_redundancy", "redundancy", -w["redundancy"]),
        ]
        for fkey, pool_key, coef in mapping:
            v = feat.get(fkey, "")
            if v == "" or v is None:
                continue
            delta += coef * _z(pools[pool_key], float(v))
        adjusted[i] = adjusted[i] + delta

    return adjusted, {"weights": w, "adjusted": True}


def replay_sp_active_set(
    *,
    block_ids: Sequence[int],
    silm_scores: Sequence[float],
    silm_selected: Set[int],
    sp_features: Mapping[int, Mapping[str, Any]],
    budget: int,
    must_keep: Set[int],
) -> Tuple[Set[int], Dict[str, Any]]:
    scores = torch.tensor(list(silm_scores), dtype=torch.float32)
    adjusted, adj_meta = apply_sp_score_adjustment(scores, block_ids, sp_features)
    k = min(max(1, budget), len(block_ids))
    _, top_idx = torch.topk(adjusted, k=k)
    top_ids = {block_ids[i] for i in top_idx.tolist()} | {int(x) for x in must_keep}
    final = _trim_to_budget(
        top_ids,
        original_block_ids=block_ids,
        block_scores=adjusted,
        must_keep_block_ids=must_keep,
        num_blocks_to_keep=k,
    )
    overlap = len(final & silm_selected)
    union = len(final | silm_selected)
    jaccard = overlap / max(1, union)
    churn = 1.0 - (len(final & silm_selected) / max(1, k))
    return final, {
        **adj_meta,
        "jaccard_vs_silm": round(jaccard, 6),
        "churn_rate": round(churn, 6),
        "num_selected": len(final),
        "budget_k": k,
        "sink_kept": int(block_ids[0] in final) if block_ids else 0,
    }


def active_set_churn(silm: Set[int], variant: Set[int], budget: int) -> float:
    if budget <= 0:
        return 0.0
    return 1.0 - len(silm & variant) / float(budget)


def _score_ranks(block_ids: Sequence[int], scores: Mapping[int, float]) -> Dict[int, int]:
    ordered = sorted(block_ids, key=lambda bid: scores.get(int(bid), float("-inf")), reverse=True)
    return {int(bid): rank + 1 for rank, bid in enumerate(ordered)}


def record_sp_layer_trace(
    cfg: Dict[str, Any],
    *,
    layer_idx: int,
    original_block_ids: Sequence[int],
    block_scores: torch.Tensor,
    selected_ids: Set[int],
) -> None:
    trace = cfg.setdefault("_sp_layer_trace", {})
    scores_list = block_scores.detach().float().cpu().tolist()
    score_map = {int(bid): float(scores_list[i]) for i, bid in enumerate(original_block_ids)}
    trace[int(layer_idx)] = {
        "scores": score_map,
        "selected": sorted(int(x) for x in selected_ids),
        "ranks": _score_ranks(original_block_ids, score_map),
    }


def compute_runtime_sp_features(
    cfg: Dict[str, Any],
    *,
    layer_idx: int,
    block_ids: Sequence[int],
    current_scores: Optional[Mapping[int, float]] = None,
    prune_layers: Sequence[int] = PRUNE_LAYERS,
) -> Dict[int, Dict[str, Any]]:
    """Gold-free score/rank proxies from prior prune-layer traces (runtime path)."""
    trace: Mapping[int, Mapping[str, Any]] = cfg.get("_sp_layer_trace", {})
    available = [int(pl) for pl in prune_layers if int(pl) in trace and int(pl) <= int(layer_idx)]
    cur_scores: Mapping[int, float] = current_scores or trace.get(int(layer_idx), {}).get("scores", {})
    cur_ranks = _score_ranks(block_ids, cur_scores) if cur_scores else {}

    def status(val: Optional[float]) -> str:
        return "RUNTIME_PROXY" if val is not None else "UNAVAILABLE"

    out: Dict[int, Dict[str, Any]] = {}
    for bid in block_ids:
        bid = int(bid)
        sel_fracs: List[float] = []
        inv_ranks: List[float] = []
        layer_score_vals: List[float] = []
        for pl in available:
            layer_data = trace[int(pl)]
            scores_pl: Mapping[int, float] = layer_data.get("scores", {})
            ranks_pl: Mapping[int, int] = layer_data.get("ranks", {})
            if bid in scores_pl:
                layer_score_vals.append(float(scores_pl[bid]))
                inv_ranks.append(1.0 / (1.0 + float(ranks_pl.get(bid, len(block_ids)))))
                selected_pl = set(layer_data.get("selected", []))
                sel_fracs.append(1.0 if bid in selected_pl else 0.0)

        persistence = None
        if sel_fracs and inv_ranks:
            persistence = (sum(sel_fracs) / len(sel_fracs)) * (sum(inv_ranks) / len(inv_ranks))

        drift = None
        if len(layer_score_vals) >= 2:
            drift = float(max(layer_score_vals) - min(layer_score_vals))
            if cur_scores:
                span = max(cur_scores.values()) - min(cur_scores.values())
                drift = drift / max(float(span), 1e-6)

        absorption = None
        if available:
            first_pl = int(available[0])
            first = trace[first_pl]
            first_scores: Mapping[int, float] = first.get("scores", {})
            first_ranks: Mapping[int, int] = first.get("ranks", {})
            if bid in first_scores and int(layer_idx) > first_pl and bid in cur_scores:
                n_blocks = max(1, len(first_scores))
                was_early_strong = int(first_ranks.get(bid, n_blocks)) <= max(1, n_blocks // 4)
                late_rank = int(cur_ranks.get(bid, n_blocks))
                if was_early_strong and late_rank > max(1, len(cur_scores) // 2):
                    band = 0.05 * max(abs(float(cur_scores[bid])), 1e-6)
                    similar = sum(
                        1 for other, score in cur_scores.items()
                        if abs(float(score) - float(cur_scores[bid])) <= band
                    )
                    absorption = float(similar) / max(1.0, float(len(cur_scores)))

        relay = None
        if 18 in available and 9 in available:
            r9 = trace[9].get("ranks", {}).get(bid)
            r18 = trace[18].get("ranks", {}).get(bid)
            if r9 is not None and r18 is not None and int(r18) < int(r9):
                relay = float(int(r9) - int(r18)) / max(1.0, float(r9))

        redundancy = None
        if bid in cur_scores:
            band = 0.05 * max(abs(float(cur_scores[bid])), 1e-6)
            redundancy = sum(
                1 for score in cur_scores.values() if abs(float(score) - float(cur_scores[bid])) <= band
            ) / max(1.0, float(len(cur_scores)))

        out[bid] = {
            "semantic_persistence": round(persistence, 6) if persistence is not None else "",
            "semantic_absorption": round(absorption, 6) if absorption is not None else "",
            "semantic_drift": round(drift, 6) if drift is not None else "",
            "semantic_relay": round(relay, 6) if relay is not None else "",
            "active_set_redundancy": round(redundancy, 6) if redundancy is not None else "",
            "persistence_status": status(persistence),
            "absorption_status": status(absorption),
            "drift_status": status(drift),
            "relay_status": status(relay),
            "redundancy_status": status(redundancy),
            "hidden_state_trajectory": "UNAVAILABLE",
            "key_state_representation_drift": "UNAVAILABLE",
        }
    return out


def sp_sliminfer_active_ids(
    *,
    original_block_ids: Sequence[int],
    block_scores: torch.Tensor,
    top_k_indices: torch.Tensor,
    must_keep_block_ids: Set[int],
    num_blocks_to_keep: int,
    cfg: Dict[str, Any],
    layer_idx: int,
    prune_layers: Sequence[int] = PRUNE_LAYERS,
    weights: Optional[Mapping[str, float]] = None,
) -> Tuple[Set[int], Dict[str, Any]]:
    scores_list = block_scores.detach().float().cpu().tolist()
    score_map = {int(bid): float(scores_list[i]) for i, bid in enumerate(original_block_ids)}
    use_weights = dict(RUNTIME_SP_WEIGHTS)
    if weights:
        use_weights.update(weights)
    trace_layers = [
        int(pl) for pl in prune_layers
        if int(pl) in cfg.get("_sp_layer_trace", {}) and int(pl) < int(layer_idx)
    ]
    layer_scale = min(1.0, len(trace_layers) / 2.0)
    use_weights = {k: float(v) * layer_scale for k, v in use_weights.items()}
    sp_features = compute_runtime_sp_features(
        cfg,
        layer_idx=layer_idx,
        block_ids=original_block_ids,
        current_scores=score_map,
        prune_layers=prune_layers,
    )
    adjusted, adj_meta = apply_sp_score_adjustment(
        block_scores,
        original_block_ids,
        sp_features,
        weights=use_weights,
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
    meta = {
        **adj_meta,
        "mode": "sp_runtime_score_adjust",
        "layers_available": trace_layers,
        "layer_scale": round(layer_scale, 6),
        "jaccard_vs_silm_topk": round(overlap / max(1, union), 6),
        "churn_rate": round(active_set_churn(silm_top, final, k), 6),
        "num_selected": len(final),
        "sink_kept": int(bool(original_block_ids and int(original_block_ids[0]) in final)),
    }
    record_sp_layer_trace(
        cfg,
        layer_idx=layer_idx,
        original_block_ids=original_block_ids,
        block_scores=block_scores,
        selected_ids=final,
    )
    return final, meta
