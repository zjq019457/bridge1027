"""RoleFlow-KV: propagation-role-aware score adjustment for SlimInfer KV retention."""
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

import torch

from bridgekv.risk_gated_roleflow import (
    filter_removable_for_swap_guard,
    hub_bridge_policy,
    risk_gate_mode,
    should_fallback_to_silm,
)

RoleProfile = Dict[str, Any]


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(out):
        return default
    return out


def zscore_map(values: Mapping[int, float]) -> Dict[int, float]:
    finite = [float(v) for v in values.values() if math.isfinite(float(v))]
    if not finite:
        return {int(k): 0.0 for k in values}
    mean = sum(finite) / len(finite)
    var = sum((v - mean) ** 2 for v in finite) / max(1, len(finite))
    std = math.sqrt(var) if var > 1e-12 else 1.0
    return {int(k): (float(v) - mean) / std for k, v in values.items()}


def middle_late_mask(layer_idx: int, num_layers: int, *, start_ratio: float = 0.33) -> float:
    if num_layers <= 1:
        return 1.0
    return 1.0 if layer_idx >= int(num_layers * start_ratio) else 0.0


def late_layer_mask(layer_idx: int, num_layers: int, *, start_ratio: float = 0.66) -> float:
    if num_layers <= 1:
        return 0.0
    return 1.0 if layer_idx >= int(num_layers * start_ratio) else 0.0


def early_layer_mask(layer_idx: int, num_layers: int, *, end_ratio: float = 0.33) -> float:
    if num_layers <= 1:
        return 1.0
    return 1.0 if layer_idx < int(num_layers * end_ratio) else 0.0


def build_role_profiles_from_features(
    feature_rows: Sequence[Mapping[str, Any]],
    *,
    top20_fraction: float = 0.2,
) -> Dict[int, RoleProfile]:
    by_block: Dict[int, List[Mapping[str, Any]]] = {}
    for row in feature_rows:
        block_id = int(row.get("block_id", -1))
        if block_id < 0:
            continue
        by_block.setdefault(block_id, []).append(row)

    qcg_raw: Dict[int, float] = {}
    profiles: Dict[int, RoleProfile] = {}
    for block_id, rows in by_block.items():
        layer_scores: List[Tuple[int, float]] = []
        for row in rows:
            layer = int(row.get("layer", row.get("layer_id", -1)))
            qk = row.get("raw_diffusion_score_or_proxy", "")
            score = _as_float(qk, float("nan"))
            if math.isfinite(score):
                layer_scores.append((layer, score))
        layer_scores.sort(key=lambda item: item[0])
        early_layers = layer_scores[: max(1, len(layer_scores) // 2)]
        late_layers = layer_scores[len(layer_scores) // 2 :] or layer_scores
        early_peak = max((s for _, s in early_layers), default=0.0)
        late_peak = max((s for _, s in late_layers), default=0.0)
        peak_layer = max(layer_scores, key=lambda item: item[1])[0] if layer_scores else ""
        onset_layer = ""
        if layer_scores:
            ranked = sorted(layer_scores, key=lambda item: item[1], reverse=True)
            threshold_rank = max(1, int(math.ceil(len(layer_scores) * top20_fraction)))
            top_cut = ranked[threshold_rank - 1][1] if ranked else 0.0
            for layer, score in layer_scores:
                if score >= top_cut:
                    onset_layer = layer
                    break

        rep = rows[0]
        persistence = max(_as_float(r.get("persistence_ratio", 0.0)) for r in rows)
        qcg = _as_float(rep.get("query_conditioned_gain", 0.0))
        hub = _as_float(rep.get("hub_score", 0.0))
        bridge = _as_float(rep.get("bridge_flow_proxy", 0.0))
        neighbor = _as_float(rep.get("neighbor_gain_proxy", 0.0))
        diffusion_agg = max(
            (_as_float(r.get("diffusion_only_aggregate_score", r.get("raw_diffusion_score_or_proxy", 0.0))) for r in rows),
            default=0.0,
        )
        qcg_raw[block_id] = qcg
        profiles[block_id] = {
            "persistence_ratio_top20": persistence,
            "early_peak_score": early_peak,
            "late_decay_score": max(0.0, early_peak - late_peak),
            "onset_layer_top20": onset_layer,
            "peak_layer": peak_layer,
            "query_conditioned_gain": qcg,
            "hub_score": hub,
            "bridge_flow_proxy": bridge,
            "neighbor_gain_proxy": neighbor,
            "generic_hub_score": hub,
            "boundary_sensitive_score": neighbor,
            "raw_diffusion_score_or_proxy": diffusion_agg,
        }

    qcg_z = zscore_map(qcg_raw)
    for block_id, profile in profiles.items():
        profile["query_conditioned_gain_z"] = qcg_z.get(block_id, 0.0)
        profile["role_persistent_anchor"] = int(
            profile["persistence_ratio_top20"] >= 0.5
            and profile["query_conditioned_gain_z"] > 0.0
            and profile["hub_score"] < 0.4
        )
        profile["role_transient_trigger"] = int(
            profile["early_peak_score"] > profile["late_decay_score"]
            and profile["late_decay_score"] > 0.05
        )
        profile["role_bridge_relay"] = int(
            profile["bridge_flow_proxy"] > 0.02 and 0.2 <= profile["persistence_ratio_top20"] <= 0.95
        )
        profile["role_generic_hub"] = int(profile["generic_hub_score"] >= 0.35)
        profile["role_boundary_sensitive"] = int(profile["boundary_sensitive_score"] > 0.1)
        profile["roleflow_priority"] = (
            3 * profile["role_persistent_anchor"]
            + 2 * profile["role_bridge_relay"]
            + profile["role_boundary_sensitive"]
            - 2 * profile["role_generic_hub"]
            - profile["role_transient_trigger"]
        )
    return profiles


def build_layer_role_profiles_from_features(
    feature_rows: Sequence[Mapping[str, Any]],
    *,
    prune_layers: Optional[Sequence[int]] = None,
) -> Dict[int, Dict[int, RoleProfile]]:
    """Per prune-layer block role profiles: layer_id -> block_id -> profile."""
    allowed = {int(x) for x in prune_layers} if prune_layers is not None else None
    by_layer: Dict[int, List[Mapping[str, Any]]] = {}
    for row in feature_rows:
        layer = int(row.get("layer", row.get("layer_id", -1)))
        if layer < 0:
            continue
        if allowed is not None and layer not in allowed:
            continue
        by_layer.setdefault(layer, []).append(row)
    return {layer: build_role_profiles_from_features(rows) for layer, rows in sorted(by_layer.items())}


def profiles_for_layer(cfg: Mapping[str, Any], layer_idx: int) -> Mapping[int, RoleProfile]:
    by_layer = cfg.get("roleflow_profiles_by_layer") or {}
    layer_profiles = by_layer.get(int(layer_idx)) or by_layer.get(str(layer_idx))
    if layer_profiles:
        return layer_profiles
    return cfg.get("roleflow_profiles") or {}


def active_set_churn(silm_selected: Set[int], roleflow_selected: Set[int], budget: int) -> float:
    if budget <= 0:
        return 0.0
    return 1.0 - len(silm_selected.intersection(roleflow_selected)) / float(budget)


def _trim_to_budget(
    selected: Set[int],
    *,
    must_keep_block_ids: Set[int],
    num_blocks_to_keep: int,
    block_scores: torch.Tensor,
    id_to_idx: Mapping[int, int],
) -> Set[int]:
    out = set(selected).union(int(x) for x in must_keep_block_ids)
    if len(out) <= num_blocks_to_keep:
        return out
    removable = [bid for bid in out if bid not in must_keep_block_ids]
    removable.sort(key=lambda bid: float(block_scores[id_to_idx[bid]].item()) if bid in id_to_idx else float("-inf"))
    for bid in removable[: max(0, len(out) - num_blocks_to_keep)]:
        out.discard(bid)
    return out


def calibrate_active_set_churn(
    *,
    silm_selected: Set[int],
    original_block_ids: Sequence[int],
    block_scores: torch.Tensor,
    must_keep_block_ids: Set[int],
    num_blocks_to_keep: int,
    cfg: Mapping[str, Any],
) -> Tuple[Set[int], Dict[str, Any]]:
    """Swap low-priority SILM blocks for high-priority role candidates to hit target churn."""
    id_to_idx = _id_to_index(original_block_ids)
    budget = int(num_blocks_to_keep)
    silm_trimmed = _trim_to_budget(
        silm_selected,
        must_keep_block_ids=must_keep_block_ids,
        num_blocks_to_keep=budget,
        block_scores=block_scores,
        id_to_idx=id_to_idx,
    )
    target = float(cfg.get("roleflow_target_churn", 0.10))
    churn_min = float(cfg.get("roleflow_target_churn_min", 0.05))
    churn_max = float(cfg.get("roleflow_target_churn_max", 0.15))
    target = min(churn_max, max(churn_min, target))
    target_swaps = int(round(budget * target))
    target_swaps = min(target_swaps, int(math.floor(budget * churn_max)))
    max_swappable = max(0, len(silm_trimmed) - len(must_keep_block_ids))
    target_swaps = max(0, min(target_swaps, max_swappable))

    profiles = profiles_for_layer(cfg, int(cfg.get("layer_idx", 0) or 0))

    selected = set(silm_trimmed)
    removable = [bid for bid in silm_trimmed if bid not in must_keep_block_ids]
    swap_guard_meta: Dict[str, Any] = {}
    policy = str(cfg.get("roleflow_policy", cfg.get("variant_method", "")) or "")
    if risk_gate_mode(policy) == "swap_guard":
        removable, swap_guard_meta = filter_removable_for_swap_guard(removable, profiles, cfg)
    removable.sort(
        key=lambda bid: (
            -int(profiles.get(int(bid), {}).get("role_generic_hub", 0)),
            float(block_scores[id_to_idx[bid]].item()) if bid in id_to_idx else float("-inf"),
        )
    )
    outside = [int(bid) for bid in original_block_ids if int(bid) not in silm_trimmed]
    outside.sort(
        key=lambda bid: (
            int(profiles.get(int(bid), {}).get("role_persistent_anchor", 0))
            + int(profiles.get(int(bid), {}).get("role_bridge_relay", 0)),
            float(block_scores[id_to_idx[bid]].item()) if bid in id_to_idx else float("-inf"),
        ),
        reverse=True,
    )

    swapped = 0
    rem_i = 0
    add_i = 0
    swap_pairs: List[Tuple[int, int]] = []
    while swapped < target_swaps and rem_i < len(removable) and add_i < len(outside):
        out_bid = int(removable[rem_i])
        in_bid = int(outside[add_i])
        rem_i += 1
        add_i += 1
        if out_bid in must_keep_block_ids or in_bid in selected:
            continue
        selected.discard(out_bid)
        selected.add(in_bid)
        swap_pairs.append((out_bid, in_bid))
        swapped += 1

    selected = _trim_to_budget(
        selected,
        must_keep_block_ids=must_keep_block_ids,
        num_blocks_to_keep=budget,
        block_scores=block_scores,
        id_to_idx=id_to_idx,
    )
    churn = active_set_churn(silm_trimmed, selected, budget)
    meta = {
        "churn_calibration": True,
        "target_churn": target,
        "target_churn_min": churn_min,
        "target_churn_max": churn_max,
        "target_swaps": target_swaps,
        "actual_swaps": swapped,
        "churn_rate": churn,
        "churn_in_target": int(churn_min <= churn <= churn_max),
        "swap_pairs_sample": swap_pairs[:16],
    }
    meta.update(swap_guard_meta)
    return selected, meta


def role_enabled(policy: str, role_name: str) -> bool:
    policy = (policy or "none").lower()
    if policy in {"", "none", "silm_original"}:
        return False
    if policy == "roleflow_full":
        return True
    if policy == "roleflow_hubcap":
        return role_name == "generic_hub"
    if policy == "roleflow_persistence":
        return role_name in {"persistent_anchor", "generic_hub"}
    if policy == "roleflow_bridgeprotect":
        return role_name in {"bridge_relay", "generic_hub"}
    if policy == "roleflow_hub_bridge":
        return role_name in {"bridge_relay", "generic_hub"}
    if policy in {
        "roleflow_risk_task_gate",
        "roleflow_risk_sample_gate",
        "roleflow_risk_swap_guard",
    }:
        return role_name in {"bridge_relay", "generic_hub"}
    if policy == "roleflow_boundary":
        return role_name in {"boundary_sensitive", "generic_hub"}
    return False


def apply_roleflow_score_adjustment(
    block_scores: torch.Tensor,
    original_block_ids: Sequence[int],
    cfg: Mapping[str, Any],
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    policy = str(cfg.get("roleflow_policy", cfg.get("variant_method", "")) or "")
    if policy in {"", "none", "sliminfer_original"}:
        return block_scores, {"applied": False, "policy": policy}
    if hub_bridge_policy(policy) and risk_gate_mode(policy) in {"task", "sample"}:
        # Defer sample/task gate to selection where SILM active set is known.
        pass

    layer_idx = int(cfg.get("layer_idx", 0) or 0)
    profiles: Mapping[int, RoleProfile] = profiles_for_layer(cfg, layer_idx)
    num_layers = int(cfg.get("num_layers", 28) or 28)
    alpha_anchor = float(cfg.get("roleflow_alpha_anchor", 0.4))
    alpha_bridge = float(cfg.get("roleflow_alpha_bridge", 0.4))
    alpha_boundary = float(cfg.get("roleflow_alpha_boundary", 0.2))
    lambda_hub = float(cfg.get("roleflow_lambda_hub", 0.6))
    lambda_transient = float(cfg.get("roleflow_lambda_transient", 0.3))

    mid_late = middle_late_mask(layer_idx, num_layers)
    late = late_layer_mask(layer_idx, num_layers)
    adjusted = block_scores.clone()
    diagnostics: Dict[str, Any] = {
        "applied": True,
        "policy": policy,
        "layer_idx": layer_idx,
        "num_layers": num_layers,
        "per_block": {},
    }

    for idx, block_id in enumerate(original_block_ids):
        prof = profiles.get(int(block_id), {})
        anchor = float(prof.get("role_persistent_anchor", 0))
        bridge = float(prof.get("role_bridge_relay", 0))
        boundary = float(prof.get("role_boundary_sensitive", 0))
        hub = float(prof.get("role_generic_hub", 0))
        transient = float(prof.get("role_transient_trigger", 0))
        delta = 0.0
        if role_enabled(policy, "persistent_anchor"):
            delta += alpha_anchor * anchor * mid_late
        if role_enabled(policy, "bridge_relay"):
            delta += alpha_bridge * bridge * mid_late
        if role_enabled(policy, "boundary_sensitive"):
            delta += alpha_boundary * boundary
        if role_enabled(policy, "generic_hub"):
            delta -= lambda_hub * hub
        if role_enabled(policy, "transient_trigger"):
            delta -= lambda_transient * transient * late
        if delta != 0.0:
            adjusted[idx] = adjusted[idx] + float(delta)
        diagnostics["per_block"][int(block_id)] = float(delta)
    return adjusted, diagnostics


def _id_to_index(original_block_ids: Sequence[int]) -> Dict[int, int]:
    return {int(block_id): idx for idx, block_id in enumerate(original_block_ids)}


def select_roleflow_kv_ids(
    *,
    original_block_ids: Sequence[int],
    block_scores: torch.Tensor,
    top_k_indices: torch.Tensor,
    must_keep_block_ids: Set[int],
    num_blocks_to_keep: int,
    cfg: Mapping[str, Any],
    silm_block_scores: Optional[torch.Tensor] = None,
) -> Tuple[Set[int], Dict[str, Any]]:
    policy = str(cfg.get("roleflow_policy", cfg.get("variant_method", "")) or "")
    layer_idx = int(cfg.get("layer_idx", 0) or 0)
    profiles: Mapping[int, RoleProfile] = profiles_for_layer(cfg, layer_idx)
    id_to_idx = _id_to_index(original_block_ids)
    budget = int(num_blocks_to_keep)

    silm_scores = silm_block_scores if silm_block_scores is not None else block_scores
    silm_k = min(max(1, budget), len(original_block_ids))
    _, silm_top_idx = torch.topk(silm_scores, k=silm_k)
    silm_top_block_ids = {int(original_block_ids[i]) for i in silm_top_idx.tolist()}
    silm_selected = _trim_to_budget(
        silm_top_block_ids,
        must_keep_block_ids=must_keep_block_ids,
        num_blocks_to_keep=budget,
        block_scores=silm_scores,
        id_to_idx=id_to_idx,
    )

    fallback, gate_meta = should_fallback_to_silm(
        policy,
        cfg,
        silm_selected=silm_selected,
        original_block_ids=original_block_ids,
        profiles=profiles,
    )
    if fallback:
        return silm_selected, {
            "selection_policy": policy,
            "active_budget": budget,
            "selected_block_count": len(silm_selected),
            "overlap_with_silm_original": 1.0,
            "churn_rate": 0.0,
            "churn_in_target": 1,
            "risk_gate_fallback": 1,
            "over_budget": 0,
            **gate_meta,
        }

    use_churn_calibration = bool(cfg.get("roleflow_churn_calibration", True))
    churn_meta: Dict[str, Any] = {}
    if use_churn_calibration and policy not in {"", "none", "sliminfer_original"}:
        selected, churn_meta = calibrate_active_set_churn(
            silm_selected=silm_selected,
            original_block_ids=original_block_ids,
            block_scores=block_scores,
            must_keep_block_ids=must_keep_block_ids,
            num_blocks_to_keep=budget,
            cfg=cfg,
        )
    else:
        top_k_block_ids = {int(original_block_ids[i]) for i in top_k_indices.tolist()}
        selected = _trim_to_budget(
            top_k_block_ids,
            must_keep_block_ids=must_keep_block_ids,
            num_blocks_to_keep=budget,
            block_scores=block_scores,
            id_to_idx=id_to_idx,
        )

    hub_cap_ratio = float(cfg.get("roleflow_hub_cap_ratio", 0.10))
    hub_blocks = [bid for bid in selected if int(profiles.get(int(bid), {}).get("role_generic_hub", 0))]
    max_hubs = max(0, int(math.floor(num_blocks_to_keep * hub_cap_ratio)))
    boundary_added = 0
    transient_late = 0
    num_layers = int(cfg.get("num_layers", 28) or 28)
    if late_layer_mask(layer_idx, num_layers) > 0:
        transient_late = sum(
            1 for bid in selected if int(profiles.get(int(bid), {}).get("role_transient_trigger", 0))
        )

    if not use_churn_calibration and role_enabled(policy, "generic_hub") and len(hub_blocks) > max_hubs:
        hub_blocks_sorted = sorted(
            hub_blocks,
            key=lambda bid: float(block_scores[id_to_idx[bid]].item()) if bid in id_to_idx else float("-inf"),
        )
        for bid in hub_blocks_sorted[max_hubs:]:
            if bid in must_keep_block_ids:
                continue
            selected.discard(bid)

    if not use_churn_calibration and role_enabled(policy, "boundary_sensitive"):
        boundary_selected = [
            bid for bid in selected if int(profiles.get(int(bid), {}).get("role_boundary_sensitive", 0))
        ]
        for bid in boundary_selected:
            for neighbor in (bid - 1, bid + 1):
                if neighbor in selected or neighbor < 0 or neighbor >= len(original_block_ids):
                    continue
                if neighbor in must_keep_block_ids:
                    selected.add(neighbor)
                    boundary_added += 1
                    continue
                if len(selected) >= num_blocks_to_keep:
                    removable = [
                        b
                        for b in selected
                        if b not in must_keep_block_ids and b not in boundary_selected and b != neighbor
                    ]
                    if not removable:
                        break
                    removable.sort(
                        key=lambda b: float(block_scores[id_to_idx[b]].item()) if b in id_to_idx else float("-inf")
                    )
                    selected.discard(removable[0])
                selected.add(neighbor)
                boundary_added += 1
                if len(selected) > num_blocks_to_keep:
                    removable = [
                        b
                        for b in selected
                        if b not in must_keep_block_ids and b not in boundary_selected and b != neighbor
                    ]
                    if removable:
                        removable.sort(
                            key=lambda b: float(block_scores[id_to_idx[b]].item()) if b in id_to_idx else float("-inf")
                        )
                        selected.discard(removable[0])

    if len(selected) > num_blocks_to_keep:
        selected = _trim_to_budget(
            selected,
            must_keep_block_ids=must_keep_block_ids,
            num_blocks_to_keep=budget,
            block_scores=block_scores,
            id_to_idx=id_to_idx,
        )

    anchor_retained = sum(1 for bid in selected if int(profiles.get(int(bid), {}).get("role_persistent_anchor", 0)))
    bridge_retained = sum(1 for bid in selected if int(profiles.get(int(bid), {}).get("role_bridge_relay", 0)))
    hub_selected = sum(1 for bid in selected if int(profiles.get(int(bid), {}).get("role_generic_hub", 0)))

    churn_rate = active_set_churn(silm_selected, selected, budget)
    overlap = len(selected.intersection(silm_selected))
    union = len(selected.union(silm_selected))
    return selected, {
        "selection_policy": policy,
        "active_budget": budget,
        "selected_block_count": len(selected),
        "overlap_with_silm_original": float(overlap / union) if union else 1.0,
        "churn_rate": churn_rate,
        "churn_in_target": int(
            float(cfg.get("roleflow_target_churn_min", 0.05))
            <= churn_rate
            <= float(cfg.get("roleflow_target_churn_max", 0.15))
        ),
        "persistent_anchor_retained_count": anchor_retained,
        "bridge_relay_retained_count": bridge_retained,
        "generic_hub_selected_count": hub_selected,
        "boundary_neighbor_added_count": boundary_added,
        "transient_trigger_late_selected_count": transient_late,
        "over_budget": int(len(selected) > budget),
        **churn_meta,
    }
