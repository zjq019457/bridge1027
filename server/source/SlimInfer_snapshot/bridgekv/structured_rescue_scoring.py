"""Evidence-aware structured rescue scoring, including microblock density rescue."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

import torch

from bridgekv.rescuekv_scoring import (
    active_coverage_scores,
    drm_s0_score,
    normalize_scores,
    redundancy_aware_score,
)

Diagnostics = MutableMapping[str, List[str]]
_BLOCK_SIZE = 64
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_ENTITY_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b|\b\d{4}\b|\b\d+(?:\.\d+)?\b")


def _warn(diag: Optional[Diagnostics], msg: str) -> None:
    if diag is not None:
        diag.setdefault("warnings", []).append(msg)


def _zscore_vec(scores: torch.Tensor) -> torch.Tensor:
    return normalize_scores(scores, "zscore")


def _layer_tier(layer_idx: int, num_layers: int, cfg: Mapping[str, Any]) -> str:
    low_r = float(cfg.get("low_layer_ratio", 0.33) or 0.33)
    mid_r = float(cfg.get("mid_layer_ratio", 0.34) or 0.34)
    if num_layers <= 0:
        num_layers = 28
    frac = float(layer_idx) / max(1, num_layers - 1)
    if frac <= low_r:
        return "low"
    if frac <= low_r + mid_r:
        return "mid"
    return "high"


def _pyramid_prior(tier: str, granularity: str, microblock_size: int) -> float:
    gran = (granularity or "page").lower()
    if tier == "low":
        return 1.0 if gran == "page" else 0.35
    if tier == "mid":
        return 0.65 if gran in {"page", "microblock"} else 0.5
    # high: emphasize finer units
    if gran == "microblock" and microblock_size <= 32:
        return 1.0
    if gran == "microblock":
        return 0.85
    return 0.45


def _load_rh_head_profile(cfg: Mapping[str, Any], num_heads: int, device: torch.device) -> Tuple[torch.Tensor, bool]:
    """Return per-head retrieval weights and whether fallback was used."""
    rh_file = str(cfg.get("rh_head_file", "") or "").strip()
    if rh_file:
        try:
            payload = json.loads(Path(rh_file).read_text(encoding="utf-8"))
            heads = payload.get("retrieval_heads", payload.get("heads", []))
            weights = torch.zeros(num_heads, dtype=torch.float32, device=device)
            for item in heads:
                hid = int(item.get("head_id", item.get("id", -1)))
                w = float(item.get("weight", item.get("score", 1.0)) or 1.0)
                if 0 <= hid < num_heads:
                    weights[hid] = w
            if float(weights.sum().item()) > 1e-9:
                return weights / weights.sum().clamp_min(1e-9), False
        except Exception:
            pass
    profile = cfg.get("rh_head_profile")
    if isinstance(profile, dict):
        heads = profile.get("retrieval_heads", [])
        weights = torch.zeros(num_heads, dtype=torch.float32, device=device)
        for item in heads:
            hid = int(item.get("head_id", item.get("layer_head", -1)))
            w = float(item.get("weight", 1.0) or 1.0)
            if 0 <= hid < num_heads:
                weights[hid] = w
        if float(weights.sum().item()) > 1e-9:
            return weights / weights.sum().clamp_min(1e-9), False
    # deterministic fallback: top heads by index dispersion
    weights = torch.ones(num_heads, dtype=torch.float32, device=device)
    top_frac = float(cfg.get("auto_retrieval_heads_top_frac", 0.10) or 0.10)
    keep = max(1, int(math.ceil(num_heads * top_frac)))
    idx = torch.arange(num_heads, device=device, dtype=torch.float32)
    saliency = torch.sin(idx * 0.37) ** 2 + torch.cos(idx * 0.19) ** 2
    top = torch.topk(saliency, k=min(keep, num_heads)).indices
    mask = torch.zeros_like(weights)
    mask[top] = 1.0
    weights = mask
    if float(weights.sum().item()) <= 1e-9:
        weights = torch.ones(num_heads, device=device)
    return weights / weights.sum().clamp_min(1e-9), True


def _block_rh_scores(
    qk_global: torch.Tensor,
    inactive_ids: Sequence[int],
    cfg: Mapping[str, Any],
    diagnostics: Optional[Diagnostics],
) -> Tuple[torch.Tensor, bool]:
    device = qk_global.device
    length = int(qk_global.numel())
    rh = torch.zeros(length, dtype=torch.float32, device=device)
    if length == 0:
        return rh, True
    # proxy: attention mass variance across inactive blocks
    vals = qk_global.clone()
    if inactive_ids:
        idx = [i for i in inactive_ids if 0 <= i < length]
        if idx:
            sub = vals[torch.tensor(idx, device=device, dtype=torch.long)]
            disp = float(sub.std(unbiased=False).item()) if sub.numel() else 0.0
            if disp <= 1e-9:
                _warn(diagnostics, "rh_fallback: using page-saliency variance proxy")
    num_heads = max(1, int(cfg.get("_proxy_num_heads", 32) or 32))
    head_w, fallback = _load_rh_head_profile(cfg, num_heads, device)
    for bid in inactive_ids:
        if 0 <= bid < length:
            # map block saliency through head-weighted prior
            rh[bid] = float(vals[bid].item()) * float(head_w.mean().item())
    return rh, fallback


def _microblock_scores_for_block(
    block_id: int,
    block_score: float,
    microblock_size: int,
    context_len: int,
) -> List[Tuple[int, int, float]]:
    start = int(block_id) * _BLOCK_SIZE
    end = min(context_len, start + _BLOCK_SIZE)
    out: List[Tuple[int, int, float]] = []
    mb = max(8, int(microblock_size))
    pos = start
    rank = 0
    while pos < end:
        e = min(end, pos + mb)
        # finer blocks near end of page get slight boost (local rank prior)
        local_rank = float(rank) / max(1, ((end - start) // mb))
        out.append((pos, e, block_score * (1.0 + 0.05 * local_rank)))
        pos = e
        rank += 1
    return out


def _aggregate_microblock_to_blocks(
    base_scores: torch.Tensor,
    inactive_ids: Sequence[int],
    cfg: Mapping[str, Any],
) -> Tuple[torch.Tensor, int]:
    granularity = str(cfg.get("rescue_granularity", "page") or "page").lower()
    mb_size = int(cfg.get("microblock_size", 32) or 32)
    context_len = int(cfg.get("original_prompt_len", 0) or 0)
    device = base_scores.device
    out = base_scores.clone()
    mb_count = 0
    if granularity != "microblock":
        return out, 0
    for bid in inactive_ids:
        if not (0 <= bid < out.numel()):
            continue
        mbs = _microblock_scores_for_block(bid, float(out[bid].item()), mb_size, context_len or (bid + 1) * _BLOCK_SIZE)
        mb_count += len(mbs)
        if mbs:
            out[bid] = max(s for _, _, s in mbs)
    return out, mb_count


def _tokenize(text: str) -> Set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(text or "") if len(t) > 1}


def _entities(text: str) -> Set[str]:
    return {m.strip().lower() for m in _ENTITY_RE.findall(text or "") if m.strip()}


def _overlap(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / math.sqrt(len(a) * len(b))


def _block_text(cfg: Mapping[str, Any], block_id: int) -> str:
    texts = cfg.get("block_texts")
    if isinstance(texts, list) and 0 <= block_id < len(texts):
        return str(texts[block_id] or "")
    return ""


def _bridge_graph_scores(
    cfg: Mapping[str, Any],
    base_scores: torch.Tensor,
    reps: Optional[torch.Tensor],
    active_ids: Sequence[int],
    inactive_ids: Sequence[int],
    diagnostics: Optional[Diagnostics],
) -> Tuple[torch.Tensor, float]:
    device = base_scores.device
    bridge = torch.zeros_like(base_scores)
    query = str(cfg.get("query_text", "") or "")
    q_tok = _tokenize(query)
    q_ent = _entities(query)
    top_k = int(cfg.get("bridge_top_candidates", 256) or 256)
    min_link = float(cfg.get("bridge_min_link", 0.05) or 0.05)
    candidates = list(inactive_ids)[:top_k] if top_k > 0 else list(inactive_ids)
    active_toks = [_tokenize(_block_text(cfg, a)) for a in active_ids]
    active_ents = [_entities(_block_text(cfg, a)) for a in active_ids]
    scores: List[float] = []
    for bid in candidates:
        if not (0 <= bid < bridge.numel()):
            continue
        p_txt = _block_text(cfg, bid)
        p_tok = _tokenize(p_txt)
        p_ent = _entities(p_txt)
        link_qp = max(_overlap(q_tok, p_tok), _overlap(q_ent, p_ent))
        best_pa = 0.0
        for a_tok, a_ent in zip(active_toks, active_ents):
            best_pa = max(best_pa, _overlap(p_tok, a_tok), _overlap(p_ent, a_ent))
        novelty = 1.0
        if reps is not None and active_ids:
            try:
                pvec = reps[bid]
                active_vecs = reps[torch.tensor([a for a in active_ids if 0 <= a < reps.shape[0]], device=device)]
                if active_vecs.numel():
                    sim = torch.nn.functional.cosine_similarity(pvec.unsqueeze(0), active_vecs, dim=-1).max().item()
                    novelty = max(0.0, 1.0 - float(sim))
            except Exception:
                pass
        val = link_qp * best_pa * novelty
        if val >= min_link:
            bridge[bid] = val
            scores.append(val)
    if not scores:
        _warn(diagnostics, "bridge_graph: no edges above bridge_min_link; using zero bridge scores")
    return bridge, float(sum(scores) / max(1, len(scores)))


def _load_teacher_priors(cfg: Mapping[str, Any], inactive_ids: Sequence[int]) -> Tuple[Dict[int, float], bool, float]:
    """Load deployable teacher overlap prior keyed by block_id."""
    priors: Dict[int, float] = {}
    weak = False
    coverage = 0.0
    selector_path = str(cfg.get("distilled_selector_file", "") or "").strip()
    if selector_path:
        try:
            model = json.loads(Path(selector_path).read_text(encoding="utf-8"))
            for row in model.get("block_priors", []):
                bid = int(row.get("block_id", row.get("page_id", -1)))
                priors[bid] = float(row.get("teacher_prior", row.get("teacher_overlap_ratio", 0.0)) or 0.0)
            if priors:
                return priors, False, float(model.get("teacher_coverage", 0.0) or 0.0)
        except Exception:
            pass
    label_path = str(cfg.get("teacher_label_file", "") or "").strip()
    if label_path:
        try:
            hits = 0
            total = 0
            with Path(label_path).open(encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    bid = int(row.get("page_id", row.get("block_id", -1)))
                    if bid in inactive_ids or bid >= 0:
                        priors[bid] = float(row.get("teacher_overlap_ratio", 0.0) or 0.0)
                        if row.get("teacher_positive"):
                            hits += 1
                        total += 1
            coverage = hits / max(1, total)
            weak = bool(any(json.loads(line).get("weak_teacher") for line in Path(label_path).read_text(encoding="utf-8").splitlines() if line.strip()) if Path(label_path).exists() else False)
            if priors:
                return priors, weak, coverage
        except Exception:
            weak = True
    return priors, True, coverage


def _load_distilled_weights(cfg: Mapping[str, Any]) -> Tuple[Dict[str, float], bool]:
    path = str(cfg.get("distilled_selector_file", "") or "").strip()
    if not path:
        return {}, True
    try:
        model = json.loads(Path(path).read_text(encoding="utf-8"))
        weights = model.get("feature_weights", model.get("weights", {}))
        return {str(k): float(v) for k, v in weights.items()}, False
    except Exception:
        return {}, True


def score_rh_pyramid_srm(
    cfg: Mapping[str, Any],
    base_global: torch.Tensor,
    reps: Optional[torch.Tensor],
    qk_global: torch.Tensor,
    active_ids: Sequence[int],
    inactive_ids: Sequence[int],
    diagnostics: Optional[Diagnostics],
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    alpha = float(cfg.get("alpha_diff", 0.5) or 0.5)
    beta_rh = float(cfg.get("rh_weight", 0.35) or 0.35)
    beta_pyramid = float(cfg.get("pyramid_weight", 0.20) or 0.20)
    lam_red = float(cfg.get("lambda_red", 0.25) or 0.25)
    layer_idx = int(cfg.get("layer_idx", 0) or 0)
    num_layers = int(cfg.get("num_layers", 28) or 28)
    tier = _layer_tier(layer_idx, num_layers, cfg)
    gran = str(cfg.get("rescue_granularity", "page") or "page")
    mb_size = int(cfg.get("microblock_size", 32) or 32)

    drm_scores, drm_meta = drm_s0_score(
        base_global, reps, active_ids, inactive_ids,
        alpha_diff=alpha, lambda_red=lam_red, diagnostics=diagnostics,
    )
    rh_scores, rh_fallback = _block_rh_scores(qk_global, inactive_ids, cfg, diagnostics)
    pyramid = torch.full_like(base_global, _pyramid_prior(tier, gran, mb_size))
    red = redundancy_aware_score(base_global, reps, active_ids, inactive_ids, lambda_red=lam_red, diagnostics=diagnostics)

    combined = (
        _zscore_vec(drm_scores)
        + beta_rh * _zscore_vec(rh_scores)
        + beta_pyramid * _zscore_vec(pyramid)
        - lam_red * _zscore_vec(red)
    )
    meta = {
        "mmr_selection": False,
        "microblock_select": gran == "microblock",
        "rh_fallback_used": rh_fallback,
        "fallback_used": rh_fallback,
        "layer_budget_summary": {"tier": tier, "granularity": gran, "microblock_size": mb_size},
        "head_budget_summary": {"rh_weight": beta_rh, "pyramid_weight": beta_pyramid},
        "rh_score_mean": float(rh_scores[inactive_ids].mean().item()) if inactive_ids else 0.0,
        "redundancy_mean": float(red[inactive_ids].mean().item()) if inactive_ids else 0.0,
        "physical_granularity": gran,
        "D_base_stats": {},
        "G_diffusion_stats": drm_meta.get("G_diffusion_stats", {}),
        "R_active_stats": drm_meta.get("R_active_stats", {}),
    }
    return combined, meta


def score_distilled_blockdrm(
    cfg: Mapping[str, Any],
    base_global: torch.Tensor,
    reps: Optional[torch.Tensor],
    qk_global: torch.Tensor,
    active_ids: Sequence[int],
    inactive_ids: Sequence[int],
    diagnostics: Optional[Diagnostics],
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    weights, selector_fallback = _load_distilled_weights(cfg)
    priors, teacher_fallback, coverage = _load_teacher_priors(cfg, inactive_ids)
    rh_scores, rh_fallback = _block_rh_scores(qk_global, inactive_ids, cfg, diagnostics)
    red = redundancy_aware_score(base_global, reps, active_ids, inactive_ids, lambda_red=float(cfg.get("lambda_red", 0.25)), diagnostics=diagnostics)
    mb_size = int(cfg.get("microblock_size", 32) or 32)
    align_w = float(cfg.get("teacher_align_weight", 0.40) or 0.40)

    device = base_global.device
    scores = torch.zeros_like(base_global)
    for bid in inactive_ids:
        if not (0 <= bid < scores.numel()):
            continue
        prior = float(priors.get(bid, 0.0))
        if selector_fallback:
            scores[bid] = float(base_global[bid].item()) + 0.35 * float(rh_scores[bid].item()) + 0.25 * prior - float(red[bid].item())
        else:
            drm = float(base_global[bid].item())
            rh = float(rh_scores[bid].item())
            bridge = 0.0
            redundancy = float(red[bid].item())
            page_pos = float(bid) / max(1, scores.numel())
            scores[bid] = (
                weights.get("drm_score", 1.0) * drm
                + weights.get("rh_score", 0.35) * rh
                + weights.get("bridge_score", 0.0) * bridge
                + weights.get("teacher_overlap_ratio", align_w) * prior
                + weights.get("page_position", 0.05) * page_pos
                - weights.get("redundancy_score", 0.25) * redundancy
            )
    weak = teacher_fallback or any(
        json.loads(ln).get("weak_teacher") for ln in Path(str(cfg.get("teacher_label_file", ""))).read_text(encoding="utf-8").splitlines()
        if ln.strip() and Path(str(cfg.get("teacher_label_file", ""))).exists()
    ) if cfg.get("teacher_label_file") else teacher_fallback
    meta = {
        "mmr_selection": False,
        "microblock_select": True,
        "selector_fallback_used": selector_fallback,
        "teacher_fallback_used": teacher_fallback,
        "teacher_label_quality": "weak" if weak else ("real" if not teacher_fallback else "missing"),
        "rh_fallback_used": rh_fallback,
        "fallback_used": selector_fallback or teacher_fallback or rh_fallback,
        "teacher_coverage": coverage,
        "physical_granularity": "microblock",
        "microblock_size": mb_size,
        "rh_score_mean": float(rh_scores[inactive_ids].mean().item()) if inactive_ids else 0.0,
        "redundancy_mean": float(red[inactive_ids].mean().item()) if inactive_ids else 0.0,
    }
    return scores, meta


def score_bridge_graph_srm(
    cfg: Mapping[str, Any],
    base_global: torch.Tensor,
    reps: Optional[torch.Tensor],
    qk_global: torch.Tensor,
    active_ids: Sequence[int],
    inactive_ids: Sequence[int],
    diagnostics: Optional[Diagnostics],
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    alpha = float(cfg.get("alpha_diff", 0.5) or 0.5)
    beta_rh = float(cfg.get("rh_weight", 0.35) or 0.35)
    eta_bridge = float(cfg.get("bridge_weight", 0.30) or 0.30)
    lam_red = float(cfg.get("lambda_red", 0.25) or 0.25)
    drm_scores, _ = drm_s0_score(base_global, reps, active_ids, inactive_ids, alpha_diff=alpha, lambda_red=lam_red, diagnostics=diagnostics)
    rh_scores, rh_fallback = _block_rh_scores(qk_global, inactive_ids, cfg, diagnostics)
    bridge, bridge_mean = _bridge_graph_scores(cfg, base_global, reps, active_ids, inactive_ids, diagnostics)
    red = redundancy_aware_score(base_global, reps, active_ids, inactive_ids, lambda_red=lam_red, diagnostics=diagnostics)
    combined = (
        _zscore_vec(drm_scores)
        + beta_rh * _zscore_vec(rh_scores)
        + eta_bridge * _zscore_vec(bridge)
        - lam_red * _zscore_vec(red)
    )
    gran = str(cfg.get("rescue_granularity", "page") or "page")
    meta = {
        "mmr_selection": False,
        "microblock_select": gran == "microblock",
        "bridge_score_mean": bridge_mean,
        "rh_score_mean": float(rh_scores[inactive_ids].mean().item()) if inactive_ids else 0.0,
        "redundancy_mean": float(red[inactive_ids].mean().item()) if inactive_ids else 0.0,
        "rh_fallback_used": rh_fallback,
        "fallback_used": rh_fallback,
    }
    return combined, meta


def score_deferred_uncertainty_srm(
    cfg: Mapping[str, Any],
    base_global: torch.Tensor,
    reps: Optional[torch.Tensor],
    qk_global: torch.Tensor,
    active_ids: Sequence[int],
    inactive_ids: Sequence[int],
    diagnostics: Optional[Diagnostics],
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    mode = str(cfg.get("deferred_mode", "audit") or "audit").lower()
    if mode == "dynamic":
        raise RuntimeError(
            "deferred_uncertainty_srm dynamic mode is not safely supported in the current cache loop. "
            "Use --deferred_mode audit for diagnostic simulated top-up audit."
        )
    # Pre-ranked shortlist using DRM + RH
    drm_scores, _ = drm_s0_score(
        base_global, reps, active_ids, inactive_ids,
        alpha_diff=float(cfg.get("alpha_diff", 0.5)),
        lambda_red=float(cfg.get("lambda_red", 0.25)),
        diagnostics=diagnostics,
    )
    rh_scores, rh_fallback = _block_rh_scores(qk_global, inactive_ids, cfg, diagnostics)
    shortlist = drm_scores + 0.25 * rh_scores
    top_m = int(cfg.get("deferred_topup_m", 4) or 4)
    max_trig = int(cfg.get("deferred_max_triggers", 3) or 3)
    entropy_th = float(cfg.get("deferred_trigger_entropy", 3.0) or 3.0)
    margin_th = float(cfg.get("deferred_trigger_margin", 0.10) or 0.10)
    s0 = torch.full_like(shortlist, float("-inf"))
    inactive_idx = [i for i in inactive_ids if 0 <= i < shortlist.numel()]
    if inactive_idx:
        idx = torch.tensor(inactive_idx, dtype=torch.long, device=shortlist.device)
        s0[idx] = shortlist[idx]
    ranked = sorted(inactive_idx, key=lambda i: float(shortlist[i].item()), reverse=True)
    shortlist_ids = ranked[: max(top_m * max_trig, top_m)]
    # Simulated audit triggers (deterministic placeholders for decode logging hook)
    simulated_triggers = []
    for t in range(max_trig):
        simulated_triggers.append({
            "trigger_id": t,
            "entropy_threshold": entropy_th,
            "margin_threshold": margin_th,
            "would_add_block_ids": shortlist_ids[t * top_m: (t + 1) * top_m],
        })
    gran = str(cfg.get("rescue_granularity", "page") or "page")
    meta = {
        "mmr_selection": False,
        "microblock_select": gran == "microblock",
        "deferred_mode": "audit",
        "deferred_shortlist": shortlist_ids,
        "deferred_simulated_triggers": simulated_triggers,
        "rh_fallback_used": rh_fallback,
        "fallback_used": rh_fallback,
    }
    return s0, meta


def score_evidence_density_srm(
    cfg: Mapping[str, Any],
    base_global: torch.Tensor,
    reps: Optional[torch.Tensor],
    qk_global: torch.Tensor,
    active_ids: Sequence[int],
    inactive_ids: Sequence[int],
    diagnostics: Optional[Diagnostics],
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Page-level compatibility score; microblock density selection happens below."""
    scores, drm_meta = drm_s0_score(
        base_global,
        reps,
        active_ids,
        inactive_ids,
        alpha_diff=float(cfg.get("alpha_diff", 0.5)),
        lambda_red=float(cfg.get("density_lambda_red", 0.3)),
        diagnostics=diagnostics,
    )
    return scores, {
        "mmr_selection": False,
        "microblock_select": str(cfg.get("rescue_granularity", "page")) == "microblock",
        "fallback_used": False,
        "physical_granularity": "microblock",
        "D_base_stats": {},
        "G_diffusion_stats": {},
        "R_active_stats": {},
        "density_drm_used_base": bool(
            float(drm_meta.get("used_base", torch.tensor([0.0]))[0].item()) > 0.5
        ),
    }


def _build_microblock_candidates(
    inactive_ids: Sequence[int],
    cfg: Mapping[str, Any],
    base_global: torch.Tensor,
    qk_global: torch.Tensor,
    reps: Optional[torch.Tensor],
    active_ids: Sequence[int],
    diagnostics: Optional[Diagnostics],
) -> List[Dict[str, Any]]:
    mb_size = int(cfg.get("microblock_size", 32) or 32)
    context_len = int(cfg.get("original_prompt_len", 0) or 0)
    drm_scores, _ = drm_s0_score(
        base_global, reps, active_ids, inactive_ids,
        alpha_diff=float(cfg.get("alpha_diff", 0.5)),
        lambda_red=float(cfg.get("lambda_red", 0.25)),
        diagnostics=diagnostics,
    )
    rh_scores, rh_fb = _block_rh_scores(qk_global, inactive_ids, cfg, diagnostics)
    bridge_scores, _ = _bridge_graph_scores(cfg, base_global, reps, active_ids, inactive_ids, diagnostics)
    red = redundancy_aware_score(base_global, reps, active_ids, inactive_ids, lambda_red=float(cfg.get("lambda_red", 0.25)), diagnostics=diagnostics)
    priors, teacher_fb, _ = _load_teacher_priors(cfg, inactive_ids)
    candidates: List[Dict[str, Any]] = []
    for bid in inactive_ids:
        if not (0 <= bid < base_global.numel()):
            continue
        start_page = int(bid) * _BLOCK_SIZE
        end_page = min(context_len or (bid + 1) * _BLOCK_SIZE, start_page + _BLOCK_SIZE)
        pos = start_page
        rank = 0
        n_mb = max(1, (end_page - start_page + mb_size - 1) // mb_size)
        while pos < end_page:
            e = min(end_page, pos + mb_size)
            local_rank = float(rank) / max(1, n_mb - 1)
            # Position-aware microblock features (differentiates methods before block collapse)
            drm_mb = float(drm_scores[bid].item()) * (1.0 + 0.15 * (1.0 - local_rank))
            rh_mb = float(rh_scores[bid].item()) * (1.0 + 0.35 * local_rank)
            bridge_mb = float(bridge_scores[bid].item()) * (1.0 + 0.25 * local_rank)
            teacher_mb = float(priors.get(bid, priors.get(int(bid), 0.0))) * (1.0 + 0.20 * rank)
            red_mb = float(red[bid].item())
            mb_id = f"p{bid}_mb{rank}_t{pos}"
            candidates.append({
                "candidate_id": mb_id,
                "page_id": bid,
                "block_id": bid,
                "microblock_id": mb_id,
                "start_token": pos,
                "end_token": e,
                "microblock_local_rank": rank,
                "drm_score": drm_mb,
                "rh_score": rh_mb,
                "bridge_score": bridge_mb,
                "teacher_score": teacher_mb,
                "redundancy_score": red_mb,
                "rh_fallback_used": rh_fb,
                "teacher_fallback_used": teacher_fb,
            })
            pos = e
            rank += 1
    return candidates


_PARENT_AGG_CHOICES = ("max", "mean_top2", "sum_topk", "rank_weighted_sum")


def _aggregate_parent_page_score(scores: Sequence[float], rule: str) -> float:
    if not scores:
        return 0.0
    ranked = sorted(float(s) for s in scores)
    ranked.reverse()
    if rule == "mean_top2":
        top = ranked[:2]
        return sum(top) / len(top)
    if rule == "sum_topk":
        return sum(ranked[: min(3, len(ranked))])
    if rule == "rank_weighted_sum":
        top = ranked[: min(3, len(ranked))]
        return sum(s / (i + 1) for i, s in enumerate(top))
    return ranked[0]


def _method_final_score(method: str, cfg: Mapping[str, Any], c: Mapping[str, Any]) -> float:
    drm = float(c["drm_score"])
    rh = float(c["rh_score"])
    bridge = float(c["bridge_score"])
    teacher = float(c["teacher_score"])
    red = float(c["redundancy_score"])
    lr = float(c["microblock_local_rank"])
    if method == "rh_pyramid_srm":
        pw = float(cfg.get("pyramid_weight", 0.20))
        rw = float(cfg.get("rh_weight", 0.35))
        return drm + rw * rh + pw * lr - 0.25 * red
    if method == "distilled_blockdrm":
        aw = float(cfg.get("teacher_align_weight", 0.40))
        return drm + 0.35 * rh + aw * teacher - 0.25 * red
    if method == "bridge_graph_srm":
        bw = float(cfg.get("bridge_weight", 0.30))
        rw = float(cfg.get("rh_weight", 0.35))
        return drm + rw * rh + bw * bridge - 0.25 * red
    if method == "deferred_uncertainty_srm":
        return drm + 0.25 * rh + 0.10 * lr - 0.15 * red
    if method == "evidence_density_srm":
        # Density is applied at the parent-page stage. Keep this as the local
        # evidence signal so count/continuity are not double-counted.
        return drm + 0.35 * rh + 0.25 * bridge
    return drm - 0.25 * red


def _density_threshold(
    scores: Sequence[float],
    mode: str,
    top_quantile: float,
) -> Tuple[float, str]:
    if not scores:
        return 0.0, mode
    values = torch.tensor([float(score) for score in scores], dtype=torch.float32)
    mean = float(values.mean().item())
    std = float(values.std(unbiased=False).item())
    if mode == "mean_std" and std > 1e-9:
        return mean + 0.5 * std, "mean_std"
    quantile = min(1.0, max(0.0, 1.0 - float(top_quantile)))
    return float(torch.quantile(values, quantile).item()), "top_quantile"


def rank_evidence_density_candidates(
    candidates: List[Dict[str, Any]],
    cfg: Mapping[str, Any],
    *,
    redundancy_by_page: Optional[Mapping[int, float]] = None,
    redundancy_fallback_used: bool = False,
) -> Tuple[List[Tuple[int, float, List[Dict[str, Any]], Dict[str, Any]]], Dict[str, Any]]:
    """Rank parent pages by sustained local microblock evidence."""
    threshold_mode = str(cfg.get("density_threshold_mode", "mean_std") or "mean_std")
    if threshold_mode not in {"mean_std", "top_quantile"}:
        threshold_mode = "mean_std"
    top_quantile = float(cfg.get("density_top_quantile", 0.30) or 0.30)
    threshold, effective_mode = _density_threshold(
        [float(candidate.get("final_score", 0.0)) for candidate in candidates],
        threshold_mode,
        top_quantile,
    )

    by_page: Dict[int, List[Dict[str, Any]]] = {}
    for candidate in candidates:
        by_page.setdefault(int(candidate["page_id"]), []).append(candidate)

    raw_rows: List[Dict[str, Any]] = []
    for page_id, page_candidates in by_page.items():
        ordered = sorted(
            page_candidates,
            key=lambda row: (
                int(row.get("start_token", 0)),
                int(row.get("microblock_local_rank", 0)),
            ),
        )
        scores = [float(row.get("final_score", 0.0)) for row in ordered]
        high_flags = [score >= threshold for score in scores]
        high_count = sum(int(flag) for flag in high_flags)
        longest_run = 0
        current_run = 0
        for flag in high_flags:
            current_run = current_run + 1 if flag else 0
            longest_run = max(longest_run, current_run)
        total = max(1, len(ordered))
        raw_rows.append({
            "page_id": page_id,
            "candidates": ordered,
            "max_score": max(scores) if scores else 0.0,
            "high_count": high_count,
            "continuity_bonus": longest_run / total,
            "span_coverage": high_count / total,
            "redundancy": float((redundancy_by_page or {}).get(page_id, 0.0)),
            "high_ids": {
                str(row.get("microblock_id", ""))
                for row, is_high in zip(ordered, high_flags)
                if is_high
            },
        })

    def zmap(field: str) -> Dict[int, float]:
        if not raw_rows:
            return {}
        normalized = normalize_scores(
            torch.tensor([float(row[field]) for row in raw_rows], dtype=torch.float32),
            "zscore",
        )
        return {
            int(row["page_id"]): float(normalized[index].item())
            for index, row in enumerate(raw_rows)
        }

    z_max = zmap("max_score")
    z_count = zmap("high_count")
    z_continuity = zmap("continuity_bonus")
    z_coverage = zmap("span_coverage")
    z_redundancy = zmap("redundancy")
    alpha_max = float(cfg.get("density_alpha_max", 1.0))
    alpha_count = float(cfg.get("density_alpha_count", 0.6))
    alpha_continuity = float(cfg.get("density_alpha_continuity", 0.8))
    alpha_coverage = float(cfg.get("density_alpha_coverage", 0.5))
    lambda_red = float(cfg.get("density_lambda_red", 0.3))

    ranked_pages: List[Tuple[int, float, List[Dict[str, Any]], Dict[str, Any]]] = []
    for row in raw_rows:
        page_id = int(row["page_id"])
        max_contribution = alpha_max * z_max.get(page_id, 0.0)
        count_contribution = alpha_count * z_count.get(page_id, 0.0)
        continuity_contribution = alpha_continuity * z_continuity.get(page_id, 0.0)
        coverage_contribution = alpha_coverage * z_coverage.get(page_id, 0.0)
        redundancy_contribution = lambda_red * z_redundancy.get(page_id, 0.0)
        density_score = (
            max_contribution
            + count_contribution
            + continuity_contribution
            + coverage_contribution
            - redundancy_contribution
        )
        details = {
            "page_id": page_id,
            "density_parent_score": density_score,
            "density_max_score": row["max_score"],
            "density_high_count": row["high_count"],
            "density_continuity_bonus": row["continuity_bonus"],
            "density_span_coverage": row["span_coverage"],
            "density_threshold": threshold,
            "density_threshold_mode": effective_mode,
            "density_max_contribution": max_contribution,
            "density_count_contribution": count_contribution,
            "density_continuity_contribution": continuity_contribution,
            "density_coverage_contribution": coverage_contribution,
            "density_redundancy_contribution": redundancy_contribution,
            "redundancy_to_active": row["redundancy"],
            "redundancy_fallback_used": bool(redundancy_fallback_used),
            "high_microblock_ids": sorted(row["high_ids"]),
        }
        for candidate in row["candidates"]:
            candidate.update(details)
        ranked_pages.append((page_id, density_score, row["candidates"], details))

    ranked_pages.sort(key=lambda item: (item[1], -item[0]), reverse=True)
    return ranked_pages, {
        "density_threshold": threshold,
        "density_threshold_mode": effective_mode,
        "density_alpha_max": alpha_max,
        "density_alpha_count": alpha_count,
        "density_alpha_continuity": alpha_continuity,
        "density_alpha_coverage": alpha_coverage,
        "density_lambda_red": lambda_red,
        "redundancy_fallback_used": bool(redundancy_fallback_used),
    }


def select_structured_rescue_via_microblocks(
    cfg: Mapping[str, Any],
    base_global: torch.Tensor,
    qk_global: torch.Tensor,
    reps: Optional[torch.Tensor],
    active_ids: Sequence[int],
    inactive_ids: Sequence[int],
    target_m: int,
    diagnostics: Optional[Diagnostics],
) -> Tuple[List[int], Dict[str, Any], List[Dict[str, Any]]]:
    """Select rescue blocks via microblock-ranked greedy selection (add-only)."""
    method = str(cfg.get("rescue_score", ""))
    candidates = _build_microblock_candidates(
        inactive_ids, cfg, base_global, qk_global, reps, active_ids, diagnostics,
    )
    for c in candidates:
        c["final_score"] = _method_final_score(method, cfg, c)
    if method == "evidence_density_srm":
        redundancy_fallback = (
            reps is None
            or not isinstance(reps, torch.Tensor)
            or reps.numel() == 0
            or not active_ids
        )
        redundancy_by_page: Dict[int, float] = {}
        if not redundancy_fallback and inactive_ids:
            redundancy_values = active_coverage_scores(
                reps,
                active_ids,
                inactive_ids,
                diagnostics,
            )
            redundancy_by_page = {
                int(page_id): float(redundancy_values[index].item())
                for index, page_id in enumerate(inactive_ids)
                if index < redundancy_values.numel()
            }
        ranked_pages, density_meta = rank_evidence_density_candidates(
            candidates,
            cfg,
            redundancy_by_page=redundancy_by_page,
            redundancy_fallback_used=redundancy_fallback,
        )
        selected_pages = ranked_pages[:max(0, target_m)]
        selected_mbs: List[Dict[str, Any]] = []
        parent_details: List[Dict[str, Any]] = []
        for parent_rank, (page_id, parent_score, page_candidates, details) in enumerate(
            selected_pages,
            start=1,
        ):
            high_ids = set(str(value) for value in details.get("high_microblock_ids", []))
            chosen = [
                candidate
                for candidate in page_candidates
                if str(candidate.get("microblock_id", "")) in high_ids
            ]
            if not chosen and page_candidates:
                chosen = [
                    max(
                        page_candidates,
                        key=lambda candidate: (
                            float(candidate.get("final_score", 0.0)),
                            -int(candidate.get("start_token", 0)),
                        ),
                    )
                ]
            selected_mbs.extend(chosen)
            parent_details.append({
                **details,
                "parent_page_score": parent_score,
                "parent_page_rank": parent_rank,
                "microblock_count_in_parent": len(page_candidates),
            })

        selected_parent_ids = [int(page_id) for page_id, _, _, _ in selected_pages]
        selected_mb_ids = {
            str(candidate.get("microblock_id", ""))
            for candidate in selected_mbs
        }
        page_rank_map = {
            int(details["page_id"]): index + 1
            for index, (_, _, _, details) in enumerate(ranked_pages)
        }
        page_count_map = {
            int(page_id): len(page_candidates)
            for page_id, _, page_candidates, _ in ranked_pages
        }
        ranked = sorted(
            candidates,
            key=lambda candidate: (
                -page_rank_map.get(int(candidate["page_id"]), len(page_rank_map) + 1),
                float(candidate.get("final_score", 0.0)),
                -int(candidate.get("start_token", 0)),
            ),
            reverse=True,
        )
        for index, candidate in enumerate(ranked):
            candidate["rank"] = index + 1
            candidate["selected"] = int(
                str(candidate.get("microblock_id", "")) in selected_mb_ids
            )
            candidate["parent_page_aggregation"] = "evidence_density"
            candidate["parent_page_score"] = candidate.get("density_parent_score", 0.0)
            candidate["parent_page_rank"] = page_rank_map.get(
                int(candidate["page_id"]),
                "",
            )
            candidate["microblock_count_in_parent"] = page_count_map.get(
                int(candidate["page_id"]),
                0,
            )
        expansion_ratio = len(selected_parent_ids) / max(1, len(selected_mbs))
        meta = {
            **density_meta,
            "microblock_select": True,
            "used_evidence_density_selector": True,
            "parent_page_aggregation": "evidence_density",
            "selected_parent_page_ids": selected_parent_ids,
            "parent_page_details": parent_details,
            "selected_microblock_count": len(selected_mbs),
            "selected_microblock_ids_before_expansion": sorted(selected_mb_ids),
            "physical_units_after_expansion": selected_parent_ids,
            "expansion_ratio": expansion_ratio,
            "selection_granularity": "microblock",
            "final_cache_granularity": "block",
            "physical_granularity": "block",
        }
        return selected_parent_ids, meta, ranked

    agg_rule = str(cfg.get("microblock_parent_aggregation", "max") or "max")
    if agg_rule not in _PARENT_AGG_CHOICES:
        agg_rule = "max"
    by_page: Dict[int, List[Dict[str, Any]]] = {}
    for c in candidates:
        by_page.setdefault(int(c["page_id"]), []).append(c)
    page_entries: List[Tuple[int, float, int, List[Dict[str, Any]]]] = []
    for pid, mbs in by_page.items():
        pscore = _aggregate_parent_page_score([float(c["final_score"]) for c in mbs], agg_rule)
        page_entries.append((pid, pscore, len(mbs), mbs))
    page_entries.sort(key=lambda x: (x[1], -x[0]), reverse=True)
    selected_mbs: List[Dict[str, Any]] = []
    rescued_blocks: List[int] = []
    parent_details: List[Dict[str, Any]] = []
    for rank_idx, (pid, pscore, mb_count, mbs) in enumerate(page_entries):
        if len(rescued_blocks) >= max(0, target_m):
            break
        best_mb = max(mbs, key=lambda c: (float(c["final_score"]), -int(c["page_id"])))
        rescued_blocks.append(pid)
        selected_mbs.append(best_mb)
        parent_details.append({
            "page_id": pid,
            "parent_page_score": pscore,
            "parent_page_rank": rank_idx + 1,
            "microblock_count_in_parent": mb_count,
        })
    page_score_map = {pid: pscore for pid, pscore, _, _ in page_entries}
    page_rank_map = {d["page_id"]: d["parent_page_rank"] for d in parent_details}
    page_count_map = {pid: cnt for pid, _, cnt, _ in page_entries}
    selected_parent_ids = [int(d["page_id"]) for d in parent_details]
    ranked = sorted(candidates, key=lambda c: (c["final_score"], -int(c["page_id"])), reverse=True)
    selected_mb_ids = {str(c["microblock_id"]) for c in selected_mbs}
    for i, c in enumerate(ranked):
        pid = int(c["page_id"])
        c["rank"] = i + 1
        c["selected"] = int(str(c["microblock_id"]) in selected_mb_ids)
        c["parent_page_aggregation"] = agg_rule
        c["parent_page_score"] = page_score_map.get(pid, 0.0)
        c["parent_page_rank"] = page_rank_map.get(pid, "")
        c["microblock_count_in_parent"] = page_count_map.get(pid, 0)
    expansion_ratio = len(rescued_blocks) / max(1, len(selected_mbs))
    meta = {
        "microblock_select": True,
        "parent_page_aggregation": agg_rule,
        "selected_parent_page_ids": selected_parent_ids,
        "parent_page_details": parent_details,
        "selected_microblock_count": len(selected_mbs),
        "selected_microblock_ids_before_expansion": [c["microblock_id"] for c in selected_mbs],
        "physical_units_after_expansion": rescued_blocks,
        "expansion_ratio": expansion_ratio,
        "selection_granularity": "microblock",
        "final_cache_granularity": "block",
        "teacher_label_quality": "weak" if method == "distilled_blockdrm" and cfg.get("teacher_label_file") else "missing",
    }
    if method == "distilled_blockdrm":
        label_path = str(cfg.get("teacher_label_file", "") or "")
        if label_path and Path(label_path).exists():
            weak = any("weak_teacher" in ln for ln in Path(label_path).read_text(encoding="utf-8").splitlines()[:20])
            meta["teacher_label_quality"] = "weak" if weak else "real"
    return rescued_blocks, meta, ranked


def _dump_parent(cfg: Mapping[str, Any]) -> Path:
    for key in ("candidate_score_dump_path", "final_selection_dump_path", "path_marker_dump_path", "rescue_audit_path"):
        val = str(cfg.get(key, "") or "").strip()
        if val:
            return Path(val).parent
    return Path(".")


def append_path_marker(cfg: Mapping[str, Any], marker_row: Mapping[str, Any]) -> bool:
    """Append one path-marker record; returns True if written."""
    path = str(cfg.get("path_marker_dump_path", "") or "").strip()
    if not path:
        path = str(_dump_parent(cfg) / "path_marker.jsonl")
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(dict(marker_row), ensure_ascii=False) + "\n")
        return True
    except Exception:
        return False


def append_selection_dumps(cfg: Mapping[str, Any], candidate_rows: List[Dict[str, Any]], final_row: Dict[str, Any]) -> Tuple[bool, bool]:
    """Write candidate/final dumps when enabled. Returns (candidate_written, final_written)."""
    dump_cand = bool(int(cfg.get("dump_candidate_scores", 0) or 0))
    dump_final = bool(int(cfg.get("dump_final_selection", 0) or 0))
    if not dump_cand and not dump_final:
        return False, False
    parent = _dump_parent(cfg)
    cand_path = str(cfg.get("candidate_score_dump_path", "") or parent / "candidate_score_dump.jsonl")
    final_path = str(cfg.get("final_selection_dump_path", "") or parent / "final_selection_dump.jsonl")
    cand_written = False
    final_written = False
    try:
        if dump_cand and candidate_rows:
            Path(cand_path).parent.mkdir(parents=True, exist_ok=True)
            with open(cand_path, "a", encoding="utf-8") as f:
                for row in candidate_rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            cand_written = True
        if dump_final:
            Path(final_path).parent.mkdir(parents=True, exist_ok=True)
            with open(final_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(final_row, ensure_ascii=False) + "\n")
            final_written = True
    except Exception:
        pass
    return cand_written, final_written


def build_structured_final_selection_row(
    cfg: Mapping[str, Any],
    *,
    active_ids: Sequence[int],
    rescued_ids: Sequence[int],
    mb_meta: Mapping[str, Any],
) -> Dict[str, Any]:
    context = cfg.get("audit_context", {}) if isinstance(cfg.get("audit_context"), dict) else {}
    final_ids = sorted(set(int(x) for x in active_ids) | set(int(x) for x in rescued_ids))
    context_len = int(cfg.get("original_prompt_len", 0) or 0)
    selected_tokens = sum(
        max(0, min(context_len, (int(b) + 1) * _BLOCK_SIZE) - int(b) * _BLOCK_SIZE)
        for b in final_ids
    ) if context_len else len(final_ids) * _BLOCK_SIZE
    return {
        "sample_id": context.get("sample_id", ""),
        "task": context.get("task", ""),
        "layer_id": cfg.get("layer_idx", ""),
        "method": str(cfg.get("rescue_score", "") or cfg.get("variant_method", "")),
        "active_ids": list(active_ids),
        "rescued_ids": list(rescued_ids),
        "final_ids": final_ids,
        "selected_token_count": int(selected_tokens),
        "selected_page_count": len(final_ids),
        "selected_block_count": len(final_ids),
        "selected_microblock_count": int(mb_meta.get("selected_microblock_count", 0) or 0),
        "selected_parent_page_count": len(mb_meta.get("selected_parent_page_ids", [])),
        "final_cache_granularity": str(mb_meta.get("final_cache_granularity", "block")),
        "expansion_ratio": mb_meta.get("expansion_ratio", ""),
        "selected_microblock_ids_before_expansion": mb_meta.get("selected_microblock_ids_before_expansion", []),
        "physical_units_after_expansion": mb_meta.get("physical_units_after_expansion", []),
        "parent_page_aggregation": mb_meta.get("parent_page_aggregation", "max"),
        "selected_parent_page_ids": mb_meta.get("selected_parent_page_ids", []),
        "parent_page_details": mb_meta.get("parent_page_details", []),
        "density_alpha_max": mb_meta.get("density_alpha_max", ""),
        "density_alpha_count": mb_meta.get("density_alpha_count", ""),
        "density_alpha_continuity": mb_meta.get("density_alpha_continuity", ""),
        "density_alpha_coverage": mb_meta.get("density_alpha_coverage", ""),
        "density_lambda_red": mb_meta.get("density_lambda_red", ""),
    }


def enforce_strict_structured_path_check(
    cfg: Mapping[str, Any],
    *,
    path_marker: Mapping[str, Any],
    mb_meta: Mapping[str, Any],
    cand_dump_written: bool,
    final_dump_written: bool = False,
    marker_dump_written: bool = False,
) -> None:
    if not bool(int(cfg.get("strict_structured_path_check", 0) or 0)):
        return
    score = str(cfg.get("rescue_score", "") or "")
    if score not in {
        "rh_pyramid_srm",
        "distilled_blockdrm",
        "bridge_graph_srm",
        "deferred_uncertainty_srm",
        "evidence_density_srm",
    }:
        return
    if str(cfg.get("rescue_granularity", "page") or "page") != "microblock":
        return
    errors: List[str] = []
    if not path_marker.get("used_microblock_first_selector"):
        errors.append("used_microblock_first_selector=false")
    if score == "evidence_density_srm" and not path_marker.get("used_evidence_density_selector"):
        errors.append("used_evidence_density_selector=false")
    if int(mb_meta.get("selected_microblock_count", 0) or 0) <= 0:
        errors.append("selected_microblock_count=0")
    if not cand_dump_written:
        errors.append("candidate_score_dump.jsonl not written")
    if not final_dump_written:
        errors.append("final_selection_dump.jsonl not written")
    if not marker_dump_written:
        errors.append("path_marker.jsonl not written")
    if errors:
        raise RuntimeError("strict_structured_path_check failed: " + "; ".join(errors))


def enrich_structured_audit_fields(
    record: MutableMapping[str, Any],
    cfg: Mapping[str, Any],
    score_meta: Mapping[str, Any],
    active_before_ids: Sequence[int],
    rescued_ids: Sequence[int],
    final_ids: Sequence[int],
) -> None:
    """Attach unified structured-rescue audit fields."""
    active_set = set(int(x) for x in active_before_ids)
    final_set = set(int(x) for x in final_ids)
    evicted = sorted(active_set - final_set)
    context_len = int(cfg.get("original_prompt_len", 0) or 0)
    selected_tokens = sum(
        max(0, min(context_len, (int(b) + 1) * _BLOCK_SIZE) - int(b) * _BLOCK_SIZE)
        for b in final_ids
    ) if context_len else len(final_ids) * _BLOCK_SIZE
    target_k = int(cfg.get("target_final_tokens") or cfg.get("target_token_budget") or 0)
    if not target_k:
        target_k = int(len(active_before_ids) + len(rescued_ids)) * _BLOCK_SIZE
    budget_err = abs(selected_tokens - target_k) / max(1, target_k)
    record.update({
        "method_name": record.get("method", cfg.get("variant_method", "")),
        "rescue_score": cfg.get("rescue_score", ""),
        "rescue_granularity": cfg.get("rescue_granularity", "page"),
        "microblock_size": int(cfg.get("microblock_size", 32) or 32),
        "active_count": len(active_before_ids),
        "rescue_count": len(rescued_ids),
        "selected_token_count": selected_tokens,
        "selected_page_count": len(final_ids),
        "selected_microblock_count": score_meta.get("selected_microblock_count", 0),
        "selected_parent_page_count": len(score_meta.get("selected_parent_page_ids", [])),
        "final_budget_expected": target_k,
        "budget_error_pct": round(100.0 * budget_err, 4),
        "budget_warning": bool(budget_err > 0.05),
        "add_only_contract_ok": len(evicted) == 0 and active_set.issubset(final_set),
        "active_evicted_count": len(evicted),
        "fallback_used": bool(score_meta.get("fallback_used", False)),
        "rh_fallback_used": bool(score_meta.get("rh_fallback_used", False)),
        "teacher_fallback_used": bool(score_meta.get("teacher_fallback_used", False)),
        "selector_fallback_used": bool(score_meta.get("selector_fallback_used", False)),
        "physical_granularity": score_meta.get("physical_granularity", cfg.get("rescue_granularity", "")),
        "teacher_coverage": score_meta.get("teacher_coverage", ""),
        "teacher_label_quality": score_meta.get("teacher_label_quality", ""),
        "selected_microblock_ids_before_expansion": score_meta.get("selected_microblock_ids_before_expansion", []),
        "physical_units_after_expansion": score_meta.get("physical_units_after_expansion", []),
        "parent_page_aggregation": score_meta.get("parent_page_aggregation", cfg.get("microblock_parent_aggregation", "max")),
        "selected_parent_page_ids": score_meta.get("selected_parent_page_ids", []),
        "parent_page_details": score_meta.get("parent_page_details", []),
        "expansion_ratio": score_meta.get("expansion_ratio", ""),
        "selection_granularity": score_meta.get("selection_granularity", ""),
        "final_cache_granularity": score_meta.get("final_cache_granularity", "block"),
        "bridge_score_mean": score_meta.get("bridge_score_mean", ""),
        "rh_score_mean": score_meta.get("rh_score_mean", ""),
        "redundancy_mean": score_meta.get("redundancy_mean", ""),
        "layer_budget_summary": score_meta.get("layer_budget_summary", {}),
        "head_budget_summary": score_meta.get("head_budget_summary", {}),
        "deferred_mode": score_meta.get("deferred_mode", ""),
        "deferred_simulated_triggers": score_meta.get("deferred_simulated_triggers", []),
        "used_evidence_density_selector": bool(score_meta.get("used_evidence_density_selector", False)),
        "density_threshold": score_meta.get("density_threshold", ""),
        "density_threshold_mode": score_meta.get("density_threshold_mode", ""),
        "density_alpha_max": score_meta.get("density_alpha_max", ""),
        "density_alpha_count": score_meta.get("density_alpha_count", ""),
        "density_alpha_continuity": score_meta.get("density_alpha_continuity", ""),
        "density_alpha_coverage": score_meta.get("density_alpha_coverage", ""),
        "density_lambda_red": score_meta.get("density_lambda_red", ""),
        "redundancy_fallback_used": bool(score_meta.get("redundancy_fallback_used", False)),
    })
