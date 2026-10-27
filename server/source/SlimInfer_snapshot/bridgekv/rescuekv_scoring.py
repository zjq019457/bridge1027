from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, MutableMapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from .semantic_edges import STOPWORDS, extract_block_features, tokenize_block


Diagnostics = MutableMapping[str, List[str]]


def _append_warning(diagnostics: Optional[Diagnostics], message: str) -> None:
    if diagnostics is not None:
        diagnostics.setdefault("warnings", []).append(message)


def _to_float_tensor(value: Any, *, device: Optional[torch.device] = None) -> torch.Tensor:
    if value is None:
        return torch.zeros(0, dtype=torch.float32, device=device)
    tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
    return torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)


def _as_id_list(ids: Optional[Iterable[int]]) -> List[int]:
    if ids is None:
        return []
    if isinstance(ids, torch.Tensor):
        return [int(x) for x in ids.detach().cpu().flatten().tolist()]
    return [int(x) for x in ids]


def _id_tensor(ids: Sequence[int], device: torch.device) -> torch.Tensor:
    if not ids:
        return torch.zeros(0, dtype=torch.long, device=device)
    return torch.tensor(list(ids), dtype=torch.long, device=device)


def normalize_scores(scores: torch.Tensor, mode: str = "zscore") -> torch.Tensor:
    """Return finite normalized scores with robust NaN/Inf handling."""
    vec = _to_float_tensor(scores)
    if vec.numel() == 0:
        return vec
    mode = (mode or "zscore").lower()
    if mode in {"none", "identity", "raw"}:
        return vec
    if mode == "minmax":
        min_v = vec.min()
        max_v = vec.max()
        denom = max_v - min_v
        if not torch.isfinite(denom) or float(denom.abs()) <= 1e-12:
            return torch.zeros_like(vec)
        return torch.nan_to_num((vec - min_v) / denom, nan=0.0, posinf=0.0, neginf=0.0)
    if mode == "zscore":
        mean = vec.mean()
        std = vec.std(unbiased=False)
        if not torch.isfinite(std) or float(std.abs()) <= 1e-12:
            return torch.zeros_like(vec)
        return torch.nan_to_num((vec - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)
    raise ValueError(f"Unsupported score normalization mode: {mode}")


def _broadcast_values_to_attention(attn_probs: torch.Tensor, values: torch.Tensor) -> Optional[torch.Tensor]:
    target_rank = attn_probs.dim() + 1
    while values.dim() < target_rank:
        values = values.unsqueeze(0)
    if values.shape[-2] != attn_probs.shape[-1]:
        return None
    try:
        return values.expand(*attn_probs.shape[:-1], values.shape[-2], values.shape[-1])
    except RuntimeError:
        try:
            return torch.broadcast_to(values, (*attn_probs.shape[:-1], values.shape[-2], values.shape[-1]))
        except RuntimeError:
            return None


def block_obd_saliency(
    *,
    attn_probs: Optional[torch.Tensor] = None,
    values: Optional[torch.Tensor] = None,
    block_ids: Optional[torch.Tensor] = None,
    block_size: int = 64,
    block_attention: Optional[torch.Tensor] = None,
    block_values: Optional[torch.Tensor] = None,
    output: Optional[torch.Tensor] = None,
    num_blocks: Optional[int] = None,
    eps: float = 1e-6,
    diagnostics: Optional[Diagnostics] = None,
) -> torch.Tensor:
    """OBCache-style block-level output perturbation saliency.

    Computes D_OBD(B) = ||(c_B - p_B O) / (1 - p_B + eps)||^2 / (||O||^2 + eps).
    Token-level attention/value tensors are preferred. Block-level attention/value
    tensors are accepted as a fallback.
    """
    if attn_probs is not None and values is not None:
        attn = _to_float_tensor(attn_probs)
        vals = _to_float_tensor(values, device=attn.device)
        vals = _broadcast_values_to_attention(attn, vals)
        if vals is None:
            _append_warning(diagnostics, "OBD fallback: token values could not be broadcast to attention shape.")
        else:
            token_count = int(attn.shape[-1])
            if block_ids is None:
                ids = torch.arange(token_count, device=attn.device, dtype=torch.long) // max(1, int(block_size))
            else:
                ids = torch.as_tensor(block_ids, dtype=torch.long, device=attn.device).flatten()
                if ids.numel() != token_count:
                    _append_warning(diagnostics, "OBD fallback: block_ids length mismatched token attention length.")
                    ids = torch.arange(token_count, device=attn.device, dtype=torch.long) // max(1, int(block_size))
            n_blocks = int(ids.max().item()) + 1 if ids.numel() else int(num_blocks or 0)
            if num_blocks is not None:
                n_blocks = max(n_blocks, int(num_blocks))
            output_vec = (attn.unsqueeze(-1) * vals).sum(dim=-2) if output is None else _to_float_tensor(output, device=attn.device)
            norm = output_vec.pow(2).sum(dim=-1).clamp_min(eps)
            scores = torch.zeros(n_blocks, dtype=torch.float32, device=attn.device)
            for block_id in range(n_blocks):
                mask = ids == block_id
                if not bool(mask.any()):
                    continue
                p_b = attn[..., mask].sum(dim=-1)
                c_b = (attn[..., mask].unsqueeze(-1) * vals[..., mask, :]).sum(dim=-2)
                denom = (1.0 - p_b + eps).clamp_min(eps).unsqueeze(-1)
                delta = (c_b - p_b.unsqueeze(-1) * output_vec) / denom
                block_score = delta.pow(2).sum(dim=-1) / norm
                scores[block_id] = torch.nan_to_num(block_score.mean(), nan=0.0, posinf=0.0, neginf=0.0)
            return torch.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    if block_attention is not None and block_values is not None:
        weights = _to_float_tensor(block_attention)
        vals = _to_float_tensor(block_values, device=weights.device)
        while vals.dim() < weights.dim() + 1:
            vals = vals.unsqueeze(0)
        if vals.shape[-2] != weights.shape[-1]:
            _append_warning(diagnostics, "OBD fallback failed: block_values length mismatched block_attention.")
            n_blocks = int(num_blocks or weights.shape[-1])
            return torch.zeros(n_blocks, dtype=torch.float32, device=weights.device)
        _append_warning(diagnostics, "OBD fallback: using block-level attention/value representatives.")
        try:
            vals = vals.expand(*weights.shape, vals.shape[-1])
        except RuntimeError:
            vals = torch.broadcast_to(vals, (*weights.shape, vals.shape[-1]))
        output_vec = (weights.unsqueeze(-1) * vals).sum(dim=-2) if output is None else _to_float_tensor(output, device=weights.device)
        norm = output_vec.pow(2).sum(dim=-1).clamp_min(eps)
        n_blocks = int(weights.shape[-1])
        if num_blocks is not None:
            n_blocks = max(n_blocks, int(num_blocks))
        scores = torch.zeros(n_blocks, dtype=torch.float32, device=weights.device)
        for block_id in range(min(n_blocks, int(weights.shape[-1]))):
            p_b = weights[..., block_id]
            c_b = p_b.unsqueeze(-1) * vals[..., block_id, :]
            denom = (1.0 - p_b + eps).clamp_min(eps).unsqueeze(-1)
            delta = (c_b - p_b.unsqueeze(-1) * output_vec) / denom
            scores[block_id] = torch.nan_to_num((delta.pow(2).sum(dim=-1) / norm).mean(), nan=0.0, posinf=0.0, neginf=0.0)
        return torch.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    _append_warning(diagnostics, "OBD unavailable: missing attention/value tensors; returning zero saliency.")
    n_blocks = int(num_blocks or 0)
    device = output.device if isinstance(output, torch.Tensor) else torch.device("cpu")
    return torch.zeros(n_blocks, dtype=torch.float32, device=device)


def delta_o_block_score(
    *,
    existing_delta_o: Optional[torch.Tensor] = None,
    attn_active: Optional[torch.Tensor] = None,
    attn_with_block: Optional[torch.Tensor] = None,
    num_blocks: Optional[int] = None,
    eps: float = 1e-6,
    diagnostics: Optional[Diagnostics] = None,
) -> torch.Tensor:
    """Wrap an existing delta_o score or compute a lightweight perturbation proxy."""
    if existing_delta_o is not None:
        return _to_float_tensor(existing_delta_o)
    if attn_active is not None and attn_with_block is not None:
        active = _to_float_tensor(attn_active)
        full = _to_float_tensor(attn_with_block, device=active.device)
        if active.shape != full.shape:
            _append_warning(diagnostics, "delta_o proxy unavailable: active/full attention tensors have different shapes.")
        else:
            diff = (full - active).flatten(start_dim=1) if full.dim() > 1 else (full - active).reshape(1, -1)
            denom = full.flatten(start_dim=1).norm(dim=-1).clamp_min(eps) if full.dim() > 1 else full.reshape(1, -1).norm(dim=-1).clamp_min(eps)
            proxy = diff.norm(dim=-1) / denom
            return torch.nan_to_num(proxy, nan=0.0, posinf=0.0, neginf=0.0)
    _append_warning(diagnostics, "delta_o unavailable: returning zero block scores.")
    return torch.zeros(int(num_blocks or 0), dtype=torch.float32)


def active_coverage_scores(
    block_reps: Optional[torch.Tensor],
    active_ids: Sequence[int],
    candidate_ids: Sequence[int],
    diagnostics: Optional[Diagnostics] = None,
) -> torch.Tensor:
    """Return C_active(B_i) = max cosine similarity to any active block."""
    return _max_cosine_to_active(block_reps, active_ids, candidate_ids, diagnostics)


def _max_cosine_to_active(
    block_reps: torch.Tensor,
    active_ids: Sequence[int],
    candidate_ids: Sequence[int],
    diagnostics: Optional[Diagnostics] = None,
) -> torch.Tensor:
    reps = _to_float_tensor(block_reps)
    if reps.numel() == 0 or reps.dim() < 2 or not active_ids or not candidate_ids:
        if reps.numel() == 0:
            _append_warning(diagnostics, "cosine coverage unavailable: missing block representations.")
        return torch.zeros(len(candidate_ids), dtype=torch.float32, device=reps.device if reps.numel() else torch.device("cpu"))
    active = _id_tensor(active_ids, reps.device)
    cand = _id_tensor(candidate_ids, reps.device)
    valid_active = active[(active >= 0) & (active < reps.shape[0])]
    valid_cand = cand[(cand >= 0) & (cand < reps.shape[0])]
    out = torch.zeros(len(candidate_ids), dtype=torch.float32, device=reps.device)
    if valid_active.numel() == 0 or valid_cand.numel() == 0:
        return out
    active_reps = F.normalize(reps[valid_active].float(), dim=-1, eps=1e-6)
    cand_reps = F.normalize(reps[valid_cand].float(), dim=-1, eps=1e-6)
    max_sim = (cand_reps @ active_reps.t()).max(dim=-1).values
    valid_lookup = {int(idx): pos for pos, idx in enumerate(valid_cand.detach().cpu().tolist())}
    for pos, block_id in enumerate(candidate_ids):
        if int(block_id) in valid_lookup:
            out[pos] = max_sim[valid_lookup[int(block_id)]]
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def diffusion_gap_values(
    block_reps: Optional[torch.Tensor],
    active_ids: Iterable[int],
    inactive_ids: Iterable[int],
    num_blocks: Optional[int] = None,
    diagnostics: Optional[Diagnostics] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (C_active, G_diffusion, R_active) vectors indexed by block id."""
    inactive = _as_id_list(inactive_ids)
    active = _as_id_list(active_ids)
    n_blocks = int(num_blocks or 0)
    if n_blocks <= 0:
        all_ids = active + inactive
        n_blocks = (max(all_ids) + 1) if all_ids else 0
    device = block_reps.device if isinstance(block_reps, torch.Tensor) and block_reps.numel() else torch.device("cpu")
    c_active = torch.zeros(n_blocks, dtype=torch.float32, device=device)
    if block_reps is not None and inactive and active:
        coverage = active_coverage_scores(block_reps, active, inactive, diagnostics)
        idx = _id_tensor([i for i in inactive if 0 <= i < n_blocks], device)
        if idx.numel():
            c_active[idx] = coverage[: idx.numel()].to(device)
    g_diff = (1.0 - c_active).clamp_min(0.0)
    r_active = c_active.clone()
    return (
        torch.nan_to_num(c_active, nan=0.0, posinf=0.0, neginf=0.0),
        torch.nan_to_num(g_diff, nan=0.0, posinf=0.0, neginf=0.0),
        torch.nan_to_num(r_active, nan=0.0, posinf=0.0, neginf=0.0),
    )


def _query_token_sets(query_text: str) -> Dict[str, Any]:
    """Extract lexical query features for evidence scoring."""
    text = (query_text or "").strip()
    if not text:
        return {"tokens": set(), "numbers": set(), "entities": set(), "fallback": "empty_query"}
    features = extract_block_features(text)
    return {
        "tokens": set(features.get("tokens", set())),
        "numbers": set(features.get("numbers", set())),
        "entities": set(features.get("entities", set())),
        "fallback": "",
    }


def _lexical_overlap_score(block_text: str, query_features: Dict[str, Any]) -> float:
    q_tokens = query_features.get("tokens", set())
    if not q_tokens:
        return 0.0
    block_features = extract_block_features(block_text or "")
    b_tokens = block_features.get("tokens", set())
    if not b_tokens:
        return 0.0
    overlap = len(q_tokens & b_tokens) / max(1, len(q_tokens))
    num_overlap = len(query_features.get("numbers", set()) & block_features.get("numbers", set()))
    ent_overlap = len(query_features.get("entities", set()) & block_features.get("entities", set()))
    bonus = 0.15 * min(1.0, num_overlap) + 0.10 * min(1.0, ent_overlap / max(1, len(query_features.get("entities", set()))))
    return float(min(1.0, overlap + bonus))


def _bm25_like_score(block_text: str, query_text: str, avg_doc_len: float = 64.0, k1: float = 1.2, b: float = 0.75) -> float:
    q_tokens = [t for t in tokenize_block(query_text or "") if t.lower() not in STOPWORDS]
    if not q_tokens:
        return 0.0
    doc_tokens = [t.lower() for t in tokenize_block(block_text or "") if t.lower() not in STOPWORDS]
    if not doc_tokens:
        return 0.0
    doc_len = len(doc_tokens)
    tf: Dict[str, int] = {}
    for tok in doc_tokens:
        tf[tok] = tf.get(tok, 0) + 1
    score = 0.0
    for qt in q_tokens:
        freq = tf.get(qt.lower(), 0)
        if freq <= 0:
            continue
        denom = freq + k1 * (1.0 - b + b * doc_len / max(1.0, avg_doc_len))
        score += math.log1p(freq * (k1 + 1.0) / max(denom, 1e-6))
    return float(score / max(1, len(set(q_tokens))))


def query_evidence_score(
    *,
    block_texts: Optional[Sequence[str]],
    query_text: Optional[str],
    inactive_ids: Sequence[int],
    block_reps: Optional[torch.Tensor] = None,
    query_rep: Optional[torch.Tensor] = None,
    num_blocks: Optional[int] = None,
    prompt_tail_tokens: int = 128,
    diagnostics: Optional[Diagnostics] = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """E_q(B): normalized query/evidence relevance for each block id."""
    inactive = _as_id_list(inactive_ids)
    n_blocks = int(num_blocks or 0)
    if n_blocks <= 0:
        all_ids = inactive
        n_blocks = (max(all_ids) + 1) if all_ids else 0
    device = block_reps.device if isinstance(block_reps, torch.Tensor) and block_reps.numel() else torch.device("cpu")
    e_query = torch.zeros(n_blocks, dtype=torch.float32, device=device)
    meta: Dict[str, Any] = {
        "fallback_used": False,
        "fallback_reason": "",
        "query_overlap_stats": {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0},
    }
    if not inactive or n_blocks <= 0:
        meta["fallback_used"] = True
        meta["fallback_reason"] = "no_inactive_blocks"
        return e_query, meta

    texts = list(block_texts or [])
    qtext = (query_text or "").strip()
    fallback_reason = ""
    if not qtext and texts:
        tail = " ".join(texts[-max(1, prompt_tail_tokens // 8):])
        qtext = tail.strip()
        fallback_reason = "prompt_tail_fallback"
    if not qtext:
        meta["fallback_used"] = True
        meta["fallback_reason"] = "missing_query_text"
        _append_warning(diagnostics, "query_evidence_score: no query text; returning zeros.")
        return e_query, meta

    query_features = _query_token_sets(qtext)
    if query_features.get("fallback") == "empty_query" and not fallback_reason:
        meta["fallback_used"] = True
        meta["fallback_reason"] = "empty_query_tokens"
        return e_query, meta
    if fallback_reason:
        meta["fallback_used"] = True
        meta["fallback_reason"] = fallback_reason

    lexical_vals: List[float] = []
    lexical_idx: List[int] = []
    avg_len = float(sum(len(tokenize_block(t)) for t in texts) / max(1, len(texts))) if texts else 64.0
    for block_id in inactive:
        if not (0 <= block_id < n_blocks):
            continue
        block_text = texts[block_id] if block_id < len(texts) else ""
        overlap = _lexical_overlap_score(block_text, query_features)
        bm25 = _bm25_like_score(block_text, qtext, avg_doc_len=avg_len)
        bm25_norm = min(1.0, bm25 / (bm25 + 1.0)) if bm25 > 0 else 0.0
        lexical_vals.append(0.5 * overlap + 0.5 * bm25_norm)
        lexical_idx.append(block_id)

    hidden_vals: List[float] = []
    if block_reps is not None and block_reps.numel() > 0:
        reps = _to_float_tensor(block_reps, device=device)
        qrep = _to_float_tensor(query_rep, device=device) if query_rep is not None else None
        if qrep is None and len(texts) > 0:
            proxy_id = min(len(texts) - 1, n_blocks - 1)
            if 0 <= proxy_id < reps.shape[0]:
                qrep = reps[proxy_id]
        if qrep is not None and qrep.numel() > 0:
            qrep = F.normalize(qrep.float().reshape(-1), dim=0, eps=1e-6)
            for block_id in inactive:
                if 0 <= block_id < reps.shape[0]:
                    hidden_vals.append(float(F.cosine_similarity(
                        F.normalize(reps[block_id].float().reshape(1, -1), dim=-1, eps=1e-6),
                        qrep.reshape(1, -1),
                        dim=-1,
                    ).item()))
                else:
                    hidden_vals.append(0.0)

    for pos, block_id in enumerate(lexical_idx):
        lex = lexical_vals[pos]
        hid = hidden_vals[pos] if pos < len(hidden_vals) else 0.0
        if hidden_vals:
            e_query[block_id] = 0.65 * lex + 0.35 * max(0.0, hid)
        else:
            e_query[block_id] = lex

    inactive_idx = _id_tensor([i for i in inactive if 0 <= i < n_blocks], device)
    if inactive_idx.numel():
        subset = e_query[inactive_idx]
        e_query[inactive_idx] = normalize_scores(subset, "minmax")
    meta["query_overlap_stats"] = {
        k: float(v) for k, v in _score_stats_dict(e_query[inactive_idx] if inactive_idx.numel() else e_query).items()
    }
    return torch.nan_to_num(e_query, nan=0.0, posinf=0.0, neginf=0.0), meta


def _score_stats_dict(scores: torch.Tensor) -> Dict[str, float]:
    if scores is None or scores.numel() == 0:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
    finite = torch.where(torch.isfinite(scores.detach().float()), scores.detach().float(), torch.zeros_like(scores.detach().float()))
    return {
        "min": float(finite.min().detach().cpu()),
        "max": float(finite.max().detach().cpu()),
        "mean": float(finite.mean().detach().cpu()),
        "std": float(finite.std(unbiased=False).detach().cpu()),
    }


def query_drm_s0_score(
    base_scores: torch.Tensor,
    block_reps: Optional[torch.Tensor],
    active_ids: Iterable[int],
    inactive_ids: Iterable[int],
    *,
    block_texts: Optional[Sequence[str]] = None,
    query_text: Optional[str] = None,
    alpha_diff: float = 0.5,
    lambda_red: float = 0.25,
    beta_query: float = 0.5,
    diagnostics: Optional[Diagnostics] = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """S_query_drm(B) = z(D_base) + alpha*z(G) - lambda*z(R) + beta*z(E_q) on inactive blocks."""
    base = _to_float_tensor(base_scores)
    inactive = _as_id_list(inactive_ids)
    n_blocks = int(base.numel())
    s0, drm_meta = drm_s0_score(
        base, block_reps, active_ids, inactive_ids,
        alpha_diff=alpha_diff, lambda_red=lambda_red, diagnostics=diagnostics,
    )
    e_query, eq_meta = query_evidence_score(
        block_texts=block_texts,
        query_text=query_text,
        inactive_ids=inactive,
        block_reps=block_reps,
        num_blocks=n_blocks,
        diagnostics=diagnostics,
    )
    valid_inactive = [idx for idx in inactive if 0 <= idx < n_blocks]
    if valid_inactive:
        idx = _id_tensor(valid_inactive, base.device)
        s0[idx] = s0[idx] + float(beta_query) * normalize_scores(e_query[idx], "zscore")
    meta: Dict[str, Any] = {
        "C_active": drm_meta.get("C_active"),
        "G_diffusion": drm_meta.get("G_diffusion"),
        "R_active": drm_meta.get("R_active"),
        "E_query": e_query,
        "used_base": drm_meta.get("used_base"),
        "query_fallback_used": eq_meta.get("fallback_used", False),
        "query_fallback_reason": eq_meta.get("fallback_reason", ""),
        "query_overlap_stats": eq_meta.get("query_overlap_stats", {}),
        "S_query_drm_stats": _score_stats_dict(s0[idx] if valid_inactive else s0),
        "D_base_stats": _score_stats_dict(base[idx] if valid_inactive else base),
    }
    return torch.nan_to_num(s0, nan=float("-inf"), posinf=0.0, neginf=float("-inf")), meta


def apply_gumbel_noise(
    scores: torch.Tensor,
    inactive_ids: Sequence[int],
    *,
    tau: float = 0.05,
    seed: int = 0,
    layer_idx: int = 0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """S_stoch(B) = S_base(B) + tau * Gumbel(0,1) for inactive blocks."""
    noisy = scores.clone()
    inactive = [int(x) for x in inactive_ids if 0 <= int(x) < scores.numel()]
    if not inactive or tau <= 0.0:
        return noisy, {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + int(layer_idx) * 10007 + 17)
    u = torch.rand(len(inactive), generator=generator, dtype=torch.float32).clamp(1e-6, 1.0 - 1e-6)
    gumbel = -torch.log(-torch.log(u))
    noise_vals = float(tau) * gumbel
    for pos, block_id in enumerate(inactive):
        noisy[block_id] = noisy[block_id] + noise_vals[pos].to(device=scores.device, dtype=scores.dtype)
    return noisy, _score_stats_dict(noise_vals)


def pool_mmr_select_rescue_ids(
    *,
    inactive_ids: Sequence[int],
    base_scores: torch.Tensor,
    block_reps: Optional[torch.Tensor],
    active_ids: Sequence[int],
    m: int,
    candidate_pool_ratio: float = 0.3,
    temperature: float = 1.0,
    eta_rescue: float = 0.25,
    eta_active: float = 0.25,
    seed: int = 0,
    layer_idx: int = 0,
    diagnostics: Optional[Diagnostics] = None,
) -> Tuple[List[int], Dict[str, Any]]:
    """Candidate-pool sampling with MMR diversity over inactive blocks."""
    inactive = [int(x) for x in inactive_ids]
    if m <= 0 or not inactive:
        return [], {"MMR_diversity_stats": {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}}
    scores = _to_float_tensor(base_scores)
    pool_size = max(m, int(math.ceil(len(inactive) * float(candidate_pool_ratio))))
    pool_size = min(pool_size, len(inactive))
    ordered = sorted(inactive, key=lambda bid: float(scores[bid]) if 0 <= bid < scores.numel() else float("-inf"), reverse=True)
    pool = ordered[:pool_size]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + int(layer_idx) * 99991 + 3)
    rescued: List[int] = []
    remaining = list(pool)
    mmr_values: List[float] = []
    temp = max(float(temperature), 1e-6)

    while len(rescued) < m and remaining:
        logits: List[float] = []
        for block_id in remaining:
            val = float(scores[block_id]) if 0 <= block_id < scores.numel() else float("-inf")
            if block_reps is not None and block_reps.numel() > 0:
                if rescued:
                    val -= float(eta_rescue) * _max_cosine_between(block_reps, [block_id], rescued)
                if active_ids:
                    val -= float(eta_active) * _max_cosine_between(block_reps, [block_id], list(active_ids))
            logits.append(val / temp)
        logits_t = torch.tensor(logits, dtype=torch.float32)
        logits_t = logits_t - logits_t.max()
        probs = torch.softmax(logits_t, dim=0)
        choice = int(torch.multinomial(probs, num_samples=1, generator=generator).item())
        picked = remaining[choice]
        rescued.append(int(picked))
        mmr_values.append(float(logits[choice] * temp))
        remaining.remove(int(picked))

    stats = {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    if mmr_values:
        vec = torch.tensor(mmr_values, dtype=torch.float32)
        stats = {
            "mean": float(vec.mean().item()),
            "std": float(vec.std(unbiased=False).item()),
            "min": float(vec.min().item()),
            "max": float(vec.max().item()),
        }
    return rescued, {"MMR_diversity_stats": stats}


def drm_s0_score(
    base_scores: torch.Tensor,
    block_reps: Optional[torch.Tensor],
    active_ids: Iterable[int],
    inactive_ids: Iterable[int],
    *,
    alpha_diff: float = 0.5,
    lambda_red: float = 0.25,
    diagnostics: Optional[Diagnostics] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """S0(B) = z(D_base) + alpha_diff*z(G) - lambda_red*z(R_active) on inactive blocks."""
    base = _to_float_tensor(base_scores)
    inactive = _as_id_list(inactive_ids)
    n_blocks = int(base.numel())
    _, g_diff, r_active = diffusion_gap_values(block_reps, active_ids, inactive, n_blocks, diagnostics)
    s0 = torch.full_like(base, float("-inf"))
    valid_inactive = [idx for idx in inactive if 0 <= idx < n_blocks]
    if not valid_inactive:
        return s0, {"C_active": torch.zeros(n_blocks), "G_diffusion": g_diff, "R_active": r_active}
    idx = _id_tensor(valid_inactive, base.device)
    d_base = base[idx]
    use_base = bool(torch.isfinite(d_base).any() and float(d_base.std(unbiased=False)) > 1e-12)
    if use_base:
        s0[idx] = (
            normalize_scores(d_base, "zscore")
            + float(alpha_diff) * normalize_scores(g_diff[idx], "zscore")
            - float(lambda_red) * normalize_scores(r_active[idx], "zscore")
        )
    else:
        _append_warning(diagnostics, "DRM fallback: D_base unavailable or constant; using diffusion/redundancy only.")
        s0[idx] = (
            float(alpha_diff) * normalize_scores(g_diff[idx], "zscore")
            - float(lambda_red) * normalize_scores(r_active[idx], "zscore")
        )
    c_active, _, _ = diffusion_gap_values(block_reps, active_ids, inactive, n_blocks, diagnostics)
    return torch.nan_to_num(s0, nan=float("-inf"), posinf=0.0, neginf=float("-inf")), {
        "C_active": c_active,
        "G_diffusion": g_diff,
        "R_active": r_active,
        "used_base": torch.tensor([1.0 if use_base else 0.0]),
    }


def _max_cosine_between(
    block_reps: torch.Tensor,
    left_ids: Sequence[int],
    right_ids: Sequence[int],
) -> float:
    reps = _to_float_tensor(block_reps)
    if reps.numel() == 0 or not left_ids or not right_ids:
        return 0.0
    left = _id_tensor(left_ids, reps.device)
    right = _id_tensor(right_ids, reps.device)
    valid_left = left[(left >= 0) & (left < reps.shape[0])]
    valid_right = right[(right >= 0) & (right < reps.shape[0])]
    if valid_left.numel() == 0 or valid_right.numel() == 0:
        return 0.0
    l_reps = F.normalize(reps[valid_left].float(), dim=-1, eps=1e-6)
    r_reps = F.normalize(reps[valid_right].float(), dim=-1, eps=1e-6)
    return float((l_reps @ r_reps.t()).max().item())


def mmr_select_rescue_ids(
    *,
    inactive_ids: Sequence[int],
    s0_scores: torch.Tensor,
    block_reps: Optional[torch.Tensor],
    active_ids: Sequence[int],
    m: int,
    eta_rescue: float = 0.25,
    eta_active: float = 0.25,
    seed: int = 0,
    diagnostics: Optional[Diagnostics] = None,
) -> Tuple[List[int], Dict[str, Any]]:
    """Iterative MMR rescue selection over inactive blocks."""
    inactive = [int(x) for x in inactive_ids]
    if m <= 0 or not inactive:
        return [], {
            "MMR_diversity_stats": {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0},
            "MMR_selection_trace": [],
            "plain_top_m_ids": [],
            "mmr_selected_ids": [],
            "mmr_vs_plain_top_m_jaccard": 0.0,
        }
    scores = _to_float_tensor(s0_scores)
    reps = _to_float_tensor(block_reps) if block_reps is not None else None
    rescued: List[int] = []
    remaining = list(inactive)
    mmr_values: List[float] = []
    trace_rows: List[Dict[str, Any]] = []
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    plain_top_m_ids = _topk_ids(inactive, scores, m, largest=True)

    while len(rescued) < m and remaining:
        best_id = None
        best_val = float("-inf")
        ties: List[int] = []
        step = len(rescued) + 1
        step_rows: List[Dict[str, Any]] = []
        for block_id in remaining:
            if not (0 <= block_id < scores.numel()):
                continue
            s0_val = float(scores[block_id])
            rescue_redundancy = 0.0
            active_redundancy = 0.0
            val = s0_val
            if reps is not None and reps.numel() > 0:
                if rescued:
                    rescue_redundancy = _max_cosine_between(reps, [block_id], rescued)
                    val -= float(eta_rescue) * rescue_redundancy
                if active_ids:
                    active_redundancy = _max_cosine_between(reps, [block_id], list(active_ids))
                    val -= float(eta_active) * active_redundancy
            step_rows.append({
                "mmr_step": int(step),
                "candidate_block_id": int(block_id),
                "s0_score": float(s0_val),
                "rescue_redundancy": float(rescue_redundancy),
                "active_redundancy": float(active_redundancy),
                "eta_rescue": float(eta_rescue),
                "eta_active": float(eta_active),
                "mmr_score": float(val),
                "selected": False,
                "selected_so_far": [int(x) for x in rescued],
            })
            if val > best_val + 1e-12:
                best_val = val
                best_id = block_id
                ties = [block_id]
            elif abs(val - best_val) <= 1e-12:
                ties.append(block_id)
        if best_id is None:
            break
        if len(ties) > 1:
            tie_idx = int(torch.randint(len(ties), (1,), generator=generator).item())
            best_id = ties[tie_idx]
        for row in step_rows:
            if int(row["candidate_block_id"]) == int(best_id):
                row["selected"] = True
                row["tie_candidates"] = [int(x) for x in ties]
            trace_rows.append(row)
        rescued.append(int(best_id))
        mmr_values.append(best_val)
        remaining.remove(int(best_id))

    if reps is None and remaining:
        _append_warning(diagnostics, "MMR fallback: missing block reps; diversity penalties disabled.")
    stats = {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    if mmr_values:
        vec = torch.tensor(mmr_values, dtype=torch.float32)
        stats = {
            "mean": float(vec.mean().item()),
            "std": float(vec.std(unbiased=False).item()),
            "min": float(vec.min().item()),
            "max": float(vec.max().item()),
        }
    rescued_set = set(int(x) for x in rescued)
    plain_set = set(int(x) for x in plain_top_m_ids)
    union = rescued_set | plain_set
    jaccard = float(len(rescued_set & plain_set) / len(union)) if union else 0.0
    return rescued, {
        "MMR_diversity_stats": stats,
        "MMR_selection_trace": trace_rows,
        "plain_top_m_ids": [int(x) for x in plain_top_m_ids],
        "mmr_selected_ids": [int(x) for x in rescued],
        "mmr_vs_plain_top_m_jaccard": jaccard,
    }


def diffusion_gap_score(
    base_scores: torch.Tensor,
    block_reps: Optional[torch.Tensor],
    active_ids: Iterable[int],
    inactive_ids: Iterable[int],
    diagnostics: Optional[Diagnostics] = None,
) -> torch.Tensor:
    base = _to_float_tensor(base_scores)
    score = torch.zeros_like(base)
    inactive = _as_id_list(inactive_ids)
    valid_inactive = [idx for idx in inactive if 0 <= idx < base.numel()]
    if not valid_inactive:
        return score
    coverage = active_coverage_scores(_to_float_tensor(block_reps, device=base.device), _as_id_list(active_ids), valid_inactive, diagnostics)
    idx = _id_tensor(valid_inactive, base.device)
    score[idx] = base[idx] * (1.0 - coverage.to(base.device)).clamp_min(0.0)
    return torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)


def redundancy_aware_score(
    base_scores: torch.Tensor,
    block_reps: Optional[torch.Tensor],
    active_ids: Iterable[int],
    inactive_ids: Iterable[int],
    lambda_red: float = 0.25,
    diagnostics: Optional[Diagnostics] = None,
) -> torch.Tensor:
    base = _to_float_tensor(base_scores)
    score = torch.zeros_like(base)
    inactive = _as_id_list(inactive_ids)
    valid_inactive = [idx for idx in inactive if 0 <= idx < base.numel()]
    if not valid_inactive:
        return score
    redundancy = _max_cosine_to_active(_to_float_tensor(block_reps, device=base.device), _as_id_list(active_ids), valid_inactive, diagnostics)
    idx = _id_tensor(valid_inactive, base.device)
    denom = 1.0 + float(lambda_red) * redundancy.clamp_min(0.0).to(base.device)
    score[idx] = base[idx] / denom.clamp_min(1e-6)
    return torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)


def path_aware_score(
    *,
    attn_mass: Optional[torch.Tensor] = None,
    active_ids: Optional[Iterable[int]] = None,
    inactive_ids: Optional[Iterable[int]] = None,
    block_positions: Optional[torch.Tensor] = None,
    num_blocks: Optional[int] = None,
    bridge_bonus: float = 0.10,
    diagnostics: Optional[Diagnostics] = None,
) -> torch.Tensor:
    """Lightweight block path score with a deterministic fallback approximation."""
    if attn_mass is None:
        _append_warning(diagnostics, "Path score fallback: missing attention flow; using bridge-position bonus only.")
        n_blocks = int(num_blocks or 0)
        scores = torch.zeros(n_blocks, dtype=torch.float32)
    else:
        scores = normalize_scores(_to_float_tensor(attn_mass).flatten(), "minmax")
        n_blocks = int(scores.numel())
        if num_blocks is not None and int(num_blocks) > n_blocks:
            padded = torch.zeros(int(num_blocks), dtype=torch.float32, device=scores.device)
            padded[:n_blocks] = scores
            scores = padded
            n_blocks = int(num_blocks)
        _append_warning(diagnostics, "Path score fallback: using recent-query block attention mass plus bridge-position bonus.")

    inactive = set(_as_id_list(inactive_ids))
    if n_blocks == 0:
        return scores
    positions = torch.arange(n_blocks, dtype=torch.float32, device=scores.device) if block_positions is None else _to_float_tensor(block_positions, device=scores.device).flatten()
    if positions.numel() < n_blocks:
        positions = torch.arange(n_blocks, dtype=torch.float32, device=scores.device)

    if scores.numel() > 0 and float(scores.max()) > 0.0:
        k = max(2, min(n_blocks, int(round(n_blocks * 0.2))))
        high = torch.topk(scores, k=k).indices.detach().cpu().tolist()
    else:
        active = _as_id_list(active_ids)
        high = active[:]
    if len(high) >= 2:
        lo = min(high)
        hi = max(high)
        span = max(1, hi - lo)
        for idx in range(lo + 1, hi):
            if inactive and idx not in inactive:
                continue
            center_bonus = 1.0 - abs((idx - lo) / span - 0.5) * 2.0
            scores[idx] += float(bridge_bonus) * max(0.0, center_bonus)
    return torch.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)


def combined_abc_score(
    *,
    obd_scores: Optional[torch.Tensor] = None,
    block_reps: Optional[torch.Tensor] = None,
    active_ids: Optional[Iterable[int]] = None,
    inactive_ids: Optional[Iterable[int]] = None,
    path_scores: Optional[torch.Tensor] = None,
    attn_mass: Optional[torch.Tensor] = None,
    alpha_diff: float = 0.5,
    lambda_red: float = 0.25,
    beta_path: float = 0.25,
    diagnostics: Optional[Diagnostics] = None,
) -> torch.Tensor:
    active = _as_id_list(active_ids)
    inactive = _as_id_list(inactive_ids)
    if obd_scores is None:
        if block_reps is not None:
            n_blocks = int(torch.as_tensor(block_reps).shape[0])
        elif path_scores is not None:
            n_blocks = int(torch.as_tensor(path_scores).numel())
        elif attn_mass is not None:
            n_blocks = int(torch.as_tensor(attn_mass).numel())
        else:
            n_blocks = 0
        _append_warning(diagnostics, "ABC score: missing OBD saliency; using zeros for OBD term.")
        obd = torch.zeros(n_blocks, dtype=torch.float32)
    else:
        obd = _to_float_tensor(obd_scores)
    total = normalize_scores(obd, "zscore")

    if block_reps is not None and obd.numel() > 0:
        diff = diffusion_gap_score(obd, block_reps, active, inactive, diagnostics)
        total = total + float(alpha_diff) * normalize_scores(diff, "zscore")
        reps = _to_float_tensor(block_reps, device=obd.device)
        red = torch.zeros_like(obd)
        valid_inactive = [idx for idx in inactive if 0 <= idx < obd.numel()]
        if valid_inactive:
            vals = _max_cosine_to_active(reps, active, valid_inactive, diagnostics).to(obd.device)
            red[_id_tensor(valid_inactive, obd.device)] = vals.clamp_min(0.0)
        total = total - float(lambda_red) * normalize_scores(red, "zscore")
    else:
        _append_warning(diagnostics, "ABC score: missing block reps; skipped diffusion and redundancy terms.")

    if path_scores is None:
        path_scores = path_aware_score(attn_mass=attn_mass, active_ids=active, inactive_ids=inactive, num_blocks=int(obd.numel()), diagnostics=diagnostics)
    path = _to_float_tensor(path_scores, device=total.device)
    if path.numel() == total.numel():
        total = total + float(beta_path) * normalize_scores(path, "zscore")
    else:
        _append_warning(diagnostics, "ABC score: path term size mismatch; skipped path term.")
    return torch.nan_to_num(total, nan=0.0, posinf=0.0, neginf=0.0)


@dataclass
class RescuePolicyResult:
    final_ids: torch.Tensor
    rescued_ids: torch.Tensor
    dropped_ids: torch.Tensor
    final_budget_type: str
    diagnostics: Dict[str, List[str]]


def _topk_ids(ids: Sequence[int], scores: torch.Tensor, k: int, largest: bool = True) -> List[int]:
    valid = [idx for idx in ids if 0 <= int(idx) < scores.numel()]
    if not valid or k <= 0:
        return []
    ordered = sorted(valid, key=lambda idx: (float(scores[int(idx)]), -int(idx) if largest else int(idx)), reverse=largest)
    return [int(idx) for idx in ordered[: min(k, len(ordered))]]


def apply_rescue_policy(
    *,
    active_ids: Iterable[int],
    inactive_ids: Iterable[int],
    rescue_scores: torch.Tensor,
    active_scores: Optional[torch.Tensor] = None,
    obd_scores: Optional[torch.Tensor] = None,
    block_reps: Optional[torch.Tensor] = None,
    policy: str = "none",
    rescue_ratio: float = 0.05,
    swap_drop_score: str = "sliminfer",
    preserve_ids: Optional[Iterable[int]] = None,
    recent_ids: Optional[Iterable[int]] = None,
    sink_ids: Optional[Iterable[int]] = None,
    forced_keep_ids: Optional[Iterable[int]] = None,
    tau: float = 0.0,
    sort_output: bool = True,
    mmr_selection: bool = False,
    eta_rescue: float = 0.25,
    eta_active: float = 0.25,
    seed: int = 0,
    diagnostics: Optional[Diagnostics] = None,
) -> RescuePolicyResult:
    """Apply additive, budget-neutral, or conditional rescue over block ids."""
    local_diag: Dict[str, List[str]] = {"warnings": []}
    diag = diagnostics if diagnostics is not None else local_diag
    policy = (policy or "none").lower()
    if policy == "add":
        policy = "additive_rescue"
    if policy == "sliminfer_larger":
        policy = "additive_rescue"
    if policy == "swap":
        policy = "budget_neutral_swap"

    active = _as_id_list(active_ids)
    inactive = [idx for idx in _as_id_list(inactive_ids) if idx not in set(active)]
    device = rescue_scores.device if isinstance(rescue_scores, torch.Tensor) else torch.device("cpu")
    rescue = _to_float_tensor(rescue_scores, device=device)
    protected = set(_as_id_list(preserve_ids)) | set(_as_id_list(recent_ids)) | set(_as_id_list(sink_ids)) | set(_as_id_list(forced_keep_ids))

    k_active = len(active)
    m_rescue = int(round(k_active * float(rescue_ratio)))
    if rescue_ratio > 0.0 and k_active > 0 and inactive:
        m_rescue = max(1, m_rescue)
    m_rescue = min(m_rescue, len(inactive))

    if policy in {"none", ""} or m_rescue <= 0:
        final = sorted(set(active)) if sort_output else list(dict.fromkeys(active))
        return RescuePolicyResult(
            final_ids=_id_tensor(final, device),
            rescued_ids=_id_tensor([], device),
            dropped_ids=_id_tensor([], device),
            final_budget_type="K",
            diagnostics={k: list(v) for k, v in diag.items()},
        )

    rescue_ids = _topk_ids(inactive, rescue, m_rescue, largest=True)
    if mmr_selection and policy == "additive_rescue":
        rescue_ids, mmr_diag = mmr_select_rescue_ids(
            inactive_ids=inactive,
            s0_scores=rescue,
            block_reps=block_reps,
            active_ids=active,
            m=m_rescue,
            eta_rescue=float(eta_rescue),
            eta_active=float(eta_active),
            seed=int(seed),
            diagnostics=diag,
        )
        diag["mmr_diversity_stats"] = [str(mmr_diag.get("MMR_diversity_stats", {}))]

    if policy == "additive_rescue":
        final = list(dict.fromkeys(active + rescue_ids))
        if sort_output:
            final = sorted(final)
        return RescuePolicyResult(
            final_ids=_id_tensor(final, device),
            rescued_ids=_id_tensor(rescue_ids, device),
            dropped_ids=_id_tensor([], device),
            final_budget_type="K+M",
            diagnostics={k: list(v) for k, v in diag.items()},
        )

    if swap_drop_score == "obd" and obd_scores is not None:
        drop = _to_float_tensor(obd_scores, device=device)
    elif active_scores is not None:
        drop = _to_float_tensor(active_scores, device=device)
    else:
        _append_warning(diag, "Swap drop score missing; falling back to rescue score scale.")
        drop = rescue

    drop_candidates = [idx for idx in active if idx not in protected]
    drop_ids = _topk_ids(drop_candidates, drop, m_rescue, largest=False)
    if len(drop_ids) < len(rescue_ids):
        rescue_ids = rescue_ids[: len(drop_ids)]

    if policy == "conditional_swap":
        accepted_rescue: List[int] = []
        accepted_drop: List[int] = []
        for r_id, d_id in zip(rescue_ids, drop_ids):
            if float(rescue[int(r_id)]) > float(drop[int(d_id)]) + float(tau):
                accepted_rescue.append(int(r_id))
                accepted_drop.append(int(d_id))
        rescue_ids = accepted_rescue
        drop_ids = accepted_drop
    elif policy != "budget_neutral_swap":
        raise ValueError(f"Unsupported rescue policy: {policy}")

    final_set = set(active)
    for dropped in drop_ids:
        if dropped not in protected:
            final_set.discard(dropped)
    final_set.update(rescue_ids)
    final = sorted(final_set) if sort_output else list(dict.fromkeys([idx for idx in active if idx in final_set] + rescue_ids))
    if len(final) != len(set(final)):
        _append_warning(diag, "Policy produced duplicate ids before final de-duplication.")
        final = list(dict.fromkeys(final))
    return RescuePolicyResult(
        final_ids=_id_tensor(final, device),
        rescued_ids=_id_tensor(rescue_ids, device),
        dropped_ids=_id_tensor(drop_ids, device),
        final_budget_type="K",
        diagnostics={k: list(v) for k, v in diag.items()},
    )
