"""SP-SlimInfer v2: real cross-layer semantic trajectory capture (read-only hook).

Default off. Activated only when ``sp_trajectory_dump_path`` is set in SlimInfer config.
Does not modify scores, active sets, or KV budgets.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import torch

TRAJECTORY_PRUNE_LAYERS: Tuple[int, ...] = (9, 18, 26)
BLOCK_SIZE = 64
UNAVAILABLE = "UNAVAILABLE"

_SCALAR_FIELDS: Tuple[str, ...] = (
    "sample_id",
    "task",
    "layer_idx",
    "block_id",
    "token_start",
    "token_end",
    "valid_token_count",
    "alive_before_pruning",
    "selected_after_pruning",
    "candidate_selected",
    "forced_sink",
    "previously_active",
    "original_sliminfer_score",
    "original_rank",
    "query_hidden_cosine",
    "query_key_cosine",
    "representation_available",
    "trajectory_censored",
    "vector_npz",
    "vector_row",
)


def _l2_normalize(vec: torch.Tensor) -> torch.Tensor:
    norm = vec.float().norm(p=2).clamp_min(1e-8)
    return vec.float() / norm


def _cosine_np(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    if a is None or b is None:
        return float("nan")
    if a.shape != b.shape:
        return float("nan")
    dot = float(np.dot(a.astype(np.float64), b.astype(np.float64)))
    if not math.isfinite(dot):
        return float("nan")
    return dot


def block_token_span(block_id: int, prompt_len: int, block_size: int = BLOCK_SIZE) -> Tuple[int, int, int]:
    start = int(block_id) * block_size
    end = min((int(block_id) + 1) * block_size, int(prompt_len))
    return start, end, max(0, end - start)


def pool_hidden_block_rep(
    hs_tokens: Optional[torch.Tensor],
    block_id: int,
    prompt_len: int,
    abs_to_rel: Mapping[int, int],
    block_size: int = BLOCK_SIZE,
) -> Optional[np.ndarray]:
    if hs_tokens is None:
        return None
    start, end, _ = block_token_span(block_id, prompt_len, block_size)
    rel_indices = [abs_to_rel[abs_idx] for abs_idx in range(start, end) if abs_idx in abs_to_rel]
    if not rel_indices:
        return None
    with torch.no_grad():
        vecs = hs_tokens[0, rel_indices, :]
        pooled = _l2_normalize(vecs.mean(dim=0))
        return pooled.detach().cpu().numpy().astype(np.float32)


def pool_key_block_rep(
    represent_keys: torch.Tensor,
    block_to_kernel_map: Mapping[int, Tuple[int, int]],
    block_id: int,
) -> Optional[np.ndarray]:
    if block_id not in block_to_kernel_map:
        return None
    start, num_k = block_to_kernel_map[block_id]
    if num_k <= 0:
        return None
    with torch.no_grad():
        k_slice = represent_keys[:, :, start : start + num_k, :]
        pooled = _l2_normalize(k_slice.mean(dim=(0, 1, 2)))
        return pooled.detach().cpu().numpy().astype(np.float32)


def pool_hidden_query_rep(
    hs_tokens: Optional[torch.Tensor],
    prompt_len: int,
    abs_to_rel: Mapping[int, int],
    local_query_window: int = 4,
) -> Optional[np.ndarray]:
    if hs_tokens is None or prompt_len <= 0:
        return None
    window = max(1, int(local_query_window))
    start_abs = max(0, int(prompt_len) - window)
    rel_indices = [abs_to_rel[abs_idx] for abs_idx in range(start_abs, int(prompt_len)) if abs_idx in abs_to_rel]
    if not rel_indices:
        return None
    with torch.no_grad():
        vecs = hs_tokens[0, rel_indices, :]
        pooled = _l2_normalize(vecs.mean(dim=0))
        return pooled.detach().cpu().numpy().astype(np.float32)


def pool_local_query_rep(q_for_pruning: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        pooled = _l2_normalize(q_for_pruning.mean(dim=(0, 1, 2)))
        return pooled.detach().cpu().numpy().astype(np.float32)


def compute_original_ranks(original_block_ids: Sequence[int], qk_scores: torch.Tensor) -> Dict[int, int]:
    scores = qk_scores.detach().float().cpu().tolist()
    pairs = [
        (int(original_block_ids[i]), float(scores[i]))
        for i in range(min(len(original_block_ids), len(scores)))
        if math.isfinite(float(scores[i]))
    ]
    pairs.sort(key=lambda x: x[1], reverse=True)
    return {block_id: rank for rank, (block_id, _) in enumerate(pairs, start=1)}


def _dump_root(cfg: Mapping[str, Any]) -> Optional[Path]:
    raw = cfg.get("sp_trajectory_dump_path")
    if not raw:
        return None
    return Path(str(raw))


def _audit_context(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    ctx = cfg.get("audit_context")
    return dict(ctx) if isinstance(ctx, dict) else {}


def reset_trajectory_sample_state(cfg: Dict[str, Any]) -> None:
    """Clear per-sample trajectory state; call at the start of each sample."""
    for key in (
        "_sp_traj_prev_selected",
        "_sp_traj_layer_history",
        "_sp_traj_hook_calls",
        "_sp_traj_sample_flushed",
        "_sp_traj_prune_layer_selected",
    ):
        cfg.pop(key, None)
    cfg["_sp_traj_prev_selected"] = set()
    cfg["_sp_traj_layer_history"] = {}
    cfg["_sp_traj_hook_calls"] = []
    cfg["_sp_traj_sample_flushed"] = False


def finalize_trajectory_sample(cfg: Dict[str, Any]) -> None:
    """Flush any pending per-sample artifacts (no-op if already flushed)."""
    if cfg.get("_sp_traj_sample_flushed"):
        return
    cfg["_sp_traj_sample_flushed"] = True


def _ensure_out_dirs(root: Path) -> Tuple[Path, Path]:
    vec_dir = root / "trajectory_vectors"
    vec_dir.mkdir(parents=True, exist_ok=True)
    root.mkdir(parents=True, exist_ok=True)
    return root, vec_dir


def _append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _append_csv_row(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(_SCALAR_FIELDS), extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in _SCALAR_FIELDS})


def record_trajectory_prune_layer(
    *,
    cfg: Dict[str, Any],
    original_block_ids: Sequence[int],
    block_scores: torch.Tensor,
    qk_block_scores: torch.Tensor,
    q_for_pruning: torch.Tensor,
    represent_keys: torch.Tensor,
    block_to_kernel_map: Mapping[int, Tuple[int, int]],
    top_k_indices: torch.Tensor,
    must_keep_block_ids: Set[int],
    num_blocks_to_keep: int,
    final_kept_ids: Set[int],
) -> None:
    """Record one pruning-layer snapshot. Read-only w.r.t. SlimInfer decisions."""
    root = _dump_root(cfg)
    if root is None:
        return
    layer_idx = int(cfg.get("layer_idx", -1))
    if layer_idx not in TRAJECTORY_PRUNE_LAYERS:
        return
    if str(cfg.get("phase", "")) == "decode":
        return

    ctx = _audit_context(cfg)
    task = str(ctx.get("task", cfg.get("task", "")) or "")
    sample_id = str(ctx.get("sample_id", ctx.get("sample_index", "")) or "")
    prompt_len = int(cfg.get("original_prompt_len", ctx.get("context_length", 0)) or 0)

    hs_tokens = cfg.get("_trajectory_hs_tokens")
    abs_to_rel = cfg.get("_trajectory_query_abs_to_relative_map") or {}
    if not isinstance(abs_to_rel, dict):
        abs_to_rel = {}

    prev_selected: Set[int] = set(cfg.get("_sp_traj_prev_selected") or set())
    prune_hist: Dict[int, Set[int]] = dict(cfg.get("_sp_traj_prune_layer_selected") or {})
    prior_prune_selected: Set[int] = set()
    for pl in TRAJECTORY_PRUNE_LAYERS:
        if pl >= layer_idx:
            break
        prior_prune_selected.update(prune_hist.get(int(pl), set()))
    rank_map = compute_original_ranks(original_block_ids, qk_block_scores)
    id_to_qk_score = {
        int(original_block_ids[i]): float(qk_block_scores[i].item())
        for i in range(min(len(original_block_ids), int(qk_block_scores.numel())))
    }
    top_k_set = {int(original_block_ids[int(i)]) for i in top_k_indices.detach().cpu().tolist() if int(i) < len(original_block_ids)}
    local_query = pool_local_query_rep(q_for_pruning)
    local_window = int(cfg.get("local_query_window", 4) or 4)
    hidden_query = pool_hidden_query_rep(hs_tokens, prompt_len, abs_to_rel, local_window)

    alive_set = {int(x) for x in original_block_ids}
    censored_ids = sorted(prior_prune_selected - alive_set)
    all_block_ids = sorted(alive_set.union(censored_ids))

    hidden_rows: List[np.ndarray] = []
    key_rows: List[np.ndarray] = []
    rep_avail: List[bool] = []
    censored_flags: List[bool] = []

    scalar_rows: List[Dict[str, Any]] = []
    root_dir, vec_dir = _ensure_out_dirs(root)
    npz_name = f"{task}_{sample_id}_L{layer_idx}.npz"
    npz_path = vec_dir / npz_name

    for row_idx, block_id in enumerate(all_block_ids):
        block_id = int(block_id)
        start, end, span_len = block_token_span(block_id, prompt_len)
        is_censored = block_id not in alive_set
        forced_sink = bool(original_block_ids and block_id == int(original_block_ids[0]) and block_id in must_keep_block_ids)

        hidden_rep = None if is_censored else pool_hidden_block_rep(hs_tokens, block_id, prompt_len, abs_to_rel)
        key_rep = None if is_censored else pool_key_block_rep(represent_keys, block_to_kernel_map, block_id)

        if hidden_rep is None and not is_censored:
            hidden_rep = None
        if key_rep is None and not is_censored:
            key_rep = None

        rep_available = (not is_censored) and hidden_rep is not None and key_rep is not None
        qh_cos = _cosine_np(hidden_query, hidden_rep) if rep_available and hidden_query is not None else float("nan")
        qk_cos = _cosine_np(local_query, key_rep) if rep_available else float("nan")

        hidden_vec = hidden_rep if hidden_rep is not None else np.full((1,), np.nan, dtype=np.float32)
        key_vec = key_rep if key_rep is not None else np.full((1,), np.nan, dtype=np.float32)

        if hidden_rep is not None:
            hidden_rows.append(hidden_rep)
        else:
            hidden_rows.append(np.array([np.nan], dtype=np.float32))
        if key_rep is not None:
            key_rows.append(key_rep)
        else:
            key_rows.append(np.array([np.nan], dtype=np.float32))
        rep_avail.append(bool(rep_available))
        censored_flags.append(bool(is_censored))

        score_val = id_to_qk_score.get(block_id, UNAVAILABLE)
        if block_id in alive_set and score_val != UNAVAILABLE and not math.isfinite(float(score_val)):
            score_val = UNAVAILABLE

        scalar_rows.append(
            {
                "sample_id": sample_id,
                "task": task,
                "layer_idx": layer_idx,
                "block_id": block_id,
                "token_start": start,
                "token_end": end,
                "valid_token_count": span_len,
                "alive_before_pruning": int(not is_censored),
                "selected_after_pruning": int(block_id in final_kept_ids),
                "candidate_selected": int(block_id in top_k_set or block_id in must_keep_block_ids),
                "forced_sink": int(forced_sink),
                "previously_active": int(block_id in prev_selected),
                "original_sliminfer_score": score_val,
                "original_rank": rank_map.get(block_id, UNAVAILABLE),
                "query_hidden_cosine": qh_cos if rep_available else UNAVAILABLE,
                "query_key_cosine": qk_cos if rep_available else UNAVAILABLE,
                "representation_available": int(rep_available),
                "trajectory_censored": int(is_censored),
                "vector_npz": str(npz_path.relative_to(root_dir)),
                "vector_row": row_idx,
            }
        )

    block_ids_arr = np.array(all_block_ids, dtype=np.int32)
    hidden_dim = hidden_rows[0].shape[0] if hidden_rows and hidden_rows[0].ndim == 1 and hidden_rows[0].size > 1 else 0
    key_dim = key_rows[0].shape[0] if key_rows and key_rows[0].ndim == 1 and key_rows[0].size > 1 else 0

    if hidden_dim > 1 and key_dim > 1:
        hidden_stack = np.stack([r for r in hidden_rows if r.size == hidden_dim], axis=0)
        if hidden_stack.shape[0] != len(all_block_ids):
            hidden_stack = np.stack(
                [r if r.size == hidden_dim else np.full((hidden_dim,), np.nan, dtype=np.float32) for r in hidden_rows],
                axis=0,
            )
        key_stack = np.stack(
            [r if r.size == key_dim else np.full((key_dim,), np.nan, dtype=np.float32) for r in key_rows],
            axis=0,
        )
    else:
        hidden_stack = np.array([np.nan], dtype=np.float32)
        key_stack = np.array([np.nan], dtype=np.float32)

    np.savez_compressed(
        npz_path,
        block_ids=block_ids_arr,
        hidden_block_rep=hidden_stack,
        key_block_rep=key_stack,
        local_query_rep=local_query,
        hidden_query_rep=hidden_query if hidden_query is not None else np.array([np.nan], dtype=np.float32),
        representation_available=np.array(rep_avail, dtype=np.bool_),
        trajectory_censored=np.array(censored_flags, dtype=np.bool_),
        hidden_dim=np.array([hidden_dim], dtype=np.int32),
        key_dim=np.array([key_dim], dtype=np.int32),
        query_dim=np.array([local_query.shape[0]], dtype=np.int32),
    )

    index_path = root_dir / "trajectory_index.jsonl"
    csv_path = root_dir / "trajectory_scalar_features.csv"
    for row in scalar_rows:
        _append_csv_row(csv_path, row)
    _append_jsonl(
        index_path,
        {
            "task": task,
            "sample_id": sample_id,
            "layer_idx": layer_idx,
            "npz": str(npz_path.relative_to(root_dir)),
            "num_blocks": len(all_block_ids),
            "num_alive": len(alive_set),
            "num_censored": len(censored_ids),
            "num_selected": len(final_kept_ids),
            "budget_k": int(num_blocks_to_keep),
            "hidden_dim": int(hidden_dim),
            "key_dim": int(key_dim),
            "query_dim": int(local_query.shape[0]),
        },
    )

    layer_history = dict(cfg.get("_sp_traj_layer_history") or {})
    layer_history[int(layer_idx)] = {
        "block_ids": all_block_ids,
        "hidden": hidden_stack,
        "key": key_stack,
        "local_query": local_query,
        "query_hidden_cosine": [row["query_hidden_cosine"] for row in scalar_rows],
        "query_key_cosine": [row["query_key_cosine"] for row in scalar_rows],
        "scores": [row["original_sliminfer_score"] for row in scalar_rows],
        "censored": censored_flags,
        "rep_available": rep_avail,
    }
    cfg["_sp_traj_layer_history"] = layer_history
    cfg["_sp_traj_prev_selected"] = set(int(x) for x in final_kept_ids)
    prune_hist[int(layer_idx)] = set(int(x) for x in final_kept_ids)
    cfg["_sp_traj_prune_layer_selected"] = prune_hist

    hook_calls = list(cfg.get("_sp_traj_hook_calls") or [])
    hook_calls.append(
        {
            "task": task,
            "sample_id": sample_id,
            "layer_idx": layer_idx,
            "num_blocks": len(all_block_ids),
            "hidden_dim": int(hidden_dim),
            "key_dim": int(key_dim),
            "query_dim": int(local_query.shape[0]),
        }
    )
    cfg["_sp_traj_hook_calls"] = hook_calls


# ---------------------------------------------------------------------------
# Post-hoc causal feature computation (from saved trajectory, no gold)
# ---------------------------------------------------------------------------

def _prev_prune_layer(layer_idx: int) -> Optional[int]:
    prior = [l for l in TRAJECTORY_PRUNE_LAYERS if l < layer_idx]
    return prior[-1] if prior else None


def semantic_persistence(
    history: Mapping[int, Mapping[str, Any]],
    layer_idx: int,
    block_id: int,
) -> Optional[float]:
    """Persistence of query relevance across observable pruning layers."""
    if layer_idx not in history:
        return None
    prev_layer = _prev_prune_layer(layer_idx)
    if prev_layer is None or prev_layer not in history:
        return None

    cur = history[layer_idx]
    prev = history[prev_layer]
    if block_id not in cur["block_ids"] or block_id not in prev["block_ids"]:
        return None
    ci = cur["block_ids"].index(block_id)
    pi = prev["block_ids"].index(block_id)
    if cur["censored"][ci] or prev["censored"][pi]:
        return None
    if not cur["rep_available"][ci] or not prev["rep_available"][pi]:
        return None

    cur_qh = cur["query_hidden_cosine"][ci]
    prev_qh = prev["query_hidden_cosine"][pi]
    if cur_qh == UNAVAILABLE or prev_qh == UNAVAILABLE:
        return None
    cur_val = float(cur_qh)
    prev_val = float(prev_qh)
    if not math.isfinite(cur_val) or not math.isfinite(prev_val):
        return None
    hist_vals = []
    for layer in TRAJECTORY_PRUNE_LAYERS:
        if layer > layer_idx:
            break
        if layer not in history or block_id not in history[layer]["block_ids"]:
            continue
        idx = history[layer]["block_ids"].index(block_id)
        if history[layer]["censored"][idx] or not history[layer]["rep_available"][idx]:
            continue
        v = history[layer]["query_hidden_cosine"][idx]
        if v != UNAVAILABLE and math.isfinite(float(v)):
            hist_vals.append(float(v))
    if len(hist_vals) < 2:
        return None
    mean_h = float(np.mean(hist_vals))
    var_h = float(np.var(hist_vals))
    drop = max(0.0, prev_val - cur_val)
    return float(0.5 * cur_val + 0.3 * mean_h - 0.1 * var_h - 0.1 * drop)


def semantic_drift(
    history: Mapping[int, Mapping[str, Any]],
    layer_idx: int,
    block_id: int,
) -> Optional[float]:
    prev_layer = _prev_prune_layer(layer_idx)
    if prev_layer is None or layer_idx not in history or prev_layer not in history:
        return None
    cur = history[layer_idx]
    prev = history[prev_layer]
    if block_id not in cur["block_ids"] or block_id not in prev["block_ids"]:
        return None
    ci = cur["block_ids"].index(block_id)
    pi = prev["block_ids"].index(block_id)
    if cur["censored"][ci] or prev["censored"][pi]:
        return None
    if not cur["rep_available"][ci] or not prev["rep_available"][pi]:
        return None
    h_cur = cur["hidden"][ci]
    h_prev = prev["hidden"][pi]
    if h_cur.ndim != 1 or h_prev.ndim != 1 or h_cur.shape != h_prev.shape:
        return None
    cos_h = _cosine_np(h_cur, h_prev)
    if not math.isfinite(cos_h):
        return None
    rep_drift = 1.0 - cos_h
    cur_qh = float(cur["query_hidden_cosine"][ci])
    prev_qh = float(prev["query_hidden_cosine"][pi])
    rel_drop = max(0.0, prev_qh - cur_qh) if math.isfinite(cur_qh) and math.isfinite(prev_qh) else 0.0
    return float(0.7 * rep_drift + 0.3 * rel_drop)


def semantic_absorption(
    history: Mapping[int, Mapping[str, Any]],
    layer_idx: int,
    block_id: int,
    top_m: int = 3,
) -> Optional[float]:
    prev_layer = _prev_prune_layer(layer_idx)
    if prev_layer is None or layer_idx not in history or prev_layer not in history:
        return None
    cur = history[layer_idx]
    prev = history[prev_layer]
    if block_id not in prev["block_ids"]:
        return None
    pi = prev["block_ids"].index(block_id)
    if prev["censored"][pi] or not prev["rep_available"][pi]:
        return None
    h_prev = prev["hidden"][pi]
    cosines: List[float] = []
    for j, other_id in enumerate(cur["block_ids"]):
        if other_id == block_id:
            continue
        if cur["censored"][j] or not cur["rep_available"][j]:
            continue
        cos_v = _cosine_np(h_prev, cur["hidden"][j])
        if math.isfinite(cos_v):
            cosines.append(cos_v)
    if not cosines:
        return None
    cosines.sort(reverse=True)
    return float(np.mean(cosines[: max(1, min(top_m, len(cosines)))]))


def semantic_relay(
    history: Mapping[int, Mapping[str, Any]],
    layer_idx: int,
    block_id: int,
) -> Optional[float]:
    prev_layer = _prev_prune_layer(layer_idx)
    if prev_layer is None or layer_idx not in history or prev_layer not in history:
        return None
    cur = history[layer_idx]
    prev = history[prev_layer]
    if block_id not in prev["block_ids"]:
        return None
    pi = prev["block_ids"].index(block_id)
    if prev["censored"][pi] or not prev["rep_available"][pi]:
        return None
    early_qh = prev["query_hidden_cosine"][pi]
    if early_qh == UNAVAILABLE or not math.isfinite(float(early_qh)):
        return None
    h_early = prev["hidden"][pi]
    best = None
    for j, other_id in enumerate(cur["block_ids"]):
        if other_id == block_id:
            continue
        if cur["censored"][j] or not cur["rep_available"][j]:
            continue
        if other_id not in prev["block_ids"]:
            prev_j = None
        else:
            pji = prev["block_ids"].index(other_id)
            prev_j = prev["query_hidden_cosine"][pji] if not prev["censored"][pji] else UNAVAILABLE
        cur_j = cur["query_hidden_cosine"][j]
        if prev_j == UNAVAILABLE or cur_j == UNAVAILABLE:
            continue
        if not math.isfinite(float(prev_j)) or not math.isfinite(float(cur_j)):
            continue
        rel_inc = float(cur_j) - float(prev_j)
        if rel_inc <= 0:
            continue
        sim = _cosine_np(h_early, cur["hidden"][j])
        if not math.isfinite(sim):
            continue
        val = float(float(early_qh) * sim * rel_inc)
        best = val if best is None else max(best, val)
    return best


def active_set_redundancy(
    history: Mapping[int, Mapping[str, Any]],
    layer_idx: int,
    candidate_block_ids: Iterable[int],
) -> Dict[str, Optional[float]]:
    if layer_idx not in history:
        return {"max_pairwise_cosine": None, "mean_pairwise_cosine": None, "p90_pairwise_cosine": None, "high_redundancy_count": None}
    cur = history[layer_idx]
    vecs: List[np.ndarray] = []
    ids: List[int] = []
    for bid in candidate_block_ids:
        if bid not in cur["block_ids"]:
            continue
        idx = cur["block_ids"].index(bid)
        if cur["censored"][idx] or not cur["rep_available"][idx]:
            continue
        vecs.append(cur["hidden"][idx])
        ids.append(int(bid))
    if len(vecs) < 2:
        return {"max_pairwise_cosine": None, "mean_pairwise_cosine": None, "p90_pairwise_cosine": None, "high_redundancy_count": None}
    pair_cos: List[float] = []
    per_block_max: Dict[int, float] = {i: 0.0 for i in ids}
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            cos_v = _cosine_np(vecs[i], vecs[j])
            if math.isfinite(cos_v):
                pair_cos.append(cos_v)
                per_block_max[ids[i]] = max(per_block_max[ids[i]], cos_v)
                per_block_max[ids[j]] = max(per_block_max[ids[j]], cos_v)
    if not pair_cos:
        return {"max_pairwise_cosine": None, "mean_pairwise_cosine": None, "p90_pairwise_cosine": None, "high_redundancy_count": None}
    arr = np.array(pair_cos, dtype=np.float64)
    high_thr = 0.85
    return {
        "max_pairwise_cosine": float(arr.max()),
        "mean_pairwise_cosine": float(arr.mean()),
        "p90_pairwise_cosine": float(np.percentile(arr, 90)),
        "high_redundancy_count": float(sum(1 for v in per_block_max.values() if v >= high_thr)),
    }


def load_sample_history_from_npz(root: Path, index_rows: Sequence[Mapping[str, Any]], task: str, sample_id: str) -> Dict[int, Dict[str, Any]]:
    history: Dict[int, Dict[str, Any]] = {}
    for row in index_rows:
        if str(row.get("task")) != task or str(row.get("sample_id")) != str(sample_id):
            continue
        layer_idx = int(row["layer_idx"])
        npz_rel = str(row["npz"])
        npz_path = root / npz_rel
        if not npz_path.exists():
            continue
        data = np.load(npz_path, allow_pickle=False)
        block_ids = [int(x) for x in data["block_ids"].tolist()]
        history[layer_idx] = {
            "block_ids": block_ids,
            "hidden": data["hidden_block_rep"],
            "key": data["key_block_rep"],
            "local_query": data["local_query_rep"],
            "query_hidden_cosine": [],
            "query_key_cosine": [],
            "scores": [],
            "censored": [bool(x) for x in data["trajectory_censored"].tolist()],
            "rep_available": [bool(x) for x in data["representation_available"].tolist()],
        }
        lq = data["local_query_rep"]
        hq = data["hidden_query_rep"] if "hidden_query_rep" in data else None
        for i, bid in enumerate(block_ids):
            if history[layer_idx]["censored"][i] or not history[layer_idx]["rep_available"][i]:
                history[layer_idx]["query_hidden_cosine"].append(UNAVAILABLE)
                history[layer_idx]["query_key_cosine"].append(UNAVAILABLE)
            else:
                if hq is not None and hq.ndim == 1 and hq.shape[0] == history[layer_idx]["hidden"][i].shape[0]:
                    history[layer_idx]["query_hidden_cosine"].append(_cosine_np(hq, history[layer_idx]["hidden"][i]))
                else:
                    history[layer_idx]["query_hidden_cosine"].append(UNAVAILABLE)
                history[layer_idx]["query_key_cosine"].append(_cosine_np(lq, history[layer_idx]["key"][i]))
            history[layer_idx]["scores"].append(UNAVAILABLE)
    return history
