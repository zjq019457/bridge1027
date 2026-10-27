import math
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
from .bridge_score import compute_bridge_scores
from .semantic_edges import build_semantic_edge_weights, extract_block_features
from .structural_risk import compute_structural_scores


def _normalize_tensor(vec: torch.Tensor, mode: str = "per_layer") -> torch.Tensor:
    if mode == "none":
        return torch.where(torch.isfinite(vec), vec, torch.zeros_like(vec))
    if vec.numel() == 0:
        return vec
    finite = torch.where(torch.isfinite(vec), vec, torch.zeros_like(vec))
    if mode == "minmax" or mode == "per_layer":
        min_v = finite.min()
        max_v = finite.max()
        denom = (max_v - min_v).clamp_min(1e-6)
        return (finite - min_v) / denom
    if mode == "zscore":
        mean = finite.mean()
        std = finite.std().clamp_min(1e-6)
        return (finite - mean) / std
    return finite


def build_bridge_graph(
    block_texts: List[str],
    query_text: str,
    bridge_graph_edges: str = "query_semantic_pos",
    bridge_pos_edge_weight: float = 0.05,
    bridge_topk_sem_edges: int = 16,
    qk_scores: Optional[torch.Tensor] = None,
    bridge_query_edge_weight: float = 1.0,
) -> Dict[str, Any]:
    start = time.perf_counter()
    num_blocks = len(block_texts)
    token_features = [extract_block_features(text) for text in block_texts]
    query_tokens = set(extract_block_features(query_text)["tokens"])

    edge_weights = [[0.0 for _ in range(num_blocks)] for _ in range(num_blocks)]
    query_node_ids = [num_blocks - 1] if num_blocks > 0 else []

    qk_norm = None
    if qk_scores is not None and qk_scores.numel() > 0:
        qk_norm = qk_scores.detach().float()
        qk_norm = (qk_norm - qk_norm.min()).clamp_min(0.0)
        qk_denom = qk_norm.max().clamp_min(1e-6)
        qk_norm = qk_norm / qk_denom

    if bridge_graph_edges in {"query_only", "query_semantic", "query_semantic_pos", "query_semantic_pos_attention"}:
        for idx, features in enumerate(token_features):
            overlap = 0.0
            if query_tokens:
                overlap = len(features["tokens"] & query_tokens) / max(1, len(query_tokens))
            qk_val = float(qk_norm[idx]) if qk_norm is not None else 0.0
            weight = max(overlap * 0.5, qk_val * bridge_query_edge_weight)
            edge_weights[num_blocks - 1][idx] = float(weight)

    if bridge_graph_edges in {"query_semantic", "query_semantic_pos", "query_semantic_pos_attention"}:
        semantic_weights = build_semantic_edge_weights(token_features, topk=bridge_topk_sem_edges)
        for i, neighbors in enumerate(semantic_weights):
            for j, w in neighbors.items():
                edge_weights[i][j] += w

    if bridge_graph_edges in {"query_semantic_pos", "query_semantic_pos_attention"}:
        for i in range(num_blocks - 1):
            edge_weights[i][i + 1] += bridge_pos_edge_weight
            edge_weights[i + 1][i] += bridge_pos_edge_weight

    graph_summary = {
        "num_blocks": num_blocks,
        "num_edges_total": sum(1 for i in range(num_blocks) for j in range(num_blocks) if edge_weights[i][j] > 0.0),
        "ppr_alpha": None,
        "graph_build_time_ms": (time.perf_counter() - start) * 1000.0,
        "bridge_graph_edges": bridge_graph_edges,
        "bridge_pos_edge_weight": bridge_pos_edge_weight,
        "bridge_topk_sem_edges": bridge_topk_sem_edges,
        "bridge_query_edge_weight": bridge_query_edge_weight,
    }
    return {
        "num_blocks": num_blocks,
        "edge_weights": edge_weights,
        "query_node_ids": query_node_ids,
        "graph_summary": graph_summary,
        "query_text": query_text,
        "block_texts": block_texts,
    }


def select_pseudo_evidence_nodes(
    qk_scores: torch.Tensor,
    delta_o_scores: Optional[torch.Tensor],
    pseudo_evidence_ratio: float = 0.10,
) -> List[int]:
    if qk_scores.numel() == 0:
        return []
    qk_norm = _normalize_tensor(qk_scores, "minmax")
    combined = qk_norm
    if delta_o_scores is not None and delta_o_scores.numel() == qk_scores.numel():
        combined = (qk_norm + _normalize_tensor(delta_o_scores, "minmax")) / 2.0
    k = max(1, math.ceil(qk_scores.numel() * pseudo_evidence_ratio))
    _, indices = torch.topk(combined, k=k)
    return [int(i) for i in indices.tolist()]


def compute_bridge_utilities(
    state: Dict[str, Any],
    qk_scores: torch.Tensor,
    delta_o_values: Optional[torch.Tensor],
    args: Any,
) -> Dict[str, Any]:
    if state is None:
        return {}
    local_scores = torch.zeros_like(qk_scores)
    if args.local_score == "delta_o" and delta_o_values is not None:
        local_scores = _normalize_tensor(delta_o_values, args.bridge_norm)
    elif args.local_score == "qk":
        local_scores = _normalize_tensor(qk_scores, args.bridge_norm)

    structural_scores = torch.tensor(state.get("structural_scores", [0.0] * len(qk_scores)), dtype=torch.float32)
    structural_scores = _normalize_tensor(structural_scores, args.bridge_norm)

    evidence_ids = select_pseudo_evidence_nodes(
        qk_scores=qk_scores,
        delta_o_scores=delta_o_values,
        pseudo_evidence_ratio=args.bridge_pseudo_evidence_ratio,
    )
    bridge_result = compute_bridge_scores(
        edge_weights=state["edge_weights"],
        query_node_ids=state["query_node_ids"],
        evidence_node_ids=evidence_ids,
        bridge_score_type=args.bridge_score,
        ppr_alpha=args.bridge_ppr_alpha,
    )

    bridge_scores = bridge_result["bridge_scores"]
    final_utility = (
        args.bridge_alpha * local_scores
        + args.bridge_beta * bridge_scores
        + args.bridge_eta * structural_scores
    )
    final_utility = _normalize_tensor(final_utility, args.bridge_norm)

    return {
        "local_scores": local_scores,
        "bridge_scores": bridge_scores,
        "structural_scores": structural_scores,
        "final_utility": final_utility,
        "bridge_ppr_time_ms": bridge_result["ppr_time_ms"],
        "bridge_evidence_node_ids": evidence_ids,
        "bridge_score_type": args.bridge_score,
        "bridge_summary": bridge_result,
    }
