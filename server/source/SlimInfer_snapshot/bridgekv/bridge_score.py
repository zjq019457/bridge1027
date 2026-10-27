import time
from typing import Any, Dict, List, Optional

import torch


def _normalize_vector(vec: torch.Tensor) -> torch.Tensor:
    if vec.numel() == 0:
        return vec
    finite = torch.where(torch.isfinite(vec), vec, torch.zeros_like(vec))
    min_v = finite.min()
    max_v = finite.max()
    denom = (max_v - min_v).clamp_min(1e-6)
    return (finite - min_v) / denom


def _build_transition_matrix(edge_weights: List[List[float]]) -> torch.Tensor:
    mat = torch.tensor(edge_weights, dtype=torch.float32)
    out_sum = mat.sum(dim=1, keepdim=True).clamp_min(1e-9)
    return mat / out_sum


def personalized_pagerank(
    transition_matrix: torch.Tensor,
    personalization: torch.Tensor,
    alpha: float = 0.85,
    max_iter: int = 50,
    tol: float = 1e-5,
) -> torch.Tensor:
    n = transition_matrix.shape[0]
    x = personalization.clone().float()
    teleport = personalization.clone().float()
    for _ in range(max_iter):
        x_next = alpha * (transition_matrix.t() @ x) + (1.0 - alpha) * teleport
        if torch.norm(x_next - x, p=1) < tol:
            return x_next
        x = x_next
    return x


def compute_bridge_scores(
    edge_weights: List[List[float]],
    query_node_ids: List[int],
    evidence_node_ids: Optional[List[int]],
    bridge_score_type: str = "ppr",
    ppr_alpha: float = 0.85,
    max_iter: int = 50,
    tol: float = 1e-5,
) -> Dict[str, Any]:
    start = time.perf_counter()
    num_nodes = len(edge_weights)
    if num_nodes == 0:
        return {
            "bridge_scores": torch.zeros(0, dtype=torch.float32),
            "ppr_time_ms": 0.0,
            "evidence_node_ids": evidence_node_ids or [],
        }

    transition = _build_transition_matrix(edge_weights)
    query_personalization = torch.zeros(num_nodes, dtype=torch.float32)
    if query_node_ids:
        query_personalization[query_node_ids] = 1.0 / len(query_node_ids)
    else:
        query_personalization[:] = 1.0 / num_nodes

    ppr_query = personalized_pagerank(transition, query_personalization, alpha=ppr_alpha, max_iter=max_iter, tol=tol)
    evidence_scores = torch.zeros(num_nodes, dtype=torch.float32)
    if bridge_score_type == "ppr_product" and evidence_node_ids:
        evidence_personalization = torch.zeros(num_nodes, dtype=torch.float32)
        evidence_personalization[evidence_node_ids] = 1.0 / len(evidence_node_ids)
        ppr_evidence = personalized_pagerank(transition, evidence_personalization, alpha=ppr_alpha, max_iter=max_iter, tol=tol)
        bridge_scores = torch.sqrt(ppr_query * ppr_evidence)
    else:
        bridge_scores = ppr_query

    bridge_scores = _normalize_vector(bridge_scores)
    result = {
        "bridge_scores": bridge_scores,
        "ppr_time_ms": (time.perf_counter() - start) * 1000.0,
        "ppr_query": ppr_query,
        "evidence_node_ids": evidence_node_ids or [],
    }
    return result
