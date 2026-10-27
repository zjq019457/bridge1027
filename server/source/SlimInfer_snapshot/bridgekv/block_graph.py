from typing import Any, Dict, List, Optional

from .semantic_edges import extract_block_features


def build_bridge_graph(
    block_texts: List[str],
    query_text: str,
    bridge_graph_edges: str = "query_semantic_pos",
    bridge_pos_edge_weight: float = 0.05,
    bridge_topk_sem_edges: int = 16,
    qk_scores: Optional[Any] = None,
    bridge_query_edge_weight: float = 1.0,
) -> Dict[str, Any]:
    from .selection_policy import build_bridge_graph as build_graph_impl

    return build_graph_impl(
        block_texts=block_texts,
        query_text=query_text,
        bridge_graph_edges=bridge_graph_edges,
        bridge_pos_edge_weight=bridge_pos_edge_weight,
        bridge_topk_sem_edges=bridge_topk_sem_edges,
        qk_scores=qk_scores,
        bridge_query_edge_weight=bridge_query_edge_weight,
    )
