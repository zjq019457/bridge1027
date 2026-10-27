#!/usr/bin/env python3
"""CPU-only smoke for CC topology operators (no model / no GPU). Do not use as eval."""
from __future__ import annotations

import torch

from bridgekv.cc_topology_policies import (
    cc_gated_active_ids,
    cc_router_active_ids,
    coverage_cc_active_ids,
    diversify_cc_active_ids,
    score_topology_features,
)
from bridgekv.structural_sliminfer import cc_prepare_state, cluster_continuity_active_ids


def main() -> None:
    n = 32
    ids = list(range(n))
    # Cluster in middle + distant second cluster + one spike.
    scores = torch.full((n,), 0.1)
    scores[10:14] = torch.tensor([0.9, 1.0, 0.95, 0.85])
    scores[24:26] = torch.tensor([0.8, 0.75])
    scores[3] = 1.2  # spike
    topk = torch.topk(scores, k=8).indices

    state = cc_prepare_state(
        original_block_ids=ids,
        block_scores=scores,
        top_k_indices=topk,
        must_keep_block_ids={0, n - 1},
        num_blocks_to_keep=8,
    )
    feats = score_topology_features(
        original_block_ids=ids,
        idx_scores=state["idx_scores"],
        smoothed=state["smoothed"],
        clusters=state["clusters"],
        baseline=set(state["baseline"]),
        k=8,
    )
    assert "cluster_strength" in feats
    base, _ = cluster_continuity_active_ids(
        original_block_ids=ids,
        block_scores=scores,
        top_k_indices=topk,
        must_keep_block_ids={0, n - 1},
        num_blocks_to_keep=8,
    )
    gated, gmeta = cc_gated_active_ids(
        original_block_ids=ids,
        block_scores=scores,
        top_k_indices=topk,
        must_keep_block_ids={0, n - 1},
        num_blocks_to_keep=8,
    )
    cov, cmeta = coverage_cc_active_ids(
        original_block_ids=ids,
        block_scores=scores,
        top_k_indices=topk,
        must_keep_block_ids={0, n - 1},
        num_blocks_to_keep=8,
    )
    div, dmeta = diversify_cc_active_ids(
        original_block_ids=ids,
        block_scores=scores,
        top_k_indices=topk,
        must_keep_block_ids={0, n - 1},
        num_blocks_to_keep=8,
    )
    routed, rmeta = cc_router_active_ids(
        original_block_ids=ids,
        block_scores=scores,
        top_k_indices=topk,
        must_keep_block_ids={0, n - 1},
        num_blocks_to_keep=8,
    )
    for name, final in [("base", base), ("gated", gated), ("cov", cov), ("div", div), ("routed", routed)]:
        assert len(final) == 8, (name, len(final), final)
        assert 0 in final and (n - 1) in final
    assert gmeta["mode"] == "cc_gated_v0"
    assert cmeta["mode"] == "cc_coverage_v0"
    assert dmeta["mode"] == "cc_diversify_v0"
    assert rmeta["mode"] == "cc_router_v0"
    assert "policy_chosen" in rmeta
    print("ok", {"features": feats, "router_policy": rmeta["policy_chosen"]})


if __name__ == "__main__":
    main()
