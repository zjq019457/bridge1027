#!/usr/bin/env python3
"""CPU-only smoke tests for evidence-aware text packing."""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bridgekv.evidence_text_packing import (  # noqa: E402
    estimate_token_count,
    pack_evidence,
    query_link_score,
)


def _synthetic_context() -> str:
    query_anchor = (
        "Nora Celadon planned the northern expedition with a quiet mapmaker. "
        "The expedition notes say Nora needed the name of the final harbor. "
        "The mapmaker was not named in this first excerpt. "
    )
    bridge = (
        "In a later archive, Nora Celadon worked with Elias Rowan on the same "
        "northern expedition. Elias Rowan carried the bridge ledger and copied "
        "the route notes. "
    )
    support = (
        "Elias Rowan wrote that the route ended at Verdant Harbor after the "
        "winter crossing. The ledger calls Verdant Harbor the final harbor of "
        "Nora's expedition. "
    )
    distractor = (
        "Nora Celadon expedition Nora Celadon expedition northern expedition "
        "mapmaker harbor question repeated many times, but this isolated note "
        "is an unrelated scoreboard with no route ledger or bridge person. "
    )
    filler = (
        "Ordinary background text about kitchens, weather, and unrelated city "
        "records fills the remaining context without useful evidence. "
    )
    return "\n\n".join([query_anchor, bridge, support, distractor, filler])


def main() -> None:
    assert os.environ.get("CUDA_VISIBLE_DEVICES", "") == "", (
        "Smoke test must run with CUDA_VISIBLE_DEVICES=''"
    )
    query = "What final harbor did Nora Celadon's expedition reach?"
    context = _synthetic_context()

    q_distractor = query_link_score(query, "Nora Celadon expedition harbor " * 8)
    q_bridge = query_link_score(query, "Elias Rowan copied the route notes to Verdant Harbor.")
    assert q_distractor > q_bridge, "synthetic distractor should be a high lexical span"

    span_only, _ = pack_evidence(
        "synthetic_0",
        "synthetic",
        query,
        context,
        "span_only_source_order",
        220,
        page_size_tokens=36,
        span_size_tokens=24,
        stride_tokens=24,
    )
    neighbor, _ = pack_evidence(
        "synthetic_0",
        "synthetic",
        query,
        context,
        "neighbor32_source_order",
        360,
        page_size_tokens=36,
        span_size_tokens=24,
        stride_tokens=24,
    )
    bridge, bridge_audit = pack_evidence(
        "synthetic_0",
        "synthetic",
        query,
        context,
        "bridge_neighbor32_path_order",
        360,
        page_size_tokens=36,
        span_size_tokens=24,
        stride_tokens=24,
    )
    tagged, _ = pack_evidence(
        "synthetic_0",
        "synthetic",
        query,
        context,
        "bridge_neighbor32_path_order_tags",
        420,
        page_size_tokens=36,
        span_size_tokens=24,
        stride_tokens=24,
    )

    assert "Verdant Harbor" in bridge.packed_text, "bridge path missed answer-support span"
    assert "Elias Rowan" in bridge.packed_text, "bridge path missed bridge entity"
    assert "unrelated scoreboard" not in bridge.packed_text, (
        "bridge path should avoid the isolated high-score distractor"
    )
    assert bridge.used_bridge_path is True
    assert int(bridge_audit.get("path_count", 0)) > 0
    assert bridge_audit.get("cross_page_path_count", 0) > 0

    assert neighbor.packed_token_count > span_only.packed_token_count, (
        "neighbor expansion should increase context tokens in this synthetic sample"
    )
    assert "[Evidence id=" in tagged.packed_text and "[/Evidence]" in tagged.packed_text
    for packed in (span_only, neighbor, bridge, tagged):
        assert packed.packed_token_count <= packed.budget
        assert estimate_token_count(packed.packed_text) == packed.packed_token_count

    print("SMOKE_EVIDENCE_TEXT_PACKING_PASS")
    print({
        "span_only_tokens": span_only.packed_token_count,
        "neighbor32_tokens": neighbor.packed_token_count,
        "bridge_tokens": bridge.packed_token_count,
        "tagged_tokens": tagged.packed_token_count,
        "bridge_pages": bridge.selected_page_ids,
        "bridge_path_count": bridge_audit.get("path_count", 0),
    })


if __name__ == "__main__":
    main()
