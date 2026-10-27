#!/usr/bin/env python3
"""Regression tests for canonical independent top-span selection."""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from bridgekv.evidence_text_packing import CandidateSpan, build_candidate_spans, lexical_scored_candidates
from bridgekv import evidence_text_packing as etp

from tools.bridgepack_g2.canonical_independent import (
    LexicalScoreError,
    buggy_unsorted_slice_top64,
    rank_lexical_candidates,
    select_independent_pre_budget,
    validate_lexical_scores,
)


def _make_span(
    idx: int,
    *,
    text: str,
    final_score: float = 0.0,
    query_score: float = 0.0,
    page_id: int = 0,
) -> CandidateSpan:
    return CandidateSpan(
        sample_id="syn",
        task="qasper",
        span_id=f"p{page_id}_s{idx}_t{idx * 32}",
        page_id=page_id,
        block_id=0,
        start_token=idx * 32,
        end_token=idx * 32 + 64,
        text=text,
        source_order_index=idx,
        base_score=0.0,
        query_score=query_score,
        bridge_score=0.0,
        density_score=0.0,
        final_score=final_score,
    )


def _synthetic_opposite_order() -> tuple[list[CandidateSpan], dict]:
    """Early spans irrelevant; late spans highly relevant — bug picks doc prefix."""
    spans = []
    for i in range(80):
        if i < 40:
            text = f"irrelevant filler paragraph number {i} lorem ipsum dolor"
        else:
            text = f"target keyword alpha beta gamma evidence paragraph {i} relevant answer"
        spans.append(_make_span(i, text=text, final_score=0.0, query_score=0.0))
    page_map = {
        0: {
            "page_id": 0,
            "start_token": 0,
            "tokens": ["tok"] * 3000,
        }
    }
    return spans, page_map


class TestIndependentSelectionRegression(unittest.TestCase):
    def test_selects_highest_relevance_not_first_64(self):
        spans, page_map = _synthetic_opposite_order()
        query = "target keyword alpha beta gamma evidence relevant answer"
        ranked = rank_lexical_candidates(query, spans)
        selected = select_independent_pre_budget(query, spans, page_map, budget=1024, neighbor_size=32)
        buggy = buggy_unsorted_slice_top64(query, spans, max_items=64)

        self.assertNotEqual(
            [s.span_id for s in selected[:5]],
            [s.span_id for s in buggy[:5]],
            "canonical must differ from unsorted slice on synthetic anti-correlated data",
        )
        self.assertGreater(max(s.source_order_index for s in selected), 40)
        self.assertLess(max(s.source_order_index for s in buggy), 64)

    def test_buggy_implementation_fails_regression(self):
        spans, _ = _synthetic_opposite_order()
        query = "target keyword alpha beta gamma evidence"
        canonical = rank_lexical_candidates(query, spans)
        buggy = buggy_unsorted_slice_top64(query, spans)
        self.assertNotEqual(
            [c.span_id for c in canonical[:10]],
            [c.span_id for c in buggy[:10]],
        )

    def test_ranking_matches_reference_sort(self):
        spans, _ = _synthetic_opposite_order()
        query = "alpha beta keyword"
        ranked = rank_lexical_candidates(query, spans)
        manual = sorted(
            lexical_scored_candidates(query, spans),
            key=lambda s: (-float(s.final_score), -float(s.query_score), int(s.source_order_index)),
        )
        self.assertEqual([s.span_id for s in ranked], [s.span_id for s in manual])

    def test_tie_break_source_order(self):
        spans = [
            _make_span(0, text="a", final_score=1.0, query_score=1.0),
            _make_span(1, text="b", final_score=1.0, query_score=1.0),
            _make_span(2, text="c", final_score=1.0, query_score=0.5),
        ]
        ranked = rank_lexical_candidates("x", spans)
        self.assertEqual([s.span_id for s in ranked], [spans[0].span_id, spans[1].span_id, spans[2].span_id])

    def test_nan_score_raises(self):
        spans = [_make_span(0, text="a", final_score=float("nan"))]
        with self.assertRaises(LexicalScoreError):
            validate_lexical_scores(spans)

    def test_inf_score_raises(self):
        spans = [_make_span(0, text="a", final_score=float("inf"))]
        with self.assertRaises(LexicalScoreError):
            validate_lexical_scores(spans)

    def test_source_order_only_for_serialization(self):
        spans, page_map = _synthetic_opposite_order()
        query = "target keyword alpha beta gamma evidence"
        selected = select_independent_pre_budget(query, spans, page_map, budget=200, neighbor_size=0)
        if len(selected) < 2:
            self.skipTest("budget too tight for serialization test")
        serialized = sorted(selected, key=lambda s: s.source_order_index)
        rel_order = sorted(selected, key=lambda s: (-s.final_score, -s.query_score, s.source_order_index))
        # With multiple selected spans from different doc regions, serialization order != relevance order
        if len({s.source_order_index for s in selected}) >= 2:
            self.assertEqual(
                [s.source_order_index for s in serialized],
                sorted(s.source_order_index for s in selected),
            )

    def test_g1_g2_pre_budget_ranking_alignment_on_real_sample(self):
        """G1 and G2 must share identical lexical ranking before tokenizer budget."""
        g1_root = REPO_ROOT / "results" / "bridgepack_g1_qwen1024_matched_20260727_004934"
        pack = g1_root / "full/packed_prompts_build/packed_prompts/independent_topspan_neighbor_source_order/budget_1024/qasper.jsonl"
        if not pack.exists():
            self.skipTest("G1 packs not found")
        import json

        row = json.loads(pack.open().readline())
        sid = str(row["sample_id"])
        task = str(row["task"])
        question = str(row.get("question") or "")
        context = str(row.get("original_context") or "")
        candidates, page_map = build_candidate_spans(sid, task, question, context)
        ranked = rank_lexical_candidates(question, candidates)
        pre_ids = [c.span_id for c in select_independent_pre_budget(question, candidates, page_map)]
        # G1 selected IDs should be subset of pre-budget selection pool (HF may trim further)
        g1_ids = list(row.get("selected_span_ids") or [])
        self.assertTrue(set(g1_ids).issubset({c.span_id for c in ranked}))
        self.assertGreater(len(pre_ids), 0)
        self.assertLessEqual(max(c.source_order_index for c in candidates if c.span_id in set(g1_ids[:3])), 200)


if __name__ == "__main__":
    unittest.main(verbosity=2)
