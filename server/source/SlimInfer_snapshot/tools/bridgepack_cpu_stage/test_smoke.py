#!/usr/bin/env python3
"""Unit + synthetic smoke tests for BridgePack CPU stage."""
from __future__ import annotations

import os
import unittest

os.environ["CUDA_VISIBLE_DEVICES"] = ""

from bridgekv.bridgepack_cpu_extensions import (
    freeze_lexical_bridgepack_selection,
    pack_strong_baseline,
    pack_with_mode,
    reorder_frozen,
    selection_only_control,
)
from bridgekv.evidence_text_packing import estimate_token_count


SYN_CONTEXT = (
    "Albert Einstein was born in Ulm. "
    "Ulm is a city in Germany. "
    "Marie Curie worked on radioactivity. "
    "The theory of relativity was developed by Einstein. "
) * 40
SYN_QUERY = "Where was Albert Einstein born?"


class BridgePackCPUStageTests(unittest.TestCase):
    def test_cuda_disabled(self):
        try:
            import torch

            self.assertFalse(torch.cuda.is_available())
        except ImportError:
            pass

    def test_backfill_not_over_regex_budget(self):
        for stop in (True, False):
            packed, audit = pack_with_mode(
                "syn1",
                "hotpotqa",
                SYN_QUERY,
                SYN_CONTEXT,
                "lexical_bridgepack_path_order",
                128,
                stop_at_first_nonfit=stop,
            )
            self.assertLessEqual(estimate_token_count(packed.packed_text), 128)
            self.assertLessEqual(packed.packed_token_count, 128)
            self.assertFalse(audit.get("selector_reads_answers"))

    def test_order_only_span_set(self):
        frozen = freeze_lexical_bridgepack_selection("syn1", "hotpotqa", SYN_QUERY, SYN_CONTEXT, 160)
        packed, audit = reorder_frozen(frozen, "source_order")
        self.assertTrue(audit["matched_order_proof"]["only_permutation_changed"])
        # path_order identity
        packed_p, audit_p = reorder_frozen(frozen, "path_order")
        self.assertEqual(set(packed_p.selected_span_ids), set(frozen["span_ids"]))

    def test_selection_only_no_answer_leak_flag(self):
        packed, audit = selection_only_control(
            "syn1", "hotpotqa", SYN_QUERY, SYN_CONTEXT, 160, "independent_lexical_top"
        )
        self.assertFalse(audit.get("selector_reads_answers"))
        self.assertLessEqual(packed.packed_token_count, 160)

    def test_strong_baseline_bm25_mmr(self):
        packed, audit = pack_strong_baseline(
            "syn1", "hotpotqa", SYN_QUERY, SYN_CONTEXT, "bm25_mmr_neighbor", 160
        )
        self.assertLessEqual(estimate_token_count(packed.packed_text), 160)
        self.assertFalse(audit.get("selector_reads_answers"))


if __name__ == "__main__":
    unittest.main()
