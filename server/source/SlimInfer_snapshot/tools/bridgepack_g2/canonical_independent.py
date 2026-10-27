#!/usr/bin/env python3
"""Canonical independent top-span selection — shared by G1/G2 fix paths."""
from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from bridgekv.bridgepack_cpu_extensions import selection_only_control
from bridgekv.evidence_text_packing import (
    CandidateSpan,
    build_candidate_spans,
    lexical_scored_candidates,
)
from bridgekv import evidence_text_packing as etp

from tools.bridgepack_g1.common import BUDGET
from tools.bridgepack_g1.run_g1 import _exact_pack_from_ordered

INDEP_VARIANT = "independent_topspan_neighbor_source_order"
SELECTION_RULE = "independent_lexical_top"


class LexicalScoreError(RuntimeError):
    """Raised when lexical scores are missing or non-finite."""


def rank_lexical_candidates(
    query: str,
    candidates: Sequence[CandidateSpan],
) -> List[CandidateSpan]:
    """Canonical pre-budget ranking: final_score desc, query_score desc, source_order asc."""
    lexical = lexical_scored_candidates(query, candidates)
    validate_lexical_scores(lexical)
    return sorted(
        lexical,
        key=lambda s: (-float(s.final_score), -float(s.query_score), int(s.source_order_index)),
    )


def validate_lexical_scores(candidates: Sequence[CandidateSpan]) -> None:
    for span in candidates:
        for name, val in (
            ("final_score", span.final_score),
            ("query_score", span.query_score),
        ):
            if val is None:
                raise LexicalScoreError(f"missing {name} for span {span.span_id}")
            fv = float(val)
            if math.isnan(fv) or math.isinf(fv):
                raise LexicalScoreError(f"non-finite {name}={fv} for span {span.span_id}")


def select_independent_pre_budget(
    query: str,
    candidates: Sequence[CandidateSpan],
    page_map: Mapping[int, Mapping[str, Any]],
    budget: int = BUDGET,
    neighbor_size: int = 32,
) -> List[CandidateSpan]:
    """Canonical selection before HF tokenizer budget execution."""
    ranked = rank_lexical_candidates(query, candidates)
    return etp._select_ranked_for_budget(
        query,
        ranked,
        page_map,
        budget,
        neighbor_size=neighbor_size,
        use_tags=False,
    )


def pack_canonical_independent(
    tokenizer,
    *,
    model_tokenizer_path: str,
    sample_id: str,
    task: str,
    question: str,
    context: str,
    answers: Sequence[str],
    all_classes: Any = None,
    variant_name: str = INDEP_VARIANT,
    neighbor_size: int = 32,
    prompt_wrapper_hash: Optional[str] = None,
    decoding_config_hash: Optional[str] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    G1-validated independent path:
    selection_only_control(independent_lexical_top) → source-order serialization → HF exact budget.
    """
    packed_obj, sel_audit = selection_only_control(
        sample_id,
        task,
        question,
        context,
        BUDGET,
        SELECTION_RULE,
        tokenizer=tokenizer,
    )
    candidates, page_map = build_candidate_spans(sample_id, task, question, context)
    by_id = {c.span_id: c for c in candidates}
    selected = [by_id[i] for i in packed_obj.selected_span_ids if i in by_id]
    ordered = [(s, "lexical") for s in sorted(selected, key=lambda x: x.source_order_index)]
    extra: Dict[str, Any] = {
        "matched_family": "selection_only",
        "selection_rule": SELECTION_RULE,
        "answers": list(answers),
        "gold_analysis_only_answers": list(answers),
        "all_classes": all_classes,
        "canonical_selection": True,
    }
    if prompt_wrapper_hash:
        extra["prompt_wrapper_hash"] = prompt_wrapper_hash
    if decoding_config_hash:
        extra["decoding_config_hash"] = decoding_config_hash
    payload = _exact_pack_from_ordered(
        tokenizer,
        sample_id,
        task,
        question,
        context,
        variant_name,
        ordered,
        page_map,
        neighbor_size,
        False,
        "source_order",
        audit_extra=extra,
    )
    payload["model_tokenizer"] = str(model_tokenizer_path)
    payload["ordering"] = "source_order"
    payload["selection_pre_budget_span_ids"] = list(packed_obj.selected_span_ids)
    return payload, sel_audit


def buggy_unsorted_slice_top64(
    query: str,
    candidates: Sequence[CandidateSpan],
    max_items: int = 64,
) -> List[CandidateSpan]:
    """Old G2 bug — must fail regression tests."""
    lexical = lexical_scored_candidates(query, candidates)
    by_id = {c.span_id: c for c in candidates}
    return sorted(
        [c for c in lexical[:max_items] if c.span_id in by_id],
        key=lambda x: x.source_order_index,
    )
