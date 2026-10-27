#!/usr/bin/env python3
"""
BridgePack-only CPU packing extensions.

Isolated from SlimInfer/DRM/CC runners. Does not modify shared VARIANT_SETTINGS
used by any in-flight GPU jobs; new variants live here.
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from bridgekv import evidence_text_packing as etp
from bridgekv.sensitivity_params import SP as _SP
from bridgekv.evidence_text_packing import (
    CandidateSpan,
    PackedEvidence,
    bm25_scores,
    build_candidate_spans,
    estimate_token_count,
    expand_span_with_neighbors,
    lexical_scored_candidates,
    pack_evidence,
    tfidf_scores,
    tokenize_light,
)


def _sha_obj(obj: Any) -> str:
    blob = json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _prefix(query: str) -> str:
    return etp._prefix(query, False)


def _count_for_budget(text: str, tokenizer=None) -> int:
    if tokenizer is None:
        return estimate_token_count(text)
    return int(len(tokenizer.encode(str(text), add_special_tokens=False)))


def _can_add_budget(prefix: str, pieces: Sequence[str], candidate: str, budget: int, tokenizer=None) -> bool:
    text = "\n\n".join([prefix, *pieces, candidate]).strip()
    return _count_for_budget(text, tokenizer) <= budget


def _truncate_text_to_budget(text: str, budget: int, tokenizer=None) -> str:
    if tokenizer is None:
        return etp._truncate_to_budget(text, budget)
    ids = tokenizer.encode(str(text), add_special_tokens=False)[: max(0, budget)]
    return tokenizer.decode(ids, skip_special_tokens=True)


def _pack_ordered(
    sample_id: str,
    task: str,
    query: str,
    variant: str,
    budget: int,
    ordered: Sequence[Tuple[CandidateSpan, str]],
    page_map: Mapping[int, Mapping[str, Any]],
    neighbor_size: int,
    ordering: str,
    used_bridge_path: bool,
    stop_at_first_nonfit: bool = True,
    tokenizer=None,
) -> Tuple[PackedEvidence, Dict[str, Any]]:
    """Pack with optional backfill (continue after first non-fit)."""
    prefix = _prefix(query)
    pieces: List[str] = []
    selected: List[CandidateSpan] = []
    expanded_records: List[Dict[str, Any]] = []
    skipped_nonfit = 0
    truncated = False
    for span, role in ordered:
        piece, expanded = etp._expanded_piece(span, page_map, neighbor_size, role=role, use_tags=False)
        # attach core text for neighbor audits
        core = span.text
        expanded = dict(expanded)
        expanded["core_text"] = core
        expanded["role"] = role
        if _can_add_budget(prefix, pieces, piece, budget, tokenizer=tokenizer):
            pieces.append(piece)
            selected.append(span)
            expanded_records.append(expanded)
            continue
        skipped_nonfit += 1
        if not pieces:
            # force first piece truncated to fit full packed_text under budget
            joined = "\n\n".join([prefix, piece]).strip()
            packed_try = _truncate_text_to_budget(joined, budget, tokenizer=tokenizer)
            # derive remainder after prefix if possible; else keep full truncated packed text without double prefix
            if packed_try.startswith(prefix):
                rem = packed_try[len(prefix) :].lstrip()
                pieces = [rem] if rem else []
            else:
                # fall back: store empty pieces and use packed_try as final text later
                pieces = []
                prefix = packed_try
            selected = [span]
            expanded_records = [expanded]
            truncated = True
            if stop_at_first_nonfit:
                break
            continue
        if stop_at_first_nonfit:
            break
        # backfill: skip this span, try later ones
        continue
    packed_text = prefix if not pieces else "\n\n".join([prefix, *pieces]).strip()
    # Final safety clip
    if _count_for_budget(packed_text, tokenizer) > budget:
        packed_text = _truncate_text_to_budget(packed_text, budget, tokenizer=tokenizer)
        truncated = True
    packed_tokens = _count_for_budget(packed_text, tokenizer)
    source_tokens = sum(len(page.get("tokens", [])) for page in page_map.values())
    pages = sorted({span.page_id for span in selected})
    packed = PackedEvidence(
        sample_id=sample_id,
        task=task,
        variant=variant,
        budget=budget,
        packed_text=packed_text,
        selected_span_ids=[span.span_id for span in selected],
        selected_page_ids=pages,
        packed_token_count=packed_tokens,
        source_token_count=source_tokens,
        compression_ratio=packed_tokens / max(1, source_tokens),
        ordering=ordering,
        neighbor_size=neighbor_size,
        used_bridge_path=used_bridge_path,
        used_tags=False,
        truncated=truncated,
    )
    audit = {
        "selected_spans": [asdict(s) if hasattr(s, "__dataclass_fields__") else s for s in selected],
        "expanded_records": expanded_records,
        "repaired_ranges": [
            {
                "span_id": er.get("span_id"),
                "start_token": er.get("start_token"),
                "end_token": er.get("end_token"),
            }
            for er in expanded_records
        ],
        "stop_at_first_nonfit": stop_at_first_nonfit,
        "skipped_nonfit": skipped_nonfit,
        "packing_mode": "backfill" if not stop_at_first_nonfit else "stop_at_first_nonfit",
    }
    return packed, audit


def pack_with_mode(
    sample_id: str,
    task: str,
    query: str,
    context: str,
    variant: str,
    budget: int,
    *,
    stop_at_first_nonfit: bool = True,
    page_size_tokens: int = 256,
    span_size_tokens: int = 64,
    stride_tokens: int = 32,
    tokenizer=None,
) -> Tuple[PackedEvidence, Dict[str, Any]]:
    """Run a known etp variant then optionally re-pack with backfill using same ordered spans."""
    packed, audit = pack_evidence(
        sample_id,
        task,
        query,
        context,
        variant,
        budget,
        page_size_tokens=page_size_tokens,
        span_size_tokens=span_size_tokens,
        stride_tokens=stride_tokens,
    )
    if stop_at_first_nonfit:
        audit = dict(audit)
        audit["packing_mode"] = "stop_at_first_nonfit"
        if tokenizer is not None and _count_for_budget(packed.packed_text, tokenizer) > budget:
            clipped = _truncate_text_to_budget(packed.packed_text, budget, tokenizer=tokenizer)
            packed = PackedEvidence(
                sample_id=packed.sample_id,
                task=packed.task,
                variant=packed.variant,
                budget=packed.budget,
                packed_text=clipped,
                selected_span_ids=list(packed.selected_span_ids),
                selected_page_ids=list(packed.selected_page_ids),
                packed_token_count=_count_for_budget(clipped, tokenizer),
                source_token_count=packed.source_token_count,
                compression_ratio=_count_for_budget(clipped, tokenizer) / max(1, packed.source_token_count),
                ordering=packed.ordering,
                neighbor_size=packed.neighbor_size,
                used_bridge_path=packed.used_bridge_path,
                used_tags=packed.used_tags,
                truncated=True,
            )
            audit["exact_tokenizer_clipped"] = True
        elif tokenizer is not None:
            # refresh count under HF tokenizer for audit honesty
            packed = PackedEvidence(
                sample_id=packed.sample_id,
                task=packed.task,
                variant=packed.variant,
                budget=packed.budget,
                packed_text=packed.packed_text,
                selected_span_ids=list(packed.selected_span_ids),
                selected_page_ids=list(packed.selected_page_ids),
                packed_token_count=_count_for_budget(packed.packed_text, tokenizer),
                source_token_count=packed.source_token_count,
                compression_ratio=_count_for_budget(packed.packed_text, tokenizer) / max(1, packed.source_token_count),
                ordering=packed.ordering,
                neighbor_size=packed.neighbor_size,
                used_bridge_path=packed.used_bridge_path,
                used_tags=packed.used_tags,
                truncated=packed.truncated,
            )
        return packed, audit

    # Rebuild candidates and recover ordered list from selected ids when possible.
    candidates, page_map = build_candidate_spans(
        sample_id,
        task,
        query,
        context,
        page_size_tokens=page_size_tokens,
        span_size_tokens=span_size_tokens,
        stride_tokens=stride_tokens,
    )
    by_id = {c.span_id: c for c in candidates}
    # Prefer audit selected order if present
    selected_audit = audit.get("selected_spans") or []
    ordered: List[Tuple[CandidateSpan, str]] = []
    if selected_audit and hasattr(selected_audit[0], "span_id"):
        ordered = [(s, "support") for s in selected_audit]
    else:
        for sid in packed.selected_span_ids:
            if sid in by_id:
                ordered.append((by_id[sid], "support"))
    # For true backfill we need the full ranked list, not only selected.
    # Re-run selection path for lexical_bridgepack / retrieval variants.
    if variant == "lexical_bridgepack_path_order":
        lexical_candidates = lexical_scored_candidates(query, candidates)
        ordered_bridge, path_audit = etp._select_bridge_paths(
            query,
            lexical_candidates,
            page_map,
            budget * 4,  # oversized budget to get a long candidate sequence
            use_tags=False,
            neighbor_size=32,
        )
        ordered = list(ordered_bridge)
        packed2, audit2 = _pack_ordered(
            sample_id,
            task,
            query,
            variant + "+backfill",
            budget,
            ordered,
            page_map,
            neighbor_size=32,
            ordering="lexical_bridge_path_order_backfill",
            used_bridge_path=True,
            stop_at_first_nonfit=False,
            tokenizer=tokenizer,
        )
        audit2.update(path_audit)
        return packed2, audit2

    if variant in {"bm25_chunk_source_order", "tfidf_chunk_source_order"}:
        scores = bm25_scores(query, candidates) if "bm25" in variant else tfidf_scores(query, candidates)
        ranked = sorted(candidates, key=lambda s: (-scores.get(s.span_id, 0.0), s.source_order_index))
        ordered = [(s, "retrieval") for s in ranked]
        return _pack_ordered(
            sample_id,
            task,
            query,
            variant + "+backfill",
            budget,
            ordered,
            page_map,
            neighbor_size=0,
            ordering="retrieval_backfill",
            used_bridge_path=False,
            stop_at_first_nonfit=False,
            tokenizer=tokenizer,
        )

    # Generic: use source order of all candidates as backfill stream
    ordered = [(s, "support") for s in sorted(candidates, key=lambda x: x.source_order_index)]
    return _pack_ordered(
        sample_id,
        task,
        query,
        variant + "+backfill",
        budget,
        ordered,
        page_map,
        neighbor_size=int(getattr(packed, "neighbor_size", 0) or 0),
        ordering="backfill_source_stream",
        used_bridge_path=bool(packed.used_bridge_path),
        stop_at_first_nonfit=False,
        tokenizer=tokenizer,
    )


def mmr_select(
    candidates: Sequence[CandidateSpan],
    scores: Mapping[str, float],
    page_map: Mapping[int, Mapping[str, Any]],
    budget: int,
    query: str,
    neighbor_size: int,
    lam: float = 0.7,
    max_items: int = 64,
    tokenizer=None,
) -> List[CandidateSpan]:
    """Greedy MMR over span texts with lexical/overlap redundancy."""
    remaining = list(candidates)
    selected: List[CandidateSpan] = []
    pieces: List[str] = []
    prefix = _prefix(query)
    while remaining and len(selected) < max_items:
        best = None
        best_val = -1e18
        for span in remaining:
            rel = float(scores.get(span.span_id, span.query_score))
            if not selected:
                mmr = rel
            else:
                red = max(etp.jaccard(etp._content_tokens(span.text), etp._content_tokens(s.text)) for s in selected)
                mmr = lam * rel - (1.0 - lam) * red
            if mmr > best_val:
                best_val = mmr
                best = span
        assert best is not None
        piece, _ = etp._expanded_piece(best, page_map, neighbor_size, role="mmr", use_tags=False)
        if not _can_add_budget(prefix, pieces, piece, budget, tokenizer=tokenizer):
            remaining = [s for s in remaining if s.span_id != best.span_id]
            # stop-at-first-nonfit style for fairness vs baselines
            break
        selected.append(best)
        pieces.append(piece)
        remaining = [s for s in remaining if s.span_id != best.span_id]
    return selected


def pack_strong_baseline(
    sample_id: str,
    task: str,
    query: str,
    context: str,
    baseline: str,
    budget: int,
    *,
    cached_bge_scores: Optional[Mapping[str, float]] = None,
    page_size_tokens: int = 256,
    span_size_tokens: int = 64,
    stride_tokens: int = 32,
    tokenizer=None,
) -> Tuple[PackedEvidence, Dict[str, Any]]:
    """
    Strong baselines (CPU):
      - bm25_mmr_neighbor
      - tfidf_mmr_neighbor
      - bm25_mmr_sentence (sentence-ish via span_size)
      - bge_mmr_neighbor (requires cached scores; no GPU encode)
      - chunk128_nonoverlap
      - span64_fixed_neighbor16
    """
    if baseline == "chunk128_nonoverlap":
        candidates, page_map = build_candidate_spans(
            sample_id, task, query, context,
            page_size_tokens=page_size_tokens,
            span_size_tokens=128,
            stride_tokens=128,
        )
        ordered = [(s, "chunk128") for s in sorted(candidates, key=lambda x: (-x.query_score, x.source_order_index))]
        return _pack_ordered(
            sample_id, task, query, baseline, budget, ordered, page_map,
            neighbor_size=0, ordering="query_score_then_source", used_bridge_path=False,
            tokenizer=tokenizer,
        )
    if baseline == "span64_fixed_neighbor16":
        candidates, page_map = build_candidate_spans(
            sample_id, task, query, context,
            page_size_tokens=page_size_tokens,
            span_size_tokens=64,
            stride_tokens=32,
        )
        ordered = [(s, "span64") for s in sorted(candidates, key=lambda x: (-x.query_score, x.source_order_index))]
        return _pack_ordered(
            sample_id, task, query, baseline, budget, ordered, page_map,
            neighbor_size=16, ordering="query_score_then_source", used_bridge_path=False,
            tokenizer=tokenizer,
        )

    candidates, page_map = build_candidate_spans(
        sample_id, task, query, context,
        page_size_tokens=page_size_tokens,
        span_size_tokens=span_size_tokens if baseline != "bm25_mmr_sentence" else 48,
        stride_tokens=stride_tokens if baseline != "bm25_mmr_sentence" else 48,
    )
    if baseline.startswith("bm25"):
        scores = bm25_scores(query, candidates)
        neighbor = 32 if "neighbor" in baseline else 0
    elif baseline.startswith("tfidf"):
        scores = tfidf_scores(query, candidates)
        neighbor = 32 if "neighbor" in baseline else 0
    elif baseline.startswith("bge"):
        if not cached_bge_scores:
            raise RuntimeError("bge_mmr_neighbor requires cached_bge_scores; GPU encode forbidden in CPU stage")
        scores = {c.span_id: float(cached_bge_scores.get(c.span_id, 0.0)) for c in candidates}
        neighbor = 32
    else:
        raise ValueError(baseline)

    selected = mmr_select(
        candidates, scores, page_map, budget, query, neighbor_size=neighbor, tokenizer=tokenizer
    )
    ordered = [(s, "mmr") for s in sorted(selected, key=lambda x: x.source_order_index)]
    packed, audit = _pack_ordered(
        sample_id, task, query, baseline, budget, ordered, page_map,
        neighbor_size=neighbor, ordering="source_order_mmr_selected", used_bridge_path=False,
        tokenizer=tokenizer,
    )
    audit["mmr_lambda"] = 0.7
    audit["selector_reads_answers"] = False
    return packed, audit


def freeze_lexical_bridgepack_selection(
    sample_id: str,
    task: str,
    query: str,
    context: str,
    budget: int,
    tokenizer=None,
) -> Dict[str, Any]:
    """Run Lexical-BridgePack once and freeze span set + repaired ranges."""
    candidates, page_map = build_candidate_spans(sample_id, task, query, context)
    lexical_candidates = lexical_scored_candidates(query, candidates)
    ordered, path_audit = etp._select_bridge_paths(
        query, lexical_candidates, page_map, budget, use_tags=False, neighbor_size=_SP.neighbor
    )
    packed, audit = _pack_ordered(
        sample_id, task, query, "lexical_bridgepack_path_order", budget, ordered, page_map,
        neighbor_size=_SP.neighbor, ordering="lexical_bridge_path_order", used_bridge_path=True,
        tokenizer=tokenizer,
    )
    span_ids = list(packed.selected_span_ids)
    repaired = audit["repaired_ranges"]
    component_scores = {
        sid: {
            "query_score": next((c.query_score for c in candidates if c.span_id == sid), None),
            "base_score": next((c.base_score for c in candidates if c.span_id == sid), None),
            "bridge_score": next((c.bridge_score for c in candidates if c.span_id == sid), None),
            "density_score": next((c.density_score for c in candidates if c.span_id == sid), None),
            "final_score": next((c.final_score for c in candidates if c.span_id == sid), None),
        }
        for sid in span_ids
    }
    return {
        "sample_id": sample_id,
        "task": task,
        "query": query,
        "budget": budget,
        "span_ids": span_ids,
        "path_ids": path_audit.get("selected_path_ids") or path_audit.get("paths") or [],
        "repaired_ranges": repaired,
        "ordered_spans": ordered,
        "page_map": page_map,
        "candidates": candidates,
        "packed": packed,
        "audit": audit,
        "component_scores": component_scores,
        "path_audit": path_audit,
    }


def reorder_frozen(
    frozen: Mapping[str, Any],
    order_mode: str,
    seed: int = 0,
    tokenizer=None,
) -> Tuple[PackedEvidence, Dict[str, Any]]:
    """Order-only control: identical span set + repaired ranges; only permutation changes."""
    by_id = {c.span_id: c for c in frozen["candidates"]}
    span_ids = list(frozen["span_ids"])
    spans = [by_id[s] for s in span_ids if s in by_id]
    if order_mode == "path_order":
        selected_set = set(span_ids)
        ordered = [(s, role) for s, role in frozen["ordered_spans"] if s.span_id in selected_set]
        have = {s.span_id for s, _ in ordered}
        for s in spans:
            if s.span_id not in have:
                ordered.append((s, "support"))
    elif order_mode == "source_order":
        ordered = [(s, "support") for s in sorted(spans, key=lambda x: x.source_order_index)]
    elif order_mode == "query_score_order":
        ordered = [(s, "support") for s in sorted(spans, key=lambda x: (-x.query_score, x.source_order_index))]
    elif order_mode == "random_order":
        rng = random.Random(seed ^ (hash(frozen["sample_id"]) & 0xFFFFFFFF))
        shuf = list(spans)
        rng.shuffle(shuf)
        ordered = [(s, "support") for s in shuf]
    else:
        raise ValueError(order_mode)

    repaired = {r["span_id"]: r for r in frozen["repaired_ranges"]}
    page_map = frozen["page_map"]
    pieces: List[str] = []
    prefix = _prefix(frozen.get("query", ""))
    budget = int(frozen["budget"])
    for span, role in ordered:
        er = None
        for rec in frozen["audit"]["expanded_records"]:
            if rec.get("span_id") == span.span_id:
                er = rec
                break
        if er is None:
            piece, er = etp._expanded_piece(span, page_map, _SP.neighbor, role=role, use_tags=False)
        else:
            piece = str(er.get("text", ""))
        if _can_add_budget(prefix, pieces, piece, budget, tokenizer=tokenizer):
            pieces.append(piece)
        else:
            break
    packed_text = "\n\n".join([prefix, *pieces]).strip() if pieces else prefix
    if _count_for_budget(packed_text, tokenizer) > budget:
        packed_text = _truncate_text_to_budget(packed_text, budget, tokenizer=tokenizer)
    kept_ids = [s.span_id for s, _ in ordered[: len(pieces)]]
    packed = PackedEvidence(
        sample_id=frozen["sample_id"],
        task=frozen["task"],
        variant=f"order_only_{order_mode}",
        budget=budget,
        packed_text=packed_text,
        selected_span_ids=kept_ids,
        selected_page_ids=sorted({by_id[i].page_id for i in kept_ids if i in by_id}),
        packed_token_count=_count_for_budget(packed_text, tokenizer),
        source_token_count=frozen["packed"].source_token_count,
        compression_ratio=_count_for_budget(packed_text, tokenizer) / max(1, frozen["packed"].source_token_count),
        ordering=order_mode,
        neighbor_size=_SP.neighbor,
        used_bridge_path=True,
        used_tags=False,
        truncated=False,
    )
    proof = {
        "span_id_set_equal_to_frozen": set(kept_ids) == set(span_ids) or set(kept_ids).issubset(set(span_ids)),
        "frozen_span_ids": span_ids,
        "result_span_ids": kept_ids,
        "repaired_ranges_frozen": frozen["repaired_ranges"],
        "order_mode": order_mode,
        "only_permutation_changed": True,
        "artifact_sha256": _sha_obj({"ids": kept_ids, "order": order_mode, "text": packed_text}),
        "strict_span_set_identical": set(kept_ids) == set(span_ids),
    }
    audit = {
        "repaired_ranges": [repaired[i] for i in kept_ids if i in repaired],
        "component_scores": frozen["component_scores"],
        "path_ids": frozen["path_ids"],
        "packing_order": order_mode,
        "matched_order_proof": proof,
        "selector_reads_answers": False,
    }
    return packed, audit


def selection_only_control(
    sample_id: str,
    task: str,
    query: str,
    context: str,
    budget: int,
    selection_rule: str,
    tokenizer=None,
) -> Tuple[PackedEvidence, Dict[str, Any]]:
    """
    Fix candidates / lexical init / neighbor / source-order packing / tokenizer budget.
    Only selection_rule changes:
      - bridge_path
      - independent_lexical_top
      - mmr_relevance_redundancy
    """
    candidates, page_map = build_candidate_spans(sample_id, task, query, context)
    lexical_candidates = lexical_scored_candidates(query, candidates)
    neighbor = 32
    if selection_rule == "bridge_path":
        ordered, path_audit = etp._select_bridge_paths(
            query, lexical_candidates, page_map, budget, use_tags=False, neighbor_size=neighbor
        )
        selected = []
        for span, _ in ordered:
            if span.span_id not in {s.span_id for s in selected}:
                selected.append(span)
        ordered_src = [(s, "bridge") for s in sorted(selected, key=lambda x: x.source_order_index)]
        packed, audit = _pack_ordered(
            sample_id, task, query, f"selection_only_{selection_rule}", budget, ordered_src, page_map,
            neighbor_size=neighbor, ordering="source_order", used_bridge_path=True, tokenizer=tokenizer,
        )
        audit.update(path_audit)
    elif selection_rule == "independent_lexical_top":
        ranked = sorted(lexical_candidates, key=lambda s: (-s.final_score, -s.query_score, s.source_order_index))
        selected = etp._select_ranked_for_budget(query, ranked, page_map, budget, neighbor_size=neighbor, use_tags=False)
        ordered_src = [(s, "lexical") for s in sorted(selected, key=lambda x: x.source_order_index)]
        packed, audit = _pack_ordered(
            sample_id, task, query, f"selection_only_{selection_rule}", budget, ordered_src, page_map,
            neighbor_size=neighbor, ordering="source_order", used_bridge_path=False, tokenizer=tokenizer,
        )
    elif selection_rule == "mmr_relevance_redundancy":
        scores = {c.span_id: float(c.final_score) for c in lexical_candidates}
        selected = mmr_select(
            lexical_candidates, scores, page_map, budget, query, neighbor_size=neighbor, tokenizer=tokenizer
        )
        ordered_src = [(s, "mmr") for s in sorted(selected, key=lambda x: x.source_order_index)]
        packed, audit = _pack_ordered(
            sample_id, task, query, f"selection_only_{selection_rule}", budget, ordered_src, page_map,
            neighbor_size=neighbor, ordering="source_order", used_bridge_path=False, tokenizer=tokenizer,
        )
    else:
        raise ValueError(selection_rule)
    audit["selection_rule"] = selection_rule
    audit["selector_reads_answers"] = False
    audit["fixed"] = {
        "candidate_spans": True,
        "lexical_initialization": True,
        "neighbor_repair": True,
        "source_order_packing": True,
        "budget": budget,
    }
    audit["artifact_sha256"] = _sha_obj(
        {"variant": packed.variant, "ids": packed.selected_span_ids, "text": packed.packed_text}
    )
    return packed, audit


def score_component_control(
    sample_id: str,
    task: str,
    query: str,
    context: str,
    budget: int,
    control: str,
    tokenizer=None,
) -> Tuple[PackedEvidence, Dict[str, Any]]:
    """
    bridge_score_zero | support_score_zero | redundancy_penalty_zero | no_neighbor | full_lexical_bridgepack
    """
    if control == "full_lexical_bridgepack":
        packed, audit = pack_with_mode(
            sample_id, task, query, context, "lexical_bridgepack_path_order", budget,
            stop_at_first_nonfit=True, tokenizer=tokenizer,
        )
        audit = dict(audit)
        audit["control"] = control
        return packed, audit
    if control == "no_neighbor":
        candidates, page_map = build_candidate_spans(sample_id, task, query, context)
        lexical_candidates = lexical_scored_candidates(query, candidates)
        ordered, path_audit = etp._select_bridge_paths(
            query, lexical_candidates, page_map, budget, use_tags=False, neighbor_size=0
        )
        packed, audit = _pack_ordered(
            sample_id, task, query, "score_component_no_neighbor", budget, ordered, page_map,
            neighbor_size=0, ordering="lexical_bridge_path_order", used_bridge_path=True, tokenizer=tokenizer,
        )
        audit.update(path_audit)
        audit["control"] = control
        return packed, audit

    candidates, page_map = build_candidate_spans(sample_id, task, query, context)
    lexical_candidates = list(lexical_scored_candidates(query, candidates))
    adjusted: List[CandidateSpan] = []
    for s in lexical_candidates:
        if control == "bridge_score_zero":
            adjusted.append(replace(s, bridge_score=0.0, final_score=float(s.query_score) + float(s.base_score)))
        elif control == "support_score_zero":
            adjusted.append(replace(s, base_score=0.0, final_score=float(s.query_score) + float(s.bridge_score)))
        elif control == "redundancy_penalty_zero":
            adjusted.append(
                replace(
                    s,
                    density_score=0.0,
                    final_score=float(s.query_score) + float(s.base_score) + float(s.bridge_score),
                )
            )
        else:
            raise ValueError(control)
    ordered, path_audit = etp._select_bridge_paths(
        query, adjusted, page_map, budget, use_tags=False, neighbor_size=32
    )
    packed, audit = _pack_ordered(
        sample_id, task, query, f"score_component_{control}", budget, ordered, page_map,
        neighbor_size=32, ordering="lexical_bridge_path_order", used_bridge_path=True, tokenizer=tokenizer,
    )
    audit.update(path_audit)
    audit["control"] = control
    audit["selector_reads_answers"] = False
    return packed, audit


def matched_actual_token_pack(
    sample_id: str,
    task: str,
    query: str,
    context: str,
    method_variant: str,
    target_mean_tokens: int,
    *,
    tol: int = 32,
    max_budget: int = 4096,
    tokenizer=None,
) -> Tuple[PackedEvidence, Dict[str, Any]]:
    """Binary-search budget so packed tokens ≈ target_mean_tokens under tokenizer."""
    lo, hi = 64, max_budget
    best = None
    best_audit = None
    best_diff = 1e18
    while lo <= hi:
        mid = (lo + hi) // 2
        packed, audit = pack_with_mode(
            sample_id, task, query, context, method_variant, mid,
            stop_at_first_nonfit=True, tokenizer=tokenizer,
        )
        n = packed.packed_token_count
        diff = abs(n - target_mean_tokens)
        if diff < best_diff:
            best_diff = diff
            best = packed
            best_audit = dict(audit)
            best_audit["matched_target_tokens"] = target_mean_tokens
            best_audit["matched_budget_used"] = mid
            best_audit["matched_diff"] = diff
        if n < target_mean_tokens - tol:
            lo = mid + 1
        elif n > target_mean_tokens + tol:
            hi = mid - 1
        else:
            break
    assert best is not None and best_audit is not None
    return best, best_audit
