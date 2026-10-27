"""Unified rescue-candidate builder for the A-F block-system diagnostic.

Design contract (this round only):
- The fixed SlimInfer active set is NEVER changed by this module. It only builds
  *rescue candidates*.
- Every candidate is expressed as a token span and the set of fixed 64-token KV
  pages it expands to. The real attention unit stays the fixed page, so the
  selection cost is always the incremental fixed-page cost.
- Candidates can come from several sources (fixed / overlap / neighbor /
  hierarchy / page_aligned_semantic / evidence / fusion). The selector downstream
  scores them with DRM + length penalty (+ optional query / semantic bonus) and
  picks a token-budget-matched subset.

The output is a ``RescueCandidateSet`` whose ``to_dict()`` is JSON-serialisable so
it can be threaded through ``sliminfer_config['rescue_candidate_layout']`` exactly
like the existing ``block_layout`` payload.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:  # reuse the mature semantic parser instead of re-implementing it
    from .block_layout import build_blocks, extract_query_text
except Exception:  # pragma: no cover - fallback when imported oddly
    build_blocks = None  # type: ignore

    def extract_query_text(sample, prompt_text=None, query_text=None):  # type: ignore
        if query_text:
            return str(query_text), "provided", ""
        sample = sample or {}
        for key in ("question", "query"):
            val = sample.get(key)
            if val:
                return str(val), f"sample.{key}", ""
        return "", "none", "no_query_text"


_PAGE_TOKENS = 64
_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
_STOPWORDS = {
    "the", "a", "an", "of", "to", "in", "on", "for", "and", "or", "is", "are",
    "was", "were", "be", "by", "with", "as", "at", "that", "this", "it", "its",
    "what", "which", "who", "whom", "whose", "when", "where", "how", "why",
    "do", "does", "did", "from", "about", "into", "than", "then", "there",
}
_STRATEGIES = {
    "fixed",
    "overlap",
    "neighbor",
    "hierarchy",
    "page_aligned_semantic",
    "evidence_centered",
    "fusion_fine",
    "fusion_fine_no_query",
    "unified",
}


@dataclass
class RescueCandidate:
    candidate_id: int
    strategy_source: str
    token_span: Tuple[int, int]
    covered_fixed_pages: List[int]
    expanded_page_tokens: int
    semantic_span_tokens: int
    text_preview: str = ""
    query_score: float = 0.0
    semantic_source: str = ""
    coarse_region_id: int = -1
    neighbor_source_page: int = -1
    fallback_used: bool = False
    fallback_reason: str = ""
    query_covered_idx: List[int] = field(default_factory=list)
    answer_score: float = 0.0


@dataclass
class RescueCandidateSet:
    strategy: str
    candidates: List[RescueCandidate]
    page_tokens: int = _PAGE_TOKENS
    prompt_len: int = 0
    fallback_used: bool = False
    fallback_reason: str = ""
    fallback_reason_counts: Dict[str, int] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy_name": self.strategy,
            "page_tokens": int(self.page_tokens),
            "prompt_len": int(self.prompt_len),
            "fallback_used": bool(self.fallback_used),
            "fallback_reason": str(self.fallback_reason),
            "fallback_reason_counts": {k: int(v) for k, v in self.fallback_reason_counts.items()},
            "metadata": self.metadata,
            "token_spans": [[int(c.token_span[0]), int(c.token_span[1])] for c in self.candidates],
            "mapping_to_fixed_pages": [[int(p) for p in c.covered_fixed_pages] for c in self.candidates],
            "source_types": [c.strategy_source for c in self.candidates],
            "block_text_preview": [c.text_preview[:160] for c in self.candidates],
            "expanded_page_tokens": [int(c.expanded_page_tokens) for c in self.candidates],
            "semantic_span_tokens": [int(c.semantic_span_tokens) for c in self.candidates],
            "query_score": [float(c.query_score) for c in self.candidates],
            "semantic_source": [c.semantic_source for c in self.candidates],
            "coarse_region_id": [int(c.coarse_region_id) for c in self.candidates],
            "neighbor_source_page": [int(c.neighbor_source_page) for c in self.candidates],
            "fallback_used_per_candidate": [bool(c.fallback_used) for c in self.candidates],
            "query_covered_idx": [[int(x) for x in c.query_covered_idx] for c in self.candidates],
            "answer_score": [float(c.answer_score) for c in self.candidates],
            "query_vocab_size": int(self.metadata.get("query_vocab_size", 0)),
        }


def _as_token_list(input_ids: Any) -> List[int]:
    if hasattr(input_ids, "detach"):
        return [int(x) for x in input_ids.detach().cpu().reshape(-1).tolist()]
    if isinstance(input_ids, (list, tuple)):
        if input_ids and isinstance(input_ids[0], (list, tuple)):
            return [int(x) for x in input_ids[0]]
        return [int(x) for x in input_ids]
    return []


def _pages_for_span(start: int, end: int, prompt_len: int, page_tokens: int) -> List[int]:
    if end <= start:
        return []
    first = max(0, int(start)) // page_tokens
    last = max(0, int(end) - 1) // page_tokens
    max_page = max(0, (max(1, int(prompt_len)) - 1) // page_tokens)
    return [int(x) for x in range(first, min(last, max_page) + 1)]


def _span_for_pages(pages: Sequence[int], prompt_len: int, page_tokens: int) -> Tuple[int, int]:
    if not pages:
        return (0, 0)
    start = max(0, min(int(p) for p in pages) * page_tokens)
    end = min(int(prompt_len), (max(int(p) for p in pages) + 1) * page_tokens)
    return start, end


def _decode(tokenizer: Any, ids: Sequence[int]) -> str:
    if tokenizer is None:
        return ""
    try:
        return tokenizer.decode(list(ids), skip_special_tokens=True, clean_up_tokenization_spaces=False)
    except Exception:
        return ""


def _make_candidate(
    cid: int,
    source: str,
    pages: Sequence[int],
    prompt_len: int,
    page_tokens: int,
    *,
    semantic_span_tokens: Optional[int] = None,
    text_preview: str = "",
    query_score: float = 0.0,
    semantic_source: str = "",
    coarse_region_id: int = -1,
    neighbor_source_page: int = -1,
) -> Optional[RescueCandidate]:
    pages = sorted({int(p) for p in pages})
    if not pages:
        return None
    start, end = _span_for_pages(pages, prompt_len, page_tokens)
    if end <= start:
        return None
    span_tokens = semantic_span_tokens if semantic_span_tokens is not None else (end - start)
    return RescueCandidate(
        candidate_id=int(cid),
        strategy_source=source,
        token_span=(int(start), int(end)),
        covered_fixed_pages=pages,
        expanded_page_tokens=int(len(pages) * page_tokens),
        semantic_span_tokens=int(max(1, span_tokens)),
        text_preview=text_preview,
        query_score=float(query_score),
        semantic_source=semantic_source,
        coarse_region_id=int(coarse_region_id),
        neighbor_source_page=int(neighbor_source_page),
    )


def _query_word_set(query_text: str) -> set:
    return {w.lower() for w in _TOKEN_RE.findall(query_text or "") if len(w) > 1}


def _page_query_scores(
    tokens: List[int],
    tokenizer: Any,
    query_text: str,
    num_pages: int,
    page_tokens: int,
) -> List[float]:
    """Lexical (overlap + light idf) query relevance per fixed page."""
    qwords = _query_word_set(query_text)
    scores = [0.0] * num_pages
    if not qwords:
        return scores
    for page in range(num_pages):
        seg = tokens[page * page_tokens:(page + 1) * page_tokens]
        text = _decode(tokenizer, seg)
        words = [w.lower() for w in _TOKEN_RE.findall(text)]
        if not words:
            continue
        wset = set(words)
        overlap = len(qwords & wset)
        if overlap:
            # normalise by query size; mild length damping
            scores[page] = float(overlap) / float(max(1, len(qwords))) / math.log(2 + len(words))
    return scores


# --------------------------------------------------------------------------
# individual strategies
# --------------------------------------------------------------------------

def _build_fixed(num_pages: int, prompt_len: int, page_tokens: int) -> List[RescueCandidate]:
    out: List[RescueCandidate] = []
    for page in range(num_pages):
        cand = _make_candidate(len(out), "fixed", [page], prompt_len, page_tokens)
        if cand:
            out.append(cand)
    return out


def _build_overlap(num_pages: int, prompt_len: int, page_tokens: int, window: int, stride: int) -> List[RescueCandidate]:
    out: List[RescueCandidate] = []
    seen: set = set()
    # fixed pages are always available as 1-page candidates
    for page in range(num_pages):
        cand = _make_candidate(len(out), "fixed", [page], prompt_len, page_tokens)
        if cand:
            out.append(cand)
            seen.add(tuple(cand.covered_fixed_pages))
    if window <= 0 or stride <= 0:
        return out
    pos = 0
    while pos < prompt_len:
        start = pos
        end = min(prompt_len, pos + window)
        pages = _pages_for_span(start, end, prompt_len, page_tokens)
        key = tuple(pages)
        if pages and key not in seen:
            cand = _make_candidate(len(out), "overlap", pages, prompt_len, page_tokens, semantic_span_tokens=end - start)
            if cand:
                out.append(cand)
                seen.add(key)
        if end >= prompt_len:
            break
        pos += stride
    return out


def _build_neighbor(num_pages: int, prompt_len: int, page_tokens: int) -> List[RescueCandidate]:
    out: List[RescueCandidate] = []
    seen: set = set()
    for page in range(num_pages):
        cand = _make_candidate(len(out), "fixed", [page], prompt_len, page_tokens)
        if cand:
            out.append(cand)
            seen.add(tuple(cand.covered_fixed_pages))
    # neighbor-expanded 3-page windows centred on each page; the selector / length
    # penalty decides whether expanding is worth the extra cost.
    for page in range(num_pages):
        grp = [p for p in (page - 1, page, page + 1) if 0 <= p < num_pages]
        key = tuple(grp)
        if len(grp) > 1 and key not in seen:
            cand = _make_candidate(len(out), "neighbor", grp, prompt_len, page_tokens, neighbor_source_page=page)
            if cand:
                out.append(cand)
                seen.add(key)
    return out


def _build_hierarchy(
    num_pages: int,
    prompt_len: int,
    page_tokens: int,
    coarse_tokens: int,
    fine_tokens: int,
    medium_tokens: int,
) -> List[RescueCandidate]:
    out: List[RescueCandidate] = []
    seen: set = set()
    coarse_pages = max(1, int(round(coarse_tokens / page_tokens)))
    fine_pages = max(1, int(round(fine_tokens / page_tokens)))
    medium_pages = max(1, int(round(medium_tokens / page_tokens)))
    all_pages = list(range(num_pages))
    coarse_groups = [all_pages[i:i + coarse_pages] for i in range(0, num_pages, coarse_pages)]
    for region_id, group in enumerate(coarse_groups):
        # coarse-region candidate (whole region)
        cand = _make_candidate(len(out), "hierarchy_coarse", group, prompt_len, page_tokens, coarse_region_id=region_id)
        if cand and tuple(cand.covered_fixed_pages) not in seen:
            out.append(cand)
            seen.add(tuple(cand.covered_fixed_pages))
        # medium chunks within the region
        for i in range(0, len(group), medium_pages):
            sub = group[i:i + medium_pages]
            cand = _make_candidate(len(out), "hierarchy_medium", sub, prompt_len, page_tokens, coarse_region_id=region_id)
            if cand and tuple(cand.covered_fixed_pages) not in seen:
                out.append(cand)
                seen.add(tuple(cand.covered_fixed_pages))
        # fine chunks within the region
        for i in range(0, len(group), fine_pages):
            sub = group[i:i + fine_pages]
            cand = _make_candidate(len(out), "hierarchy_fine", sub, prompt_len, page_tokens, coarse_region_id=region_id)
            if cand and tuple(cand.covered_fixed_pages) not in seen:
                out.append(cand)
                seen.add(tuple(cand.covered_fixed_pages))
    return out


def _build_page_aligned_semantic(
    input_ids: Any,
    tokenizer: Any,
    sample: Optional[Dict[str, Any]],
    task_name: str,
    prompt_text: Optional[str],
    query_text: str,
    num_pages: int,
    prompt_len: int,
    page_tokens: int,
    config: Dict[str, Any],
) -> Tuple[List[RescueCandidate], bool, str]:
    """Snap semantic units (sentence/paragraph/document) to fixed-page boundaries."""
    fallback_used = False
    fallback_reason = ""
    semantic_layout = None
    if build_blocks is not None:
        try:
            semantic_layout = build_blocks(
                input_ids,
                tokenizer,
                sample if isinstance(sample, dict) else {},
                prompt_text=prompt_text,
                task_name=task_name,
                block_strategy="semantic_v2",
                target_block_tokens=int(config.get("semantic_target_block_tokens", 64) or 64),
                min_block_tokens=int(config.get("semantic_min_block_tokens", 32) or 32),
                max_block_tokens=int(config.get("semantic_max_block_tokens", 128) or 128),
                query_text=query_text,
                config={
                    "block_size": page_tokens,
                    "fixed_page_tokens": page_tokens,
                    "semantic_page_align": True,
                },
            )
        except Exception as exc:  # pragma: no cover
            fallback_used = True
            fallback_reason = f"semantic_build_error:{type(exc).__name__}"
            semantic_layout = None
    else:
        fallback_used = True
        fallback_reason = "build_blocks_unavailable"

    out: List[RescueCandidate] = []
    seen: set = set()
    # always include fixed pages so semantic rescue never loses coverage options
    for page in range(num_pages):
        cand = _make_candidate(len(out), "fixed", [page], prompt_len, page_tokens)
        if cand:
            out.append(cand)
            seen.add(tuple(cand.covered_fixed_pages))

    if semantic_layout is None:
        return out, True, fallback_reason or "semantic_layout_none"

    spans = semantic_layout.token_spans or []
    sources = semantic_layout.source_types or []
    tolerance = int(config.get("semantic_page_tolerance", 16) or 16)
    for idx, span in enumerate(spans):
        try:
            s, e = int(span[0]), int(span[1])
        except Exception:
            continue
        pages = _pages_for_span(s, e, prompt_len, page_tokens)
        if not pages:
            continue
        # Page-aligned semantic rescue: real attention unit is the page. We keep
        # the covering pages, but if the semantic span is far inside a page
        # boundary (within tolerance), we do not force extra pages.
        key = tuple(pages)
        if key in seen:
            continue
        src = str(sources[idx]) if idx < len(sources) else "semantic"
        cand = _make_candidate(
            len(out),
            "page_aligned_semantic",
            pages,
            prompt_len,
            page_tokens,
            semantic_span_tokens=e - s,
            semantic_source=src,
        )
        if cand:
            out.append(cand)
            seen.add(key)
    return out, fallback_used, fallback_reason


def _build_evidence_centered(
    tokens: List[int],
    tokenizer: Any,
    query_text: str,
    num_pages: int,
    prompt_len: int,
    page_tokens: int,
    half_window_tokens: int,
    top_ratio: float,
) -> Tuple[List[RescueCandidate], bool, str]:
    out: List[RescueCandidate] = []
    seen: set = set()
    for page in range(num_pages):
        cand = _make_candidate(len(out), "fixed", [page], prompt_len, page_tokens)
        if cand:
            out.append(cand)
            seen.add(tuple(cand.covered_fixed_pages))
    page_scores = _page_query_scores(tokens, tokenizer, query_text, num_pages, page_tokens)
    if not any(s > 0 for s in page_scores):
        return out, True, "no_query_evidence"
    top_n = max(1, int(math.ceil(num_pages * max(0.05, float(top_ratio)))))
    centers = sorted(range(num_pages), key=lambda p: page_scores[p], reverse=True)[:top_n]
    half_pages = max(0, int(round(half_window_tokens / page_tokens)))
    for c in centers:
        if page_scores[c] <= 0:
            continue
        grp = [p for p in range(c - half_pages, c + half_pages + 1) if 0 <= p < num_pages]
        key = tuple(grp)
        if key in seen:
            continue
        cand = _make_candidate(
            len(out),
            "evidence_centered",
            grp,
            prompt_len,
            page_tokens,
            query_score=page_scores[c],
            neighbor_source_page=c,
        )
        if cand:
            out.append(cand)
            seen.add(key)
    # attach per-page query score onto fixed candidates too (useful for fusion)
    for cand in out:
        if cand.strategy_source == "fixed" and len(cand.covered_fixed_pages) == 1:
            p = cand.covered_fixed_pages[0]
            if 0 <= p < num_pages:
                cand.query_score = float(page_scores[p])
    return out, False, ""


def _dedup_merge(candidates: List[RescueCandidate]) -> List[RescueCandidate]:
    """Merge by covered-page key, keeping the richest (max query score, prefer
    non-fixed source) representative; re-id sequentially."""
    best: Dict[Tuple[int, ...], RescueCandidate] = {}
    for cand in candidates:
        key = tuple(cand.covered_fixed_pages)
        cur = best.get(key)
        if cur is None:
            best[key] = cand
            continue
        # prefer a candidate carrying query signal or a non-fixed semantic source
        score_new = (cand.query_score, 0 if cand.strategy_source == "fixed" else 1)
        score_cur = (cur.query_score, 0 if cur.strategy_source == "fixed" else 1)
        if score_new > score_cur:
            best[key] = cand
    out = sorted(best.values(), key=lambda c: (c.token_span[0], c.token_span[1]))
    for i, cand in enumerate(out):
        cand.candidate_id = i
    return out


def _ordered_query_vocab(query_text: str) -> List[str]:
    seen: Dict[str, int] = {}
    out: List[str] = []
    for w in _TOKEN_RE.findall(query_text or ""):
        wl = w.lower()
        if len(wl) <= 1 or wl in _STOPWORDS:
            continue
        if wl not in seen:
            seen[wl] = len(out)
            out.append(wl)
    return out


def _attach_query_answer_features(
    candidates: List[RescueCandidate],
    tokens: List[int],
    tokenizer: Any,
    query_text: str,
    answer_text: str,
    page_tokens: int,
) -> int:
    """Universal per-candidate features for the structural selectors:
    - query_score: lexical relevance to query (overlap of content words)
    - query_covered_idx: which query-vocab indices the candidate covers (for coverage gain)
    - answer_score: lexical overlap with gold answer (oracle-diagnostic only)
    Returns the query vocab size."""
    vocab = _ordered_query_vocab(query_text)
    vocab_idx = {w: i for i, w in enumerate(vocab)}
    ans_words = {w.lower() for w in _TOKEN_RE.findall(answer_text or "") if len(w) > 1 and w.lower() not in _STOPWORDS}
    # decode page texts once
    num_pages = max(1, (len(tokens) + page_tokens - 1) // page_tokens)
    page_words: List[set] = []
    for page in range(num_pages):
        seg = tokens[page * page_tokens:(page + 1) * page_tokens]
        words = {w.lower() for w in _TOKEN_RE.findall(_decode(tokenizer, seg))}
        page_words.append(words)
    for cand in candidates:
        words: set = set()
        for p in cand.covered_fixed_pages:
            if 0 <= p < num_pages:
                words |= page_words[p]
        covered = sorted({vocab_idx[w] for w in words if w in vocab_idx})
        cand.query_covered_idx = covered
        cand.query_score = float(len(covered)) / float(max(1, len(vocab))) if vocab else 0.0
        if ans_words:
            cand.answer_score = float(len(words & ans_words)) / float(max(1, len(ans_words)))
        else:
            cand.answer_score = 0.0
    return len(vocab)


def _build_unified(
    input_ids: Any,
    tokenizer: Any,
    sample: Optional[Dict[str, Any]],
    task_name: str,
    prompt_text: Optional[str],
    query_text: str,
    tokens: List[int],
    num_pages: int,
    prompt_len: int,
    page_tokens: int,
    config: Dict[str, Any],
) -> Tuple[List[RescueCandidate], bool, str]:
    """Union of every A-F candidate source (Selector-Isolation pool)."""
    merged: List[RescueCandidate] = []
    fb_used = False
    fb_reason = ""
    merged.extend(_build_fixed(num_pages, prompt_len, page_tokens))
    merged.extend(_build_overlap(
        num_pages, prompt_len, page_tokens,
        int(config.get("overlap_window_tokens", 64) or 64),
        int(config.get("overlap_stride_tokens", 32) or 32)))
    merged.extend(_build_neighbor(num_pages, prompt_len, page_tokens))
    merged.extend(_build_hierarchy(
        num_pages, prompt_len, page_tokens,
        int(config.get("coarse_block_tokens", 256) or 256),
        int(config.get("fine_block_tokens", 64) or 64),
        int(config.get("medium_block_tokens", 128) or 128)))
    sem_cands, sem_fb, sem_reason = _build_page_aligned_semantic(
        input_ids, tokenizer, sample, task_name, prompt_text,
        query_text, num_pages, prompt_len, page_tokens, config)
    merged.extend(sem_cands)
    ev_cands, ev_fb, ev_reason = _build_evidence_centered(
        tokens, tokenizer, query_text, num_pages, prompt_len, page_tokens,
        int(config.get("evidence_half_window_tokens", 64) or 64),
        float(config.get("evidence_top_ratio", 0.2) or 0.2))
    merged.extend(ev_cands)
    if sem_fb:
        fb_used = True
        fb_reason = (fb_reason + ";" + sem_reason).strip(";")
    if ev_fb:
        fb_used = True
        fb_reason = (fb_reason + ";" + ev_reason).strip(";")
    return _dedup_merge(merged), fb_used, fb_reason


def build_rescue_candidates(
    input_ids: Any,
    tokenizer: Any,
    sample: Optional[Dict[str, Any]],
    task_name: str,
    fixed_active_pages: Optional[Sequence[int]] = None,
    block_layout_fixed: Optional[Dict[str, Any]] = None,
    strategy: str = "fixed",
    model_states: Optional[Dict[str, Any]] = None,
    query_text: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> RescueCandidateSet:
    """Build a strategy-specific rescue candidate pool.

    ``fixed_active_pages`` is accepted for interface completeness but is NOT used
    to mutate the active set; candidates may overlap the active set and the
    downstream selector removes fully-covered duplicates.
    """
    config = dict(config or {})
    strategy = str(strategy or "fixed")
    page_tokens = int(config.get("page_tokens", _PAGE_TOKENS) or _PAGE_TOKENS)
    tokens = _as_token_list(input_ids)
    prompt_len = len(tokens)
    if prompt_len <= 0:
        return RescueCandidateSet(strategy=strategy, candidates=[], page_tokens=page_tokens,
                                  prompt_len=0, fallback_used=True, fallback_reason="empty_prompt")
    num_pages = max(1, (prompt_len + page_tokens - 1) // page_tokens)
    q_text, _, q_fallback_reason = extract_query_text(sample, config.get("prompt_text"), query_text)

    fallback_used = False
    fallback_reason = ""
    candidates: List[RescueCandidate] = []

    if strategy == "fixed":
        candidates = _build_fixed(num_pages, prompt_len, page_tokens)
    elif strategy == "overlap":
        candidates = _build_overlap(
            num_pages, prompt_len, page_tokens,
            int(config.get("overlap_window_tokens", 64) or 64),
            int(config.get("overlap_stride_tokens", 32) or 32),
        )
    elif strategy == "neighbor":
        candidates = _build_neighbor(num_pages, prompt_len, page_tokens)
    elif strategy == "hierarchy":
        candidates = _build_hierarchy(
            num_pages, prompt_len, page_tokens,
            int(config.get("coarse_block_tokens", 256) or 256),
            int(config.get("fine_block_tokens", 64) or 64),
            int(config.get("medium_block_tokens", 128) or 128),
        )
    elif strategy == "page_aligned_semantic":
        candidates, fallback_used, fallback_reason = _build_page_aligned_semantic(
            input_ids, tokenizer, sample, task_name, config.get("prompt_text"),
            q_text, num_pages, prompt_len, page_tokens, config,
        )
    elif strategy == "evidence_centered":
        candidates, fallback_used, fallback_reason = _build_evidence_centered(
            tokens, tokenizer, q_text, num_pages, prompt_len, page_tokens,
            int(config.get("evidence_half_window_tokens", 64) or 64),
            float(config.get("evidence_top_ratio", 0.2) or 0.2),
        )
    elif strategy in {"fusion_fine", "fusion_fine_no_query"}:
        use_query = strategy == "fusion_fine"
        merged: List[RescueCandidate] = []
        merged.extend(_build_overlap(
            num_pages, prompt_len, page_tokens,
            int(config.get("overlap_window_tokens", 64) or 64),
            int(config.get("overlap_stride_tokens", 32) or 32),
        ))
        sem_cands, sem_fb, sem_reason = _build_page_aligned_semantic(
            input_ids, tokenizer, sample, task_name, config.get("prompt_text"),
            q_text, num_pages, prompt_len, page_tokens, config,
        )
        merged.extend(sem_cands)
        if use_query:
            ev_cands, ev_fb, ev_reason = _build_evidence_centered(
                tokens, tokenizer, q_text, num_pages, prompt_len, page_tokens,
                int(config.get("evidence_half_window_tokens", 64) or 64),
                float(config.get("evidence_top_ratio", 0.2) or 0.2),
            )
            merged.extend(ev_cands)
            if ev_fb:
                fallback_used = True
                fallback_reason = (fallback_reason + ";" + ev_reason).strip(";")
        if sem_fb:
            fallback_used = True
            fallback_reason = (fallback_reason + ";" + sem_reason).strip(";")
        candidates = _dedup_merge(merged)
    elif strategy == "unified":
        candidates, fallback_used, fallback_reason = _build_unified(
            input_ids, tokenizer, sample, task_name, config.get("prompt_text"),
            q_text, tokens, num_pages, prompt_len, page_tokens, config)
    else:
        # unknown strategy -> safe fixed fallback
        candidates = _build_fixed(num_pages, prompt_len, page_tokens)
        fallback_used = True
        fallback_reason = f"unknown_strategy:{strategy}"

    if not candidates:
        candidates = _build_fixed(num_pages, prompt_len, page_tokens)
        fallback_used = True
        fallback_reason = fallback_reason or "empty_candidate_pool"

    # universal features needed by the structural selectors (query / coverage / oracle)
    vocab_size = _attach_query_answer_features(
        candidates, tokens, tokenizer, q_text,
        str(config.get("answer_text", "") or ""), page_tokens)

    counts: Dict[str, int] = {}
    for cand in candidates:
        counts[cand.strategy_source] = counts.get(cand.strategy_source, 0) + 1

    return RescueCandidateSet(
        strategy=strategy,
        candidates=candidates,
        page_tokens=page_tokens,
        prompt_len=prompt_len,
        fallback_used=fallback_used,
        fallback_reason=fallback_reason,
        fallback_reason_counts={},
        metadata={
            "num_candidates": len(candidates),
            "num_pages": int(num_pages),
            "source_counts": counts,
            "query_fallback_reason": q_fallback_reason,
            "query_available": bool(_query_word_set(q_text)),
            "query_vocab_size": int(vocab_size),
        },
    )
