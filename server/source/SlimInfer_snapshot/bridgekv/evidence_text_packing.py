"""CPU-only evidence-aware text packing utilities.

This module deliberately does not implement KV pruning. It builds compact text
contexts from long-context samples using lexical evidence signals only, so it can
run offline without loading model weights or tokenizers.
"""
from __future__ import annotations

import math
import hashlib
import random
import re
from collections import Counter, defaultdict

from bridgekv.sensitivity_params import SP as _SP
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple


TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)?|[^\w\s]", re.UNICODE)
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)?", re.UNICODE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "he", "her", "his", "in", "is", "it", "its", "of", "on", "or", "she",
    "that", "the", "their", "them", "there", "they", "this", "to", "was",
    "were", "what", "when", "where", "which", "who", "whom", "whose", "why",
    "with", "how", "did", "does", "do", "into", "than", "then", "also",
    "about", "after", "before", "between", "during", "over", "under",
}


@dataclass(frozen=True)
class CandidateSpan:
    sample_id: str
    task: str
    span_id: str
    page_id: int
    block_id: int
    start_token: int
    end_token: int
    text: str
    source_order_index: int
    base_score: float
    query_score: float
    bridge_score: float
    density_score: float
    final_score: float


@dataclass(frozen=True)
class PackedEvidence:
    sample_id: str
    task: str
    variant: str
    budget: int
    packed_text: str
    selected_span_ids: List[str]
    selected_page_ids: List[int]
    packed_token_count: int
    source_token_count: int
    compression_ratio: float
    ordering: str
    neighbor_size: int
    used_bridge_path: bool
    used_tags: bool
    truncated: bool = False


@dataclass(frozen=True)
class EvidencePath:
    span_i: CandidateSpan
    span_j: CandidateSpan
    query_link: float
    bridge_link: float
    path_score: float


def tokenize_light(text: str) -> List[str]:
    """Tokenize with a tiny CPU-only regex tokenizer."""
    if not text:
        return []
    return TOKEN_RE.findall(text)


def estimate_token_count(text: str) -> int:
    return len(tokenize_light(text))


def _word_tokens(text: str) -> List[str]:
    return [tok.lower() for tok in WORD_RE.findall(text or "")]


def _content_tokens(text: str) -> List[str]:
    return [tok for tok in _word_tokens(text) if tok not in STOPWORDS and len(tok) > 1]


def _detokenize(tokens: Sequence[str]) -> str:
    text = " ".join(tokens)
    text = re.sub(r"\s+([,.;:!?%)\]\}])", r"\1", text)
    text = re.sub(r"([\[(\{])\s+", r"\1", text)
    text = re.sub(r"\s+'", "'", text)
    return text.strip()


def split_into_pages(text: str, page_size_tokens: int = 256) -> List[Dict[str, Any]]:
    tokens = tokenize_light(text)
    pages: List[Dict[str, Any]] = []
    for page_id, start in enumerate(range(0, len(tokens), page_size_tokens)):
        end = min(len(tokens), start + page_size_tokens)
        page_tokens = tokens[start:end]
        pages.append({
            "page_id": page_id,
            "block_id": page_id,
            "start_token": start,
            "end_token": end,
            "tokens": page_tokens,
            "text": _detokenize(page_tokens),
        })
    return pages


def split_page_into_spans(
    page: Mapping[str, Any],
    span_size_tokens: int = 64,
    stride_tokens: int = 32,
) -> List[Dict[str, Any]]:
    tokens = list(page.get("tokens", []))
    if not tokens:
        return []
    spans: List[Dict[str, Any]] = []
    page_start = int(page.get("start_token", 0))
    page_id = int(page.get("page_id", 0))
    block_id = int(page.get("block_id", page_id))
    local_starts = list(range(0, max(1, len(tokens)), stride_tokens))
    for local_start in local_starts:
        local_end = min(len(tokens), local_start + span_size_tokens)
        if local_start >= local_end:
            continue
        absolute_start = page_start + local_start
        absolute_end = page_start + local_end
        spans.append({
            "page_id": page_id,
            "block_id": block_id,
            "start_token": absolute_start,
            "end_token": absolute_end,
            "local_start": local_start,
            "local_end": local_end,
            "tokens": tokens[local_start:local_end],
            "text": _detokenize(tokens[local_start:local_end]),
        })
        if local_end == len(tokens):
            break
    return spans


def expand_span_with_neighbors(
    span: Mapping[str, Any] | CandidateSpan,
    neighbor_tokens: int,
    page_tokens: Optional[Sequence[str]] = None,
    page_start_token: int = 0,
) -> Dict[str, Any]:
    if isinstance(span, CandidateSpan):
        start = span.start_token
        end = span.end_token
        text = span.text
        page_id = span.page_id
        span_id = span.span_id
    else:
        start = int(span.get("start_token", 0))
        end = int(span.get("end_token", start))
        text = str(span.get("text", ""))
        page_id = int(span.get("page_id", 0))
        span_id = str(span.get("span_id", ""))
    if page_tokens is None:
        return {
            "span_id": span_id,
            "page_id": page_id,
            "start_token": start,
            "end_token": end,
            "text": text,
        }
    local_start = max(0, start - page_start_token - neighbor_tokens)
    local_end = min(len(page_tokens), end - page_start_token + neighbor_tokens)
    return {
        "span_id": span_id,
        "page_id": page_id,
        "start_token": page_start_token + local_start,
        "end_token": page_start_token + local_end,
        "text": _detokenize(page_tokens[local_start:local_end]),
    }


def normalize_scores(values: Sequence[float]) -> List[float]:
    vals = [float(v) for v in values]
    if not vals:
        return []
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    std = math.sqrt(var)
    if std <= 1e-12:
        return [0.0 for _ in vals]
    return [(v - mean) / std for v in vals]


def jaccard(a: Iterable[Any], b: Iterable[Any]) -> float:
    left, right = set(a), set(b)
    if not left and not right:
        return 1.0
    return len(left & right) / max(1, len(left | right))


def query_link_score(query: str, span_text: str) -> float:
    q_tokens = _content_tokens(query)
    s_tokens = _content_tokens(span_text)
    if not q_tokens or not s_tokens:
        return 0.0
    q_counts = Counter(q_tokens)
    s_counts = Counter(s_tokens)
    q_set = set(q_tokens)
    length_norm = 1.0 + math.log1p(len(s_tokens))
    score = 0.0
    for tok, q_tf in q_counts.items():
        tf = s_counts.get(tok, 0)
        if tf:
            score += (1.0 + math.log(tf)) * (1.0 + math.log(q_tf))
    coverage = len(q_set & set(s_tokens)) / max(1, len(q_set))
    return score / length_norm + 1.5 * coverage


def bridge_link_score(span_i: CandidateSpan | str, span_j: CandidateSpan | str) -> float:
    text_i = span_i.text if isinstance(span_i, CandidateSpan) else str(span_i)
    text_j = span_j.text if isinstance(span_j, CandidateSpan) else str(span_j)
    toks_i = set(_content_tokens(text_i))
    toks_j = set(_content_tokens(text_j))
    if not toks_i or not toks_j:
        return 0.0
    overlap = toks_i & toks_j
    shared = len(overlap) / math.sqrt(max(1, len(toks_i) * len(toks_j)))
    if _SP.bridge_zero:
        return 0.0
    too_similar_penalty = _SP.c_jac * jaccard(toks_i, toks_j)
    novelty_bonus = _SP.c_div * (1.0 - jaccard(toks_i, toks_j))
    return shared + novelty_bonus - too_similar_penalty


def _entity_like_tokens(text: str) -> Set[str]:
    out: Set[str] = set()
    for token in WORD_RE.findall(text or ""):
        if len(token) > 2 and (token[:1].isupper() or any(ch.isdigit() for ch in token)):
            low = token.lower()
            if low not in STOPWORDS:
                out.add(low)
    return out


def density_score(
    span: CandidateSpan | Mapping[str, Any],
    page_spans: Sequence[CandidateSpan | Mapping[str, Any]],
    query: str = "",
) -> float:
    text = span.text if isinstance(span, CandidateSpan) else str(span.get("text", ""))
    span_tokens = set(_content_tokens(text))
    if not span_tokens:
        return 0.0
    page_texts = [
        s.text if isinstance(s, CandidateSpan) else str(s.get("text", ""))
        for s in page_spans
    ]
    page_counts = Counter(tok for txt in page_texts for tok in set(_content_tokens(txt)))
    repeated = sum(1 for tok in span_tokens if page_counts[tok] >= 2) / max(1, len(span_tokens))
    entities = _entity_like_tokens(text)
    entity_repeats = sum(1 for tok in entities if page_counts[tok] >= 2) / max(1, len(entities) or 1)
    query_overlap = jaccard(span_tokens, set(_content_tokens(query))) if query else 0.0
    return 0.55 * repeated + 0.30 * entity_repeats + 0.15 * query_overlap


def final_span_score(
    base_score: float,
    query_score: float,
    density: float,
    base_weight: float = 0.30,
    query_weight: float = 1.00,
    density_weight: float = 0.75,
) -> float:
    return base_weight * base_score + query_weight * query_score + density_weight * density


def build_candidate_spans(
    sample_id: str,
    task: str,
    query: str,
    context: str,
    page_size_tokens: int = 256,
    span_size_tokens: int = 64,
    stride_tokens: int = 32,
) -> Tuple[List[CandidateSpan], Dict[int, Dict[str, Any]]]:
    pages = split_into_pages(context, page_size_tokens=page_size_tokens)
    span_dicts: List[Dict[str, Any]] = []
    page_to_span_dicts: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for page in pages:
        for span in split_page_into_spans(
            page,
            span_size_tokens=span_size_tokens,
            stride_tokens=stride_tokens,
        ):
            span_dicts.append(span)
            page_to_span_dicts[int(page["page_id"])].append(span)

    query_scores = [query_link_score(query, span["text"]) for span in span_dicts]
    density_scores = [
        density_score(span, page_to_span_dicts[int(span["page_id"])], query=query)
        for span in span_dicts
    ]
    base_scores = [
        0.5 * query_scores[idx] + 0.5 * density_scores[idx]
        for idx in range(len(span_dicts))
    ]
    q_norm = normalize_scores(query_scores)
    d_norm = normalize_scores(density_scores)
    b_norm = normalize_scores(base_scores)

    candidates: List[CandidateSpan] = []
    for idx, span in enumerate(span_dicts):
        score = final_span_score(b_norm[idx], q_norm[idx], d_norm[idx])
        span_id = f"p{span['page_id']}_s{idx}_t{span['start_token']}"
        candidates.append(CandidateSpan(
            sample_id=sample_id,
            task=task,
            span_id=span_id,
            page_id=int(span["page_id"]),
            block_id=int(span["block_id"]),
            start_token=int(span["start_token"]),
            end_token=int(span["end_token"]),
            text=str(span["text"]),
            source_order_index=idx,
            base_score=b_norm[idx],
            query_score=q_norm[idx],
            bridge_score=0.0,
            density_score=d_norm[idx],
            final_score=score,
        ))
    page_map = {int(page["page_id"]): page for page in pages}
    return candidates, page_map


def _overlaps(left: CandidateSpan, right: CandidateSpan, max_fraction: float = 0.55) -> bool:
    if left.page_id != right.page_id:
        return False
    overlap = max(0, min(left.end_token, right.end_token) - max(left.start_token, right.start_token))
    denom = max(1, min(left.end_token - left.start_token, right.end_token - right.start_token))
    return overlap / denom > max_fraction


def _dedupe_candidates(candidates: Sequence[CandidateSpan]) -> List[CandidateSpan]:
    selected: List[CandidateSpan] = []
    for cand in candidates:
        if any(_overlaps(cand, prev) for prev in selected):
            continue
        selected.append(cand)
    return selected


def _replace_scores(candidates: Sequence[CandidateSpan], scores: Mapping[str, float]) -> List[CandidateSpan]:
    out: List[CandidateSpan] = []
    for span in candidates:
        score = float(scores.get(span.span_id, 0.0))
        out.append(CandidateSpan(
            sample_id=span.sample_id,
            task=span.task,
            span_id=span.span_id,
            page_id=span.page_id,
            block_id=span.block_id,
            start_token=span.start_token,
            end_token=span.end_token,
            text=span.text,
            source_order_index=span.source_order_index,
            base_score=span.base_score,
            query_score=span.query_score,
            bridge_score=span.bridge_score,
            density_score=span.density_score,
            final_score=score,
        ))
    return out


def bm25_scores(query: str, candidates: Sequence[CandidateSpan]) -> Dict[str, float]:
    """BM25-like lexical relevance between the question and each candidate span."""
    q_terms = _content_tokens(query)
    if not q_terms or not candidates:
        return {span.span_id: 0.0 for span in candidates}
    docs = [_content_tokens(span.text) for span in candidates]
    doc_count = len(docs)
    df = Counter()
    for terms in docs:
        df.update(set(terms))
    avgdl = sum(len(terms) for terms in docs) / max(1, doc_count)
    q_counts = Counter(q_terms)
    k1 = 1.2
    b = 0.75
    scores: Dict[str, float] = {}
    for span, terms in zip(candidates, docs):
        tf = Counter(terms)
        dl = len(terms)
        score = 0.0
        for term, q_tf in q_counts.items():
            if not tf.get(term):
                continue
            idf = math.log(1.0 + (doc_count - df[term] + 0.5) / (df[term] + 0.5))
            denom = tf[term] + k1 * (1.0 - b + b * dl / max(1e-9, avgdl))
            score += idf * (tf[term] * (k1 + 1.0) / max(1e-9, denom)) * (1.0 + math.log(q_tf))
        scores[span.span_id] = score
    return scores


def tfidf_scores(query: str, candidates: Sequence[CandidateSpan]) -> Dict[str, float]:
    """Sklearn-free TF-IDF cosine between the question and each candidate span."""
    q_terms = _content_tokens(query)
    if not q_terms or not candidates:
        return {span.span_id: 0.0 for span in candidates}
    docs = [_content_tokens(span.text) for span in candidates]
    doc_count = len(docs)
    df = Counter()
    for terms in docs:
        df.update(set(terms))
    vocab = set(q_terms)
    for terms in docs:
        vocab.update(terms)
    idf = {
        term: math.log((1.0 + doc_count) / (1.0 + df.get(term, 0))) + 1.0
        for term in vocab
    }

    def vector(terms: Sequence[str]) -> Dict[str, float]:
        counts = Counter(terms)
        return {
            term: (1.0 + math.log(tf)) * idf.get(term, 1.0)
            for term, tf in counts.items()
            if tf > 0
        }

    q_vec = vector(q_terms)
    q_norm = math.sqrt(sum(value * value for value in q_vec.values()))
    scores: Dict[str, float] = {}
    for span, terms in zip(candidates, docs):
        d_vec = vector(terms)
        d_norm = math.sqrt(sum(value * value for value in d_vec.values()))
        if q_norm <= 1e-12 or d_norm <= 1e-12:
            scores[span.span_id] = 0.0
            continue
        dot = sum(q_vec.get(term, 0.0) * d_vec.get(term, 0.0) for term in q_vec)
        scores[span.span_id] = dot / (q_norm * d_norm)
    return scores


def lexical_scores(query: str, candidates: Sequence[CandidateSpan]) -> Dict[str, float]:
    """Z-normalized blend of BM25-like and TF-IDF lexical scores."""
    bm25 = bm25_scores(query, candidates)
    tfidf = tfidf_scores(query, candidates)
    bm25_norm = normalize_scores([bm25.get(span.span_id, 0.0) for span in candidates])
    tfidf_norm = normalize_scores([tfidf.get(span.span_id, 0.0) for span in candidates])
    scores: Dict[str, float] = {}
    for idx, span in enumerate(candidates):
        scores[span.span_id] = _SP.alpha * bm25_norm[idx] + (1.0 - _SP.alpha) * tfidf_norm[idx]
    return scores


def lexical_scored_candidates(query: str, candidates: Sequence[CandidateSpan]) -> List[CandidateSpan]:
    return _replace_scores(candidates, lexical_scores(query, candidates))


def lexical_hybrid_candidates(query: str, candidates: Sequence[CandidateSpan]) -> List[CandidateSpan]:
    """Fixed-weight lexical/query/bridge hybrid scores.

    Weights are intentionally fixed:
    lexical=1.0, query=0.8, bridge=0.4. Redundancy is applied during selection.
    """
    if not candidates:
        return []
    lexical = lexical_scores(query, candidates)
    anchor = max(candidates, key=lambda s: (s.query_score, s.final_score))
    bridge_values = [bridge_link_score(anchor, span) for span in candidates]
    bridge_norm = normalize_scores(bridge_values)
    scores: Dict[str, float] = {}
    for idx, span in enumerate(candidates):
        scores[span.span_id] = (
            1.0 * lexical.get(span.span_id, 0.0)
            + 0.8 * span.query_score
            + 0.4 * bridge_norm[idx]
        )
    return _replace_scores(candidates, scores)


def _expanded_piece(
    span: CandidateSpan,
    page_map: Mapping[int, Mapping[str, Any]],
    neighbor_size: int,
    role: str = "support",
    use_tags: bool = False,
) -> Tuple[str, Dict[str, Any]]:
    page = page_map.get(span.page_id, {})
    expanded = expand_span_with_neighbors(
        span,
        neighbor_size,
        page_tokens=page.get("tokens", None),
        page_start_token=int(page.get("start_token", 0)),
    )
    text = str(expanded["text"])
    if use_tags:
        text = (
            f"[Evidence id={span.span_id} page={span.page_id} span={span.span_id} "
            f"position={expanded['start_token']}-{expanded['end_token']} role={role}]\n"
            f"{text}\n[/Evidence]"
        )
    return text, expanded


def _can_add(prefix: str, pieces: Sequence[str], candidate: str, budget: int) -> bool:
    text = "\n\n".join([prefix, *pieces, candidate]).strip()
    return estimate_token_count(text) <= budget


def _truncate_to_budget(text: str, budget: int) -> str:
    tokens = tokenize_light(text)
    if len(tokens) <= budget:
        return text
    return _detokenize(tokens[:budget])


def _prefix(query: str, use_tags: bool) -> str:
    if use_tags:
        return (
            "The following excerpts are selected from the original long context. "
            "They may come from different positions. Use them together to answer "
            "the question.\n\n"
            f"Question: {query}"
        )
    return f"Question: {query}\n\nSelected evidence:"


def _pack_from_ordered_spans(
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
    used_tags: bool,
) -> Tuple[PackedEvidence, Dict[str, Any]]:
    prefix = _prefix(query, used_tags)
    pieces: List[str] = []
    selected: List[CandidateSpan] = []
    expanded_records: List[Dict[str, Any]] = []
    truncated = False
    for span, role in ordered:
        piece, expanded = _expanded_piece(
            span,
            page_map,
            neighbor_size,
            role=role,
            use_tags=used_tags,
        )
        if _can_add(prefix, pieces, piece, budget):
            pieces.append(piece)
            selected.append(span)
            expanded_records.append(expanded)
            continue
        if not pieces:
            truncated = _truncate_to_budget("\n\n".join([prefix, piece]), budget)
            pieces = [truncated.replace(prefix, "", 1).strip()]
            selected = [span]
            expanded_records = [expanded]
            truncated = True
        break
    packed_text = "\n\n".join([prefix, *pieces]).strip()
    packed_tokens = estimate_token_count(packed_text)
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
        used_tags=used_tags,
        truncated=truncated,
    )
    audit = {
        "selected_spans": selected,
        "expanded_records": expanded_records,
        "path_count": 0,
        "query_link_values": [span.query_score for span in selected],
        "bridge_link_values": [],
        "cross_page_path_count": 0,
        "same_page_path_count": 0,
    }
    return packed, audit


def _select_ranked_for_budget(
    query: str,
    ranked: Sequence[CandidateSpan],
    page_map: Mapping[int, Mapping[str, Any]],
    budget: int,
    neighbor_size: int,
    use_tags: bool,
    max_fraction: float = 0.55,
) -> List[CandidateSpan]:
    prefix = _prefix(query, use_tags)
    pieces: List[str] = []
    selected: List[CandidateSpan] = []
    for span in ranked:
        if any(_overlaps(span, prior, max_fraction=max_fraction) for prior in selected):
            continue
        piece, _ = _expanded_piece(span, page_map, neighbor_size, role="support", use_tags=use_tags)
        if not _can_add(prefix, pieces, piece, budget):
            continue
        selected.append(span)
        pieces.append(piece)
        if estimate_token_count("\n\n".join([prefix, *pieces])) >= budget * 0.95:
            break
    if not selected and ranked:
        selected = [ranked[0]]
    return selected


def _pack_retrieval_style(
    sample_id: str,
    task: str,
    query: str,
    variant: str,
    budget: int,
    candidates: Sequence[CandidateSpan],
    page_map: Mapping[int, Mapping[str, Any]],
    scores: Mapping[str, float],
) -> Tuple[PackedEvidence, Dict[str, Any]]:
    scored_candidates = _replace_scores(candidates, scores)
    ranked = sorted(scored_candidates, key=lambda s: (-s.final_score, s.source_order_index))
    selected = _select_ranked_for_budget(
        query,
        ranked,
        page_map,
        budget,
        neighbor_size=0,
        use_tags=False,
    )
    ordered = [(span, "support") for span in sorted(selected, key=lambda s: s.source_order_index)]
    packed, audit = _pack_from_ordered_spans(
        sample_id,
        task,
        query,
        variant,
        budget,
        ordered,
        page_map,
        neighbor_size=0,
        ordering="source_order",
        used_bridge_path=False,
        used_tags=False,
    )
    audit.update({
        "retrieval_score_values": [scores.get(span.span_id, 0.0) for span in selected],
        "retrieval_baseline": variant,
    })
    return packed, audit


def _select_hybrid_for_budget(
    query: str,
    ranked: Sequence[CandidateSpan],
    page_map: Mapping[int, Mapping[str, Any]],
    budget: int,
    neighbor_size: int,
    redundancy_penalty: float = 0.3,
) -> List[CandidateSpan]:
    prefix = _prefix(query, False)
    pieces: List[str] = []
    selected: List[CandidateSpan] = []
    selected_terms: Set[str] = set()
    remaining = list(ranked)
    while remaining:
        best: Optional[Tuple[float, CandidateSpan]] = None
        for span in remaining:
            terms = set(_content_tokens(span.text))
            redundancy = jaccard(terms, selected_terms) if selected_terms else 0.0
            adjusted = span.final_score - redundancy_penalty * redundancy
            if best is None or adjusted > best[0]:
                best = (adjusted, span)
        if best is None:
            break
        span = best[1]
        remaining = [item for item in remaining if item.span_id != span.span_id]
        if any(_overlaps(span, prior, max_fraction=0.30) for prior in selected):
            continue
        piece, _ = _expanded_piece(span, page_map, neighbor_size, role="lexical_hybrid", use_tags=False)
        if not _can_add(prefix, pieces, piece, budget):
            continue
        selected.append(span)
        pieces.append(piece)
        selected_terms.update(_content_tokens(span.text))
        if estimate_token_count("\n\n".join([prefix, *pieces])) >= budget * 0.95:
            break
    if not selected and ranked:
        selected = [ranked[0]]
    return selected


def _independent_order(
    candidates: Sequence[CandidateSpan],
    variant: str,
) -> Tuple[List[CandidateSpan], str]:
    ranked = _dedupe_candidates(sorted(candidates, key=lambda s: (-s.final_score, s.source_order_index)))
    if variant.endswith("score_order"):
        return ranked, "score_order"
    if variant.endswith("path_order"):
        query_anchor = max(candidates, key=lambda s: (s.query_score, s.final_score))
        ordered = sorted(
            ranked,
            key=lambda s: (
                -s.query_score,
                -bridge_link_score(query_anchor, s),
                s.source_order_index,
            ),
        )
        return ordered, "path_order"
    return sorted(ranked, key=lambda s: s.source_order_index), "source_order"


def _query_only_path_order(
    candidates: Sequence[CandidateSpan],
) -> List[CandidateSpan]:
    ranked = _dedupe_candidates(sorted(
        candidates,
        key=lambda s: (-s.query_score, -s.density_score, s.source_order_index),
    ))
    return ranked


def build_candidate_paths(
    query: str,
    candidates: Sequence[CandidateSpan],
    max_query_spans: int = 32,
    max_target_spans: int = 48,
) -> List[EvidencePath]:
    query_spans = sorted(candidates, key=lambda s: (-s.query_score, -s.final_score))[:max_query_spans]
    target_spans = sorted(candidates, key=lambda s: (-s.final_score, -s.query_score))[:max_target_spans]
    paths: List[EvidencePath] = []
    for left in query_spans:
        if left.query_score <= 0.0:
            continue
        for right in target_spans:
            if left.span_id == right.span_id or _overlaps(left, right, max_fraction=0.10):
                continue
            if right.final_score <= 0.0:
                continue
            bridge = bridge_link_score(left, right)
            query_link = query_link_score(query, left.text)
            score = query_link + bridge + _SP.c_sup * right.final_score
            paths.append(EvidencePath(left, right, query_link, bridge, score))
    return sorted(paths, key=lambda p: (-p.path_score, p.span_i.source_order_index, p.span_j.source_order_index))


def _select_bridge_paths(
    query: str,
    candidates: Sequence[CandidateSpan],
    page_map: Mapping[int, Mapping[str, Any]],
    budget: int,
    use_tags: bool,
    neighbor_size: int = 32,
) -> Tuple[List[Tuple[CandidateSpan, str]], Dict[str, Any]]:
    prefix = _prefix(query, use_tags)
    selected_paths: List[EvidencePath] = []
    selected_spans: List[CandidateSpan] = []
    selected_roles: Dict[str, str] = {}
    selected_terms: Set[str] = set()
    pieces: List[str] = []
    remaining = build_candidate_paths(query, candidates)
    path_terms = {
        id(path): set(_content_tokens(path.span_i.text)) | set(_content_tokens(path.span_j.text))
        for path in remaining
    }
    for _ in range(24):
        best: Optional[Tuple[float, EvidencePath]] = None
        for path in remaining:
            pair_terms = path_terms.get(id(path), set())
            redundancy = jaccard(pair_terms, selected_terms) if selected_terms else 0.0
            adjusted = path.path_score - _SP.lam * redundancy
            if best is None or adjusted > best[0]:
                best = (adjusted, path)
        if best is None:
            break
        if best[0] <= 0.0:
            break
        path = best[1]
        trial: List[Tuple[CandidateSpan, str]] = [(path.span_i, "query"), (path.span_j, "bridge")]
        trial_pieces = list(pieces)
        can_take = False
        for span, role in trial:
            if any(_overlaps(span, prior, max_fraction=0.30) for prior in selected_spans):
                continue
            piece, _ = _expanded_piece(span, page_map, neighbor_size, role=role, use_tags=use_tags)
            if not _can_add(prefix, trial_pieces, piece, budget):
                continue
            trial_pieces.append(piece)
            can_take = True
        if not can_take:
            remaining = [p for p in remaining if p is not path]
            if not remaining:
                break
            continue
        selected_paths.append(path)
        pieces = trial_pieces
        for span, _role in trial:
            if not any(_overlaps(span, prior, max_fraction=0.30) for prior in selected_spans):
                selected_spans.append(span)
                selected_roles.setdefault(span.span_id, _role)
                selected_terms.update(_content_tokens(span.text))
        remaining = [p for p in remaining if p is not path]
        if estimate_token_count("\n\n".join([prefix, *pieces])) >= budget * 0.95:
            break

    ordered: List[Tuple[CandidateSpan, str]] = [
        (span, selected_roles.get(span.span_id, "support"))
        for span in selected_spans
    ]
    if not ordered:
        ranked = sorted(candidates, key=lambda s: (-s.final_score, s.source_order_index))
        ordered = [(span, "support") for span in _dedupe_candidates(ranked)]
    audit = {
        "path_count": len(selected_paths),
        "query_link_values": [path.query_link for path in selected_paths],
        "bridge_link_values": [path.bridge_link for path in selected_paths],
        "cross_page_path_count": sum(path.span_i.page_id != path.span_j.page_id for path in selected_paths),
        "same_page_path_count": sum(path.span_i.page_id == path.span_j.page_id for path in selected_paths),
    }
    return ordered, audit


def _stable_seed(seed: int, sample_id: str, task: str, variant: str) -> int:
    payload = f"{seed}:{task}:{sample_id}:{variant}".encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:16], 16)


def _select_random_bridge_paths(
    sample_id: str,
    task: str,
    query: str,
    candidates: Sequence[CandidateSpan],
    page_map: Mapping[int, Mapping[str, Any]],
    budget: int,
    neighbor_size: int = 32,
    seed: int = 0,
) -> Tuple[List[Tuple[CandidateSpan, str]], Dict[str, Any]]:
    prefix = _prefix(query, False)
    rng = random.Random(_stable_seed(seed, sample_id, task, "bridge_random_neighbor32_path_order"))
    query_spans = _dedupe_candidates(sorted(
        candidates,
        key=lambda s: (-s.query_score, -s.final_score, s.source_order_index),
    ))[:32]
    pool = _dedupe_candidates(sorted(
        candidates,
        key=lambda s: (-s.final_score, -s.query_score, s.source_order_index),
    ))[:96]
    pieces: List[str] = []
    selected: List[CandidateSpan] = []
    roles: Dict[str, str] = {}
    path_count = 0
    bridge_values: List[float] = []
    query_values: List[float] = []
    cross_page = 0
    same_page = 0
    for seed_span in query_spans:
        bridge_pool = [
            span for span in pool
            if span.span_id != seed_span.span_id
            and not _overlaps(span, seed_span, max_fraction=0.10)
            and not any(_overlaps(span, prior, max_fraction=0.30) for prior in selected)
        ]
        if not bridge_pool:
            continue
        bridge_span = rng.choice(bridge_pool)
        took_any = False
        for span, role in ((seed_span, "query"), (bridge_span, "bridge_random")):
            if any(_overlaps(span, prior, max_fraction=0.30) for prior in selected):
                continue
            piece, _ = _expanded_piece(span, page_map, neighbor_size, role=role, use_tags=False)
            if not _can_add(prefix, pieces, piece, budget):
                continue
            pieces.append(piece)
            selected.append(span)
            roles.setdefault(span.span_id, role)
            took_any = True
        if took_any:
            path_count += 1
            query_values.append(seed_span.query_score)
            bridge_values.append(bridge_link_score(seed_span, bridge_span))
            cross_page += int(seed_span.page_id != bridge_span.page_id)
            same_page += int(seed_span.page_id == bridge_span.page_id)
        if estimate_token_count("\n\n".join([prefix, *pieces])) >= budget * 0.95:
            break
    if not selected:
        selected = _query_only_path_order(candidates)[:1]
    ordered = [(span, roles.get(span.span_id, "support")) for span in selected]
    audit = {
        "path_count": path_count,
        "query_link_values": query_values,
        "bridge_link_values": bridge_values,
        "cross_page_path_count": cross_page,
        "same_page_path_count": same_page,
        "random_seed": seed,
    }
    return ordered, audit


VARIANT_SETTINGS: Dict[str, Dict[str, Any]] = {
    "head_truncate_source_order": {"neighbor": 0, "tags": False, "bridge": False, "head": True},
    "span_only_source_order": {"neighbor": 0, "tags": False, "bridge": False},
    "bm25_chunk_source_order": {"neighbor": 0, "tags": False, "bridge": False, "retrieval": "bm25"},
    "tfidf_chunk_source_order": {"neighbor": 0, "tags": False, "bridge": False, "retrieval": "tfidf"},
    "neighbor16_source_order": {"neighbor": 16, "tags": False, "bridge": False},
    "neighbor32_source_order": {"neighbor": 32, "tags": False, "bridge": False},
    "neighbor32_score_order": {"neighbor": 32, "tags": False, "bridge": False},
    "neighbor32_path_order": {"neighbor": 32, "tags": False, "bridge": False},
    "bridge_neighbor32_path_order": {"neighbor": 32, "tags": False, "bridge": True},
    "bridge_neighbor32_path_order_tags": {"neighbor": 32, "tags": True, "bridge": True},
    "query_only_neighbor32_path_order": {"neighbor": 32, "tags": False, "bridge": False, "query_only": True},
    "bridge_random_neighbor32_path_order": {"neighbor": 32, "tags": False, "bridge": True, "random_bridge": True},
    "bridge_same_spans_source_order": {"neighbor": 32, "tags": False, "bridge": True, "same_spans_source_order": True},
    "lexical_bridgepack_source_order": {"neighbor": 32, "tags": False, "bridge": False, "lexical_source": True},
    "lexical_bridgepack_path_order": {"neighbor": 32, "tags": False, "bridge": True, "lexical_path": True},
    "lexical_bridgepack_hybrid": {"neighbor": 32, "tags": False, "bridge": False, "lexical_hybrid": True},
    "full_context_4096": {"neighbor": 0, "tags": False, "bridge": False, "full_context": True},
}


def _pack_head_truncate(
    sample_id: str,
    task: str,
    query: str,
    context: str,
    variant: str,
    budget: int,
    page_size_tokens: int,
    full_context_tokens: Optional[int] = None,
) -> Tuple[PackedEvidence, Dict[str, Any]]:
    pages = split_into_pages(context, page_size_tokens=page_size_tokens)
    source_tokens = sum(len(page.get("tokens", [])) for page in pages)
    prefix = _prefix(query, False)
    token_budget = full_context_tokens if full_context_tokens is not None else budget
    context_text = _truncate_to_budget(context, token_budget)
    packed_text = "\n\n".join([prefix, context_text]).strip()
    truncated = estimate_token_count(packed_text) > budget
    if full_context_tokens is None and truncated:
        packed_text = _truncate_to_budget(packed_text, budget)
    packed_tokens = estimate_token_count(packed_text)
    selected_pages = [
        int(page["page_id"])
        for page in pages
        if int(page.get("start_token", 0)) < min(source_tokens, token_budget)
    ]
    packed = PackedEvidence(
        sample_id=sample_id,
        task=task,
        variant=variant,
        budget=budget,
        packed_text=packed_text,
        selected_span_ids=[f"head_t0_{min(source_tokens, token_budget)}"],
        selected_page_ids=selected_pages,
        packed_token_count=packed_tokens,
        source_token_count=source_tokens,
        compression_ratio=packed_tokens / max(1, source_tokens),
        ordering="source_order",
        neighbor_size=0,
        used_bridge_path=False,
        used_tags=False,
        truncated=truncated or source_tokens > token_budget,
    )
    audit = {
        "selected_spans": [],
        "expanded_records": [],
        "path_count": 0,
        "query_link_values": [],
        "bridge_link_values": [],
        "cross_page_path_count": 0,
        "same_page_path_count": 0,
    }
    return packed, audit


def pack_evidence(
    sample_id: str,
    task: str,
    query: str,
    context: str,
    variant: str,
    budget: int,
    page_size_tokens: int = 256,
    span_size_tokens: int = 64,
    stride_tokens: int = 32,
) -> Tuple[PackedEvidence, Dict[str, Any]]:
    if variant not in VARIANT_SETTINGS:
        raise ValueError(f"Unknown packing variant: {variant}")
    settings = VARIANT_SETTINGS[variant]
    if settings.get("head"):
        return _pack_head_truncate(
            sample_id,
            task,
            query,
            context,
            variant,
            budget,
            page_size_tokens=page_size_tokens,
        )
    if settings.get("full_context"):
        return _pack_head_truncate(
            sample_id,
            task,
            query,
            context,
            variant,
            budget,
            page_size_tokens=page_size_tokens,
            full_context_tokens=4096,
        )
    candidates, page_map = build_candidate_spans(
        sample_id,
        task,
        query,
        context,
        page_size_tokens=page_size_tokens,
        span_size_tokens=span_size_tokens,
        stride_tokens=stride_tokens,
    )
    if not candidates:
        empty = PackedEvidence(
            sample_id=sample_id,
            task=task,
            variant=variant,
            budget=budget,
            packed_text=_prefix(query, bool(settings["tags"])),
            selected_span_ids=[],
            selected_page_ids=[],
            packed_token_count=estimate_token_count(_prefix(query, bool(settings["tags"]))),
            source_token_count=0,
            compression_ratio=0.0,
            ordering="path_order" if "path_order" in variant else "source_order",
            neighbor_size=int(settings["neighbor"]),
            used_bridge_path=bool(settings["bridge"]),
            used_tags=bool(settings["tags"]),
            truncated=False,
        )
        return empty, {"selected_spans": [], "path_count": 0}

    if settings.get("retrieval") == "bm25":
        return _pack_retrieval_style(
            sample_id,
            task,
            query,
            variant,
            budget,
            candidates,
            page_map,
            bm25_scores(query, candidates),
        )

    if settings.get("retrieval") == "tfidf":
        return _pack_retrieval_style(
            sample_id,
            task,
            query,
            variant,
            budget,
            candidates,
            page_map,
            tfidf_scores(query, candidates),
        )

    if settings.get("query_only"):
        ranked = _query_only_path_order(candidates)
        selected = _select_ranked_for_budget(
            query,
            ranked,
            page_map,
            budget,
            neighbor_size=int(settings["neighbor"]),
            use_tags=False,
            max_fraction=0.30,
        )
        ordered = [(span, "query") for span in selected]
        packed, audit = _pack_from_ordered_spans(
            sample_id,
            task,
            query,
            variant,
            budget,
            ordered,
            page_map,
            neighbor_size=int(settings["neighbor"]),
            ordering="query_path_order",
            used_bridge_path=False,
            used_tags=False,
        )
        audit.update({"query_only_control": True})
        return packed, audit

    if settings.get("lexical_source"):
        lexical_candidates = lexical_scored_candidates(query, candidates)
        ranked = _dedupe_candidates(sorted(
            lexical_candidates,
            key=lambda s: (-s.final_score, -s.query_score, s.source_order_index),
        ))
        selected = _select_ranked_for_budget(
            query,
            ranked,
            page_map,
            budget,
            neighbor_size=int(settings["neighbor"]),
            use_tags=False,
            max_fraction=0.30,
        )
        ordered = [(span, "lexical") for span in sorted(selected, key=lambda s: s.source_order_index)]
        packed, audit = _pack_from_ordered_spans(
            sample_id,
            task,
            query,
            variant,
            budget,
            ordered,
            page_map,
            neighbor_size=int(settings["neighbor"]),
            ordering="source_order",
            used_bridge_path=False,
            used_tags=False,
        )
        audit.update({"lexical_initialized": True})
        return packed, audit

    if settings.get("lexical_path"):
        lexical_candidates = lexical_scored_candidates(query, candidates)
        ordered, path_audit = _select_bridge_paths(
            query,
            lexical_candidates,
            page_map,
            budget,
            use_tags=False,
            neighbor_size=int(settings["neighbor"]),
        )
        packed, audit = _pack_from_ordered_spans(
            sample_id,
            task,
            query,
            variant,
            budget,
            ordered,
            page_map,
            neighbor_size=int(settings["neighbor"]),
            ordering="lexical_bridge_path_order",
            used_bridge_path=True,
            used_tags=False,
        )
        audit.update(path_audit)
        audit.update({"lexical_initialized": True})
        return packed, audit

    if settings.get("lexical_hybrid"):
        hybrid_candidates = lexical_hybrid_candidates(query, candidates)
        ranked = _dedupe_candidates(sorted(
            hybrid_candidates,
            key=lambda s: (-s.final_score, -s.query_score, s.source_order_index),
        ))
        selected = _select_hybrid_for_budget(
            query,
            ranked,
            page_map,
            budget,
            neighbor_size=int(settings["neighbor"]),
            redundancy_penalty=0.3,
        )
        ordered = [(span, "lexical_hybrid") for span in sorted(selected, key=lambda s: s.source_order_index)]
        packed, audit = _pack_from_ordered_spans(
            sample_id,
            task,
            query,
            variant,
            budget,
            ordered,
            page_map,
            neighbor_size=int(settings["neighbor"]),
            ordering="source_order",
            used_bridge_path=False,
            used_tags=False,
        )
        audit.update({
            "lexical_initialized": True,
            "hybrid_weights": {
                "lexical": 1.0,
                "query": 0.8,
                "bridge": 0.4,
                "redundancy_penalty": 0.3,
            },
        })
        return packed, audit

    if settings.get("random_bridge"):
        ordered, path_audit = _select_random_bridge_paths(
            sample_id,
            task,
            query,
            candidates,
            page_map,
            budget,
            neighbor_size=int(settings["neighbor"]),
            seed=0,
        )
        packed, audit = _pack_from_ordered_spans(
            sample_id,
            task,
            query,
            variant,
            budget,
            ordered,
            page_map,
            neighbor_size=int(settings["neighbor"]),
            ordering="random_bridge_path_order",
            used_bridge_path=True,
            used_tags=False,
        )
        audit.update(path_audit)
        return packed, audit

    if settings.get("same_spans_source_order"):
        ordered_bridge, path_audit = _select_bridge_paths(
            query,
            candidates,
            page_map,
            budget,
            use_tags=False,
            neighbor_size=int(settings["neighbor"]),
        )
        selected: List[CandidateSpan] = []
        for span, _role in ordered_bridge:
            if span.span_id not in {prev.span_id for prev in selected}:
                selected.append(span)
        ordered = [(span, "bridge_selected") for span in sorted(selected, key=lambda s: s.source_order_index)]
        packed, audit = _pack_from_ordered_spans(
            sample_id,
            task,
            query,
            variant,
            budget,
            ordered,
            page_map,
            neighbor_size=int(settings["neighbor"]),
            ordering="source_order",
            used_bridge_path=True,
            used_tags=False,
        )
        audit.update(path_audit)
        audit.update({"same_spans_as_bridgepack_best_effort": True})
        return packed, audit

    if settings["bridge"]:
        ordered, path_audit = _select_bridge_paths(
            query,
            candidates,
            page_map,
            budget,
            use_tags=bool(settings["tags"]),
            neighbor_size=int(settings["neighbor"]),
        )
        packed, audit = _pack_from_ordered_spans(
            sample_id,
            task,
            query,
            variant,
            budget,
            ordered,
            page_map,
            neighbor_size=int(settings["neighbor"]),
            ordering="bridge_path_order",
            used_bridge_path=True,
            used_tags=bool(settings["tags"]),
        )
        audit.update(path_audit)
        return packed, audit

    ordered_spans, ordering = _independent_order(candidates, variant)
    ordered = [(span, "support") for span in ordered_spans]
    return _pack_from_ordered_spans(
        sample_id,
        task,
        query,
        variant,
        budget,
        ordered,
        page_map,
        neighbor_size=int(settings["neighbor"]),
        ordering=ordering,
        used_bridge_path=False,
        used_tags=False,
    )


def packed_to_json(packed: PackedEvidence) -> Dict[str, Any]:
    return asdict(packed)
