from __future__ import annotations

import math
import re
import statistics
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


FALLBACK_REASONS = [
    "no_raw_text",
    "no_query_text",
    "no_offsets",
    "no_document_boundary",
    "no_paragraph_boundary",
    "no_sentence_boundary",
    "too_short_units",
    "too_long_units",
    "coverage_gap",
    "overlap_error",
    "tokenizer_alignment_error",
    "kernel_incompatible",
    "fallback_fixed",
    "unknown",
]

_KNOWN_STRATEGIES = {
    "fixed",
    "semantic",
    "semantic_v2",
    "semantic_v3",
    "attention_boundary",
    "query_dynamic",
    "query_dynamic_v2",
    "query_dynamic_v3",
    "hierarchical",
    "hybrid_attention_fixed",
    "hybrid_attention_semantic",
}


@dataclass
class BlockLayout:
    block_ids: List[int]
    token_spans: List[Tuple[int, int]]
    block_lengths: List[int]
    block_text_preview: List[str]
    source_types: List[str]
    strategy_name: str
    metadata: Dict[str, Any]
    mapping_to_fixed_pages: List[List[int]]
    char_spans: List[Tuple[int, int]] = field(default_factory=list)
    fallback_used: bool = False
    fallback_reason: str = ""
    fallback_reason_counts: Dict[str, int] = field(default_factory=dict)
    expanded_fixed_pages: List[List[int]] = field(default_factory=list)
    semantic_span_tokens: List[int] = field(default_factory=list)
    expanded_page_tokens: List[int] = field(default_factory=list)
    page_expansion_overhead: List[float] = field(default_factory=list)
    boundary_confidence: List[float] = field(default_factory=list)
    boundary_score_left: List[float] = field(default_factory=list)
    boundary_score_right: List[float] = field(default_factory=list)
    coverage_ok: bool = True
    no_gap: bool = True
    no_overlap: bool = True
    ordered_spans: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "block_ids": [int(x) for x in self.block_ids],
            "token_spans": [[int(s), int(e)] for s, e in self.token_spans],
            "block_lengths": [int(x) for x in self.block_lengths],
            "block_text_preview": list(self.block_text_preview),
            "source_types": list(self.source_types),
            "strategy_name": self.strategy_name,
            "metadata": self.metadata,
            "mapping_to_fixed_pages": [[int(x) for x in pages] for pages in self.mapping_to_fixed_pages],
            "char_spans": [[int(s), int(e)] for s, e in self.char_spans],
            "fallback_used": bool(self.fallback_used),
            "fallback_reason": str(self.fallback_reason),
            "fallback_reason_counts": {k: int(v) for k, v in self.fallback_reason_counts.items()},
            "expanded_fixed_pages": [[int(x) for x in pages] for pages in self.expanded_fixed_pages],
            "semantic_span_tokens": [int(x) for x in self.semantic_span_tokens],
            "expanded_page_tokens": [int(x) for x in self.expanded_page_tokens],
            "page_expansion_overhead": [float(x) for x in self.page_expansion_overhead],
            "boundary_confidence": [float(x) for x in self.boundary_confidence],
            "boundary_score_left": [float(x) for x in self.boundary_score_left],
            "boundary_score_right": [float(x) for x in self.boundary_score_right],
            "coverage_ok": bool(self.coverage_ok),
            "no_gap": bool(self.no_gap),
            "no_overlap": bool(self.no_overlap),
            "ordered_spans": bool(self.ordered_spans),
        }


_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?;:。！？；：])\s+|(?<=[.!?;:。！？；：])(?=[A-Z0-9])")
_SOFT_BOUNDARY_RE = re.compile(r"[,，、]\s+|\s+-\s+")
_DOC_START_RE = re.compile(
    r"(?im)^\s*(?:#{1,6}\s+|={2,}\s*$|(?:Passage|Document|Doc|Article|Title|Section|Paper|Abstract)\s*\d*[:：]|\[[^\]]{1,80}\]\s*$)"
)
_QUESTION_RE = re.compile(r"(?is)(?:^|\n)\s*(?:question|query|问题)\s*[:：]\s*(.+?)(?:\n\s*(?:answer|context|passage)\s*[:：]|\Z)")
_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _as_list(input_ids: Any) -> List[int]:
    if hasattr(input_ids, "detach"):
        return [int(x) for x in input_ids.detach().cpu().reshape(-1).tolist()]
    if isinstance(input_ids, (list, tuple)):
        if input_ids and isinstance(input_ids[0], (list, tuple)):
            return [int(x) for x in input_ids[0]]
        return [int(x) for x in input_ids]
    return []


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _token_len(tokenizer: Any, text: str) -> int:
    if not text:
        return 0
    try:
        return int(len(tokenizer(text, add_special_tokens=False).input_ids))
    except Exception:
        return max(1, len(text.split()))


def _decode_tokens(tokenizer: Any, ids: Sequence[int]) -> str:
    try:
        return tokenizer.decode(
            list(ids),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    except Exception:
        return ""


def _fixed_pages_for_span(start: int, end: int, fixed_page_tokens: int, seq_len: int) -> List[int]:
    if end <= start or fixed_page_tokens <= 0:
        return []
    first = max(0, int(start)) // fixed_page_tokens
    last = max(0, int(end) - 1) // fixed_page_tokens
    max_page = max(0, (max(1, seq_len) - 1) // fixed_page_tokens)
    return [int(x) for x in range(first, min(last, max_page) + 1)]


def _stats(values: Sequence[float]) -> Dict[str, float]:
    vals = [float(x) for x in values]
    if not vals:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(statistics.mean(vals)),
        "std": float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0,
        "min": float(min(vals)),
        "max": float(max(vals)),
    }


def _empty_reason_counts() -> Dict[str, int]:
    return {name: 0 for name in FALLBACK_REASONS}


def _bump(counts: Dict[str, int], reason: str, amount: int = 1) -> None:
    counts[reason if reason in FALLBACK_REASONS else "unknown"] = int(counts.get(reason if reason in FALLBACK_REASONS else "unknown", 0)) + int(amount)


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        parts: List[str] = []
        for key in ["title", "question", "query", "abstract", "text", "content", "paragraphs", "sentences"]:
            if key in value:
                parts.append(_stringify(value.get(key)))
        if not parts:
            parts.extend(_stringify(v) for v in value.values())
        return "\n".join(p for p in parts if p)
    if isinstance(value, (list, tuple)):
        return "\n\n".join(_stringify(v) for v in value if _stringify(v))
    return str(value)


def _extract_prompt_text(sample: Optional[Dict[str, Any]], prompt_text: Optional[str], decoded_text: str) -> Tuple[str, str]:
    # The final prompt is the only text guaranteed to cover every token. Sample
    # fields are still inspected below for query extraction and metadata.
    if prompt_text:
        return prompt_text, "prompt_text"
    sample = sample or {}
    for key in ["context", "input", "prompt", "documents", "article", "text"]:
        text = _stringify(sample.get(key))
        if text.strip():
            return text, f"sample.{key}"
    if decoded_text:
        return decoded_text, "decoded_input_ids"
    return "", "none"


def extract_query_text(sample: Optional[Dict[str, Any]], prompt_text: Optional[str] = None, query_text: Optional[str] = None) -> Tuple[str, str, str]:
    if query_text and str(query_text).strip():
        return str(query_text).strip(), "provided", ""
    sample = sample or {}
    for key in ["question", "query"]:
        text = _stringify(sample.get(key)).strip()
        if text:
            return text, f"sample.{key}", ""
    raw_input = _stringify(sample.get("input")).strip()
    match = _QUESTION_RE.search(raw_input)
    if match:
        return match.group(1).strip()[:1200], "sample.input_question_marker", ""
    prompt = prompt_text or _stringify(sample.get("prompt"))
    match = _QUESTION_RE.search(prompt or "")
    if match:
        return match.group(1).strip()[:1200], "prompt_question_marker", ""
    tail = (prompt or "").strip()[-1200:]
    if tail:
        return tail, "prompt_tail", "prompt_tail_fallback"
    return "", "none", "no_query_text"


def _spans_from_boundaries(text: str, starts: List[int], source_type: str) -> List[Tuple[int, int, str]]:
    clean = sorted({0, *[int(s) for s in starts if 0 <= int(s) < len(text)], len(text)})
    spans: List[Tuple[int, int, str]] = []
    for left, right in zip(clean[:-1], clean[1:]):
        piece = text[left:right]
        if piece.strip():
            offset = len(piece) - len(piece.lstrip())
            rstrip = len(piece.rstrip())
            spans.append((left + offset, left + rstrip, source_type))
    return spans


def _split_paragraph_spans(text: str) -> List[Tuple[int, int, str]]:
    spans: List[Tuple[int, int, str]] = []
    cursor = 0
    for match in re.finditer(r"\n\s*\n+", text):
        piece = text[cursor:match.start()]
        if piece.strip():
            offset = len(piece) - len(piece.lstrip())
            spans.append((cursor + offset, cursor + len(piece.rstrip()), "paragraph"))
        cursor = match.end()
    piece = text[cursor:]
    if piece.strip():
        offset = len(piece) - len(piece.lstrip())
        spans.append((cursor + offset, cursor + len(piece.rstrip()), "paragraph"))
    return spans


def _split_line_spans(text: str, source_type: str) -> List[Tuple[int, int, str]]:
    spans: List[Tuple[int, int, str]] = []
    cursor = 0
    for line in text.splitlines(True):
        raw = line.rstrip("\n\r")
        if raw.strip():
            offset = len(raw) - len(raw.lstrip())
            spans.append((cursor + offset, cursor + len(raw.rstrip()), source_type))
        cursor += len(line)
    return spans


def _split_regex_spans(text: str, regex: re.Pattern[str], source_type: str) -> List[Tuple[int, int, str]]:
    spans: List[Tuple[int, int, str]] = []
    cursor = 0
    for match in regex.finditer(text):
        end = match.end()
        piece = text[cursor:end]
        if piece.strip():
            offset = len(piece) - len(piece.lstrip())
            spans.append((cursor + offset, cursor + len(piece.rstrip()), source_type))
        cursor = end
    piece = text[cursor:]
    if piece.strip():
        offset = len(piece) - len(piece.lstrip())
        spans.append((cursor + offset, cursor + len(piece.rstrip()), source_type))
    return spans


def _task_aware_units(text: str, task_name: str, counts: Dict[str, int]) -> Tuple[List[Tuple[str, str, Tuple[int, int]]], Dict[str, Any]]:
    meta: Dict[str, Any] = {}
    if not text.strip():
        _bump(counts, "no_raw_text")
        return [], meta

    doc_starts = [m.start() for m in _DOC_START_RE.finditer(text)]
    if len(doc_starts) > 1:
        coarse = _spans_from_boundaries(text, doc_starts, "document")
        meta["document_boundary_count"] = len(coarse)
    else:
        _bump(counts, "no_document_boundary")
        coarse = [(0, len(text), "document")]
        meta["document_boundary_count"] = 0

    task = (task_name or "").lower()
    units: List[Tuple[int, int, str]] = []
    paragraph_count = 0
    sentence_count = 0
    for start, end, source in coarse:
        piece = text[start:end]
        if task in {"samsum"}:
            sub = _split_line_spans(piece, "dialogue_turn")
        elif task in {"lcc", "repobench-p"}:
            sub = _split_paragraph_spans(piece)
            if len(sub) <= 1:
                sub = _split_line_spans(piece, "code_line")
        else:
            sub = _split_paragraph_spans(piece)
        if len(sub) <= 1:
            _bump(counts, "no_paragraph_boundary")
            sub = [(0, len(piece), source)]
        else:
            paragraph_count += len(sub)
        for ps, pe, psource in sub:
            paragraph = piece[ps:pe]
            if task in {"gov_report", "qmsum", "multi_news", "vcsum"}:
                sent_spans = [(0, len(paragraph), psource)]
            else:
                sent_spans = _split_regex_spans(paragraph, _SENTENCE_BOUNDARY_RE, "sentence")
                if len(sent_spans) <= 1:
                    _bump(counts, "no_sentence_boundary")
                    soft = _split_regex_spans(paragraph, _SOFT_BOUNDARY_RE, "soft_boundary")
                    sent_spans = soft if len(soft) > 1 else [(0, len(paragraph), psource)]
                else:
                    sentence_count += len(sent_spans)
            for ss, se, ssource in sent_spans:
                units.append((start + ps + ss, start + ps + se, ssource))

    out = [(text[s:e], source, (s, e)) for s, e, source in units if e > s and text[s:e].strip()]
    meta["paragraph_unit_count"] = int(paragraph_count)
    meta["sentence_unit_count"] = int(sentence_count)
    meta["semantic_unit_count"] = int(len(out))
    return out, meta


def _split_unit_by_tokens(
    tokenizer: Any,
    text: str,
    source: str,
    char_span: Tuple[int, int],
    max_tokens: int,
    counts: Dict[str, int],
) -> List[Tuple[str, str, Tuple[int, int], int]]:
    length = max(1, _token_len(tokenizer, text))
    if length <= max_tokens:
        return [(text, source, char_span, length)]

    _bump(counts, "too_long_units")
    pieces: List[Tuple[str, str, Tuple[int, int], int]] = []
    for split_re, split_source in [(_SENTENCE_BOUNDARY_RE, "sentence"), (_SOFT_BOUNDARY_RE, "soft_boundary")]:
        sub_spans = _split_regex_spans(text, split_re, split_source)
        if len(sub_spans) > 1:
            for ss, se, ssource in sub_spans:
                pieces.extend(_split_unit_by_tokens(tokenizer, text[ss:se], ssource, (char_span[0] + ss, char_span[0] + se), max_tokens, counts))
            return pieces

    try:
        ids = tokenizer(text, add_special_tokens=False).input_ids
    except Exception:
        ids = []
    if ids:
        _bump(counts, "fallback_fixed")
        for start in range(0, len(ids), max(1, int(max_tokens))):
            piece_ids = ids[start:start + max(1, int(max_tokens))]
            piece = _decode_tokens(tokenizer, piece_ids)
            if piece.strip():
                pieces.append((piece, "fallback_fixed", char_span, max(1, len(piece_ids))))
    if not pieces:
        _bump(counts, "fallback_fixed")
        pieces.append((text, "fallback_fixed", char_span, length))
    return pieces


def _greedy_merge_units(
    tokenizer: Any,
    units: Sequence[Tuple[str, str, Tuple[int, int], int]],
    target_block_tokens: int,
    min_block_tokens: int,
    max_block_tokens: int,
) -> List[Tuple[str, str, Tuple[int, int], int]]:
    merged: List[Tuple[str, str, Tuple[int, int], int]] = []
    cur_text = ""
    cur_sources: List[str] = []
    cur_start: Optional[int] = None
    cur_end: Optional[int] = None
    cur_len = 0
    target = max(1, int(target_block_tokens))
    max_len = max(target, int(max_block_tokens))
    min_len = max(1, int(min_block_tokens))

    def flush() -> None:
        nonlocal cur_text, cur_sources, cur_start, cur_end, cur_len
        if not cur_text:
            return
        source = cur_sources[0] if len(set(cur_sources)) == 1 else "merged"
        merged.append((cur_text, source, (int(cur_start or 0), int(cur_end or cur_start or 0)), max(1, int(cur_len))))
        cur_text = ""
        cur_sources = []
        cur_start = None
        cur_end = None
        cur_len = 0

    for text, source, span, length in units:
        length = max(1, int(length or _token_len(tokenizer, text)))
        if cur_text and cur_len + length > max_len:
            flush()
        if cur_text and cur_len + length > target and cur_len >= min_len:
            flush()
        sep = "\n\n" if cur_text and source in {"document", "paragraph"} else " "
        cur_text = (cur_text + sep + text) if cur_text else text
        cur_sources.append(source)
        cur_start = span[0] if cur_start is None else min(cur_start, span[0])
        cur_end = span[1] if cur_end is None else max(cur_end, span[1])
        cur_len += length
    flush()
    return merged


def _tokenizer_offsets_available(tokenizer: Any, text: str) -> bool:
    try:
        enc = tokenizer(text[:4096], add_special_tokens=False, return_offsets_mapping=True)
        return bool(getattr(enc, "offset_mapping", None) is not None or (isinstance(enc, dict) and enc.get("offset_mapping") is not None))
    except Exception:
        return False


def _layout_from_blocks(
    *,
    input_ids: Sequence[int],
    tokenizer: Any,
    strategy_name: str,
    blocks: Sequence[Tuple[str, str, Tuple[int, int], int]],
    fixed_page_tokens: int,
    metadata: Dict[str, Any],
    counts: Dict[str, int],
    alignment_mode: str = "tail_fatal",
) -> BlockLayout:
    seq_len = len(input_ids)
    spans: List[Tuple[int, int]] = []
    char_spans: List[Tuple[int, int]] = []
    source_types: List[str] = []
    block_list = list(blocks)
    if alignment_mode == "rescale_to_prompt" and block_list and seq_len > 0:
        raw_lengths = [max(1, int(length)) for _, _, _, length in block_list]
        raw_total = max(1, sum(raw_lengths))
        scaled = [max(1, int(round(length * seq_len / raw_total))) for length in raw_lengths]
        diff = seq_len - sum(scaled)
        idx = len(scaled) - 1
        while diff != 0 and scaled:
            if diff > 0:
                scaled[idx] += 1
                diff -= 1
            elif scaled[idx] > 1:
                scaled[idx] -= 1
                diff += 1
            idx = (idx - 1) % len(scaled)
        cursor = 0
        for (text, source, char_span, _), length in zip(block_list, scaled):
            if cursor >= seq_len:
                break
            end = min(seq_len, cursor + max(1, int(length)))
            spans.append((cursor, end))
            char_spans.append((int(char_span[0]), int(char_span[1])))
            source_types.append(source)
            cursor = end
        metadata["token_span_alignment_mode"] = "rescale_to_prompt"
        metadata["token_length_rescale_factor"] = float(seq_len / raw_total)
    else:
        cursor = 0
        for text, source, char_span, length in block_list:
            if cursor >= seq_len:
                break
            end = min(seq_len, cursor + max(1, int(length)))
            spans.append((cursor, end))
            char_spans.append((int(char_span[0]), int(char_span[1])))
            source_types.append(source)
            cursor = end
        if cursor < seq_len:
            spans.append((cursor, seq_len))
            char_spans.append((0, 0))
            source_types.append("tail_alignment")
            _bump(counts, "tokenizer_alignment_error")
    if spans:
        spans[-1] = (spans[-1][0], seq_len)
    lengths = [max(0, e - s) for s, e in spans]
    previews = [_decode_tokens(tokenizer, input_ids[s:e])[:160].replace("\n", "\\n") for s, e in spans]
    pages = [_fixed_pages_for_span(s, e, fixed_page_tokens, seq_len) for s, e in spans]
    expanded_tokens = [int(len(p) * fixed_page_tokens) for p in pages]
    overhead = [float(et / max(1, length)) for et, length in zip(expanded_tokens, lengths)]
    metadata["block_len_stats"] = _stats(lengths)
    metadata["page_expansion_overhead_stats"] = _stats(overhead)
    metadata["fallback_reason_counts"] = {k: int(counts.get(k, 0)) for k in FALLBACK_REASONS}
    fatal_reasons = ["no_raw_text", "coverage_gap", "overlap_error", "tokenizer_alignment_error", "kernel_incompatible", "unknown"]
    fallback_used = any(int(counts.get(reason, 0)) > 0 for reason in fatal_reasons)
    fallback_reason = ";".join(reason for reason in fatal_reasons if int(counts.get(reason, 0)) > 0)
    layout = BlockLayout(
        block_ids=list(range(len(spans))),
        token_spans=spans,
        block_lengths=lengths,
        block_text_preview=previews,
        source_types=source_types,
        strategy_name=strategy_name,
        metadata=metadata,
        mapping_to_fixed_pages=pages,
        char_spans=char_spans,
        fallback_used=fallback_used,
        fallback_reason=fallback_reason,
        fallback_reason_counts={k: int(counts.get(k, 0)) for k in FALLBACK_REASONS},
        expanded_fixed_pages=pages,
        semantic_span_tokens=lengths,
        expanded_page_tokens=expanded_tokens,
        page_expansion_overhead=overhead,
    )
    validate_block_layout(layout, seq_len)
    return layout


def validate_block_layout(layout: BlockLayout, input_len: int) -> BlockLayout:
    spans = [(int(s), int(e)) for s, e in layout.token_spans]
    ordered = all(spans[i][0] <= spans[i][1] and (i == 0 or spans[i - 1][0] <= spans[i][0]) for i in range(len(spans)))
    no_gap = bool(spans and spans[0][0] == 0 and spans[-1][1] == int(input_len))
    no_overlap = True
    legal = True
    for i, (s, e) in enumerate(spans):
        if s < 0 or e < s or e > int(input_len):
            legal = False
        if i > 0:
            if spans[i - 1][1] != s:
                no_gap = False
            if spans[i - 1][1] > s:
                no_overlap = False
    layout.ordered_spans = bool(ordered)
    layout.no_gap = bool(no_gap)
    layout.no_overlap = bool(no_overlap)
    layout.coverage_ok = bool(ordered and no_gap and no_overlap and legal)
    if not layout.coverage_ok:
        counts = dict(layout.fallback_reason_counts or _empty_reason_counts())
        if not no_gap:
            _bump(counts, "coverage_gap")
        if not no_overlap:
            _bump(counts, "overlap_error")
        if not legal:
            _bump(counts, "tokenizer_alignment_error")
        layout.fallback_reason_counts = counts
        layout.fallback_used = True
        layout.fallback_reason = ";".join(k for k in FALLBACK_REASONS if counts.get(k, 0) and k in {"coverage_gap", "overlap_error", "tokenizer_alignment_error"})
    if not layout.expanded_fixed_pages:
        layout.expanded_fixed_pages = layout.mapping_to_fixed_pages
    if not layout.semantic_span_tokens:
        layout.semantic_span_tokens = [int(e - s) for s, e in spans]
    if not layout.expanded_page_tokens:
        page_size = 64
        layout.expanded_page_tokens = [len(pages) * page_size for pages in layout.expanded_fixed_pages]
    if not layout.page_expansion_overhead:
        layout.page_expansion_overhead = [
            float(tokens / max(1, span_tokens))
            for tokens, span_tokens in zip(layout.expanded_page_tokens, layout.semantic_span_tokens)
        ]
    return layout


def _fixed_layout(input_ids: Sequence[int], tokenizer: Any, block_tokens: int) -> BlockLayout:
    seq_len = len(input_ids)
    block_tokens = max(1, int(block_tokens))
    spans = [(s, min(seq_len, s + block_tokens)) for s in range(0, seq_len, block_tokens)]
    lengths = [e - s for s, e in spans]
    previews = [_decode_tokens(tokenizer, input_ids[s:e])[:160].replace("\n", "\\n") for s, e in spans]
    pages = [[i] for i in range(len(spans))]
    expanded = [max(1, len(p)) * block_tokens for p in pages]
    layout = BlockLayout(
        block_ids=list(range(len(spans))),
        token_spans=spans,
        block_lengths=lengths,
        block_text_preview=previews,
        source_types=["fixed"] * len(spans),
        strategy_name="fixed",
        metadata={"block_len_stats": _stats(lengths), "fallback_used": False, "fallback_reason": "", "fallback_reason_counts": _empty_reason_counts()},
        mapping_to_fixed_pages=pages,
        char_spans=[(0, 0) for _ in spans],
        fallback_used=False,
        fallback_reason="",
        fallback_reason_counts=_empty_reason_counts(),
        expanded_fixed_pages=pages,
        semantic_span_tokens=lengths,
        expanded_page_tokens=expanded,
        page_expansion_overhead=[float(x / max(1, y)) for x, y in zip(expanded, lengths)],
    )
    return validate_block_layout(layout, seq_len)


def _semantic_layout_v2(
    *,
    input_ids: Sequence[int],
    tokenizer: Any,
    sample: Optional[Dict[str, Any]],
    prompt_text: Optional[str],
    task_name: str,
    target_block_tokens: int,
    min_block_tokens: int,
    max_block_tokens: int,
    fixed_page_tokens: int,
    config: Dict[str, Any],
    strategy_name: str = "semantic_v2",
) -> Tuple[BlockLayout, List[Tuple[str, str, Tuple[int, int], int]]]:
    ids = list(input_ids)
    decoded = _decode_tokens(tokenizer, ids)
    text, text_source = _extract_prompt_text(sample, prompt_text, decoded)
    counts = _empty_reason_counts()
    if not text.strip():
        text = decoded
        _bump(counts, "no_raw_text")
    if not _tokenizer_offsets_available(tokenizer, text):
        _bump(counts, "no_offsets")
    units, parse_meta = _task_aware_units(text, task_name, counts)
    split_units: List[Tuple[str, str, Tuple[int, int], int]] = []
    for unit_text, source, char_span in units:
        split_units.extend(_split_unit_by_tokens(tokenizer, unit_text, source, char_span, max_block_tokens, counts))
    if not split_units:
        _bump(counts, "fallback_fixed")
        fixed = _fixed_layout(ids, tokenizer, fixed_page_tokens)
        fixed.strategy_name = strategy_name
        fixed.metadata.update({"raw_text_source": text_source, "fallback_reason_counts": counts})
        fixed.fallback_reason_counts = counts
        fixed.fallback_used = True
        fixed.fallback_reason = "fallback_fixed"
        return fixed, []
    if any(length < int(min_block_tokens) for _, _, _, length in split_units):
        _bump(counts, "too_short_units", sum(1 for _, _, _, length in split_units if length < int(min_block_tokens)))
    merged = _greedy_merge_units(tokenizer, split_units, target_block_tokens, min_block_tokens, max_block_tokens)
    metadata = {
        "semantic_parser": config.get("semantic_parser", "task_aware"),
        "raw_text_source": text_source,
        "raw_text_chars": int(len(text)),
        "offset_alignment_success_rate": 0.0 if counts.get("no_offsets", 0) else 1.0,
        "source_unit_count": int(len(units)),
        "split_unit_count": int(len(split_units)),
        "merged_block_count": int(len(merged)),
        "target_block_tokens": int(target_block_tokens),
        "min_block_tokens": int(min_block_tokens),
        "max_block_tokens": int(max_block_tokens),
        "block_overlap_tokens": 0,
        "max_page_expansion_overhead": float(config.get("max_page_expansion_overhead", 2.0)),
        **parse_meta,
    }
    alignment_mode = "rescale_to_prompt" if strategy_name in {"semantic_v3", "query_dynamic_v3_base"} else "tail_fatal"
    layout = _layout_from_blocks(
        input_ids=ids,
        tokenizer=tokenizer,
        strategy_name=strategy_name,
        blocks=merged,
        fixed_page_tokens=fixed_page_tokens,
        metadata=metadata,
        counts=counts,
        alignment_mode=alignment_mode,
    )
    high_overhead = sum(1 for value in layout.page_expansion_overhead if value > float(config.get("max_page_expansion_overhead", 2.0)))
    layout.metadata["high_page_expansion_block_count"] = int(high_overhead)
    return layout, split_units


def _query_features(text: str) -> Dict[str, set]:
    toks = {t.lower() for t in _TOKEN_RE.findall(text or "") if len(t) > 1}
    nums = set(re.findall(r"\d[\d,.]*", text or ""))
    ents = set(re.findall(r"\b[A-Z][A-Za-z0-9_]{2,}\b", text or ""))
    return {"tokens": toks, "numbers": nums, "entities": ents}


def _query_score(unit_text: str, query_text: str, avg_len: float) -> float:
    q = _query_features(query_text)
    u = _query_features(unit_text)
    if not q["tokens"]:
        return 0.0
    overlap = len(q["tokens"] & u["tokens"]) / max(1, len(q["tokens"]))
    num_bonus = 0.15 if q["numbers"] & u["numbers"] else 0.0
    ent_bonus = 0.10 * min(1.0, len(q["entities"] & u["entities"]) / max(1, len(q["entities"])))
    toks = [t.lower() for t in _TOKEN_RE.findall(unit_text or "")]
    tf: Dict[str, int] = {}
    for tok in toks:
        tf[tok] = tf.get(tok, 0) + 1
    bm25 = 0.0
    for tok in q["tokens"]:
        freq = tf.get(tok, 0)
        if freq:
            denom = freq + 1.2 * (1.0 - 0.75 + 0.75 * len(toks) / max(1.0, avg_len))
            bm25 += math.log1p(freq * 2.2 / max(denom, 1e-6))
    bm25 = bm25 / max(1, len(q["tokens"]))
    bm25_norm = min(1.0, bm25 / (bm25 + 1.0)) if bm25 > 0 else 0.0
    title_bonus = 0.05 if re.search(r"(?im)^\s*(?:title|#)", unit_text or "") and (q["tokens"] & u["tokens"]) else 0.0
    return float(min(1.0, 0.45 * overlap + 0.45 * bm25_norm + num_bonus + ent_bonus + title_bonus))


def _query_dynamic_layout_v2(
    *,
    input_ids: Sequence[int],
    tokenizer: Any,
    sample: Optional[Dict[str, Any]],
    prompt_text: Optional[str],
    task_name: str,
    query_text: Optional[str],
    target_block_tokens: int,
    min_block_tokens: int,
    max_block_tokens: int,
    fixed_page_tokens: int,
    config: Dict[str, Any],
    strategy_name: str = "query_dynamic_v2",
) -> BlockLayout:
    base_strategy = "query_dynamic_v3_base" if strategy_name == "query_dynamic_v3" else "query_dynamic_v2_base"
    base_layout, semantic_units = _semantic_layout_v2(
        input_ids=input_ids,
        tokenizer=tokenizer,
        sample=sample,
        prompt_text=prompt_text,
        task_name=task_name,
        target_block_tokens=target_block_tokens,
        min_block_tokens=min_block_tokens,
        max_block_tokens=max_block_tokens,
        fixed_page_tokens=fixed_page_tokens,
        config=config,
        strategy_name=base_strategy,
    )
    query, source, fallback_reason = extract_query_text(sample, prompt_text, query_text)
    counts = dict(base_layout.fallback_reason_counts or _empty_reason_counts())
    if fallback_reason == "no_query_text":
        _bump(counts, "no_query_text")
    units = semantic_units
    if not units:
        units = [
            (preview, source_type, char_span, length)
            for preview, source_type, char_span, length in zip(
                base_layout.block_text_preview,
                base_layout.source_types,
                base_layout.char_spans,
                base_layout.block_lengths,
            )
        ]
    avg_len = statistics.mean([max(1, int(u[3])) for u in units]) if units else 64.0
    scores = [_query_score(unit_text, query, avg_len) for unit_text, _, _, _ in units]
    top_ratio = float(config.get("query_top_ratio", 0.2))
    top_n = max(1, int(math.ceil(len(scores) * top_ratio))) if scores else 0
    high = set(sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n])
    expanded_high = set(high)
    neighbors = max(0, int(config.get("query_neighbor_sentences", 1)))
    for idx in list(high):
        for delta in range(1, neighbors + 1):
            if idx - delta >= 0:
                expanded_high.add(idx - delta)
            if idx + delta < len(scores):
                expanded_high.add(idx + delta)
    high_max = max(1, int(config.get("query_high_max_block_tokens", 64)))
    low_max = max(high_max, int(config.get("query_low_max_block_tokens", 128)))
    dynamic_units: List[Tuple[str, str, Tuple[int, int], int]] = []
    for idx, (unit_text, source_type, char_span, _) in enumerate(units):
        cap = high_max if idx in expanded_high else low_max
        pieces = _split_unit_by_tokens(tokenizer, unit_text, "query_high" if idx in expanded_high else source_type, char_span, cap, counts)
        dynamic_units.extend(pieces)
    merged = _greedy_merge_units(tokenizer, dynamic_units, target_block_tokens, min_block_tokens, low_max)
    metadata = dict(base_layout.metadata)
    metadata.update({
        "query_score_mode": config.get("query_score_mode", "lexical_bm25"),
        "E_query_stats": _stats(scores),
        "E_query_mean": _stats(scores)["mean"],
        "E_query_std": _stats(scores)["std"],
        "high_query_units_count": int(len(expanded_high)),
        "query_tokens": sorted(_query_features(query)["tokens"])[:64],
        "query_extraction_source": source,
        "query_fallback_reason": fallback_reason,
        "query_top_ratio": float(top_ratio),
        "query_neighbor_sentences": int(neighbors),
        "query_high_max_block_tokens": int(high_max),
        "query_low_max_block_tokens": int(low_max),
    })
    alignment_mode = "rescale_to_prompt" if strategy_name == "query_dynamic_v3" else "tail_fatal"
    layout = _layout_from_blocks(
        input_ids=list(input_ids),
        tokenizer=tokenizer,
        strategy_name=strategy_name,
        blocks=merged,
        fixed_page_tokens=fixed_page_tokens,
        metadata=metadata,
        counts=counts,
        alignment_mode=alignment_mode,
    )
    return layout


def _legacy_semantic_layout(
    *,
    input_ids: Sequence[int],
    tokenizer: Any,
    target_block_tokens: int,
    min_block_tokens: int,
    max_block_tokens: int,
    fixed_page_tokens: int,
) -> BlockLayout:
    layout, _ = _semantic_layout_v2(
        input_ids=input_ids,
        tokenizer=tokenizer,
        sample={},
        prompt_text=_decode_tokens(tokenizer, input_ids),
        task_name="",
        target_block_tokens=target_block_tokens,
        min_block_tokens=min_block_tokens,
        max_block_tokens=max_block_tokens,
        fixed_page_tokens=fixed_page_tokens,
        config={},
        strategy_name="semantic",
    )
    # Preserve the v1 conservative signal for old reports: fixed subchunks count
    # as fallback for legacy semantic, but not for semantic_v2.
    if layout.fallback_reason_counts.get("fallback_fixed", 0):
        layout.fallback_used = True
        layout.fallback_reason = "long_semantic_unit_split_to_fixed_chunks"
    return layout


def build_blocks(
    input_ids: Any,
    tokenizer: Any,
    sample: Optional[Dict[str, Any]] = None,
    prompt_text: Optional[str] = None,
    task_name: Optional[str] = None,
    block_strategy: str = "fixed",
    target_block_tokens: int = 64,
    min_block_tokens: int = 32,
    max_block_tokens: int = 128,
    model_states: Optional[Dict[str, Any]] = None,
    query_text: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> BlockLayout:
    """Build a prompt-covering block layout.

    The interface is intentionally usable before model profiling. Semantic and
    query-dynamic layouts are materialized here; attention-boundary,
    hierarchical, and hybrid layouts are materialized later in the cache path
    from page-level scores/reps.
    """
    del model_states
    if isinstance(prompt_text, str) and prompt_text.lower() in _KNOWN_STRATEGIES and block_strategy == "fixed":
        block_strategy = prompt_text
        prompt_text = None
    cfg = config or {}
    fixed_page_tokens = _safe_int(cfg.get("fixed_page_tokens", cfg.get("block_size", target_block_tokens)), target_block_tokens)
    ids = _as_list(input_ids)
    strategy = (block_strategy or "fixed").lower()
    if strategy == "fixed":
        return _fixed_layout(ids, tokenizer, fixed_page_tokens)
    if strategy == "semantic":
        return _legacy_semantic_layout(
            input_ids=ids,
            tokenizer=tokenizer,
            target_block_tokens=target_block_tokens,
            min_block_tokens=min_block_tokens,
            max_block_tokens=max_block_tokens,
            fixed_page_tokens=fixed_page_tokens,
        )
    if strategy == "semantic_v2":
        layout, _ = _semantic_layout_v2(
            input_ids=ids,
            tokenizer=tokenizer,
            sample=sample,
            prompt_text=prompt_text,
            task_name=task_name or "",
            target_block_tokens=int(cfg.get("semantic_target_block_tokens", target_block_tokens)),
            min_block_tokens=int(cfg.get("semantic_min_block_tokens", min_block_tokens)),
            max_block_tokens=int(cfg.get("semantic_max_block_tokens", max_block_tokens)),
            fixed_page_tokens=fixed_page_tokens,
            config=cfg,
            strategy_name="semantic_v2",
        )
        return layout
    if strategy == "semantic_v3":
        layout, _ = _semantic_layout_v2(
            input_ids=ids,
            tokenizer=tokenizer,
            sample=sample,
            prompt_text=prompt_text,
            task_name=task_name or "",
            target_block_tokens=int(cfg.get("semantic_target_block_tokens", target_block_tokens)),
            min_block_tokens=int(cfg.get("semantic_min_block_tokens", min_block_tokens)),
            max_block_tokens=int(cfg.get("semantic_max_block_tokens", max_block_tokens)),
            fixed_page_tokens=fixed_page_tokens,
            config={**cfg, "semantic_version": "v3"},
            strategy_name="semantic_v3",
        )
        return layout
    if strategy == "query_dynamic":
        return _query_dynamic_layout_v2(
            input_ids=ids,
            tokenizer=tokenizer,
            sample=sample,
            prompt_text=prompt_text,
            task_name=task_name or "",
            query_text=query_text,
            target_block_tokens=target_block_tokens,
            min_block_tokens=min_block_tokens,
            max_block_tokens=max_block_tokens,
            fixed_page_tokens=fixed_page_tokens,
            config={**cfg, "query_high_max_block_tokens": cfg.get("query_high_max_block_tokens", 64), "query_low_max_block_tokens": cfg.get("query_low_max_block_tokens", 128)},
        )
    if strategy == "query_dynamic_v2":
        return _query_dynamic_layout_v2(
            input_ids=ids,
            tokenizer=tokenizer,
            sample=sample,
            prompt_text=prompt_text,
            task_name=task_name or "",
            query_text=query_text,
            target_block_tokens=target_block_tokens,
            min_block_tokens=min_block_tokens,
            max_block_tokens=max_block_tokens,
            fixed_page_tokens=fixed_page_tokens,
            config=cfg,
        )
    if strategy == "query_dynamic_v3":
        return _query_dynamic_layout_v2(
            input_ids=ids,
            tokenizer=tokenizer,
            sample=sample,
            prompt_text=prompt_text,
            task_name=task_name or "",
            query_text=query_text,
            target_block_tokens=target_block_tokens,
            min_block_tokens=min_block_tokens,
            max_block_tokens=max_block_tokens,
            fixed_page_tokens=fixed_page_tokens,
            config={**cfg, "query_version": "v3"},
            strategy_name="query_dynamic_v3",
        )
    if strategy == "hybrid_attention_semantic":
        layout, _ = _semantic_layout_v2(
            input_ids=ids,
            tokenizer=tokenizer,
            sample=sample,
            prompt_text=prompt_text,
            task_name=task_name or "",
            target_block_tokens=int(cfg.get("semantic_target_block_tokens", target_block_tokens)),
            min_block_tokens=int(cfg.get("semantic_min_block_tokens", min_block_tokens)),
            max_block_tokens=int(cfg.get("semantic_max_block_tokens", max_block_tokens)),
            fixed_page_tokens=fixed_page_tokens,
            config=cfg,
            strategy_name="hybrid_attention_semantic_rescue_layout",
        )
        return layout

    layout = _fixed_layout(ids, tokenizer, fixed_page_tokens)
    layout.strategy_name = strategy
    layout.metadata.update({
        "fallback_used": True,
        "fallback_reason": "dynamic_layout_materialized_in_cache_path",
        "fallback_reason_counts": _empty_reason_counts(),
        "target_block_tokens": int(target_block_tokens),
        "min_block_tokens": int(min_block_tokens),
        "max_block_tokens": int(max_block_tokens),
    })
    layout.fallback_used = False
    layout.fallback_reason = ""
    layout.fallback_reason_counts = _empty_reason_counts()
    return layout
