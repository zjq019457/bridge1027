import re
from typing import List

QUESTION_TOK_RE = re.compile(r"[A-Za-z0-9_]+")
CODE_SYMBOL_RE = re.compile(r"[()\[\]{}_.:;=<>+-]|")
TABLE_MARKER_RE = re.compile(r"\||^\s*\d+[.)]\s*|^-{3,}|^\s*[*+-]\s+")


def normalize(values: List[float]) -> List[float]:
    if not values:
        return []
    mn = min(values)
    mx = max(values)
    denom = mx - mn if mx > mn else 1.0
    return [(v - mn) / denom for v in values]


def safe_tokens(text: str):
    return QUESTION_TOK_RE.findall(text)


def compute_structural_scores(block_texts: List[str], query_text: str) -> List[float]:
    query_tokens = {token.lower() for token in safe_tokens(query_text)}
    q_len = max(1, len(query_tokens))
    scores = []
    for idx, text in enumerate(block_texts):
        tokens = safe_tokens(text)
        lower_tokens = [token.lower() for token in tokens]
        token_set = set(lower_tokens)
        overlap = len(token_set & query_tokens) / q_len
        numbers = sum(1 for token in tokens if any(ch.isdigit() for ch in token)) / max(1, len(tokens))
        entities = sum(1 for token in tokens if token[0].isupper() and len(token) > 1) / max(1, len(tokens))
        code_symbols = len(CODE_SYMBOL_RE.findall(text)) / max(1, len(text))
        table_marker = 1.0 if TABLE_MARKER_RE.search(text) else 0.0
        scores.append(overlap * 1.0 + numbers * 0.5 + entities * 0.5 + code_symbols * 0.5 + table_marker * 0.5)
    return normalize(scores)
