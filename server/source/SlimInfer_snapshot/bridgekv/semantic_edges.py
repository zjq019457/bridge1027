import re
from collections import defaultdict
from typing import Dict, List, Set, Tuple

STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "in", "on", "for", "with", "to", "from", "by",
    "is", "are", "was", "were", "that", "this", "it", "as", "at", "be", "has", "have",
}

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
NUMBER_RE = re.compile(r"\d[\d,.]*")
IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{1,}")
ENTITY_RE = re.compile(r"[A-Z][a-z]{2,}")


def tokenize_block(text: str) -> List[str]:
    tokens = TOKEN_RE.findall(text)
    return [token for token in tokens if token]


def normalize_token(token: str) -> str:
    return token.lower()


def extract_block_features(text: str) -> Dict[str, Set[str]]:
    tokens = tokenize_block(text)
    lower_tokens = {normalize_token(t) for t in tokens if len(t) > 1}
    lower_tokens = {t for t in lower_tokens if t not in STOPWORDS}

    numbers = {m.group(0) for m in NUMBER_RE.finditer(text)}
    capitalized = {t for t in tokens if ENTITY_RE.fullmatch(t)}
    identifiers = {t for t in tokens if IDENTIFIER_RE.fullmatch(t) and ("_" in t or t.islower() is False)}
    rare_words = {t for t in lower_tokens if len(t) > 3}

    return {
        "tokens": lower_tokens,
        "numbers": numbers,
        "entities": capitalized,
        "identifiers": identifiers,
        "rare_words": rare_words,
    }


def build_semantic_edge_weights(
    block_features: List[Dict[str, Set[str]]],
    topk: int = 16,
    w_num: float = 1.0,
    w_ent: float = 0.7,
    w_kw: float = 0.5,
    w_id: float = 0.8,
) -> List[Dict[int, float]]:
    num_blocks = len(block_features)
    feature_inv: Dict[str, List[int]] = defaultdict(list)

    for idx, features in enumerate(block_features):
        for num in features["numbers"]:
            feature_inv[f"num:{num}"].append(idx)
        for ent in features["entities"]:
            feature_inv[f"ent:{ent}"].append(idx)
        for kw in features["rare_words"]:
            feature_inv[f"kw:{kw}"].append(idx)
        for ident in features["identifiers"]:
            feature_inv[f"id:{ident}"].append(idx)

    neighbor_weights: List[Dict[int, float]] = [defaultdict(float) for _ in range(num_blocks)]
    for feature, blocks in feature_inv.items():
        if len(blocks) < 2 or len(blocks) > 64:
            continue
        if feature.startswith("num:"):
            weight = w_num
        elif feature.startswith("ent:"):
            weight = w_ent
        elif feature.startswith("kw:"):
            weight = w_kw
        elif feature.startswith("id:"):
            weight = w_id
        else:
            weight = 0.3

        for i in blocks:
            for j in blocks:
                if i == j:
                    continue
                neighbor_weights[i][j] += weight

    trimmed: List[Dict[int, float]] = []
    for weights in neighbor_weights:
        items = sorted(weights.items(), key=lambda item: item[1], reverse=True)[:topk]
        trimmed.append({k: float(v) for k, v in items})
    return trimmed
