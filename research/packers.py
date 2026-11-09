"""Packers evaluated in the research loop. `bp_main` reproduces the paper's main configuration
(run_g1 `same_spans_source_order`): frozen Lexical-BridgePack selection with tokenizer-exact
admission, repaired pieces emitted in document order.
"""
from bridgekv import bridgepack_cpu_extensions as ext
from bridgekv import evidence_text_packing as etp


def _emit(tok, question, pieces, budget):
    text = "\n\n".join([etp._prefix(question, False), *pieces]).strip()
    ids = tok(text).input_ids
    if len(ids) > budget:
        text = tok.decode(ids[:budget])
    return text


def bp_main(s, tok, budget=1024):
    fr = ext.freeze_lexical_bridgepack_selection(s["sample_id"], s["task"], s["question"], s["context"], budget, tokenizer=tok)
    by_er = {er["span_id"]: er for er in fr["audit"]["expanded_records"]}
    by_id = {c.span_id: c for c in fr["candidates"]}
    order = sorted((by_id[i] for i in fr["span_ids"] if i in by_id), key=lambda x: x.source_order_index)
    return _emit(tok, s["question"], [str(by_er[c.span_id]["text"]) for c in order if c.span_id in by_er], budget)


# ---------------------------------------------------------------------------------------------
# Research variants: in-context second-hop retrieval through bridge entities.
import math
import re
from collections import Counter

_STOP_CAP = set("""The A An In On At Of For From By With And But Or If When While After Before During Since
Until As This That These Those It Its He She They His Her Their We Our You Your I Who What Which Where Why How
Is Was Are Were Be Been Being Has Have Had Do Does Did Not No Yes Passage Question Answer Also However Although
Both Each Many Most Some Such Other Another There Here Then Than Then Thus So Because According Despite""".split())
_ENT = re.compile(r"(?:[A-Z][\w'’\-\.]*(?:\s+(?:of|de|da|von|van|the|and|du|la|le)\s+)?)+[A-Z][\w'’\-]*|[A-Z][\w'’\-]{2,}|\b(?:1[0-9]{3}|20[0-9]{2})\b")


def entities(text):
    out = []
    for m in _ENT.finditer(text):
        e = m.group(0).strip(" .")
        words = e.split()
        while words and words[0] in _STOP_CAP:
            words = words[1:]
        if words:
            out.append(" ".join(words))
    return out


def _ekey(e):
    return " ".join(etp._content_tokens(e))


_SENT_END = {".", "?", "!"}


def _sentence_range(toks, ps, a, b, radius):
    """Smallest extension of [a, b) to sentence boundaries, at most `radius` tokens per side."""
    lo = max(ps, a - radius)
    left = a
    while left > lo and toks[left - 1 - ps] not in _SENT_END:
        left -= 1
    hi = min(ps + len(toks), b + radius)
    right = b
    while right < hi and toks[right - 1 - ps] not in _SENT_END:
        right += 1
    return left, right


def _merge_pieces(selected, page_map, radius, sent=0):
    """Repair each span by +-radius within its page (or, with sent=1, only up to the nearest
    sentence boundaries within that radius) and merge overlapping repaired ranges, so that
    overlapping spans do not spend budget twice. Returns [(page_id, start, end)]."""
    rng = []
    for c in selected:
        page = page_map[c.page_id]
        ps, toks = int(page["start_token"]), page["tokens"]
        if sent:
            a, b = _sentence_range(toks, ps, c.start_token, c.end_token, radius)
            rng.append((c.page_id, a, b))
            continue
        rng.append((c.page_id, max(ps, c.start_token - radius), min(ps + len(toks), c.end_token + radius)))
    rng.sort()
    merged = []
    for p, a, b in rng:
        if merged and merged[-1][0] == p and a <= merged[-1][2]:
            merged[-1] = (p, merged[-1][1], max(merged[-1][2], b))
        else:
            merged.append((p, a, b))
    return merged


def _range_text(page_map, p, a, b):
    page = page_map[p]
    ps = int(page["start_token"])
    return etp._detokenize(page["tokens"][a - ps:b - ps])


def _admit(tok, s, priority, page_map, budget, radius, sent=0):
    """Add spans in priority order while the merged, repaired, document-ordered context fits the
    exact budget (spans that do not fit are skipped, later ones may still fit)."""
    prefix = etp._prefix(s["question"], False)
    chosen = []
    for c in priority:
        trial = chosen + [c]
        text = "\n\n".join([prefix] + [_range_text(page_map, *r) for r in _merge_pieces(trial, page_map, radius, sent)])
        if len(tok(text).input_ids) <= budget:
            chosen = trial
        elif len(tok(text).input_ids) > budget + 400:
            continue
    text = "\n\n".join([prefix] + [_range_text(page_map, *r) for r in _merge_pieces(chosen, page_map, radius, sent)])
    return _emit(tok, s["question"], [text[len(prefix):].strip()], budget) if chosen else _emit(tok, s["question"], [], budget)


def _z(d, keys):
    v = [d.get(k, 0.0) for k in keys]
    n = etp.normalize_scores(v)
    return dict(zip(keys, n))


_BGE = {}


def _bge_sims(s):
    """Cached bge-small question-span cosine similarities (see embed_cache.py)."""
    if not _BGE:
        import json
        from pathlib import Path
        import os
        for f in Path(__file__).with_name("cache").glob(os.environ.get("BP_ENC", "bge") + "_*.jsonl"):
            for line in f.open(encoding="utf-8"):
                r = json.loads(line)
                _BGE[r["key"]] = r["sim"]
    return _BGE.get(f"{s['task']}/{s['sample_id']}")


def _seed(s, cands, mode):
    ids = [c.span_id for c in cands]
    if mode == 0:  # lexical
        return etp.lexical_scores(s["question"], cands)
    sims = _bge_sims(s)
    dense = _z({i: sims.get(i, 0.0) for i in ids}, ids)
    if mode == 1:  # dense
        return dense
    lex = _z(etp.lexical_scores(s["question"], cands), ids)  # mode 2: hybrid
    return {i: lex[i] + dense[i] for i in ids}


def bridge_hop(s, tok, budget=1024, k1=6, radius=32, beta=0.3, hop2_share=0.5, max_ents=12, seed_mode=0, sent=0):
    q = s["question"]
    cands, page_map = etp.build_candidate_spans(s["sample_id"], s["task"], q, s["context"])
    if not cands:
        return _emit(tok, q, [], budget)
    ids = [c.span_id for c in cands]
    seed = _seed(s, cands, seed_mode)
    hop1 = sorted(cands, key=lambda c: (-seed[c.span_id], c.source_order_index))
    # bridge entities: named entities / years in the top hop-1 spans that the question does not mention
    qkeys = {_ekey(e) for e in entities(q)} | {""}
    qterms = set(etp._content_tokens(q))
    cnt = Counter()
    for rank, c in enumerate(hop1[:k1]):
        for e in entities(c.text):
            k = _ekey(e)
            if k and k not in qkeys and not set(k.split()) <= qterms:
                cnt[k] += 1.0 / (1 + rank)
    bridge = [k for k, _ in cnt.most_common(max_ents)]
    if bridge:
        b2 = etp.bm25_scores(" ".join(bridge), cands)
        h2 = _z(b2, ids)
        sz = _z(seed, ids)
        hop2_score = {i: h2[i] + beta * sz[i] for i in ids}
        hop2 = sorted(cands, key=lambda c: (-hop2_score[c.span_id], c.source_order_index))
    else:
        hop2 = hop1
    # interleave hop-1 and hop-2 rankings (hop2_share of the picks come from hop-2)
    priority, seen, i1, i2, acc = [], set(), 0, 0, 0.0
    while i1 < len(hop1) or i2 < len(hop2):
        acc += hop2_share
        take2 = acc >= 1.0 and i2 < len(hop2)
        if take2:
            acc -= 1.0
            c = hop2[i2]; i2 += 1
        elif i1 < len(hop1):
            c = hop1[i1]; i1 += 1
        else:
            c = hop2[i2]; i2 += 1
        if c.span_id not in seen:
            seen.add(c.span_id)
            priority.append(c)
        if len(priority) >= 60:
            break
    return _admit(tok, s, priority, page_map, budget, radius, sent)


def lex_rep(s, tok, budget=1024, radius=32, seed_mode=0, sent=0):
    """Control: hop-1 ranking only (lexical / dense / hybrid seeds), same repair/merge/admission."""
    return bridge_hop(s, tok, budget, radius=radius, hop2_share=0.0, seed_mode=seed_mode, sent=sent)


def bge_chunk(s, tok, budget=1024, neighbor=32, top_k=64):
    """BGE chunk packing as in the repo's _dense_pack (top-64 spans by cached BGE similarity,
    +-32 neighbours, each admitted if it fits, emitted in similarity order)."""
    q = s["question"]
    cands, page_map = etp.build_candidate_spans(s["sample_id"], s["task"], q, s["context"])
    sims = _bge_sims(s) or {}
    ranked = sorted(cands, key=lambda c: -sims.get(c.span_id, 0.0))[:top_k]
    prefix = etp._prefix(q, False)
    pieces = []
    for c in ranked:
        piece, _ = etp._expanded_piece(c, page_map, neighbor, role="support", use_tags=False)
        if len(tok("\n\n".join([prefix, *pieces, piece]).strip()).input_ids) <= budget:
            pieces.append(piece)
    return _emit(tok, q, pieces, budget)


# ---------------------------------------------------------------------------------------------
# Component ablations of the final packer (hybrid seeds + sentence repair + merged backfill
# admission + document-order emission). Each flag switches one component off.
def _abl_ranges(sel, page_map, radius, sent, merge, docorder):
    rng = []
    for k, c in enumerate(sel):
        page = page_map[c.page_id]
        ps, toks = int(page["start_token"]), page["tokens"]
        if sent:
            a, b = _sentence_range(toks, ps, c.start_token, c.end_token, radius)
        else:
            a, b = max(ps, c.start_token - radius), min(ps + len(toks), c.end_token + radius)
        rng.append((c.page_id, a, b, k))
    if merge:
        rng.sort()
        m = []
        for p, a, b, k in rng:
            if m and m[-1][0] == p and a <= m[-1][2]:
                m[-1] = (p, m[-1][1], max(m[-1][2], b), min(m[-1][3], k))
            else:
                m.append((p, a, b, k))
        rng = m
    rng.sort(key=(lambda r: (r[0], r[1])) if docorder else (lambda r: r[3]))
    return [r[:3] for r in rng]


def final_abl(s, tok, budget=1024, seed_mode=2, sent=1, merge=1, docorder=1, backfill=1, radius=32):
    q = s["question"]
    cands, page_map = etp.build_candidate_spans(s["sample_id"], s["task"], q, s["context"])
    if not cands:
        return _emit(tok, q, [], budget)
    seed = _seed(s, cands, seed_mode)
    priority = sorted(cands, key=lambda c: (-seed[c.span_id], c.source_order_index))[:60]
    prefix = etp._prefix(q, False)

    def render(sel):
        return "\n\n".join([prefix] + [_range_text(page_map, *r) for r in _abl_ranges(sel, page_map, radius, sent, merge, docorder)])
    chosen = []
    for c in priority:
        n = len(tok(render(chosen + [c])).input_ids)
        if n <= budget:
            chosen.append(c)
        elif not backfill:
            break
    if not chosen:
        return _emit(tok, q, [], budget)
    return _emit(tok, q, [render(chosen)[len(prefix):].strip()], budget)


# ---------------------------------------------------------------------------------------------
# Marginal-cost packing: budgeted maximum coverage over merged token intervals.
# gain(s) = shifted seed score; cost(s | S) = new (not yet covered) regex tokens that s's repaired
# range adds to the selection. Greedy by gain / marginal cost (cost-benefit greedy), each pick
# verified against the exact rendered budget; compared with the best single span (Khuller et al.).
def _ivl_new(rngs, p, a, b):
    """Tokens of [a, b) on page p not covered by the ranges already selected."""
    cov = 0
    for q, x, y in rngs:
        if q == p:
            lo, hi = max(a, x), min(b, y)
            if hi > lo:
                cov += hi - lo
    return max(0, (b - a) - cov)


def density_pack(s, tok, budget=1024, seed_mode=2, sent=1, radius=32, alpha=1.0, pool=60):
    q = s["question"]
    cands, page_map = etp.build_candidate_spans(s["sample_id"], s["task"], q, s["context"])
    if not cands:
        return _emit(tok, q, [], budget)
    seed = _seed(s, cands, seed_mode)
    pri = sorted(cands, key=lambda c: (-seed[c.span_id], c.source_order_index))[:pool]
    lo = min(seed[c.span_id] for c in pri)
    gain = {c.span_id: seed[c.span_id] - lo + 0.1 for c in pri}
    rep = {}
    for c in pri:
        page = page_map[c.page_id]; ps, toks = int(page["start_token"]), page["tokens"]
        rep[c.span_id] = (c.page_id,) + (_sentence_range(toks, ps, c.start_token, c.end_token, radius) if sent
                                         else (max(ps, c.start_token - radius), min(ps + len(toks), c.end_token + radius)))
    prefix = etp._prefix(q, False)

    def render(sel):
        return "\n\n".join([prefix] + [_range_text(page_map, *r) for r in _merge_pieces(sel, page_map, radius, sent)])
    chosen, rngs, left = [], [], list(pri)
    while left:
        best = max(left, key=lambda c: gain[c.span_id] / (max(1, _ivl_new(rngs, *rep[c.span_id])) ** alpha))
        left.remove(best)
        if len(tok(render(chosen + [best])).input_ids) <= budget:
            chosen.append(best); rngs.append(rep[best.span_id])
    if not chosen:
        return _emit(tok, q, [], budget)
    return _emit(tok, q, [render(chosen)[len(prefix):].strip()], budget)


# ---------------------------------------------------------------------------------------------
# Controls for the merge ablation (reviewer question: is the gain from removing duplication or
# from longer contiguous pieces?).
def dedup_skip(s, tok, budget=1024, seed_mode=1, radius=32):
    """Deduplicate WITHOUT merging: chunk packing in rank order, but a candidate whose padded range
    overlaps an already selected range on the same page is skipped. Pieces stay separate and short."""
    q = s["question"]
    cands, page_map = etp.build_candidate_spans(s["sample_id"], s["task"], q, s["context"])
    if not cands:
        return _emit(tok, q, [], budget)
    seed = _seed(s, cands, seed_mode)
    priority = sorted(cands, key=lambda c: (-seed[c.span_id], c.source_order_index))[:60]
    prefix = etp._prefix(q, False)

    def rng(c):
        page = page_map[c.page_id]; ps, toks = int(page["start_token"]), page["tokens"]
        return c.page_id, max(ps, c.start_token - radius), min(ps + len(toks), c.end_token + radius)

    def render(sel):
        return "\n\n".join([prefix] + [_range_text(page_map, *rng(c)) for c in sel])
    chosen, taken = [], []
    for c in priority:
        p, a, b = rng(c)
        if any(p == p2 and a < b2 and a2 < b for p2, a2, b2 in taken):
            continue
        if len(tok(render(chosen + [c])).input_ids) <= budget:
            chosen.append(c); taken.append((p, a, b))
    if not chosen:
        return _emit(tok, q, [], budget)
    return _emit(tok, q, [render(chosen)[len(prefix):].strip()], budget)


def mmr_pack(s, tok, budget=1024, seed_mode=1, radius=32, lam=0.7, pool=60):
    """MMR re-ranking (Carbonell & Goldstein, 1998) followed by ordinary chunk packing.
    relevance = min-max normalised ranker score; redundancy = max TF-IDF cosine to selected pieces."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    q = s["question"]
    cands, page_map = etp.build_candidate_spans(s["sample_id"], s["task"], q, s["context"])
    if not cands:
        return _emit(tok, q, [], budget)
    seed = _seed(s, cands, seed_mode)
    pri = sorted(cands, key=lambda c: (-seed[c.span_id], c.source_order_index))[:pool]

    def rng(c):
        page = page_map[c.page_id]; ps, toks = int(page["start_token"]), page["tokens"]
        return c.page_id, max(ps, c.start_token - radius), min(ps + len(toks), c.end_token + radius)
    texts = [_range_text(page_map, *rng(c)) for c in pri]
    X = TfidfVectorizer().fit_transform(texts); Sm = (X @ X.T).toarray()
    v = [seed[c.span_id] for c in pri]; lo, hi = min(v), max(v)
    rel = [(x - lo) / (hi - lo + 1e-9) for x in v]
    prefix = etp._prefix(q, False)
    chosen, left = [], list(range(len(pri)))
    while left:
        best = max(left, key=lambda i: lam * rel[i] - (1 - lam) * max([Sm[i, j] for j in chosen], default=0.0))
        left.remove(best)
        cand = chosen + [best]
        if len(tok("\n\n".join([prefix] + [texts[i] for i in cand])).input_ids) <= budget:
            chosen = cand
    if not chosen:
        return _emit(tok, q, [], budget)
    return _emit(tok, q, ["\n\n".join(texts[i] for i in chosen)], budget)
