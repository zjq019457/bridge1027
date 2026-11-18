

def hyb_chunk(s, tok, budget=1024, neighbor=32, top_k=64):
    """Standard chunk packing with the hybrid ranker (z(lexical) + z(BGE-small), as in lex_rep seed_mode=2):
    identical to bge_chunk except for the ranking."""
    q = s["question"]
    cands, page_map = etp.build_candidate_spans(s["sample_id"], s["task"], q, s["context"])
    if not cands:
        return _emit(tok, q, [], budget)
    seed = _seed(s, cands, 2)
    ranked = sorted(cands, key=lambda c: (-seed[c.span_id], c.source_order_index))[:top_k]
    prefix = etp._prefix(q, False)
    pieces = []
    for c in ranked:
        piece, _ = etp._expanded_piece(c, page_map, neighbor, role="support", use_tags=False)
        if len(tok("\n\n".join([prefix, *pieces, piece]).strip()).input_ids) <= budget:
            pieces.append(piece)
    return _emit(tok, q, pieces, budget)
