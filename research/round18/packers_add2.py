

_SCU = {}


def sel_ctx2(s, tok, budget=1024):
    """Selective Context from cached phrase units (sc_units.py): identical masking rule to the official
    self_info_mask at phrase level (drop units below the reduce-ratio percentile, join with '')."""
    import json, numpy as np
    if not _SCU:
        for l in open(_os.environ.get("SC_CACHE", "cache/sc_units.jsonl"), encoding="utf-8"):
            r = json.loads(l); _SCU[r["key"]] = (r["text"], r["info"])
    q = s["question"]
    room = max(1, _room(tok, q, budget))
    n = len(tok(s["context"]).input_ids)
    if n <= room:
        return _emit(tok, q, [s["context"]], budget)
    texts, info = _SCU[f"{s['task']}/{s['sample_id']}"]
    thr = np.nanpercentile(info, min(0.99, 1 - room / n) * 100)
    kept = "".join(t for t, i in zip(texts, info) if not (i < thr))
    return _emit(tok, q, [kept], budget)
