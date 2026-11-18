"""packers.py: (1) recomp_ext reads the cached RECOMP order (computed once for both tokenizers);
(2) sel_ctx2 finds its cache relative to packers.py (evalcov runs from a sub-directory);
(3) Selective Context device from SC_DEVICE (default cpu)."""
t = open("packers.py", encoding="utf-8").read()
reps = [
    ('_AUX = _os.environ.get("BP_AUX", "/root/autodl-tmp/aux")\n',
     '_AUX = _os.environ.get("BP_AUX", "/root/autodl-tmp/aux")\n_HERE = _os.path.dirname(_os.path.abspath(__file__))\n'),
    ("""    E = _recomp_embed([q] + sents)
    sc = (E[1:] @ E[0]).tolist()
    order = sorted(range(len(sents)), key=lambda i: -sc[i])
""", """    order = _recomp_order(s, q, sents)
"""),
    ("def recomp_ext(s, tok, budget=1024):\n", '''_RCO = {}


def _recomp_order(s, q, sents):
    """Sentence order by RECOMP score, from cache/recomp_order.jsonl (rc_scores.py: same checkpoint and
    pooling) when present, so the scores are computed once for every tokenizer."""
    if not _RCO:
        import json as _json
        p, _RCO["d"] = _os.path.join(_HERE, "cache", "recomp_order.jsonl"), {}
        if _os.path.exists(p):
            for l in open(p, encoding="utf-8"):
                r = _json.loads(l); _RCO["d"][r["key"]] = r
    r = _RCO["d"].get(f"{s['task']}/{s['sample_id']}")
    if r is not None and r["n"] == len(sents):
        return r["order"]
    E = _recomp_embed([q] + sents)
    sc = (E[1:] @ E[0]).tolist()
    return sorted(range(len(sents)), key=lambda i: -sc[i])


def recomp_ext(s, tok, budget=1024):
'''),
    ('open(_os.environ.get("SC_CACHE", "cache/sc_units.jsonl"), encoding="utf-8")',
     'open(_os.environ.get("SC_CACHE", _os.path.join(_HERE, "cache", "sc_units.jsonl")), encoding="utf-8")'),
    ('sc.model_type, sc.lang, sc.device = "gpt2", "en", torch.device("cpu")',
     'sc.model_type, sc.lang, sc.device = "gpt2", "en", torch.device(_os.environ.get("SC_DEVICE", "cpu"))\n'
     '        if sc.device.type == "cuda":\n'
     '            torch.cuda.set_per_process_memory_fraction(0.03)  # shared card'),
]
for a, b in reps:
    assert t.count(a) == 1, a[:60]
    t = t.replace(a, b)
open("packers.py", "w", encoding="utf-8", newline="").write(t)
print("patched")
