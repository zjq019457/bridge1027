"""Packing latency (single CPU process, per instance, 1024 budget) on a fixed sample of the test
set. BGE encoding time is measured separately in embed_cache (GPU). usage: python timing.py N"""
import random, sys, time, json
import localpack as L, packers as P
tok = L.HFLike(L.TOKDIR)
S = L.load_samples(); random.Random(0).shuffle(S); S = S[:int(sys.argv[1])]
cfg = {"bp_main": (P.bp_main, {}), "bge_chunk": (P.bge_chunk, {}),
       "ours_hybrid": (P.lex_rep, dict(seed_mode=2, sent=1)), "ours_bge": (P.lex_rep, dict(seed_mode=1, sent=1))}
for s in S[:3]:
    for f, kw in cfg.values(): f(s, tok, **kw)  # warm caches
res = {}
for name, (f, kw) in cfg.items():
    ts, nt = [], []
    for s in S:
        t = time.perf_counter(); out = f(s, tok, **kw); ts.append(time.perf_counter() - t); nt.append(len(tok(out).input_ids))
    ts.sort()
    res[name] = dict(mean_ms=round(1000 * sum(ts) / len(ts), 1), p50_ms=round(1000 * ts[len(ts) // 2], 1),
                     p95_ms=round(1000 * ts[int(.95 * len(ts))], 1), max_tok=max(nt), mean_tok=round(sum(nt) / len(nt), 1))
    print(name, res[name], flush=True)
json.dump(res, open("timing.json", "w"), indent=1)
