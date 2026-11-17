"""Provence + DedupPack: Provence prunes the sentences of each top-60 BGE span (sentence-repaired as in
DedupPack); DedupPack then merges the kept sentence ranges across spans (no sentence paid twice),
emits them in document order, and admits spans in BGE rank order with backfill under the exact budget.
stage 1 (GPU):  python provdp.py prune                -> cache/provdp_keep.jsonl
stage 2 (CPU):  BP_TOK=... python provdp.py pack OUTDIR BUDGET
"""
import json, os, sys
from pathlib import Path
import localpack as L
import packers as P
from bridgekv import evidence_text_packing as etp

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache" / "provdp_keep.jsonl"
RADIUS, TOPK = 32, 60
S = L.load_samples()[:int(os.environ.get("LIMIT", 10**9))]


def spans(s):
    cands, page_map = etp.build_candidate_spans(s["sample_id"], s["task"], s["question"], s["context"])
    sims = P._bge_sims(s) or {}
    top = sorted(cands, key=lambda c: -sims.get(c.span_id, 0.0))[:TOPK]
    return top, page_map


def sentences(page_map, p, a, b):
    """Split token range [a, b) of page p into sentence ranges."""
    page = page_map[p]; ps = int(page["start_token"]); toks = page["tokens"]
    out, st = [], a
    for i in range(a, b):
        if toks[i - ps] in P._SENT_END:
            out.append((p, st, i + 1)); st = i + 1
    if st < b:
        out.append((p, st, b))
    return out


def norm(x):
    return " ".join(x.split())


def prune():
    import torch
    from transformers import AutoModel
    pv = AutoModel.from_pretrained(os.environ["PV_PATH"], trust_remote_code=True).cuda().half().eval()
    done = set()
    if CACHE.exists():
        done = {json.loads(l)["key"] for l in CACHE.open(encoding="utf-8")}
    with CACHE.open("a", encoding="utf-8") as f:
        for n, s in enumerate(S):
            key = f"{s['task']}/{s['sample_id']}"
            if key in done:
                continue
            top, page_map = spans(s)
            units = []  # per span: list of sentence ranges
            for c in top:
                page = page_map[c.page_id]; ps = int(page["start_token"])
                a, b = P._sentence_range(page["tokens"], ps, c.start_token, c.end_token, RADIUS)
                units.append(sentences(page_map, c.page_id, a, b))
            texts = [" ".join(P._range_text(page_map, *r) for r in u) for u in units]
            keep = []
            if texts:
                with torch.no_grad():
                    out = pv.process([s["question"]], [texts], always_select_title=False, enable_warnings=False, batch_size=8)
                for u, pr in zip(units, out["pruned_context"][0]):
                    pr = norm(pr)
                    keep.append([list(r) for r in u if norm(P._range_text(page_map, *r)) and norm(P._range_text(page_map, *r)) in pr])
            f.write(json.dumps({"key": key, "keep": keep}) + "\n"); f.flush()
            if n % 50 == 0:
                print("prune", n, flush=True)
    print("PRUNE_DONE")


def render(page_map, ranges):
    rs = sorted(set(map(tuple, ranges)))
    merged = []
    for p, a, b in rs:
        if merged and merged[-1][0] == p and a <= merged[-1][2]:
            merged[-1] = (p, merged[-1][1], max(merged[-1][2], b))
        else:
            merged.append((p, a, b))
    return [P._range_text(page_map, *r) for r in merged]


def pack(outdir, B):
    tok = L.HFLike(os.environ["BP_TOK"])
    K = {}
    for l in CACHE.open(encoding="utf-8"):
        r = json.loads(l); K[r["key"]] = r["keep"]
    prefix_of = lambda s: etp._prefix(s["question"], False)
    with open(f"{outdir}/out/provence_dedup_budget={B}.jsonl", "w", encoding="utf-8") as f:
        for s in S:
            _, page_map = etp.build_candidate_spans(s["sample_id"], s["task"], s["question"], s["context"])
            prefix = prefix_of(s); chosen = []
            for k in K[f"{s['task']}/{s['sample_id']}"]:
                if not k:
                    continue
                trial = chosen + k
                text = "\n\n".join([prefix] + render(page_map, trial))
                if len(tok(text).input_ids) <= B:
                    chosen = trial
            pieces = render(page_map, chosen)
            f.write(json.dumps({"task": s["task"], "sample_id": s["sample_id"], "question": s["question"],
                                "answers": s["answers"], "packed_text": P._emit(tok, s["question"], pieces, B)}) + "\n")
    print("PACK_DONE", outdir, B)


if __name__ == "__main__":
    if sys.argv[1] == "prune":
        prune()
    else:
        pack(sys.argv[2], int(sys.argv[3]))
