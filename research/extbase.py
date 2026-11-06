"""External baselines on LongBench (exact budget, same wrapper/tokenizers):
  window    LlamaIndex SentenceWindowNodeParser (window 3) + BGE-small sentence retrieval,
            windows admitted in rank order while they fit (MetadataReplacement behaviour)
  automerge LlamaIndex HierarchicalNodeParser (2048/512/128) + BGE-small leaf retrieval +
            AutoMergingRetriever merge rule (parent replaces children if > 50% retrieved)
  rerank    bge-reranker-v2-m3 scores for the top-100 BGE spans -> cache/rr_test.jsonl
            (then packed by bge_chunk / SentPack via BP_ENC=rr)
  provence  Provence (naver) prunes each of the top BGE chunks; pruned chunks admitted in rank order
usage: python extbase.py MODE OUTDIR_QWEN:TOK OUTDIR_LLAMA:TOK  (budget 1024)"""
import json, os, sys
from pathlib import Path
import torch
import localpack as L
import packers as P
from bridgekv import evidence_text_packing as etp
HERE = Path(__file__).resolve().parent
mode = sys.argv[1]; outs = [a.split(":", 1) for a in sys.argv[2:]]; B = 1024
S = L.load_samples()[:int(os.environ.get("LIMIT", 10**9))]
toks = [(d, L.HFLike(t)) for d, t in outs]
INSTR = "Represent this sentence for searching relevant passages: "


def bge():
    from transformers import AutoModel, AutoTokenizer
    p = os.environ["BGE_PATH"]; tk = AutoTokenizer.from_pretrained(p); m = AutoModel.from_pretrained(p).cuda().half().eval()
    @torch.no_grad()
    def enc(texts):
        r = []
        for i in range(0, len(texts), 256):
            b = tk(texts[i:i + 256], padding=True, truncation=True, max_length=512, return_tensors="pt").to("cuda")
            r.append(torch.nn.functional.normalize(m(**b).last_hidden_state[:, 0].float(), dim=-1))
        return torch.cat(r)
    return enc


def pack_rank_order(s, tok, pieces):
    prefix = etp._prefix(s["question"], False); kept = []
    for p in pieces:
        if len(tok("\n\n".join([prefix, *kept, p]).strip()).input_ids) <= B:
            kept.append(p)
    return P._emit(tok, s["question"], kept, B)


def write(name, fn):
    for d, tok in toks:
        with open(f"{d}/out/{name}_budget={B}.jsonl", "w", encoding="utf-8") as f:
            for s in S:
                f.write(json.dumps({"task": s["task"], "sample_id": s["sample_id"], "question": s["question"],
                                    "answers": s["answers"], "packed_text": fn(s, tok)}) + "\n")


if mode in ("window", "automerge"):
    from llama_index.core import Document
    from llama_index.core.node_parser import SentenceWindowNodeParser, HierarchicalNodeParser, get_leaf_nodes
    enc = bge(); ranked = {}
    for n, s in enumerate(S):
        doc = Document(text=s["context"])
        q = enc([INSTR + s["question"]])[0]
        if mode == "window":
            nodes = SentenceWindowNodeParser.from_defaults(window_size=3).get_nodes_from_documents([doc])
            sims = (enc([x.text for x in nodes]) @ q).tolist() if nodes else []
            order = sorted(range(len(nodes)), key=lambda i: -sims[i])[:100]
            ranked[s["sample_id"] + s["task"]] = [nodes[i].metadata["window"] for i in order]
        else:
            allnodes = HierarchicalNodeParser.from_defaults(chunk_sizes=[2048, 512, 128]).get_nodes_from_documents([doc])
            byid = {x.node_id: x for x in allnodes}; leaves = get_leaf_nodes(allnodes)
            sims = (enc([x.text for x in leaves]) @ q).tolist() if leaves else []
            order = sorted(range(len(leaves)), key=lambda i: -sims[i])[:12]  # similarity_top_k = 12
            cur = [(leaves[i], sims[i]) for i in order]
            changed = True
            while changed:  # AutoMergingRetriever._try_merging, simple_ratio_thresh = 0.5
                changed = False; groups = {}
                for nd, sc in cur:
                    if nd.parent_node is not None:
                        groups.setdefault(nd.parent_node.node_id, []).append((nd, sc))
                for pid, ch in groups.items():
                    par = byid[pid]
                    if len(ch) / max(1, len(par.child_nodes or [])) > 0.5:
                        ids = {c.node_id for c, _ in ch}
                        sc = sum(x for _, x in ch) / len(ch)
                        cur = [(nd, x) for nd, x in cur if nd.node_id not in ids] + [(par, sc)]; changed = True
                        break
            cur.sort(key=lambda t: -t[1])
            ranked[s["sample_id"] + s["task"]] = [nd.text for nd, _ in cur] + [leaves[i].text for i in sorted(range(len(leaves)), key=lambda i: -sims[i])[12:100]]
        if n % 100 == 0: print(mode, n, flush=True)
    write("li_" + mode, lambda s, tok: pack_rank_order(s, tok, ranked[s["sample_id"] + s["task"]]))
elif mode == "rerank":
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    p = os.environ["RR_PATH"]; tk = AutoTokenizer.from_pretrained(p)
    m = AutoModelForSequenceClassification.from_pretrained(p, torch_dtype=torch.float16).cuda().eval()
    with open(HERE / "cache" / "rr_test.jsonl", "w", encoding="utf-8") as f:
        for n, s in enumerate(S):
            cands, _ = etp.build_candidate_spans(s["sample_id"], s["task"], s["question"], s["context"])
            sims = P._bge_sims(s) or {}
            top = sorted(cands, key=lambda c: -sims.get(c.span_id, 0.0))[:100]
            sc = []
            with torch.no_grad():
                for i in range(0, len(top), 50):
                    b = tk([[s["question"], c.text] for c in top[i:i + 50]], padding=True, truncation=True, max_length=512, return_tensors="pt").to("cuda")
                    sc += m(**b).logits.view(-1).float().tolist()
            low = min(sc) - 1 if sc else 0.0
            d = {c.span_id: low for c in cands}; d.update({c.span_id: round(x, 4) for c, x in zip(top, sc)})
            f.write(json.dumps({"key": f"{s['task']}/{s['sample_id']}", "sim": d}) + "\n")
            if n % 100 == 0: print("rerank", n, flush=True)
    print("RR_DONE")
elif mode == "provence":
    from transformers import AutoModel
    pv = AutoModel.from_pretrained(os.environ["PV_PATH"], trust_remote_code=True).cuda().eval()
    pruned = {}
    for n, s in enumerate(S):
        cands, page_map = etp.build_candidate_spans(s["sample_id"], s["task"], s["question"], s["context"])
        sims = P._bge_sims(s) or {}
        top = sorted(cands, key=lambda c: -sims.get(c.span_id, 0.0))[:60]
        pieces = [etp._expanded_piece(c, page_map, 32, role="support", use_tags=False)[0] for c in top]
        out = pv.process([s["question"]], [pieces], always_select_title=False, enable_warnings=False) if pieces else {"pruned_context": [[]]}
        pruned[s["sample_id"] + s["task"]] = [p for p in out["pruned_context"][0] if p.strip()]
        if n % 100 == 0: print("provence", n, flush=True)
    write("provence", lambda s, tok: pack_rank_order(s, tok, pruned[s["sample_id"] + s["task"]]))
print("EXT_DONE", mode)
