"""MultiHop-RAG end-to-end traces with LlamaIndex sentence-window retrieval (window=3).
Retrieval: BGE-small over single sentences (GPU). Packing: standard (windows in rank order, as
LlamaIndex's MetadataReplacementPostProcessor does) vs DedupPack-style (overlapping windows of the
same article merged, so each token is paid once). Budget measured with the downstream tokenizer.
Writes gen_stream traces (task=hotpotqa_e -> LongBench QA template + F1) to DIR/out.
usage: BP_TOK=<tokdir> python mhrag_traces.py DIR"""
import json, os, pickle, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).parent))
import localpack as L

HERE = Path(__file__).parent
D = HERE / "data_mhrag"
OUT = Path(sys.argv[1]) / "out"; OUT.mkdir(parents=True, exist_ok=True)
Q = [q for q in json.load(open(D / "MultiHopRAG.json", encoding="utf-8")) if q["question_type"] != "null_query"]
C = json.load(open(D / "corpus.json", encoding="utf-8"))
docs = [c["title"] + "\n" + c["body"] for c in C]
BGE = os.environ.get("BP_BGE", "/root/autodl-tmp/models/models/BAAI--bge-small-en-v1.5")
INSTR = "Represent this sentence for searching relevant passages: "


def windows():
    f = HERE / "cache" / "mhrag_win.pkl"
    if f.exists():
        return pickle.load(open(f, "rb"))
    from llama_index.core import Document
    from llama_index.core.node_parser import SentenceWindowNodeParser
    sp = SentenceWindowNodeParser.from_defaults(window_size=3)
    out = []
    for i, d in enumerate(docs):
        for n in sp.get_nodes_from_documents([Document(text=d)]):
            win = n.metadata["window"]; s = d.find(win[:200])
            if s < 0: s = d.find(n.text[:100])
            out.append((i, max(0, s), max(0, s) + len(win), n.text, win))
    pickle.dump(out, open(f, "wb"))
    return out


def ranking(ch):
    f = HERE / "cache" / "mhrag_top.npy"
    if f.exists():
        return np.load(f)
    from transformers import AutoModel, AutoTokenizer
    tk = AutoTokenizer.from_pretrained(BGE); m = AutoModel.from_pretrained(BGE).half().cuda().eval()

    @torch.no_grad()
    def enc(texts, bs=256):
        out = []
        for i in range(0, len(texts), bs):
            b = tk(texts[i:i + bs], padding=True, truncation=True, max_length=512, return_tensors="pt").to("cuda")
            out.append(torch.nn.functional.normalize(m(**b).last_hidden_state[:, 0].float(), dim=-1))
        return torch.cat(out)
    E = enc([c[3] for c in ch]); qe = enc([INSTR + q["query"] for q in Q])
    top = torch.topk(qe @ E.T, 200, dim=1).indices.cpu().numpy()
    np.save(f, top); del m; torch.cuda.empty_cache()
    return top


tok = ch = None


def _init():
    global tok, ch
    tok = L.HFLike(L.TOKDIR); ch = windows()


def render(sel):
    by, order = {}, []
    for j in sel:
        d = ch[j][0]; by.setdefault(d, []).append([ch[j][1], ch[j][2]])
        if d not in order: order.append(d)
    out = []
    for d in order:
        iv = sorted(by[d]); m = [iv[0]]
        for a, b in iv[1:]:
            if a <= m[-1][1]: m[-1][1] = max(m[-1][1], b)
            else: m.append([a, b])
        out += [docs[d][a:b] for a, b in m]
    return "\n\n".join(out)


def one(args):
    qi, order, B, mode = args
    if mode == "std":
        pieces = []
        for j in order:
            if len(tok("\n\n".join(pieces + [ch[j][4]])).input_ids) <= B: pieces.append(ch[j][4])
        text = "\n\n".join(pieces)
    else:
        sel = []
        for j in order:
            if len(tok(render(sel + [j])).input_ids) <= B: sel.append(j)
        text = render(sel)
    q = Q[qi]
    return {"task": "hotpotqa_e", "sample_id": str(qi), "packed_text": text, "question": q["query"],
            "answers": [q["answer"]], "qtype": q["question_type"], "ntok": len(tok(text).input_ids)}


if __name__ == "__main__":
    ch = windows(); top = ranking(ch)
    for B in (1024, 2048):
        for mode in ("std", "dedup"):
            f = OUT / f"mhrag_win_{mode}_budget={B}_hotpotqa_e.jsonl"
            if f.exists(): continue
            with Pool(int(os.environ.get("NP", "12")), initializer=_init) as p:
                rows = p.map(one, [(i, list(top[i]), B, mode) for i in range(len(Q))], chunksize=8)
            tmp = f.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as g:
                for r in rows: g.write(json.dumps(r) + "\n")
            tmp.rename(f)
            print(f.name, "mean ntok", round(np.mean([r["ntok"] for r in rows])), flush=True)
