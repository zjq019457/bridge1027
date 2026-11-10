"""MultiHop-RAG: how much duplication does standard RAG packing produce with common chunkers, and
how much evidence would deduplicated packing recover? CPU only (BGE-small on CPU).
Output: per config x budget: duplication %, gold-evidence recall %, answer coverage % (standard vs dedup)."""
import json, re, sys, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import localpack as L
from transformers import AutoModel, AutoTokenizer

D = Path(__file__).parent / "data_mhrag"
Q = json.load(open(D / "MultiHopRAG.json", encoding="utf-8"))
C = json.load(open(D / "corpus.json", encoding="utf-8"))
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 10**9
Q = [q for q in Q if q["question_type"] != "null_query"][:LIMIT]
tok = L.HFLike(L.TOKDIR)
docs = [c["title"] + "\n" + c["body"] for c in C]

BGE = "D:/BridgePack_backup_20261027/models/bge-small"
btk = AutoTokenizer.from_pretrained(BGE); bm = AutoModel.from_pretrained(BGE).eval(); torch.set_num_threads(8)
INSTR = "Represent this sentence for searching relevant passages: "


@torch.no_grad()
def enc(texts, bs=64):
    out = []
    for i in range(0, len(texts), bs):
        b = btk(texts[i:i + bs], padding=True, truncation=True, max_length=512, return_tensors="pt")
        out.append(torch.nn.functional.normalize(bm(**b).last_hidden_state[:, 0], dim=-1))
    return torch.cat(out).numpy()


def chunks_llama(size, overlap):
    from llama_index.core import Document
    from llama_index.core.node_parser import SentenceSplitter
    sp = SentenceSplitter(chunk_size=size, chunk_overlap=overlap)
    out = []
    for i, d in enumerate(docs):
        for n in sp.get_nodes_from_documents([Document(text=d)]):
            out.append((i, n.start_char_idx, n.end_char_idx, n.text, n.text))
    return out


def chunks_langchain(size, overlap):
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    sp = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap, add_start_index=True)
    out = []
    for i, d in enumerate(docs):
        for x in sp.create_documents([d]):
            s = x.metadata["start_index"]; out.append((i, s, s + len(x.page_content), x.page_content, x.page_content))
    return out


def chunks_window(w=3):
    from llama_index.core import Document
    from llama_index.core.node_parser import SentenceWindowNodeParser
    sp = SentenceWindowNodeParser.from_defaults(window_size=w)
    out = []
    for i, d in enumerate(docs):
        for n in sp.get_nodes_from_documents([Document(text=d)]):
            win = n.metadata["window"]; s = d.find(win[:200])
            if s < 0: s = d.find(n.text[:100])
            out.append((i, max(0, s), max(0, s) + len(win), n.text, win))  # retrieve by sentence, pack window
    return out


def norm(s): return L.norm(s)


def dup_rate(text):
    w = text.split(); seen = set(); d = 0
    for i in range(len(w) - 7):
        g = tuple(w[i:i + 8]); d += g in seen; seen.add(g)
    return d, max(1, len(w))


def pack_standard(order, ch, B):
    pieces = []
    for j in order[:200]:
        cand = "\n\n".join(pieces + [ch[j][4]])
        if len(tok(cand).input_ids) <= B: pieces.append(ch[j][4])
    return "\n\n".join(pieces)


def render(sel, ch):
    by = {}
    for j in sel:
        d, a, b = ch[j][0], ch[j][1], ch[j][2]
        by.setdefault(d, []).append([a, b])
    rank_doc = []
    for j in sel:
        if ch[j][0] not in rank_doc: rank_doc.append(ch[j][0])
    out = []
    for d in rank_doc:
        iv = sorted(by[d]); m = [iv[0]]
        for a, b in iv[1:]:
            if a <= m[-1][1]: m[-1][1] = max(m[-1][1], b)
            else: m.append([a, b])
        out += [docs[d][a:b] for a, b in m]
    return "\n\n".join(out)


def pack_dedup(order, ch, B):
    sel = []
    for j in order[:200]:
        if len(tok(render(sel + [j], ch)).input_ids) <= B: sel.append(j)
    return render(sel, ch)


qe = enc([INSTR + q["query"] for q in Q])
res = {}
ONLY = __import__("os").environ.get("ONLY")
cfgs = {"LlamaIndex 1024/200 (default)": lambda: chunks_llama(1024, 200), "LlamaIndex 256/50": lambda: chunks_llama(256, 50),
        "LangChain 1000/200 chars": lambda: chunks_langchain(1000, 200), "LlamaIndex sentence window (3)": chunks_window}
for name, make in cfgs.items():
    if ONLY and ONLY not in name: continue
    ch = make(); E = enc([c[3] for c in ch]); S = qe @ E.T
    for B in (1024, 2048):
        st = {"std": [0, 0, 0, 0, 0], "dedup": [0, 0, 0, 0, 0]}
        for qi, q in enumerate(Q):
            order = list(np.argsort(-S[qi]))
            facts = [norm(e["fact"]) for e in q["evidence_list"]]
            for k, ctx in (("std", pack_standard(order, ch, B)), ("dedup", pack_dedup(order, ch, B))):
                d, n = dup_rate(ctx); nc = norm(ctx)
                st[k][0] += d; st[k][1] += n
                st[k][2] += np.mean([f[:120] in nc for f in facts]) if facts else 0
                if q["answer"].lower() not in ("yes", "no"):
                    st[k][3] += norm(q["answer"]) in nc; st[k][4] += 1
        nq = len(Q)
        res[f"{name} | B={B}"] = {k: dict(dup=round(100 * v[0] / v[1], 1), evidence_recall=round(100 * v[2] / nq, 1),
                                          answer_cov=round(100 * v[3] / max(1, v[4]), 1)) for k, v in st.items()}
        print(name, B, res[f"{name} | B={B}"], f"chunks={len(ch)}", flush=True)
json.dump(res, open(Path(__file__).parent / "mhrag_dup.json", "w"), indent=1)
