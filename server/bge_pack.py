"""BGE chunk packing (bge-small-en-v1.5, CLS pooling + L2 norm) via the repo's _dense_pack,
on the same candidate spans, neighbor window and exact-budget admission as the other baselines.
usage: bge_pack.py TOKENIZER OUT
"""
import json
import sys
from pathlib import Path

W = Path("/diskE/jiaqizhang/BRIDGEPACK_SENSITIVITY_V1_20261025")
TOKP, OUT = sys.argv[1], Path(sys.argv[2])
sys.path.insert(0, str(W / "source" / "SlimInfer_snapshot"))
sys.path.insert(0, str(W / "scripts"))
import portable_runner as pr  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModel, AutoTokenizer  # noqa: E402
from tools.run_bridgepack_dense_retriever_packing import _dense_pack  # noqa: E402

BGE = str(W / "models_small_bge")
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class BGE_Embedder:
    def __init__(self):
        self.tok = AutoTokenizer.from_pretrained(BGE)
        self.model = AutoModel.from_pretrained(BGE).cuda().eval()

    @torch.no_grad()
    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
        out = []
        for i in range(0, len(texts), 128):
            b = self.tok(texts[i:i + 128], padding=True, truncation=True, max_length=512, return_tensors="pt").to("cuda")
            e = self.model(**b).last_hidden_state[:, 0]
            out.append(torch.nn.functional.normalize(e, dim=-1).float().cpu())
        return torch.cat(out).tolist()


class QueryAware(BGE_Embedder):
    # bge-v1.5 retrieval convention: the instruction is prepended to the query only
    def encode(self, texts, **kw):
        if len(texts) == 1 and getattr(self, "_next_is_query", False):
            self._next_is_query = False
            return super().encode([QUERY_INSTRUCTION + texts[0]], **kw)
        return super().encode(texts, **kw)


ids, samples, _ = pr.load_selection_inputs(W / "assets" / "longbench_g1_data", W / "golden", None)
tok = AutoTokenizer.from_pretrained(TOKP, local_files_only=True, use_fast=True)
emb = QueryAware()
with OUT.open("w") as f:
    for t in pr.TASKS:
        for s in samples[t]:
            emb._next_is_query = True
            r = _dense_pack(emb, tok, s, 1024, 32, 64, BGE)
            r["variant"] = "bge_chunk_packing"
            r["model_token_count"] = len(tok(r["packed_text"], add_special_tokens=False).input_ids)
            f.write(json.dumps(r) + "\n")
print("BGE_PACK_DONE", OUT)
