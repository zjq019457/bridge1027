"""Controlled cost benchmark: same 100 LongBench test instances (seed 0), one method at a time on an
otherwise idle A100, one instance at a time, 1,024-token budget (Qwen tokenizer).
Reports mean / median seconds per instance, device, and peak GPU memory.
usage: python timing2.py OUT.json"""
import json, os, random, sys, time
import numpy as np
import torch
import localpack as L
import packers as P
from bridgekv import evidence_text_packing as etp

N = 100
S = L.load_samples(); random.Random(0).shuffle(S); S = S[:N]
tok = L.HFLike(L.TOKDIR)
R = {}


def bench(name, fn, device):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    fn(S[0])  # warm-up
    ts = []
    for s in S:
        if device == "gpu": torch.cuda.synchronize()
        t = time.perf_counter(); fn(s)
        if device == "gpu": torch.cuda.synchronize()
        ts.append(time.perf_counter() - t)
    R[name] = dict(mean_s=round(float(np.mean(ts)), 3), median_s=round(float(np.median(ts)), 3), device=device,
                   peak_gpu_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2))
    print(name, R[name], flush=True)
    json.dump(R, open(sys.argv[1], "w"), indent=1)


# --- shared BGE-small encoder (live, not cached) ---
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
bp = open(os.path.join(os.environ["BP_ROOT"], "path_bge")).read().strip()
btk = AutoTokenizer.from_pretrained(bp); bm = AutoModel.from_pretrained(bp).cuda().half().eval()
INSTR = "Represent this sentence for searching relevant passages: "


@torch.no_grad()
def enc(texts):
    out = []
    for i in range(0, len(texts), 256):
        b = btk(texts[i:i + 256], padding=True, truncation=True, max_length=512, return_tensors="pt").to("cuda")
        out.append(torch.nn.functional.normalize(bm(**b).last_hidden_state[:, 0].float(), dim=-1))
    return torch.cat(out)


def bge_sims(s):
    cands, _ = etp.build_candidate_spans(s["sample_id"], s["task"], s["question"], s["context"])
    if not cands: return {}
    v = (enc([c.text for c in cands]) @ enc([INSTR + s["question"]])[0]).tolist()
    return {c.span_id: x for c, x in zip(cands, v)}


bench("BGE-small encoding (shared)", bge_sims, "gpu")
bench("DedupPack packing (CPU, after encoding)", lambda s: P.lex_rep(s, tok, budget=1024, seed_mode=1, sent=1), "cpu")
bench("Chunk packing (CPU, after encoding)", lambda s: P.bge_chunk(s, tok, budget=1024), "cpu")

rp = "/root/models/models/BAAI--bge-reranker-v2-m3/snapshots/master"
rtk = AutoTokenizer.from_pretrained(rp); rm = AutoModelForSequenceClassification.from_pretrained(rp, torch_dtype=torch.float16).cuda().eval()


@torch.no_grad()
def rerank(s):
    cands, _ = etp.build_candidate_spans(s["sample_id"], s["task"], s["question"], s["context"])
    sims = P._bge_sims(s) or {}
    top = sorted(cands, key=lambda c: -sims.get(c.span_id, 0.0))[:100]
    for i in range(0, len(top), 50):
        b = rtk([[s["question"], c.text] for c in top[i:i + 50]], padding=True, truncation=True, max_length=512, return_tensors="pt").to("cuda")
        rm(**b)
bench("Reranker (top-100, after encoding)", rerank, "gpu")
del rm; torch.cuda.empty_cache()

pv = AutoModel.from_pretrained("/root/models/provence", trust_remote_code=True).cuda().eval()


def provence(s):
    cands, pm = etp.build_candidate_spans(s["sample_id"], s["task"], s["question"], s["context"])
    sims = P._bge_sims(s) or {}
    top = sorted(cands, key=lambda c: -sims.get(c.span_id, 0.0))[:60]
    pieces = [etp._expanded_piece(c, pm, 32, role="support", use_tags=False)[0] for c in top]
    if pieces: pv.process([s["question"]], [pieces], always_select_title=False, enable_warnings=False)
bench("Provence pruning (top-60, after encoding)", provence, "gpu")
del pv; torch.cuda.empty_cache()

from llmlingua import PromptCompressor
l2 = PromptCompressor(model_name="/root/models/llmlingua-2-xlm-roberta-large-meetingbank", use_llmlingua2=True, device_map="cuda")
bench("LLMLingua-2", lambda s: l2.compress_prompt(s["context"], target_token=1000, force_tokens=["\n", ".", "?"]), "gpu")
del l2; torch.cuda.empty_cache()

ll = PromptCompressor(model_name=open(os.path.join(os.environ["BP_ROOT"], "path_llama")).read().strip(), device_map="cuda")


def lll(s):
    paras = [p for p in s["context"].split("\n") if p.strip()] or [s["context"]]
    try:
        ll.compress_prompt(paras, question=s["question"], target_token=1000, rank_method="longllmlingua",
                           condition_in_question="after_condition", reorder_context="sort",
                           dynamic_context_compression_ratio=0.3, condition_compare=True, context_budget="+100")
    except Exception:
        pass
bench("LongLLMLingua (Llama-3.1-8B LM)", lll, "gpu")
print("TIMING_DONE")
