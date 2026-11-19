"""EXIT (Hwang et al., Findings of ACL 2025) sentence scores for every test instance.
Retrieval is the same as for the Provence baseline: the top-60 BGE-small spans, each padded by 32 tokens, are the
retrieved documents. Each document is split with spaCy's rule-based sentencizer and every sentence is scored with
the official prompt (compressors/baselines/exit/core.py) and the released adapter: score = P(Yes)/(P(Yes)+P(No))
on the next token. The adapter runs in float16 rather than 4-bit (bitsandbytes is not installed).
usage: CUDA_VISIBLE_DEVICES=0 EXIT_BASE=... EXIT_ADAPTER=... extenv/bin/python exit_scores.py OUT.jsonl [LIMIT]"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import localpack as L
import packers as P
from bridgekv import evidence_text_packing as etp
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
sys.path.insert(0, os.path.join(P._AUX, "sc_pkgs"))
from spacy.lang.en import English

TEMPLATE = """<start_of_turn>user
Query:
{query}
Full context:
{context}
Sentence:
{sentence}
Is this sentence useful in answering the query? Answer only "Yes" or "No".<end_of_turn>
<start_of_turn>model
"""
OUT, LIMIT = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 10**9
nlp = English(); nlp.add_pipe("sentencizer")
tok = AutoTokenizer.from_pretrained(os.environ["EXIT_BASE"], use_fast=True)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"
base = AutoModelForCausalLM.from_pretrained(os.environ["EXIT_BASE"], torch_dtype=torch.float16).cuda()
model = PeftModel.from_pretrained(base, os.environ["EXIT_ADAPTER"]).eval()
inner = model.get_base_model()  # GemmaForCausalLM with the LoRA layers injected
ids = [tok.encode(w, add_special_tokens=False) for w in ("Yes", "No")]
assert all(len(i) == 1 for i in ids), ids
YES, NO = ids[0][0], ids[1][0]


@torch.inference_mode()
def score(prompts, bs=32):
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    out = [0.0] * len(prompts)
    for a in range(0, len(order), bs):
        ii = order[a:a + bs]
        enc = tok([prompts[i] for i in ii], return_tensors="pt", padding=True).to("cuda")
        with torch.autocast("cuda", dtype=torch.float16):
            h = inner.model(**enc).last_hidden_state[:, -1]  # left padding: last position is the prompt end
            logits = inner.lm_head(h).float()                # only the next-token logits are needed
        p = torch.softmax(logits[:, [YES, NO]], dim=-1)[:, 0].tolist()
        for i, v in zip(ii, p):
            out[i] = v
    return out


done = set()
if os.path.exists(OUT):
    done = {json.loads(l)["key"] for l in open(OUT, encoding="utf-8")}
S = [s for s in L.load_samples() if f"{s['task']}/{s['sample_id']}" not in done][:LIMIT]
print("todo", len(S), flush=True)
t0 = time.time()
with open(OUT, "a", encoding="utf-8") as f:
    for n, s in enumerate(S):
        cands, page_map = etp.build_candidate_spans(s["sample_id"], s["task"], s["question"], s["context"])
        sims = P._bge_sims(s) or {}
        top = sorted(cands, key=lambda c: -sims.get(c.span_id, 0.0))[:60]
        pieces = [etp._expanded_piece(c, page_map, 32, role="support", use_tags=False)[0] for c in top]
        sents = [[x.text.strip() for x in nlp(p.strip()).sents if x.text.strip()] for p in pieces]
        flat = [(i, j) for i, ss in enumerate(sents) for j in range(len(ss))]
        sc = score([TEMPLATE.format(query=s["question"], context=pieces[i].strip(), sentence=sents[i][j]) for i, j in flat])
        res = [[] for _ in pieces]
        for (i, j), v in zip(flat, sc):
            res[i].append([sents[i][j], round(v, 5)])
        f.write(json.dumps({"key": f"{s['task']}/{s['sample_id']}", "pieces": res}) + "\n"); f.flush()
        if n % 50 == 0:
            print(n, len(flat), round(time.time() - t0), flush=True)
print("EXIT_DONE", round(time.time() - t0), flush=True)
