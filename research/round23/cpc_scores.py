"""CPC (Liskavets et al., AAAI 2025) sentence relevance scores on the full context of every test instance, with the
released Mistral-7B compressor (LoRA, bidirectional attention). Sentence splitting, chunking (6,144 tokens) and
scoring follow the official code (util/preprocessing.py: SamplePreprocessor and compress_sample); only the final
selection is left to the packer, which fills the exact budget of the reader's tokenizer in score order and emits
the sentences in their original order, as CPC does. SDPA attention replaces flash-attention (not installed).
usage: CUDA_VISIBLE_DEVICES=0 CPC_DIR=... CPC_BASE=... CPC_LORA=... CPC_TOK=... extenv/bin/python cpc_scores.py OUT.jsonl [LIMIT]"""
import json, os, sys, time, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import localpack as L
CPC = os.environ["CPC_DIR"]
sys.path.insert(0, CPC)
# util/util.py calls evaluate.load("accuracy") at import time for training metrics; scoring never uses it
sys.path.insert(0, os.path.join(os.path.dirname(CPC), "cpc_stub"))
import torch
from model.model import load_model_and_tokenizer
from util.preprocessing import SamplePreprocessor
from util.util import SentenceEmbeddingType
from util.torch_util import mean_pooling, cos_sim

OUT, LIMIT = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 10**9
cfg = json.load(open(os.path.join(CPC, "configs", "cpc-1.0-mistral.json")))
cfg.update(model_name_or_path=os.environ["CPC_BASE"], attn_implementation="sdpa",
           output_dir=tempfile.mkdtemp(), dataset_path=tempfile.mkdtemp())
cfg_path = os.path.join(tempfile.mkdtemp(), "cpc-1.0-mistral-local.json")
json.dump(cfg, open(cfg_path, "w"))
model, tokenizer = load_model_and_tokenizer(config_path=cfg_path, tokenizer_name_or_path=os.environ["CPC_TOK"],
                                            lora_name_or_path=os.environ["CPC_LORA"])
model.eval()
proc = SamplePreprocessor(tokenizer=tokenizer, max_context_len=cfg["max_seq_length"], use_question_as_suffix=False,
                          sentence_embedding_type=SentenceEmbeddingType.AVG)


@torch.no_grad()
def scores(encodings):
    """compress_sample() from util/preprocessing.py up to the similarity list (no selection)."""
    eq = encodings["question"]
    iq = {"input_ids": torch.LongTensor(eq["input_ids"]).cuda(), "attention_mask": torch.LongTensor(eq["attention_mask"]).cuda()}
    q_emb = mean_pooling(model(**iq, is_train=False), iq["attention_mask"])
    out = []
    for ec in encodings["context"]:
        ic = {"input_ids": torch.LongTensor(ec["text_input_ids"])[None].cuda(),
              "attention_mask": torch.LongTensor([1 for _ in ec["text_input_ids"]])[None].cuda()}
        seg_ids = torch.LongTensor(ec["text_segment_ids"])[None].cuda()
        oc = model(**ic, is_train=False)
        embs = torch.cat([mean_pooling(oc, seg_ids == k) for k in range(ec["text_segment_ids"][-1] + 1)], 0)
        sim = cos_sim(embs, q_emb)[:, 0].float().cpu().numpy()
        assert len(sim) == len(ec["segments"])
        out += [[seg, round(float(v), 5)] for seg, fake, v in zip(ec["segments"], ec["segments_labels"], sim) if not fake]
    return out


done = set()
if os.path.exists(OUT):
    done = {json.loads(l)["key"] for l in open(OUT, encoding="utf-8")}
S = [s for s in L.load_samples() if f"{s['task']}/{s['sample_id']}" not in done][:LIMIT]
print("todo", len(S), flush=True)
t0 = time.time()
with open(OUT, "a", encoding="utf-8") as f:
    for n, s in enumerate(S):
        enc = proc(context=s["context"], question=s["question"], question_for_suffix=s["question"])
        f.write(json.dumps({"key": f"{s['task']}/{s['sample_id']}", "sents": scores(enc)}) + "\n"); f.flush()
        if n % 50 == 0:
            print(n, round(time.time() - t0), flush=True)
print("CPC_DONE", round(time.time() - t0), flush=True)
