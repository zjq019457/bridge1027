"""LongLLMLingua baseline (question-aware, Llama-2-7B compressor), compressed once to ~1,000
tokens and trimmed per downstream tokenizer with lingua.py --trim. Sharded over GPUs.
usage: python longlingua.py OUT_RAW.jsonl SHARD NSHARD"""
import json, sys, time
import localpack as L
from llmlingua import PromptCompressor
out, k, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
pc = PromptCompressor(model_name=__import__("os").environ["LLL_PATH"], device_map="cuda")
S = [s for i, s in enumerate(L.load_samples()) if i % n == k]
t0 = time.time()
with open(out, "w", encoding="utf-8") as f:
    for j, s in enumerate(S):
        paras = [p for p in s["context"].split("\n") if p.strip()] or [s["context"]]
        try:
            c = pc.compress_prompt(paras, question=s["question"], target_token=1000, rank_method="longllmlingua",
                                   condition_in_question="after_condition", reorder_context="sort",
                                   dynamic_context_compression_ratio=0.3, condition_compare=True,
                                   context_budget="+100")["compressed_prompt"]
        except Exception as e:  # keep going; record the failure
            c = ""; print("ERR", s["sample_id"], repr(e)[:200], flush=True)
        f.write(json.dumps({"task": s["task"], "sample_id": s["sample_id"], "question": s["question"],
                            "answers": s["answers"], "compressed": c}) + "\n")
        if j % 20 == 0: print(j, len(S), round(time.time() - t0), flush=True)
print("LLL_DONE", round(time.time() - t0))
