"""LLMLingua-2 compression baseline at an exact budget: compress the context to the budget
(minus the question prefix), then trim with the target model tokenizer to the exact budget.
usage: python lingua.py OUT_RAW.jsonl  (compress once)  |  python lingua.py --trim RAW TOKDIR OUT BUDGET"""
import json, sys, time
import localpack as L
if sys.argv[1] == "--trim":
    raw, tokdir, out, budget = sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
    tok = L.HFLike(tokdir)
    import packers as P
    from bridgekv import evidence_text_packing as etp
    with open(out, "w", encoding="utf-8") as f:
        for line in open(raw, encoding="utf-8"):
            r = json.loads(line)
            r["packed_text"] = P._emit(tok, r["question"], [r["compressed"]], budget)
            f.write(json.dumps(r) + "\n")
    sys.exit()
from llmlingua import PromptCompressor
pc = PromptCompressor(model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank", use_llmlingua2=True, device_map="cuda")
t0 = time.time()
with open(sys.argv[1], "w", encoding="utf-8") as f:
    for n, s in enumerate(L.load_samples()):
        c = pc.compress_prompt(s["context"], target_token=1000, force_tokens=["\n", ".", "?"])["compressed_prompt"]
        f.write(json.dumps({"task": s["task"], "sample_id": s["sample_id"], "question": s["question"],
                            "answers": s["answers"], "compressed": c}) + "\n")
        if n % 100 == 0: print(n, round(time.time() - t0), flush=True)
print("LINGUA_DONE", round(time.time() - t0))
