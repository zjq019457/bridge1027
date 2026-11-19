"""CompAct (Yoon et al., EMNLP 2024) on every test instance with the released CompAct-7b compressor.
Retrieval is the same as for the Provence baseline (top BGE-small spans padded by 32 tokens, in rank order); as in
the official inference script, documents are read in segments of 5 for at most 6 iterations (30 documents), each
iteration rewrites the summary given the previous summary and evaluation, and the loop stops at [COMPLETE].
Prompts, greedy decoding with 500 new tokens and output parsing follow utils.py / run_prompt.py; the reader gets
the final summary (--read_wo_prev_eval, as in the official script). Batched across instances with vLLM.
usage: CUDA_VISIBLE_DEVICES=0 COMPACT=... python compact_run.py OUT.jsonl [LIMIT]"""
import json, os, re, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import localpack as L
import packers as P
from bridgekv import evidence_text_packing as etp
from vllm import LLM, SamplingParams

OUT, LIMIT = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 10**9
SEG, MAXIT = 5, 6
I0 = "1. Generate a summary of source documents to answer the question. Ensure the summary is under 200 words and does not include any pronouns. DO NOT make assumptions or attempt to answer the question; your job is to summarize only.\n\n2. Evaluate the summary based solely on the information of it, without any additional background context: if it lacks sufficient details to answer the question, print '[INCOMPLETE]'. If it provides all necessary details, print '[COMPLETE]'. You should provide the reason of evalution."
I1 = "1. Generate a summary of the previous summary and the source documents to answer the question based on the evaluation of the previous summary. The evaluation indicates the missing information needed to answer the question. Ensure the summary is under 200 words and does not include any pronouns. DO NOT make assumptions or attempt to answer the question; your job is to summarize only.\n\n2. Evaluate the summary based solely on the information of it, without any additional background context: if it lacks sufficient details to answer the question, print '[INCOMPLETE]'. If it provides all necessary details, print '[COMPLETE]'. You should provide the reason of evalution."


def create_prompt(question, it, document_input, prev_summary, prev_eval):
    if it == 0:
        p = f"{I0}\n\nQuestion: {question}\n\nSource documents: {document_input}\n\nSummary:"
    else:
        p = f"{I1}\n\nQuestion: {question}\n\nPrevious summary: {prev_summary}\n\nEvaluation of previous summary: {prev_eval}\n\nSource documents: {document_input}\n\nSummary:"
    return tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)


def parse(text):  # parse_output_without_sentence (utils.py); a missing evaluation is treated as empty
    m1 = re.search(r'(Summary:)(.*?)(?=Evaluation:|$)', text, re.DOTALL)
    m2 = re.search(r'(^.*?)(?=Evaluation:|$)', text, re.DOTALL)
    me = re.search(r'(Evaluation:)(.*?)(?=Summary:|$)', text, re.DOTALL)
    summary = m1.group(2).strip() if m1 else (m2.group(1).strip() if m2 else "")
    ev = me.group(2).strip() if me else ""
    return summary.replace("\n\n", ""), ev.replace("\n\n", "")


S = L.load_samples()[:LIMIT]
docs = {}
for s in S:
    cands, page_map = etp.build_candidate_spans(s["sample_id"], s["task"], s["question"], s["context"])
    sims = P._bge_sims(s) or {}
    top = sorted(cands, key=lambda c: -sims.get(c.span_id, 0.0))[:SEG * MAXIT]
    # official document string: f"{doc['title']} {doc['text']}" (our spans have no title)
    docs[f"{s['task']}/{s['sample_id']}"] = [f" {etp._expanded_piece(c, page_map, 32, role='support', use_tags=False)[0].strip()}" for c in top]
llm = LLM(model=os.environ["COMPACT"], dtype="bfloat16", max_model_len=8192, gpu_memory_utilization=0.9)
tok = llm.get_tokenizer()
sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=500)
st = {k: {"summary": [], "eval": [], "done": False} for k in docs}
t0 = time.time()
for it in range(MAXIT):
    keys, prompts = [], []
    for s in S:
        k = f"{s['task']}/{s['sample_id']}"; seg = docs[k][it * SEG:(it + 1) * SEG]
        if st[k]["done"] or not seg:
            continue
        ps = st[k]["summary"][-1] if it else ""
        pe = st[k]["eval"][-1].replace("[INCOMPLETE]", "").strip() if it else ""
        keys.append(k); prompts.append(create_prompt(s["question"], it, "\n".join(seg), ps, pe))
    if not prompts:
        break
    outs = llm.generate(prompts, sp, use_tqdm=False)
    for k, o in zip(keys, outs):
        summ, ev = parse(o.outputs[0].text.strip())
        st[k]["summary"].append(summ); st[k]["eval"].append(ev)
        if "[COMPLETE]" in ev:
            st[k]["done"] = True
    print("iteration", it + 1, "prompts", len(prompts), "complete", sum(v["done"] for v in st.values()), round(time.time() - t0), flush=True)
with open(OUT, "w", encoding="utf-8") as f:
    for k, v in st.items():
        f.write(json.dumps({"key": k, "summary": v["summary"][-1] if v["summary"] else "", "iterations": len(v["summary"]),
                            "complete": v["done"], "history": v["summary"]}) + "\n")
print("COMPACT_DONE", round(time.time() - t0), flush=True)
