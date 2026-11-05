"""Streaming generation: keep one vLLM model loaded and score every trace that appears in DIR/out
(skipping ones already scored) until DIR/PACK_DONE exists and nothing is left.
usage: python gen_stream.py MODEL DIR [GPU_UTIL]"""
import json, sys, time
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from colab_gen import TEMPL, MAXLEN, base_task, scorer
from vllm import LLM, SamplingParams
model, d = sys.argv[1], Path(sys.argv[2])
util = float(sys.argv[3]) if len(sys.argv) > 3 else 0.9
llm = LLM(model=model, dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=util)
tok = llm.get_tokenizer()
pd = d / "preds"; pd.mkdir(exist_ok=True)
while True:
    todo = [t for t in sorted((d / "out").glob("*.jsonl")) if "fullctx" not in t.name
            and not (pd / f"{t.stem}.preds.jsonl").exists() and not (pd / f"{t.stem}.lock").exists() and time.time() - t.stat().st_mtime > 20]
    if not todo:
        if (d / "PACK_DONE").exists(): break
        time.sleep(15); continue
    tr = None
    for t in todo:  # claim a trace (several generators may share one directory)
        try:
            import os; os.close(os.open(pd / f"{t.stem}.lock", os.O_CREAT | os.O_EXCL)); tr = t; break
        except FileExistsError:
            continue
    if tr is None:
        if (d / "PACK_DONE").exists(): break
        time.sleep(15); continue
    rows = [json.loads(l) for l in open(tr, encoding="utf-8")]
    prompts, params = [], []
    for x in rows:
        t = base_task(x["task"])
        user = TEMPL[t].format(context=x["packed_text"], input=x["question"])
        prompts.append(tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True))
        params.append(SamplingParams(temperature=0.0, max_tokens=int(MAXLEN[t])))
    t0 = time.time(); outs = llm.generate(prompts, params, use_tqdm=False)
    by = defaultdict(list)
    tmp = pd / f"{tr.stem}.preds.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for x, o in zip(rows, outs):
            p = o.outputs[0].text; t = base_task(x["task"])
            sc = float(scorer(t, [p], [x["answers"]], [0], x.get("all_classes")))
            by[x["task"]].append(sc)
            f.write(json.dumps({"task": x["task"], "sample_id": str(x["sample_id"]), "prediction": p, "score": sc, "answers": x["answers"]}) + "\n")
    tmp.rename(pd / f"{tr.stem}.preds.jsonl")
    ts = {t: sum(v) / len(v) for t, v in by.items()}
    print(tr.stem, {t: round(v, 2) for t, v in ts.items()}, "macro", round(sum(ts.values()) / len(ts), 2), f"{time.time()-t0:.0f}s", flush=True)
print("STREAM_DONE")
