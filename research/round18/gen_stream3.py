"""gen_stream2 with optional sampling: GEN_TEMP / GEN_TOPP / GEN_SEED (env) and a SUFFIX appended to the
prediction file name, so several sampled runs of the same trace can share one directory.
usage: GEN_TEMP=0.7 GEN_TOPP=0.95 GEN_SEED=1 SUFFIX=_t07s1 python gen_stream3.py MODEL DIR [GPU_UTIL]"""
import json, os, sys, time
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from colab_gen import TEMPL, MAXLEN, base_task, scorer
from vllm import LLM, SamplingParams
model, d = sys.argv[1], Path(sys.argv[2])
util = float(sys.argv[3]) if len(sys.argv) > 3 else 0.9
TEMP, TOPP = float(os.environ.get("GEN_TEMP", "0")), float(os.environ.get("GEN_TOPP", "1"))
SEED, SUF = int(os.environ.get("GEN_SEED", "0")), os.environ.get("SUFFIX", "")
llm = LLM(model=model, dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=util, seed=SEED)
tok = llm.get_tokenizer()
pd = d / "preds"; pd.mkdir(exist_ok=True)
while True:
    todo = [t for t in sorted((d / "out").glob("*.jsonl")) if "fullctx" not in t.name and t.stat().st_size > 0
            and not (pd / f"{t.stem}{SUF}.preds.jsonl").exists() and not (pd / f"{t.stem}{SUF}.lock").exists()
            and time.time() - t.stat().st_mtime > 20]
    if not todo:
        if (d / "PACK_DONE").exists(): break
        time.sleep(15); continue
    tr = None
    for t in todo:
        try:
            os.close(os.open(pd / f"{t.stem}{SUF}.lock", os.O_CREAT | os.O_EXCL)); tr = t; break
        except FileExistsError:
            continue
    if tr is None:
        if (d / "PACK_DONE").exists(): break
        time.sleep(15); continue
    rows = [json.loads(l) for l in open(tr, encoding="utf-8")]
    prompts, params = [], []
    for k, x in enumerate(rows):
        t = base_task(x["task"])
        user = TEMPL[t].format(context=x["packed_text"], input=x["question"])
        prompts.append(tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True))
        params.append(SamplingParams(temperature=TEMP, top_p=TOPP, seed=(SEED * 100003 + k) if TEMP > 0 else None,
                                     max_tokens=int(MAXLEN[t])))
    t0 = time.time(); outs = llm.generate(prompts, params, use_tqdm=False)
    by = defaultdict(list)
    tmp = pd / f"{tr.stem}{SUF}.preds.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for x, o in zip(rows, outs):
            p = o.outputs[0].text; t = base_task(x["task"])
            sc = float(scorer(t, [p], [x["answers"]], [0], x.get("all_classes")))
            by[x["task"]].append(sc)
            f.write(json.dumps({"task": x["task"], "sample_id": str(x["sample_id"]), "prediction": p, "score": sc, "answers": x["answers"]}) + "\n")
    tmp.rename(pd / f"{tr.stem}{SUF}.preds.jsonl")
    ts = {t: sum(v) / len(v) for t, v in by.items()}
    print(tr.stem + SUF, {t: round(v, 2) for t, v in ts.items()}, "macro", round(sum(ts.values()) / len(ts), 2), f"{time.time()-t0:.0f}s", flush=True)
print("STREAM_DONE")
