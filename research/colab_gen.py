"""QA generation + LongBench scoring for packed traces on a Colab T4 (vLLM, greedy).

Same prompt construction as gen_runner.py on the A100 server: the LongBench task template
(dataset2prompt.json) with the packed context, wrapped in the model's chat template, greedy
decoding, per-task max_new_tokens (dataset2maxlen.json), LongBench scorer. The T4 cannot hold
fp16 7B weights, so development runs use an AWQ 4-bit checkpoint; numbers are comparable
between methods within this setup, not with the A100 tables.

usage: python colab_gen.py --model Qwen/Qwen2.5-7B-Instruct-AWQ --out preds out/a.jsonl out/b.jsonl ...
Each trace line needs task, sample_id, packed_text, answers (+ question or generation_prompt).
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from longbench_eval import scorer  # noqa: E402

TEMPL = json.loads((HERE / "dataset2prompt.json").read_text(encoding="utf-8"))
MAXLEN = json.loads((HERE / "dataset2maxlen.json").read_text(encoding="utf-8"))


def base_task(t):
    # LongBench-E dev tasks share the template, length and metric of their LongBench task
    return t[:-2] if t.endswith("_e") else t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct-AWQ")
    ap.add_argument("--out", default="preds")
    ap.add_argument("--questions", default="", help="jsonl with task/sample_id/question if traces lack it")
    ap.add_argument("traces", nargs="+")
    a = ap.parse_args()

    from vllm import LLM, SamplingParams

    llm = LLM(model=a.model, dtype="float16", max_model_len=4096, gpu_memory_utilization=0.9, enforce_eager=False)
    tok = llm.get_tokenizer()
    qs = {}
    if a.questions:
        for line in open(a.questions, encoding="utf-8"):
            r = json.loads(line)
            qs[(r["task"], str(r["sample_id"]))] = r
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for tr in a.traces:
        rows = [json.loads(line) for line in open(tr, encoding="utf-8")]
        for x in rows:  # older traces carry only the packed text
            meta = qs.get((x["task"], str(x["sample_id"])), {})
            x.setdefault("question", meta.get("question"))
            x.setdefault("answers", meta.get("answers", []))
        prompts, params = [], []
        for x in rows:
            t = base_task(x["task"])
            q = x["question"]
            user = x.get("generation_prompt") or TEMPL[t].format(context=x["packed_text"], input=q)
            prompts.append(tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                                   add_generation_prompt=True))
            params.append(SamplingParams(temperature=0.0, max_tokens=int(MAXLEN[t])))
        outs = llm.generate(prompts, params)
        by = defaultdict(list)
        name = Path(tr).stem
        with open(Path(a.out) / f"{name}.preds.jsonl", "w", encoding="utf-8") as f:
            for x, o in zip(rows, outs):
                pred = o.outputs[0].text
                t = base_task(x["task"])
                sc = float(scorer(t, [pred], [x["answers"]], [0], x.get("all_classes")))
                by[x["task"]].append(sc)
                f.write(json.dumps({"task": x["task"], "sample_id": str(x["sample_id"]), "prediction": pred,
                                    "score": sc, "answers": x["answers"]}) + "\n")
        task_scores = {t: sum(v) / len(v) for t, v in by.items()}
        print(name, {t: round(v, 2) for t, v in task_scores.items()},
              "macro", round(sum(task_scores.values()) / len(task_scores), 2), flush=True)


if __name__ == "__main__":
    main()
