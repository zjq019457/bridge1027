"""QA generation over an existing packing trace, with a choice of model and decoding.

Same chat-template wrapping, per-task max_new_tokens and LongBench scorer as
scripts/portable_runner.py; adds nucleus sampling with a per-sample seed and uses
the model's generation_config EOS ids (needed for Llama-3-Instruct's <|eot_id|>).
"""
import argparse
import json
import sys
from pathlib import Path

W = Path("/diskE/jiaqizhang/BRIDGEPACK_SENSITIVITY_V1_20261025")
sys.path.insert(0, str(W / "source" / "SlimInfer_snapshot"))
TASKS = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", "narrativeqa"]

ap = argparse.ArgumentParser()
ap.add_argument("--trace", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--sample", action="store_true")
ap.add_argument("--temperature", type=float, default=0.7)
ap.add_argument("--top_p", type=float, default=0.9)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from longbench_eval import scorer as _lb_scorer  # noqa: E402


def scorer(task, preds, answers, lengths, all_classes):
    if not task.startswith("babilong"):
        return _lb_scorer(task, preds, answers, lengths, all_classes)
    # BABILong: correct if the gold location word appears in the prediction
    hits = [any(str(a).lower() in str(p).lower().split()[:8] or str(a).lower() in str(p).lower() for a in ans)
            for p, ans in zip(preds, answers)]
    return 100.0 * sum(hits) / max(1, len(hits))
from tools.run_evidence_text_packing_generation_qwen32 import _max_new_tokens_for  # noqa: E402

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
traces = [json.loads(l) for l in open(args.trace)]
tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16, device_map={"": 0},
                                             attn_implementation="sdpa", trust_remote_code=True)
model.eval()
eos = model.generation_config.eos_token_id or tok.eos_token_id

done = {}
pred_path = out / "raw_predictions.jsonl"
if pred_path.exists():
    for l in open(pred_path):
        r = json.loads(l)
        done[(r["task"], r["sample_id"])] = r
with pred_path.open("a") as f:
    for i, x in enumerate(traces):
        key = (x["task"], str(x["sample_id"]))
        if key in done:
            continue
        chat = tok.apply_chat_template([{"role": "user", "content": x["generation_prompt"]}],
                                       tokenize=False, add_generation_prompt=True)
        inputs = tok(chat, return_tensors="pt", add_special_tokens=False).to(model.device)
        n_in = inputs.input_ids.shape[1]
        kw = dict(max_new_tokens=_max_new_tokens_for(x["task"], 0), num_beams=1, use_cache=True,
                  eos_token_id=eos, pad_token_id=tok.pad_token_id or tok.eos_token_id)
        if args.sample:
            torch.manual_seed(args.seed * 100003 + i)
            kw.update(do_sample=True, temperature=args.temperature, top_p=args.top_p, top_k=0)
        else:
            kw.update(do_sample=False)
        with torch.no_grad():
            o = model.generate(**inputs, **kw)[0]
        pred = tok.decode(o[n_in:], skip_special_tokens=True)
        score = float(scorer(x["task"], [pred], [x.get("answers", [])], [n_in], x.get("all_classes")))
        r = {"task": x["task"], "sample_id": key[1], "prediction": pred, "score": score,
             "answers": x.get("answers", []), "input_token_count": n_in}
        done[key] = r
        f.write(json.dumps(r) + "\n")
        f.flush()

rows = list(done.values())
by = {}
for t in sorted({r["task"] for r in rows}, key=lambda t: (TASKS.index(t) if t in TASKS else 99, t)):
    sub = [r for r in rows if r["task"] == t]
    if not sub:
        continue
    by[t] = float(scorer(t, [r["prediction"] for r in sub], [r["answers"] for r in sub],
                         [r["input_token_count"] for r in sub], None))
(out / "macro_summary.json").write_text(json.dumps({
    "sample_count": len(rows), "task_scores": by, "macro": sum(by.values()) / max(1, len(by)),
    "model": args.model, "trace": args.trace, "sample": args.sample, "seed": args.seed,
    "temperature": args.temperature, "top_p": args.top_p}, indent=2))
print("GEN_DONE", out, sum(by.values()) / max(1, len(by)))
