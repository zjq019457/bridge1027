"""Truncation baselines and full-context traces in the portable runner's trace format.

usage: simple_pack.py MODE TOKENIZER OUT [BUDGET]
  MODE head   : "Question/Selected evidence" prefix + first tokens of the context, exact BUDGET model tokens
  MODE middle : same prefix + first half and last half of the context (LongBench-style), exact BUDGET
  MODE full   : whole context in the task template; LongBench middle truncation of the prompt to BUDGET
"""
import json
import sys
from pathlib import Path

W = Path("/diskE/jiaqizhang/BRIDGEPACK_SENSITIVITY_V1_20261025")
MODE, TOKP, OUT = sys.argv[1], sys.argv[2], Path(sys.argv[3])
BUDGET = int(sys.argv[4]) if len(sys.argv) > 4 else 1024
sys.path.insert(0, str(W / "source" / "SlimInfer_snapshot"))
sys.path.insert(0, str(W / "scripts"))
import portable_runner as pr  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

templates = json.loads((W / "source" / "SlimInfer_snapshot" / "configs" / "dataset2prompt.json").read_text())
ids, samples, _ = pr.load_selection_inputs(W / "assets" / "longbench_g1_data", W / "golden", None)
tok = AutoTokenizer.from_pretrained(TOKP, local_files_only=True, trust_remote_code=True, use_fast=True)
enc = lambda s: tok(s, add_special_tokens=False).input_ids  # noqa: E731
dec = lambda x: tok.decode(x, skip_special_tokens=True)  # noqa: E731


def fit(text_ids, keep, middle):
    if len(text_ids) <= keep:
        return dec(text_ids)
    if middle:
        return dec(text_ids[:keep // 2]) + dec(text_ids[-(keep - keep // 2):])
    return dec(text_ids[:keep])


with OUT.open("w") as f:
    for t in pr.TASKS:
        for s in samples[t]:
            q, ctx = s["question"], s["context"]
            full_len = None
            if MODE == "full":
                gp = templates[t].format(context=ctx, input=q, question=q)
                g = enc(gp)
                full_len = len(g)
                packed = None
                if len(g) > BUDGET:
                    gp = fit(g, BUDGET, True)
                n = len(enc(gp))
            else:
                prefix = f"Question: {q}\n\nSelected evidence:"
                room = BUDGET - len(enc(prefix + "\n\n"))
                c = enc(ctx)
                while True:
                    packed = "\n\n".join([prefix, fit(c, room, MODE == "middle")]).strip()
                    n = len(enc(packed))
                    if n <= BUDGET:
                        break
                    room -= max(1, n - BUDGET)
                gp = templates[t].format(context=packed, input=q, question=q)
            f.write(json.dumps({
                "task": t, "sample_id": s["sample_id"], "variant": f"{MODE}_{BUDGET}", "budget": BUDGET,
                "packed_text": packed, "generation_prompt": gp, "answers": s["answers"],
                "all_classes": s["all_classes"], "question": q, "model_token_count": n,
                "full_prompt_tokens": full_len,
            }) + "\n")
print("PACK_DONE", MODE, OUT)
