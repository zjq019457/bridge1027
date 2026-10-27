#!/usr/bin/env python3
"""Compare the nine decoded mismatches without fabricating historical IDs/logits."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
RUN = REPO / "results" / "bridgepack_g1_decoded_reproduction_20260928_180900"
AUDIT = REPO / "results" / "bridgepack_g1_strict_reproduction_audit_20260928_102546"
REF = REPO / "results" / "bridgepack_g1_qwen1024_matched_20260727_004934"
MODEL = Path("/home/yu-bai/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28")
TASKS = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", "narrativeqa"]


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def digest(x: str) -> str:
    return hashlib.sha256(x.encode("utf-8", errors="replace")).hexdigest()


def score_one(task, pred, answers, all_classes):
    from longbench_eval import dataset2metric
    metric = dataset2metric[task]
    vals = []
    for answer in answers or []:
        vals.append(float(metric(pred, answer, all_classes=all_classes)))
    return 100.0 * max(vals or [0.0])


def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    new_rows = [x for x in read_jsonl(RUN / "raw_predictions.jsonl") if not x["decoded_prediction_exact"]]
    ref_packs = {}
    new_packs = {}
    for task in TASKS:
        ref_path = REF / "full" / "packed_prompts_build" / "packed_prompts" / "same_spans_source_order" / "budget_1024" / f"{task}.jsonl"
        new_path = RUN / "packing" / "packed_prompts_build" / "packed_prompts" / "same_spans_source_order" / "budget_1024" / f"{task}.jsonl"
        ref_packs.update({f"{task}/{x['sample_id']}": x for x in read_jsonl(ref_path)})
        new_packs.update({f"{task}/{x['sample_id']}": x for x in read_jsonl(new_path)})

    from transformers import AutoTokenizer
    from tools.bridgepack_g1.generate import load_qwen_fp16
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), trust_remote_code=True)
    model, tokenizer = load_qwen_fp16(str(MODEL), torch)
    eos_ids = {x for x in [tokenizer.eos_token_id, getattr(tokenizer, "pad_token_id", None), 151645, 151643] if x is not None}
    output = []
    for row in new_rows:
        key = f"{row['task']}/{row['sample_id']}"
        old_pack = ref_packs[key]
        new_pack = new_packs[key]
        old_prompt = str(old_pack["generation_prompt"])
        new_prompt = str(new_pack["generation_prompt"])
        old_chat = tokenizer.apply_chat_template([{"role": "user", "content": old_prompt}], tokenize=False, add_generation_prompt=True)
        new_chat = tokenizer.apply_chat_template([{"role": "user", "content": new_prompt}], tokenize=False, add_generation_prompt=True)
        old_retokenized_ids = tokenizer(old_chat, truncation=False, return_tensors="pt").input_ids[0].tolist()
        new_retokenized_ids = tokenizer(new_chat, truncation=False, return_tensors="pt").input_ids[0].tolist()

        device = getattr(model, "device", None) or next(model.parameters()).device
        model_input = tokenizer(new_chat, truncation=False, return_tensors="pt").to(device)
        with torch.no_grad():
            generated = model.generate(
                **model_input,
                max_new_tokens=int(row["max_new_tokens"]),
                num_beams=1,
                do_sample=False,
                use_cache=True,
                eos_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )
        new_full_ids = generated.sequences[0].tolist()
        new_input_ids = new_full_ids[: len(new_retokenized_ids)]
        new_generated_ids = new_full_ids[len(new_retokenized_ids):]
        new_decoded = tokenizer.decode(new_generated_ids, skip_special_tokens=True)
        old_decoded_ids = tokenizer(row["historical_decoded_prediction"], add_special_tokens=False).input_ids
        new_decoded_ids = tokenizer(new_decoded, add_special_tokens=False).input_ids
        first_diff = None
        for i, (a, b) in enumerate(zip(old_decoded_ids, new_decoded_ids)):
            if a != b:
                first_diff = i
                break
        if first_diff is None and len(old_decoded_ids) != len(new_decoded_ids):
            first_diff = min(len(old_decoded_ids), len(new_decoded_ids))

        top = {}
        if first_diff is not None and first_diff < len(generated.scores):
            vals, ids = torch.topk(generated.scores[first_diff][0].float(), k=2)
            top = {
                "new_top1_token_id": int(ids[0]),
                "new_top1_logit": float(vals[0]),
                "new_top2_token_id": int(ids[1]),
                "new_top2_logit": float(vals[1]),
                "new_top1_top2_margin": float(vals[0] - vals[1]),
            }
        new_eos = next((i for i, token_id in enumerate(new_generated_ids) if token_id in eos_ids), None)
        output.append({
            "sample_id": row["sample_id"],
            "task": row["task"],
            "packed_context_same": old_pack.get("packed_text") == new_pack.get("packed_text"),
            "packed_context_sha256": digest(str(new_pack.get("packed_text", ""))),
            "generation_prompt_same": old_prompt == new_prompt,
            "generation_prompt_sha256": digest(new_prompt),
            "new_input_token_ids": new_input_ids,
            "current_retokenized_old_prompt_input_token_ids": old_retokenized_ids,
            "input_ids_same_under_current_tokenizer": new_input_ids == old_retokenized_ids,
            "historical_input_token_ids": "REFERENCE_LAYER_NOT_AVAILABLE",
            "old_decoded_prediction": row["historical_decoded_prediction"],
            "new_decoded_prediction": row["pred"],
            "new_generation_matches_saved_row": new_decoded == row["pred"],
            "first_differing_generated_token": "HISTORICAL_PREDICTION_TOKEN_IDS_UNAVAILABLE",
            "decoded_retokenization_first_diff_position": first_diff,
            "decoded_retokenization_old_token_id": old_decoded_ids[first_diff] if first_diff is not None and first_diff < len(old_decoded_ids) else None,
            "decoded_retokenization_new_token_id": new_decoded_ids[first_diff] if first_diff is not None and first_diff < len(new_decoded_ids) else None,
            "new_generated_token_id_at_diff": new_generated_ids[first_diff] if first_diff is not None and first_diff < len(new_generated_ids) else None,
            "old_logits_top1_top2_margin": "HISTORICAL_LOGITS_UNAVAILABLE",
            **top,
            "old_score_rescored_from_decoded": score_one(row["task"], row["historical_decoded_prediction"], row.get("answers"), row.get("all_classes")),
            "new_score_rescored_from_decoded": score_one(row["task"], row["pred"], row.get("answers"), row.get("all_classes")),
            "old_generation_length": "HISTORICAL_PREDICTION_TOKEN_IDS_UNAVAILABLE",
            "old_decoded_retokenized_length": len(old_decoded_ids),
            "new_generation_length": len(new_generated_ids),
            "old_eos_position": "HISTORICAL_PREDICTION_TOKEN_IDS_UNAVAILABLE",
            "new_eos_position": new_eos,
        })
    write_jsonl(AUDIT / "G1_9_MISMATCH_COMPARISON.jsonl", output)
    fields = [
        "sample_id", "task", "packed_context_same", "generation_prompt_same", "input_ids_same_under_current_tokenizer",
        "historical_input_token_ids", "first_differing_generated_token", "decoded_retokenization_first_diff_position",
        "old_logits_top1_top2_margin", "new_top1_top2_margin", "new_generated_token_id_at_diff", "old_score_rescored_from_decoded",
        "new_score_rescored_from_decoded", "old_generation_length", "old_decoded_retokenized_length",
        "new_generation_length", "old_eos_position", "new_eos_position",
    ]
    with (AUDIT / "G1_9_MISMATCH_COMPARISON.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        w.writeheader(); w.writerows(output)
    all_packed = all(x["packed_context_same"] for x in output)
    all_prompt = all(x["generation_prompt_same"] for x in output)
    all_retokenized_input = all(x["input_ids_same_under_current_tokenizer"] for x in output)
    report = f"""# Nine decoded mismatches: layer comparison

- Rows audited: {len(output)}.
- Same packed context text: `{all_packed}` for all nine.
- Same generation prompt text: `{all_prompt}` for all nine.
- New input IDs equal current-tokenizer re-tokenization of the historical prompt: `{all_retokenized_input}` for all nine.
- Direct comparison with historical input IDs: **not possible**; they were not saved.
- Direct comparison with historical generated token IDs or old logits: **not possible**; they were not saved.
- New-side top-1/top-2 logits and margins are recorded in `G1_9_MISMATCH_COMPARISON.jsonl`.
- Old/new scores are rescored from the two decoded strings, not read from historical per-sample score artifacts.
- Old generation lengths and EOS positions are unavailable as historical token-level facts; decoded-string re-tokenized lengths are labeled as proxies.

The evidence is strongly consistent with divergence after identical *reconstructed/current-tokenizer* model inputs, but the strict statement “historical input token IDs were identical” cannot be certified without the missing historical token-ID artifact. The appropriate cause label remains `GENERATION_ENVIRONMENT_OR_HISTORICAL_CODE_PROVENANCE_UNRESOLVED`.
"""
    (AUDIT / "G1_9_MISMATCH_LAYER_AUDIT.md").write_text(report, encoding="utf-8")
    print(json.dumps({"rows": len(output), "packed_context_all_same": all_packed, "prompt_all_same": all_prompt, "retokenized_input_all_same": all_retokenized_input}))


if __name__ == "__main__":
    main()
