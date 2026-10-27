#!/usr/bin/env python3
"""Fresh G1 1150-sample decoded-prediction reproduction.

The historical token-ID criterion is intentionally not used as a gate because
the G1 reference did not persist token IDs. This runner uses the recovered G1
method/protocol unchanged, compares decoded predictions to historical raw
predictions, and saves token IDs for the new run's future internal replay.

It never writes under the historical G1 reference directory.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
REF = REPO / "results" / "bridgepack_g1_qwen1024_matched_20260727_004934"
DATA_ROOT = REPO.parent / "HeadKV" / "data" / "LongBench"
TASKS = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", "narrativeqa"]
BUDGET = 1024
VARIANT = "same_spans_source_order"
MODEL_PATH = Path("/home/yu-bai/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28")


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
            f.flush()


def write_csv(path: Path, fields: List[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        for row in rows:
            w.writerow({field: row.get(field, "") for field in fields})


def dataset_rows(task: str) -> Dict[str, Dict[str, Any]]:
    return {str(x["_id"]): x for x in read_jsonl(DATA_ROOT / f"{task}.jsonl")}


def reference_rows(task: str) -> List[Dict[str, Any]]:
    return read_jsonl(REF / "full" / "packed_prompts_build" / "packed_prompts" / VARIANT / f"budget_{BUDGET}" / f"{task}.jsonl")


def reference_predictions(task: str) -> Dict[str, Dict[str, Any]]:
    return {str(x["sample_id"]): x for x in read_jsonl(REF / "full" / "predictions" / VARIANT / f"{task}.jsonl")}


def load_inputs() -> tuple[Dict[str, List[str]], Dict[str, List[Dict[str, Any]]], Dict[str, Dict[str, Any]]]:
    ids: Dict[str, List[str]] = {}
    samples: Dict[str, List[Dict[str, Any]]] = {}
    reference_pred: Dict[str, Dict[str, Any]] = {}
    for task in TASKS:
        packs = reference_rows(task)
        data = dataset_rows(task)
        ids[task] = [str(x["sample_id"]) for x in packs]
        samples[task] = []
        reference_pred.update({f"{task}/{sid}": row for sid, row in reference_predictions(task).items()})
        for packed in packs:
            sid = str(packed["sample_id"])
            d = data[sid]
            samples[task].append({
                "task": task,
                "sample_id": sid,
                "question": d["input"],
                "context": d["context"],
                "original_context": d["context"],
                "answers": d.get("answers", []),
                "all_classes": d.get("all_classes"),
                "length": d.get("length", 0),
            })
    return ids, samples, reference_pred


def compare_packs(run_dir: Path, ids: Mapping[str, List[str]]) -> Dict[str, Any]:
    rows = []
    all_exact = True
    for task in TASKS:
        ref = {str(x["sample_id"]): x for x in reference_rows(task)}
        new_path = run_dir / "packing" / "packed_prompts_build" / "packed_prompts" / VARIANT / f"budget_{BUDGET}" / f"{task}.jsonl"
        new = {str(x["sample_id"]): x for x in read_jsonl(new_path)}
        for sid in ids[task]:
            r, n = ref[sid], new.get(sid, {})
            checks = {
                "sample_id_exact": sid == str(n.get("sample_id")),
                "question_exact": str(r.get("question", "")) == str(n.get("question", "")),
                "context_exact": str(r.get("original_context", "")) == str(n.get("original_context", "")),
                "selected_span_ids_exact": list(r.get("selected_span_ids") or []) == list(n.get("selected_span_ids") or []),
                "repaired_ranges_exact": (r.get("repaired_ranges") or []) == (n.get("repaired_ranges") or []),
                "packed_text_exact": str(r.get("packed_text", "")) == str(n.get("packed_text", "")),
                "packed_token_count_exact": int(r.get("model_token_count", r.get("packed_token_count", -1))) == int(n.get("model_token_count", n.get("packed_token_count", -2))),
                "within_budget": int(n.get("model_token_count", n.get("packed_token_count", 10**9))) <= BUDGET,
            }
            exact = all(checks.values())
            all_exact &= exact
            rows.append({"task": task, "sample_id": sid, **checks, "status": "PASS" if exact else "FAIL"})
    fields = list(rows[0].keys())
    write_csv(run_dir / "PACKING_COMPARISON.csv", fields, rows)
    return {"all_exact": all_exact, "rows": len(rows), "failures": sum(x["status"] == "FAIL" for x in rows)}


def model_input_and_generate(model: Any, tokenizer: Any, torch: Any, prompt: str, max_new_tokens: int) -> tuple[str, List[int], List[int], int, str]:
    if hasattr(tokenizer, "apply_chat_template"):
        chat_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
    else:
        chat_prompt = prompt
    device = getattr(model, "device", None) or next(model.parameters()).device
    model_input = tokenizer(chat_prompt, truncation=False, return_tensors="pt").to(device)
    input_ids = model_input.input_ids[0].tolist()
    with torch.no_grad():
        output = model.generate(
            **model_input,
            max_new_tokens=max_new_tokens,
            num_beams=1,
            do_sample=False,
            use_cache=True,
            eos_token_id=tokenizer.eos_token_id,
        )[0]
    output_ids = output.tolist()
    prediction_ids = output_ids[len(input_ids):]
    prediction = tokenizer.decode(prediction_ids, skip_special_tokens=True)
    return prediction, input_ids, prediction_ids, len(input_ids), sha_text(chat_prompt)


def run_qa(run_dir: Path, ids: Mapping[str, List[str]], reference_pred: Mapping[str, Dict[str, Any]], model: Any, tokenizer: Any, torch: Any) -> List[Dict[str, Any]]:
    import tools.run_evidence_text_packing_generation_qwen32 as gen

    rows: List[Dict[str, Any]] = []
    for task in TASKS:
        path = run_dir / "packing" / "packed_prompts_build" / "packed_prompts" / VARIANT / f"budget_{BUDGET}" / f"{task}.jsonl"
        packs = {str(x["sample_id"]): x for x in read_jsonl(path)}
        out_path = run_dir / "predictions" / VARIANT / f"{task}.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as out:
            for index, sid in enumerate(ids[task]):
                row = packs[sid]
                prompt = str(row["generation_prompt"])
                max_new = gen._max_new_tokens_for(task, 0)
                pred, input_ids, prediction_ids, input_count, chat_sha = model_input_and_generate(model, tokenizer, torch, prompt, max_new)
                hist = reference_pred[f"{task}/{sid}"]
                payload = {
                    "sample_id": sid,
                    "sample_index": index,
                    "task": task,
                    "variant": VARIANT,
                    "pred": pred,
                    "answers": row.get("answers", []),
                    "all_classes": row.get("all_classes"),
                    "length": input_count,
                    "packed_token_count": int(row.get("model_token_count", row.get("packed_token_count", -1))),
                    "prompt_token_count": input_count,
                    "budget": BUDGET,
                    "question": row.get("question", ""),
                    "selected_span_ids": row.get("selected_span_ids", []),
                    "selected_page_ids": row.get("selected_page_ids", []),
                    "input_token_ids": input_ids,
                    "prediction_token_ids": prediction_ids,
                    "chat_prompt_sha256": chat_sha,
                    "max_new_tokens": max_new,
                    "historical_decoded_prediction": hist.get("pred", ""),
                    "decoded_prediction_exact": pred == str(hist.get("pred", "")),
                    "historical_token_ids": "REFERENCE_LAYER_NOT_AVAILABLE",
                }
                out.write(json.dumps(payload, ensure_ascii=False) + "\n")
                out.flush()
                rows.append(payload)
                if len(rows) % 25 == 0:
                    write_json(run_dir / "progress.json", {"timestamp": utc_now(), "completed": len(rows), "total": 1150, "task": task, "sample_id": sid})
                    print(f"[decoded-repro] {len(rows)}/1150 {task}/{sid}", flush=True)
    write_jsonl(run_dir / "raw_predictions.jsonl", rows)
    return rows


def score_rows(run_dir: Path, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    from longbench_eval import scorer
    output = []
    task_scores: Dict[str, float] = {}
    for task in TASKS:
        task_rows = [x for x in rows if x["task"] == task]
        score = float(scorer(task, [x["pred"] for x in task_rows], [x.get("answers", []) for x in task_rows], [x["length"] for x in task_rows], task_rows[0].get("all_classes") if task_rows else None))
        task_scores[task] = score
        output.append({"task": task, "sample_count": len(task_rows), "score": score})
    macro = sum(task_scores.values()) / len(TASKS)
    output.append({"task": "UNWEIGHTED_MACRO", "sample_count": len(rows), "score": macro})
    write_csv(run_dir / "per_task_scores.csv", ["task", "sample_count", "score"], output)
    ref_scores = {"qasper": 41.76, "multifieldqa_en": 49.83, "hotpotqa": 53.91, "2wikimqa": 45.98, "musique": 28.05, "narrativeqa": 16.78, "UNWEIGHTED_MACRO": 39.385}
    comparison = [{"task": x["task"], "new_score": x["score"], "historical_score": ref_scores[x["task"]], "delta": x["score"] - ref_scores[x["task"]], "exact": x["score"] == ref_scores[x["task"]]} for x in output]
    write_csv(run_dir / "G1_1150_DECODED_SCORE_COMPARISON.csv", ["task", "new_score", "historical_score", "delta", "exact"], comparison)
    summary = {
        "decoded_exact_count": sum(bool(x.get("decoded_prediction_exact")) for x in rows),
        "sample_count": len(rows),
        "all_decoded_exact": all(bool(x.get("decoded_prediction_exact")) for x in rows),
        "task_scores": task_scores,
        "macro": macro,
        "historical_macro": 39.385,
        "macro_exact": macro == 39.385,
        "historical_token_ids_comparable": False,
        "status": "G1_END_TO_END_DECODED_PREDICTION_EXACT_REPRO_PASS" if all(bool(x.get("decoded_prediction_exact")) for x in rows) and macro == 39.385 else "G1_END_TO_END_DECODED_PREDICTION_REPRO_FAIL",
    }
    write_json(run_dir / "macro_summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--packing-only", action="store_true")
    parser.add_argument("--resume-packing", action="store_true", help="Reuse an existing all-exact CPU packing gate in run-dir")
    args = parser.parse_args()
    run_dir = Path(args.run_dir).resolve()
    if run_dir.exists() and any(run_dir.iterdir()) and not args.resume_packing:
        raise SystemExit(f"Refusing non-empty run directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    ids, samples, reference_pred = load_inputs()
    if args.resume_packing:
        gate = json.loads((run_dir / "packing_gate.json").read_text(encoding="utf-8"))
        if not gate.get("all_exact") or int(gate.get("rows", 0)) != 1150:
            raise SystemExit("--resume-packing requires an existing all-exact 1150 packing_gate.json")
    write_json(run_dir / "run_manifest.json", {
        "created_at": utc_now(), "reference_root": str(REF), "variant": VARIANT, "budget": BUDGET,
        "task_counts": {t: len(ids[t]) for t in TASKS}, "sample_count": sum(len(v) for v in ids.values()),
        "model_path": str(MODEL_PATH), "historical_token_ids_comparable": False,
        "selection_method_unchanged": True, "question_binding": "historical G1 prompt retained for decoded comparison",
        "resumed_from_exact_packing": args.resume_packing,
    })
    write_json(run_dir / "source_hashes.json", {name: sha_file(REPO / name) for name in [
        "bridgekv/evidence_text_packing.py", "tools/run_bridgepack_exact_budget_repack.py",
        "tools/bridgepack_g1/run_g1.py", "tools/bridgepack_g1/common.py", "tools/bridgepack_g1/generate.py",
        "configs/dataset2prompt.json", "configs/dataset2maxlen.json", "longbench_eval.py",
    ]})
    (run_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")

    if args.resume_packing:
        pack_result = json.loads((run_dir / "packing_gate.json").read_text(encoding="utf-8"))
    else:
        from transformers import AutoTokenizer
        from tools.bridgepack_g1.run_g1 import build_matched_variant_packs
        tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), trust_remote_code=True, use_fast=True)
        build_matched_variant_packs(run_dir, ids, samples, tokenizer, stage="packing", variant_filter=[VARIANT])
        pack_result = compare_packs(run_dir, ids)
        write_json(run_dir / "packing_gate.json", pack_result)
        if not pack_result["all_exact"]:
            write_json(run_dir / "macro_summary.json", {"status": "PACKING_OR_CODE_DIVERGENCE", **pack_result})
            return 3
    # Keep a unified trace artifact separate from historical files.
    trace_rows = []
    for task in TASKS:
        trace_rows.extend(read_jsonl(run_dir / "packing" / "packed_prompts_build" / "packed_prompts" / VARIANT / f"budget_{BUDGET}" / f"{task}.jsonl"))
    write_jsonl(run_dir / "packing_trace.jsonl", trace_rows)
    if args.packing_only:
        write_json(run_dir / "macro_summary.json", {"status": "PACKING_EXACT_GPU_NOT_RUN", **pack_result})
        return 0

    import torch
    from tools.bridgepack_g1.generate import load_qwen_fp16
    model, tokenizer = load_qwen_fp16(str(MODEL_PATH), torch)
    environment = {
        "timestamp": utc_now(), "torch": torch.__version__, "cuda": torch.version.cuda,
        "model_path": str(MODEL_PATH), "dtype": "float16", "do_sample": False, "num_beams": 1,
        "tokenizer_class": tokenizer.__class__.__name__, "tokenizer_json_sha256": sha_file(MODEL_PATH / "tokenizer.json"),
    }
    write_json(run_dir / "environment.json", environment)
    rows = run_qa(run_dir, ids, reference_pred, model, tokenizer, torch)
    summary = score_rows(run_dir, rows)
    write_json(run_dir / "budget_audit.json", {
        "budget": BUDGET, "sample_count": len(rows),
        "packed_over_budget": sum(int(x["packed_token_count"]) > BUDGET for x in rows),
        "max_packed_token_count": max(int(x["packed_token_count"]) for x in rows),
        "final_prompt_over_budget": sum(int(x["prompt_token_count"]) > BUDGET for x in rows),
        "max_final_prompt_token_count": max(int(x["prompt_token_count"]) for x in rows),
    })
    write_json(run_dir / "run_manifest.json", {**json.loads((run_dir / "run_manifest.json").read_text()), "completed_at": utc_now(), "status": summary["status"]})
    return 0 if summary["status"] == "G1_END_TO_END_DECODED_PREDICTION_EXACT_REPRO_PASS" else 4


if __name__ == "__main__":
    raise SystemExit(main())
