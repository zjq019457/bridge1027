#!/usr/bin/env python3
"""Deferred Qwen32 generation diagnostic for evidence text packing.

This script is prepared for later GPU use. Importing it, running --help, or
using its scoring helpers does not load model weights. The heavy torch /
transformers imports are inside _load_model().
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bridgekv.evidence_text_packing import estimate_token_count  # noqa: E402

DEFAULT_TASKS = "hotpotqa,musique,2wikimqa"
DEFAULT_VARIANTS = (
    "span_only_source_order,neighbor32_source_order,"
    "bridge_neighbor32_path_order_tags"
)


def _parse_list(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _latest_packed_dir() -> Path:
    candidates = sorted(
        (REPO_ROOT / "results").glob("evidence_text_packing_offline_*"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("No results/evidence_text_packing_offline_* directory found")
    return candidates[0]


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(fields),
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _load_packed_rows(
    packed_dir: Path,
    variant: str,
    task: str,
    budget: int,
    max_samples: int,
) -> List[Dict[str, Any]]:
    path = packed_dir / "packed_prompts" / variant / f"budget_{budget}" / f"{task}.jsonl"
    rows = _read_jsonl(path)
    if max_samples > 0:
        rows = rows[:max_samples]
    return rows


def _build_chat(tokenizer: Any, prompt: str) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt


def _generation_prompt_from_row(task: str, row: Mapping[str, Any]) -> str:
    prompt = row.get("generation_prompt")
    if isinstance(prompt, str) and prompt.strip():
        return prompt
    packed_text = str(row.get("packed_text") or row.get("packed_prompt") or "")
    templates = _load_json(REPO_ROOT / "configs" / "dataset2prompt.json")
    template = str(templates.get(task, ""))
    if template:
        try:
            return template.format(
                **{
                    "context": packed_text,
                    "input": row.get("question", ""),
                    "question": row.get("question", ""),
                }
            )
        except Exception:
            pass
    return packed_text


def _normalize_dtype(dtype: str) -> str:
    aliases = {
        "float16": "fp16",
        "torch.float16": "fp16",
        "float32": "fp32",
        "torch.float32": "fp32",
        "bfloat16": "bf16",
        "torch.bfloat16": "bf16",
    }
    return aliases.get(dtype, dtype)


def _load_model(model_name_or_path: str, dtype: str, device: str) -> tuple[Any, Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. This generation runner is intentionally "
            "GPU-only; do not run it while CUDA_VISIBLE_DEVICES is empty."
        )
    dtype = _normalize_dtype(dtype)
    torch_dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[dtype]
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    load_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch_dtype,
        "device_map": "auto" if device == "auto" else {"": device},
        "attn_implementation": "flash_attention_2",
    }
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **load_kwargs)
    except Exception as exc:
        print(f"Warning: flash_attention_2 load failed: {exc}. Retrying with sdpa.", flush=True)
        load_kwargs["attn_implementation"] = "sdpa"
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **load_kwargs)
    if getattr(model.config, "pad_token_id", None) is None and tokenizer.eos_token_id is not None:
        model.config.pad_token_id = tokenizer.eos_token_id
    model.eval()
    return model, tokenizer, torch


def _wait_for_gpu_free_memory(min_free_mb: int, poll_seconds: int) -> None:
    if min_free_mb <= 0:
        return
    while True:
        try:
            proc = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                check=False,
            )
            if proc.returncode != 0 or not proc.stdout.strip():
                return
            used_s, total_s = proc.stdout.strip().splitlines()[0].split(",")[:2]
            used = int(used_s.strip())
            total = int(total_s.strip())
            free = total - used
            if free >= min_free_mb:
                return
            print(
                f"Waiting for GPU memory before Qwen load: free={free}MB, "
                f"required={min_free_mb}MB, used={used}MB/{total}MB",
                flush=True,
            )
        except Exception:
            return
        time.sleep(max(5, poll_seconds))


def _generate_one(
    model: Any,
    tokenizer: Any,
    torch: Any,
    prompt: str,
    max_new_tokens: int,
) -> tuple[str, int]:
    chat_prompt = _build_chat(tokenizer, prompt)
    model_device = getattr(model, "device", None)
    if model_device is None:
        model_device = next(model.parameters()).device
    model_input = tokenizer(chat_prompt, truncation=False, return_tensors="pt").to(model_device)
    context_length = int(model_input.input_ids.shape[1])
    if hasattr(model, "model") and hasattr(model, "lm_head"):
        return _generate_one_last_token_logits(model, tokenizer, torch, model_input, context_length, max_new_tokens)
    with torch.no_grad():
        output = model.generate(
            **model_input,
            max_new_tokens=max_new_tokens,
            num_beams=1,
            do_sample=False,
            use_cache=True,
            eos_token_id=tokenizer.eos_token_id,
        )[0]
    pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
    return pred, context_length


def _generate_one_last_token_logits(
    model: Any,
    tokenizer: Any,
    torch: Any,
    model_input: Any,
    context_length: int,
    max_new_tokens: int,
) -> tuple[str, int]:
    """Greedy decode while projecting only the final hidden state.

    Some local Transformers versions materialize full-sequence logits during
    `generate()`. For long full-context prompts, that can allocate several extra
    GiB even though greedy decoding only needs the final token logits.
    """
    input_ids = model_input.input_ids
    attention_mask = getattr(model_input, "attention_mask", None)
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)
    generated: list[int] = []
    past_key_values = None
    eos_token_id = tokenizer.eos_token_id
    with torch.inference_mode():
        for step in range(max(1, int(max_new_tokens))):
            if step == 0:
                current_ids = input_ids
            else:
                current_ids = torch.tensor([[generated[-1]]], dtype=input_ids.dtype, device=input_ids.device)
            outputs = model.model(
                input_ids=current_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = outputs.past_key_values
            last_hidden = outputs.last_hidden_state[:, -1:, :]
            logits = model.lm_head(last_hidden)
            next_token = int(torch.argmax(logits[:, -1, :], dim=-1).item())
            if eos_token_id is not None and next_token == int(eos_token_id):
                break
            generated.append(next_token)
            next_mask = torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=attention_mask.device)
            attention_mask = torch.cat([attention_mask, next_mask], dim=-1)
    pred = tokenizer.decode(generated, skip_special_tokens=True)
    return pred, context_length


def _existing_prediction_count(path: Path) -> int:
    return len(_read_jsonl(path))


def _max_new_tokens_for(task: str, override: int) -> int:
    if override > 0:
        return override
    config = REPO_ROOT / "configs" / "dataset2maxlen.json"
    if config.exists():
        payload = json.loads(config.read_text(encoding="utf-8"))
        if task in payload:
            return int(payload[task])
    return 32


def run_generation(args: argparse.Namespace, out_dir: Path) -> None:
    packed_dir = Path(args.packed_dir) if args.packed_dir else _latest_packed_dir()
    tasks = _parse_list(args.tasks)
    variants = _parse_list(args.variants)
    _wait_for_gpu_free_memory(args.min_free_gpu_mb, args.gpu_wait_poll_seconds)
    model, tokenizer, torch = _load_model(args.model_name_or_path, args.dtype, args.device)
    for variant in variants:
        for task in tasks:
            rows = _load_packed_rows(
                packed_dir,
                variant,
                task,
                args.budget,
                args.max_samples,
            )
            if not rows:
                raise FileNotFoundError(
                    f"No packed prompt rows for variant={variant} task={task} budget={args.budget} "
                    f"under {packed_dir}"
                )
            pred_path = out_dir / "predictions" / variant / f"{task}.jsonl"
            pred_path.parent.mkdir(parents=True, exist_ok=True)
            start = _existing_prediction_count(pred_path) if args.resume else 0
            mode = "a" if start else "w"
            with pred_path.open(mode, encoding="utf-8") as handle:
                for sample_index, row in enumerate(rows[start:], start=start):
                    prompt = _generation_prompt_from_row(task, row)
                    if not prompt:
                        raise ValueError(f"Missing prompt for {variant}/{task} row {sample_index}")
                    pred, context_length = _generate_one(
                        model,
                        tokenizer,
                        torch,
                        prompt,
                        _max_new_tokens_for(task, args.max_new_tokens),
                    )
                    payload = {
                        "sample_id": row.get("sample_id", sample_index),
                        "sample_index": sample_index,
                        "task": task,
                        "variant": variant,
                        "pred": pred,
                        "answers": row.get("answers", row.get("gold_analysis_only_answers", [])),
                        "all_classes": row.get("all_classes", None),
                        "length": context_length,
                        "packed_token_count": row.get("packed_token_count", estimate_token_count(prompt)),
                        "prompt_token_count": estimate_token_count(prompt),
                        "budget": args.budget,
                        "question": row.get("question", ""),
                        "selected_span_ids": row.get("selected_span_ids", []),
                        "selected_page_ids": row.get("selected_page_ids", []),
                    }
                    json.dump(payload, handle, ensure_ascii=False)
                    handle.write("\n")
                    handle.flush()


def _score_predictions(out_dir: Path, tasks: Sequence[str], variants: Sequence[str]) -> List[Dict[str, Any]]:
    from longbench_eval import scorer

    rows: List[Dict[str, Any]] = []
    for variant in variants:
        for task in tasks:
            pred_path = out_dir / "predictions" / variant / f"{task}.jsonl"
            predictions: List[str] = []
            answers: List[Any] = []
            lengths: List[int] = []
            all_classes: Any = None
            for row in _read_jsonl(pred_path):
                predictions.append(str(row.get("pred", "")))
                answers.append(row.get("answers", []))
                lengths.append(int(row.get("length", row.get("packed_token_count", 0)) or 0))
                if row.get("all_classes") is not None:
                    all_classes = row.get("all_classes")
            status = "ok" if predictions else "missing"
            score = ""
            error = ""
            if predictions:
                try:
                    score = scorer(task, predictions, answers, lengths, all_classes)
                except Exception as exc:
                    status = "failed"
                    error = repr(exc)
            score_dir = out_dir / "scores" / variant
            score_dir.mkdir(parents=True, exist_ok=True)
            (score_dir / f"{task}.json").write_text(
                json.dumps({
                    "variant": variant,
                    "task": task,
                    "score": score,
                    "num_samples": len(predictions),
                    "status": status,
                    "error": error,
                }, indent=2),
                encoding="utf-8",
            )
            rows.append({
                "variant": variant,
                "task": task,
                "score": score,
                "num_samples": len(predictions),
                "status": status,
                "error": error,
            })
    return rows


def _prompt_budget_stats(
    packed_dir: Path,
    out_dir: Path,
    tasks: Sequence[str],
    variants: Sequence[str],
    budget: int,
    max_samples: int,
) -> None:
    rows: List[Dict[str, Any]] = []
    for variant in variants:
        for task in tasks:
            for row in _load_packed_rows(packed_dir, variant, task, budget, max_samples):
                text = row.get("packed_text") or row.get("packed_prompt") or ""
                token_count = estimate_token_count(text)
                rows.append({
                    "variant": variant,
                    "task": task,
                    "sample_id": row.get("sample_id", ""),
                    "budget": budget,
                    "packed_token_count": token_count,
                    "stored_packed_token_count": row.get("packed_token_count", ""),
                    "budget_ok": token_count <= budget or bool(row.get("truncated", False)),
                    "selected_span_count": len(row.get("selected_span_ids", [])),
                    "selected_page_count": len(row.get("selected_page_ids", [])),
                    "used_bridge_path": row.get("used_bridge_path", ""),
                    "used_tags": row.get("used_tags", ""),
                    "truncated": row.get("truncated", ""),
                })
    _write_csv(
        out_dir / "audits" / "01_prompt_budget_stats.csv",
        rows,
        [
            "variant", "task", "sample_id", "budget", "packed_token_count",
            "stored_packed_token_count", "budget_ok", "selected_span_count",
            "selected_page_count", "used_bridge_path", "used_tags", "truncated",
        ],
    )


def _write_summaries(out_dir: Path, score_rows: List[Dict[str, Any]], tasks: Sequence[str], variants: Sequence[str]) -> None:
    score_by_variant_task = {
        (row["variant"], row["task"]): row for row in score_rows
    }
    summary_rows: List[Dict[str, Any]] = []
    for variant in variants:
        values = [
            float(score_by_variant_task[(variant, task)]["score"])
            for task in tasks
            if score_by_variant_task.get((variant, task), {}).get("status") == "ok"
            and score_by_variant_task[(variant, task)]["score"] not in ("", None)
        ]
        macro = round(statistics.mean(values), 4) if values else ""
        summary_rows.append({
            "variant": variant,
            "macro_avg": macro,
            "tasks_completed": len(values),
            "budget": "",
            "delta_vs_span_only": "",
        })
    span_macro = next((row["macro_avg"] for row in summary_rows if row["variant"] == "span_only_source_order"), "")
    for row in summary_rows:
        if span_macro != "" and row["macro_avg"] != "":
            row["delta_vs_span_only"] = round(float(row["macro_avg"]) - float(span_macro), 4)

    breakdown_rows = [
        {
            "variant": row["variant"],
            "task": row["task"],
            "score": row["score"],
            "num_samples": row["num_samples"],
            "status": row["status"],
            "error": row["error"],
        }
        for row in score_rows
    ]
    delta_rows: List[Dict[str, Any]] = []
    for variant in variants:
        for task in ["_macro", *tasks]:
            if task == "_macro":
                score = next((r["macro_avg"] for r in summary_rows if r["variant"] == variant), "")
                base = span_macro
            else:
                score = score_by_variant_task.get((variant, task), {}).get("score", "")
                base = score_by_variant_task.get(("span_only_source_order", task), {}).get("score", "")
            delta = ""
            if score not in ("", None) and base not in ("", None):
                delta = round(float(score) - float(base), 4)
            delta_rows.append({
                "variant": variant,
                "task": task,
                "score": score,
                "span_only_score": base,
                "delta_vs_span_only": delta,
            })

    _write_csv(out_dir / "summary" / "summary.csv", summary_rows, [
        "variant", "macro_avg", "tasks_completed", "budget", "delta_vs_span_only",
    ])
    _write_csv(out_dir / "summary" / "method_task_breakdown.csv", breakdown_rows, [
        "variant", "task", "score", "num_samples", "status", "error",
    ])
    _write_csv(out_dir / "summary" / "delta_vs_span_only.csv", delta_rows, [
        "variant", "task", "score", "span_only_score", "delta_vs_span_only",
    ])
    _write_csv(out_dir / "audits" / "02_variant_score_comparison.csv", delta_rows, [
        "variant", "task", "score", "span_only_score", "delta_vs_span_only",
    ])


def _write_musique_examples(out_dir: Path, variants: Sequence[str]) -> None:
    task = "musique"
    predictions = {
        variant: _read_jsonl(out_dir / "predictions" / variant / f"{task}.jsonl")
        for variant in variants
    }
    by_variant = {
        variant: {str(row.get("sample_id")): row for row in rows}
        for variant, rows in predictions.items()
    }
    sample_ids = sorted(set().union(*(set(rows) for rows in by_variant.values())))[:5]
    lines = ["# MuSiQue Case Examples", ""]
    for sample_id in sample_ids:
        base = next((rows.get(sample_id) for rows in by_variant.values() if rows.get(sample_id)), {})
        lines.extend([
            f"## sample_id={sample_id}",
            "",
            f"Question: {base.get('question', '')}",
            "",
            f"Answers: `{base.get('answers', [])}`",
            "",
        ])
        for variant in variants:
            row = by_variant.get(variant, {}).get(sample_id, {})
            lines.extend([
                f"### {variant}",
                "",
                f"- pred: `{str(row.get('pred', ''))[:300]}`",
                f"- selected_span_ids: `{','.join(str(x) for x in row.get('selected_span_ids', [])[:12])}`",
                f"- selected_page_ids: `{','.join(str(x) for x in row.get('selected_page_ids', [])[:12])}`",
                "",
            ])
    (out_dir / "audits" / "03_musique_case_examples.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def _write_recommendation(out_dir: Path, tasks: Sequence[str]) -> None:
    summary_path = out_dir / "summary" / "delta_vs_span_only.csv"
    rows = _read_csv(summary_path)

    def delta(variant: str, task: str) -> float | None:
        for row in rows:
            if row.get("variant") == variant and row.get("task") == task:
                value = row.get("delta_vs_span_only", "")
                if value in ("", None):
                    return None
                return float(value)
        return None

    bridge = "bridge_neighbor32_path_order_tags"
    neigh = "neighbor32_source_order"
    bridge_macro = delta(bridge, "_macro")
    neigh_macro = delta(neigh, "_macro")
    bridge_improves = bridge_macro is not None and bridge_macro > 0
    neigh_helps = neigh_macro is not None and neigh_macro > 0
    bridge_musique = delta(bridge, "musique")
    bridge_2wiki = delta(bridge, "2wikimqa")
    multihop_improves = any(
        value is not None and value > 0 for value in (bridge_musique, bridge_2wiki)
    )
    prompt_rows = _read_csv(out_dir / "audits" / "01_prompt_budget_stats.csv")
    budget_ok = bool(prompt_rows) and all(str(row.get("budget_ok")) == "True" for row in prompt_rows)
    justify_scale = bool(
        bridge_improves and multihop_improves and budget_ok and bridge_macro >= 0.3
    )
    lines = [
        "# Evidence Text Packing Generation Recommendation",
        "",
        "This report is generated after predictions are available.",
        "",
        "## Required Answers",
        "",
        f"1. Bridge-aware packing improves over `span_only_source_order`: **{bridge_improves}** "
        f"(macro delta={bridge_macro}).",
        f"2. `neighbor32_source_order` helps compared with span-only: **{neigh_helps}** "
        f"(macro delta={neigh_macro}).",
        f"3. Bridge-aware packing improves musique or 2wikimqa: **{multihop_improves}** "
        f"(musique delta={bridge_musique}, 2wikimqa delta={bridge_2wiki}).",
        f"4. Budget 1024 is sufficient: **{budget_ok}**.",
        f"5. Larger 64-sample or budget scaling run justified: **{justify_scale}**.",
        "",
        "Do not treat this as a full LongBench result; it is a 32-sample diagnostic gate.",
        "",
    ]
    (out_dir / "audits" / "04_recommendation.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_pending_artifacts(out_dir: Path, tasks: Sequence[str], variants: Sequence[str]) -> None:
    rows = [
        {
            "variant": variant,
            "task": task,
            "score": "",
            "num_samples": 0,
            "status": "pending_generation",
            "error": "generation not run",
        }
        for variant in variants
        for task in tasks
    ]
    _write_summaries(out_dir, rows, tasks, variants)
    (out_dir / "audits" / "03_musique_case_examples.md").write_text(
        "# MuSiQue Case Examples\n\nPending generation.\n",
        encoding="utf-8",
    )
    (out_dir / "audits" / "04_recommendation.md").write_text(
        "# Evidence Text Packing Generation Recommendation\n\nPending generation.\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packed_dir", default="")
    parser.add_argument("--tasks", default=DEFAULT_TASKS)
    parser.add_argument("--max_samples", type=int, default=32)
    parser.add_argument("--budget", type=int, default=1024)
    parser.add_argument("--variants", default=DEFAULT_VARIANTS)
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--out_dir", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32", "float16", "bfloat16", "float32"], default="fp16")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max_new_tokens", type=int, default=0)
    parser.add_argument("--score_only", action="store_true")
    parser.add_argument("--prepare_only", action="store_true")
    parser.add_argument("--min_free_gpu_mb", type=int, default=18000)
    parser.add_argument("--gpu_wait_poll_seconds", type=int, default=60)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    packed_dir = Path(args.packed_dir) if args.packed_dir else _latest_packed_dir()
    tasks = _parse_list(args.tasks)
    variants = _parse_list(args.variants)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / "results" / f"evidence_text_packing_generation_qwen32_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "packed_dir": str(packed_dir),
        "tasks": tasks,
        "max_samples": args.max_samples,
        "budget": args.budget,
        "variants": variants,
        "model_name_or_path": args.model_name_or_path,
        "dtype": args.dtype,
        "device": args.device,
        "max_new_tokens": args.max_new_tokens,
        "min_free_gpu_mb": args.min_free_gpu_mb,
        "gpu_wait_poll_seconds": args.gpu_wait_poll_seconds,
        "created_at": stamp,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    _prompt_budget_stats(packed_dir, out_dir, tasks, variants, args.budget, args.max_samples)

    if args.prepare_only:
        write_pending_artifacts(out_dir, tasks, variants)
        print(f"PREPARED_OUT_DIR={out_dir}")
        return

    started = time.time()
    if not args.score_only:
        run_generation(args, out_dir)
    score_rows = _score_predictions(out_dir, tasks, variants)
    _write_summaries(out_dir, score_rows, tasks, variants)
    _write_musique_examples(out_dir, variants)
    _write_recommendation(out_dir, tasks)
    status = {
        "status": "complete",
        "elapsed_s": round(time.time() - started, 2),
        "out_dir": str(out_dir),
    }
    (out_dir / "run_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(f"OUT_DIR={out_dir}")


if __name__ == "__main__":
    main()
