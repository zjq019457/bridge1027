#!/usr/bin/env python3
"""Prepare or run a full-context LongBench QA upper-bound evaluation.

This wrapper builds a `packed_prompts` tree whose single variant is the original
uncompressed context. By default it only prepares inputs and audits exact
tokenizer lengths. Passing `--run_generation` delegates to the existing
BridgePack generation/scoring runner.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.run_bridgepack_exact_budget_repack import (  # noqa: E402
    _candidate_dataset_roots,
    _generation_prompt,
    _load_samples,
    _load_tokenizer,
    _token_count,
)

DEFAULT_TASKS = "qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,narrativeqa"
VARIANT = "full_context_original"


def _parse_list(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _reference_sample_ids(reference_run_dir: Path, tasks: Sequence[str], max_samples: int) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for task in tasks:
        path = (
            reference_run_dir / "packed_prompts_build" / "packed_prompts" /
            "span_only_source_order" / "budget_1024" / f"{task}.jsonl"
        )
        ids = [str(row.get("sample_id", "")) for row in _read_jsonl(path) if row.get("sample_id")]
        out[task] = ids[:max_samples] if max_samples > 0 else ids
    return out


def build_inputs(args: argparse.Namespace) -> tuple[Path, list[dict[str, Any]]]:
    out_dir = Path(args.out_dir)
    tasks = _parse_list(args.tasks)
    tokenizer = _load_tokenizer(args.model_name_or_path)
    roots = _candidate_dataset_roots(args.longbench_root)
    ref_ids = _reference_sample_ids(Path(args.reference_run_dir), tasks, args.max_samples)
    packed_root = out_dir / "packed_prompts_build"
    audit_rows: list[dict[str, Any]] = []
    for task in tasks:
        samples = _load_samples(task, roots, 0)
        by_id = {str(sample["sample_id"]): sample for sample in samples}
        selected_ids = ref_ids.get(task) or list(by_id)
        if args.max_samples > 0:
            selected_ids = selected_ids[: args.max_samples]
        path = packed_root / "packed_prompts" / VARIANT / f"budget_{args.budget}" / f"{task}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for sample_id in selected_ids:
                sample = by_id.get(sample_id)
                if not sample:
                    audit_rows.append({
                        "task": task,
                        "sample_id": sample_id,
                        "status": "missing_in_dataset",
                    })
                    continue
                context = str(sample.get("context", ""))
                question = str(sample.get("question", ""))
                prompt = _generation_prompt(task, context, question)
                context_tokens = _token_count(tokenizer, context)
                prompt_tokens = _token_count(tokenizer, prompt)
                exceeds = prompt_tokens > int(args.model_max_length)
                if exceeds and args.truncate_strategy == "head_tail":
                    ids = tokenizer(prompt, add_special_tokens=False).input_ids
                    half = int(args.model_max_length) // 2
                    prompt = tokenizer.decode(ids[:half], skip_special_tokens=True) + tokenizer.decode(
                        ids[-half:], skip_special_tokens=True
                    )
                    prompt_tokens = _token_count(tokenizer, prompt)
                    context = prompt
                payload = {
                    "sample_id": sample_id,
                    "task": task,
                    "variant": VARIANT,
                    "budget": args.budget,
                    "packed_text": context,
                    "packed_prompt": context,
                    "generation_prompt": prompt,
                    "question": question,
                    "answers": sample.get("answers", []),
                    "gold_analysis_only_answers": sample.get("answers", []),
                    "all_classes": sample.get("all_classes", None),
                    "length": sample.get("length", 0),
                    "source_file": sample.get("source_file", ""),
                    "packed_token_count": context_tokens,
                    "model_token_count": context_tokens,
                    "prompt_token_count": prompt_tokens,
                    "exact_model_token_budget": True,
                    "model_tokenizer": args.model_name_or_path,
                    "selected_span_ids": [],
                    "selected_page_ids": [],
                    "truncated": exceeds and args.truncate_strategy == "head_tail",
                    "full_context_original": True,
                }
                json.dump(payload, handle, ensure_ascii=False)
                handle.write("\n")
                audit_rows.append({
                    "task": task,
                    "sample_id": sample_id,
                    "status": "ok",
                    "original_context_tokens": context_tokens,
                    "full_prompt_tokens": prompt_tokens,
                    "model_max_length": args.model_max_length,
                    "exceeds_model_max_length": exceeds,
                    "truncate_strategy": args.truncate_strategy,
                    "handled": "head_tail_truncated" if payload["truncated"] else "reported_not_truncated",
                })
    return packed_root, audit_rows


def run_generation(args: argparse.Namespace, packed_root: Path) -> None:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "run_evidence_text_packing_generation_qwen32.py"),
        "--packed_dir", str(packed_root),
        "--tasks", args.tasks,
        "--max_samples", str(args.max_samples),
        "--budget", str(args.budget),
        "--variants", VARIANT,
        "--model_name_or_path", args.model_name_or_path,
        "--out_dir", args.out_dir,
        "--dtype", args.dtype,
        "--device", args.device,
        "--resume",
    ]
    log_path = Path(args.out_dir) / "logs" / "full_context_generation.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n")
        proc = subprocess.run(cmd, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, text=True, check=False)
        log.write(f"returncode={proc.returncode}\n")
    if proc.returncode != 0:
        raise RuntimeError(f"Generation failed; see {log_path}")


def parse_args() -> argparse.Namespace:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--tasks", default=DEFAULT_TASKS)
    parser.add_argument("--max_samples", type=int, default=2)
    parser.add_argument("--budget", type=int, default=131072)
    parser.add_argument("--model_max_length", type=int, default=131072)
    parser.add_argument("--reference_run_dir", default="results/bridgepack_exact_qwen_budget_1024_20260619_181059")
    parser.add_argument("--longbench_root", default="")
    parser.add_argument("--out_dir", default=f"results/bridgepack_full_context_qwen_{stamp}")
    parser.add_argument("--truncate_strategy", choices=["none", "head_tail"], default="none")
    parser.add_argument("--run_generation", action="store_true")
    parser.add_argument("--dtype", default="fp16")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    packed_root, audit_rows = build_inputs(args)
    fields = [
        "task", "sample_id", "status", "original_context_tokens", "full_prompt_tokens",
        "model_max_length", "exceeds_model_max_length", "truncate_strategy", "handled",
    ]
    _write_csv(out_dir / "full_context_input_audit.csv", audit_rows, fields)
    token_values = [int(row.get("full_prompt_tokens", 0) or 0) for row in audit_rows if row.get("status") == "ok"]
    over = [row for row in audit_rows if str(row.get("exceeds_model_max_length")) == "True"]
    _write_text(out_dir / "full_context_input_report.md", "\n".join([
        "# Full-Context Input Audit",
        "",
        "- This script prepares original-context inputs; it does not compress text.",
        f"- Rows: {len(audit_rows)}.",
        f"- Prompt token mean: {statistics.mean(token_values):.2f}." if token_values else "- Prompt token mean: TODO.",
        f"- Prompt token max: {max(token_values) if token_values else 0}.",
        f"- Over model max length: {len(over)}.",
        f"- Truncate strategy: `{args.truncate_strategy}`.",
        f"- Generation executed: **{bool(args.run_generation)}**.",
    ]))
    if args.run_generation:
        if over and args.truncate_strategy == "none":
            raise RuntimeError("Over-length full-context prompts exist; use --truncate_strategy head_tail or inspect audit.")
        run_generation(args, packed_root)
    print(f"OUT_DIR={out_dir}")


if __name__ == "__main__":
    main()
