#!/usr/bin/env python3
"""Run a fail-closed Qwen QA smoke on eligible official LLMLingua audits."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
TASKS = (
    "qasper",
    "multifieldqa_en",
    "hotpotqa",
    "2wikimqa",
    "musique",
    "narrativeqa",
)
METHODS = (
    "official_rate_search",
    "official_plus_exact_truncation",
)
DEFAULT_MODEL = Path(
    "/home/yu-bai/.cache/huggingface/hub/"
    "models--Qwen--Qwen2.5-7B-Instruct/snapshots/"
    "a09a35458c702b33eeacc393d103063234e8bc28"
)


class OfficialQASmokeError(RuntimeError):
    pass


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_csv(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise OfficialQASmokeError(
                    f"invalid JSONL {path}:{line_number}: {exc}"
                ) from exc
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def write_csv(
    path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]
) -> None:
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


def packed_path(root: Path, method: str, task: str, budget: int) -> Path:
    return (
        root
        / "packed_prompts_build"
        / "packed_prompts"
        / method
        / f"budget_{budget}"
        / f"{task}.jsonl"
    )


def token_count(tokenizer: Any, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False).input_ids)


def build_chat(tokenizer: Any, prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def max_new_tokens(task: str) -> int:
    values = read_json(REPO_ROOT / "configs" / "dataset2maxlen.json")
    if task not in values:
        raise OfficialQASmokeError(f"missing task generation length: {task}")
    return int(values[task])


def wrapper_hash() -> str:
    return sha256_text(
        canonical_json(
            {
                "dataset2prompt_sha256": sha256_file(
                    REPO_ROOT / "configs" / "dataset2prompt.json"
                ),
                "chat_wrapper": (
                    "tokenizer.apply_chat_template(user, "
                    "add_generation_prompt=True)"
                ),
            }
        )
    )


def generation_hash(task: str, model: str) -> str:
    return sha256_text(
        canonical_json(
            {
                "model": model,
                "revision": Path(model).name,
                "dtype": "float16",
                "max_new_tokens": max_new_tokens(task),
                "num_beams": 1,
                "do_sample": False,
                "use_cache": True,
                "wrapper_hash": wrapper_hash(),
            }
        )
    )


def gpu_rows() -> List[Dict[str, Any]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    rows = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        rows.append(
            {
                "physical_gpu_index": int(parts[0]),
                "gpu_name": parts[1],
                "total_mib": int(parts[2]),
                "used_mib": int(parts[3]),
                "free_mib": int(parts[4]),
            }
        )
    return rows


def check_gpu(index: int, minimum_gb: float) -> Dict[str, Any]:
    matches = [row for row in gpu_rows() if row["physical_gpu_index"] == index]
    if len(matches) != 1:
        raise OfficialQASmokeError(f"GPU {index} unavailable")
    row = matches[0]
    required = int(minimum_gb * 1024)
    if row["free_mib"] < required:
        raise OfficialQASmokeError(
            f"GPU {index} free={row['free_mib']} MiB, required={required} MiB"
        )
    return row


def score_twice(
    task: str,
    predictions: Sequence[str],
    answers: Sequence[Any],
    lengths: Sequence[int],
    all_classes: Any,
) -> Tuple[float, float]:
    from longbench_eval import scorer

    first = float(scorer(task, list(predictions), list(answers), list(lengths), all_classes))
    second = float(scorer(task, list(predictions), list(answers), list(lengths), all_classes))
    if not math.isfinite(first) or first != second:
        raise OfficialQASmokeError(
            f"{task}: scorer non-finite or mismatch {first} vs {second}"
        )
    return first, second


def preflight(args: argparse.Namespace, tokenizer: Any) -> List[Dict[str, Any]]:
    summary = read_json(args.packed_context_dir / "official_compression_summary.json")
    rows: List[Dict[str, Any]] = []
    ids_by_task: Dict[str, List[str]] = {}
    for method in METHODS:
        status = summary["protocol_status"].get(method, {})
        if not status.get("qa_eligible"):
            raise OfficialQASmokeError(f"{method}: compression audit not QA eligible")
        for task in TASKS:
            source = read_jsonl(
                packed_path(args.packed_context_dir, method, task, args.budget)
            )[: args.max_samples_per_task]
            if len(source) != args.max_samples_per_task:
                raise OfficialQASmokeError(
                    f"{method}/{task}: expected {args.max_samples_per_task} rows, "
                    f"got {len(source)}"
                )
            ids = [str(row["sample_id"]) for row in source]
            if task not in ids_by_task:
                ids_by_task[task] = ids
            elif ids_by_task[task] != ids:
                raise OfficialQASmokeError(f"{task}: sample IDs differ across methods")
            for expected_index, row in enumerate(source):
                if int(row["sample_index"]) != expected_index:
                    raise OfficialQASmokeError(
                        f"{method}/{task}: sample index mismatch"
                    )
                packed = str(row["packed_text"])
                actual = token_count(tokenizer, packed)
                if actual != int(row["packed_token_count"]) or actual > args.budget:
                    raise OfficialQASmokeError(
                        f"{method}/{task}/{row['sample_id']}: budget mismatch"
                    )
                if not packed.strip() or not str(row["generation_prompt"]).strip():
                    raise OfficialQASmokeError(
                        f"{method}/{task}/{row['sample_id']}: empty prompt"
                    )
                rows.append(
                    {
                        "method_name": method,
                        "task": task,
                        "sample_id": row["sample_id"],
                        "sample_index": expected_index,
                        "packed_qwen_tokens": actual,
                        "budget": args.budget,
                        "over_budget": False,
                    }
                )
    return rows


def worker(args: argparse.Namespace) -> None:
    cell_path = (
        args.output_dir / "cells" / args.worker_method / f"{args.worker_task}.json"
    )
    started = time.time()
    cell: Dict[str, Any] = {
        "method_name": args.worker_method,
        "task": args.worker_task,
        "status": "running",
        "oom": False,
        "error": "",
    }
    try:
        gpu = check_gpu(args.physical_gpu_index, args.min_free_gpu_gb)
        cell["gpu_before"] = gpu
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.physical_gpu_index)
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        load_kwargs = {
            "trust_remote_code": True,
            "local_files_only": True,
            "dtype": torch.float16,
            "device_map": {"": "cuda:0"},
            "attn_implementation": "flash_attention_2",
        }
        try:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path, **load_kwargs
            )
        except Exception as exc:
            cell["flash_attention_load_error"] = f"{type(exc).__name__}: {exc}"
            load_kwargs["attn_implementation"] = "sdpa"
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path, **load_kwargs
            )
        model.eval()
        torch.cuda.reset_peak_memory_stats()
        source = read_jsonl(
            packed_path(
                args.packed_context_dir,
                args.worker_method,
                args.worker_task,
                args.budget,
            )
        )[: args.max_samples_per_task]
        outputs: List[Dict[str, Any]] = []
        for row in source:
            packed = str(row["packed_text"])
            packed_tokens = token_count(tokenizer, packed)
            if packed_tokens > args.budget:
                raise OfficialQASmokeError("worker received over-budget input")
            prompt = str(row["generation_prompt"])
            chat = build_chat(tokenizer, prompt)
            inputs = tokenizer(chat, truncation=False, return_tensors="pt").to("cuda:0")
            context_length = int(inputs.input_ids.shape[-1])
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens(args.worker_task),
                    num_beams=1,
                    do_sample=False,
                    use_cache=True,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )[0]
            prediction = tokenizer.decode(
                generated[context_length:], skip_special_tokens=True
            ).strip()
            if not prediction:
                raise OfficialQASmokeError(
                    f"{args.worker_method}/{args.worker_task}/"
                    f"{row['sample_id']}: empty prediction"
                )
            outputs.append(
                {
                    "method": args.worker_method,
                    "method_name": args.worker_method,
                    "task": args.worker_task,
                    "sample_id": row["sample_id"],
                    "sample_index": row["sample_index"],
                    "prediction": prediction,
                    "pred": prediction,
                    "gold": row.get(
                        "answers", row.get("gold_analysis_only_answers", [])
                    ),
                    "answers": row.get(
                        "answers", row.get("gold_analysis_only_answers", [])
                    ),
                    "all_classes": row.get("all_classes"),
                    "length": context_length,
                    "compressed_qwen_tokens": packed_tokens,
                    "packed_model_tokens": packed_tokens,
                    "over_budget": False,
                    "empty_prediction": False,
                    "model_name": args.model_name_or_path,
                    "model_revision": Path(args.model_name_or_path).name,
                    "generation_config_hash": generation_hash(
                        args.worker_task, args.model_name_or_path
                    ),
                    "prompt_wrapper_hash": wrapper_hash(),
                    "prompt_hash": sha256_text(chat),
                    "scorer_hash": sha256_file(REPO_ROOT / "longbench_eval.py"),
                    "official_llmlingua": True,
                    "official_unmodified": (
                        args.worker_method == "official_rate_search"
                    ),
                    "post_trimmed": (
                        args.worker_method
                        == "official_plus_exact_truncation"
                    ),
                }
            )
        all_classes = next(
            (row["all_classes"] for row in outputs if row["all_classes"] is not None),
            None,
        )
        score, rescore = score_twice(
            args.worker_task,
            [row["prediction"] for row in outputs],
            [row["gold"] for row in outputs],
            [row["length"] for row in outputs],
            all_classes,
        )
        for row in outputs:
            one, one_again = score_twice(
                args.worker_task,
                [row["prediction"]],
                [row["gold"]],
                [row["length"]],
                row.get("all_classes"),
            )
            row["score"] = one
            row["score_rescore"] = one_again
        write_jsonl(
            args.output_dir
            / "predictions"
            / args.worker_method
            / f"{args.worker_task}.jsonl",
            outputs,
        )
        cell.update(
            {
                "status": "PASS",
                "num_samples": len(outputs),
                "score": score,
                "rescore": rescore,
                "scorer_recomputed": True,
                "sample_ids": [row["sample_id"] for row in outputs],
                "over_budget": 0,
                "empty_predictions": 0,
                "scorer_failure": False,
                "peak_gpu_memory_mib": (
                    torch.cuda.max_memory_allocated() / 1024**2
                ),
                "elapsed_seconds": time.time() - started,
            }
        )
        del model
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as exc:
        cell.update(
            {
                "status": "FAIL",
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_seconds": time.time() - started,
            }
        )
        if "out of memory" in str(exc).lower():
            cell["oom"] = True
        write_json(cell_path, cell)
        raise
    write_json(cell_path, cell)


def parent(args: argparse.Namespace) -> None:
    if tuple(parse_csv(args.tasks)) != TASKS:
        raise OfficialQASmokeError(f"tasks must exactly match {TASKS}")
    if tuple(parse_csv(args.methods)) != METHODS:
        raise OfficialQASmokeError(f"methods must exactly match {METHODS}")
    if args.max_samples_per_task != 2 or args.batch_size != 1:
        raise OfficialQASmokeError("official QA is restricted to 2 samples/task, batch=1")
    if args.budget != 1024 or args.seed != 0:
        raise OfficialQASmokeError("official QA requires budget=1024 and seed=0")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise OfficialQASmokeError(f"refusing to overwrite: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    from transformers import AutoTokenizer
    from longbench_eval import scorer as repository_scorer

    if not callable(repository_scorer):
        raise OfficialQASmokeError("repository scorer unavailable")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    preflight_rows = preflight(args, tokenizer)
    write_csv(
        args.output_dir / "preflight_budget_audit.csv",
        preflight_rows,
        tuple(preflight_rows[0]),
    )
    eligible = [
        row
        for row in gpu_rows()
        if row["free_mib"] >= int(args.min_free_gpu_gb * 1024)
    ]
    if not eligible:
        payload = {
            "status": "STOPPED_LOW_GPU_MEMORY",
            "gpu_rows": gpu_rows(),
            "required_free_gpu_gb": args.min_free_gpu_gb,
        }
        write_json(args.output_dir / "STOPPED_LOW_GPU_MEMORY.json", payload)
        raise OfficialQASmokeError("no eligible GPU")
    gpu_index = max(eligible, key=lambda row: row["free_mib"])[
        "physical_gpu_index"
    ]
    cells: List[Dict[str, Any]] = []
    for method in METHODS:
        for task in TASKS:
            try:
                gpu_before = check_gpu(gpu_index, args.min_free_gpu_gb)
            except Exception as exc:
                payload = {
                    "status": "STOPPED_LOW_GPU_MEMORY",
                    "stopped_before_cell": f"{method}/{task}",
                    "error": f"{type(exc).__name__}: {exc}",
                    "completed_cells": cells,
                    "no_retry": True,
                }
                write_json(args.output_dir / "STOPPED_LOW_GPU_MEMORY.json", payload)
                raise
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--worker_method",
                method,
                "--worker_task",
                task,
                "--physical_gpu_index",
                str(gpu_index),
                "--output_dir",
                str(args.output_dir),
                "--packed_context_dir",
                str(args.packed_context_dir),
                "--model_name_or_path",
                args.model_name_or_path,
                "--max_samples_per_task",
                "2",
                "--budget",
                "1024",
                "--batch_size",
                "1",
                "--min_free_gpu_gb",
                str(args.min_free_gpu_gb),
                "--seed",
                "0",
            ]
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
            environment["HF_HUB_OFFLINE"] = "1"
            environment["TRANSFORMERS_OFFLINE"] = "1"
            result = subprocess.run(
                command,
                cwd=REPO_ROOT,
                env=environment,
                text=True,
                check=False,
            )
            cell_path = args.output_dir / "cells" / method / f"{task}.json"
            cell = read_json(cell_path) if cell_path.exists() else {
                "method_name": method,
                "task": task,
                "status": "FAIL",
                "error": f"worker exited {result.returncode} without audit",
            }
            cell["parent_gpu_before"] = gpu_before
            cells.append(cell)
            if result.returncode != 0 or cell["status"] != "PASS":
                write_json(
                    args.output_dir / "qa_smoke_summary.json",
                    {
                        "status": "FAIL",
                        "stopped_after_cell": f"{method}/{task}",
                        "cells": cells,
                        "no_retry": True,
                    },
                )
                raise OfficialQASmokeError(f"cell failed: {method}/{task}")
            time.sleep(1)
    score_rows = [
        {
            "method_name": cell["method_name"],
            "task": cell["task"],
            "num_samples": cell["num_samples"],
            "score": cell["score"],
            "rescore": cell["rescore"],
            "scorer_recomputed": cell["scorer_recomputed"],
            "gpu_free_mib_before": cell["parent_gpu_before"]["free_mib"],
            "peak_gpu_memory_mib": cell["peak_gpu_memory_mib"],
            "over_budget": cell["over_budget"],
            "empty_predictions": cell["empty_predictions"],
            "scorer_failure": cell["scorer_failure"],
            "oom": cell["oom"],
            "status": cell["status"],
        }
        for cell in cells
    ]
    write_csv(
        args.output_dir / "method_task_scores.csv",
        score_rows,
        tuple(score_rows[0]),
    )
    write_json(
        args.output_dir / "qa_smoke_summary.json",
        {
            "status": "PASS",
            "created_at": datetime.now().astimezone().isoformat(),
            "packed_context_dir": str(args.packed_context_dir.resolve()),
            "methods": list(METHODS),
            "tasks": list(TASKS),
            "samples_per_task": 2,
            "total_predictions": 24,
            "over_budget": 0,
            "empty_predictions": 0,
            "scorer_failures": 0,
            "oom": False,
            "scorer_recomputed_all": True,
            "official_unmodified_qa_run": False,
            "official_rate_search_qa_run": True,
            "official_plus_exact_truncation_qa_run": True,
            "full_qa_run_performed": False,
            "cells": cells,
        },
    )


def parse_args() -> argparse.Namespace:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=(
            REPO_ROOT
            / "results"
            / f"bridgepack_official_llmlingua_qa_smoke_{stamp}"
        ),
    )
    parser.add_argument("--packed_context_dir", type=Path, required=True)
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--model_name_or_path", default=str(DEFAULT_MODEL))
    parser.add_argument("--max_samples_per_task", type=int, default=2)
    parser.add_argument("--budget", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--min_free_gpu_gb", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker_method", help=argparse.SUPPRESS)
    parser.add_argument("--worker_task", help=argparse.SUPPRESS)
    parser.add_argument("--physical_gpu_index", type=int, default=0, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.worker:
        worker(args)
    else:
        parent(args)
    print(str(args.output_dir.resolve()))


if __name__ == "__main__":
    main()
