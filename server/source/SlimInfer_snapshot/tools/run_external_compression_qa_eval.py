#!/usr/bin/env python3
"""Run a strictly bounded Qwen QA smoke on exact-budget packed contexts.

The parent process performs a CPU tokenizer preflight over every requested
sample, then launches one method/task cell at a time.  Each cell checks physical
GPU free memory before loading Qwen, uses batch size one, generates a bounded
sample count, scores it, unloads the model by exiting, and returns a cell audit.
No full-context row is accepted.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gc
import hashlib
import json
import math
import os
import subprocess
import sys
import time
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
    "llmlingua_style_exact_query_aware",
    "llmlingua_style_exact_query_agnostic",
)


class QASmokeError(RuntimeError):
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


def prompt_wrapper_hash() -> str:
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


def generation_config_hash(task: str, model_name_or_path: str) -> str:
    return sha256_text(
        canonical_json(
            {
                "model_name_or_path": model_name_or_path,
                "model_revision_or_snapshot": Path(model_name_or_path).name,
                "dtype": "float16",
                "device": "cuda:0",
                "max_new_tokens": max_new_tokens(task),
                "num_beams": 1,
                "do_sample": False,
                "use_cache": True,
                "prompt_wrapper_hash": prompt_wrapper_hash(),
            }
        )
    )


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
                raise QASmokeError(f"invalid JSONL {path}:{line_number}: {exc}") from exc
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


def gpu_rows() -> List[Dict[str, Any]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,memory.free",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except Exception as exc:
        raise QASmokeError(f"nvidia-smi check failed: {type(exc).__name__}: {exc}") from exc
    rows: List[Dict[str, Any]] = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            raise QASmokeError(f"unexpected nvidia-smi row: {line!r}")
        rows.append(
            {
                "physical_gpu_index": int(parts[0]),
                "gpu_name": parts[1],
                "total_mib": int(parts[2]),
                "used_mib": int(parts[3]),
                "free_mib": int(parts[4]),
            }
        )
    if not rows:
        raise QASmokeError("nvidia-smi returned no GPUs")
    return rows


def check_gpu(physical_index: int, min_free_gb: float) -> Dict[str, Any]:
    matches = [
        row for row in gpu_rows() if row["physical_gpu_index"] == physical_index
    ]
    if len(matches) != 1:
        raise QASmokeError(f"physical GPU {physical_index} not uniquely available")
    row = matches[0]
    required_mib = int(min_free_gb * 1024)
    if row["free_mib"] < required_mib:
        raise QASmokeError(
            f"GPU {physical_index} free={row['free_mib']} MiB, "
            f"required={required_mib} MiB"
        )
    return row


def token_count(tokenizer: Any, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False).input_ids)


def build_chat(tokenizer: Any, prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def preflight(
    args: argparse.Namespace,
    tasks: Sequence[str],
    methods: Sequence[str],
    tokenizer: Any,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    reference_ids: Dict[str, List[str]] = {}
    for method in methods:
        for task in tasks:
            path = packed_path(args.packed_context_dir, method, task, args.budget)
            if not path.exists():
                raise QASmokeError(f"packed context file missing: {path}")
            packed = read_jsonl(path)[: args.max_samples_per_task]
            if len(packed) != args.max_samples_per_task:
                raise QASmokeError(
                    f"{method}/{task}: expected {args.max_samples_per_task} rows, got {len(packed)}"
                )
            ids = [str(row.get("sample_id", "")) for row in packed]
            if task not in reference_ids:
                reference_ids[task] = ids
            elif reference_ids[task] != ids:
                raise QASmokeError(f"{task}: sample IDs differ across methods")
            for expected_index, row in enumerate(packed):
                if int(row.get("sample_index", -1)) != expected_index:
                    raise QASmokeError(
                        f"{method}/{task}: sample_index sequence mismatch"
                    )
                packed_text = str(row.get("packed_text", ""))
                if not packed_text.strip():
                    raise QASmokeError(f"{method}/{task}/{ids[expected_index]}: empty context")
                actual = token_count(tokenizer, packed_text)
                stored = int(row.get("packed_token_count", -1))
                if actual != stored:
                    raise QASmokeError(
                        f"{method}/{task}/{ids[expected_index]}: "
                        f"stored tokens={stored}, actual={actual}"
                    )
                if actual > args.budget:
                    raise QASmokeError(
                        f"{method}/{task}/{ids[expected_index]}: over budget {actual}"
                    )
                prompt = str(row.get("generation_prompt", ""))
                if not prompt:
                    raise QASmokeError(
                        f"{method}/{task}/{ids[expected_index]}: missing generation prompt"
                    )
                chat_prompt = build_chat(tokenizer, prompt)
                rows.append(
                    {
                        "method_name": method,
                        "task": task,
                        "sample_id": ids[expected_index],
                        "sample_index": expected_index,
                        "packed_model_tokens": actual,
                        "budget": args.budget,
                        "over_budget": False,
                        "empty_context": False,
                        "generation_prompt_tokens": token_count(tokenizer, prompt),
                        "chat_prompt_tokens": token_count(tokenizer, chat_prompt),
                        "sample_id_aligned": True,
                        "tokenizer_exact_match": True,
                    }
                )
    return rows


def validate_reference_smoke_ids(
    reference_smoke_dir: Path,
    packed_context_dir: Path,
    tasks: Sequence[str],
    methods: Sequence[str],
    budget: int,
) -> Dict[str, Any]:
    if not reference_smoke_dir.exists():
        raise QASmokeError(f"reference smoke directory missing: {reference_smoke_dir}")
    checked = 0
    for method in methods:
        for task in tasks:
            old_path = reference_smoke_dir / "predictions" / method / f"{task}.jsonl"
            if not old_path.exists():
                raise QASmokeError(f"reference smoke prediction missing: {old_path}")
            old_rows = read_jsonl(old_path)
            if len(old_rows) != 2:
                raise QASmokeError(
                    f"reference smoke {method}/{task}: expected 2 rows, got {len(old_rows)}"
                )
            current_rows = read_jsonl(
                packed_path(packed_context_dir, method, task, budget)
            )[:2]
            old_ids = [str(row.get("sample_id", "")) for row in old_rows]
            current_ids = [str(row.get("sample_id", "")) for row in current_rows]
            if old_ids != current_ids:
                raise QASmokeError(
                    f"reference smoke ID mismatch for {method}/{task}: "
                    f"{old_ids} != {current_ids}"
                )
            checked += len(old_ids)
    return {
        "reference_smoke_dir": str(reference_smoke_dir.resolve()),
        "cells_checked": len(tasks) * len(methods),
        "sample_ids_checked": checked,
        "first_two_sample_ids_aligned": True,
    }


def validate_reference_pilot10_ids(
    reference_pilot10_dir: Path,
    packed_context_dir: Path,
    tasks: Sequence[str],
    methods: Sequence[str],
    budget: int,
) -> Dict[str, Any]:
    if not reference_pilot10_dir.exists():
        raise QASmokeError(
            f"reference pilot10 directory missing: {reference_pilot10_dir}"
        )
    checked = 0
    for method in methods:
        for task in tasks:
            old_path = reference_pilot10_dir / "predictions" / method / f"{task}.jsonl"
            if not old_path.exists():
                raise QASmokeError(f"reference pilot10 prediction missing: {old_path}")
            old_rows = read_jsonl(old_path)
            if len(old_rows) != 10:
                raise QASmokeError(
                    f"reference pilot10 {method}/{task}: expected 10 rows, "
                    f"got {len(old_rows)}"
                )
            current_rows = read_jsonl(
                packed_path(packed_context_dir, method, task, budget)
            )[:10]
            old_ids = [str(row.get("sample_id", "")) for row in old_rows]
            current_ids = [str(row.get("sample_id", "")) for row in current_rows]
            if old_ids != current_ids:
                raise QASmokeError(
                    f"reference pilot10 ID mismatch for {method}/{task}"
                )
            checked += len(old_ids)
    return {
        "reference_pilot10_dir": str(reference_pilot10_dir.resolve()),
        "cells_checked": len(tasks) * len(methods),
        "sample_ids_checked": checked,
        "first_ten_sample_ids_aligned": True,
    }


def validate_reference_pilot50_ids(
    reference_pilot50_dir: Path,
    packed_context_dir: Path,
    tasks: Sequence[str],
    methods: Sequence[str],
    budget: int,
) -> Dict[str, Any]:
    if not reference_pilot50_dir.exists():
        raise QASmokeError(
            f"reference pilot50 directory missing: {reference_pilot50_dir}"
        )
    checked = 0
    for method in methods:
        for task in tasks:
            old_path = reference_pilot50_dir / "predictions" / method / f"{task}.jsonl"
            if not old_path.exists():
                raise QASmokeError(f"reference pilot50 prediction missing: {old_path}")
            old_rows = read_jsonl(old_path)
            if len(old_rows) != 50:
                raise QASmokeError(
                    f"reference pilot50 {method}/{task}: expected 50 rows, "
                    f"got {len(old_rows)}"
                )
            current_rows = read_jsonl(
                packed_path(packed_context_dir, method, task, budget)
            )[:50]
            old_ids = [str(row.get("sample_id", "")) for row in old_rows]
            current_ids = [str(row.get("sample_id", "")) for row in current_rows]
            if old_ids != current_ids:
                raise QASmokeError(
                    f"reference pilot50 ID mismatch for {method}/{task}"
                )
            checked += len(old_ids)
    return {
        "reference_pilot50_dir": str(reference_pilot50_dir.resolve()),
        "cells_checked": len(tasks) * len(methods),
        "sample_ids_checked": checked,
        "first_fifty_sample_ids_aligned": True,
    }


def model_path_from_packing(packing_root: Path) -> str:
    config = read_json(packing_root / "config.json")
    tokenizer = config.get("tokenizer", {})
    path = str(tokenizer.get("name_or_path", ""))
    if not path or not Path(path).exists():
        raise QASmokeError(f"local Qwen model/tokenizer path unavailable: {path!r}")
    if "Qwen2.5-7B-Instruct" not in path:
        raise QASmokeError(f"only official Qwen2.5-7B-Instruct path is allowed: {path}")
    return path


def max_new_tokens(task: str) -> int:
    config = read_json(REPO_ROOT / "configs" / "dataset2maxlen.json")
    if task not in config:
        raise QASmokeError(f"official max_new_tokens missing for {task}")
    return int(config[task])


def score_predictions(
    task: str,
    predictions: Sequence[str],
    answers: Sequence[Any],
    lengths: Sequence[int],
    all_classes: Any,
) -> Tuple[float, float]:
    from longbench_eval import scorer

    first = float(scorer(task, list(predictions), list(answers), list(lengths), all_classes))
    second = float(scorer(task, list(predictions), list(answers), list(lengths), all_classes))
    if not math.isfinite(first) or not math.isfinite(second):
        raise QASmokeError(f"{task}: scorer returned NaN/Inf")
    if abs(first - second) > 1e-12:
        raise QASmokeError(f"{task}: scorer recomputation mismatch {first} vs {second}")
    return first, second


def worker(args: argparse.Namespace) -> None:
    if not args.worker_method or not args.worker_task:
        raise QASmokeError("worker method/task required")
    started = time.time()
    cell_path = (
        args.output_dir
        / "cells"
        / args.worker_method
        / f"{args.worker_task}.json"
    )
    cell: Dict[str, Any] = {
        "method_name": args.worker_method,
        "task": args.worker_task,
        "status": "running",
        "oom": False,
        "error": "",
    }
    try:
        gpu_before = check_gpu(args.physical_gpu_index, args.min_free_gpu_gb)
        cell["gpu_before"] = gpu_before
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.physical_gpu_index)
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ.setdefault(
            "PYTORCH_CUDA_ALLOC_CONF",
            "expandable_segments:True,max_split_size_mb:128",
        )
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.manual_seed(args.seed)
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
        except Exception as flash_error:
            cell["flash_attention_load_error"] = (
                f"{type(flash_error).__name__}: {flash_error}"
            )
            load_kwargs["attn_implementation"] = "sdpa"
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path, **load_kwargs
            )
        model.eval()
        torch.cuda.reset_peak_memory_stats()
        source_rows = read_jsonl(
            packed_path(
                args.packed_context_dir,
                args.worker_method,
                args.worker_task,
                args.budget,
            )
        )[: args.max_samples_per_task]
        predictions: List[Dict[str, Any]] = []
        wrapper_hash = prompt_wrapper_hash()
        generation_hash = generation_config_hash(
            args.worker_task, args.model_name_or_path
        )
        scorer_hash = sha256_file(REPO_ROOT / "longbench_eval.py")
        for source in source_rows:
            packed_text = str(source.get("packed_text", ""))
            actual_packed = token_count(tokenizer, packed_text)
            if actual_packed > args.budget:
                raise QASmokeError(
                    f"worker exact budget failure: {actual_packed}>{args.budget}"
                )
            prompt = str(source["generation_prompt"])
            chat = build_chat(tokenizer, prompt)
            model_input = tokenizer(
                chat,
                truncation=False,
                return_tensors="pt",
            ).to("cuda:0")
            context_length = int(model_input.input_ids.shape[-1])
            with torch.inference_mode():
                output = model.generate(
                    **model_input,
                    max_new_tokens=max_new_tokens(args.worker_task),
                    num_beams=1,
                    do_sample=False,
                    use_cache=True,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )[0]
            prediction = tokenizer.decode(
                output[context_length:],
                skip_special_tokens=True,
            ).strip()
            if not prediction:
                raise QASmokeError(
                    f"{args.worker_method}/{args.worker_task}/"
                    f"{source.get('sample_id')}: empty prediction"
                )
            predictions.append(
                {
                    "sample_id": str(source["sample_id"]),
                    "sample_index": int(source["sample_index"]),
                    "task": args.worker_task,
                    "method": args.worker_method,
                    "method_name": args.worker_method,
                    "pred": prediction,
                    "prediction": prediction,
                    "answers": source.get(
                        "answers", source.get("gold_analysis_only_answers", [])
                    ),
                    "gold": source.get(
                        "answers", source.get("gold_analysis_only_answers", [])
                    ),
                    "all_classes": source.get("all_classes"),
                    "length": context_length,
                    "packed_token_count": actual_packed,
                    "packed_model_tokens": actual_packed,
                    "prompt_token_count": token_count(tokenizer, prompt),
                    "chat_prompt_token_count": context_length,
                    "prompt_hash": sha256_text(chat),
                    "prompt_wrapper_hash": wrapper_hash,
                    "generation_config_hash": generation_hash,
                    "scorer_hash": scorer_hash,
                    "model_name": args.model_name_or_path,
                    "model_name_or_path": args.model_name_or_path,
                    "model_revision": Path(args.model_name_or_path).name,
                    "model_revision_or_snapshot": Path(
                        args.model_name_or_path
                    ).name,
                    "max_new_tokens": max_new_tokens(args.worker_task),
                    "budget": args.budget,
                    "over_budget": False,
                    "empty_prediction": False,
                }
            )
        all_classes = next(
            (
                row["all_classes"]
                for row in predictions
                if row.get("all_classes") is not None
            ),
            None,
        )
        score, rescore = score_predictions(
            args.worker_task,
            [row["pred"] for row in predictions],
            [row["answers"] for row in predictions],
            [row["length"] for row in predictions],
            all_classes,
        )
        for row in predictions:
            sample_score, sample_rescore = score_predictions(
                args.worker_task,
                [row["pred"]],
                [row["answers"]],
                [row["length"]],
                row.get("all_classes"),
            )
            if sample_score != sample_rescore:
                raise QASmokeError(
                    f"{args.worker_method}/{args.worker_task}/"
                    f"{row['sample_id']}: per-sample scorer mismatch"
                )
            row["score"] = sample_score
            row["score_rescore"] = sample_rescore
        prediction_path = (
            args.output_dir
            / "predictions"
            / args.worker_method
            / f"{args.worker_task}.jsonl"
        )
        write_jsonl(prediction_path, predictions)
        cell.update(
            {
                "status": "PASS",
                "num_samples": len(predictions),
                "score": score,
                "rescore": rescore,
                "scorer_recomputed": True,
                "sample_ids": [row["sample_id"] for row in predictions],
                "over_budget": 0,
                "empty_predictions": 0,
                "scorer_failure": False,
                "peak_gpu_memory_mib": torch.cuda.max_memory_allocated() / 1024**2,
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
        if "out of memory" in str(exc).lower() or type(exc).__name__ == "OutOfMemoryError":
            cell["oom"] = True
        write_json(cell_path, cell)
        raise
    write_json(cell_path, cell)


def parent(args: argparse.Namespace) -> None:
    tasks = parse_csv(args.tasks)
    methods = parse_csv(args.methods)
    if tuple(tasks) != TASKS:
        raise QASmokeError(f"tasks must exactly equal {TASKS}")
    if tuple(methods) != METHODS:
        raise QASmokeError(f"methods must exactly equal {METHODS}")
    if args.model != "qwen":
        raise QASmokeError("only --model qwen is allowed")
    if args.budget != 1024:
        raise QASmokeError("QA smoke is locked to budget=1024")
    if args.batch_size != 1:
        raise QASmokeError("batch_size must be exactly 1")
    if args.max_samples_per_task not in {1, 2, 10, 50, 100}:
        raise QASmokeError(
            "bounded QA runner allows only 1, 2, 10, 50, or 100 samples per task"
        )
    if args.seed != 0:
        raise QASmokeError("only deterministic seed 0 is supported")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise QASmokeError(f"refusing to overwrite non-empty output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    packing_summary = read_json(args.packed_context_dir / "packing_summary.json")
    if packing_summary.get("status") != "PASS":
        raise QASmokeError("full packing audit is not PASS")
    if int(packing_summary.get("style_over_budget", -1)) != 0:
        raise QASmokeError("full packing audit contains over-budget rows")
    if int(packing_summary.get("style_empty_context", -1)) != 0:
        raise QASmokeError("full packing audit contains empty contexts")
    if int(packing_summary.get("style_successful", 0)) != 2300:
        raise QASmokeError("full packing audit does not contain all 2,300 method-samples")
    model_path = model_path_from_packing(args.packed_context_dir)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    preflight_rows = preflight(args, tasks, methods, tokenizer)
    reference_alignment = validate_reference_smoke_ids(
        args.reference_smoke_dir,
        args.packed_context_dir,
        tasks,
        methods,
        args.budget,
    )
    pilot10_alignment = None
    if args.max_samples_per_task >= 10:
        pilot10_alignment = validate_reference_pilot10_ids(
            args.reference_pilot10_dir,
            args.packed_context_dir,
            tasks,
            methods,
            args.budget,
        )
    pilot50_alignment = None
    if args.max_samples_per_task >= 50:
        pilot50_alignment = validate_reference_pilot50_ids(
            args.reference_pilot50_dir,
            args.packed_context_dir,
            tasks,
            methods,
            args.budget,
        )
    # Fail before any GPU/model activity if the exact repository scorer is not
    # importable in the current entry-point environment.
    from longbench_eval import scorer as _repository_scorer

    if not callable(_repository_scorer):
        raise QASmokeError("repository longbench_eval.scorer is not callable")
    write_csv(
        args.output_dir / "preflight_budget_audit.csv",
        preflight_rows,
        (
            "method_name",
            "task",
            "sample_id",
            "sample_index",
            "packed_model_tokens",
            "budget",
            "over_budget",
            "empty_context",
            "generation_prompt_tokens",
            "chat_prompt_tokens",
            "sample_id_aligned",
            "tokenizer_exact_match",
        ),
    )
    initial_gpus = gpu_rows()
    eligible = [
        row for row in initial_gpus if row["free_mib"] >= int(args.min_free_gpu_gb * 1024)
    ]
    if not eligible:
        stopped_payload = {
            "status": "STOPPED_LOW_GPU_MEMORY",
            "gpu_rows": initial_gpus,
            "required_free_gb": args.min_free_gpu_gb,
            "model_loaded": False,
        }
        write_json(
            args.output_dir / "qa_smoke_summary.json",
            stopped_payload,
        )
        write_json(
            args.output_dir / "STOPPED_LOW_GPU_MEMORY.json",
            stopped_payload,
        )
        raise QASmokeError("no GPU meets minimum free-memory requirement")
    physical_gpu = max(eligible, key=lambda row: row["free_mib"])[
        "physical_gpu_index"
    ]
    cell_rows: List[Dict[str, Any]] = []
    for method in methods:
        for task in tasks:
            try:
                gpu_before = check_gpu(physical_gpu, args.min_free_gpu_gb)
            except Exception as exc:
                stopped_payload = {
                    "status": "STOPPED_LOW_GPU_MEMORY",
                    "stopped_before_cell": f"{method}/{task}",
                    "error": f"{type(exc).__name__}: {exc}",
                    "completed_cells": cell_rows,
                    "completed_predictions": sum(
                        int(row.get("num_samples", 0)) for row in cell_rows
                    ),
                    "required_free_gpu_gb": args.min_free_gpu_gb,
                    "no_retry": True,
                }
                write_json(
                    args.output_dir / "qa_smoke_summary.json",
                    stopped_payload,
                )
                write_json(
                    args.output_dir / "STOPPED_LOW_GPU_MEMORY.json",
                    stopped_payload,
                )
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
                str(physical_gpu),
                "--output_dir",
                str(args.output_dir),
                "--packed_context_dir",
                str(args.packed_context_dir),
                "--max_samples_per_task",
                str(args.max_samples_per_task),
                "--budget",
                str(args.budget),
                "--batch_size",
                "1",
                "--min_free_gpu_gb",
                str(args.min_free_gpu_gb),
                "--seed",
                str(args.seed),
                "--model_name_or_path",
                model_path,
            ]
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
            environment["HF_HUB_OFFLINE"] = "1"
            environment["TRANSFORMERS_OFFLINE"] = "1"
            environment.setdefault(
                "PYTORCH_CUDA_ALLOC_CONF",
                "expandable_segments:True,max_split_size_mb:128",
            )
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
                "error": f"worker exited {result.returncode} without cell audit",
            }
            cell["parent_gpu_before"] = gpu_before
            cell_rows.append(cell)
            if result.returncode != 0 or cell.get("status") != "PASS":
                write_json(
                    args.output_dir / "qa_smoke_summary.json",
                    {
                        "status": "FAIL",
                        "stopped_after_cell": f"{method}/{task}",
                        "cells": cell_rows,
                        "no_retry": True,
                    },
                )
                raise QASmokeError(
                    f"cell failed and smoke stopped: {method}/{task}: {cell.get('error')}"
                )
            # The worker exits after every cell, releasing all model memory.
            time.sleep(1.0)

    score_rows = [
        {
            "method_name": row["method_name"],
            "task": row["task"],
            "num_samples": row["num_samples"],
            "score": row["score"],
            "rescore": row["rescore"],
            "scorer_recomputed": row["scorer_recomputed"],
            "gpu_free_mib_before": row["parent_gpu_before"]["free_mib"],
            "peak_gpu_memory_mib": row["peak_gpu_memory_mib"],
            "over_budget": row["over_budget"],
            "empty_predictions": row["empty_predictions"],
            "scorer_failure": row["scorer_failure"],
            "oom": row["oom"],
            "status": row["status"],
        }
        for row in cell_rows
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
            "created_at": dt.datetime.now().astimezone().isoformat(),
            "packed_context_dir": str(args.packed_context_dir.resolve()),
            "model_name_or_path": model_path,
            "tasks": tasks,
            "methods": methods,
            "samples_per_task": args.max_samples_per_task,
            "total_predictions": sum(row["num_samples"] for row in cell_rows),
            "batch_size": 1,
            "serial_cells": True,
            "physical_gpu_index": physical_gpu,
            "min_free_gpu_gb": args.min_free_gpu_gb,
            "preflight_rows": len(preflight_rows),
            "preflight_over_budget": 0,
            "preflight_empty_context": 0,
            "sample_id_alignment": True,
            "reference_smoke_alignment": reference_alignment,
            "reference_pilot10_alignment": pilot10_alignment,
            "reference_pilot50_alignment": pilot50_alignment,
            "tokenizer_exact_match": True,
            "over_budget": sum(row["over_budget"] for row in cell_rows),
            "empty_predictions": sum(row["empty_predictions"] for row in cell_rows),
            "scorer_failures": sum(bool(row["scorer_failure"]) for row in cell_rows),
            "oom": any(bool(row["oom"]) for row in cell_rows),
            "scorer_recomputed_all": all(
                bool(row["scorer_recomputed"]) for row in cell_rows
            ),
            "cells": cell_rows,
            "full_qa_run_performed": False,
            "full_context_qa_performed": False,
        },
    )


def parse_args() -> argparse.Namespace:
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, default=REPO_ROOT / "results" / f"bridgepack_external_compression_qa_smoke_{stamp}")
    parser.add_argument("--packed_context_dir", type=Path, required=True)
    parser.add_argument(
        "--reference_smoke_dir",
        type=Path,
        default=(
            REPO_ROOT
            / "results"
            / "bridgepack_external_compression_qa_smoke_retry_20260706_002105"
        ),
    )
    parser.add_argument(
        "--reference_pilot10_dir",
        type=Path,
        default=(
            REPO_ROOT
            / "results"
            / "bridgepack_external_compression_qa_pilot10_retry_20260706_004441"
        ),
    )
    parser.add_argument(
        "--reference_pilot50_dir",
        type=Path,
        default=(
            REPO_ROOT
            / "results"
            / "bridgepack_external_compression_qa_pilot50_20260706_005645"
        ),
    )
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument("--max_samples_per_task", type=int, default=2)
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--model", choices=("qwen",), default="qwen")
    parser.add_argument("--budget", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--min_free_gpu_gb", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker_method", help=argparse.SUPPRESS)
    parser.add_argument("--worker_task", help=argparse.SUPPRESS)
    parser.add_argument("--physical_gpu_index", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--model_name_or_path", default="", help=argparse.SUPPRESS)
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
