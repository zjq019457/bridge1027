#!/usr/bin/env python3
"""Audit official LongLLMLingua outputs under an exact Qwen-token budget.

This runner imports Microsoft's unmodified PromptCompressor implementation and
uses a fully local causal-LM checkpoint.  It never loads the Qwen QA model and
never uses a GPU.  Three output protocols are kept distinct:

* official_unmodified: one official call, no Qwen-token post-processing;
* official_rate_search: official target_token calls only, no text trimming;
* official_plus_exact_truncation: the unmodified output followed by an
  explicitly disclosed Qwen-token prefix truncation.
"""

from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import hashlib
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
LLMLINGUA_ROOT = REPO_ROOT / "LLMLingua"
REFERENCE_DIR = (
    REPO_ROOT / "results" / "bridgepack_exact_qwen_budget_1024_20260619_181059"
)
REFERENCE_VARIANT = "bridge_neighbor32_path_order"
TASKS = (
    "qasper",
    "multifieldqa_en",
    "hotpotqa",
    "2wikimqa",
    "musique",
    "narrativeqa",
)
EXPECTED_COUNTS = {
    "qasper": 200,
    "multifieldqa_en": 150,
    "hotpotqa": 200,
    "2wikimqa": 200,
    "musique": 200,
    "narrativeqa": 200,
}
PROTOCOLS = (
    "official_unmodified",
    "official_rate_search",
    "official_plus_exact_truncation",
)
DEFAULT_QWEN = Path(
    "/home/yu-bai/.cache/huggingface/hub/"
    "models--Qwen--Qwen2.5-7B-Instruct/snapshots/"
    "a09a35458c702b33eeacc393d103063234e8bc28"
)
DEFAULT_COMPRESSOR = Path(
    "/home/yu-bai/.cache/huggingface/hub/"
    "models--facebook--opt-125m/snapshots/"
    "27dcfa74d334bc871f3234de431e71c6eeba5dd6"
)


class OfficialAuditError(RuntimeError):
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
                raise OfficialAuditError(
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


def token_count(tokenizer: Any, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False).input_ids)


def exact_qwen_prefix(tokenizer: Any, text: str, budget: int) -> str:
    ids = tokenizer(text, add_special_tokens=False).input_ids[:budget]
    while ids:
        decoded = tokenizer.decode(
            ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
        if token_count(tokenizer, decoded) <= budget:
            return decoded
        ids = ids[:-1]
    return ""


def format_packed(question: str, official_output: str) -> str:
    return (
        f"Question: {question}\n\nSelected compressed evidence:\n\n"
        f"{official_output.strip()}"
    ).strip()


def generation_prompt(task: str, packed_text: str, question: str) -> str:
    templates = read_json(REPO_ROOT / "configs" / "dataset2prompt.json")
    if task not in templates:
        raise OfficialAuditError(f"missing generation template for {task}")
    return str(templates[task]).format(
        context=packed_text,
        input=question,
        question=question,
    )


def percentile_nearest(values: Sequence[int], percentile: float) -> Optional[int]:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1))
    return ordered[index]


def reference_rows(
    reference_dir: Path, task: str, max_samples: int
) -> List[Dict[str, Any]]:
    path = (
        reference_dir
        / "packed_prompts_build"
        / "packed_prompts"
        / REFERENCE_VARIANT
        / "budget_1024"
        / f"{task}.jsonl"
    )
    if not path.exists():
        raise OfficialAuditError(f"reference manifest missing: {path}")
    rows = read_jsonl(path)
    if len(rows) != EXPECTED_COUNTS[task]:
        raise OfficialAuditError(
            f"{task}: reference rows={len(rows)}, expected={EXPECTED_COUNTS[task]}"
        )
    if len({str(row.get("sample_id", "")) for row in rows}) != len(rows):
        raise OfficialAuditError(f"{task}: missing or duplicate reference sample IDs")
    for row in rows:
        if not str(row.get("original_context", "")).strip():
            raise OfficialAuditError(f"{task}: reference row lacks original_context")
        if not str(row.get("question", "")).strip():
            raise OfficialAuditError(f"{task}: reference row lacks question")
    return rows[:max_samples] if max_samples > 0 else rows


def split_official_context(
    compressor: Any, context: str, chunk_tokens: int
) -> Tuple[List[str], int, int]:
    ids = compressor.tokenizer(context, add_special_tokens=False).input_ids
    chunks = [
        compressor.tokenizer.decode(
            ids[start : start + chunk_tokens],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        for start in range(0, len(ids), chunk_tokens)
    ]
    chunks = [chunk for chunk in chunks if chunk.strip()]
    maximum = max(
        (
            len(compressor.tokenizer(chunk, add_special_tokens=False).input_ids)
            for chunk in chunks
        ),
        default=0,
    )
    return chunks, len(ids), maximum


def official_call(
    compressor: Any,
    chunks: Sequence[str],
    question: str,
    target_token: int,
    iterative_size: int,
) -> Tuple[Dict[str, Any], float]:
    started = time.perf_counter()
    response = compressor.compress_prompt(
        list(chunks),
        question=question,
        target_token=target_token,
        iterative_size=iterative_size,
        use_sentence_level_filter=False,
        use_context_level_filter=True,
        use_token_level_filter=True,
        condition_in_question="after_condition",
        reorder_context="sort",
        dynamic_context_compression_ratio=0.4,
        condition_compare=True,
        context_budget="+100",
        rank_method="longllmlingua",
        concate_question=False,
    )
    return response, (time.perf_counter() - started) * 1000.0


def response_record(
    qwen_tokenizer: Any,
    question: str,
    response: Mapping[str, Any],
) -> Dict[str, Any]:
    official_output = str(response.get("compressed_prompt", "")).strip()
    packed = format_packed(question, official_output)
    return {
        "official_output": official_output,
        "compressed_text": packed,
        "compressed_qwen_tokens": token_count(qwen_tokenizer, packed),
        "official_origin_tokens": response.get("origin_tokens"),
        "official_compressed_tokens": response.get("compressed_tokens"),
        "official_reported_ratio": response.get("ratio"),
        "official_reported_rate": response.get("rate"),
    }


def rate_search(
    compressor: Any,
    qwen_tokenizer: Any,
    chunks: Sequence[str],
    question: str,
    budget: int,
    initial_target: int,
    initial_response: Mapping[str, Any],
    initial_time_ms: float,
    iterative_size: int,
    max_calls: int,
) -> Dict[str, Any]:
    initial = response_record(qwen_tokenizer, question, initial_response)
    trace: List[Dict[str, Any]] = [
        {
            "target_token": initial_target,
            "compressed_qwen_tokens": initial["compressed_qwen_tokens"],
            "compressor_time_ms": initial_time_ms,
        }
    ]
    candidates: List[Tuple[int, Dict[str, Any], float]] = []
    if initial["compressed_qwen_tokens"] <= budget and initial["official_output"]:
        candidates.append((initial_target, initial, initial_time_ms))
        return {
            "best": candidates[0],
            "trace": trace,
            "total_time_ms": initial_time_ms,
            "calls": 1,
        }
    low, high = 32, initial_target - 1
    calls = 1
    while low <= high and calls < max_calls:
        target = (low + high) // 2
        response, elapsed = official_call(
            compressor, chunks, question, target, iterative_size
        )
        record = response_record(qwen_tokenizer, question, response)
        trace.append(
            {
                "target_token": target,
                "compressed_qwen_tokens": record["compressed_qwen_tokens"],
                "compressor_time_ms": elapsed,
            }
        )
        calls += 1
        if (
            record["compressed_qwen_tokens"] <= budget
            and record["official_output"]
        ):
            candidates.append((target, record, elapsed))
            low = target + 1
        else:
            high = target - 1
    best = (
        max(candidates, key=lambda item: (item[1]["compressed_qwen_tokens"], item[0]))
        if candidates
        else None
    )
    return {
        "best": best,
        "trace": trace,
        "total_time_ms": sum(float(item["compressor_time_ms"]) for item in trace),
        "calls": calls,
    }


def base_row(
    *,
    protocol: str,
    task: str,
    sample_id: str,
    sample_index: int,
    question: str,
    context: str,
    source_qwen_tokens: int,
    source_official_tokens: int,
    chunks: Sequence[str],
    max_chunk_tokens: int,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    return {
        "protocol": protocol,
        "method_name": protocol,
        "task": task,
        "sample_id": sample_id,
        "sample_index": sample_index,
        "question": question,
        "question_char_length": len(question),
        "source_context_char_length": len(context),
        "source_context_qwen_tokens": source_qwen_tokens,
        "source_context_official_tokens": source_official_tokens,
        "official_context_chunks": len(chunks),
        "official_max_chunk_tokens": max_chunk_tokens,
        "official_model_max_position": args.official_model_max_position,
        "budget": args.budget,
        "compressor_name": "Microsoft LLMLingua PromptCompressor",
        "compressor_backend": str(args.compressor_model.resolve()),
        "compressor_backend_revision": args.compressor_model.name,
        "compressor_version": "0.2.2",
        "official_source_commit": "e0e9d99beb94098bbd924aa53c2c112eac41c758",
        "rank_method": "longllmlingua",
        "whether_query_aware": True,
        "sentence_level_filter": False,
        "sentence_level_filter_reason": "NLTK punkt_tab unavailable offline",
        "context_level_filter": True,
        "token_level_filter": True,
        "iterative_size": args.iterative_size,
        "input_chunk_tokens": args.input_chunk_tokens,
        "whether_official_unmodified": protocol != "official_plus_exact_truncation",
        "whether_post_trimmed": False,
        "compressed_text": "",
        "compressed_qwen_tokens": "",
        "over_budget": "",
        "empty_compressed_text": "",
        "compressor_error": "",
        "compressor_time_ms": "",
        "postprocess_time_ms": 0.0,
        "target_token": "",
        "compression_rate": "",
        "official_origin_tokens": "",
        "official_compressed_tokens": "",
        "search_calls": "",
        "search_trace": [],
    }


def summarize(
    rows: Sequence[Mapping[str, Any]],
    protocols: Sequence[str],
    tasks: Sequence[str],
) -> Dict[str, Any]:
    cells: Dict[str, Any] = {}
    for protocol in protocols:
        for task in tasks:
            cell = [
                row
                for row in rows
                if row["protocol"] == protocol and row["task"] == task
            ]
            successful = [row for row in cell if not row["compressor_error"]]
            tokens = [int(row["compressed_qwen_tokens"]) for row in successful]
            times = [float(row["compressor_time_ms"]) for row in successful]
            cells[f"{protocol}/{task}"] = {
                "samples": len(cell),
                "successful": len(successful),
                "mean_tokens": statistics.mean(tokens) if tokens else None,
                "p50_tokens": statistics.median(tokens) if tokens else None,
                "p95_tokens": percentile_nearest(tokens, 0.95),
                "max_tokens": max(tokens) if tokens else None,
                "over_budget_count": sum(
                    bool(row["over_budget"]) for row in successful
                ),
                "empty_count": sum(
                    bool(row["empty_compressed_text"]) for row in successful
                ),
                "compressor_error_count": len(cell) - len(successful),
                "mean_compression_time_ms": (
                    statistics.mean(times) if times else None
                ),
            }
    protocol_status: Dict[str, Any] = {}
    for protocol in protocols:
        selected = [row for row in rows if row["protocol"] == protocol]
        protocol_status[protocol] = {
            "rows": len(selected),
            "over_budget": sum(
                bool(row["over_budget"])
                for row in selected
                if not row["compressor_error"]
            ),
            "empty": sum(
                bool(row["empty_compressed_text"])
                for row in selected
                if not row["compressor_error"]
            ),
            "errors": sum(bool(row["compressor_error"]) for row in selected),
        }
        protocol_status[protocol]["qa_eligible"] = bool(selected) and all(
            protocol_status[protocol][key] == 0
            for key in ("over_budget", "empty", "errors")
        )
    return {"cells": cells, "protocol_status": protocol_status}


def parse_args() -> argparse.Namespace:
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=(
            REPO_ROOT
            / "results"
            / f"bridgepack_official_llmlingua_audit_{stamp}"
        ),
    )
    parser.add_argument("--reference_dir", type=Path, default=REFERENCE_DIR)
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument("--max_samples_per_task", type=int, default=10)
    parser.add_argument("--budget", type=int, default=1024)
    parser.add_argument("--qwen_tokenizer", type=Path, default=DEFAULT_QWEN)
    parser.add_argument("--compressor_model", type=Path, default=DEFAULT_COMPRESSOR)
    parser.add_argument("--input_chunk_tokens", type=int, default=400)
    parser.add_argument("--iterative_size", type=int, default=1800)
    parser.add_argument("--initial_target_token", type=int, default=1024)
    parser.add_argument("--rate_search_max_calls", type=int, default=6)
    parser.add_argument("--cpu_threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--official_model_max_position", type=int, default=2048)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = tuple(item.strip() for item in args.tasks.split(",") if item.strip())
    if tasks != TASKS:
        raise OfficialAuditError(f"tasks must exactly match {TASKS}")
    if args.budget != 1024:
        raise OfficialAuditError("official audit is locked to exact Qwen budget=1024")
    if args.seed != 0:
        raise OfficialAuditError("only deterministic seed=0 is supported")
    if args.max_samples_per_task < 1 or args.max_samples_per_task > 50:
        raise OfficialAuditError("compression audit permits 1..50 samples/task")
    if args.iterative_size >= args.official_model_max_position:
        raise OfficialAuditError("iterative_size must be below compressor max position")
    for path in (LLMLINGUA_ROOT, args.qwen_tokenizer, args.compressor_model):
        if not path.exists():
            raise OfficialAuditError(f"required local path missing: {path}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise OfficialAuditError(f"refusing to overwrite non-empty output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    import torch
    from transformers import AutoTokenizer

    torch.set_num_threads(max(1, args.cpu_threads))
    if str(LLMLINGUA_ROOT) not in sys.path:
        sys.path.insert(0, str(LLMLINGUA_ROOT))
    import llmlingua
    from llmlingua import PromptCompressor

    qwen_tokenizer = AutoTokenizer.from_pretrained(
        str(args.qwen_tokenizer),
        trust_remote_code=True,
        local_files_only=True,
    )
    load_started = time.perf_counter()
    compressor = PromptCompressor(
        model_name=str(args.compressor_model),
        device_map="cpu",
        model_config={"local_files_only": True},
    )
    load_ms = (time.perf_counter() - load_started) * 1000.0
    actual_max_position = int(
        getattr(compressor.model.config, "max_position_embeddings", -1)
    )
    if actual_max_position != args.official_model_max_position:
        raise OfficialAuditError(
            f"compressor max position={actual_max_position}, "
            f"expected={args.official_model_max_position}"
        )

    refs = {
        task: reference_rows(args.reference_dir, task, args.max_samples_per_task)
        for task in tasks
    }
    manifest_hash = sha256_text(
        canonical_json(
            {
                task: [str(row["sample_id"]) for row in rows]
                for task, rows in refs.items()
            }
        )
    )
    rows: List[Dict[str, Any]] = []
    packed_outputs: Dict[Tuple[str, str], List[Dict[str, Any]]] = (
        collections.defaultdict(list)
    )
    for task in tasks:
        for sample_index, source in enumerate(refs[task]):
            sample_id = str(source["sample_id"])
            question = str(source["question"])
            context = str(source["original_context"])
            source_qwen_tokens = token_count(qwen_tokenizer, context)
            chunks, source_official_tokens, max_chunk = split_official_context(
                compressor, context, args.input_chunk_tokens
            )
            common = {
                "task": task,
                "sample_id": sample_id,
                "sample_index": sample_index,
                "question": question,
                "context": context,
                "source_qwen_tokens": source_qwen_tokens,
                "source_official_tokens": source_official_tokens,
                "chunks": chunks,
                "max_chunk_tokens": max_chunk,
                "args": args,
            }
            unmodified_row = base_row(protocol="official_unmodified", **common)
            try:
                response, elapsed = official_call(
                    compressor,
                    chunks,
                    question,
                    args.initial_target_token,
                    args.iterative_size,
                )
                unmodified = response_record(qwen_tokenizer, question, response)
                unmodified_row.update(unmodified)
                unmodified_row.update(
                    {
                        "target_token": args.initial_target_token,
                        "compression_rate": response.get("rate", ""),
                        "compressor_time_ms": elapsed,
                        "over_budget": (
                            unmodified["compressed_qwen_tokens"] > args.budget
                        ),
                        "empty_compressed_text": not bool(
                            unmodified["official_output"].strip()
                        ),
                        "search_calls": 1,
                    }
                )
            except Exception as exc:
                unmodified_row["compressor_error"] = (
                    f"{type(exc).__name__}: {exc}"
                )
                unmodified = None
                response = None
                elapsed = 0.0
            rows.append(unmodified_row)

            search_row = base_row(protocol="official_rate_search", **common)
            if response is None or unmodified is None:
                search_row["compressor_error"] = (
                    "initial official call failed: "
                    + str(unmodified_row["compressor_error"])
                )
            else:
                try:
                    searched = rate_search(
                        compressor,
                        qwen_tokenizer,
                        chunks,
                        question,
                        args.budget,
                        args.initial_target_token,
                        response,
                        elapsed,
                        args.iterative_size,
                        args.rate_search_max_calls,
                    )
                    best = searched["best"]
                    search_row["search_trace"] = searched["trace"]
                    search_row["search_calls"] = searched["calls"]
                    search_row["compressor_time_ms"] = searched["total_time_ms"]
                    if best is None:
                        search_row["compressor_error"] = (
                            "official target_token search found no non-empty "
                            "output within exact Qwen budget"
                        )
                    else:
                        target, record, _best_elapsed = best
                        search_row.update(record)
                        search_row.update(
                            {
                                "target_token": target,
                                "compression_rate": record[
                                    "official_reported_rate"
                                ],
                                "over_budget": (
                                    record["compressed_qwen_tokens"] > args.budget
                                ),
                                "empty_compressed_text": not bool(
                                    record["official_output"].strip()
                                ),
                            }
                        )
                except Exception as exc:
                    search_row["compressor_error"] = (
                        f"{type(exc).__name__}: {exc}"
                    )
            rows.append(search_row)

            truncation_row = base_row(
                protocol="official_plus_exact_truncation", **common
            )
            if unmodified is None:
                truncation_row["compressor_error"] = (
                    "initial official call failed: "
                    + str(unmodified_row["compressor_error"])
                )
            else:
                post_started = time.perf_counter()
                trimmed = exact_qwen_prefix(
                    qwen_tokenizer,
                    str(unmodified["compressed_text"]),
                    args.budget,
                )
                post_ms = (time.perf_counter() - post_started) * 1000.0
                trimmed_tokens = token_count(qwen_tokenizer, trimmed)
                evidence = (
                    trimmed.split("Selected compressed evidence:", 1)[1].strip()
                    if "Selected compressed evidence:" in trimmed
                    else ""
                )
                truncation_row.update(
                    {
                        "official_output": unmodified["official_output"],
                        "compressed_text": trimmed,
                        "compressed_qwen_tokens": trimmed_tokens,
                        "official_origin_tokens": unmodified[
                            "official_origin_tokens"
                        ],
                        "official_compressed_tokens": unmodified[
                            "official_compressed_tokens"
                        ],
                        "official_reported_ratio": unmodified[
                            "official_reported_ratio"
                        ],
                        "official_reported_rate": unmodified[
                            "official_reported_rate"
                        ],
                        "target_token": args.initial_target_token,
                        "compression_rate": unmodified[
                            "official_reported_rate"
                        ],
                        "compressor_time_ms": elapsed,
                        "postprocess_time_ms": post_ms,
                        "over_budget": trimmed_tokens > args.budget,
                        "empty_compressed_text": not bool(evidence),
                        "whether_post_trimmed": (
                            unmodified["compressed_qwen_tokens"] > args.budget
                        ),
                        "search_calls": 1,
                    }
                )
            rows.append(truncation_row)

            for protocol_row in (
                unmodified_row,
                search_row,
                truncation_row,
            ):
                if (
                    protocol_row["compressor_error"]
                    or protocol_row["over_budget"]
                    or protocol_row["empty_compressed_text"]
                ):
                    continue
                packed_outputs[(protocol_row["protocol"], task)].append(
                    {
                        "sample_id": sample_id,
                        "sample_index": sample_index,
                        "task": task,
                        "variant": protocol_row["protocol"],
                        "method_name": protocol_row["protocol"],
                        "budget": args.budget,
                        "packed_text": protocol_row["compressed_text"],
                        "packed_prompt": protocol_row["compressed_text"],
                        "generation_prompt": generation_prompt(
                            task,
                            str(protocol_row["compressed_text"]),
                            question,
                        ),
                        "question": question,
                        "answers": source.get(
                            "answers",
                            source.get("gold_analysis_only_answers", []),
                        ),
                        "gold_analysis_only_answers": source.get(
                            "gold_analysis_only_answers",
                            source.get("answers", []),
                        ),
                        "all_classes": source.get("all_classes"),
                        "length": source.get("length", 0),
                        "packed_token_count": protocol_row[
                            "compressed_qwen_tokens"
                        ],
                        "model_token_count": protocol_row[
                            "compressed_qwen_tokens"
                        ],
                        "exact_model_token_budget": True,
                        "model_tokenizer": str(args.qwen_tokenizer.resolve()),
                        "official_llmlingua": True,
                        "official_unmodified": protocol_row[
                            "whether_official_unmodified"
                        ],
                        "post_trimmed": protocol_row["whether_post_trimmed"],
                        "compressor_backend": str(
                            args.compressor_model.resolve()
                        ),
                    }
                )

    for (protocol, task), output_rows in packed_outputs.items():
        write_jsonl(
            args.output_dir
            / "packed_prompts_build"
            / "packed_prompts"
            / protocol
            / f"budget_{args.budget}"
            / f"{task}.jsonl",
            output_rows,
        )
    write_jsonl(args.output_dir / "official_compression_audit.jsonl", rows)
    audit_fields = (
        "protocol",
        "task",
        "sample_id",
        "sample_index",
        "question_char_length",
        "source_context_char_length",
        "source_context_qwen_tokens",
        "source_context_official_tokens",
        "official_context_chunks",
        "official_max_chunk_tokens",
        "target_token",
        "compression_rate",
        "compressed_qwen_tokens",
        "over_budget",
        "empty_compressed_text",
        "compressor_error",
        "compressor_time_ms",
        "whether_query_aware",
        "whether_official_unmodified",
        "whether_post_trimmed",
    )
    write_csv(args.output_dir / "official_compression_audit.csv", rows, audit_fields)
    summary = summarize(rows, PROTOCOLS, tasks)
    availability = {
        "official_found": True,
        "official_repository": "https://github.com/microsoft/LLMLingua",
        "official_code_path": str(LLMLINGUA_ROOT.resolve()),
        "official_prompt_compressor_path": str(
            (LLMLINGUA_ROOT / "llmlingua" / "prompt_compressor.py").resolve()
        ),
        "official_source_commit": "e0e9d99beb94098bbd924aa53c2c112eac41c758",
        "package_version": str(getattr(llmlingua, "__version__", "unknown")),
        "compressor_class": "llmlingua.prompt_compressor.PromptCompressor",
        "compressor_backend": str(args.compressor_model.resolve()),
        "compressor_backend_note": (
            "Local facebook/opt-125m causal-LM backend; this is not the "
            "default NousResearch/Llama-2-7b-hf backend."
        ),
        "offline": True,
        "cpu_only": True,
        "gpu_used": False,
        "model_load_time_ms": load_ms,
        "query_aware": True,
        "long_context_adapter": (
            f"official context-list API with {args.input_chunk_tokens}-token "
            "compressor-tokenizer chunks"
        ),
        "official_model_max_position": actual_max_position,
        "sentence_filter_status": (
            "disabled via official parameter because NLTK punkt_tab is "
            "not installed and network download is forbidden"
        ),
        "transformers_5_token_cache_compatibility": (
            f"official iterative_size={args.iterative_size} keeps the compressed "
            "sequence in one cache iteration; no official source modification"
        ),
    }
    summary.update(
        {
            "status": "COMPLETE",
            "created_at": dt.datetime.now().astimezone().isoformat(),
            "output_dir": str(args.output_dir.resolve()),
            "reference_dir": str(args.reference_dir.resolve()),
            "tasks": list(tasks),
            "samples_per_task": args.max_samples_per_task,
            "budget": args.budget,
            "sample_manifest_hash": manifest_hash,
            "sample_id_alignment": True,
            "qwen_tokenizer": str(args.qwen_tokenizer.resolve()),
            "prompt_wrapper": (
                "configs/dataset2prompt.json; Qwen chat template deferred to QA"
            ),
            "prompt_config_sha256": sha256_file(
                REPO_ROOT / "configs" / "dataset2prompt.json"
            ),
            "availability": availability,
            "qa_run_performed": False,
        }
    )
    write_json(args.output_dir / "availability_audit.json", availability)
    write_json(args.output_dir / "official_compression_summary.json", summary)
    write_json(
        args.output_dir / "config.json",
        {
            **vars(args),
            "output_dir": str(args.output_dir.resolve()),
            "reference_dir": str(args.reference_dir.resolve()),
            "qwen_tokenizer": str(args.qwen_tokenizer.resolve()),
            "compressor_model": str(args.compressor_model.resolve()),
            "tasks": list(tasks),
            "protocols": list(PROTOCOLS),
            "network_download_allowed": False,
            "gpu_used": False,
        },
    )
    print(str(args.output_dir.resolve()))


if __name__ == "__main__":
    main()
