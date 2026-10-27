#!/usr/bin/env python3
"""Stage 2 / G3 official LLMLingua external-compression baseline.

BridgePack-only wrapper: imports official PromptCompressor helpers, never
co-resides compressor with the QA model on GPU, and keeps raw vs adapted
labels strictly separated.
"""
from __future__ import annotations

import argparse
import collections
import csv
import gc
import json
import os
import sys
import time
import traceback
import types
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.bridgepack_cpu_stage.common import FAIRNESS_DIR, read_csv
from tools.bridgepack_g1.common import (
    QWEN_PATH,
    TASKS,
    read_jsonl,
    select_fixed_sample_ids,
    utc_now,
    write_json,
    write_jsonl,
    write_text,
)
from tools.bridgepack_g1.generate import load_qwen_fp16
from tools.bridgepack_g1.run_g1 import score_predictions
from tools.bridgepack_g1.stats import (
    approx_p_from_bootstrap,
    holm_correction,
    load_pred_scores,
    paired_bootstrap_delta,
)
from tools.bridgepack_g2.run_g2 import LLAMA_MODEL_PATH, g2_safety_gate
import tools.run_official_llmlingua_exact_budget_audit as official

PIPELINE_ROOT = REPO_ROOT / "results" / "bridgepack_completion_pipeline_20260727_140453"
G1_ROOT = REPO_ROOT / "results" / "bridgepack_g1_qwen1024_matched_20260727_004934"
G2_ROOT = REPO_ROOT / "results" / "bridgepack_g2_llama1024_sourceorder_20260727_023300"
BUDGET = 1024
ADAPTED_QWEN = "exact_budget_adapted_qwen"
ADAPTED_LLAMA = "exact_budget_adapted_llama3"
RAW = "official_raw_output"
RATE = "official_rate_search"


def _args_ns(**kwargs: Any) -> argparse.Namespace:
    defaults = {
        "budget": BUDGET,
        "qwen_tokenizer": official.DEFAULT_QWEN,
        "compressor_model": official.DEFAULT_COMPRESSOR,
        "input_chunk_tokens": 400,
        "iterative_size": 1800,
        "initial_target_token": 1024,
        "rate_search_max_calls": 6,
        "cpu_threads": 8,
        "seed": 0,
        "official_model_max_position": 2048,
        "reference_dir": official.REFERENCE_DIR,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def _load_refs(sample_ids: Optional[Mapping[str, Sequence[str]]] = None) -> Dict[str, List[Dict[str, Any]]]:
    refs = {}
    for task in TASKS:
        rows = official.reference_rows(official.REFERENCE_DIR, task, -1)  # full task, then subset
        if sample_ids is not None:
            by_id = {str(r["sample_id"]): r for r in rows}
            rows = [by_id[i] for i in sample_ids[task] if i in by_id]
            if len(rows) != len(sample_ids[task]):
                raise RuntimeError(f"{task}: missing refs {len(rows)}/{len(sample_ids[task])}")
        refs[task] = rows
    return refs


def _audit_done_ids(audit_path: Path) -> set[Tuple[str, str, str]]:
    if not audit_path.exists():
        return set()
    done = set()
    for row in read_jsonl(audit_path):
        done.add((str(row["protocol"]), str(row["task"]), str(row["sample_id"])))
    return done


def _append_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def run_compression_audit(
    out_dir: Path,
    *,
    stage: str,
    sample_ids: Optional[Mapping[str, Sequence[str]]],
    protocols: Sequence[str],
    resume: bool = True,
) -> Dict[str, Any]:
    """CPU-only official compression. Compressor never uses GPU."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    import torch
    from transformers import AutoTokenizer

    args = _args_ns()
    torch.set_num_threads(max(1, args.cpu_threads))
    if str(official.LLMLINGUA_ROOT) not in sys.path:
        sys.path.insert(0, str(official.LLMLINGUA_ROOT))
    import llmlingua
    from llmlingua import PromptCompressor

    stage_dir = out_dir / stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    audit_path = stage_dir / "official_compression_audit.jsonl"
    done = _audit_done_ids(audit_path) if resume else set()

    qwen_tokenizer = AutoTokenizer.from_pretrained(
        str(args.qwen_tokenizer), trust_remote_code=True, local_files_only=True
    )
    llama_tokenizer = AutoTokenizer.from_pretrained(
        str(LLAMA_MODEL_PATH), trust_remote_code=True, local_files_only=True
    )
    load_started = time.perf_counter()
    compressor = PromptCompressor(
        model_name=str(args.compressor_model),
        device_map="cpu",
        model_config={"local_files_only": True},
    )
    load_ms = (time.perf_counter() - load_started) * 1000.0
    actual_max_position = int(getattr(compressor.model.config, "max_position_embeddings", -1))
    if actual_max_position != args.official_model_max_position:
        raise RuntimeError(f"compressor max position={actual_max_position}")

    refs = _load_refs(sample_ids)
    packed_buffers: Dict[Tuple[str, str], List[Dict[str, Any]]] = collections.defaultdict(list)
    new_rows: List[Dict[str, Any]] = []

    for task in TASKS:
        for sample_index, source in enumerate(refs[task]):
            sample_id = str(source["sample_id"])
            print(f"[audit {stage}] {task} {sample_index+1}/{len(refs[task])} {sample_id}", flush=True)
            # skip only if all requested protocols already present
            if all((p, task, sample_id) in done for p in protocols):
                continue
            question = str(source["question"])
            context = str(source["original_context"])
            source_qwen_tokens = official.token_count(qwen_tokenizer, context)
            chunks, source_official_tokens, max_chunk = official.split_official_context(
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

            unmodified_row = official.base_row(protocol=RAW, **common)
            unmodified = None
            response = None
            elapsed = 0.0
            if (RAW, task, sample_id) not in done and RAW in protocols:
                try:
                    response, elapsed = official.official_call(
                        compressor, chunks, question, args.initial_target_token, args.iterative_size
                    )
                    unmodified = official.response_record(qwen_tokenizer, question, response)
                    unmodified_row.update(unmodified)
                    unmodified_row.update(
                        {
                            "target_token": args.initial_target_token,
                            "compression_rate": response.get("rate", ""),
                            "compressor_time_ms": elapsed,
                            "over_budget": unmodified["compressed_qwen_tokens"] > args.budget,
                            "empty_compressed_text": not bool(unmodified["official_output"].strip()),
                            "search_calls": 1,
                            "label_class": "official_raw_output",
                        }
                    )
                except Exception as exc:
                    unmodified_row["compressor_error"] = f"{type(exc).__name__}: {exc}"
                new_rows.append(unmodified_row)
            elif (RAW, task, sample_id) in done:
                # reload raw for derived protocols
                for row in read_jsonl(audit_path):
                    if row.get("protocol") == RAW and row.get("task") == task and str(row.get("sample_id")) == sample_id:
                        unmodified_row = row
                        if not row.get("compressor_error") and row.get("official_output"):
                            unmodified = {
                                "official_output": row["official_output"],
                                "compressed_text": row["compressed_text"],
                                "compressed_qwen_tokens": int(row["compressed_qwen_tokens"]),
                                "official_origin_tokens": row.get("official_origin_tokens"),
                                "official_compressed_tokens": row.get("official_compressed_tokens"),
                                "official_reported_ratio": row.get("official_reported_ratio"),
                                "official_reported_rate": row.get("official_reported_rate"),
                            }
                            elapsed = float(row.get("compressor_time_ms") or 0.0)
                        break

            if RATE in protocols and (RATE, task, sample_id) not in done:
                search_row = official.base_row(protocol=RATE, **common)
                if unmodified is None:
                    # need a fresh call for rate search
                    try:
                        response, elapsed = official.official_call(
                            compressor, chunks, question, args.initial_target_token, args.iterative_size
                        )
                        unmodified = official.response_record(qwen_tokenizer, question, response)
                    except Exception as exc:
                        search_row["compressor_error"] = f"initial call failed: {type(exc).__name__}: {exc}"
                        new_rows.append(search_row)
                        response = None
                if unmodified is not None and response is not None:
                    try:
                        searched = official.rate_search(
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
                        search_row["search_trace"] = searched["trace"]
                        search_row["search_calls"] = searched["calls"]
                        search_row["compressor_time_ms"] = searched["total_time_ms"]
                        search_row["label_class"] = "official_rate_search"
                        best = searched["best"]
                        if best is None:
                            search_row["compressor_error"] = (
                                "official target_token search found no non-empty output within exact Qwen budget"
                            )
                        else:
                            target, record, _ = best
                            search_row.update(record)
                            search_row.update(
                                {
                                    "target_token": target,
                                    "compression_rate": record["official_reported_rate"],
                                    "over_budget": record["compressed_qwen_tokens"] > args.budget,
                                    "empty_compressed_text": not bool(record["official_output"].strip()),
                                }
                            )
                    except Exception as exc:
                        search_row["compressor_error"] = f"{type(exc).__name__}: {exc}"
                    new_rows.append(search_row)

            # Always derive Qwen/Llama adapted from raw official output when available.
            for adapted_name, tokenizer, tok_field in (
                (ADAPTED_QWEN, qwen_tokenizer, "compressed_qwen_tokens"),
                (ADAPTED_LLAMA, llama_tokenizer, "compressed_llama_tokens"),
            ):
                if adapted_name not in protocols or (adapted_name, task, sample_id) in done:
                    continue
                trunc_row = official.base_row(protocol=adapted_name, **common)
                trunc_row["label_class"] = "exact_budget_adapted_output"
                trunc_row["whether_official_unmodified"] = False
                if unmodified is None:
                    trunc_row["compressor_error"] = (
                        "missing official_raw_output: " + str(unmodified_row.get("compressor_error", ""))
                    )
                else:
                    post_started = time.perf_counter()
                    trimmed = official.exact_qwen_prefix(
                        tokenizer, str(unmodified["compressed_text"]), args.budget
                    )
                    post_ms = (time.perf_counter() - post_started) * 1000.0
                    trimmed_tokens = official.token_count(tokenizer, trimmed)
                    evidence = (
                        trimmed.split("Selected compressed evidence:", 1)[1].strip()
                        if "Selected compressed evidence:" in trimmed
                        else ""
                    )
                    trunc_row.update(
                        {
                            "official_output": unmodified["official_output"],
                            "compressed_text": trimmed,
                            tok_field: trimmed_tokens,
                            "compressed_qwen_tokens": (
                                official.token_count(qwen_tokenizer, trimmed)
                                if adapted_name == ADAPTED_QWEN
                                else unmodified.get("compressed_qwen_tokens", "")
                            ),
                            "official_origin_tokens": unmodified["official_origin_tokens"],
                            "official_compressed_tokens": unmodified["official_compressed_tokens"],
                            "official_reported_ratio": unmodified["official_reported_ratio"],
                            "official_reported_rate": unmodified["official_reported_rate"],
                            "target_token": args.initial_target_token,
                            "compression_rate": unmodified["official_reported_rate"],
                            "compressor_time_ms": elapsed,
                            "postprocess_time_ms": post_ms,
                            "over_budget": trimmed_tokens > args.budget,
                            "empty_compressed_text": not bool(evidence),
                            "whether_post_trimmed": True,
                            "search_calls": 1,
                            "model_tokenizer": str(
                                (args.qwen_tokenizer if adapted_name == ADAPTED_QWEN else LLAMA_MODEL_PATH).resolve()
                            ),
                        }
                    )
                new_rows.append(trunc_row)

            if len(new_rows) >= 12:
                _append_jsonl(audit_path, new_rows)
                new_rows = []

    if new_rows:
        _append_jsonl(audit_path, new_rows)

    # Rebuild packed prompts from full audit for requested adapted/rate protocols.
    all_rows = read_jsonl(audit_path)
    for row in all_rows:
        protocol = str(row["protocol"])
        if protocol not in protocols:
            continue
        if row.get("compressor_error") or row.get("over_budget") or row.get("empty_compressed_text"):
            # raw over-budget is retained in audit but never packed for fair QA
            if protocol == RAW:
                continue
            # adapted/rate must be fully eligible
            continue
        task = str(row["task"])
        source = next(r for r in refs[task] if str(r["sample_id"]) == str(row["sample_id"]))
        tok_count = int(
            row.get("compressed_llama_tokens")
            or row.get("compressed_qwen_tokens")
            or 0
        )
        packed_buffers[(protocol, task)].append(
            {
                "sample_id": row["sample_id"],
                "sample_index": row["sample_index"],
                "task": task,
                "variant": protocol,
                "method_name": protocol,
                "budget": args.budget,
                "packed_text": row["compressed_text"],
                "packed_prompt": row["compressed_text"],
                "generation_prompt": official.generation_prompt(task, str(row["compressed_text"]), str(row["question"])),
                "question": row["question"],
                "answers": source.get("answers", source.get("gold_analysis_only_answers", [])),
                "gold_analysis_only_answers": source.get("gold_analysis_only_answers", source.get("answers", [])),
                "all_classes": source.get("all_classes"),
                "length": source.get("length", 0),
                "packed_token_count": tok_count,
                "model_token_count": tok_count,
                "exact_model_token_budget": True,
                "model_tokenizer": row.get(
                    "model_tokenizer",
                    str(args.qwen_tokenizer.resolve() if protocol == ADAPTED_QWEN else LLAMA_MODEL_PATH.resolve()),
                ),
                "official_llmlingua": True,
                "label_class": row.get("label_class", protocol),
                "post_trimmed": row.get("whether_post_trimmed", False),
                "compressor_backend": str(args.compressor_model.resolve()),
            }
        )

    for (protocol, task), output_rows in packed_buffers.items():
        output_rows = sorted(output_rows, key=lambda r: int(r["sample_index"]))
        write_jsonl(
            stage_dir
            / "packed_prompts_build"
            / "packed_prompts"
            / protocol
            / f"budget_{args.budget}"
            / f"{task}.jsonl",
            output_rows,
        )

    summary = official.summarize(all_rows, tuple(protocols), TASKS)
    # Remap qa_eligible for adapted: require full sample coverage and 0 over-budget.
    expected_n = sum(len(v) for v in refs.values())
    for protocol in protocols:
        selected = [r for r in all_rows if r["protocol"] == protocol]
        overs = sum(bool(r.get("over_budget")) for r in selected if not r.get("compressor_error"))
        empties = sum(bool(r.get("empty_compressed_text")) for r in selected if not r.get("compressor_error"))
        errors = sum(bool(r.get("compressor_error")) for r in selected)
        status = {
            "rows": len(selected),
            "expected_rows": expected_n,
            "over_budget": overs,
            "empty": empties,
            "errors": errors,
        }
        if protocol == RAW:
            status["qa_eligible"] = False
            status["note"] = "official raw never enters fair main QA comparison"
        else:
            status["qa_eligible"] = (
                len(selected) == expected_n and overs == 0 and empties == 0 and errors == 0
            )
        summary["protocol_status"][protocol] = status

    write_json(stage_dir / "official_compression_summary.json", summary)
    write_json(
        stage_dir / "availability_audit.json",
        {
            "official_found": True,
            "package_version": str(getattr(llmlingua, "__version__", "unknown")),
            "compressor_backend": str(args.compressor_model.resolve()),
            "compressor_backend_note": "Local facebook/opt-125m; not NousResearch/Llama-2-7b-hf",
            "cpu_only": True,
            "gpu_used": False,
            "model_load_time_ms": load_ms,
            "stage": stage,
            "generated_at": utc_now(),
        },
    )

    # Release compressor before any QA.
    del compressor
    gc.collect()
    return summary



def write_token_budget_and_failure_audit(out_dir: Path, stage: str) -> None:
    """Write token/budget audit + failure sample stats for raw vs adapted."""
    import statistics

    stage_dir = out_dir / stage
    rows = read_jsonl(stage_dir / "official_compression_audit.jsonl")
    token_rows = []
    fail_rows = []
    for row in rows:
        protocol = str(row["protocol"])
        tok = row.get("compressed_llama_tokens")
        if tok in ("", None):
            tok = row.get("compressed_qwen_tokens")
        try:
            tok_i = int(tok) if tok not in ("", None) else ""
        except Exception:
            tok_i = ""
        over = bool(row.get("over_budget"))
        empty = bool(row.get("empty_compressed_text"))
        err = str(row.get("compressor_error") or "")
        token_rows.append(
            {
                "protocol": protocol,
                "label_class": row.get("label_class", protocol),
                "task": row["task"],
                "sample_id": row["sample_id"],
                "compressed_tokens": tok_i,
                "budget": BUDGET,
                "over_budget": over,
                "empty_compressed_text": empty,
                "compressor_error": err,
                "in_exact_budget_main_table": protocol in {ADAPTED_QWEN, ADAPTED_LLAMA} and not over and not empty and not err,
            }
        )
        if over or empty or err or protocol == RAW:
            fail_rows.append(
                {
                    "protocol": protocol,
                    "task": row["task"],
                    "sample_id": row["sample_id"],
                    "over_budget": over,
                    "empty_compressed_text": empty,
                    "compressor_error": err,
                    "compressed_tokens": tok_i,
                    "role": "diagnostic_raw_or_failure",
                }
            )

    with (out_dir / "G3_TOKEN_BUDGET_AUDIT.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = list(token_rows[0].keys()) if token_rows else [
            "protocol","label_class","task","sample_id","compressed_tokens","budget","over_budget","empty_compressed_text","compressor_error","in_exact_budget_main_table"
        ]
        w = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for r in token_rows:
            w.writerow(r)
    with (out_dir / "G3_FAILURE_SAMPLE_STATS.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["protocol","task","sample_id","over_budget","empty_compressed_text","compressor_error","compressed_tokens","role"]
        w = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for r in fail_rows:
            w.writerow(r)

    # summary by protocol
    lines = ["# G3_TOKEN_BUDGET_AUDIT", "", f"Generated: {utc_now()}", ""]
    by_p = {}
    for r in token_rows:
        by_p.setdefault(r["protocol"], []).append(r)
    for protocol, items in by_p.items():
        toks = [int(x["compressed_tokens"]) for x in items if x["compressed_tokens"] != ""]
        overs = sum(1 for x in items if x["over_budget"])
        empties = sum(1 for x in items if x["empty_compressed_text"])
        errs = sum(1 for x in items if x["compressor_error"])
        main = sum(1 for x in items if x["in_exact_budget_main_table"])
        lines.append(f"## {protocol}")
        lines.append(f"- rows: {len(items)}")
        lines.append(f"- over_budget: {overs}")
        lines.append(f"- empty: {empties}")
        lines.append(f"- errors: {errs}")
        lines.append(f"- exact-budget main-table eligible: {main}")
        if toks:
            lines.append(f"- tokens mean/p50/max: {statistics.mean(toks):.1f}/{statistics.median(toks):.1f}/{max(toks)}")
        lines.append("")
    write_text(out_dir / "G3_TOKEN_BUDGET_AUDIT.md", "\n".join(lines))


def write_audit_markdown(out_dir: Path, smoke_summary: Mapping[str, Any], full_summary: Optional[Mapping[str, Any]]) -> None:
    lines = [
        "# G3_EXTERNAL_COMPRESSION_AUDIT",
        "",
        f"Generated: {utc_now()}",
        "",
        "## Label separation",
        "",
        f"- `{RAW}`: unmodified official compressor output; over-budget kept as FAIL records; **not** in fair QA.",
        f"- `{ADAPTED_QWEN}` / `{ADAPTED_LLAMA}`: exact-budget adapted via tokenizer prefix truncation; labeled adapted, never unmodified official.",
        f"- `{RATE}`: official target_token search only (no text trim); smoke diagnostic.",
        "",
        "## Smoke12 protocol status",
        "",
        "```json",
        json.dumps(smoke_summary.get("protocol_status", {}), indent=2),
        "```",
        "",
    ]
    if full_summary is not None:
        lines.extend(
            [
                "## Full1150 protocol status",
                "",
                "```json",
                json.dumps(full_summary.get("protocol_status", {}), indent=2),
                "```",
                "",
            ]
        )
    write_text(out_dir / "G3_EXTERNAL_COMPRESSION_AUDIT.md", "\n".join(lines))


def run_qa_stage(
    out_dir: Path,
    packed_stage_dir: Path,
    variant: str,
    model_name: str,
    sample_ids: Mapping[str, Sequence[str]],
    stage: str,
) -> None:
    gate = g2_safety_gate(out_dir, f"{stage}_{model_name}_before_load", phase="before_load")
    if gate["decision"] != "GO":
        write_text(out_dir / "WAITING_FOR_SAFE_GPU.md", json.dumps(gate, indent=2))
        raise RuntimeError(f"WAITING_FOR_SAFE_GPU: {gate['reasons']}")

    import tools.run_evidence_text_packing_generation_qwen32 as gen
    from tools.bridgepack_g1.generate import generate_greedy_historical

    model_path = QWEN_PATH if model_name == "qwen" else LLAMA_MODEL_PATH
    model, tokenizer = load_qwen_fp16(str(model_path), __import__("torch"))
    torch = __import__("torch")
    g2_safety_gate(out_dir, f"{stage}_{model_name}_after_load", phase="runtime")

    pred_root = out_dir / stage / "predictions"
    for task in TASKS:
        path = (
            packed_stage_dir
            / "packed_prompts_build"
            / "packed_prompts"
            / variant
            / f"budget_{BUDGET}"
            / f"{task}.jsonl"
        )
        rows = read_jsonl(path)
        by_id = {str(r["sample_id"]): r for r in rows}
        want = list(sample_ids[task])
        rows = [by_id[i] for i in want if i in by_id]
        if len(rows) != len(want):
            raise RuntimeError(f"Missing packs {variant}/{task}: {len(rows)}/{len(want)} at {path}")
        pred_path = pred_root / variant / f"{task}.jsonl"
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        done = {str(r["sample_id"]): r for r in read_jsonl(pred_path)}
        with pred_path.open("w", encoding="utf-8") as handle:
            for sample_index, row in enumerate(rows):
                sid = str(row.get("sample_id", sample_index))
                if sid in done and str(done[sid].get("pred", "")) != "":
                    handle.write(json.dumps(done[sid], ensure_ascii=False) + "\n")
                    handle.flush()
                    continue
                gate = g2_safety_gate(out_dir, f"{stage}_{variant}_{task}_{sample_index}", phase="runtime")
                if gate["decision"] != "GO":
                    raise RuntimeError(f"Safety stop: {gate['reasons']}")
                mt = int(row.get("model_token_count") or row.get("packed_token_count") or 0)
                if mt > BUDGET:
                    raise RuntimeError(f"Over-budget before QA: {variant}/{task}/{sid} tokens={mt}")
                prompt = str(row.get("generation_prompt") or "")
                if not prompt.strip():
                    prompt = gen._generation_prompt_from_row(task, row)
                max_new = gen._max_new_tokens_for(task, 0)
                pred, context_length = generate_greedy_historical(model, tokenizer, torch, prompt, max_new)
                payload = {
                    "sample_id": sid,
                    "sample_index": sample_index,
                    "task": task,
                    "variant": variant,
                    "pred": pred,
                    "answers": row.get("answers", row.get("gold_analysis_only_answers", [])),
                    "all_classes": row.get("all_classes", None),
                    "length": context_length,
                    "packed_token_count": mt,
                    "prompt_token_count": context_length,
                    "budget": BUDGET,
                    "question": row.get("question", ""),
                    "label_class": "exact_budget_adapted_output",
                }
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                handle.flush()
    score_predictions(pred_root, [variant])
    del model
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass


def _fairness_scores(model: str, method: str) -> Dict[Tuple[str, str], float]:
    out: Dict[Tuple[str, str], float] = {}
    for row in read_csv(FAIRNESS_DIR / "per_sample_scores.csv"):
        if row["model"] == model and int(row["budget"]) == BUDGET and row["method"] == method:
            out[(row["task"], row["sample_id"])] = float(row["score_0_100"])
    return out


def _pred_scores_from_root(root: Path, variant: str) -> Dict[Tuple[str, str], float]:
    return load_pred_scores(root / "full" / "predictions", variant, TASKS)


def write_stage2_stats(out_dir: Path) -> None:
    qwen_adapted = load_pred_scores(out_dir / "qa_full_qwen" / "predictions", ADAPTED_QWEN, TASKS)
    llama_adapted = load_pred_scores(out_dir / "qa_full_llama3" / "predictions", ADAPTED_LLAMA, TASKS)
    qwen_source = _pred_scores_from_root(G1_ROOT, "same_spans_source_order")
    llama_source = _pred_scores_from_root(G2_ROOT, "source_order_bridge_selection")
    llama_mmr = _pred_scores_from_root(G2_ROOT, "mmr_neighbor_source_order")

    result_rows = []
    for model_name, stage, variant in (
        ("qwen", "qa_full_qwen", ADAPTED_QWEN),
        ("llama3", "qa_full_llama3", ADAPTED_LLAMA),
    ):
        for row in score_predictions(out_dir / stage / "predictions", [variant]):
            row = dict(row)
            row["model"] = model_name
            row["budget"] = BUDGET
            row["label_class"] = "exact_budget_adapted_output"
            result_rows.append(row)

    comps = [
        ("qwen adapted vs source_order", qwen_adapted, qwen_source),
        ("qwen adapted vs bm25", qwen_adapted, _fairness_scores("qwen", "bm25_chunk_source_order")),
        ("qwen adapted vs bge", qwen_adapted, _fairness_scores("qwen", "bge_chunk_packing")),
        ("llama3 adapted vs source_order", llama_adapted, llama_source),
        ("llama3 adapted vs bm25", llama_adapted, _fairness_scores("llama3", "bm25_chunk_source_order")),
        ("llama3 adapted vs bge", llama_adapted, _fairness_scores("llama3", "bge_chunk_packing")),
        ("llama3 adapted vs mmr", llama_adapted, llama_mmr),
    ]
    # pooled adapted vs source
    pooled_a = {**{("qwen",) + k: v for k, v in qwen_adapted.items()}, **{("llama3",) + k: v for k, v in llama_adapted.items()}}
    pooled_s = {**{("qwen",) + k: v for k, v in qwen_source.items()}, **{("llama3",) + k: v for k, v in llama_source.items()}}
    # Remap keys to (task, sample_id) with model prefix in sample to keep strata by task only:
    # use task = f"{model}:{task}"
    def _retag(scores: Mapping[Tuple[str, str], float], model: str) -> Dict[Tuple[str, str], float]:
        return {(f"{model}:{t}", sid): sc for (t, sid), sc in scores.items()}

    pooled_adapted = {**_retag(qwen_adapted, "qwen"), **_retag(llama_adapted, "llama3")}
    pooled_source = {**_retag(qwen_source, "qwen"), **_retag(llama_source, "llama3")}
    comps.append(("pooled adapted vs source_order", pooled_adapted, pooled_source))

    boot_rows = []
    holm_inputs = []
    for label, a, b in comps:
        boot = paired_bootstrap_delta(a, b)
        if int(boot.get("n", 0)) <= 0:
            raise RuntimeError(f"Empty overlap for {label}")
        p_raw = approx_p_from_bootstrap(boot["delta_macro"], boot.get("boot_std", float("nan")))
        holm_inputs.append((label, p_raw))
        boot_rows.append(
            {
                "comparison": label,
                "n": boot["n"],
                "delta_macro": boot["delta_macro"],
                "ci95_low": boot["ci95_low"],
                "ci95_high": boot["ci95_high"],
                "crosses_zero": boot["crosses_zero"],
                **{f"delta_{t}": boot.get("task_deltas", {}).get(t, "") for t in TASKS},
            }
        )
    holm = {r["label"]: r for r in holm_correction(holm_inputs)}
    for row in boot_rows:
        h = holm[row["comparison"]]
        row["p_raw"] = h["p_raw"]
        row["p_holm"] = h["p_holm"]
        row["reject_holm_0.05"] = h["reject_holm_0.05"]

    with (out_dir / "G3_EXTERNAL_COMPRESSION_RESULTS.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["model", "budget", "variant", "label_class", "task", "n", "score", "mean_length"]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in result_rows:
            writer.writerow(row)
    with (out_dir / "G3_EXTERNAL_COMPRESSION_BOOTSTRAP.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "comparison",
            "n",
            "delta_macro",
            "ci95_low",
            "ci95_high",
            "crosses_zero",
            "p_raw",
            "p_holm",
            "reject_holm_0.05",
        ] + [f"delta_{t}" for t in TASKS]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in boot_rows:
            writer.writerow(row)

    q_macro = next(r for r in result_rows if r["model"] == "qwen" and r["task"] == "MACRO")
    l_macro = next(r for r in result_rows if r["model"] == "llama3" and r["task"] == "MACRO")
    # Aggregate GPU resource rows if present
    gpu_src = out_dir / "G2_GPU_RESOURCE_REPORT.csv"
    if gpu_src.exists():
        write_text(out_dir / "G3_GPU_RESOURCE_REPORT.csv", gpu_src.read_text(encoding="utf-8"))
    else:
        # minimal stub from last safety gate scans during QA
        write_text(out_dir / "G3_GPU_RESOURCE_REPORT.csv", "note\nSee per-unit G2_GPU_RESOURCE_REPORT.csv under qa_* dirs\n")

    write_text(
        out_dir / "G3_CLAIM_GUARD.md",
        f"""# G3_CLAIM_GUARD

Generated: {utc_now()}

| Claim | Allowed? | Basis |
|---|---|---|
| Report exact-budget adapted LLMLingua as unmodified official | **NO** | Adapted uses tokenizer prefix truncation |
| Use official raw over-budget packs in fair main table | **NO** | Raw retained only as FAIL audit records |
| Compressor backend is NousResearch/Llama-2-7b-hf | **NO** | Local facebook/opt-125m; disclosed in audit |
| Adapted Qwen macro ({float(q_macro['score']):.2f}) / Llama macro ({float(l_macro['score']):.2f}) | YES | Full 1150 QA under exact 1024 |
| Qwen MMR comparison | **NO / NA** | Qwen MMR full predictions were not run in G1 HOLD |
| Style extractive baseline as official | **NO** | Style remains diagnostic only |
""",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", required=True)
    parser.add_argument(
        "--stop_after",
        default="audit_smoke12",
        choices=[
            "audit_smoke12",
            "audit_full",
            "qa_smoke_qwen",
            "qa_full_qwen",
            "qa_smoke_llama3",
            "qa_full_llama3",
            "finalize",
        ],
    )
    parser.add_argument(
        "--resume_from",
        default="",
        choices=["", "audit_smoke12", "audit_full", "qa_smoke_qwen", "qa_full_qwen", "qa_smoke_llama3", "qa_full_llama3", "finalize"],
    )
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logs = out_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    order = [
        "audit_smoke12",
        "qa_smoke_qwen",
        "qa_smoke_llama3",
        "audit_full",
        "qa_full_qwen",
        "qa_full_llama3",
        "finalize",
    ]
    # resume_from = first stage to run (skip earlier completed stages)
    start_idx = order.index(args.resume_from) if args.resume_from else 0
    stop_idx = order.index(args.stop_after)

    ids12 = select_fixed_sample_ids(2)
    smoke_protocols = [RAW, RATE, ADAPTED_QWEN, ADAPTED_LLAMA]
    full_protocols = [RAW, ADAPTED_QWEN, ADAPTED_LLAMA]  # skip expensive rate_search on full

    smoke_summary = None
    full_summary = None

    if start_idx <= order.index("audit_smoke12") <= stop_idx:
        smoke_summary = run_compression_audit(
            out_dir, stage="audit_smoke12", sample_ids=ids12, protocols=smoke_protocols
        )
        write_audit_markdown(out_dir, smoke_summary, None)
        for p in (ADAPTED_QWEN, ADAPTED_LLAMA, RATE):
            if not smoke_summary["protocol_status"][p]["qa_eligible"]:
                raise RuntimeError(f"smoke12 {p} not QA eligible: {smoke_summary['protocol_status'][p]}")
        write_text(out_dir / "AUDIT_SMOKE12.DONE", f"{utc_now()}\n")

    if start_idx <= order.index("qa_smoke_qwen") <= stop_idx:
        run_qa_stage(out_dir, out_dir / "audit_smoke12", ADAPTED_QWEN, "qwen", ids12, "qa_smoke_qwen")
        write_text(out_dir / "QA_SMOKE_QWEN.DONE", f"{utc_now()}\n")
    if start_idx <= order.index("qa_smoke_llama3") <= stop_idx:
        run_qa_stage(out_dir, out_dir / "audit_smoke12", ADAPTED_LLAMA, "llama3", ids12, "qa_smoke_llama3")
        write_text(out_dir / "QA_SMOKE_LLAMA3.DONE", f"{utc_now()}\n")

    if start_idx <= order.index("audit_full") <= stop_idx:
        full_summary = run_compression_audit(
            out_dir, stage="audit_full", sample_ids=None, protocols=full_protocols
        )
        smoke_path = out_dir / "audit_smoke12" / "official_compression_summary.json"
        smoke_summary = smoke_summary or json.loads(smoke_path.read_text(encoding="utf-8"))
        write_audit_markdown(out_dir, smoke_summary, full_summary)
        for p in (ADAPTED_QWEN, ADAPTED_LLAMA):
            st = full_summary["protocol_status"][p]
            if not st["qa_eligible"]:
                raise RuntimeError(f"full {p} not QA eligible: {st}")
        write_token_budget_and_failure_audit(out_dir, "audit_full")
        write_text(out_dir / "AUDIT_FULL.DONE", f"{utc_now()}\n")

    if start_idx <= order.index("qa_full_qwen") <= stop_idx:
        full_ids = {t: [str(r["sample_id"]) for r in official.reference_rows(official.REFERENCE_DIR, t, -1)] for t in TASKS}
        run_qa_stage(out_dir, out_dir / "audit_full", ADAPTED_QWEN, "qwen", full_ids, "qa_full_qwen")
        write_text(out_dir / "QA_FULL_QWEN.DONE", f"{utc_now()}\n")
    if start_idx <= order.index("qa_full_llama3") <= stop_idx:
        full_ids = {t: [str(r["sample_id"]) for r in official.reference_rows(official.REFERENCE_DIR, t, -1)] for t in TASKS}
        run_qa_stage(out_dir, out_dir / "audit_full", ADAPTED_LLAMA, "llama3", full_ids, "qa_full_llama3")
        write_text(out_dir / "QA_FULL_LLAMA3.DONE", f"{utc_now()}\n")
    if start_idx <= order.index("finalize") <= stop_idx:
        write_stage2_stats(out_dir)
        write_text(out_dir / "STAGE_2.DONE", f"Stage 2 complete at {utc_now()}\n")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        out = Path(sys.argv[sys.argv.index("--out_dir") + 1]) if "--out_dir" in sys.argv else Path(".")
        write_text(out / "STAGE2_FATAL_ERROR.md", f"{exc}\n\n{traceback.format_exc()}")
        raise
