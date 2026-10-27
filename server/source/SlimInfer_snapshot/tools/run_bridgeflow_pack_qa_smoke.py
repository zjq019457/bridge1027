#!/usr/bin/env python3
"""Small-scale QA comparison: Diffusion-only Pack vs Q-BridgeFlow Pack vs BridgePack baselines.

Controlled setting:
- same LongBench samples
- same QA model
- same exact packed-context token budget (Qwen tokenizer)
- same longbench_eval.scorer

BridgeFlow arms pack passage context chunks ranked by:
- diffusion_only_pack: max-layer QK saliency mapped from SILM prefill blocks
- q_bridgeflow_pack: query-conditioned / anti-hub / persistence / bridge-aware score
- q_bridgeflow_no_wrapper_pack: same without prompt-wrapper penalty (ablation)

Baselines reuse exact-budget BridgePack repack for BM25 / Lexical-BridgePack.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.run_bridgeflow_silm_property_probe import (  # noqa: E402
    SampleContext,
    build_sample_contexts,
    check_gpu,
    compute_bridge_proxy,
    compute_neighbor_gain,
    compute_query_proxies,
    compute_redundancy_proxy,
    layer_filter,
    maybe_float,
    null_question_text,
    prompt_wrapper_penalty,
    read_jsonl,
    run_prefill_probe,
)
from tools.run_bridgepack_exact_budget_repack import (  # noqa: E402
    _exact_repack,
    _generation_prompt,
    _load_tokenizer,
    _token_count,
    _truncate_exact,
)
from tools.run_evidence_text_packing_offline import (  # noqa: E402
    _candidate_dataset_roots,
    _load_samples,
)

DEFAULT_MODEL_PATH = (
    "/home/yu-bai/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/"
    "snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
)
BRIDGEFLOW_PACK_VARIANTS = {
    "diffusion_only_pack",
    "q_bridgeflow_pack",
    "q_bridgeflow_no_wrapper_pack",
}
BRIDGEPACK_BASELINE_VARIANTS = {
    "bm25_chunk_source_order",
    "lexical_bridgepack_path_order",
}
DISPLAY = {
    "diffusion_only_pack": "Diffusion-only Pack",
    "q_bridgeflow_pack": "Q-BridgeFlow Pack",
    "q_bridgeflow_no_wrapper_pack": "Q-BridgeFlow Pack (no wrapper penalty)",
    "bm25_chunk_source_order": "BM25 Chunk Packing",
    "lexical_bridgepack_path_order": "Lexical-BridgePack Path Order",
}


def _parse_list(value: str) -> List[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _context_chunks(tokenizer: Any, context: str, block_size: int) -> List[str]:
    ids = tokenizer(context, add_special_tokens=False).input_ids
    chunks: List[str] = []
    for start in range(0, len(ids), block_size):
        piece_ids = ids[start : start + block_size]
        if not piece_ids:
            continue
        chunks.append(tokenizer.decode(piece_ids, skip_special_tokens=True))
    return chunks if chunks else [context]


def _passage_token_range(tokenizer: Any, prompt: str, context: str) -> Tuple[int, int]:
    context = str(context or "")
    if not context.strip():
        return 0, _token_count(tokenizer, prompt)
    pos = prompt.find(context)
    if pos < 0:
        marker = "The following are given passages."
        pos = prompt.find(marker)
        if pos >= 0:
            pos += len(marker)
    if pos < 0:
        return 0, _token_count(tokenizer, prompt)
    prefix = prompt[:pos]
    start = len(tokenizer(prefix, add_special_tokens=False).input_ids)
    end = start + len(tokenizer(context, add_special_tokens=False).input_ids)
    return start, end


def _overlap(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0))


def _map_context_chunks_to_prompt_blocks(
    passage_start: int,
    passage_end: int,
    num_context_chunks: int,
    num_prompt_blocks: int,
    block_size: int,
) -> Dict[int, List[int]]:
    mapping: Dict[int, List[int]] = defaultdict(list)
    for chunk_id in range(num_context_chunks):
        c0 = passage_start + chunk_id * block_size
        c1 = min(passage_end, c0 + block_size)
        for block_id in range(num_prompt_blocks):
            b0 = block_id * block_size
            b1 = b0 + block_size
            if _overlap(c0, c1, b0, b1) > 0:
                mapping[chunk_id].append(block_id)
    return mapping


def _aggregate_qk_scores(
    uasr_rows: Sequence[Mapping[str, Any]],
    start_layer: int,
    end_layer: int,
) -> Dict[int, float]:
    rows = layer_filter(uasr_rows, start_layer, end_layer)
    history: Dict[int, List[float]] = defaultdict(list)
    for row in rows:
        block_ids = [int(x) for x in row.get("all_candidate_block_ids", row.get("block_ids", []))]
        qk_scores = [maybe_float(x) for x in row.get("all_candidate_qk_scores", [])]
        for idx, block_id in enumerate(block_ids):
            if idx < len(qk_scores) and qk_scores[idx] is not None:
                history[block_id].append(float(qk_scores[idx]))
    return {block_id: max(values) for block_id, values in history.items() if values}


def _chunk_persistence_ratios(
    mapping: Mapping[int, Sequence[int]],
    uasr_rows: Sequence[Mapping[str, Any]],
    start_layer: int,
    end_layer: int,
    num_chunks: int,
) -> List[float]:
    rows = layer_filter(uasr_rows, start_layer, end_layer)
    if not rows:
        return [0.0] * num_chunks
    ratios: List[float] = []
    for chunk_id in range(num_chunks):
        prompt_blocks = set(int(x) for x in mapping.get(chunk_id, []))
        if not prompt_blocks:
            ratios.append(0.0)
            continue
        selected_layers = 0
        for row in rows:
            selected = {int(x) for x in row.get("selected_block_ids", [])}
            if selected & prompt_blocks:
                selected_layers += 1
        ratios.append(float(selected_layers / max(1, len(rows))))
    return ratios


def compute_bridgeflow_chunk_scores(
    chunk_texts: Sequence[str],
    question: str,
    null_question: str,
    empty_question: str,
    masked_question: str,
    *,
    alpha: float,
    beta: float,
    gamma: float,
    delta: float,
    lambda_hub: float,
    lambda_red: float,
    wrapper_penalty_weight: float,
    persistence_ratios: Optional[Sequence[float]] = None,
) -> List[float]:
    query_proxy = compute_query_proxies(chunk_texts, question, null_question, empty_question, masked_question)
    query_scores = query_proxy["query_score"]
    scores: List[float] = []
    for block_id in range(len(chunk_texts)):
        query_conditioned_gain = (
            query_proxy["query_conditioned_gain"][block_id]
            if block_id < len(query_proxy["query_conditioned_gain"])
            else None
        )
        hub_score = query_proxy["hub_score"][block_id] if block_id < len(query_proxy["hub_score"]) else None
        anti_hub_score = (
            query_proxy["anti_hub_score"][block_id]
            if block_id < len(query_proxy["anti_hub_score"])
            else None
        )
        neighbor_gain = compute_neighbor_gain(chunk_texts, question, block_id)
        redundancy = compute_redundancy_proxy(chunk_texts, query_scores, block_id)
        bridge_flow = compute_bridge_proxy(chunk_texts, query_scores, anti_hub_score, block_id)
        wrapper_penalty = prompt_wrapper_penalty(chunk_texts[block_id])
        persistence_ratio = 0.0
        if persistence_ratios is not None and block_id < len(persistence_ratios):
            persistence_ratio = float(persistence_ratios[block_id])
        candidate_score = 0.0
        if query_conditioned_gain is not None:
            candidate_score += alpha * query_conditioned_gain
        candidate_score += beta * persistence_ratio
        if bridge_flow is not None:
            candidate_score += gamma * bridge_flow
        if neighbor_gain is not None:
            candidate_score += delta * neighbor_gain
        if hub_score is not None:
            candidate_score -= lambda_hub * hub_score
        if redundancy is not None:
            candidate_score -= lambda_red * redundancy
        if wrapper_penalty > 0.0 and wrapper_penalty_weight > 0.0:
            candidate_score -= wrapper_penalty_weight * wrapper_penalty
        scores.append(float(candidate_score))
    return scores


def _rank_ids(scores: Sequence[float]) -> List[int]:
    return [idx for idx, _ in sorted(enumerate(scores), key=lambda item: item[1], reverse=True)]


def _pack_chunks_exact(
    tokenizer: Any,
    chunks: Sequence[str],
    ranked_ids: Sequence[int],
    budget: int,
) -> Tuple[str, List[int], bool]:
    pieces: List[str] = []
    selected: List[int] = []
    truncated = False
    for chunk_id in ranked_ids:
        if chunk_id < 0 or chunk_id >= len(chunks):
            continue
        piece = str(chunks[chunk_id]).strip()
        if not piece:
            continue
        candidate = "\n\n".join([*pieces, piece]).strip()
        if _token_count(tokenizer, candidate) <= budget:
            pieces.append(piece)
            selected.append(int(chunk_id))
            continue
        if not pieces:
            pieces = [_truncate_exact(tokenizer, piece, budget)]
            selected = [int(chunk_id)]
            truncated = True
        break
    packed = "\n\n".join(pieces).strip()
    if _token_count(tokenizer, packed) > budget:
        packed = _truncate_exact(tokenizer, packed, budget)
        truncated = True
    return packed, selected, truncated


def _packed_row_from_context_pack(
    sample: Mapping[str, Any],
    variant: str,
    budget: int,
    model_name_or_path: str,
    tokenizer: Any,
    packed_text: str,
    selected_chunk_ids: Sequence[int],
    *,
    truncated: bool,
    ranking_signal: str,
    chunk_scores: Sequence[float],
) -> Dict[str, Any]:
    task = str(sample["task"])
    question = str(sample.get("question", ""))
    context = str(sample.get("context", ""))
    model_token_count = _token_count(tokenizer, packed_text)
    source_tokens = _token_count(tokenizer, context)
    return {
        "sample_id": str(sample["sample_id"]),
        "task": task,
        "variant": variant,
        "budget": budget,
        "question": question,
        "packed_text": packed_text,
        "packed_prompt": packed_text,
        "generation_prompt": _generation_prompt(task, packed_text, question),
        "original_context": context,
        "answers": sample.get("answers", []),
        "gold_analysis_only_answers": sample.get("gold_analysis_only_answers", []),
        "all_classes": sample.get("all_classes", None),
        "length": sample.get("length", 0),
        "source_file": sample.get("source_file", ""),
        "selected_span_ids": [f"chunk_{idx}" for idx in selected_chunk_ids],
        "selected_page_ids": [],
        "packed_token_count": model_token_count,
        "model_token_count": model_token_count,
        "source_token_count": source_tokens,
        "compression_ratio": model_token_count / max(1, source_tokens),
        "truncated": truncated,
        "exact_model_token_budget": True,
        "model_tokenizer": model_name_or_path,
        "ranking_signal": ranking_signal,
        "selected_chunk_ids": list(selected_chunk_ids),
        "chunk_score_preview": [
            {"chunk_id": idx, "score": float(chunk_scores[idx])}
            for idx in selected_chunk_ids
            if 0 <= int(idx) < len(chunk_scores)
        ],
    }


@dataclass
class PackCliArgs:
    block_size: int
    start_layer: int
    end_layer: int
    null_question_mode: str
    seed: int
    alpha: float
    beta: float
    gamma: float
    delta: float
    lambda_hub: float
    lambda_red: float
    wrapper_penalty_weight: float


def build_bridgeflow_packed_row(
    sample: Mapping[str, Any],
    ctx: SampleContext,
    variant: str,
    uasr_rows: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    budget: int,
    model_name_or_path: str,
    pack_args: PackCliArgs,
) -> Dict[str, Any]:
    context = str(sample.get("context", ""))
    chunks = _context_chunks(tokenizer, context, pack_args.block_size)
    passage_start, passage_end = _passage_token_range(tokenizer, ctx.prompt, context)
    mapping = _map_context_chunks_to_prompt_blocks(
        passage_start,
        passage_end,
        len(chunks),
        len(ctx.block_texts),
        pack_args.block_size,
    )
    qk_by_block = _aggregate_qk_scores(uasr_rows, pack_args.start_layer, pack_args.end_layer)

    if variant == "diffusion_only_pack":
        chunk_scores: List[float] = []
        for chunk_id in range(len(chunks)):
            mapped = mapping.get(chunk_id, [])
            if not mapped:
                chunk_scores.append(float("-inf"))
            else:
                chunk_scores.append(max(qk_by_block.get(block_id, float("-inf")) for block_id in mapped))
        ranking_signal = "max_layer_all_candidate_qk_scores_mapped_to_context_chunks"
        ranked = _rank_ids(chunk_scores)
    else:
        wrapper_weight = 0.0 if variant == "q_bridgeflow_no_wrapper_pack" else pack_args.wrapper_penalty_weight
        persistence_ratios = _chunk_persistence_ratios(
            mapping,
            uasr_rows,
            pack_args.start_layer,
            pack_args.end_layer,
            len(chunks),
        )
        chunk_scores = compute_bridgeflow_chunk_scores(
            chunks,
            ctx.question,
            ctx.null_question,
            ctx.empty_question,
            ctx.masked_question,
            alpha=pack_args.alpha,
            beta=pack_args.beta,
            gamma=pack_args.gamma,
            delta=pack_args.delta,
            lambda_hub=pack_args.lambda_hub,
            lambda_red=pack_args.lambda_red,
            wrapper_penalty_weight=wrapper_weight,
            persistence_ratios=persistence_ratios,
        )
        ranking_signal = "q_bridgeflow_chunk_score" + ("" if wrapper_weight > 0 else "_no_wrapper_penalty")
        ranked = _rank_ids(chunk_scores)

    packed_text, selected_chunk_ids, truncated = _pack_chunks_exact(tokenizer, chunks, ranked, budget)
    return _packed_row_from_context_pack(
        sample,
        variant,
        budget,
        model_name_or_path,
        tokenizer,
        packed_text,
        selected_chunk_ids,
        truncated=truncated,
        ranking_signal=ranking_signal,
        chunk_scores=chunk_scores,
    )


def _context_by_key(contexts: Sequence[SampleContext]) -> Dict[Tuple[str, int], SampleContext]:
    return {(ctx.task, ctx.sample_index): ctx for ctx in contexts}


def _uasr_by_key(uasr_rows: Sequence[Mapping[str, Any]]) -> Dict[Tuple[str, int], List[Dict[str, Any]]]:
    grouped: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
    for row in uasr_rows:
        key = (str(row.get("task", "")), int(row.get("sample_index", -1)))
        grouped[key].append(dict(row))
    return grouped


def build_all_packed_prompts(
    out_dir: Path,
    tasks: Sequence[str],
    variants: Sequence[str],
    budget: int,
    max_samples: int,
    model_name_or_path: str,
    contexts: Sequence[SampleContext],
    uasr_rows: Sequence[Mapping[str, Any]],
    pack_args: PackCliArgs,
) -> Path:
    tokenizer = _load_tokenizer(model_name_or_path)
    roots = _candidate_dataset_roots("")
    ctx_by_key = _context_by_key(contexts)
    uasr_by_key = _uasr_by_key(uasr_rows)
    packed_root = out_dir / "packed_prompts"
    ranking_rows: List[Dict[str, Any]] = []

    for variant in variants:
        for task in tasks:
            out_path = packed_root / variant / f"budget_{budget}" / f"{task}.jsonl"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            samples = _load_samples(task, roots, max_samples)
            with out_path.open("w", encoding="utf-8") as handle:
                for sample_index, sample in enumerate(samples):
                    key = (task, sample_index)
                    ctx = ctx_by_key.get(key)
                    if ctx is None:
                        raise RuntimeError(f"missing SampleContext for {key}")
                    if variant in BRIDGEFLOW_PACK_VARIANTS:
                        payload = build_bridgeflow_packed_row(
                            sample,
                            ctx,
                            variant,
                            uasr_by_key.get(key, []),
                            tokenizer,
                            budget,
                            model_name_or_path,
                            pack_args,
                        )
                    elif variant in BRIDGEPACK_BASELINE_VARIANTS:
                        payload = _exact_repack(tokenizer, sample, variant, budget, model_name_or_path)
                        payload["ranking_signal"] = f"bridgepack::{variant}"
                    else:
                        raise ValueError(f"unsupported variant: {variant}")
                    ranking_rows.append(
                        {
                            "variant": variant,
                            "task": task,
                            "sample_id": payload.get("sample_id", ""),
                            "model_token_count": payload.get("model_token_count", ""),
                            "selected_chunk_ids": payload.get("selected_chunk_ids", ""),
                            "selected_span_ids": payload.get("selected_span_ids", ""),
                            "ranking_signal": payload.get("ranking_signal", ""),
                            "truncated": payload.get("truncated", False),
                        }
                    )
                    json.dump(payload, handle, ensure_ascii=False)
                    handle.write("\n")
    _write_csv(
        out_dir / "audits" / "01_pack_ranking_audit.csv",
        ranking_rows,
        ["variant", "task", "sample_id", "model_token_count", "selected_chunk_ids", "selected_span_ids", "ranking_signal", "truncated"],
    )
    return packed_root


def write_qa_comparison_report(out_dir: Path, variants: Sequence[str], tasks: Sequence[str], budget: int) -> None:
    breakdown = _read_csv(out_dir / "summary" / "method_task_breakdown.csv")
    summary = _read_csv(out_dir / "summary" / "summary.csv")
    by_variant_task = {
        (row.get("variant", ""), row.get("task", "")): row for row in breakdown if row.get("status") == "ok"
    }
    macro = {row.get("variant", ""): row.get("macro_avg", "") for row in summary}
    pack_audit = _read_csv(out_dir / "audits" / "01_pack_ranking_audit.csv")

    lines = [
        "# BridgeFlow Pack QA Smoke Comparison",
        "",
        f"- Budget: exact {budget} Qwen packed-context tokens",
        f"- Tasks: `{','.join(tasks)}`",
        f"- Variants: `{','.join(variants)}`",
        "",
        "## Scores (longbench_eval.scorer, 0-100)",
        "",
        "| Variant | Macro avg | " + " | ".join(tasks) + " |",
        "| --- | ---: | " + " | ".join(["---:"] * len(tasks)) + " |",
    ]
    for variant in variants:
        task_scores = [by_variant_task.get((variant, task), {}).get("score", "") for task in tasks]
        lines.append(
            f"| {DISPLAY.get(variant, variant)} | {macro.get(variant, '')} | " + " | ".join(str(v) for v in task_scores) + " |"
        )
    lines.extend(["", "## Pack audit highlights", ""])
    for variant in variants:
        rows = [row for row in pack_audit if row.get("variant") == variant]
        wrapper_like = sum(
            1
            for row in rows
            for span in str(row.get("selected_span_ids", "")).split(",")
            if "answer:" in span.lower() or "assistant" in span.lower()
        )
        mean_tokens = statistics.mean(float(row.get("model_token_count") or 0) for row in rows) if rows else 0.0
        lines.append(
            f"- **{DISPLAY.get(variant, variant)}**: mean packed tokens={mean_tokens:.1f}, samples={len(rows)}, wrapper-like selected spans={wrapper_like}"
        )
    lines.extend(
        [
            "",
            "## Interpretation guardrails",
            "",
            "- This is a small-sample answer-quality smoke, not full LongBench.",
            "- Diffusion-only Pack uses SILM QK saliency mapped onto passage chunks; Q-BridgeFlow Pack uses the cleaned ranking signal.",
            "- BM25 / Lexical-BridgePack reuse the existing exact-budget BridgePack repack path.",
        ]
    )
    (out_dir / "qa_comparison_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _release_gpu_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass


def _run_generation(
    out_dir: Path,
    packed_dir: Path,
    model_name_or_path: str,
    tasks: Sequence[str],
    variants: Sequence[str],
    budget: int,
    max_samples: int,
    resume: bool,
    dtype: str,
    device: str,
    min_free_gpu_mb: int,
) -> None:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "run_evidence_text_packing_generation_qwen32.py"),
        "--packed_dir",
        str(packed_dir),
        "--tasks",
        ",".join(tasks),
        "--max_samples",
        str(max_samples),
        "--budget",
        str(budget),
        "--variants",
        ",".join(variants),
        "--model_name_or_path",
        model_name_or_path,
        "--out_dir",
        str(out_dir),
        "--dtype",
        dtype,
        "--device",
        device,
        "--min_free_gpu_mb",
        str(min_free_gpu_mb),
    ]
    if resume:
        cmd.append("--resume")
    log_path = out_dir / "logs" / "generation.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        proc = subprocess.run(cmd, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"generation failed rc={proc.returncode}; see {log_path}")


def parse_args() -> argparse.Namespace:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name_or_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--tokenizer_path", default="")
    parser.add_argument("--longbench_dir", default="/home/yu-bai/Desktop/jiaqizhang/HeadKV/data/LongBench")
    parser.add_argument("--tasks", default="hotpotqa,musique")
    parser.add_argument("--max_samples", type=int, default=6, help="Per-task sample cap; 6+6=12 smoke.")
    parser.add_argument("--budget", type=int, default=1024)
    parser.add_argument(
        "--variants",
        default="diffusion_only_pack,q_bridgeflow_pack,q_bridgeflow_no_wrapper_pack,bm25_chunk_source_order,lexical_bridgepack_path_order",
    )
    parser.add_argument("--out_dir", default=f"results/bridgeflow_pack_qa_smoke_{stamp}")
    parser.add_argument("--block_size", type=int, default=64)
    parser.add_argument("--start_layer", type=int, default=0)
    parser.add_argument("--end_layer", type=int, default=27)
    parser.add_argument("--null_question_mode", default="shuffle_tokens", choices=["empty", "shuffle_tokens", "mask"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.8)
    parser.add_argument("--delta", type=float, default=0.3)
    parser.add_argument("--lambda_hub", type=float, default=0.6)
    parser.add_argument("--lambda_red", type=float, default=0.3)
    parser.add_argument("--wrapper_penalty_weight", type=float, default=0.8)
    parser.add_argument("--dtype", default="fp16")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--min_free_gpu_gb", type=float, default=12.0)
    parser.add_argument("--min_free_gpu_mb", type=int, default=12000)
    parser.add_argument("--skip_prefill", action="store_true")
    parser.add_argument("--pack_only", action="store_true")
    parser.add_argument("--score_only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _probe_cli_args(args: argparse.Namespace, out_dir: Path) -> argparse.Namespace:
    probe = argparse.Namespace(
        model_name_or_path=Path(args.model_name_or_path),
        tokenizer_path=Path(args.tokenizer_path) if args.tokenizer_path else None,
        longbench_dir=Path(args.longbench_dir),
        dataset="hotpotqa",
        tasks=args.tasks,
        method="sliminfer_original",
        max_samples=args.max_samples,
        sample_offset=0,
        block_size=args.block_size,
        unit_size=8,
        start_layer=args.start_layer,
        end_layer=args.end_layer,
        device="cuda",
        dtype=args.dtype,
        out_dir=out_dir,
        null_question_mode=args.null_question_mode,
        dry_run=False,
        block_strategy="fixed",
        max_model_len=32768,
        backend="flash_attention_2",
        seed=args.seed,
        min_free_gpu_gb=args.min_free_gpu_gb,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        delta=args.delta,
        lambda_hub=args.lambda_hub,
        lambda_red=args.lambda_red,
        top_k=10,
        analyze_only=False,
    )
    return probe


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = _parse_list(args.tasks)
    variants = _parse_list(args.variants)
    for variant in variants:
        if variant not in BRIDGEFLOW_PACK_VARIANTS and variant not in BRIDGEPACK_BASELINE_VARIANTS:
            raise ValueError(f"unsupported variant: {variant}")

    probe_args = _probe_cli_args(args, out_dir)
    from experiments import run_qwen25_sliminfer_variants as variants_mod

    tokenizer = variants_mod.AutoTokenizer.from_pretrained(
        str(args.tokenizer_path or args.model_name_or_path),
        trust_remote_code=True,
    )
    contexts = build_sample_contexts(probe_args, tokenizer)
    total_samples = len(contexts)
    config = {
        "model_name_or_path": str(args.model_name_or_path),
        "tasks": tasks,
        "max_samples_per_task": args.max_samples,
        "total_samples": total_samples,
        "budget": args.budget,
        "variants": variants,
        "exact_model_token_budget": True,
        "scorer": "longbench_eval.scorer",
    }
    _write_json(out_dir / "config.json", config)

    packed_root = out_dir / "packed_prompts"
    uasr_path = out_dir / "uasr_score_dump.jsonl"

    if not args.score_only:
        needs_prefill = any(variant in BRIDGEFLOW_PACK_VARIANTS for variant in variants)
        if needs_prefill and not args.skip_prefill:
            gpu_info = check_gpu(float(args.min_free_gpu_gb))
            if float(gpu_info["selected"]["memory_free_gib"]) < float(args.min_free_gpu_gb):
                raise RuntimeError(f"GPU busy: {gpu_info}")
            probe_summary = run_prefill_probe(probe_args, out_dir, contexts)
            if str(probe_summary.get("status")) != "prefill_probe_ok":
                raise RuntimeError(f"prefill failed: {probe_summary}")
            _release_gpu_memory()
            time.sleep(2)
        elif needs_prefill and not uasr_path.exists():
            raise RuntimeError("missing uasr_score_dump.jsonl; run without --skip_prefill first")

        uasr_rows = read_jsonl(uasr_path) if uasr_path.exists() else []
        pack_args = PackCliArgs(
            block_size=args.block_size,
            start_layer=args.start_layer,
            end_layer=args.end_layer,
            null_question_mode=args.null_question_mode,
            seed=args.seed,
            alpha=args.alpha,
            beta=args.beta,
            gamma=args.gamma,
            delta=args.delta,
            lambda_hub=args.lambda_hub,
            lambda_red=args.lambda_red,
            wrapper_penalty_weight=args.wrapper_penalty_weight,
        )
        build_all_packed_prompts(
            out_dir,
            tasks,
            variants,
            args.budget,
            args.max_samples,
            str(args.model_name_or_path),
            contexts,
            uasr_rows,
            pack_args,
        )
        if args.pack_only:
            print(json.dumps({"status": "pack_only_ok", "out_dir": str(out_dir), "total_samples": total_samples}, indent=2))
            return

    if not packed_root.exists():
        raise RuntimeError(f"missing packed prompts at {packed_root}")

    _run_generation(
        out_dir,
        out_dir,
        str(args.model_name_or_path),
        tasks,
        variants,
        args.budget,
        args.max_samples,
        args.resume,
        args.dtype,
        args.device,
        int(args.min_free_gpu_mb),
    )
    write_qa_comparison_report(out_dir, variants, tasks, args.budget)
    print(json.dumps({"status": "qa_smoke_ok", "out_dir": str(out_dir), "total_samples": total_samples}, indent=2))


if __name__ == "__main__":
    main()
