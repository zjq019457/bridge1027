#!/usr/bin/env python3
"""BridgeFlow property probe for SILM/SlimInfer.

This runner is intentionally lightweight:
- it reuses the existing SlimInfer/SILM loading, prompt, block, and scoring path;
- it can run a tokenizer/static `--dry_run` with no model loading;
- it can run a tiny prefill-only probe without full QA generation;
- it never edits prior official outputs or paper tables.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bridgekv.block_layout import build_blocks
from bridgekv.rescuekv_scoring import query_evidence_score
from experiments import run_qwen25_sliminfer_variants as variants


DEFAULT_MODEL_PATH = Path(
    "/home/yu-bai/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/"
    "snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
)
DEFAULT_LONGBENCH_DIR = Path("/home/yu-bai/Desktop/jiaqizhang/HeadKV/data/LongBench")
CHAT_EXCLUDED_TASKS = {"trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def read_jsonl(path: Path, *, missing_ok: bool = False) -> List[Dict[str, Any]]:
    if not path.exists():
        if missing_ok:
            return []
        raise RuntimeError(f"missing jsonl file: {path}")
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
    return rows


def append_run_command(path: Path, argv: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = " ".join(_shell_quote(str(arg)) for arg in argv)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _shell_quote(value: str) -> str:
    if not value:
        return "''"
    if re.fullmatch(r"[A-Za-z0-9_./:=,@+-]+", value):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"


def parse_tasks(value: str) -> List[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name_or_path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--tokenizer_path", type=Path, default=None)
    parser.add_argument("--longbench_dir", type=Path, default=DEFAULT_LONGBENCH_DIR)
    parser.add_argument("--dataset", type=str, default="hotpotqa")
    parser.add_argument("--tasks", type=str, default="")
    parser.add_argument("--method", type=str, default="sliminfer_original")
    parser.add_argument("--max_samples", type=int, default=2)
    parser.add_argument("--sample_offset", type=int, default=0)
    parser.add_argument("--block_size", type=int, default=64)
    parser.add_argument("--unit_size", type=int, default=8)
    parser.add_argument("--start_layer", type=int, default=0)
    parser.add_argument("--end_layer", type=int, default=27)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--out_dir", type=Path, default=None)
    parser.add_argument("--null_question_mode", type=str, default="shuffle_tokens", choices=["empty", "shuffle_tokens", "mask"])
    parser.add_argument("--no_generate", action="store_true", help="Accepted for API compatibility; this script never runs full QA generation.")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--block_strategy", type=str, default="fixed")
    parser.add_argument("--max_model_len", type=int, default=32768)
    parser.add_argument("--backend", type=str, default="flash_attention_2", choices=["flash_attention_2", "sdpa", "auto"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min_free_gpu_gb", type=float, default=12.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.8)
    parser.add_argument("--delta", type=float, default=0.3)
    parser.add_argument("--lambda_hub", type=float, default=0.6)
    parser.add_argument("--lambda_red", type=float, default=0.3)
    parser.add_argument("--top_k", type=int, default=10, help="Top-k size for ranking control comparisons.")
    parser.add_argument(
        "--analyze_only",
        action="store_true",
        help="Skip GPU prefill; rebuild contexts and re-aggregate from existing dumps in --out_dir.",
    )
    return parser.parse_args()


def default_output_dir(args: argparse.Namespace) -> Path:
    if args.out_dir is not None:
        return args.out_dir.resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (REPO_ROOT / "results" / f"bridgeflow_silm_property_probe_{timestamp}").resolve()


def dataset_list(args: argparse.Namespace) -> List[str]:
    tasks = parse_tasks(args.tasks)
    return tasks if tasks else [args.dataset]


def make_runner_args(cli_args: argparse.Namespace, out_dir: Path) -> argparse.Namespace:
    old_argv = sys.argv[:]
    try:
        sys.argv = ["run_qwen25_sliminfer_variants.py"]
        runner_args = variants.normalize_args(variants.parse_args())
    finally:
        sys.argv = old_argv
    runner_args.model_path = str(cli_args.model_name_or_path)
    runner_args.tokenizer_path = str(cli_args.tokenizer_path or cli_args.model_name_or_path)
    runner_args.method = cli_args.method
    runner_args.dtype = cli_args.dtype
    runner_args.device = cli_args.device
    runner_args.backend = cli_args.backend
    runner_args.output_dir = str(out_dir)
    runner_args.block_size = int(cli_args.block_size)
    runner_args.token_unit_size = int(cli_args.unit_size)
    runner_args.semantic_kernel_size = int(cli_args.unit_size)
    runner_args.longbench_dir = str(cli_args.longbench_dir)
    runner_args.longbench_tasks = ",".join(dataset_list(cli_args))
    runner_args.longbench_max_samples = int(cli_args.max_samples)
    runner_args.max_model_len = int(cli_args.max_model_len)
    runner_args.seed = int(cli_args.seed)
    runner_args.block_strategy = str(cli_args.block_strategy)
    runner_args.save_block_scores = "sampled"
    runner_args.eval = "gate_audit"
    runner_args.generation_max_new_tokens = 1
    runner_args.debug_used_block_audit_path = str(out_dir / "used_blocks.jsonl")
    runner_args.debug_used_block_include_full = True
    runner_args.debug_used_block_full_layers = str(cli_args.end_layer)
    runner_args.uasr_score_dump_path = str(out_dir / "uasr_score_dump.jsonl")
    runner_args.min_free_gpu_gb = float(cli_args.min_free_gpu_gb)
    return runner_args


def tokenize_words(text: str) -> List[str]:
    return [token.lower() for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9_\-']*", text or "")]


def lexical_jaccard(left: str, right: str) -> float:
    a = set(tokenize_words(left))
    b = set(tokenize_words(right))
    if not a or not b:
        return 0.0
    return float(len(a & b) / len(a | b))


def mean(values: Iterable[float]) -> float:
    vals = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(sum(vals) / len(vals)) if vals else 0.0


def entropy_from_scores(scores: Sequence[float]) -> Optional[float]:
    finite = [max(0.0, float(x)) for x in scores if x is not None and math.isfinite(float(x))]
    if not finite:
        return None
    total = sum(finite)
    if total <= 1e-12:
        return 0.0
    probs = [value / total for value in finite if value > 0]
    return float(-sum(p * math.log(max(p, 1e-12)) for p in probs))


def maybe_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def null_question_text(question: str, mode: str, seed: int) -> str:
    text = (question or "").strip()
    if mode == "empty":
        return ""
    if mode == "mask":
        return " ".join(["[MASK]"] * max(1, len(tokenize_words(text))))
    tokens = text.split()
    if len(tokens) <= 1:
        return ""
    rng = random.Random(int(seed))
    shuffled = list(tokens)
    rng.shuffle(shuffled)
    shuffled_text = " ".join(shuffled)
    return "" if shuffled_text == text else shuffled_text


def load_prompt_templates() -> Dict[str, str]:
    return variants.load_json(REPO_ROOT / "configs" / "dataset2prompt.json")


def build_prompt(tokenizer: Any, sample: Dict[str, Any], dataset: str, max_model_len: int, prompt_templates: Mapping[str, str]) -> str:
    prompt_format = prompt_templates[dataset]
    prompt = prompt_format.format(**sample)
    tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
    if len(tokenized_prompt) > max_model_len:
        half = max_model_len // 2
        prompt = (
            tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)
            + tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        )
    if dataset not in CHAT_EXCLUDED_TASKS:
        prompt = variants.build_chat(tokenizer, prompt)
    return prompt


@dataclass
class SampleContext:
    task: str
    sample_index: int
    sample_id: Any
    sample: Dict[str, Any]
    question: str
    null_question: str
    empty_question: str
    masked_question: str
    prompt: str
    context_length: int
    block_texts: List[str]
    block_layout: Dict[str, Any]


def build_sample_contexts(cli_args: argparse.Namespace, tokenizer: Any) -> List[SampleContext]:
    prompt_templates = load_prompt_templates()
    selected: List[SampleContext] = []
    for task in dataset_list(cli_args):
        samples = variants.load_longbench_samples(Path(cli_args.longbench_dir), task)
        subset = samples[int(cli_args.sample_offset): int(cli_args.sample_offset) + int(cli_args.max_samples)]
        for relative_idx, sample in enumerate(subset):
            sample_index = int(cli_args.sample_offset) + relative_idx
            prompt = build_prompt(tokenizer, sample, task, int(cli_args.max_model_len), prompt_templates)
            model_input = tokenizer(prompt, truncation=False, return_tensors="pt")
            input_ids = model_input.input_ids
            question = variants._sample_query_text(sample if isinstance(sample, dict) else {})
            block_texts = variants.build_prompt_block_texts(tokenizer, input_ids, int(cli_args.block_size))
            block_layout = build_blocks(
                input_ids[0],
                tokenizer,
                sample if isinstance(sample, dict) else {},
                prompt_text=prompt,
                task_name=task,
                block_strategy=cli_args.block_strategy,
                target_block_tokens=cli_args.block_size,
                min_block_tokens=max(16, cli_args.block_size // 2),
                max_block_tokens=max(cli_args.block_size, cli_args.block_size * 2),
                query_text=question,
                config={
                    "block_size": cli_args.block_size,
                    "fixed_page_tokens": cli_args.block_size,
                    "semantic_parser": "task_aware",
                    "semantic_target_block_tokens": cli_args.block_size,
                    "semantic_min_block_tokens": max(16, cli_args.block_size // 2),
                    "semantic_max_block_tokens": max(cli_args.block_size, cli_args.block_size * 2),
                    "semantic_page_align": 1,
                    "max_page_expansion_overhead": 2.0,
                    "query_top_ratio": 0.2,
                    "query_neighbor_sentences": 1,
                    "query_high_max_block_tokens": cli_args.block_size,
                    "query_low_max_block_tokens": max(cli_args.block_size, cli_args.block_size * 2),
                    "query_score_mode": "lexical_bm25",
                    "query_parser": "task_aware",
                },
            ).to_dict()
            selected.append(
                SampleContext(
                    task=task,
                    sample_index=sample_index,
                    sample_id=sample.get("_id", sample.get("id", sample_index)),
                    sample=sample,
                    question=question,
                    null_question=null_question_text(question, cli_args.null_question_mode, cli_args.seed + sample_index),
                    empty_question=null_question_text(question, "empty", cli_args.seed + sample_index),
                    masked_question=null_question_text(question, "mask", cli_args.seed + sample_index),
                    prompt=prompt,
                    context_length=int(input_ids.shape[1]),
                    block_texts=block_texts,
                    block_layout=block_layout,
                )
            )
    return selected


def write_dry_run_summary(out_dir: Path, cli_args: argparse.Namespace, contexts: Sequence[SampleContext]) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for ctx in contexts:
        rows.append(
            {
                "task": ctx.task,
                "sample_index": ctx.sample_index,
                "sample_id": ctx.sample_id,
                "question_present": bool(ctx.question.strip()),
                "null_question_mode": cli_args.null_question_mode,
                "prompt_tokens": ctx.context_length,
                "block_count": len(ctx.block_texts),
                "block_size": cli_args.block_size,
                "can_probe_without_model": True,
                "likely_model_probe_feasible": ctx.context_length > 0 and len(ctx.block_texts) > 0,
            }
        )
    summary = {
        "status": "dry_run_ok" if rows else "dry_run_no_samples",
        "sample_count": len(rows),
        "tasks": dataset_list(cli_args),
        "rows": rows,
    }
    write_json(out_dir / "dry_run_summary.json", summary)
    write_csv(
        out_dir / "dry_run_summary.csv",
        rows,
        ["task", "sample_index", "sample_id", "question_present", "null_question_mode", "prompt_tokens", "block_count", "block_size", "can_probe_without_model", "likely_model_probe_feasible"],
    )
    return summary


def check_gpu(min_free_gpu_gb: float) -> Dict[str, Any]:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free", "--format=csv,noheader,nounits"],
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip().splitlines()
    rows: List[Dict[str, Any]] = []
    for line in output:
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 4:
            continue
        rows.append(
            {
                "index": int(parts[0]),
                "name": parts[1],
                "memory_total_mib": float(parts[2]),
                "memory_free_mib": float(parts[3]),
                "memory_free_gib": float(parts[3]) / 1024.0,
            }
        )
    if not rows:
        raise RuntimeError("nvidia-smi returned no GPU rows")
    selected = max(rows, key=lambda item: item["memory_free_gib"])
    return {"rows": rows, "selected": selected, "min_free_gpu_gb": float(min_free_gpu_gb)}


def write_gpu_blocked(out_dir: Path, gpu_info: Mapping[str, Any]) -> None:
    selected = dict(gpu_info.get("selected", {}))
    lines = [
        "# BLOCKED: GPU Busy Or Insufficient Memory",
        "",
        f"- Selected GPU index: {selected.get('index', '')}",
        f"- Selected GPU name: {selected.get('name', '')}",
        f"- Free memory (GiB): {selected.get('memory_free_gib', '')}",
        f"- Required minimum free GPU memory (GiB): {gpu_info.get('min_free_gpu_gb', '')}",
        "",
        "Model execution was skipped before loading the model, as requested.",
    ]
    (out_dir / "BLOCKED_GPU_BUSY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def layer_filter(rows: Sequence[Mapping[str, Any]], start_layer: int, end_layer: int) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    for row in rows:
        layer_id = int(row.get("layer_id", row.get("layer_idx", -1)))
        phase = str(row.get("phase", ""))
        if phase != "prefill":
            continue
        if layer_id < int(start_layer) or layer_id > int(end_layer):
            continue
        selected.append(dict(row))
    return selected


def run_prefill_probe(cli_args: argparse.Namespace, out_dir: Path, contexts: Sequence[SampleContext]) -> Dict[str, Any]:
    runner_args = make_runner_args(cli_args, out_dir)
    results: List[Dict[str, Any]] = []
    for path in [out_dir / "used_blocks.jsonl", out_dir / "uasr_score_dump.jsonl", out_dir / "block_scores.jsonl"]:
        path.write_text("", encoding="utf-8")
    gpu_info = check_gpu(float(cli_args.min_free_gpu_gb))
    if float(gpu_info["selected"]["memory_free_gib"]) < float(cli_args.min_free_gpu_gb):
        write_gpu_blocked(out_dir, gpu_info)
        return {
            "status": "blocked_gpu_busy",
            "actual_attn_impl": "",
            "gpu_info": gpu_info,
            "profile": {},
            "notes": ["GPU check failed before model loading."],
        }

    model, tokenizer, cache_classes, actual_attn = variants.load_model_and_tokenizer(runner_args, attn_impl=runner_args.backend)
    cfg, profile, notes = variants.build_sliminfer_config(runner_args, out_dir, int(model.config.num_hidden_layers))
    SlimInferCache, HsCache = cache_classes

    for ctx in contexts:
        model_input = tokenizer(ctx.prompt, truncation=False, return_tensors="pt").to(model.device)
        sample_cfg = dict(cfg)
        sample_cfg["audit_context"] = {
            "task": ctx.task,
            "sample_id": ctx.sample_id,
            "sample_index": ctx.sample_index,
            "context_length": ctx.context_length,
            "method": cli_args.method,
            "query_text": ctx.question,
            "block_texts": ctx.block_texts,
            "prompt_text_preview": ctx.prompt[:512],
        }
        sample_cfg["query_text"] = ctx.question
        sample_cfg["block_texts"] = ctx.block_texts
        sample_cfg["block_layout"] = ctx.block_layout
        sample_cfg["phase"] = "prefill"

        if runner_args.method == "bridgekv":
            variants.initialize_bridgekv_state(runner_args, tokenizer, model_input.input_ids, sample_cfg)

        slim_cache = SlimInferCache(model.device)
        slim_cache.debug_used_block_audit_path = str(out_dir / "used_blocks.jsonl")
        slim_cache.debug_used_block_include_full = True
        slim_cache.debug_used_block_full_layers = {int(cli_args.end_layer)}
        slim_cache.debug_context = {
            "task": ctx.task,
            "sample_id": ctx.sample_id,
            "sample_index": ctx.sample_index,
            "method": cli_args.method,
            "block_strategy": cli_args.block_strategy,
            "rescue_policy": runner_args.rescue_policy,
            "rescue_score": runner_args.rescue_score,
            "seed": cli_args.seed,
        }
        kwargs: Dict[str, Any] = {
            "past_key_values": slim_cache,
            "hs_cache": HsCache(),
            "sliminfer_config": sample_cfg,
            "all_original_prompt_indices_abs": list(range(ctx.context_length)),
            "use_cache": True,
        }
        with torch.no_grad():
            variants.reset_runtime_stats(sample_cfg)
            outputs = model(
                input_ids=model_input.input_ids,
                attention_mask=model_input.attention_mask,
                **kwargs,
            )
        output_shape: List[int] = []
        if hasattr(outputs, "logits") and getattr(outputs, "logits") is not None:
            output_shape = list(outputs.logits.shape)
        elif hasattr(outputs, "last_hidden_state") and getattr(outputs, "last_hidden_state") is not None:
            output_shape = list(outputs.last_hidden_state.shape)
        results.append(
            {
                "task": ctx.task,
                "sample_index": ctx.sample_index,
                "sample_id": ctx.sample_id,
                "context_length": ctx.context_length,
                "model_output_shape": output_shape,
            }
        )
    write_json(
        out_dir / "prefill_probe_summary.json",
        {
            "status": "prefill_probe_ok",
            "actual_attn_impl": actual_attn,
            "gpu_info": gpu_info,
            "profile": profile,
            "notes": notes,
            "sample_results": results,
        },
    )
    return {
        "status": "prefill_probe_ok",
        "actual_attn_impl": actual_attn,
        "gpu_info": gpu_info,
        "profile": profile,
        "notes": notes,
        "sample_results": results,
    }


def compute_query_scores(block_texts: Sequence[str], question: str) -> List[Optional[float]]:
    if not block_texts:
        return []
    scores, _ = query_evidence_score(
        block_texts=list(block_texts),
        query_text=question,
        inactive_ids=list(range(len(block_texts))),
        block_reps=None,
        query_rep=None,
        num_blocks=len(block_texts),
        diagnostics={"warnings": []},
    )
    values = scores.detach().float().cpu().tolist()
    return [maybe_float(value) for value in values]


def normalize_score_list(values: Sequence[Optional[float]]) -> List[Optional[float]]:
    finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not finite:
        return [None for _ in values]
    lo = min(finite)
    hi = max(finite)
    denom = max(hi - lo, 1e-6)
    return [
        None if value is None else float((float(value) - lo) / denom)
        for value in values
    ]


def block_genericity(block_texts: Sequence[str], block_id: int) -> Optional[float]:
    if not (0 <= block_id < len(block_texts)):
        return None
    current = block_texts[block_id]
    other_scores = [
        lexical_jaccard(current, block_texts[idx])
        for idx in range(len(block_texts))
        if idx != block_id
    ]
    if not other_scores:
        return 0.0
    topk = sorted(other_scores, reverse=True)[: min(3, len(other_scores))]
    return float(sum(topk) / len(topk))


PROMPT_WRAPPER_PATTERNS = [
    "system",
    "user",
    "assistant",
    "answer:",
    "question:",
    "only give me the answer",
    "you are qwen",
    "helpful assistant",
    "the following are given passages",
]


def prompt_wrapper_penalty(block_text: str) -> float:
    text = (block_text or "").lower()
    if not text.strip():
        return 0.0
    hits = 0
    for pattern in PROMPT_WRAPPER_PATTERNS:
        if pattern in text:
            hits += 1
    penalty = min(1.0, 0.22 * hits)
    if "answer:" in text and "assistant" in text:
        penalty = min(1.0, penalty + 0.25)
    if text.startswith("system") or text.startswith("user") or text.startswith("assistant"):
        penalty = min(1.0, penalty + 0.15)
    return float(penalty)


def rank_positions(scores: Sequence[Optional[float]]) -> Dict[int, int]:
    ranked = sorted(
        [(idx, float(score)) for idx, score in enumerate(scores) if score is not None],
        key=lambda item: item[1],
        reverse=True,
    )
    return {idx: rank for rank, (idx, _) in enumerate(ranked, 1)}


def compute_query_proxies(
    block_texts: Sequence[str],
    question: str,
    null_question: str,
    empty_question: str,
    masked_question: str,
) -> Dict[str, List[Optional[float]]]:
    real_scores_raw = compute_query_scores(block_texts, question)
    null_scores_raw = compute_query_scores(block_texts, null_question)
    empty_scores_raw = compute_query_scores(block_texts, empty_question)
    masked_scores_raw = compute_query_scores(block_texts, masked_question)

    real_scores = normalize_score_list(real_scores_raw)
    null_scores = normalize_score_list(null_scores_raw)
    empty_scores = normalize_score_list(empty_scores_raw)
    masked_scores = normalize_score_list(masked_scores_raw)

    real_rank = rank_positions(real_scores)
    null_rank = rank_positions(null_scores)
    empty_rank = rank_positions(empty_scores)
    masked_rank = rank_positions(masked_scores)

    query_gain_scores: List[Optional[float]] = []
    hub_scores: List[Optional[float]] = []
    anti_hub_scores: List[Optional[float]] = []

    n_blocks = len(block_texts)
    for block_id in range(n_blocks):
        real = real_scores[block_id] if block_id < len(real_scores) else None
        null_v = null_scores[block_id] if block_id < len(null_scores) else None
        empty_v = empty_scores[block_id] if block_id < len(empty_scores) else None
        masked_v = masked_scores[block_id] if block_id < len(masked_scores) else None
        controls = [value for value in [null_v, empty_v, masked_v] if value is not None]
        control_mean = float(sum(controls) / len(controls)) if controls else None

        genericity = block_genericity(block_texts, block_id)
        rank_terms: List[float] = []
        if block_id in null_rank:
            rank_terms.append(1.0 / float(null_rank[block_id]))
        if block_id in empty_rank:
            rank_terms.append(1.0 / float(empty_rank[block_id]))
        if block_id in masked_rank:
            rank_terms.append(1.0 / float(masked_rank[block_id]))
        rank_hub = float(sum(rank_terms) / len(rank_terms)) if rank_terms else 0.0

        hub_score = None
        if control_mean is not None:
            hub_score = float(0.7 * control_mean + 0.2 * rank_hub + 0.1 * (genericity or 0.0))

        query_gain = None
        if real is not None:
            if control_mean is not None:
                margin = float(real - control_mean)
            else:
                margin = float(real)
            genericity_penalty = 0.15 * float(genericity or 0.0)
            real_rank_bonus = 0.0
            if block_id in real_rank:
                real_rank_bonus = 1.0 / float(real_rank[block_id])
            query_gain = float(margin + 0.10 * real_rank_bonus - genericity_penalty)

        anti_hub = None
        if hub_score is not None:
            anti_hub = float(max(0.0, 1.0 - hub_score))

        query_gain_scores.append(query_gain)
        hub_scores.append(hub_score)
        anti_hub_scores.append(anti_hub)

    return {
        "query_score_raw": real_scores_raw,
        "null_question_score_raw": null_scores_raw,
        "empty_question_score_raw": empty_scores_raw,
        "masked_question_score_raw": masked_scores_raw,
        "query_score": real_scores,
        "null_question_score": null_scores,
        "empty_question_score": empty_scores,
        "masked_question_score": masked_scores,
        "query_conditioned_gain": query_gain_scores,
        "hub_score": hub_scores,
        "anti_hub_score": anti_hub_scores,
    }


def compute_neighbor_gain(block_texts: Sequence[str], question: str, block_id: int) -> Optional[float]:
    if not (0 <= block_id < len(block_texts)):
        return None
    base_scores = compute_query_scores([block_texts[block_id]], question)
    left = block_texts[block_id - 1] if block_id - 1 >= 0 else ""
    right = block_texts[block_id + 1] if block_id + 1 < len(block_texts) else ""
    merged = " ".join([left, block_texts[block_id], right]).strip()
    merged_scores = compute_query_scores([merged], question)
    base = base_scores[0] if base_scores else None
    merged_score = merged_scores[0] if merged_scores else None
    if base is None or merged_score is None:
        return None
    return float(merged_score - base)


def compute_redundancy_proxy(block_texts: Sequence[str], query_scores: Sequence[Optional[float]], block_id: int) -> Optional[float]:
    if not (0 <= block_id < len(block_texts)):
        return None
    ranked = sorted(
        [(idx, score) for idx, score in enumerate(query_scores) if score is not None],
        key=lambda item: float(item[1]),
        reverse=True,
    )
    anchor = next((idx for idx, _ in ranked if idx != block_id), None)
    if anchor is None:
        return None
    return lexical_jaccard(block_texts[block_id], block_texts[anchor])


def jaccard_sets(left: Sequence[int], right: Sequence[int]) -> float:
    a = set(int(x) for x in left)
    b = set(int(x) for x in right)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return float(len(a & b) / len(a | b))


def wrapper_hit(text: str) -> bool:
    return prompt_wrapper_penalty(text) > 0.0


def block_feature_lookup(
    feature_rows: Sequence[Mapping[str, Any]],
    sample_id: str,
    task: str,
    block_id: int,
) -> Dict[str, Any]:
    for row in feature_rows:
        if row.get("sample_id") == sample_id and row.get("task") == task and int(row.get("block_id", -1)) == int(block_id):
            return dict(row)
    return {}


def per_sample_block_features(
    feature_rows: Sequence[Mapping[str, Any]],
    sample_id: str,
    task: str,
) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for row in feature_rows:
        if row.get("sample_id") != sample_id or row.get("task") != task:
            continue
        block_id = int(row.get("block_id", -1))
        if block_id < 0:
            continue
        if block_id not in out:
            out[block_id] = dict(row)
    return out


def compute_bridge_proxy(
    block_texts: Sequence[str],
    query_scores: Sequence[Optional[float]],
    anti_hub_score: Optional[float],
    block_id: int,
) -> Optional[float]:
    if not (0 <= block_id < len(block_texts)):
        return None
    ranked = sorted(
        [(idx, score) for idx, score in enumerate(query_scores) if score is not None and idx != block_id],
        key=lambda item: float(item[1]),
        reverse=True,
    )[:3]
    if not ranked:
        return None
    link_strength = max(lexical_jaccard(block_texts[block_id], block_texts[idx]) for idx, _ in ranked)
    anti_hub = anti_hub_score if anti_hub_score is not None else 0.0
    q = query_scores[block_id] if block_id < len(query_scores) else None
    if q is None:
        return None
    return float(max(0.0, q) * (0.5 + 0.5 * anti_hub) * link_strength)


def aggregate_sample_rows(
    cli_args: argparse.Namespace,
    contexts: Sequence[SampleContext],
    uasr_rows: Sequence[Mapping[str, Any]],
    used_rows: Sequence[Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    context_by_key = {(ctx.task, ctx.sample_index): ctx for ctx in contexts}
    score_rows = layer_filter(uasr_rows, cli_args.start_layer, cli_args.end_layer)
    used_rows_prefill = layer_filter(used_rows, cli_args.start_layer, cli_args.end_layer)
    used_by_sample_layer: Dict[Tuple[str, int, int], Dict[str, Any]] = {}
    for row in used_rows_prefill:
        used_by_sample_layer[(str(row.get("task", "")), int(row.get("sample_index", -1)), int(row.get("layer_id", -1)))] = dict(row)

    rows_out: List[Dict[str, Any]] = []
    top_silm: List[Dict[str, Any]] = []
    top_bridge: List[Dict[str, Any]] = []
    top_diffusion: List[Dict[str, Any]] = []
    sample_summary: Dict[str, Any] = {
        "sample_count": 0,
        "blocked": False,
        "difference_count": 0,
        "diffusion_bridge_difference_count": 0,
        "control_samples": [],
    }
    top_k = max(1, int(cli_args.top_k))

    grouped: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
    for row in score_rows:
        grouped[(str(row.get("task", "")), int(row.get("sample_index", -1)))].append(dict(row))

    for key, rows in grouped.items():
        ctx = context_by_key.get(key)
        if ctx is None:
            continue
        rows.sort(key=lambda item: int(item.get("layer_id", item.get("layer_idx", -1))))
        scoring_kind = str(rows[0].get("scoring", rows[0].get("fairness_base_scoring", "unknown")))
        sample_summary["sample_count"] += 1
        query_proxy = compute_query_proxies(
            ctx.block_texts,
            ctx.question,
            ctx.null_question,
            ctx.empty_question,
            ctx.masked_question,
        )
        query_scores = query_proxy["query_score"]
        null_scores = query_proxy["null_question_score"]
        num_blocks = len(ctx.block_texts)
        score_history: Dict[int, List[Tuple[int, Optional[float], bool, Optional[int]]]] = defaultdict(list)
        qk_history: Dict[int, List[Tuple[int, Optional[float]]]] = defaultdict(list)
        for row in rows:
            layer = int(row.get("layer_id", row.get("layer_idx", -1)))
            block_ids = [int(x) for x in row.get("all_candidate_block_ids", row.get("block_ids", []))]
            scores = [maybe_float(x) for x in row.get("all_candidate_scores", row.get("scores", []))]
            qk_scores = [maybe_float(x) for x in row.get("all_candidate_qk_scores", [])]
            selected_ids = {int(x) for x in row.get("selected_block_ids", [])}
            rank_map = {int(block_id): rank for rank, block_id in enumerate(row.get("sorted_block_ids", []), 1)}
            for idx, block_id in enumerate(block_ids):
                score = scores[idx] if idx < len(scores) else None
                qk_score = qk_scores[idx] if idx < len(qk_scores) else None
                score_history[block_id].append((layer, score, block_id in selected_ids, rank_map.get(block_id)))
                qk_history[block_id].append((layer, qk_score))

        silm_aggregate: List[Tuple[int, float]] = []
        bridge_aggregate: List[Tuple[int, float]] = []
        diffusion_aggregate: List[Tuple[int, float]] = []
        block_bridge_scores: Dict[int, float] = {}
        block_meta: Dict[int, Dict[str, Any]] = {}

        for block_id in range(num_blocks):
            history = sorted(score_history.get(block_id, []), key=lambda item: item[0])
            selected_count = sum(1 for _, _, selected, _ in history if selected)
            persistence_ratio = float(selected_count / max(1, len(rows)))
            finite_scores = [(layer, score) for layer, score, _, _ in history if score is not None]
            onset_layer = next((layer for layer, _, selected, _ in history if selected), None)
            peak_layer = None
            if finite_scores:
                peak_layer = max(finite_scores, key=lambda item: float(item[1]))[0]
            first_mean = mean(score for _, score, _, _ in history[: max(1, len(history) // 2)])
            last_mean = mean(score for _, score, _, _ in history[max(0, len(history) // 2):])
            directional_in = max(0.0, last_mean - first_mean)
            directional_out = max(0.0, first_mean - last_mean)
            query_score = query_scores[block_id] if block_id < len(query_scores) else None
            null_score = null_scores[block_id] if block_id < len(null_scores) else None
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
            neighbor_gain = compute_neighbor_gain(ctx.block_texts, ctx.question, block_id)
            redundancy = compute_redundancy_proxy(ctx.block_texts, query_scores, block_id)
            bridge_flow = compute_bridge_proxy(ctx.block_texts, query_scores, anti_hub_score, block_id)
            wrapper_penalty = prompt_wrapper_penalty(ctx.block_texts[block_id] if block_id < len(ctx.block_texts) else "")
            available_terms: List[str] = []
            candidate_score = 0.0
            if query_conditioned_gain is not None:
                candidate_score += cli_args.alpha * query_conditioned_gain
                available_terms.append("query_conditioned_gain")
            candidate_score += cli_args.beta * persistence_ratio
            available_terms.append("persistence_ratio")
            if bridge_flow is not None:
                candidate_score += cli_args.gamma * bridge_flow
                available_terms.append("bridge_flow_proxy")
            if neighbor_gain is not None:
                candidate_score += cli_args.delta * neighbor_gain
                available_terms.append("neighbor_gain_proxy")
            if hub_score is not None:
                candidate_score -= cli_args.lambda_hub * hub_score
                available_terms.append("hub_score")
            if redundancy is not None:
                candidate_score -= cli_args.lambda_red * redundancy
                available_terms.append("redundancy_proxy")
            if wrapper_penalty > 0.0:
                candidate_score -= 0.8 * wrapper_penalty
                available_terms.append("prompt_wrapper_penalty")

            if finite_scores:
                silm_aggregate.append((block_id, max(float(score) for _, score in finite_scores if score is not None)))
            bridge_aggregate.append((block_id, float(candidate_score)))
            block_bridge_scores[block_id] = float(candidate_score)

            qk_finite = [(layer, score) for layer, score in qk_history.get(block_id, []) if score is not None]
            diffusion_score = None
            if qk_finite:
                diffusion_score = max(float(score) for _, score in qk_finite)
                diffusion_aggregate.append((block_id, diffusion_score))

            block_meta[block_id] = {
                "query_conditioned_gain": query_conditioned_gain,
                "hub_score": hub_score,
                "persistence_ratio": persistence_ratio,
                "bridge_flow_proxy": bridge_flow,
                "wrapper_penalty": wrapper_penalty,
                "text_preview": ctx.block_texts[block_id][:240] if block_id < len(ctx.block_texts) else "",
            }

            for layer, score, selected, rank in history:
                layer_row = next((row for row in rows if int(row.get("layer_id", row.get("layer_idx", -1))) == layer), None)
                layer_entropy = None
                if layer_row is not None:
                    layer_scores = [maybe_float(x) for x in layer_row.get("all_candidate_scores", layer_row.get("scores", []))]
                    layer_entropy = entropy_from_scores([x for x in layer_scores if x is not None])
                layer_qk = next((qk for lyr, qk in qk_history.get(block_id, []) if lyr == layer), None)
                used_row = used_by_sample_layer.get((ctx.task, ctx.sample_index, layer), {})
                rows_out.append(
                    {
                        "sample_id": ctx.sample_id,
                        "task": ctx.task,
                        "question": ctx.question,
                        "block_id": block_id,
                        "span_id": "",
                        "token_start": block_id * cli_args.block_size,
                        "token_end": min(ctx.context_length, (block_id + 1) * cli_args.block_size),
                        "text_preview": ctx.block_texts[block_id][:240] if block_id < len(ctx.block_texts) else "",
                        "layer": layer,
                        "raw_diffusion_score_or_proxy": layer_qk if layer_qk is not None else "",
                        "diffusion_only_aggregate_score": diffusion_score if diffusion_score is not None else "",
                        "selected_by_silm": int(bool(selected)),
                        "rank_by_silm": rank if rank is not None else "",
                        "query_score": query_score if query_score is not None else "",
                        "null_question_score": null_score if null_score is not None else "",
                        "query_conditioned_gain": query_conditioned_gain if query_conditioned_gain is not None else "",
                        "persistence_count": selected_count,
                        "persistence_ratio": persistence_ratio,
                        "onset_layer": onset_layer if onset_layer is not None else "",
                        "peak_layer": peak_layer if peak_layer is not None else "",
                        "propagation_entropy_or_proxy": layer_entropy if layer_entropy is not None else "",
                        "hub_score": hub_score if hub_score is not None else "",
                        "anti_hub_score": anti_hub_score if anti_hub_score is not None else "",
                        "directional_inflow_proxy": directional_in,
                        "directional_outflow_proxy": directional_out,
                        "bridge_flow_proxy": bridge_flow if bridge_flow is not None else "",
                        "neighbor_gain_proxy": neighbor_gain if neighbor_gain is not None else "",
                        "redundancy_proxy": redundancy if redundancy is not None else "",
                        "prompt_wrapper_penalty": wrapper_penalty,
                        "new_bridgeflow_score": candidate_score,
                        "selected_by_bridgeflow_score": "",
                        "selected_by_diffusion_only_score": "",
                        "feature_terms_used": ",".join(available_terms),
                        "used_blocks_layer_recorded": int(bool(used_row)),
                        "diffusion_score_kind": scoring_kind,
                    }
                )

        silm_aggregate.sort(key=lambda item: item[1], reverse=True)
        bridge_aggregate.sort(key=lambda item: item[1], reverse=True)
        diffusion_aggregate.sort(key=lambda item: item[1], reverse=True)
        top_silm_ids = [block_id for block_id, _ in silm_aggregate[:top_k]]
        top_bridge_ids = [block_id for block_id, _ in bridge_aggregate[:top_k]]
        top_diffusion_ids = [block_id for block_id, _ in diffusion_aggregate[:top_k]]
        if top_silm_ids != top_bridge_ids:
            sample_summary["difference_count"] += 1
        if top_diffusion_ids != top_bridge_ids:
            sample_summary["diffusion_bridge_difference_count"] += 1

        def topk_stats(block_ids: Sequence[int]) -> Dict[str, Any]:
            wrappers = 0
            hubs: List[float] = []
            qcg: List[float] = []
            persistence: List[float] = []
            for block_id in block_ids:
                meta = block_meta.get(int(block_id), {})
                if wrapper_hit(str(meta.get("text_preview", ""))):
                    wrappers += 1
                hub = maybe_float(meta.get("hub_score"))
                if hub is not None:
                    hubs.append(hub)
                gain = maybe_float(meta.get("query_conditioned_gain"))
                if gain is not None:
                    qcg.append(gain)
                persistence.append(float(meta.get("persistence_ratio", 0.0) or 0.0))
            return {
                "wrapper_hits": wrappers,
                "mean_hub_score": mean(hubs),
                "mean_query_conditioned_gain": mean(qcg),
                "mean_persistence_ratio": mean(persistence),
            }

        diffusion_stats = topk_stats(top_diffusion_ids)
        bridge_stats = topk_stats(top_bridge_ids)
        only_diffusion = [bid for bid in top_diffusion_ids if bid not in set(top_bridge_ids)]
        only_bridge = [bid for bid in top_bridge_ids if bid not in set(top_diffusion_ids)]
        sample_summary["control_samples"].append(
            {
                "sample_id": ctx.sample_id,
                "task": ctx.task,
                "question": ctx.question,
                "scoring_kind": scoring_kind,
                "top_k": top_k,
                "top_diffusion_ids": top_diffusion_ids,
                "top_bridge_ids": top_bridge_ids,
                "top_silm_ids": top_silm_ids,
                "jaccard_diffusion_bridge": jaccard_sets(top_diffusion_ids, top_bridge_ids),
                "jaccard_silm_bridge": jaccard_sets(top_silm_ids, top_bridge_ids),
                "only_diffusion_topk": only_diffusion,
                "only_bridgeflow_topk": only_bridge,
                "diffusion_topk_stats": diffusion_stats,
                "bridgeflow_topk_stats": bridge_stats,
            }
        )

        for rank, (block_id, score) in enumerate(silm_aggregate[:top_k], 1):
            top_silm.append(
                {
                    "sample_id": ctx.sample_id,
                    "task": ctx.task,
                    "rank": rank,
                    "block_id": block_id,
                    "score": score,
                    "text_preview": ctx.block_texts[block_id][:240] if block_id < len(ctx.block_texts) else "",
                }
            )
        selected_bridge_set = set(top_bridge_ids)
        selected_diffusion_set = set(top_diffusion_ids)
        for row in rows_out:
            if row["sample_id"] == ctx.sample_id and row["task"] == ctx.task:
                if row["block_id"] in selected_bridge_set:
                    row["selected_by_bridgeflow_score"] = 1
                if row["block_id"] in selected_diffusion_set:
                    row["selected_by_diffusion_only_score"] = 1
        for rank, (block_id, score) in enumerate(bridge_aggregate[:top_k], 1):
            top_bridge.append(
                {
                    "sample_id": ctx.sample_id,
                    "task": ctx.task,
                    "rank": rank,
                    "block_id": block_id,
                    "score": score,
                    "text_preview": ctx.block_texts[block_id][:240] if block_id < len(ctx.block_texts) else "",
                }
            )
        for rank, (block_id, score) in enumerate(diffusion_aggregate[:top_k], 1):
            top_diffusion.append(
                {
                    "sample_id": ctx.sample_id,
                    "task": ctx.task,
                    "rank": rank,
                    "block_id": block_id,
                    "score": score,
                    "score_kind": scoring_kind,
                    "text_preview": ctx.block_texts[block_id][:240] if block_id < len(ctx.block_texts) else "",
                }
            )
    return rows_out, top_silm, top_bridge, top_diffusion, sample_summary


def write_diffusion_control_report(
    out_dir: Path,
    cli_args: argparse.Namespace,
    contexts: Sequence[SampleContext],
    feature_rows: Sequence[Mapping[str, Any]],
    diffusion_rows: Sequence[Mapping[str, Any]],
    bridge_rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> None:
    top_k = max(1, int(cli_args.top_k))
    control_samples = list(summary.get("control_samples", []))
    context_by_id = {(str(ctx.sample_id), ctx.task): ctx for ctx in contexts}

    lines = [
        "# Diffusion-Only vs Q-BridgeFlow Control Report",
        "",
        "This is a **controlled ranking comparison**, not a new BridgeFlow score term.",
        "Both arms rank the same block universe with the same `top_k`; only the ranking signal differs.",
        "",
        f"- Method / scoring path: `{cli_args.method}`",
        f"- Tasks: `{','.join(dataset_list(cli_args))}`",
        f"- Samples: {summary.get('sample_count', 0)}",
        f"- Top-k: {top_k}",
        f"- Layers probed: {cli_args.start_layer}..{cli_args.end_layer}",
        "",
        "## Arm Definitions",
        "",
        "| Arm | Ranking signal | Source |",
        "| --- | --- | --- |",
        "| **Diffusion-only baseline** | `max_layer(all_candidate_qk_scores)` | Exact per-layer QK saliency from `uasr_score_dump.jsonl`; in `sliminfer_original` this is `scoring=token_unit_qk`. |",
        "| **Q-BridgeFlow** | `new_bridgeflow_score` | Query-conditioned gain + persistence + bridge + neighbor − hub − redundancy − prompt-wrapper penalty. |",
        "| **SILM operational (reference)** | `max_layer(all_candidate_scores)` | Layer-normalized operational score from the same dump; should be rank-equivalent to the QK baseline. |",
        "",
        "## Headline Answers",
        "",
        f"1. **What does diffusion-only ranking select?** In {summary.get('sample_count', 0)} tiny samples, "
        f"{summary.get('diffusion_bridge_difference_count', 0)}/{max(1, summary.get('sample_count', 0))} samples have different top-{top_k} vs Q-BridgeFlow.",
        " Diffusion-only often elevates prompt tail / `Answer:` / `assistant` wrapper blocks and generic high-QK spans; see per-sample tables below.",
        "2. **Where does Q-BridgeFlow differ?** It re-ranks toward query-conditioned, persistent, bridge-linked evidence spans and away from wrapper/hub-like blocks via explicit penalties—not by adding another diffusion term.",
        "",
        "## Aggregate Top-k Diagnostics",
        "",
        "| Metric | Diffusion-only | Q-BridgeFlow |",
        "| --- | ---: | ---: |",
    ]

    if control_samples:
        lines.extend(
            [
                f"| Mean top-{top_k} Jaccard (per sample) | {mean(s.get('jaccard_diffusion_bridge', 0.0) for s in control_samples):.3f} | — |",
                f"| Mean wrapper hits in top-{top_k} | {mean(s.get('diffusion_topk_stats', {}).get('wrapper_hits', 0.0) for s in control_samples):.2f} | {mean(s.get('bridgeflow_topk_stats', {}).get('wrapper_hits', 0.0) for s in control_samples):.2f} |",
                f"| Mean hub_score in top-{top_k} | {mean(s.get('diffusion_topk_stats', {}).get('mean_hub_score', 0.0) for s in control_samples):.3f} | {mean(s.get('bridgeflow_topk_stats', {}).get('mean_hub_score', 0.0) for s in control_samples):.3f} |",
                f"| Mean query_conditioned_gain in top-{top_k} | {mean(s.get('diffusion_topk_stats', {}).get('mean_query_conditioned_gain', 0.0) for s in control_samples):.3f} | {mean(s.get('bridgeflow_topk_stats', {}).get('mean_query_conditioned_gain', 0.0) for s in control_samples):.3f} |",
                f"| Mean persistence_ratio in top-{top_k} | {mean(s.get('diffusion_topk_stats', {}).get('mean_persistence_ratio', 0.0) for s in control_samples):.3f} | {mean(s.get('bridgeflow_topk_stats', {}).get('mean_persistence_ratio', 0.0) for s in control_samples):.3f} |",
            ]
        )
    else:
        lines.append("| (no per-sample control rows) | — | — |")

    lines.extend(["", "## Per-Sample Comparison", ""])
    for sample in control_samples:
        sample_id = str(sample.get("sample_id", ""))
        task = str(sample.get("task", ""))
        ctx = context_by_id.get((sample_id, task))
        lines.append(f"### {task} / `{sample_id[:12]}...`")
        lines.append("")
        lines.append(f"- Question: {sample.get('question', '')}")
        lines.append(f"- Scoring kind: `{sample.get('scoring_kind', 'unknown')}`")
        lines.append(f"- Top-{top_k} Jaccard (diffusion vs bridgeflow): **{float(sample.get('jaccard_diffusion_bridge', 0.0)):.3f}**")
        lines.append(f"- Top-{top_k} Jaccard (silm operational vs bridgeflow): {float(sample.get('jaccard_silm_bridge', 0.0)):.3f}")
        lines.append("")

        def format_topk(label: str, block_ids: Sequence[int], score_lookup: Mapping[int, float]) -> List[str]:
            out = [f"**{label}**"]
            for rank, block_id in enumerate(block_ids, 1):
                preview = ""
                if ctx is not None and 0 <= int(block_id) < len(ctx.block_texts):
                    preview = ctx.block_texts[int(block_id)].replace("\n", " ")[:120]
                score = score_lookup.get(int(block_id), "")
                wrap = " [wrapper]" if wrapper_hit(preview) else ""
                out.append(f"{rank}. block {block_id} (score={score}){wrap}: {preview}")
            return out

        diffusion_lookup = {
            int(row["block_id"]): float(row["score"])
            for row in diffusion_rows
            if row.get("sample_id") == sample_id and row.get("task") == task
        }
        bridge_lookup = {
            int(row["block_id"]): float(row["score"])
            for row in bridge_rows
            if row.get("sample_id") == sample_id and row.get("task") == task
        }
        lines.extend(format_topk("Diffusion-only top-k", sample.get("top_diffusion_ids", []), diffusion_lookup))
        lines.append("")
        lines.extend(format_topk("Q-BridgeFlow top-k", sample.get("top_bridge_ids", []), bridge_lookup))
        lines.append("")
        only_diff = sample.get("only_diffusion_topk", [])
        only_bridge = sample.get("only_bridgeflow_topk", [])
        lines.append(f"- Only in diffusion top-{top_k}: `{only_diff}`")
        lines.append(f"- Only in Q-BridgeFlow top-{top_k}: `{only_bridge}`")
        lines.append("")

    lines.extend(
        [
            "## Exact vs Proxy Notes",
            "",
            "- **Exact (this control):** per-layer `all_candidate_qk_scores` from SILM/UASR dump; same block ids and layers as BridgeFlow probe.",
            "- **Not used here:** exact `G_diffusion = 1 - C_active` with hidden-state block reps (would need block_reps in dump).",
            "- **Q-BridgeFlow terms remain proxy** where noted in `comparison_report.md`; this file only isolates the diffusion-only baseline arm.",
            "",
            "## Output Files",
            "",
            "- `top_blocks_by_diffusion_only.csv`",
            "- `top_blocks_by_bridgeflow.csv`",
            "- `property_features.csv` (`raw_diffusion_score_or_proxy`, `diffusion_only_aggregate_score`)",
        ]
    )
    (out_dir / "diffusion_vs_bridgeflow_control_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_comparison_report(
    out_dir: Path,
    cli_args: argparse.Namespace,
    feature_rows: Sequence[Mapping[str, Any]],
    silm_rows: Sequence[Mapping[str, Any]],
    bridge_rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    status: str,
) -> None:
    high_hub = [row for row in feature_rows if maybe_float(row.get("hub_score")) is not None and float(row["hub_score"]) >= 0.5]
    high_qcg = [row for row in feature_rows if maybe_float(row.get("query_conditioned_gain")) is not None and float(row["query_conditioned_gain"]) > 0.0]
    persistent = [row for row in feature_rows if float(row.get("persistence_ratio", 0.0) or 0.0) >= 0.5]
    boundary_risk = [row for row in feature_rows if maybe_float(row.get("neighbor_gain_proxy")) is not None and float(row["neighbor_gain_proxy"]) > 0.1]

    lines = [
        "# BridgeFlow SILM Comparison Report",
        "",
        f"- Status: `{status}`",
        f"- Method: `{cli_args.method}`",
        f"- Tasks: `{','.join(dataset_list(cli_args))}`",
        f"- Samples analyzed: {summary.get('sample_count', 0)}",
        f"- Null question mode: `{cli_args.null_question_mode}`",
        "",
        "## Answers To Required Questions",
        "",
        f"1. Do SILM top blocks and BridgeFlow top blocks differ? {'Yes' if int(summary.get('difference_count', 0)) > 0 else 'No or not enough evidence'}; differing samples: {summary.get('difference_count', 0)}.",
        f"2. Do diffusion-only and Q-BridgeFlow top blocks differ? {'Yes' if int(summary.get('diffusion_bridge_difference_count', 0)) > 0 else 'No or not enough evidence'}; differing samples: {summary.get('diffusion_bridge_difference_count', 0)}. See `diffusion_vs_bridgeflow_control_report.md`.",
        f"3. Are high-diffusion blocks often generic hubs? Diffusion-only top-k wrapper hits average {mean(s.get('diffusion_topk_stats', {}).get('wrapper_hits', 0.0) for s in summary.get('control_samples', [])):.2f} vs BridgeFlow {mean(s.get('bridgeflow_topk_stats', {}).get('wrapper_hits', 0.0) for s in summary.get('control_samples', [])):.2f}; hub scores use null-question proxies.",
        f"4. Does query-conditioned gain suppress generic hubs? Preliminary signal: rows with positive query-conditioned gain = {len(high_qcg)}; BridgeFlow top-k mean query gain = {mean(s.get('bridgeflow_topk_stats', {}).get('mean_query_conditioned_gain', 0.0) for s in summary.get('control_samples', [])):.3f} vs diffusion-only {mean(s.get('diffusion_topk_stats', {}).get('mean_query_conditioned_gain', 0.0) for s in summary.get('control_samples', [])):.3f}.",
        f"5. Are persistent blocks more stable than one-layer spikes? Persistent rows (`persistence_ratio >= 0.5`) observed: {len(persistent)}.",
        f"6. Are potential bridge blocks selected that diffusion-only would miss? {'Possibly' if int(summary.get('diffusion_bridge_difference_count', 0)) > 0 else 'No clear evidence in this tiny run'}; inspect `only_bridgeflow_topk` in the control report.",
        f"7. Does neighbor sensitivity identify boundary-risk spans? Rows with positive `neighbor_gain_proxy > 0.1`: {len(boundary_risk)}.",
        "8. What is the most promising next experiment? Keep the same prefill-only setup and compare whether the diffusion-only vs Q-BridgeFlow gap persists on held-out samples before any QA generation.",
        "",
        "## Exact Vs Proxy Notes",
        "",
        "- Exact from current SILM path: per-layer selected blocks, per-layer raw SILM scores, per-layer `all_candidate_qk_scores` (diffusion-only baseline), QK-derived score dumps, layer ids, block ids, block text previews, and selected/pruned splits.",
        "- Proxy-only in this report: query-conditioned gain, hub score, anti-hub score, bridge-flow, directional flow, propagation entropy, and neighbor sensitivity.",
        "- Blank fields are intentional when no trustworthy exact value or conservative proxy was available.",
    ]
    (out_dir / "comparison_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_stop_or_pass(
    out_dir: Path,
    status: str,
    gpu_used: bool,
    dry_run_summary: Mapping[str, Any],
    probe_summary: Optional[Mapping[str, Any]],
) -> None:
    lines = [
        "# STOP OR PASS",
        "",
        f"- Dry run status: `{dry_run_summary.get('status', 'unknown')}`",
        f"- Probe status: `{status}`",
        f"- GPU used: {'Yes' if gpu_used else 'No'}",
    ]
    if probe_summary:
        gpu_info = probe_summary.get("gpu_info", {})
        selected = dict(gpu_info.get("selected", {})) if isinstance(gpu_info, dict) else {}
        if selected:
            lines.append(f"- Selected GPU: `{selected.get('index', '')} {selected.get('name', '')}`")
            lines.append(f"- Free GPU memory before run (GiB): {selected.get('memory_free_gib', '')}")
    if status == "blocked_gpu_busy":
        lines.append("- Final decision: `STOP`")
        lines.append("- Reason: GPU was checked before model execution and did not satisfy the safety threshold.")
    elif status == "prefill_probe_ok":
        lines.append("- Final decision: `PASS`")
        lines.append("- Reason: dry run and tiny prefill-only probe completed within the requested scope.")
    else:
        lines.append("- Final decision: `STOP`")
        lines.append("- Reason: only dry run completed.")
    (out_dir / "STOP_OR_PASS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_aggregated_outputs(
    out_dir: Path,
    cli_args: argparse.Namespace,
    contexts: Sequence[SampleContext],
    uasr_rows: Sequence[Mapping[str, Any]],
    used_rows: Sequence[Mapping[str, Any]],
    status: str,
) -> Dict[str, Any]:
    feature_rows, silm_rows, bridge_rows, diffusion_rows, summary = aggregate_sample_rows(
        cli_args, contexts, uasr_rows, used_rows
    )
    feature_fields = [
        "sample_id", "task", "question", "block_id", "span_id", "token_start", "token_end", "text_preview", "layer",
        "raw_diffusion_score_or_proxy", "diffusion_only_aggregate_score", "selected_by_silm", "rank_by_silm",
        "query_score", "null_question_score", "query_conditioned_gain", "persistence_count", "persistence_ratio",
        "onset_layer", "peak_layer", "propagation_entropy_or_proxy", "hub_score", "anti_hub_score",
        "directional_inflow_proxy", "directional_outflow_proxy", "bridge_flow_proxy", "neighbor_gain_proxy",
        "redundancy_proxy", "prompt_wrapper_penalty", "new_bridgeflow_score", "selected_by_bridgeflow_score",
        "selected_by_diffusion_only_score", "feature_terms_used", "used_blocks_layer_recorded", "diffusion_score_kind",
    ]
    write_jsonl(out_dir / "property_features.jsonl", feature_rows)
    write_csv(out_dir / "property_features.csv", feature_rows, feature_fields)
    write_csv(out_dir / "top_blocks_by_silm.csv", silm_rows, ["sample_id", "task", "rank", "block_id", "score", "text_preview"])
    write_csv(
        out_dir / "top_blocks_by_bridgeflow.csv",
        bridge_rows,
        ["sample_id", "task", "rank", "block_id", "score", "text_preview"],
    )
    write_csv(
        out_dir / "top_blocks_by_diffusion_only.csv",
        diffusion_rows,
        ["sample_id", "task", "rank", "block_id", "score", "score_kind", "text_preview"],
    )
    write_comparison_report(out_dir, cli_args, feature_rows, silm_rows, bridge_rows, summary, status)
    write_diffusion_control_report(out_dir, cli_args, contexts, feature_rows, diffusion_rows, bridge_rows, summary)
    write_json(out_dir / "ranking_control_summary.json", summary)
    return {
        "feature_rows": len(feature_rows),
        "silm_top_rows": len(silm_rows),
        "bridge_top_rows": len(bridge_rows),
        "diffusion_top_rows": len(diffusion_rows),
        "summary": summary,
    }


def main() -> None:
    cli_args = parse_args()
    out_dir = default_output_dir(cli_args)
    out_dir.mkdir(parents=True, exist_ok=True)
    append_run_command(out_dir / "run_commands.sh", [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]])

    # Dry run always comes first.
    tokenizer = variants.AutoTokenizer.from_pretrained(
        str(cli_args.tokenizer_path or cli_args.model_name_or_path),
        trust_remote_code=True,
    )
    contexts = build_sample_contexts(cli_args, tokenizer)
    dry_run_summary = write_dry_run_summary(out_dir, cli_args, contexts)

    if cli_args.analyze_only:
        uasr_rows = read_jsonl(out_dir / "uasr_score_dump.jsonl")
        used_rows = read_jsonl(out_dir / "used_blocks.jsonl", missing_ok=True)
        agg = write_aggregated_outputs(out_dir, cli_args, contexts, uasr_rows, used_rows, "analyze_only")
        write_stop_or_pass(out_dir, "analyze_only", gpu_used=False, dry_run_summary=dry_run_summary, probe_summary=None)
        print(
            json.dumps(
                {
                    "status": "analyze_only",
                    "out_dir": str(out_dir),
                    "sample_count": len(contexts),
                    **{k: agg[k] for k in ("feature_rows", "silm_top_rows", "bridge_top_rows", "diffusion_top_rows")},
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    if cli_args.dry_run:
        write_stop_or_pass(out_dir, "dry_run_only", gpu_used=False, dry_run_summary=dry_run_summary, probe_summary=None)
        print(json.dumps({"status": "dry_run_only", "out_dir": str(out_dir), "sample_count": len(contexts)}, ensure_ascii=False, indent=2))
        return

    probe_summary = run_prefill_probe(cli_args, out_dir, contexts)
    status = str(probe_summary.get("status", "unknown"))
    if status == "blocked_gpu_busy":
        write_stop_or_pass(out_dir, status, gpu_used=False, dry_run_summary=dry_run_summary, probe_summary=probe_summary)
        print(json.dumps({"status": status, "out_dir": str(out_dir)}, ensure_ascii=False, indent=2))
        return

    uasr_rows = read_jsonl(out_dir / "uasr_score_dump.jsonl", missing_ok=True)
    used_rows = read_jsonl(out_dir / "used_blocks.jsonl", missing_ok=True)
    agg = write_aggregated_outputs(out_dir, cli_args, contexts, uasr_rows, used_rows, status)
    write_stop_or_pass(out_dir, status, gpu_used=True, dry_run_summary=dry_run_summary, probe_summary=probe_summary)
    write_json(
        out_dir / "probe_manifest.json",
        {
            "status": status,
            "out_dir": str(out_dir),
            "sample_count": len(contexts),
            "tasks": dataset_list(cli_args),
            "method": cli_args.method,
            "null_question_mode": cli_args.null_question_mode,
            "top_k": int(cli_args.top_k),
            "gpu_used": True,
        },
    )
    print(
        json.dumps(
            {
                "status": status,
                "out_dir": str(out_dir),
                "sample_count": len(contexts),
                "feature_rows": agg["feature_rows"],
                "silm_top_rows": agg["silm_top_rows"],
                "bridge_top_rows": agg["bridge_top_rows"],
                "diffusion_top_rows": agg["diffusion_top_rows"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
