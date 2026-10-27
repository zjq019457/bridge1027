#!/usr/bin/env python3
"""Fair small-sample latency, memory, and token-efficiency diagnostic."""
from __future__ import annotations

import argparse
import copy
import gc
import math
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

import run_qwen25_sliminfer_variants as R  # noqa: E402
from tools.fup_guarded_swap_lib import FupHarness, release_cuda  # noqa: E402
from tools.span_drm_validation_lib import (  # noqa: E402
    BASELINE_ROOT,
    LONGBENCH_DIR,
    MODEL_PATH,
    SANITY_TASKS,
    fixed_audit,
    markdown_table,
    mean,
    read_csv,
    read_json,
    reconstruct_span_selection,
    span_metadata,
    write_csv,
)

METHODS = ["FullKV", "SlimInfer original", "SlimInfer larger K+M", "fixed DRM-Add", "SpanDRM"]
WORD_RE = re.compile(r"[A-Za-z0-9_]+")


def method_config(task: str, method: str) -> Optional[Dict[str, Any]]:
    if method == "SlimInfer original":
        path = (
            REPO_ROOT / "results" / "rescuekv_qwen25_full_longbench_20260520_170047"
            / "raw_metrics" / "M1_sliminfer_original" / task / "config.json"
        )
    elif method == "SlimInfer larger K+M":
        path = BASELINE_ROOT / "raw_metrics" / "sliminfer_larger" / task / "config.json"
    elif method == "fixed DRM-Add":
        path = BASELINE_ROOT / "raw_metrics" / "drm_add" / task / "config.json"
    else:
        return None
    payload = read_json(path, {})
    cfg = copy.deepcopy(payload.get("sliminfer_config", {}))
    for key in [
        "block_scores_path", "layer_stats_path", "rescue_audit_path",
        "block_layout_audit_path", "feature_dump_path",
    ]:
        cfg.pop(key, None)
    return cfg


def prepare_config(cfg: Dict[str, Any], ctx: Any, method: str) -> Dict[str, Any]:
    cfg = copy.deepcopy(cfg)
    cfg["audit_context"] = {
        "task": ctx.task, "sample_id": str(ctx.sample_id), "sample_index": ctx.sample_index,
        "context_length": ctx.context_length, "method": method, "query_text": ctx.query_text,
        "block_texts": ctx.block_texts,
    }
    cfg["query_text"] = ctx.query_text
    cfg["block_texts"] = ctx.block_texts
    cfg["block_layout"] = ctx.block_layout
    cfg["rescue_candidate_layout"] = ctx.fixed_rescue_layout
    if method == "fixed DRM-Add":
        cfg["block_strategy"] = "fixed"
        cfg["rescue_candidate_strategy"] = ""
        cfg["rescue_policy"] = "add"
        cfg["rescue_score"] = "drm"
        cfg["rescue_selector"] = "drm_mmr"
    return cfg


def span_score_overhead_ms(ctx: Any, tokenizer: Any, active_pages: Sequence[int]) -> float:
    """Measure the query-only candidate scoring workload used by SpanDRM."""
    query = {word.lower() for word in WORD_RE.findall(ctx.query_text or "") if len(word) > 1}
    active = {int(page) for page in active_pages}
    ids = [int(x) for x in ctx.input_ids[0].detach().cpu().tolist()]
    start = time.perf_counter()
    scores: List[float] = []
    for token_start in range(0, len(ids), 24):
        page = token_start // 64
        if page in active:
            continue
        text = tokenizer.decode(
            ids[token_start:min(len(ids), token_start + 24)],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        span_words = {word.lower() for word in WORD_RE.findall(text) if len(word) > 1}
        overlap = len(query & span_words) / max(1, len(query | span_words))
        keyword = len(query & span_words) / max(1, len(query))
        scores.append(overlap + 0.5 * keyword)
    scores.sort(reverse=True)
    _ = scores[: max(1, len(scores) // 20)]
    return (time.perf_counter() - start) * 1000.0


@torch.no_grad()
def timed_generate(
    harness: FupHarness,
    ctx: Any,
    method: str,
    max_new_tokens: int,
    selection: Mapping[str, Any],
    base_cfg: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    model = harness.model
    tokenizer = harness.tokenizer
    input_ids = ctx.input_ids
    attention_mask = ctx.attention_mask
    kwargs: Dict[str, Any] = {}
    cfg = None
    cache = None
    if method == "SpanDRM":
        index_t = torch.tensor(selection["selected_abs_indices"], dtype=torch.long, device=input_ids.device)
        input_ids = torch.index_select(input_ids, dim=1, index=index_t)
        attention_mask = torch.ones_like(input_ids)
    elif method not in {"FullKV"}:
        SlimInferCache, HsCache = harness.cache_classes
        cache = SlimInferCache(model.device)
        cfg = prepare_config(base_cfg or {}, ctx, method)
        R.reset_runtime_stats(cfg)
        kwargs = {
            "past_key_values": cache,
            "hs_cache": HsCache(),
            "sliminfer_config": cfg,
            "all_original_prompt_indices_abs": list(range(ctx.context_length)),
        }

    release_cuda()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    generated = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        num_beams=1,
        do_sample=False,
        use_cache=True,
        eos_token_id=tokenizer.eos_token_id,
        **kwargs,
    )
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    peak_mb = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
    generated_count = max(0, int(generated.shape[1] - input_ids.shape[1]))
    if method == "FullKV":
        selected_tokens = int(input_ids.shape[1])
        selected_semantics = "full prompt tokens"
    elif method == "SpanDRM":
        selected_tokens = int(input_ids.shape[1])
        selected_semantics = "actual text-packed prompt tokens"
    elif method == "SlimInfer original":
        selected_tokens = min(int(ctx.context_length), 4096)
        selected_semantics = "configured final retained-token target"
    else:
        selected_tokens = int(selection["target_final_tokens"])
        selected_semantics = "matched configured final retained-token target"
    runtime = R.summarize_runtime_stats(cfg)
    del generated, kwargs
    return {
        "elapsed_ms": elapsed_ms,
        "peak_gpu_memory_MB": peak_mb,
        "generated_tokens": generated_count,
        "selected_tokens": selected_tokens,
        "selected_token_semantics": selected_semantics,
        "score_overhead_ms": float(runtime.get("score_overhead_ms", 0.0) or 0.0),
    }


def kv_memory_proxy_mb(model: Any, selected_tokens: int) -> float:
    config = model.config
    layers = int(getattr(config, "num_hidden_layers", 28))
    kv_heads = int(getattr(config, "num_key_value_heads", 4))
    head_dim = int(getattr(config, "head_dim", config.hidden_size // config.num_attention_heads))
    byte_count = selected_tokens * layers * kv_heads * head_dim * 2 * 2
    return byte_count / (1024.0 * 1024.0)


def run(out: Path, samples_per_task: int, decode_tokens: int, tasks: Sequence[str]) -> None:
    harness = FupHarness(str(MODEL_PATH), str(LONGBENCH_DIR), str(out / "_latency_work"), max_new_tokens=decode_tokens)
    raw_path = out / "latency_memory_raw_samples.csv"
    raw: List[Dict[str, Any]] = [dict(row) for row in read_csv(raw_path)]
    completed = {
        (str(row.get("task", "")), str(row.get("sample_id", "")), str(row.get("method", "")))
        for row in raw
        if str(row.get("runner_fail", "1")) == "0"
    }
    warmed = set()
    for task in tasks:
        metadata = span_metadata(task)
        audits = fixed_audit(task)
        configs = {method: method_config(task, method) for method in METHODS}
        for sample_index, sample in enumerate(harness.load_samples(task, samples_per_task)):
            ctx = harness.build_sample_ctx(task, sample, sample_index)
            sid = str(ctx.sample_id)
            selection = reconstruct_span_selection(task, sid, metadata.get(sid), audits.get(sid))
            page_count = len({int(index) // 64 for index in selection["selected_abs_indices"]})
            span_count = len(selection["selected_spans"])
            for method in METHODS:
                if (task, sid, method) in completed:
                    continue
                try:
                    if method not in warmed:
                        _ = timed_generate(harness, ctx, method, 1, selection, configs[method])
                        warmed.add(method)
                    ttft = timed_generate(harness, ctx, method, 1, selection, configs[method])
                    e2e = timed_generate(harness, ctx, method, decode_tokens, selection, configs[method])
                    selected = int(e2e["selected_tokens"])
                    if method == "SpanDRM":
                        scorer_ms = span_score_overhead_ms(ctx, harness.tokenizer, selection["active_pages"])
                    else:
                        scorer_ms = float(e2e["score_overhead_ms"])
                    raw.append({
                        "task": task, "sample_id": sid, "method": method,
                        "full_context_tokens": ctx.context_length,
                        "selected_token_count": selected,
                        "selected_token_semantics": e2e["selected_token_semantics"],
                        "selected_page_count": (
                            page_count if method == "SpanDRM"
                            else len(selection["final_pages"]) if method == "fixed DRM-Add"
                            else math.ceil(selected / 64)
                        ),
                        "selected_span_count": span_count if method == "SpanDRM" else 0,
                        "compression_ratio_vs_full_context": round(selected / max(1, ctx.context_length), 7),
                        "KV_memory_proxy_MB": round(kv_memory_proxy_mb(harness.model, selected), 4),
                        "scorer_overhead_ms": round(scorer_ms, 4),
                        "prefill_time_ms": round(float(ttft["elapsed_ms"]), 4),
                        "decode_time_ms": round(max(0.0, float(e2e["elapsed_ms"]) - float(ttft["elapsed_ms"])), 4),
                        "end_to_end_time_ms": round(float(e2e["elapsed_ms"]), 4),
                        "TTFT_ms": round(float(ttft["elapsed_ms"]), 4),
                        "peak_gpu_memory_MB": round(max(float(ttft["peak_gpu_memory_MB"]), float(e2e["peak_gpu_memory_MB"])), 4),
                        "runner_fail": 0,
                        "failure_reason": "",
                        "runtime_semantics": (
                            "text-packed ordinary FullKV prompt" if method == "SpanDRM"
                            else "native FullKV" if method == "FullKV"
                            else "native block-oriented SlimInfer cache"
                        ),
                    })
                except BaseException as exc:
                    raw.append({
                        "task": task, "sample_id": sid, "method": method,
                        "full_context_tokens": ctx.context_length, "selected_token_count": "",
                        "selected_token_semantics": "",
                        "selected_page_count": "", "selected_span_count": "",
                        "compression_ratio_vs_full_context": "", "KV_memory_proxy_MB": "",
                        "scorer_overhead_ms": "", "prefill_time_ms": "", "decode_time_ms": "",
                        "end_to_end_time_ms": "", "TTFT_ms": "", "peak_gpu_memory_MB": "",
                        "runner_fail": 1, "failure_reason": f"{type(exc).__name__}: {exc}"[:500],
                        "runtime_semantics": "failed",
                    })
                    release_cuda()
                write_csv(raw_path, raw)
                print(f"[latency] {task} {sample_index + 1}/{samples_per_task} {method}", flush=True)
    write_csv(raw_path, raw)
    by_task: List[Dict[str, Any]] = []
    fields = [
        "selected_token_count", "selected_page_count", "selected_span_count",
        "compression_ratio_vs_full_context", "KV_memory_proxy_MB", "scorer_overhead_ms",
        "prefill_time_ms", "decode_time_ms", "end_to_end_time_ms", "TTFT_ms",
        "peak_gpu_memory_MB",
    ]
    for method in METHODS:
        for task in tasks:
            rows = [row for row in raw if row["method"] == method and row["task"] == task]
            item: Dict[str, Any] = {
                "method": method, "task": task, "n_samples": len(rows),
                "runner_fail": sum(int(row["runner_fail"]) for row in rows),
            }
            for field in fields:
                values = [float(row[field]) for row in rows if row.get(field) not in ("", None)]
                item[field] = round(mean(values), 4) if values else ""
            by_task.append(item)
    write_csv(out / "latency_memory_by_task.csv", by_task)
    summary: List[Dict[str, Any]] = []
    for method in METHODS:
        rows = [row for row in raw if row["method"] == method]
        item = {
            "method": method, "evaluation": f"{samples_per_task} samples/task; {decode_tokens} decode-token cap",
            "n_samples": len(rows), "runner_fail": sum(int(row["runner_fail"]) for row in rows),
        }
        for field in fields:
            values = [float(row[field]) for row in rows if row.get(field) not in ("", None)]
            item[field] = round(mean(values), 4) if values else ""
        summary.append(item)
    write_csv(out / "latency_memory_summary.csv", summary)
    span = next(row for row in summary if row["method"] == "SpanDRM")
    full = next(row for row in summary if row["method"] == "FullKV")
    speedup = (
        float(full["end_to_end_time_ms"]) / float(span["end_to_end_time_ms"])
        if span.get("end_to_end_time_ms") and not span["runner_fail"] and not full["runner_fail"] else 0.0
    )
    report = [
        "# Latency, Memory, and Token Efficiency Diagnostic",
        "",
        "This is a controlled small-sample diagnostic with one model, identical prompts, greedy decoding, and a uniform decode-token cap. SpanDRM runtime is the observed runtime of its text-packed prompt. KV memory is a formula-based retained-token proxy and is not measured physical sparse-cache allocation.",
        "",
        markdown_table(summary, ["method", "n_samples", "selected_token_count", "compression_ratio_vs_full_context", "KV_memory_proxy_MB", "scorer_overhead_ms", "TTFT_ms", "end_to_end_time_ms", "peak_gpu_memory_MB", "runner_fail"]),
        "",
        "## Required Answers",
        "",
        f"1. SpanDRM average retained tokens: {span.get('selected_token_count')}; compression ratio: {span.get('compression_ratio_vs_full_context')}; theoretical KV proxy: {span.get('KV_memory_proxy_MB')} MB.",
        f"2. SpanDRM measured scoring overhead proxy: {span.get('scorer_overhead_ms')} ms.",
        f"3. Observed diagnostic text-packed E2E speedup vs FullKV: {speedup:.3f}x.",
        "4. Without a native arbitrary-token sparse KV cache, retained-token memory and physical-cache speed claims remain proxies; text-packed runtime is an actual context-packing runtime result.",
        "5. Broad paper latency-speedup claim allowed: **False**. The paper may report the observed diagnostic text-packing runtime, but must not generalize it to full LongBench or a physical sparse-KV system.",
        "",
    ]
    (out / "latency_memory_report.md").write_text("\n".join(report), encoding="utf-8")
    del harness
    gc.collect()
    release_cuda()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--samples_per_task", type=int, default=32)
    parser.add_argument("--decode_tokens", type=int, default=8)
    parser.add_argument("--tasks", default=",".join(SANITY_TASKS))
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    run(out, args.samples_per_task, args.decode_tokens, [task for task in args.tasks.split(",") if task])


if __name__ == "__main__":
    main()
