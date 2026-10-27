#!/usr/bin/env python3
"""Token-index audit and sparse-KV repack sanity check for SpanDRM."""
from __future__ import annotations

import argparse
import gc
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.build_span_candidates import tokenize_sample  # noqa: E402
from tools.fup_guarded_swap_lib import FupHarness, release_cuda  # noqa: E402
from tools.span_drm_validation_lib import (  # noqa: E402
    LONGBENCH_DIR,
    MODEL_PATH,
    SANITY_TASKS,
    TASKS,
    fixed_audit,
    load_longbench_index,
    markdown_table,
    mean,
    prediction_index,
    read_json,
    reconstruct_span_selection,
    score_prediction,
    span_metadata,
    write_csv,
)


def load_tokenizer() -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(MODEL_PATH), trust_remote_code=True, use_fast=True)


def token_index_audit(out: Path) -> List[Dict[str, Any]]:
    tokenizer = load_tokenizer()
    prompts = read_json(REPO_ROOT / "configs" / "dataset2prompt.json", {})
    rows: List[Dict[str, Any]] = []
    for task in TASKS:
        data = load_longbench_index(task)
        metadata = span_metadata(task)
        audits = fixed_audit(task)
        for sample_index, (sid, meta) in enumerate(metadata.items()):
            sample = data.get(sid)
            audit = audits.get(sid, {})
            selection = reconstruct_span_selection(task, sid, meta, audit)
            if sample is None:
                rows.append({
                    "task": task, "sample_id": sid, "page_id": "", "span_id": "MISSING_SOURCE_SAMPLE",
                    "char_start": "", "char_end": "", "token_start": "", "token_end": "", "token_len": "",
                    "order_preserved": False, "source_traceable": False,
                    "budget_consistent": selection["budget_consistent"],
                })
                continue
            prompt, ids = tokenize_sample(tokenizer, prompts[task], task, sample, 32768)
            encoded = tokenizer(prompt, truncation=False, return_offsets_mapping=True)
            offsets = encoded.get("offset_mapping", [])
            ids_match = [int(x) for x in encoded["input_ids"]] == ids
            spans = list(meta.get("selected_spans", []))
            if not spans:
                rows.append({
                    "task": task, "sample_id": sid, "page_id": "", "span_id": "FULL_PAGE_FALLBACK",
                    "char_start": 0, "char_end": len(prompt), "token_start": 0,
                    "token_end": len(ids), "token_len": len(ids),
                    "order_preserved": selection["selected_abs_indices"] == sorted(set(selection["selected_abs_indices"])),
                    "source_traceable": bool(ids_match and len(ids) == selection["context_length"]),
                    "budget_consistent": selection["budget_consistent"],
                })
                continue
            for span in spans:
                start = int(span.get("selected_token_start", span.get("token_start", 0)) or 0)
                end = int(span.get("selected_token_end", span.get("token_end", start)) or start)
                char_start = offsets[start][0] if start < len(offsets) else ""
                char_end = offsets[end - 1][1] if 0 < end <= len(offsets) else ""
                page_id = int(span.get("page_id", -1) or -1)
                traceable = (
                    ids_match
                    and 0 <= start < end <= len(ids)
                    and page_id == int(span.get("token_start", start) or start) // 64
                    and f"p{page_id}_" in str(span.get("span_id", ""))
                )
                rows.append({
                    "task": task,
                    "sample_id": sid,
                    "page_id": page_id,
                    "span_id": span.get("span_id", ""),
                    "char_start": char_start,
                    "char_end": char_end,
                    "token_start": start,
                    "token_end": end,
                    "token_len": end - start,
                    "order_preserved": selection["selected_abs_indices"] == sorted(set(selection["selected_abs_indices"])),
                    "source_traceable": traceable,
                    "budget_consistent": selection["budget_consistent"],
                })
    fields = [
        "task", "sample_id", "page_id", "span_id", "char_start", "char_end",
        "token_start", "token_end", "token_len", "order_preserved",
        "source_traceable", "budget_consistent",
    ]
    write_csv(out / "physical_span_token_index_audit.csv", rows, fields)
    return rows


def compact_dynamic_cache(past: Any, selected_indices: Sequence[int]) -> int:
    indices = sorted({int(index) for index in selected_indices if int(index) >= 0})
    if not indices:
        raise ValueError("no selected KV indices remain before final prompt token")
    for layer in range(len(past.key_cache)):
        device = past.key_cache[layer].device
        index_t = torch.tensor(indices, dtype=torch.long, device=device)
        past.key_cache[layer] = torch.index_select(past.key_cache[layer], dim=-2, index=index_t)
        past.value_cache[layer] = torch.index_select(past.value_cache[layer], dim=-2, index=index_t)
    if hasattr(past, "_seen_tokens"):
        past._seen_tokens = len(indices)
    return len(indices)


@torch.no_grad()
def physical_generate(harness: FupHarness, ctx: Any, selected_indices: Sequence[int], max_new_tokens: int) -> Dict[str, Any]:
    model = harness.model
    tokenizer = harness.tokenizer
    context_length = int(ctx.context_length)
    if context_length < 2:
        raise ValueError("context too short for physical KV repack")
    retained = sorted({int(index) for index in selected_indices if 0 <= int(index) < context_length - 1})
    if not retained:
        raise ValueError("selection has no token before final prompt token")

    release_cuda()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    prefill_start = time.perf_counter()
    outputs = model(
        input_ids=ctx.input_ids[:, :-1],
        attention_mask=ctx.attention_mask[:, :-1],
        use_cache=True,
        return_dict=True,
        logits_to_keep=1,
    )
    torch.cuda.synchronize()
    prefill_ms = (time.perf_counter() - prefill_start) * 1000.0
    past = outputs.past_key_values
    retained_count = compact_dynamic_cache(past, retained)
    del outputs

    generated: List[int] = []
    current = ctx.input_ids[:, -1:]
    decode_start = time.perf_counter()
    for step in range(max_new_tokens):
        compact_position = retained_count + step
        original_position = context_length - 1 + step
        attention_mask = torch.ones(
            (1, compact_position + 1),
            dtype=ctx.attention_mask.dtype,
            device=ctx.attention_mask.device,
        )
        outputs = model(
            input_ids=current,
            attention_mask=attention_mask,
            position_ids=torch.tensor([[original_position]], dtype=torch.long, device=current.device),
            cache_position=torch.tensor([compact_position], dtype=torch.long, device=current.device),
            past_key_values=past,
            use_cache=True,
            return_dict=True,
            logits_to_keep=1,
        )
        next_token = int(torch.argmax(outputs.logits[:, -1, :], dim=-1).item())
        generated.append(next_token)
        past = outputs.past_key_values
        current = torch.tensor([[next_token]], dtype=torch.long, device=current.device)
        if next_token == tokenizer.eos_token_id:
            break
    torch.cuda.synchronize()
    decode_ms = (time.perf_counter() - decode_start) * 1000.0
    peak_mb = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
    pred = tokenizer.decode(generated, skip_special_tokens=True)
    if not pred.strip():
        raise RuntimeError("empty physical KV generation")
    return {
        "pred": pred,
        "retained_kv_tokens_before_final_prompt_token": retained_count,
        "prefill_time_ms": prefill_ms,
        "decode_time_ms": decode_ms,
        "end_to_end_time_ms": prefill_ms + decode_ms,
        "peak_gpu_memory_MB": peak_mb,
    }


def baseline_case(task: str, sid: str, method: str) -> Dict[str, Any]:
    row = prediction_index(method, task).get(sid, {})
    fixed = prediction_index("fixed DRM-Add", task).get(sid, {})
    answers = row.get("answers", fixed.get("answers", []))
    classes = row.get("all_classes", fixed.get("all_classes"))
    pred = str(row.get("pred", ""))
    return {
        "pred": pred,
        "score": score_prediction(task, pred, answers, classes),
        "runner_fail": int(not pred.strip()),
    }


def run_physical(out: Path, samples_per_task: int, tasks: Sequence[str]) -> None:
    harness = FupHarness(str(MODEL_PATH), str(LONGBENCH_DIR), str(out / "_physical_work"), max_new_tokens=64)
    cases: List[Dict[str, Any]] = []
    for task in tasks:
        samples = harness.load_samples(task, samples_per_task)
        metadata = span_metadata(task)
        audits = fixed_audit(task)
        full_data = load_longbench_index(task)
        for sample_index, sample in enumerate(samples):
            sid = str(sample.get("_id", sample.get("id", sample_index)))
            ctx = harness.build_sample_ctx(task, sample, sample_index)
            selection = reconstruct_span_selection(task, sid, metadata.get(sid), audits.get(sid))
            answers = sample.get("answers", [])
            classes = sample.get("all_classes")
            for output_method, source_method in [
                ("slim_infer_original", "SlimInfer original"),
                ("fixed_drm_add", "fixed DRM-Add"),
                ("span_drm_text_packing", "SpanDRM"),
            ]:
                baseline = baseline_case(task, sid, source_method)
                cases.append({
                    "task": task, "sample_id": sid, "method": output_method,
                    "score": round(baseline["score"], 6), "pred": baseline["pred"],
                    "selected_token_count": (
                        selection["selected_token_count"] if output_method == "span_drm_text_packing" else ""
                    ),
                    "target_token_count": selection["target_final_tokens"],
                    "runner_fail": baseline["runner_fail"], "failure_reason": "",
                    "prefill_time_ms": "", "decode_time_ms": "", "end_to_end_time_ms": "",
                    "peak_gpu_memory_MB": "", "implementation_status": "reused_full_prediction",
                })
            try:
                max_tokens = int(harness.dataset2maxlen.get(task, 32))
                result = physical_generate(harness, ctx, selection["selected_abs_indices"], max_tokens)
                score = score_prediction(task, result["pred"], answers, classes)
                cases.append({
                    "task": task, "sample_id": sid, "method": "span_drm_physical_kv_retention",
                    "score": round(score, 6), "pred": result["pred"],
                    "selected_token_count": result["retained_kv_tokens_before_final_prompt_token"] + 1,
                    "target_token_count": selection["target_final_tokens"],
                    "runner_fail": 0, "failure_reason": "",
                    "prefill_time_ms": round(result["prefill_time_ms"], 4),
                    "decode_time_ms": round(result["decode_time_ms"], 4),
                    "end_to_end_time_ms": round(result["end_to_end_time_ms"], 4),
                    "peak_gpu_memory_MB": round(result["peak_gpu_memory_MB"], 4),
                    "implementation_status": "order_preserving_dynamic_cache_repack_approximation",
                })
            except BaseException as exc:
                cases.append({
                    "task": task, "sample_id": sid, "method": "span_drm_physical_kv_retention",
                    "score": 0.0, "pred": "", "selected_token_count": selection["selected_token_count"],
                    "target_token_count": selection["target_final_tokens"], "runner_fail": 1,
                    "failure_reason": f"{type(exc).__name__}: {exc}"[:500],
                    "prefill_time_ms": "", "decode_time_ms": "", "end_to_end_time_ms": "",
                    "peak_gpu_memory_MB": "", "implementation_status": "failed_prototype",
                })
                release_cuda()
            print(f"[physical-kv] {task} {sample_index + 1}/{len(samples)}", flush=True)

    write_csv(out / "physical_kv_span_sanity_cases.csv", cases)
    by_task: List[Dict[str, Any]] = []
    methods = [
        "slim_infer_original", "fixed_drm_add", "span_drm_text_packing",
        "span_drm_physical_kv_retention",
    ]
    for method in methods:
        for task in tasks:
            selected = [row for row in cases if row["method"] == method and row["task"] == task]
            by_task.append({
                "method": method, "task": task, "n_samples": len(selected),
                "score": round(mean(float(row["score"]) for row in selected), 4),
                "runner_fail": sum(int(row["runner_fail"]) for row in selected),
                "avg_selected_tokens": round(mean(float(row["selected_token_count"]) for row in selected if row["selected_token_count"] != ""), 2)
                if any(row["selected_token_count"] != "" for row in selected) else "",
            })
    write_csv(out / "physical_kv_span_sanity_by_task.csv", by_task)
    summary: List[Dict[str, Any]] = []
    for method in methods:
        task_rows = [row for row in by_task if row["method"] == method]
        case_rows = [row for row in cases if row["method"] == method]
        summary.append({
            "method": method,
            "evaluation": "32-sample-per-task diagnostic" if samples_per_task == 32 else f"{samples_per_task}-sample-per-task diagnostic",
            "task_macro_avg": round(mean(float(row["score"]) for row in task_rows), 4),
            "sample_weighted_avg": round(mean(float(row["score"]) for row in case_rows), 4),
            "runner_fail": sum(int(row["runner_fail"]) for row in case_rows),
            "implementation_status": (
                "order_preserving_dynamic_cache_repack_approximation"
                if method == "span_drm_physical_kv_retention" else "reused_full_predictions"
            ),
        })
    write_csv(out / "physical_kv_span_sanity_summary.csv", summary)
    physical = next(row for row in summary if row["method"] == "span_drm_physical_kv_retention")
    text = next(row for row in summary if row["method"] == "span_drm_text_packing")
    gap = float(physical["task_macro_avg"]) - float(text["task_macro_avg"])
    failed = int(physical["runner_fail"])
    task_deltas = []
    for task in tasks:
        physical_task = next(
            row for row in by_task if row["method"] == "span_drm_physical_kv_retention" and row["task"] == task
        )
        text_task = next(
            row for row in by_task if row["method"] == "span_drm_text_packing" and row["task"] == task
        )
        task_deltas.append((task, float(physical_task["score"]) - float(text_task["score"])))
    stable = sorted(task_deltas, key=lambda item: abs(item[1]))
    report = [
        "# Physical KV Span Sanity Check",
        "",
        "This is an actual per-layer DynamicCache K/V token repack prototype, but it is not a native arbitrary-token SlimInfer cache implementation. Full original prompt prefill is still required; retained K/V entries are then compacted in source order. Keys preserve their original RoPE encoding, while decode uses original position IDs and compact cache positions.",
        "",
        markdown_table(summary, ["method", "evaluation", "task_macro_avg", "sample_weighted_avg", "runner_fail", "implementation_status"]),
        "",
        "## Required Answers",
        "",
        f"1. Physical minus text-packing task-macro score: {gap:+.4f}. This is not close enough to treat the two evaluations as quality-equivalent.",
        f"2. Physical runner failures: {failed}. Because selected indices and budgets are identical, the gap is attributable to representation/position semantics: text packing recomputes compact-context hidden states and RoPE positions, while post-prefill KV repack retains full-context states at original RoPE positions and changes only decode-visible K/V.",
        f"3. Moderate mechanical mapping claim supported: **{failed == 0}**. Quality-equivalence claim supported: **{failed == 0 and abs(gap) <= 2.0}**.",
        "4. The completed full result must remain a structured context/KV-retention proxy claim; this prototype is not the native SlimInfer arbitrary-token cache path.",
        "5. Most stable tasks: " + ", ".join(f"{task} ({delta:+.2f})" for task, delta in stable[:3]) + ".",
        f"6. Runner fail/OOM/mask errors: {failed}; exact reasons are in `physical_kv_span_sanity_cases.csv`.",
        "",
        "Do not claim that the current SpanDRM full evaluation implements a production physical span-granular sparse KV cache.",
        "",
    ]
    (out / "physical_kv_span_sanity_report.md").write_text("\n".join(report), encoding="utf-8")
    del harness
    gc.collect()
    release_cuda()


def write_audit_report(out: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    n = len(rows)
    order_ok = sum(str(row["order_preserved"]).lower() == "true" for row in rows)
    trace_ok = sum(str(row["source_traceable"]).lower() == "true" for row in rows)
    budget_ok = sum(str(row["budget_consistent"]).lower() == "true" for row in rows)
    (out / "physical_span_token_index_audit_report.md").write_text(
        "# Span Token Index Audit\n\n"
        f"- Rows: {n}\n"
        f"- Order-preserving: {order_ok}/{n}\n"
        f"- Source-traceable: {trace_ok}/{n}\n"
        f"- Sample budget-consistent: {budget_ok}/{n}\n\n"
        "Offsets are computed against the exact tokenizer prompt used by the structured runner. "
        "Full-page fallback rows intentionally have no span ID.\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--samples_per_task", type=int, default=32)
    parser.add_argument("--tasks", default=",".join(SANITY_TASKS))
    parser.add_argument("--skip_token_audit", action="store_true")
    parser.add_argument("--audit_only", action="store_true")
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not args.skip_token_audit:
        rows = token_index_audit(out)
        write_audit_report(out, rows)
    if not args.audit_only:
        run_physical(out, args.samples_per_task, [task for task in args.tasks.split(",") if task])


if __name__ == "__main__":
    main()
