#!/usr/bin/env python3
"""Stage 3 / G4: backfill + matched-actual-token fairness (BridgePack-only)."""
from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.bridgepack_cpu_stage.common import FAIRNESS_DIR, read_csv
from tools.bridgepack_g1.common import (
    QWEN_PATH,
    TASKS,
    load_hist_lbp_rows,
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
from tools.bridgepack_g2.run_g2 import DECODING_CONFIG_HASH, LLAMA_MODEL_PATH, PROMPT_WRAPPER_HASH, g2_safety_gate

BUDGET = 1024
G1_ROOT = REPO_ROOT / "results" / "bridgepack_g1_qwen1024_matched_20260727_004934"
G2_ROOT = REPO_ROOT / "results" / "bridgepack_g2_llama1024_sourceorder_20260727_023300"
BASELINE_VARIANT = {
    "qwen": ("same_spans_source_order", G1_ROOT),
    "llama3": ("source_order_bridge_selection", G2_ROOT),
}
MODEL_PATH = {"qwen": QWEN_PATH, "llama3": LLAMA_MODEL_PATH}
VARIANT_STOP = "source_order_stop"
VARIANT_BACKFILL = "source_order_backfill"
VARIANT_MATCHED = "bm25_matched_actual_token"


def _all_ids() -> Dict[str, List[str]]:
    hist = REPO_ROOT / "results" / "bridgepack_exact_qwen_budget_1024_20260619_181059"
    out = {}
    for task in TASKS:
        rows = read_jsonl(
            hist
            / "packed_prompts_build"
            / "packed_prompts"
            / "lexical_bridgepack_path_order"
            / "budget_1024"
            / f"{task}.jsonl"
        )
        out[task] = [str(r["sample_id"]) for r in rows]
    return out


def build_backfill_unit(out_dir: Path, model_name: str, sample_ids: Mapping[str, Sequence[str]]) -> Dict[str, Any]:
    from transformers import AutoTokenizer
    from bridgekv.bridgepack_cpu_extensions import selection_only_control, _pack_ordered
    from bridgekv.evidence_text_packing import build_candidate_spans

    model_path = str(MODEL_PATH[model_name])
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    hist_rows = load_hist_lbp_rows(sample_ids)
    unit = out_dir / f"{model_name}_budget_{BUDGET}"
    stop_root = unit / "stop_at_first_nonfit" / "packed_prompts_build" / "packed_prompts" / "source_order_stop"
    back_root = unit / "backfill" / "packed_prompts_build" / "packed_prompts" / "source_order_backfill"
    audit_rows = []
    changed = 0
    for task in TASKS:
        stop_rows = []
        back_rows = []
        for sid in sample_ids[task]:
            src = next(r for r in hist_rows[task] if str(r["sample_id"]) == sid)
            question = str(src["question"])
            context = str(src["original_context"])
            answers = src.get("answers", src.get("gold_analysis_only_answers", []))
            # freeze selection via bridge_path, then pack stop vs backfill on same ordered stream
            packed_obj, sel_audit = selection_only_control(
                sid, task, question, context, BUDGET, "bridge_path", tokenizer=tokenizer
            )
            candidates, page_map = build_candidate_spans(sid, task, question, context)
            by_id = {c.span_id: c for c in candidates}
            # Prefer pre-budget selected order by source_order for both packers' candidate stream:
            # use all candidates sorted by source_order as the packing stream; selection preference
            # comes from selected set first then remaining.
            selected_ids = list(packed_obj.selected_span_ids)
            selected = [by_id[i] for i in selected_ids if i in by_id]
            remaining = [c for c in sorted(candidates, key=lambda x: x.source_order_index) if c.span_id not in set(selected_ids)]
            ordered = [(s, "bridge") for s in sorted(selected, key=lambda x: x.source_order_index)] + [
                (s, "bridge") for s in remaining
            ]
            stop_packed, stop_audit = _pack_ordered(
                sid, task, question, "source_order_stop", BUDGET, ordered, page_map,
                neighbor_size=32, ordering="source_order", used_bridge_path=True,
                stop_at_first_nonfit=True, tokenizer=tokenizer,
            )
            back_packed, back_audit = _pack_ordered(
                sid, task, question, "source_order_backfill", BUDGET, ordered, page_map,
                neighbor_size=32, ordering="source_order", used_bridge_path=True,
                stop_at_first_nonfit=False, tokenizer=tokenizer,
            )
            def to_row(packed, variant: str) -> Dict[str, Any]:
                return {
                    "sample_id": sid,
                    "task": task,
                    "variant": variant,
                    "budget": BUDGET,
                    "question": question,
                    "answers": answers,
                    "gold_analysis_only_answers": answers,
                    "all_classes": src.get("all_classes"),
                    "original_context": context,
                    "packed_text": packed.packed_text,
                    "selected_span_ids": list(packed.selected_span_ids),
                    "selected_page_ids": list(packed.selected_page_ids),
                    "model_token_count": int(packed.packed_token_count),
                    "packed_token_count": int(packed.packed_token_count),
                    "source_token_count": int(packed.source_token_count),
                    "exact_model_token_budget": True,
                    "model_tokenizer": model_path,
                    "ordering": "source_order",
                    "neighbor_size": 32,
                    "prompt_wrapper_hash": PROMPT_WRAPPER_HASH,
                    "decoding_config_hash": DECODING_CONFIG_HASH,
                    "over_budget": int(packed.packed_token_count) > BUDGET,
                }

            stop_row = to_row(stop_packed, "source_order_stop")
            back_row = to_row(back_packed, "source_order_backfill")
            stop_rows.append(stop_row)
            back_rows.append(back_row)
            same = stop_row["selected_span_ids"] == back_row["selected_span_ids"] and stop_row["packed_text"] == back_row["packed_text"]
            if not same:
                changed += 1
            audit_rows.append(
                {
                    "model": model_name,
                    "task": task,
                    "sample_id": sid,
                    "stop_tokens": stop_row["model_token_count"],
                    "backfill_tokens": back_row["model_token_count"],
                    "token_delta": back_row["model_token_count"] - stop_row["model_token_count"],
                    "changed": int(not same),
                    "stop_over_budget": int(stop_row["over_budget"]),
                    "backfill_over_budget": int(back_row["over_budget"]),
                }
            )
        write_jsonl(stop_root / f"budget_{BUDGET}" / f"{task}.jsonl", stop_rows)
        write_jsonl(back_root / f"budget_{BUDGET}" / f"{task}.jsonl", back_rows)

    summary = {
        "model": model_name,
        "budget": BUDGET,
        "n": len(audit_rows),
        "changed_samples": changed,
        "over_budget_stop": sum(int(r["stop_over_budget"]) for r in audit_rows),
        "over_budget_backfill": sum(int(r["backfill_over_budget"]) for r in audit_rows),
        "noop": changed == 0,
        "generated_at": utc_now(),
    }
    write_json(unit / "backfill_cpu_summary.json", summary)
    with (unit / "backfill_cpu_audit.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = list(audit_rows[0].keys())
        w = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for row in audit_rows:
            w.writerow(row)
    return summary


def _source_token_targets(model_name: str, sample_ids: Mapping[str, Sequence[str]]) -> Dict[Tuple[str, str], int]:
    variant, root = BASELINE_VARIANT[model_name]
    out: Dict[Tuple[str, str], int] = {}
    for task in TASKS:
        path = root / "full" / "packed_prompts_build" / "packed_prompts" / variant / f"budget_{BUDGET}" / f"{task}.jsonl"
        rows = {str(r["sample_id"]): r for r in read_jsonl(path)}
        for sid in sample_ids[task]:
            out[(task, sid)] = int(rows[sid].get("model_token_count") or rows[sid].get("packed_token_count") or BUDGET)
    return out


def build_matched_actual_token_unit(
    out_dir: Path, model_name: str, sample_ids: Mapping[str, Sequence[str]]
) -> Dict[str, Any]:
    """Pack BGE-style stream under per-sample source BridgePack token caps."""
    from transformers import AutoTokenizer
    from bridgekv.bridgepack_cpu_extensions import matched_actual_token_pack
    from tools.bridgepack_g1.common import load_hist_lbp_rows

    model_path = str(MODEL_PATH[model_name])
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    hist_rows = load_hist_lbp_rows(sample_ids)
    targets = _source_token_targets(model_name, sample_ids)
    unit = out_dir / f"{model_name}_budget_{BUDGET}"
    pack_root = unit / "matched_actual_token" / "packed_prompts_build" / "packed_prompts" / "bm25_matched_actual_token"
    audit_rows = []
    for task in TASKS:
        rows_out = []
        for sid in sample_ids[task]:
            src = next(r for r in hist_rows[task] if str(r["sample_id"]) == sid)
            target = targets[(task, sid)]
            packed, audit = matched_actual_token_pack(
                sid,
                task,
                str(src["question"]),
                str(src["original_context"]),
                "bm25_chunk_source_order",  # exact-budget retrieval baseline packing path
                target,
                tokenizer=tokenizer,
            )
            if int(packed.packed_token_count) > BUDGET:
                raise RuntimeError(f"matched-token over global budget {model_name}/{task}/{sid}")
            answers = src.get("answers", src.get("gold_analysis_only_answers", []))
            if not isinstance(answers, list):
                answers = [answers]
            packed_l = str(packed.packed_text).lower()
            answer_hit = 0
            for a in answers:
                a_s = str(a).strip()
                if a_s and a_s.lower() in packed_l:
                    answer_hit = 1
                    break
            row = {
                "sample_id": sid,
                "task": task,
                "variant": "bm25_matched_actual_token",
                "budget": BUDGET,
                "matched_target_tokens": target,
                "matched_method_base": "bm25_chunk_source_order",
                "question": src["question"],
                "answers": answers,
                "gold_analysis_only_answers": src.get("gold_analysis_only_answers", answers),
                "all_classes": src.get("all_classes"),
                "original_context": src["original_context"],
                "packed_text": packed.packed_text,
                "selected_span_ids": list(packed.selected_span_ids),
                "selected_page_ids": list(packed.selected_page_ids),
                "model_token_count": int(packed.packed_token_count),
                "packed_token_count": int(packed.packed_token_count),
                "exact_model_token_budget": True,
                "model_tokenizer": model_path,
                "prompt_wrapper_hash": PROMPT_WRAPPER_HASH,
                "decoding_config_hash": DECODING_CONFIG_HASH,
                "matched_budget_used": audit.get("matched_budget_used"),
                "matched_diff": audit.get("matched_diff"),
                "answer_string_in_pack": answer_hit,
            }
            rows_out.append(row)
            audit_rows.append(
                {
                    "model": model_name,
                    "task": task,
                    "sample_id": sid,
                    "target_tokens": target,
                    "packed_tokens": row["model_token_count"],
                    "matched_diff": audit.get("matched_diff"),
                    "over_budget": int(row["model_token_count"] > BUDGET),
                    "answer_string_in_pack": answer_hit,
                }
            )
        write_jsonl(pack_root / f"budget_{BUDGET}" / f"{task}.jsonl", rows_out)
    summary = {
        "model": model_name,
        "n": len(audit_rows),
        "mean_abs_diff": sum(abs(float(r["matched_diff"] or 0)) for r in audit_rows) / max(1, len(audit_rows)),
        "over_budget": sum(int(r["over_budget"]) for r in audit_rows),
        "answer_string_recall": sum(int(r["answer_string_in_pack"]) for r in audit_rows) / max(1, len(audit_rows)),
        "generated_at": utc_now(),
    }
    write_json(unit / "matched_actual_token_summary.json", summary)
    with (unit / "matched_actual_token_audit.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = list(audit_rows[0].keys())
        w = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for row in audit_rows:
            w.writerow(row)
    return summary


def _pack_root(out_dir: Path, model_name: str, arm: str, variant: str) -> Path:
    unit = out_dir / f"{model_name}_budget_{BUDGET}"
    if arm == "stop":
        return unit / "stop_at_first_nonfit" / "packed_prompts_build" / "packed_prompts" / variant
    if arm == "backfill":
        return unit / "backfill" / "packed_prompts_build" / "packed_prompts" / variant
    if arm == "matched":
        return unit / "matched_actual_token" / "packed_prompts_build" / "packed_prompts" / variant
    raise ValueError(arm)


def run_qa_on_pack_root(
    out_dir: Path,
    pack_root: Path,
    variant: str,
    model_name: str,
    sample_ids: Mapping[str, Sequence[str]],
    stage: str,
) -> Path:
    """Run greedy QA on packed prompts; returns prediction root."""
    gate = g2_safety_gate(out_dir, f"{stage}_{model_name}_before_load", phase="before_load")
    if gate["decision"] != "GO":
        write_text(out_dir / "WAITING_FOR_SAFE_GPU.md", json.dumps(gate, indent=2))
        raise RuntimeError(f"WAITING_FOR_SAFE_GPU: {gate['reasons']}")

    import tools.run_evidence_text_packing_generation_qwen32 as gen
    from tools.bridgepack_g1.generate import generate_greedy_historical

    torch = __import__("torch")
    model, tokenizer = load_qwen_fp16(str(MODEL_PATH[model_name]), torch)
    g2_safety_gate(out_dir, f"{stage}_{model_name}_after_load", phase="runtime")

    pred_root = out_dir / stage / "predictions"
    for task in TASKS:
        path = pack_root / f"budget_{BUDGET}" / f"{task}.jsonl"
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
    return pred_root


def _fairness_scores(model: str, method: str) -> Dict[Tuple[str, str], float]:
    out: Dict[Tuple[str, str], float] = {}
    for row in read_csv(FAIRNESS_DIR / "per_sample_scores.csv"):
        if row["model"] == model and int(row["budget"]) == BUDGET and row["method"] == method:
            out[(row["task"], row["sample_id"])] = float(row["score_0_100"])
    return out


def _retag(scores: Mapping[Tuple[str, str], float], model: str) -> Dict[Tuple[str, str], float]:
    return {(f"{model}:{t}", sid): sc for (t, sid), sc in scores.items()}


def _write_stage3_tables(out_dir: Path, summaries: List[Dict[str, Any]], matched: List[Dict[str, Any]], any_change: bool) -> None:
    """Write CPU-level G4 CSVs (used before QA; finalize overwrites with scored rows when QA runs)."""
    from tools.bridgepack_g1.run_g1 import score_predictions as _sp

    backfill_rows = []
    for model_name, root, variant in (
        ("qwen", G1_ROOT, "same_spans_source_order"),
        ("llama3", G2_ROOT, "source_order_bridge_selection"),
    ):
        for row in _sp(root / "full" / "predictions", [variant]):
            row = dict(row)
            row["model"] = model_name
            row["budget"] = BUDGET
            row["comparison_arm"] = "source_order_stop_baseline"
            row["note"] = "backfill_qa_skipped_noop" if not any_change else "baseline_for_backfill_pending_qa"
            backfill_rows.append(row)
        if not any_change:
            for row in _sp(root / "full" / "predictions", [variant]):
                row = dict(row)
                row["model"] = model_name
                row["budget"] = BUDGET
                row["variant"] = "source_order_backfill"
                row["comparison_arm"] = "source_order_backfill_noop_eq_stop"
                backfill_rows.append(row)

    with (out_dir / "G4_BACKFILL_RESULTS.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = sorted({k for r in backfill_rows for k in r.keys()})
        w = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        for r in backfill_rows:
            w.writerow(r)

    matched_rows = []
    for m in matched:
        matched_rows.append(
            {
                "model": m["model"],
                "n": m["n"],
                "mean_abs_token_diff": m["mean_abs_diff"],
                "over_budget": m["over_budget"],
                "answer_string_recall": m.get("answer_string_recall", ""),
                "status": "packs_ready_qa_pending" if any_change or m.get("qa_pending") else "packs_ready_cpu_fairness",
            }
        )
    with (out_dir / "G4_MATCHED_ACTUAL_TOKEN_RESULTS.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = list(matched_rows[0].keys()) if matched_rows else ["model", "n", "status"]
        w = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for r in matched_rows:
            w.writerow(r)

    boot_rows = []
    for model_name in ("qwen", "llama3"):
        boot_rows.append(
            {
                "comparison": f"{model_name} backfill vs stop",
                "n": 1150,
                "delta_macro": 0.0 if not any_change else "",
                "ci95_low": 0.0 if not any_change else "",
                "ci95_high": 0.0 if not any_change else "",
                "crosses_zero": True if not any_change else "",
                "p_holm": 1.0 if not any_change else "",
                "note": "NO_OP_identical_artifacts" if not any_change else "requires_backfill_QA",
            }
        )
    with (out_dir / "G4_BOOTSTRAP.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = list(boot_rows[0].keys())
        w = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for r in boot_rows:
            w.writerow(r)


def finalize_stage3_with_qa(out_dir: Path, any_change: bool) -> None:
    """Score QA preds, write G4 tables + bootstrap/Holm, STAGE_3.DONE."""
    result_rows: List[Dict[str, Any]] = []
    boot_rows: List[Dict[str, Any]] = []
    holm_inputs: List[Tuple[str, float]] = []
    matched_score_rows: List[Dict[str, Any]] = []

    for model_name in ("qwen", "llama3"):
        stop_stage = f"qa_full_{model_name}_stop"
        back_stage = f"qa_full_{model_name}_backfill"
        matched_stage = f"qa_full_{model_name}_matched"
        if any_change:
            for stage, variant, arm in (
                (stop_stage, VARIANT_STOP, "stop"),
                (back_stage, VARIANT_BACKFILL, "backfill"),
            ):
                pred_root = out_dir / stage / "predictions"
                if not (pred_root / variant).exists():
                    raise RuntimeError(f"Missing predictions for {stage}/{variant}")
                for row in score_predictions(pred_root, [variant]):
                    row = dict(row)
                    row["model"] = model_name
                    row["budget"] = BUDGET
                    row["comparison_arm"] = arm
                    result_rows.append(row)
            stop_scores = load_pred_scores(out_dir / stop_stage / "predictions", VARIANT_STOP, TASKS)
            back_scores = load_pred_scores(out_dir / back_stage / "predictions", VARIANT_BACKFILL, TASKS)
            boot = paired_bootstrap_delta(back_scores, stop_scores)
            label = f"{model_name} backfill vs stop"
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
                    "note": "qa_full",
                    **{f"delta_{t}": boot.get("task_deltas", {}).get(t, "") for t in TASKS},
                }
            )
        else:
            # NO_OP: cite frozen source-order twice
            variant, root = BASELINE_VARIANT[model_name]
            for row in score_predictions(root / "full" / "predictions", [variant]):
                row = dict(row)
                row["model"] = model_name
                row["budget"] = BUDGET
                row["comparison_arm"] = "source_order_stop_baseline"
                row["note"] = "backfill_qa_skipped_noop"
                result_rows.append(row)
            for row in score_predictions(root / "full" / "predictions", [variant]):
                row = dict(row)
                row["model"] = model_name
                row["budget"] = BUDGET
                row["variant"] = VARIANT_BACKFILL
                row["comparison_arm"] = "source_order_backfill_noop_eq_stop"
                result_rows.append(row)
            boot_rows.append(
                {
                    "comparison": f"{model_name} backfill vs stop",
                    "n": 1150,
                    "delta_macro": 0.0,
                    "ci95_low": 0.0,
                    "ci95_high": 0.0,
                    "crosses_zero": True,
                    "note": "NO_OP_identical_artifacts",
                    **{f"delta_{t}": 0.0 for t in TASKS},
                }
            )
            holm_inputs.append((f"{model_name} backfill vs stop", 1.0))

        # Matched-actual-token QA vs source_order and bm25
        matched_pred = out_dir / matched_stage / "predictions"
        if (matched_pred / VARIANT_MATCHED).exists():
            for row in score_predictions(matched_pred, [VARIANT_MATCHED]):
                row = dict(row)
                row["model"] = model_name
                row["budget"] = BUDGET
                row["comparison_arm"] = "matched_actual_token"
                matched_score_rows.append(row)
            matched_scores = load_pred_scores(matched_pred, VARIANT_MATCHED, TASKS)
            src_var, src_root = BASELINE_VARIANT[model_name]
            source_scores = load_pred_scores(src_root / "full" / "predictions", src_var, TASKS)
            bm25_scores = _fairness_scores(model_name, "bm25_chunk_source_order")
            for label, a, b in (
                (f"{model_name} matched vs source_order", matched_scores, source_scores),
                (f"{model_name} matched vs bm25", matched_scores, bm25_scores),
            ):
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
                        "note": "matched_actual_token_qa",
                        **{f"delta_{t}": boot.get("task_deltas", {}).get(t, "") for t in TASKS},
                    }
                )

    holm = {r["label"]: r for r in holm_correction(holm_inputs)} if holm_inputs else {}
    for row in boot_rows:
        h = holm.get(row["comparison"])
        if h:
            row["p_raw"] = h["p_raw"]
            row["p_holm"] = h["p_holm"]
            row["reject_holm_0.05"] = h["reject_holm_0.05"]
        else:
            row.setdefault("p_raw", "")
            row.setdefault("p_holm", "")
            row.setdefault("reject_holm_0.05", "")

    with (out_dir / "G4_BACKFILL_RESULTS.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = sorted({k for r in result_rows for k in r.keys()}) or ["model", "task", "score"]
        w = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        for r in result_rows:
            w.writerow(r)

    # Enrich matched CSV with per-task scores when available
    matched_out = []
    summary = json.loads((out_dir / "G4_BACKFILL_CPU_SUMMARY.json").read_text(encoding="utf-8"))
    matched_meta = {m["model"]: m for m in summary.get("matched", [])}
    for model_name in ("qwen", "llama3"):
        meta = matched_meta.get(model_name, {})
        row = {
            "model": model_name,
            "n": meta.get("n", ""),
            "mean_abs_token_diff": meta.get("mean_abs_diff", ""),
            "over_budget": meta.get("over_budget", ""),
            "answer_string_recall": meta.get("answer_string_recall", ""),
            "status": "qa_scored" if matched_score_rows else "cpu_only",
        }
        for r in matched_score_rows:
            if r.get("model") == model_name and str(r.get("task", "")).upper() == "MACRO":
                row["macro_score"] = r.get("score")
        # per-task
        for r in matched_score_rows:
            if r.get("model") == model_name and str(r.get("task", "")).upper() != "MACRO":
                row[f"score_{r['task']}"] = r.get("score")
        matched_out.append(row)
    with (out_dir / "G4_MATCHED_ACTUAL_TOKEN_RESULTS.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = sorted({k for r in matched_out for k in r.keys()})
        w = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        for r in matched_out:
            w.writerow(r)

    with (out_dir / "G4_BOOTSTRAP.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = sorted({k for r in boot_rows for k in r.keys()})
        w = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        for r in boot_rows:
            w.writerow(r)

    report = (out_dir / "G4_PACKING_FAIRNESS_REPORT.md").read_text(encoding="utf-8") if (out_dir / "G4_PACKING_FAIRNESS_REPORT.md").exists() else "# G4_PACKING_FAIRNESS_REPORT\n"
    report += f"\n\n## QA finalize\n\nGenerated: {utc_now()}\n\n"
    report += f"- any_change (backfill): {any_change}\n"
    report += f"- bootstrap comparisons: {len(boot_rows)}\n"
    write_text(out_dir / "G4_PACKING_FAIRNESS_REPORT.md", report)
    write_text(out_dir / "STAGE_3.DONE", f"Stage 3 complete at {utc_now()}\n")


def _load_pack_summaries(out_dir: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], bool]:
    summary = json.loads((out_dir / "G4_BACKFILL_CPU_SUMMARY.json").read_text(encoding="utf-8"))
    units = summary["units"]
    matched = summary["matched"]
    any_change = any(not u.get("noop", True) for u in units)
    return units, matched, any_change


def run_pack_phase(out_dir: Path, units: Sequence[str], ids: Mapping[str, Sequence[str]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], bool]:
    if (out_dir / "STAGE3_PACK.DONE").exists() and (out_dir / "G4_BACKFILL_CPU_SUMMARY.json").exists():
        print("[stage3] resume: STAGE3_PACK.DONE present; skipping rebuild", flush=True)
        return _load_pack_summaries(out_dir)

    summaries = []
    for model_name in units:
        print(f"[stage3] building backfill packs for {model_name}", flush=True)
        summaries.append(build_backfill_unit(out_dir, model_name, ids))
    matched = []
    for model_name in units:
        print(f"[stage3] building matched-actual-token packs for {model_name}", flush=True)
        matched.append(build_matched_actual_token_unit(out_dir, model_name, ids))
    write_json(out_dir / "G4_BACKFILL_CPU_SUMMARY.json", {"units": summaries, "matched": matched, "generated_at": utc_now()})
    lines = [
        "# G4_PACKING_FAIRNESS_REPORT",
        "",
        f"Generated: {utc_now()}",
        "",
        "## Backfill CPU",
        "",
    ]
    any_change = False
    for s in summaries:
        lines.append(
            f"- {s['model']}: changed={s['changed_samples']}/{s['n']}, "
            f"over_budget stop/backfill={s['over_budget_stop']}/{s['over_budget_backfill']}, noop={s['noop']}"
        )
        any_change = any_change or (not s["noop"])
        if s["over_budget_stop"] or s["over_budget_backfill"]:
            raise RuntimeError(f"over-budget in stage3 packs: {s}")
    lines.append("")
    lines.append("## Matched-actual-token CPU")
    lines.append("")
    for m in matched:
        lines.append(
            f"- {m['model']}: n={m['n']}, mean_abs_diff={m['mean_abs_diff']:.2f}, "
            f"over_budget={m['over_budget']}, answer_string_recall={m.get('answer_string_recall', float('nan')):.4f}"
        )
        if m["over_budget"]:
            raise RuntimeError(f"matched-token over-budget: {m}")
    if not any_change:
        lines.append("")
        lines.append("Decision: **NO_OP** for G4-A backfill — identical to stop-at-first-nonfit; skip backfill QA.")
        write_text(out_dir / "G4_BACKFILL_NO_OP.md", "\n".join(lines))
    write_text(out_dir / "G4_PACKING_FAIRNESS_REPORT.md", "\n".join(lines))
    _write_stage3_tables(out_dir, summaries, matched, any_change)
    write_text(out_dir / "STAGE3_PACK.DONE", f"{utc_now()}\n")
    return summaries, matched, any_change


def run_qa_phases(out_dir: Path, ids: Mapping[str, Sequence[str]], any_change: bool, smoke_only: bool) -> None:
    sample_ids = select_fixed_sample_ids(2) if smoke_only else ids
    tag = "smoke12" if smoke_only else "full"
    for model_name in ("qwen", "llama3"):
        if any_change:
            for arm, variant in (("stop", VARIANT_STOP), ("backfill", VARIANT_BACKFILL)):
                stage = f"qa_{tag}_{model_name}_{arm}"
                done = out_dir / f"QA_{tag.upper()}_{model_name.upper()}_{arm.upper()}.DONE"
                if done.exists():
                    print(f"[stage3] skip {stage} ({done.name})", flush=True)
                    continue
                print(f"[stage3] QA {stage}", flush=True)
                run_qa_on_pack_root(
                    out_dir,
                    _pack_root(out_dir, model_name, arm, variant),
                    variant,
                    model_name,
                    sample_ids,
                    stage,
                )
                write_text(done, f"{utc_now()}\n")
        # matched always QA for token-fairness scores
        stage = f"qa_{tag}_{model_name}_matched"
        done = out_dir / f"QA_{tag.upper()}_{model_name.upper()}_MATCHED.DONE"
        if done.exists():
            print(f"[stage3] skip {stage}", flush=True)
            continue
        print(f"[stage3] QA {stage}", flush=True)
        run_qa_on_pack_root(
            out_dir,
            _pack_root(out_dir, model_name, "matched", VARIANT_MATCHED),
            VARIANT_MATCHED,
            model_name,
            sample_ids,
            stage,
        )
        write_text(done, f"{utc_now()}\n")
    if smoke_only:
        write_text(out_dir / "STAGE3_SMOKE_QA.DONE", f"{utc_now()}\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--stop_after", default="pack", choices=["pack", "smoke12", "full", "finalize"])
    parser.add_argument(
        "--resume_from",
        default="",
        choices=["", "pack", "smoke12", "full", "finalize"],
    )
    parser.add_argument("--unit", default="all", choices=["all", "qwen", "llama3"])
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = _all_ids()
    units = ["qwen", "llama3"] if args.unit == "all" else [args.unit]

    order = ["pack", "smoke12", "full", "finalize"]
    start = order.index(args.resume_from) if args.resume_from else 0
    stop = order.index(args.stop_after)

    summaries: List[Dict[str, Any]] = []
    matched: List[Dict[str, Any]] = []
    any_change = False

    if start <= order.index("pack") <= stop:
        summaries, matched, any_change = run_pack_phase(out_dir, units, ids)
        if args.stop_after == "pack":
            if not any_change and all(m["over_budget"] == 0 for m in matched):
                write_text(out_dir / "STAGE_3.DONE", f"Stage 3 complete (backfill NO_OP + matched packs) at {utc_now()}\n")
            else:
                write_text(
                    out_dir / "STAGE3_PENDING_BACKFILL_QA.md",
                    f"Backfill changed artifacts (any_change={any_change}); run --resume_from smoke12 --stop_after finalize.\n",
                )
            return 0
    else:
        summaries, matched, any_change = _load_pack_summaries(out_dir)

    if start <= order.index("smoke12") <= stop:
        run_qa_phases(out_dir, ids, any_change, smoke_only=True)
        if args.stop_after == "smoke12":
            return 0

    if start <= order.index("full") <= stop:
        run_qa_phases(out_dir, ids, any_change, smoke_only=False)

    if start <= order.index("finalize") <= stop or args.stop_after in ("full", "finalize"):
        # Always finalize after full QA
        if order.index("full") <= stop:
            finalize_stage3_with_qa(out_dir, any_change)
        elif args.stop_after == "finalize":
            finalize_stage3_with_qa(out_dir, any_change)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        out = Path(sys.argv[sys.argv.index("--out_dir") + 1]) if "--out_dir" in sys.argv else Path(".")
        write_text(out / "STAGE3_FATAL_ERROR.md", f"{exc}\n\n{traceback.format_exc()}")
        raise
