#!/usr/bin/env python3
"""Stage 1 source-order multibudget runner."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.bridgepack_cpu_stage.common import FAIRNESS_DIR, read_csv
from tools.bridgepack_g1.common import (
    BUDGET,
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
from tools.bridgepack_g1.run_g1 import _exact_pack_from_ordered, score_predictions
from tools.bridgepack_g1.stats import approx_p_from_bootstrap, holm_correction, load_pred_scores, paired_bootstrap_delta
from tools.bridgepack_g2.run_g2 import DECODING_CONFIG_HASH, LLAMA_MODEL_PATH, PROMPT_WRAPPER_HASH, g2_safety_gate
from tools.bridgepack_g2.run_independent_fix import ORIG_G2_ROOT

MODEL_SPECS = {
    "qwen": {"model_path": QWEN_PATH, "fairness_model": "qwen"},
    "llama3": {"model_path": LLAMA_MODEL_PATH, "fairness_model": "llama3"},
}
STAGE1_UNITS = [
    ("qwen", 256),
    ("qwen", 512),
    ("llama3", 256),
    ("llama3", 512),
]
SOURCE_VARIANT = "source_order_bridge_selection"


def _fairness_scores(model: str, budget: int, method: str) -> Dict[Tuple[str, str], float]:
    out: Dict[Tuple[str, str], float] = {}
    for row in read_csv(FAIRNESS_DIR / "per_sample_scores.csv"):
        if row["model"] == model and int(row["budget"]) == budget and row["method"] == method:
            out[(row["task"], row["sample_id"])] = float(row["score_0_100"])
    return out


def _all_sample_ids() -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for task in TASKS:
        path = REPO_ROOT / "results" / "bridgepack_exact_qwen_budget_1024_20260619_181059" / "packed_prompts_build" / "packed_prompts" / "lexical_bridgepack_path_order" / "budget_1024" / f"{task}.jsonl"
        rows = read_jsonl(path)
        out[task] = [str(r["sample_id"]) for r in rows]
    return out


def _pack_source_order_bridge(
    tokenizer,
    model_tokenizer_path: str,
    sample_id: str,
    task: str,
    question: str,
    context: str,
    answers: Sequence[str],
    all_classes: Any,
    budget: int,
) -> Dict[str, Any]:
    from bridgekv.bridgepack_cpu_extensions import selection_only_control
    from bridgekv.evidence_text_packing import build_candidate_spans

    packed_obj, audit = selection_only_control(
        sample_id,
        task,
        question,
        context,
        budget,
        "bridge_path",
        tokenizer=tokenizer,
    )
    candidates, page_map = build_candidate_spans(sample_id, task, question, context)
    by_id = {c.span_id: c for c in candidates}
    selected = [by_id[i] for i in packed_obj.selected_span_ids if i in by_id]
    ordered = [(s, "bridge") for s in sorted(selected, key=lambda x: x.source_order_index)]
    payload = _exact_pack_from_ordered(
        tokenizer,
        sample_id,
        task,
        question,
        context,
        SOURCE_VARIANT,
        ordered,
        page_map,
        32,
        True,
        "source_order",
        audit_extra={
            "answers": list(answers),
            "gold_analysis_only_answers": list(answers),
            "all_classes": all_classes,
            "prompt_wrapper_hash": PROMPT_WRAPPER_HASH,
            "decoding_config_hash": DECODING_CONFIG_HASH,
            "canonical_source_order_bridge": True,
            "selection_rule": "bridge_path",
        },
    )
    payload["model_tokenizer"] = str(model_tokenizer_path)
    payload["budget"] = budget
    payload["selection_pre_budget_span_ids"] = list(packed_obj.selected_span_ids)
    payload["packed_artifact_sha256"] = audit.get("artifact_sha256", "")
    return payload


def build_unit_packs(out_dir: Path, model_name: str, budget: int, sample_ids: Mapping[str, Sequence[str]]) -> Dict[str, Any]:
    from transformers import AutoTokenizer

    model_path = str(MODEL_SPECS[model_name]["model_path"])
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    hist_rows = load_hist_lbp_rows(sample_ids)
    unit_dir = out_dir / f"{model_name}_budget_{budget}"
    packed_root = unit_dir / "full" / "packed_prompts_build"

    audit_rows: List[Dict[str, Any]] = []
    toks: List[int] = []
    for task in TASKS:
        rows = []
        for hist in hist_rows[task]:
            payload = _pack_source_order_bridge(
                tokenizer,
                model_path,
                str(hist["sample_id"]),
                task,
                str(hist.get("question") or ""),
                str(hist.get("original_context") or ""),
                hist.get("answers") or hist.get("gold_analysis_only_answers") or [],
                hist.get("all_classes"),
                budget,
            )
            rows.append(payload)
            mt = int(payload.get("model_token_count") or 0)
            toks.append(mt)
            audit_rows.append(
                {
                    "model": model_name,
                    "budget": budget,
                    "task": task,
                    "sample_id": str(hist["sample_id"]),
                    "model_token_count": mt,
                    "over_budget": int(mt > budget),
                    "selected_span_count": len(payload.get("selected_span_ids") or []),
                    "repair_count": len(payload.get("repaired_ranges") or []),
                    "serialization": "source_order",
                    "selection_rule": "bridge_path",
                }
            )
        write_jsonl(packed_root / "packed_prompts" / SOURCE_VARIANT / f"budget_{budget}" / f"{task}.jsonl", rows)

    overs = sum(r["over_budget"] for r in audit_rows)
    summary = {
        "model": model_name,
        "budget": budget,
        "n": len(audit_rows),
        "over_budget": overs,
        "mean_tokens": float(np.mean(toks)),
        "p50_tokens": float(np.quantile(toks, 0.5)),
        "p95_tokens": float(np.quantile(toks, 0.95)),
        "max_tokens": int(max(toks)),
    }
    write_json(unit_dir / "pack_summary.json", summary)
    if overs:
        raise RuntimeError(f"Over-budget packs in {model_name}/{budget}: {overs}")
    return {"unit_dir": unit_dir, "packed_root": packed_root, "summary": summary, "audit_rows": audit_rows}


def ensure_subset(unit_dir: Path, stage: str, budget: int, sample_ids: Mapping[str, Sequence[str]]) -> Path:
    full_root = unit_dir / "full" / "packed_prompts_build" / "packed_prompts"
    stage_root = unit_dir / stage / "packed_prompts_build" / "packed_prompts"
    for task in TASKS:
        rows = read_jsonl(full_root / SOURCE_VARIANT / f"budget_{budget}" / f"{task}.jsonl")
        by_id = {str(r["sample_id"]): r for r in rows}
        subset = [by_id[sid] for sid in sample_ids[task]]
        write_jsonl(stage_root / SOURCE_VARIANT / f"budget_{budget}" / f"{task}.jsonl", subset)
    return unit_dir / stage / "packed_prompts_build"


def run_qa_on_packs_with_budget(
    out_dir: Path,
    packed_root: Path,
    variants: Sequence[str],
    sample_ids: Mapping[str, Sequence[str]],
    model,
    tokenizer,
    torch,
    stage: str,
    budget: int,
) -> Dict[str, Any]:
    """BridgePack stage1 wrapper: same as G1 QA but honors per-unit budget paths."""
    import tools.run_evidence_text_packing_generation_qwen32 as gen
    from tools.bridgepack_g1.generate import generate_greedy_historical

    pred_root = out_dir / stage / "predictions"
    meta = []
    for variant in variants:
        for task in TASKS:
            path = packed_root / "packed_prompts" / variant / f"budget_{budget}" / f"{task}.jsonl"
            rows = read_jsonl(path)
            want = list(sample_ids[task])
            by_id = {str(r["sample_id"]): r for r in rows}
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
                        meta.append({"variant": variant, "task": task, "sample_id": sid, "ok": True, "resumed": True})
                        continue
                    gate = g2_safety_gate(out_dir, f"{stage}_{variant}_{task}_{sample_index}", phase="runtime")
                    if gate["decision"] != "GO":
                        raise RuntimeError(f"Safety stop: {gate['reasons']}")
                    mt = int(row.get("model_token_count") or row.get("packed_token_count") or 0)
                    if mt > budget:
                        raise RuntimeError(f"Over-budget before QA: {variant}/{task}/{sid} tokens={mt} budget={budget}")
                    prompt = gen._generation_prompt_from_row(task, row)
                    max_new = gen._max_new_tokens_for(task, 0)
                    pred, context_length = generate_greedy_historical(model, tokenizer, torch, prompt, max_new)
                    payload = {
                        "sample_id": row.get("sample_id", sample_index),
                        "sample_index": sample_index,
                        "task": task,
                        "variant": variant,
                        "pred": pred,
                        "answers": row.get("answers", row.get("gold_analysis_only_answers", [])),
                        "all_classes": row.get("all_classes", None),
                        "length": context_length,
                        "packed_token_count": mt,
                        "prompt_token_count": context_length,
                        "budget": budget,
                        "question": row.get("question", ""),
                        "selected_span_ids": row.get("selected_span_ids", []),
                        "selected_page_ids": row.get("selected_page_ids", []),
                    }
                    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                    handle.flush()
                    meta.append({"variant": variant, "task": task, "sample_id": payload["sample_id"], "ok": True, "resumed": False})
    return {"n_preds": len(meta), "pred_root": pred_root}


def _write_smoke_report(unit_dir: Path, model_name: str, budget: int, stage: str, expected: int) -> None:
    preds = sum(
        len(read_jsonl(unit_dir / stage / "predictions" / SOURCE_VARIANT / f"{task}.jsonl"))
        for task in TASKS
    )
    write_text(
        unit_dir / f"{stage.upper()}_REPORT.md",
        f"""# {stage.upper()}_REPORT

Generated: {utc_now()}

- model: `{model_name}`
- budget: `{budget}`
- variant: `{SOURCE_VARIANT}`
- predictions: {preds} (expected {expected})
- over-budget packs: 0
- scorer: PASS
""",
    )


def _load_model_for_name(model_name: str):
    import torch

    model, tokenizer = load_qwen_fp16(str(MODEL_SPECS[model_name]["model_path"]), torch)
    return model, tokenizer, torch


def run_unit_qa(unit_dir: Path, model_name: str, budget: int, stop_after: str, resume_from: str = "") -> None:
    gate = g2_safety_gate(unit_dir, f"{model_name}_{budget}_before_load", phase="before_load")
    if gate["decision"] != "GO":
        write_text(unit_dir / "WAITING_FOR_SAFE_GPU.md", json.dumps(gate, indent=2))
        raise RuntimeError(f"WAITING_FOR_SAFE_GPU: {gate['reasons']}")

    model, tokenizer, torch = _load_model_for_name(model_name)
    g2_safety_gate(unit_dir, f"{model_name}_{budget}_after_load", phase="runtime")

    ids12 = select_fixed_sample_ids(2)
    if resume_from in {"", "pack", "smoke12"}:
        smoke_root = ensure_subset(unit_dir, "smoke12", budget, ids12)
        run_qa_on_packs_with_budget(unit_dir, smoke_root, [SOURCE_VARIANT], ids12, model, tokenizer, torch, "smoke12", budget)
        score_predictions(unit_dir / "smoke12" / "predictions", [SOURCE_VARIANT])
        _write_smoke_report(unit_dir, model_name, budget, "smoke12", 12)
    if stop_after == "smoke12":
        return

    full_ids = _all_sample_ids()
    run_qa_on_packs_with_budget(
        unit_dir,
        unit_dir / "full" / "packed_prompts_build",
        [SOURCE_VARIANT],
        full_ids,
        model,
        tokenizer,
        torch,
        "full",
        budget,
    )
    score_predictions(unit_dir / "full" / "predictions", [SOURCE_VARIANT])


def _unit_full_ready(unit_dir: Path) -> bool:
    pred_root = unit_dir / "full" / "predictions" / SOURCE_VARIANT
    if not pred_root.exists():
        return False
    return all((pred_root / f"{task}.jsonl").exists() and len(read_jsonl(pred_root / f"{task}.jsonl")) > 0 for task in TASKS)


def _fairness_method_names(budget: int) -> Dict[str, str]:
    # Fairness audit used different method labels for 256 vs 512/1024.
    if int(budget) == 256:
        return {
            "path_order": "lexical_bridgepack",
            "bm25": "bm25_chunk_packing",
            "tfidf": "tfidf_chunk_packing",
            "bge": "bge_chunk_packing",
        }
    return {
        "path_order": "lexical_bridgepack_path_order",
        "bm25": "bm25_chunk_source_order",
        "tfidf": "tfidf_chunk_source_order",
        "bge": "bge_chunk_packing",
    }


def write_stage1_stats(out_dir: Path) -> None:
    result_rows = []
    boot_rows = []
    holm_inputs = []
    ready_units = []
    for model_name, budget in STAGE1_UNITS:
        unit_dir = out_dir / f"{model_name}_budget_{budget}"
        if not _unit_full_ready(unit_dir):
            continue
        ready_units.append((model_name, budget))
        source_scores = load_pred_scores(unit_dir / "full" / "predictions", SOURCE_VARIANT, TASKS)
        fairness_model = str(MODEL_SPECS[model_name]["fairness_model"])
        methods = _fairness_method_names(budget)
        comps = [
            (f"{model_name} budget{budget} source_order vs path_order", source_scores, _fairness_scores(fairness_model, budget, methods["path_order"])),
            (f"{model_name} budget{budget} source_order vs bm25", source_scores, _fairness_scores(fairness_model, budget, methods["bm25"])),
            (f"{model_name} budget{budget} source_order vs tfidf", source_scores, _fairness_scores(fairness_model, budget, methods["tfidf"])),
            (f"{model_name} budget{budget} source_order vs bge", source_scores, _fairness_scores(fairness_model, budget, methods["bge"])),
        ]
        score_rows = score_predictions(unit_dir / "full" / "predictions", [SOURCE_VARIANT])
        for row in score_rows:
            row = dict(row)
            row["model"] = model_name
            row["budget"] = budget
            result_rows.append(row)
        for label, a, b in comps:
            boot = paired_bootstrap_delta(a, b)
            if int(boot.get("n", 0)) <= 0:
                raise RuntimeError(f"Empty paired bootstrap overlap for {label}")
            p_raw = approx_p_from_bootstrap(boot["delta_macro"], boot.get("boot_std", float("nan")))
            holm_inputs.append((label, p_raw))
            boot_rows.append(
                {
                    "comparison": label,
                    "model": model_name,
                    "budget": budget,
                    "n": boot["n"],
                    "delta_macro": boot["delta_macro"],
                    "ci95_low": boot["ci95_low"],
                    "ci95_high": boot["ci95_high"],
                    "crosses_zero": boot["crosses_zero"],
                    **{f"delta_{t}": boot.get("task_deltas", {}).get(t, "") for t in TASKS},
                }
            )
    if len(ready_units) != len(STAGE1_UNITS):
        raise RuntimeError(f"Stage1 finalize blocked: ready={ready_units} expected={STAGE1_UNITS}")
    holm = {r["label"]: r for r in holm_correction(holm_inputs)}
    for row in boot_rows:
        h = holm[row["comparison"]]
        row["p_raw"] = h["p_raw"]
        row["p_holm"] = h["p_holm"]
        row["reject_holm_0.05"] = h["reject_holm_0.05"]

    with (out_dir / "SOURCE_ORDER_MULTIBUDGET_RESULTS.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["model", "budget", "variant", "task", "n", "score", "mean_length"]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in result_rows:
            writer.writerow(row)
    with (out_dir / "SOURCE_ORDER_MULTIBUDGET_BOOTSTRAP.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["comparison", "model", "budget", "n", "delta_macro", "ci95_low", "ci95_high", "crosses_zero", "p_raw", "p_holm", "reject_holm_0.05"] + [f"delta_{t}" for t in TASKS]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in boot_rows:
            writer.writerow(row)

    # Canonical 1024 source-order macros from frozen G1/G2 (never hardcode / never use path-order LBP).
    g1_results = REPO_ROOT / "results" / "bridgepack_g1_qwen1024_matched_20260727_004934" / "G1_FULL_RESULTS.csv"
    g2_results = REPO_ROOT / "results" / "bridgepack_g2_llama1024_sourceorder_20260727_023300" / "G2_FULL_RESULTS.csv"
    qwen_1024 = None
    llama_1024 = None
    for row in read_csv(g1_results):
        if row.get("variant") == "same_spans_source_order" and row.get("task") == "MACRO":
            qwen_1024 = float(row["score"])
    for row in read_csv(g2_results):
        if row.get("variant") == "source_order_bridge_selection" and row.get("task") == "MACRO":
            llama_1024 = float(row["score"])
    if qwen_1024 is None or llama_1024 is None:
        raise RuntimeError(f"Missing frozen 1024 macros: qwen={qwen_1024} llama={llama_1024}")

    result_rows.extend(
        [
            {"model": "qwen", "budget": 1024, "variant": "same_spans_source_order", "task": "MACRO", "n": 1150, "score": qwen_1024, "mean_length": ""},
            {"model": "llama3", "budget": 1024, "variant": "source_order_bridge_selection", "task": "MACRO", "n": 1150, "score": llama_1024, "mean_length": ""},
        ]
    )
    with (out_dir / "SOURCE_ORDER_MULTIBUDGET_RESULTS.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["model", "budget", "variant", "task", "n", "score", "mean_length"]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in result_rows:
            writer.writerow(row)

    scaling = {}
    for model_name in {"qwen", "llama3"}:
        scaling[model_name] = {}
        for budget in [256, 512]:
            row = next(r for r in result_rows if r["model"] == model_name and int(r["budget"]) == budget and r["task"] == "MACRO")
            scaling[model_name][str(budget)] = float(row["score"])
        scaling[model_name]["1024"] = qwen_1024 if model_name == "qwen" else llama_1024
    write_text(
        out_dir / "SOURCE_ORDER_BUDGET_SCALING.md",
        f"""# SOURCE_ORDER_BUDGET_SCALING

Generated: {utc_now()}

## Macro trend

- qwen: 256={scaling['qwen']['256']:.2f}, 512={scaling['qwen']['512']:.2f}, 1024={scaling['qwen']['1024']:.2f}
- llama3: 256={scaling['llama3']['256']:.2f}, 512={scaling['llama3']['512']:.2f}, 1024={scaling['llama3']['1024']:.2f}

## 1024 provenance (frozen; not re-run in Stage 1)

- qwen 1024 = G1 `same_spans_source_order` MACRO = **{qwen_1024:.3f}** from `results/bridgepack_g1_qwen1024_matched_20260727_004934/` (1150 predictions rescore PASS).
- llama3 1024 = G2 `source_order_bridge_selection` MACRO = **{llama_1024:.3f}** from `results/bridgepack_g2_llama1024_sourceorder_20260727_023300/`.
- Method config SHA256: `15886731816aa2eb4e08afa121bd8ccc9631dd232474a82a8f20214cc2ce16bf`.
- Do **not** cite path-order LBP (38.34) as source-order. Prior erroneous hardcoded 38.05 is invalidated.
""",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--stop_after", default="pack", choices=["pack", "smoke12", "full", "finalize"])
    parser.add_argument("--resume_from", default="", choices=["", "pack", "smoke12", "full"])
    parser.add_argument("--unit", default="all", choices=["all", "qwen_256", "qwen_512", "llama3_256", "llama3_512"])
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_ids_full = _all_sample_ids()

    targets = STAGE1_UNITS
    if args.unit != "all":
        model_name, budget = args.unit.split("_")
        targets = [(model_name, int(budget))]

    audit_rows = []
    for model_name, budget in targets:
        unit_dir = out_dir / f"{model_name}_budget_{budget}"
        need_pack = not (unit_dir / "full" / "packed_prompts_build").exists()
        if args.resume_from in {"", "pack"} and need_pack and args.stop_after != "finalize":
            info = build_unit_packs(out_dir, model_name, budget, sample_ids_full)
            audit_rows.extend(info["audit_rows"])
        if args.stop_after in {"pack", "finalize"}:
            continue
        run_unit_qa(unit_dir, model_name, budget, args.stop_after, resume_from=args.resume_from)

    if audit_rows:
        with (out_dir / "STAGE1_PACKING_AUDIT.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(audit_rows[0].keys()), lineterminator="\n")
            writer.writeheader()
            for row in audit_rows:
                writer.writerow(row)

    if args.stop_after == "finalize" or (args.stop_after == "full" and args.unit == "all"):
        write_stage1_stats(out_dir)
        write_text(out_dir / "STAGE_1.DONE", f"Stage 1 complete at {utc_now()}\n")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        out = Path(sys.argv[sys.argv.index("--out_dir") + 1]) if "--out_dir" in sys.argv else Path(".")
        write_text(out / "STAGE1_FATAL_ERROR.md", f"{exc}\n\n{traceback.format_exc()}")
        raise
