#!/usr/bin/env python3
"""P0 matched order/selection ablation runner (BridgePack-only).

Reuses G1/G2 full predictions where protocol-equivalent; only packs+QA missing variants.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import random
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

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
from tools.bridgepack_g1.stats import (
    approx_p_from_bootstrap,
    holm_correction,
    load_pred_scores,
    paired_bootstrap_delta,
)
from tools.bridgepack_g2.run_g2 import LLAMA_MODEL_PATH, g2_safety_gate

FREEZE_SHA = "15886731816aa2eb4e08afa121bd8ccc9631dd232474a82a8f20214cc2ce16bf"
G1_ROOT = REPO_ROOT / "results" / "bridgepack_g1_qwen1024_matched_20260727_004934"
G2_ROOT = REPO_ROOT / "results" / "bridgepack_g2_llama1024_sourceorder_20260727_023300"
IND_FIX = REPO_ROOT / "results" / "bridgepack_g2_independent_fix_20260727_132221"
# Plan: fixed_random seed=0 (or SEED). Prefer explicit seed=0 for reproducibility.
FIXED_RANDOM_SEED = 0
BOOTSTRAP_SEED = 20260727

ORDER_VARIANTS = {
    "same_spans_source_order": "source_order",
    "same_spans_path_order": "path_order",
    "same_spans_query_score_order": "query_score_order",
    "same_spans_fixed_random_order": "random_order",
}

# Existing reusable prediction roots (variant -> (root, pred_subdir_variant))
REUSE_QWEN = {
    "same_spans_source_order": (G1_ROOT / "full" / "predictions", "same_spans_source_order"),
    "independent_topspan_neighbor_source_order": (G1_ROOT / "full" / "predictions", "independent_topspan_neighbor_source_order"),
    "mmr_neighbor_source_order": (G1_ROOT / "deferred" / "predictions", "mmr_neighbor_source_order"),
    "bridge_score_zero_source_order": (G1_ROOT / "deferred" / "predictions", "bridge_score_zero_source_order"),
}
REUSE_LLAMA = {
    "same_spans_source_order": (G2_ROOT / "full" / "predictions", "source_order_bridge_selection"),
    "independent_topspan_neighbor_source_order": (IND_FIX / "full" / "predictions", "independent_topspan_neighbor_source_order"),
    "mmr_neighbor_source_order": (G2_ROOT / "full" / "predictions", "mmr_neighbor_source_order"),
}


def _all_ids() -> Dict[str, List[str]]:
    out = {}
    for task in TASKS:
        rows = read_jsonl(
            REPO_ROOT
            / "results"
            / "bridgepack_exact_qwen_budget_1024_20260619_181059"
            / "packed_prompts_build"
            / "packed_prompts"
            / "lexical_bridgepack_path_order"
            / f"budget_{BUDGET}"
            / f"{task}.jsonl"
        )
        out[task] = [str(r["sample_id"]) for r in rows]
    return out


def _order_ids(frozen: Mapping[str, Any], order_mode: str) -> List[str]:
    by_id = {c.span_id: c for c in frozen["candidates"]}
    frozen_ids = list(frozen["span_ids"])
    spans = [by_id[i] for i in frozen_ids if i in by_id]
    if order_mode == "path_order":
        selected_set = set(frozen_ids)
        ordered = [s.span_id for s, _ in frozen["ordered_spans"] if s.span_id in selected_set]
        have = set(ordered)
        for s in spans:
            if s.span_id not in have:
                ordered.append(s.span_id)
        return ordered
    if order_mode == "source_order":
        return [s.span_id for s in sorted(spans, key=lambda x: x.source_order_index)]
    if order_mode == "query_score_order":
        return [s.span_id for s in sorted(spans, key=lambda x: (-x.query_score, x.source_order_index))]
    if order_mode == "random_order":
        # Deterministic per-sample shuffle with global seed=0 (plan fixed_random).
        rng = random.Random(FIXED_RANDOM_SEED ^ (hash(str(frozen["sample_id"])) & 0xFFFFFFFF))
        shuf = list(spans)
        rng.shuffle(shuf)
        return [s.span_id for s in shuf]
    raise ValueError(order_mode)


def _pack_order_only_payload(
    frozen: Mapping[str, Any],
    order_mode: str,
    vname: str,
    tokenizer,
    model_tokenizer_path: str,
    answers: Any,
    all_classes: Any,
    context: str,
) -> Dict[str, Any]:
    import json as _json
    from bridgekv import evidence_text_packing as etp

    by_id = {c.span_id: c for c in frozen["candidates"]}
    frozen_ids = list(frozen["span_ids"])
    frozen_ranges = [(r["span_id"], r["start_token"], r["end_token"]) for r in frozen["repaired_ranges"]]
    by_er = {er["span_id"]: er for er in frozen["audit"]["expanded_records"]}
    order_ids = _order_ids(frozen, order_mode)
    prefix = etp._prefix(frozen["query"], False)
    pieces, kept, repaired = [], [], []
    for iid in order_ids:
        er = by_er.get(iid)
        if not er:
            continue
        pieces.append(str(er.get("text") or ""))
        kept.append(iid)
        repaired.append(
            {"span_id": er.get("span_id"), "start_token": er.get("start_token"), "end_token": er.get("end_token")}
        )
    packed_text = "\n\n".join([prefix, *pieces]).strip()
    ntok = len(tokenizer(packed_text, add_special_tokens=False).input_ids)
    truncated = False
    if ntok > BUDGET:
        ids = tokenizer(packed_text, add_special_tokens=False).input_ids[:BUDGET]
        packed_text = tokenizer.decode(ids, skip_special_tokens=True)
        ntok = len(tokenizer(packed_text, add_special_tokens=False).input_ids)
        truncated = True
    templates = _json.loads((REPO_ROOT / "configs" / "dataset2prompt.json").read_text(encoding="utf-8"))
    template = str(templates.get(frozen["task"], ""))
    try:
        gen_prompt = template.format(context=packed_text, input=frozen["query"], question=frozen["query"])
    except Exception:
        gen_prompt = packed_text
    return {
        "sample_id": frozen["sample_id"],
        "task": frozen["task"],
        "variant": vname,
        "budget": BUDGET,
        "packed_text": packed_text,
        "generation_prompt": gen_prompt,
        "selected_span_ids": kept,
        "selected_page_ids": sorted({by_id[i].page_id for i in kept if i in by_id}),
        "packed_token_count": ntok,
        "model_token_count": ntok,
        "exact_model_token_budget": True,
        "model_tokenizer": str(model_tokenizer_path),
        "source_token_count": frozen["packed"].source_token_count,
        "compression_ratio": ntok / max(1, frozen["packed"].source_token_count),
        "ordering": order_mode,
        "neighbor_size": 32,
        "used_bridge_path": True,
        "used_tags": False,
        "truncated": truncated,
        "question": frozen["query"],
        "original_context": context,
        "answers": answers,
        "gold_analysis_only_answers": answers,
        "all_classes": all_classes,
        "repaired_ranges": repaired,
        "frozen_span_ids": frozen_ids,
        "matched_family": "order_only",
        "selector_reads_answers": False,
        "span_set_equal_frozen": set(kept) == set(frozen_ids),
        "repaired_equal_frozen": set((r["span_id"], r["start_token"], r["end_token"]) for r in repaired) == set(frozen_ranges),
        "method_config_sha256": FREEZE_SHA,
    }


def build_order_packs(
    out_dir: Path,
    model_name: str,
    variants: Sequence[str],
    sample_ids: Mapping[str, Sequence[str]],
    stage: str,
) -> Dict[str, Any]:
    from transformers import AutoTokenizer
    from bridgekv.bridgepack_cpu_extensions import freeze_lexical_bridgepack_selection

    model_path = str(QWEN_PATH if model_name == "qwen" else LLAMA_MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    hist = load_hist_lbp_rows(sample_ids)
    packed_root = out_dir / model_name / stage / "packed_prompts_build"
    collected: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    audit = []
    for task in TASKS:
        for hist_row in hist[task]:
            sid = str(hist_row["sample_id"])
            question = str(hist_row.get("question") or "")
            context = str(hist_row.get("original_context") or "")
            answers = hist_row.get("answers") or hist_row.get("gold_analysis_only_answers") or []
            frozen = freeze_lexical_bridgepack_selection(sid, task, question, context, BUDGET, tokenizer=tokenizer)
            for vname in variants:
                order_mode = ORDER_VARIANTS[vname]
                payload = _pack_order_only_payload(
                    frozen, order_mode, vname, tokenizer, model_path, answers, hist_row.get("all_classes"), context
                )
                collected.setdefault((vname, task), []).append(payload)
                audit.append(
                    {
                        "model": model_name,
                        "variant": vname,
                        "task": task,
                        "sample_id": sid,
                        "tokens": payload["model_token_count"],
                        "over_budget": int(payload["model_token_count"] > BUDGET),
                        "span_set_equal_frozen": int(payload["span_set_equal_frozen"]),
                        "truncated": int(payload["truncated"]),
                    }
                )
    for (v, t), rows in collected.items():
        write_jsonl(packed_root / "packed_prompts" / v / f"budget_{BUDGET}" / f"{t}.jsonl", rows)
    overs = sum(a["over_budget"] for a in audit)
    span_fail = sum(1 for a in audit if not a["span_set_equal_frozen"])
    write_json(
        out_dir / model_name / stage / "pack_summary.json",
        {"n": len(audit), "over_budget": overs, "span_set_fail": span_fail, "variants": list(variants)},
    )
    with (out_dir / model_name / stage / "pack_audit.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = list(audit[0].keys()) if audit else []
        w = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for row in audit:
            w.writerow(row)
    if overs:
        raise RuntimeError(f"{model_name}/{stage} over-budget packs: {overs}")
    if span_fail:
        raise RuntimeError(f"{model_name}/{stage} span_set_equal_frozen failures: {span_fail}")
    return {"audit": audit, "packed_root": packed_root}


def build_selection_packs(
    out_dir: Path,
    model_name: str,
    controls: Sequence[str],
    sample_ids: Mapping[str, Sequence[str]],
    stage: str,
) -> Dict[str, Any]:
    """Pack selection controls with neighbor=32 and source-order serialization."""
    from transformers import AutoTokenizer
    from bridgekv.bridgepack_cpu_extensions import score_component_control
    from bridgekv.evidence_text_packing import build_candidate_spans

    model_path = str(QWEN_PATH if model_name == "qwen" else LLAMA_MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    hist = load_hist_lbp_rows(sample_ids)
    packed_root = out_dir / model_name / stage / "packed_prompts_build"
    collected: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    audit = []
    control_to_variant = {
        "bridge_score_zero": "bridge_score_zero_source_order",
        "redundancy_penalty_zero": "redundancy_penalty_zero_source_order",
        "support_score_zero": "support_score_zero_source_order",
        "no_neighbor": "no_neighbor_source_order",
    }
    for task in TASKS:
        for hist_row in hist[task]:
            sid = str(hist_row["sample_id"])
            question = str(hist_row.get("question") or "")
            context = str(hist_row.get("original_context") or "")
            answers = hist_row.get("answers") or hist_row.get("gold_analysis_only_answers") or []
            candidates, page_map = build_candidate_spans(sid, task, question, context)
            by_id = {c.span_id: c for c in candidates}
            for control in controls:
                vname = control_to_variant[control]
                packed_obj, _audit = score_component_control(
                    sid, task, question, context, BUDGET, control, tokenizer=tokenizer
                )
                selected = [by_id[i] for i in packed_obj.selected_span_ids if i in by_id]
                # Force source-order packing for matched selection table
                neighbor = 0 if control == "no_neighbor" else 32
                ordered = [(s, "bridge") for s in sorted(selected, key=lambda x: x.source_order_index)]
                payload = _exact_pack_from_ordered(
                    tokenizer,
                    sid,
                    task,
                    question,
                    context,
                    vname,
                    ordered,
                    page_map,
                    neighbor,
                    True,
                    "source_order",
                    audit_extra={
                        "matched_family": "selection_score_component",
                        "control": control,
                        "answers": answers,
                        "gold_analysis_only_answers": answers,
                        "all_classes": hist_row.get("all_classes"),
                        "method_config_sha256": FREEZE_SHA,
                    },
                )
                payload["model_tokenizer"] = model_path
                payload["budget"] = BUDGET
                collected.setdefault((vname, task), []).append(payload)
                mt = int(payload.get("model_token_count") or 0)
                audit.append(
                    {
                        "model": model_name,
                        "variant": vname,
                        "control": control,
                        "task": task,
                        "sample_id": sid,
                        "tokens": mt,
                        "over_budget": int(mt > BUDGET),
                        "n_selected": len(payload.get("selected_span_ids") or []),
                    }
                )
    for (v, t), rows in collected.items():
        write_jsonl(packed_root / "packed_prompts" / v / f"budget_{BUDGET}" / f"{t}.jsonl", rows)
    overs = sum(a["over_budget"] for a in audit)
    write_json(out_dir / model_name / stage / "selection_pack_summary.json", {"n": len(audit), "over_budget": overs})
    with (out_dir / model_name / stage / "selection_pack_audit.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = list(audit[0].keys()) if audit else []
        w = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for row in audit:
            w.writerow(row)
    if overs:
        raise RuntimeError(f"selection over-budget: {overs}")
    return {"audit": audit, "packed_root": packed_root}


def run_qa(
    out_dir: Path,
    model_name: str,
    packed_stage: str,
    pred_stage: str,
    variants: Sequence[str],
    sample_ids: Mapping[str, Sequence[str]],
) -> None:
    import tools.run_evidence_text_packing_generation_qwen32 as gen
    from tools.bridgepack_g1.generate import generate_greedy_historical

    unit = out_dir / model_name
    gate = g2_safety_gate(unit, f"{pred_stage}_before_load", phase="before_load")
    if gate["decision"] != "GO":
        write_text(unit / "WAITING_FOR_SAFE_GPU.md", json.dumps(gate, indent=2))
        raise RuntimeError(f"WAITING_FOR_SAFE_GPU: {gate['reasons']}")
    model_path = str(QWEN_PATH if model_name == "qwen" else LLAMA_MODEL_PATH)
    torch = __import__("torch")
    model, tokenizer = load_qwen_fp16(model_path, torch)
    g2_safety_gate(unit, f"{pred_stage}_after_load", phase="runtime")
    pred_root = unit / pred_stage / "predictions"
    packed_root = unit / packed_stage / "packed_prompts_build" / "packed_prompts"
    for variant in variants:
        for task in TASKS:
            path = packed_root / variant / f"budget_{BUDGET}" / f"{task}.jsonl"
            rows = read_jsonl(path)
            by_id = {str(r["sample_id"]): r for r in rows}
            want = list(sample_ids[task])
            rows = [by_id[i] for i in want if i in by_id]
            if len(rows) != len(want):
                raise RuntimeError(f"Missing packs {variant}/{task}: {len(rows)}/{len(want)}")
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
                    gate = g2_safety_gate(unit, f"{pred_stage}_{variant}_{task}_{sample_index}", phase="runtime")
                    if gate["decision"] != "GO":
                        raise RuntimeError(f"Safety stop: {gate['reasons']}")
                    mt = int(row.get("model_token_count") or row.get("packed_token_count") or 0)
                    if mt > BUDGET:
                        raise RuntimeError(f"Over-budget QA: {variant}/{task}/{sid}")
                    prompt = str(row.get("generation_prompt") or "") or gen._generation_prompt_from_row(task, row)
                    max_new = gen._max_new_tokens_for(task, 0)
                    pred, context_length = generate_greedy_historical(model, tokenizer, torch, prompt, max_new)
                    payload = {
                        "sample_id": sid,
                        "sample_index": sample_index,
                        "task": task,
                        "variant": variant,
                        "pred": pred,
                        "answers": row.get("answers", row.get("gold_analysis_only_answers", [])),
                        "all_classes": row.get("all_classes"),
                        "length": context_length,
                        "packed_token_count": mt,
                        "budget": BUDGET,
                        "question": row.get("question", ""),
                    }
                    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                    handle.flush()
    score_predictions(pred_root, list(variants))
    del model
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass


def link_reused_predictions(out_dir: Path, model_name: str) -> None:
    reuse = REUSE_QWEN if model_name == "qwen" else REUSE_LLAMA
    dest_root = out_dir / model_name / "full" / "predictions"
    for vname, (src_root, src_variant) in reuse.items():
        for task in TASKS:
            src = src_root / src_variant / f"{task}.jsonl"
            if not src.exists():
                # try alternate name for bridge_zero
                alt = src_root / "bridge_score_zero" / f"{task}.jsonl"
                src = alt if alt.exists() else src
            if not src.exists():
                continue
            dst = dest_root / vname / f"{task}.jsonl"
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists() or dst.is_symlink():
                continue
            os.symlink(src.resolve(), dst)


def write_order_selection_stats(out_dir: Path) -> None:
    order_rows = []
    boot_rows = []
    holm_inputs = []
    for model_name, reuse_map in [("qwen", REUSE_QWEN), ("llama3", REUSE_LLAMA)]:
        pred_root = out_dir / model_name / "full" / "predictions"
        # discover variants present
        if not pred_root.exists():
            continue
        variants = sorted(p.name for p in pred_root.iterdir() if p.is_dir())
        for v in variants:
            for row in score_predictions(pred_root, [v]):
                row = dict(row)
                row["model"] = model_name
                row["budget"] = BUDGET
                order_rows.append(row)
        # pairwise vs source
        if "same_spans_source_order" not in variants:
            continue
        source = load_pred_scores(pred_root, "same_spans_source_order", TASKS)
        for v in variants:
            if v == "same_spans_source_order":
                continue
            other = load_pred_scores(pred_root, v, TASKS)
            boot = paired_bootstrap_delta(source, other)
            if int(boot.get("n", 0)) <= 0:
                continue
            label = f"{model_name} source_order vs {v}"
            p_raw = approx_p_from_bootstrap(boot["delta_macro"], boot.get("boot_std", float("nan")))
            holm_inputs.append((label, p_raw))
            boot_rows.append(
                {
                    "comparison": label,
                    "model": model_name,
                    "n": boot["n"],
                    "delta_macro": boot["delta_macro"],
                    "ci95_low": boot["ci95_low"],
                    "ci95_high": boot["ci95_high"],
                    "crosses_zero": boot["crosses_zero"],
                    **{f"delta_{t}": boot.get("task_deltas", {}).get(t, "") for t in TASKS},
                }
            )
    if holm_inputs:
        holm = {r["label"]: r for r in holm_correction(holm_inputs)}
        for row in boot_rows:
            h = holm[row["comparison"]]
            row["p_raw"] = h["p_raw"]
            row["p_holm"] = h["p_holm"]
            row["reject_holm_0.05"] = h["reject_holm_0.05"]

    # split order vs selection by variant name
    order_names = set(ORDER_VARIANTS)
    with (out_dir / "MATCHED_ORDER_RESULTS.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["model", "budget", "variant", "task", "n", "score", "mean_length"]
        w = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for row in order_rows:
            if row["variant"] in order_names or row["variant"] == "same_spans_source_order":
                w.writerow(row)
    with (out_dir / "MATCHED_SELECTION_RESULTS.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["model", "budget", "variant", "task", "n", "score", "mean_length"]
        w = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for row in order_rows:
            if row["variant"] not in order_names:
                w.writerow(row)
    with (out_dir / "MATCHED_ORDER_BOOTSTRAP.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["comparison", "model", "n", "delta_macro", "ci95_low", "ci95_high", "crosses_zero", "p_raw", "p_holm", "reject_holm_0.05"] + [
            f"delta_{t}" for t in TASKS
        ]
        w = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for row in boot_rows:
            if any(x in row["comparison"] for x in ("path_order", "query_score", "fixed_random", "random")):
                w.writerow(row)
    with (out_dir / "MATCHED_SELECTION_BOOTSTRAP.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["comparison", "model", "n", "delta_macro", "ci95_low", "ci95_high", "crosses_zero", "p_raw", "p_holm", "reject_holm_0.05"] + [
            f"delta_{t}" for t in TASKS
        ]
        w = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for row in boot_rows:
            if not any(x in row["comparison"] for x in ("path_order", "query_score", "fixed_random", "random")):
                w.writerow(row)

    write_text(
        out_dir / "MECHANISM_CLAIM_GUARD.md",
        f"""# MECHANISM_CLAIM_GUARD

Generated: {utc_now()}

| 声明 | 允许？ | 依据 |
|---|---|---|
| Qwen-1024 source-order 宏平均 39.39 | YES | G1 same_spans_source_order 精确 39.385 |
| Path-order 已证优于 source-order | 仅当 bootstrap 拒绝且 CI 不含 0 | 见 MATCHED_ORDER_BOOTSTRAP |
| Source-order 作为默认输出顺序 | 若 path vs source 不显著则可 YES | 协议默认，非已证更优机制 |
| Bridge selection 是增益的唯一来源 | 仅当 vs independent/MMR/分数消融支持 | 见 MATCHED_SELECTION_BOOTSTRAP |
| 引用 38.05 作为 source-order | **NO** | 已作废数字 |
| 本方法是 KV-cache pruning | **NO** | 一律禁止 |

Config SHA: `{FREEZE_SHA}`
""",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--model", choices=["qwen", "llama3", "all"], default="all")
    parser.add_argument(
        "--mode",
        choices=["pack_order", "pack_selection", "qa", "link_reuse", "finalize", "ladder_order", "ladder_selection"],
        required=True,
    )
    parser.add_argument("--stage", default="full", choices=["smoke12", "gate60", "full"])
    parser.add_argument("--stop_after", default="", choices=["", "smoke12", "gate60", "full"])
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    models = ["qwen", "llama3"] if args.model == "all" else [args.model]
    ids_full = _all_ids()
    ids12 = select_fixed_sample_ids(2)
    ids60 = select_fixed_sample_ids(10)

    if args.mode == "link_reuse":
        for m in models:
            link_reused_predictions(out_dir, m)
        return 0

    if args.mode == "pack_order":
        qwen_vars = ["same_spans_path_order", "same_spans_query_score_order", "same_spans_fixed_random_order"]
        llama_vars = ["same_spans_path_order", "same_spans_query_score_order"]
        for m in models:
            variants = qwen_vars if m == "qwen" else llama_vars
            sample_ids = ids_full if args.stage == "full" else (ids12 if args.stage == "smoke12" else ids60)
            print(f"[pack_order] {m} {args.stage} {variants}", flush=True)
            build_order_packs(out_dir, m, variants, sample_ids, args.stage)
        return 0

    if args.mode == "pack_selection":
        qwen_ctrls = ["redundancy_penalty_zero", "support_score_zero", "no_neighbor"]
        llama_ctrls = ["bridge_score_zero"]
        for m in models:
            controls = qwen_ctrls if m == "qwen" else llama_ctrls
            sample_ids = ids_full if args.stage == "full" else (ids12 if args.stage == "smoke12" else ids60)
            print(f"[pack_selection] {m} {args.stage} {controls}", flush=True)
            build_selection_packs(out_dir, m, controls, sample_ids, args.stage)
        return 0

    if args.mode == "qa":
        for m in models:
            packed_stage = args.stage
            # variants = dirs under packed
            packed_root = out_dir / m / packed_stage / "packed_prompts_build" / "packed_prompts"
            variants = sorted(p.name for p in packed_root.iterdir() if p.is_dir()) if packed_root.exists() else []
            sample_ids = ids_full if args.stage == "full" else (ids12 if args.stage == "smoke12" else ids60)
            print(f"[qa] {m} {args.stage} {variants}", flush=True)
            run_qa(out_dir, m, packed_stage, args.stage, variants, sample_ids)
        return 0

    if args.mode == "finalize":
        for m in models:
            link_reused_predictions(out_dir, m)
        write_order_selection_stats(out_dir)
        write_text(out_dir / "WAVE1_MATCHED.DONE", f"{utc_now()}\n")
        return 0

    if args.mode in {"ladder_order", "ladder_selection"}:
        pack_mode = "pack_order" if args.mode == "ladder_order" else "pack_selection"
        stop = args.stop_after or "full"
        stages = ["smoke12", "gate60", "full"]
        for stage in stages:
            done_marker = out_dir / f"LADDER_{pack_mode.upper()}_{stage.upper()}.DONE"
            if done_marker.exists() and stage != "full":
                print(f"[ladder] skip completed {pack_mode}/{stage}", flush=True)
                if stage == stop:
                    break
                continue
            # pack
            if pack_mode == "pack_order":
                qwen_vars = ["same_spans_path_order", "same_spans_query_score_order", "same_spans_fixed_random_order"]
                llama_vars = ["same_spans_path_order", "same_spans_query_score_order"]
                for m in models:
                    variants = qwen_vars if m == "qwen" else llama_vars
                    sample_ids = ids_full if stage == "full" else (ids12 if stage == "smoke12" else ids60)
                    # Skip pack if already present for this stage/variant
                    packed_root = out_dir / m / stage / "packed_prompts_build" / "packed_prompts"
                    need_pack = False
                    for v in variants:
                        for task in TASKS:
                            p = packed_root / v / f"budget_{BUDGET}" / f"{task}.jsonl"
                            if not p.exists() or len(read_jsonl(p)) < len(sample_ids[task]):
                                need_pack = True
                                break
                        if need_pack:
                            break
                    if need_pack:
                        print(f"[pack_order] {m} {stage} {variants}", flush=True)
                        build_order_packs(out_dir, m, variants, sample_ids, stage)
                    else:
                        print(f"[pack_order] reuse packs {m}/{stage}", flush=True)
            else:
                qwen_ctrls = ["redundancy_penalty_zero", "support_score_zero", "no_neighbor"]
                llama_ctrls = ["bridge_score_zero"]
                for m in models:
                    controls = qwen_ctrls if m == "qwen" else llama_ctrls
                    sample_ids = ids_full if stage == "full" else (ids12 if stage == "smoke12" else ids60)
                    packed_root = out_dir / m / stage / "packed_prompts_build" / "packed_prompts"
                    control_to_variant = {
                        "bridge_score_zero": "bridge_score_zero_source_order",
                        "redundancy_penalty_zero": "redundancy_penalty_zero_source_order",
                        "support_score_zero": "support_score_zero_source_order",
                        "no_neighbor": "no_neighbor_source_order",
                    }
                    variants = [control_to_variant[c] for c in controls]
                    need_pack = False
                    for v in variants:
                        for task in TASKS:
                            p = packed_root / v / f"budget_{BUDGET}" / f"{task}.jsonl"
                            if not p.exists() or len(read_jsonl(p)) < len(sample_ids[task]):
                                need_pack = True
                                break
                        if need_pack:
                            break
                    if need_pack:
                        print(f"[pack_selection] {m} {stage} {controls}", flush=True)
                        build_selection_packs(out_dir, m, controls, sample_ids, stage)
                    else:
                        print(f"[pack_selection] reuse packs {m}/{stage}", flush=True)
            # qa
            for m in models:
                packed_root = out_dir / m / stage / "packed_prompts_build" / "packed_prompts"
                variants = sorted(p.name for p in packed_root.iterdir() if p.is_dir())
                sample_ids = ids_full if stage == "full" else (ids12 if stage == "smoke12" else ids60)
                run_qa(out_dir, m, stage, stage, variants, sample_ids)
            write_text(out_dir / f"LADDER_{pack_mode.upper()}_{stage.upper()}.DONE", f"{utc_now()}\n")
            if stage == stop:
                break
        if stop == "full":
            for m in models:
                link_reused_predictions(out_dir, m)
            write_order_selection_stats(out_dir)
            write_text(out_dir / "WAVE1_MATCHED.DONE", f"{utc_now()}\n")
        return 0

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        out = Path(sys.argv[sys.argv.index("--out_dir") + 1]) if "--out_dir" in sys.argv else Path(".")
        write_text(out / "P0_MATCHED_FATAL.md", f"{exc}\n\n{traceback.format_exc()}")
        raise
