#!/usr/bin/env python3
"""G1 deferred GPU runs: full MMR + bridge-score-zero (Qwen@1024)."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.bridgepack_cpu_stage.common import FAIRNESS_DIR, read_csv
from tools.bridgepack_g1.common import (
    BUDGET,
    HIST_ROOT,
    LBP_VARIANT,
    QWEN_PATH,
    TASKS,
    load_hist_lbp_rows,
    read_jsonl,
    safety_gate,
    utc_now,
    write_json,
    write_jsonl,
    write_text,
)
from tools.bridgepack_g1.run_g1 import _exact_pack_from_ordered, build_matched_variant_packs, run_qa_on_packs, score_predictions
from tools.bridgepack_g1.stats import (
    BOOTSTRAP_N,
    approx_p_from_bootstrap,
    holm_correction,
    load_pred_scores,
    paired_bootstrap_delta,
)

MMR_VARIANT = "mmr_neighbor_source_order"
BRIDGE_ZERO_VARIANT = "bridge_score_zero_source_order"
MAIN_VARIANT = "same_spans_source_order"
INDEP_VARIANT = "independent_topspan_neighbor_source_order"

# Frozen before any test-set QA (do not tune on results).
MMR_HYPERPARAMS = {
    "lam": 0.7,
    "max_items": 64,
    "neighbor_size": 32,
    "ordering": "source_order",
    "redundancy": "jaccard_content_tokens",
    "selection_rule": "mmr_relevance_redundancy",
    "frozen_at": "2026-07-27T09:00:00Z",
    "note": "Defaults from bridgekv.bridgepack_cpu_extensions.mmr_select; not tuned on test set.",
}

BRIDGE_ZERO_HYPERPARAMS = {
    "control": "bridge_score_zero",
    "neighbor_size": 32,
    "ordering": "source_order",
    "only_change": "bridge_score set to 0.0 before bridge-path selection",
    "frozen_at": "2026-07-27T09:00:00Z",
}

JACCARD_NOOP_THRESHOLD = 0.98  # if mean Jaccard >= this vs full selection, stop QA


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _fairness_scores(method: str) -> Dict[Tuple[str, str], float]:
    out: Dict[Tuple[str, str], float] = {}
    for r in read_csv(FAIRNESS_DIR / "per_sample_scores.csv"):
        if r["model"] == "qwen" and int(r["budget"]) == 1024 and r["method"] == method:
            out[(r["task"], r["sample_id"])] = float(r["score_0_100"])
    return out


def pack_mmr_full(out_dir: Path, hist_full: Mapping[str, Sequence[Mapping[str, Any]]], tokenizer) -> Dict[str, Any]:
    from bridgekv.bridgepack_cpu_extensions import mmr_select
    from bridgekv.evidence_text_packing import build_candidate_spans, lexical_scored_candidates

    stage = "deferred"
    packed_root = out_dir / stage / "packed_prompts_build"
    rows_out: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    overs = 0
    for task in TASKS:
        for hist in hist_full[task]:
            sid = str(hist["sample_id"])
            question = str(hist.get("question") or "")
            context = str(hist.get("original_context") or "")
            answers = hist.get("answers") or hist.get("gold_analysis_only_answers") or []
            candidates, page_map = build_candidate_spans(sid, task, question, context)
            lexical = lexical_scored_candidates(question, candidates)
            scores = {c.span_id: float(c.final_score) for c in lexical}
            selected = mmr_select(
                lexical,
                scores,
                page_map,
                BUDGET,
                question,
                neighbor_size=MMR_HYPERPARAMS["neighbor_size"],
                lam=MMR_HYPERPARAMS["lam"],
                max_items=MMR_HYPERPARAMS["max_items"],
                tokenizer=tokenizer,
            )
            ordered = [(s, "mmr") for s in sorted(selected, key=lambda x: x.source_order_index)]
            payload = _exact_pack_from_ordered(
                tokenizer,
                sid,
                task,
                question,
                context,
                MMR_VARIANT,
                ordered,
                page_map,
                MMR_HYPERPARAMS["neighbor_size"],
                False,
                MMR_HYPERPARAMS["ordering"],
                audit_extra={
                    "answers": answers,
                    "gold_analysis_only_answers": answers,
                    "all_classes": hist.get("all_classes"),
                    "mmr_hyperparams": MMR_HYPERPARAMS,
                    "matched_family": "strong_baseline",
                },
            )
            if int(payload["model_token_count"]) > BUDGET:
                overs += 1
            rows_out.setdefault((MMR_VARIANT, task), []).append(payload)
    for (v, t), rows in rows_out.items():
        write_jsonl(packed_root / "packed_prompts" / v / f"budget_{BUDGET}" / f"{t}.jsonl", rows)
    summary = [{"variant": v, "task": t, "n": len(rows), "over_budget": sum(1 for r in rows if int(r.get("model_token_count", 0)) > BUDGET)} for (v, t), rows in sorted(rows_out.items())]
    write_json(out_dir / stage / "mmr_pack_summary.json", summary)
    return {"packed_root": packed_root, "overs": overs, "summary": summary}


def pack_bridge_zero_full(out_dir: Path, hist_full: Mapping[str, Sequence[Mapping[str, Any]]], tokenizer) -> Dict[str, Any]:
    from bridgekv.bridgepack_cpu_extensions import score_component_control
    from bridgekv.evidence_text_packing import build_candidate_spans

    stage = "deferred"
    packed_root = out_dir / stage / "packed_prompts_build"
    # reference full selection spans from same_spans packs
    ref_pack = out_dir / "full" / "packed_prompts_build" / "packed_prompts" / MAIN_VARIANT / f"budget_{BUDGET}"
    jaccards: List[float] = []
    changed = 0
    token_diffs: List[int] = []
    rows_out: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    overs = 0
    for task in TASKS:
        ref_by_id = {str(r["sample_id"]): r for r in read_jsonl(ref_pack / f"{task}.jsonl")}
        for hist in hist_full[task]:
            sid = str(hist["sample_id"])
            question = str(hist.get("question") or "")
            context = str(hist.get("original_context") or "")
            answers = hist.get("answers") or hist.get("gold_analysis_only_answers") or []
            packed_obj, audit = score_component_control(
                sid, task, question, context, BUDGET, "bridge_score_zero", tokenizer=tokenizer
            )
            candidates, page_map = build_candidate_spans(sid, task, question, context)
            by_id = {c.span_id: c for c in candidates}
            selected = [by_id[i] for i in packed_obj.selected_span_ids if i in by_id]
            sel_set = set(packed_obj.selected_span_ids)
            ref_set = set(ref_by_id.get(sid, {}).get("selected_span_ids") or [])
            jac = _jaccard(sel_set, ref_set)
            jaccards.append(jac)
            if jac < 1.0:
                changed += 1
            ordered = [(s, "bridge_zero") for s in sorted(selected, key=lambda x: x.source_order_index)]
            payload = _exact_pack_from_ordered(
                tokenizer,
                sid,
                task,
                question,
                context,
                BRIDGE_ZERO_VARIANT,
                ordered,
                page_map,
                BRIDGE_ZERO_HYPERPARAMS["neighbor_size"],
                True,
                BRIDGE_ZERO_HYPERPARAMS["ordering"],
                audit_extra={
                    "answers": answers,
                    "gold_analysis_only_answers": answers,
                    "all_classes": hist.get("all_classes"),
                    "control": BRIDGE_ZERO_HYPERPARAMS,
                    "jaccard_vs_full_selection": jac,
                    "span_set_changed": jac < 1.0,
                },
            )
            ref_tok = int(ref_by_id.get(sid, {}).get("model_token_count") or 0)
            token_diffs.append(int(payload["model_token_count"]) - ref_tok)
            if int(payload["model_token_count"]) > BUDGET:
                overs += 1
            rows_out.setdefault((BRIDGE_ZERO_VARIANT, task), []).append(payload)

    mean_jac = sum(jaccards) / max(1, len(jaccards))
    audit = {
        "n": len(jaccards),
        "mean_jaccard_vs_full_selection": mean_jac,
        "changed_rate": changed / max(1, len(jaccards)),
        "mean_token_diff_vs_full": sum(token_diffs) / max(1, len(token_diffs)),
        "noop_threshold": JACCARD_NOOP_THRESHOLD,
        "is_noop": mean_jac >= JACCARD_NOOP_THRESHOLD,
    }
    write_json(out_dir / stage / "bridge_zero_selection_audit.json", audit)
    if audit["is_noop"]:
        return {"packed_root": packed_root, "overs": overs, "audit": audit, "noop": True}

    for (v, t), rows in rows_out.items():
        write_jsonl(packed_root / "packed_prompts" / v / f"budget_{BUDGET}" / f"{t}.jsonl", rows)
    write_json(out_dir / stage / "bridge_zero_pack_summary.json", [{"variant": v, "task": t, "n": len(rows), "over_budget": sum(1 for r in rows if int(r.get("model_token_count", 0)) > BUDGET)} for (v, t), rows in sorted(rows_out.items())])
    return {"packed_root": packed_root, "overs": overs, "audit": audit, "noop": False}


def smoke12_interface(out_dir: Path, variant: str, hist_rows_12, tokenizer) -> None:
    """Historical interface smoke: 12 samples, pack only."""
    if variant == MMR_VARIANT:
        pack_mmr_full(out_dir, hist_rows_12, tokenizer)
    else:
        pack_bridge_zero_full(out_dir, hist_rows_12, tokenizer)


def write_deferred_bootstrap(out_dir: Path, mmr_scores, bridge_scores) -> None:
    same = load_pred_scores(out_dir / "full" / "predictions", MAIN_VARIANT, TASKS)
    indep = load_pred_scores(out_dir / "full" / "predictions", INDEP_VARIANT, TASKS)
    bge = _fairness_scores("bge_chunk_packing")
    bm25 = _fairness_scores("bm25_chunk_source_order")

    specs = [
        ("same_spans_source_order vs mmr_neighbor_source_order", same, mmr_scores),
        ("mmr_neighbor_source_order vs independent_topspan_neighbor_source_order", mmr_scores, indep),
        ("mmr_neighbor_source_order vs bge_chunk_packing", mmr_scores, bge),
        ("mmr_neighbor_source_order vs bm25_chunk_source_order", mmr_scores, bm25),
        ("same_spans_source_order vs bridge_score_zero_source_order", same, bridge_scores),
    ]
    rows = []
    p_inputs = []
    for label, a, b in specs:
        if not a or not b:
            continue
        boot = paired_bootstrap_delta(a, b)
        p_raw = approx_p_from_bootstrap(boot["delta_macro"], boot["boot_std"])
        p_inputs.append((label, p_raw))
        row = {
            "comparison": label,
            "n": boot["n"],
            "delta_macro": boot["delta_macro"],
            "ci95_low": boot["ci95_low"],
            "ci95_high": boot["ci95_high"],
            "crosses_zero": boot["crosses_zero"],
        }
        for t in TASKS:
            row[f"delta_{t}"] = boot["task_deltas"].get(t, "")
        rows.append(row)
    holm = {h["label"]: h for h in holm_correction(p_inputs)}
    for row in rows:
        h = holm.get(row["comparison"], {})
        row["p_raw"] = h.get("p_raw", "")
        row["p_holm"] = h.get("p_holm", "")
        row["reject_holm_0.05"] = h.get("reject_holm_0.05", "")

    path = out_dir / "G1_DEFERRED_BOOTSTRAP.csv"
    fields = ["comparison", "n", "delta_macro", "ci95_low", "ci95_high", "crosses_zero", "p_raw", "p_holm", "reject_holm_0.05"] + [f"delta_{t}" for t in TASKS]
    with path.open("w", encoding="utf-8", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def mechanism_decision(out_dir: Path, mmr_scores, bridge_scores, bridge_audit: Mapping[str, Any]) -> str:
    same = load_pred_scores(out_dir / "full" / "predictions", MAIN_VARIANT, TASKS)
    indep = load_pred_scores(out_dir / "full" / "predictions", INDEP_VARIANT, TASKS)
    c_sel = paired_bootstrap_delta(same, indep)
    c_mmr = paired_bootstrap_delta(same, mmr_scores) if mmr_scores else {"delta_macro": float("nan"), "crosses_zero": True, "ci95_low": float("nan"), "ci95_high": float("nan")}
    c_bz = paired_bootstrap_delta(same, bridge_scores) if bridge_scores else {"delta_macro": float("nan"), "crosses_zero": True, "ci95_low": float("nan"), "ci95_high": float("nan")}

    sel_ok = c_sel["delta_macro"] > 0 and not c_sel["crosses_zero"]
    mmr_ok = c_mmr["delta_macro"] > 0 and not c_mmr["crosses_zero"]
    bz_ok = c_bz["delta_macro"] > 0 and not c_bz["crosses_zero"]
    bz_noop = bool(bridge_audit.get("is_noop"))

    if sel_ok and mmr_ok and bz_ok and not bz_noop:
        scenario = "A_IDEAL"
        decision = "GO_G2_FULL"
        claim = "Bridge-aware selection contributes beyond relevance-only and MMR; source-order serialization preferred over path-order."
    elif sel_ok and not mmr_ok:
        scenario = "B_BEATS_INDEP_NOT_MMR"
        decision = "GO_G2_SLIM"
        claim = "Structured selection beats independent top-span but not standard MMR; emphasize neighbor repair + structured candidates."
    elif bz_noop or (bridge_scores and c_bz["crosses_zero"] and abs(c_bz["delta_macro"]) < 0.5):
        scenario = "C_BRIDGE_SCORE_NOOP"
        decision = "HOLD_REWRITE"
        claim = "Bridge connectivity score appears non-functional; rewrite as lexical structured selection + neighbor repair + source-order packing."
    elif mmr_scores and c_mmr["delta_macro"] < -1.0 and not c_mmr["crosses_zero"]:
        scenario = "D_MMR_STRONGER"
        decision = "NO_GO_G2"
        claim = "MMR clearly stronger; stop G2 expansion and reposition paper."
    else:
        scenario = "MIXED_HOLD"
        decision = "HOLD"
        claim = "Mixed deferred evidence; do not auto-start G2."

    md = f"""# G1_FINAL_MECHANISM_DECISION

Generated: {utc_now()}

## Scenario
**{scenario}** → decision: **{decision}**

## Core comparisons (macro Δ, 95% CI)
| comparison | Δ | CI | crosses 0 |
|---|---:|---|---|
| same_spans − independent (selection) | {c_sel['delta_macro']:.4f} | [{c_sel['ci95_low']:.4f}, {c_sel['ci95_high']:.4f}] | {c_sel['crosses_zero']} |
| same_spans − MMR | {c_mmr['delta_macro']:.4f} | [{c_mmr.get('ci95_low', float('nan')):.4f}, {c_mmr.get('ci95_high', float('nan')):.4f}] | {c_mmr['crosses_zero']} |
| same_spans − bridge_score_zero | {c_bz['delta_macro']:.4f} | [{c_bz.get('ci95_low', float('nan')):.4f}, {c_bz.get('ci95_high', float('nan')):.4f}] | {c_bz['crosses_zero']} |

## Bridge-score-zero selection audit
- mean Jaccard vs full selection: {bridge_audit.get('mean_jaccard_vs_full_selection', 'NA')}
- changed rate: {bridge_audit.get('changed_rate', 'NA')}
- noop flagged: {bz_noop}

## Recommended claim (draft)
{claim}

## G2
See G2_GO_NO_GO.md — **not auto-started**.
"""
    write_text(out_dir / "G1_FINAL_MECHANISM_DECISION.md", md)

    g2 = f"""# G2_GO_NO_GO

Generated: {utc_now()}

## Decision
**{decision}**

## Rationale
- Scenario: {scenario}
- Selection vs independent: {'PASS' if sel_ok else 'FAIL'}
- Selection vs MMR: {'PASS' if mmr_ok else 'FAIL/NA'}
- Selection vs bridge-score-zero: {'PASS' if bz_ok else 'FAIL/NA'}
- Bridge score noop: {bz_noop}

## If GO
- Llama-3 @1024: source-order bridge selection, independent top-span, MMR{'' if decision == 'GO_G2_FULL' else ' (slim set)'}
- bridge-score-zero: {'include' if decision == 'GO_G2_FULL' else 'defer'}

## Status
G2 **not started** in this run.
"""
    write_text(out_dir / "G2_GO_NO_GO.md", g2)

    guard = f"""# G1_REVISED_CLAIM_GUARD

Generated: {utc_now()}

| Claim | Allowed? |
|---|---|
| Bridge-aware selection > independent top-span (Qwen@1024) | {'YES' if sel_ok else 'NO'} |
| Bridge-aware selection > MMR (Qwen@1024) | {'YES' if mmr_ok else 'NO/CONDITIONAL'} |
| Bridge connectivity score is functional component | {'NO' if bz_noop else ('YES' if bz_ok else 'CONDITIONAL')} |
| Path-order packing effective | **NO** |
| Source-order is default serialization | **YES** |
| Auto-start G2 | **NO** (manual gate: {decision}) |
"""
    write_text(out_dir / "G1_REVISED_CLAIM_GUARD.md", guard)
    return decision


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--skip_gpu", action="store_true", help="pack/audit only")
    parser.add_argument("--skip_pack", action="store_true", help="reuse existing deferred packs")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Ensure BridgePack GPU worker can see the device (CPU-stage scripts may set "").
    import os
    if not args.skip_gpu:
        if os.environ.get("CUDA_VISIBLE_DEVICES", None) == "":
            del os.environ["CUDA_VISIBLE_DEVICES"]

    gate = safety_gate(out_dir, "deferred_preflight", phase="before_load")
    if gate["decision"] != "GO" and not args.skip_gpu:
        write_text(out_dir / "WAITING_FOR_SAFE_GPU.md", json.dumps(gate, indent=2))
        print("WAITING_FOR_SAFE_GPU", gate["reasons"])
        return 2

    write_json(out_dir / "deferred" / "MMR_HYPERPARAMS.json", MMR_HYPERPARAMS)
    write_json(out_dir / "deferred" / "BRIDGE_ZERO_HYPERPARAMS.json", BRIDGE_ZERO_HYPERPARAMS)

    from transformers import AutoTokenizer
    from tools.bridgepack_g1.common import select_fixed_sample_ids

    tokenizer = AutoTokenizer.from_pretrained(str(QWEN_PATH), trust_remote_code=True)
    sample_ids_full = {t: [str(r["sample_id"]) for r in read_jsonl(HIST_ROOT / "packed_prompts_build" / "packed_prompts" / LBP_VARIANT / f"budget_{BUDGET}" / f"{t}.jsonl")] for t in TASKS}
    hist_full = load_hist_lbp_rows(sample_ids_full)
    hist_12 = load_hist_lbp_rows(select_fixed_sample_ids(2))

    print("[deferred] CPU pack MMR full...", flush=True)
    if args.skip_pack and (out_dir / "deferred" / "mmr_pack_summary.json").exists():
        mmr_pack = {"packed_root": out_dir / "deferred" / "packed_prompts_build", "overs": 0}
        print("[deferred] reusing existing MMR packs", flush=True)
    else:
        mmr_pack = pack_mmr_full(out_dir, hist_full, tokenizer)
    if mmr_pack["overs"]:
        write_text(out_dir / "G1_FATAL_DEFERRED.md", f"MMR over-budget: {mmr_pack['overs']}\n")
        return 1

    print("[deferred] CPU pack bridge-score-zero + selection audit...", flush=True)
    if args.skip_pack and (out_dir / "deferred" / "bridge_zero_selection_audit.json").exists():
        audit = json.loads((out_dir / "deferred" / "bridge_zero_selection_audit.json").read_text())
        bz_pack = {
            "packed_root": out_dir / "deferred" / "packed_prompts_build",
            "overs": 0,
            "audit": audit,
            "noop": audit.get("is_noop", False),
        }
        print("[deferred] reusing existing bridge-zero packs/audit", flush=True)
    else:
        bz_pack = pack_bridge_zero_full(out_dir, hist_full, tokenizer)
    if bz_pack.get("noop"):
        write_text(
            out_dir / "G1_BRIDGE_ZERO_FULL_RESULTS.csv",
            "status,noop,mean_jaccard\nSTOPPED,True,{}\n".format(bz_pack["audit"]["mean_jaccard_vs_full_selection"]),
        )
        mechanism_decision(out_dir, {}, {}, bz_pack["audit"])
        print("bridge_score_zero NOOP — QA skipped", flush=True)
        return 1
    if bz_pack["overs"]:
        write_text(out_dir / "G1_FATAL_DEFERRED.md", f"bridge_zero over-budget: {bz_pack['overs']}\n")
        return 1

    if args.skip_gpu:
        print("skip_gpu set; packs ready", flush=True)
        return 0

    import torch as _torch
    if not _torch.cuda.is_available():
        write_text(out_dir / "WAITING_FOR_SAFE_GPU.md", "CUDA not available after env cleanup\n")
        print("WAITING_FOR_SAFE_GPU: CUDA not available")
        return 2
    from tools.bridgepack_g1.generate import load_qwen_fp16

    model, tokenizer = load_qwen_fp16(str(QWEN_PATH), _torch)
    safety_gate(out_dir, "deferred_after_load", phase="runtime")

    packed_root = out_dir / "deferred" / "packed_prompts_build"
    for variant in [MMR_VARIANT, BRIDGE_ZERO_VARIANT]:
        print(f"[deferred] QA {variant}...", flush=True)
        qa = run_qa_on_packs(out_dir, packed_root, [variant], sample_ids_full, model, tokenizer, _torch, "deferred")
        if qa["n_preds"] != 1150:
            raise RuntimeError(f"QA incomplete {variant}: {qa['n_preds']}/1150")
        scores = score_predictions(Path(qa["pred_root"]), [variant])
        fname = "G1_MMR_FULL_RESULTS.csv" if variant == MMR_VARIANT else "G1_BRIDGE_ZERO_FULL_RESULTS.csv"
        with (out_dir / fname).open("w", encoding="utf-8", newline="") as handle:
            w = csv.DictWriter(handle, fieldnames=["variant", "task", "n", "score", "mean_length"], lineterminator="\n")
            w.writeheader()
            for r in scores:
                w.writerow(r)

    mmr_scores = load_pred_scores(out_dir / "deferred" / "predictions", MMR_VARIANT, TASKS)
    bz_scores = load_pred_scores(out_dir / "deferred" / "predictions", BRIDGE_ZERO_VARIANT, TASKS)
    write_deferred_bootstrap(out_dir, mmr_scores, bz_scores)
    mechanism_decision(out_dir, mmr_scores, bz_scores, bz_pack["audit"])
    print("DONE deferred GPU", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        out = Path(sys.argv[sys.argv.index("--out_dir") + 1]) if "--out_dir" in sys.argv else Path(".")
        write_text(out / "G1_FATAL_DEFERRED.md", f"{exc}\n\n{traceback.format_exc()}")
        raise
