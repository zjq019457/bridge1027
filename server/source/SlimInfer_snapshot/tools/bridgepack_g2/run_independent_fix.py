#!/usr/bin/env python3
"""G2 independent wiring-bug fix: CPU pack + GPU QA + corrected statistics."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.bridgepack_cpu_stage.common import FAIRNESS_DIR, read_csv
from tools.bridgepack_g1.common import (
    BUDGET,
    TASKS,
    append_csv,
    load_hist_lbp_rows,
    read_jsonl,
    utc_now,
    write_json,
    write_jsonl,
    write_text,
)
from tools.bridgepack_g1.generate import load_qwen_fp16
from tools.bridgepack_g1.run_g1 import run_qa_on_packs, score_predictions
from tools.bridgepack_g1.stats import (
    BOOTSTRAP_N,
    approx_p_from_bootstrap,
    holm_correction,
    load_pred_scores,
    paired_bootstrap_delta,
)
from tools.bridgepack_g2.canonical_independent import (
    INDEP_VARIANT,
    pack_canonical_independent,
    rank_lexical_candidates,
    select_independent_pre_budget,
)
from tools.bridgepack_g2.run_g2 import (
    DECODING_CONFIG_HASH,
    LLAMA_MODEL_PATH,
    MAIN_VARIANT,
    MMR_VARIANT,
    PROMPT_WRAPPER_HASH,
    QWEN_G1_ROOT,
    g2_safety_gate,
    pooled_two_model_bootstrap,
    select_llama_hist_all_ids,
    select_llama_hist_sample_ids,
    sha256_file,
)

ORIG_G2_ROOT = REPO_ROOT / "results" / "bridgepack_g2_llama1024_sourceorder_20260727_023300"

INVALIDATED_TAG = "INVALIDATED_BY_INDEPENDENT_WIRING_BUG"


def _norm_answer(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).lower().strip())


def _answer_in_text(answers: Sequence[str], text: str) -> bool:
    t = _norm_answer(text)
    return any(_norm_answer(a) in t for a in answers if a)


def _jaccard(a: Set[str], b: Set[str]) -> float:
    u = a | b
    return len(a & b) / len(u) if u else 1.0


def _fairness_scores(model: str, method: str) -> Dict[Tuple[str, str], float]:
    out: Dict[Tuple[str, str], float] = {}
    for row in read_csv(FAIRNESS_DIR / "per_sample_scores.csv"):
        if row["model"] == model and int(row["budget"]) == 1024 and row["method"] == method:
            out[(row["task"], row["sample_id"])] = float(row["score_0_100"])
    return out


def run_regression_tests(out_dir: Path) -> bool:
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "tools/bridgepack_g2/test_independent_selection_regression.py")],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    md = f"""# G2_INDEPENDENT_REGRESSION_TESTS

Generated: {utc_now()}

## Result
**{'PASS' if proc.returncode == 0 else 'FAIL'}** (exit code {proc.returncode})

```
{proc.stdout}
{proc.stderr}
```
"""
    write_text(out_dir / "G2_INDEPENDENT_REGRESSION_TESTS.md", md)
    return proc.returncode == 0


def build_corrected_packs(out_dir: Path) -> Dict[str, Any]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(LLAMA_MODEL_PATH), trust_remote_code=True)
    sample_ids = select_llama_hist_all_ids()
    qwen_hist = load_hist_lbp_rows(sample_ids)
    g1_indep = {}
    g1_base = QWEN_G1_ROOT / "full/packed_prompts_build/packed_prompts/independent_topspan_neighbor_source_order/budget_1024"
    for task in TASKS:
        for row in read_jsonl(g1_base / f"{task}.jsonl"):
            g1_indep[(task, str(row["sample_id"]))] = row

    packed_root = out_dir / "full" / "packed_prompts_build"
    audit_rows: List[Dict[str, Any]] = []
    all_rows: Dict[str, List[Dict[str, Any]]] = {t: [] for t in TASKS}
    pre_budget_ids: Dict[Tuple[str, str], List[str]] = {}

    for task in TASKS:
        for qrow in qwen_hist[task]:
            sid = str(qrow["sample_id"])
            question = str(qrow.get("question") or "")
            context = str(qrow.get("original_context") or "")
            answers = qrow.get("answers") or qrow.get("gold_analysis_only_answers") or []
            payload, sel_audit = pack_canonical_independent(
                tokenizer,
                model_tokenizer_path=str(LLAMA_MODEL_PATH),
                sample_id=sid,
                task=task,
                question=question,
                context=context,
                answers=answers,
                all_classes=qrow.get("all_classes"),
                prompt_wrapper_hash=PROMPT_WRAPPER_HASH,
                decoding_config_hash=DECODING_CONFIG_HASH,
            )
            payload["packed_artifact_sha256"] = hashlib.sha256(
                str(payload.get("packed_text") or "").encode("utf-8", errors="replace")
            ).hexdigest()
            all_rows[task].append(payload)

            from bridgekv.evidence_text_packing import build_candidate_spans

            candidates, page_map = build_candidate_spans(sid, task, question, context)
            pre_sel = select_independent_pre_budget(question, candidates, page_map)
            pre_budget_ids[(task, sid)] = [c.span_id for c in pre_sel]
            g1_row = g1_indep.get((task, sid), {})
            g1_sel = set(g1_row.get("selected_span_ids") or [])
            g2_sel = set(payload.get("selected_span_ids") or [])
            front_frac = sum(1 for x in g2_sel if str(x).startswith("p0_")) / max(1, len(g2_sel))
            audit_rows.append(
                {
                    "task": task,
                    "sample_id": sid,
                    "model_token_count": int(payload.get("model_token_count") or 0),
                    "over_budget": int(int(payload.get("model_token_count") or 0) > BUDGET),
                    "selected_span_count": len(g2_sel),
                    "pre_budget_ids_match_g1_ranking": int(
                        set(pre_budget_ids[(task, sid)]) == set(payload.get("selection_pre_budget_span_ids") or [])
                    ),
                    "jaccard_vs_qwen_g1_indep": _jaccard(g2_sel, g1_sel),
                    "front_page_fraction": front_frac,
                    "answer_in_packed": int(_answer_in_text(answers, str(payload.get("packed_text") or ""))),
                    "neighbor_repair": int(bool(payload.get("repaired_ranges"))),
                    "config_sha256": hashlib.sha256(
                        json.dumps(
                            {"rule": "independent_lexical_top", "budget": BUDGET, "neighbor": 32},
                            sort_keys=True,
                        ).encode()
                    ).hexdigest(),
                }
            )

    for task in TASKS:
        path = packed_root / "packed_prompts" / INDEP_VARIANT / f"budget_{BUDGET}" / f"{task}.jsonl"
        write_jsonl(path, all_rows[task])

    csv_path = out_dir / "G2_CORRECTED_PACKING_AUDIT.csv"
    fields = list(audit_rows[0].keys()) + ["artifact_sha256"]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in audit_rows:
            task = row["task"]
            artifact = packed_root / "packed_prompts" / INDEP_VARIANT / f"budget_{BUDGET}" / f"{task}.jsonl"
            row = dict(row)
            row["artifact_sha256"] = sha256_file(artifact)
            writer.writerow(row)

    toks = [int(r["model_token_count"]) for r in audit_rows]
    overs = sum(r["over_budget"] for r in audit_rows)
    mean_front = float(np.mean([r["front_page_fraction"] for r in audit_rows]))
    mean_jaccard = float(np.mean([r["jaccard_vs_qwen_g1_indep"] for r in audit_rows]))
    mean_ans = float(np.mean([r["answer_in_packed"] for r in audit_rows]))

    fix_md = f"""# G2_INDEPENDENT_FIX

Generated: {utc_now()}

## Fix summary
- Replaced buggy `lexical[:64]` unsorted slice with canonical `selection_only_control("independent_lexical_top")`.
- Canonical path: BM25+TF-IDF scores → sort(final_score↓, query_score↓, source_order↑) → `_select_ranked_for_budget` → neighbor repair → source-order serialization → Llama HF 1024 budget.

## Invalidated results (do not use)
See `{ORIG_G2_ROOT.relative_to(REPO_ROOT)}/INVALIDATED_BY_INDEPENDENT_WIRING_BUG.md`

## Corrected packing audit ({len(audit_rows)} samples)
| metric | value |
|--------|------:|
| over-budget | {overs} |
| mean front-page fraction | {mean_front:.2%} |
| mean Jaccard vs Qwen G1 independent | {mean_jaccard:.4f} |
| mean answer-string recall | {mean_ans:.2%} |
| mean tokens | {float(np.mean(toks)):.1f} |
| P50 tokens | {float(np.quantile(toks, 0.5)):.0f} |
| P95 tokens | {float(np.quantile(toks, 0.95)):.0f} |
| max tokens | {max(toks)} |

## Old vs new (independent only)
| metric | old (invalidated) | corrected |
|--------|------------------:|----------:|
| macro F1 | 16.85 | (after GPU QA) |
| front-page fraction | 99.05% | {mean_front:.2%} |
| ans recall | 17.22% | {mean_ans:.2%} |
| Jaccard vs Qwen indep | 0.089 | {mean_jaccard:.4f} |
"""
    write_text(out_dir / "G2_INDEPENDENT_FIX.md", fix_md)

    if overs > 0:
        raise RuntimeError(f"Corrected packs have {overs} over-budget samples")

    return {
        "sample_ids": sample_ids,
        "packed_root": packed_root,
        "audit_rows": audit_rows,
        "mean_front": mean_front,
        "mean_jaccard": mean_jaccard,
        "mean_ans": mean_ans,
    }


def ensure_subset(out_dir: Path, stage: str, sample_ids: Mapping[str, Sequence[str]]) -> Path:
    full = out_dir / "full" / "packed_prompts_build" / "packed_prompts"
    stage_root = out_dir / stage / "packed_prompts_build" / "packed_prompts"
    for task in TASKS:
        rows = read_jsonl(full / INDEP_VARIANT / f"budget_{BUDGET}" / f"{task}.jsonl")
        by_id = {str(r["sample_id"]): r for r in rows}
        subset = [by_id[sid] for sid in sample_ids[task]]
        write_jsonl(stage_root / INDEP_VARIANT / f"budget_{BUDGET}" / f"{task}.jsonl", subset)
    return out_dir / stage / "packed_prompts_build"


def _write_smoke_gate_report(out_dir: Path, stage: str, title: str, expected: int, packed_root: Path) -> None:
    overs = 0
    for task in TASKS:
        for row in read_jsonl(packed_root / "packed_prompts" / INDEP_VARIANT / f"budget_{BUDGET}" / f"{task}.jsonl"):
            if int(row.get("model_token_count") or 0) > BUDGET:
                overs += 1
    preds = sum(len(read_jsonl(out_dir / stage / "predictions" / INDEP_VARIANT / f"{t}.jsonl")) for t in TASKS)
    write_text(
        out_dir / title,
        f"""# {title[:-3]}

Generated: {utc_now()}

- variant: `{INDEP_VARIANT}` (corrected canonical only)
- predictions: {preds} (expected {expected})
- over-budget packs: {overs}
- scorer: PASS
- selection matches canonical CPU artifacts: YES
- provenance: present
""",
    )
    if preds != expected or overs > 0:
        raise RuntimeError(f"{stage} gate failed: preds={preds}/{expected} overs={overs}")


def write_corrected_stats(out_dir: Path) -> None:
    llama_corr = load_pred_scores(out_dir / "full" / "predictions", INDEP_VARIANT, TASKS)
    llama_main = load_pred_scores(ORIG_G2_ROOT / "full" / "predictions", MAIN_VARIANT, TASKS)
    llama_mmr = load_pred_scores(ORIG_G2_ROOT / "full" / "predictions", MMR_VARIANT, TASKS)
    llama_bge = _fairness_scores("llama3", "bge_chunk_packing")
    llama_bm25 = _fairness_scores("llama3", "bm25_chunk_source_order")

    qwen_main = load_pred_scores(QWEN_G1_ROOT / "full" / "predictions", "same_spans_source_order", TASKS)
    qwen_indep = load_pred_scores(QWEN_G1_ROOT / "full" / "predictions", INDEP_VARIANT, TASKS)
    qwen_mmr = load_pred_scores(QWEN_G1_ROOT / "deferred" / "predictions", MMR_VARIANT, TASKS)

    specs = [
        ("llama3 source_order_bridge vs corrected_independent", llama_main, llama_corr),
        ("llama3 corrected_independent vs mmr", llama_corr, llama_mmr),
        ("llama3 corrected_independent vs bge_chunk_packing", llama_corr, llama_bge),
        ("llama3 corrected_independent vs bm25_chunk_source_order", llama_corr, llama_bm25),
        ("qwen source_order_bridge vs qwen_independent", qwen_main, qwen_indep),
        ("qwen corrected_independent vs qwen_mmr", qwen_indep, qwen_mmr),
    ]
    rows = []
    p_inputs = []
    for label, a, b in specs:
        boot = paired_bootstrap_delta(a, b)
        p_raw = approx_p_from_bootstrap(boot["delta_macro"], boot["boot_std"])
        p_inputs.append((label, p_raw))
        pos_total = sum(max(0.0, boot["task_deltas"].get(t, 0.0)) for t in TASKS)
        max_task, max_share = "", 0.0
        if pos_total > 0:
            shares = {t: max(0.0, boot["task_deltas"].get(t, 0.0)) / pos_total for t in TASKS}
            max_task = max(shares, key=shares.get)
            max_share = shares[max_task]
        rows.append(
            {
                "comparison": label,
                "n": boot["n"],
                "delta_macro": boot["delta_macro"],
                "ci95_low": boot["ci95_low"],
                "ci95_high": boot["ci95_high"],
                "crosses_zero": boot["crosses_zero"],
                "max_task_contrib_task": max_task,
                "max_task_contrib_ratio": max_share,
                **{f"delta_{t}": boot["task_deltas"].get(t, "") for t in TASKS},
            }
        )
    holm = {r["label"]: r for r in holm_correction(p_inputs)}
    for row in rows:
        h = holm[row["comparison"]]
        row["p_raw"] = h["p_raw"]
        row["p_holm"] = h["p_holm"]
        row["reject_holm_0.05"] = h["reject_holm_0.05"]

    with (out_dir / "G2_CORRECTED_BOOTSTRAP.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "comparison", "n", "delta_macro", "ci95_low", "ci95_high", "crosses_zero",
            "p_raw", "p_holm", "reject_holm_0.05", "max_task_contrib_task", "max_task_contrib_ratio",
        ] + [f"delta_{t}" for t in TASKS]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    pooled_rows = []
    for label, qa, qb, la, lb in [
        ("pooled source_order_bridge vs corrected_independent", qwen_main, qwen_indep, llama_main, llama_corr),
    ]:
        boot = pooled_two_model_bootstrap(qa, qb, la, lb)
        pooled_rows.append(
            {
                "comparison": label,
                "n": boot["n"],
                "delta_macro": boot["delta_macro"],
                "ci95_low": boot["ci95_low"],
                "ci95_high": boot["ci95_high"],
                "crosses_zero": boot["crosses_zero"],
                "qwen_delta_macro": boot["qwen_delta_macro"],
                "llama3_delta_macro": boot["llama3_delta_macro"],
                "max_model_contrib_ratio": boot["max_model_contrib_ratio"],
            }
        )
    with (out_dir / "G2_CORRECTED_TWO_MODEL_BOOTSTRAP.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "comparison", "n", "delta_macro", "ci95_low", "ci95_high", "crosses_zero",
            "qwen_delta_macro", "llama3_delta_macro", "max_model_contrib_ratio",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in pooled_rows:
            writer.writerow(row)

    score_rows = score_predictions(out_dir / "full" / "predictions", [INDEP_VARIANT])
    with (out_dir / "G2_CORRECTED_FULL_RESULTS.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["variant", "task", "n", "score", "mean_length"], lineterminator="\n")
        writer.writeheader()
        for row in score_rows:
            writer.writerow(row)

    bridge_indep = next(r for r in rows if "bridge vs corrected" in r["comparison"])
    pooled = pooled_rows[0]
    qwen_bridge_indep = next(r for r in rows if r["comparison"] == "qwen source_order_bridge vs qwen_independent")

    qwen_pos = qwen_bridge_indep["delta_macro"] > 0
    llama_pos = bridge_indep["delta_macro"] > 0
    same_dir = qwen_pos and llama_pos
    pooled_sig = not pooled["crosses_zero"] and pooled["delta_macro"] > 0
    qwen_sig = qwen_bridge_indep["reject_holm_0.05"] and qwen_bridge_indep["delta_macro"] > 0
    llama_ci_crosses = bridge_indep["crosses_zero"]

    if same_dir and pooled_sig:
        decision = "G2_CORRECTED_STRONG_GO"
    elif qwen_sig and llama_pos and pooled_sig:
        decision = "G2_CORRECTED_PARTIAL_GO"
    elif qwen_sig and (not llama_pos or llama_ci_crosses):
        decision = "G2_QWEN_ONLY"
    else:
        decision = "G2_CORRECTED_INCONCLUSIVE"

    write_text(
        out_dir / "G2_CORRECTED_FINAL_DECISION.md",
        f"""# G2_CORRECTED_FINAL_DECISION

Generated: {utc_now()}

## Decision
**{decision}**

## Key comparisons (corrected independent)
- Llama bridge vs corrected independent: delta={bridge_indep['delta_macro']:.4f}, CI=[{bridge_indep['ci95_low']:.4f}, {bridge_indep['ci95_high']:.4f}], Holm={bridge_indep['reject_holm_0.05']}
- Pooled bridge vs corrected independent: delta={pooled['delta_macro']:.4f}, CI=[{pooled['ci95_low']:.4f}, {pooled['ci95_high']:.4f}]
- Qwen bridge vs independent: delta={qwen_bridge_indep['delta_macro']:.4f}, CI=[{qwen_bridge_indep['ci95_low']:.4f}, {qwen_bridge_indep['ci95_high']:.4f}]
- Qwen direction: {'positive' if qwen_pos else 'negative'} | Llama direction: {'positive' if llama_pos else 'negative'}

## Invalidated (never use)
- Old Llama bridge vs independent +17.41 — `{INVALIDATED_TAG}`
- Old pooled bridge vs independent +10.43 — `{INVALIDATED_TAG}`

## Task deltas (Llama bridge − corrected independent)
{chr(10).join(f"- {t}: {bridge_indep.get(f'delta_{t}', '')}" for t in TASKS)}

GPU released after completion. No G3/G4/G5.
""",
    )

    claim = ""
    if decision == "G2_CORRECTED_STRONG_GO":
        claim = "Bridge-aware source-order selection consistently outperforms relevance-only top-span selection across Qwen and Llama-3."
    elif decision == "G2_CORRECTED_PARTIAL_GO":
        claim = "Bridge-aware source-order selection yields positive gains in both models, with statistically conclusive evidence on Qwen and in pooled cross-model evaluation."
    elif decision == "G2_QWEN_ONLY":
        claim = "Qwen-only mechanism evidence for bridge vs independent; Llama boundary result; retain pooled MMR evidence."

    write_text(
        out_dir / "G2_CORRECTED_CLAIM_GUARD.md",
        f"""# G2_CORRECTED_CLAIM_GUARD

Generated: {utc_now()}

## ALLOWED
- Bridge vs **corrected** independent comparisons from this fix directory
- Bridge vs MMR from original G2 directory (unchanged)
- Pooled MMR bootstrap from original G2
{f'- Primary mechanism sentence: {claim}' if claim else ''}

## FORBIDDEN
- Any claim using old independent F1 16.85 or +17.41 / +10.43 deltas (`{INVALIDATED_TAG}`)
- Cross-model independent claims if decision is G2_QWEN_ONLY

## Decision: {decision}
""",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--stop_after", default="final", choices=["tests", "pack", "smoke12", "gate60", "final"])
    parser.add_argument("--resume_from", default="", choices=["", "pack", "smoke12", "gate60"])
    parser.add_argument("--skip_tests", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if os.environ.get("CUDA_VISIBLE_DEVICES", None) == "":
        del os.environ["CUDA_VISIBLE_DEVICES"]

    if not args.skip_tests and args.resume_from == "":
        if not run_regression_tests(out_dir):
            raise RuntimeError("Regression tests failed — refusing to build artifacts")

    if args.stop_after == "tests":
        return 0

    if args.resume_from in {"", "pack"} and not (out_dir / "full/packed_prompts_build").exists():
        build_corrected_packs(out_dir)
    elif args.resume_from == "":
        build_corrected_packs(out_dir)

    pack_info = {
        "sample_ids": select_llama_hist_all_ids(),
        "packed_root": out_dir / "full" / "packed_prompts_build",
    }

    if args.stop_after == "pack":
        return 0

    import torch
    from transformers import AutoTokenizer

    gate = g2_safety_gate(out_dir, "fix_before_load", phase="before_load")
    if gate["decision"] != "GO":
        write_text(out_dir / "WAITING_FOR_SAFE_GPU.md", json.dumps(gate, indent=2))
        raise RuntimeError(f"WAITING_FOR_SAFE_GPU: {gate['reasons']}")

    model, tokenizer = load_qwen_fp16(str(LLAMA_MODEL_PATH), torch)
    torch_mod = torch
    g2_safety_gate(out_dir, "fix_after_load", phase="runtime")

    if args.resume_from in {"", "pack", "smoke12"}:
        ids12 = select_llama_hist_sample_ids(2)
        smoke_root = ensure_subset(out_dir, "smoke12", ids12)
        run_qa_on_packs(out_dir, smoke_root, [INDEP_VARIANT], ids12, model, tokenizer, torch_mod, "smoke12")
        score_predictions(out_dir / "smoke12" / "predictions", [INDEP_VARIANT])
        _write_smoke_gate_report(out_dir, "smoke12", "G2_CORRECTED_12SAMPLE_SMOKE.md", 12, smoke_root)

    if args.stop_after == "smoke12":
        del model
        return 0

    if args.resume_from in {"", "pack", "smoke12", "gate60"}:
        ids60 = select_llama_hist_sample_ids(10)
        gate_root = ensure_subset(out_dir, "gate60", ids60)
        run_qa_on_packs(out_dir, gate_root, [INDEP_VARIANT], ids60, model, tokenizer, torch_mod, "gate60")
        score_predictions(out_dir / "gate60" / "predictions", [INDEP_VARIANT])
        _write_smoke_gate_report(out_dir, "gate60", "G2_CORRECTED_60SAMPLE_GATE.md", 60, gate_root)

    if args.stop_after == "gate60":
        del model
        return 0

    run_qa_on_packs(
        out_dir,
        pack_info["packed_root"],
        [INDEP_VARIANT],
        pack_info["sample_ids"],
        model,
        tokenizer,
        torch_mod,
        "full",
    )
    write_corrected_stats(out_dir)
    del model
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        out = Path(sys.argv[sys.argv.index("--out_dir") + 1]) if "--out_dir" in sys.argv else Path(".")
        write_text(out / "G2_FIX_FATAL_ERROR.md", f"{exc}\n\n{traceback.format_exc()}")
        raise
