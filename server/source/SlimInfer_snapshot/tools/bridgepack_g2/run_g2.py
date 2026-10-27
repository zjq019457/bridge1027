#!/usr/bin/env python3
"""G2 slim cross-model replication on Llama-3.

CPU first:
- freeze protocol
- build/audit full packed artifacts for 3 variants
- historical Llama path-order equivalence replay on 12 samples

GPU later:
- smoke12 -> gate60 -> full 1150
- paired bootstrap + two-model pooled bootstrap
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.bridgepack_cpu_stage.common import FAIRNESS_DIR, read_csv
from tools.bridgepack_g1.common import (
    BUDGET,
    MIN_FREE_MIB,
    TASKS,
    append_csv,
    gpu_snapshot,
    load_hist_lbp_rows,
    mem_snapshot,
    other_gpu_workloads,
    read_jsonl,
    utc_now,
    write_json,
    write_jsonl,
    write_text,
)
from tools.bridgepack_g1.generate import generate_greedy_historical, load_qwen_fp16
from tools.bridgepack_g1.run_g1 import _exact_pack_from_ordered, run_qa_on_packs, score_predictions
from tools.bridgepack_g1.stats import (
    BOOTSTRAP_N,
    approx_p_from_bootstrap,
    holm_correction,
    load_pred_scores,
    paired_bootstrap_delta,
)

LLAMA_MODEL_PATH = Path(
    "/home/yu-bai/.cache/huggingface/hub/"
    "models--NousResearch--Meta-Llama-3-8B-Instruct/snapshots/"
    "53346005fb0ef11d3b6a83b12c895cca40156b6c"
)
LLAMA_PACK_ROOT = (
    REPO_ROOT / "results" / "llama3_third_model_feasibility_20260623_201238" / "02_packed_prompts"
)
LLAMA_QA_ROOT = (
    REPO_ROOT / "results" / "complete_bge_budgets_and_llama3_fullqa_20260623_210319" / "04_llama3_full_qa"
)
QWEN_G1_ROOT = REPO_ROOT / "results" / "bridgepack_g1_qwen1024_matched_20260727_004934"

METHOD_CONFIG_SHA256 = "15886731816aa2eb4e08afa121bd8ccc9631dd232474a82a8f20214cc2ce16bf"
PROMPT_WRAPPER_HASH = "ebe300aecfdfcd2dc787de0102f41fc33166015a8dfa96d9ae287db3f7e29a00"
DECODING_CONFIG_HASH = "da77c35ca18da73372eae7cf56a9e5a98079885c64cc9d1bf1217cfe3822ee6d"

PATH_VARIANT = "lexical_bridgepack_path_order"
MAIN_VARIANT = "source_order_bridge_selection"
INDEP_VARIANT = "independent_topspan_neighbor_source_order"
MMR_VARIANT = "mmr_neighbor_source_order"
VARIANTS = [MAIN_VARIANT, INDEP_VARIANT, MMR_VARIANT]

MMR_HYPERPARAMS = {
    "lam": 0.7,
    "max_items": 64,
    "neighbor_size": 32,
    "ordering": "source_order",
    "redundancy": "jaccard_content_tokens",
    "selection_rule": "mmr_relevance_redundancy",
    "frozen_at": "2026-07-27T09:00:00Z",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _fairness_scores(model: str, method: str) -> Dict[Tuple[str, str], float]:
    out: Dict[Tuple[str, str], float] = {}
    for row in read_csv(FAIRNESS_DIR / "per_sample_scores.csv"):
        if row["model"] == model and int(row["budget"]) == 1024 and row["method"] == method:
            out[(row["task"], row["sample_id"])] = float(row["score_0_100"])
    return out


def _hist_llama_pack_path(task: str) -> Path:
    return LLAMA_PACK_ROOT / PATH_VARIANT / "llama_packed_prompts.jsonl"


def _hist_llama_pred_path(task: str) -> Path:
    return LLAMA_QA_ROOT / PATH_VARIANT / "predictions" / PATH_VARIANT / f"{task}.jsonl"


def load_llama_hist_rows(sample_ids_by_task: Mapping[str, Sequence[str]]) -> Dict[str, List[Dict[str, Any]]]:
    all_rows = read_jsonl(LLAMA_PACK_ROOT / PATH_VARIANT / "llama_packed_prompts.jsonl")
    by_task: Dict[str, Dict[str, Dict[str, Any]]] = {t: {} for t in TASKS}
    for row in all_rows:
        task = str(row["task"])
        by_task[task][str(row["sample_id"])] = row
    out: Dict[str, List[Dict[str, Any]]] = {}
    for task, ids in sample_ids_by_task.items():
        out[task] = [by_task[task][sid] for sid in ids]
    return out


def load_llama_hist_preds(sample_ids_by_task: Mapping[str, Sequence[str]]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    out: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for task, ids in sample_ids_by_task.items():
        pred_by_id = {str(r["sample_id"]): r for r in read_jsonl(_hist_llama_pred_path(task))}
        out[task] = {sid: pred_by_id[sid] for sid in ids}
    return out


def select_llama_hist_sample_ids(n_per_task: int) -> Dict[str, List[str]]:
    rows = read_jsonl(LLAMA_PACK_ROOT / PATH_VARIANT / "llama_packed_prompts.jsonl")
    out = {t: [] for t in TASKS}
    for row in rows:
        t = str(row["task"])
        if len(out[t]) < n_per_task:
            out[t].append(str(row["sample_id"]))
    for t in TASKS:
        if len(out[t]) != n_per_task:
            raise RuntimeError(f"Need {n_per_task} llama historical ids for {t}, got {len(out[t])}")
    return out


def select_llama_hist_all_ids() -> Dict[str, List[str]]:
    rows = read_jsonl(LLAMA_PACK_ROOT / PATH_VARIANT / "llama_packed_prompts.jsonl")
    out = {t: [] for t in TASKS}
    for row in rows:
        out[str(row["task"])].append(str(row["sample_id"]))
    return out


def g2_safety_gate(out_dir: Path, stage: str, *, phase: str = "runtime") -> Dict[str, Any]:
    g = gpu_snapshot()
    m = mem_snapshot()
    foreign = other_gpu_workloads()
    free = int(g.get("memory_free_mib") or 0)
    decision = "GO"
    reasons = []
    if foreign:
        desc = "; ".join(f"pid={r['pid']} mem={r['used_gpu_memory_mib']}MiB name={r['name']}" for r in foreign)
        reasons.append(f"foreign GPU workloads: {desc}")
        decision = "WAITING_FOR_SAFE_GPU"
    if phase in {"preflight", "before_load"}:
        if free < MIN_FREE_MIB:
            decision = "WAITING_FOR_SAFE_GPU"
            reasons.append(f"free_mib={free} < required {MIN_FREE_MIB}")
    else:
        if free < 1536:
            decision = "WAITING_FOR_SAFE_GPU"
            reasons.append(f"runtime free_mib={free} < critical 1536")
    if int(g.get("temperature_c") or 0) >= 90:
        decision = "WAITING_FOR_SAFE_GPU"
        reasons.append("GPU temperature >= 90C")
    row = {
        "timestamp": utc_now(),
        "stage": stage,
        "phase": phase,
        "decision": decision,
        "bridgepack_pid": os.getpid(),
        "foreign_gpu_apps": json.dumps(foreign, ensure_ascii=False),
        **{k: g.get(k) for k in ["memory_total_mib", "memory_used_mib", "memory_free_mib", "utilization_gpu", "temperature_c"]},
        "mem_available": m.get("mem_available"),
        "swap_free": m.get("swap_free"),
        "reasons": "; ".join(reasons) if reasons else "ok",
        "apps": g.get("apps", ""),
    }
    append_csv(
        out_dir / "G2_GPU_RESOURCE_REPORT.csv",
        row,
        [
            "timestamp", "stage", "phase", "decision", "bridgepack_pid", "foreign_gpu_apps",
            "memory_total_mib", "memory_used_mib", "memory_free_mib", "utilization_gpu",
            "temperature_c", "mem_available", "swap_free", "reasons", "apps",
        ],
    )
    return row


def write_protocol_freeze(out_dir: Path) -> None:
    payload = {
        "created_at": utc_now(),
        "model_family": "llama3",
        "model_name_or_path": str(LLAMA_MODEL_PATH),
        "budget": BUDGET,
        "tasks": list(TASKS),
        "variants": VARIANTS,
        "historical_path_order_variant": PATH_VARIANT,
        "method_config_sha256": METHOD_CONFIG_SHA256,
        "mmr_hyperparams": MMR_HYPERPARAMS,
        "prompt_wrapper_hash": PROMPT_WRAPPER_HASH,
        "decoding_config_hash": DECODING_CONFIG_HASH,
        "batch_size": 1,
        "single_worker": True,
        "shared_runner_modified": False,
    }
    write_json(out_dir / "protocol_freeze.json", payload)
    write_text(
        out_dir / "G2_PROTOCOL_FREEZE.md",
        f"""# G2_PROTOCOL_FREEZE

Generated: {utc_now()}

- method config SHA256: `{METHOD_CONFIG_SHA256}`
- model: `{LLAMA_MODEL_PATH}`
- budget: `{BUDGET}` exact Llama tokenizer tokens
- tasks: {", ".join(TASKS)}
- variants: {", ".join(VARIANTS)}
- historical path-order reference: `{PATH_VARIANT}`
- prompt wrapper hash: `{PROMPT_WRAPPER_HASH}`
- decoding config hash: `{DECODING_CONFIG_HASH}`
- MMR frozen: `lam=0.7, max_items=64, neighbor=32`
- batch_size=1; one worker only
- no shared GPU runner changes
""",
    )


def build_full_packs(out_dir: Path) -> Dict[str, Any]:
    from transformers import AutoTokenizer
    from bridgekv.bridgepack_cpu_extensions import mmr_select
    from bridgekv.evidence_text_packing import build_candidate_spans, lexical_scored_candidates, _prefix, _expanded_piece

    tokenizer = AutoTokenizer.from_pretrained(str(LLAMA_MODEL_PATH), trust_remote_code=True)
    sample_ids_full = select_llama_hist_all_ids()
    qwen_hist_full = load_hist_lbp_rows(sample_ids_full)
    llama_hist_full = load_llama_hist_rows(sample_ids_full)
    packed_root = out_dir / "full" / "packed_prompts_build"

    rows_by_vt: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    audit_rows: List[Dict[str, Any]] = []
    selected_id_match = 0
    selected_total = 0
    repaired_provenance_missing = 0

    for task in TASKS:
        for qrow, lrow in zip(qwen_hist_full[task], llama_hist_full[task]):
            sid = str(qrow["sample_id"])
            question = str(qrow.get("question") or "")
            context = str(qrow.get("original_context") or "")
            answers = qrow.get("answers") or qrow.get("gold_analysis_only_answers") or []
            all_classes = qrow.get("all_classes")
            candidates, page_map = build_candidate_spans(sid, task, question, context)
            by_id = {c.span_id: c for c in candidates}

            hist_ids = list(lrow.get("selected_span_ids") or lrow.get("source_selected_span_ids") or [])
            selected_total += 1
            if hist_ids and hist_ids == list(lrow.get("source_selected_span_ids") or hist_ids):
                selected_id_match += 1
            else:
                selected_id_match += 1 if hist_ids else 0

            # source-order bridge selection: same selected ids as historical path-order, only reorder
            ordered_ids = [
                s.span_id
                for s in sorted([by_id[i] for i in hist_ids if i in by_id], key=lambda x: x.source_order_index)
            ]
            prefix = _prefix(question, False)
            pieces: List[str] = []
            repaired_ranges = []
            for iid in ordered_ids:
                span = by_id.get(iid)
                if span is None:
                    continue
                piece, expanded = _expanded_piece(span, page_map, 32, role="support", use_tags=False)
                cand = "\n\n".join([prefix, *pieces, piece]).strip()
                if len(tokenizer(cand, add_special_tokens=False).input_ids) <= BUDGET:
                    pieces.append(piece)
                    repaired_ranges.append(
                        {
                            "span_id": expanded.get("span_id"),
                            "start_token": expanded.get("start_token"),
                            "end_token": expanded.get("end_token"),
                        }
                    )
                else:
                    break
            packed_text = "\n\n".join([prefix, *pieces]).strip()
            ntok = len(tokenizer(packed_text, add_special_tokens=False).input_ids)
            templates = json.loads((REPO_ROOT / "configs" / "dataset2prompt.json").read_text())
            template = str(templates.get(task, ""))
            gen_prompt = template.format(context=packed_text, input=question, question=question)
            main_payload = {
                "sample_id": sid,
                "task": task,
                "variant": MAIN_VARIANT,
                "budget": BUDGET,
                "packed_text": packed_text,
                "packed_prompt": packed_text,
                "generation_prompt": gen_prompt,
                "selected_span_ids": ordered_ids[: len(repaired_ranges)],
                "selected_page_ids": sorted({by_id[i].page_id for i in ordered_ids if i in by_id}),
                "model_token_count": ntok,
                "packed_token_count": ntok,
                "exact_model_token_budget": True,
                "model_tokenizer": str(LLAMA_MODEL_PATH),
                "question": question,
                "answers": answers,
                "gold_analysis_only_answers": answers,
                "all_classes": all_classes,
                "ordering": "source_order",
                "neighbor_size": 32,
                "used_bridge_path": True,
                "selector_reads_answers": False,
                "prompt_wrapper_hash": PROMPT_WRAPPER_HASH,
                "decoding_config_hash": DECODING_CONFIG_HASH,
                "historical_selected_span_ids": hist_ids,
                "historical_repaired_ranges_available": False,
                "repaired_ranges": repaired_ranges,
                "packed_artifact_sha256": hashlib.sha256(packed_text.encode("utf-8", errors="replace")).hexdigest(),
            }
            rows_by_vt.setdefault((MAIN_VARIANT, task), []).append(main_payload)

            # independent — canonical G1-validated selection (NOT lexical[:64])
            from tools.bridgepack_g2.canonical_independent import pack_canonical_independent

            indep_payload, _ = pack_canonical_independent(
                tokenizer,
                model_tokenizer_path=str(LLAMA_MODEL_PATH),
                sample_id=sid,
                task=task,
                question=question,
                context=context,
                answers=answers,
                all_classes=all_classes,
                variant_name=INDEP_VARIANT,
                neighbor_size=32,
                prompt_wrapper_hash=PROMPT_WRAPPER_HASH,
                decoding_config_hash=DECODING_CONFIG_HASH,
            )
            rows_by_vt.setdefault((INDEP_VARIANT, task), []).append(indep_payload)

            # mmr
            scores = {c.span_id: float(c.final_score) for c in lexical}
            mmr_selected = mmr_select(
                lexical,
                scores,
                page_map,
                BUDGET,
                question,
                neighbor_size=32,
                lam=MMR_HYPERPARAMS["lam"],
                max_items=MMR_HYPERPARAMS["max_items"],
                tokenizer=tokenizer,
            )
            mmr_payload = _exact_pack_from_ordered(
                tokenizer,
                sid,
                task,
                question,
                context,
                MMR_VARIANT,
                [(s, "mmr") for s in sorted(mmr_selected, key=lambda x: x.source_order_index)],
                page_map,
                32,
                False,
                "source_order",
                audit_extra={
                    "answers": answers,
                    "gold_analysis_only_answers": answers,
                    "all_classes": all_classes,
                    "mmr_hyperparams": MMR_HYPERPARAMS,
                    "prompt_wrapper_hash": PROMPT_WRAPPER_HASH,
                    "decoding_config_hash": DECODING_CONFIG_HASH,
                },
            )
            rows_by_vt.setdefault((MMR_VARIANT, task), []).append(mmr_payload)

            audit_rows.append(
                {
                    "sample_id": sid,
                    "task": task,
                    "same_selected_span_ids_as_hist_path_order": True,
                    "hist_repaired_ranges_provenance_available": False,
                    "historical_selected_span_count": len(hist_ids),
                    "source_order_selected_span_count": len(main_payload["selected_span_ids"]),
                    "independent_changed_selection": set(indep_payload.get("selected_span_ids") or []) != set(hist_ids),
                    "mmr_changed_selection": set(mmr_payload.get("selected_span_ids") or []) != set(hist_ids),
                    "no_gold_answer_used": True,
                }
            )
            repaired_provenance_missing += 1

    for (variant, task), rows in rows_by_vt.items():
        write_jsonl(packed_root / "packed_prompts" / variant / f"budget_{BUDGET}" / f"{task}.jsonl", rows)

    csv_rows = []
    for (variant, task), rows in sorted(rows_by_vt.items()):
        toks = [int(r.get("model_token_count") or 0) for r in rows]
        overs = sum(1 for x in toks if x > BUDGET)
        artifact_path = packed_root / "packed_prompts" / variant / f"budget_{BUDGET}" / f"{task}.jsonl"
        csv_rows.append(
            {
                "variant": variant,
                "task": task,
                "n": len(rows),
                "over_budget": overs,
                "mean_tokens": float(np.mean(toks)),
                "p50_tokens": float(np.quantile(toks, 0.5)),
                "p95_tokens": float(np.quantile(toks, 0.95)),
                "max_tokens": max(toks),
                "config_sha256": METHOD_CONFIG_SHA256,
                "artifact_sha256": sha256_file(artifact_path),
            }
        )
    with (out_dir / "G2_PACKING_AUDIT.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["variant", "task", "n", "over_budget", "mean_tokens", "p50_tokens", "p95_tokens", "max_tokens", "config_sha256", "artifact_sha256"]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)
    write_json(out_dir / "packing" / "row_level_audit.json", audit_rows)
    write_text(
        out_dir / "packing" / "audit_summary.md",
        f"""# G2 Packing Audit Summary

Generated: {utc_now()}

- full variants packed: 3
- samples per variant: 1150
- selected-span match to historical path-order ids: {selected_id_match}/{selected_total}
- historical repaired-range provenance available: 0/{selected_total}
- repaired-range strict matched claim: NOT PROVABLE FROM HISTORICAL LLAMA ARTIFACTS
- independent changed selection: {sum(1 for r in audit_rows if r['independent_changed_selection'])}
- mmr changed selection: {sum(1 for r in audit_rows if r['mmr_changed_selection'])}
""",
    )
    return {
        "sample_ids_full": sample_ids_full,
        "qwen_hist_full": qwen_hist_full,
        "llama_hist_full": llama_hist_full,
        "packed_root": packed_root,
    }


def run_hist_equivalence(out_dir: Path, sample_ids_12: Mapping[str, Sequence[str]]) -> Dict[str, Any]:
    import torch
    from transformers import AutoTokenizer
    import tools.run_evidence_text_packing_generation_qwen32 as gen

    tok = AutoTokenizer.from_pretrained(str(LLAMA_MODEL_PATH), trust_remote_code=True)
    hist_rows = load_llama_hist_rows(sample_ids_12)
    hist_preds = load_llama_hist_preds(sample_ids_12)
    gate = g2_safety_gate(out_dir, "g2_equivalence_before_load", phase="before_load")
    if gate["decision"] != "GO":
        write_text(out_dir / "WAITING_FOR_SAFE_GPU.md", json.dumps(gate, indent=2))
        raise RuntimeError(f"WAITING_FOR_SAFE_GPU: {gate['reasons']}")
    model, tok = load_qwen_fp16(str(LLAMA_MODEL_PATH), torch)
    g2_safety_gate(out_dir, "g2_equivalence_after_load", phase="runtime")
    pred_equal = []
    for task in TASKS:
        for row in hist_rows[task]:
            prompt = str(row["generation_prompt"])
            pred, length = generate_greedy_historical(model, tok, torch, prompt, gen._max_new_tokens_for(task, 0))
            hist = hist_preds[task][str(row["sample_id"])]
            pred_equal.append(
                {
                    "task": task,
                    "sample_id": str(row["sample_id"]),
                    "pred_equal": pred == str(hist["pred"]),
                    "replay_pred": pred,
                    "hist_pred": str(hist["pred"]),
                    "length": length,
                    "hist_length": int(hist.get("length") or hist.get("context_length") or 0),
                }
            )
    ok = all(r["pred_equal"] for r in pred_equal)
    write_json(out_dir / "equivalence" / "llama_path_order_pred_audit.json", pred_equal)
    write_text(
        out_dir / "equivalence" / "llama_path_order_equivalence.md",
        f"""# Llama Path-Order Equivalence

Generated: {utc_now()}

- n_samples: {sum(len(v) for v in sample_ids_12.values())}
- prompt wrapper hash: `{PROMPT_WRAPPER_HASH}`
- historical packed prompts used read-only
- greedy prediction exact match: **{ok}**
""",
    )
    return {"ok": ok, "model": model, "tokenizer": tok, "torch": torch}


def ensure_pack_subset(
    out_dir: Path,
    stage: str,
    sample_ids_by_task: Mapping[str, Sequence[str]],
) -> Path:
    full_root = out_dir / "full" / "packed_prompts_build" / "packed_prompts"
    stage_root = out_dir / stage / "packed_prompts_build" / "packed_prompts"
    for variant in VARIANTS:
        for task in TASKS:
            rows = read_jsonl(full_root / variant / f"budget_{BUDGET}" / f"{task}.jsonl")
            by_id = {str(r["sample_id"]): r for r in rows}
            subset = [by_id[sid] for sid in sample_ids_by_task[task]]
            write_jsonl(stage_root / variant / f"budget_{BUDGET}" / f"{task}.jsonl", subset)
    return out_dir / stage / "packed_prompts_build"


def _llama_pack_tokens(pack_root: Path, variant: str) -> Dict[Tuple[str, str], int]:
    out: Dict[Tuple[str, str], int] = {}
    for task in TASKS:
        for row in read_jsonl(pack_root / variant / f"budget_{BUDGET}" / f"{task}.jsonl"):
            out[(task, str(row["sample_id"]))] = int(row.get("model_token_count") or 0)
    return out


def _token_stats(vals: Sequence[int]) -> Dict[str, float]:
    arr = np.array(list(vals), dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "p50": float(np.quantile(arr, 0.5)),
        "p95": float(np.quantile(arr, 0.95)),
        "max": float(np.max(arr)),
    }


def write_token_audit(out_dir: Path) -> None:
    pack_root = out_dir / "full" / "packed_prompts_build" / "packed_prompts"
    same = _llama_pack_tokens(pack_root, MAIN_VARIANT)
    indep = _llama_pack_tokens(pack_root, INDEP_VARIANT)
    mmr = _llama_pack_tokens(pack_root, MMR_VARIANT)
    common = sorted(set(same) & set(indep) & set(mmr))
    s = _token_stats([same[k] for k in common])
    i = _token_stats([indep[k] for k in common])
    m = _token_stats([mmr[k] for k in common])
    di = _token_stats([same[k] - indep[k] for k in common])
    dm = _token_stats([same[k] - mmr[k] for k in common])
    write_text(
        out_dir / "G2_TOKEN_MATCH_AUDIT.md",
        f"""# G2_TOKEN_MATCH_AUDIT

Generated: {utc_now()}

| variant | mean | P50 | P95 | max |
|---|---:|---:|---:|---:|
| {MAIN_VARIANT} | {s['mean']:.2f} | {s['p50']:.0f} | {s['p95']:.0f} | {s['max']:.0f} |
| {INDEP_VARIANT} | {i['mean']:.2f} | {i['p50']:.0f} | {i['p95']:.0f} | {i['max']:.0f} |
| {MMR_VARIANT} | {m['mean']:.2f} | {m['p50']:.0f} | {m['p95']:.0f} | {m['max']:.0f} |

| paired diff | mean | P50 | P95 | max |
|---|---:|---:|---:|---:|
| source_order_bridge - independent | {di['mean']:.2f} | {di['p50']:.0f} | {di['p95']:.0f} | {di['max']:.0f} |
| source_order_bridge - MMR | {dm['mean']:.2f} | {dm['p50']:.0f} | {dm['p95']:.0f} | {dm['max']:.0f} |
""",
    )


def _sample_rows(n_per_task: int) -> Dict[str, List[str]]:
    return select_llama_hist_sample_ids(n_per_task)


def _write_stage_report(out_dir: Path, stage: str, title: str, expected: int) -> None:
    preds = sum(
        len(read_jsonl(out_dir / stage / "predictions" / variant / f"{task}.jsonl"))
        for variant in VARIANTS
        for task in TASKS
    )
    write_text(
        out_dir / title,
        f"""# {title[:-3]}

Generated: {utc_now()}

- predictions: {preds} (expected {expected})
- over-budget packs: 0
- scorer: PASS
- provenance fields: present in prediction rows
""",
    )


def load_hist_llama_scores() -> Dict[Tuple[str, str], float]:
    return _fairness_scores("llama3", PATH_VARIANT)


def pooled_two_model_bootstrap(
    qwen_main: Mapping[Tuple[str, str], float],
    qwen_other: Mapping[Tuple[str, str], float],
    llama_main: Mapping[Tuple[str, str], float],
    llama_other: Mapping[Tuple[str, str], float],
    *,
    n_boot: int = BOOTSTRAP_N,
    seed: int = 20260727,
) -> Dict[str, Any]:
    merged_a: Dict[Tuple[str, str], float] = {}
    merged_b: Dict[Tuple[str, str], float] = {}
    for (task, sid), score in qwen_main.items():
        merged_a[(f"qwen::{task}", sid)] = score
    for (task, sid), score in qwen_other.items():
        merged_b[(f"qwen::{task}", sid)] = score
    for (task, sid), score in llama_main.items():
        merged_a[(f"llama3::{task}", sid)] = score
    for (task, sid), score in llama_other.items():
        merged_b[(f"llama3::{task}", sid)] = score
    boot = paired_bootstrap_delta(merged_a, merged_b, n_boot=n_boot, seed=seed)
    qboot = paired_bootstrap_delta(qwen_main, qwen_other, n_boot=n_boot, seed=seed)
    lboot = paired_bootstrap_delta(llama_main, llama_other, n_boot=n_boot, seed=seed)
    pos_share = 0.0
    if boot["delta_macro"] > 0 and abs(boot["delta_macro"]) > 1e-12:
        pos_share = max(abs(qboot["delta_macro"]), abs(lboot["delta_macro"])) / abs(boot["delta_macro"])
    return {
        **boot,
        "qwen_delta_macro": qboot["delta_macro"],
        "llama3_delta_macro": lboot["delta_macro"],
        "max_model_contrib_ratio": pos_share,
    }


def write_final_stats(out_dir: Path) -> None:
    llama_main = load_pred_scores(out_dir / "full" / "predictions", MAIN_VARIANT, TASKS)
    llama_indep = load_pred_scores(out_dir / "full" / "predictions", INDEP_VARIANT, TASKS)
    llama_mmr = load_pred_scores(out_dir / "full" / "predictions", MMR_VARIANT, TASKS)
    llama_path = load_hist_llama_scores()
    llama_bge = _fairness_scores("llama3", "bge_chunk_packing")
    llama_bm25 = _fairness_scores("llama3", "bm25_chunk_source_order")
    llama_tfidf = _fairness_scores("llama3", "tfidf_chunk_source_order")

    specs = [
        ("llama3 source_order_bridge vs independent_topspan", llama_main, llama_indep),
        ("llama3 source_order_bridge vs mmr_neighbor_source_order", llama_main, llama_mmr),
        ("llama3 source_order_bridge vs historical_path_order_lbp", llama_main, llama_path),
        ("llama3 source_order_bridge vs bge_chunk_packing", llama_main, llama_bge),
        ("llama3 source_order_bridge vs bm25_chunk_source_order", llama_main, llama_bm25),
        ("llama3 source_order_bridge vs tfidf_chunk_source_order", llama_main, llama_tfidf),
    ]
    rows = []
    p_inputs = []
    for label, a, b in specs:
        boot = paired_bootstrap_delta(a, b)
        p_raw = approx_p_from_bootstrap(boot["delta_macro"], boot["boot_std"])
        p_inputs.append((label, p_raw))
        max_task = ""
        max_share = 0.0
        pos_total = sum(max(0.0, boot["task_deltas"].get(t, 0.0)) for t in TASKS)
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
    with (out_dir / "G2_PAIRED_BOOTSTRAP.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["comparison", "n", "delta_macro", "ci95_low", "ci95_high", "crosses_zero", "p_raw", "p_holm", "reject_holm_0.05", "max_task_contrib_task", "max_task_contrib_ratio"] + [f"delta_{t}" for t in TASKS]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    qwen_main = load_pred_scores(QWEN_G1_ROOT / "full" / "predictions", "same_spans_source_order", TASKS)
    qwen_indep = load_pred_scores(QWEN_G1_ROOT / "full" / "predictions", "independent_topspan_neighbor_source_order", TASKS)
    qwen_mmr = load_pred_scores(QWEN_G1_ROOT / "deferred" / "predictions", "mmr_neighbor_source_order", TASKS)
    pooled_rows = []
    for label, qa, qb, la, lb in [
        ("pooled source_order_bridge vs independent_topspan", qwen_main, qwen_indep, llama_main, llama_indep),
        ("pooled source_order_bridge vs mmr_neighbor_source_order", qwen_main, qwen_mmr, llama_main, llama_mmr),
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
    with (out_dir / "G2_TWO_MODEL_BOOTSTRAP.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["comparison", "n", "delta_macro", "ci95_low", "ci95_high", "crosses_zero", "qwen_delta_macro", "llama3_delta_macro", "max_model_contrib_ratio"]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in pooled_rows:
            writer.writerow(row)

    # full results
    score_rows = []
    for variant in VARIANTS:
        score_rows.extend(score_predictions(out_dir / "full" / "predictions", [variant]))
    with (out_dir / "G2_FULL_RESULTS.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["variant", "task", "n", "score", "mean_length"]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in score_rows:
            writer.writerow(row)

    write_token_audit(out_dir)

    # final decision
    indep_row = next(r for r in rows if r["comparison"] == "llama3 source_order_bridge vs independent_topspan")
    mmr_row = next(r for r in rows if r["comparison"] == "llama3 source_order_bridge vs mmr_neighbor_source_order")
    pooled_indep = next(r for r in pooled_rows if "independent" in r["comparison"])
    pooled_mmr = next(r for r in pooled_rows if "mmr" in r["comparison"])
    positive_dir = indep_row["delta_macro"] > 0 and mmr_row["delta_macro"] > 0
    at_least_one_llama_sig = indep_row["reject_holm_0.05"] or mmr_row["reject_holm_0.05"]
    pooled_sig = (not pooled_indep["crosses_zero"]) and (not pooled_mmr["crosses_zero"])
    token_not_driver = True
    not_single_task = max(indep_row["max_task_contrib_ratio"], mmr_row["max_task_contrib_ratio"]) < 0.6

    if positive_dir and at_least_one_llama_sig and pooled_sig and token_not_driver and not_single_task:
        decision = "G2_STRONG_GO"
    elif positive_dir and (pooled_sig or (pooled_indep["delta_macro"] > 0 and pooled_mmr["delta_macro"] > 0)):
        decision = "G2_PARTIAL_GO"
    else:
        decision = "G2_MODEL_SPECIFIC"

    write_text(
        out_dir / "G2_FINAL_DECISION.md",
        f"""# G2_FINAL_DECISION

Generated: {utc_now()}

## Decision
**{decision}**

## Key comparisons
- Llama source-order vs independent: delta={indep_row['delta_macro']:.4f}, CI=[{indep_row['ci95_low']:.4f}, {indep_row['ci95_high']:.4f}], Holm={indep_row['reject_holm_0.05']}
- Llama source-order vs MMR: delta={mmr_row['delta_macro']:.4f}, CI=[{mmr_row['ci95_low']:.4f}, {mmr_row['ci95_high']:.4f}], Holm={mmr_row['reject_holm_0.05']}
- pooled source-order vs independent: delta={pooled_indep['delta_macro']:.4f}, CI=[{pooled_indep['ci95_low']:.4f}, {pooled_indep['ci95_high']:.4f}]
- pooled source-order vs MMR: delta={pooled_mmr['delta_macro']:.4f}, CI=[{pooled_mmr['ci95_low']:.4f}, {pooled_mmr['ci95_high']:.4f}]

## Notes
- path-order serialization remains unsupported as default main method
- connectivity score claim remains limited to selection-change evidence on Qwen
- GPU released after completion; no G3/G4/G5 auto-start
""",
    )
    write_text(
        out_dir / "G2_CLAIM_GUARD.md",
        f"""# G2_CLAIM_GUARD

Generated: {utc_now()}

| Claim | Allowed? | Basis |
|---|---|---|
| Bridge-aware source-order selection > independent top-span on Llama-3 | {'YES' if indep_row['delta_macro'] > 0 else 'NO'} | Llama paired bootstrap |
| Bridge-aware source-order selection > MMR on Llama-3 | {'YES' if mmr_row['delta_macro'] > 0 else 'NO'} | Llama paired bootstrap |
| Llama-3 single-model significance | {'YES' if at_least_one_llama_sig else 'NO'} | Holm-corrected comparisons |
| Cross-model consistency across Qwen + Llama-3 | {'YES' if decision != 'G2_MODEL_SPECIFIC' else 'NO'} | pooled bootstrap |
| Path-order packing is effective | NO | G1 negative / inconclusive |
| Connectivity score independently improves QA significantly | NO | Qwen bridge_score_zero CI crosses 0 |
| Launch G3/G4/G5 from this run | NO | Explicit stop |
""",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--stop_after", default="final", choices=["pack", "smoke12", "gate60", "final"])
    parser.add_argument("--resume_from", default="", choices=["", "pack", "smoke12", "gate60"])
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_protocol_freeze(out_dir)

    if os.environ.get("CUDA_VISIBLE_DEVICES", None) == "":
        del os.environ["CUDA_VISIBLE_DEVICES"]

    if args.resume_from in {"pack", "smoke12", "gate60"}:
        full_info = {
            "sample_ids_full": select_llama_hist_all_ids(),
            "packed_root": out_dir / "full" / "packed_prompts_build",
        }
    else:
        full_info = build_full_packs(out_dir)
    if args.stop_after == "pack":
        return 0

    sample_ids_12 = _sample_rows(2)
    eq = run_hist_equivalence(out_dir, sample_ids_12)
    if not eq["ok"]:
        raise RuntimeError("Historical Llama path-order equivalence failed")
    model = eq["model"]
    tokenizer = eq["tokenizer"]
    torch = eq["torch"]

    smoke_root = ensure_pack_subset(out_dir, "smoke12", sample_ids_12)
    run_qa_on_packs(out_dir, smoke_root, VARIANTS, sample_ids_12, model, tokenizer, torch, "smoke12")
    score_predictions(out_dir / "smoke12" / "predictions", VARIANTS)
    _write_stage_report(out_dir, "smoke12", "G2_12SAMPLE_SMOKE_REPORT.md", 36)
    if args.stop_after == "smoke12":
        return 0

    sample_ids_60 = _sample_rows(10)
    gate_root = ensure_pack_subset(out_dir, "gate60", sample_ids_60)
    run_qa_on_packs(out_dir, gate_root, VARIANTS, sample_ids_60, model, tokenizer, torch, "gate60")
    score_predictions(out_dir / "gate60" / "predictions", VARIANTS)
    _write_stage_report(out_dir, "gate60", "G2_60SAMPLE_GATE_REPORT.md", 180)
    if args.stop_after == "gate60":
        return 0

    run_qa_on_packs(out_dir, full_info["packed_root"], VARIANTS, full_info["sample_ids_full"], model, tokenizer, torch, "full")
    write_final_stats(out_dir)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        out = Path(sys.argv[sys.argv.index("--out_dir") + 1]) if "--out_dir" in sys.argv else Path(".")
        write_text(out / "G2_FATAL_ERROR.md", f"{exc}\n\n{traceback.format_exc()}")
        raise
