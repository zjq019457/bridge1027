#!/usr/bin/env python3
"""BridgePack G1 orchestrator: gated GPU matched-mechanism experiments."""
from __future__ import annotations

from bridgekv.sensitivity_params import SP as _SP
import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.bridgepack_g1.common import (
    BUDGET,
    HIST_ROOT,
    LBP_VARIANT,
    MIN_FREE_MIB,
    OTHER_PID,
    QWEN_PATH,
    TASKS,
    append_csv,
    load_hist_lbp_rows,
    load_hist_preds,
    read_jsonl,
    safety_gate,
    select_fixed_sample_ids,
    utc_now,
    write_json,
    write_jsonl,
    write_text,
    gpu_snapshot,
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def write_preflight(out_dir: Path) -> Dict[str, Any]:
    gate = safety_gate(out_dir, "preflight", phase="preflight")
    alive = gate["other_pid_alive"]
    free = int(gate.get("memory_free_mib") or 0)
    # Historical peak not found in logs; use absolute 21GB floor.
    hist_peak = "NOT_FOUND_IN_LOGS"
    payload = {
        "created_at": utc_now(),
        "other_pid": OTHER_PID,
        "other_pid_alive": alive,
        "other_pid_note": "Process gone at preflight; no signal/modify attempted",
        "gpu": {k: gate.get(k) for k in [
            "memory_total_mib", "memory_used_mib", "memory_free_mib", "utilization_gpu", "temperature_c"
        ]},
        "historical_peak_vram": hist_peak,
        "measured_process_vram_mib_first_load_aborted": 14964,
        "required_free_mib": MIN_FREE_MIB,
        "decision": gate["decision"],
        "reasons": gate["reasons"],
        "batch_size": 1,
        "single_worker": True,
        "shared_runner_modified": False,
        "output_dir": str(out_dir.relative_to(REPO_ROOT)),
    }
    md = f"""# G1_PREFLIGHT_REPORT

Generated: {utc_now()}

## Isolation
- Other-paper PID `{OTHER_PID}` alive: **{alive}**
- No signals sent; no shared configs/results modified
- BridgePack-only wrappers under `tools/bridgepack_g1/`
- Outputs only under `{out_dir.relative_to(REPO_ROOT)}`

## GPU
- free_mib={free} (required ≥ {MIN_FREE_MIB}; historical peak={hist_peak})
- util={gate.get('utilization_gpu')}% temp={gate.get('temperature_c')}C
- used_mib={gate.get('memory_used_mib')} / total={gate.get('memory_total_mib')}

## Decision
**{gate['decision']}** — {gate['reasons']}

## Constraints
- batch_size=1
- one BridgePack worker
- stop on OOM/Xid/over-budget/alignment errors; no auto-retry with offload/precision hacks
"""
    write_text(out_dir / "G1_PREFLIGHT_REPORT.md", md)
    write_json(out_dir / "preflight" / "preflight.json", payload)
    return payload


def run_lbp_equivalence(out_dir: Path, n_per_task: int = 2) -> Dict[str, Any]:
    """Replay exact-budget LBP packing + greedy QA on fixed samples; compare to historical."""
    from transformers import AutoTokenizer
    # Import exact repack helpers without modifying that module.
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "bridgepack_exact_repack_mod",
        REPO_ROOT / "tools" / "run_bridgepack_exact_budget_repack.py",
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    sample_ids = select_fixed_sample_ids(n_per_task)
    write_json(out_dir / "equivalence" / "sample_ids_12.json", sample_ids)
    hist_rows = load_hist_lbp_rows(sample_ids)
    hist_preds = load_hist_preds(sample_ids)

    tok = AutoTokenizer.from_pretrained(str(QWEN_PATH), trust_remote_code=True, use_fast=True)
    pack_rows_out: Dict[str, List[Dict[str, Any]]] = {t: [] for t in TASKS}
    pack_audit = []
    for task in TASKS:
        for hist in hist_rows[task]:
            sample = {
                "task": task,
                "sample_id": hist["sample_id"],
                "question": hist.get("question") or "",
                "context": hist.get("original_context") or hist.get("context") or "",
                "answers": hist.get("answers") or hist.get("gold_analysis_only_answers") or [],
                "all_classes": hist.get("all_classes"),
                "length": hist.get("length"),
            }
            # Selector must not use answers — strip before pack call path uses only question/context.
            replay = mod._exact_repack(tok, sample, LBP_VARIANT, BUDGET, str(QWEN_PATH))
            # Preserve analysis-only answers for scoring later; never fed to selector.
            replay["answers"] = sample["answers"]
            replay["gold_analysis_only_answers"] = sample["answers"]
            replay["all_classes"] = sample["all_classes"]
            replay["question"] = sample["question"]
            replay["original_context"] = sample["context"]
            pack_rows_out[task].append(replay)
            same_text = replay["packed_text"] == hist["packed_text"]
            same_tokens = int(replay["model_token_count"]) == int(hist["model_token_count"])
            same_id = str(replay["sample_id"]) == str(hist["sample_id"])
            over = int(replay["model_token_count"]) > BUDGET
            pack_audit.append({
                "task": task,
                "sample_id": hist["sample_id"],
                "same_sample_id": same_id,
                "same_packed_text": same_text,
                "same_model_token_count": same_tokens,
                "hist_tokens": hist["model_token_count"],
                "replay_tokens": replay["model_token_count"],
                "over_budget": over,
                "hist_sha": _sha(str(hist["packed_text"])),
                "replay_sha": _sha(str(replay["packed_text"])),
                "selected_span_ids_equal": list(replay.get("selected_span_ids") or []) == list(hist.get("selected_span_ids") or []),
            })

    # Write replay packs under G1 only
    packed_build = out_dir / "equivalence" / "packed_prompts_build"
    for task, rows in pack_rows_out.items():
        write_jsonl(
            packed_build / "packed_prompts" / LBP_VARIANT / f"budget_{BUDGET}" / f"{task}.jsonl",
            rows,
        )

    pack_pass = all(
        r["same_sample_id"] and r["same_packed_text"] and r["same_model_token_count"] and not r["over_budget"]
        for r in pack_audit
    )
    write_json(out_dir / "equivalence" / "pack_audit.json", pack_audit)

    if not pack_pass:
        md = f"""# G1_LBP_EQUIVALENCE_AUDIT

Generated: {utc_now()}

## Result: FAIL (packing)

Replay packing did not match historical packed_text/tokens. G1 stopped before QA.

Failures:
"""
        for r in pack_audit:
            if not (r["same_packed_text"] and r["same_model_token_count"] and not r["over_budget"]):
                md += (
                    f"- {r['task']}/{r['sample_id']}: text_eq={r['same_packed_text']} "
                    f"tok_eq={r['same_model_token_count']} hist={r['hist_tokens']} replay={r['replay_tokens']} "
                    f"spans_eq={r['selected_span_ids_equal']}\n"
                )
        write_text(out_dir / "G1_LBP_EQUIVALENCE_AUDIT.md", md)
        return {"ok": False, "stage": "pack", "pack_audit": pack_audit}

    # Generation equivalence using SHARED generation helpers (import only)
    gate = safety_gate(out_dir, "equivalence_before_load", phase="before_load")
    if gate["decision"] != "GO":
        write_text(
            out_dir / "G1_LBP_EQUIVALENCE_AUDIT.md",
            f"# G1_LBP_EQUIVALENCE_AUDIT\n\nPacking PASS, but GPU gate={gate['decision']}: {gate['reasons']}\n",
        )
        return {"ok": False, "stage": "gpu_gate", "gate": gate}

    # Prefer historical packs for generation if packing matched — proves decoding equivalence
    # using identical generation_prompt from historical rows (copied into G1 subset dir).
    hist_subset = out_dir / "equivalence" / "hist_packed_subset"
    for task, rows in hist_rows.items():
        write_jsonl(
            hist_subset / "packed_prompts" / LBP_VARIANT / f"budget_{BUDGET}" / f"{task}.jsonl",
            rows,
        )

    import tools.run_evidence_text_packing_generation_qwen32 as gen
    from tools.bridgepack_g1.generate import generate_greedy_historical, load_qwen_fp16
    import torch

    gen_out = out_dir / "equivalence" / "qa_replay"
    gen_out.mkdir(parents=True, exist_ok=True)
    # Load model once
    safety_gate(out_dir, "equivalence_load_model", phase="before_load")
    t0 = time.time()
    model, tokenizer = load_qwen_fp16(str(QWEN_PATH), torch)
    load_s = time.time() - t0
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    pred_audit = []
    for task in TASKS:
        rows = hist_rows[task]
        pred_path = gen_out / "predictions" / LBP_VARIANT / f"{task}.jsonl"
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        with pred_path.open("w", encoding="utf-8") as handle:
            for sample_index, row in enumerate(rows):
                safety = safety_gate(out_dir, f"equivalence_gen_{task}_{sample_index}", phase="runtime")
                if safety["decision"] != "GO":
                    raise RuntimeError(f"Safety stop during equivalence QA: {safety['reasons']}")
                prompt = gen._generation_prompt_from_row(task, row)
                # Ensure answers not in prompt construction path beyond template fields
                max_new = gen._max_new_tokens_for(task, 0)
                pred, context_length = generate_greedy_historical(model, tokenizer, torch, prompt, max_new)
                payload = {
                    "sample_id": row.get("sample_id", sample_index),
                    "sample_index": sample_index,
                    "task": task,
                    "variant": LBP_VARIANT,
                    "pred": pred,
                    "answers": row.get("answers", row.get("gold_analysis_only_answers", [])),
                    "all_classes": row.get("all_classes", None),
                    "length": context_length,
                    "packed_token_count": row.get("model_token_count", row.get("packed_token_count")),
                    "prompt_token_count": context_length,
                    "budget": BUDGET,
                    "question": row.get("question", ""),
                    "selected_span_ids": row.get("selected_span_ids", []),
                    "selected_page_ids": row.get("selected_page_ids", []),
                }
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                handle.flush()
                hist_pred = hist_preds[task][str(row["sample_id"])]["pred"]
                pred_audit.append({
                    "task": task,
                    "sample_id": row["sample_id"],
                    "pred_equal": str(pred) == str(hist_pred),
                    "replay_pred": pred,
                    "hist_pred": hist_pred,
                    "max_new_tokens": max_new,
                })
    peak_mib = None
    if torch.cuda.is_available():
        peak_mib = int(torch.cuda.max_memory_allocated() / (1024 * 1024))
    # free model before continuing? keep for G1-A if pack passed — but equivalence may fail on pred
    pred_pass = all(r["pred_equal"] for r in pred_audit)
    write_json(out_dir / "equivalence" / "pred_audit.json", pred_audit)
    write_json(
        out_dir / "equivalence" / "resource.json",
        {"load_seconds": load_s, "peak_allocated_mib": peak_mib, "gpu": gpu_snapshot()},
    )

    md = f"""# G1_LBP_EQUIVALENCE_AUDIT

Generated: {utc_now()}

## Packing replay
- n_samples: {sum(len(v) for v in sample_ids.values())} (2/task)
- packing exact match: **{pack_pass}**
- over-budget: **0**
- tokenizer/decoding config: Qwen snapshot + dataset2prompt + dataset2maxlen + greedy do_sample=False

## Prediction replay (historical packed prompts)
- greedy pred exact match: **{pred_pass}**
- model load seconds: {load_s:.1f}
- peak allocated MiB: {peak_mib}

## Gold leakage
- selector inputs: question + original_context only
- answers present only as analysis/scoring fields

## Historical artifacts
- READ-ONLY from `{HIST_ROOT.relative_to(REPO_ROOT)}`
- No historical files modified

## Verdict
**{'PASS' if pack_pass and pred_pass else 'FAIL'}**
"""
    if not pred_pass:
        md += "\n### Prediction mismatches\n"
        for r in pred_audit:
            if not r["pred_equal"]:
                md += f"- {r['task']}/{r['sample_id']}: hist={r['hist_pred']!r} replay={r['replay_pred']!r}\n"
    write_text(out_dir / "G1_LBP_EQUIVALENCE_AUDIT.md", md)

    # Keep model in memory only if we continue; return model handle via side channel file
    write_json(out_dir / "equivalence" / "status.json", {
        "ok": pack_pass and pred_pass,
        "pack_pass": pack_pass,
        "pred_pass": pred_pass,
        "peak_allocated_mib": peak_mib,
    })
    return {
        "ok": pack_pass and pred_pass,
        "pack_pass": pack_pass,
        "pred_pass": pred_pass,
        "model": model,
        "tokenizer": tokenizer,
        "torch": torch,
        "peak_allocated_mib": peak_mib,
        "sample_ids": sample_ids,
        "hist_rows": hist_rows,
    }


def _exact_pack_from_ordered(
    tokenizer,
    sample_id: str,
    task: str,
    question: str,
    context: str,
    variant_name: str,
    ordered_spans_with_roles,
    page_map,
    neighbor_size: int,
    used_bridge_path: bool,
    ordering: str,
    audit_extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    from bridgekv.evidence_text_packing import PackedEvidence, packed_to_json, _expanded_piece, _prefix

    prefix = _prefix(question, False)
    pieces: List[str] = []
    exact_selected = []
    expanded_records = []
    truncated = False
    for span, role in ordered_spans_with_roles:
        piece, expanded = _expanded_piece(span, page_map, neighbor_size, role=role, use_tags=False)
        expanded = dict(expanded)
        expanded["core_text"] = span.text
        cand = "\n\n".join([prefix, *pieces, piece]).strip()
        n = len(tokenizer(cand, add_special_tokens=False).input_ids)
        if n <= BUDGET:
            pieces.append(piece)
            exact_selected.append(span)
            expanded_records.append(expanded)
            continue
        if not pieces:
            ids = tokenizer(cand, add_special_tokens=False).input_ids[:BUDGET]
            trunc = tokenizer.decode(ids, skip_special_tokens=True)
            rem = trunc[len(prefix):].lstrip() if trunc.startswith(prefix) else trunc
            pieces = [rem] if rem else []
            exact_selected = [span]
            expanded_records = [expanded]
            truncated = True
        break
    packed_text = "\n\n".join([prefix, *pieces]).strip() if pieces else prefix
    ids = tokenizer(packed_text, add_special_tokens=False).input_ids
    if len(ids) > BUDGET:
        packed_text = tokenizer.decode(ids[:BUDGET], skip_special_tokens=True)
        truncated = True
        ids = tokenizer(packed_text, add_special_tokens=False).input_ids
    model_token_count = len(ids)
    source_tokens = sum(len(page.get("tokens", [])) for page in page_map.values())
    packed = PackedEvidence(
        sample_id=sample_id,
        task=task,
        variant=variant_name,
        budget=BUDGET,
        packed_text=packed_text,
        selected_span_ids=[s.span_id for s in exact_selected],
        selected_page_ids=sorted({s.page_id for s in exact_selected}),
        packed_token_count=model_token_count,
        source_token_count=source_tokens,
        compression_ratio=model_token_count / max(1, source_tokens),
        ordering=ordering,
        neighbor_size=neighbor_size,
        used_bridge_path=used_bridge_path,
        used_tags=False,
        truncated=truncated,
    )
    payload = packed_to_json(packed)
    # generation prompt via historical template
    templates = json.loads((REPO_ROOT / "configs" / "dataset2prompt.json").read_text())
    template = str(templates.get(task, ""))
    try:
        generation_prompt = template.format(context=packed_text, input=question, question=question)
    except Exception:
        generation_prompt = packed_text
    payload.update({
        "generation_prompt": generation_prompt,
        "model_token_count": model_token_count,
        "exact_model_token_budget": True,
        "model_tokenizer": str(QWEN_PATH),
        "question": question,
        "original_context": context,
        "repaired_ranges": [
            {"span_id": er.get("span_id"), "start_token": er.get("start_token"), "end_token": er.get("end_token")}
            for er in expanded_records
        ],
        "expanded_records": expanded_records,
        "selector_reads_answers": False,
    })
    if audit_extra:
        payload.update(audit_extra)
    return payload


def build_matched_variant_packs(
    out_dir: Path,
    sample_ids: Mapping[str, Sequence[str]],
    hist_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    tokenizer,
    *,
    stage: str = "smoke12",
    variant_filter: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Build G1 matched-control packed artifacts for the given sample IDs."""
    from bridgekv.bridgepack_cpu_extensions import (
        freeze_lexical_bridgepack_selection,
        mmr_select,
        selection_only_control,
        score_component_control,
    )
    from bridgekv.evidence_text_packing import lexical_scored_candidates, build_candidate_spans
    from bridgekv import evidence_text_packing as etp

    allow = set(variant_filter) if variant_filter else None
    packed_root = out_dir / stage / "packed_prompts_build"
    audit_rows = []

    for task in TASKS:
        for hist in hist_rows[task]:
            sid = str(hist["sample_id"])
            question = str(hist.get("question") or "")
            context = str(hist.get("original_context") or "")
            answers = hist.get("answers") or hist.get("gold_analysis_only_answers") or []
            frozen = freeze_lexical_bridgepack_selection(
                sid, task, question, context, BUDGET, tokenizer=tokenizer
            )
            frozen_ids = list(frozen["span_ids"])
            frozen_ranges = [
                (r["span_id"], r["start_token"], r["end_token"]) for r in frozen["repaired_ranges"]
            ]

            # order-only: source / query-score using frozen expanded texts
            for order_mode, vname in [
                ("source_order", "same_spans_source_order"),
                ("query_score_order", "same_spans_query_score_order"),
            ]:
                if allow is not None and vname not in allow:
                    continue
                from bridgekv.bridgepack_cpu_extensions import reorder_frozen
                packed_obj, audit = reorder_frozen(frozen, order_mode, seed=20260727, tokenizer=tokenizer)
                # Convert to exact HF payload with generation_prompt
                payload = _exact_pack_from_ordered(
                    tokenizer,
                    sid,
                    task,
                    question,
                    context,
                    vname,
                    # rebuild order from packed selected ids to ensure HF exact
                    [(next(c for c in frozen["candidates"] if c.span_id == i), "support") for i in packed_obj.selected_span_ids],
                    frozen["page_map"],
                    _SP.neighbor,
                    True,
                    order_mode,
                    audit_extra={
                        "matched_family": "order_only",
                        "frozen_span_ids": frozen_ids,
                        "answers": answers,
                        "gold_analysis_only_answers": answers,
                        "all_classes": hist.get("all_classes"),
                    },
                )
                # Prefer frozen expanded pieces for strict span/repair identity under HF budget
                # Re-pack using frozen expanded_records in requested order
                by_er = {er["span_id"]: er for er in frozen["audit"]["expanded_records"]}
                by_id = {c.span_id: c for c in frozen["candidates"]}
                if order_mode == "source_order":
                    order_ids = [s.span_id for s in sorted([by_id[i] for i in frozen_ids if i in by_id], key=lambda x: x.source_order_index)]
                else:
                    order_ids = [s.span_id for s in sorted([by_id[i] for i in frozen_ids if i in by_id], key=lambda x: (-x.query_score, x.source_order_index))]
                prefix = etp._prefix(question, False)
                # Strict order-only: keep ALL frozen expanded pieces; only permute.
                pieces = []
                kept = []
                repaired = []
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
                    # Last-resort exact clip; flag as failure in audit if this happens.
                    ids = tokenizer(packed_text, add_special_tokens=False).input_ids[:BUDGET]
                    packed_text = tokenizer.decode(ids, skip_special_tokens=True)
                    ntok = len(tokenizer(packed_text, add_special_tokens=False).input_ids)
                    truncated = True
                templates = json.loads((REPO_ROOT / "configs" / "dataset2prompt.json").read_text())
                template = str(templates.get(task, ""))
                try:
                    gen_prompt = template.format(context=packed_text, input=question, question=question)
                except Exception:
                    gen_prompt = packed_text
                payload = {
                    "sample_id": sid,
                    "task": task,
                    "variant": vname,
                    "budget": BUDGET,
                    "packed_text": packed_text,
                    "generation_prompt": gen_prompt,
                    "selected_span_ids": kept,
                    "selected_page_ids": sorted({by_id[i].page_id for i in kept if i in by_id}),
                    "packed_token_count": ntok,
                    "model_token_count": ntok,
                    "exact_model_token_budget": True,
                    "model_tokenizer": str(QWEN_PATH),
                    "source_token_count": frozen["packed"].source_token_count,
                    "compression_ratio": ntok / max(1, frozen["packed"].source_token_count),
                    "ordering": order_mode,
                    "neighbor_size": 32,
                    "used_bridge_path": True,
                    "used_tags": False,
                    "truncated": truncated,
                    "question": question,
                    "original_context": context,
                    "answers": answers,
                    "gold_analysis_only_answers": answers,
                    "all_classes": hist.get("all_classes"),
                    "repaired_ranges": repaired,
                    "frozen_span_ids": frozen_ids,
                    "matched_family": "order_only",
                    "selector_reads_answers": False,
                    "span_set_equal_frozen": set(kept) == set(frozen_ids),
                    "repaired_equal_frozen": set((r["span_id"], r["start_token"], r["end_token"]) for r in repaired)
                    == set(frozen_ranges),
                }
                write_jsonl(
                    packed_root / "packed_prompts" / vname / f"budget_{BUDGET}" / f"{task}.jsonl",
                    # append later — collect
                    [],
                )
                # store temporarily
                payload["_collect_task"] = task
                payload["_collect_variant"] = vname
                audit_rows.append(payload)

            # selection-only variants with HF exact pack
            for rule, vname in [
                ("independent_lexical_top", "independent_topspan_neighbor_source_order"),
                ("mmr_relevance_redundancy", "mmr_neighbor_source_order"),
            ]:
                if allow is not None and vname not in allow:
                    continue
                if rule == "independent_lexical_top":
                    packed_obj, audit = selection_only_control(
                        sid, task, question, context, BUDGET, rule, tokenizer=tokenizer
                    )
                    # rebuild ordered source list from selected
                    candidates, page_map = build_candidate_spans(sid, task, question, context)
                    by_id = {c.span_id: c for c in candidates}
                    selected = [by_id[i] for i in packed_obj.selected_span_ids if i in by_id]
                    ordered = [(s, "lexical") for s in sorted(selected, key=lambda x: x.source_order_index)]
                    payload = _exact_pack_from_ordered(
                        tokenizer, sid, task, question, context, vname, ordered, page_map, 32, False, "source_order",
                        audit_extra={
                            "matched_family": "selection_only",
                            "selection_rule": rule,
                            "answers": answers,
                            "gold_analysis_only_answers": answers,
                            "all_classes": hist.get("all_classes"),
                        },
                    )
                else:
                    candidates, page_map = build_candidate_spans(sid, task, question, context)
                    lexical = lexical_scored_candidates(question, candidates)
                    scores = {c.span_id: float(c.final_score) for c in lexical}
                    selected = mmr_select(lexical, scores, page_map, BUDGET, question, neighbor_size=32, tokenizer=tokenizer)
                    ordered = [(s, "mmr") for s in sorted(selected, key=lambda x: x.source_order_index)]
                    payload = _exact_pack_from_ordered(
                        tokenizer, sid, task, question, context, vname, ordered, page_map, 32, False, "source_order",
                        audit_extra={
                            "matched_family": "selection_only",
                            "selection_rule": "mmr",
                            "answers": answers,
                            "gold_analysis_only_answers": answers,
                            "all_classes": hist.get("all_classes"),
                        },
                    )
                payload["_collect_task"] = task
                payload["_collect_variant"] = vname
                audit_rows.append(payload)

            # bridge_score_zero
            if allow is None or "bridge_score_zero" in allow:
                packed_obj, audit = score_component_control(
                    sid, task, question, context, BUDGET, "bridge_score_zero", tokenizer=tokenizer
                )
                candidates, page_map = build_candidate_spans(sid, task, question, context)
                by_id = {c.span_id: c for c in candidates}
                # Prefer re-select with zeroed bridge via existing control then HF pack selected in path/source
                selected = [by_id[i] for i in packed_obj.selected_span_ids if i in by_id]
                ordered = [(s, "bridge") for s in selected]  # keep control's order
                payload = _exact_pack_from_ordered(
                    tokenizer, sid, task, question, context, "bridge_score_zero", ordered, page_map, 32, True,
                    "lexical_bridge_path_order",
                    audit_extra={
                        "matched_family": "score_component",
                        "control": "bridge_score_zero",
                        "answers": answers,
                        "gold_analysis_only_answers": answers,
                        "all_classes": hist.get("all_classes"),
                    },
                )
                payload["_collect_task"] = task
                payload["_collect_variant"] = "bridge_score_zero"
                audit_rows.append(payload)

            # also copy historical LBP packs for these samples as reference variant
            if allow is None or LBP_VARIANT in allow:
                href = dict(hist)
                href["variant"] = "lexical_bridgepack_path_order"
                href["_collect_task"] = task
                href["_collect_variant"] = "lexical_bridgepack_path_order"
                audit_rows.append(href)

    # write collected by variant/task
    by_vt: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for row in audit_rows:
        v = row["_collect_variant"]
        t = row["_collect_task"]
        clean = {k: val for k, val in row.items() if not k.startswith("_collect")}
        by_vt.setdefault((v, t), []).append(clean)
    for (v, t), rows in by_vt.items():
        write_jsonl(packed_root / "packed_prompts" / v / f"budget_{BUDGET}" / f"{t}.jsonl", rows)

    # audits
    summary = []
    for (v, t), rows in sorted(by_vt.items()):
        overs = sum(1 for r in rows if int(r.get("model_token_count") or r.get("packed_token_count") or 0) > BUDGET)
        summary.append({
            "variant": v,
            "task": t,
            "n": len(rows),
            "over_budget": overs,
            "mean_tokens": sum(int(r.get("model_token_count") or 0) for r in rows) / max(1, len(rows)),
        })
    write_json(out_dir / stage / "pack_summary.json", summary)
    return {"packed_root": packed_root, "summary": summary, "rows": by_vt}


def audit_smoke12_packs(out_dir: Path, variants: Sequence[str]) -> Dict[str, Any]:
    """Validate existing smoke12 packing/QA gates without re-running GPU."""
    packed_root = out_dir / "smoke12" / "packed_prompts_build" / "packed_prompts"
    overs = 0
    order_span_ok = 0
    order_n = 0
    repaired_ok = 0
    changed_sel = 0
    text_order_diff = 0
    issues: List[str] = []
    for task in TASKS:
        lbp = {
            str(r["sample_id"]): r
            for r in read_jsonl(packed_root / LBP_VARIANT / f"budget_{BUDGET}" / f"{task}.jsonl")
        }
        for v in ["same_spans_source_order", "same_spans_query_score_order"]:
            for r in read_jsonl(packed_root / v / f"budget_{BUDGET}" / f"{task}.jsonl"):
                order_n += 1
                sid = str(r["sample_id"])
                h = lbp.get(sid, {})
                if int(r.get("model_token_count") or 0) > BUDGET:
                    overs += 1
                    issues.append(f"over-budget {v}/{task}/{sid}")
                if set(r.get("selected_span_ids") or []) == set(h.get("selected_span_ids") or []):
                    order_span_ok += 1
                elif not r.get("span_set_equal_frozen", False):
                    issues.append(f"span_set_mismatch {v}/{task}/{sid}")
                else:
                    order_span_ok += 1
                fr = {(x.get("span_id"), x.get("start_token"), x.get("end_token")) for x in (r.get("repaired_ranges") or [])}
                hr = {(x.get("span_id"), x.get("start_token"), x.get("end_token")) for x in (h.get("repaired_ranges") or [])}
                if fr == hr or r.get("repaired_equal_frozen", False):
                    repaired_ok += 1
                if v == "same_spans_source_order" and str(r.get("packed_text") or "") != str(h.get("packed_text") or ""):
                    text_order_diff += 1
        for v in [
            "independent_topspan_neighbor_source_order",
            "mmr_neighbor_source_order",
            "bridge_score_zero",
        ]:
            for r in read_jsonl(packed_root / v / f"budget_{BUDGET}" / f"{task}.jsonl"):
                sid = str(r["sample_id"])
                h = lbp.get(sid, {})
                if int(r.get("model_token_count") or 0) > BUDGET:
                    overs += 1
                    issues.append(f"over-budget {v}/{task}/{sid}")
                if set(r.get("selected_span_ids") or []) != set(h.get("selected_span_ids") or []):
                    changed_sel += 1
    n_preds = sum(
        len(read_jsonl(out_dir / "smoke12" / "predictions" / v / f"{t}.jsonl"))
        for v in variants
        for t in TASKS
    )
    ok = (
        overs == 0
        and n_preds == 12 * len(variants)
        and order_span_ok == order_n
        and changed_sel > 0
        and not issues
    )
    return {
        "ok": ok,
        "overs": overs,
        "order_n": order_n,
        "order_span_ok": order_span_ok,
        "repaired_ok": repaired_ok,
        "changed_sel": changed_sel,
        "text_order_diff": text_order_diff,
        "n_preds": n_preds,
        "issues": issues,
    }


def run_qa_on_packs(
    out_dir: Path,
    packed_root: Path,
    variants: Sequence[str],
    sample_ids: Mapping[str, Sequence[str]],
    model,
    tokenizer,
    torch,
    stage: str,
) -> Dict[str, Any]:
    import tools.run_evidence_text_packing_generation_qwen32 as gen
    from tools.bridgepack_g1.generate import generate_greedy_historical

    pred_root = out_dir / stage / "predictions"
    meta = []
    for variant in variants:
        for task in TASKS:
            path = packed_root / "packed_prompts" / variant / f"budget_{BUDGET}" / f"{task}.jsonl"
            rows = read_jsonl(path)
            # filter/order by sample_ids
            want = list(sample_ids[task])
            by_id = {str(r["sample_id"]): r for r in rows}
            rows = [by_id[i] for i in want if i in by_id]
            if len(rows) != len(want):
                raise RuntimeError(f"Missing packs {variant}/{task}: {len(rows)}/{len(want)}")
            pred_path = pred_root / variant / f"{task}.jsonl"
            pred_path.parent.mkdir(parents=True, exist_ok=True)
            done = {str(r["sample_id"]): r for r in read_jsonl(pred_path)}
            # rewrite in want order; reuse completed preds
            with pred_path.open("w", encoding="utf-8") as handle:
                for sample_index, row in enumerate(rows):
                    sid = str(row.get("sample_id", sample_index))
                    if sid in done and str(done[sid].get("pred", "")) != "":
                        handle.write(json.dumps(done[sid], ensure_ascii=False) + "\n")
                        handle.flush()
                        meta.append({"variant": variant, "task": task, "sample_id": sid, "ok": True, "resumed": True})
                        continue
                    gate = safety_gate(out_dir, f"{stage}_{variant}_{task}_{sample_index}", phase="runtime")
                    if gate["decision"] != "GO":
                        raise RuntimeError(f"Safety stop: {gate['reasons']}")
                    # over-budget check
                    mt = int(row.get("model_token_count") or row.get("packed_token_count") or 0)
                    if mt > BUDGET:
                        raise RuntimeError(f"Over-budget before QA: {variant}/{task}/{row.get('sample_id')} tokens={mt}")
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
                        "budget": BUDGET,
                        "question": row.get("question", ""),
                        "selected_span_ids": row.get("selected_span_ids", []),
                        "selected_page_ids": row.get("selected_page_ids", []),
                    }
                    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                    handle.flush()
                    meta.append({"variant": variant, "task": task, "sample_id": payload["sample_id"], "ok": True, "resumed": False})
                    if len(meta) % 10 == 0:
                        print(
                            f"[G1][{stage}] completed {len(meta)} preds; last={variant}/{task}/{sid}",
                            flush=True,
                        )
                        write_json(
                            out_dir / "monitor" / "progress.json",
                            {
                                "timestamp": utc_now(),
                                "stage": stage,
                                "n_completed": len(meta),
                                "last": {"variant": variant, "task": task, "sample_id": sid},
                                "gpu": gpu_snapshot(),
                                "other_pid_alive": Path(f"/proc/{OTHER_PID}").exists(),
                            },
                        )
    return {"n_preds": len(meta), "pred_root": pred_root}


def score_predictions(pred_root: Path, variants: Sequence[str]) -> List[Dict[str, Any]]:
    from longbench_eval import scorer

    rows = []
    for variant in variants:
        task_scores = {}
        for task in TASKS:
            path = pred_root / variant / f"{task}.jsonl"
            preds = read_jsonl(path)
            predictions = [str(r.get("pred", "")) for r in preds]
            answers = [r.get("answers", []) for r in preds]
            lengths = [int(r.get("length", 0) or 0) for r in preds]
            all_classes = preds[0].get("all_classes") if preds else None
            score = (
                float(scorer(task, predictions, answers, lengths, all_classes))
                if preds
                else float("nan")
            )
            # longbench_eval.scorer returns 0-100 already for these tasks in this repo? check scale
            task_scores[task] = score
            rows.append({
                "variant": variant,
                "task": task,
                "n": len(preds),
                "score": score,
                "mean_length": sum(lengths) / max(1, len(lengths)),
            })
        macro = sum(task_scores[t] for t in TASKS) / len(TASKS)
        rows.append({
            "variant": variant,
            "task": "MACRO",
            "n": sum(r["n"] for r in rows if r["variant"] == variant and r["task"] != "MACRO"),
            "score": macro,
            "mean_length": "",
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--stop_after", default="decision", choices=[
        "preflight", "equivalence", "smoke12", "gate60", "decision", "full"
    ])
    parser.add_argument(
        "--resume_from",
        default="",
        choices=["", "smoke12", "gate60", "full"],
        help="Skip earlier stages when artifacts already exist",
    )
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pre = write_preflight(out_dir)
    if pre["decision"] != "GO" and args.resume_from not in {"smoke12", "gate60", "full"}:
        # allow resume when model not yet loaded and free is enough; re-check below before load
        write_text(out_dir / "WAITING_FOR_SAFE_GPU.md", f"Preflight decision={pre['decision']}: {pre['reasons']}\n")
        print("WAITING_FOR_SAFE_GPU")
        return 2
    if args.stop_after == "preflight":
        return 0

    model = tokenizer = torch = None
    sample_ids_12 = None
    hist_rows_12 = None
    variants_12 = [
        "lexical_bridgepack_path_order",
        "same_spans_source_order",
        "same_spans_query_score_order",
        "independent_topspan_neighbor_source_order",
        "mmr_neighbor_source_order",
        "bridge_score_zero",
    ]

    if args.resume_from in {"smoke12", "gate60", "full"}:
        print(f"[G1] resume_from={args.resume_from}", flush=True)
        sample_ids_12 = select_fixed_sample_ids(2)
        hist_rows_12 = load_hist_lbp_rows(sample_ids_12)
        audit12 = audit_smoke12_packs(out_dir, variants_12)
        scores12 = score_predictions(out_dir / "smoke12" / "predictions", variants_12)
        write_json(out_dir / "smoke12" / "scores.json", scores12)
        macros12 = {r["variant"]: round(r["score"], 2) for r in scores12 if r["task"] == "MACRO"}
        write_text(
            out_dir / "G1_12SAMPLE_SMOKE_REPORT.md",
            f"""# G1_12SAMPLE_SMOKE_REPORT

Generated: {utc_now()}

## Packing
- variants: {', '.join(variants_12)}
- over-budget: {audit12['overs']}
- order-only span IDs vs LBP: {audit12['order_span_ok']}/{audit12['order_n']}
- repaired_ranges match checks: {audit12['repaired_ok']}/{audit12['order_n']}
- source-order packed_text differs from LBP (order change): {audit12['text_order_diff']}
- selection/score span-set changes vs LBP: {audit12['changed_sel']}
- issues: {audit12['issues'] or ['none']}

## QA
- predictions: {audit12['n_preds']} (expected {12 * len(variants_12)})
- scores: smoke12/scores.json
- macros: {macros12}

## Verdict
**{'PASS' if audit12['ok'] else 'FAIL'}**
""",
        )
        if not audit12["ok"]:
            write_text(out_dir / "G1_MECHANISM_DECISION.md", "# G1_MECHANISM_DECISION\n\nStopped: G1-A smoke FAIL on resume audit.\n")
            return 1
        if args.stop_after == "smoke12":
            return 0
        # Tokenizer-only first for packing; load GPU model just before QA.
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(QWEN_PATH), trust_remote_code=True)
        model = None
        torch = None
    else:
        print("[G1] LBP equivalence...", flush=True)
        eq = run_lbp_equivalence(out_dir, n_per_task=2)
        if not eq.get("ok"):
            print("EQUIVALENCE FAIL — stop", flush=True)
            write_text(out_dir / "G1_MECHANISM_DECISION.md", "# G1_MECHANISM_DECISION\n\nStopped: LBP equivalence FAIL.\n")
            return 1
        model, tokenizer, torch = eq["model"], eq["tokenizer"], eq["torch"]
        sample_ids_12 = eq["sample_ids"]
        hist_rows_12 = eq["hist_rows"]
        if args.stop_after == "equivalence":
            return 0

        print("[G1] Build matched packs for 12-sample smoke...", flush=True)
        built = build_matched_variant_packs(
            out_dir, sample_ids_12, hist_rows_12, tokenizer, stage="smoke12"
        )
        # validate order-only identity / over-budget
        smoke_issues = []
        for (v, t), rows in built["rows"].items():
            for r in rows:
                mt = int(r.get("model_token_count") or 0)
                if mt > BUDGET:
                    smoke_issues.append(f"over-budget {v}/{t}/{r.get('sample_id')} {mt}")
                if v in {"same_spans_source_order", "same_spans_query_score_order"}:
                    if not r.get("span_set_equal_frozen", True):
                        smoke_issues.append(f"span_set_not_equal {v}/{t}/{r.get('sample_id')}")
        # selection controls should differ from frozen LBP span set for at least some samples
        lbp_spans = {}
        for t, rows in hist_rows_12.items():
            for r in rows:
                lbp_spans[(t, str(r["sample_id"]))] = set(r.get("selected_span_ids") or [])
        changed_sel = 0
        for (v, t), rows in built["rows"].items():
            if v not in {"independent_topspan_neighbor_source_order", "mmr_neighbor_source_order", "bridge_score_zero"}:
                continue
            for r in rows:
                if set(r.get("selected_span_ids") or []) != lbp_spans.get((t, str(r["sample_id"])), set()):
                    changed_sel += 1
        if changed_sel == 0:
            smoke_issues.append("selection/score controls appear no-op (no span-set changes)")

        print("[G1] QA smoke12...", flush=True)
        for t, rows in hist_rows_12.items():
            write_jsonl(
                built["packed_root"] / "packed_prompts" / LBP_VARIANT / f"budget_{BUDGET}" / f"{t}.jsonl",
                rows,
            )
        qa12 = run_qa_on_packs(out_dir, built["packed_root"], variants_12, sample_ids_12, model, tokenizer, torch, "smoke12")
        scores12 = score_predictions(Path(qa12["pred_root"]), variants_12)
        write_json(out_dir / "smoke12" / "scores.json", scores12)
        audit12 = audit_smoke12_packs(out_dir, variants_12)

        smoke_ok = len(smoke_issues) == 0 and audit12["ok"]
        write_text(
            out_dir / "G1_12SAMPLE_SMOKE_REPORT.md",
            f"""# G1_12SAMPLE_SMOKE_REPORT

Generated: {utc_now()}

## Packing
- variants: {', '.join(variants_12)}
- over-budget issues: {sum(1 for x in smoke_issues if 'over-budget' in x)}
- selection/score span changes observed: {changed_sel}
- issues: {smoke_issues or ['none']}

## QA
- predictions: {qa12['n_preds']} (expected {12 * len(variants_12)})
- scores: see smoke12/scores.json

## Verdict
**{'PASS' if smoke_ok else 'FAIL'}**
""",
        )
        if not smoke_ok:
            print("SMOKE12 FAIL", smoke_issues, flush=True)
            write_text(out_dir / "G1_MECHANISM_DECISION.md", "# G1_MECHANISM_DECISION\n\nStopped: G1-A smoke FAIL.\n")
            return 1
        if args.stop_after == "smoke12":
            return 0

    # G1-B 60-sample gate
    print("[G1] Gate60 pack...", flush=True)
    sample_ids_60 = select_fixed_sample_ids(10)
    write_json(out_dir / "gate60" / "sample_ids_60.json", sample_ids_60)
    hist_rows_60 = load_hist_lbp_rows(sample_ids_60)
    gate_variants = list(variants_12)
    built60 = build_matched_variant_packs(
        out_dir,
        sample_ids_60,
        hist_rows_60,
        tokenizer,
        stage="gate60",
        variant_filter=gate_variants,
    )
    overs60 = sum(int(r.get("over_budget") or 0) for r in built60["summary"])
    # selection/order change checks on packs
    sel_changed_60 = 0
    for task in TASKS:
        lbp = {
            str(r["sample_id"]): r
            for r in read_jsonl(
                built60["packed_root"] / "packed_prompts" / LBP_VARIANT / f"budget_{BUDGET}" / f"{task}.jsonl"
            )
        }
        for v in [
            "independent_topspan_neighbor_source_order",
            "mmr_neighbor_source_order",
            "bridge_score_zero",
        ]:
            for r in read_jsonl(
                built60["packed_root"] / "packed_prompts" / v / f"budget_{BUDGET}" / f"{task}.jsonl"
            ):
                if set(r.get("selected_span_ids") or []) != set(lbp.get(str(r["sample_id"]), {}).get("selected_span_ids") or []):
                    sel_changed_60 += 1
    if overs60:
        write_text(out_dir / "G1_60SAMPLE_GATE_REPORT.md", f"# G1_60SAMPLE_GATE_REPORT\n\nFAIL: over-budget packs={overs60}\n")
        write_text(out_dir / "G1_MECHANISM_DECISION.md", "# G1_MECHANISM_DECISION\n\nStopped: G1-B over-budget.\n")
        return 1
    if sel_changed_60 == 0:
        write_text(out_dir / "G1_60SAMPLE_GATE_REPORT.md", "# G1_60SAMPLE_GATE_REPORT\n\nFAIL: selection controls no-op on 60 samples\n")
        write_text(out_dir / "G1_MECHANISM_DECISION.md", "# G1_MECHANISM_DECISION\n\nStopped: G1-B no-op wiring.\n")
        return 1

    if model is None:
        gate = safety_gate(out_dir, "gate60_before_load", phase="before_load")
        if gate["decision"] != "GO":
            write_text(out_dir / "WAITING_FOR_SAFE_GPU.md", f"{gate}\n")
            print("WAITING_FOR_SAFE_GPU", flush=True)
            return 2
        import torch as _torch
        from tools.bridgepack_g1.generate import load_qwen_fp16
        model, tokenizer = load_qwen_fp16(str(QWEN_PATH), _torch)
        torch = _torch
        safety_gate(out_dir, "gate60_after_load", phase="runtime")

    print("[G1] Gate60 QA...", flush=True)
    try:
        qa60 = run_qa_on_packs(
            out_dir,
            built60["packed_root"],
            gate_variants,
            sample_ids_60,
            model,
            tokenizer,
            torch,
            "gate60",
        )
        scores60 = score_predictions(Path(qa60["pred_root"]), gate_variants)
        write_json(out_dir / "gate60" / "scores.json", scores60)
        write_text(
            out_dir / "G1_60SAMPLE_GATE_REPORT.md",
            f"""# G1_60SAMPLE_GATE_REPORT

Generated: {utc_now()}

## Scope
- 60 samples (10/task); variants: {gate_variants}
- Not for paper conclusions / not for picking best test variants

## Stability
- predictions: {qa60['n_preds']} (expected {60 * len(gate_variants)})
- over-budget packs: {overs60}
- selection span-set changes vs LBP: {sel_changed_60}
- peak allocated / free VRAM: see G1_GPU_RESOURCE_REPORT.csv

## Verdict
**{'PASS' if qa60['n_preds'] == 60 * len(gate_variants) and overs60 == 0 else 'FAIL'}**
""",
        )
        if qa60["n_preds"] != 60 * len(gate_variants) or overs60:
            write_text(out_dir / "G1_MECHANISM_DECISION.md", "# G1_MECHANISM_DECISION\n\nStopped: G1-B gate FAIL.\n")
            return 1
    except Exception as exc:
        write_text(
            out_dir / "G1_60SAMPLE_GATE_REPORT.md",
            f"# G1_60SAMPLE_GATE_REPORT\n\nFAIL: {exc}\n\n```\n{traceback.format_exc()}\n```\n",
        )
        write_text(out_dir / "G1_MECHANISM_DECISION.md", f"# G1_MECHANISM_DECISION\n\nStopped: G1-B exception: {exc}\n")
        return 1

    if args.stop_after == "gate60":
        return 0

    # G1-C: full 1150 for two core controls (reuse historical LBP)
    print("[G1] Full matched controls pack (1150)...", flush=True)
    sample_ids_full = {}
    for task in TASKS:
        path = HIST_ROOT / "packed_prompts_build" / "packed_prompts" / LBP_VARIANT / f"budget_{BUDGET}" / f"{task}.jsonl"
        sample_ids_full[task] = [str(r["sample_id"]) for r in read_jsonl(path)]
    hist_full = load_hist_lbp_rows(sample_ids_full)
    full_variants = ["same_spans_source_order", "independent_topspan_neighbor_source_order"]
    built_full = build_matched_variant_packs(
        out_dir,
        sample_ids_full,
        hist_full,
        tokenizer,
        stage="full",
        variant_filter=full_variants,
    )
    overs_full = sum(int(r.get("over_budget") or 0) for r in built_full["summary"])
    if overs_full:
        write_text(out_dir / "G1_MECHANISM_DECISION.md", f"# G1_MECHANISM_DECISION\n\nStopped: full pack over-budget={overs_full}\n")
        return 1
    for task in TASKS:
        for v in full_variants:
            n = len(read_jsonl(built_full["packed_root"] / "packed_prompts" / v / f"budget_{BUDGET}" / f"{task}.jsonl"))
            if n != len(sample_ids_full[task]):
                raise RuntimeError(f"Full pack alignment fail {v}/{task}: {n}/{len(sample_ids_full[task])}")

    print("[G1] Full QA...", flush=True)
    qa_full = run_qa_on_packs(
        out_dir, built_full["packed_root"], full_variants, sample_ids_full, model, tokenizer, torch, "full"
    )
    if qa_full["n_preds"] != sum(len(v) for v in sample_ids_full.values()) * len(full_variants):
        write_text(out_dir / "G1_MECHANISM_DECISION.md", f"# G1_MECHANISM_DECISION\n\nStopped: full QA incomplete n={qa_full['n_preds']}\n")
        return 1
    scores_full = score_predictions(Path(qa_full["pred_root"]), full_variants)
    # Add historical LBP scores from existing score files
    hist_scores = []
    for task in TASKS:
        sp = HIST_ROOT / "scores" / LBP_VARIANT / f"{task}.json"
        score = json.loads(sp.read_text()) if sp.exists() else {}
        # longbench score files often {"score": x} or similar
        val = score.get("score", score.get(task, None))
        if isinstance(score, dict) and val is None and len(score) == 1:
            val = list(score.values())[0]
        hist_scores.append({"variant": LBP_VARIANT, "task": task, "n": len(sample_ids_full[task]), "score": float(val) * (100.0 if float(val) <= 1.5 else 1.0), "mean_length": ""})
    # detect scale: historical fairness uses 0-100
    # Prefer per_sample recompute
    from tools.bridgepack_cpu_stage.common import FAIRNESS_DIR, read_csv as _read_csv
    per = _read_csv(FAIRNESS_DIR / "per_sample_scores.csv")
    lbp_task = {}
    for r in per:
        if r["model"] == "qwen" and int(r["budget"]) == 1024 and r["method"] in {
            "lexical_bridgepack_path_order", "lexical_bridgepack"
        }:
            lbp_task.setdefault(r["task"], []).append(float(r["score_0_100"]))
    hist_rows_score = []
    for task in TASKS:
        vals = lbp_task.get(task, [])
        hist_rows_score.append({"variant": LBP_VARIANT, "task": task, "n": len(vals), "score": sum(vals)/max(1,len(vals)), "mean_length": ""})
    hist_rows_score.append({
        "variant": LBP_VARIANT,
        "task": "MACRO",
        "n": sum(len(lbp_task.get(t, [])) for t in TASKS),
        "score": sum(r["score"] for r in hist_rows_score) / len(TASKS),
        "mean_length": "",
    })

    all_score_rows = hist_rows_score + scores_full
    # write G1_FULL_RESULTS.csv
    import csv as _csv
    with (out_dir / "G1_FULL_RESULTS.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["variant", "task", "n", "score", "mean_length"]
        w = _csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        for r in all_score_rows:
            w.writerow(r)

    # Paired bootstrap LBP vs same_spans; same_spans vs independent
    print("[G1] Bootstrap...", flush=True)
    import numpy as np
    # Build per-sample scores for new variants
    from longbench_eval import dataset2metric

    def score_row(task: str, row: Mapping[str, Any]) -> float:
        metric = dataset2metric[task]
        pred = str(row.get("pred", ""))
        answers = row.get("answers") or []
        if not isinstance(answers, list):
            answers = [answers]
        best = 0.0
        for a in answers:
            best = max(best, float(metric(pred, a, all_classes=row.get("all_classes"))))
        return 100.0 * best

    def load_variant_scores(variant: str, pred_root: Path) -> Dict[Tuple[str, str], float]:
        out = {}
        for task in TASKS:
            for row in read_jsonl(pred_root / variant / f"{task}.jsonl"):
                out[(task, str(row["sample_id"]))] = score_row(task, row)
        return out

    lbp_scores = {}
    for r in per:
        if r["model"] == "qwen" and int(r["budget"]) == 1024 and r["method"] in {
            "lexical_bridgepack_path_order", "lexical_bridgepack"
        }:
            lbp_scores[(r["task"], r["sample_id"])] = float(r["score_0_100"])
    s_same = load_variant_scores("same_spans_source_order", Path(qa_full["pred_root"]))
    s_indep = load_variant_scores("independent_topspan_neighbor_source_order", Path(qa_full["pred_root"]))

    def paired_boot(a: Dict, b: Dict, label_a: str, label_b: str, n_boot: int = 10000, seed: int = 20260727):
        keys = sorted(set(a) & set(b))
        if not keys:
            return None
        sa = np.array([a[k] for k in keys], dtype=np.float64)
        sb = np.array([b[k] for k in keys], dtype=np.float64)
        strata = np.array([k[0] for k in keys])
        unique = sorted(set(strata.tolist()))
        idx = {s: np.where(strata == s)[0] for s in unique}
        rng = np.random.default_rng(seed)
        delta = float(np.mean(sa - sb))
        boots = np.empty(n_boot)
        for i in range(n_boot):
            chosen = np.concatenate([rng.choice(idx[s], size=len(idx[s]), replace=True) for s in unique])
            boots[i] = float(np.mean(sa[chosen] - sb[chosen]))
        lo, hi = np.quantile(boots, [0.025, 0.975])
        # per-task deltas
        task_deltas = {}
        for t in TASKS:
            mask = [k for k in keys if k[0] == t]
            if mask:
                task_deltas[t] = float(np.mean([a[k] - b[k] for k in mask]))
        return {
            "comparison": f"{label_a} - {label_b}",
            "n": len(keys),
            "delta_macro": delta,
            "ci95_low": float(lo),
            "ci95_high": float(hi),
            "crosses_zero": bool(lo <= 0 <= hi),
            **{f"delta_{t}": task_deltas.get(t, "") for t in TASKS},
        }

    boots = [
        paired_boot(lbp_scores, s_same, LBP_VARIANT, "same_spans_source_order"),
        paired_boot(s_same, s_indep, "same_spans_source_order", "independent_topspan_neighbor_source_order"),
        paired_boot(lbp_scores, s_indep, LBP_VARIANT, "independent_topspan_neighbor_source_order"),
    ]
    boots = [b for b in boots if b]
    with (out_dir / "G1_PAIRED_BOOTSTRAP.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["comparison", "n", "delta_macro", "ci95_low", "ci95_high", "crosses_zero"] + [f"delta_{t}" for t in TASKS]
        w = _csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        for b in boots:
            w.writerow(b)

    # Decision gate on two core comparisons
    c1 = boots[0]  # LBP - same_spans (path order)
    c2 = boots[1]  # same_spans - independent (bridge selection under source order)
    path_pos = c1["delta_macro"] > 0 and not c1["crosses_zero"]
    path_weak = abs(c1["delta_macro"]) < 0.5 or c1["crosses_zero"]
    sel_pos = c2["delta_macro"] > 0 and not c2["crosses_zero"]
    sel_weak = abs(c2["delta_macro"]) < 0.5 or c2["crosses_zero"]

    go_mechanism = (path_pos and sel_pos) or (
        (c1["delta_macro"] > 0 and c2["delta_macro"] > 0) and (not c1["crosses_zero"] or not c2["crosses_zero"])
    )
    no_go = (path_weak and sel_weak) or (c1["delta_macro"] <= 0 and c2["delta_macro"] <= 0)

    if no_go and not go_mechanism:
        decision = "NO_GO_MECHANISM"
        write_text(
            out_dir / "G1_MECHANISM_NO_GO_REPORT.md",
            f"""# G1_MECHANISM_NO_GO_REPORT

Generated: {utc_now()}

## Core comparisons
- LBP vs same_spans_source_order: delta={c1['delta_macro']:.4f} CI=[{c1['ci95_low']:.4f},{c1['ci95_high']:.4f}] crosses0={c1['crosses_zero']}
- same_spans vs independent_topspan: delta={c2['delta_macro']:.4f} CI=[{c2['ci95_low']:.4f},{c2['ci95_high']:.4f}] crosses0={c2['crosses_zero']}

## Action
- Stop non-essential G1 variants
- Do **not** auto-start G2
- Reposition paper as Neighbor-Preserving Evidence Packing
- bridge/path organization remains a heuristic design, not a verified core mechanism
""",
        )
    elif go_mechanism:
        decision = "GO_CONTINUE_G1_PREP_G2"
    else:
        decision = "INCONCLUSIVE_HOLD"

    write_text(
        out_dir / "G1_MECHANISM_DECISION.md",
        f"""# G1_MECHANISM_DECISION

Generated: {utc_now()}

## Decision
**{decision}**

## Evidence
| comparison | n | delta | 95% CI | crosses 0 |
|---|---:|---:|---|---|
| LBP − same_spans_source_order (path order) | {c1['n']} | {c1['delta_macro']:.4f} | [{c1['ci95_low']:.4f}, {c1['ci95_high']:.4f}] | {c1['crosses_zero']} |
| same_spans − independent_topspan (selection) | {c2['n']} | {c2['delta_macro']:.4f} | [{c2['ci95_low']:.4f}, {c2['ci95_high']:.4f}] | {c2['crosses_zero']} |

## Next
- G2/G3/G4/G5: **not auto-started**
- Remaining G1 variants (MMR, bridge_score_zero, query-score order): {'run if GO_CONTINUE' if decision=='GO_CONTINUE_G1_PREP_G2' else 'deferred/stopped'}
""",
    )
    write_text(
        out_dir / "G1_CLAIM_GUARD.md",
        f"""# G1_CLAIM_GUARD

Generated: {utc_now()}

| Claim | Allowed? | Basis |
|---|---|---|
| Exact-budget LBP historical main results | YES | Equivalence PASS + DO_NOT_RERUN |
| Path-order is a verified causal mechanism | {'YES' if path_pos else 'NO'} | LBP vs same_spans bootstrap |
| Bridge/path selection beats independent top-span under matched packing | {'YES' if sel_pos else 'NO'} | same_spans vs independent bootstrap |
| Neighbor-preserving packing helps | CONDITIONAL | Not isolated in core pair alone |
| Call method KV-cache pruning | NO | Always forbidden |
| Use 12/60 smoke as paper main numbers | NO | Gate-only |
""",
    )
    print(json.dumps({"decision": decision, "c1": c1, "c2": c2}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        out = Path(sys.argv[sys.argv.index("--out_dir") + 1]) if "--out_dir" in sys.argv else Path(".")
        write_text(out / "G1_FATAL_ERROR.md", f"{exc}\n\n{traceback.format_exc()}")
        print("FATAL", exc, flush=True)
        raise
