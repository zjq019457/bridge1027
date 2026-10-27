#!/usr/bin/env python3
"""Evidence recall / chain metrics from packed contexts (no LLM calls)."""
from __future__ import annotations

import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .common import (
    FAIRNESS_DIR,
    LONGBENCH_ROOT,
    MULTIHOP_TASKS,
    OUT_ROOT,
    REPO_ROOT,
    TASKS,
    ensure_cpu_only,
    normalize_answer,
    normalize_method,
    read_csv,
    read_jsonl,
    resolve_repo_path,
    utc_now,
    write_csv,
    write_json,
    write_text,
)


COMPARE_METHODS = [
    "bm25_chunk_source_order",
    "bge_chunk_packing",
    "bridge_neighbor32_path_order",
    "lexical_bridgepack_path_order",
    "bge_recall_bridgepack",
    "neighbor32_source_order",  # no-neighbor proxy if present; else span_only
    "span_only_source_order",
]


def _answer_in_text(answers: Sequence[Any], text: str) -> bool:
    hay = normalize_answer(text)
    for a in answers:
        needle = normalize_answer(str(a))
        if needle and needle in hay:
            return True
    return False


def _token_jaccard(a: str, b: str) -> float:
    ta = set(normalize_answer(a).split())
    tb = set(normalize_answer(b).split())
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _repetition_ratio(text: str) -> float:
    # Rough duplicate-span rate: fraction of whitespace tokens that appear >1 time.
    toks = normalize_answer(text).split()
    if not toks:
        return 0.0
    from collections import Counter

    c = Counter(toks)
    dup = sum(v for v in c.values() if v > 1)
    return dup / max(1, len(toks))


def _overlap_span_ratio(text: str, window: int = 8) -> float:
    toks = normalize_answer(text).split()
    if len(toks) < window * 2:
        return 0.0
    seen = set()
    overlap = 0
    total = 0
    for i in range(0, len(toks) - window + 1, window // 2):
        gram = tuple(toks[i : i + window])
        total += 1
        if gram in seen:
            overlap += 1
        else:
            seen.add(gram)
    return overlap / max(1, total)


def _load_dataset_answers() -> Dict[Tuple[str, str], Dict[str, Any]]:
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for task in MULTIHOP_TASKS:
        path = LONGBENCH_ROOT / f"{task}.jsonl"
        if not path.exists():
            continue
        for row in read_jsonl(path):
            sid = str(row.get("_id") or row.get("sample_id") or "")
            # BridgePack packing often hashes sample ids; also index by order later.
            out[(task, sid)] = row
    return out


def _iter_packed_for_method(method: str, budget: int = 1024, model: str = "qwen") -> List[Dict[str, Any]]:
    """Resolve packed artifacts via fairness manifest."""
    manifest = read_csv(FAIRNESS_DIR / "result_file_manifest.csv")
    rows = []
    for m in manifest:
        if m["model"] != model or int(m["budget"]) != budget:
            continue
        if normalize_method(m["method"]) != normalize_method(method) and m["method"] != method:
            continue
        packed = m["packed_path"]
        if "{task}" in packed:
            for task in MULTIHOP_TASKS:
                p = resolve_repo_path(packed.replace("{task}", task))
                if p.exists():
                    for row in read_jsonl(p):
                        row = dict(row)
                        row["_resolved_method"] = method
                        row["_resolved_budget"] = budget
                        row["_resolved_model"] = model
                        rows.append(row)
        else:
            p = resolve_repo_path(packed)
            if p.exists():
                for row in read_jsonl(p):
                    if row.get("task") not in MULTIHOP_TASKS:
                        continue
                    row = dict(row)
                    row["_resolved_method"] = method
                    row["_resolved_budget"] = budget
                    row["_resolved_model"] = model
                    rows.append(row)
        if rows:
            break
    return rows


def _f1_lookup() -> Dict[Tuple[str, str, str], float]:
    """(method_norm, task, sample_id) -> score for qwen/1024."""
    out = {}
    for r in read_csv(FAIRNESS_DIR / "per_sample_scores.csv"):
        if r["model"] != "qwen" or int(r["budget"]) != 1024:
            continue
        out[(normalize_method(r["method"]), r["task"], r["sample_id"])] = float(r["score_0_100"])
    return out


def run_evidence_metrics() -> Dict[str, Any]:
    ensure_cpu_only()
    dataset = _load_dataset_answers()
    f1_map = _f1_lookup()

    # Detect supporting-fact availability once
    sf_available = {}
    for task in MULTIHOP_TASKS:
        path = LONGBENCH_ROOT / f"{task}.jsonl"
        row = read_jsonl(path)[0] if path.exists() else {}
        sf_available[task] = any(k for k in row.keys() if "support" in k.lower())

    metric_rows = []
    neighbor_rows = []

    for method in COMPARE_METHODS:
        packed_rows = _iter_packed_for_method(method)
        if not packed_rows and method == "neighbor32_source_order":
            # try isolation run
            iso = REPO_ROOT / "results/bridgepack_neighbor_path_isolation_20260701_080809/packed_prompts_build/packed_prompts"
            if iso.exists():
                for task in MULTIHOP_TASKS:
                    for variant_dir in iso.glob("*neighbor*"):
                        for p in variant_dir.rglob(f"{task}.jsonl"):
                            for row in read_jsonl(p):
                                row = dict(row)
                                row["_resolved_method"] = variant_dir.name
                                row["_resolved_budget"] = 1024
                                row["_resolved_model"] = "qwen"
                                packed_rows.append(row)
                        break
        by_task: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        for row in packed_rows:
            task = str(row.get("task"))
            if task not in MULTIHOP_TASKS:
                continue
            text = str(row.get("packed_text") or row.get("packed_prompt") or row.get("generation_prompt") or "")
            answers = row.get("answers") or row.get("gold_analysis_only_answers") or []
            if not isinstance(answers, list):
                answers = [answers]
            # Prefer answers embedded in packed artifact (analysis-only); never used by selector.
            ans_recall = 1.0 if _answer_in_text(answers, text) else 0.0
            # Answer-bearing span recall: same as answer-string for LongBench without span offsets.
            ans_span_recall = ans_recall

            sf_recall = float("nan")
            hop1 = float("nan")
            hop2 = float("nan")
            complete_chain = float("nan")
            any_hop_missing = float("nan")
            if not sf_available.get(task, False):
                # Explicit NA policy
                pass
            else:
                # Would compute if annotations existed
                pass

            density = ans_recall  # without SF, evidence density ≈ answer presence
            rep = _repetition_ratio(text)
            overlap = _overlap_span_ratio(text)
            sid = str(row.get("sample_id") or "")
            f1 = f1_map.get((normalize_method(method), task, sid), float("nan"))
            # Also try hash/id mismatches: leave nan

            rec = {
                "model": "qwen",
                "budget": 1024,
                "method": normalize_method(method),
                "task": task,
                "sample_id": sid,
                "answer_string_recall": ans_recall,
                "answer_bearing_span_recall": ans_span_recall,
                "supporting_fact_recall": "NA" if not sf_available.get(task) else sf_recall,
                "hop1_evidence_recall": "NA" if not sf_available.get(task) else hop1,
                "hop2_evidence_recall": "NA" if not sf_available.get(task) else hop2,
                "complete_chain_recall": "NA" if not sf_available.get(task) else complete_chain,
                "any_hop_missing_rate": "NA" if not sf_available.get(task) else any_hop_missing,
                "effective_evidence_density": density,
                "repeat_token_ratio": rep,
                "overlap_span_ratio": overlap,
                "qa_f1": f1,
                "supporting_fact_annotation": "present" if sf_available.get(task) else "absent_in_longbench",
            }
            metric_rows.append(rec)
            by_task[task].append(rec)

            # Neighbor repair audit: compare lexical vs no-neighbor when both exist is done later
            if int(row.get("neighbor_size") or 0) > 0:
                # Approximate "neighbor-added" text unavailable without pre-repair spans;
                # use expanded provenance if present.
                expanded = row.get("expanded_records") or row.get("audit_expanded_records") or []
                neighbor_added_has_answer = float("nan")
                if expanded and isinstance(expanded, list):
                    # If records distinguish core vs expanded, check delta text.
                    hits = 0
                    tot = 0
                    for er in expanded:
                        if not isinstance(er, dict):
                            continue
                        core = str(er.get("core_text") or er.get("span_text") or "")
                        full = str(er.get("text") or "")
                        if not full:
                            continue
                        # crude: tokens in full not in core
                        added = full
                        if core and core in full:
                            added = full.replace(core, " ", 1)
                        tot += 1
                        if _answer_in_text(answers, added):
                            hits += 1
                    if tot:
                        neighbor_added_has_answer = hits / tot
                neighbor_rows.append(
                    {
                        "method": normalize_method(method),
                        "task": task,
                        "sample_id": sid,
                        "neighbor_size": row.get("neighbor_size"),
                        "neighbor_added_contains_answer_rate": neighbor_added_has_answer,
                        "neighbor_added_contains_sf_rate": "NA",
                        "note": "SF annotations absent in local LongBench jsonl",
                    }
                )

    write_csv(OUT_ROOT / "evidence" / "EVIDENCE_METRICS.csv", metric_rows)
    write_csv(OUT_ROOT / "EVIDENCE_METRICS.csv", metric_rows)

    # Aggregates + correlation with F1
    lines = [
        "# CHAIN_COMPLETENESS_REPORT",
        "",
        f"Generated: {utc_now()}",
        "",
        "## Supporting-fact annotation status",
        "",
    ]
    for task, ok in sf_available.items():
        lines.append(f"- `{task}`: {'PRESENT' if ok else 'ABSENT → hop/chain metrics = NA; answer-string recall only'}")

    lines.extend(["", "## Answer-string recall by method × task (mean)", "", "| method | task | answer_string_recall | n | corr(recall, F1) |", "|---|---|---:|---:|---:|"])

    by_mt: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in metric_rows:
        by_mt[(r["method"], r["task"])].append(r)
    for (method, task), items in sorted(by_mt.items()):
        recalls = np.array([float(x["answer_string_recall"]) for x in items], dtype=np.float64)
        f1s = np.array([float(x["qa_f1"]) for x in items if not (isinstance(x["qa_f1"], float) and math.isnan(x["qa_f1"]))], dtype=np.float64)
        # align
        pairs = [(float(x["answer_string_recall"]), float(x["qa_f1"])) for x in items if isinstance(x["qa_f1"], (int, float)) and not math.isnan(float(x["qa_f1"]))]
        if len(pairs) >= 3:
            a = np.array([p[0] for p in pairs])
            b = np.array([p[1] for p in pairs])
            if np.std(a) < 1e-12 or np.std(b) < 1e-12:
                corr = float("nan")
            else:
                corr = float(np.corrcoef(a, b)[0, 1])
        else:
            corr = float("nan")
        lines.append(f"| {method} | {task} | {recalls.mean():.4f} | {len(items)} | {corr if not (isinstance(corr,float) and math.isnan(corr)) else 'NA'} |")

    lines.extend(
        [
            "",
            "## Claim policy",
            "",
            "- May claim answer-string containment differences across packers on HotpotQA/2Wiki/MuSiQue.",
            "- Must NOT claim supporting-fact / hop-complete-chain improvements on this LongBench dump (annotations absent).",
            "- Must NOT invent gold evidence spans.",
            "",
        ]
    )
    write_text(OUT_ROOT / "evidence" / "CHAIN_COMPLETENESS_REPORT.md", "\n".join(lines))
    write_text(OUT_ROOT / "CHAIN_COMPLETENESS_REPORT.md", "\n".join(lines))

    nlines = [
        "# NEIGHBOR_REPAIR_EVIDENCE_AUDIT",
        "",
        f"Generated: {utc_now()}",
        "",
        "Neighbor-added answer containment is computed only when expanded_records distinguish core vs expanded text.",
        "Supporting-fact containment is NA (LongBench local files lack supporting_facts).",
        "",
        f"Rows audited: {len(neighbor_rows)}",
        "",
    ]
    # summarize finite rates
    finite = [r for r in neighbor_rows if isinstance(r["neighbor_added_contains_answer_rate"], float) and not math.isnan(r["neighbor_added_contains_answer_rate"])]
    if finite:
        by_m = defaultdict(list)
        for r in finite:
            by_m[r["method"]].append(r["neighbor_added_contains_answer_rate"])
        nlines.append("| method | mean neighbor-added answer hit rate | n |")
        nlines.append("|---|---:|---:|")
        for m, vals in sorted(by_m.items()):
            nlines.append(f"| {m} | {float(np.mean(vals)):.4f} | {len(vals)} |")
    else:
        nlines.append(
            "No expanded_records with separable neighbor-added text found in packed artifacts; "
            "metric marked unavailable for historical packs. New matched-ablation artifacts will record repaired ranges explicitly."
        )
    write_text(OUT_ROOT / "evidence" / "NEIGHBOR_REPAIR_EVIDENCE_AUDIT.md", "\n".join(nlines))
    write_text(OUT_ROOT / "NEIGHBOR_REPAIR_EVIDENCE_AUDIT.md", "\n".join(nlines))
    write_csv(OUT_ROOT / "evidence" / "neighbor_repair_rows.csv", neighbor_rows)
    return {"n_metric_rows": len(metric_rows), "sf_available": sf_available}


if __name__ == "__main__":
    print(run_evidence_metrics())
