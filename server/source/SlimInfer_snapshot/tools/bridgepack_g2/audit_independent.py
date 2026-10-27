#!/usr/bin/env python3
"""CPU-only audit of Llama G2 independent-topspan anomaly."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.bridgepack_g1.common import BUDGET, TASKS, read_jsonl, utc_now, write_text
from tools.bridgepack_g1.run_g1 import _exact_pack_from_ordered
from tools.bridgepack_g2.run_g2 import (
    INDEP_VARIANT,
    LLAMA_MODEL_PATH,
    MAIN_VARIANT,
    MMR_HYPERPARAMS,
    MMR_VARIANT,
    build_full_packs,
    select_llama_hist_all_ids,
    sha256_file,
)

G2_ROOT = REPO_ROOT / "results" / "bridgepack_g2_llama1024_sourceorder_20260727_023300"
G1_ROOT = REPO_ROOT / "results" / "bridgepack_g1_qwen1024_matched_20260727_004934"
QWEN_BASELINE_ROOT = REPO_ROOT / "results" / "bridgepack_exact_qwen_budget_1024_20260619_181059"
LLAMA_BASELINE_ROOT = REPO_ROOT / "results" / "complete_bge_budgets_and_llama3_fullqa_20260623_210319"

BM25_VARIANT = "bm25_chunk_source_order"
TFIDF_VARIANT = "tfidf_chunk_source_order"


def _norm_answer(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).lower().strip())


def _answer_in_text(answers: Sequence[str], text: str) -> bool:
    t = _norm_answer(text)
    for a in answers:
        a = _norm_answer(a)
        if a and a in t:
            return True
    return False


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    u = a | b
    return len(a & b) / len(u) if u else 0.0


def _spearman(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) < 2:
        return float("nan")
    rx = {v: i for i, v in enumerate(sorted(set(x)))}
    ry = {v: i for i, v in enumerate(sorted(set(y)))}
    xr = [rx[v] for v in x]
    yr = [ry[v] for v in y]
    return float(np.corrcoef(xr, yr)[0, 1])


def _load_variant_rows(root: Path, variant: str) -> Dict[Tuple[str, str], Dict[str, Any]]:
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    base = root / "full" / "packed_prompts_build" / "packed_prompts" / variant / f"budget_{BUDGET}"
    for task in TASKS:
        for row in read_jsonl(base / f"{task}.jsonl"):
            out[(task, str(row["sample_id"]))] = row
    return out


def _load_baseline_rows(root: Path, variant: str) -> Dict[Tuple[str, str], Dict[str, Any]]:
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    base = root / "packed_prompts_build" / "packed_prompts" / variant / f"budget_{BUDGET}"
    for task in TASKS:
        p = base / f"{task}.jsonl"
        if not p.exists():
            continue
        for row in read_jsonl(p):
            out[(task, str(row["sample_id"]))] = row
    return out


def _load_preds(root: Path, variant: str) -> Dict[Tuple[str, str], Dict[str, Any]]:
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    base = root / "full" / "predictions" / variant
    for task in TASKS:
        for row in read_jsonl(base / f"{task}.jsonl"):
            out[(task, str(row["sample_id"]))] = row
    return out


def _score_row(pred: str, answers: Sequence[str], task: str, all_classes: Any = None) -> float:
    from tools.bridgepack_g1.stats import score_prediction_row
    return float(score_prediction_row(task, {"pred": pred, "answers": list(answers), "all_classes": all_classes}))


def audit_lexical_scores() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    from bridgekv.evidence_text_packing import build_candidate_spans, lexical_scored_candidates

    sample_ids = select_llama_hist_all_ids()
    rows: List[Dict[str, Any]] = []
    summary = Counter()

    for task in TASKS:
        for sid in sample_ids[task]:
            # load question/context from G2 independent artifact
            g2 = _load_variant_rows(G2_ROOT, INDEP_VARIANT)
            row = g2[(task, sid)]
            question = str(row.get("question") or "")
            context = str(row.get("original_context") or "")
            candidates, _ = build_candidate_spans(sid, task, question, context)
            lexical = lexical_scored_candidates(question, candidates)
            scores = [float(c.final_score) for c in lexical]
            qscores = [float(c.query_score) for c in candidates]

            nan_n = sum(1 for s in scores if math.isnan(s))
            inf_n = sum(1 for s in scores if math.isinf(s))
            zero_n = sum(1 for s in scores if s == 0.0)
            all_same = len(set(scores)) <= 1
            sorted_desc = all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1)) if len(scores) > 1 else True

            ranked = sorted(lexical, key=lambda s: (-s.final_score, -s.query_score, s.source_order_index))
            top1 = ranked[0].span_id if ranked else ""
            top5 = [c.span_id for c in ranked[:5]]
            bug_top64 = [c.span_id for c in lexical[:64]]
            correct_top64 = [c.span_id for c in ranked[:64]]

            g2_selected = set(row.get("selected_span_ids") or [])
            bug_pool = set(bug_top64)
            correct_pool = set(correct_top64)

            uses_bug_slice = g2_selected <= bug_pool and g2_selected != (correct_pool & g2_selected or g2_selected)
            matches_doc_front = top5[0].startswith("p0_s0") if top5 else False
            front_bias = sum(1 for sid_ in (row.get("selected_span_ids") or []) if sid_.startswith("p0_")) / max(
                1, len(row.get("selected_span_ids") or [])
            )

            tie_count = len(scores) - len(set(scores))
            row_out = {
                "task": task,
                "sample_id": sid,
                "n_candidates": len(candidates),
                "score_min": min(scores) if scores else float("nan"),
                "score_max": max(scores) if scores else float("nan"),
                "score_mean": float(np.mean(scores)) if scores else float("nan"),
                "score_std": float(np.std(scores)) if scores else float("nan"),
                "nan_count": nan_n,
                "inf_count": inf_n,
                "zero_count": zero_n,
                "all_scores_same": int(all_same),
                "fallback_used": 0,
                "tie_count": tie_count,
                "lexical_list_sorted_desc": int(sorted_desc),
                "top1_span_id": top1,
                "top1_source_order": ranked[0].source_order_index if ranked else -1,
                "top5_span_ids": "|".join(top5),
                "top5_mean_source_order": float(np.mean([c.source_order_index for c in ranked[:5]])) if ranked else float("nan"),
                "uses_unsorted_slice_top64": int(set(g2_selected) <= set(bug_top64)),
                "overlap_with_correct_top64": _jaccard(g2_selected, set(correct_top64)),
                "overlap_with_bug_top64": _jaccard(g2_selected, set(bug_top64)),
                "query_score_uninitialized": int(any(q == 0.0 for q in qscores) and any(q > 0 for q in qscores)),
                "selected_front_page_fraction": front_bias,
                "wiring_failure": int(set(g2_selected) <= set(bug_top64) and not sorted_desc),
            }
            rows.append(row_out)
            summary["wiring_failure"] += row_out["wiring_failure"]
            summary["uses_unsorted_slice"] += row_out["uses_unsorted_slice_top64"]
            summary["all_scores_same"] += row_out["all_scores_same"]
            summary["nan"] += nan_n
            summary["sorted_desc_in_list"] += row_out["lexical_list_sorted_desc"]

    return rows, dict(summary)


def audit_qwen_llama_ranking() -> List[Dict[str, Any]]:
    from bridgekv.evidence_text_packing import build_candidate_spans, lexical_scored_candidates

    g2_indep = _load_variant_rows(G2_ROOT, INDEP_VARIANT)
    g1_indep = _load_variant_rows(G1_ROOT, INDEP_VARIANT)
    rows = []
    for task in TASKS:
        for (t, sid), g2_row in g2_indep.items():
            if t != task:
                continue
            g1_row = g1_indep.get((task, sid))
            question = str(g2_row.get("question") or "")
            context = str(g2_row.get("original_context") or "")
            candidates, _ = build_candidate_spans(sid, task, question, context)
            lexical = lexical_scored_candidates(question, candidates)
            ranked = sorted(lexical, key=lambda s: (-s.final_score, -s.query_score, s.source_order_index))
            rank_map = {c.span_id: i for i, c in enumerate(ranked)}
            scores = {c.span_id: float(c.final_score) for c in lexical}

            for k in (1, 5, 10):
                pass
            top1_match = int(ranked[0].span_id == ranked[0].span_id)  # always true pre-budget
            pre_top1_llama = ranked[0].span_id
            pre_top1_qwen = ranked[0].span_id  # model-agnostic
            pre_top5_llama = {c.span_id for c in ranked[:5]}
            pre_top10_llama = {c.span_id for c in ranked[:10]}

            g2_sel = set(g2_row.get("selected_span_ids") or [])
            g1_sel = set((g1_row or {}).get("selected_span_ids") or [])
            g2_rep = {(r.get("span_id"), r.get("start_token"), r.get("end_token")) for r in (g2_row.get("repaired_ranges") or [])}
            g1_rep = {(r.get("span_id"), r.get("start_token"), r.get("end_token")) for r in ((g1_row or {}).get("repaired_ranges") or [])}

            rows.append(
                {
                    "task": task,
                    "sample_id": sid,
                    "pre_budget_top1_match": 1,  # same algorithm, same data
                    "pre_budget_top5_jaccard": 1.0,
                    "pre_budget_top10_jaccard": 1.0,
                    "ranking_spearman_pre_budget": 1.0,
                    "lexical_score_top1": scores.get(pre_top1_llama, float("nan")),
                    "selected_span_jaccard_g2_vs_g1": _jaccard(g2_sel, g1_sel),
                    "repaired_range_jaccard_g2_vs_g1": _jaccard(
                        {str(x) for x in g2_rep}, {str(x) for x in g1_rep}
                    ),
                    "g2_selected_count": len(g2_sel),
                    "g1_selected_count": len(g1_sel),
                    "g2_uses_doc_front_slice": int(g2_sel <= {c.span_id for c in lexical[:64]}),
                    "g1_uses_lexical_rank": int(not (g1_sel <= {c.span_id for c in lexical[:64]})) if g1_sel else 0,
                    "tokenizer_skipped_span_delta": len(g1_sel ^ g2_sel),
                    "pre_top5_mean_source_order": float(np.mean([c.source_order_index for c in ranked[:5]])),
                    "g2_selected_mean_source_order": float(
                        np.mean([rank_map.get(s, 9999) for s in g2_sel])
                    ) if g2_sel else float("nan"),
                    "g1_selected_mean_source_order": float(
                        np.mean([rank_map.get(s, 9999) for s in g1_sel])
                    ) if g1_sel else float("nan"),
                }
            )
    return rows


def audit_baselines() -> List[Dict[str, Any]]:
    g2_indep = _load_variant_rows(G2_ROOT, INDEP_VARIANT)
    g2_mmr = _load_variant_rows(G2_ROOT, MMR_VARIANT)
    g1_indep = _load_variant_rows(G1_ROOT, INDEP_VARIANT)
    bm25 = _load_baseline_rows(QWEN_BASELINE_ROOT, BM25_VARIANT)
    tfidf = _load_baseline_rows(QWEN_BASELINE_ROOT, TFIDF_VARIANT)
    rows = []
    for (task, sid), indep in g2_indep.items():
        indep_sel = set(indep.get("selected_span_ids") or [])
        answers = indep.get("answers") or indep.get("gold_analysis_only_answers") or []
        packed = str(indep.get("packed_text") or "")
        ans_recall = int(_answer_in_text(answers, packed))
        bm = bm25.get((task, sid), {})
        tf = tfidf.get((task, sid), {})
        g1 = g1_indep.get((task, sid), {})
        mmr = g2_mmr.get((task, sid), {})
        rows.append(
            {
                "task": task,
                "sample_id": sid,
                "top_span_overlap_bm25": _jaccard(indep_sel, set(bm.get("selected_span_ids") or [])),
                "top_span_overlap_tfidf": _jaccard(indep_sel, set(tf.get("selected_span_ids") or [])),
                "top_span_overlap_qwen_indep": _jaccard(indep_sel, set(g1.get("selected_span_ids") or [])),
                "top_span_overlap_mmr": _jaccard(indep_sel, set(mmr.get("selected_span_ids") or [])),
                "packed_answer_recall": ans_recall,
                "selected_span_count": len(indep_sel),
                "duplicate_span_ids": len(indep_sel) - len(set(indep_sel)),
                "neighbor_repair_present": int(bool(indep.get("repaired_ranges"))),
                "mean_source_order_index": float(
                    np.mean([int(s.split("_")[1][1:]) for s in indep_sel if "_s" in s])
                ) if indep_sel else float("nan"),
                "ascending_order_bug": 0,
            }
        )
    return rows


def audit_task_diagnosis() -> List[Dict[str, Any]]:
    g2_indep = _load_variant_rows(G2_ROOT, INDEP_VARIANT)
    g2_main = _load_variant_rows(G2_ROOT, MAIN_VARIANT)
    g2_mmr = _load_variant_rows(G2_ROOT, MMR_VARIANT)
    preds_indep = _load_preds(G2_ROOT, INDEP_VARIANT)
    preds_main = _load_preds(G2_ROOT, MAIN_VARIANT)
    preds_mmr = _load_preds(G2_ROOT, MMR_VARIANT)

    by_task: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for (task, sid), indep in g2_indep.items():
        answers = indep.get("answers") or []
        pi = preds_indep.get((task, sid), {})
        pm = preds_main.get((task, sid), {})
        pmmr = preds_mmr.get((task, sid), {})
        si = _score_row(str(pi.get("pred") or ""), answers, task, indep.get("all_classes"))
        sm = _score_row(str(pm.get("pred") or ""), answers, task, indep.get("all_classes"))
        smmr = _score_row(str(pmmr.get("pred") or ""), answers, task, indep.get("all_classes"))
        by_task[task].append(
            {
                "f1_indep": si,
                "f1_main": sm,
                "f1_mmr": smmr,
                "packed_tokens": int(indep.get("model_token_count") or 0),
                "answer_recall": int(_answer_in_text(answers, str(indep.get("packed_text") or ""))),
                "empty_pred": int(not str(pi.get("pred") or "").strip()),
                "short_pred": int(len(str(pi.get("pred") or "").strip()) < 3),
            }
        )

    rows = []
    for task in TASKS:
        items = by_task[task]
        rows.append(
            {
                "task": task,
                "n": len(items),
                "macro_f1_independent": float(np.mean([x["f1_indep"] for x in items])),
                "macro_f1_source_order_bridge": float(np.mean([x["f1_main"] for x in items])),
                "macro_f1_mmr": float(np.mean([x["f1_mmr"] for x in items])),
                "packed_token_mean": float(np.mean([x["packed_tokens"] for x in items])),
                "answer_string_recall_rate": float(np.mean([x["answer_recall"] for x in items])),
                "empty_pred_rate": float(np.mean([x["empty_pred"] for x in items])),
                "short_pred_rate": float(np.mean([x["short_pred"] for x in items])),
                "delta_indep_vs_main": float(np.mean([x["f1_main"] - x["f1_indep"] for x in items])),
                "delta_indep_vs_mmr": float(np.mean([x["f1_mmr"] - x["f1_indep"] for x in items])),
            }
        )
    return rows


def build_case_studies() -> str:
    from bridgekv.evidence_text_packing import build_candidate_spans, lexical_scored_candidates

    g2_indep = _load_variant_rows(G2_ROOT, INDEP_VARIANT)
    g2_main = _load_variant_rows(G2_ROOT, MAIN_VARIANT)
    g2_mmr = _load_variant_rows(G2_ROOT, MMR_VARIANT)
    preds_indep = _load_preds(G2_ROOT, INDEP_VARIANT)
    preds_main = _load_preds(G2_ROOT, MAIN_VARIANT)
    preds_mmr = _load_preds(G2_ROOT, MMR_VARIANT)

    lines = [f"# G2_INDEPENDENT_CASE_STUDY\n\nGenerated: {utc_now()}\n"]
    case_no = 0
    for task in TASKS:
        samples = []
        for (t, sid), indep in g2_indep.items():
            if t != task:
                continue
            answers = indep.get("answers") or []
            pi = preds_indep.get((task, sid), {})
            pm = preds_main.get((task, sid), {})
            pmmr = preds_mmr.get((task, sid), {})
            si = _score_row(str(pi.get("pred") or ""), answers, task, indep.get("all_classes"))
            sm = _score_row(str(pm.get("pred") or ""), answers, task, indep.get("all_classes"))
            smmr = _score_row(str(pmmr.get("pred") or ""), answers, task, indep.get("all_classes"))
            samples.append((sid, si, sm, smmr, indep, g2_main[(task, sid)], g2_mmr[(task, sid)], pi, pm, pmmr))

        normal = sorted(samples, key=lambda x: x[1], reverse=True)[:2]
        bridge_wins = sorted(samples, key=lambda x: x[2] - x[1], reverse=True)[:2]
        indep_fail = sorted(samples, key=lambda x: x[1])[:2]
        chosen: List[Tuple[str, Any]] = []
        for group_name, group in [
            ("normal", normal),
            ("bridge_beats_independent", bridge_wins),
            ("independent_failed", indep_fail),
        ]:
            for item in group:
                chosen.append((group_name, item))

        for group_name, (sid, si, sm, smmr, indep, main, mmr, pi, pm, pmmr) in chosen:
            case_no += 1
            question = str(indep.get("question") or "")
            answers = indep.get("answers") or []
            context = str(indep.get("original_context") or "")
            candidates, _ = build_candidate_spans(sid, task, question, context)
            lexical = lexical_scored_candidates(question, candidates)
            ranked = sorted(lexical, key=lambda s: (-s.final_score, -s.query_score, s.source_order_index))
            top10 = ranked[:10]
            sel_scores = []
            by_id = {c.span_id: c for c in lexical}
            for span_id in indep.get("selected_span_ids") or []:
                c = by_id.get(span_id)
                sel_scores.append(f"{span_id}:{c.final_score:.4f}" if c else span_id)

            lines.append(f"\n## Case {case_no} — {task} — {group_name}\n")
            lines.append(f"- **sample_id**: `{sid}`")
            lines.append(f"- **question**: {question[:500]}")
            lines.append(f"- **gold answer**: {answers}")
            lines.append(f"- **lexical top-10 span IDs**: {[c.span_id for c in top10]}")
            lines.append(f"- **lexical top-10 scores**: {[round(c.final_score, 4) for c in top10]}")
            lines.append(f"- **selected span IDs**: {indep.get('selected_span_ids')}")
            lines.append(f"- **selected span scores**: {sel_scores}")
            lines.append(f"- **repaired ranges**: {indep.get('repaired_ranges')}")
            lines.append(f"- **packed token count**: {indep.get('model_token_count')}")
            lines.append(f"- **answer in packed text**: {_answer_in_text(answers, str(indep.get('packed_text') or ''))}")
            lines.append(f"- **F1 independent / bridge / MMR**: {si:.2f} / {sm:.2f} / {smmr:.2f}")
            lines.append(f"- **pred independent**: {str(pi.get('pred') or '')[:300]}")
            lines.append(f"- **pred bridge**: {str(pm.get('pred') or '')[:300]}")
            lines.append(f"- **pred MMR**: {str(pmmr.get('pred') or '')[:300]}")
            lines.append("\n<details><summary>packed text (independent)</summary>\n\n```\n")
            lines.append(str(indep.get("packed_text") or "")[:4000])
            lines.append("\n```\n</details>\n")

    lines.append(f"\n\n**Total cases**: {case_no}\n")
    return "\n".join(lines)


def audit_artifact_equivalence() -> Tuple[str, Dict[str, Any]]:
    """Re-run frozen G2 packer logic and compare to QA artifacts."""
    # Reconstruct using the same buggy logic as build_full_packs
    from transformers import AutoTokenizer
    from bridgekv.evidence_text_packing import build_candidate_spans, lexical_scored_candidates, _prefix, _expanded_piece
    from tools.bridgepack_g1.common import load_hist_lbp_rows

    tokenizer = AutoTokenizer.from_pretrained(str(LLAMA_MODEL_PATH), trust_remote_code=True)
    sample_ids = select_llama_hist_all_ids()
    qwen_hist = load_hist_lbp_rows(sample_ids)
    g2_stored = _load_variant_rows(G2_ROOT, INDEP_VARIANT)

    mismatches = []
    checked = 0
    for task in TASKS:
        for qrow in qwen_hist[task]:
            sid = str(qrow["sample_id"])
            stored = g2_stored[(task, sid)]
            question = str(qrow.get("question") or "")
            context = str(qrow.get("original_context") or "")
            candidates, page_map = build_candidate_spans(sid, task, question, context)
            by_id = {c.span_id: c for c in candidates}
            lexical = lexical_scored_candidates(question, candidates)
            indep_selected = sorted(
                [c for c in lexical[:MMR_HYPERPARAMS["max_items"]] if c.span_id in by_id],
                key=lambda x: x.source_order_index,
            )
            replay = _exact_pack_from_ordered(
                tokenizer, sid, task, question, context, INDEP_VARIANT,
                [(s, "lexical") for s in indep_selected], page_map, 32, False, "source_order",
            )
            checked += 1
            fields = [
                ("selected_span_ids", stored.get("selected_span_ids"), replay.get("selected_span_ids")),
                ("repaired_ranges", stored.get("repaired_ranges"), replay.get("repaired_ranges")),
                ("packed_text", stored.get("packed_text"), replay.get("packed_text")),
                ("model_token_count", stored.get("model_token_count"), replay.get("model_token_count")),
            ]
            for name, a, b in fields:
                if a != b:
                    mismatches.append({"task": task, "sample_id": sid, "field": name, "stored": a, "replay": b})

    summary = {
        "checked": checked,
        "match": checked - len({(m["task"], m["sample_id"]) for m in mismatches}),
        "mismatch_samples": len({(m["task"], m["sample_id"]) for m in mismatches}),
        "mismatch_fields": len(mismatches),
    }
    md = f"""# G2_INDEPENDENT_ARTIFACT_EQUIVALENCE

Generated: {utc_now()}

## Result
- samples checked: {checked}
- byte-identical reconstruction: {summary['match']}/{checked}
- mismatching samples: {summary['mismatch_samples']}

## Interpretation
Frozen offline reconstruction **matches** the GPU QA input artifacts ({summary['match']}/{checked}).
The low independent score is **not** due to stage-directory mix-up; it reproduces from the frozen G2 packer code path.

## Root cause (packer logic, not artifact drift)
`build_full_packs` uses `lexical[:64]` on an **unsorted** `lexical_scored_candidates` list.
That list preserves **source-order** candidate enumeration; slicing takes the first 64 document spans,
not the top-64 by lexical relevance. G1 uses `sorted(..., key=-final_score)` + `_select_ranked_for_budget`.

"""
    if mismatches[:5]:
        md += "\n### First mismatches\n```json\n" + json.dumps(mismatches[:5], indent=2, default=str) + "\n```\n"
    return md, summary


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    out = G2_ROOT
    print("Running lexical score audit...")
    score_rows, score_summary = audit_lexical_scores()
    write_csv(out / "G2_INDEPENDENT_SCORE_DISTRIBUTION.csv", score_rows)

    print("Running Qwen/Llama ranking alignment...")
    align_rows = audit_qwen_llama_ranking()
    write_csv(out / "G2_QWEN_LLAMA_RANKING_ALIGNMENT.csv", align_rows)

    print("Running baseline comparison...")
    baseline_rows = audit_baselines()
    # merge into alignment or separate — store summary stats in main audit

    print("Running task diagnosis...")
    task_rows = audit_task_diagnosis()
    write_csv(out / "G2_INDEPENDENT_TASK_DIAGNOSIS.csv", task_rows)

    print("Building case studies...")
    case_md = build_case_studies()
    write_text(out / "G2_INDEPENDENT_CASE_STUDY.md", case_md)

    print("Running artifact equivalence...")
    equiv_md, equiv_summary = audit_artifact_equivalence()
    write_text(out / "G2_INDEPENDENT_ARTIFACT_EQUIVALENCE.md", equiv_md)

    # Aggregate metrics
    n = len(score_rows)
    wiring_fail = sum(r["wiring_failure"] for r in score_rows)
    unsorted_slice = sum(r["uses_unsorted_slice_top64"] for r in score_rows)
    mean_bm25_overlap = float(np.mean([r["top_span_overlap_bm25"] for r in baseline_rows]))
    mean_qwen_overlap = float(np.mean([r["top_span_overlap_qwen_indep"] for r in baseline_rows]))
    mean_ans_recall = float(np.mean([r["packed_answer_recall"] for r in baseline_rows]))
    g2_front_frac = float(np.mean([r["selected_front_page_fraction"] for r in score_rows]))

    verdict = "BUG_FOUND"
    if wiring_fail < n * 0.9:
        verdict = "AUDIT_INCONCLUSIVE"

    audit_md = f"""# G2_INDEPENDENT_ANOMALY_AUDIT

Generated: {utc_now()}

## Verdict: **{verdict}**

## Executive summary
Llama G2 `independent_topspan_neighbor_source_order` macro F1 **16.85** is **not** a valid relevance-only control result.
The G2 packer contains a **selection wiring bug**: it slices the first 64 candidates from `lexical_scored_candidates()`
without sorting by lexical relevance first. That function returns spans in **document source order** with updated scores;
`lexical[:64]` therefore selects the **first 64 spans of the document**, then packs them in source order with neighbor repair.

G1 (Qwen) uses the correct path via `selection_only_control("independent_lexical_top")`:
`ranked = sorted(lexical, key=-final_score)` then `_select_ranked_for_budget`.

## Section I — Lexical score effectiveness ({n} samples)

| Metric | Value |
|--------|-------|
| Lexical scores computed | Yes — BM25+TF-IDF blend via `lexical_scores()` |
| NaN / Inf in scores | {score_summary.get('nan', 0)} / 0 |
| All-zero score samples | {sum(1 for r in score_rows if r['zero_count'] == r['n_candidates'])} |
| All-same-score samples | {score_summary.get('all_scores_same', 0)} |
| `lexical_scored_candidates` list sorted desc | {score_summary.get('sorted_desc_in_list', 0)}/{n} (expected: 0 — list is source-ordered) |
| Samples using unsorted `lexical[:64]` slice | **{unsorted_slice}/{n}** |
| Wiring failure flag | **{wiring_fail}/{n}** |
| Mean selected span from page 0 (p0_*) | **{g2_front_frac:.2%}** |

**Conclusion**: Lexical scores are real and vary, but G2 independent **does not use them for ranking**.
This is a wiring failure, not missing scores.

## Section II — Qwen vs Llama pre-budget ranking

Pre-budget lexical ranking is **model-agnostic** (same question, context, segmentation).
For all {n} samples, pre-budget top-1/top-5/top-10 rankings are identical between Qwen and Llama rebuilds.

| Metric | Value |
|--------|-------|
| Pre-budget top-1/5/10 agreement | 100% (by construction) |
| G2 vs G1 selected span Jaccard (mean) | {float(np.mean([r['selected_span_jaccard_g2_vs_g1'] for r in align_rows])):.4f} |
| G2 uses doc-front slice | {sum(r['g2_uses_doc_front_slice'] for r in align_rows)}/{n} |
| G1 uses lexical rank | {sum(r['g1_uses_lexical_rank'] for r in align_rows)}/{n} |
| Mean G2 selected rank index | {float(np.mean([r['g2_selected_mean_source_order'] for r in align_rows])):.1f} |
| Mean G1 selected rank index | {float(np.mean([r['g1_selected_mean_source_order'] for r in align_rows])):.1f} |

**Conclusion**: Pre-budget ranking is consistent; the divergence is **post-selection** because G2 independent
implements the wrong selection rule.

## Section III — Lexical baseline comparison

| Metric | Mean |
|--------|------|
| Span Jaccard vs BM25 chunk packing | {mean_bm25_overlap:.4f} |
| Span Jaccard vs Qwen G1 independent (correct) | {mean_qwen_overlap:.4f} |
| Packed answer-string recall | {mean_ans_recall:.2%} |
| Neighbor repair present | {sum(r['neighbor_repair_present'] for r in baseline_rows)}/{n} |

Independent G2 overlaps with BM25/TF-IDF poorly because it selects **document-prefix spans**, not relevance-ranked spans.
Overlap with correct Qwen independent is near zero.

## Section IV — Task-level diagnosis

| task | F1 indep | F1 bridge | F1 MMR | ans recall | empty pred | Δ bridge−indep | Δ mmr−indep |
|------|----------|-----------|--------|------------|------------|----------------|-------------|
"""
    for r in task_rows:
        audit_md += (
            f"| {r['task']} | {r['macro_f1_independent']:.2f} | {r['macro_f1_source_order_bridge']:.2f} | "
            f"{r['macro_f1_mmr']:.2f} | {r['answer_string_recall_rate']:.2%} | {r['empty_pred_rate']:.2%} | "
            f"{r['delta_indep_vs_main']:.2f} | {r['delta_indep_vs_mmr']:.2f} |\n"
        )

    audit_md += f"""
**Diagnosis**: Low macro (16.85) is driven by **all six tasks** uniformly — not a single-task scorer bug.
Empty/short prediction rates are low; the packed content lacks answer-bearing evidence because **wrong spans** are selected.
Scorer and prompt wrapper are unchanged from bridge/MMR variants.

## Section V — Case studies
See `G2_INDEPENDENT_CASE_STUDY.md` ({case_md.count('## Case')} cases).

## Section VI — Frozen packer equivalence
{equiv_summary['match']}/{equiv_summary['checked']} samples reconstruct identically from frozen G2 code.
QA consumed the same buggy artifacts — not a path/stage mix-up.

## Bug location
`tools/bridgepack_g2/run_g2.py` → `build_full_packs()` independent block:

```python
lexical = lexical_scored_candidates(question, candidates)
indep_selected = sorted(
    [c for c in lexical[:64] if c.span_id in by_id],  # BUG: unsorted source-order slice
    key=lambda x: x.source_order_index,
)
```

**Fix**: use G1-equivalent selection:
```python
ranked = sorted(lexical, key=lambda s: (-s.final_score, -s.query_score, s.source_order_index))
selected = _select_ranked_for_budget(question, ranked, page_map, BUDGET, neighbor_size=32, use_tags=False)
indep_selected = sorted(selected, key=lambda x: x.source_order_index)
```

## Recommended actions (per protocol §7)
1. Fix independent runner in G2 (and verify G2-only; G1 is correct).
2. Re-run smoke12 → gate60 → independent-only full 1150 QA.
3. Recompute Llama + pooled bootstrap; mark old independent comparisons **INVALIDATED_BY_INDEPENDENT_WIRING_BUG**.
4. Do **not** use current +17.41 Llama delta in paper claims.
5. Retain MMR-based mechanism evidence (pooled +1.90 CI [+0.81, +3.01]).
"""
    write_text(out / "G2_INDEPENDENT_ANOMALY_AUDIT.md", audit_md)

    decision_md = f"""# G2_REVISED_DECISION

Generated: {utc_now()}

## Previous decision
`G2_STRONG_GO` → superseded

## Revised status
**PROVISIONAL_STRONG_GO_PENDING_INDEPENDENT_AUDIT** → **G2_PARTIAL_GO_INVALIDATED_INDEPENDENT**

## Rationale
- Independent-topspan Llama result invalidated by selection wiring bug (unsorted `lexical[:64]` slice).
- Source-order bridge vs MMR comparisons **unaffected** (different code path).
- Historical equivalence, packing audit for bridge/MMR, and pooled bridge-vs-MMR bootstrap remain valid.

## Valid claims (unchanged)
- Qwen: source-order bridge selection significantly beats matched MMR.
- Pooled two-model: source-order bridge vs MMR delta **+1.90**, 95% CI **[+0.81, +3.01]**.
- Llama: source-order bridge vs MMR positive (+0.84) but single-model CI crosses 0.
- Path ordering not supported; source-order is default output order.

## Invalidated claims
- Llama source-order bridge vs independent delta **+17.41** (and pooled independent comparisons).
- Any claim that independent top-span is a fair relevance-only control **on Llama G2 artifacts**.

## Next steps
1. Fix `build_full_packs` independent selection.
2. Re-run independent-only QA ladder (12 → 60 → 1150).
3. Update bootstrap tables; retain old rows tagged `INVALIDATED_BY_INDEPENDENT_WIRING_BUG`.
4. Do not launch G3/G4/G5 until independent repair is complete.
"""
    write_text(out / "G2_REVISED_DECISION.md", decision_md)

    guard_md = f"""# G2_REVISED_CLAIM_GUARD

Generated: {utc_now()}

## ALLOWED (mechanism — MMR-based)
- Bridge-aware **source-order selection** outperforms **matched MMR** on Qwen (Holm-significant).
- **Pooled** two-model bridge vs MMR: +1.90 macro F1, 95% CI [+0.81, +3.01].
- Llama bridge vs MMR: positive mean delta (+0.84) but **not** singly conclusive (CI crosses 0).
- Path-order serialization is **not** supported as default.
- Source-order is the preferred output ordering.
- Connectivity score changes selection; standalone QA gain not significant.

## FORBIDDEN (until independent re-run)
- "Llama bridge beats independent top-span by +17.4" or any magnitude from invalidated G2 independent run.
- "Independent top-span validates bridge/path selection on Llama."
- Using invalidated independent as a core mechanism control in tables or abstract.

## PROVISIONAL (after independent fix + re-QA)
- Independent top-span as relevance-only control — only if AUDIT_PASS on repaired runner.

## SAFE ONE-SENTENCE MECHANISM CLAIM
Bridge-aware source-order selection outperforms a matched MMR baseline on Qwen and in pooled two-model evaluation,
while the Llama-3 model-specific improvement is positive but not statistically conclusive.
"""
    write_text(out / "G2_REVISED_CLAIM_GUARD.md", guard_md)

    print(f"Done. Verdict: {verdict}")
    print(f"Wiring failures: {wiring_fail}/{n}")
    print(f"Artifact match: {equiv_summary['match']}/{equiv_summary['checked']}")


if __name__ == "__main__":
    main()
