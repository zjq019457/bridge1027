#!/usr/bin/env python3
"""Run BridgePack variants under an exact generation-token packed-context budget.

The completed BridgePack paper run used a lightweight regex token estimator for
packing. This runner rebuilds packed prompts with the actual generation
tokenizer, then runs the same generation/scoring pipeline on the exact-budget
packed contexts.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bridgekv.evidence_text_packing import (  # noqa: E402
    CandidateSpan,
    PackedEvidence,
    _expanded_piece,
    _prefix,
    build_candidate_spans,
    pack_evidence,
    packed_to_json,
)
from tools.run_evidence_text_packing_offline import (  # noqa: E402
    _candidate_dataset_roots,
    _load_samples,
)

DEFAULT_TASKS = "qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,narrativeqa"
DEFAULT_VARIANTS = (
    "span_only_source_order,bm25_chunk_source_order,tfidf_chunk_source_order,"
    "query_only_neighbor32_path_order,bridge_random_neighbor32_path_order,"
    "bridge_same_spans_source_order,bridge_neighbor32_path_order,"
    "bridge_neighbor32_path_order_tags,lexical_bridgepack_source_order,"
    "lexical_bridgepack_path_order,lexical_bridgepack_hybrid"
)
MULTIHOP_TASKS = ["hotpotqa", "2wikimqa", "musique"]
DISPLAY = {
    "span_only_source_order": "Span-only Packing",
    "bm25_chunk_source_order": "BM25 Chunk Packing",
    "tfidf_chunk_source_order": "TF-IDF Chunk Packing",
    "query_only_neighbor32_path_order": "Query-only Neighbor + Path Order",
    "bridge_random_neighbor32_path_order": "Random-Bridge Control",
    "bridge_same_spans_source_order": "Bridge Same-Spans Source Order",
    "bridge_neighbor32_path_order": "BridgePack",
    "bridge_neighbor32_path_order_tags": "BridgePack + Tags",
    "lexical_bridgepack_source_order": "Lexical BridgePack Source Order",
    "lexical_bridgepack_path_order": "Lexical BridgePack Path Order",
    "lexical_bridgepack_hybrid": "Lexical BridgePack Hybrid",
}
ROLE = {
    "span_only_source_order": "baseline",
    "bm25_chunk_source_order": "external_lexical_baseline",
    "tfidf_chunk_source_order": "external_lexical_baseline",
    "query_only_neighbor32_path_order": "control_no_bridge_link",
    "bridge_random_neighbor32_path_order": "control_random_bridge",
    "bridge_same_spans_source_order": "control_source_order_same_spans",
    "bridge_neighbor32_path_order": "ours_original",
    "bridge_neighbor32_path_order_tags": "optional_formatting",
    "lexical_bridgepack_source_order": "lexical_initialized_candidate",
    "lexical_bridgepack_path_order": "lexical_initialized_path_ablation",
    "lexical_bridgepack_hybrid": "lexical_hybrid_candidate",
}


def _parse_list(value: str) -> List[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _detect_model(run_dir: Path, user_value: str) -> str:
    candidates = [
        user_value,
        os.environ.get("QWEN_MODEL", ""),
        _load_json(run_dir / "paper_tables" / "00_generation_config.json").get("model_name_or_path", ""),
        _load_json(run_dir / "config.json").get("model_name_or_path", ""),
    ]
    for candidate in candidates:
        if candidate:
            return str(candidate)
    raise ValueError("No model path found. Pass --model_name_or_path or set QWEN_MODEL.")


def _load_tokenizer(model_name_or_path: str) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _token_count(tokenizer: Any, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False).input_ids)


def _truncate_exact(tokenizer: Any, text: str, budget: int) -> str:
    ids = tokenizer(text, add_special_tokens=False).input_ids
    if len(ids) <= budget:
        return text
    return tokenizer.decode(ids[:budget], skip_special_tokens=True)


def _run(cmd: Sequence[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    printable = " ".join(str(part) for part in cmd)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n$ {printable}\n")
        log.flush()
        proc = subprocess.run(
            list(cmd),
            cwd=REPO_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        log.write(f"returncode={proc.returncode}\n")
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed rc={proc.returncode}: {printable}. See {log_path}")


def _generation_prompt(task: str, packed_text: str, question: str) -> str:
    templates = _load_json(REPO_ROOT / "configs" / "dataset2prompt.json")
    template = str(templates.get(task, ""))
    if template:
        try:
            return template.format(context=packed_text, input=question, question=question)
        except Exception:
            pass
    return packed_text


def _span_order_from_nominal(
    sample_id: str,
    task: str,
    question: str,
    context: str,
    variant: str,
    budget: int,
) -> Tuple[PackedEvidence, Dict[str, CandidateSpan], Dict[int, Dict[str, Any]]]:
    nominal, _audit = pack_evidence(sample_id, task, question, context, variant, budget)
    candidates, page_map = build_candidate_spans(sample_id, task, question, context)
    by_id = {span.span_id: span for span in candidates}
    return nominal, by_id, page_map


def _exact_repack(
    tokenizer: Any,
    sample: Mapping[str, Any],
    variant: str,
    budget: int,
    model_name_or_path: str,
) -> Dict[str, Any]:
    task = str(sample["task"])
    sample_id = str(sample["sample_id"])
    question = str(sample.get("question", ""))
    context = str(sample.get("context", ""))
    nominal, by_id, page_map = _span_order_from_nominal(sample_id, task, question, context, variant, budget)
    selected_spans = [by_id[span_id] for span_id in nominal.selected_span_ids if span_id in by_id]
    if not selected_spans and by_id:
        selected_spans = [next(iter(by_id.values()))]

    prefix = _prefix(question, nominal.used_tags)
    pieces: List[str] = []
    exact_selected: List[CandidateSpan] = []
    truncated = False
    for span in selected_spans:
        piece, _expanded = _expanded_piece(
            span,
            page_map,
            nominal.neighbor_size,
            role="support",
            use_tags=nominal.used_tags,
        )
        candidate_text = "\n\n".join([prefix, *pieces, piece]).strip()
        if _token_count(tokenizer, candidate_text) <= budget:
            pieces.append(piece)
            exact_selected.append(span)
            continue
        if not pieces:
            candidate_text = _truncate_exact(tokenizer, candidate_text, budget)
            pieces = [candidate_text.replace(prefix, "", 1).strip()]
            exact_selected = [span]
            truncated = True
        break

    packed_text = "\n\n".join([prefix, *pieces]).strip()
    if _token_count(tokenizer, packed_text) > budget:
        packed_text = _truncate_exact(tokenizer, packed_text, budget)
        truncated = True
    model_token_count = _token_count(tokenizer, packed_text)
    source_tokens = nominal.source_token_count
    payload = packed_to_json(PackedEvidence(
        sample_id=sample_id,
        task=task,
        variant=variant,
        budget=budget,
        packed_text=packed_text,
        selected_span_ids=[span.span_id for span in exact_selected],
        selected_page_ids=sorted({span.page_id for span in exact_selected}),
        packed_token_count=model_token_count,
        source_token_count=source_tokens,
        compression_ratio=model_token_count / max(1, source_tokens),
        ordering=nominal.ordering,
        neighbor_size=nominal.neighbor_size,
        used_bridge_path=nominal.used_bridge_path,
        used_tags=nominal.used_tags,
        truncated=truncated or nominal.truncated,
    ))
    payload.update({
        "question": question,
        "packed_prompt": packed_text,
        "generation_prompt": _generation_prompt(task, packed_text, question),
        "original_context": context,
        "answers": sample.get("answers", []),
        "gold_analysis_only_answers": sample.get("gold_analysis_only_answers", []),
        "all_classes": sample.get("all_classes", None),
        "length": sample.get("length", 0),
        "source_file": sample.get("source_file", ""),
        "model_token_count": model_token_count,
        "exact_model_token_budget": True,
        "model_tokenizer": model_name_or_path,
        "nominal_regex_selected_span_count": len(nominal.selected_span_ids),
        "exact_selected_span_count": len(exact_selected),
    })
    return payload


def _line_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _expected_counts(run_dir: Path, tasks: Sequence[str], max_samples: int) -> Dict[str, int]:
    roots = _candidate_dataset_roots("")
    counts: Dict[str, int] = {}
    for task in tasks:
        count = len(_load_samples(task, roots, max_samples))
        counts[task] = count
    return counts


def _packed_complete(packed_dir: Path, variants: Sequence[str], tasks: Sequence[str], budget: int, expected: Mapping[str, int]) -> bool:
    for variant in variants:
        for task in tasks:
            path = packed_dir / "packed_prompts" / variant / f"budget_{budget}" / f"{task}.jsonl"
            if _line_count(path) != int(expected.get(task, 0)):
                return False
    return True


def build_exact_packed_prompts(
    tokenizer: Any,
    model_name_or_path: str,
    out_dir: Path,
    tasks: Sequence[str],
    variants: Sequence[str],
    budget: int,
    max_samples: int,
    resume: bool,
) -> Path:
    packed_dir = out_dir / "packed_prompts_build"
    expected = _expected_counts(out_dir, tasks, max_samples)
    if resume and _packed_complete(packed_dir, variants, tasks, budget, expected):
        return packed_dir
    roots = _candidate_dataset_roots("")
    for variant in variants:
        for task in tasks:
            path = packed_dir / "packed_prompts" / variant / f"budget_{budget}" / f"{task}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            samples = _load_samples(task, roots, max_samples)
            with path.open("w", encoding="utf-8") as handle:
                for sample in samples:
                    payload = _exact_repack(tokenizer, sample, variant, budget, model_name_or_path)
                    json.dump(payload, handle, ensure_ascii=False)
                    handle.write("\n")
    return packed_dir


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _mean(values: Sequence[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _percentile(values: Sequence[int], q: float) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    rank = (len(vals) - 1) * q
    lo = int(rank)
    hi = min(lo + 1, len(vals) - 1)
    weight = rank - lo
    return vals[lo] * (1.0 - weight) + vals[hi] * weight


def write_fairness_tables(
    out_dir: Path,
    packed_dir: Path,
    tasks: Sequence[str],
    variants: Sequence[str],
    budget: int,
    model_name_or_path: str,
) -> None:
    paper_dir = out_dir / "paper_tables"
    stats_rows: List[Dict[str, Any]] = []
    missing_rows: List[Dict[str, Any]] = []
    sample_sets: Dict[Tuple[str, str], set[str]] = {}
    for variant in variants:
        for task in tasks:
            rows = _read_jsonl(packed_dir / "packed_prompts" / variant / f"budget_{budget}" / f"{task}.jsonl")
            sample_ids = {str(row.get("sample_id")) for row in rows}
            sample_sets[(variant, task)] = sample_ids
            counts = [int(row.get("model_token_count", row.get("packed_token_count", 0)) or 0) for row in rows]
            stats_rows.append({
                "variant": variant,
                "task": task,
                "budget": budget,
                "num_samples": len(rows),
                "model_token_count_mean": round(_mean(counts), 4),
                "model_token_count_p95": round(_percentile(counts, 0.95), 4),
                "model_token_count_max": max(counts) if counts else 0,
                "over_budget_samples": sum(1 for value in counts if value > budget),
                "truncated_samples": sum(1 for row in rows if bool(row.get("truncated", False))),
                "exact_model_token_budget": all(bool(row.get("exact_model_token_budget")) for row in rows),
            })
    for task in tasks:
        union = set().union(*(sample_sets.get((variant, task), set()) for variant in variants))
        for variant in variants:
            missing = sorted(union - sample_sets.get((variant, task), set()))
            extra = sorted(sample_sets.get((variant, task), set()) - union)
            if missing or extra:
                missing_rows.append({
                    "task": task,
                    "variant": variant,
                    "missing_sample_ids": ",".join(missing[:20]),
                    "extra_sample_ids": ",".join(extra[:20]),
                    "missing_count": len(missing),
                    "extra_count": len(extra),
                })
    _write_csv(
        paper_dir / "00_model_token_budget_stats.csv",
        stats_rows,
        [
            "variant", "task", "budget", "num_samples", "model_token_count_mean",
            "model_token_count_p95", "model_token_count_max", "over_budget_samples",
            "truncated_samples", "exact_model_token_budget",
        ],
    )
    _write_csv(
        paper_dir / "00_missing_samples.csv",
        missing_rows,
        ["task", "variant", "missing_sample_ids", "extra_sample_ids", "missing_count", "extra_count"],
    )
    same_samples = not missing_rows
    over_budget = sum(int(row["over_budget_samples"]) for row in stats_rows)
    lines = [
        "# Exact Qwen-Token Budget Fairness Report",
        "",
        "BridgePack exact-budget repacking is evaluated as evidence-aware context compression / text repacking, not KV-cache pruning.",
        "",
        "## Required Checks",
        "",
        f"1. same_sample_ids_for_every_variant: **{same_samples}**",
        f"2. packed_context_uses_actual_generation_tokenizer: **True**",
        f"3. packed_context_token_count_lte_budget: **{over_budget == 0}**",
        f"4. same_prompt_wrapper_for_generation: **True**",
        f"5. same_generation_config_and_scoring_pipeline: **True**",
        f"6. no_gold_metadata_used_for_selection: **True**",
        "",
        "## Configuration",
        "",
        f"- Model/tokenizer: `{model_name_or_path}`",
        f"- Packed context budget: {budget} model-tokenizer tokens.",
        f"- Variants: `{','.join(variants)}`",
        "- Generation/scoring: existing `tools/run_evidence_text_packing_generation_qwen32.py` pipeline.",
        "",
        "## Notes",
        "",
        "The budget is enforced on `packed_text`, the compressed context inserted into the shared prompt wrapper.",
        "The final full prompt contains task instructions in addition to the packed context.",
    ]
    _write_text(paper_dir / "00_exact_budget_fairness_report.md", "\n".join(lines))


def _predictions_complete(out_dir: Path, variants: Sequence[str], tasks: Sequence[str], expected: Mapping[str, int]) -> bool:
    for variant in variants:
        for task in tasks:
            path = out_dir / "predictions" / variant / f"{task}.jsonl"
            if _line_count(path) != int(expected.get(task, 0)):
                return False
    return True


def run_generation(
    out_dir: Path,
    packed_dir: Path,
    model_name_or_path: str,
    tasks: Sequence[str],
    variants: Sequence[str],
    budget: int,
    max_samples: int,
    resume: bool,
    dtype: str,
    device: str,
) -> None:
    expected = _expected_counts(out_dir, tasks, max_samples)
    score_only = resume and _predictions_complete(out_dir, variants, tasks, expected)
    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "run_evidence_text_packing_generation_qwen32.py"),
        "--packed_dir", str(packed_dir),
        "--tasks", ",".join(tasks),
        "--max_samples", str(max_samples),
        "--budget", str(budget),
        "--variants", ",".join(variants),
        "--model_name_or_path", model_name_or_path,
        "--out_dir", str(out_dir),
        "--dtype", dtype,
        "--device", device,
    ]
    if resume:
        cmd.append("--resume")
    if score_only:
        cmd.append("--score_only")
    _run(cmd, out_dir / "logs" / "generation.log")


def _float(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt(value: Any) -> str:
    number = _float(value)
    if number is None:
        return ""
    return f"{number:.4f}".rstrip("0").rstrip(".")


def _score_maps(out_dir: Path) -> Tuple[Dict[Tuple[str, str], float], Dict[str, float]]:
    task_scores: Dict[Tuple[str, str], float] = {}
    macro_scores: Dict[str, float] = {}
    for row in _read_csv(out_dir / "summary" / "method_task_breakdown.csv"):
        score = _float(row.get("score"))
        if score is not None:
            task_scores[(row.get("variant", ""), row.get("task", ""))] = score
    for row in _read_csv(out_dir / "summary" / "summary.csv"):
        score = _float(row.get("macro_avg"))
        if score is not None:
            macro_scores[row.get("variant", "")] = score
    return task_scores, macro_scores


def _main_row(
    variant: str,
    tasks: Sequence[str],
    task_scores: Mapping[Tuple[str, str], float],
    macro_scores: Mapping[str, float],
    span_macro: float,
) -> Dict[str, Any]:
    avg = macro_scores.get(variant)
    row: Dict[str, Any] = {
        "method": variant,
        "display_name": DISPLAY.get(variant, variant),
        "paper_role": ROLE.get(variant, ""),
        "budget": 1024,
        "avg": _fmt(avg),
        "delta_vs_span_only": _fmt(avg - span_macro) if avg is not None else "",
    }
    for task in tasks:
        row[task] = _fmt(task_scores.get((variant, task)))
    return row


def write_result_tables(out_dir: Path, tasks: Sequence[str], variants: Sequence[str]) -> None:
    paper_dir = out_dir / "paper_tables"
    task_scores, macro_scores = _score_maps(out_dir)
    span_macro = macro_scores.get("span_only_source_order", 0.0)
    bm25 = macro_scores.get("bm25_chunk_source_order")
    tfidf = macro_scores.get("tfidf_chunk_source_order")
    bridge = macro_scores.get("bridge_neighbor32_path_order")
    main_rows = [_main_row(variant, tasks, task_scores, macro_scores, span_macro) for variant in variants]
    fields = ["method", "display_name", "paper_role", "budget", *tasks, "avg", "delta_vs_span_only"]
    _write_csv(paper_dir / "01_exact_budget_main_table_1024.csv", main_rows, fields)

    delta_rows: List[Dict[str, Any]] = []
    for variant in variants:
        for task in ["_macro", *tasks]:
            score = macro_scores.get(variant) if task == "_macro" else task_scores.get((variant, task))
            base = span_macro if task == "_macro" else task_scores.get(("span_only_source_order", task))
            delta_rows.append({
                "method": variant,
                "task": task,
                "score": _fmt(score),
                "span_only_score": _fmt(base),
                "delta_vs_span_only": _fmt(score - base) if score is not None and base is not None else "",
            })
    _write_csv(
        paper_dir / "02_exact_budget_delta_vs_span_only.csv",
        delta_rows,
        ["method", "task", "score", "span_only_score", "delta_vs_span_only"],
    )

    lexical_rows: List[Dict[str, Any]] = []
    for variant in variants:
        avg = macro_scores.get(variant)
        lexical_rows.append({
            "method": variant,
            "avg": _fmt(avg),
            "bm25_avg": _fmt(bm25),
            "tfidf_avg": _fmt(tfidf),
            "delta_vs_bm25": _fmt(avg - bm25) if avg is not None and bm25 is not None else "",
            "delta_vs_tfidf": _fmt(avg - tfidf) if avg is not None and tfidf is not None else "",
        })
    _write_csv(
        paper_dir / "03_exact_budget_delta_vs_bm25_tfidf.csv",
        lexical_rows,
        ["method", "avg", "bm25_avg", "tfidf_avg", "delta_vs_bm25", "delta_vs_tfidf"],
    )

    multihop_rows: List[Dict[str, Any]] = []
    span_mh = statistics.mean(
        task_scores.get(("span_only_source_order", task), 0.0)
        for task in MULTIHOP_TASKS
    )
    for variant in variants:
        values = [task_scores.get((variant, task)) for task in MULTIHOP_TASKS]
        values_f = [float(value) for value in values if value is not None]
        mh_avg = statistics.mean(values_f) if values_f else None
        row: Dict[str, Any] = {
            "method": variant,
            "multihop_avg": _fmt(mh_avg),
            "delta_vs_span_only_multihop": _fmt(mh_avg - span_mh) if mh_avg is not None else "",
        }
        for task in MULTIHOP_TASKS:
            row[task] = _fmt(task_scores.get((variant, task)))
        multihop_rows.append(row)
    _write_csv(
        paper_dir / "04_exact_budget_multihop_table.csv",
        multihop_rows,
        ["method", "multihop_avg", "delta_vs_span_only_multihop", *MULTIHOP_TASKS],
    )

    lexical_methods = [
        "lexical_bridgepack_source_order",
        "lexical_bridgepack_path_order",
        "lexical_bridgepack_hybrid",
    ]
    best_lexical_bridgepack = max(
        [macro_scores.get(method, float("-inf")) for method in lexical_methods],
        default=float("-inf"),
    )
    best_lexical_baseline = max(
        [value for value in [bm25, tfidf] if value is not None],
        default=float("-inf"),
    )
    bridge_beats_span = bridge is not None and bridge > span_macro
    lexical_beats_baselines = best_lexical_bridgepack > best_lexical_baseline
    bm25_tfidf_beat_bridge = any(
        value is not None and bridge is not None and value > bridge
        for value in [bm25, tfidf]
    )
    source_order = macro_scores.get("bridge_same_spans_source_order")
    path_order_help = source_order is not None and bridge is not None and bridge > source_order
    lines = [
        "# Exact-Budget BridgePack Claim Summary",
        "",
        f"- BridgePack beats span-only: **{bridge_beats_span}** (BridgePack={_fmt(bridge)}, span-only={_fmt(span_macro)}).",
        f"- BM25/TF-IDF beat BridgePack: **{bm25_tfidf_beat_bridge}** (BM25={_fmt(bm25)}, TF-IDF={_fmt(tfidf)}).",
        f"- Best lexical-initialized BridgePack beats BM25/TF-IDF: **{lexical_beats_baselines}**.",
        f"- Path-order helps after same-span source-order control: **{path_order_help}**.",
        "",
        "Safe claim level depends on these exact-budget results and should not reuse the earlier regex-budget claim unchanged.",
    ]
    _write_text(paper_dir / "05_exact_budget_claim_summary.md", "\n".join(lines))

    if lexical_beats_baselines:
        positioning = (
            "Position the work as evidence-aware text repacking under exact tokenizer budgets, "
            "with the lexical-initialized path variant as the strongest observed candidate and "
            "BM25/TF-IDF retained as explicit reviewer-facing baselines."
        )
    elif bm25_tfidf_beat_bridge:
        positioning = (
            "Position the work as a careful audit of evidence-aware text repacking under exact tokenizer budgets. "
            "Because BM25/TF-IDF remain stronger, emphasize controls and the lexical-hybrid direction rather than "
            "overclaiming BridgePack as the final best method."
        )
    else:
        positioning = (
            "Position the work as evidence-aware text repacking under exact tokenizer budgets, "
            "with BridgePack compared directly against BM25/TF-IDF and same-span source-order controls."
        )

    interp = [
        "# Reviewer-Safe Exact-Budget Interpretation",
        "",
        "## Required Answers",
        "",
        f"1. Under exact Qwen-token budget, BridgePack still beats span-only: **{bridge_beats_span}** "
        f"(BridgePack={_fmt(bridge)}, span-only={_fmt(span_macro)}).",
        f"2. Under exact Qwen-token budget, BM25/TF-IDF still beat BridgePack: **{bm25_tfidf_beat_bridge}** "
        f"(BM25={_fmt(bm25)}, TF-IDF={_fmt(tfidf)}).",
        f"3. Lexical-initialized BridgePack beats BM25/TF-IDF: **{lexical_beats_baselines}** "
        f"(best lexical BridgePack={_fmt(best_lexical_bridgepack)}, best BM25/TF-IDF={_fmt(best_lexical_baseline)}).",
        f"4. Path-order helps after controlling selected spans: **{path_order_help}** "
        f"(BridgePack={_fmt(bridge)}, same-spans source-order={_fmt(source_order)}).",
        "",
        "## Safe Claims",
        "",
        "- BridgePack-style methods are evidence-aware context compression / text repacking, not KV-cache pruning.",
        "- Exact-budget results use the actual Qwen tokenizer for packed-context budget accounting.",
        "- Claims should be restricted to the observed table values and should compare against BM25/TF-IDF explicitly.",
        "",
        "## Claims To Remove",
        "",
        "- Remove any claim that BridgePack uniquely outperforms common lexical chunk-packing baselines unless the exact-budget table supports it.",
        "- Remove any claim that bridge-link/path-order is the unique source of gains unless controls support it.",
        "- Remove strict KV-cache pruning or original-prefill KV-memory reduction claims.",
        "- Remove the earlier exact 1024-token fairness claim based on regex token counts.",
        "",
        "## AAAI Positioning",
        "",
        positioning,
    ]
    _write_text(paper_dir / "06_reviewer_safe_interpretation.md", "\n".join(interp))


def parse_args() -> argparse.Namespace:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="")
    parser.add_argument("--run_dir", default="results/bridgepack_paper_20260618_070828")
    parser.add_argument("--tasks", default=DEFAULT_TASKS)
    parser.add_argument("--budget", type=int, default=1024)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--variants", default=DEFAULT_VARIANTS)
    parser.add_argument("--out_dir", default=f"results/bridgepack_exact_qwen_budget_1024_{stamp}")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--paper_tables_only", action="store_true")
    parser.add_argument("--dtype", default="fp16")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seed != 0:
        raise ValueError("Only deterministic seed 0 is supported.")
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    tasks = _parse_list(args.tasks)
    variants = _parse_list(args.variants)
    model_name_or_path = _detect_model(run_dir, args.model_name_or_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "model_name_or_path": model_name_or_path,
        "source_run_dir": str(run_dir),
        "tasks": tasks,
        "budget": args.budget,
        "max_samples": args.max_samples,
        "variants": variants,
        "seed": args.seed,
        "dtype": args.dtype,
        "device": args.device,
        "exact_model_token_budget": True,
        "context_compression_not_kv_pruning": True,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    if not args.paper_tables_only:
        tokenizer = _load_tokenizer(model_name_or_path)
        packed_dir = build_exact_packed_prompts(
            tokenizer,
            model_name_or_path,
            out_dir,
            tasks,
            variants,
            args.budget,
            args.max_samples,
            args.resume,
        )
        write_fairness_tables(out_dir, packed_dir, tasks, variants, args.budget, model_name_or_path)
        run_generation(out_dir, packed_dir, model_name_or_path, tasks, variants, args.budget, args.max_samples, args.resume, args.dtype, args.device)
    write_result_tables(out_dir, tasks, variants)
    print(f"OUT_DIR={out_dir}")


if __name__ == "__main__":
    main()
