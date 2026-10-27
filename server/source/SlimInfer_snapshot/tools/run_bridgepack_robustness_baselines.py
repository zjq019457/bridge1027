#!/usr/bin/env python3
"""Run BridgePack reviewer-robustness baselines.

This wrapper reuses the existing offline packing, checker, generation, and
LongBench scoring utilities. It does not modify the completed main results.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_TASKS = "qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,narrativeqa"
DEFAULT_VARIANTS = (
    "bm25_chunk_source_order,tfidf_chunk_source_order,"
    "query_only_neighbor32_path_order,bridge_random_neighbor32_path_order,"
    "bridge_same_spans_source_order"
)
DISPLAY = {
    "head_truncate_source_order": "Head Truncation",
    "span_only_source_order": "Span-only Packing",
    "neighbor32_source_order": "Neighbor Packing",
    "neighbor32_path_order": "Neighbor + Path Order",
    "bridge_neighbor32_path_order": "BridgePack",
    "bridge_neighbor32_path_order_tags": "BridgePack + Tags",
    "bm25_chunk_source_order": "BM25 Chunk Packing",
    "tfidf_chunk_source_order": "TF-IDF Chunk Packing",
    "query_only_neighbor32_path_order": "Query-only Neighbor + Path Order",
    "bridge_random_neighbor32_path_order": "Random-Bridge Control",
    "bridge_same_spans_source_order": "Bridge Same-Spans Source Order",
}
ROLE = {
    "head_truncate_source_order": "baseline",
    "span_only_source_order": "baseline",
    "neighbor32_source_order": "ablation",
    "neighbor32_path_order": "ablation",
    "bridge_neighbor32_path_order": "ours_main",
    "bridge_neighbor32_path_order_tags": "optional_formatting",
    "bm25_chunk_source_order": "external_lexical_baseline",
    "tfidf_chunk_source_order": "external_lexical_baseline",
    "query_only_neighbor32_path_order": "control_no_bridge_link",
    "bridge_random_neighbor32_path_order": "control_random_bridge",
    "bridge_same_spans_source_order": "control_source_order_same_spans",
}
ORIGINAL_METHODS = [
    "head_truncate_source_order",
    "span_only_source_order",
    "neighbor32_source_order",
    "neighbor32_path_order",
    "bridge_neighbor32_path_order",
    "bridge_neighbor32_path_order_tags",
]


def _parse_list(value: str) -> List[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


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
    raise ValueError("No Qwen model path found. Pass --model_name_or_path or set QWEN_MODEL.")


def _line_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _expected_counts(run_dir: Path, tasks: Sequence[str], max_samples: int) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for task in tasks:
        ref = (
            run_dir / "main_1024" / "packed_prompts_build" / "packed_prompts" /
            "span_only_source_order" / "budget_1024" / f"{task}.jsonl"
        )
        count = _line_count(ref)
        if max_samples > 0:
            count = min(count, max_samples)
        counts[task] = count
    return counts


def _packed_complete(packed_dir: Path, variants: Sequence[str], tasks: Sequence[str], budget: int, expected: Mapping[str, int]) -> bool:
    if not packed_dir.exists():
        return False
    for variant in variants:
        for task in tasks:
            path = packed_dir / "packed_prompts" / variant / f"budget_{budget}" / f"{task}.jsonl"
            if _line_count(path) != int(expected.get(task, 0)):
                return False
    return True


def _predictions_complete(out_dir: Path, variants: Sequence[str], tasks: Sequence[str], expected: Mapping[str, int]) -> bool:
    for variant in variants:
        for task in tasks:
            path = out_dir / "predictions" / variant / f"{task}.jsonl"
            if _line_count(path) != int(expected.get(task, 0)):
                return False
    return True


def build_packed_prompts(
    run_dir: Path,
    tasks: Sequence[str],
    variants: Sequence[str],
    budget: int,
    max_samples: int,
    resume: bool,
) -> Path:
    packed_dir = run_dir / "robustness" / "packed_prompts_build"
    expected = _expected_counts(run_dir, tasks, max_samples)
    if resume and _packed_complete(packed_dir, variants, tasks, budget, expected):
        return packed_dir
    if packed_dir.exists():
        shutil.rmtree(packed_dir)
    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "run_evidence_text_packing_offline.py"),
        "--tasks",
        ",".join(tasks),
        "--max_samples",
        str(max_samples),
        "--budgets",
        str(budget),
        "--variants",
        ",".join(variants),
        "--out_dir",
        str(packed_dir),
    ]
    _run(cmd, run_dir / "robustness" / "logs" / "offline_packing.log")
    return packed_dir


def run_checker(run_dir: Path, packed_dir: Path, tasks: Sequence[str], variants: Sequence[str], budget: int, max_samples: int) -> None:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "check_evidence_text_packing_generation_inputs.py"),
        "--packed_dir",
        str(packed_dir),
        "--tasks",
        ",".join(tasks),
        "--budget",
        str(budget),
        "--variants",
        ",".join(variants),
        "--max_samples",
        str(max_samples),
    ]
    _run(cmd, run_dir / "robustness" / "logs" / "input_checker.log")


def run_generation(
    run_dir: Path,
    packed_dir: Path,
    model_name_or_path: str,
    tasks: Sequence[str],
    variants: Sequence[str],
    budget: int,
    max_samples: int,
    resume: bool,
) -> None:
    out_dir = run_dir / "robustness"
    expected = _expected_counts(run_dir, tasks, max_samples)
    score_only = resume and _predictions_complete(out_dir, variants, tasks, expected)
    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "run_evidence_text_packing_generation_qwen32.py"),
        "--packed_dir",
        str(packed_dir),
        "--tasks",
        ",".join(tasks),
        "--max_samples",
        str(max_samples),
        "--budget",
        str(budget),
        "--variants",
        ",".join(variants),
        "--model_name_or_path",
        model_name_or_path,
        "--out_dir",
        str(out_dir),
    ]
    if resume:
        cmd.append("--resume")
    if score_only:
        cmd.append("--score_only")
    _run(cmd, run_dir / "robustness" / "logs" / "generation.log")


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


def _original_scores(run_dir: Path) -> Tuple[Dict[Tuple[str, str], float], Dict[str, float]]:
    task_scores: Dict[Tuple[str, str], float] = {}
    macro_scores: Dict[str, float] = {}
    for row in _read_csv(run_dir / "paper_tables" / "03_component_ablation.csv"):
        value = _float(row.get("score"))
        if value is not None:
            task_scores[(row.get("variant", ""), row.get("task", ""))] = value
    for row in _read_csv(run_dir / "paper_tables" / "01_main_table_1024.csv"):
        value = _float(row.get("macro_avg"))
        if value is not None:
            macro_scores[row.get("variant", "")] = value
    return task_scores, macro_scores


def _robust_scores(run_dir: Path) -> Tuple[Dict[Tuple[str, str], float], Dict[str, float]]:
    task_scores: Dict[Tuple[str, str], float] = {}
    macro_scores: Dict[str, float] = {}
    for row in _read_csv(run_dir / "robustness" / "summary" / "method_task_breakdown.csv"):
        value = _float(row.get("score"))
        if value is not None:
            task_scores[(row.get("variant", ""), row.get("task", ""))] = value
    for row in _read_csv(run_dir / "robustness" / "summary" / "summary.csv"):
        value = _float(row.get("macro_avg"))
        if value is not None:
            macro_scores[row.get("variant", "")] = value
    return task_scores, macro_scores


def _row_for_method(
    method: str,
    tasks: Sequence[str],
    task_scores: Mapping[Tuple[str, str], float],
    macro_scores: Mapping[str, float],
    span_macro: float,
    bridge_macro: float,
) -> Dict[str, Any]:
    avg = macro_scores.get(method)
    row: Dict[str, Any] = {
        "method": method,
        "display_name": DISPLAY.get(method, method),
        "paper_role": ROLE.get(method, ""),
        "budget": 1024,
        "avg": _fmt(avg),
        "delta_vs_span_only": _fmt(avg - span_macro) if avg is not None else "",
        "delta_vs_bridgepack": _fmt(avg - bridge_macro) if avg is not None else "",
    }
    for task in tasks:
        row[task] = _fmt(task_scores.get((method, task)))
    return row


def _write_paper_tables(run_dir: Path, tasks: Sequence[str], variants: Sequence[str]) -> None:
    paper_dir = run_dir / "robustness" / "paper_tables"
    original_task_scores, original_macro_scores = _original_scores(run_dir)
    robust_task_scores, robust_macro_scores = _robust_scores(run_dir)
    task_scores = {**original_task_scores, **robust_task_scores}
    macro_scores = {**original_macro_scores, **robust_macro_scores}
    span_macro = macro_scores.get("span_only_source_order", 0.0)
    bridge_macro = macro_scores.get("bridge_neighbor32_path_order", 0.0)

    robust_rows = [
        _row_for_method(method, tasks, task_scores, macro_scores, span_macro, bridge_macro)
        for method in ["span_only_source_order", "bridge_neighbor32_path_order", *variants]
    ]
    updated_rows = [
        _row_for_method(method, tasks, task_scores, macro_scores, span_macro, bridge_macro)
        for method in [*ORIGINAL_METHODS, *variants]
    ]
    fields = [
        "method", "display_name", "paper_role", "budget", *tasks,
        "avg", "delta_vs_span_only", "delta_vs_bridgepack",
    ]
    _write_csv(paper_dir / "01_robustness_main_table.csv", robust_rows, fields)
    _write_csv(paper_dir / "04_updated_main_table_with_external_baselines.csv", updated_rows, fields)

    delta_rows: List[Dict[str, Any]] = []
    for method in variants:
        for task in ["_macro", *tasks]:
            score = macro_scores.get(method) if task == "_macro" else task_scores.get((method, task))
            bridge = bridge_macro if task == "_macro" else task_scores.get(("bridge_neighbor32_path_order", task))
            delta_rows.append({
                "method": method,
                "task": task,
                "score": _fmt(score),
                "bridgepack_score": _fmt(bridge),
                "delta_vs_bridgepack": _fmt(score - bridge) if score is not None and bridge is not None else "",
                "interpretation": ROLE.get(method, ""),
            })
    _write_csv(
        paper_dir / "02_delta_vs_bridgepack.csv",
        delta_rows,
        ["method", "task", "score", "bridgepack_score", "delta_vs_bridgepack", "interpretation"],
    )

    bm25 = macro_scores.get("bm25_chunk_source_order")
    tfidf = macro_scores.get("tfidf_chunk_source_order")
    query_only = macro_scores.get("query_only_neighbor32_path_order")
    random_bridge = macro_scores.get("bridge_random_neighbor32_path_order")
    same_spans = macro_scores.get("bridge_same_spans_source_order")
    lines = [
        "# BridgePack Robustness Control Interpretation",
        "",
        f"- Span-only baseline macro: {_fmt(span_macro)}.",
        f"- BridgePack macro: {_fmt(bridge_macro)}.",
        f"- BM25 chunk packing macro: {_fmt(bm25)}.",
        f"- TF-IDF chunk packing macro: {_fmt(tfidf)}.",
        f"- Query-only control macro: {_fmt(query_only)}.",
        f"- Random-bridge control macro: {_fmt(random_bridge)}.",
        f"- Same-spans source-order control macro: {_fmt(same_spans)}.",
        "",
        "## Interpretation",
        "",
        "- BM25 and TF-IDF address the concern that the span-only baseline may be too weak by adding common lexical retrieval-style context compression baselines.",
        "- `query_only_neighbor32_path_order` disables explicit bridge-link scoring while keeping neighbor/path-style packing pressure.",
        "- `bridge_random_neighbor32_path_order` keeps query-linked seeds but randomizes bridge/support choices with deterministic seed 0.",
        "- `bridge_same_spans_source_order` is a best-effort same-span control that reorders BridgePack-selected spans back into source order to isolate path-ordering effects.",
        "- These controls should be interpreted as robustness checks, not new main methods.",
    ]
    if same_spans is not None and bridge_macro:
        same_spans_delta = same_spans - bridge_macro
        if same_spans_delta < 0:
            same_spans_text = "This supports path-order organization as useful beyond span selection alone."
        elif same_spans_delta > 0:
            same_spans_text = (
                "This does not support path-order as the source of gains in this run; "
                "source-ordering the same selected spans is slightly stronger."
            )
        else:
            same_spans_text = "This is neutral on whether path-order contributes beyond span selection."
        lines.append(
            f"- Same-spans source-order delta vs BridgePack: {_fmt(same_spans_delta)}. {same_spans_text}"
        )
    _write_text(paper_dir / "03_control_interpretation.md", "\n".join(lines))

    audit_path = run_dir / "paper_tables" / "11_model_tokenizer_budget_audit.csv"
    audit_rows = _read_csv(audit_path)
    main_1024_over_budget = sum(
        int(float(row.get("over_budget_untruncated_samples", 0) or 0))
        for row in audit_rows
        if row.get("source_scope") == "main_1024/packed_prompts_build"
        and str(row.get("budget")) == "1024"
    )
    robustness_1024_over_budget = sum(
        int(float(row.get("over_budget_untruncated_samples", 0) or 0))
        for row in audit_rows
        if row.get("source_scope") == "robustness/packed_prompts_build"
        and str(row.get("budget")) == "1024"
    )
    audit_status = "available" if audit_rows else "pending"
    budget_risk = (
        "Audit reveals tokenizer-budget mismatch; stored packed_token_count is a regex estimate, "
        f"not the generation tokenizer. Main-1024 over-budget untruncated samples: {main_1024_over_budget}; "
        f"robustness-1024 over-budget untruncated samples: {robustness_1024_over_budget}. "
        "Do not claim exact Qwen-token budget fairness until model-tokenizer-budget repacking is run."
        if audit_rows else
        "Tokenizer audit pending; exact generation-token budget fairness is not yet established."
    )
    risk_lines = [
        "# BridgePack Reviewer Risk Assessment",
        "",
        "| Reviewer concern | Status after robustness pass |",
        "| --- | --- |",
        f"| Baselines may be too weak | Addressed with BM25 and TF-IDF chunk-packing baselines. Best lexical baseline macro: {_fmt(max([x for x in [bm25, tfidf] if x is not None], default=0.0))}; this exceeds BridgePack in the current run, so BridgePack should not be claimed as stronger than lexical retrieval baselines. |",
        f"| Bridge-aware path may be only lexical relevance | Query-only and same-spans controls are competitive or stronger than BridgePack, while random-bridge is weaker. This weakens the claim that bridge-link scoring/path order is the unique source of gains. BridgePack macro: {_fmt(bridge_macro)}. |",
        f"| Token budget may not use generation tokenizer | Tokenizer audit status: {audit_status}. {budget_risk} |",
        "| Tags are not core | Main method remains no-tags BridgePack; tags are optional formatting. |",
        "| Case studies may be cherry-picked | Existing writeup includes objective positive cases plus a failure case. |",
        "| Only one model is evaluated | Optional second-model gate runner is prepared; do not claim second-model robustness until run. |",
        "",
        "## Paper Guidance",
        "",
        "Keep BridgePack framed as evidence-aware text repacking. Do not claim strict KV-cache pruning or original-prefill KV-memory reduction.",
        "Until model-tokenizer-budget repacking is complete, describe the completed runs as using the offline regex packing budget and report the tokenizer audit as a limitation.",
    ]
    _write_text(paper_dir / "05_reviewer_risk_assessment.md", "\n".join(risk_lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="")
    parser.add_argument("--run_dir", default="results/bridgepack_paper_20260618_070828")
    parser.add_argument("--tasks", default=DEFAULT_TASKS)
    parser.add_argument("--budget", type=int, default=1024)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--variants", default=DEFAULT_VARIANTS)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--paper_tables_only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    tasks = _parse_list(args.tasks)
    variants = _parse_list(args.variants)
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir does not exist: {run_dir}")
    if args.seed != 0:
        raise ValueError("Only deterministic seed 0 is currently supported by the packing controls.")

    if not args.paper_tables_only:
        model = _detect_model(run_dir, args.model_name_or_path)
        packed_dir = build_packed_prompts(run_dir, tasks, variants, args.budget, args.max_samples, args.resume)
        run_checker(run_dir, packed_dir, tasks, variants, args.budget, args.max_samples)
        run_generation(run_dir, packed_dir, model, tasks, variants, args.budget, args.max_samples, args.resume)
    _write_paper_tables(run_dir, tasks, variants)
    print(f"ROBUSTNESS_DIR={run_dir / 'robustness'}")


if __name__ == "__main__":
    main()
