#!/usr/bin/env python3
"""Run exact-token BridgePack budget ablations via the existing exact runner."""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]


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


def _latest_exact_1024() -> Path | None:
    candidates = sorted((REPO_ROOT / "results").glob("bridgepack_exact_qwen_budget_1024_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _model_from_exact() -> str:
    latest = _latest_exact_1024()
    if latest and (latest / "config.json").exists():
        config = json.loads((latest / "config.json").read_text(encoding="utf-8"))
        return str(config.get("model_name_or_path", ""))
    return ""


def _run(cmd: Sequence[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("$ " + " ".join(str(part) for part in cmd) + "\n")
        log.flush()
        proc = subprocess.run(list(cmd), cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, text=True, check=False)
        log.write(f"returncode={proc.returncode}\n")
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed rc={proc.returncode}: {' '.join(str(x) for x in cmd)}. See {log_path}")


def _score_rows(run_dir: Path, budget: int) -> List[Dict[str, Any]]:
    rows = _read_csv(run_dir / "paper_tables" / "01_exact_budget_main_table_1024.csv")
    for row in rows:
        row["budget"] = budget
    return rows


def _budget_stats(run_dir: Path) -> Dict[str, Any]:
    rows = _read_csv(run_dir / "paper_tables" / "00_model_token_budget_stats.csv")
    return {
        "rows": rows,
        "num_samples": sum(int(row.get("num_samples", 0) or 0) for row in rows),
        "over_budget": sum(int(row.get("over_budget_samples", 0) or 0) for row in rows),
        "max_token": max((int(row.get("model_token_count_max", 0) or 0) for row in rows), default=0),
        "same_samples": not _read_csv(run_dir / "paper_tables" / "00_missing_samples.csv"),
    }


def _float(value: Any) -> float | None:
    try:
        if value in ("", None):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt(value: Any) -> str:
    number = _float(value)
    if number is None:
        return "TODO"
    return f"{number:.4f}".rstrip("0").rstrip(".")


def _missing_or_failed_rows(run_dir: Path, budget: int, variants: Sequence[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for row in _read_csv(run_dir / "paper_tables" / "00_missing_samples.csv"):
        rows.append({
            "budget": budget,
            "issue_type": "sample_id_mismatch",
            "task": row.get("task", ""),
            "method": row.get("variant", row.get("method", "")),
            "expected_count": "",
            "actual_count": "",
            "missing_count": row.get("missing_count", ""),
            "extra_count": row.get("extra_count", ""),
            "details": f"missing={row.get('missing_sample_ids', '')}; extra={row.get('extra_sample_ids', '')}",
        })
    for row in _read_csv(run_dir / "paper_tables" / "00_model_token_budget_stats.csv"):
        method = row.get("variant", row.get("method", ""))
        if method not in variants:
            continue
        task = row.get("task", "")
        expected = int(row.get("num_samples", 0) or 0)
        pred_path = run_dir / "predictions" / method / f"{task}.jsonl"
        actual = 0
        if pred_path.exists():
            with pred_path.open(encoding="utf-8") as handle:
                actual = sum(1 for line in handle if line.strip())
        if actual != expected:
            rows.append({
                "budget": budget,
                "issue_type": "prediction_count_mismatch",
                "task": task,
                "method": method,
                "expected_count": expected,
                "actual_count": actual,
                "missing_count": max(0, expected - actual),
                "extra_count": max(0, actual - expected),
                "details": str(pred_path),
            })
    return rows


def _aggregate(out_dir: Path, budgets: Sequence[int], budget_dirs: Mapping[int, Path], variants: Sequence[str], tasks: Sequence[str]) -> None:
    paper_dir = out_dir / "paper_tables"
    fairness_lines = ["# Exact-Token Budget Ablation Fairness", ""]
    table_rows: List[Dict[str, Any]] = []
    task_rows: List[Dict[str, Any]] = []
    delta_rows: List[Dict[str, Any]] = []
    missing_or_failed: List[Dict[str, Any]] = []
    for budget in budgets:
        run_dir = budget_dirs[budget]
        stats = _budget_stats(run_dir)
        missing_or_failed.extend(_missing_or_failed_rows(run_dir, budget, variants))
        fairness_lines.extend([
            f"## Budget {budget}",
            "",
            f"- Variant-sample rows: `{stats['num_samples']}`.",
            f"- Over-budget packed contexts: `{stats['over_budget']}`.",
            f"- Max packed-context token count: `{stats['max_token']}`.",
            f"- Same sample ids across variants/tasks: **{stats['same_samples']}**.",
            "",
        ])
        scores = _score_rows(run_dir, budget)
        by_method = {row["method"]: row for row in scores}
        bm25 = _float(by_method.get("bm25_chunk_source_order", {}).get("avg"))
        tfidf = _float(by_method.get("tfidf_chunk_source_order", {}).get("avg"))
        for row in scores:
            if row["method"] not in variants:
                continue
            avg = _float(row.get("avg"))
            table_rows.append({
                "budget": budget,
                "method": row["method"],
                "avg": row.get("avg", "TODO"),
                "delta_vs_span_only": row.get("delta_vs_span_only", "TODO"),
                "delta_vs_bm25": _fmt(avg - bm25) if avg is not None and bm25 is not None else "TODO",
                "delta_vs_tfidf": _fmt(avg - tfidf) if avg is not None and tfidf is not None else "TODO",
            })
            for task in tasks:
                task_rows.append({"budget": budget, "method": row["method"], "task": task, "score": row.get(task, "TODO")})
            if row["method"] == "lexical_bridgepack_path_order":
                delta_rows.append({
                    "budget": budget,
                    "method": row["method"],
                    "avg": row.get("avg", "TODO"),
                    "bm25_avg": _fmt(bm25),
                    "tfidf_avg": _fmt(tfidf),
                    "delta_vs_bm25": _fmt(avg - bm25) if avg is not None and bm25 is not None else "TODO",
                    "delta_vs_tfidf": _fmt(avg - tfidf) if avg is not None and tfidf is not None else "TODO",
                })
    _write_text(paper_dir / "00_budget_ablation_fairness.md", "\n".join(fairness_lines))
    _write_csv(paper_dir / "01_budget_ablation_table.csv", table_rows, ["budget", "method", "avg", "delta_vs_span_only", "delta_vs_bm25", "delta_vs_tfidf"])
    _write_csv(paper_dir / "02_budget_ablation_by_task.csv", task_rows, ["budget", "method", "task", "score"])
    _write_csv(paper_dir / "03_budget_ablation_delta_vs_bm25.csv", delta_rows, ["budget", "method", "avg", "bm25_avg", "tfidf_avg", "delta_vs_bm25", "delta_vs_tfidf"])
    _write_csv(
        paper_dir / "06_budget_missing_or_failed_samples.csv",
        missing_or_failed,
        ["budget", "issue_type", "task", "method", "expected_count", "actual_count", "missing_count", "extra_count", "details"],
    )
    by_budget = {int(row["budget"]): row for row in table_rows if row["method"] == "lexical_bridgepack_path_order"}
    improves_span = all(_float(row.get("delta_vs_span_only")) is not None and _float(row.get("delta_vs_span_only")) > 0 for row in by_budget.values())
    improves_bm25_tfidf = all(
        _float(row.get("delta_vs_bm25")) is not None and _float(row.get("delta_vs_bm25")) > 0
        and _float(row.get("delta_vs_tfidf")) is not None and _float(row.get("delta_vs_tfidf")) > 0
        for row in by_budget.values()
    ) if by_budget else False
    best_budget = max(by_budget, key=lambda b: _float(by_budget[b].get("avg")) or float("-inf")) if by_budget else "TODO"
    delta_2048_1024 = (
        (_float(by_budget.get(2048, {}).get("avg")) or 0) - (_float(by_budget.get(1024, {}).get("avg")) or 0)
        if 2048 in by_budget and 1024 in by_budget else None
    )
    main_budget = 1024 if 1024 in by_budget else best_budget
    rec = [
        "# Exact-Token Budget Ablation Recommendation",
        "",
        f"1. Lexical-BridgePack improves over span-only at every completed budget: **{improves_span}**.",
        f"2. Lexical-BridgePack improves over BM25/TF-IDF at every completed budget: **{improves_bm25_tfidf}**.",
        f"3. Best completed budget by Lexical-BridgePack average: **{best_budget}**.",
        f"4. 2048 improves quality over 1024: **{delta_2048_1024 is not None and delta_2048_1024 > 0}**"
        f"{' (delta=' + _fmt(delta_2048_1024) + ')' if delta_2048_1024 is not None else ' (TODO)'}.",
        f"5. Main paper table budget: **{main_budget}**. Use 1024 as the main default setting and report other completed budgets as scaling unless the paper explicitly reframes the target budget.",
    ]
    _write_text(paper_dir / "04_budget_recommendation.md", "\n".join(rec))
    _write_text(paper_dir / "05_budget_claims.md", "\n".join([
        "# Budget Ablation Claims",
        "",
        "- Use only completed exact-token budget rows.",
        "- Do not reuse regex-budget 512/2048 values.",
        "- Treat 1024 as the official main setting unless a completed exact-token budget ablation justifies otherwise.",
    ]))


def parse_args() -> argparse.Namespace:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="")
    parser.add_argument("--tasks", default="qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,narrativeqa")
    parser.add_argument("--budgets", default="512,1024,2048")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--variants", default="span_only_source_order,bm25_chunk_source_order,tfidf_chunk_source_order,bridge_neighbor32_path_order,lexical_bridgepack_path_order")
    parser.add_argument("--out_dir", default=f"results/bridgepack_exact_budget_ablation_{stamp}")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seed != 0:
        raise ValueError("Only seed 0 is supported.")
    model = args.model_name_or_path or os.environ.get("QWEN_MODEL", "") or _model_from_exact()
    if not model:
        raise ValueError("QWEN_MODEL is not set and no exact-run Qwen model path could be found.")
    out_dir = Path(args.out_dir)
    budgets = [int(item) for item in _parse_list(args.budgets)]
    variants = _parse_list(args.variants)
    budget_dirs: Dict[int, Path] = {}
    for budget in budgets:
        if budget == 1024 and _latest_exact_1024() and args.max_samples == 0:
            budget_dirs[budget] = _latest_exact_1024()  # Reuse official exact-token 1024 run.
            continue
        run_dir = out_dir / f"budget_{budget}"
        cmd = [
            sys.executable,
            str(REPO_ROOT / "tools" / "run_bridgepack_exact_budget_repack.py"),
            "--model_name_or_path", model,
            "--tasks", args.tasks,
            "--budget", str(budget),
            "--max_samples", str(args.max_samples),
            "--variants", ",".join(variants),
            "--out_dir", str(run_dir),
            "--seed", str(args.seed),
        ]
        if args.resume:
            cmd.append("--resume")
        _run(cmd, out_dir / "logs" / f"budget_{budget}.log")
        budget_dirs[budget] = run_dir
    config = {
        "model_name_or_path": model,
        "tasks": _parse_list(args.tasks),
        "budgets": budgets,
        "variants": variants,
        "max_samples": args.max_samples,
        "reused_official_1024": 1024 in budget_dirs and budget_dirs[1024] == _latest_exact_1024(),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    _aggregate(out_dir, budgets, budget_dirs, variants, _parse_list(args.tasks))
    print(f"OUT_DIR={out_dir}")


if __name__ == "__main__":
    main()
