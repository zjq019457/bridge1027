#!/usr/bin/env python3
"""Aggregate structured rescue experiment directories."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def _read_scores(result_dir: Path) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for path in result_dir.rglob("longbench_scores.csv"):
        try:
            with path.open(encoding="utf-8") as f:
                vals = [float(r["score"]) for r in csv.DictReader(f) if r.get("status") == "ok" and r.get("score")]
            if vals:
                method = path.parent.name if path.parent.name != "runs" else path.parent.parent.name
                scores[f"{method}:macro"] = sum(vals) / len(vals)
                for r in csv.DictReader(path.open(encoding="utf-8")):
                    if r.get("status") == "ok":
                        scores[f"{method}:{r.get('dataset', '')}"] = float(r["score"])
        except Exception:
            pass
    for path in result_dir.rglob("longbench_scores.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            for task, val in payload.items():
                if isinstance(val, (int, float)):
                    scores[f"{path.parent.name}:{task}"] = float(val)
        except Exception:
            pass
    for path in result_dir.rglob("*summary*.csv"):
        try:
            with path.open(encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    method = row.get("method", path.parent.name)
                    score = row.get("task_macro_avg", row.get("score", ""))
                    if score not in ("", None):
                        scores[str(method)] = float(score)
        except Exception:
            pass
    return scores


def _mean(vals: List[float]) -> float:
    return sum(vals) / max(1, len(vals))


def summarize(root: Path, output: Path) -> None:
    method_scores: Dict[str, List[float]] = defaultdict(list)
    task_breakdown: List[Dict[str, Any]] = []
    budget_rows: List[Dict[str, Any]] = []
    teacher_rows: List[Dict[str, Any]] = []

    method_dirs = []
    if (root / "runs").is_dir():
        method_dirs = [p for p in (root / "runs").iterdir() if p.is_dir()]
    else:
        method_dirs = [p for p in root.iterdir() if p.is_dir() and p.name not in {"summary", "audit", "tie_postmortem"}]

    for child in sorted(method_dirs):
        method = child.name
        scores = _read_scores(child if child.parent.name == "runs" else root / "runs" / method if (root / "runs" / method).exists() else child)
        if not scores and child.exists():
            scores = _read_scores(child)
        if scores:
            macro = _mean(list(scores.values()))
            method_scores[method].append(macro)
            for task, sc in scores.items():
                task_breakdown.append({"method": method, "task": task, "score": sc})
        for audit in list(child.rglob("rescue_audit.jsonl")) + list((root / "audit").glob(f"{method}.jsonl")):
            for row in _load_jsonl(audit):
                budget_rows.append({
                    "method": method,
                    "task": row.get("task", ""),
                    "sample_id": row.get("sample_id", ""),
                    "budget_error_pct": row.get("budget_error_pct", row.get("selected_token_error_pct", "")),
                    "budget_warning": row.get("budget_warning", ""),
                    "add_only_contract_ok": row.get("add_only_contract_ok", ""),
                    "active_evicted_count": row.get("active_evicted_count", 0),
                })
                if row.get("teacher_coverage") not in ("", None):
                    teacher_rows.append({
                        "method": method,
                        "teacher_coverage": row.get("teacher_coverage"),
                        "selector_fallback_used": row.get("selector_fallback_used", ""),
                    })

    summary = []
    for method, vals in sorted(method_scores.items()):
        summary.append({
            "method": method,
            "task_macro_avg": round(_mean(vals), 4),
            "n_result_dirs": len(vals),
        })
    slim = next((r["task_macro_avg"] for r in summary if r["method"] == "fixed_drm_add"), None)
    fixed = slim
    for r in summary:
        r["delta_vs_sliminfer"] = round(r["task_macro_avg"] - slim, 4) if slim is not None else ""
        r["delta_vs_fixed_drm_add"] = round(r["task_macro_avg"] - fixed, 4) if fixed is not None else ""

    output.mkdir(parents=True, exist_ok=True)
    def write_csv(name: str, rows: List[Dict[str, Any]], fields: List[str]) -> None:
        path = output / name
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for row in rows:
                w.writerow({k: row.get(k, "") for k in fields})

    write_csv("summary.csv", summary, ["method", "task_macro_avg", "n_result_dirs", "delta_vs_sliminfer", "delta_vs_fixed_drm_add"])
    write_csv("delta_vs_sliminfer.csv", [r for r in summary if r.get("delta_vs_sliminfer") != ""], ["method", "task_macro_avg", "delta_vs_sliminfer"])
    write_csv("delta_vs_fixed_drm_add.csv", [r for r in summary if r.get("delta_vs_fixed_drm_add") != ""], ["method", "task_macro_avg", "delta_vs_fixed_drm_add"])
    write_csv("budget_audit.csv", budget_rows, ["method", "task", "sample_id", "budget_error_pct", "budget_warning", "add_only_contract_ok", "active_evicted_count"])
    if teacher_rows:
        write_csv("teacher_coverage.csv", teacher_rows, ["method", "teacher_coverage", "selector_fallback_used"])
    write_csv("method_task_breakdown.csv", task_breakdown, ["method", "task", "score"])
    print(json.dumps({"output": str(output), "methods": len(summary), "budget_rows": len(budget_rows)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_root", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    summarize(Path(args.result_root), Path(args.output_dir))


if __name__ == "__main__":
    main()
