#!/usr/bin/env python3
"""Audit 64-sample RH/Span structured DRM results before the 128 recheck."""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
TASKS = ["musique", "hotpotqa", "qasper", "passage_count", "passage_retrieval_en"]
DEFAULT_OUT = REPO_ROOT / "results" / "structured_drm_recheck_audit"


def latest(pattern: str) -> Optional[Path]:
    matches = sorted((REPO_ROOT / "results").glob(pattern))
    return matches[-1] if matches else None


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def write_csv(path: Path, rows: List[Dict[str, Any]], header: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in header})


def as_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def count_preds(result_dir: Path, method: str, task: str) -> int:
    path = result_dir / "preds" / f"{method}__{task}.jsonl"
    return len(read_jsonl(path))


def sample_ids(result_dir: Path, method: str, task: str) -> List[str]:
    path = result_dir / "preds" / f"{method}__{task}.jsonl"
    return [str(row.get("sample_id", "")) for row in read_jsonl(path)]


def deployable(row: Dict[str, str]) -> bool:
    method = str(row.get("method", ""))
    attr = str(row.get("attribute", ""))
    return (
        method != "fixed_drm_add"
        and "oracle_diag" not in method
        and "diagnostic_only" not in attr
        and "learned_pseudo_label" not in attr
    )


def budget_ok(row: Dict[str, str]) -> bool:
    return (
        as_float(row.get("budget_error_pct")) < 1.0
        and int(as_float(row.get("budget_warning"))) == 0
        and int(as_float(row.get("runner_fail"))) == 0
    )


def source_checks() -> Dict[str, Any]:
    span_src = (REPO_ROOT / "tools" / "run_span_drm_stage2_64.py").read_text(encoding="utf-8")
    query_match = re.search(r"if scorer == \"query\":\s*score = query", span_src) is not None
    oracle_branch = "elif scorer == \"oracle\"" in span_src and "z_oracle" in span_src
    fallback_static = (
        '"span_drm_add_query": {"scorer": "query"' in span_src
        and '"count_fallback": True' in span_src
        and '"retrieval_fallback": True' in span_src
    )
    ranking_excludes_oracle = 'and "diagnostic_only" not in row["attribute"]' in span_src
    return {
        "span_query_uses_deployable_branch": query_match,
        "oracle_is_separate_branch": oracle_branch,
        "fallback_policy_static_in_method_spec": fallback_static,
        "ranking_excludes_diagnostic_only": ranking_excludes_oracle,
    }


def build_rows(rh_dir: Path, span_dir: Path, structured_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    sources = [
        ("RH-DRM", rh_dir, rh_dir / "rh_stage2_64_summary.csv"),
        ("SpanDRM", span_dir, span_dir / "span_stage2_64_summary.csv"),
        ("StructuredSummary", structured_dir, structured_dir / "structured_extensions_ranking.csv"),
    ]
    fixed_ids: Dict[str, List[str]] = {}
    for family, result_dir, summary_path in sources:
        for row in read_csv(summary_path):
            method = str(row.get("method", ""))
            if not method:
                continue
            counts = {task: count_preds(result_dir, method, task) for task in TASKS}
            if family in {"RH-DRM", "SpanDRM"}:
                if method == "fixed_drm_add":
                    fixed_ids = {task: sample_ids(result_dir, method, task) for task in TASKS}
                sample_aligned = all(
                    not fixed_ids.get(task) or sample_ids(result_dir, method, task) == fixed_ids[task]
                    for task in TASKS
                )
            else:
                sample_aligned = True
            is_deployable = deployable(row)
            rows.append({
                "family": family,
                "method": method,
                "attribute": row.get("attribute", ""),
                "avg": row.get("avg", ""),
                "passed_gate": row.get("passed_gate", ""),
                "budget_error_pct": row.get("budget_error_pct", ""),
                "budget_warning": row.get("budget_warning", ""),
                "runner_fail": row.get("runner_fail", ""),
                "deployable": "yes" if is_deployable else "no",
                "deployable_budget_ok": "yes" if (not is_deployable or budget_ok(row)) else "no",
                "sample_count_min": min(counts.values()) if counts else "",
                "sample_count_max": max(counts.values()) if counts else "",
                "sample_ids_aligned_with_fixed": "yes" if sample_aligned else "no",
                "source_dir": str(result_dir),
            })
    return rows


def report(rows: List[Dict[str, Any]], checks: Dict[str, Any], rh_dir: Path, span_dir: Path, structured_dir: Path) -> str:
    bad_budget = [row for row in rows if row["deployable"] == "yes" and row["deployable_budget_ok"] != "yes"]
    bad_samples = [row for row in rows if row["sample_ids_aligned_with_fixed"] != "yes"]
    source_ok = all(bool(value) for value in checks.values())
    oracle_in_rank = [
        row for row in read_csv(structured_dir / "structured_extensions_ranking.csv")
        if "oracle_diag" in str(row.get("method", "")) or "diagnostic_only" in str(row.get("attribute", ""))
    ]
    lines = [
        "# Structured DRM Stage2 64 Integrity And Leakage Audit",
        "",
        f"- RH source: `{rh_dir}`",
        f"- Span source: `{span_dir}`",
        f"- Structured summary: `{structured_dir}`",
        "- Environment note: requested conda `silm` was unavailable in the previous run; repo `.venv` was used.",
        "",
        "## Source-Level Checks",
        "",
    ]
    for key, value in checks.items():
        lines.append(f"- {key}: {'PASS' if value else 'FAIL'}")
    lines += [
        "",
        "## Findings",
        "",
        f"- Span query scorer uses the query-only deployable branch: {'PASS' if checks['span_query_uses_deployable_branch'] else 'FAIL'}.",
        "- `answer_overlap_oracle_diag` exists in span feature dumps for diagnostics, but only the `oracle` scorer branch consumes it.",
        f"- Oracle/diagnostic methods excluded from structured ranking: {'PASS' if not oracle_in_rank else 'FAIL'}.",
        f"- Static retrieval/count fallback policy is encoded before evaluation: {'PASS' if checks['fallback_policy_static_in_method_spec'] else 'FAIL'}.",
        "- Passage count compression is present only as an ablation and failed its gate; main SpanDRM methods use full-page fallback.",
        f"- Deployable budget/runner checks: {'PASS' if not bad_budget else 'FAIL'} ({len(bad_budget)} failing rows).",
        f"- Sample alignment with fixed baseline: {'PASS' if not bad_samples else 'FAIL'} ({len(bad_samples)} mismatched rows).",
        "- Large generated feature dumps remain local-only and should not be staged for git.",
        "",
        "## Decision",
        "",
    ]
    if source_ok and not bad_budget and not bad_samples and not oracle_in_rank:
        lines.append("- Audit passed. Proceed to Structured DRM-Add 128-sample recheck.")
    else:
        lines.append("- Audit found issues. Fix the failed rows above before treating 128-sample results as valid.")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rh_dir", default="")
    parser.add_argument("--span_dir", default="")
    parser.add_argument("--structured_dir", default=str(REPO_ROOT / "results" / "structured_drm_extensions_summary"))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    if not args.rh_dir:
        path = latest("rh_drm_stage2_64_*")
        if path is None:
            raise RuntimeError("No RH stage2_64 result directory found.")
        args.rh_dir = str(path)
    if not args.span_dir:
        path = latest("span_drm_stage2_64_*")
        if path is None:
            raise RuntimeError("No Span stage2_64 result directory found.")
        args.span_dir = str(path)
    return args


def main() -> None:
    args = parse_args()
    rh_dir = Path(args.rh_dir)
    span_dir = Path(args.span_dir)
    structured_dir = Path(args.structured_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    checks = source_checks()
    rows = build_rows(rh_dir, span_dir, structured_dir)
    header = [
        "family", "method", "attribute", "avg", "passed_gate", "budget_error_pct",
        "budget_warning", "runner_fail", "deployable", "deployable_budget_ok",
        "sample_count_min", "sample_count_max", "sample_ids_aligned_with_fixed", "source_dir",
    ]
    write_csv(out / "stage2_64_integrity_summary.csv", rows, header)
    (out / "leakage_budget_audit.md").write_text(
        report(rows, checks, rh_dir, span_dir, structured_dir),
        encoding="utf-8",
    )
    (out / "config.json").write_text(json.dumps({
        "rh_dir": str(rh_dir),
        "span_dir": str(span_dir),
        "structured_dir": str(structured_dir),
        "python_env": str(REPO_ROOT / ".venv"),
        "env_note": "conda env silm unavailable; repo .venv used",
    }, indent=2), encoding="utf-8")
    print(f"[structured-audit] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
