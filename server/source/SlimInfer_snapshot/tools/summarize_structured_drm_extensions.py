#!/usr/bin/env python3
"""Summarize FUP, RH-DRM, and SpanDRM stage2 diagnostic results."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
TASKS = ["musique", "hotpotqa", "qasper", "passage_count", "passage_retrieval_en"]
DEFAULT_FUP = REPO_ROOT / "results" / "fup_guarded_swap_stage2_64_20260601_051737"
DEFAULT_OUT = REPO_ROOT / "results" / "structured_drm_extensions_summary"

METHODS = [
    "fixed_drm_add",
    "fup_oracle_keepdrop",
    "rh_drm_add_static",
    "rh_drm_add_query",
    "rh_drm_add_learned",
    "span_drm_add_query",
    "span_drm_add_drm",
    "span_drm_add_fup",
    "span_drm_add_hybrid",
    "rh_then_span_drm_add",
]

FAMILY = {
    "fixed_drm_add": "baseline",
    "fup_oracle_keepdrop": "FUP",
    "rh_drm_add_static": "RH-DRM",
    "rh_drm_add_query": "RH-DRM",
    "rh_drm_add_learned": "RH-DRM",
    "span_drm_add_query": "SpanDRM",
    "span_drm_add_drm": "SpanDRM",
    "span_drm_add_fup": "SpanDRM",
    "span_drm_add_hybrid": "SpanDRM",
    "rh_then_span_drm_add": "RH-then-Span",
}


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, Any]], header: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in header})


def latest_result(pattern: str) -> Optional[Path]:
    matches = sorted((REPO_ROOT / "results").glob(pattern))
    return matches[-1] if matches else None


def as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def normalize_row(row: Dict[str, str], source_dir: Path) -> Dict[str, Any]:
    method = str(row.get("method", ""))
    out: Dict[str, Any] = {
        "method": method,
        "family": FAMILY.get(method, ""),
        "attribute": row.get("attribute", ""),
        "avg": round(as_float(row.get("avg")), 4),
        "passed_gate": row.get("passed_gate", ""),
        "gate_reasons": row.get("gate_reasons", ""),
        "source_dir": str(source_dir),
    }
    for task in TASKS:
        out[task] = round(as_float(row.get(task)), 3)
    return out


def find_method(rows: Iterable[Dict[str, str]], method: str) -> Optional[Dict[str, str]]:
    return next((row for row in rows if row.get("method") == method), None)


def build_rows(fup_dir: Path, rh_dir: Path, span_dir: Path) -> List[Dict[str, Any]]:
    fup_rows = read_csv(fup_dir / "fup_recheck_summary.csv") or read_csv(fup_dir / "recheck_summary.csv")
    rh_rows = read_csv(rh_dir / "rh_stage2_64_summary.csv")
    span_rows = read_csv(span_dir / "span_stage2_64_summary.csv")
    sources = [(fup_rows, fup_dir), (rh_rows, rh_dir), (span_rows, span_dir)]

    rows: List[Dict[str, Any]] = []
    for method in METHODS:
        found: Optional[Dict[str, str]] = None
        source: Optional[Path] = None
        for source_rows, source_dir in sources:
            found = find_method(source_rows, method)
            if found is not None:
                source = source_dir
                break
        if found is not None and source is not None:
            rows.append(normalize_row(found, source))

    baseline = next((row for row in rows if row["method"] == "fixed_drm_add"), None)
    if baseline is None:
        raise RuntimeError("fixed_drm_add is missing from the source summaries")
    for row in rows:
        row["delta_vs_fixed_drm_add"] = round(float(row["avg"]) - float(baseline["avg"]), 4)
        row["worst_task_drop_vs_fixed"] = round(
            min(float(row[task]) - float(baseline[task]) for task in TASKS),
            4,
        )
    return rows


def report(rows: List[Dict[str, Any]], fup_dir: Path, rh_dir: Path, span_dir: Path) -> str:
    by_name = {row["method"]: row for row in rows}
    baseline = by_name["fixed_drm_add"]
    rh_names = ["rh_drm_add_static", "rh_drm_add_query", "rh_drm_add_learned"]
    span_names = ["span_drm_add_query", "span_drm_add_drm", "span_drm_add_fup", "span_drm_add_hybrid"]
    rh_pass = [name for name in rh_names if by_name.get(name, {}).get("passed_gate") == "yes"]
    span_pass = [name for name in span_names if by_name.get(name, {}).get("passed_gate") == "yes"]
    fup_pass = by_name.get("fup_oracle_keepdrop", {}).get("passed_gate") == "yes"
    deployable = [
        row for row in rows
        if row["method"] != "fixed_drm_add" and "diagnostic_only" not in str(row.get("attribute", ""))
    ]
    best = max(deployable, key=lambda row: float(row["avg"])) if deployable else baseline

    if rh_pass and span_pass:
        decision = "Both structured directions pass; test an RH-then-Span hybrid next."
    elif rh_pass:
        decision = "RH-DRM is the main structural-upgrade candidate; consider a larger Stage2 or full run."
    elif span_pass:
        decision = "SpanDRM is the granularity-upgrade candidate; consider a larger Stage2 or full run."
    elif fup_pass:
        decision = "Only the learned FUP selector passes; keep fixed DRM-Add as mainline and FUP as secondary."
    else:
        decision = "No new structure passes; keep fixed DRM-Add as the final mainline and treat RH/Span as negative diagnostics."

    lines = [
        "# Structured DRM Extensions Summary",
        "",
        f"- Fixed DRM-Add baseline Avg: {baseline['avg']}",
        f"- Best non-baseline method: `{best['method']}` Avg {best['avg']} ({best['delta_vs_fixed_drm_add']:+.4f})",
        "- Environment: requested conda `silm` was unavailable; the repo `.venv` was used.",
        "- Scope: 64 samples per task only. No full LongBench or latency run was performed.",
        "- SpanDRM uses packed ordinary FullKV contexts as a granularity diagnostic, not a sparse physical-KV implementation.",
        "",
        "## Ranking",
        "",
    ]
    for row in sorted(deployable, key=lambda item: float(item["avg"]), reverse=True):
        lines.append(
            f"- `{row['method']}` ({row['family']}): Avg {row['avg']}, "
            f"delta {row['delta_vs_fixed_drm_add']:+.4f}, gate {row['passed_gate'] or 'n/a'}"
        )
    lines += [
        "",
        "## Decision",
        "",
        f"- RH-DRM methods passing gate: {rh_pass if rh_pass else 'none'}.",
        f"- SpanDRM methods passing gate: {span_pass if span_pass else 'none'}.",
        f"- FUP oracle-keepdrop passes gate: {fup_pass}.",
        f"- {decision}",
        "",
        "## Sources",
        "",
        f"- FUP: `{fup_dir}`",
        f"- RH-DRM: `{rh_dir}`",
        f"- SpanDRM: `{span_dir}`",
    ]
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fup_dir", default=str(DEFAULT_FUP))
    parser.add_argument("--rh_dir", default="")
    parser.add_argument("--span_dir", default="")
    parser.add_argument("--output_dir", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    if not args.rh_dir:
        path = latest_result("rh_drm_stage2_64_*")
        if path is None:
            raise RuntimeError("no RH-DRM result directory found; pass --rh_dir")
        args.rh_dir = str(path)
    if not args.span_dir:
        path = latest_result("span_drm_stage2_64_*")
        if path is None:
            raise RuntimeError("no SpanDRM result directory found; pass --span_dir")
        args.span_dir = str(path)
    return args


def main() -> None:
    args = parse_args()
    fup_dir, rh_dir, span_dir = Path(args.fup_dir), Path(args.rh_dir), Path(args.span_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = build_rows(fup_dir, rh_dir, span_dir)
    ranking = sorted(rows, key=lambda row: float(row["avg"]), reverse=True)
    header = [
        "method", "family", "attribute", "avg", "delta_vs_fixed_drm_add",
        *TASKS, "worst_task_drop_vs_fixed", "passed_gate", "gate_reasons", "source_dir",
    ]
    write_csv(out / "structured_extensions_ranking.csv", ranking, header)
    (out / "structured_extensions_report.md").write_text(
        report(rows, fup_dir, rh_dir, span_dir),
        encoding="utf-8",
    )
    (out / "config.json").write_text(json.dumps({
        "fup_dir": str(fup_dir),
        "rh_dir": str(rh_dir),
        "span_dir": str(span_dir),
        "methods": METHODS,
        "python_env": str(REPO_ROOT / ".venv"),
        "env_note": "conda env silm unavailable; repo .venv used",
    }, indent=2), encoding="utf-8")
    print(f"[structured-summary] wrote {len(rows)} methods -> {out}", flush=True)


if __name__ == "__main__":
    main()
