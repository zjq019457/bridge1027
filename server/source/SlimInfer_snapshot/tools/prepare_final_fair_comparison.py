#!/usr/bin/env python3
"""Prepare the final fair-comparison table without launching full LongBench."""
from __future__ import annotations

import csv
import datetime as dt
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results"
OUT = RESULTS / "final_fair_comparison_prep"

FULL_SOURCE = RESULTS / "final_drm_add_mainline" / "main_table_full_longbench.csv"
RH64_SOURCE = RESULTS / "rh_drm_stage2_64_20260604_072019" / "rh_stage2_64_summary.csv"
SPAN64_SOURCE = RESULTS / "span_drm_stage2_64_20260604_181036" / "span_stage2_64_summary.csv"
STRUCTURED128_SOURCE = (
    RESULTS / "structured_drm_stage2_128_20260605_070455" / "structured_stage2_128_summary.csv"
)
AB128_SOURCE = RESULTS / "ab_calibrated_recheck_128_20260608_071200" / "ab_recheck_128_summary.csv"
DRM_AUDIT_SOURCE = RESULTS / "rescuekv_drm_add_full_longbench_20260523_233633" / "audit_summary.csv"

TASKS = [
    "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique",
    "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa",
    "samsum", "lsht", "passage_count", "passage_retrieval_en", "lcc", "repobench-p",
]
STRUCTURED_CANDIDATES = ["span_drm_add_query", "rh_then_span_drm_add_query"]
STRUCTURED_FULL_NAMES = {
    "rh_drm_add_static_fixed_pool", "rh_drm_add_query_fixed_pool",
    "span_drm_add_query", "span_drm_add_hybrid",
    "rh_then_span_drm_add_query", "rh_then_span_drm_add_hybrid",
}

TABLE_FIELDS = [
    "Method", "Role", "Evaluation", "Budget setting", "Avg",
    "Delta_vs_SlimInfer", "Delta_vs_SlimInfer_Larger", "Delta_vs_fixed_DRM_Add",
    "Budget_error_mean", "Budget_error_max", "Budget_warning", "Runner_fail",
    "Use_in_main_table", "Notes",
]


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def to_float(value: Any) -> Optional[float]:
    try:
        if value in (None, "", "n/a", "N/A"):
            return None
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def fmt(value: Any, digits: int = 3, signed: bool = False) -> str:
    number = to_float(value)
    if number is None:
        return ""
    return f"{number:+.{digits}f}" if signed else f"{number:.{digits}f}"


def row_by(rows: Iterable[Dict[str, str]], key: str, value: str) -> Dict[str, str]:
    return next((row for row in rows if str(row.get(key, "")) == value), {})


def structured_budget(method: str) -> Dict[str, str]:
    rows = read_csv(STRUCTURED128_SOURCE.parent / "structured_stage2_128_budget_audit.csv")
    values = [
        abs(float(row.get("budget_error_pct", 0) or 0))
        for row in rows if row.get("method") == method
    ]
    warnings = sum(int(float(row.get("budget_warning", 0) or 0)) for row in rows if row.get("method") == method)
    return {
        "mean": fmt(sum(values) / len(values) if values else None, 4),
        "max": fmt(max(values) if values else None, 4),
        "warning": "No" if not warnings else f"Yes ({warnings})",
    }


def find_structured_full_results() -> List[str]:
    found: List[str] = []
    for path in RESULTS.rglob("*.csv"):
        lower = str(path).lower()
        if "full_longbench" not in lower or any(token in lower for token in ("stage2_64", "stage2_128")):
            continue
        for row in read_csv(path):
            method = str(row.get("method", row.get("Method", ""))).strip()
            if method in STRUCTURED_FULL_NAMES and method not in found:
                found.append(method)
    return found


def classify_artifact(path: Path) -> Dict[str, str]:
    rel = path.relative_to(REPO_ROOT).as_posix()
    lower = rel.lower()
    if "calibration" in lower or "budget_calibration" in lower:
        evaluation = "calibration / budget-only"
    elif "128" in lower:
        evaluation = "128-sample diagnostic"
    elif "64" in lower:
        evaluation = "64-sample diagnostic"
    elif "full_longbench" in lower or path.name in {"main_table_summary.csv", "main_table_ranking.csv"}:
        evaluation = "Full LongBench"
    else:
        evaluation = "diagnostic / other"
    negative_tokens = (
        "ab_", "ab-", "boundary", "fup", "guarded_swap", "ac_gated",
        "block_system", "structural_drm_add_diagnostic",
    )
    role = "negative diagnostic" if any(token in lower for token in negative_tokens) else "candidate/baseline evidence"
    if "final_drm_add_mainline" in lower or "rescuekv_drm_add_full_longbench" in lower:
        role = "authoritative full baseline"
    if "structured_drm_stage2_128" in lower or "rh_drm_stage2_64" in lower or "span_drm_stage2_64" in lower:
        role = "structured candidate diagnostic"
    return {"source_file": rel, "evaluation": evaluation, "role_class": role}


def discover_result_artifacts() -> List[Path]:
    exact = {
        "main_table_summary.csv", "main_table_ranking.csv", "final_mainline_report.md",
        "structured_extensions_report.md", "structured_extensions_ranking.csv",
        "structured_stage2_128_summary.csv", "structured_stage2_128_ranking.csv",
        "span_stage2_64_summary.csv", "rh_stage2_64_summary.csv",
        "ab_recheck_128_summary.csv", "report_ab_recheck_128.md",
    }
    paths = set()
    for path in RESULTS.rglob("*"):
        if not path.is_file():
            continue
        if OUT in path.parents:
            continue
        if path.name in exact or path.name.endswith("_summary.csv") or path.name.endswith("_ranking.csv"):
            paths.add(path)
    return sorted(paths)


def build_diagnostic_inventory() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in discover_result_artifacts():
        meta = classify_artifact(path)
        rows.append({
            **meta, "method": "", "avg": "", "delta_vs_local_fixed": "",
            "budget_error_pct": "", "budget_warning": "", "runner_fail": "",
            "notes": "artifact-level recursive inventory entry",
        })

    normalized_sources = [FULL_SOURCE, RH64_SOURCE, SPAN64_SOURCE, STRUCTURED128_SOURCE, AB128_SOURCE]
    for path in normalized_sources:
        meta = classify_artifact(path)
        for source_row in read_csv(path):
            rows.append({
                **meta,
                "method": source_row.get("method", source_row.get("Method", "")),
                "avg": source_row.get(
                    "avg",
                    source_row.get("Avg", source_row.get("avg_score", source_row.get("LongBench_avg", ""))),
                ),
                "delta_vs_local_fixed": source_row.get(
                    "delta_vs_baseline", source_row.get("delta_vs_fixed", "")
                ),
                "budget_error_pct": source_row.get(
                    "budget_error_pct",
                    source_row.get("budget_mean_abs_error", source_row.get("mean_abs_error", "")),
                ),
                "budget_warning": source_row.get("budget_warning", ""),
                "runner_fail": source_row.get("runner_fail", ""),
                "notes": "normalized target-method row; diagnostics are never promoted to full",
            })
    return rows


def build_main_table() -> List[Dict[str, Any]]:
    full_rows = read_csv(FULL_SOURCE)
    structured_rows = read_csv(STRUCTURED128_SOURCE)
    full_map = {row["Method"]: row for row in full_rows}
    slim = float(full_map["SlimInfer"]["Avg"])
    larger = float(full_map["SlimInfer larger K+M"]["Avg"])
    fixed = float(full_map["DRM-Add"]["Avg"])
    full_specs = [
        ("FullKV", "FullKV", "external baseline / uncompressed upper bound", "Full KV"),
        ("SlimInfer original", "SlimInfer", "external baseline", "Original SlimInfer K"),
        ("SlimInfer larger K+M", "SlimInfer larger K+M", "external budget-matched baseline", "Larger K+M"),
        ("Random-Add", "Random-Add", "external/random rescue control", "Matched K+M additive budget"),
        ("DeltaO-Add", "DeltaO-Add", "external functional rescue control", "Matched K+M additive budget"),
        ("fixed DRM-Add", "DRM-Add", "base Ours / old main", "Matched K+M additive budget"),
    ]
    rows: List[Dict[str, Any]] = []
    for display, source_name, role, budget in full_specs:
        source = full_map[source_name]
        avg = float(source["Avg"])
        audit_note = ""
        warning = "No token-mismatch warning reported"
        if source_name == "DRM-Add":
            drm_audit = row_by(read_csv(DRM_AUDIT_SOURCE), "method", "drm_add")
            if str(drm_audit.get("fallback_used", "")).lower() == "true":
                warning = "No target-error metric; aggregate audit has fallback_used=True"
                audit_note = " Historical full audit lacks mean/max target-token error."
        rows.append({
            "Method": display,
            "Role": role,
            "Evaluation": "Full LongBench",
            "Budget setting": budget,
            "Avg": fmt(avg, 2),
            "Delta_vs_SlimInfer": fmt(avg - slim, 2, signed=True),
            "Delta_vs_SlimInfer_Larger": fmt(avg - larger, 2, signed=True),
            "Delta_vs_fixed_DRM_Add": fmt(avg - fixed, 2, signed=True),
            "Budget_error_mean": "N/A",
            "Budget_error_max": "N/A",
            "Budget_warning": warning,
            "Runner_fail": "0 (3950/3950 complete)",
            "Use_in_main_table": "Yes",
            "Notes": f"Authoritative full result from {FULL_SOURCE.relative_to(REPO_ROOT)}.{audit_note}",
        })

    diagnostic_specs = [
        ("RH-DRM", "rh_drm_add_query_fixed_pool", "head-aware rescue"),
        ("SpanDRM", "span_drm_add_query", "span-level evidence rescue"),
        ("RH-then-Span", "rh_then_span_drm_add_hybrid", "final structured candidate"),
    ]
    fixed_128 = float(row_by(structured_rows, "method", "fixed_drm_add").get("avg", 0) or 0)
    for display, method, role in diagnostic_specs:
        source = row_by(structured_rows, "method", method)
        budget = structured_budget(method)
        avg = float(source.get("avg", 0) or 0)
        rows.append({
            "Method": display,
            "Role": role,
            "Evaluation": "128-sample diagnostic",
            "Budget setting": "Matched fixed DRM-Add selected-token budget",
            "Avg": fmt(avg, 3),
            "Delta_vs_SlimInfer": "",
            "Delta_vs_SlimInfer_Larger": "",
            "Delta_vs_fixed_DRM_Add": "",
            "Budget_error_mean": budget["mean"],
            "Budget_error_max": budget["max"],
            "Budget_warning": budget["warning"],
            "Runner_fail": source.get("runner_fail", ""),
            "Use_in_main_table": "No, diagnostic only until full is available",
            "Notes": (
                f"Need full LongBench before final claim. "
                f"128-sample delta vs fixed DRM-Add: {avg - fixed_128:+.3f}."
            ),
        })
    return rows


def markdown_table(rows: Sequence[Dict[str, Any]], fields: Sequence[str]) -> str:
    def clean(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(clean(row.get(field, "")) for field in fields) + " |")
    return "\n".join(lines)


def write_scripts() -> None:
    tasks = ",".join(TASKS)
    methods = ",".join(STRUCTURED_CANDIDATES)
    common = f'''#!/usr/bin/env bash
set -euo pipefail
cd "{REPO_ROOT}"

MODEL_PATH="${{MODEL_PATH:-/home/yu-bai/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28}}"
LONGBENCH_DIR="${{LONGBENCH_DIR:-/home/yu-bai/Desktop/jiaqizhang/HeadKV/data/LongBench}}"
PROFILE_DIR="${{PROFILE_DIR:-{REPO_ROOT}/profiles/retrieval_head_profiles}}"
TASKS="{tasks}"
METHODS="{methods}"
# 100000 is an upper bound; the loader slices the complete local LongBench test split.
SAMPLES_PER_TASK="${{SAMPLES_PER_TASK:-100000}}"
export PYTHONPATH="{REPO_ROOT}:${{PYTHONPATH:-}}"
export PYTORCH_CUDA_ALLOC_CONF="${{PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}}"
'''
    run_body = common + f'''
if [[ "${{RUN_STRUCTURED_FULL:-0}}" != "1" ]]; then
  echo "Prepared only. Set RUN_STRUCTURED_FULL=1 to launch structured full LongBench."
  exit 0
fi

STAMP="${{STRUCTURED_FULL_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}}"
OUT_DIR="${{STRUCTURED_FULL_OUT_DIR:-{RESULTS}/structured_drm_full_longbench_${{STAMP}}}}"
mkdir -p "$OUT_DIR"
printf '%s\\n' "$OUT_DIR" > "{OUT}/latest_structured_full_dir.txt"

scripts/run_with_silm.sh tools/run_structured_drm_stage2_128.py \\
  --output_dir "$OUT_DIR" \\
  --model_path "$MODEL_PATH" \\
  --longbench_dir "$LONGBENCH_DIR" \\
  --profile_dir "$PROFILE_DIR" \\
  --tasks "$TASKS" \\
  --samples_per_task "$SAMPLES_PER_TASK" \\
  --evaluation_mode full \\
  --methods "$METHODS" \\
  --skip_existing 2>&1 | tee -a "$OUT_DIR/run.log"
'''
    resume_body = common + f'''
OUT_DIR="${{STRUCTURED_FULL_OUT_DIR:-}}"
if [[ -z "$OUT_DIR" && -f "{OUT}/latest_structured_full_dir.txt" ]]; then
  OUT_DIR="$(cat "{OUT}/latest_structured_full_dir.txt")"
fi
if [[ -z "$OUT_DIR" || ! -d "$OUT_DIR" ]]; then
  echo "No structured full output directory found; set STRUCTURED_FULL_OUT_DIR." >&2
  exit 2
fi

scripts/run_with_silm.sh tools/run_structured_drm_stage2_128.py \\
  --output_dir "$OUT_DIR" \\
  --model_path "$MODEL_PATH" \\
  --longbench_dir "$LONGBENCH_DIR" \\
  --profile_dir "$PROFILE_DIR" \\
  --tasks "$TASKS" \\
  --samples_per_task "$SAMPLES_PER_TASK" \\
  --evaluation_mode full \\
  --methods "$METHODS" \\
  --skip_existing \\
  --overwrite_failed 2>&1 | tee -a "$OUT_DIR/resume.log"
'''
    for name, body in (
        ("run_structured_full_if_needed.sh", run_body),
        ("resume_structured_full_if_needed.sh", resume_body),
    ):
        path = OUT / name
        path.write_text(body, encoding="utf-8")
        path.chmod(0o755)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    structured_full = find_structured_full_results()
    table = build_main_table()
    inventory = build_diagnostic_inventory()
    found_set = set(structured_full)
    missing = []
    if not found_set.intersection({"rh_drm_add_static_fixed_pool", "rh_drm_add_query_fixed_pool"}):
        missing.append("RH-DRM: rh_drm_add_static_fixed_pool or rh_drm_add_query_fixed_pool")
    if not found_set.intersection({"span_drm_add_query", "span_drm_add_hybrid"}):
        missing.append("SpanDRM: span_drm_add_query or span_drm_add_hybrid")
    if not found_set.intersection({"rh_then_span_drm_add_query", "rh_then_span_drm_add_hybrid"}):
        missing.append("RH-then-Span: rh_then_span_drm_add_query or rh_then_span_drm_add_hybrid")

    write_csv(OUT / "final_fair_main_table.csv", table, TABLE_FIELDS)
    (OUT / "final_fair_main_table.md").write_text(
        "# Final Fair Main Table (Current Evidence)\n\n"
        "Full and diagnostic results are intentionally kept in the same status table but never share deltas.\n\n"
        + markdown_table(table, TABLE_FIELDS) + "\n",
        encoding="utf-8",
    )
    diagnostic_fields = [
        "source_file", "evaluation", "role_class", "method", "avg",
        "delta_vs_local_fixed", "budget_error_pct", "budget_warning", "runner_fail", "notes",
    ]
    write_csv(OUT / "diagnostic_results_summary.csv", inventory, diagnostic_fields)
    (OUT / "full_missing_methods.txt").write_text(
        "\n".join(missing)
        + "\n\nRecommended full candidates:\n"
        + "\n".join(STRUCTURED_CANDIDATES)
        + "\n",
        encoding="utf-8",
    )

    status = f"""# Final Fair Main Table Status

Generated: {dt.datetime.now().isoformat(timespec="seconds")}

## Required Answers

1. **Current final full main table available?** No. The six baseline/base-Ours rows form a complete full table, but Structured DRM candidates are still diagnostic-only.
2. **Methods with Full LongBench:** FullKV, SlimInfer original, SlimInfer larger K+M, Random-Add, DeltaO-Add, and fixed DRM-Add (3950/3950 each).
3. **Methods with diagnostic evidence only:** RH-DRM, SpanDRM, and RH-then-Span have 128-sample results; RH-DRM and SpanDRM also have 64-sample diagnostics. AB/FUP/Guarded Swap/AC-Gated and selector families remain negative diagnostics.
4. **Can RH-DRM / SpanDRM / RH-then-Span enter the final main table?** No. Their Avg values must not be presented as full LongBench until a complete 17-task run finishes.
5. **Current final Ours label:** fixed DRM-Add. Structured DRM-Add is a promoted candidate, not yet the final full-backed method.
6. **Next missing full runs:** `span_drm_add_query` and `rh_then_span_drm_add_hybrid`. The latter passed the 128-sample gate; the query RH-then-Span variant did not.
7. **Fair comparison with SlimInfer original?** Yes for fixed DRM-Add and all existing full baselines, using the same 17 tasks, test split, model, prompt/eval pipeline, and generation setup.
8. **Budget-fair comparison with SlimInfer larger K+M?** Yes for the existing full fixed-page table. The historical full runner did not report normalized mean/max target-token errors, so those cells are N/A rather than invented zeros.
9. **Budget warning / runner fail?** No runner failures in the complete 3950/3950 full rows. The historical fixed DRM-Add aggregate audit records `fallback_used=True`; it does not expose a target-token error metric. Structured 128 candidates report zero warnings and zero runner failures.
10. **Was any partial incorrectly treated as full?** No. All Structured values are explicitly labeled `128-sample diagnostic`, and full-only delta columns are blank for them.

## Decision

Use fixed DRM-Add as the current paper Ours. Run the two prepared Structured candidates only when `RUN_STRUCTURED_FULL=1`; until completion, do not replace the full-backed claim.

## Inventory

- Indexed result artifacts: {len(discover_result_artifacts())}
- Normalized inventory rows: {len(inventory)}
- Structured full methods found: {structured_full or "none"}
- Authoritative full source: `{FULL_SOURCE.relative_to(REPO_ROOT)}`
"""
    (OUT / "final_fair_main_table_status.md").write_text(status, encoding="utf-8")

    checklist = """# Full Fairness Checklist

- [x] Same Qwen2.5-7B-Instruct model checkpoint.
- [x] Same 17 LongBench tasks and local test split.
- [x] Same LongBench prompt templates and task-specific generation lengths.
- [x] Greedy generation (`temperature=0`, `top_p=1.0`).
- [x] Fixed SlimInfer active construction and add-only structured rescue.
- [x] Per-sample selected-token budget audit emitted by the structured runner.
- [x] `skip_existing` and `overwrite_failed` resume paths prepared.
- [x] Full output uses `results/structured_drm_full_longbench_YYYYMMDD_HHMMSS/`.
- [ ] Run both Structured candidates to complete sample counts on all 17 tasks.
- [ ] Verify every task status is complete and `runner_fail=0`.
- [ ] Verify budget warning is zero, mean budget error is <3%, and every method-task error is <5%.
- [ ] Merge completed Structured rows with the existing six authoritative full baselines.
- [ ] Keep large predictions, selected contexts, feature dumps, audits, logs, and model/cache files out of git.

## Known Generalization Caveat

The frozen RH profile was calibrated on five held-out diagnostic tasks. The full script reuses it unchanged on all 17 tasks to avoid test-time fitting. This is leakage-safe, but its cross-task generalization remains an empirical full-run question.
"""
    (OUT / "full_fairness_checklist.md").write_text(checklist, encoding="utf-8")
    write_scripts()

    config = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "run_structured_full": False,
        "run_gate_env": "RUN_STRUCTURED_FULL=1",
        "authoritative_full_source": str(FULL_SOURCE),
        "diagnostic_sources": [str(RH64_SOURCE), str(SPAN64_SOURCE), str(STRUCTURED128_SOURCE), str(AB128_SOURCE)],
        "model": "/home/yu-bai/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28",
        "longbench_dir": "/home/yu-bai/Desktop/jiaqizhang/HeadKV/data/LongBench",
        "split": "test/full local LongBench files",
        "tasks": TASKS,
        "generation": {"temperature": 0, "top_p": 1.0, "max_new_tokens": "configs/dataset2maxlen.json"},
        "structured_candidates": STRUCTURED_CANDIDATES,
        "structured_full_found": structured_full,
        "baseline_full_complete": True,
        "artifact_count_indexed": len(discover_result_artifacts()),
        "normalized_inventory_rows": len(inventory),
        "large_artifacts_committed": False,
    }
    (OUT / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
