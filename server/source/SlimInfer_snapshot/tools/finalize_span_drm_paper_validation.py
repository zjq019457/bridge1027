#!/usr/bin/env python3
"""Finalize paper-facing SpanDRM validation tables, claims, and run scripts."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.span_drm_validation_lib import (  # noqa: E402
    LONGBENCH_DIR,
    MODEL_PATH,
    PAPER_ROOT,
    SANITY_TASKS,
    STRUCTURED_ROOT,
    markdown_table,
    read_csv,
    read_json,
    write_csv,
    write_json,
)


def find_bootstrap(rows: Sequence[Mapping[str, Any]], comparison: str, scheme: str = "task_stratified") -> Mapping[str, Any]:
    return next(
        (row for row in rows if row.get("comparison") == comparison and row.get("bootstrap_type") == scheme),
        {},
    )


def method_row(rows: Sequence[Mapping[str, Any]], method: str) -> Mapping[str, Any]:
    return next((row for row in rows if row.get("method") == method or row.get("Method") == method), {})


def bool_value(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def write_run_scripts(out: Path) -> None:
    scripts = {
        "run_physical_kv_span_sanity.sh": (
            f"scripts/run_with_silm.sh tools/run_span_drm_physical_kv.py "
            f"--output_dir {out.relative_to(REPO_ROOT)} --samples_per_task 32"
        ),
        "run_latency_memory_eval.sh": (
            f"scripts/run_with_silm.sh tools/run_span_drm_latency_memory.py "
            f"--output_dir {out.relative_to(REPO_ROOT)} --samples_per_task 32 --decode_tokens 8"
        ),
        "run_bootstrap_significance.sh": (
            f"scripts/run_with_silm.sh tools/run_span_drm_static_validation.py "
            f"--output_dir {out.relative_to(REPO_ROOT)} --bootstrap_n 10000 --seed 20260609"
        ),
        "run_case_study.sh": (
            f"scripts/run_with_silm.sh tools/run_span_drm_static_validation.py "
            f"--output_dir {out.relative_to(REPO_ROOT)} --bootstrap_n 10000 --seed 20260609"
        ),
    }
    for name, command in scripts.items():
        path = out / name
        path.write_text(
            "#!/usr/bin/env bash\nset -euo pipefail\n"
            f"cd {REPO_ROOT}\n{command}\n",
            encoding="utf-8",
        )
        path.chmod(0o755)


def updated_main_table(out: Path, bootstrap: Sequence[Mapping[str, Any]], physical: Sequence[Mapping[str, Any]], latency: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    source = read_csv(PAPER_ROOT / "paper_main_table_full.csv")
    fields = list(source[0].keys()) + [
        "Task_stratified_CI_vs_fixed_DRM_Add",
        "Physical_KV_validation",
        "Latency_validation",
    ]
    fixed_ci = find_bootstrap(bootstrap, "SpanDRM vs fixed DRM-Add")
    physical_span = method_row(physical, "span_drm_physical_kv_retention")
    latency_span = method_row(latency, "SpanDRM")
    rows: List[Dict[str, Any]] = []
    for original in source:
        row = dict(original)
        row["Task_stratified_CI_vs_fixed_DRM_Add"] = ""
        row["Physical_KV_validation"] = ""
        row["Latency_validation"] = ""
        if row.get("Method") == "SpanDRM":
            row["Role"] = "final Ours"
            row["Task_stratified_CI_vs_fixed_DRM_Add"] = (
                f"[{fixed_ci.get('ci95_low')}, {fixed_ci.get('ci95_high')}]"
                if fixed_ci else "not available"
            )
            row["Physical_KV_validation"] = (
                str(physical_span.get("implementation_status", "not run")) if physical_span else "not run"
            )
            row["Latency_validation"] = (
                f"diagnostic text-packed E2E={latency_span.get('end_to_end_time_ms')} ms"
                if latency_span else "not run"
            )
            row["Notes"] = (
                "Complete full quality result under exact matched selected-token targets. "
                "The full-average gain is modest and its paired bootstrap CI crosses zero. "
                "Physical validation is an order-preserving DynamicCache repack approximation, not a native sparse-KV implementation."
            )
        elif row.get("Method") == "RH-then-Span":
            row["Role"] = "full failed structured variant"
            row["Use_in_claim"] = "No"
            row["Notes"] = (
                "Complete full diagnostic row, but below fixed DRM-Add and therefore not final Ours."
            )
        rows.append(row)
    write_csv(out / "updated_paper_main_table_full.csv", rows, fields)
    (out / "updated_paper_main_table_full.md").write_text(
        "# Updated Paper Main Table\n\n"
        + markdown_table(rows, fields)
        + "\n\nSpanDRM remains final Ours by full LongBench average. RH-then-Span is a complete but failed structured variant.\n",
        encoding="utf-8",
    )
    return rows


def write_claims_and_analysis(out: Path, table: Sequence[Mapping[str, Any]]) -> None:
    bootstrap = read_csv(out / "bootstrap_significance.csv")
    physical = read_csv(out / "physical_kv_span_sanity_summary.csv")
    latency = read_csv(out / "latency_memory_summary.csv")
    fixed_ci = find_bootstrap(bootstrap, "SpanDRM vs fixed DRM-Add")
    larger_ci = find_bootstrap(bootstrap, "SpanDRM vs SlimInfer larger K+M")
    physical_row = method_row(physical, "span_drm_physical_kv_retention")
    text_row = method_row(physical, "span_drm_text_packing")
    latency_span = method_row(latency, "SpanDRM")
    latency_full = method_row(latency, "FullKV")
    physical_gap = (
        float(physical_row.get("task_macro_avg", 0)) - float(text_row.get("task_macro_avg", 0))
        if physical_row and text_row else None
    )
    physical_fail = int(float(physical_row.get("runner_fail", 0) or 0)) if physical_row else -1
    physical_mapping_ok = physical_fail == 0
    physical_quality_equivalent = physical_gap is not None and abs(physical_gap) <= 2.0 and physical_fail == 0
    latency_complete = (
        bool(latency_span) and bool(latency_full)
        and int(float(latency_span.get("runner_fail", 1) or 1)) == 0
        and int(float(latency_full.get("runner_fail", 1) or 1)) == 0
    )

    claims = [
        "# Updated Paper Claims",
        "",
        "## Strong Claims",
        "",
        "- On the completed 3950-sample LongBench evaluation, SpanDRM has the highest average among SlimInfer original, SlimInfer larger K+M, Random-Add, DeltaO-Add, fixed DRM-Add, SpanDRM, and RH-then-Span.",
        "- SpanDRM uses the exact fixed DRM-Add selected-token target in the structured full audit, with zero budget warnings and zero runner failures.",
        "- The deployable SpanDRM query selector does not use answers, gold evidence, metric gain, or oracle labels.",
        "",
        "## Moderate Claims",
        "",
        "- SpanDRM provides a modest full-average quality improvement: +0.36 vs SlimInfer original, +0.25 vs SlimInfer larger K+M, and +0.15 vs fixed DRM-Add.",
        f"- The task-stratified 95% CI vs larger K+M is [{larger_ci.get('ci95_low')}, {larger_ci.get('ci95_high')}], and vs fixed DRM-Add is [{fixed_ci.get('ci95_low')}, {fixed_ci.get('ci95_high')}]; both cross zero, so statistical certainty is limited.",
        "- Gains are concentrated in MuSiQue and Qasper, while several tasks regress; the method should not be described as uniformly better.",
        "",
        "## Diagnostic Claims",
        "",
        f"- The token-index audit maps all selected spans to source token offsets in order and exactly matches reported budgets.",
        (
            f"- The physical DynamicCache repack diagnostic differs from text packing by {physical_gap:+.2f} task-macro points with {physical_fail} runner failures; it supports mechanical mapping feasibility but not quality equivalence."
            if physical_mapping_ok and not physical_quality_equivalent else
            f"- The physical DynamicCache repack diagnostic supports mechanical mapping and approximate quality consistency (gap {physical_gap:+.2f}, failures {physical_fail})."
            if physical_quality_equivalent else
            f"- The physical DynamicCache repack is an approximation and does not establish production sparse-KV equivalence"
            + (f" (gap {physical_gap:+.2f}, failures {physical_fail})." if physical_gap is not None else ".")
        ),
        "- RH-DRM showed diagnostic signal, but RH-then-Span did not generalize and is a failed full structured variant.",
        (
            f"- The latency run is a {latency_span.get('evaluation')} diagnostic of actual text-packed runtime and theoretical retained-token KV memory."
            if latency_complete else
            "- Latency validation is incomplete or has failures."
        ),
        "",
        "## Do-Not-Claim",
        "",
        "- Do not claim statistically significant superiority over SlimInfer larger K+M or fixed DRM-Add; both paired 95% CIs cross zero.",
        "- Do not claim that SpanDRM implements a production physical span-granular sparse KV cache.",
        "- Do not claim full-workload latency speedup; only a small diagnostic was run.",
        "- Do not equate the retained-token KV memory formula with observed physical GPU allocation.",
        "- Do not claim improvement on every task.",
        "- Do not present RH-then-Span as the final method.",
        "- Do not present 32/64/128-sample diagnostics as full LongBench results.",
        "",
    ]
    (out / "updated_paper_claims.md").write_text("\n".join(claims), encoding="utf-8")

    limitations = [
        "# Updated Paper Limitations",
        "",
        "- Statistical uncertainty: SpanDRM's full-average improvements are small and paired bootstrap confidence intervals cross zero.",
        "- Task concentration: MuSiQue and Qasper contribute much of the gain, while 2WikiMQA, VCSum, LSHT, and other tasks regress.",
        "- Systems scope: the full quality result uses order-preserving text/token packing with ordinary FullKV. The physical K/V experiment is a post-prefill DynamicCache repack approximation.",
        "- Efficiency scope: latency and peak-memory measurements are 32-sample-per-task diagnostics with an 8-token generation cap, not full LongBench systems benchmarks.",
        "- Memory interpretation: the KV-memory figure is computed from retained token count and model geometry; it excludes model weights, allocator behavior, temporary activations, and full-prefill cost.",
        "- Selection analysis: case-study evidence overlap is post hoc and must not be confused with a selection feature.",
        "- Generalization: results cover Qwen2.5-7B-Instruct and LongBench only.",
        "",
    ]
    (out / "updated_paper_limitations.md").write_text("\n".join(limitations), encoding="utf-8")

    span = next(row for row in table if row.get("Method") == "SpanDRM")
    analysis = [
        "# Updated SpanDRM Paper Analysis",
        "",
        "## Final Result",
        "",
        f"SpanDRM remains final Ours with full LongBench Avg {span.get('Avg')}. Its quality result is complete, budget-matched, and leakage-free, but the improvement is modest rather than statistically decisive.",
        "",
        "## Validation Decisions",
        "",
        f"- Bootstrap vs SlimInfer larger K+M significant positive: **{bool_value(larger_ci.get('significant_positive'))}**.",
        f"- Bootstrap vs fixed DRM-Add significant positive: **{bool_value(fixed_ci.get('significant_positive'))}**.",
        f"- Physical KV mechanical mapping completed without failures: **{physical_mapping_ok}**.",
        f"- Physical KV quality-equivalence criterion (absolute gap <=2 points): **{physical_quality_equivalent}**.",
        f"- Fair small-sample latency diagnostic completed without FullKV/SpanDRM failures: **{latency_complete}**.",
        "",
        "## Paper Positioning",
        "",
        (
            "A defensible moderate description is: SpanDRM is evidence-density-aware structured context retention with mechanically validated order-preserving KV-index mapping. The full quality result remains a proxy because physical and text-packed quality are not equivalent."
            if physical_mapping_ok and not physical_quality_equivalent else
            "A defensible moderate description is: SpanDRM is evidence-density-aware span-level context retention with a small-scale mapping to order-preserving KV retention."
            if physical_quality_equivalent else
            "The defensible description remains: SpanDRM is evidence-density-aware structured context/KV-retention proxy. The physical prototype is not strong enough to rename it as a physical sparse-KV method."
        ),
        "",
        "The main claim should emphasize completed full quality under an exactly matched selected-token budget. It should not emphasize statistical significance, universal task gains, production sparse-KV implementation, or full-workload speedup.",
        "",
    ]
    (out / "updated_paper_final_analysis.md").write_text("\n".join(analysis), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    bootstrap = read_csv(out / "bootstrap_significance.csv")
    physical = read_csv(out / "physical_kv_span_sanity_summary.csv")
    latency = read_csv(out / "latency_memory_summary.csv")
    table = updated_main_table(out, bootstrap, physical, latency)
    write_claims_and_analysis(out, table)
    write_run_scripts(out)
    config = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "stage": "paper_span_drm_validation",
        "model_path": str(MODEL_PATH),
        "longbench_dir": str(LONGBENCH_DIR),
        "paper_full_root": str(PAPER_ROOT.relative_to(REPO_ROOT)),
        "structured_full_root": str(STRUCTURED_ROOT.relative_to(REPO_ROOT)),
        "sanity_tasks": SANITY_TASKS,
        "samples_per_task": 32,
        "latency_decode_tokens": 8,
        "bootstrap_n": 10000,
        "bootstrap_seed": 20260609,
        "physical_kv_status": "order-preserving post-prefill DynamicCache repack approximation",
        "selection_policy": "query-only deployable selection; no answer/gold/oracle inputs",
        "full_latency_run": False,
    }
    write_json(out / "config.json", config)
    required = [
        "physical_kv_span_sanity_summary.csv", "physical_kv_span_sanity_report.md",
        "latency_memory_summary.csv", "latency_memory_report.md",
        "bootstrap_significance.csv", "bootstrap_significance_report.md",
        "case_study_index.csv", "case_study_report.md",
        "updated_paper_main_table_full.csv", "updated_paper_main_table_full.md",
        "updated_paper_claims.md", "updated_paper_limitations.md",
        "updated_paper_final_analysis.md", "run_physical_kv_span_sanity.sh",
        "run_latency_memory_eval.sh", "run_bootstrap_significance.sh",
        "run_case_study.sh", "config.json", "README.md",
    ]
    readme = [
        "# SpanDRM Paper Validation",
        "",
        "This directory validates the completed SpanDRM full LongBench result with source/token audits, an explicitly approximate physical K/V repack sanity check, a small fair latency diagnostic, paired bootstrap statistics, and balanced qualitative cases.",
        "",
        "## Interpretation Guardrails",
        "",
        "- Full quality results and 32-sample diagnostics are labeled separately.",
        "- Text span packing is not described as a production physical sparse KV cache.",
        "- Gold-answer overlap appears only in post-hoc case analysis.",
        "- Latency results are diagnostic; no full-workload speedup claim is made.",
        "",
        "## Required Artifacts",
        "",
    ]
    readme.extend(f"- `{name}`" for name in required)
    readme.extend([
        "",
        "Additional audit files include `current_full_result_audit.*`, "
        "`physical_span_token_index_audit.csv`, per-task/raw latency and physical outputs, "
        "bootstrap breakdowns, and `case_study_cases.json`.",
        "",
    ])
    (out / "README.md").write_text("\n".join(readme), encoding="utf-8")


if __name__ == "__main__":
    main()
