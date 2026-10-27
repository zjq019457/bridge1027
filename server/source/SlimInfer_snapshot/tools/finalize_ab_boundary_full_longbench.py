#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import shutil
import stat
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = REPO_ROOT / "results" / "rescuekv_drm_add_full_longbench_20260523_233633"
CORE_GATE_DIR = REPO_ROOT / "results" / "rescuekv_ab_budget_calibration_fix_20260526_174503" / "core_gate_review"

TASKS = [
    "qasper",
    "multifieldqa_en",
    "hotpotqa",
    "2wikimqa",
    "musique",
    "gov_report",
    "qmsum",
    "multi_news",
    "vcsum",
    "trec",
    "triviaqa",
    "samsum",
    "lsht",
    "passage_count",
    "passage_retrieval_en",
    "lcc",
    "repobench-p",
]

TASK_COLUMNS = [
    ("qasper", "Qasper"),
    ("multifieldqa_en", "MFQA"),
    ("hotpotqa", "HPQA"),
    ("2wikimqa", "2Wiki"),
    ("musique", "MuSiQue"),
    ("gov_report", "GovRep"),
    ("qmsum", "QMSum"),
    ("multi_news", "MNews"),
    ("vcsum", "VCSum"),
    ("trec", "TREC"),
    ("triviaqa", "TQA"),
    ("samsum", "SAMSum"),
    ("lsht", "LSHT"),
    ("passage_count", "Count"),
    ("passage_retrieval_en", "PassR"),
    ("lcc", "LCC"),
    ("repobench-p", "RepoB-p"),
]

CATEGORIES = {
    "SingleDocQA": ["qasper", "multifieldqa_en", "hotpotqa"],
    "MultiDocQA": ["2wikimqa", "musique"],
    "Summarization": ["gov_report", "qmsum", "multi_news", "vcsum"],
    "FewShot": ["trec", "triviaqa", "samsum"],
    "Synthetic": ["lsht", "passage_count", "passage_retrieval_en"],
    "Code": ["lcc", "repobench-p"],
}

AB_METHODS = {
    "AB-SlimInfer": "AB-SlimInfer",
    "AB-SlimInfer_larger_KplusM": "AB-SlimInfer larger K+M",
    "AB-DRM": "AB-DRM",
    "AB-BoundaryDRM": "AB-BoundaryDRM (ours)",
}


def fnum(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except Exception:
        return default


def mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    return sum(vals) / len(vals) if vals else 0.0


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, Any]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def fmt(value: Any) -> str:
    if value in ("", None):
        return ""
    try:
        return f"{float(value):.2f}"
    except Exception:
        return str(value)


def markdown_table(rows: List[Dict[str, Any]], fields: List[str]) -> str:
    if not rows:
        return "_No rows._"
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    return "\n".join(lines)


def expected_counts() -> Dict[str, int]:
    counts = load_json(BASELINE_DIR / "task_sample_counts.json")
    return {task: int(counts.get(task, 0)) for task in TASKS}


def parse_method_metrics(out_dir: Path, run_name: str, display_name: str, counts: Dict[str, int]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    metrics_path = out_dir / "runs" / run_name / "metrics.json"
    metrics = load_json(metrics_path)
    rows = []
    row_by_task = {r.get("dataset"): r for r in metrics.get("longbench", {}).get("rows", []) if isinstance(r, dict)}
    for task in TASKS:
        row = row_by_task.get(task, {})
        sample_count = int(row.get("num_samples") or 0)
        expected = int(counts.get(task, 0))
        status = row.get("status") or ("missing" if not row else "unknown")
        if sample_count != expected or status != "ok":
            status = "partial" if sample_count else status
        rows.append({
            "method": display_name,
            "task": task,
            "score_or_metric": fnum(row.get("score"), None),
            "sample_count": sample_count,
            "expected_count": expected,
            "status": status,
            "source": "run",
        })
    return rows, metrics


def audit_rows_for_method(out_dir: Path, run_name: str, display_name: str) -> List[Dict[str, Any]]:
    path = out_dir / "audit" / f"{run_name}.jsonl"
    if not path.exists():
        return []
    by_sample: Dict[Tuple[str, str], Dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except Exception:
                continue
            if row.get("phase") != "prefill":
                continue
            task = str(row.get("task", ""))
            sample = str(row.get("sample_id") or row.get("sample_index") or len(by_sample))
            key = (task, sample)
            by_sample.setdefault(key, row)

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for (task, _sample), row in by_sample.items():
        grouped.setdefault(task, []).append(row)

    out_rows: List[Dict[str, Any]] = []
    for task in TASKS:
        rows = grouped.get(task, [])
        selected = [fnum(r.get("observed_final_selected_tokens_used_by_attention", r.get("selected_token_count"))) for r in rows]
        target = [fnum(r.get("target_tokens_used_by_cache", r.get("target_token_budget"))) for r in rows]
        pages = [fnum(r.get("selected_pages", r.get("final_block_count_used_by_attention"))) for r in rows]
        overhead = [fnum(r.get("page_expansion_overhead_mean", r.get("page_expansion_overhead", 1.0)), 1.0) for r in rows]
        loaded = [bool(r.get("calibrated_budget_config_loaded")) for r in rows]
        found = [bool(r.get("calibration_entry_found")) for r in rows]
        selected_mean = mean(selected)
        target_mean = mean(target)
        error = (selected_mean - target_mean) / target_mean if target_mean else 0.0
        status = "ok"
        if abs(error) > 0.05:
            status = "budget_warning"
        if not rows:
            status = "missing"
        out_rows.append({
            "method": display_name,
            "task": task,
            "selected_tokens_mean": selected_mean,
            "selected_pages_mean": mean(pages),
            "selected_token_error_pct": error,
            "page_expansion_overhead": mean(overhead),
            "calibrated_config_loaded": all(loaded) if loaded else False,
            "calibration_entry_found": all(found) if found else False,
            "target_tokens_used_by_cache": target_mean,
            "observed_final_selected_tokens_used_by_attention": selected_mean,
            "budget_status": status,
            "sample_count": len(rows),
        })
    return out_rows


def build_main_rows(baseline_rows: List[Dict[str, str]], ab_summary: List[Dict[str, Any]], budget_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    main_rows: List[Dict[str, Any]] = []
    for row in baseline_rows:
        out = dict(row)
        out.setdefault("SelectedTokens_if_available", "")
        out.setdefault("SelectedPages_if_available", "")
        out.setdefault("PageExpansionOverhead_if_available", "")
        main_rows.append(out)

    budget_by_method: Dict[str, List[Dict[str, Any]]] = {}
    for row in budget_rows:
        budget_by_method.setdefault(str(row["method"]), []).append(row)

    for run_name, display in AB_METHODS.items():
        task_rows = [r for r in ab_summary if r["method"] == display]
        by_task = {r["task"]: r for r in task_rows}
        full_ok = all(by_task.get(task, {}).get("status") == "ok" and int(by_task.get(task, {}).get("sample_count", 0)) == int(by_task.get(task, {}).get("expected_count", -1)) for task in TASKS)
        scores = [fnum(by_task.get(task, {}).get("score_or_metric"), math.nan) for task in TASKS]
        scores_clean = [s for s in scores if not math.isnan(s)]
        sample_count = sum(int(by_task.get(task, {}).get("sample_count", 0)) for task in TASKS)
        expected = sum(int(by_task.get(task, {}).get("expected_count", 0)) for task in TASKS)
        status = "ok" if full_ok else "partial"
        method_budget = budget_by_method.get(display, [])
        selected_tokens = mean([fnum(r.get("selected_tokens_mean")) for r in method_budget if r.get("budget_status") != "missing"])
        selected_pages = mean([fnum(r.get("selected_pages_mean")) for r in method_budget if r.get("budget_status") != "missing"])
        overhead = mean([fnum(r.get("page_expansion_overhead"), 1.0) for r in method_budget if r.get("budget_status") != "missing"])
        out: Dict[str, Any] = {"Method": display}
        for task, col in TASK_COLUMNS:
            value = by_task.get(task, {}).get("score_or_metric")
            out[col] = fmt(value) if value is not None else ""
        out["Avg"] = fmt(mean(scores_clean)) if len(scores_clean) == len(TASKS) else ""
        out["SampleMode"] = "full" if full_ok else "partial"
        out["SampleCount"] = f"{sample_count}/{expected}"
        out["Status"] = status
        out["E2E_32k_s_if_available"] = ""
        out["TTFT_32k_s_if_available"] = ""
        out["SelectedTokens_if_available"] = fmt(selected_tokens)
        out["SelectedPages_if_available"] = fmt(selected_pages)
        out["PageExpansionOverhead_if_available"] = fmt(overhead)
        main_rows.append(out)
    return main_rows


def category_rows(main_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    col_by_task = dict(TASK_COLUMNS)
    for row in main_rows:
        out = {"Method": row.get("Method", "")}
        all_scores = []
        for cat, tasks in CATEGORIES.items():
            vals = [fnum(row.get(col_by_task[t]), math.nan) for t in tasks]
            vals = [v for v in vals if not math.isnan(v)]
            out[cat] = fmt(mean(vals)) if len(vals) == len(tasks) else ""
            all_scores.extend(vals)
        out["Avg"] = fmt(fnum(row.get("Avg"))) if row.get("Avg") not in ("", None) else ""
        rows.append(out)
    return rows


def latex_table(main_rows: List[Dict[str, Any]]) -> str:
    complete = [r for r in main_rows if r.get("SampleMode") == "full" and r.get("Status") in {"ok", "reused"}]
    fields = ["Method"] + [col for _, col in TASK_COLUMNS] + ["Avg"]
    best: Dict[str, float] = {}
    for field in fields[1:]:
        vals = [fnum(r.get(field), math.nan) for r in complete]
        vals = [v for v in vals if not math.isnan(v)]
        best[field] = max(vals) if vals else math.nan
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{l" + "r" * (len(fields) - 1) + "}",
        "\\toprule",
        " & ".join(fields) + " \\\\",
        "\\midrule",
    ]
    for row in complete:
        cells = []
        for field in fields:
            value = str(row.get(field, ""))
            if field == "Method" and value == "AB-BoundaryDRM (ours)":
                value = "AB-BoundaryDRM (ours)"
            elif field != "Method":
                numeric = fnum(value, math.nan)
                value = f"{numeric:.2f}" if not math.isnan(numeric) else ""
                if not math.isnan(numeric) and not math.isnan(best.get(field, math.nan)) and abs(numeric - best[field]) < 0.005:
                    value = "\\textbf{" + value + "}"
            cells.append(value)
        lines.append(" & ".join(cells) + " \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Full LongBench results. Fixed baselines are reused from the prior full run; AB-BoundaryDRM is the proposed method.}",
        "\\end{table*}",
    ]
    return "\n".join(lines) + "\n"


def copy_predictions(out_dir: Path) -> None:
    raw_pred = out_dir / "raw_predictions"
    raw_metrics = out_dir / "raw_metrics"
    for run_name in AB_METHODS:
        run_dir = out_dir / "runs" / run_name
        pred_src = run_dir / "longbench_predictions"
        if pred_src.exists():
            dest = raw_pred / run_name
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(pred_src, dest)
        metrics_src = run_dir / "longbench_scores.csv"
        if metrics_src.exists():
            dest_dir = raw_metrics / run_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(metrics_src, dest_dir / "longbench_scores.csv")
            metrics_json = run_dir / "metrics.json"
            if metrics_json.exists():
                shutil.copy2(metrics_json, dest_dir / "metrics.json")


def write_report(out_dir: Path, main_rows: List[Dict[str, Any]], cat_rows: List[Dict[str, Any]], budget_rows: List[Dict[str, Any]], partial_rows: List[Dict[str, Any]]) -> None:
    fields = ["Method"] + [col for _, col in TASK_COLUMNS] + [
        "Avg", "SampleMode", "SampleCount", "Status",
        "E2E_32k_s_if_available", "TTFT_32k_s_if_available",
        "SelectedTokens_if_available", "SelectedPages_if_available", "PageExpansionOverhead_if_available",
    ]
    cat_fields = ["Method", "SingleDocQA", "MultiDocQA", "Summarization", "FewShot", "Synthetic", "Code", "Avg"]
    by_method = {r["Method"]: r for r in main_rows}
    def avg(name: str) -> float:
        return fnum(by_method.get(name, {}).get("Avg"), math.nan)
    abd = avg("AB-BoundaryDRM (ours)")
    comparisons = [
        ("AB-SlimInfer vs SlimInfer original", avg("AB-SlimInfer") - avg("SlimInfer")),
        ("AB-SlimInfer larger K+M vs SlimInfer larger K+M", avg("AB-SlimInfer larger K+M") - avg("SlimInfer larger K+M")),
        ("AB-DRM vs fixed DRM-Add", avg("AB-DRM") - avg("DRM-Add")),
        ("AB-BoundaryDRM vs SlimInfer original", abd - avg("SlimInfer")),
        ("AB-BoundaryDRM vs SlimInfer larger K+M", abd - avg("SlimInfer larger K+M")),
        ("AB-BoundaryDRM vs fixed DRM-Add", abd - avg("DRM-Add")),
        ("AB-BoundaryDRM vs AB-DRM", abd - avg("AB-DRM")),
        ("AB-BoundaryDRM vs Random-Add", abd - avg("Random-Add")),
        ("AB-BoundaryDRM vs DeltaO-Add", abd - avg("DeltaO-Add")),
        ("AB-BoundaryDRM vs FullKV", abd - avg("FullKV")),
    ]
    budget_warnings = [r for r in budget_rows if r.get("budget_status") == "budget_warning"]
    full_done = not partial_rows and all(r.get("SampleMode") == "full" for r in main_rows if str(r.get("Method", "")).startswith("AB-"))
    lines = [
        "# Attention-Boundary Full LongBench Report",
        "",
        "## 1. Goal",
        "This is the full LongBench run for the Attention-Boundary AB methods. It is not calibration and not smoke.",
        "",
        "## 2. Core Gate",
        "- FULL_READY_CORE = true",
        "- CORE mean abs token error = 1.32%",
        "- CORE max abs token error = 4.88%",
        "- High-budget control is diagnostic-only and does not block this full table.",
        "",
        "## 3. Method Definitions",
        "- FullKV: quality upper bound, no pruning.",
        "- SlimInfer original: fixed 64-token pages with SlimInfer K budget.",
        "- SlimInfer larger K+M: fixed pages with larger token budget and no rescue.",
        "- Random-Add: fixed active set plus random additive rescue.",
        "- DeltaO-Add: fixed active set plus DeltaO additive rescue.",
        "- fixed DRM-Add: fixed active set plus DRM additive rescue.",
        "- AB-SlimInfer: attention-boundary virtual blocks with token-budget SlimInfer selection.",
        "- AB-SlimInfer larger K+M: attention-boundary virtual blocks with K+M token budget and no rescue.",
        "- AB-DRM: attention-boundary blocks with ordinary DRM additive rescue.",
        "- AB-BoundaryDRM: attention-boundary blocks with boundary-confidence weighted DRM additive rescue.",
        "",
        "## 4. Formula",
        "Attention boundary: `d_i = 1 - cos(r_i, r_{i+1})`; `C_bd(B) = mean(boundary_left, boundary_right)`.",
        "",
        "Boundary DRM: `S = z(D_base)+alpha*z(G)-lambda*z(R)+gamma*z(C_bd)`.",
        "",
        "MMR: `argmax[S - eta_rescue max cos(B,R) - eta_active max cos(B,A)]`.",
        "",
        "## 5. Main Table",
        markdown_table(main_rows, fields),
        "",
        "## 6. Category Average",
        markdown_table(cat_rows, cat_fields),
        "",
        "## 7. Key Comparisons",
    ]
    for name, delta in comparisons:
        lines.append(f"- {name}: {delta:.2f}" if not math.isnan(delta) else f"- {name}: unavailable")
    lines += [
        "",
        f"- Full LongBench complete: {full_done}",
        f"- Budget warnings: {len(budget_warnings)}",
        "",
        "## 8. Ablation Interpretation",
        "- Block construction effect: compare AB-SlimInfer against SlimInfer original.",
        "- Budget effect: compare AB-SlimInfer larger K+M against SlimInfer larger K+M.",
        "- Ordinary rescue effect: compare AB-DRM against AB-SlimInfer larger K+M.",
        "- Boundary confidence effect: compare AB-BoundaryDRM against AB-DRM.",
        "- Old fixed rescue vs new boundary rescue: compare AB-BoundaryDRM against fixed DRM-Add.",
        "",
        "## 9. Fairness And Budget Audit",
    ]
    if budget_warnings:
        lines.append("Some method/task rows exceeded 5% selected-token error; do not make strong fairness claims for those rows.")
        for row in budget_warnings[:20]:
            lines.append(f"- {row['method']} / {row['task']}: error={float(row['selected_token_error_pct']):.4f}")
    else:
        lines.append("No AB method/task exceeded 5% selected-token error in the available audit.")
    lines += [
        "",
        "See `budget_audit.csv` for selected tokens, selected pages, page expansion overhead, and cache-path config checks.",
        "",
        "## 10. Failures And Limits",
    ]
    if partial_rows:
        lines.append("Partial or failed rows exist; they are listed in `partial_debug_table.csv` and are not treated as full results.")
        for row in partial_rows:
            lines.append(f"- {row['method']} / {row['task']}: {row['sample_count']}/{row['expected_count']} status={row['status']}")
    else:
        lines.append("- No partial AB task rows detected.")
    lines += [
        "- Latency was not run in this step; a ready-to-run latency script was generated.",
        "- High-budget matched attention control remains diagnostic-only.",
        "",
        "## 11. Next Step",
        "Run the latency script, then add case studies and gamma ablations if AB-BoundaryDRM is stable.",
        "",
        "## Artifacts",
        f"- main_table: {out_dir / 'main_table.csv'}",
        f"- main_table_latex: {out_dir / 'main_table_latex.tex'}",
        f"- category_table: {out_dir / 'category_table.csv'}",
        f"- budget_audit: {out_dir / 'budget_audit.csv'}",
        f"- latency_script: {out_dir / 'latency' / 'run_latency_32k.sh'}",
    ]
    write_text(out_dir / "report.md", "\n".join(lines) + "\n")


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: finalize_ab_boundary_full_longbench.py OUT_DIR", file=sys.stderr)
        return 2
    out_dir = Path(sys.argv[1]).resolve()
    counts = expected_counts()
    baseline_main = read_csv(BASELINE_DIR / "main_table.csv")
    baseline_cat = read_csv(BASELINE_DIR / "category_table.csv")
    baseline_summary = read_csv(BASELINE_DIR / "summary.csv")

    ab_summary: List[Dict[str, Any]] = []
    metrics_by_method: Dict[str, Dict[str, Any]] = {}
    for run_name, display in AB_METHODS.items():
        rows, metrics = parse_method_metrics(out_dir, run_name, display, counts)
        ab_summary.extend(rows)
        metrics_by_method[display] = metrics

    budget_rows: List[Dict[str, Any]] = []
    for run_name, display in AB_METHODS.items():
        budget_rows.extend(audit_rows_for_method(out_dir, run_name, display))

    summary_fields = ["method", "task", "score_or_metric", "sample_count", "expected_count", "status", "source"]
    write_csv(out_dir / "summary.csv", baseline_summary + ab_summary, summary_fields)

    budget_fields = [
        "method", "task", "selected_tokens_mean", "selected_pages_mean", "selected_token_error_pct",
        "page_expansion_overhead", "calibrated_config_loaded", "calibration_entry_found",
        "target_tokens_used_by_cache", "observed_final_selected_tokens_used_by_attention",
        "budget_status", "sample_count",
    ]
    write_csv(out_dir / "budget_audit.csv", budget_rows, budget_fields)
    write_csv(out_dir / "audit_summary.csv", budget_rows, budget_fields)

    main_fields = [
        "Method", "Qasper", "MFQA", "HPQA", "2Wiki", "MuSiQue", "GovRep", "QMSum", "MNews", "VCSum",
        "TREC", "TQA", "SAMSum", "LSHT", "Count", "PassR", "LCC", "RepoB-p", "Avg",
        "SampleMode", "SampleCount", "Status", "E2E_32k_s_if_available", "TTFT_32k_s_if_available",
        "SelectedTokens_if_available", "SelectedPages_if_available", "PageExpansionOverhead_if_available",
    ]
    main_rows = build_main_rows(baseline_main, ab_summary, budget_rows)
    write_csv(out_dir / "main_table.csv", main_rows, main_fields)

    cat_fields = ["Method", "SingleDocQA", "MultiDocQA", "Summarization", "FewShot", "Synthetic", "Code", "Avg"]
    cats = category_rows(main_rows)
    write_csv(out_dir / "category_table.csv", cats, cat_fields)
    write_text(out_dir / "main_table_latex.tex", latex_table(main_rows))

    partial_rows = [r for r in ab_summary if r.get("status") != "ok" or int(r.get("sample_count", 0)) != int(r.get("expected_count", -1))]
    if partial_rows:
        write_csv(out_dir / "partial_debug_table.csv", partial_rows, summary_fields)
    elif (out_dir / "partial_debug_table.csv").exists():
        (out_dir / "partial_debug_table.csv").unlink()

    copy_predictions(out_dir)
    write_report(out_dir, main_rows, cats, budget_rows, partial_rows)

    cfg = load_json(out_dir / "config.json")
    cfg.update({
        "full_longbench_completed": not partial_rows and all(r.get("SampleMode") == "full" for r in main_rows if str(r.get("Method", "")).startswith("AB-")),
        "full_longbench_run_started": True,
        "ab_methods": list(AB_METHODS.values()),
        "main_table": str(out_dir / "main_table.csv"),
        "category_table": str(out_dir / "category_table.csv"),
        "budget_audit": str(out_dir / "budget_audit.csv"),
        "report": str(out_dir / "report.md"),
    })
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Finalized {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
