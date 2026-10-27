#!/usr/bin/env python3
"""Resume-safe task orchestrator for Structured DRM full LongBench."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results"
PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
TASK_RUNNER = REPO_ROOT / "tools" / "run_structured_drm_stage2_128.py"
BASELINE_ROOT = RESULTS / "rescuekv_drm_add_full_longbench_20260523_233633"
DEFAULT_MODEL = (
    "/home/yu-bai/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/"
    "snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
)
DEFAULT_LONGBENCH = "/home/yu-bai/Desktop/jiaqizhang/HeadKV/data/LongBench"
DEFAULT_PROFILE = REPO_ROOT / "profiles" / "retrieval_head_profiles"
TASKS = [
    "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique",
    "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa",
    "samsum", "lsht", "passage_count", "passage_retrieval_en", "lcc", "repobench-p",
]
DEFAULT_METHODS = ["span_drm_add_query", "rh_then_span_drm_add_query"]
SOURCE_METHODS = ["fixed_drm_add", "rh_drm_add_query_fixed_pool"]


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def source_status(path: Path, expected: int) -> Dict[str, Any]:
    latest = {
        str(row.get("sample_id", "")): row
        for row in read_jsonl(path)
        if str(row.get("sample_id", ""))
    }
    success = sum(1 for row in latest.values() if str(row.get("status", "")) == "success")
    failed = sum(1 for row in latest.values() if str(row.get("status", "")) == "failed")
    return {
        "records": len(latest),
        "success": success,
        "failed": failed,
        "complete": len(latest) == expected and success == expected,
    }


def prediction_status(path: Path, expected: int) -> Dict[str, Any]:
    rows = read_jsonl(path)
    latest = {str(row.get("sample_id", "")): row for row in rows if str(row.get("sample_id", ""))}
    success = sum(
        1 for row in latest.values()
        if str(row.get("status", "success")) == "success" and str(row.get("pred", "")).strip()
    )
    failed = [
        sid for sid, row in latest.items()
        if str(row.get("status", "success")) != "success" or not str(row.get("pred", "")).strip()
    ]
    return {
        "records": len(latest),
        "success": success,
        "failed": failed,
        "complete": success == expected and len(latest) == expected,
    }


def task_status(out_root: Path, task: str, methods: Sequence[str], expected: int) -> Dict[str, Any]:
    task_dir = out_root / "task_runs" / task
    method_rows: Dict[str, Any] = {}
    for method in methods:
        method_rows[method] = prediction_status(task_dir / "preds" / f"{method}__{task}.jsonl", expected)
    source_rows = {
        method: source_status(
            task_dir / "preds" / f"{method}__{task}__source_status.jsonl",
            expected,
        )
        for method in SOURCE_METHODS
    }
    complete = (
        all(row["complete"] for row in method_rows.values())
        and all(row["complete"] for row in source_rows.values())
    )
    return {
        "task": task,
        "expected": expected,
        "methods": method_rows,
        "source_status": source_rows,
        "complete": complete,
    }


def ensure_prediction_links(out_root: Path, task: str, methods: Sequence[str]) -> None:
    for method in ["fixed_drm_add", *methods]:
        source = out_root / "task_runs" / task / "preds" / f"{method}__{task}.jsonl"
        if not source.exists():
            continue
        target = out_root / "predictions" / method / f"{task}.jsonl"
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            target.unlink()
        target.symlink_to(os.path.relpath(source, target.parent))


def aggregate(out_root: Path, tasks: Sequence[str], methods: Sequence[str], counts: Dict[str, int]) -> None:
    score_rows: List[Dict[str, Any]] = []
    budget_rows: List[Dict[str, Any]] = []
    failed_rows: List[Dict[str, Any]] = []
    status_rows: List[Dict[str, Any]] = []
    for task in tasks:
        expected = int(counts[task])
        status = task_status(out_root, task, methods, expected)
        status_rows.append({
            "task": task,
            "expected": expected,
            "complete": status["complete"],
            "source_fixed_complete": status["source_status"]["fixed_drm_add"]["success"],
            "source_rh_complete": status["source_status"]["rh_drm_add_query_fixed_pool"]["success"],
            "source_failures": sum(row["failed"] for row in status["source_status"].values()),
            **{
                f"{method}_success": status["methods"][method]["success"]
                for method in methods
            },
        })
        task_dir = out_root / "task_runs" / task
        summary = read_csv(task_dir / "structured_full_summary.csv")
        for row in summary:
            method = row.get("method", "")
            if method not in {"fixed_drm_add", *methods}:
                continue
            score_rows.append({
                "method": method,
                "task": task,
                "score": row.get(task, ""),
                "avg_single_task": row.get("avg", ""),
                "sample_count": (
                    expected if method == "fixed_drm_add"
                    else status["methods"].get(method, {}).get("success", 0)
                ),
                "expected_count": expected,
                "status": (
                    "reused"
                    if method == "fixed_drm_add" and status["source_status"]["fixed_drm_add"]["complete"]
                    else "ok"
                    if status["methods"].get(method, {}).get("complete")
                    else "partial"
                ),
                "runner_fail": row.get("runner_fail", ""),
            })
        task_budget_rows = read_csv(task_dir / "structured_full_budget_audit.csv")
        context_rows = read_jsonl(task_dir / "_span_selected_contexts_structured.jsonl")
        target_values: Dict[str, List[float]] = {}
        for context_row in context_rows:
            method = str(context_row.get("method", ""))
            try:
                target = float(context_row.get("target_final_tokens", 0) or 0)
            except (TypeError, ValueError):
                target = 0.0
            if method and target > 0:
                target_values.setdefault(method, []).append(target)
        for budget_row in task_budget_rows:
            if str(budget_row.get("avg_target_tokens", "")).strip():
                continue
            method = str(budget_row.get("method", ""))
            values = target_values.get(method, [])
            if values:
                budget_row["avg_target_tokens"] = round(sum(values) / len(values), 3)
            elif method == "fixed_drm_add":
                budget_row["avg_target_tokens"] = budget_row.get("avg_selected_tokens", "")
        budget_rows.extend(task_budget_rows)
        for method in methods:
            for sid in status["methods"][method]["failed"]:
                failed_rows.append({"method": method, "task": task, "sample_id": sid})
        ensure_prediction_links(out_root, task, methods)

    write_csv(
        out_root / "structured_full_scores.csv",
        score_rows,
        ["method", "task", "score", "avg_single_task", "sample_count", "expected_count", "status", "runner_fail"],
    )
    if budget_rows:
        fields = sorted({key for row in budget_rows for key in row})
        preferred = [
            "method", "task", "n_samples", "avg_target_tokens", "avg_selected_tokens",
            "budget_error_pct", "budget_warning",
        ]
        write_csv(
            out_root / "structured_full_budget_audit.csv",
            budget_rows,
            preferred + [field for field in fields if field not in preferred],
        )
    write_csv(
        out_root / "structured_full_task_status.csv",
        status_rows,
        list(status_rows[0]) if status_rows else ["task"],
    )
    if failed_rows:
        with (out_root / "failed_samples.jsonl").open("w", encoding="utf-8") as handle:
            for row in failed_rows:
                handle.write(json.dumps(row) + "\n")
    elif (out_root / "failed_samples.jsonl").exists():
        (out_root / "failed_samples.jsonl").unlink()


def write_progress(out_root: Path, tasks: Sequence[str], methods: Sequence[str], counts: Dict[str, int]) -> Dict[str, Any]:
    statuses = [task_status(out_root, task, methods, int(counts[task])) for task in tasks]
    complete_tasks = [row["task"] for row in statuses if row["complete"]]
    total_expected = sum(int(counts[task]) for task in tasks) * len(methods)
    total_success = sum(
        int(row["methods"][method]["success"])
        for row in statuses for method in methods
    )
    progress = {
        "updated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "full_complete": len(complete_tasks) == len(tasks),
        "completed_tasks": complete_tasks,
        "tasks_complete": len(complete_tasks),
        "tasks_total": len(tasks),
        "structured_generations_success": total_success,
        "structured_generations_expected": total_expected,
        "methods": list(methods),
        "task_status": statuses,
    }
    write_json(out_root / "progress.json", progress)
    lines = [
        "# Structured DRM Full Progress",
        "",
        f"- Updated: {progress['updated_at']}",
        f"- Full complete: **{progress['full_complete']}**",
        f"- Tasks complete: {len(complete_tasks)}/{len(tasks)}",
        f"- Structured generations: {total_success}/{total_expected}",
        "",
        "| Task | Expected | Complete | " + " | ".join(methods) + " | Fixed audit | RH audit |",
        "| --- | ---: | --- | " + " | ".join("---:" for _ in methods) + " | ---: | ---: |",
    ]
    for row in statuses:
        lines.append(
            f"| {row['task']} | {row['expected']} | {row['complete']} | "
            + " | ".join(str(row["methods"][method]["success"]) for method in methods)
            + f" | {row['source_status']['fixed_drm_add']['success']} | "
            f"{row['source_status']['rh_drm_add_query_fixed_pool']['success']} |"
        )
    (out_root / "progress.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return progress


def write_scripts(out_root: Path, args: argparse.Namespace) -> None:
    methods = ",".join(args.methods)
    tasks = ",".join(args.tasks)
    base = (
        f'"{PYTHON}" tools/run_structured_drm_full_longbench.py '
        f'--output_dir "{out_root}" --methods "{methods}" --tasks "{tasks}" --full '
        f'--model_path "{args.model_path}" --longbench_dir "{args.longbench_dir}" '
        f'--profile_dir "{args.profile_dir}" --reuse_full_baseline_dir "{args.reuse_full_baseline_dir}" '
        "--budget_audit --same_eval_as_baseline"
    )
    run = (
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        f'cd "{REPO_ROOT}"\n'
        f"{base} --skip_existing\n"
    )
    resume = (
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        f'cd "{REPO_ROOT}"\n'
        f"{base} --skip_existing --overwrite_failed --resume\n"
    )
    for name, body in (("run_missing_structured_full.sh", run), ("resume_missing_structured_full.sh", resume), ("resume.sh", resume)):
        path = out_root / name
        path.write_text(body, encoding="utf-8")
        path.chmod(0o755)


def run_task(out_root: Path, task: str, expected: int, args: argparse.Namespace) -> int:
    task_dir = out_root / "task_runs" / task
    task_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_root / "logs" / f"{task}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(PYTHON), str(TASK_RUNNER),
        "--output_dir", str(task_dir),
        "--tasks", task,
        "--samples_per_task", str(expected),
        "--evaluation_mode", "full",
        "--source_audit_only",
        "--reuse_full_baseline_dir", args.reuse_full_baseline_dir,
        "--model_path", args.model_path,
        "--longbench_dir", args.longbench_dir,
        "--profile_dir", args.profile_dir,
        "--methods", ",".join(args.methods),
    ]
    if args.skip_existing:
        command.append("--skip_existing")
    if args.overwrite_failed:
        command.append("--overwrite_failed")
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_ROOT}:{env.get('PYTHONPATH', '')}"
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{dt.datetime.now().isoformat()}] {' '.join(command)}\n")
        log.flush()
        process = subprocess.run(
            command,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    return process.returncode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--tasks", default="all_longbench")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--overwrite_failed", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--budget_audit", action="store_true")
    parser.add_argument("--same_eval_as_baseline", action="store_true")
    parser.add_argument("--model_path", default=DEFAULT_MODEL)
    parser.add_argument("--longbench_dir", default=DEFAULT_LONGBENCH)
    parser.add_argument("--profile_dir", default=str(DEFAULT_PROFILE))
    parser.add_argument("--reuse_full_baseline_dir", default=str(BASELINE_ROOT))
    args = parser.parse_args()
    args.methods = [item for item in args.methods.split(",") if item]
    args.tasks = TASKS if args.tasks == "all_longbench" else [item for item in args.tasks.split(",") if item]
    if not args.output_dir:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = str(RESULTS / f"structured_drm_full_longbench_{stamp}")
    return args


def main() -> int:
    args = parse_args()
    if not args.full:
        raise SystemExit("--full is required; diagnostic runs use run_structured_drm_stage2_128.py")
    unsupported = sorted(set(args.methods) - set(DEFAULT_METHODS))
    if unsupported:
        raise SystemExit(f"unsupported paper-full methods: {unsupported}")
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    counts = read_json(Path(args.reuse_full_baseline_dir) / "task_sample_counts.json", {})
    missing_counts = [task for task in args.tasks if int(counts.get(task, 0) or 0) <= 0]
    if missing_counts:
        raise SystemExit(f"missing full sample counts: {missing_counts}")
    config = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "evaluation": "Full LongBench",
        "methods": args.methods,
        "tasks": args.tasks,
        "task_sample_counts": {task: int(counts[task]) for task in args.tasks},
        "model_path": args.model_path,
        "longbench_dir": args.longbench_dir,
        "split": "local LongBench full/test",
        "profile_dir": args.profile_dir,
        "reuse_full_baseline_dir": args.reuse_full_baseline_dir,
        "baseline_quality_reused": True,
        "source_audit_only": True,
        "same_eval_as_baseline": bool(args.same_eval_as_baseline),
        "budget_audit": bool(args.budget_audit),
        "environment": ".venv (conda silm unavailable)",
        "latency_run": False,
        "span_implementation": "order-preserving structured context/KV-retention proxy via compact token context",
    }
    write_json(out_root / "config.json", config)
    write_scripts(out_root, args)
    write_progress(out_root, args.tasks, args.methods, counts)

    for task in args.tasks:
        expected = int(counts[task])
        current = task_status(out_root, task, args.methods, expected)
        if args.skip_existing and current["complete"]:
            print(f"[structured-full] skip complete {task}", flush=True)
            continue
        print(f"[structured-full] run {task} expected={expected}", flush=True)
        returncode = run_task(out_root, task, expected, args)
        aggregate(out_root, args.tasks, args.methods, counts)
        progress = write_progress(out_root, args.tasks, args.methods, counts)
        print(
            f"[structured-full] {task} rc={returncode}; "
            f"tasks={progress['tasks_complete']}/{progress['tasks_total']}",
            flush=True,
        )
        if returncode != 0:
            print(f"[structured-full] stop after failed task {task}; use resume script", flush=True)
            return returncode

    aggregate(out_root, args.tasks, args.methods, counts)
    progress = write_progress(out_root, args.tasks, args.methods, counts)
    print(out_root)
    return 0 if progress["full_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
