#!/usr/bin/env python3
"""Run the gated 6-task DRM-Add follow-up without launching full LongBench."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.run_drm_add_mechanism_revalidation import (  # noqa: E402
    DEFAULT_LONGBENCH,
    DEFAULT_MODEL,
    METHODS,
)

SIX_TASKS = (
    "qasper",
    "multifieldqa_en",
    "hotpotqa",
    "2wikimqa",
    "musique",
    "passage_retrieval_en",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--model_path", type=Path, default=Path(DEFAULT_MODEL))
    parser.add_argument("--longbench_dir", type=Path, default=Path(DEFAULT_LONGBENCH))
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--tasks", type=str, default=",".join(SIX_TASKS))
    parser.add_argument("--longbench_max_samples", type=int, default=200)
    parser.add_argument(
        "--methods",
        type=str,
        default=(
            "sliminfer_original,drm_swap_original_k,"
            "sliminfer_larger_km,sliminfer_random_add_km,sliminfer_drm_add_km"
        ),
        help="Comma-separated subset of METHODS to run.",
    )
    parser.add_argument("--force", action="store_true", help="Rerun even when a completed metrics.json exists.")
    return parser.parse_args()


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def metrics_complete(metrics_path: Path, tasks: Sequence[str]) -> bool:
    if not metrics_path.exists():
        return False
    try:
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    rows = data.get("longbench", {}).get("rows", [])
    ok_tasks = {row.get("dataset") for row in rows if row.get("status") == "ok"}
    failed = data.get("longbench", {}).get("failed_samples", [])
    return set(tasks).issubset(ok_tasks) and not failed


def command_for(args: argparse.Namespace, method_name: str, run_dir: Path) -> List[str]:
    spec = METHODS[method_name]
    return [
        args.python,
        str(REPO_ROOT / "experiments" / "run_qwen25_sliminfer_variants.py"),
        "--model_path", str(args.model_path),
        "--tokenizer_path", str(args.model_path),
        "--model_family", "qwen2",
        "--dtype", "fp16",
        "--device", "cuda",
        "--backend", "flash_attention_2",
        "--longbench_dir", str(args.longbench_dir),
        "--longbench_tasks", args.tasks,
        "--longbench_max_samples", str(args.longbench_max_samples),
        "--max_model_len", "131072",
        "--method", str(spec["runner"]),
        "--eval", "longbench",
        "--block_strategy", "fixed",
        "--block_size", "64",
        "--token_unit_size", "8",
        "--local_query_window", "4",
        "--swap_threshold", "0.9",
        "--prune_layers", "9,18,26",
        "--retain_tokens", "12288,6144,4096",
        "--decode_policy", "original",
        "--rescue_policy", str(spec["policy"]),
        "--rescue_score", str(spec["score"]),
        "--rescue_ratio", "0.05",
        "--alpha_diff", "0.5",
        "--lambda_red", "0.25",
        "--eta_rescue", "0.25",
        "--eta_active", "0.25",
        "--seed", "0",
        "--min_free_gpu_gb", "0",
        "--output_dir", str(run_dir),
    ]


def run_cmd(cmd: List[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n$ " + " ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), stdout=log, stderr=subprocess.STDOUT)
        log.write(f"\n[returncode] {proc.returncode}\n")
        return int(proc.returncode)


def write_summary(out_dir: Path, method_names: Sequence[str], tasks: Sequence[str]) -> None:
    rows: List[Dict[str, Any]] = []
    for method_name in method_names:
        metrics_path = out_dir / f"run_{method_name}" / "metrics.json"
        row: Dict[str, Any] = {
            "method": method_name,
            "budget": METHODS[method_name]["budget"],
            "base_method": METHODS[method_name].get("base") or "",
            "LongBench_avg": "",
            "failed_samples": "",
            "status": "missing",
        }
        for task in tasks:
            row[task] = ""
        if metrics_path.exists():
            data = json.loads(metrics_path.read_text(encoding="utf-8"))
            lb = data.get("longbench", {})
            row["LongBench_avg"] = data.get("LongBench_avg", lb.get("avg", ""))
            row["failed_samples"] = len(lb.get("failed_samples", []))
            row["status"] = data.get("status", "ok")
            for item in lb.get("rows", []):
                task = item.get("dataset")
                if task in tasks:
                    row[task] = item.get("score", "")
        rows.append(row)
    fieldnames = ["method", "budget", "base_method", "LongBench_avg", *tasks, "failed_samples", "status"]
    with (out_dir / "six_task_quality_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tasks = [task for task in args.tasks.split(",") if task]
    method_names = [method for method in args.methods.split(",") if method]
    unknown = [method for method in method_names if method not in METHODS]
    if unknown:
        raise ValueError(f"Unknown method(s): {unknown}")
    write_json(args.out_dir / "six_task_run_config.json", {
        "tasks": tasks,
        "longbench_max_samples": args.longbench_max_samples,
        "methods": method_names,
        "frozen_drm": {
            "rescue_ratio": 0.05,
            "alpha_diff": 0.5,
            "lambda_red": 0.25,
            "eta_rescue": 0.25,
            "eta_active": 0.25,
            "seed": 0,
        },
    })
    for method_name in method_names:
        run_dir = args.out_dir / f"run_{method_name}"
        metrics_path = run_dir / "metrics.json"
        if not args.force and metrics_complete(metrics_path, tasks):
            continue
        cmd = command_for(args, method_name, run_dir)
        write_json(args.out_dir / f"command_run_{method_name}.json", {"command": cmd})
        rc = run_cmd(cmd, args.out_dir / f"run_{method_name}.log")
        if rc != 0:
            write_json(args.out_dir / "six_task_status.json", {
                "status": "failed",
                "failed_method": method_name,
                "returncode": rc,
            })
            write_summary(args.out_dir, method_names, tasks)
            return rc
        write_summary(args.out_dir, method_names, tasks)
    write_summary(args.out_dir, method_names, tasks)
    write_json(args.out_dir / "six_task_status.json", {"status": "ok", "methods": method_names})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
