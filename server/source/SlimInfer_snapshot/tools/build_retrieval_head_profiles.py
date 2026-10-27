#!/usr/bin/env python3
"""Build static retrieval/reasoning/query-focused head profiles for RH-DRM."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.fup_guarded_swap_lib import FupHarness, release_cuda

DEFAULT_MODEL = "/home/yu-bai/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
DEFAULT_LB = "/home/yu-bai/Desktop/jiaqizhang/HeadKV/data/LongBench"
TASKS = ["musique", "hotpotqa", "qasper", "passage_count", "passage_retrieval_en"]


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: List[Dict[str, Any]], header: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in header})


def mean(values: Iterable[float]) -> float:
    vals = [float(x) for x in values]
    return sum(vals) / len(vals) if vals else 0.0


def lift(row: Dict[str, Any], mass_key: str, frac_key: str) -> float:
    frac = float(row.get(frac_key, 0.0) or 0.0)
    if frac <= 0:
        return 0.0
    return float(row.get(mass_key, 0.0) or 0.0) / frac


def top_heads(rows: List[Dict[str, Any]], key: str, fraction: float = 0.20, minimum: int = 2) -> set:
    valid = [row for row in rows if float(row.get(key, 0.0) or 0.0) > 0.0]
    if not valid:
        return set()
    n = min(len(valid), max(minimum, int(round(len(rows) * fraction))))
    valid.sort(key=lambda row: float(row.get(key, 0.0) or 0.0), reverse=True)
    return {(int(row["layer_id"]), int(row["head_id"])) for row in valid[:n]}


def aggregate_profiles(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_head: Dict[Tuple[int, int], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_head[(int(row["layer_id"]), int(row["head_id"]))].append(row)

    profiles: List[Dict[str, Any]] = []
    for (layer_id, head_id), group in sorted(by_head.items()):
        task_distribution: Dict[str, Dict[str, Any]] = {}
        for task in sorted({str(row.get("task", "")) for row in group}):
            task_rows = [row for row in group if str(row.get("task", "")) == task]
            task_distribution[task] = {
                "support": len(task_rows),
                "evidence_attention_mass": round(mean(float(r.get("evidence_attention_mass", 0.0)) for r in task_rows), 7),
                "query_attention_mass": round(mean(float(r.get("query_attention_mass", 0.0)) for r in task_rows), 7),
                "reasoning_attention_mass": round(mean(float(r.get("reasoning_attention_mass", 0.0)) for r in task_rows), 7),
            }
        profiles.append({
            "layer_id": layer_id,
            "head_id": head_id,
            "head_type": "unknown",
            "evidence_attention_mass": round(mean(float(r.get("evidence_attention_mass", 0.0)) for r in group), 7),
            "query_attention_mass": round(mean(float(r.get("query_attention_mass", 0.0)) for r in group), 7),
            "reasoning_attention_mass": round(mean(float(r.get("reasoning_attention_mass", 0.0)) for r in group), 7),
            "local_attention_mass": round(mean(float(r.get("local_attention_mass", 0.0)) for r in group), 7),
            "active_coverage_score": round(mean(float(r.get("active_coverage_score", 0.0)) for r in group), 7),
            "calibration_support": int(sum(int(r.get("calibration_support", 0) or 0) for r in group)),
            "retrieval_score": round(mean(lift(r, "evidence_attention_mass", "evidence_page_fraction") for r in group), 7),
            "query_focused_score": round(mean(lift(r, "query_attention_mass", "query_page_fraction") for r in group), 7),
            "reasoning_score": round(mean(lift(r, "reasoning_attention_mass", "reasoning_page_fraction") for r in group), 7),
            "local_score": round(mean(lift(r, "local_attention_mass", "local_page_fraction") for r in group), 7),
            "task_distribution": task_distribution,
        })

    for layer_id in sorted({int(row["layer_id"]) for row in profiles}):
        layer_rows = [row for row in profiles if int(row["layer_id"]) == layer_id]
        retrieval = top_heads(layer_rows, "retrieval_score")
        reasoning = top_heads(layer_rows, "reasoning_score", fraction=0.15, minimum=1)
        query = top_heads(layer_rows, "query_focused_score")
        local = top_heads(layer_rows, "local_score")
        for row in layer_rows:
            key = (layer_id, int(row["head_id"]))
            if key in retrieval:
                row["head_type"] = "retrieval"
            elif key in reasoning:
                row["head_type"] = "reasoning"
            elif key in query:
                row["head_type"] = "query_focused"
            elif key in local:
                row["head_type"] = "local"
    return profiles


def write_report(out: Path, profiles: List[Dict[str, Any]], args: argparse.Namespace, failures: List[str]) -> None:
    counts: Dict[str, int] = defaultdict(int)
    for row in profiles:
        counts[str(row["head_type"])] += 1
    lines = [
        "# Retrieval Head Profile Report",
        "",
        f"- Calibration tasks: {', '.join(args.tasks)}",
        f"- Calibration samples per task: {args.calibration_samples}",
        f"- Calibration sample range: [{args.calibration_start}, {args.calibration_start + args.calibration_samples})",
        "- Evaluation range is the first 64 samples, so the static profile is held out from the diagnostic.",
        "- Environment: requested conda `silm` is unavailable on this machine; runs use the repo `.venv`.",
        "",
        "## Head Types",
        "",
    ]
    for key in ["retrieval", "reasoning", "query_focused", "local", "unknown"]:
        lines.append(f"- {key}: {counts.get(key, 0)}")
    lines += [
        "",
        "## Construction",
        "",
        "- Retrieval heads are ranked by attention lift on answer-overlap evidence pages.",
        "- Query-focused heads are ranked by attention lift on query-related pages.",
        "- Reasoning heads are ranked by attention lift on multi-hop query pages that do not contain the final answer.",
        "- Local heads are ranked by attention lift on the most recent 10% of available pages.",
        "- These are QK attention-mass proxies from SlimInfer's pruning query, not full dense attention dumps.",
        "",
        "## Failures",
        "",
        f"- Failed calibration samples: {len(failures)}",
    ]
    if failures:
        lines.extend(f"- {item}" for item in failures[:20])
    (out / "head_profile_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Path:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dump_path = out / "_head_profile_dump.jsonl"
    if dump_path.exists() and not args.reuse_dump:
        dump_path.unlink()
    failures: List[str] = []
    if not args.reuse_dump or not dump_path.exists():
        harness = FupHarness(args.model_path, args.longbench_dir, str(out / "_hwork"), max_new_tokens=1)
        for task in args.tasks:
            samples = harness.load_samples(task, args.calibration_start + args.calibration_samples)
            selected = samples[args.calibration_start:args.calibration_start + args.calibration_samples]
            for offset, sample in enumerate(selected):
                sample_index = args.calibration_start + offset
                try:
                    ctx = harness.build_sample_ctx(task, sample, sample_index)
                    harness.eval_pageset(ctx, {
                        "_method": "head_profile_calibration",
                        "block_strategy": "fixed_active_rescue",
                        "rescue_candidate_strategy": "fixed",
                        "rescue_selector": "drm_mmr",
                        "head_profile_dump_path": str(dump_path),
                    }, mode="gen", gen_tokens=1)
                except Exception as exc:
                    failures.append(f"{task}/{sample_index}: {type(exc).__name__}: {exc}")
                release_cuda()
            print(f"[head-profile] {task}: {len(selected)} calibration samples", flush=True)

    raw_rows = read_jsonl(dump_path)
    profiles = aggregate_profiles(raw_rows)
    payload = {
        "created_at": dt.datetime.now().isoformat(),
        "profile_method": "held_out_qk_attention_lift",
        "calibration_start": args.calibration_start,
        "calibration_samples_per_task": args.calibration_samples,
        "tasks": args.tasks,
        "heads": profiles,
    }
    (out / "head_profile.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    csv_rows = [{**row, "task_distribution": json.dumps(row["task_distribution"], sort_keys=True)} for row in profiles]
    header = [
        "layer_id", "head_id", "head_type", "evidence_attention_mass", "query_attention_mass",
        "reasoning_attention_mass", "local_attention_mass", "active_coverage_score",
        "calibration_support", "retrieval_score", "query_focused_score", "reasoning_score",
        "local_score", "task_distribution",
    ]
    write_csv(out / "head_profile_summary.csv", csv_rows, header)
    write_report(out, profiles, args, failures)
    (out / "config.json").write_text(json.dumps({
        "tasks": args.tasks,
        "calibration_start": args.calibration_start,
        "calibration_samples": args.calibration_samples,
        "model_path": args.model_path,
        "longbench_dir": args.longbench_dir,
        "python_env": str(REPO_ROOT / ".venv"),
        "env_note": "conda env silm unavailable; repo .venv used",
    }, indent=2), encoding="utf-8")
    print(f"[head-profile] wrote {len(profiles)} heads -> {out}", flush=True)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument("--calibration_start", type=int, default=64)
    parser.add_argument("--calibration_samples", type=int, default=16)
    parser.add_argument("--model_path", default=DEFAULT_MODEL)
    parser.add_argument("--longbench_dir", default=DEFAULT_LB)
    parser.add_argument("--output_dir", default=str(REPO_ROOT / "profiles" / "retrieval_head_profiles"))
    parser.add_argument("--reuse_dump", action="store_true")
    args = parser.parse_args()
    args.tasks = [task for task in args.tasks.split(",") if task]
    return args


if __name__ == "__main__":
    run(parse_args())
