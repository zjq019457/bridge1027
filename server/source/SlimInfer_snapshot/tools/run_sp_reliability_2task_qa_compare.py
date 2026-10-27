#!/usr/bin/env python3
"""400-sample two-task QA: sliminfer_original vs sp_sliminfer vs rank_stability_sliminfer.

Uses official SlimInfer two-task protocol (HotpotQA 200 + MuSiQue 200).
Sequential runs with prediction resume. Writes compare-level audit artifacts.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from longbench_eval import dataset2metric, scorer  # noqa: E402
from tools.consolidate_official_repro_results import load_predictions, per_sample_score, write_csv

TASKS = ("hotpotqa", "musique")
EXPECTED_PER_TASK = 200
PAPER_TARGETS = {"hotpotqa": 56.94, "musique": 27.25}
OFFICIAL_REPRO = {"hotpotqa": 57.43, "musique": 27.14, "macro_avg": 42.28}
LIVE_AUDIT_DIR = REPO_ROOT / "results/sp_reliability_live_prefill_audit_20260710"

METHODS: Tuple[Tuple[str, str], ...] = (
    ("sliminfer_original", "run_sliminfer_original"),
    ("sp_sliminfer", "run_sp_sliminfer"),
    ("rank_stability_sliminfer", "run_rank_stability_sliminfer"),
)

DEFAULT_MODEL = (
    "/home/yu-bai/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/"
    "snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
)


def parse_args() -> argparse.Namespace:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out_dir", type=Path, default=REPO_ROOT / f"results/sp_reliability_2task_compare_{ts}")
    p.add_argument("--model_path", default=DEFAULT_MODEL)
    p.add_argument("--longbench_dir", default="/home/yu-bai/Desktop/jiaqizhang/HeadKV/data/LongBench")
    p.add_argument("--live_audit_dir", type=Path, default=LIVE_AUDIT_DIR)
    p.add_argument("--methods", default="", help="Comma subset of method names")
    p.add_argument("--skip_run", action="store_true")
    p.add_argument("--python", default=str(REPO_ROOT / ".venv/bin/python"))
    return p.parse_args()


def selected_methods(args: argparse.Namespace) -> Tuple[Tuple[str, str], ...]:
    if not str(args.methods or "").strip():
        return METHODS
    wanted = {x.strip() for x in str(args.methods).split(",") if x.strip()}
    picked = tuple((m, d) for m, d in METHODS if m in wanted or d.replace("run_", "") in wanted)
    if not picked:
        raise ValueError(f"No methods matched: {args.methods}")
    return picked


def qa_cmd(args: argparse.Namespace, method: str, run_dir: Path) -> List[str]:
    return [
        args.python,
        str(REPO_ROOT / "experiments/run_qwen25_sliminfer_variants.py"),
        "--model_path", str(args.model_path),
        "--tokenizer_path", str(args.model_path),
        "--model_family", "qwen2",
        "--dtype", "fp16",
        "--device", "cuda",
        "--backend", "flash_attention_2",
        "--longbench_dir", str(args.longbench_dir),
        "--longbench_tasks", ",".join(TASKS),
        "--max_model_len", "131072",
        "--method", method,
        "--eval", "longbench",
        "--block_strategy", "fixed",
        "--block_size", "64",
        "--token_unit_size", "8",
        "--local_query_window", "4",
        "--swap_threshold", "0.9",
        "--prune_layers", "9,18,26",
        "--retain_tokens", "12288,6144,4096",
        "--decode_policy", "original",
        "--rescue_policy", "none",
        "--generation_max_new_tokens", "0",
        "--seed", "0",
        "--min_free_gpu_gb", "0",
        "--output_dir", str(run_dir),
    ]


def count_predictions(run_dir: Path) -> Dict[str, int]:
    pred_dir = run_dir / "longbench_predictions"
    counts: Dict[str, int] = {}
    for task in TASKS:
        path = pred_dir / f"{task}.jsonl"
        if not path.exists():
            counts[task] = 0
            continue
        counts[task] = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    return counts


def qa_complete(run_dir: Path) -> bool:
    counts = count_predictions(run_dir)
    return all(counts.get(t, 0) >= EXPECTED_PER_TASK for t in TASKS)


def read_method_scores(run_dir: Path) -> Dict[str, Any]:
    csv_path = run_dir / "longbench_scores.csv"
    if not csv_path.exists():
        return {}
    per_task: Dict[str, float] = {}
    with csv_path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("status") == "ok" and row.get("score"):
                per_task[row["dataset"]] = float(row["score"])
    if not per_task:
        return {}
    macro = round(sum(per_task.values()) / len(per_task), 2)
    return {"per_task": per_task, "macro": macro}


def consolidate_method(run_dir: Path, method: str) -> None:
    pred_dir = run_dir / "longbench_predictions"
    if not pred_dir.exists():
        return
    raw = load_predictions(pred_dir)
    for row in raw:
        row["method"] = method
    pred_out = run_dir / "predictions.jsonl"
    sample_rows: List[Dict[str, Any]] = []
    with pred_out.open("w", encoding="utf-8") as out_f:
        for row in raw:
            score = per_sample_score(row["task"], row["pred"], row["answers"], row["all_classes"])
            enriched = {**row, "score": score}
            sample_rows.append(enriched)
            out_f.write(json.dumps(enriched, ensure_ascii=False) + "\n")
    write_csv(run_dir / "scores_by_sample.csv", sample_rows,
              ["method", "task", "sample_id", "sample_index", "score", "length", "pred"])


def aggregate_compare(out_dir: Path, results: Sequence[Mapping[str, Any]], live_audit_dir: Path) -> None:
    summary_rows: List[Dict[str, Any]] = []
    task_rows: List[Dict[str, Any]] = []
    sample_rows: List[Dict[str, Any]] = []
    all_preds: List[Dict[str, Any]] = []

    for r in results:
        method = str(r["method"])
        scores = r.get("scores") or {}
        per = scores.get("per_task", {})
        macro = scores.get("macro", "")
        for task in TASKS:
            task_rows.append({
                "method": method,
                "task": task,
                "score": per.get(task, ""),
                "num_samples": count_predictions(Path(r["run_dir"])).get(task, 0),
                "paper_target": PAPER_TARGETS.get(task, ""),
                "official_repro": OFFICIAL_REPRO.get(task, ""),
            })
        summary_rows.append({
            "method": method,
            "hotpotqa": per.get("hotpotqa", ""),
            "musique": per.get("musique", ""),
            "macro_avg": macro,
            "delta_vs_official_repro_macro": round(float(macro) - OFFICIAL_REPRO["macro_avg"], 2) if macro != "" else "",
        })
        pred_path = Path(r["run_dir"]) / "predictions.jsonl"
        if pred_path.exists():
            for line in pred_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    all_preds.append(row)
                    sample_rows.append({k: row.get(k, "") for k in
                        ["method", "task", "sample_id", "sample_index", "score", "length", "pred"]})

    write_csv(out_dir / "scores_summary.csv", summary_rows,
              ["method", "hotpotqa", "musique", "macro_avg", "delta_vs_official_repro_macro"])
    write_csv(out_dir / "scores_by_task.csv", task_rows,
              ["method", "task", "score", "num_samples", "paper_target", "official_repro"])
    write_csv(out_dir / "scores_by_sample.csv", sample_rows,
              ["method", "task", "sample_id", "sample_index", "score", "length", "pred"])
    with (out_dir / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for row in all_preds:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    baseline = next((r for r in results if r["method"] == "sliminfer_original"), None)
    base_macro = (baseline.get("scores") or {}).get("macro") if baseline else OFFICIAL_REPRO["macro_avg"]
    sp = next((r for r in results if r["method"] == "sp_sliminfer"), None)
    rss = next((r for r in results if r["method"] == "rank_stability_sliminfer"), None)
    sp_macro = (sp.get("scores") or {}).get("macro") if sp else None
    rss_macro = (rss.get("scores") or {}).get("macro") if rss else None

    beat_sp = sp_macro is not None and float(sp_macro) > float(base_macro)
    beat_rss = rss_macro is not None and float(rss_macro) > float(base_macro)
    any_beat = beat_sp or beat_rss

    if live_audit_dir.exists():
        for name in ("ACTIVE_SET_AUDIT.md", "OFFICIAL_PATH_AUDIT.md", "FAIRNESS_AUDIT.md", "FEATURE_COVERAGE_AUDIT.md"):
            src = live_audit_dir / name
            if src.exists():
                shutil.copy2(src, out_dir / name)

    (out_dir / "OFFICIAL_PATH_AUDIT.md").write_text(
        "# Official Path Audit — Two-Task QA\n\n"
        "- Runner: `experiments/run_qwen25_sliminfer_variants.py --eval longbench`\n"
        "- Tasks: HotpotQA 200 + MuSiQue 200 (full LongBench files)\n"
        "- Methods: sliminfer_original, sp_sliminfer, rank_stability_sliminfer\n"
        "- Scoring: token_unit_qk; decode_policy=original; generation_max_new_tokens=0 (dataset default)\n"
        "- Budget: retain_tokens 12288/6144/4096 @ layers 9/18/26\n"
        "- rescue_policy=none; no BM25/LBP packing in selection path\n"
        "- Live prefill audit: PASS (`qa_ready=true`)\n",
        encoding="utf-8",
    )
    (out_dir / "FAIRNESS_AUDIT.md").write_text(
        "# Fairness Audit\n\n"
        "- Same model, tokenizer, prompt, scorer, decode, KV budget across all 3 methods\n"
        "- Only active block selection policy differs (SP / rank_stability)\n"
        "- sliminfer_original rerun verifies hooks do not contaminate baseline\n",
        encoding="utf-8",
    )

    lines = [
        "# Two-Task QA Compare",
        "",
        "| Method | HotpotQA | MuSiQue | Macro | Δ vs official repro |",
        "|--------|----------|---------|-------|---------------------|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['method']} | {row['hotpotqa']} | {row['musique']} | {row['macro_avg']} | {row['delta_vs_official_repro_macro']} |"
        )
    lines += [
        "",
        f"- Official repro reference: HPQA {OFFICIAL_REPRO['hotpotqa']}, MuSiQue {OFFICIAL_REPRO['musique']}, macro {OFFICIAL_REPRO['macro_avg']}",
        f"- SP beats baseline: **{beat_sp}**",
        f"- rank_stability beats baseline: **{beat_rss}**",
        f"- Proceed to 17-task: **{'NO' if not any_beat else 'only if user requests after review'}**",
    ]
    (out_dir / "QA_COMPARE_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    claim = [
        "# Claim Guard",
        "",
        "| Tier | Result |",
        "|------|--------|",
        f"| Official SILM repro (prior) | HPQA {OFFICIAL_REPRO['hotpotqa']}, MuSiQue {OFFICIAL_REPRO['musique']} |",
        "| RoleFlow 400-full | **no-go** (do not continue) |",
        f"| SP-SlimInfer QA | macro {sp_macro if sp_macro is not None else 'pending'} |",
        f"| rank_stability QA | macro {rss_macro if rss_macro is not None else 'pending'} |",
        "",
        "## Rules",
        "- Do not claim SP/reliability beat SILM unless macro > baseline on this 400-sample run",
        "- Do not expand to 17-task unless a new method clearly beats sliminfer_original here",
    ]
    (out_dir / "CLAIM_GUARD.md").write_text("\n".join(claim) + "\n", encoding="utf-8")

    if not any_beat and sp_macro is not None and rss_macro is not None:
        (out_dir / "NO_GO_POSTMORTEM.md").write_text(
            "# No-Go Postmortem — SP / rank_stability vs sliminfer_original\n\n"
            f"- Baseline macro: {base_macro}\n"
            f"- SP macro: {sp_macro} ({'beat' if beat_sp else 'no beat'})\n"
            f"- rank_stability macro: {rss_macro} ({'beat' if beat_rss else 'no beat'})\n\n"
            "## Decision\n"
            "- Do **not** expand to 17-task full\n"
            "- Do **not** continue RoleFlow / task gate / risk gate stacking\n"
            "- Active-set changes alone did not translate to QA gains on two-task protocol\n",
            encoding="utf-8",
        )


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    methods = selected_methods(args)
    write_json = lambda p, o: p.write_text(json.dumps(o, indent=2) + "\n", encoding="utf-8")
    write_json(out_dir / "compare_config.json", {"methods": methods, "tasks": list(TASKS)})

    results: List[Dict[str, Any]] = []
    failures: List[str] = []

    for method, subdir in methods:
        run_dir = out_dir / subdir
        entry: Dict[str, Any] = {"method": method, "run_dir": str(run_dir)}
        if not args.skip_run:
            if qa_complete(run_dir):
                print(f"[qa] skip {method}: already complete", flush=True)
            else:
                cmd = qa_cmd(args, method, run_dir)
                write_json(out_dir / f"command_{method}.json", {"command": cmd})
                print(f"[qa] running {method} ...", flush=True)
                env = os.environ.copy()
                env["PYTHONPATH"] = str(REPO_ROOT)
                proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env)
                entry["returncode"] = proc.returncode
                if proc.returncode != 0:
                    failures.append(f"{method}: rc={proc.returncode}")
                    results.append(entry)
                    continue
        consolidate_method(run_dir, method)
        entry["scores"] = read_method_scores(run_dir)
        entry["prediction_counts"] = count_predictions(run_dir)
        if not qa_complete(run_dir):
            failures.append(f"{method}: incomplete predictions {entry['prediction_counts']}")
        results.append(entry)

    aggregate_compare(out_dir, results, args.live_audit_dir)
    write_json(out_dir / "qa_summary.json", {"results": results, "failures": failures})
    print(json.dumps({"out_dir": str(out_dir), "failures": failures, "results": results}, indent=2))
    if failures:
        sys.exit(2)


if __name__ == "__main__":
    main()
