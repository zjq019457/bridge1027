#!/usr/bin/env python3
"""Qwen2.5 17-task LongBench full: sliminfer_original vs frozen cc_sliminfer_v1."""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.consolidate_official_repro_results import load_predictions, per_sample_score, write_csv

# Official SlimInfer LongBench English set (same order as experiments/run_qwen25_sliminfer_variants.py).
TASKS = (
    "passage_retrieval_en",
    "qasper",
    "multifieldqa_en",
    "hotpotqa",
    "2wikimqa",
    "musique",
    "gov_report",
    "qmsum",
    "multi_news",
    "vcsum",
    "passage_count",
    "trec",
    "triviaqa",
    "samsum",
    "lsht",
    "lcc",
    "repobench-p",
)
# Cap at 200 (lcc/repobench-p raw files have 500). multifieldqa_en has only 150.
EXPECTED_BY_TASK = {t: (150 if t == "multifieldqa_en" else 200) for t in TASKS}
MAX_SAMPLES = 200

CATEGORY_TASKS = {
    "Single-Doc QA": ["qasper", "multifieldqa_en"],
    "Multi-Doc QA": ["hotpotqa", "2wikimqa", "musique"],
    "Summarization": ["gov_report", "qmsum", "multi_news", "vcsum"],
    "Few-shot": ["trec", "triviaqa", "samsum", "lsht"],
    "Synthetic": ["passage_count", "passage_retrieval_en"],
    "Code": ["lcc", "repobench-p"],
}

METHODS = (
    ("sliminfer_original", "run_sliminfer_original"),
    ("cc_sliminfer_v1", "run_cc_sliminfer_v1"),
)
DEFAULT_MODEL = (
    "/home/yu-bai/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/"
    "snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
)
DEFAULT_LONGBENCH = Path("/home/yu-bai/Desktop/jiaqizhang/HeadKV/data/LongBench")
SIX_TASK_SRC = REPO_ROOT / "results/cc_sliminfer_v1_qwen25_6task_20260712_043551"
SIX_TASK_OVERLAP = ("qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", "triviaqa")


def parse_args() -> argparse.Namespace:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out_dir", type=Path, default=REPO_ROOT / f"results/cc_sliminfer_v1_qwen25_17task_{ts}")
    p.add_argument("--model_path", default=DEFAULT_MODEL)
    p.add_argument("--longbench_dir", default=str(DEFAULT_LONGBENCH))
    p.add_argument("--python", default=str(REPO_ROOT / ".venv/bin/python"))
    p.add_argument("--skip_run", action="store_true")
    p.add_argument("--no_reuse_6task", action="store_true")
    p.add_argument("--six_task_src", type=Path, default=SIX_TASK_SRC)
    return p.parse_args()


def qa_cmd(args: argparse.Namespace, method: str, run_dir: Path) -> List[str]:
    run_dir.mkdir(parents=True, exist_ok=True)
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
        "--longbench_max_samples", str(MAX_SAMPLES),
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


def count_preds(run_dir: Path) -> Dict[str, int]:
    pred = run_dir / "longbench_predictions"
    out = {}
    for t in TASKS:
        p = pred / f"{t}.jsonl"
        out[t] = sum(1 for l in p.read_text(encoding="utf-8").splitlines() if l.strip()) if p.exists() else 0
    return out


def complete(run_dir: Path) -> bool:
    c = count_preds(run_dir)
    return all(c.get(t, 0) >= EXPECTED_BY_TASK[t] for t in TASKS)


def read_scores(run_dir: Path) -> Dict[str, Any]:
    path = run_dir / "longbench_scores.csv"
    if not path.exists():
        return {}
    per = {}
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("status") == "ok" and row.get("score") not in (None, ""):
                per[row["dataset"]] = float(row["score"])
    if not per:
        return {}
    # Prefer full 17-task macro when all present; else mean of available.
    vals = [per[t] for t in TASKS if t in per]
    return {"per_task": per, "macro": round(sum(vals) / len(vals), 2) if vals else None}


def consolidate(run_dir: Path, method: str) -> List[Dict[str, Any]]:
    rows = load_predictions(run_dir / "longbench_predictions")
    out = []
    with (run_dir / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            sc = per_sample_score(row["task"], row["pred"], row["answers"], row["all_classes"])
            e = {**row, "method": method, "score": sc}
            out.append(e)
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    write_csv(
        run_dir / "scores_by_sample.csv",
        out,
        ["method", "task", "sample_id", "sample_index", "score", "length", "pred"],
    )
    return out


def paired_delta(base: List[Dict[str, Any]], var: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    bmap = {(r["task"], str(r["sample_id"])): r for r in base}
    out = []
    for v in var:
        b = bmap.get((v["task"], str(v["sample_id"])))
        if not b:
            continue
        bs, vs = float(b["score"]), float(v["score"])
        out.append({
            "method": v["method"],
            "task": v["task"],
            "sample_id": v["sample_id"],
            "baseline_score": bs,
            "variant_score": vs,
            "delta": round(vs - bs, 4),
            "pred_changed": int(str(b.get("pred", "")) != str(v.get("pred", ""))),
        })
    return out


def bootstrap_ci(deltas: Sequence[float], n: int = 10000, seed: int = 0) -> Dict[str, float]:
    rng = random.Random(seed)
    if not deltas:
        return {}
    macros = []
    for _ in range(n):
        sample = [deltas[rng.randrange(len(deltas))] for _ in range(len(deltas))]
        macros.append(sum(sample) / len(sample))
    macros.sort()
    return {
        "mean_delta": round(sum(deltas) / len(deltas), 4),
        "ci95_low": round(macros[int(0.025 * len(macros))], 4),
        "ci95_high": round(macros[int(0.975 * len(macros))], 4),
    }


def maybe_seed_from_6task(run_dir: Path, method_subdir: str, src_root: Path) -> None:
    src = src_root / method_subdir / "longbench_predictions"
    if not src.exists():
        return
    dst = run_dir / "longbench_predictions"
    dst.mkdir(parents=True, exist_ok=True)
    for task in SIX_TASK_OVERLAP:
        s = src / f"{task}.jsonl"
        t = dst / f"{task}.jsonl"
        if s.exists() and not t.exists():
            shutil.copy2(s, t)


def category_macros(per_task: Dict[str, float]) -> Dict[str, float]:
    out = {}
    for cat, tasks in CATEGORY_TASKS.items():
        vals = [per_task[t] for t in tasks if t in per_task]
        if vals:
            out[cat] = round(sum(vals) / len(vals), 2)
    return out


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    all_rows: List[Dict[str, Any]] = []
    scores_by_method: Dict[str, Dict[str, Any]] = {}

    for method, subdir in METHODS:
        run_dir = out_dir / subdir
        entry: Dict[str, Any] = {"method": method, "run_dir": str(run_dir)}
        if not args.no_reuse_6task:
            maybe_seed_from_6task(run_dir, subdir, args.six_task_src)
        if not args.skip_run and not complete(run_dir):
            cmd = qa_cmd(args, method, run_dir)
            (out_dir / f"command_{method}.json").write_text(json.dumps({"command": cmd}, indent=2) + "\n")
            env = os.environ.copy()
            env["PYTHONPATH"] = str(REPO_ROOT)
            log = out_dir / f"run_{method}.log"
            with log.open("w", encoding="utf-8") as lf:
                rc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
            entry["returncode"] = rc
            if rc != 0:
                results.append(entry)
                (out_dir / "FAILED.json").write_text(json.dumps(entry, indent=2) + "\n")
                print(json.dumps(entry, indent=2))
                return 1
        if not complete(run_dir):
            entry["error"] = "incomplete"
            entry["prediction_counts"] = count_preds(run_dir)
            results.append(entry)
            (out_dir / "FAILED.json").write_text(json.dumps(entry, indent=2) + "\n")
            return 1
        rows = consolidate(run_dir, method)
        all_rows.extend(rows)
        scores = read_scores(run_dir)
        scores_by_method[method] = scores
        entry["scores"] = scores
        entry["prediction_counts"] = count_preds(run_dir)
        results.append(entry)

    base_rows = [r for r in all_rows if r["method"] == "sliminfer_original"]
    cc_rows = [r for r in all_rows if r["method"] == "cc_sliminfer_v1"]
    deltas = paired_delta(base_rows, cc_rows)
    dv = [float(d["delta"]) for d in deltas]
    wins = sum(1 for d in dv if d > 0)
    losses = sum(1 for d in dv if d < 0)
    ties = sum(1 for d in dv if d == 0)
    changed = sum(int(d["pred_changed"]) for d in deltas)
    boot = bootstrap_ci(dv)
    base_scores = scores_by_method.get("sliminfer_original", {})
    cc_scores = scores_by_method.get("cc_sliminfer_v1", {})
    macro_delta = round(float(cc_scores.get("macro", 0)) - float(base_scores.get("macro", 0)), 2)

    task_rows = []
    for task in TASKS:
        b = base_scores.get("per_task", {}).get(task)
        c = cc_scores.get("per_task", {}).get(task)
        task_rows.append({
            "task": task,
            "sliminfer_original": b,
            "cc_sliminfer_v1": c,
            "delta": None if b is None or c is None else round(float(c) - float(b), 2),
            "n_expected": EXPECTED_BY_TASK[task],
        })

    base_cat = category_macros(base_scores.get("per_task", {}))
    cc_cat = category_macros(cc_scores.get("per_task", {}))
    cat_rows = []
    for cat in CATEGORY_TASKS:
        b, c = base_cat.get(cat), cc_cat.get(cat)
        cat_rows.append({
            "category": cat,
            "sliminfer_original": b,
            "cc_sliminfer_v1": c,
            "delta": None if b is None or c is None else round(float(c) - float(b), 2),
        })

    summary_rows = [
        {
            "method": "sliminfer_original",
            "macro_avg": base_scores.get("macro"),
            **{t: base_scores.get("per_task", {}).get(t) for t in TASKS},
        },
        {
            "method": "cc_sliminfer_v1",
            "macro_avg": cc_scores.get("macro"),
            "macro_delta": macro_delta,
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "changed_predictions": changed,
            **{t: cc_scores.get("per_task", {}).get(t) for t in TASKS},
        },
    ]
    write_csv(
        out_dir / "scores_summary.csv",
        summary_rows,
        ["method", "macro_avg", "macro_delta", "wins", "losses", "ties", "changed_predictions", *TASKS],
    )
    write_csv(
        out_dir / "scores_by_task.csv",
        task_rows,
        ["task", "sliminfer_original", "cc_sliminfer_v1", "delta", "n_expected"],
    )
    write_csv(
        out_dir / "scores_by_category.csv",
        cat_rows,
        ["category", "sliminfer_original", "cc_sliminfer_v1", "delta"],
    )
    write_csv(
        out_dir / "scores_by_sample.csv",
        all_rows,
        ["method", "task", "sample_id", "sample_index", "score", "length", "pred"],
    )
    with (out_dir / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    write_csv(
        out_dir / "per_sample_deltas.csv",
        deltas,
        ["method", "task", "sample_id", "baseline_score", "variant_score", "delta", "pred_changed"],
    )

    fairness = {
        "matched_model": True,
        "matched_prune_layers": [9, 18, 26],
        "matched_retain_tokens": [12288, 6144, 4096],
        "matched_block_size": 64,
        "matched_token_unit_size": 8,
        "longbench_max_samples": MAX_SAMPLES,
        "matched_tasks_and_n": EXPECTED_BY_TASK,
        "methods": ["sliminfer_original", "cc_sliminfer_v1"],
        "cc_hyperparams_frozen": {
            "cc_alpha": 0.25,
            "cc_min_cluster_len": 2,
            "cc_max_neighbors": 2,
            "cc_replacement_cap": 0.05,
        },
        "only_difference": "active-set construction via cluster_continuity_active_ids",
        "reused_6task_overlap": (not args.no_reuse_6task),
        "six_task_src": str(args.six_task_src),
    }
    (out_dir / "CC_17TASK_FAIRNESS_AUDIT.md").write_text(
        "# CC 17-task Fairness Audit\n\n"
        + json.dumps(fairness, indent=2)
        + "\n\nOfficial SlimInfer path; CC only changes final active-set under the same budget. "
        "See `CC_METHOD_SPEC.md`.\n",
        encoding="utf-8",
    )

    task_deltas = {r["task"]: r["delta"] for r in task_rows}
    cat_deltas = {r["category"]: r["delta"] for r in cat_rows}
    regress_tasks = [t for t, d in task_deltas.items() if d is not None and d <= -1.0]
    regress_cats = [c for c, d in cat_deltas.items() if d is not None and d < -1.0]
    positive = macro_delta > 0 and not regress_tasks and not regress_cats

    (out_dir / "CC_17TASK_CLAIM_GUARD.md").write_text(
        "# CC 17-task Claim Guard\n\n"
        f"- Macro Δ = **{macro_delta}**\n"
        f"- Per-task Δ = {task_deltas}\n"
        f"- Category Δ = {cat_deltas}\n"
        f"- Clear task regressions (Δ≤−1.0): {regress_tasks or 'none'}\n"
        f"- Category collapses (Δ<−1.0): {regress_cats or 'none'}\n\n"
        "## Allowed if complete and documented\n\n"
        "> On Qwen2.5-7B-Instruct, frozen `cc_sliminfer_v1` vs `sliminfer_original` on the "
        "official SlimInfer 17-task LongBench protocol (max 200/task; multifieldqa_en=150), "
        "matched KV budget/path.\n\n"
        "## Still forbidden without extra evidence\n\n"
        "- Cross-model claim\n"
        "- Latency / memory superiority from this QA-only pack\n"
        "- Retuning CC hyperparameters from these results\n",
        encoding="utf-8",
    )

    lines = [
        "# CC 17-task LongBench Report (`cc_sliminfer_v1`)\n\n",
        f"| Method | Macro | Δ |\n|--------|------:|--:|\n",
        f"| sliminfer_original | {base_scores.get('macro')} |  |\n",
        f"| cc_sliminfer_v1 | {cc_scores.get('macro')} | {macro_delta} |\n\n",
        f"Wins/Losses/Ties: {wins}/{losses}/{ties}; changed preds: {changed}\n\n",
        f"Bootstrap mean Δ={boot.get('mean_delta')} CI=[{boot.get('ci95_low')}, {boot.get('ci95_high')}]\n\n",
        "## Per-task\n\n",
        "| Task | original | cc_v1 | Δ | n |\n|------|---------:|------:|--:|--:|\n",
    ]
    for r in task_rows:
        lines.append(
            f"| {r['task']} | {r['sliminfer_original']} | {r['cc_sliminfer_v1']} | "
            f"{r['delta']} | {r['n_expected']} |\n"
        )
    lines.append("\n## Category macros\n\n| Category | original | cc_v1 | Δ |\n|----------|---------:|------:|--:|\n")
    for r in cat_rows:
        lines.append(
            f"| {r['category']} | {r['sliminfer_original']} | {r['cc_sliminfer_v1']} | {r['delta']} |\n"
        )
    lines.append(
        f"\n## Gate\n\n- macro_positive: {macro_delta > 0}\n"
        f"- no_clear_task_regression: {not regress_tasks}\n"
        f"- no_category_collapse: {not regress_cats}\n"
        f"- overall_pass: {positive}\n"
    )
    (out_dir / "CC_17TASK_REPORT.md").write_text("".join(lines), encoding="utf-8")
    (out_dir / "results.json").write_text(
        json.dumps(
            {
                "macro_delta": macro_delta,
                "bootstrap": boot,
                "task_deltas": task_deltas,
                "category_deltas": cat_deltas,
                "wins": wins,
                "losses": losses,
                "ties": ties,
                "overall_pass": positive,
                "methods": results,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"out_dir": str(out_dir), "macro_delta": macro_delta, "overall_pass": positive}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
