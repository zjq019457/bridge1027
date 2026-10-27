#!/usr/bin/env python3
"""Evaluate CC / UD / DivCov SlimInfer vs baseline on two-task 400 QA."""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.consolidate_official_repro_results import load_predictions, per_sample_score, write_csv

TASKS = ("hotpotqa", "musique")
EXPECTED = 200
METHODS = (
    ("cc_sliminfer", "run_cc_sliminfer"),
    ("ud_sliminfer", "run_ud_sliminfer"),
    ("divcov_sliminfer", "run_divcov_sliminfer"),
)
BASELINE_SRC = REPO_ROOT / "results/sp_sliminfer_v2_2task_full_20260710_210901/run_sliminfer_original"
DEFAULT_MODEL = (
    "/home/yu-bai/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/"
    "snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
)
DEFAULT_LONGBENCH = Path("/home/yu-bai/Desktop/jiaqizhang/HeadKV/data/LongBench")


def parse_args() -> argparse.Namespace:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out_dir", type=Path, default=REPO_ROOT / f"results/structural_sliminfer_2task_{ts}")
    p.add_argument("--model_path", default=DEFAULT_MODEL)
    p.add_argument("--longbench_dir", default=str(DEFAULT_LONGBENCH))
    p.add_argument("--baseline_src", type=Path, default=BASELINE_SRC)
    p.add_argument("--skip_run", action="store_true")
    p.add_argument("--python", default=str(REPO_ROOT / ".venv/bin/python"))
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
    return all(c.get(t, 0) >= EXPECTED for t in TASKS)


def read_scores(run_dir: Path) -> Dict[str, Any]:
    path = run_dir / "longbench_scores.csv"
    if not path.exists():
        return {}
    per = {}
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("status") == "ok" and row.get("score"):
                per[row["dataset"]] = float(row["score"])
    if not per:
        return {}
    return {"per_task": per, "macro": round(sum(per.values()) / len(per), 2)}


def consolidate(run_dir: Path, method: str) -> List[Dict[str, Any]]:
    rows = load_predictions(run_dir / "longbench_predictions")
    out = []
    with (run_dir / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            sc = per_sample_score(row["task"], row["pred"], row["answers"], row["all_classes"])
            e = {**row, "method": method, "score": sc}
            out.append(e)
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    write_csv(run_dir / "scores_by_sample.csv", out,
              ["method", "task", "sample_id", "sample_index", "score", "length", "pred"])
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


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    base_dst = out_dir / "run_sliminfer_original"
    if not base_dst.exists():
        if not complete(args.baseline_src):
            raise RuntimeError(f"baseline source incomplete: {args.baseline_src}")
        shutil.copytree(args.baseline_src, base_dst)

    base_rows = consolidate(base_dst, "sliminfer_original")
    base_scores = read_scores(base_dst)

    results: List[Dict[str, Any]] = []
    all_deltas: List[Dict[str, Any]] = []

    for method, subdir in METHODS:
        run_dir = out_dir / subdir
        entry: Dict[str, Any] = {"method": method, "run_dir": str(run_dir)}
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
                continue
        if not complete(run_dir):
            entry["error"] = "incomplete"
            results.append(entry)
            continue
        var_rows = consolidate(run_dir, method)
        var_scores = read_scores(run_dir)
        deltas = paired_delta(base_rows, var_rows)
        all_deltas.extend(deltas)
        dv = [float(d["delta"]) for d in deltas]
        wins = sum(1 for d in dv if d > 0)
        losses = sum(1 for d in dv if d < 0)
        ties = sum(1 for d in dv if d == 0)
        changed = sum(int(d["pred_changed"]) for d in deltas)
        boot = bootstrap_ci(dv)
        macro_delta = round(float(var_scores.get("macro", 0)) - float(base_scores.get("macro", 0)), 2)
        entry.update({
            "scores": var_scores,
            "macro_delta_vs_baseline": macro_delta,
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "changed_predictions": changed,
            "bootstrap": boot,
            "prediction_counts": count_preds(run_dir),
        })
        results.append(entry)

    summary_rows = [{
        "method": "sliminfer_original",
        "hotpotqa": base_scores.get("per_task", {}).get("hotpotqa"),
        "musique": base_scores.get("per_task", {}).get("musique"),
        "macro_avg": base_scores.get("macro"),
    }]
    for r in results:
        if "scores" not in r:
            continue
        per = r["scores"].get("per_task", {})
        summary_rows.append({
            "method": r["method"],
            "hotpotqa": per.get("hotpotqa"),
            "musique": per.get("musique"),
            "macro_avg": r["scores"].get("macro"),
            "macro_delta": r.get("macro_delta_vs_baseline"),
            "wins": r.get("wins"),
            "losses": r.get("losses"),
            "ties": r.get("ties"),
            "changed_predictions": r.get("changed_predictions"),
        })
    write_csv(out_dir / "scores_summary.csv", summary_rows,
              ["method", "hotpotqa", "musique", "macro_avg", "macro_delta", "wins", "losses", "ties", "changed_predictions"])
    write_csv(out_dir / "per_sample_deltas.csv", all_deltas,
              ["method", "task", "sample_id", "baseline_score", "variant_score", "delta", "pred_changed"])

    lines = ["# Structural SlimInfer Two-Task Results\n\n",
               "| Method | HotpotQA | MuSiQue | Macro | Δ macro | Wins | Losses | Changed preds |\n",
               "|--------|----------|---------|-------|---------|------|--------|---------------|\n"]
    for row in summary_rows:
        lines.append(
            f"| {row.get('method')} | {row.get('hotpotqa','')} | {row.get('musique','')} | "
            f"{row.get('macro_avg','')} | {row.get('macro_delta','')} | {row.get('wins','')} | "
            f"{row.get('losses','')} | {row.get('changed_predictions','')} |\n"
        )
    for r in results:
        if r.get("bootstrap"):
            b = r["bootstrap"]
            lines.append(f"\n### {r['method']} bootstrap\n\nmean Δ={b.get('mean_delta')} CI=[{b.get('ci95_low')}, {b.get('ci95_high')}]\n")
    (out_dir / "STRUCTURAL_SLIMINFER_REPORT.md").write_text("".join(lines), encoding="utf-8")
    (out_dir / "results.json").write_text(json.dumps({"baseline": base_scores, "methods": results}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "results": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
