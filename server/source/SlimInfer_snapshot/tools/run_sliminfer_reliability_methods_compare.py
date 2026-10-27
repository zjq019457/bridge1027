#!/usr/bin/env python3
"""Compare SlimInfer reliability methods vs official silm_original on two-task LongBench."""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]

METHODS = [
    ("sliminfer_original", "silm_original", "Official SlimInfer baseline"),
    ("confidence_gated_sliminfer", "confidence_gated_sliminfer", "Method 1: Confidence-Gated"),
    ("margin_stable_sliminfer", "margin_stable_sliminfer", "Method 2: Margin-Stable"),
    ("multi_window_sliminfer", "multi_window_sliminfer", "Method 3: Multi-Window Consensus"),
    ("adaptive_budget_sliminfer", "adaptive_budget_sliminfer", "Method 4: Adaptive-Budget"),
    ("reliability_gated_sliminfer", "reliability_gated_sliminfer", "Method 5: Reliability-Gated"),
]

OFFICIAL_TARGETS = {"hotpotqa": 56.94, "musique": 27.25}
PAPER_REPRO = {"hotpotqa": 57.43, "musique": 27.14}


def parse_args() -> argparse.Namespace:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_path", default=(
        "/home/yu-bai/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/"
        "snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
    ))
    p.add_argument("--longbench_dir", default="/home/yu-bai/Desktop/jiaqizhang/HeadKV/data/LongBench")
    p.add_argument("--tasks", default="hotpotqa,musique")
    p.add_argument("--max_samples_per_task", type=int, default=50)
    p.add_argument("--methods", default="", help="Comma-separated subset; default=all 6")
    p.add_argument("--out_dir", default=f"results/sliminfer_reliability_compare_{stamp}")
    p.add_argument("--python", default=str(REPO_ROOT / ".venv/bin/python"))
    p.add_argument("--skip_run", action="store_true", help="Only aggregate existing subdirs")
    return p.parse_args()


def base_cmd(args: argparse.Namespace, method: str, out_dir: Path) -> List[str]:
    return [
        args.python,
        str(REPO_ROOT / "experiments/run_qwen25_sliminfer_variants.py"),
        "--model_path", args.model_path,
        "--tokenizer_path", args.model_path,
        "--longbench_dir", args.longbench_dir,
        "--longbench_tasks", args.tasks,
        "--max_samples_per_task", str(args.max_samples_per_task),
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
        "--dtype", "fp16",
        "--device", "cuda",
        "--backend", "flash_attention_2",
        "--seed", "0",
        "--output_dir", str(out_dir),
    ]


def read_scores(run_dir: Path) -> Dict[str, Any]:
    csv_path = run_dir / "longbench_scores.csv"
    if not csv_path.exists():
        return {}
    rows: Dict[str, float] = {}
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status") == "ok" and row.get("score"):
                rows[row["dataset"]] = float(row["score"])
    if not rows:
        return {}
    macro = round(sum(rows.values()) / len(rows), 2)
    return {"per_task": rows, "macro": macro, "num_tasks": len(rows)}


def write_report(out_dir: Path, results: List[Dict[str, Any]]) -> None:
  baseline = next((r for r in results if r["label"] == "silm_original"), None)
  base_macro = baseline["scores"]["macro"] if baseline and baseline.get("scores") else None

  lines = [
      "# SlimInfer Reliability Methods Comparison",
      "",
      f"Output: `{out_dir}`",
      "",
      "## Scores (two-task macro)",
      "",
      "| Method | HPQA | MuSiQue | Macro | Δ vs SILM | vs paper repro |",
      "|--------|------|---------|-------|-----------|----------------|",
  ]
  for r in results:
      s = r.get("scores") or {}
      per = s.get("per_task", {})
      macro = s.get("macro", "")
      delta = ""
      if base_macro is not None and macro != "":
          delta = f"{float(macro) - float(base_macro):+.2f}"
      hp = per.get("hotpotqa", "")
      mq = per.get("musique", "")
      vs_repro = ""
      if hp != "" and mq != "":
          vs_repro = f"{(float(hp)+float(mq))/2 - (PAPER_REPRO['hotpotqa']+PAPER_REPRO['musique'])/2:+.2f}"
      lines.append(
          f"| {r['label']} | {hp} | {mq} | {macro} | {delta} | {vs_repro} |"
      )

  lines += [
      "",
      "## Reference",
      f"- Paper Table 1 targets: HPQA {OFFICIAL_TARGETS['hotpotqa']}, MuSiQue {OFFICIAL_TARGETS['musique']}",
      f"- Our official repro (400 full): HPQA {PAPER_REPRO['hotpotqa']}, MuSiQue {PAPER_REPRO['musique']}",
      "",
      "## Notes",
      "- adaptive_budget_sliminfer changes per-sample token budget (report budget separately for fair Pareto).",
      "- All other methods keep the same 12k/6k/4k nominal budget.",
  ]
  (out_dir / "RELIABILITY_METHODS_COMPARE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

  with (out_dir / "scores_summary.csv").open("w", encoding="utf-8", newline="") as f:
      w = csv.writer(f)
      w.writerow(["method", "hotpotqa", "musique", "macro", "delta_vs_silm_original"])
      for r in results:
          s = r.get("scores") or {}
          per = s.get("per_task", {})
          macro = s.get("macro", "")
          delta = ""
          if base_macro is not None and macro != "":
              delta = round(float(macro) - float(base_macro), 2)
          w.writerow([
              r["label"],
              per.get("hotpotqa", ""),
              per.get("musique", ""),
              macro,
              delta,
          ])


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.methods.strip() and not args.skip_run:
        selected = {m.strip() for m in args.methods.split(",") if m.strip()}
        run_list = [(cli, label, d) for cli, label, d in METHODS if cli in selected or label in selected]
    else:
        run_list = METHODS

    manifest = {"methods": run_list, "args": vars(args), "runs": []}
    results: List[Dict[str, Any]] = []

    for cli_method, label, desc in run_list:
        run_dir = out_dir / label
        entry: Dict[str, Any] = {
            "method": cli_method,
            "label": label,
            "description": desc,
            "run_dir": str(run_dir),
        }
        if not args.skip_run:
            cmd = base_cmd(args, cli_method, run_dir)
            print(f"\n=== Running {label} ({cli_method}) ===")
            print(" ".join(cmd))
            env = os.environ.copy()
            env["PYTHONPATH"] = str(REPO_ROOT)
            proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env)
            entry["returncode"] = proc.returncode
            if proc.returncode != 0:
                entry["scores"] = {}
                manifest["runs"].append(entry)
                results.append(entry)
                continue
        entry["scores"] = read_scores(run_dir)
        if args.skip_run and not entry["scores"]:
            entry["returncode"] = entry.get("returncode", -1)
        manifest["runs"].append(entry)
        results.append(entry)

    if args.skip_run and len(results) < len(METHODS):
        seen = {r["label"] for r in results}
        for cli_method, label, desc in METHODS:
            if label in seen:
                continue
            run_dir = out_dir / label
            results.append({
                "method": cli_method,
                "label": label,
                "description": desc,
                "run_dir": str(run_dir),
                "scores": read_scores(run_dir),
            })
        results.sort(key=lambda r: [x[1] for x in METHODS].index(r["label"]))

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_report(out_dir, results)
    print(f"\nWrote {out_dir / 'RELIABILITY_METHODS_COMPARE.md'}")


if __name__ == "__main__":
    main()
