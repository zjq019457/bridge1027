#!/usr/bin/env python3
"""6-task QA compare: topology CC policies vs original + always-on CC.

Reuses locked seeds for sliminfer_original / cc_sliminfer_v1 when present.
Runs (in order): cc_gated_v0, cc_diversify_v0, cc_coverage_v0, cc_router_v0.
cc_spike_hybrid_v0 is an alias of cc_gated_v0 and is not double-run.

Does not retune frozen CC operator hyperparameters.
"""
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
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.consolidate_official_repro_results import load_predictions, per_sample_score, write_csv

TASKS = (
    "qasper",
    "multifieldqa_en",
    "hotpotqa",
    "2wikimqa",
    "musique",
    "triviaqa",
)
EXPECTED_BY_TASK = {
    "qasper": 200,
    "multifieldqa_en": 150,
    "hotpotqa": 200,
    "2wikimqa": 200,
    "musique": 200,
    "triviaqa": 200,
}
# Reference always-on CC + original (locked 6-task expansion).
SEED_6TASK = REPO_ROOT / "results/cc_sliminfer_v1_qwen25_6task_20260712_043551"
SEED_17TASK = REPO_ROOT / "results/cc_sliminfer_v1_qwen25_17task_full_official_20260712_075238"

# New policies only (reference methods are seeded, not re-run unless missing).
NEW_METHODS = (
    ("cc_gated_v0", "run_cc_gated_v0"),
    ("cc_diversify_v0", "run_cc_diversify_v0"),
    ("cc_coverage_v0", "run_cc_coverage_v0"),
    ("cc_router_v0", "run_cc_router_v0"),
)
REF_METHODS = (
    ("sliminfer_original", "run_sliminfer_original"),
    ("cc_sliminfer_v1", "run_cc_sliminfer_v1"),
)
ALL_METHODS = REF_METHODS + NEW_METHODS

DEFAULT_MODEL = (
    "/home/yu-bai/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/"
    "snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
)
DEFAULT_LONGBENCH = Path("/home/yu-bai/Desktop/jiaqizhang/HeadKV/data/LongBench")
DEFAULT_GATE = REPO_ROOT / "configs" / "cc_gate_v0.json"


def parse_args() -> argparse.Namespace:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out_dir",
        type=Path,
        default=REPO_ROOT / f"results/cc_topology_6task_{ts}",
    )
    p.add_argument("--model_path", default=DEFAULT_MODEL)
    p.add_argument("--longbench_dir", default=str(DEFAULT_LONGBENCH))
    p.add_argument("--python", default=str(REPO_ROOT / ".venv/bin/python"))
    p.add_argument("--cc_gate_config_path", default=str(DEFAULT_GATE))
    p.add_argument("--seed_6task_dir", type=Path, default=SEED_6TASK)
    p.add_argument("--seed_17task_dir", type=Path, default=SEED_17TASK)
    p.add_argument(
        "--methods",
        type=str,
        default="cc_gated_v0,cc_diversify_v0,cc_coverage_v0,cc_router_v0",
        help="Comma list of NEW methods to run (refs always seeded/compared).",
    )
    p.add_argument("--skip_run", action="store_true")
    p.add_argument("--no_seed", action="store_true")
    return p.parse_args()


def qa_cmd(args: argparse.Namespace, method: str, run_dir: Path) -> List[str]:
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
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
        "--longbench_max_samples", "0",
        "--max_model_len", "131072",
        "--method", method if method != "cc_sliminfer_v1" else "cc_sliminfer_v1",
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
    if method in {m for m, _ in NEW_METHODS} or method == "cc_spike_hybrid_v0":
        audit = run_dir / "cc_selection_audit.jsonl"
        cmd.extend([
            "--cc_gate_config_path", str(args.cc_gate_config_path),
            "--cc_selection_audit_path", str(audit),
        ])
    return cmd


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
    return {"per_task": per, "macro": round(sum(per[t] for t in TASKS if t in per) / len([t for t in TASKS if t in per]), 2)}


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


def seed_method(out_run: Path, method_subdir: str, seed_roots: Sequence[Path]) -> Dict[str, Any]:
    """Copy per-task prediction jsonl for the 6 tasks from the first complete seed."""
    dst = out_run / "longbench_predictions"
    dst.mkdir(parents=True, exist_ok=True)
    report = {"copied": [], "missing": [], "source": None}
    for root in seed_roots:
        src_run = root / method_subdir
        src_pred = src_run / "longbench_predictions"
        if not src_pred.exists():
            continue
        ok = True
        for t in TASKS:
            sp = src_pred / f"{t}.jsonl"
            if not sp.exists():
                ok = False
                break
            n = sum(1 for l in sp.read_text(encoding="utf-8").splitlines() if l.strip())
            if n < EXPECTED_BY_TASK[t]:
                ok = False
                break
        if not ok:
            continue
        report["source"] = str(src_run)
        for t in TASKS:
            sp = src_pred / f"{t}.jsonl"
            tp = dst / f"{t}.jsonl"
            if tp.exists():
                existing = sum(1 for l in tp.read_text(encoding="utf-8").splitlines() if l.strip())
                if existing >= EXPECTED_BY_TASK[t]:
                    continue
            # Trim to expected n.
            lines = [l for l in sp.read_text(encoding="utf-8").splitlines() if l.strip()]
            tp.write_text("\n".join(lines[: EXPECTED_BY_TASK[t]]) + "\n", encoding="utf-8")
            report["copied"].append(t)
        # Also copy longbench_scores if present for ref methods.
        src_scores = src_run / "longbench_scores.csv"
        if src_scores.exists():
            shutil.copy2(src_scores, out_run / "longbench_scores.csv")
        return report
    report["missing"] = list(TASKS)
    return report


def gate_checklist(vs_orig: Mapping[str, float], vs_cc: Mapping[str, Optional[float]], macro_d_orig: float, macro_d_cc: float) -> Dict[str, Any]:
    """Applicability GO criteria from CC_TOPOLOGY_POLICY_DESIGN.md §10."""
    bad = [t for t, d in vs_orig.items() if d <= -0.5]
    musique_ok = vs_orig.get("musique", -999) >= 0.5
    macro_vs_cc_ok = macro_d_cc >= 0.0  # macro >= always-on CC
    return {
        "no_task_delta_le_minus_0_5": len(bad) == 0,
        "bad_tasks": bad,
        "musique_delta_ge_0_5": musique_ok,
        "musique_delta": vs_orig.get("musique"),
        "macro_ge_always_on_cc": macro_vs_cc_ok,
        "macro_delta_vs_original": macro_d_orig,
        "macro_delta_vs_cc": macro_d_cc,
        "GO": bool(len(bad) == 0 and musique_ok and macro_vs_cc_ok),
    }


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    wanted_new = [m.strip() for m in args.methods.split(",") if m.strip()]
    methods_plan: List[Tuple[str, str]] = list(REF_METHODS)
    for m, sub in NEW_METHODS:
        if m in wanted_new:
            methods_plan.append((m, sub))

    seed_roots = [p for p in (args.seed_6task_dir, args.seed_17task_dir) if p and p.exists()]
    seed_audit = {}
    if not args.no_seed:
        for method, subdir in REF_METHODS:
            seed_audit[method] = seed_method(out_dir / subdir, subdir, seed_roots)
        (out_dir / "SEED_REUSE_AUDIT.json").write_text(json.dumps(seed_audit, indent=2) + "\n")

    results: List[Dict[str, Any]] = []
    all_rows: List[Dict[str, Any]] = []
    scores_by_method: Dict[str, Dict[str, Any]] = {}

    for method, subdir in methods_plan:
        run_dir = out_dir / subdir
        entry: Dict[str, Any] = {"method": method, "run_dir": str(run_dir)}
        need_run = method in wanted_new or not complete(run_dir)
        # Never re-run refs if already complete after seed.
        if method in {m for m, _ in REF_METHODS} and complete(run_dir):
            need_run = False
        if not args.skip_run and need_run and not complete(run_dir):
            cmd = qa_cmd(args, method, run_dir)
            (out_dir / f"command_{method}.json").write_text(json.dumps({"command": cmd}, indent=2) + "\n")
            env = os.environ.copy()
            env["PYTHONPATH"] = str(REPO_ROOT)
            env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
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
        # If seeded refs lack scores csv, recompute via consolidate path still works;
        # but ensure longbench_scores exists by re-scoring if needed.
        if not (run_dir / "longbench_scores.csv").exists():
            # Build minimal scores from consolidate then write longbench_scores.
            rows_tmp = load_predictions(run_dir / "longbench_predictions")
            from longbench_eval import scorer
            by_task: Dict[str, List[Any]] = {t: [] for t in TASKS}
            for r in rows_tmp:
                by_task.setdefault(r["task"], []).append(r)
            score_rows = []
            for t in TASKS:
                items = by_task.get(t, [])
                preds = [x["pred"] for x in items]
                answers = [x["answers"] for x in items]
                lengths = [int(x.get("length") or 0) for x in items]
                ac = items[0].get("all_classes", []) if items else []
                sc = scorer(t, preds, answers, lengths, ac)
                score_rows.append({"dataset": t, "score": sc, "num_samples": len(items), "status": "ok", "error": ""})
            write_csv(run_dir / "longbench_scores.csv", score_rows, ["dataset", "score", "num_samples", "status", "error"])

        rows = consolidate(run_dir, method)
        all_rows.extend(rows)
        scores = read_scores(run_dir)
        scores_by_method[method] = scores
        entry["scores"] = scores
        entry["prediction_counts"] = count_preds(run_dir)
        results.append(entry)

    base_rows = [r for r in all_rows if r["method"] == "sliminfer_original"]
    base_scores = scores_by_method.get("sliminfer_original", {})
    cc_scores = scores_by_method.get("cc_sliminfer_v1", {})

    # Per-method vs original table
    method_names = [m for m, _ in methods_plan]
    task_rows = []
    for task in TASKS:
        row: Dict[str, Any] = {"task": task}
        b = base_scores.get("per_task", {}).get(task)
        row["sliminfer_original"] = b
        for m in method_names:
            if m == "sliminfer_original":
                continue
            c = scores_by_method.get(m, {}).get("per_task", {}).get(task)
            row[m] = c
            row[f"delta_{m}"] = None if b is None or c is None else round(float(c) - float(b), 2)
        task_rows.append(row)

    summary_rows = []
    checklist = {}
    for m in method_names:
        sc = scores_by_method.get(m, {})
        macro = sc.get("macro")
        d_orig = None if macro is None or base_scores.get("macro") is None else round(float(macro) - float(base_scores["macro"]), 2)
        d_cc = None if macro is None or cc_scores.get("macro") is None else round(float(macro) - float(cc_scores["macro"]), 2)
        entry = {
            "method": m,
            "macro_avg": macro,
            "macro_delta_vs_original": d_orig,
            "macro_delta_vs_cc_v1": d_cc,
            **{t: sc.get("per_task", {}).get(t) for t in TASKS},
        }
        if m != "sliminfer_original" and base_rows:
            var_rows = [r for r in all_rows if r["method"] == m]
            deltas = paired_delta(base_rows, var_rows)
            dv = [float(d["delta"]) for d in deltas]
            entry["wins"] = sum(1 for d in dv if d > 0)
            entry["losses"] = sum(1 for d in dv if d < 0)
            entry["ties"] = sum(1 for d in dv if d == 0)
            entry["changed_predictions"] = sum(int(d["pred_changed"]) for d in deltas)
            boot = bootstrap_ci(dv)
            entry.update({f"boot_{k}": v for k, v in boot.items()})
            vs_orig = {
                t: round(float(sc.get("per_task", {}).get(t, 0)) - float(base_scores.get("per_task", {}).get(t, 0)), 2)
                for t in TASKS
                if sc.get("per_task", {}).get(t) is not None and base_scores.get("per_task", {}).get(t) is not None
            }
            if m in wanted_new:
                checklist[m] = gate_checklist(vs_orig, {}, float(d_orig or 0), float(d_cc or 0))
        summary_rows.append(entry)

    write_csv(
        out_dir / "scores_summary.csv",
        summary_rows,
        [
            "method", "macro_avg", "macro_delta_vs_original", "macro_delta_vs_cc_v1",
            "wins", "losses", "ties", "changed_predictions",
            "boot_mean_delta", "boot_ci95_low", "boot_ci95_high",
            *TASKS,
        ],
    )
    # task csv with dynamic columns
    task_fields = ["task", "sliminfer_original"]
    for m in method_names:
        if m == "sliminfer_original":
            continue
        task_fields.append(m)
        task_fields.append(f"delta_{m}")
    write_csv(out_dir / "scores_by_task.csv", task_rows, task_fields)
    write_csv(
        out_dir / "scores_by_sample.csv",
        all_rows,
        ["method", "task", "sample_id", "sample_index", "score", "length", "pred"],
    )
    (out_dir / "GATE_CHECKLIST.json").write_text(json.dumps(checklist, indent=2) + "\n")

    # Report markdown
    lines = [
        "# CC Topology 6-task Compare\n\n",
        f"- gate_config: `{args.cc_gate_config_path}`\n",
        f"- seed_6task: `{args.seed_6task_dir}`\n",
        "- frozen CC operator: α=0.25, min_cluster_len=2, max_neighbors=2, replacement_cap=0.05\n\n",
        "## Macro\n\n",
        "| Method | Macro | Δ vs original | Δ vs always-on CC |\n",
        "|--------|------:|-------------:|------------------:|\n",
    ]
    for e in summary_rows:
        lines.append(
            f"| {e['method']} | {e.get('macro_avg')} | {e.get('macro_delta_vs_original')} | {e.get('macro_delta_vs_cc_v1')} |\n"
        )
    lines.append("\n## Per-task Δ vs original\n\n")
    lines.append("| Task | " + " | ".join(m for m in method_names if m != "sliminfer_original") + " |\n")
    lines.append("|------|" + "|".join(["------:" for m in method_names if m != "sliminfer_original"]) + "|\n")
    for tr in task_rows:
        cells = []
        for m in method_names:
            if m == "sliminfer_original":
                continue
            cells.append(str(tr.get(f"delta_{m}")))
        lines.append(f"| {tr['task']} | " + " | ".join(cells) + " |\n")
    lines.append("\n## Gate checklist (Applicability GO)\n\n")
    lines.append("```json\n" + json.dumps(checklist, indent=2) + "\n```\n")
    (out_dir / "CC_TOPOLOGY_6TASK_REPORT.md").write_text("".join(lines), encoding="utf-8")

    (out_dir / "results.json").write_text(json.dumps({"results": results, "checklist": checklist}, indent=2) + "\n")
    print(json.dumps({"out_dir": str(out_dir), "checklist": checklist, "macros": {e["method"]: e.get("macro_avg") for e in summary_rows}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
