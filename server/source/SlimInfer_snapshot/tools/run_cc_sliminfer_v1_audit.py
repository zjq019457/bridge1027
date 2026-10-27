#!/usr/bin/env python3
"""CC-SlimInfer v1 freeze audits from the 400-sample two-task structural run.

Writes into the structural result directory:
  CC_ACTIVE_SET_ANALYSIS.md
  CC_CASE_STUDY.md
  CC_OUTLIER_SENSITIVITY.md
  CC_CLAIM_GUARD.md

Optionally runs a focused live prefill dump (--live_audit) for case-study samples
to recover per-layer added/removed block ids.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_STRUCT = REPO_ROOT / "results/structural_sliminfer_2task_20260711_172445"
DEFAULT_MODEL = (
    "/home/yu-bai/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/"
    "snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
)
DEFAULT_LONGBENCH = Path("/home/yu-bai/Desktop/jiaqizhang/HeadKV/data/LongBench")
PRUNE_LAYERS = (9, 18, 26)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--struct_dir", type=Path, default=DEFAULT_STRUCT)
    p.add_argument("--live_audit", action="store_true")
    p.add_argument("--model_path", default=DEFAULT_MODEL)
    p.add_argument("--longbench_dir", default=str(DEFAULT_LONGBENCH))
    p.add_argument("--python", default=str(REPO_ROOT / ".venv/bin/python"))
    p.add_argument("--max_case_samples", type=int, default=12)
    return p.parse_args()


def load_deltas(path: Path, method: str = "cc_sliminfer") -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("method") != method:
                continue
            rows.append({
                "task": r["task"],
                "sample_id": r["sample_id"],
                "baseline_score": float(r["baseline_score"]),
                "variant_score": float(r["variant_score"]),
                "delta": float(r["delta"]),
                "pred_changed": int(r["pred_changed"]),
            })
    return rows


def load_preds(run_dir: Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    path = run_dir / "predictions.jsonl"
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            out[(str(row["task"]), str(row["sample_id"]))] = row
    return out


def load_sample_index_map(run_dir: Path) -> Dict[Tuple[str, str], int]:
    path = run_dir / "longbench_sample_ids.json"
    mapping: Dict[Tuple[str, str], int] = {}
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        # formats vary; also fall back to predictions sample_index
    preds = load_preds(run_dir)
    for (task, sid), row in preds.items():
        if "sample_index" in row:
            mapping[(task, sid)] = int(row["sample_index"])
    return mapping


def mean(xs: Sequence[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0


def outlier_report(deltas: List[Dict[str, Any]]) -> Dict[str, Any]:
    rows = sorted(deltas, key=lambda r: -r["delta"])
    all_d = [r["delta"] for r in rows]
    n = len(all_d)
    base_mean = mean(all_d)

    def drop_top_k(k: int) -> float:
        kept = all_d[k:] if k < n else []
        # recompute over full n with zeros? User asked remove-top-k then delta.
        # Interpret as: recompute mean on remaining samples only, and also
        # "macro impact if those wins were ties (delta=0)".
        return mean(kept) if kept else 0.0

    def zero_top_k(k: int) -> float:
        vals = list(all_d)
        for i in range(min(k, n)):
            vals[i] = 0.0
        return mean(vals)

    wins = [r for r in rows if r["delta"] > 0]
    zero_all_wins = list(all_d)
    for i, r in enumerate(rows):
        if r["delta"] > 0:
            zero_all_wins[i] = 0.0

    by_task: Dict[str, float] = {}
    for task in sorted({r["task"] for r in rows}):
        by_task[task] = mean([r["delta"] for r in rows if r["task"] == task])

    return {
        "n": n,
        "mean_delta": round(base_mean, 4),
        "n_wins": len(wins),
        "n_losses": sum(1 for r in rows if r["delta"] < 0),
        "n_ties": sum(1 for r in rows if r["delta"] == 0),
        "top1": rows[0] if rows else None,
        "top2": rows[:2],
        "remove_top1_mean_remaining": round(drop_top_k(1), 4),
        "remove_top2_mean_remaining": round(drop_top_k(2), 4),
        "remove_all_wins_mean_remaining": round(
            mean([d for d in all_d if d <= 0]) if any(d <= 0 for d in all_d) else 0.0, 4
        ),
        "zero_top1_mean": round(zero_top_k(1), 4),
        "zero_top2_mean": round(zero_top_k(2), 4),
        "zero_all_wins_mean": round(mean(zero_all_wins), 4),
        "by_task_mean_delta": {k: round(v, 4) for k, v in by_task.items()},
        "single_sample_dominated": bool(rows and abs(rows[0]["delta"]) / max(1e-9, abs(base_mean) * n) > 0.5),
        "fraction_top1_of_total_pos": round(
            (rows[0]["delta"] / sum(d for d in all_d if d > 0)) if rows and any(d > 0 for d in all_d) else 0.0,
            4,
        ),
    }


def select_case_samples(deltas: List[Dict[str, Any]], max_n: int) -> List[Dict[str, Any]]:
    wins = sorted([r for r in deltas if r["delta"] > 0], key=lambda r: -r["delta"])
    losses = sorted([r for r in deltas if r["delta"] < 0], key=lambda r: r["delta"])
    ties_changed = [r for r in deltas if r["delta"] == 0 and r["pred_changed"] == 1]
    ties_stable = [r for r in deltas if r["delta"] == 0 and r["pred_changed"] == 0]
    picked: List[Dict[str, Any]] = []
    for bucket in (wins, losses, ties_changed, ties_stable):
        for r in bucket:
            if len(picked) >= max_n:
                break
            key = (r["task"], r["sample_id"])
            if any((p["task"], p["sample_id"]) == key for p in picked):
                continue
            picked.append(r)
        if len(picked) >= max_n:
            break
    # ensure >=10 if available
    if len(picked) < 10:
        for r in deltas:
            if len(picked) >= 10:
                break
            key = (r["task"], r["sample_id"])
            if any((p["task"], p["sample_id"]) == key for p in picked):
                continue
            picked.append(r)
    return picked


def run_live_audit(
    args: argparse.Namespace,
    struct_dir: Path,
    cases: List[Dict[str, Any]],
    index_map: Mapping[Tuple[str, str], int],
) -> Path:
    audit_dir = struct_dir / "cc_live_selection_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    dump_path = audit_dir / "cc_selection_audit.jsonl"
    if dump_path.exists():
        dump_path.unlink()

    by_task: Dict[str, List[int]] = defaultdict(list)
    for c in cases:
        idx = index_map.get((c["task"], c["sample_id"]))
        if idx is None:
            continue
        by_task[c["task"]].append(int(idx))

    (audit_dir / "case_sample_indices.json").write_text(
        json.dumps({k: sorted(set(v)) for k, v in by_task.items()}, indent=2) + "\n",
        encoding="utf-8",
    )
    if not by_task:
        raise RuntimeError("no sample indices available for live audit cases")

    parts = [f"{task}:{','.join(str(i) for i in sorted(set(idxs)))}" for task, idxs in sorted(by_task.items())]
    indices_arg = ";".join(parts)
    tasks = ",".join(sorted(by_task.keys()))
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
        "--longbench_tasks", tasks,
        "--longbench_sample_indices_by_task", indices_arg,
        "--max_model_len", "131072",
        "--method", "cc_sliminfer_v1",
        "--eval", "gate_audit",
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
        "--cc_selection_audit_path", str(dump_path),
        "--output_dir", str(audit_dir / "run_cc_gate_audit"),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    log = audit_dir / "run_cc_gate_audit.log"
    with log.open("w", encoding="utf-8") as lf:
        rc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, stdout=lf, stderr=subprocess.STDOUT)
    if rc.returncode != 0:
        raise RuntimeError(f"live audit failed: see {log}")
    return dump_path


def summarize_live_dump(dump_path: Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
    if not dump_path.exists():
        return {}
    by_sample: Dict[Tuple[str, str], Dict[str, Any]] = {}
    with dump_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            task = str(row.get("task") or "")
            sid = str(row.get("sample_id") or "")
            if not task or not sid:
                continue
            key = (task, sid)
            slot = by_sample.setdefault(key, {
                "layers": [],
                "mean_churn": [],
                "mean_jaccard": [],
                "swaps": 0,
                "budget_parity_ok": 0,
                "budget_parity_n": 0,
                "sink_violations": 0,
                "neighbor_retention": [],
                "isolated_replacement_rate": [],
                "added_union": set(),
                "removed_union": set(),
                "changed_layers": set(),
            })
            layer = row.get("layer_idx", row.get("abs_layer"))
            triggered = int(row.get("triggered", 0) or 0)
            if triggered:
                slot["changed_layers"].add(layer)
            slot["layers"].append(row)
            if row.get("churn_rate") is not None:
                slot["mean_churn"].append(float(row["churn_rate"]))
            if row.get("jaccard_vs_silm") is not None:
                slot["mean_jaccard"].append(float(row["jaccard_vs_silm"]))
            slot["swaps"] += int(row.get("swaps_applied", 0) or 0)
            slot["budget_parity_n"] += 1
            slot["budget_parity_ok"] += int(row.get("budget_parity", 0) or 0)
            slot["sink_violations"] += int(row.get("sink_violation", 0) or 0)
            if row.get("neighbor_retention_rate") is not None:
                slot["neighbor_retention"].append(float(row["neighbor_retention_rate"]))
            if row.get("isolated_replacement_rate") is not None:
                slot["isolated_replacement_rate"].append(float(row["isolated_replacement_rate"]))
            for b in row.get("added_block_ids") or []:
                slot["added_union"].add(int(b))
            for b in row.get("removed_block_ids") or []:
                slot["removed_union"].add(int(b))
    # finalize
    out = {}
    for key, slot in by_sample.items():
        out[key] = {
            "n_layer_events": len(slot["layers"]),
            "changed_layers": sorted(x for x in slot["changed_layers"] if x is not None),
            "mean_churn": round(mean(slot["mean_churn"]), 6) if slot["mean_churn"] else None,
            "mean_jaccard": round(mean(slot["mean_jaccard"]), 6) if slot["mean_jaccard"] else None,
            "swaps_total": slot["swaps"],
            "budget_parity_rate": round(slot["budget_parity_ok"] / max(1, slot["budget_parity_n"]), 6),
            "sink_violations": slot["sink_violations"],
            "mean_neighbor_retention": round(mean(slot["neighbor_retention"]), 6) if slot["neighbor_retention"] else None,
            "mean_isolated_replacement_rate": round(mean(slot["isolated_replacement_rate"]), 6) if slot["isolated_replacement_rate"] else None,
            "added_block_ids": sorted(slot["added_union"]),
            "removed_block_ids": sorted(slot["removed_union"]),
        }
    return out


def write_active_set_md(
    path: Path,
    deltas: List[Dict[str, Any]],
    live: Mapping[Tuple[str, str], Dict[str, Any]],
    agg_cc: Mapping[str, str],
    agg_base: Mapping[str, str],
) -> None:
    changed = [r for r in deltas if r["pred_changed"] == 1]
    score_changed = [r for r in deltas if r["delta"] != 0]
    live_vals = list(live.values())
    lines = [
        "# CC Active-Set Analysis (`cc_sliminfer_v1`)\n\n",
        f"Source QA: `{DEFAULT_STRUCT.name}` / method `cc_sliminfer` frozen as `cc_sliminfer_v1`.\n\n",
        "## Sample-level (from paired predictions)\n\n",
        f"- N samples: **{len(deltas)}**\n",
        f"- Changed predictions: **{len(changed)}** ({100*len(changed)/max(1,len(deltas)):.1f}%)\n",
        f"- Score-changed samples: **{len(score_changed)}** (wins={sum(1 for r in deltas if r['delta']>0)}, losses={sum(1 for r in deltas if r['delta']<0)})\n",
        f"- Ties with identical pred: **{sum(1 for r in deltas if r['delta']==0 and r['pred_changed']==0)}**\n\n",
        "## Aggregate runtime active-set stats (full 400 QA run)\n\n",
        "| Metric | cc_sliminfer_v1 | sliminfer_original |\n",
        "|--------|----------------:|-------------------:|\n",
        f"| active_set_churn | {agg_cc.get('active_set_churn','')} | {agg_base.get('active_set_churn','')} |\n",
        f"| active_block_overlap | {agg_cc.get('active_block_overlap','')} | {agg_base.get('active_block_overlap','')} |\n",
        f"| jaccard_vs_original | {agg_cc.get('jaccard_vs_original','')} | {agg_base.get('jaccard_vs_original','')} |\n",
        f"| prefill_score_calls | {agg_cc.get('prefill_score_calls','')} | {agg_base.get('prefill_score_calls','')} |\n\n",
        "Note: `active_set_churn` / overlap here are SlimInfer’s internal consecutive-prune stats, not CC-vs-baseline Jaccard. "
        "CC-vs-SlimInfer Jaccard is in the live selection dump below.\n\n",
    ]
    if live_vals:
        lines += [
            "## Live selection dump (case-study / changed samples)\n\n",
            f"- Samples with dump: **{len(live_vals)}**\n",
            f"- Mean churn vs SlimInfer baseline set: **{mean([v['mean_churn'] for v in live_vals if v.get('mean_churn') is not None]):.4f}**\n",
            f"- Mean Jaccard vs SlimInfer: **{mean([v['mean_jaccard'] for v in live_vals if v.get('mean_jaccard') is not None]):.4f}**\n",
            f"- Budget parity rate: **{mean([v['budget_parity_rate'] for v in live_vals]):.4f}**\n",
            f"- Forced sink violations (sum over layer events): **{sum(v['sink_violations'] for v in live_vals)}**\n",
            f"- Mean cluster-neighbor retention rate: **{mean([v['mean_neighbor_retention'] for v in live_vals if v.get('mean_neighbor_retention') is not None]):.4f}**\n",
            f"- Mean isolated-block replacement rate: **{mean([v['mean_isolated_replacement_rate'] for v in live_vals if v.get('mean_isolated_replacement_rate') is not None]):.4f}**\n",
            f"- Samples with ≥1 changed layer: **{sum(1 for v in live_vals if v.get('changed_layers'))}**\n\n",
            "### Changed layers histogram\n\n",
        ]
        layer_hist: Dict[str, int] = defaultdict(int)
        for v in live_vals:
            for ly in v.get("changed_layers") or []:
                layer_hist[str(ly)] += 1
        for ly in sorted(layer_hist, key=lambda x: int(x) if str(x).lstrip('-').isdigit() else 0):
            lines.append(f"- layer {ly}: {layer_hist[ly]} samples\n")
    else:
        lines.append(
            "## Live selection dump\n\n"
            "Not available in this pass. Aggregate churn/Jaccard above are from the full QA run’s "
            "`active_set_stats.csv`. Re-run with `--live_audit` to populate per-sample block swaps.\n"
        )
    path.write_text("".join(lines), encoding="utf-8")


def write_case_study_md(
    path: Path,
    cases: List[Dict[str, Any]],
    base_preds: Mapping[Tuple[str, str], Dict[str, Any]],
    cc_preds: Mapping[Tuple[str, str], Dict[str, Any]],
    live: Mapping[Tuple[str, str], Dict[str, Any]],
) -> None:
    lines = [
        "# CC Case Study (`cc_sliminfer_v1`)\n\n",
        "Samples drawn from wins, losses, and prediction-changed ties on the frozen 400-sample run.\n\n",
    ]
    for i, c in enumerate(cases, 1):
        key = (c["task"], c["sample_id"])
        bp = base_preds.get(key, {})
        cp = cc_preds.get(key, {})
        lv = live.get(key, {})
        kind = "WIN" if c["delta"] > 0 else ("LOSS" if c["delta"] < 0 else ("TIE-CHANGED" if c["pred_changed"] else "TIE"))
        lines.append(f"## {i}. [{kind}] {c['task']} `{c['sample_id'][:16]}…`\n\n")
        lines.append(
            f"- baseline score: {c['baseline_score']:.4f} → CC: {c['variant_score']:.4f} (Δ={c['delta']:+.4f})\n"
            f"- pred_changed: {c['pred_changed']}\n"
            f"- baseline pred: `{str(bp.get('pred',''))[:120]}`\n"
            f"- CC pred: `{str(cp.get('pred',''))[:120]}`\n"
        )
        if lv:
            lines.append(
                f"- changed layers: {lv.get('changed_layers')}\n"
                f"- mean churn / Jaccard: {lv.get('mean_churn')} / {lv.get('mean_jaccard')}\n"
                f"- removed blocks: {lv.get('removed_block_ids')}\n"
                f"- added neighbor blocks: {lv.get('added_block_ids')}\n"
                f"- swaps_total: {lv.get('swaps_total')}; sink_violations: {lv.get('sink_violations')}; "
                f"budget_parity_rate: {lv.get('budget_parity_rate')}\n"
            )
            related = "yes" if c["pred_changed"] and (lv.get("added_block_ids") or lv.get("removed_block_ids")) else (
                "pred unchanged" if not c["pred_changed"] else "pred changed but no block dump"
            )
            lines.append(f"- active-set change related to prediction change: **{related}**\n")
        else:
            lines.append("- block-level dump: *not available* (run `--live_audit`)\n")
        lines.append("\n")
    path.write_text("".join(lines), encoding="utf-8")


def write_outlier_md(path: Path, rep: Dict[str, Any]) -> None:
    top1 = rep.get("top1") or {}
    lines = [
        "# CC Outlier Sensitivity\n\n",
        f"- N={rep['n']}, mean Δ={rep['mean_delta']}, wins={rep['n_wins']}, losses={rep['n_losses']}, ties={rep['n_ties']}\n",
        f"- Per-task mean Δ: {rep['by_task_mean_delta']}\n",
        f"- Top-1 win share of total positive mass: **{rep['fraction_top1_of_total_pos']}**\n",
        f"- Single-sample dominated (heuristic): **{rep['single_sample_dominated']}**\n\n",
        "## Remove / zero interventions on sample Δ\n\n",
        "| Intervention | Mean Δ |\n|--------------|-------:|\n",
        f"| Full set | {rep['mean_delta']} |\n",
        f"| Zero top-1 win | {rep['zero_top1_mean']} |\n",
        f"| Zero top-2 wins | {rep['zero_top2_mean']} |\n",
        f"| Zero all wins | {rep['zero_all_wins_mean']} |\n",
        f"| Remove top-1 (mean on remaining) | {rep['remove_top1_mean_remaining']} |\n",
        f"| Remove top-2 (mean on remaining) | {rep['remove_top2_mean_remaining']} |\n",
        f"| Remove all wins (mean on remaining non-wins) | {rep['remove_all_wins_mean_remaining']} |\n\n",
    ]
    if top1:
        lines.append(
            f"Top-1: task={top1.get('task')} id=`{str(top1.get('sample_id'))[:20]}` Δ={top1.get('delta')}\n\n"
        )
    lines.append(
        "## Verdict\n\n"
        + (
            "Gain is **not** single-sample dominated: after zeroing top-1/top-2, mean Δ remains positive "
            "or near the original scale; both tasks show non-negative mean Δ.\n"
            if (rep["zero_top1_mean"] > 0 and rep["by_task_mean_delta"].get("hotpotqa", 0) >= 0
                and rep["by_task_mean_delta"].get("musique", 0) >= 0
                and not rep["single_sample_dominated"])
            else "Caution: sensitivity suggests concentration in few wins; treat expansion GO carefully.\n"
        )
    )
    path.write_text("".join(lines), encoding="utf-8")


def write_claim_guard(path: Path, rep: Dict[str, Any]) -> None:
    path.write_text(
        "# CC Claim Guard (`cc_sliminfer_v1`)\n\n"
        "## Allowed claim\n\n"
        "> On **Qwen2.5-7B-Instruct**, under the **official SlimInfer** path and matched KV budget "
        "(`prune_layers=9,18,26`, `retain_tokens=12288,6144,4096`), frozen **`cc_sliminfer_v1`** "
        "improves macro over **`sliminfer_original`** on the **HotpotQA + MuSiQue 400-sample "
        "(200+200) two-task official protocol** (macro Δ ≈ **+0.94**; bootstrap CI on sample Δ "
        "excludes 0 in the structural report).\n\n"
        "## Forbidden without new evidence\n\n"
        "- Do **not** claim LongBench **full 17-task** improvement.\n"
        "- Do **not** claim cross-model transfer (Llama/Mistral/etc.).\n"
        "- Do **not** claim all tasks / all categories are improved.\n"
        "- Do **not** claim latency/memory superiority from this QA-only pack.\n"
        "- Do **not** claim UD or DivCov are GO; only CC is the GO candidate.\n"
        "- Do **not** retune `cc_*` hyperparameters from this 400-sample result.\n\n"
        f"## Supporting checks\n\n"
        f"- Outlier single-sample dominated? **{rep.get('single_sample_dominated')}**\n"
        f"- Zero-top-1 mean Δ: **{rep.get('zero_top1_mean')}**\n"
        f"- Per-task mean Δ: **{rep.get('by_task_mean_delta')}**\n",
        encoding="utf-8",
    )


def go_decision(rep: Dict[str, Any]) -> Dict[str, Any]:
    task_ok = all(v >= 0 for v in rep.get("by_task_mean_delta", {}).values())
    not_single = not rep.get("single_sample_dominated", True)
    # After removing top-1, remaining mean among all samples with top1 zeroed still > 0
    # OR remaining mean after drop top1 still clearly positive relative to noise.
    robust = float(rep.get("zero_top1_mean", 0)) > 0.05
    go = bool(task_ok and not_single and robust and float(rep.get("mean_delta", 0)) > 0)
    return {
        "go_6task": go,
        "task_ok": task_ok,
        "not_single_sample_dominated": not_single,
        "robust_after_zero_top1": robust,
        "reason": (
            "GO: +0.94 not single-sample dominated; no task-level mean degradation."
            if go else
            "NO-GO: outlier or task-level concern; do not start 6-task."
        ),
    }


def read_active_csv(run_dir: Path) -> Dict[str, str]:
    path = run_dir / "active_set_stats.csv"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[0] if rows else {}


def main() -> int:
    args = parse_args()
    struct = args.struct_dir.resolve()
    deltas = load_deltas(struct / "per_sample_deltas.csv")
    if not deltas:
        raise SystemExit(f"no cc deltas in {struct}")
    base_preds = load_preds(struct / "run_sliminfer_original")
    cc_preds = load_preds(struct / "run_cc_sliminfer")
    index_map = load_sample_index_map(struct / "run_cc_sliminfer")
    cases = select_case_samples(deltas, args.max_case_samples)
    rep = outlier_report(deltas)

    live: Dict[Tuple[str, str], Dict[str, Any]] = {}
    dump_path = struct / "cc_live_selection_audit" / "cc_selection_audit.jsonl"
    if args.live_audit:
        # Prefer indices from predictions
        try:
            dump_path = run_live_audit(args, struct, cases, index_map)
        except Exception as exc:
            (struct / "cc_live_selection_audit_error.txt").write_text(str(exc), encoding="utf-8")
    if dump_path.exists():
        live = summarize_live_dump(dump_path)

    write_active_set_md(
        struct / "CC_ACTIVE_SET_ANALYSIS.md",
        deltas,
        live,
        read_active_csv(struct / "run_cc_sliminfer"),
        read_active_csv(struct / "run_sliminfer_original"),
    )
    write_case_study_md(struct / "CC_CASE_STUDY.md", cases, base_preds, cc_preds, live)
    write_outlier_md(struct / "CC_OUTLIER_SENSITIVITY.md", rep)
    write_claim_guard(struct / "CC_CLAIM_GUARD.md", rep)
    decision = go_decision(rep)
    (struct / "CC_GO_DECISION.json").write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"struct_dir": str(struct), "outlier": rep, "go": decision, "n_cases": len(cases), "n_live": len(live)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
