#!/usr/bin/env python3
"""Phase-1B: Spike Protection calibration + DRM audit-semantic fix (no QA).

Freeze: results/acc_drm_v0_audit (pointer to Phase-1A).
Only spike joint thresholds change for acc_drm_v1.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_MODEL = (
    "/home/yu-bai/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/"
    "snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
)
DEFAULT_LONGBENCH = Path("/home/yu-bai/Desktop/jiaqizhang/HeadKV/data/LongBench")
V0_DIR = REPO_ROOT / "results" / "acc_drm_audit_20260714_083153"
TASKS = ("musique", "2wikimqa", "triviaqa")
MAIN_METHODS = (
    "sliminfer_original",
    "cc_sliminfer_v1",
    "acc_sliminfer_v1",
    "drm_swap_sliminfer",
    "drm_acc_swap_sliminfer_v1",
)

# Frozen Phase-1B spike joint thresholds (global).
TAU_SPIKE = 0.28
TAU_MARGIN = 0.20
TAU_SPIKE_NEI_MAX = 0.18


def parse_args() -> argparse.Namespace:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out_dir",
        type=Path,
        default=REPO_ROOT / f"results/acc_drm_phase1b_spike_calibration_{ts}",
    )
    p.add_argument("--model_path", default=DEFAULT_MODEL)
    p.add_argument("--longbench_dir", default=str(DEFAULT_LONGBENCH))
    p.add_argument("--python", default=str(REPO_ROOT / ".venv/bin/python"))
    p.add_argument("--max_samples", type=int, default=24)
    p.add_argument("--v0_dir", type=Path, default=V0_DIR)
    p.add_argument("--skip_run", action="store_true")
    return p.parse_args()


def quantile(xs: Sequence[float], p: float) -> float:
    if not xs:
        return float("nan")
    ys = sorted(float(x) for x in xs)
    i = (len(ys) - 1) * p
    lo = int(math.floor(i))
    hi = int(math.ceil(i))
    if lo == hi:
        return ys[lo]
    return ys[lo] * (hi - i) + ys[hi] * (i - lo)


def qa_cmd(args: argparse.Namespace, method: str, run_dir: Path) -> List[str]:
    run_dir.mkdir(parents=True, exist_ok=True)
    audit = run_dir / "selection_audit.jsonl"
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
        "--longbench_max_samples", str(int(args.max_samples)),
        "--max_model_len", "131072",
        "--method", method,
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
        "--cc_selection_audit_path", str(audit),
        "--output_dir", str(run_dir),
    ]


def load_audit(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: List[Dict[str, Any]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def mean(xs: Sequence[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0


def feat_of(r: Mapping[str, Any]) -> Dict[str, float]:
    f = r.get("features") if isinstance(r.get("features"), dict) else {}
    if not f and isinstance(r.get("acc"), dict):
        f = r["acc"].get("features") or {}
    return {
        "isolated_spike_ratio": float(f.get("isolated_spike_ratio") or 0.0),
        "topk_margin": float(f.get("topk_margin") or 0.0),
        "neighbor_score_ratio": float(f.get("neighbor_score_ratio") or 0.0),
    }


def write_spike_calibration_spec(out_dir: Path, v0_dir: Path) -> Dict[str, Any]:
    v0_rows = load_audit(v0_dir / "run_acc_sliminfer" / "selection_audit.jsonl")
    dist_rows = []
    for r in v0_rows:
        f = feat_of(r)
        dist_rows.append(
            {
                "source": "phase1a_v0",
                "task": r.get("task"),
                "sample_index": r.get("sample_index"),
                "layer_idx": r.get("layer_idx"),
                **f,
                "joint_v1_would_trigger": int(
                    f["isolated_spike_ratio"] >= TAU_SPIKE
                    and f["topk_margin"] >= TAU_MARGIN
                    and f["neighbor_score_ratio"] <= TAU_SPIKE_NEI_MAX
                ),
            }
        )
    isos = [d["isolated_spike_ratio"] for d in dist_rows]
    margs = [d["topk_margin"] for d in dist_rows]
    neis = [d["neighbor_score_ratio"] for d in dist_rows]
    expected_rate = mean([d["joint_v1_would_trigger"] for d in dist_rows])
    write_csv(
        out_dir / "spike_distribution.csv",
        dist_rows,
        [
            "source", "task", "sample_index", "layer_idx",
            "isolated_spike_ratio", "topk_margin", "neighbor_score_ratio",
            "joint_v1_would_trigger",
        ],
    )
    spec = {
        "version": "acc_drm_v1",
        "frozen_v0_dir": str(v0_dir),
        "calibration_n_layer_records": len(dist_rows),
        "unlabeled_features": [
            "isolated_spike_ratio",
            "topk_margin",
            "neighbor_score_ratio",
        ],
        "forbidden": ["task_name", "gold", "prediction", "qa_score", "scorer"],
        "joint_condition": (
            "isolated_spike_ratio >= tau_spike AND "
            "topk_margin >= tau_margin AND "
            "neighbor_score_ratio <= tau_spike_nei_max"
        ),
        "distribution": {
            "isolated_spike_ratio": {
                "min": min(isos) if isos else None,
                "p50": quantile(isos, 0.50),
                "p90": quantile(isos, 0.90),
                "p95": quantile(isos, 0.95),
                "max": max(isos) if isos else None,
            },
            "topk_margin": {
                "min": min(margs) if margs else None,
                "p50": quantile(margs, 0.50),
                "p90": quantile(margs, 0.90),
                "max": max(margs) if margs else None,
            },
            "neighbor_score_ratio": {
                "min": min(neis) if neis else None,
                "p10": quantile(neis, 0.10),
                "p50": quantile(neis, 0.50),
                "p90": quantile(neis, 0.90),
                "max": max(neis) if neis else None,
            },
        },
        "frozen_thresholds": {
            "tau_spike": TAU_SPIKE,
            "tau_margin": TAU_MARGIN,
            "tau_spike_nei_max": TAU_SPIKE_NEI_MAX,
        },
        "tau_spike_rationale": (
            f"Chosen from unlabeled isolated_spike_ratio p95≈{quantile(isos, 0.95):.4f} "
            f"within the requested 90–95% quantile band (p90≈{quantile(isos, 0.90):.4f}). "
            "Frozen globally; never task-specific."
        ),
        "offline_expected_trigger_rate_on_v0_feats": round(expected_rate, 6),
        "target_trigger_band": [0.02, 0.08],
        "unchanged": [
            "multi_locus",
            "safe_sacrifice_pool",
            "cluster_budget_cap",
            "drm_score",
            "drm_swap_cap/rescue_ratio",
        ],
    }
    md = [
        "# Spike Calibration Spec (Phase-1B / acc_drm_v1)\n\n",
        f"- Frozen v0: `{v0_dir}` / `results/acc_drm_v0_audit`\n",
        f"- Calibration records: **{len(dist_rows)}** layer dumps from Phase-1A ACC\n",
        "- Joint unlabeled condition only (no task/gold/pred/scorer)\n\n",
        "## Frozen thresholds\n\n",
        f"- `tau_spike` = **{TAU_SPIKE}** (≈ p95 of isolated_spike_ratio)\n",
        f"- `tau_margin` = **{TAU_MARGIN}** (unchanged from v0)\n",
        f"- `tau_spike_nei_max` = **{TAU_SPIKE_NEI_MAX}** (cold-neighbor joint arm)\n\n",
        "## Offline expected trigger rate on v0 features\n\n",
        f"- **{expected_rate:.4f}** (target band 2%–8%)\n\n",
        "## Distribution summary\n\n",
        "```json\n",
        json.dumps(spec["distribution"], indent=2),
        "\n```\n",
        "\n## Explicit non-changes\n\n",
        "- Multi-Locus / Safe Sacrifice / Cluster Cap / DRM score / DRM swap count untouched.\n",
    ]
    (out_dir / "SPIKE_CALIBRATION_SPEC.md").write_text("".join(md), encoding="utf-8")
    (out_dir / "spike_calibration_spec.json").write_text(json.dumps(spec, indent=2) + "\n")
    return spec


def key_of(r: Mapping[str, Any]) -> Tuple[Any, Any, Any]:
    return (r.get("task"), r.get("sample_index"), r.get("layer_idx"))


def set_ids(r: Mapping[str, Any]) -> set:
    return {int(x) for x in (r.get("selected_block_ids") or [])}


def churn(a: set, b: set, k: int) -> float:
    return 1.0 - len(a & b) / max(1, k)


def band(name: str, value: Optional[float], lo: float, hi: float, *, allow_below_nonzero: bool = False) -> str:
    if value is None:
        return "MISSING"
    if value == 0.0 and not allow_below_nonzero:
        return "FAIL_zero"
    if lo <= value <= hi:
        return "PASS"
    if allow_below_nonzero and 0 < value < lo:
        return "WARN_low_nonzero"
    if value > hi:
        return "FAIL_high"
    return "FAIL_low"


def main() -> int:
    args = parse_args()
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Ensure v0 freeze pointer exists.
    freeze = REPO_ROOT / "results" / "acc_drm_v0_audit"
    freeze.mkdir(parents=True, exist_ok=True)
    (freeze / "SOURCE_DIR.txt").write_text(str(args.v0_dir.resolve()) + "\n")

    spec = write_spike_calibration_spec(out_dir, args.v0_dir)

    methods = list(MAIN_METHODS)
    if not args.skip_run:
        for method in methods:
            run_dir = out_dir / f"run_{method}"
            audit = run_dir / "selection_audit.jsonl"
            if audit.exists() and audit.stat().st_size > 0:
                continue
            cmd = qa_cmd(args, method, run_dir)
            (out_dir / f"command_{method}.json").write_text(json.dumps({"command": cmd}, indent=2) + "\n")
            env = os.environ.copy()
            env["PYTHONPATH"] = str(REPO_ROOT)
            env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
            log = out_dir / f"run_{method}.log"
            with log.open("w", encoding="utf-8") as lf:
                rc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
            if rc != 0:
                (out_dir / "FAILED.json").write_text(
                    json.dumps({"method": method, "returncode": rc, "log": str(log)}, indent=2) + "\n"
                )
                print(json.dumps({"failed": method, "returncode": rc, "log": str(log)}, indent=2))
                return 1

    audits = {m: load_audit(out_dir / f"run_{m}" / "selection_audit.jsonl") for m in methods}
    maps = {m: {key_of(r): r for r in rows} for m, rows in audits.items()}

    # active_set_audit.csv
    active_rows = []
    for method, rows in audits.items():
        for r in rows:
            f = feat_of(r)
            acc = r.get("acc") if isinstance(r.get("acc"), dict) else {}
            active_rows.append(
                {
                    "method": method,
                    "task": r.get("task"),
                    "sample_index": r.get("sample_index"),
                    "layer_idx": r.get("layer_idx"),
                    "num_selected": r.get("num_selected"),
                    "num_keep_target": r.get("num_keep_target"),
                    "churn_vs_silm": r.get("churn_vs_silm", r.get("churn_rate", "")),
                    "budget_parity": r.get("budget_parity", ""),
                    "over_budget": r.get("over_budget", 0),
                    "sink_violation": r.get("sink_violation", 0),
                    "spike_condition_triggered": r.get(
                        "spike_condition_triggered", acc.get("spike_condition_triggered", 0)
                    ),
                    "spike_protection_triggered": r.get(
                        "spike_protection_triggered", acc.get("spike_protection_triggered", 0)
                    ),
                    "spike_protection_changed_selection": r.get(
                        "spike_protection_changed_selection",
                        acc.get("spike_protection_changed_selection", 0),
                    ),
                    "multi_locus_triggered": r.get("multi_locus_triggered", acc.get("multi_locus_triggered", 0)),
                    "cluster_budget_cap_triggered": r.get(
                        "cluster_budget_cap_triggered", acc.get("cluster_budget_cap_triggered", 0)
                    ),
                    "drm_candidate_available": r.get("drm_candidate_available", ""),
                    "drm_candidate_eligible": r.get("drm_candidate_eligible", ""),
                    "drm_swap_accepted": r.get("drm_swap_accepted", r.get("drm_swaps_applied", "")),
                    "drm_swap_rejected": r.get("drm_swap_rejected", ""),
                    "rejection_reason": "|".join(r.get("rejection_reason") or [])
                    if isinstance(r.get("rejection_reason"), list)
                    else r.get("rejection_reason", ""),
                    "accepted_swaps_per_layer": r.get(
                        "accepted_swaps_per_layer", r.get("drm_swaps_applied", "")
                    ),
                    "high_confidence_spike_replacement_count": r.get(
                        "high_confidence_spike_replacement_count", 0
                    ),
                    "second_locus_replacement_count": r.get("second_locus_replacement_count", 0),
                    **f,
                }
            )
    write_csv(
        out_dir / "active_set_audit.csv",
        active_rows,
        list(active_rows[0].keys()) if active_rows else ["method"],
    )

    # spike_trigger_cases.csv
    spike_cases = []
    for method in ("acc_sliminfer_v1", "drm_acc_swap_sliminfer_v1"):
        for r in audits.get(method, []):
            cond = int(
                r.get("spike_condition_triggered")
                or (r.get("acc") or {}).get("spike_condition_triggered")
                or 0
            )
            if not cond:
                continue
            f = feat_of(r)
            spike_cases.append(
                {
                    "method": method,
                    "task": r.get("task"),
                    "sample_index": r.get("sample_index"),
                    "layer_idx": r.get("layer_idx"),
                    "spike_condition_triggered": 1,
                    "spike_protection_changed_selection": r.get(
                        "spike_protection_changed_selection",
                        (r.get("acc") or {}).get("spike_protection_changed_selection", 0),
                    ),
                    "fallback_policy": r.get("fallback_policy", (r.get("acc") or {}).get("fallback_policy", "")),
                    **f,
                }
            )
    write_csv(
        out_dir / "spike_trigger_cases.csv",
        spike_cases,
        [
            "method", "task", "sample_index", "layer_idx",
            "spike_condition_triggered", "spike_protection_changed_selection",
            "fallback_policy", "isolated_spike_ratio", "topk_margin", "neighbor_score_ratio",
        ],
    )

    # Cross-method churn
    def pair_churn(a: str, b: str) -> Dict[str, Any]:
        vals = []
        per: Dict[str, List[float]] = defaultdict(list)
        for k, ra in maps.get(a, {}).items():
            rb = maps.get(b, {}).get(k)
            if not rb:
                continue
            kk = int(ra.get("num_keep_target") or len(set_ids(ra)) or 1)
            c = churn(set_ids(ra), set_ids(rb), kk)
            vals.append(c)
            per[str(ra.get("task"))].append(c)
        return {
            "mean": round(mean(vals), 6),
            "n": len(vals),
            "per_task": {t: round(mean(vs), 6) for t, vs in sorted(per.items())},
        }

    churn_acc_cc = pair_churn("acc_sliminfer_v1", "cc_sliminfer_v1")
    churn_drm_cc = pair_churn("drm_acc_swap_sliminfer_v1", "cc_sliminfer_v1")
    churn_drm_acc = pair_churn("drm_acc_swap_sliminfer_v1", "acc_sliminfer_v1")

    def method_stats(method: str) -> Dict[str, Any]:
        rows = audits.get(method, [])
        n = max(1, len(rows))

        def rate(field: str, nested: str = "acc") -> float:
            s = 0
            for r in rows:
                v = r.get(field)
                if v is None and isinstance(r.get(nested), dict):
                    v = r[nested].get(field)
                s += int(v or 0)
            return s / n

        def sumf(field: str) -> int:
            return int(sum(int(r.get(field) or 0) for r in rows))

        per_task = {}
        for t in TASKS:
            tr = [r for r in rows if r.get("task") == t]
            tn = max(1, len(tr))
            per_task[t] = {
                "n": len(tr),
                "spike_condition_rate": sum(
                    int(r.get("spike_condition_triggered") or (r.get("acc") or {}).get("spike_condition_triggered") or 0)
                    for r in tr
                )
                / tn,
                "spike_changed_rate": sum(
                    int(
                        r.get("spike_protection_changed_selection")
                        or (r.get("acc") or {}).get("spike_protection_changed_selection")
                        or 0
                    )
                    for r in tr
                )
                / tn,
                "drm_swap_accepted_mean": mean(
                    [float(r.get("drm_swap_accepted") or r.get("accepted_swaps_per_layer") or 0) for r in tr]
                ),
            }

        return {
            "n_records": len(rows),
            "spike_condition_trigger_count": sumf("spike_condition_triggered")
            if method.startswith("acc") or "drm_acc" in method
            else sum(
                int(r.get("spike_condition_triggered") or (r.get("acc") or {}).get("spike_condition_triggered") or 0)
                for r in rows
            ),
            "spike_condition_trigger_rate": rate("spike_condition_triggered"),
            "spike_protection_changed_selection_count": sum(
                int(
                    r.get("spike_protection_changed_selection")
                    or (r.get("acc") or {}).get("spike_protection_changed_selection")
                    or 0
                )
                for r in rows
            ),
            "drm_candidate_available_rate": rate("drm_candidate_available"),
            "drm_candidate_eligible_mean": mean(
                [float(r.get("drm_candidate_eligible") or 0) for r in rows]
            ),
            "drm_swap_accepted_mean": mean(
                [float(r.get("drm_swap_accepted") or r.get("accepted_swaps_per_layer") or 0) for r in rows]
            ),
            "drm_swap_accepted_layers": sum(1 for r in rows if int(r.get("drm_swap_accepted") or 0) > 0),
            "drm_swap_rejected_mean": mean([float(r.get("drm_swap_rejected") or 0) for r in rows]),
            "over_budget_count": sumf("over_budget"),
            "forced_sink_violations": sumf("sink_violation"),
            "high_confidence_spike_replacement_count": sumf("high_confidence_spike_replacement_count"),
            "second_locus_replacement_count": sumf("second_locus_replacement_count"),
            "per_task": per_task,
        }

    summaries = {m: method_stats(m) for m in methods}
    summaries["churn"] = {
        "acc_vs_cc": churn_acc_cc,
        "drm_acc_vs_cc": churn_drm_cc,
        "drm_acc_vs_acc": churn_drm_acc,
    }
    (out_dir / "audit_summary.json").write_text(json.dumps(summaries, indent=2) + "\n")

    acc_s = summaries["acc_sliminfer_v1"]
    drm_s = summaries["drm_acc_swap_sliminfer_v1"]
    spike_rate = float(acc_s.get("spike_condition_trigger_rate") or 0.0)
    decision = {
        "version": "acc_drm_v1",
        "phase": "1B",
        "frozen_thresholds": spec["frozen_thresholds"],
        "spike_trigger_rate": round(spike_rate, 6),
        "spike_condition_trigger_count": acc_s.get("spike_condition_trigger_count"),
        "spike_protection_changed_selection_count": acc_s.get(
            "spike_protection_changed_selection_count"
        ),
        "acc_vs_cc_churn": churn_acc_cc["mean"],
        "drm_acc_vs_cc_churn": churn_drm_cc["mean"],
        "drm_acc_vs_acc_churn": churn_drm_acc["mean"],
        "drm_swap_accepted_mean": drm_s.get("drm_swap_accepted_mean"),
        "drm_swap_accepted_layers": drm_s.get("drm_swap_accepted_layers"),
        "over_budget_acc": acc_s.get("over_budget_count"),
        "over_budget_drm": drm_s.get("over_budget_count"),
        "sink_violations_acc": acc_s.get("forced_sink_violations"),
        "sink_violations_drm": drm_s.get("forced_sink_violations"),
        "high_confidence_spike_replacement_count": int(
            (acc_s.get("high_confidence_spike_replacement_count") or 0)
            + (drm_s.get("high_confidence_spike_replacement_count") or 0)
        ),
        "second_locus_replacement_count": int(
            (acc_s.get("second_locus_replacement_count") or 0)
            + (drm_s.get("second_locus_replacement_count") or 0)
        ),
        "bands": {
            "spike_2_8": band("spike", spike_rate, 0.02, 0.08),
            "acc_vs_cc_3_12": band("acc_cc", churn_acc_cc["mean"], 0.03, 0.12),
            "drm_vs_cc_3_12": band("drm_cc", churn_drm_cc["mean"], 0.03, 0.12),
            "drm_vs_acc_nonzero": band(
                "drm_acc", churn_drm_acc["mean"], 0.0, 1.0, allow_below_nonzero=True
            )
            if (churn_drm_acc["mean"] or 0) > 0
            else "FAIL_zero",
        },
        "run_qa": False,
    }
    hard_ok = (
        decision["bands"]["spike_2_8"] == "PASS"
        and decision["bands"]["acc_vs_cc_3_12"] == "PASS"
        and decision["bands"]["drm_vs_cc_3_12"] == "PASS"
        and decision["bands"]["drm_vs_acc_nonzero"] != "FAIL_zero"
        and decision["over_budget_acc"] == 0
        and decision["over_budget_drm"] == 0
        and decision["sink_violations_acc"] == 0
        and decision["sink_violations_drm"] == 0
        and decision["high_confidence_spike_replacement_count"] == 0
        and decision["second_locus_replacement_count"] == 0
    )
    decision["phase1b_go"] = bool(hard_ok)
    decision["phase2_go"] = bool(hard_ok)
    decision["reason"] = (
        "Phase-1B GO — proceed to Phase-2 live prefill wiring audit (no QA)."
        if hard_ok
        else "Phase-1B FAIL — inspect bands / safety counters before live."
    )
    (out_dir / "PHASE1B_DECISION.json").write_text(json.dumps(decision, indent=2) + "\n")

    md = [
        "# Phase-1B Active-Set Audit (Spike Calibration)\n\n",
        f"- out: `{out_dir}`\n",
        f"- frozen v0: `{args.v0_dir}`\n",
        f"- thresholds: tau_spike={TAU_SPIKE}, tau_margin={TAU_MARGIN}, tau_spike_nei_max={TAU_SPIKE_NEI_MAX}\n\n",
        "## GO metrics\n\n",
        f"| Metric | Value | Band |\n|--------|------:|------|\n",
        f"| spike trigger rate | {spike_rate:.4f} | {decision['bands']['spike_2_8']} |\n",
        f"| spike condition count | {decision['spike_condition_trigger_count']} | — |\n",
        f"| spike changed selection | {decision['spike_protection_changed_selection_count']} | — |\n",
        f"| ACC vs CC churn | {churn_acc_cc['mean']:.4f} | {decision['bands']['acc_vs_cc_3_12']} |\n",
        f"| DRM-ACC vs CC churn | {churn_drm_cc['mean']:.4f} | {decision['bands']['drm_vs_cc_3_12']} |\n",
        f"| DRM-ACC vs ACC churn | {churn_drm_acc['mean']:.4f} | {decision['bands']['drm_vs_acc_nonzero']} |\n",
        f"| DRM accepted swaps/layer | {drm_s.get('drm_swap_accepted_mean')} | (no strength bump) |\n",
        f"| DRM accepted layers | {drm_s.get('drm_swap_accepted_layers')} | — |\n",
        f"| over-budget | acc={decision['over_budget_acc']} drm={decision['over_budget_drm']} | must 0 |\n",
        f"| sink violations | acc={decision['sink_violations_acc']} drm={decision['sink_violations_drm']} | must 0 |\n",
        f"| high-conf spike replacement | {decision['high_confidence_spike_replacement_count']} | must 0 |\n",
        f"| second-locus replacement | {decision['second_locus_replacement_count']} | must 0 |\n\n",
        f"**Decision: `phase1b_go={decision['phase1b_go']}`** — {decision['reason']}\n\n",
        "## Per-task spike rates (info only; no task-specific threshold)\n\n",
        "```json\n",
        json.dumps(acc_s.get("per_task", {}), indent=2),
        "\n```\n",
        "\n## DRM semantics (drm_acc_swap_v1)\n\n",
        "```json\n",
        json.dumps(
            {
                k: drm_s.get(k)
                for k in (
                    "drm_candidate_available_rate",
                    "drm_candidate_eligible_mean",
                    "drm_swap_accepted_mean",
                    "drm_swap_accepted_layers",
                    "drm_swap_rejected_mean",
                )
            },
            indent=2,
        ),
        "\n```\n",
    ]
    (out_dir / "PHASE1B_ACTIVE_SET_AUDIT.md").write_text("".join(md), encoding="utf-8")

    print(json.dumps({"out_dir": str(out_dir), "decision": decision}, indent=2))
    return 0 if hard_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
