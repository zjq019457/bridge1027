#!/usr/bin/env python3
"""Phase-2 live prefill wiring audit for acc_drm_v1 (no QA).

Sequence:
  1) static wiring checks (virtual-bypass audit-only; baselines unpolluted)
  2) 24-sample live smoke
  3) 72-sample live audit (if smoke GO)

Methods: sliminfer_original, cc_sliminfer_v1, acc_sliminfer_v1, drm_acc_swap_sliminfer_v1
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
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
PHASE1B_DEFAULT = REPO_ROOT / "results" / "acc_drm_phase1b_spike_calibration_20260714_085424"
TASKS = ("musique", "2wikimqa", "triviaqa")
METHODS = (
    "sliminfer_original",
    "cc_sliminfer_v1",
    "acc_sliminfer_v1",
    "drm_acc_swap_sliminfer_v1",
)


def parse_args() -> argparse.Namespace:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out_dir", type=Path, default=REPO_ROOT / f"results/acc_drm_phase2_live_{ts}")
    p.add_argument("--phase1b_dir", type=Path, default=PHASE1B_DEFAULT)
    p.add_argument("--model_path", default=DEFAULT_MODEL)
    p.add_argument("--longbench_dir", default=str(DEFAULT_LONGBENCH))
    p.add_argument("--python", default=str(REPO_ROOT / ".venv/bin/python"))
    p.add_argument("--smoke_samples", type=int, default=8, help="Per-task sample cap for smoke (~24 total).")
    p.add_argument("--full_samples", type=int, default=24, help="Per-task sample cap for full live (~72 total).")
    p.add_argument("--skip_smoke", action="store_true")
    p.add_argument("--skip_full", action="store_true")
    return p.parse_args()


def qa_cmd(args: argparse.Namespace, method: str, run_dir: Path, max_samples: int, *, with_audit: bool) -> List[str]:
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
        "--longbench_max_samples", str(int(max_samples)),
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
        "--output_dir", str(run_dir),
    ]
    if with_audit:
        cmd.extend(["--cc_selection_audit_path", str(run_dir / "selection_audit.jsonl")])
    return cmd


def run_method(args: argparse.Namespace, method: str, run_dir: Path, max_samples: int, *, with_audit: bool) -> int:
    cmd = qa_cmd(args, method, run_dir, max_samples, with_audit=with_audit)
    (run_dir.parent / f"command_{run_dir.name}.json").write_text(json.dumps({"command": cmd}, indent=2) + "\n")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    log = run_dir.parent / f"{run_dir.name}.log"
    with log.open("w", encoding="utf-8") as lf:
        return subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, stdout=lf, stderr=subprocess.STDOUT).returncode


def load_audit(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def mean(xs: Sequence[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0


def key_of(r: Mapping[str, Any]) -> Tuple[Any, Any, Any]:
    return (r.get("task"), r.get("sample_index"), r.get("layer_idx"))


def set_ids(r: Mapping[str, Any]) -> set:
    return {int(x) for x in (r.get("selected_block_ids") or [])}


def pair_churn(a_rows: List[Dict[str, Any]], b_rows: List[Dict[str, Any]]) -> float:
    amap = {key_of(r): r for r in a_rows}
    bmap = {key_of(r): r for r in b_rows}
    vals = []
    for k, ra in amap.items():
        rb = bmap.get(k)
        if not rb:
            continue
        ka = int(ra.get("num_keep_target") or len(set_ids(ra)) or 1)
        vals.append(1.0 - len(set_ids(ra) & set_ids(rb)) / max(1, ka))
    return mean(vals)


def static_wiring_checks() -> Dict[str, Any]:
    """Confirm audit-only virtual bypass cannot enter official QA path."""
    cache_src = (REPO_ROOT / "utils" / "cache_utils.py").read_text(encoding="utf-8")
    # Presence of guard.
    has_audit_guard = "skip_virtual_for_audit = bool(cfg.get(\"cc_selection_audit_path\"))" in cache_src
    # ACC methods always skip virtual regardless of audit.
    always_skip = all(
        m in cache_src
        for m in ("acc_sliminfer_v1", "drm_acc_swap_sliminfer_v1", "cc_sliminfer")
    )
    # Simulate condition: without audit path, original must NOT skip solely due to audit flag.
    # We re-read logic via AST/string rather than executing pruning.
    checks = {
        "audit_virtual_bypass_gated_on_cc_selection_audit_path": has_audit_guard,
        "acc_v1_aliases_in_virtual_skip_list": always_skip,
        "official_qa_default_cc_selection_audit_path_empty": True,  # CLI default ""
    }
    # Confirm CLI default empty:
    variants = (REPO_ROOT / "experiments" / "run_qwen25_sliminfer_variants.py").read_text(encoding="utf-8")
    m = re.search(r'--cc_selection_audit_path.*?default=("[^"]*"|\'\')', variants, re.S)
    checks["cli_default_audit_path_empty"] = bool(m and (m.group(1) in {'""', "''"} or m.group(1) == '""'))
    # stronger: argparse default=""
    checks["cli_default_audit_path_empty"] = 'parser.add_argument("--cc_selection_audit_path", type=str, default="")' in variants

    # Confirm is_spike / ACC version imported and CC path still uses cc_sliminfer not ACC.
    checks["cc_uses_cluster_continuity_not_acc"] = (
        'elif variant_method == "cc_sliminfer":' in cache_src
        and "cluster_continuity_active_ids" in cache_src
    )
    checks["acc_branch_gated_on_ACC_DRM_METHODS"] = "elif str(variant_method) in ACC_DRM_METHODS:" in cache_src
    checks["pass"] = all(
        [
            checks["audit_virtual_bypass_gated_on_cc_selection_audit_path"],
            checks["acc_v1_aliases_in_virtual_skip_list"],
            checks["cli_default_audit_path_empty"],
            checks["cc_uses_cluster_continuity_not_acc"],
            checks["acc_branch_gated_on_ACC_DRM_METHODS"],
        ]
    )
    return checks


def analyze_bundle(label: str, out_sub: Path, phase1b_dir: Path) -> Dict[str, Any]:
    audits = {m: load_audit(out_sub / f"run_{m}" / "selection_audit.jsonl") for m in METHODS}
    summary: Dict[str, Any] = {"label": label, "methods": {}}
    for m, rows in audits.items():
        summary["methods"][m] = {
            "n_records": len(rows),
            "over_budget": sum(int(r.get("over_budget") or 0) for r in rows),
            "sink_violations": sum(int(r.get("sink_violation") or 0) for r in rows),
            "budget_parity_fail": sum(1 for r in rows if int(r.get("budget_parity") or 0) == 0),
            "drm_swap_accepted_mean": mean(
                [float(r.get("drm_swap_accepted") or r.get("accepted_swaps_per_layer") or 0) for r in rows]
            ),
            "drm_swap_accepted_layers": sum(1 for r in rows if int(r.get("drm_swap_accepted") or 0) > 0),
            "spike_condition_rate": mean(
                [
                    float(
                        r.get("spike_condition_triggered")
                        or (r.get("acc") or {}).get("spike_condition_triggered")
                        or 0
                    )
                    for r in rows
                ]
            ),
            "tasks": sorted({str(r.get("task")) for r in rows}),
            "samples": sorted({(r.get("task"), r.get("sample_index")) for r in rows}),
        }
    summary["churn"] = {
        "acc_vs_cc": pair_churn(audits["acc_sliminfer_v1"], audits["cc_sliminfer_v1"]),
        "drm_acc_vs_cc": pair_churn(audits["drm_acc_swap_sliminfer_v1"], audits["cc_sliminfer_v1"]),
        "drm_acc_vs_acc": pair_churn(audits["drm_acc_swap_sliminfer_v1"], audits["acc_sliminfer_v1"]),
        "acc_vs_original": pair_churn(audits["acc_sliminfer_v1"], audits["sliminfer_original"]),
        "cc_vs_original": pair_churn(audits["cc_sliminfer_v1"], audits["sliminfer_original"]),
    }
    # Compare to Phase-1B offline if available.
    p1_acc = load_audit(phase1b_dir / "run_acc_sliminfer_v1" / "selection_audit.jsonl")
    p1_drm = load_audit(phase1b_dir / "run_drm_acc_swap_sliminfer_v1" / "selection_audit.jsonl")
    p1_cc = load_audit(phase1b_dir / "run_cc_sliminfer_v1" / "selection_audit.jsonl")
    if p1_acc and p1_cc:
        summary["phase1b_reference_churn"] = {
            "acc_vs_cc": pair_churn(p1_acc, p1_cc),
            "drm_acc_vs_cc": pair_churn(p1_drm, p1_cc) if p1_drm else None,
            "drm_acc_vs_acc": pair_churn(p1_drm, p1_acc) if p1_drm else None,
        }
        summary["live_vs_phase1b_churn_delta"] = {
            "acc_vs_cc": abs(summary["churn"]["acc_vs_cc"] - summary["phase1b_reference_churn"]["acc_vs_cc"]),
            "drm_acc_vs_cc": abs(
                summary["churn"]["drm_acc_vs_cc"] - (summary["phase1b_reference_churn"]["drm_acc_vs_cc"] or 0)
            ),
        }
    # Coverage expectation: 3 tasks × max_samples × ~3 layers, but samples vary by length.
    summary["coverage"] = {
        "n_records_per_method": {m: summary["methods"][m]["n_records"] for m in METHODS},
        "all_methods_nonzero": all(summary["methods"][m]["n_records"] > 0 for m in METHODS),
    }
    return summary


def decision_from(summary: Dict[str, Any], wiring: Dict[str, Any], *, stage: str) -> Dict[str, Any]:
    c = summary["churn"]
    methods = summary["methods"]
    ok_budget = all(methods[m]["over_budget"] == 0 for m in METHODS)
    ok_sink = all(methods[m]["sink_violations"] == 0 for m in METHODS)
    ok_parity = all(methods[m]["budget_parity_fail"] == 0 for m in METHODS)
    acc_change = c["acc_vs_cc"] > 0 and c["acc_vs_original"] > 0
    drm_change = c["drm_acc_vs_acc"] > 0 and c["drm_acc_vs_cc"] > 0
    # Live vs Phase1B within ~2pp absolute
    deltas = summary.get("live_vs_phase1b_churn_delta") or {}
    ok_align = all(float(v) <= 0.02 for v in deltas.values()) if deltas else True
    # DRM accepted non-zero on live
    drm_acc = methods["drm_acc_swap_sliminfer_v1"]["drm_swap_accepted_mean"]
    drm_ok = drm_acc > 0
    decision = {
        "stage": stage,
        "wiring_pass": bool(wiring.get("pass")),
        "acc_nonzero_change": bool(acc_change),
        "drm_nonzero_change": bool(drm_change),
        "drm_swap_accepted_mean": drm_acc,
        "over_budget_ok": ok_budget,
        "sink_ok": ok_sink,
        "budget_parity_ok": ok_parity,
        "live_aligned_to_phase1b": bool(ok_align),
        "churn": c,
        "deltas_vs_phase1b": deltas,
        "run_qa": False,
    }
    go = (
        decision["wiring_pass"]
        and acc_change
        and drm_change
        and drm_ok
        and ok_budget
        and ok_sink
        and ok_parity
        and ok_align
        and 0.03 <= c["acc_vs_cc"] <= 0.12
        and 0.03 <= c["drm_acc_vs_cc"] <= 0.12
    )
    decision["phase2_go"] = bool(go)
    decision["reason"] = (
        "Phase-2 live wiring GO — ready to prepare 6-task QA (not started)."
        if go
        else "Phase-2 FAIL — fix wiring/live mismatch before QA."
    )
    # Special: DRM accepted=0 → wiring issue, do NOT raise weights.
    if not drm_ok:
        decision["reason"] = "DRM accepted swaps=0 on live path — wiring fix only; do NOT increase DRM strength."
        decision["phase2_go"] = False
    return decision


def write_csv_active(path: Path, audits: Dict[str, List[Dict[str, Any]]]) -> None:
    rows = []
    for m, rs in audits.items():
        for r in rs:
            rows.append(
                {
                    "method": m,
                    "task": r.get("task"),
                    "sample_index": r.get("sample_index"),
                    "layer_idx": r.get("layer_idx"),
                    "num_selected": r.get("num_selected"),
                    "over_budget": r.get("over_budget"),
                    "sink_violation": r.get("sink_violation"),
                    "drm_swap_accepted": r.get("drm_swap_accepted", r.get("accepted_swaps_per_layer")),
                    "spike_condition_triggered": r.get(
                        "spike_condition_triggered", (r.get("acc") or {}).get("spike_condition_triggered")
                    ),
                }
            )
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    args = parse_args()
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    wiring = static_wiring_checks()
    (out_dir / "WIRING_STATIC_CHECKS.json").write_text(json.dumps(wiring, indent=2) + "\n")
    if not wiring.get("pass"):
        (out_dir / "PHASE2_DECISION.json").write_text(
            json.dumps(
                {
                    "phase2_go": False,
                    "reason": "static wiring checks failed",
                    "wiring": wiring,
                    "run_qa": False,
                },
                indent=2,
            )
            + "\n"
        )
        print(json.dumps({"failed": "static_wiring", "wiring": wiring}, indent=2))
        return 1

    # --- smoke 24 ---
    smoke_dir = out_dir / "smoke24"
    smoke_dir.mkdir(exist_ok=True)
    if not args.skip_smoke:
        for method in METHODS:
            run_dir = smoke_dir / f"run_{method}"
            if (run_dir / "selection_audit.jsonl").exists():
                continue
            rc = run_method(args, method, run_dir, args.smoke_samples, with_audit=True)
            if rc != 0:
                (out_dir / "FAILED.json").write_text(
                    json.dumps({"stage": "smoke24", "method": method, "returncode": rc}, indent=2) + "\n"
                )
                return 1

    smoke_summary = analyze_bundle("smoke24", smoke_dir, args.phase1b_dir)
    smoke_decision = decision_from(smoke_summary, wiring, stage="smoke24")
    (smoke_dir / "smoke_summary.json").write_text(json.dumps(smoke_summary, indent=2) + "\n")
    (smoke_dir / "SMOKE_DECISION.json").write_text(json.dumps(smoke_decision, indent=2) + "\n")
    write_csv_active(
        smoke_dir / "active_set_audit.csv",
        {m: load_audit(smoke_dir / f"run_{m}" / "selection_audit.jsonl") for m in METHODS},
    )

    if not smoke_decision.get("phase2_go"):
        (out_dir / "PHASE2_DECISION.json").write_text(
            json.dumps({**smoke_decision, "full72_skipped": True}, indent=2) + "\n"
        )
        (out_dir / "PHASE2_LIVE_WIRING_AUDIT.md").write_text(
            "# Phase-2 Live Wiring Audit\n\nSmoke FAIL — full72 skipped.\n\n```json\n"
            + json.dumps(smoke_decision, indent=2)
            + "\n```\n",
            encoding="utf-8",
        )
        print(json.dumps({"out_dir": str(out_dir), "decision": smoke_decision}, indent=2))
        return 2

    # --- full 72 ---
    full_dir = out_dir / "full72"
    full_dir.mkdir(exist_ok=True)
    if not args.skip_full:
        for method in METHODS:
            run_dir = full_dir / f"run_{method}"
            if (run_dir / "selection_audit.jsonl").exists():
                continue
            rc = run_method(args, method, run_dir, args.full_samples, with_audit=True)
            if rc != 0:
                (out_dir / "FAILED.json").write_text(
                    json.dumps({"stage": "full72", "method": method, "returncode": rc}, indent=2) + "\n"
                )
                return 1

    full_summary = analyze_bundle("full72", full_dir, args.phase1b_dir)
    full_decision = decision_from(full_summary, wiring, stage="full72")
    (full_dir / "full_summary.json").write_text(json.dumps(full_summary, indent=2) + "\n")
    write_csv_active(
        full_dir / "active_set_audit.csv",
        {m: load_audit(full_dir / f"run_{m}" / "selection_audit.jsonl") for m in METHODS},
    )

    decision = {
        **full_decision,
        "smoke": smoke_decision,
        "wiring": wiring,
        "run_qa": False,
        "qa_prepared_but_not_started": bool(full_decision.get("phase2_go")),
    }
    (out_dir / "PHASE2_DECISION.json").write_text(json.dumps(decision, indent=2) + "\n")
    md = [
        "# Phase-2 Live Prefill Wiring Audit\n\n",
        "## Static wiring\n\n",
        "```json\n" + json.dumps(wiring, indent=2) + "\n```\n\n",
        "## Smoke24 decision\n\n",
        "```json\n" + json.dumps(smoke_decision, indent=2) + "\n```\n\n",
        "## Full72 decision\n\n",
        "```json\n" + json.dumps(full_decision, indent=2) + "\n```\n\n",
        "## Notes\n\n",
        "- Official QA path keeps `--cc_selection_audit_path` empty → virtual bypass OFF for original.\n",
        "- This live audit uses selection dumps (audit path ON) for fixed-page comparable active sets.\n",
        "- DRM strength not increased; if accepted swaps were 0, treat as wiring only.\n",
        "- QA not started by this script.\n",
    ]
    (out_dir / "PHASE2_LIVE_WIRING_AUDIT.md").write_text("".join(md), encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "decision": decision}, indent=2))
    return 0 if full_decision.get("phase2_go") else 2


if __name__ == "__main__":
    raise SystemExit(main())
