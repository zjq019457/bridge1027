#!/usr/bin/env python3
"""Phase-1 ACC/DRM active-set audit (prefill only, no QA).

Runs gate_audit for:
  sliminfer_original, cc_sliminfer_v1, acc_sliminfer,
  drm_swap_sliminfer, drm_acc_swap_sliminfer
optional: drm_acc_add_sliminfer

Writes required audit artifacts under results/acc_drm_audit_<ts>/.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_MODEL = (
    "/home/yu-bai/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/"
    "snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
)
DEFAULT_LONGBENCH = Path("/home/yu-bai/Desktop/jiaqizhang/HeadKV/data/LongBench")
TASKS = ("musique", "2wikimqa", "triviaqa")
MAIN_METHODS = (
    "sliminfer_original",
    "cc_sliminfer_v1",
    "acc_sliminfer",
    "drm_swap_sliminfer",
    "drm_acc_swap_sliminfer",
)
OPTIONAL_METHODS = ("drm_acc_add_sliminfer",)


def parse_args() -> argparse.Namespace:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out_dir", type=Path, default=REPO_ROOT / f"results/acc_drm_audit_{ts}")
    p.add_argument("--model_path", default=DEFAULT_MODEL)
    p.add_argument("--longbench_dir", default=str(DEFAULT_LONGBENCH))
    p.add_argument("--python", default=str(REPO_ROOT / ".venv/bin/python"))
    p.add_argument("--max_samples", type=int, default=32, help="Per-task sample cap for audit.")
    p.add_argument("--include_add", action="store_true", help="Also audit drm_acc_add_sliminfer (K+M).")
    p.add_argument("--skip_run", action="store_true")
    return p.parse_args()


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


def mean(xs: Sequence[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0


def write_csv(path: Path, rows: List[Dict[str, Any]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def aggregate(method: str, rows: List[Dict[str, Any]], cc_map: Dict[Tuple[Any, ...], Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {"method": method, "n_records": 0}
    churn_silm = [float(r.get("churn_vs_silm", r.get("churn_rate", 0)) or 0) for r in rows]
    jacc_silm = [float(r.get("jaccard_vs_silm", 0) or 0) for r in rows]
    churn_cc = []
    jacc_cc = []
    for r in rows:
        key = (r.get("task"), r.get("sample_index"), r.get("layer_idx"))
        cref = cc_map.get(key)
        if cref and r.get("selected_block_ids") is not None and cref.get("selected_block_ids") is not None:
            a = set(int(x) for x in r["selected_block_ids"])
            b = set(int(x) for x in cref["selected_block_ids"])
            k = max(1, int(r.get("num_keep_target") or len(a) or 1))
            churn_cc.append(1.0 - len(a & b) / k)
            jacc_cc.append(len(a & b) / max(1, len(a | b)))
        elif r.get("churn_vs_cc") is not None:
            churn_cc.append(float(r["churn_vs_cc"]))
        elif r.get("jaccard_vs_cc") is not None:
            jacc_cc.append(float(r["jaccard_vs_cc"]))

    def rate(key: str) -> float:
        vals = [int(bool(r.get(key))) for r in rows if key in r]
        return mean(vals) if vals else 0.0

    swaps = [float(r.get("drm_swaps_applied") or r.get("swaps_applied") or 0) for r in rows]
    over = sum(int(r.get("over_budget") or 0) for r in rows)
    sink_v = sum(int(r.get("sink_violation") or 0) for r in rows)
    per_task = {}
    for t in TASKS:
        tr = [r for r in rows if r.get("task") == t]
        if not tr:
            continue
        per_task[t] = {
            "n": len(tr),
            "multi_locus_rate": mean([float(r.get("multi_locus_triggered") or 0) for r in tr]),
            "spike_rate": mean([float(r.get("spike_protection_triggered") or 0) for r in tr]),
            "drm_rate": mean([float(r.get("drm_rescue_triggered") or 0) for r in tr]),
            "churn_vs_silm": mean([float(r.get("churn_vs_silm") or r.get("churn_rate") or 0) for r in tr]),
        }

    return {
        "method": method,
        "n_records": len(rows),
        "churn_vs_original_mean": round(mean(churn_silm), 6),
        "churn_vs_cc_mean": round(mean(churn_cc), 6) if churn_cc else "",
        "jaccard_vs_original_mean": round(mean(jacc_silm), 6),
        "jaccard_vs_cc_mean": round(mean(jacc_cc), 6) if jacc_cc else "",
        "multi_locus_trigger_rate": round(rate("multi_locus_triggered"), 6),
        "spike_protection_trigger_rate": round(rate("spike_protection_triggered"), 6),
        "cluster_budget_cap_trigger_rate": round(rate("cluster_budget_cap_triggered"), 6),
        "drm_rescue_trigger_rate": round(rate("drm_rescue_triggered"), 6),
        "avg_swaps_per_layer": round(mean(swaps), 6),
        "over_budget_count": over,
        "forced_sink_violations": sink_v,
        "budget_parity_rate": round(mean([float(r.get("budget_parity") or 0) for r in rows]), 6),
        "per_task": per_task,
    }


def band_check(name: str, churn: Optional[float], lo: float = 0.03, hi: float = 0.12) -> str:
    if churn is None or churn == "":
        return "N/A"
    c = float(churn)
    if c <= 1e-12:
        return "FAIL_zero_churn"
    if c > 0.20:
        return "FAIL_too_aggressive"
    if lo <= c <= hi:
        return "PASS_band"
    if c < lo:
        return "WARN_low_churn"
    return "WARN_high_churn"


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(REPO_ROOT / "ACC_DRM_IDEA.md", out_dir / "ACC_DRM_IDEA.md")

    methods = list(MAIN_METHODS)
    if args.include_add:
        methods.extend(OPTIONAL_METHODS)

    # Live runs.
    for method in methods:
        run_dir = out_dir / f"run_{method}"
        audit = run_dir / "selection_audit.jsonl"
        if args.skip_run and audit.exists():
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
            (out_dir / "FAILED.json").write_text(json.dumps({"method": method, "returncode": rc}, indent=2) + "\n")
            print(json.dumps({"failed": method, "returncode": rc, "log": str(log)}, indent=2))
            return 1

    audits: Dict[str, List[Dict[str, Any]]] = {}
    for method in methods:
        audits[method] = load_audit(out_dir / f"run_{method}" / "selection_audit.jsonl")

    cc_map = {
        (r.get("task"), r.get("sample_index"), r.get("layer_idx")): r
        for r in audits.get("cc_sliminfer_v1", [])
    }

    # active_set_audit.csv — one row per layer record.
    active_rows = []
    for method, rows in audits.items():
        for r in rows:
            key = (r.get("task"), r.get("sample_index"), r.get("layer_idx"))
            cref = cc_map.get(key)
            churn_cc = r.get("churn_vs_cc", "")
            jacc_cc = r.get("jaccard_vs_cc", "")
            if cref and r.get("selected_block_ids") is not None and cref.get("selected_block_ids") is not None:
                a = set(int(x) for x in r["selected_block_ids"])
                b = set(int(x) for x in cref["selected_block_ids"])
                k = max(1, int(r.get("num_keep_target") or len(a) or 1))
                churn_cc = round(1.0 - len(a & b) / k, 6)
                jacc_cc = round(len(a & b) / max(1, len(a | b)), 6)
            topo = r.get("cluster_topology") if isinstance(r.get("cluster_topology"), dict) else {}
            acc = r.get("acc") if isinstance(r.get("acc"), dict) else {}
            active_rows.append({
                "method": method,
                "task": r.get("task"),
                "sample_id": r.get("sample_id"),
                "sample_index": r.get("sample_index"),
                "layer_idx": r.get("layer_idx"),
                "num_selected": r.get("num_selected"),
                "num_keep_target": r.get("num_keep_target"),
                "churn_vs_original": r.get("churn_vs_silm", r.get("churn_rate", "")),
                "churn_vs_cc": churn_cc,
                "jaccard_vs_original": r.get("jaccard_vs_silm", ""),
                "jaccard_vs_cc": jacc_cc,
                "multi_locus_triggered": r.get("multi_locus_triggered", acc.get("multi_locus_triggered", 0)),
                "spike_protection_triggered": r.get("spike_protection_triggered", acc.get("spike_protection_triggered", 0)),
                "cluster_budget_cap_triggered": r.get("cluster_budget_cap_triggered", acc.get("cluster_budget_cap_triggered", 0)),
                "drm_rescue_triggered": r.get("drm_rescue_triggered", 0),
                "drm_swaps_applied": r.get("drm_swaps_applied", r.get("swaps_applied", 0)),
                "over_budget": r.get("over_budget", 0),
                "sink_violation": r.get("sink_violation", 0),
                "budget_parity": r.get("budget_parity", ""),
                "fallback_policy": r.get("fallback_policy", r.get("policy_chosen", "")),
                "second_over_first": topo.get("second_over_first", ""),
                "inter_cluster_gap_ratio": topo.get("inter_cluster_gap_ratio", ""),
                "n_selected_preview": ",".join(str(x) for x in (r.get("selected_block_ids") or [])[:12]),
            })
    write_csv(
        out_dir / "active_set_audit.csv",
        active_rows,
        [
            "method", "task", "sample_id", "sample_index", "layer_idx",
            "num_selected", "num_keep_target",
            "churn_vs_original", "churn_vs_cc",
            "jaccard_vs_original", "jaccard_vs_cc",
            "multi_locus_triggered", "spike_protection_triggered",
            "cluster_budget_cap_triggered", "drm_rescue_triggered",
            "drm_swaps_applied", "over_budget", "sink_violation", "budget_parity",
            "fallback_policy", "second_over_first", "inter_cluster_gap_ratio",
            "n_selected_preview",
        ],
    )

    # cluster_topology_stats.csv
    topo_rows = []
    for method in ("acc_sliminfer", "drm_acc_swap_sliminfer", "cc_sliminfer_v1"):
        for r in audits.get(method, []):
            topo = r.get("cluster_topology") if isinstance(r.get("cluster_topology"), dict) else {}
            feats = r.get("features") if isinstance(r.get("features"), dict) else {}
            if not topo and not feats:
                continue
            topo_rows.append({
                "method": method,
                "task": r.get("task"),
                "sample_index": r.get("sample_index"),
                "layer_idx": r.get("layer_idx"),
                "n_clusters": topo.get("n_clusters", feats.get("n_clusters", "")),
                "first_cluster_mass": topo.get("first_cluster_mass", ""),
                "second_cluster_mass": topo.get("second_cluster_mass", ""),
                "second_over_first": topo.get("second_over_first", feats.get("second_cluster_mass_ratio", "")),
                "inter_cluster_gap_ratio": topo.get("inter_cluster_gap_ratio", feats.get("cluster_gap_ratio", "")),
                "cluster_length": topo.get("cluster_length_max", feats.get("cluster_length", "")),
                "isolated_spike_ratio": feats.get("isolated_spike_ratio", ""),
                "topk_margin": feats.get("topk_margin", ""),
                "multi_locus_triggered": r.get("multi_locus_triggered", 0),
                "spike_protection_triggered": r.get("spike_protection_triggered", 0),
            })
    write_csv(
        out_dir / "cluster_topology_stats.csv",
        topo_rows,
        [
            "method", "task", "sample_index", "layer_idx", "n_clusters",
            "first_cluster_mass", "second_cluster_mass", "second_over_first",
            "inter_cluster_gap_ratio", "cluster_length", "isolated_spike_ratio",
            "topk_margin", "multi_locus_triggered", "spike_protection_triggered",
        ],
    )

    # drm_rescue_candidates.csv + sacrifice_pool_stats.csv
    drm_rows = []
    sac_rows = []
    for method in ("drm_swap_sliminfer", "drm_acc_swap_sliminfer", "drm_acc_add_sliminfer"):
        for r in audits.get(method, []):
            for bid in (r.get("drm_candidates") or r.get("rescued_ids") or [])[:32]:
                drm_rows.append({
                    "method": method,
                    "task": r.get("task"),
                    "sample_index": r.get("sample_index"),
                    "layer_idx": r.get("layer_idx"),
                    "candidate_or_rescued_id": bid,
                    "drm_swaps_applied": r.get("drm_swaps_applied", 0),
                    "drm_rescue_triggered": r.get("drm_rescue_triggered", 0),
                })
            sac = r.get("safe_sacrifice") if isinstance(r.get("safe_sacrifice"), dict) else {}
            if sac or r.get("sacrifice_used"):
                sac_rows.append({
                    "method": method,
                    "task": r.get("task"),
                    "sample_index": r.get("sample_index"),
                    "layer_idx": r.get("layer_idx"),
                    "sacrifice_pool_size": sac.get("sacrifice_pool_size", ""),
                    "protected_size": sac.get("protected_size", ""),
                    "protected_centers": sac.get("protected_centers", ""),
                    "protected_spikes": sac.get("protected_spikes", ""),
                    "protected_sole_coverage": sac.get("protected_sole_coverage", ""),
                    "protected_second_locus": sac.get("protected_second_locus", ""),
                    "n_sacrifice_used": len(r.get("sacrifice_used") or []),
                    "sacrifice_used_preview": ",".join(str(x) for x in (r.get("sacrifice_used") or [])[:12]),
                })
    write_csv(
        out_dir / "drm_rescue_candidates.csv",
        drm_rows,
        ["method", "task", "sample_index", "layer_idx", "candidate_or_rescued_id", "drm_swaps_applied", "drm_rescue_triggered"],
    )
    write_csv(
        out_dir / "sacrifice_pool_stats.csv",
        sac_rows,
        [
            "method", "task", "sample_index", "layer_idx", "sacrifice_pool_size",
            "protected_size", "protected_centers", "protected_spikes",
            "protected_sole_coverage", "protected_second_locus",
            "n_sacrifice_used", "sacrifice_used_preview",
        ],
    )

    summaries = {m: aggregate(m, audits.get(m, []), cc_map) for m in methods}
    (out_dir / "audit_summary.json").write_text(json.dumps(summaries, indent=2) + "\n")

    # Markdown reports.
    def md_table(methods_list: Sequence[str]) -> str:
        lines = [
            "| Method | churn vs orig | churn vs CC | multi-locus | spike | DRM rescue | avg swaps | over-budget | sink viol | band |\n",
            "|--------|--------------:|------------:|------------:|------:|-----------:|----------:|------------:|----------:|------|\n",
        ]
        for m in methods_list:
            s = summaries[m]
            churn_cc = s.get("churn_vs_cc_mean", "")
            ref_churn = s.get("churn_vs_cc_mean") if m.startswith("drm_acc") else s.get("churn_vs_original_mean")
            if m == "acc_sliminfer":
                band = band_check(m, s.get("churn_vs_cc_mean"))
            elif m == "drm_acc_swap_sliminfer":
                # churn vs ACC approximated via summary field if present; else vs CC as proxy note
                band = band_check(m, s.get("churn_vs_cc_mean"))
            elif m == "drm_swap_sliminfer":
                band = band_check(m, s.get("churn_vs_original_mean"))
            else:
                band = "ref"
            lines.append(
                f"| {m} | {s.get('churn_vs_original_mean')} | {churn_cc} | "
                f"{s.get('multi_locus_trigger_rate')} | {s.get('spike_protection_trigger_rate')} | "
                f"{s.get('drm_rescue_trigger_rate')} | {s.get('avg_swaps_per_layer')} | "
                f"{s.get('over_budget_count')} | {s.get('forced_sink_violations')} | {band} |\n"
            )
        return "".join(lines)

    acc_md = [
        "# ACC Active-Set Audit\n\n",
        f"- samples/task: {args.max_samples}\n",
        f"- tasks: {', '.join(TASKS)}\n",
        f"- prune layers: 9,18,26\n\n",
        md_table(["sliminfer_original", "cc_sliminfer_v1", "acc_sliminfer"]),
        "\n## Per-task triggers (ACC)\n\n",
        "```json\n" + json.dumps(summaries.get("acc_sliminfer", {}).get("per_task", {}), indent=2) + "\n```\n",
        "\n## GO band (ACC vs CC churn 3%–12%)\n\n",
        f"- status: **{band_check('acc_sliminfer', summaries.get('acc_sliminfer', {}).get('churn_vs_cc_mean'))}**\n",
        f"- over_budget: {summaries.get('acc_sliminfer', {}).get('over_budget_count')}\n",
        f"- sink_violations: {summaries.get('acc_sliminfer', {}).get('forced_sink_violations')}\n",
    ]
    (out_dir / "ACC_ACTIVE_SET_AUDIT.md").write_text("".join(acc_md), encoding="utf-8")

    drm_md = [
        "# DRM-ACC-Swap Active-Set Audit\n\n",
        md_table(["cc_sliminfer_v1", "acc_sliminfer", "drm_swap_sliminfer", "drm_acc_swap_sliminfer"]),
        "\n## Notes\n\n",
        "- Mainline is **budget-neutral K**. `drm_acc_add` (if present) is K+M only.\n",
        "- Target: DRM-ACC-Swap churn vs ACC/CC in 3%–12%; over-budget=0; sink violations=0.\n",
        f"- DRM-ACC-Swap band: **{band_check('drm_acc_swap', summaries.get('drm_acc_swap_sliminfer', {}).get('churn_vs_cc_mean'))}**\n",
        "\n## Per-task (drm_acc_swap)\n\n",
        "```json\n"
        + json.dumps(summaries.get("drm_acc_swap_sliminfer", {}).get("per_task", {}), indent=2)
        + "\n```\n",
    ]
    (out_dir / "DRM_ACC_SWAP_ACTIVE_SET_AUDIT.md").write_text("".join(drm_md), encoding="utf-8")

    # Case study focus.
    case_lines = [
        "# Case Study: MuSiQue / 2Wiki / TriviaQA\n\n",
        "Trigger rates from Phase-1 selection audit (prefill layers).\n\n",
    ]
    for t in TASKS:
        case_lines.append(f"## {t}\n\n")
        case_lines.append("| Method | multi-locus | spike | DRM | churn vs orig |\n")
        case_lines.append("|--------|------------:|------:|----:|--------------:|\n")
        for m in methods:
            pt = summaries.get(m, {}).get("per_task", {}).get(t, {})
            case_lines.append(
                f"| {m} | {pt.get('multi_locus_rate', '')} | {pt.get('spike_rate', '')} | "
                f"{pt.get('drm_rate', '')} | {pt.get('churn_vs_silm', '')} |\n"
            )
        case_lines.append("\n")
    (out_dir / "case_study_musique_2wiki_triviaqa.md").write_text("".join(case_lines), encoding="utf-8")

    # Decision gate for Phase-2/3.
    acc_churn = summaries.get("acc_sliminfer", {}).get("churn_vs_cc_mean")
    drm_churn = summaries.get("drm_acc_swap_sliminfer", {}).get("churn_vs_cc_mean")
    decision = {
        "acc_band": band_check("acc", acc_churn),
        "drm_acc_swap_band": band_check("drm_acc_swap", drm_churn),
        "acc_over_budget": summaries.get("acc_sliminfer", {}).get("over_budget_count"),
        "drm_over_budget": summaries.get("drm_acc_swap_sliminfer", {}).get("over_budget_count"),
        "acc_sink_violations": summaries.get("acc_sliminfer", {}).get("forced_sink_violations"),
        "drm_sink_violations": summaries.get("drm_acc_swap_sliminfer", {}).get("forced_sink_violations"),
        "run_qa": False,
    }
    ok_acc = decision["acc_band"] in {"PASS_band", "WARN_low_churn", "WARN_high_churn"} and decision["acc_over_budget"] == 0 and decision["acc_sink_violations"] == 0
    ok_drm = decision["drm_acc_swap_band"] in {"PASS_band", "WARN_low_churn", "WARN_high_churn"} and decision["drm_over_budget"] == 0 and decision["drm_sink_violations"] == 0
    # Zero churn on ACC relative to CC is fail.
    if decision["acc_band"] == "FAIL_zero_churn" or decision["drm_acc_swap_band"] == "FAIL_zero_churn":
        decision["run_qa"] = False
        decision["reason"] = "zero churn — hook/trigger ineffective"
    elif decision["acc_band"] == "FAIL_too_aggressive" or decision["drm_acc_swap_band"] == "FAIL_too_aggressive":
        decision["run_qa"] = False
        decision["reason"] = "churn >20% — reduce strength before QA"
    elif ok_acc and ok_drm:
        decision["run_qa"] = False  # Phase-1 only; user asked not to run QA yet
        decision["reason"] = "Phase-1 audit bands acceptable; QA deferred until explicit Phase-3"
        decision["phase2_go"] = True
    else:
        decision["reason"] = "audit incomplete or hard fail"
        decision["phase2_go"] = False
    (out_dir / "PHASE1_DECISION.json").write_text(json.dumps(decision, indent=2) + "\n")

    print(json.dumps({"out_dir": str(out_dir), "summaries": {m: {k: summaries[m].get(k) for k in (
        "n_records", "churn_vs_original_mean", "churn_vs_cc_mean",
        "multi_locus_trigger_rate", "spike_protection_trigger_rate",
        "drm_rescue_trigger_rate", "over_budget_count", "forced_sink_violations",
    )} for m in methods}, "decision": decision}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
