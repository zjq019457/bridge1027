#!/usr/bin/env python3
"""BridgePack completion pipeline orchestrator with persistent state."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.bridgepack_cpu_stage.common import FAIRNESS_DIR, read_csv
from tools.bridgepack_g1.common import append_csv, gpu_snapshot, mem_snapshot, utc_now, write_json, write_text

STAGE_NAMES = {
    0: "freeze_valid_results",
    1: "source_order_multibudget",
    2: "external_compression_baseline",
    3: "backfill_and_token_fairness",
    4: "efficiency",
    5: "final_paper_integration",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def scan_gpu(out_dir: Path, stage: str, phase: str) -> Dict[str, Any]:
    g = gpu_snapshot()
    m = mem_snapshot()
    foreign = []
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_gpu_memory",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        for line in proc.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3 and "gnome-remote-desktop" not in parts[1]:
                foreign.append(
                    {
                        "pid": parts[0],
                        "name": parts[1],
                        "used_gpu_memory": parts[2],
                    }
                )
    except Exception as exc:
        foreign.append({"pid": "scan_error", "name": str(exc), "used_gpu_memory": "unknown"})

    decision = "GO" if not foreign else "WAITING_FOR_SAFE_GPU"
    row = {
        "timestamp": utc_now(),
        "stage": stage,
        "phase": phase,
        "decision": decision,
        "bridgepack_pid": os.getpid(),
        "foreign_gpu_apps": json.dumps(foreign, ensure_ascii=False),
        "memory_total_mib": g.get("memory_total_mib"),
        "memory_used_mib": g.get("memory_used_mib"),
        "memory_free_mib": g.get("memory_free_mib"),
        "utilization_gpu": g.get("utilization_gpu"),
        "temperature_c": g.get("temperature_c"),
        "mem_available": m.get("mem_available"),
        "swap_free": m.get("swap_free"),
        "reasons": "foreign GPU workloads detected" if foreign else "ok",
        "apps": g.get("apps", ""),
    }
    append_csv(
        out_dir / "GPU_RESOURCE_LOG.csv",
        row,
        [
            "timestamp",
            "stage",
            "phase",
            "decision",
            "bridgepack_pid",
            "foreign_gpu_apps",
            "memory_total_mib",
            "memory_used_mib",
            "memory_free_mib",
            "utilization_gpu",
            "temperature_c",
            "mem_available",
            "swap_free",
            "reasons",
            "apps",
        ],
    )
    return row


def build_initial_state(out_dir: Path) -> Dict[str, Any]:
    return {
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "pipeline_status": "INITIALIZING",
        "current_stage": 0,
        "last_completed_stage": None,
        "hard_stop": None,
        "shared_codebase_isolation": True,
        "single_bridgepack_gpu_worker": True,
        "batch_size": 1,
        "stages": {
            str(idx): {
                "name": name,
                "status": "pending",
                "notes": "",
            }
            for idx, name in STAGE_NAMES.items()
        },
        "invalidated_results": [
            {
                "root": "results/bridgepack_g2_llama1024_sourceorder_20260727_023300",
                "tag": "INVALIDATED_BY_INDEPENDENT_WIRING_BUG",
                "forbidden": [
                    "Llama bridge vs independent +17.41",
                    "pooled bridge vs independent +10.43",
                    "all conclusions derived from buggy independent control",
                ],
            }
        ],
    }


def write_status_md(out_dir: Path, state: Mapping[str, Any]) -> None:
    lines = [
        "# PIPELINE_STATUS",
        "",
        f"Generated: {utc_now()}",
        "",
        f"- pipeline_status: `{state['pipeline_status']}`",
        f"- current_stage: `{state['current_stage']}`",
        f"- last_completed_stage: `{state['last_completed_stage']}`",
        f"- hard_stop: `{state['hard_stop']}`",
        "",
        "## Rules",
        "",
        "- Shared codebase; isolate BridgePack processes, configs, caches, and outputs.",
        "- Scan all GPU processes before GPU usage; wait with `WAITING_FOR_SAFE_GPU` if foreign work is present.",
        "- One BridgePack GPU worker only; batch size 1.",
        "- Historical results are read-only.",
        "- Invalidated independent results stay invalidated permanently.",
        "- Hard-stop on over-budget, sample mismatch, provenance/hash mismatch, wiring/no-op, CUDA OOM, Xid, scorer error, foreign-paper impact, RAM/swap anomaly, or repeated prediction failure.",
        "",
        "## Stages",
        "",
    ]
    for idx in range(6):
        item = state["stages"][str(idx)]
        marker = "DONE" if item["status"] == "done" else item["status"].upper()
        lines.append(f"- Stage {idx}: `{item['name']}` — **{marker}**")
        if item.get("notes"):
            lines.append(f"  - {item['notes']}")
    write_text(out_dir / "PIPELINE_STATUS.md", "\n".join(lines) + "\n")


def stop_pipeline(out_dir: Path, state: Dict[str, Any], reason: str, details: Optional[Dict[str, Any]] = None) -> None:
    state["pipeline_status"] = "STOPPED"
    state["hard_stop"] = reason
    state["updated_at"] = utc_now()
    write_json(out_dir / "pipeline_state.json", state)
    write_json(
        out_dir / "STOPPED.json",
        {
            "timestamp": utc_now(),
            "reason": reason,
            "details": details or {},
        },
    )
    write_status_md(out_dir, state)


def update_stage(state: Dict[str, Any], idx: int, status: str, notes: str = "") -> None:
    state["stages"][str(idx)]["status"] = status
    state["stages"][str(idx)]["notes"] = notes
    state["current_stage"] = idx
    if status == "done":
        state["last_completed_stage"] = idx
    state["updated_at"] = utc_now()


def freeze_stage0(out_dir: Path, state: Dict[str, Any]) -> None:
    update_stage(state, 0, "running", "Freezing currently valid G1/G2/fixed-independent results.")
    write_json(out_dir / "pipeline_state.json", state)
    write_status_md(out_dir, state)

    orig_g2 = REPO_ROOT / "results" / "bridgepack_g2_llama1024_sourceorder_20260727_023300"
    fix_g2 = REPO_ROOT / "results" / "bridgepack_g2_independent_fix_20260727_132221"
    g1 = REPO_ROOT / "results" / "bridgepack_g1_qwen1024_matched_20260727_004934"

    required = [
        g1 / "G1_FINAL_MECHANISM_DECISION.md",
        orig_g2 / "G2_FINAL_DECISION.md",
        fix_g2 / "G2_CORRECTED_FINAL_DECISION.md",
        fix_g2 / "G2_CORRECTED_BOOTSTRAP.csv",
        fix_g2 / "G2_CORRECTED_TWO_MODEL_BOOTSTRAP.csv",
        orig_g2 / "INVALIDATED_BY_INDEPENDENT_WIRING_BUG.md",
    ]
    missing = [str(p.relative_to(REPO_ROOT)) for p in required if not p.exists()]
    if missing:
        stop_pipeline(out_dir, state, "missing_required_stage0_artifacts", {"missing": missing})
        raise RuntimeError(f"Stage 0 missing artifacts: {missing}")

    corrected_boot = {}
    with (fix_g2 / "G2_CORRECTED_BOOTSTRAP.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            corrected_boot[row["comparison"]] = row
    corrected_pooled = {}
    with (fix_g2 / "G2_CORRECTED_TWO_MODEL_BOOTSTRAP.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            corrected_pooled[row["comparison"]] = row

    pooled_mmr = None
    with (orig_g2 / "G2_TWO_MODEL_BOOTSTRAP.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["comparison"] == "pooled source_order_bridge vs mmr_neighbor_source_order":
                pooled_mmr = row
                break
    if pooled_mmr is None:
        stop_pipeline(out_dir, state, "missing_pooled_mmr_result")
        raise RuntimeError("Missing pooled source_order_bridge vs mmr_neighbor_source_order")

    stage0_summary = {
        "generated_at": utc_now(),
        "confirmed_claims": {
            "qwen_source_vs_independent_delta": float(
                corrected_boot["qwen source_order_bridge vs qwen_independent"]["delta_macro"]
            ),
            "llama_source_vs_corrected_independent_delta": float(
                corrected_boot["llama3 source_order_bridge vs corrected_independent"]["delta_macro"]
            ),
            "pooled_source_vs_corrected_independent_delta": float(
                corrected_pooled["pooled source_order_bridge vs corrected_independent"]["delta_macro"]
            ),
            "pooled_source_vs_mmr_delta": float(pooled_mmr["delta_macro"]),
            "path_order_supported": False,
            "connectivity_score_independent_conclusion": "undetermined",
        },
        "read_only_roots": [
            "results/bridgepack_g1_qwen1024_matched_20260727_004934",
            "results/bridgepack_g2_llama1024_sourceorder_20260727_023300",
            "results/bridgepack_g2_independent_fix_20260727_132221",
        ],
        "valid_results_roots": [
            {
                "root": "results/bridgepack_g1_qwen1024_matched_20260727_004934",
                "valid_outputs": [
                    "Qwen source-order bridge selection",
                    "Qwen independent top-span",
                    "Qwen MMR",
                    "Qwen bridge-zero",
                ],
            },
            {
                "root": "results/bridgepack_g2_llama1024_sourceorder_20260727_023300",
                "valid_outputs": [
                    "Llama source-order bridge selection",
                    "Llama MMR",
                    "historical path-order reference",
                ],
            },
            {
                "root": "results/bridgepack_g2_independent_fix_20260727_132221",
                "valid_outputs": [
                    "corrected Llama independent top-span",
                    "corrected pooled bridge-vs-independent bootstrap",
                ],
            },
        ],
        "invalidated_outputs": [
            {
                "root": "results/bridgepack_g2_llama1024_sourceorder_20260727_023300",
                "tag": "INVALIDATED_BY_INDEPENDENT_WIRING_BUG",
            }
        ],
    }
    write_json(out_dir / "stage0_frozen_results_manifest.json", stage0_summary)

    hash_rows = []
    for path in required + [
        orig_g2 / "G2_TWO_MODEL_BOOTSTRAP.csv",
        FAIRNESS_DIR / "per_sample_scores.csv",
        REPO_ROOT / "configs" / "dataset2prompt.json",
        REPO_ROOT / "configs" / "dataset2maxlen.json",
    ]:
        hash_rows.append(
            {
                "path": str(path.relative_to(REPO_ROOT)),
                "sha256": sha256_file(path),
            }
        )
    with (out_dir / "FILE_SHA256.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "sha256"], lineterminator="\n")
        writer.writeheader()
        for row in hash_rows:
            writer.writerow(row)

    stage0_md = f"""# STAGE_0_FREEZE_VALID_RESULTS

Generated: {utc_now()}

## Frozen effective results

- Qwen source vs independent: **{stage0_summary['confirmed_claims']['qwen_source_vs_independent_delta']:.2f}**
- Llama source vs corrected independent: **{stage0_summary['confirmed_claims']['llama_source_vs_corrected_independent_delta']:.2f}**
- pooled source vs corrected independent: **{stage0_summary['confirmed_claims']['pooled_source_vs_corrected_independent_delta']:.2f}**
- pooled source vs MMR: **{stage0_summary['confirmed_claims']['pooled_source_vs_mmr_delta']:.2f}**
- path order supported: **NO**
- connectivity score independent contribution: **undetermined**

## Accounting rule

All future paper-facing statistics in this pipeline must read **corrected independent**
from `results/bridgepack_g2_independent_fix_20260727_132221/` and must never use
the invalidated G2 independent outputs.
"""
    write_text(out_dir / "STAGE_0_FREEZE_VALID_RESULTS.md", stage0_md)
    write_text(out_dir / "STAGE_0.DONE", f"{utc_now()}\n")

    update_stage(state, 0, "done", "Stage 0 frozen; corrected independent is canonical for future stats.")
    state["pipeline_status"] = "STAGE_0_COMPLETE_STAGE_1_PENDING"
    write_json(out_dir / "pipeline_state.json", state)
    write_status_md(out_dir, state)


def write_stage_plan(out_dir: Path) -> None:
    plan = {
        "generated_at": utc_now(),
        "stages": {
            "1": {
                "summary": "Fill source-order bridge 256/512 for Qwen and Llama-3; 12-sample smoke then full 1150 per unit.",
                "outputs": [
                    "SOURCE_ORDER_MULTIBUDGET_RESULTS.csv",
                    "SOURCE_ORDER_MULTIBUDGET_BOOTSTRAP.csv",
                    "SOURCE_ORDER_BUDGET_SCALING.md",
                ],
            },
            "2": {
                "summary": "Run official/adapted external compression baseline with strict raw/adapted separation.",
                "outputs": [
                    "G3_EXTERNAL_COMPRESSION_AUDIT.md",
                    "G3_EXTERNAL_COMPRESSION_RESULTS.csv",
                    "G3_EXTERNAL_COMPRESSION_BOOTSTRAP.csv",
                    "G3_CLAIM_GUARD.md",
                ],
            },
            "3": {
                "summary": "Backfill and matched-actual-token fairness checks.",
                "outputs": [
                    "G4_BACKFILL_RESULTS.csv",
                    "G4_MATCHED_ACTUAL_TOKEN_RESULTS.csv",
                    "G4_PACKING_FAIRNESS_REPORT.md",
                    "G4_BOOTSTRAP.csv",
                ],
            },
            "4": {
                "summary": "Efficiency subset with latency/memory/token breakdown.",
                "outputs": [
                    "G5_EFFICIENCY_RESULTS.csv",
                    "G5_LATENCY_BREAKDOWN.csv",
                    "G5_MEMORY_REPORT.csv",
                    "G5_EFFICIENCY_SUMMARY.md",
                ],
            },
            "5": {
                "summary": "CPU-only final tables, claims, and draft revisions.",
                "outputs": [
                    "FINAL_EXPERIMENT_SUMMARY.md",
                    "FINAL_VALID_RESULTS_MANIFEST.json",
                    "FINAL_MAIN_TABLES.csv",
                    "FINAL_CLAIM_GUARD.md",
                    "METHODS_REVISION_DRAFT.md",
                    "RESULTS_REVISION_DRAFT.md",
                    "ABSTRACT_REVISION_DRAFT.md",
                    "FIGURE_REVISION_PLAN.md",
                ],
            },
        },
    }
    write_json(out_dir / "stage_execution_plan.json", plan)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--through_stage", type=int, default=0, choices=[0, 1, 2, 3, 4, 5])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    state_path = out_dir / "pipeline_state.json"
    state = read_json(state_path) if args.resume and state_path.exists() else build_initial_state(out_dir)
    write_json(state_path, state)
    write_stage_plan(out_dir)
    scan_gpu(out_dir, "pipeline_init", "startup")
    write_status_md(out_dir, state)

    if args.through_stage >= 0 and state["stages"]["0"]["status"] != "done":
        freeze_stage0(out_dir, state)

    if args.through_stage > 0:
        state = read_json(state_path)
        state["pipeline_status"] = "PLANNED_STAGE_EXECUTION_PENDING"
        state["updated_at"] = utc_now()
        for idx in range(1, args.through_stage + 1):
            if state["stages"][str(idx)]["status"] == "pending":
                state["stages"][str(idx)]["notes"] = "Planned but not auto-executed by this initialization run."
        write_json(state_path, state)
        write_status_md(out_dir, state)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
