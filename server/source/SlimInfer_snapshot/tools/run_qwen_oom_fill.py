#!/usr/bin/env python3
"""Audit and, only when equivalence-gated, execute final per-sample OOM fills."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

_REPO_BOOTSTRAP = Path(__file__).resolve().parents[1]
if str(_REPO_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(_REPO_BOOTSTRAP))

from tools.qwen_pipeline_common import (
    PROTOCOL_ID,
    REPO_ROOT,
    atomic_write_csv,
    atomic_write_json,
    atomic_write_text,
    load_json,
    now_iso,
    sha256_file,
    stage_decision_path,
)
from tools.qwen_protocol_gate import run_protocol_gate


ACCEPTED = {"PASS", "SOFT_BLOCKED"}


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _collect_raw_failures(pipeline_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    stage3 = load_json(stage_decision_path(pipeline_dir, 3))
    if stage3.get("decision") == "PASS":
        rows.extend(_read_csv(Path(stage3["export_root"]) / "OOM_MANIFEST.csv"))
    stage4 = load_json(stage_decision_path(pipeline_dir, 4))
    if stage4.get("decision") == "PASS":
        rows.extend(_read_csv(Path(stage4["run_root"]) / "full/OOM_MANIFEST.csv"))
    for stage in (5, 6, 7):
        decision = load_json(stage_decision_path(pipeline_dir, stage))
        if decision.get("decision") == "PASS" and decision.get("run_root"):
            rows.extend(_read_csv(Path(decision["run_root"]) / "OOM_MANIFEST.csv"))
    return rows


def build_plan(pipeline_dir: Path, *, execute: bool = False, dry_run: bool = False) -> Dict[str, Any]:
    gate = run_protocol_gate(pipeline_dir, 11)
    if gate["decision"] != "PASS":
        raise RuntimeError("Stage 11 protocol gate failed")
    prerequisite_failures = []
    for stage in range(11):
        path = stage_decision_path(pipeline_dir, stage)
        if not path.exists() or load_json(path).get("decision") not in ACCEPTED:
            prerequisite_failures.append(stage)
    if prerequisite_failures:
        raise RuntimeError(f"OOM fill is final-only; prerequisites not accepted: {prerequisite_failures}")
    pointer = pipeline_dir / "QWEN_OOM_FILL_ROOT.json"
    if pointer.exists():
        fill_root = Path(load_json(pointer)["path"])
    else:
        fill_root = REPO_ROOT / "results" / f"qwen_oom_offload_fill_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        atomic_write_json(pointer, {"created_at": now_iso(), "path": str(fill_root)})
    rows = _collect_raw_failures(pipeline_dir)
    raw_files = []
    stage3 = load_json(stage_decision_path(pipeline_dir, 3))
    if stage3.get("decision") == "PASS":
        source_map = Path(stage3["export_root"]) / "SOURCE_MAP.csv"
        for source in _read_csv(source_map):
            path = Path(source["path"])
            if path.exists():
                raw_files.append({"path": str(path), "sha256": sha256_file(path)})
    adapter = REPO_ROOT / "tools/qwen_whole_model_offload_adapter.py"
    plan = {
        "created_at": now_iso(), "protocol_id": PROTOCOL_ID,
        "fill_root": str(fill_root), "raw_rows": len(rows),
        "equivalence_gate_samples": 20, "one_sample_per_process": True,
        "batch_size": 1, "raw_predictions_immutable": True,
        "single_sample_timeout_required": True, "raw_hash_lock": raw_files,
        "whole_model_offload_adapter": str(adapter), "adapter_present": adapter.exists(),
        "execute_requested": execute, "dry_run": dry_run,
    }
    atomic_write_json(pipeline_dir / "STAGE_11_OOM_FILL_PLAN.json", plan)
    if dry_run or not execute:
        decision = {"stage": 11, "name": "oom_offload_fill", "decision": "NOT_RUN", **plan}
        atomic_write_json(stage_decision_path(pipeline_dir, 11), decision)
        return decision
    fill_root.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(fill_root / "raw_oom_failed_manifest.csv", rows, sorted({key for row in rows for key in row} or {"method"}))
    atomic_write_json(fill_root / "RAW_UNCHANGED_HASH_LOCK.json", {
        "created_at": now_iso(), "files": raw_files, "raw_predictions_immutable": True,
    })
    if not adapter.exists():
        decision = {
            "stage": 11, "name": "oom_offload_fill", "created_at": now_iso(),
            "decision": "SOFT_BLOCKED", "reason": "equivalence_gated_whole_model_offload_adapter_absent",
            "run_root": str(fill_root), "raw_failures": len(rows), "filled": 0,
            "raw_predictions_modified": False, "next_stage": 12,
        }
        atomic_write_json(stage_decision_path(pipeline_dir, 11), decision)
        atomic_write_text(
            pipeline_dir / "STAGE_11_SOFT_BLOCKER.md",
            "# Stage 11 Soft Blocker\n\nAll raw stages were allowed to finish first and their prediction "
            "files were hash-locked. The frozen Qwen runner has no audited whole-model CPU+GPU offload "
            "adapter, so the required 20-sample exact-equivalence gate cannot be run. No OOM row was filled "
            "and no raw prediction was modified. The final export may continue with this limitation.\n",
        )
        atomic_write_text(
            fill_root / "OFFLOAD_EQUIVALENCE_BLOCKED.md",
            "# Offload Equivalence Blocked\n\nNo adapter means no equivalence gate and therefore no fill. "
            "KV-cache swapping intrinsic to SlimInfer is not relabeled as whole-model OOM offload.\n",
        )
        return decision
    raise RuntimeError(
        "An offload adapter appeared after this pipeline implementation; it requires an explicit identity audit before execution"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = build_plan(args.pipeline_dir.resolve(), execute=args.execute, dry_run=args.dry_run)
    print(json.dumps(result, indent=2))
    return 0 if result.get("decision") in {"PASS", "SOFT_BLOCKED", "NOT_RUN"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
