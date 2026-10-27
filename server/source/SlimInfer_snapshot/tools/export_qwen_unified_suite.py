#!/usr/bin/env python3
"""Export a checksum-addressed Qwen experiment registry without paper prose."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

_REPO_BOOTSTRAP = Path(__file__).resolve().parents[1]
if str(_REPO_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(_REPO_BOOTSTRAP))

from tools.qwen_pipeline_common import (
    PROTOCOL_DIR,
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


def _copy_text(source: Path, target: Path) -> None:
    if source.exists():
        atomic_write_text(target, source.read_text(encoding="utf-8"))


def _source_entry(stage: int, method: str, kind: str, path: Path) -> Dict[str, Any]:
    return {
        "stage": stage, "method": method, "kind": kind, "path": str(path),
        "exists": path.exists(),
        "sha256": sha256_file(path) if path.is_file() else "directory",
    }


def _write_sha(root: Path) -> None:
    files = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS.txt"
    )
    atomic_write_text(
        root / "SHA256SUMS.txt",
        "".join(f"{sha256_file(path)}  {path.relative_to(root)}\n" for path in files),
    )


def _focused_export(
    root: Path,
    *,
    paper_scope: str,
    methods: Sequence[str],
    sources: Sequence[Mapping[str, Any]],
    shared_analysis: Path,
    efficiency: Path | None,
    mechanism: Path | None,
) -> None:
    root.mkdir(parents=True, exist_ok=False)
    selected = [row for row in sources if row.get("method") in methods or row.get("method") == "shared"]
    atomic_write_json(root / "SCOPE.json", {
        "created_at": now_iso(), "protocol_id": PROTOCOL_ID,
        "experiment_scope": paper_scope, "methods": list(methods),
        "paper_text_included": False,
    })
    atomic_write_csv(root / "SOURCE_MAP.csv", selected, [
        "stage", "method", "kind", "path", "exists", "sha256",
    ])
    for name in (
        "scores_by_sample.csv", "scores_by_task.csv", "common_valid_scores.csv",
        "pairwise_common_valid.csv", "win_loss_tie.csv", "paired_bootstrap.json",
        "BUDGET_SELECTION_AUDIT.csv", "ACTIVE_TRACE_COVERAGE.csv",
    ):
        _copy_text(shared_analysis / name, root / name)
    if efficiency:
        atomic_write_json(root / "EFFICIENCY_SOURCE.json", {"path": str(efficiency)})
    if mechanism:
        atomic_write_json(root / "MECHANISM_SOURCE.json", {"path": str(mechanism)})
    _write_sha(root)


def export_suite(pipeline_dir: Path, *, dry_run: bool = False) -> Dict[str, Any]:
    gate = run_protocol_gate(pipeline_dir, 12)
    if gate["decision"] != "PASS":
        raise RuntimeError("Stage 12 protocol gate failed")
    decisions: List[Dict[str, Any]] = []
    incomplete: List[int] = []
    for stage in range(12):
        path = stage_decision_path(pipeline_dir, stage)
        if not path.exists() or load_json(path).get("decision") not in ACCEPTED:
            incomplete.append(stage)
        else:
            decisions.append(load_json(path))
    pointer = pipeline_dir / "QWEN_FINAL_EXPORT_ROOT.json"
    if pointer.exists():
        root = Path(load_json(pointer)["path"])
    else:
        root = REPO_ROOT / "results_exports" / f"qwen_unified_protocol_v1_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        atomic_write_json(pointer, {"created_at": now_iso(), "path": str(root)})
    plan = {
        "created_at": now_iso(), "protocol_id": PROTOCOL_ID,
        "pipeline_dir": str(pipeline_dir), "incomplete_stages": incomplete,
        "accepted_soft_blockers": [row["stage"] for row in decisions if row.get("decision") == "SOFT_BLOCKED"],
        "export_root": str(root),
    }
    atomic_write_json(pipeline_dir / "STAGE_12_EXPORT_PLAN.json", plan)
    if incomplete:
        raise RuntimeError(f"Stages are not accepted; refusing final export: {incomplete}")
    if dry_run:
        result = {"stage": 12, "name": "final_standardized_export", "decision": "NOT_RUN", **plan}
        atomic_write_json(stage_decision_path(pipeline_dir, 12), result)
        return result
    if root.exists() and (root / "EXPORT_COMPLETE.json").exists():
        decision = {
            "stage": 12, "name": "final_standardized_export", "created_at": now_iso(),
            "decision": "PASS", "export_root": str(root), "next_stage": None, "resumed": True,
        }
        atomic_write_json(stage_decision_path(pipeline_dir, 12), decision)
        return decision
    root.mkdir(parents=True, exist_ok=False)
    stage2 = load_json(stage_decision_path(pipeline_dir, 2))
    stage3 = load_json(stage_decision_path(pipeline_dir, 3))
    analysis_root = Path(stage3["export_root"])
    sources: List[Dict[str, Any]] = []
    for row in _read_csv(analysis_root / "SOURCE_MAP.csv"):
        sources.append({
            "stage": 2 if row["method"] == "cc_sliminfer_v1" else 0,
            "method": row["method"], "kind": f"prediction:{row['task']}",
            "path": row["path"], "exists": Path(row["path"]).exists(), "sha256": row["sha256"],
        })
    sources.append(_source_entry(2, "cc_sliminfer_v1", "run_root", Path(stage2["run_root"])))
    stage4 = load_json(stage_decision_path(pipeline_dir, 4))
    if stage4.get("run_root"):
        sources.append(_source_entry(4, "full_kv_fa2", "run_root", Path(stage4["run_root"])))
    for stage in (5, 6, 7):
        decision = load_json(stage_decision_path(pipeline_dir, stage))
        method = {5: "minference_official", 6: "flexprefill_official", 7: "lazyllm_official"}[stage]
        path = Path(decision.get("run_root", pipeline_dir / f"STAGE_{stage:02d}_OFFICIAL_PATH_AUDIT.json"))
        sources.append(_source_entry(stage, method, "official_stage", path))
    stage8 = load_json(stage_decision_path(pipeline_dir, 8))
    efficiency_root = Path(stage8["run_root"]) if stage8.get("run_root") else None
    if efficiency_root:
        sources.append(_source_entry(8, "shared", "efficiency", efficiency_root))
    stage9 = load_json(stage_decision_path(pipeline_dir, 9))
    stage10 = load_json(stage_decision_path(pipeline_dir, 10))
    cc_mechanism = Path(stage9["run_root"]) if stage9.get("run_root") else None
    drm_mechanism = Path(stage10["run_root"]) if stage10.get("run_root") else None
    if cc_mechanism:
        sources.append(_source_entry(9, "cc_sliminfer_v1", "mechanism", cc_mechanism))
    if drm_mechanism:
        sources.append(_source_entry(10, "sliminfer_drm_add_km", "mechanism", drm_mechanism))
    stage11 = load_json(stage_decision_path(pipeline_dir, 11))
    if stage11.get("run_root"):
        sources.append(_source_entry(11, "shared", "oom_fill", Path(stage11["run_root"])))

    atomic_write_json(root / "PROTOCOL_ATTESTATION.json", load_json(pipeline_dir / "PROTOCOL_ATTESTATION.json"))
    atomic_write_json(root / "PIPELINE_MANIFEST.json", load_json(pipeline_dir / "PIPELINE_MANIFEST.json"))
    atomic_write_json(root / "STAGE_DECISIONS.json", {str(row["stage"]): row for row in decisions})
    atomic_write_json(root / "CANONICAL_ORIGINAL_HASH_LOCK.json", load_json(pipeline_dir / "CANONICAL_ORIGINAL_HASH_LOCK.json"))
    _copy_text(pipeline_dir / "RUN_REGISTRY.csv", root / "RUN_REGISTRY.csv")
    atomic_write_csv(root / "RESULT_REGISTRY.csv", [
        {
            "stage": row["stage"], "name": row.get("name"), "decision": row["decision"],
            "reason": row.get("reason", ""),
            "root": row.get("run_root", row.get("export_root", "")),
        }
        for row in decisions
    ], ["stage", "name", "decision", "reason", "root"])
    atomic_write_csv(root / "SOURCE_MAP.csv", sources, [
        "stage", "method", "kind", "path", "exists", "sha256",
    ])
    _copy_text(PROTOCOL_DIR / "PROTOCOL_IDENTITY_MATRIX.csv", root / "PROTOCOL_IDENTITY_MATRIX.csv")
    _copy_text(analysis_root / "FIELD_COVERAGE.csv", root / "FIELD_COVERAGE.csv")
    _copy_text(analysis_root / "ACTIVE_TRACE_COVERAGE.csv", root / "ACTIVE_TRACE_COVERAGE.csv")
    _copy_text(analysis_root / "OOM_MANIFEST.csv", root / "OOM_REGISTRY.csv")
    for name in (
        "scores_by_sample.csv", "scores_by_task.csv", "common_valid_scores.csv",
        "pairwise_common_valid.csv", "win_loss_tie.csv", "outlier_sensitivity.csv",
        "category_summary.csv", "paired_bootstrap.json", "BOOTSTRAP_CONFIG.json",
        "BUDGET_SELECTION_AUDIT.csv", "GLOBAL_COMMON_VALID.json", "PAIRWISE_COMMON_VALID.json",
    ):
        _copy_text(analysis_root / name, root / name)
    limitations = [
        "# Known Limitations", "",
        "This export contains experiment facts and provenance only; it contains no paper text or claims.", "",
    ]
    for row in decisions:
        if row.get("decision") == "SOFT_BLOCKED":
            limitations.append(f"- Stage {row['stage']:02d} `{row.get('name', '')}`: {row.get('reason', 'soft blocked')}")
    atomic_write_text(root / "KNOWN_LIMITATIONS.md", "\n".join(limitations) + "\n")

    cc_root = REPO_ROOT / "results_exports" / f"cc_qwen_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    drm_root = REPO_ROOT / "results_exports" / f"drm_qwen_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    _focused_export(
        cc_root, paper_scope="cc_qwen_experiments",
        methods=("sliminfer_original", "cc_sliminfer_v1", "full_kv_fa2", "minference_official", "flexprefill_official", "lazyllm_official"),
        sources=sources, shared_analysis=analysis_root, efficiency=efficiency_root, mechanism=cc_mechanism,
    )
    _focused_export(
        drm_root, paper_scope="drm_qwen_experiments",
        methods=("sliminfer_original", "drm_swap_original_k", "sliminfer_larger_km", "sliminfer_random_add_km", "sliminfer_drm_add_km", "full_kv_fa2", "minference_official", "flexprefill_official", "lazyllm_official"),
        sources=sources, shared_analysis=analysis_root, efficiency=efficiency_root, mechanism=drm_mechanism,
    )
    atomic_write_json(root / "FOCUSED_EXPORTS.json", {"cc_qwen": str(cc_root), "drm_qwen": str(drm_root)})
    atomic_write_json(root / "EXPORT_COMPLETE.json", {
        "created_at": now_iso(), "protocol_id": PROTOCOL_ID,
        "source_entries": len(sources), "soft_blocked_stages": plan["accepted_soft_blockers"],
        "cc_export": str(cc_root), "drm_export": str(drm_root),
    })
    _write_sha(root)
    decision = {
        "stage": 12, "name": "final_standardized_export", "created_at": now_iso(),
        "decision": "PASS", "export_root": str(root), "cc_export_root": str(cc_root),
        "drm_export_root": str(drm_root), "next_stage": None,
    }
    atomic_write_json(stage_decision_path(pipeline_dir, 12), decision)
    return decision


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = export_suite(args.pipeline_dir.resolve(), dry_run=args.dry_run)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
