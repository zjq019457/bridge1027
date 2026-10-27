#!/usr/bin/env python3
"""Immutable identity gate used before every unified Qwen pipeline stage."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

_REPO_BOOTSTRAP = Path(__file__).resolve().parents[1]
if str(_REPO_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(_REPO_BOOTSTRAP))

from tools.qwen_pipeline_common import (
    PROTOCOL_ID,
    QWEN_DRM_ROOT,
    REPO_ROOT,
    TASKS,
    atomic_write_json,
    canonical_bytes,
    implementation_revision,
    load_json,
    now_iso,
    protocol_manifest_id,
    sha256_file,
    sha256_bytes,
)


FROZEN_BASE_COMMIT = "2fdc9815dc0cd9467bfbbc855c7805101dc15706"
FROZEN_IMPLEMENTATION_REVISION = (
    "ff48d133659070d8634f80873fbb1cfb58f232f3604a402d0bad2c8a269bb068"
)
FROZEN_DIFF_SHA256 = (
    "257ab1d4005587326fc3cf66db47e460213807e9a766cc5ccb2077887b4cc42a"
)
RUNTIME_FILES = (
    REPO_ROOT / "experiments/run_qwen25_sliminfer_variants.py",
    REPO_ROOT / "utils/cache_utils.py",
    REPO_ROOT / "utils/processor.py",
    REPO_ROOT / "models/modeling_qwen2.py",
)
CANONICAL_ORIGINAL = QWEN_DRM_ROOT / "run_sliminfer_original"
CONTROL_PLANE_FILES = (
    REPO_ROOT / "tools/qwen_protocol_gate.py",
    REPO_ROOT / "tools/qwen_runtime_watchdog.py",
    REPO_ROOT / "tools/run_qwen_unified_task.py",
    REPO_ROOT / "tools/run_qwen_unified_pipeline.py",
    REPO_ROOT / "tools/analyze_qwen_unified_suite.py",
    REPO_ROOT / "tools/run_qwen_unified_efficiency.py",
    REPO_ROOT / "tools/qwen_efficiency_direct_gpu_loader.py",
    REPO_ROOT / "tools/qwen_efficiency_loader_canary.py",
    REPO_ROOT / "tools/diagnose_qwen_stage8_swap.py",
    REPO_ROOT / "tools/analyze_qwen_mechanisms.py",
    REPO_ROOT / "tools/qwen_ablation_adapter.py",
    REPO_ROOT / "tools/run_qwen_ablation_suite.py",
    REPO_ROOT / "tools/run_qwen_oom_fill.py",
    REPO_ROOT / "tools/export_qwen_unified_suite.py",
)


def _git_output(*args: str) -> str:
    import subprocess

    result = subprocess.run(
        ["git", *args], cwd=REPO_ROOT, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stdout}")
    return result.stdout


def _canonical_files() -> Iterable[Path]:
    if not CANONICAL_ORIGINAL.is_dir():
        return ()
    return tuple(sorted(path for path in CANONICAL_ORIGINAL.rglob("*") if path.is_file()))


def _canonical_snapshot() -> Dict[str, Any]:
    files = [
        {
            "path": str(path.relative_to(CANONICAL_ORIGINAL)),
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in _canonical_files()
    ]
    return {
        "root": str(CANONICAL_ORIGINAL),
        "file_count": len(files),
        "files": files,
    }


def create_or_validate_canonical_lock(pipeline_dir: Path) -> Dict[str, Any]:
    """Freeze the canonical tree once, then byte-verify it on every stage entry."""
    lock_path = pipeline_dir / "CANONICAL_ORIGINAL_HASH_LOCK.json"
    actual = _canonical_snapshot()
    if not lock_path.exists():
        if actual["file_count"] == 0:
            raise RuntimeError(f"Canonical Original is absent: {CANONICAL_ORIGINAL}")
        lock = {
            "created_at": now_iso(),
            "protocol_id": PROTOCOL_ID,
            "immutable": True,
            **actual,
        }
        atomic_write_json(lock_path, lock)
        return {"pass": True, "created": True, "mismatches": [], "lock": lock}
    lock = load_json(lock_path)
    expected = {
        row["path"]: (int(row["size"]), str(row["sha256"]))
        for row in lock.get("files", [])
    }
    observed = {
        row["path"]: (int(row["size"]), str(row["sha256"]))
        for row in actual.get("files", [])
    }
    mismatches: List[Dict[str, Any]] = []
    for path in sorted(set(expected) | set(observed)):
        if expected.get(path) != observed.get(path):
            mismatches.append({
                "path": path,
                "expected": expected.get(path),
                "actual": observed.get(path),
            })
    return {"pass": not mismatches, "created": False, "mismatches": mismatches, "lock": lock}


def run_protocol_gate(pipeline_dir: Path, stage: int) -> Dict[str, Any]:
    manifest_id, manifest_sha = protocol_manifest_id()
    commit = _git_output("rev-parse", "HEAD").strip()
    diff = _git_output("diff", "--binary")
    import hashlib

    diff_sha = hashlib.sha256(diff.encode("utf-8")).hexdigest()
    revision = implementation_revision(RUNTIME_FILES)
    stage0 = load_json(pipeline_dir / "STAGE_00_DECISION.json")
    stage1 = load_json(pipeline_dir / "STAGE_01_DECISION.json")
    frozen_revision = load_json(pipeline_dir / "IMPLEMENTATION_REVISION.json")
    canonical = create_or_validate_canonical_lock(pipeline_dir)
    control_files = [
        {"path": str(path.relative_to(REPO_ROOT)), "sha256": sha256_file(path)}
        for path in CONTROL_PLANE_FILES
    ]
    control_revision = sha256_bytes(canonical_bytes(control_files))
    atomic_write_json(pipeline_dir / "CONTROL_PLANE_IMPLEMENTATION_REVISION.json", {
        "created_at": now_iso(), "protocol_id": PROTOCOL_ID,
        "runtime_revision": revision, "control_plane_revision": control_revision,
        "files": control_files,
        "note": "Control-plane hashes are separate from the Stage-1-frozen runtime revision.",
    })
    methods = load_json(REPO_ROOT / "protocol/unified_longbench_v1/method_configs.json")
    task_manifest = load_json(REPO_ROOT / "protocol/unified_longbench_v1/task_manifest.json")

    checks: Dict[str, bool] = {
        "protocol_id": manifest_id == PROTOCOL_ID,
        "base_commit": commit == FROZEN_BASE_COMMIT,
        "tracked_diff": diff_sha == FROZEN_DIFF_SHA256,
        "runtime_revision": revision == FROZEN_IMPLEMENTATION_REVISION,
        "frozen_revision_record": (
            frozen_revision.get("runtime_revision") == FROZEN_IMPLEMENTATION_REVISION
            and frozen_revision.get("tracked_diff_sha256") == FROZEN_DIFF_SHA256
        ),
        "stage_00": stage0.get("decision") == "PASS",
        "stage_01": stage1.get("decision") == "PASS",
        "canonical_original_immutable": bool(canonical["pass"]),
        "method_manifest": set(methods.get("methods", {})) >= {
            "sliminfer_original", "cc_sliminfer_v1", "drm_swap_original_k",
            "sliminfer_larger_km", "sliminfer_random_add_km",
            "sliminfer_drm_add_km", "full_kv_fa2",
        },
        "task_manifest": len(task_manifest.get("tasks", [])) == len(TASKS),
    }
    report = {
        "created_at": now_iso(),
        "stage": int(stage),
        "decision": "PASS" if all(checks.values()) else "HARD_BLOCKED",
        "protocol_id": PROTOCOL_ID,
        "computed_protocol_id": manifest_id,
        "protocol_manifest_sha256": manifest_sha,
        "git_commit": commit,
        "tracked_diff_sha256": diff_sha,
        "implementation_revision": revision,
        "control_plane_revision": control_revision,
        "checks": checks,
        "canonical_lock": {
            "path": str(pipeline_dir / "CANONICAL_ORIGINAL_HASH_LOCK.json"),
            "created": canonical["created"],
            "mismatches": canonical["mismatches"],
        },
        "failures": [name for name, passed in checks.items() if not passed],
    }
    atomic_write_json(pipeline_dir / f"STAGE_{stage:02d}_PROTOCOL_GATE.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-dir", type=Path, required=True)
    parser.add_argument("--stage", type=int, required=True)
    args = parser.parse_args()
    report = run_protocol_gate(args.pipeline_dir.resolve(), args.stage)
    print(json.dumps(report, indent=2))
    return 0 if report["decision"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
