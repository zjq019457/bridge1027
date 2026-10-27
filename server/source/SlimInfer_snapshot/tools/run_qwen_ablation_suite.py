#!/usr/bin/env python3
"""Run equivalence-gated six-task CC or DRM ablations sequentially."""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

_REPO_BOOTSTRAP = Path(__file__).resolve().parents[1]
if str(_REPO_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(_REPO_BOOTSTRAP))

from tools.analyze_qwen_mechanisms import _cc_analysis, _drm_analysis
from tools.consolidate_official_repro_results import per_sample_score
from tools.qwen_ablation_adapter import ADAPTER_ID
from tools.qwen_pipeline_common import (
    COUNTS,
    DATA_ROOT,
    PROTOCOL_ID,
    PYTHON,
    QWEN_DRM_ROOT,
    QWEN_MODEL_ROOT,
    REPO_ROOT,
    RUN_DIR_BY_METHOD,
    atomic_write_csv,
    atomic_write_json,
    atomic_write_text,
    canonical_bytes,
    canonical_sample_identity,
    implementation_revision,
    iter_jsonl,
    load_json,
    load_task_samples,
    now_iso,
    per_task_score,
    prediction_status,
    sha256_bytes,
    sha256_file,
    stage_decision_path,
)
from tools.qwen_protocol_gate import (
    FROZEN_IMPLEMENTATION_REVISION,
    RUNTIME_FILES,
    run_protocol_gate,
)
from tools.qwen_resource_guard import collect_snapshot, monitor_resources, write_blocker
from tools.qwen_runtime_watchdog import RuntimeWatchdog


REPRESENTATIVE_TASKS = (
    "qasper", "hotpotqa", "2wikimqa", "musique", "gov_report", "repobench-p",
)
CANARY_PLAN = "ORIGINAL_EQUIVALENCE_CANARY_PLAN.json"
EXPECTED_K = {9: 192, 18: 96, 26: 64}


def _trace_layer(event: Mapping[str, Any]) -> int:
    value = event.get("abs_layer")
    if value is None:
        value = event.get("layer_idx", -1)
    return int(value)


def _cc_specs() -> List[Dict[str, Any]]:
    common = {
        "family": "cc", "alpha": 0.25, "min_cluster_len": 2,
        "max_neighbors": 2, "replacement_cap": 0.05, "component": "full", "seed": 0,
    }
    return [
        {"id": "sliminfer_original", "reuse": "sliminfer_original", "role": "baseline"},
        {"id": "cc_full", "reuse": "cc_sliminfer_v1", "role": "frozen_main", **common},
        {"id": "cc_continuity_alpha_0", **common, "alpha": 0.0, "axis": "continuity_strength"},
        {"id": "cc_continuity_alpha_0_5", **common, "alpha": 0.5, "axis": "continuity_strength"},
        {"id": "cc_neighbor_radius_1", **common, "max_neighbors": 1, "axis": "neighbor_radius"},
        {"id": "cc_neighbor_radius_4", **common, "max_neighbors": 4, "axis": "neighbor_radius"},
        {
            "id": "cc_budget_conservative", **common, "replacement_cap": 0.025,
            "axis": "replacement_budget", "budget_label": "conservative",
        },
        {
            "id": "cc_budget_aggressive", **common, "replacement_cap": 0.10,
            "axis": "replacement_budget", "budget_label": "aggressive",
        },
        {"id": "cc_neighbor_only", **common, "component": "neighbor_only", "axis": "component"},
        {"id": "cc_cluster_only", **common, "component": "cluster_only", "axis": "component"},
        {
            "id": "cc_random_swap_fixed_k", **common, "component": "random_swap_fixed_k",
            "axis": "control", "seed": 0,
        },
    ]


def _drm_specs() -> List[Dict[str, Any]]:
    common = {
        "family": "drm", "runtime_method": "sliminfer_original", "policy": "add",
        "score": "drm", "rescue_ratio": 0.05, "alpha_diff": 0.5,
        "lambda_red": 0.25, "eta_rescue": 0.25, "eta_active": 0.25, "seed": 0,
    }
    return [
        {"id": "sliminfer_original", "reuse": "sliminfer_original", "role": "baseline"},
        {"id": "drm_full_ratio_0_05", "reuse": "sliminfer_drm_add_km", "role": "frozen_main", **common},
        {"id": "drm_ratio_0_02", **common, "rescue_ratio": 0.02, "axis": "rescue_ratio"},
        {"id": "drm_ratio_0_10", **common, "rescue_ratio": 0.10, "axis": "rescue_ratio"},
        {
            "id": "random_add_seed_0", "reuse": "sliminfer_random_add_km", "role": "control",
            **common, "score": "random",
        },
        {"id": "random_add_seed_1", **common, "score": "random", "seed": 1, "axis": "random_seed"},
        {"id": "random_add_seed_2", **common, "score": "random", "seed": 2, "axis": "random_seed"},
        {"id": "drm_without_gain", **common, "alpha_diff": 0.0, "axis": "component"},
        {
            "id": "drm_without_active_core_redundancy", **common,
            "lambda_red": 0.0, "eta_active": 0.0, "axis": "component",
        },
        {
            "id": "drm_without_rescue_rescue_mmr", **common,
            "eta_rescue": 0.0, "axis": "component",
        },
        {
            "id": "drm_base_utility_only", **common, "alpha_diff": 0.0,
            "lambda_red": 0.0, "eta_rescue": 0.0, "eta_active": 0.0,
            "axis": "component",
        },
        {
            "id": "drm_swap_protected_core_fixed_k", "reuse": "drm_swap_original_k",
            "role": "add_vs_swap_and_protected_core", **common,
            "policy": "budget_neutral_drm_swap",
        },
    ]


def variant_specs(stage: int) -> List[Dict[str, Any]]:
    return _cc_specs() if stage == 9 else _drm_specs()


def _config_hash(spec: Mapping[str, Any]) -> str:
    payload = {
        "adapter_id": ADAPTER_ID,
        "adapter_sha256": sha256_file(REPO_ROOT / "tools/qwen_ablation_adapter.py"),
        "frozen_runtime_revision": FROZEN_IMPLEMENTATION_REVISION,
        "protocol_id": PROTOCOL_ID,
        "spec": dict(spec),
    }
    return sha256_bytes(canonical_bytes(payload))


def _canonical_prediction_path(pipeline_dir: Path, method: str, task: str) -> Path:
    if method == "cc_sliminfer_v1":
        stage2 = load_json(stage_decision_path(pipeline_dir, 2))
        return Path(stage2["run_root"]) / "tasks" / task / "run/longbench_predictions" / f"{task}.jsonl"
    return QWEN_DRM_ROOT / RUN_DIR_BY_METHOD[method] / "longbench_predictions" / f"{task}.jsonl"


def _read_predictions(path: Path) -> List[Dict[str, Any]]:
    return list(iter_jsonl(path)) if path.exists() else []


def _base_command(
    spec: Mapping[str, Any], task: str, task_root: Path, config_path: Path,
    config_hash: str, sample_indices: Optional[Sequence[int]] = None,
) -> List[str]:
    family = str(spec["family"])
    runtime_method = "cc_sliminfer_v1" if family == "cc" else str(spec["runtime_method"])
    policy = "none" if family == "cc" else str(spec["policy"])
    score = "delta_o" if family == "cc" else str(spec["score"])
    command = [
        str(PYTHON), str(REPO_ROOT / "tools/qwen_ablation_adapter.py"),
        "--ablation-config", str(config_path),
        "--model_path", str(QWEN_MODEL_ROOT), "--tokenizer_path", str(QWEN_MODEL_ROOT),
        "--model_family", "qwen2", "--dtype", "fp16", "--device", "cuda",
        "--backend", "flash_attention_2", "--longbench_dir", str(DATA_ROOT),
        "--longbench_tasks", task, "--longbench_max_samples", "0",
        "--max_model_len", "131072", "--method", runtime_method, "--eval", "longbench",
        "--block_strategy", "fixed", "--selection_path", "canonical_fixed_page",
        "--block_size", "64", "--token_unit_size", "8", "--local_query_window", "4",
        "--swap_threshold", "0.9", "--prune_layers", "9,18,26",
        "--retain_tokens", "12288,6144,4096", "--decode_policy", "original",
        "--rescue_policy", policy, "--rescue_score", score,
        "--rescue_ratio", str(spec.get("rescue_ratio", 0.05)),
        "--alpha_diff", str(spec.get("alpha_diff", 0.5)),
        "--lambda_red", str(spec.get("lambda_red", 0.25)),
        "--eta_rescue", str(spec.get("eta_rescue", 0.25)),
        "--eta_active", str(spec.get("eta_active", 0.25)),
        "--temperature", "0.0", "--top_p", "1.0", "--generation_max_new_tokens", "0",
        "--seed", str(spec.get("seed", 0)), "--min_free_gpu_gb", "0",
        "--protocol_id", PROTOCOL_ID, "--implementation_revision", FROZEN_IMPLEMENTATION_REVISION,
        "--source_config_hash", config_hash, "--output_dir", str(task_root / "run"),
    ]
    if sample_indices is not None:
        values = ",".join(str(int(value)) for value in sample_indices)
        command.extend(["--longbench_sample_indices_by_task", f"{task}:{values}"])
    if family == "cc":
        command.extend([
            "--active_block_trace", "full",
            "--active_block_trace_dir", str(task_root / "active_block_trace"),
        ])
    else:
        command.extend([
            "--rescue_audit_path", str(task_root / "audits/rescue_policy_audit.jsonl"),
            "--drm_trace_dir", str(task_root / "drm_trace"),
        ])
    return command


def _identity_reference(pipeline_dir: Path, family: str, task: str) -> Dict[int, Dict[str, Any]]:
    method = "cc_sliminfer_v1" if family == "cc" else "sliminfer_original"
    return {
        int(row["sample_index"]): row
        for row in _read_predictions(_canonical_prediction_path(pipeline_dir, method, task))
    }


def _validate_cc_trace(path: Path, task: str, expected_indices: Sequence[int]) -> Dict[str, Any]:
    report = {
        "path": str(path), "exists": path.exists(), "events": 0,
        "samples": 0, "budget_bad": 0, "sink_bad": 0, "duplicate_bad": 0,
        "invalid_bad": 0, "set_accounting_bad": 0, "identity_bad": 0,
    }
    if not path.exists():
        return report
    samples = set()
    expected = set(int(value) for value in expected_indices)
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            event = json.loads(line)
            report["events"] += 1
            index = int(event.get("sample_index", -1))
            samples.add(index)
            selected = list(event.get("cc_active_ids", []))
            selected_set = set(selected)
            baseline = set(event.get("base_original_active_ids", []))
            candidates = set(event.get("candidate_ids", []))
            added = set(event.get("added_ids", []))
            removed = set(event.get("removed_ids", []))
            layer = _trace_layer(event)
            if (
                int(event.get("budget_parity", 0)) != 1
                or int(event.get("over_budget", 0)) != 0
                or len(selected) != int(event.get("selected_count", -1))
                or (layer in EXPECTED_K and len(selected) != EXPECTED_K[layer])
            ):
                report["budget_bad"] += 1
            if int(event.get("sink_violation", 0)) or not set(event.get("sink_ids", [])).issubset(selected_set):
                report["sink_bad"] += 1
            if len(selected) != len(selected_set) or int(event.get("duplicate_violation", 0)):
                report["duplicate_bad"] += 1
            if any(not isinstance(value, int) or value < 0 or value not in candidates for value in selected):
                report["invalid_bad"] += 1
            if added != selected_set - baseline or removed != baseline - selected_set or len(added) != len(removed):
                report["set_accounting_bad"] += 1
            if (
                event.get("protocol_id") != PROTOCOL_ID
                or event.get("implementation_revision") != FROZEN_IMPLEMENTATION_REVISION
                or event.get("task") != task
                or index not in expected
            ):
                report["identity_bad"] += 1
    report["samples"] = len(samples & expected)
    return report


def _validate_drm_audit(path: Path, spec: Mapping[str, Any]) -> Dict[str, Any]:
    report = {
        "path": str(path), "exists": path.exists(), "events": 0, "prefill_events": 0,
        "budget_bad": 0, "duplicate_bad": 0, "protected_bad": 0,
        "add_preservation_bad": 0, "swap_pairing_bad": 0,
    }
    if not path.exists():
        return report
    policy = str(spec["policy"])
    for event in iter_jsonl(path):
        report["events"] += 1
        if event.get("phase") != "prefill":
            continue
        report["prefill_events"] += 1
        active = int(event.get("active_before_count", -1))
        final = int(event.get("final_block_count_used_by_attention", event.get("final_block_count", -1)))
        rescued = int(event.get("rescued_count", -1))
        swapped = int(event.get("swapped_count", -1))
        if bool(event.get("duplicate_block_ids")):
            report["duplicate_bad"] += 1
        if bool(event.get("protected_block_violation")):
            report["protected_bad"] += 1
        if policy == "add":
            if final != active + rescued or swapped != 0:
                report["budget_bad"] += 1
                report["add_preservation_bad"] += 1
        else:
            if final != active or rescued != swapped:
                report["budget_bad"] += 1
                report["swap_pairing_bad"] += 1
    return report


def validate_task(
    pipeline_dir: Path, stage: int, spec: Mapping[str, Any], task: str,
    task_root: Path, config_hash: str, sample_indices: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    indices = list(sample_indices) if sample_indices is not None else list(range(COUNTS[task]))
    path = task_root / "run/longbench_predictions" / f"{task}.jsonl"
    rows = _read_predictions(path)
    family = str(spec["family"])
    runtime_method = "cc_sliminfer_v1" if family == "cc" else str(spec["runtime_method"])
    references = _identity_reference(pipeline_dir, family, task)
    identity_fields = (
        "sample_id", "content_sha256", "prompt_sha256", "input_token_ids_sha256",
        "input_tokens", "max_new_tokens", "generation_config", "generation_config_sha256",
        "runtime_dtype", "attention_implementation", "tokenizer_id",
    )
    identity_bad: List[Dict[str, Any]] = []
    statuses: Dict[str, int] = defaultdict(int)
    for position, row in enumerate(rows):
        status = prediction_status(row)
        statuses[status] += 1
        if position >= len(indices):
            identity_bad.append({"position": position, "reason": "extra_row"})
            continue
        index = int(indices[position])
        ref = references.get(index)
        if ref is None:
            identity_bad.append({"position": position, "reason": "missing_reference"})
            continue
        expected = {
            "sample_index": index, "protocol_id": PROTOCOL_ID,
            "implementation_revision": FROZEN_IMPLEMENTATION_REVISION,
            "source_config_hash": config_hash, "method": runtime_method,
            **{field: ref.get(field) for field in identity_fields},
        }
        for field, value in expected.items():
            if row.get(field) != value:
                identity_bad.append({
                    "position": position, "field": field,
                    "expected": value, "actual": row.get(field),
                })
    invalid_status = {key: value for key, value in statuses.items() if key not in {"ok", "oom"}}
    if family == "cc":
        selection = _validate_cc_trace(
            task_root / "active_block_trace" / f"{task}.jsonl.gz", task, indices,
        )
        selection_bad = sum(int(selection[key]) for key in (
            "budget_bad", "sink_bad", "duplicate_bad", "invalid_bad",
            "set_accounting_bad", "identity_bad",
        ))
        selection_complete = selection["exists"] and selection_bad == 0
    else:
        selection = _validate_drm_audit(task_root / "audits/rescue_policy_audit.jsonl", spec)
        selection_bad = sum(int(selection[key]) for key in (
            "budget_bad", "duplicate_bad", "protected_bad",
            "add_preservation_bad", "swap_pairing_bad",
        ))
        selection_complete = selection["exists"] and selection["prefill_events"] > 0 and selection_bad == 0
    complete = (
        len(rows) == len(indices) and not identity_bad and not invalid_status and selection_complete
    )
    report = {
        "created_at": now_iso(), "stage": stage, "variant": spec["id"], "task": task,
        "decision": "PASS" if complete else "FAIL", "expected": len(indices),
        "predictions": len(rows), "statuses": dict(statuses), "identity_failures": identity_bad,
        "invalid_statuses": invalid_status, "selection_audit": selection,
        "score": per_task_score(task, rows) if complete else None,
        "prediction_path": str(path), "config_hash": config_hash,
    }
    return report


def _prediction_path(task_root: Path, task: str) -> Path:
    return task_root / "run/longbench_predictions" / f"{task}.jsonl"


def _resume_artifact_paths(task_root: Path, task: str) -> List[Path]:
    paths = [_prediction_path(task_root, task)]
    paths.extend(sorted((task_root / "audits").glob("*.jsonl")))
    paths.extend(sorted((task_root / "drm_trace").glob("*.jsonl")))
    paths.extend(sorted((task_root / "active_block_trace").glob("*")))
    return [path for path in paths if path.is_file()]


def _command_value(command: Sequence[str], option: str) -> Optional[str]:
    try:
        return str(command[command.index(option) + 1])
    except (ValueError, IndexError):
        return None


def _partial_resume_preflight(
    pipeline_dir: Path,
    stage: int,
    spec: Mapping[str, Any],
    task: str,
    task_root: Path,
    config_hash: str,
    command: Sequence[str],
    sample_indices: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    indices = list(sample_indices) if sample_indices is not None else list(range(COUNTS[task]))
    path = _prediction_path(task_root, task)
    rows = _read_predictions(path)
    family = str(spec["family"])
    runtime_method = "cc_sliminfer_v1" if family == "cc" else str(spec["runtime_method"])
    references = _identity_reference(pipeline_dir, family, task)
    identity_fields = (
        "sample_id", "content_sha256", "prompt_sha256", "input_token_ids_sha256",
        "input_tokens", "max_new_tokens", "generation_config", "generation_config_sha256",
        "runtime_dtype", "attention_implementation", "tokenizer_id",
    )
    failures: List[Dict[str, Any]] = []
    if not rows:
        failures.append({"check": "prediction_prefix_nonempty", "actual": 0})
    if len(rows) >= len(indices):
        failures.append({
            "check": "prediction_prefix_partial",
            "actual": len(rows),
            "expected_less_than": len(indices),
        })
    observed_indices = [row.get("sample_index") for row in rows]
    expected_indices = indices[:len(rows)]
    if observed_indices != expected_indices:
        failures.append({
            "check": "contiguous_sample_index_prefix",
            "expected": expected_indices,
            "actual": observed_indices,
        })
    observed_ids = [row.get("sample_id") for row in rows]
    if len(observed_ids) != len(set(observed_ids)):
        failures.append({"check": "duplicate_sample_ids", "actual": observed_ids})
    for position, row in enumerate(rows):
        if position >= len(expected_indices):
            break
        index = int(expected_indices[position])
        reference = references.get(index)
        if reference is None:
            failures.append({
                "check": "canonical_reference",
                "position": position,
                "sample_index": index,
            })
            continue
        expected = {
            "sample_index": index,
            "protocol_id": PROTOCOL_ID,
            "implementation_revision": FROZEN_IMPLEMENTATION_REVISION,
            "source_config_hash": config_hash,
            "method": runtime_method,
            **{field: reference.get(field) for field in identity_fields},
        }
        for field, value in expected.items():
            if row.get(field) != value:
                failures.append({
                    "check": "prediction_identity",
                    "position": position,
                    "field": field,
                    "expected": value,
                    "actual": row.get(field),
                })
    statuses = [prediction_status(row) for row in rows]
    invalid_statuses = sorted(set(statuses) - {"ok", "oom"})
    if invalid_statuses:
        failures.append({"check": "prediction_status", "actual": invalid_statuses})

    expected_seed = int(spec.get("seed", 0))
    config = load_json(task_root.parent.parent / "ABLATION_CONFIG.json")
    if int(config.get("seed", -1)) != expected_seed:
        failures.append({
            "check": "config_seed",
            "expected": expected_seed,
            "actual": config.get("seed"),
        })
    command_checks = {
        "--seed": str(expected_seed),
        "--source_config_hash": config_hash,
        "--rescue_score": str(spec.get("score", "delta_o")),
        "--rescue_policy": (
            "none" if family == "cc" else str(spec.get("policy", "none"))
        ),
    }
    for option, expected in command_checks.items():
        actual = _command_value(command, option)
        if actual != expected:
            failures.append({
                "check": "command_wiring",
                "option": option,
                "expected": expected,
                "actual": actual,
            })

    seed_evidence = 0
    seed_mismatches: List[Dict[str, Any]] = []
    if family == "drm":
        for artifact in _resume_artifact_paths(task_root, task):
            if artifact == path or artifact.suffix == ".gz":
                continue
            for event in iter_jsonl(artifact):
                if "seed" not in event:
                    continue
                seed_evidence += 1
                if int(event["seed"]) != expected_seed:
                    seed_mismatches.append({
                        "path": str(artifact),
                        "sample_index": event.get("sample_index"),
                        "actual": event.get("seed"),
                    })
    if family == "drm" and not seed_evidence:
        failures.append({"check": "trace_seed_evidence", "actual": 0})
    if seed_mismatches:
        failures.append({
            "check": "trace_seed_wiring",
            "expected": expected_seed,
            "mismatches": seed_mismatches[:20],
            "total_mismatches": len(seed_mismatches),
        })
    artifacts = {
        str(artifact.relative_to(task_root)): {
            "bytes": artifact.stat().st_size,
            "sha256": sha256_file(artifact),
        }
        for artifact in _resume_artifact_paths(task_root, task)
    }
    return {
        "created_at": now_iso(),
        "stage": stage,
        "variant": spec["id"],
        "task": task,
        "decision": "PASS" if not failures else "FAIL",
        "expected_total": len(indices),
        "preserved_prediction_count": len(rows),
        "resume_from_position": len(rows),
        "resume_from_sample_index": (
            int(indices[len(rows)]) if len(rows) < len(indices) else None
        ),
        "sample_indices": observed_indices,
        "sample_ids": observed_ids,
        "statuses": dict(
            (status, statuses.count(status)) for status in sorted(set(statuses))
        ),
        "expected_seed": expected_seed,
        "trace_seed_evidence": seed_evidence,
        "config_hash": config_hash,
        "artifacts": artifacts,
        "failures": failures,
    }


def _snapshot_partial_artifacts(
    root: Path, spec: Mapping[str, Any], task: str, task_root: Path
) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_root = (
        root / "resume_snapshots" / str(spec["id"]) / task / stamp
    )
    manifest: Dict[str, Any] = {
        "created_at": now_iso(),
        "variant": spec["id"],
        "task": task,
        "source": str(task_root),
        "files": {},
    }
    for source in _resume_artifact_paths(task_root, task):
        relative = source.relative_to(task_root)
        target = snapshot_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        manifest["files"][str(relative)] = {
            "bytes": source.stat().st_size,
            "sha256": sha256_file(source),
            "snapshot": str(target),
        }
    atomic_write_json(snapshot_root / "SNAPSHOT_MANIFEST.json", manifest)
    return snapshot_root


def _restore_truncated_audit_prefix(task_root: Path, snapshot_root: Path) -> Dict[str, Any]:
    relative = Path("audits/rescue_policy_audit.jsonl")
    prefix_path = snapshot_root / relative
    current_path = task_root / relative
    if not prefix_path.exists():
        return {"required": False, "merged": False}
    prefix = prefix_path.read_bytes()
    suffix = current_path.read_bytes() if current_path.exists() else b""
    if suffix.startswith(prefix):
        return {
            "required": True,
            "merged": False,
            "reason": "runtime_preserved_prefix",
            "prefix_bytes": len(prefix),
            "suffix_bytes": max(0, len(suffix) - len(prefix)),
        }
    separator = b"" if not prefix or prefix.endswith(b"\n") or not suffix else b"\n"
    current_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = current_path.with_suffix(current_path.suffix + ".resume-merge.tmp")
    with temporary.open("wb") as handle:
        handle.write(prefix)
        handle.write(separator)
        handle.write(suffix)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, current_path)
    return {
        "required": True,
        "merged": True,
        "prefix_bytes": len(prefix),
        "suffix_bytes": len(suffix),
        "merged_bytes": current_path.stat().st_size,
    }


def _partial_resume_postcheck(
    task_root: Path,
    snapshot_root: Path,
    task: str,
    expected_seed: int,
    expected_indices: Sequence[int],
) -> Dict[str, Any]:
    failures: List[Dict[str, Any]] = []
    artifacts: Dict[str, Any] = {}
    manifest = load_json(snapshot_root / "SNAPSHOT_MANIFEST.json")
    for relative_text, frozen in manifest["files"].items():
        relative = Path(relative_text)
        frozen_path = Path(frozen["snapshot"])
        current_path = task_root / relative
        prefix = frozen_path.read_bytes()
        current = current_path.read_bytes() if current_path.exists() else b""
        preserved = current.startswith(prefix)
        artifacts[relative_text] = {
            "frozen_bytes": len(prefix),
            "current_bytes": len(current),
            "frozen_sha256": frozen["sha256"],
            "prefix_preserved": preserved,
        }
        if not preserved:
            failures.append({
                "check": "artifact_prefix_preserved",
                "path": str(current_path),
            })

    prediction_rows = _read_predictions(_prediction_path(task_root, task))
    sample_indices = [row.get("sample_index") for row in prediction_rows]
    expected_prefix = list(expected_indices[:len(prediction_rows)])
    if sample_indices != expected_prefix:
        failures.append({
            "check": "prediction_indices_after_resume",
            "expected": expected_prefix,
            "actual": sample_indices,
        })
    sample_ids = [row.get("sample_id") for row in prediction_rows]
    if len(sample_ids) != len(set(sample_ids)):
        failures.append({"check": "duplicate_sample_ids_after_resume"})

    seed_evidence = 0
    seed_mismatches = 0
    for artifact in _resume_artifact_paths(task_root, task):
        if artifact == _prediction_path(task_root, task) or artifact.suffix == ".gz":
            continue
        for event in iter_jsonl(artifact):
            if "seed" not in event:
                continue
            seed_evidence += 1
            if int(event["seed"]) != int(expected_seed):
                seed_mismatches += 1
    if seed_evidence == 0 or seed_mismatches:
        failures.append({
            "check": "seed_wiring_after_resume",
            "expected_seed": expected_seed,
            "evidence": seed_evidence,
            "mismatches": seed_mismatches,
        })
    return {
        "created_at": now_iso(),
        "decision": "PASS" if not failures else "FAIL",
        "prediction_count": len(prediction_rows),
        "sample_indices": sample_indices,
        "unique_sample_ids": len(set(sample_ids)),
        "expected_seed": expected_seed,
        "trace_seed_evidence": seed_evidence,
        "seed_mismatches": seed_mismatches,
        "artifacts": artifacts,
        "failures": failures,
    }


def _archive_partial(path: Path, label: str) -> None:
    if not path.exists():
        return
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path.rename(path.with_name(f"{path.name}.incomplete_{label}_{stamp}"))


def _transient_post_worker_utilization(gate: Mapping[str, Any]) -> bool:
    failures = [
        failure
        for check in gate.get("checks", [])
        for failure in check.get("failures", [])
    ]
    if not failures or {row.get("check") for row in failures} != {"idle_gpu_utilization_percent"}:
        return False
    snapshots = list(gate.get("snapshots", []))
    return bool(snapshots) and all(
        not [
            row for row in snapshot.get("gpu_compute_processes", [])
            if not row.get("desktop_allowlisted")
        ]
        and not snapshot.get("experiment_processes", [])
        for snapshot in snapshots
    )


def _worker_resource_gate(
    pipeline_dir: Path, policy: Mapping[str, Any], allowed_pids: Sequence[int],
) -> Dict[str, Any]:
    attempts = []
    for gate_attempt in (1, 2):
        gate = monitor_resources(
            policy,
            int(policy["thresholds"]["resource_gate_duration_seconds"]),
            int(policy["thresholds"]["resource_gate_interval_seconds"]),
            pipeline_dir=pipeline_dir,
            allowed_pids=allowed_pids,
        )
        attempts.append(gate)
        if gate["pass"]:
            return {**gate, "cooldown_attempts": attempts}
        if gate_attempt == 1 and _transient_post_worker_utilization(gate):
            time.sleep(30)
            continue
        return {**gate, "cooldown_attempts": attempts}
    raise AssertionError("unreachable")


def _post_worker_cooldown(
    pipeline_dir: Path,
    policy: Mapping[str, Any],
    *,
    worker_pid: int,
    pre_worker_swap_bytes: int,
) -> Dict[str, Any]:
    thresholds = policy["thresholds"]
    duration = int(thresholds.get("post_worker_cooldown_seconds", 30))
    interval = max(1, int(thresholds.get("resource_gate_interval_seconds", 5)))
    snapshots: List[Dict[str, Any]] = []
    deadline = time.monotonic() + duration
    while True:
        snapshot = collect_snapshot(pipeline_dir)
        snapshots.append(snapshot)
        if time.monotonic() >= deadline:
            break
        time.sleep(min(interval, max(0.0, deadline - time.monotonic())))
    swap_values = [
        int(snapshot["host_memory"]["swap_used_bytes"]) for snapshot in snapshots
    ]
    hard_failures: List[Dict[str, Any]] = []
    final = snapshots[-1]
    if int(final["host_memory"]["ram_available_bytes"]) < int(
        thresholds["min_host_available_bytes"]
    ):
        hard_failures.append({
            "check": "host_available_bytes_after_worker",
            "value": final["host_memory"]["ram_available_bytes"],
            "threshold": thresholds["min_host_available_bytes"],
        })
    if final["recent_driver_errors"].get("matches"):
        hard_failures.append({
            "check": "recent_driver_errors_after_worker",
            "value": final["recent_driver_errors"]["matches"],
        })
    worker_still_present = any(
        int(row.get("pid", -1)) == int(worker_pid)
        for row in final.get("gpu_compute_processes", [])
    ) or Path(f"/proc/{int(worker_pid)}").exists()
    if worker_still_present:
        hard_failures.append({
            "check": "worker_still_present_after_cooldown",
            "value": worker_pid,
        })
    return {
        "created_at": now_iso(),
        "decision": "PASS" if not hard_failures else "FAIL",
        "duration_seconds": duration,
        "interval_seconds": interval,
        "worker_pid": worker_pid,
        "pre_worker_swap_bytes": int(pre_worker_swap_bytes),
        "swap_start_bytes": swap_values[0],
        "swap_peak_bytes": max(swap_values),
        "swap_final_bytes": swap_values[-1],
        "swap_final_delta_from_pre_worker_bytes": (
            swap_values[-1] - int(pre_worker_swap_bytes)
        ),
        "swap_recovered_during_cooldown_bytes": max(swap_values) - swap_values[-1],
        "snapshots": snapshots,
        "hard_failures": hard_failures,
    }


def _next_attempt_number(root: Path, spec: Mapping[str, Any], task: str) -> int:
    command_root = root / "commands" / str(spec["id"])
    attempts: List[int] = []
    for path in command_root.glob(f"{task}.attempt-*.json"):
        try:
            attempts.append(int(path.stem.rsplit("-", 1)[1]))
        except (IndexError, ValueError):
            continue
    return max(attempts, default=0) + 1


def _set_pid(pipeline_dir: Path, *, stage: int, method: str, task: str, worker: int) -> None:
    path = pipeline_dir / "PID.json"
    state = load_json(path) if path.exists() else {}
    state.update({
        "updated_at": now_iso(), "task_worker_pid": os.getpid(), "gpu_worker_pid": worker,
        "current_stage": stage, "current_method": method, "current_task": task,
    })
    atomic_write_json(path, state)


def _clear_pid(pipeline_dir: Path) -> None:
    path = pipeline_dir / "PID.json"
    state = load_json(path) if path.exists() else {}
    state.update({
        "updated_at": now_iso(), "task_worker_pid": None, "gpu_worker_pid": None,
        "current_method": None, "current_task": None,
    })
    atomic_write_json(path, state)


def _run_worker(
    *, pipeline_dir: Path, root: Path, stage: int, spec: Mapping[str, Any],
    task: str, task_root: Path, config_path: Path, config_hash: str,
    policy: Mapping[str, Any], stage_swap_baseline: int,
    sample_indices: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    task_root.parent.mkdir(parents=True, exist_ok=True)
    indices = list(sample_indices) if sample_indices is not None else list(range(COUNTS[task]))
    prediction_path = _prediction_path(task_root, task)
    resume_mode = False
    resume_snapshot: Optional[Path] = None
    first_attempt = _next_attempt_number(root, spec, task)
    command = _base_command(spec, task, task_root, config_path, config_hash, sample_indices)
    if task_root.exists() and prediction_path.exists():
        existing_rows = _read_predictions(prediction_path)
        if len(existing_rows) == len(indices):
            report = validate_task(
                pipeline_dir, stage, spec, task, task_root, config_hash,
                sample_indices=sample_indices,
            )
            if report["decision"] == "PASS":
                report.update({"attempt": first_attempt - 1, "recovered_without_worker": True})
                atomic_write_json(task_root / "TASK_DECISION.json", report)
                return report
            return {
                "decision": "HARD_BLOCKED",
                "reason": "complete_prediction_set_failed_validation",
                "details": report,
            }
        if existing_rows:
            preflight = _partial_resume_preflight(
                pipeline_dir, stage, spec, task, task_root, config_hash,
                command, sample_indices=sample_indices,
            )
            atomic_write_json(task_root / "PARTIAL_RESUME_PREFLIGHT.json", preflight)
            if preflight["decision"] != "PASS":
                return {
                    "decision": "HARD_BLOCKED",
                    "reason": "partial_resume_preflight_failed",
                    "details": preflight,
                }
            resume_snapshot = _snapshot_partial_artifacts(root, spec, task, task_root)
            preflight["snapshot_root"] = str(resume_snapshot)
            atomic_write_json(task_root / "PARTIAL_RESUME_PREFLIGHT.json", preflight)
            resume_mode = True

    attempts = (first_attempt,) if resume_mode else (first_attempt, first_attempt + 1)
    for attempt in attempts:
        if task_root.exists() and not resume_mode:
            _archive_partial(task_root, f"attempt{attempt}")
        command = _base_command(spec, task, task_root, config_path, config_hash, sample_indices)
        command_path = root / "commands" / str(spec["id"]) / f"{task}.attempt-{attempt:02d}.json"
        atomic_write_json(command_path, {
            "created_at": now_iso(),
            "command": command,
            "resume_mode": resume_mode,
            "resume_from_position": len(_read_predictions(prediction_path)) if resume_mode else 0,
            "partial_snapshot": str(resume_snapshot) if resume_snapshot else None,
        })
        gate = _worker_resource_gate(pipeline_dir, policy, (os.getpid(),))
        atomic_write_json(
            root / "resource_gates" / str(spec["id"]) / f"{task}.attempt-{attempt:02d}.json",
            gate,
        )
        if not gate["pass"]:
            write_blocker(pipeline_dir, gate)
            return {"decision": "HARD_BLOCKED", "reason": "resource_gate_failed", "details": gate}
        pre_worker_swap = int(gate["snapshots"][-1]["host_memory"]["swap_used_bytes"])
        atomic_write_json(
            root / "resource_gates" / str(spec["id"])
            / f"{task}.attempt-{attempt:02d}.worker-swap-baseline.json",
            {
                "created_at": now_iso(),
                "stage_swap_baseline_bytes": int(stage_swap_baseline),
                "pre_worker_stable_swap_bytes": pre_worker_swap,
                "preexisting_stage_swap_delta_bytes": (
                    pre_worker_swap - int(stage_swap_baseline)
                ),
                "watchdog_primary_metrics": [
                    "worker_vm_swap_delta",
                    "cgroup_memory_swap_current_delta",
                ],
                "watchdog_auxiliary_metric": "system_swap_delta",
            },
        )
        task_root.mkdir(parents=True, exist_ok=resume_mode)
        log_path = root / "logs" / str(spec["id"]) / f"{task}.attempt-{attempt:02d}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.update({key: str(value) for key, value in policy["environment"].items() if key.isupper()})
        with log_path.open("a", encoding="utf-8") as log:
            proc = subprocess.Popen(command, cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            _set_pid(
                pipeline_dir, stage=stage, method=str(spec["id"]), task=task, worker=proc.pid,
            )
            trace_path = (
                task_root / "active_block_trace" / f"{task}.jsonl.gz.tmp"
                if spec["family"] == "cc" else task_root / "audits/rescue_policy_audit.jsonl"
            )
            watchdog = RuntimeWatchdog(
                pipeline_dir=pipeline_dir, run_root=root, policy=policy, stage=stage,
                method=str(spec["id"]), task=task, worker_pid=proc.pid,
                prediction_path=prediction_path, trace_path=trace_path,
                log_path=log_path,
                resume_command=(pipeline_dir / "RESUME_COMMAND.txt").read_text(encoding="utf-8").strip(),
                interval_seconds=30, swap_baseline_bytes=pre_worker_swap,
            )
            watchdog.start()
            returncode = proc.wait()
            watchdog_failure = watchdog.stop()
        _clear_pid(pipeline_dir)
        merge_report: Optional[Dict[str, Any]] = None
        postcheck: Optional[Dict[str, Any]] = None
        if resume_mode:
            assert resume_snapshot is not None
            merge_report = _restore_truncated_audit_prefix(task_root, resume_snapshot)
            postcheck = _partial_resume_postcheck(
                task_root, resume_snapshot, task, int(spec.get("seed", 0)), indices,
            )
            postcheck["audit_prefix_merge"] = merge_report
            atomic_write_json(task_root / "PARTIAL_RESUME_POSTCHECK.json", postcheck)
            if postcheck["decision"] != "PASS":
                return {
                    "decision": "HARD_BLOCKED",
                    "reason": "partial_resume_postcheck_failed",
                    "details": postcheck,
                }
        cooldown = _post_worker_cooldown(
            pipeline_dir,
            policy,
            worker_pid=proc.pid,
            pre_worker_swap_bytes=pre_worker_swap,
        )
        atomic_write_json(
            root / "resource_gates" / str(spec["id"])
            / f"{task}.attempt-{attempt:02d}.post-worker-cooldown.json",
            cooldown,
        )
        if cooldown["decision"] != "PASS":
            return {
                "decision": "HARD_BLOCKED",
                "reason": "post_worker_cooldown_failed",
                "details": cooldown,
            }
        if watchdog_failure:
            return {
                "decision": "HARD_BLOCKED", "reason": "runtime_watchdog_failure",
                "details": watchdog_failure, "partial_resume_postcheck": postcheck,
            }
        if returncode == 0:
            report = validate_task(
                pipeline_dir, stage, spec, task, task_root, config_hash,
                sample_indices=sample_indices,
            )
            report.update({
                "attempt": attempt,
                "log": str(log_path),
                "resume_mode": resume_mode,
                "partial_resume_postcheck": postcheck,
                "post_worker_cooldown": str(
                    root / "resource_gates" / str(spec["id"])
                    / f"{task}.attempt-{attempt:02d}.post-worker-cooldown.json"
                ),
            })
            if report["decision"] == "PASS":
                atomic_write_json(task_root / "TASK_DECISION.json", report)
                return report
            atomic_write_json(
                task_root / f"TASK_VALIDATION_FAILURE.attempt-{attempt:02d}.json",
                report,
            )
        else:
            report = {
                "decision": "FAILED", "returncode": returncode,
                "attempt": attempt, "log": str(log_path), "resume_mode": resume_mode,
            }
        if resume_mode or attempt == attempts[-1]:
            return report
    raise AssertionError("unreachable")


def _canonical_trace_events(pipeline_dir: Path, task: str, indices: Sequence[int]) -> Dict[Tuple[Any, ...], Dict[str, Any]]:
    stage2 = load_json(stage_decision_path(pipeline_dir, 2))
    path = Path(stage2["run_root"]) / "tasks" / task / "active_block_trace" / f"{task}.jsonl.gz"
    wanted = set(int(value) for value in indices)
    result: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            event = json.loads(line)
            if int(event.get("sample_index", -1)) not in wanted:
                continue
            key = (
                int(event["sample_index"]), event.get("phase"), int(event.get("decode_step", 0)),
                int(event.get("selection_event_index", -1)),
                _trace_layer(event),
            )
            result[key] = event
    return result


def _current_trace_events(path: Path) -> Dict[Tuple[Any, ...], Dict[str, Any]]:
    result: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            event = json.loads(line)
            key = (
                int(event["sample_index"]), event.get("phase"), int(event.get("decode_step", 0)),
                int(event.get("selection_event_index", -1)),
                _trace_layer(event),
            )
            result[key] = event
    return result


def _canonical_drm_prefill(indices_by_task: Mapping[str, Sequence[int]]) -> Dict[Tuple[str, int, int], Dict[str, Any]]:
    wanted = {task: set(int(value) for value in values) for task, values in indices_by_task.items()}
    path = QWEN_DRM_ROOT / "audits/sliminfer_drm_add_km_rescue_policy_audit.jsonl"
    result: Dict[Tuple[str, int, int], Dict[str, Any]] = {}
    for event in iter_jsonl(path):
        task = str(event.get("task"))
        index = int(event.get("sample_index", -1))
        if event.get("phase") != "prefill" or index not in wanted.get(task, set()):
            continue
        result[(task, index, int(event.get("layer_id", -1)))] = event
    return result


def _run_canary(
    pipeline_dir: Path, root: Path, stage: int, policy: Mapping[str, Any],
    stage_swap_baseline: int,
) -> Dict[str, Any]:
    canary_root = root / "adapter_equivalence_canary"
    result_path = canary_root / "CANARY_DECISION.json"
    if result_path.exists() and load_json(result_path).get("decision") == "PASS":
        return load_json(result_path)
    plan = load_json(pipeline_dir / CANARY_PLAN)
    indices_by_task: Dict[str, List[int]] = defaultdict(list)
    for sample in plan["samples"]:
        indices_by_task[str(sample["task"])].append(int(sample["sample_index"]))
    spec = (
        {row["id"]: row for row in _cc_specs()}["cc_full"]
        if stage == 9 else {row["id"]: row for row in _drm_specs()}["drm_full_ratio_0_05"]
    )
    spec = {key: value for key, value in spec.items() if key != "reuse"}
    spec["id"] = f"stage_{stage:02d}_adapter_canary"
    config_hash = _config_hash(spec)
    config_path = canary_root / "ABLATION_CONFIG.json"
    atomic_write_json(config_path, {**spec, "config_hash": config_hash})
    task_reports = []
    for task in REPRESENTATIVE_TASKS:
        task_root = canary_root / "tasks" / task
        existing = task_root / "TASK_DECISION.json"
        if existing.exists() and load_json(existing).get("decision") == "PASS":
            report = load_json(existing)
        else:
            report = _run_worker(
                pipeline_dir=pipeline_dir, root=canary_root, stage=stage, spec=spec,
                task=task, task_root=task_root, config_path=config_path,
                config_hash=config_hash, policy=policy,
                stage_swap_baseline=stage_swap_baseline,
                sample_indices=indices_by_task[task],
            )
        task_reports.append(report)
        if report.get("decision") == "HARD_BLOCKED":
            atomic_write_json(result_path, report)
            return report
        if report.get("decision") != "PASS":
            failure = {
                "created_at": now_iso(), "stage": stage, "decision": "FAIL",
                "reason": "canary_task_failed", "task": task, "details": report,
            }
            atomic_write_json(result_path, failure)
            return failure

    prediction_mismatches: List[Dict[str, Any]] = []
    selection_mismatches: List[Dict[str, Any]] = []
    if stage == 9:
        for task, indices in indices_by_task.items():
            canonical_rows = {
                int(row["sample_index"]): row
                for row in _read_predictions(_canonical_prediction_path(pipeline_dir, "cc_sliminfer_v1", task))
            }
            current_rows = _read_predictions(
                canary_root / "tasks" / task / "run/longbench_predictions" / f"{task}.jsonl"
            )
            for row in current_rows:
                ref = canonical_rows[int(row["sample_index"])]
                if row.get("pred") != ref.get("pred"):
                    prediction_mismatches.append({"task": task, "sample_index": row["sample_index"]})
            expected_events = _canonical_trace_events(pipeline_dir, task, indices)
            current_events = _current_trace_events(
                canary_root / "tasks" / task / "active_block_trace" / f"{task}.jsonl.gz"
            )
            fields = (
                "base_original_active_ids", "cc_active_ids", "added_ids", "removed_ids",
                "selected_count", "sink_ids", "budget_parity", "sink_violation",
            )
            if set(expected_events) != set(current_events):
                selection_mismatches.append({
                    "task": task, "reason": "event_keys", "expected": len(expected_events),
                    "actual": len(current_events),
                })
            for key in set(expected_events) & set(current_events):
                bad = [field for field in fields if expected_events[key].get(field) != current_events[key].get(field)]
                if bad:
                    selection_mismatches.append({"task": task, "event": list(key), "fields": bad})
    else:
        canonical_events = _canonical_drm_prefill(indices_by_task)
        current_events: Dict[Tuple[str, int, int], Dict[str, Any]] = {}
        for task in REPRESENTATIVE_TASKS:
            canonical_rows = {
                int(row["sample_index"]): row
                for row in _read_predictions(_canonical_prediction_path(pipeline_dir, "sliminfer_drm_add_km", task))
            }
            rows = _read_predictions(
                canary_root / "tasks" / task / "run/longbench_predictions" / f"{task}.jsonl"
            )
            for row in rows:
                ref = canonical_rows[int(row["sample_index"])]
                if row.get("pred") != ref.get("pred"):
                    prediction_mismatches.append({"task": task, "sample_index": row["sample_index"]})
            for event in iter_jsonl(canary_root / "tasks" / task / "audits/rescue_policy_audit.jsonl"):
                if event.get("phase") == "prefill":
                    current_events[(task, int(event["sample_index"]), int(event["layer_id"]))] = event
        fields = (
            "K", "M", "active_before_count", "final_block_count_used_by_attention",
            "rescued_count", "swapped_count", "rescued_block_ids_first50", "dropped_ids_sampled",
        )
        if set(canonical_events) != set(current_events):
            selection_mismatches.append({
                "reason": "event_keys", "expected": len(canonical_events), "actual": len(current_events),
            })
        for key in set(canonical_events) & set(current_events):
            bad = [field for field in fields if canonical_events[key].get(field) != current_events[key].get(field)]
            if bad:
                selection_mismatches.append({"event": list(key), "fields": bad})

    result = {
        "created_at": now_iso(), "stage": stage,
        "decision": "PASS" if not prediction_mismatches and not selection_mismatches else "FAIL",
        "adapter_id": ADAPTER_ID,
        "adapter_sha256": sha256_file(REPO_ROOT / "tools/qwen_ablation_adapter.py"),
        "config_hash": config_hash, "samples": 12,
        "prediction_mismatches": prediction_mismatches,
        "selection_mismatches": selection_mismatches,
        "task_reports": task_reports,
    }
    atomic_write_json(result_path, result)
    return result


def _reuse_rows(pipeline_dir: Path, method: str, task: str) -> List[Dict[str, Any]]:
    return _read_predictions(_canonical_prediction_path(pipeline_dir, method, task))


def _variant_rows(
    pipeline_dir: Path, root: Path, spec: Mapping[str, Any], task: str,
) -> List[Dict[str, Any]]:
    if spec.get("reuse"):
        return _reuse_rows(pipeline_dir, str(spec["reuse"]), task)
    return _read_predictions(
        root / "variants" / str(spec["id"]) / "tasks" / task
        / "run/longbench_predictions" / f"{task}.jsonl"
    )


def _write_outputs(pipeline_dir: Path, root: Path, stage: int, specs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    sample_rows: List[Dict[str, Any]] = []
    task_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    for spec in specs:
        task_scores: Dict[str, float] = {}
        for task in REPRESENTATIVE_TASKS:
            rows = _variant_rows(pipeline_dir, root, spec, task)
            score = per_task_score(task, rows)
            task_scores[task] = score
            statuses: Dict[str, int] = defaultdict(int)
            for row in rows:
                status = prediction_status(row)
                statuses[status] += 1
                sample_rows.append({
                    "variant": spec["id"], "task": task,
                    "sample_id": row.get("sample_id"), "sample_index": row.get("sample_index"),
                    "status": status,
                    "score": per_sample_score(
                        task, str(row.get("pred", "")), row.get("answers", []), row.get("all_classes", []),
                    ),
                    "prediction_sha256": sha256_bytes(str(row.get("pred", "")).encode("utf-8")),
                    "source": str(spec.get("reuse", "fresh_ablation")),
                })
            task_rows.append({
                "variant": spec["id"], "task": task, "score": score,
                "samples": len(rows), "ok": statuses.get("ok", 0), "oom": statuses.get("oom", 0),
                "source": str(spec.get("reuse", "fresh_ablation")),
            })
        summary_rows.append({
            "variant": spec["id"],
            "macro_6task": sum(task_scores.values()) / len(REPRESENTATIVE_TASKS),
            "reuse": str(spec.get("reuse", "")), "axis": str(spec.get("axis", spec.get("role", ""))),
        })
    atomic_write_csv(root / "scores_by_sample.csv", sample_rows, [
        "variant", "task", "sample_id", "sample_index", "status", "score",
        "prediction_sha256", "source",
    ])
    atomic_write_csv(root / "scores_by_task.csv", task_rows, [
        "variant", "task", "score", "samples", "ok", "oom", "source",
    ])
    atomic_write_csv(root / "scores_summary.csv", summary_rows, [
        "variant", "macro_6task", "reuse", "axis",
    ])

    score_map = {(row["variant"], row["task"]): float(row["score"]) for row in task_rows}
    baseline = "sliminfer_original" if stage == 9 else "drm_full_ratio_0_05"
    delta_rows = []
    for spec in specs:
        if spec["id"] == baseline:
            continue
        deltas = []
        for task in REPRESENTATIVE_TASKS:
            delta = score_map[(spec["id"], task)] - score_map[(baseline, task)]
            deltas.append(delta)
            delta_rows.append({
                "variant": spec["id"], "baseline": baseline, "task": task, "delta": delta,
            })
        delta_rows.append({
            "variant": spec["id"], "baseline": baseline, "task": "6task_macro",
            "delta": sum(deltas) / len(deltas),
        })
    atomic_write_csv(root / "deltas_vs_preregistered_baseline.csv", delta_rows, [
        "variant", "baseline", "task", "delta",
    ])

    mechanism_root = root / "mechanism_analysis"
    mechanism_root.mkdir(parents=True, exist_ok=True)
    mechanism = _cc_analysis(pipeline_dir, mechanism_root) if stage == 9 else _drm_analysis(pipeline_dir, mechanism_root)
    atomic_write_json(mechanism_root / "MECHANISM_ANALYSIS_SUMMARY.json", mechanism)
    atomic_write_text(
        mechanism_root / "GOLD_EVIDENCE_ANALYSIS_BLOCKED.md",
        "# Gold Evidence Analysis Blocked\n\nThe official LongBench rows do not provide a common, "
        "verified supporting-fact annotation. No answer-derived pseudo-gold evidence recall is computed.\n",
    )
    return {
        "variants": len(specs), "fresh_variants": sum(not bool(spec.get("reuse")) for spec in specs),
        "tasks": len(REPRESENTATIVE_TASKS), "sample_score_rows": len(sample_rows),
        "mechanism": mechanism,
    }


def _write_manifest(root: Path, stage: int, specs: Sequence[Mapping[str, Any]]) -> None:
    adapter = REPO_ROOT / "tools/qwen_ablation_adapter.py"
    rows = []
    for order, spec in enumerate(specs, start=1):
        rows.append({
            "order": order, "variant": spec["id"], "family": "reuse" if spec.get("reuse") else spec["family"],
            "reuse": spec.get("reuse", ""), "axis": spec.get("axis", spec.get("role", "")),
            "config_hash": _config_hash(spec), "spec_json": json.dumps(dict(spec), sort_keys=True),
        })
    atomic_write_csv(root / "ABLATION_MANIFEST.csv", rows, [
        "order", "variant", "family", "reuse", "axis", "config_hash", "spec_json",
    ])
    atomic_write_json(root / "IMPLEMENTATION_IDENTITY.json", {
        "created_at": now_iso(), "stage": stage, "protocol_id": PROTOCOL_ID,
        "frozen_runtime_revision": implementation_revision(RUNTIME_FILES),
        "adapter_id": ADAPTER_ID, "adapter_path": str(adapter),
        "adapter_sha256": sha256_file(adapter),
        "injection_scope": "ablation worker process only",
        "frozen_runtime_files_modified": False,
    })


def run_stage(pipeline_dir: Path, stage: int, *, execute: bool, dry_run: bool) -> Dict[str, Any]:
    if stage not in {9, 10}:
        raise ValueError(stage)
    gate = run_protocol_gate(pipeline_dir, stage)
    if gate["decision"] != "PASS":
        raise RuntimeError(f"Stage {stage} protocol gate failed")
    if implementation_revision(RUNTIME_FILES) != FROZEN_IMPLEMENTATION_REVISION:
        raise RuntimeError("Frozen runtime revision changed")
    prefix = "qwen_cc_ablation_mechanism" if stage == 9 else "qwen_drm_ablation_mechanism"
    pointer = pipeline_dir / f"STAGE_{stage:02d}_MECHANISM_ROOT.json"
    if pointer.exists():
        root = Path(load_json(pointer)["path"])
    else:
        root = REPO_ROOT / "results" / f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        atomic_write_json(pointer, {"created_at": now_iso(), "path": str(root)})
    specs = variant_specs(stage)
    plan = {
        "created_at": now_iso(), "stage": stage, "protocol_id": PROTOCOL_ID,
        "root": str(root), "tasks": list(REPRESENTATIVE_TASKS),
        "variants": specs, "adapter_id": ADAPTER_ID,
        "adapter_sha256": sha256_file(REPO_ROOT / "tools/qwen_ablation_adapter.py"),
        "main_runtime_unchanged": True, "canary_required": True,
        "one_gpu_worker_at_a_time": True, "one_model_per_worker": True,
    }
    atomic_write_json(pipeline_dir / f"STAGE_{stage:02d}_PLAN.json", plan)
    if dry_run or not execute:
        decision = {
            "stage": stage, "name": prefix, "decision": "NOT_RUN", "run_root": str(root),
            "plan": plan,
        }
        atomic_write_json(stage_decision_path(pipeline_dir, stage), decision)
        return decision

    root.mkdir(parents=True, exist_ok=True)
    _write_manifest(root, stage, specs)
    policy = load_json(pipeline_dir / "SAFETY_POLICY.json")
    stage_gate = monitor_resources(
        policy,
        int(policy["thresholds"]["resource_gate_duration_seconds"]),
        int(policy["thresholds"]["resource_gate_interval_seconds"]),
        pipeline_dir=pipeline_dir,
        allowed_pids=(os.getpid(),),
    )
    atomic_write_json(root / "RESOURCE_GATE.json", stage_gate)
    if not stage_gate["pass"]:
        write_blocker(pipeline_dir, stage_gate)
        decision = {
            "stage": stage, "name": prefix, "created_at": now_iso(),
            "decision": "HARD_BLOCKED", "reason": "resource_gate_failed",
            "run_root": str(root), "details": stage_gate,
        }
        atomic_write_json(stage_decision_path(pipeline_dir, stage), decision)
        return decision
    swap_baseline = int(stage_gate["snapshots"][-1]["host_memory"]["swap_used_bytes"])
    atomic_write_json(root / "STAGE_SWAP_BASELINE.json", {
        "created_at": now_iso(), "swap_used_bytes": swap_baseline,
        "max_growth_bytes": int(policy["thresholds"]["max_swap_growth_bytes_during_gate"]),
    })

    canary = _run_canary(pipeline_dir, root, stage, policy, swap_baseline)
    atomic_write_json(pipeline_dir / f"STAGE_{stage:02d}_ABLATION_ADAPTER_EQUIVALENCE.json", canary)
    if canary.get("decision") != "PASS":
        decision = {
            "stage": stage, "name": prefix, "created_at": now_iso(),
            "decision": "HARD_BLOCKED", "reason": "ablation_adapter_equivalence_failed",
            "run_root": str(root), "canary": canary,
        }
        atomic_write_json(stage_decision_path(pipeline_dir, stage), decision)
        return decision

    fresh = [spec for spec in specs if not spec.get("reuse")]
    task_total = len(fresh) * len(REPRESENTATIVE_TASKS)
    task_done = 0
    failures: List[Dict[str, Any]] = []
    for spec in fresh:
        variant_root = root / "variants" / str(spec["id"])
        config_hash = _config_hash(spec)
        config_path = variant_root / "ABLATION_CONFIG.json"
        atomic_write_json(config_path, {**spec, "config_hash": config_hash})
        for task in REPRESENTATIVE_TASKS:
            task_root = variant_root / "tasks" / task
            decision_path = task_root / "TASK_DECISION.json"
            if decision_path.exists() and load_json(decision_path).get("decision") == "PASS":
                report = load_json(decision_path)
            else:
                report = _run_worker(
                    pipeline_dir=pipeline_dir, root=root, stage=stage, spec=spec,
                    task=task, task_root=task_root, config_path=config_path,
                    config_hash=config_hash, policy=policy,
                    stage_swap_baseline=swap_baseline,
                )
            task_done += 1
            atomic_write_json(pipeline_dir / "PROGRESS.json", {
                "updated_at": now_iso(), "stage": stage, "method": spec["id"], "task": task,
                "completed_ablation_tasks": task_done, "total_ablation_tasks": task_total,
            })
            if report.get("decision") == "HARD_BLOCKED":
                decision = {
                    "stage": stage, "name": prefix, "created_at": now_iso(),
                    "decision": "HARD_BLOCKED", "reason": report.get("reason"),
                    "run_root": str(root), "details": report,
                }
                atomic_write_json(stage_decision_path(pipeline_dir, stage), decision)
                return decision
            if report.get("decision") != "PASS":
                failures.append({"variant": spec["id"], "task": task, "details": report})

    atomic_write_json(root / "FAILED_TASKS.json", failures)
    if failures:
        decision = {
            "stage": stage, "name": prefix, "created_at": now_iso(),
            "decision": "SOFT_BLOCKED", "reason": "non_core_ablation_task_failures",
            "run_root": str(root), "failed_tasks": len(failures), "next_stage": stage + 1,
        }
        atomic_write_json(stage_decision_path(pipeline_dir, stage), decision)
        atomic_write_text(
            pipeline_dir / f"STAGE_{stage:02d}_SOFT_BLOCKER.md",
            f"# Stage {stage:02d} Soft Blocker\n\n{len(failures)} non-core ablation task(s) failed "
            "after one retry. Completed raw results remain intact and resumable; see `FAILED_TASKS.json`.\n",
        )
        return decision

    output_summary = _write_outputs(pipeline_dir, root, stage, specs)
    atomic_write_json(root / "ABLATION_COMPLETE.json", {
        "created_at": now_iso(), "stage": stage, "decision": "PASS", **output_summary,
    })
    decision = {
        "stage": stage, "name": prefix, "created_at": now_iso(), "decision": "PASS",
        "run_root": str(root), "canary": "PASS", "summary": output_summary,
        "next_stage": stage + 1,
    }
    atomic_write_json(stage_decision_path(pipeline_dir, stage), decision)
    return decision


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-dir", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--stage-9", action="store_true")
    group.add_argument("--stage-10", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    stage = 9 if args.stage_9 else 10
    result = run_stage(
        args.pipeline_dir.resolve(), stage, execute=args.execute, dry_run=args.dry_run,
    )
    print(json.dumps(result, indent=2))
    return 0 if result.get("decision") in {"PASS", "SOFT_BLOCKED", "NOT_RUN"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
