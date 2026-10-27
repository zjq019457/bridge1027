#!/usr/bin/env python3
"""SP-SlimInfer v2 real cross-layer trajectory probe (prefill-only, no QA).

Phase 1: smoke 4 samples (HotpotQA 2 + MuSiQue 2)
Phase 2: 24-sample audit (only if smoke PASS)

Does NOT modify SP v1 proxy, does NOT run QA, does NOT run 17-task / 4-task full.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bridgekv.sp_trajectory import (
    TRAJECTORY_PRUNE_LAYERS,
    active_set_redundancy,
    load_sample_history_from_npz,
    semantic_absorption,
    semantic_drift,
    semantic_persistence,
    semantic_relay,
)
from tools.run_bridgeflow_silm_property_probe import read_jsonl

TASKS = ("hotpotqa", "musique")
PRUNE_LAYERS = list(TRAJECTORY_PRUNE_LAYERS)
BLOCK_SIZE = 64
MAX_MODEL_LEN = 32768
DEFAULT_MODEL = (
    "/home/yu-bai/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/"
    "snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
)
DEFAULT_LONGBENCH = Path("/home/yu-bai/Desktop/jiaqizhang/HeadKV/data/LongBench")
DEFAULT_BASELINE_UASR = (
    REPO_ROOT / "results/bridgeflow_silm_property_probe_2task_full_20260709_211600/uasr_score_dump.jsonl"
)


def parse_args() -> argparse.Namespace:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["smoke", "audit24", "both"], default="both")
    p.add_argument("--out_dir", type=Path, default=None)
    p.add_argument("--model_path", type=Path, default=Path(DEFAULT_MODEL))
    p.add_argument("--longbench_dir", type=Path, default=DEFAULT_LONGBENCH)
    p.add_argument("--baseline_uasr", type=Path, default=DEFAULT_BASELINE_UASR)
    p.add_argument("--smoke_samples_per_task", type=int, default=2)
    p.add_argument("--audit_samples_per_task", type=int, default=12)
    p.add_argument("--skip_gpu_runs", action="store_true")
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def python_bin() -> Path:
    candidate = REPO_ROOT / ".venv" / "bin" / "python"
    return candidate if candidate.exists() else Path(sys.executable)


def build_sample_spec(samples_per_task: int) -> str:
    indices = ",".join(str(i) for i in range(samples_per_task))
    return ";".join(f"{task}:{indices}" for task in TASKS)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def check_gpu_or_block(out_dir: Path) -> Tuple[bool, str]:
    try:
        proc = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        text = (proc.stdout or "") + (proc.stderr or "")
        if proc.returncode != 0:
            msg = f"nvidia-smi failed rc={proc.returncode}\n{text}"
            write_text(out_dir / "BLOCKED_GPU_OR_OOM.md", msg)
            return False, msg
        if "No running processes found" not in text and "MiB /" in text:
            used = re.findall(r"(\d+)MiB\s*/\s*(\d+)MiB", text)
            if used:
                used_mb, total_mb = int(used[0][0]), int(used[0][1])
                if used_mb / max(1, total_mb) > 0.92:
                    msg = f"GPU heavily utilized ({used_mb}/{total_mb} MiB). Aborting.\n{text}"
                    write_text(out_dir / "BLOCKED_GPU_OR_OOM.md", msg)
                    return False, msg
        return True, text
    except Exception as exc:
        msg = f"GPU check failed: {exc}"
        write_text(out_dir / "BLOCKED_GPU_OR_OOM.md", msg)
        return False, msg


def gate_audit_cmd(
    *,
    run_dir: Path,
    sample_spec: str,
    model_path: Path,
    longbench_dir: Path,
    sp_trajectory_dump_path: Optional[Path] = None,
) -> List[str]:
    score_dump = run_dir / "uasr_score_dump.jsonl"
    used_blocks = run_dir / "used_blocks.jsonl"
    for path in (score_dump, used_blocks):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
    cmd = [
        str(python_bin()),
        str(REPO_ROOT / "experiments" / "run_qwen25_sliminfer_variants.py"),
        "--model_path", str(model_path),
        "--tokenizer_path", str(model_path),
        "--model_family", "qwen2",
        "--dtype", "fp16",
        "--device", "cuda",
        "--backend", "flash_attention_2",
        "--longbench_dir", str(longbench_dir),
        "--longbench_tasks", ",".join(TASKS),
        "--longbench_sample_indices_by_task", sample_spec,
        "--longbench_subset_name", "sp_trajectory_probe",
        "--max_model_len", str(MAX_MODEL_LEN),
        "--method", "sliminfer_original",
        "--scoring", "token_unit",
        "--decode_policy", "original",
        "--rescue_policy", "none",
        "--prune_layers", ",".join(str(x) for x in PRUNE_LAYERS),
        "--retain_tokens", "12288,6144,4096",
        "--block_size", str(BLOCK_SIZE),
        "--token_unit_size", "8",
        "--local_query_window", "4",
        "--swap_threshold", "0.9",
        "--seed", "0",
        "--output_dir", str(run_dir),
        "--min_free_gpu_gb", "0",
        "--eval", "gate_audit",
        "--uasr_score_dump_path", str(score_dump),
        "--uasr_tail_width", "8",
        "--save_block_scores", "sampled",
        "--debug_used_block_audit_path", str(used_blocks),
        "--debug_used_block_include_full", "true",
        "--debug_used_block_full_layers", "27",
    ]
    if sp_trajectory_dump_path is not None:
        sp_trajectory_dump_path.mkdir(parents=True, exist_ok=True)
        cmd.extend(["--sp_trajectory_dump_path", str(sp_trajectory_dump_path)])
    return cmd


def run_command(cmd: Sequence[str], log_prefix: Path) -> float:
    log_prefix.parent.mkdir(parents=True, exist_ok=True)
    stdout_path = Path(f"{log_prefix}_stdout.log")
    stderr_path = Path(f"{log_prefix}_stderr.log")
    started = time.perf_counter()
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        proc = subprocess.run(list(cmd), cwd=str(REPO_ROOT), text=True, stdout=stdout, stderr=stderr)
    elapsed = time.perf_counter() - started
    out_text = stdout_path.read_text(encoding="utf-8", errors="ignore")
    err_text = stderr_path.read_text(encoding="utf-8", errors="ignore")
    if "CUDA out of memory" in out_text or "CUDA out of memory" in err_text:
        raise RuntimeError(f"CUDA OOM: see {stderr_path}")
    if proc.returncode != 0:
        raise RuntimeError(f"runner failed rc={proc.returncode}; see {stderr_path}")
    return elapsed


def gate_audit_complete(run_dir: Path, expected_samples: int) -> bool:
    manifest = run_dir / "gate_audit_manifest.jsonl"
    if not manifest.exists():
        return False
    keys: Set[Tuple[str, int]] = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        keys.add((str(row.get("task", "")), int(row.get("sample_index", -1))))
    return len(keys) >= expected_samples


def load_prefill_uasr(path: Path) -> List[Dict[str, Any]]:
    rows = read_jsonl(path)
    return [r for r in rows if str(r.get("phase", "")) == "prefill" and str(r.get("task", "")) in TASKS]


def index_uasr(rows: Sequence[Mapping[str, Any]]) -> Dict[Tuple[str, str, int], Mapping[str, Any]]:
    out: Dict[Tuple[str, str, int], Mapping[str, Any]] = {}
    for row in rows:
        task = str(row.get("task", ""))
        sid = str(row.get("sample_id", row.get("sample_index", "")))
        layer = int(row.get("layer_id", -1))
        if task and sid and layer >= 0:
            out[(task, sid, layer)] = row
    return out


def compare_selection_baseline(
    baseline_rows: Sequence[Mapping[str, Any]],
    hook_rows: Sequence[Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    base_idx = index_uasr(baseline_rows)
    hook_idx = index_uasr(hook_rows)
    keys = sorted(set(base_idx.keys()) & set(hook_idx.keys()), key=lambda k: (k[0], k[1], k[2]))
    detail: List[Dict[str, Any]] = []
    mismatches = 0
    for task, sid, layer in keys:
        b_sel = sorted(int(x) for x in base_idx[(task, sid, layer)].get("selected_block_ids", []))
        h_sel = sorted(int(x) for x in hook_idx[(task, sid, layer)].get("selected_block_ids", []))
        b_score = base_idx[(task, sid, layer)].get("block_scores", [])
        h_score = hook_idx[(task, sid, layer)].get("block_scores", [])
        changed = int(b_sel != h_sel)
        mismatches += changed
        detail.append(
            {
                "task": task,
                "sample_id": sid,
                "layer_id": layer,
                "baseline_selected_count": len(b_sel),
                "hook_selected_count": len(h_sel),
                "selection_match": int(b_sel == h_sel),
                "score_list_match": int(b_score == h_score),
            }
        )
    summary = {
        "layers_compared": len(keys),
        "selection_mismatches": mismatches,
        "zero_pollution": mismatches == 0,
    }
    return detail, summary


def read_scalar_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def analyze_trajectory_dump(traj_root: Path) -> Dict[str, Any]:
    index_rows = read_jsonl(traj_root / "trajectory_index.jsonl")
    scalar_rows = read_scalar_csv(traj_root / "trajectory_scalar_features.csv")
    hook_calls = defaultdict(int)
    for row in index_rows:
        hook_calls[int(row["layer_idx"])] += 1

    samples: Set[Tuple[str, str]] = set()
    for row in index_rows:
        samples.add((str(row["task"]), str(row["sample_id"])))

    dim_checks: Dict[str, Any] = {}
    cos_finite = 0
    cos_total = 0
    nan_inf = 0
    censored = 0
    rep_avail = 0
    total_rows = len(scalar_rows)

    for row in scalar_rows:
        if str(row.get("trajectory_censored", "0")) in {"1", "True", "true"}:
            censored += 1
        if str(row.get("representation_available", "0")) in {"1", "True", "true"}:
            rep_avail += 1
        for field in ("query_hidden_cosine", "query_key_cosine"):
            val = row.get(field, "")
            if val in ("", "UNAVAILABLE"):
                continue
            cos_total += 1
            try:
                f = float(val)
                if math.isfinite(f):
                    cos_finite += 1
                else:
                    nan_inf += 1
            except ValueError:
                nan_inf += 1

    npz_dims: List[Tuple[int, int, int]] = []
    for row in index_rows:
        npz_path = traj_root / str(row["npz"])
        if not npz_path.exists():
            continue
        data = np.load(npz_path, allow_pickle=False)
        npz_dims.append(
            (
                int(data["hidden_dim"][0]) if "hidden_dim" in data else int(data["hidden_block_rep"].shape[-1]),
                int(data["key_dim"][0]) if "key_dim" in data else int(data["key_block_rep"].shape[-1]),
                int(data["query_dim"][0]) if "query_dim" in data else int(data["local_query_rep"].shape[0]),
            )
        )
    if npz_dims:
        h_dims = {d[0] for d in npz_dims}
        k_dims = {d[1] for d in npz_dims}
        q_dims = {d[2] for d in npz_dims}
        dim_checks = {
            "hidden_dim_consistent": len(h_dims) == 1,
            "key_dim_consistent": len(k_dims) == 1,
            "query_dim_consistent": len(q_dims) == 1,
            "hidden_dim": sorted(h_dims),
            "key_dim": sorted(k_dims),
            "query_dim": sorted(q_dims),
        }

    return {
        "num_samples": len(samples),
        "num_index_rows": len(index_rows),
        "num_scalar_rows": total_rows,
        "hook_calls_by_layer": dict(hook_calls),
        "layers_hooked": sorted(hook_calls.keys()),
        "censored_ratio": censored / max(1, total_rows),
        "representation_available_ratio": rep_avail / max(1, total_rows),
        "cosine_finite_ratio": cos_finite / max(1, cos_total),
        "nan_inf_cosine_count": nan_inf,
        "dim_checks": dim_checks,
        "samples": sorted(samples),
    }


def compute_feature_audit(traj_root: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    index_rows = read_jsonl(traj_root / "trajectory_index.jsonl")
    samples: Dict[Tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in index_rows:
        samples[(str(row["task"]), str(row["sample_id"]))].append(row)

    feature_rows: List[Dict[str, Any]] = []
    feature_values: Dict[str, List[float]] = defaultdict(list)
    score_vals: List[float] = []

    for (task, sample_id), rows in sorted(samples.items()):
        history = load_sample_history_from_npz(traj_root, rows, task, sample_id)
        scalar_rows = [r for r in read_scalar_csv(traj_root / "trajectory_scalar_features.csv") if r.get("task") == task and str(r.get("sample_id")) == str(sample_id)]
        for sr in scalar_rows:
            try:
                sc = sr.get("original_sliminfer_score", "")
                if sc not in ("", "UNAVAILABLE"):
                    score_vals.append(float(sc))
            except ValueError:
                pass

        for layer_idx in PRUNE_LAYERS:
            if layer_idx not in history:
                continue
            cur = history[layer_idx]
            candidate_ids = [bid for i, bid in enumerate(cur["block_ids"]) if cur["rep_available"][i] and not cur["censored"][i]]
            redundancy = active_set_redundancy(history, layer_idx, candidate_ids)
            for block_id in cur["block_ids"]:
                idx = cur["block_ids"].index(block_id)
                if cur["censored"][idx]:
                    continue
                row = {
                    "task": task,
                    "sample_id": sample_id,
                    "layer_idx": layer_idx,
                    "block_id": block_id,
                    "semantic_persistence": semantic_persistence(history, layer_idx, block_id),
                    "semantic_drift": semantic_drift(history, layer_idx, block_id),
                    "semantic_absorption": semantic_absorption(history, layer_idx, block_id),
                    "semantic_relay": semantic_relay(history, layer_idx, block_id),
                    "redundancy_max_pairwise": redundancy.get("max_pairwise_cosine"),
                }
                feature_rows.append(row)
                for fk in ("semantic_persistence", "semantic_drift", "semantic_absorption", "semantic_relay"):
                    val = row[fk]
                    if val is not None and math.isfinite(float(val)):
                        feature_values[fk].append(float(val))

    def cov(name: str) -> float:
        vals = feature_values[name]
        return len(vals) / max(1, len(feature_rows))

    summary = {
        "feature_rows": len(feature_rows),
        "persistence_coverage": cov("semantic_persistence"),
        "drift_coverage": cov("semantic_drift"),
        "absorption_coverage": cov("semantic_absorption"),
        "relay_coverage": cov("semantic_relay"),
        "feature_stats": {
            k: {
                "count": len(v),
                "mean": float(statistics.mean(v)) if v else None,
                "std": float(statistics.pstdev(v)) if len(v) > 1 else None,
                "min": float(min(v)) if v else None,
                "max": float(max(v)) if v else None,
            }
            for k, v in feature_values.items()
        },
        "score_stats": {
            "count": len(score_vals),
            "std": float(statistics.pstdev(score_vals)) if len(score_vals) > 1 else None,
        },
    }
    return feature_rows, summary


def disk_size_mb(root: Path) -> float:
    total = 0
    for p in root.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total / (1024 * 1024)


def run_phase(
    *,
    phase_name: str,
    out_dir: Path,
    sample_spec: str,
    expected_samples: int,
    model_path: Path,
    longbench_dir: Path,
    baseline_uasr: Path,
    skip_gpu: bool,
    resume: bool,
) -> Dict[str, Any]:
    baseline_dir = out_dir / "baseline_no_hook"
    traj_dir = out_dir / "trajectory_hook"
    traj_dump = traj_dir / "trajectory_dump"

    result: Dict[str, Any] = {"phase": phase_name, "expected_samples": expected_samples}

    if not skip_gpu:
        ok, gpu_text = check_gpu_or_block(out_dir)
        result["gpu_check_ok"] = ok
        result["gpu_text_head"] = gpu_text[:500]
        if not ok:
            result["blocked"] = True
            return result

        if not (resume and gate_audit_complete(baseline_dir, expected_samples)):
            result["baseline_elapsed_s"] = run_command(
                gate_audit_cmd(
                    run_dir=baseline_dir,
                    sample_spec=sample_spec,
                    model_path=model_path,
                    longbench_dir=longbench_dir,
                ),
                out_dir / f"{phase_name}_baseline",
            )
        if not (resume and gate_audit_complete(traj_dir, expected_samples)):
            result["trajectory_elapsed_s"] = run_command(
                gate_audit_cmd(
                    run_dir=traj_dir,
                    sample_spec=sample_spec,
                    model_path=model_path,
                    longbench_dir=longbench_dir,
                    sp_trajectory_dump_path=traj_dump,
                ),
                out_dir / f"{phase_name}_trajectory",
            )

    baseline_rows = load_prefill_uasr(baseline_dir / "uasr_score_dump.jsonl")
    hook_rows = load_prefill_uasr(traj_dir / "uasr_score_dump.jsonl")
    ref_rows = load_prefill_uasr(baseline_uasr) if baseline_uasr.exists() else []

    compare_hook, compare_summary = compare_selection_baseline(baseline_rows, hook_rows)
    result["baseline_non_interference"] = compare_summary

    traj_analysis = analyze_trajectory_dump(traj_dump)
    result["trajectory_analysis"] = traj_analysis
    result["samples_completed"] = traj_analysis["num_samples"]

    manifest_count = 0
    manifest = traj_dir / "gate_audit_manifest.jsonl"
    if manifest.exists():
        manifest_count = len(manifest.read_text(encoding="utf-8").splitlines())
    result["manifest_lines"] = manifest_count

    # Smoke checks
    checks = {
        "samples_complete": traj_analysis["num_samples"] >= expected_samples,
        "hooks_all_layers": all(l in traj_analysis["hook_calls_by_layer"] for l in PRUNE_LAYERS),
        "hook_calls_per_layer": all(traj_analysis["hook_calls_by_layer"].get(l, 0) >= expected_samples for l in PRUNE_LAYERS),
        "dim_consistent": bool(traj_analysis.get("dim_checks", {}).get("hidden_dim_consistent", False)),
        "cosine_finite": traj_analysis["cosine_finite_ratio"] >= 0.99,
        "no_nan_inf": traj_analysis["nan_inf_cosine_count"] == 0,
        "zero_pollution": compare_summary["zero_pollution"],
    }
    result["checks"] = checks
    result["smoke_pass"] = all(checks.values())

    write_csv(out_dir / "HOOK_CALL_AUDIT.csv", [
        {"layer_idx": l, "hook_calls": traj_analysis["hook_calls_by_layer"].get(l, 0), "expected": expected_samples}
        for l in PRUNE_LAYERS
    ], ["layer_idx", "hook_calls", "expected"])

    write_csv(out_dir / "BASELINE_NON_INTERFERENCE_AUDIT.csv", compare_hook,
              ["task", "sample_id", "layer_id", "baseline_selected_count", "hook_selected_count", "selection_match", "score_list_match"])

    write_text(
        out_dir / "BASELINE_NON_INTERFERENCE_AUDIT.md",
        "# Baseline Non-Interference Audit\n\n"
        f"- layers compared: {compare_summary['layers_compared']}\n"
        f"- selection mismatches: {compare_summary['selection_mismatches']}\n"
        f"- zero pollution: **{compare_summary['zero_pollution']}**\n",
    )

    write_text(
        out_dir / "TRAJECTORY_SCHEMA.md",
        "# Trajectory Schema\n\n"
        "- `trajectory_vectors/*.npz`: block_ids, hidden_block_rep, key_block_rep, local_query_rep, flags\n"
        "- `trajectory_index.jsonl`: per sample/layer index\n"
        "- `trajectory_scalar_features.csv`: scalar fields per block\n\n"
        f"Dim checks: `{json.dumps(traj_analysis.get('dim_checks', {}), ensure_ascii=False)}`\n",
    )

    if phase_name.startswith("smoke"):
        write_text(
            out_dir / "SMOKE_REPORT.md",
            "# SP Trajectory Smoke Report\n\n"
            f"- expected samples: {expected_samples}\n"
            f"- completed: {traj_analysis['num_samples']}\n"
            f"- smoke pass: **{result['smoke_pass']}**\n\n"
            "## Checks\n" + "\n".join(f"- {k}: {v}" for k, v in checks.items()) + "\n",
        )
    else:
        feature_rows, feature_summary = compute_feature_audit(traj_dump)
        write_csv(out_dir / "TRUE_FEATURE_AUDIT.csv", feature_rows,
                  ["task", "sample_id", "layer_idx", "block_id", "semantic_persistence", "semantic_drift",
                   "semantic_absorption", "semantic_relay", "redundancy_max_pairwise"])
        write_json(out_dir / "TRUE_FEATURE_AUDIT_SUMMARY.json", feature_summary)
        write_text(
            out_dir / "TRUE_FEATURE_AUDIT_REPORT.md",
            "# True Feature Audit\n\n" + json.dumps(feature_summary, ensure_ascii=False, indent=2) + "\n",
        )
        write_text(
            out_dir / "TRAJECTORY_COVERAGE_AUDIT.md",
            "# Trajectory Coverage\n\n"
            f"- censored ratio: {traj_analysis['censored_ratio']:.4f}\n"
            f"- representation available: {traj_analysis['representation_available_ratio']:.4f}\n"
            f"- persistence coverage: {feature_summary['persistence_coverage']:.4f}\n"
            f"- drift coverage: {feature_summary['drift_coverage']:.4f}\n",
        )
        write_text(
            out_dir / "CENSORING_ANALYSIS.md",
            f"# Censoring Analysis\n\nCensored ratio: {traj_analysis['censored_ratio']:.4f}\n",
        )
        elapsed = float(result.get("baseline_elapsed_s", 0) or 0) + float(result.get("trajectory_elapsed_s", 0) or 0)
        write_text(
            out_dir / "MEMORY_AND_OVERHEAD_REPORT.md",
            f"# Memory & Overhead\n\n- wall time (s): {elapsed:.1f}\n- disk MB: {disk_size_mb(traj_dump):.2f}\n",
        )

        go = (
            traj_analysis["num_samples"] >= expected_samples
            and feature_summary["persistence_coverage"] >= 0.05
            and feature_summary["drift_coverage"] >= 0.05
            and checks["zero_pollution"]
            and traj_analysis["representation_available_ratio"] >= 0.3
        )
        feas_path = out_dir / ("SP_V2_TRAJECTORY_GO.md" if go else "SP_V2_TRAJECTORY_NO_GO.md")
        write_text(
            feas_path,
            f"# SP-v2 Trajectory {'GO' if go else 'NO-GO'}\n\n"
            f"See TRUE_FEATURE_AUDIT_REPORT.md and TRAJECTORY_COVERAGE_AUDIT.md.\n",
        )
        write_text(
            out_dir / "SP_V2_RUNTIME_FEASIBILITY.md",
            f"# Runtime Feasibility\n\nDecision: **{'GO' if go else 'NO-GO'}**\n",
        )
        result["go_decision"] = go

    result["disk_mb"] = disk_size_mb(traj_dump)
    return result


def main() -> int:
    args = parse_args()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results: Dict[str, Any] = {"started_at": ts, "phases": {}}

    if args.mode in ("smoke", "both"):
        smoke_dir = args.out_dir or (REPO_ROOT / f"results/sp_trajectory_probe_smoke_{ts}")
        if args.out_dir and args.mode == "both":
            smoke_dir = REPO_ROOT / f"results/sp_trajectory_probe_smoke_{ts}"
        smoke_spec = build_sample_spec(args.smoke_samples_per_task)
        expected = args.smoke_samples_per_task * len(TASKS)
        smoke_res = run_phase(
            phase_name="smoke",
            out_dir=smoke_dir,
            sample_spec=smoke_spec,
            expected_samples=expected,
            model_path=args.model_path,
            longbench_dir=args.longbench_dir,
            baseline_uasr=args.baseline_uasr,
            skip_gpu=args.skip_gpu_runs,
            resume=args.resume,
        )
        results["phases"]["smoke"] = smoke_res
        write_json(smoke_dir / "smoke_result.json", smoke_res)

        if args.mode == "both" and not smoke_res.get("smoke_pass", False):
            write_text(
                smoke_dir / "SMOKE_REPORT.md",
                (smoke_dir / "SMOKE_REPORT.md").read_text(encoding="utf-8")
                + "\n\n**STOP**: smoke failed; 24-sample audit not started.\n",
            )
            write_json(REPO_ROOT / "results/sp_trajectory_probe_latest.json", results)
            print(json.dumps(results, ensure_ascii=False, indent=2))
            return 1

    if args.mode in ("audit24", "both"):
        if args.mode == "both":
            smoke_pass = results.get("phases", {}).get("smoke", {}).get("smoke_pass", False)
            if not smoke_pass:
                return 1
        audit_dir = REPO_ROOT / f"results/sp_trajectory_probe_24sample_{ts}"
        audit_spec = build_sample_spec(args.audit_samples_per_task)
        expected = args.audit_samples_per_task * len(TASKS)
        audit_res = run_phase(
            phase_name="audit24",
            out_dir=audit_dir,
            sample_spec=audit_spec,
            expected_samples=expected,
            model_path=args.model_path,
            longbench_dir=args.longbench_dir,
            baseline_uasr=args.baseline_uasr,
            skip_gpu=args.skip_gpu_runs,
            resume=args.resume,
        )
        results["phases"]["audit24"] = audit_res
        write_json(audit_dir / "audit24_result.json", audit_res)

    write_json(REPO_ROOT / "results/sp_trajectory_probe_latest.json", results)
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
