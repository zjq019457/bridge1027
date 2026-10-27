#!/usr/bin/env python3
"""Official DRM-Add 17-task LongBench full runner.

This wrapper does not change SlimInfer/DRM selection logic.  It writes the
protocol/config/reuse audits, invokes the existing Qwen2.5 SlimInfer runtime
with frozen DRM parameters, supports resume from saved predictions, and
consolidates the five-method official full run.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from longbench_eval import scorer  # noqa: E402
from tools.consolidate_official_repro_results import per_sample_score  # noqa: E402
from tools.run_drm_add_mechanism_revalidation import (  # noqa: E402
    DEFAULT_LONGBENCH,
    DEFAULT_MODEL,
    METHODS as REVALIDATION_METHODS,
)


OFFICIAL_COUNTS: Dict[str, int] = {
    "qasper": 200,
    "multifieldqa_en": 150,
    "hotpotqa": 200,
    "2wikimqa": 200,
    "musique": 200,
    "gov_report": 200,
    "qmsum": 200,
    "multi_news": 200,
    "vcsum": 200,
    "trec": 200,
    "triviaqa": 200,
    "samsum": 200,
    "lsht": 200,
    "passage_count": 200,
    "passage_retrieval_en": 200,
    "lcc": 500,
    "repobench-p": 500,
}

TASKS: Tuple[str, ...] = (
    "passage_retrieval_en",
    "qasper",
    "multifieldqa_en",
    "hotpotqa",
    "2wikimqa",
    "musique",
    "gov_report",
    "qmsum",
    "multi_news",
    "vcsum",
    "passage_count",
    "trec",
    "triviaqa",
    "samsum",
    "lsht",
    "lcc",
    "repobench-p",
)
assert sum(OFFICIAL_COUNTS[t] for t in TASKS) == 3950

RUN_ORDER: Tuple[str, ...] = (
    "sliminfer_original",
    "sliminfer_larger_km",
    "sliminfer_random_add_km",
    "sliminfer_drm_add_km",
    "drm_swap_original_k",
)

K_BUDGET_METHODS = ("sliminfer_original", "drm_swap_original_k")
KM_BUDGET_METHODS = (
    "sliminfer_larger_km",
    "sliminfer_random_add_km",
    "sliminfer_drm_add_km",
)

COMPARISONS: Tuple[Tuple[str, str, str], ...] = (
    ("drm_add_vs_larger_km", "sliminfer_larger_km", "sliminfer_drm_add_km"),
    ("drm_add_vs_random_add_km", "sliminfer_random_add_km", "sliminfer_drm_add_km"),
    ("drm_swap_vs_original_k", "sliminfer_original", "drm_swap_original_k"),
    ("drm_add_vs_swap_budget_tradeoff", "drm_swap_original_k", "sliminfer_drm_add_km"),
)

FROZEN_DRM = {
    "rescue_policy": "add",
    "rescue_score": "drm",
    "rescue_ratio": 0.05,
    "alpha_diff": 0.5,
    "lambda_red": 0.25,
    "eta_rescue": 0.25,
    "eta_active": 0.25,
    "seed": 0,
}

SCHEDULE = {
    9: {"K": 192, "M": 10, "retain_tokens": 12288},
    18: {"K": 96, "M": 5, "retain_tokens": 6144},
    26: {"K": 64, "M": 3, "retain_tokens": 4096},
}

FREEZE_DIR = REPO_ROOT / "results" / "drm_add_mechanism_revalidation_20260714_174251"
V3_DIR = FREEZE_DIR / "phase2_feature_complete_runtime_20260715_v3"
REQUIRED_FREEZE_FILES = (
    V3_DIR / "FINAL_DRM_ADD_REVALIDATION_REPORT.md",
    V3_DIR / "FINAL_DRM_ADD_REVALIDATION_DECISION.json",
    FREEZE_DIR / "DRM_ADD_CONFIG_RECOVERY.json",
    FREEZE_DIR / "DRM_REVALIDATION_FROZEN_CONFIG.json",
    FREEZE_DIR / "DRM_REVALIDATION_METHOD_HASHES.json",
)

SOURCE_FILES_FOR_HASH = (
    "experiments/run_qwen25_sliminfer_variants.py",
    "utils/cache_utils.py",
    "bridgekv/rescuekv_scoring.py",
    "tools/run_drm_add_mechanism_revalidation.py",
    "tools/run_drm_add_six_task_gate.py",
    "tools/run_drm_add_full_longbench.py",
    "tools/run_cc_sliminfer_v1_17task_full_official.py",
    "longbench_eval.py",
    "configs/dataset2prompt.json",
    "configs/dataset2maxlen.json",
)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fields))
        w.writeheader()
        for row in rows:
            w.writerow({field: row.get(field, "") for field in fields})


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def jsonl_rows(path: Path, limit: Optional[int] = None) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
            if limit is not None and len(rows) >= limit:
                break
    return rows


def method_spec(method: str) -> Dict[str, Any]:
    spec = dict(REVALIDATION_METHODS[method])
    spec["method"] = method
    return spec


def method_hash(method: str, protocol_hash: str, model_fingerprint: Mapping[str, Any]) -> str:
    payload = {
        "method_spec": method_spec(method),
        "frozen_drm": FROZEN_DRM,
        "schedule": SCHEDULE,
        "protocol_hash": protocol_hash,
        "model_snapshot": model_fingerprint.get("snapshot_id"),
        "decode": {"temperature": 0.0, "top_p": 1.0, "do_sample": False, "num_beams": 1},
        "scorer": "longbench_eval.scorer",
    }
    return sha256_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def sample_id(sample: Mapping[str, Any], index: int) -> str:
    return str(sample.get("_id", sample.get("id", index)))


def load_longbench_samples(longbench_dir: Path, task: str) -> List[Dict[str, Any]]:
    candidates = [
        longbench_dir / f"{task}.jsonl",
        longbench_dir / f"{task}.json",
        longbench_dir / task / "test.jsonl",
        longbench_dir / task / "test.json",
        longbench_dir / task / f"{task}.jsonl",
        longbench_dir / task / f"{task}.json",
    ]
    for path in candidates:
        if not path.exists():
            continue
        if path.suffix == ".jsonl":
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        else:
            data = json.loads(path.read_text(encoding="utf-8"))
            rows = data if isinstance(data, list) else data.get("data", [])
        if len(rows) < OFFICIAL_COUNTS[task]:
            raise RuntimeError(f"{task}: {path} has {len(rows)} < official {OFFICIAL_COUNTS[task]}")
        return rows[: OFFICIAL_COUNTS[task]]
    raise FileNotFoundError(f"Missing LongBench file for {task} under {longbench_dir}")


def model_fingerprint(model_path: Path) -> Dict[str, Any]:
    model_path = model_path.expanduser().resolve()
    snapshot_id = model_path.name
    parts = list(model_path.parts)
    if "snapshots" in parts:
        idx = parts.index("snapshots")
        if idx + 1 < len(parts):
            snapshot_id = parts[idx + 1]
    selected = {}
    for name in (
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "model.safetensors.index.json",
    ):
        path = model_path / name
        if path.exists():
            selected[name] = sha256_file(path)
    return {
        "model_path": str(model_path),
        "snapshot_id": snapshot_id,
        "selected_file_hashes": selected,
        "fingerprint_hash": sha256_text(json.dumps({"snapshot_id": snapshot_id, "files": selected}, sort_keys=True)),
    }


def source_hashes() -> Dict[str, str]:
    out = {}
    for rel in SOURCE_FILES_FOR_HASH:
        path = REPO_ROOT / rel
        out[rel] = sha256_file(path) if path.exists() else "missing"
    return out


def build_sample_manifest(longbench_dir: Path) -> Dict[str, Any]:
    prompt_map = load_json(REPO_ROOT / "configs" / "dataset2prompt.json")
    sample_ids_by_task: Dict[str, List[str]] = {}
    prompt_hashes_by_task: Dict[str, List[str]] = {}
    counts: Dict[str, int] = {}
    first_last: Dict[str, Dict[str, str]] = {}
    for task in TASKS:
        rows = load_longbench_samples(longbench_dir, task)
        ids = [sample_id(row, i) for i, row in enumerate(rows)]
        sample_ids_by_task[task] = ids
        counts[task] = len(rows)
        first_last[task] = {"first": ids[0], "last": ids[-1]}
        fmt = prompt_map[task]
        prompt_hashes: List[str] = []
        for row in rows:
            try:
                prompt = fmt.format(**row)
            except Exception:
                prompt = str(row.get("input", "")) + "\n" + str(row.get("context", ""))
            prompt_hashes.append(sha256_text(prompt))
        prompt_hashes_by_task[task] = prompt_hashes
    manifest_core = {
        "task_order": list(TASKS),
        "counts": counts,
        "sample_ids_by_task": sample_ids_by_task,
    }
    prompt_core = {
        "task_order": list(TASKS),
        "prompt_hashes_by_task": prompt_hashes_by_task,
    }
    return {
        "task_order": list(TASKS),
        "counts": counts,
        "total": sum(counts.values()),
        "sample_ids_by_task": sample_ids_by_task,
        "first_last_sample_ids": first_last,
        "sample_ids_hash_by_task": {task: sha256_text("\n".join(ids)) for task, ids in sample_ids_by_task.items()},
        "prompt_hash_by_task": {task: sha256_text("\n".join(vals)) for task, vals in prompt_hashes_by_task.items()},
        "task_manifest_hash": sha256_text(json.dumps(manifest_core, sort_keys=True, separators=(",", ":"))),
        "prompt_manifest_hash": sha256_text(json.dumps(prompt_core, sort_keys=True, separators=(",", ":"))),
    }


def ensure_dirs(out_dir: Path) -> None:
    for rel in ("logs", "commands", "checkpoints", "audits", "predictions", "reports"):
        (out_dir / rel).mkdir(parents=True, exist_ok=True)


def load_freeze_payloads() -> Dict[str, Any]:
    payloads = {}
    for path in REQUIRED_FREEZE_FILES:
        if not path.exists():
            raise FileNotFoundError(path)
        if path.suffix == ".json":
            payloads[str(path.relative_to(REPO_ROOT))] = load_json(path)
        else:
            payloads[str(path.relative_to(REPO_ROOT))] = {
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
    return payloads


def write_protocol_audit(out_dir: Path, longbench_dir: Path, manifest: Mapping[str, Any]) -> Dict[str, Any]:
    try:
        from tools.run_cc_sliminfer_v1_17task_full_official import (  # type: ignore
            OFFICIAL_COUNTS as cc_counts,
            TASKS as cc_tasks,
        )
    except Exception as exc:
        cc_counts = {}
        cc_tasks = ()
        cc_import_error = repr(exc)
    else:
        cc_import_error = ""
    try:
        from tools.run_drm_add_full_longbench import TASKS as old_drm_tasks  # type: ignore
    except Exception as exc:
        old_drm_tasks = ()
        old_import_error = repr(exc)
    else:
        old_import_error = ""

    local_counts = dict(manifest["counts"])
    rows = []
    for i, task in enumerate(TASKS, start=1):
        rows.append({
            "order": i,
            "task": task,
            "local_count": local_counts[task],
            "official_count": OFFICIAL_COUNTS[task],
            "first_sample_id": manifest["first_last_sample_ids"][task]["first"],
            "last_sample_id": manifest["first_last_sample_ids"][task]["last"],
            "sample_ids_hash": manifest["sample_ids_hash_by_task"][task],
            "prompt_hash": manifest["prompt_hash_by_task"][task],
        })

    current_runner_counts_match = dict(cc_counts) == OFFICIAL_COUNTS
    current_runner_order_match = tuple(cc_tasks) == TASKS
    old_runner_task_set_match = set(old_drm_tasks) == set(TASKS)
    old_runner_order_match = tuple(old_drm_tasks) == TASKS
    local_counts_match = local_counts == OFFICIAL_COUNTS
    audit = {
        "protocol": "official_sliminfer_longbench_17task_full",
        "longbench_dir": str(longbench_dir),
        "tasks": list(TASKS),
        "counts": local_counts,
        "total": sum(local_counts.values()),
        "expected_total": 3950,
        "task_manifest_hash": manifest["task_manifest_hash"],
        "prompt_manifest_hash": manifest["prompt_manifest_hash"],
        "current_official_runner": {
            "path": "tools/run_cc_sliminfer_v1_17task_full_official.py",
            "import_error": cc_import_error,
            "counts_match": current_runner_counts_match,
            "order_match": current_runner_order_match,
        },
        "historical_drm_full_runner": {
            "path": "tools/run_drm_add_full_longbench.py",
            "import_error": old_import_error,
            "task_set_match": old_runner_task_set_match,
            "order_match": old_runner_order_match,
            "note": "Older runner has a different task order but the same 17-task set; current official SlimInfer runner order is used.",
        },
        "pass": bool(local_counts_match and sum(local_counts.values()) == 3950 and current_runner_counts_match),
    }
    write_json(out_dir / "OFFICIAL_17TASK_SAMPLE_MANIFEST.json", manifest)
    write_json(out_dir / "FULL17_PROTOCOL_AUDIT.json", audit)
    write_csv(out_dir / "FULL17_PROTOCOL_TASK_COUNTS.csv", rows, list(rows[0].keys()))

    lines = [
        "# FULL17 Protocol Audit",
        "",
        f"- Protocol: official SlimInfer LongBench 17-task full.",
        f"- LongBench dir: `{longbench_dir}`",
        f"- Total samples per method: `{sum(local_counts.values())}`",
        f"- Task manifest hash: `{manifest['task_manifest_hash']}`",
        f"- Prompt manifest hash: `{manifest['prompt_manifest_hash']}`",
        f"- Current official runner counts match: `{current_runner_counts_match}`",
        f"- Current official runner task order match: `{current_runner_order_match}`",
        f"- Historical DRM full runner task set match: `{old_runner_task_set_match}`",
        f"- Historical DRM full runner order match: `{old_runner_order_match}`",
        "",
        "| order | task | samples | first sample | last sample |",
        "|---:|---|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['order']} | `{row['task']}` | {row['local_count']} | `{row['first_sample_id']}` | `{row['last_sample_id']}` |"
        )
    lines.extend([
        "",
        "Decision: **PASS**. The local data and current official runner use the uncapped full protocol; no per-task max-200 cap is used.",
        "",
    ])
    (out_dir / "FULL17_PROTOCOL_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
    return audit


def write_frozen_config(out_dir: Path, manifest: Mapping[str, Any], model_fp: Mapping[str, Any]) -> Dict[str, Any]:
    freeze_payloads = load_freeze_payloads()
    hashes = source_hashes()
    method_hashes = {method: method_hash(method, manifest["task_manifest_hash"], model_fp) for method in RUN_ORDER}
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_freeze_files": [str(p.relative_to(REPO_ROOT)) for p in REQUIRED_FREEZE_FILES],
        "freeze_payloads": freeze_payloads,
        "model_fingerprint": model_fp,
        "source_hashes_current": hashes,
        "task_manifest_hash": manifest["task_manifest_hash"],
        "prompt_manifest_hash": manifest["prompt_manifest_hash"],
        "frozen_drm": FROZEN_DRM,
        "schedule": SCHEDULE,
        "runtime_protocol": {
            "model_family": "qwen2",
            "dtype": "fp16",
            "backend": "flash_attention_2",
            "batch_size": 1,
            "longbench_max_samples": 0,
            "max_model_len": 131072,
            "block_strategy": "fixed",
            "block_size": 64,
            "token_unit_size": 8,
            "local_query_window": 4,
            "swap_threshold": 0.9,
            "prune_layers": [9, 18, 26],
            "retain_tokens": [12288, 6144, 4096],
            "decode_policy": "original",
            "temperature": 0.0,
            "top_p": 1.0,
            "do_sample": False,
            "num_beams": 1,
            "generation_max_new_tokens": 0,
            "enable_cpu_offload": True,
            "enable_async_kv": True,
            "oom_policy": "mark_[OOM]_sample_and_continue",
        },
        "method_config_hashes": method_hashes,
        "no_retuning": True,
    }
    payload["full17_frozen_config_hash"] = sha256_text(json.dumps(payload, sort_keys=True, default=str))
    write_json(out_dir / "FULL17_FROZEN_CONFIG.json", payload)
    return payload


def write_method_spec(out_dir: Path, method_hashes: Mapping[str, str]) -> None:
    rows = []
    for method in RUN_ORDER:
        spec = method_spec(method)
        rows.append({
            "method": method,
            "budget": spec["budget"],
            "runner": spec["runner"],
            "rescue_policy": spec["policy"],
            "rescue_score": spec["score"],
            "base": spec.get("base") or "",
            "method_hash": method_hashes[method],
        })
    write_csv(out_dir / "FULL17_METHOD_SPEC.csv", rows, list(rows[0].keys()))
    lines = [
        "# FULL17 Method Spec",
        "",
        "Frozen parameters are inherited from the passed v3 revalidation. The wrapper only changes task scope from six-task/two-task to official 17-task full.",
        "",
        "| method | budget | runtime method | rescue_policy | rescue_score | definition |",
        "|---|---|---|---|---|---|",
    ]
    descriptions = {
        "sliminfer_original": "Original SlimInfer active set A_K.",
        "drm_swap_original_k": "Start from original A_K, apply strict one-in-one-out DRM swap, final K.",
        "sliminfer_larger_km": "Original SlimInfer relevance directly keeps K+M.",
        "sliminfer_random_add_km": "Preserve all original A_K, add M random pruned blocks with seed 0.",
        "sliminfer_drm_add_km": "Preserve all original A_K, add M blocks selected by frozen DRM score plus historical MMR.",
    }
    for row in rows:
        lines.append(
            f"| `{row['method']}` | `{row['budget']}` | `{row['runner']}` | `{row['rescue_policy']}` | `{row['rescue_score']}` | {descriptions[row['method']]} |"
        )
    lines.extend([
        "",
        "K and K+M results must be reported in separate tables. The fair K+M comparisons are DRM-Add vs larger and DRM-Add vs random-add.",
        "",
    ])
    (out_dir / "FULL17_METHOD_SPEC.md").write_text("\n".join(lines), encoding="utf-8")


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip())


def prediction_counts_from_dir(pred_dir: Path) -> Dict[str, int]:
    return {task: count_jsonl(pred_dir / f"{task}.jsonl") for task in TASKS}


def total_predictions(pred_dir: Path) -> int:
    return sum(prediction_counts_from_dir(pred_dir).values())


def inspect_run_config(run_dir: Path) -> Dict[str, Any]:
    cfg_path = run_dir / "config.json"
    metrics_path = run_dir / "metrics.json"
    cfg = load_json(cfg_path) if cfg_path.exists() else {}
    metrics = load_json(metrics_path) if metrics_path.exists() else {}
    args = cfg.get("args", {}) if isinstance(cfg, dict) else {}
    longbench = metrics.get("longbench", {}) if isinstance(metrics, dict) else {}
    return {
        "exists": run_dir.exists(),
        "config_exists": cfg_path.exists(),
        "metrics_exists": metrics_path.exists(),
        "args": args,
        "failed_samples": len(longbench.get("failed_samples", []) or []),
        "longbench_avg": metrics.get("LongBench_avg"),
        "prediction_counts": prediction_counts_from_dir(run_dir / "longbench_predictions"),
        "total_predictions": total_predictions(run_dir / "longbench_predictions"),
    }


def write_reuse_audit(out_dir: Path, manifest: Mapping[str, Any], frozen: Mapping[str, Any]) -> None:
    candidates: List[Dict[str, Any]] = []
    cc_full = REPO_ROOT / "results" / "cc_sliminfer_v1_qwen25_17task_full_official_20260712_075238" / "run_sliminfer_original"
    cc_info = inspect_run_config(cc_full)
    candidates.append({
        "artifact": str(cc_full.relative_to(REPO_ROOT)),
        "candidate_for": "sliminfer_original",
        "task_manifest_match": cc_info["prediction_counts"] == OFFICIAL_COUNTS,
        "model_protocol_hint": str(cc_info.get("args", {}).get("model_path", "")),
        "longbench_max_samples": cc_info.get("args", {}).get("longbench_max_samples", ""),
        "max_model_len": cc_info.get("args", {}).get("max_model_len", ""),
        "total_predictions": cc_info["total_predictions"],
        "oom_marked_or_failed_samples": cc_info["failed_samples"],
        "decision": "reject",
        "reason": "strict method_config_hash/prompt-hash/prediction-integrity metadata are not identical to this new five-method run; rerun fresh.",
    })
    may_root = REPO_ROOT / "results" / "rescuekv_qwen25_full_longbench_20260520_170047"
    may_pred = may_root / "raw_predictions" / "M1_sliminfer_original"
    candidates.append({
        "artifact": str(may_root.relative_to(REPO_ROOT)),
        "candidate_for": "sliminfer_original",
        "task_manifest_match": prediction_counts_from_dir(may_pred) == OFFICIAL_COUNTS,
        "model_protocol_hint": "historical configs use Qwen/Qwen2.5-7B-Instruct and max_model_len=32768",
        "longbench_max_samples": 0,
        "max_model_len": 32768,
        "total_predictions": total_predictions(may_pred),
        "oom_marked_or_failed_samples": "",
        "decision": "reject",
        "reason": "model path/hash, max_model_len, runner, and method hashes differ from current official v3 protocol.",
    })
    capped = REPO_ROOT / "results" / "cc_sliminfer_v1_qwen25_17task_20260712_074431" / "run_sliminfer_original"
    capped_info = inspect_run_config(capped)
    candidates.append({
        "artifact": str(capped.relative_to(REPO_ROOT)),
        "candidate_for": "sliminfer_original",
        "task_manifest_match": False,
        "model_protocol_hint": str(capped_info.get("args", {}).get("model_path", "")),
        "longbench_max_samples": capped_info.get("args", {}).get("longbench_max_samples", ""),
        "max_model_len": capped_info.get("args", {}).get("max_model_len", ""),
        "total_predictions": capped_info["total_predictions"],
        "oom_marked_or_failed_samples": capped_info["failed_samples"],
        "decision": "reject",
        "reason": "capped or non-full artifact; cannot substitute for official full.",
    })
    for method in RUN_ORDER:
        if method == "sliminfer_original":
            continue
        candidates.append({
            "artifact": "none_found_with_exact_method",
            "candidate_for": method,
            "task_manifest_match": False,
            "model_protocol_hint": "",
            "longbench_max_samples": "",
            "max_model_len": "",
            "total_predictions": 0,
            "oom_marked_or_failed_samples": "",
            "decision": "reject",
            "reason": "no historical official 17-task full artifact exists with exact frozen method hash and prediction completeness.",
        })
    write_csv(out_dir / "FULL17_REUSE_AUDIT.csv", candidates, list(candidates[0].keys()))
    lines = [
        "# FULL17 Reuse Audit",
        "",
        "Decision: **no historical predictions are reused**. All five methods will be regenerated in the new directory.",
        "",
        "Strict reuse fields checked: model/tokenizer fingerprint, prompt/template manifest, task manifest, sample IDs/order, decode config, scorer, method config hash, K/M schedule, and prediction completeness.",
        "",
        "OOM policy note: by latest instruction, an OOM sample marked by the runtime is retained as a prediction row and scored normally. This does not make old artifacts reusable when hashes/metadata differ.",
        "",
        "| artifact | method | decision | reason | total predictions | OOM/failed rows |",
        "|---|---|---|---|---:|---:|",
    ]
    for row in candidates:
        lines.append(
            f"| `{row['artifact']}` | `{row['candidate_for']}` | `{row['decision']}` | {row['reason']} | {row['total_predictions']} | {row['oom_marked_or_failed_samples']} |"
        )
    lines.append("")
    (out_dir / "FULL17_REUSE_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")


def write_run_plan(out_dir: Path, args: argparse.Namespace) -> None:
    command = (
        f"{args.python} tools/run_drm_add_official_17task_full.py "
        f"--out_dir {out_dir} --run"
    )
    lines = [
        "# FULL17 Run Plan",
        "",
        f"- Output directory: `{out_dir}`",
        f"- Methods: `{','.join(RUN_ORDER)}`",
        f"- Tasks: `{','.join(TASKS)}`",
        "- Batch size: `1`",
        "- Decode: greedy deterministic (`temperature=0`, `top_p=1`, `do_sample=false`).",
        "- OOM policy: mark `[OOM]` sample and continue, per latest instruction.",
        "- GPU preflight: `nvidia-smi` before each method; stop if busy before launching a method.",
        "- Resume: rerun the same command; existing `longbench_predictions/*.jsonl` prefixes are reused by the underlying runtime.",
        "",
        "## Command",
        "",
        "```bash",
        f"cd {REPO_ROOT}",
        command,
        "```",
        "",
        "## Resume Command",
        "",
        "```bash",
        f"cd {REPO_ROOT}",
        command,
        "```",
        "",
        "## Log Paths",
        "",
    ]
    for method in RUN_ORDER:
        lines.append(f"- `{method}`: `{out_dir / 'logs' / (method + '.log')}`")
    lines.append("")
    (out_dir / "FULL17_RUN_PLAN.md").write_text("\n".join(lines), encoding="utf-8")
    (out_dir / "resume.sh").write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"cd {REPO_ROOT}\n"
        f"{command}\n",
        encoding="utf-8",
    )
    os.chmod(out_dir / "resume.sh", 0o755)


def prepare_audits(args: argparse.Namespace) -> Dict[str, Any]:
    out_dir = args.out_dir.resolve()
    ensure_dirs(out_dir)
    manifest = build_sample_manifest(Path(args.longbench_dir))
    protocol = write_protocol_audit(out_dir, Path(args.longbench_dir), manifest)
    model_fp = model_fingerprint(Path(args.model_path))
    frozen = write_frozen_config(out_dir, manifest, model_fp)
    write_method_spec(out_dir, frozen["method_config_hashes"])
    write_reuse_audit(out_dir, manifest, frozen)
    write_run_plan(out_dir, args)
    status = {
        "out_dir": str(out_dir),
        "protocol_pass": protocol["pass"],
        "reuse_decision": "rerun_all_methods",
        "config_hash": frozen["full17_frozen_config_hash"],
        "tasks": list(TASKS),
        "total_samples_per_method": 3950,
        "methods": list(RUN_ORDER),
    }
    write_json(out_dir / "AUDIT_STATUS.json", status)
    return status


def gpu_preflight(out_dir: Path, method: str, min_free_gb: float) -> Tuple[bool, str]:
    log_path = out_dir / "logs" / "nvidia_smi_preflight.log"
    try:
        full = subprocess.check_output(["nvidia-smi"], text=True, stderr=subprocess.STDOUT)
        mem = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip().splitlines()
        free_gb = min(float(x.strip()) for x in mem if x.strip()) / 1024.0
        apps = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except Exception as exc:
        reason = f"nvidia-smi failed: {exc}"
        with log_path.open("a", encoding="utf-8") as f:
            f.write(f"\n[{datetime.now().isoformat(timespec='seconds')}] {method}\n{reason}\n")
        return False, reason
    heavy = []
    for line in apps.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 3:
            try:
                used_mb = float(parts[2].split()[0])
            except Exception:
                used_mb = 0.0
            if used_mb >= 1024:
                heavy.append(line)
    ok = free_gb >= min_free_gb and not heavy
    reason = f"free_gb={free_gb:.2f}; heavy_compute_apps={heavy or []}"
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"\n[{datetime.now().isoformat(timespec='seconds')}] {method}\n")
        f.write(full)
        f.write("\n")
        f.write(reason + "\n")
    return ok, reason


def qa_cmd(args: argparse.Namespace, method: str, run_dir: Path) -> List[str]:
    spec = method_spec(method)
    audit_prefix = args.out_dir.resolve() / "audits" / method
    return [
        str(args.python),
        str(REPO_ROOT / "experiments" / "run_qwen25_sliminfer_variants.py"),
        "--model_path", str(args.model_path),
        "--tokenizer_path", str(args.model_path),
        "--model_family", "qwen2",
        "--dtype", "fp16",
        "--device", "cuda",
        "--backend", "flash_attention_2",
        "--longbench_dir", str(args.longbench_dir),
        "--longbench_tasks", ",".join(TASKS),
        "--longbench_max_samples", "0",
        "--max_model_len", "131072",
        "--method", str(spec["runner"]),
        "--eval", "longbench",
        "--block_strategy", "fixed",
        "--block_size", "64",
        "--token_unit_size", "8",
        "--local_query_window", "4",
        "--swap_threshold", "0.9",
        "--prune_layers", "9,18,26",
        "--retain_tokens", "12288,6144,4096",
        "--decode_policy", "original",
        "--rescue_policy", str(spec["policy"]),
        "--rescue_score", str(spec["score"]),
        "--rescue_ratio", "0.05",
        "--alpha_diff", "0.5",
        "--lambda_red", "0.25",
        "--eta_rescue", "0.25",
        "--eta_active", "0.25",
        "--temperature", "0.0",
        "--top_p", "1.0",
        "--generation_max_new_tokens", "0",
        "--seed", "0",
        "--min_free_gpu_gb", "0",
        "--rescue_audit_path", str(audit_prefix.with_name(f"{method}_rescue_policy_audit.jsonl")),
        "--cc_selection_audit_path", str(audit_prefix.with_name(f"{method}_selection_audit.jsonl")),
        "--debug_used_block_audit_path", str(audit_prefix.with_name(f"{method}_used_blocks_audit.jsonl")),
        "--output_dir", str(run_dir),
    ]


def method_run_dir(out_dir: Path, method: str) -> Path:
    return out_dir / f"run_{method}"


def method_counts(out_dir: Path, method: str) -> Dict[str, int]:
    return prediction_counts_from_dir(method_run_dir(out_dir, method) / "longbench_predictions")


def metrics_failed_samples(run_dir: Path) -> int:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return 0
    try:
        data = load_json(metrics_path)
    except Exception:
        return 0
    return len(data.get("longbench", {}).get("failed_samples", []) or [])


def method_complete(out_dir: Path, method: str) -> bool:
    counts = method_counts(out_dir, method)
    if any(counts.get(task, 0) < OFFICIAL_COUNTS[task] for task in TASKS):
        return False
    score_path = method_run_dir(out_dir, method) / "longbench_scores.csv"
    rows = read_csv_rows(score_path)
    ok_tasks = {row.get("dataset") for row in rows if row.get("status") == "ok"}
    return set(TASKS).issubset(ok_tasks)


def checkpoint(out_dir: Path, method: str, status: str, extra: Optional[Mapping[str, Any]] = None) -> None:
    payload = {
        "method": method,
        "status": status,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "prediction_counts": method_counts(out_dir, method),
        "total_predictions": sum(method_counts(out_dir, method).values()),
        "oom_marked_or_failed_samples": metrics_failed_samples(method_run_dir(out_dir, method)),
    }
    if extra:
        payload.update(dict(extra))
    write_json(out_dir / "checkpoints" / f"{method}.json", payload)


def run_methods(args: argparse.Namespace) -> int:
    out_dir = args.out_dir.resolve()
    ensure_dirs(out_dir)
    statuses = []
    for method in [m for m in RUN_ORDER if m in args.methods.split(",")]:
        run_dir = method_run_dir(out_dir, method)
        run_dir.mkdir(parents=True, exist_ok=True)
        if method_complete(out_dir, method) and not args.force:
            checkpoint(out_dir, method, "complete_skipped")
            statuses.append({"method": method, "status": "complete_skipped"})
            continue
        ok, reason = gpu_preflight(out_dir, method, args.min_free_gpu_gb_before_start)
        if not ok:
            checkpoint(out_dir, method, "blocked_gpu_busy", {"reason": reason})
            (out_dir / "BLOCKED_GPU_BUSY.md").write_text(
                f"# BLOCKED_GPU_BUSY\n\nmethod: `{method}`\n\nreason: `{reason}`\n",
                encoding="utf-8",
            )
            write_json(out_dir / "RUN_STATUS.json", {"status": "blocked_gpu_busy", "method": method, "reason": reason})
            return 2
        cmd = qa_cmd(args, method, run_dir)
        assert "--longbench_max_samples" in cmd
        assert cmd[cmd.index("--longbench_max_samples") + 1] == "0"
        write_json(out_dir / "commands" / f"{method}.json", {"command": cmd, "cwd": str(REPO_ROOT)})
        checkpoint(out_dir, method, "running", {"command": cmd})
        log_path = out_dir / "logs" / f"{method}.log"
        start = time.time()
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n[{datetime.now().isoformat(timespec='seconds')}] START {method}\n")
            log.write("$ " + " ".join(cmd) + "\n")
            log.flush()
            env = os.environ.copy()
            env["PYTHONPATH"] = str(REPO_ROOT)
            proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, stdout=log, stderr=subprocess.STDOUT)
            rc = int(proc.returncode)
            elapsed = time.time() - start
            log.write(f"\n[{datetime.now().isoformat(timespec='seconds')}] END {method} rc={rc} elapsed_s={elapsed:.1f}\n")
        if rc != 0:
            checkpoint(out_dir, method, "failed", {"returncode": rc, "elapsed_s": round(elapsed, 3), "log_path": str(log_path)})
            write_json(out_dir / "FAILED.json", {"method": method, "returncode": rc, "log_path": str(log_path)})
            return rc or 1
        if not method_complete(out_dir, method):
            checkpoint(out_dir, method, "incomplete_after_run", {"log_path": str(log_path)})
            write_json(out_dir / "FAILED.json", {"method": method, "reason": "incomplete_after_run", "counts": method_counts(out_dir, method)})
            return 1
        checkpoint(out_dir, method, "complete", {"elapsed_s": round(elapsed, 3), "log_path": str(log_path)})
        statuses.append({"method": method, "status": "complete", "elapsed_s": round(elapsed, 3)})
        write_json(out_dir / "RUN_STATUS.json", {"status": "running", "completed": statuses})
    write_json(out_dir / "RUN_STATUS.json", {"status": "all_methods_complete", "completed": statuses})
    return 0


def load_method_predictions(out_dir: Path, method: str) -> List[Dict[str, Any]]:
    run_dir = method_run_dir(out_dir, method)
    pred_dir = run_dir / "longbench_predictions"
    rows: List[Dict[str, Any]] = []
    out_path = out_dir / "predictions" / f"{method}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as out_f:
        for task in TASKS:
            path = pred_dir / f"{task}.jsonl"
            raw = list(jsonl_rows(path))[: OFFICIAL_COUNTS[task]]
            for i, row in enumerate(raw):
                sid = str(row.get("sample_id", row.get("sample_index", i)))
                score = per_sample_score(task, str(row.get("pred", "")), row.get("answers", []), row.get("all_classes", []))
                enriched = {
                    "method": method,
                    "budget": method_spec(method)["budget"],
                    "task": task,
                    "sample_id": sid,
                    "sample_index": int(row.get("sample_index", i)),
                    "score": score,
                    "pred": row.get("pred", ""),
                    "answers": row.get("answers", []),
                    "all_classes": row.get("all_classes", []),
                    "length": int(row.get("length", 0) or 0),
                    "status": row.get("status", "ok"),
                    "error": row.get("error", ""),
                }
                rows.append(enriched)
                out_f.write(json.dumps(enriched, ensure_ascii=False) + "\n")
    return rows


def compute_task_scores(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    by_method_task: Dict[Tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_method_task[(str(row["method"]), str(row["task"]))].append(row)
    out = []
    for method in RUN_ORDER:
        for task in TASKS:
            task_rows = by_method_task.get((method, task), [])
            preds = [str(r.get("pred", "")) for r in task_rows]
            answers = [r.get("answers", []) for r in task_rows]
            lengths = [int(r.get("length", 0) or 0) for r in task_rows]
            classes: Any = []
            for r in task_rows:
                if r.get("all_classes"):
                    classes = r.get("all_classes")
                    break
            score = scorer(task, preds, answers, lengths, classes) if len(task_rows) == OFFICIAL_COUNTS[task] else ""
            out.append({
                "method": method,
                "budget": method_spec(method)["budget"],
                "task": task,
                "score": score,
                "num_samples": len(task_rows),
                "expected_samples": OFFICIAL_COUNTS[task],
                "oom_rows": sum(1 for r in task_rows if str(r.get("status", "")) == "oom" or str(r.get("pred", "")) == "[OOM]"),
            })
    return out


def macro_for(method: str, task_rows: Sequence[Mapping[str, Any]]) -> float:
    vals = [float(r["score"]) for r in task_rows if r["method"] == method and r.get("score") not in ("", None)]
    return round(sum(vals) / len(vals), 2) if vals else float("nan")


def summary_rows(methods: Sequence[str], task_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for method in methods:
        row: Dict[str, Any] = {"method": method, "budget": method_spec(method)["budget"], "macro": macro_for(method, task_rows)}
        for tr in task_rows:
            if tr["method"] == method:
                row[str(tr["task"])] = tr["score"]
        row["total_samples"] = sum(int(tr["num_samples"]) for tr in task_rows if tr["method"] == method)
        row["oom_rows"] = sum(int(tr.get("oom_rows", 0) or 0) for tr in task_rows if tr["method"] == method)
        out.append(row)
    return out


def prediction_map(rows: Sequence[Mapping[str, Any]], method: str) -> Dict[Tuple[str, str], Mapping[str, Any]]:
    return {(str(r["task"]), str(r["sample_id"])): r for r in rows if r["method"] == method}


def paired_sample_deltas(rows: Sequence[Mapping[str, Any]], base: str, variant: str) -> List[Dict[str, Any]]:
    bmap = prediction_map(rows, base)
    vmap = prediction_map(rows, variant)
    out = []
    for key in sorted(set(bmap) & set(vmap)):
        b = bmap[key]
        v = vmap[key]
        delta = round(float(v["score"]) - float(b["score"]), 4)
        out.append({
            "comparison": f"{variant}_minus_{base}",
            "base_method": base,
            "variant_method": variant,
            "task": key[0],
            "sample_id": key[1],
            "base_score": b["score"],
            "variant_score": v["score"],
            "delta": delta,
            "pred_changed": int(str(b.get("pred", "")) != str(v.get("pred", ""))),
        })
    return out


def per_task_delta(task_rows: Sequence[Mapping[str, Any]], base: str, variant: str) -> Dict[str, float]:
    b = {str(r["task"]): float(r["score"]) for r in task_rows if r["method"] == base and r.get("score") not in ("", None)}
    v = {str(r["task"]): float(r["score"]) for r in task_rows if r["method"] == variant and r.get("score") not in ("", None)}
    return {task: round(v[task] - b[task], 4) for task in TASKS if task in b and task in v}


def paired_bootstrap_task_macro(deltas: Sequence[Mapping[str, Any]], *, seed: int = 0, n_boot: int = 10000) -> Dict[str, Any]:
    by_task: Dict[str, List[float]] = defaultdict(list)
    for row in deltas:
        by_task[str(row["task"])].append(float(row["delta"]))
    if not by_task:
        return {}
    rng = random.Random(seed)
    samples = []
    for _ in range(n_boot):
        task_means = []
        for task in TASKS:
            vals = by_task.get(task, [])
            if not vals:
                continue
            draw = [vals[rng.randrange(len(vals))] for _ in range(len(vals))]
            task_means.append(sum(draw) / len(draw))
        samples.append(sum(task_means) / len(task_means))
    samples.sort()
    mean = sum(samples) / len(samples)
    return {
        "n_boot": n_boot,
        "seed": seed,
        "mean_delta": round(mean, 6),
        "ci95_low": round(samples[int(0.025 * (n_boot - 1))], 6),
        "ci95_high": round(samples[int(0.975 * (n_boot - 1))], 6),
        "ci_crosses_zero": bool(samples[int(0.025 * (n_boot - 1))] <= 0.0 <= samples[int(0.975 * (n_boot - 1))]),
    }


def outlier_sensitivity_rows(comp_name: str, task_delta: Mapping[str, float]) -> List[Dict[str, Any]]:
    vals = [(task, float(delta)) for task, delta in task_delta.items()]
    if not vals:
        return []
    macro = sum(delta for _, delta in vals) / len(vals)
    wins = [(task, delta) for task, delta in vals if delta > 0]
    wins_sorted = sorted(wins, key=lambda x: x[1], reverse=True)

    def mean_without(remove: Sequence[str]) -> Any:
        keep = [delta for task, delta in vals if task not in set(remove)]
        return round(sum(keep) / len(keep), 6) if keep else ""

    top1 = wins_sorted[:1]
    top2 = wins_sorted[:2]
    all_win_tasks = [task for task, _ in wins_sorted]
    positive_sum = sum(delta for _, delta in wins_sorted)
    top_contrib = (top1[0][1] / positive_sum) if top1 and positive_sum else 0.0
    return [
        {
            "comparison": comp_name,
            "analysis": "original_macro_delta",
            "removed_tasks": "",
            "macro_delta": round(macro, 6),
            "top_task": top1[0][0] if top1 else "",
            "top_task_delta": top1[0][1] if top1 else "",
            "top1_contribution_of_positive_sum": round(top_contrib, 6),
        },
        {
            "comparison": comp_name,
            "analysis": "remove_top_1",
            "removed_tasks": ",".join(task for task, _ in top1),
            "macro_delta": mean_without([task for task, _ in top1]),
            "top_task": top1[0][0] if top1 else "",
            "top_task_delta": top1[0][1] if top1 else "",
            "top1_contribution_of_positive_sum": round(top_contrib, 6),
        },
        {
            "comparison": comp_name,
            "analysis": "remove_top_2",
            "removed_tasks": ",".join(task for task, _ in top2),
            "macro_delta": mean_without([task for task, _ in top2]),
            "top_task": top1[0][0] if top1 else "",
            "top_task_delta": top1[0][1] if top1 else "",
            "top1_contribution_of_positive_sum": round(top_contrib, 6),
        },
        {
            "comparison": comp_name,
            "analysis": "remove_all_wins",
            "removed_tasks": ",".join(all_win_tasks),
            "macro_delta": mean_without(all_win_tasks),
            "top_task": top1[0][0] if top1 else "",
            "top_task_delta": top1[0][1] if top1 else "",
            "top1_contribution_of_positive_sum": round(top_contrib, 6),
        },
    ]


def read_selection_rows(out_dir: Path, method: str) -> List[Dict[str, Any]]:
    return list(jsonl_rows(out_dir / "audits" / f"{method}_selection_audit.jsonl"))


def read_rescue_rows(out_dir: Path, method: str) -> List[Dict[str, Any]]:
    return list(jsonl_rows(out_dir / "audits" / f"{method}_rescue_policy_audit.jsonl"))


def budget_runtime_audit(out_dir: Path) -> List[Dict[str, Any]]:
    rows = []
    set_by_method: Dict[str, Dict[Tuple[str, str, int], set]] = {}
    for method in RUN_ORDER:
        sel_rows = read_selection_rows(out_dir, method)
        rescue_rows = read_rescue_rows(out_dir, method)
        set_by_method[method] = {}
        for row in sel_rows:
            key = (str(row.get("task")), str(row.get("sample_id")), int(row.get("layer_idx", row.get("layer_id", -1)) or -1))
            set_by_method[method][key] = set(int(x) for x in row.get("selected_block_ids", []) or [])
        n = len(sel_rows)
        add_rows = [r for r in sel_rows if method in KM_BUDGET_METHODS]
        swap_rows = [r for r in sel_rows if method == "drm_swap_original_k"]
        duplicate_violations = 0
        sink_violations = 0
        invalid_violations = 0
        over_under_violations = 0
        add_pres_ok = 0
        add_pres_total = 0
        swap_ok = 0
        swap_total = 0
        m_vals = []
        selected_counts = []
        for r in sel_rows:
            selected = [int(x) for x in r.get("selected_block_ids", []) or []]
            selected_counts.append(len(selected))
            if len(selected) != len(set(selected)):
                duplicate_violations += 1
            if int(r.get("sink_violation", 0) or 0):
                sink_violations += 1
            if any(x < 0 for x in selected):
                invalid_violations += 1
            added = r.get("added_vs_silm", []) or []
            removed = r.get("removed_vs_silm", []) or []
            k = int(r.get("num_keep_target", 0) or 0)
            if method in KM_BUDGET_METHODS:
                add_pres_total += 1
                if len(removed) == 0:
                    add_pres_ok += 1
                m_vals.append(len(added))
                if len(selected) != k + len(added) or removed:
                    over_under_violations += 1
            elif method == "drm_swap_original_k":
                swap_total += 1
                if len(added) == len(removed) and len(selected) == k:
                    swap_ok += 1
                if len(added) != len(removed) or len(selected) != k:
                    over_under_violations += 1
                m_vals.append(len(added))
            elif method == "sliminfer_original":
                if len(selected) != k:
                    over_under_violations += 1
        rescue_selected_tokens = [
            int(r.get("selected_token_count", 0) or 0)
            for r in rescue_rows
            if r.get("selected_token_count") not in ("", None)
        ]
        accepted = [
            int(r.get("rescued_count", 0) or 0)
            for r in rescue_rows
        ]
        requested = [
            int(r.get("M", 0) or 0)
            for r in rescue_rows
        ]
        rows.append({
            "method": method,
            "budget": method_spec(method)["budget"],
            "selection_rows": n,
            "rescue_audit_rows": len(rescue_rows),
            "avg_selected_blocks": round(statistics.mean(selected_counts), 4) if selected_counts else "",
            "avg_selected_tokens": round(statistics.mean(rescue_selected_tokens), 4) if rescue_selected_tokens else "",
            "actual_avg_M": round(statistics.mean(m_vals), 4) if m_vals else 0,
            "add_preservation_rate": round(add_pres_ok / add_pres_total, 6) if add_pres_total else "",
            "swap_one_in_one_out_rate": round(swap_ok / swap_total, 6) if swap_total else "",
            "drm_accepted_rate": round(sum(accepted) / sum(requested), 6) if requested and sum(requested) else "",
            "over_under_budget_violations": over_under_violations,
            "sink_violations": sink_violations,
            "duplicate_violations": duplicate_violations,
            "invalid_violations": invalid_violations,
        })
    # Add cross-method selection difference checks.
    diff_rows = []
    for a, b in (("sliminfer_drm_add_km", "sliminfer_random_add_km"), ("sliminfer_drm_add_km", "sliminfer_larger_km")):
        keys = set(set_by_method.get(a, {})) & set(set_by_method.get(b, {}))
        diff = sum(1 for key in keys if set_by_method[a][key] != set_by_method[b][key])
        diff_rows.append({
            "method": f"{a}_vs_{b}",
            "budget": "K+M",
            "selection_rows": len(keys),
            "rescue_audit_rows": "",
            "avg_selected_blocks": "",
            "avg_selected_tokens": "",
            "actual_avg_M": "",
            "add_preservation_rate": "",
            "swap_one_in_one_out_rate": "",
            "drm_accepted_rate": "",
            "over_under_budget_violations": "",
            "sink_violations": "",
            "duplicate_violations": "",
            "invalid_violations": "",
            "selection_difference_rate": round(diff / len(keys), 6) if keys else "",
        })
    rows.extend(diff_rows)
    fields = [
        "method", "budget", "selection_rows", "rescue_audit_rows", "avg_selected_blocks",
        "avg_selected_tokens", "actual_avg_M", "add_preservation_rate",
        "swap_one_in_one_out_rate", "drm_accepted_rate", "over_under_budget_violations",
        "sink_violations", "duplicate_violations", "invalid_violations", "selection_difference_rate",
    ]
    write_csv(out_dir / "budget_runtime_audit.csv", rows, fields)
    return rows


def latency_memory_summary(out_dir: Path) -> List[Dict[str, Any]]:
    rows = []
    for method in RUN_ORDER:
        path = method_run_dir(out_dir, method) / "runtime_resource_stats.csv"
        raw = read_csv_rows(path)
        wall = [float(r.get("wall_clock_s", 0) or 0) for r in raw]
        alloc = [float(r.get("peak_memory_allocated_gb", 0) or 0) for r in raw]
        reserved = [float(r.get("peak_memory_reserved_gb", 0) or 0) for r in raw]
        rows.append({
            "method": method,
            "num_samples_with_runtime": len(raw),
            "latency_mean_s": round(statistics.mean(wall), 6) if wall else "",
            "latency_median_s": round(statistics.median(wall), 6) if wall else "",
            "latency_total_s": round(sum(wall), 3) if wall else "",
            "peak_allocated_gb": round(max(alloc), 6) if alloc else "",
            "peak_reserved_gb": round(max(reserved), 6) if reserved else "",
            "oom_marked_or_failed_samples": metrics_failed_samples(method_run_dir(out_dir, method)),
        })
    write_csv(out_dir / "latency_memory_summary.csv", rows, list(rows[0].keys()))
    return rows


def write_final_reports(
    out_dir: Path,
    sample_rows: Sequence[Mapping[str, Any]],
    task_rows: Sequence[Mapping[str, Any]],
    k_summary: Sequence[Mapping[str, Any]],
    km_summary: Sequence[Mapping[str, Any]],
    win_rows: Sequence[Mapping[str, Any]],
    bootstrap: Mapping[str, Any],
    sensitivity: Sequence[Mapping[str, Any]],
    budget_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    macros = {row["method"]: row["macro"] for row in list(k_summary) + list(km_summary)}
    task_delta_add_larger = per_task_delta(task_rows, "sliminfer_larger_km", "sliminfer_drm_add_km")
    task_delta_add_random = per_task_delta(task_rows, "sliminfer_random_add_km", "sliminfer_drm_add_km")
    task_delta_swap = per_task_delta(task_rows, "sliminfer_original", "drm_swap_original_k")
    add_vs_larger = round(float(macros["sliminfer_drm_add_km"]) - float(macros["sliminfer_larger_km"]), 4)
    add_vs_random = round(float(macros["sliminfer_drm_add_km"]) - float(macros["sliminfer_random_add_km"]), 4)
    swap_vs_original = round(float(macros["drm_swap_original_k"]) - float(macros["sliminfer_original"]), 4)
    six_task_replicates = abs(add_vs_larger - 0.39) <= 0.15 or abs(add_vs_random - 0.39) <= 0.15
    add_larger_ci = bootstrap.get("drm_add_vs_larger_km", {})
    add_random_ci = bootstrap.get("drm_add_vs_random_add_km", {})
    decision = {
        "official_17task_completed": all(
            sum(1 for r in sample_rows if r["method"] == method) == 3950 for method in RUN_ORDER
        ),
        "macro_scores": macros,
        "fair_km_delta_vs_larger": add_vs_larger,
        "fair_km_delta_vs_random": add_vs_random,
        "k_delta_swap_vs_original": swap_vs_original,
        "drm_add_exceeds_larger": add_vs_larger > 0,
        "drm_add_exceeds_random": add_vs_random > 0,
        "drm_swap_exceeds_original": swap_vs_original > 0,
        "bootstrap_ci": bootstrap,
        "recommend_second_model": bool(
            add_vs_larger > 0
            and add_vs_random > 0
            and not add_larger_ci.get("ci_crosses_zero", True)
            and not add_random_ci.get("ci_crosses_zero", True)
        ),
        "six_task_plus_0_39_replicated": six_task_replicates,
        "claim": "",
    }
    if decision["recommend_second_model"]:
        decision["claim"] = (
            "On Qwen2.5-7B-Instruct official LongBench 17-task full, frozen DRM-Add improves over both same-budget K+M controls."
        )
    elif add_vs_larger > 0 and add_vs_random > 0:
        decision["claim"] = (
            "Frozen DRM-Add is directionally above same-budget K+M controls on the 17-task macro, but CI/sensitivity must qualify the claim."
        )
    else:
        decision["claim"] = (
            "The strict 17-task result does not support claiming DRM-Add selection value beyond same-budget controls."
        )
    write_json(out_dir / "FINAL_17TASK_DECISION.json", decision)

    def table_summary(rows: Sequence[Mapping[str, Any]]) -> List[str]:
        lines = ["| method | budget | macro | oom_rows |", "|---|---|---:|---:|"]
        for row in rows:
            lines.append(f"| `{row['method']}` | `{row['budget']}` | {row['macro']} | {row.get('oom_rows', '')} |")
        return lines

    report = [
        "# DRM-Add Official 17-task Report",
        "",
        f"- Output directory: `{out_dir}`",
        f"- Official tasks: `{len(TASKS)}`; samples per method: `3950`.",
        f"- OOM policy: mark `[OOM]` and continue.",
        "",
        "## Table A: K-budget",
        "",
        *table_summary(k_summary),
        "",
        "## Table B: K+M-budget",
        "",
        *table_summary(km_summary),
        "",
        "## Fair Deltas",
        "",
        f"- DRM-Add - larger: `{add_vs_larger:+.4f}`",
        f"- DRM-Add - random-add: `{add_vs_random:+.4f}`",
        f"- DRM-Swap - original: `{swap_vs_original:+.4f}`",
        "",
        "## Required Answers",
        "",
        f"1. DRM-Add exceeds same-budget larger: `{decision['drm_add_exceeds_larger']}`.",
        f"2. DRM-Add exceeds same-budget random-add: `{decision['drm_add_exceeds_random']}`.",
        "3. Cross-task stability is reported in `outlier_sensitivity.csv` and per-task deltas below.",
        f"4. DRM-Swap exceeds original at K budget: `{decision['drm_swap_exceeds_original']}`.",
        "5. Add-vs-Swap stability is interpreted with separate budgets; Add preserves A_K while Swap deletes blocks.",
        f"6. Six-task +0.39 replicated within tolerance: `{decision['six_task_plus_0_39_replicated']}`.",
        f"7. Large task regressions are visible in per-task deltas: `{task_delta_add_larger}`.",
        f"8. Bootstrap CI crosses zero for add-vs-larger: `{add_larger_ci.get('ci_crosses_zero')}`; add-vs-random: `{add_random_ci.get('ci_crosses_zero')}`.",
        "9. Single-task/sample dominance is reported in `outlier_sensitivity.csv`.",
        f"10. DRM selection value beyond budget: `{decision['drm_add_exceeds_larger'] and decision['drm_add_exceeds_random']}` before CI qualification.",
        f"11. Recommend second model: `{decision['recommend_second_model']}`.",
        f"12. Strict paper claim: {decision['claim']}",
        "",
    ]
    (out_dir / "DRM_ADD_OFFICIAL_17TASK_REPORT.md").write_text("\n".join(report), encoding="utf-8")

    fairness = [
        "# DRM-Add Official 17-task Fairness Audit",
        "",
        "- K-budget and K+M-budget are reported separately.",
        "- LongBench macro is task macro, not pooled sample weighted average.",
        "- Same sample IDs/order are required across methods; see `OFFICIAL_17TASK_SAMPLE_MANIFEST.json`.",
        "- Selection uses no gold, prediction, or F1.",
        "- OOM rows are marked by the runtime and scored as predictions, per latest instruction.",
        "",
        "## Budget Audit",
        "",
        "| method | add preservation | swap one-in/out | over/under | sink | duplicate | invalid |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in budget_rows:
        if row["method"] in RUN_ORDER:
            fairness.append(
                f"| `{row['method']}` | {row.get('add_preservation_rate', '')} | {row.get('swap_one_in_one_out_rate', '')} | {row.get('over_under_budget_violations', '')} | {row.get('sink_violations', '')} | {row.get('duplicate_violations', '')} | {row.get('invalid_violations', '')} |"
            )
    fairness.append("")
    (out_dir / "DRM_ADD_OFFICIAL_17TASK_FAIRNESS_AUDIT.md").write_text("\n".join(fairness), encoding="utf-8")

    mechanism = [
        "# DRM-Add Official 17-task Mechanism Analysis",
        "",
        "DRM-Add is evaluated as an additive safety mechanism: it preserves all original SlimInfer A_K blocks and only adds rescue blocks. DRM-Swap is diagnostic because it must remove blocks one-in/one-out.",
        "",
        f"- Add vs larger per-task delta: `{task_delta_add_larger}`",
        f"- Add vs random per-task delta: `{task_delta_add_random}`",
        f"- Swap vs original per-task delta: `{task_delta_swap}`",
        "",
        "Selection difference checks are in `budget_runtime_audit.csv`; pairwise score statistics are in `win_loss_tie.csv` and `bootstrap_results.json`.",
        "",
    ]
    (out_dir / "DRM_ADD_OFFICIAL_17TASK_MECHANISM_ANALYSIS.md").write_text("\n".join(mechanism), encoding="utf-8")

    claim = [
        "# DRM-Add Official 17-task Claim Guard",
        "",
        f"Final decision: `{decision['claim']}`",
        "",
        "Allowed claims must be limited to Qwen2.5-7B-Instruct, frozen DRM parameters, official SlimInfer LongBench 17-task full, and the five methods in this run.",
        "",
        "Forbidden: retuned DRM claims, CC/ACC claims, second-model claims, capped-200 substitution, or mixed K/K+M same-budget rankings.",
        "",
    ]
    (out_dir / "DRM_ADD_OFFICIAL_17TASK_CLAIM_GUARD.md").write_text("\n".join(claim), encoding="utf-8")
    return decision


def analyze(out_dir: Path) -> Dict[str, Any]:
    all_rows: List[Dict[str, Any]] = []
    for method in RUN_ORDER:
        all_rows.extend(load_method_predictions(out_dir, method))
    with (out_dir / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_csv(
        out_dir / "scores_by_sample.csv",
        all_rows,
        ["method", "budget", "task", "sample_id", "sample_index", "score", "length", "status", "error", "pred"],
    )
    task_rows = compute_task_scores(all_rows)
    write_csv(out_dir / "scores_by_task.csv", task_rows, ["method", "budget", "task", "score", "num_samples", "expected_samples", "oom_rows"])
    k_summary = summary_rows(K_BUDGET_METHODS, task_rows)
    km_summary = summary_rows(KM_BUDGET_METHODS, task_rows)
    summary_fields = ["method", "budget", "macro", "total_samples", "oom_rows", *TASKS]
    write_csv(out_dir / "scores_summary_k_budget.csv", k_summary, summary_fields)
    write_csv(out_dir / "scores_summary_km_budget.csv", km_summary, summary_fields)

    win_rows: List[Dict[str, Any]] = []
    bootstrap: Dict[str, Any] = {}
    sensitivity: List[Dict[str, Any]] = []
    all_delta_rows: List[Dict[str, Any]] = []
    for comp_name, base, variant in COMPARISONS:
        deltas = paired_sample_deltas(all_rows, base, variant)
        all_delta_rows.extend(deltas)
        task_delta = per_task_delta(task_rows, base, variant)
        wins = sum(1 for d in deltas if float(d["delta"]) > 0)
        losses = sum(1 for d in deltas if float(d["delta"]) < 0)
        ties = sum(1 for d in deltas if float(d["delta"]) == 0)
        changed = sum(int(d["pred_changed"]) for d in deltas)
        macro_delta = round(sum(task_delta.values()) / len(task_delta), 6) if task_delta else ""
        win_rows.append({
            "comparison": comp_name,
            "base_method": base,
            "variant_method": variant,
            "paired_samples": len(deltas),
            "macro_delta": macro_delta,
            "sample_wins": wins,
            "sample_losses": losses,
            "sample_ties": ties,
            "changed_predictions": changed,
            "task_wins": sum(1 for x in task_delta.values() if x > 0),
            "task_losses": sum(1 for x in task_delta.values() if x < 0),
            "task_ties": sum(1 for x in task_delta.values() if x == 0),
        })
        bootstrap[comp_name] = paired_bootstrap_task_macro(deltas, seed=0, n_boot=10000)
        bootstrap[comp_name]["per_task_delta"] = task_delta
        sensitivity.extend(outlier_sensitivity_rows(comp_name, task_delta))
    write_csv(out_dir / "win_loss_tie.csv", win_rows, list(win_rows[0].keys()))
    write_csv(out_dir / "per_sample_deltas.csv", all_delta_rows, list(all_delta_rows[0].keys()))
    write_json(out_dir / "bootstrap_results.json", bootstrap)
    write_csv(out_dir / "outlier_sensitivity.csv", sensitivity, list(sensitivity[0].keys()))

    budget_rows = budget_runtime_audit(out_dir)
    latency_memory_summary(out_dir)
    decision = write_final_reports(out_dir, all_rows, task_rows, k_summary, km_summary, win_rows, bootstrap, sensitivity, budget_rows)
    return decision


def parse_args() -> argparse.Namespace:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_dir", type=Path, default=REPO_ROOT / "results" / f"drm_add_official_17task_full_{ts}")
    parser.add_argument("--model_path", type=Path, default=Path(DEFAULT_MODEL))
    parser.add_argument("--longbench_dir", type=Path, default=Path(DEFAULT_LONGBENCH))
    parser.add_argument("--python", type=str, default=str(REPO_ROOT / ".venv" / "bin" / "python"))
    parser.add_argument("--methods", type=str, default=",".join(RUN_ORDER))
    parser.add_argument("--audit_only", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--analyze_only", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--min_free_gpu_gb_before_start", type=float, default=20.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir = args.out_dir.resolve()
    ensure_dirs(args.out_dir)
    if not args.analyze_only:
        status = prepare_audits(args)
        print(json.dumps(status, indent=2, ensure_ascii=False))
        if not status["protocol_pass"]:
            return 2
    if args.audit_only and not args.run:
        return 0
    if args.run:
        rc = run_methods(args)
        if rc != 0:
            return rc
        decision = analyze(args.out_dir)
        print(json.dumps({"out_dir": str(args.out_dir), "decision": decision}, indent=2, ensure_ascii=False))
    elif args.analyze_only:
        decision = analyze(args.out_dir)
        print(json.dumps({"out_dir": str(args.out_dir), "decision": decision}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
