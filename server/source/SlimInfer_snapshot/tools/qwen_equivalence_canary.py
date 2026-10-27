#!/usr/bin/env python3
"""Plan, run, and audit the Stage 1 Qwen Original equivalence canary."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

_REPO_BOOTSTRAP = Path(__file__).resolve().parents[1]
if str(_REPO_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(_REPO_BOOTSTRAP))

from tools.qwen_pipeline_common import (
    DATA_ROOT,
    PROTOCOL_DIR,
    PROTOCOL_ID,
    PYTHON,
    QWEN_DRM_ROOT,
    QWEN_MODEL_ROOT,
    REPO_ROOT,
    atomic_write_json,
    atomic_write_text,
    canonical_bytes,
    canonical_sample_identity,
    implementation_revision,
    iter_jsonl,
    load_json,
    load_task_samples,
    now_iso,
    prediction_status,
    read_stage_decision,
    run_command,
    sha256_bytes,
    sha256_file,
)
from tools.qwen_resource_guard import collect_snapshot, monitor_resources, write_blocker


CANARY_INDICES: Dict[str, List[int]] = {
    "qasper": [75, 123],
    "hotpotqa": [72, 49],
    "2wikimqa": [167, 176],
    "musique": [138, 182],
    "gov_report": [180, 67],
    "repobench-p": [418, 224],
}
CANARY_TASKS = tuple(CANARY_INDICES)
RUNTIME_FILES = (
    REPO_ROOT / "experiments/run_qwen25_sliminfer_variants.py",
    REPO_ROOT / "utils/cache_utils.py",
    REPO_ROOT / "utils/processor.py",
    REPO_ROOT / "models/modeling_qwen2.py",
)
IMPLEMENTATION_FILES = RUNTIME_FILES + tuple(
    REPO_ROOT / relative for relative in (
        "tools/qwen_pipeline_common.py",
        "tools/qwen_resource_guard.py",
        "tools/qwen_protocol_attest.py",
        "tools/qwen_stage_preflight.py",
        "tools/qwen_equivalence_canary.py",
        "tools/run_qwen_unified_task.py",
        "tools/run_qwen_unified_pipeline.py",
        "tools/analyze_qwen_unified_suite.py",
        "tools/export_qwen_unified_suite.py",
        "tools/run_qwen_oom_fill.py",
        "configs/qwen_machine_safety_policy.json",
    )
)
NO_CHAT_TASKS = {"trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"}
MAX_MODEL_LEN = 131072


def _historical_rows(task: str) -> List[Dict[str, Any]]:
    path = (
        QWEN_DRM_ROOT / "run_sliminfer_original/longbench_predictions" / f"{task}.jsonl"
    )
    return list(iter_jsonl(path))


def build_plan() -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    lengths: List[int] = []
    for task, indices in CANARY_INDICES.items():
        samples = load_task_samples(task)
        historical = _historical_rows(task)
        for position, index in enumerate(indices):
            sample_id, content_hash = canonical_sample_identity(task, index, samples[index])
            historical_row = historical[index]
            context_length = int(historical_row.get("length", 0) or 0)
            lengths.append(context_length)
            role = "short_or_medium" if position == 0 else "long"
            if task == "repobench-p" and index == 224:
                role = "longest_reproducible_historical_original_success"
            rows.append({
                "task": task,
                "sample_index": index,
                "sample_id": sample_id,
                "content_sha256": content_hash,
                "historical_context_tokens": context_length,
                "historical_status": prediction_status(historical_row),
                "historical_prediction_sha256": sha256_bytes(
                    str(historical_row.get("pred", "")).encode("utf-8")
                ),
                "coverage_role": role,
            })
    return {
        "created_at": now_iso(),
        "protocol_id": PROTOCOL_ID,
        "sample_count": len(rows),
        "task_count": len(CANARY_TASKS),
        "tasks": list(CANARY_TASKS),
        "length_range_tokens": [min(lengths), max(lengths)],
        "selection_policy": (
            "Two frozen, previously successful Original samples per required task. The final "
            "repobench-p sample is the longest historical Original success that reproduced "
            "in an isolated clean resource-gated run (34,855 tokens). The longer 39,268-token "
            "historical success was excluded after deterministic OOM under both the current "
            "implementation and the clean base commit; see stage_01_diagnostics."
        ),
        "samples": rows,
    }


def _selection_spec(plan: Mapping[str, Any]) -> str:
    by_task: Dict[str, List[int]] = {}
    for row in plan["samples"]:
        by_task.setdefault(str(row["task"]), []).append(int(row["sample_index"]))
    return ";".join(
        f"{task}:{','.join(str(index) for index in by_task[task])}" for task in CANARY_TASKS
    )


def _runtime_revision() -> str:
    return implementation_revision(RUNTIME_FILES)


def freeze_implementation_state(pipeline_dir: Path) -> Dict[str, Any]:
    diff = run_command(["git", "diff", "--binary"])
    stat = run_command(["git", "diff", "--stat"])
    status = run_command(["git", "status", "--short"])
    commit = run_command(["git", "rev-parse", "HEAD"])["output"].strip()
    if any(result["returncode"] != 0 for result in (diff, stat, status)):
        raise RuntimeError("Unable to freeze the pre-canary git state")
    atomic_write_text(pipeline_dir / "IMPLEMENTATION_PATCH.diff", diff["output"])
    atomic_write_text(pipeline_dir / "IMPLEMENTATION_DIFF_STAT.txt", stat["output"])
    atomic_write_text(pipeline_dir / "IMPLEMENTATION_GIT_STATUS.txt", status["output"])
    files = []
    for path in IMPLEMENTATION_FILES:
        files.append({
            "path": str(path.relative_to(REPO_ROOT)),
            "exists": path.exists(),
            "sha256": sha256_file(path) if path.exists() else None,
        })
    revision = {
        "created_at": now_iso(),
        "protocol_id": PROTOCOL_ID,
        "base_commit": commit,
        "dirty": bool(status["output"].strip()),
        "git_status_sha256": sha256_bytes(status["output"].encode("utf-8")),
        "tracked_diff_sha256": sha256_bytes(diff["output"].encode("utf-8")),
        "tracked_diff_stat": stat["output"],
        "runtime_revision": _runtime_revision(),
        "files": files,
        "note": "No commit or reset was performed. Untracked implementation files are identified by git status and frozen by individual SHA256.",
    }
    atomic_write_json(pipeline_dir / "IMPLEMENTATION_REVISION.json", revision)
    return revision


def _ids_sha256(values: Sequence[int]) -> str:
    return sha256_bytes(",".join(str(int(value)) for value in values).encode("ascii"))


def build_expected_inputs(plan: Mapping[str, Any]) -> Dict[str, Any]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_ROOT, trust_remote_code=True)
    prompt_templates = load_json(REPO_ROOT / "configs/dataset2prompt.json")
    task_max_new_tokens = load_json(REPO_ROOT / "configs/dataset2maxlen.json")
    expected_rows: List[Dict[str, Any]] = []
    for planned in plan["samples"]:
        task = str(planned["task"])
        sample_index = int(planned["sample_index"])
        sample = load_task_samples(task)[sample_index]
        prompt = prompt_templates[task].format(**sample)
        raw_ids = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        if int(raw_ids.shape[0]) > MAX_MODEL_LEN:
            half = MAX_MODEL_LEN // 2
            prompt = (
                tokenizer.decode(raw_ids[:half], skip_special_tokens=True)
                + tokenizer.decode(raw_ids[-half:], skip_special_tokens=True)
            )
        if task not in NO_CHAT_TASKS:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        input_ids = [
            int(value) for value in tokenizer(
                prompt, truncation=False, return_tensors="pt"
            ).input_ids[0].tolist()
        ]
        generation_config = {
            "max_new_tokens": int(task_max_new_tokens[task]),
            "num_beams": 1,
            "do_sample": False,
            "use_cache": True,
            "eos_token_id": tokenizer.eos_token_id,
            "temperature_effective": None,
            "top_p_effective": None,
        }
        expected_rows.append({
            "task": task,
            "sample_index": sample_index,
            "sample_id": planned["sample_id"],
            "prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
            "input_token_ids_sha256": _ids_sha256(input_ids),
            "input_tokens": len(input_ids),
            "historical_input_tokens": int(planned["historical_context_tokens"]),
            "historical_input_tokens_match": len(input_ids) == int(planned["historical_context_tokens"]),
            "max_new_tokens": int(task_max_new_tokens[task]),
            "generation_config": generation_config,
            "generation_config_sha256": sha256_bytes(canonical_bytes(generation_config)),
        })
    return {
        "created_at": now_iso(),
        "protocol_id": PROTOCOL_ID,
        "tokenizer_path": str(QWEN_MODEL_ROOT),
        "tokenizer_class": tokenizer.__class__.__name__,
        "chat_template_sha256": sha256_bytes(str(tokenizer.chat_template or "").encode("utf-8")),
        "all_historical_input_lengths_match": all(
            row["historical_input_tokens_match"] for row in expected_rows
        ),
        "samples": expected_rows,
    }


def _parse_command(command: Sequence[str]) -> Dict[str, Any]:
    parsed: Dict[str, Any] = {}
    index = 2
    while index < len(command):
        token = command[index]
        if not token.startswith("--"):
            index += 1
            continue
        value: Any = True
        if index + 1 < len(command) and not command[index + 1].startswith("--"):
            value = command[index + 1]
            index += 1
        parsed[token[2:]] = value
        index += 1
    return parsed


def generation_identity(commands: Mapping[str, Sequence[str]]) -> Dict[str, Any]:
    off = _parse_command(commands["trace_off"])
    full = _parse_command(commands["trace_full"])
    semantic_fields = (
        "model_path", "tokenizer_path", "model_family", "dtype", "device", "backend",
        "longbench_dir", "longbench_tasks", "longbench_sample_indices_by_task",
        "longbench_max_samples", "max_model_len", "method", "eval", "block_strategy",
        "selection_path", "block_size", "token_unit_size", "local_query_window",
        "swap_threshold", "prune_layers", "retain_tokens", "decode_policy",
        "rescue_policy", "rescue_score", "rescue_ratio", "alpha_diff", "lambda_red",
        "eta_rescue", "eta_active", "temperature", "top_p",
        "generation_max_new_tokens", "seed", "protocol_id", "source_config_hash",
    )
    off_full_mismatches = {
        field: {"trace_off": off.get(field), "trace_full": full.get(field)}
        for field in semantic_fields if off.get(field) != full.get(field)
    }
    canonical_config = load_json(
        QWEN_DRM_ROOT / "run_sliminfer_original/config.json"
    )
    canonical_args = canonical_config["args"]
    historical_mapping = {
        "model_path": "model_path", "tokenizer_path": "tokenizer_path",
        "dtype": "dtype", "backend": "backend", "longbench_dir": "longbench_dir",
        "max_model_len": "max_model_len", "method": "method", "block_strategy": "block_strategy",
        "block_size": "block_size", "token_unit_size": "token_unit_size",
        "local_query_window": "local_query_window", "swap_threshold": "swap_threshold",
        "prune_layers": "prune_layers", "retain_tokens": "retain_tokens",
        "decode_policy": "decode_policy", "rescue_policy": "rescue_policy",
        "rescue_score": "rescue_score", "rescue_ratio": "rescue_ratio",
        "alpha_diff": "alpha_diff", "lambda_red": "lambda_red",
        "eta_rescue": "eta_rescue", "eta_active": "eta_active",
        "temperature": "temperature", "top_p": "top_p",
        "generation_max_new_tokens": "generation_max_new_tokens", "seed": "seed",
    }
    canonical_mismatches: Dict[str, Any] = {}
    for command_field, canonical_field in historical_mapping.items():
        actual = off.get(command_field)
        expected = canonical_args.get(canonical_field)
        if str(actual) != str(expected):
            canonical_mismatches[command_field] = {"expected": expected, "actual": actual}
    expected_backend = canonical_config.get("attn_implementation")
    if off.get("backend") != expected_backend:
        canonical_mismatches["attn_implementation"] = {
            "expected": expected_backend, "actual": off.get("backend")
        }
    return {
        "trace_off_full_semantic_fields_exact": not off_full_mismatches,
        "canonical_historical_config_exact": not canonical_mismatches,
        "trace_off_full_mismatches": off_full_mismatches,
        "canonical_mismatches": canonical_mismatches,
        "semantic_fields": list(semantic_fields),
        "explicit_selection_path": off.get("selection_path"),
        "historical_selection_path_note": "canonical_fixed_page explicitly reproduces the historical cc_selection_audit_path fixed-page behavior without enabling the legacy audit flag",
    }


def build_command(
    pipeline_dir: Path,
    plan: Mapping[str, Any],
    variant: str,
) -> List[str]:
    if variant not in {"trace_off", "trace_full"}:
        raise ValueError(variant)
    run_dir = pipeline_dir / "stage_01_canary" / variant
    method_hash = load_json(PROTOCOL_DIR / "method_configs.json")["methods"][
        "sliminfer_original"
    ]["method_config_sha256"]
    command = [
        str(PYTHON),
        str(REPO_ROOT / "experiments/run_qwen25_sliminfer_variants.py"),
        "--model_path", str(QWEN_MODEL_ROOT),
        "--tokenizer_path", str(QWEN_MODEL_ROOT),
        "--model_family", "qwen2",
        "--dtype", "fp16",
        "--device", "cuda",
        "--backend", "flash_attention_2",
        "--longbench_dir", str(DATA_ROOT),
        "--longbench_tasks", ",".join(CANARY_TASKS),
        "--longbench_sample_indices_by_task", _selection_spec(plan),
        "--longbench_max_samples", "0",
        "--max_model_len", "131072",
        "--method", "sliminfer_original",
        "--eval", "longbench",
        "--block_strategy", "fixed",
        "--selection_path", "canonical_fixed_page",
        "--block_size", "64",
        "--token_unit_size", "8",
        "--local_query_window", "4",
        "--swap_threshold", "0.9",
        "--prune_layers", "9,18,26",
        "--retain_tokens", "12288,6144,4096",
        "--decode_policy", "original",
        "--rescue_policy", "none",
        "--rescue_score", "delta_o",
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
        "--protocol_id", PROTOCOL_ID,
        "--implementation_revision", _runtime_revision(),
        "--source_config_hash", method_hash,
        "--selection_digest_dir", str(run_dir / "selection_digests"),
        "--active_block_trace", "full" if variant == "trace_full" else "none",
        "--output_dir", str(run_dir),
    ]
    if variant == "trace_full":
        command.extend(["--active_block_trace_dir", str(run_dir / "active_block_trace")])
    return command


def _read_gzip_jsonl(path: Path) -> List[Dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _event_key(row: Mapping[str, Any]) -> Tuple[Any, ...]:
    layer = row.get("abs_layer")
    if layer is None:
        layer = row.get("layer_idx", -1)
    return (
        row.get("task"), int(row.get("sample_index", -1)), row.get("phase"),
        int(row.get("decode_step", 0)), int(row.get("selection_event_index", -1)),
        int(layer),
    )


def _digest_ids(values: Iterable[Any]) -> str:
    text = ",".join(str(int(value)) for value in sorted(int(value) for value in values))
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def _load_predictions(run_dir: Path) -> Dict[Tuple[str, int], Dict[str, Any]]:
    result: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for task in CANARY_TASKS:
        for row in iter_jsonl(run_dir / "longbench_predictions" / f"{task}.jsonl"):
            key = (task, int(row["sample_index"]))
            if key in result:
                raise RuntimeError(f"Duplicate canary prediction: {key}")
            result[key] = row
    return result


def _load_digests(run_dir: Path) -> Dict[Tuple[Any, ...], Dict[str, Any]]:
    result: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for task in CANARY_TASKS:
        for row in _read_gzip_jsonl(run_dir / "selection_digests" / f"{task}.jsonl.gz"):
            key = _event_key(row)
            if key in result:
                raise RuntimeError(f"Duplicate selection digest event: {key}")
            result[key] = row
    return result


def audit_runs(
    pipeline_dir: Path,
    plan: Mapping[str, Any],
    expected_inputs: Mapping[str, Any],
    generation_audit: Mapping[str, Any],
) -> Dict[str, Any]:
    root = pipeline_dir / "stage_01_canary"
    off = _load_predictions(root / "trace_off")
    full = _load_predictions(root / "trace_full")
    expected: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for task in CANARY_TASKS:
        historical = _historical_rows(task)
        for index in CANARY_INDICES[task]:
            expected[(task, index)] = historical[index]
    expected_runtime = {
        (str(row["task"]), int(row["sample_index"])): row
        for row in expected_inputs["samples"]
    }
    prediction_fields = (
        "sample_id", "content_sha256", "prompt_sha256", "input_token_ids_sha256",
        "status", "pred", "input_tokens", "output_tokens", "source_config_hash",
        "max_new_tokens", "generation_config", "generation_config_sha256",
        "runtime_dtype", "attention_implementation", "tf32_matmul_allowed",
        "tf32_cudnn_allowed", "tokenizer_id",
    )
    sample_rows: List[Dict[str, Any]] = []
    for key in sorted(expected):
        off_row = off.get(key)
        full_row = full.get(key)
        historical = expected[key]
        expected_row = expected_runtime[key]
        field_mismatches = [
            field for field in prediction_fields
            if off_row is None or full_row is None or off_row.get(field) != full_row.get(field)
        ]
        historical_prediction_match = bool(
            off_row is not None
            and str(off_row.get("pred")) == str(historical.get("pred"))
            and prediction_status(off_row) == prediction_status(historical)
        )
        sample_id_match = bool(
            off_row is not None and full_row is not None
            and off_row.get("sample_id") == expected_row["sample_id"]
            and full_row.get("sample_id") == expected_row["sample_id"]
        )
        prompt_match = bool(
            off_row is not None and full_row is not None
            and off_row.get("prompt_sha256") == expected_row["prompt_sha256"]
            and full_row.get("prompt_sha256") == expected_row["prompt_sha256"]
        )
        token_ids_match = bool(
            off_row is not None and full_row is not None
            and off_row.get("input_token_ids_sha256") == expected_row["input_token_ids_sha256"]
            and full_row.get("input_token_ids_sha256") == expected_row["input_token_ids_sha256"]
        )
        input_length_match = bool(
            off_row is not None and full_row is not None
            and int(off_row.get("input_tokens", -1)) == int(expected_row["input_tokens"])
            and int(full_row.get("input_tokens", -1)) == int(expected_row["input_tokens"])
            and expected_row["historical_input_tokens_match"]
        )
        max_new_tokens_match = bool(
            off_row is not None and full_row is not None
            and int(off_row.get("max_new_tokens", -1)) == int(expected_row["max_new_tokens"])
            and int(full_row.get("max_new_tokens", -1)) == int(expected_row["max_new_tokens"])
        )
        generation_config_match = bool(
            off_row is not None and full_row is not None
            and off_row.get("generation_config_sha256") == expected_row["generation_config_sha256"]
            and full_row.get("generation_config_sha256") == expected_row["generation_config_sha256"]
            and off_row.get("generation_config") == expected_row["generation_config"]
            and full_row.get("generation_config") == expected_row["generation_config"]
        )
        runtime_backend_match = bool(
            off_row is not None and full_row is not None
            and off_row.get("runtime_dtype") == "torch.float16"
            and full_row.get("runtime_dtype") == "torch.float16"
            and off_row.get("attention_implementation") == "flash_attention_2"
            and full_row.get("attention_implementation") == "flash_attention_2"
            and off_row.get("tokenizer_id") == str(QWEN_MODEL_ROOT)
            and full_row.get("tokenizer_id") == str(QWEN_MODEL_ROOT)
        )
        sample_rows.append({
            "task": key[0],
            "sample_index": key[1],
            "trace_off_present": off_row is not None,
            "trace_full_present": full_row is not None,
            "trace_equivalent": not field_mismatches,
            "field_mismatches": field_mismatches,
            "sample_id_match": sample_id_match,
            "prompt_hash_match": prompt_match,
            "token_ids_match": token_ids_match,
            "input_length_match": input_length_match,
            "max_new_tokens_match": max_new_tokens_match,
            "generation_config_match": generation_config_match,
            "runtime_backend_match": runtime_backend_match,
            "canonical_prediction_match": historical_prediction_match,
        })
    off_digest = _load_digests(root / "trace_off")
    full_digest = _load_digests(root / "trace_full")
    digest_keys_match = set(off_digest) == set(full_digest)
    digest_mismatches = [
        key for key in sorted(set(off_digest) & set(full_digest))
        if off_digest[key] != full_digest[key]
    ]
    full_trace: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    full_trace_duplicate_keys: List[Tuple[Any, ...]] = []
    for task in CANARY_TASKS:
        path = root / "trace_full/active_block_trace" / f"{task}.jsonl.gz"
        for row in _read_gzip_jsonl(path):
            key = _event_key(row)
            if key in full_trace:
                full_trace_duplicate_keys.append(key)
            full_trace[key] = row
    trace_digest_mismatches: List[Tuple[Any, ...]] = []
    budget_or_sink_failures: List[Tuple[Any, ...]] = []
    duplicate_failures: List[Tuple[Any, ...]] = []
    for key, digest in full_digest.items():
        trace = full_trace.get(key)
        if trace is None or (
            digest["selected_ids_sha256"] != _digest_ids(trace.get("selected_block_ids", []))
            or digest["base_original_ids_sha256"]
            != _digest_ids(trace.get("base_original_active_ids", []))
        ):
            trace_digest_mismatches.append(key)
        if int(digest.get("budget_parity", 0)) != 1 or int(digest.get("sink_violation", 0)) != 0:
            budget_or_sink_failures.append(key)
        trace_ids = trace.get("selected_block_ids", []) if trace is not None else []
        if (
            int(digest.get("duplicate_violation", 1)) != 0
            or len(trace_ids) != len(set(int(value) for value in trace_ids))
        ):
            duplicate_failures.append(key)
    historical_trace_path = QWEN_DRM_ROOT / "audits/sliminfer_original_selection_audit.jsonl"
    historical_trace: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    planned = {(row["task"], int(row["sample_index"])) for row in plan["samples"]}
    for row in iter_jsonl(historical_trace_path):
        pair = (str(row.get("task")), int(row.get("sample_index", -1)))
        if pair in planned:
            old_layer = row.get("abs_layer")
            if old_layer is None:
                old_layer = row.get("layer_idx", -1)
            key = (
                pair[0], pair[1], str(row.get("phase", "prefill") or "prefill"),
                int(old_layer),
            )
            historical_trace[key] = row
    current_prefill_trace: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for row in full_trace.values():
        if str(row.get("phase", "")) != "prefill":
            continue
        layer = row.get("abs_layer")
        if layer is None:
            layer = row.get("layer_idx", -1)
        current_prefill_trace[(
            str(row.get("task")), int(row.get("sample_index", -1)), "prefill", int(layer)
        )] = row
    historical_selection_mismatches: List[Dict[str, Any]] = []
    for key, old in historical_trace.items():
        current = current_prefill_trace.get(key)
        if current is None:
            historical_selection_mismatches.append({"event": list(key), "reason": "missing_current_event"})
            continue
        if int(old.get("num_selected", -1)) != int(current.get("selected_count", -1)):
            historical_selection_mismatches.append({"event": list(key), "reason": "selected_count"})
        if sorted(old.get("selected_block_ids", [])) != sorted(current.get("selected_block_ids", [])):
            historical_selection_mismatches.append({"event": list(key), "reason": "selected_ids"})
    checks = {
        "sample_count_12": len(sample_rows) == 12,
        "six_required_tasks": set(CANARY_TASKS) == {row["task"] for row in plan["samples"]},
        "sample_ids_12_of_12": all(row["sample_id_match"] for row in sample_rows),
        "prompt_hashes_12_of_12": all(row["prompt_hash_match"] for row in sample_rows),
        "token_ids_12_of_12": all(row["token_ids_match"] for row in sample_rows),
        "historical_input_lengths_12_of_12": all(row["input_length_match"] for row in sample_rows),
        "task_max_new_tokens_12_of_12": all(row["max_new_tokens_match"] for row in sample_rows),
        "generation_config_12_of_12": all(row["generation_config_match"] for row in sample_rows),
        "runtime_dtype_attention_tokenizer_12_of_12": all(row["runtime_backend_match"] for row in sample_rows),
        "command_generation_semantics_exact": bool(
            generation_audit["trace_off_full_semantic_fields_exact"]
            and generation_audit["canonical_historical_config_exact"]
        ),
        "trace_off_full_prediction_fields_exact": all(row["trace_equivalent"] for row in sample_rows),
        "canonical_predictions_exact": all(row["canonical_prediction_match"] for row in sample_rows),
        "selection_digest_event_keys_exact": digest_keys_match,
        "selection_digest_rows_exact": not digest_mismatches,
        "full_trace_event_keys_unique": not full_trace_duplicate_keys,
        "full_trace_matches_digest": not trace_digest_mismatches and set(full_trace) == set(full_digest),
        "selected_counts_exact": all(
            off_digest[key].get("selected_count") == full_digest[key].get("selected_count")
            for key in set(off_digest) & set(full_digest)
        ) and digest_keys_match,
        "budget_and_sink_valid": not budget_or_sink_failures,
        "duplicates_zero": not duplicate_failures,
        "historical_prefill_selected_ids_exact": bool(historical_trace) and not historical_selection_mismatches,
    }
    return {
        "created_at": now_iso(),
        "protocol_id": PROTOCOL_ID,
        "implementation_revision": _runtime_revision(),
        "decision": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "generation_identity": dict(generation_audit),
        "expected_input_manifest": str(pipeline_dir / "ORIGINAL_EQUIVALENCE_EXPECTED_INPUTS.json"),
        "samples": sample_rows,
        "selection_event_counts": {
            "trace_off_digest": len(off_digest),
            "trace_full_digest": len(full_digest),
            "full_trace": len(full_trace),
            "historical_prefill": len(historical_trace),
        },
        "digest_mismatches": [list(key) for key in digest_mismatches],
        "trace_digest_mismatches": [list(key) for key in trace_digest_mismatches],
        "full_trace_duplicate_keys": [list(key) for key in full_trace_duplicate_keys],
        "budget_or_sink_failures": [list(key) for key in budget_or_sink_failures],
        "duplicate_failures": [list(key) for key in duplicate_failures],
        "historical_selection_mismatches": historical_selection_mismatches,
    }


def _write_report(pipeline_dir: Path, report: Mapping[str, Any]) -> None:
    atomic_write_json(pipeline_dir / "ORIGINAL_EQUIVALENCE_CANARY.json", report)
    lines = [
        "# Original Equivalence Canary",
        "",
        f"- Decision: **{report['decision']}**",
        f"- Protocol: `{report['protocol_id']}`",
        f"- Implementation revision: `{report['implementation_revision']}`",
        f"- Samples: `{len(report['samples'])}` across `{len(CANARY_TASKS)}` tasks",
        "",
        "| check | pass |",
        "|---|---|",
    ]
    lines.extend(f"| `{name}` | `{passed}` |" for name, passed in report["checks"].items())
    atomic_write_text(
        pipeline_dir / "ORIGINAL_EQUIVALENCE_CANARY_REPORT.md", "\n".join(lines) + "\n"
    )
    decision = {
        "stage": 1,
        "name": "original_equivalence_canary",
        "created_at": now_iso(),
        "decision": report["decision"],
        "protocol_id": PROTOCOL_ID,
        "implementation_revision": report["implementation_revision"],
        "next_stage": 2 if report["decision"] == "PASS" else None,
    }
    atomic_write_json(pipeline_dir / "STAGE_01_DECISION.json", decision)


def run_canary(pipeline_dir: Path, safety_policy: Path, *, execute: bool) -> Dict[str, Any]:
    stage0 = read_stage_decision(pipeline_dir, 0)
    if not stage0 or stage0.get("decision") != "PASS":
        raise RuntimeError("Stage 00 is not PASS; refusing the Stage 01 canary")
    implementation_state = freeze_implementation_state(pipeline_dir)
    plan = build_plan()
    atomic_write_json(pipeline_dir / "ORIGINAL_EQUIVALENCE_CANARY_PLAN.json", plan)
    commands = {
        variant: build_command(pipeline_dir, plan, variant)
        for variant in ("trace_off", "trace_full")
    }
    atomic_write_json(pipeline_dir / "stage_01_canary/COMMANDS.json", commands)
    generation_audit = generation_identity(commands)
    atomic_write_json(
        pipeline_dir / "ORIGINAL_EQUIVALENCE_GENERATION_IDENTITY.json",
        generation_audit,
    )
    if not execute:
        return {
            "decision": "NOT_RUN",
            "reason": "dry_run_only",
            "sample_count": plan["sample_count"],
            "implementation_revision": implementation_state,
            "generation_identity": generation_audit,
            "commands": commands,
        }
    policy = load_json(safety_policy)
    env = os.environ.copy()
    env.update({key: str(value) for key, value in policy["environment"].items() if key.isupper()})
    expected_inputs: Optional[Dict[str, Any]] = None
    for variant in ("trace_off", "trace_full"):
        if variant == "trace_full":
            cooldown_started = collect_snapshot()
            time.sleep(30)
            cooldown_completed = collect_snapshot()
            atomic_write_json(
                pipeline_dir / "stage_01_canary/INTER_VARIANT_COOLDOWN.json",
                {
                    "started_at": cooldown_started["captured_at"],
                    "completed_at": cooldown_completed["captured_at"],
                    "requested_seconds": 30,
                    "before": cooldown_started,
                    "after": cooldown_completed,
                    "note": "Cooldown follows trace-off process exit; the normal continuous resource gate still runs afterward.",
                },
            )
        gate = monitor_resources(
            policy,
            int(policy["thresholds"]["resource_gate_duration_seconds"]),
            int(policy["thresholds"]["resource_gate_interval_seconds"]),
            pipeline_dir=pipeline_dir,
            allowed_pids=(os.getpid(),),
        )
        atomic_write_json(pipeline_dir / f"stage_01_canary/{variant}_RESOURCE_GATE.json", gate)
        if not gate["pass"]:
            write_blocker(pipeline_dir, gate)
            raise RuntimeError(f"Resource gate blocked {variant}")
        if expected_inputs is None:
            expected_inputs = build_expected_inputs(plan)
            atomic_write_json(
                pipeline_dir / "ORIGINAL_EQUIVALENCE_EXPECTED_INPUTS.json",
                expected_inputs,
            )
            if not expected_inputs["all_historical_input_lengths_match"]:
                raise RuntimeError(
                    "Independent tokenizer reconstruction disagrees with historical context lengths"
                )
            if not (
                generation_audit["trace_off_full_semantic_fields_exact"]
                and generation_audit["canonical_historical_config_exact"]
            ):
                raise RuntimeError("Canary command generation semantics do not match canonical Original")
        log_path = pipeline_dir / f"logs/stage_01_{variant}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as log:
            proc = subprocess.Popen(
                commands[variant], cwd=REPO_ROOT, env=env, stdout=log,
                stderr=subprocess.STDOUT,
            )
            pid_path = pipeline_dir / "PID.json"
            pid_state = load_json(pid_path)
            pid_state.update({
                "updated_at": now_iso(),
                "gpu_worker_pid": proc.pid,
                "current_method": "sliminfer_original",
                "current_task": f"stage_01_{variant}",
            })
            atomic_write_json(pid_path, pid_state)
            try:
                returncode = proc.wait()
            finally:
                pid_state = load_json(pid_path)
                pid_state.update({
                    "updated_at": now_iso(),
                    "gpu_worker_pid": None,
                    "current_method": None,
                    "current_task": None,
                })
                atomic_write_json(pid_path, pid_state)
        if returncode != 0:
            raise RuntimeError(f"Canary {variant} failed with exit code {returncode}")
    if expected_inputs is None:
        raise RuntimeError("Expected input manifest was not created")
    report = audit_runs(pipeline_dir, plan, expected_inputs, generation_audit)
    _write_report(pipeline_dir, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-dir", type=Path, required=True)
    parser.add_argument("--safety-policy", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_canary(args.pipeline_dir, args.safety_policy, execute=args.execute)
    print(json.dumps(result, indent=2))
    return 0 if result.get("decision") in {"PASS", "NOT_RUN"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
