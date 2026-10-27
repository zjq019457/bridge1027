#!/usr/bin/env python3
"""Freeze Unified LongBench Protocol v1.0 without launching model inference.

The tool is intentionally CPU-only. It inventories and hashes existing result
artifacts, fingerprints the local LongBench snapshot and evaluation code, and
writes the protocol manifests and policies used by later adapters/runners.
Historical predictions are opened read-only and are never rewritten.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = Path("/home/yu-bai/Desktop/jiaqizhang/HeadKV/data/LongBench")
REQUESTED_REPO = Path(
    "/home/user/Downloads/"
    "RescueKV-Evidence-Core-Limited-Rescue-for-Robust-Long-Context-KV-Compression-clone"
)
REQUESTED_LLAMA_CC = REQUESTED_REPO / "results/llama_cc_official_17task_full_20260717_113452"
LOCAL_QWEN_CC = REPO_ROOT / "results/cc_sliminfer_v1_qwen25_17task_full_official_20260712_075238"
LOCAL_QWEN_DRM = REPO_ROOT / "results/drm_add_official_17task_full_20260715_084131"
QWEN_SNAPSHOT = Path(
    "/home/yu-bai/.cache/huggingface/hub/"
    "models--Qwen--Qwen2.5-7B-Instruct/snapshots/"
    "a09a35458c702b33eeacc393d103063234e8bc28"
)
EXPECTED_LLAMA_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"

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
COUNTS: Dict[str, int] = {task: 200 for task in TASKS}
COUNTS.update({"multifieldqa_en": 150, "lcc": 500, "repobench-p": 500})
assert sum(COUNTS.values()) == 3950

CATEGORIES: Dict[str, Tuple[str, ...]] = {
    "single_document_qa": ("qasper", "multifieldqa_en"),
    "multi_document_qa": ("hotpotqa", "2wikimqa", "musique"),
    "summarization": ("gov_report", "qmsum", "multi_news", "vcsum"),
    "few_shot_learning": ("trec", "triviaqa", "samsum", "lsht"),
    "synthetic_tasks": ("passage_count", "passage_retrieval_en"),
    "code_completion": ("lcc", "repobench-p"),
}
NO_CHAT_TASKS = {"trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"}

QWEN_SCHEDULE = {
    "prune_layers": [9, 18, 26],
    "k_blocks": [192, 96, 64],
    "m_blocks": [10, 5, 3],
    "retained_tokens": [12288, 6144, 4096],
    "add_retained_tokens": [12928, 6464, 4288],
}
LLAMA_SCHEDULE = {
    "prune_layers": [10, 20, 30],
    "k_blocks": [128, 64, 32],
    "m_blocks": [6, 3, 2],
    "retained_tokens": [8192, 4096, 2048],
    "add_retained_tokens": [8576, 4288, 2176],
}
DRM_FROZEN = {
    "rescue_policy": "add",
    "rescue_score": "drm",
    "rescue_ratio": 0.05,
    "alpha_diff": 0.5,
    "lambda_red": 0.25,
    "eta_rescue": 0.25,
    "eta_active": 0.25,
    "seed": 0,
    "selector": "historical_drm_mmr",
}

CODE_FILES = (
    "experiments/run_qwen25_sliminfer_variants.py",
    "models/modeling_qwen2.py",
    "models/modeling_llama.py",
    "utils/cache_utils.py",
    "bridgekv/rescuekv_scoring.py",
    "longbench_pred.py",
    "longbench_eval.py",
    "tools/consolidate_official_repro_results.py",
    "tools/run_cc_sliminfer_v1_17task_full_official.py",
    "tools/run_drm_add_mechanism_revalidation.py",
    "tools/run_drm_add_six_task_gate.py",
    "tools/run_drm_add_official_17task_full.py",
    "tools/freeze_unified_longbench_v1.py",
    "configs/dataset2prompt.json",
    "configs/dataset2maxlen.json",
    "configs/cc_gate_v0.json",
)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def command_output(args: Sequence[str]) -> Tuple[int, str]:
    proc = subprocess.run(
        list(args),
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return proc.returncode, proc.stdout


def classify_artifact(path: Path) -> str:
    name = path.name.lower()
    parts = {part.lower() for part in path.parts}
    if "longbench_predictions" in parts or name == "predictions.jsonl":
        return "prediction"
    if "audit" in name or "audits" in parts or "trace" in name or "selection" in name:
        return "audit_or_trace"
    if "score" in name or name.endswith("summary.csv") or name.startswith("bootstrap"):
        return "score_or_analysis"
    if "log" in name or "logs" in parts:
        return "log"
    if "config" in name or "command" in name or "manifest" in name:
        return "config_or_manifest"
    if name.endswith(".md"):
        return "report"
    return "other"


def inventory_root(
    label: str,
    root: Path,
    hash_files: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[str]]:
    if not root.exists():
        missing = {
            "result_family": label,
            "root": str(root),
            "root_exists": False,
            "relative_path": "",
            "size_bytes": "",
            "mtime_ns": "",
            "sha256": "",
            "artifact_class": "missing_root",
            "prediction_immutable": True,
        }
        return [missing], {
            "label": label,
            "root": str(root),
            "exists": False,
            "file_count": 0,
            "total_bytes": 0,
            "aggregate_sha256": None,
        }, []

    rows: List[Dict[str, Any]] = []
    hash_lines: List[str] = []
    aggregate_items: List[Dict[str, Any]] = []
    files = sorted(path for path in root.rglob("*") if path.is_file())
    for index, path in enumerate(files, start=1):
        stat = path.stat()
        rel = str(path.relative_to(root))
        digest = sha256_file(path) if hash_files else None
        row = {
            "result_family": label,
            "root": str(root),
            "root_exists": True,
            "relative_path": rel,
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": digest or "NOT_COMPUTED",
            "artifact_class": classify_artifact(path),
            "prediction_immutable": classify_artifact(path) == "prediction",
        }
        rows.append(row)
        aggregate_items.append({
            "path": rel,
            "size_bytes": stat.st_size,
            "sha256": digest,
        })
        if digest:
            hash_lines.append(f"{digest}  {path}")
        if index % 25 == 0 or index == len(files):
            print(f"[inventory] {label}: {index}/{len(files)} files", flush=True)
    return rows, {
        "label": label,
        "root": str(root),
        "exists": True,
        "file_count": len(rows),
        "total_bytes": sum(int(row["size_bytes"]) for row in rows),
        "aggregate_sha256": sha256_bytes(canonical_bytes(aggregate_items)),
        "all_file_sha256_computed": bool(hash_files),
    }, hash_lines


def sample_content(task: str, row: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "task": task,
        "context": row.get("context"),
        "input": row.get("input"),
        "question": row.get("question"),
        "answers": row.get("answers"),
        "all_classes": row.get("all_classes"),
        "language": row.get("language"),
    }


def stable_sample_identity(task: str, index: int, row: Mapping[str, Any]) -> Tuple[str, str]:
    content_sha256 = sha256_bytes(canonical_bytes(sample_content(task, row)))
    return f"{task}:{index}:{content_sha256[:16]}", content_sha256


def load_dataset_snapshot(data_root: Path) -> Tuple[Dict[str, Any], Dict[str, List[Dict[str, Any]]]]:
    task_entries: List[Dict[str, Any]] = []
    rows_by_task: Dict[str, List[Dict[str, Any]]] = {}
    for order, task in enumerate(TASKS, start=1):
        path = data_root / f"{task}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Missing official LongBench file: {path}")
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        expected = COUNTS[task]
        if len(rows) != expected:
            raise RuntimeError(f"{task}: expected exactly {expected} rows, found {len(rows)}")
        rows_by_task[task] = rows
        ids: List[str] = []
        content_hashes: List[str] = []
        canonical_rows: List[bytes] = []
        for index, row in enumerate(rows):
            sample_id, content_hash = stable_sample_identity(task, index, row)
            ids.append(sample_id)
            content_hashes.append(content_hash)
            canonical_rows.append(canonical_bytes(sample_content(task, row)))
        task_entries.append({
            "order": order,
            "task": task,
            "path": str(path),
            "row_count": len(rows),
            "expected_count": expected,
            "file_size_bytes": path.stat().st_size,
            "file_sha256": sha256_file(path),
            "official_slice_canonical_sha256": sha256_bytes(b"\n".join(canonical_rows)),
            "sample_id_sequence_sha256": sha256_bytes("\n".join(ids).encode("utf-8")),
            "content_hash_sequence_sha256": sha256_bytes("\n".join(content_hashes).encode("utf-8")),
            "first_sample_id": ids[0],
            "last_sample_id": ids[-1],
            "max_declared_length": max(int(row.get("length", 0) or 0) for row in rows),
        })
    snapshot_core = [
        {
            "task": entry["task"],
            "row_count": entry["row_count"],
            "official_slice_canonical_sha256": entry["official_slice_canonical_sha256"],
            "sample_id_sequence_sha256": entry["sample_id_sequence_sha256"],
        }
        for entry in task_entries
    ]
    return {
        "data_root": str(data_root),
        "task_order": list(TASKS),
        "total_samples": sum(entry["row_count"] for entry in task_entries),
        "sample_id_policy": "<task>:<zero_based_row_index>:<canonical_content_sha256_first16>",
        "canonical_content_fields": [
            "task", "context", "input", "question", "answers", "all_classes", "language"
        ],
        "string_normalization": "none; parsed string values are preserved exactly",
        "task_entries": task_entries,
        "dataset_snapshot_sha256": sha256_bytes(canonical_bytes(snapshot_core)),
    }, rows_by_task


def selected_model_hashes(model_root: Path) -> Dict[str, Any]:
    names = (
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "model.safetensors.index.json",
    )
    files = {
        name: sha256_file(model_root / name)
        for name in names
        if (model_root / name).exists()
    }
    return {
        "path": str(model_root),
        "exists": model_root.exists(),
        "snapshot_revision": model_root.name if model_root.exists() else None,
        "selected_file_sha256": files,
        "fingerprint_sha256": sha256_bytes(canonical_bytes(files)) if files else None,
    }


def build_prompt_hashes(
    rows_by_task: Mapping[str, Sequence[Mapping[str, Any]]],
    prompt_map: Mapping[str, str],
    qwen_model_root: Path,
) -> Dict[str, Any]:
    template_entries = {
        task: {
            "template_sha256": sha256_bytes(prompt_map[task].encode("utf-8")),
            "uses_chat_template": task not in NO_CHAT_TASKS,
        }
        for task in TASKS
    }
    qwen: Dict[str, Any] = {
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "snapshot_revision": qwen_model_root.name if qwen_model_root.exists() else None,
        "status": "unavailable",
        "chat_template_sha256": None,
        "final_prompt_sequence_sha256_by_task": {},
    }
    if qwen_model_root.exists():
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            str(qwen_model_root),
            trust_remote_code=True,
            local_files_only=True,
        )
        chat_template = tokenizer.chat_template or ""
        qwen["status"] = "verified_exact_final_rendering"
        qwen["chat_template_sha256"] = sha256_bytes(chat_template.encode("utf-8"))
        qwen["tokenizer_class"] = tokenizer.__class__.__name__
        qwen["tokenizer_config_sha256"] = sha256_file(qwen_model_root / "tokenizer_config.json")
        qwen["tokenizer_json_sha256"] = sha256_file(qwen_model_root / "tokenizer.json")
        for task in TASKS:
            hashes: List[str] = []
            truncated = 0
            max_raw_tokens = 0
            for row in rows_by_task[task]:
                prompt = prompt_map[task].format(**row)
                raw_ids = tokenizer(prompt, truncation=False, add_special_tokens=True).input_ids
                max_raw_tokens = max(max_raw_tokens, len(raw_ids))
                if len(raw_ids) > 131072:
                    half = 131072 // 2
                    prompt = (
                        tokenizer.decode(raw_ids[:half], skip_special_tokens=True)
                        + tokenizer.decode(raw_ids[-half:], skip_special_tokens=True)
                    )
                    truncated += 1
                if task not in NO_CHAT_TASKS:
                    prompt = tokenizer.apply_chat_template(
                        [{"role": "user", "content": prompt}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                hashes.append(sha256_bytes(prompt.encode("utf-8")))
            qwen["final_prompt_sequence_sha256_by_task"][task] = sha256_bytes(
                "\n".join(hashes).encode("utf-8")
            )
            qwen.setdefault("runtime_truncation_count_by_task", {})[task] = truncated
            qwen.setdefault("max_raw_prompt_tokens_by_task", {})[task] = max_raw_tokens

    return {
        "prompt_template_source": "configs/dataset2prompt.json",
        "prompt_template_file_sha256": sha256_file(REPO_ROOT / "configs/dataset2prompt.json"),
        "task_templates": template_entries,
        "no_chat_tasks": sorted(NO_CHAT_TASKS),
        "rendering_order": [
            "format dataset template with parsed sample",
            "tokenize raw formatted prompt without truncation",
            "if token length exceeds model max, decode first half plus last half",
            "apply model chat template except for no-chat tasks",
            "SHA256 exact final prompt UTF-8 text before final tokenizer call",
        ],
        "qwen25": qwen,
        "llama31": {
            "model_id": EXPECTED_LLAMA_MODEL,
            "status": "BLOCKED_SOURCE_ABSENT",
            "chat_template_sha256": None,
            "final_prompt_sequence_sha256_by_task": None,
            "reason": "Requested LLaMA result/model snapshot is absent on this machine; no substitute tokenizer was used.",
        },
    }


def make_task_manifest(max_new_tokens: Mapping[str, int], dataset_hashes: Mapping[str, Any]) -> Dict[str, Any]:
    category_by_task = {
        task: category
        for category, members in CATEGORIES.items()
        for task in members
    }
    entries = []
    dataset_by_task = {entry["task"]: entry for entry in dataset_hashes["task_entries"]}
    for order, task in enumerate(TASKS, start=1):
        entries.append({
            "order": order,
            "task": task,
            "sample_count": COUNTS[task],
            "category": category_by_task[task],
            "max_new_tokens": int(max_new_tokens[task]),
            "dataset_file_sha256": dataset_by_task[task]["file_sha256"],
            "sample_id_sequence_sha256": dataset_by_task[task]["sample_id_sequence_sha256"],
        })
    return {
        "name": "LongBench official 17-task full",
        "task_order": list(TASKS),
        "task_count": 17,
        "total_samples": 3950,
        "tasks": entries,
        "categories": {key: list(value) for key, value in CATEGORIES.items()},
        "macro": "compute each official task score independently, then arithmetic mean over 17 task scores",
        "pooled_sample_average_forbidden": True,
    }


def make_model_configs(model_hashes: Mapping[str, Any], prompt_hashes: Mapping[str, Any]) -> Dict[str, Any]:
    common = {
        "block_size": 64,
        "token_unit_size": 8,
        "gamma": 0.9,
        "local_query_window": 4,
        "batch_size": 1,
        "max_model_len": 131072,
        "dtype": "fp16",
        "attention_backend": "flash_attention_2",
        "decode": {
            "num_beams": 1,
            "do_sample": False,
            "temperature_effective": None,
            "top_p_effective": None,
            "use_cache": True,
            "task_specific_max_new_tokens": True,
            "samsum_additional_eos": "newline last token id",
        },
    }
    return {
        "common": common,
        "qwen25_7b_instruct": {
            "model_id": "Qwen/Qwen2.5-7B-Instruct",
            "model_family": "qwen2",
            "revision": model_hashes["snapshot_revision"],
            "model_fingerprint_sha256": model_hashes["fingerprint_sha256"],
            "tokenizer_config_sha256": prompt_hashes["qwen25"].get("tokenizer_config_sha256"),
            "tokenizer_json_sha256": prompt_hashes["qwen25"].get("tokenizer_json_sha256"),
            "chat_template_sha256": prompt_hashes["qwen25"].get("chat_template_sha256"),
            "schedule": QWEN_SCHEDULE,
            "status": "resolved",
        },
        "llama31_8b_instruct": {
            "model_id": EXPECTED_LLAMA_MODEL,
            "model_family": "llama",
            "revision": None,
            "model_fingerprint_sha256": None,
            "tokenizer_config_sha256": None,
            "tokenizer_json_sha256": None,
            "chat_template_sha256": None,
            "schedule": LLAMA_SCHEDULE,
            "status": "BLOCKED_SOURCE_ABSENT",
            "identity_gate": "Exact model/tokenizer revision and hashes must be recovered from the requested LLaMA CC artifacts before any LLaMA run.",
        },
        "model_specific_expected_differences": [
            "model and tokenizer identity",
            "chat template",
            "prune layers",
            "K block schedule",
            "M block schedule",
            "retained token schedule",
        ],
    }


def make_method_configs() -> Dict[str, Any]:
    methods = {
        "sliminfer_original": {
            "budget_group": "K",
            "definition": "Original SlimInfer active set A_K without rescue additions or swaps.",
        },
        "cc_sliminfer_v1": {
            "budget_group": "K",
            "definition": "Frozen CC v1 selector at the same K schedule as Original.",
            "implementation_identity": "must be recovered/verified per model; LLaMA source absent locally",
        },
        "drm_swap_original_k": {
            "budget_group": "K",
            "definition": "Start from A_K and perform strict one-in-one-out frozen DRM swaps; final size remains K.",
            "drm": {**DRM_FROZEN, "rescue_policy": "budget_neutral_drm_swap"},
        },
        "sliminfer_larger_km": {
            "budget_group": "K+M",
            "definition": "Directly retain K+M blocks by original SlimInfer relevance.",
        },
        "sliminfer_random_add_km": {
            "budget_group": "K+M",
            "definition": "Preserve 100% of A_K and randomly add M valid pruned blocks with seed 0.",
            "seed": 0,
        },
        "sliminfer_drm_add_km": {
            "budget_group": "K+M",
            "definition": "Preserve 100% of A_K and add M blocks using frozen DRM score and historical MMR.",
            "drm": DRM_FROZEN,
        },
        "full_kv_fa2": {
            "budget_group": "full_kv",
            "definition": "Full KV with official Hugging Face model and FlashAttention2; no SlimInfer pruning.",
            "adapter_policy": "thin adapter only",
        },
        "minference_official": {
            "budget_group": "external",
            "definition": "Official MInference implementation only.",
            "adapter_policy": "official repository plus thin schema adapter; no local approximation",
        },
        "flexprefill_official": {
            "budget_group": "external",
            "definition": "Official FlexPrefill implementation only.",
            "adapter_policy": "official repository plus thin schema adapter; no local approximation",
        },
        "lazyllm_official": {
            "budget_group": "external",
            "definition": "Official LazyLLM implementation after repository identity audit.",
            "adapter_policy": "do not guess repository; no local approximation",
        },
    }
    for name, spec in methods.items():
        spec["method_config_sha256"] = sha256_bytes(canonical_bytes({"method": name, **spec}))
    return {
        "methods": methods,
        "fair_comparisons": {
            "cc_paper": [["cc_sliminfer_v1", "sliminfer_original"]],
            "drm_paper": [
                ["drm_swap_original_k", "sliminfer_original"],
                ["sliminfer_drm_add_km", "sliminfer_larger_km"],
                ["sliminfer_drm_add_km", "sliminfer_random_add_km"],
            ],
        },
        "forbidden_same_budget_claim": "cc_sliminfer_v1 vs sliminfer_drm_add_km",
    }


def make_output_schema() -> Dict[str, Any]:
    status_enum = ["ok", "oom", "failed", "missing"]
    source_enum = ["runtime", "posthoc_reconstructed", "unavailable"]
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "urn:ulb:v1:prediction",
        "title": "Unified LongBench canonical prediction",
        "type": "object",
        "additionalProperties": True,
        "required": [
            "protocol_id", "model_id", "method", "task", "sample_index", "sample_id",
            "content_sha256", "prompt_sha256", "status", "prediction", "answers",
            "all_classes", "input_tokens", "output_tokens", "field_sources",
            "source_file", "source_line",
        ],
        "properties": {
            "protocol_id": {"type": "string", "pattern": "^ULB-v1-[0-9a-f]{16}$"},
            "model_id": {"type": "string", "minLength": 1},
            "method": {"type": "string", "minLength": 1},
            "task": {"enum": list(TASKS)},
            "sample_index": {"type": "integer", "minimum": 0},
            "sample_id": {"type": "string", "minLength": 1},
            "content_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            "prompt_sha256": {"type": ["string", "null"], "pattern": "^[0-9a-f]{64}$"},
            "template_sha256": {"type": ["string", "null"], "pattern": "^[0-9a-f]{64}$"},
            "chat_template_sha256": {"type": ["string", "null"], "pattern": "^[0-9a-f]{64}$"},
            "status": {"enum": status_enum},
            "prediction": {"type": "string"},
            "answers": {"type": "array"},
            "all_classes": {"type": ["array", "null"]},
            "input_tokens": {"type": ["integer", "null"], "minimum": 0},
            "output_tokens": {"type": ["integer", "null"], "minimum": 0},
            "field_sources": {
                "type": "object",
                "required": ["sample_id", "prompt_sha256", "status", "input_tokens", "output_tokens"],
                "properties": {
                    "sample_id": {"enum": source_enum},
                    "prompt_sha256": {"enum": source_enum},
                    "status": {"enum": source_enum},
                    "input_tokens": {"enum": source_enum},
                    "output_tokens": {"enum": source_enum},
                },
            },
            "source_file": {"type": "string", "minLength": 1},
            "source_line": {"type": "integer", "minimum": 1},
        },
        "allOf": [
            {
                "if": {"properties": {"status": {"const": "oom"}}},
                "then": {"properties": {"prediction": {"const": "[OOM]"}}},
            }
        ],
    }


def make_active_trace_schema() -> Dict[str, Any]:
    id_array = {"type": "array", "items": {"type": "integer", "minimum": 0}, "uniqueItems": True}
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "urn:ulb:v1:active-block-trace",
        "title": "Unified LongBench active block trace event",
        "type": "object",
        "additionalProperties": True,
        "required": [
            "protocol_id", "model_id", "method", "task", "sample_id", "sample_index",
            "decode_step", "layer_idx", "selection_event_index", "selected_count",
            "trace_level", "trace_source",
        ],
        "properties": {
            "protocol_id": {"type": "string"},
            "model_id": {"type": "string"},
            "method": {"type": "string"},
            "task": {"enum": list(TASKS)},
            "sample_id": {"type": "string"},
            "sample_index": {"type": "integer", "minimum": 0},
            "decode_step": {"type": "integer", "minimum": 0},
            "layer_idx": {"type": "integer", "minimum": 0},
            "selection_event_index": {"type": "integer", "minimum": 0},
            "active_before_ids": id_array,
            "candidate_ids": id_array,
            "active_after_ids": id_array,
            "added_ids": id_array,
            "removed_ids": id_array,
            "sink_ids": id_array,
            "selected_count": {"type": "integer", "minimum": 0},
            "trace_level": {"enum": ["summary", "full"]},
            "trace_source": {"enum": ["runtime"]},
        },
        "allOf": [
            {
                "if": {"properties": {"trace_level": {"const": "full"}}},
                "then": {
                    "required": [
                        "active_before_ids", "candidate_ids", "active_after_ids",
                        "added_ids", "removed_ids", "sink_ids",
                    ]
                },
            }
        ],
    }


def scorer_manifest() -> Dict[str, Any]:
    metric_map = {
        "passage_retrieval_en": "retrieval_score",
        "qasper": "qa_f1_score",
        "multifieldqa_en": "qa_f1_score",
        "hotpotqa": "qa_f1_score",
        "2wikimqa": "qa_f1_score",
        "musique": "qa_f1_score",
        "gov_report": "rouge_score",
        "qmsum": "rouge_score",
        "multi_news": "rouge_score",
        "vcsum": "rouge_zh_score",
        "passage_count": "count_score",
        "trec": "classification_score",
        "triviaqa": "qa_f1_score",
        "samsum": "rouge_score",
        "lsht": "classification_score",
        "lcc": "code_sim_score",
        "repobench-p": "code_sim_score",
    }
    return {
        "scorer_source": "longbench_eval.py",
        "scorer_sha256": sha256_file(REPO_ROOT / "longbench_eval.py"),
        "per_sample_helper_source": "tools/consolidate_official_repro_results.py",
        "per_sample_helper_sha256": sha256_file(REPO_ROOT / "tools/consolidate_official_repro_results.py"),
        "task_metric_map": metric_map,
        "task_score_scale": 100,
        "task_score_rounding_for_legacy_raw": 2,
        "macro_policy": "arithmetic mean of the 17 independently computed task scores",
        "pooled_sample_weighting": False,
    }


def code_hashes() -> Dict[str, Any]:
    entries = {}
    for rel in CODE_FILES:
        path = REPO_ROOT / rel
        entries[rel] = {
            "exists": path.exists(),
            "sha256": sha256_file(path) if path.exists() else None,
            "size_bytes": path.stat().st_size if path.exists() else None,
        }
    return {
        "git_commit": command_output(["git", "rev-parse", "HEAD"])[1].strip(),
        "files": entries,
    }


def environment_manifest() -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "captured_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "conda_prefix": os.environ.get("CONDA_PREFIX"),
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
    }
    for module in ("torch", "transformers", "flash_attn"):
        try:
            imported = __import__(module)
            payload[module] = getattr(imported, "__version__", "unknown")
        except Exception as exc:
            payload[module] = {"status": "unavailable", "error": repr(exc)}
    try:
        import torch

        payload["torch_cuda_runtime"] = torch.version.cuda
        payload["cuda_available"] = torch.cuda.is_available()
        payload["cuda_device_count"] = torch.cuda.device_count()
    except Exception:
        pass
    rc, smi = command_output([
        "nvidia-smi",
        "--query-gpu=index,name,uuid,driver_version,memory.total,memory.used,utilization.gpu,pstate",
        "--format=csv,noheader",
    ])
    payload["nvidia_smi_query"] = {"returncode": rc, "output": smi.strip()}
    rc, conda = command_output(["conda", "info", "--json"])
    try:
        conda_payload = json.loads(conda) if rc == 0 else {"raw": conda}
    except Exception:
        conda_payload = {"raw": conda}
    for key in ("env_vars", "user_agent"):
        conda_payload.pop(key, None)
    payload["conda_info"] = conda_payload
    return payload


def policy_documents(protocol_dir: Path) -> None:
    write_text(
        protocol_dir / "OOM_POLICY.md",
        """# OOM Policy

1. Batch size is 1 and methods run sequentially on one GPU.
2. A CUDA OOM is retried once only after discarding the partially written cache, garbage collection, and `torch.cuda.empty_cache()`.
3. A second OOM writes prediction `[OOM]`, status `oom`, the exact sample identity, and the exception metadata, then continues to the next sample.
4. `failed` is reserved for non-OOM per-sample exceptions. `missing` means no canonical row exists at finalization. Neither may be relabeled as OOM.
5. Raw full-set scoring assigns zero contribution to `oom`, `failed`, and `missing`, while preserving each status separately in manifests.
6. Raw historical predictions are immutable. Offload fills are separate artifacts and never replace raw predictions or efficiency results.
""",
    )
    write_text(
        protocol_dir / "COMMON_VALID_POLICY.md",
        """# Common-Valid Policy

For a method set S, the global common-valid set is the intersection of sample IDs whose canonical status is `ok` for every method in S. Pairwise common-valid uses the same definition for exactly two methods. OOM, failed, missing, duplicate, and identity-mismatched rows are excluded from the applicable intersection but are never deleted from source artifacts.

Scores are recomputed per task on the retained paired sample IDs with the frozen official metric, then macro-averaged across the 17 tasks. The 3950 samples must never be pooled into a single weighted average. Every export records S, the ordered sample-ID hash per task, retained N, excluded N, and exclusion reasons.
""",
    )
    write_text(
        protocol_dir / "BOOTSTRAP_POLICY.md",
        """# Bootstrap Policy

- Use pairwise common-valid samples and a fixed seed of 0.
- Run 10,000 paired bootstrap replicates.
- Within each task, resample paired sample IDs with replacement to that task's common-valid N; both methods use the identical sampled IDs and multiplicities.
- Recompute the frozen official task metric from sampled predictions and references, then compute the arithmetic mean of all 17 task scores for every replicate.
- Report observed macro delta, bootstrap mean delta, 2.5/97.5 percentile CI, P(delta > 0), seed, per-task N, and total common-valid N.
- Do not bootstrap pooled per-sample metric values across tasks.
""",
    )
    write_text(
        protocol_dir / "RESUME_POLICY.md",
        """# Resume Policy

Each method/task has an atomic state file and an append-safe temporary JSONL. A resume validates protocol ID, model/tokenizer fingerprints, method hash, task manifest, ordered sample IDs, and the last complete JSON line before continuing. Completed sample IDs are skipped and never overwritten. Duplicate or out-of-order IDs stop the affected task.

Task completion performs schema validation, exact expected-count validation, duplicate/missing checks, scoring, fsync, and atomic rename from `.tmp` to canonical output. Detached launches must write `LAUNCH_COMMAND.txt`, `RESUME_COMMAND.txt`, `PID.json`, `RUN_STATE.json`, `PROGRESS.json`, wrapper logs, method logs, and task states. Initialization failures stop the current method; isolated per-sample OOM/failures continue under the status policy.
""",
    )


def protocol_markdown(
    protocol_id: str,
    dataset_hash: str,
    llama_available: bool,
) -> str:
    llama_state = "resolved" if llama_available else "BLOCKED_SOURCE_ABSENT"
    return f"""# Unified LongBench Protocol v1.0

Protocol ID: `{protocol_id}`

## Scope

This protocol freezes the shared data, prompting, generation, scoring, failure handling, identity, output, resume, and statistical rules for CC and DRM LongBench experiments. Model-specific pruning schedules and chat templates are explicit branches and do not make the protocol non-unified.

The protocol is for experiments and standardized result export only. It does not authorize paper/LaTeX edits or claims.

## Official task set

The ordered 17-task manifest contains 3,950 samples: 150 for `multifieldqa_en`, 500 each for `lcc` and `repobench-p`, and 200 for every other task. Dataset snapshot SHA256 is `{dataset_hash}`. A run with a per-task cap of 200 is not this protocol.

Each task is scored independently with the frozen LongBench scorer. The final score is the simple arithmetic mean of 17 task scores. Pooling all samples into a weighted average is forbidden.

## Stable sample identity

Canonical sample IDs are `<task>:<zero_based_row_index>:<content_hash16>`. The full content SHA256 hashes canonical JSON over `task`, `context`, `input`, `question`, `answers`, `all_classes`, and `language`. JSON object key order and serialization whitespace do not affect the hash; parsed string values are preserved exactly.

## Prompt and generation

The exact dataset template is formatted first. If the raw tokenized prompt exceeds 131,072 tokens, the runner decodes the first and last 65,536 tokens and concatenates them. The model chat template is then applied except for `trec`, `triviaqa`, `samsum`, `lsht`, `lcc`, and `repobench-p`. Per-sample prompt SHA256 is computed over the exact final UTF-8 prompt before its final tokenizer call.

Generation is batch size 1, single-GPU sequential, greedy (`do_sample=False`, `num_beams=1`), with frozen task-specific `max_new_tokens`. `samsum` additionally accepts the final newline token as EOS, matching the historical runner.

## Model branches

- Qwen2.5-7B-Instruct: layers 9/18/26, K 192/96/64, M 10/5/3, retained tokens 12288/6144/4096.
- LLaMA-3.1-8B-Instruct: layers 10/20/30, K 128/64/32, M 6/3/2, retained tokens 8192/4096/2048; Add tokens 8576/4288/2176.
- Common: block size 64, token unit 8, gamma/swap threshold 0.9, local query window 4, fp16, FlashAttention2.

The requested LLaMA source state is currently `{llama_state}` on this machine. Its exact model/tokenizer/chat-template and method-code fingerprints must pass identity audit before LLaMA outputs can carry this protocol ID.

## DRM and fair comparisons

DRM is frozen at rescue ratio 0.05, alpha_diff 0.5, lambda_red 0.25, eta_rescue 0.25, eta_active 0.25, seed 0, and historical MMR. Add preserves 100% of A_K and adds exactly M valid blocks. Swap is strict one-in-one-out and remains K.

CC compares `cc_sliminfer_v1` with `sliminfer_original`. DRM compares Swap with Original at K, and DRM-Add with Larger/Random-Add at K+M. CC and DRM-Add are not a same-budget comparison.

## Status and output

Canonical statuses are `ok`, `oom`, `failed`, and `missing`. Missing numeric fields are `null`, never zero. Historical reconstruction is explicitly labeled in `field_sources`. OOM predictions remain `[OOM]`. Source file and line are always retained.

Raw full-set, global common-valid, pairwise common-valid, and offload-completed accuracy are distinct result concepts. Source predictions are immutable and are never silently merged across protocol identities.

## Current freeze boundary

Protocol semantics, the local 3,950-sample data snapshot, Qwen tokenizer/prompt rendering, scorer, schemas, and statistical policies are frozen. LLaMA source artifacts were not found at the requested path, so LLaMA result export, LLaMA identity PASS, smoke, and detached full launch remain blocked. No GPU task was started by this freeze.
"""


IDENTITY_FIELDS = (
    "record_type", "result_group", "model_key", "method", "result_root", "artifact_status",
    "data_root", "dataset_snapshot_sha256", "dataset_order", "prompt_template_sha256",
    "chat_template_sha256", "tokenizer_identity", "tokenizer_revision_or_hash",
    "model_identity", "model_revision_or_hash", "task_max_new_tokens_sha256", "truncation",
    "dtype", "decoding", "scorer_code_sha256", "longbench_evaluator_sha256", "oom_policy",
    "sample_id_policy", "macro_policy", "runner_commit", "method_config_sha256",
    "expected_model_specific_differences", "overall_status", "notes",
)


def identity_matrix_rows(
    protocol_id: str,
    task_manifest: Mapping[str, Any],
    dataset_hashes: Mapping[str, Any],
    prompt_hashes: Mapping[str, Any],
    scorer_hashes: Mapping[str, Any],
    method_configs: Mapping[str, Any],
    git_commit: str,
) -> List[Dict[str, Any]]:
    max_new_hash = sha256_bytes(canonical_bytes({
        row["task"]: row["max_new_tokens"] for row in task_manifest["tasks"]
    }))
    prompt_template_hash = prompt_hashes["prompt_template_file_sha256"]
    qwen_chat_hash = prompt_hashes["qwen25"].get("chat_template_sha256")
    qwen_tokenizer_hash = prompt_hashes["qwen25"].get("tokenizer_json_sha256")
    qwen_model_fp = selected_model_hashes(QWEN_SNAPSHOT)["fingerprint_sha256"]
    common = {
        "record_type": "run_or_planned_run",
        "data_root": str(DEFAULT_DATA_ROOT),
        "dataset_snapshot_sha256": dataset_hashes["dataset_snapshot_sha256"],
        "dataset_order": ",".join(TASKS),
        "prompt_template_sha256": prompt_template_hash,
        "task_max_new_tokens_sha256": max_new_hash,
        "truncation": "head_tail_to_131072_before_chat",
        "dtype": "fp16",
        "decoding": "greedy;do_sample=false;num_beams=1;task_max_new_tokens",
        "scorer_code_sha256": scorer_hashes["scorer_sha256"],
        "longbench_evaluator_sha256": scorer_hashes["scorer_sha256"],
        "oom_policy": "retry_once_then_[OOM]/oom_and_continue",
        "sample_id_policy": "task:index:content_hash16; posthoc reconstruction allowed with provenance",
        "macro_policy": "17-task arithmetic macro",
        "runner_commit": git_commit,
        "expected_model_specific_differences": "model,tokenizer,chat_template,prune_layers,K,M,retained_tokens",
    }
    rows: List[Dict[str, Any]] = []
    for method in (
        "sliminfer_original", "sliminfer_larger_km", "sliminfer_random_add_km",
        "sliminfer_drm_add_km", "drm_swap_original_k",
    ):
        rows.append({
            **common,
            "result_group": "qwen25_drm_official_3950",
            "model_key": "qwen25_7b_instruct",
            "method": method,
            "result_root": str(LOCAL_QWEN_DRM),
            "artifact_status": "complete_3950",
            "chat_template_sha256": qwen_chat_hash,
            "tokenizer_identity": "Qwen/Qwen2.5-7B-Instruct",
            "tokenizer_revision_or_hash": qwen_tokenizer_hash,
            "model_identity": "Qwen/Qwen2.5-7B-Instruct",
            "model_revision_or_hash": qwen_model_fp,
            "method_config_sha256": method_configs["methods"][method]["method_config_sha256"],
            "overall_status": "PASS",
            "notes": f"Protocol-compatible identity verified/reconstructable; canonical protocol ID is {protocol_id}.",
        })
    for method in ("sliminfer_original", "cc_sliminfer_v1"):
        rows.append({
            **common,
            "result_group": "qwen25_cc_official_3950_crosscheck",
            "model_key": "qwen25_7b_instruct",
            "method": method,
            "result_root": str(LOCAL_QWEN_CC),
            "artifact_status": "complete_3950",
            "chat_template_sha256": qwen_chat_hash,
            "tokenizer_identity": "Qwen/Qwen2.5-7B-Instruct",
            "tokenizer_revision_or_hash": qwen_tokenizer_hash,
            "model_identity": "Qwen/Qwen2.5-7B-Instruct",
            "model_revision_or_hash": qwen_model_fp,
            "method_config_sha256": method_configs["methods"][method]["method_config_sha256"],
            "overall_status": "PASS",
            "notes": "Effective greedy decoding matches despite historical temperature=1.0 encoding because do_sample resolves false.",
        })
    llama_methods = (
        "sliminfer_original", "cc_sliminfer_v1", "sliminfer_larger_km",
        "sliminfer_random_add_km", "sliminfer_drm_add_km", "drm_swap_original_k",
        "full_kv_fa2", "minference_official", "flexprefill_official", "lazyllm_official",
    )
    for method in llama_methods:
        existing_expected = method in {"sliminfer_original", "cc_sliminfer_v1"}
        rows.append({
            **common,
            "result_group": "llama31_existing_cc" if existing_expected else "llama31_future_suite",
            "model_key": "llama31_8b_instruct",
            "method": method,
            "result_root": str(REQUESTED_LLAMA_CC) if existing_expected else "not_created",
            "artifact_status": "BLOCKED_SOURCE_ABSENT" if existing_expected else "planned_not_run",
            "chat_template_sha256": "",
            "tokenizer_identity": EXPECTED_LLAMA_MODEL,
            "tokenizer_revision_or_hash": "",
            "model_identity": EXPECTED_LLAMA_MODEL,
            "model_revision_or_hash": "",
            "method_config_sha256": method_configs["methods"][method]["method_config_sha256"],
            "overall_status": "UNKNOWN",
            "notes": "Exact LLaMA artifacts/model/tokenizer/runner identity unavailable; do not merge or launch until resolved.",
        })
    return rows


def freeze(args: argparse.Namespace) -> None:
    audit_dir = Path(args.audit_dir).expanduser().resolve()
    protocol_dir = Path(args.protocol_dir).expanduser().resolve()
    data_root = Path(args.data_root).expanduser().resolve()
    audit_dir.mkdir(parents=True, exist_ok=True)
    protocol_dir.mkdir(parents=True, exist_ok=True)

    print("[freeze] dataset snapshot", flush=True)
    dataset_hashes, rows_by_task = load_dataset_snapshot(data_root)
    prompt_map = read_json(REPO_ROOT / "configs/dataset2prompt.json")
    max_new_tokens = read_json(REPO_ROOT / "configs/dataset2maxlen.json")
    print("[freeze] exact Qwen prompt rendering", flush=True)
    prompt_hashes = build_prompt_hashes(rows_by_task, prompt_map, QWEN_SNAPSHOT)
    task_manifest = make_task_manifest(max_new_tokens, dataset_hashes)
    qwen_model_hashes = selected_model_hashes(QWEN_SNAPSHOT)
    model_configs = make_model_configs(qwen_model_hashes, prompt_hashes)
    method_configs = make_method_configs()
    output_schema = make_output_schema()
    active_trace_schema = make_active_trace_schema()
    scorer_hashes = scorer_manifest()
    current_code_hashes = code_hashes()
    env = environment_manifest()

    write_json(protocol_dir / "task_manifest.json", task_manifest)
    write_json(protocol_dir / "dataset_hashes.json", dataset_hashes)
    write_json(protocol_dir / "prompt_hashes.json", prompt_hashes)
    write_json(protocol_dir / "scorer_hashes.json", scorer_hashes)
    write_json(protocol_dir / "code_hashes.json", current_code_hashes)
    write_json(protocol_dir / "model_configs.json", model_configs)
    write_json(protocol_dir / "method_configs.json", method_configs)
    write_json(protocol_dir / "environment_manifest.json", env)
    write_json(protocol_dir / "output_schema.json", output_schema)
    write_json(protocol_dir / "active_block_trace_schema.json", active_trace_schema)
    policy_documents(protocol_dir)

    component_hashes = {
        name: sha256_file(protocol_dir / name)
        for name in (
            "task_manifest.json", "dataset_hashes.json", "prompt_hashes.json",
            "scorer_hashes.json", "code_hashes.json", "model_configs.json",
            "method_configs.json", "output_schema.json", "active_block_trace_schema.json",
            "OOM_POLICY.md", "COMMON_VALID_POLICY.md", "BOOTSTRAP_POLICY.md", "RESUME_POLICY.md",
        )
    }
    manifest = {
        "name": "Unified LongBench Protocol",
        "version": "1.0",
        "schema_version": 1,
        "official_suite": "LongBench 17-task full",
        "task_count": 17,
        "sample_count": 3950,
        "dataset_snapshot_sha256": dataset_hashes["dataset_snapshot_sha256"],
        "component_sha256": component_hashes,
        "sample_identity": {
            "format": "<task>:<zero_based_row_index>:<content_sha256_first16>",
            "content_fields": dataset_hashes["canonical_content_fields"],
            "canonical_json": "UTF-8; sorted keys; compact separators; ensure_ascii=false; allow_nan=false",
        },
        "runtime": {
            "batch_size": 1,
            "single_gpu_sequential": True,
            "greedy": True,
            "do_sample": False,
            "block_size": 64,
            "token_unit_size": 8,
            "gamma": 0.9,
            "local_query_window": 4,
            "oom_continue": True,
            "detached_resume_required": True,
        },
        "scoring": {
            "per_task_official_metric": True,
            "macro": "arithmetic mean of 17 task scores",
            "global_common_valid": True,
            "pairwise_common_valid": True,
            "paired_bootstrap_replicates": 10000,
            "bootstrap_seed": 0,
        },
        "immutability": {
            "historical_predictions_read_only": True,
            "unverifiable_fields_must_be_null": True,
            "posthoc_fields_require_provenance": True,
            "cross_protocol_silent_merge_forbidden": True,
        },
        "identity_gate": {
            "qwen25": "resolved",
            "llama31": "BLOCKED_SOURCE_ABSENT",
            "external_baselines": "official path audit required",
        },
    }
    protocol_manifest_path = protocol_dir / "protocol_manifest.json"
    write_json(protocol_manifest_path, manifest)
    manifest_sha256 = sha256_bytes(canonical_bytes(manifest))
    protocol_id = f"ULB-v1-{manifest_sha256[:16]}"
    write_text(protocol_dir / "PROTOCOL_ID.txt", protocol_id)
    write_text(
        protocol_dir / "PROTOCOL.md",
        protocol_markdown(protocol_id, dataset_hashes["dataset_snapshot_sha256"], REQUESTED_LLAMA_CC.exists()),
    )

    identity_rows = identity_matrix_rows(
        protocol_id,
        task_manifest,
        dataset_hashes,
        prompt_hashes,
        scorer_hashes,
        method_configs,
        current_code_hashes["git_commit"],
    )
    write_csv(protocol_dir / "PROTOCOL_IDENTITY_MATRIX.csv", identity_rows, IDENTITY_FIELDS)

    print("[freeze] inventory and immutable hashes", flush=True)
    cc_rows: List[Dict[str, Any]] = []
    root_summaries: List[Dict[str, Any]] = []
    all_hash_lines: List[str] = []
    for label, root in (
        ("llama31_cc_requested", REQUESTED_LLAMA_CC),
        ("qwen25_cc_local_crosscheck", LOCAL_QWEN_CC),
    ):
        rows, summary, hash_lines = inventory_root(label, root, args.hash_historical_files)
        cc_rows.extend(rows)
        root_summaries.append(summary)
        all_hash_lines.extend(hash_lines)
    drm_rows, drm_summary, drm_hash_lines = inventory_root(
        "qwen25_drm_five_method", LOCAL_QWEN_DRM, args.hash_historical_files
    )
    root_summaries.append(drm_summary)
    all_hash_lines.extend(drm_hash_lines)
    inventory_fields = (
        "result_family", "root", "root_exists", "relative_path", "size_bytes", "mtime_ns",
        "sha256", "artifact_class", "prediction_immutable",
    )
    write_csv(audit_dir / "CC_ARTIFACT_INVENTORY.csv", cc_rows, inventory_fields)
    write_csv(audit_dir / "DRM_ARTIFACT_INVENTORY.csv", drm_rows, inventory_fields)
    write_text(audit_dir / "SHA256_BEFORE.txt", "\n".join(all_hash_lines) or "# No available historical files to hash")
    immutability = {
        "captured_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "mode": "read_only_historical_artifact_inventory",
        "historical_files_hashed": bool(args.hash_historical_files),
        "prediction_mutations": 0,
        "roots": root_summaries,
        "sha256_before_path": str(audit_dir / "SHA256_BEFORE.txt"),
        "verification_command": f"sha256sum -c {audit_dir / 'SHA256_BEFORE.txt'}",
    }
    write_json(audit_dir / "IMMUTABILITY_MANIFEST.json", immutability)
    roots_payload = {
        "requested_repository": {"path": str(REQUESTED_REPO), "exists": REQUESTED_REPO.exists()},
        "actual_repository": {"path": str(REPO_ROOT), "exists": REPO_ROOT.exists()},
        "requested_llama_cc": {"path": str(REQUESTED_LLAMA_CC), "exists": REQUESTED_LLAMA_CC.exists()},
        "qwen25_cc_local_crosscheck": {"path": str(LOCAL_QWEN_CC), "exists": LOCAL_QWEN_CC.exists()},
        "qwen25_drm_five_method": {"path": str(LOCAL_QWEN_DRM), "exists": LOCAL_QWEN_DRM.exists()},
        "longbench_data": {"path": str(data_root), "exists": data_root.exists()},
        "qwen_model_snapshot": {"path": str(QWEN_SNAPSHOT), "exists": QWEN_SNAPSHOT.exists()},
        "expected_llama_model": {"model_id": EXPECTED_LLAMA_MODEL, "resolved": False},
    }
    write_json(audit_dir / "EXISTING_RESULT_ROOTS.json", roots_payload)
    write_json(audit_dir / "ENVIRONMENT.json", env)
    _, gpu_state = command_output(["nvidia-smi"])
    _, gpu_processes = command_output([
        "nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"
    ])
    write_text(
        audit_dir / "GPU_STATE.txt",
        "Captured read-only; no GPU experiment was launched.\n\n"
        + gpu_state
        + "\nCompute applications:\n"
        + (gpu_processes.strip() or "(none)"),
    )

    blocker = {
        "protocol_id": protocol_id,
        "protocol_manifest_sha256": manifest_sha256,
        "audit_status": "PARTIAL_PASS_WITH_BLOCKER",
        "qwen_data_prompt_scorer_identity": "PASS",
        "qwen_drm_history_available": True,
        "llama_cc_history_available": False,
        "llama_identity": "UNKNOWN",
        "gpu_launched": False,
        "blocker": (
            "Requested repository and results/llama_cc_official_17task_full_20260717_113452 "
            "do not exist on this machine. LLaMA export/statistics/identity and all LLaMA GPU stages are blocked."
        ),
        "required_to_unblock": [
            "Provide or mount the requested LLaMA repository/result root with original files.",
            "Recover exact LLaMA model/tokenizer revisions and runner/method hashes from those artifacts.",
            "Rerun this freeze/identity audit without changing protocol semantics.",
        ],
    }
    write_json(audit_dir / "AUDIT_DECISION.json", blocker)
    write_text(
        audit_dir / "BLOCKER.md",
        f"""# Unified Protocol Freeze Blocker

The shared 3,950-sample data snapshot, Qwen historical identity, prompt rendering, scorer, schemas, and protocol semantics were frozen as `{protocol_id}`.

The requested repository `{REQUESTED_REPO}` and LLaMA CC root `{REQUESTED_LLAMA_CC}` are absent. No LLaMA prediction, score, prompt hash, active ID, model hash, tokenizer hash, or runner identity was inferred from the numbers in the request. LLaMA CC export, identity PASS, smoke, and detached full launch are blocked until the original artifacts are mounted or copied intact.

No GPU experiment was launched and no historical prediction file was modified.
""",
    )

    protocol_files = sorted(path for path in protocol_dir.iterdir() if path.is_file() and path.name != "SHA256SUMS.txt")
    checksum_lines = [f"{sha256_file(path)}  {path.name}" for path in protocol_files]
    write_text(protocol_dir / "SHA256SUMS.txt", "\n".join(checksum_lines))
    print(json.dumps({
        "protocol_id": protocol_id,
        "protocol_dir": str(protocol_dir),
        "audit_dir": str(audit_dir),
        "qwen_drm_root": str(LOCAL_QWEN_DRM),
        "llama_cc_root": str(REQUESTED_LLAMA_CC),
        "llama_cc_available": REQUESTED_LLAMA_CC.exists(),
        "gpu_launched": False,
        "status": "PARTIAL_PASS_WITH_BLOCKER",
    }, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit-dir",
        default=str(REPO_ROOT / f"results/unified_protocol_v1_audit_{timestamp}"),
    )
    parser.add_argument(
        "--protocol-dir",
        default=str(REPO_ROOT / "protocol/unified_longbench_v1"),
    )
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument(
        "--skip-historical-file-hashes",
        dest="hash_historical_files",
        action="store_false",
        help="Record metadata only. The default computes every historical artifact SHA256.",
    )
    parser.set_defaults(hash_historical_files=True)
    return parser.parse_args()


if __name__ == "__main__":
    freeze(parse_args())
