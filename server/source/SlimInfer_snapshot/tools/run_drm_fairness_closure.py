#!/usr/bin/env python3
"""Protocol-matched DRM-Add fairness-closure runner.

This entry point is intentionally separate from the historical DRM-Add full
runner. It provides:

1. A CPU-only preflight over real LongBench prompts and K/K+M schedules.
2. Sequential, GPU-gated final-QA smoke cells.
3. Per-sample audit consolidation and fail-closed protocol validation.

It never reuses or writes the historical DRM-Add result directory.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
BASE_RUNNER = REPO_ROOT / "experiments" / "run_qwen25_sliminfer_variants.py"
PROMPT_CONFIG = REPO_ROOT / "configs" / "dataset2prompt.json"
DATASET2MAXLEN_CONFIG = REPO_ROOT / "configs" / "dataset2maxlen.json"
SCORER_SOURCE = REPO_ROOT / "longbench_eval.py"
DEFAULT_MODEL = Path(
    "/home/yu-bai/.cache/huggingface/hub/"
    "models--Qwen--Qwen2.5-7B-Instruct/snapshots/"
    "a09a35458c702b33eeacc393d103063234e8bc28"
)
DEFAULT_LONGBENCH = Path("/home/yu-bai/Desktop/jiaqizhang/HeadKV/data/LongBench")
DEFAULT_RETROSPECTIVE_RESULT = (
    REPO_ROOT
    / "results"
    / "drm_budget_neutral_swap_multihop200_20260704_181544"
)

METHOD_ORDER = [
    "sliminfer_original",
    "sliminfer_larger",
    "random_add",
    "delta_o_add",
    "drm_add",
]

METHOD_SPECS: Dict[str, Dict[str, Any]] = {
    "sliminfer_original": {
        "method_name": "SlimInfer original K",
        "rescue_policy": "none",
        "rescue_score": "not_applicable",
        "runner_rescue_score": "delta_o",
        "target_budget_type": "K",
        "final_budget_type": "K",
    },
    "sliminfer_larger": {
        "method_name": "SlimInfer larger K+M",
        "rescue_policy": "sliminfer_larger",
        "rescue_score": "not_applicable",
        "runner_rescue_score": "delta_o",
        "target_budget_type": "K+M",
        "final_budget_type": "K+M",
    },
    "random_add": {
        "method_name": "Random-Add K+M",
        "rescue_policy": "add",
        "rescue_score": "random",
        "runner_rescue_score": "random",
        "target_budget_type": "K+M",
        "final_budget_type": "K+M",
    },
    "delta_o_add": {
        "method_name": "DeltaO-Add K+M",
        "rescue_policy": "add",
        "rescue_score": "delta_o",
        "runner_rescue_score": "delta_o",
        "target_budget_type": "K+M",
        "final_budget_type": "K+M",
    },
    "drm_add": {
        "method_name": "DRM-Add K+M",
        "rescue_policy": "add",
        "rescue_score": "drm",
        "runner_rescue_score": "drm",
        "target_budget_type": "K+M",
        "final_budget_type": "K+M",
    },
    "budget_neutral_drm_swap": {
        "method_name": "budget_neutral_drm_swap",
        "rescue_policy": "budget_neutral_drm_swap",
        "rescue_score": "drm",
        "runner_rescue_score": "drm",
        "target_budget_type": "K",
        "final_budget_type": "K",
    },
    "gated_budget_neutral_drm_swap": {
        "method_name": "gated_budget_neutral_drm_swap",
        "rescue_policy": "gated_budget_neutral_drm_swap",
        "rescue_score": "drm",
        "runner_rescue_score": "drm",
        "target_budget_type": "K",
        "final_budget_type": "K",
    },
    "residual_aware_gate_a_removal_risk": {
        "method_name": "residual_aware_gate_a_removal_risk",
        "rescue_policy": "gated_budget_neutral_drm_swap",
        "rescue_score": "drm",
        "runner_rescue_score": "drm",
        "target_budget_type": "K",
        "final_budget_type": "K",
    },
    "random_budget_neutral_swap": {
        "method_name": "random_budget_neutral_swap",
        "rescue_policy": "budget_neutral_drm_swap",
        "rescue_score": "random",
        "runner_rescue_score": "random",
        "target_budget_type": "K",
        "final_budget_type": "K",
    },
}

BUDGET_NEUTRAL_SWAP_METHODS = {
    "budget_neutral_drm_swap",
    "random_budget_neutral_swap",
    "gated_budget_neutral_drm_swap",
    "residual_aware_gate_a_removal_risk",
}
GATED_SWAP_METHODS = {
    "gated_budget_neutral_drm_swap",
    "residual_aware_gate_a_removal_risk",
}
FIXED_SWAP_SELECTOR_METHODS = {
    "budget_neutral_drm_swap",
    "random_budget_neutral_swap",
}

REQUIRED_AUDIT_FIELDS = [
    "method_name",
    "task",
    "sample_id",
    "sample_index",
    "decode_policy",
    "base_scoring",
    "rescue_score",
    "seed",
    "target_budget_type",
    "K",
    "M",
    "target_K_plus_M",
    "selected_token_count",
    "selected_block_count",
    "active_set_size_before",
    "active_set_size_after",
    "pruned_candidate_count",
    "rescue_count",
    "added_count",
    "final_budget_type",
    "budget_equal_to_target",
    "over_budget",
    "used_blocks",
    "used_tokens",
    "per_layer_block_schedule",
    "prompt_hash",
    "prompt_wrapper_hash",
    "generation_config_hash",
    "scorer_hash",
    "model_name",
    "model_revision_or_unknown",
    "fallback_used",
    "latency_available",
    "memory_available",
    "kv_size_available",
    "compression_ratio_available",
    "throughput_available",
    "swap_count",
    "removed_count",
    "budget_equal_to_sliminfer",
    "removed_block_ids",
    "rescued_block_ids",
    "rescue_score_mean",
    "removed_score_mean",
    "active_redundancy_mean",
    "removed_value_norm_mean",
    "removed_value_norm_max",
    "rescued_value_norm_mean",
    "rescued_value_norm_max",
    "removed_qk_or_delta_score_mean",
    "removed_qk_or_delta_score_max",
    "rescued_qk_or_delta_score_mean",
    "rescued_qk_or_delta_score_max",
    "delete_risk_proxy_mean",
    "delete_risk_proxy_max",
    "rescue_gain_proxy_mean",
    "rescue_gain_proxy_max",
    "residual_benefit_proxy",
    "rescue_score_margin",
    "removed_block_text_preview",
    "rescued_block_text_preview",
    "residual_proxy_unavailable_fields",
    "residual_proxy_version",
    "protected_block_violation",
    "fallback_reason",
    "gate_mode",
    "gate_policy",
    "gate_triggered",
    "would_gate_trigger",
    "gate_score",
    "gate_reason",
    "gate_default_policy",
    "gate_fallback_to_original",
    "selected_policy",
    "selected_policy_before_gate",
    "selected_policy_after_gate",
    "gate_feature_missing",
    "gate_feature_missing_fields",
    "gate_feature_version",
    "rescue_coverage_gap",
    "coverage_gap_v2",
    "coverage_gap_v3",
    "context_entity_novelty",
    "title_or_heading_novelty",
    "title_or_heading_available",
    "rescued_novel_sentence_ratio",
    "bridge_entity_novelty_ratio",
    "bridge_entity_count",
    "rescued_active_semantic_novelty_mean",
    "rescued_active_semantic_novelty_max",
    "rescued_active_semantic_novelty_min",
    "deletion_evidence_risk_v3",
    "candidate_pool_exhausted",
    "question_entity_units",
    "rescued_entity_units",
    "active_entity_units",
    "rescued_only_entity_units",
    "removed_entity_units",
    "removed_title_like_units",
    "removed_date_number_units",
    "answer_type_proxy",
    "coverage_gap_v3_raw",
    "coverage_gap_v3_1",
    "rescued_vs_removed_entity_gain",
    "rescued_vs_active_entity_gain",
    "sentence_novelty_gain",
    "semantic_novelty_rank",
    "semantic_novelty_rank_mean",
    "semantic_novelty_rank_min",
    "bridge_entity_gain",
    "answer_type_support_gain",
    "rescued_answer_type_support",
    "removed_answer_type_support",
    "deletion_query_overlap_risk",
    "deletion_entity_risk",
    "deletion_date_number_risk",
    "deletion_title_risk",
    "deletion_answer_type_risk",
    "deletion_base_importance_risk",
    "deletion_protected_proximity_risk",
    "deletion_risk_v3_raw",
    "deletion_risk_v3_1",
    "removed_block_risk_sources",
    "rescue_selected_min_percentile",
    "rescue_selected_mean_percentile",
    "rescue_unselected_max_percentile",
    "rescue_boundary_margin_rank",
    "candidate_pool_size",
    "candidate_pool_effective_size",
    "candidate_score_std",
    "candidate_score_iqr",
    "candidate_score_entropy_proxy",
    "candidate_score_flat",
    "would_gate_trigger_v3_1",
    "gate_reason_v3_1",
    "coverage_components_available",
    "coverage_components_missing",
    "deletion_risk",
    "rescue_boundary_margin",
    "query_overlap_rescued",
    "query_overlap_removed",
    "entity_overlap_rescued",
    "entity_overlap_removed",
    "date_number_overlap_rescued",
    "date_number_overlap_removed",
    "rescue_score_top1",
    "rescue_score_top2",
    "rescue_score_min_selected",
    "rescue_score_boundary_selected",
    "rescue_score_boundary_unselected",
    "removed_score_min",
    "removed_score_max",
    "removed_drop_utility_mean",
    "removed_drop_utility_max",
    "rescued_vs_removed_score_gap",
    "active_redundancy_max",
    "protected_reasons",
]

NO_CHAT_TASKS = {"trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"}


class ClosureError(RuntimeError):
    """Fail-closed protocol or execution error."""


def parse_csv(value: str) -> List[str]:
    return [part.strip() for part in str(value).split(",") if part.strip()]


def parse_int_csv(value: str) -> List[int]:
    return [int(part) for part in parse_csv(value)]


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return list(iter_jsonl(path))


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ClosureError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def profile_availability(cell_dir: Path) -> Dict[str, bool]:
    latency_rows: List[Dict[str, str]] = []
    latency_path = cell_dir / "latency.csv"
    if latency_path.exists():
        with latency_path.open("r", newline="", encoding="utf-8") as handle:
            latency_rows = list(csv.DictReader(handle))
    memory_payload = read_json(cell_dir / "memory_profile.json", {})
    memory_rows = memory_payload.get("rows", []) if isinstance(memory_payload, dict) else []

    def has_value(field: str) -> bool:
        return any(str(row.get(field, "")).strip() for row in latency_rows)

    return {
        "latency_available": bool(latency_rows and (has_value("TTFT_s") or has_value("E2E_s"))),
        "memory_available": bool(memory_rows),
        "kv_size_available": has_value("prompt_kv_mem_GB"),
        "throughput_available": any(
            any("throughput" in str(key).lower() and str(value).strip() for key, value in row.items())
            for row in latency_rows
        ),
        "compression_ratio_available": False,
    }


def resolve_model_revision(model_path: Path, explicit_revision: str) -> str:
    if explicit_revision:
        return explicit_revision
    parts = model_path.resolve().parts if model_path.exists() else model_path.parts
    if "snapshots" in parts:
        index = parts.index("snapshots")
        if index + 1 < len(parts):
            return parts[index + 1]
    return "unknown"


def parse_sample_indices_by_task(value: str) -> Dict[str, List[int]]:
    value = str(value or "").strip()
    if not value:
        return {}
    parsed: Dict[str, List[int]] = {}
    for chunk in value.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise ClosureError(
                "--sample_indices_by_task must use task:i,j;task:k format"
            )
        task, indices = chunk.split(":", 1)
        task = task.strip()
        if not task:
            raise ClosureError("--sample_indices_by_task contains empty task")
        seen: set[int] = set()
        selected: List[int] = []
        for part in indices.split(","):
            part = part.strip()
            if not part:
                continue
            index = int(part)
            if index < 0:
                raise ClosureError(
                    f"--sample_indices_by_task has negative index for {task}: {index}"
                )
            if index not in seen:
                selected.append(index)
                seen.add(index)
        if not selected:
            raise ClosureError(f"--sample_indices_by_task has no indices for {task}")
        parsed[task] = selected
    return parsed


def load_longbench_rows(
    longbench_dir: Path,
    task: str,
    limit: int,
    sample_indices_by_task: Optional[Mapping[str, Sequence[int]]] = None,
) -> List[Dict[str, Any]]:
    candidates = [
        longbench_dir / f"{task}.jsonl",
        longbench_dir / task / "test.jsonl",
        longbench_dir / task / f"{task}.jsonl",
    ]
    for path in candidates:
        if path.exists():
            rows = read_jsonl(path)
            selected_indices = list((sample_indices_by_task or {}).get(task, []))
            if selected_indices:
                max_index = max(selected_indices)
                if max_index >= len(rows):
                    raise ClosureError(
                        f"sample index {max_index} out of range for {task}: rows={len(rows)}"
                    )
                return [
                    {**rows[index], "__longbench_original_index": int(index)}
                    for index in selected_indices
                ]
            return rows[:limit] if limit > 0 else rows
    raise ClosureError(f"LongBench task file not found for {task} under {longbench_dir}")


def sample_id_of(row: Mapping[str, Any], index: int) -> str:
    return str(row.get("_id", row.get("id", index)))


def prompt_wrapper_hash() -> str:
    payload = {
        "dataset2prompt_sha256": sha256_file(PROMPT_CONFIG),
        "chat_wrapper": "tokenizer.apply_chat_template(user, add_generation_prompt=True)",
        "no_chat_tasks": sorted(NO_CHAT_TASKS),
        "truncation": "tokenize raw prompt; if >max_model_len keep first/last halves; then chat-wrap",
    }
    return sha256_text(canonical_json(payload))


def load_dataset2maxlen() -> Dict[str, int]:
    payload = read_json(DATASET2MAXLEN_CONFIG, {})
    if not isinstance(payload, dict) or not payload:
        raise ClosureError(f"missing or empty dataset2maxlen config: {DATASET2MAXLEN_CONFIG}")
    return {str(key): int(value) for key, value in payload.items()}


def task_max_new_tokens(args: argparse.Namespace, task: str) -> int:
    if int(args.max_new_tokens) > 0:
        return int(args.max_new_tokens)
    dataset2maxlen = load_dataset2maxlen()
    if task not in dataset2maxlen:
        raise ClosureError(
            f"task {task} missing from {DATASET2MAXLEN_CONFIG}; "
            "cannot use task-specific max_new_tokens"
        )
    return int(dataset2maxlen[task])


def generation_config(args: argparse.Namespace) -> Dict[str, Any]:
    base = {
        "temperature": 0.0,
        "top_p": 1.0,
        "do_sample": False,
        "num_beams": 1,
        "use_cache": True,
        "max_model_len": args.max_model_len,
        "samsum_newline_eos": True,
    }
    if int(args.max_new_tokens) > 0:
        base["max_new_tokens"] = int(args.max_new_tokens)
        base["max_new_tokens_mode"] = "uniform_override"
    else:
        dataset2maxlen = load_dataset2maxlen()
        per_task = {task: int(dataset2maxlen[task]) for task in args.tasks}
        base["max_new_tokens_mode"] = "dataset2maxlen_per_task"
        base["max_new_tokens_per_task"] = per_task
        base["dataset2maxlen_sha256"] = sha256_file(DATASET2MAXLEN_CONFIG)
    return base


def generation_config_for_task(args: argparse.Namespace, task: str) -> Dict[str, Any]:
    return {
        "temperature": 0.0,
        "top_p": 1.0,
        "do_sample": False,
        "num_beams": 1,
        "use_cache": True,
        "max_new_tokens": task_max_new_tokens(args, task),
        "max_model_len": args.max_model_len,
        "samsum_newline_eos": task == "samsum",
        "source": (
            "uniform_override"
            if int(args.max_new_tokens) > 0
            else f"{DATASET2MAXLEN_CONFIG.name}:{task}"
        ),
    }


def generation_config_hash_for_task(args: argparse.Namespace, task: str) -> str:
    return sha256_text(canonical_json(generation_config_for_task(args, task)))


def scorer_config() -> Dict[str, Any]:
    return {
        "source": str(SCORER_SOURCE),
        "source_sha256": sha256_file(SCORER_SOURCE),
        "entry_point": "longbench_eval.scorer",
    }


def budget_schedule(
    prompt_token_count: int,
    block_size: int,
    final_retain_tokens: int,
    rescue_ratio: float,
    budget_type: str,
) -> Dict[str, int]:
    total_blocks = int(math.ceil(prompt_token_count / max(1, block_size)))
    k = min(total_blocks, int(math.ceil(final_retain_tokens / max(1, block_size))))
    inactive = max(0, total_blocks - k)
    m = 0
    if budget_type == "K+M" and inactive > 0 and rescue_ratio > 0.0 and k > 0:
        m = min(inactive, max(1, int(round(k * rescue_ratio))))
    return {
        "prompt_block_count": total_blocks,
        "K": k,
        "M": m,
        "target_K_plus_M": k + m,
        "pruned_candidate_count": inactive,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["preflight", "gate_audit", "smoke", "run"],
        default="preflight",
    )
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--model_path", type=str, default=str(DEFAULT_MODEL))
    parser.add_argument("--model_revision", type=str, default="")
    parser.add_argument("--longbench_dir", type=str, default=str(DEFAULT_LONGBENCH))
    parser.add_argument("--tasks", type=str, default="qasper,musique")
    parser.add_argument("--methods", type=str, default=",".join(METHOD_ORDER))
    parser.add_argument("--allow_method_subset", action="store_true")
    parser.add_argument("--max_samples_per_task", type=int, default=3)
    parser.add_argument(
        "--sample_indices_by_task",
        type=str,
        default="",
        help="optional exact sample selection, e.g. hotpotqa:0,56;musique:20,50",
    )
    parser.add_argument("--max_new_tokens", type=int, default=0,
                        help="0 = use configs/dataset2maxlen.json per task (recommended)")
    parser.add_argument("--decode_policy", choices=["original", "reuse_prefill"], default="reuse_prefill")
    parser.add_argument("--base_scoring", choices=["token_unit_qk", "delta_o"], default="delta_o")
    parser.add_argument("--rescue_ratio", type=float, default=0.05)
    parser.add_argument("--prune_layers", type=str, default="9,18,26")
    parser.add_argument("--retain_tokens", type=str, default="12288,6144,4096")
    parser.add_argument("--block_size", type=int, default=64)
    parser.add_argument("--token_unit_size", type=int, default=8)
    parser.add_argument("--max_model_len", type=int, default=32768)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min_free_gpu_gb", type=float, default=20.0)
    parser.add_argument("--gate_mode", choices=["shadow", "apply"], default="shadow")
    parser.add_argument(
        "--gate_policy",
        choices=["conservative_v1", "gate_a_removal_risk"],
        default="conservative_v1",
    )
    parser.add_argument("--gate_default_policy", choices=["original"], default="original")
    parser.add_argument("--gate_rescue_margin_min", type=float, default=0.25)
    parser.add_argument("--gate_coverage_gap_min", type=float, default=0.0)
    parser.add_argument("--gate_deletion_risk_max", type=float, default=0.20)
    parser.add_argument(
        "--gate_feature_version",
        choices=[
            "selector_native_v2",
            "selector_native_v3",
            "selector_native_v3_1",
        ],
        default="selector_native_v2",
    )
    parser.add_argument(
        "--gate_sentence_jaccard_novel_max", type=float, default=0.35
    )
    parser.add_argument(
        "--retrospective_result_dir",
        type=str,
        default=str(DEFAULT_RETROSPECTIVE_RESULT),
        help="read-only historical score labels for post-hoc association",
    )
    parser.add_argument("--confirm_non_smoke", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="resume an interrupted run in --output_dir; skip completed cells")
    return parser.parse_args()


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    args.tasks = parse_csv(args.tasks)
    args.methods = parse_csv(args.methods)
    args.prune_layers_list = parse_int_csv(args.prune_layers)
    args.retain_tokens_list = parse_int_csv(args.retain_tokens)
    args.sample_indices_by_task_map = parse_sample_indices_by_task(
        args.sample_indices_by_task
    )
    unknown = [method for method in args.methods if method not in METHOD_SPECS]
    if unknown:
        raise ClosureError(f"unknown methods: {unknown}; allowed={sorted(METHOD_SPECS)}")
    if not args.allow_method_subset and args.methods != METHOD_ORDER:
        raise ClosureError(
            "fairness closure requires all five methods in canonical order; "
            "use --allow_method_subset only for method-local development"
        )
    if len(args.prune_layers_list) != len(args.retain_tokens_list) or not args.prune_layers_list:
        raise ClosureError("--prune_layers and --retain_tokens must be nonempty and have equal lengths")
    if args.max_samples_per_task <= 0:
        raise ClosureError("--max_samples_per_task must be positive")
    for task, indices in args.sample_indices_by_task_map.items():
        if task not in args.tasks:
            raise ClosureError(
                f"--sample_indices_by_task includes task outside --tasks: {task}"
            )
        if len(indices) != args.max_samples_per_task:
            raise ClosureError(
                f"{task}: selected sample count={len(indices)} does not match "
                f"--max_samples_per_task={args.max_samples_per_task}"
            )
    if args.mode == "smoke" and args.max_samples_per_task > 3:
        raise ClosureError("smoke mode is capped at 3 samples per task")
    if args.mode == "gate_audit":
        if args.methods != ["gated_budget_neutral_drm_swap"]:
            raise ClosureError(
                "gate_audit runs only gated_budget_neutral_drm_swap"
            )
        if args.gate_mode != "shadow":
            raise ClosureError("gate_audit is shadow-only")
        if args.max_samples_per_task > 200:
            raise ClosureError("gate_audit is capped at 200 samples per task")
    if args.mode == "run" and not args.confirm_non_smoke:
        raise ClosureError("run mode requires explicit --confirm_non_smoke")
    if args.rescue_ratio < 0.0:
        raise ClosureError("--rescue_ratio must be nonnegative")
    if args.gate_default_policy != "original":
        raise ClosureError("gated diagnostic must default to Original")
    if args.gate_rescue_margin_min < 0.0:
        raise ClosureError("--gate_rescue_margin_min must be nonnegative")
    if not 0.0 <= args.gate_deletion_risk_max <= 1.0:
        raise ClosureError("--gate_deletion_risk_max must be in [0, 1]")
    if not 0.0 <= args.gate_sentence_jaccard_novel_max <= 1.0:
        raise ClosureError(
            "--gate_sentence_jaccard_novel_max must be in [0, 1]"
        )
    dataset2maxlen = load_dataset2maxlen()
    for task in args.tasks:
        if task not in dataset2maxlen:
            raise ClosureError(
                f"task {task} missing from {DATASET2MAXLEN_CONFIG}; "
                "cannot run fair LongBench diagnostic"
            )
    if args.mode == "run" and int(args.max_new_tokens) > 0:
        raise ClosureError(
            "run-mode 64-sample diagnostic requires task-specific max_new_tokens; "
            "omit --max_new_tokens or pass --max_new_tokens 0"
        )
    model_path = Path(args.model_path)
    longbench_dir = Path(args.longbench_dir)
    if not model_path.exists():
        raise ClosureError(f"model path does not exist locally: {model_path}")
    if not longbench_dir.exists():
        raise ClosureError(f"LongBench path does not exist: {longbench_dir}")
    if not PYTHON.exists():
        raise ClosureError(f"repo Python not found: {PYTHON}")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_path = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else REPO_ROOT / "results" / f"drm_fairness_closure_{timestamp}"
    )
    historical = REPO_ROOT / "results" / "rescuekv_drm_add_full_longbench_20260523_233633"
    try:
        args.output_path.relative_to(historical)
    except ValueError:
        pass
    else:
        raise ClosureError(f"refusing to write inside historical result path: {historical}")
    return args


def shared_protocol(args: argparse.Namespace) -> Dict[str, Any]:
    model_path = Path(args.model_path)
    return {
        "model_name": str(model_path),
        "model_revision_or_unknown": resolve_model_revision(model_path, args.model_revision),
        "longbench_dir": str(Path(args.longbench_dir)),
        "tasks": args.tasks,
        "max_samples_per_task": args.max_samples_per_task,
        "sample_indices_by_task": args.sample_indices_by_task_map,
        "decode_policy": args.decode_policy,
        "base_scoring": args.base_scoring,
        "max_new_tokens": args.max_new_tokens,
        "generation_config": generation_config(args),
        "generation_config_hash": sha256_text(canonical_json(generation_config(args))),
        "generation_config_hash_by_task": {
            task: generation_config_hash_for_task(args, task) for task in args.tasks
        },
        "prompt_wrapper_hash": prompt_wrapper_hash(),
        "scorer": scorer_config(),
        "scorer_hash": scorer_config()["source_sha256"],
        "prune_layers": args.prune_layers_list,
        # Cache pruning at layer L is consumed by attention at layer L+1.
        # The used-block audit must therefore read the next layer.
        "final_used_block_audit_layer": args.prune_layers_list[-1] + 1,
        "retain_tokens": args.retain_tokens_list,
        "block_size": args.block_size,
        "token_unit_size": args.token_unit_size,
        "rescue_ratio": args.rescue_ratio,
        "seed": args.seed,
        "gate_mode": args.gate_mode,
        "gate_policy": args.gate_policy,
        "gate_default_policy": args.gate_default_policy,
        "gate_rescue_margin_min": args.gate_rescue_margin_min,
        "gate_coverage_gap_min": args.gate_coverage_gap_min,
        "gate_deletion_risk_max": args.gate_deletion_risk_max,
        "gate_feature_version": args.gate_feature_version,
        "gate_sentence_jaccard_novel_max": (
            args.gate_sentence_jaccard_novel_max
        ),
    }


def method_matrix(args: argparse.Namespace) -> List[Dict[str, Any]]:
    protocol = shared_protocol(args)
    rows: List[Dict[str, Any]] = []
    for method in args.methods:
        spec = METHOD_SPECS[method]
        rows.append(
            {
                "method_key": method,
                "method_name": spec["method_name"],
                "runner_method": "sliminfer_variant",
                "decode_policy": protocol["decode_policy"],
                "base_scoring": protocol["base_scoring"],
                "rescue_policy": spec["rescue_policy"],
                "rescue_score": spec["rescue_score"],
                "runner_rescue_score": spec["runner_rescue_score"],
                "target_budget_type": spec["target_budget_type"],
                "final_budget_type": spec["final_budget_type"],
                "rescue_ratio": args.rescue_ratio,
                "gate_mode": (
                    args.gate_mode
                    if method in GATED_SWAP_METHODS
                    else "not_applicable"
                ),
                "gate_policy": (
                    args.gate_policy
                    if method in GATED_SWAP_METHODS
                    else "not_applicable"
                ),
            }
        )
    return rows


def validate_static_matrix(rows: Sequence[Mapping[str, Any]], allow_subset: bool) -> List[str]:
    errors: List[str] = []
    if not rows:
        return ["empty method matrix"]
    for field in ["runner_method", "decode_policy", "base_scoring", "rescue_ratio"]:
        values = {canonical_json(row.get(field)) for row in rows}
        if len(values) != 1:
            errors.append(f"method matrix differs on shared field {field}: {sorted(values)}")
    add_rows = [row for row in rows if row.get("method_key") in {"random_add", "delta_o_add", "drm_add"}]
    if len(add_rows) >= 2:
        for field in [
            "runner_method",
            "decode_policy",
            "base_scoring",
            "rescue_policy",
            "target_budget_type",
            "final_budget_type",
            "rescue_ratio",
        ]:
            if len({canonical_json(row.get(field)) for row in add_rows}) != 1:
                errors.append(f"Add trio differs on {field}")
        expected_by_method = {
            "random_add": "random",
            "delta_o_add": "delta_o",
            "drm_add": "drm",
        }
        expected_scores = {
            expected_by_method[str(row.get("method_key"))] for row in add_rows
        }
        actual_scores = {str(row.get("rescue_score")) for row in add_rows}
        if actual_scores != expected_scores:
            errors.append(f"Add trio rescue scores are {actual_scores}, expected {expected_scores}")
    elif not allow_subset:
        errors.append("Add trio is incomplete")
    return errors


def validate_preflight_matrix(
    args: argparse.Namespace,
    rows: Sequence[Mapping[str, Any]],
) -> List[str]:
    errors: List[str] = []
    expected_count = len(args.methods) * len(args.tasks) * args.max_samples_per_task
    if len(rows) != expected_count:
        errors.append(f"preflight rows={len(rows)}, expected={expected_count}")
    grouped: Dict[Tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("method_key")), str(row.get("task")))].append(row)
    for task in args.tasks:
        sequences: Dict[str, List[Tuple[int, str]]] = {}
        hashes: Dict[str, List[Tuple[str, str, str]]] = {}
        for method in args.methods:
            cell_rows = sorted(
                grouped.get((method, task), []),
                key=lambda row: int(row.get("sample_index", -1)),
            )
            sequences[method] = [
                (int(row.get("sample_index", -1)), str(row.get("sample_id", "")))
                for row in cell_rows
            ]
            hashes[method] = [
                (
                    str(row.get("sample_id", "")),
                    str(row.get("prompt_hash", "")),
                    str(row.get("generation_config_hash", "")),
                )
                for row in cell_rows
            ]
            if len(cell_rows) != args.max_samples_per_task:
                errors.append(f"{method}/{task}: preflight samples={len(cell_rows)}")
            if any(int(row.get("max_new_tokens", -1)) != task_max_new_tokens(args, task) for row in cell_rows):
                errors.append(f"{method}/{task}: task-specific max_new_tokens mismatch")
        if len({canonical_json(value) for value in sequences.values()}) != 1:
            errors.append(f"{task}: preflight sample_id sequences differ across methods")
        if len({canonical_json(value) for value in hashes.values()}) != 1:
            errors.append(f"{task}: preflight prompt/generation hashes differ across methods")
    return errors


def cpu_preflight(args: argparse.Namespace) -> List[Dict[str, Any]]:
    from transformers import AutoTokenizer

    protocol = shared_protocol(args)
    prompts = read_json(PROMPT_CONFIG, {})
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, local_files_only=True)
    rows: List[Dict[str, Any]] = []
    final_retain = args.retain_tokens_list[-1]
    for task in args.tasks:
        samples = load_longbench_rows(
            Path(args.longbench_dir),
            task,
            args.max_samples_per_task,
            args.sample_indices_by_task_map,
        )
        if task not in prompts:
            raise ClosureError(f"missing prompt template for task {task}")
        for position, sample in enumerate(samples):
            index = int(sample.get("__longbench_original_index", position))
            prompt = prompts[task].format(**sample)
            tokenized = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
            if len(tokenized) > args.max_model_len:
                half = args.max_model_len // 2
                prompt = tokenizer.decode(tokenized[:half], skip_special_tokens=True) + tokenizer.decode(
                    tokenized[-half:], skip_special_tokens=True
                )
            if task not in NO_CHAT_TASKS:
                prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            prompt_tokens = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
            sid = sample_id_of(sample, index)
            for method in args.methods:
                spec = METHOD_SPECS[method]
                schedule = budget_schedule(
                    len(prompt_tokens),
                    args.block_size,
                    final_retain,
                    args.rescue_ratio,
                    spec["target_budget_type"],
                )
                rows.append(
                    {
                        "method_key": method,
                        "method_name": spec["method_name"],
                        "task": task,
                        "sample_id": sid,
                        "sample_index": index,
                        "prompt_token_count": len(prompt_tokens),
                        "prompt_hash": sha256_text(prompt),
                        "prompt_wrapper_hash": protocol["prompt_wrapper_hash"],
                        "decode_policy": args.decode_policy,
                        "base_scoring": args.base_scoring,
                        "target_budget_type": spec["target_budget_type"],
                        "max_new_tokens": task_max_new_tokens(args, task),
                        "generation_config_hash": generation_config_hash_for_task(args, task),
                        "used_block_audit_layer": args.prune_layers_list[-1] + 1,
                        **schedule,
                    }
                )
    return rows


def gpu_free_mib() -> Tuple[int, Dict[str, Any]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.free",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = subprocess.run(command, text=True, capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise ClosureError(f"nvidia-smi preflight failed: {exc}") from exc
    rows = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if not rows:
        raise ClosureError("nvidia-smi returned no GPU rows")
    first = [part.strip() for part in rows[0].split(",")]
    if len(first) != 4:
        raise ClosureError(f"unexpected nvidia-smi output: {rows[0]}")
    return int(first[3]), {
        "gpu_index": int(first[0]),
        "gpu_name": first[1],
        "memory_total_mib": int(first[2]),
        "memory_free_mib": int(first[3]),
    }


def build_cell_command(
    args: argparse.Namespace,
    method: str,
    task: str,
    cell_dir: Path,
) -> List[str]:
    spec = METHOD_SPECS[method]
    audit_dir = cell_dir / "audit"
    return [
        str(PYTHON),
        str(BASE_RUNNER),
        "--model_path",
        args.model_path,
        "--tokenizer_path",
        args.model_path,
        "--model_family",
        "qwen2",
        "--dtype",
        "fp16",
        "--device",
        "cuda",
        "--backend",
        "flash_attention_2",
        "--eval",
        "gate_audit" if args.mode == "gate_audit" else "longbench",
        "--longbench_dir",
        args.longbench_dir,
        "--longbench_tasks",
        task,
        "--longbench_max_samples",
        str(args.max_samples_per_task),
        "--longbench_sample_indices_by_task",
        args.sample_indices_by_task,
        "--longbench_subset_name",
        "drm_fairness_closure",
        "--generation_max_new_tokens",
        str(args.max_new_tokens if int(args.max_new_tokens) > 0 else 0),
        "--max_model_len",
        str(args.max_model_len),
        "--method",
        "sliminfer_variant",
        "--scoring",
        args.base_scoring,
        "--fairness_base_scoring",
        args.base_scoring,
        "--decode_policy",
        args.decode_policy,
        "--rescue_policy",
        spec["rescue_policy"],
        "--rescue_score",
        spec["runner_rescue_score"],
        "--rescue_ratio",
        str(args.rescue_ratio),
        "--gate_mode",
        args.gate_mode,
        "--gate_policy",
        args.gate_policy,
        "--gate_default_policy",
        args.gate_default_policy,
        "--gate_rescue_margin_min",
        str(args.gate_rescue_margin_min),
        "--gate_coverage_gap_min",
        str(args.gate_coverage_gap_min),
        "--gate_deletion_risk_max",
        str(args.gate_deletion_risk_max),
        "--gate_feature_version",
        args.gate_feature_version,
        "--gate_sentence_jaccard_novel_max",
        str(args.gate_sentence_jaccard_novel_max),
        "--prune_layers",
        args.prune_layers,
        "--retain_tokens",
        args.retain_tokens,
        "--block_size",
        str(args.block_size),
        "--token_unit_size",
        str(args.token_unit_size),
        "--local_query_window",
        "4",
        "--swap_threshold",
        "0.9",
        "--temperature",
        "0",
        "--top_p",
        "1.0",
        "--seed",
        str(args.seed),
        "--fd_alpha_o",
        "1.0",
        "--fd_beta_h",
        "0.0",
        "--fd_gamma_s",
        "0.0",
        "--alpha_diff",
        "0.5",
        "--lambda_red",
        "0.25",
        "--eta_rescue",
        "0.25",
        "--eta_active",
        "0.25",
        "--candidate_pool_ratio",
        "0.3",
        "--sampling_temperature",
        "1.0",
        "--stochastic_mode",
        "gumbel",
        "--gumbel_tau",
        "0.05",
        "--output_dir",
        str(cell_dir),
        "--isolated_run_id",
        f"fairness__{method}__{task}",
        "--rescue_audit_path",
        str(audit_dir / "rescue_policy_audit.jsonl"),
        "--debug_used_block_audit_path",
        str(audit_dir / "used_blocks.jsonl"),
        "--debug_used_block_include_full",
        "true",
        "--debug_used_block_full_layers",
        ",".join(str(layer + 1) for layer in args.prune_layers_list),
        "--debug_generation_jsonl",
        str(audit_dir / "generations.jsonl"),
        "--profile_io",
        "--profile_memory",
        "--profile_score_overhead",
        "--save_metrics_json",
        "--smoke_mode",
        "1" if args.mode == "smoke" else "0",
        "--max_smoke_samples",
        str(args.max_samples_per_task),
    ]


def selected_prefill_records(cell_dir: Path, target_layer: int) -> Dict[str, Dict[str, Any]]:
    selected: Dict[str, Dict[str, Any]] = {}
    for row in iter_jsonl(cell_dir / "audit" / "used_blocks.jsonl"):
        if str(row.get("phase")) != "prefill" or int(row.get("layer_id", -1)) != target_layer:
            continue
        sid = str(row.get("sample_id", ""))
        if sid and sid not in selected:
            selected[sid] = row
    return selected


def per_layer_prefill_schedule(
    cell_dir: Path,
    prune_layers: Sequence[int],
) -> Dict[str, List[Dict[str, int]]]:
    used_layers = {int(layer) + 1: int(layer) for layer in prune_layers}
    selected: Dict[str, Dict[int, Dict[str, int]]] = defaultdict(dict)
    for row in iter_jsonl(cell_dir / "audit" / "used_blocks.jsonl"):
        used_layer = int(row.get("layer_id", -1))
        if str(row.get("phase")) != "prefill" or used_layer not in used_layers:
            continue
        sid = str(row.get("sample_id", ""))
        if not sid or used_layer in selected[sid]:
            continue
        selected[sid][used_layer] = {
            "prune_layer": used_layers[used_layer],
            "used_layer": used_layer,
            "selected_block_count": int(row.get("final_block_count_used_by_attention", -1)),
            "selected_token_count": int(row.get("kv_abs_indices_count", -1)),
        }
    result: Dict[str, List[Dict[str, int]]] = {}
    expected_layers = sorted(used_layers)
    for sid, by_layer in selected.items():
        if sorted(by_layer) == expected_layers:
            result[sid] = [by_layer[layer] for layer in expected_layers]
    return result


def quick_budget_rows(args: argparse.Namespace, method: str, task: str, cell_dir: Path) -> List[Dict[str, Any]]:
    spec = METHOD_SPECS[method]
    pred_path = cell_dir / "longbench_predictions" / f"{task}.jsonl"
    predictions = read_jsonl(pred_path)
    used = selected_prefill_records(cell_dir, args.prune_layers_list[-1] + 1)
    layer_schedules = per_layer_prefill_schedule(cell_dir, args.prune_layers_list)
    rows: List[Dict[str, Any]] = []
    for prediction in predictions:
        sid = str(prediction.get("sample_id", ""))
        if sid not in used:
            raise ClosureError(f"{method}/{task}/{sid}: missing prefill used-block record at final prune layer")
        usage = used[sid]
        blocks = usage.get("used_block_ids")
        tokens = usage.get("used_tokens")
        if not isinstance(blocks, list) or not isinstance(tokens, list):
            raise ClosureError(f"{method}/{task}/{sid}: full used_blocks/used_tokens were not emitted")
        schedule = budget_schedule(
            int(usage.get("prompt_token_count", prediction.get("length", 0))),
            args.block_size,
            args.retain_tokens_list[-1],
            args.rescue_ratio,
            spec["target_budget_type"],
        )
        actual_blocks = int(usage.get("final_block_count_used_by_attention", -1))
        actual_tokens = int(usage.get("kv_abs_indices_count", -1))
        if actual_blocks != schedule["target_K_plus_M"]:
            raise ClosureError(
                f"{method}/{task}/{sid}: actual blocks={actual_blocks}, "
                f"target={schedule['target_K_plus_M']}"
            )
        if actual_blocks != len(blocks) or actual_tokens != len(tokens):
            raise ClosureError(f"{method}/{task}/{sid}: used-block/token list length mismatch")
        layer_schedule = layer_schedules.get(sid)
        if layer_schedule is None:
            raise ClosureError(f"{method}/{task}/{sid}: incomplete per-prune-layer used-block schedule")
        rows.append(
            {
                "sample_id": sid,
                "sample_index": int(prediction.get("sample_index", -1)),
                "selected_block_count": actual_blocks,
                "selected_token_count": actual_tokens,
                "used_blocks": blocks,
                "used_tokens": tokens,
                "per_layer_block_schedule": layer_schedule,
                **schedule,
            }
        )
    if len(rows) != args.max_samples_per_task:
        raise ClosureError(
            f"{method}/{task}: expected {args.max_samples_per_task} predictions, got {len(rows)}"
        )
    return rows


def compare_actual_to_larger(
    method: str,
    task: str,
    actual: Sequence[Mapping[str, Any]],
    larger: Sequence[Mapping[str, Any]],
) -> None:
    larger_by_id = {str(row["sample_id"]): row for row in larger}
    for row in actual:
        sid = str(row["sample_id"])
        reference = larger_by_id.get(sid)
        if reference is None:
            raise ClosureError(f"{method}/{task}/{sid}: sample is absent from SlimInfer larger")
        for field in [
            "target_K_plus_M",
            "selected_block_count",
            "selected_token_count",
            "per_layer_block_schedule",
        ]:
            if row.get(field) != reference.get(field):
                raise ClosureError(
                    f"{method}/{task}/{sid}: {field}={row.get(field)} does not match "
                    f"SlimInfer larger={reference.get(field)}"
                )


def compare_budget_neutral_to_original(
    task: str,
    actual: Sequence[Mapping[str, Any]],
    original: Sequence[Mapping[str, Any]],
    method: str = "budget_neutral_drm_swap",
) -> None:
    original_by_id = {str(row["sample_id"]): row for row in original}
    for row in actual:
        sid = str(row["sample_id"])
        reference = original_by_id.get(sid)
        if reference is None:
            raise ClosureError(
                f"{method}/{task}/{sid}: missing SlimInfer original reference"
            )
        for field in [
            "selected_block_count",
            "selected_token_count",
            "per_layer_block_schedule",
        ]:
            if row.get(field) != reference.get(field):
                raise ClosureError(
                    f"{method}/{task}/{sid}: {field}={row.get(field)} "
                    f"does not match SlimInfer original={reference.get(field)}"
                )


def cell_prediction_count(cell_dir: Path, task: str) -> int:
    pred_path = cell_dir / "longbench_predictions" / f"{task}.jsonl"
    return len(read_jsonl(pred_path)) if pred_path.exists() else 0


def quick_prediction_integrity(
    args: argparse.Namespace,
    method: str,
    task: str,
    cell_dir: Path,
    expected_preflight: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    predictions = read_jsonl(cell_dir / "longbench_predictions" / f"{task}.jsonl")
    if len(predictions) != args.max_samples_per_task:
        raise ClosureError(
            f"{method}/{task}: expected {args.max_samples_per_task} predictions, "
            f"got {len(predictions)}"
        )
    actual_sequence = [
        (int(row.get("sample_index", -1)), str(row.get("sample_id", "")))
        for row in predictions
    ]
    expected_sequence = [
        (int(row.get("sample_index", -1)), str(row.get("sample_id", "")))
        for row in expected_preflight
    ]
    if actual_sequence != expected_sequence:
        raise ClosureError(
            f"{method}/{task}: prediction sample sequence differs from CPU preflight; "
            f"actual={actual_sequence[:5]} expected={expected_sequence[:5]}"
        )
    if len({sid for _, sid in actual_sequence}) != len(actual_sequence):
        raise ClosureError(f"{method}/{task}: duplicate sample_id in predictions")
    for row in predictions:
        sid = str(row.get("sample_id", ""))
        pred = str(row.get("pred", ""))
        if not pred.strip():
            raise ClosureError(f"{method}/{task}/{sid}: empty prediction")
        if is_nan(row):
            raise ClosureError(f"{method}/{task}/{sid}: NaN/Inf in prediction record")
    generations = generation_records(cell_dir)
    expected_hash = {
        str(row.get("sample_id", "")): str(row.get("prompt_hash", ""))
        for row in expected_preflight
    }
    for _, sid in actual_sequence:
        generation = generations.get(sid)
        if generation is None:
            raise ClosureError(f"{method}/{task}/{sid}: missing generation audit")
        if not str(generation.get("output_text", "")).strip():
            raise ClosureError(f"{method}/{task}/{sid}: empty generation audit output")
        if str(generation.get("prompt_hash", "")) != expected_hash.get(sid):
            raise ClosureError(f"{method}/{task}/{sid}: runtime/preflight prompt_hash mismatch")
    scorer_accepts_task(task, predictions)
    return predictions


def run_cells(args: argparse.Namespace) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    minimum_mib = int(args.min_free_gpu_gb * 1024)
    actual_usage: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    sample_sequences: Dict[str, List[Tuple[int, str]]] = {}
    commands_path = args.output_path / "commands.jsonl"
    preflight_by_cell: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(args.output_path / "preflight_selector_schedule.jsonl"):
        preflight_by_cell[(str(row.get("method_key")), str(row.get("task")))].append(row)
    for key in preflight_by_cell:
        preflight_by_cell[key].sort(key=lambda row: int(row.get("sample_index", -1)))
    for method in args.methods:
        for task in args.tasks:
            cell_dir = args.output_path / "runs" / method / task
            free_mib, gpu = gpu_free_mib()
            if free_mib < minimum_mib:
                raise ClosureError(
                    f"GPU gate failed before {method}/{task}: free={free_mib} MiB, "
                    f"required={minimum_mib} MiB"
                )
            expected_preflight = preflight_by_cell.get((method, task), [])
            if len(expected_preflight) != args.max_samples_per_task:
                raise ClosureError(
                    f"{method}/{task}: CPU preflight rows={len(expected_preflight)}, "
                    f"expected={args.max_samples_per_task}"
                )
            existing_predictions = cell_prediction_count(cell_dir, task)
            if existing_predictions > args.max_samples_per_task:
                raise ClosureError(
                    f"{method}/{task}: prediction file has {existing_predictions} rows, "
                    f"expected at most {args.max_samples_per_task}"
                )
            completed = existing_predictions == args.max_samples_per_task
            if completed and not args.resume:
                raise ClosureError(f"{method}/{task}: completed cell already exists without --resume")
            if not completed:
                if cell_dir.exists() and not args.resume:
                    raise ClosureError(f"{method}/{task}: partial cell exists without --resume")
                cell_dir.mkdir(parents=True, exist_ok=args.resume)
            command = build_cell_command(args, method, task, cell_dir)
            command_record = {
                "event": "resume_skip_completed" if completed else (
                    "resume_partial" if existing_predictions > 0 else "run_new"
                ),
                "method": method,
                "task": task,
                "existing_predictions": existing_predictions,
                "gpu_preflight": gpu,
                "command": command,
            }
            with commands_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(command_record, ensure_ascii=False) + "\n")
            if not completed:
                log_path = args.output_path / "logs" / f"{method}__{task}.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                env = os.environ.copy()
                env["PYTHONPATH"] = str(REPO_ROOT)
                env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
                log_mode = "a" if args.resume and log_path.exists() else "w"
                with log_path.open(log_mode, encoding="utf-8") as log_handle:
                    if log_mode == "a":
                        log_handle.write(
                            f"\n[fairness-closure resume {datetime.now().isoformat()} "
                            f"from_prediction={existing_predictions}]\n"
                        )
                    proc = subprocess.run(
                        command,
                        cwd=str(REPO_ROOT),
                        env=env,
                        stdout=log_handle,
                        stderr=subprocess.STDOUT,
                    )
                if proc.returncode != 0:
                    log_text = log_path.read_text(encoding="utf-8", errors="replace")
                    reason = "CUDA OOM" if "out of memory" in log_text.lower() else f"returncode={proc.returncode}"
                    raise ClosureError(f"cell {method}/{task} failed ({reason}); see {log_path}")
            prediction_src = cell_dir / "longbench_predictions" / f"{task}.jsonl"
            if not prediction_src.exists():
                raise ClosureError(f"cell {method}/{task} emitted no predictions: {prediction_src}")
            predictions = quick_prediction_integrity(
                args,
                method,
                task,
                cell_dir,
                expected_preflight,
            )
            sequence = [
                (int(row.get("sample_index", -1)), str(row.get("sample_id", "")))
                for row in predictions
            ]
            reference_sequence = sample_sequences.get(task)
            if reference_sequence is None:
                sample_sequences[task] = sequence
            elif sequence != reference_sequence:
                raise ClosureError(f"{method}/{task}: sample_id sequence differs across methods")
            prediction_dst = args.output_path / "predictions" / method / f"{task}.jsonl"
            prediction_dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(prediction_src, prediction_dst)
            current = quick_budget_rows(args, method, task, cell_dir)
            actual_usage[(method, task)] = current
            if method in BUDGET_NEUTRAL_SWAP_METHODS:
                original = actual_usage.get(("sliminfer_original", task))
                if original is None:
                    raise ClosureError(
                        f"cannot budget-check {method}/{task}: "
                        "SlimInfer original must run first"
                    )
                compare_budget_neutral_to_original(task, current, original, method)
            if method in {"random_add", "delta_o_add", "drm_add"}:
                larger = actual_usage.get(("sliminfer_larger", task))
                if larger is None:
                    if not args.allow_method_subset:
                        raise ClosureError(f"cannot budget-check {method}/{task}: SlimInfer larger not run first")
                else:
                    compare_actual_to_larger(method, task, current, larger)
    return actual_usage


GATE_AUDIT_REQUIRED_FIELDS = [
    "task",
    "sample_id",
    "sample_index",
    "gate_mode",
    "gate_policy",
    "selected_policy",
    "selected_policy_before_gate",
    "selected_policy_after_gate",
    "would_gate_trigger",
    "gate_triggered",
    "gate_reason",
    "gate_feature_missing",
    "gate_feature_missing_fields",
    "rescue_boundary_margin",
    "rescue_score_top1",
    "rescue_score_top2",
    "rescue_score_mean",
    "rescue_score_min_selected",
    "rescue_score_boundary_selected",
    "rescue_score_boundary_unselected",
    "deletion_risk",
    "rescue_coverage_gap",
    "coverage_gap_v2",
    "coverage_gap_v3",
    "context_entity_novelty",
    "title_or_heading_novelty",
    "title_or_heading_available",
    "rescued_novel_sentence_ratio",
    "bridge_entity_novelty_ratio",
    "bridge_entity_count",
    "rescued_active_semantic_novelty_mean",
    "rescued_active_semantic_novelty_max",
    "rescued_active_semantic_novelty_min",
    "deletion_evidence_risk_v3",
    "candidate_pool_exhausted",
    "question_entity_units",
    "rescued_entity_units",
    "active_entity_units",
    "rescued_only_entity_units",
    "removed_entity_units",
    "removed_title_like_units",
    "removed_date_number_units",
    "answer_type_proxy",
    "coverage_gap_v3_raw",
    "coverage_gap_v3_1",
    "rescued_vs_removed_entity_gain",
    "rescued_vs_active_entity_gain",
    "sentence_novelty_gain",
    "semantic_novelty_rank",
    "semantic_novelty_rank_mean",
    "semantic_novelty_rank_min",
    "bridge_entity_gain",
    "answer_type_support_gain",
    "rescued_answer_type_support",
    "removed_answer_type_support",
    "deletion_query_overlap_risk",
    "deletion_entity_risk",
    "deletion_date_number_risk",
    "deletion_title_risk",
    "deletion_answer_type_risk",
    "deletion_base_importance_risk",
    "deletion_protected_proximity_risk",
    "deletion_risk_v3_raw",
    "deletion_risk_v3_1",
    "removed_block_risk_sources",
    "rescue_selected_min_percentile",
    "rescue_selected_mean_percentile",
    "rescue_unselected_max_percentile",
    "rescue_boundary_margin_rank",
    "candidate_pool_size",
    "candidate_pool_effective_size",
    "candidate_score_std",
    "candidate_score_iqr",
    "candidate_score_entropy_proxy",
    "candidate_score_flat",
    "would_gate_trigger_v3_1",
    "gate_reason_v3_1",
    "coverage_components_available",
    "coverage_components_missing",
    "query_overlap_rescued",
    "query_overlap_removed",
    "entity_overlap_rescued",
    "entity_overlap_removed",
    "date_number_overlap_rescued",
    "date_number_overlap_removed",
    "rescued_vs_removed_score_gap",
    "removed_score_mean",
    "removed_drop_utility_mean",
    "active_redundancy_mean",
    "active_redundancy_max",
    "removed_value_norm_mean",
    "removed_value_norm_max",
    "rescued_value_norm_mean",
    "rescued_value_norm_max",
    "removed_qk_or_delta_score_mean",
    "removed_qk_or_delta_score_max",
    "rescued_qk_or_delta_score_mean",
    "rescued_qk_or_delta_score_max",
    "delete_risk_proxy_mean",
    "delete_risk_proxy_max",
    "rescue_gain_proxy_mean",
    "rescue_gain_proxy_max",
    "residual_benefit_proxy",
    "rescue_score_margin",
    "removed_block_text_preview",
    "rescued_block_text_preview",
    "residual_proxy_unavailable_fields",
    "residual_proxy_version",
    "pruned_candidate_count",
    "swap_count",
    "candidate_rescued_block_ids",
    "candidate_removed_block_ids",
    "protected_blocks_with_reasons",
    "all_pruned_candidates",
    "selected_rescued_candidates",
    "unselected_pruned_candidates",
    "all_active_drop_candidates",
    "selected_removed_candidates",
    "fallback_used",
    "fallback_reason",
    "selected_block_count",
    "used_blocks",
    "used_tokens",
    "over_budget",
    "protected_block_violation",
]


def run_gate_audit_cells(args: argparse.Namespace) -> None:
    minimum_mib = int(args.min_free_gpu_gb * 1024)
    commands_path = args.output_path / "commands.jsonl"
    preflight_by_cell: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(args.output_path / "preflight_selector_schedule.jsonl"):
        preflight_by_cell[(str(row["method_key"]), str(row["task"]))].append(row)
    for key in preflight_by_cell:
        preflight_by_cell[key].sort(key=lambda row: int(row["sample_index"]))

    method = "gated_budget_neutral_drm_swap"
    for task in args.tasks:
        cell_dir = args.output_path / "runs" / method / task
        manifest_path = cell_dir / "gate_audit_manifest.jsonl"
        existing_rows = read_jsonl(manifest_path)
        if len(existing_rows) > args.max_samples_per_task:
            raise ClosureError(f"{method}/{task}: oversized gate audit manifest")
        expected = preflight_by_cell[(method, task)]
        actual_prefix = [
            (int(row["sample_index"]), str(row["sample_id"]))
            for row in existing_rows
        ]
        expected_prefix = [
            (int(row["sample_index"]), str(row["sample_id"]))
            for row in expected[:len(existing_rows)]
        ]
        if actual_prefix != expected_prefix:
            raise ClosureError(
                f"{method}/{task}: partial gate audit sequence mismatch"
            )
        completion_status = read_json(
            cell_dir / "gate_audit_status.json", {}
        )
        completed = (
            len(existing_rows) == args.max_samples_per_task
            and completion_status.get("status") == "ok"
            and int(completion_status.get("samples", -1))
            == args.max_samples_per_task
            and completion_status.get("quality_generation_performed") is False
            and completion_status.get("scorer_invoked") is False
        )
        if completed and not args.resume:
            raise ClosureError(f"{method}/{task}: completed gate audit cell exists")
        if cell_dir.exists() and not args.resume and not completed:
            raise ClosureError(f"{method}/{task}: partial gate audit cell exists")
        quarantine_path: Optional[Path] = None
        if args.resume and not completed and cell_dir.exists():
            partial_files = [
                path for path in cell_dir.rglob("*") if path.is_file()
            ]
            if partial_files:
                quarantine_root = (
                    args.output_path
                    / f"quarantine_partial_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    / method
                )
                quarantine_root.mkdir(parents=True, exist_ok=True)
                quarantine_path = quarantine_root / task
                if quarantine_path.exists():
                    raise ClosureError(
                        f"quarantine destination already exists: {quarantine_path}"
                    )
                print(json.dumps({
                    "event": "quarantine_incomplete_gate_cell",
                    "method": method,
                    "task": task,
                    "source": str(cell_dir),
                    "destination": str(quarantine_path),
                    "files": [
                        str(path.relative_to(cell_dir))
                        for path in partial_files
                    ],
                }, ensure_ascii=False))
                shutil.move(str(cell_dir), str(quarantine_path))
                existing_rows = []
                manifest_path = cell_dir / "gate_audit_manifest.jsonl"
        cell_dir.mkdir(parents=True, exist_ok=args.resume)
        free_mib, gpu = gpu_free_mib()
        if free_mib < minimum_mib:
            raise ClosureError(
                f"GPU gate failed before {method}/{task}: free={free_mib} MiB, "
                f"required={minimum_mib} MiB"
            )
        command = build_cell_command(args, method, task, cell_dir)
        with commands_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "event": "resume_skip_completed" if completed else (
                    "resume_restart_incomplete"
                    if quarantine_path is not None
                    else ("resume_from_empty_cell" if args.resume else "run_new")
                ),
                "method": method,
                "task": task,
                "existing_audit_rows": len(existing_rows),
                "quarantine_path": (
                    str(quarantine_path) if quarantine_path is not None else ""
                ),
                "gpu_preflight": gpu,
                "command": command,
            }, ensure_ascii=False) + "\n")
        if not completed:
            log_path = args.output_path / "logs" / f"{method}__{task}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            env = os.environ.copy()
            env["PYTHONPATH"] = str(REPO_ROOT)
            env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
            with log_path.open(
                "a" if args.resume and log_path.exists() else "w",
                encoding="utf-8",
            ) as log_handle:
                proc = subprocess.run(
                    command,
                    cwd=str(REPO_ROOT),
                    env=env,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                )
            if proc.returncode != 0:
                log_text = log_path.read_text(encoding="utf-8", errors="replace")
                reason = (
                    "CUDA OOM"
                    if "out of memory" in log_text.lower()
                    else f"returncode={proc.returncode}"
                )
                raise ClosureError(
                    f"gate audit cell {method}/{task} failed ({reason}); see {log_path}"
                )
        manifest = read_jsonl(manifest_path)
        actual_sequence = [
            (int(row["sample_index"]), str(row["sample_id"])) for row in manifest
        ]
        expected_sequence = [
            (int(row["sample_index"]), str(row["sample_id"])) for row in expected
        ]
        if actual_sequence != expected_sequence:
            raise ClosureError(f"{method}/{task}: gate audit sample sequence mismatch")
        if len(manifest) != args.max_samples_per_task:
            raise ClosureError(
                f"{method}/{task}: gate audit rows={len(manifest)}, "
                f"expected={args.max_samples_per_task}"
            )
        if any(bool(row.get("quality_generation_performed")) for row in manifest):
            raise ClosureError(f"{method}/{task}: quality generation was performed")
        if any(bool(row.get("scorer_invoked")) for row in manifest):
            raise ClosureError(f"{method}/{task}: scorer was invoked")
        status = read_json(cell_dir / "gate_audit_status.json", {})
        if status.get("quality_generation_performed") is not False:
            raise ClosureError(f"{method}/{task}: invalid gate audit status")
        prediction_path = cell_dir / "longbench_predictions" / f"{task}.jsonl"
        if prediction_path.exists() and read_jsonl(prediction_path):
            raise ClosureError(f"{method}/{task}: gate audit emitted predictions")
        # Fail before starting the next task cell. The final aggregate validator
        # repeats these checks, but doing them here enforces the audit-only
        # safety contract even when an early task exposes a malformed feature.
        gate_records = select_rescue_records(
            cell_dir, args.prune_layers_list[-1]
        )
        for sample_id, gate_record in gate_records.items():
            candidate_count = int(
                gate_record.get("pruned_candidate_count", 0) or 0
            )
            if candidate_count > 0 and bool(
                gate_record.get("gate_feature_missing")
            ):
                fields = gate_record.get("gate_feature_missing_fields", [])
                raise ClosureError(
                    f"{method}/{task}/{sample_id}: missing gate features "
                    f"after cell: {fields}"
                )
            if bool(gate_record.get("over_budget")) or bool(
                gate_record.get("protected_block_violation")
            ):
                raise ClosureError(
                    f"{method}/{task}/{sample_id}: budget/protected failure "
                    "after cell"
                )


def consolidate_gate_audit(args: argparse.Namespace) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    method = "gated_budget_neutral_drm_swap"
    target_layer = args.prune_layers_list[-1]
    used_layer = target_layer + 1
    for task in args.tasks:
        cell_dir = args.output_path / "runs" / method / task
        manifest = {
            str(row["sample_id"]): row
            for row in read_jsonl(cell_dir / "gate_audit_manifest.jsonl")
        }
        used = selected_prefill_records(cell_dir, used_layer)
        rescue = select_rescue_records(cell_dir, target_layer)
        preflight = {
            str(row["sample_id"]): row
            for row in read_jsonl(args.output_path / "preflight_selector_schedule.jsonl")
            if row.get("method_key") == method and row.get("task") == task
        }
        for sid, audit_manifest in manifest.items():
            usage = used.get(sid)
            schedule = preflight.get(sid)
            if usage is None or schedule is None:
                raise ClosureError(f"{task}/{sid}: missing gate audit usage/preflight")
            used_blocks = usage.get("used_block_ids")
            used_tokens = usage.get("used_tokens")
            if not isinstance(used_blocks, list) or not isinstance(used_tokens, list):
                raise ClosureError(f"{task}/{sid}: missing full used block/token lists")
            gate = rescue.get(sid)
            if gate is None:
                gate = {
                    "gate_mode": "shadow",
                    "gate_policy": args.gate_policy,
                    "selected_policy": "original",
                    "selected_policy_before_gate": "original",
                    "selected_policy_after_gate": "original",
                    "would_gate_trigger": False,
                    "gate_triggered": False,
                    "gate_reason": "no_final_pruning_candidates",
                    "gate_feature_missing": True,
                    "gate_feature_missing_fields": ["final_pruning_candidates"],
                    "fallback_used": True,
                    "fallback_reason": "no_final_pruning_candidates",
                    "pre_gate_active_block_ids": used_blocks,
                }
            selected_blocks = int(
                usage.get("final_block_count_used_by_attention", -1)
            )
            k = int(schedule["K"])
            candidate_rescued = [
                int(block_id)
                for block_id in gate.get("candidate_rescued_block_ids", [])
            ]
            candidate_removed = [
                int(block_id)
                for block_id in gate.get("candidate_removed_block_ids", [])
            ]
            row = {
                "task": task,
                "sample_id": sid,
                "sample_index": int(audit_manifest["sample_index"]),
                "gate_mode": str(gate.get("gate_mode", "shadow")),
                "gate_policy": str(gate.get("gate_policy", args.gate_policy)),
                "selected_policy": str(gate.get("selected_policy", "original")),
                "selected_policy_before_gate": str(
                    gate.get("selected_policy_before_gate", "original")
                ),
                "selected_policy_after_gate": str(
                    gate.get("selected_policy_after_gate", "original")
                ),
                "would_gate_trigger": bool(gate.get("would_gate_trigger", False)),
                "gate_triggered": bool(gate.get("gate_triggered", False)),
                "gate_reason": str(gate.get("gate_reason", "")),
                "gate_feature_missing": bool(
                    gate.get("gate_feature_missing", False)
                ),
                "gate_feature_missing_fields": list(
                    gate.get("gate_feature_missing_fields", [])
                ),
                "rescue_boundary_margin": gate.get("rescue_boundary_margin"),
                "rescue_score_top1": gate.get("rescue_score_top1"),
                "rescue_score_top2": gate.get("rescue_score_top2"),
                "rescue_score_mean": gate.get("rescue_score_mean"),
                "rescue_score_min_selected": gate.get(
                    "rescue_score_min_selected"
                ),
                "rescue_score_boundary_selected": gate.get(
                    "rescue_score_boundary_selected"
                ),
                "rescue_score_boundary_unselected": gate.get(
                    "rescue_score_boundary_unselected"
                ),
                "deletion_risk": gate.get("deletion_risk"),
                "rescue_coverage_gap": gate.get("rescue_coverage_gap"),
                "coverage_gap_v2": gate.get(
                    "coverage_gap_v2", gate.get("rescue_coverage_gap")
                ),
                "coverage_gap_v3": gate.get("coverage_gap_v3"),
                "context_entity_novelty": gate.get(
                    "context_entity_novelty"
                ),
                "title_or_heading_novelty": gate.get(
                    "title_or_heading_novelty"
                ),
                "title_or_heading_available": bool(
                    gate.get("title_or_heading_available", False)
                ),
                "rescued_novel_sentence_ratio": gate.get(
                    "rescued_novel_sentence_ratio"
                ),
                "bridge_entity_novelty_ratio": gate.get(
                    "bridge_entity_novelty_ratio"
                ),
                "bridge_entity_count": int(
                    gate.get("bridge_entity_count", 0) or 0
                ),
                "rescued_active_semantic_novelty_mean": gate.get(
                    "rescued_active_semantic_novelty_mean"
                ),
                "rescued_active_semantic_novelty_max": gate.get(
                    "rescued_active_semantic_novelty_max"
                ),
                "rescued_active_semantic_novelty_min": gate.get(
                    "rescued_active_semantic_novelty_min"
                ),
                "deletion_evidence_risk_v3": gate.get(
                    "deletion_evidence_risk_v3"
                ),
                "candidate_pool_exhausted": bool(
                    gate.get("candidate_pool_exhausted", False)
                ),
                "question_entity_units": list(
                    gate.get("question_entity_units", [])
                ),
                "rescued_entity_units": list(
                    gate.get("rescued_entity_units", [])
                ),
                "active_entity_units": list(
                    gate.get("active_entity_units", [])
                ),
                "rescued_only_entity_units": list(
                    gate.get("rescued_only_entity_units", [])
                ),
                "removed_entity_units": list(
                    gate.get("removed_entity_units", [])
                ),
                "removed_title_like_units": list(
                    gate.get("removed_title_like_units", [])
                ),
                "removed_date_number_units": list(
                    gate.get("removed_date_number_units", [])
                ),
                "answer_type_proxy": str(
                    gate.get("answer_type_proxy", "other")
                ),
                "coverage_gap_v3_raw": gate.get(
                    "coverage_gap_v3_raw", gate.get("coverage_gap_v3")
                ),
                "coverage_gap_v3_1": gate.get("coverage_gap_v3_1"),
                "rescued_vs_removed_entity_gain": gate.get(
                    "rescued_vs_removed_entity_gain"
                ),
                "rescued_vs_active_entity_gain": gate.get(
                    "rescued_vs_active_entity_gain"
                ),
                "sentence_novelty_gain": gate.get(
                    "sentence_novelty_gain"
                ),
                "semantic_novelty_rank": gate.get(
                    "semantic_novelty_rank"
                ),
                "semantic_novelty_rank_mean": gate.get(
                    "semantic_novelty_rank_mean"
                ),
                "semantic_novelty_rank_p50": gate.get(
                    "semantic_novelty_rank_p50"
                ),
                "semantic_novelty_rank_min": gate.get(
                    "semantic_novelty_rank_min"
                ),
                "bridge_entity_gain": gate.get("bridge_entity_gain"),
                "answer_type_support_gain": gate.get(
                    "answer_type_support_gain"
                ),
                "rescued_answer_type_support": gate.get(
                    "rescued_answer_type_support"
                ),
                "removed_answer_type_support": gate.get(
                    "removed_answer_type_support"
                ),
                "deletion_query_overlap_risk": gate.get(
                    "deletion_query_overlap_risk"
                ),
                "deletion_entity_risk": gate.get(
                    "deletion_entity_risk"
                ),
                "deletion_date_number_risk": gate.get(
                    "deletion_date_number_risk"
                ),
                "deletion_title_risk": gate.get(
                    "deletion_title_risk"
                ),
                "deletion_answer_type_risk": gate.get(
                    "deletion_answer_type_risk"
                ),
                "deletion_base_importance_risk": gate.get(
                    "deletion_base_importance_risk"
                ),
                "deletion_protected_proximity_risk": gate.get(
                    "deletion_protected_proximity_risk"
                ),
                "deletion_risk_v3_raw": gate.get(
                    "deletion_risk_v3_raw",
                    gate.get("deletion_evidence_risk_v3"),
                ),
                "deletion_risk_v3_1": gate.get(
                    "deletion_risk_v3_1"
                ),
                "removed_block_risk_sources": list(
                    gate.get("removed_block_risk_sources", [])
                ),
                "rescue_selected_min_percentile": gate.get(
                    "rescue_selected_min_percentile"
                ),
                "rescue_selected_mean_percentile": gate.get(
                    "rescue_selected_mean_percentile"
                ),
                "rescue_unselected_max_percentile": gate.get(
                    "rescue_unselected_max_percentile"
                ),
                "rescue_boundary_margin_rank": gate.get(
                    "rescue_boundary_margin_rank"
                ),
                "candidate_pool_size": int(
                    gate.get("candidate_pool_size", 0) or 0
                ),
                "candidate_pool_effective_size": int(
                    gate.get("candidate_pool_effective_size", 0) or 0
                ),
                "candidate_score_std": gate.get("candidate_score_std"),
                "candidate_score_iqr": gate.get("candidate_score_iqr"),
                "candidate_score_entropy_proxy": gate.get(
                    "candidate_score_entropy_proxy"
                ),
                "candidate_score_flat": bool(
                    gate.get("candidate_score_flat", False)
                ),
                "would_gate_trigger_v3_1": bool(
                    gate.get("would_gate_trigger_v3_1", False)
                ),
                "gate_reason_v3_1": str(
                    gate.get(
                        "gate_reason_v3_1",
                        gate.get("gate_reason", "not_applicable")
                        if args.gate_feature_version
                        == "selector_native_v3_1"
                        else "not_applicable",
                    )
                ),
                "coverage_components_available": list(
                    gate.get("coverage_components_available", [])
                ),
                "coverage_components_missing": list(
                    gate.get("coverage_components_missing", [])
                ),
                "query_overlap_rescued": gate.get("query_overlap_rescued"),
                "query_overlap_removed": gate.get("query_overlap_removed"),
                "entity_overlap_rescued": gate.get("entity_overlap_rescued"),
                "entity_overlap_removed": gate.get("entity_overlap_removed"),
                "date_number_overlap_rescued": gate.get(
                    "date_number_overlap_rescued"
                ),
                "date_number_overlap_removed": gate.get(
                    "date_number_overlap_removed"
                ),
                "rescued_vs_removed_score_gap": gate.get(
                    "rescued_vs_removed_score_gap"
                ),
                "removed_score_mean": gate.get("removed_score_mean"),
                "removed_drop_utility_mean": gate.get(
                    "removed_drop_utility_mean"
                ),
                "active_redundancy_mean": gate.get("active_redundancy_mean"),
                "active_redundancy_max": gate.get("active_redundancy_max"),
                "removed_value_norm_mean": gate.get("removed_value_norm_mean"),
                "removed_value_norm_max": gate.get("removed_value_norm_max"),
                "rescued_value_norm_mean": gate.get("rescued_value_norm_mean"),
                "rescued_value_norm_max": gate.get("rescued_value_norm_max"),
                "removed_qk_or_delta_score_mean": gate.get(
                    "removed_qk_or_delta_score_mean"
                ),
                "removed_qk_or_delta_score_max": gate.get(
                    "removed_qk_or_delta_score_max"
                ),
                "rescued_qk_or_delta_score_mean": gate.get(
                    "rescued_qk_or_delta_score_mean"
                ),
                "rescued_qk_or_delta_score_max": gate.get(
                    "rescued_qk_or_delta_score_max"
                ),
                "delete_risk_proxy_mean": gate.get("delete_risk_proxy_mean"),
                "delete_risk_proxy_max": gate.get("delete_risk_proxy_max"),
                "rescue_gain_proxy_mean": gate.get("rescue_gain_proxy_mean"),
                "rescue_gain_proxy_max": gate.get("rescue_gain_proxy_max"),
                "residual_benefit_proxy": gate.get("residual_benefit_proxy"),
                "rescue_score_margin": gate.get("rescue_score_margin"),
                "removed_block_text_preview": list(
                    gate.get("removed_block_text_preview", [])
                ),
                "rescued_block_text_preview": list(
                    gate.get("rescued_block_text_preview", [])
                ),
                "residual_proxy_unavailable_fields": list(
                    gate.get("residual_proxy_unavailable_fields", [])
                ),
                "residual_proxy_version": str(
                    gate.get("residual_proxy_version", "")
                ),
                "pruned_candidate_count": max(
                    0, int(schedule["prompt_block_count"]) - k
                ),
                "swap_count": int(gate.get("swap_count", 0) or 0),
                "candidate_swap_count": int(
                    gate.get("candidate_swap_count", 0) or 0
                ),
                "candidate_rescued_block_ids": candidate_rescued,
                "candidate_removed_block_ids": candidate_removed,
                "rescued_block_ids": [
                    int(block_id) for block_id in gate.get("rescued_block_ids", [])
                ],
                "removed_block_ids": [
                    int(block_id) for block_id in gate.get("removed_block_ids", [])
                ],
                "protected_blocks_with_reasons": gate.get(
                    "protected_blocks_with_reasons", []
                ),
                "all_pruned_candidates": gate.get("all_pruned_candidates", []),
                "selected_rescued_candidates": gate.get(
                    "selected_rescued_candidates", []
                ),
                "unselected_pruned_candidates": gate.get(
                    "unselected_pruned_candidates", []
                ),
                "all_active_drop_candidates": gate.get(
                    "all_active_drop_candidates", []
                ),
                "selected_removed_candidates": gate.get(
                    "selected_removed_candidates", []
                ),
                "pre_gate_active_block_ids": [
                    int(block_id)
                    for block_id in gate.get(
                        "pre_gate_active_block_ids", used_blocks
                    )
                ],
                "fallback_used": bool(gate.get("fallback_used", False)),
                "fallback_reason": str(gate.get("fallback_reason", "") or ""),
                "K": k,
                "selected_block_count": selected_blocks,
                "used_blocks": used_blocks,
                "used_tokens": used_tokens,
                "over_budget": selected_blocks > k,
                "protected_block_violation": bool(
                    gate.get("protected_block_violation", False)
                ),
                "prompt_hash": str(audit_manifest.get("prompt_hash", "")),
                "prompt_token_count": int(
                    audit_manifest.get("prompt_token_count", 0)
                ),
                "prefill_elapsed_s": float(
                    audit_manifest.get("prefill_elapsed_s", 0.0)
                ),
                "quality_generation_performed": False,
                "scorer_invoked": False,
                "gate_feature_version": str(
                    gate.get("gate_feature_version", args.gate_feature_version)
                ),
            }
            rows.append(row)
    rows.sort(key=lambda row: (args.tasks.index(row["task"]), row["sample_index"]))
    return rows


def gate_feature_distribution(values: Sequence[Any]) -> Dict[str, Any]:
    numeric = sorted(
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    )
    if not numeric:
        return {
            "n_available": 0,
            "mean": None,
            "median": None,
            "p25": None,
            "p75": None,
            "p90": None,
            "p95": None,
            "max": None,
            "nonzero_rate": None,
        }
    return {
        "n_available": len(numeric),
        "mean": round(sum(numeric) / len(numeric), 8),
        "median": round(percentile(numeric, 0.50), 8),
        "p25": round(percentile(numeric, 0.25), 8),
        "p75": round(percentile(numeric, 0.75), 8),
        "p90": round(percentile(numeric, 0.90), 8),
        "p95": round(percentile(numeric, 0.95), 8),
        "max": round(max(numeric), 8),
        "nonzero_rate": round(
            sum(abs(value) > 1e-12 for value in numeric) / len(numeric), 8
        ),
    }


def validate_and_summarize_gate_audit(
    args: argparse.Namespace,
    rows: Sequence[Mapping[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    errors: List[str] = []
    missing_reasons: Dict[str, int] = defaultdict(int)
    numeric_feature_names = [
        "coverage_gap_v2",
        "coverage_gap_v3",
        "coverage_gap_v3_1",
        "rescued_vs_removed_entity_gain",
        "rescued_vs_active_entity_gain",
        "sentence_novelty_gain",
        "semantic_novelty_rank",
        "semantic_novelty_rank_mean",
        "semantic_novelty_rank_min",
        "bridge_entity_gain",
        "answer_type_support_gain",
        "context_entity_novelty",
        "title_or_heading_novelty",
        "rescued_novel_sentence_ratio",
        "bridge_entity_novelty_ratio",
        "rescued_active_semantic_novelty_mean",
        "rescued_active_semantic_novelty_max",
        "rescued_active_semantic_novelty_min",
        "deletion_evidence_risk_v3",
        "coverage_gap_v3_1",
        "rescued_vs_removed_entity_gain",
        "rescued_vs_active_entity_gain",
        "sentence_novelty_gain",
        "semantic_novelty_rank",
        "semantic_novelty_rank_mean",
        "semantic_novelty_rank_min",
        "bridge_entity_gain",
        "answer_type_support_gain",
        "deletion_query_overlap_risk",
        "deletion_entity_risk",
        "deletion_date_number_risk",
        "deletion_title_risk",
        "deletion_answer_type_risk",
        "deletion_base_importance_risk",
        "deletion_protected_proximity_risk",
        "deletion_risk_v3_1",
        "rescue_selected_min_percentile",
        "rescue_selected_mean_percentile",
        "rescue_unselected_max_percentile",
        "rescue_boundary_margin_rank",
        "candidate_score_std",
        "candidate_score_iqr",
        "candidate_score_entropy_proxy",
        "rescue_boundary_margin",
    ]
    for row in rows:
        missing = [field for field in GATE_AUDIT_REQUIRED_FIELDS if field not in row]
        if missing:
            errors.append(f"{row.get('task')}/{row.get('sample_id')}: missing={missing}")
        if row.get("gate_mode") != "shadow":
            errors.append(f"{row['task']}/{row['sample_id']}: not shadow")
        if row.get("gate_feature_version") != args.gate_feature_version:
            errors.append(
                f"{row['task']}/{row['sample_id']}: gate feature version mismatch"
            )
        if row.get("selected_policy_after_gate") != "original":
            errors.append(f"{row['task']}/{row['sample_id']}: not Original")
        if bool(row.get("gate_triggered")):
            errors.append(f"{row['task']}/{row['sample_id']}: gate triggered")
        if int(row.get("selected_block_count", -1)) != int(row.get("K", -2)):
            errors.append(f"{row['task']}/{row['sample_id']}: selected count != K")
        if list(row.get("used_blocks", [])) != list(
            row.get("pre_gate_active_block_ids", [])
        ):
            errors.append(f"{row['task']}/{row['sample_id']}: shadow changed used blocks")
        if bool(row.get("over_budget")) or bool(
            row.get("protected_block_violation")
        ):
            errors.append(f"{row['task']}/{row['sample_id']}: budget/protected failure")
        if bool(row.get("quality_generation_performed")) or bool(
            row.get("scorer_invoked")
        ):
            errors.append(f"{row['task']}/{row['sample_id']}: quality path invoked")
        if bool(row.get("gate_feature_missing")):
            for reason in row.get("gate_feature_missing_fields", []) or ["unknown"]:
                missing_reasons[str(reason)] += 1
            if int(row.get("pruned_candidate_count", 0)) > 0:
                errors.append(
                    f"{row['task']}/{row['sample_id']}: unexpected missing gate features"
                )
        if bool(row.get("candidate_pool_exhausted")):
            if row.get("gate_reason") != "candidate_pool_exhausted":
                errors.append(
                    f"{row['task']}/{row['sample_id']}: exhausted pool reason mismatch"
                )
            if row.get("selected_policy_after_gate") != "original":
                errors.append(
                    f"{row['task']}/{row['sample_id']}: exhausted pool did not use Original"
                )
        if args.gate_feature_version == "selector_native_v3_1":
            if bool(row.get("would_gate_trigger_v3_1")) != bool(
                row.get("would_gate_trigger")
            ):
                errors.append(
                    f"{row['task']}/{row['sample_id']}: v3_1 trigger mismatch"
                )
            if str(row.get("gate_reason_v3_1")) != str(
                row.get("gate_reason")
            ):
                errors.append(
                    f"{row['task']}/{row['sample_id']}: v3_1 reason mismatch"
                )
            if bool(row.get("candidate_score_flat")) and bool(
                row.get("would_gate_trigger_v3_1")
            ):
                errors.append(
                    f"{row['task']}/{row['sample_id']}: flat pool triggered"
                )
            if (
                not bool(row.get("gate_feature_missing"))
                and int(row.get("pruned_candidate_count", 0)) > 0
                and len(row.get("coverage_components_available", [])) < 3
            ):
                errors.append(
                    f"{row['task']}/{row['sample_id']}: fewer than 3 coverage components"
                )
        for feature in numeric_feature_names:
            value = row.get(feature)
            if value is not None and not math.isfinite(float(value)):
                errors.append(
                    f"{row['task']}/{row['sample_id']}: nonfinite {feature}"
                )
        if int(row.get("pruned_candidate_count", 0)) > 0:
            if not row.get("all_pruned_candidates"):
                errors.append(f"{row['task']}/{row['sample_id']}: empty candidate audit")
            if not str(row.get("gate_reason", "")).strip():
                errors.append(f"{row['task']}/{row['sample_id']}: empty gate reason")
    expected = len(args.tasks) * args.max_samples_per_task
    if len(rows) != expected:
        errors.append(f"audit rows={len(rows)} expected={expected}")

    features = [
        "rescue_coverage_gap",
        "coverage_gap_v2",
        "coverage_gap_v3",
        "context_entity_novelty",
        "title_or_heading_novelty",
        "rescued_novel_sentence_ratio",
        "bridge_entity_novelty_ratio",
        "bridge_entity_count",
        "rescued_active_semantic_novelty_mean",
        "rescued_active_semantic_novelty_max",
        "rescued_active_semantic_novelty_min",
        "deletion_evidence_risk_v3",
        "coverage_gap_v3_raw",
        "coverage_gap_v3_1",
        "rescued_vs_removed_entity_gain",
        "rescued_vs_active_entity_gain",
        "sentence_novelty_gain",
        "semantic_novelty_rank",
        "semantic_novelty_rank_mean",
        "semantic_novelty_rank_min",
        "bridge_entity_gain",
        "answer_type_support_gain",
        "rescued_answer_type_support",
        "removed_answer_type_support",
        "deletion_query_overlap_risk",
        "deletion_entity_risk",
        "deletion_date_number_risk",
        "deletion_title_risk",
        "deletion_answer_type_risk",
        "deletion_base_importance_risk",
        "deletion_protected_proximity_risk",
        "deletion_risk_v3_raw",
        "deletion_risk_v3_1",
        "rescue_selected_min_percentile",
        "rescue_selected_mean_percentile",
        "rescue_unselected_max_percentile",
        "rescue_boundary_margin_rank",
        "candidate_pool_size",
        "candidate_pool_effective_size",
        "candidate_score_std",
        "candidate_score_iqr",
        "candidate_score_entropy_proxy",
        "candidate_score_flat",
        "rescue_boundary_margin",
        "deletion_risk",
        "query_overlap_rescued",
        "query_overlap_removed",
        "entity_overlap_rescued",
        "entity_overlap_removed",
        "date_number_overlap_rescued",
        "date_number_overlap_removed",
    ]
    groups: Dict[str, Any] = {}
    for scope in ["overall", *args.tasks]:
        selected = list(rows) if scope == "overall" else [
            row for row in rows if row["task"] == scope
        ]
        groups[scope] = {
            "samples": len(selected),
            "would_gate_trigger": sum(
                bool(row.get("would_gate_trigger")) for row in selected
            ),
            "gate_reason_counts": dict(sorted(
                (
                    reason,
                    sum(row.get("gate_reason") == reason for row in selected),
                )
                for reason in sorted({str(row.get("gate_reason")) for row in selected})
            )),
            "features": {
                feature: gate_feature_distribution(
                    [row.get(feature) for row in selected]
                )
                for feature in features
            },
        }
    trigger_groups: Dict[str, Any] = {}
    for triggered in [False, True]:
        selected = [
            row for row in rows if bool(row.get("would_gate_trigger")) == triggered
        ]
        trigger_groups[str(triggered).lower()] = {
            "samples": len(selected),
            "feature_means": {
                feature: gate_feature_distribution(
                    [row.get(feature) for row in selected]
                )["mean"]
                for feature in features
            },
        }

    top_cases = {}
    for feature in [
        "coverage_gap_v2",
        "coverage_gap_v3",
        "coverage_gap_v3_1",
        "rescued_vs_removed_entity_gain",
        "sentence_novelty_gain",
        "semantic_novelty_rank",
        "bridge_entity_gain",
        "answer_type_support_gain",
        "deletion_risk_v3_1",
        "rescue_boundary_margin_rank",
        "context_entity_novelty",
        "rescued_novel_sentence_ratio",
        "bridge_entity_novelty_ratio",
        "rescued_active_semantic_novelty_mean",
        "deletion_evidence_risk_v3",
        "deletion_risk_v3_1",
        "rescue_boundary_margin_rank",
        "candidate_score_std",
        "candidate_score_iqr",
        "candidate_score_entropy_proxy",
        "rescue_boundary_margin",
    ]:
        available = [row for row in rows if row.get(feature) is not None]
        top_cases[feature] = [
            {
                "sample_id": row["sample_id"],
                "task": row["task"],
                "gate_reason": row["gate_reason"],
                "candidate_rescued_block_ids": row[
                    "candidate_rescued_block_ids"
                ],
                "candidate_removed_block_ids": row[
                    "candidate_removed_block_ids"
                ],
                "rescue_coverage_gap": row.get("rescue_coverage_gap"),
                "coverage_gap_v2": row.get("coverage_gap_v2"),
                "coverage_gap_v3": row.get("coverage_gap_v3"),
                "coverage_gap_v3_1": row.get("coverage_gap_v3_1"),
                "rescued_vs_removed_entity_gain": row.get(
                    "rescued_vs_removed_entity_gain"
                ),
                "sentence_novelty_gain": row.get(
                    "sentence_novelty_gain"
                ),
                "semantic_novelty_rank": row.get(
                    "semantic_novelty_rank"
                ),
                "bridge_entity_gain": row.get("bridge_entity_gain"),
                "answer_type_support_gain": row.get(
                    "answer_type_support_gain"
                ),
                "deletion_risk_v3_1": row.get(
                    "deletion_risk_v3_1"
                ),
                "rescue_boundary_margin_rank": row.get(
                    "rescue_boundary_margin_rank"
                ),
                "context_entity_novelty": row.get(
                    "context_entity_novelty"
                ),
                "rescued_novel_sentence_ratio": row.get(
                    "rescued_novel_sentence_ratio"
                ),
                "bridge_entity_novelty_ratio": row.get(
                    "bridge_entity_novelty_ratio"
                ),
                "rescued_active_semantic_novelty_mean": row.get(
                    "rescued_active_semantic_novelty_mean"
                ),
                "rescue_boundary_margin": row.get("rescue_boundary_margin"),
                "deletion_risk": row.get("deletion_risk"),
                "deletion_evidence_risk_v3": row.get(
                    "deletion_evidence_risk_v3"
                ),
            }
            for row in sorted(
                available,
                key=lambda item: float(item.get(feature) or 0.0),
                reverse=True,
            )[:10]
        ]
    summary = {
        "mode": "gate_feature_audit_only",
        "quality_generation_performed": False,
        "scorer_invoked": False,
        "total_samples": len(rows),
        "samples_per_task": {
            task: sum(row["task"] == task for row in rows) for task in args.tasks
        },
        "gate_feature_missing": sum(
            bool(row.get("gate_feature_missing")) for row in rows
        ),
        "gate_feature_missing_reasons": dict(sorted(missing_reasons.items())),
        "would_gate_trigger": sum(
            bool(row.get("would_gate_trigger")) for row in rows
        ),
        "would_gate_trigger_rate": round(
            sum(bool(row.get("would_gate_trigger")) for row in rows)
            / max(1, len(rows)),
            8,
        ),
        "candidate_pool_exhausted": sum(
            bool(row.get("candidate_pool_exhausted")) for row in rows
        ),
        "candidate_score_flat": sum(
            bool(row.get("candidate_score_flat")) for row in rows
        ),
        "would_gate_trigger_v3_1": sum(
            bool(row.get("would_gate_trigger_v3_1")) for row in rows
        ),
        "title_or_heading_available": sum(
            bool(row.get("title_or_heading_available")) for row in rows
        ),
        "groups": groups,
        "would_trigger_groups": trigger_groups,
        "top_cases": top_cases,
        "total_prefill_elapsed_s": round(
            sum(float(row.get("prefill_elapsed_s", 0.0)) for row in rows), 4
        ),
    }
    validation = {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors[:50],
        "audit_rows": len(rows),
        "expected_rows": expected,
        "quality_generation_performed": False,
        "scorer_invoked": False,
    }
    return validation, summary


def retrospective_gate_association(
    args: argparse.Namespace,
    rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Join precomputed historical scores only after gate decisions are frozen."""
    result_dir = Path(args.retrospective_result_dir).resolve()
    audit_path = result_dir / "audit" / "per_sample_audit.jsonl"
    if not audit_path.exists():
        return {
            "status": "unavailable",
            "reason": f"missing historical audit: {audit_path}",
            "used_as_gate_input": False,
        }

    historical_scores: Dict[Tuple[str, str, str], float] = {}
    for row in read_jsonl(audit_path):
        method = str(row.get("method_key", ""))
        task = str(row.get("task", ""))
        sample_id = str(row.get("sample_id", ""))
        score = row.get("prediction_score")
        if (
            method
            and task
            and sample_id
            and score is not None
            and math.isfinite(float(score))
        ):
            historical_scores[(method, task, sample_id)] = float(score)

    comparisons = {
        "drm_minus_original": (
            "budget_neutral_drm_swap",
            "sliminfer_original",
        ),
        "drm_minus_random_swap": (
            "budget_neutral_drm_swap",
            "random_budget_neutral_swap",
        ),
    }
    feature_names = [
        "coverage_gap_v2",
        "coverage_gap_v3",
        "coverage_gap_v3_1",
        "rescued_vs_removed_entity_gain",
        "rescued_vs_active_entity_gain",
        "sentence_novelty_gain",
        "semantic_novelty_rank",
        "semantic_novelty_rank_mean",
        "semantic_novelty_rank_min",
        "bridge_entity_gain",
        "answer_type_support_gain",
        "context_entity_novelty",
        "title_or_heading_novelty",
        "rescued_novel_sentence_ratio",
        "bridge_entity_novelty_ratio",
        "rescued_active_semantic_novelty_mean",
        "deletion_evidence_risk_v3",
        "deletion_risk_v3_1",
        "rescue_boundary_margin_rank",
        "candidate_score_std",
        "candidate_score_iqr",
        "candidate_score_entropy_proxy",
        "rescue_boundary_margin",
    ]
    output: Dict[str, Any] = {}
    for comparison, (left_method, right_method) in comparisons.items():
        matched: List[Dict[str, Any]] = []
        for gate_row in rows:
            task = str(gate_row["task"])
            sample_id = str(gate_row["sample_id"])
            left = historical_scores.get((left_method, task, sample_id))
            right = historical_scores.get((right_method, task, sample_id))
            if left is None or right is None:
                continue
            delta = float(left - right)
            label = "positive" if delta > 0.0 else (
                "negative" if delta < 0.0 else "zero"
            )
            matched.append({
                **dict(gate_row),
                "historical_delta": delta,
                "historical_label": label,
            })
        scopes: Dict[str, Any] = {}
        for scope in ["overall", *args.tasks]:
            scoped = matched if scope == "overall" else [
                row for row in matched if row["task"] == scope
            ]
            labels: Dict[str, Any] = {}
            for label in ["positive", "negative", "zero"]:
                selected = [
                    row for row in scoped if row["historical_label"] == label
                ]
                labels[label] = {
                    "samples": len(selected),
                    "mean_historical_delta": (
                        round(
                            sum(float(row["historical_delta"]) for row in selected)
                            / len(selected),
                            8,
                        )
                        if selected
                        else None
                    ),
                    "would_gate_trigger": sum(
                        bool(row.get("would_gate_trigger")) for row in selected
                    ),
                    "would_gate_trigger_rate": (
                        round(
                            sum(
                                bool(row.get("would_gate_trigger"))
                                for row in selected
                            )
                            / len(selected),
                            8,
                        )
                        if selected
                        else None
                    ),
                    "features": {
                        feature: gate_feature_distribution(
                            [row.get(feature) for row in selected]
                        )
                        for feature in feature_names
                    },
                }
            scopes[scope] = labels
        output[comparison] = {
            "matched_samples": len(matched),
            "groups": scopes,
        }
    return {
        "status": "ok",
        "note": (
            "Retrospective association only. Historical gold-scored deltas "
            "are post-hoc labels and were never gate inputs."
        ),
        "used_as_gate_input": False,
        "historical_result_dir": str(result_dir),
        "comparisons": output,
    }


def select_rescue_records(cell_dir: Path, target_layer: int) -> Dict[str, Dict[str, Any]]:
    path = cell_dir / "audit" / "rescue_policy_audit.jsonl"
    rows = read_jsonl(path)
    selected: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if int(row.get("layer_id", -1)) != target_layer:
            continue
        phase = str(row.get("phase", ""))
        if phase and phase != "prefill":
            continue
        sid = str(row.get("sample_id", ""))
        if sid and sid not in selected:
            selected[sid] = row
    return selected


def generation_records(cell_dir: Path) -> Dict[str, Dict[str, Any]]:
    return {
        str(row.get("sample_id", "")): row
        for row in read_jsonl(cell_dir / "audit" / "generations.jsonl")
        if row.get("sample_id") is not None
    }


def score_prediction(task: str, prediction: str, answers: Sequence[Any], all_classes: Any) -> float:
    from longbench_eval import dataset2metric

    pred = str(prediction)
    if task in {"trec", "triviaqa", "samsum", "lsht"}:
        pred = pred.lstrip("\n").split("\n")[0]
    score = 0.0
    for answer in answers:
        score = max(score, float(dataset2metric[task](pred, answer, all_classes=all_classes)))
    return 100.0 * score


def scorer_accepts_task(task: str, predictions: Sequence[Mapping[str, Any]]) -> float:
    from longbench_eval import scorer

    score = scorer(
        task,
        [str(row.get("pred", "")) for row in predictions],
        [row.get("answers", []) for row in predictions],
        [int(row.get("length", 0) or 0) for row in predictions],
        next((row.get("all_classes") for row in predictions if row.get("all_classes")), []),
    )
    if score is None or not math.isfinite(float(score)):
        raise ClosureError(f"longbench_eval scorer failed for {task}")
    return float(score)


def consolidate_audit(args: argparse.Namespace) -> List[Dict[str, Any]]:
    protocol = shared_protocol(args)
    rows: List[Dict[str, Any]] = []
    target_layer = args.prune_layers_list[-1]
    used_block_layer = target_layer + 1
    final_retain = args.retain_tokens_list[-1]
    for method in args.methods:
        spec = METHOD_SPECS[method]
        for task in args.tasks:
            cell_dir = args.output_path / "runs" / method / task
            predictions = read_jsonl(cell_dir / "longbench_predictions" / f"{task}.jsonl")
            scorer_accepts_task(task, predictions)
            availability = profile_availability(cell_dir)
            used = selected_prefill_records(cell_dir, used_block_layer)
            layer_schedules = per_layer_prefill_schedule(cell_dir, args.prune_layers_list)
            rescue = select_rescue_records(cell_dir, target_layer)
            generations = generation_records(cell_dir)
            for prediction in predictions:
                sid = str(prediction.get("sample_id", ""))
                usage = used.get(sid)
                generation = generations.get(sid)
                if usage is None or generation is None:
                    raise ClosureError(f"{method}/{task}/{sid}: missing usage or generation audit")
                used_blocks = usage.get("used_block_ids")
                used_tokens = usage.get("used_tokens")
                if not isinstance(used_blocks, list) or not isinstance(used_tokens, list):
                    raise ClosureError(f"{method}/{task}/{sid}: missing full used block/token lists")
                layer_schedule = layer_schedules.get(sid)
                if layer_schedule is None:
                    raise ClosureError(f"{method}/{task}/{sid}: incomplete per-layer block schedule")
                schedule = budget_schedule(
                    int(usage.get("prompt_token_count", prediction.get("length", 0))),
                    args.block_size,
                    final_retain,
                    args.rescue_ratio,
                    spec["target_budget_type"],
                )
                rescue_row = rescue.get(sid, {})
                is_swap = method in BUDGET_NEUTRAL_SWAP_METHODS
                is_gated = method in GATED_SWAP_METHODS
                rescue_count = (
                    int(rescue_row.get("rescued_count", 0) or 0)
                    if spec["rescue_policy"] in {
                        "add",
                        "budget_neutral_drm_swap",
                        "gated_budget_neutral_drm_swap",
                    }
                    else 0
                )
                fallback_reason = str(rescue_row.get("fallback_reason", "") or "")
                fallback_used = bool(rescue_row.get("fallback_used", False))
                if is_swap and not rescue_row:
                    fallback_used = True
                    fallback_reason = "no_final_pruning_candidates"
                removed_block_ids = (
                    [int(block_id) for block_id in rescue_row.get("removed_block_ids", [])]
                    if is_swap
                    else []
                )
                rescued_block_ids = (
                    [int(block_id) for block_id in rescue_row.get("rescued_block_ids", [])]
                    if is_swap
                    else []
                )
                swap_count = (
                    int(rescue_row.get("swap_count", len(removed_block_ids)) or 0)
                    if is_swap
                    else 0
                )
                candidate_block_count = int(
                    rescue_row.get("block_count_total")
                    or usage.get("prompt_block_count")
                    or 0
                )
                selected_blocks = int(usage.get("final_block_count_used_by_attention", -1))
                selected_tokens = int(usage.get("kv_abs_indices_count", -1))
                pred_text = str(prediction.get("pred", ""))
                score = score_prediction(
                    task,
                    pred_text,
                    prediction.get("answers", []),
                    prediction.get("all_classes", []),
                )
                row = {
                    "method_key": method,
                    "method_name": spec["method_name"],
                    "task": task,
                    "sample_id": sid,
                    "sample_index": int(prediction.get("sample_index", -1)),
                    "decode_policy": args.decode_policy,
                    "base_scoring": args.base_scoring,
                    "rescue_score": spec["rescue_score"],
                    "seed": int(rescue_row.get("seed", args.seed) or 0),
                    "target_budget_type": spec["target_budget_type"],
                    "K": schedule["K"],
                    "M": schedule["M"],
                    "target_K_plus_M": schedule["target_K_plus_M"],
                    "selected_token_count": selected_tokens,
                    "selected_block_count": selected_blocks,
                    "active_set_size_before": int(
                        rescue_row.get("active_before_count")
                        or schedule["K"]
                    ),
                    "active_set_size_after": selected_blocks,
                    "pruned_candidate_count": max(0, candidate_block_count - schedule["K"]),
                    "rescue_count": rescue_count,
                    "added_count": rescue_count,
                    "swap_count": swap_count,
                    "removed_count": (
                        int(rescue_row.get("removed_count", len(removed_block_ids)) or 0)
                        if is_swap
                        else 0
                    ),
                    "final_budget_type": spec["final_budget_type"],
                    "budget_equal_to_target": selected_blocks == schedule["target_K_plus_M"],
                    "budget_equal_to_sliminfer": selected_blocks == schedule["K"],
                    "over_budget": selected_blocks > schedule["target_K_plus_M"],
                    "used_blocks": used_blocks,
                    "used_tokens": used_tokens,
                    "per_layer_block_schedule": layer_schedule,
                    "removed_block_ids": removed_block_ids,
                    "rescued_block_ids": rescued_block_ids,
                    "protected_block_ids": (
                        [int(block_id) for block_id in rescue_row.get("protected_block_ids", [])]
                        if is_swap
                        else []
                    ),
                    "query_critical_block_ids": (
                        [int(block_id) for block_id in rescue_row.get("query_critical_block_ids", [])]
                        if is_swap
                        else []
                    ),
                    "forced_keep_block_ids": (
                        [int(block_id) for block_id in rescue_row.get("forced_keep_block_ids", [])]
                        if is_swap
                        else []
                    ),
                    "protected_block_violation": bool(
                        rescue_row.get("protected_block_violation", False)
                    ),
                    "rescue_score_mean": (
                        float(rescue_row.get("rescue_score_mean", 0.0) or 0.0)
                        if is_swap
                        else 0.0
                    ),
                    "removed_score_mean": (
                        float(rescue_row.get("removed_score_mean", 0.0) or 0.0)
                        if is_swap
                        else 0.0
                    ),
                    "active_redundancy_mean": (
                        float(rescue_row.get("active_redundancy_mean", 0.0) or 0.0)
                        if is_swap
                        else 0.0
                    ),
                    "removed_value_norm_mean": (
                        rescue_row.get("removed_value_norm_mean") if is_swap else None
                    ),
                    "removed_value_norm_max": (
                        rescue_row.get("removed_value_norm_max") if is_swap else None
                    ),
                    "rescued_value_norm_mean": (
                        rescue_row.get("rescued_value_norm_mean") if is_swap else None
                    ),
                    "rescued_value_norm_max": (
                        rescue_row.get("rescued_value_norm_max") if is_swap else None
                    ),
                    "removed_qk_or_delta_score_mean": (
                        rescue_row.get("removed_qk_or_delta_score_mean") if is_swap else None
                    ),
                    "removed_qk_or_delta_score_max": (
                        rescue_row.get("removed_qk_or_delta_score_max") if is_swap else None
                    ),
                    "rescued_qk_or_delta_score_mean": (
                        rescue_row.get("rescued_qk_or_delta_score_mean") if is_swap else None
                    ),
                    "rescued_qk_or_delta_score_max": (
                        rescue_row.get("rescued_qk_or_delta_score_max") if is_swap else None
                    ),
                    "delete_risk_proxy_mean": (
                        rescue_row.get("delete_risk_proxy_mean") if is_swap else None
                    ),
                    "delete_risk_proxy_max": (
                        rescue_row.get("delete_risk_proxy_max") if is_swap else None
                    ),
                    "rescue_gain_proxy_mean": (
                        rescue_row.get("rescue_gain_proxy_mean") if is_swap else None
                    ),
                    "rescue_gain_proxy_max": (
                        rescue_row.get("rescue_gain_proxy_max") if is_swap else None
                    ),
                    "residual_benefit_proxy": (
                        rescue_row.get("residual_benefit_proxy") if is_swap else None
                    ),
                    "rescue_score_margin": (
                        rescue_row.get("rescue_score_margin") if is_swap else None
                    ),
                    "removed_block_text_preview": (
                        list(rescue_row.get("removed_block_text_preview", []))
                        if is_swap
                        else []
                    ),
                    "rescued_block_text_preview": (
                        list(rescue_row.get("rescued_block_text_preview", []))
                        if is_swap
                        else []
                    ),
                    "residual_proxy_unavailable_fields": (
                        list(rescue_row.get("residual_proxy_unavailable_fields", []))
                        if is_swap
                        else []
                    ),
                    "residual_proxy_version": (
                        str(rescue_row.get("residual_proxy_version", "unavailable"))
                        if is_swap
                        else "not_applicable"
                    ),
                    "gate_mode": (
                        str(rescue_row.get("gate_mode", args.gate_mode))
                        if is_gated
                        else "not_applicable"
                    ),
                    "gate_policy": (
                        str(rescue_row.get("gate_policy", args.gate_policy))
                        if is_gated
                        else "not_applicable"
                    ),
                    "gate_triggered": bool(
                        rescue_row.get("gate_triggered", False)
                    ) if is_swap else False,
                    "would_gate_trigger": bool(
                        rescue_row.get("would_gate_trigger", False)
                    ) if is_gated else False,
                    "gate_score": (
                        rescue_row.get("gate_score") if is_gated else None
                    ),
                    "gate_reason": (
                        str(
                            rescue_row.get(
                                "gate_reason",
                                fallback_reason or "not_applicable_audit_only",
                            )
                        )
                        if is_swap
                        else "not_applicable"
                    ),
                    "gate_default_policy": (
                        str(
                            rescue_row.get(
                                "gate_default_policy", args.gate_default_policy
                            )
                        )
                        if is_gated
                        else "not_applicable"
                    ),
                    "gate_fallback_to_original": bool(
                        rescue_row.get(
                            "gate_fallback_to_original", not bool(rescue_row)
                        )
                    ) if is_gated else False,
                    "selected_policy": (
                        str(rescue_row.get("selected_policy", "original"))
                        if is_gated
                        else "not_applicable"
                    ),
                    "selected_policy_before_gate": (
                        str(
                            rescue_row.get(
                                "selected_policy_before_gate", "original"
                            )
                        )
                        if is_gated
                        else "not_applicable"
                    ),
                    "selected_policy_after_gate": (
                        str(
                            rescue_row.get(
                                "selected_policy_after_gate", "original"
                            )
                        )
                        if is_gated
                        else "not_applicable"
                    ),
                    "gate_feature_missing": bool(
                        rescue_row.get("gate_feature_missing", not bool(rescue_row))
                    ) if is_gated else False,
                    "gate_feature_missing_fields": (
                        list(
                            rescue_row.get(
                                "gate_feature_missing_fields",
                                ["final_pruning_audit"] if not rescue_row else [],
                            )
                        )
                        if is_gated
                        else []
                    ),
                    "gate_feature_version": (
                        str(
                            rescue_row.get(
                                "gate_feature_version", args.gate_feature_version
                            )
                        )
                        if is_gated
                        else "not_applicable"
                    ),
                    "rescue_coverage_gap": (
                        rescue_row.get("rescue_coverage_gap")
                        if is_gated
                        else None
                    ),
                    "coverage_gap_v2": (
                        rescue_row.get(
                            "coverage_gap_v2",
                            rescue_row.get("rescue_coverage_gap"),
                        )
                        if is_gated
                        else None
                    ),
                    "coverage_gap_v3": (
                        rescue_row.get("coverage_gap_v3")
                        if is_gated
                        else None
                    ),
                    "context_entity_novelty": (
                        rescue_row.get("context_entity_novelty")
                        if is_gated
                        else None
                    ),
                    "title_or_heading_novelty": (
                        rescue_row.get("title_or_heading_novelty")
                        if is_gated
                        else None
                    ),
                    "title_or_heading_available": bool(
                        rescue_row.get("title_or_heading_available", False)
                    ) if is_gated else False,
                    "rescued_novel_sentence_ratio": (
                        rescue_row.get("rescued_novel_sentence_ratio")
                        if is_gated
                        else None
                    ),
                    "bridge_entity_novelty_ratio": (
                        rescue_row.get("bridge_entity_novelty_ratio")
                        if is_gated
                        else None
                    ),
                    "bridge_entity_count": (
                        int(rescue_row.get("bridge_entity_count", 0) or 0)
                        if is_gated
                        else 0
                    ),
                    "rescued_active_semantic_novelty_mean": (
                        rescue_row.get(
                            "rescued_active_semantic_novelty_mean"
                        )
                        if is_gated
                        else None
                    ),
                    "rescued_active_semantic_novelty_max": (
                        rescue_row.get(
                            "rescued_active_semantic_novelty_max"
                        )
                        if is_gated
                        else None
                    ),
                    "rescued_active_semantic_novelty_min": (
                        rescue_row.get(
                            "rescued_active_semantic_novelty_min"
                        )
                        if is_gated
                        else None
                    ),
                    "deletion_evidence_risk_v3": (
                        rescue_row.get("deletion_evidence_risk_v3")
                        if is_gated
                        else None
                    ),
                    "candidate_pool_exhausted": bool(
                        rescue_row.get("candidate_pool_exhausted", False)
                    ) if is_gated else False,
                    "question_entity_units": (
                        list(rescue_row.get("question_entity_units", []))
                        if is_gated
                        else []
                    ),
                    "rescued_entity_units": (
                        list(rescue_row.get("rescued_entity_units", []))
                        if is_gated
                        else []
                    ),
                    "active_entity_units": (
                        list(rescue_row.get("active_entity_units", []))
                        if is_gated
                        else []
                    ),
                    "rescued_only_entity_units": (
                        list(rescue_row.get("rescued_only_entity_units", []))
                        if is_gated
                        else []
                    ),
                    "removed_entity_units": (
                        list(rescue_row.get("removed_entity_units", []))
                        if is_gated
                        else []
                    ),
                    "removed_title_like_units": (
                        list(
                            rescue_row.get("removed_title_like_units", [])
                        )
                        if is_gated
                        else []
                    ),
                    "removed_date_number_units": (
                        list(
                            rescue_row.get("removed_date_number_units", [])
                        )
                        if is_gated
                        else []
                    ),
                    "answer_type_proxy": (
                        str(rescue_row.get("answer_type_proxy", "other"))
                        if is_gated
                        else "not_applicable"
                    ),
                    "coverage_gap_v3_raw": (
                        rescue_row.get(
                            "coverage_gap_v3_raw",
                            rescue_row.get("coverage_gap_v3"),
                        )
                        if is_gated
                        else None
                    ),
                    "coverage_gap_v3_1": (
                        rescue_row.get("coverage_gap_v3_1")
                        if is_gated
                        else None
                    ),
                    "rescued_vs_removed_entity_gain": (
                        rescue_row.get("rescued_vs_removed_entity_gain")
                        if is_gated
                        else None
                    ),
                    "rescued_vs_active_entity_gain": (
                        rescue_row.get("rescued_vs_active_entity_gain")
                        if is_gated
                        else None
                    ),
                    "sentence_novelty_gain": (
                        rescue_row.get("sentence_novelty_gain")
                        if is_gated
                        else None
                    ),
                    "semantic_novelty_rank": (
                        rescue_row.get("semantic_novelty_rank")
                        if is_gated
                        else None
                    ),
                    "semantic_novelty_rank_mean": (
                        rescue_row.get("semantic_novelty_rank_mean")
                        if is_gated
                        else None
                    ),
                    "semantic_novelty_rank_min": (
                        rescue_row.get("semantic_novelty_rank_min")
                        if is_gated
                        else None
                    ),
                    "bridge_entity_gain": (
                        rescue_row.get("bridge_entity_gain")
                        if is_gated
                        else None
                    ),
                    "answer_type_support_gain": (
                        rescue_row.get("answer_type_support_gain")
                        if is_gated
                        else None
                    ),
                    "rescued_answer_type_support": (
                        rescue_row.get("rescued_answer_type_support")
                        if is_gated
                        else None
                    ),
                    "removed_answer_type_support": (
                        rescue_row.get("removed_answer_type_support")
                        if is_gated
                        else None
                    ),
                    "deletion_query_overlap_risk": (
                        rescue_row.get("deletion_query_overlap_risk")
                        if is_gated
                        else None
                    ),
                    "deletion_entity_risk": (
                        rescue_row.get("deletion_entity_risk")
                        if is_gated
                        else None
                    ),
                    "deletion_date_number_risk": (
                        rescue_row.get("deletion_date_number_risk")
                        if is_gated
                        else None
                    ),
                    "deletion_title_risk": (
                        rescue_row.get("deletion_title_risk")
                        if is_gated
                        else None
                    ),
                    "deletion_answer_type_risk": (
                        rescue_row.get("deletion_answer_type_risk")
                        if is_gated
                        else None
                    ),
                    "deletion_base_importance_risk": (
                        rescue_row.get("deletion_base_importance_risk")
                        if is_gated
                        else None
                    ),
                    "deletion_protected_proximity_risk": (
                        rescue_row.get(
                            "deletion_protected_proximity_risk"
                        )
                        if is_gated
                        else None
                    ),
                    "deletion_risk_v3_raw": (
                        rescue_row.get(
                            "deletion_risk_v3_raw",
                            rescue_row.get("deletion_evidence_risk_v3"),
                        )
                        if is_gated
                        else None
                    ),
                    "deletion_risk_v3_1": (
                        rescue_row.get("deletion_risk_v3_1")
                        if is_gated
                        else None
                    ),
                    "removed_block_risk_sources": (
                        list(
                            rescue_row.get(
                                "removed_block_risk_sources", []
                            )
                        )
                        if is_gated
                        else []
                    ),
                    "rescue_selected_min_percentile": (
                        rescue_row.get("rescue_selected_min_percentile")
                        if is_gated
                        else None
                    ),
                    "rescue_selected_mean_percentile": (
                        rescue_row.get("rescue_selected_mean_percentile")
                        if is_gated
                        else None
                    ),
                    "rescue_unselected_max_percentile": (
                        rescue_row.get("rescue_unselected_max_percentile")
                        if is_gated
                        else None
                    ),
                    "rescue_boundary_margin_rank": (
                        rescue_row.get("rescue_boundary_margin_rank")
                        if is_gated
                        else None
                    ),
                    "candidate_pool_size": (
                        int(rescue_row.get("candidate_pool_size", 0) or 0)
                        if is_gated
                        else 0
                    ),
                    "candidate_pool_effective_size": (
                        int(
                            rescue_row.get(
                                "candidate_pool_effective_size", 0
                            )
                            or 0
                        )
                        if is_gated
                        else 0
                    ),
                    "candidate_score_std": (
                        rescue_row.get("candidate_score_std")
                        if is_swap
                        else None
                    ),
                    "candidate_score_iqr": (
                        rescue_row.get("candidate_score_iqr")
                        if is_swap
                        else None
                    ),
                    "candidate_score_entropy_proxy": (
                        rescue_row.get("candidate_score_entropy_proxy")
                        if is_gated
                        else None
                    ),
                    "candidate_score_flat": bool(
                        rescue_row.get("candidate_score_flat", False)
                    ) if is_gated else False,
                    "would_gate_trigger_v3_1": bool(
                        rescue_row.get(
                            "would_gate_trigger_v3_1", False
                        )
                    ) if is_gated else False,
                    "gate_reason_v3_1": (
                        str(
                            rescue_row.get(
                                "gate_reason_v3_1", "not_applicable"
                            )
                        )
                        if is_gated
                        else "not_applicable"
                    ),
                    "coverage_components_available": (
                        list(
                            rescue_row.get(
                                "coverage_components_available", []
                            )
                        )
                        if is_gated
                        else []
                    ),
                    "coverage_components_missing": (
                        list(
                            rescue_row.get(
                                "coverage_components_missing", []
                            )
                        )
                        if is_gated
                        else []
                    ),
                    "deletion_risk": (
                        rescue_row.get("deletion_risk") if is_gated else None
                    ),
                    "rescue_boundary_margin": (
                        rescue_row.get("rescue_boundary_margin")
                        if is_gated
                        else None
                    ),
                    "query_overlap_rescued": (
                        rescue_row.get("query_overlap_rescued")
                        if is_gated
                        else None
                    ),
                    "query_overlap_removed": (
                        rescue_row.get("query_overlap_removed")
                        if is_gated
                        else None
                    ),
                    "entity_overlap_rescued": (
                        rescue_row.get("entity_overlap_rescued")
                        if is_gated
                        else None
                    ),
                    "entity_overlap_removed": (
                        rescue_row.get("entity_overlap_removed")
                        if is_gated
                        else None
                    ),
                    "date_number_overlap_rescued": (
                        rescue_row.get("date_number_overlap_rescued")
                        if is_gated
                        else None
                    ),
                    "date_number_overlap_removed": (
                        rescue_row.get("date_number_overlap_removed")
                        if is_gated
                        else None
                    ),
                    "rescue_score_top1": (
                        rescue_row.get("rescue_score_top1")
                        if is_swap
                        else None
                    ),
                    "rescue_score_top2": (
                        rescue_row.get("rescue_score_top2")
                        if is_swap
                        else None
                    ),
                    "rescue_score_min_selected": (
                        rescue_row.get("rescue_score_min_selected")
                        if is_gated
                        else None
                    ),
                    "rescue_score_boundary_selected": (
                        rescue_row.get("rescue_score_boundary_selected")
                        if is_gated
                        else None
                    ),
                    "rescue_score_boundary_unselected": (
                        rescue_row.get("rescue_score_boundary_unselected")
                        if is_gated
                        else None
                    ),
                    "removed_score_min": (
                        rescue_row.get("removed_score_min")
                        if is_gated
                        else None
                    ),
                    "removed_score_max": (
                        rescue_row.get("removed_score_max")
                        if is_gated
                        else None
                    ),
                    "removed_drop_utility_mean": (
                        rescue_row.get("removed_drop_utility_mean")
                        if is_gated
                        else None
                    ),
                    "removed_drop_utility_max": (
                        rescue_row.get("removed_drop_utility_max")
                        if is_gated
                        else None
                    ),
                    "rescued_vs_removed_score_gap": (
                        rescue_row.get("rescued_vs_removed_score_gap")
                        if is_gated
                        else None
                    ),
                    "active_redundancy_max": (
                        rescue_row.get("active_redundancy_max")
                        if is_gated
                        else None
                    ),
                    "protected_reasons": (
                        rescue_row.get("protected_reasons", {})
                        if is_swap
                        else {}
                    ),
                    "candidate_swap_count": (
                        int(rescue_row.get("candidate_swap_count", 0) or 0)
                        if is_gated
                        else 0
                    ),
                    "candidate_rescued_block_ids": (
                        [
                            int(block_id)
                            for block_id in rescue_row.get(
                                "candidate_rescued_block_ids", []
                            )
                        ]
                        if is_swap
                        else []
                    ),
                    "candidate_removed_block_ids": (
                        [
                            int(block_id)
                            for block_id in rescue_row.get(
                                "candidate_removed_block_ids", []
                            )
                        ]
                        if is_swap
                        else []
                    ),
                    "all_pruned_candidates": (
                        rescue_row.get("all_pruned_candidates", [])
                        if is_gated
                        else []
                    ),
                    "selected_rescued_candidates": (
                        rescue_row.get("selected_rescued_candidates", [])
                        if is_gated
                        else []
                    ),
                    "unselected_pruned_candidates": (
                        rescue_row.get("unselected_pruned_candidates", [])
                        if is_gated
                        else []
                    ),
                    "all_active_drop_candidates": (
                        rescue_row.get("all_active_drop_candidates", [])
                        if is_gated
                        else []
                    ),
                    "selected_removed_candidates": (
                        rescue_row.get("selected_removed_candidates", [])
                        if is_gated
                        else []
                    ),
                    "protected_blocks_with_reasons": (
                        rescue_row.get("protected_blocks_with_reasons", [])
                        if is_gated
                        else []
                    ),
                    "fallback_reason": fallback_reason,
                    "prompt_hash": str(generation.get("prompt_hash", "")),
                    "prompt_wrapper_hash": protocol["prompt_wrapper_hash"],
                    "generation_config_hash": generation_config_hash_for_task(args, task),
                    "scorer_hash": protocol["scorer_hash"],
                    "model_name": str(Path(args.model_path)),
                    "model_revision_or_unknown": protocol["model_revision_or_unknown"],
                    "fallback_used": fallback_used,
                    **availability,
                    "prediction": pred_text,
                    "gold_answers": prediction.get("answers", []),
                    "prediction_score": score,
                    "prediction_file": str(
                        args.output_path / "predictions" / method / f"{task}.jsonl"
                    ),
                    "prompt_token_count": int(
                        usage.get("prompt_token_count", prediction.get("length", 0))
                    ),
                    "prompt_block_count": int(usage.get("prompt_block_count", 0)),
                    "selector_candidate_block_count": candidate_block_count,
                    "max_new_tokens": task_max_new_tokens(args, task),
                    "max_new_tokens_mode": (
                        "uniform_override" if int(args.max_new_tokens) > 0 else "dataset2maxlen_per_task"
                    ),
                    "scorer": "longbench_eval.scorer",
                    "used_block_audit_layer": used_block_layer,
                    "rescue_audit_layer": target_layer,
                }
                rows.append(row)
    return rows


def material_config_signature(config: Mapping[str, Any]) -> Dict[str, Any]:
    args = config.get("args", {})
    slim = config.get("sliminfer_config", {})
    return {
        "model_path": args.get("model_path"),
        "tokenizer_path": args.get("tokenizer_path"),
        "runner_method": args.get("method"),
        "decode_policy": args.get("decode_policy"),
        "scoring": args.get("scoring"),
        "fairness_base_scoring": args.get("fairness_base_scoring"),
        "temperature": args.get("temperature"),
        "top_p": args.get("top_p"),
        "generation_max_new_tokens": args.get("generation_max_new_tokens"),
        "max_model_len": args.get("max_model_len"),
        "prune_layers": args.get("prune_layers"),
        "retain_tokens": args.get("retain_tokens"),
        "block_size": args.get("block_size"),
        "token_unit_size": args.get("token_unit_size"),
        "local_query_window": args.get("local_query_window"),
        "swap_threshold": args.get("swap_threshold"),
        "rescue_policy": args.get("rescue_policy"),
        "rescue_ratio": args.get("rescue_ratio"),
        "seed": args.get("seed"),
        "fd_alpha_o": args.get("fd_alpha_o"),
        "fd_beta_h": args.get("fd_beta_h"),
        "fd_gamma_s": args.get("fd_gamma_s"),
        "alpha_diff": args.get("alpha_diff"),
        "lambda_red": args.get("lambda_red"),
        "eta_rescue": args.get("eta_rescue"),
        "eta_active": args.get("eta_active"),
        "candidate_pool_ratio": args.get("candidate_pool_ratio"),
        "sampling_temperature": args.get("sampling_temperature"),
        "stochastic_mode": args.get("stochastic_mode"),
        "gumbel_tau": args.get("gumbel_tau"),
        "gate_mode": args.get("gate_mode"),
        "gate_policy": args.get("gate_policy"),
        "gate_default_policy": args.get("gate_default_policy"),
        "gate_rescue_margin_min": args.get("gate_rescue_margin_min"),
        "gate_coverage_gap_min": args.get("gate_coverage_gap_min"),
        "gate_deletion_risk_max": args.get("gate_deletion_risk_max"),
        "gate_feature_version": args.get("gate_feature_version"),
        "score_kind": slim.get("score_kind"),
        "configured_fairness_base_scoring": slim.get("fairness_base_scoring"),
        "pruning_config": slim.get("pruning_config"),
    }


def is_nan(value: Any) -> bool:
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, list):
        return any(is_nan(item) for item in value)
    if isinstance(value, dict):
        return any(is_nan(item) for item in value.values())
    return False


def validate_smoke(args: argparse.Namespace, rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []

    def add_check(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    missing_fields: List[str] = []
    invalid_rows: List[str] = []
    for row in rows:
        missing = [field for field in REQUIRED_AUDIT_FIELDS if field not in row]
        if missing:
            missing_fields.append(f"{row.get('method_key')}/{row.get('task')}/{row.get('sample_id')}: {missing}")
        if is_nan(dict(row)):
            invalid_rows.append(f"{row.get('method_key')}/{row.get('task')}/{row.get('sample_id')}: NaN/Inf")
        if not str(row.get("prediction", "")).strip():
            invalid_rows.append(f"{row.get('method_key')}/{row.get('task')}/{row.get('sample_id')}: empty prediction")
        if int(row.get("selected_block_count", 0)) <= 0 or int(row.get("selected_token_count", 0)) <= 0:
            invalid_rows.append(f"{row.get('method_key')}/{row.get('task')}/{row.get('sample_id')}: empty selection")
        if int(row.get("selected_token_count", 0)) > int(row.get("prompt_token_count", 0)):
            invalid_rows.append(f"{row.get('method_key')}/{row.get('task')}/{row.get('sample_id')}: selection exceeds prompt")
        if len(row.get("used_blocks", [])) != int(row.get("selected_block_count", -1)):
            invalid_rows.append(f"{row.get('method_key')}/{row.get('task')}/{row.get('sample_id')}: used_blocks mismatch")
        if len(row.get("used_tokens", [])) != int(row.get("selected_token_count", -1)):
            invalid_rows.append(f"{row.get('method_key')}/{row.get('task')}/{row.get('sample_id')}: used_tokens mismatch")
        if int(row.get("K", -1)) + int(row.get("M", -1)) != int(row.get("target_K_plus_M", -1)):
            invalid_rows.append(f"{row.get('method_key')}/{row.get('task')}/{row.get('sample_id')}: K+M target mismatch")
        method = str(row.get("method_key", ""))
        expected_budget = METHOD_SPECS.get(method, {}).get("final_budget_type")
        if row.get("final_budget_type") != expected_budget:
            invalid_rows.append(f"{method}/{row.get('task')}/{row.get('sample_id')}: final budget type mismatch")
        if method in {"random_add", "delta_o_add", "drm_add"}:
            if int(row.get("rescue_count", -1)) != int(row.get("M", -1)):
                invalid_rows.append(f"{method}/{row.get('task')}/{row.get('sample_id')}: rescue_count != M")
            if int(row.get("added_count", -1)) != int(row.get("rescue_count", -1)):
                invalid_rows.append(f"{method}/{row.get('task')}/{row.get('sample_id')}: added_count mismatch")
        if method in BUDGET_NEUTRAL_SWAP_METHODS:
            if not bool(row.get("budget_equal_to_sliminfer")):
                invalid_rows.append(f"{method}/{row.get('task')}/{row.get('sample_id')}: budget != K")
            counts = {
                int(row.get("swap_count", -1)),
                int(row.get("rescue_count", -1)),
                int(row.get("removed_count", -1)),
                len(row.get("removed_block_ids", [])),
                len(row.get("rescued_block_ids", [])),
            }
            if len(counts) != 1:
                invalid_rows.append(f"{method}/{row.get('task')}/{row.get('sample_id')}: swap counts differ")
            if bool(row.get("protected_block_violation")) or (
                set(row.get("removed_block_ids", []))
                & set(row.get("protected_block_ids", []))
            ):
                invalid_rows.append(
                    f"{method}/{row.get('task')}/{row.get('sample_id')}: protected block removed"
                )
            if bool(row.get("fallback_used")):
                if counts != {0} or not str(row.get("fallback_reason", "")).strip():
                    invalid_rows.append(
                        f"{method}/{row.get('task')}/{row.get('sample_id')}: invalid fallback audit"
                    )
            elif method not in GATED_SWAP_METHODS and int(row.get("swap_count", 0)) <= 0:
                invalid_rows.append(
                    f"{method}/{row.get('task')}/{row.get('sample_id')}: no swap without fallback"
                )
            if int(row.get("seed", -1)) != int(args.seed):
                invalid_rows.append(
                    f"{method}/{row.get('task')}/{row.get('sample_id')}: seed mismatch"
                )
    add_check("required_audit_fields", not missing_fields, "; ".join(missing_fields[:10]) or "all present")
    add_check("no_nan_empty_prediction_or_selection", not invalid_rows, "; ".join(invalid_rows[:10]) or "clean")

    by_method_task: Dict[Tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_method_task[(str(row["method_key"]), str(row["task"]))].append(row)
    expected_row_count = len(args.methods) * len(args.tasks) * args.max_samples_per_task
    add_check(
        "audit_row_count",
        len(rows) == expected_row_count,
        f"actual={len(rows)} expected={expected_row_count}",
    )
    id_errors: List[str] = []
    for task in args.tasks:
        sequences = {
            method: [
                (int(row["sample_index"]), str(row["sample_id"]))
                for row in sorted(by_method_task[(method, task)], key=lambda item: int(item["sample_index"]))
            ]
            for method in args.methods
        }
        short = {
            method: len(sequence)
            for method, sequence in sequences.items()
            if len(sequence) != args.max_samples_per_task
        }
        if short:
            id_errors.append(f"{task}: sample counts={short}")
        if len({canonical_json(value) for value in sequences.values()}) != 1:
            id_errors.append(f"{task}: {sequences}")
    add_check("sample_id_alignment", not id_errors, "; ".join(id_errors) or "aligned")

    decode_values = {str(row["decode_policy"]) for row in rows}
    base_values = {str(row["base_scoring"]) for row in rows}
    add_check(
        "decode_and_base_scoring_closure",
        decode_values == {args.decode_policy} and base_values == {args.base_scoring},
        f"decode={sorted(decode_values)} base={sorted(base_values)}",
    )

    config_errors: List[str] = []
    for task in args.tasks:
        configs = {
            method: read_json(args.output_path / "runs" / method / task / "config.json", {})
            for method in args.methods
        }
        for method, config in configs.items():
            signature = material_config_signature(config)
            if signature["decode_policy"] != args.decode_policy:
                config_errors.append(f"{method}/{task}: decode_policy={signature['decode_policy']}")
            if signature["fairness_base_scoring"] != args.base_scoring:
                config_errors.append(
                    f"{method}/{task}: fairness_base_scoring={signature['fairness_base_scoring']}"
                )
            if signature["configured_fairness_base_scoring"] != args.base_scoring:
                config_errors.append(
                    f"{method}/{task}: slim config fairness override="
                    f"{signature['configured_fairness_base_scoring']}"
                )
            if int(signature["generation_max_new_tokens"] or 0) != int(args.max_new_tokens):
                config_errors.append(
                    f"{method}/{task}: generation_max_new_tokens="
                    f"{signature['generation_max_new_tokens']} expected={args.max_new_tokens}"
                )
        add_methods = [method for method in ["random_add", "delta_o_add", "drm_add"] if method in configs]
        if len(add_methods) >= 2:
            signatures = {method: material_config_signature(configs[method]) for method in add_methods}
            if len({canonical_json(value) for value in signatures.values()}) != 1:
                config_errors.append(f"{task}: Add trio material config differs: {signatures}")
            rescue_scores = {
                method: configs[method].get("args", {}).get("rescue_score")
                for method in add_methods
            }
            expected_rescue_scores = {
                method: {
                    "random_add": "random",
                    "delta_o_add": "delta_o",
                    "drm_add": "drm",
                }[method]
                for method in add_methods
            }
            if rescue_scores != expected_rescue_scores:
                config_errors.append(f"{task}: Add rescue scores={rescue_scores}")
        swap_methods = [
            method for method in sorted(FIXED_SWAP_SELECTOR_METHODS) if method in configs
        ]
        if len(swap_methods) == 2:
            signatures = {
                method: material_config_signature(configs[method])
                for method in swap_methods
            }
            if len({canonical_json(value) for value in signatures.values()}) != 1:
                config_errors.append(
                    f"{task}: budget-neutral swap material configs differ: {signatures}"
                )
            swap_scores = {
                method: configs[method].get("args", {}).get("rescue_score")
                for method in swap_methods
            }
            if swap_scores != {
                "budget_neutral_drm_swap": "drm",
                "random_budget_neutral_swap": "random",
            }:
                config_errors.append(f"{task}: swap rescue scores={swap_scores}")
    add_check("material_config_and_add_trio_identity", not config_errors, "; ".join(config_errors) or "matched")

    budget_errors: List[str] = []
    over_budget = [row for row in rows if bool(row.get("over_budget"))]
    mismatch = [row for row in rows if not bool(row.get("budget_equal_to_target"))]
    for task in args.tasks:
        larger = {
            str(row["sample_id"]): row
            for row in by_method_task.get(("sliminfer_larger", task), [])
        }
        for method in ["random_add", "delta_o_add", "drm_add"]:
            for row in by_method_task.get((method, task), []):
                reference = larger.get(str(row["sample_id"]))
                if reference is None:
                    if not args.allow_method_subset:
                        budget_errors.append(f"{method}/{task}: missing larger reference")
                    continue
                for field in [
                    "target_K_plus_M",
                    "selected_block_count",
                    "selected_token_count",
                    "per_layer_block_schedule",
                ]:
                    if row.get(field) != reference.get(field):
                        budget_errors.append(
                            f"{method}/{task}/{row['sample_id']}: {field}={row.get(field)} "
                            f"vs larger={reference.get(field)}"
                        )
        original = {
            str(row["sample_id"]): row
            for row in by_method_task.get(("sliminfer_original", task), [])
        }
        for method in sorted(BUDGET_NEUTRAL_SWAP_METHODS):
            for row in by_method_task.get((method, task), []):
                reference = original.get(str(row["sample_id"]))
                if reference is None:
                    budget_errors.append(f"{method}/{task}: missing original reference")
                    continue
                for field in [
                    "selected_block_count",
                    "selected_token_count",
                    "per_layer_block_schedule",
                ]:
                    if row.get(field) != reference.get(field):
                        budget_errors.append(
                            f"{method}/{task}/{row['sample_id']}: "
                            f"{field} differs from original"
                        )
    if over_budget:
        budget_errors.append(f"over_budget rows={len(over_budget)}")
    if mismatch:
        budget_errors.append(f"budget mismatch rows={len(mismatch)}")
    add_check("K_and_KplusM_actual_alignment", not budget_errors, "; ".join(budget_errors) or "aligned")

    gate_errors: List[str] = []
    for task in args.tasks:
        original = {
            str(row["sample_id"]): row
            for row in by_method_task.get(("sliminfer_original", task), [])
        }
        gated_rows: List[Mapping[str, Any]] = []
        for gated_method in GATED_SWAP_METHODS:
            gated_rows.extend(by_method_task.get((gated_method, task), []))
        for row in gated_rows:
            sid = str(row["sample_id"])
            selected = str(row.get("selected_policy_after_gate", ""))
            if selected not in {"original", "drm_swap"}:
                gate_errors.append(f"{task}/{sid}: selected_policy={selected}")
            if str(row.get("selected_policy", "")) != selected:
                gate_errors.append(f"{task}/{sid}: selected_policy aliases disagree")
            if selected == "random_swap":
                gate_errors.append(f"{task}/{sid}: random_swap is forbidden")
            if row.get("final_budget_type") != "K":
                gate_errors.append(f"{task}/{sid}: gated final budget is not K")
            if str(row.get("gate_mode", "")) != args.gate_mode:
                gate_errors.append(f"{task}/{sid}: gate_mode mismatch")
            if not str(row.get("gate_reason", "")).strip():
                gate_errors.append(f"{task}/{sid}: empty gate_reason")
            if bool(row.get("gate_feature_missing")) or bool(row.get("fallback_used")):
                if selected != "original":
                    gate_errors.append(
                        f"{task}/{sid}: missing/fallback did not return Original"
                    )
            if bool(row.get("gate_triggered")) != (
                args.gate_mode == "apply"
                and bool(row.get("would_gate_trigger"))
                and selected == "drm_swap"
            ):
                gate_errors.append(f"{task}/{sid}: gate_triggered inconsistent")
            reference = original.get(sid)
            if reference is None:
                gate_errors.append(f"{task}/{sid}: missing Original reference")
                continue
            if args.gate_mode == "shadow" and selected != "original":
                gate_errors.append(f"{task}/{sid}: shadow did not select Original")
            if selected == "original":
                for field in ["used_blocks", "used_tokens", "per_layer_block_schedule"]:
                    if row.get(field) != reference.get(field):
                        gate_errors.append(
                            f"{task}/{sid}: Original-selected {field} differs"
                        )
                if (
                    args.gate_mode == "shadow"
                    and str(row.get("prediction", ""))
                    != str(reference.get("prediction", ""))
                ):
                    gate_errors.append(
                        f"{task}/{sid}: deterministic shadow prediction differs from Original"
                    )
            elif selected == "drm_swap":
                expected_blocks = (
                    set(int(block_id) for block_id in reference.get("used_blocks", []))
                    - set(
                        int(block_id)
                        for block_id in row.get("candidate_removed_block_ids", [])
                    )
                ) | set(
                    int(block_id)
                    for block_id in row.get("candidate_rescued_block_ids", [])
                )
                if set(int(block_id) for block_id in row.get("used_blocks", [])) != expected_blocks:
                    gate_errors.append(
                        f"{task}/{sid}: DRM-selected blocks differ from its audited candidate swap"
                    )
    add_check(
        "gated_policy_safety",
        not gate_errors,
        "; ".join(gate_errors[:20]) or "shadow/apply semantics aligned",
    )

    forbidden_fallback_rows = [
        row
        for row in rows
        if bool(row.get("fallback_used"))
        and str(row.get("method_key")) not in BUDGET_NEUTRAL_SWAP_METHODS
    ]
    swap_fallback_rows = [
        row
        for row in rows
        if bool(row.get("fallback_used"))
        and str(row.get("method_key")) in BUDGET_NEUTRAL_SWAP_METHODS
    ]
    swap_rows = [
        row for row in rows if str(row.get("method_key")) in BUDGET_NEUTRAL_SWAP_METHODS
    ]
    add_check(
        "fallback_scope_and_reason",
        not forbidden_fallback_rows
        and all(str(row.get("fallback_reason", "")).strip() for row in swap_fallback_rows),
        (
            f"non_swap_fallback=0; swap_fallback={len(swap_fallback_rows)}/"
            f"{len(swap_rows)}"
        ),
    )

    hash_errors: List[str] = []
    preflight_rows = read_jsonl(args.output_path / "preflight_selector_schedule.jsonl")
    preflight_by_key = {
        (str(row.get("method_key")), str(row.get("task")), str(row.get("sample_id"))): row
        for row in preflight_rows
    }
    for task in args.tasks:
        by_sample: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
        for method in args.methods:
            for row in by_method_task[(method, task)]:
                by_sample[str(row["sample_id"])].append(row)
        for sid, sample_rows in by_sample.items():
            for field in ["prompt_hash", "prompt_wrapper_hash", "generation_config_hash", "scorer_hash"]:
                values = {str(row.get(field, "")) for row in sample_rows}
                if len(values) != 1 or "" in values:
                    hash_errors.append(f"{task}/{sid}: {field}={values}")
            for row in sample_rows:
                key = (str(row.get("method_key")), task, sid)
                preflight_row = preflight_by_key.get(key)
                if preflight_row is None:
                    hash_errors.append(f"{key}: missing preflight row")
                else:
                    if row.get("prompt_hash") != preflight_row.get("prompt_hash"):
                        hash_errors.append(f"{key}: runtime/preflight prompt_hash mismatch")
                    if row.get("generation_config_hash") != preflight_row.get("generation_config_hash"):
                        hash_errors.append(f"{key}: runtime/preflight generation_config_hash mismatch")
    add_check("prompt_generation_scorer_hash_alignment", not hash_errors, "; ".join(hash_errors) or "aligned")

    scorer_errors: List[str] = []
    for method in args.methods:
        for task in args.tasks:
            predictions = read_jsonl(args.output_path / "predictions" / method / f"{task}.jsonl")
            try:
                scorer_accepts_task(task, predictions)
            except ClosureError as exc:
                scorer_errors.append(str(exc))
    add_check("longbench_scorer_accepts_predictions", not scorer_errors, "; ".join(scorer_errors) or "all scored")

    status = "PASS" if all(check["passed"] for check in checks) else "FAIL"
    return {
        "status": status,
        "checks": checks,
        "num_audit_rows": len(rows),
        "tasks": args.tasks,
        "methods": args.methods,
        "samples_per_task": args.max_samples_per_task,
    }


def audit_summary(args: argparse.Namespace, rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []

    def nullable_mean(selected_rows: Sequence[Mapping[str, Any]], field: str) -> Any:
        values = [
            float(row[field])
            for row in selected_rows
            if row.get(field) is not None
        ]
        return round(sum(values) / len(values), 6) if values else "unavailable"

    for method in args.methods:
        selected = [row for row in rows if row.get("method_key") == method]
        fallback_reasons: Dict[str, int] = defaultdict(int)
        for row in selected:
            if bool(row.get("fallback_used")):
                fallback_reasons[str(row.get("fallback_reason", "") or "unknown")] += 1
        result.append(
            {
                "method_key": method,
                "method_name": METHOD_SPECS[method]["method_name"],
                "samples": len(selected),
                "decode_policy": args.decode_policy,
                "base_scoring": args.base_scoring,
                "rescue_score": METHOD_SPECS[method]["rescue_score"],
                "final_budget_type": METHOD_SPECS[method]["final_budget_type"],
                "K_mean": round(sum(float(row["K"]) for row in selected) / max(1, len(selected)), 4),
                "M_mean": round(sum(float(row["M"]) for row in selected) / max(1, len(selected)), 4),
                "selected_blocks_mean": round(
                    sum(float(row["selected_block_count"]) for row in selected) / max(1, len(selected)),
                    4,
                ),
                "selected_blocks_min": min(
                    (int(row["selected_block_count"]) for row in selected),
                    default=0,
                ),
                "selected_blocks_max": max(
                    (int(row["selected_block_count"]) for row in selected),
                    default=0,
                ),
                "selected_tokens_mean": round(
                    sum(float(row["selected_token_count"]) for row in selected) / max(1, len(selected)),
                    4,
                ),
                "used_tokens_mean": round(
                    sum(float(len(row["used_tokens"])) for row in selected)
                    / max(1, len(selected)),
                    4,
                ),
                "swap_count_mean": round(
                    sum(float(row["swap_count"]) for row in selected)
                    / max(1, len(selected)),
                    6,
                ),
                "rescue_count_mean": round(
                    sum(float(row["rescue_count"]) for row in selected)
                    / max(1, len(selected)),
                    6,
                ),
                "removed_count_mean": round(
                    sum(float(row["removed_count"]) for row in selected)
                    / max(1, len(selected)),
                    6,
                ),
                "rescue_count_sum": sum(int(row["rescue_count"]) for row in selected),
                "added_count_sum": sum(int(row["added_count"]) for row in selected),
                "swap_count_sum": sum(int(row["swap_count"]) for row in selected),
                "removed_count_sum": sum(int(row["removed_count"]) for row in selected),
                "rescue_score_mean": round(
                    sum(float(row["rescue_score_mean"]) for row in selected)
                    / max(1, len(selected)),
                    6,
                ),
                "removed_score_mean": round(
                    sum(float(row["removed_score_mean"]) for row in selected)
                    / max(1, len(selected)),
                    6,
                ),
                "active_redundancy_mean": round(
                    sum(float(row["active_redundancy_mean"]) for row in selected)
                    / max(1, len(selected)),
                    6,
                ),
                "gate_mode": (
                    args.gate_mode
                    if method in GATED_SWAP_METHODS
                    else "not_applicable"
                ),
                "gate_triggered": sum(
                    bool(row.get("gate_triggered")) for row in selected
                ),
                "would_gate_trigger": sum(
                    bool(row.get("would_gate_trigger")) for row in selected
                ),
                "gate_feature_missing": sum(
                    bool(row.get("gate_feature_missing")) for row in selected
                ),
                "selected_policy_original": sum(
                    row.get("selected_policy_after_gate") == "original"
                    for row in selected
                ),
                "selected_policy_drm_swap": sum(
                    row.get("selected_policy_after_gate") == "drm_swap"
                    for row in selected
                ),
                "rescue_boundary_margin_mean": nullable_mean(
                    selected, "rescue_boundary_margin"
                ),
                "deletion_risk_mean": nullable_mean(selected, "deletion_risk"),
                "rescue_coverage_gap_mean": nullable_mean(
                    selected, "rescue_coverage_gap"
                ),
                "budget_not_equal_to_sliminfer": sum(
                    not bool(row["budget_equal_to_sliminfer"]) for row in selected
                ),
                "budget_mismatch": sum(not bool(row["budget_equal_to_target"]) for row in selected),
                "over_budget": sum(bool(row["over_budget"]) for row in selected),
                "fallback_used": sum(bool(row["fallback_used"]) for row in selected),
                "fallback_rate": round(
                    sum(bool(row["fallback_used"]) for row in selected) / max(1, len(selected)),
                    6,
                ),
                "fallback_reason_counts": canonical_json(dict(sorted(fallback_reasons.items()))),
                "protected_block_violation": sum(
                    bool(row.get("protected_block_violation")) for row in selected
                ),
                "empty_predictions": sum(not str(row["prediction"]).strip() for row in selected),
                "latency_available": any(bool(row["latency_available"]) for row in selected),
                "memory_available": any(bool(row["memory_available"]) for row in selected),
                "kv_size_available": any(bool(row["kv_size_available"]) for row in selected),
                "compression_ratio_available": any(
                    bool(row["compression_ratio_available"]) for row in selected
                ),
                "throughput_available": any(bool(row["throughput_available"]) for row in selected),
            }
        )
    return result


def compute_method_task_scores(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["method_key"]), str(row["task"]))].append(float(row["prediction_score"]))
    result: List[Dict[str, Any]] = []
    for (method, task), scores in sorted(grouped.items()):
        result.append(
            {
                "method_key": method,
                "task": task,
                "n": len(scores),
                # Match longbench_eval.scorer, which reports each task at 2 decimals.
                "score": round(sum(scores) / max(1, len(scores)), 2),
            }
        )
    return result


def compute_macro_scores(task_scores: Sequence[Mapping[str, Any]], methods: Sequence[str]) -> List[Dict[str, Any]]:
    by_method: Dict[str, List[float]] = defaultdict(list)
    for row in task_scores:
        by_method[str(row["method_key"])].append(float(row["score"]))
    result: List[Dict[str, Any]] = []
    for method in methods:
        scores = by_method.get(method, [])
        macro = sum(scores) / max(1, len(scores))
        result.append({"method_key": method, "macro": round(macro, 4), "tasks": len(scores)})
    return result


def percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    position = max(0.0, min(1.0, float(quantile))) * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def swap_pair_analysis(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    original_rows = {
        (str(row["task"]), str(row["sample_id"])): row
        for row in rows
        if row.get("method_key") == "sliminfer_original"
    }
    random_rows = {
        (str(row["task"]), str(row["sample_id"])): row
        for row in rows
        if row.get("method_key") == "random_budget_neutral_swap"
    }
    drm_rows = {
        (str(row["task"]), str(row["sample_id"])): row
        for row in rows
        if row.get("method_key") == "budget_neutral_drm_swap"
    }
    paired: List[Dict[str, Any]] = []
    for key in sorted(set(random_rows).intersection(drm_rows)):
        random_row = random_rows[key]
        drm_row = drm_rows[key]
        original_row = original_rows.get(key, {})
        random_blocks = set(int(block_id) for block_id in random_row.get("used_blocks", []))
        drm_blocks = set(int(block_id) for block_id in drm_row.get("used_blocks", []))
        union = random_blocks | drm_blocks
        jaccard = len(random_blocks & drm_blocks) / max(1, len(union))
        paired.append(
            {
                "task": key[0],
                "sample_id": key[1],
                "sample_index": int(drm_row.get("sample_index", -1)),
                "gold": drm_row.get("gold_answers", []),
                "original_prediction": original_row.get("prediction", ""),
                "random_prediction": random_row.get("prediction", ""),
                "drm_prediction": drm_row.get("prediction", ""),
                "original_score": float(original_row.get("prediction_score", 0.0)),
                "random_score": float(random_row.get("prediction_score", 0.0)),
                "drm_score": float(drm_row.get("prediction_score", 0.0)),
                "score_delta_drm_minus_original": round(
                    float(drm_row.get("prediction_score", 0.0))
                    - float(original_row.get("prediction_score", 0.0)),
                    6,
                ),
                "score_delta_drm_minus_random": round(
                    float(drm_row.get("prediction_score", 0.0))
                    - float(random_row.get("prediction_score", 0.0)),
                    6,
                ),
                "used_blocks_jaccard": round(jaccard, 6),
                "random_removed_block_ids": random_row.get("removed_block_ids", []),
                "random_rescued_block_ids": random_row.get("rescued_block_ids", []),
                "drm_removed_block_ids": drm_row.get("removed_block_ids", []),
                "drm_rescued_block_ids": drm_row.get("rescued_block_ids", []),
                "random_fallback_used": bool(random_row.get("fallback_used")),
                "random_fallback_reason": random_row.get("fallback_reason", ""),
                "drm_fallback_used": bool(drm_row.get("fallback_used")),
                "drm_fallback_reason": drm_row.get("fallback_reason", ""),
            }
        )
    jaccards = [float(row["used_blocks_jaccard"]) for row in paired]
    per_task: Dict[str, Dict[str, float]] = {}
    for task in sorted({str(row["task"]) for row in paired}):
        task_values = [
            float(row["used_blocks_jaccard"]) for row in paired if row["task"] == task
        ]
        per_task[task] = {
            "n": len(task_values),
            "mean": round(sum(task_values) / max(1, len(task_values)), 6),
            "p50": round(percentile(task_values, 0.50), 6),
            "p95": round(percentile(task_values, 0.95), 6),
        }
    return {
        "n": len(paired),
        "used_blocks_jaccard": {
            "mean": round(sum(jaccards) / max(1, len(jaccards)), 6),
            "p50": round(percentile(jaccards, 0.50), 6),
            "p95": round(percentile(jaccards, 0.95), 6),
            "per_task": per_task,
        },
        "top_positive_cases": sorted(
            paired,
            key=lambda row: (
                float(row["score_delta_drm_minus_random"]),
                row["task"],
                row["sample_id"],
            ),
            reverse=True,
        )[:10],
        "top_negative_cases": sorted(
            paired,
            key=lambda row: (
                float(row["score_delta_drm_minus_random"]),
                row["task"],
                row["sample_id"],
            ),
        )[:10],
        "paired_cases": paired,
    }


def budget_neutral_stage_recommendation(
    args: argparse.Namespace,
    macro_scores: Sequence[Mapping[str, Any]],
    task_scores: Sequence[Mapping[str, Any]],
    validation: Mapping[str, Any],
    summaries: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    macro_by_method = {str(row["method_key"]): float(row["macro"]) for row in macro_scores}
    task_by_method = {
        (str(row["method_key"]), str(row["task"])): float(row["score"])
        for row in task_scores
    }
    summary_by_method = {
        str(row["method_key"]): row
        for row in summaries
    }
    drm_method = "budget_neutral_drm_swap"
    random_method = "random_budget_neutral_swap"
    drm_macro = macro_by_method.get(drm_method, 0.0)
    baselines = [
        "sliminfer_original",
        random_method,
        "sliminfer_larger",
        "random_add",
        "drm_add",
    ]
    macro_deltas = {
        f"delta_drm_swap_vs_{baseline}": round(
            drm_macro - macro_by_method[baseline],
            4,
        )
        for baseline in baselines
        if baseline in macro_by_method
    }
    random_vs_original = None
    if random_method in macro_by_method and "sliminfer_original" in macro_by_method:
        random_vs_original = round(
            macro_by_method[random_method] - macro_by_method["sliminfer_original"],
            4,
        )
    per_task_drm_vs_random = []
    for task in args.tasks:
        drm_score = task_by_method.get((drm_method, task))
        random_score = task_by_method.get((random_method, task))
        if drm_score is None or random_score is None:
            continue
        per_task_drm_vs_random.append(
            {
                "task": task,
                "drm_swap": drm_score,
                "random_swap": random_score,
                "delta_drm_swap_vs_random_swap": round(drm_score - random_score, 4),
            }
        )

    multihop_deltas = []
    for task in ["hotpotqa", "2wikimqa", "musique"]:
        drm_score = task_by_method.get((drm_method, task))
        original_score = task_by_method.get(("sliminfer_original", task))
        if drm_score is None or original_score is None:
            continue
        multihop_deltas.append(
            {
                "task": task,
                "delta_drm_swap_vs_original": round(drm_score - original_score, 4),
            }
        )
    multihop_mean_delta = (
        sum(float(row["delta_drm_swap_vs_original"]) for row in multihop_deltas)
        / len(multihop_deltas)
        if multihop_deltas
        else None
    )
    # "No obvious overall decline" is operationalized as a non-negative mean
    # across the three requested multi-hop tasks. Exact per-task deltas remain
    # in the report so this aggregate cannot conceal an isolated regression.
    multihop_no_obvious_overall_decline = (
        len(multihop_deltas) == 3
        and multihop_mean_delta is not None
        and multihop_mean_delta >= 0.0
    )

    drm_summary = summary_by_method.get(drm_method, {})
    fallback_rate = float(drm_summary.get("fallback_rate", 1.0))
    fallback_reason_counts = json.loads(
        str(drm_summary.get("fallback_reason_counts", "{}") or "{}")
    )
    fallback_total = sum(int(count) for count in fallback_reason_counts.values())
    reasonable_fallbacks = int(
        fallback_reason_counts.get("no_final_pruning_candidates", 0)
    )
    reasonable_fallback_fraction = (
        reasonable_fallbacks / fallback_total if fallback_total else 1.0
    )
    fallback_reasons_reasonable = reasonable_fallback_fraction >= 0.5
    audit_pass = validation.get("status") == "PASS"
    drm_vs_original = macro_deltas.get("delta_drm_swap_vs_sliminfer_original")
    drm_vs_random = macro_deltas.get(
        "delta_drm_swap_vs_random_budget_neutral_swap"
    )
    automatic_stage_recommendation_applicable = int(args.max_samples_per_task) <= 64
    recommend_next_stage = (
        (
            args.mode == "run"
            and audit_pass
            and drm_vs_original is not None
            and drm_vs_original >= 0.30
            and drm_vs_random is not None
            and drm_vs_random >= 0.20
            and multihop_no_obvious_overall_decline
            and fallback_rate <= 0.20
            and fallback_reasons_reasonable
        )
        if automatic_stage_recommendation_applicable
        else None
    )
    return {
        "recommend_next_stage_128_200_multihop_diagnostic": recommend_next_stage,
        "automatic_stage_recommendation_applicable": automatic_stage_recommendation_applicable,
        "automatic_stage_recommendation_deferred_reason": (
            ""
            if automatic_stage_recommendation_applicable
            else (
                "Runs with more than 64 samples require the requested paired-bootstrap, "
                "gain-concentration, and per-task stability analysis before a next-stage decision."
            )
        ),
        "recommend_full_run": False,
        "macro_scores": macro_by_method,
        "macro_deltas": macro_deltas,
        "delta_random_swap_vs_original": random_vs_original,
        "per_task_drm_swap_vs_random_swap": per_task_drm_vs_random,
        "multihop_deltas_vs_original": multihop_deltas,
        "multihop_mean_delta_vs_original": (
            round(multihop_mean_delta, 4)
            if multihop_mean_delta is not None
            else None
        ),
        "multihop_no_obvious_overall_decline": multihop_no_obvious_overall_decline,
        "multihop_rule": (
            "mean DRM-Swap minus SlimInfer-original delta over "
            "hotpotqa, 2wikimqa, and musique must be >= 0"
        ),
        "drm_swap_fallback_rate": fallback_rate,
        "drm_swap_fallback_reason_counts": fallback_reason_counts,
        "reasonable_fallback_fraction": round(reasonable_fallback_fraction, 6),
        "fallback_reasons_reasonable": fallback_reasons_reasonable,
        "audit_pass": audit_pass,
        "thresholds": {
            "delta_drm_swap_vs_original_min": 0.30,
            "delta_drm_swap_vs_random_swap_min": 0.20,
            "drm_swap_fallback_rate_max": 0.20,
            "reasonable_fallback_fraction_min": 0.50,
        },
        "note": (
            "Diagnostic only; never a full-LongBench recommendation. "
            "No statistical significance or systems-performance advantage claimed."
        ),
    }


def full_run_recommendation(
    args: argparse.Namespace,
    macro_scores: Sequence[Mapping[str, Any]],
    task_scores: Sequence[Mapping[str, Any]],
    validation: Mapping[str, Any],
) -> Dict[str, Any]:
    macro_by_method = {str(row["method_key"]): float(row["macro"]) for row in macro_scores}
    original = macro_by_method.get("sliminfer_original", 0.0)
    larger = macro_by_method.get("sliminfer_larger", 0.0)
    drm = macro_by_method.get("drm_add", 0.0)
    multihop = {"hotpotqa", "2wikimqa", "musique"}
    task_by_method = {
        (str(row["method_key"]), str(row["task"])): float(row["score"]) for row in task_scores
    }
    multihop_drops = []
    for task in multihop:
        if task not in args.tasks:
            continue
        r_score = task_by_method.get(("sliminfer_original", task))
        d_score = task_by_method.get(("drm_add", task))
        if r_score is None or d_score is None:
            continue
        multihop_drops.append({"task": task, "delta_drm_vs_original": round(d_score - r_score, 4)})
    audit_pass = validation.get("status") == "PASS"
    recommend = (
        audit_pass
        and drm >= original + 0.30
        and drm >= larger
        and all(item["delta_drm_vs_original"] >= 0.0 for item in multihop_drops)
    )
    return {
        "recommend_full_run": recommend,
        "drm_macro": drm,
        "original_macro": original,
        "larger_macro": larger,
        "delta_drm_vs_original": round(drm - original, 4),
        "delta_drm_vs_larger": round(drm - larger, 4),
        "multihop_deltas": multihop_drops,
        "multihop_no_decline": all(
            item["delta_drm_vs_original"] >= 0.0 for item in multihop_drops
        ),
        "audit_pass": audit_pass,
        "note": (
            "64-sample diagnostic only; not full LongBench. "
            "No statistical significance claimed."
        ),
    }


def write_report(
    args: argparse.Namespace,
    validation: Mapping[str, Any],
    summaries: Sequence[Mapping[str, Any]],
    audit_rows: Sequence[Mapping[str, Any]],
) -> None:
    task_scores = compute_method_task_scores(audit_rows)
    macro_scores = compute_macro_scores(task_scores, args.methods)
    write_csv(
        args.output_path / "method_task_scores.csv",
        task_scores,
        ["method_key", "task", "n", "score"],
    )
    write_csv(
        args.output_path / "macro_scores.csv",
        macro_scores,
        ["method_key", "macro", "tasks"],
    )
    is_swap_comparison = BUDGET_NEUTRAL_SWAP_METHODS.issubset(set(args.methods))
    if is_swap_comparison:
        pair_analysis = swap_pair_analysis(audit_rows)
        write_json(args.output_path / "swap_pair_analysis.json", pair_analysis)
        recommendation = budget_neutral_stage_recommendation(
            args,
            macro_scores,
            task_scores,
            validation,
            summaries,
        )
        write_json(args.output_path / "next_stage_recommendation.json", recommendation)
    else:
        pair_analysis = {}
        recommendation = full_run_recommendation(
            args,
            macro_scores,
            task_scores,
            validation,
        )
        write_json(args.output_path / "full_run_recommendation.json", recommendation)
    max_tokens_desc = (
        f"task-specific dataset2maxlen ({DATASET2MAXLEN_CONFIG.name})"
        if int(args.max_new_tokens) <= 0
        else str(args.max_new_tokens)
    )
    title = (
        f"# DRM Fairness Closure {args.max_samples_per_task}-Sample Diagnostic"
        if args.mode == "run"
        else "# DRM Fairness Closure Smoke"
    )
    lines = [
        title,
        "",
        f"- Status: **{validation['status']}**",
        f"- Mode: `{args.mode}`",
        f"- Tasks: {', '.join(args.tasks)}",
        f"- Samples per task: {args.max_samples_per_task}",
        f"- Decode policy: `{args.decode_policy}`",
        f"- Base scoring: `{args.base_scoring}`",
        f"- Max new tokens: {max_tokens_desc}",
        "- Execution: sequential; GPU free-memory gate before every method/task cell.",
        (
            f"- This is a {args.max_samples_per_task}-sample fairness/quality diagnostic, "
            "not full LongBench."
        ),
        "",
        "## Method-task scores",
        "",
    ]
    for row in task_scores:
        lines.append(f"- `{row['method_key']}` / `{row['task']}`: **{row['score']:.2f}** (n={row['n']})")
    lines.extend(["", "## Macro averages", ""])
    macro_by_method = {str(row["method_key"]): float(row["macro"]) for row in macro_scores}
    for row in macro_scores:
        lines.append(f"- `{row['method_key']}`: **{row['macro']:.2f}**")
    if "drm_add" in macro_by_method:
        drm = macro_by_method["drm_add"]
        for baseline in ["sliminfer_original", "sliminfer_larger", "random_add", "delta_o_add"]:
            if baseline in macro_by_method:
                lines.append(f"- DRM-Add vs `{baseline}`: **{drm - macro_by_method[baseline]:+.2f}**")
    if "budget_neutral_drm_swap" in macro_by_method:
        drm_swap = macro_by_method["budget_neutral_drm_swap"]
        for baseline in [
            "sliminfer_original",
            "random_budget_neutral_swap",
            "sliminfer_larger",
            "random_add",
            "drm_add",
        ]:
            if baseline in macro_by_method:
                lines.append(
                    f"- DRM-Swap vs `{baseline}`: "
                    f"**{drm_swap - macro_by_method[baseline]:+.2f}**"
                )
    if (
        "random_budget_neutral_swap" in macro_by_method
        and "sliminfer_original" in macro_by_method
    ):
        lines.append(
            "- Random-Swap vs `sliminfer_original`: "
            f"**{macro_by_method['random_budget_neutral_swap'] - macro_by_method['sliminfer_original']:+.2f}**"
        )
    lines.extend(["", "## Checks", ""])
    for check in validation["checks"]:
        lines.append(f"- {'PASS' if check['passed'] else 'FAIL'} `{check['check']}`: {check['detail']}")
    lines.extend(["", "## Audit summary", ""])
    for row in summaries:
        lines.append(
            f"- `{row['method_key']}`: n={row['samples']}, K_mean={row['K_mean']}, "
            f"M_mean={row['M_mean']}, blocks_mean={row['selected_blocks_mean']}, "
            f"tokens_mean={row['selected_tokens_mean']}, over_budget={row['over_budget']}, "
            f"budget_mismatch={row['budget_mismatch']}, fallback_used={row['fallback_used']}, "
            f"fallback_rate={row['fallback_rate']:.6f}, "
            f"rescue_count_sum={row['rescue_count_sum']}, added_count_sum={row['added_count_sum']}, "
            f"swap_count_sum={row['swap_count_sum']}, removed_count_sum={row['removed_count_sum']}, "
            f"protected_violation={row['protected_block_violation']}, "
            f"latency_available={row['latency_available']}, memory_available={row['memory_available']}, "
            f"kv_size_available={row['kv_size_available']}, "
            f"throughput_available={row['throughput_available']}."
        )
    if is_swap_comparison:
        jaccard = pair_analysis["used_blocks_jaccard"]
        lines.extend(
            [
                "",
                "## Budget-neutral swap comparison",
                "",
                f"- paired samples: {pair_analysis['n']}",
                (
                    "- DRM-Swap/Random-Swap used-block Jaccard: "
                    f"mean={jaccard['mean']:.6f}, P50={jaccard['p50']:.6f}, "
                    f"P95={jaccard['p95']:.6f}"
                ),
                "",
                "## Next-stage threshold",
                "",
                (
                    "- automatic next-stage recommendation: "
                    + (
                        f"**{recommendation['recommend_next_stage_128_200_multihop_diagnostic']}**"
                        if recommendation["automatic_stage_recommendation_applicable"]
                        else (
                            "**deferred** pending paired-bootstrap, gain-concentration, "
                            "and per-task stability analysis"
                        )
                    )
                ),
                "- recommend_full_run: **False**",
                f"- {recommendation['note']}",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "## Full-run threshold",
                "",
                f"- recommend_full_run: **{recommendation['recommend_full_run']}**",
                f"- DRM macro: {recommendation['drm_macro']:.2f}",
                f"- delta vs original: {recommendation['delta_drm_vs_original']:+.2f}",
                f"- delta vs larger: {recommendation['delta_drm_vs_larger']:+.2f}",
                f"- {recommendation['note']}",
            ]
        )
    report_name = "diagnostic_report.md" if args.mode == "run" else "smoke_report.md"
    (args.output_path / report_name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = normalize_args(parse_args())
    if args.resume:
        if not args.output_dir:
            raise ClosureError("--resume requires an explicit --output_dir")
        if not args.output_path.exists():
            raise ClosureError(f"resume requested but output_dir does not exist: {args.output_path}")
        args.output_path.mkdir(parents=True, exist_ok=True)
    else:
        args.output_path.mkdir(parents=True, exist_ok=False)
    try:
        protocol = shared_protocol(args)
        matrix = method_matrix(args)
        static_errors = validate_static_matrix(matrix, args.allow_method_subset)
        if static_errors:
            raise ClosureError("; ".join(static_errors))
        config_path = args.output_path / "config.json"
        matrix_path = args.output_path / "method_matrix.json"
        if args.resume:
            existing_config = read_json(config_path, {})
            existing_matrix = read_json(matrix_path, [])
            if not existing_config:
                raise ClosureError(f"resume contract missing: {config_path}")
            if str(existing_config.get("mode")) != args.mode:
                raise ClosureError(
                    f"resume mode mismatch: existing={existing_config.get('mode')} requested={args.mode}"
                )
            if canonical_json(existing_config.get("protocol", {})) != canonical_json(protocol):
                raise ClosureError("resume protocol differs from existing config.json")
            if canonical_json(existing_matrix) != canonical_json(matrix):
                raise ClosureError("resume method matrix differs from existing method_matrix.json")
        else:
            write_json(
                config_path,
                {
                    "stage": "drm_fairness_closure",
                    "mode": args.mode,
                    "output_dir": str(args.output_path),
                    "protocol": protocol,
                    "method_matrix": matrix,
                    "historical_results_modified": False,
                },
            )
            write_json(matrix_path, matrix)
        preflight_path = args.output_path / "preflight_selector_schedule.jsonl"
        if args.resume and preflight_path.exists():
            preflight_rows = read_jsonl(preflight_path)
            regenerated_preflight = cpu_preflight(args)
            if canonical_json(preflight_rows) != canonical_json(regenerated_preflight):
                raise ClosureError(
                    "resume CPU preflight differs from existing schedule; "
                    "data, prompt wrapper, or task protocol changed"
                )
        else:
            preflight_rows = cpu_preflight(args)
            write_jsonl(preflight_path, preflight_rows)
        preflight_errors = validate_preflight_matrix(args, preflight_rows)
        if preflight_errors:
            raise ClosureError("; ".join(preflight_errors))
        preflight = {
            "status": "PASS",
            "cpu_only": True,
            "model_weights_loaded": False,
            "num_rows": len(preflight_rows),
            "method_matrix_errors": [],
            "prompt_hashes_present": all(bool(row.get("prompt_hash")) for row in preflight_rows),
        }
        write_json(args.output_path / "preflight_report.json", preflight)
        if args.mode == "preflight":
            print(json.dumps({"status": "PASS", "output_dir": str(args.output_path), **preflight}, indent=2))
            return 0

        if args.mode == "gate_audit":
            run_gate_audit_cells(args)
            gate_rows = consolidate_gate_audit(args)
            write_jsonl(
                args.output_path / "audit" / "gate_feature_audit.jsonl",
                gate_rows,
            )
            validation, feature_summary = validate_and_summarize_gate_audit(
                args, gate_rows
            )
            write_json(
                args.output_path / "gate_audit_validation.json", validation
            )
            write_json(
                args.output_path / "gate_feature_distribution.json",
                feature_summary,
            )
            write_json(
                args.output_path / "retrospective_association.json",
                retrospective_gate_association(args, gate_rows),
            )
            write_json(
                args.output_path / "run_status.json",
                {
                    "status": validation["status"],
                    "mode": "gate_audit",
                    "completed_at": datetime.now().isoformat(),
                    "audit_rows": len(gate_rows),
                    "quality_generation_performed": False,
                    "scorer_invoked": False,
                },
            )
            print(json.dumps({
                "status": validation["status"],
                "output_dir": str(args.output_path),
                "audit_rows": len(gate_rows),
                "quality_generation_performed": False,
                "scorer_invoked": False,
            }, indent=2))
            return 0 if validation["status"] == "PASS" else 2

        run_cells(args)
        audit_rows = consolidate_audit(args)
        write_jsonl(args.output_path / "audit" / "per_sample_audit.jsonl", audit_rows)
        summaries = audit_summary(args, audit_rows)
        write_csv(
            args.output_path / "audit" / "audit_summary.csv",
            summaries,
            [
                "method_key",
                "method_name",
                "samples",
                "decode_policy",
                "base_scoring",
                "rescue_score",
                "final_budget_type",
                "K_mean",
                "M_mean",
                "selected_blocks_mean",
                "selected_blocks_min",
                "selected_blocks_max",
                "selected_tokens_mean",
                "used_tokens_mean",
                "swap_count_mean",
                "rescue_count_mean",
                "removed_count_mean",
                "rescue_count_sum",
                "added_count_sum",
                "swap_count_sum",
                "removed_count_sum",
                "rescue_score_mean",
                "removed_score_mean",
                "active_redundancy_mean",
                "gate_mode",
                "gate_triggered",
                "would_gate_trigger",
                "gate_feature_missing",
                "selected_policy_original",
                "selected_policy_drm_swap",
                "rescue_boundary_margin_mean",
                "deletion_risk_mean",
                "rescue_coverage_gap_mean",
                "budget_not_equal_to_sliminfer",
                "budget_mismatch",
                "over_budget",
                "fallback_used",
                "fallback_rate",
                "fallback_reason_counts",
                "protected_block_violation",
                "empty_predictions",
                "latency_available",
                "memory_available",
                "kv_size_available",
                "compression_ratio_available",
                "throughput_available",
            ],
        )
        validation = validate_smoke(args, audit_rows)
        validation_name = "diagnostic_validation.json" if args.mode == "run" else "smoke_validation.json"
        write_json(args.output_path / validation_name, validation)
        write_report(args, validation, summaries, audit_rows)
        write_json(
            args.output_path / "run_status.json",
            {
                "status": validation["status"],
                "mode": args.mode,
                "completed_at": datetime.now().isoformat(),
                "audit_rows": len(audit_rows),
            },
        )
        print(
            json.dumps(
                {
                    "status": validation["status"],
                    "output_dir": str(args.output_path),
                    "audit_rows": len(audit_rows),
                },
                indent=2,
            )
        )
        return 0 if validation["status"] == "PASS" else 2
    except Exception as exc:
        failure = {
            "status": "FAIL",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "failed_at": datetime.now().isoformat(),
        }
        write_json(args.output_path / "failure.json", failure)
        write_json(
            args.output_path / "failures" / f"failure_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            failure,
        )
        print(json.dumps({"output_dir": str(args.output_path), **failure}, indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
