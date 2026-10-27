#!/usr/bin/env python3
"""Build authoritative DRM CLI commands for LLaMA on the Qwen machine."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from tools.llama_drm_on_qwen_v1.common import resolve_method_key
from tools.llama_drm_on_qwen_v1.constants import (
    ALPHA_DIFF,
    AUTHORITATIVE_RUNNER,
    BACKEND,
    BLOCK_SIZE,
    DATA_ROOT,
    DTYPE,
    ETA_ACTIVE,
    ETA_RESCUE,
    LAMBDA_RED,
    LOCAL_QUERY_WINDOW,
    MAX_MODEL_LEN,
    MODEL_FAMILY,
    MODEL_PATH,
    PRUNE_LAYERS,
    PYTHON,
    RESCUE_RATIO,
    RETAIN_TOKENS,
    SELECTION_PATH,
    SWAP_THRESHOLD,
    TOKEN_UNIT_SIZE,
)


def base_runtime_flags(
    *,
    method_key: str,
    output_dir: Path,
    tasks: Sequence[str],
    sample_indices_by_task: Optional[Mapping[str, Sequence[int]]] = None,
    seed: Optional[int] = None,
    rescue_ratio: Optional[float] = None,
    alpha_diff: Optional[float] = None,
    lambda_red: Optional[float] = None,
    eta_rescue: Optional[float] = None,
    eta_active: Optional[float] = None,
    policy: Optional[str] = None,
    score: Optional[str] = None,
    eval_mode: str = "longbench",
    context_lengths: Optional[str] = None,
    decode_tokens: int = 16,
    active_block_trace: str = "full",
    drm_trace_dir: Optional[Path] = None,
    rescue_audit_path: Optional[Path] = None,
    protocol_id: str = "",
    implementation_revision: str = "",
    longbench_max_samples: int = 0,
) -> List[str]:
    spec = resolve_method_key(method_key)
    use_seed = int(seed if seed is not None else spec["seed"])
    use_policy = policy if policy is not None else str(spec["policy"])
    use_score = score if score is not None else str(spec["score"])
    cmd = [
        str(PYTHON),
        str(AUTHORITATIVE_RUNNER),
        "--model_path", str(MODEL_PATH),
        "--tokenizer_path", str(MODEL_PATH),
        "--model_family", MODEL_FAMILY,
        "--dtype", DTYPE,
        "--device", "cuda",
        "--backend", BACKEND,
        "--longbench_dir", str(DATA_ROOT),
        "--longbench_tasks", ",".join(tasks),
        "--longbench_max_samples", str(longbench_max_samples),
        "--max_model_len", str(MAX_MODEL_LEN),
        "--method", str(spec["runner"]),
        "--eval", eval_mode,
        "--block_strategy", "fixed",
        "--selection_path", SELECTION_PATH,
        "--block_size", str(BLOCK_SIZE),
        "--token_unit_size", str(TOKEN_UNIT_SIZE),
        "--local_query_window", str(LOCAL_QUERY_WINDOW),
        "--swap_threshold", str(SWAP_THRESHOLD),
        "--prune_layers", ",".join(str(x) for x in PRUNE_LAYERS),
        "--retain_tokens", ",".join(str(x) for x in RETAIN_TOKENS),
        "--decode_policy", "original",
        "--rescue_policy", use_policy,
        "--rescue_score", use_score,
        "--rescue_ratio", str(RESCUE_RATIO if rescue_ratio is None else rescue_ratio),
        "--alpha_diff", str(ALPHA_DIFF if alpha_diff is None else alpha_diff),
        "--lambda_red", str(LAMBDA_RED if lambda_red is None else lambda_red),
        "--eta_rescue", str(ETA_RESCUE if eta_rescue is None else eta_rescue),
        "--eta_active", str(ETA_ACTIVE if eta_active is None else eta_active),
        "--temperature", "0.0",
        "--top_p", "1.0",
        "--generation_max_new_tokens", "0",
        "--seed", str(use_seed),
        "--min_free_gpu_gb", "0",
        "--output_dir", str(output_dir),
    ]
    if protocol_id:
        cmd.extend(["--protocol_id", protocol_id])
    if implementation_revision:
        cmd.extend(["--implementation_revision", implementation_revision])
    if sample_indices_by_task:
        parts = []
        for task, indices in sample_indices_by_task.items():
            parts.append(f"{task}:{','.join(str(int(i)) for i in indices)}")
        cmd.extend(["--longbench_sample_indices_by_task", ";".join(parts)])
    if eval_mode == "latency":
        if context_lengths:
            # replace default longbench-only settings
            if "--longbench_tasks" in cmd:
                # keep longbench_dir unused for latency
                pass
            cmd.extend(["--context_lengths", context_lengths, "--decode_tokens", str(decode_tokens)])
    if active_block_trace and active_block_trace != "none":
        cmd.extend(["--active_block_trace", active_block_trace])
        if drm_trace_dir is not None:
            cmd.extend(["--active_block_trace_dir", str(drm_trace_dir)])
    if drm_trace_dir is not None:
        cmd.extend(["--drm_trace_dir", str(drm_trace_dir)])
    if rescue_audit_path is not None:
        cmd.extend(["--rescue_audit_path", str(rescue_audit_path)])
    return cmd


def strip_unsupported_flags(cmd: List[str]) -> List[str]:
    """Drop flags unknown to the authoritative runner if present."""
    unsupported = {"--do_sample", "--num_beams"}
    out: List[str] = []
    skip_next = False
    for item in cmd:
        if skip_next:
            skip_next = False
            continue
        if item in unsupported:
            skip_next = True
            continue
        out.append(item)
    return out
