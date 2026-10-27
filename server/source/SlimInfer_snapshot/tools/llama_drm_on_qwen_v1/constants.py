#!/usr/bin/env python3
"""Frozen constants for LLAMA_DRM_ON_QWEN_V1."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_NAME = "LLAMA_DRM_ON_QWEN_V1"
PROTOCOL_DIR = REPO_ROOT / "protocol" / "llama_drm_on_qwen_v1"
ULB_PROTOCOL_DIR = REPO_ROOT / "protocol" / "unified_longbench_v1"
ULB_PROTOCOL_ID = "ULB-v1-052cfbbeaa2cc42f"

MODEL_PATH = Path("/home/yu-bai/Desktop/jiaqizhang/models/llama31")
DATA_ROOT = Path("/home/yu-bai/Desktop/jiaqizhang/HeadKV/data/LongBench")
PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
AUTHORITATIVE_RUNNER = REPO_ROOT / "experiments" / "run_qwen25_sliminfer_variants.py"
AUTHORITATIVE_CACHE = REPO_ROOT / "utils" / "cache_utils.py"
AUTHORITATIVE_SCORING = REPO_ROOT / "bridgekv" / "rescuekv_scoring.py"
AUTHORITATIVE_BUNDLE = Path(
    "/home/yu-bai/Desktop/jiaqizhang/exports/qwen_drm_authoritative_bundle_20260727_074100"
)

MODEL_REVISION = "UNKNOWN_LOCAL_SNAPSHOT"
DTYPE = "bf16"
MODEL_FAMILY = "llama"
BACKEND = "flash_attention_2"
MAX_MODEL_LEN = 131072
BLOCK_SIZE = 64
TOKEN_UNIT_SIZE = 8
LOCAL_QUERY_WINDOW = 4
SWAP_THRESHOLD = 0.9
PRUNE_LAYERS = (10, 20, 30)
RETAIN_TOKENS = (8192, 4096, 2048)
K_BLOCKS = (128, 64, 32)
M_BLOCKS = (6, 3, 2)
RESCUE_RATIO = 0.05
ALPHA_DIFF = 0.5
LAMBDA_RED = 0.25
ETA_RESCUE = 0.25
ETA_ACTIVE = 0.25
SELECTION_PATH = "canonical_fixed_page"
EFFICIENCY_TAG = "LLAMA_ON_QWEN_3090TI_EFFICIENCY"
GPU_POLL_SECONDS = 60
GLOBAL_FLOCK_NAME = "llama_drm_on_qwen_v1.scheduler.lock"
SERVICE_NAME = "llama-drm-on-qwen-v1.service"

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

SIX_TASKS: Tuple[str, ...] = (
    "qasper",
    "hotpotqa",
    "2wikimqa",
    "musique",
    "gov_report",
    "repobench-p",
)

# Matched Qwen Stage-1 canary indices on the shared LongBench sample order.
CANARY_INDICES: Dict[str, List[int]] = {
    "qasper": [75, 123],
    "hotpotqa": [72, 49],
    "2wikimqa": [167, 176],
    "musique": [138, 182],
    "gov_report": [180, 67],
    "repobench-p": [418, 224],
}

DRM_METHODS: Tuple[str, ...] = (
    "sliminfer_original",
    "sliminfer_larger_km",
    "sliminfer_random_add_km",
    "sliminfer_drm_add_km",
    "drm_swap_original_k",
)

METHOD_SPECS: Dict[str, Dict[str, Any]] = {
    "sliminfer_original": {
        "runner": "sliminfer_original",
        "policy": "none",
        "score": "delta_o",
        "budget": "K",
        "seed": 0,
    },
    "sliminfer_larger_km": {
        "runner": "sliminfer_original",
        "policy": "sliminfer_larger",
        "score": "delta_o",
        "budget": "K+M",
        "seed": 0,
    },
    "sliminfer_random_add_km": {
        "runner": "sliminfer_original",
        "policy": "add",
        "score": "random",
        "budget": "K+M",
        "seed": 0,
    },
    "sliminfer_drm_add_km": {
        "runner": "sliminfer_original",
        "policy": "add",
        "score": "drm",
        "budget": "K+M",
        "seed": 0,
    },
    "drm_swap_original_k": {
        "runner": "sliminfer_original",
        "policy": "budget_neutral_drm_swap",
        "score": "drm",
        "budget": "K",
        "seed": 0,
    },
}

SIX_TASK_METHOD_KEYS: Tuple[str, ...] = (
    "sliminfer_original",
    "sliminfer_larger_km",
    "random_seed_0",
    "random_seed_1",
    "random_seed_2",
    "sliminfer_drm_add_km",
    "drm_swap_original_k",
)

FULL17_METHOD_KEYS: Tuple[str, ...] = (
    "sliminfer_original",
    "sliminfer_larger_km",
    "random_seed_0",
    "sliminfer_drm_add_km",
    "drm_swap_original_k",
)

CANARY_RUNS: Tuple[Dict[str, Any], ...] = (
    {"id": "original_run1", "method_key": "sliminfer_original", "seed": 0},
    {"id": "original_run2", "method_key": "sliminfer_original", "seed": 0},
    {"id": "noop_adapter", "method_key": "sliminfer_original", "seed": 0, "noop": True},
    {"id": "larger", "method_key": "sliminfer_larger_km", "seed": 0},
    {"id": "random_seed0_run1", "method_key": "random_seed_0", "seed": 0},
    {"id": "random_seed0_run2", "method_key": "random_seed_0", "seed": 0},
    {"id": "random_seed1", "method_key": "random_seed_1", "seed": 1},
    {"id": "drm_add", "method_key": "sliminfer_drm_add_km", "seed": 0},
    {"id": "drm_swap", "method_key": "drm_swap_original_k", "seed": 0},
)

ABLATION_VARIANTS: Tuple[Dict[str, Any], ...] = (
    {
        "id": "drm_ratio_0_02",
        "policy": "add",
        "score": "drm",
        "rescue_ratio": 0.02,
        "alpha_diff": 0.5,
        "lambda_red": 0.25,
        "eta_rescue": 0.25,
        "eta_active": 0.25,
        "seed": 0,
        "reuse": None,
    },
    {
        "id": "drm_ratio_0_10",
        "policy": "add",
        "score": "drm",
        "rescue_ratio": 0.10,
        "alpha_diff": 0.5,
        "lambda_red": 0.25,
        "eta_rescue": 0.25,
        "eta_active": 0.25,
        "seed": 0,
        "reuse": None,
    },
    {
        "id": "drm_full_ratio_0_05",
        "policy": "add",
        "score": "drm",
        "rescue_ratio": 0.05,
        "reuse": "sliminfer_drm_add_km",
    },
    {
        "id": "drm_without_gain",
        "policy": "add",
        "score": "drm",
        "rescue_ratio": 0.05,
        "alpha_diff": 0.0,
        "lambda_red": 0.25,
        "eta_rescue": 0.25,
        "eta_active": 0.25,
        "seed": 0,
        "reuse": None,
    },
    {
        "id": "drm_without_active_core_redundancy",
        "policy": "add",
        "score": "drm",
        "rescue_ratio": 0.05,
        "alpha_diff": 0.5,
        "lambda_red": 0.0,
        "eta_rescue": 0.25,
        "eta_active": 0.0,
        "seed": 0,
        "reuse": None,
    },
    {
        "id": "drm_without_rescue_rescue_mmr",
        "policy": "add",
        "score": "drm",
        "rescue_ratio": 0.05,
        "alpha_diff": 0.5,
        "lambda_red": 0.25,
        "eta_rescue": 0.0,
        "eta_active": 0.25,
        "seed": 0,
        "reuse": None,
    },
    {
        "id": "drm_base_utility_only",
        "policy": "add",
        "score": "drm",
        "rescue_ratio": 0.05,
        "alpha_diff": 0.0,
        "lambda_red": 0.0,
        "eta_rescue": 0.0,
        "eta_active": 0.0,
        "seed": 0,
        "reuse": None,
    },
)

EFFICIENCY_METHODS: Tuple[str, ...] = (
    "full_kv",
    "sliminfer_original",
    "sliminfer_larger_km",
    "random_seed_0",
    "sliminfer_drm_add_km",
    "drm_swap_original_k",
)
EFFICIENCY_LENGTHS: Tuple[int, ...] = (8192, 16384, 24576, 28672, 32768)

STAGES: Tuple[Dict[str, Any], ...] = (
    {"id": "static_audit", "name": "static_audit", "gpu": False},
    {"id": "canary_12", "name": "canary_12_sample", "gpu": True},
    {"id": "six_task", "name": "six_task_quality", "gpu": True},
    {"id": "full17", "name": "full_17_task", "gpu": True},
    {"id": "stats_full17", "name": "full17_statistics", "gpu": False},
    {"id": "oom_completion", "name": "native_oom_accuracy_completion", "gpu": True},
    {"id": "ablation", "name": "six_task_budget_component_ablation", "gpu": True},
    {"id": "efficiency", "name": "llama_on_qwen_3090ti_efficiency", "gpu": True},
    {"id": "final_export", "name": "final_statistics_and_paper_export", "gpu": False},
)

DESKTOP_GPU_ALLOWLIST = {
    "/usr/libexec/gnome-remote-desktop-daemon",
    "/usr/lib/xorg/Xorg",
    "/usr/bin/gnome-shell",
    "gnome-remote-desktop-daemon",
    "Xorg",
    "gnome-shell",
}

FORBIDDEN_RUNTIME_SUBSTRINGS = (
    "codex",
    "git pull",
    "pip install",
    "huggingface.co",
    "curl ",
    "wget ",
    "wandb login",
)

AUTHORITATIVE_SOURCE_FILES = (
    AUTHORITATIVE_RUNNER,
    AUTHORITATIVE_CACHE,
    AUTHORITATIVE_SCORING,
    REPO_ROOT / "models" / "modeling_llama.py",
    REPO_ROOT / "bridgekv" / "rescuekv_scoring.py",
    REPO_ROOT / "utils" / "processor.py",
    REPO_ROOT / "longbench_eval.py",
)

CC_FORBIDDEN_IMPORT_MARKERS = (
    "cc" + "_sliminfer_v1",
    "continuity" + "_cluster",
    "run_" + "cc_sliminfer",
)
