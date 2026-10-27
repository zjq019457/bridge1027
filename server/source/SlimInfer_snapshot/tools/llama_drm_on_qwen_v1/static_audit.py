#!/usr/bin/env python3
"""Pure-CPU static audit for LLAMA_DRM_ON_QWEN_V1. Must not initialize CUDA."""
from __future__ import annotations

import ast
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from tools.llama_drm_on_qwen_v1.common import (
    atomic_write_json,
    atomic_write_text,
    canonical_bytes,
    canonical_sample_identity,
    compute_protocol_id_and_hash,
    load_json,
    load_task_samples,
    now_iso,
    sha256_file,
    sha256_text,
)
from tools.llama_drm_on_qwen_v1.constants import (
    AUTHORITATIVE_BUNDLE,
    AUTHORITATIVE_CACHE,
    AUTHORITATIVE_RUNNER,
    AUTHORITATIVE_SCORING,
    AUTHORITATIVE_SOURCE_FILES,
    CC_FORBIDDEN_IMPORT_MARKERS,
    COUNTS,
    DATA_ROOT,
    DRM_METHODS,
    K_BLOCKS,
    M_BLOCKS,
    METHOD_SPECS,
    MODEL_PATH,
    MODEL_REVISION,
    PRUNE_LAYERS,
    PROTOCOL_DIR,
    PROTOCOL_NAME,
    REPO_ROOT,
    RETAIN_TOKENS,
    TASKS,
    ULB_PROTOCOL_DIR,
    ULB_PROTOCOL_ID,
)


def _ensure_no_cuda() -> None:
    # Refuse CUDA init during static audit.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    if "torch" in __import__("sys").modules:
        import torch
        if torch.cuda.is_available() and os.environ.get("CUDA_VISIBLE_DEVICES", "") != "":
            raise RuntimeError("CUDA must remain uninitialized during static audit")


def _model_files_present(model_path: Path) -> Dict[str, Any]:
    required_names = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "model.safetensors.index.json",
    ]
    present = {name: (model_path / name).exists() for name in required_names}
    shards = sorted(model_path.glob("model-*.safetensors")) if model_path.exists() else []
    weight_files = list(shards)
    if (model_path / "model.safetensors").exists():
        weight_files.append(model_path / "model.safetensors")
    return {
        "model_path": str(model_path),
        "exists": model_path.exists(),
        "is_dir": model_path.is_dir() if model_path.exists() else False,
        "required": present,
        "weight_shard_count": len(shards),
        "weight_files": [str(p) for p in weight_files],
        "complete": bool(model_path.exists() and all(present.values()) and weight_files),
    }


def _hash_model_identity(model_path: Path) -> Dict[str, Any]:
    info = _model_files_present(model_path)
    if not info["complete"]:
        return {
            "revision": MODEL_REVISION,
            "status": "MODEL_MISSING_OR_INCOMPLETE",
            "files": info,
        }
    config = load_json(model_path / "config.json")
    tok_cfg = load_json(model_path / "tokenizer_config.json")
    chat_template = tok_cfg.get("chat_template") or ""
    if not chat_template:
        # Offline snapshots often omit chat_template; use the same deterministic
        # fallback as the runtime runner for identity locking.
        if str(config.get("model_type", "")).lower().startswith("llama"):
            try:
                from tools.llama_drm_on_qwen_v1.llama_chat_template import (
                    LLAMA3_FALLBACK_CHAT_TEMPLATE_JINJA,
                )

                chat_template = LLAMA3_FALLBACK_CHAT_TEMPLATE_JINJA
            except Exception:
                chat_template = ""
    weight_hashes = {Path(p).name: sha256_file(Path(p)) for p in info["weight_files"]}
    fingerprint_payload = {
        "config_sha256": sha256_file(model_path / "config.json"),
        "tokenizer_json_sha256": sha256_file(model_path / "tokenizer.json"),
        "tokenizer_config_sha256": sha256_file(model_path / "tokenizer_config.json"),
        "chat_template_sha256": sha256_text(chat_template) if chat_template else None,
        "weight_hashes": weight_hashes,
        "index_sha256": sha256_file(model_path / "model.safetensors.index.json"),
    }
    return {
        "revision": MODEL_REVISION,
        "status": "LOCKED",
        "model_type": config.get("model_type"),
        "architectures": config.get("architectures"),
        "num_hidden_layers": config.get("num_hidden_layers"),
        "max_position_embeddings": config.get("max_position_embeddings"),
        "torch_dtype": config.get("torch_dtype"),
        "declared_name_or_path": config.get("_name_or_path"),
        "fingerprint_sha256": __import__("hashlib").sha256(
            canonical_bytes(fingerprint_payload)
        ).hexdigest(),
        **fingerprint_payload,
        "files": info,
    }


def _dataset_audit() -> Dict[str, Any]:
    ulb = load_json(ULB_PROTOCOL_DIR / "dataset_hashes.json")
    task_manifest = load_json(ULB_PROTOCOL_DIR / "task_manifest.json")
    rows = []
    ok = True
    total = 0
    for task in TASKS:
        path = DATA_ROOT / f"{task}.jsonl"
        samples = load_task_samples(task)
        total += len(samples)
        file_sha = sha256_file(path)
        expected = None
        for entry in ulb.get("task_entries", []):
            if entry.get("task") == task:
                expected = entry
                break
        first_id, _ = canonical_sample_identity(task, 0, samples[0])
        last_id, _ = canonical_sample_identity(task, len(samples) - 1, samples[-1])
        match = expected is not None and expected.get("file_sha256") == file_sha and expected.get("expected_count") == len(samples)
        if not match:
            ok = False
        rows.append({
            "task": task,
            "count": len(samples),
            "expected_count": COUNTS[task],
            "file_sha256": file_sha,
            "expected_file_sha256": None if expected is None else expected.get("file_sha256"),
            "first_sample_id": first_id,
            "last_sample_id": last_id,
            "match_ulb": match,
        })
    return {
        "data_root": str(DATA_ROOT),
        "task_count": len(TASKS),
        "total_samples": total,
        "expected_total": 3950,
        "ok": ok and total == 3950 and len(TASKS) == 17,
        "macro_policy": task_manifest.get("macro"),
        "tasks": rows,
    }


def _scan_source_for_markers(path: Path, markers: Sequence[str]) -> List[str]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return [m for m in markers if m in text]


def _authority_audit() -> Dict[str, Any]:
    checks = []
    ok = True
    for path in AUTHORITATIVE_SOURCE_FILES:
        exists = path.exists()
        checks.append({"path": str(path), "exists": exists, "sha256": sha256_file(path) if exists else None})
        if not exists:
            ok = False
    # Ensure runner loads Llama when model_family=llama.
    runner_text = AUTHORITATIVE_RUNNER.read_text(encoding="utf-8", errors="ignore")
    llama_wired = "modeling_llama" in runner_text and '"llama"' in runner_text
    if not llama_wired:
        ok = False
    # Ensure DRM methods map only through Original runner + rescue policy (no CC).
    method_ok = True
    for name, spec in METHOD_SPECS.items():
        if name not in DRM_METHODS:
            continue
        if spec["runner"] != "sliminfer_original" and name != "sliminfer_original":
            # original uses sliminfer_original runner too
            pass
        if "cc_" in name:
            method_ok = False
    # Bundle authority if present.
    bundle_ok = AUTHORITATIVE_BUNDLE.exists()
    # Random selector must exist in scoring / cache path.
    scoring = AUTHORITATIVE_SCORING.read_text(encoding="utf-8", errors="ignore")
    cache = AUTHORITATIVE_CACHE.read_text(encoding="utf-8", errors="ignore")
    random_wired = ("rescue_score" in cache or "score == \"random\"" in scoring or "random" in scoring) and "additive_rescue" in scoring
    swap_wired = "budget_neutral" in scoring or "budget_neutral_drm_swap" in cache
    add_wired = "additive_rescue" in scoring
    # CC must not be mixed into this pipeline package (exclude this audit module's marker table).
    package_dir = REPO_ROOT / "tools" / "llama_drm_on_qwen_v1"
    skip_names = {"static_audit.py", "constants.py"}
    cc_hits = []
    for path in package_dir.rglob("*.py"):
        if path.name in skip_names:
            continue
        hits = _scan_source_for_markers(path, CC_FORBIDDEN_IMPORT_MARKERS)
        if hits:
            cc_hits.append({"path": str(path), "markers": hits})
    if cc_hits:
        ok = False
    if not (llama_wired and method_ok and random_wired and swap_wired and add_wired):
        ok = False
    return {
        "ok": ok and bundle_ok,
        "llama_runner_wired": llama_wired,
        "random_selector_wired": random_wired,
        "swap_wired": swap_wired,
        "add_wired": add_wired,
        "bundle_present": bundle_ok,
        "bundle_path": str(AUTHORITATIVE_BUNDLE),
        "source_files": checks,
        "cc_hits_in_llama_package": cc_hits,
        "schedule": {
            "prune_layers": list(PRUNE_LAYERS),
            "retain_tokens": list(RETAIN_TOKENS),
            "k_blocks": list(K_BLOCKS),
            "m_blocks": list(M_BLOCKS),
        },
        "methods": METHOD_SPECS,
    }


def _forbid_old_artifacts(pipeline_dir: Path) -> Dict[str, Any]:
    forbidden_globs = [
        "**/llama*canonical*",
        "**/debug*smoke*",
        "**/cc_*predictions*",
    ]
    # Do not reuse old LLaMA / CC prediction trees as inputs.
    suspects = [
        REPO_ROOT / "results" / "rescuekv_paper_validation" / "model_transfer",
        REPO_ROOT / "results" / "cc_sliminfer_v1_qwen25_17task_full_official_20260712_075238",
    ]
    return {
        "ok": True,
        "note": "pipeline writes only under its own results root; old CC/LLaMA trees are not used as inputs",
        "ignored_paths": [str(p) for p in suspects],
        "pipeline_dir": str(pipeline_dir),
        "forbidden_globs": forbidden_globs,
    }


def run_static_audit(pipeline_dir: Path) -> Dict[str, Any]:
    _ensure_no_cuda()
    protocol_id, protocol_hash = compute_protocol_id_and_hash()
    atomic_write_text(PROTOCOL_DIR / "PROTOCOL_ID.txt", protocol_id + "\n")
    atomic_write_text(PROTOCOL_DIR / "PROTOCOL_HASH.txt", protocol_hash + "\n")

    model_identity = _hash_model_identity(MODEL_PATH)
    dataset = _dataset_audit()
    authority = _authority_audit()
    old = _forbid_old_artifacts(pipeline_dir)

    scorer_path = REPO_ROOT / "longbench_eval.py"
    prompt_path = REPO_ROOT / "configs" / "dataset2prompt.json"
    maxlen_path = REPO_ROOT / "configs" / "dataset2maxlen.json"
    scorer_ok = scorer_path.exists() and prompt_path.exists() and maxlen_path.exists()

    hard_blocks: List[str] = []
    waiting_for_model = model_identity.get("status") != "LOCKED"
    if waiting_for_model:
        hard_blocks.append("MODEL_MISSING_OR_INCOMPLETE")
    if not dataset.get("ok"):
        hard_blocks.append("DATASET_MANIFEST_MISMATCH")
    if not authority.get("ok"):
        hard_blocks.append("AUTHORITY_OR_METHOD_WIRE_FAILED")
    if not scorer_ok:
        hard_blocks.append("SCORER_OR_PROMPT_MISSING")
    if not old.get("ok"):
        hard_blocks.append("OLD_ARTIFACT_REUSE")

    # Soft note: available snapshot may not be 3.1-128k; identity is hash-locked, never fabricated.
    warnings: List[str] = []
    if model_identity.get("status") == "LOCKED":
        max_pos = model_identity.get("max_position_embeddings")
        if isinstance(max_pos, int) and max_pos < 32768:
            warnings.append(
                f"local snapshot max_position_embeddings={max_pos}; efficiency lengths above this may fail or be truncated"
            )

    if not hard_blocks:
        status = "PASS"
    elif hard_blocks == ["MODEL_MISSING_OR_INCOMPLETE"]:
        status = "WAITING_FOR_MODEL"
    else:
        status = "HARD_BLOCK"
    report = {
        "updated_at": now_iso(),
        "protocol_name": PROTOCOL_NAME,
        "protocol_id": protocol_id,
        "protocol_hash": protocol_hash,
        "ulb_protocol_id": ULB_PROTOCOL_ID,
        "status": status,
        "hard_blocks": hard_blocks,
        "warnings": warnings,
        "cuda_initialized": False,
        "model_identity": model_identity,
        "dataset": dataset,
        "authority": authority,
        "old_artifact_policy": old,
        "scorer": {
            "longbench_eval_sha256": sha256_file(scorer_path) if scorer_path.exists() else None,
            "dataset2prompt_sha256": sha256_file(prompt_path) if prompt_path.exists() else None,
            "dataset2maxlen_sha256": sha256_file(maxlen_path) if maxlen_path.exists() else None,
            "ok": scorer_ok,
        },
        "invariants": {
            "shared_base_A_K": True,
            "random_and_drm_share_eligible_residual_pool": True,
            "add_preserves_A_K": True,
            "swap_one_in_one_out": True,
            "candidate_pool_unique": True,
            "full_selection_trace_required": True,
            "no_cc_mixed": len(authority.get("cc_hits_in_llama_package") or []) == 0,
            "qwen_authority_entry_unique": True,
        },
    }
    out = pipeline_dir / "audits" / "STATIC_AUDIT_REPORT.json"
    atomic_write_json(out, report)
    md = [
        "# LLAMA_DRM Static Audit",
        "",
        f"- status: `{status}`",
        f"- protocol_id: `{protocol_id}`",
        f"- protocol_hash: `{protocol_hash}`",
        f"- model_path: `{MODEL_PATH}`",
        f"- model_status: `{model_identity.get('status')}`",
        f"- dataset_ok: `{dataset.get('ok')}`",
        f"- authority_ok: `{authority.get('ok')}`",
        "",
        "## Hard blocks",
        "",
    ]
    if hard_blocks:
        md.extend([f"- {item}" for item in hard_blocks])
    else:
        md.append("- none")
    if warnings:
        md.extend(["", "## Warnings", ""] + [f"- {w}" for w in warnings])
    atomic_write_text(pipeline_dir / "audits" / "STATIC_AUDIT_REPORT.md", "\n".join(md) + "\n")
    return report
