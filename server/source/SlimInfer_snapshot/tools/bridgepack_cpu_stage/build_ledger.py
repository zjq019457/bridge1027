#!/usr/bin/env python3
"""Build BridgePack EXPERIMENT_LEDGER from existing manifests + extended result dirs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .common import (
    FAIRNESS_DIR,
    OUT_ROOT,
    REPO_ROOT,
    TASKS,
    EXPECTED_COUNTS,
    ensure_cpu_only,
    normalize_method,
    read_csv,
    read_json,
    resolve_repo_path,
    sha256_file,
    sha256_path_or_glob,
    utc_now,
    write_csv,
    write_json,
    write_text,
)


EXTRA_UNITS: List[Dict[str, Any]] = [
    # Mistral 1024 (3-task gate; incomplete vs 6-task protocol)
    {
        "model": "mistral",
        "checkpoint": "mistralai/Mistral-7B-Instruct-v0.3",
        "method": "lexical_bridgepack_path_order",
        "budget": 1024,
        "final_root": "results/bridgepack_second_model_exact_gate_mistral7b_instruct_v03_20260620_055432",
        "packed_path": "results/bridgepack_second_model_exact_gate_mistral7b_instruct_v03_20260620_055432/packed_prompts_build/packed_prompts/lexical_bridgepack_path_order/budget_1024/{task}.jsonl",
        "prediction_dir": "results/bridgepack_second_model_exact_gate_mistral7b_instruct_v03_20260620_055432/predictions/lexical_bridgepack_path_order",
        "score_dir": "results/bridgepack_second_model_exact_gate_mistral7b_instruct_v03_20260620_055432/scores/lexical_bridgepack_path_order",
        "config_path": "results/bridgepack_second_model_exact_gate_mistral7b_instruct_v03_20260620_055432/config.json",
        "notes": "3-task second-model gate (not full 6-task)",
    },
    {
        "model": "mistral",
        "checkpoint": "mistralai/Mistral-7B-Instruct-v0.3",
        "method": "bm25_chunk_source_order",
        "budget": 1024,
        "final_root": "results/bridgepack_second_model_exact_gate_mistral7b_instruct_v03_20260620_055432",
        "packed_path": "results/bridgepack_second_model_exact_gate_mistral7b_instruct_v03_20260620_055432/packed_prompts_build/packed_prompts/bm25_chunk_source_order/budget_1024/{task}.jsonl",
        "prediction_dir": "results/bridgepack_second_model_exact_gate_mistral7b_instruct_v03_20260620_055432/predictions/bm25_chunk_source_order",
        "score_dir": "results/bridgepack_second_model_exact_gate_mistral7b_instruct_v03_20260620_055432/scores/bm25_chunk_source_order",
        "config_path": "results/bridgepack_second_model_exact_gate_mistral7b_instruct_v03_20260620_055432/config.json",
        "notes": "3-task second-model gate",
    },
    {
        "model": "mistral",
        "checkpoint": "mistralai/Mistral-7B-Instruct-v0.3",
        "method": "tfidf_chunk_source_order",
        "budget": 1024,
        "final_root": "results/bridgepack_second_model_exact_gate_mistral7b_instruct_v03_20260620_055432",
        "packed_path": "results/bridgepack_second_model_exact_gate_mistral7b_instruct_v03_20260620_055432/packed_prompts_build/packed_prompts/tfidf_chunk_source_order/budget_1024/{task}.jsonl",
        "prediction_dir": "results/bridgepack_second_model_exact_gate_mistral7b_instruct_v03_20260620_055432/predictions/tfidf_chunk_source_order",
        "score_dir": "results/bridgepack_second_model_exact_gate_mistral7b_instruct_v03_20260620_055432/scores/tfidf_chunk_source_order",
        "config_path": "results/bridgepack_second_model_exact_gate_mistral7b_instruct_v03_20260620_055432/config.json",
        "notes": "3-task second-model gate",
    },
    {
        "model": "mistral",
        "checkpoint": "mistralai/Mistral-7B-Instruct-v0.3",
        "method": "bridge_neighbor32_path_order",
        "budget": 1024,
        "final_root": "results/bridgepack_second_model_exact_gate_mistral7b_instruct_v03_20260620_055432",
        "packed_path": "results/bridgepack_second_model_exact_gate_mistral7b_instruct_v03_20260620_055432/packed_prompts_build/packed_prompts/bridge_neighbor32_path_order/budget_1024/{task}.jsonl",
        "prediction_dir": "results/bridgepack_second_model_exact_gate_mistral7b_instruct_v03_20260620_055432/predictions/bridge_neighbor32_path_order",
        "score_dir": "results/bridgepack_second_model_exact_gate_mistral7b_instruct_v03_20260620_055432/scores/bridge_neighbor32_path_order",
        "config_path": "results/bridgepack_second_model_exact_gate_mistral7b_instruct_v03_20260620_055432/config.json",
        "notes": "3-task second-model gate",
    },
    {
        "model": "mistral",
        "checkpoint": "mistralai/Mistral-7B-Instruct-v0.3",
        "method": "span_only_source_order",
        "budget": 1024,
        "final_root": "results/bridgepack_second_model_exact_gate_mistral7b_instruct_v03_20260620_055432",
        "packed_path": "results/bridgepack_second_model_exact_gate_mistral7b_instruct_v03_20260620_055432/packed_prompts_build/packed_prompts/span_only_source_order/budget_1024/{task}.jsonl",
        "prediction_dir": "results/bridgepack_second_model_exact_gate_mistral7b_instruct_v03_20260620_055432/predictions/span_only_source_order",
        "score_dir": "results/bridgepack_second_model_exact_gate_mistral7b_instruct_v03_20260620_055432/scores/span_only_source_order",
        "config_path": "results/bridgepack_second_model_exact_gate_mistral7b_instruct_v03_20260620_055432/config.json",
        "notes": "3-task second-model gate",
    },
    # Qwen 2048
    {
        "model": "qwen",
        "checkpoint": "Qwen2.5-7B-Instruct",
        "method": "lexical_bridgepack_path_order",
        "budget": 2048,
        "final_root": "results/bridgepack_exact_budget_ablation_20260619_231854/budget_2048",
        "packed_path": "results/bridgepack_exact_budget_ablation_20260619_231854/budget_2048/packed_prompts_build/packed_prompts/lexical_bridgepack_path_order/budget_2048/{task}.jsonl",
        "prediction_dir": "results/bridgepack_exact_budget_ablation_20260619_231854/budget_2048/predictions/lexical_bridgepack_path_order",
        "score_dir": "results/bridgepack_exact_budget_ablation_20260619_231854/budget_2048/scores/lexical_bridgepack_path_order",
        "config_path": "results/bridgepack_exact_budget_ablation_20260619_231854/budget_2048/config.json",
        "notes": "budget ablation 2048",
    },
    {
        "model": "qwen",
        "checkpoint": "Qwen2.5-7B-Instruct",
        "method": "bm25_chunk_source_order",
        "budget": 2048,
        "final_root": "results/bridgepack_exact_budget_ablation_20260619_231854/budget_2048",
        "packed_path": "results/bridgepack_exact_budget_ablation_20260619_231854/budget_2048/packed_prompts_build/packed_prompts/bm25_chunk_source_order/budget_2048/{task}.jsonl",
        "prediction_dir": "results/bridgepack_exact_budget_ablation_20260619_231854/budget_2048/predictions/bm25_chunk_source_order",
        "score_dir": "results/bridgepack_exact_budget_ablation_20260619_231854/budget_2048/scores/bm25_chunk_source_order",
        "config_path": "results/bridgepack_exact_budget_ablation_20260619_231854/budget_2048/config.json",
        "notes": "budget ablation 2048",
    },
    {
        "model": "qwen",
        "checkpoint": "Qwen2.5-7B-Instruct",
        "method": "tfidf_chunk_source_order",
        "budget": 2048,
        "final_root": "results/bridgepack_exact_budget_ablation_20260619_231854/budget_2048",
        "packed_path": "results/bridgepack_exact_budget_ablation_20260619_231854/budget_2048/packed_prompts_build/packed_prompts/tfidf_chunk_source_order/budget_2048/{task}.jsonl",
        "prediction_dir": "results/bridgepack_exact_budget_ablation_20260619_231854/budget_2048/predictions/tfidf_chunk_source_order",
        "score_dir": "results/bridgepack_exact_budget_ablation_20260619_231854/budget_2048/scores/tfidf_chunk_source_order",
        "config_path": "results/bridgepack_exact_budget_ablation_20260619_231854/budget_2048/config.json",
        "notes": "budget ablation 2048",
    },
    {
        "model": "qwen",
        "checkpoint": "Qwen2.5-7B-Instruct",
        "method": "bridge_neighbor32_path_order",
        "budget": 2048,
        "final_root": "results/bridgepack_exact_budget_ablation_20260619_231854/budget_2048",
        "packed_path": "results/bridgepack_exact_budget_ablation_20260619_231854/budget_2048/packed_prompts_build/packed_prompts/bridge_neighbor32_path_order/budget_2048/{task}.jsonl",
        "prediction_dir": "results/bridgepack_exact_budget_ablation_20260619_231854/budget_2048/predictions/bridge_neighbor32_path_order",
        "score_dir": "results/bridgepack_exact_budget_ablation_20260619_231854/budget_2048/scores/bridge_neighbor32_path_order",
        "config_path": "results/bridgepack_exact_budget_ablation_20260619_231854/budget_2048/config.json",
        "notes": "budget ablation 2048",
    },
    {
        "model": "qwen",
        "checkpoint": "Qwen2.5-7B-Instruct",
        "method": "span_only_source_order",
        "budget": 2048,
        "final_root": "results/bridgepack_exact_budget_ablation_20260619_231854/budget_2048",
        "packed_path": "results/bridgepack_exact_budget_ablation_20260619_231854/budget_2048/packed_prompts_build/packed_prompts/span_only_source_order/budget_2048/{task}.jsonl",
        "prediction_dir": "results/bridgepack_exact_budget_ablation_20260619_231854/budget_2048/predictions/span_only_source_order",
        "score_dir": "results/bridgepack_exact_budget_ablation_20260619_231854/budget_2048/scores/span_only_source_order",
        "config_path": "results/bridgepack_exact_budget_ablation_20260619_231854/budget_2048/config.json",
        "notes": "budget ablation 2048",
    },
]


CHECKPOINT_BY_MODEL = {
    "qwen": "/home/yu-bai/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28",
    "llama3": "meta-llama/Meta-Llama-3-8B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "llama2": "meta-llama/Llama-2-7b-chat-hf",
}


def _count_prediction_samples(pred_dir: Path) -> int:
    if not pred_dir.exists():
        return 0
    total = 0
    for task in TASKS:
        p = pred_dir / f"{task}.jsonl"
        if p.exists():
            total += sum(1 for line in p.open(encoding="utf-8") if line.strip())
    # flat layout fallback
    if total == 0:
        for p in pred_dir.glob("*.jsonl"):
            total += sum(1 for line in p.open(encoding="utf-8") if line.strip())
    return total


def _count_packed_samples(packed_path: str) -> int:
    if not packed_path:
        return 0
    if "{task}" in packed_path:
        total = 0
        for task in TASKS:
            p = resolve_repo_path(packed_path.replace("{task}", task))
            if p.exists():
                total += sum(1 for line in p.open(encoding="utf-8") if line.strip())
        return total
    p = resolve_repo_path(packed_path)
    if not p.exists():
        return 0
    return sum(1 for line in p.open(encoding="utf-8") if line.strip())


def _score_present(score_dir: Path) -> bool:
    if not score_dir.exists():
        return False
    return any(score_dir.rglob("*.json"))


def _infer_status(
    *,
    packed_n: int,
    pred_n: int,
    has_scores: bool,
    exact_budget_pass: Optional[bool],
    expected: int = 1150,
    notes: str = "",
) -> str:
    if "GPU_PENDING" in notes:
        return "GPU_PENDING"
    if packed_n == 0 and pred_n == 0 and not has_scores:
        return "MISSING"
    if pred_n < expected or packed_n < expected or not has_scores:
        if packed_n > 0 or pred_n > 0 or has_scores:
            return "PARTIAL"
        return "MISSING"
    if exact_budget_pass is False:
        return "BLOCKED"
    return "COMPLETE"


def _load_prompt_wrapper() -> Dict[str, str]:
    path = REPO_ROOT / "configs" / "dataset2prompt.json"
    return read_json(path) if path.exists() else {}


def _load_maxlen() -> Dict[str, int]:
    path = REPO_ROOT / "configs" / "dataset2maxlen.json"
    raw = read_json(path) if path.exists() else {}
    return {k: int(v) for k, v in raw.items()}


def _unit_from_manifest_row(row: Dict[str, str], wrappers: Dict[str, str], maxlen: Dict[str, int]) -> Dict[str, Any]:
    model = row["model"]
    method = row["method"]
    budget = int(row["budget"])
    packed_path = row.get("packed_path", "")
    pred_dir = resolve_repo_path(row.get("prediction_dir", ""))
    score_dir = resolve_repo_path(row.get("score_dir", ""))
    config_path = row.get("config_or_provenance_path") or row.get("config_path") or ""
    packed_n = _count_packed_samples(packed_path)
    pred_n = _count_prediction_samples(pred_dir)
    has_scores = _score_present(score_dir)
    # Fairness audit already verified 0 over-budget for 60 cells.
    exact_pass = True if row.get("all_required_paths_exist", "").lower() == "true" else None
    status = _infer_status(
        packed_n=packed_n,
        pred_n=pred_n,
        has_scores=has_scores,
        exact_budget_pass=exact_pass,
        expected=1150,
    )
    provenance_complete = bool(
        row.get("all_required_paths_exist", "").lower() == "true"
        and config_path
        and resolve_repo_path(config_path).exists()
    )
    return {
        "model": model,
        "checkpoint": CHECKPOINT_BY_MODEL.get(model, ""),
        "method": method,
        "method_normalized": normalize_method(method),
        "budget": budget,
        "tasks": ",".join(TASKS),
        "n_samples_expected": 1150,
        "n_packed": packed_n,
        "n_predictions": pred_n,
        "packed_artifact": packed_path,
        "per_sample_prediction": row.get("prediction_dir", ""),
        "per_sample_score": "paper/aaai2027_bridgepack_exact/submission_fairness_audit_20260701/per_sample_scores.csv",
        "tokenizer": CHECKPOINT_BY_MODEL.get(model, ""),
        "prompt_wrapper": "configs/dataset2prompt.json",
        "decoding_config": json.dumps(
            {
                "temperature": 0.0 if "qwen" in model or "llama" in model else None,
                "top_p": None,
                "max_new_tokens_by_task": {t: maxlen.get(t) for t in TASKS},
                "dtype": "fp16",
            },
            ensure_ascii=False,
        ),
        "exact_budget_pass": exact_pass,
        "full_provenance": provenance_complete,
        "status": status,
        "config_path": config_path,
        "final_root": row.get("final_root", ""),
        "audit_path": row.get("audit_path", ""),
        "packed_sha256": sha256_path_or_glob(packed_path),
        "config_sha256": sha256_file(resolve_repo_path(config_path)) if config_path else None,
        "source_manifest": "submission_fairness_audit_20260701/result_file_manifest.csv",
        "notes": "fairness-audit 60-cell unit; DO_NOT_RERUN",
    }


def _unit_from_extra(extra: Dict[str, Any], maxlen: Dict[str, int]) -> Dict[str, Any]:
    packed_path = extra["packed_path"]
    pred_dir = resolve_repo_path(extra["prediction_dir"])
    score_dir = resolve_repo_path(extra["score_dir"])
    packed_n = _count_packed_samples(packed_path)
    pred_n = _count_prediction_samples(pred_dir)
    has_scores = _score_present(score_dir)
    expected = 1150 if extra["model"] != "mistral" else 96
    # Mistral gate is known partial (3 tasks × ~32)
    if extra["model"] == "mistral":
        status = "PARTIAL"
    else:
        status = _infer_status(
            packed_n=packed_n,
            pred_n=pred_n,
            has_scores=has_scores,
            exact_budget_pass=True if packed_n > 0 else None,
            expected=expected,
            notes=extra.get("notes", ""),
        )
    cfg = extra.get("config_path", "")
    return {
        "model": extra["model"],
        "checkpoint": extra.get("checkpoint") or CHECKPOINT_BY_MODEL.get(extra["model"], ""),
        "method": extra["method"],
        "method_normalized": normalize_method(extra["method"]),
        "budget": extra["budget"],
        "tasks": ",".join(TASKS),
        "n_samples_expected": expected,
        "n_packed": packed_n,
        "n_predictions": pred_n,
        "packed_artifact": packed_path,
        "per_sample_prediction": extra["prediction_dir"],
        "per_sample_score": extra["score_dir"],
        "tokenizer": extra.get("checkpoint") or CHECKPOINT_BY_MODEL.get(extra["model"], ""),
        "prompt_wrapper": "configs/dataset2prompt.json",
        "decoding_config": json.dumps({"max_new_tokens_by_task": {t: maxlen.get(t) for t in TASKS}}),
        "exact_budget_pass": True if packed_n > 0 else None,
        "full_provenance": bool(cfg and resolve_repo_path(cfg).exists()),
        "status": status,
        "config_path": cfg,
        "final_root": extra.get("final_root", ""),
        "audit_path": "",
        "packed_sha256": sha256_path_or_glob(packed_path),
        "config_sha256": sha256_file(resolve_repo_path(cfg)) if cfg else None,
        "source_manifest": "EXTRA_UNITS",
        "notes": extra.get("notes", ""),
    }


def _discover_llama2_units(maxlen: Dict[str, int]) -> List[Dict[str, Any]]:
    root = REPO_ROOT / "results" / "llama2_7b_chat_exact_budget_full_experiment_20260625_003442"
    units: List[Dict[str, Any]] = []
    if not root.exists():
        return units
    # Prefer reports CSV if present
    scores_csv = root / "06_reports" / "llama2_finalqa_scores.csv"
    score_rows = read_csv(scores_csv) if scores_csv.exists() else []
    methods_budgets = set()
    for row in score_rows:
        methods_budgets.add((row.get("method") or row.get("variant") or "", int(float(row.get("budget", 0) or 0))))
    # Discover from directory layout
    prompt_root = root / "03_prompt_generation"
    qa_root = root / "04_finalqa"
    discovered = set()
    if qa_root.exists():
        for method_dir in qa_root.iterdir():
            if not method_dir.is_dir():
                continue
            for budget_dir in method_dir.iterdir():
                if not budget_dir.is_dir():
                    continue
                name = budget_dir.name
                if not name.startswith("budget_"):
                    # sometimes nested differently
                    continue
                budget = int(name.split("_")[1])
                discovered.add((method_dir.name, budget))
    # Flat alternative
    if not discovered and (root / "04_finalqa").exists():
        for p in (root / "04_finalqa").rglob("run_status.json"):
            parts = p.relative_to(root / "04_finalqa").parts
            if len(parts) >= 2:
                method = parts[0]
                budget_part = parts[1]
                if budget_part.startswith("budget_"):
                    discovered.add((method, int(budget_part.split("_")[1])))
    for method, budget in sorted(discovered):
        packed = root / "03_prompt_generation" / method / f"budget_{budget}" / "llama2_packed_prompts.jsonl"
        if not packed.exists():
            packed = root / "03_prompt_generation" / method / f"budget_{budget}" / "packed_prompts.jsonl"
        pred = root / "04_finalqa" / method / f"budget_{budget}" / "predictions" / method
        score = root / "04_finalqa" / method / f"budget_{budget}" / "scores" / method
        packed_rel = str(packed.relative_to(REPO_ROOT)) if packed.exists() else ""
        units.append(
            {
                "model": "llama2",
                "checkpoint": CHECKPOINT_BY_MODEL["llama2"],
                "method": method,
                "method_normalized": normalize_method(method),
                "budget": budget,
                "tasks": ",".join(TASKS),
                "n_samples_expected": 1150,
                "n_packed": _count_packed_samples(packed_rel) if packed_rel else 0,
                "n_predictions": _count_prediction_samples(pred) if pred.exists() else 0,
                "packed_artifact": packed_rel,
                "per_sample_prediction": str(pred.relative_to(REPO_ROOT)) if pred.exists() else "",
                "per_sample_score": str(score.relative_to(REPO_ROOT)) if score.exists() else "",
                "tokenizer": CHECKPOINT_BY_MODEL["llama2"],
                "prompt_wrapper": "configs/dataset2prompt.json",
                "decoding_config": json.dumps({"max_new_tokens_by_task": {t: maxlen.get(t) for t in TASKS}}),
                "exact_budget_pass": True if packed.exists() else None,
                "full_provenance": True,
                "status": _infer_status(
                    packed_n=_count_packed_samples(packed_rel) if packed_rel else 0,
                    pred_n=_count_prediction_samples(pred) if pred.exists() else 0,
                    has_scores=_score_present(score),
                    exact_budget_pass=True if packed.exists() else None,
                ),
                "config_path": str((root / "README.md").relative_to(REPO_ROOT)),
                "final_root": str(root.relative_to(REPO_ROOT)),
                "audit_path": str((root / "06_reports" / "llama2_packed_prompt_audit_summary.csv").relative_to(REPO_ROOT))
                if (root / "06_reports" / "llama2_packed_prompt_audit_summary.csv").exists()
                else "",
                "packed_sha256": sha256_file(packed) if packed.exists() else None,
                "config_sha256": None,
                "source_manifest": "llama2_7b_chat_exact_budget_full_experiment",
                "notes": "llama2 extension; prefer reuse over rerun",
            }
        )
    if not units:
        units.append(
            {
                "model": "llama2",
                "checkpoint": CHECKPOINT_BY_MODEL["llama2"],
                "method": "ALL",
                "method_normalized": "ALL",
                "budget": 1024,
                "tasks": ",".join(TASKS),
                "n_samples_expected": 1150,
                "n_packed": 0,
                "n_predictions": 0,
                "packed_artifact": "",
                "per_sample_prediction": "",
                "per_sample_score": str((root / "06_reports" / "llama2_finalqa_scores.csv").relative_to(REPO_ROOT))
                if scores_csv.exists()
                else "",
                "tokenizer": CHECKPOINT_BY_MODEL["llama2"],
                "prompt_wrapper": "configs/dataset2prompt.json",
                "decoding_config": "{}",
                "exact_budget_pass": None,
                "full_provenance": False,
                "status": "PARTIAL" if scores_csv.exists() else "MISSING",
                "config_path": str(root.relative_to(REPO_ROOT)),
                "final_root": str(root.relative_to(REPO_ROOT)),
                "audit_path": "",
                "packed_sha256": None,
                "config_sha256": None,
                "source_manifest": "llama2_reports_only",
                "notes": "directory layout atypical; see 06_reports",
            }
        )
    return units


def _neighbor_units(maxlen: Dict[str, int]) -> List[Dict[str, Any]]:
    root = REPO_ROOT / "results" / "bridgepack_neighbor_path_isolation_20260701_080809"
    units: List[Dict[str, Any]] = []
    if not root.exists():
        return units
    packed_root = root / "packed_prompts_build" / "packed_prompts"
    if not packed_root.exists():
        return [
            {
                "model": "qwen",
                "checkpoint": CHECKPOINT_BY_MODEL["qwen"],
                "method": "neighbor_path_isolation",
                "method_normalized": "neighbor_path_isolation",
                "budget": 1024,
                "tasks": ",".join(TASKS),
                "n_samples_expected": 1150,
                "n_packed": 0,
                "n_predictions": 0,
                "packed_artifact": str(root.relative_to(REPO_ROOT)),
                "per_sample_prediction": "",
                "per_sample_score": "",
                "tokenizer": CHECKPOINT_BY_MODEL["qwen"],
                "prompt_wrapper": "configs/dataset2prompt.json",
                "decoding_config": "{}",
                "exact_budget_pass": None,
                "full_provenance": True,
                "status": "PARTIAL",
                "config_path": str((root / "config.json").relative_to(REPO_ROOT)) if (root / "config.json").exists() else "",
                "final_root": str(root.relative_to(REPO_ROOT)),
                "audit_path": str((root / "packing_audit.json").relative_to(REPO_ROOT))
                if (root / "packing_audit.json").exists()
                else "",
                "packed_sha256": None,
                "config_sha256": sha256_file(root / "config.json") if (root / "config.json").exists() else None,
                "source_manifest": "neighbor_path_isolation",
                "notes": "Qwen-1024 neighbor ablation; reuse packed artifacts",
            }
        ]
    for variant_dir in sorted(packed_root.iterdir()):
        if not variant_dir.is_dir():
            continue
        budget_dirs = list(variant_dir.glob("budget_*"))
        for bdir in budget_dirs or [variant_dir]:
            budget = 1024
            if bdir.name.startswith("budget_"):
                budget = int(bdir.name.split("_")[1])
                packed_path = str((bdir / "{task}.jsonl").relative_to(REPO_ROOT)).replace("%7Btask%7D", "{task}")
                # pathlib may not keep braces; reconstruct
                packed_path = str(bdir.relative_to(REPO_ROOT)) + "/{task}.jsonl"
            else:
                packed_path = str(variant_dir.relative_to(REPO_ROOT)) + "/{task}.jsonl"
            packed_n = _count_packed_samples(packed_path)
            qa = root / "final_qa_cells" / variant_dir.name
            pred = qa / "predictions" / variant_dir.name if qa.exists() else Path()
            score = qa / "scores" / variant_dir.name if qa.exists() else Path()
            units.append(
                {
                    "model": "qwen",
                    "checkpoint": CHECKPOINT_BY_MODEL["qwen"],
                    "method": variant_dir.name,
                    "method_normalized": normalize_method(variant_dir.name),
                    "budget": budget,
                    "tasks": ",".join(TASKS),
                    "n_samples_expected": 1150,
                    "n_packed": packed_n,
                    "n_predictions": _count_prediction_samples(pred) if pred and pred.exists() else 0,
                    "packed_artifact": packed_path,
                    "per_sample_prediction": str(pred.relative_to(REPO_ROOT)) if pred and pred.exists() else "",
                    "per_sample_score": str(score.relative_to(REPO_ROOT)) if score and score.exists() else "",
                    "tokenizer": CHECKPOINT_BY_MODEL["qwen"],
                    "prompt_wrapper": "configs/dataset2prompt.json",
                    "decoding_config": json.dumps({"max_new_tokens_by_task": {t: maxlen.get(t) for t in TASKS}}),
                    "exact_budget_pass": True if packed_n > 0 else None,
                    "full_provenance": True,
                    "status": _infer_status(
                        packed_n=packed_n,
                        pred_n=_count_prediction_samples(pred) if pred and pred.exists() else 0,
                        has_scores=_score_present(score) if score else False,
                        exact_budget_pass=True if packed_n > 0 else None,
                    ),
                    "config_path": str((root / "config.json").relative_to(REPO_ROOT)) if (root / "config.json").exists() else "",
                    "final_root": str(root.relative_to(REPO_ROOT)),
                    "audit_path": str((root / "packing_audit.json").relative_to(REPO_ROOT))
                    if (root / "packing_audit.json").exists()
                    else "",
                    "packed_sha256": sha256_path_or_glob(packed_path),
                    "config_sha256": sha256_file(root / "config.json") if (root / "config.json").exists() else None,
                    "source_manifest": "neighbor_path_isolation",
                    "notes": "Qwen-1024 neighbor/path isolation; DO_NOT_RERUN if COMPLETE",
                }
            )
    return units


def _external_compression_units() -> List[Dict[str, Any]]:
    units = []
    for name, status, notes in [
        (
            "results/bridgepack_official_llmlingua_audit10_20260706_012314",
            "PARTIAL",
            "LLMLingua official interface audit (10-sample); GPU_PENDING for full QA",
        ),
        (
            "results/bridgepack_official_llmlingua_qa_smoke_20260706_014713",
            "PARTIAL",
            "LLMLingua QA smoke only; not paper-grade",
        ),
        (
            "results/bridgepack_external_compression_full_packing_20260705_204655",
            "PARTIAL",
            "External compression packing; budget failures must be retained",
        ),
    ]:
        root = REPO_ROOT / name
        units.append(
            {
                "model": "qwen",
                "checkpoint": CHECKPOINT_BY_MODEL["qwen"],
                "method": "llmlingua_or_external",
                "method_normalized": "external_compression",
                "budget": 1024,
                "tasks": ",".join(TASKS),
                "n_samples_expected": 1150,
                "n_packed": 0,
                "n_predictions": 0,
                "packed_artifact": name if root.exists() else "",
                "per_sample_prediction": "",
                "per_sample_score": "",
                "tokenizer": CHECKPOINT_BY_MODEL["qwen"],
                "prompt_wrapper": "configs/dataset2prompt.json",
                "decoding_config": "{}",
                "exact_budget_pass": None,
                "full_provenance": root.exists(),
                "status": status if root.exists() else "MISSING",
                "config_path": name,
                "final_root": name,
                "audit_path": name,
                "packed_sha256": None,
                "config_sha256": None,
                "source_manifest": "external_compression",
                "notes": notes + ("" if root.exists() else " | directory missing"),
            }
        )
    # Interface-only GPU pending placeholders
    for method in [
        "matched_ablation_order_only",
        "matched_ablation_selection_only",
        "matched_ablation_score_components",
        "strong_baseline_bm25_mmr_neighbor",
        "strong_baseline_tfidf_mmr_neighbor",
        "strong_baseline_bge_mmr_neighbor_cached",
        "backfill_packing",
        "matched_actual_token_packing",
        "llmlingua_full_qa",
    ]:
        units.append(
            {
                "model": "qwen",
                "checkpoint": CHECKPOINT_BY_MODEL["qwen"],
                "method": method,
                "method_normalized": method,
                "budget": 1024,
                "tasks": ",".join(TASKS),
                "n_samples_expected": 1150,
                "n_packed": 0,
                "n_predictions": 0,
                "packed_artifact": "",
                "per_sample_prediction": "",
                "per_sample_score": "",
                "tokenizer": CHECKPOINT_BY_MODEL["qwen"],
                "prompt_wrapper": "configs/dataset2prompt.json",
                "decoding_config": "{}",
                "exact_budget_pass": None,
                "full_provenance": False,
                "status": "GPU_PENDING",
                "config_path": "",
                "final_root": "",
                "audit_path": "",
                "packed_sha256": None,
                "config_sha256": None,
                "source_manifest": "cpu_stage_interfaces",
                "notes": "Interface implemented in CPU stage; QA/full pack GPU_PENDING",
            }
        )
    return units


def build_ledger() -> Dict[str, Any]:
    ensure_cpu_only()
    wrappers = _load_prompt_wrapper()
    maxlen = _load_maxlen()
    manifest = read_csv(FAIRNESS_DIR / "result_file_manifest.csv")
    rows: List[Dict[str, Any]] = []
    for row in manifest:
        rows.append(_unit_from_manifest_row(row, wrappers, maxlen))
    for extra in EXTRA_UNITS:
        rows.append(_unit_from_extra(extra, maxlen))
    rows.extend(_discover_llama2_units(maxlen))
    rows.extend(_neighbor_units(maxlen))
    rows.extend(_external_compression_units())

    # Deduplicate by model/budget/method/final_root
    dedup = {}
    for row in rows:
        key = (row["model"], row["budget"], row["method"], row.get("final_root", ""))
        dedup[key] = row
    rows = list(dedup.values())
    rows.sort(key=lambda r: (r["model"], int(r["budget"]), r["method"], r.get("final_root", "")))

    fields = [
        "model",
        "checkpoint",
        "method",
        "method_normalized",
        "budget",
        "tasks",
        "n_samples_expected",
        "n_packed",
        "n_predictions",
        "packed_artifact",
        "per_sample_prediction",
        "per_sample_score",
        "tokenizer",
        "prompt_wrapper",
        "decoding_config",
        "exact_budget_pass",
        "full_provenance",
        "status",
        "config_path",
        "final_root",
        "audit_path",
        "packed_sha256",
        "config_sha256",
        "source_manifest",
        "notes",
    ]
    out_csv = OUT_ROOT / "ledger" / "EXPERIMENT_LEDGER.csv"
    out_json = OUT_ROOT / "ledger" / "EXPERIMENT_LEDGER.json"
    write_csv(out_csv, rows, fields)
    payload = {
        "created_at": utc_now(),
        "stage_id": OUT_ROOT.name,
        "n_units": len(rows),
        "status_counts": {},
        "units": rows,
    }
    for row in rows:
        payload["status_counts"][row["status"]] = payload["status_counts"].get(row["status"], 0) + 1
    write_json(out_json, payload)

    missing = [r for r in rows if r["status"] in {"MISSING", "BLOCKED", "PARTIAL", "GPU_PENDING"}]
    missing_md = ["# MISSING_ARTIFACTS", "", f"Generated: {utc_now()}", ""]
    missing_md.append("| model | budget | method | status | notes |")
    missing_md.append("|---|---:|---|---|---|")
    for r in missing:
        missing_md.append(
            f"| {r['model']} | {r['budget']} | {r['method']} | {r['status']} | {r.get('notes','').replace('|','/')} |"
        )
    if not missing:
        missing_md.append("_No missing/partial/blocked/GPU_PENDING units._")
    write_text(OUT_ROOT / "ledger" / "MISSING_ARTIFACTS.md", "\n".join(missing_md))

    donot = [
        "# DO_NOT_RERUN",
        "",
        f"Generated: {utc_now()}",
        "",
        "Do not overwrite or re-run the following completed BridgePack result roots while GPU is occupied.",
        "Recompute statistics / evidence metrics from existing artifacts only.",
        "",
        "## Official / fairness-complete cells",
        "",
    ]
    for r in rows:
        if r["status"] == "COMPLETE":
            donot.append(
                f"- `{r['model']}` budget={r['budget']} method=`{r['method']}` root=`{r.get('final_root','')}`"
            )
    donot.extend(
        [
            "",
            "## Absolute bans during CPU stage",
            "",
            "- Do not kill/pause PID of other-paper GPU jobs.",
            "- Do not initialize CUDA or load QA/compressor model weights.",
            "- Do not write into historical `results/bridgepack_*` directories.",
            "- All new outputs go under `" + str(OUT_ROOT.relative_to(REPO_ROOT)) + "`.",
            "",
        ]
    )
    write_text(OUT_ROOT / "ledger" / "DO_NOT_RERUN.md", "\n".join(donot))
    # Also copy top-level convenience links
    write_csv(OUT_ROOT / "EXPERIMENT_LEDGER.csv", rows, fields)
    write_json(OUT_ROOT / "EXPERIMENT_LEDGER.json", payload)
    write_text(OUT_ROOT / "MISSING_ARTIFACTS.md", "\n".join(missing_md))
    write_text(OUT_ROOT / "DO_NOT_RERUN.md", "\n".join(donot))
    return payload


if __name__ == "__main__":
    payload = build_ledger()
    print(json.dumps(payload["status_counts"], indent=2))
    print("units", payload["n_units"])
