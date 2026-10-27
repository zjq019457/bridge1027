#!/usr/bin/env python3
"""Audit Elastic-DRM-Add side-budget policies without generation.

Elastic-DRM-Add keeps SlimInfer original A_K intact and allocates a small
sample-level additive side budget M(x).  This script only performs offline
feature/budget analysis from existing artifacts; it does not decode, generate,
score, swap, or delete active blocks.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


DEFAULT_INPUT_DIR = Path("results/drm_budget_neutral_swap_multihop200_20260706_042717")
DEFAULT_LONGBENCH_DIR = Path("/home/yu-bai/Desktop/jiaqizhang/HeadKV/data/LongBench")
POLICIES = ["E0", "E1", "E2", "E3", "E4"]
METHOD_ORIGINAL = "sliminfer_original"
METHOD_DRM_ADD = "drm_add"
METHOD_RANDOM_ADD = "random_add"
METHOD_LARGER = "sliminfer_larger"
METHOD_DRM_SWAP = "budget_neutral_drm_swap"
SIDE_BUDGET_SET = {0, 4, 8, 16, 32}
E4_SIDE_BUDGET_SET = {0, 4, 8}
E4_THRESHOLDS = {
    "rescue_score_margin": 0.80,
    "selected_coverage_risk": 0.70,
    "candidate_evidence_density": 0.80,
    "selected_redundancy": 0.50,
    "pruned_query_overlap": 0.55,
    "missing_query_fail_closed_extreme": 0.92,
    "strong_pass": 0.90,
}
JsonRow = Dict[str, Any]
Key = Tuple[str, int]


def read_jsonl(path: Path, *, missing_ok: bool = False) -> List[JsonRow]:
    if not path.exists():
        if missing_ok:
            return []
        raise RuntimeError(f"missing JSONL: {path}")
    rows: List[JsonRow] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def as_float(value: Any) -> Optional[float]:
    if value in (None, "", [], {}):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def safe_mean(values: Iterable[float]) -> float:
    items = [float(value) for value in values if math.isfinite(float(value))]
    return float(sum(items) / len(items)) if items else 0.0


def key_of(row: Mapping[str, Any]) -> Key:
    return str(row.get("task", "")), int(row.get("sample_index", -1))


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_longbench_samples(longbench_dir: Path, dataset: str) -> List[JsonRow]:
    candidates = [
        longbench_dir / f"{dataset}.jsonl",
        longbench_dir / f"{dataset}.json",
        longbench_dir / dataset / "test.jsonl",
        longbench_dir / dataset / "test.json",
        longbench_dir / dataset / f"{dataset}.jsonl",
        longbench_dir / dataset / f"{dataset}.json",
    ]
    for path in candidates:
        if not path.exists():
            continue
        if path.suffix == ".jsonl":
            return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict) and isinstance(payload.get("data"), list):
            return payload["data"]
    return []


def sample_query_text(sample: Mapping[str, Any]) -> str:
    for key in ("question", "query", "input", "instruction"):
        value = sample.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def content_tokens(text: str) -> set[str]:
    stop = {
        "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "by", "with",
        "was", "were", "is", "are", "be", "as", "at", "from", "which", "who", "whom",
        "whose", "what", "when", "where", "why", "how", "did", "does", "do", "this",
        "that", "these", "those", "only", "give", "answer", "question", "based",
    }
    tokens = {
        token.lower()
        for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9_\-']*", str(text or ""))
    }
    return {token for token in tokens if len(token) > 1 and token not in stop}


def overlap_with_query(block_text: str, query_tokens: set[str]) -> Optional[float]:
    if not query_tokens:
        return None
    block_tokens = content_tokens(block_text)
    return float(len(block_tokens.intersection(query_tokens)) / max(1, len(query_tokens)))


def build_chat_prompt(tokenizer: Any, prompt: str) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt


def build_prompt_block_texts(tokenizer: Any, prompt: str, block_size: int) -> List[str]:
    encoded = tokenizer(prompt, truncation=False, add_special_tokens=True)
    input_ids = list(getattr(encoded, "input_ids", encoded["input_ids"]))
    block_texts: List[str] = []
    for start in range(0, len(input_ids), int(block_size)):
        block_texts.append(
            tokenizer.decode(
                input_ids[start:start + int(block_size)],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        )
    return block_texts


def load_tokenizer(tokenizer_path: Optional[Path]) -> Tuple[Any, Optional[str]]:
    if tokenizer_path is None:
        return None, "tokenizer_path_unavailable"
    try:
        from transformers import AutoTokenizer  # type: ignore
    except Exception as exc:
        return None, f"transformers_unavailable:{type(exc).__name__}"
    try:
        return AutoTokenizer.from_pretrained(str(tokenizer_path), trust_remote_code=True), None
    except Exception as exc:
        return None, f"tokenizer_load_failed:{type(exc).__name__}"


def build_query_overlap_provider(
    *,
    input_dir: Path,
    longbench_dir: Path,
    tokenizer_path: Optional[Path],
    block_size: int,
    max_model_len: int = 32768,
) -> Tuple[Dict[Key, Dict[str, Any]], Dict[str, Any]]:
    tokenizer, tokenizer_error = load_tokenizer(tokenizer_path)
    if tokenizer is None:
        return {}, {
            "available": False,
            "reason": tokenizer_error or "tokenizer_unavailable",
            "source": "longbench_prompt_reconstruction",
        }
    prompt_path = Path("configs") / "dataset2prompt.json"
    if not prompt_path.exists():
        return {}, {
            "available": False,
            "reason": "dataset2prompt_missing",
            "source": "longbench_prompt_reconstruction",
        }
    dataset2prompt = load_json(prompt_path)
    tasks = sorted({task for task, _ in read_per_sample(input_dir).get(METHOD_ORIGINAL, {})})
    block_texts_by_key: Dict[Key, Dict[str, Any]] = {}
    missing: List[str] = []
    for task in tasks:
        samples = load_longbench_samples(longbench_dir, task)
        prompt_format = dataset2prompt.get(task)
        if not samples or not isinstance(prompt_format, str):
            missing.append(task)
            continue
        for sample_index, sample in enumerate(samples):
            key = (task, int(sample_index))
            try:
                prompt = prompt_format.format(**sample)
                raw_ids = list(getattr(tokenizer(prompt, truncation=False), "input_ids", []))
                if int(max_model_len) > 0 and len(raw_ids) > int(max_model_len):
                    half = int(max_model_len) // 2
                    prompt = (
                        tokenizer.decode(raw_ids[:half], skip_special_tokens=True)
                        + tokenizer.decode(raw_ids[-half:], skip_special_tokens=True)
                    )
                if task not in {"trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"}:
                    prompt = build_chat_prompt(tokenizer, prompt)
                query_text = sample_query_text(sample)
                block_texts_by_key[key] = {
                    "query_text": query_text,
                    "query_tokens": content_tokens(query_text),
                    "block_texts": build_prompt_block_texts(tokenizer, prompt, block_size),
                }
            except Exception:
                continue
    return block_texts_by_key, {
        "available": bool(block_texts_by_key),
        "reason": "" if block_texts_by_key else "no_prompt_blocks_reconstructed",
        "source": "longbench_prompt_reconstruction",
        "tokenizer_path": str(tokenizer_path),
        "longbench_dir": str(longbench_dir),
        "block_size": int(block_size),
        "max_model_len": int(max_model_len),
        "reconstructed_samples": len(block_texts_by_key),
        "missing_tasks": missing,
    }


def query_overlap_features_for_sample(
    provider: Mapping[Key, Mapping[str, Any]],
    key: Key,
    *,
    active_ids: Sequence[Any],
    rescued_ids: Sequence[Any],
    num_original_blocks: int,
    top_m: int,
) -> Dict[str, Any]:
    payload = provider.get(key)
    if not payload:
        return {"available": False, "reason": "prompt_block_text_unavailable"}
    block_texts = payload.get("block_texts")
    query_tokens = payload.get("query_tokens")
    if not isinstance(block_texts, list) or not isinstance(query_tokens, set) or not query_tokens:
        return {"available": False, "reason": "query_or_block_text_unavailable"}
    all_ids = set(range(max(0, int(num_original_blocks))))
    active = {int(x) for x in active_ids if 0 <= int(x) < len(block_texts)}
    rescued = {int(x) for x in rescued_ids if 0 <= int(x) < len(block_texts)}
    if not all_ids:
        all_ids = set(range(len(block_texts)))
    pruned = sorted((all_ids - active).intersection(range(len(block_texts))))
    if not pruned:
        return {"available": False, "reason": "no_pruned_blocks_with_text"}

    def value(block_id: int) -> float:
        overlap = overlap_with_query(str(block_texts[int(block_id)]), query_tokens)
        return float(overlap or 0.0)

    pruned_values = [value(block_id) for block_id in pruned]
    active_values = [value(block_id) for block_id in sorted(active)]
    rescued_values = [value(block_id) for block_id in sorted(rescued)]
    top_count = max(1, min(int(top_m or 1), len(pruned_values)))
    pruned_top = sorted(pruned_values, reverse=True)[:top_count]
    selected_mean = safe_mean(active_values)
    pruned_top_mean = safe_mean(pruned_top)
    return {
        "available": True,
        "reason": "",
        "pruned_query_overlap_max": max(pruned_values) if pruned_values else 0.0,
        "pruned_query_overlap_mean_topM": pruned_top_mean,
        "rescued_candidate_query_overlap": safe_mean(rescued_values),
        "selected_query_overlap_mean": selected_mean,
        "selected_vs_pruned_query_overlap_gap": pruned_top_mean - selected_mean,
        "pruned_block_count_for_overlap": len(pruned_values),
        "rescued_block_count_for_overlap": len(rescued_values),
    }


def read_per_sample(input_dir: Path) -> Dict[str, Dict[Key, JsonRow]]:
    rows = read_jsonl(input_dir / "audit" / "per_sample_audit.jsonl")
    by_method: Dict[str, Dict[Key, JsonRow]] = defaultdict(dict)
    for row in rows:
        method = str(row.get("method_key", ""))
        key = key_of(row)
        if method and key[0] and key[1] >= 0:
            by_method[method][key] = row
    return by_method


def read_rescue_rows(input_dir: Path, method: str) -> Dict[Key, List[JsonRow]]:
    grouped: Dict[Key, List[JsonRow]] = defaultdict(list)
    for path in sorted((input_dir / "runs" / method).glob("*/audit/rescue_policy_audit.jsonl")):
        for row in read_jsonl(path, missing_ok=True):
            key = key_of(row)
            if key[0] and key[1] >= 0:
                grouped[key].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: int(row.get("layer_id", -1)))
    return grouped


def stat_value(row: Mapping[str, Any], stat_name: str, field: str) -> Optional[float]:
    value = row.get(stat_name)
    if isinstance(value, Mapping):
        return as_float(value.get(field))
    return None


def max_available(values: Iterable[Optional[float]]) -> Optional[float]:
    items = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return max(items) if items else None


def min_available(values: Iterable[Optional[float]]) -> Optional[float]:
    items = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return min(items) if items else None


def normalize_feature(values_by_key: Mapping[Key, Optional[float]], *, higher_is_risk: bool = True) -> Dict[Key, Optional[float]]:
    values = [float(value) for value in values_by_key.values() if value is not None and math.isfinite(float(value))]
    if not values:
        return {key: None for key in values_by_key}
    lo, hi = min(values), max(values)
    if abs(hi - lo) < 1e-12:
        return {key: 0.0 if values_by_key[key] is not None else None for key in values_by_key}
    result: Dict[Key, Optional[float]] = {}
    for key, value in values_by_key.items():
        if value is None or not math.isfinite(float(value)):
            result[key] = None
            continue
        norm = (float(value) - lo) / (hi - lo)
        result[key] = norm if higher_is_risk else 1.0 - norm
    return result


def fixed_extra_blocks(original: Mapping[Key, JsonRow], drm_add: Mapping[Key, JsonRow]) -> Dict[Key, float]:
    extras: Dict[Key, float] = {}
    for key, add_row in drm_add.items():
        original_row = original.get(key)
        if not original_row:
            continue
        add_count = as_float(add_row.get("added_count"))
        if add_count is not None:
            extras[key] = max(0.0, add_count)
            continue
        add_selected = as_float(add_row.get("selected_block_count"))
        original_selected = as_float(original_row.get("selected_block_count"))
        if add_selected is not None and original_selected is not None:
            extras[key] = max(0.0, add_selected - original_selected)
    return extras


def feature_coverage(rows: Sequence[Mapping[str, Any]], feature_names: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    coverage: Dict[str, Dict[str, Any]] = {}
    total = len(rows)
    for name in feature_names:
        available = sum(1 for row in rows if row.get(f"{name}_available"))
        coverage[name] = {
            "available": available,
            "total": total,
            "coverage": float(available / max(1, total)),
        }
    return coverage


def build_feature_rows(
    by_method: Mapping[str, Mapping[Key, JsonRow]],
    drm_add_rescue: Mapping[Key, List[JsonRow]],
    query_overlap_provider: Optional[Mapping[Key, Mapping[str, Any]]] = None,
) -> Tuple[List[JsonRow], Dict[str, Any]]:
    original = by_method.get(METHOD_ORIGINAL, {})
    drm_add = by_method.get(METHOD_DRM_ADD, {})
    random_add = by_method.get(METHOD_RANDOM_ADD, {})
    larger = by_method.get(METHOD_LARGER, {})
    keys = sorted(set(original).intersection(drm_add))

    raw: Dict[str, Dict[Key, Optional[float]]] = {
        "rescue_score_margin": {},
        "pruned_query_overlap": {},
        "selected_coverage_risk": {},
        "selected_redundancy": {},
        "candidate_evidence_density": {},
    }
    candidate_capacity: Dict[Key, int] = {}
    metadata: Dict[Key, Dict[str, Any]] = {}

    for key in keys:
        o = original[key]
        add = drm_add[key]
        rescue_rows = drm_add_rescue.get(key, [])
        final_rescue = rescue_rows[-1] if rescue_rows else {}
        g_margins = []
        g_means = []
        r_means = []
        coverage_values = []
        for row in rescue_rows:
            g_max = stat_value(row, "G_diffusion_stats", "max")
            g_mean = stat_value(row, "G_diffusion_stats", "mean")
            if g_max is not None and g_mean is not None:
                g_margins.append(max(0.0, g_max - g_mean))
            if g_mean is not None:
                g_means.append(g_mean)
            r_mean = stat_value(row, "R_active_stats", "mean")
            if r_mean is not None:
                r_means.append(r_mean)
            active = row.get("active_fixed_page_ids")
            num_original = as_float(row.get("num_original_blocks"))
            if isinstance(active, list) and num_original and num_original > 0:
                coverage_values.append(min(1.0, len(active) / num_original))
            elif as_float(row.get("K")) is not None and num_original and num_original > 0:
                coverage_values.append(min(1.0, float(row.get("K")) / num_original))

        raw["rescue_score_margin"][key] = max_available(g_margins)
        raw["candidate_evidence_density"][key] = max_available(g_means)
        raw["selected_redundancy"][key] = max_available(r_means)
        min_coverage = min_available(coverage_values)
        raw["selected_coverage_risk"][key] = (1.0 - min_coverage) if min_coverage is not None else None

        num_original_blocks = int(as_float(final_rescue.get("num_original_blocks")) or as_float(o.get("prompt_block_count")) or 0)
        active_for_overlap = (
            final_rescue.get("active_fixed_page_ids")
            if isinstance(final_rescue.get("active_fixed_page_ids"), list)
            else []
        )
        rescued_for_overlap = (
            final_rescue.get("rescued_fixed_page_ids")
            if isinstance(final_rescue.get("rescued_fixed_page_ids"), list)
            else []
        )
        fixed_add_m = int(as_float(add.get("added_count")) or as_float(final_rescue.get("rescued_count")) or 1)
        overlap_features = query_overlap_features_for_sample(
            query_overlap_provider or {},
            key,
            active_ids=active_for_overlap,
            rescued_ids=rescued_for_overlap,
            num_original_blocks=num_original_blocks,
            top_m=fixed_add_m,
        )
        if overlap_features.get("available"):
            pruned_max = as_float(overlap_features.get("pruned_query_overlap_max"))
            pruned_topm = as_float(overlap_features.get("pruned_query_overlap_mean_topM"))
            raw["pruned_query_overlap"][key] = max_available([pruned_max, pruned_topm])
        else:
            raw["pruned_query_overlap"][key] = None

        capacity = as_float(o.get("pruned_candidate_count"))
        if capacity is None:
            prompt_blocks = as_float(o.get("prompt_block_count"))
            selected = as_float(o.get("selected_block_count"))
            capacity = max(0.0, prompt_blocks - selected) if prompt_blocks is not None and selected is not None else 0.0
        candidate_capacity[key] = max(0, int(capacity))

        metadata[key] = {
            "task": key[0],
            "sample_index": key[1],
            "sample_id": o.get("sample_id", add.get("sample_id", "")),
            "K": int(as_float(o.get("selected_block_count")) or as_float(o.get("K")) or 0),
            "prompt_block_count": int(as_float(o.get("prompt_block_count")) or 0),
            "candidate_capacity": candidate_capacity[key],
            "fixed_drm_add_extra_blocks": float(max(0.0, as_float(add.get("added_count")) or 0.0)),
            "random_add_extra_blocks": float(max(0.0, as_float(random_add.get(key, {}).get("added_count")) or 0.0)),
            "larger_extra_blocks": float(max(0.0, (as_float(larger.get(key, {}).get("selected_block_count")) or 0.0) - (as_float(o.get("selected_block_count")) or 0.0))) if key in larger else 0.0,
            "fixed_drm_add_rescued_blocks": list(final_rescue.get("rescued_fixed_page_ids", [])) if isinstance(final_rescue.get("rescued_fixed_page_ids"), list) else [],
            "fixed_drm_add_final_layer": int(final_rescue.get("layer_id", -1)) if final_rescue else "",
            "pruned_query_overlap_reconstruction_available": bool(overlap_features.get("available")),
            "pruned_query_overlap_reconstruction_reason": str(overlap_features.get("reason", "")),
            "pruned_query_overlap_max": overlap_features.get("pruned_query_overlap_max"),
            "pruned_query_overlap_mean_topM": overlap_features.get("pruned_query_overlap_mean_topM"),
            "rescued_candidate_query_overlap": overlap_features.get("rescued_candidate_query_overlap"),
            "selected_query_overlap_mean": overlap_features.get("selected_query_overlap_mean"),
            "selected_vs_pruned_query_overlap_gap": overlap_features.get("selected_vs_pruned_query_overlap_gap"),
            "pruned_block_count_for_overlap": overlap_features.get("pruned_block_count_for_overlap"),
            "rescued_block_count_for_overlap": overlap_features.get("rescued_block_count_for_overlap"),
        }

    normalized = {
        name: normalize_feature(values, higher_is_risk=True)
        for name, values in raw.items()
    }
    rows: List[JsonRow] = []
    for key in keys:
        components = {
            name: normalized[name].get(key)
            for name in raw
        }
        available_components = [value for value in components.values() if value is not None]
        risk_score = safe_mean(available_components)
        if risk_score >= 0.66:
            risk_level = "high"
        elif risk_score >= 0.40:
            risk_level = "mid"
        else:
            risk_level = "low"
        row: JsonRow = {
            **metadata[key],
            "risk_score": risk_score,
            "risk_level": risk_level,
            "risk_feature_count": len(available_components),
        }
        for name in raw:
            row[f"{name}_raw"] = raw[name].get(key)
            row[f"{name}_risk_component"] = components[name]
            row[f"{name}_available"] = raw[name].get(key) is not None
        rows.append(row)
    diagnostics = {
        "sample_count": len(rows),
        "feature_names": list(raw),
    }
    return rows, diagnostics


def apply_policies(rows: List[JsonRow], fixed_avg_extra: float) -> None:
    for row in rows:
        risk = float(row.get("risk_score", 0.0) or 0.0)
        level = str(row.get("risk_level", "low"))
        capacity = int(row.get("candidate_capacity", 0) or 0)
        requested = {
            "E0": 0,
            "E1": 8 if risk >= 0.66 else 0,
            "E2": 16 if level == "high" else 8 if level == "mid" else 0,
            "E3": 0,
            "E4": 0,
        }
        for policy, value in requested.items():
            if value not in SIDE_BUDGET_SET:
                raise RuntimeError(f"invalid side budget {value} for {policy}")
            row[f"{policy}_M_requested"] = value
            row[f"{policy}_M_effective"] = min(value, capacity)
        row["E4_gate_reason"] = "not_evaluated"
        row["E4_gate_eligible_before_cap"] = False

    cap_total = max(0.0, 0.5 * float(fixed_avg_extra) * len(rows))
    used = 0.0
    for row in sorted(rows, key=lambda item: (-float(item.get("risk_score", 0.0) or 0.0), item["task"], item["sample_index"])):
        capacity = int(row.get("candidate_capacity", 0) or 0)
        if capacity <= 0:
            continue
        requested = 4
        effective = min(requested, capacity)
        if effective <= 0:
            continue
        if used + effective <= cap_total + 1e-9:
            row["E3_M_requested"] = requested
            row["E3_M_effective"] = effective
            used += effective

    e4_candidates: List[JsonRow] = []
    for row in rows:
        components = {
            "rescue_score_margin": row.get("rescue_score_margin_risk_component"),
            "selected_coverage_risk": row.get("selected_coverage_risk_risk_component"),
            "candidate_evidence_density": row.get("candidate_evidence_density_risk_component"),
            "selected_redundancy": row.get("selected_redundancy_risk_component"),
            "pruned_query_overlap": row.get("pruned_query_overlap_risk_component"),
        }
        numeric = {
            name: float(value)
            for name, value in components.items()
            if value is not None and math.isfinite(float(value))
        }
        query_available = bool(row.get("pruned_query_overlap_available"))
        required_without_query = [
            numeric.get("rescue_score_margin", -1.0) >= E4_THRESHOLDS["rescue_score_margin"],
            numeric.get("selected_coverage_risk", -1.0) >= E4_THRESHOLDS["selected_coverage_risk"],
            numeric.get("candidate_evidence_density", -1.0) >= E4_THRESHOLDS["candidate_evidence_density"],
            numeric.get("selected_redundancy", -1.0) >= E4_THRESHOLDS["selected_redundancy"],
        ]
        query_pass = (
            query_available
            and numeric.get("pruned_query_overlap", -1.0) >= E4_THRESHOLDS["pruned_query_overlap"]
        )
        missing_query_extreme = (
            not query_available
            and all(
                numeric.get(name, -1.0) >= E4_THRESHOLDS["missing_query_fail_closed_extreme"]
                for name in [
                    "rescue_score_margin",
                    "selected_coverage_risk",
                    "candidate_evidence_density",
                    "selected_redundancy",
                ]
            )
        )
        eligible = bool(all(required_without_query) and (query_pass or missing_query_extreme))
        row["E4_gate_eligible_before_cap"] = eligible
        row["E4_pruned_query_overlap_gate_available"] = query_available
        row["E4_gate_thresholds"] = dict(E4_THRESHOLDS)
        if not eligible:
            if not all(required_without_query):
                row["E4_gate_reason"] = "risk_and_gate_failed"
            elif not query_available:
                row["E4_gate_reason"] = "pruned_query_overlap_unavailable_fail_closed"
            else:
                row["E4_gate_reason"] = "pruned_query_overlap_below_threshold"
            continue
        strong_pass = all(
            numeric.get(name, -1.0) >= E4_THRESHOLDS["strong_pass"]
            for name in components
            if name != "pruned_query_overlap" or query_available
        )
        requested = 8 if strong_pass and query_available else 4
        if requested not in E4_SIDE_BUDGET_SET:
            raise RuntimeError(f"invalid E4 side budget {requested}")
        row["E4_M_requested"] = requested
        row["E4_gate_reason"] = "eligible_pending_cap"
        row["E4_gate_strength"] = safe_mean(numeric.values())
        e4_candidates.append(row)

    used = 0.0
    for row in sorted(
        e4_candidates,
        key=lambda item: (
            -float(item.get("E4_gate_strength", 0.0) or 0.0),
            -float(item.get("pruned_query_overlap_risk_component", 0.0) or 0.0),
            item["task"],
            item["sample_index"],
        ),
    ):
        capacity = int(row.get("candidate_capacity", 0) or 0)
        requested = int(row.get("E4_M_requested", 0) or 0)
        effective = min(requested, capacity)
        if effective >= 8 and used + 8 <= cap_total + 1e-9:
            row["E4_M_effective"] = 8
            row["E4_gate_reason"] = "accepted_m8"
            used += 8
        elif effective >= 4 and used + 4 <= cap_total + 1e-9:
            row["E4_M_effective"] = 4
            row["E4_gate_reason"] = "accepted_m4"
            used += 4
        else:
            row["E4_M_effective"] = 0
            row["E4_gate_reason"] = "rejected_by_half_overhead_cap_or_capacity"


def policy_distribution(rows: Sequence[Mapping[str, Any]], policy: str, field_suffix: str) -> Dict[str, int]:
    counts = Counter(int(row.get(f"{policy}_{field_suffix}", 0) or 0) for row in rows)
    return {str(key): counts[key] for key in sorted(counts)}


def historical_cases(by_method: Mapping[str, Mapping[Key, JsonRow]]) -> Dict[Key, Dict[str, Any]]:
    original = by_method.get(METHOD_ORIGINAL, {})
    drm_add = by_method.get(METHOD_DRM_ADD, {})
    cases: Dict[Key, Dict[str, Any]] = {}
    for key, add_row in drm_add.items():
        original_row = original.get(key)
        if not original_row:
            continue
        o_score = as_float(original_row.get("prediction_score"))
        add_score = as_float(add_row.get("prediction_score"))
        if o_score is None or add_score is None:
            continue
        delta = add_score - o_score
        if delta > 1e-9:
            case_type = "positive"
        elif delta < -1e-9:
            case_type = "negative"
        else:
            case_type = "zero"
        cases[key] = {
            "original_score": o_score,
            "drm_add_score": add_score,
            "drm_add_delta": delta,
            "case_type": case_type,
            "obvious_negative": bool(delta <= -5.0),
        }
    return cases


def summarize_policy(
    rows: Sequence[Mapping[str, Any]],
    policy: str,
    fixed_avg_extra: float,
    fixed_avg_overhead: float,
    cases: Mapping[Key, Mapping[str, Any]],
) -> JsonRow:
    extras = [float(row.get(f"{policy}_M_effective", 0) or 0) for row in rows]
    avg_extra = safe_mean(extras)
    overheads = [
        float(row.get(f"{policy}_M_effective", 0) or 0) / max(1.0, float(row.get("K", 0) or 0))
        for row in rows
    ]
    triggered = {
        (str(row["task"]), int(row["sample_index"]))
        for row in rows
        if int(row.get(f"{policy}_M_effective", 0) or 0) > 0
    }
    positives = {key for key, case in cases.items() if case.get("case_type") == "positive"}
    negatives = {key for key, case in cases.items() if case.get("case_type") == "negative"}
    obvious_negatives = {key for key, case in cases.items() if case.get("obvious_negative")}
    return {
        "policy": policy,
        "trigger_count": len(triggered),
        "trigger_rate": float(len(triggered) / max(1, len(rows))),
        "dynamic_M_requested_distribution": policy_distribution(rows, policy, "M_requested"),
        "effective_extra_distribution": policy_distribution(rows, policy, "M_effective"),
        "avg_extra_blocks": avg_extra,
        "avg_kv_overhead": safe_mean(overheads),
        "avg_extra_vs_fixed_drm_add": (avg_extra / fixed_avg_extra) if fixed_avg_extra > 0 else 0.0,
        "avg_overhead_vs_fixed_drm_add": (safe_mean(overheads) / fixed_avg_overhead) if fixed_avg_overhead > 0 else 0.0,
        "historical_positive_total": len(positives),
        "historical_positive_covered": len(positives.intersection(triggered)),
        "historical_negative_total": len(negatives),
        "historical_negative_triggered": len(negatives.intersection(triggered)),
        "historical_obvious_negative_total": len(obvious_negatives),
        "historical_obvious_negative_triggered": len(obvious_negatives.intersection(triggered)),
        "triggered_samples": sorted(f"{task}:{index}" for task, index in triggered),
    }


def write_report(
    output_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    feature_cov: Mapping[str, Mapping[str, Any]],
    policy_summaries: Sequence[Mapping[str, Any]],
    baseline: Mapping[str, Any],
    recommendation: Mapping[str, Any],
    query_overlap_meta: Mapping[str, Any],
) -> None:
    lines = [
        "# Elastic-DRM-Add Offline Audit",
        "",
        f"- Result directory: `{output_dir}`",
        "- Stage: audit/offline budget analysis only",
        "- Generation run: **No**",
        "- Scorer run: **No**",
        "- Swap/deletion: **No**",
        "- Gold/final prediction used for policy: **No**",
        "",
        "## Risk Feature Coverage",
        "",
    ]
    for name, stats in feature_cov.items():
        lines.append(f"- {name}: {stats['available']}/{stats['total']} ({stats['coverage']:.1%})")
    lines.extend([
        "",
        "## Pruned Query Overlap Reconstruction",
        "",
        f"- Reconstructed: {bool(query_overlap_meta.get('available'))}",
        f"- Source: `{query_overlap_meta.get('source', 'unavailable')}`",
        f"- Reason: `{query_overlap_meta.get('reason', '')}`",
        f"- Reconstructed samples: {query_overlap_meta.get('reconstructed_samples', 0)}",
        f"- Thresholds: `{json.dumps(E4_THRESHOLDS, sort_keys=True)}`",
    ])
    lines.extend([
        "",
        "## Baseline Overhead",
        "",
        f"- Fixed DRM-Add avg extra blocks: {baseline['fixed_drm_add_avg_extra_blocks']:.4f}",
        f"- Fixed DRM-Add avg KV overhead: {baseline['fixed_drm_add_avg_kv_overhead']:.4%}",
        f"- Random-Add avg extra blocks: {baseline['random_add_avg_extra_blocks']:.4f}",
        f"- Larger avg extra blocks: {baseline['larger_avg_extra_blocks']:.4f}",
        "",
        "## Dynamic Policies",
        "",
    ])
    for summary in policy_summaries:
        lines.extend([
            f"### Policy {summary['policy']}",
            f"- Requested M distribution: `{json.dumps(summary['dynamic_M_requested_distribution'], sort_keys=True)}`",
            f"- Effective extra distribution: `{json.dumps(summary['effective_extra_distribution'], sort_keys=True)}`",
            f"- Trigger count: {summary['trigger_count']} ({summary['trigger_rate']:.1%})",
            f"- Avg extra blocks: {summary['avg_extra_blocks']:.4f}",
            f"- Avg KV overhead: {summary['avg_kv_overhead']:.4%}",
            f"- Overhead vs fixed DRM-Add: {summary['avg_extra_vs_fixed_drm_add']:.4f}",
            f"- Historical DRM-Add positives covered: {summary['historical_positive_covered']}/{summary['historical_positive_total']}",
            f"- Historical negatives triggered: {summary['historical_negative_triggered']}/{summary['historical_negative_total']}",
            f"- Historical obvious negatives triggered: {summary['historical_obvious_negative_triggered']}/{summary['historical_obvious_negative_total']}",
            "",
        ])
    e3 = next((item for item in policy_summaries if item["policy"] == "E3"), {})
    e4 = next((item for item in policy_summaries if item["policy"] == "E4"), {})
    e4_triggered = list(e4.get("triggered_samples", []))
    e3_triggered = set(e3.get("triggered_samples", []))
    e4_triggered_set = set(e4_triggered)
    e4_new_vs_e3 = sorted(e4_triggered_set.difference(e3_triggered))
    e4_dropped_vs_e3 = sorted(e3_triggered.difference(e4_triggered_set))
    lines.extend([
        "## E4 vs E3",
        "",
        f"- E3 trigger count: {e3.get('trigger_count', 0)}",
        f"- E4 trigger count: {e4.get('trigger_count', 0)}",
        f"- E3 obvious negatives triggered: {e3.get('historical_obvious_negative_triggered', 0)}/{e3.get('historical_obvious_negative_total', 0)}",
        f"- E4 obvious negatives triggered: {e4.get('historical_obvious_negative_triggered', 0)}/{e4.get('historical_obvious_negative_total', 0)}",
        f"- E3 positives covered: {e3.get('historical_positive_covered', 0)}/{e3.get('historical_positive_total', 0)}",
        f"- E4 positives covered: {e4.get('historical_positive_covered', 0)}/{e4.get('historical_positive_total', 0)}",
        f"- E4 new triggers vs E3: `{json.dumps(e4_new_vs_e3, ensure_ascii=False)}`",
        f"- E4 removed E3 triggers preview: `{json.dumps(e4_dropped_vs_e3[:80], ensure_ascii=False)}`",
        "",
        "## Recommendation",
        "",
        f"- Audit decision: **{recommendation['decision']}**",
        f"- Worth tiny quality smoke: {recommendation['worth_tiny_quality_smoke']}",
        f"- Primary policy for smoke: {recommendation['primary_policy']}",
        f"- E4 triggered samples: `{json.dumps(e4_triggered, ensure_ascii=False)}`",
    ])
    for reason in recommendation.get("reasons", []):
        lines.append(f"- Reason: `{reason}`")
    lines.append("")
    lines.append("Historical positive/negative coverage is diagnostic only; no policy uses gold answers, prediction text, or final score.")
    (output_dir / "elastic_drm_add_audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def decide(policy_summaries: Sequence[Mapping[str, Any]], feature_cov: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    e4 = next(summary for summary in policy_summaries if summary["policy"] == "E4")
    reasons: List[str] = []
    core_names = [
        "rescue_score_margin",
        "selected_coverage_risk",
        "candidate_evidence_density",
        "selected_redundancy",
    ]
    available_core = sum(1 for name in core_names if feature_cov.get(name, {}).get("coverage", 0.0) >= 0.95)
    if available_core < len(core_names):
        reasons.append("core_risk_feature_coverage_incomplete")
    if feature_cov.get("pruned_query_overlap", {}).get("coverage", 0.0) < 0.95:
        reasons.append("pruned_query_overlap_or_equivalent_signal_unavailable")
    if int(e4["trigger_count"]) <= 0:
        reasons.append("e4_rejects_all_samples")
    if int(e4["trigger_count"]) >= int(feature_cov.get("rescue_score_margin", {}).get("total", 0)):
        reasons.append("e4_degenerates_to_fixed_add")
    if float(e4["avg_extra_vs_fixed_drm_add"]) > 0.50 + 1e-9:
        reasons.append("e4_overhead_exceeds_half_fixed_drm_add")
    if int(e4["historical_positive_total"]) > 0 and int(e4["historical_positive_covered"]) < 2:
        reasons.append("e4_retains_fewer_than_two_historical_positives")
    if int(e4["historical_obvious_negative_triggered"]) >= 2:
        reasons.append("e4_triggers_too_many_obvious_negatives")
    return {
        "decision": "GO" if not reasons else "NO-GO",
        "worth_tiny_quality_smoke": bool(not reasons),
        "primary_policy": "E4",
        "reasons": reasons,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output_root", type=Path, default=Path("results"))
    parser.add_argument("--longbench_dir", type=Path, default=None)
    parser.add_argument("--tokenizer_path", type=Path, default=None)
    parser.add_argument("--block_size", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (args.output_root / f"elastic_drm_add_e4_audit_{timestamp}").resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    by_method = read_per_sample(args.input_dir)
    drm_add_rescue = read_rescue_rows(args.input_dir, METHOD_DRM_ADD)

    run_config = {}
    config_path = args.input_dir / "config.json"
    if config_path.exists():
        run_config = load_json(config_path)
    protocol = run_config.get("protocol", {}) if isinstance(run_config.get("protocol"), dict) else {}
    longbench_dir = args.longbench_dir or Path(protocol.get("longbench_dir", DEFAULT_LONGBENCH_DIR))
    tokenizer_path = args.tokenizer_path or (
        Path(protocol["model_name"]) if isinstance(protocol.get("model_name"), str) and protocol.get("model_name") else None
    )
    block_size = int(args.block_size or protocol.get("block_size", 64) or 64)
    generation_config = protocol.get("generation_config", {}) if isinstance(protocol.get("generation_config"), dict) else {}
    max_model_len = int(generation_config.get("max_model_len", 32768) or 32768)
    query_overlap_provider, query_overlap_meta = build_query_overlap_provider(
        input_dir=args.input_dir,
        longbench_dir=longbench_dir,
        tokenizer_path=tokenizer_path,
        block_size=block_size,
        max_model_len=max_model_len,
    )
    rows, diagnostics = build_feature_rows(by_method, drm_add_rescue, query_overlap_provider)
    if not rows:
        raise RuntimeError("no overlapping SlimInfer/DRM-Add samples found")

    fixed_extras = fixed_extra_blocks(by_method[METHOD_ORIGINAL], by_method[METHOD_DRM_ADD])
    fixed_avg_extra = safe_mean(fixed_extras.values())
    fixed_overheads = [
        fixed_extras.get((row["task"], int(row["sample_index"])), 0.0) / max(1.0, float(row.get("K", 0) or 0))
        for row in rows
    ]
    fixed_avg_overhead = safe_mean(fixed_overheads)
    apply_policies(rows, fixed_avg_extra)

    cases = historical_cases(by_method)
    for row in rows:
        key = (str(row["task"]), int(row["sample_index"]))
        case = cases.get(key, {})
        row["historical_drm_add_case_type"] = case.get("case_type", "unavailable")
        row["historical_drm_add_delta"] = case.get("drm_add_delta")
        row["policy_features_use_historical_quality"] = False
        for policy in POLICIES:
            row[f"{policy}_keeps_A_K_intact"] = True
            row[f"{policy}_deletes_original_blocks"] = False

    feature_names = diagnostics["feature_names"]
    feature_cov = feature_coverage(rows, feature_names)
    random_avg_extra = safe_mean(float(row.get("random_add_extra_blocks", 0.0) or 0.0) for row in rows)
    larger_avg_extra = safe_mean(float(row.get("larger_extra_blocks", 0.0) or 0.0) for row in rows)
    baseline = {
        "fixed_drm_add_avg_extra_blocks": fixed_avg_extra,
        "fixed_drm_add_avg_kv_overhead": fixed_avg_overhead,
        "random_add_avg_extra_blocks": random_avg_extra,
        "larger_avg_extra_blocks": larger_avg_extra,
    }
    policy_summaries = [
        summarize_policy(rows, policy, fixed_avg_extra, fixed_avg_overhead, cases)
        for policy in POLICIES
    ]
    recommendation = decide(policy_summaries, feature_cov)

    write_jsonl(output_dir / "elastic_drm_add_audit.jsonl", rows)
    write_csv(
        output_dir / "elastic_drm_add_policy_summary.csv",
        policy_summaries,
        [
            "policy",
            "trigger_count",
            "trigger_rate",
            "avg_extra_blocks",
            "avg_kv_overhead",
            "avg_extra_vs_fixed_drm_add",
            "avg_overhead_vs_fixed_drm_add",
            "historical_positive_total",
            "historical_positive_covered",
            "historical_negative_total",
            "historical_negative_triggered",
            "historical_obvious_negative_total",
            "historical_obvious_negative_triggered",
        ],
    )
    write_json(output_dir / "elastic_drm_add_audit_summary.json", {
        "result_dir": str(output_dir),
        "input_dir": str(args.input_dir),
        "sample_count": len(rows),
        "risk_feature_coverage": feature_cov,
        "query_overlap_reconstruction": query_overlap_meta,
        "e4_thresholds": E4_THRESHOLDS,
        "baseline": baseline,
        "policy_summaries": policy_summaries,
        "recommendation": recommendation,
        "quality_generation_performed": False,
        "scorer_invoked": False,
        "swap_performed": False,
        "deletes_A_K_blocks": False,
        "uses_gold_or_prediction_for_policy": False,
    })
    write_report(output_dir, rows, feature_cov, policy_summaries, baseline, recommendation, query_overlap_meta)
    print(json.dumps({
        "status": "ok",
        "output_dir": str(output_dir),
        "recommendation": recommendation,
        "primary_policy_summary": next(item for item in policy_summaries if item["policy"] == recommendation["primary_policy"]),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
