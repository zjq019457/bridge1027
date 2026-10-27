#!/usr/bin/env python3
"""Audit BridgePack packed-text budgets with the actual generation tokenizer.

This script loads only a tokenizer. It does not load model weights or use CUDA.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _detect_model(run_dir: Path, user_value: str) -> str:
    candidates = [
        user_value,
        os.environ.get("QWEN_MODEL", ""),
        _load_json(run_dir / "paper_tables" / "00_generation_config.json").get("tokenizer", ""),
        _load_json(run_dir / "paper_tables" / "00_generation_config.json").get("model_name_or_path", ""),
        _load_json(run_dir / "config.json").get("model_name_or_path", ""),
    ]
    for candidate in candidates:
        if candidate:
            return str(candidate)
    raise ValueError("No tokenizer/model path found. Pass --model_name_or_path or set QWEN_MODEL.")


def _load_tokenizer(model_name_or_path: str) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _packed_files(run_dir: Path) -> List[Path]:
    return sorted(run_dir.glob("**/packed_prompts/*/budget_*/*.jsonl"))


def _parse_packed_path(run_dir: Path, path: Path) -> Tuple[str, str, int, str]:
    rel = path.relative_to(run_dir)
    parts = rel.parts
    packed_idx = parts.index("packed_prompts")
    source_scope = "/".join(parts[:packed_idx]) or "."
    variant = parts[packed_idx + 1]
    budget_part = parts[packed_idx + 2]
    budget = int(str(budget_part).replace("budget_", ""))
    task = path.stem
    return source_scope, variant, budget, task


def _percentile(values: Sequence[int], percentile: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    rank = (len(sorted_values) - 1) * percentile
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = rank - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _mean(values: Sequence[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _token_count(tokenizer: Any, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False).input_ids)


def build_audit(run_dir: Path, model_name_or_path: str) -> List[Dict[str, Any]]:
    tokenizer = _load_tokenizer(model_name_or_path)
    groups: Dict[Tuple[str, str, int, str], List[Dict[str, Any]]] = defaultdict(list)
    for path in _packed_files(run_dir):
        source_scope, variant, budget, task = _parse_packed_path(run_dir, path)
        for row in _read_jsonl(path):
            packed_text = str(row.get("packed_text") or row.get("packed_prompt") or "")
            if not packed_text:
                continue
            model_count = _token_count(tokenizer, packed_text)
            stored_count = int(row.get("packed_token_count", 0) or 0)
            truncated = bool(row.get("truncated", False))
            groups[(source_scope, variant, budget, task)].append({
                "model_token_count": model_count,
                "stored_packed_token_count": stored_count,
                "over_budget": model_count > budget,
                "over_budget_untruncated": model_count > budget and not truncated,
                "truncated": truncated,
            })

    rows: List[Dict[str, Any]] = []
    for (source_scope, variant, budget, task), items in sorted(groups.items()):
        model_counts = [int(item["model_token_count"]) for item in items]
        stored_counts = [int(item["stored_packed_token_count"]) for item in items]
        exact_matches = sum(
            int(item["model_token_count"]) == int(item["stored_packed_token_count"])
            for item in items
        )
        rows.append({
            "source_scope": source_scope,
            "variant": variant,
            "budget": budget,
            "task": task,
            "num_samples": len(items),
            "model_token_count_mean": round(_mean(model_counts), 4),
            "model_token_count_p95": round(_percentile(model_counts, 0.95), 4),
            "model_token_count_max": max(model_counts) if model_counts else 0,
            "stored_packed_token_count_mean": round(_mean(stored_counts), 4),
            "over_budget_samples": sum(1 for item in items if item["over_budget"]),
            "over_budget_untruncated_samples": sum(1 for item in items if item["over_budget_untruncated"]),
            "truncated_samples": sum(1 for item in items if item["truncated"]),
            "stored_count_exact_match_rate": round(exact_matches / max(1, len(items)), 6),
            "original_packed_token_counter": "bridgekv.evidence_text_packing.estimate_token_count regex tokenizer",
            "generation_tokenizer": model_name_or_path,
            "same_tokenizer_as_generation": "False; stored packed_token_count is an offline regex estimate",
        })
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", default="results/bridgepack_paper_20260618_070828")
    parser.add_argument("--model_name_or_path", default="")
    parser.add_argument(
        "--out_csv",
        default="results/bridgepack_paper_20260618_070828/paper_tables/11_model_tokenizer_budget_audit.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir does not exist: {run_dir}")
    model_name_or_path = _detect_model(run_dir, args.model_name_or_path)
    rows = build_audit(run_dir, model_name_or_path)
    fields = [
        "source_scope",
        "variant",
        "budget",
        "task",
        "num_samples",
        "model_token_count_mean",
        "model_token_count_p95",
        "model_token_count_max",
        "stored_packed_token_count_mean",
        "over_budget_samples",
        "over_budget_untruncated_samples",
        "truncated_samples",
        "stored_count_exact_match_rate",
        "original_packed_token_counter",
        "generation_tokenizer",
        "same_tokenizer_as_generation",
    ]
    _write_csv(Path(args.out_csv), rows, fields)
    print(f"Wrote {len(rows)} audit rows to {args.out_csv}")


if __name__ == "__main__":
    main()
