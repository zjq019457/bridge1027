#!/usr/bin/env python3
"""Summarize compact overhead/budget statistics for exact-token BridgePack runs."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _mean(values: Sequence[float]) -> float | None:
    return statistics.mean(values) if values else None


def _percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    vals = sorted(values)
    idx = (len(vals) - 1) * q
    lo = int(idx)
    hi = min(lo + 1, len(vals) - 1)
    weight = idx - lo
    return vals[lo] * (1.0 - weight) + vals[hi] * weight


def _fmt(value: Any) -> str:
    if value is None or value == "":
        return "TODO"
    return f"{float(value):.4f}".rstrip("0").rstrip(".")


def _maybe_float(value: Any) -> float | None:
    try:
        if value in ("", None):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _prediction_lengths(exact_dir: Path, variant: str) -> List[float]:
    values: List[float] = []
    for path in sorted((exact_dir / "predictions" / variant).glob("*.jsonl")):
        for row in _read_jsonl(path):
            length = _maybe_float(row.get("length"))
            if length is not None:
                values.append(length)
    return values


def _packed_rows(exact_dir: Path, variant: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    root = exact_dir / "packed_prompts_build" / "packed_prompts" / variant
    for path in sorted(root.glob("budget_*/*.jsonl")):
        rows.extend(_read_jsonl(path))
    return rows


def _summarize_rows(exact_dir: Path, stats: Sequence[Mapping[str, str]]) -> List[Dict[str, Any]]:
    variants = sorted({row["variant"] for row in stats})
    rows: List[Dict[str, Any]] = []
    for variant in variants:
        budget_rows = [row for row in stats if row["variant"] == variant]
        packed = _packed_rows(exact_dir, variant)
        packed_counts: List[float] = []
        source_counts: List[float] = []
        ratios: List[float] = []
        for row in packed:
            count = _maybe_float(row.get("model_token_count", row.get("packed_token_count")))
            source = _maybe_float(row.get("source_token_count"))
            ratio = _maybe_float(row.get("compression_ratio"))
            if count is not None:
                packed_counts.append(count)
            if source is not None:
                source_counts.append(source)
            if ratio is not None:
                ratios.append(ratio)
        input_lengths = _prediction_lengths(exact_dir, variant)
        rows.append({
            "variant": variant,
            "num_samples": sum(int(row.get("num_samples", 0) or 0) for row in budget_rows),
            "num_tasks": len({row.get("task") for row in budget_rows}),
            "packed_context_token_count_mean": _fmt(_mean(packed_counts)),
            "packed_context_token_count_p95": _fmt(_percentile(packed_counts, 0.95)),
            "packed_context_token_count_max": _fmt(max(packed_counts) if packed_counts else None),
            "source_context_token_count_mean": _fmt(_mean(source_counts)),
            "compression_ratio_mean": _fmt(_mean(ratios)),
            "generation_input_length_mean": _fmt(_mean(input_lengths)),
            "generation_input_length_p95": _fmt(_percentile(input_lengths, 0.95)),
            "packing_time_per_sample_mean": "TODO",
            "packing_time_per_sample_p95": "TODO",
            "timing_source": "TODO: no packing timing log found",
            "exact_qwen_tokenizer": all(str(row.get("exact_model_token_budget")) == "True" for row in budget_rows),
        })
    return rows


def summarize(exact_dir: Path, paper_dir: Path) -> List[Dict[str, Any]]:
    stats = _read_csv(exact_dir / "paper_tables" / "00_model_token_budget_stats.csv")
    rows = _summarize_rows(exact_dir, stats)
    out_csv = exact_dir / "paper_tables" / "08_packing_overhead.csv"
    fields = [
        "variant", "num_samples", "num_tasks", "packed_context_token_count_mean",
        "packed_context_token_count_p95", "packed_context_token_count_max",
        "source_context_token_count_mean", "compression_ratio_mean",
        "generation_input_length_mean", "generation_input_length_p95",
        "packing_time_per_sample_mean", "packing_time_per_sample_p95",
        "timing_source", "exact_qwen_tokenizer",
    ]
    _write_csv(out_csv, rows, fields)
    main = next((row for row in rows if row["variant"] == "lexical_bridgepack_path_order"), rows[0] if rows else {})
    md = [
        "# Exact-Token Packing Overhead Summary",
        "",
        "This summary reads existing exact-token packed prompts and predictions. No model was loaded and no generation was run.",
        "",
        f"- Variants summarized: `{len(rows)}`.",
        f"- Main method samples: `{main.get('num_samples', 'TODO')}`.",
        f"- Main method packed-token mean/p95/max: `{main.get('packed_context_token_count_mean', 'TODO')}` / `{main.get('packed_context_token_count_p95', 'TODO')}` / `{main.get('packed_context_token_count_max', 'TODO')}`.",
        f"- Main method source-token mean: `{main.get('source_context_token_count_mean', 'TODO')}`.",
        f"- Main method compression-ratio mean: `{main.get('compression_ratio_mean', 'TODO')}`.",
        f"- Main method generation input length mean/p95: `{main.get('generation_input_length_mean', 'TODO')}` / `{main.get('generation_input_length_p95', 'TODO')}`.",
        "- Packing time per sample: TODO. No timing log was found in the exact-token run.",
        "- Exact Qwen tokenizer budget: True for all summarized rows.",
    ]
    _write_text(exact_dir / "paper_tables" / "08_packing_overhead.md", "\n".join(md))
    paragraph = (
        "The exact-token run stores compact packing statistics without timing instrumentation. "
        f"For Lexical-BridgePack, the packed context has mean/p95/max token counts of "
        f"{main.get('packed_context_token_count_mean', 'TODO')}/"
        f"{main.get('packed_context_token_count_p95', 'TODO')}/"
        f"{main.get('packed_context_token_count_max', 'TODO')} under the Qwen tokenizer. "
        f"The mean source-context token count is {main.get('source_context_token_count_mean', 'TODO')}, "
        f"and the mean stored compression ratio is {main.get('compression_ratio_mean', 'TODO')}. "
        "Packing-time overhead remains TODO because no timing log was recorded for the exact-token run."
    )
    _write_text(paper_dir / "overhead_paragraph.md", paragraph)
    return rows


def summarize_budget_ablation(ablation_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for budget_dir in sorted(ablation_dir.glob("budget_*")):
        if not budget_dir.is_dir():
            continue
        try:
            budget = int(budget_dir.name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        stats = _read_csv(budget_dir / "paper_tables" / "00_model_token_budget_stats.csv")
        for row in _summarize_rows(budget_dir, stats):
            out = {"budget": budget}
            out.update(row)
            rows.append(out)
    fields = [
        "budget", "variant", "num_samples", "num_tasks",
        "packed_context_token_count_mean", "packed_context_token_count_p95",
        "packed_context_token_count_max", "source_context_token_count_mean",
        "compression_ratio_mean", "generation_input_length_mean",
        "generation_input_length_p95", "packing_time_per_sample_mean",
        "packing_time_per_sample_p95", "timing_source", "exact_qwen_tokenizer",
    ]
    out_dir = ablation_dir / "paper_tables"
    _write_csv(out_dir / "07_budget_packing_overhead.csv", rows, fields)
    by_budget = sorted({row["budget"] for row in rows})
    main_rows = [row for row in rows if row.get("variant") == "lexical_bridgepack_path_order"]
    lines = [
        "# Budget Ablation Packing Overhead Summary",
        "",
        "This summary reads completed exact-token budget-ablation packed prompts and predictions. No additional model was loaded.",
        "",
        f"- Budgets summarized: `{','.join(str(item) for item in by_budget)}`.",
        "- Packing time per sample: TODO. No timing log was found in the budget-ablation runs.",
    ]
    for row in sorted(main_rows, key=lambda item: int(item["budget"])):
        lines.append(
            f"- Lexical-BridgePack budget {row['budget']} packed-token mean/p95/max: "
            f"`{row['packed_context_token_count_mean']}` / `{row['packed_context_token_count_p95']}` / "
            f"`{row['packed_context_token_count_max']}`."
        )
    _write_text(out_dir / "07_budget_packing_overhead.md", "\n".join(lines))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exact_dir", required=True)
    parser.add_argument("--paper_dir", default="paper/aaai2027_bridgepack_exact")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exact_dir = Path(args.exact_dir)
    paper_dir = Path(args.paper_dir)
    if (exact_dir / "paper_tables" / "00_model_token_budget_stats.csv").exists():
        summarize(exact_dir, paper_dir)
        print(f"OUT_CSV={exact_dir / 'paper_tables' / '08_packing_overhead.csv'}")
        print(f"OUT_MD={exact_dir / 'paper_tables' / '08_packing_overhead.md'}")
    else:
        summarize_budget_ablation(exact_dir)
        print(f"OUT_CSV={exact_dir / 'paper_tables' / '07_budget_packing_overhead.csv'}")
        print(f"OUT_MD={exact_dir / 'paper_tables' / '07_budget_packing_overhead.md'}")


if __name__ == "__main__":
    main()
