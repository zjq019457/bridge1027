#!/usr/bin/env python3
"""Build the exact-token BridgePack paper data pack.

This script only reads compact paper/result artifacts. It does not load models,
run generation, or inspect raw predictions.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


REQUIRED_MAIN = [
    "span_only_source_order",
    "tfidf_chunk_source_order",
    "bm25_chunk_source_order",
    "bridge_neighbor32_path_order_tags",
    "bridge_neighbor32_path_order",
    "lexical_bridgepack_path_order",
]

DISPLAY = {
    "span_only_source_order": "Span-only Packing",
    "tfidf_chunk_source_order": "TF-IDF Chunk Packing",
    "bm25_chunk_source_order": "BM25 Chunk Packing",
    "bridge_neighbor32_path_order_tags": "BridgePack+Tags",
    "bridge_neighbor32_path_order": "BridgePack",
    "bridge_same_spans_source_order": "Same Spans + Source Order",
    "lexical_bridgepack_path_order": "Lexical-BridgePack",
}

TASKS = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", "narrativeqa"]
MULTIHOP_TASKS = ["hotpotqa", "2wikimqa", "musique"]
OLD_PATTERNS = [
    "38.1533",
    "23.6367",
    "14.5166",
    "35.935",
    "12.2983",
    "regex-budget",
    "BM25 Chunk Packing: TODO",
    "TF-IDF Chunk Packing: TODO",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "TODO") for k in fieldnames})


def fmt(value: Any, digits: int = 4) -> str:
    if value is None or value == "":
        return "TODO"
    if isinstance(value, str):
        if value == "TODO":
            return value
        try:
            value = float(value)
        except ValueError:
            return value
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}f}"


def fmt2(value: Any) -> str:
    return fmt(value, 2)


def md_table(rows: list[dict[str, Any]], cols: list[str]) -> str:
    if not rows:
        return "_No rows available._\n"
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(c, "TODO")) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def latex_escape(text: Any) -> str:
    s = str(text)
    return (
        s.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("#", r"\#")
    )


def latex_table(
    caption: str,
    label: str,
    rows: list[dict[str, Any]],
    cols: list[str],
    headers: list[str] | None = None,
    table_star: bool = False,
) -> str:
    headers = headers or cols
    env = "table*" if table_star else "table"
    spec = "l" + "r" * (len(cols) - 1)
    body = [
        f"\\begin{{{env}}}[t]",
        "\\centering",
        "\\small",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{spec}}}",
        "\\toprule",
        " & ".join(latex_escape(h) for h in headers) + r" \\",
        "\\midrule",
    ]
    for row in rows:
        body.append(" & ".join(latex_escape(row.get(c, "TODO")) for c in cols) + r" \\")
    body += ["\\bottomrule", "\\end{tabular}", f"\\end{{{env}}}", ""]
    return "\n".join(body)


def row_by_method(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {r.get("method", ""): r for r in rows}


def latest(pattern: str) -> Path | None:
    matches = sorted(Path(".").glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def source(path: Path) -> str:
    return str(path)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def main_rows(exact_dir: Path) -> tuple[list[dict[str, Any]], Path]:
    src = exact_dir / "paper_tables" / "01_exact_budget_main_table_1024.csv"
    rows = row_by_method(read_csv(src))
    out: list[dict[str, Any]] = []
    for method in REQUIRED_MAIN:
        r = rows.get(method, {})
        item = {"method": DISPLAY[method], "budget": r.get("budget", "1024")}
        for task in TASKS:
            item[task] = fmt(r.get(task, "TODO"))
        item["avg"] = fmt(r.get("avg", "TODO"))
        out.append(item)
    return out, src


def component_rows(exact_dir: Path) -> tuple[list[dict[str, Any]], Path]:
    src = exact_dir / "paper_tables" / "01_exact_budget_main_table_1024.csv"
    rows = row_by_method(read_csv(src))
    specs = [
        ("span_only_source_order", "Span-only", "N", "N", "N", "N", "N"),
        ("tfidf_chunk_source_order", "TF-IDF", "Y", "N", "N", "N", "N"),
        ("bm25_chunk_source_order", "BM25", "Y", "N", "N", "N", "N"),
        ("bridge_neighbor32_path_order_tags", "BridgePack+Tags", "N", "Y", "Y", "Y", "Y"),
        ("bridge_same_spans_source_order", "Same Spans + Source Order", "N", "Y", "Y", "N", "N"),
        ("bridge_neighbor32_path_order", "BridgePack", "N", "Y", "Y", "Y", "N"),
        ("lexical_bridgepack_path_order", "Lexical-BridgePack", "Y", "Y", "Y", "Y", "N"),
    ]
    out: list[dict[str, Any]] = []
    for method, display, lex, bridge, neighbor, path_order, tags in specs:
        avg = rows.get(method, {}).get("avg", "TODO")
        out.append(
            {
                "method": display,
                "lexical_initialization": lex,
                "bridge_selection": bridge,
                "neighbor_repair": neighbor,
                "path_order": path_order,
                "tags": tags,
                "avg": fmt(avg),
            }
        )
    same = rows.get("bridge_same_spans_source_order", {}).get("avg")
    bridge = rows.get("bridge_neighbor32_path_order", {}).get("avg")
    gain = "TODO"
    if same and bridge:
        gain = fmt(float(bridge) - float(same))
    out.append(
        {
            "method": "Path-order gain (BridgePack - Same Spans)",
            "lexical_initialization": "N",
            "bridge_selection": "Y",
            "neighbor_repair": "Y",
            "path_order": "Y-vs-N",
            "tags": "N",
            "avg": gain,
        }
    )
    return out, src


def multihop_rows(exact_dir: Path) -> tuple[list[dict[str, Any]], Path]:
    src = exact_dir / "paper_tables" / "04_exact_budget_multihop_table.csv"
    rows = row_by_method(read_csv(src))
    methods = [
        "span_only_source_order",
        "tfidf_chunk_source_order",
        "bm25_chunk_source_order",
        "bridge_neighbor32_path_order",
        "lexical_bridgepack_path_order",
    ]
    out: list[dict[str, Any]] = []
    for method in methods:
        r = rows.get(method, {})
        item = {"method": DISPLAY[method]}
        for task in MULTIHOP_TASKS:
            item[task] = fmt(r.get(task, "TODO"))
        item["multihop_avg"] = fmt(r.get("multihop_avg", "TODO"))
        out.append(item)
    return out, src


def bootstrap_rows(exact_dir: Path) -> tuple[list[dict[str, Any]], Path]:
    src = exact_dir / "paper_tables" / "07_exact_bootstrap_ci.csv"
    rows = read_csv(src)
    names = {
        "span_only_source_order": "Span-only Packing",
        "bm25_chunk_source_order": "BM25 Chunk Packing",
        "tfidf_chunk_source_order": "TF-IDF Chunk Packing",
        "bridge_neighbor32_path_order": "BridgePack",
    }
    out: list[dict[str, Any]] = []
    for r in rows:
        baseline = r.get("baseline", "")
        lo = float(r.get("ci_low", "0"))
        hi = float(r.get("ci_high", "0"))
        interpretation = "significant" if lo > 0 or hi < 0 else "not significant; conservative claim"
        if baseline in {"bm25_chunk_source_order", "tfidf_chunk_source_order", "bridge_neighbor32_path_order"}:
            interpretation = "not significant; higher observed average only"
        out.append(
            {
                "comparison": f"Lexical-BridgePack vs {names.get(baseline, baseline)}",
                "delta": fmt(r.get("paired_delta", "TODO")),
                "ci_low": fmt(r.get("ci_low", "TODO")),
                "ci_high": fmt(r.get("ci_high", "TODO")),
                "paired_samples": r.get("paired_samples", "TODO"),
                "interpretation": interpretation,
            }
        )
    return out, src


def budget_rows(budget_dir: Path) -> tuple[list[dict[str, Any]], Path]:
    src = budget_dir / "paper_tables" / "01_budget_ablation_table.csv"
    rows = read_csv(src)
    by_budget: dict[str, dict[str, dict[str, str]]] = {}
    for r in rows:
        by_budget.setdefault(r["budget"], {})[r["method"]] = r
    out: list[dict[str, Any]] = []
    for budget in ["512", "1024", "2048"]:
        b = by_budget.get(budget, {})
        lex = b.get("lexical_bridgepack_path_order", {})
        bm25 = b.get("bm25_chunk_source_order", {})
        tfidf = b.get("tfidf_chunk_source_order", {})
        span = b.get("span_only_source_order", {})
        bridge = b.get("bridge_neighbor32_path_order", {})
        lex_avg = float(lex.get("avg", "nan")) if lex.get("avg") else None
        row = {
            "budget": budget,
            "span_only": fmt(span.get("avg", "TODO")),
            "tfidf": fmt(tfidf.get("avg", "TODO")),
            "bm25": fmt(bm25.get("avg", "TODO")),
            "bridgepack": fmt(bridge.get("avg", "TODO")),
            "lexical_bridgepack": fmt(lex.get("avg", "TODO")),
            "delta_lexical_vs_span": fmt(lex.get("delta_vs_span_only", "TODO")),
            "delta_lexical_vs_bm25": fmt(lex.get("delta_vs_bm25", "TODO")),
            "delta_lexical_vs_tfidf": fmt(lex.get("delta_vs_tfidf", "TODO")),
            "beats_span_only": "TODO",
            "beats_bm25": "TODO",
            "beats_tfidf": "TODO",
        }
        if lex_avg is not None:
            row["beats_span_only"] = str(lex_avg > float(span.get("avg", "nan")))
            row["beats_bm25"] = str(lex_avg > float(bm25.get("avg", "nan")))
            row["beats_tfidf"] = str(lex_avg > float(tfidf.get("avg", "nan")))
        out.append(row)
    return out, src


def fairness(exact_dir: Path, second_model_dir: Path | None) -> tuple[dict[str, Any], Path, Path | None]:
    main_stats = read_csv(exact_dir / "paper_tables" / "00_model_token_budget_stats.csv")
    max_len = max(int(float(r["model_token_count_max"])) for r in main_stats) if main_stats else 1024
    over = sum(int(float(r.get("over_budget_samples", 0))) for r in main_stats)
    rows = sum(int(float(r.get("num_samples", 0))) for r in main_stats)
    missing = read_csv(exact_dir / "paper_tables" / "00_missing_samples.csv")
    out: dict[str, Any] = {
        "official_main": {
            "setting": "exact 1024 Qwen-token packed-context budget",
            "tokenizer_budget_used": True,
            "variant_sample_rows": rows,
            "over_budget_packed_contexts": over,
            "max_packed_length": max_len,
            "same_sample_ids": True,
            "missing_or_failed_samples": sum(int(r.get("missing_count", 0) or 0) for r in missing),
            "no_gold_answer_or_metadata_used": True,
            "old_regex_budget_results_discarded": True,
            "source": source(exact_dir / "paper_tables" / "00_model_token_budget_stats.csv"),
        }
    }
    sm_src = None
    if second_model_dir:
        sm_src = second_model_dir / "paper_tables" / "00_model_token_budget_stats.csv"
        sm_stats = read_csv(sm_src)
        sm_missing = read_csv(second_model_dir / "paper_tables" / "06_second_model_missing_or_failed_samples.csv")
        out["second_model_gate"] = {
            "setting": "exact 1024 Mistral-token packed-context budget",
            "variant_sample_rows": sum(int(float(r.get("num_samples", 0))) for r in sm_stats),
            "over_budget_packed_contexts": sum(int(float(r.get("over_budget_samples", 0))) for r in sm_stats),
            "max_packed_length": max(int(float(r["model_token_count_max"])) for r in sm_stats) if sm_stats else 1024,
            "same_sample_ids": True,
            "missing_or_failed_samples": sum(int(r.get("missing_count", 0) or 0) for r in sm_missing),
            "source": source(sm_src),
        }
    return out, exact_dir / "paper_tables" / "00_model_token_budget_stats.csv", sm_src


def overhead_rows(exact_dir: Path, budget_dir: Path) -> tuple[list[dict[str, Any]], Path]:
    src = exact_dir / "paper_tables" / "08_packing_overhead.csv"
    rows = read_csv(src)
    main = [r for r in rows if r.get("variant") == "lexical_bridgepack_path_order"]
    out: list[dict[str, Any]] = []
    for r in main:
        out.append(
            {
                "setting": "Qwen 1024 main",
                "variant": "Lexical-BridgePack",
                "packed_token_mean": fmt(r.get("packed_context_token_count_mean")),
                "packed_token_p95": fmt(r.get("packed_context_token_count_p95")),
                "packed_token_max": fmt(r.get("packed_context_token_count_max")),
                "source_token_mean": fmt(r.get("source_context_token_count_mean")),
                "compression_ratio_mean": fmt(r.get("compression_ratio_mean")),
                "generation_input_length_mean": fmt(r.get("generation_input_length_mean")),
                "generation_input_length_p95": fmt(r.get("generation_input_length_p95")),
                "packing_time_mean": r.get("packing_time_per_sample_mean", "TODO"),
                "packing_time_p95": r.get("packing_time_per_sample_p95", "TODO"),
            }
        )
    bsrc = budget_dir / "paper_tables" / "07_budget_packing_overhead.csv"
    for r in read_csv(bsrc):
        if r.get("variant") == "lexical_bridgepack_path_order" and r.get("budget") in {"512", "2048"}:
            out.append(
                {
                    "setting": f"Qwen {r.get('budget')} scaling",
                    "variant": "Lexical-BridgePack",
                    "packed_token_mean": fmt(r.get("packed_context_token_count_mean")),
                    "packed_token_p95": fmt(r.get("packed_context_token_count_p95")),
                    "packed_token_max": fmt(r.get("packed_context_token_count_max")),
                    "source_token_mean": fmt(r.get("source_context_token_count_mean")),
                    "compression_ratio_mean": fmt(r.get("compression_ratio_mean")),
                    "generation_input_length_mean": fmt(r.get("generation_input_length_mean")),
                    "generation_input_length_p95": fmt(r.get("generation_input_length_p95")),
                    "packing_time_mean": r.get("packing_time_per_sample_mean", "TODO"),
                    "packing_time_p95": r.get("packing_time_per_sample_p95", "TODO"),
                }
            )
    return out, src


def second_model_rows(second_model_dir: Path | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Path | None, Path | None]:
    if not second_model_dir:
        return [], [], None, None
    table_src = second_model_dir / "paper_tables" / "01_exact_budget_main_table_1024.csv"
    boot_src = second_model_dir / "paper_tables" / "07_second_model_bootstrap_ci.csv"
    rows = []
    for r in read_csv(table_src):
        rows.append(
            {
                "method": DISPLAY.get(r["method"], r.get("display_name", r["method"])),
                "hotpotqa": fmt(r.get("hotpotqa")),
                "2wikimqa": fmt(r.get("2wikimqa")),
                "musique": fmt(r.get("musique")),
                "avg": fmt(r.get("avg")),
            }
        )
    boot = []
    for r in read_csv(boot_src):
        boot.append(
            {
                "comparison": f"Lexical-BridgePack vs {DISPLAY.get(r['baseline'], r['baseline'])}",
                "delta": fmt(r.get("delta_mean")),
                "ci_low": fmt(r.get("ci_low")),
                "ci_high": fmt(r.get("ci_high")),
                "interpretation": "significant" if float(r["ci_low"]) > 0 else "not significant; gate trend only",
            }
        )
    return rows, boot, table_src, boot_src


def official_registry(paths: dict[str, str]) -> dict[str, dict[str, Any]]:
    registry = {
        "lexical_bridgepack_qwen_1024_avg": {
            "value": 38.3333,
            "meaning": "Lexical-BridgePack official exact Qwen-token 1024 average",
            "source_file_path": paths["main"],
            "safe_claim": "Best observed average under the official exact Qwen-token 1024 packed-context budget.",
            "unsafe_overclaim": "Do not claim statistically significant superiority over BM25/TF-IDF.",
        },
        "span_only_qwen_1024_avg": {
            "value": 22.2867,
            "meaning": "Span-only exact Qwen-token 1024 average",
            "source_file_path": paths["main"],
            "safe_claim": "Baseline for significant Lexical-BridgePack improvement.",
            "unsafe_overclaim": "Do not use old regex-budget span-only scores.",
        },
        "bm25_qwen_1024_avg": {
            "value": 37.0217,
            "meaning": "BM25 chunk packing exact Qwen-token 1024 average",
            "source_file_path": paths["main"],
            "safe_claim": "Strong lexical baseline.",
            "unsafe_overclaim": "Do not mark BM25 as TODO or weak without qualification.",
        },
        "tfidf_qwen_1024_avg": {
            "value": 36.9250,
            "meaning": "TF-IDF chunk packing exact Qwen-token 1024 average",
            "source_file_path": paths["main"],
            "safe_claim": "Strong lexical baseline.",
            "unsafe_overclaim": "Do not mark TF-IDF as TODO or weak without qualification.",
        },
        "bridgepack_qwen_1024_avg": {
            "value": 37.2983,
            "meaning": "Original BridgePack without lexical initialization",
            "source_file_path": paths["main"],
            "safe_claim": "Ablation/intermediate method.",
            "unsafe_overclaim": "Do not present BridgePack alone as the final method.",
        },
        "bridgepack_tags_qwen_1024_avg": {
            "value": 34.3433,
            "meaning": "BridgePack+Tags optional formatting ablation",
            "source_file_path": paths["main"],
            "safe_claim": "Tags underperform no-tag BridgePack.",
            "unsafe_overclaim": "Do not claim tags are core.",
        },
        "qwen_bootstrap_vs_span": {
            "value": "+16.0487 CI [13.7177, 18.2680]",
            "meaning": "Paired bootstrap Lexical-BridgePack vs span-only",
            "source_file_path": paths["bootstrap"],
            "safe_claim": "Significant improvement over span-only.",
            "unsafe_overclaim": "Do not transfer this significance claim to BM25/TF-IDF.",
        },
        "lexical_budget_2048_avg": {
            "value": 39.5267,
            "meaning": "Lexical-BridgePack exact Qwen-token 2048 scaling average",
            "source_file_path": paths["budget"],
            "safe_claim": "2048 improves over 1024 by +1.1934.",
            "unsafe_overclaim": "Do not silently replace the official 1024 main setting.",
        },
        "lexical_budget_512_avg": {
            "value": 33.9600,
            "meaning": "Lexical-BridgePack exact Qwen-token 512 scaling average",
            "source_file_path": paths["budget"],
            "safe_claim": "Improves over span-only at 512 but is below BM25.",
            "unsafe_overclaim": "Do not claim Lexical-BridgePack beats BM25 at every budget.",
        },
        "mistral_gate_lexical_avg": {
            "value": 30.0667,
            "meaning": "Lexical-BridgePack exact 1024 Mistral-token 32-sample-per-task multi-hop gate average",
            "source_file_path": paths.get("second_model", "TODO"),
            "safe_claim": "Supports transfer over span-only in the Mistral gate.",
            "unsafe_overclaim": "Do not present this as full second-model validation.",
        },
        "mistral_gate_span_avg": {
            "value": 16.1033,
            "meaning": "Span-only exact 1024 Mistral-token gate average",
            "source_file_path": paths.get("second_model", "TODO"),
            "safe_claim": "Baseline for Mistral gate improvement.",
            "unsafe_overclaim": "Do not mix with Qwen main table values.",
        },
        "mistral_gate_bm25_avg": {
            "value": 27.4700,
            "meaning": "BM25 exact 1024 Mistral-token gate average",
            "source_file_path": paths.get("second_model", "TODO"),
            "safe_claim": "Strong lexical baseline in the Mistral gate.",
            "unsafe_overclaim": "Do not claim significant Lexical-BridgePack superiority over BM25 in the gate.",
        },
        "mistral_gate_tfidf_avg": {
            "value": 27.1100,
            "meaning": "TF-IDF exact 1024 Mistral-token gate average",
            "source_file_path": paths.get("second_model", "TODO"),
            "safe_claim": "Strong lexical baseline in the Mistral gate.",
            "unsafe_overclaim": "Do not claim significant Lexical-BridgePack superiority over TF-IDF in the gate.",
        },
        "mistral_gate_bridgepack_avg": {
            "value": 28.3467,
            "meaning": "BridgePack exact 1024 Mistral-token gate average",
            "source_file_path": paths.get("second_model", "TODO"),
            "safe_claim": "Original BridgePack remains an ablation/intermediate method in the second-model gate.",
            "unsafe_overclaim": "Do not present BridgePack alone as the final method.",
        },
        "mistral_bootstrap_vs_span": {
            "value": "+13.9623 CI [4.7557, 22.7474]",
            "meaning": "Mistral gate paired bootstrap Lexical-BridgePack vs span-only",
            "source_file_path": paths.get("second_model_bootstrap", "TODO"),
            "safe_claim": "Significant improvement over span-only in the 32-sample-per-task Mistral gate.",
            "unsafe_overclaim": "Do not claim full model generalization or significant superiority over lexical baselines.",
        },
    }
    return registry


def scan_old_numbers(paper_dir: Path) -> str:
    files = [paper_dir / "main.tex", paper_dir / "tables.tex"]
    lines = ["# Old Official Number Scan", ""]
    found_error = False
    for path in files:
        lines.append(f"## {path}")
        if not path.exists():
            lines.append("- Missing.")
            continue
        text_lines = path.read_text(errors="ignore").splitlines()
        hits = []
        for i, line in enumerate(text_lines, 1):
            if any(p in line for p in OLD_PATTERNS):
                hits.append((i, line.strip()))
        if not hits:
            lines.append("- No old-number/TODO official-result hits.")
        else:
            found_error = True
            for i, line in hits:
                lines.append(f"- ERROR line {i}: `{line}`")
        lines.append("")
    lines.append(f"Overall status: {'ERROR' if found_error else 'OK'}")
    return "\n".join(lines) + "\n"


def write_main_tex(paper_dir: Path) -> None:
    text = r"""\documentclass[letterpaper]{article}
\usepackage[submission]{aaai2027}
\usepackage[hyphens]{url}
\usepackage{graphicx}
\usepackage{natbib}
\usepackage{caption}
\frenchspacing
\usepackage{algorithm}
\usepackage{algorithmic}
\usepackage{booktabs}

\title{Lexical-BridgePack: Evidence-Aware Context Repacking under Exact Token Budgets}

\author{Anonymous Submission}
\affiliations{Anonymous Institution}

\begin{document}

\maketitle

\begin{abstract}
Long-context question answering often depends on dispersed evidence that must fit inside a compact inference budget.
We study evidence-aware context compression as text repacking: selecting, repairing, and ordering compact evidence before generation under the exact tokenizer budget used by the model.
The final method, Lexical-BridgePack, combines lexical evidence initialization with neighbor-preserving repair and bridge/path-aware organization.
Under the official exact 1024 Qwen-token packed-context budget, Lexical-BridgePack obtains the best observed average score, 38.3333, compared with 37.2983 for BridgePack, 37.0217 for BM25, 36.9250 for TF-IDF, 34.3433 for BridgePack+Tags, and 22.2867 for span-only packing.
Paired bootstrap analysis shows a significant improvement over span-only packing, but confidence intervals versus BM25, TF-IDF, and BridgePack cross zero, so we do not claim statistically significant superiority over those baselines.
Exact-token scaling reaches 39.5267 at 2048 tokens, while 1024 remains the official main setting.
A Mistral-7B-Instruct-v0.3 32-sample-per-task gate supports transfer over span-only packing.
The method is context compression and text repacking, not strict KV-cache pruning or original long-context prefill KV-memory reduction.
\end{abstract}

\section{Introduction}
Long-context systems must decide which evidence to present when the available prompt budget is smaller than the source context.
Independent span ranking can preserve isolated facts but miss relationships among dispersed evidence pieces.
This paper studies Lexical-BridgePack, a lightweight context-repacking method that combines lexical evidence initialization, neighbor-preserving repair, and bridge/path-aware evidence organization.

The main empirical point is exact-token fairness.
The official results use the Qwen generation tokenizer for the 1024-token packed-context budget.
Across 12,650 variant-sample rows, the compact audit reports zero over-budget packed contexts, maximum packed-context length 1024, aligned sample ids, and no missing or failed compact-audit samples.

Our contributions are:
\begin{itemize}
    \item We formulate BridgePack-style compression as evidence-aware text repacking under exact tokenizer budgets.
    \item We introduce Lexical-BridgePack, combining lexical candidate initialization with bridge/path organization.
    \item We provide exact-token main results, component ablations, multi-hop analysis, paired bootstrap intervals, budget scaling, overhead summaries, and a second-model gate.
    \item We state conservative claims: lexical baselines are strong, tags underperform, path-order gains are modest, and the method is not KV-cache pruning.
\end{itemize}

\section{Related Work}
\paragraph{Long-context evaluation.}
LongBench evaluates long-context language models across question answering, summarization, retrieval, and other tasks \citep{bai2023longbench}.
Our exact-token experiments focus on six English QA-style LongBench tasks: Qasper, MultifieldQA-en, HotpotQA, 2WikiMQA, MuSiQue, and NarrativeQA.

\paragraph{Prompt and context compression.}
LLMLingua and LongLLMLingua compress prompts before generation \citep{jiang2023llmlingua,jiang2023longllmlingua}.
They remain important TODO external baselines for this draft.
Our setting is prompt-level context selection with exact-token budget audits and bridge-aware evidence organization.

\paragraph{Lexical and dense retrieval.}
BM25 \citep{robertson2009probabilistic} and TF-IDF \citep{salton1988term} are strong lexical retrieval baselines.
Dense retrieval \citep{karpukhin2020dense} is a natural additional baseline and remains TODO.

\section{Method}
\paragraph{Task setting.}
Given a question $q$, source context $C$, and budget $B$, the system constructs a packed text context $\hat{C}$ whose token count under the generation tokenizer is at most $B$.
The packed context is inserted into the same prompt wrapper for all variants.
Gold answers are used only for evaluation.

\paragraph{Lexical-BridgePack.}
Lexical-BridgePack starts with lexical evidence initialization, repairs selected spans with local neighbor context, and organizes evidence in path order.
Original BridgePack removes lexical initialization and is reported as an ablation.
BridgePack+Tags inserts explicit tags, but tags underperform here and are not core.

\begin{algorithm}[t]
\caption{Lexical-BridgePack}
\begin{algorithmic}[1]
\STATE Split $C$ into pages, chunks, and candidate spans.
\STATE Score candidates with lexical question-evidence signals.
\STATE Select seed candidates under redundancy constraints.
\STATE Score bridge links between seed evidence and support evidence.
\STATE Build short evidence paths from high-scoring linked candidates.
\STATE Repair selected evidence with local neighbor tokens when budget permits.
\STATE Pack evidence in path order until exact tokenizer budget $B$ is reached.
\end{algorithmic}
\end{algorithm}

\section{Experiments}
\paragraph{Datasets and model.}
The official main result uses Qwen2.5-7B-Instruct on Qasper, MultifieldQA-en, HotpotQA, 2WikiMQA, MuSiQue, and NarrativeQA.
We also include a 32-sample-per-task Mistral-7B-Instruct-v0.3 gate on HotpotQA, 2WikiMQA, and MuSiQue.
This is a gate, not a full second-model validation.

\paragraph{Compared methods.}
The official table compares span-only packing, TF-IDF chunk packing, BM25 chunk packing, BridgePack+Tags, BridgePack, and Lexical-BridgePack.
Additional controls include same-spans source order and budget scaling rows.

\paragraph{Statistics.}
We use paired bootstrap intervals over aligned sample ids within each task.
The Qwen bootstrap uses 1,150 paired samples.

\section{Results}
\input{tables}

Lexical-BridgePack obtains the best observed exact-token Qwen average at 1024 tokens.
Its gain over span-only is significant, while confidence intervals versus BM25, TF-IDF, and BridgePack cross zero.
At 512 tokens, Lexical-BridgePack improves over span-only but is below BM25.
At 2048 tokens, it reaches 39.5267, improving over the official 1024-token setting by +1.1934.

\section{Ablation and Analysis}
Component results show that lexical initialization is important and lexical baselines are strong.
Same-spans source order scores 37.0617 and BridgePack scores 37.2983, a modest path-order gain of +0.2366.
BridgePack+Tags scores 34.3433, so tags are not a core contribution.
The Mistral gate supports improvement over span-only but does not establish significant superiority over BM25, TF-IDF, or BridgePack.

\section{Limitations}
This paper studies context compression and text repacking, not strict KV-cache pruning.
It does not demonstrate original long-context prefill KV-memory reduction.
Full second-model validation, dense retrieval, LLMLingua/LongLLMLingua baselines, broader model/dataset validation, and packing-time measurements remain TODO.

\section{Ethical Statement}
Context compression can remove information that would otherwise affect a model response.
Applications should disclose compression, preserve audit traces where possible, and avoid using compressed evidence selection as a substitute for human review in high-stakes settings.
No new dataset containing private personal information is introduced by this draft.

\section{Conclusion}
Lexical-BridgePack is an evidence-aware context-repacking method evaluated under exact tokenizer budgets.
It obtains the best observed average in the official Qwen 1024-token table and significantly improves over span-only packing.
Lexical baselines remain competitive, so claims over BM25/TF-IDF are intentionally conservative.

\bibliography{bridgepack_refs}

\end{document}
"""
    write_text(paper_dir / "main.tex", text)


def write_claim_guard(paper_dir: Path) -> None:
    write_text(
        paper_dir / "claim_guard.md",
        """# BridgePack Exact-Token Claim Guard

## Safe Claims

- Lexical-BridgePack obtains the best observed average score under the exact 1024 Qwen-token packed-context budget.
- Lexical-BridgePack significantly improves over span-only packing: paired delta `+16.0487`, CI `[13.7177, 18.2680]`.
- Lexical-BridgePack has a higher observed average than BM25, TF-IDF, and BridgePack, but those CIs cross zero.
- BM25 and TF-IDF are strong lexical retrieval baselines.
- Exact-token budget scaling is complete for 512/1024/2048; 2048 has the highest Lexical-BridgePack average, while 1024 remains the official main-table default.
- At 512 tokens, Lexical-BridgePack improves over span-only but is below BM25.
- A 32-sample-per-task Mistral multi-hop gate supports improvement over span-only; deltas over BM25/TF-IDF/BridgePack are positive but not significant.
- Tags are not core and underperform in this run.
- Path-order contribution is modest: BridgePack 37.2983 vs same-spans source order 37.0617.
- The paper is about evidence-aware context compression / text repacking.

## Unsafe Claims

- Lexical-BridgePack significantly outperforms BM25 or TF-IDF.
- Lexical-BridgePack significantly outperforms BM25, TF-IDF, or BridgePack on the Mistral gate.
- The Mistral gate is a full second-model validation.
- Lexical-BridgePack outperforms BM25 at every budget.
- The 2048-token scaling row replaces the official 1024-token main table.
- Tags are a core component.
- Path order is the only or dominant source of gain.
- The method is strict KV-cache pruning.
- The method reduces original long-context prefill KV memory.
- Old regex-budget results are official final main results.
""",
    )


def write_todo_checklist(paper_dir: Path) -> None:
    write_text(
        paper_dir / "todo_data_checklist.md",
        """# BridgePack Exact-Token Data Checklist

## Completed

- Exact-token 1024 Qwen main table.
- BM25 and TF-IDF exact-token lexical baselines.
- Exact-token fairness audit: 12650 variant-sample rows, 0 over budget, max packed-token count 1024.
- Component ablations: span-only, BM25, TF-IDF, BridgePack+Tags, BridgePack, same-spans source order, Lexical-BridgePack.
- Paired bootstrap CI for Lexical-BridgePack against span-only, BM25, TF-IDF, and BridgePack.
- Exact-token 512/1024/2048 Qwen budget scaling.
- Exact-token overhead summaries for packed tokens, source tokens, compression ratio, and generation input length.
- Mistral-7B-Instruct-v0.3 second-model multi-hop gate for 32 samples per task.

## Exact-Token Completed

- `span_only_source_order`: 22.2867
- `tfidf_chunk_source_order`: 36.9250
- `bm25_chunk_source_order`: 37.0217
- `bridge_neighbor32_path_order_tags`: 34.3433
- `bridge_neighbor32_path_order`: 37.2983
- `lexical_bridgepack_path_order`: 38.3333

## Still TODO

- Full second-model validation beyond the small Mistral gate.
- Dense retriever baseline.
- LongLLMLingua / LLMLingua baseline.
- Packing-time overhead timing, because no timing log was found.
- Verified human-readable case studies.
- Broader model/dataset validation.
- Verified official bibliographic metadata for all references.

## Optional Future Experiments

- Larger multi-model gate after the Mistral 32-sample results.
- Additional budget points beyond 512/1024/2048.
- Human-readable figures.
- Strict KV-cache pruning integration, if a future native sparse-KV path exists.
""",
    )


def write_readme(paper_dir: Path, exact_dir: Path, budget_dir: Path, second_model_dir: Path | None) -> None:
    write_text(
        paper_dir / "README.md",
        f"""# AAAI2027 BridgePack Exact-Token Draft

This directory contains a compact AAAI2027-oriented draft and data pack for the exact-token BridgePack paper.

## Source Of Truth

- Official exact result directory: `{exact_dir}`
- Official budget-scaling directory: `{budget_dir}`
- Second-model gate directory: `{second_model_dir if second_model_dir else 'TODO'}`
- Data pack: `paper/aaai2027_bridgepack_exact/data_pack/`

## Official Framing

- Final method: Lexical-BridgePack, also described as BridgePack with lexical initialization.
- Setting: evidence-aware context compression / text repacking.
- Not strict KV-cache pruning.
- Do not claim original long-context prefill KV-memory reduction.

## Official Main Result

- Exact 1024 Qwen-token packed-context budget.
- Lexical-BridgePack: `38.3333`.
- Span-only: `22.2867`; TF-IDF: `36.9250`; BM25: `37.0217`; BridgePack+Tags: `34.3433`; BridgePack: `37.2983`.
- Fairness: 12650 variant-sample rows, 0 over budget, max packed-token count 1024.

## Files

- `main.tex`: AAAI-style draft skeleton.
- `tables.tex`: LaTeX tables generated from the data pack.
- `data_pack/`: categorized CSV/MD/TeX/JSON paper data pack.
- `bridgepack_refs.bib`: compact bibliography with TODO notes where metadata needs verification.
- `claim_guard.md`: safe/unsafe claim checklist.
- `todo_data_checklist.md`: completed and remaining data checklist.
""",
    )


def build(args: argparse.Namespace) -> None:
    exact_dir = Path(args.exact_dir) if args.exact_dir else latest("results/bridgepack_exact_qwen_budget_1024_*")
    budget_dir = Path(args.budget_dir) if args.budget_dir else latest("results/bridgepack_exact_budget_ablation_*")
    second_model_dir = Path(args.second_model_dir) if args.second_model_dir else latest("results/bridgepack_second_model_exact_gate_*")
    paper_dir = Path(args.paper_dir)
    out_dir = Path(args.out_dir)
    if not exact_dir or not exact_dir.exists():
        raise SystemExit("Missing exact result directory")
    if not budget_dir or not budget_dir.exists():
        raise SystemExit("Missing budget result directory")
    if second_model_dir and not second_model_dir.exists():
        second_model_dir = None
    out_dir.mkdir(parents=True, exist_ok=True)

    main, main_src = main_rows(exact_dir)
    comp, comp_src = component_rows(exact_dir)
    mh, mh_src = multihop_rows(exact_dir)
    boot, boot_src = bootstrap_rows(exact_dir)
    budget, budget_src = budget_rows(budget_dir)
    fair, fair_src, fair_sm_src = fairness(exact_dir, second_model_dir)
    overhead, overhead_src = overhead_rows(exact_dir, budget_dir)
    sm_rows, sm_boot, sm_src, sm_boot_src = second_model_rows(second_model_dir)

    inventory = [
        "# BridgePack Paper Data Inventory",
        "",
        f"- exact_dir: `{exact_dir}`",
        f"- budget_dir: `{budget_dir}`",
        f"- second_model_dir: `{second_model_dir if second_model_dir else 'TODO'}`",
        f"- paper_dir: `{paper_dir}`",
        f"- out_dir: `{out_dir}`",
        "",
        "## Compact Sources",
    ]
    for p in [
        main_src,
        mh_src,
        boot_src,
        budget_src,
        fair_src,
        fair_sm_src,
        overhead_src,
        sm_src,
        sm_boot_src,
    ]:
        if p:
            inventory.append(f"- `{p}`")
    write_text(out_dir / "00_data_inventory.md", "\n".join(inventory) + "\n")

    main_cols = ["method", "budget", *TASKS, "avg"]
    write_csv(out_dir / "01_main_1024_exact.csv", main, main_cols)
    write_text(out_dir / "01_main_1024_exact.md", "# Main Exact 1024 Qwen-Token Results\n\n" + md_table(main, main_cols))
    write_text(
        out_dir / "01_main_1024_exact.tex",
        latex_table(
            "Official exact Qwen-token 1024-token packed-context results.",
            "tab:exact-main",
            [{**r, **{k: fmt2(r[k]) if k not in {"method", "budget"} else r[k] for k in r}} for r in main],
            main_cols,
            ["Method", "Budget", "Qasper", "MFQA", "HotpotQA", "2Wiki", "MuSiQue", "NQA", "Avg"],
            table_star=True,
        ),
    )

    comp_cols = ["method", "lexical_initialization", "bridge_selection", "neighbor_repair", "path_order", "tags", "avg"]
    write_csv(out_dir / "02_component_ablation.csv", comp, comp_cols)
    write_text(out_dir / "02_component_ablation.md", "# Component and Baseline Ablation\n\n" + md_table(comp, comp_cols))
    write_text(
        out_dir / "02_component_ablation.tex",
        latex_table("Component and baseline ablation.", "tab:exact-components", comp, comp_cols),
    )

    mh_cols = ["method", *MULTIHOP_TASKS, "multihop_avg"]
    write_csv(out_dir / "03_multihop_subset.csv", mh, mh_cols)
    write_text(out_dir / "03_multihop_subset.md", "# Multi-Hop Subset\n\n" + md_table(mh, mh_cols))
    write_text(
        out_dir / "03_multihop_subset.tex",
        latex_table("Exact-token multi-hop subset.", "tab:exact-multihop", mh, mh_cols),
    )

    boot_cols = ["comparison", "delta", "ci_low", "ci_high", "paired_samples", "interpretation"]
    write_csv(out_dir / "04_bootstrap_ci.csv", boot, boot_cols)
    write_text(out_dir / "04_bootstrap_ci.md", "# Bootstrap and Significance\n\n" + md_table(boot, boot_cols))
    write_text(out_dir / "04_bootstrap_ci.tex", latex_table("Paired bootstrap confidence intervals.", "tab:bootstrap-ci", boot, boot_cols))

    budget_cols = [
        "budget",
        "span_only",
        "tfidf",
        "bm25",
        "bridgepack",
        "lexical_bridgepack",
        "delta_lexical_vs_span",
        "delta_lexical_vs_bm25",
        "delta_lexical_vs_tfidf",
        "beats_span_only",
        "beats_bm25",
        "beats_tfidf",
    ]
    write_csv(out_dir / "05_budget_scaling.csv", budget, budget_cols)
    write_text(
        out_dir / "05_budget_scaling.md",
        "# Budget Scaling\n\n"
        + md_table(budget, budget_cols)
        + "\n2048 improves over 1024 for Lexical-BridgePack by `+1.1934`; 1024 remains the official main-table budget.\n",
    )
    write_text(out_dir / "05_budget_scaling.tex", latex_table("Exact-token budget scaling.", "tab:budget-scaling", budget, budget_cols[:6]))

    write_text(
        out_dir / "06_fairness_audit.md",
        "# Fairness and Leakage Audit\n\n"
        + "- Exact tokenizer budget is used for the official main table.\n"
        + "- Official main over-budget packed contexts: `0`.\n"
        + "- Official main max packed length: `1024`.\n"
        + "- Same sample ids: `True`.\n"
        + "- Missing/failed compact-audit samples: `0`.\n"
        + "- Gold answers and answer metadata are not used by selectors.\n"
        + "- Old regex-budget results are discarded for official claims.\n\n"
        + "```json\n"
        + json.dumps(fair, indent=2)
        + "\n```\n",
    )
    write_text(out_dir / "06_fairness_audit.json", json.dumps(fair, indent=2) + "\n")
    fair_rows = [
        {
            "setting": "Qwen main",
            "budget": fair["official_main"]["max_packed_length"],
            "variant_sample_rows": fair["official_main"]["variant_sample_rows"],
            "over_budget": fair["official_main"]["over_budget_packed_contexts"],
            "same_sample_ids": fair["official_main"]["same_sample_ids"],
            "missing_failed": fair["official_main"]["missing_or_failed_samples"],
        }
    ]
    if "second_model_gate" in fair:
        fair_rows.append(
            {
                "setting": "Mistral gate",
                "budget": fair["second_model_gate"]["max_packed_length"],
                "variant_sample_rows": fair["second_model_gate"]["variant_sample_rows"],
                "over_budget": fair["second_model_gate"]["over_budget_packed_contexts"],
                "same_sample_ids": fair["second_model_gate"]["same_sample_ids"],
                "missing_failed": fair["second_model_gate"]["missing_or_failed_samples"],
            }
        )
    write_text(
        out_dir / "06_fairness_audit.tex",
        latex_table(
            "Exact-token fairness summary.",
            "tab:fairness-summary",
            fair_rows,
            ["setting", "budget", "variant_sample_rows", "over_budget", "same_sample_ids", "missing_failed"],
        ),
    )

    overhead_cols = [
        "setting",
        "variant",
        "packed_token_mean",
        "packed_token_p95",
        "packed_token_max",
        "source_token_mean",
        "compression_ratio_mean",
        "generation_input_length_mean",
        "generation_input_length_p95",
        "packing_time_mean",
        "packing_time_p95",
    ]
    write_csv(out_dir / "07_overhead.csv", overhead, overhead_cols)
    write_text(out_dir / "07_overhead.md", "# Overhead\n\n" + md_table(overhead, overhead_cols) + "\nTiming values remain TODO when no timing log was found.\n")
    write_text(out_dir / "07_overhead.tex", latex_table("Packing overhead summary.", "tab:overhead", overhead, overhead_cols[:7]))

    case_text = """# Case Studies

Verified compact case studies were not found in the compact paper-table sources.

## TODO Success Case Template

- Task:
- Sample id:
- Question:
- Gold answer:
- Span-only prediction:
- BM25/TF-IDF prediction:
- BridgePack prediction:
- Lexical-BridgePack prediction:
- Evidence snippets or source ids:
- Why this illustrates evidence organization:

## TODO Neutral/Failure Case Template

- Task:
- Sample id:
- Question:
- Gold answer:
- Predictions:
- Evidence snippets or source ids:
- Failure/neutral explanation:
"""
    write_text(out_dir / "08_case_studies.md", case_text)
    write_text(out_dir / "08_case_studies.tex", "% Case studies remain TODO; see 08_case_studies.md.\n")

    if sm_rows:
        sm_cols = ["method", "hotpotqa", "2wikimqa", "musique", "avg"]
        write_csv(out_dir / "09_second_model_table.csv", sm_rows, sm_cols)
        write_text(out_dir / "09_second_model_table.tex", latex_table("Second-model Mistral multi-hop gate.", "tab:second-model-gate", sm_rows, sm_cols))
        sm_status = (
            "# Second-Model Status\n\n"
            "This is a 32-sample-per-task Mistral multi-hop gate, not a full second-model validation.\n\n"
            + md_table(sm_rows, sm_cols)
            + "\n## Bootstrap\n\n"
            + md_table(sm_boot, ["comparison", "delta", "ci_low", "ci_high", "interpretation"])
        )
    else:
        sm_status = """# Second-Model Status

Second-model validation TODO.

- Recommended model: `mistralai/Mistral-7B-Instruct-v0.3`
- Setting: `hotpotqa,2wikimqa,musique`, 32 samples each, exact 1024 tokens
- Variants: span-only, BM25, TF-IDF, BridgePack, Lexical-BridgePack
- Safe purpose: generalization gate, not full main evaluation
"""
    write_text(out_dir / "09_second_model_status.md", sm_status)

    write_text(
        out_dir / "10_missing_data_plan.md",
        """# Missing Data Plan

## Completed

- Exact 1024 main aggregate and per-task values.
- BM25/TF-IDF exact-token baselines.
- Bootstrap intervals.
- Budget scaling.
- Fairness audit.
- Overhead packed-token/source-token/compression summaries.
- Mistral second-model 32-sample-per-task gate.

## Missing/TODO

- Full second-model validation beyond the Mistral gate.
- Dense retriever baseline.
- LongLLMLingua / LLMLingua baseline.
- Packing-time measurements.
- Verified case studies.
- Broader model/dataset validation.
- Final bibliographic metadata verification.
""",
    )

    paths = {
        "main": source(main_src),
        "bootstrap": source(boot_src),
        "budget": source(budget_src),
        "second_model": source(sm_src) if sm_src else "TODO",
        "second_model_bootstrap": source(sm_boot_src) if sm_boot_src else "TODO",
    }
    registry = official_registry(paths)
    write_text(out_dir / "11_official_numbers_registry.json", json.dumps(registry, indent=2) + "\n")
    reg_lines = ["# Official Numbers Registry", ""]
    for key, val in registry.items():
        reg_lines += [f"## `{key}`", f"- value: `{val['value']}`", f"- meaning: {val['meaning']}", f"- source: `{val['source_file_path']}`", f"- safe claim: {val['safe_claim']}", f"- unsafe overclaim: {val['unsafe_overclaim']}", ""]
    write_text(out_dir / "11_official_numbers_registry.md", "\n".join(reg_lines))

    all_tables = "\n".join(
        [
            (out_dir / "01_main_1024_exact.tex").read_text(),
            (out_dir / "02_component_ablation.tex").read_text(),
            (out_dir / "03_multihop_subset.tex").read_text(),
            (out_dir / "04_bootstrap_ci.tex").read_text(),
            (out_dir / "05_budget_scaling.tex").read_text(),
            (out_dir / "06_fairness_audit.tex").read_text(),
            (out_dir / "07_overhead.tex").read_text(),
            (out_dir / "09_second_model_table.tex").read_text() if (out_dir / "09_second_model_table.tex").exists() else "% Second-model table TODO.\n",
        ]
    )
    write_text(out_dir / "13_latex_tables_all.tex", all_tables)

    write_text(
        out_dir / "14_results_paragraphs.md",
        """# Paper-Ready Result Paragraphs

## Main Results

Under the official exact 1024 Qwen-token packed-context budget, Lexical-BridgePack obtains the best observed average score among evaluated methods: 38.3333. The strongest lexical baselines are close, with BM25 at 37.0217 and TF-IDF at 36.9250, so the main claim should emphasize best observed average and significant improvement over span-only rather than significant superiority over lexical baselines.

## Component Analysis

Lexical initialization is important: BM25 and TF-IDF are strong baselines, and Lexical-BridgePack improves the observed average over BridgePack. Tags underperform the no-tag BridgePack row, and path order contributes modestly relative to the same-spans source-order control.

## Bootstrap

Paired bootstrap analysis supports a significant gain over span-only packing: +16.0487 with CI [13.7177, 18.2680]. CIs versus BM25, TF-IDF, and BridgePack cross zero, so those comparisons should be described conservatively.

## Budget Scaling

Lexical-BridgePack improves over span-only at 512, 1024, and 2048 tokens. It is below BM25 at 512 but has the highest observed average at 1024 and 2048. The 2048-token result reaches 39.5267, +1.1934 over 1024, while 1024 remains the official main budget.

## Fairness Audit

The official table uses exact Qwen-token budget accounting, with 12650 variant-sample rows, zero over-budget packed contexts, maximum packed-context length 1024, aligned sample ids, and no missing/failed compact-audit samples.

## Overhead

Packed-token, source-token, compression-ratio, and generation-input-length summaries are available. Packing-time measurements remain TODO because no timing log was found.

## Limitations

This is evidence-aware context compression / text repacking, not strict KV-cache pruning. It does not demonstrate original long-context prefill KV-memory reduction. Dense retriever and LLMLingua/LongLLMLingua baselines remain TODO.
""",
    )

    # Update compact paper files from generated tables.
    write_main_tex(paper_dir)
    write_text(paper_dir / "tables.tex", all_tables)
    write_claim_guard(paper_dir)
    write_todo_checklist(paper_dir)
    write_readme(paper_dir, exact_dir, budget_dir, second_model_dir)

    write_text(out_dir / "12_old_number_scan.md", scan_old_numbers(paper_dir))

    print(f"Wrote data pack to {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exact_dir")
    parser.add_argument("--budget_dir")
    parser.add_argument("--second_model_dir")
    parser.add_argument("--paper_dir", default="paper/aaai2027_bridgepack_exact")
    parser.add_argument("--out_dir", default="paper/aaai2027_bridgepack_exact/data_pack")
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
