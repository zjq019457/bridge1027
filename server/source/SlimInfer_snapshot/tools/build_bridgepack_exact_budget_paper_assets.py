#!/usr/bin/env python3
"""Build paper-ready BridgePack assets from exact Qwen-token results."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]

TASKS = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", "narrativeqa"]
MULTIHOP_TASKS = ["hotpotqa", "2wikimqa", "musique"]
METHOD_ORDER = [
    "head_truncate_source_order",
    "span_only_source_order",
    "bm25_chunk_source_order",
    "tfidf_chunk_source_order",
    "query_only_neighbor32_path_order",
    "bridge_random_neighbor32_path_order",
    "bridge_same_spans_source_order",
    "bridge_neighbor32_path_order_tags",
    "bridge_neighbor32_path_order",
    "lexical_bridgepack_source_order",
    "lexical_bridgepack_path_order",
    "lexical_bridgepack_hybrid",
]
DISPLAY = {
    "head_truncate_source_order": "Head Truncation",
    "span_only_source_order": "Span-only Packing",
    "bm25_chunk_source_order": "BM25 Chunk Packing",
    "tfidf_chunk_source_order": "TF-IDF Chunk Packing",
    "query_only_neighbor32_path_order": "Query-only Neighbor + Path Order",
    "bridge_random_neighbor32_path_order": "Random Bridge Control",
    "bridge_same_spans_source_order": "Same Spans + Source Order",
    "bridge_neighbor32_path_order_tags": "BridgePack + Tags",
    "bridge_neighbor32_path_order": "BridgePack",
    "lexical_bridgepack_source_order": "Lexical BridgePack Source Order",
    "lexical_bridgepack_path_order": "Lexical-BridgePack",
    "lexical_bridgepack_hybrid": "Lexical BridgePack Hybrid",
}
CATEGORY = {
    "head_truncate_source_order": "truncation_baseline",
    "span_only_source_order": "span_baseline",
    "bm25_chunk_source_order": "lexical_baseline",
    "tfidf_chunk_source_order": "lexical_baseline",
    "query_only_neighbor32_path_order": "control",
    "bridge_random_neighbor32_path_order": "control",
    "bridge_same_spans_source_order": "ablation",
    "bridge_neighbor32_path_order_tags": "optional_formatting",
    "bridge_neighbor32_path_order": "bridgepack",
    "lexical_bridgepack_source_order": "lexical_bridgepack_ablation",
    "lexical_bridgepack_path_order": "lexical_bridgepack",
    "lexical_bridgepack_hybrid": "lexical_bridgepack_ablation",
}
PAPER_ROLE = {
    "head_truncate_source_order": "baseline",
    "span_only_source_order": "baseline",
    "bm25_chunk_source_order": "external_baseline",
    "tfidf_chunk_source_order": "external_baseline",
    "query_only_neighbor32_path_order": "control",
    "bridge_random_neighbor32_path_order": "control",
    "bridge_same_spans_source_order": "ablation",
    "bridge_neighbor32_path_order_tags": "optional_formatting",
    "bridge_neighbor32_path_order": "ablation",
    "lexical_bridgepack_source_order": "ablation",
    "lexical_bridgepack_path_order": "ours_final",
    "lexical_bridgepack_hybrid": "ablation",
}


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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


def _float(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt(value: Any, digits: int = 4) -> str:
    number = _float(value)
    if number is None:
        return "TODO"
    return f"{number:.{digits}f}".rstrip("0").rstrip(".")


def _latex_num(value: Any) -> str:
    number = _float(value)
    if number is None:
        return "--"
    return f"{number:.2f}"


def _latex_escape(text: str) -> str:
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
    )


def _load_exact_main(exact_dir: Path) -> Dict[str, Dict[str, str]]:
    rows = _read_csv(exact_dir / "paper_tables" / "01_exact_budget_main_table_1024.csv")
    return {row["method"]: row for row in rows}


def _load_multihop(exact_dir: Path) -> Dict[str, Dict[str, str]]:
    rows = _read_csv(exact_dir / "paper_tables" / "04_exact_budget_multihop_table.csv")
    return {row["method"]: row for row in rows}


def _budget_summary(exact_dir: Path) -> Dict[str, Any]:
    rows = _read_csv(exact_dir / "paper_tables" / "00_model_token_budget_stats.csv")
    missing_rows = _read_csv(exact_dir / "paper_tables" / "00_missing_samples.csv")
    total_samples = sum(int(row.get("num_samples", 0) or 0) for row in rows)
    over_budget = sum(int(row.get("over_budget_samples", 0) or 0) for row in rows)
    max_token = max((int(row.get("model_token_count_max", 0) or 0) for row in rows), default=0)
    truncated = sum(int(row.get("truncated_samples", 0) or 0) for row in rows)
    exact = bool(rows) and all(str(row.get("exact_model_token_budget")) == "True" for row in rows)
    same_samples = not any(
        int(row.get("missing_count", 0) or 0) or int(row.get("extra_count", 0) or 0)
        for row in missing_rows
    )
    return {
        "rows": rows,
        "total_samples": total_samples,
        "over_budget": over_budget,
        "max_token": max_token,
        "truncated": truncated,
        "exact": exact,
        "same_samples": same_samples,
    }


def build_clean_main(exact_dir: Path, out_dir: Path) -> List[Dict[str, Any]]:
    table = _load_exact_main(exact_dir)
    bm25 = _float(table.get("bm25_chunk_source_order", {}).get("avg"))
    tfidf = _float(table.get("tfidf_chunk_source_order", {}).get("avg"))
    rows: List[Dict[str, Any]] = []
    for method in METHOD_ORDER:
        if method not in table:
            continue
        source = table[method]
        avg = _float(source.get("avg"))
        rows.append({
            "method": method,
            "display_name": DISPLAY.get(method, source.get("display_name", method)),
            "category": CATEGORY.get(method, ""),
            "budget": source.get("budget", "1024"),
            **{task: source.get(task, "TODO") for task in TASKS},
            "avg": source.get("avg", "TODO"),
            "delta_vs_span_only": source.get("delta_vs_span_only", "TODO"),
            "delta_vs_bm25": _fmt(avg - bm25) if avg is not None and bm25 is not None else "TODO",
            "delta_vs_tfidf": _fmt(avg - tfidf) if avg is not None and tfidf is not None else "TODO",
            "paper_role": PAPER_ROLE.get(method, ""),
        })
    fields = [
        "method", "display_name", "category", "budget", *TASKS, "avg",
        "delta_vs_span_only", "delta_vs_bm25", "delta_vs_tfidf", "paper_role",
    ]
    _write_csv(out_dir / "01_exact_main_table_clean.csv", rows, fields)
    return rows


def build_component_ablation(main_rows: Sequence[Mapping[str, Any]], out_dir: Path) -> List[Dict[str, Any]]:
    explanations = {
        "span_only_source_order": "Tests independent span selection without bridge-aware ordering.",
        "bm25_chunk_source_order": "Tests lexical retrieval-style packing with a BM25-style score.",
        "tfidf_chunk_source_order": "Tests lexical retrieval-style packing with TF-IDF scoring.",
        "bridge_same_spans_source_order": "Uses the same selected spans as BridgePack but restores source order, isolating ordering.",
        "bridge_neighbor32_path_order": "Tests bridge-aware evidence organization.",
        "lexical_bridgepack_path_order": "Combines lexical retrieval initialization with BridgePack path organization.",
        "bridge_neighbor32_path_order_tags": "Tests optional evidence tags; tags are not part of the core method.",
    }
    rows: List[Dict[str, Any]] = []
    source_by_method = {row["method"]: row for row in main_rows}
    for method in [
        "span_only_source_order",
        "bm25_chunk_source_order",
        "tfidf_chunk_source_order",
        "bridge_same_spans_source_order",
        "bridge_neighbor32_path_order",
        "lexical_bridgepack_path_order",
        "bridge_neighbor32_path_order_tags",
    ]:
        if method not in source_by_method:
            continue
        source = source_by_method[method]
        rows.append({
            "method": method,
            "display_name": source["display_name"],
            "component_test": explanations[method],
            "lexical_init": method.startswith("lexical") or method in {"bm25_chunk_source_order", "tfidf_chunk_source_order"},
            "bridge_aware": method in {
                "bridge_same_spans_source_order",
                "bridge_neighbor32_path_order",
                "bridge_neighbor32_path_order_tags",
                "lexical_bridgepack_path_order",
            },
            "path_order": method in {
                "bridge_neighbor32_path_order",
                "bridge_neighbor32_path_order_tags",
                "lexical_bridgepack_path_order",
            },
            "source_order": method in {
                "span_only_source_order",
                "bm25_chunk_source_order",
                "tfidf_chunk_source_order",
                "bridge_same_spans_source_order",
            },
            "tags": method == "bridge_neighbor32_path_order_tags",
            "avg": source.get("avg", "TODO"),
            "delta_vs_span_only": source.get("delta_vs_span_only", "TODO"),
            "paper_role": source.get("paper_role", ""),
        })
    _write_csv(
        out_dir / "02_exact_component_ablation.csv",
        rows,
        [
            "method", "display_name", "component_test", "lexical_init", "bridge_aware",
            "path_order", "source_order", "tags", "avg", "delta_vs_span_only", "paper_role",
        ],
    )
    return rows


def build_fairness_protocol(exact_dir: Path, out_dir: Path) -> None:
    summary = _budget_summary(exact_dir)
    config = json.loads((exact_dir / "config.json").read_text(encoding="utf-8"))
    lines = [
        "# Exact-Token Fairness Protocol",
        "",
        "The official BridgePack main table now uses exact Qwen-token packed-context budgets. "
        "The earlier regex-estimated budget results are treated as preliminary diagnostics and must not be used as the final main table.",
        "",
        "## Protocol",
        "",
        f"- Model/tokenizer: `{config.get('model_name_or_path', 'TODO')}`.",
        "- Packed context budget: `1024` tokens under the actual generation tokenizer.",
        f"- Variant-sample rows audited: `{summary['total_samples']}`.",
        f"- Over-budget packed contexts: `{summary['over_budget']}`.",
        f"- Maximum packed-context token count: `{summary['max_token']}`.",
        f"- Truncated packed contexts: `{summary['truncated']}`.",
        f"- Same sample ids for every variant/task: **{summary['same_samples']}**.",
        "- Same model, tokenizer, greedy decoding, scoring code, and prompt wrapper are used across variants.",
        "- Gold answers are used only for evaluation; selector logic uses the question and context.",
        "",
        "## Interpretation",
        "",
        "This is a context-compression / text-repacking evaluation. It is not a KV-cache pruning or KV-memory-reduction evaluation.",
    ]
    _write_text(out_dir / "03_exact_fairness_protocol.md", "\n".join(lines))


def build_results_section(main_rows: Sequence[Mapping[str, Any]], out_dir: Path) -> None:
    rows = {row["method"]: row for row in main_rows}
    span = rows.get("span_only_source_order", {})
    bm25 = rows.get("bm25_chunk_source_order", {})
    tfidf = rows.get("tfidf_chunk_source_order", {})
    bridge = rows.get("bridge_neighbor32_path_order", {})
    same = rows.get("bridge_same_spans_source_order", {})
    tags = rows.get("bridge_neighbor32_path_order_tags", {})
    lexical = rows.get("lexical_bridgepack_path_order", {})
    hybrid = rows.get("lexical_bridgepack_hybrid", {})
    source = rows.get("lexical_bridgepack_source_order", {})
    lines = [
        "# Exact-Token Results Section Draft",
        "",
        "We report the exact-token BridgePack results in Table X. Unlike preliminary runs that used a lightweight regex token estimate, "
        "all methods here are repacked under the actual Qwen tokenizer with a 1024-token packed-context limit. "
        "The audit covers 12,650 variant-sample rows, contains zero over-budget packed contexts, and has a maximum packed-context length of 1024 tokens.",
        "",
        "Lexical-BridgePack is the strongest method in the exact-budget table. "
        f"It reaches an average score of {_fmt(lexical.get('avg'))}, compared with {_fmt(bridge.get('avg'))} for BridgePack, "
        f"{_fmt(bm25.get('avg'))} for BM25 chunk packing, {_fmt(tfidf.get('avg'))} for TF-IDF chunk packing, "
        f"and {_fmt(span.get('avg'))} for span-only packing. "
        f"Thus, lexical initialization improves the BridgePack-style path organization by {_fmt(_float(lexical.get('avg')) - _float(bridge.get('avg')) if _float(lexical.get('avg')) is not None and _float(bridge.get('avg')) is not None else None)} average points.",
        "",
        f"BridgePack alone remains competitive under the exact-token fairness check. It scores {_fmt(bridge.get('avg'))}, "
        f"slightly above BM25 ({_fmt(bm25.get('avg'))}) and TF-IDF ({_fmt(tfidf.get('avg'))}). "
        f"The same-spans source-order control scores {_fmt(same.get('avg'))}, so path-order contributes a small gain of "
        f"{_fmt(_float(bridge.get('avg')) - _float(same.get('avg')) if _float(bridge.get('avg')) is not None and _float(same.get('avg')) is not None else None)} average points when selected spans are controlled.",
        "",
        f"Formatting tags are not part of the core method. BridgePack + Tags scores {_fmt(tags.get('avg'))}, below the untagged BridgePack score. "
        "We therefore treat tags as an optional negative ablation rather than as a main component.",
        "",
        f"Among lexical variants, Lexical-BridgePack ({_fmt(lexical.get('avg'))}) is stronger than lexical source order ({_fmt(source.get('avg'))}) "
        f"and lexical hybrid packing ({_fmt(hybrid.get('avg'))}). This suggests that lexical candidate initialization and bridge/path organization are complementary in this setting.",
        "",
        "All claims in this section refer to evidence-aware context compression with a packed prompt. These results do not imply KV-cache pruning or original-prefill memory reduction.",
    ]
    _write_text(out_dir / "04_exact_results_section_draft.md", "\n".join(lines))


def build_claims(out_dir: Path) -> None:
    lines = [
        "# Exact-Token Claims and Limitations",
        "",
        "## Safe Claims",
        "",
        "- Lexical-BridgePack is the final best method under the exact Qwen-token 1024-token packed-context budget.",
        "- BridgePack beats BM25 and TF-IDF under exact-token fairness, but only slightly.",
        "- Lexical initialization strengthens BridgePack.",
        "- Tags hurt in this run and should be treated as optional or as a negative ablation.",
        "- The evaluation is context compression / text repacking, not KV-cache pruning.",
        "",
        "## Unsafe Claims",
        "",
        "- Tags are a core component.",
        "- Path-order is the only source of gains.",
        "- The method reduces KV memory or implements KV-cache pruning.",
        "- Exact-token results prove universal superiority over all prompt-compression systems.",
        "- Regex-estimated budget results are the official main results.",
    ]
    _write_text(out_dir / "05_exact_claims_and_limitations.md", "\n".join(lines))


def _latex_table_main(rows: Sequence[Mapping[str, Any]]) -> str:
    best = max((_float(row.get("avg")) or float("-inf") for row in rows), default=float("-inf"))
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Exact Qwen-token 1024-token context-compression results.}",
        "\\label{tab:exact-bridgepack-main}",
        "\\small",
        "\\begin{tabular}{lrrrrrrrr}",
        "\\toprule",
        "Method & Qasper & MFQA & HotpotQA & 2Wiki & MuSiQue & NarrativeQA & Avg & $\\Delta$ \\\\",
        "\\midrule",
    ]
    for row in rows:
        avg = _float(row.get("avg"))
        avg_text = _latex_num(avg)
        if avg is not None and abs(avg - best) < 1e-9:
            avg_text = f"\\textbf{{{avg_text}}}"
        lines.append(
            f"{_latex_escape(str(row['display_name']))} & "
            + " & ".join(_latex_num(row.get(task)) for task in TASKS)
            + f" & {avg_text} & {_latex_num(row.get('delta_vs_span_only'))} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}"])
    return "\n".join(lines)


def _latex_table_components(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Exact-budget BridgePack component ablations.}",
        "\\label{tab:exact-bridgepack-components}",
        "\\small",
        "\\begin{tabular}{lcccr}",
        "\\toprule",
        "Method & Lexical & Bridge & Path & Avg \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{_latex_escape(str(row['display_name']))} & "
            f"{'Y' if str(row.get('lexical_init')) == 'True' else 'N'} & "
            f"{'Y' if str(row.get('bridge_aware')) == 'True' else 'N'} & "
            f"{'Y' if str(row.get('path_order')) == 'True' else 'N'} & "
            f"{_latex_num(row.get('avg'))} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    return "\n".join(lines)


def _latex_table_multihop(multihop: Mapping[str, Mapping[str, str]], methods: Sequence[str]) -> str:
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Exact-budget multi-hop subset.}",
        "\\label{tab:exact-bridgepack-multihop}",
        "\\small",
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Method & Multi-hop Avg & $\\Delta$ & HotpotQA & 2Wiki & MuSiQue \\\\",
        "\\midrule",
    ]
    for method in methods:
        if method not in multihop:
            continue
        row = multihop[method]
        lines.append(
            f"{_latex_escape(DISPLAY.get(method, method))} & {_latex_num(row.get('multihop_avg'))} & "
            f"{_latex_num(row.get('delta_vs_span_only_multihop'))} & "
            f"{_latex_num(row.get('hotpotqa'))} & {_latex_num(row.get('2wikimqa'))} & {_latex_num(row.get('musique'))} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    return "\n".join(lines)


def _latex_table_fairness(summary: Mapping[str, Any]) -> str:
    rows = [
        ("Variant-sample rows", str(summary["total_samples"])),
        ("Over-budget packed contexts", str(summary["over_budget"])),
        ("Maximum packed-context tokens", str(summary["max_token"])),
        ("Same sample ids", "Yes" if summary["same_samples"] else "No"),
    ]
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Exact-token budget audit.}",
        "\\label{tab:exact-budget-audit}",
        "\\small",
        "\\begin{tabular}{lr}",
        "\\toprule",
        "Audit item & Value \\\\",
        "\\midrule",
    ]
    for key, value in rows:
        lines.append(f"{_latex_escape(key)} & {_latex_escape(value)} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    return "\n".join(lines)


def build_latex(exact_dir: Path, main_rows: Sequence[Mapping[str, Any]], component_rows: Sequence[Mapping[str, Any]], out_dir: Path) -> None:
    multihop = _load_multihop(exact_dir)
    summary = _budget_summary(exact_dir)
    methods = [row["method"] for row in main_rows]
    text = "\n\n".join([
        "% Exact-token BridgePack paper tables. Requires booktabs.",
        _latex_table_main(main_rows),
        _latex_table_components(component_rows),
        _latex_table_multihop(multihop, methods),
        _latex_table_fairness(summary),
    ])
    _write_text(out_dir / "06_latex_tables_exact.tex", text)


def build_abstract_intro(out_dir: Path) -> None:
    lines = [
        "# Updated Abstract and Introduction Bullets",
        "",
        "## Abstract Bullets",
        "",
        "- We study evidence-aware context compression under exact tokenizer budgets for long-context QA.",
        "- The final method, Lexical-BridgePack, initializes evidence candidates with lexical retrieval signals and then organizes selected evidence using bridge-aware path ordering.",
        "- Under an exact Qwen-token 1024-token packed-context budget, Lexical-BridgePack achieves 38.3333 average score, outperforming Span-only Packing (22.2867), BM25 Chunk Packing (37.0217), TF-IDF Chunk Packing (36.9250), and BridgePack (37.2983).",
        "- The audit covers 12,650 variant-sample rows with zero over-budget packed contexts and a maximum packed-context length of 1024 tokens.",
        "- The method is evaluated as prompt/context compression, not as KV-cache pruning.",
        "",
        "## Introduction Bullets",
        "",
        "- Motivation: long-context models still need robust evidence selection when the prompt budget is small.",
        "- Problem: independent span selection misses multi-hop evidence organization, while naive bridge formatting can be sensitive to budget accounting.",
        "- Method: Lexical-BridgePack first uses lexical candidate initialization, then applies bridge/path organization to pack related evidence in a compact prompt.",
        "- Fairness: all official results use exact Qwen tokenizer accounting, shared sample ids, shared prompt wrapper, shared decoding, and shared LongBench scoring.",
        "- Finding: bridge organization alone is slightly stronger than BM25/TF-IDF, and lexical initialization yields the best final method.",
        "- Limitation: results are for context compression and do not demonstrate KV memory reduction.",
    ]
    _write_text(out_dir / "07_updated_abstract_and_intro_bullets.md", "\n".join(lines))


def build_todos(out_dir: Path) -> None:
    lines = [
        "# Remaining TODOs After Exact-Token Main Result",
        "",
        "## Validation TODOs",
        "",
        "- Second-model validation: TODO. Exact-token results currently use the Qwen run.",
        "- Exact-token budget ablation at 512/2048: TODO unless rerun with exact Qwen-token accounting.",
        "- LongLLMLingua external baseline: TODO.",
        "- Dense retriever baseline: TODO.",
        "- Runtime/packing overhead: TODO if not already measured under the exact-token pipeline.",
        "",
        "## Paper Hygiene",
        "",
        "- Replace regex-estimated tables with exact-token tables.",
        "- Keep old regex-budget tables only as preliminary diagnostics, not official results.",
        "- Avoid KV-cache pruning language for BridgePack context-compression results.",
    ]
    _write_text(out_dir / "08_missing_todos_after_exact.md", "\n".join(lines))


def build_draft_patch(out_dir: Path) -> None:
    tex_files = [
        path for path in REPO_ROOT.rglob("*.tex")
        if ".git" not in path.parts and "bridgepack_exact_qwen_budget_1024" not in str(path)
    ]
    draft_like = [
        str(path.relative_to(REPO_ROOT)) for path in tex_files
        if "aaai" in path.name.lower() or "draft" in path.name.lower() or "paper" in path.parts
    ]
    lines = [
        "# Exact-Token AAAI Draft Patch",
        "",
        "No draft was overwritten.",
        "",
        "## Candidate Local TeX Files",
        "",
    ]
    if draft_like:
        lines.extend(f"- `{path}`" for path in draft_like)
    else:
        lines.append("- TODO: No local AAAI draft TeX file was detected. Apply these replacements when the draft is available.")
    lines.extend([
        "",
        "## Required Replacements",
        "",
        "- Replace BridgePack as the final method with Lexical-BridgePack.",
        "- Replace the old regex-budget main table with the exact-token main table.",
        "- Replace old claim `+14.5166` with exact-token claims:",
        "  - Lexical-BridgePack: `38.3333`.",
        "  - BridgePack: `37.2983`.",
        "  - BM25: `37.0217`.",
        "  - TF-IDF: `36.9250`.",
        "  - Span-only: `22.2867`.",
        "- Mark old regex-estimated results as preliminary diagnostics, not the official main result.",
        "- State that the exact-token audit has 12,650 variant-sample rows, zero over-budget packed contexts, and max packed-context length 1024.",
    ])
    _write_text(out_dir / "09_exact_aaai_draft_patch.md", "\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exact_dir", required=True)
    parser.add_argument("--legacy_run_dir", default="results/bridgepack_paper_20260618_070828")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seed != 0:
        raise ValueError("Only seed 0 is supported for deterministic paper asset generation.")
    exact_dir = Path(args.exact_dir)
    out_dir = Path(args.out_dir)
    required = [
        "00_exact_budget_fairness_report.md",
        "00_model_token_budget_stats.csv",
        "01_exact_budget_main_table_1024.csv",
        "02_exact_budget_delta_vs_span_only.csv",
        "03_exact_budget_delta_vs_bm25_tfidf.csv",
        "04_exact_budget_multihop_table.csv",
        "05_exact_budget_claim_summary.md",
        "06_reviewer_safe_interpretation.md",
    ]
    missing = [name for name in required if not (exact_dir / "paper_tables" / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing exact paper table inputs: {missing}")
    main_rows = build_clean_main(exact_dir, out_dir)
    component_rows = build_component_ablation(main_rows, out_dir)
    build_fairness_protocol(exact_dir, out_dir)
    build_results_section(main_rows, out_dir)
    build_claims(out_dir)
    build_latex(exact_dir, main_rows, component_rows, out_dir)
    build_abstract_intro(out_dir)
    build_todos(out_dir)
    build_draft_patch(out_dir)
    print(f"OUT_DIR={out_dir}")


if __name__ == "__main__":
    main()
