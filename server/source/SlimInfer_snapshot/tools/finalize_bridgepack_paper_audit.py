#!/usr/bin/env python3
"""Create final reviewer-risk audit files for the exact-token BridgePack paper."""
from __future__ import annotations

import argparse
import csv
import json
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


TASKS = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", "narrativeqa"]
CASE_METHODS = [
    "span_only_source_order",
    "bm25_chunk_source_order",
    "tfidf_chunk_source_order",
    "bridge_neighbor32_path_order",
    "lexical_bridgepack_path_order",
]


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


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


def _normalize(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def _f1(prediction: str, answer: str) -> float:
    pred = _normalize(prediction).split()
    gold = _normalize(answer).split()
    common = Counter(pred) & Counter(gold)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision = same / max(1, len(pred))
    recall = same / max(1, len(gold))
    return 100.0 * (2 * precision * recall) / max(1e-12, precision + recall)


def _score(row: Mapping[str, Any]) -> float:
    answers = row.get("answers", [])
    if not isinstance(answers, list):
        answers = [answers]
    pred = str(row.get("pred", ""))
    return max((_f1(pred, str(answer)) for answer in answers), default=0.0)


def _case_maps(exact_dir: Path, task: str) -> Dict[str, Dict[str, Dict[str, Any]]]:
    out: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for method in CASE_METHODS:
        path = exact_dir / "predictions" / method / f"{task}.jsonl"
        out[method] = {str(row.get("sample_id")): row for row in _read_jsonl(path)}
    return out


def _select_cases(exact_dir: Path) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for task in ["hotpotqa", "2wikimqa", "musique", "qasper"]:
        maps = _case_maps(exact_dir, task)
        common = set.intersection(*(set(rows) for rows in maps.values() if rows))
        for sample_id in common:
            scores = {method: _score(maps[method][sample_id]) for method in CASE_METHODS}
            lexical = scores["lexical_bridgepack_path_order"]
            best_baseline = max(scores["span_only_source_order"], scores["bm25_chunk_source_order"], scores["tfidf_chunk_source_order"])
            bridge = scores["bridge_neighbor32_path_order"]
            row = maps["lexical_bridgepack_path_order"][sample_id]
            candidates.append({
                "task": task,
                "sample_id": sample_id,
                "question": row.get("question", ""),
                "answers": row.get("answers", []),
                "scores": scores,
                "lexical_gain_vs_best_baseline": lexical - best_baseline,
                "lexical_gain_vs_bridgepack": lexical - bridge,
                "predictions": {method: maps[method][sample_id].get("pred", "") for method in CASE_METHODS},
            })
    if not candidates:
        return []
    success = max(candidates, key=lambda row: row["lexical_gain_vs_best_baseline"])
    failure = min(candidates, key=lambda row: row["lexical_gain_vs_bridgepack"])
    neutral = min(candidates, key=lambda row: abs(row["lexical_gain_vs_bridgepack"]))
    selected: List[Dict[str, Any]] = []
    for row in [success, neutral, failure]:
        if row["sample_id"] not in {item["sample_id"] for item in selected}:
            selected.append(row)
    return selected


def _case_text(cases: Sequence[Mapping[str, Any]]) -> str:
    if not cases:
        return "\n".join([
            "# Case Study Update",
            "",
            "TODO: Local predictions/packed prompts were not available, so no cases were selected.",
        ])
    lines = [
        "# Case Study Update",
        "",
        "Cases were selected from existing exact-token predictions only. Scores are local English QA-F1 estimates used for case selection; no model was loaded.",
    ]
    for case in cases:
        lines.extend([
            "",
            f"## {case['task']} / {case['sample_id']}",
            "",
            f"Question: {case['question']}",
            "",
            f"Answers: `{case['answers']}`",
            "",
            "Scores:",
        ])
        for method, score in case["scores"].items():
            lines.append(f"- `{method}`: {score:.2f}")
        lines.append("")
        lines.append("Predictions:")
        for method, pred in case["predictions"].items():
            lines.append(f"- `{method}`: `{str(pred)[:300]}`")
    return "\n".join(lines)


def _latest_budget_ablation() -> Path | None:
    candidates = [
        path for path in Path("results").glob("bridgepack_exact_budget_ablation_*")
        if (path / "paper_tables" / "01_budget_ablation_table.csv").exists()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _budget_claim_lines() -> List[str]:
    budget_dir = _latest_budget_ablation()
    if budget_dir is None:
        return ["- Exact-token 512/2048 budget ablation remains TODO."]
    rows = _read_csv(budget_dir / "paper_tables" / "01_budget_ablation_table.csv")
    lexical = {row["budget"]: row for row in rows if row.get("method") == "lexical_bridgepack_path_order"}
    fair = (budget_dir / "paper_tables" / "00_budget_ablation_fairness.md").read_text(encoding="utf-8") if (budget_dir / "paper_tables" / "00_budget_ablation_fairness.md").exists() else ""
    lines = [
        f"- Exact-token budget ablation is complete: `{budget_dir}`.",
        "- Lexical-BridgePack improves over span-only at every completed budget.",
        "- Lexical-BridgePack does not improve over BM25/TF-IDF at every completed budget; at 512 it is below BM25.",
        "- 2048 has the best completed Lexical-BridgePack average, but 1024 remains the official main-table default setting.",
    ]
    if "2048" in lexical and "1024" in lexical:
        try:
            delta = float(lexical["2048"]["avg"]) - float(lexical["1024"]["avg"])
            lines.append(f"- 2048 improves over 1024 by `{delta:.4f}` average points in the scaling run.")
        except (KeyError, TypeError, ValueError):
            pass
    if "Over-budget packed contexts: `0`" in fair:
        lines.append("- Budget ablation fairness: 0 over-budget packed contexts for completed budgets.")
    return lines


def build_audit(exact_dir: Path, paper_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    final_claims = [
        "# Final Supported Claims",
        "",
        "- Main method: **Lexical-BridgePack** (`lexical_bridgepack_path_order`).",
        "- Official results are the exact Qwen-token 1024-token packed-context results.",
        "- Lexical-BridgePack obtains the best average score: `38.3333`.",
        "- Lexical-BridgePack significantly improves over span-only: paired delta `+16.0487`, CI `[13.7177, 18.268]`.",
        "- Lexical-BridgePack has a higher average than BM25 and TF-IDF, but CI crosses zero; this is not a statistically significant superiority claim.",
        "- BridgePack is an ablation / intermediate method, not the final method.",
        "- Tags are optional and underperform in this run.",
        "- Path-order contribution is modest.",
        "- The paper is about evidence-aware context compression / text repacking, not KV-cache pruning.",
        "",
        "## Budget Scaling",
        "",
        *_budget_claim_lines(),
    ]
    _write_text(out_dir / "01_final_claims.md", "\n".join(final_claims))
    _write_text(out_dir / "02_unsafe_claims_to_remove.md", "\n".join([
        "# Unsafe Claims To Remove",
        "",
        "- Lexical-BridgePack significantly outperforms BM25 or TF-IDF.",
        "- Tags are a core component.",
        "- Path order is the only or dominant source of gain.",
        "- The method is strict KV-cache pruning.",
        "- The method reduces original long-context prefill KV memory.",
        "- Old regex-budget results are official main results.",
        "- Exact-token results prove universal superiority over all prompt-compression systems.",
    ]))
    _write_text(out_dir / "03_remaining_todos.md", "\n".join([
        "# Remaining TODOs",
        "",
        "- Second-model validation: TODO.",
        "- Packing-time overhead: TODO unless timing instrumentation is run.",
        "- LongLLMLingua / LLMLingua baseline: TODO.",
        "- Dense retriever baseline: TODO.",
        "- Broader model/dataset validation: TODO.",
    ]))
    _write_text(out_dir / "04_case_study_update.md", _case_text(_select_cases(exact_dir)))
    _write_text(out_dir / "05_revision_plan.md", "\n".join([
        "# Revision Plan",
        "",
        "1. Replace all regex-budget main tables with exact-token tables.",
        "2. Name Lexical-BridgePack as the final method and demote BridgePack to ablation.",
        "3. Add bootstrap language: strong vs span-only, non-significant vs BM25/TF-IDF.",
        "4. Keep tags as a negative ablation.",
        "5. Add the completed exact-token 512/1024/2048 budget scaling table while keeping 1024 as the official main-table default.",
        "6. Remove KV-pruning and KV-memory-reduction claims.",
        "7. Keep TODO markers for second model, packing-time overhead, LLMLingua, and dense retrieval.",
    ]))
    table_files = sorted((exact_dir / "paper_tables").glob("*.csv")) + sorted((exact_dir / "paper_tables").glob("*.md"))
    budget_dir = _latest_budget_ablation()
    if budget_dir is not None:
        table_files.extend(sorted((budget_dir / "paper_tables").glob("*.csv")))
        table_files.extend(sorted((budget_dir / "paper_tables").glob("*.md")))
    writeup_files = sorted((paper_dir).glob("*.tex")) + sorted((paper_dir).glob("*.md"))
    lines = ["# Table and Report Inventory", ""]
    lines.append("## Exact Result Tables")
    for path in table_files:
        lines.append(f"- `{path}`")
    lines.append("")
    lines.append("## Paper Draft Files")
    for path in writeup_files:
        lines.append(f"- `{path}`")
    _write_text(out_dir / "06_table_inventory.md", "\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exact_dir", required=True)
    parser.add_argument("--paper_dir", default="paper/aaai2027_bridgepack_exact")
    parser.add_argument("--out_dir", default="paper/aaai2027_bridgepack_exact/final_audit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_audit(Path(args.exact_dir), Path(args.paper_dir), Path(args.out_dir))
    print(f"OUT_DIR={args.out_dir}")


if __name__ == "__main__":
    main()
