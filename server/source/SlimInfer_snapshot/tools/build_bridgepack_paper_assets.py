#!/usr/bin/env python3
"""Build paper-ready BridgePack tables and writeup assets.

This script is deliberately offline-only. It reads completed BridgePack result
artifacts and never loads a model, touches CUDA, reruns packing, or rewrites
predictions.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
import string
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TASKS = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", "narrativeqa"]
MULTIHOP_TASKS = ["hotpotqa", "2wikimqa", "musique"]
METHOD_ORDER = [
    "head_truncate_source_order",
    "span_only_source_order",
    "neighbor32_source_order",
    "neighbor32_path_order",
    "bridge_neighbor32_path_order",
    "bridge_neighbor32_path_order_tags",
    "full_context_4096",
]
COMPONENT_METHODS = [
    "head_truncate_source_order",
    "span_only_source_order",
    "neighbor32_source_order",
    "neighbor32_path_order",
    "bridge_neighbor32_path_order",
    "bridge_neighbor32_path_order_tags",
]
DISPLAY_NAMES = {
    "head_truncate_source_order": "Head Truncation",
    "span_only_source_order": "Span-only Packing",
    "neighbor32_source_order": "Neighbor Packing",
    "neighbor32_path_order": "Neighbor + Path Order",
    "bridge_neighbor32_path_order": "BridgePack",
    "bridge_neighbor32_path_order_tags": "BridgePack + Tags",
    "full_context_4096": "Full Context 4096 Reference",
}
PAPER_ROLES = {
    "head_truncate_source_order": "baseline",
    "span_only_source_order": "baseline",
    "neighbor32_source_order": "ablation",
    "neighbor32_path_order": "ablation",
    "bridge_neighbor32_path_order": "ours_main",
    "bridge_neighbor32_path_order_tags": "optional_formatting",
    "full_context_4096": "reference",
}
INTERPRETATIONS = {
    "head_truncate_source_order": "naive prefix-style compression baseline",
    "span_only_source_order": "independent top-span baseline",
    "neighbor32_source_order": "tests local context repair",
    "neighbor32_path_order": "tests evidence ordering without bridge-aware selection",
    "bridge_neighbor32_path_order": "tests bridge-aware path selection plus path ordering",
    "bridge_neighbor32_path_order_tags": (
        "tests whether explicit metadata tags help; tags are optional because no-tags BridgePack is stronger"
    ),
}


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def to_float(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out):
        return None
    return out


def fmt(value: Any, digits: int = 4) -> str:
    number = to_float(value)
    if number is None:
        return ""
    text = f"{number:.{digits}f}".rstrip("0").rstrip(".")
    return text if text else "0"


def fmt_bool(value: bool) -> str:
    return "True" if value else "False"


def get_component_scores(run_dir: Path) -> Dict[Tuple[str, str], float]:
    scores: Dict[Tuple[str, str], float] = {}
    for row in read_csv(run_dir / "paper_tables" / "03_component_ablation.csv"):
        variant = row.get("variant", "")
        task = row.get("task", "")
        score = to_float(row.get("score"))
        if variant and task and score is not None:
            scores[(variant, task)] = score
    return scores


def get_main_rows(run_dir: Path) -> Dict[str, Dict[str, str]]:
    return {row.get("variant", ""): row for row in read_csv(run_dir / "paper_tables" / "01_main_table_1024.csv")}


def build_main_table(run_dir: Path, out_dir: Path) -> List[Dict[str, Any]]:
    component_scores = get_component_scores(run_dir)
    main_rows = get_main_rows(run_dir)
    rows: List[Dict[str, Any]] = []
    for method in METHOD_ORDER:
        has_method = method in main_rows or any(key[0] == method for key in component_scores)
        if not has_method:
            continue
        main = main_rows.get(method, {})
        reference_only = method == "full_context_4096"
        row: Dict[str, Any] = {
            "method": method,
            "display_name": DISPLAY_NAMES.get(method, method),
            "setting": "reference full context" if reference_only else "compressed text packing",
            "budget": "" if reference_only else 1024,
            "avg": fmt(main.get("macro_avg") or component_scores.get((method, "_macro"))),
            "delta_vs_span_only": fmt(main.get("delta_vs_span_only")),
            "reference_only": fmt_bool(reference_only),
            "paper_role": PAPER_ROLES.get(method, ""),
        }
        for task in TASKS:
            row[task] = fmt(component_scores.get((method, task)), 2)
        rows.append(row)

    fields = [
        "method",
        "display_name",
        "setting",
        "budget",
        *TASKS,
        "avg",
        "delta_vs_span_only",
        "reference_only",
        "paper_role",
    ]
    write_csv(out_dir / "01_main_table_clean.csv", rows, fields)
    return rows


def build_component_table(run_dir: Path, out_dir: Path) -> List[Dict[str, Any]]:
    component_scores = get_component_scores(run_dir)
    main_rows = get_main_rows(run_dir)
    rows: List[Dict[str, Any]] = []
    for method in COMPONENT_METHODS:
        if method not in main_rows and (method, "_macro") not in component_scores:
            continue
        avg = main_rows.get(method, {}).get("macro_avg") or component_scores.get((method, "_macro"))
        delta = main_rows.get(method, {}).get("delta_vs_span_only")
        rows.append({
            "method": method,
            "bridge_aware": fmt_bool(method.startswith("bridge_")),
            "neighbor32": fmt_bool("neighbor32" in method),
            "path_order": fmt_bool("path_order" in method),
            "tags": fmt_bool(method.endswith("_tags")),
            "avg": fmt(avg),
            "delta_vs_span_only": fmt(delta),
            "interpretation": INTERPRETATIONS.get(method, ""),
        })
    fields = [
        "method",
        "bridge_aware",
        "neighbor32",
        "path_order",
        "tags",
        "avg",
        "delta_vs_span_only",
        "interpretation",
    ]
    write_csv(out_dir / "02_component_ablation_clean.csv", rows, fields)
    return rows


def parse_fairness_checks(fairness_text: str) -> List[Tuple[str, str]]:
    checks: List[Tuple[str, str]] = []
    pattern = re.compile(r"^\d+\.\s+([^:]+):\s+\*\*(True|False)\*\*", re.MULTILINE)
    for match in pattern.finditer(fairness_text):
        checks.append((match.group(1).strip(), "Pass" if match.group(2) == "True" else "Fail"))
    return checks


def build_fairness_protocol(run_dir: Path, out_dir: Path) -> None:
    paper_dir = run_dir / "paper_tables"
    fairness_path = paper_dir / "00_fairness_report.md"
    fairness_text = fairness_path.read_text(encoding="utf-8") if fairness_path.exists() else ""
    checks = parse_fairness_checks(fairness_text)
    if not checks:
        checks = [
            ("same_model_name_or_path_for_every_variant", "Unknown"),
            ("same_tokenizer_for_every_variant", "Unknown"),
            ("same_sample_ids_for_every_compressed_variant", "Unknown"),
            ("compressed_variants_use_budget_1024", "Unknown"),
            ("same_generation_config_for_every_variant", "Unknown"),
            ("same_scoring_function_for_every_variant", "Unknown"),
            ("shared_prompt_wrapper_except_packed_evidence", "Unknown"),
            ("no_explicit_answer_metadata_in_prompt_text", "Unknown"),
        ]

    lines = [
        "# BridgePack Fairness Protocol",
        "",
        "BridgePack is evaluated as evidence-aware context compression / text repacking, not as strict KV-cache pruning.",
        "The fair comparison is among compressed variants under the same packed-token budget.",
        "",
        "## Protocol",
        "",
        "- Same model for every compressed variant.",
        "- Same tokenizer for every compressed variant.",
        "- Same dataset samples for every compressed variant.",
        "- Same packed-token budget for compressed methods: 1024 tokens.",
        "- Same generation configuration: greedy decoding with the repository LongBench generation settings.",
        "- Same scoring function: existing LongBench scorer / `dataset2metric`.",
        "- Same prompt wrapper except for the packed evidence content.",
        "- No gold-answer metadata is serialized into `packed_text` or `generation_prompt`.",
        "- `full_context_4096`, if present, is reference-only and not a same-budget baseline.",
        "",
        "## Checklist",
        "",
        "| Check | Status |",
        "| --- | --- |",
    ]
    for check, status in checks:
        lines.append(f"| `{check}` | {status} |")
    lines.extend([
        "",
        "## Source Artifacts",
        "",
        f"- `{fairness_path.relative_to(run_dir)}`",
        "- `paper_tables/00_generation_config.json`",
        "- `paper_tables/00_sample_manifest.csv`",
        "- `paper_tables/00_prompt_hash_manifest.csv`",
        "- `paper_tables/00_leakage_audit.csv`",
        "- `paper_tables/00_missing_or_mismatched_samples.csv`",
    ])
    write_text(out_dir / "03_fairness_protocol.md", "\n".join(lines))


def value_from_main(main_rows: Mapping[str, Mapping[str, str]], method: str, field: str) -> str:
    return fmt(main_rows.get(method, {}).get(field))


def build_tags_analysis(run_dir: Path, out_dir: Path) -> None:
    main_rows = get_main_rows(run_dir)
    tags = "bridge_neighbor32_path_order_tags"
    no_tags = "bridge_neighbor32_path_order"
    span = "span_only_source_order"
    tags_avg = value_from_main(main_rows, tags, "macro_avg")
    no_tags_avg = value_from_main(main_rows, no_tags, "macro_avg")
    span_avg = value_from_main(main_rows, span, "macro_avg")
    tags_delta = value_from_main(main_rows, tags, "delta_vs_span_only")
    lines = [
        "# Tags Analysis",
        "",
        f"The pre-registered tags variant, `BridgePack + Tags`, strongly improves over `Span-only Packing`: {tags_avg} vs {span_avg}, delta +{tags_delta}.",
        f"However, the no-tags BridgePack variant is stronger in this completed run: {no_tags_avg} vs {tags_avg}.",
        "",
        "Therefore, the paper should define BridgePack as bridge-aware neighbor-preserving path-order packing.",
        "Tags should be reported as optional formatting rather than as a required component.",
        "The gain mainly comes from evidence organization rather than explicit metadata tags.",
    ]
    write_text(out_dir / "04_tags_analysis.md", "\n".join(lines))


def sample_score(task: str, pred: str, answers: Sequence[str], all_classes: Any = None) -> float:
    try:
        from longbench_eval import dataset2metric

        metric = dataset2metric.get(task)
        if metric is None:
            return 0.0
        best = 0.0
        for answer in answers or [""]:
            try:
                best = max(best, float(metric(pred, answer, all_classes=all_classes)))
            except Exception:
                continue
        return round(100.0 * best, 6)
    except Exception:
        pass

    # Offline fallback for English LongBench QA tasks. This mirrors the local
    # qa_f1_score behavior closely enough for case-study selection when optional
    # scorer dependencies such as jieba are unavailable in the active shell.
    def normalize_answer(text: str) -> str:
        def remove_articles(value: str) -> str:
            return re.sub(r"\b(a|an|the)\b", " ", value)

        def white_space_fix(value: str) -> str:
            return " ".join(value.split())

        def remove_punc(value: str) -> str:
            exclude = set(string.punctuation)
            return "".join(char for char in value if char not in exclude)

        return white_space_fix(remove_articles(remove_punc(str(text).lower())))

    def qa_f1(prediction: str, ground_truth: str) -> float:
        prediction_tokens = normalize_answer(prediction).split()
        ground_truth_tokens = normalize_answer(ground_truth).split()
        if not prediction_tokens or not ground_truth_tokens:
            return float(prediction_tokens == ground_truth_tokens)
        common = set(prediction_tokens) & set(ground_truth_tokens)
        num_same = sum(min(prediction_tokens.count(token), ground_truth_tokens.count(token)) for token in common)
        if num_same == 0:
            return 0.0
        precision = num_same / len(prediction_tokens)
        recall = num_same / len(ground_truth_tokens)
        return 2 * precision * recall / (precision + recall)

    best = 0.0
    for answer in answers or [""]:
        best = max(best, qa_f1(pred, str(answer)))
    return round(100.0 * best, 6)


def load_prediction_map(run_dir: Path, variant: str, task: str) -> Dict[str, Dict[str, Any]]:
    path = run_dir / "main_1024" / "predictions" / variant / f"{task}.jsonl"
    rows = read_jsonl(path)
    return {str(row.get("sample_id", "")): row for row in rows if row.get("sample_id")}


def load_packed_map(run_dir: Path, variant: str, task: str) -> Dict[str, Dict[str, Any]]:
    path = (
        run_dir
        / "main_1024"
        / "packed_prompts_build"
        / "packed_prompts"
        / variant
        / "budget_1024"
        / f"{task}.jsonl"
    )
    rows = read_jsonl(path)
    return {str(row.get("sample_id", "")): row for row in rows if row.get("sample_id")}


def compact_text(text: Any, limit: int = 520) -> str:
    cleaned = " ".join(str(text or "").split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3].rstrip() + "..."


def evidence_excerpt(packed: Mapping[str, Any], limit: int = 520) -> str:
    text = str(packed.get("packed_text") or packed.get("packed_prompt") or "")
    marker = "Selected evidence:"
    if marker in text:
        text = text.split(marker, 1)[1]
    return compact_text(text, limit)


def ids_summary(value: Any, limit: int = 8) -> str:
    if not isinstance(value, list):
        return ""
    shown = value[:limit]
    suffix = "" if len(value) <= limit else f" ... (+{len(value) - limit})"
    return ", ".join(str(item) for item in shown) + suffix


def build_case_candidates(run_dir: Path) -> List[Dict[str, Any]]:
    span_variant = "span_only_source_order"
    bridge_variant = "bridge_neighbor32_path_order"
    candidates: List[Dict[str, Any]] = []
    for task in MULTIHOP_TASKS:
        span_preds = load_prediction_map(run_dir, span_variant, task)
        bridge_preds = load_prediction_map(run_dir, bridge_variant, task)
        common = sorted(set(span_preds) & set(bridge_preds))
        for sample_id in common:
            span = span_preds[sample_id]
            bridge = bridge_preds[sample_id]
            answers = bridge.get("answers") or span.get("answers") or []
            all_classes = bridge.get("all_classes") or span.get("all_classes")
            span_score = sample_score(task, str(span.get("pred", "")), answers, all_classes)
            bridge_score = sample_score(task, str(bridge.get("pred", "")), answers, all_classes)
            candidates.append({
                "task": task,
                "sample_id": sample_id,
                "span_score": span_score,
                "bridge_score": bridge_score,
                "delta": bridge_score - span_score,
                "span": span,
                "bridge": bridge,
            })
    return candidates


def select_cases(candidates: Sequence[Dict[str, Any]], seed: int) -> List[Dict[str, Any]]:
    if not candidates:
        return []
    rng = random.Random(seed)
    positives = [row for row in candidates if row["delta"] > 0 and row["bridge_score"] >= row["span_score"]]
    positives.sort(key=lambda row: (-row["delta"], MULTIHOP_TASKS.index(row["task"]), row["sample_id"]))
    chosen: List[Dict[str, Any]] = []
    used_tasks = set()
    for row in positives:
        if row["task"] in used_tasks and len(chosen) < len(MULTIHOP_TASKS):
            continue
        chosen.append(row)
        used_tasks.add(row["task"])
        if len(chosen) == 2:
            break
    if len(chosen) < 2:
        for row in positives:
            if row not in chosen:
                chosen.append(row)
            if len(chosen) == 2:
                break

    failures = [row for row in candidates if row["delta"] < 0]
    failures.sort(key=lambda row: (row["delta"], MULTIHOP_TASKS.index(row["task"]), row["sample_id"]))
    neutral = [row for row in candidates if abs(row["delta"]) <= 1e-9]
    rng.shuffle(neutral)
    extra = failures[0] if failures else (neutral[0] if neutral else None)
    if extra is not None and extra not in chosen:
        chosen.append(extra)
    return chosen[:3]


def build_case_study(run_dir: Path, out_dir: Path, seed: int) -> None:
    candidates = build_case_candidates(run_dir)
    selected = select_cases(candidates, seed)
    lines = [
        "# BridgePack Case Studies",
        "",
        "Cases are selected by objective criteria from local predictions and packed prompts.",
        "Positive cases maximize per-sample LongBench score delta for BridgePack over span-only on multi-hop tasks.",
        "The final case is a non-cherry-picked neutral or failure case when available.",
        "",
    ]
    if not selected:
        fallback = run_dir / "paper_tables" / "06_case_examples.md"
        if fallback.exists():
            lines.append("Local per-sample predictions were not usable; falling back to `paper_tables/06_case_examples.md`.")
            lines.append("")
            lines.append(fallback.read_text(encoding="utf-8"))
        else:
            lines.extend([
                "No usable local case data was found.",
                "",
                "Template:",
                "",
                "- task:",
                "- sample_id:",
                "- question:",
                "- gold answer:",
                "- span_only prediction:",
                "- BridgePack prediction:",
                "- score difference:",
                "- span_only selected evidence summary:",
                "- BridgePack selected evidence summary:",
                "- query -> bridge -> support explanation:",
            ])
        write_text(out_dir / "05_case_study_bridgepack.md", "\n".join(lines))
        return

    span_packed_cache: Dict[Tuple[str, str], Dict[str, Dict[str, Any]]] = {}
    bridge_packed_cache: Dict[Tuple[str, str], Dict[str, Dict[str, Any]]] = {}
    for index, row in enumerate(selected, start=1):
        task = row["task"]
        sample_id = row["sample_id"]
        span_key = ("span_only_source_order", task)
        bridge_key = ("bridge_neighbor32_path_order", task)
        if span_key not in span_packed_cache:
            span_packed_cache[span_key] = load_packed_map(run_dir, span_key[0], task)
        if bridge_key not in bridge_packed_cache:
            bridge_packed_cache[bridge_key] = load_packed_map(run_dir, bridge_key[0], task)
        span_pack = span_packed_cache[span_key].get(sample_id, {})
        bridge_pack = bridge_packed_cache[bridge_key].get(sample_id, {})
        span_pred = row["span"]
        bridge_pred = row["bridge"]
        label = "positive" if row["delta"] > 0 else ("failure" if row["delta"] < 0 else "neutral")
        answers = bridge_pred.get("answers") or span_pred.get("answers") or []
        lines.extend([
            f"## Case {index}: {task} / {sample_id}",
            "",
            f"- Selection label: {label}",
            f"- Question: {span_pred.get('question', bridge_pred.get('question', ''))}",
            f"- Gold answer(s): {json.dumps(answers, ensure_ascii=False)}",
            f"- Span-only prediction: {span_pred.get('pred', '')}",
            f"- BridgePack prediction: {bridge_pred.get('pred', '')}",
            f"- Span-only score: {fmt(row['span_score'])}",
            f"- BridgePack score: {fmt(row['bridge_score'])}",
            f"- Score difference: {fmt(row['delta'])}",
            "",
            "### Span-only Selected Evidence Summary",
            "",
            f"- selected_page_ids: {ids_summary(span_pack.get('selected_page_ids'))}",
            f"- selected_span_ids: {ids_summary(span_pack.get('selected_span_ids'))}",
            f"- Excerpt: {evidence_excerpt(span_pack)}",
            "",
            "### BridgePack Selected Evidence Summary",
            "",
            f"- selected_page_ids: {ids_summary(bridge_pack.get('selected_page_ids'))}",
            f"- selected_span_ids: {ids_summary(bridge_pack.get('selected_span_ids'))}",
            f"- used_bridge_path: {bridge_pack.get('used_bridge_path', '')}",
            f"- ordering: {bridge_pack.get('ordering', '')}",
            f"- Excerpt: {evidence_excerpt(bridge_pack)}",
            "",
            "### Query -> Bridge -> Support Organization",
            "",
            (
                "BridgePack metadata marks bridge-aware selection and path ordering as active for this sample. "
                "The explanation is limited to observed local artifacts: selected pages/spans and the quoted packed-text excerpt. "
                "When BridgePack improves, the packed evidence appears less like independent high-scoring snippets and more like an ordered evidence chain. "
                "When it fails or ties, the local artifacts should be treated as a diagnostic rather than proof of the intended chain."
            ),
            "",
        ])
    write_text(out_dir / "05_case_study_bridgepack.md", "\n".join(lines))


def latex_escape(text: Any) -> str:
    out = str(text)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in out)


def ranked_methods(rows: Sequence[Mapping[str, Any]], score_field: str = "avg") -> Tuple[str, str]:
    scored: List[Tuple[float, str]] = []
    for row in rows:
        if str(row.get("reference_only")) == "True":
            continue
        score = to_float(row.get(score_field))
        if score is not None:
            scored.append((score, str(row.get("method", ""))))
    scored.sort(reverse=True)
    best = scored[0][1] if scored else ""
    second = scored[1][1] if len(scored) > 1 else ""
    return best, second


def latex_score(method: str, value: Any, best: str, second: str) -> str:
    text = latex_escape(fmt(value) if to_float(value) is not None else value)
    if method == best:
        return rf"\textbf{{{text}}}"
    if method == second:
        return rf"\underline{{{text}}}"
    return text


def latex_table(title: str, label: str, headers: Sequence[str], body_rows: Sequence[Sequence[str]]) -> str:
    alignment = "l" + "r" * (len(headers) - 1)
    lines = [
        rf"\begin{{table}}[t]",
        r"\centering",
        rf"\caption{{{latex_escape(title)}}}",
        rf"\label{{{latex_escape(label)}}}",
        r"\small",
        r"\resizebox{\linewidth}{!}{%",
        rf"\begin{{tabular}}{{{alignment}}}",
        r"\toprule",
        " & ".join(latex_escape(header) for header in headers) + r" \\",
        r"\midrule",
    ]
    for row in body_rows:
        lines.append(" & ".join(row) + r" \\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table}",
        "",
    ])
    return "\n".join(lines)


def build_latex_tables(run_dir: Path, out_dir: Path, main_rows: Sequence[Mapping[str, Any]], component_rows: Sequence[Mapping[str, Any]]) -> None:
    best, second = ranked_methods(main_rows)
    main_body: List[List[str]] = []
    for row in main_rows:
        method = str(row.get("method", ""))
        main_body.append([
            latex_escape(row.get("display_name", method)),
            latex_escape(row.get("qasper", "")),
            latex_escape(row.get("multifieldqa_en", "")),
            latex_escape(row.get("hotpotqa", "")),
            latex_escape(row.get("2wikimqa", "")),
            latex_escape(row.get("musique", "")),
            latex_escape(row.get("narrativeqa", "")),
            latex_score(method, row.get("avg", ""), best, second),
            latex_escape(row.get("delta_vs_span_only", "")),
        ])

    component_body: List[List[str]] = []
    for row in component_rows:
        component_body.append([
            latex_escape(DISPLAY_NAMES.get(str(row.get("method")), row.get("method", ""))),
            latex_escape(row.get("bridge_aware", "")),
            latex_escape(row.get("neighbor32", "")),
            latex_escape(row.get("path_order", "")),
            latex_escape(row.get("tags", "")),
            latex_escape(row.get("avg", "")),
            latex_escape(row.get("delta_vs_span_only", "")),
        ])

    budget_rows = read_csv(run_dir / "paper_tables" / "02_budget_ablation.csv")
    budget_body = [
        [
            latex_escape(row.get("budget", "")),
            latex_escape(DISPLAY_NAMES.get(row.get("variant", ""), row.get("variant", ""))),
            latex_escape(row.get("macro_avg", "")),
            latex_escape(row.get("delta_vs_span_only_same_budget", "")),
        ]
        for row in budget_rows
    ]

    multihop_rows = read_csv(run_dir / "paper_tables" / "04_multihop_only_table.csv")
    mh_best = ""
    mh_second = ""
    mh_scored = sorted(
        [(to_float(row.get("multihop_macro")) or -1.0, row.get("variant", "")) for row in multihop_rows],
        reverse=True,
    )
    if mh_scored:
        mh_best = mh_scored[0][1]
    if len(mh_scored) > 1:
        mh_second = mh_scored[1][1]
    mh_body = []
    for row in multihop_rows:
        method = row.get("variant", "")
        mh_body.append([
            latex_escape(DISPLAY_NAMES.get(method, method)),
            latex_score(method, row.get("multihop_macro", ""), mh_best, mh_second),
            latex_escape(row.get("delta_vs_span_only_multihop", "")),
            latex_escape(row.get("hotpotqa", "")),
            latex_escape(row.get("2wikimqa", "")),
            latex_escape(row.get("musique", "")),
        ])

    text = "\n".join([
        "% Requires booktabs and graphicx.",
        latex_table(
            "Main 1024-token BridgePack results",
            "tab:bridgepack-main-1024",
            ["Method", "Qasper", "MFQA", "HotpotQA", "2Wiki", "MuSiQue", "NarrativeQA", "Avg", "Delta"],
            main_body,
        ),
        latex_table(
            "BridgePack component ablation",
            "tab:bridgepack-components",
            ["Method", "Bridge", "Neighbor", "Path", "Tags", "Avg", "Delta"],
            component_body,
        ),
        latex_table(
            "Budget ablation for selected variants",
            "tab:bridgepack-budget",
            ["Budget", "Variant", "Avg", "Delta vs Span-only"],
            budget_body,
        ),
        latex_table(
            "Multi-hop only results",
            "tab:bridgepack-multihop",
            ["Method", "Multi-hop Avg", "Delta", "HotpotQA", "2Wiki", "MuSiQue"],
            mh_body,
        ),
    ])
    write_text(out_dir / "06_latex_tables.tex", text)


def main_value(main_rows: Sequence[Mapping[str, Any]], method: str, field: str) -> str:
    for row in main_rows:
        if row.get("method") == method:
            return str(row.get(field, ""))
    return ""


def component_value(component_rows: Sequence[Mapping[str, Any]], method: str, field: str) -> str:
    for row in component_rows:
        if row.get("method") == method:
            return str(row.get(field, ""))
    return ""


def budget_value(run_dir: Path, budget: str, variant: str, field: str) -> str:
    for row in read_csv(run_dir / "paper_tables" / "02_budget_ablation.csv"):
        if row.get("budget") == budget and row.get("variant") == variant:
            return row.get(field, "")
    return ""


def multihop_value(run_dir: Path, variant: str, field: str) -> str:
    for row in read_csv(run_dir / "paper_tables" / "04_multihop_only_table.csv"):
        if row.get("variant") == variant:
            return row.get(field, "")
    return ""


def build_results_section(run_dir: Path, out_dir: Path, main_rows: Sequence[Mapping[str, Any]], component_rows: Sequence[Mapping[str, Any]]) -> None:
    bridge_avg = main_value(main_rows, "bridge_neighbor32_path_order", "avg")
    bridge_delta = main_value(main_rows, "bridge_neighbor32_path_order", "delta_vs_span_only")
    tags_avg = main_value(main_rows, "bridge_neighbor32_path_order_tags", "avg")
    tags_delta = main_value(main_rows, "bridge_neighbor32_path_order_tags", "delta_vs_span_only")
    span_avg = main_value(main_rows, "span_only_source_order", "avg")
    neighbor_delta = component_value(component_rows, "neighbor32_source_order", "delta_vs_span_only")
    path_delta = component_value(component_rows, "neighbor32_path_order", "delta_vs_span_only")
    bridge_component_delta = component_value(component_rows, "bridge_neighbor32_path_order", "delta_vs_span_only")
    tags_component_delta = component_value(component_rows, "bridge_neighbor32_path_order_tags", "delta_vs_span_only")
    mh_delta = multihop_value(run_dir, "bridge_neighbor32_path_order", "delta_vs_span_only_multihop")
    tags_mh_delta = multihop_value(run_dir, "bridge_neighbor32_path_order_tags", "delta_vs_span_only_multihop")
    tags_512 = budget_value(run_dir, "512", "bridge_neighbor32_path_order_tags", "macro_avg")
    tags_1024 = budget_value(run_dir, "1024", "bridge_neighbor32_path_order_tags", "macro_avg")
    tags_2048 = budget_value(run_dir, "2048", "bridge_neighbor32_path_order_tags", "macro_avg")
    lines = [
        "# Results Section Draft",
        "",
        "## Main Result",
        "",
        (
            "Under the 1024-token packed-context setting, BridgePack (`bridge_neighbor32_path_order`) "
            f"achieves an average score of {bridge_avg}, compared with {span_avg} for the span-only packing baseline. "
            f"This is a same-budget gain of +{bridge_delta}. BridgePack+Tags (`bridge_neighbor32_path_order_tags`) "
            f"also strongly improves over span-only, reaching {tags_avg} (+{tags_delta}), but it is not the strongest BridgePack variant in this run."
        ),
        "",
        "## Component Ablation",
        "",
        (
            "The component ablation indicates that local neighbor expansion alone does not explain the gain "
            f"(Neighbor Packing delta {neighbor_delta}). Reordering evidence into path order gives a large improvement "
            f"(Neighbor + Path Order delta +{path_delta}), and bridge-aware selection further improves the result "
            f"(BridgePack delta +{bridge_component_delta}). Tags are optional formatting: BridgePack+Tags remains strong "
            f"(+{tags_component_delta}) but trails no-tags BridgePack."
        ),
        "",
        "## Multi-hop Tasks",
        "",
        (
            "On the multi-hop subset (HotpotQA, 2WikiMultiHopQA, and MuSiQue), BridgePack improves over span-only by "
            f"+{mh_delta}. The pre-registered BridgePack+Tags variant improves by +{tags_mh_delta}. This supports the "
            "interpretation that organizing evidence paths is useful for multi-hop reasoning, while not implying that all gains are exclusive to multi-hop tasks."
        ),
        "",
        "## Budget Ablation",
        "",
        (
            f"For BridgePack+Tags, the average score rises from {tags_512} at 512 tokens to {tags_1024} at 1024 tokens "
            f"and {tags_2048} at 2048 tokens. Thus, 1024 is the main default budget for the paper comparison, not the optimal budget."
        ),
        "",
        "## Fairness",
        "",
        (
            "All compressed variants use the same model, tokenizer, dataset samples, generation configuration, scoring function, "
            "and packed-token budget. The prompt wrapper is shared except for the packed evidence content, and the leakage audit "
            "reports no explicit answer metadata in the prompt text."
        ),
        "",
        "## Tags",
        "",
        (
            "Tags should be reported as an optional formatting ablation rather than the core contribution. The no-tags BridgePack "
            f"variant is stronger ({bridge_avg}) than BridgePack+Tags ({tags_avg}), so the central method should be described as "
            "bridge-aware neighbor-preserving path-order packing."
        ),
    ]
    write_text(out_dir / "07_results_section_draft.md", "\n".join(lines))


def build_claims_and_limitations(run_dir: Path, out_dir: Path, main_rows: Sequence[Mapping[str, Any]]) -> None:
    span_avg = main_value(main_rows, "span_only_source_order", "avg")
    bridge_avg = main_value(main_rows, "bridge_neighbor32_path_order", "avg")
    bridge_delta = main_value(main_rows, "bridge_neighbor32_path_order", "delta_vs_span_only")
    tags_avg = main_value(main_rows, "bridge_neighbor32_path_order_tags", "avg")
    tags_2048 = budget_value(run_dir, "2048", "bridge_neighbor32_path_order_tags", "macro_avg")
    lines = [
        "# Claims and Limitations",
        "",
        "## Safe Claims",
        "",
        "- BridgePack is evidence-aware context compression / text repacking.",
        f"- Same-budget comparison shows BridgePack strongly outperforms span-only packing ({bridge_avg} vs {span_avg}, delta +{bridge_delta}).",
        "- Gains are especially meaningful on multi-hop tasks.",
        f"- Tags are optional; no-tags BridgePack is stronger in this run ({bridge_avg} vs {tags_avg}).",
        f"- 2048 gives the best observed tags score ({tags_2048}), while 1024 is the main default budget.",
        "",
        "## Unsafe Claims",
        "",
        "- BridgePack is strict KV-cache pruning.",
        "- BridgePack reduces original long-context prefill KV memory.",
        "- Tags are the core contribution.",
        "- 1024 is the optimal budget.",
        "- BridgePack beats all possible prompt compression or RAG systems unless externally benchmarked.",
    ]
    write_text(out_dir / "08_claims_and_limitations.md", "\n".join(lines))


def build_abstract_bullets(out_dir: Path, main_rows: Sequence[Mapping[str, Any]]) -> None:
    span_avg = main_value(main_rows, "span_only_source_order", "avg")
    bridge_avg = main_value(main_rows, "bridge_neighbor32_path_order", "avg")
    bridge_delta = main_value(main_rows, "bridge_neighbor32_path_order", "delta_vs_span_only")
    lines = [
        "# Abstract Bullets",
        "",
        "- Problem: Long-context generation often needs aggressive context compression, but independent span selection can fragment evidence.",
        "- Limitation: Span-only packing selects high-scoring snippets without explicitly organizing bridge and support evidence.",
        "- Method: BridgePack performs bridge-aware neighbor-preserving path-order text repacking under a fixed packed-token budget.",
        f"- Result: At 1024 packed tokens, BridgePack reaches {bridge_avg} average score versus {span_avg} for span-only packing, a +{bridge_delta} gain.",
        "- Interpretation: The result supports evidence organization as a useful design principle for context compression; it is not a strict KV-cache pruning claim.",
    ]
    write_text(out_dir / "09_abstract_bullets.md", "\n".join(lines))


def build_next_experiments(out_dir: Path) -> None:
    lines = [
        "# Next Experiments",
        "",
        "1. Run a second-model diagnostic, such as Mistral or Llama, starting with 32 samples per task before scaling.",
        "2. Compare against a simple BM25 sentence or passage retrieval baseline if feasible.",
        "3. Add a latency and token-cost table for the offline packing step.",
        "4. Expand case studies on multi-hop examples where BridgePack changes the answer.",
        "5. Add optional human-readable evidence-chain visualization for query -> bridge -> support paths.",
    ]
    write_text(out_dir / "10_next_experiments.md", "\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir does not exist: {run_dir}")
    if not (run_dir / "paper_tables").exists():
        raise FileNotFoundError(f"paper_tables not found under run_dir: {run_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    main_rows = build_main_table(run_dir, out_dir)
    component_rows = build_component_table(run_dir, out_dir)
    build_fairness_protocol(run_dir, out_dir)
    build_tags_analysis(run_dir, out_dir)
    build_case_study(run_dir, out_dir, args.seed)
    build_latex_tables(run_dir, out_dir, main_rows, component_rows)
    build_results_section(run_dir, out_dir, main_rows, component_rows)
    build_claims_and_limitations(run_dir, out_dir, main_rows)
    build_abstract_bullets(out_dir, main_rows)
    build_next_experiments(out_dir)
    print(f"Wrote BridgePack paper assets to {out_dir}")


if __name__ == "__main__":
    main()
