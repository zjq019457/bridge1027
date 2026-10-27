#!/usr/bin/env python3
"""CPU-only offline diagnostics for evidence-aware text packing."""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import statistics
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bridgekv.evidence_text_packing import (  # noqa: E402
    PackedEvidence,
    jaccard,
    pack_evidence,
    packed_to_json,
)

DEFAULT_TASKS = "qasper,hotpotqa,2wikimqa,musique,multifieldqa_en"
DEFAULT_BUDGETS = "512,1024,2048"
DEFAULT_VARIANTS = (
    "head_truncate_source_order,span_only_source_order,neighbor16_source_order,neighbor32_source_order,"
    "neighbor32_score_order,neighbor32_path_order,bridge_neighbor32_path_order,"
    "bridge_neighbor32_path_order_tags"
)
DEFAULT_LONGBENCH = "/home/yu-bai/Desktop/jiaqizhang/HeadKV/data/LongBench"


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(fields),
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _parse_list(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_ints(value: str) -> List[int]:
    return [int(item) for item in _parse_list(value)]


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _candidate_dataset_roots(extra_root: str = "") -> List[Path]:
    roots = []
    if extra_root:
        roots.append(Path(extra_root).expanduser())
    roots.extend([
        Path(DEFAULT_LONGBENCH),
        REPO_ROOT / "data",
        REPO_ROOT / "datasets",
        REPO_ROOT / "LongBench",
        REPO_ROOT.parent / "HeadKV" / "data" / "LongBench",
    ])
    seen = set()
    out = []
    for root in roots:
        resolved = str(root)
        if resolved in seen:
            continue
        seen.add(resolved)
        if root.exists():
            out.append(root)
    return out


def _find_task_file(task: str, roots: Sequence[Path]) -> Path:
    names = [f"{task}.jsonl", f"{task}.json", f"{task}_e.jsonl"]
    for root in roots:
        for name in names:
            direct = root / name
            if direct.exists():
                return direct
        for path in root.rglob("*"):
            if path.name in names and path.is_file():
                return path
    raise FileNotFoundError(f"Could not find local LongBench file for task {task}")


def _sample_id(task: str, row: Mapping[str, Any], index: int) -> str:
    for key in ("sample_id", "_id", "id"):
        if row.get(key) not in (None, ""):
            return str(row[key])
    return f"{task}_{index:05d}"


def _load_samples(task: str, roots: Sequence[Path], max_samples: int) -> List[Dict[str, Any]]:
    path = _find_task_file(task, roots)
    rows = _read_jsonl(path) if path.suffix == ".jsonl" else json.loads(path.read_text(encoding="utf-8"))
    if isinstance(rows, dict):
        rows = rows.get("data", rows.get("rows", []))
    samples: List[Dict[str, Any]] = []
    selected_rows = rows if max_samples == 0 else rows[:max_samples]
    for idx, row in enumerate(selected_rows):
        query = str(row.get("input", row.get("question", row.get("query", ""))))
        context = str(row.get("context", row.get("document", row.get("passage", ""))))
        if not context:
            context = "\n\n".join(str(v) for k, v in row.items() if k not in {"answers", "all_classes"})
        samples.append({
            "task": task,
            "sample_id": _sample_id(task, row, idx),
            "question": query,
            "context": context,
            "answers": row.get("answers", []),
            "gold_analysis_only_answers": row.get("answers", []),
            "all_classes": row.get("all_classes", None),
            "length": row.get("length", 0),
            "source_file": str(path),
        })
    return samples


def _mean(values: Sequence[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _preview(text: str, max_chars: int = 900) -> str:
    cleaned = " ".join((text or "").split())
    return cleaned[:max_chars] + ("..." if len(cleaned) > max_chars else "")


def _write_packed_prompt(
    out_dir: Path,
    packed: PackedEvidence,
    sample: Mapping[str, Any],
    prompt_templates: Mapping[str, str],
) -> None:
    path = (
        out_dir / "packed_prompts" / packed.variant /
        f"budget_{packed.budget}" / f"{packed.task}.jsonl"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = packed_to_json(packed)
    generation_prompt = packed.packed_text
    prompt_template = str(prompt_templates.get(packed.task, ""))
    if prompt_template:
        try:
            generation_prompt = prompt_template.format(
                **{
                    "context": packed.packed_text,
                    "input": sample.get("question", ""),
                    "question": sample.get("question", ""),
                }
            )
        except Exception:
            generation_prompt = packed.packed_text
    payload.update({
        "question": sample["question"],
        "packed_prompt": packed.packed_text,
        "generation_prompt": generation_prompt,
        "original_context": sample.get("context", ""),
        "answers": sample.get("answers", []),
        "gold_analysis_only_answers": sample.get("gold_analysis_only_answers", []),
        "all_classes": sample.get("all_classes", None),
        "length": sample.get("length", 0),
        "source_file": sample.get("source_file", ""),
    })
    with path.open("a", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)
        handle.write("\n")


def _packing_stats_row(packed: PackedEvidence) -> Dict[str, Any]:
    return {
        "task": packed.task,
        "sample_id": packed.sample_id,
        "variant": packed.variant,
        "budget": packed.budget,
        "source_token_count": packed.source_token_count,
        "packed_token_count": packed.packed_token_count,
        "compression_ratio": round(packed.compression_ratio, 6),
        "selected_span_count": len(packed.selected_span_ids),
        "selected_page_count": len(packed.selected_page_ids),
        "used_bridge_path": packed.used_bridge_path,
        "used_tags": packed.used_tags,
        "neighbor_size": packed.neighbor_size,
        "truncated": packed.truncated,
    }


def _path_stats_row(packed: PackedEvidence, audit: Mapping[str, Any]) -> Dict[str, Any]:
    q_vals = [float(v) for v in audit.get("query_link_values", [])]
    b_vals = [float(v) for v in audit.get("bridge_link_values", [])]
    return {
        "task": packed.task,
        "sample_id": packed.sample_id,
        "variant": packed.variant,
        "budget": packed.budget,
        "path_count": int(audit.get("path_count", 0) or 0),
        "query_link_mean": round(_mean(q_vals), 6),
        "bridge_link_mean": round(_mean(b_vals), 6),
        "selected_page_count": len(packed.selected_page_ids),
        "cross_page_path_count": int(audit.get("cross_page_path_count", 0) or 0),
        "same_page_path_count": int(audit.get("same_page_path_count", 0) or 0),
    }


def _build_overlap_rows(
    packed_by_key: Mapping[Tuple[str, str, int, str], PackedEvidence],
    variants: Sequence[str],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    task_sample_budgets = sorted({(k[0], k[1], k[2]) for k in packed_by_key})
    for task, sample_id, budget in task_sample_budgets:
        for left, right in itertools.combinations(variants, 2):
            a = packed_by_key.get((task, sample_id, budget, left))
            b = packed_by_key.get((task, sample_id, budget, right))
            if not a or not b:
                continue
            rows.append({
                "task": task,
                "sample_id": sample_id,
                "budget": budget,
                "variant_a": left,
                "variant_b": right,
                "span_jaccard": round(jaccard(a.selected_span_ids, b.selected_span_ids), 6),
                "page_jaccard": round(jaccard(a.selected_page_ids, b.selected_page_ids), 6),
            })
    return rows


def _build_budget_rows(packing_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str, int], List[Mapping[str, Any]]] = defaultdict(list)
    for row in packing_rows:
        groups[(str(row["task"]), str(row["variant"]), int(row["budget"]))].append(row)
    out: List[Dict[str, Any]] = []
    for (task, variant, budget), rows in sorted(groups.items()):
        out.append({
            "task": task,
            "variant": variant,
            "budget": budget,
            "mean_packed_tokens": round(_mean([float(r["packed_token_count"]) for r in rows]), 3),
            "mean_compression_ratio": round(_mean([float(r["compression_ratio"]) for r in rows]), 6),
            "mean_selected_spans": round(_mean([float(r["selected_span_count"]) for r in rows]), 3),
            "mean_selected_pages": round(_mean([float(r["selected_page_count"]) for r in rows]), 3),
        })
    return out


def _write_examples(
    out_dir: Path,
    packed_by_key: Mapping[Tuple[str, str, int, str], PackedEvidence],
    sample_by_key: Mapping[Tuple[str, str], Mapping[str, Any]],
    budgets: Sequence[int],
) -> None:
    budget = 1024 if 1024 in budgets else max(budgets)
    variants = [
        "span_only_source_order",
        "neighbor32_source_order",
        "bridge_neighbor32_path_order_tags",
    ]
    musique_ids = [
        sample_id
        for task, sample_id in sorted(sample_by_key)
        if task == "musique"
    ][:3]
    lines = ["# MuSiQue Evidence Packing Examples", ""]
    for sample_id in musique_ids:
        sample = sample_by_key[("musique", sample_id)]
        lines.extend([
            f"## sample_id={sample_id}",
            "",
            f"Question: {sample['question']}",
            "",
        ])
        for variant in variants:
            packed = packed_by_key.get(("musique", sample_id, budget, variant))
            if not packed:
                continue
            lines.extend([
                f"### {variant} budget={budget}",
                "",
                f"- selected evidence ids: `{','.join(packed.selected_span_ids[:12])}`",
                f"- source pages: `{','.join(str(p) for p in packed.selected_page_ids[:12])}`",
                "",
                "```text",
                _preview(packed.packed_text, max_chars=1100),
                "```",
                "",
            ])
    (out_dir / "audits" / "05_examples_musique.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def _recommendation(
    packed_by_key: Mapping[Tuple[str, str, int, str], PackedEvidence],
    overlap_rows: Sequence[Mapping[str, Any]],
    path_rows: Sequence[Mapping[str, Any]],
    budget_rows: Sequence[Mapping[str, Any]],
    variants: Sequence[str],
    budgets: Sequence[int],
) -> str:
    def overlaps_for(a: str, b: str) -> List[Mapping[str, Any]]:
        return [
            row for row in overlap_rows
            if {row["variant_a"], row["variant_b"]} == {a, b}
        ]

    bridge_vs_span = overlaps_for("span_only_source_order", "bridge_neighbor32_path_order_tags")
    bridge_diff = bool(bridge_vs_span) and any(
        float(row["span_jaccard"]) < 1.0 or float(row["page_jaccard"]) < 1.0
        for row in bridge_vs_span
    )
    path_vs_source = [
        (
            packed_by_key.get((task, sid, budget, "neighbor32_source_order")),
            packed_by_key.get((task, sid, budget, "neighbor32_path_order")),
        )
        for task, sid, budget, variant in packed_by_key
        if variant == "neighbor32_source_order"
    ]
    path_order_diff = any(
        a and b and a.selected_span_ids != b.selected_span_ids
        for a, b in path_vs_source
    )
    span_rows = [
        row for row in budget_rows
        if row["variant"] == "span_only_source_order"
    ]
    neighbor_rows = [
        row for row in budget_rows
        if row["variant"] == "neighbor32_source_order"
    ]
    span_mean = _mean([float(row["mean_packed_tokens"]) for row in span_rows])
    neighbor_mean = _mean([float(row["mean_packed_tokens"]) for row in neighbor_rows])
    token_ratio = neighbor_mean / max(1.0, span_mean)
    neighbor_increases = token_ratio > 1.10
    bridge_paths = [
        row for row in path_rows
        if str(row["variant"]).startswith("bridge_neighbor32")
    ]
    cross_page_mean = _mean([float(row["cross_page_path_count"]) for row in bridge_paths])
    promising = (
        "bridge_neighbor32_path_order_tags"
        if bridge_diff and cross_page_mean > 0
        else "neighbor32_path_order"
    )
    first_budget = 1024 if 1024 in budgets else max(budgets)
    ready = bridge_diff and path_order_diff and all(
        packed.packed_token_count <= packed.budget
        for packed in packed_by_key.values()
    )
    return "\n".join([
        "# Evidence Text Packing Offline Recommendation",
        "",
        "This is CPU-only prompt construction. No generation or model loading was run.",
        "",
        "## Required Answers",
        "",
        f"1. Bridge-aware packing selects different pages/spans than independent top-span packing: **{bridge_diff}**.",
        f"2. `path_order` changes evidence organization compared with `source_order`: **{path_order_diff}**.",
        f"3. Neighbor packing substantially increases packed tokens: **{neighbor_increases}** "
        f"(neighbor32/source token ratio {token_ratio:.3f}).",
        f"4. Most promising later GPU-generation variant: **`{promising}`**.",
        f"5. First budget to try when GPU is available: **{first_budget}**.",
        f"6. Ready for a later 32-sample generation diagnostic: **{ready}**.",
        "",
        "## Notes",
        "",
        f"- Mean bridge cross-page path count: {cross_page_mean:.3f}.",
        "- Gold answers, when present, are written only as `gold_analysis_only_answers` "
        "and are not used by selector logic.",
        "- Treat these outputs as a context-compression baseline/teacher candidate, not as strict KV pruning.",
        "",
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default=DEFAULT_TASKS)
    parser.add_argument("--max_samples", type=int, default=32)
    parser.add_argument("--budgets", default=DEFAULT_BUDGETS)
    parser.add_argument("--variants", default=DEFAULT_VARIANTS)
    parser.add_argument("--out_dir", default="")
    parser.add_argument("--longbench_dir", default=DEFAULT_LONGBENCH)
    args = parser.parse_args()

    tasks = _parse_list(args.tasks)
    budgets = _parse_ints(args.budgets)
    variants = _parse_list(args.variants)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / "results" / f"evidence_text_packing_offline_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    roots = _candidate_dataset_roots(args.longbench_dir)

    sample_by_key: Dict[Tuple[str, str], Mapping[str, Any]] = {}
    packing_rows: List[Dict[str, Any]] = []
    path_rows: List[Dict[str, Any]] = []
    packed_by_key: Dict[Tuple[str, str, int, str], PackedEvidence] = {}

    manifest = {
        "tasks": tasks,
        "max_samples": args.max_samples,
        "budgets": budgets,
        "variants": variants,
        "dataset_roots": [str(root) for root in roots],
        "cpu_only": True,
        "model_inference": False,
        "samples": {},
    }
    prompt_templates = _load_json(REPO_ROOT / "configs" / "dataset2prompt.json")
    for task in tasks:
        samples = _load_samples(task, roots, args.max_samples)
        manifest["samples"][task] = len(samples)
        for sample in samples:
            sample_by_key[(task, sample["sample_id"])] = sample
            for budget in budgets:
                for variant in variants:
                    packed, audit = pack_evidence(
                        sample_id=sample["sample_id"],
                        task=task,
                        query=sample["question"],
                        context=sample["context"],
                        variant=variant,
                        budget=budget,
                    )
                    _write_packed_prompt(out_dir, packed, sample, prompt_templates)
                    packed_by_key[(task, sample["sample_id"], budget, variant)] = packed
                    packing_rows.append(_packing_stats_row(packed))
                    path_rows.append(_path_stats_row(packed, audit))

    overlap_rows = _build_overlap_rows(packed_by_key, variants)
    budget_rows = _build_budget_rows(packing_rows)
    audits = out_dir / "audits"
    _write_csv(
        audits / "01_packing_stats.csv",
        packing_rows,
        [
            "task", "sample_id", "variant", "budget",
            "source_token_count", "packed_token_count", "compression_ratio",
            "selected_span_count", "selected_page_count", "used_bridge_path",
            "used_tags", "neighbor_size", "truncated",
        ],
    )
    _write_csv(
        audits / "02_variant_overlap.csv",
        overlap_rows,
        [
            "task", "sample_id", "budget", "variant_a", "variant_b",
            "span_jaccard", "page_jaccard",
        ],
    )
    _write_csv(
        audits / "03_path_structure_stats.csv",
        path_rows,
        [
            "task", "sample_id", "variant", "budget", "path_count",
            "query_link_mean", "bridge_link_mean", "selected_page_count",
            "cross_page_path_count", "same_page_path_count",
        ],
    )
    _write_csv(
        audits / "04_budget_stats.csv",
        budget_rows,
        [
            "task", "variant", "budget", "mean_packed_tokens",
            "mean_compression_ratio", "mean_selected_spans", "mean_selected_pages",
        ],
    )
    _write_examples(out_dir, packed_by_key, sample_by_key, budgets)
    (audits / "06_recommendation.md").write_text(
        _recommendation(packed_by_key, overlap_rows, path_rows, budget_rows, variants, budgets),
        encoding="utf-8",
    )
    (out_dir / "config.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(f"OUT_DIR={out_dir}")


if __name__ == "__main__":
    main()
