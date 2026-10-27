#!/usr/bin/env python3
"""Audit BridgePack evidence spans/pages against frozen SlimInfer selections.

This is a CPU-only diagnostic.  It does not run a model, does not generate or
score predictions, and does not modify the SlimInfer selector.  The alignment is
explicitly marked as proxy alignment: BridgePack span ids encode regex-token
positions from the original context, while SlimInfer final-layer used_blocks are
KV block ids over the generation prompt.  We therefore map BridgePack span
start/end ranges into 64-token proxy blocks and report the limitation.
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
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


DEFAULT_HISTORICAL_DIR = Path("results/drm_budget_neutral_swap_multihop200_20260706_042717")
DEFAULT_BRIDGEPACK_DIR = Path("results/bridgepack_exact_qwen_budget_1024_20260619_181059")
DEFAULT_VARIANT = "lexical_bridgepack_path_order"
DEFAULT_BUDGET = 1024
BLOCK_SIZE = 64
PAGE_SIZE_TOKENS = 256
SPAN_SIZE_TOKENS = 64

METHOD_ORIGINAL = "sliminfer_original"
METHOD_RANDOM_SWAP = "random_budget_neutral_swap"
METHOD_DRM_SWAP = "budget_neutral_drm_swap"

TINY_SAMPLE_INDICES = {
    "hotpotqa": [0, 1, 2, 3, 4, 5, 41, 56, 57, 117],
    "musique": [0, 1, 2, 3, 4, 20, 50, 54, 56, 134],
    "2wikimqa": list(range(10)),
}


JsonRow = Dict[str, Any]
Key = Tuple[str, int]


SPAN_ID_RE = re.compile(r"^p(?P<page>\d+)_s(?P<span>\d+)_t(?P<start>\d+)$")


def read_jsonl(path: Path) -> List[JsonRow]:
    if not path.exists():
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
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: serialize_csv(row.get(field, "")) for field in fields})


def serialize_csv(value: Any) -> Any:
    if isinstance(value, (list, dict, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return value


def key_of(row: Mapping[str, Any]) -> Key:
    return str(row["task"]), int(row["sample_index"])


def as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def finite_float_or_none(value: Any) -> float | None:
    try:
        if value in (None, "", "unavailable"):
            return None
        result = float(value)
        if not math.isfinite(result):
            return None
        return result
    except Exception:
        return None


def jaccard(left: Iterable[Any], right: Iterable[Any]) -> float:
    a, b = set(left), set(right)
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def preview(text: str, max_chars: int = 360) -> str:
    clean = re.sub(r"\s+", " ", text or "").strip()
    if len(clean) <= max_chars:
        return clean
    return clean[: max_chars - 3].rstrip() + "..."


def load_historical(historical_dir: Path) -> Dict[str, Dict[Key, JsonRow]]:
    path = historical_dir / "audit" / "per_sample_audit.jsonl"
    by_method: Dict[str, Dict[Key, JsonRow]] = defaultdict(dict)
    for row in read_jsonl(path):
        method = str(row.get("method_key") or "")
        if method:
            by_method[method][key_of(row)] = row
    missing = [method for method in [METHOD_ORIGINAL, METHOD_RANDOM_SWAP, METHOD_DRM_SWAP] if method not in by_method]
    if missing:
        raise RuntimeError(f"historical audit missing methods: {missing}")
    return dict(by_method)


def bridgepack_path(bridgepack_dir: Path, variant: str, budget: int, task: str) -> Path:
    return (
        bridgepack_dir
        / "packed_prompts_build"
        / "packed_prompts"
        / variant
        / f"budget_{budget}"
        / f"{task}.jsonl"
    )


def load_bridgepack_rows(bridgepack_dir: Path, variant: str, budget: int, tasks: Sequence[str]) -> Dict[Key, JsonRow]:
    rows_by_key: Dict[Key, JsonRow] = {}
    for task in tasks:
        path = bridgepack_path(bridgepack_dir, variant, budget, task)
        if not path.exists():
            continue
        rows = read_jsonl(path)
        for index, row in enumerate(rows):
            row = dict(row)
            row["sample_index"] = index
            rows_by_key[(task, index)] = row
    return rows_by_key


def span_range_from_id(span_id: str, neighbor_size: int) -> Tuple[int, int, int] | None:
    match = SPAN_ID_RE.match(str(span_id))
    if not match:
        return None
    page = int(match.group("page"))
    start = int(match.group("start"))
    page_start = page * PAGE_SIZE_TOKENS
    page_end = page_start + PAGE_SIZE_TOKENS
    core_end = start + SPAN_SIZE_TOKENS
    expanded_start = max(page_start, start - int(neighbor_size or 0))
    expanded_end = min(page_end, core_end + int(neighbor_size or 0))
    if expanded_end <= expanded_start:
        expanded_end = core_end
    return page, expanded_start, expanded_end


def proxy_blocks_for_spans(span_ids: Sequence[Any], neighbor_size: int) -> Tuple[List[int], List[Dict[str, Any]], List[str]]:
    blocks = set()
    ranges: List[Dict[str, Any]] = []
    invalid: List[str] = []
    for raw in span_ids:
        span_id = str(raw)
        parsed = span_range_from_id(span_id, neighbor_size)
        if parsed is None:
            invalid.append(span_id)
            continue
        page, start, end = parsed
        block_start = start // BLOCK_SIZE
        block_end = max(block_start, (max(start, end - 1)) // BLOCK_SIZE)
        block_ids = list(range(block_start, block_end + 1))
        blocks.update(block_ids)
        ranges.append({
            "span_id": span_id,
            "page_id": page,
            "expanded_start_token": start,
            "expanded_end_token": end,
            "proxy_block_ids": block_ids,
        })
    return sorted(blocks), ranges, invalid


def score_delta(
    historical: Mapping[str, Mapping[Key, Mapping[str, Any]]],
    method: str,
    key: Key,
    base_method: str = METHOD_ORIGINAL,
) -> float | None:
    if key not in historical.get(method, {}) or key not in historical.get(base_method, {}):
        return None
    left = finite_float_or_none(historical[method][key].get("prediction_score"))
    right = finite_float_or_none(historical[base_method][key].get("prediction_score"))
    if left is None or right is None:
        return None
    return left - right


def case_type(delta: float | None) -> str:
    if delta is None:
        return "unavailable"
    if delta > 1e-9:
        return "positive"
    if delta < -1e-9:
        return "negative"
    return "zero"


def audit_one(
    key: Key,
    historical: Mapping[str, Mapping[Key, Mapping[str, Any]]],
    bridge_rows: Mapping[Key, Mapping[str, Any]],
    variant: str,
    budget: int,
) -> JsonRow:
    task, sample_index = key
    original = historical.get(METHOD_ORIGINAL, {}).get(key)
    bridge = bridge_rows.get(key)
    sample_id = ""
    if original:
        sample_id = str(original.get("sample_id") or "")
    elif bridge:
        sample_id = str(bridge.get("sample_id") or "")
    base: JsonRow = {
        "task": task,
        "sample_index": sample_index,
        "sample_id": sample_id,
        "bridgepack_variant": variant,
        "bridgepack_budget": budget,
        "alignment_mode": "proxy_regex_token_to_sliminfer_block64",
        "exact_block_alignment_available": False,
        "query": bridge.get("question", "") if bridge else "",
        "bridgepack_span_ids": [],
        "bridgepack_page_ids": [],
        "bridgepack_evidence_text_preview": "",
        "bridgepack_proxy_block_ids": [],
        "bridgepack_span_proxy_ranges": [],
        "sliminfer_selected_block_ids": [],
        "sliminfer_used_tokens": [],
        "bridgepack_covered_by_sliminfer": False,
        "missing_bridgepack_evidence_blocks": [],
        "missing_bridgepack_evidence_count": None,
        "coverage_gap_score": None,
        "overlap_jaccard": None,
        "drm_swap_delta_if_available": score_delta(historical, METHOD_DRM_SWAP, key),
        "random_swap_delta_if_available": score_delta(historical, METHOD_RANDOM_SWAP, key),
        "case_type_positive_negative_zero_if_available": case_type(score_delta(historical, METHOD_DRM_SWAP, key)),
        "audit_status": "blocked",
        "blocked_reason": "",
    }
    blocked: List[str] = []
    if original is None:
        blocked.append("missing_sliminfer_original_audit")
    if bridge is None:
        blocked.append("missing_bridgepack_packed_prompt_row")
    if original is not None and bridge is not None:
        bridge_sample_id = str(bridge.get("sample_id") or "")
        if sample_id and bridge_sample_id and sample_id != bridge_sample_id:
            blocked.append(f"sample_id_mismatch_bridgepack={bridge_sample_id}")
    if blocked:
        base["blocked_reason"] = ";".join(blocked)
        return base

    span_ids = [str(x) for x in as_list(bridge.get("selected_span_ids"))]
    page_ids = sorted({int(x) for x in as_list(bridge.get("selected_page_ids")) if str(x) != ""})
    selected_blocks = sorted({int(x) for x in as_list(original.get("used_blocks"))})
    used_tokens = [int(x) for x in as_list(original.get("used_tokens"))]
    proxy_blocks, ranges, invalid = proxy_blocks_for_spans(span_ids, int(bridge.get("neighbor_size") or 0))
    if invalid:
        blocked.append(f"invalid_span_ids={invalid[:5]}")
    if not span_ids:
        blocked.append("missing_bridgepack_span_ids")
    if not selected_blocks:
        blocked.append("missing_sliminfer_selected_block_ids")
    missing_blocks = sorted(set(proxy_blocks) - set(selected_blocks))
    coverage_gap = len(missing_blocks) / max(1, len(set(proxy_blocks)))
    covered = bool(proxy_blocks) and not missing_blocks
    status = "pass_proxy_alignment" if not blocked else "partial_proxy_alignment_blocked_fields"
    blocked_reason = (
        "exact SlimInfer context-token/block alignment unavailable; "
        "BridgePack regex-token span ranges mapped to 64-token proxy blocks"
    )
    if blocked:
        blocked_reason += "; " + ";".join(blocked)
    base.update({
        "query": str(bridge.get("question") or ""),
        "bridgepack_span_ids": span_ids,
        "bridgepack_page_ids": page_ids,
        "bridgepack_evidence_text_preview": preview(str(bridge.get("packed_text") or bridge.get("packed_prompt") or "")),
        "bridgepack_proxy_block_ids": proxy_blocks,
        "bridgepack_span_proxy_ranges": ranges,
        "sliminfer_selected_block_ids": selected_blocks,
        "sliminfer_used_tokens": used_tokens,
        "bridgepack_covered_by_sliminfer": covered,
        "missing_bridgepack_evidence_blocks": missing_blocks,
        "missing_bridgepack_evidence_count": len(missing_blocks),
        "coverage_gap_score": coverage_gap,
        "overlap_jaccard": jaccard(proxy_blocks, selected_blocks),
        "audit_status": status,
        "blocked_reason": blocked_reason,
    })
    return base


def summarize(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    valid = [r for r in rows if r.get("missing_bridgepack_evidence_count") is not None]
    by_task: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    by_case: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in valid:
        by_task[str(row["task"])].append(row)
        by_case[str(row.get("case_type_positive_negative_zero_if_available"))].append(row)

    def mean(values: Sequence[float]) -> float | None:
        vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
        return sum(vals) / len(vals) if vals else None

    task_summary = {}
    for task, task_rows in sorted(by_task.items()):
        task_summary[task] = {
            "samples": len(task_rows),
            "fully_covered_count": sum(bool(r.get("bridgepack_covered_by_sliminfer")) for r in task_rows),
            "mean_missing_bridgepack_evidence_count": mean([r.get("missing_bridgepack_evidence_count") for r in task_rows]),
            "mean_coverage_gap_score": mean([r.get("coverage_gap_score") for r in task_rows]),
            "mean_overlap_jaccard": mean([r.get("overlap_jaccard") for r in task_rows]),
        }
    case_summary = {}
    for label, case_rows in sorted(by_case.items()):
        case_summary[label] = {
            "samples": len(case_rows),
            "mean_missing_bridgepack_evidence_count": mean([r.get("missing_bridgepack_evidence_count") for r in case_rows]),
            "mean_coverage_gap_score": mean([r.get("coverage_gap_score") for r in case_rows]),
            "mean_overlap_jaccard": mean([r.get("overlap_jaccard") for r in case_rows]),
        }
    missing_distribution = Counter()
    for row in valid:
        count = int(row.get("missing_bridgepack_evidence_count") or 0)
        if count == 0:
            bucket = "0"
        elif count <= 2:
            bucket = "1-2"
        elif count <= 5:
            bucket = "3-5"
        else:
            bucket = ">5"
        missing_distribution[bucket] += 1
    exact_block_alignment_available = all(bool(row.get("exact_block_alignment_available")) for row in rows) if rows else False
    nonidentical = any((row.get("missing_bridgepack_evidence_count") or 0) > 0 for row in valid)
    blocked_exact = not exact_block_alignment_available
    appendix_recommended = bool(valid) and nonidentical
    decision = "GO_APPENDIX_PROXY_DIAGNOSTIC" if appendix_recommended else "NO-GO"
    if blocked_exact:
        decision = "GO_APPENDIX_PROXY_DIAGNOSTIC_NO_EXACT_COVERAGE_CLAIM" if appendix_recommended else "NO-GO"
    reasons = []
    if not valid:
        reasons.append("no_overlap_rows_computable")
    if blocked_exact:
        reasons.append("exact_block_alignment_unavailable_proxy_only")
    if nonidentical:
        reasons.append("bridgepack_proxy_evidence_not_fully_covered_by_sliminfer")
    else:
        reasons.append("bridgepack_proxy_evidence_fully_overlaps_sliminfer")
    return {
        "samples": len(rows),
        "valid_proxy_overlap_rows": len(valid),
        "bridgepack_evidence_file_exists": bool(valid),
        "sliminfer_selected_block_info_exists": all(bool(row.get("sliminfer_selected_block_ids")) for row in valid) if valid else False,
        "exact_block_alignment_available": exact_block_alignment_available,
        "alignment_mode": "proxy_regex_token_to_sliminfer_block64",
        "overall": {
            "fully_covered_count": sum(bool(r.get("bridgepack_covered_by_sliminfer")) for r in valid),
            "fully_covered_rate": sum(bool(r.get("bridgepack_covered_by_sliminfer")) for r in valid) / max(1, len(valid)),
            "mean_missing_bridgepack_evidence_count": mean([r.get("missing_bridgepack_evidence_count") for r in valid]),
            "mean_coverage_gap_score": mean([r.get("coverage_gap_score") for r in valid]),
            "mean_overlap_jaccard": mean([r.get("overlap_jaccard") for r in valid]),
            "missing_distribution": dict(sorted(missing_distribution.items())),
        },
        "by_task": task_summary,
        "by_case_type": case_summary,
        "audit_status_counts": dict(Counter(str(r.get("audit_status")) for r in rows)),
        "go_no_go": {
            "decision": decision,
            "reasons": reasons,
            "recommend_bridgepack_appendix": appendix_recommended,
            "claim_level": (
                "appendix proxy diagnostic only; do not claim exact BridgePack-span/KV-block coverage"
                if blocked_exact else
                "exact coverage diagnostic"
            ),
        },
    }


def write_report(output_dir: Path, args: argparse.Namespace, summary: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# BridgePack + Frozen-SlimInfer Coverage Audit",
        "",
        f"- Result directory: `{output_dir.resolve()}`",
        f"- Historical SlimInfer input: `{args.historical_dir}`",
        f"- BridgePack input: `{args.bridgepack_dir}`",
        f"- BridgePack variant: `{args.variant}`",
        f"- Budget: `{args.budget}`",
        "- GPU/model generation used: **No**",
        "- Full LongBench run: **No**",
        "- SlimInfer selector modified: **No**",
        "",
        "## Availability",
        "",
        f"- BridgePack evidence files exist / rows computable: {summary['valid_proxy_overlap_rows']}/{summary['samples']}",
        f"- SlimInfer selected block information exists: **{summary['sliminfer_selected_block_info_exists']}**",
        f"- Exact block alignment available: **{summary['exact_block_alignment_available']}**",
        f"- Alignment mode: `{summary['alignment_mode']}`",
        "- Limitation: BridgePack span ids use regex-token source positions; SlimInfer used_blocks are KV block ids over the generation prompt. The reported overlap is proxy alignment, not exact text-span/KV-block alignment.",
        "",
        "## Overall proxy overlap / coverage",
        "",
        f"- Fully covered by SlimInfer: {summary['overall']['fully_covered_count']}/{summary['valid_proxy_overlap_rows']} ({summary['overall']['fully_covered_rate']:.2%})",
        f"- Mean missing BridgePack proxy evidence blocks: {summary['overall']['mean_missing_bridgepack_evidence_count']:.4f}",
        f"- Mean coverage gap score: {summary['overall']['mean_coverage_gap_score']:.4f}",
        f"- Mean overlap Jaccard: {summary['overall']['mean_overlap_jaccard']:.4f}",
        f"- Missing evidence distribution: `{json.dumps(summary['overall']['missing_distribution'], ensure_ascii=False)}`",
        "",
        "## By task",
        "",
        "| task | samples | fully covered | mean missing | mean gap | mean Jaccard |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for task, item in summary["by_task"].items():
        lines.append(
            f"| {task} | {item['samples']} | {item['fully_covered_count']} | "
            f"{item['mean_missing_bridgepack_evidence_count']:.4f} | {item['mean_coverage_gap_score']:.4f} | {item['mean_overlap_jaccard']:.4f} |"
        )
    lines.extend([
        "",
        "## By DRM-Swap case type",
        "",
        "| case type | samples | mean missing | mean gap | mean Jaccard |",
        "|---|---:|---:|---:|---:|",
    ])
    for label, item in summary["by_case_type"].items():
        lines.append(
            f"| {label} | {item['samples']} | {item['mean_missing_bridgepack_evidence_count']:.4f} | "
            f"{item['mean_coverage_gap_score']:.4f} | {item['mean_overlap_jaccard']:.4f} |"
        )
    lines.extend([
        "",
        "## Highest missing-evidence proxy cases",
        "",
        "| task | sample_index | case | missing | gap | jaccard | sample_id |",
        "|---|---:|---|---:|---:|---:|---|",
    ])
    top_rows = sorted(
        rows,
        key=lambda row: (
            float(row.get("missing_bridgepack_evidence_count") or -1),
            float(row.get("coverage_gap_score") or -1),
        ),
        reverse=True,
    )[:8]
    for row in top_rows:
        lines.append(
            f"| {row.get('task')} | {row.get('sample_index')} | {row.get('case_type_positive_negative_zero_if_available')} | "
            f"{row.get('missing_bridgepack_evidence_count')} | {float(row.get('coverage_gap_score') or 0):.4f} | "
            f"{float(row.get('overlap_jaccard') or 0):.4f} | `{row.get('sample_id')}` |"
        )
    lines.extend([
        "",
        "## GO / NO-GO",
        "",
        f"- Decision: **{summary['go_no_go']['decision']}**",
        f"- Recommend BridgePack appendix: **{summary['go_no_go']['recommend_bridgepack_appendix']}**",
        f"- Claim level: {summary['go_no_go']['claim_level']}",
    ])
    for reason in summary["go_no_go"]["reasons"]:
        lines.append(f"- Reason: `{reason}`")
    lines.extend([
        "",
        "This audit is diagnostic only. It does not replay quality and does not claim performance improvement.",
    ])
    (output_dir / "bridgepack_sliminfer_coverage_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Path:
    historical_dir = Path(args.historical_dir)
    bridgepack_dir = Path(args.bridgepack_dir)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("results") / f"bridgepack_sliminfer_coverage_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    historical = load_historical(historical_dir)
    tasks = list(TINY_SAMPLE_INDICES)
    bridge_rows = load_bridgepack_rows(bridgepack_dir, args.variant, int(args.budget), tasks)
    rows: List[JsonRow] = []
    for task, indices in TINY_SAMPLE_INDICES.items():
        for sample_index in indices:
            rows.append(audit_one((task, sample_index), historical, bridge_rows, args.variant, int(args.budget)))
    summary = summarize(rows)
    summary.update({
        "input_artifacts": {
            "historical_dir": str(historical_dir),
            "bridgepack_dir": str(bridgepack_dir),
            "variant": args.variant,
            "budget": int(args.budget),
        },
        "model_or_gpu_used": False,
        "generation_performed": False,
        "quality_replay_performed": False,
        "sliminfer_selector_modified": False,
        "sample_indices_by_task": TINY_SAMPLE_INDICES,
    })

    fields = [
        "task",
        "sample_index",
        "sample_id",
        "query",
        "bridgepack_variant",
        "bridgepack_span_ids",
        "bridgepack_page_ids",
        "bridgepack_evidence_text_preview",
        "bridgepack_proxy_block_ids",
        "sliminfer_selected_block_ids",
        "sliminfer_used_tokens",
        "bridgepack_covered_by_sliminfer",
        "missing_bridgepack_evidence_blocks",
        "missing_bridgepack_evidence_count",
        "coverage_gap_score",
        "overlap_jaccard",
        "drm_swap_delta_if_available",
        "random_swap_delta_if_available",
        "case_type_positive_negative_zero_if_available",
        "audit_status",
        "blocked_reason",
    ]
    write_jsonl(output_dir / "bridgepack_sliminfer_coverage_audit.jsonl", rows)
    write_csv(output_dir / "bridgepack_sliminfer_coverage_audit.csv", rows, fields)
    write_json(output_dir / "bridgepack_sliminfer_coverage_summary.json", summary)
    write_report(output_dir, args, summary, rows)
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical_dir", default=str(DEFAULT_HISTORICAL_DIR))
    parser.add_argument("--bridgepack_dir", default=str(DEFAULT_BRIDGEPACK_DIR))
    parser.add_argument("--variant", default=DEFAULT_VARIANT)
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    parser.add_argument("--output_dir", default="")
    return parser.parse_args()


def main() -> None:
    print(run(parse_args()))


if __name__ == "__main__":
    main()
