#!/usr/bin/env python3
"""Expansion collapse postmortem from 64-sample structured rescue dumps (no inference)."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _agg_rules(scores: List[float]) -> Dict[str, float]:
    if not scores:
        return {"max": 0.0, "mean_top2": 0.0, "sum_topk": 0.0, "rank_weighted_sum": 0.0}
    ranked = sorted(scores, reverse=True)
    top2 = ranked[:2]
    topk = ranked[: min(3, len(ranked))]
    rw = sum(s / (i + 1) for i, s in enumerate(ranked[:5]))
    return {
        "max": max(scores),
        "mean_top2": sum(top2) / len(top2),
        "sum_topk": sum(topk),
        "rank_weighted_sum": rw,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    root = Path(args.input_root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    mapping_rows: List[Dict[str, Any]] = []
    page_scores: Dict[Tuple[str, str, str, int], List[float]] = defaultdict(list)
    component_rows: List[Dict[str, Any]] = []

    for method in ("rh_pyramid_srm", "bridge_graph_srm"):
        for dump in (root / "runs" / method).rglob("candidate_score_dump.jsonl"):
            for row in _read_jsonl(dump):
                if not row.get("selected"):
                    continue
                sid = str(row.get("sample_id", ""))
                task = str(row.get("task", ""))
                mb = str(row.get("microblock_id", ""))
                page = int(row.get("page_id", row.get("block_id", -1)))
                mapping_rows.append({
                    "sample_id": sid,
                    "task": task,
                    "method": method,
                    "microblock_id": mb,
                    "parent_page_id": page,
                    "final_score": row.get("final_score", ""),
                    "selected_rank": row.get("rank", ""),
                    "drm_score": row.get("drm_score", ""),
                    "rh_score": row.get("rh_score", ""),
                    "bridge_score": row.get("bridge_score", ""),
                })
                try:
                    page_scores[(sid, task, method, page)].append(float(row["final_score"]))
                except (TypeError, ValueError, KeyError):
                    pass
                rank = int(row.get("rank", 9999) or 9999)
                if rank <= 20:
                    component_rows.append({
                        "sample_id": sid,
                        "task": task,
                        "method": method,
                        "candidate_id": row.get("candidate_id", ""),
                        "rank": rank,
                        "drm_score": row.get("drm_score", ""),
                        "rh_score": row.get("rh_score", ""),
                        "bridge_score": row.get("bridge_score", ""),
                        "redundancy_score": row.get("redundancy_score", ""),
                        "final_score": row.get("final_score", ""),
                        "selected": row.get("selected", 0),
                    })

    with (out / "01_microblock_to_page_mapping.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(mapping_rows[0].keys()) if mapping_rows else [
            "sample_id", "task", "method", "microblock_id", "parent_page_id", "final_score", "selected_rank",
            "drm_score", "rh_score", "bridge_score",
        ])
        w.writeheader()
        for row in mapping_rows:
            w.writerow(row)

    agg_rows: List[Dict[str, Any]] = []
    for (sid, task, method, page), scores in sorted(page_scores.items()):
        aggs = _agg_rules(scores)
        agg_rows.append({
            "sample_id": sid,
            "task": task,
            "method": method,
            "parent_page_id": page,
            "n_microblocks_selected": len(scores),
            **aggs,
        })
    with (out / "02_parent_page_aggregation_audit.csv").open("w", encoding="utf-8", newline="") as f:
        fields = ["sample_id", "task", "method", "parent_page_id", "n_microblocks_selected",
                  "max", "mean_top2", "sum_topk", "rank_weighted_sum"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in agg_rows:
            w.writerow(row)

    with (out / "03_rh_bridge_score_component_audit.csv").open("w", encoding="utf-8", newline="") as f:
        fields = ["sample_id", "task", "method", "candidate_id", "rank", "drm_score", "rh_score",
                  "bridge_score", "redundancy_score", "final_score", "selected"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in sorted(component_rows, key=lambda r: (r["task"], r["sample_id"], r["method"], r["rank"]))[:5000]:
            w.writerow(row)

    # Erasure cases from final_selection dumps
    rh_sel: Dict[str, Set[str]] = {}
    br_sel: Dict[str, Set[str]] = {}
    rh_pages: Dict[str, Set[int]] = {}
    br_pages: Dict[str, Set[int]] = {}
    for method, mb_store, pg_store in [
        ("rh_pyramid_srm", rh_sel, rh_pages),
        ("bridge_graph_srm", br_sel, br_pages),
    ]:
        for dump in (root / "runs" / method).rglob("final_selection_dump.jsonl"):
            for row in _read_jsonl(dump):
                key = f"{row.get('task')}:{row.get('sample_id')}"
                mbs = row.get("selected_microblock_ids_before_expansion") or []
                mb_store[key] = {str(x) for x in mbs}
                pg_store[key] = {int(x) for x in row.get("rescued_ids", [])}

    erasure_cases: List[str] = []
    keys = sorted(set(rh_sel) | set(br_sel))
    for key in keys:
        mb_r, mb_b = rh_sel.get(key, set()), br_sel.get(key, set())
        pg_r, pg_b = rh_pages.get(key, set()), br_pages.get(key, set())
        if mb_r != mb_b and pg_r == pg_b:
            erasure_cases.append(
                f"- **{key}**: microblocks rh={sorted(mb_r)[:5]} bridge={sorted(mb_b)[:5]} → same pages {sorted(pg_b)}"
            )
    (out / "04_expansion_erasure_cases.md").write_text(
        "# Expansion erasure cases\n\n"
        f"Count: {len(erasure_cases)} samples where rh≠bridge microblocks but identical rescued pages.\n\n"
        + "\n".join(erasure_cases[:30])
        + ("\n\n...(truncated)" if len(erasure_cases) > 30 else "")
        + "\n",
        encoding="utf-8",
    )

    # Would aggregation change ranking?
    rule_changes = 0
    for key in keys:
        for method in ("rh_pyramid_srm", "bridge_graph_srm"):
            pages = [r for r in agg_rows if f"{r['task']}:{r['sample_id']}" == key and r["method"] == method]
            if len(pages) < 2:
                continue
            by_max = sorted(pages, key=lambda r: r["max"], reverse=True)[:3]
            by_rw = sorted(pages, key=lambda r: r["rank_weighted_sum"], reverse=True)[:3]
            if [r["parent_page_id"] for r in by_max] != [r["parent_page_id"] for r in by_rw]:
                rule_changes += 1

    rec = [
        "# Patch recommendation",
        "",
        "## Diagnosis",
        "- Real path wiring is correct; collapse is at parent-page selection after microblock scoring.",
        f"- Erasure cases (different microblocks, same pages): {len(erasure_cases)}",
        f"- Parent-page ranking would change under alternate aggregation in {rule_changes} method×sample slices.",
        "",
        "## Recommended minimal patch: **Option A** — parent page aggregation by top-k microblock evidence",
        "",
        "Add `--microblock_parent_aggregation max|mean_top2|sum_topk|rank_weighted_sum` (default `max`).",
        "For structured methods only, aggregate microblock scores per parent page before greedy page pick.",
        "Keep `fixed_drm_add` on existing DRM block path unchanged.",
        "",
        "Rationale: score components differ between rh and bridge, but greedy per-microblock pick + one-page-per-block",
        "collapse makes methods agree. Aggregating within page may preserve scorer differences.",
        "",
        "Do **not** run 128-sample or full LongBench until ablation shows rh≠bridge page Jaccard mean < 0.95.",
        "",
        "Option D (negative ablation only) applies if aggregation ablation still ties — structured methods document",
        "path fix success but scoring/expansion does not beat fixed DRM-Add.",
    ]
    (out / "05_patch_recommendation.md").write_text("\n".join(rec) + "\n", encoding="utf-8")
    print(f"wrote postmortem to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
