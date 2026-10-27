#!/usr/bin/env python3
"""Post-32-sample real-path audit: microblock vs page expansion overlap (no inference)."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _jaccard(a: Set[Any], b: Set[Any]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def _find_files(root: Path, name: str) -> List[Path]:
    return sorted(root.rglob(name))


def _layer26_rows(root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    audit_dir = root / "audit"
    if audit_dir.is_dir():
        for p in sorted(audit_dir.glob("*.jsonl")):
            method = p.stem
            for row in _read_jsonl(p):
                if int(row.get("layer_id", -1)) == 26:
                    row = dict(row)
                    row.setdefault("method", method)
                    rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    root = Path(args.input_root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    structured = {
        "rh_pyramid_srm",
        "bridge_graph_srm",
        "distilled_blockdrm",
        "deferred_uncertainty_srm",
        "evidence_density_srm",
    }

    # 01 path marker
    markers: List[Dict[str, Any]] = []
    for p in _find_files(root, "path_marker.jsonl"):
        for row in _read_jsonl(p):
            row["source_file"] = str(p)
            markers.append(row)
    with (out / "01_path_marker_audit.csv").open("w", encoding="utf-8", newline="") as f:
        fields = [
            "sample_id", "task", "method", "entered_structured_selector", "entered_real_fixed_block_path",
            "entered_cache_construction_path", "used_microblock_first_selector",
            "used_candidate_score_dump", "used_final_selection_dump", "source_file",
        ]
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in markers:
            w.writerow(row)

    # 02 microblock count (layer 26 preferred)
    l26 = _layer26_rows(root)
    mb_rows: List[Dict[str, Any]] = []
    seen_mb: Set[Tuple[str, str, str]] = set()
    for row in l26:
        method = str(row.get("method", row.get("rescue_score", "")))
        key = (str(row.get("task", "")), str(row.get("sample_id", "")), method)
        if key in seen_mb:
            continue
        seen_mb.add(key)
        mb_rows.append({
            "sample_id": row.get("sample_id", ""),
            "task": row.get("task", ""),
            "method": method,
            "layer_id": 26,
            "selected_microblock_count": row.get("selected_microblock_count", 0),
            "rescued_count": row.get("rescued_count", row.get("rescue_count", "")),
            "rescue_selector": row.get("rescue_selector", ""),
        })
    for row in _read_jsonl_batch(root, "final_selection_dump.jsonl"):
        key = (str(row.get("task", "")), str(row.get("sample_id", "")), str(row.get("method", "")))
        if key in seen_mb:
            continue
        seen_mb.add(key)
        mb_rows.append({
            "sample_id": row.get("sample_id", ""),
            "task": row.get("task", ""),
            "method": row.get("method", ""),
            "layer_id": "",
            "selected_microblock_count": row.get("selected_microblock_count", 0),
            "rescued_count": len(row.get("rescued_ids", [])),
            "rescue_selector": "",
        })
    with (out / "02_microblock_count_audit.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "sample_id", "task", "method", "layer_id", "selected_microblock_count", "rescued_count", "rescue_selector",
        ])
        w.writeheader()
        for row in mb_rows:
            w.writerow(row)

    # Build selection sets per method/sample
    page_by_method: Dict[str, Dict[str, Set[int]]] = defaultdict(dict)
    mb_by_method: Dict[str, Dict[str, Set[str]]] = defaultdict(dict)

    for row in l26:
        method = str(row.get("method", row.get("rescue_score", "")))
        key = f"{row.get('task')}:{row.get('sample_id')}"
        pages = row.get("rescued_fixed_page_ids") or row.get("physical_units_after_expansion") or []
        page_by_method[method][key] = {int(x) for x in pages}
        mbs = row.get("selected_microblock_ids_before_expansion") or []
        mb_by_method[method][key] = {str(x) for x in mbs}

    for row in _read_jsonl_batch(root, "final_selection_dump.jsonl"):
        method = str(row.get("method", ""))
        key = f"{row.get('task')}:{row.get('sample_id')}"
        if key not in page_by_method[method]:
            page_by_method[method][key] = {int(x) for x in row.get("rescued_ids", [])}
        if key not in mb_by_method[method]:
            mbs = row.get("selected_microblock_ids_before_expansion") or []
            mb_by_method[method][key] = {str(x) for x in mbs}

    pairs = [
        ("fixed_drm_add", "rh_pyramid_srm"),
        ("fixed_drm_add", "bridge_graph_srm"),
        ("rh_pyramid_srm", "bridge_graph_srm"),
    ]

    # 03 page-level overlap
    with (out / "03_selection_overlap.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["granularity", "task_sample", "method_a", "method_b", "jaccard", "n_a", "n_b"])
        for gran, store in [("page", page_by_method), ("microblock", mb_by_method)]:
            for a, b in pairs:
                keys = set(store.get(a, {})) | set(store.get(b, {}))
                for key in sorted(keys):
                    sa = store.get(a, {}).get(key, set())
                    sb = store.get(b, {}).get(key, set())
                    w.writerow([gran, key, a, b, round(_jaccard(sa, sb), 4), len(sa), len(sb)])

    # 04 candidate diversity
    diversity_rows: List[Dict[str, Any]] = []
    for p in _find_files(root, "candidate_score_dump.jsonl"):
        by_key: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in _read_jsonl(p):
            key = f"{row.get('task')}:{row.get('sample_id')}:{row.get('method')}"
            by_key[key].append(row)
        for key, rows in by_key.items():
            ranked = sorted(rows, key=lambda r: r.get("rank", 9999))[:10]
            diversity_rows.append({
                "key": key,
                "candidate_count": len(rows),
                "unique_final_score_count": len({r.get("final_score") for r in rows}),
                "unique_selected_microblock_count": len({r.get("microblock_id") for r in rows if r.get("selected")}),
                "top10_ids": "|".join(str(r.get("candidate_id", "")) for r in ranked),
                "selected_microblock_ids": "|".join(str(r.get("microblock_id", "")) for r in rows if r.get("selected")),
            })
    with (out / "04_candidate_score_diversity.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "key", "candidate_count", "unique_final_score_count", "unique_selected_microblock_count",
            "top10_ids", "selected_microblock_ids",
        ])
        w.writeheader()
        for row in diversity_rows:
            w.writerow(row)

    # 05 expansion overlap audit
    expansion_rows: List[Dict[str, Any]] = []
    rh_pages = page_by_method.get("rh_pyramid_srm", {})
    br_pages = page_by_method.get("bridge_graph_srm", {})
    rh_mbs = mb_by_method.get("rh_pyramid_srm", {})
    br_mbs = mb_by_method.get("bridge_graph_srm", {})
    keys = sorted(set(rh_pages) | set(br_pages) | set(rh_mbs) | set(br_mbs))
    for key in keys:
        pj = _jaccard(rh_pages.get(key, set()), br_pages.get(key, set()))
        mj = _jaccard(rh_mbs.get(key, set()), br_mbs.get(key, set()))
        expansion_rows.append({
            "task_sample": key,
            "microblock_jaccard_rh_vs_bridge": round(mj, 4),
            "page_jaccard_rh_vs_bridge": round(pj, 4),
            "expansion_erases_difference": bool(mj < 1.0 and pj >= 1.0),
            "both_collapsed": bool(mj >= 1.0 and pj >= 1.0),
            "rh_microblock_count": len(rh_mbs.get(key, set())),
            "bridge_microblock_count": len(br_mbs.get(key, set())),
            "rh_page_count": len(rh_pages.get(key, set())),
            "bridge_page_count": len(br_pages.get(key, set())),
        })
    with (out / "05_expansion_overlap_audit.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(expansion_rows[0].keys()) if expansion_rows else [
            "task_sample", "microblock_jaccard_rh_vs_bridge", "page_jaccard_rh_vs_bridge",
            "expansion_erases_difference", "both_collapsed",
            "rh_microblock_count", "bridge_microblock_count", "rh_page_count", "bridge_page_count",
        ])
        w.writeheader()
        for row in expansion_rows:
            w.writerow(row)

    # 06 task failure breakdown
    failure_rows: List[Dict[str, Any]] = []
    for method_dir in sorted((root / "runs").glob("*")) if (root / "runs").is_dir() else []:
        method = method_dir.name
        scores_csv = list(method_dir.rglob("longbench_scores.csv"))
        if scores_csv:
            with scores_csv[0].open(encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    failure_rows.append({
                        "method": method,
                        "task": row.get("dataset", ""),
                        "status": row.get("status", ""),
                        "score": row.get("score", ""),
                        "error": row.get("error", ""),
                    })
        else:
            status = method_dir / "run_status.json"
            if status.exists():
                payload = json.loads(status.read_text(encoding="utf-8"))
                failure_rows.append({
                    "method": method,
                    "task": "_all",
                    "status": "ok" if payload.get("exit_code") == 0 else "fail",
                    "score": "",
                    "error": f"exit_code={payload.get('exit_code')}",
                })
    with (out / "06_task_failure_breakdown.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "task", "status", "score", "error"])
        w.writeheader()
        for row in failure_rows:
            w.writerow(row)

    # Dump existence per structured method actually run
    runs_dir = root / "runs"
    run_methods = sorted(p.name for p in runs_dir.iterdir() if p.is_dir()) if runs_dir.is_dir() else []
    structured_run = [m for m in run_methods if m in structured]
    dump_status: Dict[str, Dict[str, bool]] = {}
    for method in structured_run:
        run_dir = runs_dir / method
        dump_status[method] = {
            "candidate_score_dump": any(run_dir.rglob("candidate_score_dump.jsonl")),
            "final_selection_dump": any(run_dir.rglob("final_selection_dump.jsonl")),
            "path_marker": any(run_dir.rglob("path_marker.jsonl")),
        }

    mb_counts = [
        int(r.get("selected_microblock_count", 0) or 0)
        for r in mb_rows if str(r.get("method", "")) in structured
    ]
    path_ok = bool(markers) and all(
        row.get("used_microblock_first_selector") and row.get("entered_real_fixed_block_path")
        for row in markers if row.get("method") in structured
    )
    dumps_ok = all(all(v.values()) for v in dump_status.values() if v)

    def _pair_stats(store: Dict[str, Dict[str, Set[Any]]], a: str, b: str) -> Tuple[float, float, int]:
        js = []
        for key in set(store.get(a, {})) | set(store.get(b, {})):
            js.append(_jaccard(store.get(a, {}).get(key, set()), store.get(b, {}).get(key, set())))
        if not js:
            return 1.0, 1.0, 0
        return min(js), sum(js) / len(js), len(js)

    rh_br_mb_min, rh_br_mb_mean, _ = _pair_stats(mb_by_method, "rh_pyramid_srm", "bridge_graph_srm")
    rh_br_pg_min, rh_br_pg_mean, _ = _pair_stats(page_by_method, "rh_pyramid_srm", "bridge_graph_srm")
    fix_rh_pg_min, _, _ = _pair_stats(page_by_method, "fixed_drm_add", "rh_pyramid_srm")

    expansion_erased = sum(1 for r in expansion_rows if r.get("expansion_erases_difference"))
    both_collapsed = sum(1 for r in expansion_rows if r.get("both_collapsed"))

    justify_64 = (
        path_ok and dumps_ok and mb_counts and min(mb_counts) > 0
        and fix_rh_pg_min < 1.0
        and rh_br_mb_min < 1.0
        and rh_br_pg_min < 1.0
    )

    rec = [
        "# Recommendation (64-sample)",
        "",
        "## Real-path wiring",
        f"- path_marker hook active for structured methods: {path_ok}",
        f"- all structured dumps present: {dumps_ok}",
        f"- dump status: {json.dumps(dump_status, indent=2)}",
        "",
        "## selected_microblock_count (structured, layer-26/dump)",
        f"- count={len(mb_counts)}, min={min(mb_counts) if mb_counts else 0}, "
        f"max={max(mb_counts) if mb_counts else 0}, "
        f"mean={round(statistics.mean(mb_counts), 2) if mb_counts else 0}",
        "",
        "## Overlap interpretation",
        f"- fixed_drm_add vs rh_pyramid_srm page Jaccard min: {round(fix_rh_pg_min, 4)}",
        f"- rh vs bridge microblock Jaccard min/mean: {round(rh_br_mb_min, 4)} / {round(rh_br_mb_mean, 4)}",
        f"- rh vs bridge page Jaccard min/mean: {round(rh_br_pg_min, 4)} / {round(rh_br_pg_mean, 4)}",
        f"- samples where microblock differs but page ties (expansion erases): {expansion_erased}",
        f"- samples where both microblock and page tied: {both_collapsed}",
        "",
        "If microblock Jaccard < 1.0 but page Jaccard = 1.0, scorer differs but page expansion erases difference.",
        "If both = 1.0, scoring/weights still effectively collapsed at selection time.",
        "",
        ("**64-sample rerun is justified.** Real path is active and at least one overlap axis shows method diversity."
         if justify_64 else
         "**64-sample rerun is NOT justified.** Fix wiring or scoring diversity before scaling up."),
    ]
    (out / "07_recommendation.md").write_text("\n".join(rec) + "\n", encoding="utf-8")
    print(f"wrote audit to {out}")
    return 0


def _read_jsonl_batch(root: Path, name: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for p in _find_files(root, name):
        rows.extend(_read_jsonl(p))
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
