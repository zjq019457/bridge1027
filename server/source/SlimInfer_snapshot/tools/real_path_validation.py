#!/usr/bin/env python3
"""Post-run validation for real-path microblock hook (no model)."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _jaccard(a: Set[int], b: Set[int]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def _find_files(root: Path, name: str) -> List[Path]:
    return sorted(root.rglob(name))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    root = Path(args.input_root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    markers = []
    for p in _find_files(root, "path_marker.jsonl"):
        for row in _read_jsonl(p):
            row["source_file"] = str(p)
            markers.append(row)
    with (out / "01_path_marker_audit.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "sample_id", "task", "method", "entered_structured_selector", "entered_real_fixed_block_path",
            "entered_cache_construction_path", "used_microblock_first_selector",
            "used_candidate_score_dump", "used_final_selection_dump", "source_file",
        ], extrasaction="ignore")
        w.writeheader()
        for row in markers:
            w.writerow(row)

    micro_rows: List[Dict[str, Any]] = []
    for p in _find_files(root, "final_selection_dump.jsonl"):
        micro_rows.extend(_read_jsonl(p))
    for p in _find_files(root, "*.jsonl"):
        if "audit" in p.name or "pruning" in p.name:
            for row in _read_jsonl(p):
                if "selected_microblock_count" in row:
                    micro_rows.append(row)
    with (out / "02_microblock_count_audit.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sample_id", "task", "method", "selected_microblock_count"], extrasaction="ignore")
        w.writeheader()
        seen = set()
        for row in micro_rows:
            key = (row.get("sample_id"), row.get("task"), row.get("method"))
            if key in seen:
                continue
            seen.add(key)
            w.writerow({
                "sample_id": row.get("sample_id", ""),
                "task": row.get("task", ""),
                "method": row.get("method", row.get("rescue_score", "")),
                "selected_microblock_count": row.get("selected_microblock_count", 0),
            })

    rescued_by_method: Dict[str, Dict[str, Set[int]]] = {}
    for p in _find_files(root, "final_selection_dump.jsonl"):
        for row in _read_jsonl(p):
            method = str(row.get("method", ""))
            sid = str(row.get("sample_id", ""))
            task = str(row.get("task", ""))
            key = f"{task}:{sid}"
            rescued_by_method.setdefault(method, {})[key] = {int(x) for x in row.get("rescued_ids", [])}
    for p in sorted((root / "audit").glob("*.jsonl")) if (root / "audit").exists() else []:
        method = p.stem
        for row in _read_jsonl(p):
            if int(row.get("layer_id", -1)) != 26:
                continue
            sid = str(row.get("sample_id", ""))
            task = str(row.get("task", ""))
            key = f"{task}:{sid}"
            rescued = row.get("rescued_fixed_page_ids") or row.get("rescued_virtual_ids_first50") or []
            rescued_by_method.setdefault(method, {})[key] = {int(x) for x in rescued}

    pairs = [
        ("fixed_drm_add", "rh_pyramid_srm"),
        ("fixed_drm_add", "bridge_graph_srm"),
        ("rh_pyramid_srm", "bridge_graph_srm"),
    ]
    with (out / "03_selection_overlap.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task_sample", "method_a", "method_b", "jaccard"])
        for a, b in pairs:
            keys = set(rescued_by_method.get(a, {})) | set(rescued_by_method.get(b, {}))
            for key in sorted(keys):
                ja = rescued_by_method.get(a, {}).get(key, set())
                jb = rescued_by_method.get(b, {}).get(key, set())
                w.writerow([key, a, b, round(_jaccard(ja, jb), 4)])

    diversity_rows: List[Dict[str, Any]] = []
    for p in _find_files(root, "candidate_score_dump.jsonl"):
        by_key: Dict[str, List[Dict[str, Any]]] = {}
        for row in _read_jsonl(p):
            key = f"{row.get('task')}:{row.get('sample_id')}:{row.get('method')}"
            by_key.setdefault(key, []).append(row)
        for key, rows in by_key.items():
            scores = [r.get("final_score") for r in rows]
            selected = [r.get("candidate_id") for r in rows if r.get("selected")]
            ranked = sorted(rows, key=lambda r: r.get("rank", 9999))[:10]
            diversity_rows.append({
                "key": key,
                "candidate_count": len(rows),
                "unique_final_score_count": len({s for s in scores if s is not None}),
                "top10_ids": "|".join(str(r.get("candidate_id", "")) for r in ranked),
                "selected_ids": "|".join(str(x) for x in selected),
            })
    with (out / "04_candidate_score_diversity.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["key", "candidate_count", "unique_final_score_count", "top10_ids", "selected_ids"])
        w.writeheader()
        for row in diversity_rows:
            w.writerow(row)

    has_cand = bool(_find_files(root, "candidate_score_dump.jsonl"))
    has_final = bool(_find_files(root, "final_selection_dump.jsonl"))
    has_marker = bool(_find_files(root, "path_marker.jsonl"))
    structured_methods = {
        "rh_pyramid_srm",
        "bridge_graph_srm",
        "distilled_blockdrm",
        "deferred_uncertainty_srm",
        "evidence_density_srm",
    }
    path_ok = all(
        row.get("used_microblock_first_selector") and row.get("entered_real_fixed_block_path")
        for row in markers if row.get("method") in structured_methods
    ) if markers else False
    mb_ok = any(
        int(row.get("selected_microblock_count", 0) or 0) > 0
        for row in micro_rows if str(row.get("method", row.get("rescue_score", ""))) in structured_methods
    )
    jaccards_vs_fixed: List[float] = []
    jaccards_structured: List[float] = []
    with (out / "03_selection_overlap.csv").open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ja = float(row["jaccard"])
            if row["method_a"] == "fixed_drm_add" and row["method_b"] in structured_methods:
                jaccards_vs_fixed.append(ja)
            if {row["method_a"], row["method_b"]} == {"rh_pyramid_srm", "bridge_graph_srm"}:
                jaccards_structured.append(ja)
    diverse_vs_fixed = any(j < 1.0 for j in jaccards_vs_fixed)
    rh_bridge_diverse = any(j < 1.0 for j in jaccards_structured)
    justify = (
        has_cand and has_final and has_marker and path_ok and mb_ok
        and diverse_vs_fixed and rh_bridge_diverse
    )
    rec = [
        "# Recommendation",
        "",
        f"- candidate_score_dump exists: {has_cand}",
        f"- final_selection_dump exists: {has_final}",
        f"- path_marker.jsonl exists: {has_marker}",
        f"- path_marker confirms real path hook (entered_real_fixed_block_path + used_microblock_first_selector): {path_ok}",
        f"- selected_microblock_count > 0 for structured methods: {mb_ok}",
        f"- structured vs fixed_drm_add rescued-id Jaccard < 1.0 on at least one pair: {diverse_vs_fixed}",
        f"- rh_pyramid_srm vs bridge_graph_srm not 100% identical (Jaccard < 1.0): {rh_bridge_diverse}",
        "",
        ("**32-sample rerun is justified.** Real fixed-block microblock hook is active on the generation path." if justify
         else "**32-sample rerun is NOT yet justified.** Real path hook is not fully confirmed."),
    ]
    (out / "05_recommendation.md").write_text("\n".join(rec) + "\n", encoding="utf-8")
    print(f"wrote validation to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
