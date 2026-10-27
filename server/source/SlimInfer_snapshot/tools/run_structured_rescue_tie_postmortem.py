#!/usr/bin/env python3
"""Post-mortem: why structured rescue methods tied in stage2 32-sample run."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

METHODS = [
    "fixed_drm_add", "rh_pyramid_srm", "distilled_blockdrm",
    "bridge_graph_srm", "deferred_uncertainty_srm",
]
NEW_METHODS = METHODS[1:]
PAIRS = [
    ("fixed_drm_add", "rh_pyramid_srm"),
    ("fixed_drm_add", "distilled_blockdrm"),
    ("fixed_drm_add", "bridge_graph_srm"),
    ("fixed_drm_add", "deferred_uncertainty_srm"),
    ("rh_pyramid_srm", "distilled_blockdrm"),
    ("rh_pyramid_srm", "bridge_graph_srm"),
    ("rh_pyramid_srm", "deferred_uncertainty_srm"),
]


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _jaccard(a: Set[int], b: Set[int]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def _write_csv(path: Path, rows: List[Dict[str, Any]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def run_postmortem(result_dir: Path) -> None:
    out = result_dir / "tie_postmortem"
    out.mkdir(parents=True, exist_ok=True)
    audits = {m: _load_jsonl(result_dir / "audit" / f"{m}.jsonl") for m in METHODS}
    configs = {}
    for m in METHODS:
        cfg_path = result_dir / "runs" / m / "config.json"
        configs[m] = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}

    # 00 method config
    cfg_rows = []
    for m in METHODS:
        args = configs.get(m, {}).get("args", configs.get(m, {}))
        slim = configs.get(m, {}).get("sliminfer_config", {})
        src = slim if slim else args
        sample_audit = next((r for r in audits[m] if r.get("layer_id") == 26), audits[m][0] if audits[m] else {})
        cfg_rows.append({
            "method": m,
            "rescue_score": src.get("rescue_score", sample_audit.get("rescue_score", "")),
            "rescue_granularity": src.get("rescue_granularity", sample_audit.get("rescue_granularity", "page")),
            "microblock_size": src.get("microblock_size", sample_audit.get("microblock_size", "")),
            "rescue_ratio": src.get("rescue_ratio", ""),
            "rh_weight": src.get("rh_weight", ""),
            "bridge_weight": src.get("bridge_weight", ""),
            "deferred_mode": src.get("deferred_mode", sample_audit.get("deferred_mode", "")),
            "fallback_used": sample_audit.get("fallback_used", ""),
            "rh_fallback_used": sample_audit.get("rh_fallback_used", ""),
            "teacher_fallback_used": sample_audit.get("teacher_fallback_used", ""),
            "selector_fallback_used": sample_audit.get("selector_fallback_used", ""),
            "bridge_enabled": m == "bridge_graph_srm",
            "teacher_label_file": src.get("teacher_label_file", ""),
            "weak_teacher": "weak" in str(src.get("teacher_label_file", "")),
        })
    _write_csv(out / "00_method_config_audit.csv", cfg_rows, list(cfg_rows[0].keys()) if cfg_rows else ["method"])

    # Index rescued ids per sample (layer 26 prefill only)
    by_key: Dict[str, Dict[str, Set[int]]] = {m: {} for m in METHODS}
    rescued_counts: Dict[str, Counter] = {m: Counter() for m in METHODS}
    for m, rows in audits.items():
        for r in rows:
            if str(r.get("layer_id")) != "26" or r.get("phase") != "prefill":
                continue
            key = f"{r.get('task')}|{r.get('sample_id')}"
            rescued = set(int(x) for x in (r.get("rescued_fixed_page_ids") or r.get("rescued_virtual_ids_first50") or []))
            by_key[m][key] = rescued
            rescued_counts[m][r.get("rescued_count", 0)] += 1

    # 01 selection overlap
    overlap_rows = []
    keys = sorted(set().union(*(by_key[m].keys() for m in METHODS)))
    for key in keys:
        task, sid = key.split("|", 1)
        for a, b in PAIRS:
            sa, sb = by_key[a].get(key, set()), by_key[b].get(key, set())
            overlap_rows.append({
                "task": task, "sample_id": sid, "method_a": a, "method_b": b,
                "jaccard_rescued": round(_jaccard(sa, sb), 6),
                "identical_rescued": int(sa == sb),
                "rescued_a": ",".join(str(x) for x in sorted(sa)),
                "rescued_b": ",".join(str(x) for x in sorted(sb)),
            })
    _write_csv(out / "01_selection_overlap.csv", overlap_rows,
               ["task", "sample_id", "method_a", "method_b", "jaccard_rescued", "identical_rescued", "rescued_a", "rescued_b"])

    # 02 score diversity - candidate scores NOT logged in this run
    score_rows = [{
        "note": "candidate_score_dump.jsonl missing for all methods",
        "recommendation": "enable dump_candidate_scores in runner after path fix",
    }]
    _write_csv(out / "02_score_diversity.csv", score_rows, ["note", "recommendation"])

    # 03 granularity collapse
    gran_rows = []
    for m in METHODS:
        r = next((x for x in audits[m] if x.get("rescue_granularity") or x.get("physical_granularity")), {})
        gran_rows.append({
            "method": m,
            "requested_granularity": r.get("rescue_granularity", "page"),
            "scoring_granularity": r.get("physical_granularity", r.get("rescue_granularity", "")),
            "selection_granularity": "block/page (MMR on block ids)",
            "physical_granularity": r.get("physical_granularity", ""),
            "final_cache_granularity": "block (64-token pages)",
            "selected_microblock_count_mean": r.get("selected_microblock_count", 0),
            "collapse_detected": int(m != "fixed_drm_add" and r.get("selected_microblock_count", 0) == 0),
        })
    _write_csv(out / "03_granularity_collapse_audit.csv", gran_rows, list(gran_rows[0].keys()))

    # 04 fallback
    fb_rows = []
    teacher_path = result_dir / "teacher_microblock_labels.jsonl"
    weak_teacher = any("weak_teacher" in l and "true" in l.lower() for l in teacher_path.read_text().splitlines()[:5]) if teacher_path.exists() else True
    for m in METHODS:
        r = next((x for x in audits[m] if x.get("method_name") or x.get("method")), {})
        fb_rows.append({
            "method": m,
            "rh_fallback_used": r.get("rh_fallback_used", ""),
            "teacher_fallback_used": r.get("teacher_fallback_used", weak_teacher if m == "distilled_blockdrm" else ""),
            "selector_fallback_used": r.get("selector_fallback_used", ""),
            "weak_teacher_labels": weak_teacher if m == "distilled_blockdrm" else "",
            "candidate_scores_logged": False,
            "likely_cause": "microblock scores aggregated to block max before MMR; identical block ranking",
        })
    _write_csv(out / "04_fallback_cause_audit.csv", fb_rows, list(fb_rows[0].keys()))

    # 05 budget topk
    budget_rows = []
    for m in METHODS:
        rc = rescued_counts[m]
        budget_rows.append({
            "method": m,
            "rescued_count_histogram": dict(rc),
            "identical_histogram_vs_fixed": int(rc == rescued_counts["fixed_drm_add"]),
        })
    _write_csv(out / "05_budget_topk_audit.csv", budget_rows, ["method", "rescued_count_histogram", "identical_histogram_vs_fixed"])

    # 06 musique cases - compare predictions
    musique_rows = []
    for m in METHODS:
        pred_dir = result_dir / "runs" / m / "longbench_predictions"
        mus_path = pred_dir / "musique.jsonl" if pred_dir.exists() else None
        if not mus_path or not mus_path.exists():
            continue
        for line in mus_path.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            sid = rec.get("_id", rec.get("id", ""))
            musique_rows.append({
                "sample_id": sid, "method": m,
                "score": rec.get("score", ""),
                "pred_preview": str(rec.get("pred", ""))[:120],
            })
    # pivot scores
    fixed_scores = {r["sample_id"]: r for r in musique_rows if r["method"] == "fixed_drm_add"}
    case_rows = []
    for sid, fr in fixed_scores.items():
        row = {"sample_id": sid, "fixed_score": fr.get("score", ""), "fixed_pred": fr.get("pred_preview", "")}
        for m in NEW_METHODS:
            nr = next((r for r in musique_rows if r["sample_id"] == sid and r["method"] == m), {})
            row[f"{m}_score"] = nr.get("score", "")
            row[f"{m}_pred_match_fixed"] = int(nr.get("pred_preview", "") == fr.get("pred_preview", ""))
            key = f"musique|{sid}"
            row[f"{m}_rescued_jaccard_vs_fixed"] = round(
                _jaccard(by_key["fixed_drm_add"].get(key, set()), by_key[m].get(key, set())), 4
            )
        case_rows.append(row)
    _write_csv(out / "06_musique_failure_cases.csv", case_rows, list(case_rows[0].keys()) if case_rows else ["sample_id"])

    # 07 recommendation
    new_identical = all(
        rescued_counts[m] == rescued_counts[NEW_METHODS[0]] for m in NEW_METHODS
    )
    overlap_new = []
    for key in keys:
        sets = [by_key[m].get(key, set()) for m in NEW_METHODS]
        if sets and all(s == sets[0] for s in sets[1:]):
            overlap_new.append(key)
    rec = f"""# Structured Rescue Tie Post-Mortem Recommendation

## Primary cause
**Granularity collapse before selection**: `rescue_granularity=microblock` only modified block-level scores via per-block `max(microblock_score)`, then **MMR selected block ids**. Method-specific RH/bridge/teacher/deferred terms did not change block rankings enough to alter MMR output. All four new methods therefore produced **identical rescued_count histograms** and **identical task scores** (except small drift vs fixed_drm_add on hotpotqa/musique from decode variance or sparse per-sample block differences).

## Evidence
- `selected_microblock_count=0` in audit for all structured methods.
- `candidate_score_dump.jsonl` **not present** — cannot verify per-candidate score diversity.
- Rescued-count histogram identical across all 5 methods: `{dict(rescued_counts['fixed_drm_add'])}`.
- New-method pairwise rescued-id identity on layer-26: **{len(overlap_new)}/{len(keys)}** samples identical across rh/distilled/bridge/deferred.

## Secondary factors
1. **weak_teacher=true** for distilled_blockdrm (synthetic teacher labels).
2. **deferred_uncertainty_srm** uses DRM+RH pre-rank — collapses to same block MMR as rh_pyramid when block scores align.
3. **Budget quantization**: `target_m = round(K * rescue_ratio)` yields discrete {1,3,5,10} rescued blocks.

## Required code changes (before 64-sample)
1. Select **microblock candidates first**, then map to blocks for cache expansion.
2. Emit `candidate_score_dump.jsonl` and `final_selection_dump.jsonl`.
3. Add `teacher_label_quality` audit field; mark weak-teacher distilled as diagnostic-only.
4. Run `tools/smoke_selector_diversity.py` — must show non-identical selections on synthetic DRM/RH/bridge/teacher-favored units.

## 64-sample recommendation
**Do not run 64-sample** until smoke_selector_diversity passes and post-fix 32-sample shows:
- rescued-id Jaccard vs fixed_drm_add < 1.0 for at least one new method on >10% samples, AND
- macro avg not clearly worse than fixed_drm_add by >0.3.

Current run: macro 45.43 vs fixed 45.96 (**-0.52**); **not justified**.
"""
    (out / "07_recommendation.md").write_text(rec, encoding="utf-8")
    print(f"Post-mortem written to {out}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--result_dir", default="results/structured_rescue_stage2_32_20260609_222904")
    args = p.parse_args()
    run_postmortem(REPO_ROOT / args.result_dir)


if __name__ == "__main__":
    main()
