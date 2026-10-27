#!/usr/bin/env python3
"""Smoke test: sustained evidence must beat one isolated maximum."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bridgekv.structured_rescue_scoring import rank_evidence_density_candidates


def _candidate(page_id: int, local_rank: int, score: float) -> Dict[str, Any]:
    start = page_id * 64 + local_rank * 16
    return {
        "candidate_id": f"p{page_id}_mb{local_rank}",
        "page_id": page_id,
        "block_id": page_id,
        "microblock_id": f"p{page_id}_mb{local_rank}",
        "start_token": start,
        "end_token": start + 16,
        "microblock_local_rank": local_rank,
        "final_score": score,
    }


def main() -> int:
    # Page A: one isolated spike. Page B: three adjacent moderate signals.
    candidates: List[Dict[str, Any]] = []
    for rank, score in enumerate([10.0, 0.0, 0.0, 0.0]):
        candidates.append(_candidate(4, rank, score))
    for rank, score in enumerate([6.0, 6.0, 6.0, 0.0]):
        candidates.append(_candidate(5, rank, score))

    ranked, meta = rank_evidence_density_candidates(
        candidates,
        {
            "density_alpha_max": 1.0,
            "density_alpha_count": 0.6,
            "density_alpha_continuity": 0.8,
            "density_alpha_coverage": 0.5,
            "density_lambda_red": 0.3,
            "density_threshold_mode": "mean_std",
            "density_top_quantile": 0.30,
        },
        redundancy_by_page={4: 0.0, 5: 0.0},
    )
    errors: List[str] = []
    if not ranked or ranked[0][0] != 5:
        errors.append(f"dense page B did not outrank isolated page A: {ranked}")
    max_only_winner = max(
        {4: 10.0, 5: 6.0},
        key={4: 10.0, 5: 6.0}.get,
    )
    if max_only_winner == ranked[0][0]:
        errors.append("density components did not change ranking versus max-only")
    details = {page_id: info for page_id, _, _, info in ranked}
    if details.get(5, {}).get("density_high_count", 0) <= details.get(4, {}).get("density_high_count", 0):
        errors.append("dense page high-count feature did not exceed isolated page")
    if details.get(5, {}).get("density_continuity_bonus", 0.0) <= details.get(4, {}).get("density_continuity_bonus", 0.0):
        errors.append("dense page continuity feature did not exceed isolated page")
    required = {
        "density_parent_score",
        "density_max_score",
        "density_high_count",
        "density_continuity_bonus",
        "density_span_coverage",
        "density_threshold",
        "density_threshold_mode",
        "redundancy_fallback_used",
    }
    if required - candidates[0].keys():
        errors.append(f"candidate density fields missing: {sorted(required - candidates[0].keys())}")

    print(
        "ranking:",
        [
            {
                "page": page_id,
                "density_score": round(score, 4),
                "high_count": info["density_high_count"],
                "continuity": info["density_continuity_bonus"],
            }
            for page_id, score, _, info in ranked
        ],
    )
    print("threshold:", meta["density_threshold"], meta["density_threshold_mode"])
    if errors:
        print("FAIL")
        for error in errors:
            print(" -", error)
        return 1
    print("PASS: evidence-density selector favors sustained adjacent evidence")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
