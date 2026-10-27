#!/usr/bin/env python3
"""Smoke: structured rescue selectors must differ on synthetic DRM/RH/bridge/teacher-favored units."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Set

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bridgekv.rescuekv_scoring import apply_rescue_policy, drm_s0_score
from bridgekv.structured_rescue_scoring import select_structured_rescue_via_microblocks


def _jaccard(a: Set[int], b: Set[int]) -> float:
    return len(a & b) / max(1, len(a | b))


def main() -> int:
    torch.manual_seed(0)
    errors: List[str] = []
    num_blocks = 16
    active_ids = list(range(8))
    inactive_ids = [8, 9, 10, 11]
    device = torch.device("cpu")

    # Block 8: DRM-favored; block 9: RH-favored; block 10: bridge-favored; block 11: teacher-favored
    base = torch.zeros(num_blocks, device=device)
    qk = torch.zeros(num_blocks, device=device)
    base[8], base[9], base[10], base[11] = 10.0, 1.0, 1.0, 1.0
    qk[8], qk[9], qk[10], qk[11] = 0.1, 10.0, 0.2, 0.1
    reps = torch.randn(num_blocks, 8, device=device)

    cfg_base = {
        "original_prompt_len": num_blocks * 64,
        "microblock_size": 32,
        "rescue_granularity": "microblock",
        "rescue_ratio": 0.25,
        "alpha_diff": 0.5,
        "lambda_red": 0.25,
        "query_text": "Microsoft Seattle bridge entity Alpha",
        "block_texts": [
            "" if i < 8 else (
                "drm evidence damage" if i == 8 else
                "retrieval head salient query" if i == 9 else
                "Alpha Seattle bridge connection" if i == 10 else
                "teacher span evidence dense"
            )
            for i in range(num_blocks)
        ],
        "teacher_label_file": "",
        "distilled_selector_file": "",
    }

    # fixed drm_add path (block-level MMR)
    drm_scores, _ = drm_s0_score(base, reps, active_ids, inactive_ids, diagnostics={"warnings": []})
    drm_res = apply_rescue_policy(
        active_ids=active_ids, inactive_ids=inactive_ids, rescue_scores=drm_scores,
        active_scores=base, block_reps=reps, policy="add", rescue_ratio=0.25,
        mmr_selection=True, diagnostics={"warnings": []},
    )
    drm_picks = set(int(x) for x in drm_res.rescued_ids.tolist())

    teacher_path = Path(tempfile.mkdtemp()) / "teacher.jsonl"
    teacher_path.write_text(
        json.dumps({"page_id": 11, "teacher_overlap_ratio": 5.0, "teacher_positive": 1, "weak_teacher": False}) + "\n",
        encoding="utf-8",
    )

    selections: Dict[str, Set[int]] = {"fixed_drm_add": drm_picks}
    for method, extra in [
        ("rh_pyramid_srm", {"rh_weight": 3.0, "pyramid_weight": 0.1}),
        ("distilled_blockdrm", {"teacher_align_weight": 5.0, "teacher_label_file": str(teacher_path)}),
        ("bridge_graph_srm", {"bridge_weight": 5.0, "rh_weight": 0.05, "bridge_min_link": 0.01}),
        ("deferred_uncertainty_srm", {"deferred_mode": "audit"}),
        ("evidence_density_srm", {}),
    ]:
        cfg = {**cfg_base, "rescue_score": method, **extra}
        rescued, _, _ = select_structured_rescue_via_microblocks(
            cfg, base, qk, reps, active_ids, inactive_ids, target_m=1, diagnostics={"warnings": []},
        )
        selections[method] = set(rescued)

    # add-only checks on rh_pyramid
    rescued_list, meta, _ = select_structured_rescue_via_microblocks(
        {**cfg_base, "rescue_score": "rh_pyramid_srm", "rh_weight": 0.35},
        base, qk, reps, active_ids, inactive_ids, target_m=2, diagnostics={"warnings": []},
    )
    final_ids = sorted(set(active_ids) | set(rescued_list))
    if set(active_ids) - set(final_ids):
        errors.append("add_only_contract violated")
    if len(rescued_list) != len(set(rescued_list)):
        errors.append("duplicate rescued units")

    new_methods = [
        "rh_pyramid_srm",
        "distilled_blockdrm",
        "bridge_graph_srm",
        "deferred_uncertainty_srm",
        "evidence_density_srm",
    ]
    unique_sets = {tuple(sorted(v)) for v in selections.values()}
    if len(unique_sets) < 3:
        errors.append(f"insufficient selector diversity (unique={len(unique_sets)}): {selections}")
    if selections.get("fixed_drm_add") == selections.get("rh_pyramid_srm"):
        errors.append("rh_pyramid_srm did not differ from fixed_drm_add")
    if selections.get("bridge_graph_srm") == selections.get("distilled_blockdrm") == selections.get("rh_pyramid_srm"):
        errors.append("bridge/distilled/rh collapsed to identical picks")

    print("selections:", {k: sorted(v) for k, v in selections.items()})
    if errors:
        print("FAIL")
        for e in errors:
            print(" -", e)
        return 1
    print("PASS: selector diversity smoke ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
