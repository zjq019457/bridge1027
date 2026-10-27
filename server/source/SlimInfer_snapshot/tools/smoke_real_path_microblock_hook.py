#!/usr/bin/env python3
"""Dry-run: structured microblock hook on fixed virtual-block rescue path (no model)."""
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

from utils.cache_utils import _structured_microblock_rescue_in_fixed_virtual_path


def _jaccard(a: Set[int], b: Set[int]) -> float:
    return len(a & b) / max(1, len(a | b))


def main() -> int:
    torch.manual_seed(0)
    errors: List[str] = []
    num_blocks = 16
    original_block_ids = list(range(num_blocks))
    selected = set(range(8))
    inactive = list(range(8, num_blocks))
    device = torch.device("cpu")
    virtual_delta = torch.zeros(num_blocks, device=device)
    virtual_delta[8:12] = torch.tensor([10.0, 1.0, 1.0, 1.0], device=device)
    virtual_reps = torch.randn(num_blocks, 8, device=device)
    qk_block_scores = torch.zeros(num_blocks, device=device)
    qk_block_scores[9] = 10.0

    out_dir = Path(tempfile.mkdtemp(prefix="real_path_microblock_hook_"))
    base_cfg: Dict[str, object] = {
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
        "dump_candidate_scores": 1,
        "dump_final_selection": 1,
        "strict_structured_path_check": 1,
        "candidate_score_dump_path": str(out_dir / "candidate_score_dump.jsonl"),
        "final_selection_dump_path": str(out_dir / "final_selection_dump.jsonl"),
        "path_marker_dump_path": str(out_dir / "path_marker.jsonl"),
        "audit_context": {"sample_id": "dry0", "task": "hook"},
    }

    picks: Dict[str, Set[int]] = {}
    for method, extra in [
        ("rh_pyramid_srm", {"rh_weight": 3.0}),
        ("bridge_graph_srm", {"bridge_weight": 5.0, "bridge_min_link": 0.01}),
        ("evidence_density_srm", {
            "density_alpha_max": 1.0,
            "density_alpha_count": 0.6,
            "density_alpha_continuity": 0.8,
            "density_alpha_coverage": 0.5,
            "density_lambda_red": 0.3,
        }),
        ("fixed_drm_proxy", {}),
    ]:
        if method == "fixed_drm_proxy":
            continue
        cfg = {**base_cfg, "rescue_score": method, **extra}
        rescued_virtual, meta = _structured_microblock_rescue_in_fixed_virtual_path(
            cfg=cfg,
            selected=selected,
            inactive=inactive,
            original_block_ids=original_block_ids,
            virtual_delta=virtual_delta,
            virtual_reps=virtual_reps,
            qk_block_scores=qk_block_scores,
            target_m=1,
            device=device,
        )
        rescued_pages = {original_block_ids[v] for v in rescued_virtual}
        picks[method] = rescued_pages
        if int(meta.get("selected_microblock_count", 0) or 0) <= 0:
            errors.append(f"{method}: selected_microblock_count=0")
        marker_path = Path(str(cfg["path_marker_dump_path"]))
        if not marker_path.exists():
            errors.append(f"{method}: path_marker missing")
        else:
            row = json.loads(marker_path.read_text(encoding="utf-8").strip().splitlines()[-1])
            if not row.get("used_microblock_first_selector"):
                errors.append(f"{method}: used_microblock_first_selector=false")
            if not row.get("entered_real_fixed_block_path"):
                errors.append(f"{method}: entered_real_fixed_block_path=false")
            if method == "evidence_density_srm" and not row.get("used_evidence_density_selector"):
                errors.append("evidence_density_srm: used_evidence_density_selector=false")

        if method == "evidence_density_srm":
            final_rows = [
                json.loads(line)
                for line in Path(str(cfg["final_selection_dump_path"])).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            density_row = final_rows[-1] if final_rows else {}
            if density_row.get("method") != "evidence_density_srm":
                errors.append("evidence_density_srm: final selection method mismatch")
            if int(density_row.get("selected_parent_page_count", 0) or 0) <= 0:
                errors.append("evidence_density_srm: selected_parent_page_count=0")

    cand_path = out_dir / "candidate_score_dump.jsonl"
    final_path = out_dir / "final_selection_dump.jsonl"
    marker_path = out_dir / "path_marker.jsonl"
    if not cand_path.exists() or cand_path.stat().st_size == 0:
        errors.append("candidate_score_dump.jsonl missing or empty")
    if not final_path.exists() or final_path.stat().st_size == 0:
        errors.append("final_selection_dump.jsonl missing or empty")
    if not marker_path.exists() or marker_path.stat().st_size == 0:
        errors.append("path_marker.jsonl missing or empty")

    if _jaccard(picks.get("rh_pyramid_srm", set()), picks.get("bridge_graph_srm", set())) >= 1.0:
        errors.append("structured methods produced identical rescued pages on hook path")

    print(f"hook_out_dir={out_dir}")
    if errors:
        print("FAIL")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("PASS: real fixed-block microblock hook ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
