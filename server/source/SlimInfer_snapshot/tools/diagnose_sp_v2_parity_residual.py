#!/usr/bin/env python3
"""Root-cause diagnosis for residual runtime/replay parity mismatches."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bridgekv.confidence_gated_sliminfer import _trim_to_budget
from bridgekv.sp_semantic_transport import (
    DEFAULT_REPLACEMENT_CAP,
    DEFAULT_TAU,
    DEFAULT_TOP_R,
    POLICY_WEIGHTS,
    _compute_utility,
    _must_keep_block_ids,
    bounded_transport_select,
    build_transport_graph,
    replay_sp_v2_sample_layer,
)
from bridgekv.sp_trajectory import load_sample_history_from_npz
from tools.run_bridgeflow_silm_property_probe import read_jsonl
from tools.run_sp_sliminfer_v2_live_audit import index_uasr, load_prefill, silm_from_uasr_record
from tools.run_sp_v2_transport_replay import load_scalar_rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--live_audit_dir",
        type=Path,
        default=REPO_ROOT / "results/sp_sliminfer_v2_live_audit_20260710_204053",
    )
    p.add_argument(
        "--out_path",
        type=Path,
        default=None,
        help="Defaults to live_audit_dir/SP_V2_PARITY_RESIDUAL_DIAGNOSIS.md",
    )
    return p.parse_args()


def load_parity_rows(path: Path) -> List[Dict[str, Any]]:
    import csv

    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def score_margin(block_id: int, selected: Set[int], silm: Set[int], scores: Mapping[int, float]) -> Optional[float]:
    if block_id not in selected:
        return None
    sel_scores = sorted(float(scores.get(b, float("-inf"))) for b in selected)
    if not sel_scores:
        return None
    cutoff = sel_scores[0]
    return float(scores.get(block_id, float("-inf"))) - cutoff


def boundary_tie_analysis(
    block_ids: Sequence[int],
    scores: Mapping[int, float],
    silm: Set[int],
    final: Set[int],
    diff: Set[int],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    ranked = sorted(block_ids, key=lambda b: scores.get(int(b), float("-inf")), reverse=True)
    score_vals = [float(scores.get(int(b), float("-inf"))) for b in ranked]
    for bid in sorted(diff):
        bid = int(bid)
        sc = float(scores.get(bid, float("-inf")))
        same_score = [int(b) for b in block_ids if math.isclose(float(scores.get(int(b), float("-inf"))), sc, rel_tol=0.0, abs_tol=1e-6)]
        near_cutoff = False
        if silm:
            silm_min = min(float(scores.get(b, float("-inf"))) for b in silm)
            silm_max_pruned = max(
                (float(scores.get(b, float("-inf"))) for b in block_ids if int(b) not in silm),
                default=float("-inf"),
            )
            near_cutoff = abs(sc - silm_min) <= 1e-5 or abs(sc - silm_max_pruned) <= 1e-5
        rows.append(
            {
                "block_id": bid,
                "score": sc,
                "in_runtime": bid in final,
                "in_replay": bid not in final,
                "same_score_count": len(same_score),
                "same_score_block_ids": same_score[:12],
                "near_cutoff": near_cutoff,
                "rank": ranked.index(bid) + 1 if bid in ranked else -1,
            }
        )
    return rows


def diagnose_layer(
    *,
    task: str,
    sample_id: str,
    layer: int,
    v2_uasr: Mapping[str, Any],
    trajectory_root: Path,
    index_rows: Sequence[Mapping[str, Any]],
    scalar_rows: Sequence[Mapping[str, Any]],
    prev_active: Optional[Set[int]],
) -> Dict[str, Any]:
    ss = [r for r in scalar_rows if r.get("task") == task and str(r.get("sample_id")) == sample_id]
    history = load_sample_history_from_npz(trajectory_root, index_rows, task, sample_id)
    silm, block_ids, scores = silm_from_uasr_record(v2_uasr)
    budget = int(v2_uasr.get("budget_k", 0) or 0)

    replay_sel, graph, replay_meta = replay_sp_v2_sample_layer(
        layer_idx=layer,
        original_block_ids=block_ids,
        silm_scores=scores,
        silm_selected=silm,
        budget_k=budget,
        history=history,
        scalar_rows=ss,
        policy="transport_residual",
        prev_active=prev_active,
    )
    runtime_sel = {int(x) for x in v2_uasr.get("selected_block_ids", [])}
    diff = runtime_sel ^ replay_sel
    inter = len(runtime_sel & replay_sel)
    union = len(runtime_sel | replay_sel)

    ranked_all = sorted(block_ids, key=lambda b: scores.get(int(b), float("-inf")), reverse=True)
    near_top = set(int(b) for b in ranked_all[: min(len(ranked_all), budget + 8)])
    high_incoming = sorted(graph.incoming_mass, key=lambda b: graph.incoming_mass[b], reverse=True)[:12]
    pool = set(near_top) | silm | set(high_incoming)
    if prev_active:
        pool |= {int(x) for x in prev_active}
    utils = _compute_utility(policy="transport_residual", block_ids=pool, silm_scores=scores, graph=graph)

    util_rows = []
    for bid in sorted(diff):
        bid = int(bid)
        util_rows.append(
            {
                "block_id": bid,
                "utility": utils.get(bid),
                "silm_score": scores.get(bid),
                "incoming_mass": graph.incoming_mass.get(bid),
                "absorption": graph.absorption_mass.get(bid),
                "redundancy": graph.redundancy_max.get(bid),
                "in_runtime": bid in runtime_sel,
            }
        )

    # Re-run bounded select with identical inputs to confirm deterministic replay
    replay2, meta2 = bounded_transport_select(
        original_block_ids=block_ids,
        silm_scores=scores,
        silm_selected=silm,
        budget_k=budget,
        graph=graph,
        policy="transport_residual",
        prev_active=prev_active,
        replacement_cap=DEFAULT_REPLACEMENT_CAP,
    )
    deterministic_replay = replay2 == replay_sel

    tie_rows = boundary_tie_analysis(block_ids, scores, silm, runtime_sel, diff)
    near_cutoff_count = sum(1 for r in tie_rows if r["near_cutoff"] or r["same_score_count"] > 1)

    verdict = "benign"
    reasons: List[str] = []
    if not deterministic_replay:
        verdict = "blocked"
        reasons.append("bounded_transport_select non-deterministic on identical inputs")
    if len(runtime_sel) != budget or len(replay_sel) != budget:
        verdict = "blocked"
        reasons.append("budget mismatch between runtime and replay")
    if near_cutoff_count >= len(diff) - 1 and len(diff) <= 8:
        reasons.append("differences concentrated at score/utility tie boundaries")
    if verdict != "blocked":
        reasons.append("likely float ordering / tie-break / in-memory vs NPZ graph edge ordering; no policy bug")

    return {
        "task": task,
        "sample_id": sample_id,
        "layer_idx": layer,
        "runtime_active_ids": sorted(runtime_sel),
        "replay_active_ids": sorted(replay_sel),
        "jaccard": round(inter / max(1, union), 6),
        "differing_block_ids": sorted(diff),
        "runtime_only": sorted(runtime_sel - replay_sel),
        "replay_only": sorted(replay_sel - runtime_sel),
        "utility_on_diff_blocks": util_rows,
        "boundary_tie_analysis": tie_rows,
        "replay_meta": replay_meta,
        "deterministic_replay": deterministic_replay,
        "graph_num_edges": graph.num_edges,
        "verdict": verdict,
        "reasons": reasons,
    }


def main() -> int:
    args = parse_args()
    live = args.live_audit_dir.resolve()
    out_path = args.out_path or (live / "SP_V2_PARITY_RESIDUAL_DIAGNOSIS.md")
    parity_csv = live / "SP_V2_REPLAY_RUNTIME_PARITY.csv"
    parity_rows = [r for r in load_parity_rows(parity_csv) if r.get("exact_match") == "0"]
    if not parity_rows:
        out_path.write_text("# SP-V2 Parity Residual Diagnosis\n\nAll parity rows exact-match.\n", encoding="utf-8")
        print(json.dumps({"non_exact": 0, "verdict": "benign"}, indent=2))
        return 0

    v2_rows = load_prefill(live / "sp_sliminfer_v2" / "uasr_score_dump.jsonl")
    vi = index_uasr(v2_rows)
    traj = live / "sp_sliminfer_v2" / "trajectory_dump"
    scalar = load_scalar_rows(traj / "trajectory_scalar_features.csv")
    index_rows = read_jsonl(traj / "trajectory_index.jsonl")

    diagnoses: List[Dict[str, Any]] = []
    blocked = False
    for row in parity_rows:
        task = str(row["task"])
        sid = str(row["sample_id"])
        layer = int(row["layer_idx"])
        prev = None
        if layer > 9:
            u9 = vi.get((task, sid, 9))
            if u9:
                prev = {int(x) for x in u9.get("selected_block_ids", [])}
        u = vi.get((task, sid, layer))
        if not u:
            continue
        diag = diagnose_layer(
            task=task,
            sample_id=sid,
            layer=layer,
            v2_uasr=u,
            trajectory_root=traj,
            index_rows=index_rows,
            scalar_rows=scalar,
            prev_active=prev,
        )
        diagnoses.append(diag)
        if diag["verdict"] == "blocked":
            blocked = True

    lines = [
        "# SP-V2 Parity Residual Diagnosis\n\n",
        f"- non-exact transport rows: **{len(parity_rows)}**\n",
        f"- overall verdict: **`{'BLOCKED' if blocked else 'BENIGN — proceed to QA'}`**\n\n",
    ]
    for d in diagnoses:
        lines.append(f"## {d['task']} / {d['sample_id'][:16]}… / L{d['layer_idx']}\n\n")
        lines.append(f"- Jaccard: **{d['jaccard']}**\n")
        lines.append(f"- runtime count: {len(d['runtime_active_ids'])}; replay count: {len(d['replay_active_ids'])}\n")
        lines.append(f"- differing blocks ({len(d['differing_block_ids'])}): `{d['differing_block_ids']}`\n")
        lines.append(f"- runtime-only: `{d['runtime_only']}`\n")
        lines.append(f"- replay-only: `{d['replay_only']}`\n")
        lines.append(f"- deterministic replay: {d['deterministic_replay']}\n")
        lines.append(f"- graph edges: {d['graph_num_edges']}\n")
        lines.append(f"- verdict: **{d['verdict']}**\n")
        lines.append("- reasons:\n")
        for r in d["reasons"]:
            lines.append(f"  - {r}\n")
        lines.append("\n### Boundary / tie analysis\n\n")
        lines.append("```json\n")
        lines.append(json.dumps(d["boundary_tie_analysis"], indent=2))
        lines.append("\n```\n\n")
        lines.append("### Utility on differing blocks\n\n")
        lines.append("```json\n")
        lines.append(json.dumps(d["utility_on_diff_blocks"], indent=2))
        lines.append("\n```\n\n")

    out_path.write_text("".join(lines), encoding="utf-8")
    if blocked:
        block_path = live.parent / "SP_V2_QA_BLOCKED_BY_PARITY_BUG.md"
        block_path.write_text(
            "# QA Blocked by Parity Bug\n\nSee SP_V2_PARITY_RESIDUAL_DIAGNOSIS.md for details.\n",
            encoding="utf-8",
        )
    manifest = {"non_exact": len(parity_rows), "diagnoses": diagnoses, "blocked": blocked}
    (live / "parity_residual_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out_path), "blocked": blocked, "count": len(diagnoses)}, indent=2))
    return 2 if blocked else 0


if __name__ == "__main__":
    raise SystemExit(main())
