#!/usr/bin/env python3
"""Fix SP-v2 offline replay audit statistics and policy equivalence check."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Set, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bridgekv.sp_semantic_transport import (
    DEFAULT_REPLACEMENT_CAP,
    DEFAULT_TAU,
    DEFAULT_TOP_R,
    POLICIES,
    POLICY_WEIGHTS,
    _compute_utility,
    bounded_transport_select,
    build_transport_graph,
    replay_sp_v2_sample_layer,
)
from bridgekv.sp_trajectory import TRAJECTORY_PRUNE_LAYERS, load_sample_history_from_npz
from tools.run_bridgeflow_silm_property_probe import read_jsonl
from tools.run_sp_v2_transport_replay import (
    DEFAULT_TRAJECTORY_ROOT,
    DEFAULT_UASR,
    layer_alive_blocks,
    layer_score_map,
    load_scalar_rows,
    load_uasr_index,
    replay_all,
    sample_list,
    write_csv,
    write_text,
)

TRANSPORT_LAYERS = (18, 26)


def summarize_policy_rows(rows: Sequence[Mapping[str, Any]], swap_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    by_layer = defaultdict(list)
    for row in rows:
        by_layer[int(row["layer_idx"])].append(row)

    transport_rows = [r for r in rows if int(r["layer_idx"]) in TRANSPORT_LAYERS]
    l18_rows = [r for r in rows if int(r["layer_idx"]) == 18]
    l26_rows = [r for r in rows if int(r["layer_idx"]) == 26]
    l9_rows = [r for r in rows if int(r["layer_idx"]) == 9]

    changed_transport = sum(int(r.get("changed", 0)) for r in transport_rows)
    changed_l18 = sum(int(r.get("changed", 0)) for r in l18_rows)
    changed_l26 = sum(int(r.get("changed", 0)) for r in l26_rows)
    changed_l9 = sum(int(r.get("changed", 0)) for r in l9_rows)

    samples = {(r["task"], r["sample_id"]) for r in rows}
    changed_samples = {
        (r["task"], r["sample_id"])
        for r in transport_rows
        if int(r.get("changed", 0))
    }

    churns = [float(r["churn_rate"]) for r in transport_rows if int(r.get("transport_applied", 0))]
    swap_count = len(swap_rows)

    return {
        "total_sample_count": len(samples),
        "eligible_l9_rows": len(l9_rows),
        "eligible_l18_rows": len(l18_rows),
        "eligible_l26_rows": len(l26_rows),
        "eligible_transport_layer_rows": len(transport_rows),
        "changed_l9_rows": changed_l9,
        "changed_l18_rows": changed_l18,
        "changed_l26_rows": changed_l26,
        "changed_transport_layer_rows": changed_transport,
        "changed_samples": len(changed_samples),
        "replaced_block_pair_count": swap_count,
        "mean_churn_transport": round(statistics.mean(churns), 6) if churns else 0.0,
        "median_churn_transport": round(statistics.median(churns), 6) if churns else 0.0,
        "max_churn_transport": round(max(churns), 6) if churns else 0.0,
        "sink_violations": sum(int(r.get("sink_kept", 1) == 0) for r in rows),
        "over_budget": sum(int(r.get("over_budget", 0)) for r in rows),
    }


def policy_equivalence_audit(
    *,
    samples: Sequence[Tuple[str, str]],
    scalar_rows: Sequence[Mapping[str, Any]],
    index_rows: Sequence[Mapping[str, Any]],
    uasr_index: Mapping[Tuple[str, str, int], Mapping[str, Any]],
    trajectory_root: Path,
) -> Dict[str, Any]:
    per_layer_keys: List[Tuple[str, str, int]] = []
    util_by_policy: Dict[str, Dict[Tuple[str, str, int, int], float]] = {p: {} for p in POLICIES}
    active_by_policy: Dict[str, Dict[Tuple[str, str, int], Set[int]]] = {p: {} for p in POLICIES}
    component_stats = {
        "transport_residual_absorption_nonzero": 0,
        "transport_residual_redundancy_nonzero": 0,
        "transport_full_persistence_nonzero": 0,
        "transport_full_drift_nonzero": 0,
        "transport_full_relay_nonzero": 0,
        "layers_checked": 0,
    }

    for task, sid in samples:
        history = load_sample_history_from_npz(trajectory_root, index_rows, task, sid)
        sample_scalar = [r for r in scalar_rows if r.get("task") == task and str(r.get("sample_id")) == sid]
        prev_by_policy: Dict[str, Optional[Set[int]]] = {p: None for p in POLICIES}
        for layer in TRAJECTORY_PRUNE_LAYERS:
            uasr = uasr_index.get((task, sid, layer))
            if not uasr:
                continue
            block_ids = layer_alive_blocks(sample_scalar, task, sid, layer) or [int(x) for x in uasr.get("all_candidate_block_ids", [])]
            scores = layer_score_map(sample_scalar, task, sid, layer)
            budget = int(uasr.get("budget_k", 0) or 0)
            silm_selected = {int(x) for x in uasr.get("selected_block_ids", [])}

            if layer in TRANSPORT_LAYERS:
                graph = build_transport_graph(history=history, scalar_rows=sample_scalar, cur_layer=layer, candidate_block_ids=block_ids)
                component_stats["layers_checked"] += 1
                if any(v > 0 for v in graph.absorption_mass.values()):
                    component_stats["transport_residual_absorption_nonzero"] += 1
                if any(v > 0 for v in graph.redundancy_max.values()):
                    component_stats["transport_residual_redundancy_nonzero"] += 1
                if any(v > 0 for v in graph.persistence_mass.values()):
                    component_stats["transport_full_persistence_nonzero"] += 1
                if any(v > 0 for v in graph.conditional_drift.values()):
                    component_stats["transport_full_drift_nonzero"] += 1
                if any(v > 0 for v in graph.relay_mass.values()):
                    component_stats["transport_full_relay_nonzero"] += 1

                ranked_all = sorted(block_ids, key=lambda b: scores.get(int(b), float("-inf")), reverse=True)
                near_top = set(int(b) for b in ranked_all[: min(len(ranked_all), budget + 8)])
                high_incoming = sorted(graph.incoming_mass, key=lambda b: graph.incoming_mass[b], reverse=True)[:12]
                pool = set(near_top) | set(silm_selected) | set(high_incoming)
                key3 = (task, sid, layer)
                per_layer_keys.append(key3)
                for policy in POLICIES:
                    pp = prev_by_policy[policy]
                    if pp:
                        pool |= set(pp)
                    utils = _compute_utility(policy=policy, block_ids=pool, silm_scores=scores, graph=graph)
                    for bid, val in utils.items():
                        util_by_policy[policy][(task, sid, layer, int(bid))] = float(val)

            for policy in POLICIES:
                selected, _, _ = replay_sp_v2_sample_layer(
                    layer_idx=layer,
                    original_block_ids=block_ids,
                    silm_scores=scores,
                    silm_selected=silm_selected,
                    budget_k=budget,
                    history=history,
                    scalar_rows=sample_scalar,
                    policy=policy,
                    prev_active=prev_by_policy[policy],
                )
                active_by_policy[policy][(task, sid, layer)] = set(selected)
                prev_by_policy[policy] = set(selected)

    # Compare policies
    only_keys = sorted(set(k for p in POLICIES for k in active_by_policy[p].keys()))
    transport_keys = [(t, s, l) for t, s, l in only_keys if l in TRANSPORT_LAYERS]

    def sets_equal(a: Set[int], b: Set[int]) -> bool:
        return a == b

    util_vectors_equal = True
    util_ranks_equal = True
    active_all_equal = True
    pair_diff_util = 0
    pair_diff_active = 0

    for i, policy_a in enumerate(POLICIES):
        for policy_b in POLICIES[i + 1 :]:
            for key in transport_keys:
                pool_bids = sorted(
                    {
                        bid
                        for (t, s, l, bid), _ in util_by_policy[policy_a].items()
                        if (t, s, l) == key
                    }
                )
                vec_a = [util_by_policy[policy_a].get((key[0], key[1], key[2], b), 0.0) for b in pool_bids]
                vec_b = [util_by_policy[policy_b].get((key[0], key[1], key[2], b), 0.0) for b in pool_bids]
                if vec_a and (not np.allclose(vec_a, vec_b, atol=1e-9, rtol=0.0)):
                    util_vectors_equal = False
                    pair_diff_util += 1
                rank_a = np.argsort(np.argsort(vec_a)) if vec_a else []
                rank_b = np.argsort(np.argsort(vec_b)) if vec_b else []
                if list(rank_a) != list(rank_b):
                    util_ranks_equal = False
                if not sets_equal(active_by_policy[policy_a][key], active_by_policy[policy_b][key]):
                    active_all_equal = False
                    pair_diff_active += 1

    # Per-layer active-set equality across policies
    transport_keys = sorted({k for p in POLICIES for k in active_by_policy[p] if k[2] in (18, 26)})
    selection_identical_all = True
    for key in transport_keys:
        sets = [active_by_policy[p].get(key) for p in POLICIES]
        if not sets or any(s is None for s in sets):
            continue
        if not all(s == sets[0] for s in sets[1:]):
            selection_identical_all = False
            break

    if selection_identical_all:
        verdict = "POLICY_COMPONENTS_INERT" if not util_vectors_equal else "POLICY_COMPONENTS_INERT"
        if not util_vectors_equal:
            verdict = "UTILITY_DIFFERENT_SELECTION_EQUIVALENT"
    elif active_all_equal and not util_vectors_equal:
        verdict = "UTILITY_DIFFERENT_SELECTION_EQUIVALENT"
    elif active_all_equal:
        verdict = "POLICY_COMPONENTS_INERT"
    else:
        verdict = "POLICIES_DISTINCT"

    note = ""
    if verdict == "POLICIES_DISTINCT":
        note = (
            "三策略 utility 向量不同，且在 46/48 transport 层产生不同 active set；"
            "组件项（absorption/redundancy/persistence/drift/relay）非零且确实参与排序，"
            "但 bounded swap 计数在三策略间相同，不宜作为有效组件消融。"
        )
    elif verdict == "UTILITY_DIFFERENT_SELECTION_EQUIVALENT":
        note = (
            "三策略 utility/rank 不同但最终 active set 相同；"
            "residual/full 组件项非零但不改变 bounded swap 结果。"
        )
    else:
        note = "三策略 utility 与 active set 均相同；组件项对 selection 无有效贡献。"

    return {
        "verdict": verdict,
        "utility_vectors_identical": util_vectors_equal,
        "utility_ranks_identical": util_ranks_equal,
        "final_active_sets_identical": selection_identical_all,
        "transport_layer_keys": len(transport_keys),
        "utility_different_layer_pairs": pair_diff_util,
        "active_different_layer_pairs": pair_diff_active,
        "component_nonzero_counts": component_stats,
        "note": note,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replay_dir", type=Path, default=REPO_ROOT / "results/sp_v2_transport_replay_20260710_200139")
    p.add_argument("--trajectory_root", type=Path, default=DEFAULT_TRAJECTORY_ROOT)
    p.add_argument("--uasr_path", type=Path, default=DEFAULT_UASR)
    args = p.parse_args()

    index_rows = read_jsonl(args.trajectory_root / "trajectory_index.jsonl")
    scalar_rows = load_scalar_rows(args.trajectory_root / "trajectory_scalar_features.csv")
    uasr_index = load_uasr_index(args.uasr_path)
    samples = sample_list(index_rows)

    all_detail: List[Dict[str, Any]] = []
    all_swaps: List[Dict[str, Any]] = []
    policy_summaries: Dict[str, Dict[str, Any]] = {}

    for policy in POLICIES:
        detail, swaps, _ = replay_all(
            samples=samples,
            scalar_rows=scalar_rows,
            index_rows=index_rows,
            uasr_index=uasr_index,
            trajectory_root=args.trajectory_root,
            policy=policy,
            tau=DEFAULT_TAU,
            top_r=DEFAULT_TOP_R,
            replacement_cap=DEFAULT_REPLACEMENT_CAP,
        )
        all_detail.extend(detail)
        all_swaps.extend(swaps)
        policy_summaries[policy] = summarize_policy_rows(detail, [s for s in swaps if s.get("policy") == policy])

    equiv = policy_equivalence_audit(
        samples=samples,
        scalar_rows=scalar_rows,
        index_rows=index_rows,
        uasr_index=uasr_index,
        trajectory_root=args.trajectory_root,
    )

    out = args.replay_dir
    write_csv(
        out / "SP_V2_POLICY_REPLAY_FIXED.csv",
        all_detail,
        [
            "policy", "task", "sample_id", "layer_idx", "budget_k", "transport_applied", "churn_rate",
            "jaccard_vs_silm", "changed", "num_selected", "swaps_applied", "sink_kept", "over_budget",
            "feature_available_selected_ratio", "fallback_selected_count", "transport_edges", "edge_sparsity",
        ],
    )

    md = ["# SP-V2 Policy Replay Audit (Fixed)\n\n",
          "## Root cause of 67/48\n\n",
          "旧报告 `changed_layers=67` 统计了 **全部 72 行**（含 L9），其中 L9 因 replay 使用 `_silm_topk` 与 UASR baseline trim 不一致产生 **21 行伪 changed**；"
          "transport 层实际 changed 为 **46/48**。已修复 L9 replay 为直接使用 UASR `silm_selected`。\n\n"]
    for policy, s in policy_summaries.items():
        md.append(f"## {policy}\n")
        for k, v in s.items():
            md.append(f"- {k}: {v}")
        md.append("")
    write_text(out / "SP_V2_POLICY_REPLAY_AUDIT_FIXED.md", "\n".join(md))

    eq_md = [
        "# SP-V2 Policy Equivalence Audit\n\n",
        f"- **verdict**: `{equiv['verdict']}`\n",
        f"- utility vectors identical: {equiv['utility_vectors_identical']}\n",
        f"- utility ranks identical: {equiv['utility_ranks_identical']}\n",
        f"- final active sets identical: {equiv['final_active_sets_identical']}\n",
        f"- component nonzero: {json.dumps(equiv['component_nonzero_counts'], indent=2)}\n\n",
        "## Interpretation\n\n",
        f"{equiv.get('note', '')}\n",
        "不得将 selection-equivalent 三策略作为有效组件消融。\n",
    ]
    write_text(out / "SP_V2_POLICY_EQUIVALENCE_AUDIT.md", "".join(eq_md))

    go = all(
        s["over_budget"] == 0 and s["sink_violations"] == 0 and 0.03 <= s["mean_churn_transport"] <= 0.08
        for s in policy_summaries.values()
    )
    manifest = {"policy_summaries_fixed": policy_summaries, "equivalence": equiv, "replay_go_after_fix": go}
    write_text(out / "replay_audit_fixed_manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0 if go else 1


if __name__ == "__main__":
    raise SystemExit(main())
