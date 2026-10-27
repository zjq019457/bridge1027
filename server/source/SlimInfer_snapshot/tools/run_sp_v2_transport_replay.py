#!/usr/bin/env python3
"""SP-SlimInfer v2 offline policy replay on 24-sample trajectory dumps (no model, no QA)."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

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
    build_transport_graph,
    replay_sp_v2_sample_layer,
)
from bridgekv.sp_sliminfer import active_set_churn
from bridgekv.sp_trajectory import TRAJECTORY_PRUNE_LAYERS, load_sample_history_from_npz
from tools.run_bridgeflow_silm_property_probe import read_jsonl

DEFAULT_TRAJECTORY_ROOT = (
    REPO_ROOT / "results/sp_trajectory_probe_24sample_20260710_195052/trajectory_hook/trajectory_dump"
)
DEFAULT_UASR = REPO_ROOT / "results/sp_trajectory_probe_24sample_20260710_195052/baseline_no_hook/uasr_score_dump.jsonl"


def parse_args() -> argparse.Namespace:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trajectory_root", type=Path, default=DEFAULT_TRAJECTORY_ROOT)
    p.add_argument("--uasr_path", type=Path, default=DEFAULT_UASR)
    p.add_argument("--out_dir", type=Path, default=REPO_ROOT / f"results/sp_v2_transport_replay_{ts}")
    p.add_argument("--tau", type=float, default=DEFAULT_TAU)
    p.add_argument("--top_r", type=int, default=DEFAULT_TOP_R)
    p.add_argument("--replacement_cap", type=float, default=DEFAULT_REPLACEMENT_CAP)
    return p.parse_args()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def load_scalar_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_uasr_index(path: Path) -> Dict[Tuple[str, str, int], Mapping[str, Any]]:
    rows = [r for r in read_jsonl(path) if str(r.get("phase", "")) == "prefill"]
    out: Dict[Tuple[str, str, int], Mapping[str, Any]] = {}
    for row in rows:
        task = str(row.get("task", ""))
        sid = str(row.get("sample_id", ""))
        layer = int(row.get("layer_id", -1))
        if task and sid and layer >= 0:
            out[(task, sid, layer)] = row
    return out


def sample_list(index_rows: Sequence[Mapping[str, Any]]) -> List[Tuple[str, str]]:
    seen: Set[Tuple[str, str]] = set()
    for row in index_rows:
        seen.add((str(row["task"]), str(row["sample_id"])))
    return sorted(seen)


def layer_alive_blocks(scalar_rows: Sequence[Mapping[str, Any]], task: str, sid: str, layer: int) -> List[int]:
    bids: List[int] = []
    for row in scalar_rows:
        if row.get("task") != task or str(row.get("sample_id")) != sid:
            continue
        if int(row.get("layer_idx", -1)) != layer:
            continue
        if str(row.get("alive_before_pruning", "0")) not in {"1", "True", "true"}:
            continue
        bids.append(int(row["block_id"]))
    return sorted(set(bids))


def layer_score_map(scalar_rows: Sequence[Mapping[str, Any]], task: str, sid: str, layer: int) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for row in scalar_rows:
        if row.get("task") != task or str(row.get("sample_id")) != sid:
            continue
        if int(row.get("layer_idx", -1)) != layer:
            continue
        if str(row.get("alive_before_pruning", "0")) not in {"1", "True", "true"}:
            continue
        try:
            out[int(row["block_id"])] = float(row["original_sliminfer_score"])
        except (TypeError, ValueError):
            pass
    return out


def replay_all(
    *,
    samples: Sequence[Tuple[str, str]],
    scalar_rows: Sequence[Mapping[str, Any]],
    index_rows: Sequence[Mapping[str, Any]],
    uasr_index: Mapping[Tuple[str, str, int], Mapping[str, Any]],
    trajectory_root: Path,
    policy: str,
    tau: float,
    top_r: int,
    replacement_cap: float,
    weight_scale: float = 1.0,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    detail_rows: List[Dict[str, Any]] = []
    swap_rows: List[Dict[str, Any]] = []
    churns: List[float] = []
    layer_churns: Dict[int, List[float]] = defaultdict(list)
    changed_samples: Set[Tuple[str, str]] = set()
    changed_layers = 0
    sink_bad = 0
    over_budget = 0
    fallback_counts: List[int] = []
    fa_ratios: List[float] = []
    silm_fa_ratios: List[float] = []
    edge_sparsities: List[float] = []
    incoming_means: List[float] = []

    weights = {k: v * weight_scale for k, v in POLICY_WEIGHTS[policy].items()} if weight_scale != 1.0 else None

    for task, sid in samples:
        history = load_sample_history_from_npz(trajectory_root, index_rows, task, sid)
        sample_scalar = [r for r in scalar_rows if r.get("task") == task and str(r.get("sample_id")) == sid]
        prev_active: Optional[Set[int]] = None

        for layer in TRAJECTORY_PRUNE_LAYERS:
            uasr = uasr_index.get((task, sid, layer))
            if uasr is None:
                continue
            block_ids = layer_alive_blocks(sample_scalar, task, sid, layer)
            if not block_ids:
                block_ids = [int(x) for x in uasr.get("all_candidate_block_ids", [])]
            scores = layer_score_map(sample_scalar, task, sid, layer)
            budget = int(uasr.get("budget_k", 0) or 0)
            silm_selected = {int(x) for x in uasr.get("selected_block_ids", [])}

            selected, graph, meta = replay_sp_v2_sample_layer(
                layer_idx=layer,
                original_block_ids=block_ids,
                silm_scores=scores,
                silm_selected=silm_selected,
                budget_k=budget,
                history=history,
                scalar_rows=sample_scalar,
                policy=policy,
                prev_active=prev_active,
                tau=tau,
                top_r=top_r,
                replacement_cap=replacement_cap,
                weights=weights,
            )
            prev_active = set(selected)

            churn = float(meta.get("churn_rate", 0.0) or 0.0)
            if layer > 9:
                churns.append(churn)
                layer_churns[layer].append(churn)
            if selected != silm_selected:
                changed_layers += 1
                changed_samples.add((task, sid))
            if block_ids and int(block_ids[0]) not in selected:
                sink_bad += 1
            if len(selected) > budget:
                over_budget += 1

            fb = int(meta.get("fallback_selected_count", 0) or 0)
            fa_ratio = float(meta.get("feature_available_selected_ratio", 0.0) or 0.0)
            fallback_counts.append(fb)
            fa_ratios.append(fa_ratio)

            alive_fa = sum(
                1
                for r in sample_scalar
                if int(r.get("layer_idx", -1)) == layer
                and str(r.get("alive_before_pruning", "0")) in {"1", "True", "true"}
                and str(r.get("representation_available", "0")) in {"1", "True", "true"}
            )
            alive_total = sum(
                1
                for r in sample_scalar
                if int(r.get("layer_idx", -1)) == layer
                and str(r.get("alive_before_pruning", "0")) in {"1", "True", "true"}
            )
            silm_fa = sum(1 for b in silm_selected if any(
                int(r.get("block_id", -1)) == b
                and int(r.get("layer_idx", -1)) == layer
                and str(r.get("representation_available", "0")) in {"1", "True", "true"}
                for r in sample_scalar
            ))
            silm_fa_ratios.append(silm_fa / max(1, len(silm_selected)))

            if graph.num_edges > 0:
                edge_sparsities.append(graph.edge_sparsity)
            if graph.incoming_mass:
                incoming_means.append(float(np.mean(list(graph.incoming_mass.values()))))

            row = {
                "policy": policy,
                "task": task,
                "sample_id": sid,
                "layer_idx": layer,
                "budget_k": budget,
                "transport_applied": int(meta.get("transport_applied", False)),
                "churn_rate": round(churn, 6),
                "jaccard_vs_silm": meta.get("jaccard_vs_silm", 1.0),
                "changed": int(selected != silm_selected),
                "num_selected": len(selected),
                "swaps_applied": meta.get("swaps_applied", 0),
                "sink_kept": meta.get("sink_kept", 0),
                "over_budget": int(len(selected) > budget),
                "feature_available_selected_ratio": fa_ratio,
                "fallback_selected_count": fb,
                "transport_edges": graph.num_edges,
                "edge_sparsity": round(graph.edge_sparsity, 6),
                "incoming_mass_mean": meta.get("incoming_mass_mean", 0.0),
                "weight_scale": weight_scale,
            }
            detail_rows.append(row)
            for sw in meta.get("swaps", []) or []:
                swap_rows.append(
                    {
                        "policy": policy,
                        "task": task,
                        "sample_id": sid,
                        "layer_idx": layer,
                        "block_out": sw.get("out"),
                        "block_in": sw.get("in"),
                        "u_old": sw.get("u_old"),
                        "u_new": sw.get("u_new"),
                    }
                )

    summary = {
        "policy": policy,
        "num_samples": len(samples),
        "num_layers": len(detail_rows),
        "changed_samples": len(changed_samples),
        "changed_layers": changed_layers,
        "mean_churn": round(statistics.mean(churns), 6) if churns else 0.0,
        "median_churn": round(statistics.median(churns), 6) if churns else 0.0,
        "max_churn": round(max(churns), 6) if churns else 0.0,
        "layer_mean_churn": {str(k): round(statistics.mean(v), 6) for k, v in sorted(layer_churns.items())},
        "sink_violations": sink_bad,
        "over_budget": over_budget,
        "mean_feature_available_selected_ratio": round(statistics.mean(fa_ratios), 6) if fa_ratios else 0.0,
        "mean_silm_feature_available_ratio": round(statistics.mean(silm_fa_ratios), 6) if silm_fa_ratios else 0.0,
        "mean_fallback_selected": round(statistics.mean(fallback_counts), 6) if fallback_counts else 0.0,
        "mean_edge_sparsity": round(statistics.mean(edge_sparsities), 6) if edge_sparsities else 0.0,
        "mean_incoming_mass": round(statistics.mean(incoming_means), 6) if incoming_means else 0.0,
        "weight_scale": weight_scale,
    }
    return detail_rows, swap_rows, summary


def stability_audit(
    *,
    samples: Sequence[Tuple[str, str]],
    scalar_rows: Sequence[Mapping[str, Any]],
    index_rows: Sequence[Mapping[str, Any]],
    uasr_index: Mapping[Tuple[str, str, int], Mapping[str, Any]],
    trajectory_root: Path,
    policy: str,
    tau: float,
    top_r: int,
    replacement_cap: float,
) -> List[Dict[str, Any]]:
    _, _, base = replay_all(
        samples=samples,
        scalar_rows=scalar_rows,
        index_rows=index_rows,
        uasr_index=uasr_index,
        trajectory_root=trajectory_root,
        policy=policy,
        tau=tau,
        top_r=top_r,
        replacement_cap=replacement_cap,
        weight_scale=1.0,
    )
    base_sel: Dict[Tuple[str, str, int], Set[int]] = {}
    base_detail, _, _ = replay_all(
        samples=samples,
        scalar_rows=scalar_rows,
        index_rows=index_rows,
        uasr_index=uasr_index,
        trajectory_root=trajectory_root,
        policy=policy,
        tau=tau,
        top_r=top_r,
        replacement_cap=replacement_cap,
    )
    for row in base_detail:
        key = (row["task"], row["sample_id"], int(row["layer_idx"]))
        uasr = uasr_index.get(key)
        if not uasr:
            continue
        silm = {int(x) for x in uasr.get("selected_block_ids", [])}
        if int(row.get("changed", 0)):
            base_sel[key] = silm  # placeholder; recompute below

    # Re-run to capture actual selected sets for jaccard
    selected_map: Dict[Tuple[str, str, int], Set[int]] = {}
    for task, sid in samples:
        history = load_sample_history_from_npz(trajectory_root, index_rows, task, sid)
        sample_scalar = [r for r in scalar_rows if r.get("task") == task and str(r.get("sample_id")) == sid]
        prev_active = None
        for layer in TRAJECTORY_PRUNE_LAYERS:
            uasr = uasr_index.get((task, sid, layer))
            if not uasr:
                continue
            block_ids = layer_alive_blocks(sample_scalar, task, sid, layer) or [int(x) for x in uasr.get("all_candidate_block_ids", [])]
            scores = layer_score_map(sample_scalar, task, sid, layer)
            budget = int(uasr.get("budget_k", 0) or 0)
            silm_selected = {int(x) for x in uasr.get("selected_block_ids", [])}
            selected, _, _ = replay_sp_v2_sample_layer(
                layer_idx=layer,
                original_block_ids=block_ids,
                silm_scores=scores,
                silm_selected=silm_selected,
                budget_k=budget,
                history=history,
                scalar_rows=sample_scalar,
                policy=policy,
                prev_active=prev_active,
                tau=tau,
                top_r=top_r,
                replacement_cap=replacement_cap,
            )
            selected_map[(task, sid, layer)] = selected
            prev_active = set(selected)

    rows: List[Dict[str, Any]] = []
    for scale in (0.9, 1.1):
        pert_detail, _, pert_summary = replay_all(
            samples=samples,
            scalar_rows=scalar_rows,
            index_rows=index_rows,
            uasr_index=uasr_index,
            trajectory_root=trajectory_root,
            policy=policy,
            tau=tau * scale,
            top_r=top_r,
            replacement_cap=replacement_cap,
            weight_scale=scale,
        )
        jaccards: List[float] = []
        for row in pert_detail:
            key = (row["task"], row["sample_id"], int(row["layer_idx"]))
            if key not in selected_map:
                continue
            a = selected_map[key]
            # approximate perturbed selection via changed flag + silm; rerun layer individually for exact
            history = load_sample_history_from_npz(trajectory_root, index_rows, key[0], key[1])
            sample_scalar = [r for r in scalar_rows if r.get("task") == key[0] and str(r.get("sample_id")) == key[1]]
            uasr = uasr_index[key]
            block_ids = layer_alive_blocks(sample_scalar, key[0], key[1], key[2]) or [int(x) for x in uasr.get("all_candidate_block_ids", [])]
            scores = layer_score_map(sample_scalar, key[0], key[1], key[2])
            silm_selected = {int(x) for x in uasr.get("selected_block_ids", [])}
            prev_active = None
            for pl in TRAJECTORY_PRUNE_LAYERS:
                if pl >= key[2]:
                    break
                if (key[0], key[1], pl) in selected_map:
                    prev_active = selected_map[(key[0], key[1], pl)]
            pert_sel, _, _ = replay_sp_v2_sample_layer(
                layer_idx=key[2],
                original_block_ids=block_ids,
                silm_scores=scores,
                silm_selected=silm_selected,
                budget_k=int(uasr.get("budget_k", 0) or 0),
                history=history,
                scalar_rows=sample_scalar,
                policy=policy,
                prev_active=prev_active,
                tau=tau * scale,
                top_r=top_r,
                replacement_cap=replacement_cap,
                weights={k: v * scale for k, v in POLICY_WEIGHTS[policy].items()},
            )
            inter = len(a & pert_sel)
            union = len(a | pert_sel)
            jaccards.append(inter / max(1, union))
        rows.append(
            {
                "policy": policy,
                "perturbation": f"scale_{scale}",
                "tau": tau * scale,
                "weight_scale": scale,
                "mean_churn": pert_summary["mean_churn"],
                "mean_jaccard_vs_base": round(statistics.mean(jaccards), 6) if jaccards else 0.0,
                "min_jaccard_vs_base": round(min(jaccards), 6) if jaccards else 0.0,
            }
        )
    return rows


def missingness_bias_audit(detail_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    by_policy: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in detail_rows:
        if int(row.get("transport_applied", 0)):
            by_policy[str(row["policy"])].append(row)
    out: Dict[str, Any] = {}
    for policy, rows in by_policy.items():
        fa = [float(r.get("feature_available_selected_ratio", 0.0) or 0.0) for r in rows]
        changed = [r for r in rows if int(r.get("changed", 0))]
        out[policy] = {
            "layers_with_transport": len(rows),
            "mean_feature_available_selected_ratio": round(statistics.mean(fa), 6) if fa else 0.0,
            "changed_layers": len(changed),
            "mean_churn_changed_only": round(
                statistics.mean(float(r.get("churn_rate", 0.0) or 0.0) for r in changed), 6
            )
            if changed
            else 0.0,
            "bias_flag": int(statistics.mean(fa) > 0.98) if fa else 0,
        }
    return out


def tau_sensitivity_rows(
    *,
    samples: Sequence[Tuple[str, str]],
    scalar_rows: Sequence[Mapping[str, Any]],
    index_rows: Sequence[Mapping[str, Any]],
    uasr_index: Mapping[Tuple[str, str, int], Mapping[str, Any]],
    trajectory_root: Path,
    policy: str,
    replacement_cap: float,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for tau in (0.05, 0.10, 0.20):
        for top_r in (3, 5, 8):
            _, _, summary = replay_all(
                samples=samples,
                scalar_rows=scalar_rows,
                index_rows=index_rows,
                uasr_index=uasr_index,
                trajectory_root=trajectory_root,
                policy=policy,
                tau=tau,
                top_r=top_r,
                replacement_cap=replacement_cap,
            )
            rows.append({"policy": policy, "tau": tau, "top_r": top_r, **summary})
    return rows


def choose_primary_policy(summaries: Mapping[str, Mapping[str, Any]]) -> Tuple[str, str]:
    best = None
    best_score = -1.0
    rationale: List[str] = []
    for policy, s in summaries.items():
        churn = float(s.get("mean_churn", 0.0) or 0.0)
        score = 0.0
        if 0.03 <= churn <= 0.08:
            score += 3.0
        elif 0.0 < churn < 0.03:
            score += 1.0
        elif churn <= 0.15:
            score += 0.5
        if int(s.get("sink_violations", 0)) == 0:
            score += 2.0
        if int(s.get("over_budget", 0)) == 0:
            score += 2.0
        if churn > 0.0:
            score += 1.0
        if best is None or score > best_score:
            best = policy
            best_score = score
    if best is None:
        return "transport_residual", "fallback_default"
    rationale.append(f"highest structural score={best_score:.2f}")
    rationale.append(f"mean_churn={summaries[best].get('mean_churn')}")
    return best, "; ".join(rationale)


def evaluate_go(summaries: Mapping[str, Mapping[str, Any]], stability_rows: Sequence[Mapping[str, Any]]) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    ok = True
    for policy, s in summaries.items():
        churn = float(s.get("mean_churn", 0.0) or 0.0)
        if churn == 0.0:
            ok = False
            reasons.append(f"{policy}: churn=0")
        if churn > 0.15:
            ok = False
            reasons.append(f"{policy}: churn>{0.15}")
        if int(s.get("sink_violations", 0)) > 0:
            ok = False
            reasons.append(f"{policy}: sink violations")
        if int(s.get("over_budget", 0)) > 0:
            ok = False
            reasons.append(f"{policy}: over budget")
    stab = [float(r.get("mean_jaccard_vs_base", 0.0) or 0.0) for r in stability_rows]
    if stab and min(stab) < 0.85:
        reasons.append(f"stability min jaccard {min(stab):.3f} < 0.85 (warning)")
    if not reasons:
        reasons.append("all structural checks passed")
    return ok, reasons


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    index_rows = read_jsonl(args.trajectory_root / "trajectory_index.jsonl")
    scalar_rows = load_scalar_rows(args.trajectory_root / "trajectory_scalar_features.csv")
    uasr_index = load_uasr_index(args.uasr_path)
    samples = sample_list(index_rows)

    all_detail: List[Dict[str, Any]] = []
    all_swaps: List[Dict[str, Any]] = []
    summaries: Dict[str, Dict[str, Any]] = {}
    sensitivity_all: List[Dict[str, Any]] = []
    stability_all: List[Dict[str, Any]] = []

    for policy in POLICIES:
        detail, swaps, summary = replay_all(
            samples=samples,
            scalar_rows=scalar_rows,
            index_rows=index_rows,
            uasr_index=uasr_index,
            trajectory_root=args.trajectory_root,
            policy=policy,
            tau=args.tau,
            top_r=args.top_r,
            replacement_cap=args.replacement_cap,
        )
        all_detail.extend(detail)
        all_swaps.extend(swaps)
        summaries[policy] = summary
        sensitivity_all.extend(
            tau_sensitivity_rows(
                samples=samples,
                scalar_rows=scalar_rows,
                index_rows=index_rows,
                uasr_index=uasr_index,
                trajectory_root=args.trajectory_root,
                policy=policy,
                replacement_cap=args.replacement_cap,
            )
        )
        stability_all.extend(
            stability_audit(
                samples=samples,
                scalar_rows=scalar_rows,
                index_rows=index_rows,
                uasr_index=uasr_index,
                trajectory_root=args.trajectory_root,
                policy=policy,
                tau=args.tau,
                top_r=args.top_r,
                replacement_cap=args.replacement_cap,
            )
        )

    missingness = missingness_bias_audit(all_detail)
    primary, rationale = choose_primary_policy(summaries)
    go, go_reasons = evaluate_go(summaries, stability_all)

    write_csv(
        args.out_dir / "SP_V2_POLICY_REPLAY.csv",
        all_detail,
        [
            "policy", "task", "sample_id", "layer_idx", "budget_k", "transport_applied", "churn_rate",
            "jaccard_vs_silm", "changed", "num_selected", "swaps_applied", "sink_kept", "over_budget",
            "feature_available_selected_ratio", "fallback_selected_count", "transport_edges",
            "edge_sparsity", "incoming_mass_mean",
        ],
    )
    write_csv(args.out_dir / "SP_V2_SWAP_AUDIT.csv", all_swaps,
              ["policy", "task", "sample_id", "layer_idx", "block_out", "block_in", "u_old", "u_new"])
    write_csv(args.out_dir / "SP_V2_TAU_SENSITIVITY.csv", sensitivity_all,
              ["policy", "tau", "top_r", "mean_churn", "median_churn", "max_churn", "changed_layers", "sink_violations"])
    write_csv(args.out_dir / "SP_V2_POLICY_STABILITY.csv", stability_all,
              ["policy", "perturbation", "tau", "weight_scale", "mean_churn", "mean_jaccard_vs_base", "min_jaccard_vs_base"])

    replay_md = ["# SP-V2 Policy Replay Audit\n"]
    for policy, s in summaries.items():
        replay_md.append(f"## {policy}\n")
        for k, v in s.items():
            replay_md.append(f"- {k}: {v}")
        replay_md.append("")
    write_text(args.out_dir / "SP_V2_POLICY_REPLAY_AUDIT.md", "\n".join(replay_md))

    miss_md = ["# SP-V2 Missingness Bias Audit\n", "```json\n", json.dumps(missingness, indent=2), "\n```\n"]
    write_text(args.out_dir / "SP_V2_MISSINGNESS_BIAS_AUDIT.md", "".join(miss_md))

    stab_md = ["# SP-V2 Policy Stability Audit\n"]
    for row in stability_all:
        stab_md.append(f"- {row}")
    stab_md.append("\n## Tau/Top-R sensitivity\n")
    for row in sensitivity_all:
        stab_md.append(
            f"- {row['policy']} tau={row['tau']} top_r={row['top_r']}: mean_churn={row.get('mean_churn')} changed_layers={row.get('changed_layers')}"
        )
    write_text(args.out_dir / "SP_V2_POLICY_STABILITY_AUDIT.md", "\n".join(stab_md))

    verdict = "GO" if go else "NO-GO"
    verdict_path = args.out_dir / (f"SP_V2_POLICY_{verdict}.md" if go else "SP_V2_POLICY_NO_GO.md")
    write_text(
        verdict_path,
        f"# SP-V2 Policy Replay {verdict}\n\n"
        f"- primary policy (pre-registered recommendation): **{primary}**\n"
        f"- rationale: {rationale}\n\n"
        f"## Checks\n" + "\n".join(f"- {r}" for r in go_reasons) + "\n",
    )

    manifest = {
        "samples": len(samples),
        "policies": list(POLICIES),
        "summaries": summaries,
        "missingness": missingness,
        "primary_policy": primary,
        "primary_rationale": rationale,
        "verdict": verdict,
        "go_reasons": go_reasons,
    }
    write_text(args.out_dir / "replay_manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")

    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0 if go else 1


if __name__ == "__main__":
    raise SystemExit(main())
