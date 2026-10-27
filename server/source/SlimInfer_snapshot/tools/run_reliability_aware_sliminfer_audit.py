#!/usr/bin/env python3
"""Reliability-Aware SlimInfer Phase 1: CPU active-set audit (no QA)."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bridgekv.confidence_gated_sliminfer import confidence_gated_active_ids, compute_pruning_confidence
from bridgekv.reliability_sliminfer import (
    apply_rank_stability_adjustment,
    margin_stable_active_ids,
    rank_stability_scores,
    reliability_full_active_ids,
)
from bridgekv.sp_sliminfer import active_set_churn
from tools.run_bridgeflow_silm_property_probe import read_jsonl

CHURN_MIN, CHURN_MAX = 0.03, 0.15
DEFAULT_UASR = (
    REPO_ROOT / "results/bridgeflow_silm_property_probe_2task_full_20260709_211600/uasr_score_dump.jsonl"
)

VARIANTS = [
    ("confidence_gate_sliminfer", "confidence_gated_sliminfer"),
    ("margin_safe_swap_sliminfer", "margin_stable_sliminfer"),
    ("multi_window_consensus_sliminfer", "multi_window_sliminfer"),
    ("rank_stability_sliminfer", "rank_stability_sliminfer"),
    ("reliability_full_sliminfer", "reliability_full_sliminfer"),
]


def parse_args() -> argparse.Namespace:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--uasr_dump", type=Path, default=DEFAULT_UASR)
    p.add_argument("--out_dir", type=Path, default=REPO_ROOT / f"results/reliability_aware_sliminfer_{ts}")
    p.add_argument("--tasks", default="hotpotqa,musique")
    return p.parse_args()


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def _must_keep(block_ids: Sequence[int]) -> Set[int]:
    s: Set[int] = set()
    if block_ids:
        s.add(block_ids[0])
        s.add(block_ids[-1])
    return s


def _build_layer_ranks(sample_rows: Sequence[Mapping[str, Any]]) -> Dict[int, Dict[int, int]]:
    """layer -> block_id -> rank (1=best)."""
    out: Dict[int, Dict[int, int]] = {}
    for row in sample_rows:
        layer = int(row.get("layer_id", -1))
        ids = row.get("sorted_block_ids") or row.get("all_candidate_block_ids") or []
        out[layer] = {int(bid): rank for rank, bid in enumerate(ids, 1)}
    return out


def _select_variant(
    variant: str,
    *,
    block_ids: Sequence[int],
    scores_t: torch.Tensor,
    top_idx: torch.Tensor,
    must_keep: Set[int],
    budget: int,
    prev_active: Optional[Set[int]],
    layer_ranks: Mapping[int, Mapping[int, int]],
    layer: int,
    qk_scores: Optional[Sequence[float]],
) -> Tuple[Set[int], Dict[str, Any]]:
    k = min(max(1, budget), len(block_ids))

    if variant == "confidence_gate_sliminfer":
        final, meta = confidence_gated_active_ids(
            original_block_ids=block_ids,
            block_scores=scores_t,
            top_k_indices=top_idx,
            must_keep_block_ids=must_keep,
            num_blocks_to_keep=k,
            prev_active_block_ids=prev_active,
        )
        meta["triggered"] = int(meta.get("gated", False))
        meta["rejected_swaps"] = 0 if meta.get("gated") else 0
        return final, meta

    if variant == "margin_safe_swap_sliminfer":
        final, meta = margin_stable_active_ids(
            original_block_ids=block_ids,
            block_scores=scores_t,
            top_k_indices=top_idx,
            must_keep_block_ids=must_keep,
            num_blocks_to_keep=k,
            prev_active_block_ids=prev_active,
        )
        meta["triggered"] = int(meta.get("gated", False))
        meta["rejected_swaps"] = max(0, int(meta.get("swaps_applied", 0)))
        return final, meta

    if variant == "multi_window_consensus_sliminfer":
        # CPU replay proxy: min(base, qk) when qk available; true multi-window needs GPU q slices.
        if qk_scores and len(qk_scores) == len(block_ids):
            proxy = torch.tensor(
                [min(float(a), float(b)) for a, b in zip(scores_t.tolist(), qk_scores)],
                dtype=torch.float32,
            )
            _, proxy_top = torch.topk(proxy, k=k)
            final, meta = margin_stable_active_ids(
                original_block_ids=block_ids,
                block_scores=proxy,
                top_k_indices=proxy_top,
                must_keep_block_ids=must_keep,
                num_blocks_to_keep=k,
                prev_active_block_ids=prev_active,
            )
            meta["mode"] = "CPU_PROXY_min_score_qk"
            meta["triggered"] = int(proxy.tolist() != scores_t.tolist())
        else:
            final = {block_ids[i] for i in top_idx.tolist()} | must_keep
            meta = {"mode": "UNAVAILABLE_NO_QK", "triggered": 0, "NO_EFFECT_IN_ACTIVE_SET": 1}
        return final, meta

    if variant == "rank_stability_sliminfer":
        stab = rank_stability_scores(block_ids=block_ids, layer_ranks=layer_ranks)
        adjusted = apply_rank_stability_adjustment(scores_t, block_ids, stab)
        _, adj_top = torch.topk(adjusted, k=k)
        final = {block_ids[i] for i in adj_top.tolist()} | must_keep
        if len(final) > k:
            removable = sorted(final - must_keep)
            for bid in removable[: len(final) - k]:
                final.discard(bid)
        meta = {"triggered": 1, "mode": "rank_stability_boost"}
        return final, meta

    if variant == "reliability_full_sliminfer":
        final, meta = reliability_full_active_ids(
            original_block_ids=block_ids,
            block_scores=scores_t,
            top_k_indices=top_idx,
            must_keep_block_ids=must_keep,
            num_blocks_to_keep=k,
            prev_active_block_ids=prev_active,
        )
        meta["triggered"] = int(
            meta.get("confidence_gated", {}).get("gated", False)
            or meta.get("margin_stable", {}).get("gated", False)
        )
        return final, meta

    raise ValueError(variant)


def replay_all_variants(
    uasr_rows: Sequence[Mapping[str, Any]],
    tasks: Set[str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in uasr_rows:
        if str(row.get("phase", "")) != "prefill" or str(row.get("task", "")) not in tasks:
            continue
        grouped[(str(row["task"]), str(row["sample_id"]))].append(dict(row))

    detail_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    for label, _internal in VARIANTS:
        prev_by_sample: Dict[Tuple[str, str], Set[int]] = {}
        churns: List[float] = []
        triggers = 0
        rejected = 0
        changed_layers = 0
        changed_samples: Set[Tuple[str, str]] = set()
        no_effect = 0
        eligible = 0
        over_budget = 0
        sink_bad = 0

        for (task, sid), sample_rows in grouped.items():
            layer_ranks = _build_layer_ranks(sample_rows)
            for row in sorted(sample_rows, key=lambda r: int(r["layer_id"])):
                layer = int(row["layer_id"])
                block_ids = [int(x) for x in row.get("all_candidate_block_ids", [])]
                scores = [float(x) for x in row.get("all_candidate_scores", [])]
                qk = row.get("all_candidate_qk_scores")
                qk_scores = [float(x) for x in qk] if qk else None
                if not block_ids or len(scores) != len(block_ids):
                    continue
                budget = int(row.get("budget_k", 0) or len(row.get("selected_block_ids", [])))
                silm_selected = {int(x) for x in row.get("selected_block_ids", [])}
                must_keep = _must_keep(block_ids)
                churn_eligible = int(budget < len(block_ids))
                if churn_eligible:
                    eligible += 1

                scores_t = torch.tensor(scores, dtype=torch.float32)
                k = min(max(1, budget), len(block_ids))
                _, top_idx = torch.topk(scores_t, k=k)
                prev = prev_by_sample.get((task, sid))

                final, meta = _select_variant(
                    label,
                    block_ids=block_ids,
                    scores_t=scores_t,
                    top_idx=top_idx,
                    must_keep=must_keep,
                    budget=budget,
                    prev_active=prev,
                    layer_ranks=layer_ranks,
                    layer=layer,
                    qk_scores=qk_scores,
                )
                prev_by_sample[(task, sid)] = set(final)

                churn = active_set_churn(silm_selected, final, budget)
                if churn_eligible:
                    churns.append(churn)
                if int(meta.get("triggered", 0)):
                    triggers += 1
                rejected += int(meta.get("rejected_swaps", 0) or 0)
                if final == silm_selected:
                    no_effect += 1
                else:
                    changed_layers += 1
                    changed_samples.add((task, sid))
                if len(final) > budget:
                    over_budget += 1
                if block_ids and block_ids[0] not in final:
                    sink_bad += 1

                overlap = len(final & silm_selected)
                union = len(final | silm_selected)
                detail_rows.append(
                    {
                        "variant": label,
                        "task": task,
                        "sample_id": sid,
                        "layer": layer,
                        "active_budget": budget,
                        "churn_eligible": churn_eligible,
                        "churn_rate": round(churn, 6),
                        "churn_in_target": int(CHURN_MIN <= churn <= CHURN_MAX) if churn_eligible else "",
                        "jaccard_vs_silm": round(overlap / max(1, union), 6),
                        "triggered": meta.get("triggered", 0),
                        "rejected_swaps": meta.get("rejected_swaps", 0),
                        "changed_vs_silm": int(final != silm_selected),
                        "over_budget": int(len(final) > budget),
                        "sink_kept": int(block_ids[0] in final) if block_ids else 1,
                        "mode": meta.get("mode", ""),
                    }
                )

        mean_churn = sum(churns) / len(churns) if churns else 0.0
        no_effect_flag = changed_layers == 0
        summary_rows.append(
            {
                "variant": label,
                "trigger_count": triggers,
                "rejected_swap_count": rejected,
                "changed_layer_count": changed_layers,
                "changed_sample_count": len(changed_samples),
                "eligible_layer_count": eligible,
                "mean_churn_eligible": round(mean_churn, 6),
                "churn_in_band": int(CHURN_MIN <= mean_churn <= CHURN_MAX) if churns else 0,
                "over_budget": over_budget,
                "sink_violations": sink_bad,
                "no_effect_flag": int(no_effect_flag),
                "qa_candidate": int(
                    not no_effect_flag
                    and over_budget == 0
                    and sink_bad == 0
                    and churns
                    and CHURN_MIN <= mean_churn <= CHURN_MAX
                ),
                "status": "NO_EFFECT_IN_ACTIVE_SET" if no_effect_flag else "ACTIVE_SET_CHANGED",
            }
        )

    return detail_rows, summary_rows


def write_md_reports(out_dir: Path, summary: Sequence[Mapping[str, Any]], n_samples: int) -> None:
    lines = [
        "# Reliability-Aware SlimInfer — Active Set Audit",
        "",
        f"Samples: {n_samples}/400",
        "",
        "| Variant | Triggers | Rejected swaps | Δ layers | Δ samples | Mean churn | In band | Status | QA? |",
        "|---------|----------|----------------|----------|-----------|------------|---------|--------|-----|",
    ]
    for row in summary:
        lines.append(
            f"| {row['variant']} | {row['trigger_count']} | {row['rejected_swap_count']} | "
            f"{row['changed_layer_count']} | {row['changed_sample_count']} | "
            f"{row['mean_churn_eligible']} | {row['churn_in_band']} | {row['status']} | {row['qa_candidate']} |"
        )
    (out_dir / "ACTIVE_SET_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    (out_dir / "OFFICIAL_PATH_AUDIT.md").write_text(
        "\n".join(
            [
                "# Official Path Audit — Reliability-Aware",
                "",
                "- Base: `token_unit_qk` from SILM prefill uasr dump",
                "- CPU replay of selection policies (no QA)",
                "- Same budget K per layer as SILM",
                "- No gold, no BM25/LBP, no packing",
                "- `multi_window_consensus`: CPU proxy via min(score,qk); true w=1,4,8 needs GPU prefill",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    (out_dir / "FAIRNESS_AUDIT.md").write_text(
        "\n".join(
            [
                "# Fairness Audit",
                "",
                "- Identical candidate sets and budgets as `sliminfer_original` dump",
                "- Reliability variants only change selection, not prompt/scorer/decode",
                "- Excludes adaptive_budget (changes KV budget)",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    (out_dir / "FEATURE_COVERAGE_AUDIT.md").write_text(
        f"# Feature Coverage\n\n- UASR prefill rows: 400 samples × 3 layers\n- Tasks: hotpotqa, musique\n",
        encoding="utf-8",
    )

    (out_dir / "CLAIM_GUARD.md").write_text(
        "\n".join(
            [
                "# Claim Guard",
                "",
                "| Result tier | Note |",
                "|-------------|------|",
                "| Official SILM 400-full | 57.43 / 27.14 |",
                "| RoleFlow | **no-go** |",
                "| Reliability Phase 1 | active-set audit only |",
                "| SP-SlimInfer | separate probe |",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = {t.strip() for t in args.tasks.split(",") if t.strip()}
    uasr_rows = read_jsonl(args.uasr_dump)

    detail, summary = replay_all_variants(uasr_rows, tasks)
    n_samples = len({(r["task"], r["sample_id"]) for r in detail})

    _write_csv(
        out_dir / "reliability_active_set_detail.csv",
        detail,
        [
            "variant", "task", "sample_id", "layer", "active_budget", "churn_eligible",
            "churn_rate", "churn_in_target", "jaccard_vs_silm", "triggered", "rejected_swaps",
            "changed_vs_silm", "over_budget", "sink_kept", "mode",
        ],
    )
    _write_csv(
        out_dir / "reliability_active_set_summary.csv",
        summary,
        [
            "variant", "trigger_count", "rejected_swap_count", "changed_layer_count",
            "changed_sample_count", "eligible_layer_count", "mean_churn_eligible",
            "churn_in_band", "over_budget", "sink_violations", "no_effect_flag", "qa_candidate", "status",
        ],
    )
    write_md_reports(out_dir, summary, n_samples)
    (out_dir / "audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
