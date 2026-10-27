#!/usr/bin/env python3
"""SP-SlimInfer Phase 1: CPU replay active-set audit from prefill dumps (no QA)."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Set, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bridgekv.sp_sliminfer import (  # noqa: E402
    FEATURE_SOURCES,
    PRUNE_LAYERS,
    active_set_churn,
    build_block_layer_index,
    compute_sp_features_for_block,
    replay_sp_active_set,
)
from tools.run_bridgeflow_silm_property_probe import read_jsonl  # noqa: E402

CHURN_MIN, CHURN_MAX = 0.03, 0.15
DEFAULT_FEATURE_DIR = (
    REPO_ROOT / "results/bridgeflow_silm_property_probe_2task_full_20260709_211600"
)
DEFAULT_UASR = DEFAULT_FEATURE_DIR / "uasr_score_dump.jsonl"


def parse_args() -> argparse.Namespace:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--feature_dir", type=Path, default=DEFAULT_FEATURE_DIR)
    p.add_argument("--uasr_dump", type=Path, default=DEFAULT_UASR)
    p.add_argument("--out_dir", type=Path, default=REPO_ROOT / f"results/sp_sliminfer_probe_{ts}")
    p.add_argument("--tasks", default="hotpotqa,musique")
    return p.parse_args()


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def build_feature_audit_rows(
    feature_rows: Sequence[Mapping[str, Any]],
    tasks: Set[str],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    layer_index = build_block_layer_index(feature_rows)
    audit_rows: List[Dict[str, Any]] = []
    samples_seen: Set[Tuple[str, str]] = set()
    status_counts: Dict[str, int] = defaultdict(int)

    grouped_blocks: Dict[Tuple[str, str, int], Set[int]] = defaultdict(set)
    for row in feature_rows:
        if str(row.get("task", "")) not in tasks:
            continue
        key = (str(row["task"]), str(row["sample_id"]), int(row["layer"]))
        grouped_blocks[key].add(int(row["block_id"]))

    for (task, sid, layer), blocks in sorted(grouped_blocks.items()):
        samples_seen.add((task, sid))
        for bid in sorted(blocks):
            feat = compute_sp_features_for_block(
                task=task,
                sample_id=sid,
                block_id=bid,
                layer=layer,
                layer_index=layer_index,
            )
            row = {
                "task": task,
                "sample_id": sid,
                "layer": layer,
                "block_id": bid,
                **feat,
            }
            audit_rows.append(row)
            for fname in ("persistence", "absorption", "drift", "relay", "redundancy"):
                status_counts[feat.get(f"{fname}_status", "UNAVAILABLE")] += 1

    meta = {
        "unique_samples": len(samples_seen),
        "expected_samples": 400,
        "feature_row_count": len(audit_rows),
        "status_counts": dict(status_counts),
        "feature_sources": FEATURE_SOURCES,
    }
    return audit_rows, meta


def replay_active_set_audit(
    uasr_rows: Sequence[Mapping[str, Any]],
    feature_rows: Sequence[Mapping[str, Any]],
    tasks: Set[str],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    layer_index = build_block_layer_index(feature_rows)
    audit: List[Dict[str, Any]] = []
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in uasr_rows:
        if str(row.get("phase", "")) != "prefill":
            continue
        if str(row.get("task", "")) not in tasks:
            continue
        grouped[(str(row["task"]), str(row["sample_id"]))].append(dict(row))

    churns: List[float] = []
    changed_layers = 0
    changed_samples: Set[Tuple[str, str]] = set()
    over_budget = 0
    sink_violations = 0
    no_effect_rows = 0
    eligible = 0

    for (task, sid), rows in grouped.items():
        for row in sorted(rows, key=lambda r: int(r.get("layer_id", -1))):
            layer = int(row.get("layer_id", -1))
            block_ids = [int(x) for x in row.get("all_candidate_block_ids", [])]
            scores = [float(x) for x in row.get("all_candidate_scores", [])]
            if not block_ids or len(scores) != len(block_ids):
                continue
            budget = int(row.get("budget_k", 0) or 0)
            silm_selected = {int(x) for x in row.get("selected_block_ids", [])}
            if budget <= 0:
                budget = len(silm_selected)
            churn_eligible = int(budget < len(block_ids))
            if churn_eligible:
                eligible += 1

            must_keep: Set[int] = set()
            if block_ids:
                must_keep.add(block_ids[0])
                must_keep.add(block_ids[-1])

            sp_feat_map = {
                bid: compute_sp_features_for_block(
                    task=task,
                    sample_id=sid,
                    block_id=bid,
                    layer=layer,
                    layer_index=layer_index,
                )
                for bid in block_ids
            }
            final, meta = replay_sp_active_set(
                block_ids=block_ids,
                silm_scores=scores,
                silm_selected=silm_selected,
                sp_features=sp_feat_map,
                budget=budget,
                must_keep=must_keep,
            )
            churn = active_set_churn(silm_selected, final, budget)
            if churn_eligible:
                churns.append(churn)
            if final != silm_selected:
                changed_layers += 1
                changed_samples.add((task, sid))
            else:
                no_effect_rows += 1
            if len(final) > budget:
                over_budget += 1
            if block_ids and block_ids[0] not in final:
                sink_violations += 1

            audit.append(
                {
                    "task": task,
                    "sample_id": sid,
                    "layer": layer,
                    "variant": "sp_sliminfer",
                    "active_budget": budget,
                    "churn_eligible": churn_eligible,
                    "churn_rate": round(churn, 6),
                    "churn_in_target": int(CHURN_MIN <= churn <= CHURN_MAX) if churn_eligible else "",
                    "jaccard_vs_silm": meta.get("jaccard_vs_silm", ""),
                    "selected_block_count": len(final),
                    "over_budget": int(len(final) > budget),
                    "sink_kept": meta.get("sink_kept", ""),
                    "changed_vs_silm": int(final != silm_selected),
                }
            )

    mean_churn = sum(churns) / len(churns) if churns else 0.0
    summary = {
        "unique_samples": len(grouped),
        "eligible_layers": eligible,
        "mean_churn_eligible": round(mean_churn, 6),
        "churn_in_band": int(CHURN_MIN <= mean_churn <= CHURN_MAX) if churns else 0,
        "changed_layers": changed_layers,
        "changed_samples": len(changed_samples),
        "no_effect_rows": no_effect_rows,
        "over_budget": over_budget,
        "sink_violations": sink_violations,
        "no_effect_flag": int(changed_layers == 0),
    }
    return audit, summary


def write_reports(out_dir: Path, feat_meta: Dict[str, Any], act_summary: Dict[str, Any]) -> None:
    coverage_ok = feat_meta["unique_samples"] == 400
    churn_ok = act_summary.get("churn_in_band", 0) == 1 and act_summary.get("over_budget", 1) == 0
    qa_ready = coverage_ok and churn_ok and act_summary.get("no_effect_flag", 1) == 0

    (out_dir / "SP_FEATURE_AUDIT_REPORT.md").write_text(
        "\n".join(
            [
                "# SP Feature Audit Report",
                "",
                f"**Coverage:** {feat_meta['unique_samples']}/400 samples "
                f"({'PASS' if coverage_ok else 'FAIL'})",
                "",
                "## Feature availability",
                "",
                "| Feature | Source |",
                "|---------|--------|",
                *[f"| {k} | {v} |" for k, v in FEATURE_SOURCES.items()],
                "",
                "## Proxy status counts",
                f"{feat_meta.get('status_counts', {})}",
                "",
                "## Active-set replay (SP score adjustment)",
                f"- Mean churn (eligible): **{act_summary.get('mean_churn_eligible', 0)}** "
                f"(target {CHURN_MIN:.0%}–{CHURN_MAX:.0%})",
                f"- Changed layers: {act_summary.get('changed_layers', 0)}",
                f"- Changed samples: {act_summary.get('changed_samples', 0)}",
                f"- NO_EFFECT rows: {act_summary.get('no_effect_rows', 0)}",
                f"- Over budget: {act_summary.get('over_budget', 0)}",
                f"- Sink violations: {act_summary.get('sink_violations', 0)}",
                "",
                f"**QA ready:** {'YES' if qa_ready else 'NO — fix hook/weights first'}",
                "",
                "## Claim tier",
                "- SP-SlimInfer Phase 1: active-set audit only",
                "- RoleFlow: **no-go** (400-full failed)",
                "- Official repro: separate (57.43 / 27.14 on 400-full SILM)",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    (out_dir / "FEATURE_COVERAGE_AUDIT.md").write_text(
        f"# Feature Coverage Audit\n\n"
        f"- Samples: {feat_meta['unique_samples']}/400\n"
        f"- Feature audit rows: {feat_meta['feature_row_count']}\n"
        f"- Source dump: property_features from prefill probe\n"
        f"- Hidden-state trajectory: **UNAVAILABLE**\n",
        encoding="utf-8",
    )

    (out_dir / "ACTIVE_SET_AUDIT.md").write_text(
        (out_dir / "SP_FEATURE_AUDIT_REPORT.md").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    (out_dir / "OFFICIAL_PATH_AUDIT.md").write_text(
        "\n".join(
            [
                "# Official Path Audit — SP-SlimInfer Phase 1",
                "",
                "- Base scoring: `token_unit_qk` (from uasr dump, SILM prefill)",
                "- Selection: CPU replay of SP score adjustment + same budget K",
                "- No QA decode, no scorer, no gold",
                "- No BM25/LBP/packing",
                "- Forced sink: block 0 in must_keep",
                "- KV budget: unchanged vs SILM row `budget_k`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    (out_dir / "FAIRNESS_AUDIT.md").write_text(
        "\n".join(
            [
                "# Fairness Audit — SP-SlimInfer Phase 1",
                "",
                "- Same candidate blocks/scores as SILM uasr dump",
                "- Same budget per layer",
                "- SP only adjusts scores before top-k replay",
                "- No task-name routing, no gold",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    (out_dir / "CLAIM_GUARD.md").write_text(
        "\n".join(
            [
                "# Claim Guard",
                "",
                "| Tier | Status |",
                "|------|--------|",
                "| Official SILM repro (400 two-task) | 57.43 / 27.14 |",
                "| RoleFlow 400-full | **no-go** |",
                "| SP-SlimInfer Phase 1 | active-set audit only |",
                f"| SP QA candidate | {'pending audit pass' if qa_ready else 'blocked'} |",
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

    feature_rows = read_jsonl(args.feature_dir / "property_features.jsonl")
    uasr_rows = read_jsonl(args.uasr_dump)

    feat_audit, feat_meta = build_feature_audit_rows(feature_rows, tasks)
    _write_csv(
        out_dir / "sp_feature_audit.csv",
        feat_audit,
        [
            "task", "sample_id", "layer", "block_id",
            "semantic_persistence", "semantic_absorption", "semantic_drift",
            "semantic_relay", "active_set_redundancy",
            "persistence_status", "absorption_status", "drift_status",
            "relay_status", "redundancy_status",
            "hidden_state_trajectory", "key_state_representation_drift",
        ],
    )

    act_audit, act_summary = replay_active_set_audit(uasr_rows, feature_rows, tasks)
    _write_csv(
        out_dir / "sp_active_set_audit.csv",
        act_audit,
        [
            "task", "sample_id", "layer", "variant", "active_budget",
            "churn_eligible", "churn_rate", "churn_in_target", "jaccard_vs_silm",
            "selected_block_count", "over_budget", "sink_kept", "changed_vs_silm",
        ],
    )

    write_reports(out_dir, feat_meta, act_summary)
    (out_dir / "audit_summary.json").write_text(
        json.dumps({"feature": feat_meta, "active_set": act_summary}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"out_dir": str(out_dir), "feature": feat_meta, "active_set": act_summary}, indent=2))


if __name__ == "__main__":
    main()
