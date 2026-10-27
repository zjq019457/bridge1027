#!/usr/bin/env python3
"""Replay-only 2Wiki CC topology diagnosis (no live generation).

Usage:
  .venv/bin/python tools/diagnose_cc_2wiki_topology.py \
    --scores_by_sample results/.../scores_by_sample.csv \
    --cc_audit results/.../cc_selection_audit.jsonl \
    --out_dir results/.../cc_2wiki_diagnosis
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional


def _load_scores(path: Path) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            task = str(r.get("task") or r.get("dataset") or "")
            if task not in {"2wikimqa", "2wiki", "2WikiMQA"} and "2wiki" not in task.lower():
                # Keep all tasks if user passes mixed file; filter later.
                pass
            sid = str(r.get("sample_id") or r.get("sample_index") or "")
            key = f"{task}:{sid}"
            rows[key] = r
    return rows


def _mean(xs: List[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else float("nan")


def main() -> int:
    p = argparse.ArgumentParser(description="Diagnose CC 2Wiki hop-collapse from audits.")
    p.add_argument("--scores_by_sample", type=Path, required=True)
    p.add_argument("--cc_audit", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--task", type=str, default="2wikimqa")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    scores = _load_scores(args.scores_by_sample)
    # Prefer delta columns if present; else original/cc pair.
    win_keys: List[str] = []
    loss_keys: List[str] = []
    for key, r in scores.items():
        task = str(r.get("task") or r.get("dataset") or key.split(":", 1)[0])
        if task != args.task and args.task not in task:
            continue
        delta = r.get("delta")
        if delta in (None, ""):
            try:
                o = float(r.get("sliminfer_original") or r.get("score_original") or "nan")
                c = float(r.get("cc_sliminfer_v1") or r.get("score_cc") or "nan")
                delta_v = c - o
            except Exception:
                continue
        else:
            try:
                delta_v = float(delta)
            except Exception:
                continue
        if delta_v > 1e-9:
            win_keys.append(key)
        elif delta_v < -1e-9:
            loss_keys.append(key)

    audits_by_sample: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    with args.cc_audit.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            task = str(row.get("task") or "")
            if task != args.task and args.task not in task:
                continue
            sid = str(row.get("sample_id") or row.get("sample_index") or "")
            audits_by_sample[f"{task}:{sid}"].append(row)

    def summarize(keys: List[str]) -> Dict[str, Any]:
        feat_lists: Dict[str, List[float]] = defaultdict(list)
        multi_locus = 0
        second_hop_evict = 0
        n = 0
        for key in keys:
            rows = audits_by_sample.get(key) or []
            if not rows:
                continue
            n += 1
            # aggregate over layers: mean features if present
            for row in rows:
                feats = row.get("features") if isinstance(row.get("features"), dict) else {}
                for fk, fv in feats.items():
                    try:
                        feat_lists[fk].append(float(fv))
                    except Exception:
                        pass
                if float(feats.get("n_clusters", 0) or 0) >= 2 and float(
                    feats.get("cluster_gap_ratio", 0) or 0
                ) >= 0.08:
                    multi_locus += 1
                    break
            # heuristic: removed ids while n_clusters>=2
            for row in rows:
                feats = row.get("features") if isinstance(row.get("features"), dict) else {}
                removed = row.get("removed_block_ids") or []
                if (
                    float(feats.get("n_clusters", 0) or 0) >= 2
                    and isinstance(removed, list)
                    and len(removed) > 0
                ):
                    second_hop_evict += 1
                    break
        return {
            "n_samples_with_audit": n,
            "multi_locus_rate": round(multi_locus / max(1, n), 6),
            "second_hop_evict_rate": round(second_hop_evict / max(1, n), 6),
            "feature_means": {k: round(_mean(v), 6) for k, v in sorted(feat_lists.items())},
        }

    report = {
        "task": args.task,
        "n_win": len(win_keys),
        "n_loss": len(loss_keys),
        "win": summarize(win_keys),
        "loss": summarize(loss_keys),
        "hypothesis": (
            "If loss.second_hop_evict_rate >> win and loss.multi_locus_rate high, "
            "prefer cc_diversify_v0 over always-on CC."
        ),
    }
    (args.out_dir / "CC_2WIKI_DIAGNOSIS.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# CC 2Wiki Topology Diagnosis (replay)\n\n",
        f"- task: `{args.task}`\n",
        f"- wins: {report['n_win']} / losses: {report['n_loss']}\n\n",
        "## Loss vs Win\n\n",
        f"- loss multi_locus_rate: {report['loss']['multi_locus_rate']}\n",
        f"- win multi_locus_rate: {report['win']['multi_locus_rate']}\n",
        f"- loss second_hop_evict_rate: {report['loss']['second_hop_evict_rate']}\n",
        f"- win second_hop_evict_rate: {report['win']['second_hop_evict_rate']}\n\n",
        f"{report['hypothesis']}\n",
    ]
    (args.out_dir / "CC_2WIKI_DIAGNOSIS.md").write_text("".join(lines), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
