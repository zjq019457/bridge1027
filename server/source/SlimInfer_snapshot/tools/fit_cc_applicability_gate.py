#!/usr/bin/env python3
"""Offline threshold fit for CC applicability gate (no live generation).

Fits simple thresholds on audit JSONL by correlating enable decisions with
already-computed per-sample deltas (analysis only). Writes a frozen gate JSON.

Usage:
  .venv/bin/python tools/fit_cc_applicability_gate.py \
    --cc_audit path/to/cc_selection_audit.jsonl \
    --scores_by_sample path/to/scores_by_sample.csv \
    --out configs/cc_gate_v0_fitted.json
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from bridgekv.cc_topology_policies import DEFAULT_CC_GATE_CONFIG, cc_applicable, classify_spike_vs_cluster


def _sample_delta(path: Path) -> Dict[Tuple[str, str], float]:
    out: Dict[Tuple[str, str], float] = {}
    with path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            task = str(r.get("task") or r.get("dataset") or "")
            sid = str(r.get("sample_id") or r.get("sample_index") or "")
            if r.get("delta") not in (None, ""):
                out[(task, sid)] = float(r["delta"])
                continue
            try:
                o = float(r.get("sliminfer_original") or "nan")
                c = float(r.get("cc_sliminfer_v1") or r.get("cc_sliminfer") or "nan")
                out[(task, sid)] = c - o
            except Exception:
                continue
    return out


def _agg_features(audit_path: Path) -> Dict[Tuple[str, str], Dict[str, float]]:
    """Mean features across prune layers per sample."""
    buckets: Dict[Tuple[str, str], List[Dict[str, float]]] = defaultdict(list)
    with audit_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            feats = row.get("features")
            if not isinstance(feats, dict):
                continue
            key = (str(row.get("task") or ""), str(row.get("sample_id") or row.get("sample_index") or ""))
            clean = {}
            for k, v in feats.items():
                try:
                    clean[str(k)] = float(v)
                except Exception:
                    pass
            if clean:
                buckets[key].append(clean)
    out: Dict[Tuple[str, str], Dict[str, float]] = {}
    for key, rows in buckets.items():
        keys = set().union(*(r.keys() for r in rows))
        out[key] = {k: sum(r.get(k, 0.0) for r in rows) / len(rows) for k in keys}
    return out


def _score_gate(features_by_sample, deltas, gate_cfg) -> float:
    """Higher is better: reward enabling when delta>0, disabling when delta<0."""
    score = 0.0
    n = 0
    for key, feats in features_by_sample.items():
        if key not in deltas:
            continue
        d = deltas[key]
        pattern = classify_spike_vs_cluster(feats, gate_cfg)
        applicable, _ = cc_applicable(feats, gate_cfg)
        enable = bool(applicable and pattern != "spike")
        n += 1
        if enable and d > 0:
            score += 1.0
        elif (not enable) and d < 0:
            score += 1.0
        elif enable and d < 0:
            score -= 1.0
        elif (not enable) and d > 0:
            score -= 0.5
    return score / max(1, n)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cc_audit", type=Path, required=True)
    p.add_argument("--scores_by_sample", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    deltas = _sample_delta(args.scores_by_sample)
    feats = _agg_features(args.cc_audit)
    if not feats:
        raise SystemExit(
            "No features in audit. Re-run with topology methods (cc_gated_v0/...) "
            "or ensure audit JSONL contains `features`."
        )

    base = dict(DEFAULT_CC_GATE_CONFIG)
    best_cfg = dict(base)
    best_score = _score_gate(feats, deltas, best_cfg)

    # Tiny grid around defaults (no gold in features; deltas only for offline selection).
    grid = {
        "tau_nei": [0.25, 0.35, 0.45],
        "tau_spike": [0.35, 0.45, 0.55],
        "tau_cont": [0.10, 0.15, 0.25],
        "tau_str": [0.15, 0.25, 0.35],
        "tau_margin": [0.10, 0.20, 0.30],
    }
    from itertools import product

    keys = list(grid.keys())
    for values in product(*(grid[k] for k in keys)):
        cfg = dict(base)
        for k, v in zip(keys, values):
            cfg[k] = float(v)
        s = _score_gate(feats, deltas, cfg)
        if s > best_score:
            best_score = s
            best_cfg = cfg

    best_cfg["fit_score"] = round(best_score, 6)
    best_cfg["notes"] = "Fitted offline from CC audits + existing score deltas; runtime remains gold-free."
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(best_cfg, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "fit_score": best_score, "cfg": best_cfg}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
