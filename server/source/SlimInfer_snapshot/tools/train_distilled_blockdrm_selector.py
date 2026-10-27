#!/usr/bin/env python3
"""Train/calibrate distilled BlockDRM microblock selector from teacher labels."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

FEATURE_NAMES = [
    "drm_score", "rh_score", "bridge_score", "redundancy_score",
    "page_position", "relative_position_to_query", "microblock_local_rank",
    "teacher_overlap_ratio",
]


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _deterministic_weights(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    pos = [r for r in rows if int(r.get("teacher_positive", 0))]
    neg = [r for r in rows if not int(r.get("teacher_positive", 0))]
    def avg(key: str, data: List[Dict[str, Any]], default: float = 0.0) -> float:
        vals = [float(r.get(key, default) or default) for r in data]
        return sum(vals) / max(1, len(vals))
    weights = {
        "drm_score": 1.0,
        "rh_score": 0.35,
        "bridge_score": 0.0,
        "redundancy_score": -0.25,
        "page_position": 0.05,
        "relative_position_to_query": 0.0,
        "microblock_local_rank": 0.02,
        "teacher_overlap_ratio": 0.40,
    }
    if pos:
        weights["teacher_overlap_ratio"] = max(0.2, avg("teacher_overlap_ratio", pos) - avg("teacher_overlap_ratio", neg, 0.0))
    return weights


def train_selector(
    label_file: Path,
    feature_dump: Path,
    output: Path,
) -> Dict[str, Any]:
    rows = _load_jsonl(label_file)
    block_priors = []
    for row in rows:
        bid = int(row.get("page_id", -1))
        block_priors.append({
            "block_id": bid,
            "page_id": bid,
            "teacher_prior": float(row.get("teacher_overlap_ratio", 0.0) or 0.0),
            "teacher_positive": int(row.get("teacher_positive", 0)),
        })
    X = []
    y = []
    for row in rows:
        feat = [
            float(row.get("drm_score", 0.0) or 0.0),
            float(row.get("rh_score", 0.0) or 0.0),
            float(row.get("bridge_score", 0.0) or 0.0),
            float(row.get("redundancy_score", 0.0) or 0.0),
            float(row.get("page_id", 0)) / 100.0,
            float(row.get("relative_position_to_query", 0.0) or 0.0),
            float(row.get("teacher_rank", 0.0) or 0.0),
            float(row.get("teacher_overlap_ratio", 0.0) or 0.0),
        ]
        X.append(feat)
        y.append(int(row.get("teacher_positive", 0)))
    weights: Dict[str, float]
    backend = "deterministic_linear"
    if len(X) >= 8:
        try:
            from sklearn.linear_model import LogisticRegression
            import numpy as np
            model = LogisticRegression(max_iter=200, C=1.0)
            model.fit(np.array(X), np.array(y))
            coef = model.coef_.reshape(-1)
            weights = {FEATURE_NAMES[i]: float(coef[i]) for i in range(len(FEATURE_NAMES))}
            backend = "sklearn_logistic"
        except Exception:
            weights = _deterministic_weights(rows)
    else:
        weights = _deterministic_weights(rows)
    coverage = sum(int(r.get("teacher_positive", 0)) for r in rows) / max(1, len(rows))
    payload = {
        "backend": backend,
        "feature_weights": weights,
        "block_priors": block_priors[:500],
        "teacher_coverage": round(coverage, 4),
        "num_training_rows": len(rows),
        "feature_dump_used": str(feature_dump) if feature_dump.exists() else "",
        "weak_teacher": any(r.get("weak_teacher") for r in rows),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_label_file", required=True)
    parser.add_argument("--feature_dump", default="")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    meta = train_selector(
        Path(args.teacher_label_file),
        Path(args.feature_dump) if args.feature_dump else Path(""),
        Path(args.output),
    )
    print(json.dumps({"backend": meta["backend"], "output": args.output, "teacher_coverage": meta["teacher_coverage"]}, indent=2))


if __name__ == "__main__":
    main()
