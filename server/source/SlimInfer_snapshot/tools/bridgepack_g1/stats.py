#!/usr/bin/env python3
"""Bootstrap / Holm helpers for BridgePack G1 analysis."""
from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from longbench_eval import dataset2metric

BOOTSTRAP_N = 10000
BOOTSTRAP_SEED = 20260727


def score_prediction_row(task: str, row: Mapping[str, Any]) -> float:
    metric = dataset2metric[task]
    pred = str(row.get("pred", ""))
    answers = row.get("answers") or []
    if not isinstance(answers, list):
        answers = [answers]
    best = 0.0
    for a in answers:
        best = max(best, float(metric(pred, a, all_classes=row.get("all_classes"))))
    return 100.0 * best


def load_pred_scores(pred_root, variant: str, tasks: Sequence[str]) -> Dict[Tuple[str, str], float]:
    from tools.bridgepack_g1.common import read_jsonl

    out: Dict[Tuple[str, str], float] = {}
    for task in tasks:
        for row in read_jsonl(pred_root / variant / f"{task}.jsonl"):
            out[(task, str(row["sample_id"]))] = score_prediction_row(task, row)
    return out


def paired_bootstrap_delta(
    scores_a: Mapping[Tuple[str, str], float],
    scores_b: Mapping[Tuple[str, str], float],
    *,
    n_boot: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
) -> Dict[str, Any]:
    keys = sorted(set(scores_a) & set(scores_b))
    if not keys:
        return {
            "n": 0,
            "delta_macro": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
            "crosses_zero": True,
            "boot_std": float("nan"),
            "task_deltas": {},
        }
    sa = np.array([scores_a[k] for k in keys], dtype=np.float64)
    sb = np.array([scores_b[k] for k in keys], dtype=np.float64)
    strata = np.array([k[0] for k in keys])
    unique = sorted(set(strata.tolist()))
    idx = {s: np.where(strata == s)[0] for s in unique}
    rng = np.random.default_rng(seed)
    delta = float(np.mean(sa - sb))
    boots = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        chosen = np.concatenate([rng.choice(idx[s], size=len(idx[s]), replace=True) for s in unique])
        boots[i] = float(np.mean(sa[chosen] - sb[chosen]))
    lo, hi = np.quantile(boots, [0.025, 0.975])
    task_deltas = {}
    for t in unique:
        mask = [k for k in keys if k[0] == t]
        if mask:
            task_deltas[t] = float(np.mean([scores_a[k] - scores_b[k] for k in mask]))
    return {
        "n": len(keys),
        "delta_macro": delta,
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "crosses_zero": bool(lo <= 0 <= hi),
        "boot_std": float(np.std(boots)),
        "task_deltas": task_deltas,
    }


def approx_p_from_bootstrap(delta: float, boot_std: float) -> float:
    if boot_std <= 1e-12 or math.isnan(delta) or math.isnan(boot_std):
        return 1.0
    z = abs(delta) / boot_std
    return float(math.erfc(z / math.sqrt(2.0)))


def holm_correction(pvals: Sequence[Tuple[str, float]]) -> List[Dict[str, Any]]:
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i][1])
    out: List[Dict[str, Any]] = [None] * m  # type: ignore
    for rank, idx in enumerate(order):
        label, p = pvals[idx]
        adj = min(1.0, (m - rank) * p)
        if rank > 0:
            adj = max(adj, out[order[rank - 1]]["p_holm"])
        out[idx] = {"label": label, "p_raw": p, "p_holm": adj, "reject_holm_0.05": adj < 0.05}
    return out
