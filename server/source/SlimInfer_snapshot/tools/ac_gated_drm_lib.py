#!/usr/bin/env python3
"""AC-Gated DRM-Add: utility maps, FUP model load, feature CSV helpers."""
from __future__ import annotations

import csv
import math
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FUP_DIR = REPO_ROOT / "results/fup_guarded_swap_stage2_64_20260601_051737"

FEATURE_COLS = [
    "feature_drm_delta", "feature_g_diffusion", "feature_r_active", "feature_query_similarity",
    "feature_query_covered_count", "feature_coverage_gain_proxy", "feature_answer_overlap",
    "feature_page_position", "feature_page_token_len", "feature_attention_score",
]
SOURCES = ["fixed", "overlap", "neighbor", "hierarchy", "evidence_centered", "fusion_fine", "fusion_fine_no_query"]

AC_WEIGHTS = {"w_drm": 1.0, "w_fup": 0.5, "w_novelty": 0.35, "w_gap": 0.35, "w_redundancy": 0.30, "w_noise": 0.25}


def zscore(vals: List[float]) -> List[float]:
    if not vals:
        return []
    if len(vals) == 1:
        return [0.0]
    m = sum(vals) / len(vals)
    sd = (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5
    if sd <= 1e-9:
        return [0.0] * len(vals)
    return [(v - m) / sd for v in vals]


def ref_layer_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not rows:
        return []
    if "layer_id" not in rows[0]:
        return rows
    ref = min(int(r.get("layer_id", 0)) for r in rows)
    return [r for r in rows if int(r.get("layer_id", 0)) == ref]


def load_features_csv(path: Path) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    by_s: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            by_s[(str(row["task"]), str(row["sample_id"]))].append(row)
    return {k: ref_layer_rows(v) for k, v in by_s.items()}


def _feat_vec(r: Dict[str, Any], task: str, tasks: List[str]) -> List[float]:
    v = [float(r.get(c, 0.0)) for c in FEATURE_COLS]
    v += [1.0 if r.get("candidate_source") == s else 0.0 for s in SOURCES]
    v += [1.0 if task == t else 0.0 for t in tasks]
    return v


def load_fup_keepdrop_model(fup_dir: Path):
    pkl = fup_dir / "models" / "fup_oracle_keepdrop_model.pkl"
    if not pkl.exists():
        return None, None
    with pkl.open("rb") as fh:
        blob = pickle.load(fh)
    return blob.get("model"), blob


def predict_fup_maps(
    feat_by_sample: Dict[Tuple[str, str], List[Dict[str, Any]]],
    tasks: List[str],
    model,
    blob: Optional[Dict[str, Any]] = None,
) -> Dict[Tuple[str, str], Dict[int, float]]:
    out: Dict[Tuple[str, str], Dict[int, float]] = defaultdict(dict)
    if model is None:
        for key, rows in feat_by_sample.items():
            for r in rows:
                if int(r.get("is_candidate", 1)) == 1:
                    out[key][int(r["page_id"])] = float(r.get("feature_drm_delta", 0.0))
        return out
    fcols = (blob or {}).get("feature_cols", FEATURE_COLS)
    srcs = (blob or {}).get("sources", SOURCES)
    tlist = (blob or {}).get("tasks", tasks)

    def vec(r: Dict[str, Any], task: str) -> List[float]:
        v = [float(r.get(c, 0.0)) for c in fcols]
        v += [1.0 if r.get("candidate_source") == s else 0.0 for s in srcs]
        v += [1.0 if task == t else 0.0 for t in tlist]
        return v

    keys, feats = [], []
    for (task, sid), rows in feat_by_sample.items():
        for r in rows:
            if int(r.get("is_candidate", 0)) != 1:
                continue
            keys.append((task, sid, int(r["page_id"])))
            feats.append(vec(r, task))
    if not feats:
        return out
    X = np.array(feats, dtype=float)
    prob = model.predict_proba(X)[:, 1]
    for (task, sid, pid), p in zip(keys, prob.tolist()):
        out[(task, sid)][pid] = float(p)
    return out


def build_ac_utility_maps(
    feat_by_sample: Dict[Tuple[str, str], List[Dict[str, Any]]],
    fup_maps: Dict[Tuple[str, str], Dict[int, float]],
    weights: Optional[Dict[str, float]] = None,
) -> Dict[Tuple[str, str], Dict[int, float]]:
    w = {**AC_WEIGHTS, **(weights or {})}
    ac: Dict[Tuple[str, str], Dict[int, float]] = defaultdict(dict)
    for (task, sid), rows in feat_by_sample.items():
        cands = [r for r in rows if int(r.get("is_candidate", 0)) == 1]
        if not cands:
            continue
        drm = [float(r.get("feature_drm_delta", 0.0)) for r in cands]
        attn = [float(r.get("feature_attention_score", 0.0)) for r in cands]
        red_a = [float(r.get("feature_r_active", 0.0)) for r in cands]
        z_drm = zscore(drm)
        tail_attn = min(attn) if attn else 0.0
        fup_m = fup_maps.get((task, sid), {})
        for r, d, ra, za in zip(cands, z_drm, red_a, zscore(red_a)):
            pid = int(r["page_id"])
            fup_sc = float(fup_m.get(pid, 0.0))
            qsim = float(r.get("feature_query_similarity", 0.0))
            novelty = max(0.0, 1.0 - ra)
            gap = d - tail_attn
            noise = max(0.0, d) * (1.0 - min(1.0, qsim))
            src = str(r.get("candidate_source", ""))
            if src in ("fusion_fine_no_query",) and qsim < 0.05:
                noise += 0.15
            if task in ("passage_retrieval_en", "passage_count") and qsim < 0.02:
                noise += 0.10
            delta = (w["w_drm"] * d + w["w_fup"] * fup_sc + w["w_novelty"] * novelty
                     + w["w_gap"] * gap - w["w_redundancy"] * ra - w["w_noise"] * noise)
            ac[(task, sid)][pid] = float(delta)
    return ac


def candidate_pool_stats_rows(feat_by_sample: Dict[Tuple[str, str], List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    per_task: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for (task, _sid), rows in feat_by_sample.items():
        cands = [r for r in rows if int(r.get("is_candidate", 0)) == 1]
        per_task[task].extend(cands)
    rows_out = []
    all_cands = []
    for task, cands in per_task.items():
        all_cands.extend(cands)
        src_ct: Dict[str, int] = defaultdict(int)
        costs = []
        for r in cands:
            src_ct[str(r.get("candidate_source", ""))] += 1
            costs.append(int(r.get("token_cost", 64) or 64))
        n = len(cands)
        rows_out.append({
            "scope": task, "n_candidate_rows": n,
            "avg_candidates_per_sample": round(n / max(1, len({sid for t, sid in feat_by_sample if t == task})), 2),
            "avg_token_cost": round(sum(costs) / max(1, len(costs)), 2),
            "source_distribution": ";".join(f"{k}:{v}" for k, v in sorted(src_ct.items())),
        })
    src_all: Dict[str, int] = defaultdict(int)
    costs_all = []
    for r in all_cands:
        src_all[str(r.get("candidate_source", ""))] += 1
        costs_all.append(int(r.get("token_cost", 64) or 64))
    rows_out.insert(0, {
        "scope": "ALL",
        "n_candidate_rows": len(all_cands),
        "avg_candidates_per_sample": round(len(all_cands) / max(1, len(feat_by_sample)), 2),
        "avg_token_cost": round(sum(costs_all) / max(1, len(costs_all)), 2),
        "source_distribution": ";".join(f"{k}:{v}" for k, v in sorted(src_all.items())),
    })
    return rows_out
