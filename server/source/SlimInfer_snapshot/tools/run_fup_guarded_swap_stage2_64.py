#!/usr/bin/env python3
"""FUP + Guarded-Swap 64-sample diagnostic orchestrator.

Direction 1 (Functional Utility Predictor): learn / estimate per-candidate
marginal functional utility instead of hand-written selectors. Variants:
  - fup_add_gain          (learned_pseudo_label: first-token gold NLL gain, cross-fit ranker)
  - fup_oracle_keepdrop   (learned_pseudo_label: greedy-oracle membership, cross-fit classifier)
  - fup_output_delta      (deployable proxy: attention/diffusion/redundancy, no GT)
  - *_oracle_diag / oracle_upper_bound (DIAGNOSTIC-ONLY, use ground-truth labels)

Direction 2 (Guarded Swap): swap low-utility active TAIL pages for higher-utility
candidates (budget-neutral, active CORE protected). DRM + FUP utilities + sweep.

Everything is Add-only on top of the FIXED SlimInfer active set except guarded
swap, which only touches the protected tail. 64-sample diagnostic only.

Label metric: first-token gold-answer NLL via generate(max_new_tokens=1) -- cheap
and robust. fixed_drm_add is the unified in-experiment baseline.
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import math
import os
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.fup_guarded_swap_lib import FupHarness, SampleCtx  # noqa: E402

DEFAULT_MODEL = "/home/yu-bai/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
DEFAULT_LB = "/home/yu-bai/Desktop/jiaqizhang/HeadKV/data/LongBench"
# Fallback if HeadKV path missing: /diskE/datasets/LongBench
TASKS = ["musique", "hotpotqa", "qasper", "passage_count", "passage_retrieval_en"]

FEATURE_COLS = [
    "feature_drm_score", "feature_drm_delta", "feature_g_diffusion", "feature_r_active",
    "feature_query_similarity", "feature_query_covered_count", "feature_coverage_gain_proxy",
    "feature_evidence_overlap", "feature_answer_overlap", "feature_page_position",
    "feature_page_token_len", "feature_attention_score", "feature_active_tail_margin",
    "feature_damage_score",
]
SOURCES = ["fixed", "overlap", "neighbor", "hierarchy", "evidence_centered", "fusion_fine", "fusion_fine_no_query"]

GATE_TASK_DROP = 3.0
GATE_BUDGET_ERR = 0.01


# --------------------------------------------------------------------------- io
def write_csv(path: Path, rows: List[Dict[str, Any]], header: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    header = header or list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in header})


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl_index(path: Path, key: str = "sample_id") -> Dict[str, Dict[str, Any]]:
    idx: Dict[str, Dict[str, Any]] = {}
    for r in read_jsonl(path):
        idx[str(r.get(key, ""))] = r
    return idx


def upsert_jsonl(path: Path, row: Dict[str, Any], key: str = "sample_id") -> None:
    """Replace row with same key or append (avoids duplicate preds on retry)."""
    idx = read_jsonl_index(path, key=key) if path.exists() else {}
    idx[str(row.get(key, ""))] = row
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for sid in sorted(idx.keys()):
            f.write(json.dumps(idx[sid], ensure_ascii=False) + "\n")


def zscore(vals: List[float]) -> List[float]:
    import statistics as st
    if not vals:
        return []
    if len(vals) == 1:
        return [0.0]
    m = sum(vals) / len(vals)
    sd = (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5
    if sd <= 1e-9:
        return [0.0 for _ in vals]
    return [(v - m) / sd for v in vals]


# ---------------------------------------------------------------- ref features
def ref_layer_rows(feat_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """For one sample's dumped rows, keep only the reference (min) prune layer."""
    if not feat_rows:
        return []
    ref = min(int(r["layer_id"]) for r in feat_rows)
    return [r for r in feat_rows if int(r["layer_id"]) == ref]


class Driver:
    def __init__(self, args):
        self.args = args
        self._target_tasks = None
        if getattr(args, "target_task", ""):
            self._target_tasks = {t.strip() for t in str(args.target_task).split(",") if t.strip()}
        self.out = Path(args.output_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "preds").mkdir(exist_ok=True)
        (self.out / "models").mkdir(exist_ok=True)
        self.harness: Optional[FupHarness] = None
        # per (task, sample_id) state
        self.ctx_cache: Dict[Tuple[str, Any], SampleCtx] = {}
        self.sample_order: Dict[str, List[Any]] = {}        # task -> [sample_id]
        self.feat_by_sample: Dict[Tuple[str, Any], List[Dict[str, Any]]] = {}
        self.answers: Dict[Tuple[str, Any], Any] = {}
        self.lengths: Dict[Tuple[str, Any], int] = {}
        self.all_classes: Dict[str, Any] = {}

    def log(self, *a):
        print("[fup]", *a, flush=True)

    def ensure_harness(self):
        if self.harness is None:
            self.log("loading model ...")
            self.harness = FupHarness(self.args.model_path, self.args.longbench_dir, str(self.out / "_hwork"),
                                      max_new_tokens=32)

    def get_ctx(self, task: str, sample_id: Any) -> SampleCtx:
        return self.ctx_cache[(task, sample_id)]

    def _sample_ids_for_task(self, task: str) -> List[Any]:
        ids = list(self.sample_order.get(task, []))
        tsid = getattr(self.args, "target_sample_id", "") or ""
        if tsid:
            ids = [x for x in ids if str(x) == str(tsid)]
        if getattr(self.args, "retry_failed_only", False):
            # caller may still pass explicit target_sample_id; otherwise scan preds externally
            pass
        return ids

    def _should_run_sample(self, task: str, sid: Any, prev: Optional[Dict[str, Any]]) -> bool:
        tsid = getattr(self.args, "target_sample_id", "") or ""
        if tsid and str(sid) != str(tsid):
            return False
        if not getattr(self.args, "retry_failed_only", False) and not tsid:
            return False
        ok = prev is not None and str(prev.get("status", "success")) == "success" and prev.get("pred") not in (None, "")
        if ok and not self.args.overwrite_failed:
            return False
        return prev is None or str(prev.get("status", "")) != "success" or prev.get("pred") in (None, "")

    # ----------------------------------------------------- phase 0: extraction
    def phase_extract(self):
        self.ensure_harness()
        feat_path = self.out / "_feature_dump.jsonl"
        done_marker = self.out / "_extract_done.json"
        if done_marker.exists() and self.args.skip_existing:
            meta = json.loads(done_marker.read_text())
            self.sample_order = {k: v for k, v in meta["sample_order"].items()}
            # rebuild ctxs (needed for later phases) but reuse dumped features
            self._rebuild_after_extract(feat_path)
            self._write_features_csv()
            self.log("extract: resumed from cache")
            return
        if feat_path.exists():
            feat_path.unlink()
        for task in self.args.tasks:
            samples = self.harness.load_samples(task, self.args.samples_per_task)
            ids = []
            for i, s in enumerate(samples):
                try:
                    ctx = self.harness.build_sample_ctx(task, s, i)
                    self.harness.eval_pageset(ctx, {"_method": "extract", "feature_dump_path": str(feat_path)},
                                              mode="gen", gen_tokens=1)
                except Exception as exc:
                    self.log(f"extract FAIL {task}/{i}: {exc}; skipping sample")
                    continue
                self.ctx_cache[(task, ctx.sample_id)] = ctx
                self.answers[(task, ctx.sample_id)] = ctx.answers
                self.lengths[(task, ctx.sample_id)] = ctx.context_length
                self.all_classes[task] = ctx.all_classes
                ids.append(ctx.sample_id)
            self.sample_order[task] = ids
            self.log(f"extract {task}: {len(ids)} samples")
        # index features
        all_rows = read_jsonl(feat_path)
        by_s: Dict[Tuple[str, Any], List[Dict[str, Any]]] = defaultdict(list)
        for r in all_rows:
            by_s[(str(r["task"]), str(r["sample_id"]))].append(r)
        for (t, sid), rows in by_s.items():
            self.feat_by_sample[(t, sid)] = ref_layer_rows(rows)
        # canonical features csv
        self._write_features_csv()
        done_marker.write_text(json.dumps({"sample_order": {k: [str(x) for x in v] for k, v in self.sample_order.items()}}))

    def _rebuild_after_extract(self, feat_path: Path):
        for task in self.args.tasks:
            samples = self.harness.load_samples(task, self.args.samples_per_task)
            ids = []
            for i, s in enumerate(samples):
                ctx = self.harness.build_sample_ctx(task, s, i)
                self.ctx_cache[(task, ctx.sample_id)] = ctx
                self.answers[(task, ctx.sample_id)] = ctx.answers
                self.lengths[(task, ctx.sample_id)] = ctx.context_length
                self.all_classes[task] = ctx.all_classes
                ids.append(ctx.sample_id)
            self.sample_order[task] = ids
        all_rows = read_jsonl(feat_path)
        by_s: Dict[Tuple[str, Any], List[Dict[str, Any]]] = defaultdict(list)
        for r in all_rows:
            by_s[(str(r["task"]), str(r["sample_id"]))].append(r)
        for (t, sid), rows in by_s.items():
            self.feat_by_sample[(t, sid)] = ref_layer_rows(rows)

    def _write_features_csv(self):
        rows = []
        for task in self.args.tasks:
            for sid in self.sample_order.get(task, []):
                sample_rows = self.feat_by_sample.get((task, str(sid)), [])
                attn = [float(r.get("feature_attention_score", 0.0)) for r in sample_rows]
                attn_min = min(attn) if attn else 0.0
                for r in sample_rows:
                    drm = float(r.get("feature_drm_delta", 0.0))
                    red = float(r.get("feature_r_active", 0.0))
                    att = float(r.get("feature_attention_score", 0.0))
                    ev = float(r.get("feature_answer_overlap", r.get("feature_evidence_overlap", 0.0)))
                    rows.append({
                        "task": r["task"], "sample_id": r["sample_id"], "page_id": r["page_id"],
                        "candidate_source": r["candidate_source"], "token_cost": r["token_cost"],
                        "is_in_active": r["is_in_active"], "is_candidate": r["is_candidate"],
                        "feature_drm_score": drm,
                        "feature_drm_delta": drm,
                        "feature_g_diffusion": float(r.get("feature_g_diffusion", 0.0)),
                        "feature_r_active": red,
                        "feature_active_redundancy": red,
                        "feature_query_similarity": float(r.get("feature_query_similarity", 0.0)),
                        "feature_query_covered_count": int(r.get("feature_query_covered_count", 0) or 0),
                        "feature_coverage_gain_proxy": float(r.get("feature_coverage_gain_proxy", 0.0)),
                        "feature_evidence_overlap": ev,
                        "feature_answer_overlap": ev,
                        "feature_page_position": float(r.get("feature_page_position", 0.0)),
                        "feature_page_token_len": int(r.get("feature_page_token_len", 0) or 0),
                        "feature_attention_score": att,
                        "feature_active_tail_margin": round(att - attn_min, 6),
                        "feature_damage_score": drm,
                    })
        write_csv(self.out / "features_fup_candidates.csv", rows,
                  header=["task", "sample_id", "page_id", "candidate_source", "token_cost",
                          "is_in_active", "is_candidate"] + FEATURE_COLS)
        self.log(f"features_fup_candidates.csv: {len(rows)} rows")

    # candidates for a sample (is_candidate rows), sorted by drm desc
    def sample_candidates(self, task: str, sid: Any) -> List[Dict[str, Any]]:
        rows = [r for r in self.feat_by_sample.get((task, str(sid)), []) if int(r["is_candidate"]) == 1]
        rows.sort(key=lambda r: float(r.get("feature_drm_delta", 0.0)), reverse=True)
        return rows

    def sample_m_pages(self, task: str, sid: Any) -> int:
        rows = self.feat_by_sample.get((task, str(sid)), [])
        return max(1, int(rows[0].get("m_default", 3))) if rows else 3

    # ----------------------------------------------------- phase 1: labels A/B
    def phase_labels(self):
        self.ensure_harness()
        lab_path = self.out / "_labels.jsonl"
        done = {(r["task"], str(r["sample_id"])) for r in read_jsonl(lab_path)} if self.args.skip_existing else set()
        if not self.args.skip_existing and lab_path.exists():
            lab_path.unlink()
        topn = self.args.label_topn
        ub_rows = []
        for task in self.args.tasks:
            if self._target_tasks and task not in self._target_tasks:
                continue
            for sid in self._sample_ids_for_task(task):
                if (task, str(sid)) in done and not self.args.overwrite_failed:
                    continue
                ctx = self.get_ctx(task, sid)
                cands = self.sample_candidates(task, sid)[:topn]
                if not cands:
                    append_jsonl(lab_path, {"task": task, "sample_id": str(sid), "candidates": [], "oracle_set": []})
                    continue
                try:
                    base = self.harness.eval_pageset(ctx, {"_method": "lab_base", "forced_rescue_pages": []},
                                                     mode="ftnll")["first_token_nll"]
                except Exception as exc:
                    self.log(f"labels base FAIL {task}/{sid}: {exc}")
                    append_jsonl(lab_path, {"task": task, "sample_id": str(sid), "candidates": [], "oracle_set": []})
                    continue
                # label A: single-page gain
                single = {}
                for c in cands:
                    p = int(c["page_id"])
                    try:
                        nll = self.harness.eval_pageset(ctx, {"_method": "labA", "forced_rescue_pages": [p]},
                                                        mode="ftnll")["first_token_nll"]
                    except Exception:
                        nll = float("nan")
                    single[p] = (base - nll) if (nll == nll and base == base) else 0.0
                # greedy oracle (B): up to M steps
                M = min(self.args.oracle_steps, self.sample_m_pages(task, sid))
                cur: List[int] = []
                cur_base = base
                remaining = [int(c["page_id"]) for c in cands]
                step_gains = []
                for _ in range(max(1, M)):
                    best_p, best_nll, best_gain = None, None, -1e9
                    for p in remaining:
                        try:
                            nll = self.harness.eval_pageset(ctx, {"_method": "labB", "forced_rescue_pages": cur + [p]},
                                                            mode="ftnll")["first_token_nll"]
                        except Exception:
                            continue
                        if nll != nll:
                            continue
                        gain = cur_base - nll
                        if gain > best_gain:
                            best_gain, best_p, best_nll = gain, p, nll
                    if best_p is None or best_gain <= 0:
                        break
                    cur.append(best_p)
                    remaining.remove(best_p)
                    cur_base = best_nll
                    step_gains.append(best_gain)
                row = {
                    "task": task, "sample_id": str(sid), "base_nll": base,
                    "candidates": [{"page_id": int(c["page_id"]), "label_A": round(single.get(int(c["page_id"]), 0.0), 6),
                                    "in_oracle": 1 if int(c["page_id"]) in cur else 0} for c in cands],
                    "oracle_set": cur, "oracle_total_gain": round(float(base - cur_base), 6),
                }
                upsert_jsonl(lab_path, row)
                self.log(f"labels {task}/{sid}: |cand|={len(cands)} oracle={len(cur)} gain={row['oracle_total_gain']:.4f}")
        self._labels_index(lab_path)
        self._write_oracle_ub_summary()

    def _labels_index(self, lab_path: Path):
        self.labels = {}
        for r in read_jsonl(lab_path):
            self.labels[(r["task"], str(r["sample_id"]))] = r

    def labels_loaded(self) -> bool:
        return hasattr(self, "labels") and bool(self.labels)

    def _write_oracle_ub_summary(self):
        rows = []
        for task in self.args.tasks:
            gains = []
            for sid in self.sample_order.get(task, []):
                lab = self.labels.get((task, str(sid)))
                if lab:
                    gains.append(float(lab.get("oracle_total_gain", 0.0)))
            rows.append({"task": task, "oracle_gain_mean": round(sum(gains) / len(gains), 6) if gains else 0.0,
                         "n_samples": len(gains)})
        write_csv(self.out / "oracle_upper_bound_summary.csv", rows)

    # --------------------------------------------------- phase 2: train ranker
    def phase_train(self):
        if not self.labels_loaded():
            self._labels_index(self.out / "_labels.jsonl")
        from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
        import numpy as np
        # assemble training matrix over candidate rows that have labels
        feats, yA, yB, keys, groups, src = [], [], [], [], [], []
        sample_to_group: Dict[Tuple[str, str], int] = {}
        for task in self.args.tasks:
            for sid in self.sample_order.get(task, []):
                lab = self.labels.get((task, str(sid)))
                if not lab or not lab.get("candidates"):
                    continue
                gid = sample_to_group.setdefault((task, str(sid)), len(sample_to_group))
                labmap = {int(c["page_id"]): c for c in lab["candidates"]}
                for r in self.sample_candidates(task, sid):
                    p = int(r["page_id"])
                    if p not in labmap:
                        continue
                    feats.append(self._feat_vec(r, task))
                    yA.append(float(labmap[p]["label_A"]))
                    yB.append(int(labmap[p]["in_oracle"]))
                    keys.append((task, str(sid), p))
                    groups.append(gid)
                    src.append(r["candidate_source"])
        if not feats:
            self.log("train: no labeled candidates; skipping")
            self.pred_add_gain, self.pred_keepdrop = {}, {}
            return
        X = np.array(feats, dtype=float)
        yA = np.array(yA, dtype=float)
        yB = np.array(yB, dtype=int)
        groups = np.array(groups, dtype=int)
        ng = int(groups.max()) + 1
        nfold = min(4, ng)
        fold_of = {g: (i % nfold) for i, g in enumerate(sorted(set(groups.tolist())))}
        foldid = np.array([fold_of[g] for g in groups.tolist()])
        predA = np.zeros(len(yA)); predB = np.zeros(len(yB))
        for f in range(nfold):
            tr = foldid != f
            te = foldid == f
            if tr.sum() == 0 or te.sum() == 0:
                continue
            reg = HistGradientBoostingRegressor(max_iter=200, max_depth=3, learning_rate=0.08, random_state=0)
            reg.fit(X[tr], yA[tr])
            predA[te] = reg.predict(X[te])
            if len(set(yB[tr].tolist())) >= 2:
                clf = HistGradientBoostingClassifier(max_iter=200, max_depth=3, learning_rate=0.08, random_state=0)
                clf.fit(X[tr], yB[tr])
                predB[te] = clf.predict_proba(X[te])[:, 1]
            else:
                predB[te] = float(yB[tr].mean()) if tr.sum() else 0.0
        # refit-all models for pkl
        regA = HistGradientBoostingRegressor(max_iter=200, max_depth=3, learning_rate=0.08, random_state=0).fit(X, yA)
        with (self.out / "models" / "fup_add_gain_model.pkl").open("wb") as fh:
            pickle.dump({"model": regA, "feature_cols": FEATURE_COLS, "sources": SOURCES, "tasks": list(self.args.tasks)}, fh)
        if len(set(yB.tolist())) >= 2:
            clfB = HistGradientBoostingClassifier(max_iter=200, max_depth=3, learning_rate=0.08, random_state=0).fit(X, yB)
        else:
            clfB = None
        with (self.out / "models" / "fup_oracle_keepdrop_model.pkl").open("wb") as fh:
            pickle.dump({"model": clfB, "feature_cols": FEATURE_COLS, "sources": SOURCES, "tasks": list(self.args.tasks)}, fh)
        # out-of-fold utilities -> per-sample maps; also predict for ALL pages (incl active) for swap
        self.pred_add_gain = defaultdict(dict)
        self.pred_keepdrop = defaultdict(dict)
        for (task, sid, p), a, b in zip(keys, predA.tolist(), predB.tolist()):
            self.pred_add_gain[(task, sid)][p] = float(a)
            self.pred_keepdrop[(task, sid)][p] = float(b)
        # predict for active pages too (use refit-all model; acceptable for swap-side utility)
        for task in self.args.tasks:
            for sid in self.sample_order.get(task, []):
                rows = self.feat_by_sample.get((task, str(sid)), [])
                act = [r for r in rows if int(r["is_in_active"]) == 1]
                if not act:
                    continue
                Xa = np.array([self._feat_vec(r, task) for r in act], dtype=float)
                pa = regA.predict(Xa)
                pb = clfB.predict_proba(Xa)[:, 1] if clfB is not None else [0.0] * len(act)
                for r, a, b in zip(act, pa.tolist(), list(pb)):
                    self.pred_add_gain[(task, str(sid))][int(r["page_id"])] = float(a)
                    self.pred_keepdrop[(task, str(sid))][int(r["page_id"])] = float(b)
        # predictions csv
        pred_rows = []
        for (task, sid, p), a, b in zip(keys, predA.tolist(), predB.tolist()):
            lab = self.labels.get((task, sid), {})
            labmap = {int(c["page_id"]): c for c in lab.get("candidates", [])}
            pred_rows.append({"task": task, "sample_id": sid, "page_id": p,
                              "label_A": labmap.get(p, {}).get("label_A", ""), "pred_add_gain": round(a, 6),
                              "in_oracle": labmap.get(p, {}).get("in_oracle", ""), "pred_keepdrop": round(b, 6)})
        write_csv(self.out / "fup_add_gain_eval_predictions.csv", pred_rows)
        write_csv(self.out / "fup_oracle_keepdrop_predictions.csv", pred_rows)
        # train-labels csvs
        write_csv(self.out / "fup_add_gain_train_labels.csv",
                  [{"task": t, "sample_id": s, "page_id": p, "label_A": round(float(a), 6), "candidate_source": sc}
                   for (t, s, p), a, sc in zip(keys, yA.tolist(), src)])
        write_csv(self.out / "fup_oracle_keepdrop_labels.csv",
                  [{"task": t, "sample_id": s, "page_id": p, "in_oracle": int(b), "candidate_source": sc}
                   for (t, s, p), b, sc in zip(keys, yB.tolist(), src)])
        self.log(f"train: {len(yA)} labeled candidates, {ng} groups, {nfold} folds; positives={int(yB.sum())}")

    def _feat_vec(self, r: Dict[str, Any], task: str) -> List[float]:
        v = [float(r.get(c, 0.0)) for c in FEATURE_COLS]
        v += [1.0 if r.get("candidate_source") == s else 0.0 for s in SOURCES]
        v += [1.0 if task == t else 0.0 for t in self.args.tasks]
        return v

    # ----------------------------------------------- phase 3: C deployable proxy
    def build_output_delta_maps(self):
        self.pred_output_delta = defaultdict(dict)
        for task in self.args.tasks:
            for sid in self.sample_order.get(task, []):
                rows = self.feat_by_sample.get((task, str(sid)), [])
                if not rows:
                    continue
                za = zscore([float(r.get("feature_attention_score", 0.0)) for r in rows])
                zg = zscore([float(r.get("feature_g_diffusion", 0.0)) for r in rows])
                zr = zscore([float(r.get("feature_r_active", 0.0)) for r in rows])
                zq = zscore([float(r.get("feature_query_similarity", 0.0)) for r in rows])
                for r, a, g, rr, q in zip(rows, za, zg, zr, zq):
                    # deployable functional proxy: attention + diffusion novelty + query - redundancy
                    self.pred_output_delta[(task, str(sid))][int(r["page_id"])] = a + 0.5 * g + 0.5 * q - 0.5 * rr

    def oracle_label_maps(self):
        self.pred_oracle_A = defaultdict(dict)
        self.pred_oracle_B = defaultdict(dict)
        if not self.labels_loaded():
            self._labels_index(self.out / "_labels.jsonl")
        for (task, sid), lab in self.labels.items():
            for c in lab.get("candidates", []):
                self.pred_oracle_A[(task, sid)][int(c["page_id"])] = float(c["label_A"])
                self.pred_oracle_B[(task, sid)][int(c["page_id"])] = float(c["in_oracle"])

    # ----------------------------------------------- generic eval of a method
    def run_eval_method(self, name: str, override_fn, gen_tokens=None, force_rerun=False) -> Dict[str, Any]:
        self.ensure_harness()
        from tools.fup_guarded_swap_lib import release_cuda
        from longbench_eval import scorer
        result = {"method": name, "per_task": {}, "budget_warning": 0, "runner_fail": 0}
        for task in self.args.tasks:
            pred_path = self.out / "preds" / f"{name}__{task}.jsonl"
            existing = read_jsonl_index(pred_path) if not force_rerun else {}
            preds, answers, lengths = [], [], []
            audit_path = self.out / "preds" / f"{name}__{task}__audit.jsonl"
            task_in_scope = not self._target_tasks or task in self._target_tasks
            for sid in self.sample_order.get(task, []):
                prev = existing.get(str(sid))
                ok_prev = (
                    prev is not None
                    and str(prev.get("status", "success")) == "success"
                    and prev.get("pred") not in (None, "")
                )
                need_run = task_in_scope and (
                    force_rerun or not ok_prev or self._should_run_sample(task, sid, prev)
                )
                if need_run:
                    ctx = self.get_ctx(task, sid)
                    release_cuda()
                    ov = override_fn(task, sid)
                    ov["_method"] = name
                    ov["rescue_audit_path"] = str(audit_path)
                    try:
                        pr = self.harness.eval_pageset(ctx, ov, mode="gen", gen_tokens=gen_tokens)["pred"]
                        st = "success" if pr else "failed"
                    except Exception as exc:
                        pr = ""
                        st = "failed"
                        self.log(f"FAIL {name}/{task}/{sid}: {exc}")
                    existing[str(sid)] = {"sample_id": str(sid), "pred": pr, "status": st}
                pr = str(existing.get(str(sid), {}).get("pred", "") or "")
                preds.append(pr)
                answers.append(self.answers[(task, sid)])
                lengths.append(self.lengths[(task, sid)])
            with pred_path.open("w", encoding="utf-8") as f:
                for sid in self.sample_order.get(task, []):
                    row = existing.get(str(sid), {"sample_id": str(sid), "pred": "", "status": "failed"})
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            result["runner_fail"] += sum(
                1 for sid in self.sample_order.get(task, [])
                if not str(existing.get(str(sid), {}).get("pred", "") or "").strip()
                or existing.get(str(sid), {}).get("status") == "failed"
            )
            try:
                sc = scorer(task, preds, answers, lengths, self.all_classes.get(task, []))
            except Exception as exc:
                sc = 0.0
                self.log(f"score fail {name}/{task}: {exc}")
            result["per_task"][task] = float(sc)
        result["avg"] = sum(result["per_task"].values()) / max(1, len(result["per_task"]))
        # budget audit
        result.update(self._budget_audit(name))
        return result

    def _budget_audit(self, name: str) -> Dict[str, Any]:
        toks, ref = [], []
        warn = 0
        for task in self.args.tasks:
            audit = read_jsonl(self.out / "preds" / f"{name}__{task}__audit.jsonl")
            for r in audit:
                tc = r.get("selected_token_count")
                if tc:
                    toks.append(int(tc))
                if r.get("fallback_used"):
                    warn += 1
        return {"avg_selected_tokens": round(sum(toks) / len(toks), 1) if toks else 0.0,
                "budget_warning": int(warn)}

    # ----------------------------------------------- override builders
    def ov_baseline(self, task, sid):
        return {"block_strategy": "fixed", "rescue_candidate_strategy": "", "rescue_policy": "add",
                "rescue_score": "drm", "rescue_selector": "drm_mmr"}

    def ov_unified_noswap(self, task, sid):
        # shared Add backbone for FUP / guarded-swap methods (unified pool, DRM-MMR, no swap).
        # Isolates the pool effect from the FUP/swap effect.
        return {"block_strategy": "fixed_active_rescue", "rescue_candidate_strategy": "unified",
                "rescue_selector": "drm_mmr"}

    def _fup_ov(self, util_map):
        def fn(task, sid):
            return {"block_strategy": "fixed_active_rescue", "rescue_candidate_strategy": "unified",
                    "rescue_selector": "fup", "page_utility_map": util_map.get((task, str(sid)), {})}
        return fn

    def _forced_ov(self, set_map):
        def fn(task, sid):
            return {"block_strategy": "fixed_active_rescue", "rescue_candidate_strategy": "unified",
                    "forced_rescue_pages": list(set_map.get((task, str(sid)), []))}
        return fn

    def _swap_ov(self, util_src, util_map, core, swaps, tau):
        def fn(task, sid):
            ov = {"block_strategy": "fixed_active_rescue", "rescue_candidate_strategy": "unified",
                  "rescue_selector": "drm_mmr", "guarded_swap": 1, "swap_utility_source": util_src,
                  "active_core_ratio": core, "max_swaps": swaps, "swap_tau": tau}
            if util_src == "injected":
                ov["page_utility_map"] = util_map.get((task, str(sid)), {})
            return ov
        return fn


# --------------------------------------------------------------------------- main
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_fup", action="store_true")
    ap.add_argument("--run_guarded_swap", action="store_true")
    ap.add_argument("--stage", default="stage2_64")
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--samples_per_task", type=int, default=64)
    ap.add_argument("--skip_existing", action="store_true")
    ap.add_argument("--overwrite_failed", action="store_true")
    ap.add_argument("--retry_failed_only", action="store_true")
    ap.add_argument("--target_task", default="")
    ap.add_argument("--target_sample_id", default="")
    ap.add_argument("--recheck_only", action="store_true", help="Skip extract/labels/train; rescore from preds")
    ap.add_argument("--source_dir", default="", help="Copy preds/models from prior run for recheck")
    ap.add_argument("--output_dir", default="")
    ap.add_argument("--model_path", default=DEFAULT_MODEL)
    ap.add_argument("--longbench_dir", default=DEFAULT_LB)
    ap.add_argument("--label_topn", type=int, default=16)
    ap.add_argument("--oracle_steps", type=int, default=3)
    ap.add_argument("--swap_screen_samples", type=int, default=16)
    ap.add_argument("--full_swap_sweep", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.tasks = [t for t in args.tasks.split(",") if t]
    if not args.output_dir:
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = str(REPO_ROOT / "results" / f"fup_guarded_swap_stage2_64_{ts}")
    if not args.run_fup and not args.run_guarded_swap:
        args.run_fup = args.run_guarded_swap = True
    return args


def main():
    args = parse_args()
    from tools.fup_pipeline_run import run_pipeline
    run_pipeline(args, Driver)


if __name__ == "__main__":
    main()
