#!/usr/bin/env python3
"""Offline learned Feedback-DC router diagnostic on held-out64 features.

Uses existing feedback features and score-derived labels only.
No generation, no scorer, no full run, no selector changes.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_FEEDBACK_DIR = Path("results/feedback_adaptive_dc_selector_replay_20260708_184705")
DEFAULT_HELDOUT_DIR = Path("results/adaptive_dc_v3_heldout64_diagnostic_20260708_073628")

FEATURE_NAMES = [
    "original_answer_empty_or_short",
    "original_answer_generic_or_refusal",
    "original_answer_repeats_question",
    "answer_terms_supported_by_original_context",
    "query_entities_covered_by_original_context",
    "answer_entities_covered_by_original_context",
    "original_context_entity_coverage_gap",
    "dc_context_additional_entity_coverage",
    "original_vs_dc_prediction_disagreement",
    "dc_answer_terms_supported_by_dc_context",
    "answer_jaccard_original_vs_dc",
]

FORBIDDEN_FEATURES = {
    "task",
    "sample_index",
    "sample_id",
    "original_answer",
    "dc_answer",
    "original_score",
    "dc_score",
    "gold",
    "gold_answer",
    "score_delta",
    "dc_better",
    "strong_dc_better",
    "policy_f1_score",
    "policy_f2_score",
}

TASKS = ["hotpotqa", "musique"]
N_FOLDS = 5
RANDOM_SEED = 42
GO_MACRO_GAIN = 1.0
STRONG_DC_MARGIN = 1.0


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "y")


def to_float(value: Any) -> float:
    try:
        out = float(value)
        return out if math.isfinite(out) else 0.0
    except (TypeError, ValueError):
        return 0.0


def mean(values: Sequence[float]) -> float:
    items = [float(v) for v in values if math.isfinite(float(v))]
    return float(sum(items) / len(items)) if items else 0.0


def task_macro(rows: Sequence[Mapping[str, Any]], score_key: str) -> Dict[str, float]:
    by_task: Dict[str, List[float]] = {task: [] for task in TASKS}
    for row in rows:
        by_task[str(row["task"])].append(to_float(row[score_key]))
    per_task = {task: mean(by_task[task]) for task in TASKS}
    per_task["macro"] = mean([per_task[task] for task in TASKS])
    return per_task


def build_matrix(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    matrix = np.zeros((len(rows), len(FEATURE_NAMES)), dtype=np.float64)
    for row_idx, row in enumerate(rows):
        for col_idx, name in enumerate(FEATURE_NAMES):
            value = row.get(name)
            if name in {
                "original_answer_empty_or_short",
                "original_answer_generic_or_refusal",
                "original_answer_repeats_question",
                "original_vs_dc_prediction_disagreement",
            }:
                matrix[row_idx, col_idx] = 1.0 if parse_bool(value) else 0.0
            else:
                matrix[row_idx, col_idx] = to_float(value)
    return matrix


def add_labels(rows: List[Dict[str, Any]]) -> None:
    for row in rows:
        original_score = to_float(row.get("original_score"))
        dc_score = to_float(row.get("dc_score"))
        row["dc_better"] = int(dc_score > original_score + 1e-9)
        row["strong_dc_better"] = int((dc_score - original_score) >= STRONG_DC_MARGIN)
        row["oracle_score"] = max(original_score, dc_score)
        row["f1_heuristic_score"] = to_float(row.get("policy_f1_score"))


def stratified_folds(labels: np.ndarray, n_folds: int, seed: int) -> List[np.ndarray]:
    rng = np.random.RandomState(seed)
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)
    fold_buckets: List[List[int]] = [[] for _ in range(n_folds)]
    for idx, sample_idx in enumerate(pos_idx):
        fold_buckets[idx % n_folds].append(int(sample_idx))
    for idx, sample_idx in enumerate(neg_idx):
        fold_buckets[idx % n_folds].append(int(sample_idx))
    return [np.array(bucket, dtype=np.int64) for bucket in fold_buckets]


def tune_label_threshold(probs: np.ndarray, y: np.ndarray) -> float:
    best_threshold = 0.5
    best_balanced = -1.0
    for threshold in np.linspace(0.05, 0.95, 19):
        pred = (probs >= threshold).astype(np.int64)
        tp = int(((pred == 1) & (y == 1)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        tpr = tp / (tp + fn) if tp + fn else 0.0
        tnr = tn / (tn + fp) if tn + fp else 0.0
        balanced = 0.5 * (tpr + tnr)
        if balanced > best_balanced:
            best_balanced = balanced
            best_threshold = float(threshold)
    return best_threshold


class NumpyLogisticRouter:
    backend = "logistic_regression"

    def __init__(self, seed: int = RANDOM_SEED) -> None:
        self.seed = seed
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.weight_: Optional[np.ndarray] = None
        self.bias_: float = 0.0
        self.threshold_: float = 0.5
        self.feature_importance_: Dict[str, float] = {}

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        rng = np.random.RandomState(self.seed)
        self.mean_ = x.mean(axis=0)
        self.std_ = x.std(axis=0)
        self.std_[self.std_ < 1e-9] = 1.0
        x_norm = (x - self.mean_) / self.std_
        n_features = x_norm.shape[1]
        weight = rng.normal(0.0, 0.01, size=n_features)
        bias = 0.0
        positives = max(1, int(y.sum()))
        negatives = max(1, len(y) - positives)
        pos_weight = negatives / positives
        lr = 0.08
        l2 = 0.02
        sample_weight = np.where(y == 1, pos_weight, 1.0)
        for _ in range(800):
            logits = x_norm @ weight + bias
            probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
            errors = (probs - y) * sample_weight
            grad_w = (x_norm.T @ errors) / sample_weight.sum() + l2 * weight
            grad_b = float(errors.sum() / sample_weight.sum())
            weight -= lr * grad_w
            bias -= lr * grad_b
        self.weight_ = weight
        self.bias_ = bias
        train_probs = 1.0 / (1.0 + np.exp(-np.clip(x_norm @ weight + bias, -30, 30)))
        self.threshold_ = tune_label_threshold(train_probs, y)
        importance = np.abs(weight * self.std_)
        total = float(importance.sum()) or 1.0
        self.feature_importance_ = {
            FEATURE_NAMES[i]: float(importance[i] / total) for i in range(n_features)
        }

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None or self.weight_ is None:
            raise RuntimeError("model not fit")
        x_norm = (x - self.mean_) / self.std_
        logits = x_norm @ self.weight_ + self.bias_
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))

    def predict(self, x: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        cutoff = self.threshold_ if threshold is None else threshold
        return (self.predict_proba(x) >= cutoff).astype(np.int64)


class SklearnLogisticRouter(NumpyLogisticRouter):
    backend = "logistic_regression_sklearn"

    def __init__(self, seed: int = RANDOM_SEED) -> None:
        super().__init__(seed=seed)
        self.sk_model: Any = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        from sklearn.linear_model import LogisticRegression

        self.sk_model = LogisticRegression(max_iter=500, C=1.0, random_state=self.seed)
        self.sk_model.fit(x, y)
        self.mean_ = x.mean(axis=0)
        self.std_ = x.std(axis=0)
        self.std_[self.std_ < 1e-9] = 1.0
        self.weight_ = self.sk_model.coef_.reshape(-1)
        self.bias_ = float(self.sk_model.intercept_[0])
        importance = np.abs(self.weight_ * self.std_)
        total = float(importance.sum()) or 1.0
        self.feature_importance_ = {
            FEATURE_NAMES[i]: float(importance[i] / total) for i in range(len(FEATURE_NAMES))
        }
        self.threshold_ = tune_label_threshold(self.predict_proba(x), y)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.sk_model is None:
            raise RuntimeError("model not fit")
        return self.sk_model.predict_proba(x)[:, 1]


class SklearnTreeRouter:
    backend = "decision_tree_depth3"

    def __init__(self, seed: int = RANDOM_SEED) -> None:
        self.seed = seed
        self.model: Any = None
        self.threshold_: float = 0.5
        self.feature_importance_: Dict[str, float] = {}

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        from sklearn.tree import DecisionTreeClassifier

        self.model = DecisionTreeClassifier(max_depth=3, random_state=self.seed, class_weight="balanced")
        self.model.fit(x, y)
        raw = self.model.feature_importances_
        total = float(raw.sum()) or 1.0
        self.feature_importance_ = {
            FEATURE_NAMES[i]: float(raw[i] / total) for i in range(len(FEATURE_NAMES))
        }
        self.threshold_ = tune_label_threshold(self.predict_proba(x), y)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("model not fit")
        return self.model.predict_proba(x)[:, 1]

    def predict(self, x: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        cutoff = self.threshold_ if threshold is None else threshold
        return (self.predict_proba(x) >= cutoff).astype(np.int64)


class NumpyTreeRouter:
    backend = "decision_tree_depth3_numpy"

    def __init__(self, seed: int = RANDOM_SEED, max_depth: int = 3) -> None:
        self.seed = seed
        self.max_depth = max_depth
        self.tree: Optional[Dict[str, Any]] = None
        self.threshold_: float = 0.5
        self.feature_importance_: Dict[str, float] = {}

    def _gini(self, y: np.ndarray) -> float:
        if len(y) == 0:
            return 0.0
        p = float(y.mean())
        return 2 * p * (1 - p)

    def _best_split(self, x: np.ndarray, y: np.ndarray) -> Optional[Tuple[int, float, np.ndarray, np.ndarray]]:
        best_gain = 0.0
        best: Optional[Tuple[int, float, np.ndarray, np.ndarray]] = None
        base = self._gini(y)
        n_samples, n_features = x.shape
        for feat_idx in range(n_features):
            values = np.unique(x[:, feat_idx])
            if len(values) <= 1:
                continue
            thresholds = values[:-1] if len(values) <= 16 else np.quantile(values, np.linspace(0.1, 0.9, 9))
            for threshold in thresholds:
                left_mask = x[:, feat_idx] <= threshold
                right_mask = ~left_mask
                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue
                gain = base - (left_mask.mean() * self._gini(y[left_mask]) + right_mask.mean() * self._gini(y[right_mask]))
                if gain > best_gain:
                    best_gain = gain
                    best = (feat_idx, float(threshold), left_mask, right_mask)
        return best

    def _build(self, x: np.ndarray, y: np.ndarray, depth: int) -> Dict[str, Any]:
        if depth >= self.max_depth or len(np.unique(y)) == 1:
            return {"leaf": True, "prob": float(y.mean())}
        split = self._best_split(x, y)
        if split is None:
            return {"leaf": True, "prob": float(y.mean())}
        feat_idx, threshold, left_mask, right_mask = split
        return {
            "leaf": False,
            "feature_idx": feat_idx,
            "threshold": threshold,
            "left": self._build(x[left_mask], y[left_mask], depth + 1),
            "right": self._build(x[right_mask], y[right_mask], depth + 1),
        }

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        self.tree = self._build(x, y, depth=0)
        counts = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
        self._importance_walk(x, y, counts)
        total = float(counts.sum()) or 1.0
        self.feature_importance_ = {
            FEATURE_NAMES[i]: float(counts[i] / total) for i in range(len(FEATURE_NAMES))
        }
        self.threshold_ = tune_label_threshold(self.predict_proba(x), y)

    def _importance_walk(self, x: np.ndarray, y: np.ndarray, counts: np.ndarray) -> None:
        if self.tree is None:
            return
        stack = [(self.tree, x, y)]
        while stack:
            node, node_x, node_y = stack.pop()
            if node.get("leaf"):
                continue
            feat_idx = int(node["feature_idx"])
            counts[feat_idx] += float(len(node_y))
            threshold = float(node["threshold"])
            left_mask = node_x[:, feat_idx] <= threshold
            right_mask = ~left_mask
            stack.append((node["left"], node_x[left_mask], node_y[left_mask]))
            stack.append((node["right"], node_x[right_mask], node_y[right_mask]))

    def _walk_predict(self, row: np.ndarray, node: Dict[str, Any]) -> float:
        if node.get("leaf"):
            return float(node["prob"])
        feat_idx = int(node["feature_idx"])
        threshold = float(node["threshold"])
        if row[feat_idx] <= threshold:
            return self._walk_predict(row, node["left"])
        return self._walk_predict(row, node["right"])

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.tree is None:
            raise RuntimeError("model not fit")
        return np.array([self._walk_predict(row, self.tree) for row in x], dtype=np.float64)

    def predict(self, x: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        cutoff = self.threshold_ if threshold is None else threshold
        return (self.predict_proba(x) >= cutoff).astype(np.int64)


class RuleListRouter:
    backend = "rule_list_from_importance"

    def __init__(self, seed: int = RANDOM_SEED) -> None:
        self.seed = seed
        self.rules: List[Dict[str, Any]] = []
        self.threshold_: float = 0.67
        self.feature_importance_: Dict[str, float] = {}
        self.base_router = NumpyLogisticRouter(seed=seed)

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        self.base_router.fit(x, y)
        self.feature_importance_ = dict(self.base_router.feature_importance_)
        ranked = sorted(self.feature_importance_.items(), key=lambda item: item[1], reverse=True)[:3]
        self.rules = []
        for feature_name, importance in ranked:
            feat_idx = FEATURE_NAMES.index(feature_name)
            coef = float(self.base_router.weight_[feat_idx]) if self.base_router.weight_ is not None else 0.0
            direction = "ge" if coef >= 0 else "le"
            values = x[:, feat_idx]
            thresholds = np.unique(np.quantile(values, [0.2, 0.35, 0.5, 0.65, 0.8]))
            best_threshold = float(values.mean())
            best_score = -1.0
            for threshold in thresholds:
                if direction == "ge":
                    pred = (values >= threshold).astype(np.int64)
                else:
                    pred = (values <= threshold).astype(np.int64)
                tp = int(((pred == 1) & (y == 1)).sum())
                fp = int(((pred == 1) & (y == 0)).sum())
                fn = int(((pred == 0) & (y == 1)).sum())
                precision = tp / (tp + fp) if tp + fp else 0.0
                recall = tp / (tp + fn) if tp + fn else 0.0
                score = precision + recall
                if score > best_score:
                    best_score = score
                    best_threshold = float(threshold)
            self.rules.append({
                "feature": feature_name,
                "direction": direction,
                "threshold": best_threshold,
                "importance": float(importance),
            })
        self.threshold_ = tune_label_threshold(self.predict_proba(x), y)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        votes = np.zeros(len(x), dtype=np.float64)
        for rule in self.rules:
            feat_idx = FEATURE_NAMES.index(rule["feature"])
            values = x[:, feat_idx]
            if rule["direction"] == "ge":
                votes += (values >= float(rule["threshold"])).astype(np.float64)
            else:
                votes += (values <= float(rule["threshold"])).astype(np.float64)
        return votes / max(1.0, float(len(self.rules)))

    def predict(self, x: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        cutoff = self.threshold_ if threshold is None else threshold
        return (self.predict_proba(x) >= cutoff).astype(np.int64)


def make_router(model_name: str, seed: int) -> Any:
    if model_name == "logistic_regression":
        try:
            import sklearn  # noqa: F401

            return SklearnLogisticRouter(seed=seed)
        except Exception:
            return NumpyLogisticRouter(seed=seed)
    if model_name == "decision_tree_depth3":
        try:
            import sklearn  # noqa: F401

            return SklearnTreeRouter(seed=seed)
        except Exception:
            return NumpyTreeRouter(seed=seed, max_depth=3)
    if model_name == "rule_list":
        return RuleListRouter(seed=seed)
    raise ValueError(f"unknown model: {model_name}")


def router_metrics(rows: Sequence[Mapping[str, Any]], trigger_key: str, score_key: str) -> Dict[str, Any]:
    triggered = [row for row in rows if parse_bool(row.get(trigger_key))]
    trigger_count = len(triggered)
    dc_positive = [row for row in rows if int(row.get("dc_better", 0)) == 1]
    dc_negative = [row for row in rows if int(row.get("dc_better", 0)) == 0 and to_float(row["dc_score"]) < to_float(row["original_score"]) - 1e-9]

    tp = sum(1 for row in rows if parse_bool(row.get(trigger_key)) and int(row.get("dc_better", 0)) == 1)
    fp = sum(1 for row in rows if parse_bool(row.get(trigger_key)) and int(row.get("dc_better", 0)) == 0)
    fn = sum(1 for row in rows if not parse_bool(row.get(trigger_key)) and int(row.get("dc_better", 0)) == 1)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    macros = task_macro(rows, score_key)
    return {
        "trigger_count": trigger_count,
        "trigger_rate": trigger_count / len(rows) if rows else 0.0,
        "macro_feedback": macros["macro"],
        "hotpotqa_feedback_macro": macros["hotpotqa"],
        "musique_feedback_macro": macros["musique"],
        "dc_positive_precision": precision,
        "dc_positive_recall": recall,
        "dc_positive_count": len(dc_positive),
        "dc_negative_false_trigger_count": sum(
            1 for row in rows
            if parse_bool(row.get(trigger_key)) and to_float(row["dc_score"]) < to_float(row["original_score"]) - 1e-9
        ),
    }


def run_cv(rows: List[Dict[str, Any]], model_name: str) -> Tuple[List[Dict[str, Any]], Dict[str, float], List[Dict[str, Any]]]:
    x = build_matrix(rows)
    y = np.array([int(row["dc_better"]) for row in rows], dtype=np.int64)
    folds = stratified_folds(y, N_FOLDS, RANDOM_SEED)
    oof_rows: List[Dict[str, Any]] = []
    fold_macros: List[float] = []
    importance_accum: Dict[str, float] = defaultdict(float)

    for fold_id, test_idx in enumerate(folds):
        test_set = set(int(i) for i in test_idx)
        train_idx = np.array([i for i in range(len(rows)) if i not in test_set], dtype=np.int64)
        router = make_router(model_name, seed=RANDOM_SEED + fold_id)
        router.fit(x[train_idx], y[train_idx])
        probs = router.predict_proba(x[test_idx])
        preds = router.predict(x[test_idx])

        for local_idx, global_idx in enumerate(test_idx):
            row = dict(rows[int(global_idx)])
            triggered = bool(int(preds[local_idx]))
            row["model"] = model_name
            row["fold_id"] = fold_id
            row["router_probability"] = float(probs[local_idx])
            row["router_triggered"] = triggered
            row["router_chosen_selector"] = "dc" if triggered else "original"
            row["router_score"] = to_float(row["dc_score"]) if triggered else to_float(row["original_score"])
            oof_rows.append(row)

        fold_metrics = router_metrics(oof_rows[-len(test_idx):], "router_triggered", "router_score")
        fold_macros.append(float(fold_metrics["macro_feedback"]))
        for feature, weight in router.feature_importance_.items():
            importance_accum[feature] += float(weight)

    for feature in FEATURE_NAMES:
        importance_accum[feature] /= float(N_FOLDS)

    summary = router_metrics(oof_rows, "router_triggered", "router_score")
    summary["model"] = model_name
    summary["cv_fold_macros"] = fold_macros
    summary["cv_macro_std"] = float(np.std(fold_macros))
    summary["backend"] = router.backend
    importance_rows = [
        {"model": model_name, "feature": feature, "importance": float(importance_accum[feature])}
        for feature in FEATURE_NAMES
    ]
    importance_rows.sort(key=lambda row: row["importance"], reverse=True)
    return oof_rows, summary, importance_rows


def decide(
    best_summary: Mapping[str, Any],
    original_macro: float,
    f1_macro: float,
    f1_precision: float,
) -> Dict[str, Any]:
    cv_macro = float(best_summary["macro_feedback"])
    delta = cv_macro - original_macro
    trigger_count = int(best_summary["trigger_count"])
    precision = float(best_summary["dc_positive_precision"])
    fold_std = float(best_summary.get("cv_macro_std", 0.0))

    if trigger_count in (0, 128):
        return {
            "decision": "LEARNED_FEEDBACK_NO_GO",
            "reasons": [f"degenerate_trigger_count={trigger_count}/128", f"cv_macro={cv_macro:.4f}"],
        }

    clearly_better_than_f1 = cv_macro > f1_macro + 0.05
    precision_gain = precision >= max(f1_precision + 0.08, f1_precision * 1.15)

    if cv_macro >= original_macro + GO_MACRO_GAIN and clearly_better_than_f1 and precision_gain:
        return {
            "decision": "LEARNED_FEEDBACK_GO",
            "reasons": [
                f"cv_macro={cv_macro:.4f} >= original+{GO_MACRO_GAIN:.1f}",
                f"f1_heuristic_macro={f1_macro:.4f}",
                f"dc_positive_precision={precision:.3f}",
                f"trigger_count={trigger_count}/128",
            ],
        }

    if cv_macro > original_macro and (delta < GO_MACRO_GAIN or fold_std > 1.5 or not clearly_better_than_f1):
        return {
            "decision": "LEARNED_FEEDBACK_WEAK",
            "reasons": [
                f"cv_macro-original={delta:+.4f}",
                f"f1_heuristic_macro={f1_macro:.4f}",
                f"cv_macro_std={fold_std:.4f}",
                f"dc_positive_precision={precision:.3f}",
                f"trigger_count={trigger_count}/128",
            ],
        }

    return {
        "decision": "LEARNED_FEEDBACK_NO_GO",
        "reasons": [
            f"cv_macro={cv_macro:.4f}",
            f"original_macro={original_macro:.4f}",
            f"f1_heuristic_macro={f1_macro:.4f}",
            f"dc_positive_precision={precision:.3f}",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feedback_dir", type=Path, default=DEFAULT_FEEDBACK_DIR)
    parser.add_argument("--heldout_dir", type=Path, default=DEFAULT_HELDOUT_DIR)
    parser.add_argument("--output_root", type=Path, default=Path("results"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    feedback_dir = (REPO_ROOT / args.feedback_dir).resolve()
    heldout_dir = (REPO_ROOT / args.heldout_dir).resolve()
    output_dir = (
        REPO_ROOT / args.output_root / f"offline_feedback_dc_router_{datetime.now():%Y%m%d_%H%M%S}"
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    per_sample_path = feedback_dir / "feedback_selector_per_sample.csv"
    if not per_sample_path.exists():
        raise FileNotFoundError(per_sample_path)

    rows = read_csv(per_sample_path)
    if len(rows) != 128:
        raise RuntimeError(f"BUG_STOP: expected 128 rows, got {len(rows)}")

    leaked = sorted(set(FEATURE_NAMES) & FORBIDDEN_FEATURES)
    if leaked:
        raise RuntimeError(f"BUG_STOP: forbidden features in model input: {leaked}")

    add_labels(rows)

    baselines = {
        "original_macro": task_macro(rows, "original_score")["macro"],
        "dc_macro": task_macro(rows, "dc_score")["macro"],
        "f1_heuristic_macro": task_macro(rows, "f1_heuristic_score")["macro"],
        "oracle_macro": task_macro(rows, "oracle_score")["macro"],
    }
    f1_metrics = router_metrics(rows, "policy_f1_triggered", "f1_heuristic_score")

    all_oof: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []
    importance_rows: List[Dict[str, Any]] = []

    for model_name in ["logistic_regression", "decision_tree_depth3", "rule_list"]:
        oof_rows, summary, model_importance = run_cv(rows, model_name)
        all_oof.extend(oof_rows)
        summary.update(baselines)
        summary["feedback_vs_original_delta"] = float(summary["macro_feedback"]) - baselines["original_macro"]
        summary["feedback_vs_f1_delta"] = float(summary["macro_feedback"]) - baselines["f1_heuristic_macro"]
        summary["feedback_vs_oracle_delta"] = float(summary["macro_feedback"]) - baselines["oracle_macro"]
        summary["f1_heuristic_macro"] = baselines["f1_heuristic_macro"]
        summary["f1_dc_positive_precision"] = f1_metrics["dc_positive_precision"]
        summaries.append(summary)
        importance_rows.extend(model_importance)

    non_degenerate = [
        row for row in summaries
        if 0 < int(row["trigger_count"]) < 128
    ]
    candidate_pool = non_degenerate if non_degenerate else summaries
    best_summary = max(candidate_pool, key=lambda row: float(row["macro_feedback"]))
    decision = decide(
        best_summary,
        baselines["original_macro"],
        baselines["f1_heuristic_macro"],
        f1_metrics["dc_positive_precision"],
    )

    oof_fields = [
        "model", "fold_id", "task", "sample_index", "sample_id",
        "original_score", "dc_score", "dc_better", "strong_dc_better",
        "router_probability", "router_triggered", "router_chosen_selector", "router_score",
        "f1_heuristic_score", "oracle_score",
        *FEATURE_NAMES,
    ]
    write_csv(output_dir / "learned_router_cv_predictions.csv", all_oof, oof_fields)

    summary_fields = [
        "model", "backend", "macro_feedback", "hotpotqa_feedback_macro", "musique_feedback_macro",
        "original_macro", "dc_macro", "f1_heuristic_macro", "oracle_macro",
        "feedback_vs_original_delta", "feedback_vs_f1_delta", "feedback_vs_oracle_delta",
        "trigger_count", "trigger_rate", "dc_positive_precision", "dc_positive_recall",
        "dc_positive_count", "dc_negative_false_trigger_count",
        "cv_macro_std", "f1_dc_positive_precision",
    ]
    write_csv(output_dir / "learned_router_summary.csv", summaries, summary_fields)
    write_csv(
        output_dir / "learned_router_feature_importance.csv",
        importance_rows,
        ["model", "feature", "importance"],
    )

    report = [
        "# Offline Learned Feedback-DC Router Diagnostic",
        "",
        f"- Output: `{output_dir}`",
        f"- Feedback dir: `{feedback_dir}`",
        f"- Held-out dir: `{heldout_dir}`",
        "",
        "## Setup",
        "",
        "- 5-fold stratified CV on held-out64 feedback features (128 samples).",
        "- Label: `dc_better = 1` iff `dc_score > original_score`; `strong_dc_better` iff margin >= 1.0.",
        "- Forbidden runtime features: task label, sample index, gold answer, score fields, score_delta.",
        "- Scores used only for labels and offline evaluation.",
        "",
        "## Baselines",
        "",
        f"- Original macro: **{baselines['original_macro']:.4f}**",
        f"- DC macro: **{baselines['dc_macro']:.4f}**",
        f"- F1 heuristic macro: **{baselines['f1_heuristic_macro']:.4f}**",
        f"- Oracle max(original,dc) macro: **{baselines['oracle_macro']:.4f}**",
        "",
        "## CV results",
        "",
        "| model | CV macro | vs original | vs F1 | trigger | DC+ precision | DC+ recall | false trigger | fold std |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for summary in summaries:
        report.append(
            f"| {summary['model']} | {summary['macro_feedback']:.4f} | "
            f"{summary['feedback_vs_original_delta']:+.4f} | {summary['feedback_vs_f1_delta']:+.4f} | "
            f"{summary['trigger_count']}/128 | {summary['dc_positive_precision']:.3f} | "
            f"{summary['dc_positive_recall']:.3f} | {summary['dc_negative_false_trigger_count']} | "
            f"{summary['cv_macro_std']:.4f} |"
        )

    report.extend([
        "",
        "## Per-task best model",
        "",
        f"- Best overall: **{best_summary['model']}** ({best_summary['backend']})",
        f"- HotpotQA macro: {best_summary['hotpotqa_feedback_macro']:.4f}",
        f"- MuSiQue macro: {best_summary['musique_feedback_macro']:.4f}",
        "",
        "## Feature importance (top 5, best model)",
        "",
    ])
    best_importance = [row for row in importance_rows if row["model"] == best_summary["model"]][:5]
    for row in best_importance:
        report.append(f"- {row['feature']}: {row['importance']:.4f}")

    report.extend([
        "",
        "## Decision",
        "",
        f"- **{decision['decision']}**",
    ])
    for reason in decision.get("reasons", []):
        report.append(f"- `{reason}`")
    report.append("")
    (output_dir / "learned_router_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    decision_lines = [
        "# Offline Learned Feedback-DC Router Decision",
        "",
        f"- Decision: **{decision['decision']}**",
        f"- Best model: **{best_summary['model']}**",
        f"- CV macro: {float(best_summary['macro_feedback']):.4f}",
        f"- Original macro: {baselines['original_macro']:.4f}",
        f"- F1 heuristic macro: {baselines['f1_heuristic_macro']:.4f}",
        f"- Oracle macro: {baselines['oracle_macro']:.4f}",
        "",
    ]
    for reason in decision.get("reasons", []):
        decision_lines.append(f"- Reason: `{reason}`")
    decision_lines.append("")
    (output_dir / "decision.md").write_text("\n".join(decision_lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "status": "ok",
        "output_dir": str(output_dir),
        "decision": decision,
        "baselines": baselines,
        "summaries": summaries,
        "best_model": best_summary["model"],
    }, indent=2))


if __name__ == "__main__":
    main()
