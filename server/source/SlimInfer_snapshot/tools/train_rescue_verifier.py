#!/usr/bin/env python3
"""Train a lightweight rescue verifier without generating model predictions.

The default backend is a CPU-only PyTorch logistic regressor because sklearn is
not required by this repository.  Neutral examples are excluded by default.
The optional text backend is intentionally skipped unless an explicit local
cross-encoder/reranker checkpoint is supplied; this script never downloads one.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


FORBIDDEN_FEATURES = {
    "task",
    "sample_id",
    "sample_index",
    "gold_answers",
    "original_prediction",
    "random_swap_prediction",
    "drm_swap_prediction",
    "drm_add_prediction",
    "original_score",
    "random_swap_score",
    "drm_swap_score",
    "drm_add_score",
    "delta_drm_vs_original",
    "delta_drm_vs_random",
    "primary_label",
    "label_drm_better_original",
    "label_drm_worse_original",
    "label_drm_better_random",
    "label_drm_worse_random",
}


class TrainingError(RuntimeError):
    pass


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise TrainingError(f"invalid JSONL {path}:{line_number}: {exc}") from exc


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_number(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TrainingError(f"feature {field} is not numeric: {value!r}") from exc
    if not math.isfinite(result):
        raise TrainingError(f"feature {field} is not finite: {value!r}")
    return result


def feature_names(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    if not rows:
        raise TrainingError("empty training dataset")
    declared = rows[0].get("inference_numeric_feature_names")
    if not isinstance(declared, list) or not declared:
        raise TrainingError("dataset does not declare inference_numeric_feature_names")
    names = [str(name) for name in declared]
    forbidden = sorted(set(names) & FORBIDDEN_FEATURES)
    if forbidden:
        raise TrainingError(f"forbidden inference features requested: {forbidden}")
    for row in rows:
        if row.get("inference_numeric_feature_names") != declared:
            raise TrainingError("inference numeric feature declaration differs across examples")
        allowance = row.get("inference_allowed", {})
        missing = [name for name in names if not bool(allowance.get(name, False))]
        if missing:
            raise TrainingError(
                f"{row.get('task')}/{row.get('sample_index')}: "
                f"features not marked inference_allowed: {missing}"
            )
        for name in names:
            finite_number(row.get(name), name)
    return names


def matrix(rows: Sequence[Mapping[str, Any]], names: Sequence[str]) -> List[List[float]]:
    return [[finite_number(row.get(name), name) for name in names] for row in rows]


def binary_metrics(
    rows: Sequence[Mapping[str, Any]], probabilities: Sequence[float]
) -> Dict[str, Any]:
    eligible = [
        (int(row["primary_label"]), float(probability))
        for row, probability in zip(rows, probabilities)
        if int(row["primary_label"]) != 0
    ]
    labels = [1 if label == 1 else 0 for label, _ in eligible]
    probs = [probability for _, probability in eligible]
    predictions = [int(probability >= 0.5) for probability in probs]
    tp = sum(y == 1 and p == 1 for y, p in zip(labels, predictions))
    tn = sum(y == 0 and p == 0 for y, p in zip(labels, predictions))
    fp = sum(y == 0 and p == 1 for y, p in zip(labels, predictions))
    fn = sum(y == 1 and p == 0 for y, p in zip(labels, predictions))
    positive = sum(labels)
    negative = len(labels) - positive
    accuracy = (tp + tn) / len(labels) if labels else None
    tpr = tp / positive if positive else None
    tnr = tn / negative if negative else None
    balanced = (tpr + tnr) / 2 if tpr is not None and tnr is not None else None
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tpr if tpr is not None else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    auc = None
    if positive and negative:
        positive_probs = [p for y, p in zip(labels, probs) if y == 1]
        negative_probs = [p for y, p in zip(labels, probs) if y == 0]
        wins = sum(
            1.0 if pp > np else 0.5 if pp == np else 0.0
            for pp in positive_probs
            for np in negative_probs
        )
        auc = wins / (len(positive_probs) * len(negative_probs))
    return {
        "eligible_non_neutral": len(labels),
        "positive": positive,
        "negative": negative,
        "neutral_excluded_from_metrics": len(rows) - len(labels),
        "threshold": 0.5,
        "accuracy": accuracy,
        "balanced_accuracy": balanced,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc_pairwise": auc,
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "reliability_warning": (
            "Metrics use very few non-neutral examples and are not evidence of "
            "generalization or statistical significance."
        ),
    }


def train_feature_logistic(
    train_rows: Sequence[Mapping[str, Any]],
    dev_rows: Sequence[Mapping[str, Any]],
    test_rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    import torch

    torch.set_num_threads(max(1, args.cpu_threads))
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    names = feature_names(train_rows)
    feature_names(list(dev_rows) + list(test_rows))

    train_all_x = torch.tensor(matrix(train_rows, names), dtype=torch.float64)
    mean = train_all_x.mean(dim=0)
    std = train_all_x.std(dim=0, unbiased=False)
    std = torch.where(std > 1e-12, std, torch.ones_like(std))

    eligible_rows = [
        row
        for row in train_rows
        if int(row["primary_label"]) != 0
        or args.include_neutral_as_weighted_negative
    ]
    if not eligible_rows:
        raise TrainingError("no eligible training examples")
    x = torch.tensor(matrix(eligible_rows, names), dtype=torch.float64)
    x = (x - mean) / std
    labels = torch.tensor(
        [1.0 if int(row["primary_label"]) == 1 else 0.0 for row in eligible_rows],
        dtype=torch.float64,
    )
    weights = torch.tensor(
        [
            args.neutral_weight if int(row["primary_label"]) == 0 else 1.0
            for row in eligible_rows
        ],
        dtype=torch.float64,
    )
    positives = int((labels == 1).sum().item())
    negatives = int((labels == 0).sum().item())
    if positives == 0 or negatives == 0:
        raise TrainingError(
            f"logistic training needs both classes; positive={positives}, negative={negatives}"
        )

    model = torch.nn.Linear(len(names), 1, dtype=torch.float64)
    torch.nn.init.zeros_(model.weight)
    prior = positives / (positives + negatives)
    torch.nn.init.constant_(model.bias, math.log(prior / (1.0 - prior)))
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    positive_weight = negatives / positives
    class_weights = torch.where(
        labels == 1,
        torch.full_like(labels, positive_weight),
        torch.ones_like(labels),
    )
    losses: List[float] = []
    model.train()
    for epoch in range(args.epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x).reshape(-1)
        per_example = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, labels, reduction="none"
        )
        loss = (per_example * class_weights * weights).sum() / (
            class_weights * weights
        ).sum()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().item()))

    def probabilities(rows: Sequence[Mapping[str, Any]]) -> List[float]:
        values = torch.tensor(matrix(rows, names), dtype=torch.float64)
        values = (values - mean) / std
        model.eval()
        with torch.no_grad():
            return torch.sigmoid(model(values).reshape(-1)).tolist()

    dev_probabilities = probabilities(dev_rows)
    test_probabilities = probabilities(test_rows)
    state = {
        "format_version": 1,
        "backend": "feature_logistic",
        "feature_names": names,
        "feature_mean": mean,
        "feature_std": std,
        "weight": model.weight.detach().reshape(-1),
        "bias": model.bias.detach().reshape(-1),
        "seed": args.seed,
    }
    model_path = output_dir / "verifier_model.pt"
    torch.save(state, model_path)

    coefficients = model.weight.detach().reshape(-1).tolist()
    importance = sorted(
        (
            {
                "feature": name,
                "standardized_coefficient": coefficient,
                "absolute_coefficient": abs(coefficient),
            }
            for name, coefficient in zip(names, coefficients)
        ),
        key=lambda row: -row["absolute_coefficient"],
    )
    write_json(
        output_dir / "verifier_feature_importance.json",
        {
            "interpretation": (
                "Coefficients are fitted on standardized features. With only "
                f"{positives + negatives} default training labels they are exploratory."
            ),
            "features": importance,
        },
    )
    write_json(
        output_dir / "verifier_dev_metrics.json",
        binary_metrics(dev_rows, dev_probabilities),
    )
    write_json(
        output_dir / "verifier_test_metrics.json",
        binary_metrics(test_rows, test_probabilities),
    )
    write_json(
        output_dir / "verifier_config.json",
        {
            "status": "trained",
            "created_at": dt.datetime.now().astimezone().isoformat(),
            "backend": "feature_logistic",
            "implementation": "torch.nn.Linear CPU-only",
            "model_path": str(model_path.resolve()),
            "model_sha256": sha256_file(model_path),
            "dataset_dir": str(args.dataset_dir.resolve()),
            "feature_names": names,
            "forbidden_feature_used": False,
            "neutral_policy": (
                "weighted_negative"
                if args.include_neutral_as_weighted_negative
                else "excluded"
            ),
            "neutral_weight": (
                args.neutral_weight if args.include_neutral_as_weighted_negative else 0.0
            ),
            "training_examples": len(eligible_rows),
            "training_positive": positives,
            "training_negative": negatives,
            "all_train_examples": len(train_rows),
            "dev_examples": len(dev_rows),
            "test_examples": len(test_rows),
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "final_training_loss": losses[-1],
            "minimum_training_loss": min(losses),
            "class_imbalance_warning": (
                "The deterministic split contains very few non-neutral examples; "
                "especially the test positive count. Treat metrics and coefficients "
                "as a stage-gate diagnostic, not a reliable trained verifier claim."
            ),
        },
    )


def skip_text_backend(output_dir: Path, args: argparse.Namespace) -> None:
    if args.local_text_model_path is not None:
        raise TrainingError(
            "A local text checkpoint was supplied, but text fine-tuning is intentionally "
            "not activated in this audit-first run. Use feature_logistic first; this "
            "prevents an unreviewed checkpoint architecture from silently training."
        )
    write_json(
        output_dir / "verifier_config.json",
        {
            "status": "skipped_no_local_model",
            "created_at": dt.datetime.now().astimezone().isoformat(),
            "backend": "text_cross_encoder",
            "network_download_allowed": False,
            "reason": (
                "No explicit local cross-encoder/reranker checkpoint was supplied. "
                "The locally cached BGE embedding checkpoint is not assumed to be a "
                "cross encoder."
            ),
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument(
        "--backend",
        choices=("feature_logistic", "text_cross_encoder"),
        default="feature_logistic",
    )
    parser.add_argument("--local_text_model_path", type=Path)
    parser.add_argument("--include_neutral_as_weighted_negative", action="store_true")
    parser.add_argument("--neutral_weight", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--learning_rate", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu_threads", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir is None:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = (
            args.dataset_dir.parent / f"drm_rescue_verifier_model_{stamp}"
        )
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise TrainingError(f"refusing to overwrite non-empty output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.backend == "text_cross_encoder":
        skip_text_backend(args.output_dir, args)
        print(str(args.output_dir.resolve()))
        return

    train_path = args.dataset_dir / "verifier_examples_train.jsonl"
    dev_path = args.dataset_dir / "verifier_examples_dev.jsonl"
    test_path = args.dataset_dir / "verifier_examples_test.jsonl"
    for path in (train_path, dev_path, test_path):
        if not path.exists():
            raise TrainingError(f"dataset split not found: {path}")
    train_rows = list(iter_jsonl(train_path))
    dev_rows = list(iter_jsonl(dev_path))
    test_rows = list(iter_jsonl(test_path))
    train_feature_logistic(train_rows, dev_rows, test_rows, args.output_dir, args)
    print(str(args.output_dir.resolve()))


if __name__ == "__main__":
    main()
