#!/usr/bin/env python3
"""Simulate a verifier gate using historical per-sample scores only.

Thresholds are selected on the deterministic dev split and then frozen for the
test split.  This script never generates an answer and its results are an
offline policy counterfactual, not a measured model-method result.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


TASKS = ("hotpotqa", "2wikimqa", "musique")
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
}


class SimulationError(RuntimeError):
    pass


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise SimulationError(f"invalid JSONL {path}:{line_number}: {exc}") from exc


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise SimulationError(f"{name} is not numeric: {value!r}") from exc
    if not math.isfinite(result):
        raise SimulationError(f"{name} is not finite: {value!r}")
    return result


def load_model(path: Path) -> Dict[str, Any]:
    import torch

    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    if state.get("backend") != "feature_logistic":
        raise SimulationError(f"unsupported checkpoint backend: {state.get('backend')}")
    names = [str(name) for name in state.get("feature_names", [])]
    forbidden = sorted(set(names) & FORBIDDEN_FEATURES)
    if forbidden:
        raise SimulationError(f"checkpoint contains forbidden inference features: {forbidden}")
    expected = len(names)
    for field in ("feature_mean", "feature_std", "weight"):
        if len(state[field]) != expected:
            raise SimulationError(f"checkpoint {field} length mismatch")
    return state


def predict(rows: Sequence[Mapping[str, Any]], state: Mapping[str, Any]) -> List[float]:
    import torch

    names = list(state["feature_names"])
    values: List[List[float]] = []
    for row in rows:
        allowance = row.get("inference_allowed", {})
        missing_allowed = [name for name in names if not bool(allowance.get(name, False))]
        if missing_allowed:
            raise SimulationError(
                f"{row.get('task')}/{row.get('sample_index')}: "
                f"checkpoint features not inference-allowed: {missing_allowed}"
            )
        values.append([finite(row.get(name), name) for name in names])
    x = torch.tensor(values, dtype=torch.float64)
    mean = state["feature_mean"].to(dtype=torch.float64)
    std = state["feature_std"].to(dtype=torch.float64)
    weight = state["weight"].to(dtype=torch.float64)
    bias = state["bias"].to(dtype=torch.float64).reshape(-1)[0]
    logits = ((x - mean) / std) @ weight + bias
    return torch.sigmoid(logits).tolist()


def baseline_scores(rows: Sequence[Mapping[str, Any]], field: str) -> Dict[str, Any]:
    per_task = {
        task: sum(finite(row[field], field) for row in rows if row["task"] == task)
        / sum(row["task"] == task for row in rows)
        for task in TASKS
    }
    return {
        "per_task": per_task,
        "macro": sum(per_task.values()) / len(TASKS),
    }


def policy_evaluation(
    rows: Sequence[Mapping[str, Any]],
    probabilities: Sequence[float],
    threshold: float,
    deletion_risk_feature: Optional[str],
    deletion_risk_threshold: Optional[float],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    output: List[Dict[str, Any]] = []
    for row, probability in zip(rows, probabilities):
        risk_blocked = False
        risk = None
        if deletion_risk_feature is not None:
            risk = finite(row.get(deletion_risk_feature), deletion_risk_feature)
            risk_blocked = bool(risk >= float(deletion_risk_threshold))
        triggered = bool(probability >= threshold and not risk_blocked)
        selected_score = finite(
            row["drm_swap_score"] if triggered else row["original_score"],
            "selected_score",
        )
        delta = selected_score - finite(row["original_score"], "original_score")
        output.append(
            {
                "task": row["task"],
                "sample_id": row["sample_id"],
                "sample_index": int(row["sample_index"]),
                "primary_label": int(row["primary_label"]),
                "verifier_score": probability,
                "threshold": threshold,
                "deletion_risk_feature": deletion_risk_feature or "",
                "deletion_risk_score": "" if risk is None else risk,
                "deletion_risk_threshold": (
                    "" if deletion_risk_threshold is None else deletion_risk_threshold
                ),
                "risk_blocked": risk_blocked,
                "triggered": triggered,
                "selected_policy": "drm_swap" if triggered else "original",
                "selected_score": selected_score,
                "original_score": finite(row["original_score"], "original_score"),
                "random_swap_score": finite(row["random_swap_score"], "random_swap_score"),
                "drm_swap_score": finite(row["drm_swap_score"], "drm_swap_score"),
                "delta_selected_vs_original": delta,
                "delta_drm_vs_original": finite(
                    row["delta_drm_vs_original"], "delta_drm_vs_original"
                ),
                "delta_drm_vs_random": finite(
                    row["delta_drm_vs_random"], "delta_drm_vs_random"
                ),
                "question": row["question"],
            }
        )
    per_task = {
        task: sum(row["selected_score"] for row in output if row["task"] == task)
        / sum(row["task"] == task for row in output)
        for task in TASKS
    }
    macro = sum(per_task.values()) / len(TASKS)
    trigger_count = sum(bool(row["triggered"]) for row in output)
    trigger_rate = trigger_count / len(output)
    original = baseline_scores(rows, "original_score")
    random_swap = baseline_scores(rows, "random_swap_score")
    drm_swap = baseline_scores(rows, "drm_swap_score")
    outcome_counts = {
        "positive": sum(row["delta_selected_vs_original"] > 0 for row in output),
        "negative": sum(row["delta_selected_vs_original"] < 0 for row in output),
        "zero": sum(row["delta_selected_vs_original"] == 0 for row in output),
    }
    triggered_outcomes = {
        "positive": sum(
            row["triggered"] and row["delta_selected_vs_original"] > 0 for row in output
        ),
        "negative": sum(
            row["triggered"] and row["delta_selected_vs_original"] < 0 for row in output
        ),
        "zero": sum(
            row["triggered"] and row["delta_selected_vs_original"] == 0 for row in output
        ),
    }
    result = {
        "examples": len(rows),
        "threshold": threshold,
        "trigger_count": trigger_count,
        "trigger_rate": trigger_rate,
        "per_task": per_task,
        "macro": macro,
        "baselines": {
            "original": original,
            "random_swap": random_swap,
            "drm_swap": drm_swap,
        },
        "delta_vs_original": macro - original["macro"],
        "delta_vs_random_swap": macro - random_swap["macro"],
        "delta_vs_drm_swap": macro - drm_swap["macro"],
        "per_task_delta_vs_original": {
            task: per_task[task] - original["per_task"][task] for task in TASKS
        },
        "selected_outcome_counts_vs_original": outcome_counts,
        "triggered_outcome_counts_vs_original": triggered_outcomes,
        "label_distribution": {
            str(label): sum(int(row["primary_label"]) == label for row in rows)
            for label in (-1, 0, 1)
        },
    }
    return result, output


def concentration(policy_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    positives = sorted(
        (
            row
            for row in policy_rows
            if row["triggered"] and row["delta_selected_vs_original"] > 0
        ),
        key=lambda row: -row["delta_selected_vs_original"],
    )
    total = sum(row["delta_selected_vs_original"] for row in positives)
    task_contribution = {
        task: sum(
            row["delta_selected_vs_original"]
            for row in positives
            if row["task"] == task
        )
        for task in TASKS
    }
    task_share = {
        task: (value / total if total > 0 else 0.0)
        for task, value in task_contribution.items()
    }
    top3 = sum(row["delta_selected_vs_original"] for row in positives[:3])
    top5 = sum(row["delta_selected_vs_original"] for row in positives[:5])
    tasks_with_positive_gain = sum(value > 0 for value in task_contribution.values())
    not_few_musique_only = bool(
        total > 0
        and tasks_with_positive_gain >= 2
        and (top3 / total if total else 1.0) <= 0.80
    )
    return {
        "positive_triggered_count": len(positives),
        "positive_gain_sum": total,
        "positive_gain_by_task": task_contribution,
        "positive_gain_share_by_task": task_share,
        "top3_positive_gain_share": top3 / total if total > 0 else None,
        "top5_positive_gain_share": top5 / total if total > 0 else None,
        "tasks_with_positive_gain": tasks_with_positive_gain,
        "not_concentrated_in_few_musique_cases": not_few_musique_only,
        "criterion": (
            "positive gains occur in >=2 tasks and top-3 positive cases account "
            "for <=80% of total positive gain"
        ),
    }


def top_cases(policy_rows: Sequence[Mapping[str, Any]], count: int = 10) -> Dict[str, Any]:
    triggered = [row for row in policy_rows if row["triggered"]]
    common_fields = (
        "task",
        "sample_id",
        "sample_index",
        "verifier_score",
        "primary_label",
        "delta_selected_vs_original",
        "delta_drm_vs_random",
        "question",
    )

    def compact(row: Mapping[str, Any]) -> Dict[str, Any]:
        return {field: row[field] for field in common_fields}

    return {
        "positive": [
            compact(row)
            for row in sorted(
                triggered, key=lambda row: -row["delta_selected_vs_original"]
            )[:count]
            if row["delta_selected_vs_original"] > 0
        ],
        "negative": [
            compact(row)
            for row in sorted(
                triggered, key=lambda row: row["delta_selected_vs_original"]
            )[:count]
            if row["delta_selected_vs_original"] < 0
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--model_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--min_trigger_rate", type=float, default=0.05)
    parser.add_argument("--max_trigger_rate", type=float, default=0.30)
    parser.add_argument("--dev_task_floor_delta", type=float, default=-0.25)
    parser.add_argument("--test_task_floor_delta", type=float, default=-0.25)
    parser.add_argument("--deletion_risk_feature")
    parser.add_argument("--deletion_risk_threshold", type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (args.deletion_risk_feature is None) != (
        args.deletion_risk_threshold is None
    ):
        raise SimulationError(
            "deletion_risk_feature and deletion_risk_threshold must be set together"
        )
    if args.output_dir is None:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = (
            args.dataset_dir.parent / f"drm_rescue_verifier_simulation_{stamp}"
        )
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SimulationError(f"refusing to overwrite non-empty output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dev_path = args.dataset_dir / "verifier_examples_dev.jsonl"
    test_path = args.dataset_dir / "verifier_examples_test.jsonl"
    dev_rows = list(iter_jsonl(dev_path))
    test_rows = list(iter_jsonl(test_path))
    if len(dev_rows) != 150 or len(test_rows) != 150:
        raise SimulationError(
            f"expected dev/test=150/150, got {len(dev_rows)}/{len(test_rows)}"
        )
    state = load_model(args.model_path)
    if args.deletion_risk_feature is not None:
        if args.deletion_risk_feature not in state["feature_names"]:
            raise SimulationError(
                f"deletion risk feature is not in verifier features: "
                f"{args.deletion_risk_feature}"
            )
    dev_probabilities = predict(dev_rows, state)
    test_probabilities = predict(test_rows, state)

    unique_scores = sorted(set(float(value) for value in dev_probabilities))
    thresholds = unique_scores + [math.nextafter(max(unique_scores), math.inf)]
    search_rows: List[Dict[str, Any]] = []
    evaluations: List[Tuple[Dict[str, Any], List[Dict[str, Any]]]] = []
    for threshold in thresholds:
        evaluation, policy_rows = policy_evaluation(
            dev_rows,
            dev_probabilities,
            threshold,
            args.deletion_risk_feature,
            args.deletion_risk_threshold,
        )
        rate_ok = args.min_trigger_rate <= evaluation["trigger_rate"] <= args.max_trigger_rate
        macro_ok = evaluation["delta_vs_original"] >= 0.0
        task_ok = all(
            delta >= args.dev_task_floor_delta
            for delta in evaluation["per_task_delta_vs_original"].values()
        )
        evaluation["dev_rate_constraint_pass"] = rate_ok
        evaluation["dev_macro_constraint_pass"] = macro_ok
        evaluation["dev_task_floor_constraint_pass"] = task_ok
        evaluation["dev_constraints_pass"] = rate_ok and macro_ok and task_ok
        evaluations.append((evaluation, policy_rows))
        search_rows.append(
            {
                "threshold": threshold,
                "trigger_count": evaluation["trigger_count"],
                "trigger_rate": evaluation["trigger_rate"],
                "macro": evaluation["macro"],
                "delta_vs_original": evaluation["delta_vs_original"],
                "hotpotqa": evaluation["per_task"]["hotpotqa"],
                "2wikimqa": evaluation["per_task"]["2wikimqa"],
                "musique": evaluation["per_task"]["musique"],
                "hotpotqa_delta_vs_original": evaluation["per_task_delta_vs_original"]["hotpotqa"],
                "2wikimqa_delta_vs_original": evaluation["per_task_delta_vs_original"]["2wikimqa"],
                "musique_delta_vs_original": evaluation["per_task_delta_vs_original"]["musique"],
                "rate_constraint_pass": rate_ok,
                "macro_constraint_pass": macro_ok,
                "task_floor_constraint_pass": task_ok,
                "all_dev_constraints_pass": rate_ok and macro_ok and task_ok,
            }
        )

    feasible = [item for item in evaluations if item[0]["dev_constraints_pass"]]
    if feasible:
        chosen = sorted(
            feasible,
            key=lambda item: (
                -item[0]["macro"],
                item[0]["trigger_rate"],
                -item[0]["threshold"],
            ),
        )[0]
        selection_status = "selected_feasible_dev_threshold"
    else:
        rate_feasible = [
            item for item in evaluations if item[0]["dev_rate_constraint_pass"]
        ]
        pool = rate_feasible or evaluations
        chosen = sorted(
            pool,
            key=lambda item: (
                -item[0]["macro"],
                abs(item[0]["trigger_rate"] - 0.15),
                item[0]["trigger_rate"],
            ),
        )[0]
        selection_status = (
            "diagnostic_threshold_only_no_full_dev_feasible_threshold"
        )
    dev_evaluation, dev_policy_rows = chosen
    threshold = float(dev_evaluation["threshold"])
    test_evaluation, test_policy_rows = policy_evaluation(
        test_rows,
        test_probabilities,
        threshold,
        args.deletion_risk_feature,
        args.deletion_risk_threshold,
    )
    test_concentration = concentration(test_policy_rows)
    test_task_floor_pass = all(
        delta >= args.test_task_floor_delta
        for delta in test_evaluation["per_task_delta_vs_original"].values()
    )
    criteria = {
        "dev_threshold_constraints_pass": bool(
            dev_evaluation["dev_constraints_pass"]
        ),
        "test_delta_vs_original_at_least_0_30": (
            test_evaluation["delta_vs_original"] >= 0.30
        ),
        "test_delta_vs_random_swap_at_least_0_20": (
            test_evaluation["delta_vs_random_swap"] >= 0.20
        ),
        "test_trigger_rate_5_to_30_percent": (
            args.min_trigger_rate
            <= test_evaluation["trigger_rate"]
            <= args.max_trigger_rate
        ),
        "test_no_task_below_original_minus_0_25": test_task_floor_pass,
        "test_gain_not_concentrated_in_few_musique_cases": test_concentration[
            "not_concentrated_in_few_musique_cases"
        ],
    }
    go = all(criteria.values())

    write_csv(args.output_dir / "threshold_search.csv", search_rows)
    write_csv(args.output_dir / "simulated_policy_dev.csv", dev_policy_rows)
    write_csv(args.output_dir / "simulated_policy_test.csv", test_policy_rows)
    write_json(
        args.output_dir / "simulation_summary.json",
        {
            "status": "GO" if go else "NO_GO",
            "created_at": dt.datetime.now().astimezone().isoformat(),
            "offline_simulation_only": True,
            "not_a_measured_method_result": True,
            "dataset_dir": str(args.dataset_dir.resolve()),
            "model_path": str(args.model_path.resolve()),
            "model_sha256": sha256_file(args.model_path),
            "threshold_selection_status": selection_status,
            "threshold_selected_on": "dev_only",
            "selected_threshold": threshold,
            "deletion_risk_feature": args.deletion_risk_feature,
            "deletion_risk_threshold": args.deletion_risk_threshold,
            "dev": dev_evaluation,
            "test": test_evaluation,
            "test_gain_concentration": test_concentration,
            "test_top_triggered_cases": top_cases(test_policy_rows),
            "go_no_go_criteria": criteria,
            "go_no_go": "GO" if go else "NO_GO",
            "next_action": (
                "eligible_for_shadow_integration"
                if go
                else "stop_before_generate_pipeline_integration"
            ),
            "warnings": [
                "Threshold was never tuned on test.",
                "Historical gold-derived scores are used only to simulate the selected policy.",
                "The deterministic test split contains very few non-neutral labels.",
                "No statistical-significance claim is supported.",
            ],
        },
    )
    print(str(args.output_dir.resolve()))


if __name__ == "__main__":
    main()
