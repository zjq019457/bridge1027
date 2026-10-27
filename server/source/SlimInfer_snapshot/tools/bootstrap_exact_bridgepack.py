#!/usr/bin/env python3
"""Paired bootstrap CIs for exact-token BridgePack predictions."""
from __future__ import annotations

import argparse
import csv
import json
import re
import random
import string
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _parse_list(value: str) -> List[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _fmt(value: Any, digits: int = 4) -> str:
    if value in ("", None):
        return "TODO"
    try:
        return f"{float(value):.{digits}f}".rstrip("0").rstrip(".")
    except (TypeError, ValueError):
        return "TODO"


def _normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def _f1_score(prediction_tokens: Sequence[str], ground_truth_tokens: Sequence[str]) -> float:
    from collections import Counter

    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision = same / max(1, len(prediction_tokens))
    recall = same / max(1, len(ground_truth_tokens))
    return (2 * precision * recall) / max(1e-12, precision + recall)


def _qa_f1_score(prediction: str, ground_truth: str, **_: Any) -> float:
    return _f1_score(_normalize_answer(prediction).split(), _normalize_answer(ground_truth).split())


def _metric_map() -> Tuple[Dict[str, Any], str]:
    try:
        from longbench_eval import dataset2metric

        return dict(dataset2metric), "longbench_eval.dataset2metric"
    except Exception as exc:
        english_qa_tasks = [
            "narrativeqa",
            "qasper",
            "multifieldqa_en",
            "hotpotqa",
            "2wikimqa",
            "musique",
        ]
        return {task: _qa_f1_score for task in english_qa_tasks}, (
            "local English QA-F1 fallback matching longbench_eval.qa_f1_score "
            f"because importing longbench_eval failed with {exc!r}"
        )


def _score_prediction(dataset: str, row: Mapping[str, Any], metric: Any) -> float:
    prediction = str(row.get("pred", ""))
    if dataset in ["trec", "triviaqa", "samsum", "lsht"] and prediction:
        prediction = prediction.lstrip("\n").split("\n")[0]
    answers = row.get("answers", [])
    if not isinstance(answers, list):
        answers = [answers]
    if not prediction:
        return 0.0
    best = 0.0
    for answer in answers:
        best = max(best, float(metric(prediction, answer, all_classes=row.get("all_classes"))))
    return 100.0 * best


def _load_scores(exact_dir: Path, method: str, metric_map: Mapping[str, Any]) -> Dict[str, Dict[str, float]]:
    by_task: Dict[str, Dict[str, float]] = {}
    pred_root = exact_dir / "predictions" / method
    for path in sorted(pred_root.glob("*.jsonl")):
        task = path.stem
        if task not in metric_map:
            continue
        scores: Dict[str, float] = {}
        for row in _read_jsonl(path):
            sample_id = str(row.get("sample_id", row.get("sample_index", "")))
            if not sample_id:
                continue
            scores[sample_id] = _score_prediction(task, row, metric_map[task])
        if scores:
            by_task[task] = scores
    return by_task


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _task_macro(scores_by_task: Mapping[str, Mapping[str, float]]) -> float:
    task_means = [_mean(list(scores.values())) for scores in scores_by_task.values() if scores]
    return _mean(task_means)


def _paired_delta(
    main_scores: Mapping[str, Mapping[str, float]],
    base_scores: Mapping[str, Mapping[str, float]],
    rng: random.Random | None = None,
) -> float:
    task_deltas: List[float] = []
    for task in sorted(main_scores):
        if task not in base_scores:
            continue
        ids = sorted(set(main_scores[task]) & set(base_scores[task]))
        if not ids:
            continue
        if rng is not None:
            ids = [rng.choice(ids) for _ in ids]
        diffs = [main_scores[task][sample_id] - base_scores[task][sample_id] for sample_id in ids]
        task_deltas.append(_mean(diffs))
    return _mean(task_deltas)


def _paired_count(main_scores: Mapping[str, Mapping[str, float]], base_scores: Mapping[str, Mapping[str, float]]) -> Tuple[int, int]:
    tasks = 0
    samples = 0
    for task in sorted(main_scores):
        if task not in base_scores:
            continue
        overlap = set(main_scores[task]) & set(base_scores[task])
        if overlap:
            tasks += 1
            samples += len(overlap)
    return tasks, samples


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    idx = (len(vals) - 1) * q
    lo = int(idx)
    hi = min(lo + 1, len(vals) - 1)
    weight = idx - lo
    return vals[lo] * (1.0 - weight) + vals[hi] * weight


def _todo_outputs(out_csv: Path, out_md: Path, reason: str, baselines: Sequence[str], main_method: str) -> None:
    rows = [
        {
            "main_method": main_method,
            "baseline": baseline,
            "status": "TODO",
            "reason": reason,
        }
        for baseline in baselines
    ]
    _write_csv(
        out_csv,
        rows,
        [
            "main_method", "baseline", "status", "reason", "paired_tasks", "paired_samples",
            "main_avg", "baseline_avg", "paired_delta", "ci_low", "ci_high", "p_delta_le_0",
        ],
    )
    _write_text(
        out_md,
        "\n".join([
            "# Exact-Token Paired Bootstrap CI",
            "",
            "Bootstrap CI was not computed.",
            "",
            f"Reason: {reason}",
            "",
            "TODO: provide per-sample prediction/score files for paired bootstrap.",
        ]),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exact_dir", required=True)
    parser.add_argument("--main_method", default="lexical_bridgepack_path_order")
    parser.add_argument(
        "--baselines",
        default="span_only_source_order,bm25_chunk_source_order,tfidf_chunk_source_order,bridge_neighbor32_path_order",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--out_md", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exact_dir = Path(args.exact_dir)
    out_csv = Path(args.out_csv)
    out_md = Path(args.out_md)
    baselines = _parse_list(args.baselines)
    dataset2metric, metric_source = _metric_map()

    if not (exact_dir / "predictions" / args.main_method).exists():
        _todo_outputs(out_csv, out_md, f"Missing predictions for main method {args.main_method}", baselines, args.main_method)
        return
    main_scores = _load_scores(exact_dir, args.main_method, dataset2metric)
    if not main_scores:
        _todo_outputs(out_csv, out_md, f"No per-sample scores could be computed for {args.main_method}", baselines, args.main_method)
        return

    rng = random.Random(args.seed)
    rows: List[Dict[str, Any]] = []
    for baseline in baselines:
        base_scores = _load_scores(exact_dir, baseline, dataset2metric)
        paired_tasks, paired_samples = _paired_count(main_scores, base_scores)
        if not base_scores or paired_samples == 0:
            rows.append({
                "main_method": args.main_method,
                "baseline": baseline,
                "status": "TODO",
                "reason": "Missing aligned per-sample predictions or scores.",
                "paired_tasks": paired_tasks,
                "paired_samples": paired_samples,
            })
            continue
        observed = _paired_delta(main_scores, base_scores)
        boot = [_paired_delta(main_scores, base_scores, rng) for _ in range(args.resamples)]
        rows.append({
            "main_method": args.main_method,
            "baseline": baseline,
            "status": "ok",
            "reason": "",
            "paired_tasks": paired_tasks,
            "paired_samples": paired_samples,
            "main_avg": _fmt(_task_macro(main_scores)),
            "baseline_avg": _fmt(_task_macro(base_scores)),
            "paired_delta": _fmt(observed),
            "ci_low": _fmt(_percentile(boot, 0.025)),
            "ci_high": _fmt(_percentile(boot, 0.975)),
            "p_delta_le_0": _fmt(sum(1 for value in boot if value <= 0.0) / max(1, len(boot)), digits=6),
        })
    fields = [
        "main_method", "baseline", "status", "reason", "paired_tasks", "paired_samples",
        "main_avg", "baseline_avg", "paired_delta", "ci_low", "ci_high", "p_delta_le_0",
    ]
    _write_csv(out_csv, rows, fields)
    lines = [
        "# Exact-Token Paired Bootstrap CI",
        "",
        f"- Main method: `{args.main_method}`.",
        f"- Resamples: `{args.resamples}`.",
        f"- Seed: `{args.seed}`.",
        "- Unit: paired sample ids within each task; each bootstrap replicate averages paired deltas by task, then macro-averages tasks.",
        f"- Metric source: `{metric_source}`.",
        "- Scores are recomputed from existing prediction JSONL files; no model was loaded.",
        "",
        "| baseline | paired samples | paired delta | 95% CI | p(delta <= 0) | status |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        if row.get("status") == "ok":
            ci = f"[{row['ci_low']}, {row['ci_high']}]"
            lines.append(
                f"| `{row['baseline']}` | {row['paired_samples']} | {row['paired_delta']} | {ci} | {row['p_delta_le_0']} | ok |"
            )
        else:
            lines.append(
                f"| `{row['baseline']}` | {row.get('paired_samples', 0)} | TODO | TODO | TODO | {row.get('reason', 'TODO')} |"
            )
    _write_text(out_md, "\n".join(lines))
    print(f"OUT_CSV={out_csv}")
    print(f"OUT_MD={out_md}")


if __name__ == "__main__":
    main()
