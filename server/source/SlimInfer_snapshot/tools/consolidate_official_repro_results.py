#!/usr/bin/env python3
"""Consolidate official SlimInfer reproduction outputs into standard report files."""
import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from longbench_eval import dataset2metric, scorer  # noqa: E402


def parse_targets(value: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for chunk in value.split(","):
        chunk = chunk.strip()
        if not chunk or ":" not in chunk:
            continue
        task, score = chunk.split(":", 1)
        out[task.strip()] = float(score.strip())
    return out


def per_sample_score(dataset: str, pred: str, answers: List[str], all_classes: Any) -> float:
    if not pred:
        return 0.0
    if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
        pred = pred.lstrip("\n").split("\n")[0]
    metric_fn = dataset2metric[dataset]
    best = 0.0
    for gt in answers:
        best = max(best, float(metric_fn(pred, gt, all_classes=all_classes)))
    return round(100.0 * best, 4)


def load_predictions(pred_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(pred_dir.glob("*.jsonl")):
        dataset = path.stem
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                rows.append({
                    "task": dataset,
                    "sample_id": row.get("sample_id", row.get("sample_index")),
                    "sample_index": row.get("sample_index"),
                    "pred": row.get("pred", ""),
                    "answers": row.get("answers", []),
                    "all_classes": row.get("all_classes", []),
                    "length": int(row.get("length", 0) or 0),
                })
    return rows


def write_csv(path: Path, rows: List[Dict[str, Any]], fields: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--paper_targets", type=str, default="hotpotqa:56.94,musique:27.25")
    parser.add_argument("--tolerance", type=float, default=3.0)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    pred_dir = run_dir / "longbench_predictions"
    if not pred_dir.exists():
        raise FileNotFoundError(f"Missing predictions dir: {pred_dir}")

    paper_targets = parse_targets(args.paper_targets)
    raw_rows = load_predictions(pred_dir)

    # Sample count audit
    counts: Dict[str, int] = {}
    for row in raw_rows:
        counts[row["task"]] = counts.get(row["task"], 0) + 1

    expected = {"hotpotqa": 200, "musique": 200}
    mismatch = {k: (counts.get(k, 0), v) for k, v in expected.items() if counts.get(k, 0) != v}
    if mismatch:
        md = run_dir / "SAMPLE_COUNT_MISMATCH.md"
        md.write_text(
            "# SAMPLE_COUNT_MISMATCH\n\n"
            f"Expected hotpotqa=200, musique=200.\n\n"
            f"Actual counts: {counts}\n\n"
            f"Mismatches: {mismatch}\n",
            encoding="utf-8",
        )
        print(f"Wrote {md}")
        sys.exit(2)

    sample_rows: List[Dict[str, Any]] = []
    by_task_preds: Dict[str, List[str]] = {}
    by_task_answers: Dict[str, List[Any]] = {}
    by_task_lengths: Dict[str, List[int]] = {}
    by_task_classes: Dict[str, Any] = {}

    predictions_jsonl = run_dir / "predictions.jsonl"
    with predictions_jsonl.open("w", encoding="utf-8") as out_f:
        for row in raw_rows:
            task = row["task"]
            score = per_sample_score(task, row["pred"], row["answers"], row["all_classes"])
            enriched = {**row, "score": score}
            sample_rows.append(enriched)
            out_f.write(json.dumps(enriched, ensure_ascii=False) + "\n")

            by_task_preds.setdefault(task, []).append(row["pred"])
            by_task_answers.setdefault(task, []).append(row["answers"])
            by_task_lengths.setdefault(task, []).append(row["length"])
            if row["all_classes"]:
                by_task_classes[task] = row["all_classes"]

    write_csv(
        run_dir / "scores_by_sample.csv",
        sample_rows,
        ["task", "sample_id", "sample_index", "score", "length", "pred"],
    )

    task_rows: List[Dict[str, Any]] = []
    for task in sorted(by_task_preds.keys()):
        classes = by_task_classes.get(task, [])
        task_score = scorer(
            task,
            by_task_preds[task],
            by_task_answers[task],
            by_task_lengths[task],
            classes,
        )
        target = paper_targets.get(task)
        delta = round(float(task_score) - float(target), 2) if target is not None else ""
        task_rows.append({
            "task": task,
            "score": task_score,
            "num_samples": len(by_task_preds[task]),
            "paper_target": target if target is not None else "",
            "delta_vs_paper": delta,
        })

    write_csv(
        run_dir / "scores_by_task.csv",
        task_rows,
        ["task", "score", "num_samples", "paper_target", "delta_vs_paper"],
    )

    macro = round(sum(float(r["score"]) for r in task_rows) / len(task_rows), 2) if task_rows else None
    summary_rows = task_rows + [{
        "task": "macro_avg",
        "score": macro,
        "num_samples": sum(int(r["num_samples"]) for r in task_rows),
        "paper_target": "",
        "delta_vs_paper": "",
    }]
    write_csv(
        run_dir / "scores_summary.csv",
        summary_rows,
        ["task", "score", "num_samples", "paper_target", "delta_vs_paper"],
    )

    pass_tasks = []
    fail_tasks = []
    for row in task_rows:
        target = paper_targets.get(row["task"])
        if target is None:
            continue
        delta = abs(float(row["score"]) - float(target))
        if delta <= args.tolerance:
            pass_tasks.append((row["task"], row["score"], target, delta))
        else:
            fail_tasks.append((row["task"], row["score"], target, delta))

    overall_pass = len(fail_tasks) == 0 and len(pass_tasks) == len(paper_targets)

    report_lines = [
        "# Two-Task Reproduction Report",
        "",
        f"Run directory: `{run_dir}`",
        f"Method: silm_original (`sliminfer_original`)",
        f"Tolerance: ±{args.tolerance} vs paper Table 1",
        "",
        "## Sample counts",
        "",
        "| task | count | expected |",
        "|------|-------|----------|",
    ]
    for task, exp in expected.items():
        report_lines.append(f"| {task} | {counts.get(task, 0)} | {exp} |")

    report_lines += ["", "## Task scores", "", "| task | score | paper | delta | status |", "|------|-------|-------|-------|--------|"]
    for row in task_rows:
        target = paper_targets.get(row["task"])
        if target is None:
            continue
        delta = float(row["delta_vs_paper"])
        status = "PASS" if abs(delta) <= args.tolerance else "FAIL"
        report_lines.append(
            f"| {row['task']} | {row['score']} | {target} | {delta:+.2f} | {status} |"
        )
    report_lines += ["", f"**Macro avg (2 tasks): {macro}**", ""]
    (run_dir / "TWO_TASK_REPRODUCTION_REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    verdict = "PASS" if overall_pass else "FAIL"
    verdict_lines = [
        f"# REPRODUCTION_{verdict}",
        "",
        f"Overall: **{verdict}**",
        "",
        "## Passed",
    ]
    for t in pass_tasks:
        verdict_lines.append(f"- {t[0]}: {t[1]} (target {t[2]}, |Δ|={t[3]:.2f})")
    verdict_lines.append("")
    verdict_lines.append("## Failed")
    if fail_tasks:
        for t in fail_tasks:
            verdict_lines.append(f"- {t[0]}: {t[1]} (target {t[2]}, |Δ|={t[3]:.2f})")
    else:
        verdict_lines.append("- (none)")
    verdict_lines += [
        "",
        "## Next step",
        "- PASS → proceed to Phase 3 RoleFlow two-task comparison",
        "- FAIL → do NOT run RoleFlow; fix official path first",
    ]
    (run_dir / "REPRODUCTION_PASS_OR_FAIL.md").write_text("\n".join(verdict_lines) + "\n", encoding="utf-8")

    print(f"Wrote reports to {run_dir}")
    print(f"Verdict: {verdict}")


if __name__ == "__main__":
    main()
