#!/usr/bin/env python3
"""Bootstrap paired deltas for the second-model BridgePack gate."""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

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


def _score_row(task: str, row: Mapping[str, Any]) -> float:
    from longbench_eval import scorer

    return float(scorer(
        task,
        [str(row.get("pred", ""))],
        [row.get("answers", [])],
        [int(row.get("length", row.get("packed_token_count", 0)) or 0)],
        row.get("all_classes", None),
    ))


def _paired_scores(second_dir: Path, main_method: str, baseline: str) -> List[float]:
    diffs: List[float] = []
    pred_root = second_dir / "predictions"
    tasks = sorted({path.stem for path in (pred_root / main_method).glob("*.jsonl")})
    for task in tasks:
        main_rows = {str(row.get("sample_id")): row for row in _read_jsonl(pred_root / main_method / f"{task}.jsonl")}
        base_rows = {str(row.get("sample_id")): row for row in _read_jsonl(pred_root / baseline / f"{task}.jsonl")}
        common = sorted(set(main_rows) & set(base_rows))
        for sample_id in common:
            diffs.append(_score_row(task, main_rows[sample_id]) - _score_row(task, base_rows[sample_id]))
    return diffs


def _bootstrap(values: Sequence[float], seed: int, resamples: int) -> tuple[float, float, float]:
    if not values:
        raise ValueError("No paired values available for bootstrap.")
    rng = random.Random(seed)
    n = len(values)
    draws: List[float] = []
    for _ in range(resamples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        draws.append(sum(sample) / n)
    draws.sort()
    lo = draws[int(0.025 * (len(draws) - 1))]
    hi = draws[int(0.975 * (len(draws) - 1))]
    return sum(values) / n, lo, hi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--second_dir", required=True)
    parser.add_argument("--main_method", default="lexical_bridgepack_path_order")
    parser.add_argument("--baselines", default="span_only_source_order,bm25_chunk_source_order,tfidf_chunk_source_order,bridge_neighbor32_path_order")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--out_csv", default="")
    parser.add_argument("--out_md", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    second_dir = Path(args.second_dir)
    out_csv = Path(args.out_csv) if args.out_csv else second_dir / "paper_tables" / "07_second_model_bootstrap_ci.csv"
    out_md = Path(args.out_md) if args.out_md else second_dir / "paper_tables" / "07_second_model_bootstrap_ci.md"
    baselines = _parse_list(args.baselines)
    rows: List[Dict[str, Any]] = []
    todo_lines: List[str] = []
    for baseline in baselines:
        try:
            diffs = _paired_scores(second_dir, args.main_method, baseline)
            mean, lo, hi = _bootstrap(diffs, args.seed, args.resamples)
            rows.append({
                "main_method": args.main_method,
                "baseline": baseline,
                "paired_samples": len(diffs),
                "delta_mean": round(mean, 4),
                "ci_low": round(lo, 4),
                "ci_high": round(hi, 4),
                "resamples": args.resamples,
                "seed": args.seed,
                "status": "ok",
            })
        except Exception as exc:  # Keep this diagnostic non-fabricating.
            rows.append({
                "main_method": args.main_method,
                "baseline": baseline,
                "paired_samples": 0,
                "delta_mean": "",
                "ci_low": "",
                "ci_high": "",
                "resamples": args.resamples,
                "seed": args.seed,
                "status": f"TODO: {exc!r}",
            })
            todo_lines.append(f"- `{baseline}`: TODO, {exc!r}")
    _write_csv(out_csv, rows, ["main_method", "baseline", "paired_samples", "delta_mean", "ci_low", "ci_high", "resamples", "seed", "status"])
    lines = [
        "# Second-Model Bootstrap CI",
        "",
        "Paired bootstrap over available aligned per-sample predictions. This is a 32-sample-per-task gate, not a full transfer validation.",
        "",
    ]
    for row in rows:
        if row["status"] == "ok":
            significant = float(row["ci_low"]) > 0 or float(row["ci_high"]) < 0
            lines.append(
                f"- `{args.main_method}` vs `{row['baseline']}`: delta `{row['delta_mean']}`, "
                f"CI [`{row['ci_low']}`, `{row['ci_high']}`], n=`{row['paired_samples']}`, "
                f"CI excludes 0: **{significant}**."
            )
    if todo_lines:
        lines.extend(["", "## TODO / Unavailable", "", *todo_lines])
    _write_text(out_md, "\n".join(lines))
    print(f"OUT_CSV={out_csv}")
    print(f"OUT_MD={out_md}")


if __name__ == "__main__":
    main()
