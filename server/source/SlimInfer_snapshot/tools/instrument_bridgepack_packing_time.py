#!/usr/bin/env python3
"""Optional CPU-only timing instrumentation for BridgePack packing.

This script does not run generation or load a model. It times the selector and
packing function before tokenizer-exact repacking. Use it only when a dedicated
packing-time measurement is needed.
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bridgekv.evidence_text_packing import pack_evidence  # noqa: E402
from tools.run_evidence_text_packing_offline import _candidate_dataset_roots, _load_samples  # noqa: E402


def _parse_list(value: str) -> List[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _mean(values: Sequence[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    idx = (len(vals) - 1) * q
    lo = int(idx)
    hi = min(lo + 1, len(vals) - 1)
    weight = idx - lo
    return vals[lo] * (1.0 - weight) + vals[hi] * weight


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default="qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,narrativeqa")
    parser.add_argument("--variants", default="span_only_source_order,bm25_chunk_source_order,tfidf_chunk_source_order,bridge_neighbor32_path_order,lexical_bridgepack_path_order")
    parser.add_argument("--budget", type=int, default=1024)
    parser.add_argument("--max_samples", type=int, default=32)
    parser.add_argument("--out_csv", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = _parse_list(args.tasks)
    variants = _parse_list(args.variants)
    roots = _candidate_dataset_roots("")
    detail_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    for variant in variants:
        durations: List[float] = []
        for task in tasks:
            for sample in _load_samples(task, roots, args.max_samples):
                start = time.perf_counter()
                pack_evidence(
                    str(sample.get("sample_id", "")),
                    str(sample.get("task", task)),
                    str(sample.get("question", "")),
                    str(sample.get("context", "")),
                    variant,
                    args.budget,
                )
                elapsed = time.perf_counter() - start
                durations.append(elapsed)
                detail_rows.append({
                    "variant": variant,
                    "task": task,
                    "sample_id": sample.get("sample_id", ""),
                    "packing_time_s": round(elapsed, 6),
                })
        summary_rows.append({
            "variant": variant,
            "budget": args.budget,
            "samples": len(durations),
            "packing_time_mean_s": round(_mean(durations), 6),
            "packing_time_p95_s": round(_percentile(durations, 0.95), 6),
            "instrumentation": "cpu_pack_evidence_only_no_generation_no_model",
        })
    out_csv = Path(args.out_csv)
    _write_csv(
        out_csv,
        summary_rows,
        ["variant", "budget", "samples", "packing_time_mean_s", "packing_time_p95_s", "instrumentation"],
    )
    detail_path = out_csv.with_name(out_csv.stem + "_detail.csv")
    _write_csv(detail_path, detail_rows, ["variant", "task", "sample_id", "packing_time_s"])
    print(f"OUT_CSV={out_csv}")
    print(f"DETAIL_CSV={detail_path}")


if __name__ == "__main__":
    main()
