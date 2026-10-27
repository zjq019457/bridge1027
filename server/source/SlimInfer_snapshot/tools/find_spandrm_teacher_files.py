#!/usr/bin/env python3
"""Search result directories for SpanDRM teacher / span metadata files."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
KEYWORDS = (
    "selected_spans", "span_selection", "packed_spans", "teacher_spans",
    "evidence_spans", "offset", "structured", "span_drm", "teacher",
)
NAMES = (
    "selected_spans", "span_selection", "packed_spans", "teacher_spans",
    "evidence_spans", "offset_map", "tokenizer_offset", "structured",
    "_span_selected_contexts_structured.jsonl",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="results")
    parser.add_argument("--output", default="results/spandrm_teacher_file_search.csv")
    args = parser.parse_args()
    root = REPO_ROOT / args.root
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix not in {".json", ".jsonl", ".csv", ".md"}:
            continue
        name_l = path.name.lower()
        if not any(k in name_l for k in NAMES) and "span" not in name_l:
            continue
        weak = ""
        has_spans = False
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")[:8000]
            has_spans = "selected_spans" in text or "span_id" in text
            weak = str("weak_teacher" in text)
        except Exception:
            pass
        rows.append({
            "path": str(path.relative_to(REPO_ROOT)),
            "size_bytes": path.stat().st_size,
            "has_selected_spans": has_spans,
            "mentions_weak_teacher": weak,
        })
    out = REPO_ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "size_bytes", "has_selected_spans", "mentions_weak_teacher"])
        w.writeheader()
        w.writerows(rows)
    print(json.dumps({"found": len(rows), "output": str(out)}, indent=2))


if __name__ == "__main__":
    main()
