#!/usr/bin/env python3
"""Build SpanDRM teacher microblock labels from span-selection JSONL."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PAGE_TOKENS = 64


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            yield json.loads(line)


def _microblocks_for_span(start: int, end: int, mb_size: int) -> List[Dict[str, Any]]:
    out = []
    pos = start
    rank = 0
    while pos < end:
        e = min(end, pos + mb_size)
        out.append({"start_token": pos, "end_token": e, "teacher_rank": rank})
        pos = e
        rank += 1
    return out


def build_labels(
    span_jsonl: Optional[Path],
    output: Path,
    microblock_size: int = 32,
    weak_fallback: bool = False,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    weak = weak_fallback
    if span_jsonl and span_jsonl.exists():
        for rec in _iter_jsonl(span_jsonl):
            task = str(rec.get("task", ""))
            sid = str(rec.get("sample_id", ""))
            spans = rec.get("selected_spans", [])
            if not spans and rec.get("selected_token_start") is not None:
                spans = [rec]
            for span in spans:
                start = int(span.get("selected_token_start", span.get("token_start", 0)) or 0)
                end = int(span.get("selected_token_end", span.get("token_end", start)) or start)
                page_id = int(span.get("page_id", start // PAGE_TOKENS))
                span_len = max(1, end - start)
                density = float(span.get("evidence_density", span.get("final_span_score", 0.0)) or 0.0)
                for rank, mb in enumerate(_microblocks_for_span(start, end, microblock_size)):
                    overlap = (min(end, mb["end_token"]) - max(start, mb["start_token"])) / max(1, mb["end_token"] - mb["start_token"])
                    rows.append({
                        "sample_id": sid,
                        "task": task,
                        "page_id": page_id,
                        "microblock_id": f"p{page_id}_mb{mb['start_token']//microblock_size}",
                        "start_token": mb["start_token"],
                        "end_token": mb["end_token"],
                        "teacher_positive": int(overlap >= 0.5),
                        "teacher_overlap_ratio": round(overlap, 6),
                        "teacher_rank": rank,
                        "evidence_density": density,
                        "weak_teacher": False,
                    })
    if not rows:
        weak = True
        for page_id in range(4):
            for mb_idx in range(2):
                start = page_id * PAGE_TOKENS + mb_idx * microblock_size
                end = min(start + microblock_size, (page_id + 1) * PAGE_TOKENS)
                rows.append({
                    "sample_id": "synthetic_smoke",
                    "task": "hotpotqa",
                    "page_id": page_id,
                    "microblock_id": f"p{page_id}_mb{mb_idx}",
                    "start_token": start,
                    "end_token": end,
                    "teacher_positive": int(mb_idx == 0),
                    "teacher_overlap_ratio": 1.0 if mb_idx == 0 else 0.1,
                    "teacher_rank": mb_idx,
                    "evidence_density": 0.5,
                    "weak_teacher": True,
                })
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"num_labels": len(rows), "weak_teacher": weak, "output": str(output)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--span_jsonl", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--microblock_size", type=int, default=32, choices=[16, 32, 64])
    parser.add_argument("--weak_fallback", action="store_true")
    args = parser.parse_args()
    span_path = Path(args.span_jsonl) if args.span_jsonl else None
    meta = build_labels(span_path, Path(args.output), args.microblock_size, args.weak_fallback)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
