#!/usr/bin/env python3
"""CPU-only input checker for evidence text-packing generation diagnostics."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bridgekv.evidence_text_packing import estimate_token_count  # noqa: E402

DEFAULT_TASKS = "hotpotqa,musique,2wikimqa"
DEFAULT_VARIANTS = (
    "head_truncate_source_order,span_only_source_order,neighbor32_source_order,"
    "neighbor32_path_order,bridge_neighbor32_path_order,"
    "bridge_neighbor32_path_order_tags"
)


def _parse_list(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _latest_packed_dir() -> Path:
    candidates = sorted(
        (REPO_ROOT / "results").glob("evidence_text_packing_offline_*"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("No results/evidence_text_packing_offline_* directory found")
    return candidates[0]


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except Exception as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def _answer_metadata_leak(row: Dict[str, Any], packed_text: str) -> bool:
    lowered = packed_text.lower()
    if "gold_analysis_only_answers" in lowered:
        return True
    if "\nanswers:" in lowered or "\nanswer:" in lowered:
        return True
    answers = row.get("gold_analysis_only_answers", [])
    if answers and json.dumps(answers, ensure_ascii=False).lower() in lowered:
        return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packed_dir", default="")
    parser.add_argument("--tasks", default=DEFAULT_TASKS)
    parser.add_argument("--variants", default=DEFAULT_VARIANTS)
    parser.add_argument("--budget", type=int, default=1024)
    parser.add_argument("--max_samples", type=int, default=0)
    args = parser.parse_args()

    packed_dir = Path(args.packed_dir) if args.packed_dir else _latest_packed_dir()
    tasks = _parse_list(args.tasks)
    variants = _parse_list(args.variants)
    errors: List[str] = []
    warnings: List[str] = []
    checked_rows = 0
    exact_answer_mentions = 0

    if not packed_dir.exists():
        errors.append(f"packed_dir does not exist: {packed_dir}")
    else:
        for variant in variants:
            for task in tasks:
                path = (
                    packed_dir / "packed_prompts" / variant /
                    f"budget_{args.budget}" / f"{task}.jsonl"
                )
                if not path.exists():
                    errors.append(f"missing packed prompt file: {path}")
                    continue
                rows = _read_jsonl(path)
                if args.max_samples > 0:
                    rows = rows[: args.max_samples]
                if not rows:
                    errors.append(f"empty packed prompt file: {path}")
                    continue
                for idx, row in enumerate(rows):
                    checked_rows += 1
                    prefix = f"{path}:{idx + 1}"
                    for field in ("question", "task", "sample_id", "variant", "budget"):
                        if row.get(field) in (None, ""):
                            errors.append(f"{prefix}: missing {field}")
                    if "answers" not in row:
                        errors.append(f"{prefix}: missing answers")
                    if "gold_analysis_only_answers" not in row:
                        errors.append(f"{prefix}: missing gold_analysis_only_answers")
                    if "original_context" not in row:
                        errors.append(f"{prefix}: missing original_context")
                    packed_text = row.get("packed_text") or row.get("packed_prompt")
                    if not isinstance(packed_text, str) or not packed_text.strip():
                        errors.append(f"{prefix}: missing packed_text/packed_prompt")
                        continue
                    if row.get("task") != task:
                        errors.append(f"{prefix}: task mismatch {row.get('task')} != {task}")
                    if row.get("variant") != variant:
                        errors.append(f"{prefix}: variant mismatch {row.get('variant')} != {variant}")
                    token_count = estimate_token_count(packed_text)
                    stored_count = int(row.get("packed_token_count", token_count) or token_count)
                    truncated = bool(row.get("truncated", False))
                    if token_count > args.budget and not truncated:
                        errors.append(
                            f"{prefix}: token_count {token_count} exceeds budget {args.budget}"
                        )
                    if stored_count > args.budget and not truncated:
                        errors.append(
                            f"{prefix}: stored packed_token_count {stored_count} exceeds budget {args.budget}"
                        )
                    if _answer_metadata_leak(row, packed_text):
                        errors.append(f"{prefix}: gold answer metadata appears in packed_text")
                    for answer in row.get("gold_analysis_only_answers", []) or []:
                        if isinstance(answer, str) and answer and answer.lower() in packed_text.lower():
                            exact_answer_mentions += 1
                            break

    if exact_answer_mentions:
        warnings.append(
            f"{exact_answer_mentions} rows contain an exact answer string in selected evidence; "
            "this is allowed as natural context evidence and is not selector leakage."
        )

    report = {
        "packed_dir": str(packed_dir),
        "tasks": tasks,
        "variants": variants,
        "budget": args.budget,
        "checked_rows": checked_rows,
        "errors": errors,
        "warnings": warnings,
        "status": "ok" if not errors else "failed",
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
