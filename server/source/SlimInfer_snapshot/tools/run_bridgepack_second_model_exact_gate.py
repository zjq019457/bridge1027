#!/usr/bin/env python3
"""Run a small second-model exact-token BridgePack gate."""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]


def _parse_list(value: str) -> List[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


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


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(value: Any) -> float | None:
    try:
        if value in ("", None):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt(value: Any) -> str:
    number = _float(value)
    if number is None:
        return "TODO"
    return f"{number:.4f}".rstrip("0").rstrip(".")


def _run(cmd: Sequence[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("$ " + " ".join(str(part) for part in cmd) + "\n")
        log.flush()
        proc = subprocess.run(list(cmd), cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, text=True, check=False)
        log.write(f"returncode={proc.returncode}\n")
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed rc={proc.returncode}: {' '.join(str(x) for x in cmd)}. See {log_path}")


def _missing_or_failed_rows(run_dir: Path, variants: Sequence[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for row in _read_csv(run_dir / "paper_tables" / "00_missing_samples.csv"):
        rows.append({
            "issue_type": "sample_id_mismatch",
            "task": row.get("task", ""),
            "method": row.get("variant", row.get("method", "")),
            "expected_count": "",
            "actual_count": "",
            "missing_count": row.get("missing_count", ""),
            "extra_count": row.get("extra_count", ""),
            "details": f"missing={row.get('missing_sample_ids', '')}; extra={row.get('extra_sample_ids', '')}",
        })
    for row in _read_csv(run_dir / "paper_tables" / "00_model_token_budget_stats.csv"):
        method = row.get("variant", row.get("method", ""))
        if method not in variants:
            continue
        task = row.get("task", "")
        expected = int(row.get("num_samples", 0) or 0)
        pred_path = run_dir / "predictions" / method / f"{task}.jsonl"
        actual = 0
        if pred_path.exists():
            with pred_path.open(encoding="utf-8") as handle:
                actual = sum(1 for line in handle if line.strip())
        if actual != expected:
            rows.append({
                "issue_type": "prediction_count_mismatch",
                "task": task,
                "method": method,
                "expected_count": expected,
                "actual_count": actual,
                "missing_count": max(0, expected - actual),
                "extra_count": max(0, actual - expected),
                "details": str(pred_path),
            })
    return rows


def _budget_summary(run_dir: Path) -> Dict[str, Any]:
    rows = _read_csv(run_dir / "paper_tables" / "00_model_token_budget_stats.csv")
    missing = _read_csv(run_dir / "paper_tables" / "00_missing_samples.csv")
    return {
        "variant_task_rows": len(rows),
        "variant_sample_rows": sum(int(row.get("num_samples", 0) or 0) for row in rows),
        "over_budget": sum(int(row.get("over_budget_samples", 0) or 0) for row in rows),
        "max_token": max((int(row.get("model_token_count_max", 0) or 0) for row in rows), default=0),
        "same_samples": not missing,
    }


def _aggregate(run_dir: Path, args: argparse.Namespace) -> None:
    paper_dir = run_dir / "paper_tables"
    main = _read_csv(run_dir / "paper_tables" / "01_exact_budget_main_table_1024.csv")
    by_method = {row["method"]: row for row in main}
    span = _float(by_method.get("span_only_source_order", {}).get("avg"))
    bm25 = _float(by_method.get("bm25_chunk_source_order", {}).get("avg"))
    tfidf = _float(by_method.get("tfidf_chunk_source_order", {}).get("avg"))
    budget_summary = _budget_summary(run_dir)
    rows: List[Dict[str, Any]] = []
    for row in main:
        avg = _float(row.get("avg"))
        rows.append({
            "method": row.get("method", ""),
            "avg": row.get("avg", "TODO"),
            "delta_vs_span_only": _fmt(avg - span) if avg is not None and span is not None else "TODO",
            "delta_vs_bm25": _fmt(avg - bm25) if avg is not None and bm25 is not None else "TODO",
            "delta_vs_tfidf": _fmt(avg - tfidf) if avg is not None and tfidf is not None else "TODO",
        })
    _write_text(paper_dir / "00_second_model_fairness.md", "\n".join([
        "# Second-Model Exact Gate Fairness",
        "",
        "This run uses a second-model tokenizer for exact packed-context budgeting.",
        "",
        "## Required Checks",
        "",
        f"- Model/tag: `{args.model_name_or_path}` / `{args.model_tag}`.",
        f"- Tasks: `{args.tasks}`.",
        f"- Max samples per task: `{args.max_samples}`.",
        f"- Packed-context budget: `{args.budget}` second-model tokenizer tokens.",
        f"- Variants: `{args.variants}`.",
        "- Same prompt wrapper, generation config, and scoring pipeline are used across variants.",
        "- Gold answers are carried only for scoring and are not used for packing selection.",
        "",
        "Inspect `00_model_token_budget_stats.csv` and `00_missing_samples.csv` for the low-level audit.",
        "",
        "## Low-Level Audit Summary",
        "",
        f"- Variant-task rows: `{budget_summary['variant_task_rows']}`.",
        f"- Variant-sample rows: `{budget_summary['variant_sample_rows']}`.",
        f"- Over-budget packed contexts: `{budget_summary['over_budget']}`.",
        f"- Maximum packed-context token count: `{budget_summary['max_token']}`.",
        f"- Same sample ids across variants/tasks: **{budget_summary['same_samples']}**.",
    ]))
    _write_csv(paper_dir / "01_second_model_gate_table.csv", rows, ["method", "avg", "delta_vs_span_only", "delta_vs_bm25", "delta_vs_tfidf"])
    _write_csv(paper_dir / "02_delta_vs_span_only.csv", rows, ["method", "avg", "delta_vs_span_only"])
    _write_csv(paper_dir / "03_delta_vs_bm25_tfidf.csv", rows, ["method", "avg", "delta_vs_bm25", "delta_vs_tfidf"])
    _write_csv(
        paper_dir / "06_second_model_missing_or_failed_samples.csv",
        _missing_or_failed_rows(run_dir, _parse_list(args.variants)),
        ["issue_type", "task", "method", "expected_count", "actual_count", "missing_count", "extra_count", "details"],
    )
    lexical = by_method.get("lexical_bridgepack_path_order", {})
    lex_avg = _float(lexical.get("avg"))
    span_avg = span
    lex_beats_span = lex_avg is not None and span_avg is not None and lex_avg > span_avg
    lex_beats_bm25 = lex_avg is not None and bm25 is not None and lex_avg > bm25
    lex_beats_tfidf = lex_avg is not None and tfidf is not None and lex_avg > tfidf
    lex_beats_lexical = lex_beats_bm25 and lex_beats_tfidf
    include_level = "main paper small gate paragraph" if lex_beats_span else "appendix or limitation only"
    safe_claim = (
        "Lexical-BridgePack improves over span-only in this 32-sample second-model multi-hop gate."
        if lex_beats_span else
        "The second-model gate does not support a generalization claim over span-only."
    )
    if lex_beats_span and not lex_beats_lexical:
        safe_claim += " Lexical baselines remain competitive on the gate."
    if lex_beats_lexical:
        safe_claim += " It also shows a positive average trend over BM25 and TF-IDF; use bootstrap before claiming significance."
    _write_text(paper_dir / "04_second_model_recommendation.md", "\n".join([
        "# Second-Model Exact Gate Recommendation",
        "",
        f"- Lexical-BridgePack avg: `{lexical.get('avg', 'TODO')}`.",
        "",
        "## Required Answers",
        "",
        f"1. Lexical-BridgePack beats span-only on the second model: **{lex_beats_span}**.",
        f"2. Lexical-BridgePack beats BM25/TF-IDF on average: **{lex_beats_lexical}**.",
        f"3. Consistent with Qwen: **{lex_beats_span}** for the span-only comparison; lexical-baseline consistency depends on the deltas above.",
        f"4. Include level: **{include_level}**.",
        f"5. Safe claim: {safe_claim}",
        "",
        "Treat this as a 32-sample multi-hop gate, not a full second-model validation.",
    ]))
    _write_text(paper_dir / "05_second_model_claims.md", "\n".join([
        "# Second-Model Exact Gate Claims",
        "",
        "- This is a small exact-token second-model gate, not a full transfer validation.",
        "- Use only completed rows from `01_second_model_gate_table.csv`.",
        "- Do not claim broad model generalization from this gate alone.",
        "- Keep the Qwen exact-token 1024 table as the official main result unless the paper is explicitly expanded to multi-model results.",
    ]))
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    config = {
        "model_name_or_path": args.model_name_or_path,
        "model_tag": args.model_tag,
        "tokenizer_class": tokenizer.__class__.__name__,
        "dtype": args.dtype,
        "device": args.device,
        "tasks": _parse_list(args.tasks),
        "max_samples": args.max_samples,
        "budget": args.budget,
        "variants": _parse_list(args.variants),
        "seed": args.seed,
    }
    (paper_dir / "00_second_model_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def write_todo(path: Path, *, model_name_or_path: str = "", model_tag: str = "", error: str = "", retry_command: str = "") -> None:
    lines = [
        "# Second-Model Validation TODO",
        "",
        "- Second-model validation remains TODO.",
    ]
    if model_name_or_path:
        lines.append(f"- Attempted model: `{model_name_or_path}` (`{model_tag}`).")
    if error:
        lines.extend(["", "## Error Summary", "", "```text", error.strip()[:4000], "```"])
    if retry_command:
        lines.extend(["", "## Retry Command", "", "```bash", retry_command, "```"])
    lines.extend([
        "",
        "- No manual git clone is required now.",
        "- If HuggingFace auth is required, run `huggingface-cli login` first.",
        "- Use a local Hugging Face model path if available; otherwise Transformers may download into the normal HF cache.",
        "- Avoid git clone unless the provider explicitly requires git-lfs cloning.",
    ])
    _write_text(path, "\n".join(lines))


def parse_args() -> argparse.Namespace:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--model_tag", required=True)
    parser.add_argument("--tasks", default="hotpotqa,2wikimqa,musique")
    parser.add_argument("--max_samples", type=int, default=32)
    parser.add_argument("--budget", type=int, default=1024)
    parser.add_argument("--variants", default="span_only_source_order,bm25_chunk_source_order,tfidf_chunk_source_order,bridge_neighbor32_path_order,lexical_bridgepack_path_order")
    parser.add_argument("--out_dir", default=f"results/bridgepack_second_model_exact_gate_model_{stamp}")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "run_bridgepack_exact_budget_repack.py"),
        "--model_name_or_path", args.model_name_or_path,
        "--tasks", args.tasks,
        "--budget", str(args.budget),
        "--max_samples", str(args.max_samples),
        "--variants", args.variants,
        "--out_dir", str(out_dir),
        "--seed", str(args.seed),
        "--dtype", args.dtype,
        "--device", args.device,
    ]
    if args.resume:
        cmd.append("--resume")
    retry_command = " ".join(str(part) for part in [sys.executable, __file__, *sys.argv[1:]])
    try:
        _run(cmd, out_dir / "logs" / "second_model_gate.log")
        _aggregate(out_dir, args)
        print(f"OUT_DIR={out_dir}")
    except Exception as exc:
        error = "".join(traceback.format_exception_only(type(exc), exc))
        detail = traceback.format_exc()
        paper_todo = REPO_ROOT / "paper" / "aaai2027_bridgepack_exact" / "second_model_todo.md"
        write_todo(
            paper_todo,
            model_name_or_path=args.model_name_or_path,
            model_tag=args.model_tag,
            error=detail,
            retry_command=retry_command,
        )
        fail_dir = out_dir / "paper_tables"
        fail_dir.mkdir(parents=True, exist_ok=True)
        _write_text(fail_dir / "04_second_model_recommendation.md", "\n".join([
            "# Second-Model Exact Gate Recommendation",
            "",
            "Second-model gate did not complete.",
            "",
            "## Error Summary",
            "",
            "```text",
            error.strip(),
            "```",
            "",
            "See `paper/aaai2027_bridgepack_exact/second_model_todo.md` for the retry command.",
        ]))
        print(f"SECOND_MODEL_TODO={paper_todo}")


if __name__ == "__main__":
    main()
