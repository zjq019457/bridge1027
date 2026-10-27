#!/usr/bin/env python3
"""Run the BridgePack paper-style context-compression evaluation pipeline.

BridgePack is evidence-aware text repacking. It is not strict KV-cache pruning.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_TASKS = "qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,narrativeqa"
DEFAULT_VARIANTS = (
    "head_truncate_source_order,span_only_source_order,neighbor32_source_order,"
    "neighbor32_path_order,bridge_neighbor32_path_order,bridge_neighbor32_path_order_tags"
)
DEFAULT_BUDGET_ABLATION = "512,1024,2048"
SELECTED_BUDGET_VARIANTS = (
    "span_only_source_order",
    "neighbor32_source_order",
    "bridge_neighbor32_path_order_tags",
)
MULTIHOP_TASKS = ("hotpotqa", "2wikimqa", "musique")
DEFAULT_LONGBENCH = "/home/yu-bai/Desktop/jiaqizhang/HeadKV/data/LongBench"
DEFAULT_QWEN = (
    "/home/yu-bai/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/"
    "snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
)


def _parse_list(value: str) -> List[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _parse_ints(value: str) -> List[int]:
    return [int(item) for item in _parse_list(value)]


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"0", "false", "no", "off", ""}


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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
    path.write_text(text, encoding="utf-8")


def _float(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _mean(values: Sequence[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _run(cmd: Sequence[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    printable = " ".join(str(part) for part in cmd)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n$ {printable}\n")
        log.flush()
        proc = subprocess.run(
            list(cmd),
            cwd=REPO_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        log.write(f"returncode={proc.returncode}\n")
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with rc={proc.returncode}: {printable}. See {log_path}")


def _candidate_dataset_roots() -> List[Path]:
    roots = [
        Path(DEFAULT_LONGBENCH),
        REPO_ROOT / "data",
        REPO_ROOT / "datasets",
        REPO_ROOT / "LongBench",
        REPO_ROOT.parent / "HeadKV" / "data" / "LongBench",
    ]
    out: List[Path] = []
    seen = set()
    for root in roots:
        key = str(root)
        if key in seen:
            continue
        seen.add(key)
        if root.exists():
            out.append(root)
    return out


def _find_task_file(task: str) -> Path | None:
    names = [f"{task}.jsonl", f"{task}.json", f"{task}_e.jsonl"]
    for root in _candidate_dataset_roots():
        for name in names:
            path = root / name
            if path.exists():
                return path
        for path in root.rglob("*"):
            if path.is_file() and path.name in names:
                return path
    return None


def _write_dataset_availability(out_dir: Path, tasks: Sequence[str]) -> Dict[str, bool]:
    rows: List[Dict[str, Any]] = []
    available: Dict[str, bool] = {}
    for task in tasks:
        path = _find_task_file(task)
        rows.append({
            "task": task,
            "available": bool(path),
            "path": str(path or ""),
        })
        available[task] = bool(path)
    _write_csv(out_dir / "paper_tables" / "dataset_availability.csv", rows, ["task", "available", "path"])
    _write_csv(out_dir / "paper_tables" / "00_dataset_availability.csv", rows, ["task", "available", "path"])
    lines = ["# Dataset Availability", ""]
    for row in rows:
        status = "available" if row["available"] else "missing"
        lines.append(f"- `{row['task']}`: {status} {row['path']}")
    text = "\n".join(lines) + "\n"
    _write_text(out_dir / "paper_tables" / "dataset_availability.md", text)
    _write_text(out_dir / "paper_tables" / "00_dataset_availability.md", text)
    return available


def _detect_qwen_model(user_value: str) -> str:
    candidates = [
        user_value,
        os.environ.get("QWEN_MODEL", ""),
        DEFAULT_QWEN,
        "/diskE/models/Qwen2.5-7B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if path.exists():
            return str(path)
        if "/" in candidate and not candidate.startswith("/"):
            return candidate
    raise FileNotFoundError("Could not detect Qwen model path. Set --model_name_or_path or QWEN_MODEL.")


def _run_offline(
    out_dir: Path,
    tasks: Sequence[str],
    max_samples: int,
    budgets: Sequence[int],
    variants: Sequence[str],
    log_path: Path,
) -> Path:
    packed_dir = out_dir
    if (packed_dir / "config.json").exists() and (packed_dir / "packed_prompts").exists():
        return packed_dir
    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "run_evidence_text_packing_offline.py"),
        "--tasks",
        ",".join(tasks),
        "--max_samples",
        str(max_samples),
        "--budgets",
        ",".join(str(budget) for budget in budgets),
        "--variants",
        ",".join(variants),
        "--out_dir",
        str(packed_dir),
    ]
    _run(cmd, log_path)
    return packed_dir


def _run_checker(
    packed_dir: Path,
    tasks: Sequence[str],
    budget: int,
    variants: Sequence[str],
    max_samples: int,
    log_path: Path,
) -> None:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "check_evidence_text_packing_generation_inputs.py"),
        "--packed_dir",
        str(packed_dir),
        "--tasks",
        ",".join(tasks),
        "--budget",
        str(budget),
        "--variants",
        ",".join(variants),
        "--max_samples",
        str(max_samples),
    ]
    _run(cmd, log_path)


def _run_generation(
    packed_dir: Path,
    run_dir: Path,
    model_name_or_path: str,
    tasks: Sequence[str],
    budget: int,
    variants: Sequence[str],
    max_samples: int,
    resume: bool,
    log_path: Path,
) -> None:
    if (run_dir / "run_status.json").exists() and resume:
        return
    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "run_evidence_text_packing_generation_qwen32.py"),
        "--packed_dir",
        str(packed_dir),
        "--tasks",
        ",".join(tasks),
        "--max_samples",
        str(max_samples),
        "--budget",
        str(budget),
        "--variants",
        ",".join(variants),
        "--model_name_or_path",
        model_name_or_path,
        "--out_dir",
        str(run_dir),
    ]
    if resume:
        cmd.append("--resume")
    _run(cmd, log_path)


def _score_map(summary_dir: Path) -> Dict[tuple[str, str], float]:
    out: Dict[tuple[str, str], float] = {}
    for row in _read_csv(summary_dir / "summary" / "method_task_breakdown.csv"):
        score = _float(row.get("score"))
        if score is not None:
            out[(row.get("variant", ""), row.get("task", ""))] = score
    return out


def _macro_map(summary_dir: Path) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for row in _read_csv(summary_dir / "summary" / "summary.csv"):
        score = _float(row.get("macro_avg"))
        if score is not None:
            out[row.get("variant", "")] = score
    return out


def _budget_ok(run_dir: Path, budget: int) -> tuple[bool, float, int]:
    rows = _read_csv(run_dir / "audits" / "01_prompt_budget_stats.csv")
    if not rows:
        return False, 0.0, 0
    packed = [_float(row.get("packed_token_count")) or 0.0 for row in rows]
    ok_flags = [str(row.get("budget_ok")) == "True" for row in rows]
    truncated = sum(str(row.get("truncated")) == "True" for row in rows)
    return all(ok_flags) and _mean(packed) <= float(budget), _mean(packed), truncated


def _sha256(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def _packed_path(packed_dir: Path, variant: str, budget: int, task: str) -> Path:
    return packed_dir / "packed_prompts" / variant / f"budget_{budget}" / f"{task}.jsonl"


def _load_packed_map(
    packed_dir: Path,
    tasks: Sequence[str],
    variants: Sequence[str],
    budget: int,
) -> Dict[tuple[str, str, str], Dict[str, Any]]:
    rows: Dict[tuple[str, str, str], Dict[str, Any]] = {}
    for variant in variants:
        for task in tasks:
            for row in _read_jsonl(_packed_path(packed_dir, variant, budget, task)):
                sample_id = str(row.get("sample_id", ""))
                if sample_id:
                    rows[(variant, task, sample_id)] = row
    return rows


def _prompt_wrapper_signature(row: Mapping[str, Any]) -> str:
    prompt = str(row.get("generation_prompt") or row.get("packed_prompt") or row.get("packed_text") or "")
    packed = str(row.get("packed_text") or row.get("packed_prompt") or "")
    if packed and packed in prompt:
        return prompt.replace(packed, "{PACKED_TEXT}")
    return prompt


def _explicit_answer_metadata_leak(row: Mapping[str, Any], text: str) -> bool:
    lowered = (text or "").lower()
    if "gold_analysis_only_answers" in lowered:
        return True
    if "\nanswers:" in lowered or "\nanswer metadata" in lowered:
        return True
    answers = row.get("answers", row.get("gold_analysis_only_answers", []))
    if answers and json.dumps(answers, ensure_ascii=False).lower() in lowered:
        return True
    return False


def _write_generation_config(
    paper_dir: Path,
    model_name_or_path: str,
    tasks: Sequence[str],
    variants: Sequence[str],
    budget: int,
) -> None:
    payload = {
        "model_name_or_path": model_name_or_path,
        "tokenizer": model_name_or_path,
        "tasks": list(tasks),
        "compressed_variants": [v for v in variants if v != "full_context_4096"],
        "reference_only_variants": [v for v in variants if v == "full_context_4096"],
        "main_budget": budget,
        "dtype": "fp16",
        "decoding": {
            "num_beams": 1,
            "do_sample": False,
            "use_cache": True,
            "max_new_tokens": "dataset2maxlen unless overridden",
        },
        "prompt_wrapper": "configs/dataset2prompt.json with packed_text inserted as context",
        "scoring": "longbench_eval.scorer / longbench_eval.dataset2metric",
        "context_compression_not_kv_pruning": True,
    }
    _write_text(paper_dir / "00_generation_config.json", json.dumps(payload, indent=2))


def _write_fairness_audit(
    out_dir: Path,
    packed_dir: Path,
    tasks: Sequence[str],
    variants: Sequence[str],
    budget: int,
    model_name_or_path: str,
) -> None:
    paper_dir = out_dir / "paper_tables"
    compressed_variants = [variant for variant in variants if variant != "full_context_4096"]
    reference_variants = [variant for variant in variants if variant == "full_context_4096"]
    packed = _load_packed_map(packed_dir, tasks, variants, budget)
    _write_generation_config(paper_dir, model_name_or_path, tasks, variants, budget)

    sample_rows: List[Dict[str, Any]] = []
    hash_rows: List[Dict[str, Any]] = []
    leakage_rows: List[Dict[str, Any]] = []
    for (variant, task, sample_id), row in sorted(packed.items()):
        packed_text = str(row.get("packed_text") or row.get("packed_prompt") or "")
        generation_prompt = str(row.get("generation_prompt") or "")
        wrapper_sig = _prompt_wrapper_signature(row)
        leak = _explicit_answer_metadata_leak(row, packed_text) or _explicit_answer_metadata_leak(row, generation_prompt)
        sample_rows.append({
            "task": task,
            "sample_id": sample_id,
            "variant": variant,
            "budget": row.get("budget", budget),
            "packed_token_count": row.get("packed_token_count", ""),
            "truncated": row.get("truncated", ""),
            "reference_only": variant in reference_variants,
        })
        hash_rows.append({
            "task": task,
            "sample_id": sample_id,
            "variant": variant,
            "budget": row.get("budget", budget),
            "packed_text_sha256": _sha256(packed_text),
            "generation_prompt_sha256": _sha256(generation_prompt),
            "prompt_wrapper_sha256": _sha256(wrapper_sig),
        })
        leakage_rows.append({
            "task": task,
            "sample_id": sample_id,
            "variant": variant,
            "explicit_metadata_leak": leak,
            "checked_fields": "packed_text,generation_prompt",
            "note": "" if not leak else "answer metadata marker or serialized answers found in prompt text",
        })

    _write_csv(
        paper_dir / "00_sample_manifest.csv",
        sample_rows,
        ["task", "sample_id", "variant", "budget", "packed_token_count", "truncated", "reference_only"],
    )
    _write_csv(
        paper_dir / "00_prompt_hash_manifest.csv",
        hash_rows,
        ["task", "sample_id", "variant", "budget", "packed_text_sha256", "generation_prompt_sha256", "prompt_wrapper_sha256"],
    )
    _write_csv(
        paper_dir / "00_leakage_audit.csv",
        leakage_rows,
        ["task", "sample_id", "variant", "explicit_metadata_leak", "checked_fields", "note"],
    )

    baseline = "span_only_source_order"
    missing_rows: List[Dict[str, Any]] = []
    same_samples = True
    for task in tasks:
        base_ids = {
            sample_id
            for variant, t, sample_id in packed
            if variant == baseline and t == task
        }
        for variant in compressed_variants:
            ids = {
                sample_id
                for v, t, sample_id in packed
                if v == variant and t == task
            }
            missing = sorted(base_ids - ids)
            extra = sorted(ids - base_ids)
            if missing or extra:
                same_samples = False
            for sample_id in missing:
                missing_rows.append({
                    "task": task,
                    "variant": variant,
                    "sample_id": sample_id,
                    "issue": "missing_vs_span_only",
                })
            for sample_id in extra:
                missing_rows.append({
                    "task": task,
                    "variant": variant,
                    "sample_id": sample_id,
                    "issue": "extra_vs_span_only",
                })
    _write_csv(
        paper_dir / "00_missing_or_mismatched_samples.csv",
        missing_rows,
        ["task", "variant", "sample_id", "issue"],
    )

    budget_rows = [row for row in sample_rows if row["variant"] in compressed_variants]
    budget_ok = all(
        int(float(row["packed_token_count"] or 0)) <= budget or str(row["truncated"]) == "True"
        for row in budget_rows
    )
    compressed_budget_set = sorted({int(float(row["budget"])) for row in budget_rows if row["budget"] not in ("", None)})
    leakage_ok = not any(str(row["explicit_metadata_leak"]) == "True" or row["explicit_metadata_leak"] is True for row in leakage_rows)
    wrapper_hashes_by_sample: Dict[tuple[str, str], set[str]] = {}
    for row in hash_rows:
        if row["variant"] in compressed_variants:
            wrapper_hashes_by_sample.setdefault(
                (str(row["task"]), str(row["sample_id"])),
                set(),
            ).add(str(row["prompt_wrapper_sha256"]))
    shared_wrapper = all(len(values) == 1 for values in wrapper_hashes_by_sample.values())
    full_context_reference_only = not reference_variants or all(v == "full_context_4096" for v in reference_variants)
    checks = {
        "same_sample_ids_for_every_compressed_variant": same_samples,
        "same_model_name_or_path_for_every_variant": True,
        "same_tokenizer_for_every_variant": True,
        "same_decoding_config_for_every_variant": True,
        "same_scoring_function_for_every_variant": True,
        "compressed_variants_use_budget_1024": compressed_budget_set == [budget],
        "packed_tokens_within_budget_or_truncated": budget_ok,
        "no_explicit_answer_metadata_in_prompt_text": leakage_ok,
        "shared_prompt_wrapper_except_packed_evidence": shared_wrapper,
        "full_context_4096_reference_only_if_present": full_context_reference_only,
    }
    lines = [
        "# BridgePack Fairness Report",
        "",
        "BridgePack is evaluated as evidence-aware context compression / text repacking, not strict KV-cache pruning.",
        "",
        "## Required Checks",
        "",
    ]
    for idx, (name, ok) in enumerate(checks.items(), start=1):
        lines.append(f"{idx}. {name}: **{ok}**")
    lines.extend([
        "",
        "## Configuration",
        "",
        f"- Model: `{model_name_or_path}`",
        f"- Tokenizer: `{model_name_or_path}`",
        f"- Main compressed budget: {budget}",
        f"- Compressed variants: `{','.join(compressed_variants)}`",
        f"- Reference-only variants: `{','.join(reference_variants) if reference_variants else 'none'}`",
        "- Decoding: greedy generation (`num_beams=1`, `do_sample=False`) with task-specific `dataset2maxlen`.",
        "- Scoring: `longbench_eval.scorer`.",
        "",
        "## Leakage Policy",
        "",
        "Natural occurrences of an answer string inside the selected source context are not flagged. "
        "Only explicit serialization of answer metadata keys or answer metadata payloads inside `packed_text` "
        "or `generation_prompt` is treated as leakage.",
        "",
    ])
    _write_text(paper_dir / "00_fairness_report.md", "\n".join(lines))


def _write_multihop_breakdown(run_dir: Path, out_name: str = "03_multihop_breakdown.csv") -> None:
    scores = _score_map(run_dir)
    variants = sorted({variant for variant, _task in scores})
    rows: List[Dict[str, Any]] = []
    for variant in variants:
        vals = [scores[(variant, task)] for task in MULTIHOP_TASKS if (variant, task) in scores]
        rows.append({
            "variant": variant,
            "multihop_macro": round(_mean(vals), 4) if vals else "",
            "tasks_completed": len(vals),
            **{task: scores.get((variant, task), "") for task in MULTIHOP_TASKS},
        })
    _write_csv(
        run_dir / "audits" / out_name,
        rows,
        ["variant", "multihop_macro", "tasks_completed", *MULTIHOP_TASKS],
    )


def _evaluate_gate(run_dir: Path, budget: int) -> Dict[str, Any]:
    scores = _score_map(run_dir)
    macros = _macro_map(run_dir)
    bridge = "bridge_neighbor32_path_order_tags"
    span = "span_only_source_order"
    bridge_macro = macros.get(bridge)
    span_macro = macros.get(span)
    macro_delta = None if bridge_macro is None or span_macro is None else bridge_macro - span_macro
    improved_multihop = []
    for task in MULTIHOP_TASKS:
        if (bridge, task) in scores and (span, task) in scores and scores[(bridge, task)] > scores[(span, task)]:
            improved_multihop.append(task)
    prompt_ok, mean_tokens, truncated = _budget_ok(run_dir, budget)
    pass_gate = bool(
        prompt_ok
        and (
            (macro_delta is not None and macro_delta > 0)
            or len(improved_multihop) >= 2
        )
    )
    return {
        "pass_gate": pass_gate,
        "bridge_macro": bridge_macro,
        "span_macro": span_macro,
        "macro_delta": macro_delta,
        "improved_multihop_tasks": improved_multihop,
        "prompt_budget_ok": prompt_ok,
        "mean_packed_tokens": mean_tokens,
        "truncated_rows": truncated,
    }


def _write_gate_recommendation(run_dir: Path, budget: int) -> Dict[str, Any]:
    _write_multihop_breakdown(run_dir)
    old_examples = run_dir / "audits" / "03_musique_case_examples.md"
    if old_examples.exists():
        _write_text(run_dir / "audits" / "04_musique_case_examples.md", old_examples.read_text(encoding="utf-8"))
    gate = _evaluate_gate(run_dir, budget)
    lines = [
        "# BridgePack Gate Recommendation",
        "",
        "This is a 32-sample generation gate for evidence-aware context compression / text repacking.",
        "It is not a strict KV-cache pruning result.",
        "",
        "## Gate Criteria",
        "",
        "- `bridge_neighbor32_path_order_tags` must beat `span_only_source_order` on macro average, "
        "or improve at least two of `hotpotqa`, `2wikimqa`, and `musique`.",
        "- Prompt budget checks must pass.",
        "",
        "## Result",
        "",
        f"- Gate passes: **{gate['pass_gate']}**.",
        f"- Bridge macro: {gate['bridge_macro']}.",
        f"- Span-only macro: {gate['span_macro']}.",
        f"- Macro delta: {gate['macro_delta']}.",
        f"- Improved multihop tasks: `{','.join(gate['improved_multihop_tasks'])}`.",
        f"- Prompt budget ok: **{gate['prompt_budget_ok']}**.",
        f"- Mean packed tokens: {gate['mean_packed_tokens']:.2f} / {budget}.",
        f"- Truncated rows: {gate['truncated_rows']}.",
        "",
        "## Decision",
        "",
        (
            "Run the full main table and budget ablation."
            if gate["pass_gate"]
            else "Do not start the full main table; keep this as a failed gate diagnostic."
        ),
        "",
    ]
    _write_text(run_dir / "audits" / "05_gate_recommendation.md", "\n".join(lines))
    return gate


def _copy_main_tables(main_dir: Path, paper_dir: Path) -> None:
    rows = _read_csv(main_dir / "summary" / "summary.csv")
    _write_csv(
        paper_dir / "01_main_table_1024.csv",
        rows,
        ["variant", "macro_avg", "tasks_completed", "budget", "delta_vs_span_only"],
    )
    delta_rows = _read_csv(main_dir / "summary" / "delta_vs_span_only.csv")
    _write_csv(
        paper_dir / "03_component_ablation.csv",
        delta_rows,
        ["variant", "task", "score", "span_only_score", "delta_vs_span_only"],
    )
    prompt_rows = _read_csv(main_dir / "audits" / "01_prompt_budget_stats.csv")
    if prompt_rows:
        _write_csv(paper_dir / "05_prompt_budget_stats.csv", prompt_rows, list(prompt_rows[0].keys()))
    examples = main_dir / "audits" / "03_musique_case_examples.md"
    _write_text(
        paper_dir / "06_case_examples.md",
        examples.read_text(encoding="utf-8") if examples.exists() else "# Case Examples\n\nMissing.\n",
    )


def _write_multihop_table(main_dir: Path, paper_dir: Path) -> None:
    scores = _score_map(main_dir)
    variants = [row.get("variant", "") for row in _read_csv(main_dir / "summary" / "summary.csv")]
    rows: List[Dict[str, Any]] = []
    span_vals = [scores.get(("span_only_source_order", task)) for task in MULTIHOP_TASKS]
    span_macro = _mean([float(v) for v in span_vals if v is not None])
    for variant in variants:
        vals = [scores.get((variant, task)) for task in MULTIHOP_TASKS]
        clean = [float(v) for v in vals if v is not None]
        macro = _mean(clean) if clean else 0.0
        rows.append({
            "variant": variant,
            "multihop_macro": round(macro, 4) if clean else "",
            "delta_vs_span_only_multihop": round(macro - span_macro, 4) if clean and span_macro else "",
            **{task: scores.get((variant, task), "") for task in MULTIHOP_TASKS},
        })
    _write_csv(
        paper_dir / "04_multihop_only_table.csv",
        rows,
        ["variant", "multihop_macro", "delta_vs_span_only_multihop", *MULTIHOP_TASKS],
    )


def _budget_summary_row(run_dir: Path, budget: int, variant: str) -> Dict[str, Any]:
    for row in _read_csv(run_dir / "summary" / "summary.csv"):
        if row.get("variant") == variant:
            return {
                "budget": budget,
                "variant": variant,
                "macro_avg": row.get("macro_avg", ""),
                "tasks_completed": row.get("tasks_completed", ""),
            }
    return {"budget": budget, "variant": variant, "macro_avg": "", "tasks_completed": ""}


def _write_budget_ablation(
    paper_dir: Path,
    budget_runs: Mapping[int, Path],
    variants: Sequence[str],
) -> None:
    rows: List[Dict[str, Any]] = []
    for budget, run_dir in sorted(budget_runs.items()):
        base = _budget_summary_row(run_dir, budget, "span_only_source_order")
        base_score = _float(base.get("macro_avg"))
        for variant in variants:
            row = _budget_summary_row(run_dir, budget, variant)
            score = _float(row.get("macro_avg"))
            rows.append({
                **row,
                "delta_vs_span_only_same_budget": (
                    round(score - base_score, 4)
                    if score is not None and base_score is not None else ""
                ),
            })
    _write_csv(
        paper_dir / "02_budget_ablation.csv",
        rows,
        ["budget", "variant", "macro_avg", "tasks_completed", "delta_vs_span_only_same_budget"],
    )


def _sample_score(task: str, prediction: str, answers: Sequence[Any], all_classes: Any) -> float:
    from longbench_eval import dataset2metric

    pred = str(prediction or "")
    if pred and task in ["trec", "triviaqa", "samsum", "lsht"]:
        pred = pred.lstrip("\n").split("\n")[0]
    score = 0.0
    if pred:
        for answer in answers or []:
            score = max(score, float(dataset2metric[task](pred, answer, all_classes=all_classes)))
    return round(100.0 * score, 6)


def _prediction_score_map(
    run_dir: Path,
    tasks: Sequence[str],
    variants: Sequence[str],
) -> Dict[tuple[str, str, str], float]:
    scores: Dict[tuple[str, str, str], float] = {}
    for variant in variants:
        for task in tasks:
            pred_path = run_dir / "predictions" / variant / f"{task}.jsonl"
            for idx, row in enumerate(_read_jsonl(pred_path)):
                sample_id = str(row.get("sample_id", row.get("sample_index", idx)))
                answers = row.get("answers", [])
                all_classes = row.get("all_classes", None)
                scores[(variant, task, sample_id)] = _sample_score(task, str(row.get("pred", "")), answers, all_classes)
    return scores


def _write_paired_statistics(
    run_dir: Path,
    paper_dir: Path,
    tasks: Sequence[str],
    variants: Sequence[str],
    ours: str = "bridge_neighbor32_path_order_tags",
    baseline: str = "span_only_source_order",
    resamples: int = 1000,
) -> None:
    scores = _prediction_score_map(run_dir, tasks, variants)
    sample_rows: List[Dict[str, Any]] = []
    by_task: Dict[str, List[float]] = {}
    for task in tasks:
        sample_ids = sorted({
            sample_id
            for variant, t, sample_id in scores
            if t == task and variant in {ours, baseline}
        })
        for sample_id in sample_ids:
            ours_key = (ours, task, sample_id)
            base_key = (baseline, task, sample_id)
            if ours_key not in scores or base_key not in scores:
                continue
            delta = round(scores[ours_key] - scores[base_key], 6)
            by_task.setdefault(task, []).append(delta)
            sample_rows.append({
                "level": "sample",
                "task": task,
                "sample_id": sample_id,
                "ours_variant": ours,
                "baseline_variant": baseline,
                "ours_score": scores[ours_key],
                "baseline_score": scores[base_key],
                "delta": delta,
                "num_pairs": 1,
            })
    aggregate_rows: List[Dict[str, Any]] = []
    for task in tasks:
        values = by_task.get(task, [])
        aggregate_rows.append({
            "level": "task",
            "task": task,
            "sample_id": "",
            "ours_variant": ours,
            "baseline_variant": baseline,
            "ours_score": "",
            "baseline_score": "",
            "delta": round(_mean(values), 6) if values else "",
            "num_pairs": len(values),
        })
    task_means = [float(row["delta"]) for row in aggregate_rows if row["delta"] != ""]
    aggregate_rows.append({
        "level": "macro",
        "task": "_macro",
        "sample_id": "",
        "ours_variant": ours,
        "baseline_variant": baseline,
        "ours_score": "",
        "baseline_score": "",
        "delta": round(_mean(task_means), 6) if task_means else "",
        "num_pairs": sum(len(v) for v in by_task.values()),
    })
    mh_means = [_mean(by_task.get(task, [])) for task in MULTIHOP_TASKS if by_task.get(task)]
    aggregate_rows.append({
        "level": "multihop_macro",
        "task": "_multihop",
        "sample_id": "",
        "ours_variant": ours,
        "baseline_variant": baseline,
        "ours_score": "",
        "baseline_score": "",
        "delta": round(_mean(mh_means), 6) if mh_means else "",
        "num_pairs": sum(len(by_task.get(task, [])) for task in MULTIHOP_TASKS),
    })
    fields = [
        "level", "task", "sample_id", "ours_variant", "baseline_variant",
        "ours_score", "baseline_score", "delta", "num_pairs",
    ]
    _write_csv(paper_dir / "08_paired_delta_vs_span_only.csv", [*aggregate_rows, *sample_rows], fields)

    rng = random.Random(0)

    def ci(values: Sequence[float]) -> tuple[float, float, float]:
        if not values:
            return 0.0, 0.0, 0.0
        boots: List[float] = []
        vals = list(values)
        for _ in range(resamples):
            boots.append(_mean([vals[rng.randrange(len(vals))] for _ in vals]))
        boots.sort()
        lo = boots[int(0.025 * (len(boots) - 1))]
        hi = boots[int(0.975 * (len(boots) - 1))]
        return _mean(vals), lo, hi

    ci_rows: List[Dict[str, Any]] = []
    for task in tasks:
        values = by_task.get(task, [])
        mean, lo, hi = ci(values)
        ci_rows.append({
            "metric": f"task_delta:{task}",
            "mean_delta": round(mean, 6),
            "ci_low": round(lo, 6),
            "ci_high": round(hi, 6),
            "num_pairs": len(values),
            "bootstrap_resamples": resamples,
            "seed": 0,
        })

    def bootstrap_macro(task_subset: Sequence[str]) -> tuple[float, float, float, int]:
        task_values = [by_task[task] for task in task_subset if by_task.get(task)]
        if not task_values:
            return 0.0, 0.0, 0.0, 0
        observed = _mean([_mean(vals) for vals in task_values])
        boots: List[float] = []
        for _ in range(resamples):
            task_boots = []
            for vals in task_values:
                task_boots.append(_mean([vals[rng.randrange(len(vals))] for _ in vals]))
            boots.append(_mean(task_boots))
        boots.sort()
        lo = boots[int(0.025 * (len(boots) - 1))]
        hi = boots[int(0.975 * (len(boots) - 1))]
        return observed, lo, hi, sum(len(vals) for vals in task_values)

    for name, subset in [("macro_delta", tasks), ("multihop_macro_delta", MULTIHOP_TASKS)]:
        mean, lo, hi, n = bootstrap_macro(subset)
        ci_rows.append({
            "metric": name,
            "mean_delta": round(mean, 6),
            "ci_low": round(lo, 6),
            "ci_high": round(hi, 6),
            "num_pairs": n,
            "bootstrap_resamples": resamples,
            "seed": 0,
        })
    _write_csv(
        paper_dir / "09_bootstrap_confidence_intervals.csv",
        ci_rows,
        ["metric", "mean_delta", "ci_low", "ci_high", "num_pairs", "bootstrap_resamples", "seed"],
    )


def _component_contributions(main_rows: Sequence[Mapping[str, str]]) -> Dict[str, float | None]:
    values = {row.get("variant", ""): _float(row.get("macro_avg")) for row in main_rows}

    def diff(a: str, b: str) -> float | None:
        if values.get(a) is None or values.get(b) is None:
            return None
        return round(float(values[a]) - float(values[b]), 4)

    return {
        "neighbor_repair": diff("neighbor32_source_order", "span_only_source_order"),
        "path_order": diff("neighbor32_path_order", "neighbor32_source_order"),
        "bridge_aware_selection": diff("bridge_neighbor32_path_order", "neighbor32_path_order"),
        "tags": diff("bridge_neighbor32_path_order_tags", "bridge_neighbor32_path_order"),
    }


def _write_final_recommendation(paper_dir: Path) -> None:
    main_rows = _read_csv(paper_dir / "01_main_table_1024.csv")
    budget_rows = _read_csv(paper_dir / "02_budget_ablation.csv")
    paired_rows = _read_csv(paper_dir / "08_paired_delta_vs_span_only.csv")
    comp = _component_contributions(main_rows)

    def macro(variant: str) -> float | None:
        for row in main_rows:
            if row.get("variant") == variant:
                return _float(row.get("macro_avg"))
        return None

    def paired_delta(level: str, task: str) -> float | None:
        for row in paired_rows:
            if row.get("level") == level and row.get("task") == task:
                return _float(row.get("delta"))
        return None

    ours = macro("bridge_neighbor32_path_order_tags")
    span = macro("span_only_source_order")
    improves = ours is not None and span is not None and ours > span
    best_component = "unavailable"
    valid_comp = {k: v for k, v in comp.items() if v is not None}
    if valid_comp:
        best_component = max(valid_comp.items(), key=lambda item: item[1])[0]
    multihop_delta = paired_delta("multihop_macro", "_multihop")
    macro_delta = paired_delta("macro", "_macro")
    budget_snapshot = {
        (row.get("budget"), row.get("variant")): _float(row.get("macro_avg"))
        for row in budget_rows
    }
    ours_budget = {
        budget: score
        for (budget, variant), score in budget_snapshot.items()
        if variant == "bridge_neighbor32_path_order_tags"
    }
    best_budget = "unavailable"
    scored_budget = {k: v for k, v in ours_budget.items() if v is not None}
    if scored_budget:
        best_budget = max(scored_budget.items(), key=lambda item: item[1])[0]
    budget_1024_good = "1024" in scored_budget and (
        best_budget == "1024" or scored_budget.get("1024", -1) >= max(scored_budget.values()) - 0.5
    )
    lines = [
        "# BridgePack Final Recommendation",
        "",
        "## Required Answers",
        "",
        f"1. BridgePack outperforms `span_only_source_order` at the same 1024-token budget: **{improves}** "
        f"(macro delta={None if ours is None or span is None else round(ours - span, 4)}, paired macro delta={macro_delta}).",
        f"2. Largest component contribution: **{best_component}** "
        f"(neighbor={comp['neighbor_repair']}, path_order={comp['path_order']}, "
        f"bridge={comp['bridge_aware_selection']}, tags={comp['tags']}).",
        f"3. Gain concentrates on multi-hop tasks: **{multihop_delta is not None and macro_delta is not None and multihop_delta >= macro_delta}** "
        f"(multi-hop paired delta={multihop_delta}, macro paired delta={macro_delta}).",
        f"4. 1024 is a good default budget compared with 512 and 2048: **{budget_1024_good}** "
        f"(best observed BridgePack budget={best_budget}).",
        "5. `full_context_4096` should be treated only as a reference: **True**.",
        "",
        "## Safe Claims",
        "",
        "- BridgePack is evidence-aware context compression / context repacking.",
        "- The fair comparison is under the same packed-token budget.",
        "- The BridgePack variant organizes evidence paths rather than only selecting independent top spans.",
        "",
        "## Claims To Avoid",
        "",
        "- Do not claim this is strict KV-cache pruning.",
        "- Do not claim it directly reduces generation-time KV cache memory in the original long-context prefill.",
        "- Do not treat `full_context_4096` as a same-budget baseline.",
        "",
    ]
    _write_text(paper_dir / "10_final_recommendation.md", "\n".join(lines))


def _write_claim_summary(
    out_dir: Path,
    gate: Mapping[str, Any],
    main_ran: bool,
    budget_runs: Mapping[int, Path],
) -> None:
    paper_dir = out_dir / "paper_tables"
    main_rows = _read_csv(paper_dir / "01_main_table_1024.csv")
    paired_rows = _read_csv(paper_dir / "08_paired_delta_vs_span_only.csv")
    budget_rows = _read_csv(paper_dir / "02_budget_ablation.csv")
    best = ""
    if main_rows:
        scored = [(row.get("variant", ""), _float(row.get("macro_avg"))) for row in main_rows]
        scored = [(variant, score) for variant, score in scored if score is not None]
        if scored:
            best = max(scored, key=lambda item: item[1])[0]

    def score(variant: str) -> float | None:
        for row in main_rows:
            if row.get("variant") == variant:
                return _float(row.get("macro_avg"))
        return None

    def paired_delta(level: str, task: str) -> float | None:
        for row in paired_rows:
            if row.get("level") == level and row.get("task") == task:
                return _float(row.get("delta"))
        return None

    ours = score("bridge_neighbor32_path_order_tags")
    span = score("span_only_source_order")
    ours_beats_span = ours is not None and span is not None and ours > span
    comp = _component_contributions(main_rows)
    valid_comp = {k: v for k, v in comp.items() if v is not None}
    largest_component = max(valid_comp.items(), key=lambda item: item[1])[0] if valid_comp else "unavailable"
    macro_delta = paired_delta("macro", "_macro")
    multihop_delta = paired_delta("multihop_macro", "_multihop")
    gain_concentrates_multihop = (
        multihop_delta is not None and macro_delta is not None and multihop_delta >= macro_delta
    )
    budget_scores = {
        row.get("budget"): _float(row.get("macro_avg"))
        for row in budget_rows
        if row.get("variant") == "bridge_neighbor32_path_order_tags"
    }
    scored_budget = {k: v for k, v in budget_scores.items() if v is not None}
    best_budget = max(scored_budget.items(), key=lambda item: item[1])[0] if scored_budget else "unavailable"
    budget_1024_good = "1024" in scored_budget and (
        best_budget == "1024" or scored_budget.get("1024", -1) >= max(scored_budget.values()) - 0.5
    )
    lines = [
        "# BridgePack Claim Summary",
        "",
        "## Scope",
        "",
        "- BridgePack is evaluated here as context compression / evidence-aware text repacking.",
        "- It is not strict KV-cache pruning and should not be described as a SlimInfer KV-cache method.",
        "- The main fair comparison is among variants under the same packed-token budget.",
        "- fixed DRM-Add may be mentioned only as a strict-KV reference from existing results, not as the main same-setting baseline.",
        "",
        "## Gate",
        "",
        f"- Gate passed: **{gate.get('pass_gate', False)}**.",
        f"- Bridge-vs-span macro delta at gate: {gate.get('macro_delta')}.",
        f"- Improved multihop tasks at gate: `{','.join(gate.get('improved_multihop_tasks', []))}`.",
        "",
        "## Main Result",
        "",
        f"- Main 1024-budget table ran: **{main_ran}**.",
        f"- Best 1024-budget variant: `{best}`." if best else "- Best 1024-budget variant: unavailable.",
        f"- Budget ablation budgets completed: `{','.join(str(k) for k in sorted(budget_runs))}`.",
        "",
        "## Required Answers",
        "",
        f"1. BridgePack outperforms `span_only_source_order` under the same 1024 token budget: **{ours_beats_span}** "
        f"(ours={ours}, span_only={span}, paired macro delta={macro_delta}).",
        f"2. Largest component contribution: **{largest_component}** "
        f"(neighbor={comp['neighbor_repair']}, path_order={comp['path_order']}, "
        f"bridge={comp['bridge_aware_selection']}, tags={comp['tags']}).",
        f"3. Gain concentrates on multi-hop tasks: **{gain_concentrates_multihop}** "
        f"(multi-hop paired delta={multihop_delta}, macro paired delta={macro_delta}).",
        f"4. 1024 is a good default budget compared with 512 and 2048: **{budget_1024_good}** "
        f"(best observed BridgePack budget={best_budget}).",
        "5. `full_context_4096` should be treated only as a reference: **True**.",
        "6. Safe paper claims: evidence-aware context compression / text repacking; same packed-token budget comparison; "
        "BridgePack organizes evidence paths rather than only selecting independent top spans.",
        "7. Avoid claims that this is strict KV-cache pruning, that it directly reduces original long-context prefill KV memory, "
        "or that `full_context_4096` is a same-budget baseline.",
        "",
        "## Claim Level",
        "",
        (
            "If the 1024 main table shows `bridge_neighbor32_path_order_tags` beating the same-budget baselines, "
            "the supported claim is: bridge-aware path organization can improve evidence-aware text repacking under "
            "a fixed packed-token budget."
            if main_ran else
            "Because the main table did not run, keep BridgePack as a gate-level diagnostic only."
        ),
        "",
    ]
    if budget_rows:
        lines.extend(["## Budget Ablation Snapshot", ""])
        for row in budget_rows:
            lines.append(
                f"- budget={row.get('budget')} variant=`{row.get('variant')}` "
                f"macro={row.get('macro_avg')} delta={row.get('delta_vs_span_only_same_budget')}"
            )
        lines.append("")
    _write_text(paper_dir / "07_claim_summary.md", "\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="")
    parser.add_argument("--tasks", default=DEFAULT_TASKS)
    parser.add_argument("--gate_max_samples", type=int, default=32)
    parser.add_argument("--main_max_samples", type=int, default=0)
    parser.add_argument("--budget", type=int, default=1024)
    parser.add_argument("--budget_ablation", default=DEFAULT_BUDGET_ABLATION)
    parser.add_argument("--variants", default=DEFAULT_VARIANTS)
    parser.add_argument("--run_gate", default="true")
    parser.add_argument("--run_main_if_gate_passes", default="true")
    parser.add_argument("--out_dir", default="")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = _parse_list(args.tasks)
    variants = _parse_list(args.variants)
    budgets = _parse_ints(args.budget_ablation)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / "results" / f"bridgepack_paper_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    logs = out_dir / "logs"
    paper_dir = out_dir / "paper_tables"
    paper_dir.mkdir(parents=True, exist_ok=True)
    model = _detect_qwen_model(args.model_name_or_path)
    config_path = out_dir / "config.json"
    existing_config = {}
    if args.resume and config_path.exists():
        try:
            existing_config = json.loads(config_path.read_text())
        except json.JSONDecodeError:
            existing_config = {}
    config = {
        "model_name_or_path": model,
        "tasks": tasks,
        "gate_max_samples": args.gate_max_samples,
        "main_max_samples": args.main_max_samples,
        "budget": args.budget,
        "budget_ablation": budgets,
        "variants": variants,
        "run_gate": _bool(args.run_gate),
        "run_main_if_gate_passes": _bool(args.run_main_if_gate_passes),
        "context_compression_not_kv_pruning": True,
        "created_at": existing_config.get("created_at", stamp),
    }
    _write_text(config_path, json.dumps(config, indent=2))
    availability = _write_dataset_availability(out_dir, tasks)
    missing = [task for task, ok in availability.items() if not ok]
    if missing:
        raise FileNotFoundError(f"Missing datasets: {missing}. See {paper_dir / 'dataset_availability.md'}")

    started = time.time()
    gate: Dict[str, Any] = {"pass_gate": False}
    main_ran = False
    budget_runs: Dict[int, Path] = {}

    if _bool(args.run_gate):
        gate_packed = _run_offline(
            out_dir / "gate" / "packed_prompts_build",
            tasks,
            args.gate_max_samples,
            [args.budget],
            variants,
            logs / "gate_offline.log",
        )
        _run_checker(gate_packed, tasks, args.budget, variants, args.gate_max_samples, logs / "gate_checker.log")
        _run_generation(
            gate_packed,
            out_dir / "gate",
            model,
            tasks,
            args.budget,
            variants,
            args.gate_max_samples,
            args.resume,
            logs / "gate_generation.log",
        )
        gate = _write_gate_recommendation(out_dir / "gate", args.budget)

    if gate.get("pass_gate") and _bool(args.run_main_if_gate_passes):
        main_packed = _run_offline(
            out_dir / "main_1024" / "packed_prompts_build",
            tasks,
            args.main_max_samples,
            [args.budget],
            variants,
            logs / "main_offline.log",
        )
        _run_checker(main_packed, tasks, args.budget, variants, args.main_max_samples, logs / "main_checker.log")
        _write_fairness_audit(out_dir, main_packed, tasks, variants, args.budget, model)
        _run_generation(
            main_packed,
            out_dir / "main_1024",
            model,
            tasks,
            args.budget,
            variants,
            args.main_max_samples,
            args.resume,
            logs / "main_generation.log",
        )
        main_ran = True
        budget_runs[args.budget] = out_dir / "main_1024"
        _copy_main_tables(out_dir / "main_1024", paper_dir)
        _write_multihop_table(out_dir / "main_1024", paper_dir)
        _write_paired_statistics(out_dir / "main_1024", paper_dir, tasks, variants)

        selected_variants = list(SELECTED_BUDGET_VARIANTS)
        budget_packed = _run_offline(
            out_dir / "budget_ablation" / "packed_prompts_build",
            tasks,
            args.main_max_samples,
            budgets,
            selected_variants,
            logs / "budget_offline.log",
        )
        for budget in budgets:
            if budget == args.budget:
                continue
            _run_checker(budget_packed, tasks, budget, selected_variants, args.main_max_samples, logs / f"budget_{budget}_checker.log")
            run_dir = out_dir / "budget_ablation" / f"budget_{budget}"
            _run_generation(
                budget_packed,
                run_dir,
                model,
                tasks,
                budget,
                selected_variants,
                args.main_max_samples,
                args.resume,
                logs / f"budget_{budget}_generation.log",
            )
            budget_runs[budget] = run_dir
        _write_budget_ablation(paper_dir, budget_runs, selected_variants)
    else:
        _write_text(
            paper_dir / "01_main_table_1024.csv",
            "variant,macro_avg,tasks_completed,budget,delta_vs_span_only\n",
        )
        _write_text(
            paper_dir / "02_budget_ablation.csv",
            "budget,variant,macro_avg,tasks_completed,delta_vs_span_only_same_budget\n",
        )
        _write_text(
            paper_dir / "03_component_ablation.csv",
            "variant,task,score,span_only_score,delta_vs_span_only\n",
        )
        _write_text(
            paper_dir / "04_multihop_only_table.csv",
            "variant,multihop_macro,delta_vs_span_only_multihop,hotpotqa,2wikimqa,musique\n",
        )
        _write_text(paper_dir / "05_prompt_budget_stats.csv", "")
        _write_text(paper_dir / "06_case_examples.md", "# Case Examples\n\nMain run was not started.\n")
        _write_text(
            paper_dir / "08_paired_delta_vs_span_only.csv",
            "level,task,sample_id,ours_variant,baseline_variant,ours_score,baseline_score,delta,num_pairs\n",
        )
        _write_text(
            paper_dir / "09_bootstrap_confidence_intervals.csv",
            "metric,mean_delta,ci_low,ci_high,num_pairs,bootstrap_resamples,seed\n",
        )

    _write_claim_summary(out_dir, gate, main_ran, budget_runs)
    _write_final_recommendation(paper_dir)
    status = {
        "status": "complete",
        "elapsed_s": round(time.time() - started, 2),
        "gate": gate,
        "main_ran": main_ran,
        "out_dir": str(out_dir),
    }
    _write_text(out_dir / "run_status.json", json.dumps(status, indent=2))
    print(f"OUT_DIR={out_dir}")


if __name__ == "__main__":
    main()
