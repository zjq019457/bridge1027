#!/usr/bin/env python3
"""Run the fixed 20-sample DC-SlimInfer tiny quality smoke.

This runner compares only SlimInfer original and DC-SlimInfer.  It does not run
full LongBench, 64-sample diagnostics, threshold sweeps, DRM-Add, or DRM-Swap.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from longbench_eval import dataset2metric, scorer


DEFAULT_MODEL_PATH = Path(
    "/home/yu-bai/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/"
    "snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
)
DEFAULT_LONGBENCH_DIR = Path("/home/yu-bai/Desktop/jiaqizhang/HeadKV/data/LongBench")
METHOD_ORIGINAL = "sliminfer_original"
METHOD_DC = "dc_sliminfer"
TASKS = ["hotpotqa", "musique"]
FIXED_HOTPOTQA = [0, 1, 2, 3, 4, 5, 41, 56, 57, 117]
FIXED_MUSIQUE = [0, 1, 2, 3, 4, 20, 50, 54, 56, 134]
FIXED_INDICES = {"hotpotqa": FIXED_HOTPOTQA, "musique": FIXED_MUSIQUE}
FIXED_SPEC = (
    "hotpotqa:" + ",".join(str(x) for x in FIXED_HOTPOTQA)
    + ";musique:" + ",".join(str(x) for x in FIXED_MUSIQUE)
)
PRUNE_LAYERS = [9, 18, 26]
FINAL_BUDGET_TOKENS = 4096
HOTPOTQA_DROP_TOLERANCE = 5.0

JsonRow = Dict[str, Any]
Key = Tuple[str, int]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def read_jsonl(path: Path, *, missing_ok: bool = False) -> List[JsonRow]:
    if not path.exists():
        if missing_ok:
            return []
        raise RuntimeError(f"missing JSONL file: {path}")
    rows: List[JsonRow] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
    return rows


def mean(values: Iterable[float]) -> float:
    items = [float(value) for value in values if math.isfinite(float(value))]
    return float(sum(items) / len(items)) if items else 0.0


def check_gpu(min_free_gpu_gb: float) -> Dict[str, Any]:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free", "--format=csv,noheader,nounits"],
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip().splitlines()
    rows = []
    for line in output:
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 4:
            continue
        rows.append({
            "index": int(parts[0]),
            "name": parts[1],
            "memory_total_mib": float(parts[2]),
            "memory_free_mib": float(parts[3]),
            "memory_free_gib": float(parts[3]) / 1024.0,
        })
    if not rows:
        raise RuntimeError("nvidia-smi returned no GPU rows")
    selected = max(rows, key=lambda row: row["memory_free_gib"])
    if float(selected["memory_free_gib"]) < float(min_free_gpu_gb):
        raise RuntimeError(
            f"free GPU memory {selected['memory_free_gib']:.2f} GiB < required {float(min_free_gpu_gb):.2f} GiB"
        )
    return {"rows": rows, "selected": selected, "min_free_gpu_gb": float(min_free_gpu_gb)}


def python_bin() -> Path:
    candidate = REPO_ROOT / ".venv" / "bin" / "python"
    return candidate if candidate.exists() else Path(sys.executable)


def base_runner_cmd(args: argparse.Namespace, output_dir: Path) -> List[str]:
    return [
        str(python_bin()),
        str(REPO_ROOT / "experiments" / "run_qwen25_sliminfer_variants.py"),
        "--model_path", str(args.model_path),
        "--tokenizer_path", str(args.model_path),
        "--model_family", "qwen2",
        "--dtype", "fp16",
        "--device", "cuda",
        "--backend", "flash_attention_2",
        "--longbench_dir", str(args.longbench_dir),
        "--longbench_tasks", ",".join(TASKS),
        "--longbench_sample_indices_by_task", FIXED_SPEC,
        "--longbench_subset_name", "dc_sliminfer_tiny_quality_smoke",
        "--generation_max_new_tokens", "32",
        "--max_model_len", "32768",
        "--decode_policy", "original",
        "--rescue_policy", "none",
        "--prune_layers", ",".join(str(x) for x in PRUNE_LAYERS),
        "--retain_tokens", "12288,6144,4096",
        "--block_size", "64",
        "--token_unit_size", "8",
        "--local_query_window", "4",
        "--swap_threshold", "0.9",
        "--temperature", "0",
        "--top_p", "1.0",
        "--seed", "0",
        "--dc_alpha", "0.10",
        "--dc_beta", "0.05",
        "--dc_lambda", "0.15",
        "--dc_gamma", "0.05",
        "--output_dir", str(output_dir),
        "--min_free_gpu_gb", "0",
    ]


def run_command(cmd: Sequence[str], cwd: Path, stdout_path: Path, stderr_path: Path) -> None:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        proc = subprocess.run(list(cmd), cwd=str(cwd), text=True, stdout=stdout, stderr=stderr)
    stdout_text = stdout_path.read_text(encoding="utf-8", errors="ignore")
    stderr_text = stderr_path.read_text(encoding="utf-8", errors="ignore")
    if "CUDA out of memory" in stdout_text or "CUDA out of memory" in stderr_text:
        raise RuntimeError(f"CUDA OOM detected while running: {' '.join(cmd)}")
    if proc.returncode != 0:
        raise RuntimeError(f"runner failed with return code {proc.returncode}; see {stderr_path}")


def run_dc_gate_audit(args: argparse.Namespace, output_dir: Path) -> Tuple[Path, Path]:
    run_dir = output_dir / "run_dc_gate_audit"
    activation_path = output_dir / "dc_gate_activation_audit.jsonl"
    selection_path = output_dir / "dc_gate_selection_audit.jsonl"
    activation_path.write_text("", encoding="utf-8")
    selection_path.write_text("", encoding="utf-8")
    cmd = base_runner_cmd(args, run_dir) + [
        "--eval", "gate_audit",
        "--method", METHOD_DC,
        "--dc_activation_audit_path", str(activation_path),
        "--dc_selection_audit_path", str(selection_path),
    ]
    write_json(output_dir / "dc_gate_audit_command.json", {"command": cmd})
    run_command(cmd, REPO_ROOT, output_dir / "dc_gate_audit_stdout.log", output_dir / "dc_gate_audit_stderr.log")
    return activation_path, selection_path


def run_quality_method(args: argparse.Namespace, output_dir: Path, method: str) -> Path:
    run_dir = output_dir / f"run_{method}"
    audit_dir = run_dir / "audit"
    audit_dir.mkdir(parents=True, exist_ok=False)
    cmd = base_runner_cmd(args, run_dir) + [
        "--eval", "longbench",
        "--method", method,
        "--debug_generation_jsonl", str(audit_dir / "generations.jsonl"),
        "--debug_used_block_audit_path", str(audit_dir / "used_blocks.jsonl"),
        "--debug_used_block_include_full", "true",
        "--debug_used_block_full_layers", "27",
    ]
    if method == METHOD_DC:
        cmd.extend([
            "--dc_activation_audit_path", str(audit_dir / "dc_runtime_activation_audit.jsonl"),
            "--dc_selection_audit_path", str(audit_dir / "dc_runtime_selection_audit.jsonl"),
        ])
        (audit_dir / "dc_runtime_activation_audit.jsonl").write_text("", encoding="utf-8")
        (audit_dir / "dc_runtime_selection_audit.jsonl").write_text("", encoding="utf-8")
    write_json(output_dir / f"{method}_quality_command.json", {"command": cmd})
    run_command(cmd, REPO_ROOT, output_dir / f"{method}_quality_stdout.log", output_dir / f"{method}_quality_stderr.log")
    return run_dir


def metric_score(task: str, prediction: str, answers: Sequence[Any], all_classes: Any) -> float:
    pred = prediction
    if task in {"trec", "triviaqa", "samsum", "lsht"}:
        pred = pred.lstrip("\n").split("\n")[0]
    best = 0.0
    for answer in answers:
        best = max(best, float(dataset2metric[task](pred, answer, all_classes=all_classes)))
    return float(100.0 * best)


def read_predictions(run_dir: Path, method: str) -> Dict[Key, JsonRow]:
    out: Dict[Key, JsonRow] = {}
    for task in TASKS:
        pred_path = run_dir / "longbench_predictions" / f"{task}.jsonl"
        rows = read_jsonl(pred_path)
        expected = FIXED_INDICES[task]
        if len(rows) != len(expected):
            raise RuntimeError(f"{method} {task} predictions count mismatch: {len(rows)} != {len(expected)}")
        all_classes = None
        for row in rows:
            if row.get("all_classes") is not None:
                all_classes = row.get("all_classes")
                break
        predictions = [str(row.get("pred", "")) for row in rows]
        answers = [row.get("answers", []) for row in rows]
        lengths = [int(row.get("length", 0) or 0) for row in rows]
        recomputed = float(scorer(task, predictions, answers, lengths, all_classes))
        runner_scores = read_runner_scores(run_dir)
        if abs(recomputed - float(runner_scores[task])) > 1e-9:
            raise RuntimeError(f"independent scorer mismatch for {method} {task}: {recomputed} vs {runner_scores[task]}")
        for expected_index, row in zip(expected, rows):
            sample_index = int(row.get("sample_index", expected_index))
            if sample_index != int(expected_index):
                raise RuntimeError(f"{method} {task} sample mismatch: expected {expected_index}, got {sample_index}")
            pred = str(row.get("pred", ""))
            if not pred.strip():
                raise RuntimeError(f"{method} {task}:{sample_index} produced empty prediction")
            key = (task, sample_index)
            out[key] = {
                "task": task,
                "sample_index": sample_index,
                "sample_id": row.get("sample_id", sample_index),
                "method": method,
                "prediction": pred,
                "gold": row.get("answers", []),
                "score": metric_score(task, pred, row.get("answers", []), all_classes),
                "length": int(row.get("length", 0) or 0),
            }
    return out


def read_runner_scores(run_dir: Path) -> Dict[str, float]:
    path = run_dir / "longbench_scores.csv"
    if not path.exists():
        raise RuntimeError(f"missing score file: {path}")
    scores: Dict[str, float] = {}
    with path.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") == "ok":
                scores[str(row.get("dataset"))] = float(row.get("score"))
    missing = [task for task in TASKS if task not in scores]
    if missing:
        raise RuntimeError(f"missing task scores in {path}: {missing}")
    return scores


def key_of(row: Mapping[str, Any]) -> Key:
    return str(row.get("task", "")), int(row.get("sample_index", -1))


def jaccard(left: Sequence[Any], right: Sequence[Any]) -> float:
    a = {int(x) for x in left}
    b = {int(x) for x in right}
    union = a | b
    return float(len(a & b) / len(union)) if union else 1.0


def summarize_dc_audit(activation_path: Path, selection_path: Path) -> Tuple[Dict[Key, JsonRow], Dict[str, Any]]:
    activation_rows = [
        row for row in read_jsonl(activation_path)
        if str(row.get("phase", "")) == "prefill" and int(row.get("layer_id", -1)) in PRUNE_LAYERS
    ]
    selection_rows = [
        row for row in read_jsonl(selection_path)
        if str(row.get("phase", "")) == "prefill" and int(row.get("layer_id", -1)) in PRUNE_LAYERS
    ]
    if len(activation_rows) != 60:
        raise RuntimeError(f"DC gate activation row count mismatch: {len(activation_rows)} != 60")
    if len(selection_rows) != 60:
        raise RuntimeError(f"DC gate selection row count mismatch: {len(selection_rows)} != 60")

    by_sample: Dict[Key, List[JsonRow]] = defaultdict(list)
    for row in activation_rows:
        by_sample[key_of(row)].append(row)
    per_sample: Dict[Key, JsonRow] = {}
    for key, rows in by_sample.items():
        rows = sorted(rows, key=lambda row: int(row.get("layer_id", -1)))
        per_sample[key] = {
            "selection_changed": any(bool(row.get("selection_changed")) for row in rows),
            "selected_jaccard_vs_original": mean(float(row.get("selected_jaccard", 1.0) or 1.0) for row in rows),
            "original_redundancy": mean(float(row.get("original_redundancy", 0.0) or 0.0) for row in rows),
            "dc_redundancy": mean(float(row.get("dc_redundancy", 0.0) or 0.0) for row in rows),
            "original_query_overlap": mean(float(row.get("original_query_overlap", 0.0) or 0.0) for row in rows),
            "dc_query_overlap": mean(float(row.get("dc_query_overlap", 0.0) or 0.0) for row in rows),
            "original_coverage_bin_count": mean(len(row.get("original_coverage_bins", []) or []) for row in rows),
            "dc_coverage_bin_count": mean(len(row.get("dc_coverage_bins", []) or []) for row in rows),
            "dc_entered_before_virtual_return": any(bool(row.get("entered_dc_selector_before_virtual_return")) for row in rows),
            "virtual_block_path_returned_before_dc_selector": any(
                str(row.get("virtual_return_reason", "")) == "virtual_block_path_returned_before_dc_selector"
                for row in rows
            ),
            "fallback_used": any(bool(row.get("fallback_reason")) for row in rows),
            "budget_mismatch": any(not bool(row.get("budget_preserved", True)) for row in rows),
        }

    selection_by_sample: Dict[Key, List[JsonRow]] = defaultdict(list)
    for row in selection_rows:
        selection_by_sample[key_of(row)].append(row)
    for key, rows in selection_by_sample.items():
        if key not in per_sample:
            continue
        per_sample[key]["selection_audit_fallback_used"] = any(bool(row.get("fallback_used")) for row in rows)
        per_sample[key]["selection_audit_budget_mismatch"] = any(bool(row.get("budget_violation")) for row in rows)

    summary = {
        "activation_rows": len(activation_rows),
        "selection_rows": len(selection_rows),
        "selection_changed_samples": sum(1 for row in per_sample.values() if bool(row.get("selection_changed"))),
        "fallback_count": sum(1 for row in per_sample.values() if bool(row.get("fallback_used") or row.get("selection_audit_fallback_used"))),
        "budget_mismatch_count": sum(1 for row in per_sample.values() if bool(row.get("budget_mismatch") or row.get("selection_audit_budget_mismatch"))),
        "entered_before_virtual_return_samples": sum(1 for row in per_sample.values() if bool(row.get("dc_entered_before_virtual_return"))),
        "virtual_return_before_dc_samples": sum(1 for row in per_sample.values() if bool(row.get("virtual_block_path_returned_before_dc_selector"))),
    }
    return per_sample, summary


def read_final_prefill_usage(run_dir: Path) -> Dict[Key, JsonRow]:
    rows = read_jsonl(run_dir / "audit" / "used_blocks.jsonl")
    selected: Dict[Key, JsonRow] = {}
    for row in rows:
        if str(row.get("phase", "")) != "prefill":
            continue
        if int(row.get("layer_id", -1)) != 27:
            continue
        key = key_of(row)
        if key[0] and key[1] >= 0 and key not in selected:
            selected[key] = row
    missing = [(task, idx) for task in TASKS for idx in FIXED_INDICES[task] if (task, idx) not in selected]
    if missing:
        raise RuntimeError(f"missing final prefill used-block rows: {missing[:10]}")
    return selected


def quality_rows(
    predictions: Mapping[str, Mapping[Key, JsonRow]],
    usage: Mapping[str, Mapping[Key, JsonRow]],
    dc_audit: Mapping[Key, JsonRow],
) -> List[JsonRow]:
    rows: List[JsonRow] = []
    for method in [METHOD_ORIGINAL, METHOD_DC]:
        for task in TASKS:
            for sample_index in FIXED_INDICES[task]:
                key = (task, int(sample_index))
                pred = dict(predictions[method][key])
                used = usage[method][key]
                if method == METHOD_DC:
                    audit = dict(dc_audit[key])
                else:
                    audit = {
                        "selection_changed": False,
                        "selected_jaccard_vs_original": 1.0,
                        "original_redundancy": None,
                        "dc_redundancy": None,
                        "original_query_overlap": None,
                        "dc_query_overlap": None,
                        "original_coverage_bin_count": None,
                        "dc_coverage_bin_count": None,
                        "dc_entered_before_virtual_return": False,
                        "virtual_block_path_returned_before_dc_selector": False,
                        "fallback_used": False,
                        "budget_mismatch": False,
                    }
                selected_token_count = int(used.get("kv_abs_indices_count", 0) or 0)
                row = {
                    **pred,
                    "selected_token_count": selected_token_count,
                    "budget": FINAL_BUDGET_TOKENS,
                    "budget_mismatch": bool(audit.get("budget_mismatch")) or selected_token_count > FINAL_BUDGET_TOKENS,
                    "fallback_used": bool(audit.get("fallback_used")),
                    "selection_changed": bool(audit.get("selection_changed")),
                    "selected_jaccard_vs_original": audit.get("selected_jaccard_vs_original"),
                    "original_redundancy": audit.get("original_redundancy"),
                    "dc_redundancy": audit.get("dc_redundancy"),
                    "original_query_overlap": audit.get("original_query_overlap"),
                    "dc_query_overlap": audit.get("dc_query_overlap"),
                    "original_coverage_bin_count": audit.get("original_coverage_bin_count"),
                    "dc_coverage_bin_count": audit.get("dc_coverage_bin_count"),
                    "dc_entered_before_virtual_return": bool(audit.get("dc_entered_before_virtual_return")),
                    "virtual_block_path_returned_before_dc_selector": bool(audit.get("virtual_block_path_returned_before_dc_selector")),
                }
                rows.append(row)
    if len(rows) != 40:
        raise RuntimeError(f"per-sample quality audit row count mismatch: {len(rows)} != 40")
    return rows


def comparison_rows(predictions: Mapping[str, Mapping[Key, JsonRow]], dc_audit: Mapping[Key, JsonRow]) -> List[JsonRow]:
    rows: List[JsonRow] = []
    for task in TASKS:
        for sample_index in FIXED_INDICES[task]:
            key = (task, int(sample_index))
            original = predictions[METHOD_ORIGINAL][key]
            dc = predictions[METHOD_DC][key]
            rows.append({
                "task": task,
                "sample_index": sample_index,
                "sample_id": original.get("sample_id"),
                "original_score": original["score"],
                "dc_score": dc["score"],
                "delta": float(dc["score"]) - float(original["score"]),
                "original_prediction": original["prediction"],
                "dc_prediction": dc["prediction"],
                "prediction_changed": str(original["prediction"]).strip() != str(dc["prediction"]).strip(),
                "selected_jaccard": dc_audit[key].get("selected_jaccard_vs_original"),
                "selection_changed": bool(dc_audit[key].get("selection_changed")),
            })
    if len(rows) != 20:
        raise RuntimeError(f"comparison row count mismatch: {len(rows)} != 20")
    return rows


def task_scores_from_rows(comparisons: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, float]]:
    scores: Dict[str, Dict[str, float]] = {}
    for task in TASKS:
        task_rows = [row for row in comparisons if row["task"] == task]
        scores[task] = {
            "original": mean(float(row["original_score"]) for row in task_rows),
            "dc": mean(float(row["dc_score"]) for row in task_rows),
        }
        scores[task]["delta"] = scores[task]["dc"] - scores[task]["original"]
    return scores


def correlation(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    x_mean = mean(xs)
    y_mean = mean(ys)
    x_var = sum((x - x_mean) ** 2 for x in xs)
    y_var = sum((y - y_mean) ** 2 for y in ys)
    if x_var <= 1e-12 or y_var <= 1e-12:
        return None
    cov = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    return float(cov / math.sqrt(x_var * y_var))


def summarize_quality(
    comparisons: Sequence[Mapping[str, Any]],
    quality_audit: Sequence[Mapping[str, Any]],
    dc_audit_summary: Mapping[str, Any],
) -> Dict[str, Any]:
    task_scores = task_scores_from_rows(comparisons)
    original_macro = mean(task_scores[task]["original"] for task in TASKS)
    dc_macro = mean(task_scores[task]["dc"] for task in TASKS)
    deltas = [float(row["delta"]) for row in comparisons]
    prediction_changed = [row for row in comparisons if bool(row.get("prediction_changed"))]
    dc_quality_rows = [row for row in quality_audit if row["method"] == METHOD_DC]
    corr = correlation(
        [1.0 - float(row.get("selected_jaccard") or 1.0) for row in comparisons],
        deltas,
    )
    summary = {
        "task_scores": task_scores,
        "macro": {
            "original": original_macro,
            "dc": dc_macro,
            "delta": dc_macro - original_macro,
        },
        "positive_sample_count": sum(1 for value in deltas if value > 1e-9),
        "negative_sample_count": sum(1 for value in deltas if value < -1e-9),
        "zero_sample_count": sum(1 for value in deltas if abs(value) <= 1e-9),
        "musique_positive_sample_count": sum(
            1 for row in comparisons if row["task"] == "musique" and float(row["delta"]) > 1e-9
        ),
        "prediction_changed_count": len(prediction_changed),
        "prediction_change_rate": float(len(prediction_changed) / max(1, len(comparisons))),
        "budget_mismatch_count": sum(1 for row in quality_audit if bool(row.get("budget_mismatch"))),
        "fallback_count": sum(1 for row in dc_quality_rows if bool(row.get("fallback_used"))),
        "dc_activation_rows": int(dc_audit_summary.get("activation_rows", 0)),
        "dc_selection_rows": int(dc_audit_summary.get("selection_rows", 0)),
        "dc_entered_before_virtual_return_samples": int(dc_audit_summary.get("entered_before_virtual_return_samples", 0)),
        "virtual_return_before_dc_samples": int(dc_audit_summary.get("virtual_return_before_dc_samples", 0)),
        "selection_delta_correlation": corr,
        "selection_score_not_fully_negative": corr is None or corr > -0.99,
    }
    return summary


def decide(summary: Mapping[str, Any]) -> Dict[str, Any]:
    reasons: List[str] = []
    bug_reasons: List[str] = []
    task_scores = summary["task_scores"]
    if int(summary["budget_mismatch_count"]) != 0:
        bug_reasons.append("budget_mismatch")
    if int(summary["fallback_count"]) != 0:
        bug_reasons.append("dc_fallback_used")
    if int(summary["dc_activation_rows"]) != 60 or int(summary["dc_selection_rows"]) != 60:
        bug_reasons.append("dc_gate_audit_row_count_unreliable")
    if int(summary["virtual_return_before_dc_samples"]) != 0:
        bug_reasons.append("virtual_block_path_returned_before_dc_selector_nonzero")
    if int(summary["dc_entered_before_virtual_return_samples"]) <= 0:
        bug_reasons.append("dc_did_not_enter_before_virtual_return")
    if bug_reasons:
        return {"decision": "BUG_STOP_FIX_REQUIRED", "reasons": bug_reasons}

    if int(summary["prediction_changed_count"]) <= 0:
        reasons.append("prediction_nearly_unchanged")
    if float(summary["macro"]["dc"]) + 1e-9 < float(summary["macro"]["original"]):
        reasons.append("macro_below_original")
    if float(task_scores["hotpotqa"]["delta"]) < -HOTPOTQA_DROP_TOLERANCE:
        reasons.append("hotpotqa_obvious_drop")
    if int(summary["musique_positive_sample_count"]) <= 0:
        reasons.append("musique_no_positive_samples")
    if not bool(summary["selection_score_not_fully_negative"]):
        reasons.append("selection_change_score_delta_fully_negative")
    return {
        "decision": "QUALITY_GO_TO_64_DIAGNOSTIC" if not reasons else "QUALITY_NO_GO_STOP",
        "reasons": reasons,
    }


def write_report(output_dir: Path, summary: Mapping[str, Any], decision: Mapping[str, Any]) -> None:
    task_scores = summary["task_scores"]
    lines = [
        "# DC-SlimInfer Tiny Quality Smoke",
        "",
        f"- Result directory: `{output_dir}`",
        "- Methods: `sliminfer_original`, `dc_sliminfer`",
        "- Full LongBench: **No**",
        "- 64/128/200 sample diagnostic: **No**",
        "- Threshold sweep: **No**",
        "- DRM-Add/DRM-Swap: **No**",
        "",
        "## Task Scores",
        "",
        f"- HotpotQA original / dc / delta: {task_scores['hotpotqa']['original']:.4f} / {task_scores['hotpotqa']['dc']:.4f} / {task_scores['hotpotqa']['delta']:+.4f}",
        f"- MuSiQue original / dc / delta: {task_scores['musique']['original']:.4f} / {task_scores['musique']['dc']:.4f} / {task_scores['musique']['delta']:+.4f}",
        f"- 2-task macro original / dc / delta: {summary['macro']['original']:.4f} / {summary['macro']['dc']:.4f} / {summary['macro']['delta']:+.4f}",
        "",
        "## Sample Summary",
        "",
        f"- Positive / negative / zero sample count: {summary['positive_sample_count']} / {summary['negative_sample_count']} / {summary['zero_sample_count']}",
        f"- Prediction change rate: {summary['prediction_change_rate']:.2%} ({summary['prediction_changed_count']}/20)",
        f"- Budget mismatch count: {summary['budget_mismatch_count']}",
        f"- Fallback count: {summary['fallback_count']}",
        f"- DC gate activation/selection rows: {summary['dc_activation_rows']} / {summary['dc_selection_rows']}",
        f"- DC entered before virtual return samples: {summary['dc_entered_before_virtual_return_samples']}/20",
        f"- virtual_block_path_returned_before_dc_selector samples: {summary['virtual_return_before_dc_samples']}",
        f"- Selection-change/score-delta correlation: {summary['selection_delta_correlation']}",
        "",
        "## Decision",
        "",
        f"- Final conclusion: **{decision['decision']}**",
    ]
    for reason in decision.get("reasons", []):
        lines.append(f"- Reason: `{reason}`")
    lines.append("")
    lines.append("This is a tiny quality smoke only, not a full LongBench result or a statistical claim.")
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--longbench_dir", type=Path, default=DEFAULT_LONGBENCH_DIR)
    parser.add_argument("--output_root", type=Path, default=Path("results"))
    parser.add_argument("--min_free_gpu_gb", type=float, default=20.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (REPO_ROOT / args.output_root / f"dc_sliminfer_tiny_quality_smoke_{timestamp}").resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json(output_dir / "fixed_samples.json", {
        "hotpotqa": FIXED_HOTPOTQA,
        "musique": FIXED_MUSIQUE,
        "methods": [METHOD_ORIGINAL, METHOD_DC],
        "generation_max_new_tokens": 32,
        "seed": 0,
        "decode_policy": "original",
        "budget_tokens": FINAL_BUDGET_TOKENS,
    })

    gpu_checks = [check_gpu(args.min_free_gpu_gb)]
    activation_path, selection_path = run_dc_gate_audit(args, output_dir)
    dc_audit, dc_audit_summary = summarize_dc_audit(activation_path, selection_path)
    if dc_audit_summary["fallback_count"] != 0 or dc_audit_summary["budget_mismatch_count"] != 0:
        raise RuntimeError(f"DC gate audit failed safety checks: {dc_audit_summary}")

    run_dirs: Dict[str, Path] = {}
    predictions: Dict[str, Dict[Key, JsonRow]] = {}
    usage: Dict[str, Dict[Key, JsonRow]] = {}
    for method in [METHOD_ORIGINAL, METHOD_DC]:
        gpu_checks.append(check_gpu(args.min_free_gpu_gb))
        run_dir = run_quality_method(args, output_dir, method)
        run_dirs[method] = run_dir
        predictions[method] = read_predictions(run_dir, method)
        usage[method] = read_final_prefill_usage(run_dir)

    original_config = json.loads((run_dirs[METHOD_ORIGINAL] / "config.json").read_text(encoding="utf-8"))
    original_variant = (original_config.get("sliminfer_config") or {}).get("variant_method")
    if original_variant == METHOD_DC:
        raise RuntimeError("sliminfer_original run was contaminated with dc_sliminfer variant_method")
    original_dc_audit = run_dirs[METHOD_ORIGINAL] / "audit" / "dc_runtime_activation_audit.jsonl"
    if original_dc_audit.exists() and read_jsonl(original_dc_audit, missing_ok=True):
        raise RuntimeError("sliminfer_original produced DC activation audit rows")

    quality_audit = quality_rows(predictions, usage, dc_audit)
    comparisons = comparison_rows(predictions, dc_audit)
    summary = summarize_quality(comparisons, quality_audit, dc_audit_summary)
    decision = decide(summary)

    write_jsonl(output_dir / "per_sample_predictions.jsonl", quality_audit)
    write_jsonl(output_dir / "per_sample_comparison.jsonl", comparisons)
    write_csv(
        output_dir / "per_sample_comparison.csv",
        comparisons,
        [
            "task",
            "sample_index",
            "sample_id",
            "original_score",
            "dc_score",
            "delta",
            "original_prediction",
            "dc_prediction",
            "prediction_changed",
            "selected_jaccard",
            "selection_changed",
        ],
    )
    score_rows = [
        {
            "task": task,
            "original": summary["task_scores"][task]["original"],
            "dc": summary["task_scores"][task]["dc"],
            "delta": summary["task_scores"][task]["delta"],
        }
        for task in TASKS
    ]
    score_rows.append({"task": "macro", **summary["macro"]})
    write_csv(output_dir / "task_summary.csv", score_rows, ["task", "original", "dc", "delta"])
    write_json(output_dir / "summary.json", {
        "result_dir": str(output_dir),
        "run_dirs": {method: str(path) for method, path in run_dirs.items()},
        "gpu_checks": gpu_checks,
        "summary": summary,
        "decision": decision,
        "quality_audit_rows": len(quality_audit),
        "comparison_rows": len(comparisons),
        "original_not_dc_contaminated": True,
        "predictions_exist": True,
        "scores_recomputed_zero_diff": True,
    })
    write_report(output_dir, summary, decision)
    print(json.dumps({
        "status": "ok",
        "output_dir": str(output_dir),
        "decision": decision,
        "macro": summary["macro"],
        "prediction_changed_count": summary["prediction_changed_count"],
        "budget_mismatch_count": summary["budget_mismatch_count"],
        "fallback_count": summary["fallback_count"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
