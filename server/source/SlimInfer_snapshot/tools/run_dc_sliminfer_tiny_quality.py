#!/usr/bin/env python3
"""Run the fixed 20-sample DC-SlimInfer audit and tiny quality smoke."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


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
FIXED_HOTPOTQA = [0, 1, 2, 3, 4, 5, 41, 56, 57, 117]
FIXED_MUSIQUE = [0, 1, 2, 3, 4, 20, 50, 54, 56, 134]
FIXED_SPEC = (
    "hotpotqa:" + ",".join(str(x) for x in FIXED_HOTPOTQA)
    + ";musique:" + ",".join(str(x) for x in FIXED_MUSIQUE)
)
TASKS = ["hotpotqa", "musique"]
FALLBACK_MAX_RATE = 0.20
TASK_COLLAPSE_TOLERANCE = 5.0

JsonRow = Dict[str, Any]


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


def read_jsonl(path: Path) -> List[JsonRow]:
    if not path.exists():
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


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="ignore")).hexdigest()


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
        "--longbench_subset_name", "dc_sliminfer_tiny",
        "--generation_max_new_tokens", "32",
        "--max_model_len", "32768",
        "--decode_policy", "original",
        "--rescue_policy", "none",
        "--prune_layers", "9,18,26",
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


def run_selection_audit(args: argparse.Namespace, output_dir: Path) -> Path:
    audit_path = output_dir / "selection_audit.jsonl"
    audit_path.write_text("", encoding="utf-8")
    run_dir = output_dir / "run_dc_selection_audit"
    cmd = base_runner_cmd(args, run_dir) + [
        "--eval", "gate_audit",
        "--method", METHOD_DC,
        "--dc_selection_audit_path", str(audit_path),
    ]
    write_json(output_dir / "selection_audit_command.json", {"command": cmd})
    run_command(
        cmd,
        REPO_ROOT,
        output_dir / "selection_audit_stdout.log",
        output_dir / "selection_audit_stderr.log",
    )
    rows = read_jsonl(audit_path)
    if not rows:
        raise RuntimeError("DC selection audit produced no rows")
    return audit_path


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
        "--debug_used_block_full_layers", "10,19,27",
    ]
    write_json(output_dir / f"{method}_quality_command.json", {"command": cmd})
    run_command(
        cmd,
        REPO_ROOT,
        output_dir / f"{method}_quality_stdout.log",
        output_dir / f"{method}_quality_stderr.log",
    )
    return run_dir


def per_sample_score(task: str, prediction: str, answers: Sequence[Any], all_classes: Sequence[Any]) -> float:
    score = 0.0
    pred = prediction
    if task in ["trec", "triviaqa", "samsum", "lsht"]:
        pred = pred.lstrip("\n").split("\n")[0]
    for answer in answers:
        score = max(score, dataset2metric[task](pred, answer, all_classes=all_classes))
    return float(100.0 * score)


def read_predictions(run_dir: Path, method: str) -> Tuple[List[JsonRow], Dict[str, float]]:
    rows: List[JsonRow] = []
    task_scores: Dict[str, float] = {}
    for task in TASKS:
        pred_path = run_dir / "longbench_predictions" / f"{task}.jsonl"
        pred_rows = read_jsonl(pred_path)
        expected = FIXED_HOTPOTQA if task == "hotpotqa" else FIXED_MUSIQUE
        if len(pred_rows) != len(expected):
            raise RuntimeError(f"{method} {task} predictions count mismatch: {len(pred_rows)} != {len(expected)}")
        predictions = [str(row.get("pred", "")) for row in pred_rows]
        answers = [row.get("answers", []) for row in pred_rows]
        lengths = [int(row.get("length", 0) or 0) for row in pred_rows]
        all_classes: List[Any] = []
        for row in pred_rows:
            if row.get("all_classes"):
                all_classes = row.get("all_classes", all_classes)
        task_scores[task] = float(scorer(task, predictions, answers, lengths, all_classes))
        for expected_index, row in zip(expected, pred_rows):
            sample_index = int(row.get("sample_index", expected_index))
            if sample_index != int(expected_index):
                raise RuntimeError(f"{method} {task} sample mismatch: expected {expected_index}, got {sample_index}")
            pred = str(row.get("pred", ""))
            if pred == "":
                raise RuntimeError(f"{method} {task}:{sample_index} produced empty prediction")
            rows.append({
                "method": method,
                "task": task,
                "sample_index": sample_index,
                "sample_id": row.get("sample_id", sample_index),
                "prediction": pred,
                "gold": row.get("answers", []),
                "score": per_sample_score(task, pred, row.get("answers", []), all_classes),
                "used_tokens": int(row.get("length", 0) or 0),
            })
    return rows, task_scores


def read_runner_scores(run_dir: Path) -> Dict[str, float]:
    score_path = run_dir / "longbench_scores.csv"
    if not score_path.exists():
        raise RuntimeError(f"missing score file: {score_path}")
    scores: Dict[str, float] = {}
    with score_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("status") == "ok":
                scores[str(row.get("dataset"))] = float(row.get("score"))
    return scores


def mean(values: Iterable[float]) -> float:
    items = [float(value) for value in values if math.isfinite(float(value))]
    return float(sum(items) / len(items)) if items else 0.0


def summarize_selection_audit(path: Path) -> Dict[str, Any]:
    rows = read_jsonl(path)
    total = len(rows)
    changed_rows = [row for row in rows if bool(row.get("selection_changed"))]
    fallback_rows = [row for row in rows if bool(row.get("fallback_used"))]
    budget_violations = [row for row in rows if bool(row.get("budget_violation")) or not bool(row.get("budget_preserved", True))]
    redundancy_original = mean(float(row.get("selected_redundancy_original", 0.0) or 0.0) for row in rows)
    redundancy_dc = mean(float(row.get("selected_redundancy_dc", 0.0) or 0.0) for row in rows)
    query_original = mean(float(row.get("query_overlap_original", 0.0) or 0.0) for row in rows)
    query_dc = mean(float(row.get("query_overlap_dc", 0.0) or 0.0) for row in rows)
    coverage_original = mean(float(row.get("coverage_bin_count_original", 0.0) or 0.0) for row in rows)
    coverage_dc = mean(float(row.get("coverage_bin_count_dc", 0.0) or 0.0) for row in rows)
    changed_samples = sorted({
        f"{row.get('task')}:{row.get('sample_index')}"
        for row in changed_rows
    })
    fallback_reasons: Dict[str, int] = {}
    for row in fallback_rows:
        reason = str(row.get("fallback_reason", "") or "unknown")
        fallback_reasons[reason] = fallback_reasons.get(reason, 0) + 1
    return {
        "row_count": total,
        "selection_changed_count": int(len(changed_rows)),
        "selection_changed_sample_count": int(len(changed_samples)),
        "selection_changed_samples": changed_samples,
        "selected_redundancy_original": redundancy_original,
        "selected_redundancy_dc": redundancy_dc,
        "selected_redundancy_delta": redundancy_dc - redundancy_original,
        "query_overlap_original": query_original,
        "query_overlap_dc": query_dc,
        "query_overlap_delta": query_dc - query_original,
        "coverage_bins_original_mean": coverage_original,
        "coverage_bins_dc_mean": coverage_dc,
        "coverage_bins_delta": coverage_dc - coverage_original,
        "budget_violation_count": int(len(budget_violations)),
        "fallback_count": int(len(fallback_rows)),
        "fallback_rate": float(len(fallback_rows) / max(1, total)),
        "fallback_reasons": fallback_reasons,
    }


def score_table(task_scores_by_method: Mapping[str, Mapping[str, float]]) -> List[JsonRow]:
    rows: List[JsonRow] = []
    for method, scores in task_scores_by_method.items():
        macro = mean(scores[task] for task in TASKS)
        rows.append({
            "method": method,
            "hotpotqa": scores.get("hotpotqa", ""),
            "musique": scores.get("musique", ""),
            "macro": macro,
        })
    return rows


def write_report(output_dir: Path, selection: Mapping[str, Any], scores: Mapping[str, Mapping[str, float]], decision: Mapping[str, Any]) -> None:
    original_macro = mean(scores[METHOD_ORIGINAL][task] for task in TASKS)
    dc_macro = mean(scores[METHOD_DC][task] for task in TASKS)
    lines = [
        "# DC-SlimInfer Tiny Quality",
        "",
        f"- Result directory: `{output_dir}`",
        "- Methods run: `sliminfer_original`, `dc_sliminfer`",
        "- Full LongBench run: **No**",
        "- Threshold sweep: **No**",
        "- Historical DRM proxy used: **No**",
        "- Gold/final score/final prediction used for selection: **No**",
        "",
        "## Selection Audit",
        "",
        f"- Layer rows: {selection['row_count']}",
        f"- Selection changed count: {selection['selection_changed_count']}",
        f"- Changed samples: `{json.dumps(selection['selection_changed_samples'], ensure_ascii=False)}`",
        f"- Redundancy original: {selection['selected_redundancy_original']:.6f}",
        f"- Redundancy DC: {selection['selected_redundancy_dc']:.6f}",
        f"- Query overlap original: {selection['query_overlap_original']:.6f}",
        f"- Query overlap DC: {selection['query_overlap_dc']:.6f}",
        f"- Coverage bins original mean: {selection['coverage_bins_original_mean']:.4f}",
        f"- Coverage bins DC mean: {selection['coverage_bins_dc_mean']:.4f}",
        f"- Budget violation: {selection['budget_violation_count']}",
        f"- Fallback count: {selection['fallback_count']} ({selection['fallback_rate']:.2%})",
        "",
        "## Scores",
        "",
        f"- Original task scores: `{json.dumps(scores[METHOD_ORIGINAL], ensure_ascii=False, sort_keys=True)}`",
        f"- DC task scores: `{json.dumps(scores[METHOD_DC], ensure_ascii=False, sort_keys=True)}`",
        f"- Original 2-task macro: {original_macro:.4f}",
        f"- DC 2-task macro: {dc_macro:.4f}",
        f"- DC vs Original macro: {dc_macro - original_macro:+.4f}",
        f"- HotpotQA delta: {scores[METHOD_DC]['hotpotqa'] - scores[METHOD_ORIGINAL]['hotpotqa']:+.4f}",
        f"- MuSiQue delta: {scores[METHOD_DC]['musique'] - scores[METHOD_ORIGINAL]['musique']:+.4f}",
        "",
        "## GO / NO-GO",
        "",
        f"- Decision: **{decision['decision']}**",
        f"- Recommend 64-sample diagnostic: {decision['recommend_64_sample_diagnostic']}",
    ]
    for reason in decision.get("reasons", []):
        lines.append(f"- Reason: `{reason}`")
    lines.append("")
    lines.append("This is a tiny quality smoke, not a full LongBench result and not a statistical claim.")
    (output_dir / "dc_sliminfer_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def decide(selection: Mapping[str, Any], scores: Mapping[str, Mapping[str, float]]) -> Dict[str, Any]:
    reasons: List[str] = []
    original_macro = mean(scores[METHOD_ORIGINAL][task] for task in TASKS)
    dc_macro = mean(scores[METHOD_DC][task] for task in TASKS)
    task_deltas = {
        task: float(scores[METHOD_DC][task] - scores[METHOD_ORIGINAL][task])
        for task in TASKS
    }
    if int(selection["selection_changed_count"]) <= 0:
        reasons.append("selection_unchanged")
    if float(selection["selected_redundancy_delta"]) >= 0.0:
        reasons.append("redundancy_not_reduced")
    if (
        float(selection["query_overlap_delta"]) <= 0.0
        and float(selection["coverage_bins_delta"]) <= 0.0
    ):
        reasons.append("query_and_coverage_not_improved")
    if int(selection["budget_violation_count"]) != 0:
        reasons.append("budget_violation")
    if float(selection["fallback_rate"]) > FALLBACK_MAX_RATE:
        reasons.append("fallback_rate_above_20pct")
    if dc_macro < original_macro:
        reasons.append("quality_macro_below_original")
    if max(task_deltas.values()) <= 0.0:
        reasons.append("no_task_improved")
    if min(task_deltas.values()) < -TASK_COLLAPSE_TOLERANCE:
        reasons.append("task_obvious_collapse")
    return {
        "decision": "GO" if not reasons else "NO-GO",
        "reasons": reasons,
        "recommend_64_sample_diagnostic": bool(not reasons),
        "task_deltas": task_deltas,
        "original_macro": original_macro,
        "dc_macro": dc_macro,
    }


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
    output_dir = (REPO_ROOT / args.output_root / f"dc_sliminfer_tiny_{timestamp}").resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json(output_dir / "fixed_samples.json", {
        "hotpotqa": FIXED_HOTPOTQA,
        "musique": FIXED_MUSIQUE,
        "generation_max_new_tokens": 32,
        "seed": 0,
        "dc_params": {"alpha": 0.10, "beta": 0.05, "lambda": 0.15, "gamma": 0.05},
    })

    gpu_checks: List[Dict[str, Any]] = []
    gpu_checks.append(check_gpu(args.min_free_gpu_gb))
    selection_audit_path = run_selection_audit(args, output_dir)
    selection_summary = summarize_selection_audit(selection_audit_path)

    run_dirs: Dict[str, Path] = {}
    all_prediction_rows: List[JsonRow] = []
    task_scores_by_method: Dict[str, Dict[str, float]] = {}
    runner_scores_by_method: Dict[str, Dict[str, float]] = {}
    for method in [METHOD_ORIGINAL, METHOD_DC]:
        gpu_checks.append(check_gpu(args.min_free_gpu_gb))
        run_dir = run_quality_method(args, output_dir, method)
        run_dirs[method] = run_dir
        prediction_rows, task_scores = read_predictions(run_dir, method)
        runner_scores = read_runner_scores(run_dir)
        for task in TASKS:
            if abs(float(task_scores[task]) - float(runner_scores[task])) > 1e-9:
                raise RuntimeError(
                    f"independent scorer mismatch for {method} {task}: "
                    f"{task_scores[task]} vs runner {runner_scores[task]}"
                )
        task_scores_by_method[method] = task_scores
        runner_scores_by_method[method] = runner_scores
        all_prediction_rows.extend(prediction_rows)

    table = score_table(task_scores_by_method)
    decision = decide(selection_summary, task_scores_by_method)
    summary = {
        "result_dir": str(output_dir),
        "selection_audit": selection_summary,
        "task_scores": task_scores_by_method,
        "runner_scores": runner_scores_by_method,
        "score_table": table,
        "go_no_go": decision,
        "run_dirs": {method: str(path) for method, path in run_dirs.items()},
        "gpu_checks": gpu_checks,
        "config_hash": sha256_text(json.dumps({
            "fixed_samples": FIXED_SPEC,
            "methods": [METHOD_ORIGINAL, METHOD_DC],
            "generation_max_new_tokens": 32,
            "seed": 0,
            "dc_params": [0.10, 0.05, 0.15, 0.05],
        }, sort_keys=True)),
    }
    write_jsonl(output_dir / "per_sample_predictions.jsonl", all_prediction_rows)
    write_csv(output_dir / "method_task_scores.csv", table, ["method", "hotpotqa", "musique", "macro"])
    write_json(output_dir / "dc_sliminfer_summary.json", summary)
    write_report(output_dir, selection_summary, task_scores_by_method, decision)
    print(json.dumps({
        "status": "ok",
        "output_dir": str(output_dir),
        "go_no_go": decision,
        "dc_macro": decision["dc_macro"],
        "original_macro": decision["original_macro"],
        "selection_changed_count": selection_summary["selection_changed_count"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
