#!/usr/bin/env python3
"""MCU-SlimInfer tiny quality smoke on fixed 12 samples.

Compares only:
  1. sliminfer_original
  2. mcu_sliminfer
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from longbench_eval import dataset2metric, scorer
from tools.mcu_sliminfer_selection import (
    load_frozen_selections_from_audit_csv,
    write_frozen_selections_jsonl,
)

DEFAULT_MODEL_PATH = Path(
    "/home/yu-bai/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/"
    "snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
)
DEFAULT_LONGBENCH_DIR = Path("/home/yu-bai/Desktop/jiaqizhang/HeadKV/data/LongBench")
DEFAULT_AUDIT_DIR = Path("results/mcu_sliminfer_counterfactual_audit_20260708_203500")
METHOD_ORIGINAL = "sliminfer_original"
METHOD_MCU = "mcu_sliminfer"
ALL_METHODS = [METHOD_ORIGINAL, METHOD_MCU]
TASKS = ["hotpotqa", "musique"]
PRUNE_LAYERS = [9, 18, 26]
FINAL_BUDGET_TOKENS = 4096
TASK_DROP_TOLERANCE = 5.0

MCU_SAMPLE_INDICES: Dict[str, List[int]] = {
    "hotpotqa": [0, 1, 2, 3, 4, 5],
    "musique": [0, 1, 2, 3, 4, 20],
}

Key = Tuple[str, int]
JsonRow = Dict[str, Any]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
    items = [float(v) for v in values if math.isfinite(float(v))]
    return float(sum(items) / len(items)) if items else 0.0


def python_bin() -> Path:
    candidate = REPO_ROOT / ".venv" / "bin" / "python"
    return candidate if candidate.exists() else Path(sys.executable)


def build_sample_spec() -> Tuple[str, Dict[str, List[int]]]:
    spec = ";".join(f"{task}:" + ",".join(str(i) for i in MCU_SAMPLE_INDICES[task]) for task in TASKS)
    return spec, dict(MCU_SAMPLE_INDICES)


def sample_ids_from_frozen(frozen: Mapping[str, Mapping[str, Any]]) -> List[str]:
    ids: List[str] = []
    for task in TASKS:
        for idx in MCU_SAMPLE_INDICES[task]:
            row = frozen.get(f"{task}:{idx}", {})
            ids.append(str(row.get("sample_id", f"{task}:{idx}")))
    return ids


def base_runner_cmd(args: argparse.Namespace, output_dir: Path, sample_spec: str) -> List[str]:
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
        "--longbench_sample_indices_by_task", sample_spec,
        "--longbench_subset_name", "mcu_sliminfer_tiny_quality_smoke",
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
        "--output_dir", str(output_dir),
        "--min_free_gpu_gb", "0",
    ]


def run_command(cmd: Sequence[str], cwd: Path, stdout_path: Path, stderr_path: Path) -> None:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8") as so, stderr_path.open("w", encoding="utf-8") as se:
        proc = subprocess.run(list(cmd), cwd=str(cwd), text=True, stdout=so, stderr=se)
    stdout_text = stdout_path.read_text(encoding="utf-8", errors="ignore")
    stderr_text = stderr_path.read_text(encoding="utf-8", errors="ignore")
    if "CUDA out of memory" in stdout_text or "CUDA out of memory" in stderr_text:
        raise RuntimeError(f"CUDA OOM: {' '.join(cmd)}")
    if proc.returncode != 0:
        raise RuntimeError(f"runner failed rc={proc.returncode}; see {stderr_path}")


def run_quality_method(
    args: argparse.Namespace,
    output_dir: Path,
    method: str,
    sample_spec: str,
    frozen_path: Optional[Path] = None,
) -> Path:
    run_dir = output_dir / f"run_{method}"
    audit_dir = run_dir / "audit"
    audit_dir.mkdir(parents=True, exist_ok=False)
    cmd = base_runner_cmd(args, run_dir, sample_spec) + [
        "--eval", "longbench",
        "--method", method,
        "--debug_generation_jsonl", str(audit_dir / "generations.jsonl"),
        "--debug_used_block_audit_path", str(audit_dir / "used_blocks.jsonl"),
        "--debug_used_block_include_full", "true",
        "--debug_used_block_full_layers", "27",
    ]
    if method == METHOD_MCU and frozen_path is not None:
        mcu_audit = audit_dir / "mcu_runtime_selection_audit.jsonl"
        mcu_audit.write_text("", encoding="utf-8")
        cmd.extend([
            "--mcu_frozen_selection_path", str(frozen_path),
            "--mcu_selection_audit_path", str(mcu_audit),
        ])
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
        raise RuntimeError(f"missing task scores: {missing}")
    return scores


def read_predictions(run_dir: Path, method: str, expected_indices: Dict[str, List[int]]) -> Dict[Key, JsonRow]:
    out: Dict[Key, JsonRow] = {}
    for task in TASKS:
        pred_path = run_dir / "longbench_predictions" / f"{task}.jsonl"
        rows = read_jsonl(pred_path)
        expected = expected_indices[task]
        if len(rows) != len(expected):
            raise RuntimeError(f"{method} {task} count mismatch: {len(rows)} != {len(expected)}")
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
            raise RuntimeError(f"scorer mismatch {method} {task}: {recomputed} vs {runner_scores[task]}")
        for expected_idx, row in zip(expected, rows):
            sample_index = int(row.get("sample_index", expected_idx))
            if sample_index != int(expected_idx):
                raise RuntimeError(f"{method} {task} sample mismatch: expected {expected_idx}, got {sample_index}")
            pred = str(row.get("pred", ""))
            if not pred.strip():
                raise RuntimeError(f"{method} {task}:{sample_index} empty prediction")
            out[(task, sample_index)] = {
                "task": task,
                "sample_index": sample_index,
                "sample_id": row.get("sample_id", sample_index),
                "method": method,
                "prediction": pred,
                "score": metric_score(task, pred, row.get("answers", []), all_classes),
                "length": int(row.get("length", 0) or 0),
            }
    return out


def key_of(row: Mapping[str, Any]) -> Key:
    return str(row.get("task", "")), int(row.get("sample_index", -1))


def read_final_prefill_usage(run_dir: Path, expected_indices: Dict[str, List[int]]) -> Dict[Key, JsonRow]:
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
    missing = [(task, idx) for task in TASKS for idx in expected_indices[task] if (task, idx) not in selected]
    if missing:
        raise RuntimeError(f"missing final prefill used-block rows: {missing[:10]}")
    return selected


def read_mcu_runtime_audit(run_dir: Path, expected_indices: Dict[str, List[int]], frozen: Mapping[str, Mapping[str, Any]]) -> Dict[Key, JsonRow]:
    path = run_dir / "audit" / "mcu_runtime_selection_audit.jsonl"
    rows = read_jsonl(path, missing_ok=True)
    by_key: Dict[Key, JsonRow] = {}
    for row in rows:
        if int(row.get("layer_id", -1)) != PRUNE_LAYERS[-1]:
            continue
        if str(row.get("phase", "")) != "prefill":
            continue
        key = key_of(row)
        if key[0] and key[1] >= 0 and key not in by_key:
            by_key[key] = row

    usage = read_final_prefill_usage(run_dir, expected_indices)
    for task in TASKS:
        for sample_index in expected_indices[task]:
            key = (task, sample_index)
            if key in by_key:
                continue
            frozen_row = frozen.get(f"{task}:{sample_index}", {})
            used = usage.get(key, {})
            by_key[key] = {
                "task": task,
                "sample_index": sample_index,
                "sample_id": frozen_row.get("sample_id", used.get("sample_id", sample_index)),
                "fallback_used": bool(frozen_row.get("fallback_used")),
                "fallback_reason": frozen_row.get("fallback_reason", "missing_runtime_mcu_audit_row"),
                "selection_changed": bool(frozen_row.get("selection_changed")),
                "selected_jaccard_vs_original": frozen_row.get("selected_jaccard_vs_original", 1.0),
                "utility_margin": frozen_row.get("utility_margin", 0.0),
                "removed_block_ids": frozen_row.get("removed_block_ids", []),
                "added_block_ids": frozen_row.get("added_block_ids", []),
                "budget_preserved": frozen_row.get("budget_preserved", True),
            }
    return by_key


def compute_method_summary(predictions: Dict[str, Dict[Key, JsonRow]], expected_indices: Dict[str, List[int]]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for method, pred_map in predictions.items():
        task_scores: Dict[str, float] = {}
        for task in TASKS:
            scores = [float(pred_map[(task, idx)]["score"]) for idx in expected_indices[task]]
            task_scores[task] = mean(scores)
        task_scores["macro"] = mean(task_scores[t] for t in TASKS)
        out[method] = task_scores
    return out


def build_per_sample_quality(
    predictions: Dict[str, Dict[Key, JsonRow]],
    usage: Dict[str, Dict[Key, JsonRow]],
    mcu_audit: Dict[Key, JsonRow],
    frozen: Mapping[str, Mapping[str, Any]],
    expected_indices: Dict[str, List[int]],
) -> List[JsonRow]:
    rows: List[JsonRow] = []
    for task in TASKS:
        for sample_index in expected_indices[task]:
            key = (task, sample_index)
            orig = predictions[METHOD_ORIGINAL][key]
            mcu = predictions[METHOD_MCU][key]
            used_orig = usage[METHOD_ORIGINAL].get(key, {})
            used_mcu = usage[METHOD_MCU].get(key, {})
            audit = mcu_audit.get(key, {})
            frozen_row = frozen.get(f"{task}:{sample_index}", {})
            score_delta = float(mcu["score"]) - float(orig["score"])
            rows.append({
                "task": task,
                "sample_index": sample_index,
                "sample_id": orig.get("sample_id", sample_index),
                "method": METHOD_MCU,
                "prediction": mcu["prediction"],
                "score": mcu["score"],
                "original_score": orig["score"],
                "mcu_score": mcu["score"],
                "score_delta": score_delta,
                "selected_token_count": int(used_mcu.get("final_token_count_used_by_attention", 0) or 0),
                "budget": FINAL_BUDGET_TOKENS,
                "budget_mismatch": not bool(audit.get("budget_preserved", frozen_row.get("budget_preserved", True))),
                "fallback_used": bool(audit.get("fallback_used", frozen_row.get("fallback_used"))),
                "fallback_reason": audit.get("fallback_reason", frozen_row.get("fallback_reason", "")),
                "selection_changed": bool(audit.get("selection_changed", frozen_row.get("selection_changed"))),
                "selected_jaccard_vs_original": audit.get("selected_jaccard_vs_original", frozen_row.get("selected_jaccard_vs_original", 1.0)),
                "original_selected_block_ids": frozen_row.get("original_selected_block_ids", []),
                "mcu_selected_block_ids": frozen_row.get("mcu_selected_block_ids", []),
                "removed_block_ids": audit.get("removed_block_ids", frozen_row.get("removed_block_ids", [])),
                "added_block_ids": audit.get("added_block_ids", frozen_row.get("added_block_ids", [])),
                "removed_block_utility": frozen_row.get("removed_block_utility", []),
                "added_block_utility": frozen_row.get("added_block_utility", []),
                "utility_margin": audit.get("utility_margin", frozen_row.get("utility_margin", 0.0)),
                "hidden_change_mean": frozen_row.get("hidden_change_mean", audit.get("hidden_change_mean", 0.0)),
                "logit_kl_mean": frozen_row.get("logit_kl_mean", audit.get("logit_kl_mean", 0.0)),
                "entropy_change_mean": frozen_row.get("entropy_change_mean", audit.get("entropy_change_mean", 0.0)),
                "top_logit_margin_change_mean": frozen_row.get("top_logit_margin_change_mean", audit.get("top_logit_margin_change_mean", 0.0)),
                "probe_runtime_ms": frozen_row.get("probe_runtime_ms", audit.get("probe_runtime_ms", 0.0)),
                "original_query_overlap": frozen_row.get("original_query_overlap", audit.get("original_query_overlap", 0.0)),
                "mcu_query_overlap": frozen_row.get("mcu_query_overlap", audit.get("mcu_query_overlap", 0.0)),
                "query_overlap_delta": frozen_row.get("query_overlap_delta", audit.get("query_overlap_delta", 0.0)),
                "original_redundancy": frozen_row.get("original_redundancy", audit.get("original_redundancy", 0.0)),
                "mcu_redundancy": frozen_row.get("mcu_redundancy", audit.get("mcu_redundancy", 0.0)),
                "redundancy_delta": frozen_row.get("redundancy_delta", audit.get("redundancy_delta", 0.0)),
                "original_selected_token_count": int(used_orig.get("final_token_count_used_by_attention", 0) or 0),
            })
    return rows


def summarize_mcu_audit(per_sample: Sequence[JsonRow]) -> Dict[str, Any]:
    mcu_rows = list(per_sample)
    return {
        "budget_mismatch_count": sum(1 for r in mcu_rows if r.get("budget_mismatch")),
        "fallback_count": sum(1 for r in mcu_rows if r.get("fallback_used")),
        "selection_changed_count": sum(1 for r in mcu_rows if r.get("selection_changed")),
        "mean_probe_runtime_ms": mean(float(r.get("probe_runtime_ms", 0.0) or 0.0) for r in mcu_rows),
        "mean_utility_margin": mean(float(r.get("utility_margin", 0.0) or 0.0) for r in mcu_rows),
        "selection_changed_sample_ids": [r.get("sample_id") for r in mcu_rows if r.get("selection_changed")],
    }


def decide(
    method_scores: Dict[str, Dict[str, float]],
    per_sample: Sequence[JsonRow],
    audit_summary: Dict[str, Any],
    original_pollution_count: int,
) -> Dict[str, Any]:
    bug_reasons: List[str] = []
    no_go_reasons: List[str] = []
    weak_reasons: List[str] = []

    orig_macro = float(method_scores[METHOD_ORIGINAL]["macro"])
    mcu_macro = float(method_scores[METHOD_MCU]["macro"])
    macro_delta = mcu_macro - orig_macro
    hotpot_delta = float(method_scores[METHOD_MCU]["hotpotqa"]) - float(method_scores[METHOD_ORIGINAL]["hotpotqa"])
    musique_delta = float(method_scores[METHOD_MCU]["musique"]) - float(method_scores[METHOD_ORIGINAL]["musique"])

    if audit_summary.get("budget_mismatch_count", 0) > 0:
        bug_reasons.append(f"budget_mismatch_count={audit_summary['budget_mismatch_count']}")
    if original_pollution_count:
        bug_reasons.append(f"original_selector_pollution={original_pollution_count}")
    if len(per_sample) != sum(len(v) for v in MCU_SAMPLE_INDICES.values()):
        bug_reasons.append("per_sample_row_count_mismatch")
    empty_preds = [r for r in per_sample if not str(r.get("prediction", "")).strip()]
    if empty_preds:
        bug_reasons.append(f"empty_predictions={len(empty_preds)}")

    margin_score_pairs = []
    for row in per_sample:
        try:
            margin_score_pairs.append((float(row.get("utility_margin", 0.0) or 0.0), float(row.get("score_delta", 0.0) or 0.0)))
        except (TypeError, ValueError):
            continue
    neg_corr_count = sum(1 for margin, delta in margin_score_pairs if margin > 0.2 and delta < -2.0)

    if bug_reasons:
        return {"decision": "BUG_STOP_FIX_REQUIRED", "reasons": bug_reasons, "macro_delta": macro_delta}

    if audit_summary.get("fallback_count", 0) > len(per_sample) // 2:
        bug_reasons.append(f"fallback_count={audit_summary['fallback_count']}")
        return {"decision": "BUG_STOP_FIX_REQUIRED", "reasons": bug_reasons, "macro_delta": macro_delta}

    if mcu_macro <= orig_macro:
        no_go_reasons.append("mcu_macro_not_above_original")
    if hotpot_delta <= -TASK_DROP_TOLERANCE:
        no_go_reasons.append(f"hotpotqa_task_drop={hotpot_delta:.2f}")
    if musique_delta <= -TASK_DROP_TOLERANCE:
        no_go_reasons.append(f"musique_task_drop={musique_delta:.2f}")
    if neg_corr_count >= max(3, len(per_sample) // 3):
        no_go_reasons.append(f"utility_margin_score_negative_correlation={neg_corr_count}")

    go_ready = (
        mcu_macro > orig_macro
        and hotpot_delta > -TASK_DROP_TOLERANCE
        and musique_delta > -TASK_DROP_TOLERANCE
        and audit_summary.get("budget_mismatch_count", 0) == 0
        and audit_summary.get("fallback_count", 0) <= max(2, len(per_sample) // 4)
    )
    if go_ready and not no_go_reasons:
        return {
            "decision": "MCU_TINY_QUALITY_GO",
            "reasons": [
                f"macro_delta={macro_delta:+.4f}",
                f"hotpot_delta={hotpot_delta:+.4f}",
                f"musique_delta={musique_delta:+.4f}",
            ],
            "macro_delta": macro_delta,
            "worth_heldout_diagnostic": True,
        }

    if mcu_macro > orig_macro or (hotpot_delta > 0 and musique_delta > -TASK_DROP_TOLERANCE):
        weak_reasons.append("partial_positive_signal")
    if no_go_reasons:
        return {
            "decision": "MCU_TINY_QUALITY_NO_GO",
            "reasons": no_go_reasons,
            "macro_delta": macro_delta,
            "worth_heldout_diagnostic": False,
        }
    return {
        "decision": "MCU_TINY_QUALITY_WEAK_GO",
        "reasons": weak_reasons or [f"macro_delta={macro_delta:+.4f}"],
        "macro_delta": macro_delta,
        "worth_heldout_diagnostic": True,
    }


def build_case_study(per_sample: Sequence[JsonRow], top_n: int = 4) -> str:
    changed = sorted(
        [r for r in per_sample if r.get("selection_changed")],
        key=lambda r: -float(r.get("utility_margin", 0.0) or 0.0),
    )[:top_n]
    lines = ["## Added/Removed Block Case Study", ""]
    for row in changed:
        lines.extend([
            f"### {row['task']}:{row['sample_index']} ({row.get('sample_id', '')})",
            f"- utility margin: {float(row.get('utility_margin') or 0.0):.3f}",
            f"- score delta (MCU-original): {float(row.get('score_delta') or 0.0):+.2f}",
            f"- removed: {row.get('removed_block_ids')}",
            f"- added: {row.get('added_block_ids')}",
            "",
        ])
    if not changed:
        lines.append("_No selection changes in smoke set._")
    return "\n".join(lines)


def build_report(
    output_dir: Path,
    sample_ids: Sequence[str],
    method_scores: Dict[str, Dict[str, float]],
    per_sample: Sequence[JsonRow],
    audit_summary: Dict[str, Any],
    decision: Dict[str, Any],
) -> str:
    hotpot_orig = method_scores[METHOD_ORIGINAL]["hotpotqa"]
    hotpot_mcu = method_scores[METHOD_MCU]["hotpotqa"]
    musique_orig = method_scores[METHOD_ORIGINAL]["musique"]
    musique_mcu = method_scores[METHOD_MCU]["musique"]
    macro_orig = method_scores[METHOD_ORIGINAL]["macro"]
    macro_mcu = method_scores[METHOD_MCU]["macro"]

    margin_score_pairs = [
        (float(r.get("utility_margin", 0.0) or 0.0), float(r.get("score_delta", 0.0) or 0.0))
        for r in per_sample
    ]
    pos_margin_pos_score = sum(1 for m, d in margin_score_pairs if m > 0 and d > 0)
    pos_margin_neg_score = sum(1 for m, d in margin_score_pairs if m > 0 and d < 0)

    lines = [
        "# MCU-SlimInfer Tiny Quality Smoke",
        "",
        f"- Output dir: `{output_dir}`",
        f"- Decision: **{decision.get('decision')}**",
        "",
        "## Sample IDs (12)",
    ]
    for sid in sample_ids:
        lines.append(f"- `{sid}`")
    lines.extend([
        "",
        "## Task Scores",
        f"- HotpotQA original / MCU / delta: {hotpot_orig:.4f} / {hotpot_mcu:.4f} / {hotpot_mcu - hotpot_orig:+.4f}",
        f"- MuSiQue original / MCU / delta: {musique_orig:.4f} / {musique_mcu:.4f} / {musique_mcu - musique_orig:+.4f}",
        f"- 2-task macro original / MCU / delta: {macro_orig:.4f} / {macro_mcu:.4f} / {macro_mcu - macro_orig:+.4f}",
        "",
        "## Runtime Audit",
        f"- Selection changed count: {audit_summary.get('selection_changed_count', 0)}",
        f"- Budget mismatch count: {audit_summary.get('budget_mismatch_count', 0)}",
        f"- Fallback count: {audit_summary.get('fallback_count', 0)}",
        f"- Mean probe runtime (ms): {audit_summary.get('mean_probe_runtime_ms', 0.0):.0f}",
        "",
        "## Utility Margin vs Score Delta",
        f"- Positive margin & positive score delta: {pos_margin_pos_score}",
        f"- Positive margin & negative score delta: {pos_margin_neg_score}",
        "",
        build_case_study(per_sample),
        "",
        "## Held-out Diagnostic",
        f"- Worth held-out diagnostic: **{decision.get('worth_heldout_diagnostic', False)}**",
        "",
        "## Decision Reasons",
    ])
    for reason in decision.get("reasons", []):
        lines.append(f"- {reason}")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MCU-SlimInfer tiny quality smoke.")
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--longbench_dir", type=Path, default=DEFAULT_LONGBENCH_DIR)
    parser.add_argument("--audit_dir", type=Path, default=DEFAULT_AUDIT_DIR)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--skip_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample_spec, expected_indices = build_sample_spec()
    audit_dir = (REPO_ROOT / args.audit_dir).resolve()
    audit_csv = audit_dir / "mcu_counterfactual_audit.csv"
    if not audit_csv.exists():
        raise RuntimeError(f"MCU audit CSV missing: {audit_csv}")

    frozen = load_frozen_selections_from_audit_csv(audit_csv)
    sample_ids = sample_ids_from_frozen(frozen)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (REPO_ROOT / f"results/mcu_sliminfer_tiny_quality_smoke_{timestamp}")
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    frozen_path = output_dir / "mcu_frozen_selections.jsonl"
    write_frozen_selections_jsonl(frozen_path, frozen)

    run_config = {
        "methods": ALL_METHODS,
        "tasks": TASKS,
        "sample_spec": sample_spec,
        "expected_indices": expected_indices,
        "sample_ids": sample_ids,
        "audit_dir": str(audit_dir),
        "frozen_selection_path": str(frozen_path),
        "model_path": str(args.model_path),
        "longbench_dir": str(args.longbench_dir),
    }
    write_json(output_dir / "run_config.json", run_config)

    run_dirs: Dict[str, Path] = {}
    predictions: Dict[str, Dict[Key, JsonRow]] = {}
    usage: Dict[str, Dict[Key, JsonRow]] = {}

    if args.skip_run:
        for method in ALL_METHODS:
            run_dir = output_dir / f"run_{method}"
            if not run_dir.exists():
                raise RuntimeError(f"skip_run requested but missing {run_dir}")
            run_dirs[method] = run_dir
            predictions[method] = read_predictions(run_dir, method, expected_indices)
            usage[method] = read_final_prefill_usage(run_dir, expected_indices)
    else:
        for method in ALL_METHODS:
            frozen_arg = frozen_path if method == METHOD_MCU else None
            run_dirs[method] = run_quality_method(args, output_dir, method, sample_spec, frozen_arg)
            predictions[method] = read_predictions(run_dirs[method], method, expected_indices)
            usage[method] = read_final_prefill_usage(run_dirs[method], expected_indices)

    original_config = json.loads((run_dirs[METHOD_ORIGINAL] / "config.json").read_text(encoding="utf-8"))
    original_pollution = 0
    if (original_config.get("sliminfer_config") or {}).get("variant_method") not in (METHOD_ORIGINAL, None, ""):
        original_pollution += 1
    mcu_config = json.loads((run_dirs[METHOD_MCU] / "config.json").read_text(encoding="utf-8"))
    if (mcu_config.get("sliminfer_config") or {}).get("variant_method") != METHOD_MCU:
        original_pollution += 1

    mcu_audit = read_mcu_runtime_audit(run_dirs[METHOD_MCU], expected_indices, frozen)
    method_scores = compute_method_summary(predictions, expected_indices)
    per_sample = build_per_sample_quality(predictions, usage, mcu_audit, frozen, expected_indices)
    audit_summary = summarize_mcu_audit(per_sample)
    decision = decide(method_scores, per_sample, audit_summary, original_pollution)

    quality_fields = [
        "task", "sample_index", "sample_id", "method", "prediction", "score",
        "selected_token_count", "budget", "budget_mismatch", "fallback_used", "fallback_reason",
        "selection_changed", "selected_jaccard_vs_original",
        "original_selected_block_ids", "mcu_selected_block_ids",
        "removed_block_ids", "added_block_ids",
        "removed_block_utility", "added_block_utility", "utility_margin",
        "hidden_change_mean", "logit_kl_mean", "entropy_change_mean", "top_logit_margin_change_mean",
        "probe_runtime_ms",
        "original_query_overlap", "mcu_query_overlap", "query_overlap_delta",
        "original_redundancy", "mcu_redundancy", "redundancy_delta",
        "original_score", "mcu_score", "score_delta",
    ]
    write_csv(output_dir / "per_sample_quality.csv", per_sample, quality_fields)

    summary_rows = []
    for task in TASKS + ["macro"]:
        summary_rows.append({
            "task": task,
            "original": method_scores[METHOD_ORIGINAL][task],
            "mcu": method_scores[METHOD_MCU][task],
            "delta": method_scores[METHOD_MCU][task] - method_scores[METHOD_ORIGINAL][task],
        })
    write_csv(output_dir / "mcu_quality_summary.csv", summary_rows, ["task", "original", "mcu", "delta"])

    runtime_rows = []
    for row in per_sample:
        runtime_rows.append({
            "task": row["task"],
            "sample_index": row["sample_index"],
            "sample_id": row["sample_id"],
            "fallback_used": row.get("fallback_used"),
            "fallback_reason": row.get("fallback_reason"),
            "budget_mismatch": row.get("budget_mismatch"),
            "selection_changed": row.get("selection_changed"),
            "selected_jaccard_vs_original": row.get("selected_jaccard_vs_original"),
            "utility_margin": row.get("utility_margin"),
            "probe_runtime_ms": row.get("probe_runtime_ms"),
            "score_delta": row.get("score_delta"),
            "removed_block_ids": row.get("removed_block_ids"),
            "added_block_ids": row.get("added_block_ids"),
        })
    write_csv(
        output_dir / "mcu_runtime_audit.csv",
        runtime_rows,
        list(runtime_rows[0].keys()) if runtime_rows else [
            "task", "sample_index", "sample_id", "fallback_used", "fallback_reason",
            "budget_mismatch", "selection_changed", "utility_margin", "score_delta",
        ],
    )

    report = build_report(output_dir, sample_ids, method_scores, per_sample, audit_summary, decision)
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    (output_dir / "decision.md").write_text(
        f"# Decision: {decision.get('decision')}\n\n"
        + "\n".join(f"- {r}" for r in decision.get("reasons", []))
        + f"\n\n- Worth held-out diagnostic: {decision.get('worth_heldout_diagnostic', False)}\n",
        encoding="utf-8",
    )
    write_json(output_dir / "decision.json", decision)

    print(report)
    print(f"Wrote results to {output_dir}")


if __name__ == "__main__":
    main()
