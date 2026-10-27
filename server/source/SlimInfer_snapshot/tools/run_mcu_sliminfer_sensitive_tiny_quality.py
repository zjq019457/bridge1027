#!/usr/bin/env python3
"""MCU-SlimInfer sensitivity-aware tiny quality smoke on 20 selected samples."""
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
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

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
METHOD_ORIGINAL = "sliminfer_original"
METHOD_MCU = "mcu_sliminfer"
ALL_METHODS = [METHOD_ORIGINAL, METHOD_MCU]
TASKS = ["hotpotqa", "musique"]
PRUNE_LAYERS = [9, 18, 26]
RUNTIME_KV_LAYER = 27
TASK_DROP_TOLERANCE = 5.0

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


def sha_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def load_selection(sensitivity_dir: Path) -> Tuple[Dict[str, List[int]], Dict[str, Any], List[str]]:
    selection_path = sensitivity_dir / "mcu_sensitive_sample_selection.json"
    if not selection_path.exists():
        raise RuntimeError(f"missing selection file: {selection_path}")
    payload = json.loads(selection_path.read_text(encoding="utf-8"))
    expected_indices = payload.get("expected_indices") or {}
    for task in TASKS:
        if len(expected_indices.get(task, [])) < 10:
            raise RuntimeError(f"selection missing 10 samples for {task}")
    sample_ids: List[str] = []
    for task in TASKS:
        sample_ids.extend(str(x) for x in (payload.get("selected_sample_ids", {}).get(task, [])))
    return {task: [int(x) for x in expected_indices[task]] for task in TASKS}, payload, sample_ids


def build_sample_spec(expected_indices: Mapping[str, Sequence[int]]) -> str:
    return ";".join(
        f"{task}:" + ",".join(str(i) for i in expected_indices[task])
        for task in TASKS
    )


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
        "--longbench_subset_name", "mcu_sliminfer_sensitive_tiny_quality",
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
    audit_dir.mkdir(parents=True, exist_ok=True)
    cmd = base_runner_cmd(args, run_dir, sample_spec) + [
        "--eval", "longbench",
        "--method", method,
        "--debug_generation_jsonl", str(audit_dir / "generations.jsonl"),
        "--debug_used_block_audit_path", str(audit_dir / "used_blocks.jsonl"),
        "--debug_used_block_include_full", "true",
        "--debug_used_block_full_layers", str(RUNTIME_KV_LAYER),
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


def read_final_prefill_usage(run_dir: Path, expected_indices: Dict[str, List[int]]) -> Dict[Key, JsonRow]:
    rows = read_jsonl(run_dir / "audit" / "used_blocks.jsonl")
    selected: Dict[Key, JsonRow] = {}
    for row in rows:
        if str(row.get("phase", "")) != "prefill":
            continue
        if int(row.get("layer_id", -1)) != RUNTIME_KV_LAYER:
            continue
        key = (str(row["task"]), int(row["sample_index"]))
        if key[0] and key[1] >= 0 and key not in selected:
            selected[key] = row
    missing = [(task, idx) for task in TASKS for idx in expected_indices[task] if (task, idx) not in selected]
    if missing:
        raise RuntimeError(f"missing final prefill used-block rows: {missing[:10]}")
    return selected


def verify_runtime_wiring(
    usage_orig: Mapping[Key, JsonRow],
    usage_mcu: Mapping[Key, JsonRow],
    expected_indices: Dict[str, List[int]],
) -> Tuple[int, List[str]]:
    issues: List[str] = []
    changed = 0
    for task in TASKS:
        for sample_index in expected_indices[task]:
            key = (task, sample_index)
            o_row = usage_orig[key]
            m_row = usage_mcu[key]
            o_tokens = o_row.get("used_tokens", o_row.get("kv_abs_indices_first100", []))
            m_tokens = m_row.get("used_tokens", m_row.get("kv_abs_indices_first100", []))
            if sha_json(o_tokens) != sha_json(m_tokens):
                changed += 1
            else:
                issues.append(f"runtime_kv_unchanged:{task}:{sample_index}")
    return changed, issues


def read_mcu_runtime_audit(
    run_dir: Path,
    expected_indices: Dict[str, List[int]],
    frozen: Mapping[str, Mapping[str, Any]],
) -> Dict[Key, JsonRow]:
    path = run_dir / "audit" / "mcu_runtime_selection_audit.jsonl"
    rows = read_jsonl(path, missing_ok=True)
    by_key: Dict[Key, JsonRow] = {}
    for row in rows:
        if int(row.get("layer_id", -1)) != PRUNE_LAYERS[-1]:
            continue
        if str(row.get("phase", "")) != "prefill":
            continue
        key = (str(row["task"]), int(row["sample_index"]))
        if key[0] and key[1] >= 0 and key not in by_key:
            by_key[key] = row
    for task in TASKS:
        for sample_index in expected_indices[task]:
            key = (task, sample_index)
            if key in by_key:
                continue
            frozen_row = frozen.get(f"{task}:{sample_index}", {})
            by_key[key] = {
                "task": task,
                "sample_index": sample_index,
                "sample_id": frozen_row.get("sample_id", sample_index),
                "fallback_used": bool(frozen_row.get("fallback_used")),
                "fallback_reason": frozen_row.get("fallback_reason", "missing_runtime_mcu_audit_row"),
                "selection_changed": bool(frozen_row.get("selection_changed")),
                "selected_jaccard_vs_original": frozen_row.get("selected_jaccard_vs_original", 1.0),
                "utility_margin": frozen_row.get("utility_margin", 0.0),
                "budget_preserved": frozen_row.get("budget_preserved", True),
            }
    return by_key


def build_frozen_from_selection(
    sensitivity_dir: Path,
    expected_indices: Dict[str, List[int]],
) -> Dict[str, Dict[str, Any]]:
    frozen_path = sensitivity_dir / "mcu_sensitive_frozen_selections.jsonl"
    if frozen_path.exists():
        frozen: Dict[str, Dict[str, Any]] = {}
        for row in read_jsonl(frozen_path):
            task = str(row.get("task", ""))
            sample_index = int(row.get("sample_index", -1))
            if task and sample_index >= 0:
                frozen[f"{task}:{sample_index}"] = row
        return frozen

    audit_csv = sensitivity_dir / "mcu_sensitivity_heldout_audit.csv"
    if audit_csv.exists():
        all_frozen = load_frozen_selections_from_audit_csv(audit_csv)
        return {
            key: all_frozen[key]
            for task in TASKS
            for idx in expected_indices[task]
            if (key := f"{task}:{idx}") in all_frozen
        }
    raise RuntimeError(f"no frozen selections in {sensitivity_dir}")


def build_per_sample_quality(
    predictions: Dict[str, Dict[Key, JsonRow]],
    usage: Dict[str, Dict[Key, JsonRow]],
    mcu_audit: Dict[Key, JsonRow],
    frozen: Mapping[str, Mapping[str, Any]],
    sensitivity_rows: Mapping[str, Mapping[str, Any]],
    expected_indices: Dict[str, List[int]],
) -> List[JsonRow]:
    rows: List[JsonRow] = []
    for task in TASKS:
        for sample_index in expected_indices[task]:
            key = (task, sample_index)
            orig = predictions[METHOD_ORIGINAL][key]
            mcu = predictions[METHOD_MCU][key]
            used_mcu = usage[METHOD_MCU].get(key, {})
            audit = mcu_audit.get(key, {})
            frozen_row = frozen.get(f"{task}:{sample_index}", {})
            sens = sensitivity_rows.get(f"{task}:{sample_index}", {})
            score_delta = float(mcu["score"]) - float(orig["score"])
            rows.append({
                "task": task,
                "sample_index": sample_index,
                "sample_id": orig.get("sample_id", sample_index),
                "method": METHOD_MCU,
                "prediction": mcu["prediction"],
                "score": mcu["score"],
                "original_prediction": orig["prediction"],
                "mcu_prediction": mcu["prediction"],
                "prediction_changed": orig["prediction"] != mcu["prediction"],
                "original_score": orig["score"],
                "mcu_score": mcu["score"],
                "score_delta": score_delta,
                "selected_token_count": int(used_mcu.get("final_block_count_used_by_attention", 0) or 0),
                "budget_mismatch": not bool(audit.get("budget_preserved", frozen_row.get("budget_preserved", True))),
                "fallback_used": bool(audit.get("fallback_used", frozen_row.get("fallback_used"))),
                "selection_changed": bool(audit.get("selection_changed", frozen_row.get("selection_changed"))),
                "selected_jaccard_vs_original": audit.get(
                    "selected_jaccard_vs_original", frozen_row.get("selected_jaccard_vs_original", 1.0),
                ),
                "utility_margin": audit.get("utility_margin", frozen_row.get("utility_margin", 0.0)),
                "hidden_change_mean": sens.get("hidden_change_mean", frozen_row.get("hidden_change_mean", 0.0)),
                "logit_kl_mean": sens.get("logit_kl_mean", frozen_row.get("logit_kl_mean", 0.0)),
                "top_logit_margin_change_mean": sens.get(
                    "top_logit_margin_change_mean", frozen_row.get("top_logit_margin_change_mean", 0.0),
                ),
                "mcu_sensitivity_score": sens.get("mcu_sensitivity_score", frozen_row.get("mcu_sensitivity_score", 0.0)),
            })
    return rows


def compute_method_summary(predictions: Dict[str, Dict[Key, JsonRow]], expected_indices: Dict[str, List[int]]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for method, pred_map in predictions.items():
        task_scores: Dict[str, float] = {}
        for task in TASKS:
            task_scores[task] = mean(float(pred_map[(task, idx)]["score"]) for idx in expected_indices[task])
        task_scores["macro"] = mean(task_scores[t] for t in TASKS)
        out[method] = task_scores
    return out


def decide(
    method_scores: Dict[str, Dict[str, float]],
    per_sample: Sequence[JsonRow],
    audit_summary: Mapping[str, Any],
    wiring_issues: Sequence[str],
    original_pollution_count: int,
) -> Dict[str, Any]:
    bug_reasons = list(wiring_issues)
    no_go_reasons: List[str] = []
    weak_reasons: List[str] = []

    if audit_summary.get("budget_mismatch_count", 0) > 0:
        bug_reasons.append(f"budget_mismatch_count={audit_summary['budget_mismatch_count']}")
    if original_pollution_count:
        bug_reasons.append(f"original_selector_pollution={original_pollution_count}")
    if audit_summary.get("runtime_kv_changed_count", 0) < len(per_sample):
        bug_reasons.append(
            f"runtime_kv_unchanged={len(per_sample) - int(audit_summary.get('runtime_kv_changed_count', 0))}"
        )
    empty_preds = [r for r in per_sample if not str(r.get("prediction", "")).strip()]
    if empty_preds:
        bug_reasons.append(f"empty_predictions={len(empty_preds)}")
    if bug_reasons:
        return {"decision": "BUG_STOP_FIX_REQUIRED", "reasons": bug_reasons}

    orig_macro = float(method_scores[METHOD_ORIGINAL]["macro"])
    mcu_macro = float(method_scores[METHOD_MCU]["macro"])
    macro_delta = mcu_macro - orig_macro
    hotpot_delta = float(method_scores[METHOD_MCU]["hotpotqa"]) - float(method_scores[METHOD_ORIGINAL]["hotpotqa"])
    musique_delta = float(method_scores[METHOD_MCU]["musique"]) - float(method_scores[METHOD_ORIGINAL]["musique"])
    prediction_changed_count = sum(1 for r in per_sample if r.get("prediction_changed"))
    nonzero_score_delta_count = sum(1 for r in per_sample if abs(float(r.get("score_delta", 0.0) or 0.0)) > 1e-9)

    if audit_summary.get("fallback_count", 0) > len(per_sample) // 2:
        return {
            "decision": "BUG_STOP_FIX_REQUIRED",
            "reasons": [f"fallback_count={audit_summary['fallback_count']}"],
        }

    if prediction_changed_count == 0:
        no_go_reasons.append("prediction_unchanged_on_sensitive_set")
    if mcu_macro <= orig_macro:
        no_go_reasons.append("mcu_macro_not_above_original")
    if hotpot_delta <= -TASK_DROP_TOLERANCE:
        no_go_reasons.append(f"hotpotqa_task_drop={hotpot_delta:.2f}")
    if musique_delta <= -TASK_DROP_TOLERANCE:
        no_go_reasons.append(f"musique_task_drop={musique_delta:.2f}")

    go_ready = (
        prediction_changed_count > 0
        and mcu_macro > orig_macro
        and hotpot_delta > -TASK_DROP_TOLERANCE
        and musique_delta > -TASK_DROP_TOLERANCE
        and audit_summary.get("budget_mismatch_count", 0) == 0
        and audit_summary.get("fallback_count", 0) <= max(2, len(per_sample) // 4)
    )
    if go_ready and not no_go_reasons:
        return {
            "decision": "MCU_SENSITIVE_TINY_GO",
            "reasons": [
                f"macro_delta={macro_delta:+.4f}",
                f"prediction_changed={prediction_changed_count}",
                f"nonzero_score_delta={nonzero_score_delta_count}",
            ],
            "worth_heldout_diagnostic": True,
        }

    if prediction_changed_count > 0 and (mcu_macro > orig_macro or hotpot_delta > 0 or musique_delta > 0):
        weak_reasons.append("prediction_changed_with_partial_positive_signal")
        return {
            "decision": "MCU_SENSITIVE_TINY_WEAK_GO",
            "reasons": weak_reasons + [f"macro_delta={macro_delta:+.4f}"],
            "worth_heldout_diagnostic": True,
        }

    return {
        "decision": "MCU_SENSITIVE_TINY_NO_GO",
        "reasons": no_go_reasons or [f"macro_delta={macro_delta:+.4f}"],
        "worth_heldout_diagnostic": prediction_changed_count > 0,
    }


def build_report(
    output_dir: Path,
    selection_payload: Mapping[str, Any],
    sample_ids: Sequence[str],
    method_scores: Dict[str, Dict[str, float]],
    per_sample: Sequence[JsonRow],
    audit_summary: Mapping[str, Any],
    decision: Mapping[str, Any],
) -> str:
    hotpot_orig = method_scores[METHOD_ORIGINAL]["hotpotqa"]
    hotpot_mcu = method_scores[METHOD_MCU]["hotpotqa"]
    musique_orig = method_scores[METHOD_ORIGINAL]["musique"]
    musique_mcu = method_scores[METHOD_MCU]["musique"]
    macro_orig = method_scores[METHOD_ORIGINAL]["macro"]
    macro_mcu = method_scores[METHOD_MCU]["macro"]
    prediction_changed_count = sum(1 for r in per_sample if r.get("prediction_changed"))
    nonzero_score_delta_count = sum(1 for r in per_sample if abs(float(r.get("score_delta", 0.0) or 0.0)) > 1e-9)

    pairs = [
        (float(r.get("mcu_sensitivity_score", 0.0) or 0.0), float(r.get("score_delta", 0.0) or 0.0))
        for r in per_sample
    ]
    pos_sens_pos_delta = sum(1 for s, d in pairs if s > 0 and d > 0)
    pos_sens_neg_delta = sum(1 for s, d in pairs if s > 0 and d < 0)

    lines = [
        "# MCU-SlimInfer Sensitive Tiny Quality Smoke",
        "",
        f"- Output dir: `{output_dir}`",
        f"- Decision: **{decision.get('decision')}**",
        "",
        "## Sample selection rule",
        "- Selected from held-out sensitivity audit by `mcu_sensitivity_score`.",
        "- HotpotQA top 10 + MuSiQue top 10.",
        "- **No gold answer, no score, no prediction, no task-label gate.**",
        "",
        f"- Formula: `{selection_payload.get('sensitivity_formula', '')}`",
        "",
        "## 20 sample ids",
    ]
    for sid in sample_ids:
        lines.append(f"- `{sid}`")
    lines.extend([
        "",
        "## Task Scores",
        f"- HotpotQA original / MCU / delta: {hotpot_orig:.4f} / {hotpot_mcu:.4f} / {hotpot_mcu - hotpot_orig:+.4f}",
        f"- MuSiQue original / MCU / delta: {musique_orig:.4f} / {musique_mcu:.4f} / {musique_mcu - musique_orig:+.4f}",
        f"- macro original / MCU / delta: {macro_orig:.4f} / {macro_mcu:.4f} / {macro_mcu - macro_orig:+.4f}",
        "",
        "## Runtime / quality signals",
        f"- prediction_changed count: {prediction_changed_count}",
        f"- score_delta non-zero count: {nonzero_score_delta_count}",
        f"- runtime_kv_changed count: {audit_summary.get('runtime_kv_changed_count', 0)}",
        f"- budget mismatch count: {audit_summary.get('budget_mismatch_count', 0)}",
        f"- fallback count: {audit_summary.get('fallback_count', 0)}",
        "",
        "## Sensitivity score vs score_delta",
        f"- positive sensitivity & positive score delta: {pos_sens_pos_delta}",
        f"- positive sensitivity & negative score delta: {pos_sens_neg_delta}",
        "",
        "## Held-out diagnostic",
        f"- Worth held-out diagnostic: **{decision.get('worth_heldout_diagnostic', False)}**",
        "",
        "## Decision reasons",
    ])
    for reason in decision.get("reasons", []):
        lines.append(f"- {reason}")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MCU sensitive tiny quality smoke.")
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--longbench_dir", type=Path, default=DEFAULT_LONGBENCH_DIR)
    parser.add_argument("--sensitivity_audit_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--skip_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sensitivity_dir = (REPO_ROOT / args.sensitivity_audit_dir).resolve()
    expected_indices, selection_payload, sample_ids = load_selection(sensitivity_dir)
    sample_spec = build_sample_spec(expected_indices)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (REPO_ROOT / f"results/mcu_sliminfer_sensitive_tiny_quality_{timestamp}")
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    frozen = build_frozen_from_selection(sensitivity_dir, expected_indices)
    frozen_path = output_dir / "mcu_frozen_selections.jsonl"
    write_frozen_selections_jsonl(frozen_path, frozen)

    sensitivity_rows = {
        f"{task}:{int(row['sample_index'])}": row
        for task in TASKS
        for row in selection_payload.get("selected_rows", {}).get(task, [])
    }

    write_json(output_dir / "run_config.json", {
        "sensitivity_audit_dir": str(sensitivity_dir),
        "sample_spec": sample_spec,
        "expected_indices": expected_indices,
        "sample_ids": sample_ids,
        "selection_rule": selection_payload.get("selection_rule", ""),
    })

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

    runtime_kv_changed, wiring_issues = verify_runtime_wiring(
        usage[METHOD_ORIGINAL], usage[METHOD_MCU], expected_indices,
    )

    original_pollution = 0
    orig_cfg = json.loads((run_dirs[METHOD_ORIGINAL] / "config.json").read_text(encoding="utf-8"))
    mcu_cfg = json.loads((run_dirs[METHOD_MCU] / "config.json").read_text(encoding="utf-8"))
    if (orig_cfg.get("sliminfer_config") or {}).get("variant_method") not in (METHOD_ORIGINAL, None, ""):
        original_pollution += 1
    if (mcu_cfg.get("sliminfer_config") or {}).get("variant_method") != METHOD_MCU:
        original_pollution += 1

    mcu_audit = read_mcu_runtime_audit(run_dirs[METHOD_MCU], expected_indices, frozen)
    method_scores = compute_method_summary(predictions, expected_indices)
    per_sample = build_per_sample_quality(
        predictions, usage, mcu_audit, frozen, sensitivity_rows, expected_indices,
    )
    audit_summary = {
        "budget_mismatch_count": sum(1 for r in per_sample if r.get("budget_mismatch")),
        "fallback_count": sum(1 for r in per_sample if r.get("fallback_used")),
        "selection_changed_count": sum(1 for r in per_sample if r.get("selection_changed")),
        "runtime_kv_changed_count": runtime_kv_changed,
        "prediction_changed_count": sum(1 for r in per_sample if r.get("prediction_changed")),
        "nonzero_score_delta_count": sum(1 for r in per_sample if abs(float(r.get("score_delta", 0.0) or 0.0)) > 1e-9),
    }
    decision = decide(method_scores, per_sample, audit_summary, wiring_issues, original_pollution)

    quality_fields = [
        "task", "sample_index", "sample_id", "method", "prediction", "score",
        "original_prediction", "mcu_prediction", "prediction_changed",
        "original_score", "mcu_score", "score_delta",
        "selected_token_count", "budget_mismatch", "fallback_used",
        "selection_changed", "selected_jaccard_vs_original",
        "utility_margin", "hidden_change_mean", "logit_kl_mean",
        "top_logit_margin_change_mean", "mcu_sensitivity_score",
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

    report = build_report(output_dir, selection_payload, sample_ids, method_scores, per_sample, audit_summary, decision)
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
