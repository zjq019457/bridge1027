#!/usr/bin/env python3
"""Run the fixed 20-sample QDC-SlimInfer tiny quality smoke.

Compares only:
  1. sliminfer_original
  2. qdc_sliminfer

Sample selection comes from the prior QDC selection audit CSV:
  10 qdc_triggered=true + 10 qdc_triggered=false controls,
  balanced across HotpotQA and MuSiQue when possible.
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
DEFAULT_SELECTION_AUDIT_DIR = Path("results/qdc_sliminfer_selection_audit_20260708_191113")
METHOD_ORIGINAL = "sliminfer_original"
METHOD_QDC = "qdc_sliminfer"
ALL_METHODS = [METHOD_ORIGINAL, METHOD_QDC]
TASKS = ["hotpotqa", "musique"]
PRUNE_LAYERS = [9, 18, 26]
FINAL_BUDGET_TOKENS = 4096
CONTROL_DROP_TOLERANCE = 2.0

JsonRow = Dict[str, Any]
Key = Tuple[str, int]


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


def check_gpu(min_free_gpu_gb: float) -> Dict[str, Any]:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free", "--format=csv,noheader,nounits"],
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip().splitlines()
    rows = []
    for line in output:
        parts = [p.strip() for p in line.split(",")]
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
    selected = max(rows, key=lambda r: r["memory_free_gib"])
    if float(selected["memory_free_gib"]) < float(min_free_gpu_gb):
        raise RuntimeError(
            f"free GPU {selected['memory_free_gib']:.2f} GiB < required {float(min_free_gpu_gb):.2f} GiB"
        )
    return {"rows": rows, "selected": selected, "min_free_gpu_gb": float(min_free_gpu_gb)}


def select_tiny_samples(audit_csv: Path, per_group: int = 5) -> Tuple[List[JsonRow], Dict[str, Any]]:
    if not audit_csv.exists():
        raise RuntimeError(f"selection audit missing: {audit_csv}")
    rows = list(csv.DictReader(audit_csv.open(encoding="utf-8")))
    trigger_rows = [r for r in rows if str(r.get("qdc_triggered", "")).lower() == "true"]
    control_rows = [r for r in rows if str(r.get("qdc_triggered", "")).lower() == "false"]

    def pick(pool: List[JsonRow], task: str, count: int, label: str) -> List[JsonRow]:
        task_rows = sorted(
            [r for r in pool if str(r.get("task", "")) == task],
            key=lambda r: int(r.get("sample_index", -1)),
        )
        if len(task_rows) < count:
            raise RuntimeError(
                f"insufficient {label} samples for {task}: need {count}, have {len(task_rows)}"
            )
        return task_rows[:count]

    selected: List[JsonRow] = []
    selection_meta: Dict[str, Any] = {
        "audit_csv": str(audit_csv),
        "per_task_per_group": per_group,
        "trigger_available": {task: sum(1 for r in trigger_rows if r["task"] == task) for task in TASKS},
        "control_available": {task: sum(1 for r in control_rows if r["task"] == task) for task in TASKS},
        "groups": {},
    }
    for task in TASKS:
        trig = pick(trigger_rows, task, per_group, "trigger")
        ctrl = pick(control_rows, task, per_group, "control")
        selected.extend(trig)
        selected.extend(ctrl)
        selection_meta["groups"][task] = {
            "trigger": [{"sample_index": int(r["sample_index"]), "sample_id": r.get("sample_id", "")} for r in trig],
            "control": [{"sample_index": int(r["sample_index"]), "sample_id": r.get("sample_id", "")} for r in ctrl],
        }

    selected = sorted(selected, key=lambda r: (str(r.get("task", "")), int(r.get("sample_index", -1))))
    selection_meta["selected_count"] = len(selected)
    selection_meta["trigger_count"] = sum(1 for r in selected if str(r.get("qdc_triggered", "")).lower() == "true")
    selection_meta["control_count"] = sum(1 for r in selected if str(r.get("qdc_triggered", "")).lower() == "false")
    selection_meta["sample_ids"] = [str(r.get("sample_id", "")) for r in selected]
    return selected, selection_meta


def build_sample_spec(selected_rows: Sequence[JsonRow]) -> Tuple[str, Dict[str, List[int]]]:
    indices: Dict[str, List[int]] = defaultdict(list)
    for row in selected_rows:
        indices[str(row["task"])].append(int(row["sample_index"]))
    for task in indices:
        indices[task] = sorted(indices[task])
    spec = ";".join(f"{task}:" + ",".join(str(i) for i in indices[task]) for task in TASKS if indices[task])
    return spec, dict(indices)


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
        "--longbench_subset_name", "qdc_sliminfer_tiny_quality_smoke",
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


def run_quality_method(args: argparse.Namespace, output_dir: Path, method: str, sample_spec: str) -> Path:
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
    if method == METHOD_QDC:
        qdc_audit = audit_dir / "qdc_runtime_selection_audit.jsonl"
        qdc_audit.write_text("", encoding="utf-8")
        cmd.extend(["--qdc_selection_audit_path", str(qdc_audit)])
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
                "gold": row.get("answers", []),
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


def read_qdc_runtime_audit(run_dir: Path, expected_indices: Dict[str, List[int]]) -> Dict[Key, JsonRow]:
    path = run_dir / "audit" / "qdc_runtime_selection_audit.jsonl"
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
            used = usage.get(key, {})
            prompt_block_count = int(used.get("prompt_block_count", 0) or 0)
            final_block_count = int(used.get("final_block_count_used_by_attention", 0) or 0)
            by_key[key] = {
                "task": task,
                "sample_index": sample_index,
                "sample_id": used.get("sample_id", sample_index),
                "qdc_triggered": False,
                "qdc_trigger_reason": (
                    "all_blocks_kept_no_gate_layer_prune"
                    if prompt_block_count > 0 and final_block_count >= prompt_block_count
                    else "missing_runtime_qdc_audit_row"
                ),
                "fallback_used": False,
                "budget_preserved": True,
                "original_component_coverage": None,
                "qdc_component_coverage": None,
                "coverage_delta": 0.0,
                "selected_jaccard_vs_original": 1.0,
                "removed_block_ids": [],
                "added_block_ids": [],
                "original_query_overlap": None,
                "qdc_query_overlap": None,
                "query_overlap_delta": 0.0,
                "original_redundancy": None,
                "qdc_redundancy": None,
                "redundancy_delta": 0.0,
            }
    return by_key


def load_offline_audit_map(audit_csv: Path) -> Dict[Key, JsonRow]:
    out: Dict[Key, JsonRow] = {}
    for row in csv.DictReader(audit_csv.open(encoding="utf-8")):
        key = (str(row.get("task", "")), int(row.get("sample_index", -1)))
        out[key] = row
    return out


def compute_method_summary(
    predictions: Dict[str, Dict[Key, JsonRow]],
    expected_indices: Dict[str, List[int]],
) -> Dict[str, Dict[str, float]]:
    method_scores: Dict[str, Dict[str, float]] = {}
    for method in ALL_METHODS:
        preds = predictions[method]
        task_means: Dict[str, float] = {}
        for task in TASKS:
            task_scores = [float(preds[(task, si)]["score"]) for si in expected_indices[task]]
            task_means[task] = mean(task_scores)
        task_means["macro"] = mean(task_means[t] for t in TASKS)
        method_scores[method] = task_means
    return method_scores


def subset_mean(
    predictions: Dict[Key, JsonRow],
    keys: Sequence[Key],
) -> float:
    return mean(float(predictions[k]["score"]) for k in keys)


def build_per_sample_quality(
    selected_rows: Sequence[JsonRow],
    predictions: Dict[str, Dict[Key, JsonRow]],
    usage: Dict[str, Dict[Key, JsonRow]],
    qdc_audit: Dict[Key, JsonRow],
    offline_audit: Dict[Key, JsonRow],
) -> List[JsonRow]:
    rows: List[JsonRow] = []
    for sample_row in selected_rows:
        task = str(sample_row["task"])
        sample_index = int(sample_row["sample_index"])
        key = (task, sample_index)
        offline = offline_audit.get(key, {})
        runtime_qdc = qdc_audit.get(key, {})
        orig = predictions[METHOD_ORIGINAL][key]
        qdc = predictions[METHOD_QDC][key]
        orig_usage = usage[METHOD_ORIGINAL].get(key, {})
        qdc_usage = usage[METHOD_QDC].get(key, {})
        offline_trigger = str(offline.get("qdc_triggered", "")).lower() == "true"

        for method, pred_row, used in (
            (METHOD_ORIGINAL, orig, orig_usage),
            (METHOD_QDC, qdc, qdc_usage),
        ):
            selected_token_count = int(used.get("kv_abs_indices_count", 0) or 0)
            if method == METHOD_QDC:
                audit = runtime_qdc
                qdc_triggered = bool(audit.get("qdc_triggered"))
                fallback = bool(audit.get("fallback_used"))
                original_component_coverage = audit.get("original_component_coverage", offline.get("original_component_coverage"))
                qdc_component_coverage = audit.get("qdc_component_coverage", offline.get("qdc_component_coverage"))
                coverage_delta = audit.get("coverage_delta", offline.get("coverage_delta"))
                selected_jaccard = audit.get("selected_jaccard_vs_original", offline.get("selected_jaccard_vs_original"))
                removed_block_ids = audit.get("removed_block_ids", offline.get("removed_block_ids", []))
                added_block_ids = audit.get("added_block_ids", offline.get("added_block_ids", []))
                original_query_overlap = audit.get("original_query_overlap", offline.get("original_query_overlap"))
                qdc_query_overlap = audit.get("qdc_query_overlap", offline.get("qdc_query_overlap"))
                query_overlap_delta = audit.get("query_overlap_delta", offline.get("query_overlap_delta"))
                original_redundancy = audit.get("original_redundancy", offline.get("original_redundancy"))
                qdc_redundancy = audit.get("qdc_redundancy", offline.get("qdc_redundancy"))
                redundancy_delta = audit.get("redundancy_delta", offline.get("redundancy_delta"))
                qdc_trigger_reason = str(audit.get("qdc_trigger_reason", offline.get("qdc_trigger_reason", "")))
            else:
                qdc_triggered = False
                fallback = False
                original_component_coverage = offline.get("original_component_coverage")
                qdc_component_coverage = offline.get("qdc_component_coverage")
                coverage_delta = offline.get("coverage_delta")
                selected_jaccard = offline.get("selected_jaccard_vs_original", 1.0)
                removed_block_ids = offline.get("removed_block_ids", [])
                added_block_ids = offline.get("added_block_ids", [])
                original_query_overlap = offline.get("original_query_overlap")
                qdc_query_overlap = offline.get("qdc_query_overlap")
                query_overlap_delta = offline.get("query_overlap_delta")
                original_redundancy = offline.get("original_redundancy")
                qdc_redundancy = offline.get("qdc_redundancy")
                redundancy_delta = offline.get("redundancy_delta")
                qdc_trigger_reason = ""

            rows.append({
                "task": task,
                "sample_index": sample_index,
                "sample_id": pred_row.get("sample_id", sample_row.get("sample_id", sample_index)),
                "method": method,
                "prediction": pred_row["prediction"],
                "score": pred_row["score"],
                "selected_token_count": selected_token_count,
                "budget": FINAL_BUDGET_TOKENS,
                "budget_mismatch": selected_token_count > FINAL_BUDGET_TOKENS,
                "fallback": fallback,
                "qdc_triggered": qdc_triggered if method == METHOD_QDC else offline_trigger,
                "qdc_trigger_reason": qdc_trigger_reason,
                "original_component_coverage": original_component_coverage,
                "qdc_component_coverage": qdc_component_coverage,
                "coverage_delta": coverage_delta,
                "selected_jaccard_vs_original": selected_jaccard,
                "removed_block_ids": removed_block_ids,
                "added_block_ids": added_block_ids,
                "original_query_overlap": original_query_overlap,
                "qdc_query_overlap": qdc_query_overlap,
                "query_overlap_delta": query_overlap_delta,
                "original_redundancy": original_redundancy,
                "qdc_redundancy": qdc_redundancy,
                "redundancy_delta": redundancy_delta,
                "offline_qdc_triggered": offline_trigger,
                "original_score": orig["score"],
                "qdc_score": qdc["score"],
                "qdc_delta_vs_original": float(qdc["score"]) - float(orig["score"]),
            })
    return rows


def summarize_qdc_audit(per_sample: Sequence[JsonRow]) -> Dict[str, Any]:
    qdc_rows = [r for r in per_sample if r["method"] == METHOD_QDC]
    changed = [
        r for r in qdc_rows
        if bool(r.get("qdc_triggered"))
        or (r.get("removed_block_ids") not in ("", "[]", None) and r.get("removed_block_ids"))
        or float(r.get("selected_jaccard_vs_original") or 1.0) < 0.999
    ]
    return {
        "fallback_count": sum(1 for r in qdc_rows if bool(r.get("fallback"))),
        "budget_mismatch_count": sum(1 for r in qdc_rows if bool(r.get("budget_mismatch"))),
        "runtime_trigger_count": sum(1 for r in qdc_rows if bool(r.get("qdc_triggered"))),
        "selection_changed_count": len(changed),
        "selection_changed_sample_ids": [r.get("sample_id") for r in changed],
    }


def decide(
    method_scores: Dict[str, Dict[str, float]],
    per_sample: Sequence[JsonRow],
    qdc_audit_summary: Dict[str, Any],
    original_pollution_count: int,
    expected_indices: Dict[str, List[int]],
) -> Dict[str, Any]:
    bug_reasons: List[str] = []
    no_go_reasons: List[str] = []
    weak_reasons: List[str] = []

    orig_macro = method_scores[METHOD_ORIGINAL]["macro"]
    qdc_macro = method_scores[METHOD_QDC]["macro"]
    macro_delta = qdc_macro - orig_macro

    qdc_rows = [r for r in per_sample if r["method"] == METHOD_QDC]
    trigger_keys = [
        (str(r["task"]), int(r["sample_index"]))
        for r in qdc_rows
        if bool(r.get("offline_qdc_triggered"))
    ]
    control_keys = [
        (str(r["task"]), int(r["sample_index"]))
        for r in qdc_rows
        if not bool(r.get("offline_qdc_triggered"))
    ]
    trigger_orig = subset_mean({(r["task"], int(r["sample_index"])): r for r in per_sample if r["method"] == METHOD_ORIGINAL}, trigger_keys)
    trigger_qdc = subset_mean({(r["task"], int(r["sample_index"])): r for r in qdc_rows}, trigger_keys)
    control_orig = subset_mean({(r["task"], int(r["sample_index"])): r for r in per_sample if r["method"] == METHOD_ORIGINAL}, control_keys)
    control_qdc = subset_mean({(r["task"], int(r["sample_index"])): r for r in qdc_rows}, control_keys)

    budget_mismatch_count = int(qdc_audit_summary.get("budget_mismatch_count", 0))
    fallback_count = int(qdc_audit_summary.get("fallback_count", 0))

    if budget_mismatch_count != 0:
        bug_reasons.append("budget_mismatch_nonzero")
    if fallback_count != 0:
        bug_reasons.append("fallback_nonzero")
    if original_pollution_count != 0:
        bug_reasons.append("original_selector_polluted")
    if len(qdc_rows) != sum(len(v) for v in expected_indices.values()):
        bug_reasons.append("qdc_row_count_mismatch")
    missing_runtime_audit = sum(
        1 for r in qdc_rows
        if str(r.get("qdc_trigger_reason", "")) == "missing_runtime_qdc_audit_row"
    )
    if missing_runtime_audit:
        bug_reasons.append(f"missing_runtime_qdc_audit_rows={missing_runtime_audit}")
    if bug_reasons:
        return {
            "decision": "BUG_STOP_FIX_REQUIRED",
            "reasons": bug_reasons,
            "macro_delta": macro_delta,
            "trigger_delta": trigger_qdc - trigger_orig,
            "control_delta": control_qdc - control_orig,
        }

    trigger_delta = trigger_qdc - trigger_orig
    control_delta = control_qdc - control_orig

    if qdc_macro <= orig_macro:
        no_go_reasons.append("qdc_macro_not_above_original")
    if trigger_delta <= 0:
        no_go_reasons.append("trigger_subset_no_positive_signal")
    if control_delta <= -CONTROL_DROP_TOLERANCE:
        no_go_reasons.append("control_subset_clear_drop")

    if no_go_reasons:
        return {
            "decision": "QDC_TINY_QUALITY_NO_GO",
            "reasons": no_go_reasons,
            "macro_delta": macro_delta,
            "trigger_delta": trigger_delta,
            "control_delta": control_delta,
        }

    go_ready = (
        qdc_macro > orig_macro
        and trigger_delta > 0
        and control_delta >= -CONTROL_DROP_TOLERANCE
    )
    if go_ready:
        return {
            "decision": "QDC_TINY_QUALITY_GO",
            "reasons": ["budget_clean", "macro_gain", "trigger_positive", "control_stable"],
            "macro_delta": macro_delta,
            "trigger_delta": trigger_delta,
            "control_delta": control_delta,
        }

    if qdc_macro > orig_macro or trigger_delta > 0:
        weak_reasons.append("macro_or_trigger_positive_but_not_both_strong")
    return {
        "decision": "QDC_TINY_QUALITY_WEAK_GO",
        "reasons": weak_reasons or ["overall_unstable"],
        "macro_delta": macro_delta,
        "trigger_delta": trigger_delta,
        "control_delta": control_delta,
    }


def build_report(
    output_dir: Path,
    selection_meta: Dict[str, Any],
    selected_rows: Sequence[JsonRow],
    method_scores: Dict[str, Dict[str, float]],
    per_sample: Sequence[JsonRow],
    qdc_audit_summary: Dict[str, Any],
    decision: Dict[str, Any],
) -> str:
    qdc_rows = [r for r in per_sample if r["method"] == METHOD_QDC]
    trigger_count = sum(1 for r in selected_rows if str(r.get("qdc_triggered", "")).lower() == "true")
    control_count = len(selected_rows) - trigger_count
    sample_ids = [str(r.get("sample_id", "")) for r in selected_rows]

    hotpot_orig = method_scores[METHOD_ORIGINAL]["hotpotqa"]
    hotpot_qdc = method_scores[METHOD_QDC]["hotpotqa"]
    musique_orig = method_scores[METHOD_ORIGINAL]["musique"]
    musique_qdc = method_scores[METHOD_QDC]["musique"]
    macro_orig = method_scores[METHOD_ORIGINAL]["macro"]
    macro_qdc = method_scores[METHOD_QDC]["macro"]

    trigger_keys = {(r["task"], int(r["sample_index"])) for r in qdc_rows if bool(r.get("offline_qdc_triggered"))}
    control_keys = {(r["task"], int(r["sample_index"])) for r in qdc_rows if not bool(r.get("offline_qdc_triggered"))}
    trigger_orig = subset_mean({(r["task"], int(r["sample_index"])): r for r in per_sample if r["method"] == METHOD_ORIGINAL}, trigger_keys)
    trigger_qdc = subset_mean({(r["task"], int(r["sample_index"])): r for r in qdc_rows}, trigger_keys)
    control_orig = subset_mean({(r["task"], int(r["sample_index"])): r for r in per_sample if r["method"] == METHOD_ORIGINAL}, control_keys)
    control_qdc = subset_mean({(r["task"], int(r["sample_index"])): r for r in qdc_rows}, control_keys)

    coverage_score_pairs = []
    for row in qdc_rows:
        try:
            coverage_score_pairs.append(
                (float(row.get("coverage_delta") or 0.0), float(row.get("qdc_delta_vs_original") or 0.0))
            )
        except (TypeError, ValueError):
            continue
    positive_cov_positive_score = sum(1 for cov, sc in coverage_score_pairs if cov > 0 and sc > 0)
    positive_cov_negative_score = sum(1 for cov, sc in coverage_score_pairs if cov > 0 and sc < 0)
    zero_cov_score_gain = sum(1 for cov, sc in coverage_score_pairs if abs(cov) <= 1e-9 and sc > 0)

    lines = [
        "# QDC-SlimInfer Tiny Quality Smoke",
        "",
        f"- Output dir: `{output_dir}`",
        f"- Decision: **{decision.get('decision')}**",
        "",
        "## Sample Set",
        f"- Sample count: {len(selected_rows)}",
        f"- Trigger count: {trigger_count}",
        f"- Control count: {control_count}",
        "- Sample ids:",
    ]
    for sid in sample_ids:
        lines.append(f"  - `{sid}`")

    lines.extend([
        "",
        "## Task Scores",
        f"- HotpotQA original / qdc / delta: {hotpot_orig:.4f} / {hotpot_qdc:.4f} / {hotpot_qdc - hotpot_orig:+.4f}",
        f"- MuSiQue original / qdc / delta: {musique_orig:.4f} / {musique_qdc:.4f} / {musique_qdc - musique_orig:+.4f}",
        f"- 2-task macro original / qdc / delta: {macro_orig:.4f} / {macro_qdc:.4f} / {macro_qdc - macro_orig:+.4f}",
        "",
        "## Subsets",
        f"- Trigger subset original / qdc / delta: {trigger_orig:.4f} / {trigger_qdc:.4f} / {trigger_qdc - trigger_orig:+.4f}",
        f"- Control subset original / qdc / delta: {control_orig:.4f} / {control_qdc:.4f} / {control_qdc - control_orig:+.4f}",
        "",
        "## Audit",
        f"- Budget mismatch count: {qdc_audit_summary.get('budget_mismatch_count', 0)}",
        f"- Fallback count: {qdc_audit_summary.get('fallback_count', 0)}",
        f"- Runtime QDC trigger count: {qdc_audit_summary.get('runtime_trigger_count', 0)}",
        f"- Selection changed count: {qdc_audit_summary.get('selection_changed_count', 0)}",
        f"- Selection changed sample ids: {qdc_audit_summary.get('selection_changed_sample_ids', [])}",
        "",
        "## Coverage vs Score",
        f"- Positive coverage delta & positive score delta: {positive_cov_positive_score}",
        f"- Positive coverage delta & negative score delta: {positive_cov_negative_score}",
        f"- Zero coverage delta & positive score delta: {zero_cov_score_gain}",
        "",
        "## Decision Reasons",
    ])
    for reason in decision.get("reasons", []):
        lines.append(f"- {reason}")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run QDC-SlimInfer tiny quality smoke.")
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--longbench_dir", type=Path, default=DEFAULT_LONGBENCH_DIR)
    parser.add_argument("--selection_audit_dir", type=Path, default=DEFAULT_SELECTION_AUDIT_DIR)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--min_free_gpu_gb", type=float, default=8.0)
    parser.add_argument("--skip_run", action="store_true", help="Only rebuild reports from existing run dirs in output_dir.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit_csv = (REPO_ROOT / args.selection_audit_dir / "qdc_selection_audit.csv").resolve()
    selected_rows, selection_meta = select_tiny_samples(audit_csv)
    sample_spec, expected_indices = build_sample_spec(selected_rows)
    offline_audit = load_offline_audit_map(audit_csv)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (REPO_ROOT / f"results/qdc_sliminfer_tiny_quality_smoke_{timestamp}")
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_config = {
        "methods": ALL_METHODS,
        "tasks": TASKS,
        "prune_layers": PRUNE_LAYERS,
        "final_budget_tokens": FINAL_BUDGET_TOKENS,
        "sample_spec": sample_spec,
        "expected_indices": expected_indices,
        "selection_meta": selection_meta,
        "model_path": str(args.model_path),
        "longbench_dir": str(args.longbench_dir),
        "selection_audit_csv": str(audit_csv),
    }
    write_json(output_dir / "run_config.json", run_config)

    if not args.skip_run:
        gpu_info = check_gpu(args.min_free_gpu_gb)
        run_config["gpu"] = gpu_info
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
            run_dirs[method] = run_quality_method(args, output_dir, method, sample_spec)
            predictions[method] = read_predictions(run_dirs[method], method, expected_indices)
            usage[method] = read_final_prefill_usage(run_dirs[method], expected_indices)

    original_config = json.loads((run_dirs[METHOD_ORIGINAL] / "config.json").read_text(encoding="utf-8"))
    original_variant = (original_config.get("sliminfer_config") or {}).get("variant_method")
    original_pollution_count = 0
    if original_variant not in (METHOD_ORIGINAL, None, ""):
        original_pollution_count += 1
    qdc_config = json.loads((run_dirs[METHOD_QDC] / "config.json").read_text(encoding="utf-8"))
    qdc_variant = (qdc_config.get("sliminfer_config") or {}).get("variant_method")
    if qdc_variant != METHOD_QDC:
        original_pollution_count += 1

    qdc_audit = read_qdc_runtime_audit(run_dirs[METHOD_QDC], expected_indices)
    method_scores = compute_method_summary(predictions, expected_indices)
    per_sample_quality = build_per_sample_quality(
        selected_rows, predictions, usage, qdc_audit, offline_audit
    )
    qdc_audit_summary = summarize_qdc_audit(per_sample_quality)
    decision = decide(
        method_scores,
        per_sample_quality,
        qdc_audit_summary,
        original_pollution_count,
        expected_indices,
    )

    quality_fields = [
        "task", "sample_index", "sample_id", "method", "prediction", "score",
        "selected_token_count", "budget", "budget_mismatch", "fallback",
        "qdc_triggered", "qdc_trigger_reason",
        "original_component_coverage", "qdc_component_coverage", "coverage_delta",
        "selected_jaccard_vs_original", "removed_block_ids", "added_block_ids",
        "original_query_overlap", "qdc_query_overlap", "query_overlap_delta",
        "original_redundancy", "qdc_redundancy", "redundancy_delta",
    ]
    write_csv(output_dir / "per_sample_quality.csv", per_sample_quality, quality_fields)

    summary_rows = []
    for task in TASKS + ["macro"]:
        summary_rows.append({
            "task": task,
            "original": method_scores[METHOD_ORIGINAL][task],
            "qdc": method_scores[METHOD_QDC][task],
            "delta": method_scores[METHOD_QDC][task] - method_scores[METHOD_ORIGINAL][task],
        })
    write_csv(output_dir / "qdc_quality_summary.csv", summary_rows, ["task", "original", "qdc", "delta"])

    gate_rows = []
    for row in per_sample_quality:
        if row["method"] != METHOD_QDC:
            continue
        gate_rows.append({
            "task": row["task"],
            "sample_index": row["sample_index"],
            "sample_id": row["sample_id"],
            "qdc_triggered": row["qdc_triggered"],
            "qdc_trigger_reason": row.get("qdc_trigger_reason", ""),
            "fallback": row.get("fallback"),
            "budget_mismatch": row.get("budget_mismatch"),
            "original_component_coverage": row.get("original_component_coverage"),
            "qdc_component_coverage": row.get("qdc_component_coverage"),
            "coverage_delta": row.get("coverage_delta"),
            "selected_jaccard_vs_original": row.get("selected_jaccard_vs_original"),
            "removed_block_ids": row.get("removed_block_ids"),
            "added_block_ids": row.get("added_block_ids"),
            "original_score": row.get("original_score"),
            "qdc_score": row.get("qdc_score"),
            "qdc_delta_vs_original": row.get("qdc_delta_vs_original"),
        })
    write_csv(
        output_dir / "qdc_gate_audit.csv",
        gate_rows,
        [
            "task", "sample_index", "sample_id",
            "qdc_triggered", "qdc_trigger_reason", "fallback", "budget_mismatch",
            "original_component_coverage", "qdc_component_coverage", "coverage_delta",
            "selected_jaccard_vs_original", "removed_block_ids", "added_block_ids",
            "original_score", "qdc_score", "qdc_delta_vs_original",
        ],
    )

    report = build_report(
        output_dir, selection_meta, selected_rows, method_scores,
        per_sample_quality, qdc_audit_summary, decision,
    )
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    write_json(output_dir / "decision.json", decision)
    decision_md = "\n".join([
        f"# Decision: {decision.get('decision')}",
        "",
        "## Reasons",
        *[f"- {reason}" for reason in decision.get("reasons", [])],
        "",
        f"- Macro delta: {decision.get('macro_delta', 0):+.4f}",
        f"- Trigger subset delta: {decision.get('trigger_delta', 0):+.4f}",
        f"- Control subset delta: {decision.get('control_delta', 0):+.4f}",
        "",
    ])
    (output_dir / "decision.md").write_text(decision_md, encoding="utf-8")

    print(report)
    print(f"Wrote results to {output_dir}")


if __name__ == "__main__":
    main()
