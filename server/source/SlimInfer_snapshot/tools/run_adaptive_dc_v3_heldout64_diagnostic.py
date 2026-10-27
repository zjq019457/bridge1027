#!/usr/bin/env python3
"""Run held-out 64-sample Adaptive DC-SlimInfer v3 diagnostic per task.

Compares sliminfer_original, dc_sliminfer, and adaptive_dc_sliminfer_v3 on
128 held-out samples (64 HotpotQA + 64 MuSiQue), excluding the tiny 20-sample
development set. Uses frozen threshold/normalization from the tiny GO run.
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
DEFAULT_TINY_GO_RUN = Path("results/adaptive_dc_v3_tiny_quality_smoke_20260708_060002")
DEFAULT_V3_AUDIT_DIR = Path("results/adaptive_dc_router_v3_feature_audit_20260708_043456")
OBSOLETE_PROXY_RUN = Path("results/adaptive_dc_v3_tiny_quality_smoke_20260708_053337")
VALID_TINY_GO_RUN = Path("results/adaptive_dc_v3_tiny_quality_smoke_20260708_060002")

METHOD_ORIGINAL = "sliminfer_original"
METHOD_DC = "dc_sliminfer"
METHOD_ADAPTIVE = "adaptive_dc_sliminfer_v3"
ALL_METHODS = [METHOD_ORIGINAL, METHOD_DC, METHOD_ADAPTIVE]
TASKS = ["hotpotqa", "musique"]
TINY_EXCLUDE = {
    "hotpotqa": {0, 1, 2, 3, 4, 5, 41, 56, 57, 117},
    "musique": {0, 1, 2, 3, 4, 20, 50, 54, 56, 134},
}
HELDOUT_COUNT = 64
PRUNE_LAYERS = [9, 18, 26]
GATE_APPLY_LAYER = 26
FINAL_BUDGET_TOKENS = 4096
HOTPOTQA_DROP_TOLERANCE = 2.0
HOTPOT_CATASTROPHIC_THRESHOLD = -5.0

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
    items = [float(v) for v in values if math.isfinite(float(v))]
    return float(sum(items) / len(items)) if items else 0.0


def heldout_indices(task: str, count: int = HELDOUT_COUNT, total: int = 200) -> List[int]:
    exclude = TINY_EXCLUDE[task]
    selected: List[int] = []
    for idx in range(total):
        if idx in exclude:
            continue
        selected.append(idx)
        if len(selected) >= count:
            break
    if len(selected) != count:
        raise RuntimeError(f"held-out selection for {task} produced {len(selected)} != {count}")
    return selected


def build_heldout_spec() -> str:
    parts = [f"{task}:" + ",".join(str(i) for i in heldout_indices(task)) for task in TASKS]
    return ";".join(parts)


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


def python_bin() -> Path:
    candidate = REPO_ROOT / ".venv" / "bin" / "python"
    return candidate if candidate.exists() else Path(sys.executable)


def load_frozen_gate_config(tiny_go_run: Path, v3_audit_dir: Path) -> Dict[str, Any]:
    run_config_path = tiny_go_run / "run_config.json"
    gate_json_path = tiny_go_run / "adaptive_dc_v3_gate_config.json"
    threshold: Optional[float] = None
    norm_min: Dict[str, float] = {}
    norm_max: Dict[str, float] = {}
    source_notes: List[str] = []

    if run_config_path.exists():
        run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
        frozen = run_config.get("frozen_gate_config") or {}
        if frozen.get("evidence_dispersion_threshold") is not None:
            threshold = float(frozen["evidence_dispersion_threshold"])
            norm_min = {k: float(v) for k, v in (frozen.get("norm_min") or {}).items()}
            norm_max = {k: float(v) for k, v in (frozen.get("norm_max") or {}).items()}
            source_notes.append(str(run_config_path))

    if threshold is None and gate_json_path.exists():
        gate_json = json.loads(gate_json_path.read_text(encoding="utf-8"))
        if gate_json.get("evidence_dispersion_threshold") is not None:
            threshold = float(gate_json["evidence_dispersion_threshold"])
            norm_min = {k: float(v) for k, v in (gate_json.get("norm_min") or {}).items()}
            norm_max = {k: float(v) for k, v in (gate_json.get("norm_max") or {}).items()}
            source_notes.append(str(gate_json_path))

    if threshold is None:
        rule_csv = v3_audit_dir / "adaptive_dc_v3_rule_replay.csv"
        feature_csv = v3_audit_dir / "adaptive_dc_v3_feature_table.csv"
        if not rule_csv.exists() or not feature_csv.exists():
            raise RuntimeError("BUG_STOP_FIX_REQUIRED: frozen threshold source missing")
        with rule_csv.open("r", encoding="utf-8") as handle:
            rule_rows = list(csv.DictReader(handle))
        rule3 = [row for row in rule_rows if "Rule 3" in str(row.get("rule", ""))]
        if not rule3:
            raise RuntimeError("BUG_STOP_FIX_REQUIRED: Rule 3 not found in v3 audit")
        threshold = float(rule3[0]["diagnostic_median_evidence_dispersion_score"])
        with feature_csv.open("r", encoding="utf-8") as handle:
            feature_rows = list(csv.DictReader(handle))
        for feat in ["original_doc_entropy", "entity_coverage_gap", "pruned_query_overlap_max"]:
            vals = [float(row[feat]) for row in feature_rows if row.get(feat) not in ("", "None", "nan", None)]
            if not vals:
                raise RuntimeError(f"BUG_STOP_FIX_REQUIRED: feature {feat} missing in v3 audit")
            norm_min[feat] = min(vals)
            norm_max[feat] = max(vals)
        source_notes.append(str(v3_audit_dir))

    if threshold is None:
        raise RuntimeError("BUG_STOP_FIX_REQUIRED: frozen threshold unavailable")

    return {
        "evidence_dispersion_threshold": threshold,
        "norm_min": norm_min,
        "norm_max": norm_max,
        "gate_apply_layer": GATE_APPLY_LAYER,
        "top_pruned_m": 16,
        "source_tiny_go_run": str(tiny_go_run),
        "source_notes": source_notes,
        "threshold_NOT_recomputed_from_current_batch": True,
        "runtime_feature_mode": "runtime_frozen_norm",
    }


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
        "--longbench_subset_name", "adaptive_dc_v3_heldout64_diagnostic",
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
    gate_json_path: Optional[Path] = None,
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
    if method in (METHOD_DC, METHOD_ADAPTIVE):
        cmd.extend([
            "--dc_activation_audit_path", str(audit_dir / "dc_runtime_activation_audit.jsonl"),
            "--dc_selection_audit_path", str(audit_dir / "dc_runtime_selection_audit.jsonl"),
        ])
        (audit_dir / "dc_runtime_activation_audit.jsonl").write_text("", encoding="utf-8")
        (audit_dir / "dc_runtime_selection_audit.jsonl").write_text("", encoding="utf-8")
    if method == METHOD_ADAPTIVE and gate_json_path is not None:
        cmd.extend(["--adaptive_dc_v3_gate_json", str(gate_json_path)])
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
            sample_idx = int(row.get("sample_index", expected_idx))
            if sample_idx != int(expected_idx):
                raise RuntimeError(f"{method} {task} sample mismatch: expected {expected_idx}, got {sample_idx}")
            pred = str(row.get("pred", ""))
            if not pred.strip():
                raise RuntimeError(f"{method} {task}:{sample_idx} empty prediction")
            out[(task, sample_idx)] = {
                "task": task,
                "sample_index": sample_idx,
                "sample_id": row.get("sample_id", sample_idx),
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


def read_adaptive_gate_audit(run_dir: Path, expected_indices: Dict[str, List[int]]) -> Dict[Key, JsonRow]:
    path = run_dir / "audit" / "dc_runtime_selection_audit.jsonl"
    rows = [
        row for row in read_jsonl(path, missing_ok=True)
        if str(row.get("phase", "")) == "prefill" and int(row.get("layer_id", -1)) == GATE_APPLY_LAYER
    ]
    by_key: Dict[Key, JsonRow] = {}
    for row in rows:
        key = key_of(row)
        if key[0] and key[1] >= 0:
            by_key[key] = row
    missing = [(task, idx) for task in TASKS for idx in expected_indices[task] if (task, idx) not in by_key]
    if missing:
        raise RuntimeError(f"missing adaptive gate audit rows at layer {GATE_APPLY_LAYER}: {missing[:10]}")
    return by_key


def summarize_dc_audit(run_dir: Path, expected_indices: Dict[str, List[int]]) -> Tuple[Dict[Key, JsonRow], Dict[str, Any]]:
    activation_rows = [
        row for row in read_jsonl(run_dir / "audit" / "dc_runtime_activation_audit.jsonl", missing_ok=True)
        if str(row.get("phase", "")) == "prefill" and int(row.get("layer_id", -1)) in PRUNE_LAYERS
    ]
    selection_rows = [
        row for row in read_jsonl(run_dir / "audit" / "dc_runtime_selection_audit.jsonl", missing_ok=True)
        if str(row.get("phase", "")) == "prefill" and int(row.get("layer_id", -1)) in PRUNE_LAYERS
    ]
    by_sample: Dict[Key, List[JsonRow]] = defaultdict(list)
    for row in activation_rows:
        by_sample[key_of(row)].append(row)
    per_sample: Dict[Key, JsonRow] = {}
    for key, rows_list in by_sample.items():
        rows_list = sorted(rows_list, key=lambda row: int(row.get("layer_id", -1)))
        per_sample[key] = {
            "selection_changed": any(bool(row.get("selection_changed")) for row in rows_list),
            "selected_jaccard_vs_original": mean(float(row.get("selected_jaccard", 1.0) or 1.0) for row in rows_list),
            "dc_entered_before_virtual_return": any(bool(row.get("entered_dc_selector_before_virtual_return")) for row in rows_list),
            "virtual_block_path_returned_before_dc_selector": any(
                str(row.get("virtual_return_reason", "")) == "virtual_block_path_returned_before_dc_selector"
                for row in rows_list
            ),
            "fallback_used": any(bool(row.get("fallback_reason")) for row in rows_list),
            "budget_mismatch": any(not bool(row.get("budget_preserved", True)) for row in rows_list),
        }
    expected_total = sum(len(expected_indices[task]) for task in TASKS)
    summary = {
        "activation_rows": len(activation_rows),
        "selection_rows": len(selection_rows),
        "expected_samples": expected_total,
        "budget_mismatch_count": sum(1 for row in per_sample.values() if bool(row.get("budget_mismatch"))),
        "fallback_count": sum(1 for row in per_sample.values() if bool(row.get("fallback_used"))),
        "entered_before_virtual_return_samples": sum(
            1 for row in per_sample.values() if bool(row.get("dc_entered_before_virtual_return"))
        ),
    }
    return per_sample, summary


def build_per_sample_quality(
    predictions: Dict[str, Dict[Key, JsonRow]],
    usage: Dict[str, Dict[Key, JsonRow]],
    gate_audit: Dict[Key, JsonRow],
    dc_audit: Dict[Key, JsonRow],
    gate_config: Dict[str, Any],
    expected_indices: Dict[str, List[int]],
) -> List[JsonRow]:
    rows: List[JsonRow] = []
    original_preds = predictions[METHOD_ORIGINAL]
    dc_preds = predictions[METHOD_DC]
    adaptive_preds = predictions[METHOD_ADAPTIVE]
    threshold = float(gate_config.get("evidence_dispersion_threshold", 0))

    for task in TASKS:
        for sample_idx in expected_indices[task]:
            key = (task, sample_idx)
            orig = original_preds[key]
            dc = dc_preds[key]
            adap = adaptive_preds[key]
            used = usage.get(METHOD_ADAPTIVE, {}).get(key, {})
            gate_row = gate_audit.get(key, {})
            dc_row = dc_audit.get(key, {})

            selected_token_count = int(used.get("kv_abs_indices_count", 0) or 0)
            adaptive_delta_vs_original = float(adap["score"]) - float(orig["score"])
            adaptive_delta_vs_dc = float(adap["score"]) - float(dc["score"])

            gate_triggered = bool(gate_row.get("adaptive_gate_triggered"))
            if "adaptive_gate_triggered" not in gate_row and gate_row.get("adaptive_dc_v3_gate"):
                gate_triggered = bool((gate_row.get("adaptive_dc_v3_gate") or {}).get("triggered"))

            adap_pred = str(adap["prediction"]).strip()
            orig_pred = str(orig["prediction"]).strip()
            dc_pred = str(dc["prediction"]).strip()
            if adap_pred == orig_pred:
                chosen_selector = "original"
            elif adap_pred == dc_pred:
                chosen_selector = "dc"
            else:
                chosen_selector = "unknown"

            rows.append({
                "task": task,
                "sample_index": sample_idx,
                "sample_id": adap.get("sample_id", sample_idx),
                "method": METHOD_ADAPTIVE,
                "prediction": adap["prediction"],
                "score": adap["score"],
                "selected_token_count": selected_token_count,
                "budget": FINAL_BUDGET_TOKENS,
                "budget_mismatch": selected_token_count > FINAL_BUDGET_TOKENS,
                "fallback": bool(dc_row.get("fallback_used")),
                "adaptive_gate_triggered": gate_triggered,
                "adaptive_gate_reason": gate_row.get("adaptive_gate_reason", ""),
                "evidence_dispersion_score": gate_row.get("evidence_dispersion_score"),
                "evidence_dispersion_threshold": gate_row.get("evidence_dispersion_threshold", threshold),
                "entity_coverage_gap": gate_row.get("entity_coverage_gap"),
                "pruned_query_overlap_max": gate_row.get("pruned_query_overlap_max"),
                "selected_query_overlap_mean": gate_row.get("selected_query_overlap_mean"),
                "selected_jaccard_vs_original": gate_row.get("selected_jaccard_vs_original"),
                "chosen_selector": chosen_selector,
                "original_score": orig["score"],
                "dc_score": dc["score"],
                "adaptive_score": adap["score"],
                "adaptive_delta_vs_original": adaptive_delta_vs_original,
                "adaptive_delta_vs_dc": adaptive_delta_vs_dc,
                "original_entered_dc_selector": False,
                "dc_entered_before_virtual_return": bool(dc_row.get("dc_entered_before_virtual_return")),
                "virtual_block_path_returned_before_dc_selector": bool(
                    dc_row.get("virtual_block_path_returned_before_dc_selector")
                ),
            })
    return rows


def compute_method_summary(
    predictions: Dict[str, Dict[Key, JsonRow]],
    expected_indices: Dict[str, List[int]],
) -> Dict[str, Dict[str, float]]:
    method_scores: Dict[str, Dict[str, float]] = {}
    for method in ALL_METHODS:
        preds = predictions[method]
        task_means: Dict[str, float] = {}
        for task in TASKS:
            task_scores = [float(preds[(task, idx)]["score"]) for idx in expected_indices[task]]
            task_means[task] = mean(task_scores)
        task_means["macro"] = mean(task_means[task] for task in TASKS)
        method_scores[method] = task_means
    return method_scores


def decide(
    method_scores: Dict[str, Dict[str, float]],
    per_sample: List[JsonRow],
    gate_config: Dict[str, Any],
    original_pollution_count: int,
    adaptive_dc_audit_summary: Dict[str, Any],
) -> Dict[str, Any]:
    bug_reasons: List[str] = []
    no_go_reasons: List[str] = []
    weak_go_reasons: List[str] = []

    orig_macro = method_scores[METHOD_ORIGINAL]["macro"]
    dc_macro = method_scores[METHOD_DC]["macro"]
    adap_macro = method_scores[METHOD_ADAPTIVE]["macro"]
    hotpotqa_delta = method_scores[METHOD_ADAPTIVE]["hotpotqa"] - method_scores[METHOD_ORIGINAL]["hotpotqa"]
    musique_delta = method_scores[METHOD_ADAPTIVE]["musique"] - method_scores[METHOD_ORIGINAL]["musique"]

    trigger_count = sum(1 for row in per_sample if bool(row.get("adaptive_gate_triggered")))
    total_samples = len(per_sample)
    positive_vs_orig = sum(1 for row in per_sample if float(row.get("adaptive_delta_vs_original", 0)) > 1e-9)
    negative_vs_orig = sum(1 for row in per_sample if float(row.get("adaptive_delta_vs_original", 0)) < -1e-9)
    zero_vs_orig = sum(1 for row in per_sample if abs(float(row.get("adaptive_delta_vs_original", 0))) <= 1e-9)
    hotpot_catastrophic = sum(
        1 for row in per_sample
        if row["task"] == "hotpotqa"
        and bool(row.get("adaptive_gate_triggered"))
        and float(row.get("adaptive_delta_vs_original", 0)) <= HOTPOT_CATASTROPHIC_THRESHOLD
    )
    musique_positive_retained = sum(
        1 for row in per_sample
        if row["task"] == "musique"
        and bool(row.get("adaptive_gate_triggered"))
        and float(row.get("adaptive_delta_vs_original", 0)) > 1e-9
    )
    musique_negative_retained = sum(
        1 for row in per_sample
        if row["task"] == "musique"
        and bool(row.get("adaptive_gate_triggered"))
        and float(row.get("adaptive_delta_vs_original", 0)) < -1e-9
    )
    budget_mismatch_count = sum(1 for row in per_sample if bool(row.get("budget_mismatch")))
    fallback_count = int(adaptive_dc_audit_summary.get("fallback_count", 0))

    if gate_config.get("evidence_dispersion_threshold") is None:
        bug_reasons.append("frozen_threshold_missing")
    if not gate_config.get("threshold_NOT_recomputed_from_current_batch"):
        bug_reasons.append("threshold_not_marked_frozen")
    if budget_mismatch_count != 0:
        bug_reasons.append("budget_mismatch_nonzero")
    if fallback_count != 0:
        bug_reasons.append("fallback_nonzero")
    if original_pollution_count != 0:
        bug_reasons.append("original_selector_polluted")
    if bug_reasons:
        return {"decision": "BUG_STOP_FIX_REQUIRED", "reasons": bug_reasons}

    if trigger_count == 0:
        no_go_reasons.append("gate_all_reject")
    if trigger_count == total_samples:
        no_go_reasons.append("gate_all_trigger")
    if adap_macro <= orig_macro:
        no_go_reasons.append("adaptive_macro_not_above_original")
    if hotpotqa_delta < -HOTPOTQA_DROP_TOLERANCE:
        no_go_reasons.append(f"hotpotqa_delta={hotpotqa_delta:.4f}_below_tolerance")
    if musique_delta <= 0 and musique_positive_retained <= musique_negative_retained:
        no_go_reasons.append("musique_no_clear_positive_retention")
    if hotpot_catastrophic >= 3:
        no_go_reasons.append(f"hotpot_catastrophic_triggered={hotpot_catastrophic}")

    if no_go_reasons:
        return {"decision": "ADAPTIVE_DC_V3_HELDOUT64_NO_GO", "reasons": no_go_reasons}

    if adap_macro <= dc_macro - 1e-9:
        weak_go_reasons.append("adaptive_macro_not_above_unconditional_dc")
    if hotpotqa_delta < 0:
        weak_go_reasons.append(f"hotpotqa_small_drop={hotpotqa_delta:.4f}")
    if trigger_count <= max(2, total_samples // 20):
        weak_go_reasons.append(f"sparse_trigger_count={trigger_count}")
    if hotpot_catastrophic > 0:
        weak_go_reasons.append(f"hotpot_catastrophic_triggered={hotpot_catastrophic}")

    if weak_go_reasons:
        return {
            "decision": "ADAPTIVE_DC_V3_HELDOUT64_WEAK_GO",
            "reasons": weak_go_reasons + [
                f"adaptive_macro={adap_macro:.4f} > original={orig_macro:.4f}",
                f"trigger_count={trigger_count}/{total_samples}",
            ],
        }

    return {
        "decision": "ADAPTIVE_DC_V3_HELDOUT64_GO",
        "reasons": [
            f"trigger_count={trigger_count}/{total_samples}",
            f"adaptive_macro={adap_macro:.4f} > original={orig_macro:.4f}",
            f"adaptive_macro={adap_macro:.4f} >= unconditional_dc={dc_macro:.4f}",
            f"hotpotqa_delta={hotpotqa_delta:+.4f}",
            f"musique_delta={musique_delta:+.4f}",
            f"musique_positive_retained={musique_positive_retained}",
            f"hotpot_catastrophic={hotpot_catastrophic}",
        ],
    }


def write_obsolete_proxy_note(output_dir: Path) -> None:
    lines = [
        "# Obsolete Proxy Selection Run Note",
        "",
        "The following early Adaptive DC v3 tiny run used the **pre-fix proxy selection gate**.",
        "It is **superseded** and must not be used for method decisions.",
        "",
        f"- Obsolete run: `{OBSOLETE_PROXY_RUN}`",
        f"- Valid frozen-features tiny GO run: `{VALID_TINY_GO_RUN}`",
        "",
        "## Obsolete Run Summary",
        "",
        "- Gate scheme: proxy selection (repair-before-frozen-features)",
        "- Adaptive macro: **25.05** (below original macro 28.86)",
        "- Decision: `ADAPTIVE_DC_V3_TINY_CONFIRMATION_NO_GO`",
        "- Trigger mismatch vs offline Rule 3 replay",
        "",
        "## Valid Run Summary",
        "",
        "- Gate scheme: frozen-features runtime gate (tiny) / runtime_frozen_norm (held-out)",
        "- Adaptive macro: **37.19**",
        "- Decision: `ADAPTIVE_DC_V3_TINY_CONFIRMATION_GO`",
        "",
        "## Policy",
        "",
        "- Do **not** compare or aggregate obsolete proxy runs with current frozen-features Adaptive DC v3.",
        "- Do **not** delete obsolete results; keep them only as historical failure evidence.",
        "- All subsequent Adaptive DC v3 decisions must use frozen threshold + frozen normalization",
        "  from the valid tiny GO run, not batch-recomputed thresholds.",
        "",
    ]
    (output_dir / "obsolete_proxy_run_note.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(
    output_dir: Path,
    method_scores: Dict[str, Dict[str, float]],
    per_sample: List[JsonRow],
    gate_config: Dict[str, Any],
    decision: Dict[str, Any],
    original_pollution_count: int,
    adaptive_dc_audit_summary: Dict[str, Any],
    expected_indices: Dict[str, List[int]],
) -> None:
    orig = method_scores[METHOD_ORIGINAL]
    dc = method_scores[METHOD_DC]
    adap = method_scores[METHOD_ADAPTIVE]
    trigger_count = sum(1 for row in per_sample if bool(row.get("adaptive_gate_triggered")))
    triggered_keys = sorted(f"{row['task']}:{row['sample_index']}" for row in per_sample if bool(row.get("adaptive_gate_triggered")))
    positive = sum(1 for row in per_sample if float(row.get("adaptive_delta_vs_original", 0)) > 1e-9)
    negative = sum(1 for row in per_sample if float(row.get("adaptive_delta_vs_original", 0)) < -1e-9)
    zero = sum(1 for row in per_sample if abs(float(row.get("adaptive_delta_vs_original", 0))) <= 1e-9)
    hotpot_cat = sum(
        1 for row in per_sample
        if row["task"] == "hotpotqa"
        and bool(row.get("adaptive_gate_triggered"))
        and float(row.get("adaptive_delta_vs_original", 0)) <= HOTPOT_CATASTROPHIC_THRESHOLD
    )
    mus_pos = sum(
        1 for row in per_sample
        if row["task"] == "musique"
        and bool(row.get("adaptive_gate_triggered"))
        and float(row.get("adaptive_delta_vs_original", 0)) > 1e-9
    )
    budget_mm = sum(1 for row in per_sample if bool(row.get("budget_mismatch")))
    fallback_count = int(adaptive_dc_audit_summary.get("fallback_count", 0))

    lines = [
        "# Adaptive DC-SlimInfer v3 Held-Out 64-Sample Diagnostic",
        "",
        f"- Result directory: `{output_dir}`",
        "- Methods: `sliminfer_original`, `dc_sliminfer`, `adaptive_dc_sliminfer_v3`",
        "- Generation: **Yes** (128 held-out samples x 3 methods)",
        "- Scorer: **Yes**",
        "- Full LongBench: **No**",
        "- Threshold sweep: **No**",
        "",
        "## 1. Artifact Self-Check",
        "",
        f"- Prediction files exist: **Yes**",
        f"- Original/DC/Adaptive each have 128 rows: **Yes**",
        f"- HotpotQA rows per method: 64",
        f"- MuSiQue rows per method: 64",
        f"- Budget mismatch count: {budget_mm}",
        f"- Fallback count: {fallback_count}",
        f"- Original selector pollution count: {original_pollution_count}",
        f"- DC audit activation rows: {adaptive_dc_audit_summary.get('activation_rows', 'N/A')}",
        f"- DC audit selection rows: {adaptive_dc_audit_summary.get('selection_rows', 'N/A')}",
        "",
        "## 2. Frozen Threshold",
        "",
        f"- Source tiny GO run: `{gate_config.get('source_tiny_go_run', 'N/A')}`",
        f"- Source notes: {gate_config.get('source_notes', [])}",
        f"- evidence_dispersion_threshold: {gate_config.get('evidence_dispersion_threshold')}",
        f"- Normalization min: {json.dumps(gate_config.get('norm_min', {}), sort_keys=True)}",
        f"- Normalization max: {json.dumps(gate_config.get('norm_max', {}), sort_keys=True)}",
        f"- Runtime feature mode: `{gate_config.get('runtime_feature_mode')}`",
        f"- Threshold recomputed from current batch: **No**",
        "",
        "## 3. Held-Out Sample Selection",
        "",
        f"- Excluded tiny HotpotQA: {sorted(TINY_EXCLUDE['hotpotqa'])}",
        f"- Excluded tiny MuSiQue: {sorted(TINY_EXCLUDE['musique'])}",
        f"- Held-out HotpotQA count: {len(expected_indices['hotpotqa'])}",
        f"- Held-out MuSiQue count: {len(expected_indices['musique'])}",
        f"- Held-out HotpotQA ids (first 10): {expected_indices['hotpotqa'][:10]}",
        f"- Held-out MuSiQue ids (first 10): {expected_indices['musique'][:10]}",
        "",
        "## 4. Gate Trigger",
        "",
        f"- Trigger count: {trigger_count}/128",
        f"- Trigger sample list (first 20): {', '.join(triggered_keys[:20]) if triggered_keys else '(none)'}",
        f"- Non-all-reject: {trigger_count > 0}",
        f"- Non-all-trigger: {trigger_count < 128}",
        "",
        "## 5. Task Scores",
        "",
        "| task | original | dc | adaptive |",
        "| --- | --- | --- | --- |",
        f"| HotpotQA | {orig['hotpotqa']:.4f} | {dc['hotpotqa']:.4f} | {adap['hotpotqa']:.4f} |",
        f"| MuSiQue | {orig['musique']:.4f} | {dc['musique']:.4f} | {adap['musique']:.4f} |",
        f"| 2-task macro | {orig['macro']:.4f} | {dc['macro']:.4f} | {adap['macro']:.4f} |",
        "",
        f"- Adaptive vs original delta: {adap['macro'] - orig['macro']:+.4f}",
        f"- Adaptive vs unconditional DC delta: {adap['macro'] - dc['macro']:+.4f}",
        f"- HotpotQA adaptive delta: {adap['hotpotqa'] - orig['hotpotqa']:+.4f}",
        f"- MuSiQue adaptive delta: {adap['musique'] - orig['musique']:+.4f}",
        "",
        "## 6. Sample Analysis",
        "",
        f"- Positive / negative / zero count vs original: {positive} / {negative} / {zero}",
        f"- Triggered HotpotQA catastrophic negative count: {hotpot_cat}",
        f"- Retained MuSiQue positive count (triggered): {mus_pos}",
        "",
        "## 7. Obsolete Proxy Run",
        "",
        f"- Obsolete proxy run marked superseded: `{OBSOLETE_PROXY_RUN}`",
        f"- Valid tiny GO reference: `{VALID_TINY_GO_RUN}`",
        "",
        "## 8. Decision",
        "",
        f"- Final conclusion: **{decision['decision']}**",
    ]
    for reason in decision.get("reasons", []):
        lines.append(f"- Reason: `{reason}`")
    lines.append("")
    lines.append("This is a held-out diagnostic only, not a full LongBench result or a statistical claim.")
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_decision(output_dir: Path, decision: Dict[str, Any], gate_config: Dict[str, Any]) -> None:
    lines = [
        "# Adaptive DC-SlimInfer v3 Held-Out 64 Diagnostic Decision",
        "",
        f"- Decision: **{decision['decision']}**",
    ]
    for reason in decision.get("reasons", []):
        lines.append(f"- Reason: `{reason}`")
    lines.append("")
    lines.append(f"- Frozen threshold source: `{gate_config.get('source_tiny_go_run', 'N/A')}`")
    lines.append(f"- evidence_dispersion_threshold: {gate_config.get('evidence_dispersion_threshold')}")
    lines.append(f"- Output directory: `{output_dir}`")
    (output_dir / "decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--longbench_dir", type=Path, default=DEFAULT_LONGBENCH_DIR)
    parser.add_argument("--tiny_go_run", type=Path, default=DEFAULT_TINY_GO_RUN)
    parser.add_argument("--v3_audit_dir", type=Path, default=DEFAULT_V3_AUDIT_DIR)
    parser.add_argument("--output_root", type=Path, default=Path("results"))
    parser.add_argument("--min_free_gpu_gb", type=float, default=20.0)
    parser.add_argument("--heldout_count", type=int, default=HELDOUT_COUNT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (REPO_ROOT / args.output_root / f"adaptive_dc_v3_heldout64_diagnostic_{timestamp}").resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    expected_indices = {task: heldout_indices(task, count=args.heldout_count) for task in TASKS}
    sample_spec = build_heldout_spec()
    gate_config = load_frozen_gate_config(REPO_ROOT / args.tiny_go_run, REPO_ROOT / args.v3_audit_dir)
    gate_json_path = output_dir / "adaptive_dc_v3_gate_config.json"
    write_json(gate_json_path, gate_config)

    run_config = {
        "methods": ALL_METHODS,
        "tasks": TASKS,
        "heldout_samples": expected_indices,
        "excluded_tiny_samples": {task: sorted(values) for task, values in TINY_EXCLUDE.items()},
        "sample_spec": sample_spec,
        "generation_max_new_tokens": 32,
        "seed": 0,
        "decode_policy": "original",
        "budget_tokens": FINAL_BUDGET_TOKENS,
        "prune_layers": PRUNE_LAYERS,
        "frozen_gate_config": gate_config,
        "valid_tiny_go_run": str(VALID_TINY_GO_RUN),
        "obsolete_proxy_run": str(OBSOLETE_PROXY_RUN),
        "threshold_NOT_recomputed_from_current_batch": True,
    }
    write_json(output_dir / "run_config.json", run_config)
    write_obsolete_proxy_note(output_dir)

    gpu_checks = [check_gpu(args.min_free_gpu_gb)]
    run_dirs: Dict[str, Path] = {}
    predictions: Dict[str, Dict[Key, JsonRow]] = {}
    usage: Dict[str, Dict[Key, JsonRow]] = {}

    for method in ALL_METHODS:
        gpu_checks.append(check_gpu(args.min_free_gpu_gb))
        run_dir = run_quality_method(
            args,
            output_dir,
            method,
            sample_spec=sample_spec,
            gate_json_path=gate_json_path if method == METHOD_ADAPTIVE else None,
        )
        run_dirs[method] = run_dir
        predictions[method] = read_predictions(run_dir, method, expected_indices)
        usage[method] = read_final_prefill_usage(run_dir, expected_indices)

    original_config = json.loads((run_dirs[METHOD_ORIGINAL] / "config.json").read_text(encoding="utf-8"))
    original_variant = (original_config.get("sliminfer_config") or {}).get("variant_method")
    original_pollution_count = 0
    if original_variant in (METHOD_DC, METHOD_ADAPTIVE):
        original_pollution_count += 1
    original_dc_audit = run_dirs[METHOD_ORIGINAL] / "audit" / "dc_runtime_activation_audit.jsonl"
    if original_dc_audit.exists() and read_jsonl(original_dc_audit, missing_ok=True):
        original_pollution_count += 1

    adaptive_dc_audit, adaptive_dc_audit_summary = summarize_dc_audit(run_dirs[METHOD_ADAPTIVE], expected_indices)
    gate_audit = read_adaptive_gate_audit(run_dirs[METHOD_ADAPTIVE], expected_indices)
    method_scores = compute_method_summary(predictions, expected_indices)
    per_sample_quality = build_per_sample_quality(
        predictions, usage, gate_audit, adaptive_dc_audit, gate_config, expected_indices
    )
    decision = decide(method_scores, per_sample_quality, gate_config, original_pollution_count, adaptive_dc_audit_summary)

    quality_fields = [
        "task", "sample_index", "sample_id", "method", "score",
        "selected_token_count", "budget", "budget_mismatch", "fallback",
        "adaptive_gate_triggered", "adaptive_gate_reason",
        "evidence_dispersion_score", "evidence_dispersion_threshold",
        "entity_coverage_gap", "pruned_query_overlap_max",
        "selected_query_overlap_mean", "selected_jaccard_vs_original",
        "chosen_selector",
        "original_score", "dc_score", "adaptive_score",
        "adaptive_delta_vs_original", "adaptive_delta_vs_dc",
        "original_entered_dc_selector",
        "dc_entered_before_virtual_return",
        "virtual_block_path_returned_before_dc_selector",
    ]
    write_csv(output_dir / "per_sample_quality.csv", per_sample_quality, quality_fields)
    write_jsonl(output_dir / "per_sample_quality.jsonl", per_sample_quality)

    gate_audit_rows = []
    for row in per_sample_quality:
        gate_audit_rows.append({
            "task": row["task"],
            "sample_index": row["sample_index"],
            "sample_id": row["sample_id"],
            "adaptive_gate_triggered": row["adaptive_gate_triggered"],
            "adaptive_gate_reason": row.get("adaptive_gate_reason", ""),
            "chosen_selector": row["chosen_selector"],
            "evidence_dispersion_score": row.get("evidence_dispersion_score"),
            "evidence_dispersion_threshold": row.get("evidence_dispersion_threshold"),
            "entity_coverage_gap": row.get("entity_coverage_gap"),
            "pruned_query_overlap_max": row.get("pruned_query_overlap_max"),
            "selected_query_overlap_mean": row.get("selected_query_overlap_mean"),
            "selected_jaccard_vs_original": row.get("selected_jaccard_vs_original"),
            "original_score": row["original_score"],
            "dc_score": row["dc_score"],
            "adaptive_score": row["adaptive_score"],
            "adaptive_delta_vs_original": row["adaptive_delta_vs_original"],
            "adaptive_delta_vs_dc": row["adaptive_delta_vs_dc"],
        })
    gate_fields = [
        "task", "sample_index", "sample_id",
        "adaptive_gate_triggered", "adaptive_gate_reason", "chosen_selector",
        "evidence_dispersion_score", "evidence_dispersion_threshold",
        "entity_coverage_gap", "pruned_query_overlap_max",
        "selected_query_overlap_mean", "selected_jaccard_vs_original",
        "original_score", "dc_score", "adaptive_score",
        "adaptive_delta_vs_original", "adaptive_delta_vs_dc",
    ]
    write_csv(output_dir / "adaptive_gate_audit.csv", gate_audit_rows, gate_fields)

    summary_rows = []
    for task in TASKS:
        summary_rows.append({
            "task": task,
            "original": method_scores[METHOD_ORIGINAL][task],
            "dc": method_scores[METHOD_DC][task],
            "adaptive": method_scores[METHOD_ADAPTIVE][task],
            "adaptive_delta_vs_original": method_scores[METHOD_ADAPTIVE][task] - method_scores[METHOD_ORIGINAL][task],
            "adaptive_delta_vs_dc": method_scores[METHOD_ADAPTIVE][task] - method_scores[METHOD_DC][task],
        })
    summary_rows.append({
        "task": "macro",
        "original": method_scores[METHOD_ORIGINAL]["macro"],
        "dc": method_scores[METHOD_DC]["macro"],
        "adaptive": method_scores[METHOD_ADAPTIVE]["macro"],
        "adaptive_delta_vs_original": method_scores[METHOD_ADAPTIVE]["macro"] - method_scores[METHOD_ORIGINAL]["macro"],
        "adaptive_delta_vs_dc": method_scores[METHOD_ADAPTIVE]["macro"] - method_scores[METHOD_DC]["macro"],
    })
    write_csv(
        output_dir / "method_summary.csv",
        summary_rows,
        ["task", "original", "dc", "adaptive", "adaptive_delta_vs_original", "adaptive_delta_vs_dc"],
    )

    write_report(
        output_dir,
        method_scores,
        per_sample_quality,
        gate_config,
        decision,
        original_pollution_count,
        adaptive_dc_audit_summary,
        expected_indices,
    )
    write_decision(output_dir, decision, gate_config)
    write_json(
        output_dir / "summary.json",
        {
            "result_dir": str(output_dir),
            "run_dirs": {method: str(path) for method, path in run_dirs.items()},
            "gpu_checks": gpu_checks,
            "method_scores": method_scores,
            "decision": decision,
            "gate_config_source": gate_config.get("source_tiny_go_run"),
            "frozen_threshold": gate_config.get("evidence_dispersion_threshold"),
            "heldout_samples": expected_indices,
            "trigger_count": sum(1 for row in per_sample_quality if bool(row.get("adaptive_gate_triggered"))),
            "original_not_polluted": original_pollution_count == 0,
        },
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "output_dir": str(output_dir),
                "decision": decision,
                "method_scores": method_scores,
                "trigger_count": sum(1 for row in per_sample_quality if bool(row.get("adaptive_gate_triggered"))),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
