#!/usr/bin/env python3
"""Run the LSSR Policy B tiny quality smoke.

This script runs only the fixed HotpotQA/MuSiQue 20-sample smoke for
`lssr_policy_b_middle_donor_late_receiver`.  It does not run Policy A/C, does
not run full LongBench, and does not train or call any learned verifier.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from longbench_eval import dataset2metric, scorer


DEFAULT_INSTRUMENTATION_DIR = Path("results/lssr_layer_action_instrumentation_tiny_20260707_083604")
DEFAULT_LAYER_AUDIT_DIR = Path("results/layer_stratified_sliminfer_audit_20260707_083609")
DEFAULT_REPLAY_DIR = Path("results/lssr_controller_replay_20260707_090048")
DEFAULT_HISTORICAL_DIR = Path("results/drm_budget_neutral_swap_multihop200_20260706_042717")
DEFAULT_MODEL_PATH = Path(
    "/home/yu-bai/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/"
    "snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
)
DEFAULT_LONGBENCH_DIR = Path("/home/yu-bai/Desktop/jiaqizhang/HeadKV/data/LongBench")

METHOD_ORIGINAL = "sliminfer_original"
METHOD_DRM_SWAP = "budget_neutral_drm_swap"
METHOD_LSSR = "lssr_policy_b_middle_donor_late_receiver"
FIXED_HOTPOTQA = [0, 1, 2, 3, 4, 5, 41, 56, 57, 117]
FIXED_MUSIQUE = [0, 1, 2, 3, 4, 20, 50, 54, 56, 134]
FIXED_KEYS = {
    ("hotpotqa", index) for index in FIXED_HOTPOTQA
}.union({("musique", index) for index in FIXED_MUSIQUE})
EXPECTED_ACCEPTED = {
    "musique:0",
    "musique:1",
    "musique:2",
    "musique:20",
    "musique:56",
    "musique:134",
}
HOTPOT_TOP_NEGATIVE = {0, 56, 117, 41, 57}
MUSIQUE_TOP_POSITIVE = {20, 50, 54, 134, 56}
OFFLINE_PROXY_DROP_TOLERANCE = 5.0

JsonRow = Dict[str, Any]
Key = Tuple[str, int]


def key_label(key: Key) -> str:
    return f"{key[0]}:{key[1]}"


def key_of(row: Mapping[str, Any]) -> Key:
    return str(row["task"]), int(row["sample_index"])


def read_jsonl(path: Path) -> List[JsonRow]:
    if not path.exists():
        raise RuntimeError(f"missing JSONL: {path}")
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


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def finite_score(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"non-finite score: {value!r}")
    return result


def check_gpu(min_free_gpu_gb: float) -> Dict[str, Any]:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free", "--format=csv,noheader,nounits"],
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip().splitlines()
    rows = []
    for line in out:
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
    best = max(rows, key=lambda row: row["memory_free_gib"])
    if best["memory_free_gib"] < min_free_gpu_gb:
        raise RuntimeError(
            f"free GPU memory {best['memory_free_gib']:.2f} GiB < required {min_free_gpu_gb:.2f} GiB"
        )
    return {"rows": rows, "selected": best, "min_free_gpu_gb": min_free_gpu_gb}


def load_replay_policy_b(replay_dir: Path) -> Tuple[Dict[Key, JsonRow], JsonRow]:
    decisions = {}
    for row in read_jsonl(replay_dir / "lssr_controller_replay_decisions.jsonl"):
        if row.get("policy_id") != "policy_b":
            continue
        key = key_of(row)
        if key in FIXED_KEYS:
            decisions[key] = row
    missing = sorted(FIXED_KEYS.difference(decisions), key=lambda key: (key[0], key[1]))
    if missing:
        raise RuntimeError(f"missing Policy B replay decisions: {[key_label(key) for key in missing]}")
    accepted = {key_label(key) for key, row in decisions.items() if bool(row.get("lssr_triggered"))}
    if len(accepted) != 6 or accepted != EXPECTED_ACCEPTED:
        raise RuntimeError(f"unexpected Policy B accepted set: {sorted(accepted)}")
    summary = json.loads((replay_dir / "lssr_controller_replay_summary.json").read_text(encoding="utf-8"))
    policy_b = next(
        (row for row in summary["policy_results"] if row.get("policy_id") == "policy_b"),
        None,
    )
    if policy_b is None:
        raise RuntimeError("Policy B summary missing from replay")
    if int(policy_b.get("trigger_count") or 0) != 6:
        raise RuntimeError(f"Policy B replay trigger count is not 6: {policy_b.get('trigger_count')}")
    return decisions, policy_b


def load_instrumentation(instrumentation_dir: Path) -> Dict[Key, JsonRow]:
    rows = {
        key_of(row): row
        for row in read_jsonl(instrumentation_dir / "lssr_layer_action_instrumentation.jsonl")
        if key_of(row) in FIXED_KEYS
    }
    missing = sorted(FIXED_KEYS.difference(rows), key=lambda key: (key[0], key[1]))
    if missing:
        raise RuntimeError(f"missing instrumentation rows: {[key_label(key) for key in missing]}")
    return rows


def load_historical(historical_dir: Path) -> Dict[str, Dict[Key, JsonRow]]:
    by_method: Dict[str, Dict[Key, JsonRow]] = defaultdict(dict)
    for row in read_jsonl(historical_dir / "audit" / "per_sample_audit.jsonl"):
        key = key_of(row)
        if key in FIXED_KEYS:
            by_method[str(row.get("method_key") or "")][key] = row
    for method in [METHOD_ORIGINAL, METHOD_DRM_SWAP]:
        missing = sorted(FIXED_KEYS.difference(by_method[method]), key=lambda key: (key[0], key[1]))
        if missing:
            raise RuntimeError(f"historical {method} missing samples: {[key_label(key) for key in missing]}")
    return by_method


def collect_prompt_hashes(historical_dir: Path) -> Dict[Key, str]:
    out: Dict[Key, str] = {}
    for task in ["hotpotqa", "musique"]:
        path = historical_dir / "runs" / METHOD_ORIGINAL / task / "audit" / "generations.jsonl"
        for row in read_jsonl(path):
            key = key_of(row)
            if key in FIXED_KEYS and row.get("prompt_hash"):
                out[key] = str(row["prompt_hash"])
    return out


def plan_receiver_actions(row: Mapping[str, Any], donor_count: int) -> Dict[str, List[int]]:
    by_layer: Dict[int, List[Tuple[float, int]]] = defaultdict(list)
    for layer in row.get("per_layer") or []:
        if layer.get("layer_group") != "late":
            continue
        layer_id = int(layer["layer_id"])
        ids = [int(x) for x in layer.get("candidate_receiver_block_ids") or []]
        gains = [float(x) for x in layer.get("candidate_receiver_gain_proxy") or []]
        for pos, block_id in enumerate(ids):
            gain = gains[pos] if pos < len(gains) else 0.0
            by_layer[layer_id].append((gain, block_id))
    selected_by_layer: Dict[str, List[int]] = defaultdict(list)
    remaining = int(donor_count)
    for layer_id in sorted(by_layer):
        ranked = sorted(by_layer[layer_id], key=lambda item: (-item[0], item[1]))
        for _, block_id in ranked:
            if remaining <= 0:
                break
            selected_by_layer[str(layer_id)].append(block_id)
            remaining -= 1
        if remaining <= 0:
            break
    return dict(selected_by_layer)


def ranked_receivers_by_layer(row: Mapping[str, Any]) -> Dict[str, List[int]]:
    by_layer: Dict[str, List[Tuple[float, int]]] = defaultdict(list)
    for layer in row.get("per_layer") or []:
        if layer.get("layer_group") != "late":
            continue
        layer_id = str(int(layer["layer_id"]))
        ids = [int(x) for x in layer.get("candidate_receiver_block_ids") or []]
        gains = [float(x) for x in layer.get("candidate_receiver_gain_proxy") or []]
        for pos, block_id in enumerate(ids):
            gain = gains[pos] if pos < len(gains) else 0.0
            by_layer[layer_id].append((gain, block_id))
    ranked: Dict[str, List[int]] = {}
    for layer_id, items in by_layer.items():
        items.sort(key=lambda item: (-item[0], item[1]))
        ranked[layer_id] = [block_id for _, block_id in items]
    return ranked


def build_policy_b_plan(
    output_dir: Path,
    replay_dir: Path,
    replay_decisions: Mapping[Key, Mapping[str, Any]],
    instrumentation: Mapping[Key, Mapping[str, Any]],
    replay_summary: Mapping[str, Any],
) -> Path:
    thresholds = json.loads((replay_dir / "lssr_controller_replay_summary.json").read_text(encoding="utf-8"))[
        "controller_constants"
    ]
    samples: Dict[str, JsonRow] = {}
    for key in sorted(FIXED_KEYS, key=lambda item: (item[0], item[1])):
        row = instrumentation[key]
        replay = replay_decisions[key]
        gate_triggered = bool(replay.get("lssr_triggered"))
        actions_by_layer: Dict[str, JsonRow] = {}
        middle_donors: List[int] = []
        layer_group_definition = row.get("layer_group_definition", {})
        for layer in row.get("per_layer") or []:
            layer_id = str(int(layer["layer_id"]))
            layer_group = str(layer.get("layer_group", ""))
            actions_by_layer[layer_id] = {
                "layer_group": layer_group,
                "remove_block_ids": [],
                "add_block_ids": [],
            }
            if layer_group == "middle":
                middle_donors.extend(int(x) for x in layer.get("candidate_donor_block_ids") or [])
        donor_count = len(middle_donors)
        receiver_by_layer = plan_receiver_actions(row, donor_count) if gate_triggered else {}
        receiver_ranked_by_layer = ranked_receivers_by_layer(row)
        if gate_triggered:
            for layer in row.get("per_layer") or []:
                layer_id = str(int(layer["layer_id"]))
                if layer.get("layer_group") == "middle":
                    actions_by_layer[layer_id]["remove_block_ids"] = list(middle_donors)
                if layer.get("layer_group") == "late":
                    actions_by_layer[layer_id]["add_block_ids"] = receiver_by_layer.get(layer_id, [])
                    actions_by_layer[layer_id]["add_target_count"] = len(receiver_by_layer.get(layer_id, []))
                    actions_by_layer[layer_id]["candidate_add_block_ids"] = receiver_ranked_by_layer.get(layer_id, [])
        total_remove = sum(len(action["remove_block_ids"]) for action in actions_by_layer.values())
        total_add = sum(len(action["add_block_ids"]) for action in actions_by_layer.values())
        if gate_triggered and total_remove != total_add:
            raise RuntimeError(f"LSSR budget plan mismatch for {key_label(key)}: remove={total_remove} add={total_add}")
        late_receiver_block_ids: List[int] = []
        for action in actions_by_layer.values():
            late_receiver_block_ids.extend(action["add_block_ids"])
        samples[key_label(key)] = {
            "task": key[0],
            "sample_index": key[1],
            "sample_id": row.get("sample_id"),
            "gate_triggered": gate_triggered,
            "gate_reason": "policy_b_middle_donor_late_receiver_pass" if gate_triggered else replay.get("reject_reasons", ""),
            "layer_group_definition": layer_group_definition,
            "middle_donor_capacity": replay.get("middle_donor_capacity"),
            "late_receiver_need": replay.get("late_receiver_need"),
            "late_query_or_evidence_candidate_count": replay.get("late_query_or_evidence_candidate_count"),
            "middle_donor_risk_mean": replay.get("middle_donor_risk_mean"),
            "late_receiver_gain_max": replay.get("late_receiver_gain_max"),
            "budget_preserved": bool(replay.get("budget_preserved")),
            "protected_violation": int(replay.get("protected_violation") or 0),
            "fallback_reason": replay.get("fallback_reason"),
            "selection_behavior_unchanged": bool(replay.get("selection_behavior_unchanged")),
            "middle_donor_block_ids": middle_donors,
            "late_receiver_block_ids": late_receiver_block_ids,
            "sample_budget_delta_blocks": total_add - total_remove,
            "actions_by_layer": actions_by_layer,
        }
    plan = {
        "policy_id": "policy_b",
        "policy_name": "Conservative Middle-Donor",
        "method": METHOD_LSSR,
        "thresholds": thresholds,
        "fixed_samples": {"hotpotqa": FIXED_HOTPOTQA, "musique": FIXED_MUSIQUE},
        "expected_trigger_count": 6,
        "expected_accepted_samples": sorted(EXPECTED_ACCEPTED),
        "offline_proxy_macro": replay_summary.get("macro"),
        "samples": samples,
    }
    plan_path = output_dir / "lssr_policy_b_action_plan.json"
    write_json(plan_path, plan)
    return plan_path


def task_scores_from_per_sample(scores: Mapping[Key, float]) -> Dict[str, float]:
    grouped: Dict[str, List[float]] = defaultdict(list)
    for (task, _), score in scores.items():
        grouped[task].append(float(score))
    return {task: sum(values) / len(values) for task, values in sorted(grouped.items())}


def macro(scores: Mapping[str, float]) -> float:
    return sum(scores.values()) / max(1, len(scores))


def longbench_metric_score(task: str, prediction: str, answers: Sequence[str], all_classes: Any) -> float:
    best = 0.0
    if prediction:
        pred = prediction
        if task in ["trec", "triviaqa", "samsum", "lsht"]:
            pred = pred.lstrip("\n").split("\n")[0]
        for answer in answers:
            best = max(best, float(dataset2metric[task](pred, answer, all_classes=all_classes)))
    return 100.0 * best


def read_predictions(run_dir: Path) -> Dict[Key, JsonRow]:
    rows: Dict[Key, JsonRow] = {}
    for task in ["hotpotqa", "musique"]:
        pred_path = run_dir / "longbench_predictions" / f"{task}.jsonl"
        for row in read_jsonl(pred_path):
            key = key_of({"task": task, "sample_index": row["sample_index"]})
            if key in FIXED_KEYS:
                rows[key] = row
    missing = sorted(FIXED_KEYS.difference(rows), key=lambda key: (key[0], key[1]))
    if missing:
        raise RuntimeError(f"LSSR predictions missing samples: {[key_label(key) for key in missing]}")
    for key, row in rows.items():
        if not str(row.get("pred", "")).strip():
            raise RuntimeError(f"empty prediction for {key_label(key)}")
    return rows


def read_generation_hashes(run_dir: Path) -> Dict[Key, str]:
    out: Dict[Key, str] = {}
    for row in read_jsonl(run_dir / "audit" / "generations.jsonl"):
        key = key_of(row)
        if key in FIXED_KEYS:
            out[key] = str(row.get("prompt_hash", ""))
    return out


def aggregate_gate_rows(run_dir: Path, plan: Mapping[str, Any]) -> Dict[Key, JsonRow]:
    rows_by_key: Dict[Key, List[JsonRow]] = defaultdict(list)
    for row in read_jsonl(run_dir / "audit" / "rescue_policy_audit.jsonl"):
        key = key_of(row)
        if key in FIXED_KEYS:
            rows_by_key[key].append(row)
    missing = sorted(FIXED_KEYS.difference(rows_by_key), key=lambda key: (key[0], key[1]))
    if missing:
        raise RuntimeError(f"gate audit missing samples: {[key_label(key) for key in missing]}")
    used_rows: Dict[Key, JsonRow] = {}
    used_path = run_dir / "audit" / "used_blocks.jsonl"
    for row in read_jsonl(used_path):
        key = key_of(row)
        if key in FIXED_KEYS and row.get("phase") == "prefill" and int(row.get("layer_id", -1)) == 27:
            used_rows[key] = row
    aggregated: Dict[Key, JsonRow] = {}
    for key, rows in rows_by_key.items():
        sample_plan = plan["samples"][key_label(key)]
        removed: List[int] = []
        rescued: List[int] = []
        protected_violation = False
        fallback_reasons: List[str] = []
        selected_block_count = None
        for row in rows:
            removed.extend(int(x) for x in row.get("removed_block_ids") or [])
            rescued.extend(int(x) for x in row.get("rescued_block_ids") or [])
            protected_violation = protected_violation or bool(row.get("protected_block_violation")) or bool(row.get("protected_violation"))
            if row.get("fallback_reason"):
                fallback_reasons.append(str(row.get("fallback_reason")))
            if int(row.get("layer_id", -1)) == 26:
                selected_block_count = int(row.get("final_block_count_used_by_attention") or row.get("final_block_count") or 0)
        used = used_rows.get(key, {})
        aggregated[key] = {
            "task": key[0],
            "sample_index": key[1],
            "sample_id": sample_plan.get("sample_id"),
            "method": METHOD_LSSR,
            "gate_triggered": bool(sample_plan.get("gate_triggered")),
            "gate_reason": sample_plan.get("gate_reason"),
            "layer_group_definition": sample_plan.get("layer_group_definition"),
            "donor_layer_group": "middle" if removed else "",
            "receiver_layer_group": "late" if rescued else "",
            "middle_donor_block_ids": removed if removed else sample_plan.get("middle_donor_block_ids", []),
            "late_receiver_block_ids": rescued if rescued else sample_plan.get("late_receiver_block_ids", []),
            "removed_block_ids": removed,
            "rescued_block_ids": rescued,
            "budget_preserved": len(removed) == len(rescued),
            "selected_block_count": selected_block_count,
            "used_blocks": used.get("used_block_ids", used.get("used_block_ids_first100", [])),
            "used_tokens": used.get("used_tokens", used.get("kv_abs_indices_first100", [])),
            "protected_violation": protected_violation,
            "fallback_reason": None if not fallback_reasons else ";".join(fallback_reasons),
            "audit_layer_rows": len(rows),
        }
    return aggregated


def recompute_scores(run_dir: Path, predictions: Mapping[Key, Mapping[str, Any]]) -> Tuple[Dict[str, float], Dict[Key, float]]:
    per_sample_scores: Dict[Key, float] = {}
    task_scores: Dict[str, float] = {}
    for task in ["hotpotqa", "musique"]:
        task_rows = [predictions[(task, index)] for index in (FIXED_HOTPOTQA if task == "hotpotqa" else FIXED_MUSIQUE)]
        preds = [str(row["pred"]) for row in task_rows]
        answers = [row["answers"] for row in task_rows]
        lengths = [int(row["length"]) for row in task_rows]
        all_classes = []
        for row in task_rows:
            if row.get("all_classes"):
                all_classes = row.get("all_classes", all_classes)
        score = scorer(task, preds, answers, lengths, all_classes)
        task_scores[task] = float(score)
        for row in task_rows:
            key = (task, int(row["sample_index"]))
            per_sample_scores[key] = longbench_metric_score(task, str(row["pred"]), row["answers"], row.get("all_classes"))

    runner_scores = {}
    scores_csv = run_dir / "longbench_scores.csv"
    with scores_csv.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("dataset") in {"hotpotqa", "musique"}:
                runner_scores[row["dataset"]] = float(row["score"])
    mismatches = {
        task: (task_scores[task], runner_scores.get(task))
        for task in task_scores
        if abs(task_scores[task] - float(runner_scores.get(task, float("nan")))) > 1e-9
    }
    if mismatches:
        raise RuntimeError(f"independent LongBench scorer mismatch: {mismatches}")
    return task_scores, per_sample_scores


def run_generation(args: argparse.Namespace, output_dir: Path, plan_path: Path) -> Path:
    run_dir = output_dir / "run_lssr_policy_b_middle_donor_late_receiver"
    run_dir.mkdir(parents=True, exist_ok=False)
    audit_dir = run_dir / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    python_bin = REPO_ROOT / ".venv" / "bin" / "python"
    if not python_bin.exists():
        python_bin = Path(sys.executable)
    cmd = [
        str(python_bin),
        str(REPO_ROOT / "experiments" / "run_qwen25_sliminfer_variants.py"),
        "--model_path", str(args.model_path),
        "--tokenizer_path", str(args.model_path),
        "--model_family", "qwen2",
        "--dtype", "fp16",
        "--device", "cuda",
        "--backend", "flash_attention_2",
        "--eval", "longbench",
        "--longbench_dir", str(args.longbench_dir),
        "--longbench_tasks", "hotpotqa,musique",
        "--longbench_sample_indices_by_task",
        "hotpotqa:" + ",".join(str(x) for x in FIXED_HOTPOTQA) + ";musique:" + ",".join(str(x) for x in FIXED_MUSIQUE),
        "--longbench_subset_name", "lssr_policy_b_quality_smoke",
        "--generation_max_new_tokens", "32",
        "--max_model_len", "32768",
        "--method", "sliminfer_variant",
        "--scoring", "delta_o",
        "--fairness_base_scoring", "delta_o",
        "--decode_policy", "reuse_prefill",
        "--rescue_policy", METHOD_LSSR,
        "--rescue_score", "delta_o",
        "--rescue_ratio", "0.05",
        "--lssr_policy_b_plan_path", str(plan_path),
        "--gate_mode", "shadow",
        "--gate_policy", "conservative_v1",
        "--gate_default_policy", "original",
        "--gate_rescue_margin_min", "0.25",
        "--gate_coverage_gap_min", "0.0",
        "--gate_deletion_risk_max", "0.2",
        "--gate_feature_version", "selector_native_v2",
        "--gate_sentence_jaccard_novel_max", "0.35",
        "--prune_layers", "9,18,26",
        "--retain_tokens", "12288,6144,4096",
        "--block_size", "64",
        "--token_unit_size", "8",
        "--local_query_window", "4",
        "--swap_threshold", "0.9",
        "--temperature", "0",
        "--top_p", "1.0",
        "--seed", "0",
        "--fd_alpha_o", "1.0",
        "--fd_beta_h", "0.0",
        "--fd_gamma_s", "0.0",
        "--alpha_diff", "0.5",
        "--lambda_red", "0.25",
        "--eta_rescue", "0.25",
        "--eta_active", "0.25",
        "--candidate_pool_ratio", "0.3",
        "--sampling_temperature", "1.0",
        "--swap_tau", "0.0",
        "--debug_generation_jsonl", str(audit_dir / "generations.jsonl"),
        "--debug_used_block_audit_path", str(audit_dir / "used_blocks.jsonl"),
        "--debug_used_block_include_full", "true",
        "--debug_used_block_full_layers", "10,19,27",
        "--rescue_audit_path", str(audit_dir / "rescue_policy_audit.jsonl"),
        "--output_dir", str(run_dir),
        "--min_free_gpu_gb", "0",
    ]
    write_json(output_dir / "generation_command.json", {"command": cmd})
    with (output_dir / "generation_stdout.log").open("w", encoding="utf-8") as stdout, (
        output_dir / "generation_stderr.log"
    ).open("w", encoding="utf-8") as stderr:
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), text=True, stdout=stdout, stderr=stderr)
    stderr_text = (output_dir / "generation_stderr.log").read_text(encoding="utf-8", errors="ignore")
    stdout_text = (output_dir / "generation_stdout.log").read_text(encoding="utf-8", errors="ignore")
    if "CUDA out of memory" in stderr_text or "CUDA out of memory" in stdout_text:
        raise RuntimeError("CUDA OOM detected during LSSR Policy B smoke")
    if proc.returncode != 0:
        raise RuntimeError(f"LSSR generation failed with return code {proc.returncode}")
    return run_dir


def write_report(output_dir: Path, summary: Mapping[str, Any]) -> None:
    lines = [
        "# LSSR Policy B Tiny Quality Smoke",
        "",
        f"- Result directory: `{output_dir.resolve()}`",
        "- Actual runtime method: `lssr_policy_b_middle_donor_late_receiver`",
        "- Baseline Original: historical 200-sample diagnostic rows for the same fixed 20 samples",
        "- Policy A/C run: **No**",
        "- Full LongBench run: **No**",
        "- Gold/final score/final prediction used for gate: **No**",
        "",
        "## Gate",
        "",
        f"- Actual trigger count: {summary['actual_trigger_count']}/20",
        f"- Accepted samples: `{json.dumps(summary['accepted_samples'], ensure_ascii=False)}`",
        f"- Matches offline replay: {summary['gate_matches_offline_replay']}",
        f"- HotpotQA top negative rejected: {summary['hotpotqa_top_negative_rejected']}/{summary['hotpotqa_top_negative_total']}",
        f"- MuSiQue positive retained: {summary['musique_positive_retained']}/{summary['musique_positive_total']}",
        "",
        "## Scores",
        "",
        f"- Task scores: `{json.dumps(summary['task_scores'], ensure_ascii=False)}`",
        f"- 2-task macro: {summary['macro']:.4f}",
        f"- LSSR vs Original: {summary['delta_vs_original']:+.4f}",
        f"- LSSR vs historical DRM-Swap: {summary['delta_vs_historical_drm_swap']:+.4f}",
        f"- LSSR vs offline proxy replay: {summary['delta_vs_offline_proxy']:+.4f}",
        "",
        "## Safety",
        "",
        f"- over_budget: {summary['over_budget_count']}",
        f"- protected violation: {summary['protected_violation_count']}",
        f"- fallback: {summary['fallback_count']}",
        f"- OOM: {summary['oom_count']}",
        f"- Independent scorer zero-diff: {summary['independent_scorer_zero_diff']}",
        "",
        "## GO / NO-GO",
        "",
        f"- Decision: **{summary['go_no_go']['decision']}**",
        f"- Recommend 64-sample diagnostic: {summary['recommend_64_sample_diagnostic']}",
        f"- Recommend stop LSSR: {summary['recommend_stop_lssr']}",
    ]
    for reason in summary["go_no_go"]["reasons"]:
        lines.append(f"- Reason: `{reason}`")
    lines.append("")
    lines.append("This is a tiny quality smoke, not a full LongBench result and not a statistical claim.")
    (output_dir / "quality_smoke_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir) if args.output_dir else Path("results") / f"lssr_policy_b_quality_smoke_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=False)
    gpu_preflight = check_gpu(args.min_free_gpu_gb)
    replay_decisions, replay_policy_b = load_replay_policy_b(Path(args.replay_dir))
    instrumentation = load_instrumentation(Path(args.instrumentation_dir))
    historical = load_historical(Path(args.historical_dir))
    historical_prompt_hashes = collect_prompt_hashes(Path(args.historical_dir))
    plan_path = build_policy_b_plan(output_dir, Path(args.replay_dir), replay_decisions, instrumentation, replay_policy_b)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    run_dir = run_generation(args, output_dir, plan_path)

    predictions = read_predictions(run_dir)
    task_scores, per_sample_scores = recompute_scores(run_dir, predictions)
    generation_prompt_hashes = read_generation_hashes(run_dir)
    gate_rows = aggregate_gate_rows(run_dir, plan)

    accepted = {key_label(key) for key, row in gate_rows.items() if bool(row["gate_triggered"])}
    if accepted != EXPECTED_ACCEPTED:
        raise RuntimeError(f"runtime accepted samples mismatch: {sorted(accepted)}")
    if len(accepted) != 6:
        raise RuntimeError(f"runtime trigger count mismatch: {len(accepted)}")
    for key in FIXED_KEYS:
        expected_id = plan["samples"][key_label(key)]["sample_id"]
        if str(predictions[key].get("sample_id")) != str(expected_id):
            raise RuntimeError(f"prediction sample_id mismatch for {key_label(key)}")
        if generation_prompt_hashes.get(key) != historical_prompt_hashes.get(key):
            raise RuntimeError(f"prompt hash mismatch for {key_label(key)}")

    per_sample_predictions = []
    per_sample_gate = []
    generation_config_hash = json.loads((Path(args.historical_dir) / "config.json").read_text(encoding="utf-8"))[
        "protocol"
    ]["generation_config_hash"]
    model_revision = json.loads((Path(args.historical_dir) / "config.json").read_text(encoding="utf-8"))[
        "protocol"
    ]["model_revision_or_unknown"]
    for key in sorted(FIXED_KEYS, key=lambda item: (item[0], item[1])):
        pred = predictions[key]
        gate = gate_rows[key]
        gate.update({
            "prediction": pred["pred"],
            "gold": pred["answers"],
            "score": per_sample_scores[key],
            "prompt_hash": generation_prompt_hashes[key],
            "generation_config_hash": generation_config_hash,
            "model_revision": model_revision,
        })
        per_sample_gate.append(gate)
        per_sample_predictions.append({
            "task": key[0],
            "sample_index": key[1],
            "sample_id": pred["sample_id"],
            "method": METHOD_LSSR,
            "gate_triggered": gate["gate_triggered"],
            "prediction": pred["pred"],
            "gold": pred["answers"],
            "score": per_sample_scores[key],
            "length": pred["length"],
            "prompt_hash": generation_prompt_hashes[key],
            "generation_config_hash": generation_config_hash,
            "model_revision": model_revision,
        })
    write_jsonl(output_dir / "per_sample_predictions.jsonl", per_sample_predictions)
    write_jsonl(output_dir / "per_sample_gate_audit.jsonl", per_sample_gate)

    original_task = task_scores_from_per_sample({
        key: finite_score(historical[METHOD_ORIGINAL][key]["prediction_score"])
        for key in FIXED_KEYS
    })
    drm_task = task_scores_from_per_sample({
        key: finite_score(historical[METHOD_DRM_SWAP][key]["prediction_score"])
        for key in FIXED_KEYS
    })
    lssr_macro = macro(task_scores)
    original_macro = macro(original_task)
    drm_macro = macro(drm_task)
    offline_proxy_macro = float(replay_policy_b["macro"])
    hotpot_rejected = sum(1 for index in HOTPOT_TOP_NEGATIVE if f"hotpotqa:{index}" not in accepted)
    musique_retained = sum(1 for index in MUSIQUE_TOP_POSITIVE if f"musique:{index}" in accepted)
    protected_violations = sum(bool(row["protected_violation"]) for row in per_sample_gate)
    fallback_count = sum(row["fallback_reason"] not in (None, "", "None") for row in per_sample_gate)
    over_budget_count = sum(not bool(row["budget_preserved"]) for row in per_sample_gate)

    reasons: List[str] = []
    if accepted != EXPECTED_ACCEPTED:
        reasons.append("runtime_gate_decision_mismatch")
    if len(accepted) != 6:
        reasons.append("trigger_count_not_6")
    if over_budget_count:
        reasons.append("budget_violation")
    if protected_violations:
        reasons.append("protected_violation")
    if hotpot_rejected != len(HOTPOT_TOP_NEGATIVE):
        reasons.append("hotpotqa_top_negative_not_all_rejected")
    if musique_retained < 3:
        reasons.append("musique_positive_retained_less_than_3")
    if lssr_macro < original_macro - 1e-9:
        reasons.append("true_generation_macro_below_original")
    if lssr_macro < offline_proxy_macro - OFFLINE_PROXY_DROP_TOLERANCE:
        reasons.append("true_generation_obviously_below_offline_proxy")
    if fallback_count:
        reasons.append("fallback_used")
    go_no_go = {"decision": "GO" if not reasons else "NO-GO", "reasons": reasons}
    summary = {
        "result_dir": str(output_dir.resolve()),
        "run_dir": str(run_dir.resolve()),
        "actual_methods_run": [METHOD_LSSR],
        "baseline_original_source": str(Path(args.historical_dir) / "audit" / "per_sample_audit.jsonl"),
        "fixed_samples": {"hotpotqa": FIXED_HOTPOTQA, "musique": FIXED_MUSIQUE},
        "actual_trigger_count": len(accepted),
        "accepted_samples": sorted(accepted),
        "gate_matches_offline_replay": accepted == EXPECTED_ACCEPTED,
        "hotpotqa_top_negative_rejected": hotpot_rejected,
        "hotpotqa_top_negative_total": len(HOTPOT_TOP_NEGATIVE),
        "musique_positive_retained": musique_retained,
        "musique_positive_total": len(MUSIQUE_TOP_POSITIVE),
        "task_scores": task_scores,
        "macro": lssr_macro,
        "original_task_scores": original_task,
        "original_macro": original_macro,
        "historical_drm_swap_task_scores": drm_task,
        "historical_drm_swap_macro": drm_macro,
        "offline_proxy_macro": offline_proxy_macro,
        "delta_vs_original": lssr_macro - original_macro,
        "delta_vs_historical_drm_swap": lssr_macro - drm_macro,
        "delta_vs_offline_proxy": lssr_macro - offline_proxy_macro,
        "over_budget_count": over_budget_count,
        "protected_violation_count": protected_violations,
        "fallback_count": fallback_count,
        "oom_count": 0,
        "independent_scorer_zero_diff": True,
        "prompt_hashes_match_historical": True,
        "generation_config_hash": generation_config_hash,
        "model_revision": model_revision,
        "scorer_hash": hash_file(REPO_ROOT / "longbench_eval.py"),
        "gpu_preflight": gpu_preflight,
        "go_no_go": go_no_go,
        "recommend_64_sample_diagnostic": go_no_go["decision"] == "GO",
        "recommend_stop_lssr": go_no_go["decision"] != "GO",
        "notes": [
            "No Policy A/C, Random-Swap, DRM-Add, Larger, verifier, BridgePack, or QFocus run.",
            "Gate uses only Policy B instrumentation features and fixed replay thresholds.",
        ],
    }
    write_json(output_dir / "gate_decision_summary.json", summary)
    write_csv(
        output_dir / "method_task_scores.csv",
        [
            {"method": METHOD_LSSR, **task_scores, "macro": lssr_macro},
            {"method": METHOD_ORIGINAL, **original_task, "macro": original_macro},
            {"method": "historical_drm_swap", **drm_task, "macro": drm_macro},
            {"method": "offline_proxy_replay_policy_b", **replay_policy_b["task_scores"], "macro": offline_proxy_macro},
        ],
        ["method", "hotpotqa", "musique", "macro"],
    )
    write_report(output_dir, summary)
    print(json.dumps({
        "status": "ok",
        "output_dir": str(output_dir.resolve()),
        "go_no_go": go_no_go,
        "macro": lssr_macro,
        "accepted_samples": sorted(accepted),
    }, ensure_ascii=False, indent=2))
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instrumentation_dir", default=str(DEFAULT_INSTRUMENTATION_DIR))
    parser.add_argument("--layer_audit_dir", default=str(DEFAULT_LAYER_AUDIT_DIR))
    parser.add_argument("--replay_dir", default=str(DEFAULT_REPLAY_DIR))
    parser.add_argument("--historical_dir", default=str(DEFAULT_HISTORICAL_DIR))
    parser.add_argument("--model_path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--longbench_dir", default=str(DEFAULT_LONGBENCH_DIR))
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--min_free_gpu_gb", type=float, default=20.0)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
