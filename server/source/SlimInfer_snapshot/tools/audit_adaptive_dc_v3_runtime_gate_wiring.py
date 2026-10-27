#!/usr/bin/env python3
"""Offline audit of fixed Adaptive DC v3 runtime gate wiring on held-out64.

Does NOT run generation or scorer. Replays fixed gate logic on held-out64
selection audit artifacts and compares against prior offline recompute.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.cache_utils import _adaptive_dc_v3_compute_gate

DEFAULT_HELDOUT_DIR = Path("results/adaptive_dc_v3_heldout64_diagnostic_20260708_073628")
DEFAULT_FEEDBACK_DIR = Path("results/adaptive_dc_feedback_combined_analysis_20260708_182536")
DEFAULT_TINY_DIR = Path("results/adaptive_dc_v3_tiny_quality_smoke_20260708_060002")
DEFAULT_LONGBENCH_DIR = Path("/home/yu-bai/Desktop/jiaqizhang/HeadKV/data/LongBench")
DEFAULT_MODEL_PATH = Path(
    "/home/yu-bai/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/"
    "snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
)
GATE_LAYER = 26
TASKS = ["hotpotqa", "musique"]
TINY_EXCLUDE = {
    "hotpotqa": {0, 1, 2, 3, 4, 5, 41, 56, 57, 117},
    "musique": {0, 1, 2, 3, 4, 20, 50, 54, 56, 134},
}


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def to_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        out = float(value)
        return out if math.isfinite(out) else None
    except (TypeError, ValueError):
        return None


def heldout_indices(task: str, count: int = 64) -> List[int]:
    selected: List[int] = []
    for idx in range(200):
        if idx in TINY_EXCLUDE[task]:
            continue
        selected.append(idx)
        if len(selected) >= count:
            break
    return selected


def load_gate_config(heldout_dir: Path) -> Dict[str, Any]:
    cfg = json.loads((heldout_dir / "run_config.json").read_text(encoding="utf-8"))
    frozen = cfg.get("frozen_gate_config") or {}
    if frozen.get("evidence_dispersion_threshold") is None:
        raise RuntimeError("BUG_STOP: frozen threshold missing")
    return {
        "evidence_dispersion_threshold": float(frozen["evidence_dispersion_threshold"]),
        "norm_min": frozen.get("norm_min") or {},
        "norm_max": frozen.get("norm_max") or {},
        "gate_apply_layer": int(frozen.get("gate_apply_layer", GATE_LAYER)),
        "top_pruned_m": int(frozen.get("top_pruned_m", 16)),
    }


def load_tokenizer_and_prompts(model_path: Path) -> Tuple[Any, Dict[str, str]]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    dataset2prompt = json.loads((REPO_ROOT / "configs" / "dataset2prompt.json").read_text(encoding="utf-8"))
    return tokenizer, dataset2prompt


def load_samples(longbench_dir: Path, task: str) -> List[Dict[str, Any]]:
    return [
        json.loads(line)
        for line in (longbench_dir / f"{task}.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_chat_prompt(tokenizer: Any, prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )


def build_prompt_blocks(
    tokenizer: Any, prompt_format: str, sample: Dict[str, Any], dataset: str,
    max_model_len: int, block_size: int,
) -> List[str]:
    prompt = prompt_format.format(**sample)
    input_ids = tokenizer(prompt, truncation=False, return_tensors=None).input_ids
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    if len(input_ids) > max_model_len:
        half = max_model_len // 2
        prompt = tokenizer.decode(input_ids[:half], skip_special_tokens=True) + tokenizer.decode(
            input_ids[-half:], skip_special_tokens=True
        )
    if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
        prompt = build_chat_prompt(tokenizer, prompt)
    chat_ids = tokenizer(prompt, truncation=False, return_tensors=None).input_ids
    if chat_ids and isinstance(chat_ids[0], list):
        chat_ids = chat_ids[0]
    blocks = []
    for start in range(0, len(chat_ids), block_size):
        blocks.append(
            tokenizer.decode(chat_ids[start : start + block_size], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        )
    return blocks


def compute_gate_row(
    gate_config: Dict[str, Any],
    task: str,
    sample_index: int,
    sample_id: str,
    query_text: str,
    block_texts: List[str],
    original_selected: Sequence[int],
    dc_selected: Sequence[int],
) -> Dict[str, Any]:
    cfg = {
        "adaptive_dc_v3_gate": gate_config,
        "block_texts": block_texts,
        "query_text": query_text,
        "audit_context": {"task": task, "sample_index": sample_index, "sample_id": sample_id},
    }
    gate = _adaptive_dc_v3_compute_gate(
        cfg=cfg,
        original_selected_block_ids=set(int(x) for x in original_selected),
        dc_selected_block_ids=set(int(x) for x in dc_selected),
        num_blocks=max(len(block_texts), 1),
    )
    threshold = float(gate_config["evidence_dispersion_threshold"])
    eds = to_float(gate.get("evidence_dispersion_score"))
    ecg = to_float(gate.get("entity_coverage_gap"))
    pqo = to_float(gate.get("pruned_query_overlap_max"))
    sqm = to_float(gate.get("selected_query_overlap_mean"))
    jac = to_float(gate.get("selected_jaccard_vs_original"))
    preds = {
        "evidence_dispersion_score_ge_threshold": eds is not None and eds >= threshold,
        "entity_coverage_gap_ge_zero": ecg is not None and ecg >= 0.0,
        "pruned_query_overlap_max_gt_selected_mean": pqo is not None and sqm is not None and pqo > sqm,
        "selected_jaccard_vs_original_ge_030": jac is not None and jac >= 0.30,
    }
    return {
        **gate,
        "predicates": preds,
        "all_predicates_pass": all(preds.values()),
        "chosen_selector": "dc" if gate.get("triggered") else "original",
    }


def predicate_summary(rows: Sequence[Dict[str, Any]], prefix: str) -> List[Dict[str, Any]]:
    names = [
        "evidence_dispersion_score_ge_threshold",
        "entity_coverage_gap_ge_zero",
        "pruned_query_overlap_max_gt_selected_mean",
        "selected_jaccard_vs_original_ge_030",
    ]
    total = len(rows)
    out = []
    for name in names:
        passes = sum(1 for row in rows if bool((row.get("predicates") or {}).get(name)))
        out.append({
            "predicate": f"{prefix}{name}",
            "pass_count": passes,
            "fail_count": total - passes,
            "pass_rate": passes / total if total else 0.0,
        })
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heldout_dir", type=Path, default=DEFAULT_HELDOUT_DIR)
    parser.add_argument("--feedback_dir", type=Path, default=DEFAULT_FEEDBACK_DIR)
    parser.add_argument("--longbench_dir", type=Path, default=DEFAULT_LONGBENCH_DIR)
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output_root", type=Path, default=Path("results"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    heldout_dir = (REPO_ROOT / args.heldout_dir).resolve()
    feedback_dir = (REPO_ROOT / args.feedback_dir).resolve()
    output_dir = (REPO_ROOT / args.output_root / f"adaptive_dc_v3_runtime_gate_fix_audit_{datetime.now():%Y%m%d_%H%M%S}").resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    gate_config = load_gate_config(heldout_dir)
    old_runtime_audit = read_csv(heldout_dir / "adaptive_gate_audit.csv")
    prior_offline = {}
    prior_path = feedback_dir / "heldout_offline_gate_recompute.csv"
    if prior_path.exists():
        for row in read_csv(prior_path):
            prior_offline[(row["task"], int(row["sample_index"]))] = row

    selection_rows = read_jsonl(heldout_dir / "run_adaptive_dc_sliminfer_v3" / "audit" / "dc_runtime_selection_audit.jsonl")
    selection_by_key: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for row in selection_rows:
        if str(row.get("phase")) != "prefill" or int(row.get("layer_id", -1)) != GATE_LAYER:
            continue
        key = (str(row.get("task")), int(row.get("sample_index")))
        if key not in selection_by_key:
            selection_by_key[key] = row

    tokenizer, dataset2prompt = load_tokenizer_and_prompts(args.model_path)
    samples_cache = {task: load_samples(args.longbench_dir, task) for task in TASKS}
    keys = [(task, idx) for task in TASKS for idx in heldout_indices(task)]

    fixed_rows: List[Dict[str, Any]] = []
    diff_rows: List[Dict[str, Any]] = []

    for task, sample_index in keys:
        sel = selection_by_key.get((task, sample_index))
        if sel is None:
            raise RuntimeError(f"missing selection audit for {task}:{sample_index}")
        sample = samples_cache[task][sample_index]
        query_text = str(sample.get("input", sample.get("question", sample.get("query", ""))))
        block_texts = build_prompt_blocks(
            tokenizer, dataset2prompt[task], sample, task, 32768, 64
        )
        orig_sel = [int(x) for x in sel.get("original_selected_block_ids", [])]
        dc_sel = [int(x) for x in sel.get("dc_selected_block_ids", [])]
        fixed = compute_gate_row(
            gate_config, task, sample_index, str(sel.get("sample_id", "")),
            query_text, block_texts, orig_sel, dc_sel,
        )
        old = next((r for r in old_runtime_audit if r["task"] == task and int(r["sample_index"]) == sample_index), {})
        prior = prior_offline.get((task, sample_index), {})

        fixed_rows.append({
            "task": task,
            "sample_index": sample_index,
            "sample_id": sel.get("sample_id", ""),
            "adaptive_gate_triggered": fixed.get("triggered"),
            "adaptive_gate_reason": fixed.get("reason"),
            "evidence_dispersion_score": fixed.get("evidence_dispersion_score"),
            "evidence_dispersion_threshold": fixed.get("evidence_dispersion_threshold"),
            "entity_coverage_gap": fixed.get("entity_coverage_gap"),
            "pruned_query_overlap_max": fixed.get("pruned_query_overlap_max"),
            "selected_query_overlap_mean": fixed.get("selected_query_overlap_mean"),
            "selected_jaccard_vs_original": fixed.get("selected_jaccard_vs_original"),
            "chosen_selector": fixed.get("chosen_selector"),
            "feature_source": fixed.get("feature_source"),
            "legacy_runtime_recorded_eds": old.get("evidence_dispersion_score", ""),
            "legacy_runtime_triggered": old.get("adaptive_gate_triggered", ""),
            "predicates": fixed.get("predicates"),
        })

        prior_trigger = prior.get("offline_frozen_triggered", "")
        prior_eds = prior.get("offline_frozen_eds", "")
        fixed_trigger = bool(fixed.get("triggered"))
        fixed_eds = to_float(fixed.get("evidence_dispersion_score"))
        mismatch = False
        if prior:
            mismatch = (
                str(prior_trigger).lower() != str(fixed_trigger).lower()
                or (to_float(prior_eds) is not None and fixed_eds is not None and abs(to_float(prior_eds) - fixed_eds) > 1e-6)
            )
        diff_rows.append({
            "task": task,
            "sample_index": sample_index,
            "fixed_triggered": fixed_trigger,
            "prior_offline_triggered": prior_trigger,
            "fixed_eds": fixed_eds,
            "prior_offline_eds": prior_eds,
            "legacy_runtime_triggered": old.get("adaptive_gate_triggered", ""),
            "legacy_runtime_eds": old.get("evidence_dispersion_score", ""),
            "mismatch_fixed_vs_prior_offline": mismatch,
        })

    feature_coverage = sum(1 for row in fixed_rows if to_float(row.get("evidence_dispersion_score")) is not None)
    fixed_trigger_count = sum(1 for row in fixed_rows if bool(row.get("adaptive_gate_triggered")))
    prior_trigger_count = sum(
        1 for row in diff_rows
        if str(row.get("prior_offline_triggered", "")).lower() in ("true", "1")
    ) if prior_offline else fixed_trigger_count
    mismatch_count = sum(1 for row in diff_rows if bool(row.get("mismatch_fixed_vs_prior_offline")))
    legacy_empty = sum(1 for row in old_runtime_audit if not str(row.get("evidence_dispersion_score", "")).strip())

    predicate_rows = predicate_summary(fixed_rows, "fixed_")

    if feature_coverage < 128 or mismatch_count > 0:
        decision = {
            "decision": "BUG_STOP_FIX_REQUIRED",
            "reasons": [
                f"feature_coverage={feature_coverage}/128",
                f"mismatch_fixed_vs_prior_offline={mismatch_count}",
            ],
        }
    elif fixed_trigger_count == 128:
        decision = {
            "decision": "STATIC_GATE_RUNTIME_NO_GO",
            "reasons": [f"fixed_trigger_count={fixed_trigger_count}/128 (all_trigger)"],
        }
    else:
        decision = {
            "decision": "BUG_FIXED_GATE_VALIDATION_PASS",
            "reasons": [
                f"feature_coverage={feature_coverage}/128",
                f"fixed_trigger_count={fixed_trigger_count}/128",
                f"mismatch_fixed_vs_prior_offline={mismatch_count}",
                f"legacy_runtime_empty_features={legacy_empty}/128",
            ],
        }

    audit_fields = [
        "task", "sample_index", "sample_id",
        "adaptive_gate_triggered", "adaptive_gate_reason",
        "evidence_dispersion_score", "evidence_dispersion_threshold",
        "entity_coverage_gap", "pruned_query_overlap_max",
        "selected_query_overlap_mean", "selected_jaccard_vs_original",
        "chosen_selector", "feature_source",
        "legacy_runtime_recorded_eds", "legacy_runtime_triggered",
    ]
    write_csv(output_dir / "runtime_gate_audit.csv", fixed_rows, audit_fields)
    write_csv(
        output_dir / "runtime_vs_offline_gate_diff.csv",
        diff_rows,
        [
            "task", "sample_index", "fixed_triggered", "prior_offline_triggered",
            "fixed_eds", "prior_offline_eds", "legacy_runtime_triggered",
            "legacy_runtime_eds", "mismatch_fixed_vs_prior_offline",
        ],
    )
    write_csv(output_dir / "gate_predicate_pass_fail.csv", predicate_rows,
              ["predicate", "pass_count", "fail_count", "pass_rate"])

    report = [
        "# Adaptive DC v3 Runtime Gate Fix Audit",
        "",
        f"- Output: `{output_dir}`",
        f"- Held-out dir: `{heldout_dir}`",
        "",
        "## Gate Wiring Validation (fixed logic, offline replay)",
        "",
        f"- Fixed runtime gate feature coverage: **{feature_coverage}/128**",
        f"- Legacy held-out runtime recorded features: **{128 - legacy_empty}/128** (expected 0)",
        f"- Fixed trigger count: **{fixed_trigger_count}/128**",
        f"- Prior offline recompute trigger count: **{prior_trigger_count}/128**",
        f"- Fixed vs prior offline mismatch: **{mismatch_count}**",
        f"- Frozen threshold: {gate_config['evidence_dispersion_threshold']}",
        f"- Threshold type: normalized evidence_dispersion_score",
        "",
        "## Predicate Pass/Fail (fixed logic)",
        "",
        "| predicate | pass | fail | pass_rate |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in predicate_rows:
        report.append(
            f"| {row['predicate'].replace('fixed_', '')} | {row['pass_count']} | {row['fail_count']} | {row['pass_rate']:.3f} |"
        )
    report.extend([
        "",
        "## Decision",
        "",
        f"- **{decision['decision']}**",
    ])
    for reason in decision.get("reasons", []):
        report.append(f"- `{reason}`")
    report.append("")
    (output_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    decision_lines = [
        "# Adaptive DC v3 Runtime Gate Fix Audit Decision",
        "",
        f"- Decision: **{decision['decision']}**",
    ]
    for reason in decision.get("reasons", []):
        decision_lines.append(f"- Reason: `{reason}`")
    decision_lines.append(f"\n- Output directory: `{output_dir}`\n")
    (output_dir / "decision.md").write_text("\n".join(decision_lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "status": "ok",
        "output_dir": str(output_dir),
        "decision": decision,
        "feature_coverage": feature_coverage,
        "fixed_trigger_count": fixed_trigger_count,
        "mismatch_count": mismatch_count,
    }, indent=2))


if __name__ == "__main__":
    main()
