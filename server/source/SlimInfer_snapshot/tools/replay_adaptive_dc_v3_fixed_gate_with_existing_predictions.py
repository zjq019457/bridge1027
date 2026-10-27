#!/usr/bin/env python3
"""Replay fixed Adaptive DC v3 gate on held-out64 using existing original/DC predictions.

No generation, no scorer. Combines per-sample scores by fixed gate trigger:
  gate=false -> original
  gate=true  -> dc
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.cache_utils import _adaptive_dc_v3_compute_gate

DEFAULT_HELDOUT_DIR = Path("results/adaptive_dc_v3_heldout64_diagnostic_20260708_073628")
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
ORACLE_GAIN_THRESHOLD = 2.0


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


def to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return False
    return str(value).strip().lower() in ("1", "true", "yes", "y")


def mean(values: Sequence[float]) -> float:
    items = [float(v) for v in values if math.isfinite(float(v))]
    return float(sum(items) / len(items)) if items else 0.0


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
    frozen = (json.loads((heldout_dir / "run_config.json").read_text(encoding="utf-8"))).get("frozen_gate_config") or {}
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
    return [
        tokenizer.decode(chat_ids[start : start + block_size], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        for start in range(0, len(chat_ids), block_size)
    ]


def compute_fixed_gate(
    gate_config: Dict[str, Any], task: str, sample_index: int, sample_id: str,
    query_text: str, block_texts: List[str],
    original_selected: Sequence[int], dc_selected: Sequence[int],
) -> Dict[str, Any]:
    cfg = {
        "adaptive_dc_v3_gate": gate_config,
        "block_texts": block_texts,
        "query_text": query_text,
        "audit_context": {"task": task, "sample_index": sample_index, "sample_id": sample_id},
    }
    return _adaptive_dc_v3_compute_gate(
        cfg=cfg,
        original_selected_block_ids=set(int(x) for x in original_selected),
        dc_selected_block_ids=set(int(x) for x in dc_selected),
        num_blocks=max(len(block_texts), 1),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heldout_dir", type=Path, default=DEFAULT_HELDOUT_DIR)
    parser.add_argument("--audit_dir", type=Path, default=None,
                        help="Optional gate fix audit dir; if omitted, recompute gate inline")
    parser.add_argument("--longbench_dir", type=Path, default=DEFAULT_LONGBENCH_DIR)
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output_dir", type=Path, default=None,
                        help="Output directory; defaults to audit_dir or new timestamp dir")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    heldout_dir = (REPO_ROOT / args.heldout_dir).resolve()
    if args.output_dir is not None:
        output_dir = (REPO_ROOT / args.output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
    elif args.audit_dir is not None:
        output_dir = (REPO_ROOT / args.audit_dir).resolve()
    else:
        output_dir = (
            REPO_ROOT / "results" / f"adaptive_dc_v3_runtime_gate_fix_audit_{datetime.now():%Y%m%d_%H%M%S}"
        ).resolve()
        output_dir.mkdir(parents=True, exist_ok=False)

    quality = read_csv(heldout_dir / "per_sample_quality.csv")
    method_summary = {row["task"]: row for row in read_csv(heldout_dir / "method_summary.csv")}
    gate_config = load_gate_config(heldout_dir)

    fixed_gate_by_key: Dict[Tuple[str, int], Dict[str, Any]] = {}
    if args.audit_dir is not None:
        audit_dir = (REPO_ROOT / args.audit_dir).resolve()
        for row in read_csv(audit_dir / "runtime_gate_audit.csv"):
            fixed_gate_by_key[(row["task"], int(row["sample_index"]))] = row
    else:
        selection_rows = read_jsonl(
            heldout_dir / "run_adaptive_dc_sliminfer_v3" / "audit" / "dc_runtime_selection_audit.jsonl"
        )
        selection_by_key = {}
        for row in selection_rows:
            if str(row.get("phase")) != "prefill" or int(row.get("layer_id", -1)) != GATE_LAYER:
                continue
            key = (str(row.get("task")), int(row.get("sample_index")))
            if key not in selection_by_key:
                selection_by_key[key] = row
        tokenizer, dataset2prompt = load_tokenizer_and_prompts(args.model_path)
        samples_cache = {task: load_samples(args.longbench_dir, task) for task in TASKS}
        for task in TASKS:
            for sample_index in heldout_indices(task):
                sel = selection_by_key[(task, sample_index)]
                sample = samples_cache[task][sample_index]
                query_text = str(sample.get("input", sample.get("question", sample.get("query", ""))))
                block_texts = build_prompt_blocks(tokenizer, dataset2prompt[task], sample, task, 32768, 64)
                gate = compute_fixed_gate(
                    gate_config, task, sample_index, str(sel.get("sample_id", "")),
                    query_text, block_texts,
                    sel.get("original_selected_block_ids", []),
                    sel.get("dc_selected_block_ids", []),
                )
                fixed_gate_by_key[(task, sample_index)] = gate

    replay_rows: List[Dict[str, Any]] = []
    task_scores: Dict[str, Dict[str, List[float]]] = {t: defaultdict(list) for t in TASKS}
    dc_positive = dc_negative = trigger_count = 0

    for row in quality:
        task = str(row["task"])
        sample_index = int(row["sample_index"])
        key = (task, sample_index)
        gate = fixed_gate_by_key.get(key, {})
        triggered = parse_bool(gate.get("adaptive_gate_triggered", gate.get("triggered")))
        if triggered:
            trigger_count += 1
        orig_score = to_float(row.get("original_score"))
        dc_score = to_float(row.get("dc_score"))
        replay_score = dc_score if triggered else orig_score
        oracle_score = max(orig_score, dc_score)
        if dc_score > orig_score + 1e-9:
            dc_positive += 1
        elif dc_score < orig_score - 1e-9:
            dc_negative += 1
        task_scores[task]["original"].append(orig_score)
        task_scores[task]["dc"].append(dc_score)
        task_scores[task]["replay"].append(replay_score)
        task_scores[task]["oracle"].append(oracle_score)
        replay_rows.append({
            "task": task,
            "sample_index": sample_index,
            "sample_id": row.get("sample_id", ""),
            "fixed_gate_triggered": triggered,
            "adaptive_gate_reason": gate.get("adaptive_gate_reason", gate.get("reason", "")),
            "chosen_selector": "dc" if triggered else "original",
            "original_score": orig_score,
            "dc_score": dc_score,
            "replay_adaptive_score": replay_score,
            "oracle_score": oracle_score,
            "replay_delta_vs_original": replay_score - orig_score,
            "oracle_gain_vs_original": oracle_score - orig_score,
            "evidence_dispersion_score": gate.get("evidence_dispersion_score", ""),
            "evidence_dispersion_threshold": gate.get("evidence_dispersion_threshold", gate_config["evidence_dispersion_threshold"]),
        })

    original_macro = mean([mean(task_scores[t]["original"]) for t in TASKS])
    dc_macro = mean([mean(task_scores[t]["dc"]) for t in TASKS])
    replay_macro = mean([mean(task_scores[t]["replay"]) for t in TASKS])
    oracle_macro = mean([mean(task_scores[t]["oracle"]) for t in TASKS])
    oracle_gain = oracle_macro - original_macro
    hotpot_oracle_gain = mean(task_scores["hotpotqa"]["oracle"]) - mean(task_scores["hotpotqa"]["original"])
    musique_oracle_gain = mean(task_scores["musique"]["oracle"]) - mean(task_scores["musique"]["original"])

    feature_coverage = sum(
        1 for row in replay_rows if str(row.get("evidence_dispersion_score", "")).strip() not in ("", "None")
    )

    if feature_coverage < 128:
        decision = {"decision": "BUG_STOP_FIX_REQUIRED", "reasons": [f"feature_coverage={feature_coverage}/128"]}
    elif replay_macro <= original_macro and trigger_count >= 120:
        if oracle_gain >= ORACLE_GAIN_THRESHOLD and dc_positive > 0:
            decision = {
                "decision": "FEEDBACK_ADAPTIVE_PROMISING",
                "reasons": [
                    f"replay_macro={replay_macro:.4f} <= original={original_macro:.4f}",
                    f"trigger_count={trigger_count}/128 (near_all_trigger)",
                    f"oracle_gain={oracle_gain:+.4f}",
                    f"dc_positive_count={dc_positive}",
                ],
            }
        else:
            decision = {
                "decision": "STATIC_GATE_RUNTIME_NO_GO",
                "reasons": [
                    f"replay_macro={replay_macro:.4f} <= original={original_macro:.4f}",
                    f"trigger_count={trigger_count}/128",
                ],
            }
    elif replay_macro <= original_macro:
        if oracle_gain >= ORACLE_GAIN_THRESHOLD and dc_positive > 0:
            decision = {
                "decision": "FEEDBACK_ADAPTIVE_PROMISING",
                "reasons": [
                    f"replay_macro={replay_macro:.4f} <= original={original_macro:.4f}",
                    f"oracle_gain={oracle_gain:+.4f}",
                    f"dc_positive_count={dc_positive}",
                ],
            }
        else:
            decision = {
                "decision": "STATIC_GATE_RUNTIME_NO_GO",
                "reasons": [f"replay_macro={replay_macro:.4f} <= original={original_macro:.4f}"],
            }
    elif oracle_gain >= ORACLE_GAIN_THRESHOLD and dc_positive > 0 and replay_macro < oracle_macro - 0.5:
        decision = {
            "decision": "FEEDBACK_ADAPTIVE_PROMISING",
            "reasons": [
                f"replay_macro={replay_macro:.4f} > original={original_macro:.4f}",
                f"oracle_gain={oracle_gain:+.4f}",
                f"dc_positive_count={dc_positive}",
            ],
        }
    else:
        decision = {
            "decision": "BUG_FIXED_GATE_VALIDATION_PASS",
            "reasons": [
                f"replay_macro={replay_macro:.4f}",
                f"original_macro={original_macro:.4f}",
                f"dc_macro={dc_macro:.4f}",
                f"trigger_count={trigger_count}/128",
            ],
        }

    write_csv(
        output_dir / "fixed_gate_existing_prediction_replay.csv",
        replay_rows,
        [
            "task", "sample_index", "sample_id", "fixed_gate_triggered", "adaptive_gate_reason",
            "chosen_selector", "original_score", "dc_score", "replay_adaptive_score", "oracle_score",
            "replay_delta_vs_original", "oracle_gain_vs_original",
            "evidence_dispersion_score", "evidence_dispersion_threshold",
        ],
    )

    summary = {
        "original_macro": original_macro,
        "dc_macro": dc_macro,
        "replay_adaptive_macro": replay_macro,
        "oracle_macro": oracle_macro,
        "oracle_gain_vs_original": oracle_gain,
        "hotpotqa_oracle_gain": hotpot_oracle_gain,
        "musique_oracle_gain": musique_oracle_gain,
        "dc_positive_count": dc_positive,
        "dc_negative_count": dc_negative,
        "fixed_gate_trigger_count": trigger_count,
        "heldout_reported_original_macro": to_float(method_summary.get("macro", {}).get("original", original_macro)),
        "heldout_reported_dc_macro": to_float(method_summary.get("macro", {}).get("dc", dc_macro)),
        "heldout_reported_adaptive_macro": to_float(method_summary.get("macro", {}).get("adaptive", replay_macro)),
    }
    write_csv(output_dir / "replay_summary.csv", [summary], list(summary.keys()))

    replay_report = [
        "",
        "## Fixed Gate Existing-Prediction Replay",
        "",
        f"- Held-out dir: `{heldout_dir}`",
        "",
        "### Scores (existing predictions, fixed gate routing)",
        "",
        f"- Original macro: {original_macro:.4f}",
        f"- DC macro: {dc_macro:.4f}",
        f"- Replay adaptive macro: {replay_macro:.4f}",
        f"- Oracle max(original,dc) macro: {oracle_macro:.4f}",
        f"- Oracle gain vs original: {oracle_gain:+.4f}",
        f"- HotpotQA oracle gain: {hotpot_oracle_gain:+.4f}",
        f"- MuSiQue oracle gain: {musique_oracle_gain:+.4f}",
        "",
        f"- Fixed gate trigger count: {trigger_count}/128",
        f"- DC-positive samples: {dc_positive}",
        f"- DC-negative samples: {dc_negative}",
        "",
        "### Feedback-Adaptive space",
        "",
        f"- Oracle gain >= +{ORACLE_GAIN_THRESHOLD:.1f} macro: **{'yes' if oracle_gain >= ORACLE_GAIN_THRESHOLD else 'no'}** ({oracle_gain:+.4f})",
        f"- DC-positive samples > 0: **{'yes' if dc_positive > 0 else 'no'}** ({dc_positive})",
        f"- Replay adaptive beats original: **{'yes' if replay_macro > original_macro else 'no'}**",
        "",
        "### Replay decision",
        "",
        f"- **{decision['decision']}**",
    ]
    for reason in decision.get("reasons", []):
        replay_report.append(f"- `{reason}`")
    replay_report.append("")

    report_path = output_dir / "report.md"
    if report_path.exists():
        existing = report_path.read_text(encoding="utf-8").rstrip()
        report_path.write_text(existing + "\n" + "\n".join(replay_report), encoding="utf-8")
    else:
        report_path.write_text(
            "# Adaptive DC v3 Runtime Gate Fix Audit\n" + "\n".join(replay_report) + "\n",
            encoding="utf-8",
        )

    decision_path = output_dir / "decision.md"
    replay_decision_lines = [
        "",
        "## Replay decision",
        "",
        f"- **{decision['decision']}**",
    ]
    for reason in decision.get("reasons", []):
        replay_decision_lines.append(f"- Reason: `{reason}`")
    replay_decision_lines.append(
        f"\n- Feedback-Adaptive space remains: **{'yes' if oracle_gain >= ORACLE_GAIN_THRESHOLD and dc_positive > 0 else 'no'}**"
    )
    if decision_path.exists():
        existing = decision_path.read_text(encoding="utf-8").rstrip()
        decision_path.write_text(existing + "\n" + "\n".join(replay_decision_lines) + "\n", encoding="utf-8")
    else:
        decision_path.write_text(
            f"# Adaptive DC v3 Runtime Gate Fix Decision\n" + "\n".join(replay_decision_lines) + "\n",
            encoding="utf-8",
        )

    print(json.dumps({"status": "ok", "output_dir": str(output_dir), "decision": decision, "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
