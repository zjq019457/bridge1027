#!/usr/bin/env python3
"""Offline combined diagnostic: held-out64 gate failure + feedback-adaptive oracle.

No generation, no scorer, no threshold sweep. Reads existing result dirs only.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.cache_utils import _adaptive_dc_v3_compute_gate

DEFAULT_HELDOUT_DIR = Path("results/adaptive_dc_v3_heldout64_diagnostic_20260708_073628")
DEFAULT_TINY_DIR = Path("results/adaptive_dc_v3_tiny_quality_smoke_20260708_060002")
DEFAULT_V3_AUDIT_DIR = Path("results/adaptive_dc_router_v3_feature_audit_20260708_043456")
DEFAULT_LONGBENCH_DIR = Path("/home/yu-bai/Desktop/jiaqizhang/HeadKV/data/LongBench")
DEFAULT_MODEL_PATH = Path(
    "/home/yu-bai/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/"
    "snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
)
GATE_APPLY_LAYER = 26
ORACLE_GAIN_THRESHOLD = 2.0
TASKS = ["hotpotqa", "musique"]

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


def read_csv(path: Path) -> List[JsonRow]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> List[JsonRow]:
    rows: List[JsonRow] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def to_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        out = float(value)
        return out if math.isfinite(out) else None
    except (TypeError, ValueError):
        return None


def mean(values: Iterable[float]) -> float:
    items = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(sum(items) / len(items)) if items else 0.0


def percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct / 100.0
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return ordered[lo]
    weight = rank - lo
    return ordered[lo] * (1 - weight) + ordered[hi] * weight


def load_frozen_gate_config(heldout_run_config: JsonRow, tiny_dir: Path, v3_audit_dir: Path) -> Dict[str, Any]:
    frozen = heldout_run_config.get("frozen_gate_config") or {}
    threshold = frozen.get("evidence_dispersion_threshold")
    if threshold is None:
        tiny_cfg = json.loads((tiny_dir / "run_config.json").read_text(encoding="utf-8"))
        frozen = tiny_cfg.get("frozen_gate_config") or {}
        threshold = frozen.get("evidence_dispersion_threshold")
    if threshold is None:
        rule_csv = v3_audit_dir / "adaptive_dc_v3_rule_replay.csv"
        with rule_csv.open("r", encoding="utf-8") as handle:
            rule3 = [row for row in csv.DictReader(handle) if "Rule 3" in str(row.get("rule", ""))]
        if not rule3:
            raise RuntimeError("BUG_STOP: frozen threshold unavailable")
        threshold = float(rule3[0]["diagnostic_median_evidence_dispersion_score"])
    return {
        "evidence_dispersion_threshold": float(threshold),
        "norm_min": frozen.get("norm_min") or {},
        "norm_max": frozen.get("norm_max") or {},
        "gate_apply_layer": int(frozen.get("gate_apply_layer", GATE_APPLY_LAYER)),
        "top_pruned_m": int(frozen.get("top_pruned_m", 16)),
        "threshold_NOT_recomputed_from_current_batch": bool(
            heldout_run_config.get("threshold_NOT_recomputed_from_current_batch", True)
        ),
        "source_tiny_go_run": str(heldout_run_config.get("valid_tiny_go_run", tiny_dir)),
    }


def load_tokenizer_and_prompts(longbench_dir: Path, model_path: Path) -> Tuple[Any, Dict[str, str]]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    dataset2prompt = json.loads((REPO_ROOT / "configs" / "dataset2prompt.json").read_text(encoding="utf-8"))
    if not isinstance(dataset2prompt, dict):
        raise RuntimeError("configs/dataset2prompt.json missing or invalid")
    return tokenizer, dataset2prompt


def load_longbench_samples(longbench_dir: Path, dataset: str) -> List[Dict[str, Any]]:
    path = longbench_dir / f"{dataset}.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_chat_prompt(tokenizer: Any, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def build_prompt_blocks(
    tokenizer: Any,
    prompt_format: str,
    sample: Dict[str, Any],
    dataset: str,
    max_model_len: int,
    block_size: int,
) -> Tuple[str, List[str]]:
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
            tokenizer.decode(
                chat_ids[start : start + block_size],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        )
    return prompt, blocks


def jaccard(a: Sequence[int], b: Sequence[int]) -> float:
    sa, sb = set(int(x) for x in a), set(int(x) for x in b)
    union = sa | sb
    return float(len(sa & sb) / len(union)) if union else 1.0


def read_final_selection_audit(adaptive_run_dir: Path) -> Dict[Key, JsonRow]:
    path = adaptive_run_dir / "audit" / "dc_runtime_selection_audit.jsonl"
    rows = read_jsonl(path)
    selected: Dict[Key, JsonRow] = {}
    for row in rows:
        if str(row.get("phase", "")) != "prefill":
            continue
        if int(row.get("layer_id", -1)) != GATE_APPLY_LAYER:
            continue
        key = (str(row.get("task", "")), int(row.get("sample_index", -1)))
        if key[0] and key[1] >= 0 and key not in selected:
            selected[key] = row
    return selected


def recompute_gate_row(
    gate_config: Dict[str, Any],
    task: str,
    sample_index: int,
    sample_id: str,
    query_text: str,
    block_texts: List[str],
    original_selected: Sequence[int],
    dc_selected: Sequence[int],
    norm_mode: str,
    batch_norm: Optional[Dict[str, Dict[str, float]]] = None,
) -> JsonRow:
    cfg_gate = dict(gate_config)
    if norm_mode == "batch" and batch_norm:
        cfg_gate = {**gate_config, "norm_min": batch_norm["min"], "norm_max": batch_norm["max"]}
    cfg = {
        "adaptive_dc_v3_gate": cfg_gate,
        "block_texts": block_texts,
        "query_text": query_text,
        "audit_context": {"task": task, "sample_index": sample_index, "sample_id": sample_id},
    }
    gate = _adaptive_dc_v3_compute_gate(
        cfg=cfg,
        original_selected_block_ids=set(int(x) for x in original_selected),
        dc_selected_block_ids=set(int(x) for x in dc_selected),
        num_blocks=len(block_texts),
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
        "norm_mode": norm_mode,
        "predicates": preds,
        "all_predicates_pass": all(preds.values()),
    }


def compute_batch_norm(raw_rows: List[JsonRow]) -> Dict[str, Dict[str, float]]:
    mins: Dict[str, float] = {}
    maxs: Dict[str, float] = {}
    for feat in ["doc_entropy", "entity_coverage_gap", "pruned_query_overlap_max"]:
        vals = [to_float(row.get(feat)) for row in raw_rows]
        clean = [v for v in vals if v is not None]
        if clean:
            mins[feat if feat != "doc_entropy" else "original_doc_entropy"] = min(clean)
            maxs[feat if feat != "doc_entropy" else "original_doc_entropy"] = max(clean)
    return {"min": mins, "max": maxs}


def distribution_row(cohort: str, values: Sequence[Optional[float]]) -> JsonRow:
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not clean:
        return {"cohort": cohort, "count": 0}
    return {
        "cohort": cohort,
        "count": len(clean),
        "mean": mean(clean),
        "median": float(statistics.median(clean)),
        "p25": percentile(clean, 25),
        "p75": percentile(clean, 75),
        "min": min(clean),
        "max": max(clean),
    }


def predicate_summary(rows: Sequence[JsonRow], prefix: str = "") -> List[JsonRow]:
    names = [
        "evidence_dispersion_score_ge_threshold",
        "entity_coverage_gap_ge_zero",
        "pruned_query_overlap_max_gt_selected_mean",
        "selected_jaccard_vs_original_ge_030",
        "budget_mismatch_eq_zero",
        "fallback_eq_zero",
        "all_rule3_predicates_pass",
        "gate_triggered",
    ]
    total = len(rows)
    out = []
    for name in names:
        if name == "budget_mismatch_eq_zero":
            passes = sum(1 for row in rows if not bool(row.get("budget_mismatch")))
            fails = total - passes
        elif name == "fallback_eq_zero":
            passes = sum(1 for row in rows if not bool(row.get("fallback")))
            fails = total - passes
        elif name == "all_rule3_predicates_pass":
            passes = sum(1 for row in rows if bool(row.get("all_rule3_predicates_pass")))
            fails = total - passes
        elif name == "gate_triggered":
            passes = sum(1 for row in rows if bool(row.get("gate_triggered")))
            fails = total - passes
        else:
            passes = sum(1 for row in rows if bool((row.get("predicates") or {}).get(name)))
            fails = total - passes
        out.append({
            "predicate": f"{prefix}{name}" if prefix else name,
            "pass_count": passes,
            "fail_count": fails,
            "pass_rate": passes / total if total else 0.0,
        })
    return out


def read_predictions_disagreement(heldout_dir: Path, keys: Sequence[Key]) -> Dict[Key, bool]:
    out: Dict[Key, bool] = {}
    for method in ("sliminfer_original", "dc_sliminfer"):
        pass
    orig_dir = heldout_dir / "run_sliminfer_original" / "longbench_predictions"
    dc_dir = heldout_dir / "run_dc_sliminfer" / "longbench_predictions"
    orig_preds: Dict[Key, str] = {}
    dc_preds: Dict[Key, str] = {}
    for task in TASKS:
        for row in read_jsonl(orig_dir / f"{task}.jsonl"):
            key = (task, int(row.get("sample_index", -1)))
            orig_preds[key] = str(row.get("pred", "")).strip()
        for row in read_jsonl(dc_dir / f"{task}.jsonl"):
            key = (task, int(row.get("sample_index", -1)))
            dc_preds[key] = str(row.get("pred", "")).strip()
    for key in keys:
        out[key] = orig_preds.get(key, "") != dc_preds.get(key, "")
    return out


def decide(
    bug_reasons: List[str],
    heldout_frozen_trigger_count: int,
    heldout_batch_trigger_count: int,
    runtime_recorded_feature_count: int,
    oracle_macro_gain: float,
    dc_positive_count: int,
    dc_negative_count: int,
    bottleneck: str,
) -> Dict[str, Any]:
    if bug_reasons:
        return {"decision": "BUG_STOP_FIX_REQUIRED", "reasons": bug_reasons}

    if runtime_recorded_feature_count == 0 and heldout_frozen_trigger_count > 0:
        return {
            "decision": "BUG_STOP_FIX_REQUIRED",
            "reasons": [
                "runtime_gate_features_missing_but_offline_frozen_norm_would_trigger",
                f"offline_frozen_trigger_count={heldout_frozen_trigger_count}",
            ],
        }

    if oracle_macro_gain > ORACLE_GAIN_THRESHOLD and dc_positive_count > 0:
        return {
            "decision": "STATIC_GATE_OVERFITTED_BUT_FEEDBACK_PROMISING",
            "reasons": [
                f"oracle_macro_gain={oracle_macro_gain:+.4f} > {ORACLE_GAIN_THRESHOLD}",
                f"dc_positive_count={dc_positive_count}",
                f"heldout_frozen_trigger_count={heldout_frozen_trigger_count}",
                f"primary_bottleneck={bottleneck}",
            ],
        }

    if heldout_frozen_trigger_count == 0 and heldout_batch_trigger_count > heldout_frozen_trigger_count + 5:
        return {
            "decision": "STATIC_GATE_CAN_BE_FIXED",
            "reasons": [
                "frozen_threshold_likely_too_strict_on_heldout_distribution",
                f"heldout_frozen_trigger_count={heldout_frozen_trigger_count}",
                f"heldout_batch_trigger_count={heldout_batch_trigger_count}",
                f"primary_bottleneck={bottleneck}",
            ],
        }

    if oracle_macro_gain <= 1.0 or dc_positive_count <= 2:
        return {
            "decision": "FEEDBACK_ADAPTIVE_DC_NO_GO",
            "reasons": [
                f"oracle_macro_gain={oracle_macro_gain:+.4f}",
                f"dc_positive_count={dc_positive_count}",
                f"dc_negative_count={dc_negative_count}",
            ],
        }

    return {
        "decision": "STATIC_GATE_CAN_BE_FIXED",
        "reasons": [
            f"primary_bottleneck={bottleneck}",
            f"heldout_frozen_trigger_count={heldout_frozen_trigger_count}",
            f"oracle_macro_gain={oracle_macro_gain:+.4f}",
        ],
    }


def write_report(
    output_dir: Path,
    gate_rows: List[JsonRow],
    predicate_rows: List[JsonRow],
    dist_rows: List[JsonRow],
    oracle_summary: JsonRow,
    decision: Dict[str, Any],
    meta: Dict[str, Any],
) -> None:
    bottleneck = meta.get("primary_bottleneck", "unknown")
    lines = [
        "# Adaptive DC Feedback Combined Analysis",
        "",
        f"- Output directory: `{output_dir}`",
        f"- Held-out dir: `{meta.get('heldout_dir')}`",
        f"- Tiny GO dir: `{meta.get('tiny_dir')}`",
        f"- V3 audit dir: `{meta.get('v3_audit_dir')}`",
        "",
        "## A. Gate Failure Analysis",
        "",
        f"- Runtime recorded gate feature rows: **{meta.get('runtime_recorded_feature_count', 0)}/128**",
        f"- Offline recomputed (frozen norm) trigger count: **{meta.get('heldout_frozen_trigger_count', 0)}/128**",
        f"- Offline recomputed (batch norm) trigger count: **{meta.get('heldout_batch_trigger_count', 0)}/128**",
        f"- Primary bottleneck predicate: **{bottleneck}**",
        f"- Threshold value: {meta.get('threshold')}",
        f"- Threshold type: **normalized evidence_dispersion_score** (mean of 3 min-max normalized components)",
        f"- Frozen threshold source: `{meta.get('threshold_source')}`",
        f"- Held-out used batch normalization at runtime: **{meta.get('runtime_used_batch_norm', 'No')}**",
        "",
        "### Predicate Pass/Fail (offline frozen norm, held-out64)",
        "",
        "| predicate | pass | fail | pass_rate |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in predicate_rows:
        if not str(row.get("predicate", "")).startswith("heldout_frozen_"):
            continue
        lines.append(
            f"| {row['predicate'].replace('heldout_frozen_', '')} | {row['pass_count']} | {row['fail_count']} | {row['pass_rate']:.3f} |"
        )
    lines.extend([
        "",
        "### Tiny vs Held-Out Feature Distribution",
        "",
        "| cohort | count | mean | median | p25 | p75 | min | max |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in dist_rows:
        if row.get("count", 0) == 0:
            continue
        lines.append(
            f"| {row['cohort']} | {row['count']} | {row['mean']:.4f} | {row['median']:.4f} | "
            f"{row['p25']:.4f} | {row['p75']:.4f} | {row['min']:.4f} | {row['max']:.4f} |"
        )
    lines.extend([
        "",
        "## B. Feedback Oracle Analysis",
        "",
        f"- Original macro: {oracle_summary.get('original_macro', 0):.4f}",
        f"- DC macro: {oracle_summary.get('dc_macro', 0):.4f}",
        f"- Oracle max(original, dc) macro: {oracle_summary.get('oracle_macro', 0):.4f}",
        f"- Oracle gain over original: {oracle_summary.get('oracle_macro_gain', 0):+.4f}",
        f"- DC-positive samples: {oracle_summary.get('dc_positive_count', 0)}",
        f"- DC-negative samples: {oracle_summary.get('dc_negative_count', 0)}",
        f"- Prediction disagreement rate: {oracle_summary.get('prediction_disagreement_rate', 0):.3f}",
        f"- HotpotQA oracle gain: {oracle_summary.get('hotpotqa_oracle_gain', 0):+.4f}",
        f"- MuSiQue oracle gain: {oracle_summary.get('musique_oracle_gain', 0):+.4f}",
        "",
        "## C. Decision",
        "",
        f"- Final conclusion: **{decision['decision']}**",
    ])
    for reason in decision.get("reasons", []):
        lines.append(f"- Reason: `{reason}`")
    for reason in meta.get("bug_reasons", []):
        lines.append(f"- Bug note: `{reason}`")
    lines.append("")
    (output_dir / "combined_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heldout_dir", type=Path, default=DEFAULT_HELDOUT_DIR)
    parser.add_argument("--tiny_dir", type=Path, default=DEFAULT_TINY_DIR)
    parser.add_argument("--v3_audit_dir", type=Path, default=DEFAULT_V3_AUDIT_DIR)
    parser.add_argument("--longbench_dir", type=Path, default=DEFAULT_LONGBENCH_DIR)
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output_root", type=Path, default=Path("results"))
    parser.add_argument("--max_model_len", type=int, default=32768)
    parser.add_argument("--block_size", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    heldout_dir = (REPO_ROOT / args.heldout_dir).resolve()
    tiny_dir = (REPO_ROOT / args.tiny_dir).resolve()
    v3_audit_dir = (REPO_ROOT / args.v3_audit_dir).resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (REPO_ROOT / args.output_root / f"adaptive_dc_feedback_combined_analysis_{timestamp}").resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    heldout_run_config = json.loads((heldout_dir / "run_config.json").read_text(encoding="utf-8"))
    gate_config = load_frozen_gate_config(heldout_run_config, tiny_dir, v3_audit_dir)
    heldout_quality = read_csv(heldout_dir / "per_sample_quality.csv")
    heldout_gate_audit = read_csv(heldout_dir / "adaptive_gate_audit.csv")
    tiny_gate_audit = read_csv(tiny_dir / "adaptive_gate_audit.csv")
    v3_feature_rows = read_csv(v3_audit_dir / "adaptive_dc_v3_feature_table.csv")

    runtime_recorded_feature_count = sum(
        1 for row in heldout_gate_audit if to_float(row.get("evidence_dispersion_score")) is not None
    )

    selection_audit = read_final_selection_audit(heldout_dir / "run_adaptive_dc_sliminfer_v3")
    tokenizer, dataset2prompt = load_tokenizer_and_prompts(args.longbench_dir, args.model_path)
    samples_cache: Dict[str, List[Dict[str, Any]]] = {
        task: load_longbench_samples(args.longbench_dir, task) for task in TASKS
    }

    quality_by_key = {(row["task"], int(row["sample_index"])): row for row in heldout_quality}
    keys = sorted(quality_by_key.keys(), key=lambda item: (item[0], item[1]))

    heldout_gate_by_key = {
        (row["task"], int(row["sample_index"])): row for row in heldout_gate_audit
    }

    raw_feature_rows: List[JsonRow] = []
    gate_rows: List[JsonRow] = []
    block_cache: Dict[Key, Tuple[str, List[str]]] = {}

    for task, sample_index in keys:
        quality = quality_by_key[(task, sample_index)]
        sel = selection_audit.get((task, sample_index))
        if sel is None:
            raise RuntimeError(f"missing selection audit for {task}:{sample_index}")

        sample = samples_cache[task][sample_index]
        query_text = str(sample.get("input", sample.get("question", sample.get("query", ""))))
        _, block_texts = build_prompt_blocks(
            tokenizer=tokenizer,
            prompt_format=dataset2prompt[task],
            sample=sample,
            dataset=task,
            max_model_len=args.max_model_len,
            block_size=args.block_size,
        )
        block_cache[(task, sample_index)] = (query_text, block_texts)
        original_selected = [int(x) for x in sel.get("original_selected_block_ids", [])]
        dc_selected = [int(x) for x in sel.get("dc_selected_block_ids", [])]

        frozen_gate = recompute_gate_row(
            gate_config=gate_config,
            task=task,
            sample_index=sample_index,
            sample_id=str(quality.get("sample_id", "")),
            query_text=query_text,
            block_texts=block_texts,
            original_selected=original_selected,
            dc_selected=dc_selected,
            norm_mode="frozen",
        )
        runtime_row = heldout_gate_by_key.get((task, sample_index), {})
        raw_feature_rows.append({
            "task": task,
            "sample_index": sample_index,
            "doc_entropy": frozen_gate.get("doc_entropy"),
            "entity_coverage_gap": frozen_gate.get("entity_coverage_gap"),
            "pruned_query_overlap_max": frozen_gate.get("pruned_query_overlap_max"),
        })
        gate_rows.append({
            "task": task,
            "sample_index": sample_index,
            "sample_id": quality.get("sample_id", ""),
            "runtime_recorded_eds": to_float(runtime_row.get("evidence_dispersion_score")),
            "offline_frozen_eds": frozen_gate.get("evidence_dispersion_score"),
            "offline_frozen_triggered": frozen_gate.get("triggered"),
            "offline_frozen_reason": frozen_gate.get("reason"),
            "entity_coverage_gap": frozen_gate.get("entity_coverage_gap"),
            "pruned_query_overlap_max": frozen_gate.get("pruned_query_overlap_max"),
            "selected_query_overlap_mean": frozen_gate.get("selected_query_overlap_mean"),
            "selected_jaccard_vs_original": frozen_gate.get("selected_jaccard_vs_original"),
            "predicates": frozen_gate.get("predicates"),
            "all_rule3_predicates_pass": frozen_gate.get("all_predicates_pass"),
            "gate_triggered": frozen_gate.get("triggered"),
            "budget_mismatch": quality.get("budget_mismatch") in (True, "True", "true", "1"),
            "fallback": quality.get("fallback") in (True, "True", "true", "1"),
            "feature_source": frozen_gate.get("feature_source"),
            "original_score": to_float(quality.get("original_score")),
            "dc_score": to_float(quality.get("dc_score")),
        })

    batch_norm = compute_batch_norm(raw_feature_rows)
    for i, (task, sample_index) in enumerate(keys):
        quality = quality_by_key[(task, sample_index)]
        sel = selection_audit[(task, sample_index)]
        query_text, block_texts = block_cache[(task, sample_index)]
        batch_gate = recompute_gate_row(
            gate_config=gate_config,
            task=task,
            sample_index=sample_index,
            sample_id=str(quality.get("sample_id", "")),
            query_text=query_text,
            block_texts=block_texts,
            original_selected=[int(x) for x in sel.get("original_selected_block_ids", [])],
            dc_selected=[int(x) for x in sel.get("dc_selected_block_ids", [])],
            norm_mode="batch",
            batch_norm=batch_norm,
        )
        gate_rows[i]["offline_batch_eds"] = batch_gate.get("evidence_dispersion_score")
        gate_rows[i]["offline_batch_triggered"] = batch_gate.get("triggered")

    heldout_frozen_trigger_count = sum(1 for row in gate_rows if bool(row.get("offline_frozen_triggered")))
    heldout_batch_trigger_count = sum(1 for row in gate_rows if bool(row.get("offline_batch_triggered")))

    predicate_rows = predicate_summary(gate_rows, prefix="heldout_frozen_")
    tiny_predicate_rows = []
    for row in tiny_gate_audit:
        threshold = float(gate_config["evidence_dispersion_threshold"])
        eds = to_float(row.get("evidence_dispersion_score"))
        ecg = to_float(row.get("entity_coverage_gap"))
        pqo = to_float(row.get("pruned_query_overlap_max"))
        sqm = to_float(row.get("selected_query_overlap_mean"))
        jac = to_float(row.get("selected_jaccard_vs_original"))
        tiny_predicate_rows.append({
            "predicates": {
                "evidence_dispersion_score_ge_threshold": eds is not None and eds >= threshold,
                "entity_coverage_gap_ge_zero": ecg is not None and ecg >= 0.0,
                "pruned_query_overlap_max_gt_selected_mean": pqo is not None and sqm is not None and pqo > sqm,
                "selected_jaccard_vs_original_ge_030": jac is not None and jac >= 0.30,
            },
            "budget_mismatch": False,
            "fallback": False,
            "all_rule3_predicates_pass": bool(row.get("adaptive_gate_triggered") == "True" or row.get("adaptive_gate_triggered") is True),
            "gate_triggered": row.get("adaptive_gate_triggered") in (True, "True", "true", "1"),
        })
    predicate_rows.extend(predicate_summary(tiny_predicate_rows, prefix="tiny_"))

    fail_counts = {
        row["predicate"].replace("heldout_frozen_", ""): row["fail_count"]
        for row in predicate_rows
        if str(row.get("predicate", "")).startswith("heldout_frozen_")
        and row["predicate"] not in ("heldout_frozen_budget_mismatch_eq_zero", "heldout_frozen_fallback_eq_zero", "heldout_frozen_gate_triggered", "heldout_frozen_all_rule3_predicates_pass")
    }
    primary_bottleneck = max(fail_counts, key=lambda k: fail_counts[k]) if fail_counts else "unknown"

    tiny_eds = [to_float(row.get("evidence_dispersion_score")) for row in tiny_gate_audit]
    tiny_eds += [to_float(row.get("evidence_dispersion_score")) for row in v3_feature_rows]
    heldout_eds = [to_float(row.get("offline_frozen_eds")) for row in gate_rows]
    tiny_ecg = [to_float(row.get("entity_coverage_gap")) for row in tiny_gate_audit]
    tiny_ecg += [to_float(row.get("entity_coverage_gap")) for row in v3_feature_rows]
    heldout_ecg = [to_float(row.get("entity_coverage_gap")) for row in gate_rows]

    dist_rows = [
        distribution_row("tiny_evidence_dispersion_score", tiny_eds),
        distribution_row("heldout_evidence_dispersion_score_frozen_norm", heldout_eds),
        distribution_row("tiny_entity_coverage_gap", tiny_ecg),
        distribution_row("heldout_entity_coverage_gap", heldout_ecg),
    ]

    disagreement = read_predictions_disagreement(heldout_dir, keys)
    oracle_rows: List[JsonRow] = []
    task_scores: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    dc_positive = dc_negative = disagree = 0

    for i, (task, sample_index) in enumerate(keys):
        quality = quality_by_key[(task, sample_index)]
        orig = float(quality.get("original_score", 0))
        dc = float(quality.get("dc_score", 0))
        oracle = max(orig, dc)
        delta = oracle - orig
        if dc > orig + 1e-9:
            dc_positive += 1
        elif dc < orig - 1e-9:
            dc_negative += 1
        if disagreement.get((task, sample_index), False):
            disagree += 1
        task_scores[task]["original"].append(orig)
        task_scores[task]["dc"].append(dc)
        task_scores[task]["oracle"].append(oracle)
        oracle_rows.append({
            "task": task,
            "sample_index": sample_index,
            "sample_id": quality.get("sample_id", ""),
            "original_score": orig,
            "dc_score": dc,
            "oracle_score": oracle,
            "oracle_gain_vs_original": delta,
            "dc_delta_vs_original": dc - orig,
            "prediction_disagreement": disagreement.get((task, sample_index), False),
            "offline_frozen_gate_triggered": gate_rows[i].get("offline_frozen_triggered"),
        })

    original_macro = mean(mean(task_scores[t]["original"]) for t in TASKS)
    dc_macro = mean(mean(task_scores[t]["dc"]) for t in TASKS)
    oracle_macro = mean(mean(task_scores[t]["oracle"]) for t in TASKS)
    oracle_summary = {
        "original_macro": original_macro,
        "dc_macro": dc_macro,
        "oracle_macro": oracle_macro,
        "oracle_macro_gain": oracle_macro - original_macro,
        "dc_positive_count": dc_positive,
        "dc_negative_count": dc_negative,
        "prediction_disagreement_rate": disagree / len(keys) if keys else 0.0,
        "hotpotqa_oracle_gain": mean(task_scores["hotpotqa"]["oracle"]) - mean(task_scores["hotpotqa"]["original"]),
        "musique_oracle_gain": mean(task_scores["musique"]["oracle"]) - mean(task_scores["musique"]["original"]),
    }

    bug_reasons: List[str] = []
    if runtime_recorded_feature_count == 0:
        bug_reasons.append("runtime_gate_features_missing_in_heldout_audit")
    sel_with_gate_field = sum(
        1 for row in selection_audit.values()
        if row.get("adaptive_dc_v3_gate") or row.get("adaptive_gate_triggered") is not None
    )
    if sel_with_gate_field == 0:
        bug_reasons.append("runtime_selection_audit_missing_adaptive_gate_payload")
    if not gate_config.get("threshold_NOT_recomputed_from_current_batch", True):
        bug_reasons.append("heldout_config_not_marked_frozen_threshold")

    hard_bug = [r for r in bug_reasons if "missing" in r or "not_marked" in r]
    if runtime_recorded_feature_count == 0 and heldout_frozen_trigger_count > 0:
        hard_bug.append("offline_recompute_shows_nonzero_triggers_but_runtime_recorded_zero")

    decision = decide(
        bug_reasons=hard_bug,
        heldout_frozen_trigger_count=heldout_frozen_trigger_count,
        heldout_batch_trigger_count=heldout_batch_trigger_count,
        runtime_recorded_feature_count=runtime_recorded_feature_count,
        oracle_macro_gain=float(oracle_summary["oracle_macro_gain"]),
        dc_positive_count=dc_positive,
        dc_negative_count=dc_negative,
        bottleneck=primary_bottleneck,
    )

    meta = {
        "heldout_dir": str(heldout_dir),
        "tiny_dir": str(tiny_dir),
        "v3_audit_dir": str(v3_audit_dir),
        "runtime_recorded_feature_count": runtime_recorded_feature_count,
        "heldout_frozen_trigger_count": heldout_frozen_trigger_count,
        "heldout_batch_trigger_count": heldout_batch_trigger_count,
        "primary_bottleneck": primary_bottleneck,
        "threshold": gate_config["evidence_dispersion_threshold"],
        "threshold_source": gate_config.get("source_tiny_go_run"),
        "runtime_used_batch_norm": heldout_batch_trigger_count != heldout_frozen_trigger_count,
        "bug_reasons": bug_reasons,
    }

    write_csv(output_dir / "gate_predicate_pass_fail.csv", predicate_rows,
              ["predicate", "pass_count", "fail_count", "pass_rate"])
    write_csv(output_dir / "tiny_vs_heldout_feature_distribution.csv", dist_rows,
              ["cohort", "count", "mean", "median", "p25", "p75", "min", "max"])
    write_csv(output_dir / "feedback_oracle_per_sample.csv", oracle_rows,
              ["task", "sample_index", "sample_id", "original_score", "dc_score", "oracle_score",
               "oracle_gain_vs_original", "dc_delta_vs_original", "prediction_disagreement",
               "offline_frozen_gate_triggered"])
    write_csv(output_dir / "feedback_oracle_summary.csv", [oracle_summary],
              list(oracle_summary.keys()))

    gate_detail_fields = [
        "task", "sample_index", "sample_id",
        "runtime_recorded_eds", "offline_frozen_eds", "offline_batch_eds",
        "offline_frozen_triggered", "offline_batch_triggered", "offline_frozen_reason",
        "entity_coverage_gap", "pruned_query_overlap_max", "selected_query_overlap_mean",
        "selected_jaccard_vs_original", "feature_source",
        "original_score", "dc_score",
    ]
    simplified_gate_rows = [{k: row.get(k) for k in gate_detail_fields} for row in gate_rows]
    write_csv(output_dir / "heldout_offline_gate_recompute.csv", simplified_gate_rows, gate_detail_fields)

    write_report(output_dir, gate_rows, predicate_rows, dist_rows, oracle_summary, decision, meta)
    write_json(output_dir / "analysis_config.json", {
        "heldout_dir": str(heldout_dir),
        "tiny_dir": str(tiny_dir),
        "v3_audit_dir": str(v3_audit_dir),
        "gate_config": gate_config,
        "meta": meta,
        "decision": decision,
    })
    lines = [
        "# Adaptive DC Feedback Combined Decision",
        "",
        f"- Decision: **{decision['decision']}**",
    ]
    for reason in decision.get("reasons", []):
        lines.append(f"- Reason: `{reason}`")
    lines.append("")
    lines.append(f"- Output directory: `{output_dir}`")
    (output_dir / "decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "status": "ok",
        "output_dir": str(output_dir),
        "decision": decision,
        "heldout_frozen_trigger_count": heldout_frozen_trigger_count,
        "oracle_macro_gain": oracle_summary["oracle_macro_gain"],
        "primary_bottleneck": primary_bottleneck,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
