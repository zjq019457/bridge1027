#!/usr/bin/env python3
"""Layer-Consensus SlimInfer (lc_sliminfer) selection audit on held-out64.

Prefill-only gate_audit captures per-layer block scores via uasr_score_dump,
then offline recomputes budget-neutral LC top-K selection and compares it against
runtime sliminfer_original. No generation, no scorer, no gold.
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

from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.run_qwen25_sliminfer_variants import (
    build_chat,
    build_prompt_block_texts,
    _sample_query_text,
)
from tools.audit_query_decomposed_coverage_sliminfer import (
    BLOCK_SIZE,
    TASKS,
    heldout_indices,
    heldout_keys,
    parse_bool,
    percentile,
    preview_text,
)
from utils.cache_utils import _dc_selected_redundancy, _gate_content_tokens

DEFAULT_LONGBENCH_DIR = Path("/home/yu-bai/Desktop/jiaqizhang/HeadKV/data/LongBench")
DEFAULT_MODEL_PATH = Path(
    "/home/yu-bai/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/"
    "snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
)
PRUNE_LAYERS = [9, 18, 26]
FINAL_LAYER = PRUNE_LAYERS[-1]
PREFILL_AUDIT_LAYER = 27
METHOD_ORIGINAL = "sliminfer_original"
MAX_MODEL_LEN = 32768

Key = Tuple[str, int]
JsonRow = Dict[str, Any]

REQUIRED_SCORE_FIELDS = (
    "layer_id",
    "all_candidate_block_ids",
    "all_candidate_scores",
    "selected_block_ids",
    "budget_k",
    "phase",
    "task",
    "sample_index",
)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _serialize(row.get(field, "")) for field in fields})


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _serialize(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, bool):
        return value
    return value


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


def python_bin() -> Path:
    candidate = REPO_ROOT / ".venv" / "bin" / "python"
    return candidate if candidate.exists() else Path(sys.executable)


def build_sample_spec() -> Tuple[str, Dict[str, List[int]]]:
    indices = {task: heldout_indices(task, count=64) for task in TASKS}
    spec = ";".join(f"{task}:" + ",".join(str(i) for i in indices[task]) for task in TASKS)
    return spec, indices


def load_samples(longbench_dir: Path, task: str) -> List[Dict[str, Any]]:
    return [
        json.loads(line)
        for line in (longbench_dir / f"{task}.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_dataset2prompt() -> Dict[str, str]:
    return json.loads((REPO_ROOT / "configs" / "dataset2prompt.json").read_text(encoding="utf-8"))


def build_runner_prompt_bundle(
    tokenizer: AutoTokenizer,
    prompt_format: str,
    sample: Dict[str, Any],
    dataset: str,
) -> Tuple[str, List[str], List[int]]:
    prompt = prompt_format.format(**sample)
    tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
    if len(tokenized_prompt) > MAX_MODEL_LEN:
        half = MAX_MODEL_LEN // 2
        prompt = (
            tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)
            + tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        )
    if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
        prompt = build_chat(tokenizer, prompt)
    model_input = tokenizer(prompt, truncation=False, return_tensors="pt")
    input_ids = model_input.input_ids[0]
    seq_len = int(input_ids.shape[0])
    block_texts = build_prompt_block_texts(tokenizer, input_ids.unsqueeze(0), BLOCK_SIZE)
    token_lengths = [min(BLOCK_SIZE, seq_len - start) for start in range(0, seq_len, BLOCK_SIZE)]
    query_text = _sample_query_text(sample)
    return query_text, block_texts, token_lengths


def base_gate_audit_cmd(
    args: argparse.Namespace,
    run_dir: Path,
    sample_spec: str,
    score_dump_path: Path,
    used_blocks_path: Path,
    block_scores_path: Path,
    dc_sink_path: Path,
) -> List[str]:
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
        "--longbench_subset_name", "lc_sliminfer_selection_audit_heldout64",
        "--max_model_len", str(MAX_MODEL_LEN),
        "--decode_policy", "original",
        "--rescue_policy", "none",
        "--prune_layers", ",".join(str(x) for x in PRUNE_LAYERS),
        "--retain_tokens", "12288,6144,4096",
        "--block_size", str(BLOCK_SIZE),
        "--token_unit_size", "8",
        "--local_query_window", "4",
        "--swap_threshold", "0.9",
        "--seed", "0",
        "--output_dir", str(run_dir),
        "--min_free_gpu_gb", "0",
        "--eval", "gate_audit",
        "--method", METHOD_ORIGINAL,
        "--uasr_score_dump_path", str(score_dump_path),
        "--uasr_tail_width", "8",
        "--dc_selection_audit_path", str(dc_sink_path),
        "--save_block_scores", "sampled",
        "--debug_used_block_audit_path", str(used_blocks_path),
        "--debug_used_block_include_full", "true",
        "--debug_used_block_full_layers", str(PREFILL_AUDIT_LAYER),
    ]


def run_command(cmd: Sequence[str], cwd: Path, stdout_path: Path, stderr_path: Path) -> None:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        proc = subprocess.run(list(cmd), cwd=str(cwd), text=True, stdout=stdout, stderr=stderr)
    stdout_text = stdout_path.read_text(encoding="utf-8", errors="ignore")
    stderr_text = stderr_path.read_text(encoding="utf-8", errors="ignore")
    if "CUDA out of memory" in stdout_text or "CUDA out of memory" in stderr_text:
        raise RuntimeError(f"CUDA OOM while running: {' '.join(cmd)}")
    if proc.returncode != 0:
        raise RuntimeError(f"runner failed rc={proc.returncode}; see {stderr_path}")


def run_score_instrumentation(args: argparse.Namespace, output_dir: Path, sample_spec: str) -> Dict[str, Path]:
    run_dir = output_dir / "run_sliminfer_original_gate_audit"
    run_dir.mkdir(parents=True, exist_ok=True)
    score_dump_path = run_dir / "uasr_score_dump.jsonl"
    used_blocks_path = run_dir / "used_blocks.jsonl"
    block_scores_path = run_dir / "block_scores.jsonl"
    dc_sink_path = run_dir / "dc_selection_audit_sink.jsonl"
    for path in (score_dump_path, used_blocks_path, block_scores_path, dc_sink_path):
        path.write_text("", encoding="utf-8")
    cmd = base_gate_audit_cmd(args, run_dir, sample_spec, score_dump_path, used_blocks_path, block_scores_path, dc_sink_path)
    write_json(output_dir / "lc_gate_audit_command.json", {"command": cmd})
    run_command(cmd, REPO_ROOT, output_dir / "lc_gate_audit_stdout.log", output_dir / "lc_gate_audit_stderr.log")
    return {
        "score_dump_path": score_dump_path,
        "used_blocks_path": used_blocks_path,
        "block_scores_path": block_scores_path,
    }


def probe_per_layer_score_availability(
    score_dump_path: Path,
    block_scores_path: Path,
    keys: Sequence[Key],
) -> Dict[str, Any]:
    rows = read_jsonl(score_dump_path, missing_ok=True)
    block_rows = read_jsonl(block_scores_path, missing_ok=True)
    probe: Dict[str, Any] = {
        "score_dump_path": str(score_dump_path),
        "block_scores_path": str(block_scores_path),
        "score_dump_rows": len(rows),
        "block_scores_rows": len(block_rows),
        "required_fields": list(REQUIRED_SCORE_FIELDS),
        "per_layer_score_available": False,
        "virtual_scores_available": False,
        "missing_fields": [],
        "available_layers": sorted({int(r.get("layer_id", -1)) for r in rows if r.get("layer_id") is not None}),
        "sample_layer_coverage": {},
        "instrumentation_gaps": [],
    }

    if not rows:
        probe["instrumentation_gaps"].append("uasr_score_dump.jsonl empty or missing")
        return probe

    sample_rows: Dict[Key, List[JsonRow]] = defaultdict(list)
    for row in rows:
        if str(row.get("phase", "")) != "prefill":
            continue
        key = (str(row.get("task", "")), int(row.get("sample_index", -1)))
        if key[0] and key[1] >= 0:
            sample_rows[key].append(row)

    missing_fields: Set[str] = set()
    for row in rows[: min(20, len(rows))]:
        for field in REQUIRED_SCORE_FIELDS:
            if field not in row:
                missing_fields.add(field)
            elif field in ("all_candidate_block_ids", "all_candidate_scores") and not row.get(field):
                missing_fields.add(f"{field}_empty")

    probe["missing_fields"] = sorted(missing_fields)
    if missing_fields:
        probe["instrumentation_gaps"].append(f"score dump missing fields: {sorted(missing_fields)}")
        return probe

    layer_set = set(PRUNE_LAYERS)
    covered_samples = 0
    for key in keys:
        layer_ids = {
            int(r.get("layer_id", -1))
            for r in sample_rows.get(key, [])
            if int(r.get("layer_id", -1)) in layer_set
        }
        has_scores = False
        for row in sample_rows.get(key, []):
            if int(row.get("layer_id", -1)) not in layer_set:
                continue
            ids = row.get("all_candidate_block_ids") or []
            scores = row.get("all_candidate_scores") or []
            if ids and scores and len(ids) == len(scores):
                has_scores = True
                break
        probe["sample_layer_coverage"][f"{key[0]}:{key[1]}"] = {
            "layers_present": sorted(layer_ids),
            "has_per_block_scores": has_scores,
        }
        if layer_set.issubset(layer_ids) and has_scores:
            covered_samples += 1

    probe["samples_with_full_layer_scores"] = covered_samples
    probe["expected_samples"] = len(keys)
    probe["per_layer_score_available"] = covered_samples == len(keys)
    probe["virtual_scores_available"] = False
    if block_rows:
        sample_block = block_rows[0]
        if sample_block.get("block_ids") and sample_block.get("scores"):
            probe["block_scores_sampled_available"] = True
    return probe


def minmax_norm(values: Dict[int, float]) -> Dict[int, float]:
    if not values:
        return {}
    finite = {k: float(v) for k, v in values.items() if math.isfinite(float(v))}
    if not finite:
        return {k: 0.0 for k in values}
    vmin = min(finite.values())
    vmax = max(finite.values())
    if abs(vmax - vmin) < 1e-12:
        return {k: 0.5 for k in values}
    return {k: (float(v) - vmin) / (vmax - vmin) for k, v in values.items()}


def rank_percentile_map(scores: Dict[int, float]) -> Dict[int, float]:
    ordered = sorted(
        ((bid, float(score)) for bid, score in scores.items() if math.isfinite(float(score))),
        key=lambda item: item[1],
        reverse=True,
    )
    n = len(ordered)
    if n == 0:
        return {}
    if n == 1:
        return {ordered[0][0]: 1.0}
    out: Dict[int, float] = {}
    for rank, (block_id, _) in enumerate(ordered, 1):
        out[int(block_id)] = 1.0 - (rank - 1) / (n - 1)
    return out


def topk_set(scores: Dict[int, float], k: int) -> Set[int]:
    ordered = sorted(
        ((bid, float(score)) for bid, score in scores.items() if math.isfinite(float(score))),
        key=lambda item: item[1],
        reverse=True,
    )
    return {int(bid) for bid, _ in ordered[: max(0, int(k))]}


def selected_jaccard(left: Set[int], right: Set[int]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def token_count(selected: Set[int], token_lengths: Sequence[int]) -> int:
    return sum(int(token_lengths[block_id]) for block_id in selected if 0 <= block_id < len(token_lengths))


def block_token_map(block_texts: Sequence[str]) -> Dict[int, Set[str]]:
    return {idx: _gate_content_tokens(text) for idx, text in enumerate(block_texts)}


def block_query_overlaps(block_texts: Sequence[str], query: str) -> Dict[int, float]:
    query_tokens = _gate_content_tokens(query)
    if not query_tokens:
        return {idx: 0.0 for idx in range(len(block_texts))}
    overlaps: Dict[int, float] = {}
    for idx, text in enumerate(block_texts):
        tokens = _gate_content_tokens(text)
        overlaps[idx] = len(tokens.intersection(query_tokens)) / len(query_tokens) if tokens else 0.0
    return overlaps


def load_layer_records(score_dump_path: Path, keys: Sequence[Key]) -> Dict[Key, Dict[int, JsonRow]]:
    rows = read_jsonl(score_dump_path)
    grouped: Dict[Key, Dict[int, JsonRow]] = defaultdict(dict)
    for row in rows:
        if str(row.get("phase", "")) != "prefill":
            continue
        key = (str(row.get("task", "")), int(row.get("sample_index", -1)))
        if key not in keys:
            continue
        layer_id = int(row.get("layer_id", -1))
        if layer_id in PRUNE_LAYERS:
            grouped[key][layer_id] = row
    return grouped


def layer_scores_from_record(record: JsonRow) -> Dict[int, float]:
    block_ids = [int(x) for x in (record.get("all_candidate_block_ids") or [])]
    scores = record.get("all_candidate_scores") or []
    out: Dict[int, float] = {}
    for block_id, score in zip(block_ids, scores):
        try:
            value = float(score)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            out[int(block_id)] = value
    return out


def compute_block_features(
    layer_records: Dict[int, JsonRow],
    budget_k: int,
) -> Tuple[Dict[int, Dict[str, float]], Set[int], List[str]]:
    gaps: List[str] = []
    for layer_id in PRUNE_LAYERS:
        if layer_id not in layer_records:
            gaps.append(f"missing_layer_{layer_id}")

    final_record = layer_records.get(FINAL_LAYER)
    if final_record is None:
        return {}, set(), gaps + ["missing_final_layer_record"]

    candidates = {int(x) for x in (final_record.get("all_candidate_block_ids") or [])}
    if not candidates:
        return {}, set(), gaps + ["empty_candidate_set"]

    per_layer_scores: Dict[int, Dict[int, float]] = {}
    per_layer_rank_pct: Dict[int, Dict[int, float]] = {}
    per_layer_topk: Dict[int, Set[int]] = {}
    for layer_id in PRUNE_LAYERS:
        record = layer_records.get(layer_id)
        if record is None:
            continue
        scores = layer_scores_from_record(record)
        per_layer_scores[layer_id] = scores
        per_layer_rank_pct[layer_id] = rank_percentile_map(scores)
        per_layer_topk[layer_id] = topk_set(scores, budget_k)

    block_features: Dict[int, Dict[str, float]] = {}
    for block_id in candidates:
        layer_score_values: List[float] = []
        layer_rank_values: List[float] = []
        consensus_count = 0
        for layer_id in PRUNE_LAYERS:
            scores = per_layer_scores.get(layer_id, {})
            if block_id in scores:
                layer_score_values.append(float(scores[block_id]))
            rank_pct = per_layer_rank_pct.get(layer_id, {})
            if block_id in rank_pct:
                layer_rank_values.append(float(rank_pct[block_id]))
            if block_id in per_layer_topk.get(layer_id, set()):
                consensus_count += 1
        if not layer_score_values or not layer_rank_values:
            continue
        mean_score = statistics.mean(layer_score_values)
        max_score = max(layer_score_values)
        score_std = statistics.pstdev(layer_score_values) if len(layer_score_values) > 1 else 0.0
        mean_rank_percentile = statistics.mean(layer_rank_values)
        rank_std = statistics.pstdev(layer_rank_values) if len(layer_rank_values) > 1 else 0.0
        spike_score = max_score - mean_score
        stability_score = -rank_std
        block_features[block_id] = {
            "mean_score": mean_score,
            "max_score": max_score,
            "score_std": score_std,
            "mean_rank_percentile": mean_rank_percentile,
            "rank_std": rank_std,
            "consensus_topk_count": float(consensus_count),
            "spike_score": spike_score,
            "stability_score": stability_score,
        }

    if not block_features:
        gaps.append("no_block_features_computed")
    return block_features, candidates, gaps


def compute_lc_scores(block_features: Dict[int, Dict[str, float]]) -> Dict[int, float]:
    mean_rank_signal = minmax_norm({bid: feat["mean_rank_percentile"] for bid, feat in block_features.items()})
    consensus_signal = minmax_norm({bid: feat["consensus_topk_count"] for bid, feat in block_features.items()})
    rank_std_signal = minmax_norm({bid: feat["rank_std"] for bid, feat in block_features.items()})
    spike_signal = minmax_norm({bid: feat["spike_score"] for bid, feat in block_features.items()})
    lc_scores: Dict[int, float] = {}
    for block_id, feat in block_features.items():
        lc_scores[block_id] = (
            mean_rank_signal.get(block_id, 0.0)
            + 0.5 * consensus_signal.get(block_id, 0.0)
            - 0.25 * rank_std_signal.get(block_id, 0.0)
            - 0.25 * spike_signal.get(block_id, 0.0)
        )
    return lc_scores


def lc_select_topk(
    candidates: Set[int],
    lc_scores: Dict[int, float],
    budget_k: int,
    original_selected: Set[int],
) -> Tuple[Set[int], str]:
    if budget_k <= 0:
        return set(), "empty_budget"
    if budget_k >= len(candidates):
        return set(candidates), "all_candidates_kept"
    ranked = sorted(
        ((bid, lc_scores.get(bid, float("-inf"))) for bid in candidates),
        key=lambda item: (item[1], -item[0]),
        reverse=True,
    )
    lc_selected = {int(bid) for bid, _ in ranked[:budget_k]}
    if lc_selected == original_selected:
        return lc_selected, "lc_equals_original"
    return lc_selected, "lc_reordered_topk"


def aggregate_selection_metric(
    selected: Set[int],
    block_features: Dict[int, Dict[str, float]],
    metric: str,
) -> float:
    values = [float(block_features[bid][metric]) for bid in selected if bid in block_features]
    return float(statistics.mean(values)) if values else 0.0


def decide_lc_audit(rows: Sequence[Mapping[str, Any]], probe: Mapping[str, Any]) -> Dict[str, Any]:
    if not probe.get("per_layer_score_available"):
        return {
            "decision": "BUG_STOP_FIX_REQUIRED",
            "reasons": probe.get("instrumentation_gaps", ["per_layer_score_unavailable"]),
            "worth_tiny_quality_smoke": False,
            "required_instrumentation": [
                "gate_audit with --uasr_score_dump_path on sliminfer_original",
                "per-layer all_candidate_block_ids + all_candidate_scores at prune layers 9/18/26",
                "optional --save_block_scores sampled as secondary validation",
            ],
        }

    total = len(rows)
    if total == 0:
        return {"decision": "BUG_STOP_FIX_REQUIRED", "reasons": ["no_audit_rows"]}

    bug_reasons: List[str] = []
    budget_violations = sum(1 for row in rows if not parse_bool(row.get("budget_preserved")))
    fallback_count = sum(1 for row in rows if parse_bool(row.get("fallback_used")))
    missing_selection = sum(1 for row in rows if not row.get("original_selected_block_ids"))
    if missing_selection:
        bug_reasons.append(f"missing_original_selection={missing_selection}")
    if budget_violations:
        bug_reasons.append(f"budget_violation_count={budget_violations}")
    if fallback_count > total // 2:
        bug_reasons.append(f"fallback_count={fallback_count}")

    if bug_reasons:
        return {
            "decision": "BUG_STOP_FIX_REQUIRED",
            "reasons": bug_reasons,
            "worth_tiny_quality_smoke": False,
        }

    changed = [row for row in rows if parse_bool(row.get("selection_changed"))]
    trigger_count = sum(1 for row in rows if parse_bool(row.get("lc_triggered")))
    jaccards = [float(row["selected_jaccard_vs_original"]) for row in rows]
    median_jaccard = statistics.median(jaccards) if jaccards else 1.0

    orig_consensus = statistics.mean(float(row["original_layer_consensus"]) for row in rows)
    lc_consensus = statistics.mean(float(row["lc_layer_consensus"]) for row in rows)
    orig_spike = statistics.mean(float(row["original_spike_score"]) for row in rows)
    lc_spike = statistics.mean(float(row["lc_spike_score"]) for row in rows)
    orig_rank_std = statistics.mean(float(row["original_rank_std"]) for row in rows)
    lc_rank_std = statistics.mean(float(row["lc_rank_std"]) for row in rows)
    redundancy_delta = statistics.mean(float(row["redundancy_delta"]) for row in rows)

    no_go: List[str] = []
    if trigger_count == 0:
        no_go.append("selection_no_op")
    if median_jaccard < 0.05:
        no_go.append(f"median_jaccard_too_low={median_jaccard:.3f}")
    if median_jaccard > 0.98 and trigger_count > 0:
        no_go.append("almost_all_changed")
    if lc_consensus <= orig_consensus + 0.01:
        no_go.append(f"lc_consensus_not_higher={lc_consensus:.4f}<={orig_consensus:.4f}")
    if lc_spike > orig_spike + 0.02:
        no_go.append(f"lc_spike_not_lower={lc_spike:.4f}>{orig_spike:.4f}")
    if redundancy_delta > 0.05:
        no_go.append(f"redundancy_increased={redundancy_delta:.4f}")

    go_ready = (
        trigger_count >= max(5, total // 10)
        and trigger_count < total
        and budget_violations == 0
        and fallback_count == 0
        and 0.05 <= median_jaccard <= 0.95
        and lc_consensus > orig_consensus + 0.02
        and lc_spike <= orig_spike
        and lc_rank_std <= orig_rank_std + 0.02
        and redundancy_delta <= 0.03
    )
    if go_ready:
        return {
            "decision": "LC_SELECTION_AUDIT_GO",
            "reasons": [
                f"trigger_count={trigger_count}/{total}",
                f"median_jaccard={median_jaccard:.3f}",
                f"lc_consensus_gain={lc_consensus - orig_consensus:+.4f}",
                f"lc_spike_delta={lc_spike - orig_spike:+.4f}",
            ],
            "worth_tiny_quality_smoke": True,
        }

    weak_ready = trigger_count > 0 and budget_violations == 0 and lc_consensus > orig_consensus
    if weak_ready and not no_go:
        return {
            "decision": "LC_SELECTION_AUDIT_WEAK",
            "reasons": [
                f"trigger_count={trigger_count}/{total}",
                f"median_jaccard={median_jaccard:.3f}",
                f"lc_consensus_gain={lc_consensus - orig_consensus:+.4f}",
            ],
            "worth_tiny_quality_smoke": trigger_count >= 3,
        }

    return {
        "decision": "LC_SELECTION_AUDIT_NO_GO",
        "reasons": no_go or [f"trigger_count={trigger_count}/{total}"],
        "worth_tiny_quality_smoke": False,
    }


def build_case_study(rows: Sequence[Mapping[str, Any]]) -> str:
    changed = sorted(
        [row for row in rows if parse_bool(row.get("selection_changed"))],
        key=lambda row: (
            -float(row.get("lc_layer_consensus", 0)) + float(row.get("original_layer_consensus", 0)),
            -float(row.get("coverage_delta", 0) if "coverage_delta" in row else 0),
        ),
    )[:5]
    lines = ["# Layer-Consensus SlimInfer Case Study", ""]
    for row in changed:
        lines.extend([
            f"## {row['task']}:{row['sample_index']} ({row['sample_id']})",
            f"- lc_trigger_reason: {row.get('lc_trigger_reason')}",
            f"- jaccard vs original: {float(row['selected_jaccard_vs_original']):.3f}",
            f"- layer consensus: {float(row['original_layer_consensus']):.3f} -> {float(row['lc_layer_consensus']):.3f}",
            f"- spike score: {float(row['original_spike_score']):.3f} -> {float(row['lc_spike_score']):.3f}",
            f"- rank std: {float(row['original_rank_std']):.3f} -> {float(row['lc_rank_std']):.3f}",
            f"- removed: {row.get('removed_block_ids')}",
            f"- added: {row.get('added_block_ids')}",
            f"- removed preview: {row.get('removed_block_text_preview', '')}",
            f"- added preview: {row.get('added_block_text_preview', '')}",
            "",
        ])
    if not changed:
        lines.append("_No selection changes observed._")
        lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--longbench_dir", type=Path, default=DEFAULT_LONGBENCH_DIR)
    parser.add_argument("--output_root", type=Path, default=Path("results"))
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--skip_gate_run", action="store_true")
    parser.add_argument("--score_dump_path", type=Path, default=None)
    parser.add_argument("--used_blocks_path", type=Path, default=None)
    parser.add_argument("--block_scores_path", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    keys = heldout_keys()
    if len(keys) != 128:
        raise RuntimeError(f"expected 128 held-out keys, got {len(keys)}")
    sample_spec, expected_indices = build_sample_spec()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (REPO_ROOT / args.output_root / f"lc_sliminfer_selection_audit_{timestamp}")
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    gate_dir = output_dir / "run_sliminfer_original_gate_audit"
    score_dump_path = args.score_dump_path or gate_dir / "uasr_score_dump.jsonl"
    used_blocks_path = args.used_blocks_path or gate_dir / "used_blocks.jsonl"
    block_scores_path = args.block_scores_path or gate_dir / "block_scores.jsonl"

    if args.skip_gate_run:
        if not score_dump_path.exists():
            raise RuntimeError(f"--skip_gate_run but score dump missing: {score_dump_path}")
    else:
        paths = run_score_instrumentation(args, output_dir, sample_spec)
        score_dump_path = paths["score_dump_path"]
        used_blocks_path = paths["used_blocks_path"]
        block_scores_path = paths["block_scores_path"]

    probe = probe_per_layer_score_availability(score_dump_path, block_scores_path, keys)
    write_json(output_dir / "lc_score_probe.json", probe)

    if not probe.get("per_layer_score_available"):
        decision = decide_lc_audit([], probe)
        write_json(output_dir / "run_config.json", {
            "sample_spec": sample_spec,
            "expected_indices": expected_indices,
            "prune_layers": PRUNE_LAYERS,
            "probe": probe,
            "quality_generation_performed": False,
            "scorer_invoked": False,
        })
        report = [
            "# Layer-Consensus SlimInfer Selection Audit",
            "",
            f"- Decision: **{decision['decision']}**",
            "",
            "## Per-layer score availability",
            f"- Per-layer score available: **{probe.get('per_layer_score_available')}**",
            f"- Score dump rows: **{probe.get('score_dump_rows', 0)}**",
            f"- Available layers in dump: **{probe.get('available_layers', [])}**",
            f"- Samples with full layer scores: **{probe.get('samples_with_full_layer_scores', 0)}/{probe.get('expected_samples', 128)}**",
            "",
            "## Instrumentation gaps",
        ]
        report.extend(f"- {gap}" for gap in probe.get("instrumentation_gaps", []))
        report.extend([
            "",
            "## Required instrumentation",
        ])
        for item in decision.get("required_instrumentation", []):
            report.append(f"- {item}")
        report.append("")
        (output_dir / "lc_audit_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
        (output_dir / "lc_decision.md").write_text(
            f"# Decision: {decision['decision']}\n\n" + "\n".join(f"- {r}" for r in decision.get("reasons", [])) + "\n",
            encoding="utf-8",
        )
        write_json(output_dir / "lc_decision.json", decision)
        print("\n".join(report))
        print(f"Wrote results to {output_dir}")
        return

    layer_records_by_key = load_layer_records(score_dump_path, keys)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), trust_remote_code=True)
    dataset2prompt = load_dataset2prompt()
    samples_cache = {task: load_samples(args.longbench_dir, task) for task in TASKS}

    coverage_rows: List[Dict[str, Any]] = []
    for layer_id in PRUNE_LAYERS:
        present = sum(1 for key in keys if layer_id in layer_records_by_key.get(key, {}))
        coverage_rows.append({
            "layer_id": layer_id,
            "samples_with_record": present,
            "sample_fraction": present / len(keys),
            "field_available": "all_candidate_scores",
        })
    write_csv(
        output_dir / "lc_layer_score_coverage.csv",
        coverage_rows,
        ["layer_id", "samples_with_record", "sample_fraction", "field_available"],
    )

    audit_rows: List[Dict[str, Any]] = []
    for task, sample_index in keys:
        key = (task, sample_index)
        sample = samples_cache[task][sample_index]
        query_text, block_texts, token_lengths = build_runner_prompt_bundle(
            tokenizer, dataset2prompt[task], sample, task,
        )
        layer_map = layer_records_by_key.get(key, {})
        final_record = layer_map[FINAL_LAYER]
        original_selected = {int(x) for x in (final_record.get("selected_block_ids") or [])}
        budget_k = int(final_record.get("budget_k", len(original_selected)) or len(original_selected))
        if not original_selected and budget_k > 0:
            fallback_reason = "missing_original_selected_block_ids"
            audit_rows.append({
                "task": task,
                "sample_index": sample_index,
                "sample_id": final_record.get("sample_id", sample.get("_id", sample_index)),
                "fallback_used": True,
                "lc_trigger_reason": fallback_reason,
            })
            continue

        block_features, candidates, gaps = compute_block_features(layer_map, budget_k)
        if gaps or not block_features:
            audit_rows.append({
                "task": task,
                "sample_index": sample_index,
                "sample_id": final_record.get("sample_id", sample.get("_id", sample_index)),
                "fallback_used": True,
                "lc_trigger_reason": ";".join(gaps) if gaps else "feature_compute_failed",
            })
            continue

        lc_scores = compute_lc_scores(block_features)
        lc_selected, select_reason = lc_select_topk(candidates, lc_scores, budget_k, original_selected)
        removed = sorted(original_selected - lc_selected)
        added = sorted(lc_selected - original_selected)
        selection_changed = bool(removed or added)
        jaccard = selected_jaccard(original_selected, lc_selected)

        token_map = block_token_map(block_texts)
        query_overlaps = block_query_overlaps(block_texts, query_text)
        original_redundancy = _dc_selected_redundancy(original_selected, token_map)
        lc_redundancy = _dc_selected_redundancy(lc_selected, token_map)
        original_query_overlap = float(
            sum(query_overlaps.get(bid, 0.0) for bid in original_selected) / max(1, len(original_selected))
        )
        lc_query_overlap = float(
            sum(query_overlaps.get(bid, 0.0) for bid in lc_selected) / max(1, len(lc_selected))
        )
        original_token_count = token_count(original_selected, token_lengths)
        lc_token_count = token_count(lc_selected, token_lengths)
        budget_preserved = len(lc_selected) == len(original_selected) and lc_token_count == original_token_count

        num_layers = len(PRUNE_LAYERS)
        original_layer_consensus = aggregate_selection_metric(original_selected, block_features, "consensus_topk_count") / num_layers
        lc_layer_consensus = aggregate_selection_metric(lc_selected, block_features, "consensus_topk_count") / num_layers

        row = {
            "task": task,
            "sample_index": sample_index,
            "sample_id": final_record.get("sample_id", sample.get("_id", sample_index)),
            "original_selected_block_ids": sorted(original_selected),
            "lc_selected_block_ids": sorted(lc_selected),
            "selected_jaccard_vs_original": jaccard,
            "selection_changed": selection_changed,
            "original_mean_score": aggregate_selection_metric(original_selected, block_features, "mean_score"),
            "lc_mean_score": aggregate_selection_metric(lc_selected, block_features, "mean_score"),
            "original_layer_consensus": original_layer_consensus,
            "lc_layer_consensus": lc_layer_consensus,
            "original_rank_std": aggregate_selection_metric(original_selected, block_features, "rank_std"),
            "lc_rank_std": aggregate_selection_metric(lc_selected, block_features, "rank_std"),
            "original_spike_score": aggregate_selection_metric(original_selected, block_features, "spike_score"),
            "lc_spike_score": aggregate_selection_metric(lc_selected, block_features, "spike_score"),
            "original_redundancy": original_redundancy,
            "lc_redundancy": lc_redundancy,
            "redundancy_delta": lc_redundancy - original_redundancy,
            "original_query_overlap": original_query_overlap,
            "lc_query_overlap": lc_query_overlap,
            "query_overlap_delta": lc_query_overlap - original_query_overlap,
            "selected_token_count_original": original_token_count,
            "selected_token_count_lc": lc_token_count,
            "budget_preserved": budget_preserved,
            "over_budget": lc_token_count > original_token_count,
            "fallback_used": False,
            "lc_triggered": selection_changed and lc_layer_consensus >= original_layer_consensus,
            "lc_trigger_reason": select_reason if selection_changed else "selection_unchanged",
            "removed_block_ids": removed,
            "added_block_ids": added,
            "removed_block_text_preview": preview_text(block_texts, removed),
            "added_block_text_preview": preview_text(block_texts, added),
            "budget_k": budget_k,
            "candidate_count": len(candidates),
        }
        audit_rows.append(row)

    valid_rows = [row for row in audit_rows if not parse_bool(row.get("fallback_used"))]
    decision = decide_lc_audit(valid_rows, probe)

    audit_fields = [
        "task", "sample_index", "sample_id",
        "original_selected_block_ids", "lc_selected_block_ids",
        "selected_jaccard_vs_original", "selection_changed",
        "original_mean_score", "lc_mean_score",
        "original_layer_consensus", "lc_layer_consensus",
        "original_rank_std", "lc_rank_std",
        "original_spike_score", "lc_spike_score",
        "original_redundancy", "lc_redundancy", "redundancy_delta",
        "original_query_overlap", "lc_query_overlap", "query_overlap_delta",
        "selected_token_count_original", "selected_token_count_lc",
        "budget_preserved", "over_budget", "fallback_used",
        "lc_triggered", "lc_trigger_reason",
    ]
    write_csv(output_dir / "lc_selection_audit.csv", audit_rows, audit_fields)

    group_rows: List[Dict[str, Any]] = []
    for group in TASKS + ["all"]:
        subset = audit_rows if group == "all" else [row for row in audit_rows if row["task"] == group]
        if not subset:
            continue
        valid = [row for row in subset if not parse_bool(row.get("fallback_used"))]
        jaccards = [float(row["selected_jaccard_vs_original"]) for row in valid] if valid else [1.0]
        group_rows.append({
            "group": group,
            "sample_count": len(subset),
            "valid_sample_count": len(valid),
            "fallback_count": sum(1 for row in subset if parse_bool(row.get("fallback_used"))),
            "budget_violation_count": sum(1 for row in valid if not parse_bool(row.get("budget_preserved"))),
            "selection_changed_count": sum(1 for row in valid if parse_bool(row.get("selection_changed"))),
            "lc_trigger_count": sum(1 for row in valid if parse_bool(row.get("lc_triggered"))),
            "mean_original_layer_consensus": statistics.mean(float(row["original_layer_consensus"]) for row in valid) if valid else 0.0,
            "mean_lc_layer_consensus": statistics.mean(float(row["lc_layer_consensus"]) for row in valid) if valid else 0.0,
            "mean_original_rank_std": statistics.mean(float(row["original_rank_std"]) for row in valid) if valid else 0.0,
            "mean_lc_rank_std": statistics.mean(float(row["lc_rank_std"]) for row in valid) if valid else 0.0,
            "mean_original_spike_score": statistics.mean(float(row["original_spike_score"]) for row in valid) if valid else 0.0,
            "mean_lc_spike_score": statistics.mean(float(row["lc_spike_score"]) for row in valid) if valid else 0.0,
            "median_selected_jaccard": statistics.median(jaccards) if jaccards else 1.0,
            "mean_redundancy_delta": statistics.mean(float(row["redundancy_delta"]) for row in valid) if valid else 0.0,
            "mean_query_overlap_delta": statistics.mean(float(row["query_overlap_delta"]) for row in valid) if valid else 0.0,
        })
    write_csv(output_dir / "lc_group_summary.csv", group_rows, list(group_rows[0].keys()))

    all_summary = next(row for row in group_rows if row["group"] == "all")
    hotpot_summary = next((row for row in group_rows if row["group"] == "hotpotqa"), {})
    musique_summary = next((row for row in group_rows if row["group"] == "musique"), {})

    case_study = build_case_study(valid_rows)
    (output_dir / "lc_case_study.md").write_text(case_study, encoding="utf-8")

    report_lines = [
        "# Layer-Consensus SlimInfer Selection Audit",
        "",
        f"- Output: `{output_dir}`",
        f"- Decision: **{decision.get('decision')}**",
        "",
        "## Instrumentation",
        f"1. Per-layer score fields available: **{probe.get('per_layer_score_available')}**",
        f"2. Available prune layers in dump: **{probe.get('available_layers', [])}**",
        f"3. Samples with full layer scores: **{probe.get('samples_with_full_layer_scores', 0)}/128**",
        f"4. virtual_scores available: **{probe.get('virtual_scores_available')}** (not emitted by current instrumentation)",
        "",
        "## Audit summary",
        f"5. Sample count: **{all_summary['sample_count']}** (valid={all_summary['valid_sample_count']})",
        f"6. LC trigger / selection changed: **{all_summary['lc_trigger_count']} / {all_summary['selection_changed_count']}**",
        f"7. Budget violation count: **{all_summary['budget_violation_count']}**",
        f"8. Fallback count: **{all_summary['fallback_count']}**",
        "",
        "## Layer consensus / stability",
        f"9. Original vs LC layer consensus: **{all_summary['mean_original_layer_consensus']:.4f} -> {all_summary['mean_lc_layer_consensus']:.4f}**",
        f"10. Original vs LC rank std: **{all_summary['mean_original_rank_std']:.4f} -> {all_summary['mean_lc_rank_std']:.4f}**",
        f"11. Original vs LC spike score: **{all_summary['mean_original_spike_score']:.4f} -> {all_summary['mean_lc_spike_score']:.4f}**",
        f"12. Median selected Jaccard: **{all_summary['median_selected_jaccard']:.4f}**",
        f"13. Mean redundancy delta: **{all_summary['mean_redundancy_delta']:+.4f}**",
        f"14. Mean query overlap delta: **{all_summary['mean_query_overlap_delta']:+.4f}**",
        "",
        "## Per-task",
        f"- HotpotQA trigger/changed: **{hotpot_summary.get('lc_trigger_count', 0)} / {hotpot_summary.get('selection_changed_count', 0)}**",
        f"- MuSiQue trigger/changed: **{musique_summary.get('lc_trigger_count', 0)} / {musique_summary.get('selection_changed_count', 0)}**",
        "",
        "## Tiny quality smoke",
        f"- Worth tiny quality smoke: **{decision.get('worth_tiny_quality_smoke', False)}**",
        "",
        "## Decision reasons",
    ]
    report_lines.extend(f"- {reason}" for reason in decision.get("reasons", []))
    report_lines.append("")
    (output_dir / "lc_audit_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    (output_dir / "lc_decision.md").write_text(
        f"# Decision: {decision.get('decision')}\n\n"
        + "\n".join(f"- {reason}" for reason in decision.get("reasons", []))
        + f"\n\n- Worth tiny quality smoke: {decision.get('worth_tiny_quality_smoke', False)}\n",
        encoding="utf-8",
    )

    write_json(output_dir / "run_config.json", {
        "sample_spec": sample_spec,
        "expected_indices": expected_indices,
        "prune_layers": PRUNE_LAYERS,
        "lc_score_formula": "mean_rank_signal + 0.5*consensus_signal - 0.25*rank_std - 0.25*spike_score",
        "score_dump_path": str(score_dump_path),
        "probe": probe,
        "quality_generation_performed": False,
        "scorer_invoked": False,
    })
    write_json(output_dir / "lc_decision.json", decision)

    print("\n".join(report_lines))
    print(f"Wrote results to {output_dir}")


if __name__ == "__main__":
    main()
