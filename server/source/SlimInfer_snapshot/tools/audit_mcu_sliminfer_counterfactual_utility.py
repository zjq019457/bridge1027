#!/usr/bin/env python3
"""MCU-SlimInfer counterfactual utility audit (tiny 12-sample prefill probes).

Runs sliminfer_original gate_audit for block selection, then performs prefill-only
forward counterfactual probes (no generation, no scorer, no gold).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

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
    parse_bool,
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
METHOD_ORIGINAL = "sliminfer_original"
MAX_MODEL_LEN = 32768
MAX_POOL_SIZE = 32
MAX_REMOVE_PROBES = 8
MAX_SWAP_PROBES = 12
MAX_REPLACE_FRAC = 0.20
UTILITY_MARGIN_MIN = 0.15
PROBE_SLOW_SAMPLE_SEC = 600.0
PROBE_SLOW_SINGLE_SEC = 90.0

MCU_SAMPLE_INDICES: Dict[str, List[int]] = {
    "hotpotqa": [0, 1, 2, 3, 4, 5],
    "musique": [0, 1, 2, 3, 4, 20],
}

Key = Tuple[str, int]
JsonRow = Dict[str, Any]

AUDIT_FIELDS = [
    "task", "sample_index", "sample_id",
    "original_selected_block_ids", "mcu_selected_block_ids",
    "candidate_pool_size", "num_counterfactual_probes",
    "selected_jaccard_vs_original", "selection_changed",
    "removed_block_ids", "added_block_ids",
    "removed_block_utility", "added_block_utility", "utility_margin",
    "hidden_change_mean", "logit_kl_mean", "entropy_change_mean",
    "top_logit_margin_change_mean", "redundancy_delta", "query_overlap_delta",
    "selected_token_count_original", "selected_token_count_mcu",
    "budget_preserved", "over_budget", "fallback_used", "fallback_reason",
    "probe_runtime_ms",
]


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


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


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


def mcu_keys() -> List[Key]:
    return [(task, idx) for task in TASKS for idx in MCU_SAMPLE_INDICES[task]]


def build_sample_spec() -> Tuple[str, Dict[str, List[int]]]:
    spec = ";".join(f"{task}:" + ",".join(str(i) for i in MCU_SAMPLE_INDICES[task]) for task in TASKS)
    return spec, dict(MCU_SAMPLE_INDICES)


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
) -> Tuple[torch.Tensor, str, List[str], List[int]]:
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
    return input_ids, query_text, block_texts, token_lengths


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
        "--longbench_subset_name", "mcu_sliminfer_counterfactual_audit_tiny12",
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
        "--debug_used_block_full_layers", "27",
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


def run_gate_audit(args: argparse.Namespace, output_dir: Path, sample_spec: str) -> Dict[str, Path]:
    run_dir = output_dir / "run_sliminfer_original_gate_audit"
    run_dir.mkdir(parents=True, exist_ok=True)
    score_dump_path = run_dir / "uasr_score_dump.jsonl"
    used_blocks_path = run_dir / "used_blocks.jsonl"
    block_scores_path = run_dir / "block_scores.jsonl"
    dc_sink_path = run_dir / "dc_selection_audit_sink.jsonl"
    for path in (score_dump_path, used_blocks_path, block_scores_path, dc_sink_path):
        path.write_text("", encoding="utf-8")
    cmd = base_gate_audit_cmd(args, run_dir, sample_spec, score_dump_path, used_blocks_path, block_scores_path, dc_sink_path)
    write_json(output_dir / "mcu_gate_audit_command.json", {"command": cmd})
    run_command(cmd, REPO_ROOT, output_dir / "mcu_gate_audit_stdout.log", output_dir / "mcu_gate_audit_stderr.log")
    return {
        "score_dump_path": score_dump_path,
        "used_blocks_path": used_blocks_path,
        "block_scores_path": block_scores_path,
    }


def load_final_layer_records(score_dump_path: Path, keys: Sequence[Key]) -> Dict[Key, JsonRow]:
    rows = read_jsonl(score_dump_path)
    grouped: Dict[Key, JsonRow] = {}
    for row in rows:
        if str(row.get("phase", "")) != "prefill":
            continue
        key = (str(row.get("task", "")), int(row.get("sample_index", -1)))
        if key not in keys:
            continue
        if int(row.get("layer_id", -1)) == FINAL_LAYER:
            grouped[key] = row
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


def find_query_span(input_ids: torch.Tensor, tokenizer: AutoTokenizer, query_text: str) -> Tuple[int, int]:
    query_ids = tokenizer(query_text, add_special_tokens=False).input_ids
    if not query_ids:
        return max(0, int(input_ids.shape[0]) - 128), int(input_ids.shape[0])
    q_tensor = torch.tensor(query_ids, dtype=input_ids.dtype, device=input_ids.device)
    seq_len = int(input_ids.shape[0])
    q_len = int(q_tensor.shape[0])
    for start in range(seq_len - q_len, -1, -1):
        if torch.equal(input_ids[start:start + q_len], q_tensor):
            return start, seq_len
    return max(0, seq_len - min(128, seq_len)), seq_len


def protected_block_ids(seq_len: int, block_size: int, query_start: int) -> Set[int]:
    num_blocks = (seq_len + block_size - 1) // block_size
    protected: Set[int] = set()
    for bid in range(num_blocks):
        bstart = bid * block_size
        bend = min(seq_len, (bid + 1) * block_size)
        if bend > query_start:
            protected.add(bid)
    return protected


def compact_input_ids(input_ids: torch.Tensor, active_blocks: Set[int], block_size: int) -> torch.Tensor:
    indices: List[int] = []
    seq_len = int(input_ids.shape[0])
    num_blocks = (seq_len + block_size - 1) // block_size
    for bid in range(num_blocks):
        if bid not in active_blocks:
            continue
        start = bid * block_size
        end = min(seq_len, (bid + 1) * block_size)
        indices.extend(range(start, end))
    if not indices:
        return input_ids[:0]
    return input_ids[torch.tensor(indices, dtype=torch.long)]


def build_candidate_pool(
    original_selected: Set[int],
    all_scores: Dict[int, float],
    query_overlaps: Dict[int, float],
    protected: Set[int],
    max_pool: int = MAX_POOL_SIZE,
) -> Set[int]:
    all_blocks = set(all_scores.keys()) | original_selected
    pruned = sorted(all_blocks - original_selected)
    selected_low = sorted(
        original_selected,
        key=lambda bid: all_scores.get(bid, float("-inf")),
    )[: max(4, max_pool // 4)]

    pruned_near: List[int] = []
    if pruned:
        pruned_by_score = sorted(pruned, key=lambda bid: all_scores.get(bid, float("-inf")), reverse=True)
        pruned_near = pruned_by_score[: max(8, max_pool // 2)]
    overlap_ranked = sorted(pruned, key=lambda bid: query_overlaps.get(bid, 0.0), reverse=True)
    overlap_top = overlap_ranked[: max(6, max_pool // 3)]

    pool: List[int] = []
    seen: Set[int] = set()
    for bid in list(selected_low) + list(pruned_near) + list(overlap_top):
        if bid in seen:
            continue
        seen.add(bid)
        pool.append(bid)
        if len(pool) >= max_pool:
            break
    for bid in sorted(protected):
        if bid not in seen:
            if len(pool) >= max_pool:
                pool.pop(0)
            pool.append(bid)
            seen.add(bid)
    return set(pool[:max_pool])


@dataclass
class ProbeState:
    last_hidden: torch.Tensor
    query_hidden_mean: torch.Tensor
    next_logits: torch.Tensor
    value_norm: float
    attention_mass: float


@dataclass
class ProbeMetrics:
    last_token_hidden_l2_change: float = 0.0
    query_token_hidden_l2_change: float = 0.0
    next_token_logit_kl: float = 0.0
    next_token_entropy_change: float = 0.0
    top_logit_margin_change: float = 0.0
    value_norm_change: float = 0.0
    attention_mass_change: float = 0.0
    probe_runtime_ms: float = 0.0


def _entropy(logits: torch.Tensor) -> float:
    probs = F.softmax(logits.float(), dim=-1)
    log_probs = F.log_softmax(logits.float(), dim=-1)
    return float(-(probs * log_probs).sum().item())


def _top_margin(logits: torch.Tensor) -> float:
    top2 = torch.topk(logits.float(), k=min(2, logits.shape[-1])).values
    if top2.numel() < 2:
        return 0.0
    return float(top2[0] - top2[1])


def run_forward_probe(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    query_start_in_compact: int,
    *,
    capture_attention: bool = True,
) -> ProbeState:
    device = next(model.parameters()).device
    compact_len = int(input_ids.shape[0])
    if compact_len == 0:
        raise RuntimeError("empty compact input for probe")

    with torch.inference_mode():
        outputs = model(
            input_ids.unsqueeze(0).to(device),
            use_cache=False,
            output_hidden_states=True,
            output_attentions=capture_attention,
            return_dict=True,
        )
    if not outputs.hidden_states:
        raise RuntimeError("model forward did not return hidden_states")
    hidden = outputs.hidden_states[-1][0].float()
    logits = outputs.logits[0].float()
    last_hidden = hidden[-1]
    q_start = max(0, min(query_start_in_compact, compact_len - 1))
    query_hidden_mean = hidden[q_start:].mean(dim=0)

    value_norm = float(hidden.norm(dim=-1).mean().item())
    attention_mass = 0.0
    if capture_attention and outputs.attentions:
        last_attn = outputs.attentions[-1][0].float()
        q_window = min(max(1, compact_len - q_start), last_attn.shape[-2])
        attention_mass = float(last_attn[:, -q_window:, :].mean().item())

    return ProbeState(
        last_hidden=last_hidden.cpu(),
        query_hidden_mean=query_hidden_mean.cpu(),
        next_logits=logits[-1].cpu(),
        value_norm=value_norm,
        attention_mass=attention_mass,
    )


def compare_probe_states(baseline: ProbeState, probe: ProbeState, runtime_ms: float) -> ProbeMetrics:
    last_l2 = float(torch.norm(probe.last_hidden - baseline.last_hidden, p=2).item())
    query_l2 = float(torch.norm(probe.query_hidden_mean - baseline.query_hidden_mean, p=2).item())

    base_probs = F.softmax(baseline.next_logits.float(), dim=-1)
    probe_probs = F.softmax(probe.next_logits.float(), dim=-1)
    kl = float(F.kl_div(probe_probs.log(), base_probs, reduction="sum").item())

    base_entropy = _entropy(baseline.next_logits)
    probe_entropy = _entropy(probe.next_logits)
    entropy_change = probe_entropy - base_entropy
    margin_change = _top_margin(probe.next_logits) - _top_margin(baseline.next_logits)

    return ProbeMetrics(
        last_token_hidden_l2_change=last_l2,
        query_token_hidden_l2_change=query_l2,
        next_token_logit_kl=kl,
        next_token_entropy_change=entropy_change,
        top_logit_margin_change=margin_change,
        value_norm_change=probe.value_norm - baseline.value_norm,
        attention_mass_change=probe.attention_mass - baseline.attention_mass,
        probe_runtime_ms=runtime_ms,
    )


def zscore_map(values: Dict[int, float]) -> Dict[int, float]:
    finite = [float(v) for v in values.values() if math.isfinite(float(v))]
    if not finite:
        return {k: 0.0 for k in values}
    mean = statistics.mean(finite)
    std = statistics.pstdev(finite) if len(finite) > 1 else 0.0
    if std < 1e-12:
        return {k: 0.0 for k in values}
    return {k: (float(v) - mean) / std for k, v in values.items()}


def compute_block_utility(
    block_id: int,
    metrics: ProbeMetrics,
    redundancy: float,
    sample_z: Dict[str, Dict[int, float]],
) -> float:
    hidden_change = metrics.last_token_hidden_l2_change + 0.5 * metrics.query_token_hidden_l2_change
    entropy_reduction = -metrics.next_token_entropy_change
    return (
        sample_z["hidden"].get(block_id, 0.0)
        + sample_z["logit_kl"].get(block_id, 0.0)
        + 0.5 * sample_z["entropy_reduction"].get(block_id, 0.0)
        + 0.5 * sample_z["margin"].get(block_id, 0.0)
        - 0.5 * sample_z["redundancy"].get(block_id, 0.0)
    )


def mcu_select_blocks(
    original_selected: Set[int],
    block_utilities: Dict[int, float],
    budget_k: int,
    protected: Set[int],
    token_lengths: Sequence[int],
    *,
    max_replace_frac: float = MAX_REPLACE_FRAC,
    utility_margin_min: float = UTILITY_MARGIN_MIN,
) -> Tuple[Set[int], str, List[Tuple[int, int]], float]:
    if budget_k <= 0 or not original_selected:
        return set(original_selected), "empty_budget", [], 0.0

    selected_sorted = sorted(
        original_selected,
        key=lambda bid: (block_utilities.get(bid, float("inf")), bid),
    )
    pruned_candidates = sorted(
        ((bid, block_utilities[bid]) for bid in block_utilities if bid not in original_selected),
        key=lambda item: (-item[1], item[0]),
    )
    max_swaps = max(0, int(math.floor(max_replace_frac * budget_k)))
    if max_swaps == 0:
        return set(original_selected), "zero_swap_budget", [], 0.0

    current = set(original_selected)
    swaps: List[Tuple[int, int]] = []
    margins: List[float] = []

    removable = [bid for bid in selected_sorted if bid not in protected]
    for _ in range(max_swaps):
        if not removable or not pruned_candidates:
            break
        victim = removable.pop(0)
        victim_u = block_utilities.get(victim, 0.0)
        best_pruned = None
        best_u = float("-inf")
        for bid, util in pruned_candidates:
            if bid in current:
                continue
            if util > best_u:
                best_u = util
                best_pruned = bid
        if best_pruned is None:
            break
        margin = best_u - victim_u
        if margin < utility_margin_min:
            break
        current.remove(victim)
        current.add(best_pruned)
        swaps.append((victim, best_pruned))
        margins.append(margin)
        pruned_candidates = [(bid, u) for bid, u in pruned_candidates if bid != best_pruned]

    if not swaps:
        return set(original_selected), "no_positive_utility_swaps", [], 0.0

    if len(current) != len(original_selected):
        return set(original_selected), "swap_count_mismatch", [], 0.0
    if token_count(current, token_lengths) != token_count(original_selected, token_lengths):
        return set(original_selected), "token_budget_mismatch", [], 0.0

    return current, "mcu_swap_applied", swaps, float(statistics.mean(margins))


INSTRUMENTATION_FALLBACK_REASONS = {
    "missing_original_selected_block_ids",
    "empty_baseline_compact_input",
    "baseline_probe_failed",
    "no_successful_probes",
    "probe_signal_failed",
}


def is_instrumentation_fallback(reason: str) -> bool:
    if not reason:
        return False
    if reason.startswith("baseline_probe_failed:"):
        return True
    if reason.startswith("CUDA OOM"):
        return True
    return reason.split(":")[0] in INSTRUMENTATION_FALLBACK_REASONS or reason in INSTRUMENTATION_FALLBACK_REASONS


def decide_mcu_audit(
    rows: Sequence[Mapping[str, Any]],
    probe_signal_ok: bool,
    stop_reason: Optional[str] = None,
) -> Dict[str, Any]:
    if not probe_signal_ok:
        return {
            "decision": "BUG_STOP_FIX_REQUIRED",
            "reasons": [stop_reason or "hidden_logit_probe_unavailable"],
            "worth_tiny_quality_smoke": False,
        }
    if stop_reason and "OOM" in stop_reason:
        return {
            "decision": "BUG_STOP_FIX_REQUIRED",
            "reasons": [stop_reason],
            "worth_tiny_quality_smoke": False,
        }

    total = len(rows)
    if total == 0:
        return {"decision": "BUG_STOP_FIX_REQUIRED", "reasons": ["no_audit_rows"], "worth_tiny_quality_smoke": False}

    budget_violations = sum(1 for row in rows if not parse_bool(row.get("budget_preserved")))
    instrumentation_fallback = sum(
        1 for row in rows
        if parse_bool(row.get("fallback_used"))
        and is_instrumentation_fallback(str(row.get("fallback_reason", "")))
    )
    selection_fallback = sum(
        1 for row in rows
        if parse_bool(row.get("fallback_used"))
        and not is_instrumentation_fallback(str(row.get("fallback_reason", "")))
    )
    changed_count = sum(1 for row in rows if parse_bool(row.get("selection_changed")))
    jaccards = [float(row["selected_jaccard_vs_original"]) for row in rows if "selected_jaccard_vs_original" in row]
    median_jaccard = statistics.median(jaccards) if jaccards else 1.0
    margins = [float(row.get("utility_margin", 0.0) or 0.0) for row in rows if not parse_bool(row.get("fallback_used"))]
    mean_margin = statistics.mean(margins) if margins else 0.0
    probe_ms = [float(row.get("probe_runtime_ms", 0.0) or 0.0) for row in rows]
    mean_probe_ms = statistics.mean(probe_ms) if probe_ms else 0.0

    if budget_violations > 0:
        return {
            "decision": "BUG_STOP_FIX_REQUIRED",
            "reasons": [f"budget_violation_count={budget_violations}"],
            "worth_tiny_quality_smoke": False,
        }
    if instrumentation_fallback > 0:
        return {
            "decision": "BUG_STOP_FIX_REQUIRED",
            "reasons": [f"instrumentation_fallback_count={instrumentation_fallback}/{total}"],
            "worth_tiny_quality_smoke": False,
        }

    no_go: List[str] = []
    if changed_count == 0:
        no_go.append("selection_no_op")
    if median_jaccard < 0.30:
        no_go.append(f"almost_all_blocks_changed_median_jaccard={median_jaccard:.3f}")
    if median_jaccard > 0.98 and changed_count > 0:
        no_go.append(f"selection_changes_minimal_median_jaccard={median_jaccard:.3f}")
    if mean_margin < 0.05:
        no_go.append(f"utility_margin_too_small={mean_margin:.3f}")
    if mean_probe_ms > 120_000:
        no_go.append(f"probe_cost_too_high_mean_ms={mean_probe_ms:.0f}")

    block_change_material = 0.05 <= median_jaccard <= 0.98
    go_ready = (
        changed_count > 0
        and budget_violations == 0
        and instrumentation_fallback == 0
        and block_change_material
        and mean_margin >= UTILITY_MARGIN_MIN
        and mean_probe_ms <= 90_000
    )
    if go_ready and not no_go:
        return {
            "decision": "MCU_COUNTERFACTUAL_AUDIT_GO",
            "reasons": [
                f"selection_changed={changed_count}/{total}",
                f"median_jaccard={median_jaccard:.3f}",
                f"mean_utility_margin={mean_margin:.3f}",
                f"mean_probe_runtime_ms={mean_probe_ms:.0f}",
            ],
            "worth_tiny_quality_smoke": True,
        }

    weak_ready = (
        changed_count > 0
        and budget_violations == 0
        and instrumentation_fallback == 0
        and not any(
            reason.startswith("almost_all_blocks_changed")
            or reason.startswith("probe_cost_too_high")
            for reason in no_go
        )
        and (
            mean_margin < UTILITY_MARGIN_MIN
            or median_jaccard > 0.90
            or selection_fallback > 0
        )
    )
    if weak_ready and not any(x in no_go for x in ("selection_no_op",)):
        return {
            "decision": "MCU_COUNTERFACTUAL_AUDIT_WEAK",
            "reasons": no_go or [
                f"selection_changed={changed_count}/{total}",
                f"median_jaccard={median_jaccard:.3f}",
                f"mean_utility_margin={mean_margin:.3f}",
            ],
            "worth_tiny_quality_smoke": True,
        }

    return {
        "decision": "MCU_COUNTERFACTUAL_AUDIT_NO_GO",
        "reasons": no_go or [f"selection_changed={changed_count}/{total}"],
        "worth_tiny_quality_smoke": False,
    }


def build_case_study(rows: Sequence[Mapping[str, Any]], block_texts_by_key: Dict[Key, List[str]]) -> str:
    changed = sorted(
        [row for row in rows if parse_bool(row.get("selection_changed"))],
        key=lambda row: -float(row.get("utility_margin", 0.0) or 0.0),
    )[:5]
    lines = ["# MCU-SlimInfer Case Study", ""]
    for row in changed:
        key = (str(row["task"]), int(row["sample_index"]))
        texts = block_texts_by_key.get(key, [])
        removed = row.get("removed_block_ids") or []
        added = row.get("added_block_ids") or []
        lines.extend([
            f"## {row['task']}:{row['sample_index']} ({row.get('sample_id', '')})",
            f"- utility margin: {float(row.get('utility_margin', 0.0)):.3f}",
            f"- jaccard vs original: {float(row.get('selected_jaccard_vs_original', 1.0)):.3f}",
            f"- hidden/logit means: hidden={float(row.get('hidden_change_mean', 0)):.4f}, kl={float(row.get('logit_kl_mean', 0)):.4f}",
            f"- removed ({removed}): {preview_text(texts, removed if isinstance(removed, list) else json.loads(removed))}",
            f"- added ({added}): {preview_text(texts, added if isinstance(added, list) else json.loads(added))}",
            "",
        ])
    if not changed:
        lines.append("_No MCU selection changes observed._")
        lines.append("")
    return "\n".join(lines)


def load_probe_model(model_path: Path, device: str = "cuda") -> AutoModelForCausalLM:
    dtype = torch.float16
    load_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": dtype,
        "attn_implementation": "flash_attention_2",
    }
    try:
        model = AutoModelForCausalLM.from_pretrained(str(model_path), **load_kwargs)
    except Exception as exc:
        print(f"Warning: flash_attention_2 load failed ({exc}); retrying sdpa.")
        load_kwargs["attn_implementation"] = "sdpa"
        model = AutoModelForCausalLM.from_pretrained(str(model_path), **load_kwargs)
    model = model.to(torch.device(device if torch.cuda.is_available() else "cpu"))
    model.eval()
    return model


def audit_sample(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    task: str,
    sample_index: int,
    sample: Dict[str, Any],
    final_record: JsonRow,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], bool]:
    input_ids, query_text, block_texts, token_lengths = build_runner_prompt_bundle(
        tokenizer, load_dataset2prompt()[task], sample, task,
    )
    seq_len = int(input_ids.shape[0])
    query_start, _ = find_query_span(input_ids, tokenizer, query_text)
    protected = protected_block_ids(seq_len, BLOCK_SIZE, query_start)

    original_selected = {int(x) for x in (final_record.get("selected_block_ids") or [])}
    budget_k = int(final_record.get("budget_k", len(original_selected)) or len(original_selected))
    all_scores = layer_scores_from_record(final_record)
    query_overlaps = block_query_overlaps(block_texts, query_text)
    token_map = block_token_map(block_texts)

    sample_id = final_record.get("sample_id", sample.get("_id", sample_index))
    probe_details: List[Dict[str, Any]] = []
    fallback_reason = ""

    if not original_selected:
        return {
            "task": task,
            "sample_index": sample_index,
            "sample_id": sample_id,
            "fallback_used": True,
            "fallback_reason": "missing_original_selected_block_ids",
        }, probe_details, False

    active_with_protected = set(original_selected) | protected
    candidate_pool = build_candidate_pool(original_selected, all_scores, query_overlaps, protected)
    candidate_pool |= protected

    baseline_compact = compact_input_ids(input_ids, active_with_protected, BLOCK_SIZE)
    if int(baseline_compact.shape[0]) == 0:
        return {
            "task": task,
            "sample_index": sample_index,
            "sample_id": sample_id,
            "fallback_used": True,
            "fallback_reason": "empty_baseline_compact_input",
        }, probe_details, False

    q_start_compact = max(0, int(baseline_compact.shape[0]) - (seq_len - query_start))
    sample_probe_start = time.perf_counter()
    probe_runtime_total = 0.0

    try:
        t0 = time.perf_counter()
        baseline_state = run_forward_probe(model, baseline_compact, q_start_compact)
        probe_runtime_total += (time.perf_counter() - t0) * 1000.0
    except torch.cuda.OutOfMemoryError as exc:
        raise RuntimeError(f"CUDA OOM at baseline probe {task}:{sample_index}: {exc}") from exc
    except Exception as exc:
        return {
            "task": task,
            "sample_index": sample_index,
            "sample_id": sample_id,
            "fallback_used": True,
            "fallback_reason": f"baseline_probe_failed:{exc}",
        }, probe_details, False

    selected_in_pool = sorted(bid for bid in candidate_pool if bid in original_selected and bid not in protected)
    pruned_in_pool = sorted(bid for bid in candidate_pool if bid not in original_selected)
    selected_in_pool = selected_in_pool[:MAX_REMOVE_PROBES]
    pruned_in_pool = pruned_in_pool[:MAX_SWAP_PROBES]

    victim_for_swap = min(
        (bid for bid in original_selected if bid not in protected),
        key=lambda bid: all_scores.get(bid, float("inf")),
        default=None,
    )

    raw_signals: Dict[int, Dict[str, float]] = {}
    probe_metrics_by_block: Dict[int, ProbeMetrics] = {}

    def record_probe(probe_type: str, block_id: int, active: Set[int], metrics: ProbeMetrics) -> None:
        probe_details.append({
            "task": task,
            "sample_index": sample_index,
            "probe_type": probe_type,
            "block_id": block_id,
            "active_block_ids": sorted(active),
            "metrics": metrics.__dict__,
        })
        hidden_sig = metrics.last_token_hidden_l2_change + 0.5 * metrics.query_token_hidden_l2_change
        entropy_reduction = -metrics.next_token_entropy_change
        raw_signals.setdefault(block_id, {})
        raw_signals[block_id].update({
            "hidden": hidden_sig,
            "logit_kl": metrics.next_token_logit_kl,
            "entropy_reduction": entropy_reduction,
            "margin": metrics.top_logit_margin_change,
        })
        probe_metrics_by_block[block_id] = metrics

    for block_id in selected_in_pool:
        if time.perf_counter() - sample_probe_start > PROBE_SLOW_SAMPLE_SEC:
            fallback_reason = "probe_runtime_exceeded_sample_budget"
            break
        active = set(active_with_protected)
        active.discard(block_id)
        compact = compact_input_ids(input_ids, active, BLOCK_SIZE)
        if int(compact.shape[0]) == 0:
            continue
        q_start = max(0, int(compact.shape[0]) - (seq_len - query_start))
        t0 = time.perf_counter()
        try:
            state = run_forward_probe(model, compact, q_start)
        except torch.cuda.OutOfMemoryError as exc:
            raise RuntimeError(f"CUDA OOM at remove probe {task}:{sample_index} block={block_id}") from exc
        runtime_ms = (time.perf_counter() - t0) * 1000.0
        if runtime_ms > PROBE_SLOW_SINGLE_SEC * 1000.0:
            fallback_reason = f"single_probe_too_slow_ms={runtime_ms:.0f}"
            break
        probe_runtime_total += runtime_ms
        metrics = compare_probe_states(baseline_state, state, runtime_ms)
        record_probe("remove", block_id, active, metrics)

    if victim_for_swap is not None:
        for block_id in pruned_in_pool:
            if time.perf_counter() - sample_probe_start > PROBE_SLOW_SAMPLE_SEC:
                fallback_reason = fallback_reason or "probe_runtime_exceeded_sample_budget"
                break
            active = set(active_with_protected)
            active.discard(victim_for_swap)
            active.add(block_id)
            compact = compact_input_ids(input_ids, active, BLOCK_SIZE)
            q_start = max(0, int(compact.shape[0]) - (seq_len - query_start))
            t0 = time.perf_counter()
            try:
                state = run_forward_probe(model, compact, q_start)
            except torch.cuda.OutOfMemoryError as exc:
                raise RuntimeError(f"CUDA OOM at swap probe {task}:{sample_index} block={block_id}") from exc
            runtime_ms = (time.perf_counter() - t0) * 1000.0
            if runtime_ms > PROBE_SLOW_SINGLE_SEC * 1000.0:
                fallback_reason = f"single_probe_too_slow_ms={runtime_ms:.0f}"
                break
            probe_runtime_total += runtime_ms
            metrics = compare_probe_states(baseline_state, state, runtime_ms)
            record_probe("swap", block_id, active, metrics)

    signal_ok = bool(probe_metrics_by_block)
    if not signal_ok:
        return {
            "task": task,
            "sample_index": sample_index,
            "sample_id": sample_id,
            "original_selected_block_ids": sorted(original_selected),
            "mcu_selected_block_ids": sorted(original_selected),
            "candidate_pool_size": len(candidate_pool),
            "num_counterfactual_probes": len(probe_details),
            "fallback_used": True,
            "fallback_reason": fallback_reason or "no_successful_probes",
            "probe_runtime_ms": probe_runtime_total,
        }, probe_details, False

    redundancy_map = {
        bid: _dc_selected_redundancy({bid}, token_map) for bid in candidate_pool
    }
    sample_z = {
        "hidden": zscore_map({bid: raw_signals.get(bid, {}).get("hidden", 0.0) for bid in raw_signals}),
        "logit_kl": zscore_map({bid: raw_signals.get(bid, {}).get("logit_kl", 0.0) for bid in raw_signals}),
        "entropy_reduction": zscore_map({bid: raw_signals.get(bid, {}).get("entropy_reduction", 0.0) for bid in raw_signals}),
        "margin": zscore_map({bid: raw_signals.get(bid, {}).get("margin", 0.0) for bid in raw_signals}),
        "redundancy": zscore_map(redundancy_map),
    }

    block_utilities: Dict[int, float] = {}
    for bid in candidate_pool:
        if bid in probe_metrics_by_block:
            block_utilities[bid] = compute_block_utility(
                bid, probe_metrics_by_block[bid], redundancy_map.get(bid, 0.0), sample_z,
            )
        elif bid in all_scores:
            block_utilities[bid] = float(all_scores[bid])

    utility_values = [float(v) for v in block_utilities.values() if math.isfinite(v)]
    utility_spread = (max(utility_values) - min(utility_values)) if len(utility_values) >= 2 else 0.0

    mcu_selected, select_reason, swaps, utility_margin = mcu_select_blocks(
        original_selected,
        block_utilities,
        budget_k,
        protected,
        token_lengths,
    )

    use_fallback = False
    if fallback_reason and is_instrumentation_fallback(fallback_reason):
        use_fallback = True
    elif utility_spread < 0.05:
        use_fallback = True
        fallback_reason = f"utility_spread_too_small={utility_spread:.4f}"
    elif select_reason in {"no_positive_utility_swaps", "zero_swap_budget"}:
        use_fallback = True
        fallback_reason = select_reason

    if use_fallback:
        mcu_selected = set(original_selected)
        swaps = []

    removed = sorted(original_selected - mcu_selected)
    added = sorted(mcu_selected - original_selected)
    selection_changed = bool(removed or added)
    jaccard = selected_jaccard(original_selected, mcu_selected)

    hidden_vals = [m.last_token_hidden_l2_change + 0.5 * m.query_token_hidden_l2_change for m in probe_metrics_by_block.values()]
    kl_vals = [m.next_token_logit_kl for m in probe_metrics_by_block.values()]
    entropy_vals = [m.next_token_entropy_change for m in probe_metrics_by_block.values()]
    margin_vals = [m.top_logit_margin_change for m in probe_metrics_by_block.values()]

    original_redundancy = _dc_selected_redundancy(original_selected, token_map)
    mcu_redundancy = _dc_selected_redundancy(mcu_selected, token_map)
    original_query_overlap = float(
        sum(query_overlaps.get(bid, 0.0) for bid in original_selected) / max(1, len(original_selected))
    )
    mcu_query_overlap = float(
        sum(query_overlaps.get(bid, 0.0) for bid in mcu_selected) / max(1, len(mcu_selected))
    )

    original_token_count = token_count(original_selected, token_lengths)
    mcu_token_count = token_count(mcu_selected, token_lengths)
    budget_preserved = len(mcu_selected) == len(original_selected) and mcu_token_count == original_token_count

    removed_util = [block_utilities.get(bid, 0.0) for bid in removed]
    added_util = [block_utilities.get(bid, 0.0) for bid in added]

    row = {
        "task": task,
        "sample_index": sample_index,
        "sample_id": sample_id,
        "original_selected_block_ids": sorted(original_selected),
        "mcu_selected_block_ids": sorted(mcu_selected),
        "candidate_pool_size": len(candidate_pool),
        "num_counterfactual_probes": len(probe_details),
        "selected_jaccard_vs_original": jaccard,
        "selection_changed": selection_changed,
        "removed_block_ids": removed,
        "added_block_ids": added,
        "removed_block_utility": removed_util,
        "added_block_utility": added_util,
        "utility_margin": utility_margin,
        "hidden_change_mean": statistics.mean(hidden_vals) if hidden_vals else 0.0,
        "logit_kl_mean": statistics.mean(kl_vals) if kl_vals else 0.0,
        "entropy_change_mean": statistics.mean(entropy_vals) if entropy_vals else 0.0,
        "top_logit_margin_change_mean": statistics.mean(margin_vals) if margin_vals else 0.0,
        "redundancy_delta": mcu_redundancy - original_redundancy,
        "query_overlap_delta": mcu_query_overlap - original_query_overlap,
        "selected_token_count_original": original_token_count,
        "selected_token_count_mcu": mcu_token_count,
        "budget_preserved": budget_preserved,
        "over_budget": mcu_token_count > original_token_count,
        "fallback_used": use_fallback,
        "fallback_reason": fallback_reason or select_reason,
        "probe_runtime_ms": probe_runtime_total,
        "mcu_select_reason": select_reason,
        "utility_spread": utility_spread,
        "swaps": swaps,
    }
    return row, probe_details, signal_ok


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--longbench_dir", type=Path, default=DEFAULT_LONGBENCH_DIR)
    parser.add_argument("--output_root", type=Path, default=Path("results"))
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--skip_gate_run", action="store_true")
    parser.add_argument("--score_dump_path", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    keys = mcu_keys()
    sample_spec, expected_indices = build_sample_spec()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (REPO_ROOT / args.output_root / f"mcu_sliminfer_counterfactual_audit_{timestamp}")
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    gate_dir = output_dir / "run_sliminfer_original_gate_audit"
    score_dump_path = args.score_dump_path or gate_dir / "uasr_score_dump.jsonl"

    stop_reason: Optional[str] = None
    probe_signal_ok = True

    try:
        if args.skip_gate_run:
            if not score_dump_path.exists():
                raise RuntimeError(f"--skip_gate_run but score dump missing: {score_dump_path}")
        else:
            paths = run_gate_audit(args, output_dir, sample_spec)
            score_dump_path = paths["score_dump_path"]

        final_records = load_final_layer_records(score_dump_path, keys)
        missing_keys = [key for key in keys if key not in final_records]
        if missing_keys:
            probe_signal_ok = False
            stop_reason = f"missing_final_layer_records={missing_keys}"

        tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), trust_remote_code=True)
        samples_cache = {task: load_samples(args.longbench_dir, task) for task in TASKS}
        block_texts_by_key: Dict[Key, List[str]] = {}

        audit_rows: List[Dict[str, Any]] = []
        all_probe_details: List[Dict[str, Any]] = []

        if probe_signal_ok:
            model = load_probe_model(args.model_path, args.device)
            for task, sample_index in keys:
                key = (task, sample_index)
                final_record = final_records[key]
                sample = samples_cache[task][sample_index]
                _, _, block_texts, _ = build_runner_prompt_bundle(
                    tokenizer, load_dataset2prompt()[task], sample, task,
                )
                block_texts_by_key[key] = block_texts

                try:
                    row, probe_details, signal_ok = audit_sample(
                        model, tokenizer, task, sample_index, sample, final_record,
                    )
                except RuntimeError as exc:
                    if "OOM" in str(exc):
                        stop_reason = str(exc)
                        probe_signal_ok = False
                        audit_rows.append({
                            "task": task,
                            "sample_index": sample_index,
                            "fallback_used": True,
                            "fallback_reason": str(exc),
                        })
                        break
                    raise

                all_probe_details.extend(probe_details)
                audit_rows.append(row)
                if not signal_ok and not parse_bool(row.get("fallback_used")):
                    probe_signal_ok = False
                    stop_reason = row.get("fallback_reason", "probe_signal_failed")

                if stop_reason:
                    break

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    except RuntimeError as exc:
        stop_reason = str(exc)
        if "OOM" in stop_reason:
            probe_signal_ok = False
        else:
            raise

    valid_rows = [row for row in audit_rows if row.get("original_selected_block_ids")]
    decision = decide_mcu_audit(valid_rows, probe_signal_ok, stop_reason)

    write_csv(output_dir / "mcu_counterfactual_audit.csv", audit_rows, AUDIT_FIELDS)
    write_jsonl(output_dir / "mcu_probe_details.jsonl", all_probe_details)

    group_rows: List[Dict[str, Any]] = []
    for group in TASKS + ["all"]:
        subset = audit_rows if group == "all" else [row for row in audit_rows if row.get("task") == group]
        if not subset:
            continue
        valid = [row for row in subset if not parse_bool(row.get("fallback_used"))]
        jaccards = [float(row["selected_jaccard_vs_original"]) for row in valid if "selected_jaccard_vs_original" in row]
        group_rows.append({
            "group": group,
            "sample_count": len(subset),
            "valid_sample_count": len(valid),
            "fallback_count": sum(1 for row in subset if parse_bool(row.get("fallback_used"))),
            "selection_fallback_count": sum(
                1 for row in subset
                if parse_bool(row.get("fallback_used"))
                and not is_instrumentation_fallback(str(row.get("fallback_reason", "")))
            ),
            "budget_violation_count": sum(1 for row in valid if not parse_bool(row.get("budget_preserved", True))),
            "selection_changed_count": sum(1 for row in valid if parse_bool(row.get("selection_changed"))),
            "mean_probe_runtime_ms": statistics.mean(
                float(row.get("probe_runtime_ms", 0.0) or 0.0) for row in subset
            ) if subset else 0.0,
            "mean_utility_margin": statistics.mean(
                float(row.get("utility_margin", 0.0) or 0.0) for row in valid
            ) if valid else 0.0,
            "median_selected_jaccard": statistics.median(jaccards) if jaccards else 1.0,
            "mean_hidden_change": statistics.mean(
                float(row.get("hidden_change_mean", 0.0) or 0.0) for row in valid
            ) if valid else 0.0,
            "mean_logit_kl": statistics.mean(
                float(row.get("logit_kl_mean", 0.0) or 0.0) for row in valid
            ) if valid else 0.0,
        })
    if group_rows:
        write_csv(output_dir / "mcu_group_summary.csv", group_rows, list(group_rows[0].keys()))

    all_summary = next((row for row in group_rows if row["group"] == "all"), {})
    mean_pool_size = (
        statistics.mean(float(r.get("candidate_pool_size", 0)) for r in valid_rows)
        if valid_rows else 0.0
    )
    case_study = build_case_study(valid_rows, block_texts_by_key)
    (output_dir / "mcu_case_study.md").write_text(case_study, encoding="utf-8")

    report_lines = [
        "# MCU-SlimInfer Counterfactual Utility Audit",
        "",
        f"- Output: `{output_dir}`",
        f"- Decision: **{decision.get('decision')}**",
        f"- Samples: **{len(keys)}** (completed={len(audit_rows)})",
        "",
        "## Probe signal",
        f"1. Hidden/logit counterfactual signal available: **{probe_signal_ok}**",
        f"2. Total probe detail rows: **{len(all_probe_details)}**",
        f"3. Mean probe runtime per sample (ms): **{all_summary.get('mean_probe_runtime_ms', 0.0):.0f}**",
        "",
        "## Candidate pool / utility",
        f"4. Mean candidate pool size: **{mean_pool_size:.1f}**",
        f"5. Mean utility margin (non-fallback): **{all_summary.get('mean_utility_margin', 0.0):.3f}**",
        f"6. Mean hidden change: **{all_summary.get('mean_hidden_change', 0.0):.4f}**",
        f"7. Mean logit KL: **{all_summary.get('mean_logit_kl', 0.0):.4f}**",
        "",
        "## Selection",
        f"8. Selection changed count: **{all_summary.get('selection_changed_count', 0)}**",
        f"9. Budget violation count: **{all_summary.get('budget_violation_count', 0)}**",
        f"10. Fallback count: **{all_summary.get('fallback_count', 0)}**",
        f"11. Median selected Jaccard: **{all_summary.get('median_selected_jaccard', 1.0):.4f}**",
        "",
        "## Tiny quality smoke",
        f"- Worth tiny quality smoke: **{decision.get('worth_tiny_quality_smoke', False)}**",
        "",
        "## Decision reasons",
    ]
    if stop_reason:
        report_lines.append(f"- Stop reason: {stop_reason}")
    report_lines.extend(f"- {reason}" for reason in decision.get("reasons", []))
    report_lines.append("")
    (output_dir / "mcu_audit_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    (output_dir / "mcu_decision.md").write_text(
        f"# Decision: {decision.get('decision')}\n\n"
        + "\n".join(f"- {reason}" for reason in decision.get("reasons", []))
        + f"\n\n- Worth tiny quality smoke: {decision.get('worth_tiny_quality_smoke', False)}\n",
        encoding="utf-8",
    )

    write_json(output_dir / "run_config.json", {
        "sample_spec": sample_spec,
        "expected_indices": expected_indices,
        "mcu_sample_indices": MCU_SAMPLE_INDICES,
        "max_pool_size": MAX_POOL_SIZE,
        "max_replace_frac": MAX_REPLACE_FRAC,
        "utility_formula": "z(hidden_change)+z(logit_kl)+0.5*z(entropy_reduction)+0.5*z(margin)-0.5*z(redundancy)",
        "score_dump_path": str(score_dump_path),
        "quality_generation_performed": False,
        "scorer_invoked": False,
        "stop_reason": stop_reason,
    })
    write_json(output_dir / "mcu_decision.json", decision)

    print("\n".join(report_lines))
    print(f"Wrote results to {output_dir}")


if __name__ == "__main__":
    main()
