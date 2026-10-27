#!/usr/bin/env python3
"""Runtime QDC selection audit on held-out64 sliminfer_original selections.

Runs prefill-only gate_audit to capture current runtime sliminfer_original block
selection, then applies budget-neutral QDC coverage repair offline on those
selections. No generation, no scorer, no gold.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import torch
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
    GATE_LAYER,
    MAX_MODEL_LEN,
    TASKS,
    TINY_EXCLUDE,
    covered_components,
    heldout_indices,
    heldout_keys,
    parse_bool,
    percentile,
    preview_text,
)
from utils.cache_utils import _qdc_apply_coverage_repair, _qdc_extract_query_components

DEFAULT_LONGBENCH_DIR = Path("/home/yu-bai/Desktop/jiaqizhang/HeadKV/data/LongBench")
DEFAULT_MODEL_PATH = Path(
    "/home/yu-bai/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/"
    "snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
)
DEFAULT_OFFLINE_AUDIT_CSV = Path("results/qdc_sliminfer_selection_audit_20260708_191113/qdc_selection_audit.csv")
PRUNE_LAYERS = [9, 18, 26]
PREFILL_AUDIT_LAYER = 27
METHOD_ORIGINAL = "sliminfer_original"

Key = Tuple[str, int]
JsonRow = Dict[str, Any]


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
) -> Tuple[str, torch.Tensor, List[str], List[int]]:
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
    token_lengths = [
        min(BLOCK_SIZE, seq_len - start)
        for start in range(0, seq_len, BLOCK_SIZE)
    ]
    query_text = _sample_query_text(sample)
    return query_text, input_ids, block_texts, token_lengths


def base_gate_audit_cmd(args: argparse.Namespace, run_dir: Path, sample_spec: str, used_blocks_path: Path, dc_sink_path: Path) -> List[str]:
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
        "--longbench_subset_name", "qdc_runtime_selection_audit_heldout64",
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
        "--dc_selection_audit_path", str(dc_sink_path),
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


def run_runtime_gate_audit(args: argparse.Namespace, output_dir: Path, sample_spec: str) -> Path:
    run_dir = output_dir / "run_sliminfer_original_gate_audit"
    run_dir.mkdir(parents=True, exist_ok=True)
    used_blocks_path = run_dir / "used_blocks.jsonl"
    dc_sink_path = run_dir / "dc_selection_audit_sink.jsonl"
    dc_sink_path.write_text("", encoding="utf-8")
    if not used_blocks_path.exists():
        used_blocks_path.write_text("", encoding="utf-8")
    cmd = base_gate_audit_cmd(args, run_dir, sample_spec, used_blocks_path, dc_sink_path)
    write_json(output_dir / "runtime_gate_audit_command.json", {"command": cmd})
    run_command(cmd, REPO_ROOT, output_dir / "runtime_gate_audit_stdout.log", output_dir / "runtime_gate_audit_stderr.log")
    manifest = run_dir / "gate_audit_manifest.jsonl"
    if not manifest.exists():
        raise RuntimeError("BUG_STOP_FIX_REQUIRED: gate_audit_manifest.jsonl missing after gate_audit run")
    return used_blocks_path


def _parse_block_ids(value: Any) -> List[int]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return [int(x) for x in value]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [int(x) for x in parsed]
        except json.JSONDecodeError:
            pass
    return []


def load_runtime_selection(used_blocks_path: Path, keys: Sequence[Key]) -> Dict[Key, JsonRow]:
    rows = read_jsonl(used_blocks_path)
    selected: Dict[Key, JsonRow] = {}
    for row in rows:
        if str(row.get("phase", "")) != "prefill":
            continue
        if int(row.get("layer_id", -1)) != PREFILL_AUDIT_LAYER:
            continue
        key = (str(row.get("task", "")), int(row.get("sample_index", -1)))
        if key[0] and key[1] >= 0 and key not in selected:
            selected[key] = row

    out: Dict[Key, JsonRow] = {}
    missing: List[Key] = []
    for key in keys:
        row = selected.get(key)
        if row is None:
            missing.append(key)
            continue
        used_block_ids = _parse_block_ids(row.get("used_block_ids"))
        if not used_block_ids:
            used_block_ids = _parse_block_ids(row.get("used_block_ids_first100"))
        prompt_block_count = int(row.get("prompt_block_count", 0) or 0)
        final_block_count = int(row.get("final_block_count_used_by_attention", 0) or 0)
        if not used_block_ids and prompt_block_count > 0 and final_block_count >= prompt_block_count:
            used_block_ids = list(range(prompt_block_count))
        if not used_block_ids:
            missing.append(key)
            continue
        out[key] = {
            "task": key[0],
            "sample_index": key[1],
            "sample_id": row.get("sample_id", key[1]),
            "runtime_original_selected_block_ids": sorted(int(x) for x in used_block_ids),
            "prompt_block_count": prompt_block_count,
            "final_block_count_used_by_attention": final_block_count,
            "selected_token_count_original": int(row.get("kv_abs_indices_count", 0) or 0),
            "selection_source": "used_blocks_prefill_layer27",
        }
    if missing:
        raise RuntimeError(
            "BUG_STOP_FIX_REQUIRED: missing runtime original selection for "
            f"{len(missing)} samples: {missing[:10]}"
        )
    return out


def load_offline_triggers(offline_csv: Path) -> Dict[Key, bool]:
    if not offline_csv.exists():
        return {}
    out: Dict[Key, bool] = {}
    for row in csv.DictReader(offline_csv.open(encoding="utf-8")):
        key = (str(row.get("task", "")), int(row.get("sample_index", -1)))
        out[key] = parse_bool(row.get("qdc_triggered"))
    return out


def token_count(selected: Set[int], token_lengths: Sequence[int]) -> int:
    return sum(int(token_lengths[block_id]) for block_id in selected if 0 <= block_id < len(token_lengths))


def added_blocks_explain_missing(
    missing_before: Sequence[str],
    missing_after: Sequence[str],
    added_block_ids: Sequence[int],
    block_texts: Sequence[str],
    query_components: Sequence[str],
) -> bool:
    if not added_block_ids:
        return not missing_before or not missing_after
    gained = set(missing_before) - set(missing_after)
    if not gained:
        return False
    for block_id in added_block_ids:
        if not (0 <= int(block_id) < len(block_texts)):
            continue
        text = block_texts[int(block_id)]
        for component in gained:
            kind, _, value = component.partition(":")
            if not value:
                continue
            if value in str(text).lower():
                return True
            if kind in ("entity", "noun", "constraint", "wh") and value in str(text).lower():
                return True
    return False


def decide_runtime_audit(rows: Sequence[Mapping[str, Any]], offline_agreement: Mapping[str, Any]) -> Dict[str, Any]:
    total = len(rows)
    if total == 0:
        return {"decision": "BUG_STOP_FIX_REQUIRED", "reasons": ["no_rows"]}

    bug_reasons: List[str] = []
    missing_selection = sum(1 for row in rows if not row.get("runtime_original_selected_block_ids"))
    if missing_selection:
        bug_reasons.append(f"missing_runtime_original_selection={missing_selection}")
    if any(row.get("runtime_original_selected_block_ids") is None for row in rows):
        bug_reasons.append("runtime_original_selected_block_ids_null")
    if any(row.get("runtime_qdc_selected_block_ids") is None for row in rows):
        bug_reasons.append("runtime_qdc_selected_block_ids_null")

    budget_violations = sum(1 for row in rows if not parse_bool(row.get("budget_preserved")))
    fallback_count = sum(1 for row in rows if parse_bool(row.get("fallback_used")))
    over_budget_count = sum(1 for row in rows if parse_bool(row.get("over_budget")))
    trigger_count = sum(1 for row in rows if parse_bool(row.get("qdc_triggered")))
    changed_count = sum(
        1 for row in rows
        if _parse_block_ids(row.get("removed_block_ids")) or _parse_block_ids(row.get("added_block_ids"))
    )

    if budget_violations or over_budget_count:
        bug_reasons.append(f"budget_violation_count={budget_violations + over_budget_count}")
    if changed_count > 0 and trigger_count == 0:
        bug_reasons.append("selection_changed_but_qdc_not_triggered")

    orig_cov = [float(row["original_component_coverage"]) for row in rows if row.get("original_component_coverage") is not None]
    qdc_cov = [float(row["qdc_component_coverage"]) for row in rows if row.get("qdc_component_coverage") is not None]
    cov_delta = [float(row["coverage_delta"]) for row in rows]
    jaccards = [float(row["selected_jaccard_vs_original"]) for row in rows]

    if not orig_cov or not qdc_cov:
        bug_reasons.append("coverage_fields_missing")

    if bug_reasons:
        return {
            "decision": "BUG_STOP_FIX_REQUIRED",
            "reasons": bug_reasons,
            "trigger_count": trigger_count,
            "worth_tiny_quality_smoke": False,
        }

    mean_orig_cov = statistics.mean(orig_cov)
    mean_qdc_cov = statistics.mean(qdc_cov)
    mean_cov_delta = statistics.mean(cov_delta)
    median_jaccard = statistics.median(jaccards)
    p25_jaccard = percentile(jaccards, 25)
    explain_count = sum(1 for row in rows if parse_bool(row.get("added_blocks_explain_missing")))

    no_go_reasons: List[str] = []
    weak_reasons: List[str] = []

    if trigger_count == 0:
        no_go_reasons.append("runtime_trigger_count=0")
    if mean_cov_delta <= 0.005:
        no_go_reasons.append(f"mean_coverage_delta={mean_cov_delta:.4f}")
    if fallback_count > total // 4:
        no_go_reasons.append(f"fallback_count={fallback_count}")

    go_ready = (
        trigger_count >= 10
        and budget_violations == 0
        and fallback_count == 0
        and over_budget_count == 0
        and mean_cov_delta >= 0.02
        and median_jaccard >= 0.35
        and explain_count >= max(1, trigger_count // 2)
    )
    if go_ready:
        return {
            "decision": "QDC_RUNTIME_AUDIT_GO",
            "reasons": [
                f"runtime_trigger_count={trigger_count}/{total}",
                f"mean_coverage_delta={mean_cov_delta:+.4f}",
                f"median_jaccard={median_jaccard:.3f}",
                f"offline_trigger_agreement={offline_agreement.get('agreement_rate', 0):.3f}",
            ],
            "trigger_count": trigger_count,
            "mean_original_component_coverage": mean_orig_cov,
            "mean_qdc_component_coverage": mean_qdc_cov,
            "mean_coverage_delta": mean_cov_delta,
            "median_selected_jaccard": median_jaccard,
            "worth_tiny_quality_smoke": True,
        }

    weak_ready = (
        trigger_count > 0
        and budget_violations == 0
        and over_budget_count == 0
        and mean_cov_delta > 0
    )
    if weak_ready:
        if trigger_count < 10:
            weak_reasons.append(f"runtime_trigger_count={trigger_count}<10")
        if median_jaccard < 0.35:
            weak_reasons.append(f"median_jaccard={median_jaccard:.3f}")
        if mean_cov_delta < 0.02:
            weak_reasons.append(f"mean_coverage_delta={mean_cov_delta:.4f}")
        return {
            "decision": "QDC_RUNTIME_AUDIT_WEAK",
            "reasons": weak_reasons or ["coverage_gain_unstable"],
            "trigger_count": trigger_count,
            "mean_coverage_delta": mean_cov_delta,
            "worth_tiny_quality_smoke": trigger_count >= 5,
        }

    return {
        "decision": "QDC_RUNTIME_AUDIT_NO_GO",
        "reasons": no_go_reasons or ["qdc_mostly_noop"],
        "trigger_count": trigger_count,
        "mean_coverage_delta": mean_cov_delta,
        "worth_tiny_quality_smoke": False,
    }


def compare_offline_triggers(rows: Sequence[Mapping[str, Any]], offline_triggers: Dict[Key, bool]) -> Dict[str, Any]:
    compared = 0
    agree = 0
    runtime_only: List[str] = []
    offline_only: List[str] = []
    for row in rows:
        key = (str(row["task"]), int(row["sample_index"]))
        if key not in offline_triggers:
            continue
        compared += 1
        runtime_trigger = parse_bool(row.get("qdc_triggered"))
        offline_trigger = bool(offline_triggers[key])
        if runtime_trigger == offline_trigger:
            agree += 1
        elif runtime_trigger and not offline_trigger:
            runtime_only.append(f"{key[0]}:{key[1]}")
        else:
            offline_only.append(f"{key[0]}:{key[1]}")
    return {
        "compared_samples": compared,
        "agreement_count": agree,
        "agreement_rate": (agree / compared) if compared else 0.0,
        "runtime_only_triggers": runtime_only[:20],
        "offline_only_triggers": offline_only[:20],
        "offline_only_trigger_count": len(offline_only),
        "runtime_only_trigger_count": len(runtime_only),
    }


def build_case_study(rows: Sequence[Mapping[str, Any]]) -> str:
    positive = sorted(
        [row for row in rows if parse_bool(row.get("qdc_triggered"))],
        key=lambda row: (-float(row.get("coverage_delta", 0)), -float(row.get("selected_jaccard_vs_original", 0))),
    )[:5]
    risky = sorted(
        rows,
        key=lambda row: (
            -float(row.get("redundancy_delta", 0) or 0),
            float(row.get("selected_jaccard_vs_original", 1)),
        ),
    )[:3]
    lines = ["# Runtime QDC Case Study", ""]
    lines.append("## Positive-looking runtime triggers")
    for row in positive:
        lines.extend([
            "",
            f"### {row['task']}:{row['sample_index']} ({row['sample_id']})",
            f"- coverage: {float(row['original_component_coverage']):.3f} -> {float(row['qdc_component_coverage']):.3f} (delta {float(row['coverage_delta']):+.3f})",
            f"- missing before: {row.get('missing_components_before')}",
            f"- missing after: {row.get('missing_components_after')}",
            f"- added: {row.get('added_block_ids')} | removed: {row.get('removed_block_ids')}",
            f"- jaccard: {float(row['selected_jaccard_vs_original']):.3f}",
            f"- offline trigger: {row.get('offline_qdc_triggered')}",
            f"- added preview: {row.get('added_block_text_preview', '')}",
        ])
    lines.append("")
    lines.append("## Risky / high-change cases")
    for row in risky:
        lines.extend([
            "",
            f"### {row['task']}:{row['sample_index']} ({row['sample_id']})",
            f"- triggered: {row.get('qdc_triggered')} | reason: {row.get('qdc_trigger_reason')}",
            f"- coverage delta: {float(row.get('coverage_delta', 0)):+.3f}",
            f"- jaccard: {float(row.get('selected_jaccard_vs_original', 1)):.3f}",
        ])
    lines.append("")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--longbench_dir", type=Path, default=DEFAULT_LONGBENCH_DIR)
    parser.add_argument("--offline_audit_csv", type=Path, default=DEFAULT_OFFLINE_AUDIT_CSV)
    parser.add_argument("--output_root", type=Path, default=Path("results"))
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--skip_gate_run", action="store_true")
    parser.add_argument("--min_free_gpu_gb", type=float, default=8.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample_spec, expected_indices = build_sample_spec()
    keys = heldout_keys()
    if len(keys) != 128:
        raise RuntimeError(f"BUG_STOP_FIX_REQUIRED: expected 128 held-out keys, got {len(keys)}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (REPO_ROOT / args.output_root / f"qdc_runtime_selection_audit_{timestamp}")
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_config = {
        "tasks": TASKS,
        "sample_spec": sample_spec,
        "expected_indices": expected_indices,
        "tiny_exclude": {task: sorted(TINY_EXCLUDE[task]) for task in TASKS},
        "prune_layers": PRUNE_LAYERS,
        "prefill_audit_layer": PREFILL_AUDIT_LAYER,
        "model_path": str(args.model_path),
        "longbench_dir": str(args.longbench_dir),
        "offline_audit_csv": str((REPO_ROOT / args.offline_audit_csv).resolve()),
        "quality_generation_performed": False,
        "scorer_invoked": False,
    }
    write_json(output_dir / "run_config.json", run_config)

    used_blocks_path = output_dir / "run_sliminfer_original_gate_audit" / "used_blocks.jsonl"
    if args.skip_gate_run:
        if not used_blocks_path.exists():
            raise RuntimeError(f"--skip_gate_run requested but missing {used_blocks_path}")
    else:
        used_blocks_path = run_runtime_gate_audit(args, output_dir, sample_spec)

    runtime_selection = load_runtime_selection(used_blocks_path, keys)
    offline_triggers = load_offline_triggers((REPO_ROOT / args.offline_audit_csv).resolve())

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), trust_remote_code=True)
    dataset2prompt = load_dataset2prompt()
    samples_cache = {task: load_samples(args.longbench_dir, task) for task in TASKS}

    audit_rows: List[Dict[str, Any]] = []
    for task, sample_index in keys:
        key = (task, sample_index)
        runtime_row = runtime_selection[key]
        sample = samples_cache[task][sample_index]
        query_text, _input_ids, block_texts, token_lengths = build_runner_prompt_bundle(
            tokenizer, dataset2prompt[task], sample, task,
        )
        original_selected = {int(x) for x in runtime_row["runtime_original_selected_block_ids"]}
        query_components = _qdc_extract_query_components(query_text)
        original_cov_set = covered_components(original_selected, block_texts, query_components)
        missing_before = [comp for comp in query_components if comp not in original_cov_set]

        repair = _qdc_apply_coverage_repair(
            cfg={
                "block_texts": block_texts,
                "query_text": query_text,
                "block_size": BLOCK_SIZE,
                "_qdc_block_token_lengths": token_lengths,
            },
            original_selected=original_selected,
            block_texts=block_texts,
            token_lengths=token_lengths,
            query_components=query_components,
            query_text=query_text,
        )
        qdc_selected = {int(x) for x in repair.get("selected", original_selected)}
        qdc_cov_set = covered_components(qdc_selected, block_texts, query_components)
        missing_after = [comp for comp in query_components if comp not in qdc_cov_set]

        num_components = len(query_components)
        original_coverage = float(repair.get("original_component_coverage", len(original_cov_set) / max(1, num_components)))
        qdc_coverage = float(repair.get("qdc_component_coverage", len(qdc_cov_set) / max(1, num_components)))
        coverage_delta = float(repair.get("coverage_delta", qdc_coverage - original_coverage))

        original_token_count = token_count(original_selected, token_lengths)
        qdc_token_count = token_count(qdc_selected, token_lengths)
        removed_ids = [int(x) for x in repair.get("removed_block_ids", [])]
        added_ids = [int(x) for x in repair.get("added_block_ids", [])]
        budget_preserved = bool(
            repair.get("budget_preserved", len(qdc_selected) == len(original_selected) and qdc_token_count == original_token_count)
        )

        row = {
            "task": task,
            "sample_index": sample_index,
            "sample_id": runtime_row.get("sample_id", sample.get("_id", sample.get("id", sample_index))),
            "runtime_original_selected_block_ids": sorted(original_selected),
            "runtime_qdc_selected_block_ids": sorted(qdc_selected),
            "qdc_triggered": bool(repair.get("qdc_triggered")),
            "qdc_trigger_reason": str(repair.get("qdc_trigger_reason", "")),
            "original_component_coverage": original_coverage,
            "qdc_component_coverage": qdc_coverage,
            "coverage_delta": coverage_delta,
            "missing_components_before": missing_before,
            "missing_components_after": missing_after,
            "selected_jaccard_vs_original": float(repair.get("selected_jaccard_vs_original", 1.0 if not (removed_ids or added_ids) else 0.0)),
            "removed_block_ids": removed_ids,
            "added_block_ids": added_ids,
            "removed_block_text_preview": preview_text(block_texts, removed_ids),
            "added_block_text_preview": preview_text(block_texts, added_ids),
            "budget_preserved": budget_preserved,
            "selected_token_count_original": original_token_count,
            "selected_token_count_qdc": qdc_token_count,
            "over_budget": bool(qdc_token_count > original_token_count),
            "fallback_used": bool(repair.get("fallback_used")),
            "offline_qdc_triggered": bool(offline_triggers.get(key, False)),
            "offline_runtime_trigger_match": bool(offline_triggers.get(key, False)) == bool(repair.get("qdc_triggered")),
            "added_blocks_explain_missing": added_blocks_explain_missing(
                missing_before, missing_after, added_ids, block_texts, query_components,
            ),
            "prompt_block_count": int(runtime_row.get("prompt_block_count", len(block_texts))),
            "num_query_components": num_components,
        }
        audit_rows.append(row)

    offline_agreement = compare_offline_triggers(audit_rows, offline_triggers)
    decision = decide_runtime_audit(audit_rows, offline_agreement)

    audit_fields = [
        "task", "sample_index", "sample_id",
        "runtime_original_selected_block_ids", "runtime_qdc_selected_block_ids",
        "qdc_triggered", "qdc_trigger_reason",
        "original_component_coverage", "qdc_component_coverage", "coverage_delta",
        "missing_components_before", "missing_components_after",
        "selected_jaccard_vs_original",
        "removed_block_ids", "added_block_ids",
        "budget_preserved", "selected_token_count_original", "selected_token_count_qdc",
        "over_budget", "fallback_used",
    ]
    write_csv(output_dir / "runtime_qdc_selection_audit.csv", audit_rows, audit_fields)

    group_rows: List[Dict[str, Any]] = []
    for group in TASKS + ["all"]:
        subset = audit_rows if group == "all" else [row for row in audit_rows if row["task"] == group]
        if not subset:
            continue
        cov_deltas = [float(row["coverage_delta"]) for row in subset]
        jaccards = [float(row["selected_jaccard_vs_original"]) for row in subset]
        group_rows.append({
            "group": group,
            "sample_count": len(subset),
            "runtime_trigger_count": sum(1 for row in subset if row["qdc_triggered"]),
            "budget_violation_count": sum(1 for row in subset if not row["budget_preserved"]),
            "fallback_count": sum(1 for row in subset if row["fallback_used"]),
            "mean_original_component_coverage": statistics.mean(float(row["original_component_coverage"]) for row in subset),
            "mean_qdc_component_coverage": statistics.mean(float(row["qdc_component_coverage"]) for row in subset),
            "mean_coverage_delta": statistics.mean(cov_deltas),
            "median_coverage_delta": statistics.median(cov_deltas),
            "p25_coverage_delta": percentile(cov_deltas, 25),
            "p75_coverage_delta": percentile(cov_deltas, 75),
            "median_selected_jaccard": statistics.median(jaccards),
            "p25_selected_jaccard": percentile(jaccards, 25),
            "p75_selected_jaccard": percentile(jaccards, 75),
            "offline_trigger_agreement_count": sum(1 for row in subset if row["offline_runtime_trigger_match"]),
        })
    write_csv(output_dir / "runtime_qdc_group_summary.csv", group_rows, list(group_rows[0].keys()))

    all_summary = next(row for row in group_rows if row["group"] == "all")
    hotpot_summary = next((row for row in group_rows if row["group"] == "hotpotqa"), {})
    musique_summary = next((row for row in group_rows if row["group"] == "musique"), {})

    case_study = build_case_study(audit_rows)
    (output_dir / "runtime_qdc_case_study.md").write_text(case_study, encoding="utf-8")

    report_lines = [
        "# QDC Runtime Selection Audit (Held-out64)",
        "",
        f"- Output: `{output_dir}`",
        f"- Offline audit reference: `{args.offline_audit_csv}`",
        f"- Decision: **{decision.get('decision')}**",
        "",
        "## Runtime trigger counts",
        f"1. Runtime trigger count: **{all_summary['runtime_trigger_count']}/{all_summary['sample_count']}**",
        f"2. HotpotQA trigger count: **{hotpot_summary.get('runtime_trigger_count', 0)}**",
        f"3. MuSiQue trigger count: **{musique_summary.get('runtime_trigger_count', 0)}**",
        f"4. Budget violation count: **{all_summary['budget_violation_count']}**",
        f"5. Fallback count: **{all_summary['fallback_count']}**",
        "",
        "## Coverage",
        f"- Original component coverage: **{all_summary['mean_original_component_coverage']:.4f}**",
        f"- QDC component coverage: **{all_summary['mean_qdc_component_coverage']:.4f}**",
        f"- Mean coverage delta: **{all_summary['mean_coverage_delta']:+.4f}**",
        f"- Median coverage delta: **{all_summary['median_coverage_delta']:+.4f}**",
        f"- P25/P75 coverage delta: **{all_summary['p25_coverage_delta']:+.4f} / {all_summary['p75_coverage_delta']:+.4f}**",
        "",
        "## Selected Jaccard",
        f"- Median: **{all_summary['median_selected_jaccard']:.4f}**",
        f"- P25/P75: **{all_summary['p25_selected_jaccard']:.4f} / {all_summary['p75_selected_jaccard']:.4f}**",
        "",
        "## Offline vs runtime trigger agreement",
        f"- Compared samples: **{offline_agreement['compared_samples']}**",
        f"- Agreement count: **{offline_agreement['agreement_count']}**",
        f"- Agreement rate: **{offline_agreement['agreement_rate']:.3f}**",
        f"- Offline-only triggers: **{offline_agreement['offline_only_trigger_count']}**",
        f"- Runtime-only triggers: **{offline_agreement['runtime_only_trigger_count']}**",
        "",
        "## Tiny quality smoke recommendation",
        f"- Worth re-running tiny quality smoke: **{decision.get('worth_tiny_quality_smoke', False)}**",
        "",
        "## Decision reasons",
    ]
    report_lines.extend(f"- {reason}" for reason in decision.get("reasons", []))
    report_lines.append("")
    (output_dir / "runtime_qdc_decision.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    write_json(output_dir / "runtime_qdc_decision.json", {**decision, "offline_agreement": offline_agreement})

    print("\n".join(report_lines))
    print(f"Wrote results to {output_dir}")


if __name__ == "__main__":
    main()
