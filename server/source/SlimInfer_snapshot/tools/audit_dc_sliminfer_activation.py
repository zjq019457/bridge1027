#!/usr/bin/env python3
"""Audit whether DC-SlimInfer reaches each pruning-layer selector.

This is audit-only: prefill forward pass only, no decode, no generation, and no
LongBench scorer.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = Path(
    "/home/yu-bai/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/"
    "snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
)
DEFAULT_LONGBENCH_DIR = Path("/home/yu-bai/Desktop/jiaqizhang/HeadKV/data/LongBench")
FIXED_HOTPOTQA = [0, 1, 2, 3, 4, 5, 41, 56, 57, 117]
FIXED_MUSIQUE = [0, 1, 2, 3, 4, 20, 50, 54, 56, 134]
PRUNE_LAYERS = [9, 18, 26]
FIXED_SPEC = (
    "hotpotqa:" + ",".join(str(x) for x in FIXED_HOTPOTQA)
    + ";musique:" + ",".join(str(x) for x in FIXED_MUSIQUE)
)
JsonRow = Dict[str, Any]
Key = Tuple[str, int, int]


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


def check_gpu(min_free_gpu_gb: float) -> Dict[str, Any]:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free", "--format=csv,noheader,nounits"],
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip().splitlines()
    rows = []
    for line in output:
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
    selected = max(rows, key=lambda row: row["memory_free_gib"])
    if float(selected["memory_free_gib"]) < float(min_free_gpu_gb):
        raise RuntimeError(
            f"free GPU memory {selected['memory_free_gib']:.2f} GiB < required {float(min_free_gpu_gb):.2f} GiB"
        )
    return {"rows": rows, "selected": selected, "min_free_gpu_gb": float(min_free_gpu_gb)}


def python_bin() -> Path:
    candidate = REPO_ROOT / ".venv" / "bin" / "python"
    return candidate if candidate.exists() else Path(sys.executable)


def runner_cmd(
    args: argparse.Namespace,
    run_dir: Path,
    activation_path: Path,
    selection_path: Path,
    retain_tokens: str,
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
        "--eval", "gate_audit",
        "--longbench_dir", str(args.longbench_dir),
        "--longbench_tasks", "hotpotqa,musique",
        "--longbench_sample_indices_by_task", FIXED_SPEC,
        "--longbench_subset_name", "dc_sliminfer_activation_audit",
        "--generation_max_new_tokens", "32",
        "--max_model_len", "32768",
        "--method", "dc_sliminfer",
        "--decode_policy", "original",
        "--rescue_policy", "none",
        "--prune_layers", ",".join(str(x) for x in PRUNE_LAYERS),
        "--retain_tokens", retain_tokens,
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
        "--dc_activation_audit_path", str(activation_path),
        "--dc_selection_audit_path", str(selection_path),
        "--output_dir", str(run_dir),
        "--min_free_gpu_gb", "0",
    ]


def run_audit(
    args: argparse.Namespace,
    output_dir: Path,
    label: str,
    retain_tokens: str,
) -> Dict[str, Any]:
    run_dir = output_dir / f"run_{label}"
    activation_path = output_dir / f"{label}_runtime_activation_audit.jsonl"
    selection_path = output_dir / f"{label}_selection_audit.jsonl"
    activation_path.write_text("", encoding="utf-8")
    selection_path.write_text("", encoding="utf-8")
    cmd = runner_cmd(args, run_dir, activation_path, selection_path, retain_tokens)
    write_json(output_dir / f"{label}_command.json", {"command": cmd})
    with (output_dir / f"{label}_stdout.log").open("w", encoding="utf-8") as stdout, (
        output_dir / f"{label}_stderr.log"
    ).open("w", encoding="utf-8") as stderr:
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), text=True, stdout=stdout, stderr=stderr)
    stdout_text = (output_dir / f"{label}_stdout.log").read_text(encoding="utf-8", errors="ignore")
    stderr_text = (output_dir / f"{label}_stderr.log").read_text(encoding="utf-8", errors="ignore")
    if "CUDA out of memory" in stdout_text or "CUDA out of memory" in stderr_text:
        raise RuntimeError(f"CUDA OOM during {label} activation audit")
    if proc.returncode != 0:
        raise RuntimeError(f"{label} activation audit failed with return code {proc.returncode}")
    manifest = read_jsonl(run_dir / "gate_audit_manifest.jsonl")
    runtime_rows = read_jsonl(activation_path, missing_ok=True)
    selection_rows = read_jsonl(selection_path, missing_ok=True)
    return {
        "label": label,
        "run_dir": str(run_dir),
        "activation_path": str(activation_path),
        "selection_path": str(selection_path),
        "manifest_rows": manifest,
        "runtime_rows": runtime_rows,
        "selection_rows": selection_rows,
    }


def row_key(row: Mapping[str, Any]) -> Key:
    return str(row.get("task", "")), int(row.get("sample_index", -1)), int(row.get("layer_id", -1))


def expected_sample_rows(manifest_rows: Sequence[Mapping[str, Any]]) -> List[Tuple[str, int, Any]]:
    seen = set()
    rows: List[Tuple[str, int, Any]] = []
    for row in manifest_rows:
        task = str(row.get("task", ""))
        sample_index = int(row.get("sample_index", -1))
        key = (task, sample_index)
        if task and sample_index >= 0 and key not in seen:
            rows.append((task, sample_index, row.get("sample_id", sample_index)))
            seen.add(key)
    rows.sort(key=lambda item: (item[0], item[1]))
    return rows


def missing_row(task: str, sample_index: int, sample_id: Any, layer_id: int) -> JsonRow:
    return {
        "task": task,
        "sample_index": sample_index,
        "sample_id": sample_id,
        "layer_id": int(layer_id),
        "method": "dc_sliminfer",
        "phase": "prefill",
        "dc_flag_received": False,
        "entered_pruning_layer": False,
        "entered_virtual_block_path": False,
        "entered_dc_selector": False,
        "entered_dc_selector_before_virtual_return": False,
        "virtual_block_path_bypassed_by_dc": False,
        "audit_row_written": False,
        "num_blocks": 0,
        "num_keep_target": 0,
        "num_pruned_candidates": 0,
        "keep_all": False,
        "early_return_reason": "missing_runtime_activation_row",
        "virtual_return_reason": "",
        "must_keep_return_reason": "",
        "selection_pressure": False,
        "candidate_block_ids_available": False,
        "candidate_scores_available": False,
        "candidate_text_available": False,
        "query_text_available": False,
        "original_virtual_selected_block_ids": [],
        "original_selected_block_ids": [],
        "dc_selected_block_ids": [],
        "selection_changed": False,
        "selected_jaccard": 1.0,
        "original_redundancy": 0.0,
        "dc_redundancy": 0.0,
        "original_query_overlap": 0.0,
        "dc_query_overlap": 0.0,
        "original_coverage_bins": [],
        "dc_coverage_bins": [],
        "budget_preserved": False,
        "fallback_reason": "",
    }


def expand_rows(result: Mapping[str, Any]) -> List[JsonRow]:
    runtime_by_key: Dict[Key, JsonRow] = {}
    duplicates = 0
    for row in result["runtime_rows"]:
        key = row_key(row)
        if key in runtime_by_key:
            duplicates += 1
        runtime_by_key[key] = dict(row)
    expanded: List[JsonRow] = []
    for task, sample_index, sample_id in expected_sample_rows(result["manifest_rows"]):
        for layer_id in PRUNE_LAYERS:
            key = (task, sample_index, layer_id)
            expanded.append(runtime_by_key.get(key, missing_row(task, sample_index, sample_id, layer_id)))
    for row in expanded:
        row.setdefault("duplicate_runtime_rows", duplicates)
    return expanded


def count_rows(rows: Sequence[Mapping[str, Any]], key: str) -> int:
    return sum(1 for row in rows if bool(row.get(key)))


def mean_float(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    values: List[float] = []
    for row in rows:
        try:
            values.append(float(row.get(key, 0.0) or 0.0))
        except (TypeError, ValueError):
            continue
    return float(sum(values) / max(1, len(values)))


def mean_bin_count(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    values = [len(row.get(key, []) or []) for row in rows]
    return float(sum(values) / max(1, len(values)))


def summarize_rows(rows: Sequence[Mapping[str, Any]], runtime_rows: Sequence[Mapping[str, Any]], selection_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    pressure_rows = [row for row in rows if bool(row.get("selection_pressure"))]
    changed_rows = [row for row in rows if bool(row.get("selection_changed"))]
    budget_equal_rows = [
        row for row in rows
        if bool(row.get("budget_preserved", False))
        or len(row.get("original_selected_block_ids", []) or []) == len(row.get("dc_selected_block_ids", []) or [])
    ]
    virtual_reasons: Dict[str, int] = {}
    early_reasons: Dict[str, int] = {}
    fallback_reasons: Dict[str, int] = {}
    for row in rows:
        for field, target in [
            ("virtual_return_reason", virtual_reasons),
            ("early_return_reason", early_reasons),
            ("fallback_reason", fallback_reasons),
        ]:
            reason = str(row.get(field, "") or "")
            if reason:
                target[reason] = target.get(reason, 0) + 1
    feature_nonzero_rows = sum(
        1 for row in rows
        if float(row.get("original_redundancy", 0.0) or 0.0) != 0.0
        or float(row.get("dc_redundancy", 0.0) or 0.0) != 0.0
        or float(row.get("original_query_overlap", 0.0) or 0.0) != 0.0
        or float(row.get("dc_query_overlap", 0.0) or 0.0) != 0.0
    )
    return {
        "expected_rows": len(rows),
        "runtime_rows": len(runtime_rows),
        "actual_rows": len(rows),
        "selection_audit_rows": len(selection_rows),
        "dc_flag_received_rows": count_rows(rows, "dc_flag_received"),
        "entered_pruning_layer_rows": count_rows(rows, "entered_pruning_layer"),
        "entered_virtual_block_path_rows": count_rows(rows, "entered_virtual_block_path"),
        "entered_dc_selector_rows": count_rows(rows, "entered_dc_selector"),
        "entered_dc_selector_before_virtual_return_rows": count_rows(rows, "entered_dc_selector_before_virtual_return"),
        "virtual_block_path_bypassed_by_dc_rows": count_rows(rows, "virtual_block_path_bypassed_by_dc"),
        "audit_row_written_rows": count_rows(rows, "audit_row_written"),
        "keep_all_rows": count_rows(rows, "keep_all"),
        "selection_pressure_rows": len(pressure_rows),
        "selection_changed_rows": len(changed_rows),
        "budget_equal_rows": len(budget_equal_rows),
        "budget_mismatch_rows": len(rows) - len(budget_equal_rows),
        "candidate_block_ids_available_rows": count_rows(rows, "candidate_block_ids_available"),
        "candidate_scores_available_rows": count_rows(rows, "candidate_scores_available"),
        "candidate_text_available_rows": count_rows(rows, "candidate_text_available"),
        "query_text_available_rows": count_rows(rows, "query_text_available"),
        "virtual_return_reasons": virtual_reasons,
        "early_return_reasons": early_reasons,
        "fallback_reasons": fallback_reasons,
        "fallback_count": sum(1 for row in rows if str(row.get("fallback_reason", "") or "")),
        "feature_nonzero_rows": feature_nonzero_rows,
        "pressure_entered_dc_selector_rows": sum(1 for row in pressure_rows if bool(row.get("entered_dc_selector"))),
        "pressure_entered_dc_selector_before_virtual_return_rows": sum(
            1 for row in pressure_rows if bool(row.get("entered_dc_selector_before_virtual_return"))
        ),
        "pressure_virtual_return_rows": sum(1 for row in pressure_rows if bool(row.get("virtual_return_reason"))),
        "selected_jaccard_mean": mean_float(rows, "selected_jaccard"),
        "pressure_selected_jaccard_mean": mean_float(pressure_rows, "selected_jaccard"),
        "original_redundancy_mean": mean_float(rows, "original_redundancy"),
        "dc_redundancy_mean": mean_float(rows, "dc_redundancy"),
        "redundancy_delta_original_minus_dc": mean_float(rows, "original_redundancy") - mean_float(rows, "dc_redundancy"),
        "original_query_overlap_mean": mean_float(rows, "original_query_overlap"),
        "dc_query_overlap_mean": mean_float(rows, "dc_query_overlap"),
        "query_overlap_delta_dc_minus_original": mean_float(rows, "dc_query_overlap") - mean_float(rows, "original_query_overlap"),
        "original_coverage_bin_count_mean": mean_bin_count(rows, "original_coverage_bins"),
        "dc_coverage_bin_count_mean": mean_bin_count(rows, "dc_coverage_bins"),
        "coverage_bin_count_delta_dc_minus_original": mean_bin_count(rows, "dc_coverage_bins") - mean_bin_count(rows, "original_coverage_bins"),
        "changed_samples": sorted({
            f"{row.get('task')}:{row.get('sample_index')}"
            for row in changed_rows
        }),
    }


def write_report(output_dir: Path, summary: Mapping[str, Any], stress_summary: Mapping[str, Any] | None) -> None:
    previous_reason = (
        "上一轮只使用 selection_audit hook；该 hook 只在 _select_dc_sliminfer_ids 被调用时写行。"
        "activation audit 显示 selection-pressure rows 是否在到达 DC selector 前被 keep-all / virtual / must-keep return 截断。"
    )
    if summary.get("pressure_virtual_return_rows", 0):
        previous_reason = (
            "上一轮只有 1 row 的主要原因是 selection-pressure rows 走了 virtual block path，"
            "在进入 _select_dc_sliminfer_ids 前返回；selection_audit 只记录到了少量 all-kept/selector rows。"
        )
    elif summary.get("keep_all_rows", 0) == summary.get("actual_rows", 0):
        previous_reason = "上一轮只有 1 row 是因为当前 fixed samples/budget 基本 keep-all，DC selector 没有 selection pressure。"

    recommend_quality = bool(
        summary.get("selection_changed_rows", 0) > 0
        and summary.get("budget_mismatch_rows", 0) == 0
        and (
            float(summary.get("redundancy_delta_original_minus_dc", 0.0) or 0.0) > 0.0
            or float(summary.get("coverage_bin_count_delta_dc_minus_original", 0.0) or 0.0) > 0.0
        )
    )
    if summary.get("selection_pressure_rows", 0) == 0:
        quality_reason = "No: current budget/sample has no selection-pressure rows."
    elif summary.get("pressure_entered_dc_selector_rows", 0) == 0:
        quality_reason = "No: pressure rows do not enter DC selector; fix activation before quality."
    elif summary.get("selection_changed_rows", 0) == 0:
        quality_reason = "No: DC selector entered but did not change selection."
    elif summary.get("budget_mismatch_rows", 0) != 0:
        quality_reason = "No: DC selector changed selection but introduced budget mismatch."
    elif not (
        float(summary.get("redundancy_delta_original_minus_dc", 0.0) or 0.0) > 0.0
        or float(summary.get("coverage_bin_count_delta_dc_minus_original", 0.0) or 0.0) > 0.0
    ):
        quality_reason = "No: selection changed, but redundancy/coverage did not improve."
    else:
        quality_reason = "Yes: DC changes selection under pressure; next turn may run tiny quality smoke."

    lines = [
        "# DC-SlimInfer Activation Audit",
        "",
        f"- Result directory: `{output_dir}`",
        f"- Expected rows: {summary['expected_rows']} = 20 x {len(PRUNE_LAYERS)} prune layers",
        f"- Actual rows: {summary['actual_rows']}",
        f"- Runtime rows written: {summary['runtime_rows']}",
        f"- Selection audit rows: {summary['selection_audit_rows']}",
        f"- Entered DC selector rows: {summary['entered_dc_selector_rows']}",
        f"- Entered virtual block path rows: {summary['entered_virtual_block_path_rows']}",
        f"- Entered DC selector before virtual return rows: {summary['entered_dc_selector_before_virtual_return_rows']}",
        f"- Virtual block path bypassed by DC rows: {summary['virtual_block_path_bypassed_by_dc_rows']}",
        f"- Keep-all rows: {summary['keep_all_rows']}",
        f"- Selection-pressure rows: {summary['selection_pressure_rows']}",
        f"- Selection-changed rows: {summary['selection_changed_rows']}",
        f"- Selected Jaccard mean: {float(summary['selected_jaccard_mean']):.4f}",
        f"- Pressure selected Jaccard mean: {float(summary['pressure_selected_jaccard_mean']):.4f}",
        f"- Original redundancy mean: {float(summary['original_redundancy_mean']):.4f}",
        f"- DC redundancy mean: {float(summary['dc_redundancy_mean']):.4f}",
        f"- Redundancy delta original-minus-DC: {float(summary['redundancy_delta_original_minus_dc']):.4f}",
        f"- Original query overlap mean: {float(summary['original_query_overlap_mean']):.4f}",
        f"- DC query overlap mean: {float(summary['dc_query_overlap_mean']):.4f}",
        f"- Query overlap delta DC-minus-original: {float(summary['query_overlap_delta_dc_minus_original']):.4f}",
        f"- Original coverage bin count mean: {float(summary['original_coverage_bin_count_mean']):.4f}",
        f"- DC coverage bin count mean: {float(summary['dc_coverage_bin_count_mean']):.4f}",
        f"- Coverage bin count delta DC-minus-original: {float(summary['coverage_bin_count_delta_dc_minus_original']):.4f}",
        f"- Budget mismatch rows: {summary['budget_mismatch_rows']}",
        f"- Fallback count: {summary['fallback_count']}",
        f"- Candidate ids available rows: {summary['candidate_block_ids_available_rows']}",
        f"- Candidate scores available rows: {summary['candidate_scores_available_rows']}",
        f"- Candidate text available rows: {summary['candidate_text_available_rows']}",
        f"- Query text available rows: {summary['query_text_available_rows']}",
        f"- Virtual return reasons: `{json.dumps(summary['virtual_return_reasons'], ensure_ascii=False, sort_keys=True)}`",
        f"- Early return reasons: `{json.dumps(summary['early_return_reasons'], ensure_ascii=False, sort_keys=True)}`",
        f"- Fallback reasons: `{json.dumps(summary['fallback_reasons'], ensure_ascii=False, sort_keys=True)}`",
        "",
        "## Diagnosis",
        "",
        f"- Why previous audit had 1 row: {previous_reason}",
        f"- virtual_block_path_returned_before_dc_selector is zero: {summary.get('pressure_virtual_return_rows', 0) == 0}",
        f"- Need lower budget to test DC: {summary['selection_pressure_rows'] == 0}",
        f"- Recommend tiny quality rerun: {recommend_quality}. {quality_reason}",
    ]
    if stress_summary is not None:
        lines.extend([
            "",
            "## Stress Audit",
            "",
            "- Stress setting was audit-only; no generation/scorer.",
            f"- Stress expected rows: {stress_summary['expected_rows']}",
            f"- Stress selection-pressure rows: {stress_summary['selection_pressure_rows']}",
            f"- Stress entered DC selector rows: {stress_summary['entered_dc_selector_rows']}",
            f"- Stress selection-changed rows: {stress_summary['selection_changed_rows']}",
            f"- Stress budget mismatch rows: {stress_summary['budget_mismatch_rows']}",
        ])
    lines.append("")
    lines.append("This is activation observability only, not a quality result.")
    (output_dir / "dc_activation_audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--longbench_dir", type=Path, default=DEFAULT_LONGBENCH_DIR)
    parser.add_argument("--output_root", type=Path, default=Path("results"))
    parser.add_argument("--min_free_gpu_gb", type=float, default=20.0)
    parser.add_argument("--stress_retain_tokens", type=str, default="2048,1024,512")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (REPO_ROOT / args.output_root / f"dc_sliminfer_activation_audit_fix_{timestamp}").resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    gpu_preflight = check_gpu(args.min_free_gpu_gb)
    write_json(output_dir / "fixed_samples.json", {
        "hotpotqa": FIXED_HOTPOTQA,
        "musique": FIXED_MUSIQUE,
        "prune_layers": PRUNE_LAYERS,
        "quality_generation_performed": False,
        "scorer_invoked": False,
    })

    main_result = run_audit(args, output_dir, "main", "12288,6144,4096")
    expanded = expand_rows(main_result)
    summary = summarize_rows(expanded, main_result["runtime_rows"], main_result["selection_rows"])
    write_jsonl(output_dir / "activation_audit.jsonl", expanded)

    stress_summary = None
    if summary["selection_pressure_rows"] == 0:
        stress_result = run_audit(args, output_dir, "stress", args.stress_retain_tokens)
        stress_expanded = expand_rows(stress_result)
        stress_summary = summarize_rows(stress_expanded, stress_result["runtime_rows"], stress_result["selection_rows"])
        write_jsonl(output_dir / "stress_activation_audit.jsonl", stress_expanded)

    write_json(output_dir / "activation_audit_summary.json", {
        "result_dir": str(output_dir),
        "gpu_preflight": gpu_preflight,
        "summary": summary,
        "stress_summary": stress_summary,
        "quality_generation_performed": False,
        "scorer_invoked": False,
    })
    write_csv(output_dir / "activation_audit_summary.csv", [summary], [
        "expected_rows",
        "actual_rows",
        "runtime_rows",
        "selection_audit_rows",
        "entered_dc_selector_rows",
        "entered_virtual_block_path_rows",
        "entered_dc_selector_before_virtual_return_rows",
        "virtual_block_path_bypassed_by_dc_rows",
        "keep_all_rows",
        "selection_pressure_rows",
        "selection_changed_rows",
        "budget_mismatch_rows",
        "fallback_count",
        "pressure_entered_dc_selector_rows",
        "pressure_entered_dc_selector_before_virtual_return_rows",
        "pressure_virtual_return_rows",
        "selected_jaccard_mean",
        "pressure_selected_jaccard_mean",
        "original_redundancy_mean",
        "dc_redundancy_mean",
        "redundancy_delta_original_minus_dc",
        "original_query_overlap_mean",
        "dc_query_overlap_mean",
        "query_overlap_delta_dc_minus_original",
        "original_coverage_bin_count_mean",
        "dc_coverage_bin_count_mean",
        "coverage_bin_count_delta_dc_minus_original",
        "feature_nonzero_rows",
    ])
    write_report(output_dir, summary, stress_summary)
    print(json.dumps({
        "status": "ok",
        "output_dir": str(output_dir),
        "summary": summary,
        "stress_summary": stress_summary,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
