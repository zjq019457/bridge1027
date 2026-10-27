#!/usr/bin/env python3
"""Tiny prefill-only score instrumentation for UASR.

This runner is intentionally narrow: it runs SlimInfer's frozen original
fairness-closure baseline on a fixed 20-sample HotpotQA/MuSiQue set, emits
selector score dumps, validates that the selected blocks match the existing
200-sample diagnostic, and then runs the CPU-only UASR audit/replay scripts.

It never decodes, never invokes a scorer, and never writes historical result
directories.
"""
from __future__ import annotations

import argparse
import csv
import json
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

PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
if not PYTHON.exists():
    PYTHON = Path(sys.executable)

BASE_RUNNER = REPO_ROOT / "experiments" / "run_qwen25_sliminfer_variants.py"
DEFAULT_HISTORICAL_DIR = REPO_ROOT / "results" / "drm_budget_neutral_swap_multihop200_20260706_042717"

TINY_SAMPLE_INDICES = {
    "hotpotqa": [0, 1, 2, 3, 4, 5, 41, 56, 57, 117],
    "musique": [0, 1, 2, 3, 4, 20, 50, 54, 56, 134],
}


JsonRow = Dict[str, Any]
Key = Tuple[str, int]


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                field: json.dumps(row.get(field), ensure_ascii=False) if isinstance(row.get(field), (list, dict)) else row.get(field, "")
                for field in fields
            })


def key_of(row: Mapping[str, Any]) -> Key:
    return str(row["task"]), int(row["sample_index"])


def gpu_free_mib() -> Tuple[int, Dict[str, Any]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.free",
        "--format=csv,noheader,nounits",
    ]
    proc = subprocess.run(command, text=True, capture_output=True, check=True)
    rows = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if not rows:
        raise RuntimeError("nvidia-smi returned no GPU rows")
    first = [part.strip() for part in rows[0].split(",")]
    if len(first) != 4:
        raise RuntimeError(f"unexpected nvidia-smi output: {rows[0]}")
    return int(first[3]), {
        "gpu_index": int(first[0]),
        "gpu_name": first[1],
        "memory_total_mib": int(first[2]),
        "memory_free_mib": int(first[3]),
    }


def load_historical_original(historical_dir: Path) -> Dict[Key, JsonRow]:
    by_key: Dict[Key, JsonRow] = {}
    for row in read_jsonl(historical_dir / "audit" / "per_sample_audit.jsonl"):
        if str(row.get("method_key")) == "sliminfer_original":
            by_key[key_of(row)] = row
    missing = [key for task, indices in TINY_SAMPLE_INDICES.items() for key in [(task, idx) for idx in indices] if key not in by_key]
    if missing:
        raise RuntimeError(f"historical sliminfer_original audit missing tiny keys: {missing}")
    return by_key


def sample_indices_arg(task: str) -> str:
    return f"{task}:{','.join(str(x) for x in TINY_SAMPLE_INDICES[task])}"


def build_prefill_command(args: argparse.Namespace, protocol: Mapping[str, Any], task: str, cell_dir: Path, raw_dump_path: Path) -> List[str]:
    prune_layers = ",".join(str(x) for x in protocol.get("prune_layers", [9, 18, 26]))
    retain_tokens = ",".join(str(x) for x in protocol.get("retain_tokens", [12288, 6144, 4096]))
    block_size = int(protocol.get("block_size", 64))
    debug_layers = ",".join(str(int(x) + 1) for x in protocol.get("prune_layers", [9, 18, 26]))
    audit_dir = cell_dir / "audit"
    return [
        str(PYTHON),
        str(BASE_RUNNER),
        "--model_path",
        str(protocol["model_name"]),
        "--tokenizer_path",
        str(protocol["model_name"]),
        "--model_family",
        "qwen2",
        "--dtype",
        "fp16",
        "--device",
        "cuda",
        "--backend",
        args.backend,
        "--eval",
        "gate_audit",
        "--longbench_dir",
        str(protocol["longbench_dir"]),
        "--longbench_tasks",
        task,
        "--longbench_max_samples",
        str(len(TINY_SAMPLE_INDICES[task])),
        "--longbench_sample_indices_by_task",
        sample_indices_arg(task),
        "--longbench_subset_name",
        "uasr_score_instrumentation_tiny",
        "--generation_max_new_tokens",
        "0",
        "--max_model_len",
        str(protocol.get("generation_config", {}).get("max_model_len", 32768)),
        "--method",
        "sliminfer_variant",
        "--scoring",
        str(protocol.get("base_scoring", "delta_o")),
        "--fairness_base_scoring",
        str(protocol.get("base_scoring", "delta_o")),
        "--decode_policy",
        str(protocol.get("decode_policy", "reuse_prefill")),
        "--rescue_policy",
        "none",
        "--rescue_score",
        "delta_o",
        "--rescue_ratio",
        str(protocol.get("rescue_ratio", 0.05)),
        "--prune_layers",
        prune_layers,
        "--retain_tokens",
        retain_tokens,
        "--block_size",
        str(block_size),
        "--token_unit_size",
        str(protocol.get("token_unit_size", 8)),
        "--local_query_window",
        "4",
        "--swap_threshold",
        "0.9",
        "--temperature",
        "0",
        "--top_p",
        "1.0",
        "--seed",
        str(protocol.get("seed", 0)),
        "--fd_alpha_o",
        "1.0",
        "--fd_beta_h",
        "0.0",
        "--fd_gamma_s",
        "0.0",
        "--output_dir",
        str(cell_dir),
        "--isolated_run_id",
        f"uasr_score_instrumentation__sliminfer_original__{task}",
        "--uasr_score_dump_path",
        str(raw_dump_path),
        "--uasr_tail_width",
        str(args.tail_width),
        "--debug_used_block_audit_path",
        str(audit_dir / "used_blocks.jsonl"),
        "--debug_used_block_include_full",
        "true",
        "--debug_used_block_full_layers",
        debug_layers,
        "--smoke_mode",
        "0",
        "--max_smoke_samples",
        str(len(TINY_SAMPLE_INDICES[task])),
    ]


def run_task_cell(args: argparse.Namespace, protocol: Mapping[str, Any], output_dir: Path, task: str) -> Dict[str, Any]:
    minimum_mib = int(args.min_free_gpu_gb * 1024)
    free_mib, gpu = gpu_free_mib()
    if free_mib < minimum_mib:
        payload = {
            "status": "STOPPED_LOW_GPU_MEMORY",
            "task": task,
            "free_mib": free_mib,
            "required_mib": minimum_mib,
            "gpu": gpu,
        }
        write_json(output_dir / "STOPPED_LOW_GPU_MEMORY.json", payload)
        raise RuntimeError(f"GPU gate failed before {task}: free={free_mib} MiB required={minimum_mib} MiB")
    cell_dir = output_dir / "runs" / "sliminfer_original" / task
    raw_dump_path = output_dir / "raw" / f"{task}_score_dump.jsonl"
    command = build_prefill_command(args, protocol, task, cell_dir, raw_dump_path)
    (output_dir / "commands.jsonl").parent.mkdir(parents=True, exist_ok=True)
    with (output_dir / "commands.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"task": task, "command": command, "gpu_before": gpu}, ensure_ascii=False) + "\n")
    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")
    proc = subprocess.run(command, cwd=str(REPO_ROOT), text=True, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"prefill instrumentation cell failed for {task}: exit={proc.returncode}")
    return {"task": task, "gpu_before": gpu, "raw_dump_path": str(raw_dump_path), "cell_dir": str(cell_dir)}


def consolidate_score_dumps(output_dir: Path) -> List[JsonRow]:
    rows: List[JsonRow] = []
    for task in TINY_SAMPLE_INDICES:
        rows.extend(read_jsonl(output_dir / "raw" / f"{task}_score_dump.jsonl"))
    rows.sort(key=lambda row: (str(row.get("task")), int(row.get("sample_index", -1)), int(row.get("layer_id", -1))))
    write_jsonl(output_dir / "score_dump_audit.jsonl", rows)
    return rows


def final_layer_records(rows: Sequence[Mapping[str, Any]]) -> Dict[Key, JsonRow]:
    grouped: Dict[Key, List[JsonRow]] = defaultdict(list)
    for row in rows:
        grouped[key_of(row)].append(dict(row))
    selected: Dict[Key, JsonRow] = {}
    for key, items in grouped.items():
        selected[key] = sorted(items, key=lambda row: int(row.get("layer_id", -1)))[-1]
    return selected


def validate_and_summarize(output_dir: Path, score_rows: Sequence[Mapping[str, Any]], historical: Mapping[Key, Mapping[str, Any]]) -> List[JsonRow]:
    final_rows = final_layer_records(score_rows)
    expected_keys = [(task, idx) for task, indices in TINY_SAMPLE_INDICES.items() for idx in indices]
    missing = [key for key in expected_keys if key not in final_rows]
    if missing:
        write_json(output_dir / "STOPPED_SCORE_MISSING.json", {"missing_keys": missing})
        raise RuntimeError(f"score dump missing final-layer records: {missing}")

    summary_rows: List[JsonRow] = []
    mismatches: List[JsonRow] = []
    for key in expected_keys:
        row = final_rows[key]
        hist = historical[key]
        selected = [int(x) for x in row.get("selected_block_ids", [])]
        hist_blocks = [int(x) for x in hist.get("used_blocks", [])]
        used_tokens = [int(x) for x in row.get("used_tokens", [])]
        hist_tokens = [int(x) for x in hist.get("used_tokens", [])]
        selected_match = selected == hist_blocks
        used_tokens_match = used_tokens == hist_tokens
        layer_count = sum(1 for r in score_rows if key_of(r) == key)
        summary = {
            "task": key[0],
            "sample_index": key[1],
            "sample_id": row.get("sample_id", ""),
            "score_dump_available": bool(row.get("score_dump_available")),
            "final_layer_id": row.get("layer_id"),
            "layer_count": layer_count,
            "prompt_block_count": row.get("prompt_block_count"),
            "budget_k": row.get("budget_k"),
            "selected_block_count": len(selected),
            "pruned_block_count": len(row.get("pruned_block_ids", [])),
            "used_tokens_count": len(used_tokens),
            "cutoff_margin": row.get("cutoff_margin"),
            "cutoff_margin_ratio": row.get("cutoff_margin_ratio"),
            "num_blocks_within_eps_0_01": row.get("num_blocks_within_eps_0_01"),
            "num_blocks_within_eps_0_02": row.get("num_blocks_within_eps_0_02"),
            "num_blocks_within_margin_eps": row.get("num_blocks_within_margin_eps"),
            "score_entropy_near_cutoff": row.get("score_entropy_near_cutoff"),
            "score_std_near_cutoff": row.get("score_std_near_cutoff"),
            "score_iqr_near_cutoff": row.get("score_iqr_near_cutoff"),
            "selected_matches_historical": selected_match,
            "used_tokens_match_historical": used_tokens_match,
            "block_text_available": bool(row.get("block_text_available")),
            "query_text_available": bool(row.get("query_text_available")),
        }
        summary_rows.append(summary)
        if not selected_match or not used_tokens_match:
            mismatches.append({
                "task": key[0],
                "sample_index": key[1],
                "sample_id": row.get("sample_id", ""),
                "selected_matches_historical": selected_match,
                "used_tokens_match_historical": used_tokens_match,
                "new_selected_first50": selected[:50],
                "historical_selected_first50": hist_blocks[:50],
            })
    if mismatches:
        write_json(output_dir / "STOPPED_SELECTION_MISMATCH.json", {"mismatches": mismatches})
        raise RuntimeError(f"selection mismatch vs historical SlimInfer original: {mismatches[:2]}")
    write_csv(
        output_dir / "score_dump_summary.csv",
        summary_rows,
        [
            "task",
            "sample_index",
            "sample_id",
            "score_dump_available",
            "final_layer_id",
            "layer_count",
            "prompt_block_count",
            "budget_k",
            "selected_block_count",
            "pruned_block_count",
            "used_tokens_count",
            "cutoff_margin",
            "cutoff_margin_ratio",
            "num_blocks_within_eps_0_01",
            "num_blocks_within_eps_0_02",
            "num_blocks_within_margin_eps",
            "score_entropy_near_cutoff",
            "score_std_near_cutoff",
            "score_iqr_near_cutoff",
            "selected_matches_historical",
            "used_tokens_match_historical",
            "block_text_available",
            "query_text_available",
        ],
    )
    return summary_rows


def run_offline_uasr(output_dir: Path, historical_dir: Path, anchor_dir: Path) -> Dict[str, str]:
    audit_dir = output_dir / "uasr_uncertainty_audit"
    replay_dir = audit_dir / "offline_replay"
    cmd_audit = [
        str(PYTHON),
        str(REPO_ROOT / "tools" / "run_uasr_uncertainty_audit.py"),
        "--historical_dir",
        str(historical_dir),
        "--score_dump_dir",
        str(output_dir),
        "--output_dir",
        str(audit_dir),
    ]
    subprocess.run(cmd_audit, cwd=str(REPO_ROOT), text=True, check=True)
    cmd_replay = [
        str(PYTHON),
        str(REPO_ROOT / "tools" / "replay_uasr_gate.py"),
        "--audit_dir",
        str(audit_dir),
        "--historical_dir",
        str(historical_dir),
        "--anchor_dir",
        str(anchor_dir),
        "--output_dir",
        str(replay_dir),
    ]
    subprocess.run(cmd_replay, cwd=str(REPO_ROOT), text=True, check=True)
    return {"audit_dir": str(audit_dir), "replay_dir": str(replay_dir)}


def write_report(
    output_dir: Path,
    args: argparse.Namespace,
    protocol: Mapping[str, Any],
    cell_results: Sequence[Mapping[str, Any]],
    summary_rows: Sequence[Mapping[str, Any]],
    offline_paths: Mapping[str, str],
) -> None:
    margins = [float(row["cutoff_margin"]) for row in summary_rows if row.get("cutoff_margin") is not None]
    min_free = min(int(row["gpu_before"]["memory_free_mib"]) for row in cell_results) if cell_results else None
    lines = [
        "# UASR Score Instrumentation Tiny Audit",
        "",
        f"- Result directory: `{output_dir.resolve()}`",
        f"- Historical diagnostic: `{Path(args.historical_dir).resolve()}`",
        "- Scope: fixed 20-sample HotpotQA/MuSiQue prefill-only selector audit",
        "- SlimInfer selector logic changed: **No**",
        "- Decode/generation performed: **No**",
        "- Scorer invoked: **No**",
        "- Gold answer used for gate: **No**",
        f"- Protocol: decode_policy=`{protocol.get('decode_policy')}`, base_scoring=`{protocol.get('base_scoring')}`, prune_layers=`{protocol.get('prune_layers')}`, retain_tokens=`{protocol.get('retain_tokens')}`",
        f"- Minimum free GPU memory before task cells: {min_free} MiB",
        "",
        "## Samples",
        "",
    ]
    for task, indices in TINY_SAMPLE_INDICES.items():
        lines.append(f"- {task}: `{indices}`")
    lines.extend([
        "",
        "## Score field coverage",
        "",
        f"- Final-layer score records: {len(summary_rows)}/20",
        f"- `cutoff_margin` available: {len(margins)}/20",
        f"- `block_text_available`: {sum(bool(row.get('block_text_available')) for row in summary_rows)}/20",
        f"- selected/used_tokens match historical SlimInfer original: {sum(bool(row.get('selected_matches_historical') and row.get('used_tokens_match_historical')) for row in summary_rows)}/20",
        "",
        "## Cutoff margin summary",
        "",
    ])
    if margins:
        lines.extend([
            f"- mean: {sum(margins) / len(margins):.6f}",
            f"- min: {min(margins):.6f}",
            f"- max: {max(margins):.6f}",
        ])
    else:
        lines.append("- unavailable")
    lines.extend([
        "",
        "## Offline UASR replay outputs",
        "",
        f"- UASR uncertainty audit: `{Path(offline_paths['audit_dir']).resolve()}`",
        f"- UASR offline replay: `{Path(offline_paths['replay_dir']).resolve()}`",
        "",
        "This report is an observability artifact only; it is not a quality generation experiment.",
    ])
    (output_dir / "uasr_score_instrumentation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical_dir", default=str(DEFAULT_HISTORICAL_DIR))
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--min_free_gpu_gb", type=float, default=20.0)
    parser.add_argument("--backend", default="flash_attention_2", choices=["flash_attention_2", "sdpa", "auto"])
    parser.add_argument("--tail_width", type=int, default=8)
    parser.add_argument("--anchor_dir", default=str(REPO_ROOT / "results" / "protected_evidence_anchor_audit_20260706_214256"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    historical_dir = Path(args.historical_dir)
    config = read_json(historical_dir / "config.json")
    protocol = config["protocol"]
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else REPO_ROOT / "results" / f"uasr_score_instrumentation_tiny_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json(output_dir / "run_config.json", {
        "historical_dir": str(historical_dir),
        "fixed_sample_indices": TINY_SAMPLE_INDICES,
        "min_free_gpu_gb": args.min_free_gpu_gb,
        "backend": args.backend,
        "tail_width": args.tail_width,
        "decode_generation_performed": False,
        "scorer_invoked": False,
    })
    historical = load_historical_original(historical_dir)
    cell_results = []
    for task in TINY_SAMPLE_INDICES:
        cell_results.append(run_task_cell(args, protocol, output_dir, task))
    score_rows = consolidate_score_dumps(output_dir)
    summary_rows = validate_and_summarize(output_dir, score_rows, historical)
    offline_paths = run_offline_uasr(output_dir, historical_dir, Path(args.anchor_dir))
    write_report(output_dir, args, protocol, cell_results, summary_rows, offline_paths)
    print(output_dir)


if __name__ == "__main__":
    main()
