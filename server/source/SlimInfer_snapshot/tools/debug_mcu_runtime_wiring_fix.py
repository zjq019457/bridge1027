#!/usr/bin/env python3
"""Prefill/debug wiring check for MCU-SlimInfer runtime KV propagation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.mcu_sliminfer_selection import (
    load_frozen_selections_from_audit_csv,
    parse_json_field,
    write_frozen_selections_jsonl,
)

DEFAULT_MODEL_PATH = Path(
    "/home/yu-bai/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/"
    "snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
)
DEFAULT_LONGBENCH_DIR = Path("/home/yu-bai/Desktop/jiaqizhang/HeadKV/data/LongBench")
DEFAULT_AUDIT_DIR = Path("results/mcu_sliminfer_counterfactual_audit_20260708_203500")
TASKS = ["hotpotqa", "musique"]
SAMPLE_INDICES: Dict[str, List[int]] = {
    "hotpotqa": [0, 1, 2, 3, 4, 5],
    "musique": [0, 1, 2, 3, 4, 20],
}
GATE_LAYER = 26
RUNTIME_KV_LAYER = GATE_LAYER + 1
PRUNE_LAYERS = [9, 18, GATE_LAYER]

Key = Tuple[str, int]


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def sha_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def python_bin() -> Path:
    candidate = REPO_ROOT / ".venv" / "bin" / "python"
    return candidate if candidate.exists() else Path(sys.executable)


def build_sample_spec() -> str:
    return ";".join(f"{task}:" + ",".join(str(i) for i in SAMPLE_INDICES[task]) for task in TASKS)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def load_used_blocks_prefill(path: Path, layer_id: int) -> Dict[Key, Dict[str, Any]]:
    out: Dict[Key, Dict[str, Any]] = {}
    for row in read_jsonl(path):
        if str(row.get("phase", "")) != "prefill":
            continue
        if int(row.get("layer_id", -1)) != layer_id:
            continue
        out[(str(row["task"]), int(row["sample_index"]))] = row
    return out


def load_mcu_audit_prefill(path: Path, layer_id: int) -> Dict[Key, Dict[str, Any]]:
    out: Dict[Key, Dict[str, Any]] = {}
    for row in read_jsonl(path):
        if str(row.get("phase", "")) != "prefill":
            continue
        if int(row.get("layer_id", -1)) != layer_id:
            continue
        key = (str(row["task"]), int(row["sample_index"]))
        if key not in out:
            out[key] = row
    return out


def runner_cmd(
    output_dir: Path,
    method: str,
    sample_spec: str,
    frozen_path: Optional[Path],
    mcu_audit_path: Optional[Path],
    args: argparse.Namespace,
) -> List[str]:
    audit_dir = output_dir / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
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
        "--longbench_subset_name", "mcu_runtime_wiring_fix_debug",
        "--generation_max_new_tokens", "1",
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
        "--eval", "longbench",
        "--method", method,
        "--output_dir", str(output_dir),
        "--min_free_gpu_gb", "0",
        "--debug_used_block_audit_path", str(audit_dir / "used_blocks.jsonl"),
        "--debug_used_block_include_full", "true",
        "--debug_used_block_full_layers", str(RUNTIME_KV_LAYER),
    ]
    if method == "mcu_sliminfer" and frozen_path is not None:
        cmd.extend([
            "--mcu_frozen_selection_path", str(frozen_path),
            "--mcu_selection_audit_path", str(mcu_audit_path or audit_dir / "mcu_runtime_selection_audit.jsonl"),
        ])
    return cmd


def run_runner(cmd: Sequence[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(list(cmd), cwd=str(REPO_ROOT), capture_output=True, text=True)
    log_path.write_text(proc.stdout + "\n" + proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"runner failed rc={proc.returncode}; see {log_path}")


def decide(rows: Sequence[Mapping[str, Any]]) -> Tuple[str, List[str]]:
    reasons: List[str] = []
    n = len(rows)
    if n == 0:
        return "BUG_STOP_FIX_REQUIRED", ["no_samples"]

    audit_changed = sum(1 for r in rows if r.get("selection_changed_in_audit"))
    runtime_changed = sum(1 for r in rows if r.get("selection_changed_in_runtime"))
    removed_still = sum(1 for r in rows if r.get("removed_still_present_in_used_blocks"))
    added_missing = sum(1 for r in rows if not r.get("added_present_in_used_blocks"))
    budget_mismatch = sum(1 for r in rows if r.get("budget_mismatch"))
    fallback = sum(1 for r in rows if r.get("fallback_used"))
    alignment_errors = sum(1 for r in rows if not r.get("frozen_alignment_ok", True))

    if alignment_errors:
        reasons.append(f"frozen_alignment_errors={alignment_errors}")
    if audit_changed and runtime_changed < n:
        reasons.append(f"runtime_used_blocks_unchanged={n - runtime_changed}/{n}")
    if removed_still:
        reasons.append(f"removed_still_present={removed_still}")
    if added_missing:
        reasons.append(f"added_missing={added_missing}")
    if budget_mismatch:
        reasons.append(f"budget_mismatch={budget_mismatch}")
    if fallback:
        reasons.append(f"fallback_used={fallback}")

    if reasons:
        return "BUG_STOP_FIX_REQUIRED", reasons

    if audit_changed == n and runtime_changed == n and removed_still == 0 and added_missing == 0:
        return "MCU_RUNTIME_WIRING_FIXED_PASS", [
            f"audit_changed={audit_changed}/{n}",
            f"runtime_changed={runtime_changed}/{n}",
            f"runtime_kv_layer={RUNTIME_KV_LAYER}",
        ]
    return "BUG_STOP_FIX_REQUIRED", ["incomplete_pass_criteria"]


def build_rows(
    orig_used: Mapping[Key, Dict[str, Any]],
    mcu_used: Mapping[Key, Dict[str, Any]],
    mcu_audit: Mapping[Key, Dict[str, Any]],
    frozen_rows: Mapping[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    keys = sorted(set(orig_used) & set(mcu_used))
    for key in keys:
        task, sample_index = key
        o_row = orig_used[key]
        m_row = mcu_used[key]
        audit = mcu_audit.get(key, {})
        frozen = frozen_rows.get(f"{task}:{sample_index}", {})

        orig_blocks = [int(x) for x in o_row.get("used_block_ids", o_row.get("used_block_ids_first100", []))]
        mcu_blocks = [int(x) for x in m_row.get("used_block_ids", m_row.get("used_block_ids_first100", []))]
        orig_tokens = [int(x) for x in o_row.get("used_tokens", o_row.get("kv_abs_indices_first100", []))]
        mcu_tokens = [int(x) for x in m_row.get("used_tokens", m_row.get("kv_abs_indices_first100", []))]

        orig_sel = set(parse_json_field(audit.get("original_selected_block_ids")) or [])
        mcu_sel = set(parse_json_field(audit.get("mcu_selected_block_ids")) or frozen.get("mcu_selected_block_ids") or [])
        removed = set(parse_json_field(audit.get("removed_block_ids")) or [])
        added = set(parse_json_field(audit.get("added_block_ids")) or [])
        mcu_block_set = set(mcu_blocks)

        orig_hash = sha_json(orig_blocks)
        mcu_hash = sha_json(mcu_blocks)
        orig_kv_hash = sha_json(orig_tokens)
        mcu_kv_hash = sha_json(mcu_tokens)

        frozen_alignment_ok = (
            str(frozen.get("sample_id", "")) == str(m_row.get("sample_id", frozen.get("sample_id", "")))
            and str(frozen.get("task", "")) == task
            and int(frozen.get("sample_index", sample_index)) == sample_index
        )

        row = {
            "task": task,
            "sample_index": sample_index,
            "sample_id": m_row.get("sample_id", frozen.get("sample_id", "")),
            "gate_layer": GATE_LAYER,
            "runtime_kv_layer": RUNTIME_KV_LAYER,
            "original_selected_block_ids": sorted(orig_sel),
            "mcu_selected_block_ids": sorted(mcu_sel),
            "original_used_blocks": sorted(orig_blocks),
            "mcu_used_blocks": sorted(mcu_blocks),
            "removed_block_ids": sorted(removed),
            "added_block_ids": sorted(added),
            "removed_still_present_in_used_blocks": bool(removed & mcu_block_set),
            "added_present_in_used_blocks": bool(added.issubset(mcu_block_set)) if added else True,
            "original_used_blocks_hash": orig_hash,
            "mcu_used_blocks_hash": mcu_hash,
            "original_kv_abs_indices_hash": orig_kv_hash,
            "mcu_kv_abs_indices_hash": mcu_kv_hash,
            "selection_changed_in_audit": bool(orig_sel != mcu_sel),
            "selection_changed_in_runtime": orig_hash != mcu_hash and orig_kv_hash != mcu_kv_hash,
            "budget_mismatch": bool(audit.get("mcu_trigger_reason") == "budget_mismatch" or audit.get("fallback_reason") == "budget_mismatch"),
            "fallback_used": bool(audit.get("fallback_used")),
            "frozen_alignment_ok": frozen_alignment_ok,
            "mcu_trigger_reason": audit.get("mcu_trigger_reason", ""),
        }
        row["decision"] = "PASS" if (
            row["selection_changed_in_audit"]
            and row["selection_changed_in_runtime"]
            and not row["removed_still_present_in_used_blocks"]
            and row["added_present_in_used_blocks"]
            and not row["budget_mismatch"]
            and not row["fallback_used"]
            and row["frozen_alignment_ok"]
        ) else "FAIL"
        rows.append(row)
    return rows


def build_report(output_dir: Path, rows: Sequence[Mapping[str, Any]], decision: str, reasons: Sequence[str]) -> str:
    lines = [
        "# MCU Runtime Wiring Debug",
        "",
        f"- Output dir: `{output_dir}`",
        f"- Gate layer: {GATE_LAYER}",
        f"- Runtime KV read layer: {RUNTIME_KV_LAYER} (pruning at gate L updates KV at L+1)",
        f"- Decision: **{decision}**",
        "",
        "## Summary",
        f"- Samples: {len(rows)}",
        f"- Audit selection changed: {sum(1 for r in rows if r.get('selection_changed_in_audit'))}",
        f"- Runtime used_blocks+KV changed: {sum(1 for r in rows if r.get('selection_changed_in_runtime'))}",
        f"- Removed still present: {sum(1 for r in rows if r.get('removed_still_present_in_used_blocks'))}",
        f"- Added missing: {sum(1 for r in rows if not r.get('added_present_in_used_blocks'))}",
        f"- Budget mismatch: {sum(1 for r in rows if r.get('budget_mismatch'))}",
        f"- Fallback used: {sum(1 for r in rows if r.get('fallback_used'))}",
        "",
        "## Reasons",
    ]
    for reason in reasons:
        lines.append(f"- {reason}")
    lines.extend(["", "## Per-sample", ""])
    for row in rows:
        lines.append(
            f"- {row['task']}:{row['sample_index']} | audit_changed={row['selection_changed_in_audit']} "
            f"| runtime_changed={row['selection_changed_in_runtime']} | removed_still={row['removed_still_present_in_used_blocks']} "
            f"| fallback={row['fallback_used']}"
        )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug MCU runtime wiring on fixed 12 samples.")
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--longbench_dir", type=Path, default=DEFAULT_LONGBENCH_DIR)
    parser.add_argument("--audit_dir", type=Path, default=DEFAULT_AUDIT_DIR)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--skip_run", action="store_true", help="Only analyze existing output_dir.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (args.output_dir or Path(f"results/mcu_runtime_wiring_fix_debug_{stamp}")).resolve()
    if not args.skip_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        audit_dir = (REPO_ROOT / args.audit_dir).resolve()
        frozen_csv = audit_dir / "mcu_counterfactual_audit.csv"
        frozen = load_frozen_selections_from_audit_csv(frozen_csv)
        frozen_path = output_dir / "mcu_frozen_selections.jsonl"
        write_frozen_selections_jsonl(frozen_path, frozen)

        sample_spec = build_sample_spec()
        orig_dir = output_dir / "run_sliminfer_original"
        mcu_dir = output_dir / "run_mcu_sliminfer"
        mcu_audit = mcu_dir / "audit" / "mcu_runtime_selection_audit.jsonl"
        mcu_audit.parent.mkdir(parents=True, exist_ok=True)
        mcu_audit.write_text("", encoding="utf-8")

        run_runner(
            runner_cmd(orig_dir, "sliminfer_original", sample_spec, None, None, args),
            output_dir / "original_runner.log",
        )
        run_runner(
            runner_cmd(mcu_dir, "mcu_sliminfer", sample_spec, frozen_path, mcu_audit, args),
            output_dir / "mcu_runner.log",
        )
    else:
        if not output_dir.exists():
            raise RuntimeError(f"output_dir does not exist: {output_dir}")

    frozen_rows = {
        f"{row['task']}:{int(row['sample_index'])}": row
        for row in read_jsonl(output_dir / "mcu_frozen_selections.jsonl")
    } if (output_dir / "mcu_frozen_selections.jsonl").exists() else {}

    orig_used = load_used_blocks_prefill(output_dir / "run_sliminfer_original" / "audit" / "used_blocks.jsonl", RUNTIME_KV_LAYER)
    mcu_used = load_used_blocks_prefill(output_dir / "run_mcu_sliminfer" / "audit" / "used_blocks.jsonl", RUNTIME_KV_LAYER)
    mcu_audit = load_mcu_audit_prefill(output_dir / "run_mcu_sliminfer" / "audit" / "mcu_runtime_selection_audit.jsonl", GATE_LAYER)

    rows = build_rows(orig_used, mcu_used, mcu_audit, frozen_rows)
    decision, reasons = decide(rows)

    fields = [
        "task", "sample_index", "sample_id", "gate_layer", "runtime_kv_layer",
        "selection_changed_in_audit", "selection_changed_in_runtime",
        "removed_still_present_in_used_blocks", "added_present_in_used_blocks",
        "budget_mismatch", "fallback_used", "frozen_alignment_ok",
        "original_used_blocks_hash", "mcu_used_blocks_hash",
        "original_kv_abs_indices_hash", "mcu_kv_abs_indices_hash",
        "mcu_trigger_reason", "decision",
        "original_selected_block_ids", "mcu_selected_block_ids",
        "original_used_blocks", "mcu_used_blocks",
        "removed_block_ids", "added_block_ids",
    ]
    write_csv(output_dir / "mcu_runtime_wiring_debug.csv", rows, fields)
    report = build_report(output_dir, rows, decision, reasons)
    (output_dir / "mcu_runtime_wiring_report.md").write_text(report, encoding="utf-8")
    (output_dir / "decision.md").write_text(
        f"# Decision\n\n**{decision}**\n\n" + "\n".join(f"- {r}" for r in reasons) + "\n",
        encoding="utf-8",
    )
    print(report)
    print(f"Wrote {output_dir / 'decision.md'}")


if __name__ == "__main__":
    main()
