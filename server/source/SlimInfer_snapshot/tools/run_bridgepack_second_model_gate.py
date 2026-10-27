#!/usr/bin/env python3
"""Prepare/run an optional BridgePack second-model 32-sample gate.

This runner intentionally does not default to Qwen. It runs only when
--model_name_or_path is passed or MISTRAL_MODEL / LLAMA_MODEL is set.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_TASKS = "hotpotqa,2wikimqa,musique"
DEFAULT_VARIANTS = "span_only_source_order,bridge_neighbor32_path_order"


def _parse_list(value: str) -> List[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _run(cmd: Sequence[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    printable = " ".join(str(part) for part in cmd)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n$ {printable}\n")
        log.flush()
        proc = subprocess.run(
            list(cmd),
            cwd=REPO_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        log.write(f"returncode={proc.returncode}\n")
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed rc={proc.returncode}: {printable}. See {log_path}")


def _model_from_args(value: str) -> str:
    for candidate in [value, os.environ.get("MISTRAL_MODEL", ""), os.environ.get("LLAMA_MODEL", "")]:
        if candidate:
            return str(candidate)
    return ""


def _slug(model_name_or_path: str) -> str:
    name = Path(model_name_or_path).name or model_name_or_path
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "second_model"


def parse_args() -> argparse.Namespace:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="")
    parser.add_argument("--tasks", default=DEFAULT_TASKS)
    parser.add_argument("--max_samples", type=int, default=32)
    parser.add_argument("--budget", type=int, default=1024)
    parser.add_argument("--variants", default=DEFAULT_VARIANTS)
    parser.add_argument("--out_dir", default=f"results/bridgepack_second_model_gate_second_model_{stamp}")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = _model_from_args(args.model_name_or_path)
    if not model:
        print("No second-model path provided; set MISTRAL_MODEL or LLAMA_MODEL, or pass --model_name_or_path. Nothing run.")
        return
    if args.out_dir.endswith("second_model_" + args.out_dir.rsplit("_", 1)[-1]):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = REPO_ROOT / "results" / f"bridgepack_second_model_gate_{_slug(model)}_{stamp}"
    else:
        out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = _parse_list(args.tasks)
    variants = _parse_list(args.variants)
    packed_dir = out_dir / "packed_prompts_build"
    config: Dict[str, Any] = {
        "model_name_or_path": model,
        "tasks": tasks,
        "max_samples": args.max_samples,
        "budget": args.budget,
        "variants": variants,
        "context_compression_not_kv_pruning": True,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    offline_cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "run_evidence_text_packing_offline.py"),
        "--tasks",
        ",".join(tasks),
        "--max_samples",
        str(args.max_samples),
        "--budgets",
        str(args.budget),
        "--variants",
        ",".join(variants),
        "--out_dir",
        str(packed_dir),
    ]
    if not args.resume or not packed_dir.exists():
        _run(offline_cmd, out_dir / "logs" / "offline_packing.log")

    checker_cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "check_evidence_text_packing_generation_inputs.py"),
        "--packed_dir",
        str(packed_dir),
        "--tasks",
        ",".join(tasks),
        "--budget",
        str(args.budget),
        "--variants",
        ",".join(variants),
        "--max_samples",
        str(args.max_samples),
    ]
    _run(checker_cmd, out_dir / "logs" / "input_checker.log")

    generation_cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "run_evidence_text_packing_generation_qwen32.py"),
        "--packed_dir",
        str(packed_dir),
        "--tasks",
        ",".join(tasks),
        "--max_samples",
        str(args.max_samples),
        "--budget",
        str(args.budget),
        "--variants",
        ",".join(variants),
        "--model_name_or_path",
        model,
        "--out_dir",
        str(out_dir),
    ]
    if args.resume:
        generation_cmd.append("--resume")
    _run(generation_cmd, out_dir / "logs" / "generation.log")
    print(f"OUT_DIR={out_dir}")


if __name__ == "__main__":
    main()
