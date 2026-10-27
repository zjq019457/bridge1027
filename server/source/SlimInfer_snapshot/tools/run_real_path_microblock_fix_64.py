#!/usr/bin/env python3
"""64-sample real-path microblock validation (3 methods, 5 tasks)."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.postfix_32_real_path_audit import main as run_postfix_audit
from tools.summarize_structured_rescue import summarize

PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
RUNNER = REPO_ROOT / "experiments" / "run_qwen25_sliminfer_variants.py"
AUDIT_SCRIPT = REPO_ROOT / "tools" / "postfix_32_real_path_audit.py"
TASKS = ["qasper", "hotpotqa", "musique", "passage_count", "passage_retrieval_en"]
DEFAULT_MODEL = "/home/yu-bai/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
DEFAULT_LB = "/home/yu-bai/Desktop/jiaqizhang/HeadKV/data/LongBench"

METHOD_SPECS: List[Dict[str, Any]] = [
    {"method": "fixed_drm_add", "rescue_score": "drm"},
    {
        "method": "rh_pyramid_srm",
        "rescue_score": "rh_pyramid_srm",
        "rescue_granularity": "microblock",
        "microblock_size": 32,
        "rh_weight": 0.50,
        "pyramid_weight": 0.20,
        "structured": True,
    },
    {
        "method": "bridge_graph_srm",
        "rescue_score": "bridge_graph_srm",
        "rescue_granularity": "microblock",
        "microblock_size": 32,
        "bridge_weight": 0.45,
        "structured": True,
    },
]


def _cmd(out_root: Path, spec: Dict[str, Any], cfg: Dict[str, Any]) -> List[str]:
    method = spec["method"]
    cmd = [
        str(PYTHON), str(RUNNER),
        "--model_path", cfg["model_path"],
        "--tokenizer_path", cfg["model_path"],
        "--model_family", "qwen2", "--dtype", "fp16", "--device", "cuda",
        "--backend", "flash_attention_2",
        "--method", "sliminfer_variant",
        "--scoring", "delta_o",
        "--decode_policy", "reuse_prefill",
        "--eval", "longbench",
        "--longbench_dir", cfg["longbench_dir"],
        "--longbench_tasks", ",".join(cfg["tasks"]),
        "--max_samples_per_task", str(cfg["max_samples"]),
        "--generation_max_new_tokens", str(cfg["max_new_tokens"]),
        "--smoke_mode", "1",
        "--max_smoke_samples", str(cfg["max_samples"]),
        "--output_dir", str(out_root / "runs" / method),
        "--isolated_run_id", method,
        "--block_strategy", "fixed",
        "--block_size", "64",
        "--prune_layers", "9,18,26",
        "--retain_tokens", "12288,6144,4096",
        "--preserve_first_block",
        "--temperature", "0", "--top_p", "1.0",
        "--enable_cpu_offload", "--enable_async_kv",
        "--save_outputs",
        "--rescue_policy", "add",
        "--rescue_score", spec["rescue_score"],
        "--rescue_ratio", str(cfg["rescue_ratio"]),
        "--rescue_audit_path", str(out_root / "audit" / f"{method}.jsonl"),
        "--seed", "0",
        "--fd_alpha_o", "1.0", "--fd_beta_h", "0.0", "--fd_gamma_s", "0.0",
    ]
    for key in ("rescue_granularity", "microblock_size", "rh_weight", "pyramid_weight", "bridge_weight"):
        if key in spec:
            cmd.extend([f"--{key}", str(spec[key])])
    if spec.get("structured"):
        cmd.extend([
            "--dump_candidate_scores", "1",
            "--dump_final_selection", "1",
            "--strict_structured_path_check", "true",
        ])
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--model_path", default=DEFAULT_MODEL)
    parser.add_argument("--longbench_dir", default=DEFAULT_LB)
    parser.add_argument("--max_samples", type=int, default=64)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--rescue_ratio", type=float, default=0.05)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--methods", default="fixed_drm_add,rh_pyramid_srm,bridge_graph_srm")
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.output_dir) if args.output_dir else REPO_ROOT / "results" / f"real_path_microblock_fix_64_{stamp}"
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "audit").mkdir(exist_ok=True)
    (out_root / "runs").mkdir(exist_ok=True)

    wanted = {m.strip() for m in args.methods.split(",") if m.strip()}
    specs = [s for s in METHOD_SPECS if s["method"] in wanted]
    cfg = {
        "model_path": args.model_path,
        "longbench_dir": args.longbench_dir,
        "tasks": TASKS,
        "max_samples": args.max_samples,
        "max_new_tokens": args.max_new_tokens,
        "rescue_ratio": args.rescue_ratio,
    }
    manifest = {"out_root": str(out_root), "specs": specs, "cfg": cfg, "runs": []}
    for spec in specs:
        method = spec["method"]
        run_dir = out_root / "runs" / method
        status_path = run_dir / "run_status.json"
        if args.skip_existing and status_path.exists():
            print(f"skip existing {method}")
            continue
        t0 = time.time()
        print(f"===== RUN {method} =====", flush=True)
        proc = subprocess.run(_cmd(out_root, spec, cfg), cwd=str(REPO_ROOT))
        elapsed = time.time() - t0
        status = {"method": method, "exit_code": proc.returncode, "elapsed_s": round(elapsed, 1)}
        run_dir.mkdir(parents=True, exist_ok=True)
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
        manifest["runs"].append(status)
        print(json.dumps(status), flush=True)

    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    summarize(out_root, out_root / "summary")
    audit_dir = out_root / "postfix_64_real_path_audit"
    subprocess.run(
        [str(PYTHON), str(AUDIT_SCRIPT), "--input_root", str(out_root), "--output_dir", str(audit_dir)],
        cwd=str(REPO_ROOT),
        check=False,
    )
    print(f"OUT_ROOT={out_root}")


if __name__ == "__main__":
    main()
