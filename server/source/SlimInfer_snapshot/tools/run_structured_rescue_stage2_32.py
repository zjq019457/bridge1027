#!/usr/bin/env python3
"""32-sample structured rescue diagnostic (5 tasks, 5 methods)."""
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

from tools.build_span_teacher_microblock_labels import build_labels
from tools.summarize_structured_rescue import summarize
from tools.train_distilled_blockdrm_selector import train_selector

PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
RUNNER = REPO_ROOT / "experiments" / "run_qwen25_sliminfer_variants.py"
TASKS = ["qasper", "hotpotqa", "musique", "passage_count", "passage_retrieval_en"]
DEFAULT_MODEL = "/home/yu-bai/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
DEFAULT_LB = "/home/yu-bai/Desktop/jiaqizhang/HeadKV/data/LongBench"
RH_PROFILE = REPO_ROOT / "profiles" / "retrieval_head_profiles" / "head_profile.json"

METHOD_SPECS: List[Dict[str, Any]] = [
    {"method": "fixed_drm_add", "runner_method": "sliminfer_variant", "scoring": "delta_o",
     "decode_policy": "reuse_prefill", "rescue_policy": "add", "rescue_score": "drm"},
    {"method": "rh_pyramid_srm", "runner_method": "sliminfer_variant", "scoring": "delta_o",
     "decode_policy": "reuse_prefill", "rescue_policy": "add", "rescue_score": "rh_pyramid_srm",
     "rescue_granularity": "microblock", "microblock_size": 32},
    {"method": "distilled_blockdrm", "runner_method": "sliminfer_variant", "scoring": "delta_o",
     "decode_policy": "reuse_prefill", "rescue_policy": "add", "rescue_score": "distilled_blockdrm",
     "rescue_granularity": "microblock", "microblock_size": 32, "needs_teacher": True},
    {"method": "bridge_graph_srm", "runner_method": "sliminfer_variant", "scoring": "delta_o",
     "decode_policy": "reuse_prefill", "rescue_policy": "add", "rescue_score": "bridge_graph_srm",
     "rescue_granularity": "microblock", "microblock_size": 32, "bridge_weight": 0.45},
    {"method": "deferred_uncertainty_srm", "runner_method": "sliminfer_variant", "scoring": "delta_o",
     "decode_policy": "reuse_prefill", "rescue_policy": "add", "rescue_score": "deferred_uncertainty_srm",
     "rescue_granularity": "microblock", "microblock_size": 32, "deferred_mode": "audit"},
]


def _cmd(out_root: Path, spec: Dict[str, Any], cfg: Dict[str, Any]) -> List[str]:
    method = spec["method"]
    cmd = [
        str(PYTHON), str(RUNNER),
        "--model_path", cfg["model_path"],
        "--tokenizer_path", cfg["model_path"],
        "--model_family", "qwen2", "--dtype", "fp16", "--device", "cuda",
        "--backend", "flash_attention_2",
        "--method", spec["runner_method"],
        "--scoring", spec["scoring"],
        "--decode_policy", spec["decode_policy"],
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
        "--rescue_policy", spec["rescue_policy"],
        "--rescue_score", spec["rescue_score"],
        "--rescue_ratio", str(cfg["rescue_ratio"]),
        "--rescue_audit_path", str(out_root / "audit" / f"{method}.jsonl"),
        "--seed", "0",
    ]
    if spec["runner_method"] == "sliminfer_variant":
        cmd.extend(["--fd_alpha_o", "1.0", "--fd_beta_h", "0.0", "--fd_gamma_s", "0.0"])
    for key in ("rescue_granularity", "microblock_size", "deferred_mode", "bridge_weight", "rh_weight", "pyramid_weight"):
        if key in spec:
            cmd.extend([f"--{key}", str(spec[key])])
    if spec.get("rescue_score") != "drm":
        cmd.extend([
            "--dump_candidate_scores", "1",
            "--dump_final_selection", "1",
            "--strict_structured_path_check", "true",
        ])
    if spec["rescue_score"] == "rh_pyramid_srm" and cfg.get("rh_head_file"):
        cmd.extend(["--rh_head_file", cfg["rh_head_file"]])
    if spec.get("needs_teacher"):
        cmd.extend([
            "--teacher_label_file", cfg["teacher_label_file"],
            "--distilled_selector_file", cfg["distilled_selector_file"],
        ])
    if spec["rescue_score"] == "bridge_graph_srm":
        cmd.extend(["--bridge_weight", "0.30"])
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--model_path", default=DEFAULT_MODEL)
    parser.add_argument("--longbench_dir", default=DEFAULT_LB)
    parser.add_argument("--max_samples", type=int, default=32)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--rescue_ratio", type=float, default=0.05)
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.output_dir) if args.output_dir else REPO_ROOT / "results" / f"structured_rescue_stage2_32_{stamp}"
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "audit").mkdir(exist_ok=True)
    (out_root / "runs").mkdir(exist_ok=True)

    teacher_labels = out_root / "teacher_microblock_labels.jsonl"
    selector_file = out_root / "distilled_blockdrm_selector.json"
    build_labels(None, teacher_labels, microblock_size=32, weak_fallback=True)
    train_selector(teacher_labels, Path(""), selector_file)

    cfg = {
        "model_path": args.model_path,
        "longbench_dir": args.longbench_dir,
        "tasks": TASKS,
        "max_samples": args.max_samples,
        "max_new_tokens": args.max_new_tokens,
        "rescue_ratio": args.rescue_ratio,
        "rh_head_file": str(RH_PROFILE) if RH_PROFILE.exists() else "",
        "teacher_label_file": str(teacher_labels),
        "distilled_selector_file": str(selector_file),
    }
    (out_root / "config.json").write_text(json.dumps({"cfg": cfg, "methods": [s["method"] for s in METHOD_SPECS]}, indent=2), encoding="utf-8")

    log_path = out_root / "run.log"
    status = []
    with log_path.open("a", encoding="utf-8") as log:
        for spec in METHOD_SPECS:
            method = spec["method"]
            score_csv = out_root / "runs" / method / "longbench_scores.csv"
            if args.skip_existing and score_csv.exists():
                log.write(f"[skip] {method} already has scores\n")
                status.append({"method": method, "status": "skipped"})
                continue
            cmd = _cmd(out_root, spec, cfg)
            log.write(f"[start] {method}: {' '.join(cmd)}\n")
            log.flush()
            print(f"[structured-rescue] running {method} ...", flush=True)
            t0 = time.time()
            proc = subprocess.run(cmd, cwd=str(REPO_ROOT), stdout=log, stderr=subprocess.STDOUT)
            elapsed = time.time() - t0
            st = "ok" if proc.returncode == 0 else f"fail_{proc.returncode}"
            log.write(f"[done] {method} status={st} elapsed={elapsed:.1f}s\n")
            status.append({"method": method, "status": st, "elapsed_s": round(elapsed, 1)})

    summarize(out_root, out_root / "summary")
    (out_root / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(f"DONE -> {out_root}", flush=True)


if __name__ == "__main__":
    main()
