#!/usr/bin/env python3
"""Full LongBench orchestrator for DRM-Add paper main table (resume-safe)."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
RUNNER = REPO_ROOT / "experiments" / "run_qwen25_sliminfer_variants.py"
ANALYZER = REPO_ROOT / "experiments" / "analyze_drm_add_full_longbench.py"

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_LONGBENCH = "/diskE/datasets/LongBench"
REUSE_ROOT = REPO_ROOT / "results" / "rescuekv_qwen25_full_longbench_20260520_170047"

TASKS = [
    "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique",
    "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum",
    "lsht", "passage_count", "passage_retrieval_en", "lcc", "repobench-p",
]

METHODS: List[Dict[str, Any]] = [
    {
        "key": "FullKV",
        "display": "FullKV",
        "reuse_key": "M0_fullkv",
        "runner_method": "fullkv",
        "scoring": "token_unit_qk",
        "decode_policy": "original",
        "rescue_policy": "none",
        "rescue_score": "delta_o",
        "seed": 0,
        "run_latency": True,
    },
    {
        "key": "sliminfer_original",
        "display": "SlimInfer",
        "reuse_key": "M1_sliminfer_original",
        "runner_method": "sliminfer_original",
        "scoring": "token_unit_qk",
        "decode_policy": "original",
        "rescue_policy": "none",
        "rescue_score": "delta_o",
        "seed": 0,
        "run_latency": True,
    },
    {
        "key": "sliminfer_larger",
        "display": "SlimInfer larger K+M",
        "runner_method": "sliminfer_original",
        "scoring": "token_unit_qk",
        "decode_policy": "original",
        "rescue_policy": "sliminfer_larger",
        "rescue_score": "delta_o",
        "seed": 0,
        "run_latency": True,
    },
    {
        "key": "random_add_seed0",
        "display": "Random-Add",
        "runner_method": "sliminfer_variant",
        "scoring": "delta_o",
        "decode_policy": "reuse_prefill",
        "rescue_policy": "add",
        "rescue_score": "random",
        "seed": 0,
        "run_latency": True,
    },
    {
        "key": "delta_o_add",
        "display": "DeltaO-Add",
        "runner_method": "sliminfer_variant",
        "scoring": "delta_o",
        "decode_policy": "reuse_prefill",
        "rescue_policy": "add",
        "rescue_score": "delta_o",
        "seed": 0,
        "run_latency": True,
    },
    {
        "key": "drm_add",
        "display": "DRM-Add",
        "runner_method": "sliminfer_variant",
        "scoring": "delta_o",
        "decode_policy": "reuse_prefill",
        "rescue_policy": "add",
        "rescue_score": "drm",
        "seed": 0,
        "run_latency": True,
    },
]

COMMON_ARGS = [
    "--model_path", DEFAULT_MODEL,
    "--tokenizer_path", DEFAULT_MODEL,
    "--model_family", "qwen2",
    "--dtype", "fp16",
    "--device", "cuda",
    "--backend", "flash_attention_2",
    "--block_size", "64",
    "--token_unit_size", "8",
    "--local_query_window", "4",
    "--swap_threshold", "0.9",
    "--prune_layers", "9,18,26",
    "--retain_tokens", "12288,6144,4096",
    "--preserve_first_block",
    "--temperature", "0",
    "--top_p", "1.0",
    "--enable_cpu_offload",
    "--enable_async_kv",
    "--profile_io",
    "--profile_memory",
    "--profile_score_overhead",
    "--save_layer_stats",
    "--save_block_scores", "aggregate_only",
    "--save_metrics_json",
    "--max_model_len", "32768",
    "--longbench_subset_name", "full",
    "--smoke_mode", "0",
]


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def load_task_counts(longbench_dir: Path) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for task in TASKS:
        for path in [
            longbench_dir / f"{task}.jsonl",
            longbench_dir / f"{task}.json",
            longbench_dir / task / "test.jsonl",
        ]:
            if not path.exists():
                continue
            if path.suffix == ".jsonl":
                counts[task] = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
            else:
                payload = json.loads(path.read_text(encoding="utf-8"))
                counts[task] = len(payload.get("data", payload)) if isinstance(payload, (dict, list)) else 0
            break
        if task not in counts:
            from datasets import load_dataset
            ds = load_dataset("THUDM/LongBench", task, split="test", trust_remote_code=True)
            counts[task] = len(ds)
    return counts


def read_score_status(path: Path) -> Tuple[str, str]:
    if not path.exists():
        return "", "missing"
    rows = list(csv.DictReader(path.open("r", encoding="utf-8")))
    if not rows:
        return "", "missing"
    return str(rows[0].get("num_samples", "")), rows[0].get("status", "unknown")


def task_complete(out_root: Path, method_key: str, task: str, expected: int) -> bool:
    score_path = out_root / "raw_metrics" / method_key / task / "longbench_scores.csv"
    meta_path = out_root / "raw_metrics" / method_key / task / "run_meta.json"
    n, status = read_score_status(score_path)
    meta = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    if status not in ("ok", "reused"):
        return False
    try:
        return int(float(n)) >= expected and expected > 0
    except Exception:
        return False


def copy_reused_task(
    out_root: Path,
    method_key: str,
    task: str,
    reuse_root: Path,
    reuse_key: str,
    expected: int,
) -> bool:
    src_dir = reuse_root / "raw_metrics" / reuse_key / task
    src_score = src_dir / "longbench_scores.csv"
    if not src_score.exists():
        return False
    n, status = read_score_status(src_score)
    if status != "ok":
        return False
    try:
        if int(float(n)) < expected:
            return False
    except Exception:
        return False
    dst_dir = out_root / "raw_metrics" / method_key / task
    dst_dir.mkdir(parents=True, exist_ok=True)
    for name in ["longbench_scores.csv", "config.json", "metrics.json"]:
        src = src_dir / name
        if src.exists():
            shutil.copy2(src, dst_dir / name)
    pred_src = src_dir / "longbench_predictions" / f"{task}.jsonl"
    if pred_src.exists():
        pred_dst = out_root / "raw_predictions" / method_key
        pred_dst.mkdir(parents=True, exist_ok=True)
        shutil.copy2(pred_src, pred_dst / f"{task}.jsonl")
    write_json(dst_dir / "run_meta.json", {
        "status": "reused",
        "source": "reused",
        "reuse_key": reuse_key,
        "source_path": str(src_dir),
        "num_samples": n,
        "expected": expected,
    })
    return True


def setup_reuse(out_root: Path, reuse_root: Path, counts: Dict[str, int]) -> Dict[str, Any]:
    manifest: Dict[str, Any] = {"reused_methods": {}, "skipped_reuse": {}}
    for spec in METHODS:
        reuse_key = spec.get("reuse_key")
        if not reuse_key or not reuse_root.exists():
            continue
        ok_tasks = 0
        for task in TASKS:
            if copy_reused_task(out_root, spec["key"], task, reuse_root, reuse_key, counts[task]):
                ok_tasks += 1
        if ok_tasks == len(TASKS):
            manifest["reused_methods"][spec["key"]] = {
                "source_path": str(reuse_root / "raw_metrics" / reuse_key),
                "sample_count": sum(counts.values()),
                "compatibility_note": (
                    "Same Qwen2.5-7B-Instruct, prune_layers=9,18,26, retain_tokens=12288,6144,4096, "
                    "full LongBench, status=ok. FullKV/rescue_policy none for SlimInfer original."
                ),
            }
            lat_src = reuse_root / "raw_metrics" / reuse_key / "latency_32k"
            if lat_src.exists():
                lat_dst = out_root / "raw_metrics" / spec["key"] / "latency_32k"
                lat_dst.mkdir(parents=True, exist_ok=True)
                for f in lat_src.iterdir():
                    if f.is_file():
                        shutil.copy2(f, lat_dst / f.name)
        else:
            manifest["skipped_reuse"][spec["key"]] = f"only {ok_tasks}/{len(TASKS)} tasks reusable"
    write_json(out_root / "reuse_manifest.json", manifest)
    return manifest


def build_runner_cmd(out_root: Path, spec: Dict[str, Any], task: str, longbench_dir: str) -> List[str]:
    out_dir = out_root / "raw_metrics" / spec["key"] / task
    audit_path = out_root / "audit" / f"{spec['key']}.jsonl"
    cmd = [
        str(PYTHON), str(RUNNER),
        *COMMON_ARGS,
        "--output_dir", str(out_dir),
        "--isolated_run_id", f"{spec['key']}__{task}",
        "--eval", "longbench",
        "--longbench_dir", longbench_dir,
        "--longbench_tasks", task,
        "--method", spec["runner_method"],
        "--scoring", spec["scoring"],
        "--decode_policy", spec["decode_policy"],
        "--rescue_policy", spec["rescue_policy"],
        "--rescue_score", spec["rescue_score"],
        "--rescue_ratio", "0.05",
        "--seed", str(spec["seed"]),
        "--alpha_diff", "0.5",
        "--lambda_red", "0.25",
        "--eta_rescue", "0.25",
        "--eta_active", "0.25",
        "--rescue_audit_path", str(audit_path),
        "--debug_used_block_audit_path", str(out_root / "used_blocks" / f"{spec['key']}.jsonl"),
    ]
    if spec["runner_method"] == "sliminfer_variant":
        cmd.extend(["--fd_alpha_o", "1.0", "--fd_beta_h", "0.0", "--fd_gamma_s", "0.0"])
    return cmd


def run_task(out_root: Path, spec: Dict[str, Any], task: str, longbench_dir: str, skip_existing: bool) -> Dict[str, Any]:
    expected = int(load_json(out_root / "task_sample_counts.json", {}).get(task, 0))
    if skip_existing and task_complete(out_root, spec["key"], task, expected):
        return {"status": "skipped", "task": task, "method": spec["key"]}
    out_dir = out_root / "raw_metrics" / spec["key"] / task
    log_dir = out_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{spec['key']}__{task}.log"
    cmd = build_runner_cmd(out_root, spec, task, longbench_dir)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    with log_path.open("w", encoding="utf-8") as logf:
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, stdout=logf, stderr=subprocess.STDOUT)
    n, status = read_score_status(out_dir / "longbench_scores.csv")
    write_json(out_dir / "run_meta.json", {
        "status": status or ("failed" if proc.returncode else "unknown"),
        "source": "run",
        "returncode": proc.returncode,
        "num_samples": n,
        "expected": expected,
    })
    pred_src = out_dir / "longbench_predictions" / f"{task}.jsonl"
    if pred_src.exists():
        pred_dst = out_root / "raw_predictions" / spec["key"]
        pred_dst.mkdir(parents=True, exist_ok=True)
        shutil.copy2(pred_src, pred_dst / f"{task}.jsonl")
    return {"status": status, "task": task, "method": spec["key"], "returncode": proc.returncode}


def run_latency(out_root: Path, spec: Dict[str, Any], skip_existing: bool) -> None:
    if not spec.get("run_latency"):
        return
    lat_dir = out_root / "raw_metrics" / spec["key"] / "latency_32k"
    lat_csv = lat_dir / "latency.csv"
    if skip_existing and lat_csv.exists():
        text = lat_csv.read_text(encoding="utf-8")
        if "32768," in text:
            return
    lat_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(PYTHON), str(RUNNER),
        *COMMON_ARGS,
        "--output_dir", str(lat_dir),
        "--isolated_run_id", f"{spec['key']}__latency_32k",
        "--eval", "latency",
        "--context_lengths", "32768",
        "--decode_tokens", "16",
        "--method", spec["runner_method"],
        "--scoring", spec["scoring"],
        "--decode_policy", spec["decode_policy"],
        "--rescue_policy", spec["rescue_policy"],
        "--rescue_score", spec["rescue_score"],
        "--rescue_ratio", "0.05",
        "--seed", str(spec["seed"]),
        "--alpha_diff", "0.5",
        "--lambda_red", "0.25",
        "--eta_rescue", "0.25",
        "--eta_active", "0.25",
    ]
    if spec["runner_method"] == "sliminfer_variant":
        cmd.extend(["--fd_alpha_o", "1.0", "--fd_beta_h", "0.0", "--fd_gamma_s", "0.0"])
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, check=False)


def write_shell_scripts(out_root: Path, longbench_dir: str) -> None:
    run_full = out_root / "run_full.sh"
    resume = out_root / "resume.sh"
    body = f"""#!/usr/bin/env bash
set -euo pipefail
cd "{REPO_ROOT}"
export PYTHONPATH="{REPO_ROOT}:${{PYTHONPATH:-}}"
export PYTORCH_CUDA_ALLOC_CONF="${{PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}}"
OUT_ROOT="{out_root}" LONGBENCH_DIR="{longbench_dir}" SKIP_EXISTING=0 \\
  {PYTHON} tools/run_drm_add_full_longbench.py --out_root "$OUT_ROOT" --longbench_dir "$LONGBENCH_DIR"
"""
    resume_body = f"""#!/usr/bin/env bash
set -euo pipefail
cd "{REPO_ROOT}"
export PYTHONPATH="{REPO_ROOT}:${{PYTHONPATH:-}}"
export PYTORCH_CUDA_ALLOC_CONF="${{PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}}"
OUT_ROOT="{out_root}" LONGBENCH_DIR="{longbench_dir}" SKIP_EXISTING=1 \\
  {PYTHON} tools/run_drm_add_full_longbench.py --out_root "$OUT_ROOT" --longbench_dir "$LONGBENCH_DIR" --skip_existing 1
"""
    run_full.write_text(body, encoding="utf-8")
    resume.write_text(resume_body, encoding="utf-8")
    run_full.chmod(0o755)
    resume.chmod(0o755)


def aggregate_audit(out_root: Path) -> None:
    rows = []
    for spec in METHODS:
        audit_path = out_root / "audit" / f"{spec['key']}.jsonl"
        if not audit_path.exists():
            continue
        audit_lines = [json.loads(l) for l in audit_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        if not audit_lines:
            continue
        rescued = [float(x.get("rescued_count", 0) or 0) for x in audit_lines]
        k_vals = [float(x.get("K", 0) or 0) for x in audit_lines if float(x.get("K", 0) or 0) > 0]
        rows.append({
            "method": spec["key"],
            "task": "aggregate",
            "sample_count": len({x.get("sample_id") for x in audit_lines}),
            "rescue_policy": spec["rescue_policy"],
            "rescue_score": spec["rescue_score"],
            "K_mean": sum(k_vals) / len(k_vals) if k_vals else 0,
            "M_mean": sum(rescued) / len(rescued) if rescued else 0,
            "rescued_count_mean": sum(rescued) / len(rescued) if rescued else 0,
            "duplicate_block_ids": any(bool(x.get("duplicate_block_ids")) for x in audit_lines),
            "protected_block_violation": any(bool(x.get("protected_block_violation")) for x in audit_lines),
            "fallback_used": any(bool(x.get("fallback_used")) for x in audit_lines),
            "D_base_std_mean": "",
            "G_diffusion_std_mean": "",
            "R_active_std_mean": "",
        })
    if rows:
        fields = list(rows[0].keys())
        with (out_root / "audit_summary.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_root", type=str, default="")
    parser.add_argument("--longbench_dir", type=str, default=DEFAULT_LONGBENCH)
    parser.add_argument("--reuse_root", type=str, default=str(REUSE_ROOT))
    parser.add_argument("--skip_existing", type=int, default=1)
    parser.add_argument("--methods", type=str, default="")
    parser.add_argument("--analyze_only", action="store_true")
    args = parser.parse_args()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.out_root) if args.out_root else REPO_ROOT / "results" / f"rescuekv_drm_add_full_longbench_{timestamp}"
    out_root.mkdir(parents=True, exist_ok=True)
    for sub in ["raw_metrics", "raw_predictions", "audit", "used_blocks", "logs"]:
        (out_root / sub).mkdir(parents=True, exist_ok=True)

    counts = load_json(out_root / "task_sample_counts.json", None)
    if counts is None:
        counts = load_task_counts(Path(args.longbench_dir))
        write_json(out_root / "task_sample_counts.json", counts)

    write_json(out_root / "config.json", {
        "timestamp": timestamp,
        "model_path": DEFAULT_MODEL,
        "tasks": TASKS,
        "total_samples": sum(counts.values()),
        "rescue_ratio": 0.05,
        "sample_mode": "full",
        "smoke_mode": False,
        "methods": [m["key"] for m in METHODS],
        "reuse_root": args.reuse_root,
    })

    if not args.analyze_only:
        setup_reuse(out_root, Path(args.reuse_root), counts)
        specs = METHODS
        if args.methods:
            wanted = {x.strip() for x in args.methods.split(",") if x.strip()}
            specs = [m for m in METHODS if m["key"] in wanted]
        for spec in specs:
            if spec.get("reuse_key") and spec["key"] in load_json(out_root / "reuse_manifest.json", {}).get("reused_methods", {}):
                print(f"Skip run {spec['key']} (fully reused)", flush=True)
                continue
            print(f"=== Method {spec['key']} ===", flush=True)
            for task in TASKS:
                result = run_task(out_root, spec, task, args.longbench_dir, bool(args.skip_existing))
                print(f"  {task}: {result.get('status')}", flush=True)
            run_latency(out_root, spec, bool(args.skip_existing))
        aggregate_audit(out_root)
        write_shell_scripts(out_root, args.longbench_dir)

    subprocess.run([str(PYTHON), str(ANALYZER), "--root", str(out_root)], check=False)
    print(out_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
