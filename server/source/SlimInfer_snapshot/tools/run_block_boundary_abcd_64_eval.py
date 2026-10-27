#!/usr/bin/env python3
"""64-sample diagnostic eval for fixed vs A/B/C/D block construction."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
RUNNER = REPO_ROOT / "experiments" / "run_qwen25_sliminfer_variants.py"

DEFAULT_MODEL = "/home/yu-bai/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
DEFAULT_LONGBENCH = "/home/yu-bai/Desktop/jiaqizhang/HeadKV/data/LongBench"
DEFAULT_TASKS = ["hotpotqa", "2wikimqa", "musique", "qasper", "multifieldqa_en"]

METHOD_SPECS: List[Dict[str, Any]] = [
    {"method": "fixed_sliminfer_original", "block_strategy": "fixed", "rescue_policy": "none", "rescue_score": "delta_o",
     "runner_method": "sliminfer_original", "scoring": "token_unit_qk", "decode_policy": "original"},
    {"method": "fixed_sliminfer_larger", "block_strategy": "fixed", "rescue_policy": "sliminfer_larger", "rescue_score": "delta_o",
     "runner_method": "sliminfer_original", "scoring": "token_unit_qk", "decode_policy": "original"},
    {"method": "fixed_drm_add", "block_strategy": "fixed", "rescue_policy": "add", "rescue_score": "drm",
     "runner_method": "sliminfer_variant", "scoring": "delta_o", "decode_policy": "reuse_prefill"},
    {"method": "semantic_sliminfer", "block_strategy": "semantic", "rescue_policy": "none", "rescue_score": "delta_o",
     "runner_method": "sliminfer_original", "scoring": "token_unit_qk", "decode_policy": "original"},
    {"method": "semantic_drm_add", "block_strategy": "semantic", "rescue_policy": "add", "rescue_score": "drm",
     "runner_method": "sliminfer_variant", "scoring": "delta_o", "decode_policy": "reuse_prefill"},
    {"method": "attention_boundary_sliminfer", "block_strategy": "attention_boundary", "rescue_policy": "none", "rescue_score": "delta_o",
     "runner_method": "sliminfer_original", "scoring": "token_unit_qk", "decode_policy": "original"},
    {"method": "attention_boundary_drm_add", "block_strategy": "attention_boundary", "rescue_policy": "add", "rescue_score": "drm",
     "runner_method": "sliminfer_variant", "scoring": "delta_o", "decode_policy": "reuse_prefill"},
    {"method": "query_dynamic_sliminfer", "block_strategy": "query_dynamic", "rescue_policy": "none", "rescue_score": "delta_o",
     "runner_method": "sliminfer_original", "scoring": "token_unit_qk", "decode_policy": "original"},
    {"method": "query_dynamic_drm_add", "block_strategy": "query_dynamic", "rescue_policy": "add", "rescue_score": "drm",
     "runner_method": "sliminfer_variant", "scoring": "delta_o", "decode_policy": "reuse_prefill"},
    {"method": "hierarchical_sliminfer", "block_strategy": "hierarchical", "rescue_policy": "none", "rescue_score": "delta_o",
     "runner_method": "sliminfer_original", "scoring": "token_unit_qk", "decode_policy": "original"},
    {"method": "hierarchical_drm_add", "block_strategy": "hierarchical", "rescue_policy": "add", "rescue_score": "drm",
     "runner_method": "sliminfer_variant", "scoring": "delta_o", "decode_policy": "reuse_prefill"},
]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
    return rows


def _mean(values: Iterable[Any]) -> Optional[float]:
    vals = []
    for value in values:
        try:
            if value not in ("", None):
                vals.append(float(value))
        except Exception:
            pass
    return statistics.mean(vals) if vals else None


def _std(values: Iterable[Any]) -> Optional[float]:
    vals = []
    for value in values:
        try:
            if value not in ("", None):
                vals.append(float(value))
        except Exception:
            pass
    if not vals:
        return None
    return statistics.pstdev(vals) if len(vals) > 1 else 0.0


def _score_rows(method_dir: Path) -> List[Dict[str, Any]]:
    path = method_dir / "longbench_scores.csv"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _method_complete(out_root: Path, spec: Dict[str, Any], tasks: List[str], max_samples: int) -> bool:
    rows = _score_rows(out_root / "runs" / spec["method"])
    by_task = {r.get("dataset"): r for r in rows}
    for task in tasks:
        row = by_task.get(task)
        if not row or row.get("status") != "ok":
            return False
        try:
            if int(float(row.get("num_samples", 0))) < max_samples:
                return False
        except Exception:
            return False
    return True


def _runner_cmd(
    out_root: Path,
    spec: Dict[str, Any],
    *,
    model_path: str,
    longbench_dir: str,
    tasks: List[str],
    max_samples: int,
    max_new_tokens: int,
    rescue_ratio: float,
) -> List[str]:
    method = spec["method"]
    cmd = [
        str(PYTHON), str(RUNNER),
        "--model_path", model_path,
        "--tokenizer_path", model_path,
        "--model_family", "qwen2",
        "--dtype", "fp16",
        "--device", "cuda",
        "--backend", "flash_attention_2",
        "--method", spec["runner_method"],
        "--scoring", spec["scoring"],
        "--decode_policy", spec["decode_policy"],
        "--eval", "longbench",
        "--longbench_dir", longbench_dir,
        "--longbench_tasks", ",".join(tasks),
        "--max_samples_per_task", str(max_samples),
        "--generation_max_new_tokens", str(max_new_tokens),
        "--smoke_mode", "1",
        "--max_smoke_samples", str(max_samples),
        "--output_dir", str(out_root / "runs" / method),
        "--isolated_run_id", method,
        "--block_strategy", spec["block_strategy"],
        "--target_block_tokens", "64",
        "--min_block_tokens", "32",
        "--max_block_tokens", "128",
        "--block_overlap_tokens", "0",
        "--attention_boundary_layer", "9",
        "--boundary_percentile", "85",
        "--query_top_ratio", "0.2",
        "--query_neighbor_sentences", "1",
        "--query_high_max_block_tokens", "64",
        "--query_low_max_block_tokens", "128",
        "--coarse_block_tokens", "256",
        "--fine_block_tokens", "64",
        "--medium_block_tokens", "128",
        "--top_coarse_ratio", "0.3",
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
        "--save_layer_stats",
        "--save_block_scores", "aggregate_only",
        "--save_metrics_json",
        "--max_model_len", "32768",
        "--rescue_policy", spec["rescue_policy"],
        "--rescue_score", spec["rescue_score"],
        "--rescue_ratio", str(rescue_ratio),
        "--seed", "0",
        "--alpha_diff", "0.5",
        "--lambda_red", "0.25",
        "--eta_rescue", "0.25",
        "--eta_active", "0.25",
        "--rescue_audit_path", str(out_root / "audit" / f"{method}.jsonl"),
        "--block_layout_audit_path", str(out_root / "block_layouts" / f"{method}.jsonl"),
        "--debug_used_block_audit_path", str(out_root / "used_blocks" / f"{method}.jsonl"),
        "--debug_generation_jsonl", str(out_root / "generations" / f"{method}.jsonl"),
        "--debug_first_token_logits_path", str(out_root / "logits" / f"{method}.jsonl"),
        "--save_outputs",
    ]
    if spec["runner_method"] == "sliminfer_variant":
        cmd.extend(["--fd_alpha_o", "1.0", "--fd_beta_h", "0.0", "--fd_gamma_s", "0.0"])
    return cmd


def run_method(
    out_root: Path,
    spec: Dict[str, Any],
    config: Dict[str, Any],
    *,
    dry_run: bool,
    skip_existing: bool,
) -> Dict[str, Any]:
    method = spec["method"]
    if skip_existing and _method_complete(out_root, spec, config["tasks"], int(config["max_samples"])):
        return {"method": method, "status": "skipped_existing", "returncode": 0}
    for sub in ["audit", "block_layouts", "generations", "logits", "used_blocks", "logs"]:
        (out_root / sub).mkdir(parents=True, exist_ok=True)
    for p in [
        out_root / "audit" / f"{method}.jsonl",
        out_root / "block_layouts" / f"{method}.jsonl",
        out_root / "generations" / f"{method}.jsonl",
        out_root / "logits" / f"{method}.jsonl",
        out_root / "used_blocks" / f"{method}.jsonl",
    ]:
        if not (skip_existing and p.exists()):
            p.write_text("", encoding="utf-8")
    cmd = _runner_cmd(
        out_root,
        spec,
        model_path=config["model_path"],
        longbench_dir=config["longbench_dir"],
        tasks=config["tasks"],
        max_samples=int(config["max_samples"]),
        max_new_tokens=int(config["max_new_tokens"]),
        rescue_ratio=float(config["rescue_ratio"]),
    )
    if dry_run:
        return {"method": method, "status": "dry_run", "returncode": 0, "cmd": cmd}
    log_path = out_root / "logs" / f"{method}.log"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    with log_path.open("w", encoding="utf-8") as logf:
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, stdout=logf, stderr=subprocess.STDOUT)
    return {"method": method, "status": "ok" if proc.returncode == 0 else "failed", "returncode": proc.returncode, "cmd": cmd}


def _audit_by_task(out_root: Path, method: str) -> Dict[str, List[Dict[str, Any]]]:
    by_task: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in _load_jsonl(out_root / "audit" / f"{method}.jsonl"):
        if row.get("phase") == "decode":
            continue
        by_task[str(row.get("task", ""))].append(row)
    return by_task


def build_summary(out_root: Path, specs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for spec in specs:
        method = spec["method"]
        audit = _audit_by_task(out_root, method)
        for score_row in _score_rows(out_root / "runs" / method):
            task = score_row.get("dataset", "")
            task_audit = audit.get(task, [])
            rows.append({
                "method": method,
                "block_strategy": spec["block_strategy"],
                "rescue_policy": spec["rescue_policy"],
                "rescue_score": spec["rescue_score"],
                "task": task,
                "sample_count": score_row.get("num_samples", ""),
                "score": score_row.get("score", ""),
                "status": score_row.get("status", "missing"),
                "avg_selected_tokens": _mean(r.get("selected_token_count") for r in task_audit) or "",
                "avg_selected_blocks": _mean(r.get("final_block_count_policy") for r in task_audit) or "",
                "notes": score_row.get("error", ""),
            })
    return rows


def build_category_summary(summary_rows: List[Dict[str, Any]], specs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    single = {"qasper", "multifieldqa_en"}
    multi = {"hotpotqa", "2wikimqa", "musique"}
    rows = []
    for spec in specs:
        method = spec["method"]
        vals = [r for r in summary_rows if r.get("method") == method]
        rows.append({
            "Method": method,
            "BlockStrategy": spec["block_strategy"],
            "SingleDocQA": _mean(r.get("score") for r in vals if r.get("task") in single) or "",
            "MultiDocQA": _mean(r.get("score") for r in vals if r.get("task") in multi) or "",
            "Avg": _mean(r.get("score") for r in vals) or "",
        })
    return rows


def build_block_layout_summary(out_root: Path, specs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for spec in specs:
        method = spec["method"]
        layout_rows = _load_jsonl(out_root / "block_layouts" / f"{method}.jsonl")
        audit = _audit_by_task(out_root, method)
        tasks = sorted({r.get("task", "") for r in layout_rows} | set(audit.keys()))
        for task in tasks:
            lrows = [r for r in layout_rows if r.get("task") == task]
            lengths = []
            for r in lrows:
                lengths.extend(r.get("block_lengths", []) or [])
            arows = audit.get(task, [])
            rows.append({
                "method": method,
                "block_strategy": spec["block_strategy"],
                "task": task,
                "num_blocks_mean": _mean(r.get("num_blocks") for r in lrows) or "",
                "block_len_mean": _mean(lengths) or "",
                "block_len_std": _std(lengths) or "",
                "block_len_min": min(lengths) if lengths else "",
                "block_len_max": max(lengths) if lengths else "",
                "fallback_rate": _mean(1.0 if r.get("fallback_used") else 0.0 for r in lrows) or 0.0,
                "selected_token_count_mean": _mean(r.get("selected_token_count") for r in arows) or "",
                "selected_token_ratio_mean": _mean(r.get("selected_token_ratio") for r in arows) or "",
            })
    return rows


def build_audit_summary(out_root: Path, specs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for spec in specs:
        method = spec["method"]
        audit = [r for r in _load_jsonl(out_root / "audit" / f"{method}.jsonl") if r.get("phase") != "decode"]
        expected = "K+M" if spec["rescue_policy"] in {"add", "sliminfer_larger"} else "K"
        rows.append({
            "method": method,
            "block_strategy": spec["block_strategy"],
            "rescue_policy": spec["rescue_policy"],
            "rescue_score": spec["rescue_score"],
            "expected_budget": expected,
            "observed_final_block_count_mean": _mean(r.get("final_block_count_policy") for r in audit) or "",
            "selected_token_count_mean": _mean(r.get("selected_token_count") for r in audit) or "",
            "duplicate_block_ids": any(bool(r.get("duplicate_block_ids")) for r in audit),
            "protected_block_violation": any(bool(r.get("protected_block_violation")) for r in audit),
            "fallback_rate": _mean(1.0 if r.get("fallback_used") else 0.0 for r in audit) or 0.0,
            "D_base_std_mean": _mean((r.get("D_base_stats") or {}).get("std", r.get("D_base_std_mean")) for r in audit) or "",
            "G_std_mean": _mean((r.get("G_diffusion_stats") or {}).get("std", r.get("G_std_mean")) for r in audit) or "",
            "R_std_mean": _mean((r.get("R_active_stats") or {}).get("std", r.get("R_std_mean")) for r in audit) or "",
        })
    return rows


def _macro(summary_rows: List[Dict[str, Any]], method: str, tasks: Optional[set] = None) -> Optional[float]:
    vals = []
    for row in summary_rows:
        if row.get("method") != method:
            continue
        if tasks and row.get("task") not in tasks:
            continue
        try:
            vals.append(float(row.get("score")))
        except Exception:
            pass
    return _mean(vals)


def build_generation_diff(out_root: Path, specs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    fixed_base = {(r.get("task"), r.get("sample_id")): r for r in _load_jsonl(out_root / "generations" / "fixed_sliminfer_original.jsonl")}
    strategy_base: Dict[str, Dict[Tuple[Any, Any], Dict[str, Any]]] = {}
    for spec in specs:
        if spec["method"].endswith("_sliminfer") or spec["method"] == "fixed_sliminfer_original":
            strategy_base[spec["block_strategy"]] = {
                (r.get("task"), r.get("sample_id")): r
                for r in _load_jsonl(out_root / "generations" / f"{spec['method']}.jsonl")
            }
    rows = []
    for spec in specs:
        for rec in _load_jsonl(out_root / "generations" / f"{spec['method']}.jsonl"):
            key = (rec.get("task"), rec.get("sample_id"))
            fixed = fixed_base.get(key, {})
            same_strategy = strategy_base.get(spec["block_strategy"], {}).get(key, {})
            rows.append({
                "task": rec.get("task", ""),
                "sample_id": rec.get("sample_id", ""),
                "method": spec["method"],
                "block_strategy": spec["block_strategy"],
                "output_hash": rec.get("output_hash", ""),
                "metric_score": rec.get("metric_score", ""),
                "output_same_as_fixed_sliminfer": rec.get("output_hash") == fixed.get("output_hash") if fixed else "",
                "output_same_as_same_strategy_sliminfer": rec.get("output_hash") == same_strategy.get("output_hash") if same_strategy else "",
                "output_text_preview": str(rec.get("output_text", ""))[:160],
            })
    return rows


def write_scripts(out_root: Path, config: Dict[str, Any]) -> None:
    args = [
        f'--out_root "{out_root}"',
        f'--model_path "{config["model_path"]}"',
        f'--longbench_dir "{config["longbench_dir"]}"',
        f'--tasks "{",".join(config["tasks"])}"',
        f'--max_samples {config["max_samples"]}',
        f'--max_new_tokens {config["max_new_tokens"]}',
        f'--rescue_ratio {config["rescue_ratio"]}',
    ]
    base = f'{PYTHON} tools/run_block_boundary_abcd_64_eval.py ' + " ".join(args)
    run = "\n".join(["#!/usr/bin/env bash", "set -euo pipefail", f'cd "{REPO_ROOT}"', base, ""])
    resume = "\n".join(["#!/usr/bin/env bash", "set -euo pipefail", f'cd "{REPO_ROOT}"', base + " --skip_existing", ""])
    (out_root / "run_abcd_64.sh").write_text(run, encoding="utf-8")
    (out_root / "resume.sh").write_text(resume, encoding="utf-8")
    (out_root / "run_abcd_64.sh").chmod(0o755)
    (out_root / "resume.sh").chmod(0o755)


def write_report(
    out_root: Path,
    summary_rows: List[Dict[str, Any]],
    block_rows: List[Dict[str, Any]],
    audit_rows: List[Dict[str, Any]],
    run_status: List[Dict[str, Any]],
    config: Dict[str, Any],
) -> None:
    def fmt(v: Any) -> str:
        try:
            return f"{float(v):.2f}"
        except Exception:
            return "n/a" if v in ("", None) else str(v)

    tasks = config["tasks"]
    methods = [s["method"] for s in METHOD_SPECS if s["method"] in {r["method"] for r in summary_rows}]
    avg_map = {m: _macro(summary_rows, m) for m in methods}
    fixed_orig = avg_map.get("fixed_sliminfer_original")
    fixed_drm = avg_map.get("fixed_drm_add")
    best = max((m for m in methods if avg_map.get(m) is not None), key=lambda m: avg_map.get(m) or -1e9, default="")
    multi = {"hotpotqa", "2wikimqa", "musique"}
    multi_best = max((m for m in methods if _macro(summary_rows, m, multi) is not None), key=lambda m: _macro(summary_rows, m, multi) or -1e9, default="")
    failed = [r for r in run_status if r.get("status") == "failed"]

    lines = [
        f"# Block Boundary ABCD {config['max_samples']}-Sample Diagnostic Report",
        "",
        "## 1. 本轮目标",
        f"这是 **block construction diagnostic**，每个任务最多 {config['max_samples']} samples；不是 full LongBench，也不覆盖已有 full-sample 结果。",
        "",
        "## 2. 已有 full main table 背景",
        "- SlimInfer original Avg = 47.59",
        "- SlimInfer larger K+M Avg = 47.70",
        "- DRM-Add Avg = 47.80",
        "",
        f"本轮只在同一 {config['max_samples']}-sample/task 子集上比较 fixed / semantic / attention boundary / query dynamic / hierarchical block construction。",
        "",
        "## 3. 方法解释",
        "- fixed blocks: 原始 64-token fixed KV pages。",
        "- semantic blocks: prompt 文本先按 document/paragraph/sentence 边界切分，短单元合并，过长单元退化为 fixed chunks。",
        "- attention boundary blocks: 在 cache path 中用 fixed-page pooled K/V representations 的相邻 cosine change 近似 boundary；未额外跑 full attention profiling。",
        "- query dynamic blocks: 用 question/query 的 lexical + BM25-style relevance 标记高相关语义单元，高相关区域更细，低相关区域更粗。",
        "- hierarchical blocks: 先按 coarse pages 估计 SlimInfer relevance，top coarse regions 细分为 fine blocks，其他区域保留 medium blocks。",
        "",
        "每个非 fixed strategy 都有 SlimInfer-only 与 DRM-Add 两版；DRM-Add 仍是 additive rescue：final = A_K ∪ R_M，不做 swap，不删除 A_K。",
        "",
        "## 4. 实现细节",
        "- selection unit 是 BlockLayout virtual block；现有 attention/cache kernel 仍使用 fixed 64-token KV pages。",
        "- variable block 被选中后会展开为覆盖它的 fixed pages，实际 attention 使用去重后的 fixed page ids。",
        "- 因此 audit 同时记录 final_block_count_policy 与 final_block_count_used_by_attention，以及 selected_token_count / selected_token_ratio。",
        "- attention_boundary 与 hierarchical 的动态 layout 在 pruning 时由 page-level reps/scores 构造；semantic/query_dynamic layout 在 tokenized prompt 上预构造。",
        "",
        "## 5. 结果表",
        "",
    ]
    lines.append("| Method | " + " | ".join(tasks) + " | Avg | Status |")
    lines.append("|---|" + "---|" * (len(tasks) + 2))
    for method in methods:
        vals = [fmt(_macro(summary_rows, method, {task})) for task in tasks]
        statuses = {r.get("status") for r in summary_rows if r.get("method") == method}
        lines.append(f"| {method} | " + " | ".join(vals) + f" | {fmt(avg_map.get(method))} | {','.join(sorted(statuses))} |")

    lines.extend(["", "## 6. Block layout 分析", ""])
    lines.append("| Method | Strategy | blocks | block_len | selected_tokens | fallback_rate |")
    lines.append("|---|---|---:|---:|---:|---:|")
    by_method = defaultdict(list)
    for row in block_rows:
        by_method[row["method"]].append(row)
    for method in methods:
        rows = by_method.get(method, [])
        spec = next((s for s in METHOD_SPECS if s["method"] == method), {})
        lines.append(
            f"| {method} | {spec.get('block_strategy', '')} | {fmt(_mean(r.get('num_blocks_mean') for r in rows))} | "
            f"{fmt(_mean(r.get('block_len_mean') for r in rows))} | {fmt(_mean(r.get('selected_token_count_mean') for r in rows))} | "
            f"{fmt(_mean(r.get('fallback_rate') for r in rows))} |"
        )

    comparisons = [
        ("semantic_sliminfer", "fixed_sliminfer_original"),
        ("semantic_drm_add", "fixed_drm_add"),
        ("attention_boundary_sliminfer", "fixed_sliminfer_original"),
        ("attention_boundary_drm_add", "fixed_drm_add"),
        ("query_dynamic_sliminfer", "fixed_sliminfer_original"),
        ("query_dynamic_drm_add", "fixed_drm_add"),
        ("hierarchical_sliminfer", "fixed_sliminfer_original"),
        ("hierarchical_drm_add", "fixed_drm_add"),
    ]
    lines.extend(["", "## 7. 关键比较"])
    for left, right in comparisons:
        lv = avg_map.get(left)
        rv = avg_map.get(right)
        delta = (lv - rv) if lv is not None and rv is not None else None
        lines.append(f"- {left} vs {right}: {fmt(lv)} vs {fmt(rv)}, delta={fmt(delta)}")
    lines.append(f"- 当前最有效 block strategy/method: {best} (Avg={fmt(avg_map.get(best))})")
    lines.append(f"- multi-hop tasks 提升最大当前方法: {multi_best} (MultiDocQA={fmt(_macro(summary_rows, multi_best, multi))})")
    drm_pairs = [
        ("semantic_drm_add", "semantic_sliminfer"),
        ("attention_boundary_drm_add", "attention_boundary_sliminfer"),
        ("query_dynamic_drm_add", "query_dynamic_sliminfer"),
        ("hierarchical_drm_add", "hierarchical_sliminfer"),
    ]
    drm_gains = []
    for drm, slim in drm_pairs:
        dv, sv = avg_map.get(drm), avg_map.get(slim)
        if dv is not None and sv is not None:
            drm_gains.append(dv - sv)
    lines.append(f"- DRM-Add 在新 block strategy 上平均增益: {fmt(_mean(drm_gains))}")
    lines.append(f"- best strategy 的 DRM-Add 是否超过 fixed_drm_add: {'是' if best and avg_map.get(best) is not None and fixed_drm is not None and (avg_map.get(best) or -1e9) > fixed_drm else '否/未观察到'}")

    lines.extend([
        "",
        "## 8. 公平性说明",
        "- variable-length blocks 可能导致 selected_token_count 不同，不能只看 block count。",
        "- 如果某策略选中更多 token，提升可能来自更多 token 而不是 boundary 更好。",
        "- 本轮未做 latency full breakdown；报告保留 token budget / block budget audit 以支持后续 matched-token 分析。",
        "",
        "## 9. 失败与限制",
    ])
    if failed:
        lines.extend([f"- {r['method']}: failed, returncode={r.get('returncode')}" for r in failed])
    else:
        lines.append("- 无 runner 级失败。")
    fallback_methods = [r for r in audit_rows if float(r.get("fallback_rate") or 0.0) > 0.0]
    for row in fallback_methods:
        lines.append(f"- {row['method']}: fallback_rate={fmt(row.get('fallback_rate'))}")
    lines.extend([
        "- attention_boundary 使用 page-level pooled representations，而不是 token-level hidden/attention profile。",
        "- variable block 通过 fixed KV pages 映射兼容现有 kernel，边界处可能带入额外 page tokens。",
        "",
        "## 10. 下一步建议",
    ])
    if best and best != "fixed_sliminfer_original" and avg_map.get(best) is not None and fixed_orig is not None and (avg_map.get(best) or 0) > fixed_orig:
        lines.append(f"- 有策略在 {config['max_samples']}-sample diagnostic 中优于 fixed original，可扩大到 full LongBench 前先补 matched-token larger baseline 与 latency。")
    else:
        lines.append("- 当前没有稳定证据表明 block construction 优于 fixed original；fixed block + DRM-Add 仍应保留为主线。")
    lines.append("- 对有效策略做 case study：展示 fixed block 切断 evidence，而 semantic/hierarchical 保留完整 evidence。")
    lines.append("")
    lines.append("## Resume")
    lines.append("```bash")
    lines.append(f"bash {out_root / 'resume.sh'}")
    lines.append("```")
    (out_root / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_root", type=str, default="")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--longbench_dir", type=str, default=DEFAULT_LONGBENCH)
    parser.add_argument("--tasks", type=str, default=",".join(DEFAULT_TASKS))
    parser.add_argument("--max_samples", type=int, default=64)
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--rescue_ratio", type=float, default=0.05)
    parser.add_argument("--methods", type=str, default="")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.out_root) if args.out_root else REPO_ROOT / "results" / f"rescuekv_block_boundary_abcd_64_{timestamp}"
    out_root.mkdir(parents=True, exist_ok=True)
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    specs = METHOD_SPECS
    if args.methods:
        wanted = {m.strip() for m in args.methods.split(",") if m.strip()}
        specs = [s for s in specs if s["method"] in wanted]

    config = {
        "timestamp": timestamp,
        "model_path": args.model_path,
        "longbench_dir": args.longbench_dir,
        "tasks": tasks,
        "max_samples": args.max_samples,
        "max_new_tokens": args.max_new_tokens,
        "rescue_ratio": args.rescue_ratio,
        "sample_mode": f"{args.max_samples}-sample diagnostic",
        "full_longbench": False,
        "methods": [s["method"] for s in specs],
    }
    _write_json(out_root / "config.json", config)
    write_scripts(out_root, config)

    run_status = []
    for spec in specs:
        print(f"Running {spec['method']} ({spec['block_strategy']}) ...", flush=True)
        run_status.append(run_method(out_root, spec, config, dry_run=args.dry_run, skip_existing=args.skip_existing))

    summary_rows = build_summary(out_root, specs)
    category_rows = build_category_summary(summary_rows, specs)
    block_rows = build_block_layout_summary(out_root, specs)
    audit_rows = build_audit_summary(out_root, specs)
    generation_rows = build_generation_diff(out_root, specs)

    _write_csv(out_root / "summary.csv", summary_rows, [
        "method", "block_strategy", "rescue_policy", "rescue_score", "task", "sample_count", "score", "status",
        "avg_selected_tokens", "avg_selected_blocks", "notes",
    ])
    _write_csv(out_root / "category_summary.csv", category_rows, ["Method", "BlockStrategy", "SingleDocQA", "MultiDocQA", "Avg"])
    _write_csv(out_root / "block_layout_summary.csv", block_rows, [
        "method", "block_strategy", "task", "num_blocks_mean", "block_len_mean", "block_len_std",
        "block_len_min", "block_len_max", "fallback_rate", "selected_token_count_mean", "selected_token_ratio_mean",
    ])
    _write_csv(out_root / "audit_summary.csv", audit_rows, [
        "method", "block_strategy", "rescue_policy", "rescue_score", "expected_budget",
        "observed_final_block_count_mean", "selected_token_count_mean", "duplicate_block_ids",
        "protected_block_violation", "fallback_rate", "D_base_std_mean", "G_std_mean", "R_std_mean",
    ])
    _write_csv(out_root / "generation_diff.csv", generation_rows, [
        "task", "sample_id", "method", "block_strategy", "output_hash", "metric_score",
        "output_same_as_fixed_sliminfer", "output_same_as_same_strategy_sliminfer", "output_text_preview",
    ])
    write_report(out_root, summary_rows, block_rows, audit_rows, run_status, config)
    print(out_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
