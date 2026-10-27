#!/usr/bin/env python3
"""Small eval orchestrator for DRM-Add vs SlimInfer baselines."""
from __future__ import annotations

import argparse
import csv
import hashlib
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
DEFAULT_TASKS = ["qasper", "hotpotqa", "musique", "2wikimqa", "multifieldqa_en"]

METHOD_SPECS: List[Dict[str, Any]] = [
    {"method": "sliminfer_original", "label": "sliminfer_original", "rescue_policy": "none", "rescue_score": "delta_o", "seed": 0,
     "runner_method": "sliminfer_original", "scoring": "token_unit_qk", "decode_policy": "original"},
    {"method": "sliminfer_larger", "label": "sliminfer_larger", "rescue_policy": "sliminfer_larger", "rescue_score": "delta_o", "seed": 0,
     "runner_method": "sliminfer_original", "scoring": "token_unit_qk", "decode_policy": "original"},
]
for seed in range(5):
    METHOD_SPECS.append({
        "method": f"random_add_seed{seed}", "label": f"random_add_seed{seed}",
        "rescue_policy": "add", "rescue_score": "random", "seed": seed,
        "runner_method": "sliminfer_variant", "scoring": "delta_o", "decode_policy": "reuse_prefill",
    })
for score in ["delta_o", "diffusion_gap", "redundancy", "mmr", "drm"]:
    METHOD_SPECS.append({
        "method": f"{score}_add", "label": f"{score}_add",
        "rescue_policy": "add", "rescue_score": score, "seed": 0,
        "runner_method": "sliminfer_variant", "scoring": "delta_o", "decode_policy": "reuse_prefill",
    })


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


def _mean(values: Iterable[float]) -> Optional[float]:
    vals = [float(v) for v in values if v not in ("", None) and not (isinstance(v, float) and math.isnan(v))]
    return statistics.mean(vals) if vals else None


def _std(values: Iterable[float]) -> Optional[float]:
    vals = [float(v) for v in values if v not in ("", None)]
    return statistics.pstdev(vals) if len(vals) > 1 else (0.0 if vals else None)


def run_method(
    out_root: Path,
    spec: Dict[str, Any],
    *,
    model_path: str,
    tasks: List[str],
    max_samples: int,
    max_new_tokens: int,
    rescue_ratio: float,
    dry_run: bool,
) -> Dict[str, Any]:
    method_dir = out_root / "runs" / spec["method"]
    audit_path = out_root / "audit" / f"{spec['method']}.jsonl"
    gen_path = out_root / "generations" / f"{spec['method']}.jsonl"
    logits_path = out_root / "logits" / f"{spec['method']}.jsonl"
    used_block_path = out_root / "used_blocks" / f"{spec['method']}.jsonl"
    for p in [audit_path, gen_path, logits_path, used_block_path]:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("", encoding="utf-8")

    cmd = [
        str(PYTHON), str(RUNNER),
        "--model_path", model_path,
        "--method", spec["runner_method"],
        "--scoring", spec["scoring"],
        "--decode_policy", spec["decode_policy"],
        "--eval", "longbench",
        "--longbench_tasks", ",".join(tasks),
        "--max_samples_per_task", str(max_samples),
        "--generation_max_new_tokens", str(max_new_tokens),
        "--smoke_mode", "1",
        "--max_smoke_samples", str(max_samples),
        "--output_dir", str(method_dir),
        "--rescue_policy", spec["rescue_policy"],
        "--rescue_score", spec["rescue_score"],
        "--rescue_ratio", str(rescue_ratio),
        "--seed", str(spec["seed"]),
        "--alpha_diff", "0.5",
        "--lambda_red", "0.25",
        "--eta_rescue", "0.25",
        "--eta_active", "0.25",
        "--rescue_audit_path", str(audit_path),
        "--debug_used_block_audit_path", str(used_block_path),
        "--debug_generation_jsonl", str(gen_path),
        "--debug_first_token_logits_path", str(logits_path),
        "--save_outputs",
    ]
    status = "ok"
    error = ""
    if dry_run:
        return {"status": "dry_run", "cmd": cmd}
    try:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(REPO_ROOT)
        subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, check=True)
    except subprocess.CalledProcessError as exc:
        status = "failed"
        error = repr(exc)
    return {"status": status, "error": error, "cmd": cmd, "method_dir": str(method_dir)}


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def aggregate_audit(out_root: Path, spec: Dict[str, Any]) -> Dict[str, Any]:
    audit_rows = load_jsonl(out_root / "audit" / f"{spec['method']}.jsonl")
    used_rows = load_jsonl(out_root / "used_blocks" / f"{spec['method']}.jsonl")
    prefill_used = [r for r in used_rows if r.get("phase") == "prefill"]
    budgets_policy = [int(r.get("final_block_count_policy", r.get("final_block_count", 0)) or 0) for r in audit_rows if r.get("final_block_count_policy") or r.get("final_block_count")]
    budgets_used = [int(r.get("final_block_count_used_by_attention", 0) or 0) for r in prefill_used if r.get("final_block_count_used_by_attention")]
    if not budgets_used and audit_rows:
        budgets_used = [int(r.get("final_block_count", 0) or 0) for r in audit_rows]
    rescued = [int(r.get("rescued_count", 0) or 0) for r in audit_rows]
    score_std = [float((r.get("score_stats") or {}).get("std", 0.0)) for r in audit_rows]
    dup = any(bool(r.get("duplicate_block_ids")) for r in audit_rows)
    prot = any(bool(r.get("protected_block_violation")) for r in audit_rows)
    fallback = any(bool(r.get("fallback_used")) for r in audit_rows)
    reasons = sorted({str(r.get("fallback_reason", "")) for r in audit_rows if r.get("fallback_reason")})
    expected = "K+M" if spec["rescue_policy"] in {"add", "sliminfer_larger"} else "K"
    return {
        "method": spec["method"],
        "rescue_policy": spec["rescue_policy"],
        "rescue_score": spec["rescue_score"],
        "seed": spec["seed"],
        "expected_budget": expected,
        "observed_final_budget_mean": _mean(budgets_used or budgets_policy) or 0.0,
        "observed_final_budget_min": min(budgets_used or budgets_policy or [0]),
        "observed_final_budget_max": max(budgets_used or budgets_policy or [0]),
        "rescued_count_mean": _mean(rescued) or 0.0,
        "duplicate_block_ids": dup,
        "protected_block_violation": prot,
        "fallback_used": fallback,
        "fallback_reason": "; ".join(reasons),
        "score_std_mean": _mean(score_std) or 0.0,
        "jaccard_vs_sliminfer_mean": "",
        "audit_rows": len(audit_rows),
        "used_block_rows": len(used_rows),
    }


def aggregate_scores(out_root: Path, spec: Dict[str, Any]) -> List[Dict[str, Any]]:
    score_path = out_root / "runs" / spec["method"] / "longbench_scores.csv"
    if not score_path.exists():
        return []
    rows = []
    with score_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "method": spec["method"],
                "rescue_policy": spec["rescue_policy"],
                "rescue_score": spec["rescue_score"],
                "seed": spec["seed"],
                "task": row.get("dataset", ""),
                "sample_count": row.get("num_samples", ""),
                "score_or_metric": row.get("score", ""),
                "status": row.get("status", ""),
                "notes": row.get("error", ""),
            })
    return rows


def build_generation_diff(out_root: Path, baseline_method: str = "sliminfer_original") -> List[Dict[str, Any]]:
    baseline = {(r.get("task"), r.get("sample_id")): r for r in load_jsonl(out_root / "generations" / f"{baseline_method}.jsonl")}
    rows = []
    for gen_path in sorted((out_root / "generations").glob("*.jsonl")):
        method = gen_path.stem
        for rec in load_jsonl(gen_path):
            key = (rec.get("task"), rec.get("sample_id"))
            base = baseline.get(key, {})
            same = rec.get("output_hash") == base.get("output_hash") if base else ""
            text = str(rec.get("output_text", ""))[:120]
            rows.append({
                "task": rec.get("task", ""),
                "sample_id": rec.get("sample_id", ""),
                "method": method,
                "output_hash": rec.get("output_hash", ""),
                "output_same_as_sliminfer": same,
                "output_text_preview": text,
            })
    return rows


def build_logit_diff(out_root: Path, baseline_method: str = "sliminfer_original") -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    baseline = {(r.get("task"), r.get("sample_id")): r for r in load_jsonl(out_root / "logits" / f"{baseline_method}.jsonl")}
    detail_rows = []
    summary: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"count": 0, "chosen_same": 0, "top20_same_rates": [], "l2_diffs": []})
    for log_path in sorted((out_root / "logits").glob("*.jsonl")):
        method = log_path.stem
        for rec in load_jsonl(log_path):
            key = (rec.get("task"), rec.get("sample_id"))
            base = baseline.get(key, {})
            top20_same = ""
            l2_diff = ""
            chosen_same = ""
            if base and "top20_token_ids" in rec and "top20_token_ids" in base:
                a = set(rec.get("top20_token_ids", []))
                b = set(base.get("top20_token_ids", []))
                top20_same = len(a & b) / max(1, len(a | b))
                chosen_same = rec.get("chosen_token_id") == base.get("chosen_token_id")
                try:
                    import torch
                    la = torch.tensor(rec.get("top20_logits", []), dtype=torch.float32)
                    lb = torch.tensor(base.get("top20_logits", []), dtype=torch.float32)
                    if la.numel() == lb.numel() and la.numel() > 0:
                        l2_diff = float((la - lb).norm().item())
                except Exception:
                    pass
            detail_rows.append({
                "task": rec.get("task", ""),
                "sample_id": rec.get("sample_id", ""),
                "method": method,
                "top20_same_rate": top20_same,
                "chosen_token_same": chosen_same,
                "l2_diff_vs_sliminfer": l2_diff,
            })
            s = summary[method]
            s["count"] += 1
            if chosen_same is True:
                s["chosen_same"] += 1
            if top20_same != "":
                s["top20_same_rates"].append(float(top20_same))
            if l2_diff != "":
                s["l2_diffs"].append(float(l2_diff))
    summary_rows = []
    for method, s in sorted(summary.items()):
        summary_rows.append({
            "method": method,
            "num_samples": s["count"],
            "chosen_token_same_rate": (s["chosen_same"] / s["count"]) if s["count"] else "",
            "top20_same_rate_mean": _mean(s["top20_same_rates"]),
            "l2_diff_mean": _mean(s["l2_diffs"]),
        })
    return detail_rows, summary_rows


def build_block_diff(out_root: Path, baseline_method: str = "sliminfer_original") -> List[Dict[str, Any]]:
    baseline_rows = load_jsonl(out_root / "used_blocks" / f"{baseline_method}.jsonl")
    baseline_by_key: Dict[Tuple[Any, Any, Any], set] = defaultdict(set)
    for r in baseline_rows:
        if r.get("phase") != "prefill":
            continue
        key = (r.get("task"), r.get("sample_id"), r.get("layer_id"))
        baseline_by_key[key].update(r.get("used_block_ids_first100", []))
    rows = []
    for path in sorted((out_root / "used_blocks").glob("*.jsonl")):
        method = path.stem
        for r in load_jsonl(path):
            if r.get("phase") != "prefill":
                continue
            key = (r.get("task"), r.get("sample_id"), r.get("layer_id"))
            cur = set(r.get("used_block_ids_first100", []))
            base = baseline_by_key.get(key, set())
            inter = len(cur & base)
            union = len(cur | base) or 1
            rows.append({
                "method": method,
                "task": r.get("task", ""),
                "sample_id": r.get("sample_id", ""),
                "layer_id": r.get("layer_id", ""),
                "block_count": r.get("final_block_count_used_by_attention", 0),
                "jaccard_vs_sliminfer": inter / union,
            })
    return rows


def write_run_script(out_root: Path, config: Dict[str, Any]) -> None:
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", f'cd "{REPO_ROOT}"', ""]
    for spec in METHOD_SPECS:
        result = run_method(out_root, spec, dry_run=True, **{k: config[k] for k in ["model_path", "tasks", "max_samples", "max_new_tokens", "rescue_ratio"]})
        lines.append(f"# {spec['method']}")
        lines.append(" ".join(result["cmd"]))
        lines.append("")
    script = out_root / "run_drm_add_eval.sh"
    script.write_text("\n".join(lines) + "\n", encoding="utf-8")
    script.chmod(0o755)


def write_report(out_root: Path, summary_rows: List[Dict[str, Any]], audit_rows: List[Dict[str, Any]], config: Dict[str, Any], run_status: List[Dict[str, Any]]) -> None:
    def score_for(method: str, task: Optional[str] = None) -> Optional[float]:
        vals = []
        for r in summary_rows:
            if r.get("method") != method:
                continue
            if task and r.get("task") != task:
                continue
            try:
                vals.append(float(r.get("score_or_metric")))
            except Exception:
                pass
        return _mean(vals)

    random_scores = [_mean([float(r["score_or_metric"]) for r in summary_rows if r["method"] == f"random_add_seed{s}" and r.get("score_or_metric") not in ("", None)]) for s in range(5)]
    random_scores = [x for x in random_scores if x is not None]
    random_mean = _mean(random_scores)
    random_std = _std(random_scores)
    drm = score_for("drm_add")
    orig = score_for("sliminfer_original")
    larger = score_for("sliminfer_larger")

    failed = [r for r in run_status if r.get("status") == "failed"]
    lines = [
        "# DRM-Add Small Eval Report",
        "",
        "## 1. 本轮目标",
        "本轮只评估 **Additive Rescue (DRM-Add)**，不使用 swap/conditional_swap，不跑 full LongBench。",
        "",
        "## 2. 方法定义",
        "- A = SlimInfer top-K active blocks",
        "- I = inactive blocks",
        "- R = DRM-selected M rescued blocks from I",
        "- final = A ∪ R，预算 K+M",
        "",
        "DRM score:",
        "```",
        "S0(B) = z(D_base(B)) + alpha_diff * z(G(B)) - lambda_red * z(R_active(B))",
        "G(B) = 1 - max_{A_j in A} cos(rep(B), rep(A_j))",
        "```",
        "",
        "MMR:",
        "```",
        "S_MMR(B) = S0(B) - eta_rescue * max_{B' in R} cos(rep(B), rep(B')) - eta_active * max_{A_j in A} cos(rep(B), rep(A_j))",
        "```",
        "",
        "## 3. 路径验证",
    ]
    for a in audit_rows:
        lines.append(
            f"- **{a['method']}**: expected={a['expected_budget']}, observed_mean={a['observed_final_budget_mean']:.1f}, "
            f"dup={a['duplicate_block_ids']}, protected_violation={a['protected_block_violation']}, fallback={a['fallback_used']}"
        )
    lines.extend(["", "## 4. 小样本结果表", ""])
    tasks = sorted({r.get("task") for r in summary_rows if r.get("task")})
    methods = sorted({r.get("method") for r in summary_rows if r.get("method")})
    lines.append("| task | " + " | ".join(methods[:8]) + " |")
    lines.append("|" + "---|" * (min(8, len(methods)) + 1))
    for task in tasks:
        vals = []
        for method in methods[:8]:
            s = next((r for r in summary_rows if r.get("method") == method and r.get("task") == task), None)
            vals.append(str(s.get("score_or_metric", "")) if s else "")
        lines.append(f"| {task} | " + " | ".join(vals) + " |")

    lines.extend([
        "",
        "## 5. 关键比较 (small eval trend only)",
        f"- drm_add vs SlimInfer original: drm={drm}, orig={orig}, delta={(drm - orig) if drm is not None and orig is not None else 'n/a'}",
        f"- drm_add vs SlimInfer larger K+M: drm={drm}, larger={larger}, delta={(drm - larger) if drm is not None and larger is not None else 'n/a'}",
        f"- drm_add vs random_add mean: drm={drm}, random_mean={random_mean}, random_std={random_std}",
        f"- drm_add vs delta_o_add: drm={drm}, delta_o={score_for('delta_o_add')}",
        f"- drm_add vs diffusion_gap_add: drm={drm}, diffusion_gap={score_for('diffusion_gap_add')}",
        f"- drm_add vs redundancy_add: drm={drm}, redundancy={score_for('redundancy_add')}",
        f"- drm_add vs mmr_add: drm={drm}, mmr={score_for('mmr_add')}",
        "",
        "## 6. random_add 解释",
        f"- 5 seeds mean/std: {random_mean} / {random_std}",
    ])
    if random_mean is not None and drm is not None and random_mean >= drm:
        lines.append("- **rescue budget itself is strong; selector superiority is not yet fully established.**")

    lines.extend([
        "",
        "## 7. larger K+M 解释",
    ])
    if larger is not None and drm is not None:
        if abs(larger - drm) < 0.5:
            lines.append("- SlimInfer larger K+M 接近 drm_add，收益可能主要来自额外预算。")
        elif drm > larger:
            lines.append("- drm_add 高于 larger K+M，DRM selector 可能有额外价值。")
        else:
            lines.append("- drm_add 未明显高于 larger K+M。")

    lines.extend(["", "## 8. generation/logit", "- 详见 generation_diff.csv 与 logit_diff_summary.csv", ""])
    if failed:
        lines.extend(["", "## 失败记录", *[f"- {f['method']}: {f.get('error','')}" for f in failed]])
    lines.extend([
        "",
        "## 9. 下一步建议",
        "- 若 drm_add > random_add_mean 且 > larger K+M：扩到 50 samples/task。",
        "- 若仅高于 original 但不及 larger K+M：写 quality-latency trade-off。",
        "- 若 random_add 最高：继续调 MMR/diversity。",
        "- 若 logits 变但 output 不变：扩大样本并筛选分歧样本。",
    ])
    (out_root / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_root", type=str, default="")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--tasks", type=str, default=",".join(DEFAULT_TASKS))
    parser.add_argument("--max_samples", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--rescue_ratio", type=float, default=0.05)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--methods", type=str, default="", help="Comma-separated subset of method names")
    args = parser.parse_args()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.out_root) if args.out_root else REPO_ROOT / "results" / f"rescuekv_drm_add_eval_{timestamp}"
    out_root.mkdir(parents=True, exist_ok=True)
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    specs = METHOD_SPECS
    if args.methods:
        wanted = {x.strip() for x in args.methods.split(",") if x.strip()}
        specs = [s for s in METHOD_SPECS if s["method"] in wanted]

    config = {
        "timestamp": timestamp,
        "model_path": args.model_path,
        "tasks": tasks,
        "max_samples": args.max_samples,
        "max_new_tokens": args.max_new_tokens,
        "rescue_ratio": args.rescue_ratio,
        "alpha_diff": 0.5,
        "lambda_red": 0.25,
        "eta_rescue": 0.25,
        "eta_active": 0.25,
        "full_longbench": False,
        "methods": [s["method"] for s in specs],
    }
    _write_json(out_root / "config.json", config)
    write_run_script(out_root, config)

    run_status = []
    for spec in specs:
        print(f"Running {spec['method']} ...", flush=True)
        status = run_method(
            out_root, spec,
            model_path=args.model_path,
            tasks=tasks,
            max_samples=args.max_samples,
            max_new_tokens=args.max_new_tokens,
            rescue_ratio=args.rescue_ratio,
            dry_run=args.dry_run,
        )
        run_status.append({"method": spec["method"], **status})

    summary_rows: List[Dict[str, Any]] = []
    audit_summary: List[Dict[str, Any]] = []
    for spec in specs:
        summary_rows.extend(aggregate_scores(out_root, spec))
        audit_summary.append(aggregate_audit(out_root, spec))

    _write_csv(out_root / "summary.csv", summary_rows, [
        "method", "rescue_policy", "rescue_score", "seed", "task", "sample_count", "score_or_metric", "status", "notes",
    ])
    _write_csv(out_root / "audit_summary.csv", audit_summary, [
        "method", "rescue_policy", "rescue_score", "seed", "expected_budget", "observed_final_budget_mean",
        "observed_final_budget_min", "observed_final_budget_max", "rescued_count_mean", "duplicate_block_ids",
        "protected_block_violation", "fallback_used", "fallback_reason", "score_std_mean", "jaccard_vs_sliminfer_mean",
    ])
    gen_diff = build_generation_diff(out_root)
    _write_csv(out_root / "generation_diff.csv", gen_diff, [
        "task", "sample_id", "method", "output_hash", "output_same_as_sliminfer", "output_text_preview",
    ])
    _, logit_summary = build_logit_diff(out_root)
    _write_csv(out_root / "logit_diff_summary.csv", logit_summary, [
        "method", "num_samples", "chosen_token_same_rate", "top20_same_rate_mean", "l2_diff_mean",
    ])
    block_diff = build_block_diff(out_root)
    _write_csv(out_root / "block_diff_summary.csv", block_diff, [
        "method", "task", "sample_id", "layer_id", "block_count", "jaccard_vs_sliminfer",
    ])
    write_report(out_root, summary_rows, audit_summary, config, run_status)
    print(out_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
