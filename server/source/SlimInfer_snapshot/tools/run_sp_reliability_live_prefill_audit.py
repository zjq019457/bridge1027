#!/usr/bin/env python3
"""Phase 2 live prefill audit: SP-SlimInfer + rank_stability_sliminfer runtime wiring.

Runs GPU prefill-only gate_audit on 400 two-task samples for:
  - sliminfer_original (contamination check vs offline baseline)
  - sp_sliminfer
  - rank_stability_sliminfer

Does NOT run QA. Emits active-set / fairness / official-path audits.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bridgekv.sp_sliminfer import active_set_churn
from tools.run_bridgeflow_silm_property_probe import read_jsonl

CHURN_MIN, CHURN_MAX = 0.03, 0.15
PRUNE_LAYERS = [9, 18, 26]
BLOCK_SIZE = 64
MAX_MODEL_LEN = 32768
TASKS = ("hotpotqa", "musique")
SAMPLES_PER_TASK = 200

DEFAULT_MODEL = (
    "/home/yu-bai/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/"
    "snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
)
DEFAULT_LONGBENCH = Path("/home/yu-bai/Desktop/jiaqizhang/HeadKV/data/LongBench")
DEFAULT_BASELINE_UASR = (
    REPO_ROOT / "results/bridgeflow_silm_property_probe_2task_full_20260709_211600/uasr_score_dump.jsonl"
)
OFFLINE_SP_AUDIT = REPO_ROOT / "results/sp_sliminfer_probe_20260710"
OFFLINE_RSS_AUDIT = REPO_ROOT / "results/reliability_aware_sliminfer_20260710"

METHODS = (
    ("sliminfer_original", "sliminfer_original"),
    ("sp_sliminfer", "sp_sliminfer"),
    ("rank_stability_sliminfer", "rank_stability_sliminfer"),
)


def parse_args() -> argparse.Namespace:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out_dir", type=Path, default=REPO_ROOT / f"results/sp_reliability_live_prefill_audit_{ts}")
    p.add_argument("--model_path", type=Path, default=Path(DEFAULT_MODEL))
    p.add_argument("--longbench_dir", type=Path, default=DEFAULT_LONGBENCH)
    p.add_argument("--baseline_uasr", type=Path, default=DEFAULT_BASELINE_UASR)
    p.add_argument("--offline_sp_audit", type=Path, default=OFFLINE_SP_AUDIT)
    p.add_argument("--offline_rss_audit", type=Path, default=OFFLINE_RSS_AUDIT)
    p.add_argument("--samples_per_task", type=int, default=SAMPLES_PER_TASK)
    p.add_argument("--smoke_samples_per_task", type=int, default=0, help="If >0, run tiny smoke instead of full 400.")
    p.add_argument("--skip_gpu_runs", action="store_true", help="Only analyze existing run dirs under out_dir.")
    p.add_argument("--analyze_only", action="store_true", help="Alias for --skip_gpu_runs.")
    p.add_argument("--methods", default="", help="Comma-separated subset: sliminfer_original,sp_sliminfer,rank_stability_sliminfer")
    p.add_argument("--resume", action="store_true", help="Skip GPU run when gate_audit_manifest already has expected samples.")
    return p.parse_args()


def selected_methods(args: argparse.Namespace) -> Tuple[Tuple[str, str], ...]:
    if not str(args.methods or "").strip():
        return METHODS
    wanted = {x.strip() for x in str(args.methods).split(",") if x.strip()}
    picked = tuple((label, method) for label, method in METHODS if label in wanted or method in wanted)
    if not picked:
        raise ValueError(f"No methods matched --methods={args.methods!r}")
    return picked


def gate_audit_complete(run_dir: Path, expected_samples: int) -> bool:
    manifest = run_dir / "gate_audit_manifest.jsonl"
    if not manifest.exists():
        return False
    keys: Set[Tuple[str, int]] = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        keys.add((str(row.get("task", "")), int(row.get("sample_index", -1))))
    return len(keys) >= expected_samples


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def python_bin() -> Path:
    candidate = REPO_ROOT / ".venv" / "bin" / "python"
    return candidate if candidate.exists() else Path(sys.executable)


def build_sample_spec(samples_per_task: int) -> str:
    indices = ",".join(str(i) for i in range(samples_per_task))
    return ";".join(f"{task}:{indices}" for task in TASKS)


def gate_audit_cmd(
    *,
    method: str,
    run_dir: Path,
    sample_spec: str,
    model_path: Path,
    longbench_dir: Path,
) -> List[str]:
    score_dump = run_dir / "uasr_score_dump.jsonl"
    used_blocks = run_dir / "used_blocks.jsonl"
    dc_sink = run_dir / "dc_selection_audit_sink.jsonl"
    for path in (score_dump, used_blocks, dc_sink):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
    return [
        str(python_bin()),
        str(REPO_ROOT / "experiments" / "run_qwen25_sliminfer_variants.py"),
        "--model_path", str(model_path),
        "--tokenizer_path", str(model_path),
        "--model_family", "qwen2",
        "--dtype", "fp16",
        "--device", "cuda",
        "--backend", "flash_attention_2",
        "--longbench_dir", str(longbench_dir),
        "--longbench_tasks", ",".join(TASKS),
        "--longbench_sample_indices_by_task", sample_spec,
        "--longbench_subset_name", "sp_reliability_live_prefill_audit",
        "--max_model_len", str(MAX_MODEL_LEN),
        "--method", method,
        "--scoring", "token_unit",
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
        "--uasr_score_dump_path", str(score_dump),
        "--uasr_tail_width", "8",
        "--dc_selection_audit_path", str(dc_sink),
        "--save_block_scores", "sampled",
        "--debug_used_block_audit_path", str(used_blocks),
        "--debug_used_block_include_full", "true",
        "--debug_used_block_full_layers", "27",
    ]


def run_command(cmd: Sequence[str], log_prefix: Path) -> None:
    log_prefix.parent.mkdir(parents=True, exist_ok=True)
    stdout_path = Path(f"{log_prefix}_stdout.log")
    stderr_path = Path(f"{log_prefix}_stderr.log")
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        proc = subprocess.run(list(cmd), cwd=str(REPO_ROOT), text=True, stdout=stdout, stderr=stderr)
    out_text = stdout_path.read_text(encoding="utf-8", errors="ignore")
    err_text = stderr_path.read_text(encoding="utf-8", errors="ignore")
    if "CUDA out of memory" in out_text or "CUDA out of memory" in err_text:
        raise RuntimeError(f"CUDA OOM: {' '.join(cmd)}")
    if proc.returncode != 0:
        raise RuntimeError(f"runner failed rc={proc.returncode}; see {stderr_path}")


def load_prefill_rows(path: Path) -> List[Dict[str, Any]]:
    rows = read_jsonl(path)
    return [r for r in rows if str(r.get("phase", "")) == "prefill" and str(r.get("task", "")) in TASKS]


def index_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[Tuple[str, str, int], Mapping[str, Any]]:
    out: Dict[Tuple[str, str, int], Mapping[str, Any]] = {}
    for row in rows:
        task = str(row.get("task", ""))
        sid = str(row.get("sample_id", row.get("sample_index", "")))
        layer = int(row.get("layer_id", -1))
        if task and sid and layer >= 0:
            out[(task, sid, layer)] = row
    return out


def must_keep(block_ids: Sequence[int]) -> Set[int]:
    s: Set[int] = set()
    if block_ids:
        s.add(int(block_ids[0]))
        s.add(int(block_ids[-1]))
    return s


def audit_method_vs_baseline(
    *,
    label: str,
    runtime_rows: Sequence[Mapping[str, Any]],
    baseline_rows: Sequence[Mapping[str, Any]],
    expected_samples: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rt_index = index_rows(runtime_rows)
    base_index = index_rows(baseline_rows)
    detail: List[Dict[str, Any]] = []
    churns: List[float] = []
    changed_layers = 0
    changed_samples: Set[Tuple[str, str]] = set()
    eligible = 0
    over_budget = 0
    sink_bad = 0
    no_effect = 0
    samples_seen: Set[Tuple[str, str]] = set()

    keys = sorted(set(rt_index.keys()) & set(base_index.keys()), key=lambda k: (k[0], k[1], k[2]))
    for task, sid, layer in keys:
        rt = rt_index[(task, sid, layer)]
        base = base_index[(task, sid, layer)]
        block_ids = [int(x) for x in rt.get("all_candidate_block_ids", [])]
        budget = int(rt.get("budget_k", 0) or len(rt.get("selected_block_ids", [])))
        silm_selected = {int(x) for x in base.get("selected_block_ids", [])}
        runtime_selected = {int(x) for x in rt.get("selected_block_ids", [])}
        churn_eligible = int(budget < len(block_ids))
        if churn_eligible:
            eligible += 1
        churn = active_set_churn(silm_selected, runtime_selected, budget)
        if churn_eligible:
            churns.append(churn)
        if runtime_selected == silm_selected:
            no_effect += 1
        else:
            changed_layers += 1
            changed_samples.add((task, sid))
        if len(runtime_selected) > budget:
            over_budget += 1
        if block_ids and int(block_ids[0]) not in runtime_selected:
            sink_bad += 1
        overlap = len(runtime_selected & silm_selected)
        union = len(runtime_selected | silm_selected)
        samples_seen.add((task, sid))
        detail.append({
            "method": label,
            "task": task,
            "sample_id": sid,
            "layer_id": layer,
            "budget_k": budget,
            "eligible_churn": churn_eligible,
            "churn_rate": round(churn, 6),
            "jaccard_vs_baseline": round(overlap / max(1, union), 6),
            "changed": int(runtime_selected != silm_selected),
            "over_budget": int(len(runtime_selected) > budget),
            "sink_preserved": int(not block_ids or int(block_ids[0]) in runtime_selected),
            "runtime_selected_count": len(runtime_selected),
            "baseline_selected_count": len(silm_selected),
        })

    mean_churn = statistics.mean(churns) if churns else 0.0
    summary = {
        "method": label,
        "unique_samples": len(samples_seen),
        "expected_samples": expected_samples,
        "eligible_layers": eligible,
        "mean_churn_eligible": round(mean_churn, 6),
        "churn_in_band": int(CHURN_MIN <= mean_churn <= CHURN_MAX) if label != "sliminfer_original" else int(mean_churn == 0.0),
        "changed_layer_count": changed_layers,
        "changed_sample_count": len(changed_samples),
        "no_effect_rows": no_effect,
        "over_budget": over_budget,
        "sink_violations": sink_bad,
        "contamination_vs_baseline": round(mean_churn, 6),
        "status": "PASS" if label == "sliminfer_original" and mean_churn == 0.0 and over_budget == 0 and sink_bad == 0 else "",
    }
    if label == "sliminfer_original":
        if mean_churn > 0.001 or changed_layers > 0:
            summary["status"] = "CONTAMINATED"
        else:
            summary["status"] = "PASS"
    elif over_budget > 0 or sink_bad > 0:
        summary["status"] = "BLOCK_QA"
    elif mean_churn == 0.0:
        summary["status"] = "NO_EFFECT_IN_ACTIVE_SET"
    elif not (CHURN_MIN <= mean_churn <= CHURN_MAX):
        summary["status"] = "CHURN_OUT_OF_BAND"
    else:
        summary["status"] = "PASS"
    return detail, summary


def load_offline_churn(offline_dir: Path, variant: str) -> Optional[float]:
    summary_path = offline_dir / "audit_summary.json"
    if not summary_path.exists():
        return None
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if variant == "sp_sliminfer":
        return float(payload.get("active_set", {}).get("mean_churn_eligible", 0.0))
    if variant == "rank_stability_sliminfer":
        if isinstance(payload, list):
            for row in payload:
                if row.get("variant") == variant:
                    return float(row.get("mean_churn_eligible", 0.0))
    return None


def render_reports(
    out_dir: Path,
    summaries: Sequence[Mapping[str, Any]],
    detail_rows: Sequence[Mapping[str, Any]],
    *,
    expected_samples: int,
    offline_churn: Mapping[str, Optional[float]],
    qa_ready: bool,
) -> None:
    write_csv(out_dir / "live_active_set_detail.csv", detail_rows, [
        "method", "task", "sample_id", "layer_id", "budget_k", "eligible_churn", "churn_rate",
        "jaccard_vs_baseline", "changed", "over_budget", "sink_preserved",
        "runtime_selected_count", "baseline_selected_count",
    ])
    write_csv(out_dir / "live_active_set_summary.csv", summaries, [
        "method", "unique_samples", "expected_samples", "eligible_layers", "mean_churn_eligible",
        "churn_in_band", "changed_layer_count", "changed_sample_count", "no_effect_rows",
        "over_budget", "sink_violations", "contamination_vs_baseline", "status",
        "offline_mean_churn", "offline_delta",
    ])

    lines = [
        "# Live Prefill Active-Set Audit",
        "",
        f"Samples expected: **{expected_samples}**",
        "",
        "| Method | Samples | Mean churn | In band | Δ layers | Over budget | Sink OK | Status |",
        "|--------|---------|------------|---------|----------|-------------|---------|--------|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['method']} | {row['unique_samples']} | {row['mean_churn_eligible']} | "
            f"{row['churn_in_band']} | {row['changed_layer_count']} | {row['over_budget']} | "
            f"{row['sink_violations']} | {row['status']} |"
        )
    lines.extend([
        "",
        "## Offline vs live churn",
    ])
    for method in ("sp_sliminfer", "rank_stability_sliminfer"):
        live = next((float(r["mean_churn_eligible"]) for r in summaries if r["method"] == method), None)
        off = offline_churn.get(method)
        if live is not None and off is not None:
            lines.append(f"- **{method}**: offline={off:.4f}, live={live:.4f}, delta={abs(live-off):.4f}")
    lines.extend([
        "",
        f"**QA ready (live audit):** {'YES' if qa_ready else 'NO — fix hooks before QA'}",
        "",
        "## Claim tiers",
        "- Official SILM repro (400 two-task): separate (57.43 / 27.14)",
        "- RoleFlow: **no-go**",
        "- SP-SlimInfer: Phase 2 live prefill audit",
        "- Reliability rank_stability: Phase 2 live prefill audit",
    ])
    (out_dir / "ACTIVE_SET_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    (out_dir / "OFFICIAL_PATH_AUDIT.md").write_text(
        "# Official Path Audit — Live Prefill\n\n"
        "- Runner: `experiments/run_qwen25_sliminfer_variants.py --eval gate_audit`\n"
        "- Methods: sliminfer_original, sp_sliminfer, rank_stability_sliminfer\n"
        "- Scoring: token_unit_qk via represent_keys (same as SILM)\n"
        "- Budget: retain_tokens 12288/6144/4096 at layers 9/18/26\n"
        "- No BM25/LBP/packing; no gold in selection\n"
        "- Prefill-only (no decode / no scorer)\n",
        encoding="utf-8",
    )
    (out_dir / "FAIRNESS_AUDIT.md").write_text(
        "# Fairness Audit\n\n"
        "- Same model, tokenizer, prompt template, block_size=64, prune layers, KV budget\n"
        "- Only `variant_method` selection policy differs for SP / rank_stability\n"
        "- sliminfer_original rerun verifies new hooks do not alter baseline path\n",
        encoding="utf-8",
    )
    (out_dir / "FEATURE_COVERAGE_AUDIT.md").write_text(
        "# Feature Coverage Audit (Runtime)\n\n"
        f"- Expected samples: {expected_samples}\n"
        "- SP runtime features: RUNTIME_PROXY from score/rank traces (`_sp_layer_trace`)\n"
        "- hidden_state_trajectory: **UNAVAILABLE**\n"
        "- key_state_representation_drift: **UNAVAILABLE**\n"
        "- rank_stability: causal ranks from prior prune layers in same prefill pass\n",
        encoding="utf-8",
    )
    (out_dir / "CLAIM_GUARD.md").write_text(
        "# Claim Guard\n\n"
        "| Tier | Note |\n"
        "|------|------|\n"
        "| Official SILM 400-full | 57.43 / 27.14 |\n"
        "| RoleFlow | **no-go** |\n"
        "| Phase 2 live prefill | audit only, no QA yet |\n"
        f"| QA gate | {'OPEN' if qa_ready else 'BLOCKED'} |\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if args.analyze_only:
        args.skip_gpu_runs = True
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    methods = selected_methods(args)

    n_per_task = int(args.smoke_samples_per_task) if args.smoke_samples_per_task > 0 else int(args.samples_per_task)
    expected_samples = n_per_task * len(TASKS)
    sample_spec = build_sample_spec(n_per_task)
    write_json(out_dir / "audit_config.json", {
        "sample_spec": sample_spec,
        "expected_samples": expected_samples,
        "tasks": list(TASKS),
        "baseline_uasr": str(args.baseline_uasr),
        "methods": [label for label, _ in methods],
    })

    run_dirs: Dict[str, Path] = {}
    failures: List[str] = []
    if not args.skip_gpu_runs:
        for label, method in methods:
            run_dir = out_dir / f"run_{label}"
            run_dirs[label] = run_dir
            if args.resume and gate_audit_complete(run_dir, expected_samples):
                print(f"[live-audit] skip {label}: manifest already has {expected_samples} samples", flush=True)
                continue
            cmd = gate_audit_cmd(
                method=method,
                run_dir=run_dir,
                sample_spec=sample_spec,
                model_path=args.model_path,
                longbench_dir=args.longbench_dir,
            )
            write_json(out_dir / f"command_{label}.json", {"command": cmd})
            print(f"[live-audit] running {label} ...", flush=True)
            try:
                run_command(cmd, out_dir / f"log_{label}")
            except Exception as exc:
                failures.append(f"{label}: {exc}")
                print(f"[live-audit] FAILED {label}: {exc}", flush=True)
    else:
        for label, _ in methods:
            run_dirs[label] = out_dir / f"run_{label}"

    baseline_rows = load_prefill_rows(args.baseline_uasr)
    all_detail: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []

    for label, _ in methods:
        dump_path = run_dirs[label] / "uasr_score_dump.jsonl"
        if not dump_path.exists():
            failures.append(f"{label}: missing {dump_path}")
            continue
        runtime_rows = load_prefill_rows(dump_path)
        detail, summary = audit_method_vs_baseline(
            label=label,
            runtime_rows=runtime_rows,
            baseline_rows=baseline_rows,
            expected_samples=expected_samples,
        )
        off_key = label
        offline = load_offline_churn(
            args.offline_sp_audit if label == "sp_sliminfer" else args.offline_rss_audit,
            off_key,
        )
        if offline is not None and label != "sliminfer_original":
            summary["offline_mean_churn"] = round(offline, 6)
            summary["offline_delta"] = round(abs(float(summary["mean_churn_eligible"]) - offline), 6)
        else:
            summary["offline_mean_churn"] = ""
            summary["offline_delta"] = ""
        all_detail.extend(detail)
        summaries.append(summary)

    offline_churn = {
        "sp_sliminfer": load_offline_churn(args.offline_sp_audit, "sp_sliminfer"),
        "rank_stability_sliminfer": load_offline_churn(args.offline_rss_audit, "rank_stability_sliminfer"),
    }

    orig = next((s for s in summaries if s["method"] == "sliminfer_original"), None)
    sp = next((s for s in summaries if s["method"] == "sp_sliminfer"), None)
    rss = next((s for s in summaries if s["method"] == "rank_stability_sliminfer"), None)

    qa_ready = (
        not failures
        and orig is not None
        and sp is not None
        and rss is not None
        and orig["status"] == "PASS"
        and orig["unique_samples"] == expected_samples
        and sp["status"] == "PASS"
        and rss["status"] == "PASS"
        and sp["over_budget"] == 0
        and rss["over_budget"] == 0
        and sp["sink_violations"] == 0
        and rss["sink_violations"] == 0
    )

    render_reports(out_dir, summaries, all_detail, expected_samples=expected_samples, offline_churn=offline_churn, qa_ready=qa_ready)
    write_json(out_dir / "audit_summary.json", {"summaries": summaries, "qa_ready": qa_ready, "failures": failures})

    print(json.dumps({"out_dir": str(out_dir), "qa_ready": qa_ready, "failures": failures, "summaries": summaries}, indent=2))
    if not qa_ready and len(methods) == len(METHODS) and not args.skip_gpu_runs:
        sys.exit(2)
    if not qa_ready and args.analyze_only:
        sys.exit(2)


if __name__ == "__main__":
    main()
