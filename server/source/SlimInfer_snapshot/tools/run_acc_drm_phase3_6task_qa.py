#!/usr/bin/env python3
"""Phase-3 official 6-task QA for acc_drm_v1 (budget-neutral only).

Preflight: require Phase-2 GO. Freeze configs. Optionally reuse historical
sliminfer_original / cc_sliminfer_v1 only under strict protocol match.
Runs official experiments/run_qwen25_sliminfer_variants.py longbench path.
Does NOT start 17-task. Does NOT tune parameters from QA scores.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.consolidate_official_repro_results import load_predictions, per_sample_score, write_csv

TASKS = (
    "qasper",
    "multifieldqa_en",
    "hotpotqa",
    "2wikimqa",
    "musique",
    "triviaqa",
)
EXPECTED_BY_TASK = {
    "qasper": 200,
    "multifieldqa_en": 150,
    "hotpotqa": 200,
    "2wikimqa": 200,
    "musique": 200,
    "triviaqa": 200,
}
METHODS = (
    "sliminfer_original",
    "cc_sliminfer_v1",
    "acc_sliminfer_v1",
    "drm_acc_swap_sliminfer_v1",
)
DEFAULT_MODEL = (
    "/home/yu-bai/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/"
    "snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
)
DEFAULT_LONGBENCH = Path("/home/yu-bai/Desktop/jiaqizhang/HeadKV/data/LongBench")
SEED_6TASK = REPO_ROOT / "results/cc_sliminfer_v1_qwen25_6task_20260712_043551"
ACC_V1 = REPO_ROOT / "configs/acc_drm_v1.json"

# Official SlimInfer protocol (frozen).
PROTOCOL = {
    "prune_layers": [9, 18, 26],
    "retain_tokens": [12288, 6144, 4096],
    "block_size": 64,
    "token_unit_size": 8,
    "gamma_swap_threshold": 0.9,
    "local_query_window": 4,
    "batch_size": 1,
    "decode_policy": "original",
    "rescue_policy": "none",
    "seed": 0,
    "dtype": "fp16",
    "backend": "flash_attention_2",
    "generation_max_new_tokens": 0,  # dataset default maxlen
    "score_kind": "token_unit_qk",
}


def parse_args() -> argparse.Namespace:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out_dir",
        type=Path,
        default=REPO_ROOT / f"results/acc_drm_phase3_6task_{ts}",
    )
    p.add_argument("--phase2_dir", type=Path, default=None)
    p.add_argument("--seed_6task_dir", type=Path, default=SEED_6TASK)
    p.add_argument("--model_path", default=DEFAULT_MODEL)
    p.add_argument("--longbench_dir", default=str(DEFAULT_LONGBENCH))
    p.add_argument("--python", default=str(REPO_ROOT / ".venv/bin/python"))
    p.add_argument("--skip_run", action="store_true")
    p.add_argument("--force_rerun_baselines", action="store_true")
    p.add_argument("--postprocess_only", action="store_true")
    return p.parse_args()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def method_config_blob(method: str, acc_cfg: Mapping[str, Any]) -> Dict[str, Any]:
    base = {
        "method": method,
        **PROTOCOL,
        "acc_drm_version": "acc_drm_v1" if "acc" in method or "drm_acc" in method else None,
    }
    if method in ("acc_sliminfer_v1", "drm_acc_swap_sliminfer_v1", "drm_swap_sliminfer"):
        base["spike"] = {
            "tau_spike": acc_cfg.get("tau_spike"),
            "tau_margin": acc_cfg.get("tau_margin"),
            "tau_spike_nei_max": acc_cfg.get("tau_spike_nei_max"),
        }
        base["acc"] = {
            "tau_second": acc_cfg.get("tau_second"),
            "tau_gap": acc_cfg.get("tau_gap"),
            "cluster_share_cap": acc_cfg.get("cluster_share_cap"),
            "rescue_ratio": acc_cfg.get("rescue_ratio"),
            "diversify_top_g": acc_cfg.get("diversify_top_g", 2),
        }
        base["drm"] = {
            "rescue_ratio": acc_cfg.get("rescue_ratio"),
            "alpha_diff": acc_cfg.get("alpha_diff", 0.5),
            "lambda_red": acc_cfg.get("lambda_red", 0.25),
            "budget_mode": "K_swap_one_in_one_out",
        }
    if method == "cc_sliminfer_v1":
        base["cc"] = {
            "cc_alpha": 0.25,
            "cc_min_cluster_len": 2,
            "cc_max_neighbors": 2,
            "cc_replacement_cap": 0.05,
        }
    return base


def find_phase2_dir(explicit: Optional[Path]) -> Path:
    if explicit and explicit.exists():
        return explicit.resolve()
    cands: List[Path] = []
    root = REPO_ROOT / "results"
    for p in root.iterdir():
        if not p.is_dir():
            continue
        name = p.name.lower()
        if not any(k in name for k in ("acc_drm_phase2", "acc_drm_live", "phase2_live_prefill")):
            continue
        if (p / "PHASE2_DECISION.json").exists() or (p / "LIVE_PREFILL_DECISION.json").exists():
            cands.append(p)
    if not cands:
        raise FileNotFoundError("No Phase-2 decision directory found under results/")
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0].resolve()


def load_phase2_decision(phase2: Path) -> Dict[str, Any]:
    for name in ("PHASE2_DECISION.json", "LIVE_PREFILL_DECISION.json"):
        p = phase2 / name
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"No decision json in {phase2}")


def phase2_running() -> List[Dict[str, str]]:
    """Detect unfinished Phase-2 live wiring jobs (not Phase-3 referencing phase2_dir)."""
    out = []
    try:
        raw = subprocess.check_output(["ps", "-ef"], text=True)
    except Exception:
        return out
    for line in raw.splitlines():
        if "grep" in line:
            continue
        # Exact Phase-2 orchestrator / scripts only.
        if "run_acc_drm_phase2_live_wiring_audit.py" in line:
            parts = line.split()
            out.append({"pid": parts[1] if len(parts) > 1 else "?", "cmd": line.strip()})
            continue
        # gate_audit under a phase2 live output dir (smoke/full), excluding phase3.
        if "run_qwen25_sliminfer_variants.py" in line and "acc_drm_phase2_live_" in line and "acc_drm_phase3" not in line:
            parts = line.split()
            out.append({"pid": parts[1] if len(parts) > 1 else "?", "cmd": line.strip()})
    return out


def gpu_busy_foreign() -> List[Dict[str, str]]:
    rows = []
    try:
        raw = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
            text=True,
        )
    except Exception:
        return rows
    for line in raw.strip().splitlines():
        if not line.strip():
            continue
        pid = line.split(",")[0].strip()
        name = line.split(",")[1].strip() if "," in line else line
        # Ignore display/remote desktop daemons.
        if any(x in name for x in ("gnome-remote", "Xorg", "gnome-shell", "cursor", "TeamViewer")):
            continue
        if "python" in name.lower() or "pt_main" in name.lower():
            rows.append({"pid": pid, "process": name, "raw": line.strip()})
    return rows


def count_preds(run_dir: Path) -> Dict[str, int]:
    pred = run_dir / "longbench_predictions"
    out = {}
    for t in TASKS:
        p = pred / f"{t}.jsonl"
        out[t] = sum(1 for l in p.read_text(encoding="utf-8").splitlines() if l.strip()) if p.exists() else 0
    return out


def complete(run_dir: Path) -> bool:
    c = count_preds(run_dir)
    return all(c.get(t, 0) >= EXPECTED_BY_TASK[t] for t in TASKS)


def qa_cmd(
    args: argparse.Namespace,
    method: str,
    run_dir: Path,
    *,
    with_selection_audit: bool,
) -> List[str]:
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        args.python,
        str(REPO_ROOT / "experiments/run_qwen25_sliminfer_variants.py"),
        "--model_path", str(args.model_path),
        "--tokenizer_path", str(args.model_path),
        "--model_family", "qwen2",
        "--dtype", "fp16",
        "--device", "cuda",
        "--backend", "flash_attention_2",
        "--longbench_dir", str(args.longbench_dir),
        "--longbench_tasks", ",".join(TASKS),
        "--max_model_len", "131072",
        "--method", method,
        "--eval", "longbench",
        "--block_strategy", "fixed",
        "--block_size", "64",
        "--token_unit_size", "8",
        "--local_query_window", "4",
        "--swap_threshold", "0.9",
        "--prune_layers", "9,18,26",
        "--retain_tokens", "12288,6144,4096",
        "--decode_policy", "original",
        "--rescue_policy", "none",
        "--generation_max_new_tokens", "0",
        "--seed", "0",
        "--min_free_gpu_gb", "0",
        "--output_dir", str(run_dir),
    ]
    # Selection audit ONLY for ACC/DRM — original/CC keep official path (no virtual bypass).
    if with_selection_audit:
        cmd.extend(["--cc_selection_audit_path", str(run_dir / "selection_audit.jsonl")])
    return cmd


def seed_protocol_ok(seed_dir: Path, method: str, model_path: str) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    run = seed_dir / f"run_{method}"
    if not complete(run):
        reasons.append("incomplete_predictions")
        return False, reasons
    cfg_path = run / "config.json"
    if not cfg_path.exists():
        reasons.append("missing_config")
        return False, reasons
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    args = cfg.get("args") or {}
    sc = cfg.get("sliminfer_config") or {}
    if str(args.get("method")) != method and not (
        method == "cc_sliminfer_v1" and str(sc.get("variant_method")) in {"cc_sliminfer", "cc_sliminfer_v1"}
    ):
        reasons.append(f"method_mismatch:{args.get('method')}/{sc.get('variant_method')}")
    if str(args.get("prune_layers")) != "9,18,26":
        reasons.append(f"prune_mismatch:{args.get('prune_layers')}")
    if str(args.get("retain_tokens")) != "12288,6144,4096":
        reasons.append(f"retain_mismatch:{args.get('retain_tokens')}")
    if int(args.get("block_size") or 0) != 64:
        reasons.append("block_size_mismatch")
    if int(args.get("token_unit_size") or 0) != 8:
        reasons.append("token_unit_mismatch")
    if float(args.get("swap_threshold") or 0) != 0.9:
        reasons.append("gamma_mismatch")
    if int(args.get("local_query_window") or 0) != 4:
        reasons.append("query_window_mismatch")
    if str(args.get("decode_policy") or "") != "original":
        reasons.append("decode_mismatch")
    if str(args.get("rescue_policy") or "none") not in {"none", ""}:
        reasons.append("rescue_policy_mismatch")
    if int(args.get("seed") if args.get("seed") is not None else -1) != 0:
        reasons.append(f"seed_mismatch:{args.get('seed')!r}")
    # Model path must resolve to same snapshot.
    mp = str(args.get("model_path") or "")
    if Path(mp).resolve() != Path(model_path).resolve() and not mp.endswith(
        "a09a35458c702b33eeacc393d103063234e8bc28"
    ):
        reasons.append(f"model_mismatch:{mp[-48:]}")
    # Official QA must not have used audit-only virtual bypass.
    if str(args.get("cc_selection_audit_path") or sc.get("cc_selection_audit_path") or "").strip():
        reasons.append("seed_used_selection_audit_path")
    return (len(reasons) == 0), reasons


def copy_seed(seed_run: Path, dst_run: Path) -> None:
    dst_run.mkdir(parents=True, exist_ok=True)
    for name in ("longbench_predictions", "longbench_scores.csv", "config.json", "metrics.json"):
        src = seed_run / name
        dst = dst_run / name
        if not src.exists():
            continue
        if src.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)


def write_sample_manifest(out_dir: Path, longbench_dir: Path) -> None:
    rows = []
    for task in TASKS:
        # LongBench jsonl naming in this repo
        cand = [
            Path(longbench_dir) / f"{task}.jsonl",
            Path(longbench_dir) / "data" / f"{task}.jsonl",
        ]
        path = next((p for p in cand if p.exists()), None)
        if path is None:
            continue
        n = 0
        with path.open(encoding="utf-8") as f:
            for i, line in enumerate(f):
                if not line.strip():
                    continue
                if n >= EXPECTED_BY_TASK[task]:
                    break
                obj = json.loads(line)
                q = str(obj.get("input") or obj.get("question") or "")
                ctx = str(obj.get("context") or "")
                sid = str(obj.get("_id") or obj.get("id") or i)
                rows.append(
                    {
                        "task": task,
                        "sample_index": n,
                        "sample_id": sid,
                        "question_hash": sha256_text(q),
                        "context_hash": sha256_text(ctx[:4096]),
                        "gold_present": int(bool(obj.get("answers"))),
                    }
                )
                n += 1
    write_csv(
        out_dir / "PHASE3_SAMPLE_MANIFEST.csv",
        rows,
        ["task", "sample_index", "sample_id", "question_hash", "context_hash", "gold_present"],
    )
    by_task = {t: sum(1 for r in rows if r["task"] == t) for t in TASKS}
    (out_dir / "PHASE3_SAMPLE_HASH_AUDIT.md").write_text(
        "# Phase-3 Sample Hash Audit\n\n"
        + f"Total rows: {len(rows)} (expect 1150)\n\n"
        + json.dumps(by_task, indent=2)
        + "\n\nGold answers are scoring-only; never used in ACC/DRM selection.\n",
        encoding="utf-8",
    )


def read_scores(run_dir: Path) -> Dict[str, Any]:
    path = run_dir / "longbench_scores.csv"
    if not path.exists():
        return {}
    per = {}
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("status") == "ok" and row.get("score"):
                per[row["dataset"]] = float(row["score"])
    if not per:
        return {}
    return {"per_task": per, "macro": round(sum(per.values()) / len(per), 2)}


def consolidate(run_dir: Path, method: str) -> List[Dict[str, Any]]:
    rows = load_predictions(run_dir / "longbench_predictions")
    out = []
    with (run_dir / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            sc = per_sample_score(row["task"], row["pred"], row["answers"], row["all_classes"])
            e = {**row, "method": method, "score": sc}
            out.append(e)
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    write_csv(
        run_dir / "scores_by_sample.csv",
        out,
        ["method", "task", "sample_id", "sample_index", "score", "length", "pred"],
    )
    return out


def paired_delta(base: List[Dict[str, Any]], var: List[Dict[str, Any]], label: str) -> List[Dict[str, Any]]:
    bmap = {(r["task"], str(r["sample_id"])): r for r in base}
    out = []
    for v in var:
        b = bmap.get((v["task"], str(v["sample_id"])))
        if not b:
            continue
        bs, vs = float(b["score"]), float(v["score"])
        out.append(
            {
                "compare": label,
                "method": v["method"],
                "task": v["task"],
                "sample_id": v["sample_id"],
                "baseline_score": bs,
                "variant_score": vs,
                "delta": round(vs - bs, 4),
                "pred_changed": int(str(b.get("pred", "")) != str(v.get("pred", ""))),
            }
        )
    return out


def bootstrap_ci(deltas: Sequence[float], n: int = 10000, seed: int = 0) -> Dict[str, float]:
    rng = random.Random(seed)
    if not deltas:
        return {}
    macros = []
    for _ in range(n):
        sample = [deltas[rng.randrange(len(deltas))] for _ in range(len(deltas))]
        macros.append(sum(sample) / len(sample))
    macros.sort()
    return {
        "mean_delta": round(sum(deltas) / len(deltas), 4),
        "ci95_low": round(macros[int(0.025 * len(macros))], 4),
        "ci95_high": round(macros[int(0.975 * len(macros))], 4),
        "n_boot": n,
        "seed": seed,
    }


def outlier_sensitivity(deltas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    wins = sorted([d for d in deltas if float(d["delta"]) > 0], key=lambda d: -float(d["delta"]))
    all_d = [float(d["delta"]) for d in deltas]
    base_mean = sum(all_d) / len(all_d) if all_d else 0.0
    rows = []
    for k in (0, 1, 2, len(wins)):
        drop = {id(x) for x in wins[:k]} if k < len(wins) else {id(x) for x in wins}
        kept = [float(d["delta"]) for d in deltas if id(d) not in drop]
        rows.append(
            {
                "remove_top_wins": k if k < len(wins) else "all_wins",
                "n_kept": len(kept),
                "mean_delta": round(sum(kept) / len(kept), 6) if kept else 0.0,
                "base_mean_delta": round(base_mean, 6),
            }
        )
    if wins:
        rows.append(
            {
                "remove_top_wins": "top1_contribution_share",
                "n_kept": "",
                "mean_delta": "",
                "base_mean_delta": "",
                "top1_delta": wins[0]["delta"],
                "top1_share_of_positive_sum": round(
                    float(wins[0]["delta"]) / max(1e-12, sum(float(w["delta"]) for w in wins)), 6
                ),
                "task": wins[0]["task"],
                "sample_id": wins[0]["sample_id"],
            }
        )
    return rows


def summarize_selection_audit(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"n": 0}
    rows = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
    n = max(1, len(rows))

    def rate(field: str) -> float:
        s = 0
        for r in rows:
            v = r.get(field)
            if v is None and isinstance(r.get("acc"), dict):
                v = r["acc"].get(field)
            s += int(v or 0)
        return s / n

    return {
        "n": len(rows),
        "spike_condition_rate": rate("spike_condition_triggered"),
        "spike_changed_rate": rate("spike_protection_changed_selection"),
        "multi_locus_rate": rate("multi_locus_triggered"),
        "cap_rate": rate("cluster_budget_cap_triggered"),
        "drm_accepted_mean": sum(float(r.get("drm_swap_accepted") or r.get("accepted_swaps_per_layer") or 0) for r in rows)
        / n,
        "drm_accepted_layers": sum(1 for r in rows if int(r.get("drm_swap_accepted") or 0) > 0),
        "over_budget": sum(int(r.get("over_budget") or 0) for r in rows),
        "sink_violations": sum(int(r.get("sink_violation") or 0) for r in rows),
        "high_conf_spike_repl": sum(int(r.get("high_confidence_spike_replacement_count") or 0) for r in rows),
        "second_locus_repl": sum(int(r.get("second_locus_replacement_count") or 0) for r in rows),
    }


def write_runtime_csvs(out_dir: Path, method_dirs: Mapping[str, Path]) -> None:
    active_rows = []
    trig_rows = []
    drm_rows = []
    for method, run_dir in method_dirs.items():
        audit = run_dir / "selection_audit.jsonl"
        if not audit.exists():
            continue
        for r in (json.loads(l) for l in audit.open(encoding="utf-8") if l.strip()):
            acc = r.get("acc") if isinstance(r.get("acc"), dict) else {}
            active_rows.append(
                {
                    "method": method,
                    "task": r.get("task"),
                    "sample_index": r.get("sample_index"),
                    "layer_idx": r.get("layer_idx"),
                    "num_selected": r.get("num_selected"),
                    "num_keep_target": r.get("num_keep_target"),
                    "churn_vs_silm": r.get("churn_vs_silm"),
                    "over_budget": r.get("over_budget"),
                    "sink_violation": r.get("sink_violation"),
                }
            )
            trig_rows.append(
                {
                    "method": method,
                    "task": r.get("task"),
                    "sample_index": r.get("sample_index"),
                    "layer_idx": r.get("layer_idx"),
                    "multi_locus": r.get("multi_locus_triggered", acc.get("multi_locus_triggered")),
                    "spike_condition": r.get("spike_condition_triggered", acc.get("spike_condition_triggered")),
                    "spike_changed": r.get(
                        "spike_protection_changed_selection", acc.get("spike_protection_changed_selection")
                    ),
                    "cap_condition": r.get(
                        "cluster_budget_cap_triggered", acc.get("cluster_budget_cap_triggered")
                    ),
                }
            )
            if "drm" in method:
                drm_rows.append(
                    {
                        "method": method,
                        "task": r.get("task"),
                        "sample_index": r.get("sample_index"),
                        "layer_idx": r.get("layer_idx"),
                        "drm_candidate_available": r.get("drm_candidate_available"),
                        "drm_candidate_eligible": r.get("drm_candidate_eligible"),
                        "drm_swap_accepted": r.get("drm_swap_accepted"),
                        "drm_swap_rejected": r.get("drm_swap_rejected"),
                        "rejection_reason": "|".join(r.get("rejection_reason") or [])
                        if isinstance(r.get("rejection_reason"), list)
                        else r.get("rejection_reason"),
                        "accepted_swaps_per_layer": r.get("accepted_swaps_per_layer"),
                        "high_confidence_spike_replacement_count": r.get(
                            "high_confidence_spike_replacement_count"
                        ),
                        "second_locus_replacement_count": r.get("second_locus_replacement_count"),
                    }
                )
    if active_rows:
        write_csv(out_dir / "runtime_active_set_stats.csv", active_rows, list(active_rows[0].keys()))
    if trig_rows:
        write_csv(out_dir / "runtime_trigger_stats.csv", trig_rows, list(trig_rows[0].keys()))
    if drm_rows:
        write_csv(out_dir / "runtime_drm_swap_stats.csv", drm_rows, list(drm_rows[0].keys()))


def decide(scores: Mapping[str, Dict[str, Any]], boots: Mapping[str, Dict[str, Any]], runtime: Mapping[str, Any]) -> Dict[str, Any]:
    base = scores.get("sliminfer_original", {})
    cc = scores.get("cc_sliminfer_v1", {})
    acc = scores.get("acc_sliminfer_v1", {})
    drm = scores.get("drm_acc_swap_sliminfer_v1", {})
    drm_vs_cc = round(float(drm.get("macro", 0)) - float(cc.get("macro", 0)), 2)
    drm_vs_acc = round(float(drm.get("macro", 0)) - float(acc.get("macro", 0)), 2)
    acc_vs_cc = round(float(acc.get("macro", 0)) - float(cc.get("macro", 0)), 2)

    def task_d(a: Dict[str, Any], b: Dict[str, Any], t: str) -> Optional[float]:
        av = (a.get("per_task") or {}).get(t)
        bv = (b.get("per_task") or {}).get(t)
        if av is None or bv is None:
            return None
        return round(float(av) - float(bv), 2)

    musique_d = task_d(drm, cc, "musique")
    wiki_d = task_d(drm, cc, "2wikimqa")
    trivia_d = task_d(drm, cc, "triviaqa")
    # negative migration of CC vs original was known; relief if DRM improves vs CC
    wiki_relief = wiki_d is not None and wiki_d > 0
    trivia_relief = trivia_d is not None and trivia_d > 0
    musique_ok = musique_d is None or musique_d >= -0.5
    hard_reg = [
        t
        for t in TASKS
        if (task_d(drm, cc, t) is not None and task_d(drm, cc, t) <= -1.0)
    ]
    boot = boots.get("drm_acc_swap_sliminfer_v1__vs__cc_sliminfer_v1") or {}
    ci_lo = boot.get("ci95_low")
    ci_hi = boot.get("ci95_high")
    ci_cross = ci_lo is not None and ci_hi is not None and float(ci_lo) < 0 < float(ci_hi)

    status = "EXPLORATORY"
    if drm_vs_cc < 0.20:
        status = "NO_GO_FOR_17TASK"
    if (not musique_ok) and (wiki_relief or trivia_relief):
        status = "TRADEOFF_NO_GO"
    if drm_vs_cc >= 0.20 and musique_ok and (wiki_relief or trivia_relief) and not hard_reg:
        if ci_cross:
            status = "EXPLORATORY"
        else:
            status = "GO_CANDIDATE"

    return {
        "status": status,
        "drm_vs_cc_macro": drm_vs_cc,
        "drm_vs_acc_macro": drm_vs_acc,
        "acc_vs_cc_macro": acc_vs_cc,
        "musique_delta_vs_cc": musique_d,
        "2wikimqa_delta_vs_cc": wiki_d,
        "triviaqa_delta_vs_cc": trivia_d,
        "hard_regressions_vs_cc": hard_reg,
        "bootstrap_drm_vs_cc": boot,
        "runtime_ok": all(
            int((runtime.get(m) or {}).get("over_budget") or 0) == 0
            and int((runtime.get(m) or {}).get("sink_violations") or 0) == 0
            for m in ("acc_sliminfer_v1", "drm_acc_swap_sliminfer_v1")
            if m in runtime
        ),
        "start_17task": False,
        "run_qa_completed": True,
    }


def postprocess(out_dir: Path) -> Dict[str, Any]:
    method_rows: Dict[str, List[Dict[str, Any]]] = {}
    scores: Dict[str, Dict[str, Any]] = {}
    for method in METHODS:
        run_dir = out_dir / f"run_{method}"
        if not complete(run_dir):
            raise RuntimeError(f"incomplete: {method} counts={count_preds(run_dir)}")
        method_rows[method] = consolidate(run_dir, method)
        scores[method] = read_scores(run_dir)

    all_rows = [r for rows in method_rows.values() for r in rows]
    write_csv(
        out_dir / "scores_by_sample.csv",
        all_rows,
        ["method", "task", "sample_id", "sample_index", "score", "length", "pred"],
    )
    with (out_dir / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # per-task table
    task_rows = []
    for t in TASKS:
        row = {"task": t}
        for m in METHODS:
            row[m] = (scores.get(m, {}).get("per_task") or {}).get(t)
        if row["sliminfer_original"] is not None and row["cc_sliminfer_v1"] is not None:
            row["delta_cc_vs_orig"] = round(float(row["cc_sliminfer_v1"]) - float(row["sliminfer_original"]), 2)
        if row["acc_sliminfer_v1"] is not None and row["cc_sliminfer_v1"] is not None:
            row["delta_acc_vs_cc"] = round(float(row["acc_sliminfer_v1"]) - float(row["cc_sliminfer_v1"]), 2)
        if row["drm_acc_swap_sliminfer_v1"] is not None and row["cc_sliminfer_v1"] is not None:
            row["delta_drm_vs_cc"] = round(
                float(row["drm_acc_swap_sliminfer_v1"]) - float(row["cc_sliminfer_v1"]), 2
            )
        if row["drm_acc_swap_sliminfer_v1"] is not None and row["acc_sliminfer_v1"] is not None:
            row["delta_drm_vs_acc"] = round(
                float(row["drm_acc_swap_sliminfer_v1"]) - float(row["acc_sliminfer_v1"]), 2
            )
        task_rows.append(row)
    write_csv(out_dir / "scores_by_task.csv", task_rows, list(task_rows[0].keys()))

    summary_rows = []
    for m in METHODS:
        summary_rows.append(
            {
                "method": m,
                "macro_avg": scores.get(m, {}).get("macro"),
                **{t: (scores.get(m, {}).get("per_task") or {}).get(t) for t in TASKS},
                "delta_vs_original": None
                if scores.get(m, {}).get("macro") is None or scores.get("sliminfer_original", {}).get("macro") is None
                else round(float(scores[m]["macro"]) - float(scores["sliminfer_original"]["macro"]), 2),
                "delta_vs_cc": None
                if scores.get(m, {}).get("macro") is None or scores.get("cc_sliminfer_v1", {}).get("macro") is None
                else round(float(scores[m]["macro"]) - float(scores["cc_sliminfer_v1"]["macro"]), 2),
            }
        )
    write_csv(
        out_dir / "scores_summary.csv",
        summary_rows,
        ["method", "macro_avg", "delta_vs_original", "delta_vs_cc", *TASKS],
    )

    # category (single/multi-hop grouping used historically)
    cat = {
        "Single-Doc QA": ["qasper", "multifieldqa_en"],
        "Multi-Hop QA": ["hotpotqa", "2wikimqa", "musique"],
        "Few-shot/Trivia": ["triviaqa"],
    }
    cat_rows = []
    for m in METHODS:
        for cname, tasks in cat.items():
            vals = [(scores.get(m, {}).get("per_task") or {}).get(t) for t in tasks]
            vals = [float(v) for v in vals if v is not None]
            cat_rows.append(
                {
                    "method": m,
                    "category": cname,
                    "macro": round(sum(vals) / len(vals), 2) if vals else None,
                    "n_tasks": len(vals),
                }
            )
    write_csv(out_dir / "scores_by_category.csv", cat_rows, ["method", "category", "macro", "n_tasks"])

    boots: Dict[str, Dict[str, float]] = {}
    wlt_rows = []
    outlier_rows = []
    compares = [
        ("acc_sliminfer_v1", "sliminfer_original"),
        ("acc_sliminfer_v1", "cc_sliminfer_v1"),
        ("drm_acc_swap_sliminfer_v1", "sliminfer_original"),
        ("drm_acc_swap_sliminfer_v1", "cc_sliminfer_v1"),
        ("drm_acc_swap_sliminfer_v1", "acc_sliminfer_v1"),
    ]
    all_deltas = []
    for var, base in compares:
        label = f"{var}__vs__{base}"
        deltas = paired_delta(method_rows[base], method_rows[var], label)
        all_deltas.extend(deltas)
        dv = [float(d["delta"]) for d in deltas]
        wins = sum(1 for d in dv if d > 0)
        losses = sum(1 for d in dv if d < 0)
        ties = sum(1 for d in dv if d == 0)
        changed = sum(int(d["pred_changed"]) for d in deltas)
        boot = bootstrap_ci(dv, n=10000, seed=0)
        boots[label] = boot
        wlt_rows.append(
            {
                "compare": label,
                "wins": wins,
                "losses": losses,
                "ties": ties,
                "changed_predictions": changed,
                "identical_ratio": round(1.0 - changed / max(1, len(deltas)), 6),
                "mean_delta": boot.get("mean_delta"),
                "ci95_low": boot.get("ci95_low"),
                "ci95_high": boot.get("ci95_high"),
            }
        )
        for row in outlier_sensitivity(deltas):
            outlier_rows.append({"compare": label, **row})

    write_csv(
        out_dir / "win_loss_tie.csv",
        wlt_rows,
        list(wlt_rows[0].keys()) if wlt_rows else ["compare"],
    )
    write_csv(
        out_dir / "per_sample_deltas.csv",
        all_deltas,
        ["compare", "method", "task", "sample_id", "baseline_score", "variant_score", "delta", "pred_changed"],
    )
    (out_dir / "bootstrap_results.json").write_text(json.dumps(boots, indent=2) + "\n")
    if outlier_rows:
        # normalize fields
        fields = sorted({k for r in outlier_rows for k in r.keys()})
        write_csv(out_dir / "outlier_sensitivity.csv", outlier_rows, fields)

    write_runtime_csvs(
        out_dir,
        {m: out_dir / f"run_{m}" for m in ("acc_sliminfer_v1", "drm_acc_swap_sliminfer_v1")},
    )
    runtime = {
        m: summarize_selection_audit(out_dir / f"run_{m}" / "selection_audit.jsonl")
        for m in ("acc_sliminfer_v1", "drm_acc_swap_sliminfer_v1")
    }
    (out_dir / "PHASE3_RUNTIME_AUDIT.md").write_text(
        "# Phase-3 Runtime Audit\n\n```json\n" + json.dumps(runtime, indent=2) + "\n```\n",
        encoding="utf-8",
    )

    # Fail-closed runtime safety for ACC/DRM
    for m, st in runtime.items():
        if st.get("n", 0) > 0 and (
            int(st.get("over_budget") or 0) > 0
            or int(st.get("sink_violations") or 0) > 0
            or int(st.get("high_conf_spike_repl") or 0) > 0
            or int(st.get("second_locus_repl") or 0) > 0
        ):
            (out_dir / "PHASE3_RUNTIME_FAILURE.md").write_text(
                f"# Runtime failure for {m}\n\n```json\n{json.dumps(st, indent=2)}\n```\n",
                encoding="utf-8",
            )

    decision = decide(scores, boots, runtime)
    (out_dir / "PHASE3_DECISION.json").write_text(json.dumps(decision, indent=2) + "\n")

    # main report table
    def fmt(x: Any) -> str:
        return "" if x is None else str(x)

    lines = [
        "# ACC/DRM Phase-3 6-task Report\n\n",
        "| Method | Qasper | MFQA | HPQA | 2Wiki | MuSiQue | TriviaQA | Macro | Δ vs original | Δ vs CC |\n",
        "|--------|-------:|-----:|-----:|------:|--------:|---------:|------:|--------------:|--------:|\n",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['method']} | {fmt(row.get('qasper'))} | {fmt(row.get('multifieldqa_en'))} | "
            f"{fmt(row.get('hotpotqa'))} | {fmt(row.get('2wikimqa'))} | {fmt(row.get('musique'))} | "
            f"{fmt(row.get('triviaqa'))} | {fmt(row.get('macro_avg'))} | "
            f"{fmt(row.get('delta_vs_original'))} | {fmt(row.get('delta_vs_cc'))} |\n"
        )
    lines.append("\n## Decision\n\n```json\n" + json.dumps(decision, indent=2) + "\n```\n")
    lines.append("\n## Win/Loss/Tie\n\n```json\n" + json.dumps(wlt_rows, indent=2) + "\n```\n")
    (out_dir / "ACC_DRM_PHASE3_6TASK_REPORT.md").write_text("".join(lines), encoding="utf-8")

    (out_dir / "ACC_DRM_PHASE3_FAIRNESS_AUDIT.md").write_text(
        "# Fairness Audit\n\n"
        "- Same model/tokenizer/prompt/scorer/decode for all methods.\n"
        "- Same prune/retain/block/unit/gamma/window.\n"
        "- DRM-ACC is **K-budget swap**, not K+M add.\n"
        "- Gold never enters selection / DRM / sacrifice.\n"
        "- Original/CC QA path: no `cc_selection_audit_path` (no audit-only virtual bypass).\n"
        "- ACC/DRM selection_audit enabled for runtime stats only; methods already skip virtual.\n",
        encoding="utf-8",
    )
    (out_dir / "ACC_DRM_PHASE3_OFFICIAL_PATH_AUDIT.md").write_text(
        "# Official Path Audit\n\n"
        "- Runner: `experiments/run_qwen25_sliminfer_variants.py --eval longbench`\n"
        "- Methods dispatched via official `variant_method` wiring in `utils/cache_utils.py`\n"
        "- No CPU-replay substitute for ACC/DRM QA.\n",
        encoding="utf-8",
    )
    (out_dir / "ACC_DRM_PHASE3_RUNTIME_AUDIT.md").write_text(
        (out_dir / "PHASE3_RUNTIME_AUDIT.md").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (out_dir / "ACC_DRM_PHASE3_CASE_STUDY.md").write_text(
        "# Case Study Focus\n\n"
        f"- MuSiQue Δ(DRM vs CC) = {decision.get('musique_delta_vs_cc')}\n"
        f"- 2Wiki Δ(DRM vs CC) = {decision.get('2wikimqa_delta_vs_cc')}\n"
        f"- TriviaQA Δ(DRM vs CC) = {decision.get('triviaqa_delta_vs_cc')}\n"
        "- See `per_sample_deltas.csv` for paired sample analysis.\n",
        encoding="utf-8",
    )
    (out_dir / "ACC_DRM_PHASE3_CLAIM_GUARD.md").write_text(
        "# Claim Guard\n\n"
        f"- Status: **{decision.get('status')}**\n"
        "- Do **not** auto-start 17-task.\n"
        "- Do **not** retune parameters from this QA.\n"
        "- K+M DRM-Add is out of scope for this table.\n"
        f"- DRM vs CC macro = {decision.get('drm_vs_cc_macro')} (GO needs ≥ +0.20).\n",
        encoding="utf-8",
    )
    return decision


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_log = REPO_ROOT / "results" / "acc_drm_phase3_6task_progress.log"

    if args.postprocess_only:
        decision = postprocess(out_dir)
        print(json.dumps({"out_dir": str(out_dir), "decision": decision}, indent=2))
        return 0

    # --- Phase2 locating ---
    try:
        phase2 = find_phase2_dir(args.phase2_dir)
        phase2_dec = load_phase2_decision(phase2)
    except Exception as exc:
        (out_dir / "PHASE3_BLOCKED_BY_PHASE2_AUDIT.md").write_text(
            f"# Phase-3 blocked\n\nCannot locate/load Phase-2 decision: {exc}\n",
            encoding="utf-8",
        )
        print(json.dumps({"blocked": True, "reason": str(exc)}, indent=2))
        return 2

    running = phase2_running()
    if running:
        (out_dir / "PHASE3_BLOCKED_PHASE2_RUNNING.md").write_text(
            "# Phase-3 blocked: Phase-2 still running\n\n"
            + json.dumps(running, indent=2)
            + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"blocked": True, "reason": "phase2_running", "procs": running}, indent=2))
        return 2

    # --- Phase2 GO gates ---
    fail_items = []
    if not phase2_dec.get("phase2_go"):
        fail_items.append("phase2_go!=true")
    if not (phase2_dec.get("wiring_pass") or (phase2_dec.get("wiring") or {}).get("pass")):
        fail_items.append("wiring_pass_failed")
    if not phase2_dec.get("acc_nonzero_change"):
        fail_items.append("acc_nonzero_change_failed")
    if not phase2_dec.get("drm_nonzero_change"):
        fail_items.append("drm_nonzero_change_failed")
    if float(phase2_dec.get("drm_swap_accepted_mean") or 0) <= 0:
        fail_items.append("drm_accepted_swaps_zero")
    if not phase2_dec.get("over_budget_ok"):
        fail_items.append("over_budget_ok_failed")
    if not phase2_dec.get("sink_ok"):
        fail_items.append("sink_ok_failed")
    wiring = phase2_dec.get("wiring") or {}
    if not wiring.get("cli_default_audit_path_empty"):
        fail_items.append("official_path_may_use_audit_bypass")
    if not wiring.get("cc_uses_cluster_continuity_not_acc"):
        fail_items.append("cc_possibly_polluted")

    # Phase1B safety counters
    phase1b = REPO_ROOT / "results" / "acc_drm_phase1b_spike_calibration_20260714_085424" / "PHASE1B_DECISION.json"
    if phase1b.exists():
        p1 = json.loads(phase1b.read_text(encoding="utf-8"))
        if int(p1.get("high_confidence_spike_replacement_count") or 0) != 0:
            fail_items.append("high_confidence_spike_replacement!=0")
        if int(p1.get("second_locus_replacement_count") or 0) != 0:
            fail_items.append("second_locus_replacement!=0")

    if fail_items:
        (out_dir / "PHASE3_BLOCKED_BY_PHASE2_AUDIT.md").write_text(
            "# Phase-3 blocked by Phase-2 audit\n\n"
            + f"phase2_dir: `{phase2}`\n\n"
            + "Failed items:\n"
            + "\n".join(f"- {x}" for x in fail_items)
            + "\n\n```json\n"
            + json.dumps(phase2_dec, indent=2)
            + "\n```\n",
            encoding="utf-8",
        )
        print(json.dumps({"blocked": True, "phase2_dir": str(phase2), "failed": fail_items}, indent=2))
        return 2

    # --- GPU preflight ---
    busy = gpu_busy_foreign()
    if busy:
        (out_dir / "BLOCKED_GPU_BUSY.md").write_text(
            "# GPU busy\n\n" + json.dumps(busy, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"blocked": True, "reason": "gpu_busy", "apps": busy}, indent=2))
        return 2
    if not Path(args.model_path).exists():
        (out_dir / "BLOCKED_GPU_BUSY.md").write_text(
            f"# Model path missing\n\n{args.model_path}\n", encoding="utf-8"
        )
        return 2

    acc_cfg = json.loads(ACC_V1.read_text(encoding="utf-8")) if ACC_V1.exists() else {
        "tau_spike": 0.28,
        "tau_margin": 0.20,
        "tau_spike_nei_max": 0.18,
        "rescue_ratio": 0.05,
        "cluster_share_cap": 0.35,
        "tau_second": 0.30,
        "tau_gap": 0.08,
    }

    # git snapshot
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), text=True).strip()
        diff_stat = subprocess.check_output(
            ["git", "diff", "--stat", "HEAD"], cwd=str(REPO_ROOT), text=True
        )[:4000]
    except Exception:
        commit, diff_stat = "unknown", ""

    hashes = {}
    for m in METHODS:
        blob = method_config_blob(m, acc_cfg)
        hashes[m] = {
            "config_hash": sha256_text(json.dumps(blob, sort_keys=True)),
            "config": blob,
        }
    phase2_hash = sha256_text(json.dumps(phase2_dec, sort_keys=True))
    frozen = {
        "phase": "3",
        "phase2_dir": str(phase2),
        "phase2_config_hash": phase2_hash,
        "git_commit": commit,
        "git_diff_stat": diff_stat,
        "model_path": args.model_path,
        "tokenizer_path": args.model_path,
        "protocol": PROTOCOL,
        "acc_drm_v1": acc_cfg,
        "tasks": list(TASKS),
        "expected_by_task": EXPECTED_BY_TASK,
        "methods": list(METHODS),
        "notes": [
            "Parameters frozen from Phase-2 / acc_drm_v1; no QA-driven retuning.",
            "DRM-Add K+M forbidden.",
            "selection_audit only for ACC/DRM methods.",
        ],
    }
    (out_dir / "PHASE3_FROZEN_CONFIG.json").write_text(json.dumps(frozen, indent=2) + "\n")
    (out_dir / "PHASE3_METHOD_HASHES.json").write_text(json.dumps(hashes, indent=2) + "\n")
    (out_dir / "PHASE3_INPUT_PROTOCOL.md").write_text(
        "# Phase-3 Input Protocol\n\n"
        f"- Phase2: `{phase2}`\n"
        f"- Phase2 hash: `{phase2_hash}`\n"
        f"- Git: `{commit}`\n"
        f"- Model: `{args.model_path}`\n\n"
        "## Official SlimInfer settings\n\n"
        + json.dumps(PROTOCOL, indent=2)
        + "\n\n## ACC/DRM spike (frozen)\n\n"
        + json.dumps(
            {
                "tau_spike": acc_cfg.get("tau_spike"),
                "tau_margin": acc_cfg.get("tau_margin"),
                "tau_spike_nei_max": acc_cfg.get("tau_spike_nei_max"),
                "rescue_ratio": acc_cfg.get("rescue_ratio"),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_sample_manifest(out_dir, Path(args.longbench_dir))

    # --- Seed reuse ---
    reuse_audit = {"seed_dir": str(args.seed_6task_dir), "reuse": {}, "rerun": []}
    for method in ("sliminfer_original", "cc_sliminfer_v1"):
        ok, reasons = seed_protocol_ok(args.seed_6task_dir, method, args.model_path)
        if args.force_rerun_baselines:
            ok, reasons = False, reasons + ["force_rerun_baselines"]
        reuse_audit["reuse"][method] = {"ok": ok, "reasons": reasons}
        if not ok:
            reuse_audit["rerun"].append(method)
    (out_dir / "PHASE3_SEED_REUSE_AUDIT.md").write_text(
        "# Seed Reuse Audit\n\n```json\n" + json.dumps(reuse_audit, indent=2) + "\n```\n"
        + "\nReuse only when protocol/runtime fields strictly match. ACC/DRM always run live.\n",
        encoding="utf-8",
    )

    # Launch methods sequentially; tee to progress log.
    progress_log.parent.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        line = f"[{datetime.now().isoformat()}] {msg}"
        print(line, flush=True)
        with progress_log.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    log(f"PHASE3_START out_dir={out_dir} phase2={phase2}")
    log(f"method_hashes={[hashes[m]['config_hash'] for m in METHODS]}")

    if not args.skip_run:
        for method in METHODS:
            run_dir = out_dir / f"run_{method}"
            if complete(run_dir):
                log(f"SKIP_COMPLETE {method}")
                continue
            # reuse baselines
            if method in ("sliminfer_original", "cc_sliminfer_v1") and reuse_audit["reuse"].get(method, {}).get("ok"):
                src = args.seed_6task_dir / f"run_{method}"
                log(f"REUSE_SEED {method} from {src}")
                copy_seed(src, run_dir)
                if complete(run_dir):
                    continue
                log(f"REUSE_INCOMPLETE_FALLBACK_RERUN {method}")

            with_audit = method in ("acc_sliminfer_v1", "drm_acc_swap_sliminfer_v1")
            cmd = qa_cmd(args, method, run_dir, with_selection_audit=with_audit)
            (out_dir / f"command_{method}.json").write_text(json.dumps({"command": cmd}, indent=2) + "\n")
            env = os.environ.copy()
            env["PYTHONPATH"] = str(REPO_ROOT)
            env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
            log_path = out_dir / f"run_{method}.log"
            log(f"RUN_START {method} audit={with_audit}")
            t0 = time.time()
            with log_path.open("w", encoding="utf-8") as lf:
                # also append short markers to global progress log
                rc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
            log(f"RUN_END {method} rc={rc} elapsed_s={time.time()-t0:.1f}")
            if rc != 0:
                (out_dir / "FAILED.json").write_text(
                    json.dumps({"method": method, "returncode": rc, "log": str(log_path)}, indent=2) + "\n"
                )
                return 1
            if not complete(run_dir):
                (out_dir / "FAILED.json").write_text(
                    json.dumps({"method": method, "error": "incomplete", "counts": count_preds(run_dir)}, indent=2)
                    + "\n"
                )
                return 1

            # Mid-run runtime safety for ACC/DRM
            if with_audit:
                st = summarize_selection_audit(run_dir / "selection_audit.jsonl")
                if st.get("n", 0) > 0 and (
                    int(st.get("over_budget") or 0) > 0
                    or int(st.get("sink_violations") or 0) > 0
                    or int(st.get("high_conf_spike_repl") or 0) > 0
                    or int(st.get("second_locus_repl") or 0) > 0
                ):
                    (out_dir / "PHASE3_RUNTIME_FAILURE.md").write_text(
                        f"# Runtime failure after {method}\n\n```json\n{json.dumps(st, indent=2)}\n```\n",
                        encoding="utf-8",
                    )
                    log(f"RUNTIME_FAILURE {method} {st}")
                    return 1

    # Postprocess if all complete
    if all(complete(out_dir / f"run_{m}") for m in METHODS):
        decision = postprocess(out_dir)
        log(f"PHASE3_DONE status={decision.get('status')}")
        print(
            json.dumps(
                {
                    "phase2_dir": str(phase2),
                    "phase2_go": True,
                    "out_dir": str(out_dir),
                    "reuse": reuse_audit["reuse"],
                    "progress_log": str(progress_log),
                    "method_hashes": {m: hashes[m]["config_hash"] for m in METHODS},
                    "decision": decision,
                    "started": True,
                },
                indent=2,
            )
        )
        return 0

    print(
        json.dumps(
            {
                "phase2_dir": str(phase2),
                "phase2_go": True,
                "out_dir": str(out_dir),
                "reuse": reuse_audit["reuse"],
                "progress_log": str(progress_log),
                "method_hashes": {m: hashes[m]["config_hash"] for m in METHODS},
                "started": True,
                "note": "runs in progress / incomplete; re-invoke with --postprocess_only after finish",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
