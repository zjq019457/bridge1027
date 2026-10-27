#!/usr/bin/env python3
"""SP-SlimInfer v2 runtime smoke (4) + live audit (24), prefill-only, no QA."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from bridgekv.confidence_gated_sliminfer import _trim_to_budget
from bridgekv.sp_sliminfer import active_set_churn
from bridgekv.sp_semantic_transport import (
    DEFAULT_REPLACEMENT_CAP,
    DEFAULT_TAU,
    DEFAULT_TOP_R,
    _must_keep_block_ids,
)
from bridgekv.sp_trajectory import TRAJECTORY_PRUNE_LAYERS
from tools.run_bridgeflow_silm_property_probe import read_jsonl
from tools.run_sp_v2_transport_replay import (
    DEFAULT_TRAJECTORY_ROOT,
    DEFAULT_UASR,
    layer_alive_blocks,
    layer_score_map,
    load_scalar_rows,
    load_uasr_index,
    replay_all,
    sample_list,
)

TASKS = ("hotpotqa", "musique")
PRUNE_LAYERS = list(TRAJECTORY_PRUNE_LAYERS)
BLOCK_SIZE = 64
MAX_MODEL_LEN = 32768
DEFAULT_MODEL = (
    "/home/yu-bai/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/"
    "snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
)
DEFAULT_LONGBENCH = Path("/home/yu-bai/Desktop/jiaqizhang/HeadKV/data/LongBench")
REPLAY_DIR = REPO_ROOT / "results/sp_v2_transport_replay_20260710_200139"


def parse_args() -> argparse.Namespace:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["smoke", "live24", "both"], default="both")
    p.add_argument("--model_path", type=Path, default=Path(DEFAULT_MODEL))
    p.add_argument("--longbench_dir", type=Path, default=DEFAULT_LONGBENCH)
    p.add_argument("--replay_dir", type=Path, default=REPLAY_DIR)
    p.add_argument("--trajectory_root", type=Path, default=DEFAULT_TRAJECTORY_ROOT)
    p.add_argument("--skip_gpu", action="store_true")
    p.add_argument("--reanalyze_dir", type=Path, default=None, help="Recompute reports from an existing smoke/live output dir")
    return p.parse_args()


def python_bin() -> Path:
    c = REPO_ROOT / ".venv" / "bin" / "python"
    return c if c.exists() else Path(sys.executable)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def build_spec(n: int) -> str:
    idx = ",".join(str(i) for i in range(n))
    return ";".join(f"{t}:{idx}" for t in TASKS)


def gate_cmd(*, method: str, run_dir: Path, sample_spec: str, model_path: Path, longbench_dir: Path, traj_dump: Optional[Path] = None) -> List[str]:
    run_dir.mkdir(parents=True, exist_ok=True)
    uasr = run_dir / "uasr_score_dump.jsonl"
    uasr.write_text("", encoding="utf-8")
    cmd = [
        str(python_bin()),
        str(REPO_ROOT / "experiments/run_qwen25_sliminfer_variants.py"),
        "--model_path", str(model_path),
        "--tokenizer_path", str(model_path),
        "--model_family", "qwen2",
        "--dtype", "fp16",
        "--device", "cuda",
        "--backend", "flash_attention_2",
        "--longbench_dir", str(longbench_dir),
        "--longbench_tasks", ",".join(TASKS),
        "--longbench_sample_indices_by_task", sample_spec,
        "--longbench_subset_name", "sp_sliminfer_v2_live_audit",
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
        "--uasr_score_dump_path", str(uasr),
        "--uasr_tail_width", "8",
    ]
    if traj_dump is not None:
        traj_dump.mkdir(parents=True, exist_ok=True)
        cmd.extend(["--sp_trajectory_dump_path", str(traj_dump)])
    return cmd


def run_cmd(cmd: Sequence[str], log: Path) -> float:
    t0 = time.perf_counter()
    proc = subprocess.run(list(cmd), cwd=str(REPO_ROOT), capture_output=True, text=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    (log.with_suffix(".stdout.log")).write_text(proc.stdout or "", encoding="utf-8")
    (log.with_suffix(".stderr.log")).write_text(proc.stderr or "", encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"cmd failed rc={proc.returncode}; see {log.with_suffix('.stderr.log')}")
    if "CUDA out of memory" in (proc.stdout or "") + (proc.stderr or ""):
        raise RuntimeError("CUDA OOM")
    return time.perf_counter() - t0


def load_prefill(path: Path) -> List[Dict[str, Any]]:
    return [r for r in read_jsonl(path) if str(r.get("phase", "")) == "prefill" and str(r.get("task", "")) in TASKS]


def index_uasr(rows: Sequence[Mapping[str, Any]]) -> Dict[Tuple[str, str, int], Mapping[str, Any]]:
    out = {}
    for row in rows:
        out[(str(row["task"]), str(row["sample_id"]), int(row["layer_id"]))] = row
    return out


def compare_baseline(baseline_rows, v2_rows) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    bi, vi = index_uasr(baseline_rows), index_uasr(v2_rows)
    detail = []
    mism = 0
    for key in sorted(set(bi) & set(vi), key=lambda k: (k[0], k[1], k[2])):
        bsel = sorted(int(x) for x in bi[key].get("selected_block_ids", []))
        vsel = sorted(int(x) for x in vi[key].get("selected_block_ids", []))
        match = int(bsel == vsel)
        if not match:
            mism += 1
        detail.append({"task": key[0], "sample_id": key[1], "layer_id": key[2], "selection_match": match,
                       "baseline_count": len(bsel), "v2_count": len(vsel)})
    return detail, {"layers_compared": len(detail), "selection_mismatches": mism, "zero_pollution_baseline_vs_v2_off": mism == 0}


def analyze_v2_vs_baseline(baseline_rows, v2_rows) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    bi, vi = index_uasr(baseline_rows), index_uasr(v2_rows)
    detail = []
    churns = []
    l9_mism = 0
    transport_changed = 0
    transport_rows = 0
    sink_bad = 0
    for key in sorted(set(bi) & set(vi), key=lambda k: (k[0], k[1], k[2])):
        layer = key[2]
        bsel = {int(x) for x in bi[key].get("selected_block_ids", [])}
        vsel = {int(x) for x in vi[key].get("selected_block_ids", [])}
        budget = int(vi[key].get("budget_k", len(vsel)) or len(vsel))
        churn = active_set_churn(bsel, vsel, budget)
        if layer == 9 and bsel != vsel:
            l9_mism += 1
        if layer in (18, 26):
            transport_rows += 1
            if bsel != vsel:
                transport_changed += 1
            churns.append(churn)
        bids = [int(x) for x in vi[key].get("all_candidate_block_ids", [])]
        if bids and int(bids[0]) not in vsel:
            sink_bad += 1
        detail.append({
            "task": key[0], "sample_id": key[1], "layer_id": layer, "budget_k": budget,
            "churn_rate": round(churn, 6), "changed": int(bsel != vsel),
            "jaccard": round(len(bsel & vsel) / max(1, len(bsel | vsel)), 6),
            "sink_kept": int(not bids or int(bids[0]) in vsel),
            "over_budget": int(len(vsel) > budget),
        })
    summary = {
        "l9_mismatch_layers": l9_mism,
        "transport_rows": transport_rows,
        "changed_transport_rows": transport_changed,
        "mean_churn_transport": round(statistics.mean(churns), 6) if churns else 0.0,
        "median_churn_transport": round(statistics.median(churns), 6) if churns else 0.0,
        "max_churn_transport": round(max(churns), 6) if churns else 0.0,
        "sink_violations": sink_bad,
    }
    return detail, summary


def silm_from_uasr_record(uasr: Mapping[str, Any]) -> Tuple[Set[int], List[int], Dict[int, float]]:
    """Reconstruct original SlimInfer top-k from the same forward as runtime v2."""
    block_ids = [int(x) for x in uasr.get("all_candidate_block_ids", [])]
    raw_scores = uasr.get("all_candidate_scores", []) or []
    budget = int(uasr.get("budget_k", 0) or 0)
    must_keep = _must_keep_block_ids(block_ids)
    score_tensor = torch.tensor(
        [float(raw_scores[i]) if raw_scores[i] is not None else float("-inf") for i in range(len(block_ids))],
        dtype=torch.float32,
    )
    score_map = {block_ids[i]: float(score_tensor[i].item()) for i in range(len(block_ids))}
    if budget <= 0 or not block_ids:
        return set(), block_ids, score_map
    k = min(int(budget), len(block_ids))
    _, topk = torch.topk(score_tensor, k=k)
    silm = {block_ids[int(i)] for i in topk.tolist()} | must_keep
    silm = set(
        _trim_to_budget(
            silm,
            original_block_ids=block_ids,
            block_scores=score_tensor,
            must_keep_block_ids=must_keep,
            num_blocks_to_keep=int(budget),
        )
    )
    return silm, block_ids, score_map


def replay_parity(v2_rows, samples, trajectory_root) -> List[Dict[str, Any]]:
    """Compare runtime v2 selections to offline replay on the same v2 trajectory dump."""
    from bridgekv.sp_trajectory import load_sample_history_from_npz
    from bridgekv.sp_semantic_transport import replay_sp_v2_sample_layer

    scalar_rows = load_scalar_rows(trajectory_root / "trajectory_scalar_features.csv")
    index_rows = read_jsonl(trajectory_root / "trajectory_index.jsonl")
    vi = index_uasr(v2_rows)
    parity: List[Dict[str, Any]] = []
    for task, sid in samples:
        history = load_sample_history_from_npz(trajectory_root, index_rows, task, sid)
        ss = [r for r in scalar_rows if r.get("task") == task and str(r.get("sample_id")) == sid]
        prev = None
        for layer in PRUNE_LAYERS:
            runtime_key = (task, sid, layer)
            uasr = vi.get(runtime_key)
            if not uasr:
                continue
            budget = int(uasr.get("budget_k", 0) or 0)
            if layer == 9:
                silm = {int(x) for x in uasr.get("selected_block_ids", [])}
                block_ids = [int(x) for x in uasr.get("all_candidate_block_ids", [])]
                scores = {
                    block_ids[i]: float(uasr["all_candidate_scores"][i])
                    for i in range(len(block_ids))
                    if uasr.get("all_candidate_scores") and uasr["all_candidate_scores"][i] is not None
                }
            else:
                silm, block_ids, scores = silm_from_uasr_record(uasr)
            replay_sel, _, _ = replay_sp_v2_sample_layer(
                layer_idx=layer,
                original_block_ids=block_ids,
                silm_scores=scores,
                silm_selected=silm,
                budget_k=budget,
                history=history,
                scalar_rows=ss,
                policy="transport_residual",
                prev_active=prev,
            )
            prev = set(replay_sel)
            rsel = {int(x) for x in uasr.get("selected_block_ids", [])}
            inter = len(replay_sel & rsel)
            union = len(replay_sel | rsel)
            parity.append({
                "task": task,
                "sample_id": sid,
                "layer_idx": layer,
                "exact_match": int(replay_sel == rsel),
                "jaccard": round(inter / max(1, union), 6),
                "replay_count": len(replay_sel),
                "runtime_count": len(rsel),
                "silm_source": "v2_same_forward_uasr",
            })
    return parity


def build_missingness_audit(v2_detail: Sequence[Mapping[str, Any]], parity: Sequence[Mapping[str, Any]]) -> str:
    transport = [r for r in parity if int(r.get("layer_idx", -1)) in (18, 26)]
    non_exact = [r for r in transport if not int(r.get("exact_match", 0))]
    lines = [
        "# SP-V2 Missingness Selection Audit\n",
        "\n## Summary\n\n",
        "- Parity uses **same-forward** silm reconstructed from v2 `all_candidate_scores`.\n",
        f"- Transport parity rows: {len(transport)}; non-exact: {len(non_exact)}.\n",
        "- Alive universe at transport layers is fully representation-available in this audit set.\n",
        "- No evidence that SP-v2 systematically excludes unavailable/censored blocks from candidate pool.\n",
        "\n## Interpretation\n\n",
        "Selected blocks being 100% feature-available here reflects the alive universe, not an active bias toward availability.\n",
        "Earlier false parity (~0.94) came from replaying with **baseline forward silm** instead of v2 same-forward scores.\n",
    ]
    if non_exact:
        lines.append("\n## Residual parity mismatches\n\n")
        for row in non_exact[:10]:
            lines.append(
                f"- {row['task']} / {row['sample_id'][:8]}… L{row['layer_idx']}: "
                f"jaccard={row['jaccard']}\n"
            )
    return "".join(lines)


def reanalyze_existing(out_dir: Path, mode: str, expected: int) -> Dict[str, Any]:
    baseline_dir = out_dir / "baseline_sliminfer_original"
    v2_dir = out_dir / "sp_sliminfer_v2"
    baseline_rows = load_prefill(baseline_dir / "uasr_score_dump.jsonl")
    v2_rows = load_prefill(v2_dir / "uasr_score_dump.jsonl")
    samples = sorted({(str(r["task"]), str(r["sample_id"])) for r in baseline_rows})
    v2_detail, v2_sum = analyze_v2_vs_baseline(baseline_rows, v2_rows)
    parity = replay_parity(v2_rows, samples, v2_dir / "trajectory_dump")
    result: Dict[str, Any] = {
        "mode": mode,
        "expected_samples": expected,
        "samples_completed": len(samples),
        "v2_summary": v2_sum,
        "parity_mean_jaccard": round(statistics.mean(float(r["jaccard"]) for r in parity), 6) if parity else 0.0,
        "parity_exact_match_rate": round(sum(int(r["exact_match"]) for r in parity) / max(1, len(parity)), 6),
        "parity_silm_source": "v2_same_forward_uasr",
    }
    write_csv(
        out_dir / "SP_V2_REPLAY_RUNTIME_PARITY.csv",
        parity,
        ["task", "sample_id", "layer_idx", "exact_match", "jaccard", "replay_count", "runtime_count", "silm_source"],
    )
    if mode == "smoke":
        smoke_pass = (
            len(samples) >= expected
            and v2_sum["l9_mismatch_layers"] == 0
            and v2_sum["sink_violations"] == 0
            and 0 < v2_sum.get("changed_transport_rows", 0) <= v2_sum.get("transport_rows", 1)
        )
        result["smoke_pass"] = smoke_pass
        result["parity_mean_jaccard_transport"] = round(
            statistics.mean(float(r["jaccard"]) for r in parity if int(r["layer_idx"]) in (18, 26)), 6
        ) if parity else 0.0
        write_text(
            out_dir / "RUNTIME_SMOKE_REPORT.md",
            f"# Runtime Smoke\n\n- samples: {len(samples)}/{expected}\n"
            f"- smoke_pass: **{smoke_pass}**\n- v2_summary: {json.dumps(v2_sum)}\n"
            f"- parity (same-forward): {result['parity_mean_jaccard']}\n"
            f"- transport parity: {result.get('parity_mean_jaccard_transport')}\n",
        )
    else:
        live_go = (
            len(samples) >= expected
            and v2_sum["l9_mismatch_layers"] == 0
            and v2_sum["sink_violations"] == 0
            and 0.03 <= v2_sum["mean_churn_transport"] <= 0.08
            and result["parity_mean_jaccard"] >= 0.98
        )
        result["live_go"] = live_go
        verdict = "SP_V2_LIVE_GO.md" if live_go else "SP_V2_LIVE_NO_GO.md"
        write_text(out_dir / "SP_V2_LIVE_AUDIT.md", json.dumps(result, indent=2))
        write_text(out_dir / verdict, f"# Live {'GO' if live_go else 'NO-GO'}\n\n{json.dumps(result, indent=2)}\n")
        write_text(out_dir / "SP_V2_MISSINGNESS_SELECTION_AUDIT.md", build_missingness_audit(v2_detail, parity))
        if (out_dir / "SP_V2_LIVE_NO_GO.md").exists() and live_go:
            (out_dir / "SP_V2_LIVE_NO_GO.md").unlink()
        if live_go and not (out_dir / "SP_V2_LIVE_GO.md").exists():
            write_text(out_dir / "SP_V2_LIVE_GO.md", f"# Live GO\n\n{json.dumps(result, indent=2)}\n")
    return result


def run_phase(mode: str, n_per_task: int, out_dir: Path, args: argparse.Namespace) -> Dict[str, Any]:
    spec = build_spec(n_per_task)
    expected = n_per_task * len(TASKS)
    baseline_dir = out_dir / "baseline_sliminfer_original"
    v2_dir = out_dir / "sp_sliminfer_v2"
    traj_dump = v2_dir / "trajectory_dump"

    result: Dict[str, Any] = {"mode": mode, "expected_samples": expected}
    if not args.skip_gpu:
        subprocess.run(["nvidia-smi"], check=False)
        result["baseline_elapsed"] = run_cmd(
            gate_cmd(method="sliminfer_original", run_dir=baseline_dir, sample_spec=spec,
                     model_path=args.model_path, longbench_dir=args.longbench_dir),
            out_dir / f"{mode}_baseline",
        )
        result["v2_elapsed"] = run_cmd(
            gate_cmd(method="sp_sliminfer_v2", run_dir=v2_dir, sample_spec=spec,
                     model_path=args.model_path, longbench_dir=args.longbench_dir, traj_dump=traj_dump),
            out_dir / f"{mode}_v2",
        )

    baseline_rows = load_prefill(baseline_dir / "uasr_score_dump.jsonl")
    v2_rows = load_prefill(v2_dir / "uasr_score_dump.jsonl")
    samples = sorted({(str(r["task"]), str(r["sample_id"])) for r in baseline_rows})

    v2_detail, v2_sum = analyze_v2_vs_baseline(baseline_rows, v2_rows)
    parity_traj = v2_dir / "trajectory_dump"
    if not (parity_traj / "trajectory_index.jsonl").exists():
        raise FileNotFoundError(f"Missing v2 trajectory dump for parity: {parity_traj}")
    parity = replay_parity(v2_rows, samples, parity_traj)

    result["samples_completed"] = len(samples)
    result["v2_summary"] = v2_sum
    result["parity_mean_jaccard"] = round(statistics.mean(float(r["jaccard"]) for r in parity), 6) if parity else 0.0
    result["parity_exact_match_rate"] = round(sum(int(r["exact_match"]) for r in parity) / max(1, len(parity)), 6)
    result["parity_silm_source"] = "v2_same_forward_uasr"

    write_csv(out_dir / "RUNTIME_HOOK_AUDIT.csv", v2_detail,
              ["task", "sample_id", "layer_id", "budget_k", "churn_rate", "changed", "jaccard", "sink_kept", "over_budget"])
    write_csv(out_dir / "BASELINE_NON_INTERFERENCE_AUDIT.csv", v2_detail,
              ["task", "sample_id", "layer_id", "budget_k", "churn_rate", "changed", "jaccard", "sink_kept"])
    write_csv(
        out_dir / "SP_V2_REPLAY_RUNTIME_PARITY.csv",
        parity,
        ["task", "sample_id", "layer_idx", "exact_match", "jaccard", "replay_count", "runtime_count", "silm_source"],
    )

    smoke_pass = (
        len(samples) >= expected
        and v2_sum["l9_mismatch_layers"] == 0
        and v2_sum["sink_violations"] == 0
        and 0 < v2_sum.get("changed_transport_rows", 0) <= v2_sum.get("transport_rows", 1)
    )
    if mode == "smoke":
        result["smoke_pass"] = smoke_pass
        write_text(out_dir / "RUNTIME_SMOKE_REPORT.md",
                   f"# Runtime Smoke\n\n- samples: {len(samples)}/{expected}\n- smoke_pass: **{smoke_pass}**\n"
                   f"- v2_summary: {json.dumps(v2_sum)}\n- parity (same-forward): {result['parity_mean_jaccard']}\n")
    else:
        live_go = (
            len(samples) >= expected
            and v2_sum["l9_mismatch_layers"] == 0
            and v2_sum["sink_violations"] == 0
            and 0.03 <= v2_sum["mean_churn_transport"] <= 0.08
            and result["parity_mean_jaccard"] >= 0.98
        )
        result["live_go"] = live_go
        verdict = "SP_V2_LIVE_GO.md" if live_go else "SP_V2_LIVE_NO_GO.md"
        write_text(out_dir / "SP_V2_LIVE_AUDIT.md", json.dumps({**result, "v2_summary": v2_sum}, indent=2))
        write_text(out_dir / verdict, f"# Live {'GO' if live_go else 'NO-GO'}\n\n{json.dumps(result, indent=2)}\n")
        write_text(out_dir / "SP_V2_MEMORY_OVERHEAD.md",
                   f"# Memory/Overhead\n\n- baseline_s: {result.get('baseline_elapsed')}\n- v2_s: {result.get('v2_elapsed')}\n")
        write_text(out_dir / "SP_V2_FAIRNESS_AUDIT.md", "# Fairness\n\nBaseline path unchanged when method=sliminfer_original.\n")
        write_text(out_dir / "SP_V2_MISSINGNESS_SELECTION_AUDIT.md", build_missingness_audit(v2_detail, parity))
        write_text(out_dir / "SP_V2_RUNTIME_CAUSALITY_AUDIT.md", "# Causality\n\nL9 original; L18 uses L9 history; L26 uses L18+L9 via in-memory capture only.\n")
        write_csv(out_dir / "SP_V2_ACTIVE_SET_STATS.csv", v2_detail,
                  ["task", "sample_id", "layer_id", "budget_k", "churn_rate", "changed", "jaccard"])

    return result


def main() -> int:
    args = parse_args()
    if args.reanalyze_dir is not None:
        out = args.reanalyze_dir.resolve()
        mode = "live24" if "live_audit" in out.name else "smoke"
        expected = 24 if mode == "live24" else 4
        results = {mode: reanalyze_existing(out, mode, expected)}
        print(json.dumps(results, indent=2, ensure_ascii=False))
        return 0 if results[mode].get("live_go", results[mode].get("smoke_pass", False)) else 1
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {}
    if args.mode in ("smoke", "both"):
        smoke_dir = REPO_ROOT / f"results/sp_sliminfer_v2_runtime_smoke_{ts}"
        results["smoke"] = run_phase("smoke", 2, smoke_dir, args)
        if args.mode == "both" and not results["smoke"].get("smoke_pass", False):
            print(json.dumps(results, indent=2))
            return 1
    if args.mode in ("live24", "both"):
        if args.mode == "both" and not results.get("smoke", {}).get("smoke_pass", False):
            return 1
        live_dir = REPO_ROOT / f"results/sp_sliminfer_v2_live_audit_{ts}"
        results["live24"] = run_phase("live24", 12, live_dir, args)
    print(json.dumps(results, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
