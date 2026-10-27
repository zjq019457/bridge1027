#!/usr/bin/env python3
"""Stage 4 / G5 efficiency subset (BridgePack-only; fixed small sample set)."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.bridgepack_g1.common import (
    QWEN_PATH,
    TASKS,
    read_jsonl,
    select_fixed_sample_ids,
    utc_now,
    write_json,
    write_text,
)
from tools.bridgepack_g2.run_g2 import LLAMA_MODEL_PATH, g2_safety_gate

BUDGET = 1024
N_PER_TASK = 10
MODEL_PATH = {"qwen": QWEN_PATH, "llama3": LLAMA_MODEL_PATH}
G1 = REPO_ROOT / "results" / "bridgepack_g1_qwen1024_matched_20260727_004934"
G2 = REPO_ROOT / "results" / "bridgepack_g2_llama1024_sourceorder_20260727_023300"
SOURCE_PACK = {
    "qwen": G1 / "full" / "packed_prompts_build" / "packed_prompts" / "same_spans_source_order" / f"budget_{BUDGET}",
    "llama3": G2 / "full" / "packed_prompts_build" / "packed_prompts" / "source_order_bridge_selection" / f"budget_{BUDGET}",
}


def _median(xs: Sequence[float]) -> float:
    return float(statistics.median(xs)) if xs else float("nan")


def measure_packing_cpu(out_dir: Path, model_name: str, ids: Mapping[str, Sequence[str]]) -> List[Dict[str, Any]]:
    from transformers import AutoTokenizer
    from bridgekv.bridgepack_cpu_extensions import selection_only_control
    from tools.bridgepack_g1.common import load_hist_lbp_rows

    tok = AutoTokenizer.from_pretrained(str(MODEL_PATH[model_name]), trust_remote_code=True)
    hist = load_hist_lbp_rows(ids)
    rows = []
    for task in TASKS:
        for sid in ids[task]:
            src = next(r for r in hist[task] if str(r["sample_id"]) == sid)
            t0 = time.perf_counter()
            packed, audit = selection_only_control(
                sid, task, str(src["question"]), str(src["original_context"]), BUDGET, "bridge_path", tokenizer=tok
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            rows.append(
                {
                    "model": model_name,
                    "method": "source_order_bridgepack",
                    "phase": "cpu_packing",
                    "task": task,
                    "sample_id": sid,
                    "latency_ms": elapsed_ms,
                    "packed_tokens": int(packed.packed_token_count),
                    "budget": BUDGET,
                    "over_budget": int(packed.packed_token_count > BUDGET),
                }
            )
    return rows


def measure_qa_gpu(out_dir: Path, model_name: str, ids: Mapping[str, Sequence[str]], n_runs: int = 3) -> List[Dict[str, Any]]:
    import torch
    import tools.run_evidence_text_packing_generation_qwen32 as gen
    from tools.bridgepack_g1.generate import generate_greedy_historical, load_qwen_fp16

    gate = g2_safety_gate(out_dir, f"eff_{model_name}_before_load", phase="before_load")
    if gate["decision"] != "GO":
        write_text(out_dir / "WAITING_FOR_SAFE_GPU.md", json.dumps(gate, indent=2))
        raise RuntimeError(f"WAITING_FOR_SAFE_GPU: {gate['reasons']}")

    model, tokenizer = load_qwen_fp16(str(MODEL_PATH[model_name]), torch)
    g2_safety_gate(out_dir, f"eff_{model_name}_after_load", phase="runtime")
    rows = []
    pack_root = SOURCE_PACK[model_name]
    # warmup
    warm_task = TASKS[0]
    warm = read_jsonl(pack_root / f"{warm_task}.jsonl")[0]
    prompt = str(warm.get("generation_prompt") or gen._generation_prompt_from_row(warm_task, warm))
    generate_greedy_historical(model, tokenizer, torch, prompt, gen._max_new_tokens_for(warm_task, 0))

    for task in TASKS:
        by_id = {str(r["sample_id"]): r for r in read_jsonl(pack_root / f"{task}.jsonl")}
        for sid in ids[task]:
            row = by_id[sid]
            prompt = str(row.get("generation_prompt") or gen._generation_prompt_from_row(task, row))
            max_new = gen._max_new_tokens_for(task, 0)
            latencies = []
            peak = 0.0
            for _ in range(n_runs):
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                pred, ctx_len = generate_greedy_historical(model, tokenizer, torch, prompt, max_new)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    peak = max(peak, torch.cuda.max_memory_allocated() / (1024**2))
                latencies.append((time.perf_counter() - t0) * 1000.0)
            rows.append(
                {
                    "model": model_name,
                    "method": "source_order_bridgepack",
                    "phase": "gpu_qa",
                    "task": task,
                    "sample_id": sid,
                    "latency_ms_median": _median(latencies),
                    "latency_ms_min": min(latencies),
                    "latency_ms_max": max(latencies),
                    "prompt_tokens": int(row.get("model_token_count") or row.get("packed_token_count") or ctx_len),
                    "peak_vram_mib": peak,
                    "n_runs": n_runs,
                }
            )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--stop_after", default="full", choices=["pack_cpu", "full", "finalize"])
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = select_fixed_sample_ids(N_PER_TASK)

    all_pack = []
    all_qa = []
    for model_name in ("qwen", "llama3"):
        print(f"[stage4] CPU packing timing {model_name}", flush=True)
        all_pack.extend(measure_packing_cpu(out_dir, model_name, ids))
        if args.stop_after == "pack_cpu":
            continue
        print(f"[stage4] GPU QA timing {model_name}", flush=True)
        all_qa.extend(measure_qa_gpu(out_dir, model_name, ids))

    with (out_dir / "G5_LATENCY_BREAKDOWN.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = sorted({k for r in (all_pack + all_qa) for k in r.keys()})
        w = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        for r in all_pack + all_qa:
            w.writerow(r)

    mem_rows = [
        {
            "model": r["model"],
            "task": r["task"],
            "sample_id": r["sample_id"],
            "peak_vram_mib": r.get("peak_vram_mib", ""),
            "prompt_tokens": r.get("prompt_tokens", ""),
        }
        for r in all_qa
    ]
    with (out_dir / "G5_MEMORY_REPORT.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["model", "task", "sample_id", "peak_vram_mib", "prompt_tokens"]
        w = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for r in mem_rows:
            w.writerow(r)

    summary_rows = []
    for model_name in ("qwen", "llama3"):
        pack = [r for r in all_pack if r["model"] == model_name]
        qa = [r for r in all_qa if r["model"] == model_name]
        summary_rows.append(
            {
                "model": model_name,
                "n_samples": len(pack),
                "pack_latency_ms_median": _median([r["latency_ms"] for r in pack]),
                "qa_latency_ms_median": _median([float(r["latency_ms_median"]) for r in qa]) if qa else "",
                "peak_vram_mib_max": max((float(r["peak_vram_mib"]) for r in qa), default=0.0) if qa else "",
                "mean_prompt_tokens": (
                    statistics.mean([float(r["prompt_tokens"]) for r in qa]) if qa else ""
                ),
            }
        )
    with (out_dir / "G5_EFFICIENCY_RESULTS.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = list(summary_rows[0].keys())
        w = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)

    write_text(
        out_dir / "G5_EFFICIENCY_SUMMARY.md",
        f"""# G5_EFFICIENCY_SUMMARY

Generated: {utc_now()}

- subset: {N_PER_TASK}/task × 6 tasks × 2 models = {N_PER_TASK*6*2} timed cells
- batch_size: 1
- budget: {BUDGET}
- note: model load excluded from per-sample latency; CUDA sync around timed region
- not KV-cache pruning

## Medians

"""
        + "\n".join(
            f"- {r['model']}: pack={r['pack_latency_ms_median']:.1f} ms, qa={r['qa_latency_ms_median']}, peak_vram={r['peak_vram_mib_max']}"
            for r in summary_rows
        )
        + "\n",
    )
    write_json(out_dir / "efficiency_ids.json", {t: list(ids[t]) for t in TASKS})
    write_text(out_dir / "STAGE_4.DONE", f"Stage 4 complete at {utc_now()}\n")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        out = Path(sys.argv[sys.argv.index("--out_dir") + 1]) if "--out_dir" in sys.argv else Path(".")
        write_text(out / "STAGE4_FATAL_ERROR.md", f"{exc}\n\n{traceback.format_exc()}")
        raise
