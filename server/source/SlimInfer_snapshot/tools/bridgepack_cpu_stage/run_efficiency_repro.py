#!/usr/bin/env python3
"""CPU efficiency measurement + reproducibility checklist (GPU metrics stubs only)."""
from __future__ import annotations

import json
import os
import resource
import time
import tracemalloc
from pathlib import Path
from typing import Any, Dict, List

from bridgekv.evidence_text_packing import pack_evidence

from .common import (
    OUT_ROOT,
    REPO_ROOT,
    ensure_cpu_only,
    load_longbench_task,
    read_json,
    utc_now,
    write_json,
    write_text,
)


def _rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KB
    return float(usage) / 1024.0


def run_efficiency_and_repro(n_time: int = 20) -> Dict[str, Any]:
    ensure_cpu_only()
    samples = []
    for task in ["hotpotqa", "2wikimqa", "musique"]:
        samples.extend(load_longbench_task(task)[: max(1, n_time // 3)])
    samples = samples[:n_time]

    tracemalloc.start()
    times = []
    stage_times = {"segmentation_scoring_path_repair_pack": []}
    t0_all = time.perf_counter()
    for sample in samples:
        sid = str(sample["_id"])
        task = str(sample.get("dataset") or "hotpotqa")
        # infer
        query = str(sample.get("input") or "")
        context = str(sample.get("context") or "")
        t0 = time.perf_counter()
        pack_evidence(sid, "hotpotqa", query, context, "lexical_bridgepack_path_order", 512)
        dt = time.perf_counter() - t0
        times.append(dt)
        stage_times["segmentation_scoring_path_repair_pack"].append(dt)
    total = time.perf_counter() - t0_all
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    import numpy as np

    arr = np.array(times, dtype=np.float64)
    # Extrapolate 1150
    mean_t = float(arr.mean()) if len(arr) else float("nan")
    report = {
        "created_at": utc_now(),
        "n_measured": len(times),
        "variant": "lexical_bridgepack_path_order",
        "budget_regex": 512,
        "cpu_time_sec": {
            "mean": mean_t,
            "p50": float(np.percentile(arr, 50)) if len(arr) else None,
            "p95": float(np.percentile(arr, 95)) if len(arr) else None,
            "total_measured": total,
            "extrapolated_1150_sec": mean_t * 1150,
            "extrapolated_1150_min": mean_t * 1150 / 60.0,
        },
        "peak_tracemalloc_mb": peak / (1024 * 1024),
        "peak_rss_mb": _rss_mb(),
        "staged": {
            "note": "Fine-grained stage timers require instrumentation hooks; currently end-to-end pack_evidence timing only.",
            "pack_evidence_end_to_end_sec": stage_times,
        },
        "cached_bge_organization_cost": "NOT_MEASURED_NO_SAFE_CACHE_WIRED",
        "gpu_metrics": {
            "bge_encoder_time": "GPU_PENDING",
            "gpu_peak_memory": "GPU_PENDING",
            "prefill_time": "GPU_PENDING",
            "ttft": "GPU_PENDING",
            "e2e_latency": "GPU_PENDING",
            "generation_peak_vram": "GPU_PENDING",
        },
    }
    write_json(OUT_ROOT / "efficiency" / "cpu_efficiency.json", report)
    md = [
        "# CPU_EFFICIENCY_REPORT",
        "",
        f"Generated: {utc_now()}",
        "",
        f"- Measured samples: {len(times)}",
        f"- Lexical-BridgePack pack mean/P50/P95 seconds: {report['cpu_time_sec']['mean']:.4f} / {report['cpu_time_sec']['p50']:.4f} / {report['cpu_time_sec']['p95']:.4f}",
        f"- Extrapolated 1150-sample preprocessing: {report['cpu_time_sec']['extrapolated_1150_min']:.2f} min",
        f"- Peak RSS≈{report['peak_rss_mb']:.1f} MB; tracemalloc peak≈{report['peak_tracemalloc_mb']:.1f} MB",
        "- GPU metrics: not run (stubs in GPU_EFFICIENCY_RUNNER_STATUS.md)",
        "",
    ]
    write_text(OUT_ROOT / "efficiency" / "CPU_EFFICIENCY_REPORT.md", "\n".join(md))
    write_text(OUT_ROOT / "CPU_EFFICIENCY_REPORT.md", "\n".join(md))

    # Reproducibility from configs
    cfg_paths = [
        REPO_ROOT / "results/bridgepack_exact_qwen_budget_1024_20260619_181059/config.json",
        REPO_ROOT / "results/bridgepack_second_model_exact_gate_mistral7b_instruct_v03_20260620_055432/config.json",
        REPO_ROOT / "configs/dataset2prompt.json",
        REPO_ROOT / "configs/dataset2maxlen.json",
        REPO_ROOT / "configs/model2path.json",
    ]
    found = {}
    missing = []
    for p in cfg_paths:
        if p.exists():
            found[str(p.relative_to(REPO_ROOT))] = True
        else:
            missing.append(str(p))

    qwen_cfg = read_json(cfg_paths[0]) if cfg_paths[0].exists() else {}
    prompts = read_json(REPO_ROOT / "configs/dataset2prompt.json")
    maxlen = read_json(REPO_ROOT / "configs/dataset2maxlen.json")

    checklist = f"""# REPRODUCIBILITY_CHECKLIST

Generated: {utc_now()}

## Checkpoints
- Qwen2.5-7B-Instruct: `{qwen_cfg.get('model_name_or_path', 'SEE config')}`
- Mistral-7B-Instruct-v0.3: `mistralai/Mistral-7B-Instruct-v0.3` (second-model gate config)
- Llama-3-8B-Instruct: see llama3 result run_status / packed configs under `results/llama3_*`
- Llama-2-7b-chat-hf: see `results/llama2_7b_chat_exact_budget_full_experiment_20260625_003442`
- BGE: typically `BAAI/bge-small-en-v1.5` in dense baseline scripts (verify per-run config); pooling/normalize/query prompts must be copied from that run's config before paper claims.

## Data
- LongBench path used: `/home/yu-bai/Desktop/jiaqizhang/HeadKV/data/LongBench`
- Split: LongBench official jsonl task files (no supporting_facts fields in this dump)
- Tasks: qasper, multifieldqa_en, hotpotqa, 2wikimqa, musique, narrativeqa (1150 samples)

## Prompt / decoding
- Prompt wrappers: `configs/dataset2prompt.json`
- max_new_tokens: `configs/dataset2maxlen.json` => {json.dumps(maxlen, ensure_ascii=False)}
- Official Qwen exact run dtype: `{qwen_cfg.get('dtype')}`
- temperature/top-p: not always recorded in packed config (`max_new_tokens` field in exact config was `{qwen_cfg.get('max_new_tokens')}` — treat generation hyperparameters as **must verify from generation runner logs**)

## Scoring
- `longbench_eval.py` dataset2metric

## Isolation
- BridgePack CPU stage outputs only under `{OUT_ROOT.relative_to(REPO_ROOT)}`
- Must not disturb other-paper GPU process
"""
    write_text(OUT_ROOT / "efficiency" / "REPRODUCIBILITY_CHECKLIST.md", checklist)
    write_text(OUT_ROOT / "REPRODUCIBILITY_CHECKLIST.md", checklist)

    missing_fields = f"""# MISSING_REPRODUCIBILITY_FIELDS

Generated: {utc_now()}

## Missing or incomplete (do not invent)

- Unified temperature / top-p / do_sample recorded in every model-budget cell config
- Exact BGE checkpoint revision + pooling + query/document prompt templates + normalize flag in one canonical file
- Hardware SKU / CUDA / drivers / transformers / torch versions pinned per main result root
- Hyperparameter selection source (dev split protocol) for neighbor=32, span=64, stride=32, bridge weights
- Fine-grained packing stage timers in historical exact runs (only aggregate TODO previously)
- Full-context reference completeness / OOM failure matrix for all models
- Llama-3 / Llama-2 absolute local snapshot paths in a single registry (some runs use HF ids only)

## Present

- Qwen snapshot path in official 1024 config
- dataset2prompt / dataset2maxlen
- Fairness audit proving 0 over-budget for 60 cells
- LongBench task files path
"""
    write_text(OUT_ROOT / "efficiency" / "MISSING_REPRODUCIBILITY_FIELDS.md", missing_fields)
    write_text(OUT_ROOT / "MISSING_REPRODUCIBILITY_FIELDS.md", missing_fields)

    gpu_eff = f"""# GPU_EFFICIENCY_RUNNER_STATUS

Generated: {utc_now()}

Status: **GPU_PENDING** (interface only; not executed)

Required measurements on identical hardware/samples/gen config:

| Method | selector/encoder | prefill | TTFT | e2e latency | GPU peak mem | CPU peak mem | input tokens | gen tokens |
|---|---|---|---|---|---|---|---|---|
| full context | GPU_PENDING | GPU_PENDING | GPU_PENDING | GPU_PENDING | GPU_PENDING | GPU_PENDING | GPU_PENDING | GPU_PENDING |
| BM25 | GPU_PENDING | GPU_PENDING | GPU_PENDING | GPU_PENDING | GPU_PENDING | GPU_PENDING | GPU_PENDING | GPU_PENDING |
| BGE | GPU_PENDING | GPU_PENDING | GPU_PENDING | GPU_PENDING | GPU_PENDING | GPU_PENDING | GPU_PENDING | GPU_PENDING |
| Lexical-BridgePack | GPU_PENDING | GPU_PENDING | GPU_PENDING | GPU_PENDING | GPU_PENDING | GPU_PENDING | GPU_PENDING | GPU_PENDING |
| BGE Recall + BridgePack | GPU_PENDING | GPU_PENDING | GPU_PENDING | GPU_PENDING | GPU_PENDING | GPU_PENDING | GPU_PENDING | GPU_PENDING |

Claim constraint: may discuss input-token reduction / prefill benefit; must NOT call this KV-cache pruning.
"""
    write_text(OUT_ROOT / "efficiency" / "GPU_EFFICIENCY_RUNNER_STATUS.md", gpu_eff)
    write_text(OUT_ROOT / "GPU_EFFICIENCY_RUNNER_STATUS.md", gpu_eff)
    return report


if __name__ == "__main__":
    print(run_efficiency_and_repro())
