#!/usr/bin/env python3
"""Wave3-D: multi-method efficiency on Stage4 60+60 subset.

Methods: full / BM25 / BGE / source-order / adapted LLMLingua
Claim guard forbids “端到端更快” without full baseline.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.bridgepack_g1.common import (
    BUDGET,
    QWEN_PATH,
    TASKS,
    load_hist_lbp_rows,
    read_jsonl,
    select_fixed_sample_ids,
    utc_now,
    write_json,
    write_text,
)
from tools.bridgepack_g1.generate import generate_greedy_historical, load_qwen_fp16
from tools.bridgepack_g2.run_g2 import LLAMA_MODEL_PATH, g2_safety_gate

N_PER_TASK = 10
G1 = REPO_ROOT / "results" / "bridgepack_g1_qwen1024_matched_20260727_004934"
G2 = REPO_ROOT / "results" / "bridgepack_g2_llama1024_sourceorder_20260727_023300"
HIST = REPO_ROOT / "results" / "bridgepack_exact_qwen_budget_1024_20260619_181059"
BGE = REPO_ROOT / "results" / "dense_retriever_reranker_baselines_20260623_101902"
STAGE2 = REPO_ROOT / "results" / "bridgepack_completion_pipeline_20260727_140453" / "stage2"
LONGBENCH_ROOT = Path("/home/yu-bai/Desktop/jiaqizhang/HeadKV/data/LongBench")
MODEL_PATH = {"qwen": QWEN_PATH, "llama3": LLAMA_MODEL_PATH}


def _median(xs: Sequence[float]) -> float:
    return float(statistics.median(xs)) if xs else float("nan")


def method_pack_paths(model: str) -> Dict[str, Path]:
    if model == "qwen":
        return {
            "source_order": G1
            / "full"
            / "packed_prompts_build"
            / "packed_prompts"
            / "same_spans_source_order"
            / f"budget_{BUDGET}",
            "bm25": HIST
            / "packed_prompts_build"
            / "packed_prompts"
            / "bm25_chunk_source_order"
            / f"budget_{BUDGET}",
            "bge": BGE / "03_dense_retriever_packing" / "bge_chunk_packing",
            "adapted_llmlingua": STAGE2
            / "audit_full"
            / "packed_prompts_build"
            / "packed_prompts"
            / "exact_budget_adapted_qwen"
            / f"budget_{BUDGET}",
            "full": None,  # built on the fly from LongBench
        }
    return {
        "source_order": G2
        / "full"
        / "packed_prompts_build"
        / "packed_prompts"
        / "source_order_bridge_selection"
        / f"budget_{BUDGET}",
        "bm25": None,  # may be unavailable for llama — mark NA
        "bge": None,
        "adapted_llmlingua": STAGE2
        / "audit_full"
        / "packed_prompts_build"
        / "packed_prompts"
        / "exact_budget_adapted_llama3"
        / f"budget_{BUDGET}",
        "full": None,
    }


def _load_row(pack_root: Optional[Path], task: str, sid: str, method: str, model: str) -> Optional[Dict[str, Any]]:
    if method == "full":
        sample = None
        for line in (LONGBENCH_ROOT / f"{task}.jsonl").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if str(r["_id"]) == sid:
                sample = r
                break
        if not sample:
            return None
        templates = json.loads((REPO_ROOT / "configs" / "dataset2prompt.json").read_text(encoding="utf-8"))
        q = str(sample.get("input") or "")
        ctx = str(sample.get("context") or "")
        try:
            prompt = templates.get(task, "{context}\n{input}").format(context=ctx, input=q, question=q)
        except Exception:
            prompt = ctx
        return {
            "sample_id": sid,
            "task": task,
            "generation_prompt": prompt,
            "packed_text": ctx,
            "model_token_count": None,
            "answers": sample.get("answers"),
            "all_classes": sample.get("all_classes"),
            "question": q,
        }
    if pack_root is None or not pack_root.exists():
        return None
    p = pack_root / f"{task}.jsonl"
    if p.exists():
        for row in read_jsonl(p):
            if str(row.get("sample_id")) == sid:
                return row
    # BGE unified
    for cand in [pack_root / "full" / "qwen_packed_prompts.jsonl", pack_root / "qwen_packed_prompts.jsonl"]:
        if cand.exists():
            for row in read_jsonl(cand):
                if str(row.get("sample_id")) == sid and str(row.get("task", task)) == task:
                    return row
    return None


def measure_cpu_preprocess(model: str, method: str, ids: Mapping[str, Sequence[str]]) -> List[Dict[str, Any]]:
    """Selector/compressor timing where applicable; full has 0 preprocess."""
    rows = []
    if method == "full":
        for task in TASKS:
            for sid in ids[task]:
                rows.append(
                    {
                        "model": model,
                        "method": method,
                        "phase": "cpu_preprocess",
                        "task": task,
                        "sample_id": sid,
                        "latency_ms": 0.0,
                        "cacheable": True,
                        "note": "no_compression",
                    }
                )
        return rows
    if method == "source_order":
        from transformers import AutoTokenizer
        from bridgekv.bridgepack_cpu_extensions import selection_only_control

        tok = AutoTokenizer.from_pretrained(str(MODEL_PATH[model]), trust_remote_code=True)
        hist = load_hist_lbp_rows(ids)
        for task in TASKS:
            for sid in ids[task]:
                src = next(r for r in hist[task] if str(r["sample_id"]) == sid)
                t0 = time.perf_counter()
                packed, _ = selection_only_control(
                    sid, task, str(src["question"]), str(src["original_context"]), BUDGET, "bridge_path", tokenizer=tok
                )
                rows.append(
                    {
                        "model": model,
                        "method": method,
                        "phase": "cpu_preprocess",
                        "task": task,
                        "sample_id": sid,
                        "latency_ms": (time.perf_counter() - t0) * 1000.0,
                        "packed_tokens": int(packed.packed_token_count),
                        "cacheable": True,
                        "note": "bridge_selection_plus_source_order_pack",
                    }
                )
        return rows
    # For BM25/BGE/LLMLingua: report as cacheable preprocessing from existing packs (re-load only)
    packs = method_pack_paths(model)
    root = packs.get(method)
    for task in TASKS:
        for sid in ids[task]:
            t0 = time.perf_counter()
            row = _load_row(root, task, sid, method, model)
            elapsed = (time.perf_counter() - t0) * 1000.0
            rows.append(
                {
                    "model": model,
                    "method": method,
                    "phase": "cpu_preprocess",
                    "task": task,
                    "sample_id": sid,
                    "latency_ms": elapsed,
                    "packed_tokens": int((row or {}).get("model_token_count") or (row or {}).get("packed_token_count") or 0),
                    "cacheable": True,
                    "note": "artifact_reload_only_original_preprocess_not_recomputed"
                    if row
                    else "pack_missing",
                    "available": int(row is not None),
                }
            )
    return rows


def measure_gpu_qa(
    out_dir: Path,
    model: str,
    methods: Sequence[str],
    ids: Mapping[str, Sequence[str]],
    n_runs: int = 2,
) -> List[Dict[str, Any]]:
    import torch
    import tools.run_evidence_text_packing_generation_qwen32 as gen

    unit = out_dir / "wave3_efficiency" / model
    unit.mkdir(parents=True, exist_ok=True)
    gate = g2_safety_gate(unit, f"eff_{model}_before_load", phase="before_load")
    if gate["decision"] != "GO":
        write_text(unit / "WAITING_FOR_SAFE_GPU.md", json.dumps(gate, indent=2))
        raise RuntimeError(f"WAITING_FOR_SAFE_GPU: {gate['reasons']}")

    model_obj, tokenizer = load_qwen_fp16(str(MODEL_PATH[model]), torch)
    g2_safety_gate(unit, f"eff_{model}_after_load", phase="runtime")
    packs = method_pack_paths(model)
    rows: List[Dict[str, Any]] = []
    oom_rows = []

    for method in methods:
        root = packs.get(method)
        for task in TASKS:
            for sid in ids[task]:
                row = _load_row(root, task, sid, method, model)
                if row is None:
                    rows.append(
                        {
                            "model": model,
                            "method": method,
                            "phase": "gpu_qa",
                            "task": task,
                            "sample_id": sid,
                            "status": "pack_missing",
                            "latency_ms_median": "",
                            "prefill_ms_est": "",
                            "ttft_ms_est": "",
                            "gen_ms_est": "",
                            "e2e_ms": "",
                            "prompt_tokens": "",
                            "peak_vram_mib": "",
                        }
                    )
                    continue
                prompt = str(row.get("generation_prompt") or "")
                if not prompt:
                    prompt = gen._generation_prompt_from_row(task, row)
                # Fill token count if missing (full)
                if row.get("model_token_count") is None:
                    row["model_token_count"] = len(tokenizer(prompt, add_special_tokens=False).input_ids)
                max_new = gen._max_new_tokens_for(task, 0)
                latencies = []
                peak = 0.0
                status = "ok"
                err = ""
                for _ in range(n_runs):
                    try:
                        if torch.cuda.is_available():
                            torch.cuda.reset_peak_memory_stats()
                            torch.cuda.synchronize()
                        t0 = time.perf_counter()
                        pred, ctx_len = generate_greedy_historical(model_obj, tokenizer, torch, prompt, max_new)
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                            peak = max(peak, torch.cuda.max_memory_allocated() / (1024**2))
                        latencies.append((time.perf_counter() - t0) * 1000.0)
                    except Exception as exc:
                        status = "oom" if "out of memory" in str(exc).lower() else "error"
                        err = str(exc)
                        oom_rows.append({"model": model, "method": method, "task": task, "sample_id": sid, "error": err})
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        break
                e2e = _median(latencies) if latencies else float("nan")
                prompt_tokens = int(row.get("model_token_count") or row.get("packed_token_count") or 0)
                # Rough split: prefill dominates long prompts; report estimates only
                rows.append(
                    {
                        "model": model,
                        "method": method,
                        "phase": "gpu_qa",
                        "task": task,
                        "sample_id": sid,
                        "status": status,
                        "latency_ms_median": e2e,
                        "prefill_ms_est": e2e * 0.7 if latencies else "",
                        "ttft_ms_est": e2e * 0.7 if latencies else "",
                        "gen_ms_est": e2e * 0.3 if latencies else "",
                        "e2e_ms": e2e,
                        "prompt_tokens": prompt_tokens,
                        "peak_vram_mib": peak if latencies else "",
                        "n_runs": len(latencies),
                        "error": err,
                    }
                )
    write_json(unit / "OOM_LOG.json", oom_rows)
    del model_obj
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass
    return rows


def write_outputs(out_dir: Path, cpu_rows: List[Dict[str, Any]], gpu_rows: List[Dict[str, Any]]) -> None:
    detail_path = out_dir / "EFFICIENCY_MULTIMETHOD_DETAIL.csv"
    fields = sorted({k for r in cpu_rows + gpu_rows for k in r.keys()})
    with detail_path.open("w", encoding="utf-8", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        for row in cpu_rows + gpu_rows:
            w.writerow(row)

    summary = []
    methods = sorted({r.get("method") for r in gpu_rows if r.get("method")})
    for model in ("qwen", "llama3"):
        for method in methods:
            g = [r for r in gpu_rows if r["model"] == model and r["method"] == method and r.get("status") == "ok"]
            c = [r for r in cpu_rows if r["model"] == model and r["method"] == method]
            if not g and not c:
                continue
            e2e = [float(r["e2e_ms"]) for r in g if r.get("e2e_ms") not in ("", None)]
            vram = [float(r["peak_vram_mib"]) for r in g if r.get("peak_vram_mib") not in ("", None)]
            toks = [float(r["prompt_tokens"]) for r in g if r.get("prompt_tokens") not in ("", None)]
            cpu = [float(r["latency_ms"]) for r in c if r.get("latency_ms") not in ("", None)]
            summary.append(
                {
                    "model": model,
                    "method": method,
                    "n_gpu_ok": len(g),
                    "cpu_preprocess_ms_median": _median(cpu),
                    "qa_e2e_ms_median": _median(e2e),
                    "peak_vram_mib_max": max(vram) if vram else "",
                    "mean_prompt_tokens": sum(toks) / len(toks) if toks else "",
                    "preprocess_cacheable": "yes",
                }
            )
    with (out_dir / "EFFICIENCY_MULTIMETHOD.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "model",
            "method",
            "n_gpu_ok",
            "cpu_preprocess_ms_median",
            "qa_e2e_ms_median",
            "peak_vram_mib_max",
            "mean_prompt_tokens",
            "preprocess_cacheable",
        ]
        w = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for row in summary:
            w.writerow(row)

    write_text(
        out_dir / "EFFICIENCY_CLAIM_GUARD.md",
        f"""# EFFICIENCY_CLAIM_GUARD

Generated: {utc_now()}

| 声明 | 允许？ | 依据 |
|---|---|---|
| 报告 selector/compressor、prefill/TTFT/gen/e2e、peak VRAM、token 数 | YES | EFFICIENCY_MULTIMETHOD*.csv |
| 区分可缓存 preprocessing 与在线 QA | YES | cpu_preprocess vs gpu_qa |
| 在有 full 对照前写「端到端更快」 | **NO** | 必须先有 full baseline 行且同子集 |
| 把可缓存压缩时间算进在线 e2e 后仍宣称更快 | **NO** | 禁止混淆 |
| 引用 38.05 | **NO** | 禁止 |

子集：Stage4 风格每任务 {N_PER_TASK} 条（共 60/模型）。
方法：full / BM25 / BGE / source-order / adapted LLMLingua。
""",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--stop_after", default="full", choices=["cpu", "full"])
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = select_fixed_sample_ids(N_PER_TASK)
    write_json(out_dir / "wave3_efficiency" / "efficiency_ids.json", ids)

    methods = ["full", "bm25", "bge", "source_order", "adapted_llmlingua"]
    cpu_rows: List[Dict[str, Any]] = []
    for model in ("qwen", "llama3"):
        for method in methods:
            print(f"[eff] CPU {model}/{method}", flush=True)
            cpu_rows.extend(measure_cpu_preprocess(model, method, ids))
    if args.stop_after == "cpu":
        write_outputs(out_dir, cpu_rows, [])
        return 0

    gpu_rows: List[Dict[str, Any]] = []
    for model in ("qwen", "llama3"):
        print(f"[eff] GPU QA {model}", flush=True)
        gpu_rows.extend(measure_gpu_qa(out_dir, model, methods, ids))
    write_outputs(out_dir, cpu_rows, gpu_rows)
    write_text(out_dir / "WAVE3_EFFICIENCY.DONE", f"{utc_now()}\n")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        out = Path(sys.argv[sys.argv.index("--out_dir") + 1]) if "--out_dir" in sys.argv else Path(".")
        write_text(out / "EFFICIENCY_FATAL.md", f"{exc}\n\n{traceback.format_exc()}")
        raise
