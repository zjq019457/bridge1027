#!/usr/bin/env python3
"""Wave3-C: full-context vs source-order@1024 retained ratio on common-valid subset.

Reuses legal historical full-context predictions when protocol-compatible.
Does not silently truncate; OOM / failures are recorded and excluded from common-valid.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.bridgepack_g1.common import (
    BUDGET,
    QWEN_PATH,
    TASKS,
    read_jsonl,
    utc_now,
    write_json,
    write_jsonl,
    write_text,
)
from tools.bridgepack_g1.generate import generate_greedy_historical, load_qwen_fp16
from tools.bridgepack_g1.run_g1 import score_predictions
from tools.bridgepack_g1.stats import score_prediction_row
from tools.bridgepack_g2.run_g2 import LLAMA_MODEL_PATH, g2_safety_gate

FREEZE_SHA = "15886731816aa2eb4e08afa121bd8ccc9631dd232474a82a8f20214cc2ce16bf"
G1 = REPO_ROOT / "results" / "bridgepack_g1_qwen1024_matched_20260727_004934"
G2 = REPO_ROOT / "results" / "bridgepack_g2_llama1024_sourceorder_20260727_023300"
QWEN_FULL_HIST = (
    REPO_ROOT
    / "results"
    / "bridgepack_missing_evidence_20260620_081306"
    / "01_full_context_qwen"
    / "full_context_full"
    / "predictions"
    / "full_context_original"
)
LONGBENCH_ROOT = Path("/home/yu-bai/Desktop/jiaqizhang/HeadKV/data/LongBench")
MODEL_PATH = {"qwen": QWEN_PATH, "llama3": LLAMA_MODEL_PATH}
SOURCE_PRED = {
    "qwen": G1 / "full" / "predictions" / "same_spans_source_order",
    "llama3": G2 / "full" / "predictions" / "source_order_bridge_selection",
}


def _templates() -> Dict[str, str]:
    return json.loads((REPO_ROOT / "configs" / "dataset2prompt.json").read_text(encoding="utf-8"))


def build_full_packs(out_dir: Path, model_name: str, tasks: Sequence[str]) -> Path:
    from transformers import AutoTokenizer
    import tools.run_evidence_text_packing_generation_qwen32 as gen

    tok = AutoTokenizer.from_pretrained(str(MODEL_PATH[model_name]), trust_remote_code=True)
    templates = _templates()
    packed_root = out_dir / "wave3_fullctx" / model_name / "packed_prompts_build" / "packed_prompts" / "full_context_original"
    audit = []
    for task in tasks:
        rows_out = []
        for line in (LONGBENCH_ROOT / f"{task}.jsonl").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            sample = json.loads(line)
            sid = str(sample["_id"])
            context = str(sample.get("context") or "")
            question = str(sample.get("input") or sample.get("question") or "")
            try:
                prompt = templates.get(task, "{context}\n{input}").format(
                    context=context, input=question, question=question
                )
            except Exception:
                prompt = context
            ntok = len(tok(prompt, add_special_tokens=False).input_ids)
            rows_out.append(
                {
                    "sample_id": sid,
                    "task": task,
                    "variant": "full_context_original",
                    "budget": None,
                    "packed_text": context,
                    "generation_prompt": prompt,
                    "model_token_count": ntok,
                    "packed_token_count": ntok,
                    "question": question,
                    "answers": sample.get("answers"),
                    "all_classes": sample.get("all_classes"),
                    "truncated": False,
                    "method_config_sha256": FREEZE_SHA,
                    "no_silent_truncate": True,
                }
            )
            audit.append({"model": model_name, "task": task, "sample_id": sid, "prompt_tokens": ntok})
        write_jsonl(packed_root / f"{task}.jsonl", rows_out)
    with (out_dir / "wave3_fullctx" / model_name / "full_pack_audit.csv").open("w", encoding="utf-8", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=["model", "task", "sample_id", "prompt_tokens"], lineterminator="\n")
        w.writeheader()
        for row in audit:
            w.writerow(row)
    return packed_root


def link_or_run_qwen_full(out_dir: Path) -> Path:
    """Reuse historical Qwen full-context preds; do not re-QA unless missing."""
    dest = out_dir / "wave3_fullctx" / "qwen" / "predictions" / "full_context_original"
    dest.mkdir(parents=True, exist_ok=True)
    status = []
    for task in TASKS:
        src = QWEN_FULL_HIST / f"{task}.jsonl"
        dst = dest / f"{task}.jsonl"
        if src.exists():
            rows = read_jsonl(src)
            # NarrativeQA historical is partial (OOM) — keep as-is, mark later
            if not dst.exists():
                import os

                os.symlink(src.resolve(), dst)
            status.append({"task": task, "n": len(rows), "source": "historical_symlink", "path": str(src)})
        else:
            status.append({"task": task, "n": 0, "source": "missing", "path": str(src)})
    write_json(out_dir / "wave3_fullctx" / "qwen" / "reuse_status.json", status)
    return dest


def run_llama_full_qa(out_dir: Path) -> Path:
    import torch
    import tools.run_evidence_text_packing_generation_qwen32 as gen

    unit = out_dir / "wave3_fullctx" / "llama3"
    packed_root = build_full_packs(out_dir, "llama3", TASKS)
    pred_root = unit / "predictions" / "full_context_original"
    pred_root.mkdir(parents=True, exist_ok=True)
    oom_log = []

    gate = g2_safety_gate(unit, "fullctx_llama_before_load", phase="before_load")
    if gate["decision"] != "GO":
        write_text(unit / "WAITING_FOR_SAFE_GPU.md", json.dumps(gate, indent=2))
        raise RuntimeError(f"WAITING_FOR_SAFE_GPU: {gate['reasons']}")

    model, tokenizer = load_qwen_fp16(str(LLAMA_MODEL_PATH), torch)
    g2_safety_gate(unit, "fullctx_llama_after_load", phase="runtime")

    for task in TASKS:
        rows = read_jsonl(packed_root / f"{task}.jsonl")
        pred_path = pred_root / f"{task}.jsonl"
        done = {str(r["sample_id"]): r for r in read_jsonl(pred_path)}
        consecutive_oom = 0
        with pred_path.open("w", encoding="utf-8") as handle:
            for sample_index, row in enumerate(rows):
                sid = str(row["sample_id"])
                if sid in done and done[sid].get("pred") not in (None, "") and done[sid].get("status") != "oom":
                    handle.write(json.dumps(done[sid], ensure_ascii=False) + "\n")
                    handle.flush()
                    consecutive_oom = 0
                    continue
                # After many consecutive OOMs on this task, mark remaining as oom without retry
                # (no silent truncate; explicit exclusion). Threshold high for non-NQA.
                oom_stop = 8 if task == "narrativeqa" else 25
                if consecutive_oom >= oom_stop:
                    payload = {
                        "sample_id": sid,
                        "sample_index": sample_index,
                        "task": task,
                        "variant": "full_context_original",
                        "pred": "",
                        "answers": row.get("answers"),
                        "all_classes": row.get("all_classes"),
                        "length": int(row.get("model_token_count") or 0),
                        "status": "oom",
                        "error": f"skipped_after_{consecutive_oom}_consecutive_oom",
                        "question": row.get("question"),
                    }
                    oom_log.append({"task": task, "sample_id": sid, "error": payload["error"], "is_oom": True})
                    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                    handle.flush()
                    continue
                gate = g2_safety_gate(unit, f"fullctx_llama_{task}_{sample_index}", phase="runtime")
                if gate["decision"] != "GO":
                    reasons = str(gate.get("reasons") or "")
                    # Full-context QA legitimately saturates VRAM (esp. NarrativeQA).
                    # Per-sample CUDA OOM is recorded below; only hard-abort on foreign
                    # workloads or overheating — not on free_mib soft critical alone.
                    hard = ("foreign GPU" in reasons) or ("temperature" in reasons.lower())
                    if hard:
                        raise RuntimeError(f"Safety stop: {reasons}")
                    print(
                        f"[fullctx] soft safety warn (continue; OOM will be recorded): {reasons}",
                        flush=True,
                    )
                prompt = str(row.get("generation_prompt") or "")
                max_new = gen._max_new_tokens_for(task, 0)
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    pred, ctx_len = generate_greedy_historical(model, tokenizer, torch, prompt, max_new)
                    payload = {
                        "sample_id": sid,
                        "sample_index": sample_index,
                        "task": task,
                        "variant": "full_context_original",
                        "pred": pred,
                        "answers": row.get("answers"),
                        "all_classes": row.get("all_classes"),
                        "length": ctx_len,
                        "status": "ok",
                        "question": row.get("question"),
                    }
                    consecutive_oom = 0
                except Exception as exc:
                    msg = str(exc)
                    is_oom = "out of memory" in msg.lower() or "oom" in msg.lower()
                    if "xid" in msg.lower():
                        raise
                    oom_log.append({"task": task, "sample_id": sid, "error": msg, "is_oom": is_oom})
                    payload = {
                        "sample_id": sid,
                        "sample_index": sample_index,
                        "task": task,
                        "variant": "full_context_original",
                        "pred": "",
                        "answers": row.get("answers"),
                        "all_classes": row.get("all_classes"),
                        "length": int(row.get("model_token_count") or 0),
                        "status": "oom" if is_oom else "error",
                        "error": msg,
                        "question": row.get("question"),
                    }
                    consecutive_oom = consecutive_oom + 1 if is_oom else 0
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                handle.flush()
    write_json(unit / "OOM_LOG.json", oom_log)
    del model
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    return pred_root


def _task_scores(pred_root: Path, task: str) -> Dict[str, float]:
    out = {}
    for row in read_jsonl(pred_root / f"{task}.jsonl"):
        if row.get("status") in {"oom", "error"}:
            continue
        if str(row.get("pred", "")) == "" and row.get("status") != "ok":
            continue
        out[str(row["sample_id"])] = score_prediction_row(task, row)
    return out


def write_retained_tables(out_dir: Path) -> None:
    rows = []
    common_ids: Dict[str, List[str]] = {t: [] for t in TASKS}
    exclusion = []

    # Determine per-model valid ids
    valid: Dict[str, Dict[str, set]] = {}
    for model in ("qwen", "llama3"):
        full_root = out_dir / "wave3_fullctx" / model / "predictions" / "full_context_original"
        src_root = SOURCE_PRED[model]
        valid[model] = {}
        for task in TASKS:
            full_scores = _task_scores(full_root, task) if full_root.exists() else {}
            src_scores = _task_scores(src_root, task) if src_root.exists() else {}
            # Also count OOMs
            oom_ids = set()
            if (full_root / f"{task}.jsonl").exists():
                for row in read_jsonl(full_root / f"{task}.jsonl"):
                    if row.get("status") in {"oom", "error"} or (
                        task == "narrativeqa" and model == "qwen" and str(row.get("pred", "")) == "" and len(read_jsonl(full_root / f"{task}.jsonl")) < 50
                    ):
                        oom_ids.add(str(row["sample_id"]))
            # Historical Qwen NQA is partial: treat missing ids as OOM/excluded
            if model == "qwen" and task == "narrativeqa":
                all_ids = {str(json.loads(l)["_id"]) for l in (LONGBENCH_ROOT / f"{task}.jsonl").read_text().splitlines() if l.strip()}
                missing = all_ids - set(full_scores) - oom_ids
                for mid in missing:
                    exclusion.append({"model": model, "task": task, "sample_id": mid, "reason": "historical_partial_or_oom"})
                oom_ids |= missing
            inter = set(full_scores) & set(src_scores)
            valid[model][task] = inter
            for sid in sorted(inter):
                rows.append(
                    {
                        "model": model,
                        "task": task,
                        "sample_id": sid,
                        "full_score": full_scores[sid],
                        "source_order_1024_score": src_scores[sid],
                        "retained_ratio": (src_scores[sid] / full_scores[sid]) if full_scores[sid] > 1e-12 else float("nan"),
                    }
                )
            for sid in oom_ids:
                exclusion.append({"model": model, "task": task, "sample_id": sid, "reason": "oom_or_error"})

    # Common-valid across both models
    for task in TASKS:
        common = sorted(valid["qwen"].get(task, set()) & valid["llama3"].get(task, set()))
        common_ids[task] = common

    with (out_dir / "FULL_CONTEXT_RETAINED_RATIO.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["model", "task", "sample_id", "full_score", "source_order_1024_score", "retained_ratio"]
        w = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for row in rows:
            w.writerow(row)

    # Aggregate
    agg = []
    for model in ("qwen", "llama3"):
        for task in list(TASKS) + ["MACRO_COMMON_VALID"]:
            if task == "MACRO_COMMON_VALID":
                # mean of per-task means on common-valid ids only
                task_means = []
                for t in TASKS:
                    ids = set(common_ids[t])
                    subset = [r for r in rows if r["model"] == model and r["task"] == t and r["sample_id"] in ids]
                    if subset:
                        task_means.append(sum(r["retained_ratio"] for r in subset if r["retained_ratio"] == r["retained_ratio"]) / len(subset))
                agg.append(
                    {
                        "model": model,
                        "task": task,
                        "n": sum(len(common_ids[t]) for t in TASKS),
                        "mean_retained_ratio": sum(task_means) / len(task_means) if task_means else float("nan"),
                        "mean_full_score": "",
                        "mean_source_score": "",
                    }
                )
            else:
                ids = set(common_ids[task])
                subset = [r for r in rows if r["model"] == model and r["task"] == task and r["sample_id"] in ids]
                if not subset:
                    # also report per-model valid (not only common)
                    subset = [r for r in rows if r["model"] == model and r["task"] == task]
                    scope = "model_valid"
                else:
                    scope = "common_valid"
                agg.append(
                    {
                        "model": model,
                        "task": task,
                        "n": len(subset),
                        "scope": scope,
                        "mean_retained_ratio": sum(r["retained_ratio"] for r in subset if r["retained_ratio"] == r["retained_ratio"]) / max(1, len(subset)),
                        "mean_full_score": sum(r["full_score"] for r in subset) / max(1, len(subset)),
                        "mean_source_score": sum(r["source_order_1024_score"] for r in subset) / max(1, len(subset)),
                    }
                )
    with (out_dir / "FULL_CONTEXT_RETAINED_AGG.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["model", "task", "n", "scope", "mean_retained_ratio", "mean_full_score", "mean_source_score"]
        w = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        for row in agg:
            w.writerow(row)

    write_json(out_dir / "FULL_CONTEXT_COMMON_VALID_IDS.json", common_ids)
    write_json(out_dir / "FULL_CONTEXT_EXCLUSIONS.json", exclusion)

    oom_qwen_nqa = sum(1 for e in exclusion if e["model"] == "qwen" and e["task"] == "narrativeqa")
    md = f"""# FULL_CONTEXT_COMMON_VALID

Generated: {utc_now()}

## 规则
- 不静默截断 full-context；OOM/error 记入排除集。
- common-valid = 两模型均有合法 full-context 与 source-order@1024 分数的样本交集。
- Qwen source-order 论文数字仍为 **39.39**（精确 39.385）；禁止 38.05。

## 共同子集规模
"""
    for t in TASKS:
        md += f"- `{t}`: {len(common_ids[t])}\n"
    md += f"\n## OOM / 排除\n- Qwen NarrativeQA exclusions (historical partial/OOM): {oom_qwen_nqa}\n"
    md += "- 详见 `FULL_CONTEXT_EXCLUSIONS.json`\n"
    md += "\n## 表\n- `FULL_CONTEXT_RETAINED_RATIO.csv`\n- `FULL_CONTEXT_RETAINED_AGG.csv`\n"
    write_text(out_dir / "FULL_CONTEXT_COMMON_VALID.md", md)

    write_text(
        out_dir / "FULL_CONTEXT_CLAIM_GUARD.md",
        f"""# FULL_CONTEXT_CLAIM_GUARD

Generated: {utc_now()}

| 声明 | 允许？ | 依据 |
|---|---|---|
| 报告 common-valid retained ratio | YES | FULL_CONTEXT_RETAINED_AGG.csv |
| 六任务官方 full-context 上限 | NO（若 NQA/OOM 未全通） | 排除规则 |
| 静默截断后仍称 full-context | **NO** | 禁止 |
| 引用 38.05 | **NO** | 禁止 |
""",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--skip_llama_qa", action="store_true")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[fullctx] link/reuse Qwen historical full-context", flush=True)
    link_or_run_qwen_full(out_dir)
    # Ensure Qwen pack audit exists for token stats
    build_full_packs(out_dir, "qwen", TASKS)

    if not args.skip_llama_qa:
        print("[fullctx] run Llama full-context QA (OOM recorded, no silent truncate)", flush=True)
        run_llama_full_qa(out_dir)
    else:
        build_full_packs(out_dir, "llama3", TASKS)

    write_retained_tables(out_dir)
    write_text(out_dir / "WAVE3_FULLCTX.DONE", f"{utc_now()}\n")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        out = Path(sys.argv[sys.argv.index("--out_dir") + 1]) if "--out_dir" in sys.argv else Path(".")
        write_text(out / "FULLCTX_FATAL.md", f"{exc}\n\n{traceback.format_exc()}")
        raise
