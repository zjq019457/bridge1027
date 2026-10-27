#!/usr/bin/env python3
"""Official SlimInfer 17-task LongBench full: 3950 samples/method.

No global max_samples=200. Official counts:
  multifieldqa_en=150, lcc=500, repobench-p=500, all other listed tasks=200.
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
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.consolidate_official_repro_results import load_predictions, per_sample_score, write_csv

OFFICIAL_COUNTS = {
    "qasper": 200,
    "multifieldqa_en": 150,
    "hotpotqa": 200,
    "2wikimqa": 200,
    "musique": 200,
    "gov_report": 200,
    "qmsum": 200,
    "multi_news": 200,
    "vcsum": 200,
    "trec": 200,
    "triviaqa": 200,
    "samsum": 200,
    "lsht": 200,
    "passage_count": 200,
    "passage_retrieval_en": 200,
    "lcc": 500,
    "repobench-p": 500,
}
# SlimInfer LONG_BENCH order
TASKS = (
    "passage_retrieval_en",
    "qasper",
    "multifieldqa_en",
    "hotpotqa",
    "2wikimqa",
    "musique",
    "gov_report",
    "qmsum",
    "multi_news",
    "vcsum",
    "passage_count",
    "trec",
    "triviaqa",
    "samsum",
    "lsht",
    "lcc",
    "repobench-p",
)
assert sum(OFFICIAL_COUNTS[t] for t in TASKS) == 3950

CATEGORY_TASKS = {
    "Single-Doc QA": ["qasper", "multifieldqa_en"],
    "Multi-Doc QA": ["hotpotqa", "2wikimqa", "musique"],
    "Summarization": ["gov_report", "qmsum", "multi_news", "vcsum"],
    "Few-shot": ["trec", "triviaqa", "samsum", "lsht"],
    "Synthetic": ["passage_count", "passage_retrieval_en"],
    "Code": ["lcc", "repobench-p"],
}

METHODS = (
    ("sliminfer_original", "run_sliminfer_original"),
    ("cc_sliminfer_v1", "run_cc_sliminfer_v1"),
)

DEFAULT_MODEL = (
    "/home/yu-bai/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/"
    "snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
)
DEFAULT_LONGBENCH = Path("/home/yu-bai/Desktop/jiaqizhang/HeadKV/data/LongBench")
SIX_TASK_SRC = REPO_ROOT / "results/cc_sliminfer_v1_qwen25_6task_20260712_043551"
CAPPED_SRC = REPO_ROOT / "results/cc_sliminfer_v1_qwen25_17task_20260712_074431"

PROTOCOL_FINGERPRINT = {
    "model_snapshot": "a09a35458c702b33eeacc393d103063234e8bc28",
    "model_family": "qwen2",
    "dtype": "fp16",
    "backend": "flash_attention_2",
    "block_strategy": "fixed",
    "block_size": 64,
    "token_unit_size": 8,
    "local_query_window": 4,
    "swap_threshold": 0.9,
    "prune_layers": [9, 18, 26],
    "retain_tokens": [12288, 6144, 4096],
    "decode_policy": "original",
    "rescue_policy": "none",
    "seed": 0,
    "score_kind": "token_unit_qk",
    "cc_alpha": 0.25,
    "cc_min_cluster_len": 2,
    "cc_max_neighbors": 2,
    "cc_replacement_cap": 0.05,
}


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def protocol_config_hash(method: str) -> str:
    payload = {**PROTOCOL_FINGERPRINT, "method": method}
    return sha256_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def parse_args() -> argparse.Namespace:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out_dir",
        type=Path,
        default=REPO_ROOT / f"results/cc_sliminfer_v1_qwen25_17task_full_official_{ts}",
    )
    p.add_argument("--model_path", default=DEFAULT_MODEL)
    p.add_argument("--longbench_dir", default=str(DEFAULT_LONGBENCH))
    p.add_argument("--python", default=str(REPO_ROOT / ".venv/bin/python"))
    p.add_argument("--skip_run", action="store_true")
    p.add_argument("--no_seed", action="store_true")
    p.add_argument("--six_task_src", type=Path, default=SIX_TASK_SRC)
    p.add_argument("--capped_src", type=Path, default=CAPPED_SRC)
    return p.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_longbench_samples(longbench_dir: Path, dataset: str) -> List[Dict[str, Any]]:
    path = longbench_dir / f"{dataset}.jsonl"
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    n = OFFICIAL_COUNTS[dataset]
    if len(rows) < n:
        raise RuntimeError(f"{dataset}: file has {len(rows)} < official {n}")
    return rows[:n]


def sample_hashes(sample: Dict[str, Any], prompt_format: str) -> Dict[str, str]:
    question = str(sample.get("input") or sample.get("question") or "")
    context = str(sample.get("context") or "")
    try:
        prompt = prompt_format.format(**sample)
    except Exception:
        prompt = question + "\n" + context
    return {
        "question_hash": sha256_text(question),
        "context_hash": sha256_text(context),
        "prompt_hash": sha256_text(prompt),
    }


def read_pred_map(path: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        sid = str(row.get("sample_id", ""))
        if sid:
            out[sid] = row
    return out


def source_config_compatible(run_dir: Path, method: str) -> Tuple[bool, str]:
    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        return False, "missing_config"
    cfg = load_json(cfg_path)
    args = cfg.get("args", {})
    meth = str(args.get("method", ""))
    if method == "sliminfer_original":
        if meth != "sliminfer_original":
            return False, f"method_mismatch:{meth}"
    elif method == "cc_sliminfer_v1":
        if meth not in ("cc_sliminfer_v1", "cc_sliminfer"):
            return False, f"method_mismatch:{meth}"
    checks = [
        (int(args.get("block_size", -1)) == 64, "block_size"),
        (int(args.get("token_unit_size", -1)) == 8, "token_unit_size"),
        (int(args.get("local_query_window", -1)) == 4, "local_query_window"),
        (abs(float(args.get("swap_threshold", -1)) - 0.9) < 1e-9, "swap_threshold"),
        (str(args.get("prune_layers")) == "9,18,26", "prune_layers"),
        (str(args.get("retain_tokens")) == "12288,6144,4096", "retain_tokens"),
        (str(args.get("decode_policy")) == "original", "decode_policy"),
        (str(args.get("rescue_policy")) == "none", "rescue_policy"),
        (int(args.get("seed", -1)) == 0, "seed"),
    ]
    for ok, name in checks:
        if not ok:
            return False, f"mismatch:{name}"
    return True, "ok"


def seed_task_prefix(
    *,
    task: str,
    method: str,
    official: List[Dict[str, Any]],
    prompt_format: str,
    sources: List[Tuple[str, Path]],
    dst_path: Path,
    config_hash: str,
) -> Dict[str, Any]:
    """Write contiguous official-order prefix of verified seeds; return audit stats."""
    by_source: Dict[str, Dict[str, Dict[str, Any]]] = {}
    source_ok: Dict[str, str] = {}
    for name, run_dir in sources:
        ok, reason = source_config_compatible(run_dir, method)
        source_ok[name] = reason if ok else f"reject:{reason}"
        if not ok:
            by_source[name] = {}
            continue
        by_source[name] = read_pred_map(run_dir / "longbench_predictions" / f"{task}.jsonl")

    prefix_rows: List[Dict[str, Any]] = []
    missing_from = 0
    rejected = 0
    reused_from: Dict[str, int] = {name: 0 for name, _ in sources}
    detail_rows: List[Dict[str, Any]] = []

    for i, sample in enumerate(official):
        sid = str(sample.get("_id", sample.get("id", i)))
        hashes = sample_hashes(sample, prompt_format)
        chosen = None
        chosen_src = None
        for name, _ in sources:
            pred = by_source.get(name, {}).get(sid)
            if pred is None:
                continue
            # sample_index must match official index for contiguous resume semantics
            pred_idx = pred.get("sample_index")
            if pred_idx is not None and int(pred_idx) != i:
                rejected += 1
                detail_rows.append({
                    "task": task, "sample_index": i, "sample_id": sid,
                    "status": "reject_index_mismatch", "source": name,
                    "pred_index": pred_idx,
                })
                continue
            # identity via sample_id + official content hashes (seed has no stored hashes)
            chosen = {
                "pred": pred.get("pred", ""),
                "answers": pred.get("answers", sample.get("answers", [])),
                "all_classes": pred.get("all_classes", sample.get("all_classes")),
                "length": pred.get("length", sample.get("length", 0)),
                "sample_id": sid,
                "sample_index": i,
                "task": task,
                "question_hash": hashes["question_hash"],
                "context_hash": hashes["context_hash"],
                "prompt_hash": hashes["prompt_hash"],
                "config_hash": config_hash,
                "seed_source": name,
            }
            chosen_src = name
            break
        if chosen is None:
            missing_from = i
            break
        prefix_rows.append(chosen)
        reused_from[chosen_src] = reused_from.get(chosen_src, 0) + 1
        detail_rows.append({
            "task": task, "sample_index": i, "sample_id": sid,
            "status": "reuse", "source": chosen_src,
            **hashes,
        })

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if prefix_rows:
        with dst_path.open("w", encoding="utf-8") as f:
            for row in prefix_rows:
                # strip audit-only fields for runner compatibility
                out = {
                    "pred": row["pred"],
                    "answers": row["answers"],
                    "all_classes": row["all_classes"],
                    "length": row["length"],
                    "sample_id": row["sample_id"],
                    "sample_index": row["sample_index"],
                }
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
    elif dst_path.exists():
        dst_path.unlink()

    n_official = len(official)
    return {
        "task": task,
        "official_n": n_official,
        "reused_prefix": len(prefix_rows),
        "missing_from_index": missing_from if len(prefix_rows) < n_official else None,
        "to_run": n_official - len(prefix_rows),
        "rejected": rejected,
        "reused_from": reused_from,
        "source_config": source_ok,
        "detail": detail_rows,
    }


def qa_cmd(args: argparse.Namespace, method: str, run_dir: Path) -> List[str]:
    run_dir.mkdir(parents=True, exist_ok=True)
    return [
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
        # CRITICAL: no global 200 cap — official full counts (lcc/repobench-p=500)
        "--longbench_max_samples", "0",
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


def count_preds(run_dir: Path) -> Dict[str, int]:
    pred = run_dir / "longbench_predictions"
    out = {}
    for t in TASKS:
        p = pred / f"{t}.jsonl"
        out[t] = sum(1 for l in p.read_text(encoding="utf-8").splitlines() if l.strip()) if p.exists() else 0
    return out


def complete(run_dir: Path) -> bool:
    c = count_preds(run_dir)
    return all(c.get(t, 0) >= OFFICIAL_COUNTS[t] for t in TASKS)


def read_scores(run_dir: Path) -> Dict[str, Any]:
    path = run_dir / "longbench_scores.csv"
    if not path.exists():
        return {}
    per = {}
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("status") == "ok" and row.get("score") not in (None, ""):
                per[row["dataset"]] = float(row["score"])
    vals = [per[t] for t in TASKS if t in per]
    return {"per_task": per, "macro": round(sum(vals) / len(vals), 2) if vals else None}


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


def paired_delta(base: List[Dict[str, Any]], var: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    bmap = {(r["task"], str(r["sample_id"])): r for r in base}
    out = []
    for v in var:
        b = bmap.get((v["task"], str(v["sample_id"])))
        if not b:
            continue
        bs, vs = float(b["score"]), float(v["score"])
        out.append({
            "method": v["method"],
            "task": v["task"],
            "sample_id": v["sample_id"],
            "baseline_score": bs,
            "variant_score": vs,
            "delta": round(vs - bs, 4),
            "pred_changed": int(str(b.get("pred", "")) != str(v.get("pred", ""))),
        })
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
    }


def category_macros(per_task: Dict[str, float]) -> Dict[str, float]:
    out = {}
    for cat, tasks in CATEGORY_TASKS.items():
        vals = [per_task[t] for t in tasks if t in per_task]
        if vals:
            out[cat] = round(sum(vals) / len(vals), 2)
    return out


def gpu_busy(min_free_gb: float = 8.0) -> Tuple[bool, str]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            text=True,
        ).strip().splitlines()
        free_mb = min(float(x.strip()) for x in out if x.strip())
        free_gb = free_mb / 1024.0
        # ignore tiny gnome processes; require enough free for 7B fp16 ~20GB peak but typically ~22 used with headroom
        apps = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader"],
            text=True,
        ).strip()
        heavy = []
        for line in apps.splitlines():
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                mem = float(parts[1].split()[0])
                if mem >= 1024:  # >=1GB
                    heavy.append(line)
        if heavy and free_gb < min_free_gb:
            return True, f"heavy_apps={heavy}; free_gb={free_gb:.1f}"
        if free_gb < min_free_gb:
            return True, f"free_gb={free_gb:.1f} < {min_free_gb}"
        return False, f"free_gb={free_gb:.1f}"
    except Exception as exc:
        return True, f"nvidia_smi_failed:{exc}"


def prepare_seeds(args: argparse.Namespace, out_dir: Path) -> None:
    prompt_map = load_json(REPO_ROOT / "configs" / "dataset2prompt.json")
    longbench = Path(args.longbench_dir)
    missing_rows: List[Dict[str, Any]] = []
    audit_lines = [
        "# SEED_REUSE_AUDIT\n\n",
        "Reuse rule: contiguous official-order prefix only; "
        "require method+config fingerprint match, sample_id match, sample_index==official index; "
        "question/context/prompt hashes derived from official loader for that sample_id.\n\n",
    ]

    for method, subdir in METHODS:
        cfg_hash = protocol_config_hash(method)
        sources = [
            ("six_task", args.six_task_src / subdir),
            ("capped_17task", args.capped_src / subdir),
        ]
        # only existing sources
        sources = [(n, p) for n, p in sources if p.exists()]
        audit_lines.append(f"## method `{method}` config_hash=`{cfg_hash}`\n\n")
        run_dir = out_dir / subdir
        pred_dir = run_dir / "longbench_predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)
        for task in TASKS:
            official = load_longbench_samples(longbench, task)
            dst = pred_dir / f"{task}.jsonl"
            # do not overwrite a longer existing resume file
            existing_n = sum(1 for l in dst.read_text(encoding="utf-8").splitlines() if l.strip()) if dst.exists() else 0
            if existing_n >= OFFICIAL_COUNTS[task]:
                audit_lines.append(f"- `{task}`: already complete ({existing_n})\n")
                continue
            if existing_n > 0:
                audit_lines.append(f"- `{task}`: keep existing resume prefix n={existing_n}\n")
                missing_rows.append({
                    "method": method, "task": task,
                    "official_n": OFFICIAL_COUNTS[task],
                    "have": existing_n,
                    "to_run": OFFICIAL_COUNTS[task] - existing_n,
                    "missing_from_index": existing_n,
                })
                continue
            stats = seed_task_prefix(
                task=task,
                method=method,
                official=official,
                prompt_format=prompt_map[task],
                sources=sources,
                dst_path=dst,
                config_hash=cfg_hash,
            )
            audit_lines.append(
                f"- `{task}`: reused_prefix={stats['reused_prefix']}/{stats['official_n']}, "
                f"to_run={stats['to_run']}, sources={stats['reused_from']}, "
                f"source_config={stats['source_config']}\n"
            )
            missing_rows.append({
                "method": method,
                "task": task,
                "official_n": stats["official_n"],
                "have": stats["reused_prefix"],
                "to_run": stats["to_run"],
                "missing_from_index": stats["missing_from_index"],
                "rejected": stats["rejected"],
            })
            # write per-task detail
            detail_path = out_dir / f"seed_detail_{method}_{task}.jsonl"
            with detail_path.open("w", encoding="utf-8") as f:
                for row in stats["detail"]:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

    (out_dir / "SEED_REUSE_AUDIT.md").write_text("".join(audit_lines), encoding="utf-8")
    write_csv(
        out_dir / "MISSING_SAMPLE_PLAN.csv",
        missing_rows,
        ["method", "task", "official_n", "have", "to_run", "missing_from_index", "rejected"],
    )


def final_sample_audit(out_dir: Path, all_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    report: Dict[str, Any] = {"methods": {}}
    lines = ["# SAMPLE_COUNT_FINAL_AUDIT\n\n"]
    ok_all = True
    for method, _ in METHODS:
        rows = [r for r in all_rows if r["method"] == method]
        by_task: Dict[str, List[str]] = {t: [] for t in TASKS}
        for r in rows:
            by_task.setdefault(r["task"], []).append(str(r["sample_id"]))
        method_ok = True
        task_info = {}
        for t in TASKS:
            ids = by_task.get(t, [])
            expected = OFFICIAL_COUNTS[t]
            dup = len(ids) - len(set(ids))
            n = len(ids)
            task_ok = n == expected and dup == 0
            if not task_ok:
                method_ok = False
            task_info[t] = {"n": n, "expected": expected, "duplicates": dup, "ok": task_ok}
        total = sum(v["n"] for v in task_info.values())
        if total != 3950:
            method_ok = False
        report["methods"][method] = {"total": total, "ok": method_ok, "tasks": task_info}
        lines.append(f"## {method}\n\n- total={total}/3950 ok={method_ok}\n")
        for t in TASKS:
            info = task_info[t]
            lines.append(f"- {t}: {info['n']}/{info['expected']} dup={info['duplicates']} ok={info['ok']}\n")
        lines.append("\n")
        ok_all = ok_all and method_ok

    # cross-method sample set equality
    base_ids = {(r["task"], str(r["sample_id"])) for r in all_rows if r["method"] == "sliminfer_original"}
    cc_ids = {(r["task"], str(r["sample_id"])) for r in all_rows if r["method"] == "cc_sliminfer_v1"}
    equal = base_ids == cc_ids
    lines.append(f"## Cross-method sample set equal: **{equal}**\n")
    lines.append(f"- only_in_baseline: {len(base_ids - cc_ids)}\n")
    lines.append(f"- only_in_cc: {len(cc_ids - base_ids)}\n")
    ok_all = ok_all and equal
    lines.append(f"\n## OVERALL_PASS: **{ok_all}**\n")
    report["sample_sets_equal"] = equal
    report["overall_pass"] = ok_all
    (out_dir / "SAMPLE_COUNT_FINAL_AUDIT.md").write_text("".join(lines), encoding="utf-8")
    return report


def write_reports(out_dir: Path, scores_by_method: Dict[str, Any], all_rows: List[Dict[str, Any]], deltas: List[Dict[str, Any]]) -> None:
    base_scores = scores_by_method["sliminfer_original"]
    cc_scores = scores_by_method["cc_sliminfer_v1"]
    macro_delta = round(float(cc_scores.get("macro", 0)) - float(base_scores.get("macro", 0)), 2)
    dv = [float(d["delta"]) for d in deltas]
    wins = sum(1 for d in dv if d > 0)
    losses = sum(1 for d in dv if d < 0)
    ties = sum(1 for d in dv if d == 0)
    changed = sum(int(d["pred_changed"]) for d in deltas)
    boot = bootstrap_ci(dv)

    task_rows = []
    for task in TASKS:
        b = base_scores.get("per_task", {}).get(task)
        c = cc_scores.get("per_task", {}).get(task)
        task_rows.append({
            "task": task,
            "sliminfer_original": b,
            "cc_sliminfer_v1": c,
            "delta": None if b is None or c is None else round(float(c) - float(b), 2),
            "n_expected": OFFICIAL_COUNTS[task],
        })
    base_cat = category_macros(base_scores.get("per_task", {}))
    cc_cat = category_macros(cc_scores.get("per_task", {}))
    cat_rows = []
    for cat in CATEGORY_TASKS:
        b, c = base_cat.get(cat), cc_cat.get(cat)
        cat_rows.append({
            "category": cat,
            "sliminfer_original": b,
            "cc_sliminfer_v1": c,
            "delta": None if b is None or c is None else round(float(c) - float(b), 2),
        })

    summary_rows = [
        {"method": "sliminfer_original", "macro_avg": base_scores.get("macro"),
         **{t: base_scores.get("per_task", {}).get(t) for t in TASKS}},
        {"method": "cc_sliminfer_v1", "macro_avg": cc_scores.get("macro"), "macro_delta": macro_delta,
         "wins": wins, "losses": losses, "ties": ties, "changed_predictions": changed,
         **{t: cc_scores.get("per_task", {}).get(t) for t in TASKS}},
    ]
    write_csv(out_dir / "scores_summary.csv", summary_rows,
              ["method", "macro_avg", "macro_delta", "wins", "losses", "ties", "changed_predictions", *TASKS])
    write_csv(out_dir / "scores_by_task.csv", task_rows,
              ["task", "sliminfer_original", "cc_sliminfer_v1", "delta", "n_expected"])
    write_csv(out_dir / "scores_by_category.csv", cat_rows,
              ["category", "sliminfer_original", "cc_sliminfer_v1", "delta"])
    write_csv(out_dir / "scores_by_sample.csv", all_rows,
              ["method", "task", "sample_id", "sample_index", "score", "length", "pred"])
    with (out_dir / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    write_csv(out_dir / "per_sample_deltas.csv", deltas,
              ["method", "task", "sample_id", "baseline_score", "variant_score", "delta", "pred_changed"])

    # subset analysis: 6-task QA subset + code full
    six = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", "triviaqa"]
    def subset_macro(scores: Dict[str, Any], tasks: List[str]) -> Optional[float]:
        vals = [scores.get("per_task", {}).get(t) for t in tasks]
        vals = [v for v in vals if v is not None]
        return round(sum(vals) / len(vals), 2) if vals else None
    subset_lines = [
        "# CC_17TASK_FULL_SUBSET_ANALYSIS\n\n",
        f"- 6-task QA subset macro original={subset_macro(base_scores, six)} "
        f"cc={subset_macro(cc_scores, six)}\n",
        f"- Code (lcc+repobench-p) original={subset_macro(base_scores, ['lcc','repobench-p'])} "
        f"cc={subset_macro(cc_scores, ['lcc','repobench-p'])}\n",
        f"- lcc n expected 500; repobench-p n expected 500\n",
    ]
    (out_dir / "CC_17TASK_FULL_SUBSET_ANALYSIS.md").write_text("".join(subset_lines), encoding="utf-8")

    fairness = {
        "protocol": "official_longbench_17task_full_3950",
        "official_counts": OFFICIAL_COUNTS,
        "total_per_method": 3950,
        "longbench_max_samples": 0,
        "matched_prune_layers": [9, 18, 26],
        "matched_retain_tokens": [12288, 6144, 4096],
        "methods": ["sliminfer_original", "cc_sliminfer_v1"],
        "cc_frozen": PROTOCOL_FINGERPRINT,
    }
    (out_dir / "CC_17TASK_FULL_FAIRNESS_AUDIT.md").write_text(
        "# CC 17-task Full Fairness Audit\n\n" + json.dumps(fairness, indent=2) + "\n",
        encoding="utf-8",
    )
    task_deltas = {r["task"]: r["delta"] for r in task_rows}
    cat_deltas = {r["category"]: r["delta"] for r in cat_rows}
    (out_dir / "CC_17TASK_FULL_CLAIM_GUARD.md").write_text(
        "# CC 17-task Full Claim Guard\n\n"
        f"- Macro Δ={macro_delta}\n- Per-task Δ={task_deltas}\n- Category Δ={cat_deltas}\n\n"
        "## Allowed\n\n"
        "> Qwen2.5-7B-Instruct, frozen `cc_sliminfer_v1` vs `sliminfer_original`, "
        "official SlimInfer 17-task LongBench full (3950/method; lcc=500, repobench-p=500).\n\n"
        "## Forbidden without extra evidence\n\n"
        "- Cross-model; latency claims; retuning CC from these numbers.\n",
        encoding="utf-8",
    )
    lines = [
        "# CC 17-task Full Official Report\n\n",
        f"| Method | Macro | Δ |\n|--------|------:|--:|\n",
        f"| sliminfer_original | {base_scores.get('macro')} |  |\n",
        f"| cc_sliminfer_v1 | {cc_scores.get('macro')} | {macro_delta} |\n\n",
        f"Wins/Losses/Ties: {wins}/{losses}/{ties}; changed={changed}\n",
        f"Bootstrap mean Δ={boot.get('mean_delta')} CI=[{boot.get('ci95_low')}, {boot.get('ci95_high')}]\n\n",
        "## Per-task\n\n| Task | original | cc | Δ | n |\n|------|---------:|---:|--:|--:|\n",
    ]
    for r in task_rows:
        lines.append(
            f"| {r['task']} | {r['sliminfer_original']} | {r['cc_sliminfer_v1']} | {r['delta']} | {r['n_expected']} |\n"
        )
    lines.append("\n## Category\n\n| Category | original | cc | Δ |\n|----------|---------:|---:|--:|\n")
    for r in cat_rows:
        lines.append(
            f"| {r['category']} | {r['sliminfer_original']} | {r['cc_sliminfer_v1']} | {r['delta']} |\n"
        )
    (out_dir / "CC_17TASK_FULL_OFFICIAL_REPORT.md").write_text("".join(lines), encoding="utf-8")
    (out_dir / "results.json").write_text(
        json.dumps({
            "macro_delta": macro_delta,
            "bootstrap": boot,
            "task_deltas": task_deltas,
            "category_deltas": cat_deltas,
            "wins": wins, "losses": losses, "ties": ties,
        }, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    counts_payload = {
        "protocol": "official_sliminfer_longbench_17task_full",
        "counts": OFFICIAL_COUNTS,
        "task_order": list(TASKS),
        "total_per_method": sum(OFFICIAL_COUNTS[t] for t in TASKS),
    }
    (out_dir / "OFFICIAL_17TASK_SAMPLE_COUNTS.json").write_text(
        json.dumps(counts_payload, indent=2) + "\n", encoding="utf-8"
    )
    assert counts_payload["total_per_method"] == 3950

    if not args.no_seed:
        prepare_seeds(args, out_dir)

    busy, reason = gpu_busy(8.0)
    if busy and not args.skip_run:
        (out_dir / "BLOCKED_GPU_BUSY.md").write_text(
            f"# BLOCKED_GPU_BUSY\n\nreason: {reason}\n\nDid not start corrected full run.\n",
            encoding="utf-8",
        )
        print(json.dumps({"blocked": True, "reason": reason, "out_dir": str(out_dir)}, indent=2))
        return 2

    results = []
    all_rows: List[Dict[str, Any]] = []
    scores_by_method: Dict[str, Any] = {}

    for method, subdir in METHODS:
        run_dir = out_dir / subdir
        entry: Dict[str, Any] = {"method": method, "run_dir": str(run_dir)}
        if not args.skip_run and not complete(run_dir):
            cmd = qa_cmd(args, method, run_dir)
            # assert no accidental 200 cap
            assert "--longbench_max_samples" in cmd
            assert cmd[cmd.index("--longbench_max_samples") + 1] == "0"
            (out_dir / f"command_{method}.json").write_text(json.dumps({"command": cmd}, indent=2) + "\n")
            env = os.environ.copy()
            env["PYTHONPATH"] = str(REPO_ROOT)
            log = out_dir / f"run_{method}.log"
            with log.open("w", encoding="utf-8") as lf:
                rc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
            entry["returncode"] = rc
            if rc != 0:
                results.append(entry)
                (out_dir / "FAILED.json").write_text(json.dumps(entry, indent=2) + "\n")
                return 1
        if not complete(run_dir):
            entry["error"] = "incomplete"
            entry["prediction_counts"] = count_preds(run_dir)
            results.append(entry)
            (out_dir / "FAILED.json").write_text(json.dumps(entry, indent=2) + "\n")
            return 1
        # trim oversize files if any
        for task in TASKS:
            p = run_dir / "longbench_predictions" / f"{task}.jsonl"
            if not p.exists():
                continue
            lines = [l for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
            if len(lines) > OFFICIAL_COUNTS[task]:
                p.write_text("\n".join(lines[: OFFICIAL_COUNTS[task]]) + "\n", encoding="utf-8")
        rows = consolidate(run_dir, method)
        all_rows.extend(rows)
        scores = read_scores(run_dir)
        scores_by_method[method] = scores
        entry["scores"] = scores
        entry["prediction_counts"] = count_preds(run_dir)
        results.append(entry)

    base_rows = [r for r in all_rows if r["method"] == "sliminfer_original"]
    cc_rows = [r for r in all_rows if r["method"] == "cc_sliminfer_v1"]
    deltas = paired_delta(base_rows, cc_rows)
    write_reports(out_dir, scores_by_method, all_rows, deltas)
    audit = final_sample_audit(out_dir, all_rows)
    print(json.dumps({"out_dir": str(out_dir), "sample_audit": audit, "results": results}, indent=2))
    return 0 if audit.get("overall_pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
