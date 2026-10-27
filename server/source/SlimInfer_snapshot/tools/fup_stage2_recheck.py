#!/usr/bin/env python3
"""Recheck FUP stage2_64 after runner_fail fix: retry one sample + recompute gate."""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

FAILED_TASK = "passage_count"
FAILED_SAMPLE = "870d882bf3ca7872472e9616aa87dc8871e722243ad4c3e9"
SOURCE = REPO_ROOT / "results/fup_guarded_swap_stage2_64_20260601_051737"

FUP_METHODS = [
    "fixed_drm_add",
    "unified_add_noswap",
    "fup_add_gain",
    "fup_oracle_keepdrop",
    "fup_output_delta",
    "fup_add_gain_oracle_diag",
    "fup_oracle_keepdrop_oracle_diag",
    "oracle_upper_bound",
]


def _copy_tree(src: Path, dst: Path, names: list[str]) -> None:
    for n in names:
        sp, dp = src / n, dst / n
        if sp.is_dir():
            if dp.exists():
                shutil.rmtree(dp)
            shutil.copytree(sp, dp)
        elif sp.exists():
            dp.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(sp, dp)


def prepare_recheck_dir(source: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for sub in ["preds", "models", "_hwork"]:
        _copy_tree(source, out, [sub])
    for f in [
        "_extract_done.json", "_feature_dump.jsonl", "_labels.jsonl", "features_fup_candidates.csv",
        "config.json", "fup_add_gain_eval_predictions.csv", "fup_oracle_keepdrop_predictions.csv",
        "fup_add_gain_train_labels.csv", "fup_oracle_keepdrop_labels.csv",
    ]:
        if (source / f).exists():
            shutil.copy2(source / f, out / f)
    (out / "env_logs.txt").write_text(
        "python_env=project .venv (conda prefix /home/yu-bai/Desktop/jiaqizhang/SlimInfer/.venv)\n"
        "note=user requested conda env 'silm'; use this project venv as silm equivalent\n",
        encoding="utf-8",
    )


def run_retry(out: Path, source: Path) -> None:
    from argparse import Namespace
    from tools.run_fup_guarded_swap_stage2_64 import Driver

    cfg = json.loads((source / "config.json").read_text(encoding="utf-8"))
    tasks = cfg.get("tasks", [])
    args = Namespace(
        run_fup=True,
        run_guarded_swap=False,
        stage="stage2_64",
        tasks=tasks,
        samples_per_task=int(cfg.get("samples_per_task", 64)),
        skip_existing=True,
        overwrite_failed=True,
        retry_failed_only=True,
        target_task=FAILED_TASK,
        target_sample_id=FAILED_SAMPLE,
        recheck_only=False,
        source_dir=str(source),
        output_dir=str(out),
        model_path=cfg.get("model_path", ""),
        longbench_dir="/home/yu-bai/Desktop/jiaqizhang/HeadKV/data/LongBench",
        label_topn=int(cfg.get("label_topn", 16)),
        oracle_steps=int(cfg.get("oracle_greedy_steps", 3)),
        swap_screen_samples=16,
        full_swap_sweep=False,
        seed=int(cfg.get("seed", 0)),
    )

    drv = Driver(args)
    drv.phase_extract()
    drv.phase_train()
    drv.build_output_delta_maps()
    drv.oracle_label_maps()
    oracle_set_map = {k: lab.get("oracle_set", []) for k, lab in drv.labels.items()}

    methods = [
        ("fixed_drm_add", drv.ov_baseline),
        ("unified_add_noswap", drv.ov_unified_noswap),
        ("fup_add_gain", drv._fup_ov(drv.pred_add_gain)),
        ("fup_oracle_keepdrop", drv._fup_ov(drv.pred_keepdrop)),
        ("fup_output_delta", drv._fup_ov(drv.pred_output_delta)),
        ("fup_add_gain_oracle_diag", drv._fup_ov(drv.pred_oracle_A)),
        ("fup_oracle_keepdrop_oracle_diag", drv._fup_ov(drv.pred_oracle_B)),
        ("oracle_upper_bound", drv._forced_ov(oracle_set_map)),
    ]
    results = {}
    for name, fn in methods:
        drv.log(f"recheck eval {name}")
        results[name] = drv.run_eval_method(name, fn, force_rerun=False)
    baseline = results["fixed_drm_add"]
    _write_recheck_reports(out, results, baseline)
    if out.resolve() != source.resolve():
        for fn in ["recheck_summary.csv", "recheck_gate_report.md", "fup_recheck_summary.csv",
                   "fup_recheck_ranking.csv", "fup_recheck_gate_report.md"]:
            if (out / fn).exists():
                shutil.copy2(out / fn, source / fn)


def _write_recheck_reports(out: Path, results: dict, baseline: dict) -> None:
    from tools.fup_pipeline_run import ATTR, _gate, _summary_rows
    from tools.run_fup_guarded_swap_stage2_64 import write_csv

    names = FUP_METHODS
    rows = _summary_rows(results, names, baseline)
    header = ["method", "attribute", "avg", "delta_vs_baseline"] + list(baseline["per_task"].keys()) + \
             ["budget_error_pct", "budget_warning", "runner_fail", "passed_gate", "gate_reasons"]
    write_csv(out / "fup_recheck_summary.csv", rows, header=header)
    write_csv(out / "recheck_summary.csv", rows, header=header)
    rank = sorted([r for r in rows if r["attribute"] != "diagnostic_only"], key=lambda x: x["avg"], reverse=True)
    write_csv(out / "fup_recheck_ranking.csv", rank, header=header)

    L = ["# FUP Stage2 Recheck Gate Report\n"]
    L.append(f"- Baseline: fixed_drm_add avg={baseline['avg']:.4f}, runner_fail={baseline.get('runner_fail', 0)}\n")
    for n in names:
        m = results[n]
        ok, reasons = _gate(m, baseline)
        L.append(f"## {n} ({ATTR.get(n)})\n- PASS: **{ok}**\n- reasons: {reasons or 'none'}\n")
    passed = [n for n in names if ATTR.get(n) != "diagnostic_only" and _gate(results[n], baseline)[0]]
    L.append("\n## 结论\n")
    qs = [
        ("fup_oracle_keepdrop 是否通过 gate？", "fup_oracle_keepdrop"),
        ("fup_add_gain 是否通过 gate？", "fup_add_gain"),
        ("fup_output_delta 是否通过 gate？", "fup_output_delta"),
    ]
    for i, (q, key) in enumerate(qs, 1):
        m = results[key]
        ok, _ = _gate(m, baseline)
        L.append(f"{i}. {q} **{'是' if ok else '否'}** (avg Δ={m['avg']-baseline['avg']:+.3f}, runner_fail={m.get('runner_fail',0)})\n")
    dep = [n for n in ["fup_output_delta", "fup_add_gain", "fup_oracle_keepdrop"]
           if ATTR.get(n) != "diagnostic_only" and results[n]["avg"] > baseline["avg"]]
    L.append(f"- deployable/learned 超过 baseline：{dep or '无'}\n")
    if passed:
        L.append(f"- 通过 gate（可部署/ learned）：{passed} → 仅 **fup_oracle_keepdrop** 若过 gate 应标为 enhanced learned variant，非默认主线。\n")
        L.append("- FUP 是否进 full：**仅当 deployable 方法过 gate**；learned 过 gate 最多作 secondary ablation。\n")
    else:
        L.append("- **全部 FUP 未过 gate → 停止 FUP，fixed DRM-Add 保持主线。**\n")
    L.append("\n## Guarded Swap（不继续）\n")
    L.append("- 保留为 negative diagnostic：swap 收益为负、tail 可换空间小、passage_retrieval 有 recall 风险；add-only fixed DRM-Add 更稳。\n")
    text = "\n".join(L)
    (out / "fup_recheck_gate_report.md").write_text(text, encoding="utf-8")
    (out / "recheck_gate_report.md").write_text(text, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source_dir", default=str(SOURCE))
    ap.add_argument("--output_dir", default="")
    ap.add_argument("--in_place", action="store_true", help="Retry in source dir (no copy)")
    args = ap.parse_args()
    source = Path(args.source_dir)
    if args.in_place:
        out = source
    else:
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        out = Path(args.output_dir) if args.output_dir else REPO_ROOT / "results" / f"fup_guarded_swap_stage2_64_recheck_{ts}"
        prepare_recheck_dir(source, out)
    run_retry(out, source)
    sh = out / "run_recheck.sh"
    sh.write_text(
        f"#!/usr/bin/env bash\nset -e\ncd {REPO_ROOT}\n.venv/bin/python tools/fup_stage2_recheck.py --source_dir {source} --output_dir {out}\n",
        encoding="utf-8",
    )
    sh.chmod(0o755)
    print(f"recheck done -> {out}")


if __name__ == "__main__":
    main()
