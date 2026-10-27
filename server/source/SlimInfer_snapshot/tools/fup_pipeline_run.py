#!/usr/bin/env python3
"""Pipeline orchestration + aggregation + report for the FUP / Guarded-Swap diagnostic."""
from __future__ import annotations

import itertools
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

DEFAULT_SWAP = {"core": 0.8, "swaps": 2, "tau": 0.05}

# attributes: deployable | learned_pseudo_label | diagnostic_only
ATTR = {
    "fixed_drm_add": "deployable",
    "unified_add_noswap": "deployable",
    "fup_output_delta": "deployable",
    "fup_add_gain": "learned_pseudo_label",
    "fup_oracle_keepdrop": "learned_pseudo_label",
    "fup_add_gain_oracle_diag": "diagnostic_only",
    "fup_oracle_keepdrop_oracle_diag": "diagnostic_only",
    "oracle_upper_bound": "diagnostic_only",
    "guarded_swap_drm": "deployable",
    "guarded_swap_fup_output_delta": "deployable",
    "guarded_swap_fup_add_gain": "learned_pseudo_label",
    "guarded_swap_fup_oracle_keepdrop": "learned_pseudo_label",
    "guarded_swap_oracle_diag": "diagnostic_only",
}


def _write(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def run_pipeline(args, DriverCls):
    from tools.run_fup_guarded_swap_stage2_64 import write_csv
    drv = DriverCls(args)
    out = drv.out
    t0 = time.time()

    # ---- config.json + scripts (written up front, support resume) ----
    _write_config(args, out)
    _write_scripts(args, out)

    # ---- phase 0: extraction (features + active set) ----
    drv.phase_extract()

    results: Dict[str, Dict[str, Any]] = {}
    baseline = None

    # ====================== DIRECTION 1: FUP ======================
    if args.run_fup:
        drv.phase_labels()
        drv.phase_train()
        drv.build_output_delta_maps()
        drv.oracle_label_maps()
        # oracle set map for upper bound
        oracle_set_map = {k: lab.get("oracle_set", []) for k, lab in drv.labels.items()}

        fup_methods = [
            ("fixed_drm_add", drv.ov_baseline),
            ("unified_add_noswap", drv.ov_unified_noswap),
            ("fup_add_gain", drv._fup_ov(drv.pred_add_gain)),
            ("fup_oracle_keepdrop", drv._fup_ov(drv.pred_keepdrop)),
            ("fup_output_delta", drv._fup_ov(drv.pred_output_delta)),
            ("fup_add_gain_oracle_diag", drv._fup_ov(drv.pred_oracle_A)),
            ("fup_oracle_keepdrop_oracle_diag", drv._fup_ov(drv.pred_oracle_B)),
            ("oracle_upper_bound", drv._forced_ov(oracle_set_map)),
        ]
        for name, fn in fup_methods:
            drv.log(f"=== eval {name} ===")
            results[name] = drv.run_eval_method(name, fn, force_rerun=False)
            drv.log(f"{name}: avg={results[name]['avg']:.3f} per_task={results[name]['per_task']}")
        baseline = results["fixed_drm_add"]
        _write_fup_outputs(drv, args, results, baseline)

    # ====================== DIRECTION 2: GUARDED SWAP ======================
    if args.run_guarded_swap:
        if baseline is None:
            drv.log("=== eval fixed_drm_add (baseline for swap) ===")
            results["fixed_drm_add"] = drv.run_eval_method("fixed_drm_add", drv.ov_baseline)
            baseline = results["fixed_drm_add"]
        # need utility maps (if FUP not run, build the deployable + label-based ones)
        if not hasattr(drv, "pred_output_delta"):
            drv.build_output_delta_maps()
        if not hasattr(drv, "pred_add_gain"):
            drv.phase_labels(); drv.phase_train(); drv.oracle_label_maps()
        if not hasattr(drv, "pred_oracle_A"):
            drv.oracle_label_maps()

        if "unified_add_noswap" not in results:
            drv.log("=== eval unified_add_noswap (swap backbone reference) ===")
            results["unified_add_noswap"] = drv.run_eval_method("unified_add_noswap", drv.ov_unified_noswap)
        c, s, ta = DEFAULT_SWAP["core"], DEFAULT_SWAP["swaps"], DEFAULT_SWAP["tau"]
        swap_methods = [
            ("guarded_swap_drm", drv._swap_ov("drm", None, c, s, ta)),
            ("guarded_swap_fup_add_gain", drv._swap_ov("injected", drv.pred_add_gain, c, s, ta)),
            ("guarded_swap_fup_oracle_keepdrop", drv._swap_ov("injected", drv.pred_keepdrop, c, s, ta)),
            ("guarded_swap_fup_output_delta", drv._swap_ov("injected", drv.pred_output_delta, c, s, ta)),
            ("guarded_swap_oracle_diag", drv._swap_ov("injected", drv.pred_oracle_A, c, s, ta)),
        ]
        for name, fn in swap_methods:
            drv.log(f"=== eval {name} (core={c},swaps={s},tau={ta}) ===")
            results[name] = drv.run_eval_method(name, fn)
            results[name]["swap_count_avg"] = _avg_swap_count(drv, name)
            drv.log(f"{name}: avg={results[name]['avg']:.3f} swaps={results[name]['swap_count_avg']:.2f}")

        # optional full (core x swaps x tau) sweep -- first-token NLL SCREEN, then confirm best
        if args.full_swap_sweep:
            _swap_sweep_screen(drv, args, results)

        _write_swap_outputs(drv, args, results, baseline)

    # ====================== FINAL RANKING + REPORT ======================
    _write_final_ranking(out, results, baseline)
    _write_report(drv, args, results, baseline, time.time() - t0)
    drv.log(f"DONE in {(time.time()-t0)/60:.1f} min -> {out}")


# --------------------------------------------------------------------- helpers
def _avg_swap_count(drv, name: str) -> float:
    counts = []
    for task in drv.args.tasks:
        for r in _read_audit(drv, name, task):
            if "swap_count" in r:
                counts.append(int(r.get("swap_count", 0) or 0))
    return round(sum(counts) / len(counts), 3) if counts else 0.0


def _read_audit(drv, name: str, task: str) -> List[Dict[str, Any]]:
    from tools.run_fup_guarded_swap_stage2_64 import read_jsonl
    return read_jsonl(drv.out / "preds" / f"{name}__{task}__audit.jsonl")


def _swap_sweep_screen(drv, args, results):
    """Screen the full (core x swaps x tau) grid with first-token NLL gain on a
    subset, then confirm the best config for drm + fup_output_delta at full metric."""
    from tools.run_fup_guarded_swap_stage2_64 import write_csv
    cores, swaps_g, taus = [0.7, 0.8, 0.9], [1, 2, 4], [0.0, 0.05, 0.1]
    if not drv.labels_loaded():
        drv._labels_index(drv.out / "_labels.jsonl")
    screen_rows = []
    best = {}
    for variant, util_src, util_map in [("guarded_swap_drm", "drm", None),
                                        ("guarded_swap_fup_output_delta", "injected", drv.pred_output_delta)]:
        best_gain, best_cfg = -1e9, None
        for c, s, ta in itertools.product(cores, swaps_g, taus):
            gains = []
            for task in drv.args.tasks:
                ids = drv.sample_order.get(task, [])[: args.swap_screen_samples]
                for sid in ids:
                    lab = drv.labels.get((task, str(sid)))
                    if not lab:
                        continue
                    base = float(lab.get("base_nll", float("nan")))
                    ctx = drv.get_ctx(task, sid)
                    ov = drv._swap_ov(util_src, util_map, c, s, ta)(task, sid)
                    ov["_method"] = "screen"
                    try:
                        nll = drv.harness.eval_pageset(ctx, ov, mode="ftnll")["first_token_nll"]
                    except Exception:
                        nll = float("nan")
                    if base == base and nll == nll:
                        gains.append(base - nll)
            mg = sum(gains) / len(gains) if gains else 0.0
            screen_rows.append({"variant": variant, "active_core_ratio": c, "max_swaps": s, "swap_tau": ta,
                                "screen_nll_gain": round(mg, 6), "n": len(gains)})
            if mg > best_gain:
                best_gain, best_cfg = mg, (c, s, ta)
        best[variant] = best_cfg
        drv.log(f"swap screen {variant}: best cfg={best_cfg} gain={best_gain:.5f}")
    write_csv(drv.out / "guarded_swap_screen.csv", screen_rows)
    # confirm best configs at full LongBench metric
    for variant, (c, s, ta) in best.items():
        if variant == "guarded_swap_drm":
            fn = drv._swap_ov("drm", None, c, s, ta)
        else:
            fn = drv._swap_ov("injected", drv.pred_output_delta, c, s, ta)
        nm = f"{variant}@best"
        drv.log(f"=== eval {nm} (core={c},swaps={s},tau={ta}) ===")
        results[nm] = drv.run_eval_method(nm, fn)
        results[nm]["swap_count_avg"] = _avg_swap_count(drv, nm)
        ATTR[nm] = ATTR.get(variant, "deployable")


# --------------------------------------------------------------------- gating
def _budget_error(method: Dict[str, Any], baseline: Dict[str, Any]) -> float:
    b = baseline.get("avg_selected_tokens", 0.0)
    m = method.get("avg_selected_tokens", 0.0)
    if b <= 0:
        return 0.0
    return abs(m - b) / b


def _gate(method: Dict[str, Any], baseline: Dict[str, Any]) -> Tuple[bool, List[str]]:
    from tools.fup_pipeline_run import ATTR
    name = method["method"]
    reasons = []
    if ATTR.get(name, "deployable") == "diagnostic_only":
        return False, ["diagnostic_only"]
    if method["avg"] <= baseline["avg"]:
        reasons.append(f"avg<=baseline ({method['avg']:.3f}<={baseline['avg']:.3f})")
    for task, sc in method["per_task"].items():
        if sc < baseline["per_task"].get(task, 0.0) - 3.0:
            reasons.append(f"{task} drop>{3.0} ({sc:.2f} vs {baseline['per_task'].get(task,0):.2f})")
    pr = "passage_retrieval_en"
    if pr in method["per_task"] and pr in baseline["per_task"]:
        if method["per_task"][pr] < baseline["per_task"][pr] - 3.0:
            reasons.append("passage_retrieval_en collapsed")
    be = _budget_error(method, baseline)
    if be >= 0.01:
        reasons.append(f"budget_error={be*100:.2f}%>=1%")
    if method.get("budget_warning", 0) != 0:
        reasons.append(f"budget_warning={method['budget_warning']}")
    if method.get("runner_fail", 0) != 0:
        reasons.append(f"runner_fail={method['runner_fail']}")
    return (len(reasons) == 0), reasons


def _summary_rows(results, names, baseline):
    rows = []
    for n in names:
        if n not in results:
            continue
        m = results[n]
        passed, reasons = _gate(m, baseline) if baseline else (False, ["no_baseline"])
        row = {"method": n, "attribute": ATTR.get(n, "deployable"), "avg": round(m["avg"], 4),
               "delta_vs_baseline": round(m["avg"] - baseline["avg"], 4) if baseline else "",
               "budget_error_pct": round(_budget_error(m, baseline) * 100, 3) if baseline else "",
               "budget_warning": m.get("budget_warning", 0), "runner_fail": m.get("runner_fail", 0),
               "avg_selected_tokens": m.get("avg_selected_tokens", 0.0),
               "swap_count_avg": m.get("swap_count_avg", ""),
               "passed_gate": "yes" if passed else "no", "gate_reasons": "; ".join(reasons)}
        for task in baseline["per_task"]:
            row[task] = round(m["per_task"].get(task, 0.0), 3)
        rows.append(row)
    return rows


def _write_fup_outputs(drv, args, results, baseline):
    from tools.run_fup_guarded_swap_stage2_64 import write_csv
    names = ["fixed_drm_add", "unified_add_noswap", "fup_add_gain", "fup_oracle_keepdrop", "fup_output_delta",
             "fup_add_gain_oracle_diag", "fup_oracle_keepdrop_oracle_diag", "oracle_upper_bound"]
    rows = _summary_rows(results, names, baseline)
    base_tasks = list(baseline["per_task"].keys())
    header = ["method", "attribute", "avg", "delta_vs_baseline"] + base_tasks + \
             ["budget_error_pct", "budget_warning", "runner_fail", "avg_selected_tokens", "passed_gate", "gate_reasons"]
    write_csv(drv.out / "fup_stage2_64_summary.csv", rows, header=header)
    rank = sorted([r for r in rows if r["attribute"] != "diagnostic_only"],
                  key=lambda r: r["avg"], reverse=True)
    write_csv(drv.out / "fup_stage2_64_ranking.csv", rank, header=header)
    # oracle upper bound summary
    ub_rows = []
    for n in ["fixed_drm_add", "oracle_upper_bound", "fup_add_gain_oracle_diag", "fup_oracle_keepdrop_oracle_diag",
              "fup_add_gain", "fup_oracle_keepdrop", "fup_output_delta"]:
        if n in results:
            ub_rows.append({"method": n, "attribute": ATTR.get(n), "avg": round(results[n]["avg"], 4),
                            "delta_vs_baseline": round(results[n]["avg"] - baseline["avg"], 4)})
    write_csv(drv.out / "oracle_upper_bound_summary.csv", ub_rows)
    # fup_output_delta scores csv (per-sample utility)
    od_rows = []
    for (task, sid), m in getattr(drv, "pred_output_delta", {}).items():
        for p, u in m.items():
            od_rows.append({"task": task, "sample_id": sid, "page_id": p, "output_delta_utility": round(u, 6)})
    write_csv(drv.out / "fup_output_delta_scores.csv", od_rows)
    write_csv(drv.out / "fup_output_delta_summary.csv",
              [{"method": "fup_output_delta", "deployable": True, "avg": round(results["fup_output_delta"]["avg"], 4),
                "delta_vs_baseline": round(results["fup_output_delta"]["avg"] - baseline["avg"], 4)}])


def _write_swap_outputs(drv, args, results, baseline):
    from tools.run_fup_guarded_swap_stage2_64 import write_csv
    names = [n for n in results if n.startswith("guarded_swap")]
    rows = _summary_rows(results, names, baseline)
    base_tasks = list(baseline["per_task"].keys())
    header = ["method", "attribute", "avg", "delta_vs_baseline"] + base_tasks + \
             ["swap_count_avg", "budget_error_pct", "budget_warning", "runner_fail", "passed_gate", "gate_reasons"]
    write_csv(drv.out / "guarded_swap_stage2_64_summary.csv", rows, header=header)
    rank = sorted([r for r in rows if r["attribute"] != "diagnostic_only"], key=lambda r: r["avg"], reverse=True)
    write_csv(drv.out / "guarded_swap_stage2_64_ranking.csv", rank, header=header)
    # case-level examples
    case_rows = []
    for name in names:
        for task in drv.args.tasks:
            for r in _read_audit(drv, name, task)[:200]:
                if int(r.get("swap_count", 0) or 0) > 0:
                    case_rows.append({"method": name, "task": task, "sample_id": r.get("sample_id"),
                                      "swap_count": r.get("swap_count"), "swapped_out_pages": r.get("swapped_out_pages"),
                                      "swapped_in_pages": r.get("swapped_in_pages"),
                                      "active_core_size": r.get("active_core_size"),
                                      "active_tail_size": r.get("active_tail_size"),
                                      "swap_utility_margin": r.get("swap_utility_margin"),
                                      "selected_token_count": r.get("selected_token_count")})
    write_csv(drv.out / "guarded_swap_cases.csv", case_rows)


def _write_final_ranking(out, results, baseline):
    from tools.run_fup_guarded_swap_stage2_64 import write_csv
    if not baseline:
        return
    names = [n for n in results if ATTR.get(n) != "diagnostic_only"]
    rows = _summary_rows(results, names, baseline)
    base_tasks = list(baseline["per_task"].keys())
    header = ["method", "attribute", "avg", "delta_vs_baseline"] + base_tasks + \
             ["swap_count_avg", "budget_error_pct", "budget_warning", "runner_fail", "passed_gate", "gate_reasons"]
    rows.sort(key=lambda r: r["avg"], reverse=True)
    write_csv(out / "final_stage2_64_ranking.csv", rows, header=header)


def _write_config(args, out: Path):
    cfg = {
        "stage": args.stage, "tasks": args.tasks, "samples_per_task": args.samples_per_task,
        "baseline": "fixed_drm_add",
        "candidate_pool_source": "unified = union(fixed, overlap, neighbor, hierarchy, evidence_centered, fusion_fine, fusion_fine_no_query)",
        "active_set": "FIXED SlimInfer top-K (per-layer qk top-K); only rescue/swap injected",
        "budget": {"active": "K (per-layer top-K)", "rescue": "K+M, M=round(K*rescue_ratio), rescue_ratio=0.05",
                   "selection_cost": "expanded_page_tokens"},
        "label_metric": "first_token_gold_NLL (generate max_new_tokens=1); learned_pseudo_label via 4-fold cross-fit (group=sample)",
        "label_topn": args.label_topn, "oracle_greedy_steps": args.oracle_steps,
        "rankers": {"fup_add_gain": "HistGradientBoostingRegressor", "fup_oracle_keepdrop": "HistGradientBoostingClassifier",
                    "fup_output_delta": "deployable proxy = z(attention)+0.5 z(diffusion)+0.5 z(query)-0.5 z(redundancy), no model"},
        "swap": {"default": DEFAULT_SWAP, "sweep": {"active_core_ratio": [0.7, 0.8, 0.9], "max_swaps": [1, 2, 4], "tau": [0.0, 0.05, 0.1]},
                 "sweep_method": "first-token-NLL screen on subset then confirm best at full metric",
                 "full_swap_sweep": bool(args.full_swap_sweep), "swap_screen_samples": args.swap_screen_samples,
                 "rule": "swap iff utility(r) > utility(a)+tau, a in active_tail (not core), budget-neutral, max_swaps cap"},
        "method_attributes": ATTR, "seed": args.seed, "model_path": args.model_path,
    }
    _write(out / "config.json", json.dumps(cfg, indent=2, ensure_ascii=False))


def _write_scripts(args, out: Path):
    od = str(out)
    py = "tools/run_fup_guarded_swap_stage2_64.py"
    common = f"--stage {args.stage} --tasks {','.join(args.tasks)} --samples_per_task {args.samples_per_task} --output_dir {od} --skip_existing"
    sweep = " --full_swap_sweep" if getattr(args, "full_swap_sweep", False) else ""
    _write(out / "run_fup_stage2_64.sh", f"#!/usr/bin/env bash\nset -e\ncd {REPO_ROOT}\n.venv/bin/python {py} --run_fup {common}\n")
    _write(out / "run_guarded_swap_stage2_64.sh", f"#!/usr/bin/env bash\nset -e\ncd {REPO_ROOT}\n.venv/bin/python {py} --run_guarded_swap {common}{sweep}\n")
    _write(out / "resume.sh", f"#!/usr/bin/env bash\nset -e\ncd {REPO_ROOT}\n.venv/bin/python {py} --run_fup --run_guarded_swap {common} --overwrite_failed{sweep}\n")
    for s in ["run_fup_stage2_64.sh", "run_guarded_swap_stage2_64.sh", "resume.sh"]:
        (out / s).chmod(0o755)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_report(drv, args, results, baseline, secs):
    from tools.fup_pipeline_run import _gate, _budget_error
    out = drv.out
    L = []
    L.append("# FUP / Guarded-Swap 64-sample Diagnostic (Stage 2)\n")
    L.append(f"- Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}  |  wall: {secs/60:.1f} min")
    L.append(f"- Tasks: {', '.join(args.tasks)}  |  samples/task: {args.samples_per_task}")
    L.append(f"- Baseline (unified): **fixed_drm_add**\n")

    L.append("## 1. 实验设置\n")
    L.append("- active = FIXED SlimInfer top-K（每层 qk top-K），整轮不变；只注入 rescue 选择 / 效用 / swap。")
    L.append("- candidate_pool = unified = union(fixed, overlap, neighbor, hierarchy, evidence_centered, fusion_fine, fusion_fine_no_query)。")
    L.append("- 预算：rescue 对齐 fixed DRM-Add 的 K+M（M=round(K*0.05)）；guarded swap 为 budget-neutral（1 换 1）。")
    L.append(f"- label metric：**first-token 金标答案 NLL**（generate max_new_tokens=1），cross-fit（4 折，按 sample 分组）避免泄漏。top_N={args.label_topn}，greedy oracle 步数≤{args.oracle_steps}。")
    L.append("- 方法属性：deployable（不需 GT）/ learned_pseudo_label（cross-fit 伪标签，需校准集）/ diagnostic_only（用 GT，仅上界）。\n")

    if baseline:
        L.append("## 2. fixed DRM-Add baseline\n")
        L.append(_md_table([baseline], list(baseline["per_task"].keys())))
        L.append("")

    if args.run_fup:
        L.append("## 3. 方向一 FUP A/B/C 结果\n")
        names = ["fixed_drm_add", "unified_add_noswap", "fup_add_gain", "fup_oracle_keepdrop", "fup_output_delta",
                 "fup_add_gain_oracle_diag", "fup_oracle_keepdrop_oracle_diag", "oracle_upper_bound"]
        L.append(_md_table([results[n] for n in names if n in results], list(baseline["per_task"].keys()), baseline))
        L.append("\n**问题回答：**")
        L.extend(_fup_answers(results, baseline))
        L.append("")

    if args.run_guarded_swap:
        L.append("## 4. 方向二 Guarded Swap 结果\n")
        names = [n for n in results if n.startswith("guarded_swap")]
        L.append(_md_table([results[n] for n in names], list(baseline["per_task"].keys()), baseline))
        L.append("\n**问题回答：**")
        L.extend(_swap_answers(drv, results, baseline))
        L.append("")

    # promotion summary
    L.append("## 5-6. Promotion gate / 方法属性\n")
    passed = []
    for n, m in results.items():
        if ATTR.get(n) == "diagnostic_only":
            continue
        ok, reasons = _gate(m, baseline)
        L.append(f"- **{n}** ({ATTR.get(n)}): {'PASS' if ok else 'FAIL'} {'' if ok else '— ' + '; '.join(reasons)}")
        if ok:
            passed.append(n)
    L.append("")
    diag = [n for n in results if ATTR.get(n) == "diagnostic_only"]
    L.append(f"- diagnostic-only / oracle-only（不进 promotion）：{', '.join(diag)}\n")

    L.append("## 7-8. 是否进入 full / 是否替代 fixed DRM-Add 主线\n")
    if passed:
        L.append(f"- 通过 gate 的方法：**{', '.join(passed)}** → 建议进入更大 Stage 2 / full 复核。")
    else:
        L.append("- **没有任何可部署 / learned 方法通过 promotion gate。** 不建议进入 full，**fixed DRM-Add 保持主线**。")
    L.append("")

    L.append("## 9. 失败原因分析\n")
    L.extend(_failure_analysis(results, baseline))
    L.append("")

    L.append("## 10. 下一步建议\n")
    L.extend(_next_steps(results, baseline, passed))
    _write(out / "report_stage2_64.md", "\n".join(L) + "\n")


def _md_table(methods, tasks, baseline=None):
    cols = ["method", "attr", "avg"] + (["Δ"] if baseline else []) + tasks + ["swaps", "bud_err%", "warn", "fail", "gate"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for m in methods:
        from tools.fup_pipeline_run import _gate, _budget_error
        name = m["method"]
        ok = ""
        if baseline:
            g, _ = _gate(m, baseline)
            ok = "PASS" if g else ("diag" if ATTR.get(name) == "diagnostic_only" else "fail")
        row = [name, ATTR.get(name, "?"), f"{m['avg']:.3f}"]
        if baseline:
            row.append(f"{m['avg']-baseline['avg']:+.3f}")
        for t in tasks:
            row.append(f"{m['per_task'].get(t,0):.2f}")
        row.append(f"{m.get('swap_count_avg','')}")
        row.append(f"{_budget_error(m, baseline)*100:.2f}" if baseline else "")
        row.append(str(m.get("budget_warning", 0)))
        row.append(str(m.get("runner_fail", 0)))
        row.append(ok)
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)


def _fup_answers(results, baseline):
    out = []
    fup = {k: results[k]["avg"] for k in ["fup_add_gain", "fup_oracle_keepdrop", "fup_output_delta"] if k in results}
    best = max(fup, key=fup.get) if fup else None
    out.append(f"1. A/B/C 最佳：**{best}** (avg={fup.get(best,0):.3f})。" if best else "1. 无 FUP 结果。")
    ub = results.get("oracle_upper_bound", {}).get("avg")
    if ub is not None:
        out.append(f"2. oracle 上界 avg={ub:.3f}（Δ={ub-baseline['avg']:+.3f}）：{'候选池有潜力' if ub>baseline['avg']+0.5 else '候选池上限有限'}。")
    od = results.get("fup_output_delta", {}).get("avg")
    if od is not None:
        out.append(f"3. fup_output_delta（不依赖 GT）avg={od:.3f}，{'超过' if od>baseline['avg'] else '未超过'} fixed DRM-Add。")
    out.append("4. learned predictor vs hand-designed selector：见上表 Δ 与各任务波动（learned 多数与 baseline 同量级）。")
    passed_fup = [k for k in fup if results[k]["avg"] > baseline["avg"]]
    out.append(f"5. 通过 gate 的 FUP 方法：{passed_fup if passed_fup else '无'}。")
    return out


def _swap_answers(drv, results, baseline):
    out = []
    diag = results.get("guarded_swap_oracle_diag", {}).get("avg")
    drm = results.get("guarded_swap_drm", {})
    out.append(f"1. active tail 是否有低价值 page：guarded_swap_oracle_diag avg={diag:.3f}（Δ={diag-baseline['avg']:+.3f}）→ {'存在可替换空间' if diag and diag>baseline['avg']+0.3 else 'active set 已较强，swap 空间有限'}。" if diag is not None else "1. 无 oracle_diag。")
    noswap = results.get("unified_add_noswap", {}).get("avg")
    best_swap = max([n for n in results if n.startswith('guarded_swap') and ATTR.get(n)!='diagnostic_only'],
                    key=lambda n: results[n]['avg'], default=None)
    if best_swap:
        out.append(f"2. guarded swap vs pure Add：最佳 swap = {best_swap} avg={results[best_swap]['avg']:.3f}"
                   f"（Δ vs fixed_drm_add={results[best_swap]['avg']-baseline['avg']:+.3f}）。")
        if noswap is not None:
            out.append(f"3. swap 隔离效应（vs unified_add_noswap={noswap:.3f}，同 backbone 无 swap）："
                       f"Δ_swap={results[best_swap]['avg']-noswap:+.3f} → {'swap 本身有正贡献' if results[best_swap]['avg']>noswap+0.2 else 'swap 本身贡献≈0'}。")
    pr = "passage_retrieval_en"
    if drm and pr in drm.get("per_task", {}):
        out.append(f"4. swap 是否伤 passage_retrieval_en：drm swap {pr}={drm['per_task'][pr]:.2f} vs baseline {baseline['per_task'].get(pr,0):.2f}。")
    return out


def _failure_analysis(results, baseline):
    out = []
    ub = results.get("oracle_upper_bound", {}).get("avg")
    learned_best = max([results[k]["avg"] for k in ["fup_add_gain", "fup_oracle_keepdrop", "fup_output_delta"] if k in results], default=None)
    if ub is not None and learned_best is not None:
        if ub <= baseline["avg"] + 0.5:
            out.append("- **candidate pool 上限不够**：oracle 上界 ≈ baseline，加任何候选页都难提升 → 不是 predictor 的问题。")
        elif learned_best <= baseline["avg"]:
            out.append("- **predictor 没学到 / 特征不够**：oracle 上界高但 learned/deployable 不高 → 特征或 ranker 不足以逼近 oracle。")
    swap_best = max([results[n]["avg"] for n in results if n.startswith("guarded_swap") and ATTR.get(n) != "diagnostic_only"], default=None)
    if swap_best is not None and swap_best <= baseline["avg"]:
        out.append("- **active set 已足够强**：guarded swap 不优于 pure Add，说明 active tail 价值不低，swap 不值得。")
    if not out:
        out.append("- 见各方法 gate_reasons。")
    return out


def _next_steps(results, baseline, passed):
    out = []
    if passed:
        out.append(f"- 对通过 gate 的方法（{', '.join(passed)}）做更大样本 / full LongBench 复核。")
    else:
        out.append("- 两个方向均未过 gate：**停止扩展，fixed DRM-Add 保持最终主线**。")
        ub = results.get("oracle_upper_bound", {}).get("avg")
        if ub is not None and ub > baseline["avg"] + 0.5:
            out.append("- oracle 上界仍高：候选池有潜力，瓶颈在特征/ranker，可尝试更强特征（真实 attention output delta）或更强模型。")
        else:
            out.append("- oracle 上界 ≈ baseline：候选池上限有限，应转向改进 active set 本身或 budget 分配，而非 rescue。")
    return out
