#!/usr/bin/env python3
"""AC-Gated DRM-Add 64-sample diagnostic (add-only, fixed active, no swap)."""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.ac_gated_drm_lib import (
    AC_WEIGHTS, DEFAULT_FUP_DIR, build_ac_utility_maps, candidate_pool_stats_rows,
    load_features_csv, load_fup_keepdrop_model, predict_fup_maps,
)
from tools.fup_guarded_swap_lib import FupHarness, SampleCtx, release_cuda

DEFAULT_MODEL = "/home/yu-bai/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
DEFAULT_LB = "/home/yu-bai/Desktop/jiaqizhang/HeadKV/data/LongBench"
TASKS = ["musique", "hotpotqa", "qasper", "passage_count", "passage_retrieval_en"]
GATE_TASK_AWARE = "gate_task_aware"

ATTR = {
    "fixed_drm_add": "Ours current main method",
    "fup_oracle_keepdrop": "learned utility enhanced diagnostic / secondary variant",
    "gated_drm_add": "risk-gated DRM diagnostic",
    "ac_drm_add": "active-conditioned DRM diagnostic",
    "ac_gated_drm_add": "candidate enhanced method",
    "ac_gated_drm_add_oracle_diag": "oracle diagnostic only, not deployable",
}


def write_csv(path: Path, rows: List[Dict[str, Any]], header: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    header = header or list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in header})


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def read_jsonl_index(path: Path, key: str = "sample_id") -> Dict[str, Dict[str, Any]]:
    return {str(r.get(key, "")): r for r in read_jsonl(path)}


class AcDriver:
    def __init__(self, args):
        self.args = args
        self.out = Path(args.output_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "preds").mkdir(exist_ok=True)
        self.harness: Optional[FupHarness] = None
        self.ctx_cache: Dict[Tuple[str, Any], SampleCtx] = {}
        self.sample_order: Dict[str, List[Any]] = {}
        self.feat_by_sample: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        self.fup_maps: Dict[Tuple[str, str], Dict[int, float]] = {}
        self.ac_maps: Dict[Tuple[str, str], Dict[int, float]] = {}
        self.answers: Dict[Tuple[str, Any], Any] = {}
        self.lengths: Dict[Tuple[str, Any], int] = {}
        self.all_classes: Dict[str, Any] = {}
        self._target_tasks = {t.strip() for t in (args.target_task or "").split(",") if t.strip()} or None

    def log(self, *a):
        print("[ac]", *a, flush=True)

    def ensure_harness(self):
        if self.harness is None:
            self.log("loading model ...")
            self.harness = FupHarness(self.args.model_path, self.args.longbench_dir,
                                      str(self.out / "_hwork"), max_new_tokens=32)

    def get_ctx(self, task: str, sid: Any) -> SampleCtx:
        return self.ctx_cache[(task, sid)]

    def phase_load_features(self):
        fup_feat = Path(self.args.reuse_fup_features)
        if fup_feat.exists():
            self.log(f"reuse features: {fup_feat}")
            self.feat_by_sample = load_features_csv(fup_feat)
        else:
            self.phase_extract()
        model, blob = load_fup_keepdrop_model(Path(self.args.fup_model_dir))
        self.fup_maps = predict_fup_maps(self.feat_by_sample, self.args.tasks, model, blob)
        self.ac_maps = build_ac_utility_maps(self.feat_by_sample, self.fup_maps, AC_WEIGHTS)
        write_csv(self.out / "candidate_pool_stats.csv", candidate_pool_stats_rows(self.feat_by_sample))
        self._rebuild_ctxs()

    def phase_extract(self):
        self.ensure_harness()
        feat_dump = self.out / "_ac_feature_dump.jsonl"
        ac_dump = self.out / "_ac_utility_dump.jsonl"
        if feat_dump.exists():
            feat_dump.unlink()
        if ac_dump.exists():
            ac_dump.unlink()
        for task in self.args.tasks:
            ids = []
            samples = self.harness.load_samples(task, self.args.samples_per_task)
            for i, s in enumerate(samples):
                try:
                    ctx = self.harness.build_sample_ctx(task, s, i)
                    ov = {
                        "block_strategy": "fixed_active_rescue",
                        "rescue_candidate_strategy": "unified",
                        "rescue_selector": "ac_util",
                        "ac_gate_type": "no_gate",
                        "ac_page_utility_map": {},
                        "page_utility_map": {},
                        "feature_dump_path": str(feat_dump),
                        "ac_utility_dump_path": str(ac_dump),
                        "ac_w_drm": AC_WEIGHTS["w_drm"], "ac_w_fup": AC_WEIGHTS["w_fup"],
                "ac_w_novelty": AC_WEIGHTS["w_novelty"], "ac_w_gap": AC_WEIGHTS["w_gap"],
                "ac_w_redundancy": AC_WEIGHTS["w_redundancy"], "ac_w_noise": AC_WEIGHTS["w_noise"],
                    }
                    self.harness.eval_pageset(ctx, ov, mode="gen", gen_tokens=1)
                except Exception as exc:
                    self.log(f"extract FAIL {task}/{i}: {exc}")
                    continue
                self.ctx_cache[(task, ctx.sample_id)] = ctx
                self.answers[(task, ctx.sample_id)] = ctx.answers
                self.lengths[(task, ctx.sample_id)] = ctx.context_length
                self.all_classes[task] = ctx.all_classes
                ids.append(ctx.sample_id)
            self.sample_order[task] = ids
            self.log(f"extract {task}: {len(ids)}")
        from tools.ac_gated_drm_lib import ref_layer_rows
        from tools.run_fup_guarded_swap_stage2_64 import read_jsonl
        by_s = defaultdict(list)
        for r in read_jsonl(feat_dump):
            by_s[(str(r["task"]), str(r["sample_id"]))].append(r)
        self.feat_by_sample = {k: ref_layer_rows(v) for k, v in by_s.items()}

    def _rebuild_ctxs(self):
        self.ensure_harness()
        for task in self.args.tasks:
            ids = []
            samples = self.harness.load_samples(task, self.args.samples_per_task)
            for i, s in enumerate(samples):
                ctx = self.harness.build_sample_ctx(task, s, i)
                sid = str(ctx.sample_id)
                if (task, sid) not in self.feat_by_sample and self.args.skip_existing:
                    continue
                self.ctx_cache[(task, ctx.sample_id)] = ctx
                self.answers[(task, ctx.sample_id)] = ctx.answers
                self.lengths[(task, ctx.sample_id)] = ctx.context_length
                self.all_classes[task] = ctx.all_classes
                ids.append(ctx.sample_id)
            self.sample_order[task] = ids

    def _ac_ov(self, gate: str):
        def fn(task, sid):
            return {
                "block_strategy": "fixed_active_rescue",
                "rescue_candidate_strategy": "unified",
                "rescue_selector": "ac_util",
                "ac_gate_type": gate,
                "ac_cost_adjusted": 1,
                "ac_page_utility_map": self.ac_maps.get((task, str(sid)), {}),
                "page_utility_map": self.fup_maps.get((task, str(sid)), {}),
                "ac_w_drm": AC_WEIGHTS["w_drm"], "ac_w_fup": AC_WEIGHTS["w_fup"],
                "ac_w_novelty": AC_WEIGHTS["w_novelty"], "ac_w_gap": AC_WEIGHTS["w_gap"],
                "ac_w_redundancy": AC_WEIGHTS["w_redundancy"], "ac_w_noise": AC_WEIGHTS["w_noise"],
            }
        return fn

    def ov_baseline(self, task, sid):
        return {"block_strategy": "fixed", "rescue_candidate_strategy": "",
                "rescue_policy": "add", "rescue_score": "drm", "rescue_selector": "drm_mmr"}

    def ov_fup(self, task, sid):
        return {"block_strategy": "fixed_active_rescue", "rescue_candidate_strategy": "unified",
                "rescue_selector": "fup", "page_utility_map": self.fup_maps.get((task, str(sid)), {})}

    def ov_gated(self, task, sid, gate: str):
        return {"block_strategy": "fixed_active_rescue", "rescue_candidate_strategy": "unified",
                "rescue_selector": "drm_mmr", "ac_gate_type": gate}

    def run_eval_method(self, name: str, override_fn) -> Dict[str, Any]:
        from longbench_eval import scorer
        self.ensure_harness()
        result = {"method": name, "per_task": {}, "budget_warning": 0, "runner_fail": 0}
        for task in self.args.tasks:
            pred_path = self.out / "preds" / f"{name}__{task}.jsonl"
            existing = read_jsonl_index(pred_path)
            preds, answers, lengths = [], [], []
            audit_path = self.out / "preds" / f"{name}__{task}__audit.jsonl"
            for sid in self.sample_order.get(task, []):
                prev = existing.get(str(sid))
                ok = prev and str(prev.get("status", "success")) == "success" and prev.get("pred") not in (None, "")
                if ok and self.args.skip_existing and not self.args.overwrite_failed:
                    pr = prev["pred"]
                else:
                    release_cuda()
                    ctx = self.get_ctx(task, sid)
                    ov = override_fn(task, sid)
                    ov["_method"] = name
                    ov["rescue_audit_path"] = str(audit_path)
                    try:
                        pr = self.harness.eval_pageset(ctx, ov, mode="gen")["pred"]
                        st = "success" if pr else "failed"
                    except Exception as exc:
                        pr, st = "", "failed"
                        self.log(f"FAIL {name}/{task}/{sid}: {exc}")
                    existing[str(sid)] = {"sample_id": str(sid), "pred": pr, "status": st}
                preds.append(str(existing[str(sid)].get("pred", "") or ""))
                answers.append(self.answers[(task, sid)])
                lengths.append(self.lengths[(task, sid)])
            with pred_path.open("w", encoding="utf-8") as f:
                for sid in self.sample_order.get(task, []):
                    f.write(json.dumps(existing[str(sid)], ensure_ascii=False) + "\n")
            result["runner_fail"] += sum(
                1 for sid in self.sample_order.get(task, [])
                if not str(existing.get(str(sid), {}).get("pred", "") or "").strip()
            )
            try:
                result["per_task"][task] = float(scorer(task, preds, answers, lengths, self.all_classes.get(task, [])))
            except Exception as exc:
                result["per_task"][task] = 0.0
                self.log(f"score fail {name}/{task}: {exc}")
        result["avg"] = sum(result["per_task"].values()) / max(1, len(result["per_task"]))
        toks = []
        for task in self.args.tasks:
            for r in read_jsonl(self.out / "preds" / f"{name}__{task}__audit.jsonl"):
                if r.get("selected_token_count"):
                    toks.append(int(r["selected_token_count"]))
                if r.get("fallback_used"):
                    result["budget_warning"] += 1
        result["avg_selected_tokens"] = round(sum(toks) / len(toks), 1) if toks else 0.0
        return result


def _budget_error(m: Dict[str, Any], b: Dict[str, Any]) -> float:
    ref = b.get("avg_selected_tokens", 0.0)
    if ref <= 0:
        return 0.0
    return abs(m.get("avg_selected_tokens", 0.0) - ref) / ref


def _gate(m: Dict[str, Any], b: Dict[str, Any], name: str) -> Tuple[bool, List[str]]:
    if "oracle_diag" in name or ATTR.get(name, "").startswith("oracle"):
        return False, ["diagnostic_only"]
    reasons = []
    if m["avg"] <= b["avg"]:
        reasons.append(f"avg<=baseline ({m['avg']:.3f}<={b['avg']:.3f})")
    for t, sc in m["per_task"].items():
        if sc < b["per_task"].get(t, 0.0) - 3.0:
            reasons.append(f"{t} drop>3.0")
    pr = "passage_retrieval_en"
    if pr in m["per_task"] and m["per_task"][pr] < b["per_task"].get(pr, 0.0) - 3.0:
        reasons.append("passage_retrieval_en collapsed")
    if _budget_error(m, b) >= float(m.get("_budget_tol", 0.01)):
        reasons.append(f"budget_error>={m.get('_budget_tol', 0.01)}")
    if m.get("budget_warning", 0) != 0:
        reasons.append("budget_warning")
    if m.get("runner_fail", 0) != 0:
        reasons.append("runner_fail")
    return len(reasons) == 0, reasons


def _write_outputs(drv: AcDriver, results: Dict[str, Dict[str, Any]], baseline: Dict[str, Any], gate_rows: List[Dict]):
    out = drv.out
    rank_names = [n for n in results if "oracle_diag" not in n and "gate_sweep" not in n]
    rows = []
    for n in rank_names:
        m = results[n]
        ok, rs = _gate(m, baseline, n)
        row = {"method": n, "attribute": ATTR.get(n, ""), "avg": round(m["avg"], 4),
               "delta_vs_baseline": round(m["avg"] - baseline["avg"], 4),
               "budget_error_pct": round(_budget_error(m, baseline) * 100, 3),
               "budget_warning": m.get("budget_warning", 0), "runner_fail": m.get("runner_fail", 0),
               "passed_gate": "yes" if ok else "no", "gate_reasons": "; ".join(rs)}
        for t in baseline["per_task"]:
            row[t] = round(m["per_task"].get(t, 0.0), 3)
        rows.append(row)
    hdr = ["method", "attribute", "avg", "delta_vs_baseline"] + list(baseline["per_task"].keys()) + \
          ["budget_error_pct", "budget_warning", "runner_fail", "passed_gate", "gate_reasons"]
    write_csv(out / "ac_stage2_64_summary.csv", rows, header=hdr)
    dep = [r for r in rows if r["attribute"] and "diagnostic" not in r["attribute"] and "main method" not in r["attribute"]]
    dep.sort(key=lambda r: r["avg"], reverse=True)
    write_csv(out / "ac_stage2_64_ranking.csv", dep, header=hdr)
    write_csv(out / "ac_gate_decisions.csv", gate_rows)
    # utility features from dump or rebuild csv
    urows = []
    for line in read_jsonl(out / "_ac_utility_dump.jsonl"):
        urows.append(line)
    if not urows and Path(drv.args.reuse_fup_features).exists():
        import csv as csvmod
        with Path(drv.args.reuse_fup_features).open() as f:
            for r in csvmod.DictReader(f):
                if int(r.get("is_candidate", 0)) == 1:
                    urows.append({
                        "task": r["task"], "sample_id": r["sample_id"], "page_id": r["page_id"],
                        "candidate_source": r.get("candidate_source", ""),
                        "token_cost": r.get("token_cost", 64),
                        "drm_score": r.get("feature_drm_delta", 0),
                        "fup_score": drv.fup_maps.get((r["task"], r["sample_id"]), {}).get(int(r["page_id"]), 0),
                        "query_similarity": r.get("feature_query_similarity", 0),
                        "active_redundancy": r.get("feature_r_active", 0),
                        "novelty_gain": "", "active_gap_score": "",
                        "noise_risk": "", "final_delta_utility":
                        drv.ac_maps.get((r["task"], r["sample_id"]), {}).get(int(r["page_id"]), 0),
                        "selected": 0,
                    })
    write_csv(out / "ac_utility_features.csv", urows)
    sel_rows = []
    for n, m in results.items():
        if "gate_sweep" in n:
            continue
        for task in drv.args.tasks:
            for r in read_jsonl(out / "preds" / f"{n}__{task}__audit.jsonl"):
                for pid in (r.get("rescued_virtual_ids_first50") or r.get("final_block_ids_first100") or [])[:20]:
                    sel_rows.append({"task": task, "sample_id": r.get("sample_id"), "method": n,
                                     "page_id": pid, "need_rescue": r.get("need_rescue"),
                                     "gate_type": r.get("ac_gate_type", r.get("rescue_gate_mode")),
                                     "budget_before": "", "budget_after": r.get("selected_token_count")})
    write_csv(out / "ac_selected_rescue_pages.csv", sel_rows)
    _write_report(out, results, baseline, rows, gate_rows)


def _write_report(out: Path, results, baseline, rows, gate_rows):
    passed = [r["method"] for r in rows if r.get("passed_gate") == "yes" and "main method" not in r.get("attribute", "")]
    gated = sum(1 for g in gate_rows if str(g.get("need_rescue", "")) in ("0", "0.0", ""))
    by_task_gate = defaultdict(int)
    for g in gate_rows:
        if str(g.get("need_rescue")) == "0":
            by_task_gate[g.get("task", "")] += 1
    best_gate = "gate_task_aware"
    sweep = [r for r in rows if "gate_sweep" in r["method"]]
    if sweep:
        best_gate = max(sweep, key=lambda r: r["avg"])["method"].replace("gate_sweep_ac_", "").replace("gate_sweep_gated_", "")
    fup = results.get("fup_oracle_keepdrop", {})
    gated_m = results.get("gated_drm_add", {})
    ac = results.get("ac_drm_add", {})
    acg = results.get("ac_gated_drm_add", {})
    L = ["# AC-Gated DRM-Add Stage2 64-sample Diagnostic\n",
         f"- Env: repo `.venv` (documented as silm equivalent). See `env_logs.txt`.\n",
         f"- Baseline **fixed_drm_add** avg={baseline['avg']:.3f}\n",
         "## 主表\n"]
    for r in rows:
        if "gate_sweep" not in r["method"]:
            L.append(f"- {r['method']}: avg={r['avg']} Δ={r['delta_vs_baseline']} gate={r['passed_gate']}\n")
    L.append(f"\n## Gate 统计\n- need_rescue=0 记录数: {gated}\n- 按任务 gate 掉: {dict(by_task_gate)}\n- sweep 最佳 gate 信号: {best_gate}\n")
    L.append("\n## 问题回答\n")
    qs = [
        ("1. fixed DRM-Add baseline 分数", f"{baseline['avg']:.3f}"),
        ("2. fup_oracle_keepdrop 是否仍超过 baseline", fup.get("avg", 0) > baseline["avg"]),
        ("3. gated_drm_add 是否更好", gated_m.get("avg", 0) > baseline["avg"]),
        ("4. ac_drm_add 是否更好", ac.get("avg", 0) > baseline["avg"]),
        ("5. ac_gated_drm_add 是否更好", acg.get("avg", 0) > baseline["avg"]),
        ("6. 哪个 gate 类型最好", best_gate),
        ("7. gate 掉多少样本", str(gated)),
        ("8. gate 是否保护 passage_count / passage_retrieval_en", "见 per-task 列与 gate_decisions"),
        ("9. active-conditioned utility vs 静态 DRM", ac.get("avg", 0) > gated_m.get("avg", 0)),
        ("10. FUP 与 DRM 是否互补", fup.get("avg", 0) > ac.get("avg", 0) and acg.get("avg", 0) >= ac.get("avg", 0)),
        ("11. ac_gated_drm_add 是否通过 promotion gate", "ac_gated_drm_add" in passed),
        ("12. 是否建议 full LongBench", "仅当 ac_gated_drm_add 过 gate 且 deployable；否则否"),
        ("14. 当前主线", "fixed DRM-Add" if "ac_gated_drm_add" not in passed else "ac_gated_drm_add (enhanced candidate)"),
    ]
    for k, v in qs:
        L.append(f"- **{k}**: {v}\n")
    if "ac_gated_drm_add" not in passed:
        L.append("- **13. 失败原因（若未过 gate）**: gate 过保守/候选池上限/active 已够强/FUP 噪声/budget — 见 gate_reasons 列。\n")
    L.append("- **15. 下一步**: 若 ac_gated_drm_add 过 gate → 更大 Stage2；否则停止扩展，保持 fixed DRM-Add。\n")
    L.append("\n## Guarded Swap / A-F / old selectors\n- 不继续（negative diagnostic）。\n")
    L.append("\n## 方法定位\n")
    for k, v in ATTR.items():
        L.append(f"- **{k}**: {v}\n")
    (out / "report_ac_stage2_64.md").write_text("".join(L), encoding="utf-8")


def run_pipeline(args):
    t0 = time.time()
    drv = AcDriver(args)
    out = drv.out
    cfg = {
        "stage": args.stage, "tasks": args.tasks, "samples_per_task": args.samples_per_task,
        "python_env": str(REPO_ROOT / ".venv"),
        "env_note": "user requested conda env 'silm'; project uses repo .venv (conda-prefix registered)",
        "candidate_pool": "unified union (fixed, overlap, neighbor, hierarchy, evidence_centered, fusion_fine, fusion_fine_no_query)",
        "ac_weights": AC_WEIGHTS,
        "gate_types": args.gate_types.split(","),
        "reuse_fup_features": args.reuse_fup_features,
        "fup_model_dir": args.fup_model_dir,
        "budget_error_tol": args.budget_error_tol,
    }
    (out / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    (out / "env_logs.txt").write_text(
        f"python={REPO_ROOT / '.venv/bin/python'}\nconda_alias=silm->.venv\nlongbench={args.longbench_dir}\n",
        encoding="utf-8",
    )
    sh = f"""#!/usr/bin/env bash
set -e
cd {REPO_ROOT}
.venv/bin/python tools/run_ac_gated_drm_stage2_64.py --output_dir {out} --skip_existing
"""
    (out / "run_ac_stage2_64.sh").write_text(sh, encoding="utf-8")
    (out / "resume.sh").write_text(sh.replace("--skip_existing", "--skip_existing --overwrite_failed"), encoding="utf-8")
    for s in ["run_ac_stage2_64.sh", "resume.sh"]:
        (out / s).chmod(0o755)

    drv.phase_load_features()
    results: Dict[str, Dict[str, Any]] = {}
    gate_rows: List[Dict[str, Any]] = []

    methods = [
        ("fixed_drm_add", drv.ov_baseline),
        ("fup_oracle_keepdrop", drv.ov_fup),
        ("gated_drm_add", lambda t, s: drv.ov_gated(t, s, GATE_TASK_AWARE)),
        ("ac_drm_add", drv._ac_ov("no_gate")),
        ("ac_gated_drm_add", drv._ac_ov(GATE_TASK_AWARE)),
    ]
    for gt in args.gate_types.split(","):
        gt = gt.strip()
        if gt and gt != GATE_TASK_AWARE:
            methods.append((f"gate_sweep_gated_{gt}", lambda t, s, g=gt: drv.ov_gated(t, s, g)))
            methods.append((f"gate_sweep_ac_{gt}", drv._ac_ov(gt)))

    for name, fn in methods:
        drv.log(f"eval {name}")
        results[name] = drv.run_eval_method(name, fn)
        results[name]["_budget_tol"] = args.budget_error_tol
        drv.log(f"  avg={results[name]['avg']:.3f} fail={results[name]['runner_fail']}")

    baseline = results["fixed_drm_add"]
    for name in results:
        for task in args.tasks:
            for r in read_jsonl(out / "preds" / f"{name}__{task}__audit.jsonl"):
                gate_rows.append({
                    "task": r.get("task", task), "sample_id": r.get("sample_id"), "method": name,
                    "need_rescue": r.get("need_rescue", ""), "rescue_need_score": r.get("rescue_need_score", ""),
                    "gate_type": r.get("ac_gate_type", r.get("rescue_gate_mode", "")),
                    "threshold": r.get("gate_threshold", ""),
                    "num_rescue_selected": r.get("num_rescue_selected", r.get("rescued_count", "")),
                    "total_rescue_tokens": r.get("selected_token_count", ""),
                })
    _write_outputs(drv, results, baseline, gate_rows)
    drv.log(f"DONE {(time.time()-t0)/60:.1f} min -> {out}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="stage2_64")
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--samples_per_task", type=int, default=64)
    ap.add_argument("--methods", default="")
    ap.add_argument("--gate_types", default="no_gate,gate_score_threshold,gate_top_margin,gate_task_aware")
    ap.add_argument("--output_dir", default="")
    ap.add_argument("--model_path", default=DEFAULT_MODEL)
    ap.add_argument("--longbench_dir", default=DEFAULT_LB)
    ap.add_argument("--reuse_fup_features", default=str(DEFAULT_FUP_DIR / "features_fup_candidates.csv"))
    ap.add_argument("--fup_model_dir", default=str(DEFAULT_FUP_DIR))
    ap.add_argument("--budget_error_tol", type=float, default=0.01)
    ap.add_argument("--skip_existing", action="store_true")
    ap.add_argument("--overwrite_failed", action="store_true")
    ap.add_argument("--retry_failed_only", action="store_true")
    ap.add_argument("--target_task", default="")
    ap.add_argument("--target_sample_id", default="")
    args = ap.parse_args()
    args.tasks = [t for t in args.tasks.split(",") if t]
    if not args.output_dir:
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = str(REPO_ROOT / "results" / f"ac_gated_drm_stage2_64_{ts}")
    return args


def main():
    run_pipeline(parse_args())


if __name__ == "__main__":
    main()
