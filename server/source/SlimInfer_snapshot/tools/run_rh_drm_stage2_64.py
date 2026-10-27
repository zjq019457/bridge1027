#!/usr/bin/env python3
"""Retrieval-Head-Conditioned DRM-Add 64-sample diagnostic."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.fup_guarded_swap_lib import FupHarness, SampleCtx, release_cuda

DEFAULT_MODEL = "/home/yu-bai/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
DEFAULT_LB = "/home/yu-bai/Desktop/jiaqizhang/HeadKV/data/LongBench"
DEFAULT_PROFILE = REPO_ROOT / "profiles" / "retrieval_head_profiles"
DEFAULT_FUP = REPO_ROOT / "results" / "fup_guarded_swap_stage2_64_20260601_051737"
TASKS = ["musique", "hotpotqa", "qasper", "passage_count", "passage_retrieval_en"]

ATTR = {
    "fixed_drm_add": "deployable baseline",
    "rh_drm_add_static": "deployable fixed-pool static head profile",
    "rh_drm_add_query": "deployable fixed-pool query-conditioned head profile",
    "rh_drm_add_learned": "learned_pseudo_label 4-fold cross-fit",
    "rh_drm_add_oracle_diag": "diagnostic_only",
    "rh_drm_add_static_union_pool": "deployable union-pool ablation",
    "rh_drm_add_query_union_pool": "deployable union-pool ablation",
}

FEATURE_COLS = [
    "drm_score", "query_similarity", "retrieval_head_mass", "reasoning_head_mass",
    "query_focused_head_mass", "active_head_coverage", "active_head_deficit",
    "head_coverage_gain", "head_diversity_gain", "head_redundancy_with_active",
    "head_redundancy_with_rescue", "page_redundancy", "token_cost", "layer_id",
]


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def read_jsonl_index(path: Path) -> Dict[str, Dict[str, Any]]:
    return {str(row.get("sample_id", "")): row for row in read_jsonl(path)}


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: List[Dict[str, Any]], header: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    header = header or list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in header})


def latest_layer_records(path: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in read_jsonl(path):
        sid = str(row.get("sample_id", ""))
        old = out.get(sid)
        if old is None or int(row.get("layer_id", -1) or -1) >= int(old.get("layer_id", -1) or -1):
            out[sid] = row
    return out


def mean(values: Iterable[float]) -> float:
    vals = [float(x) for x in values]
    return sum(vals) / len(vals) if vals else 0.0


class RhDriver:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.out = Path(args.output_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "preds").mkdir(exist_ok=True)
        (self.out / "_features").mkdir(exist_ok=True)
        self.profile = json.loads((Path(args.profile_dir) / "head_profile.json").read_text(encoding="utf-8"))
        self.harness: Optional[FupHarness] = None
        self.ctx: Dict[Tuple[str, str], SampleCtx] = {}
        self.sample_order: Dict[str, List[str]] = {}
        self.answers: Dict[Tuple[str, str], Any] = {}
        self.lengths: Dict[Tuple[str, str], int] = {}
        self.all_classes: Dict[str, Any] = {}
        self.audit_paths: Dict[Tuple[str, str], Path] = {}

    def log(self, *args: Any) -> None:
        print("[rh-drm]", *args, flush=True)

    def ensure_harness(self) -> FupHarness:
        if self.harness is None:
            self.log("loading model ...")
            self.harness = FupHarness(self.args.model_path, self.args.longbench_dir, str(self.out / "_hwork"))
        return self.harness

    def load_contexts(self) -> None:
        harness = self.ensure_harness()
        for task in self.args.tasks:
            ids: List[str] = []
            for index, sample in enumerate(harness.load_samples(task, self.args.samples_per_task)):
                ctx = harness.build_sample_ctx(task, sample, index)
                sid = str(ctx.sample_id)
                self.ctx[(task, sid)] = ctx
                self.answers[(task, sid)] = ctx.answers
                self.lengths[(task, sid)] = ctx.context_length
                self.all_classes[task] = ctx.all_classes
                ids.append(sid)
            self.sample_order[task] = ids
            self.log(f"contexts {task}: {len(ids)}")

    def override(
        self,
        mode: str,
        pool: str = "fixed",
        utility_maps: Optional[Dict[Tuple[str, str], Dict[str, Dict[int, float]]]] = None,
        feature_path: Optional[Path] = None,
    ) -> Callable[[str, str], Dict[str, Any]]:
        def fn(task: str, sid: str) -> Dict[str, Any]:
            ov: Dict[str, Any] = {
                "block_strategy": "fixed_active_rescue",
                "rescue_candidate_strategy": "fixed" if pool == "fixed" else "unified",
                "rescue_selector": mode,
                "rh_mode": mode,
                "rh_head_profile": self.profile,
                "rh_cost_adjusted": 1,
                "rh_alpha_drm": self.args.rh_alpha_drm,
                "rh_beta_retrieval": self.args.rh_beta_retrieval,
                "rh_beta_query": self.args.rh_beta_query,
                "rh_beta_reasoning": self.args.rh_beta_reasoning,
                "rh_lambda_head": self.args.rh_lambda_head,
                "rh_lambda_page": self.args.rh_lambda_page,
            }
            if feature_path is not None:
                ov["rh_candidate_dump_path"] = str(feature_path)
            if utility_maps is not None:
                ov["rh_page_utility_map"] = utility_maps.get((task, sid), {})
            return ov
        return fn

    @staticmethod
    def baseline_override(_task: str, _sid: str) -> Dict[str, Any]:
        return {
            "block_strategy": "fixed",
            "rescue_candidate_strategy": "",
            "rescue_policy": "add",
            "rescue_score": "drm",
            "rescue_selector": "drm_mmr",
        }

    def baseline_source_path(self, task: str, audit: bool = False) -> Path:
        suffix = "__audit.jsonl" if audit else ".jsonl"
        return Path(self.args.reuse_baseline_dir) / "preds" / f"fixed_drm_add__{task}{suffix}"

    def run_eval_method(self, name: str, override_fn: Callable[[str, str], Dict[str, Any]]) -> Dict[str, Any]:
        from longbench_eval import scorer

        harness = self.ensure_harness()
        result: Dict[str, Any] = {"method": name, "per_task": {}, "runner_fail": 0}
        for task in self.args.tasks:
            pred_path = self.out / "preds" / f"{name}__{task}.jsonl"
            audit_path = self.out / "preds" / f"{name}__{task}__audit.jsonl"
            if name == "fixed_drm_add" and self.baseline_source_path(task).exists():
                source_pred = self.baseline_source_path(task)
                source_audit = self.baseline_source_path(task, audit=True)
                existing = read_jsonl_index(source_pred)
                self.audit_paths[(name, task)] = source_audit
            else:
                existing = read_jsonl_index(pred_path) if self.args.skip_existing else {}
                self.audit_paths[(name, task)] = audit_path
                if audit_path.exists() and not self.args.skip_existing:
                    audit_path.unlink()

            preds, answers, lengths = [], [], []
            for sid in self.sample_order[task]:
                prev = existing.get(sid)
                ok = prev is not None and str(prev.get("status", "success")) == "success" and str(prev.get("pred", "")).strip()
                if name != "fixed_drm_add" or not self.baseline_source_path(task).exists():
                    if not ok or not self.args.skip_existing or self.args.overwrite_failed:
                        release_cuda()
                        ov = override_fn(task, sid)
                        ov["_method"] = name
                        ov["rescue_audit_path"] = str(audit_path)
                        try:
                            pred = harness.eval_pageset(self.ctx[(task, sid)], ov, mode="gen")["pred"]
                            status = "success" if str(pred).strip() else "failed"
                        except Exception as exc:
                            pred, status = "", "failed"
                            self.log(f"FAIL {name}/{task}/{sid}: {type(exc).__name__}: {exc}")
                        existing[sid] = {"sample_id": sid, "pred": pred, "status": status}
                        write_jsonl(
                            pred_path,
                            [existing[item] for item in self.sample_order[task] if item in existing],
                        )
                row = existing.get(sid, {"pred": "", "status": "failed"})
                preds.append(str(row.get("pred", "") or ""))
                answers.append(self.answers[(task, sid)])
                lengths.append(self.lengths[(task, sid)])
            if name != "fixed_drm_add" or not self.baseline_source_path(task).exists():
                write_jsonl(pred_path, [existing.get(sid, {"sample_id": sid, "pred": "", "status": "failed"}) for sid in self.sample_order[task]])
            result["runner_fail"] += sum(1 for sid in self.sample_order[task] if not str(existing.get(sid, {}).get("pred", "") or "").strip())
            result["per_task"][task] = float(scorer(task, preds, answers, lengths, self.all_classes.get(task, [])))
            self.log(f"{name}/{task}: {result['per_task'][task]:.3f}")
        result["avg"] = mean(result["per_task"].values())
        return result

    def method_budget(self, name: str, baseline_records: Optional[Dict[Tuple[str, str], Dict[str, Any]]] = None) -> Dict[str, Any]:
        records: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for task in self.args.tasks:
            for sid, row in latest_layer_records(self.audit_paths[(name, task)]).items():
                records[(task, sid)] = row
        tokens = [float(row.get("selected_token_count", 0.0) or 0.0) for row in records.values()]
        warnings = sum(1 for row in records.values() if bool(row.get("fallback_used", False)))
        errors: List[float] = []
        if baseline_records is not None:
            for key, row in records.items():
                if key not in baseline_records:
                    continue
                ref = float(baseline_records[key].get("selected_token_count", 0.0) or 0.0)
                cur = float(row.get("selected_token_count", 0.0) or 0.0)
                if ref > 0:
                    errors.append(abs(cur - ref) / ref)
        return {
            "records": records,
            "avg_selected_tokens": round(mean(tokens), 3),
            "budget_warning": warnings,
            "budget_error": mean(errors),
        }


def feature_vector(row: Dict[str, Any], task: str, tasks: List[str]) -> List[float]:
    values = [float(row.get(key, 0.0) or 0.0) for key in FEATURE_COLS]
    values.extend(1.0 if task == item else 0.0 for item in tasks)
    return values


def load_fup_labels(path: Path) -> Dict[Tuple[str, str, int], int]:
    labels: Dict[Tuple[str, str, int], int] = {}
    for row in read_jsonl(path):
        task, sid = str(row.get("task", "")), str(row.get("sample_id", ""))
        for cand in row.get("candidates", []):
            labels[(task, sid, int(cand["page_id"]))] = int(cand.get("in_oracle", 0) or 0)
    return labels


def train_crossfit_ranker(
    rows: List[Dict[str, Any]],
    labels: Dict[Tuple[str, str, int], int],
    tasks: List[str],
    out: Path,
) -> Dict[Tuple[str, str], Dict[str, Dict[int, float]]]:
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingClassifier

    by_sample: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_sample[(str(row["task"]), str(row["sample_id"]))].append(row)
    samples = sorted(by_sample)
    fold_of = {key: index % 4 for index, key in enumerate(samples)}
    maps: Dict[Tuple[str, str], Dict[str, Dict[int, float]]] = defaultdict(lambda: defaultdict(dict))
    pred_rows: List[Dict[str, Any]] = []
    for fold in range(4):
        train_rows = [
            row for key, group in by_sample.items() if fold_of[key] != fold
            for row in group if (str(row["task"]), str(row["sample_id"]), int(row["page_id"])) in labels
        ]
        test_rows = [row for key, group in by_sample.items() if fold_of[key] == fold for row in group]
        if not test_rows:
            continue
        X_train = np.array([feature_vector(row, str(row["task"]), tasks) for row in train_rows], dtype=float)
        y_train = np.array([
            labels[(str(row["task"]), str(row["sample_id"]), int(row["page_id"]))] for row in train_rows
        ], dtype=int)
        X_test = np.array([feature_vector(row, str(row["task"]), tasks) for row in test_rows], dtype=float)
        if len(train_rows) and len(set(y_train.tolist())) >= 2:
            model = HistGradientBoostingClassifier(
                max_iter=200, max_depth=3, learning_rate=0.08, random_state=fold,
            ).fit(X_train, y_train)
            probs = model.predict_proba(X_test)[:, 1]
        else:
            probs = np.full(len(test_rows), float(y_train.mean()) if len(y_train) else 0.0)
        for row, prob in zip(test_rows, probs.tolist()):
            task, sid, page = str(row["task"]), str(row["sample_id"]), int(row["page_id"])
            layer = str(int(row.get("layer_id", 0) or 0))
            maps[(task, sid)][layer][page] = max(maps[(task, sid)][layer].get(page, 0.0), float(prob))
            pred_rows.append({
                "task": task, "sample_id": sid, "layer_id": layer, "page_id": page,
                "fold": fold, "label_in_oracle": labels.get((task, sid, page), ""),
                "predicted_utility": round(float(prob), 7),
            })
    write_csv(out / "rh_learned_crossfit_predictions.csv", pred_rows)
    return maps


def gate(method: Dict[str, Any], baseline: Dict[str, Any], name: str) -> Tuple[bool, List[str]]:
    if "oracle_diag" in name:
        return False, ["diagnostic_only"]
    reasons: List[str] = []
    if float(method["avg"]) <= float(baseline["avg"]):
        reasons.append("avg<=baseline")
    for task, score in method["per_task"].items():
        if float(score) < float(baseline["per_task"].get(task, 0.0)) - 3.0:
            reasons.append(f"{task} drop>3.0")
    if float(method.get("budget_error", 0.0)) >= 0.01:
        reasons.append("budget_error>=1%")
    if int(method.get("budget_warning", 0)) != 0:
        reasons.append("budget_warning")
    if int(method.get("runner_fail", 0)) != 0:
        reasons.append("runner_fail")
    return not reasons, reasons


def write_outputs(drv: RhDriver, results: Dict[str, Dict[str, Any]], feature_paths: Dict[str, Path]) -> None:
    baseline = results["fixed_drm_add"]
    baseline_budget = drv.method_budget("fixed_drm_add")
    baseline_records = baseline_budget.pop("records")
    baseline.update(baseline_budget)
    budget_rows: List[Dict[str, Any]] = []
    for name, result in results.items():
        if name != "fixed_drm_add":
            budget = drv.method_budget(name, baseline_records)
            budget.pop("records")
            result.update(budget)
        for task in drv.args.tasks:
            method_rows = latest_layer_records(drv.audit_paths[(name, task)])
            task_tokens = [float(row.get("selected_token_count", 0.0) or 0.0) for row in method_rows.values()]
            refs = [baseline_records[(task, sid)] for sid in method_rows if (task, sid) in baseline_records]
            ref_tokens = [float(row.get("selected_token_count", 0.0) or 0.0) for row in refs]
            budget_rows.append({
                "method": name, "task": task, "n_records": len(method_rows),
                "avg_selected_tokens": round(mean(task_tokens), 3),
                "baseline_avg_selected_tokens": round(mean(ref_tokens), 3),
                "budget_error_pct": round(float(result.get("budget_error", 0.0)) * 100, 4),
                "budget_warning": result.get("budget_warning", 0),
            })
    write_csv(drv.out / "rh_budget_audit.csv", budget_rows)

    summary: List[Dict[str, Any]] = []
    for name, result in results.items():
        passed, reasons = gate(result, baseline, name)
        row: Dict[str, Any] = {
            "method": name, "attribute": ATTR.get(name, ""), "avg": round(float(result["avg"]), 4),
            "delta_vs_baseline": round(float(result["avg"]) - float(baseline["avg"]), 4),
        }
        for task in drv.args.tasks:
            row[task] = round(float(result["per_task"].get(task, 0.0)), 3)
        row.update({
            "budget_error_pct": round(float(result.get("budget_error", 0.0)) * 100, 4),
            "budget_warning": int(result.get("budget_warning", 0)),
            "runner_fail": int(result.get("runner_fail", 0)),
            "passed_gate": "yes" if passed else "no",
            "gate_reasons": "; ".join(reasons),
        })
        summary.append(row)
    header = ["method", "attribute", "avg", "delta_vs_baseline"] + drv.args.tasks + [
        "budget_error_pct", "budget_warning", "runner_fail", "passed_gate", "gate_reasons",
    ]
    write_csv(drv.out / "rh_stage2_64_summary.csv", summary, header)
    ranking = [row for row in summary if "diagnostic_only" not in row["attribute"] and row["method"] != "fixed_drm_add"]
    ranking.sort(key=lambda row: float(row["avg"]), reverse=True)
    write_csv(drv.out / "rh_stage2_64_ranking.csv", ranking, header)

    all_features: List[Dict[str, Any]] = []
    selected: List[Dict[str, Any]] = []
    for name, path in feature_paths.items():
        for row in read_jsonl(path):
            row["method"] = name
            all_features.append(row)
            if int(row.get("selected", 0) or 0) == 1:
                selected.append({
                    "task": row.get("task"), "sample_id": row.get("sample_id"), "method": name,
                    "layer_id": row.get("layer_id"), "page_id": row.get("page_id"),
                    "candidate_source": row.get("candidate_source"), "token_cost": row.get("token_cost"),
                    "final_rh_utility": row.get("final_rh_utility"),
                })
    feature_header = [
        "task", "sample_id", "method", "layer_id", "page_id", "candidate_source", "token_cost",
        "drm_score", "query_similarity", "retrieval_head_mass", "reasoning_head_mass",
        "query_focused_head_mass", "active_head_coverage", "active_head_deficit", "head_coverage_gain",
        "retrieval_head_gain", "reasoning_head_gain", "query_focused_head_gain", "head_diversity_gain",
        "head_redundancy_with_active", "head_redundancy_with_rescue", "page_redundancy",
        "final_rh_utility", "selected", "rh_mode", "answer_overlap_oracle_diag",
    ]
    write_csv(drv.out / "rh_candidate_features.csv", all_features, feature_header)
    write_csv(drv.out / "rh_selected_rescue_pages.csv", selected)
    write_report(drv, summary, all_features)


def write_report(drv: RhDriver, summary: List[Dict[str, Any]], features: List[Dict[str, Any]]) -> None:
    by_name = {row["method"]: row for row in summary}
    baseline = by_name["fixed_drm_add"]
    static = by_name.get("rh_drm_add_static", {})
    query = by_name.get("rh_drm_add_query", {})
    learned = by_name.get("rh_drm_add_learned", {})
    oracle = by_name.get("rh_drm_add_oracle_diag", {})
    selected_static = [
        row for row in features
        if row.get("method") == "rh_drm_add_static" and int(row.get("selected", 0) or 0) == 1
    ]
    contribution = {
        "retrieval": mean(float(row.get("retrieval_head_gain", 0.0) or 0.0) for row in selected_static),
        "reasoning": mean(float(row.get("reasoning_head_gain", 0.0) or 0.0) for row in selected_static),
        "query_focused": mean(float(row.get("query_focused_head_gain", 0.0) or 0.0) for row in selected_static),
    }
    largest = max(contribution, key=contribution.get) if contribution else "n/a"
    passed = [row["method"] for row in summary if row.get("passed_gate") == "yes"]
    if float(oracle.get("avg", 0.0) or 0.0) <= float(baseline["avg"]) + 0.1:
        failure = "candidate pool/fixed active set already strong; even oracle head guidance has little room"
    elif float(static.get("avg", 0.0) or 0.0) <= float(baseline["avg"]):
        failure = "head profile noise or head-conditioned ranking error"
    else:
        failure = "see per-task drops and budget audit"
    lines = [
        "# RH-DRM-Add Stage2 64-sample Diagnostic",
        "",
        "- Active construction: fixed SlimInfer top-K, unchanged.",
        "- Rescue policy: add-only; no swap and no active deletion.",
        "- Static profile calibration: held-out samples 64:80.",
        "- Head signal: per-head QK attention mass from SlimInfer's pruning query.",
        "- Environment: requested conda `silm` is unavailable; the repo `.venv` was used.",
        "",
        "## Results",
        "",
    ]
    for row in summary:
        lines.append(
            f"- `{row['method']}`: Avg {row['avg']}, delta {row['delta_vs_baseline']}, "
            f"gate {row['passed_gate']}, budget error {row['budget_error_pct']}%"
        )
    lines += [
        "",
        "## Questions",
        "",
        f"1. Retrieval-head-conditioned scoring exceeds fixed DRM-Add: {float(static.get('avg', 0.0) or 0.0) > float(baseline['avg'])}.",
        f"2. Query-conditioned profile is better than static: {float(query.get('avg', 0.0) or 0.0) > float(static.get('avg', 0.0) or 0.0)}.",
        f"3. Learned RH ranker is better than hand-written RH utility: {float(learned.get('avg', 0.0) or 0.0) > max(float(static.get('avg', 0.0) or 0.0), float(query.get('avg', 0.0) or 0.0))}.",
        f"4. Largest selected-head gain component: {largest} ({contribution}).",
        f"5. Musique / HotpotQA effect: static deltas are {float(static.get('musique', 0.0) or 0.0) - float(baseline.get('musique', 0.0) or 0.0):+.3f} / {float(static.get('hotpotqa', 0.0) or 0.0) - float(baseline.get('hotpotqa', 0.0) or 0.0):+.3f}.",
        f"6. Passage retrieval risk: query delta is {float(query.get('passage_retrieval_en', 0.0) or 0.0) - float(baseline.get('passage_retrieval_en', 0.0) or 0.0):+.3f}.",
        f"7. Failure diagnosis if no method passes: {failure}.",
        "",
        "## Promotion",
        "",
        f"- Methods passing the strict gate: {passed if passed else 'none'}.",
        "- Oracle diagnostics are excluded from deployable ranking.",
        "- No full LongBench or latency run was performed.",
    ]
    (drv.out / "report_rh_stage2_64.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Path:
    start = time.time()
    drv = RhDriver(args)
    for name in ["head_profile_summary.csv", "head_profile_report.md"]:
        src = Path(args.profile_dir) / name
        if src.exists():
            shutil.copy2(src, drv.out / name)
    config = {
        "stage": "stage2_64",
        "tasks": args.tasks,
        "samples_per_task": args.samples_per_task,
        "active_construction": "fixed SlimInfer top-K",
        "rescue_policy": "add-only",
        "candidate_pools": ["fixed", "unified"],
        "profile_dir": args.profile_dir,
        "reuse_baseline_dir": args.reuse_baseline_dir,
        "fup_labels": args.fup_labels,
        "rh_weights": {
            "alpha_drm": args.rh_alpha_drm,
            "beta_retrieval": args.rh_beta_retrieval,
            "beta_query": args.rh_beta_query,
            "beta_reasoning": args.rh_beta_reasoning,
            "lambda_head": args.rh_lambda_head,
            "lambda_page": args.rh_lambda_page,
        },
        "python_env": str(REPO_ROOT / ".venv"),
        "env_note": "conda env silm unavailable; repo .venv used",
    }
    (drv.out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    run_sh = (
        "#!/usr/bin/env bash\nset -e\n"
        f"cd {REPO_ROOT}\n"
        f"scripts/run_with_silm.sh tools/run_rh_drm_stage2_64.py --output_dir {drv.out} --skip_existing\n"
    )
    (drv.out / "run_rh_stage2_64.sh").write_text(run_sh, encoding="utf-8")
    (drv.out / "resume.sh").write_text(run_sh.replace("--skip_existing", "--skip_existing --overwrite_failed"), encoding="utf-8")
    for name in ["run_rh_stage2_64.sh", "resume.sh"]:
        (drv.out / name).chmod(0o755)

    drv.load_contexts()
    feature_paths = {
        "rh_drm_add_static": drv.out / "_features" / "rh_drm_add_static.jsonl",
        "rh_drm_add_query": drv.out / "_features" / "rh_drm_add_query.jsonl",
        "rh_drm_add_static_union_pool": drv.out / "_features" / "rh_drm_add_static_union_pool.jsonl",
        "rh_drm_add_query_union_pool": drv.out / "_features" / "rh_drm_add_query_union_pool.jsonl",
        "rh_drm_add_learned": drv.out / "_features" / "rh_drm_add_learned.jsonl",
        "rh_drm_add_oracle_diag": drv.out / "_features" / "rh_drm_add_oracle_diag.jsonl",
    }
    if not args.skip_existing:
        for path in feature_paths.values():
            if path.exists():
                path.unlink()

    results: Dict[str, Dict[str, Any]] = {}
    methods = [
        ("fixed_drm_add", drv.baseline_override),
        ("rh_drm_add_static", drv.override("rh_static", "fixed", feature_path=feature_paths["rh_drm_add_static"])),
        ("rh_drm_add_query", drv.override("rh_query", "fixed", feature_path=feature_paths["rh_drm_add_query"])),
        ("rh_drm_add_static_union_pool", drv.override("rh_static", "unified", feature_path=feature_paths["rh_drm_add_static_union_pool"])),
        ("rh_drm_add_query_union_pool", drv.override("rh_query", "unified", feature_path=feature_paths["rh_drm_add_query_union_pool"])),
    ]
    requested = {item for item in args.methods.split(",") if item}
    for name, fn in methods:
        if requested and name not in requested:
            continue
        drv.log(f"eval {name}")
        results[name] = drv.run_eval_method(name, fn)

    if not requested or "rh_drm_add_learned" in requested:
        static_rows = read_jsonl(feature_paths["rh_drm_add_static"])
        labels = load_fup_labels(Path(args.fup_labels))
        learned_maps = train_crossfit_ranker(static_rows, labels, args.tasks, drv.out)
        drv.log("eval rh_drm_add_learned")
        results["rh_drm_add_learned"] = drv.run_eval_method(
            "rh_drm_add_learned",
            drv.override("rh_learned", "fixed", learned_maps, feature_paths["rh_drm_add_learned"]),
        )
    if not requested or "rh_drm_add_oracle_diag" in requested:
        drv.log("eval rh_drm_add_oracle_diag")
        results["rh_drm_add_oracle_diag"] = drv.run_eval_method(
            "rh_drm_add_oracle_diag",
            drv.override("rh_oracle", "fixed", feature_path=feature_paths["rh_drm_add_oracle_diag"]),
        )

    if "fixed_drm_add" not in results:
        results["fixed_drm_add"] = drv.run_eval_method("fixed_drm_add", drv.baseline_override)
    write_outputs(drv, results, {name: path for name, path in feature_paths.items() if name in results})
    drv.log(f"DONE {(time.time() - start) / 60:.1f} min -> {drv.out}")
    return drv.out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument("--samples_per_task", type=int, default=64)
    parser.add_argument("--methods", default="")
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--model_path", default=DEFAULT_MODEL)
    parser.add_argument("--longbench_dir", default=DEFAULT_LB)
    parser.add_argument("--profile_dir", default=str(DEFAULT_PROFILE))
    parser.add_argument("--reuse_baseline_dir", default=str(DEFAULT_FUP))
    parser.add_argument("--fup_labels", default=str(DEFAULT_FUP / "_labels.jsonl"))
    parser.add_argument("--rh_alpha_drm", type=float, default=1.0)
    parser.add_argument("--rh_beta_retrieval", type=float, default=0.8)
    parser.add_argument("--rh_beta_query", type=float, default=0.6)
    parser.add_argument("--rh_beta_reasoning", type=float, default=0.4)
    parser.add_argument("--rh_lambda_head", type=float, default=0.25)
    parser.add_argument("--rh_lambda_page", type=float, default=0.25)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--overwrite_failed", action="store_true")
    args = parser.parse_args()
    args.tasks = [task for task in args.tasks.split(",") if task]
    if not args.output_dir:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = str(REPO_ROOT / "results" / f"rh_drm_stage2_64_{stamp}")
    return args


if __name__ == "__main__":
    run(parse_args())
