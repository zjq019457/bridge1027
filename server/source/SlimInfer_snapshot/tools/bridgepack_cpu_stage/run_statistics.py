#!/usr/bin/env python3
"""CPU-only statistical recompute: per-task, macro, paired bootstrap, Holm correction."""
from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .common import (
    FAIRNESS_DIR,
    METHOD_NORM,
    MULTIHOP_TASKS,
    OUT_ROOT,
    PRIMARY_PAIRS,
    REPO_ROOT,
    TASKS,
    ensure_cpu_only,
    normalize_method,
    read_csv,
    read_json,
    resolve_repo_path,
    utc_now,
    write_csv,
    write_json,
    write_text,
)


BOOTSTRAP_N = 10000
BOOTSTRAP_SEED = 20260725


def _canonical_method(name: str) -> str:
    return normalize_method(name)


def _load_per_sample() -> List[Dict[str, Any]]:
    rows = read_csv(FAIRNESS_DIR / "per_sample_scores.csv")
    out = []
    for r in rows:
        out.append(
            {
                "model": r["model"],
                "budget": int(r["budget"]),
                "method": r["method"],
                "method_norm": _canonical_method(r["method"]),
                "task": r["task"],
                "sample_id": r["sample_id"],
                "score": float(r["score_0_100"]),
            }
        )
    return out


def _load_extra_model_rows() -> List[Dict[str, Any]]:
    """Attach Mistral / 2048 / Llama2 rows when per-sample artifacts exist."""
    extra: List[Dict[str, Any]] = []
    # Mistral predictions -> recompute simple F1 if needed via stored scores json
    mistral_root = REPO_ROOT / "results/bridgepack_second_model_exact_gate_mistral7b_instruct_v03_20260620_055432"
    if mistral_root.exists():
        for score_dir in (mistral_root / "scores").glob("*"):
            method = score_dir.name
            for task_file in score_dir.glob("*.json"):
                task = task_file.stem
                payload = read_json(task_file)
                # LongBench score files often {score: float} only — no per-sample.
                # Try predictions for per-sample bootstrap later.
                pred = mistral_root / "predictions" / method / f"{task}.jsonl"
                if pred.exists():
                    from .common import read_jsonl, normalize_answer

                    try:
                        from longbench_eval import dataset2metric

                        metric = dataset2metric[task]
                    except Exception:
                        def metric(pred_s, gold, **kwargs):  # type: ignore
                            pt = normalize_answer(pred_s).split()
                            gt = normalize_answer(gold).split()
                            from collections import Counter

                            common = Counter(pt) & Counter(gt)
                            same = sum(common.values())
                            if same == 0:
                                return 0.0
                            p = same / max(1, len(pt))
                            r = same / max(1, len(gt))
                            return 2 * p * r / max(1e-12, p + r)

                    for i, row in enumerate(read_jsonl(pred)):
                        answers = row.get("answers") or []
                        if not isinstance(answers, list):
                            answers = [answers]
                        pred_text = str(row.get("pred", ""))
                        best = 0.0
                        for a in answers:
                            best = max(best, float(metric(pred_text, a, all_classes=row.get("all_classes"))))
                        extra.append(
                            {
                                "model": "mistral",
                                "budget": 1024,
                                "method": method,
                                "method_norm": _canonical_method(method),
                                "task": task,
                                "sample_id": str(row.get("sample_id") or row.get("_id") or f"{task}-{i}"),
                                "score": 100.0 * best,
                            }
                        )
    # Qwen 2048
    b2048 = REPO_ROOT / "results/bridgepack_exact_budget_ablation_20260619_231854/budget_2048"
    if b2048.exists():
        for score_dir in (b2048 / "scores").glob("*"):
            method = score_dir.name
            pred_root = b2048 / "predictions" / method
            for task_file in score_dir.glob("*.json"):
                task = task_file.stem
                pred = pred_root / f"{task}.jsonl"
                if not pred.exists():
                    continue
                from .common import read_jsonl, normalize_answer

                try:
                    from longbench_eval import dataset2metric

                    metric = dataset2metric[task]
                except Exception:
                    def metric(pred_s, gold, **kwargs):  # type: ignore
                        return 0.0

                for i, row in enumerate(read_jsonl(pred)):
                    answers = row.get("answers") or []
                    if not isinstance(answers, list):
                        answers = [answers]
                    pred_text = str(row.get("pred", ""))
                    best = 0.0
                    for a in answers:
                        try:
                            best = max(best, float(metric(pred_text, a, all_classes=row.get("all_classes"))))
                        except Exception:
                            best = max(best, 0.0)
                    extra.append(
                        {
                            "model": "qwen",
                            "budget": 2048,
                            "method": method,
                            "method_norm": _canonical_method(method),
                            "task": task,
                            "sample_id": str(row.get("sample_id") or row.get("_id") or f"{task}-{i}"),
                            "score": 100.0 * best,
                        }
                    )
    return extra


def _task_macro(scores_by_task: Mapping[str, Sequence[float]]) -> Tuple[Dict[str, float], float]:
    per_task = {t: (float(np.mean(v)) if len(v) else float("nan")) for t, v in scores_by_task.items()}
    vals = [per_task[t] for t in TASKS if t in per_task and not math.isnan(per_task[t])]
    macro = float(np.mean(vals)) if vals else float("nan")
    return per_task, macro


def paired_bootstrap_delta(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    strata: Sequence[str],
    n_boot: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
) -> Dict[str, Any]:
    """Stratified paired bootstrap of mean(A-B) with task strata."""
    assert len(scores_a) == len(scores_b) == len(strata)
    n = len(scores_a)
    if n == 0:
        return {
            "n": 0,
            "delta": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "crosses_zero": True,
        }
    strata_arr = np.asarray(list(strata))
    unique = sorted(set(strata_arr.tolist()))
    index_by = {s: np.where(strata_arr == s)[0] for s in unique}
    rng = np.random.default_rng(seed)
    observed = float(np.mean(scores_a - scores_b))
    boots = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        chosen = []
        for s in unique:
            idx = index_by[s]
            draw = rng.choice(idx, size=len(idx), replace=True)
            chosen.append(draw)
        samp = np.concatenate(chosen)
        boots[i] = float(np.mean(scores_a[samp] - scores_b[samp]))
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {
        "n": int(n),
        "delta": observed,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "crosses_zero": bool(lo <= 0.0 <= hi),
        "boot_mean": float(np.mean(boots)),
        "boot_std": float(np.std(boots)),
    }


def holm_correction(pvals: Sequence[Tuple[str, float]]) -> List[Dict[str, Any]]:
    """Holm–Bonferroni on (label, p_value) pairs. p approximated from CI crossing."""
    # For CI-based tests we convert crosses_zero to a conservative binary; here we expect real p.
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i][1])
    out = [None] * m  # type: ignore
    running = 0.0
    for rank, idx in enumerate(order):
        label, p = pvals[idx]
        adj = min(1.0, (m - rank) * p)
        if rank > 0:
            adj = max(adj, out[order[rank - 1]]["p_holm"])  # type: ignore
        out[idx] = {"label": label, "p_raw": p, "p_holm": adj, "reject_holm_0.05": adj < 0.05}
    return out  # type: ignore


def _approx_p_from_bootstrap(delta: float, boots_std: float, n_boot: int) -> float:
    """Two-sided normal approx p-value from bootstrap SE."""
    if boots_std <= 1e-12 or math.isnan(delta) or math.isnan(boots_std):
        return 1.0
    z = abs(delta) / boots_std
    # erfc for two-sided normal
    return float(math.erfc(z / math.sqrt(2.0)))


def run_statistics() -> Dict[str, Any]:
    ensure_cpu_only()
    rows = _load_per_sample()
    rows.extend(_load_extra_model_rows())

    # PER_TASK_RESULTS
    grouped: Dict[Tuple[str, int, str], Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        key = (r["model"], r["budget"], r["method_norm"])
        grouped[key][r["task"]].append(r["score"])

    per_task_rows = []
    for (model, budget, method), by_task in sorted(grouped.items()):
        per_task, macro = _task_macro(by_task)
        row = {
            "model": model,
            "budget": budget,
            "method": method,
            "n_total": int(sum(len(v) for v in by_task.values())),
            "macro": macro,
        }
        for t in TASKS:
            row[t] = per_task.get(t, float("nan"))
            row[f"n_{t}"] = len(by_task.get(t, []))
        per_task_rows.append(row)
    write_csv(OUT_ROOT / "statistics" / "PER_TASK_RESULTS.csv", per_task_rows)
    write_csv(OUT_ROOT / "PER_TASK_RESULTS.csv", per_task_rows)

    # Build paired maps: model,budget,method_norm,task,sample_id -> score
    score_map: Dict[Tuple[str, int, str, str, str], float] = {}
    for r in rows:
        score_map[(r["model"], r["budget"], r["method_norm"], r["task"], r["sample_id"])] = r["score"]

    # Expand primary pairs with alias resolution for available methods
    pair_specs = list(PRIMARY_PAIRS)
    # neighbor / source-order ablations when present
    pair_specs.extend(
        [
            ("lexical_bridgepack_path_order", "lexical_bridgepack_source_order"),
            ("lexical_bridgepack_path_order", "bridge_same_spans_source_order"),
            ("bridge_neighbor32_path_order", "bridge_same_spans_source_order"),
            ("lexical_bridgepack_path_order", "query_only_neighbor32_path_order"),
            ("lexical_bridgepack_path_order", "neighbor32_source_order"),
            ("bge_chunk_packing", "bm25_chunk_source_order"),
            ("bridge_neighbor32_path_order", "bm25_chunk_source_order"),
            ("lexical_bridgepack_path_order", "span_only_source_order"),
        ]
    )

    pairwise_rows = []
    pval_inputs = []
    models = sorted({r["model"] for r in rows})
    budgets = sorted({r["budget"] for r in rows})

    for model in models:
        for budget in budgets:
            available = {r["method_norm"] for r in rows if r["model"] == model and r["budget"] == budget}
            for a, b in pair_specs:
                if a not in available or b not in available:
                    continue
                # Align on sample ids
                keys_a = {
                    (t, sid)
                    for (m, bud, meth, t, sid) in score_map
                    if m == model and bud == budget and meth == a
                }
                keys_b = {
                    (t, sid)
                    for (m, bud, meth, t, sid) in score_map
                    if m == model and bud == budget and meth == b
                }
                common = sorted(keys_a & keys_b)
                if not common:
                    continue
                scores_a = np.array([score_map[(model, budget, a, t, sid)] for t, sid in common], dtype=np.float64)
                scores_b = np.array([score_map[(model, budget, b, t, sid)] for t, sid in common], dtype=np.float64)
                strata = [t for t, _ in common]
                boot = paired_bootstrap_delta(scores_a, scores_b, strata)
                label = f"{model}|{budget}|{a} vs {b}"
                p_approx = _approx_p_from_bootstrap(boot["delta"], boot["boot_std"], BOOTSTRAP_N)
                pval_inputs.append((label, p_approx))
                pairwise_rows.append(
                    {
                        "model": model,
                        "budget": budget,
                        "method_a": a,
                        "method_b": b,
                        "comparison": f"{a} - {b}",
                        "n_paired": boot["n"],
                        "delta": boot["delta"],
                        "ci95_low": boot["ci_low"],
                        "ci95_high": boot["ci_high"],
                        "crosses_zero": boot["crosses_zero"],
                        "boot_mean": boot["boot_mean"],
                        "boot_std": boot["boot_std"],
                        "p_approx_normal": p_approx,
                        "n_boot": BOOTSTRAP_N,
                        "seed": BOOTSTRAP_SEED,
                        "direction": "positive" if boot["delta"] > 0 else ("negative" if boot["delta"] < 0 else "zero"),
                    }
                )

    write_csv(OUT_ROOT / "statistics" / "PAIRWISE_BOOTSTRAP.csv", pairwise_rows)
    write_csv(OUT_ROOT / "PAIRWISE_BOOTSTRAP.csv", pairwise_rows)

    holm_rows = holm_correction(pval_inputs) if pval_inputs else []
    # join
    holm_by_label = {h["label"]: h for h in holm_rows}
    multi_rows = []
    for pr in pairwise_rows:
        label = f"{pr['model']}|{pr['budget']}|{pr['method_a']} vs {pr['method_b']}"
        h = holm_by_label.get(label, {})
        multi_rows.append(
            {
                **pr,
                "p_raw": h.get("p_raw", pr["p_approx_normal"]),
                "p_holm": h.get("p_holm", ""),
                "reject_holm_0.05": h.get("reject_holm_0.05", ""),
            }
        )
    write_csv(OUT_ROOT / "statistics" / "MULTIPLE_COMPARISON_CORRECTION.csv", multi_rows)
    write_csv(OUT_ROOT / "MULTIPLE_COMPARISON_CORRECTION.csv", multi_rows)

    # Markdown report
    lines = [
        "# FULL_STATISTICAL_REPORT",
        "",
        f"Generated: {utc_now()}",
        f"Bootstrap: stratified paired, n={BOOTSTRAP_N}, seed={BOOTSTRAP_SEED}",
        f"Source: `{FAIRNESS_DIR.relative_to(REPO_ROOT)}/per_sample_scores.csv` + extra model artifacts",
        "",
        "## Macro overview (selected)",
        "",
        "| model | budget | method | macro | n |",
        "|---|---:|---|---:|---:|",
    ]
    focus_methods = {
        "lexical_bridgepack_path_order",
        "bge_chunk_packing",
        "bm25_chunk_source_order",
        "tfidf_chunk_source_order",
        "bridge_neighbor32_path_order",
        "bge_recall_bridgepack",
        "bge_reranked_bridgepack",
        "span_only_source_order",
    }
    for row in per_task_rows:
        if row["method"] in focus_methods or row["model"] in {"mistral", "llama2"}:
            lines.append(
                f"| {row['model']} | {row['budget']} | {row['method']} | {row['macro']:.4f} | {row['n_total']} |"
            )

    lines.extend(["", "## Pairwise bootstrap (all computed; includes negative deltas)", ""])
    lines.append("| model | budget | comparison | n | delta | 95% CI | crosses 0 | Holm reject |")
    lines.append("|---|---:|---|---:|---:|---|---|---|")
    for pr in multi_rows:
        ci = f"[{pr['ci95_low']:.4f}, {pr['ci95_high']:.4f}]"
        lines.append(
            f"| {pr['model']} | {pr['budget']} | {pr['comparison']} | {pr['n_paired']} | {pr['delta']:.4f} | {ci} | {pr['crosses_zero']} | {pr.get('reject_holm_0.05','')} |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Scores are on 0–100 LongBench metric scale.",
            "- Holm correction applied across all pairwise tests in this table.",
            "- Uncorrected CI / p_approx retained in CSV.",
            "- Mistral rows use available gate predictions (often <1150); treat as partial evidence.",
            "- Both positive and negative deltas are reported; do not cherry-pick.",
            "",
        ]
    )
    write_text(OUT_ROOT / "statistics" / "FULL_STATISTICAL_REPORT.md", "\n".join(lines))
    write_text(OUT_ROOT / "FULL_STATISTICAL_REPORT.md", "\n".join(lines))
    return {"n_per_task_rows": len(per_task_rows), "n_pairwise": len(pairwise_rows)}


if __name__ == "__main__":
    print(run_statistics())
