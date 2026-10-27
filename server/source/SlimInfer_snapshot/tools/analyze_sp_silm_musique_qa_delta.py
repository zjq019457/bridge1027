#!/usr/bin/env python3
"""Post-hoc MuSiQue delta analysis: SP vs SILM (wins/losses, bootstrap, active-set)."""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bridgekv.sp_sliminfer import compute_sp_features_for_block, build_block_layer_index
from longbench_eval import dataset2metric
from tools.consolidate_official_repro_results import per_sample_score
from tools.run_bridgeflow_silm_property_probe import read_jsonl

PRUNE_LAYERS = [9, 18, 26]
TASK = "musique"
HPQA = "hotpotqa"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--compare_dir",
        type=Path,
        default=REPO_ROOT / "results/sp_reliability_2task_compare_20260710",
    )
    p.add_argument(
        "--live_audit_dir",
        type=Path,
        default=REPO_ROOT / "results/sp_reliability_live_prefill_audit_20260710",
    )
    p.add_argument(
        "--property_features",
        type=Path,
        default=REPO_ROOT
        / "results/bridgeflow_silm_property_probe_2task_full_20260709_211600/property_features.jsonl",
    )
    p.add_argument("--bootstrap_n", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", type=Path, default=None)
    return p.parse_args()


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def load_sample_scores(compare_dir: Path, method: str) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """(task, sample_id) -> row from scores_by_sample or predictions."""
    path = compare_dir / f"run_{method}" / "scores_by_sample.csv"
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("method") != method:
                continue
            key = (str(row["task"]), str(row["sample_id"]))
            out[key] = row
    return out


def load_predictions(compare_dir: Path, method: str, task: str) -> Dict[str, Dict[str, Any]]:
    path = compare_dir / f"run_{method}" / "longbench_predictions" / f"{task}.jsonl"
    out: Dict[str, Dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        out[str(row.get("sample_id"))] = row
    return out


def load_uasr_index(dump_path: Path, task: str) -> Dict[Tuple[str, int], Mapping[str, Any]]:
    idx: Dict[Tuple[str, int], Mapping[str, Any]] = {}
    for row in read_jsonl(dump_path):
        if str(row.get("phase", "")) != "prefill" or str(row.get("task", "")) != task:
            continue
        sid = str(row.get("sample_id", ""))
        layer = int(row.get("layer_id", -1))
        if sid and layer >= 0:
            idx[(sid, layer)] = row
    return idx


def sign_test_p_value(n_pos: int, n_neg: int) -> float:
    """Two-sided binomial sign test (exclude ties)."""
    n = n_pos + n_neg
    if n == 0:
        return 1.0
    k = min(n_pos, n_neg)
    # P(X <= k) for Binomial(n, 0.5), two-sided
    p_one = sum(math.comb(n, i) * (0.5 ** n) for i in range(0, k + 1))
    return min(1.0, 2.0 * p_one)


def paired_bootstrap_ci(
    deltas: Sequence[float],
    *,
    n_boot: int,
    seed: int,
    stat_fn=statistics.mean,
) -> Tuple[float, float, float]:
    rng = random.Random(seed)
    n = len(deltas)
    if n == 0:
        return 0.0, 0.0, 0.0
    stats = []
    for _ in range(n_boot):
        sample = [deltas[rng.randrange(n)] for _ in range(n)]
        stats.append(stat_fn(sample))
    stats.sort()
    lo = stats[int(0.025 * n_boot)]
    hi = stats[int(0.975 * n_boot) - 1]
    return stat_fn(deltas), lo, hi


def permutation_test_mean(deltas: Sequence[float], *, n_perm: int, seed: int) -> float:
    rng = random.Random(seed)
    obs = statistics.mean(deltas)
    n = len(deltas)
    if n == 0:
        return 1.0
    count = 0
    for _ in range(n_perm):
        flipped = [d if rng.random() > 0.5 else -d for d in deltas]
        if abs(statistics.mean(flipped)) >= abs(obs):
            count += 1
    return (count + 1) / (n_perm + 1)


def macro_delta(hp_deltas: Sequence[float], mq_deltas: Sequence[float]) -> float:
    if not hp_deltas or not mq_deltas:
        return 0.0
    return (statistics.mean(hp_deltas) + statistics.mean(mq_deltas)) / 2.0


def bootstrap_macro_ci(
    hp_deltas: Sequence[float],
    mq_deltas: Sequence[float],
    *,
    n_boot: int,
    seed: int,
) -> Tuple[float, float, float]:
    rng = random.Random(seed)
    nh, nm = len(hp_deltas), len(mq_deltas)
    stats = []
    for _ in range(n_boot):
        hp_s = [hp_deltas[rng.randrange(nh)] for _ in range(nh)]
        mq_s = [mq_deltas[rng.randrange(nm)] for _ in range(nm)]
        stats.append((statistics.mean(hp_s) + statistics.mean(mq_s)) / 2.0)
    stats.sort()
    obs = macro_delta(hp_deltas, mq_deltas)
    lo = stats[int(0.025 * n_boot)]
    hi = stats[int(0.975 * n_boot) - 1]
    return obs, lo, hi


def remove_topk_robustness(deltas: Sequence[float], k: int) -> float:
    """Mean delta after removing top-k positive deltas."""
    sorted_d = sorted(deltas, reverse=True)
    remain = sorted_d[k:] if k < len(sorted_d) else []
    return statistics.mean(remain) if remain else 0.0


def block_set(row: Mapping[str, Any]) -> Set[int]:
    return {int(x) for x in row.get("selected_block_ids", [])}


def feature_mean(feat_rows: Mapping[int, Mapping[str, Any]], bids: Iterable[int], key: str) -> Optional[float]:
    vals = []
    for bid in bids:
        v = feat_rows.get(int(bid), {}).get(key, "")
        if v != "" and v is not None:
            try:
                vals.append(float(v))
            except (TypeError, ValueError):
                pass
    return statistics.mean(vals) if vals else None


def analyze_active_set_sample(
    sid: str,
    *,
    silm_uasr: Mapping[Tuple[str, int], Mapping[str, Any]],
    sp_uasr: Mapping[Tuple[str, int], Mapping[str, Any]],
    layer_index: Mapping[Tuple[str, str, int], Mapping[int, Mapping[str, Any]]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for layer in PRUNE_LAYERS:
        silm_row = silm_uasr.get((sid, layer))
        sp_row = sp_uasr.get((sid, layer))
        if not silm_row or not sp_row:
            continue
        silm_sel = block_set(silm_row)
        sp_sel = block_set(sp_row)
        if silm_sel == sp_sel:
            continue
        added = sp_sel - silm_sel
        removed = silm_sel - sp_sel
        budget = int(sp_row.get("budget_k", len(sp_sel)))
        churn = 1.0 - len(silm_sel & sp_sel) / max(1, budget)
        sp_feats_layer: Dict[int, Dict[str, Any]] = {}
        for bid in added | removed:
            sp_feats_layer[int(bid)] = compute_sp_features_for_block(
                task=TASK,
                sample_id=sid,
                block_id=int(bid),
                layer=layer,
                layer_index=layer_index,
            )
        rows.append({
            "sample_id": sid,
            "layer_id": layer,
            "budget_k": budget,
            "churn_rate": round(churn, 6),
            "added_count": len(added),
            "removed_count": len(removed),
            "added_block_ids": json.dumps(sorted(added)),
            "removed_block_ids": json.dumps(sorted(removed)),
            "removed_mean_drift": feature_mean(sp_feats_layer, removed, "semantic_drift"),
            "removed_mean_redundancy": feature_mean(sp_feats_layer, removed, "active_set_redundancy"),
            "removed_mean_absorption": feature_mean(sp_feats_layer, removed, "semantic_absorption"),
            "added_mean_persistence": feature_mean(sp_feats_layer, added, "semantic_persistence"),
            "added_mean_relay": feature_mean(sp_feats_layer, added, "semantic_relay"),
            "added_mean_drift": feature_mean(sp_feats_layer, added, "semantic_drift"),
            "added_mean_redundancy": feature_mean(sp_feats_layer, added, "active_set_redundancy"),
        })
    return rows


def main() -> None:
    args = parse_args()
    compare_dir = args.compare_dir.resolve()
    out_dir = (args.out_dir or compare_dir / "musique_sp_vs_silm_analysis").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    silm_scores = load_sample_scores(compare_dir, "sliminfer_original")
    sp_scores = load_sample_scores(compare_dir, "sp_sliminfer")
    silm_pred_mq = load_predictions(compare_dir, "sliminfer_original", TASK)
    sp_pred_mq = load_predictions(compare_dir, "sp_sliminfer", TASK)
    silm_pred_hp = load_predictions(compare_dir, "sliminfer_original", HPQA)
    sp_pred_hp = load_predictions(compare_dir, "sp_sliminfer", HPQA)

    # --- Part 1: MuSiQue paired comparison ---
    paired_rows: List[Dict[str, Any]] = []
    wins = losses = ties = 0
    changed_pred = unchanged_pred = 0
    pos_sum = neg_sum = 0.0

    for sid, silm_p in silm_pred_mq.items():
        sp_p = sp_pred_mq.get(sid)
        if not sp_p:
            continue
        silm_sc = float(silm_scores[(TASK, sid)]["score"])
        sp_sc = float(sp_scores[(TASK, sid)]["score"])
        delta = sp_sc - silm_sc
        pred_changed = int(str(silm_p.get("pred", "")).strip() != str(sp_p.get("pred", "")).strip())
        if pred_changed:
            changed_pred += 1
        else:
            unchanged_pred += 1
        if delta > 0:
            wins += 1
            pos_sum += delta
        elif delta < 0:
            losses += 1
            neg_sum += delta
        else:
            ties += 1
        paired_rows.append({
            "sample_id": sid,
            "sample_index": silm_p.get("sample_index", ""),
            "silm_score": silm_sc,
            "sp_score": sp_sc,
            "delta": round(delta, 4),
            "pred_changed": pred_changed,
            "silm_pred": silm_p.get("pred", ""),
            "sp_pred": sp_p.get("pred", ""),
            "answers": json.dumps(silm_p.get("answers", []), ensure_ascii=False),
        })
    paired_rows.sort(key=lambda r: -float(r["delta"]))

    mq_deltas = [float(r["delta"]) for r in paired_rows]
    hp_deltas = []
    for sid in silm_pred_hp:
        if sid not in sp_pred_hp:
            continue
        hp_deltas.append(
            float(sp_scores[(HPQA, sid)]["score"]) - float(silm_scores[(HPQA, sid)]["score"])
        )

    # Top-k contribution
    topk_rows = []
    for k in (1, 2, 5, 10):
        top_deltas = mq_deltas[:k] if len(mq_deltas) >= k else mq_deltas
        topk_rows.append({
            "k": k,
            "sum_topk_delta": round(sum(top_deltas), 4),
            "mean_topk_delta": round(statistics.mean(top_deltas), 4) if top_deltas else 0,
            "musique_mean_delta_after_remove_topk": round(remove_topk_robustness(mq_deltas, k), 4),
            "macro_delta_after_remove_topk_musique_only": round(
                (statistics.mean(hp_deltas) + remove_topk_robustness(mq_deltas, k)) / 2.0, 4
            ),
        })

    # --- Part 2: Significance ---
    mq_mean, mq_lo, mq_hi = paired_bootstrap_ci(mq_deltas, n_boot=args.bootstrap_n, seed=args.seed)
    macro_obs, macro_lo, macro_hi = bootstrap_macro_ci(
        hp_deltas, mq_deltas, n_boot=args.bootstrap_n, seed=args.seed + 1
    )
    n_pos = sum(1 for d in mq_deltas if d > 0)
    n_neg = sum(1 for d in mq_deltas if d < 0)
    sign_p = sign_test_p_value(n_pos, n_neg)
    perm_p_mq = permutation_test_mean(mq_deltas, n_perm=args.bootstrap_n, seed=args.seed + 2)
    all_400 = hp_deltas + mq_deltas
    perm_p_400 = permutation_test_mean(all_400, n_perm=args.bootstrap_n, seed=args.seed + 3)

    # --- Part 3: Active-set explanation (MuSiQue score-changed samples) ---
    silm_uasr = load_uasr_index(
        args.live_audit_dir / "run_sliminfer_original/uasr_score_dump.jsonl", TASK
    )
    sp_uasr = load_uasr_index(
        args.live_audit_dir / "run_sp_sliminfer/uasr_score_dump.jsonl", TASK
    )
    prop_rows = [
        r for r in read_jsonl(args.property_features)
        if str(r.get("task", "")) == TASK
    ]
    layer_index = build_block_layer_index(prop_rows)

    score_changed_sids = {r["sample_id"] for r in paired_rows if float(r["delta"]) != 0.0}
    pred_changed_sids = {r["sample_id"] for r in paired_rows if int(r["pred_changed"]) == 1}
    active_rows: List[Dict[str, Any]] = []
    for sid in sorted(score_changed_sids):
        active_rows.extend(
            analyze_active_set_sample(
                sid,
                silm_uasr=silm_uasr,
                sp_uasr=sp_uasr,
                layer_index=layer_index,
            )
        )

    # Aggregate active-set patterns
    def _avg(key: str, rows: Sequence[Mapping[str, Any]]) -> Optional[float]:
        vals = [float(r[key]) for r in rows if r.get(key) not in ("", None)]
        return round(statistics.mean(vals), 4) if vals else None

    active_summary = {
        "score_changed_samples": len(score_changed_sids),
        "pred_changed_samples": len(pred_changed_sids),
        "pred_changed_but_score_same": len(pred_changed_sids - score_changed_sids),
        "score_changed_layers": len(active_rows),
        "mean_churn_changed_layers": _avg("churn_rate", active_rows),
        "mean_removed_drift": _avg("removed_mean_drift", active_rows),
        "mean_removed_redundancy": _avg("removed_mean_redundancy", active_rows),
        "mean_removed_absorption": _avg("removed_mean_absorption", active_rows),
        "mean_added_persistence": _avg("added_mean_persistence", active_rows),
        "mean_added_relay": _avg("added_mean_relay", active_rows),
        "mean_added_drift": _avg("added_mean_drift", active_rows),
    }

    # --- Write outputs ---
    write_csv(out_dir / "musique_paired_samples.csv", paired_rows, [
        "sample_id", "sample_index", "silm_score", "sp_score", "delta", "pred_changed",
        "silm_pred", "sp_pred", "answers",
    ])
    write_csv(out_dir / "topk_gain_contribution.csv", topk_rows, [
        "k", "sum_topk_delta", "mean_topk_delta",
        "musique_mean_delta_after_remove_topk", "macro_delta_after_remove_topk_musique_only",
    ])
    write_csv(out_dir / "active_set_changed_layers.csv", active_rows, [
        "sample_id", "layer_id", "budget_k", "churn_rate", "added_count", "removed_count",
        "added_block_ids", "removed_block_ids",
        "removed_mean_drift", "removed_mean_redundancy", "removed_mean_absorption",
        "added_mean_persistence", "added_mean_relay", "added_mean_drift", "added_mean_redundancy",
    ])

    part1 = {
        "musique_n": len(paired_rows),
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "changed_predictions": changed_pred,
        "unchanged_predictions": unchanged_pred,
        "sum_positive_delta": round(pos_sum, 4),
        "sum_negative_delta": round(neg_sum, 4),
        "net_delta_mean": round(statistics.mean(mq_deltas), 4),
        "hotpotqa_unchanged_mean_delta": round(statistics.mean(hp_deltas), 4) if hp_deltas else 0,
    }

    part2 = {
        "musique_bootstrap_mean_delta": round(mq_mean, 4),
        "musique_bootstrap_95ci": [round(mq_lo, 4), round(mq_hi, 4)],
        "musique_ci_crosses_zero": bool(mq_lo <= 0 <= mq_hi),
        "musique_sign_test_p": round(sign_p, 6),
        "musique_permutation_p": round(perm_p_mq, 6),
        "macro_bootstrap_mean_delta": round(macro_obs, 4),
        "macro_bootstrap_95ci": [round(macro_lo, 4), round(macro_hi, 4)],
        "macro_ci_crosses_zero": bool(macro_lo <= 0 <= macro_hi),
        "all400_permutation_p": round(perm_p_400, 6),
        "sign_test_pos": n_pos,
        "sign_test_neg": n_neg,
    }

    summary = {"part1_win_loss": part1, "part2_significance": part2, "part3_active_set": active_summary}
    (out_dir / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    # Report markdown
    exploratory = part2["musique_ci_crosses_zero"] or part2["macro_ci_crosses_zero"]
    lines = [
        "# MuSiQue SP vs SILM — Post-hoc Analysis",
        "",
        "## 1. Win / Loss / Tie (MuSiQue, n=200)",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Wins (SP > SILM) | **{wins}** |",
        f"| Losses (SP < SILM) | **{losses}** |",
        f"| Ties | **{ties}** |",
        f"| Changed predictions | **{changed_pred}** |",
        f"| Unchanged predictions | **{unchanged_pred}** |",
        f"| Sum positive Δ | **+{pos_sum:.2f}** |",
        f"| Sum negative Δ | **{neg_sum:.2f}** |",
        f"| Net mean Δ (MuSiQue) | **{part1['net_delta_mean']:+.2f}** |",
        "",
        "### Top-k gain contribution & robustness",
        "",
        "| Remove top-k wins | Sum top-k Δ | MuSiQue mean Δ after remove | Macro Δ after remove (HP fixed) |",
        "|-------------------|-------------|----------------------------|--------------------------------|",
    ]
    for row in topk_rows:
        lines.append(
            f"| top-{row['k']} | {row['sum_topk_delta']} | {row['musique_mean_delta_after_remove_topk']} | "
            f"{row['macro_delta_after_remove_topk_musique_only']} |"
        )

    lines += [
        "",
        "## 2. Significance (paired bootstrap / permutation)",
        "",
        f"- MuSiQue mean Δ: **{part2['musique_bootstrap_mean_delta']:+.2f}**, 95% CI [{part2['musique_bootstrap_95ci'][0]:+.2f}, {part2['musique_bootstrap_95ci'][1]:+.2f}]",
        f"- CI crosses 0: **{part2['musique_ci_crosses_zero']}**",
        f"- Sign test (wins vs losses): p={part2['musique_sign_test_p']:.4f} ({n_pos}+ / {n_neg}-)",
        f"- Permutation test (MuSiQue mean Δ): p={part2['musique_permutation_p']:.4f}",
        f"- Macro (2-task) mean Δ: **{part2['macro_bootstrap_mean_delta']:+.2f}**, 95% CI [{part2['macro_bootstrap_95ci'][0]:+.2f}, {part2['macro_bootstrap_95ci'][1]:+.2f}]",
        f"- Macro CI crosses 0: **{part2['macro_ci_crosses_zero']}**",
        "",
        f"**Interpretation:** {'exploratory positive (CI includes 0)' if exploratory else 'CI excludes 0 — stronger evidence'}",
        "",
        "## 3. Active-set explanation (score-changed MuSiQue samples only)",
        "",
        f"- Score-changed samples: **{active_summary['score_changed_samples']}** / 200",
        f"- Pred-changed but score-same: **{active_summary['pred_changed_but_score_same']}**",
        f"- Changed prune layers (rows): **{active_summary['score_changed_layers']}**",
        f"- Mean churn on changed layers: **{active_summary['mean_churn_changed_layers']}**",
        "",
        "### Semantic proxy patterns (PROXY from property_features; gold NOT used in selection)",
        "",
        f"| Signal | Removed blocks (mean) | Added blocks (mean) |",
        f"|--------|----------------------|---------------------|",
        f"| semantic_drift | {active_summary['mean_removed_drift']} | {active_summary['mean_added_drift']} |",
        f"| active_set_redundancy | {active_summary['mean_removed_redundancy']} | — |",
        f"| semantic_absorption | {active_summary['mean_removed_absorption']} | — |",
        f"| semantic_persistence | — | {active_summary['mean_added_persistence']} |",
        f"| semantic_relay | — | {active_summary['mean_added_relay']} |",
        "",
        "## Claim tier",
        "- Gold answers used **only** for post-hoc score labeling, never for selection features",
        "- hidden_state trajectory: UNAVAILABLE",
    ]
    (out_dir / "MUSIQUE_SP_VS_SILM_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"\nWrote {out_dir / 'MUSIQUE_SP_VS_SILM_REPORT.md'}")


if __name__ == "__main__":
    main()
