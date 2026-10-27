#!/usr/bin/env python3
"""CPU-only complementarity upper-bound audit: CC vs DRM-ACC.

No GPU. Uses Phase-3 scores + DRM selection_audit runtime signals.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PHASE3 = REPO_ROOT / "results/acc_drm_phase3_6task_20260714_094058"
TASKS = (
    "qasper",
    "multifieldqa_en",
    "hotpotqa",
    "2wikimqa",
    "musique",
    "triviaqa",
)


def parse_args() -> argparse.Namespace:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase3_dir", type=Path, default=DEFAULT_PHASE3)
    p.add_argument(
        "--out_dir",
        type=Path,
        default=REPO_ROOT / f"results/cc_drm_complementarity_oracle_{ts}",
    )
    return p.parse_args()


def mean(xs: Sequence[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else float("nan")


def safe_mean(xs: Sequence[Optional[float]]) -> float:
    vals = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    return mean(vals)


def write_csv(path: Path, rows: List[Dict[str, Any]], fields: Optional[List[str]] = None) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = fields or list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def load_scores(phase3: Path) -> Dict[Tuple[str, str], Dict[str, float]]:
    """(task, sample_id) -> method -> score"""
    out: Dict[Tuple[str, str], Dict[str, float]] = defaultdict(dict)
    with (phase3 / "scores_by_sample.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["task"], str(row["sample_id"]))
            out[key][row["method"]] = float(row["score"])
            out[key]["_sample_index"] = float(row.get("sample_index") or -1)
    return out


def largest_contig_segments(ids: Iterable[int]) -> List[Tuple[int, int]]:
    xs = sorted(set(int(x) for x in ids))
    if not xs:
        return []
    segs: List[Tuple[int, int]] = []
    a = b = xs[0]
    for x in xs[1:]:
        if x == b + 1:
            b = x
        else:
            segs.append((a, b))
            a = b = x
    segs.append((a, b))
    segs.sort(key=lambda ab: -(ab[1] - ab[0] + 1))
    return segs


def in_or_near_main_cluster(bid: int, selected: Sequence[int], halo: int = 2) -> bool:
    segs = largest_contig_segments(selected)
    if not segs:
        return False
    a, b = segs[0]
    return (a - halo) <= int(bid) <= (b + halo)


def aggregate_drm_audit(audit_path: Path) -> Dict[Tuple[str, int], Dict[str, Any]]:
    """Aggregate layer audits to (task, sample_index)."""
    buckets: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
    with audit_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            task = str(r.get("task") or "")
            si = int(r.get("sample_index") if r.get("sample_index") is not None else -1)
            if si < 0 or not task:
                continue
            buckets[(task, si)].append(r)

    out: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for key, rows in buckets.items():
        feats_list = []
        topo_list = []
        swaps = []
        eligible = []
        cap = []
        multi = []
        locality = []
        second = []
        gap = []
        sac_main = 0
        sac_total = 0
        rescue_margins = []  # proxy: accepted / max(eligible,1) averaged; also churn_vs_acc
        churn_acc = []
        for r in rows:
            acc = r.get("acc") if isinstance(r.get("acc"), dict) else {}
            feats = acc.get("features") if isinstance(acc.get("features"), dict) else {}
            topo = acc.get("cluster_topology") if isinstance(acc.get("cluster_topology"), dict) else {}
            feats_list.append(feats)
            topo_list.append(topo)
            sw = int(r.get("drm_swap_accepted") or r.get("accepted_swaps_per_layer") or 0)
            el = int(r.get("drm_candidate_eligible") or 0)
            swaps.append(sw)
            eligible.append(el)
            cap.append(int(r.get("cluster_budget_cap_triggered") or acc.get("cluster_budget_cap_triggered") or 0))
            multi.append(int(r.get("multi_locus_triggered") or acc.get("multi_locus_triggered") or 0))
            locality.append(float(feats.get("active_set_locality") or 0.0))
            second.append(
                float(
                    feats.get("second_cluster_mass_ratio")
                    or topo.get("second_over_first")
                    or 0.0
                )
            )
            gap.append(
                float(feats.get("cluster_gap_ratio") or topo.get("inter_cluster_gap_ratio") or 0.0)
            )
            churn_acc.append(float(r.get("churn_vs_acc") or 0.0))
            # rescue margin proxy: how "selective" acceptance was among eligible
            if el > 0:
                rescue_margins.append(float(sw) / float(el))
            selected = r.get("selected_block_ids") or []
            # Approximations of ACC-before-swap: selected - rescued + sacrifice
            rescued = set(int(x) for x in (r.get("rescued_ids") or []))
            sacr = [int(x) for x in (r.get("sacrifice_used") or [])]
            # reconstruct pre-swap active ≈ (selected - rescued) ∪ sacrifice
            pre = (set(int(x) for x in selected) - rescued) | set(sacr)
            if not pre:
                pre = set(int(x) for x in selected)
            for bid in sacr:
                sac_total += 1
                if in_or_near_main_cluster(bid, pre, halo=2):
                    sac_main += 1

        out[key] = {
            "n_layers": len(rows),
            "drm_swaps_total": int(sum(swaps)),
            "drm_swaps_mean": mean(swaps),
            "drm_eligible_mean": mean([float(x) for x in eligible]),
            "cap_trigger_rate": mean([float(x) for x in cap]),
            "multi_locus_rate": mean([float(x) for x in multi]),
            "second_locus_strength": mean(second),
            "inter_cluster_distance": mean(gap),
            "active_set_locality": mean(locality),
            "rescue_margin_proxy": mean(rescue_margins) if rescue_margins else 0.0,
            "churn_vs_acc_mean": mean(churn_acc),
            "sacrifice_main_cluster_count": sac_main,
            "sacrifice_total": sac_total,
            "sacrifice_main_cluster_frac": (sac_main / sac_total) if sac_total else 0.0,
            "sample_id_audit": rows[0].get("sample_id"),
        }
    return out


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    n = len(xs)
    if n < 3:
        return float("nan")

    def rank(vals: Sequence[float]) -> List[float]:
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        ranks = [0.0] * len(vals)
        i = 0
        while i < len(vals):
            j = i
            while j + 1 < len(vals) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg
            i = j + 1
        return ranks

    rx, ry = rank(xs), rank(ys)
    mx, my = mean(rx), mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    denx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    deny = math.sqrt(sum((b - my) ** 2 for b in ry))
    if denx < 1e-12 or deny < 1e-12:
        return float("nan")
    return num / (denx * deny)


def cohen_d(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    ma, mb = mean(a), mean(b)
    va = statistics.pvariance(a)
    vb = statistics.pvariance(b)
    pooled = math.sqrt((va + vb) / 2.0)
    if pooled < 1e-12:
        return 0.0
    return (ma - mb) / pooled


def group_signal_table(
    rows: List[Dict[str, Any]],
    signal: str,
    outcome_key: str = "outcome",
) -> Dict[str, Any]:
    wins = [float(r[signal]) for r in rows if r[outcome_key] == "drm_win" and math.isfinite(float(r[signal]))]
    losses = [float(r[signal]) for r in rows if r[outcome_key] == "cc_win" and math.isfinite(float(r[signal]))]
    ties = [float(r[signal]) for r in rows if r[outcome_key] == "tie" and math.isfinite(float(r[signal]))]
    return {
        "signal": signal,
        "win_mean": round(mean(wins), 6) if wins else None,
        "loss_mean": round(mean(losses), 6) if losses else None,
        "tie_mean": round(mean(ties), 6) if ties else None,
        "n_win": len(wins),
        "n_loss": len(losses),
        "cohen_d_win_vs_loss": round(cohen_d(wins, losses), 4) if wins and losses else None,
        "separable": bool(
            wins
            and losses
            and abs(cohen_d(wins, losses)) >= 0.35
            and math.isfinite(cohen_d(wins, losses))
        ),
    }


def main() -> int:
    args = parse_args()
    phase3 = args.phase3_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    scores = load_scores(phase3)
    audit = aggregate_drm_audit(phase3 / "run_drm_acc_swap_sliminfer_v1" / "selection_audit.jsonl")

    # Also index audit by sample_id when available.
    audit_by_sid: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for (task, si), agg in audit.items():
        sid = str(agg.get("sample_id_audit") or "")
        if sid:
            audit_by_sid[(task, sid)] = agg

    paired_rows: List[Dict[str, Any]] = []
    missing_audit = 0
    for (task, sid), m in scores.items():
        if "cc_sliminfer_v1" not in m or "drm_acc_swap_sliminfer_v1" not in m:
            continue
        cc = float(m["cc_sliminfer_v1"])
        drm = float(m["drm_acc_swap_sliminfer_v1"])
        orig = float(m.get("sliminfer_original", float("nan")))
        oracle = max(cc, drm)
        if drm > cc + 1e-12:
            outcome = "drm_win"
        elif cc > drm + 1e-12:
            outcome = "cc_win"
        else:
            outcome = "tie"
        si = int(m.get("_sample_index", -1))
        sig = audit_by_sid.get((task, sid)) or audit.get((task, si))
        if sig is None:
            missing_audit += 1
            sig = {}
        paired_rows.append(
            {
                "task": task,
                "sample_id": sid,
                "sample_index": si,
                "score_cc": cc,
                "score_drm_acc": drm,
                "score_original": orig,
                "score_oracle": oracle,
                "delta_drm_vs_cc": round(drm - cc, 6),
                "delta_oracle_vs_cc": round(oracle - cc, 6),
                "outcome": outcome,
                "drm_swaps_total": sig.get("drm_swaps_total"),
                "drm_swaps_mean": sig.get("drm_swaps_mean"),
                "second_locus_strength": sig.get("second_locus_strength"),
                "inter_cluster_distance": sig.get("inter_cluster_distance"),
                "rescue_margin_proxy": sig.get("rescue_margin_proxy"),
                "cap_trigger_rate": sig.get("cap_trigger_rate"),
                "multi_locus_rate": sig.get("multi_locus_rate"),
                "active_set_locality": sig.get("active_set_locality"),
                "churn_vs_acc_mean": sig.get("churn_vs_acc_mean"),
                "sacrifice_main_cluster_frac": sig.get("sacrifice_main_cluster_frac"),
                "sacrifice_main_cluster_count": sig.get("sacrifice_main_cluster_count"),
                "sacrifice_total": sig.get("sacrifice_total"),
                "has_audit": int(bool(sig)),
            }
        )

    write_csv(out_dir / "paired_sample_scores.csv", paired_rows)

    def macro_map(field: str) -> Tuple[float, Dict[str, float]]:
        per = {}
        for t in TASKS:
            vals = [float(r[field]) for r in paired_rows if r["task"] == t]
            per[t] = round(mean(vals), 4)
        return round(mean(list(per.values())), 4), per

    cc_macro, cc_per = macro_map("score_cc")
    drm_macro, drm_per = macro_map("score_drm_acc")
    ora_macro, ora_per = macro_map("score_oracle")
    oracle_delta = round(ora_macro - cc_macro, 4)
    drm_delta = round(drm_macro - cc_macro, 4)

    # Outcome counts
    outcome_counts = defaultdict(int)
    outcome_by_task: Dict[str, Dict[str, int]] = {t: defaultdict(int) for t in TASKS}
    for r in paired_rows:
        outcome_counts[r["outcome"]] += 1
        outcome_by_task[r["task"]][r["outcome"]] += 1

    wlt_rows = []
    for t in TASKS:
        d = outcome_by_task[t]
        n = sum(d.values()) or 1
        wlt_rows.append(
            {
                "task": t,
                "drm_wins": d.get("drm_win", 0),
                "cc_wins": d.get("cc_win", 0),
                "ties": d.get("tie", 0),
                "drm_win_rate": round(d.get("drm_win", 0) / n, 4),
                "cc_mean": cc_per[t],
                "drm_mean": drm_per[t],
                "oracle_mean": ora_per[t],
                "oracle_delta_vs_cc": round(ora_per[t] - cc_per[t], 4),
            }
        )
    wlt_rows.append(
        {
            "task": "MACRO",
            "drm_wins": outcome_counts.get("drm_win", 0),
            "cc_wins": outcome_counts.get("cc_win", 0),
            "ties": outcome_counts.get("tie", 0),
            "drm_win_rate": round(
                outcome_counts.get("drm_win", 0) / max(1, sum(outcome_counts.values())), 4
            ),
            "cc_mean": cc_macro,
            "drm_mean": drm_macro,
            "oracle_mean": ora_macro,
            "oracle_delta_vs_cc": oracle_delta,
        }
    )
    write_csv(out_dir / "win_loss_oracle_by_task.csv", wlt_rows)

    # Signal separability (wins vs losses), gold-free only
    signals = [
        "second_locus_strength",
        "inter_cluster_distance",
        "drm_swaps_total",
        "rescue_margin_proxy",
        "cap_trigger_rate",
        "active_set_locality",
        "sacrifice_main_cluster_frac",
        "multi_locus_rate",
        "churn_vs_acc_mean",
    ]
    # Only rows with audit
    audited = [r for r in paired_rows if int(r.get("has_audit") or 0) == 1]
    signal_rows = [group_signal_table(audited, s) for s in signals]
    write_csv(out_dir / "signal_separability_win_vs_loss.csv", signal_rows)

    # Task-name only separability baseline: drm_win_rate by task variance
    task_rates = [float(r["drm_win_rate"]) for r in wlt_rows if r["task"] != "MACRO"]
    task_name_effect = {
        "drm_win_rate_by_task": {r["task"]: r["drm_win_rate"] for r in wlt_rows if r["task"] != "MACRO"},
        "task_rate_range": round(max(task_rates) - min(task_rates), 4) if task_rates else None,
        "note": "If only task rates differ and unlabeled signals are weak, method is not general.",
    }

    # MuSiQue loss deep dive
    mus_loss = [r for r in audited if r["task"] == "musique" and r["outcome"] == "cc_win"]
    mus_win = [r for r in audited if r["task"] == "musique" and r["outcome"] == "drm_win"]
    mus_rows = []
    for label, group in (("musique_cc_wins_drm_losses", mus_loss), ("musique_drm_wins", mus_win)):
        mus_rows.append(
            {
                "group": label,
                "n": len(group),
                "mean_delta_drm_vs_cc": round(mean([float(r["delta_drm_vs_cc"]) for r in group]), 4)
                if group
                else None,
                "mean_drm_swaps_total": round(safe_mean([r.get("drm_swaps_total") for r in group]), 4),
                "mean_sacrifice_main_cluster_frac": round(
                    safe_mean([r.get("sacrifice_main_cluster_frac") for r in group]), 4
                ),
                "mean_active_set_locality": round(safe_mean([r.get("active_set_locality") for r in group]), 4),
                "mean_second_locus_strength": round(
                    safe_mean([r.get("second_locus_strength") for r in group]), 4
                ),
                "mean_inter_cluster_distance": round(
                    safe_mean([r.get("inter_cluster_distance") for r in group]), 4
                ),
                "frac_any_main_cluster_sacrifice": round(
                    mean([1.0 if float(r.get("sacrifice_main_cluster_count") or 0) > 0 else 0.0 for r in group]),
                    4,
                )
                if group
                else None,
            }
        )
    write_csv(out_dir / "musique_loss_vs_win_signals.csv", mus_rows)

    # Correlation of delta with signals among audited
    corr_rows = []
    deltas = [float(r["delta_drm_vs_cc"]) for r in audited]
    for s in signals:
        xs = [float(r[s]) for r in audited if r.get(s) is not None]
        ys = [float(r["delta_drm_vs_cc"]) for r in audited if r.get(s) is not None]
        if len(xs) != len(ys):
            # align
            pairs = [
                (float(r[s]), float(r["delta_drm_vs_cc"]))
                for r in audited
                if r.get(s) is not None and math.isfinite(float(r[s]))
            ]
            xs = [a for a, _ in pairs]
            ys = [b for _, b in pairs]
        corr_rows.append(
            {
                "signal": s,
                "spearman_vs_delta_drm_cc": round(spearman(xs, ys), 4) if len(xs) >= 3 else None,
                "n": len(xs),
            }
        )
    write_csv(out_dir / "signal_correlation_vs_delta.csv", corr_rows)

    n_sep = sum(1 for r in signal_rows if r.get("separable"))
    # Decision
    if oracle_delta < 0.20:
        decision = "STOP_WEAK_COMPLEMENTARITY"
        reason = (
            f"Oracle vs CC macro Δ={oracle_delta} < +0.20 — complementarity ceiling too weak; "
            "do not build Selective DRM Rescue."
        )
    elif oracle_delta >= 0.50 and n_sep >= 2:
        decision = "GO_SELECTIVE_DRM_RESCUE"
        reason = (
            f"Oracle vs CC macro Δ={oracle_delta} ≥ +0.50 and {n_sep} unlabeled signals "
            "separate wins/losses (|d|≥0.35) — Selective DRM Rescue may be justified."
        )
    elif oracle_delta >= 0.50 and n_sep < 2:
        # check if task-name dominates
        if (task_name_effect.get("task_rate_range") or 0) >= 0.15:
            decision = "STOP_TASK_ONLY_SEPARATION"
            reason = (
                f"Oracle Δ={oracle_delta} is large, but unlabeled signals weak "
                f"(separable={n_sep}) while task win-rate range="
                f"{task_name_effect.get('task_rate_range')} — not worth a general method."
            )
        else:
            decision = "BORDERLINE_NEED_BETTER_FEATURES"
            reason = (
                f"Oracle Δ={oracle_delta} large but current unlabeled signals poorly separate "
                f"wins/losses (separable={n_sep})."
            )
    else:
        # 0.2 <= oracle < 0.5
        if n_sep >= 2 and (task_name_effect.get("task_rate_range") or 0) < 0.20:
            decision = "MAYBE_SELECTIVE_IF_CHEAP"
            reason = (
                f"Oracle Δ={oracle_delta} moderate (+0.2..+0.5) with some unlabeled separation "
                f"(separable={n_sep}). Only pursue selective DRM if cheap; not strong enough for default on."
            )
        else:
            decision = "STOP_WEAK_OR_TASKY"
            reason = (
                f"Oracle Δ={oracle_delta} moderate/weak and separation insufficient or task-dominated."
            )

    # MuSiQue hypothesis
    mus_hyp = {}
    if mus_loss and mus_win:
        mus_hyp = {
            "loss_main_cluster_sac_frac": mus_rows[0].get("mean_sacrifice_main_cluster_frac"),
            "win_main_cluster_sac_frac": mus_rows[1].get("mean_sacrifice_main_cluster_frac"),
            "loss_frac_any_main_sac": mus_rows[0].get("frac_any_main_cluster_sacrifice"),
            "win_frac_any_main_sac": mus_rows[1].get("frac_any_main_cluster_sacrifice"),
            "supports_main_cluster_damage": bool(
                (mus_rows[0].get("mean_sacrifice_main_cluster_frac") or 0)
                > (mus_rows[1].get("mean_sacrifice_main_cluster_frac") or 0) + 0.05
            ),
        }

    decision_obj = {
        "decision": decision,
        "reason": reason,
        "n_paired_samples": len(paired_rows),
        "n_with_audit": len(audited),
        "missing_audit": missing_audit,
        "cc_macro": cc_macro,
        "drm_acc_macro": drm_macro,
        "oracle_macro": ora_macro,
        "drm_vs_cc_macro": drm_delta,
        "oracle_vs_cc_macro": oracle_delta,
        "thresholds": {
            "stop_if_oracle_lt": 0.20,
            "go_if_oracle_ge": 0.50,
            "cohen_d_separable": 0.35,
        },
        "n_separable_unlabeled_signals": n_sep,
        "task_name_effect": task_name_effect,
        "outcome_counts": dict(outcome_counts),
        "musique_hypothesis": mus_hyp,
        "do_not_tune_drm_strength": True,
        "phase3_dir": str(phase3),
    }
    (out_dir / "COMPLEMENTARITY_DECISION.json").write_text(json.dumps(decision_obj, indent=2) + "\n")

    # Markdown report
    md = [
        "# CC × DRM-ACC Complementarity Oracle Audit (CPU-only)\n\n",
        f"- Phase3: `{phase3}`\n",
        f"- Paired samples: **{len(paired_rows)}**; with DRM audit: **{len(audited)}**\n\n",
        "## Macro ceiling\n\n",
        f"| Method | Macro |\n|--------|------:|\n",
        f"| CC | {cc_macro} |\n",
        f"| DRM-ACC | {drm_macro} (Δ vs CC {drm_delta}) |\n",
        f"| **Oracle max(CC, DRM-ACC)** | **{ora_macro}** (Δ vs CC **{oracle_delta}**) |\n\n",
        f"**Decision: `{decision}`**\n\n{reason}\n\n",
        "## Win / loss / tie\n\n",
        f"- DRM wins: {outcome_counts.get('drm_win',0)}\n",
        f"- CC wins: {outcome_counts.get('cc_win',0)}\n",
        f"- Ties: {outcome_counts.get('tie',0)}\n\n",
        "## Unlabeled signal separability (|Cohen's d|≥0.35 win vs loss)\n\n",
        "```json\n" + json.dumps(signal_rows, indent=2) + "\n```\n\n",
        "## MuSiQue losses vs main-cluster sacrifice\n\n",
        "```json\n" + json.dumps(mus_rows, indent=2) + "\n```\n\n",
        "```json\n" + json.dumps(mus_hyp, indent=2) + "\n```\n\n",
        "## Task-name baseline\n\n",
        "```json\n" + json.dumps(task_name_effect, indent=2) + "\n```\n",
    ]
    (out_dir / "COMPLEMENTARITY_ORACLE_REPORT.md").write_text("".join(md), encoding="utf-8")

    print(json.dumps(decision_obj, indent=2))
    print(f"out_dir={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
