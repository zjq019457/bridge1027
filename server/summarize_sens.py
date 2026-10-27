"""Summarize sensitivity runs: macro F1, paired bootstrap vs default, packing statistics."""
import json
import random
import statistics
import sys
from pathlib import Path

TASKS = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", "narrativeqa"]
B = 10000
SEED = 20260701
BUDGET = 1024

root = Path(sys.argv[1])  # e.g. runs/qwen
holm_family = sys.argv[2].split(",") if len(sys.argv) > 2 else None


def load(run):
    preds = {}
    for line in (run / "raw_predictions.jsonl").open():
        r = json.loads(line)
        preds[(r["task"], str(r["sample_id"]))] = r
    traces = {}
    for line in (run / "packing_trace.jsonl").open():
        t = json.loads(line)
        traces[(t["task"], str(t["sample_id"]))] = t
    return preds, traces


def macro(run):
    # official aggregate written by the runner's LongBench scorer
    s = json.loads((run / "macro_summary.json").read_text())
    return s["macro"], s["task_scores"]


def paired(base, other):
    rng = random.Random(SEED)
    diffs = {t: [] for t in TASKS}
    for key, r in base.items():
        diffs[key[0]].append(other[key]["score"] - r["score"])  # per-sample scores are already 0-100
    point = sum(sum(v) / len(v) for v in diffs.values()) / len(TASKS)
    boots = []
    for _ in range(B):
        tot = 0.0
        for v in diffs.values():
            n = len(v)
            tot += sum(v[rng.randrange(n)] for _ in range(n)) / n
        boots.append(tot / len(TASKS))
    boots.sort()
    lo, hi = boots[int(0.025 * B)], boots[int(0.975 * B) - 1]
    # two-sided bootstrap p-value for H0: delta == 0
    p = 2 * min(sum(b <= 0 for b in boots), sum(b >= 0 for b in boots)) / B
    return point, lo, hi, min(1.0, p)


def pack_stats(traces):
    lens = sorted(t["model_token_count"] for t in traces.values())
    n = len(lens)
    regions = [len(t["selected_span_ids"]) for t in traces.values()]
    dropped = [len(set(t.get("frozen_span_ids", [])) - set(t["selected_span_ids"])) for t in traces.values()]
    return {
        "n": n,
        "len_mean": statistics.mean(lens),
        "len_p50": lens[n // 2],
        "len_p95": lens[int(0.95 * n) - 1],
        "len_max": lens[-1],
        "over_budget": sum(l > BUDGET for l in lens),
        "regions_mean": statistics.mean(regions),
        "truncated_rate": sum(bool(t.get("truncated")) for t in traces.values()) / n,
        "drop_any_rate": sum(d > 0 for d in dropped) / n,
        "dropped_mean": statistics.mean(dropped),
    }


runs = sorted(p.parent for p in root.glob("*/DONE"))
data = {r.name: load(r) for r in runs}
base_preds = data["default"][0]
rows = []
for name, (preds, traces) in data.items():
    m, per = macro(root / name)
    row = {"config": name, "params": (root / name / "SENS_PARAMS.json").read_text().strip(), "macro": m}
    row.update({f"f1_{t}": per[t] for t in TASKS})
    row.update(pack_stats(traces))
    if name != "default":
        d, lo, hi, p = paired(base_preds, preds)
        row.update({"delta_vs_default": d, "ci_lo": lo, "ci_hi": hi, "p_boot": p})
    rows.append(row)

# Holm correction over the non-default comparisons
cmp_rows = [r for r in rows if "p_boot" in r and (holm_family is None or r["config"] in holm_family)]
cmp_rows.sort(key=lambda r: r["p_boot"])
k = len(cmp_rows)
running = 0.0
for i, r in enumerate(cmp_rows):
    running = max(running, min(1.0, (k - i) * r["p_boot"]))
    r["p_holm"] = running

out = root / "SENSITIVITY_SUMMARY.json"
out.write_text(json.dumps(rows, indent=2))
keys = ["config", "macro", "delta_vs_default", "ci_lo", "ci_hi", "p_holm", "len_mean", "len_p95", "over_budget",
        "regions_mean", "truncated_rate", "drop_any_rate", "dropped_mean"]
with (root / "SENSITIVITY_SUMMARY.csv").open("w") as f:
    f.write(",".join(keys + [f"f1_{t}" for t in TASKS]) + "\n")
    for r in rows:
        f.write(",".join(str(r.get(c, "")) for c in keys + [f"f1_{t}" for t in TASKS]) + "\n")
for r in rows:
    print(f"{r['config']:12s} macro={r['macro']:.3f} "
          + (f"d={r['delta_vs_default']:+.3f} [{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}] holm_p={r.get('p_holm', float('nan')):.4f} " if 'delta_vs_default' in r else "")
          + f"len={r['len_mean']:.1f}/p95 {r['len_p95']} over={r['over_budget']} regions={r['regions_mean']:.2f} "
          + f"trunc={r['truncated_rate']:.3f} drop_any={r['drop_any_rate']:.3f}")
