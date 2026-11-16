"""Page-one summary figure: macro F1 vs. token budget (LongBench QA, 1,150 instances)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

budgets = [512, 1024, 2048]
panels = [
    ("Avg. of 7B / 8B / 14B models",
     [37.87, 41.98, 44.63], [34.20, 40.06, 42.37], 44.59, 12.5),
    ("Qwen2.5-32B",
     [40.88, 45.07, 49.15], [37.52, 42.40, 46.08], 47.79, None),
]
plt.rcParams.update({"font.size": 8, "font.family": "serif"})
fig, axes = plt.subplots(1, 2, figsize=(3.3, 1.95), sharey=False)
for ax, (title, dp, ch, full, _) in zip(axes, panels):
    ax.axhline(full, color="#555555", ls="--", lw=1)
    ax.text(512, full + 0.35, "Full context", fontsize=6, color="#555555")
    ax.plot(budgets, dp, "-o", color="#1f6fb4", lw=1.6, ms=3.5, label="DedupPack")
    ax.plot(budgets, ch, "-s", color="#d9822b", lw=1.2, ms=3, label="BGE chunk packing")
    ax.set_xscale("log", base=2)
    ax.set_xticks(budgets)
    ax.set_xticklabels(["512", "1K", "2K"])
    ax.set_title(title, fontsize=6.5)
    ax.set_xlabel("Token budget", fontsize=7)
    ax.grid(alpha=0.25, lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    lo = min(ch) - 1.5
    ax.set_ylim(lo, max(max(dp), full) + 2)
axes[0].set_ylabel("Macro F1", fontsize=7)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, fontsize=6.5, loc="lower center", ncol=2, frameon=False)
fig.tight_layout(pad=0.3, w_pad=0.6, rect=(0, 0.08, 1, 1))
fig.savefig("fig_summary.pdf")
print("ok")
