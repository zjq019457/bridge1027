"""Page-one summary figure: macro F1 vs. token budget per model (LongBench QA, 1,150 instances)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

budgets = [512, 1024, 2048]
# per model: method -> F1 at 512 / 1,024 / 2,048 tokens; "full" is the full-context reference
D = {
    "Qwen2.5-7B": {"rr": [39.56, 42.79, 44.96], "rrc": [37.19, 41.59, 43.48], "bge": [35.64, 40.76, 43.14], "chunk": [33.05, 38.85, 40.98], "full": 43.04},
    "Llama-3.1-8B": {"rr": [40.49, 44.13, 45.61], "rrc": [38.29, 42.22, 43.72], "bge": [36.16, 41.67, 43.58], "chunk": [33.27, 38.99, 40.85], "full": 43.13},
    "Qwen2.5-14B": {"rr": [42.85, 46.43, 49.54], "rrc": [41.18, 44.37, 47.30], "bge": [39.21, 45.18, 48.10], "chunk": [36.27, 42.34, 45.29], "full": 47.59},
    "Qwen2.5-32B (AWQ)": {"rr": [44.05, 47.32, 50.47], "rrc": [41.74, 45.09, 48.15], "bge": [40.34, 46.28, 49.01], "chunk": [37.52, 42.40, 46.08], "full": 47.79},
}
M = [  # key, label, colour, marker, line width, line style
    ("rr", "Reranker + DedupPack", "#1f5fa8", "o", 1.3, "-"),
    ("rrc", "Reranker + chunk packing", "#b35806", "s", 1.0, "--"),
    ("bge", "BGE-small + DedupPack", "#5aa0e0", "^", 1.0, "-"),
    ("chunk", "BGE-small + chunk packing", "#f1a340", "D", 0.9, "--"),
]
AX = 0.4  # thin axes
plt.rcParams.update({"font.size": 7, "font.family": "serif", "axes.linewidth": AX,
                     "xtick.major.width": AX, "ytick.major.width": AX,
                     "xtick.major.size": 2, "ytick.major.size": 2,
                     "xtick.labelsize": 6, "ytick.labelsize": 6})
fig, axes = plt.subplots(2, 2, figsize=(3.3, 3.0))
for ax, (title, d) in zip(axes.flat, D.items()):
    ax.axhline(d["full"], color="#444444", ls="--", lw=0.8)
    ax.text(512, d["full"] + 0.3, "Full context", fontsize=5.5, color="#444444", ha="left", va="bottom")
    for k, lab, c, mk, lw, ls in M:
        ax.plot(budgets, d[k], ls=ls, marker=mk, color=c, lw=lw, ms=2.6, label=lab, mew=0)
    ax.set_xscale("log", base=2)
    ax.set_xticks(budgets)
    ax.set_xticklabels(["512", "1K", "2K"])
    ax.minorticks_off()
    ax.set_title(title, fontsize=6.5, pad=2)
    ax.grid(alpha=0.25, lw=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    lo = min(min(d[k]) for k, *_ in M) - 1
    ax.set_ylim(lo, max(max(d["rr"]), d["full"]) + 1.5)
for ax in axes[1]:
    ax.set_xlabel("Token budget", fontsize=6.5, labelpad=1)
for ax in axes[:, 0]:
    ax.set_ylabel("Macro F1", fontsize=6.5, labelpad=1)
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, fontsize=5.8, loc="lower center", ncol=2, frameon=False, handlelength=2.2, columnspacing=1.2)
fig.tight_layout(pad=0.25, w_pad=0.5, h_pad=0.4, rect=(0, 0.09, 1, 1))
fig.savefig("fig_summary.pdf")
fig.savefig("fig_summary.png", dpi=250)
print("ok")
