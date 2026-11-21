"""Page-one summary figure: Qwen2.5-32B (AWQ), macro F1 vs. exact token budget on LongBench QA (1,150 instances).
Ours: Reranker + DedupPack. Baselines: the strongest 32B method of each family in Table 2
(chunk packing with the same reranker, sentence chunks, LlamaIndex auto-merging, two trained compressors,
LongLLMLingua) and the full-context reference. All numbers are the 32B columns of Table 2.
The four-model version is kept as fig_summary_4models.py."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

budgets = [512, 1024, 2048]
FULL = 47.79
OURS = ("Reranker + DedupPack (ours)", [44.05, 47.32, 50.47])
BASE = [  # label, F1 at 512 / 1,024 / 2,048, colour, marker, line style
    ("Reranker + chunk packing", [41.74, 45.09, 48.15], "#D55E00", "s", "--"),
    ("Sentence chunks + BGE", [39.89, 44.34, 48.62], "#E69F00", "^", "--"),
    ("RECOMP (trained)", [42.63, 45.93, 47.39], "#009E73", "D", "-."),
    ("Provence (trained)", [43.66, 45.66, 45.68], "#CC79A7", "v", "-."),
    ("LlamaIndex auto-merging", [35.11, 42.52, 47.12], "#8C6D31", "P", ":"),
    ("LongLLMLingua", [35.70, 41.88, 46.03], "#7F7F7F", "X", ":"),
]
AX = 0.5
plt.rcParams.update({"font.size": 7, "font.family": "serif", "axes.linewidth": AX,
                     "xtick.major.width": AX, "ytick.major.width": AX,
                     "xtick.major.size": 2, "ytick.major.size": 2,
                     "xtick.labelsize": 6.5, "ytick.labelsize": 6.5})
fig, ax = plt.subplots(figsize=(3.3, 2.75))
ax.axhline(FULL, color="#333333", ls=(0, (4, 2)), lw=0.9, zorder=1)
ax.text(460, FULL + 0.25, f"Full context (~12.5K tokens): {FULL:.2f}", fontsize=5.8, color="#333333", ha="left", va="bottom")
for lab, ys, c, mk, ls in BASE:
    ax.plot(budgets, ys, ls=ls, marker=mk, color=c, lw=1.0, ms=3.4, mew=0, label=lab, zorder=2)
lab, ys = OURS
ax.plot(budgets, ys, ls="-", marker="o", color="#0072B2", lw=2.0, ms=4.6, mew=0, label=lab, zorder=4)
# value labels; at 1,024 the label sits right of the point, between our line and the RECOMP line
for x, y, (dx, dy, ha, va) in zip(budgets, ys, ((1.0, 0.55, "center", "bottom"), (1.06, -0.12, "left", "center"), (1.0, 0.55, "center", "bottom"))):
    ax.text(x * dx, y + dy, f"{y:.2f}", fontsize=6, color="#0072B2", ha=ha, va=va, fontweight="bold", zorder=5)
# gap to the full context at 2,048 tokens (one sixth of the average document)
xa = 2048 * 1.13
ax.annotate("", xy=(xa, ys[2]), xytext=(xa, FULL), arrowprops=dict(arrowstyle="<->", color="#0072B2", lw=0.9, shrinkA=0, shrinkB=0))
ax.text(xa * 1.04, (ys[2] + FULL) / 2, "+2.7 F1\nwith 1/6\nof the input", fontsize=5.8, color="#0072B2", ha="left", va="center", fontweight="bold")
ax.set_xscale("log", base=2)
ax.set_xticks(budgets)
ax.set_xticklabels(["512", "1,024", "2,048"])
ax.minorticks_off()
ax.set_xlim(440, 3300)
ax.set_ylim(34.2, 52.2)
ax.set_xlabel("Token budget", fontsize=7, labelpad=1)
ax.set_ylabel("Macro F1 (Qwen2.5-32B)", fontsize=7, labelpad=1)
ax.grid(alpha=0.25, lw=0.3)
ax.spines[["top", "right"]].set_visible(False)
h, l = ax.get_legend_handles_labels()
order = [len(h) - 1] + list(range(len(h) - 1))      # ours first
fig.legend([h[i] for i in order], [l[i] for i in order], fontsize=5.9, loc="lower center", ncol=2,
           frameon=False, handlelength=2.4, columnspacing=1.0, labelspacing=0.35)
fig.tight_layout(pad=0.25, rect=(0, 0.21, 1, 1))
fig.savefig("fig_summary.pdf")
fig.savefig("fig_summary.png", dpi=300)
print("ok")
