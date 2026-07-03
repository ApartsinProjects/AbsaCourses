"""N5: re-render Figures A2 and A3 with larger, legible fonts and clean styling.

Reviewer h7LN flagged the formatting of these appendix figures. Data are taken
verbatim from the existing summary tables; only presentation is refined.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "paper/outputs/figures"
NAVY = "#14385c"
TERRA = "#c1663f"
plt.rcParams.update({
    "font.size": 15, "font.family": "DejaVu Sans", "axes.edgecolor": "#5a626c",
    "axes.linewidth": 1.0, "svg.fonttype": "none",
})


def fig_a2():
    counts = {1: 9, 2: 13, 3: 3}
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    xs = list(counts.keys())
    ys = [counts[x] for x in xs]
    bars = ax.bar(xs, ys, width=0.6, color=NAVY, edgecolor="white")
    for b, y in zip(bars, ys):
        ax.text(b.get_x() + b.get_width() / 2, y + 0.25, str(y),
                ha="center", va="bottom", fontsize=15, fontweight="bold", color=NAVY)
    ax.set_xlabel("Target aspects per review", fontsize=15)
    ax.set_ylabel("Review count", fontsize=15)
    ax.set_xticks(xs)
    ax.set_ylim(0, max(ys) + 2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=14)
    fig.tight_layout()
    fig.savefig(FIG / "canary_smoke_aspect_count_distribution.svg", format="svg")
    plt.close(fig)


def fig_a3():
    rows = [
        ("TF-IDF", 0.23530, 0.00576),
        ("DistilBERT", 0.26940, 0.00047),
        ("BERT", 0.27906, 0.01403),
    ]
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    labels = [r[0] for r in rows]
    means = [r[1] for r in rows]
    errs = [r[2] for r in rows]
    xs = range(len(rows))
    colors = ["#8a94a6", TERRA, NAVY]
    bars = ax.bar(xs, means, yerr=errs, width=0.6, color=colors, edgecolor="white",
                  capsize=8, error_kw={"elinewidth": 1.6, "ecolor": "#2c3138"})
    for i, (m, e) in enumerate(zip(means, errs)):
        ax.text(i, m + e + 0.006, f"{m:.3f}", ha="center", va="bottom",
                fontsize=14, fontweight="bold", color="#2c3138")
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels, fontsize=15)
    ax.set_ylabel("Micro-F1 (mean over 3 seeds)", fontsize=15)
    ax.set_ylim(0, 0.34)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=14)
    ax.grid(axis="y", color="#e4e2dc", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIG / "seed_stability_micro_f1.svg", format="svg")
    plt.close(fig)


if __name__ == "__main__":
    fig_a2()
    fig_a3()
    print("re-rendered A2 + A3")
