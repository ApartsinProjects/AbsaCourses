"""Build the final 2-panel headline figure for §6.7D.

Panel A: sent_mse by bucket (lower is better), with 95% bootstrap CI.
Panel B: macro_balanced_accuracy by bucket (higher is better), with 95% CI.
"""
from __future__ import annotations
import csv, random, statistics
from pathlib import Path

LONG_CSV = Path(r"E:\Claude\CourseABSA\hopeful-kowalevski-04ee10\paper\outputs\tables\phase_d2_filtering_results_multiseed_long.csv")
FIG_DIR = Path(r"E:\Claude\CourseABSA\hopeful-kowalevski-04ee10\paper\outputs\figures")
BUCKETS = ["top25","top50","full","bot25","random_5k"]
DISPLAY = {"top25":"top-25%","top50":"top-50%","full":"full (10K)","bot25":"bottom-25%","random_5k":"random 5K"}
COLORS = {"top25":"#2c7fb8","top50":"#1a5b8a","full":"#7fcdbb","bot25":"#d7191c","random_5k":"#fdae61"}


def bootstrap_ci(vals, n_boot=10000, alpha=0.05, seed=0):
    rng = random.Random(seed)
    n = len(vals)
    means = []
    for _ in range(n_boot):
        samp = [vals[rng.randrange(n)] for _ in range(n)]
        means.append(sum(samp)/n)
    means.sort()
    return means[int(alpha/2*n_boot)], means[int((1-alpha/2)*n_boot)]


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    by_bucket = {b: {} for b in BUCKETS}
    with LONG_CSV.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            by_bucket[r["bucket"]].setdefault("sent_mse", []).append(float(r["sentiment_mse_detected"]))
            by_bucket[r["bucket"]].setdefault("mba", []).append(float(r["macro_balanced_accuracy"]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.0))

    # Panel A: sentiment_mse
    labels = [DISPLAY[b] for b in BUCKETS]
    means_sm = [sum(by_bucket[b]["sent_mse"])/4 for b in BUCKETS]
    cis_sm = [bootstrap_ci(by_bucket[b]["sent_mse"], seed=hash(b)&0xfff) for b in BUCKETS]
    err_lo = [m - lo for m, (lo,_) in zip(means_sm, cis_sm)]
    err_hi = [hi - m for m, (_,hi) in zip(means_sm, cis_sm)]
    colors = [COLORS[b] for b in BUCKETS]
    ax1.bar(labels, means_sm, yerr=[err_lo, err_hi], color=colors, capsize=6)
    full_sm = means_sm[BUCKETS.index("full")]
    ax1.axhline(full_sm, color="#666", linestyle="--", linewidth=1, label=f"full = {full_sm:.3f}")
    for x, m in enumerate(means_sm):
        ax1.text(x, m + 0.015, f"{m:.3f}", ha="center", va="bottom", fontsize=9)
    ax1.set_ylabel("Sentiment MSE (lower = better)")
    ax1.set_title("A. Sentiment-polarity transfer to Herath\n(n=4 seeds, 95% bootstrap CI)")
    ax1.set_ylim(0, max(means_sm)*1.18)
    ax1.legend(loc="upper left", fontsize=8)
    plt.setp(ax1.get_xticklabels(), rotation=20, ha='right')

    # Panel B: macro_balanced_accuracy
    means_ba = [sum(by_bucket[b]["mba"])/4 for b in BUCKETS]
    cis_ba = [bootstrap_ci(by_bucket[b]["mba"], seed=hash(b)*7 & 0xfff) for b in BUCKETS]
    err_lo = [m - lo for m, (lo,_) in zip(means_ba, cis_ba)]
    err_hi = [hi - m for m, (_,hi) in zip(means_ba, cis_ba)]
    ax2.bar(labels, means_ba, yerr=[err_lo, err_hi], color=colors, capsize=6)
    full_ba = means_ba[BUCKETS.index("full")]
    ax2.axhline(full_ba, color="#666", linestyle="--", linewidth=1, label=f"full = {full_ba:.3f}")
    for x, m in enumerate(means_ba):
        ax2.text(x, m + 0.005, f"{m:.3f}", ha="center", va="bottom", fontsize=9)
    ax2.set_ylabel("Macro balanced accuracy (higher = better)")
    ax2.set_title("B. Aspect-detection transfer to Herath\n(n=4 seeds, 95% bootstrap CI)")
    ax2.set_ylim(0.49, max(means_ba)*1.05)
    ax2.legend(loc="upper left", fontsize=8)
    plt.setp(ax2.get_xticklabels(), rotation=20, ha='right')

    plt.tight_layout()
    for fmt in ("svg","png"):
        p = FIG_DIR / f"phase_d2_headline.{fmt}"
        fig.savefig(p, dpi=150 if fmt=="png" else None)
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
