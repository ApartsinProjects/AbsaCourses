"""Phase E final headline figure: 2x2 grid of sent_mse by bucket for
(arch in {BERT, DistilBERT}) x (target in {Herath, EduRABSA}).
"""
from __future__ import annotations
import csv, random
from pathlib import Path

LONG_CSV = Path(r"E:\Claude\CourseABSA\hopeful-kowalevski-04ee10\paper\outputs\tables\phase_e_long.csv")
FIG_DIR = Path(r"E:\Claude\CourseABSA\hopeful-kowalevski-04ee10\paper\outputs\figures")
BUCKETS = ["top25","top50","full","bot25","random_5k"]
DISPLAY = {"top25":"top-25%","top50":"top-50%","full":"full","bot25":"bot-25%","random_5k":"random-5k"}
COLORS = {"top25":"#2c7fb8","top50":"#1a5b8a","full":"#7fcdbb","bot25":"#d7191c","random_5k":"#fdae61"}


def bootstrap_ci(vals, n_boot=10000, alpha=0.05, seed=0):
    if len(vals) < 2:
        v = vals[0] if vals else 0
        return v, v
    rng = random.Random(seed)
    n = len(vals)
    means = sorted(sum(vals[rng.randrange(n)] for _ in range(n))/n for _ in range(n_boot))
    return means[int(alpha/2*n_boot)], means[int((1-alpha/2)*n_boot)]


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Load long-format
    by = {}
    with LONG_CSV.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            key = (r["arch"], r["target"], r["bucket"])
            by.setdefault(key, []).append(float(r["sentiment_mse_detected"]))

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    panel_title = {
        ("bert-base-uncased","herath"):    "A. BERT-base on Herath transfer",
        ("bert-base-uncased","edurabsa"):  "B. BERT-base on EduRABSA transfer",
        ("distilbert-base-uncased","herath"):    "C. DistilBERT on Herath transfer",
        ("distilbert-base-uncased","edurabsa"):  "D. DistilBERT on EduRABSA transfer",
    }
    layout = {
        (0,0): ("bert-base-uncased","herath"),
        (0,1): ("bert-base-uncased","edurabsa"),
        (1,0): ("distilbert-base-uncased","herath"),
        (1,1): ("distilbert-base-uncased","edurabsa"),
    }
    for (i, j), (arch, target) in layout.items():
        ax = axes[i][j]
        means, ci_lo, ci_hi, ns = [], [], [], []
        for b in BUCKETS:
            vals = by.get((arch, target, b), [])
            if not vals:
                means.append(0); ci_lo.append(0); ci_hi.append(0); ns.append(0)
                continue
            m = sum(vals)/len(vals)
            lo, hi = bootstrap_ci(vals, seed=hash(arch+target+b)&0xfff)
            means.append(m); ci_lo.append(lo); ci_hi.append(hi); ns.append(len(vals))
        err_lo = [m - lo for m, lo in zip(means, ci_lo)]
        err_hi = [hi - m for hi, m in zip(ci_hi, means)]
        labels = [DISPLAY[b] for b in BUCKETS]
        colors = [COLORS[b] for b in BUCKETS]
        ax.bar(labels, means, yerr=[err_lo, err_hi], color=colors, capsize=4)
        for x, (m, n) in enumerate(zip(means, ns)):
            if m > 0:
                ax.text(x, m + 0.015, f"{m:.3f}", ha="center", va="bottom", fontsize=8.5)
        full_m = means[BUCKETS.index("full")]
        if full_m > 0:
            ax.axhline(full_m, color="#666", linestyle="--", linewidth=0.8)
        n_display = max(ns) if any(n > 0 for n in ns) else 0
        ax.set_title(f"{panel_title[(arch,target)]}  (n={n_display} seeds)", fontsize=10)
        ax.set_ylabel("Sentiment MSE\n(lower = better)", fontsize=9)
        ax.set_ylim(0, max(0.95, max(ci_hi) + 0.05))
        plt.setp(ax.get_xticklabels(), rotation=18, ha='right', fontsize=8.5)
    fig.suptitle("Faithfulness-aware filtering reduces Herath and EduRABSA sentiment-polarity error\n(paired 95% bootstrap CI, two architectures × two transfer targets)", fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    for fmt in ("svg","png"):
        p = FIG_DIR / f"phase_e_headline.{fmt}"
        fig.savefig(p, dpi=150 if fmt=="png" else None)
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
