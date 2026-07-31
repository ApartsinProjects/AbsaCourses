"""Regenerate Figure 6 (sample-efficiency curve) from the artifact summary.

Two learning curves over the number of real Herath reviews N, both evaluated on the
SAME fixed 566-row real Herath test set (9-aspect detection micro-F1), 5 seeds each:
  - synthetic pretrain + real fine-tune  (curve_by_real_train_n)
  - real-only training from scratch      (real_only_curve_by_real_train_n)
Shaded bands are 95% CIs (t, n-1 dof); markers carry matching error bars. The
synthetic-only-transfer reference (0.402, Section 5.4 / Table 8) stays as a flat line.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
SUMMARY = ROOT / "outputs/rc5_finetune_curve_summary.json"
OUT = ROOT / "outputs/figures/rc5_finetune_curve.svg"

SYNTH_ONLY_REF = 0.402   # canonical five-seed synthetic-only transfer (Table 8)


def _series(curve):
    xs = sorted(int(k) for k in curve)
    ys = [curve[str(x)]["micro_f1_mean"] for x in xs]
    # prefer 95% CI half-width; fall back to std if CI absent
    es = [curve[str(x)].get("micro_f1_ci95", curve[str(x)].get("micro_f1_std", 0.0)) for x in xs]
    return xs, ys, es


def _plot_curve(ax, xs, ys, es, color, label):
    ax.fill_between(xs, [y - e for y, e in zip(ys, es)], [y + e for y, e in zip(ys, es)],
                    color=color, alpha=0.15, linewidth=0)
    ax.errorbar(xs, ys, yerr=es, marker="o", color=color, linewidth=1.8,
                capsize=3, label=label)


def main():
    d = json.load(open(SUMMARY))
    synth = d["curve_by_real_train_n"]
    real = d.get("real_only_curve_by_real_train_n")

    plt.rcParams.update({"font.size": 11, "svg.fonttype": "none",
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")

    xs, ys, es = _series(synth)
    _plot_curve(ax, xs, ys, es, "#315c88", "synthetic pretrain + real fine-tune")
    for x, y in zip(xs, ys):
        ax.text(x, y + 0.014, f"{y:.2f}", ha="center", va="bottom", fontsize=8, color="#315c88")

    if real:
        rxs, rys, res = _series(real)
        _plot_curve(ax, rxs, rys, res, "#2f6b43", "real-only training (from scratch)")
        for x, y in zip(rxs, rys):
            ax.text(x, y - 0.020, f"{y:.2f}", ha="center", va="top", fontsize=8, color="#2f6b43")

    ax.axhline(SYNTH_ONLY_REF, color="#b5742d", linestyle=":", linewidth=1.4,
               label=f"synthetic-only transfer ({SYNTH_ONLY_REF:.3f})")

    ax.set_xlabel("Number of real Herath reviews used to train / fine-tune")
    ax.set_ylabel("Real-test 9-aspect micro-F1")
    ax.set_ylim(0.35, 0.83)
    ax.set_xticks(xs)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT, format="svg", bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT}")
    print(f"synth curve: {list(zip(xs, [round(y,4) for y in ys]))}")
    if real:
        print(f"real-only curve: {list(zip(rxs, [round(y,4) for y in rys]))}")


if __name__ == "__main__":
    main()
