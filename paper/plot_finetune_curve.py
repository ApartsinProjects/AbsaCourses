"""Regenerate Figure 6 (rc5 fine-tuning-size curve) from the artifact summary.

Curve points come from outputs/rc5_finetune_curve_summary.json (two-seed sweep,
unchanged). The synthetic-only-transfer reference line is set to the canonical
five-seed headline value (0.402, Section 5.4 / Table 8), which is the authoritative
estimate of that exact quantity, replacing the noisier two-seed 0.4593 anchor.
The real-only reference (0.767) already matches the headline.
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

def main():
    d = json.load(open(SUMMARY))
    curve = d["curve_by_real_train_n"]
    xs = sorted(int(k) for k in curve)
    ys = [curve[str(x)]["micro_f1_mean"] for x in xs]
    es = [curve[str(x)]["micro_f1_std"] for x in xs]
    real_only = d["references"]["real_only_training"]

    plt.rcParams.update({"font.size": 11, "svg.fonttype": "none",
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.errorbar(xs, ys, yerr=es, marker="o", color="#315c88", linewidth=1.8,
                capsize=3, label="synthetic pretrain + real fine-tune")
    ax.axhline(real_only, color="#2f6b43", linestyle="--", linewidth=1.2,
               label=f"real-only training ({real_only:.3f})")
    ax.axhline(SYNTH_ONLY_REF, color="#b5742d", linestyle=":", linewidth=1.4,
               label=f"synthetic-only transfer ({SYNTH_ONLY_REF:.3f})")
    for x, y in zip(xs, ys):
        ax.text(x, y + 0.012, f"{y:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("Number of real Herath reviews used to fine-tune")
    ax.set_ylabel("Real-test 9-aspect micro-F1")
    ax.set_ylim(0.35, 0.83)
    ax.set_xticks(xs)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT, format="svg", bbox_inches="tight")
    print(f"wrote {OUT}")
    print(f"curve: {list(zip(xs, [round(y,4) for y in ys]))}")
    print(f"refs: synthetic-only {SYNTH_ONLY_REF}, real-only {real_only}")

if __name__ == "__main__":
    main()
