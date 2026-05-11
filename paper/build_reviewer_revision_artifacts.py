from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = ROOT / "paper"
FIG_DIR = PAPER_DIR / "outputs" / "figures"
TABLE_DIR = PAPER_DIR / "outputs" / "tables"
REAL_DIR = PAPER_DIR / "real_transfer"
BENCH_DIR = PAPER_DIR / "benchmark_outputs"


def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "#fcfbf7",
            "axes.facecolor": "#fcfbf7",
            "savefig.facecolor": "#fcfbf7",
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "axes.edgecolor": "#d0ccc2",
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.color": "#e6e1d8",
            "grid.linewidth": 0.7,
            "grid.alpha": 0.8,
            "legend.frameon": False,
            "xtick.color": "#2f3b46",
            "ytick.color": "#2f3b46",
            "text.color": "#1a1f24",
            "axes.labelcolor": "#1a1f24",
            "axes.titleweight": "bold",
        }
    )


def write_markdown_table(df: pd.DataFrame, path: Path) -> None:
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in df.fillna("").astype(str).itertuples(index=False, name=None):
        lines.append("| " + " | ".join(row) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_overlap_gap_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(REAL_DIR / "overlap_internal_vs_external_summary.csv")
    keep = df.loc[
        :,
        ["approach", "eval_split", "micro_f1", "macro_f1", "micro_recall", "sentiment_mse_detected", "n_reviews"],
    ].copy()
    for col in ["micro_f1", "macro_f1", "micro_recall", "sentiment_mse_detected"]:
        keep[col] = keep[col].map(lambda x: f"{float(x):.4f}")
    keep["n_reviews"] = keep["n_reviews"].astype(int).astype(str)

    pivot = df.pivot(index="approach", columns="eval_split", values=["micro_f1", "sentiment_mse_detected"])
    rows = []
    for approach in pivot.index:
        synth_f1 = float(pivot.loc[approach, ("micro_f1", "synthetic_overlap_test")])
        real_f1 = float(pivot.loc[approach, ("micro_f1", "real_herath_mapped")])
        synth_mse = float(pivot.loc[approach, ("sentiment_mse_detected", "synthetic_overlap_test")])
        real_mse = float(pivot.loc[approach, ("sentiment_mse_detected", "real_herath_mapped")])
        rows.append(
            {
                "approach": approach,
                "synthetic_overlap_f1": f"{synth_f1:.4f}",
                "real_overlap_f1": f"{real_f1:.4f}",
                "f1_gap_real_minus_synth": f"{(real_f1 - synth_f1):.4f}",
                "synthetic_overlap_mse": f"{synth_mse:.4f}",
                "real_overlap_mse": f"{real_mse:.4f}",
            }
        )
    gap = pd.DataFrame(rows).sort_values("real_overlap_f1", ascending=False).reset_index(drop=True)
    return keep, gap


def plot_overlap_gap() -> None:
    df = pd.read_csv(REAL_DIR / "overlap_internal_vs_external_summary.csv")
    approaches = list(dict.fromkeys(df["approach"].tolist()))
    x = np.arange(len(approaches))
    width = 0.34
    synth = [float(df[(df["approach"] == a) & (df["eval_split"] == "synthetic_overlap_test")]["micro_f1"].iloc[0]) for a in approaches]
    real = [float(df[(df["approach"] == a) & (df["eval_split"] == "real_herath_mapped")]["micro_f1"].iloc[0]) for a in approaches]

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.bar(x - width / 2, synth, width=width, color="#315c88", label="synthetic overlap test")
    ax.bar(x + width / 2, real, width=width, color="#b5742d", label="mapped real test")
    ax.set_xticks(x, approaches, rotation=12, ha="right")
    ax.set_ylim(0.0, max(max(synth), max(real)) + 0.12)
    ax.set_ylabel("Micro-F1")
    for xpos, val in zip(x - width / 2, synth):
        ax.text(xpos, val + 0.01, f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    for xpos, val in zip(x + width / 2, real):
        ax.text(xpos, val + 0.01, f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    ax.legend(ncol=2, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "overlap_internal_vs_external_f1.svg", format="svg", bbox_inches="tight")
    plt.close(fig)


def build_prompt_table() -> pd.DataFrame:
    df = pd.read_csv(BENCH_DIR / "openai_batch_eval_summary.csv").copy()
    df["variant_label"] = df["variant"].map(
        {
            "zero-shot": "zero-shot",
            "few-shot": "few-shot",
            "few-shot-diverse": "few-shot-diverse",
        }
    )
    out = df.loc[:, ["variant_label", "micro_f1", "macro_f1", "micro_recall", "sentiment_mse_detected"]].copy()
    out.columns = ["variant", "micro_f1", "macro_f1", "micro_recall", "sentiment_mse"]
    for col in ["micro_f1", "macro_f1", "micro_recall", "sentiment_mse"]:
        out[col] = out[col].map(lambda x: f"{float(x):.4f}")
    return out.sort_values("micro_f1", ascending=False).reset_index(drop=True)


def plot_prompt_baselines() -> None:
    df = pd.read_csv(BENCH_DIR / "openai_batch_eval_summary.csv").copy()
    df = df.sort_values("micro_f1", ascending=True).reset_index(drop=True)
    y = np.arange(len(df))
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.4), sharey=True)
    metrics = [("micro_f1", "Micro-F1", "#315c88"), ("micro_recall", "Micro-recall", "#4f8f5b")]
    for ax, (column, title, color) in zip(axes, metrics):
        values = df[column].astype(float).to_numpy()
        ax.hlines(y, 0, values, color=color, linewidth=2.0)
        ax.scatter(values, y, color=color, s=65, zorder=3)
        ax.set_xlabel(title)
        for yi, val in zip(y, values):
            ax.text(val + 0.005, yi, f"{val:.3f}", va="center", fontsize=8)
    axes[0].set_yticks(y, df["variant"])
    axes[0].set_xlim(0.0, max(0.30, float(df["micro_f1"].max()) + 0.06))
    axes[1].set_xlim(0.0, max(0.38, float(df["micro_recall"].max()) + 0.08))
    axes[1].tick_params(axis="y", left=False, labelleft=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "prompt_baseline_subset_comparison.svg", format="svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    configure_style()
    overlap_detail, overlap_gap = build_overlap_gap_tables()
    overlap_detail.to_csv(TABLE_DIR / "overlap_internal_vs_external_summary_publication.csv", index=False)
    overlap_gap.to_csv(TABLE_DIR / "overlap_internal_vs_external_gap_publication.csv", index=False)
    write_markdown_table(overlap_detail, TABLE_DIR / "overlap_internal_vs_external_summary_publication.md")
    write_markdown_table(overlap_gap, TABLE_DIR / "overlap_internal_vs_external_gap_publication.md")
    plot_overlap_gap()

    prompt_table = build_prompt_table()
    prompt_table.to_csv(TABLE_DIR / "prompt_baseline_subset_summary.csv", index=False)
    write_markdown_table(prompt_table, TABLE_DIR / "prompt_baseline_subset_summary.md")
    plot_prompt_baselines()


if __name__ == "__main__":
    main()
