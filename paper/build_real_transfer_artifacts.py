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


def load_transfer_summary() -> pd.DataFrame:
    return pd.read_csv(REAL_DIR / "synthetic_to_real_transfer_summary.csv")


def load_overlap_summary() -> pd.DataFrame:
    return pd.read_csv(REAL_DIR / "herath_overlap_summary.csv")


def load_transfer_per_aspect() -> pd.DataFrame:
    return pd.read_csv(REAL_DIR / "synthetic_to_real_transfer_per_aspect.csv")


def build_transfer_table(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.sort_values("micro_f1", ascending=False).reset_index(drop=True).copy()
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    out = out.loc[
        :,
        [
            "rank",
            "approach",
            "micro_f1",
            "macro_f1",
            "micro_recall",
            "sentiment_mse_detected",
            "n_real_reviews",
            "n_overlap_aspects",
        ],
    ]
    for col in ["micro_f1", "macro_f1", "micro_recall", "sentiment_mse_detected"]:
        out[col] = out[col].map(lambda x: f"{float(x):.4f}")
    out["n_real_reviews"] = out["n_real_reviews"].astype(int).astype(str)
    out["n_overlap_aspects"] = out["n_overlap_aspects"].astype(int).astype(str)
    return out


def build_overlap_table(overlap: pd.DataFrame) -> pd.DataFrame:
    out = overlap.copy().sort_values("review_count", ascending=False).reset_index(drop=True)
    for col in ["review_count", "positive", "neutral", "negative"]:
        out[col] = out[col].astype(int).astype(str)
    return out


def plot_real_transfer_overview(summary: pd.DataFrame, overlap: pd.DataFrame) -> None:
    overlap_sorted = overlap.sort_values("review_count", ascending=True).reset_index(drop=True)
    results = summary.sort_values("micro_f1", ascending=True).reset_index(drop=True)
    y_overlap = np.arange(len(overlap_sorted))
    y_results = np.arange(len(results))

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.8))

    axes[0].barh(y_overlap, overlap_sorted["review_count"], color="#4f8f5b")
    axes[0].set_yticks(y_overlap, overlap_sorted["aspect"])
    axes[0].set_xlabel("Reviews")
    for yi, val in zip(y_overlap, overlap_sorted["review_count"]):
        axes[0].text(float(val) + 10, yi, str(int(val)), va="center", fontsize=8)

    axes[1].hlines(y_results, 0, results["micro_f1"], color="#315c88", linewidth=2.2)
    axes[1].scatter(results["micro_f1"], y_results, color="#315c88", s=70, zorder=3)
    axes[1].set_yticks(y_results, results["approach"])
    axes[1].set_xlim(0.34, 0.49)
    axes[1].set_xlabel("Micro-F1")
    for yi, val in zip(y_results, results["micro_f1"]):
        axes[1].text(float(val) + 0.003, yi, f"{float(val):.3f}", va="center", fontsize=8)

    axes[2].hlines(y_results, 0, results["sentiment_mse_detected"], color="#b5742d", linewidth=2.2)
    axes[2].scatter(results["sentiment_mse_detected"], y_results, color="#b5742d", s=70, zorder=3)
    axes[2].set_yticks(y_results, [])
    axes[2].set_xlim(0.30, 0.76)
    axes[2].set_xlabel("MSE")
    for yi, val in zip(y_results, results["sentiment_mse_detected"]):
        axes[2].text(float(val) + 0.01, yi, f"{float(val):.3f}", va="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "real_transfer_overview.svg", format="svg", bbox_inches="tight")
    plt.close(fig)


def plot_real_transfer_per_aspect(per_aspect: pd.DataFrame) -> None:
    pivot = (
        per_aspect.pivot_table(index="aspect", columns="approach", values="f1", aggfunc="mean")
        .fillna(0.0)
        .sort_index()
    )
    ordered = pivot.mean(axis=1).sort_values(ascending=False).index
    pivot = pivot.loc[ordered]
    matrix = pivot.to_numpy()

    fig, ax = plt.subplots(figsize=(7.8, 5.8))
    im = ax.imshow(matrix, aspect="auto", cmap="YlGnBu", vmin=0.0, vmax=max(0.8, float(matrix.max())))
    ax.set_xticks(np.arange(len(pivot.columns)), labels=pivot.columns.tolist(), rotation=20, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)), labels=pivot.index.tolist())
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{float(matrix[i, j]):.2f}", ha="center", va="center", fontsize=7, color="#182026")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cbar.ax.set_ylabel("F1", rotation=90)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "real_transfer_per_aspect_heatmap.svg", format="svg", bbox_inches="tight")
    plt.close(fig)


def plot_real_transfer_polarity_balance(overlap: pd.DataFrame) -> None:
    ordered = overlap.sort_values("review_count", ascending=False).reset_index(drop=True)
    x = np.arange(len(ordered))
    pos = ordered["positive"].to_numpy()
    neu = ordered["neutral"].to_numpy()
    neg = ordered["negative"].to_numpy()

    fig, ax = plt.subplots(figsize=(12.8, 4.8))
    ax.bar(x, pos, color="#4f8f5b", label="positive")
    ax.bar(x, neu, bottom=pos, color="#315c88", label="neutral")
    ax.bar(x, neg, bottom=pos + neu, color="#b5742d", label="negative")
    ax.set_xticks(x, ordered["aspect"], rotation=25, ha="right")
    ax.set_ylabel("Review count")
    ax.legend(ncol=3, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "real_transfer_polarity_balance.svg", format="svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    configure_style()

    summary = load_transfer_summary()
    overlap = load_overlap_summary()
    per_aspect = load_transfer_per_aspect()

    transfer_table = build_transfer_table(summary)
    overlap_table = build_overlap_table(overlap)

    transfer_table.to_csv(TABLE_DIR / "real_transfer_summary_publication.csv", index=False)
    overlap_table.to_csv(TABLE_DIR / "real_transfer_overlap_publication.csv", index=False)
    write_markdown_table(transfer_table, TABLE_DIR / "real_transfer_summary_publication.md")
    write_markdown_table(overlap_table, TABLE_DIR / "real_transfer_overlap_publication.md")

    plot_real_transfer_overview(summary, overlap)
    plot_real_transfer_per_aspect(per_aspect)
    plot_real_transfer_polarity_balance(overlap)


if __name__ == "__main__":
    main()
