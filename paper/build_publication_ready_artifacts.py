from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = ROOT / "paper"
FIG_DIR = PAPER_DIR / "outputs" / "figures"
TABLE_DIR = PAPER_DIR / "outputs" / "tables"
BENCHMARK_DIR = PAPER_DIR / "benchmark_outputs"
VALIDATION_DIR = PAPER_DIR / "validation"
DATASET_PATH = PAPER_DIR / "generated_datasets" / "batch_69cc15c483488190941478aa4e3a976d_generated_reviews.jsonl"


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


def load_dataset() -> pd.DataFrame:
    return pd.read_json(DATASET_PATH, lines=True)


def load_realism_metrics() -> pd.DataFrame:
    return pd.read_csv(TABLE_DIR / "realism_cycle_metrics.csv")


def load_benchmark_summary() -> pd.DataFrame:
    summary = pd.read_csv(BENCHMARK_DIR / "model_comparison_summary.csv").copy()
    strengthened = TABLE_DIR / "combined_local_benchmark_with_joint.csv"
    if strengthened.exists():
        strengthened_df = pd.read_csv(strengthened).copy()
        if set(summary["approach"].astype(str)) <= set(strengthened_df["approach"].astype(str)):
            for column in [
                "macro_balanced_accuracy",
                "macro_specificity",
                "macro_mcc",
                "macro_accuracy",
                "label_accuracy",
                "subset_accuracy",
                "samples_f1",
                "samples_jaccard",
            ]:
                if column not in strengthened_df.columns and column in summary.columns:
                    strengthened_df = strengthened_df.merge(
                        summary.loc[:, ["approach", column]],
                        on="approach",
                        how="left",
                    )
            if strengthened.stat().st_mtime >= (BENCHMARK_DIR / "model_comparison_summary.csv").stat().st_mtime:
                return strengthened_df
    return summary


def load_openai_summary() -> pd.DataFrame:
    return pd.read_csv(BENCHMARK_DIR / "openai_batch_eval_summary.csv").copy()


def load_real_transfer_summary() -> pd.DataFrame:
    return pd.read_csv(PAPER_DIR / "real_transfer" / "synthetic_to_real_transfer_summary.csv").copy()


def load_best_model_aspects() -> tuple[str, pd.DataFrame]:
    summary = load_benchmark_summary().sort_values("micro_f1", ascending=False).reset_index(drop=True)
    best_model = str(summary.loc[0, "approach"])
    per_aspect = pd.read_csv(BENCHMARK_DIR / "model_comparison_per_aspect.csv")
    best = per_aspect[per_aspect["approach"] == best_model].copy().sort_values("f1", ascending=False).reset_index(drop=True)
    return best_model, best


def build_publication_benchmark_table(summary: pd.DataFrame) -> pd.DataFrame:
    table = summary.sort_values("micro_f1", ascending=False).reset_index(drop=True).copy()
    if "rank" not in table.columns:
        table.insert(0, "rank", np.arange(1, len(table) + 1))
    if "runtime_min" not in table.columns:
        table["runtime_min"] = table["elapsed_seconds"] / 60.0
    keep = ["rank", "approach", "micro_f1", "macro_f1", "micro_recall", "sentiment_mse_detected", "runtime_min"]
    table = table.loc[:, keep]
    for col in ["micro_f1", "macro_f1", "micro_recall", "sentiment_mse_detected", "runtime_min"]:
        table[col] = table[col].map(lambda x: f"{float(x):.4f}" if pd.notna(x) else "")
    return table


def build_detection_diagnostics_table(local_summary: pd.DataFrame, openai_summary: pd.DataFrame, real_summary: pd.DataFrame) -> pd.DataFrame:
    local = local_summary.copy()
    local["family"] = "Local synthetic benchmark"
    openai = openai_summary.copy()
    openai["family"] = "GPT batch inference"
    real = real_summary.copy()
    real["family"] = "Mapped real-data transfer"
    real = real[real.get("eval_split", "") == "real_herath_mapped"].copy() if "eval_split" in real.columns else real
    combined = pd.concat([local, openai, real], ignore_index=True, sort=False)
    keep = [
        "family",
        "approach",
        "micro_f1",
        "macro_f1",
        "macro_balanced_accuracy",
        "macro_specificity",
        "macro_mcc",
        "sentiment_mse_detected",
    ]
    table = combined.loc[:, [column for column in keep if column in combined.columns]].copy()
    if "macro_balanced_accuracy" in table.columns:
        table = table[table["macro_balanced_accuracy"].notna()].copy()
    for col in [c for c in table.columns if c not in {"family", "approach"}]:
        table[col] = pd.to_numeric(table[col], errors="coerce").map(lambda x: f"{x:.4f}" if pd.notna(x) else "")
    return table


def build_aspect_extremes_table(best_model: str, per_aspect: pd.DataFrame) -> pd.DataFrame:
    top = per_aspect.head(5).copy()
    top["group"] = "top_5_f1"
    bottom = per_aspect.tail(5).copy()
    bottom["group"] = "bottom_5_f1"
    out = pd.concat([top, bottom], ignore_index=True)
    out.insert(0, "model", best_model)
    out = out.loc[:, ["model", "group", "aspect", "f1", "mse", "precision", "recall"]]
    for col in ["f1", "mse", "precision", "recall"]:
        out[col] = out[col].map(lambda x: f"{float(x):.4f}" if pd.notna(x) else "")
    return out


def build_realism_publication_table(realism: pd.DataFrame) -> pd.DataFrame:
    out = realism.loc[
        :, ["cycle_name", "judge_item_accuracy", "mean_confusion", "mean_entropy_bits", "binomial_p_value_vs_chance", "editor_triggered"]
    ].copy()
    out.columns = ["cycle", "judge_accuracy", "mean_confusion", "mean_entropy_bits", "binomial_p_value", "editor_triggered"]
    for col in ["judge_accuracy", "mean_confusion", "mean_entropy_bits", "binomial_p_value"]:
        out[col] = out[col].map(lambda x: f"{float(x):.4f}")
    return out


def plot_realism_curve(realism: pd.DataFrame) -> None:
    cycles = np.arange(len(realism))
    labels = [f"C{idx}" for idx in cycles]
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.8))
    metrics = [
        ("judge_item_accuracy", "Judge accuracy", "#315c88", 0.5),
        ("mean_confusion", "Mean confusion", "#4f8f5b", None),
        ("mean_entropy_bits", "Mean entropy (bits)", "#b5742d", None),
    ]
    for ax, (column, title, color, baseline) in zip(axes, metrics):
        values = realism[column].astype(float).to_numpy()
        ax.plot(cycles, values, marker="o", markersize=7, linewidth=2.2, color=color)
        ax.set_xticks(cycles, labels)
        if baseline is not None:
            ax.axhline(baseline, color="#8a8a8a", linestyle="--", linewidth=1.1)
        ax.set_ylabel(title)
        for x, y in zip(cycles, values):
            ax.text(x, y + 0.006, f"{y:.3f}", ha="center", va="bottom", fontsize=8, color="#30363d")
    axes[0].set_ylim(0.38, 0.56)
    axes[1].set_ylim(0.54, 0.67)
    axes[2].set_ylim(0.68, 0.72)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "realism_improvement_curve.svg", format="svg", bbox_inches="tight")
    plt.close(fig)


def plot_method_comparison(summary: pd.DataFrame) -> None:
    df = summary.copy()
    if "runtime_min" in df.columns:
        df["runtime_min"] = df["runtime_min"].astype(float)
    else:
        df["runtime_min"] = df["elapsed_seconds"].astype(float) / 60.0
    df["micro_f1"] = df["micro_f1"].astype(float)
    df["macro_f1"] = df["macro_f1"].astype(float)
    df["micro_recall"] = df["micro_recall"].astype(float)
    df["sentiment_mse_detected"] = df["sentiment_mse_detected"].astype(float)

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))

    ax = axes[0]
    scatter = ax.scatter(
        df["runtime_min"],
        df["micro_f1"],
        s=240,
        c=df["sentiment_mse_detected"],
        cmap="YlOrBr_r",
        edgecolors="#2f3b46",
        linewidths=0.8,
        zorder=3,
    )
    for _, row in df.iterrows():
        ax.text(
            float(row["runtime_min"]) + 0.6,
            float(row["micro_f1"]) + 0.0015,
            str(row["approach"]),
            fontsize=8,
            va="center",
        )
    ax.set_xlabel("Runtime (minutes)")
    ax.set_ylabel("Detection micro-F1")
    ax.set_xlim(-1, max(52, df["runtime_min"].max() + 3))
    ax.set_ylim(0.14, 0.29)
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.03)
    cbar.ax.set_ylabel("Sentiment MSE", rotation=90)

    ax2 = axes[1]
    plot_df = df.sort_values("micro_f1", ascending=True).reset_index(drop=True)
    y = np.arange(len(plot_df))
    ax2.barh(y, plot_df["micro_recall"], color="#d8e4ef", edgecolor="#b7c8d8", height=0.58, label="Micro-recall")
    ax2.barh(y, plot_df["micro_f1"], color="#315c88", height=0.38, label="Micro-F1")
    ax2.set_yticks(y, plot_df["approach"])
    ax2.set_xlim(0.0, 0.75)
    ax2.set_xlabel("Score")
    ax2.legend(loc="lower right")
    for yi, f1, rec in zip(y, plot_df["micro_f1"], plot_df["micro_recall"]):
        ax2.text(float(rec) + 0.012, yi + 0.16, f"{rec:.3f}", va="center", fontsize=8, color="#4f657a")
        ax2.text(float(f1) + 0.012, yi - 0.16, f"{f1:.3f}", va="center", fontsize=8, color="#16324a")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "production_method_comparison.svg", format="svg", bbox_inches="tight")
    plt.close(fig)


def plot_dataset_profile(df: pd.DataFrame) -> None:
    word_counts = df["text"].astype(str).str.split().map(len)
    aspect_counts = df["aspects"].map(len).value_counts().sort_index()
    course_counts = df["course_name"].fillna("Unknown").replace("", "Unknown").value_counts().head(8).sort_values()
    style_counts = df["style"].fillna("").replace("", "unspecified style").value_counts().head(6).sort_values()

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.4))

    axes[0, 0].hist(word_counts, bins=28, color="#315c88", alpha=0.9, edgecolor="white")
    axes[0, 0].axvline(word_counts.mean(), color="#8a3b3b", linestyle="--", linewidth=1.3)
    axes[0, 0].set_xlabel("Words")
    axes[0, 0].set_ylabel("Count")

    axes[0, 1].bar(aspect_counts.index.astype(str), aspect_counts.values, color="#4f8f5b", width=0.6)
    axes[0, 1].set_xlabel("Target aspects")
    axes[0, 1].set_ylabel("Count")

    axes[1, 0].barh(course_counts.index, course_counts.values, color="#b5742d")
    axes[1, 0].set_xlabel("Review count")

    axes[1, 1].barh(style_counts.index, style_counts.values, color="#7a5ea8")
    axes[1, 1].set_xlabel("Review count")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "production_dataset_profile.svg", format="svg", bbox_inches="tight")
    plt.close(fig)


def plot_aspect_sentiment_heatmap(df: pd.DataFrame) -> None:
    records = []
    for aspect_map in df["aspects"]:
        for aspect, sentiment in aspect_map.items():
            records.append({"aspect": aspect, "sentiment": sentiment})
    frame = pd.DataFrame(records)
    pivot = (
        frame.assign(count=1)
        .pivot_table(index="aspect", columns="sentiment", values="count", aggfunc="sum", fill_value=0)
        .reindex(columns=["negative", "neutral", "positive"], fill_value=0)
    )
    ordered = pivot.sum(axis=1).sort_values(ascending=False).index
    pivot = pivot.loc[ordered]
    matrix = pivot.values
    total_support = pivot.sum(axis=1)
    aspect_labels = [label.replace("_", " ").title() for label in pivot.index.tolist()]

    fig, (ax_heat, ax_bar) = plt.subplots(
        1,
        2,
        figsize=(11.6, 8.3),
        gridspec_kw={"width_ratios": [3.15, 1.7], "wspace": 0.16},
        constrained_layout=True,
    )

    im = ax_heat.imshow(matrix, aspect="auto", cmap="GnBu", vmin=0, vmax=float(matrix.max()))
    ax_heat.set_xticks(np.arange(3), labels=["Negative", "Neutral", "Positive"])
    ax_heat.set_yticks(np.arange(len(aspect_labels)), labels=aspect_labels)
    ax_heat.tick_params(axis="x", labelsize=10.5, pad=8, length=0)
    ax_heat.tick_params(axis="y", labelsize=9.5, pad=8, length=0)
    ax_heat.set_xticks(np.arange(-0.5, 3, 1), minor=True)
    ax_heat.set_yticks(np.arange(-0.5, len(aspect_labels), 1), minor=True)
    ax_heat.grid(which="minor", color="#f3efe6", linestyle="-", linewidth=1.0)
    ax_heat.tick_params(which="minor", bottom=False, left=False)

    max_value = float(matrix.max()) if float(matrix.max()) > 0 else 1.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = int(matrix[i, j])
            normalized = value / max_value
            text_color = "#f8fbfd" if normalized > 0.56 else "#173042"
            ax_heat.text(
                j,
                i,
                f"{value:,}",
                ha="center",
                va="center",
                fontsize=8.6,
                fontweight="semibold",
                color=text_color,
            )

    ax_bar.barh(
        np.arange(len(total_support)),
        total_support.values,
        color="#6f8fb5",
        edgecolor="#496885",
        linewidth=0.7,
        height=0.72,
    )
    ax_bar.set_yticks(np.arange(len(aspect_labels)), labels=[""] * len(aspect_labels))
    ax_bar.invert_yaxis()
    ax_bar.tick_params(axis="y", length=0)
    ax_bar.tick_params(axis="x", labelsize=9.5)
    ax_bar.set_xlabel("Total labeled mentions", fontsize=10.5, labelpad=8)
    ax_bar.grid(axis="x", color="#e6e1d8", linewidth=0.7, alpha=0.9)
    ax_bar.grid(axis="y", visible=False)
    x_max = float(total_support.max()) * 1.17
    ax_bar.set_xlim(0, x_max)
    for y, value in enumerate(total_support.values):
        ax_bar.text(
            float(value) + x_max * 0.018,
            y,
            f"{int(value):,}",
            va="center",
            ha="left",
            fontsize=8.7,
            color="#22384a",
        )

    for axis in (ax_heat, ax_bar):
        for spine in axis.spines.values():
            spine.set_visible(False)

    cbar = fig.colorbar(im, ax=[ax_heat, ax_bar], fraction=0.028, pad=0.02)
    cbar.ax.set_ylabel("Review count per aspect-polarity cell", rotation=90, labelpad=12)
    cbar.ax.tick_params(labelsize=8.8)
    fig.savefig(FIG_DIR / "production_aspect_sentiment_heatmap.svg", format="svg", bbox_inches="tight")
    plt.close(fig)


def plot_best_model_aspects(best_model: str, per_aspect: pd.DataFrame) -> None:
    top = per_aspect.head(10).copy()
    bottom = per_aspect.tail(10).copy().sort_values("f1", ascending=False)
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 9.2), sharex="col")

    axes[0, 0].barh(top["aspect"], top["f1"], color="#4f8f5b")
    axes[0, 0].invert_yaxis()
    axes[0, 0].set_xlim(0, max(0.60, float(top["f1"].max()) + 0.05))
    axes[0, 0].set_xlabel("Detection F1")

    axes[0, 1].barh(top["aspect"], top["mse"].fillna(0.0), color="#b5742d")
    axes[0, 1].invert_yaxis()
    axes[0, 1].set_xlabel("Sentiment MSE")

    axes[1, 0].barh(bottom["aspect"], bottom["f1"], color="#9a5a6b")
    axes[1, 0].invert_yaxis()
    axes[1, 0].set_xlim(0, max(0.30, float(per_aspect["f1"].max()) + 0.05))
    axes[1, 0].set_xlabel("Detection F1")

    axes[1, 1].barh(bottom["aspect"], bottom["mse"].fillna(0.0), color="#c29037")
    axes[1, 1].invert_yaxis()
    axes[1, 1].set_xlabel("Sentiment MSE")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "production_best_model_per_aspect.svg", format="svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    configure_style()
    dataset = load_dataset()
    realism = load_realism_metrics()
    benchmark = load_benchmark_summary()
    openai_summary = load_openai_summary()
    real_transfer_summary = load_real_transfer_summary()
    best_model, best_aspects = load_best_model_aspects()

    publication_benchmark = build_publication_benchmark_table(benchmark)
    detection_diagnostics = build_detection_diagnostics_table(benchmark, openai_summary, real_transfer_summary)
    aspect_extremes = build_aspect_extremes_table(best_model, best_aspects)
    realism_table = build_realism_publication_table(realism)

    publication_benchmark.to_csv(TABLE_DIR / "publication_benchmark_summary.csv", index=False)
    detection_diagnostics.to_csv(TABLE_DIR / "publication_detection_diagnostics.csv", index=False)
    aspect_extremes.to_csv(TABLE_DIR / "publication_aspect_extremes.csv", index=False)
    realism_table.to_csv(TABLE_DIR / "publication_realism_summary.csv", index=False)
    write_markdown_table(publication_benchmark, TABLE_DIR / "publication_benchmark_summary.md")
    write_markdown_table(detection_diagnostics, TABLE_DIR / "publication_detection_diagnostics.md")
    write_markdown_table(aspect_extremes, TABLE_DIR / "publication_aspect_extremes.md")
    write_markdown_table(realism_table, TABLE_DIR / "publication_realism_summary.md")

    plot_realism_curve(realism)
    plot_method_comparison(benchmark)
    plot_dataset_profile(dataset)
    plot_aspect_sentiment_heatmap(dataset)
    plot_best_model_aspects(best_model, best_aspects)

    print(
        json.dumps(
            {
                "best_model": best_model,
                "tables": [
                    str(TABLE_DIR / "publication_benchmark_summary.csv"),
                    str(TABLE_DIR / "publication_detection_diagnostics.csv"),
                    str(TABLE_DIR / "publication_aspect_extremes.csv"),
                    str(TABLE_DIR / "publication_realism_summary.csv"),
                ],
                "figures": [
                    str(FIG_DIR / "realism_improvement_curve.svg"),
                    str(FIG_DIR / "production_method_comparison.svg"),
                    str(FIG_DIR / "production_dataset_profile.svg"),
                    str(FIG_DIR / "production_aspect_sentiment_heatmap.svg"),
                    str(FIG_DIR / "production_best_model_per_aspect.svg"),
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
