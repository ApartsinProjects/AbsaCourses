from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LOCAL_PER_ASPECT = ROOT / "paper" / "benchmark_outputs" / "model_comparison_per_aspect.csv"
GPT_PER_ASPECT = ROOT / "paper" / "benchmark_outputs" / "openai_batch_eval_per_aspect.csv"
REAL_PER_ASPECT = ROOT / "paper" / "real_transfer" / "synthetic_to_real_transfer_per_aspect.csv"
OUT_TABLE = ROOT / "paper" / "outputs" / "tables" / "groupwise_summary.csv"
OUT_MD = ROOT / "paper" / "outputs" / "tables" / "groupwise_summary.md"
OUT_FIG = ROOT / "paper" / "outputs" / "figures" / "groupwise_performance.svg"

ASPECT_GROUPS = {
    "clarity": "Instructional Quality",
    "lecturer_quality": "Instructional Quality",
    "materials": "Instructional Quality",
    "feedback_quality": "Instructional Quality",
    "exam_fairness": "Assessment & Course Mgmt",
    "assessment_design": "Assessment & Course Mgmt",
    "grading_transparency": "Assessment & Course Mgmt",
    "organization": "Assessment & Course Mgmt",
    "tooling_usability": "Assessment & Course Mgmt",
    "difficulty": "Learning Demand & Readiness",
    "workload": "Learning Demand & Readiness",
    "pacing": "Learning Demand & Readiness",
    "prerequisite_fit": "Learning Demand & Readiness",
    "support": "Learning Environment",
    "accessibility": "Learning Environment",
    "peer_interaction": "Learning Environment",
    "relevance": "Engagement & Value",
    "interest": "Engagement & Value",
    "practical_application": "Engagement & Value",
    "overall_experience": "Engagement & Value",
}

GROUP_ORDER = [
    "Instructional Quality",
    "Assessment & Course Mgmt",
    "Learning Demand & Readiness",
    "Learning Environment",
    "Engagement & Value",
]


def load_and_filter() -> pd.DataFrame:
    local = pd.read_csv(LOCAL_PER_ASPECT)
    gpt = pd.read_csv(GPT_PER_ASPECT)
    real = pd.read_csv(REAL_PER_ASPECT)

    frames = [
        local[local["approach"] == "bert-base-uncased"].assign(family="Best local two-step", eval_scope="synthetic_20"),
        local[local["approach"] == "distilbert_joint"].assign(family="Best joint", eval_scope="synthetic_20"),
        gpt[gpt["approach"] == "openai-gpt-5.2-zero-shot"].assign(family="Best GPT", eval_scope="synthetic_20"),
        real[real["approach"] == "bert-base-uncased"].assign(family="Best real-transfer", eval_scope="real_overlap_9"),
    ]
    df = pd.concat(frames, ignore_index=True)
    df["group"] = df["aspect"].map(ASPECT_GROUPS)
    df = df[df["group"].notna()].copy()
    return df


def micro_f1_from_counts(frame: pd.DataFrame) -> float:
    tp = float(frame["tp"].sum())
    fp = float(frame["fp"].sum())
    fn = float(frame["fn"].sum())
    denom = 2 * tp + fp + fn
    return 0.0 if denom == 0 else (2 * tp) / denom


def group_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (family, eval_scope, group), frame in df.groupby(["family", "eval_scope", "group"], sort=False):
        rows.append(
            {
                "family": family,
                "eval_scope": eval_scope,
                "group": group,
                "group_micro_f1": micro_f1_from_counts(frame),
                "group_macro_f1": float(frame["f1"].mean()),
                "group_balanced_accuracy": float(frame["balanced_accuracy"].mean()),
                "group_specificity": float(frame["specificity"].mean()),
                "group_mcc": float(frame["mcc"].mean()),
                "group_sentiment_mse": float(frame["mse"].mean()),
                "n_aspects": int(frame.shape[0]),
            }
        )
    out = pd.DataFrame(rows)
    out["group"] = pd.Categorical(out["group"], categories=GROUP_ORDER, ordered=True)
    out = out.sort_values(["eval_scope", "family", "group"]).reset_index(drop=True)
    return out


def write_markdown(df: pd.DataFrame, path: Path) -> None:
    lines = [
        "| Family | Eval scope | Group | Group micro-F1 | Group macro-F1 | Group balanced accuracy | Group sentiment MSE | # aspects |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"| {row['family']} | {row['eval_scope']} | {row['group']} | "
            f"{row['group_micro_f1']:.4f} | {row['group_macro_f1']:.4f} | "
            f"{row['group_balanced_accuracy']:.4f} | {row['group_sentiment_mse']:.4f} | {row['n_aspects']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_figure(df: pd.DataFrame, path: Path) -> None:
    subset = df[df["eval_scope"] == "synthetic_20"].copy()
    families = ["Best local two-step", "Best joint", "Best GPT"]
    colors = {
        "Best local two-step": "#264653",
        "Best joint": "#6c8ead",
        "Best GPT": "#c07a2c",
        "Best real-transfer": "#6f9f70",
    }
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    x = range(len(GROUP_ORDER))
    width = 0.24
    for offset, family in zip([-width, 0, width], families):
        fam = subset[subset["family"] == family].set_index("group").reindex(GROUP_ORDER)
        ax.bar(
            [i + offset for i in x],
            fam["group_micro_f1"],
            width=width,
            label=family,
            color=colors[family],
            edgecolor="white",
            linewidth=0.8,
        )
    ax.set_xticks(list(x))
    ax.set_xticklabels(GROUP_ORDER, rotation=18, ha="right")
    ax.set_ylabel("Group micro-F1")
    ax.set_ylim(0, max(0.65, float(subset["group_micro_f1"].max()) + 0.05))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.18, linewidth=0.6)
    ax.legend(frameon=False, ncol=3, loc="upper right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = load_and_filter()
    summary = group_summary(df)
    OUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT_TABLE, index=False)
    write_markdown(summary, OUT_MD)
    build_figure(summary, OUT_FIG)
    print(
        json.dumps(
            {
                "table_csv": str(OUT_TABLE),
                "table_md": str(OUT_MD),
                "figure_svg": str(OUT_FIG),
                "rows": int(summary.shape[0]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
