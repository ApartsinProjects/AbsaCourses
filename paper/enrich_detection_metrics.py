from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def safe_specificity(tp: float, tn: float, fp: float, fn: float) -> float:
    denom = tn + fp
    return float(tn / denom) if denom else 0.0


def safe_balanced_accuracy(tp: float, tn: float, fp: float, fn: float) -> float:
    recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
    specificity = safe_specificity(tp, tn, fp, fn)
    return float((recall + specificity) / 2.0)


def safe_mcc(tp: float, tn: float, fp: float, fn: float) -> float:
    numerator = (tp * tn) - (fp * fn)
    denom = float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    return float(numerator / denom) if denom else 0.0


def enrich_summary(summary_df: pd.DataFrame, per_aspect_df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    enriched = summary_df.copy()
    needed = {"tp", "tn", "fp", "fn", "accuracy", "precision", "recall", "f1"}
    missing = needed.difference(per_aspect_df.columns)
    if missing:
        raise ValueError(f"Per-aspect CSV is missing columns: {sorted(missing)}")

    group_keys = group_cols if group_cols else ["approach"]
    group_frame = per_aspect_df.copy()
    for column in ["tp", "tn", "fp", "fn", "accuracy", "precision", "recall", "f1"]:
        group_frame[column] = pd.to_numeric(group_frame[column], errors="coerce")
    group_frame["specificity"] = group_frame.apply(
        lambda row: safe_specificity(row["tp"], row["tn"], row["fp"], row["fn"]),
        axis=1,
    )
    group_frame["balanced_accuracy"] = group_frame.apply(
        lambda row: safe_balanced_accuracy(row["tp"], row["tn"], row["fp"], row["fn"]),
        axis=1,
    )
    group_frame["mcc"] = group_frame.apply(
        lambda row: safe_mcc(row["tp"], row["tn"], row["fp"], row["fn"]),
        axis=1,
    )

    derived = (
        group_frame.groupby(group_keys, dropna=False)
        .agg(
            macro_accuracy=("accuracy", "mean"),
            macro_specificity=("specificity", "mean"),
            macro_balanced_accuracy=("balanced_accuracy", "mean"),
            macro_mcc=("mcc", "mean"),
        )
        .reset_index()
    )
    return enriched.merge(derived, on=group_keys, how="left")


def main() -> None:
    parser = argparse.ArgumentParser(description="Add derived multilabel metrics to a saved summary CSV.")
    parser.add_argument("--summary", required=True)
    parser.add_argument("--per-aspect", required=True)
    parser.add_argument("--group-cols", nargs="*", default=["approach"])
    parser.add_argument("--out-summary", default="")
    parser.add_argument("--out-per-aspect", default="")
    args = parser.parse_args()

    summary_path = Path(args.summary)
    per_aspect_path = Path(args.per_aspect)
    out_summary = Path(args.out_summary) if args.out_summary else summary_path
    out_per_aspect = Path(args.out_per_aspect) if args.out_per_aspect else per_aspect_path

    summary_df = pd.read_csv(summary_path)
    per_aspect_df = pd.read_csv(per_aspect_path)
    enriched_summary = enrich_summary(summary_df, per_aspect_df, args.group_cols)

    per_aspect_numeric = per_aspect_df.copy()
    for column in ["tp", "tn", "fp", "fn"]:
        per_aspect_numeric[column] = pd.to_numeric(per_aspect_numeric[column], errors="coerce")
    per_aspect_numeric["specificity"] = per_aspect_numeric.apply(
        lambda row: safe_specificity(row["tp"], row["tn"], row["fp"], row["fn"]),
        axis=1,
    )
    per_aspect_numeric["balanced_accuracy"] = per_aspect_numeric.apply(
        lambda row: safe_balanced_accuracy(row["tp"], row["tn"], row["fp"], row["fn"]),
        axis=1,
    )
    per_aspect_numeric["mcc"] = per_aspect_numeric.apply(
        lambda row: safe_mcc(row["tp"], row["tn"], row["fp"], row["fn"]),
        axis=1,
    )

    enriched_summary.to_csv(out_summary, index=False)
    per_aspect_numeric.to_csv(out_per_aspect, index=False)
    print({"summary": str(out_summary), "per_aspect": str(out_per_aspect)})


if __name__ == "__main__":
    main()
