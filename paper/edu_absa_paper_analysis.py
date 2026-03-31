from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "edu" / "final_student_reviews.jsonl"
NOTEBOOK_PATH = ROOT / "edu" / "absa_train_new.ipynb"
OUT_DIR = ROOT / "paper" / "outputs"
FIG_DIR = OUT_DIR / "figures"

SENTIMENT_TO_SCORE = {"negative": -1.0, "neutral": 0.0, "positive": 1.0}
SEEDS = [7, 42, 123]


@dataclass
class ExperimentResult:
    seed: int
    n_train: int
    n_calib: int
    n_test: int
    micro_precision: float
    micro_recall: float
    micro_f1: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    sentiment_mse_detected: float
    sentiment_mae_detected: float
    sentiment_polarity_accuracy: float


@dataclass
class ModelBundle:
    vectorizer: TfidfVectorizer
    mlb: MultiLabelBinarizer
    detection_model: OneVsRestClassifier
    thresholds: Dict[str, float]
    sentiment_models: Dict[str, Ridge]


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_reviews(path: Path) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = (obj.get("review_text") or "").strip()
            aspects = obj.get("aspects") or {}
            if not text or not isinstance(aspects, dict) or not aspects:
                continue
            clean_aspects = {
                str(aspect): str(sentiment)
                for aspect, sentiment in aspects.items()
                if str(sentiment) in SENTIMENT_TO_SCORE
            }
            if not clean_aspects:
                continue
            rows.append(
                {
                    "course_name": obj.get("course_name", ""),
                    "lecturer": obj.get("lecturer", ""),
                    "grade": obj.get("grade", ""),
                    "style": obj.get("style", ""),
                    "review_text": text,
                    "aspects": clean_aspects,
                }
            )
    df = pd.DataFrame(rows)
    df["word_count"] = df["review_text"].str.split().str.len()
    df["char_count"] = df["review_text"].str.len()
    df["aspect_count"] = df["aspects"].map(len)
    df["aspect_list"] = df["aspects"].map(lambda item: sorted(item.keys()))
    return df


def save_dataset_summary(df: pd.DataFrame) -> None:
    summary = {
        "n_reviews": int(len(df)),
        "n_unique_courses": int(df["course_name"].nunique()),
        "n_unique_lecturers": int(df["lecturer"].nunique()),
        "word_count_mean": round(float(df["word_count"].mean()), 2),
        "word_count_median": float(df["word_count"].median()),
        "word_count_min": int(df["word_count"].min()),
        "word_count_max": int(df["word_count"].max()),
        "char_count_mean": round(float(df["char_count"].mean()), 2),
        "aspect_count_mean": round(float(df["aspect_count"].mean()), 2),
        "aspect_count_median": float(df["aspect_count"].median()),
        "aspect_count_min": int(df["aspect_count"].min()),
        "aspect_count_max": int(df["aspect_count"].max()),
    }
    (OUT_DIR / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def save_counts_and_figures(df: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")

    grade_counts = df["grade"].value_counts().rename_axis("grade").reset_index(name="count")
    style_counts = df["style"].value_counts().rename_axis("style").reset_index(name="count")
    grade_counts.to_csv(OUT_DIR / "grade_counts.csv", index=False)
    style_counts.to_csv(OUT_DIR / "style_counts.csv", index=False)

    grade_style = pd.crosstab(df["grade"], df["style"])
    grade_style.to_csv(OUT_DIR / "grade_style_crosstab.csv")

    aspect_sentiment_rows: List[Dict[str, object]] = []
    all_aspects = sorted({aspect for aspect_map in df["aspects"] for aspect in aspect_map})
    for aspect in all_aspects:
        for sentiment in ["negative", "neutral", "positive"]:
            count = int(sum(1 for aspect_map in df["aspects"] if aspect_map.get(aspect) == sentiment))
            aspect_sentiment_rows.append(
                {"aspect": aspect, "sentiment": sentiment, "count": count}
            )
    aspect_sentiment_df = pd.DataFrame(aspect_sentiment_rows)
    aspect_sentiment_df.to_csv(OUT_DIR / "aspect_sentiment_counts.csv", index=False)

    cooccurrence = pd.DataFrame(0, index=all_aspects, columns=all_aspects, dtype=int)
    for aspect_list in df["aspect_list"]:
        for aspect_i in aspect_list:
            for aspect_j in aspect_list:
                cooccurrence.loc[aspect_i, aspect_j] += 1
    cooccurrence.to_csv(OUT_DIR / "aspect_cooccurrence.csv")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.histplot(df["word_count"], bins=25, color="#2E5E4E", ax=ax)
    ax.set_title("Review Length Distribution")
    ax.set_xlabel("Words per review")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "review_length_distribution.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.8))
    sns.heatmap(grade_style, cmap="YlGnBu", annot=True, fmt="d", ax=ax)
    ax.set_title("Grade by Writing Style")
    ax.set_xlabel("Style")
    ax.set_ylabel("Grade")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "grade_style_heatmap.png", dpi=200)
    plt.close(fig)

    pivot = aspect_sentiment_df.pivot(index="aspect", columns="sentiment", values="count")
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    sns.heatmap(pivot, cmap="YlOrRd", annot=True, fmt="d", ax=ax)
    ax.set_title("Aspect-Sentiment Distribution")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Aspect")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "aspect_sentiment_heatmap.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    sns.heatmap(cooccurrence, cmap="Blues", ax=ax)
    ax.set_title("Aspect Co-occurrence Matrix")
    ax.set_xlabel("Aspect")
    ax.set_ylabel("Aspect")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "aspect_cooccurrence_heatmap.png", dpi=200)
    plt.close(fig)


def select_examples(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    def add_example(label: str, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        row = frame.sort_values(["aspect_count", "word_count"], ascending=[False, False]).iloc[0]
        rows.append(
            {
                "example_type": label,
                "grade": row["grade"],
                "style": row["style"],
                "course_name": row["course_name"],
                "lecturer": row["lecturer"],
                "aspect_count": int(row["aspect_count"]),
                "aspects": json.dumps(row["aspects"], ensure_ascii=False),
                "review_text": row["review_text"],
            }
        )

    mixed = df[
        (df["aspect_count"] >= 2)
        & df["aspects"].map(lambda item: "positive" in item.values() and "negative" in item.values())
    ]
    add_example("mixed_polarity_multi_aspect", mixed)
    add_example("analytic_style", df[df["style"].str.contains("Analytic", case=False, na=False)])
    add_example("casual_style", df[df["style"].str.contains("Casual", case=False, na=False)])
    add_example("rant_style", df[df["style"].str.contains("Rant", case=False, na=False)])
    add_example("short_style", df[df["style"].str.contains("Short", case=False, na=False)])
    add_example("failed_course_review", df[df["grade"].str.contains("F", na=False)])

    example_df = pd.DataFrame(rows).drop_duplicates(subset=["review_text"]).reset_index(drop=True)
    example_df.to_csv(OUT_DIR / "representative_examples.csv", index=False)
    return example_df


def split_three_way(df: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df, holdout = train_test_split(df, test_size=0.20, random_state=seed)
    calib_df, test_df = train_test_split(holdout, test_size=0.50, random_state=seed)
    return (
        train_df.reset_index(drop=True),
        calib_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def build_multilabel_targets(df: pd.DataFrame, mlb: MultiLabelBinarizer) -> np.ndarray:
    return mlb.transform(df["aspect_list"])


def calibrate_thresholds(y_true: np.ndarray, y_prob: np.ndarray, classes: Iterable[str]) -> Dict[str, float]:
    thresholds: Dict[str, float] = {}
    grid = np.linspace(0.05, 0.95, 19)
    for idx, aspect in enumerate(classes):
        best_f1 = -1.0
        best_threshold = 0.50
        for threshold in grid:
            preds = (y_prob[:, idx] >= threshold).astype(int)
            score = f1_score(y_true[:, idx], preds, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_threshold = float(threshold)
        thresholds[str(aspect)] = best_threshold
    return thresholds


def evaluate_detection(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    classes: Iterable[str],
    thresholds: Dict[str, float],
) -> tuple[pd.DataFrame, np.ndarray, Dict[str, float]]:
    class_list = list(classes)
    thr_vec = np.array([thresholds[aspect] for aspect in class_list], dtype=float)
    y_pred = (y_prob >= thr_vec).astype(int)

    aggregate = {
        "micro_precision": float(precision_score(y_true.ravel(), y_pred.ravel(), zero_division=0)),
        "micro_recall": float(recall_score(y_true.ravel(), y_pred.ravel(), zero_division=0)),
        "micro_f1": float(f1_score(y_true.ravel(), y_pred.ravel(), zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }

    rows: List[Dict[str, float]] = []
    n_rows = len(y_true)
    for idx, aspect in enumerate(class_list):
        yt = y_true[:, idx]
        yp = y_pred[:, idx]
        tp = int(np.sum((yp == 1) & (yt == 1)))
        tn = int(np.sum((yp == 0) & (yt == 0)))
        fp = int(np.sum((yp == 1) & (yt == 0)))
        fn = int(np.sum((yp == 0) & (yt == 1)))
        rows.append(
            {
                "aspect": aspect,
                "accuracy": (tp + tn) / n_rows,
                "precision": precision_score(yt, yp, zero_division=0),
                "recall": recall_score(yt, yp, zero_division=0),
                "f1": f1_score(yt, yp, zero_division=0),
                "threshold": thresholds[aspect],
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "n_true": int(np.sum(yt)),
            }
        )
    return pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True), y_pred, aggregate


def train_sentiment_models(train_df: pd.DataFrame, vectorizer: TfidfVectorizer, aspects: List[str]) -> Dict[str, Ridge]:
    models: Dict[str, Ridge] = {}
    train_features = vectorizer.transform(train_df["review_text"])
    for aspect in aspects:
        mask = train_df["aspects"].map(lambda item: aspect in item).to_numpy()
        targets = np.array(
            [SENTIMENT_TO_SCORE[item[aspect]] for item in train_df.loc[mask, "aspects"]],
            dtype=float,
        )
        model = Ridge(alpha=1.0)
        model.fit(train_features[mask], targets)
        models[aspect] = model
    return models


def evaluate_sentiment(
    test_df: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    y_pred_detection: np.ndarray,
    aspects: List[str],
    sentiment_models: Dict[str, Ridge],
) -> tuple[pd.DataFrame, Dict[str, float]]:
    test_features = vectorizer.transform(test_df["review_text"])
    score_lookup = np.array([-1.0, 0.0, 1.0], dtype=float)
    rows: List[Dict[str, float]] = []
    all_true_scores: List[float] = []
    all_pred_scores: List[float] = []
    all_true_labels: List[str] = []
    all_pred_labels: List[str] = []

    for idx, aspect in enumerate(aspects):
        predicted_presence = y_pred_detection[:, idx].astype(bool)
        true_presence = test_df["aspects"].map(lambda item: aspect in item).to_numpy()
        eval_mask = predicted_presence & true_presence
        pred_scores_raw = np.clip(sentiment_models[aspect].predict(test_features), -1.0, 1.0)
        pred_scores = pred_scores_raw[eval_mask]
        true_scores = np.array(
            [
                SENTIMENT_TO_SCORE[item[aspect]]
                for item in test_df.loc[eval_mask, "aspects"]
            ],
            dtype=float,
        )

        if len(true_scores) == 0:
            rows.append(
                {
                    "aspect": aspect,
                    "n_eval": 0,
                    "mse": math.nan,
                    "mae": math.nan,
                    "polarity_accuracy": math.nan,
                }
            )
            continue

        rounded_preds = score_lookup[np.argmin(np.abs(pred_scores[:, None] - score_lookup[None, :]), axis=1)]
        true_labels = ["negative" if score < 0 else "neutral" if score == 0 else "positive" for score in true_scores]
        pred_labels = ["negative" if score < 0 else "neutral" if score == 0 else "positive" for score in rounded_preds]

        all_true_scores.extend(true_scores.tolist())
        all_pred_scores.extend(pred_scores.tolist())
        all_true_labels.extend(true_labels)
        all_pred_labels.extend(pred_labels)

        rows.append(
            {
                "aspect": aspect,
                "n_eval": int(len(true_scores)),
                "mse": mean_squared_error(true_scores, pred_scores),
                "mae": mean_absolute_error(true_scores, pred_scores),
                "polarity_accuracy": float(np.mean(np.array(true_labels) == np.array(pred_labels))),
            }
        )

    overall = {
        "sentiment_mse_detected": float(mean_squared_error(all_true_scores, all_pred_scores)),
        "sentiment_mae_detected": float(mean_absolute_error(all_true_scores, all_pred_scores)),
        "sentiment_polarity_accuracy": float(np.mean(np.array(all_true_labels) == np.array(all_pred_labels))),
        "n_detected_sentiment_pairs": int(len(all_true_scores)),
    }
    return pd.DataFrame(rows).sort_values("mse", ascending=True).reset_index(drop=True), overall


def evaluate_detection_by_group(
    test_df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    column: str,
) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for group_value in sorted(test_df[column].dropna().unique()):
        mask = (test_df[column] == group_value).to_numpy()
        if mask.sum() == 0:
            continue
        rows.append(
            {
                column: group_value,
                "n_reviews": int(mask.sum()),
                "micro_f1": float(f1_score(y_true[mask].ravel(), y_pred[mask].ravel(), zero_division=0)),
                "micro_precision": float(
                    precision_score(y_true[mask].ravel(), y_pred[mask].ravel(), zero_division=0)
                ),
                "micro_recall": float(
                    recall_score(y_true[mask].ravel(), y_pred[mask].ravel(), zero_division=0)
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("micro_f1", ascending=False).reset_index(drop=True)


def run_baseline_experiment(df: pd.DataFrame, seed: int) -> tuple[ExperimentResult, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df, calib_df, test_df = split_three_way(df, seed)
    bundle = train_baseline_models(train_df, calib_df, df)
    x_test = bundle.vectorizer.transform(test_df["review_text"])
    y_test = build_multilabel_targets(test_df, bundle.mlb)

    y_test_prob = bundle.detection_model.predict_proba(x_test)
    detection_df, y_test_pred, detection_aggregate = evaluate_detection(
        y_test,
        y_test_prob,
        bundle.mlb.classes_,
        bundle.thresholds,
    )
    detection_df.to_csv(OUT_DIR / f"baseline_detection_per_aspect_seed_{seed}.csv", index=False)
    pd.DataFrame([bundle.thresholds]).to_csv(OUT_DIR / f"baseline_thresholds_seed_{seed}.csv", index=False)

    sentiment_df, sentiment_aggregate = evaluate_sentiment(
        test_df,
        bundle.vectorizer,
        y_test_pred,
        list(bundle.mlb.classes_),
        bundle.sentiment_models,
    )
    sentiment_df.to_csv(OUT_DIR / f"baseline_sentiment_per_aspect_seed_{seed}.csv", index=False)

    style_df = evaluate_detection_by_group(test_df, y_test, y_test_pred, "style")
    grade_df = evaluate_detection_by_group(test_df, y_test, y_test_pred, "grade")
    style_df.to_csv(OUT_DIR / f"baseline_style_metrics_seed_{seed}.csv", index=False)
    grade_df.to_csv(OUT_DIR / f"baseline_grade_metrics_seed_{seed}.csv", index=False)

    if seed == 42:
        plot_group_metrics(style_df, "style", FIG_DIR / "baseline_style_micro_f1.png", "Baseline Micro-F1 by Style")
        plot_group_metrics(grade_df, "grade", FIG_DIR / "baseline_grade_micro_f1.png", "Baseline Micro-F1 by Grade")
        plot_detection_and_sentiment(detection_df, sentiment_df)

    result = ExperimentResult(
        seed=seed,
        n_train=len(train_df),
        n_calib=len(calib_df),
        n_test=len(test_df),
        micro_precision=detection_aggregate["micro_precision"],
        micro_recall=detection_aggregate["micro_recall"],
        micro_f1=detection_aggregate["micro_f1"],
        macro_precision=detection_aggregate["macro_precision"],
        macro_recall=detection_aggregate["macro_recall"],
        macro_f1=detection_aggregate["macro_f1"],
        sentiment_mse_detected=sentiment_aggregate["sentiment_mse_detected"],
        sentiment_mae_detected=sentiment_aggregate["sentiment_mae_detected"],
        sentiment_polarity_accuracy=sentiment_aggregate["sentiment_polarity_accuracy"],
    )
    return result, detection_df, sentiment_df, style_df, grade_df


def train_baseline_models(train_df: pd.DataFrame, calib_df: pd.DataFrame, full_df: pd.DataFrame) -> ModelBundle:
    all_aspects = sorted({aspect for items in full_df["aspects"] for aspect in items})

    mlb = MultiLabelBinarizer(classes=all_aspects)
    mlb.fit(full_df["aspect_list"])

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_features=20000,
        sublinear_tf=True,
    )
    x_train = vectorizer.fit_transform(train_df["review_text"])
    x_calib = vectorizer.transform(calib_df["review_text"])

    y_train = build_multilabel_targets(train_df, mlb)
    y_calib = build_multilabel_targets(calib_df, mlb)

    detection_model = OneVsRestClassifier(
        LogisticRegression(
            max_iter=1000,
            solver="liblinear",
            class_weight="balanced",
        )
    )
    detection_model.fit(x_train, y_train)

    y_calib_prob = detection_model.predict_proba(x_calib)
    thresholds = calibrate_thresholds(y_calib, y_calib_prob, mlb.classes_)
    sentiment_models = train_sentiment_models(train_df, vectorizer, list(mlb.classes_))
    return ModelBundle(
        vectorizer=vectorizer,
        mlb=mlb,
        detection_model=detection_model,
        thresholds=thresholds,
        sentiment_models=sentiment_models,
    )


def plot_group_metrics(df: pd.DataFrame, label_column: str, output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.6))
    plot_df = df.sort_values("micro_f1", ascending=True)
    sns.barplot(plot_df, x="micro_f1", y=label_column, color="#3B6F8C", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Micro-F1")
    ax.set_ylabel("")
    ax.set_xlim(0.0, 1.0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_detection_and_sentiment(detection_df: pd.DataFrame, sentiment_df: pd.DataFrame) -> None:
    merged = detection_df[["aspect", "f1"]].merge(
        sentiment_df[["aspect", "mse"]],
        on="aspect",
        how="left",
    )
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    plot_df = merged.sort_values("f1", ascending=True)
    ax.barh(plot_df["aspect"], plot_df["f1"], color="#4E9A81", label="Detection F1")
    ax.set_xlabel("Detection F1")
    ax.set_title("Baseline Per-Aspect Detection F1")
    ax.set_xlim(0.0, 1.0)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "baseline_detection_f1_by_aspect.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    plot_df = sentiment_df.sort_values("mse", ascending=False)
    sns.barplot(plot_df, x="mse", y="aspect", color="#D08B55", ax=ax)
    ax.set_xlabel("Sentiment MSE on detected true aspects")
    ax.set_ylabel("")
    ax.set_title("Baseline Per-Aspect Sentiment Error")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "baseline_sentiment_mse_by_aspect.png", dpi=200)
    plt.close(fig)


def run_learning_curve_experiment(df: pd.DataFrame, seed: int = 42) -> None:
    train_df, calib_df, test_df = split_three_way(df, seed)
    rng = np.random.default_rng(seed)
    fractions = [0.25, 0.50, 0.75, 1.00]
    rows: List[Dict[str, float]] = []

    for fraction in fractions:
        if fraction < 1.0:
            sample_size = max(200, int(len(train_df) * fraction))
            indices = rng.choice(train_df.index.to_numpy(), size=sample_size, replace=False)
            sub_train_df = train_df.loc[np.sort(indices)].reset_index(drop=True)
        else:
            sub_train_df = train_df.copy()

        bundle = train_baseline_models(sub_train_df, calib_df, df)
        x_test = bundle.vectorizer.transform(test_df["review_text"])
        y_test = build_multilabel_targets(test_df, bundle.mlb)
        y_test_prob = bundle.detection_model.predict_proba(x_test)
        _, y_test_pred, detection_aggregate = evaluate_detection(
            y_test,
            y_test_prob,
            bundle.mlb.classes_,
            bundle.thresholds,
        )
        _, sentiment_aggregate = evaluate_sentiment(
            test_df,
            bundle.vectorizer,
            y_test_pred,
            list(bundle.mlb.classes_),
            bundle.sentiment_models,
        )
        rows.append(
            {
                "train_fraction": fraction,
                "n_train": int(len(sub_train_df)),
                "micro_f1": detection_aggregate["micro_f1"],
                "macro_f1": detection_aggregate["macro_f1"],
                "sentiment_mse_detected": sentiment_aggregate["sentiment_mse_detected"],
                "sentiment_polarity_accuracy": sentiment_aggregate["sentiment_polarity_accuracy"],
            }
        )

    result_df = pd.DataFrame(rows)
    result_df.to_csv(OUT_DIR / "learning_curve_results.csv", index=False)

    fig, ax1 = plt.subplots(figsize=(7.8, 4.6))
    ax1.plot(result_df["n_train"], result_df["micro_f1"], marker="o", color="#397367", label="Detection micro-F1")
    ax1.plot(result_df["n_train"], result_df["macro_f1"], marker="o", color="#7A9E7E", label="Detection macro-F1")
    ax1.set_xlabel("Training reviews")
    ax1.set_ylabel("F1")
    ax1.set_ylim(0.0, 1.0)
    ax2 = ax1.twinx()
    ax2.plot(
        result_df["n_train"],
        result_df["sentiment_mse_detected"],
        marker="s",
        color="#D17B49",
        label="Sentiment MSE",
    )
    ax2.set_ylabel("Sentiment MSE")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="lower right")
    ax1.set_title("Learning Curve on the Synthetic ABSA Dataset")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "learning_curve.png", dpi=200)
    plt.close(fig)


def run_style_holdout_experiment(df: pd.DataFrame) -> None:
    rows: List[Dict[str, float]] = []
    styles = sorted(df["style"].dropna().unique())
    all_aspects = sorted({aspect for aspect_map in df["aspects"] for aspect in aspect_map})

    for style in styles:
        test_df = df[df["style"] == style].reset_index(drop=True)
        pool_df = df[df["style"] != style].reset_index(drop=True)
        train_df, calib_df = train_test_split(pool_df, test_size=0.10, random_state=42)
        train_df = train_df.reset_index(drop=True)
        calib_df = calib_df.reset_index(drop=True)

        bundle = train_baseline_models(train_df, calib_df, df)
        x_test = bundle.vectorizer.transform(test_df["review_text"])
        y_test = build_multilabel_targets(test_df, bundle.mlb)
        y_test_prob = bundle.detection_model.predict_proba(x_test)
        _, y_test_pred, detection_aggregate = evaluate_detection(
            y_test,
            y_test_prob,
            all_aspects,
            bundle.thresholds,
        )
        _, sentiment_aggregate = evaluate_sentiment(
            test_df,
            bundle.vectorizer,
            y_test_pred,
            all_aspects,
            bundle.sentiment_models,
        )
        rows.append(
            {
                "held_out_style": style,
                "n_test": int(len(test_df)),
                "micro_f1": detection_aggregate["micro_f1"],
                "macro_f1": detection_aggregate["macro_f1"],
                "sentiment_mse_detected": sentiment_aggregate["sentiment_mse_detected"],
                "sentiment_polarity_accuracy": sentiment_aggregate["sentiment_polarity_accuracy"],
            }
        )

    result_df = pd.DataFrame(rows).sort_values("micro_f1", ascending=False).reset_index(drop=True)
    result_df.to_csv(OUT_DIR / "style_holdout_results.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    plot_df = result_df.sort_values("micro_f1", ascending=True)
    sns.barplot(plot_df, x="micro_f1", y="held_out_style", color="#5C80BC", ax=ax)
    ax.set_title("Held-Out Style Generalization")
    ax.set_xlabel("Detection micro-F1 on unseen style")
    ax.set_ylabel("")
    ax.set_xlim(0.0, 1.0)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "style_holdout_micro_f1.png", dpi=200)
    plt.close(fig)


def summarize_seed_results(results: List[ExperimentResult]) -> None:
    rows = [result.__dict__ for result in results]
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "baseline_seed_summary.csv", index=False)

    summary = {
        metric: {
            "mean": round(float(df[metric].mean()), 4),
            "std": round(float(df[metric].std(ddof=0)), 4),
        }
        for metric in [
            "micro_precision",
            "micro_recall",
            "micro_f1",
            "macro_precision",
            "macro_recall",
            "macro_f1",
            "sentiment_mse_detected",
            "sentiment_mae_detected",
            "sentiment_polarity_accuracy",
        ]
    }
    (OUT_DIR / "baseline_seed_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def extract_notebook_outputs(notebook_path: Path) -> None:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))

    test_output_text = ""
    threshold_text = ""
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        outputs = cell.get("outputs", [])
        combined = []
        for output in outputs:
            if "text" in output:
                combined.append("".join(output["text"]))
            elif "data" in output and "text/plain" in output["data"]:
                combined.append("".join(output["data"]["text/plain"]))
        text = "\n".join(combined)
        if "=== Step 4: Test Evaluation ===" in text:
            test_output_text = text
        if "Per-aspect best threshold (F1-optimal on calib)" in text:
            threshold_text = text

    threshold_rows = []
    for line in threshold_text.splitlines():
        match = re.search(r"^\s*([a-z_]+)\s*:\s*threshold=([0-9.]+)\s+\(F1=([0-9.]+)\)", line)
        if match:
            threshold_rows.append(
                {
                    "aspect": match.group(1),
                    "threshold": float(match.group(2)),
                    "calib_f1": float(match.group(3)),
                }
            )
    pd.DataFrame(threshold_rows).to_csv(OUT_DIR / "recorded_notebook_thresholds.csv", index=False)

    metric_rows = []
    for line in test_output_text.splitlines():
        match = re.search(
            r"^\s*([a-z_]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)",
            line,
        )
        if match:
            metric_rows.append(
                {
                    "aspect": match.group(1),
                    "accuracy": float(match.group(2)),
                    "precision": float(match.group(3)),
                    "recall": float(match.group(4)),
                    "mse": float(match.group(5)),
                    "threshold": float(match.group(6)),
                    "tp": int(match.group(7)),
                    "tn": int(match.group(8)),
                    "fp": int(match.group(9)),
                    "fn": int(match.group(10)),
                }
            )
    pd.DataFrame(metric_rows).to_csv(OUT_DIR / "recorded_notebook_test_results.csv", index=False)


def main() -> None:
    ensure_dirs()
    df = load_reviews(DATA_PATH)
    save_dataset_summary(df)
    save_counts_and_figures(df)
    select_examples(df)
    extract_notebook_outputs(NOTEBOOK_PATH)

    results: List[ExperimentResult] = []
    for seed in SEEDS:
        result, detection_df, sentiment_df, style_df, grade_df = run_baseline_experiment(df, seed)
        results.append(result)
        if seed == 42:
            detection_df.to_csv(OUT_DIR / "baseline_detection_per_aspect_main.csv", index=False)
            sentiment_df.to_csv(OUT_DIR / "baseline_sentiment_per_aspect_main.csv", index=False)
            style_df.to_csv(OUT_DIR / "baseline_style_metrics_main.csv", index=False)
            grade_df.to_csv(OUT_DIR / "baseline_grade_metrics_main.csv", index=False)

    summarize_seed_results(results)
    run_learning_curve_experiment(df, seed=42)
    run_style_holdout_experiment(df)
    print(f"Saved analysis artifacts to: {OUT_DIR}")


if __name__ == "__main__":
    main()
