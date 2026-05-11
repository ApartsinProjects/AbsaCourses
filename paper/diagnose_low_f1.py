from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def dataset_diagnostics(dataset_path: Path) -> Dict[str, Any]:
    df = pd.read_json(dataset_path, lines=True)
    aspect_counts = df["target_attributes"].apply(len)
    counts: Dict[str, int] = {}
    for labels in df["target_attributes"]:
        for aspect in labels:
            counts[aspect] = counts.get(aspect, 0) + 1
    return {
        "rows": int(len(df)),
        "mean_words": round(float(df["text"].str.split().apply(len).mean()), 2),
        "median_words": float(df["text"].str.split().apply(len).median()),
        "aspect_labels_total": int(sum(counts.values())),
        "aspect_inventory_size": int(len(counts)),
        "aspects_per_row_mean": round(float(aspect_counts.mean()), 3),
        "aspects_per_row_distribution": {str(k): int(v) for k, v in aspect_counts.value_counts().sort_index().items()},
        "duplicate_texts": int(df["text"].duplicated().sum()),
        "top_aspects": sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:8],
        "bottom_aspects": sorted(counts.items(), key=lambda kv: kv[1])[:8],
    }


def prediction_diagnostics(sample_predictions_path: Path) -> Dict[str, Any]:
    rows = load_jsonl(sample_predictions_path)
    aspects = sorted(rows[0]["detection_targets"])
    gold = np.array([[int(row["detection_targets"][a]) for a in aspects] for row in rows], dtype=int)
    preds = np.array([[int(row["detection_predictions"][a]) for a in aspects] for row in rows], dtype=int)
    probs = np.array([[float(row["detection_probabilities"][a]) for a in aspects] for row in rows], dtype=float)
    thresholds = {a: float(rows[0]["thresholds"][a]) for a in aspects}

    ap_scores = []
    auroc_scores = []
    prevalence_shift = []
    separation = []
    weak_f1 = []
    for idx, aspect in enumerate(aspects):
        y = gold[:, idx]
        s = probs[:, idx]
        p = preds[:, idx]
        if y.min() != y.max():
            ap_scores.append((aspect, float(average_precision_score(y, s))))
            auroc_scores.append((aspect, float(roc_auc_score(y, s))))
        prevalence_shift.append((aspect, float(y.mean()), float(p.mean()), float(p.mean() - y.mean())))
        pos = s[y == 1]
        neg = s[y == 0]
        separation.append((aspect, float(pos.mean()), float(neg.mean()), float(pos.mean() - neg.mean())))
        weak_f1.append((aspect, float(f1_score(y, p, zero_division=0))))

    oracle_k_preds = np.zeros_like(gold)
    for row_idx, row in enumerate(rows):
        k = max(1, sum(int(v) for v in row["detection_targets"].values()))
        top_idx = np.argsort(-probs[row_idx])[:k]
        oracle_k_preds[row_idx, top_idx] = 1

    return {
        "n_eval_rows": int(len(rows)),
        "gold_positive_rate": round(float(gold.mean()), 4),
        "pred_positive_rate": round(float(preds.mean()), 4),
        "threshold_summary": {
            "min": round(float(min(thresholds.values())), 3),
            "max": round(float(max(thresholds.values())), 3),
            "mean": round(float(np.mean(list(thresholds.values()))), 3),
            "thresholds": {k: round(v, 3) for k, v in thresholds.items()},
        },
        "macro_ap": round(float(np.mean([score for _, score in ap_scores])), 4),
        "macro_auroc": round(float(np.mean([score for _, score in auroc_scores])), 4),
        "oracle_k_micro_f1": round(float(f1_score(gold.ravel(), oracle_k_preds.ravel(), zero_division=0)), 4),
        "oracle_k_samples_f1": round(float(f1_score(gold, oracle_k_preds, average="samples", zero_division=0)), 4),
        "most_overpredicted_aspects": [
            {
                "aspect": aspect,
                "gold_rate": round(gold_rate, 4),
                "pred_rate": round(pred_rate, 4),
                "delta": round(delta, 4),
            }
            for aspect, gold_rate, pred_rate, delta in sorted(prevalence_shift, key=lambda item: abs(item[3]), reverse=True)[:8]
        ],
        "weakest_aspects_by_ap": [
            {"aspect": aspect, "average_precision": round(score, 4)}
            for aspect, score in sorted(ap_scores, key=lambda item: item[1])[:8]
        ],
        "weakest_aspects_by_f1": [
            {"aspect": aspect, "f1": round(score, 4)}
            for aspect, score in sorted(weak_f1, key=lambda item: item[1])[:8]
        ],
        "weakest_separation_aspects": [
            {
                "aspect": aspect,
                "positive_mean_prob": round(pos_mean, 4),
                "negative_mean_prob": round(neg_mean, 4),
                "separation": round(delta, 4),
            }
            for aspect, pos_mean, neg_mean, delta in sorted(separation, key=lambda item: item[3])[:8]
        ],
    }


def faithfulness_diagnostics(details_path: Path) -> Dict[str, Any]:
    df = pd.read_csv(details_path)
    df = df[df["aspect"] != "__row_summary__"].copy()
    return {
        "n_aspect_judgments": int(len(df)),
        "aspect_supported_rate": round(float(df["supported"].mean()), 4),
        "aspect_sentiment_match_rate": round(float(df["sentiment_match"].mean()), 4),
        "weakest_support_aspects": {
            key: round(float(value), 4)
            for key, value in df.groupby("aspect")["supported"].mean().sort_values().head(8).items()
        },
        "weakest_sentiment_match_aspects": {
            key: round(float(value), 4)
            for key, value in df.groupby("aspect")["sentiment_match"].mean().sort_values().head(8).items()
        },
    }


def render_markdown(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Low-F1 Diagnosis")
    lines.append("")
    lines.append("## Dataset")
    ds = report["dataset"]
    lines.append(f"- Rows: `{ds['rows']}`")
    lines.append(f"- Mean words per review: `{ds['mean_words']}`")
    lines.append(f"- Mean aspects per review: `{ds['aspects_per_row_mean']}`")
    lines.append(f"- Duplicate texts: `{ds['duplicate_texts']}`")
    lines.append("")
    lines.append("## Prediction Behavior")
    pdx = report["predictions"]
    lines.append(f"- Gold positive rate: `{pdx['gold_positive_rate']}`")
    lines.append(f"- Predicted positive rate: `{pdx['pred_positive_rate']}`")
    lines.append(f"- Macro average precision: `{pdx['macro_ap']}`")
    lines.append(f"- Macro AUROC: `{pdx['macro_auroc']}`")
    lines.append(f"- Oracle-k micro-F1: `{pdx['oracle_k_micro_f1']}`")
    lines.append("")
    lines.append("Most overpredicted aspects:")
    for row in pdx["most_overpredicted_aspects"]:
        lines.append(f"- `{row['aspect']}`: gold `{row['gold_rate']}`, predicted `{row['pred_rate']}`, delta `{row['delta']}`")
    lines.append("")
    lines.append("Weakest aspects by ranking quality:")
    for row in pdx["weakest_aspects_by_ap"]:
        lines.append(f"- `{row['aspect']}`: AP `{row['average_precision']}`")
    lines.append("")
    if "faithfulness" in report:
        fd = report["faithfulness"]
        lines.append("## Faithfulness")
        lines.append(f"- Aspect support rate: `{fd['aspect_supported_rate']}`")
        lines.append(f"- Aspect sentiment-match rate: `{fd['aspect_sentiment_match_rate']}`")
        lines.append("")
        lines.append("Weakest support aspects:")
        for aspect, score in fd["weakest_support_aspects"].items():
            lines.append(f"- `{aspect}`: `{score}`")
        lines.append("")
        lines.append("Weakest sentiment-match aspects:")
        for aspect, score in fd["weakest_sentiment_match_aspects"].items():
            lines.append(f"- `{aspect}`: `{score}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose low-F1 behavior for CourseABSA runs.")
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--sample-predictions", type=Path, required=True)
    parser.add_argument("--faithfulness-details", type=Path)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    args = parser.parse_args()

    report: Dict[str, Any] = {
        "dataset": dataset_diagnostics(args.dataset_path),
        "predictions": prediction_diagnostics(args.sample_predictions),
    }
    if args.faithfulness_details:
        report["faithfulness"] = faithfulness_diagnostics(args.faithfulness_details)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    args.out_md.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"out_json": str(args.out_json), "out_md": str(args.out_md)}, indent=2))


if __name__ == "__main__":
    main()
