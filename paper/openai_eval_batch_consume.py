from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score

from absa_model_comparison import SENT2VAL, discover_aspects, extract_json_block, load_jsonl, parse_aspect_map


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "paper" / "benchmark_outputs"
RUNS_DIR = OUT_DIR / "runs"
REGISTRY_PATH = OUT_DIR / "experiment_registry.jsonl"


def safe_specificity(tp: int, tn: int, fp: int, fn: int) -> float:
    denom = tn + fp
    return float(tn / denom) if denom else 0.0


def safe_balanced_accuracy(tp: int, tn: int, fp: int, fn: int) -> float:
    recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
    specificity = safe_specificity(tp, tn, fp, fn)
    return float((recall + specificity) / 2.0)


def safe_mcc(tp: int, tn: int, fp: int, fn: int) -> float:
    numerator = (tp * tn) - (fp * fn)
    denom = float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    return float(numerator / denom) if denom else 0.0


def multilabel_detection_metrics(true: np.ndarray, preds: np.ndarray) -> dict:
    per_aspect_balanced = []
    per_aspect_specificity = []
    per_aspect_mcc = []
    for idx in range(true.shape[1]):
        yt = true[:, idx].astype(int)
        yp = preds[:, idx].astype(int)
        tp = int(((yp == 1) & (yt == 1)).sum())
        tn = int(((yp == 0) & (yt == 0)).sum())
        fp = int(((yp == 1) & (yt == 0)).sum())
        fn = int(((yp == 0) & (yt == 1)).sum())
        per_aspect_balanced.append(safe_balanced_accuracy(tp, tn, fp, fn))
        per_aspect_specificity.append(safe_specificity(tp, tn, fp, fn))
        per_aspect_mcc.append(safe_mcc(tp, tn, fp, fn))
    return {
        "label_accuracy": float((true == preds).mean()),
        "subset_accuracy": float(accuracy_score(true, preds)),
        "samples_f1": float(f1_score(true, preds, average="samples", zero_division=0)),
        "samples_jaccard": float(jaccard_score(true, preds, average="samples", zero_division=0)),
        "macro_balanced_accuracy": float(np.mean(per_aspect_balanced)),
        "macro_specificity": float(np.mean(per_aspect_specificity)),
        "macro_mcc": float(np.mean(per_aspect_mcc)),
    }


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)


def make_run_dir(prefix: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    candidate = RUNS_DIR / f"{prefix}_{timestamp}"
    suffix = 1
    while candidate.exists():
        suffix += 1
        candidate = RUNS_DIR / f"{prefix}_{timestamp}_{suffix}"
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def append_registry_entry(entry: dict) -> None:
    with REGISTRY_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def extract_output_text(row: dict) -> str:
    body = row.get("response", {}).get("body", {})
    for item in body.get("output", []):
        if item.get("type") == "message":
            for content in item.get("content", []):
                if content.get("type") == "output_text":
                    return str(content.get("text", "")).strip()
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Consume a completed OpenAI ABSA evaluation batch.")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--results-path", required=True)
    parser.add_argument("--model-tag", default="")
    args = parser.parse_args()

    ensure_dirs()
    df = load_jsonl(Path(args.data_path))
    aspects = discover_aspects(df)
    manifest = pd.read_csv(args.manifest)

    by_custom = {}
    with Path(args.results_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            by_custom[row["custom_id"]] = row

    summary_rows = []
    per_aspect_rows = []
    response_rows = []

    grouping = ["model", "variant"] if "model" in manifest.columns else ["variant"]

    for group_key, subset in manifest.groupby(grouping, dropna=False):
        subset = subset.reset_index(drop=True)
        if isinstance(group_key, tuple):
            model, variant = group_key
        else:
            model, variant = (args.model_tag or "openai-batch"), group_key
        det_true, det_pred = [], []
        sentiment_true, sentiment_pred = [], []
        approach_name = (
            str(subset.loc[0, "approach"])
            if "approach" in subset.columns
            else f"openai-{model}-{variant}" if model else f"{args.model_tag}-{variant}"
        )
        parse_successes = 0

        for _, row in subset.iterrows():
            result = by_custom.get(row["custom_id"], {})
            output_text = extract_output_text(result)
            parsed = {}
            if output_text:
                try:
                    parsed = extract_json_block(output_text)
                    parse_successes += 1
                except Exception:
                    parsed = {}
            pred_map = parse_aspect_map(parsed, aspects)
            gold_map = json.loads(row["gold_aspects"])
            response_rows.append(
                {
                    "approach": approach_name,
                    "model": model,
                    "variant": variant,
                    "custom_id": row["custom_id"],
                    "text": row["text"],
                    "gold_aspects": gold_map,
                    "raw_response_text": output_text,
                    "parsed_response": parsed,
                    "predicted_aspects": pred_map,
                    "response_status": result.get("response", {}).get("status", ""),
                }
            )
            y_true = np.array([1 if aspect in gold_map else 0 for aspect in aspects], dtype=int)
            y_pred = np.array([1 if aspect in pred_map else 0 for aspect in aspects], dtype=int)
            det_true.append(y_true)
            det_pred.append(y_pred)
            for aspect in aspects:
                if aspect in pred_map and aspect in gold_map:
                    sentiment_true.append(SENT2VAL[gold_map[aspect]])
                    sentiment_pred.append(SENT2VAL[pred_map[aspect]])

        det_true_arr = np.vstack(det_true)
        det_pred_arr = np.vstack(det_pred)
        for idx, aspect in enumerate(aspects):
            yt = det_true_arr[:, idx]
            yp = det_pred_arr[:, idx]
            per_aspect_rows.append(
                {
                    "approach": approach_name,
                    "aspect": aspect,
                    "accuracy": float((yt == yp).mean()),
                    "precision": precision_score(yt, yp, zero_division=0),
                    "recall": recall_score(yt, yp, zero_division=0),
                    "f1": f1_score(yt, yp, zero_division=0),
                    "specificity": safe_specificity(int(((yp == 1) & (yt == 1)).sum()), int(((yp == 0) & (yt == 0)).sum()), int(((yp == 1) & (yt == 0)).sum()), int(((yp == 0) & (yt == 1)).sum())),
                    "balanced_accuracy": safe_balanced_accuracy(int(((yp == 1) & (yt == 1)).sum()), int(((yp == 0) & (yt == 0)).sum()), int(((yp == 1) & (yt == 0)).sum()), int(((yp == 0) & (yt == 1)).sum())),
                    "mcc": safe_mcc(int(((yp == 1) & (yt == 1)).sum()), int(((yp == 0) & (yt == 0)).sum()), int(((yp == 1) & (yt == 0)).sum()), int(((yp == 0) & (yt == 1)).sum())),
                    "mse": float("nan"),
                    "threshold": float("nan"),
                    "tp": int(((yp == 1) & (yt == 1)).sum()),
                    "tn": int(((yp == 0) & (yt == 0)).sum()),
                    "fp": int(((yp == 1) & (yt == 0)).sum()),
                    "fn": int(((yp == 0) & (yt == 1)).sum()),
                }
            )

        detection_metrics = {
            "micro_precision": float(precision_score(det_true_arr.ravel(), det_pred_arr.ravel(), zero_division=0)),
            "micro_recall": float(recall_score(det_true_arr.ravel(), det_pred_arr.ravel(), zero_division=0)),
            "micro_f1": float(f1_score(det_true_arr.ravel(), det_pred_arr.ravel(), zero_division=0)),
            "macro_precision": float(precision_score(det_true_arr, det_pred_arr, average="macro", zero_division=0)),
            "macro_recall": float(recall_score(det_true_arr, det_pred_arr, average="macro", zero_division=0)),
            "macro_f1": float(f1_score(det_true_arr, det_pred_arr, average="macro", zero_division=0)),
        }
        detection_metrics.update(multilabel_detection_metrics(det_true_arr, det_pred_arr))
        summary_rows.append(
            {
                "approach": approach_name,
                "micro_precision": detection_metrics["micro_precision"],
                "micro_recall": detection_metrics["micro_recall"],
                "micro_f1": detection_metrics["micro_f1"],
                "macro_precision": detection_metrics["macro_precision"],
                "macro_recall": detection_metrics["macro_recall"],
                "macro_f1": detection_metrics["macro_f1"],
                "macro_balanced_accuracy": detection_metrics["macro_balanced_accuracy"],
                "macro_specificity": detection_metrics["macro_specificity"],
                "macro_mcc": detection_metrics["macro_mcc"],
                "label_accuracy": detection_metrics["label_accuracy"],
                "subset_accuracy": detection_metrics["subset_accuracy"],
                "samples_f1": detection_metrics["samples_f1"],
                "samples_jaccard": detection_metrics["samples_jaccard"],
                "sentiment_mse_detected": float(np.mean([(p - t) ** 2 for p, t in zip(sentiment_pred, sentiment_true)])) if sentiment_true else float("nan"),
                "elapsed_seconds": float("nan"),
                "model": model,
                "variant": variant,
                "n_examples": int(len(subset)),
                "parse_success_rate": float(parse_successes / len(subset)) if len(subset) else float("nan"),
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("micro_f1", ascending=False).reset_index(drop=True)
    per_aspect_df = pd.DataFrame(per_aspect_rows)
    summary_path = OUT_DIR / "openai_batch_eval_summary.csv"
    per_aspect_path = OUT_DIR / "openai_batch_eval_per_aspect.csv"
    latest_metadata_path = OUT_DIR / "openai_batch_eval_metadata.json"
    summary_df.to_csv(summary_path, index=False)
    per_aspect_df.to_csv(per_aspect_path, index=False)
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_path": str(Path(args.data_path).resolve()),
        "manifest": str(Path(args.manifest).resolve()),
        "results_path": str(Path(args.results_path).resolve()),
        "model_tag": args.model_tag,
        "grouping": grouping,
        "n_rows": int(len(df)),
        "n_manifest_rows": int(len(manifest)),
        "approaches": summary_df["approach"].tolist(),
        "models": sorted(summary_df["model"].dropna().astype(str).unique().tolist()) if "model" in summary_df.columns else [],
        "variants": sorted(summary_df["variant"].dropna().astype(str).unique().tolist()) if "variant" in summary_df.columns else [],
        "artifact_policy": {
            "raw_batch_jsonl_external": True,
            "parsed_llm_responses": True,
            "predicted_aspect_maps": True,
        },
        "command": [
            "python",
            __file__,
            "--data-path",
            args.data_path,
            "--manifest",
            args.manifest,
            "--results-path",
            args.results_path,
            "--model-tag",
            args.model_tag,
        ],
    }
    latest_metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    run_dir = make_run_dir("openai_batch_eval")
    run_summary_path = run_dir / "summary.csv"
    run_per_aspect_path = run_dir / "per_aspect.csv"
    run_metadata_path = run_dir / "metadata.json"
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    run_response_path = artifacts_dir / "llm_responses.jsonl"
    summary_df.to_csv(run_summary_path, index=False)
    per_aspect_df.to_csv(run_per_aspect_path, index=False)
    with run_response_path.open("w", encoding="utf-8") as handle:
        for record in response_rows:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    run_metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    append_registry_entry(
        {
            "run_dir": str(run_dir),
            "prefix": "openai_batch_eval",
            "created_at_utc": metadata["created_at_utc"],
            "files": {
                "summary": str(run_summary_path),
                "per_aspect": str(run_per_aspect_path),
                "metadata": str(run_metadata_path),
                "llm_responses": str(run_response_path),
                "latest_summary": str(summary_path),
                "latest_per_aspect": str(per_aspect_path),
                "latest_metadata": str(latest_metadata_path),
            },
            "approaches": metadata["approaches"],
        }
    )
    print(
        json.dumps(
            {
                "summary": str(summary_path),
                "per_aspect": str(per_aspect_path),
                "run_dir": str(run_dir),
                "metadata": str(run_metadata_path),
                "llm_responses": str(run_response_path),
                "latest_metadata": str(latest_metadata_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
