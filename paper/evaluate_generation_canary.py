from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict

from absa_data_io import dataset_summary, load_absa_dataset
from consume_generation_batch import extract_output_text, iter_result_rows, load_manifest


ROOT = Path(__file__).resolve().parents[1]
BATCH_DIR = ROOT / "paper" / "batch_requests"
RESULTS_DIR = ROOT / "paper" / "batch_results"
DATASET_DIR = ROOT / "paper" / "generated_datasets"


LENGTH_BANDS = {
    "very short comment": (20, 45),
    "compact but informative review": (45, 85),
    "mid-length reflective review": (85, 140),
    "detailed review with one dominant complaint": (140, 220),
}


def normalize_text(text: str) -> str:
    return " ".join(text.split())


def evaluate_lengths(manifest: Dict[str, Dict[str, str]], results_path: Path) -> Dict[str, float]:
    checked = 0
    matched = 0
    completed = 0
    with_text = 0
    duplicates = 0
    seen = set()
    words = []

    for item in iter_result_rows(results_path):
        custom_id = item.get("custom_id")
        meta = manifest.get(custom_id)
        if meta is None:
            continue
        checked += 1
        body = item.get("response", {}).get("body", {})
        if body.get("status") == "completed":
            completed += 1
        text = extract_output_text(body)
        if not text:
            continue
        with_text += 1
        cleaned = normalize_text(text)
        wc = len(cleaned.split())
        words.append(wc)
        if cleaned in seen:
            duplicates += 1
        seen.add(cleaned)
        nuances = json.loads(meta["nuance_attributes"])
        band = str(nuances.get("review_length_band", "")).strip().lower()
        if band in LENGTH_BANDS:
            lo, hi = LENGTH_BANDS[band]
            if lo <= wc <= hi:
                matched += 1

    return {
        "checked_rows": checked,
        "completed_rate": round(completed / max(checked, 1), 4),
        "text_success_rate": round(with_text / max(checked, 1), 4),
        "duplicate_rate": round(duplicates / max(with_text, 1), 4),
        "length_band_match_rate": round(matched / max(with_text, 1), 4),
        "word_count_mean": round(sum(words) / len(words), 2) if words else 0.0,
        "word_count_min": min(words) if words else 0,
        "word_count_max": max(words) if words else 0,
    }


def loader_check(dataset_path: Path) -> Dict[str, object]:
    try:
        df = load_absa_dataset(dataset_path)
    except Exception as exc:
        return {"loader_ok": False, "loader_error": str(exc)}
    summary = dataset_summary(df)
    return {"loader_ok": True, "dataset_summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run acceptance checks for a canary generation batch.")
    parser.add_argument("--generation-prefix", required=True)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--text-success-threshold", type=float, default=0.95)
    parser.add_argument("--completed-threshold", type=float, default=0.95)
    parser.add_argument("--duplicate-threshold", type=float, default=0.10)
    parser.add_argument("--length-match-threshold", type=float, default=0.70)
    args = parser.parse_args()

    manifest_path = BATCH_DIR / f"{args.generation_prefix}_manifest.csv"
    results_path = RESULTS_DIR / f"{args.batch_id}_output.jsonl"
    dataset_path = DATASET_DIR / f"{args.batch_id}_generated_reviews.jsonl"

    manifest = load_manifest(manifest_path)
    metrics = evaluate_lengths(manifest, results_path)
    loader = loader_check(dataset_path)
    verdict = (
        metrics["text_success_rate"] >= args.text_success_threshold
        and metrics["completed_rate"] >= args.completed_threshold
        and metrics["duplicate_rate"] <= args.duplicate_threshold
        and metrics["length_band_match_rate"] >= args.length_match_threshold
        and loader.get("loader_ok", False)
    )

    summary = {
        "generation_prefix": args.generation_prefix,
        "batch_id": args.batch_id,
        "metrics": metrics,
        "loader": loader,
        "thresholds": {
            "text_success_threshold": args.text_success_threshold,
            "completed_threshold": args.completed_threshold,
            "duplicate_threshold": args.duplicate_threshold,
            "length_match_threshold": args.length_match_threshold,
        },
        "accepted": verdict,
    }
    summary_path = BATCH_DIR / f"{args.generation_prefix}_acceptance_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
