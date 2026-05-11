from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "paper" / "benchmark_outputs" / "runs"
ANALYSIS_DIR = ROOT / "paper" / "analysis"

POLARITIES = ["negative", "neutral", "positive"]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def latest_local_prediction_files() -> list[Path]:
    latest: dict[str, Path] = {}
    for path in RUNS_DIR.glob("benchmark_full_*/*/*_sample_predictions.jsonl"):
        approach = path.name.replace("_sample_predictions.jsonl", "")
        current = latest.get(approach)
        if current is None or path.stat().st_mtime > current.stat().st_mtime:
            latest[approach] = path
    return [latest[k] for k in sorted(latest)]


def llm_response_files() -> list[Path]:
    files = []
    for path in RUNS_DIR.glob("openai_batch_eval_*/artifacts/llm_responses.jsonl"):
        files.append(path)
    files.sort(key=lambda p: p.stat().st_mtime)
    return files


def analyze_rows(rows: list[dict[str, Any]], source_label: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    per_aspect: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for row in rows:
        approach = str(row.get("approach", source_label))
        gold_map = row.get("gold_aspects", {}) or {}
        pred_map = row.get("predicted_aspects", {}) or {}

        for aspect, polarity in gold_map.items():
            counts[approach][f"{polarity}_gold"] += 1
            per_aspect[(approach, aspect)][f"{polarity}_gold"] += 1

            if aspect in pred_map:
                counts[approach][f"{polarity}_detected"] += 1
                per_aspect[(approach, aspect)][f"{polarity}_detected"] += 1
                if pred_map.get(aspect) == polarity:
                    counts[approach][f"{polarity}_sentiment_match"] += 1
                    per_aspect[(approach, aspect)][f"{polarity}_sentiment_match"] += 1
                else:
                    counts[approach][f"{polarity}_sentiment_mismatch"] += 1
                    per_aspect[(approach, aspect)][f"{polarity}_sentiment_mismatch"] += 1
            else:
                counts[approach][f"{polarity}_missed"] += 1
                per_aspect[(approach, aspect)][f"{polarity}_missed"] += 1

    summary_rows: list[dict[str, Any]] = []
    for approach, stat in sorted(counts.items()):
        row: dict[str, Any] = {"approach": approach}
        for polarity in POLARITIES:
            gold = stat.get(f"{polarity}_gold", 0)
            detected = stat.get(f"{polarity}_detected", 0)
            missed = stat.get(f"{polarity}_missed", 0)
            sentiment_match = stat.get(f"{polarity}_sentiment_match", 0)
            sentiment_mismatch = stat.get(f"{polarity}_sentiment_mismatch", 0)
            row[f"{polarity}_gold"] = gold
            row[f"{polarity}_detected"] = detected
            row[f"{polarity}_missed"] = missed
            row[f"{polarity}_recall"] = (detected / gold) if gold else float("nan")
            row[f"{polarity}_miss_rate"] = (missed / gold) if gold else float("nan")
            row[f"{polarity}_sentiment_match_rate_given_detected"] = (
                sentiment_match / detected if detected else float("nan")
            )
            row[f"{polarity}_sentiment_mismatch_rate_given_detected"] = (
                sentiment_mismatch / detected if detected else float("nan")
            )
        summary_rows.append(row)

    per_aspect_rows: list[dict[str, Any]] = []
    for (approach, aspect), stat in sorted(per_aspect.items()):
        for polarity in POLARITIES:
            gold = stat.get(f"{polarity}_gold", 0)
            detected = stat.get(f"{polarity}_detected", 0)
            missed = stat.get(f"{polarity}_missed", 0)
            sentiment_match = stat.get(f"{polarity}_sentiment_match", 0)
            sentiment_mismatch = stat.get(f"{polarity}_sentiment_mismatch", 0)
            per_aspect_rows.append(
                {
                    "approach": approach,
                    "aspect": aspect,
                    "polarity": polarity,
                    "gold": gold,
                    "detected": detected,
                    "missed": missed,
                    "recall": (detected / gold) if gold else float("nan"),
                    "miss_rate": (missed / gold) if gold else float("nan"),
                    "sentiment_match_rate_given_detected": (
                        sentiment_match / detected if detected else float("nan")
                    ),
                    "sentiment_mismatch_rate_given_detected": (
                        sentiment_mismatch / detected if detected else float("nan")
                    ),
                }
            )

    return summary_rows, per_aspect_rows


def write_report(summary_df: pd.DataFrame, path: Path) -> None:
    lines: list[str] = []
    lines.append("# Detection Error By Gold Polarity\n\n")
    lines.append("This report checks whether aspect detection depends on the gold sentiment polarity of the aspect mention. Detection recall here means: among gold aspect instances of a given polarity, how often was the aspect detected at all, regardless of whether the predicted sentiment was correct.\n\n")
    for _, row in summary_df.iterrows():
        lines.append(f"## {row['approach']}\n")
        for polarity in POLARITIES:
            lines.append(
                f"- `{polarity}`: gold {int(row[f'{polarity}_gold'])}, recall {row[f'{polarity}_recall']:.4f}, miss rate {row[f'{polarity}_miss_rate']:.4f}, sentiment match given detected {row[f'{polarity}_sentiment_match_rate_given_detected']:.4f}\n"
            )
        lines.append("\n")
    path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    all_summary_rows: list[dict[str, Any]] = []
    all_per_aspect_rows: list[dict[str, Any]] = []

    for path in latest_local_prediction_files():
        rows = load_jsonl(path)
        summary_rows, per_aspect_rows = analyze_rows(rows, path.stem)
        all_summary_rows.extend(summary_rows)
        all_per_aspect_rows.extend(per_aspect_rows)

    for path in llm_response_files():
        rows = load_jsonl(path)
        summary_rows, per_aspect_rows = analyze_rows(rows, path.stem)
        all_summary_rows.extend(summary_rows)
        all_per_aspect_rows.extend(per_aspect_rows)

    summary_df = pd.DataFrame(all_summary_rows).drop_duplicates(subset=["approach"]).sort_values("approach").reset_index(drop=True)
    per_aspect_df = pd.DataFrame(all_per_aspect_rows).sort_values(["approach", "aspect", "polarity"]).reset_index(drop=True)

    summary_path = ANALYSIS_DIR / "detection_by_polarity_summary_20260404.csv"
    per_aspect_path = ANALYSIS_DIR / "detection_by_polarity_per_aspect_20260404.csv"
    report_path = ANALYSIS_DIR / "detection_by_polarity_report_20260404.md"

    summary_df.to_csv(summary_path, index=False)
    per_aspect_df.to_csv(per_aspect_path, index=False)
    write_report(summary_df, report_path)

    print(
        json.dumps(
            {
                "summary_csv": str(summary_path),
                "per_aspect_csv": str(per_aspect_path),
                "report_md": str(report_path),
                "n_approaches": int(len(summary_df)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
