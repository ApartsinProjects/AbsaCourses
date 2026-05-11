from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from absa_model_comparison import extract_json_block, parse_aspect_map


ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "paper" / "benchmark_outputs" / "runs"
ANALYSIS_DIR = ROOT / "paper" / "analysis"

ALLOWED_ASPECTS = {
    "accessibility",
    "assessment_design",
    "clarity",
    "difficulty",
    "exam_fairness",
    "feedback_quality",
    "grading_transparency",
    "interest",
    "lecturer_quality",
    "materials",
    "organization",
    "overall_experience",
    "pacing",
    "peer_interaction",
    "practical_application",
    "prerequisite_fit",
    "relevance",
    "support",
    "tooling_usability",
    "workload",
}

BROAD_LABELS = {
    "overall_experience",
    "interest",
    "support",
    "tooling_usability",
    "feedback_quality",
}


def discover_llm_runs() -> list[Path]:
    runs = []
    for run_dir in RUNS_DIR.glob("openai_batch_eval_*"):
        artifact = run_dir / "artifacts" / "llm_responses.jsonl"
        metadata = run_dir / "metadata.json"
        if artifact.exists() or metadata.exists():
            runs.append(run_dir)
    runs.sort(key=lambda p: p.stat().st_mtime)
    return runs


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def reconstruct_rows_from_metadata(run_dir: Path) -> list[dict[str, Any]]:
    metadata_path = run_dir / "metadata.json"
    if not metadata_path.exists():
        return []
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    manifest_path = Path(metadata.get("manifest", ""))
    results_path = Path(metadata.get("results_path", ""))
    if not manifest_path.exists() or not results_path.exists():
        return []

    manifest = pd.read_csv(manifest_path)
    raw_by_custom: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(results_path):
        raw_by_custom[str(row.get("custom_id", ""))] = row

    aspects = sorted(ALLOWED_ASPECTS)
    rows: list[dict[str, Any]] = []
    for _, manifest_row in manifest.iterrows():
        custom_id = str(manifest_row["custom_id"])
        raw_row = raw_by_custom.get(custom_id, {})
        output_text = ""
        body = raw_row.get("response", {}).get("body", {})
        for item in body.get("output", []):
            if item.get("type") == "message":
                for content in item.get("content", []):
                    if content.get("type") == "output_text":
                        output_text = str(content.get("text", "")).strip()
                        break
        parsed = {}
        if output_text:
            try:
                parsed = extract_json_block(output_text)
            except Exception:
                parsed = {}
        pred_map = parse_aspect_map(parsed, aspects)
        gold_map = json.loads(str(manifest_row["gold_aspects"]))
        rows.append(
            {
                "approach": str(manifest_row.get("approach", f"openai-{manifest_row.get('model', '')}-{manifest_row.get('variant', '')}")),
                "model": str(manifest_row.get("model", "")),
                "variant": str(manifest_row.get("variant", "")),
                "custom_id": custom_id,
                "text": str(manifest_row.get("text", "")),
                "gold_aspects": gold_map,
                "raw_response_text": output_text,
                "parsed_response": parsed,
                "predicted_aspects": pred_map,
                "response_status": raw_row.get("response", {}).get("status", ""),
            }
        )
    return rows


def summarize_approach(rows: list[dict[str, Any]], run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    response_statuses = Counter()
    invalid_names = Counter()
    pair_counts = Counter()
    parse_fail_examples: list[dict[str, Any]] = []
    per_aspect = {aspect: {"gold_support": 0, "pred_support": 0, "tp": 0, "fp": 0, "fn": 0, "correct_sentiment": 0, "wrong_sentiment": 0} for aspect in sorted(ALLOWED_ASPECTS)}
    broad_replacements = Counter()

    row_count = 0
    raw_nonempty = 0
    parse_fail_count = 0
    exact_match_rows = 0
    miss_only_rows = 0
    extra_only_rows = 0
    both_rows = 0
    fn_total = 0
    fp_total = 0
    gold_total = 0
    pred_total = 0
    correct_sentiment_tp_total = 0
    wrong_sentiment_tp_total = 0

    for row in rows:
        row_count += 1
        raw_text = row.get("raw_response_text", "")
        if raw_text:
            raw_nonempty += 1
        response_statuses[str(row.get("response_status", ""))] += 1

        parsed = row.get("parsed_response")
        if not parsed:
            parse_fail_count += 1
            if len(parse_fail_examples) < 25:
                parse_fail_examples.append(
                    {
                        "approach": row.get("approach", ""),
                        "run_dir": str(run_dir),
                        "custom_id": row.get("custom_id", ""),
                        "response_status": row.get("response_status", ""),
                        "raw_response_text": raw_text,
                    }
                )

        gold_map = row.get("gold_aspects", {}) or {}
        pred_map = row.get("predicted_aspects", {}) or {}
        gold = set(gold_map.keys())
        pred = set(pred_map.keys())

        for aspect in pred:
            if aspect not in ALLOWED_ASPECTS:
                invalid_names[aspect] += 1

        missed = gold - pred
        extra = pred - gold
        fn_total += len(missed)
        fp_total += len(extra)
        gold_total += len(gold)
        pred_total += len(pred)

        if not missed and not extra:
            exact_match_rows += 1
        elif missed and not extra:
            miss_only_rows += 1
        elif extra and not missed:
            extra_only_rows += 1
        else:
            both_rows += 1

        for gold_aspect in missed:
            for wrong_aspect in extra:
                pair_counts[(gold_aspect, wrong_aspect)] += 1
                if wrong_aspect in BROAD_LABELS:
                    broad_replacements[(gold_aspect, wrong_aspect)] += 1

        for aspect in ALLOWED_ASPECTS:
            gold_present = aspect in gold
            pred_present = aspect in pred
            if gold_present:
                per_aspect[aspect]["gold_support"] += 1
            if pred_present:
                per_aspect[aspect]["pred_support"] += 1
            if gold_present and pred_present:
                per_aspect[aspect]["tp"] += 1
                if pred_map.get(aspect) == gold_map.get(aspect):
                    per_aspect[aspect]["correct_sentiment"] += 1
                    correct_sentiment_tp_total += 1
                else:
                    per_aspect[aspect]["wrong_sentiment"] += 1
                    wrong_sentiment_tp_total += 1
            elif gold_present and not pred_present:
                per_aspect[aspect]["fn"] += 1
            elif pred_present and not gold_present:
                per_aspect[aspect]["fp"] += 1

    approach_name = str(rows[0].get("approach", "unknown"))
    model_name = str(rows[0].get("model", ""))
    variant = str(rows[0].get("variant", ""))

    summary = {
        "approach": approach_name,
        "model": model_name,
        "variant": variant,
        "run_dir": str(run_dir),
        "n_rows": row_count,
        "parse_fail_count": parse_fail_count,
        "parse_success_rate": (row_count - parse_fail_count) / row_count if row_count else float("nan"),
        "raw_nonempty_rate": raw_nonempty / row_count if row_count else float("nan"),
        "exact_match_rows": exact_match_rows,
        "miss_only_rows": miss_only_rows,
        "extra_only_rows": extra_only_rows,
        "both_miss_and_extra_rows": both_rows,
        "rows_with_any_miss": miss_only_rows + both_rows,
        "rows_with_any_extra": extra_only_rows + both_rows,
        "gold_total_aspects": gold_total,
        "pred_total_aspects": pred_total,
        "false_negative_total": fn_total,
        "false_positive_total": fp_total,
        "mean_gold_aspects_per_row": gold_total / row_count if row_count else float("nan"),
        "mean_pred_aspects_per_row": pred_total / row_count if row_count else float("nan"),
        "pred_minus_gold_aspects_per_row": (pred_total - gold_total) / row_count if row_count else float("nan"),
        "correct_sentiment_given_tp_total": correct_sentiment_tp_total,
        "wrong_sentiment_given_tp_total": wrong_sentiment_tp_total,
        "sentiment_match_rate_given_tp": correct_sentiment_tp_total / (correct_sentiment_tp_total + wrong_sentiment_tp_total)
        if (correct_sentiment_tp_total + wrong_sentiment_tp_total)
        else float("nan"),
        "invalid_predicted_name_count": int(sum(invalid_names.values())),
        "response_statuses": json.dumps(dict(response_statuses), ensure_ascii=False),
    }

    per_aspect_rows = []
    for aspect, stats in sorted(per_aspect.items()):
        gold_support = stats["gold_support"]
        pred_support = stats["pred_support"]
        tp = stats["tp"]
        fp = stats["fp"]
        fn = stats["fn"]
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        per_aspect_rows.append(
            {
                "approach": approach_name,
                "aspect": aspect,
                "gold_support": gold_support,
                "pred_support": pred_support,
                "pred_minus_gold": pred_support - gold_support,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "sentiment_match_rate_given_tp": stats["correct_sentiment"] / tp if tp else float("nan"),
                "wrong_sentiment_count_given_tp": stats["wrong_sentiment"],
                "correct_sentiment_count_given_tp": stats["correct_sentiment"],
            }
        )

    confusion_rows = [
        {
            "approach": approach_name,
            "gold_missed": gold_aspect,
            "wrong_predicted": wrong_aspect,
            "count": count,
            "wrong_is_broad_label": wrong_aspect in BROAD_LABELS,
        }
        for (gold_aspect, wrong_aspect), count in pair_counts.most_common()
    ]

    broad_replacement_rows = [
        {
            "approach": approach_name,
            "gold_missed": gold_aspect,
            "broad_wrong_predicted": wrong_aspect,
            "count": count,
        }
        for (gold_aspect, wrong_aspect), count in broad_replacements.most_common()
    ]

    return summary, per_aspect_rows, confusion_rows, broad_replacement_rows, parse_fail_examples


def write_markdown_report(summary_df: pd.DataFrame, confusion_df: pd.DataFrame, broad_df: pd.DataFrame, path: Path) -> None:
    lines: list[str] = []
    lines.append("# GPT Diagnostics Report\n\n")
    lines.append("This report summarizes detection-heavy diagnostics for the saved GPT ABSA batch runs. The main purpose is to separate missed-aspect behavior from extra-aspect behavior, quantify parse failures, and identify recurring confusion patterns.\n\n")
    lines.append("## Headline summary\n\n")
    for _, row in summary_df.iterrows():
        lines.append(
            f"- `{row['approach']}`: micro behavior summary = exact {int(row['exact_match_rows'])} rows, miss-only {int(row['miss_only_rows'])}, extra-only {int(row['extra_only_rows'])}, both {int(row['both_miss_and_extra_rows'])}; parse success {row['parse_success_rate']:.3f}; mean gold aspects {row['mean_gold_aspects_per_row']:.3f}; mean predicted aspects {row['mean_pred_aspects_per_row']:.3f}.\n"
        )
    lines.append("\n## Top confusion pairs\n\n")
    for approach, subset in confusion_df.groupby("approach", sort=False):
        lines.append(f"### {approach}\n")
        for _, row in subset.head(10).iterrows():
            lines.append(f"- {row['gold_missed']} -> {row['wrong_predicted']}: {int(row['count'])}\n")
        lines.append("\n")
    lines.append("## Broad-label replacement patterns\n\n")
    for approach, subset in broad_df.groupby("approach", sort=False):
        lines.append(f"### {approach}\n")
        if subset.empty:
            lines.append("- No broad-label replacement pairs recorded.\n\n")
            continue
        for _, row in subset.head(10).iterrows():
            lines.append(f"- {row['gold_missed']} -> {row['broad_wrong_predicted']}: {int(row['count'])}\n")
        lines.append("\n")
    path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []
    per_aspect_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    broad_rows: list[dict[str, Any]] = []
    parse_fail_rows: list[dict[str, Any]] = []

    for run_dir in discover_llm_runs():
        llm_path = run_dir / "artifacts" / "llm_responses.jsonl"
        rows = load_jsonl(llm_path) if llm_path.exists() else reconstruct_rows_from_metadata(run_dir)
        by_approach: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            by_approach.setdefault(str(row.get("approach", "unknown")), []).append(row)
        for approach_rows in by_approach.values():
            summary, aspect_rows, conf_rows, broad_repl_rows, parse_fails = summarize_approach(approach_rows, run_dir)
            summary_rows.append(summary)
            per_aspect_rows.extend(aspect_rows)
            confusion_rows.extend(conf_rows)
            broad_rows.extend(broad_repl_rows)
            parse_fail_rows.extend(parse_fails)

    summary_df = pd.DataFrame(summary_rows).sort_values("approach").reset_index(drop=True)
    per_aspect_df = pd.DataFrame(per_aspect_rows).sort_values(["approach", "aspect"]).reset_index(drop=True)
    confusion_df = pd.DataFrame(confusion_rows).sort_values(["approach", "count"], ascending=[True, False]).reset_index(drop=True)
    broad_df = pd.DataFrame(broad_rows).sort_values(["approach", "count"], ascending=[True, False]).reset_index(drop=True)
    parse_fail_df = pd.DataFrame(parse_fail_rows)

    summary_path = ANALYSIS_DIR / "gpt_diagnostics_summary_20260404.csv"
    per_aspect_path = ANALYSIS_DIR / "gpt_diagnostics_per_aspect_20260404.csv"
    confusion_path = ANALYSIS_DIR / "gpt_diagnostics_confusion_pairs_20260404.csv"
    broad_path = ANALYSIS_DIR / "gpt_diagnostics_broad_replacements_20260404.csv"
    parse_fail_path = ANALYSIS_DIR / "gpt_diagnostics_parse_fail_examples_20260404.csv"
    report_path = ANALYSIS_DIR / "gpt_diagnostics_report_20260404.md"

    summary_df.to_csv(summary_path, index=False)
    per_aspect_df.to_csv(per_aspect_path, index=False)
    confusion_df.to_csv(confusion_path, index=False)
    broad_df.to_csv(broad_path, index=False)
    parse_fail_df.to_csv(parse_fail_path, index=False)
    write_markdown_report(summary_df, confusion_df, broad_df, report_path)

    print(
        json.dumps(
            {
                "summary_csv": str(summary_path),
                "per_aspect_csv": str(per_aspect_path),
                "confusion_csv": str(confusion_path),
                "broad_replacements_csv": str(broad_path),
                "parse_fail_examples_csv": str(parse_fail_path),
                "report_md": str(report_path),
                "n_approaches": int(len(summary_df)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
