from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from collections import Counter
from statistics import median
from typing import Dict, Iterable, Optional


ROOT = Path(__file__).resolve().parents[1]
BATCH_DIR = ROOT / "paper" / "batch_requests"
RESULTS_DIR = ROOT / "paper" / "batch_results"
DATASET_DIR = ROOT / "paper" / "generated_datasets"


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)


def default_manifest_path() -> Path:
    candidates = [
        BATCH_DIR / "dataset_generation_10k_v2_manifest.csv",
        BATCH_DIR / "dataset_generation_10k_manifest.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def load_manifest(path: Path) -> Dict[str, Dict[str, str]]:
    rows: Dict[str, Dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows[row["custom_id"]] = row
    return rows


def extract_output_text(body: object) -> Optional[str]:
    if isinstance(body, dict):
        if isinstance(body.get("output_text"), str) and body["output_text"].strip():
            return body["output_text"].strip()
        output = body.get("output")
        if isinstance(output, list):
            parts = []
            for item in output:
                if not isinstance(item, dict):
                    continue
                content = item.get("content")
                if isinstance(content, list):
                    for chunk in content:
                        if not isinstance(chunk, dict):
                            continue
                        text = chunk.get("text")
                        if isinstance(text, str):
                            parts.append(text)
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
            joined = "".join(parts).strip()
            if joined:
                return joined
        choices = body.get("choices")
        if isinstance(choices, list) and choices:
            choice = choices[0]
            if isinstance(choice, dict):
                message = choice.get("message", {})
                if isinstance(message, dict):
                    content = message.get("content")
                    if isinstance(content, str):
                        return content.strip()
    return None


def iter_result_rows(path: Path) -> Iterable[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def merge_results(
    manifest: Dict[str, Dict[str, str]],
    results_path: Path,
    batch_id: str,
    *,
    include_incomplete: bool = False,
) -> list[Dict[str, object]]:
    rows: list[Dict[str, object]] = []
    for item in iter_result_rows(results_path):
        custom_id = item.get("custom_id")
        if custom_id not in manifest:
            continue
        response = item.get("response", {})
        body = response.get("body", {}) if isinstance(response, dict) else {}
        status = str(body.get("status", "unknown"))
        incomplete_details = body.get("incomplete_details", {}) if isinstance(body.get("incomplete_details", {}), dict) else {}
        incomplete_reason = str(incomplete_details.get("reason", "") or "")
        if not include_incomplete and status != "completed":
            continue
        text = extract_output_text(body)
        if not text:
            continue
        meta = manifest[custom_id]
        target_attributes = json.loads(meta["target_attributes"])
        nuance_attributes = json.loads(meta["nuance_attributes"])
        rows.append(
            {
                "text": text,
                "aspects": target_attributes,
                "target_attributes": target_attributes,
                "nuance_attributes": nuance_attributes,
                "source_path": str(results_path),
                "batch_id": batch_id,
                "sample_id": meta["sample_id"],
                "course_name": nuance_attributes.get("course_name", ""),
                "style": nuance_attributes.get("writing_style", nuance_attributes.get("linguistic_style", "")),
                "grade": nuance_attributes.get("grade_band", ""),
                "response_status": status,
                "incomplete_reason": incomplete_reason,
            }
        )
    return rows


def summarize_raw_results(manifest: Dict[str, Dict[str, str]], results_path: Path) -> Dict[str, object]:
    status_counts: Counter[str] = Counter()
    incomplete_reason_counts: Counter[str] = Counter()
    text_lengths: list[int] = []
    word_lengths: list[int] = []
    seen_texts: set[str] = set()
    duplicate_texts = 0
    total_rows = 0
    rows_with_manifest = 0
    rows_with_text = 0

    for item in iter_result_rows(results_path):
        total_rows += 1
        custom_id = item.get("custom_id")
        if custom_id in manifest:
            rows_with_manifest += 1
        response = item.get("response", {})
        body = response.get("body", {}) if isinstance(response, dict) else {}
        status = str(body.get("status", "unknown"))
        status_counts[status] += 1
        incomplete = body.get("incomplete_details", {})
        if isinstance(incomplete, dict):
            reason = incomplete.get("reason")
            if reason:
                incomplete_reason_counts[str(reason)] += 1
        text = extract_output_text(body)
        if not text:
            continue
        rows_with_text += 1
        cleaned = " ".join(text.split())
        text_lengths.append(len(cleaned))
        word_lengths.append(len(cleaned.split()))
        if cleaned in seen_texts:
            duplicate_texts += 1
        else:
            seen_texts.add(cleaned)

    summary: Dict[str, object] = {
        "manifest_rows": len(manifest),
        "result_rows": total_rows,
        "rows_with_matching_manifest": rows_with_manifest,
        "rows_with_text": rows_with_text,
        "text_success_rate": round(rows_with_text / max(total_rows, 1), 4),
        "response_status_counts": dict(status_counts),
        "incomplete_reason_counts": dict(incomplete_reason_counts),
        "duplicate_text_count": duplicate_texts,
        "duplicate_text_rate": round(duplicate_texts / max(rows_with_text, 1), 4),
    }
    if word_lengths:
        summary.update(
            {
                "word_count_mean": round(sum(word_lengths) / len(word_lengths), 2),
                "word_count_median": round(float(median(word_lengths)), 2),
                "word_count_min": min(word_lengths),
                "word_count_max": max(word_lengths),
                "char_count_mean": round(sum(text_lengths) / len(text_lengths), 2),
            }
        )
    return summary


def write_outputs(rows: list[Dict[str, object]], stem: str) -> Dict[str, str]:
    jsonl_path = DATASET_DIR / f"{stem}.jsonl"
    csv_path = DATASET_DIR / f"{stem}.csv"

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    fieldnames = [
        "text",
        "aspects",
        "target_attributes",
        "nuance_attributes",
        "source_path",
        "batch_id",
        "sample_id",
        "course_name",
        "style",
        "grade",
        "response_status",
        "incomplete_reason",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            serializable = dict(row)
            serializable["aspects"] = json.dumps(serializable["aspects"], ensure_ascii=False)
            serializable["target_attributes"] = json.dumps(serializable["target_attributes"], ensure_ascii=False)
            serializable["nuance_attributes"] = json.dumps(serializable["nuance_attributes"], ensure_ascii=False)
            writer.writerow(serializable)
    return {"jsonl": str(jsonl_path), "csv": str(csv_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge completed generation batch outputs with the manifest.")
    parser.add_argument("--manifest", default=str(default_manifest_path()))
    parser.add_argument("--results-path", required=True)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--output-stem", default=None)
    parser.add_argument("--include-incomplete", action="store_true", help="Allow rows whose response status is not completed.")
    args = parser.parse_args()

    ensure_dirs()
    manifest = load_manifest(Path(args.manifest))
    stem = args.output_stem or f"{args.batch_id}_generated_reviews"
    raw_summary = summarize_raw_results(manifest, Path(args.results_path))
    rows = merge_results(manifest, Path(args.results_path), args.batch_id, include_incomplete=args.include_incomplete)
    outputs = write_outputs(rows, stem)
    summary = {
        "batch_id": args.batch_id,
        "manifest_path": str(Path(args.manifest).resolve()),
        "n_rows": len(rows),
        "include_incomplete": bool(args.include_incomplete),
        "outputs": outputs,
        "raw_result_summary": raw_summary,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
