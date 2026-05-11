"""Sample Task 3 (human vs. LLM-judge agreement) rater files.

For each rater (A, B, C by default) produces:
  - rater_<L>.csv            -> one row per (review, aspect) pair sampled from the
                                250-item GPT-5.2 audit

Also writes:
  - manifest.csv             -> public per-item manifest (no LLM judgments visible)
  - _gpt_judgments.json      -> hidden GPT-5.2 decisions used for kappa later

Sampling is stratified across the four GPT-5.2 outcome buckets so that human raters
see a balanced mix of easy and hard cases:

  bucket A: supported=yes  + sentiment_match=yes
  bucket B: supported=yes  + sentiment_match=no
  bucket C: supported=no
  bucket D: supported=unclear OR sentiment_match=unclear
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Dict, List, Tuple

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    AUDIT_DETAILS_PATH,
    HUMAN_ROOT,
    rater_letters,
    write_json,
)


TASK_DIR = HUMAN_ROOT / "tasks" / "task_3_llm_judge_agreement"


def normalize_value(v: object) -> str:
    if v is None:
        return ""
    s = str(v).strip().lower()
    if s in ("yes", "true", "1", "supported", "match"):
        return "yes"
    if s in ("no", "false", "0", "not_supported", "mismatch"):
        return "no"
    if s in ("", "unclear", "ambiguous", "uncertain", "partial"):
        return "unclear"
    return s


def bucket_of(supported: str, sentiment_match: str) -> str:
    if supported == "unclear" or sentiment_match == "unclear":
        return "D"
    if supported == "yes" and sentiment_match == "yes":
        return "A"
    if supported == "yes" and sentiment_match == "no":
        return "B"
    if supported == "no":
        return "C"
    return "D"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n_items", type=int, default=80,
                        help="Total (review, aspect) pairs per rater.")
    parser.add_argument("--n_raters", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--audit_csv", type=Path, default=AUDIT_DETAILS_PATH)
    args = parser.parse_args()

    TASK_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    print(f"Loading LLM-judge audit: {args.audit_csv.name}")
    with args.audit_csv.open("r", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    print(f"  audit declared-aspect rows: {len(rows)}")

    # Normalize the GPT decisions and assign buckets.
    enriched: List[Dict[str, object]] = []
    for r in rows:
        supported = normalize_value(r.get("supported"))
        sentiment_match = normalize_value(r.get("sentiment_match"))
        enriched.append({
            "audit_row_id": r.get("row_id", ""),
            "review_text": (r.get("text") or "").strip(),
            "aspect": r.get("aspect", ""),
            "declared_polarity": r.get("declared_sentiment", ""),
            "gpt_supported": supported,
            "gpt_sentiment_match": sentiment_match,
            "gpt_note": r.get("note", ""),
            "bucket": bucket_of(supported, sentiment_match),
        })

    # Filter to rows that have a non-empty review and a non-empty aspect.
    enriched = [r for r in enriched if r["review_text"] and r["aspect"]]

    by_bucket: Dict[str, List[Dict[str, object]]] = {"A": [], "B": [], "C": [], "D": []}
    for r in enriched:
        by_bucket[r["bucket"]].append(r)
    for b in by_bucket:
        rng.shuffle(by_bucket[b])
        print(f"  bucket {b}: {len(by_bucket[b])} candidates")

    # Allocate quotas: try to split evenly across the four buckets.
    quota_per_bucket = max(args.n_items // 4, 1)
    picked: List[Dict[str, object]] = []
    for b in ("A", "B", "C", "D"):
        picked.extend(by_bucket[b][:quota_per_bucket])
    if len(picked) < args.n_items:
        remaining: List[Dict[str, object]] = []
        for b in ("A", "B", "C", "D"):
            remaining.extend(by_bucket[b][quota_per_bucket:])
        rng.shuffle(remaining)
        picked.extend(remaining[: args.n_items - len(picked)])
    picked = picked[: args.n_items]

    # Assign stable item ids in stratified order.
    items: List[Tuple[str, Dict[str, object]]] = [
        (f"J{i:03d}", r) for i, r in enumerate(picked, start=1)
    ]

    # Public manifest (no LLM judgments).
    with (TASK_DIR / "manifest.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["item_id", "audit_row_id", "review_text", "aspect", "declared_polarity"])
        for item_id, r in items:
            writer.writerow([
                item_id, r["audit_row_id"], r["review_text"], r["aspect"], r["declared_polarity"],
            ])

    # Hidden GPT judgments.
    hidden = {
        "n_items": len(items),
        "items": [
            {
                "item_id": item_id,
                "audit_row_id": r["audit_row_id"],
                "aspect": r["aspect"],
                "declared_polarity": r["declared_polarity"],
                "gpt_supported": r["gpt_supported"],
                "gpt_sentiment_match": r["gpt_sentiment_match"],
                "gpt_note": r["gpt_note"],
                "bucket": r["bucket"],
            }
            for item_id, r in items
        ],
    }
    write_json(TASK_DIR / "_gpt_judgments.json", hidden)

    # Rater files (shuffled order per rater).
    letters = rater_letters(args.n_raters)
    header = [
        "item_id", "audit_row_id", "review_text", "aspect", "declared_polarity",
        "aspect_supported", "sentiment_match", "notes",
    ]
    for letter in letters:
        rater_rng = random.Random(args.seed + ord(letter))
        order = list(range(len(items)))
        rater_rng.shuffle(order)
        path = TASK_DIR / f"rater_{letter}.csv"
        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(header)
            for idx in order:
                item_id, r = items[idx]
                writer.writerow([
                    item_id, r["audit_row_id"], r["review_text"], r["aspect"],
                    r["declared_polarity"], "", "", "",
                ])

    print()
    print("Done.")
    print(f"  task dir:  {TASK_DIR}")
    print(f"  items:     {len(items)}")
    print(f"  raters:    {args.n_raters} ({', '.join(letters)})")
    print(f"  bucket distribution in sample:")
    for b in ("A", "B", "C", "D"):
        c = sum(1 for _, r in items if r["bucket"] == b)
        print(f"    bucket {b}: {c}")


if __name__ == "__main__":
    main()
