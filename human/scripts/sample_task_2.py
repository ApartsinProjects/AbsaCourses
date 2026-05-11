"""Sample Task 2 (Herath re-annotation) rater files.

For each rater (A, B, C by default) produces:
  - rater_<L>.csv            -> one row per Herath review with 20 (discussed, polarity) column pairs

Also writes:
  - manifest.csv             -> public per-item manifest with item_id + review_text
  - _herath_mapped_labels.json  -> the Herath team's mapped labels, kept hidden during
                                   annotation but used for the post-hoc agreement table

Sampling tries to cover a broad mix of:
  - length bands (short/mid/long)
  - the aspects present in the Herath mapping (so rare aspects like accessibility,
    workload, grading_transparency are represented)
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Dict, List

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    ASPECTS,
    HERATH_PATH,
    HUMAN_ROOT,
    load_jsonl,
    rater_letters,
    word_count,
    write_json,
)


TASK_DIR = HUMAN_ROOT / "tasks" / "task_2_herath_reannotation"


def length_band(n: int) -> str:
    if n < 15:
        return "short"
    if n < 40:
        return "mid"
    return "long"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n_items", type=int, default=50)
    parser.add_argument("--min_words", type=int, default=8,
                        help="Drop Herath reviews below this word count.")
    parser.add_argument("--n_raters", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    TASK_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    print(f"Loading Herath corpus: {HERATH_PATH.name}")
    real = load_jsonl(HERATH_PATH)
    real = [r for r in real if word_count(str(r.get("text", ""))) >= args.min_words]
    print(f"  after length filter: {len(real)}")

    # Group by which aspect appears in the mapping so we cover sparse aspects.
    aspect_to_rows: Dict[str, List[Dict[str, object]]] = {}
    for r in real:
        for asp in (r.get("aspects") or {}):
            aspect_to_rows.setdefault(str(asp), []).append(r)

    # Sort aspects by population, ascending; sparse aspects get covered first so they
    # are not crowded out by the rich ones.
    aspects_by_size = sorted(aspect_to_rows.items(), key=lambda kv: len(kv[1]))
    target_per_aspect = max(args.n_items // max(len(aspects_by_size), 1), 1)

    picked_ids: set = set()
    picked: List[Dict[str, object]] = []
    for asp, rows in aspects_by_size:
        rng.shuffle(rows)
        count = 0
        for r in rows:
            tid = id(r)
            if tid in picked_ids:
                continue
            picked_ids.add(tid)
            picked.append(r)
            count += 1
            if count >= target_per_aspect:
                break
        if len(picked) >= args.n_items:
            break

    # Top up with random others if short.
    if len(picked) < args.n_items:
        leftover = [r for r in real if id(r) not in picked_ids]
        rng.shuffle(leftover)
        picked.extend(leftover[: args.n_items - len(picked)])
    picked = picked[: args.n_items]

    # Build item ids and manifest.
    items: List[Dict[str, object]] = []
    for i, r in enumerate(picked, start=1):
        text = str(r.get("text", "")).strip()
        items.append({
            "item_id": f"H{i:03d}",
            "review_text": text,
            "herath_mapped_labels": r.get("aspects") or {},
            "herath_doc_sent": r.get("doc_sent", ""),
            "length_band": length_band(word_count(text)),
        })

    # Manifest (rater-visible).
    with (TASK_DIR / "manifest.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["item_id", "review_text", "length_band"])
        for it in items:
            writer.writerow([it["item_id"], it["review_text"], it["length_band"]])

    # Hidden mapping (for post-hoc agreement only).
    hidden = {
        "n_items": len(items),
        "items": [
            {
                "item_id": it["item_id"],
                "herath_mapped_labels": it["herath_mapped_labels"],
                "herath_doc_sent": it["herath_doc_sent"],
            }
            for it in items
        ],
    }
    write_json(TASK_DIR / "_herath_mapped_labels.json", hidden)

    # Rater files.
    letters = rater_letters(args.n_raters)
    header = ["item_id", "review_text"]
    for asp in ASPECTS:
        header.append(f"discussed_{asp}")
        header.append(f"polarity_{asp}")
    header.append("notes")

    for letter in letters:
        rater_rng = random.Random(args.seed + ord(letter))
        order = list(range(len(items)))
        rater_rng.shuffle(order)
        path = TASK_DIR / f"rater_{letter}.csv"
        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(header)
            for idx in order:
                it = items[idx]
                row = [it["item_id"], it["review_text"]]
                for _ in ASPECTS:
                    row.append("")  # discussed
                    row.append("")  # polarity
                row.append("")  # notes
                writer.writerow(row)

    aspect_coverage = {
        asp: sum(1 for it in items if asp in (it["herath_mapped_labels"] or {}))
        for asp in ASPECTS
    }

    print()
    print("Done.")
    print(f"  task dir:    {TASK_DIR}")
    print(f"  items:       {len(items)}")
    print(f"  raters:      {args.n_raters} ({', '.join(letters)})")
    print(f"  aspect coverage in sample (mapped Herath labels):")
    for asp, c in sorted(aspect_coverage.items(), key=lambda kv: -kv[1]):
        if c > 0:
            print(f"    {asp:24s} {c}")


if __name__ == "__main__":
    main()
