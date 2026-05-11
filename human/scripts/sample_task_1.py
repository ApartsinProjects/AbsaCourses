"""Sample Task 1 (realism + faithfulness) rater files.

For each rater (A, B, C by default) produces:
  - rater_<L>_part_1.csv    -> realism judgment on all items (synthetic + real)
  - rater_<L>_part_2.csv    -> faithfulness check on synthetic items only

Also writes:
  - manifest.csv             -> public per-item manifest (no source label)
  - _truth.json              -> hidden ground truth (source + synthetic labels)

The sample is stratified by length so synthetic and real items occupy roughly the
same word-count range, removing the trivial length-as-cue confound.
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
    HERATH_PATH,
    HUMAN_ROOT,
    SYNTHETIC_PATH,
    load_jsonl,
    rater_letters,
    word_count,
    write_json,
)


TASK_DIR = HUMAN_ROOT / "tasks" / "task_1_realism_and_faithfulness"


def length_band(n: int) -> str:
    if n < 30:
        return "short"
    if n < 80:
        return "mid"
    if n < 180:
        return "long"
    return "very_long"


def filter_by_min_len(rows: List[Dict[str, object]], min_words: int) -> List[Dict[str, object]]:
    return [r for r in rows if word_count(str(r.get("text", ""))) >= min_words]


def stratified_sample(
    rows: List[Dict[str, object]],
    n: int,
    rng: random.Random,
    bands: List[str] | None = None,
) -> List[Dict[str, object]]:
    """Pick n rows trying to spread across length bands. bands defaults to all four."""
    bands = bands or ["short", "mid", "long", "very_long"]
    buckets: Dict[str, List[Dict[str, object]]] = {b: [] for b in bands}
    for r in rows:
        b = length_band(word_count(str(r.get("text", ""))))
        if b in buckets:
            buckets[b].append(r)
    for b in buckets:
        rng.shuffle(buckets[b])

    per_band = max(n // max(len([b for b in bands if buckets[b]]), 1), 1)
    picked: List[Dict[str, object]] = []
    for b in bands:
        picked.extend(buckets[b][:per_band])

    # Top up uniformly if short.
    if len(picked) < n:
        remaining = [r for b in bands for r in buckets[b][per_band:]]
        rng.shuffle(remaining)
        picked.extend(remaining[: n - len(picked)])
    # Trim if long.
    rng.shuffle(picked)
    return picked[:n]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n_synthetic", type=int, default=40)
    parser.add_argument("--n_real", type=int, default=40)
    parser.add_argument("--min_real_words", type=int, default=15,
                        help="Drop real reviews below this word count.")
    parser.add_argument("--min_synth_words", type=int, default=15,
                        help="Drop synthetic reviews below this word count.")
    parser.add_argument("--n_raters", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    TASK_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    print(f"Loading synthetic corpus: {SYNTHETIC_PATH.name}")
    syn = load_jsonl(SYNTHETIC_PATH)
    syn = filter_by_min_len(syn, args.min_synth_words)
    print(f"  synthetic after length filter: {len(syn)}")

    print(f"Loading real (Herath) corpus: {HERATH_PATH.name}")
    real = load_jsonl(HERATH_PATH)
    real = filter_by_min_len(real, args.min_real_words)
    print(f"  real after length filter: {len(real)}")

    syn_sample = stratified_sample(syn, args.n_synthetic, rng)
    real_sample = stratified_sample(real, args.n_real, rng)

    # Build the combined pool.
    pool: List[Dict[str, object]] = []
    for i, r in enumerate(real_sample, start=1):
        pool.append({
            "item_id": f"R{i:03d}",
            "source": "real",
            "review_text": str(r.get("text", "")).strip(),
            "aspects": {},
        })
    for i, s in enumerate(syn_sample, start=1):
        aspects = s.get("aspects") or {}
        clean_aspects: Dict[str, str] = {
            str(a): str(p) for a, p in aspects.items() if str(p) in ("positive", "neutral", "negative")
        }
        pool.append({
            "item_id": f"S{i:03d}",
            "source": "synthetic",
            "review_text": str(s.get("text", "")).strip(),
            "aspects": clean_aspects,
        })

    # Public manifest (item_id, review_text only).
    manifest_path = TASK_DIR / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["item_id", "review_text"])
        for p in pool:
            writer.writerow([p["item_id"], p["review_text"]])

    # Hidden ground truth.
    truth = {
        "n_real": sum(1 for p in pool if p["source"] == "real"),
        "n_synthetic": sum(1 for p in pool if p["source"] == "synthetic"),
        "items": [
            {"item_id": p["item_id"], "source": p["source"], "aspects": p["aspects"]}
            for p in pool
        ],
    }
    write_json(TASK_DIR / "_truth.json", truth)

    # Rater files (Part 1: blinded realism on all items).
    letters = rater_letters(args.n_raters)
    for letter in letters:
        order = list(range(len(pool)))
        rater_rng = random.Random(args.seed + ord(letter))
        rater_rng.shuffle(order)
        path = TASK_DIR / f"rater_{letter}_part_1.csv"
        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["item_id", "review_text", "judgment", "confidence", "notes"])
            for idx in order:
                p = pool[idx]
                writer.writerow([p["item_id"], p["review_text"], "", "", ""])

    # Rater files (Part 2: faithfulness on synthetic items only).
    syn_items = [p for p in pool if p["source"] == "synthetic"]
    n_part_2_rows_per_rater = sum(len(p["aspects"]) for p in syn_items)
    for letter in letters:
        rater_rng = random.Random(args.seed + ord(letter) + 1000)
        rows: List[List[object]] = []
        for p in syn_items:
            for aspect in sorted(p["aspects"]):
                rows.append([
                    p["item_id"],
                    p["review_text"],
                    aspect,
                    p["aspects"][aspect],
                    "",  # aspect_discussed
                    "",  # polarity_correct
                    "",  # notes
                ])
        rater_rng.shuffle(rows)
        path = TASK_DIR / f"rater_{letter}_part_2.csv"
        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow([
                "item_id", "review_text", "aspect", "claimed_polarity",
                "aspect_discussed", "polarity_correct", "notes",
            ])
            writer.writerows(rows)

    print()
    print("Done.")
    print(f"  task dir:       {TASK_DIR}")
    print(f"  real items:     {truth['n_real']}")
    print(f"  synth items:    {truth['n_synthetic']}")
    print(f"  raters:         {args.n_raters} ({', '.join(letters)})")
    print(f"  part-2 rows/r:  {n_part_2_rows_per_rater}")


if __name__ == "__main__":
    main()
