"""Score completed Task 1 rater files.

Inputs (reads from human/responses/task_1/):
  - rater_<L>_part_1_complete.csv   -> realism judgments per rater
  - rater_<L>_part_2_complete.csv   -> faithfulness judgments per rater (synthetic only)

Truth (reads from human/tasks/task_1_realism_and_faithfulness/_truth.json):
  - per-item source (real / synthetic) and generator-assigned aspect labels

Outputs (writes into human/tasks/task_1_realism_and_faithfulness/scoring/):
  - realism_per_rater.csv            -> accuracy, confidence-weighted accuracy
  - realism_kappa.csv                -> pairwise Cohen kappa on judgment
  - faithfulness_per_aspect.csv      -> per-aspect 'discussed' rate and polarity-match rate
  - faithfulness_summary.json        -> aggregate faithfulness numbers
  - disagreements_part_1.csv         -> items where raters split

Run: /c/Python314/python human/scripts/score_task_1.py
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, List

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import HUMAN_ROOT  # noqa: E402


TASK_DIR = HUMAN_ROOT / "tasks" / "task_1_realism_and_faithfulness"
RESPONSES_DIR = HUMAN_ROOT / "responses" / "task_1"
SCORING_DIR = TASK_DIR / "scoring"


def cohen_kappa(a: List[str], b: List[str]) -> float:
    """Compute Cohen kappa without sklearn."""
    assert len(a) == len(b)
    n = len(a)
    if n == 0:
        return float("nan")
    labels = sorted(set(a) | set(b))
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    k = len(labels)
    cm = [[0] * k for _ in range(k)]
    for x, y in zip(a, b):
        cm[label_to_idx[x]][label_to_idx[y]] += 1
    po = sum(cm[i][i] for i in range(k)) / n
    row = [sum(cm[i]) / n for i in range(k)]
    col = [sum(cm[r][i] for r in range(k)) / n for i in range(k)]
    pe = sum(row[i] * col[i] for i in range(k))
    if pe == 1.0:
        return 1.0 if po == 1.0 else 0.0
    return (po - pe) / (1.0 - pe)


def load_truth() -> Dict[str, Dict[str, object]]:
    with (TASK_DIR / "_truth.json").open("r", encoding="utf-8") as fh:
        truth = json.load(fh)
    return {item["item_id"]: item for item in truth["items"]}


def load_part_1_responses() -> Dict[str, Dict[str, Dict[str, str]]]:
    """Returns {rater_letter: {item_id: {judgment, confidence, notes}}}."""
    out: Dict[str, Dict[str, Dict[str, str]]] = {}
    for path in sorted(RESPONSES_DIR.glob("rater_*_part_1_complete.csv")):
        letter = path.stem.split("_")[1]
        with path.open("r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            out[letter] = {
                row["item_id"]: {
                    "judgment": (row.get("judgment") or "").strip().lower(),
                    "confidence": (row.get("confidence") or "").strip(),
                    "notes": (row.get("notes") or "").strip(),
                }
                for row in reader
            }
    return out


def load_part_2_responses() -> Dict[str, List[Dict[str, str]]]:
    """Returns {rater_letter: [rows...]}. Each row keys (item_id, aspect, ...)."""
    out: Dict[str, List[Dict[str, str]]] = {}
    for path in sorted(RESPONSES_DIR.glob("rater_*_part_2_complete.csv")):
        letter = path.stem.split("_")[1]
        with path.open("r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            out[letter] = list(reader)
    return out


def score_realism(truth: Dict[str, Dict[str, object]],
                  part1: Dict[str, Dict[str, Dict[str, str]]]) -> None:
    rows = []
    for letter, items in part1.items():
        correct = 0
        weighted = 0.0
        weight_total = 0.0
        total = 0
        for item_id, ans in items.items():
            if item_id not in truth:
                continue
            true_source = truth[item_id]["source"]
            pred = ans["judgment"]
            try:
                conf = int(ans["confidence"])
            except ValueError:
                conf = 1
            total += 1
            if pred == true_source:
                correct += 1
                weighted += conf
            weight_total += conf
        if total == 0:
            continue
        rows.append({
            "rater": letter,
            "n": total,
            "accuracy": round(correct / total, 4),
            "confidence_weighted_accuracy": round(weighted / weight_total, 4) if weight_total else float("nan"),
        })
    out_path = SCORING_DIR / "realism_per_rater.csv"
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["rater", "n", "accuracy", "confidence_weighted_accuracy"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"  -> {out_path.name} ({len(rows)} rater(s))")

    # Pairwise kappa.
    kappa_rows = []
    letters = sorted(part1.keys())
    common_items = set.intersection(*[set(part1[l].keys()) for l in letters]) if letters else set()
    common_items = sorted(common_items)
    for la, lb in combinations(letters, 2):
        a = [part1[la][i]["judgment"] for i in common_items]
        b = [part1[lb][i]["judgment"] for i in common_items]
        kappa_rows.append({
            "rater_a": la,
            "rater_b": lb,
            "n_common": len(common_items),
            "kappa_judgment": round(cohen_kappa(a, b), 4),
        })
    if kappa_rows:
        with (SCORING_DIR / "realism_kappa.csv").open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["rater_a", "rater_b", "n_common", "kappa_judgment"])
            writer.writeheader()
            writer.writerows(kappa_rows)
        print(f"  -> realism_kappa.csv ({len(kappa_rows)} pair(s))")

    # Disagreements (items where any pair disagrees).
    disagreements = []
    for item_id in sorted(set.union(*[set(part1[l].keys()) for l in letters])) if letters else []:
        judgments = {l: part1[l].get(item_id, {}).get("judgment", "") for l in letters}
        if len(set(judgments.values())) > 1:
            true_source = truth.get(item_id, {}).get("source", "")
            disagreements.append({
                "item_id": item_id,
                "true_source": true_source,
                **{f"rater_{l}": judgments[l] for l in letters},
            })
    if disagreements:
        fieldnames = ["item_id", "true_source"] + [f"rater_{l}" for l in letters]
        with (SCORING_DIR / "disagreements_part_1.csv").open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(disagreements)
        print(f"  -> disagreements_part_1.csv ({len(disagreements)} items)")


def score_faithfulness(part2: Dict[str, List[Dict[str, str]]]) -> None:
    # Pool all rater rows. Each row carries (item_id, aspect, claimed_polarity,
    # aspect_discussed, polarity_correct).
    all_rows: List[Dict[str, str]] = []
    for letter, rows in part2.items():
        for r in rows:
            r2 = dict(r)
            r2["rater"] = letter
            all_rows.append(r2)
    if not all_rows:
        print("  (no Part 2 responses found)")
        return

    # Per-aspect discussed rate and polarity-match rate, pooled across raters.
    per_aspect: Dict[str, Counter] = defaultdict(Counter)
    for r in all_rows:
        asp = r["aspect"]
        disc = (r.get("aspect_discussed") or "").strip().lower()
        match = (r.get("polarity_correct") or "").strip().lower()
        per_aspect[asp]["n"] += 1
        per_aspect[asp][f"disc_{disc}"] += 1
        if disc == "yes":
            per_aspect[asp][f"polmatch_{match}"] += 1
    aspect_rows = []
    for asp, c in sorted(per_aspect.items()):
        n = c["n"]
        n_disc_yes = c["disc_yes"]
        aspect_rows.append({
            "aspect": asp,
            "n_judgments": n,
            "discussed_yes_rate": round(n_disc_yes / n, 4) if n else 0,
            "discussed_no_rate": round(c["disc_no"] / n, 4) if n else 0,
            "discussed_unclear_rate": round(c["disc_unclear"] / n, 4) if n else 0,
            "polarity_match_rate_when_discussed": (
                round(c["polmatch_yes"] / n_disc_yes, 4) if n_disc_yes else 0
            ),
        })
    with (SCORING_DIR / "faithfulness_per_aspect.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "aspect", "n_judgments",
            "discussed_yes_rate", "discussed_no_rate", "discussed_unclear_rate",
            "polarity_match_rate_when_discussed",
        ])
        writer.writeheader()
        writer.writerows(aspect_rows)
    print(f"  -> faithfulness_per_aspect.csv ({len(aspect_rows)} aspects)")

    # Aggregate summary.
    total = len(all_rows)
    n_disc_yes = sum(1 for r in all_rows if (r.get("aspect_discussed") or "").strip().lower() == "yes")
    n_match_yes = sum(
        1 for r in all_rows
        if (r.get("aspect_discussed") or "").strip().lower() == "yes"
        and (r.get("polarity_correct") or "").strip().lower() == "yes"
    )
    summary = {
        "n_judgments_total": total,
        "aspect_discussed_yes_rate": round(n_disc_yes / total, 4) if total else 0,
        "polarity_match_rate_among_discussed": (
            round(n_match_yes / n_disc_yes, 4) if n_disc_yes else 0
        ),
        "polarity_match_rate_overall": round(n_match_yes / total, 4) if total else 0,
    }
    with (SCORING_DIR / "faithfulness_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"  -> faithfulness_summary.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--responses_dir", type=Path, default=RESPONSES_DIR)
    args = parser.parse_args()

    SCORING_DIR.mkdir(parents=True, exist_ok=True)

    if not args.responses_dir.exists() or not any(args.responses_dir.glob("*.csv")):
        print(f"No response files found in {args.responses_dir}")
        print("Drop completed CSVs there (with _complete suffix) and rerun.")
        return

    truth = load_truth()
    part1 = load_part_1_responses()
    part2 = load_part_2_responses()

    print(f"Found Part 1 responses from raters: {sorted(part1.keys())}")
    print(f"Found Part 2 responses from raters: {sorted(part2.keys())}")
    print()
    print("Realism scoring:")
    score_realism(truth, part1)
    print()
    print("Faithfulness scoring:")
    score_faithfulness(part2)


if __name__ == "__main__":
    main()
