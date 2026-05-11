"""Score completed Task 2 (Herath re-annotation) rater files.

Inputs (reads from human/responses/task_2/):
  - rater_<L>_complete.csv          -> annotations per rater under our 20-aspect schema

Hidden file (reads from human/tasks/task_2_herath_reannotation/_herath_mapped_labels.json):
  - Herath team's mapped labels, used for the human-vs-mapping comparison

Outputs (writes into human/tasks/task_2_herath_reannotation/scoring/):
  - kappa_per_aspect.csv             -> pairwise Cohen kappa on each aspect's
                                        'discussed' decision
  - kappa_summary.json               -> aggregate kappa stats per pair and per aspect
  - mapping_vs_human.csv             -> per-aspect agreement between Herath's mapped
                                        labels and the human majority vote
  - gold_labels.csv                  -> majority-vote final labels for the 50 items

Run: /c/Python314/python human/scripts/score_task_2.py
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Dict, List

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ASPECTS, HUMAN_ROOT  # noqa: E402


TASK_DIR = HUMAN_ROOT / "tasks" / "task_2_herath_reannotation"
RESPONSES_DIR = HUMAN_ROOT / "responses" / "task_2"
SCORING_DIR = TASK_DIR / "scoring"


def cohen_kappa(a: List[str], b: List[str]) -> float:
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


def load_mapped() -> Dict[str, Dict[str, object]]:
    p = TASK_DIR / "_herath_mapped_labels.json"
    if not p.exists():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    return {item["item_id"]: item for item in data["items"]}


def load_rater_responses() -> Dict[str, Dict[str, Dict[str, str]]]:
    """Returns {rater_letter: {item_id: {col: value}}}."""
    out: Dict[str, Dict[str, Dict[str, str]]] = {}
    for path in sorted(RESPONSES_DIR.glob("rater_*_complete.csv")):
        letter = path.stem.split("_")[1]
        with path.open("r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            out[letter] = {row["item_id"]: row for row in reader}
    return out


def majority(values: List[str]) -> str:
    """Pick the most common label, breaking ties deterministically."""
    if not values:
        return ""
    c = Counter(values)
    top_count = max(c.values())
    winners = sorted(v for v, n in c.items() if n == top_count)
    return winners[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--responses_dir", type=Path, default=RESPONSES_DIR)
    args = parser.parse_args()

    SCORING_DIR.mkdir(parents=True, exist_ok=True)

    raters = load_rater_responses()
    if not raters:
        print(f"No response files found in {args.responses_dir}")
        print("Drop completed CSVs there (with _complete suffix) and rerun.")
        return

    letters = sorted(raters.keys())
    print(f"Found responses from raters: {letters}")

    common_items = set.intersection(*[set(raters[l].keys()) for l in letters])
    common_items = sorted(common_items)
    print(f"  common items: {len(common_items)}")

    # Per-aspect kappa for the 'discussed_<aspect>' columns.
    kappa_rows = []
    for asp in ASPECTS:
        col = f"discussed_{asp}"
        for la, lb in combinations(letters, 2):
            a = [(raters[la][i].get(col) or "no").strip().lower() for i in common_items]
            b = [(raters[lb][i].get(col) or "no").strip().lower() for i in common_items]
            kappa_rows.append({
                "aspect": asp,
                "rater_a": la,
                "rater_b": lb,
                "n_common": len(common_items),
                "kappa_discussed": round(cohen_kappa(a, b), 4),
            })
    with (SCORING_DIR / "kappa_per_aspect.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "aspect", "rater_a", "rater_b", "n_common", "kappa_discussed",
        ])
        writer.writeheader()
        writer.writerows(kappa_rows)
    print(f"  -> kappa_per_aspect.csv ({len(kappa_rows)} rows)")

    # Summary kappa.
    kappa_by_aspect: Dict[str, List[float]] = {asp: [] for asp in ASPECTS}
    for row in kappa_rows:
        k = row["kappa_discussed"]
        if k == k:  # not NaN
            kappa_by_aspect[row["aspect"]].append(float(k))
    summary = {
        "raters": letters,
        "n_common_items": len(common_items),
        "kappa_per_aspect_mean": {
            asp: round(sum(v) / len(v), 4) if v else None
            for asp, v in kappa_by_aspect.items()
        },
        "overall_kappa_mean": round(
            sum(k for vs in kappa_by_aspect.values() for k in vs) /
            max(sum(1 for vs in kappa_by_aspect.values() for _ in vs), 1),
            4,
        ),
    }
    (SCORING_DIR / "kappa_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"  -> kappa_summary.json (mean kappa = {summary['overall_kappa_mean']})")

    # Majority-vote gold labels.
    gold_rows = []
    for item_id in common_items:
        rec: Dict[str, str] = {"item_id": item_id}
        for asp in ASPECTS:
            disc_votes = [
                (raters[l][item_id].get(f"discussed_{asp}") or "no").strip().lower()
                for l in letters
            ]
            pol_votes = [
                (raters[l][item_id].get(f"polarity_{asp}") or "").strip().lower()
                for l in letters
            ]
            disc_winner = majority([v for v in disc_votes if v in ("yes", "no", "unclear")])
            rec[f"discussed_{asp}"] = disc_winner
            if disc_winner == "yes":
                pol_winner = majority([v for v in pol_votes if v in ("positive", "neutral", "negative")])
                rec[f"polarity_{asp}"] = pol_winner
            else:
                rec[f"polarity_{asp}"] = ""
        gold_rows.append(rec)
    gold_fields = ["item_id"]
    for asp in ASPECTS:
        gold_fields.append(f"discussed_{asp}")
        gold_fields.append(f"polarity_{asp}")
    with (SCORING_DIR / "gold_labels.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=gold_fields)
        writer.writeheader()
        writer.writerows(gold_rows)
    print(f"  -> gold_labels.csv ({len(gold_rows)} rows)")

    # Mapping-vs-human comparison.
    mapped = load_mapped()
    rows = []
    for asp in ASPECTS:
        n_mapped_yes = 0
        n_human_yes = 0
        n_both_yes = 0
        n_polarity_agree = 0
        n_polarity_compared = 0
        for item in gold_rows:
            iid = item["item_id"]
            if iid not in mapped:
                continue
            mapped_labels = mapped[iid].get("herath_mapped_labels") or {}
            mapped_has = asp in mapped_labels
            human_has = item.get(f"discussed_{asp}") == "yes"
            if mapped_has:
                n_mapped_yes += 1
            if human_has:
                n_human_yes += 1
            if mapped_has and human_has:
                n_both_yes += 1
                m_pol = str(mapped_labels[asp])
                h_pol = item.get(f"polarity_{asp}", "")
                n_polarity_compared += 1
                if m_pol == h_pol:
                    n_polarity_agree += 1
        rows.append({
            "aspect": asp,
            "n_mapped_yes": n_mapped_yes,
            "n_human_yes": n_human_yes,
            "n_both_yes": n_both_yes,
            "support_recall": round(n_both_yes / n_mapped_yes, 4) if n_mapped_yes else None,
            "support_precision": round(n_both_yes / n_human_yes, 4) if n_human_yes else None,
            "polarity_agreement_rate": (
                round(n_polarity_agree / n_polarity_compared, 4) if n_polarity_compared else None
            ),
        })
    with (SCORING_DIR / "mapping_vs_human.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "aspect", "n_mapped_yes", "n_human_yes", "n_both_yes",
            "support_recall", "support_precision", "polarity_agreement_rate",
        ])
        writer.writeheader()
        writer.writerows(rows)
    print(f"  -> mapping_vs_human.csv ({len(rows)} aspects)")


if __name__ == "__main__":
    main()
