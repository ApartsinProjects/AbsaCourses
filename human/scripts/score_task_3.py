"""Score completed Task 3 (human vs. LLM-judge) rater files.

Inputs (reads from human/responses/task_3/):
  - rater_<L>_complete.csv          -> per (review, aspect) row decisions

Hidden file (reads from human/tasks/task_3_llm_judge_agreement/_gpt_judgments.json):
  - GPT-5.2 decisions from the original audit

Outputs (writes into human/tasks/task_3_llm_judge_agreement/scoring/):
  - human_vs_llm_summary.json        -> agreement rates and kappa, both decisions
  - human_vs_llm_per_bucket.csv      -> agreement by GPT outcome bucket (A/B/C/D)
  - human_vs_llm_per_aspect.csv      -> agreement by aspect
  - disagreements.csv                -> items where humans and GPT differ

Run: /c/Python314/python human/scripts/score_task_3.py
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


TASK_DIR = HUMAN_ROOT / "tasks" / "task_3_llm_judge_agreement"
RESPONSES_DIR = HUMAN_ROOT / "responses" / "task_3"
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


def majority(values: List[str]) -> str:
    if not values:
        return ""
    c = Counter(values)
    top_count = max(c.values())
    winners = sorted(v for v, n in c.items() if n == top_count)
    return winners[0]


def load_gpt() -> Dict[str, Dict[str, object]]:
    p = TASK_DIR / "_gpt_judgments.json"
    if not p.exists():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    return {item["item_id"]: item for item in data["items"]}


def load_raters() -> Dict[str, Dict[str, Dict[str, str]]]:
    out: Dict[str, Dict[str, Dict[str, str]]] = {}
    for path in sorted(RESPONSES_DIR.glob("rater_*_complete.csv")):
        letter = path.stem.split("_")[1]
        with path.open("r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            out[letter] = {row["item_id"]: row for row in reader}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--responses_dir", type=Path, default=RESPONSES_DIR)
    args = parser.parse_args()

    SCORING_DIR.mkdir(parents=True, exist_ok=True)

    gpt = load_gpt()
    raters = load_raters()
    if not raters:
        print(f"No response files found in {args.responses_dir}")
        print("Drop completed CSVs there (with _complete suffix) and rerun.")
        return

    letters = sorted(raters.keys())
    common_items = set.intersection(*[set(raters[l].keys()) for l in letters])
    common_items = sorted(common_items & set(gpt.keys()))
    print(f"Common items between all raters and GPT audit: {len(common_items)}")

    # Build per-item majority vote of humans.
    human_supported: Dict[str, str] = {}
    human_match: Dict[str, str] = {}
    for iid in common_items:
        sup_votes = [
            (raters[l][iid].get("aspect_supported") or "").strip().lower()
            for l in letters
        ]
        match_votes = [
            (raters[l][iid].get("sentiment_match") or "").strip().lower()
            for l in letters
        ]
        human_supported[iid] = majority([v for v in sup_votes if v in ("yes", "no", "unclear")])
        human_match[iid] = majority([v for v in match_votes if v in ("yes", "no", "unclear")])

    # Pairwise human kappa (inter-rater).
    inter_rater_rows = []
    for la, lb in combinations(letters, 2):
        a_sup = [(raters[la][i].get("aspect_supported") or "").strip().lower() for i in common_items]
        b_sup = [(raters[lb][i].get("aspect_supported") or "").strip().lower() for i in common_items]
        a_mat = [(raters[la][i].get("sentiment_match") or "").strip().lower() for i in common_items]
        b_mat = [(raters[lb][i].get("sentiment_match") or "").strip().lower() for i in common_items]
        inter_rater_rows.append({
            "rater_a": la,
            "rater_b": lb,
            "n_common": len(common_items),
            "kappa_supported": round(cohen_kappa(a_sup, b_sup), 4),
            "kappa_sentiment_match": round(cohen_kappa(a_mat, b_mat), 4),
        })
    with (SCORING_DIR / "inter_rater_kappa.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "rater_a", "rater_b", "n_common", "kappa_supported", "kappa_sentiment_match",
        ])
        writer.writeheader()
        writer.writerows(inter_rater_rows)
    print(f"  -> inter_rater_kappa.csv ({len(inter_rater_rows)} pair(s))")

    # Human-vs-GPT agreement.
    sup_h = [human_supported[i] for i in common_items]
    sup_g = [str(gpt[i]["gpt_supported"]) for i in common_items]
    match_h = [human_match[i] for i in common_items]
    match_g = [str(gpt[i]["gpt_sentiment_match"]) for i in common_items]
    n = len(common_items)
    agree_sup = sum(1 for h, g in zip(sup_h, sup_g) if h == g)
    agree_match = sum(1 for h, g in zip(match_h, match_g) if h == g)

    summary = {
        "n_items": n,
        "n_raters": len(letters),
        "agreement_rate_supported": round(agree_sup / n, 4) if n else 0,
        "agreement_rate_sentiment_match": round(agree_match / n, 4) if n else 0,
        "kappa_supported_human_vs_gpt": round(cohen_kappa(sup_h, sup_g), 4),
        "kappa_sentiment_match_human_vs_gpt": round(cohen_kappa(match_h, match_g), 4),
        "human_supported_distribution": dict(Counter(sup_h)),
        "gpt_supported_distribution": dict(Counter(sup_g)),
        "human_sentiment_match_distribution": dict(Counter(match_h)),
        "gpt_sentiment_match_distribution": dict(Counter(match_g)),
    }
    (SCORING_DIR / "human_vs_llm_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"  -> human_vs_llm_summary.json")
    print(f"     kappa_supported = {summary['kappa_supported_human_vs_gpt']}")
    print(f"     kappa_sentiment_match = {summary['kappa_sentiment_match_human_vs_gpt']}")

    # Per-bucket agreement.
    by_bucket: Dict[str, Dict[str, int]] = defaultdict(lambda: {
        "n": 0, "agree_supported": 0, "agree_sentiment_match": 0,
    })
    for iid, h_s, g_s, h_m, g_m in zip(common_items, sup_h, sup_g, match_h, match_g):
        bucket = str(gpt[iid].get("bucket", "?"))
        by_bucket[bucket]["n"] += 1
        if h_s == g_s:
            by_bucket[bucket]["agree_supported"] += 1
        if h_m == g_m:
            by_bucket[bucket]["agree_sentiment_match"] += 1
    bucket_rows = []
    for b in sorted(by_bucket):
        n_b = by_bucket[b]["n"]
        bucket_rows.append({
            "bucket": b,
            "n_items": n_b,
            "agreement_supported": round(by_bucket[b]["agree_supported"] / n_b, 4) if n_b else 0,
            "agreement_sentiment_match": round(by_bucket[b]["agree_sentiment_match"] / n_b, 4) if n_b else 0,
        })
    with (SCORING_DIR / "human_vs_llm_per_bucket.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "bucket", "n_items", "agreement_supported", "agreement_sentiment_match",
        ])
        writer.writeheader()
        writer.writerows(bucket_rows)
    print(f"  -> human_vs_llm_per_bucket.csv ({len(bucket_rows)} bucket(s))")

    # Per-aspect agreement.
    by_aspect: Dict[str, Dict[str, int]] = defaultdict(lambda: {
        "n": 0, "agree_supported": 0, "agree_sentiment_match": 0,
    })
    for iid, h_s, g_s, h_m, g_m in zip(common_items, sup_h, sup_g, match_h, match_g):
        asp = str(gpt[iid].get("aspect", "?"))
        by_aspect[asp]["n"] += 1
        if h_s == g_s:
            by_aspect[asp]["agree_supported"] += 1
        if h_m == g_m:
            by_aspect[asp]["agree_sentiment_match"] += 1
    aspect_rows = []
    for asp in sorted(by_aspect):
        n_a = by_aspect[asp]["n"]
        aspect_rows.append({
            "aspect": asp,
            "n_items": n_a,
            "agreement_supported": round(by_aspect[asp]["agree_supported"] / n_a, 4) if n_a else 0,
            "agreement_sentiment_match": round(by_aspect[asp]["agree_sentiment_match"] / n_a, 4) if n_a else 0,
        })
    with (SCORING_DIR / "human_vs_llm_per_aspect.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "aspect", "n_items", "agreement_supported", "agreement_sentiment_match",
        ])
        writer.writeheader()
        writer.writerows(aspect_rows)
    print(f"  -> human_vs_llm_per_aspect.csv ({len(aspect_rows)} aspect(s))")

    # Disagreements (rows where human majority differs from GPT on either decision).
    disagreements = []
    for iid, h_s, g_s, h_m, g_m in zip(common_items, sup_h, sup_g, match_h, match_g):
        if h_s != g_s or h_m != g_m:
            disagreements.append({
                "item_id": iid,
                "audit_row_id": gpt[iid]["audit_row_id"],
                "aspect": gpt[iid]["aspect"],
                "declared_polarity": gpt[iid]["declared_polarity"],
                "human_supported": h_s,
                "gpt_supported": g_s,
                "human_sentiment_match": h_m,
                "gpt_sentiment_match": g_m,
                "gpt_note": gpt[iid].get("gpt_note", ""),
            })
    with (SCORING_DIR / "disagreements.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "item_id", "audit_row_id", "aspect", "declared_polarity",
            "human_supported", "gpt_supported",
            "human_sentiment_match", "gpt_sentiment_match",
            "gpt_note",
        ])
        writer.writeheader()
        writer.writerows(disagreements)
    print(f"  -> disagreements.csv ({len(disagreements)} items)")


if __name__ == "__main__":
    main()
