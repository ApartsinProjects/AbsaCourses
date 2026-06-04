"""Phase E2: map EduRABSA_ASQE to the 7-aspect Herath-compatible JSONL.

Output shape: one JSON per line, schema matching the synthetic corpus and the
Herath mapped JSONL the existing transfer eval consumes:
  {"text": "...", "aspects": {"<aspect_name>": "<negative|neutral|positive>"}}

Per-row sentiment-aggregation: per aspect, if all EduRABSA mentions agree on
polarity, use that polarity; if mixed (one Positive + one Negative), drop the
aspect from that row.

Mapping from EduRABSA categories -> 7 of the 9 Herath aspects:

| EduRABSA category                  | Herath aspect       |
|------------------------------------|---------------------|
| Staff - Teaching                   | lecturer_quality    |
| Staff - Knowledge & skills         | lecturer_quality    |
| Course - Overall                   | overall_experience  |
| Staff - Overall                    | overall_experience  |
| Course - Learning activity         | organization        |
| Course - Course materials          | materials           |
| Course - Technology & tools        | materials           |
| Course - Assessment                | exam_fairness       |
| Course - Difficulty                | workload            |
| Course - Workload                  | workload            |
| Staff - Helpfulness                | accessibility       |
| University - Information & services| accessibility       |

The 2 Herath aspects with no direct EduRABSA category (assessment_design,
grading_transparency) are dropped from the E2 evaluation. The 9-aspect
overlap-restriction in evaluate_synthetic_to_real_transfer.py filters
the synthetic corpus to these 7 EduRABSA-mappable aspects automatically.
"""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path

import datasets

OUT_DIR = Path(r"E:\Claude\CourseABSA\hopeful-kowalevski-04ee10\external_data\EduRABSA_mapped")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CATEGORY_TO_ASPECT = {
    "Staff - Teaching": "lecturer_quality",
    "Staff - Knowledge & skills": "lecturer_quality",
    "Course - Overall": "overall_experience",
    "Staff - Overall": "overall_experience",
    "Course - Learning activity": "organization",
    "Course - Course materials": "materials",
    "Course - Technology & tools": "materials",
    "Course - Assessment": "exam_fairness",
    "Course - Difficulty": "workload",
    "Course - Workload": "workload",
    "Staff - Helpfulness": "accessibility",
    "University - Information & services": "accessibility",
}
SENTIMENT_NORMALIZE = {
    "Positive": "positive", "positive": "positive",
    "Negative": "negative", "negative": "negative",
    "Neutral": "neutral",   "neutral": "neutral",
}


def map_row(r: dict) -> dict | None:
    """Aggregate the EduRABSA row to one {text, aspects: {aspect: polarity}} record."""
    text = r["text"]
    by_aspect_polarities = defaultdict(list)
    for o in r.get("output", []):
        cat = o.get("category", "")
        asp = CATEGORY_TO_ASPECT.get(cat)
        if not asp: continue
        pol = SENTIMENT_NORMALIZE.get(o.get("sentiment", ""))
        if not pol: continue
        by_aspect_polarities[asp].append(pol)
    final_aspects = {}
    for asp, pols in by_aspect_polarities.items():
        unique = set(pols)
        if len(unique) == 1:
            final_aspects[asp] = next(iter(unique))
        else:
            # Mixed -> drop this aspect from this row
            continue
    if not final_aspects:
        return None  # no usable mappable aspect
    return {"text": text, "aspects": final_aspects}


def main():
    ds = datasets.load_from_disk(r"E:\Claude\CourseABSA\hopeful-kowalevski-04ee10\external_data\EduRABSA_ASQE")
    for split in ("train", "test"):
        mapped = []
        for r in ds[split]:
            m = map_row(r)
            if m is not None:
                mapped.append(m)
        out = OUT_DIR / f"edurabsa_{split}_mapped.jsonl"
        with out.open("w", encoding="utf-8") as f:
            for m in mapped:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")
        print(f"{split}: {len(ds[split])} src -> {len(mapped)} mapped rows ({len(mapped)/len(ds[split])*100:.1f}%)")
        # Aspect coverage
        from collections import Counter
        c = Counter()
        for m in mapped:
            for a, p in m["aspects"].items():
                c[(a, p)] += 1
        print(f"  per (aspect, polarity) count:")
        for k, n in sorted(c.items()):
            print(f"    {k[0]:25s} {k[1]:8s}  {n}")
    # Also produce a TRAIN+TEST combined file for the transfer eval to use as the
    # real-set, matching the way Herath consumes its whole annotated corpus.
    combined = OUT_DIR / "edurabsa_all_mapped.jsonl"
    with combined.open("w", encoding="utf-8") as f:
        for split in ("train", "test"):
            for m in [map_row(r) for r in ds[split]]:
                if m: f.write(json.dumps(m, ensure_ascii=False) + "\n")
    print(f"\nwrote combined: {combined}")


if __name__ == "__main__":
    main()
