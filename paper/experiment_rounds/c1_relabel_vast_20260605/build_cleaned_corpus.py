"""Build a CLEANED-label copy of the 10k synthetic corpus from the faithfulness audit.

C1 (relabel-and-retrain) label-cleaning rule (NO API call; pure re-derivation from
the existing per-record audit):

For each corpus row (joined to the audit by 0-based row_id == corpus line index --
verified: declared aspect-set == corpus aspect-set for all 10000 rows, zero mismatch):
  - DETECTION target for a declared aspect = PRESENT iff audit says supported==True.
    Declared-but-unsupported aspects are DROPPED from detection.
  - SENTIMENT target for a declared aspect = declared polarity iff
    (supported==True AND sentiment_match==True); otherwise MASKED (excluded from the
    sentiment loss). Polarity is NEVER flipped -- the audit provides no corrected
    polarity, only a mismatch flag.

The engine couples detection presence and sentiment polarity in one {aspect: polarity}
dict (DetectionDataset uses the keys; SentimentDataset uses the {key: polarity} pairs
with mask=1 per key). To realize "keep an aspect for detection but mask its sentiment"
without modifying the engine, we emit TWO label views per row:
  - aspects_detection : {aspect: polarity} for every SUPPORTED aspect (polarity is a
                        placeholder, only the KEY matters for detection).
  - aspects_sentiment : {aspect: polarity} only for aspects that are SUPPORTED AND
                        sentiment_match (these get mask=1 in SentimentDataset; the
                        supported-but-mismatched aspects are simply absent -> masked).

The C1 worker loads the detection view for train_detection and the sentiment view for
train_sentiment. The BASELINE worker uses the ORIGINAL single-dict labels for both.

Outputs (all under the c1 round dir):
  - cleaned_corpus_10k.jsonl : per row {row_id, text, aspects_original,
                               aspects_detection, aspects_sentiment}
  - cleaning_stats.json      : corpus-wide drop / mask counts.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CORPUS = ROOT / "paper" / "reviewer_ab_data" / "generated_reviews_10k.jsonl"
AUDIT = ROOT / "paper" / "faithfulness_audit" / "at_scale_gpt-4.1-mini_responses.jsonl"

OUT_CORPUS = HERE / "cleaned_corpus_10k.jsonl"
OUT_STATS = HERE / "cleaning_stats.json"

VALID_POLARITY = {"positive", "neutral", "negative"}


def main() -> None:
    corpus = [json.loads(l) for l in CORPUS.open(encoding="utf-8")]
    n = len(corpus)
    assert n == 10000, f"expected 10000 corpus rows, got {n}"

    # audit indexed by 0-based row_id (== corpus line index)
    audit_by_row = {}
    for line in AUDIT.open(encoding="utf-8"):
        a = json.loads(line)
        rid = int(a["row_id"])
        audit_by_row[rid] = a
    assert len(audit_by_row) == n, f"audit rows {len(audit_by_row)} != corpus {n}"

    stats = {
        "n_rows": n,
        "n_rows_with_audit": 0,
        "n_declared_aspect_labels": 0,
        "n_supported": 0,
        "n_dropped_unsupported": 0,
        "n_matched_kept_for_sentiment": 0,
        "n_masked_polarity_mismatch": 0,  # supported but sentiment_match False
        "n_rows_emptied_detection": 0,    # rows that lose ALL aspects after cleaning
        "n_audit_aspect_not_in_declared": 0,  # audit lists an aspect not declared (should be 0)
        "n_declared_aspect_not_in_audit": 0,  # declared aspect missing from audit judged list
    }

    out_rows = []
    for rid in range(n):
        row = corpus[rid]
        text = row["text"]
        original = dict(row["aspects"])  # {aspect: polarity}
        stats["n_declared_aspect_labels"] += len(original)

        a = audit_by_row[rid]
        stats["n_rows_with_audit"] += 1
        declared_audit = a.get("declared", {})
        judged = a.get("judged", {}).get("aspects", [])
        # index audit verdicts by aspect
        verdict = {}
        for j in judged:
            asp = j.get("aspect")
            if asp is None:
                continue
            verdict[asp] = {
                "supported": bool(j.get("supported", False)),
                "sentiment_match": bool(j.get("sentiment_match", False)),
            }
            if asp not in original:
                stats["n_audit_aspect_not_in_declared"] += 1

        aspects_detection = {}
        aspects_sentiment = {}
        for asp, pol in original.items():
            pol = str(pol).strip().lower()
            v = verdict.get(asp)
            if v is None:
                # declared aspect the audit did not judge: conservative -> treat as
                # unsupported (drop). Counted separately for transparency.
                stats["n_declared_aspect_not_in_audit"] += 1
                continue
            if v["supported"]:
                stats["n_supported"] += 1
                aspects_detection[asp] = pol  # placeholder polarity; key is what matters
                if v["sentiment_match"] and pol in VALID_POLARITY:
                    aspects_sentiment[asp] = pol
                    stats["n_matched_kept_for_sentiment"] += 1
                else:
                    # supported but polarity mismatch -> MASK for sentiment (omit)
                    stats["n_masked_polarity_mismatch"] += 1
            else:
                stats["n_dropped_unsupported"] += 1

        if not aspects_detection:
            stats["n_rows_emptied_detection"] += 1

        out_rows.append({
            "row_id": rid,
            "text": text,
            "aspects_original": original,
            "aspects_detection": aspects_detection,
            "aspects_sentiment": aspects_sentiment,
        })

    with OUT_CORPUS.open("w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # derived rates
    d = stats["n_declared_aspect_labels"]
    stats["drop_rate_unsupported"] = round(stats["n_dropped_unsupported"] / d, 4) if d else None
    stats["mask_rate_polarity_mismatch"] = round(stats["n_masked_polarity_mismatch"] / d, 4) if d else None
    stats["support_rate"] = round(stats["n_supported"] / d, 4) if d else None
    stats["sentiment_keep_rate"] = round(stats["n_matched_kept_for_sentiment"] / d, 4) if d else None

    OUT_STATS.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"\nWROTE {OUT_CORPUS}")
    print(f"WROTE {OUT_STATS}")


if __name__ == "__main__":
    main()
