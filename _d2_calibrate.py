"""Phase D2 calibration: re-audit the existing 250 rows with gpt-4.1-mini,
score per row, compute Spearman vs the GPT-5.2 audit, gate at rho >= 0.6."""
from __future__ import annotations

import concurrent.futures as cf
import csv
import json
import os
import time
from pathlib import Path
from openai import OpenAI

# -------- config --------
OPENAI_API_KEY = Path(r"E:\Projects\CourseABSA\.opeai.key").read_text(encoding="utf-8").strip()
INPUT_JSONL = Path(r"E:\Claude\CourseABSA\hopeful-kowalevski-04ee10\_d2_cal_input.jsonl")
GPT5_DETAILS = Path(r"E:\Projects\CourseABSA\paper\faithfulness_audit\faithfulness_audit_gpt-5_2_250_details.csv")
OUT_DIR = Path(r"E:\Claude\CourseABSA\hopeful-kowalevski-04ee10\paper\faithfulness_audit")
RAW_JSONL = OUT_DIR / "calibration_gpt-4.1-mini_responses.jsonl"
SCORES_CSV = OUT_DIR / "calibration_gpt-4.1-mini_per_row_scores.csv"
RESULT_JSON = OUT_DIR / "calibration_gpt-4.1-mini_vs_gpt-5_2.json"
MODEL = "gpt-4.1-mini"
MAX_WORKERS = 12
MIN_SPEARMAN = 0.6

AUDIT_PROMPT = """You are auditing whether aspect-sentiment labels are faithful to a student course review.

Review:
{text}

Declared labels:
{labels}

Return JSON only with this schema:
{{
  "aspects": [
    {{
      "aspect": "aspect_name",
      "supported": true,
      "sentiment_match": true,
      "note": "short explanation"
    }}
  ],
  "overall_note": "one short sentence"
}}

Rules:
- "supported" means the review text clearly expresses that aspect.
- "sentiment_match" means the declared sentiment matches the review text for that aspect.
- If an aspect is not supported, set sentiment_match to false.
- Be strict and conservative.
"""

SCHEMA = {
    "type": "json_schema", "name": "faithfulness_audit",
    "schema": {
        "type": "object", "additionalProperties": False,
        "properties": {
            "aspects": {"type": "array", "items": {
                "type": "object", "additionalProperties": False,
                "properties": {
                    "aspect": {"type": "string"}, "supported": {"type": "boolean"},
                    "sentiment_match": {"type": "boolean"}, "note": {"type": "string"},
                },
                "required": ["aspect", "supported", "sentiment_match", "note"],
            }},
            "overall_note": {"type": "string"},
        },
        "required": ["aspects", "overall_note"],
    },
    "strict": True,
}


def call_audit(client: OpenAI, row: dict) -> dict:
    labels = "\n".join(f"- {a}: {s}" for a, s in row["aspects"].items())
    prompt = AUDIT_PROMPT.format(text=str(row["text"])[:2500], labels=labels)
    for attempt in range(3):
        try:
            resp = client.responses.create(
                model=MODEL, input=prompt, max_output_tokens=500,
                text={"format": SCHEMA},
            )
            return {"row_id": row["row_id"], "ok": True,
                    "declared": row["aspects"],
                    "judged": json.loads(resp.output_text)}
        except Exception as e:
            last = e
            time.sleep(2 ** attempt)
    return {"row_id": row["row_id"], "ok": False, "error": str(last)[:200]}


def score_row(judged_payload: dict, declared: dict) -> dict:
    """Per-row faithfulness score: fraction of declared aspects that are both
    supported AND sentiment-match. Same scoring as the GPT-5.2 audit summary."""
    by_aspect = {a["aspect"]: a for a in judged_payload.get("aspects", [])}
    n = len(declared)
    n_supported = 0
    n_matched = 0
    for asp in declared:
        j = by_aspect.get(asp)
        if not j:
            continue
        if j.get("supported"):
            n_supported += 1
            if j.get("sentiment_match"):
                n_matched += 1
    return {"n_aspects": n, "n_supported": n_supported, "n_matched": n_matched,
            "support_rate": n_supported / n if n else 0.0,
            "match_rate": n_matched / n if n else 0.0,
            "row_score": n_matched / n if n else 0.0}


def gpt5_per_row(details_csv: Path) -> dict:
    """Recompute GPT-5.2 per-row scores from the existing details CSV."""
    by_row = {}
    with details_csv.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rid = r["row_id"]
            by_row.setdefault(rid, {"n": 0, "n_supported": 0, "n_matched": 0})
            by_row[rid]["n"] += 1
            sup = str(r.get("supported", "")).lower() in ("true", "1", "yes")
            mat = str(r.get("sentiment_match", "")).lower() in ("true", "1", "yes")
            if sup:
                by_row[rid]["n_supported"] += 1
                if mat:
                    by_row[rid]["n_matched"] += 1
    return {rid: (v["n_matched"] / v["n"] if v["n"] else 0.0) for rid, v in by_row.items()}


def spearman(a: list, b: list) -> float:
    """Spearman rho without scipy (rank correlation)."""
    assert len(a) == len(b)
    n = len(a)
    if n < 3:
        return float("nan")
    def ranks(xs):
        # average rank for ties
        order = sorted(range(n), key=lambda i: xs[i])
        out = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and xs[order[j + 1]] == xs[order[i]]:
                j += 1
            avg = (i + j + 2) / 2  # 1-indexed average
            for k in range(i, j + 1):
                out[order[k]] = avg
            i = j + 1
        return out
    ra, rb = ranks(a), ranks(b)
    mean_a = sum(ra) / n
    mean_b = sum(rb) / n
    cov = sum((ra[i] - mean_a) * (rb[i] - mean_b) for i in range(n))
    va = sum((ra[i] - mean_a) ** 2 for i in range(n))
    vb = sum((rb[i] - mean_b) ** 2 for i in range(n))
    if va == 0 or vb == 0:
        return float("nan")
    return cov / (va * vb) ** 0.5


def pearson(a: list, b: list) -> float:
    n = len(a)
    if n < 3:
        return float("nan")
    ma = sum(a) / n
    mb = sum(b) / n
    cov = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    va = sum((a[i] - ma) ** 2 for i in range(n))
    vb = sum((b[i] - mb) ** 2 for i in range(n))
    if va == 0 or vb == 0:
        return float("nan")
    return cov / (va * vb) ** 0.5


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in INPUT_JSONL.open(encoding="utf-8")]
    print(f"calibration over {len(rows)} rows with {MODEL}, {MAX_WORKERS} workers")

    client = OpenAI(api_key=OPENAI_API_KEY)
    t0 = time.time()
    results = []
    with cf.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(call_audit, client, r): r for r in rows}
        for i, f in enumerate(cf.as_completed(futures), 1):
            res = f.result()
            results.append(res)
            if i % 25 == 0:
                elapsed = time.time() - t0
                ok = sum(1 for r in results if r["ok"])
                print(f"  {i}/{len(rows)}  ok={ok}  elapsed={elapsed:.0f}s")

    # Persist raw responses
    with RAW_JSONL.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nwrote raw: {RAW_JSONL}")

    # Per-row scores
    scores_by_row = {}
    for res in results:
        if not res["ok"]:
            continue
        s = score_row(res["judged"], res["declared"])
        scores_by_row[res["row_id"]] = s

    with SCORES_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row_id", "n_aspects", "n_supported", "n_matched", "support_rate", "match_rate", "row_score"])
        for rid in sorted(scores_by_row.keys(), key=int):
            s = scores_by_row[rid]
            w.writerow([rid, s["n_aspects"], s["n_supported"], s["n_matched"],
                        round(s["support_rate"], 4), round(s["match_rate"], 4), round(s["row_score"], 4)])
    print(f"wrote scores: {SCORES_CSV}")

    # GPT-5.2 per-row scores
    gpt5 = gpt5_per_row(GPT5_DETAILS)
    common = sorted(set(scores_by_row.keys()) & set(gpt5.keys()), key=int)
    print(f"\nrows scored by both: {len(common)}")

    mini_scores = [scores_by_row[r]["row_score"] for r in common]
    gpt5_scores = [gpt5[r] for r in common]
    rho = spearman(mini_scores, gpt5_scores)
    rs = pearson(mini_scores, gpt5_scores)

    # Aggregate rates
    n_supported_mini = sum(scores_by_row[r]["n_supported"] for r in common)
    n_matched_mini = sum(scores_by_row[r]["n_matched"] for r in common)
    n_aspects = sum(scores_by_row[r]["n_aspects"] for r in common)

    summary = {
        "candidate_model": MODEL,
        "n_rows": len(common),
        "spearman_rho": round(rho, 4),
        "pearson_r": round(rs, 4),
        "candidate_aggregate_support_rate": round(n_supported_mini / n_aspects, 4) if n_aspects else None,
        "candidate_aggregate_match_rate": round(n_matched_mini / n_aspects, 4) if n_aspects else None,
        "candidate_avg_row_score": round(sum(mini_scores) / len(mini_scores), 4) if mini_scores else None,
        "gpt5_avg_row_score": round(sum(gpt5_scores) / len(gpt5_scores), 4) if gpt5_scores else None,
        "acceptance_threshold_spearman": MIN_SPEARMAN,
        "accepted": rho >= MIN_SPEARMAN,
        "recommendation": (
            f"USE {MODEL} for at-scale audit (rho={rho:.3f} >= {MIN_SPEARMAN})"
            if rho >= MIN_SPEARMAN
            else f"ESCALATE: rho={rho:.3f} < {MIN_SPEARMAN}; try gpt-4.1-mini or gpt-5.2"
        ),
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    RESULT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote result: {RESULT_JSON}")
    print()
    print(json.dumps(summary, indent=2))
    return 0 if rho >= MIN_SPEARMAN else 1


if __name__ == "__main__":
    raise SystemExit(main())
