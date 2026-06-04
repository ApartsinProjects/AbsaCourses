"""Phase E4: cross-provider audit-judge calibration.

Re-runs the 250-row calibration with Claude Haiku via OpenRouter (OpenAI-
compatible API), scoring per row exactly as in _d2_calibrate.py, then
computes per-aspect support/match agreement vs the existing gpt-5.2 details.

Acceptance threshold: per-aspect support agreement >= 0.75 AND
per-aspect sentiment-match agreement >= 0.60. (The thresholds the
gpt-4.1-mini Phase D2 calibration hit were 0.845 and 0.715; this is a
weaker bar so a second provider passes credibly.)
"""
from __future__ import annotations
import concurrent.futures as cf
import csv, json, time
from pathlib import Path
from openai import OpenAI

# Read OpenRouter key from .env.all
ENV_PATH = Path(r"E:\Projects\.env.all")
OPENROUTER_KEY = None
for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
    if line.startswith("OPENROUTER_API_KEY="):
        OPENROUTER_KEY = line.split("=", 1)[1].strip()
        break
assert OPENROUTER_KEY, "OPENROUTER_API_KEY not found"

INPUT_JSONL = Path(r"E:\Claude\CourseABSA\hopeful-kowalevski-04ee10\_d2_cal_input.jsonl")
GPT5_DETAILS = Path(r"E:\Projects\CourseABSA\paper\faithfulness_audit\faithfulness_audit_gpt-5_2_250_details.csv")
OUT_DIR = Path(r"E:\Claude\CourseABSA\hopeful-kowalevski-04ee10\paper\faithfulness_audit")
MODEL = "anthropic/claude-3.5-haiku"   # OpenRouter slug
TAG = "claude-3_5-haiku"
RAW_JSONL = OUT_DIR / f"calibration_{TAG}_responses.jsonl"
SCORES_CSV = OUT_DIR / f"calibration_{TAG}_per_row_scores.csv"
RESULT_JSON = OUT_DIR / f"calibration_{TAG}_vs_gpt-5_2.json"
MAX_WORKERS = 12

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


def call_audit(client, row):
    labels = "\n".join(f"- {a}: {s}" for a, s in row["aspects"].items())
    prompt = AUDIT_PROMPT.format(text=str(row["text"])[:2500], labels=labels)
    for attempt in range(3):
        try:
            # OpenRouter doesn't support json_schema across all providers, so use
            # JSON mode + post-validation.
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=600,
                temperature=0.0,
            )
            raw = resp.choices[0].message.content or ""
            # Strip ```json fences if present
            s = raw.strip()
            if s.startswith("```"):
                s = s.split("```", 2)[1]
                if s.startswith("json"):
                    s = s[4:]
                s = s.strip("` \n")
            payload = json.loads(s)
            return {"row_id": row["row_id"], "ok": True,
                    "declared": row["aspects"], "judged": payload}
        except Exception as e:
            last = e
            time.sleep(2 ** attempt)
    return {"row_id": row["row_id"], "ok": False, "error": str(last)[:200]}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in INPUT_JSONL.open(encoding="utf-8")]
    print(f"calibration over {len(rows)} rows with {MODEL} (OpenRouter)")
    print(f"raw -> {RAW_JSONL}")

    client = OpenAI(api_key=OPENROUTER_KEY, base_url="https://openrouter.ai/api/v1")
    t0 = time.time()
    results = []
    with cf.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(call_audit, client, r): r for r in rows}
        for i, f in enumerate(cf.as_completed(futures), 1):
            res = f.result()
            results.append(res)
            if i % 25 == 0:
                el = time.time() - t0
                ok = sum(1 for r in results if r["ok"])
                print(f"  {i}/{len(rows)} ok={ok} elapsed={el:.0f}s")

    with RAW_JSONL.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Per-aspect agreement vs gpt-5.2 details
    haiku_aspect = {}
    for r in results:
        if not r["ok"]: continue
        for a in r["judged"].get("aspects", []):
            key = (r["row_id"], a["aspect"])
            haiku_aspect[key] = {"sup": bool(a.get("supported")),
                                  "match": bool(a.get("sentiment_match"))}
    gpt5_aspect = {}
    with GPT5_DETAILS.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            key = (r["row_id"], r["aspect"])
            gpt5_aspect[key] = {
                "sup": str(r.get("supported","")).lower() in ("true","1","yes"),
                "match": str(r.get("sentiment_match","")).lower() in ("true","1","yes"),
            }
    common = sorted(set(haiku_aspect.keys()) & set(gpt5_aspect.keys()))
    n = len(common)
    sup_agree = sum(1 for k in common if haiku_aspect[k]["sup"] == gpt5_aspect[k]["sup"]) / n
    match_agree = sum(1 for k in common if haiku_aspect[k]["match"] == gpt5_aspect[k]["match"]) / n
    haiku_sup_rate = sum(1 for k in common if haiku_aspect[k]["sup"]) / n
    haiku_match_rate = sum(1 for k in common if haiku_aspect[k]["match"]) / n
    gpt5_sup_rate = sum(1 for k in common if gpt5_aspect[k]["sup"]) / n
    gpt5_match_rate = sum(1 for k in common if gpt5_aspect[k]["match"]) / n

    summary = {
        "candidate_model": MODEL, "n_aspects_common": n,
        "per_aspect_support_agreement": round(sup_agree, 4),
        "per_aspect_match_agreement": round(match_agree, 4),
        "candidate_support_rate": round(haiku_sup_rate, 4),
        "candidate_match_rate": round(haiku_match_rate, 4),
        "gpt5_support_rate": round(gpt5_sup_rate, 4),
        "gpt5_match_rate": round(gpt5_match_rate, 4),
        "accepted": sup_agree >= 0.75 and match_agree >= 0.60,
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    RESULT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nwrote: {RESULT_JSON}")
    print(json.dumps(summary, indent=2))
    return 0 if summary["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
