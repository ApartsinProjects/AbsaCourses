"""Phase D2 at-scale audit: gpt-4.1-mini on the full 10K corpus, real-time concurrent."""
from __future__ import annotations
import concurrent.futures as cf
import csv, json, time
from pathlib import Path
from openai import OpenAI

OPENAI_API_KEY = Path(r"E:\Projects\CourseABSA\.opeai.key").read_text(encoding="utf-8").strip()
CORPUS = Path(r"E:\Projects\CourseABSA\paper\generated_datasets\batch_69cc15c483488190941478aa4e3a976d_generated_reviews.jsonl")
OUT_DIR = Path(r"E:\Claude\CourseABSA\hopeful-kowalevski-04ee10\paper\faithfulness_audit")
RAW_JSONL = OUT_DIR / "at_scale_gpt-4.1-mini_responses.jsonl"
SCORES_CSV = OUT_DIR / "at_scale_gpt-4.1-mini_per_row_scores.csv"
MODEL = "gpt-4.1-mini"
MAX_WORKERS = 24

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


def call_audit(client, row):
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


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Load corpus; row_id = line index (0-based, matching label_faithfulness_audit convention)
    rows = []
    for i, line in enumerate(CORPUS.open(encoding="utf-8")):
        r = json.loads(line)
        rows.append({"row_id": str(i), "text": r.get("text",""), "aspects": r.get("aspects",{})})
    print(f"loaded {len(rows)} rows, submitting with {MODEL} ({MAX_WORKERS} workers)")
    print(f"raw output -> {RAW_JSONL}")
    print()

    client = OpenAI(api_key=OPENAI_API_KEY)
    t0 = time.time()
    # Stream results to disk as they arrive
    fout = RAW_JSONL.open("w", encoding="utf-8")
    n_ok = 0; n_done = 0
    with cf.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(call_audit, client, r): r for r in rows}
        for f in cf.as_completed(futures):
            res = f.result()
            fout.write(json.dumps(res, ensure_ascii=False) + "\n")
            fout.flush()
            n_done += 1
            if res["ok"]: n_ok += 1
            if n_done % 100 == 0:
                elapsed = time.time() - t0
                rate = n_done / elapsed
                eta = (len(rows) - n_done) / rate if rate > 0 else 0
                print(f"  {n_done}/{len(rows)} ok={n_ok} elapsed={elapsed:.0f}s eta={eta:.0f}s")
    fout.close()
    print(f"\nDone. {n_ok}/{len(rows)} successful, total {time.time()-t0:.0f}s")

    # Compute per-row scores
    print(f"\nScoring per-row...")
    by_row = {}
    for line in RAW_JSONL.open(encoding="utf-8"):
        r = json.loads(line)
        if not r["ok"]: continue
        declared = r["declared"]
        by_aspect = {a["aspect"]: a for a in r["judged"].get("aspects",[])}
        n = len(declared); n_sup = 0; n_mat = 0
        for asp in declared:
            j = by_aspect.get(asp)
            if not j: continue
            if j.get("supported"):
                n_sup += 1
                if j.get("sentiment_match"): n_mat += 1
        by_row[r["row_id"]] = (n, n_sup, n_mat)

    with SCORES_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row_id","n_aspects","n_supported","n_matched","support_rate","match_rate","row_score"])
        for rid in sorted(by_row.keys(), key=int):
            n, sup, mat = by_row[rid]
            sr = sup/n if n else 0
            mr = mat/n if n else 0
            w.writerow([rid, n, sup, mat, round(sr,4), round(mr,4), round(mr,4)])
    print(f"wrote {SCORES_CSV}")
    # Quick stats
    scores = [by_row[rid][2]/by_row[rid][0] for rid in by_row if by_row[rid][0]>0]
    print(f"\nAggregate stats:")
    print(f"  rows scored: {len(scores)}")
    print(f"  avg row_score: {sum(scores)/len(scores):.4f}")
    print(f"  score distribution:")
    from collections import Counter
    c = Counter(round(s,2) for s in scores)
    for k in sorted(c.keys()):
        print(f"    {k}: {c[k]}")


if __name__ == "__main__":
    main()
