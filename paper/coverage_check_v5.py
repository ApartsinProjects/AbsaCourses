"""Coverage/diversity check: does faithfulness ROW-filtering (top50/top25) skew
the corpus away from hard aspects or narrow its diversity? Justifies the
filter-and-scale recipe (or motivates per-aspect quotas).

Local CPU. Inputs: the row-id-aligned corpus + bucket row_id lists.
"""
import json
import os
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(HERE, "reviewer_ab_data", "generated_reviews_10k.jsonl")
BUCKETS = os.path.join(HERE, "faithfulness_audit", "buckets")

rows = [json.loads(l) for l in open(CORPUS, encoding="utf-8")]
n = len(rows)
# aspects per row (keys of the aspects dict)
def aspects_of(r):
    a = r.get("aspects") or r.get("aspects_original") or {}
    return list(a.keys())
def words(r):
    return len((r.get("text") or "").split())
def style(r):
    return r.get("style") or (r.get("nuance_attributes") or {}).get("writing_style") or "NA"

bucket_ids = {}
for b in ["full", "top50", "top25", "bot25", "random_5k"]:
    p = os.path.join(BUCKETS, f"{b}.row_ids.txt")
    if os.path.exists(p):
        bucket_ids[b] = set(int(x) for x in open(p) if x.strip())

ALL_ASPECTS = sorted({a for r in rows for a in aspects_of(r)})
# "hard" aspects flagged in the paper
HARD = {"peer_interaction", "support", "interest", "feedback_quality", "clarity"}

def profile(ids):
    sub = [rows[i] for i in sorted(ids) if i < len(rows)]
    asp = Counter(a for r in sub for a in aspects_of(r))
    wl = [words(r) for r in sub]
    st = Counter(style(r) for r in sub)
    ac = Counter(len(aspects_of(r)) for r in sub)
    return {"n": len(sub), "aspect_counts": asp,
            "mean_words": round(sum(wl) / max(1, len(wl)), 1),
            "n_styles": len(st), "aspect_count_dist": dict(sorted(ac.items()))}

full = profile(bucket_ids["full"]) if "full" in bucket_ids else profile(set(range(n)))
print(f"=== Corpus coverage/diversity by bucket (full n={full['n']}) ===")
print(f"{'bucket':10} {'n':>6} {'mean_wd':>8} {'styles':>7}  aspect_count_dist")
for b in ["full", "top50", "top25", "bot25", "random_5k"]:
    if b not in bucket_ids:
        continue
    pr = profile(bucket_ids[b])
    print(f"{b:10} {pr['n']:>6} {pr['mean_words']:>8} {pr['n_styles']:>7}  {pr['aspect_count_dist']}")

# Per-aspect RETENTION in top50 vs full: fraction of each aspect's full-corpus rows kept.
print("\n=== Per-aspect retention in top50 vs full (sorted asc; <0.40 = under-retained) ===")
top50 = profile(bucket_ids["top50"]) if "top50" in bucket_ids else None
if top50:
    fa = full["aspect_counts"]; ta = top50["aspect_counts"]
    overall_keep = top50["n"] / full["n"]
    print(f"(overall row-keep rate top50/full = {overall_keep:.3f}; a balanced filter retains each aspect near this)")
    rows_out = []
    for a in ALL_ASPECTS:
        ret = ta.get(a, 0) / max(1, fa.get(a, 0))
        rows_out.append((ret, a, fa.get(a, 0), ta.get(a, 0)))
    for ret, a, f, t in sorted(rows_out):
        flag = "  <-- HARD" if a in HARD else ""
        skew = "  *** UNDER" if ret < overall_keep - 0.08 else ("  +over" if ret > overall_keep + 0.08 else "")
        print(f"  {a:22} full={f:>5} top50={t:>5} retention={ret:.3f}{skew}{flag}")
