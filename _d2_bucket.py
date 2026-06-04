"""Phase D2 Stage 3: assign rows to buckets by faithfulness score and write
per-bucket corpus JSONL files.

Buckets:
  top25      : highest-scoring 25% of the corpus
  top50      : highest-scoring 50%
  full       : entire corpus (baseline)
  bot25      : lowest-scoring 25%
  random_5k  : uniform-random 5,000 rows (size-controlled negative-signal control)

Bucket files are written as JSONL with the same schema as the source corpus
so `absa_model_comparison.py --data-path <bucket>.jsonl` works unchanged.
"""
from __future__ import annotations
import csv, json, random
from pathlib import Path

CORPUS = Path(r"E:\Projects\CourseABSA\paper\generated_datasets\batch_69cc15c483488190941478aa4e3a976d_generated_reviews.jsonl")
SCORES_CSV = Path(r"E:\Claude\CourseABSA\hopeful-kowalevski-04ee10\paper\faithfulness_audit\at_scale_gpt-4.1-mini_per_row_scores.csv")
BUCKETS_DIR = Path(r"E:\Claude\CourseABSA\hopeful-kowalevski-04ee10\paper\faithfulness_audit\buckets")
MANIFEST = BUCKETS_DIR / "manifest.json"
SEED = 42
RANDOM_SIZE = 5000


def load_corpus():
    rows = []
    for i, line in enumerate(CORPUS.open(encoding="utf-8")):
        rows.append((i, line))
    return rows


def load_scores():
    out = {}
    with SCORES_CSV.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[int(r["row_id"])] = float(r["row_score"])
    return out


def write_bucket(name, row_indices, corpus_lines, scores):
    out_path = BUCKETS_DIR / f"{name}.jsonl"
    ids_path = BUCKETS_DIR / f"{name}.row_ids.txt"
    sorted_ids = sorted(row_indices)
    with out_path.open("w", encoding="utf-8") as f:
        for idx in sorted_ids:
            f.write(corpus_lines[idx])
    with ids_path.open("w", encoding="utf-8") as f:
        for idx in sorted_ids:
            f.write(f"{idx}\n")
    bucket_scores = [scores[i] for i in sorted_ids if i in scores]
    info = {
        "bucket": name, "n": len(sorted_ids),
        "score_min": round(min(bucket_scores), 4) if bucket_scores else None,
        "score_max": round(max(bucket_scores), 4) if bucket_scores else None,
        "score_mean": round(sum(bucket_scores)/len(bucket_scores), 4) if bucket_scores else None,
        "n_with_score": len(bucket_scores),
        "path": str(out_path), "ids_path": str(ids_path),
    }
    return info


def main():
    BUCKETS_DIR.mkdir(parents=True, exist_ok=True)
    corpus = load_corpus()
    corpus_lines = [line for _, line in corpus]
    n_total = len(corpus_lines)
    print(f"corpus: {n_total} rows")

    scores = load_scores()
    print(f"scored: {len(scores)} rows ({len(scores)/n_total*100:.1f}%)")
    if len(scores) < n_total * 0.95:
        print(f"WARNING: < 95% of corpus has a score; bucketing may be skewed.")

    # Sort row indices by (score desc, row_id asc) for deterministic ranking.
    ranked = sorted(
        [(scores.get(i, -1.0), i) for i in range(n_total)],
        key=lambda t: (-t[0], t[1]),
    )
    ranked_ids = [i for _, i in ranked]

    # Use ALL scored rows in the ranking; rows missing a score are placed at
    # the BOTTOM (score -1.0). They will only appear in `full` and `bot25`
    # (depending on the cut), never in `top25` / `top50`. This is conservative.

    top25_n = n_total // 4         # 2500
    top50_n = n_total // 2         # 5000
    bot25_n = n_total // 4         # 2500

    buckets = {}
    buckets["top25"] = ranked_ids[:top25_n]
    buckets["top50"] = ranked_ids[:top50_n]
    buckets["full"]  = list(range(n_total))
    buckets["bot25"] = ranked_ids[-bot25_n:]

    rng = random.Random(SEED)
    rand_indices = list(range(n_total))
    rng.shuffle(rand_indices)
    buckets["random_5k"] = rand_indices[:RANDOM_SIZE]

    manifest = {"seed": SEED, "n_corpus": n_total, "buckets": []}
    for name in ("top25","top50","full","bot25","random_5k"):
        info = write_bucket(name, buckets[name], corpus_lines, scores)
        manifest["buckets"].append(info)
        print(f"  {name:10s} n={info['n']:6d} score_mean={info['score_mean']}")

    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nwrote: {MANIFEST}")


if __name__ == "__main__":
    main()
