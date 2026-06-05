"""Reviewer-response EVIDENCE ITEM 1: test-split label faithfulness.

Co-computes, in ONE pass, faithfulness statistics (mean support_rate, mean
match_rate, mean row_score, and the quantized row_score distribution) over:
  (a) the FULL 10K synthetic corpus, and
  (b) a 1000-row held-out split reconstructed at seed 42 -- TWO ways, both
      clearly labelled as fallbacks because the exact internal-benchmark test
      row_ids cannot be reconstructed with certainty (see registry note).

Why fallback: the internal benchmark notebook (edu/absa_train_new.ipynb)
splits via TWO nested sklearn train_test_split(random_state=42) calls
(80/20 then 50/50 on the holdout) operating on a file named
'o4_mini_final_student_reviews_clean.jsonl' that is NOT present on disk.
The manuscript (Section 5.9 / line 597) instead describes a 'seeded
permutation with seed 42' + fixed 0.80/0.10/0.10 proportions. These two
procedures yield DIFFERENT row partitions, and the notebook's source corpus
file is unavailable, so the exact test row_ids cannot be reproduced with
certainty. We therefore report the full-corpus faithfulness (the
construct-relevant quantity) plus two seed-42 1000-row holdout
reconstructions as bounding fallbacks.

NO model is trained, NO API is called. All numbers come from the on-disk
per-row faithfulness CSV.
"""
from __future__ import annotations
import csv, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

REPO = Path(r"E:\Claude\CourseABSA\hopeful-kowalevski-04ee10")
SCORES_CSV = REPO / "paper" / "faithfulness_audit" / "at_scale_gpt-4.1-mini_per_row_scores.csv"
OUT_JSON = REPO / "paper" / "outputs" / "tables" / "reviewer_response_item1_testsplit_faithfulness.json"
OUT_CSV = REPO / "paper" / "outputs" / "tables" / "reviewer_response_item1_testsplit_faithfulness.csv"

N_CORPUS = 10000
SEED = 42
VAL = 0.10
TEST = 0.10

# Quantized row_score levels observed in the audit (n_aspects in {1,2,3} ->
# match fractions land on 0, 1/3, 1/2, 2/3, 1).
LEVELS = [0.0, 0.3333, 0.5, 0.6667, 1.0]


def load_scores():
    df = pd.read_csv(SCORES_CSV)
    # row_id is 0-based line index into the 10K corpus.
    df = df.sort_values("row_id").reset_index(drop=True)
    return df


def level_distribution(series):
    """% of rows whose row_score rounds to each quantized level."""
    n = len(series)
    out = {}
    rounded = series.round(2)
    # map each rounded value to the nearest canonical level for clean reporting
    canon = {0.0: "0.00", 0.33: "0.33", 0.5: "0.50", 0.67: "0.67", 1.0: "1.00"}
    counts = {v: 0 for v in canon.values()}
    other = 0
    for x in rounded:
        key = round(x, 2)
        if key in canon:
            counts[canon[key]] += 1
        else:
            other += 1
    for k in counts:
        out[k] = round(100.0 * counts[k] / n, 2)
    if other:
        out["other"] = round(100.0 * other / n, 2)
    return out


def summarize(df_subset, label, row_ids=None):
    n = len(df_subset)
    return {
        "subset": label,
        "n_rows": int(n),
        "mean_support_rate": round(float(df_subset["support_rate"].mean()), 4),
        "mean_match_rate": round(float(df_subset["match_rate"].mean()), 4),
        "mean_row_score": round(float(df_subset["row_score"].mean()), 4),
        "row_score_dist_pct": level_distribution(df_subset["row_score"]),
        "n_aspects_total": int(df_subset["n_aspects"].sum()),
        "row_ids_min_max": [int(row_ids.min()), int(row_ids.max())] if row_ids is not None else None,
    }


def main():
    df = load_scores()
    assert len(df) == N_CORPUS, f"expected {N_CORPUS} scored rows, got {len(df)}"
    row_index = df["row_id"].to_numpy()

    results = {
        "evidence_item": "1 - test-split label faithfulness",
        "input_artifact": str(SCORES_CSV),
        "n_scored_rows": int(len(df)),
        "caveats": [
            "row_id is the 0-based line index into the 10K synthetic corpus "
            "(batch_69cc15c483488190941478aa4e3a976d_generated_reviews.jsonl).",
            "The faithfulness audit indexes the 10K TRAINING corpus. The internal "
            "benchmark's 8000/1000/1000 split (manuscript Section 5.9) is defined on "
            "this same 10K corpus per the manuscript, but the EXACT test row_ids "
            "cannot be reconstructed with certainty: the training notebook "
            "(edu/absa_train_new.ipynb) splits via two nested sklearn "
            "train_test_split(random_state=42) calls on a file "
            "'o4_mini_final_student_reviews_clean.jsonl' that is NOT on disk, while "
            "the manuscript text describes a numpy 'seeded permutation with seed 42' "
            "+ fixed 0.80/0.10/0.10. These procedures give different partitions.",
            "Therefore the full-10K faithfulness is the primary number; two seed-42 "
            "1000-row holdout reconstructions are reported as labelled fallbacks.",
            "No per-row TEST predictions exist anywhere on disk (searched "
            "E:\\Projects\\CourseABSA and E:\\Claude\\CourseABSA for test_results*.csv "
            "/ *test_pred*); the high-faithfulness-subset detection F1 is therefore "
            "NOT computable from saved artifacts and is intentionally omitted.",
        ],
        "subsets": [],
    }

    # (a) FULL 10K
    results["subsets"].append(summarize(df, "full_10K", row_index))

    # (b1) Fallback A: numpy seeded permutation (matches manuscript text)
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(N_CORPUS)
    n_train = int(round(N_CORPUS * (1 - VAL - TEST)))  # 8000
    n_val = int(round(N_CORPUS * VAL))                  # 1000
    test_pos = perm[n_train + n_val:]                   # last 1000 positions
    df_test_np = df.iloc[np.sort(test_pos)].reset_index(drop=True)
    s = summarize(df_test_np, "seed42_holdout_numpy_permutation_1000 (FALLBACK: matches manuscript text)",
                  np.sort(test_pos))
    results["subsets"].append(s)

    # (b2) Fallback B: nested sklearn train_test_split (matches notebook code)
    # three_way_split(df, calib=0.10, test=0.10, seed=42):
    #   hold = 0.20; tr, ho = split(df, test_size=0.20, rs=42)
    #   ca, te = split(ho, test_size=0.10/0.20=0.5, rs=42)
    idx = pd.DataFrame({"row_id": df["row_id"].values})
    tr, ho = train_test_split(idx, test_size=VAL + TEST, random_state=SEED)
    ca, te = train_test_split(ho, test_size=TEST / (VAL + TEST), random_state=SEED)
    te_ids = te["row_id"].to_numpy()
    df_test_sk = df[df["row_id"].isin(te_ids)].reset_index(drop=True)
    s2 = summarize(df_test_sk, "seed42_holdout_nested_sklearn_1000 (FALLBACK: matches notebook code path)",
                   np.sort(te_ids))
    results["subsets"].append(s2)

    # filtered-test-F1: not computable
    results["filtered_test_f1"] = {
        "computable": False,
        "reason": "No saved per-row TEST predictions on disk. Cannot recompute "
                  "detection micro-F1 on row_score==1.0 subset vs full test. NOT fabricated.",
    }

    OUT_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")

    # Flat CSV for easy paste
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subset", "n_rows", "mean_support_rate", "mean_match_rate",
                    "mean_row_score", "pct_0.00", "pct_0.33", "pct_0.50",
                    "pct_0.67", "pct_1.00"])
        for sub in results["subsets"]:
            d = sub["row_score_dist_pct"]
            w.writerow([sub["subset"], sub["n_rows"], sub["mean_support_rate"],
                        sub["mean_match_rate"], sub["mean_row_score"],
                        d.get("0.00"), d.get("0.33"), d.get("0.50"),
                        d.get("0.67"), d.get("1.00")])

    # Console report
    print("=== EVIDENCE ITEM 1: test-split label faithfulness ===")
    for sub in results["subsets"]:
        print(f"\n[{sub['subset']}]  n={sub['n_rows']}")
        print(f"  mean support_rate = {sub['mean_support_rate']}")
        print(f"  mean match_rate   = {sub['mean_match_rate']}")
        print(f"  mean row_score    = {sub['mean_row_score']}")
        print(f"  row_score dist %% : {sub['row_score_dist_pct']}")
    print(f"\nfiltered_test_f1: {results['filtered_test_f1']}")
    print(f"\nwrote {OUT_JSON}")
    print(f"wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
