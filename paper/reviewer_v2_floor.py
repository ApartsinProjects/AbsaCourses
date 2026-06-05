"""V2 -- TRIVIAL BASELINE FLOOR (local CPU, no Modal, deterministic).

On mode A's internal 20-aspect TEST split (the same three_way_split the worker uses,
seed 42), compute detection micro-F1 for four trivial predictors so the paper's 0.277
can be shown above chance:
  (a) predict-all-negative      (every aspect absent on every row)
  (b) predict-all-positive      (every aspect present on every row)
  (c) predict-by-train-frequency (predict aspect present iff its TRAIN prevalence > 0.5)
  (d) uniform-random per-aspect at TRAIN prevalence (Bernoulli(p_a) per cell; averaged over
      many draws -> the expected-floor; also reports the analytic expectation)

Reuses eng.three_way_split + load_absa_dataset so the test split is byte-identical to the
worker's. Output: a result JSON + CSV under the v5 round dir.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))

from absa_data_io import load_absa_dataset  # noqa: E402
import absa_model_comparison as eng  # noqa: E402

CORPUS = HERE / "reviewer_ab_data" / "generated_reviews_10k.jsonl"
OUT_DIR = HERE / "experiment_rounds" / "validation_v5_20260605" / "V2_trivial_floor"
PAPER_BERT_F1_SINGLE = 0.2760


def micro(yt: np.ndarray, yp: np.ndarray) -> dict:
    return {
        "micro_precision": float(precision_score(yt.ravel(), yp.ravel(), zero_division=0)),
        "micro_recall": float(recall_score(yt.ravel(), yp.ravel(), zero_division=0)),
        "micro_f1": float(f1_score(yt.ravel(), yp.ravel(), zero_division=0)),
    }


def main(seed: int = 42) -> None:
    cfg = eng.Config(seed=seed)
    df = load_absa_dataset(CORPUS).reset_index(drop=True)
    assert len(df) == 10000, len(df)
    aspects = eng.discover_aspects(df)
    train_df, calib_df, test_df = eng.three_way_split(df, cfg.split_calib, cfg.split_test, seed)
    n_test = len(test_df)
    A = len(aspects)
    a2i = {a: i for i, a in enumerate(aspects)}

    # gold matrices
    def to_mat(d):
        M = np.zeros((len(d), A), dtype=int)
        for i, row in enumerate(d["aspects"]):
            for a in row:
                M[i, a2i[a]] = 1
        return M

    y_test = to_mat(test_df)
    y_train = to_mat(train_df)
    train_prev = y_train.mean(axis=0)  # per-aspect prevalence in TRAIN

    results = {}

    # (a) all-negative
    results["predict_all_negative"] = micro(y_test, np.zeros_like(y_test))
    # (b) all-positive
    results["predict_all_positive"] = micro(y_test, np.ones_like(y_test))
    # (c) predict by train frequency (present iff train prevalence > 0.5)
    by_freq = (train_prev > 0.5).astype(int)
    pred_c = np.tile(by_freq, (n_test, 1))
    results["predict_by_train_frequency_gt0p5"] = micro(y_test, pred_c)
    results["predict_by_train_frequency_gt0p5"]["n_aspects_predicted_present"] = int(by_freq.sum())

    # (d) uniform-random per-aspect at train prevalence -- average over draws + analytic
    rng = np.random.default_rng(seed)
    n_draws = 200
    f1s, ps, rs = [], [], []
    for _ in range(n_draws):
        draw = (rng.random((n_test, A)) < train_prev).astype(int)
        m = micro(y_test, draw)
        f1s.append(m["micro_f1"]); ps.append(m["micro_precision"]); rs.append(m["micro_recall"])
    results["random_at_train_prevalence"] = {
        "micro_f1_mean": float(np.mean(f1s)),
        "micro_f1_std": float(np.std(f1s)),
        "micro_precision_mean": float(np.mean(ps)),
        "micro_recall_mean": float(np.mean(rs)),
        "n_draws": n_draws,
    }

    out = {
        "experiment": "V2_trivial_baseline_floor",
        "seed": seed,
        "n_test_rows": int(n_test),
        "n_aspects": A,
        "aspects": aspects,
        "train_prevalence": {a: float(train_prev[i]) for i, a in enumerate(aspects)},
        "paper_bert_micro_f1": PAPER_BERT_F1_SINGLE,
        "floors": results,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "result.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    # flat CSV
    rows = [
        {"baseline": "predict_all_negative", "micro_f1": results["predict_all_negative"]["micro_f1"],
         "micro_precision": results["predict_all_negative"]["micro_precision"],
         "micro_recall": results["predict_all_negative"]["micro_recall"]},
        {"baseline": "predict_all_positive", "micro_f1": results["predict_all_positive"]["micro_f1"],
         "micro_precision": results["predict_all_positive"]["micro_precision"],
         "micro_recall": results["predict_all_positive"]["micro_recall"]},
        {"baseline": "predict_by_train_frequency_gt0p5",
         "micro_f1": results["predict_by_train_frequency_gt0p5"]["micro_f1"],
         "micro_precision": results["predict_by_train_frequency_gt0p5"]["micro_precision"],
         "micro_recall": results["predict_by_train_frequency_gt0p5"]["micro_recall"]},
        {"baseline": "random_at_train_prevalence_mean",
         "micro_f1": results["random_at_train_prevalence"]["micro_f1_mean"],
         "micro_precision": results["random_at_train_prevalence"]["micro_precision_mean"],
         "micro_recall": results["random_at_train_prevalence"]["micro_recall_mean"]},
        {"baseline": "bert_paper_reference", "micro_f1": PAPER_BERT_F1_SINGLE,
         "micro_precision": None, "micro_recall": None},
    ]
    pd.DataFrame(rows).to_csv(OUT_DIR / "floors.csv", index=False)

    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
