"""Reviewer A/B experiment worker (runs inside the Modal container).

Two paper-bound experiments for the CourseABSA TMLR revision:

EXPERIMENT A -- filtered-test detection micro-F1 (reviewer concern 1).
  Train bert-base-uncased on the 8000-row train split of the 10K synthetic
  corpus (three_way_split numpy seeded permutation, seed S), evaluate detection
  micro-F1 on the 1000-row test split. The test rows are joined by their corpus
  row_id (== positional line index, verified identical to the faithfulness-audit
  row_id) to the per-row faithfulness scores. We then recompute detection
  micro-F1 on (a) the full test set, (b) row_score == 1.0, and (c) row_score >= 0.5.

  REPRODUCE GATE: the full-test micro-F1 must land within +-0.02 of the paper's
  0.2760 (single seed) / 0.2791 (3-seed mean). If not, the run records the
  discrepancy and SKIPS the filtered number.

EXPERIMENT B -- real-Herath-trained baseline (reviewer concern 4).
  Train bert-base-uncased on a real-train split of the 2,829 mapped Herath
  reviews (9-aspect overlap), evaluate detection micro-F1 on a real-test split.
  This is the real-trained UPPER reference for the synthetic-only 0.4593.

Invocation (inside container, cwd=/app):
  python paper/reviewer_ab_worker.py --experiment A --seed 42 --out /results/...
  python paper/reviewer_ab_worker.py --experiment B --seed 42 --out /results/...
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

sys.path.insert(0, "/app/paper")

from absa_data_io import load_absa_dataset  # noqa: E402
import absa_model_comparison as eng  # noqa: E402


PAPER_BERT_F1_SINGLE = 0.2760
PAPER_BERT_F1_MEAN = 0.2791
REPRO_TOL = 0.02


def micro_f1_from_arrays(det_true: np.ndarray, det_preds: np.ndarray) -> dict:
    return {
        "micro_precision": float(precision_score(det_true.ravel(), det_preds.ravel(), zero_division=0)),
        "micro_recall": float(recall_score(det_true.ravel(), det_preds.ravel(), zero_division=0)),
        "micro_f1": float(f1_score(det_true.ravel(), det_preds.ravel(), zero_division=0)),
    }


def subset_micro_f1(det_true: np.ndarray, det_preds: np.ndarray, mask: np.ndarray) -> dict:
    """Micro-F1 restricted to the rows where mask is True."""
    sub_true = det_true[mask]
    sub_preds = det_preds[mask]
    if sub_true.size == 0:
        return {"n_rows": 0, "micro_precision": None, "micro_recall": None, "micro_f1": None}
    m = micro_f1_from_arrays(sub_true, sub_preds)
    m["n_rows"] = int(mask.sum())
    return m


# ------------------------------------------------------------------ Experiment A
def run_experiment_a(corpus_path: Path, scores_csv: Path, seed: int, out_dir: Path) -> dict:
    cfg = eng.Config(seed=seed)  # 192 tok, bs 8, lr 3e-5, 3 epochs, patience 2, 0.10/0.10 split
    eng.set_seed(seed)
    eng.log_event(f"[A seed={seed}] device={cfg.device}")

    df = load_absa_dataset(corpus_path)
    n = len(df)
    eng.log_event(f"[A seed={seed}] loaded corpus rows={n}")
    assert n == 10000, f"expected 10000 corpus rows, got {n}"

    # row_id == positional index (verified line-index == audit row_id)
    df = df.reset_index(drop=True)
    df["row_id"] = np.arange(n)

    aspects = eng.discover_aspects(df)
    eng.log_event(f"[A seed={seed}] n_aspects={len(aspects)}")

    # three_way_split reproduced locally so we KEEP the test row_ids
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    calib_size, test_size = cfg.split_calib, cfg.split_test
    counts = np.floor(np.array([1.0 - calib_size - test_size, calib_size, test_size]) * n).astype(int)
    counts = np.maximum(counts, 1)
    while counts.sum() > n:
        reducible = [idx for idx in np.argsort(-counts) if counts[idx] > 1]
        if not reducible:
            break
        counts[reducible[0]] -= 1
    if counts.sum() < n:
        counts[0] += n - counts.sum()
    train_n, calib_n, test_n = counts.tolist()
    train_idx = perm[:train_n]
    calib_idx = perm[train_n:train_n + calib_n]
    test_idx = perm[train_n + calib_n:train_n + calib_n + test_n]

    train_df = df.iloc[train_idx].reset_index(drop=True)
    calib_df = df.iloc[calib_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    test_row_ids = df.iloc[test_idx]["row_id"].to_numpy()
    eng.log_event(f"[A seed={seed}] split train={len(train_df)} calib={len(calib_df)} test={len(test_df)}")

    # Sanity vs library three_way_split (must match indices order)
    lib_tr, lib_ca, lib_te = eng.three_way_split(df, calib_size, test_size, seed)
    assert len(lib_te) == len(test_df), "library/local test size mismatch"
    assert (lib_te["row_id"].to_numpy() == test_row_ids).all(), "library/local test row_id mismatch"
    eng.log_event(f"[A seed={seed}] split matches library three_way_split exactly")

    # Train detection head (the recipe under test); sentiment head trained too for full artifact parity
    det_model, det_tok = eng.train_detection("bert-base-uncased", train_df, calib_df, aspects, cfg)
    sent_model, sent_tok = eng.train_sentiment("bert-base-uncased", train_df, calib_df, aspects, cfg)
    thresholds = eng.calibrate_thresholds(det_model, calib_df, det_tok, aspects, cfg)

    # Evaluate on test; reuse engine internals to get det_true/det_preds aligned to test_df order
    from torch.utils.data import DataLoader
    det_loader = DataLoader(eng.DetectionDataset(test_df, det_tok, aspects, cfg.max_len), batch_size=cfg.batch_size, shuffle=False)
    det_probs, det_true = eng.collect_detection(det_model, det_loader, cfg.device)
    thr_vec = np.array([thresholds[a] for a in aspects], dtype=np.float32)
    det_preds = (det_probs >= thr_vec).astype(int)

    full_metrics = micro_f1_from_arrays(det_true, det_preds)
    full_metrics["n_rows"] = int(len(test_df))
    eng.log_event(f"[A seed={seed}] FULL test micro_f1={full_metrics['micro_f1']:.4f}")

    # Reproduce gate
    repro_ok = abs(full_metrics["micro_f1"] - PAPER_BERT_F1_SINGLE) <= REPRO_TOL or \
        abs(full_metrics["micro_f1"] - PAPER_BERT_F1_MEAN) <= REPRO_TOL
    eng.log_event(f"[A seed={seed}] reproduce gate (|{full_metrics['micro_f1']:.4f} - {PAPER_BERT_F1_SINGLE}|<= {REPRO_TOL}): {repro_ok}")

    # Save per-row test predictions (always, regardless of gate)
    scores = pd.read_csv(scores_csv).set_index("row_id")
    per_row_records = []
    for i in range(len(test_df)):
        rid = int(test_row_ids[i])
        gold = test_df.iloc[i]["aspects"]
        pred = {aspects[j]: ("present" ) for j in range(len(aspects)) if int(det_preds[i, j]) == 1}
        row_score = float(scores.loc[rid, "row_score"]) if rid in scores.index else None
        per_row_records.append({
            "row_id": rid,
            "text": str(test_df.iloc[i]["text"]),
            "gold_aspects": list(gold.keys()),
            "predicted_aspects": list(pred.keys()),
            "row_faithfulness_score": row_score,
            "detection_targets": {aspects[j]: int(det_true[i, j]) for j in range(len(aspects))},
            "detection_predictions": {aspects[j]: int(det_preds[i, j]) for j in range(len(aspects))},
            "detection_probabilities": {aspects[j]: float(det_probs[i, j]) for j in range(len(aspects))},
        })
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "test_predictions.jsonl"
    with pred_path.open("w", encoding="utf-8") as f:
        for r in per_row_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Join faithfulness and compute filtered metrics
    row_scores = np.array([
        (float(scores.loc[int(rid), "row_score"]) if int(rid) in scores.index else np.nan)
        for rid in test_row_ids
    ])
    n_joined = int(np.isfinite(row_scores).sum())
    eng.log_event(f"[A seed={seed}] joined faithfulness for {n_joined}/{len(test_df)} test rows")

    result = {
        "experiment": "A_filtered_test_f1",
        "seed": seed,
        "n_aspects": len(aspects),
        "aspects": aspects,
        "thresholds": {a: float(thresholds[a]) for a in aspects},
        "paper_reference": {"bert_micro_f1_single_seed": PAPER_BERT_F1_SINGLE,
                            "bert_micro_f1_3seed_mean": PAPER_BERT_F1_MEAN, "tol": REPRO_TOL},
        "full_test": full_metrics,
        "reproduce_ok": bool(repro_ok),
        "n_test_rows_joined_to_faithfulness": n_joined,
        "test_predictions_path": str(pred_path),
    }

    if repro_ok:
        faithful_mask = np.isfinite(row_scores) & (np.isclose(row_scores, 1.0))
        half_mask = np.isfinite(row_scores) & (row_scores >= 0.5)
        result["faithful_only_row_score_eq_1"] = subset_micro_f1(det_true, det_preds, faithful_mask)
        result["row_score_ge_0p5"] = subset_micro_f1(det_true, det_preds, half_mask)
        # also the complement (noisy rows) for the contrast
        noisy_mask = np.isfinite(row_scores) & (row_scores < 1.0)
        result["noisy_row_score_lt_1"] = subset_micro_f1(det_true, det_preds, noisy_mask)
        result["delta_faithful_minus_full"] = (
            None if result["faithful_only_row_score_eq_1"]["micro_f1"] is None
            else round(result["faithful_only_row_score_eq_1"]["micro_f1"] - full_metrics["micro_f1"], 4)
        )
        eng.log_event(
            f"[A seed={seed}] FILTERED faithful(=1.0) micro_f1="
            f"{result['faithful_only_row_score_eq_1']['micro_f1']} "
            f"(n={result['faithful_only_row_score_eq_1']['n_rows']}); "
            f">=0.5 micro_f1={result['row_score_ge_0p5']['micro_f1']} "
            f"(n={result['row_score_ge_0p5']['n_rows']})"
        )
    else:
        result["filtered_skipped_reason"] = (
            "Full-test micro-F1 did not reproduce the paper's 0.2760/0.2791 within "
            f"+-{REPRO_TOL}; filtered number intentionally omitted (inconsistent "
            "filtered-F1 is worse than none)."
        )

    (out_dir / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


# ============================================================ shared mode-A split
def _load_corpus_with_row_ids(corpus_path: Path) -> pd.DataFrame:
    df = load_absa_dataset(corpus_path)
    n = len(df)
    assert n == 10000, f"expected 10000 corpus rows, got {n}"
    df = df.reset_index(drop=True)
    df["row_id"] = np.arange(n)
    return df


def _mode_a_split(df: pd.DataFrame, cfg: "eng.Config", seed: int):
    """Reproduce mode A's three_way_split, keeping per-row row_ids on every split.

    Returns (train_df, calib_df, test_df, train_row_ids, test_row_ids, aspects).
    Verified to match eng.three_way_split exactly (same assertion as run_experiment_a).
    """
    n = len(df)
    aspects = eng.discover_aspects(df)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    calib_size, test_size = cfg.split_calib, cfg.split_test
    counts = np.floor(np.array([1.0 - calib_size - test_size, calib_size, test_size]) * n).astype(int)
    counts = np.maximum(counts, 1)
    while counts.sum() > n:
        reducible = [idx for idx in np.argsort(-counts) if counts[idx] > 1]
        if not reducible:
            break
        counts[reducible[0]] -= 1
    if counts.sum() < n:
        counts[0] += n - counts.sum()
    train_n, calib_n, test_n = counts.tolist()
    train_idx = perm[:train_n]
    calib_idx = perm[train_n:train_n + calib_n]
    test_idx = perm[train_n + calib_n:train_n + calib_n + test_n]

    train_df = df.iloc[train_idx].reset_index(drop=True)
    calib_df = df.iloc[calib_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    train_row_ids = df.iloc[train_idx]["row_id"].to_numpy()
    test_row_ids = df.iloc[test_idx]["row_id"].to_numpy()

    # Sanity vs library three_way_split (must match the test indices exactly)
    lib_tr, lib_ca, lib_te = eng.three_way_split(df, calib_size, test_size, seed)
    assert len(lib_te) == len(test_df), "library/local test size mismatch"
    assert (lib_te["row_id"].to_numpy() == test_row_ids).all(), "library/local test row_id mismatch"
    return train_df, calib_df, test_df, train_row_ids, test_row_ids, aspects


def _train_eval_micro_f1(train_df, calib_df, test_df, aspects, cfg):
    """Train detection head on train_df (calibrate thresholds on calib_df) and return
    (det_true, det_preds, det_probs, thresholds, full_metrics) on test_df.
    Identical recipe to run_experiment_a's detection path."""
    from torch.utils.data import DataLoader
    det_model, det_tok = eng.train_detection("bert-base-uncased", train_df, calib_df, aspects, cfg)
    thresholds = eng.calibrate_thresholds(det_model, calib_df, det_tok, aspects, cfg)
    det_loader = DataLoader(eng.DetectionDataset(test_df, det_tok, aspects, cfg.max_len),
                            batch_size=cfg.batch_size, shuffle=False)
    det_probs, det_true = eng.collect_detection(det_model, det_loader, cfg.device)
    thr_vec = np.array([thresholds[a] for a in aspects], dtype=np.float32)
    det_preds = (det_probs >= thr_vec).astype(int)
    full_metrics = micro_f1_from_arrays(det_true, det_preds)
    full_metrics["n_rows"] = int(len(test_df))
    return det_true, det_preds, det_probs, thresholds, full_metrics


# ------------------------------------------------------------------ V1 PERMUTATION CONTROL
def run_experiment_v1(corpus_path: Path, seed: int, out_dir: Path) -> dict:
    """Permutation control: shuffle which 20-aspect label-vector is paired with which
    TRAIN text (label marginal preserved), train on the scrambled pairs, evaluate on the
    REAL-label test split (unchanged). If permuted micro-F1 ~= the real 0.277, the model
    only learned label priors; if much lower, the text genuinely predicts the labels."""
    cfg = eng.Config(seed=seed)
    eng.set_seed(seed)
    eng.log_event(f"[V1 seed={seed}] device={cfg.device}")
    df = _load_corpus_with_row_ids(corpus_path)
    train_df, calib_df, test_df, train_row_ids, test_row_ids, aspects = _mode_a_split(df, cfg, seed)
    eng.log_event(f"[V1 seed={seed}] split train={len(train_df)} calib={len(calib_df)} test={len(test_df)} aspects={len(aspects)}")

    # Permute the label column of the TRAIN split relative to the texts (seeded).
    # 'aspects' is the label dict used by DetectionDataset; scramble its row-assignment.
    rng = np.random.default_rng(seed + 1000)  # distinct stream from the split perm
    perm = rng.permutation(len(train_df))
    permuted_train = train_df.copy()
    permuted_train["aspects"] = train_df["aspects"].to_numpy()[perm]
    permuted_train["target_attributes"] = permuted_train["aspects"]
    permuted_train = permuted_train.reset_index(drop=True)
    n_fixed = int((perm == np.arange(len(train_df))).sum())
    eng.log_event(f"[V1 seed={seed}] permuted {len(train_df)} train label-vectors "
                  f"(fixed points={n_fixed}); marginal preserved by construction")

    # Calibrate on the REAL-label calib split (thresholds calibrated against true labels,
    # exactly as mode A does -- we only scramble TRAIN supervision, not calibration/test).
    det_true, det_preds, det_probs, thresholds, full_metrics = _train_eval_micro_f1(
        permuted_train, calib_df, test_df, aspects, cfg)
    eng.log_event(f"[V1 seed={seed}] PERMUTED-train test micro_f1={full_metrics['micro_f1']:.4f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    # per-row test predictions
    with (out_dir / "test_predictions.jsonl").open("w", encoding="utf-8") as f:
        for i in range(len(test_df)):
            rid = int(test_row_ids[i])
            f.write(json.dumps({
                "row_id": rid,
                "text": str(test_df.iloc[i]["text"]),
                "gold_aspects": list(test_df.iloc[i]["aspects"].keys()),
                "predicted_aspects": [aspects[j] for j in range(len(aspects)) if int(det_preds[i, j]) == 1],
            }, ensure_ascii=False) + "\n")
    (out_dir / "train_permutation.json").write_text(json.dumps({
        "seed": seed, "perm_seed": seed + 1000, "n_train": len(train_df),
        "fixed_points": n_fixed, "permutation": perm.tolist(),
    }), encoding="utf-8")

    result = {
        "experiment": "V1_permutation_control",
        "seed": seed,
        "n_aspects": len(aspects),
        "real_label_reference_micro_f1": PAPER_BERT_F1_SINGLE,
        "permuted_train_full_test": full_metrics,
        "permuted_micro_f1": full_metrics["micro_f1"],
        "gap_real_minus_permuted": round(PAPER_BERT_F1_SINGLE - full_metrics["micro_f1"], 4),
        "thresholds": {a: float(thresholds[a]) for a in aspects},
        "interpretation": ("permuted ~= 0.277 => label-prior only (no signal); "
                           "permuted << 0.277 => text genuinely predicts labels (signal present)"),
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


# ------------------------------------------------------------------ V4 CLEAN-LABEL CEILING
def run_experiment_v4(corpus_path: Path, scores_csv: Path, seed: int, out_dir: Path) -> dict:
    """Filter BOTH train and test to faithful rows (row_score==1.0), train+eval; also report
    the row_score>=0.5 variant. A large jump over the full-data 0.277 means the ceiling is
    label noise, not the task."""
    cfg = eng.Config(seed=seed)
    eng.set_seed(seed)
    eng.log_event(f"[V4 seed={seed}] device={cfg.device}")
    df = _load_corpus_with_row_ids(corpus_path)
    train_df, calib_df, test_df, train_row_ids, test_row_ids, aspects = _mode_a_split(df, cfg, seed)
    eng.log_event(f"[V4 seed={seed}] base split train={len(train_df)} calib={len(calib_df)} test={len(test_df)}")

    scores = pd.read_csv(scores_csv).set_index("row_id")["row_score"]

    def _mask(row_ids, thresh_eq1: bool):
        vals = np.array([float(scores.loc[int(r)]) if int(r) in scores.index else np.nan for r in row_ids])
        if thresh_eq1:
            return np.isfinite(vals) & np.isclose(vals, 1.0)
        return np.isfinite(vals) & (vals >= 0.5)

    out_dir.mkdir(parents=True, exist_ok=True)
    variants = {}
    for label, eq1 in (("row_score_eq_1p0", True), ("row_score_ge_0p5", False)):
        tr_mask = _mask(train_row_ids, eq1)
        te_mask = _mask(test_row_ids, eq1)
        # calib is filtered too so thresholds are calibrated on clean rows (matches the regime)
        ca_ids = calib_df["row_id"].to_numpy()
        ca_mask = _mask(ca_ids, eq1)
        tr = train_df[tr_mask].reset_index(drop=True)
        ca = calib_df[ca_mask].reset_index(drop=True)
        te = test_df[te_mask].reset_index(drop=True)
        eng.log_event(f"[V4 seed={seed}][{label}] filtered train={len(tr)} calib={len(ca)} test={len(te)}")
        det_true, det_preds, det_probs, thresholds, full_metrics = _train_eval_micro_f1(tr, ca, te, aspects, cfg)
        eng.log_event(f"[V4 seed={seed}][{label}] clean micro_f1={full_metrics['micro_f1']:.4f}")
        # per-row predictions
        te_ids = te["row_id"].to_numpy()
        with (out_dir / f"test_predictions_{label}.jsonl").open("w", encoding="utf-8") as f:
            for i in range(len(te)):
                f.write(json.dumps({
                    "row_id": int(te_ids[i]),
                    "text": str(te.iloc[i]["text"]),
                    "gold_aspects": list(te.iloc[i]["aspects"].keys()),
                    "predicted_aspects": [aspects[j] for j in range(len(aspects)) if int(det_preds[i, j]) == 1],
                }, ensure_ascii=False) + "\n")
        variants[label] = {
            "n_train": int(len(tr)), "n_calib": int(len(ca)), "n_test": int(len(te)),
            "metrics": full_metrics,
            "micro_f1": full_metrics["micro_f1"],
            "jump_over_full_0p277": round(full_metrics["micro_f1"] - PAPER_BERT_F1_SINGLE, 4),
            "thresholds": {a: float(thresholds[a]) for a in aspects},
        }

    result = {
        "experiment": "V4_clean_label_ceiling",
        "seed": seed,
        "n_aspects": len(aspects),
        "full_data_reference_micro_f1": PAPER_BERT_F1_SINGLE,
        "variants": variants,
        "interpretation": "large jump over 0.277 => ceiling is label noise, not the task",
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


# ------------------------------------------------------------------ V3 LEARNING CURVE
def run_experiment_v3(corpus_path: Path, seed: int, out_dir: Path) -> dict:
    """Train on seeded subsamples of the train split [250,500,1000,2000,4000,8000], eval the
    SAME full test split each time. Signal scales with data; noise plateaus."""
    cfg = eng.Config(seed=seed)
    eng.set_seed(seed)
    eng.log_event(f"[V3 seed={seed}] device={cfg.device}")
    df = _load_corpus_with_row_ids(corpus_path)
    train_df, calib_df, test_df, train_row_ids, test_row_ids, aspects = _mode_a_split(df, cfg, seed)
    eng.log_event(f"[V3 seed={seed}] base split train={len(train_df)} calib={len(calib_df)} test={len(test_df)}")

    sizes = [250, 500, 1000, 2000, 4000, 8000]
    rng = np.random.default_rng(seed + 2000)
    sub_perm = rng.permutation(len(train_df))  # one nested ordering; size k = first k indices
    out_dir.mkdir(parents=True, exist_ok=True)
    points = []
    for k in sizes:
        kk = min(k, len(train_df))
        sub_idx = sub_perm[:kk]
        sub_train = train_df.iloc[sub_idx].reset_index(drop=True)
        eng.log_event(f"[V3 seed={seed}] size={kk} training")
        det_true, det_preds, det_probs, thresholds, full_metrics = _train_eval_micro_f1(
            sub_train, calib_df, test_df, aspects, cfg)
        eng.log_event(f"[V3 seed={seed}] size={kk} micro_f1={full_metrics['micro_f1']:.4f}")
        points.append({"train_size": kk, "metrics": full_metrics, "micro_f1": full_metrics["micro_f1"]})

    result = {
        "experiment": "V3_learning_curve",
        "seed": seed,
        "n_aspects": len(aspects),
        "n_test": int(len(test_df)),
        "sizes": sizes,
        "subsample_seed": seed + 2000,
        "points": points,
        "full_data_reference_micro_f1": PAPER_BERT_F1_SINGLE,
        "interpretation": "rising curve => signal scales with data; flat curve => noise plateau",
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


# ------------------------------------------------------------------ B5 SYNTH->REAL FINE-TUNE
OVERLAP_9 = ["accessibility", "assessment_design", "exam_fairness", "grading_transparency",
             "lecturer_quality", "materials", "organization", "overall_experience", "workload"]


def _restrict_aspects(df: pd.DataFrame, keep: list) -> pd.DataFrame:
    keepset = set(keep)
    out = df.copy()
    out["aspects"] = df["aspects"].apply(lambda d: {a: s for a, s in d.items() if a in keepset})
    out["target_attributes"] = out["aspects"]
    # keep all rows even if they lose every overlap aspect (multi-label; row may have 0 of the 9)
    return out.reset_index(drop=True)


def run_experiment_b5(corpus_path: Path, herath_mapped_jsonl: Path, seed: int, out_dir: Path,
                      real_train_n: int = None, arm: str = "synth_pretrain_finetune") -> dict:
    """9-aspect real-test detection micro-F1 vs number of real fine-tuning examples.

    arm="synth_pretrain_finetune" (default, original B5): (1) Pretrain a 9-aspect detection
    head on the synthetic train split. (2) Fine-tune the SAME head on the real-Herath train
    split. (3) Eval on real-Herath test.
    arm="real_only": skip the synthetic pretrain entirely and train a 9-aspect detection head
    FROM SCRATCH on the real-Herath train split (same N-subsampled train / calib), then eval on
    the SAME real-Herath test split. This is the per-N real-only learning curve (RC9).
    The real train/calib/test split, the N-subsample, threshold calibration, and evaluation are
    IDENTICAL across arms, so the two curves are directly comparable point-for-point.
    References: synthetic-only transfer 0.4593, real-only training (mode B, full data) 0.7673."""
    from torch.utils.data import DataLoader
    cfg = eng.Config(seed=seed)
    eng.set_seed(seed)
    eng.log_event(f"[B5 seed={seed}] device={cfg.device}")
    aspects = list(OVERLAP_9)  # fixed 9-aspect head end-to-end

    # ---- synthetic train split (mode-A split, restricted to the 9 overlap aspects)
    syn = _load_corpus_with_row_ids(corpus_path)
    syn_train, syn_calib, syn_test, _, _, _ = _mode_a_split(syn, cfg, seed)
    syn_train9 = _restrict_aspects(syn_train, aspects)
    syn_calib9 = _restrict_aspects(syn_calib, aspects)
    eng.log_event(f"[B5 seed={seed}] synth train={len(syn_train9)} calib={len(syn_calib9)} (9-aspect)")

    # ---- real-Herath split (identical to mode B)
    rows = []
    with herath_mapped_jsonl.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            text = str(r.get("text", "")).strip()
            amap = r.get("aspects", {})
            if not text or not amap:
                continue
            rows.append({"text": text, "aspects": amap, "target_attributes": amap})
    real_df = pd.DataFrame(rows).reset_index(drop=True)
    n = len(real_df)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_test = int(round(n * 0.20))
    test_idx = np.sort(perm[:n_test])
    rest_idx = perm[n_test:]
    n_calib = max(1, int(round(len(rest_idx) * 0.125)))
    calib_idx = np.sort(rest_idx[:n_calib])
    train_idx = np.sort(rest_idx[n_calib:])
    # RC5 (min-N fine-tuning curve): optionally subsample the real-train set to N rows,
    # seeded and deterministic, leaving calib/test untouched. None => full train (original B5).
    if real_train_n and int(real_train_n) < len(train_idx):
        train_idx = np.sort(np.random.default_rng(seed + 5000).choice(
            train_idx, size=int(real_train_n), replace=False))
        eng.log_event(f"[B5 seed={seed}] RC5 subsample real train -> n={len(train_idx)} (requested {real_train_n})")
    real_train = _restrict_aspects(real_df.iloc[train_idx].reset_index(drop=True), aspects)
    real_calib = _restrict_aspects(real_df.iloc[calib_idx].reset_index(drop=True), aspects)
    real_test = _restrict_aspects(real_df.iloc[test_idx].reset_index(drop=True), aspects)
    eng.log_event(f"[B5 seed={seed}] real train={len(real_train)} calib={len(real_calib)} test={len(real_test)} (9-aspect)")

    if arm == "real_only":
        # ---- real-only: train a 9-aspect detection head FROM SCRATCH on real-Herath train.
        eng.log_event(f"[B5 seed={seed}] arm=real_only: train from scratch on N={len(real_train)} real reviews")
        eng.set_seed(seed)
        det_model, det_tok = eng.train_detection("bert-base-uncased", real_train, real_calib, aspects, cfg)
    else:
        # ---- (1) pretrain 9-aspect detection head on synthetic
        eng.log_event(f"[B5 seed={seed}] STAGE 1: pretrain on synthetic (9-aspect)")
        det_model, det_tok = eng.train_detection("bert-base-uncased", syn_train9, syn_calib9, aspects, cfg)

        # ---- (2) fine-tune the SAME head on real-Herath train (continue training the same module)
        eng.log_event(f"[B5 seed={seed}] STAGE 2: fine-tune on real-Herath (same head)")
        det_model = _finetune_detection(det_model, det_tok, real_train, real_calib, aspects, cfg)

    # ---- (3) evaluate on real-Herath test, calibrate thresholds on real calib
    thresholds = eng.calibrate_thresholds(det_model, real_calib, det_tok, aspects, cfg)
    det_loader = DataLoader(eng.DetectionDataset(real_test, det_tok, aspects, cfg.max_len),
                            batch_size=cfg.batch_size, shuffle=False)
    det_probs, det_true = eng.collect_detection(det_model, det_loader, cfg.device)
    thr_vec = np.array([thresholds[a] for a in aspects], dtype=np.float32)
    det_preds = (det_probs >= thr_vec).astype(int)
    full_metrics = micro_f1_from_arrays(det_true, det_preds)
    full_metrics["n_rows"] = int(len(real_test))
    eng.log_event(f"[B5 seed={seed}] synth->real real-test micro_f1={full_metrics['micro_f1']:.4f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    real_ids = real_df.iloc[test_idx].index.to_numpy()
    with (out_dir / "test_predictions.jsonl").open("w", encoding="utf-8") as f:
        for i in range(len(real_test)):
            f.write(json.dumps({
                "real_test_pos": int(test_idx[i]),
                "text": str(real_test.iloc[i]["text"]),
                "gold_aspects": list(real_test.iloc[i]["aspects"].keys()),
                "predicted_aspects": [aspects[j] for j in range(len(aspects)) if int(det_preds[i, j]) == 1],
            }, ensure_ascii=False) + "\n")
    (out_dir / "split_indices.json").write_text(json.dumps({
        "seed": seed, "n_real_total": n,
        "real_train_idx": train_idx.tolist(), "real_calib_idx": calib_idx.tolist(),
        "real_test_idx": test_idx.tolist(),
    }), encoding="utf-8")

    result = {
        "experiment": "B5_synth_to_real_finetune",
        "arm": arm,
        "seed": seed,
        "n_aspects": len(aspects),
        "aspects": aspects,
        "n_syn_train": (0 if arm == "real_only" else int(len(syn_train9))), "n_real_train": int(len(real_train)),
        "n_real_calib": int(len(real_calib)), "n_real_test": int(len(real_test)),
        "synth_to_real_real_test": full_metrics,
        "synth_to_real_micro_f1": full_metrics["micro_f1"],
        "reference_synthetic_only_transfer_micro_f1": 0.4593,
        "reference_real_only_training_micro_f1": 0.7673,
        "delta_over_real_only": round(full_metrics["micro_f1"] - 0.7673, 4),
        "delta_over_synth_only": round(full_metrics["micro_f1"] - 0.4593, 4),
        "thresholds": {a: float(thresholds[a]) for a in aspects},
        "interpretation": "synth->real >= real-only => transfer helps the small ~2k real set",
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def _finetune_detection(model, tokenizer, train_df, val_df, aspects, cfg):
    """Continue training an existing DetectionModel on a new (real) train set. Same recipe as
    eng.train_detection (BCEWithLogits + pos_weight, AdamW lr, epochs, patience, best-on-val-macro-F1)
    but starting from the pretrained weights instead of fresh init."""
    import torch
    import torch.nn as nn
    import time as _time
    from torch.optim import AdamW
    from torch.utils.data import DataLoader
    with eng.gpu_training_lock(cfg.device, "bert-base-uncased:detection-finetune"):
        model = model.to(cfg.device)
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=eng.compute_pos_weight(train_df, tokenizer, aspects, cfg.max_len, cfg.device))
        optimizer = AdamW(model.parameters(), lr=cfg.lr)
        train_loader = DataLoader(eng.DetectionDataset(train_df, tokenizer, aspects, cfg.max_len),
                                  batch_size=cfg.batch_size, shuffle=True)
        val_loader = DataLoader(eng.DetectionDataset(val_df, tokenizer, aspects, cfg.max_len),
                                batch_size=cfg.batch_size, shuffle=False)
        best_score, best_state, patience = -1.0, None, 0
        eng.log_event(f"[finetune] start: train_rows={len(train_df)} val_rows={len(val_df)} "
                      f"epochs={cfg.epochs_detection} from pretrained head")
        for epoch in range(cfg.epochs_detection):
            model.train()
            t0 = _time.time()
            losses = []
            for batch in train_loader:
                optimizer.zero_grad()
                logits = model(batch["input_ids"].to(cfg.device), batch["attention_mask"].to(cfg.device))
                loss = criterion(logits, batch["labels"].to(cfg.device))
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu().item()))
            val_probs, val_true = eng.collect_detection(model, val_loader, cfg.device)
            em = eng.detection_epoch_metrics(val_probs, val_true)
            score = em["macro_f1"]
            eng.log_event(f"[finetune] epoch {epoch+1}/{cfg.epochs_detection}: "
                          f"train_loss={np.mean(losses):.4f} val_micro_f1={em['micro_f1']:.4f} "
                          f"val_macro_f1={em['macro_f1']:.4f} patience={patience} secs={_time.time()-t0:.1f}")
            if score > best_score:
                best_score = score
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= cfg.patience:
                    eng.log_event(f"[finetune] early stop at epoch {epoch+1}")
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        eng.log_event(f"[finetune] complete: best_val_macro_f1={best_score:.4f}")
        return model


# ------------------------------------------------------------------ Experiment B
def run_experiment_b(herath_mapped_jsonl: Path, seed: int, out_dir: Path) -> dict:
    cfg = eng.Config(seed=seed)
    eng.set_seed(seed)
    eng.log_event(f"[B seed={seed}] device={cfg.device}")

    # Load the 2,829 mapped Herath reviews (9-aspect overlap), already in {text, aspects} schema
    rows = []
    with herath_mapped_jsonl.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            text = str(r.get("text", "")).strip()
            aspects_map = r.get("aspects", {})
            if not text or not aspects_map:
                continue
            rows.append({"text": text, "aspects": aspects_map, "target_attributes": aspects_map})
    real_df = pd.DataFrame(rows).reset_index(drop=True)
    n = len(real_df)
    eng.log_event(f"[B seed={seed}] loaded mapped Herath rows={n}")

    aspects = sorted({a for d in real_df["aspects"] for a in d.keys()})
    eng.log_event(f"[B seed={seed}] overlap aspects ({len(aspects)})={aspects}")

    # 80/20 real-train / real-test split, then carve a calibration slice from train
    # (the engine needs a validation set for early-stop + threshold calibration).
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_test = int(round(n * 0.20))
    test_idx = np.sort(perm[:n_test])
    rest_idx = perm[n_test:]
    # 1/8 of the rest -> calibration (~10% of full), rest -> train
    n_calib = max(1, int(round(len(rest_idx) * 0.125)))
    calib_idx = np.sort(rest_idx[:n_calib])
    train_idx = np.sort(rest_idx[n_calib:])

    train_df = real_df.iloc[train_idx].reset_index(drop=True)
    calib_df = real_df.iloc[calib_idx].reset_index(drop=True)
    test_df = real_df.iloc[test_idx].reset_index(drop=True)
    eng.log_event(f"[B seed={seed}] split real-train={len(train_df)} calib={len(calib_df)} real-test={len(test_df)}")

    det_model, det_tok = eng.train_detection("bert-base-uncased", train_df, calib_df, aspects, cfg)
    sent_model, sent_tok = eng.train_sentiment("bert-base-uncased", train_df, calib_df, aspects, cfg)
    thresholds = eng.calibrate_thresholds(det_model, calib_df, det_tok, aspects, cfg)

    per_aspect_df, summary, artifact = eng.evaluate_models(
        "bert-base-uncased", det_model, sent_model, test_df, det_tok, sent_tok,
        aspects, thresholds, cfg, return_artifacts=True,
    )
    eng.log_event(f"[B seed={seed}] real-test micro_f1={summary['micro_f1']:.4f} macro_f1={summary['macro_f1']:.4f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    per_aspect_df.to_csv(out_dir / "per_aspect.csv", index=False)
    with (out_dir / "test_predictions.jsonl").open("w", encoding="utf-8") as f:
        for rec in artifact["sample_predictions"]:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    # save split indices for reproducibility
    (out_dir / "split_indices.json").write_text(json.dumps({
        "seed": seed, "n_total": n,
        "train_idx": train_idx.tolist(), "calib_idx": calib_idx.tolist(), "test_idx": test_idx.tolist(),
    }), encoding="utf-8")

    result = {
        "experiment": "B_real_herath_trained_baseline",
        "seed": seed,
        "n_total_mapped": n,
        "n_overlap_aspects": len(aspects),
        "aspects": aspects,
        "n_train": len(train_df), "n_calib": len(calib_df), "n_test": len(test_df),
        "thresholds": {a: float(thresholds[a]) for a in aspects},
        "synthetic_only_reference_micro_f1": 0.4593,
        "real_test_summary": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                              for k, v in summary.items() if k != "approach"},
        "real_test_micro_f1": float(summary["micro_f1"]),
        "real_test_macro_f1": float(summary["macro_f1"]),
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", required=True, choices=["A", "B", "V1", "V3", "V4", "B5"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", required=True, help="output dir")
    p.add_argument("--corpus-path", default="/app/data/generated_reviews_10k.jsonl")
    p.add_argument("--scores-csv", default="/app/data/at_scale_per_row_scores.csv")
    p.add_argument("--herath-mapped-jsonl", default="/app/data/herath_mapped_real_reviews_2829.jsonl")
    p.add_argument("--real-train-n", type=int, default=None, help="RC5: subsample real-train to N rows (B5 only)")
    p.add_argument("--arm", default="synth_pretrain_finetune",
                   choices=["synth_pretrain_finetune", "real_only"],
                   help="B5/RC9 arm: synth-pretrain+finetune (default) or real-only from scratch")
    args = p.parse_args()

    eng.configure_console_encoding()
    eng.ensure_dirs()  # creates benchmark_outputs/ so the gpu_training_lock path exists
    out_dir = Path(args.out)
    t0 = time.time()
    if args.experiment == "A":
        res = run_experiment_a(Path(args.corpus_path), Path(args.scores_csv), args.seed, out_dir)
    elif args.experiment == "B":
        res = run_experiment_b(Path(args.herath_mapped_jsonl), args.seed, out_dir)
    elif args.experiment == "V1":
        res = run_experiment_v1(Path(args.corpus_path), args.seed, out_dir)
    elif args.experiment == "V3":
        res = run_experiment_v3(Path(args.corpus_path), args.seed, out_dir)
    elif args.experiment == "V4":
        res = run_experiment_v4(Path(args.corpus_path), Path(args.scores_csv), args.seed, out_dir)
    elif args.experiment == "B5":
        res = run_experiment_b5(Path(args.corpus_path), Path(args.herath_mapped_jsonl), args.seed, out_dir,
                                real_train_n=args.real_train_n, arm=args.arm)
    else:
        raise ValueError(args.experiment)
    res["elapsed_seconds"] = round(time.time() - t0, 1)
    (out_dir / "result.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print("RESULT_JSON_BEGIN")
    print(json.dumps(res, indent=2, ensure_ascii=False))
    print("RESULT_JSON_END")


if __name__ == "__main__":
    main()
