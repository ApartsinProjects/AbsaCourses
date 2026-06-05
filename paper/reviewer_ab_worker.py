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
    p.add_argument("--experiment", required=True, choices=["A", "B"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", required=True, help="output dir")
    p.add_argument("--corpus-path", default="/app/data/generated_reviews_10k.jsonl")
    p.add_argument("--scores-csv", default="/app/data/at_scale_per_row_scores.csv")
    p.add_argument("--herath-mapped-jsonl", default="/app/data/herath_mapped_real_reviews_2829.jsonl")
    args = p.parse_args()

    eng.configure_console_encoding()
    eng.ensure_dirs()  # creates benchmark_outputs/ so the gpu_training_lock path exists
    out_dir = Path(args.out)
    t0 = time.time()
    if args.experiment == "A":
        res = run_experiment_a(Path(args.corpus_path), Path(args.scores_csv), args.seed, out_dir)
    else:
        res = run_experiment_b(Path(args.herath_mapped_jsonl), args.seed, out_dir)
    res["elapsed_seconds"] = round(time.time() - t0, 1)
    (out_dir / "result.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print("RESULT_JSON_BEGIN")
    print(json.dumps(res, indent=2, ensure_ascii=False))
    print("RESULT_JSON_END")


if __name__ == "__main__":
    main()
