"""Multi-seed synthetic-to-real transfer worker (runs inside the Modal container).

For one (approach, seed):
  1. Load the canonical synthetic corpus and restrict it to the Herath conservative
     9-aspect overlap (aspects derived from the CORRECT XMI-mapped real Herath data).
  2. three_way_split(synthetic_overlap, 0.10, 0.10, seed) -> train / calib / INTERNAL test.
  3. Train detection + sentiment on the synthetic train split (calibrate thresholds on
     calib). set_seed(seed) is called BEFORE EACH model's training (clean seeding, mirrors
     the validated paper/multiseed_validation.py discipline).
  4. Evaluate the SAME trained models on BOTH:
       - INTERNAL: the held-out synthetic overlap test split.
       - EXTERNAL: the full mapped real Herath dataset (2,829 rows, 9 aspects).
  5. Report micro-F1, macro-F1, micro-recall, sentiment MSE per (split).

Data-provenance: the real target is loaded with load_real_from_mapped_jsonl from the
CORRECT XMI-derived mapping (paper/real_transfer/herath_mapped_real_reviews.jsonl).
The WRONG paper/reviewer_ab_data/herath_mapped_real_reviews_2829.jsonl (distilbert
Herath micro-F1 ~0.18) is never touched.

Invocation (inside container, cwd=/app):
  python paper/multiseed_transfer_worker.py --approach bert-base-uncased --seed 42 --out /results/...
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, "/app/paper")

import absa_model_comparison as eng  # noqa: E402
from evaluate_synthetic_to_real_transfer import (  # noqa: E402
    load_real_from_mapped_jsonl,
    restrict_to_overlap,
)


def _pick_metrics(summary: dict) -> dict:
    """Extract the four reported metrics from an evaluate/summary dict."""
    return {
        "micro_f1": float(summary["micro_f1"]),
        "macro_f1": float(summary["macro_f1"]),
        "micro_recall": float(summary["micro_recall"]),
        "sentiment_mse": float(summary["sentiment_mse_detected"]),
    }


def run(approach: str, seed: int, synthetic_path: Path, herath_mapped_jsonl: Path, out_dir: Path) -> dict:
    cfg = eng.Config(
        max_len=192, batch_size=8, lr=3e-5,
        epochs_detection=3, epochs_sentiment=3, seed=seed,
    )
    eng.set_seed(seed)
    eng.log_event(f"[{approach} seed={seed}] device={cfg.device}")

    # ---- data (CORRECT XMI-derived Herath mapping) ----
    synthetic_df = eng.load_jsonl(synthetic_path)
    real_df = load_real_from_mapped_jsonl(herath_mapped_jsonl)
    aspects = sorted({a for labels in real_df["aspects"] for a in labels.keys()})
    eng.log_event(f"[{approach} seed={seed}] real_rows={len(real_df)} overlap_aspects({len(aspects)})={aspects}")

    synthetic_overlap = restrict_to_overlap(synthetic_df, aspects)
    train_df, calib_df, internal_test_df = eng.three_way_split(
        synthetic_overlap, cfg.split_calib, cfg.split_test, seed)
    eng.log_event(
        f"[{approach} seed={seed}] synth_overlap={len(synthetic_overlap)} "
        f"train={len(train_df)} calib={len(calib_df)} internal_test={len(internal_test_df)}")
    if calib_df.empty or internal_test_df.empty:
        raise ValueError("Empty calib/test split after overlap restriction.")

    metrics = {}
    if approach == "tfidf_two_step":
        # Classical fit is deterministic in train_df; evaluate the same fit on both splits
        # by calling the bundled fit+eval twice (identical underlying models/thresholds).
        eng.set_seed(seed)
        _, internal_summary = eng.run_tfidf_two_step_approach(
            approach, train_df, calib_df, internal_test_df, aspects)
        eng.set_seed(seed)
        _, external_summary = eng.run_tfidf_two_step_approach(
            approach, train_df, calib_df, real_df, aspects)
    else:
        eng.set_seed(seed)  # clean seed before detection head
        det_model, det_tok = eng.train_detection(approach, train_df, calib_df, aspects, cfg)
        eng.set_seed(seed)  # clean seed before sentiment head
        sent_model, sent_tok = eng.train_sentiment(approach, train_df, calib_df, aspects, cfg)
        thresholds = eng.calibrate_thresholds(det_model, calib_df, det_tok, aspects, cfg)

        _, internal_summary = eng.evaluate_models(
            approach, det_model, sent_model, internal_test_df,
            det_tok, sent_tok, aspects, thresholds, cfg)
        _, external_summary = eng.evaluate_models(
            approach, det_model, sent_model, real_df,
            det_tok, sent_tok, aspects, thresholds, cfg)

    metrics["internal_synthetic_test"] = _pick_metrics(internal_summary)
    metrics["external_real_herath"] = _pick_metrics(external_summary)

    eng.log_event(
        f"[{approach} seed={seed}] INTERNAL micro_f1={metrics['internal_synthetic_test']['micro_f1']:.4f} "
        f"| EXTERNAL(herath) micro_f1={metrics['external_real_herath']['micro_f1']:.4f} "
        f"sent_mse={metrics['external_real_herath']['sentiment_mse']:.4f}")

    result = {
        "experiment": "multiseed_synthetic_to_real_transfer",
        "approach": approach,
        "seed": seed,
        "n_overlap_aspects": len(aspects),
        "aspects": aspects,
        "n_synth_overlap": int(len(synthetic_overlap)),
        "n_train": int(len(train_df)),
        "n_calib": int(len(calib_df)),
        "n_internal_test": int(len(internal_test_df)),
        "n_real_herath": int(len(real_df)),
        "config": {"max_len": 192, "batch_size": 8, "lr": 3e-5,
                   "epochs_detection": 3, "epochs_sentiment": 3},
        "metrics": metrics,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--approach", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--synthetic-path", default="/app/data/synthetic_generated_reviews.jsonl")
    p.add_argument("--herath-mapped-jsonl", default="/app/data/herath_mapped_real_reviews.jsonl")
    args = p.parse_args()

    eng.configure_console_encoding()
    eng.ensure_dirs()  # creates benchmark_outputs/ for the gpu_training_lock path
    t0 = time.time()
    res = run(args.approach, args.seed, Path(args.synthetic_path),
              Path(args.herath_mapped_jsonl), Path(args.out))
    res["elapsed_seconds"] = round(time.time() - t0, 1)
    (Path(args.out) / "result.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print("RESULT_JSON_BEGIN")
    print(json.dumps(res, indent=2, ensure_ascii=False))
    print("RESULT_JSON_END")


if __name__ == "__main__":
    main()
