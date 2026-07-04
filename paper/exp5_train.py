"""Experiment 5 (part B) - does matching real structure (co-occurrence + polarity)
improve transfer? Train detection+sentiment on the structured corpus vs an
EQUAL-N subsample of the original independent corpus, evaluate on OATS with both
whole-document and windowed (per-sentence OR-pool) inference. Equal N isolates
structure from corpus size.

Usage: python exp5_train.py [--smoke]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from absa_model_comparison import (
    Config,
    calibrate_thresholds,
    evaluate_models,
    set_seed,
    three_way_split,
    train_detection,
    train_sentiment,
)
from evaluate_synthetic_to_real_transfer import (
    DEFAULT_SYNTHETIC_PATH,
    load_jsonl,
    load_real_from_mapped_jsonl,
    restrict_to_overlap,
)
from exp1_windowed_transfer import windowed_eval

ROOT = Path(__file__).resolve().parents[1]
OATS = ROOT / "external_data/OATS_coursera/oats_mapped.jsonl"
STRUCT = ROOT / "paper/outputs/exp5_structured_gen.jsonl"
OUT = ROOT / "paper/outputs/exp5_structured_train.json"


def load_gen(path) -> pd.DataFrame:
    rows = []
    for l in open(path, encoding="utf-8"):
        r = json.loads(l)
        if r.get("text") and r.get("aspects"):
            rows.append({"text": r["text"], "aspects": r["aspects"], "target_attributes": r["aspects"],
                         "nuance_attributes": {}, "course_name": "", "grade": "", "style": ""})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--approach", default="bert-base-uncased")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = Config()
    if args.smoke:
        cfg.epochs_detection = cfg.epochs_sentiment = 1
    set_seed(cfg.seed)

    struct = load_gen(STRUCT)
    orig_full = load_jsonl(DEFAULT_SYNTHETIC_PATH)
    n = min(len(struct), len(orig_full))
    struct = struct.sample(n, random_state=cfg.seed).reset_index(drop=True)
    orig = orig_full.sample(n, random_state=cfg.seed).reset_index(drop=True)
    real = load_real_from_mapped_jsonl(OATS)
    if args.smoke:
        n = 300
        struct = struct.sample(min(n, len(struct)), random_state=1).reset_index(drop=True)
        orig = orig.sample(n, random_state=1).reset_index(drop=True)
        real = real.sample(120, random_state=1).reset_index(drop=True)
    aspects = sorted({a for labs in real["aspects"] for a in labs.keys()})
    print(f"[exp5train] equal-N={len(struct)} structured vs original; OATS aspects={len(aspects)} real={len(real)}", flush=True)

    results = {}
    for src_name, src in [("structured", struct), ("original", orig)]:
        pool = restrict_to_overlap(src, aspects)
        tr, cal, _ = three_way_split(pool, cfg.split_calib, cfg.split_test, cfg.seed)
        det, dtok = train_detection(args.approach, tr, cal, aspects, cfg)
        sen, stok = train_sentiment(args.approach, tr, cal, aspects, cfg)
        thr = calibrate_thresholds(det, cal, dtok, aspects, cfg)
        _, whole = evaluate_models(args.approach, det, sen, real, dtok, stok, aspects, thr, cfg)
        wins = windowed_eval(det, sen, real, dtok, stok, aspects, thr, cfg)
        results[src_name] = {
            "n_train": int(len(tr)),
            "whole_micro_f1": round(whole["micro_f1"], 4),
            "windowed_micro_f1": round(wins["micro_f1"], 4),
            "whole_sent_mse": round(whole["sentiment_mse_detected"], 4),
            "windowed_sent_mse": round(wins["sentiment_mse_detected"], 4),
        }
        print(f"[exp5train] {src_name}: whole F1={results[src_name]['whole_micro_f1']:.4f} "
              f"windowed F1={results[src_name]['windowed_micro_f1']:.4f} "
              f"whole sentMSE={results[src_name]['whole_sent_mse']:.4f}", flush=True)
        import gc, torch
        del det, sen, dtok, stok
        gc.collect(); torch.cuda.empty_cache()

    results["delta_whole_f1"] = round(results["structured"]["whole_micro_f1"] - results["original"]["whole_micro_f1"], 4)
    results["delta_windowed_f1"] = round(results["structured"]["windowed_micro_f1"] - results["original"]["windowed_micro_f1"], 4)
    results["delta_whole_sent_mse"] = round(results["structured"]["whole_sent_mse"] - results["original"]["whole_sent_mse"], 4)
    print(json.dumps(results, indent=2), flush=True)
    if not args.smoke:
        OUT.write_text(json.dumps(results, indent=2))
        print(f"[exp5train] wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
