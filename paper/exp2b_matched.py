"""Exp2b - construct-matched baseline for the sentence-level training win.

Trains BOTH arms in ONE pass per target, same config/seed: (A) review-level
(original full reviews) and (B) sentence-level (lexicon-localized), evaluated on
the same real target. Gives a co-computed, construct-matched review-vs-sentence
delta so the Exp2 win is auditable number-by-number.

Usage: python exp2b_matched.py [--smoke]
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import pandas as pd
import torch

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
    DEFAULT_HERATH_ROOT,
    DEFAULT_SYNTHETIC_PATH,
    load_herath_mapped_dataset,
    load_jsonl,
    load_real_from_mapped_jsonl,
    restrict_to_overlap,
)
from exp2_sentence_train import build_sentence_df

ROOT = Path(__file__).resolve().parents[1]
OATS = ROOT / "external_data/OATS_coursera/oats_mapped.jsonl"
MABSA = ROOT / "external_data/M-ABSA_coursera/m_absa_mapped.jsonl"
OUT = ROOT / "paper/outputs/exp2b_matched.json"


def train_eval(approach, train_pool, real, aspects, cfg):
    tr, cal, _ = three_way_split(train_pool, cfg.split_calib, cfg.split_test, cfg.seed)
    det, dtok = train_detection(approach, tr, cal, aspects, cfg)
    sen, stok = train_sentiment(approach, tr, cal, aspects, cfg)
    thr = calibrate_thresholds(det, cal, dtok, aspects, cfg)
    _, summ = evaluate_models(approach, det, sen, real, dtok, stok, aspects, thr, cfg)
    out = {"micro_f1": round(summ["micro_f1"], 4), "sent_mse": round(summ["sentiment_mse_detected"], 4), "n_train": int(len(tr))}
    del det, sen, dtok, stok
    gc.collect(); torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--approach", default="bert-base-uncased")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = Config()
    if args.smoke:
        cfg.epochs_detection = cfg.epochs_sentiment = 1
    set_seed(cfg.seed)

    synth = load_jsonl(DEFAULT_SYNTHETIC_PATH)
    if args.smoke:
        synth = synth.sample(300, random_state=42).reset_index(drop=True)
    sent_synth = build_sentence_df(synth)

    targets = {"herath": load_herath_mapped_dataset(DEFAULT_HERATH_ROOT),
               "m_absa": load_real_from_mapped_jsonl(MABSA),
               "oats": load_real_from_mapped_jsonl(OATS)}
    if args.smoke:
        targets = {k: v.sample(min(120, len(v)), random_state=1).reset_index(drop=True) for k, v in targets.items()}

    results = {}
    for name, real in targets.items():
        aspects = sorted({a for labs in real["aspects"] for a in labs.keys()})
        review_pool = restrict_to_overlap(synth, aspects)
        sent_pool = restrict_to_overlap(sent_synth, aspects)
        print(f"[exp2b] {name}: aspects={len(aspects)} review_pool={len(review_pool)} sent_pool={len(sent_pool)} real={len(real)}", flush=True)
        review = train_eval(args.approach, review_pool, real, aspects, cfg)
        sentence = train_eval(args.approach, sent_pool, real, aspects, cfg)
        results[name] = {"review_trained": review, "sentence_trained": sentence,
                         "delta_micro_f1": round(sentence["micro_f1"] - review["micro_f1"], 4),
                         "delta_sent_mse": round(sentence["sent_mse"] - review["sent_mse"], 4),
                         "n_real": int(len(real))}
        print(f"[exp2b] {name}: review F1={review['micro_f1']} sentence F1={sentence['micro_f1']} "
              f"delta={results[name]['delta_micro_f1']:+.4f}", flush=True)

    print(json.dumps(results, indent=2), flush=True)
    if not args.smoke:
        OUT.write_text(json.dumps(results, indent=2))
        print(f"[exp2b] wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
