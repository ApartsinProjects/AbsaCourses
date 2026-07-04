"""Experiment 1 - Windowed (per-sentence) inference + OR-pool aggregation.

Motivation: an aspect is expressed in one or two sentences, but whole-document
BERT pools it over ~100 words, diluting a localized signal. Split each real
review into sentences, predict per sentence, then OR-pool to the review level
(aspect present if ANY sentence fires) and take the sentiment from the sentence
with the strongest detection probability. Compare whole-document vs windowed on
multi-sentence targets. No retraining: one trained model, two eval paths.

Usage:
  python exp1_windowed_transfer.py --target oats   [--smoke]
  python exp1_windowed_transfer.py --target herath
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

from absa_model_comparison import (
    Config,
    DetectionDataset,
    SentimentDataset,
    calibrate_thresholds,
    collect_detection,
    collect_sentiment,
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

ROOT = Path(__file__).resolve().parents[1]
OATS = ROOT / "external_data/OATS_coursera/oats_mapped.jsonl"
OUT = ROOT / "paper/outputs/exp1_windowed.json"
POL2NUM = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def sentence_rows(real_df: pd.DataFrame, min_w: int = 3) -> tuple[pd.DataFrame, list[list[int]]]:
    """Explode reviews into sentences. Return the sentence frame and, for each
    review, the list of sentence-row indices belonging to it."""
    sent_texts, groups = [], []
    for _, row in real_df.reset_index(drop=True).iterrows():
        parts = [s.strip() for s in SENT_SPLIT.split(" ".join(str(row["text"]).split())) if len(s.split()) >= min_w]
        if not parts:
            parts = [str(row["text"]).strip()]
        start = len(sent_texts)
        sent_texts.extend(parts)
        groups.append(list(range(start, start + len(parts))))
    sent_df = pd.DataFrame({"text": sent_texts, "aspects": [{} for _ in sent_texts]})
    return sent_df, groups


def micro_from_matrices(gold: np.ndarray, pred: np.ndarray) -> dict:
    return {
        "micro_precision": float(precision_score(gold.ravel(), pred.ravel(), zero_division=0)),
        "micro_recall": float(recall_score(gold.ravel(), pred.ravel(), zero_division=0)),
        "micro_f1": float(f1_score(gold.ravel(), pred.ravel(), zero_division=0)),
    }


def windowed_eval(det_model, sent_model, real_df, det_tok, sent_tok, aspects, thresholds, cfg) -> dict:
    sent_df, groups = sentence_rows(real_df)
    det_loader = DataLoader(DetectionDataset(sent_df, det_tok, aspects, cfg.max_len), batch_size=cfg.batch_size, shuffle=False)
    sent_loader = DataLoader(SentimentDataset(sent_df, sent_tok, aspects, cfg.max_len), batch_size=cfg.batch_size, shuffle=False)
    sdet_probs, _ = collect_detection(det_model, det_loader, cfg.device)          # [n_sent, n_aspect]
    ssent_preds, _, _ = collect_sentiment(sent_model, sent_loader, cfg.device)    # [n_sent, n_aspect]

    thr = np.array([thresholds[a] for a in aspects], dtype=np.float32)
    n_rev, n_asp = len(real_df), len(aspects)
    pooled_prob = np.zeros((n_rev, n_asp), dtype=np.float32)
    pooled_sent = np.zeros((n_rev, n_asp), dtype=np.float32)
    for r, idxs in enumerate(groups):
        block = sdet_probs[idxs]                       # [k_sent, n_aspect]
        pooled_prob[r] = block.max(axis=0)             # OR-pool: strongest sentence per aspect
        best = block.argmax(axis=0)                    # sentence carrying that aspect
        pooled_sent[r] = ssent_preds[np.array(idxs)][best, np.arange(n_asp)]
    pooled_pred = (pooled_prob >= thr).astype(int)

    gold = np.zeros((n_rev, n_asp), dtype=np.int64)
    gold_sent = np.zeros((n_rev, n_asp), dtype=np.float32)
    gold_mask = np.zeros((n_rev, n_asp), dtype=np.float32)
    for r, (_, row) in enumerate(real_df.reset_index(drop=True).iterrows()):
        for a_i, a in enumerate(aspects):
            if a in row["aspects"]:
                gold[r, a_i] = 1
                gold_sent[r, a_i] = POL2NUM.get(str(row["aspects"][a]).lower(), 0.0)
                gold_mask[r, a_i] = 1.0
    m = micro_from_matrices(gold, pooled_pred)
    eff = pooled_pred * gold_mask
    denom = eff.sum()
    m["sentiment_mse_detected"] = float((((pooled_sent - gold_sent) ** 2) * eff).sum() / denom) if denom else float("nan")
    m["n_sentences"] = int(len(sent_df))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["oats", "herath"], default="oats")
    ap.add_argument("--approach", default="bert-base-uncased")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    cfg = Config()
    if args.smoke:
        cfg.epochs_detection = 1
        cfg.epochs_sentiment = 1
    set_seed(cfg.seed)

    synth = load_jsonl(DEFAULT_SYNTHETIC_PATH)
    if args.target == "oats":
        real = load_real_from_mapped_jsonl(OATS)
    else:
        real = load_herath_mapped_dataset(DEFAULT_HERATH_ROOT)
    if args.smoke:
        synth = synth.sample(300, random_state=42).reset_index(drop=True)
        real = real.sample(min(120, len(real)), random_state=42).reset_index(drop=True)

    aspects = sorted({a for labs in real["aspects"] for a in labs.keys()})
    synth = restrict_to_overlap(synth, aspects)
    synth_train, synth_calib, _ = three_way_split(synth, cfg.split_calib, cfg.split_test, cfg.seed)
    print(f"[exp1] target={args.target} approach={args.approach} aspects={len(aspects)} "
          f"synth_train={len(synth_train)} real={len(real)} smoke={args.smoke}", flush=True)

    det_model, det_tok = train_detection(args.approach, synth_train, synth_calib, aspects, cfg)
    sent_model, sent_tok = train_sentiment(args.approach, synth_train, synth_calib, aspects, cfg)
    thresholds = calibrate_thresholds(det_model, synth_calib, det_tok, aspects, cfg)

    _, whole = evaluate_models(args.approach, det_model, sent_model, real, det_tok, sent_tok, aspects, thresholds, cfg)
    wins = windowed_eval(det_model, sent_model, real, det_tok, sent_tok, aspects, thresholds, cfg)

    result = {
        "target": args.target, "approach": args.approach, "smoke": args.smoke,
        "n_real_reviews": int(len(real)), "n_aspects": len(aspects),
        "whole_doc": {k: whole[k] for k in ("micro_precision", "micro_recall", "micro_f1", "sentiment_mse_detected")},
        "windowed": {k: wins[k] for k in ("micro_precision", "micro_recall", "micro_f1", "sentiment_mse_detected", "n_sentences")},
    }
    result["delta_micro_f1"] = round(wins["micro_f1"] - whole["micro_f1"], 4)
    print(json.dumps(result, indent=2), flush=True)
    if not args.smoke:
        prev = json.loads(OUT.read_text()) if OUT.exists() else {}
        prev[args.target] = result
        OUT.write_text(json.dumps(prev, indent=2))
        print(f"[exp1] wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
