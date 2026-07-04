"""Experiment 4 (part B) - does correlated-sentiment training improve the
sentiment head? Train the sentiment model on the correlated-disposition corpus
(exp4_generate.py) vs an EQUAL-N subsample of the original independent-polarity
corpus, and measure sentiment MSE on gold-present aspects of real reviews
(detection-independent). Equal N isolates sentiment structure from corpus size.

Usage: python exp4_train.py [--smoke]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader

from absa_model_comparison import (
    Config,
    SentimentDataset,
    collect_sentiment,
    masked_mse_numpy,
    set_seed,
    three_way_split,
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
CORR = ROOT / "paper/outputs/exp4_correlated_gen.jsonl"
OUT = ROOT / "paper/outputs/exp4_correlated_train.json"


def load_corr() -> pd.DataFrame:
    rows = []
    for l in CORR.open(encoding="utf-8"):
        r = json.loads(l)
        if r.get("text") and r.get("aspects"):
            rows.append({"text": r["text"], "aspects": r["aspects"], "target_attributes": r["aspects"],
                         "nuance_attributes": {}, "course_name": "", "grade": "", "style": ""})
    return pd.DataFrame(rows)


def gold_sentiment_mse(sent_model, tok, real_df, aspects, cfg) -> float:
    loader = DataLoader(SentimentDataset(real_df, tok, aspects, cfg.max_len), batch_size=cfg.batch_size, shuffle=False)
    preds, tgt, mask = collect_sentiment(sent_model, loader, cfg.device)
    return float(masked_mse_numpy(preds, tgt, mask))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--approach", default="bert-base-uncased")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = Config()
    if args.smoke:
        cfg.epochs_sentiment = 1
    set_seed(cfg.seed)

    corr = load_corr()
    orig_full = load_jsonl(DEFAULT_SYNTHETIC_PATH)
    n = min(len(corr), len(orig_full))
    orig = orig_full.sample(n, random_state=cfg.seed).reset_index(drop=True)
    corr = corr.sample(n, random_state=cfg.seed).reset_index(drop=True)  # equal N
    print(f"[exp4train] equal-N={n} (correlated vs original subsample)", flush=True)

    targets = {"herath": load_herath_mapped_dataset(DEFAULT_HERATH_ROOT),
               "oats": load_real_from_mapped_jsonl(OATS)}
    if args.smoke:
        n = 250
        orig = orig.sample(n, random_state=1).reset_index(drop=True)
        corr = corr.sample(min(n, len(corr)), random_state=1).reset_index(drop=True)
        targets = {k: v.sample(min(120, len(v)), random_state=1).reset_index(drop=True) for k, v in targets.items()}

    results = {}
    for name, real in targets.items():
        aspects = sorted({a for labs in real["aspects"] for a in labs.keys()})
        row = {"n_real": int(len(real)), "n_aspects": len(aspects)}
        for src_name, src in [("correlated", corr), ("original", orig)]:
            pool = restrict_to_overlap(src, aspects)
            tr, cal, _ = three_way_split(pool, cfg.split_calib, cfg.split_test, cfg.seed)
            model, tok = train_sentiment(args.approach, tr, cal, aspects, cfg)
            mse = gold_sentiment_mse(model, tok, real, aspects, cfg)
            row[f"{src_name}_gold_sent_mse"] = round(mse, 4)
            row[f"{src_name}_n_train"] = int(len(tr))
            print(f"[exp4train] {name}/{src_name}: gold-present sentiment MSE={mse:.4f} (train={len(tr)})", flush=True)
            import gc, torch
            del model, tok
            gc.collect(); torch.cuda.empty_cache()
        row["delta_mse"] = round(row["correlated_gold_sent_mse"] - row["original_gold_sent_mse"], 4)
        results[name] = row

    print(json.dumps(results, indent=2), flush=True)
    if not args.smoke:
        OUT.write_text(json.dumps(results, indent=2))
        print(f"[exp4train] wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
