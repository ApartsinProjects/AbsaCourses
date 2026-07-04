"""Exp3b - aspect-query detection, mitigated (revives the Exp3 null).

Three fixes over Exp3, whose loss collapsed to the class prior:
  1. NLI-pretrained encoder (bert-base-uncased-MNLI) so the pairwise entailment
     representation is not learned from scratch (fresh 2-class head only).
  2. Class-imbalance-aware weighted CrossEntropy.
  3. More epochs (6) + neg_per_pos=2.
Same targets/eval as Exp3 for a matched comparison. Detection-focused.

Usage: python exp3b_aspect_query_nli.py [--smoke] [--model textattack/bert-base-uncased-MNLI]
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from absa_model_comparison import Config, gpu_training_lock, set_seed, three_way_split
from evaluate_synthetic_to_real_transfer import (
    DEFAULT_HERATH_ROOT,
    DEFAULT_SYNTHETIC_PATH,
    load_herath_mapped_dataset,
    load_jsonl,
    load_real_from_mapped_jsonl,
    restrict_to_overlap,
)
from exp3_aspect_query import PHRASE, PairDS, make_pairs, score_reviews, calibrate

ROOT = Path(__file__).resolve().parents[1]
OATS = ROOT / "external_data/OATS_coursera/oats_mapped.jsonl"
MABSA = ROOT / "external_data/M-ABSA_coursera/m_absa_mapped.jsonl"
OUT = ROOT / "paper/outputs/exp3b_aspect_query_nli.json"


def train(model_name, train_pairs, cfg, epochs, pos_weight=2.0):
    with gpu_training_lock(cfg.device, f"{model_name}:aspect_query_nli"):
        tok = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2, ignore_mismatched_sizes=True).to(cfg.device)
        opt = AdamW(model.parameters(), lr=cfg.lr)
        w = torch.tensor([1.0, float(pos_weight)], device=cfg.device)
        crit = torch.nn.CrossEntropyLoss(weight=w)
        loader = DataLoader(PairDS(train_pairs, tok, cfg.max_len), batch_size=cfg.batch_size, shuffle=True)
        for ep in range(epochs):
            model.train()
            tot = 0.0
            for b in loader:
                opt.zero_grad()
                logits = model(b["input_ids"].to(cfg.device), attention_mask=b["attention_mask"].to(cfg.device)).logits
                loss = crit(logits, b["label"].to(cfg.device))
                loss.backward()
                opt.step()
                tot += float(loss)
            print(f"[exp3b] {model_name} epoch {ep+1}/{epochs} train_loss={tot/max(len(loader),1):.4f}", flush=True)
    return model, tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="textattack/bert-base-uncased-MNLI")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = Config()
    epochs = 1 if args.smoke else args.epochs
    set_seed(cfg.seed)

    synth = load_jsonl(DEFAULT_SYNTHETIC_PATH)
    if args.smoke:
        synth = synth.sample(300, random_state=42).reset_index(drop=True)
    targets = {"herath": load_herath_mapped_dataset(DEFAULT_HERATH_ROOT),
               "m_absa": load_real_from_mapped_jsonl(MABSA)}
    if args.smoke:
        targets = {k: v.sample(min(120, len(v)), random_state=1).reset_index(drop=True) for k, v in targets.items()}

    results = {"model": args.model, "epochs": epochs}
    for name, real in targets.items():
        aspects = sorted({a for labs in real["aspects"] for a in labs.keys()})
        pool = restrict_to_overlap(synth, aspects)
        tr, cal, _ = three_way_split(pool, cfg.split_calib, cfg.split_test, cfg.seed)
        train_pairs = make_pairs(tr, aspects, neg_per_pos=2, seed=cfg.seed)
        print(f"[exp3b] target={name} aspects={len(aspects)} train_pairs={len(train_pairs)} real={len(real)}", flush=True)
        model, tok = train(args.model, train_pairs, cfg, epochs)
        thr = calibrate(model, tok, cal, aspects, cfg)
        probs = score_reviews(model, tok, real["text"].tolist(), aspects, cfg)
        pred = (probs >= np.array([thr[a] for a in aspects])).astype(int)
        gold = np.array([[1 if a in r else 0 for a in aspects] for r in real["aspects"]])
        results[name] = {
            "micro_precision": round(float(precision_score(gold.ravel(), pred.ravel(), zero_division=0)), 4),
            "micro_recall": round(float(recall_score(gold.ravel(), pred.ravel(), zero_division=0)), 4),
            "micro_f1": round(float(f1_score(gold.ravel(), pred.ravel(), zero_division=0)), 4),
            "n_real": int(len(real))}
        print(f"[exp3b] {name}: NLI aspect-query microF1={results[name]['micro_f1']} "
              f"(Exp3 bert-scratch was herath 0.288 / m_absa 0.188)", flush=True)
        del model, tok
        gc.collect(); torch.cuda.empty_cache()

    print(json.dumps(results, indent=2), flush=True)
    if not args.smoke:
        OUT.write_text(json.dumps(results, indent=2))
        print(f"[exp3b] wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
