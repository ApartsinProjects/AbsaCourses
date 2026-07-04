"""Experiment 3 - Aspect-query (NLI-style) detection.

Instead of a single [CLS] -> 20-way multi-label head (each aspect an opaque index),
represent each aspect as a natural-language phrase (multiple subword tokens) and
classify the pair (review [SEP] aspect phrase) -> present / absent. This lets the
model match aspects by MEANING, which should help cross-domain (MOOC) transfer:
a MOOC phrasing of "workload" still matches the "workload / time commitment" query
tokens even if surface vocabulary differs. Detection-focused (the aspect-query
hypothesis is about detection generalization); sentiment head unchanged elsewhere.

Usage: python exp3_aspect_query.py [--smoke]
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
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

ROOT = Path(__file__).resolve().parents[1]
OATS = ROOT / "external_data/OATS_coursera/oats_mapped.jsonl"
MABSA = ROOT / "external_data/M-ABSA_coursera/m_absa_mapped.jsonl"
OUT = ROOT / "paper/outputs/exp3_aspect_query.json"

# Natural-language query phrase per aspect (multiple subword tokens, semantic).
PHRASE = {
    "exam_fairness": "exam fairness", "accessibility": "accessibility and availability",
    "organization": "course organization and structure", "workload": "workload and time commitment",
    "overall_experience": "overall experience", "grading_transparency": "grading transparency",
    "pacing": "pacing and speed", "tooling_usability": "tools and platform usability",
    "lecturer_quality": "lecturer and teaching quality", "prerequisite_fit": "prerequisite fit and background",
    "support": "student support and office hours", "materials": "course materials and resources",
    "difficulty": "difficulty level", "clarity": "clarity of explanations",
    "assessment_design": "assignment and assessment design", "interest": "how interesting and engaging",
    "peer_interaction": "peer interaction and collaboration", "practical_application": "practical real world application",
    "feedback_quality": "quality of feedback", "relevance": "relevance and usefulness",
}


class PairDS(Dataset):
    def __init__(self, pairs, tokenizer, max_len):
        self.pairs, self.tok, self.max_len = pairs, tokenizer, max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        review, phrase, label = self.pairs[i]
        enc = self.tok(review, phrase, truncation=True, padding="max_length",
                       max_length=self.max_len, return_tensors="pt")
        return {"input_ids": enc["input_ids"][0], "attention_mask": enc["attention_mask"][0],
                "label": torch.tensor(int(label), dtype=torch.long)}


def make_pairs(df, aspects, neg_per_pos=3, seed=42):
    rng = random.Random(seed)
    pairs = []
    for _, r in df.iterrows():
        gold = set(r["aspects"].keys()) if isinstance(r["aspects"], dict) else set()
        pos = [a for a in aspects if a in gold]
        neg = [a for a in aspects if a not in gold]
        rng.shuffle(neg)
        chosen = pos + neg[: max(len(pos) * neg_per_pos, 2)]
        for a in chosen:
            pairs.append((str(r["text"]), PHRASE.get(a, a.replace("_", " ")), 1 if a in gold else 0))
    rng.shuffle(pairs)
    return pairs


@torch.no_grad()
def score_reviews(model, tok, reviews, aspects, cfg):
    """Return [n_review, n_aspect] present-probabilities."""
    pairs = [(str(t), PHRASE.get(a, a.replace("_", " ")), 0) for t in reviews for a in aspects]
    loader = DataLoader(PairDS(pairs, tok, cfg.max_len), batch_size=cfg.batch_size, shuffle=False)
    probs = []
    model.eval()
    for b in loader:
        logits = model(b["input_ids"].to(cfg.device), attention_mask=b["attention_mask"].to(cfg.device)).logits
        probs.append(torch.softmax(logits, -1)[:, 1].cpu().numpy())
    return np.concatenate(probs).reshape(len(reviews), len(aspects))


def train(model_name, train_pairs, val_pairs, cfg):
    with gpu_training_lock(cfg.device, f"{model_name}:aspect_query"):
        tok = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(cfg.device)
        opt = AdamW(model.parameters(), lr=cfg.lr)
        crit = torch.nn.CrossEntropyLoss()
        loader = DataLoader(PairDS(train_pairs, tok, cfg.max_len), batch_size=cfg.batch_size, shuffle=True)
        for ep in range(cfg.epochs_detection):
            model.train()
            tot = 0.0
            for b in loader:
                opt.zero_grad()
                logits = model(b["input_ids"].to(cfg.device), attention_mask=b["attention_mask"].to(cfg.device)).logits
                loss = crit(logits, b["label"].to(cfg.device))
                loss.backward()
                opt.step()
                tot += float(loss)
            print(f"[exp3] {model_name} epoch {ep+1}/{cfg.epochs_detection} train_loss={tot/max(len(loader),1):.4f}", flush=True)
    return model, tok


def calibrate(model, tok, val_df, aspects, cfg):
    reviews = val_df["text"].tolist()
    probs = score_reviews(model, tok, reviews, aspects, cfg)
    gold = np.array([[1 if a in r else 0 for a in aspects] for r in val_df["aspects"]])
    best = {}
    for j, a in enumerate(aspects):
        grid = np.linspace(0.1, 0.9, 17)
        f1s = [f1_score(gold[:, j], (probs[:, j] >= t).astype(int), zero_division=0) for t in grid]
        best[a] = float(grid[int(np.argmax(f1s))])
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--approach", default="bert-base-uncased")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = Config()
    if args.smoke:
        cfg.epochs_detection = 1
    set_seed(cfg.seed)

    synth = load_jsonl(DEFAULT_SYNTHETIC_PATH)
    if args.smoke:
        synth = synth.sample(300, random_state=42).reset_index(drop=True)
    targets = {"herath": load_herath_mapped_dataset(DEFAULT_HERATH_ROOT),
               "m_absa": load_real_from_mapped_jsonl(MABSA),
               "oats": load_real_from_mapped_jsonl(OATS)}
    if args.smoke:
        targets = {k: v.sample(min(120, len(v)), random_state=42).reset_index(drop=True) for k, v in targets.items()}

    results = {}
    for name, real in targets.items():
        aspects = sorted({a for labs in real["aspects"] for a in labs.keys()})
        pool = restrict_to_overlap(synth, aspects)
        tr, cal, _ = three_way_split(pool, cfg.split_calib, cfg.split_test, cfg.seed)
        train_pairs = make_pairs(tr, aspects, seed=cfg.seed)
        val_pairs = make_pairs(cal, aspects, seed=cfg.seed)
        print(f"[exp3] target={name} aspects={len(aspects)} train_pairs={len(train_pairs)} real={len(real)}", flush=True)
        model, tok = train(args.approach, train_pairs, val_pairs, cfg)
        thr = calibrate(model, tok, cal, aspects, cfg)
        probs = score_reviews(model, tok, real["text"].tolist(), aspects, cfg)
        thr_vec = np.array([thr[a] for a in aspects])
        pred = (probs >= thr_vec).astype(int)
        gold = np.array([[1 if a in r else 0 for a in aspects] for r in real["aspects"]])
        results[name] = {
            "micro_precision": float(precision_score(gold.ravel(), pred.ravel(), zero_division=0)),
            "micro_recall": float(recall_score(gold.ravel(), pred.ravel(), zero_division=0)),
            "micro_f1": float(f1_score(gold.ravel(), pred.ravel(), zero_division=0)),
            "n_real": int(len(real)),
        }
        print(f"[exp3] {name}: aspect-query detection microF1={results[name]['micro_f1']:.4f}", flush=True)
        import gc
        del model, tok
        gc.collect(); torch.cuda.empty_cache()

    print(json.dumps(results, indent=2), flush=True)
    if not args.smoke:
        OUT.write_text(json.dumps(results, indent=2))
        print(f"[exp3] wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
