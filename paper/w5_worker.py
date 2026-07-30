"""W5 held-out-generator transfer worker (runs inside the Modal A10G container).

Train the BERT detector on the GPT-generated synthetic corpus (10k) and evaluate
detection micro-F1 on each held-out generator's synthetic corpus (gpt5nano,
gemini, glm, llama). All five corpora share the identical 20-aspect schema, so the
"shared aspects" set is the full 20 aspects (no restriction needed). Also runs the
reverse direction (train on a non-GPT generator, test on GPT) for every non-GPT gen.

Uses the repo's ACTUAL harness functions (train_detection, calibrate_thresholds,
collect_detection, DetectionDataset, three_way_split, discover_aspects, Config,
set_seed) from absa_model_comparison. No re-implemented training/eval.

Invocation (inside container, cwd=/app):
  python paper/w5_worker.py --seed 42 --out /results/W5_seed42
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, "/app/paper")

import absa_model_comparison as eng  # noqa: E402

DATA = Path("/app/data")
GPT_CORPUS = DATA / "gpt_corpus_10k.jsonl"
HELDOUT = {
    "gpt5nano": DATA / "n3_gen_gpt5nano.jsonl",
    "gemini_flash": DATA / "n3_gen_gemini_flash.jsonl",
    "glm_46": DATA / "n3_gen_glm_46.jsonl",
    "llama33_70b": DATA / "n3_gen_llama33_70b.jsonl",
}
NON_GPT = ["gemini_flash", "glm_46", "llama33_70b"]


def load_gen_labels(path: Path) -> pd.DataFrame:
    """Held-out generator corpora carry the aspect dict under 'labels'."""
    rows = []
    for line in path.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        t = r.get("text")
        lab = r.get("labels") or {}
        if t and lab:
            rows.append({"text": t, "aspects": lab, "target_attributes": lab,
                         "nuance_attributes": {}, "course_name": "", "grade": "", "style": ""})
    return pd.DataFrame(rows).reset_index(drop=True)


def eval_detection(det, tok, target_df, aspects, thr, cfg) -> dict:
    loader = DataLoader(eng.DetectionDataset(target_df, tok, aspects, cfg.max_len),
                        batch_size=cfg.batch_size, shuffle=False)
    probs, true = eng.collect_detection(det, loader, cfg.device)
    thr_vec = np.array([thr[a] for a in aspects], dtype=np.float32)
    preds = (probs >= thr_vec).astype(int)
    return {
        "micro_f1": float(f1_score(true.ravel(), preds.ravel(), zero_division=0)),
        "micro_precision": float(precision_score(true.ravel(), preds.ravel(), zero_division=0)),
        "micro_recall": float(recall_score(true.ravel(), preds.ravel(), zero_division=0)),
        "n_rows": int(len(target_df)),
    }


def free(*objs):
    for o in objs:
        del o
    gc.collect()
    torch.cuda.empty_cache()


def train_det_pool(pool, aspects, cfg):
    eng.set_seed(cfg.seed)  # per-arm reseed: identical head-init + shuffle
    tr, cal, _ = eng.three_way_split(pool, cfg.split_calib, cfg.split_test, cfg.seed)
    det, tok = eng.train_detection("bert-base-uncased", tr, cal, aspects, cfg)
    thr = eng.calibrate_thresholds(det, cal, tok, aspects, cfg)
    return det, tok, thr, tr, cal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    eng.configure_console_encoding()
    eng.ensure_dirs()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = eng.Config(seed=args.seed)
    eng.set_seed(cfg.seed)
    t0 = time.time()
    eng.log_event(f"[W5 seed={cfg.seed}] device={cfg.device}")

    gpt = eng.load_jsonl(GPT_CORPUS)
    assert len(gpt) == 10000, f"expected 10000 GPT rows, got {len(gpt)}"
    aspects = eng.discover_aspects(gpt)
    eng.log_event(f"[W5] GPT rows={len(gpt)} n_aspects={len(aspects)}")

    heldout = {name: load_gen_labels(p) for name, p in HELDOUT.items()}
    for name, df in heldout.items():
        ha = sorted({a for m in df["aspects"] for a in m})
        eng.log_event(f"[W5] heldout {name}: rows={len(df)} aspects={len(ha)}")
        # sanity: shared aspect set equals full schema
        assert set(ha).issubset(set(aspects)), f"{name} has aspects outside GPT schema"

    cells = []  # {train_gen, test_gen, ...metrics}

    # ---------------- FORWARD: train on GPT, eval on each held-out generator
    det, tok, thr, gpt_tr, gpt_cal = train_det_pool(gpt, aspects, cfg)
    # in-domain reference: GPT test split (from same split)
    _, _, gpt_test = eng.three_way_split(gpt, cfg.split_calib, cfg.split_test, cfg.seed)
    ref = eval_detection(det, tok, gpt_test, aspects, thr, cfg)
    cells.append({"train_gen": "gpt", "test_gen": "gpt_indomain_testsplit", **ref})
    eng.log_event(f"[W5] gpt->gpt(in-domain test) micro_f1={ref['micro_f1']:.4f}")
    for name, df in heldout.items():
        m = eval_detection(det, tok, df, aspects, thr, cfg)
        cells.append({"train_gen": "gpt", "test_gen": name, **m})
        eng.log_event(f"[W5] gpt->{name} micro_f1={m['micro_f1']:.4f} (n={m['n_rows']})")
    free(det, tok)

    # ---------------- REVERSE: train on each non-GPT gen, eval on GPT test split
    for name in NON_GPT:
        rdet, rtok, rthr, _, _ = train_det_pool(heldout[name], aspects, cfg)
        m = eval_detection(rdet, rtok, gpt_test, aspects, rthr, cfg)
        cells.append({"train_gen": name, "test_gen": "gpt_testsplit", **m})
        eng.log_event(f"[W5] {name}->gpt micro_f1={m['micro_f1']:.4f} (n={m['n_rows']})")
        free(rdet, rtok)

    result = {
        "experiment": "W5_heldout_generator_transfer",
        "seed": cfg.seed,
        "n_aspects": len(aspects),
        "aspects": aspects,
        "shared_aspect_note": "all 5 corpora share the identical 20-aspect schema; shared = full 20",
        "cells": cells,
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(cells).to_csv(out_dir / "w5_cells.csv", index=False)
    print("RESULT_JSON_BEGIN")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("RESULT_JSON_END")


if __name__ == "__main__":
    main()
