"""Multi-seed paired-delta validation of the transfer-improvement claims.

Single-seed transfer numbers are high-variance (OATS whole-doc 0.275 vs 0.448 across
runs). The correct statistic for each claim is the PAIRED delta computed on models
that share a seed, which cancels most of that variance. We report per-seed deltas and
mean +/- std across seeds for:
  A28 windowing:  windowed_F1 - whole_F1        (same det model; OATS, Herath control)
  Exp2 sentence:  sentence_F1 - review_F1        (Herath, M-ABSA)
  Exp4 polarity:  correlated_MSE - original_MSE  (Herath, OATS; lower is better)
  Exp5 structure: structured_F1 - original_F1    (OATS whole-doc)

Detection-only where the metric is detection F1 (halves cost); sentiment-only for Exp4.
Saves incrementally to paper/outputs/multiseed_validation.json.
"""
from __future__ import annotations

import gc
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from absa_model_comparison import (
    Config, DetectionDataset, SentimentDataset, calibrate_thresholds,
    collect_detection, collect_sentiment, masked_mse_numpy, set_seed,
    three_way_split, train_detection, train_sentiment,
)
from evaluate_synthetic_to_real_transfer import (
    DEFAULT_HERATH_ROOT, DEFAULT_SYNTHETIC_PATH, load_herath_mapped_dataset,
    load_jsonl, load_real_from_mapped_jsonl, restrict_to_overlap,
)
from exp2_sentence_train import build_sentence_df

ROOT = Path(__file__).resolve().parents[1]
OATS = ROOT / "external_data/OATS_coursera/oats_mapped.jsonl"
MABSA = ROOT / "external_data/M-ABSA_coursera/m_absa_mapped.jsonl"
CORR = ROOT / "paper/outputs/exp4_correlated_gen.jsonl"
STRUCT = ROOT / "paper/outputs/exp5_structured_gen.jsonl"
INDEP = ROOT / "paper/outputs/exp_indep_gen.jsonl"  # matched independent-structure baseline (same pipeline)
OUT = ROOT / "paper/outputs/multiseed_validation.json"
SEEDS = [42, 17, 23]
POL2NUM = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def load_gen(path):
    rows = []
    for l in open(path, encoding="utf-8"):
        r = json.loads(l)
        if r.get("text") and r.get("aspects"):
            rows.append({"text": r["text"], "aspects": r["aspects"], "target_attributes": r["aspects"],
                         "nuance_attributes": {}, "course_name": "", "grade": "", "style": ""})
    return pd.DataFrame(rows)


def det_probs_gold(det, tok, df, aspects, cfg):
    loader = DataLoader(DetectionDataset(df, tok, aspects, cfg.max_len), batch_size=cfg.batch_size, shuffle=False)
    probs, true = collect_detection(det, loader, cfg.device)
    return probs, true


def whole_f1(det, tok, real, aspects, thr, cfg):
    probs, true = det_probs_gold(det, tok, real, aspects, cfg)
    pred = (probs >= np.array([thr[a] for a in aspects])).astype(int)
    return float(f1_score(true.ravel(), pred.ravel(), zero_division=0))


def windowed_f1(det, tok, real, aspects, thr, cfg):
    sent_texts, groups = [], []
    for _, row in real.reset_index(drop=True).iterrows():
        parts = [s.strip() for s in SENT_SPLIT.split(" ".join(str(row["text"]).split())) if len(s.split()) >= 3] or [str(row["text"]).strip()]
        start = len(sent_texts); sent_texts.extend(parts); groups.append(list(range(start, start + len(parts))))
    sdf = pd.DataFrame({"text": sent_texts, "aspects": [{} for _ in sent_texts]})
    sprobs, _ = det_probs_gold(det, tok, sdf, aspects, cfg)
    thrv = np.array([thr[a] for a in aspects])
    pooled = np.zeros((len(real), len(aspects)), dtype=np.float32)
    for r, idxs in enumerate(groups):
        pooled[r] = sprobs[idxs].max(axis=0)
    pred = (pooled >= thrv).astype(int)
    gold = np.array([[1 if a in m else 0 for a in aspects] for m in real["aspects"]])
    return float(f1_score(gold.ravel(), pred.ravel(), zero_division=0))


def gold_sent_mse(sen, tok, real, aspects, cfg):
    loader = DataLoader(SentimentDataset(real, tok, aspects, cfg.max_len), batch_size=cfg.batch_size, shuffle=False)
    preds, tgt, mask = collect_sentiment(sen, loader, cfg.device)
    return float(masked_mse_numpy(preds, tgt, mask))


def train_det(pool, aspects, cfg):
    set_seed(cfg.seed)  # re-seed per arm: identical head-init + shuffle, so a paired delta isolates the DATA treatment
    tr, cal, _ = three_way_split(pool, cfg.split_calib, cfg.split_test, cfg.seed)
    det, tok = train_detection("bert-base-uncased", tr, cal, aspects, cfg)
    thr = calibrate_thresholds(det, cal, tok, aspects, cfg)
    return det, tok, thr


def train_sent(pool, aspects, cfg):
    set_seed(cfg.seed)  # re-seed per arm (see train_det)
    tr, cal, _ = three_way_split(pool, cfg.split_calib, cfg.split_test, cfg.seed)
    return train_sentiment("bert-base-uncased", tr, cal, aspects, cfg)


def free(*objs):
    for o in objs:
        del o
    gc.collect(); torch.cuda.empty_cache()


def equal_pools(dfA, dfB, aspects, seed):
    """Restrict both corpora to the target aspect overlap, then subsample BOTH to the
    same (minimum) size so training-set size cannot confound the structure comparison."""
    a = restrict_to_overlap(dfA, aspects); b = restrict_to_overlap(dfB, aspects)
    m = min(len(a), len(b))
    return a.sample(m, random_state=seed).reset_index(drop=True), b.sample(m, random_state=seed).reset_index(drop=True)


def main():
    cfg = Config()
    # synth (the paper's 10k corpus) is used ONLY for windowing (single model) and exp2
    # (review vs sentence, both arms share this source) -> no cross-corpus confound there.
    synth = load_jsonl(DEFAULT_SYNTHETIC_PATH)
    sent_synth = build_sentence_df(synth)
    # Exp4/Exp5 baselines: a freshly-generated INDEPENDENT-structure corpus (same OpenRouter
    # pipeline as correlated/structured), so the only difference from corr/struct is structure,
    # not generation batch/model/date/cleanliness. Equal N is enforced AFTER restrict_to_overlap.
    corr = load_gen(CORR); struct = load_gen(STRUCT); indep = load_gen(INDEP)
    targets = {"herath": load_herath_mapped_dataset(DEFAULT_HERATH_ROOT),
               "m_absa": load_real_from_mapped_jsonl(MABSA), "oats": load_real_from_mapped_jsonl(OATS)}
    asp = {k: sorted({a for m in v["aspects"] for a in m}) for k, v in targets.items()}

    res = json.loads(OUT.read_text()) if OUT.exists() else {"seeds": [], "A28_windowing": {}, "exp2_sentence": {}, "exp4_polarity": {}, "exp5_structure": {}}

    def rec(bucket, key, val):
        res[bucket].setdefault(key, []).append(round(val, 4))
        OUT.write_text(json.dumps(res, indent=2))

    for seed in SEEDS:
        if seed in res["seeds"]:
            print(f"[ms] seed {seed} already done, skip", flush=True); continue
        cfg.seed = seed; set_seed(seed)
        print(f"===== SEED {seed} =====", flush=True)

        # A28 windowing (OATS, Herath) - detection only
        for tgt in ("oats", "herath"):
            det, tok, thr = train_det(restrict_to_overlap(synth, asp[tgt]), asp[tgt], cfg)
            w = whole_f1(det, tok, targets[tgt], asp[tgt], thr, cfg)
            v = windowed_f1(det, tok, targets[tgt], asp[tgt], thr, cfg)
            rec("A28_windowing", f"{tgt}_whole", w); rec("A28_windowing", f"{tgt}_windowed", v)
            rec("A28_windowing", f"{tgt}_delta", v - w)
            print(f"[ms:{seed}] windowing {tgt}: whole={w:.4f} windowed={v:.4f} delta={v-w:+.4f}", flush=True)
            free(det, tok)

        # Exp2 sentence vs review (herath, m_absa) - detection only
        for tgt in ("herath", "m_absa"):
            dr, tr_, thr = train_det(restrict_to_overlap(synth, asp[tgt]), asp[tgt], cfg)
            rf = whole_f1(dr, tr_, targets[tgt], asp[tgt], thr, cfg); free(dr, tr_)
            ds, ts, ths = train_det(restrict_to_overlap(sent_synth, asp[tgt]), asp[tgt], cfg)
            sf = whole_f1(ds, ts, targets[tgt], asp[tgt], ths, cfg); free(ds, ts)
            rec("exp2_sentence", f"{tgt}_review", rf); rec("exp2_sentence", f"{tgt}_sentence", sf)
            rec("exp2_sentence", f"{tgt}_delta", sf - rf)
            print(f"[ms:{seed}] exp2 {tgt}: review={rf:.4f} sentence={sf:.4f} delta={sf-rf:+.4f}", flush=True)

        # Exp4 correlated vs matched-independent polarity - sentiment only, eval herath+oats.
        # Equal pools AFTER restrict; baseline = fresh independent corpus (same pipeline).
        for tgt in ("herath", "oats"):
            cpool, ipool = equal_pools(corr, indep, asp[tgt], seed)
            for src_name, pool in (("correlated", cpool), ("original", ipool)):
                sen, tok = train_sent(pool, asp[tgt], cfg)
                rec("exp4_polarity", f"{tgt}_{src_name}", gold_sent_mse(sen, tok, targets[tgt], asp[tgt], cfg))
                free(sen, tok)
            d = res["exp4_polarity"][f"{tgt}_correlated"][-1] - res["exp4_polarity"][f"{tgt}_original"][-1]
            rec("exp4_polarity", f"{tgt}_delta", d)
            print(f"[ms:{seed}] exp4 {tgt}: delta(corr-indep) MSE={d:+.4f} (n_pool={len(cpool)})", flush=True)

        # Exp5 structured vs matched-independent - detection only, OATS whole-doc.
        spool, ipool = equal_pools(struct, indep, asp["oats"], seed)
        for src_name, pool in (("structured", spool), ("original", ipool)):
            det, tok, thr = train_det(pool, asp["oats"], cfg)
            rec("exp5_structure", f"oats_{src_name}", whole_f1(det, tok, targets["oats"], asp["oats"], thr, cfg))
            free(det, tok)
        d = res["exp5_structure"]["oats_structured"][-1] - res["exp5_structure"]["oats_original"][-1]
        rec("exp5_structure", "oats_delta", d)
        print(f"[ms:{seed}] exp5 oats: delta(struct-indep) F1={d:+.4f} (n_pool={len(spool)})", flush=True)

        res["seeds"].append(seed); OUT.write_text(json.dumps(res, indent=2))

    # summary
    print("\n===== PAIRED-DELTA SUMMARY (mean +/- std across seeds) =====", flush=True)
    for bucket in ("A28_windowing", "exp2_sentence", "exp4_polarity", "exp5_structure"):
        for k, vals in res[bucket].items():
            if k.endswith("_delta"):
                a = np.array(vals)
                print(f"  {bucket:16s} {k:18s} = {a.mean():+.4f} +/- {a.std():.4f}  (n={len(a)}: {vals})", flush=True)
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
