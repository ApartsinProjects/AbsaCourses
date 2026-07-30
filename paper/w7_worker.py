"""W7 audit-quartile dose-response worker (runs inside the Modal A10G container).

Bucket the GPT synthetic corpus into audit-score quartiles (Q1 lowest .. Q4 highest)
by row_score from the per-row faithfulness CSV, at MATCHED bucket size (rank-based
qcut gives equal-size buckets). Train the BERT detector+sentiment head on each
quartile and evaluate on:
  (a) a held-out internal synthetic test split (fixed across all arms), and
  (b) real Herath (regenerated from the canonical XMI via load_herath_mapped_dataset).

Also runs a "full" arm (entire train pool) whose Herath micro-F1 is the SANITY GATE:
it must land near ~0.40-0.48 (paper synthetic-only transfer reference 0.4593). If it
comes out ~0.18 the Herath mapping is wrong.

Uses the repo's ACTUAL harness functions and the canonical loaders. Everything is
restricted to the 9 Herath-overlap aspects so a single model per arm serves both the
internal and the Herath evaluation (construct-matched, one training pass per arm).

Invocation (inside container, cwd=/app):
  python paper/w7_worker.py --seed 42 --out /results/W7_seed42
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

import sys
sys.path.insert(0, "/app/paper")

import absa_model_comparison as eng  # noqa: E402
from evaluate_synthetic_to_real_transfer import (  # noqa: E402
    load_herath_mapped_dataset, restrict_to_overlap,
)

DATA = Path("/app/data")
GPT_CORPUS = DATA / "gpt_corpus_10k.jsonl"
SCORES_CSV = DATA / "at_scale_per_row_scores.csv"
HERATH_ROOT = Path("/app/herath")

OVERLAP_9 = ["accessibility", "assessment_design", "exam_fairness", "grading_transparency",
             "lecturer_quality", "materials", "organization", "overall_experience", "workload"]


def free(*objs):
    for o in objs:
        del o
    gc.collect()
    torch.cuda.empty_cache()


def train_and_eval(train_arm, calib, aspects, internal_test, herath, cfg) -> dict:
    eng.set_seed(cfg.seed)  # per-arm reseed: isolate the training-data treatment
    det, tok = eng.train_detection("bert-base-uncased", train_arm, calib, aspects, cfg)
    sen, stok = eng.train_sentiment("bert-base-uncased", train_arm, calib, aspects, cfg)
    thr = eng.calibrate_thresholds(det, calib, tok, aspects, cfg)
    _, s_int = eng.evaluate_models("bert-base-uncased", det, sen, internal_test, tok, stok, aspects, thr, cfg)
    _, s_her = eng.evaluate_models("bert-base-uncased", det, sen, herath, tok, stok, aspects, thr, cfg)
    free(det, sen, tok, stok)
    return {
        "n_train": int(len(train_arm)),
        "internal_micro_f1": float(s_int["micro_f1"]),
        "internal_sentiment_mse": float(s_int["sentiment_mse_detected"]),
        "herath_micro_f1": float(s_her["micro_f1"]),
        "herath_sentiment_mse": float(s_her["sentiment_mse_detected"]),
    }


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
    eng.log_event(f"[W7 seed={cfg.seed}] device={cfg.device}")

    # ---- load synthetic corpus + join row_score by positional row_id
    syn = eng.load_jsonl(GPT_CORPUS).reset_index(drop=True)
    assert len(syn) == 10000, f"expected 10000 synthetic rows, got {len(syn)}"
    scores = pd.read_csv(SCORES_CSV).set_index("row_id")["row_score"]
    syn["row_score"] = [float(scores.loc[i]) if i in scores.index else np.nan for i in range(len(syn))]
    syn = syn[np.isfinite(syn["row_score"])].reset_index(drop=True)
    eng.log_event(f"[W7] synthetic rows with audit score = {len(syn)}")

    # ---- restrict to the 9 Herath-overlap aspects (row_score column is preserved)
    syn9 = restrict_to_overlap(syn, OVERLAP_9)
    eng.log_event(f"[W7] synthetic rows with >=1 of the 9 overlap aspects = {len(syn9)}")
    aspects = sorted(OVERLAP_9)

    # ---- fixed internal split: train pool / calib / test (held-out, same for every arm)
    train_pool, calib, internal_test = eng.three_way_split(syn9, cfg.split_calib, cfg.split_test, cfg.seed)
    train_pool = train_pool.reset_index(drop=True)
    eng.log_event(f"[W7] train_pool={len(train_pool)} calib={len(calib)} internal_test={len(internal_test)}")

    # ---- canonical Herath (regenerated from XMI), restricted to 9 aspects
    herath = load_herath_mapped_dataset(HERATH_ROOT)
    herath = restrict_to_overlap(herath, OVERLAP_9).reset_index(drop=True)
    her_aspects = sorted({a for m in herath["aspects"] for a in m})
    eng.log_event(f"[W7] Herath rows={len(herath)} aspects={her_aspects}")
    assert len(herath) == 2829, f"Herath mapping produced {len(herath)} rows, expected 2829 (mapping wrong!)"
    assert her_aspects == aspects, f"Herath aspects {her_aspects} != {aspects}"

    # ---- rank-based quartiles (equal size => matched bucket size)
    ranks = train_pool["row_score"].rank(method="first")
    q = pd.qcut(ranks, 4, labels=[1, 2, 3, 4])
    buckets = {qi: train_pool[q == qi].reset_index(drop=True) for qi in [1, 2, 3, 4]}
    mn = min(len(b) for b in buckets.values())
    buckets = {qi: b.sample(mn, random_state=cfg.seed).reset_index(drop=True) for qi, b in buckets.items()}
    for qi, b in buckets.items():
        eng.log_event(f"[W7] Q{qi}: n={len(b)} row_score[min={b.row_score.min():.3f} "
                      f"mean={b.row_score.mean():.3f} max={b.row_score.max():.3f}]")

    arms = {}

    # ---- SANITY / full-corpus arm first (Herath micro-F1 must be ~0.40-0.48)
    eng.log_event("[W7] ===== FULL arm (sanity gate) =====")
    full_res = train_and_eval(train_pool, calib, aspects, internal_test, herath, cfg)
    arms["full"] = full_res
    (out_dir / "partial_full.json").write_text(json.dumps(full_res, indent=2), encoding="utf-8")
    eng.log_event(f"[W7] FULL herath_micro_f1={full_res['herath_micro_f1']:.4f} "
                  f"(sanity gate ~0.40-0.48; paper ref 0.4593)")

    # ---- quartile arms
    for qi in [1, 2, 3, 4]:
        eng.log_event(f"[W7] ===== Q{qi} arm =====")
        res = train_and_eval(buckets[qi], calib, aspects, internal_test, herath, cfg)
        res["row_score_min"] = float(buckets[qi].row_score.min())
        res["row_score_mean"] = float(buckets[qi].row_score.mean())
        res["row_score_max"] = float(buckets[qi].row_score.max())
        arms[f"Q{qi}"] = res
        (out_dir / f"partial_Q{qi}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
        eng.log_event(f"[W7] Q{qi} internal_micro_f1={res['internal_micro_f1']:.4f} "
                      f"herath_micro_f1={res['herath_micro_f1']:.4f} "
                      f"internal_mse={res['internal_sentiment_mse']:.4f}")

    sanity_ok = 0.36 <= full_res["herath_micro_f1"] <= 0.52
    result = {
        "experiment": "W7_audit_quartile_dose_response",
        "seed": cfg.seed,
        "aspects": aspects,
        "n_herath": int(len(herath)),
        "matched_bucket_size": int(mn),
        "sanity_gate": {
            "full_corpus_herath_micro_f1": full_res["herath_micro_f1"],
            "expected_range": "0.40-0.48 (paper synthetic-only transfer ref 0.4593)",
            "passed": bool(sanity_ok),
        },
        "arms": arms,
        "internal_dose_response_Q1_to_Q4": [arms[f"Q{i}"]["internal_micro_f1"] for i in [1, 2, 3, 4]],
        "herath_dose_response_Q1_to_Q4": [arms[f"Q{i}"]["herath_micro_f1"] for i in [1, 2, 3, 4]],
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    rows = []
    for arm, r in arms.items():
        rows.append({"arm": arm, **r})
    pd.DataFrame(rows).to_csv(out_dir / "w7_arms.csv", index=False)
    print("RESULT_JSON_BEGIN")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("RESULT_JSON_END")


if __name__ == "__main__":
    main()
