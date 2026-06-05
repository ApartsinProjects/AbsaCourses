"""C1 relabel-and-retrain worker (SYNTHETIC-ONLY training, transfer eval on REAL data).

For ONE seed and ONE real transfer target, co-compute in the SAME process:
  BASELINE  = synthetic-only detector+sentiment trained on ORIGINAL labels.
  TREATMENT = same recipe trained on CLEANED labels (audit-derived):
                detection trained on supported-only aspects,
                sentiment trained on supported AND sentiment_match aspects (rest masked).

Both are evaluated by TRANSFER to the real human-annotated target (Herath 9-aspect or
EduRABSA 7-aspect): detection micro-F1 AND sentiment MSE on detected aspects. The real
data is NEVER used for training -- it is the held-out, unbiased transfer metric. The
synthetic test split is NOT used as a metric (it carries the same label noise we audit).

Recipe matches evaluate_synthetic_to_real_transfer.py exactly:
  - aspects = the real target's overlap aspect set (sorted).
  - restrict synthetic corpus to those aspects, three_way_split (seed) -> synth_train/calib.
  - train_detection + train_sentiment on synth_train (val=synth_calib),
    calibrate_thresholds on synth_calib, evaluate_models on the FULL real_df.

Co-computation guarantee: baseline and treatment share the SAME seed, SAME split
(same row_ids, since the split is computed on row order then labels are swapped in
place), SAME aspect set, SAME real_df, SAME Config. Only the training labels differ.

REPRODUCE GATE (Herath only): baseline detection micro-F1 must be within +-0.05 of the
synthetic-only transfer reference 0.4593. If it is not, the result records the
discrepancy and sets reproduce_ok=False (the caller decides whether to scale).

Invocation (inside container, cwd=/app):
  python paper/.../c1_worker.py --seed 42 --target herath --out /results/herath_seed42
  python paper/.../c1_worker.py --seed 42 --target edurabsa --out /results/edurabsa_seed42
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# absa_model_comparison + absa_data_io: prefer C1_BASE/paper, then <repo>/paper,
# then /app/paper (legacy). The local-run fallback lets the same worker run on the
# RTX 2060 (repo checkout) and on the vast container.
_base = os.environ.get("C1_BASE", "")
if _base:
    sys.path.insert(0, str(Path(_base).resolve() / "paper"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # <repo>/paper
sys.path.insert(0, "/app/paper")

import absa_model_comparison as eng  # noqa: E402


SYNTH_ONLY_TRANSFER_REF_HERATH = 0.4593
REPRO_TOL = 0.05

HERATH_OVERLAP_9 = ["accessibility", "assessment_design", "exam_fairness",
                    "grading_transparency", "lecturer_quality", "materials",
                    "organization", "overall_experience", "workload"]


def _load_real(target: str, herath_jsonl: Path, edurabsa_jsonl: Path) -> pd.DataFrame:
    """Load the real transfer target as {text, aspects:{aspect:polarity}}."""
    path = herath_jsonl if target == "herath" else edurabsa_jsonl
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            text = str(r.get("text", "")).strip()
            amap = r.get("aspects", {})
            amap = {str(k).strip(): str(v).strip().lower() for k, v in amap.items()
                    if str(v).strip().lower() in eng.SENT2VAL}
            if not text or not amap:
                continue
            rows.append({"text": text, "aspects": amap, "target_attributes": amap})
    df = pd.DataFrame(rows).reset_index(drop=True)
    if df.empty:
        raise ValueError(f"No real rows loaded from {path}")
    return df


def _load_cleaned_corpus(cleaned_jsonl: Path) -> list[dict]:
    return [json.loads(l) for l in cleaned_jsonl.open(encoding="utf-8")]


def _restrict(d: dict, keep: set) -> dict:
    return {a: s for a, s in d.items() if a in keep}


def _build_corpus_frames(cleaned_rows: list[dict], aspects: list[str], condition: str):
    """Return (detection_df, sentiment_df) row-aligned for the given condition.

    condition == 'baseline'  : both views use the ORIGINAL labels.
    condition == 'treatment' : detection uses supported aspects (aspects_detection);
                               sentiment uses supported&matched aspects (aspects_sentiment).

    Detection and sentiment are returned as separate frames because the cleaned
    sentiment view can be a strict subset of the detection view (masking). The
    engine's DetectionDataset reads dict KEYS; SentimentDataset reads dict {key:pol}
    with mask=1 per key. Restricting each frame's dict realizes drop (detection) and
    mask (sentiment) without engine changes.
    """
    keep = set(aspects)
    det_rows, sent_rows = [], []
    for r in cleaned_rows:
        text = r["text"]
        if condition == "baseline":
            det_lab = _restrict(r["aspects_original"], keep)
            sent_lab = _restrict(r["aspects_original"], keep)
        elif condition == "treatment":
            det_lab = _restrict(r["aspects_detection"], keep)
            sent_lab = _restrict(r["aspects_sentiment"], keep)
        else:
            raise ValueError(condition)
        det_rows.append({"text": text, "aspects": det_lab, "target_attributes": det_lab, "row_id": r["row_id"]})
        sent_rows.append({"text": text, "aspects": sent_lab, "target_attributes": sent_lab, "row_id": r["row_id"]})
    det_df = pd.DataFrame(det_rows).reset_index(drop=True)
    sent_df = pd.DataFrame(sent_rows).reset_index(drop=True)
    return det_df, sent_df


def _split_indices(n: int, cfg: eng.Config, seed: int):
    """Reproduce three_way_split index selection on row ORDER (label-independent),
    so baseline and treatment get identical train/calib row_ids."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    calib_size, test_size = cfg.split_calib, cfg.split_test
    counts = np.floor(np.array([1.0 - calib_size - test_size, calib_size, test_size]) * n).astype(int)
    counts = np.maximum(counts, 1)
    while counts.sum() > n:
        reducible = [idx for idx in np.argsort(-counts) if counts[idx] > 1]
        if not reducible:
            break
        counts[reducible[0]] -= 1
    if counts.sum() < n:
        counts[0] += n - counts.sum()
    train_n, calib_n, test_n = counts.tolist()
    train_idx = perm[:train_n]
    calib_idx = perm[train_n:train_n + calib_n]
    test_idx = perm[train_n + calib_n:train_n + calib_n + test_n]
    return train_idx, calib_idx, test_idx


def _run_condition(condition, cleaned_rows, aspects, real_df, cfg, seed, out_dir, train_idx, calib_idx):
    """Train (det+sent) on the synthetic train split for `condition`, eval on real_df."""
    det_df, sent_df = _build_corpus_frames(cleaned_rows, aspects, condition)
    # restrict to overlap is already done in _build_corpus_frames via `keep`
    det_train = det_df.iloc[train_idx].reset_index(drop=True)
    det_calib = det_df.iloc[calib_idx].reset_index(drop=True)
    sent_train = sent_df.iloc[train_idx].reset_index(drop=True)
    sent_calib = sent_df.iloc[calib_idx].reset_index(drop=True)

    # label-count diagnostics for this condition's TRAIN split
    det_label_count = int(sum(len(d) for d in det_train["aspects"]))
    sent_label_count = int(sum(len(d) for d in sent_train["aspects"]))
    eng.log_event(f"[C1 seed={seed} {condition}] det_train_rows={len(det_train)} "
                  f"det_labels={det_label_count} sent_labels={sent_label_count}")

    eng.set_seed(seed)  # re-seed before each condition for identical init given same labels
    det_model, det_tok = eng.train_detection("bert-base-uncased", det_train, det_calib, aspects, cfg)
    sent_model, sent_tok = eng.train_sentiment("bert-base-uncased", sent_train, sent_calib, aspects, cfg)
    thresholds = eng.calibrate_thresholds(det_model, det_calib, det_tok, aspects, cfg)

    per_aspect_df, summary, artifact = eng.evaluate_models(
        "bert-base-uncased", det_model, sent_model, real_df, det_tok, sent_tok,
        aspects, thresholds, cfg, return_artifacts=True,
    )
    eng.log_event(f"[C1 seed={seed} {condition}] real-transfer micro_f1={summary['micro_f1']:.4f} "
                  f"sentiment_mse_detected={summary['sentiment_mse_detected']:.4f}")

    cond_dir = out_dir / condition
    cond_dir.mkdir(parents=True, exist_ok=True)
    per_aspect_df.to_csv(cond_dir / "per_aspect.csv", index=False)
    with (cond_dir / "test_predictions.jsonl").open("w", encoding="utf-8") as f:
        for rec in artifact["sample_predictions"]:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return {
        "condition": condition,
        "n_det_train_rows": int(len(det_train)),
        "n_det_train_labels": det_label_count,
        "n_sent_train_labels": sent_label_count,
        "thresholds": {a: float(thresholds[a]) for a in aspects},
        "detection_micro_f1": float(summary["micro_f1"]),
        "detection_micro_precision": float(summary["micro_precision"]),
        "detection_micro_recall": float(summary["micro_recall"]),
        "detection_macro_f1": float(summary["macro_f1"]),
        "sentiment_mse_detected": float(summary["sentiment_mse_detected"]),
    }


def run(seed: int, target: str, out_dir: Path, cleaned_jsonl: Path,
        herath_jsonl: Path, edurabsa_jsonl: Path) -> dict:
    cfg = eng.Config(seed=seed)
    eng.set_seed(seed)
    eng.log_event(f"[C1 seed={seed} target={target}] device={cfg.device}")

    cleaned_rows = _load_cleaned_corpus(cleaned_jsonl)
    assert len(cleaned_rows) == 10000, f"expected 10000 cleaned rows, got {len(cleaned_rows)}"

    real_df = _load_real(target, herath_jsonl, edurabsa_jsonl)
    aspects = sorted({a for d in real_df["aspects"] for a in d.keys()})
    eng.log_event(f"[C1 seed={seed} target={target}] real_rows={len(real_df)} aspects({len(aspects)})={aspects}")

    # Restrict the synthetic corpus to rows declaring >=1 overlap aspect in the ORIGINAL
    # labels, matching evaluate_synthetic_to_real_transfer.restrict_to_overlap (which
    # drops no-overlap rows BEFORE the split, then three_way_split over the restricted
    # set). Keying on aspects_original keeps the kept-row set IDENTICAL for baseline and
    # treatment, so the paired comparison stays matched; cleaning only changes labels
    # within these rows. Without this, training on all 10k rows (mostly all-negative for
    # the overlap aspects) under-trains the detector and the baseline does not reproduce
    # the 0.4593 synthetic-only transfer reference (it gave 0.301).
    keep = set(aspects)
    restricted_rows = [r for r in cleaned_rows if any(a in keep for a in r["aspects_original"].keys())]
    n = len(restricted_rows)
    eng.log_event(f"[C1 seed={seed} target={target}] restricted synth rows with >=1 overlap aspect: {n}/10000")

    # SAME split row_ids for both conditions (computed on row order, label-independent)
    train_idx, calib_idx, test_idx = _split_indices(n, cfg, seed)
    eng.log_event(f"[C1 seed={seed} target={target}] synth split train={len(train_idx)} calib={len(calib_idx)} (test split unused)")

    out_dir.mkdir(parents=True, exist_ok=True)
    baseline = _run_condition("baseline", restricted_rows, aspects, real_df, cfg, seed, out_dir, train_idx, calib_idx)
    treatment = _run_condition("treatment", restricted_rows, aspects, real_df, cfg, seed, out_dir, train_idx, calib_idx)

    reproduce_ok = None
    if target == "herath":
        reproduce_ok = abs(baseline["detection_micro_f1"] - SYNTH_ONLY_TRANSFER_REF_HERATH) <= REPRO_TOL
        eng.log_event(f"[C1 seed={seed}] reproduce gate (|{baseline['detection_micro_f1']:.4f} - "
                      f"{SYNTH_ONLY_TRANSFER_REF_HERATH}| <= {REPRO_TOL}): {reproduce_ok}")

    result = {
        "experiment": "C1_relabel_and_retrain_synthetic_only_transfer",
        "seed": seed,
        "target": target,
        "n_real_rows": int(len(real_df)),
        "aspects": aspects,
        "n_aspects": len(aspects),
        "synthetic_only_transfer_reference_micro_f1_herath": SYNTH_ONLY_TRANSFER_REF_HERATH,
        "reproduce_ok": reproduce_ok,
        "baseline": baseline,
        "treatment": treatment,
        "paired_delta_treatment_minus_baseline": {
            "detection_micro_f1": round(treatment["detection_micro_f1"] - baseline["detection_micro_f1"], 4),
            "sentiment_mse_detected": round(treatment["sentiment_mse_detected"] - baseline["sentiment_mse_detected"], 4),
        },
        "config": {"max_len": cfg.max_len, "batch_size": cfg.batch_size, "lr": cfg.lr,
                   "epochs_detection": cfg.epochs_detection, "epochs_sentiment": cfg.epochs_sentiment,
                   "patience": cfg.patience, "split_calib": cfg.split_calib, "split_test": cfg.split_test},
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--target", required=True, choices=["herath", "edurabsa"])
    p.add_argument("--out", required=True)
    p.add_argument("--cleaned-jsonl", default="/app/data/cleaned_corpus_10k.jsonl")
    p.add_argument("--herath-jsonl", default="/app/data/herath_mapped_real_reviews_2829.jsonl")
    p.add_argument("--edurabsa-jsonl", default="/app/data/edurabsa_test_mapped.jsonl")
    args = p.parse_args()

    eng.configure_console_encoding()
    eng.ensure_dirs()
    out_dir = Path(args.out)
    t0 = time.time()
    res = run(args.seed, args.target, out_dir, Path(args.cleaned_jsonl),
              Path(args.herath_jsonl), Path(args.edurabsa_jsonl))
    res["elapsed_seconds"] = round(time.time() - t0, 1)
    (out_dir / "result.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print("RESULT_JSON_BEGIN")
    print(json.dumps(res, indent=2, ensure_ascii=False))
    print("RESULT_JSON_END")


if __name__ == "__main__":
    main()
