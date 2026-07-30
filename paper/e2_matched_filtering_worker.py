"""Covariate-matched faithfulness-filtering worker (reviewer nfat N2).

Addresses two confounds in the naive faithfulness-filtering comparison:
  (a) the retained (high-faithfulness) subset differs from the control on aspect
      identity, polarity, aspect-count, review length and style -- so a raw
      retained-vs-control gap could be driven by covariate shift, not faithfulness.
  (b) sentiment MSE measured only on each model's predicted aspects confounds the
      comparison with differing prediction masks.

Fixes:
  (a) build the control by strict COVARIATE MATCHING to the retained set. Greedy
      1:1 matching: each retained row is paired with an unused control-pool row in
      the SAME cell of (aspect_count x polarity_composition x length_band x
      formality_band), choosing the candidate with the highest aspect-set Jaccard.
      Both arms therefore share an identical joint distribution of the four scalar
      covariates and a best-effort-matched aspect identity. Retained = top-50% of
      the whole corpus by audit row_score; control pool = complementary 50%.
  (b) evaluate sentiment MSE on the COMMON GOLD aspect set -- the gold aspects of
      the real Herath eval (mask = gold-present), identical for both models, NOT
      each model's predicted set. Detection micro-F1 likewise on the same gold set.

The retained and control TRAINING pools are built once (matching_seed, data-only,
identical across model seeds); set_seed(seed) is called before each arm so a paired
retained-vs-control delta isolates the faithfulness treatment.

SANITY GATE (--sanity, seed 42): full-corpus (9-aspect) synthetic -> Herath
detection micro-F1 must land ~0.40-0.48; ~0.18 means the wrong Herath mapping.

Invocation (inside container, cwd=/app):
  python paper/e2_matched_filtering_worker.py --seed 42 --out /results/... --sanity
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

sys.path.insert(0, "/app/paper")

from absa_data_io import load_absa_dataset  # noqa: E402
import absa_model_comparison as eng  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

# The 9 aspects shared by the synthetic corpus and the real Herath eval.
OVERLAP_9 = ["accessibility", "assessment_design", "exam_fairness", "grading_transparency",
             "lecturer_quality", "materials", "organization", "overall_experience", "workload"]

SANITY_LO, SANITY_HI = 0.40, 0.48


# ----------------------------------------------------------------- data loaders
def load_real_from_mapped_jsonl(path: Path) -> pd.DataFrame:
    """Load the CORRECT XMI-derived Herath mapping (schema {text, aspects:{a:pol}})."""
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            text = str(r.get("text", "")).strip()
            aspects = r.get("aspects", {})
            if not text or not aspects:
                continue
            rows.append({"text": text, "aspects": aspects, "target_attributes": aspects})
    if not rows:
        raise ValueError(f"No rows in {path}")
    return pd.DataFrame(rows).reset_index(drop=True)


def restrict_to_overlap(df: pd.DataFrame, keep: list) -> pd.DataFrame:
    keepset = set(keep)
    rows = []
    for _, row in df.iterrows():
        labels = {a: s for a, s in row["aspects"].items() if a in keepset}
        if not labels:
            continue
        upd = dict(row)
        upd["aspects"] = labels
        upd["target_attributes"] = labels
        rows.append(upd)
    return pd.DataFrame(rows).reset_index(drop=True)


# ----------------------------------------------------------------- covariates
def _polarity_comp(aspects: dict) -> str:
    pols = set(aspects.values())
    if len(pols) == 0:
        return "none"
    if len(pols) > 1:
        return "mixed"
    return "all_" + next(iter(pols))


def build_covariate_frame(corpus_path: Path, scores_csv: Path) -> pd.DataFrame:
    """One row per corpus line with row_id (== audit row_id == line index),
    the full aspect dict, and the four scalar covariates + faithfulness row_score."""
    df = load_absa_dataset(corpus_path).reset_index(drop=True)
    n = len(df)
    assert n == 10000, f"expected 10000 corpus rows, got {n}"
    df["row_id"] = np.arange(n)
    scores = pd.read_csv(scores_csv).set_index("row_id")["row_score"]
    df["row_score"] = [float(scores.loc[i]) if i in scores.index else np.nan for i in range(n)]
    assert df["row_score"].notna().all(), "some rows missing an audit row_score"

    # covariates
    df["aspect_count"] = df["aspects"].map(len).clip(upper=3)
    df["polarity_comp"] = df["aspects"].map(_polarity_comp)
    df["word_count"] = df["text"].map(lambda t: len(str(t).split()))
    q1, q2 = df["word_count"].quantile([1 / 3, 2 / 3]).tolist()
    df["length_band"] = df["word_count"].map(lambda w: 0 if w <= q1 else (1 if w <= q2 else 2))

    def _form(row):
        na = row if isinstance(row, dict) else {}
        v = na.get("formality_level")
        return "unknown" if v in (None, "", "None") else str(v)
    df["formality_band"] = df["nuance_attributes"].map(_form)
    df["aspect_set"] = df["aspects"].map(lambda d: frozenset(d.keys()))
    return df


def split_retained_control(df: pd.DataFrame, retained_frac: float = 0.5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retained = top-frac of the whole corpus by audit row_score (ties broken by
    row_id, deterministic). Control pool = complementary set."""
    n_ret = int(round(len(df) * retained_frac))
    ordered = df.sort_values(["row_score", "row_id"], ascending=[False, True]).reset_index(drop=True)
    retained = ordered.iloc[:n_ret].copy()
    control_pool = ordered.iloc[n_ret:].copy()
    return retained, control_pool


def greedy_covariate_match(retained: pd.DataFrame, control_pool: pd.DataFrame,
                           matching_seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Greedy 1:1 match: for each retained row (seeded random order) pick an unused
    control-pool row in the same (aspect_count, polarity_comp, length_band,
    formality_band) cell with the highest aspect-set Jaccard. Returns
    (matched_retained, matched_control) in paired order, equal size."""
    cell_cols = ["aspect_count", "polarity_comp", "length_band", "formality_band"]

    # index the control pool by cell
    pool_by_cell: dict = {}
    for pos, (_, r) in enumerate(control_pool.iterrows()):
        key = tuple(r[c] for c in cell_cols)
        pool_by_cell.setdefault(key, []).append(pos)
    ctrl_rows = control_pool.reset_index(drop=True)
    ctrl_sets = ctrl_rows["aspect_set"].tolist()
    used = np.zeros(len(ctrl_rows), dtype=bool)

    rng = np.random.default_rng(matching_seed)
    ret_rows = retained.reset_index(drop=True)
    order = rng.permutation(len(ret_rows))

    ret_keep, ctrl_keep = [], []
    for ri in order:
        r = ret_rows.iloc[ri]
        key = tuple(r[c] for c in cell_cols)
        cands = pool_by_cell.get(key, [])
        best_pos, best_j = -1, -2.0
        rset = r["aspect_set"]
        for pos in cands:
            if used[pos]:
                continue
            cs = ctrl_sets[pos]
            union = len(rset | cs)
            jac = (len(rset & cs) / union) if union else 1.0
            if jac > best_j:
                best_j, best_pos = jac, pos
        if best_pos >= 0:
            used[best_pos] = True
            ret_keep.append(ri)
            ctrl_keep.append(best_pos)

    matched_retained = ret_rows.iloc[ret_keep].reset_index(drop=True)
    matched_control = ctrl_rows.iloc[ctrl_keep].reset_index(drop=True)
    return matched_retained, matched_control


# ----------------------------------------------------------------- balance report
def _marginal(df: pd.DataFrame, col: str) -> dict:
    vc = df[col].value_counts(normalize=True)
    return {str(k): round(float(v), 4) for k, v in vc.items()}


def _aspect_prevalence(df: pd.DataFrame, aspects: list) -> dict:
    out = {}
    n = len(df)
    for a in aspects:
        out[a] = round(float(df["aspect_set"].map(lambda s: a in s).sum()) / max(n, 1), 4)
    return out


def _polarity_token_marginal(df: pd.DataFrame) -> dict:
    from collections import Counter
    c = Counter()
    for d in df["aspects"]:
        c.update(d.values())
    tot = sum(c.values())
    return {k: round(v / tot, 4) for k, v in c.items()}


def balance_block(df: pd.DataFrame, aspects_all: list) -> dict:
    return {
        "n": int(len(df)),
        "row_score_mean": round(float(df["row_score"].mean()), 4),
        "aspect_count": _marginal(df, "aspect_count"),
        "polarity_comp": _marginal(df, "polarity_comp"),
        "length_band": _marginal(df, "length_band"),
        "formality_band": _marginal(df, "formality_band"),
        "polarity_token_marginal": _polarity_token_marginal(df),
        "aspect_prevalence": _aspect_prevalence(df, aspects_all),
        "word_count_mean": round(float(df["word_count"].mean()), 2),
    }


def max_abs_marginal_gap(a: dict, b: dict) -> float:
    keys = set(a) | set(b)
    return round(max(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys), 4)


# ----------------------------------------------------------------- train / eval
def _train_arm(pool9: pd.DataFrame, aspects: list, cfg: "eng.Config", seed: int):
    """set_seed(seed) then carve train/calib from the 9-aspect matched pool and train
    detection + sentiment heads (paired: identical init/shuffle across arms)."""
    eng.set_seed(seed)
    tr, cal, _ = eng.three_way_split(pool9, cfg.split_calib, cfg.split_test, seed)
    det, det_tok = eng.train_detection("bert-base-uncased", tr, cal, aspects, cfg)
    thr = eng.calibrate_thresholds(det, cal, det_tok, aspects, cfg)
    eng.set_seed(seed)
    sen, sen_tok = eng.train_sentiment("bert-base-uncased", tr, cal, aspects, cfg)
    return det, det_tok, thr, sen, sen_tok, len(tr), len(cal)


def _common_gold_metrics(det, det_tok, thr, sen, sen_tok, real9: pd.DataFrame,
                         aspects: list, cfg: "eng.Config") -> dict:
    """Detection micro-F1 and sentiment MSE on the COMMON GOLD set (gold-present
    aspects of the real Herath eval), identical for every model."""
    det_loader = DataLoader(eng.DetectionDataset(real9, det_tok, aspects, cfg.max_len),
                            batch_size=cfg.batch_size, shuffle=False)
    det_probs, det_true = eng.collect_detection(det, det_loader, cfg.device)
    thr_vec = np.array([thr[a] for a in aspects], dtype=np.float32)
    det_preds = (det_probs >= thr_vec).astype(int)
    micro_f1 = float(f1_score(det_true.ravel(), det_preds.ravel(), zero_division=0))
    micro_p = float(precision_score(det_true.ravel(), det_preds.ravel(), zero_division=0))
    micro_r = float(recall_score(det_true.ravel(), det_preds.ravel(), zero_division=0))

    sen_loader = DataLoader(eng.SentimentDataset(real9, sen_tok, aspects, cfg.max_len),
                            batch_size=cfg.batch_size, shuffle=False)
    sent_preds, sent_tgt, sent_mask = eng.collect_sentiment(sen, sen_loader, cfg.device)
    # mask == gold-present cells (SentimentDataset builds it from gold aspects) -> common gold MSE
    gold_mse = float(eng.masked_mse_numpy(sent_preds, sent_tgt, sent_mask))
    n_gold_cells = int(sent_mask.sum())
    return {
        "detection_micro_f1": round(micro_f1, 4),
        "detection_micro_precision": round(micro_p, 4),
        "detection_micro_recall": round(micro_r, 4),
        "sentiment_mse_common_gold": round(gold_mse, 6),
        "n_common_gold_cells": n_gold_cells,
    }


def sanity_full_corpus_f1(cov: pd.DataFrame, real9: pd.DataFrame, aspects: list,
                          cfg: "eng.Config", seed: int) -> dict:
    """Full-corpus (9-aspect) synthetic -> Herath detection micro-F1. Provenance gate."""
    full9 = restrict_to_overlap(cov, aspects)
    eng.set_seed(seed)
    tr, cal, _ = eng.three_way_split(full9, cfg.split_calib, cfg.split_test, seed)
    det, det_tok = eng.train_detection("bert-base-uncased", tr, cal, aspects, cfg)
    thr = eng.calibrate_thresholds(det, cal, det_tok, aspects, cfg)
    det_loader = DataLoader(eng.DetectionDataset(real9, det_tok, aspects, cfg.max_len),
                            batch_size=cfg.batch_size, shuffle=False)
    det_probs, det_true = eng.collect_detection(det, det_loader, cfg.device)
    thr_vec = np.array([thr[a] for a in aspects], dtype=np.float32)
    det_preds = (det_probs >= thr_vec).astype(int)
    f1 = float(f1_score(det_true.ravel(), det_preds.ravel(), zero_division=0))
    ok = SANITY_LO <= f1 <= SANITY_HI
    return {"full_corpus_herath_micro_f1": round(f1, 4), "n_full9_rows": int(len(full9)),
            "expected_range": [SANITY_LO, SANITY_HI], "sanity_ok": bool(ok)}


# ----------------------------------------------------------------- main
def run(seed: int, out_dir: Path, corpus_path: Path, scores_csv: Path,
        herath_path: Path, matching_seed: int, do_sanity: bool) -> dict:
    cfg = eng.Config(seed=seed)
    eng.log_event(f"[E2 seed={seed}] device={cfg.device} matching_seed={matching_seed}")
    aspects = list(OVERLAP_9)

    # ---- data prep (deterministic, data-only) ----
    cov = build_covariate_frame(corpus_path, scores_csv)
    retained, control_pool = split_retained_control(cov, retained_frac=0.5)
    eng.log_event(f"[E2] retained={len(retained)} (score_mean={retained['row_score'].mean():.3f}) "
                  f"control_pool={len(control_pool)} (score_mean={control_pool['row_score'].mean():.3f})")

    matched_retained, matched_control = greedy_covariate_match(retained, control_pool, matching_seed)
    eng.log_event(f"[E2] matched pairs={len(matched_retained)} "
                  f"(retained score_mean={matched_retained['row_score'].mean():.3f} "
                  f"control score_mean={matched_control['row_score'].mean():.3f})")

    aspects_all = eng.discover_aspects(cov)
    pre = {"retained_full": balance_block(retained, aspects_all),
           "control_pool_full": balance_block(control_pool, aspects_all)}
    post = {"retained_matched": balance_block(matched_retained, aspects_all),
            "control_matched": balance_block(matched_control, aspects_all)}
    balance = {
        "pre_matching": pre,
        "post_matching": post,
        "max_abs_marginal_gap": {
            "aspect_count_pre": max_abs_marginal_gap(pre["retained_full"]["aspect_count"], pre["control_pool_full"]["aspect_count"]),
            "aspect_count_post": max_abs_marginal_gap(post["retained_matched"]["aspect_count"], post["control_matched"]["aspect_count"]),
            "polarity_comp_pre": max_abs_marginal_gap(pre["retained_full"]["polarity_comp"], pre["control_pool_full"]["polarity_comp"]),
            "polarity_comp_post": max_abs_marginal_gap(post["retained_matched"]["polarity_comp"], post["control_matched"]["polarity_comp"]),
            "length_band_pre": max_abs_marginal_gap(pre["retained_full"]["length_band"], pre["control_pool_full"]["length_band"]),
            "length_band_post": max_abs_marginal_gap(post["retained_matched"]["length_band"], post["control_matched"]["length_band"]),
            "formality_band_pre": max_abs_marginal_gap(pre["retained_full"]["formality_band"], pre["control_pool_full"]["formality_band"]),
            "formality_band_post": max_abs_marginal_gap(post["retained_matched"]["formality_band"], post["control_matched"]["formality_band"]),
            "aspect_prevalence_pre": max_abs_marginal_gap(pre["retained_full"]["aspect_prevalence"], pre["control_pool_full"]["aspect_prevalence"]),
            "aspect_prevalence_post": max_abs_marginal_gap(post["retained_matched"]["aspect_prevalence"], post["control_matched"]["aspect_prevalence"]),
        },
    }

    # ---- restrict matched arms to the 9 overlap aspects for training ----
    retained_train = restrict_to_overlap(matched_retained, aspects)
    control_train = restrict_to_overlap(matched_control, aspects)
    eng.log_event(f"[E2] overlap-9 restricted: retained_train={len(retained_train)} "
                  f"control_train={len(control_train)}")
    balance["overlap9_survival"] = {"retained_train_n": int(len(retained_train)),
                                    "control_train_n": int(len(control_train))}

    # ---- real Herath common gold eval set (correct mapping) ----
    real = load_real_from_mapped_jsonl(herath_path)
    real9 = restrict_to_overlap(real, aspects)
    eng.log_event(f"[E2] Herath rows={len(real)} overlap-9 rows={len(real9)}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- retained arm ----
    eng.log_event(f"[E2 seed={seed}] === RETAINED arm ===")
    r_det, r_dtok, r_thr, r_sen, r_stok, r_ntr, r_ncal = _train_arm(retained_train, aspects, cfg, seed)
    retained_metrics = _common_gold_metrics(r_det, r_dtok, r_thr, r_sen, r_stok, real9, aspects, cfg)
    retained_metrics.update({"n_train": r_ntr, "n_calib": r_ncal})
    eng.log_event(f"[E2 seed={seed}] RETAINED common-gold micro_f1={retained_metrics['detection_micro_f1']} "
                  f"sent_mse={retained_metrics['sentiment_mse_common_gold']}")
    del r_det, r_sen

    # ---- control arm ----
    eng.log_event(f"[E2 seed={seed}] === CONTROL arm (covariate-matched) ===")
    c_det, c_dtok, c_thr, c_sen, c_stok, c_ntr, c_ncal = _train_arm(control_train, aspects, cfg, seed)
    control_metrics = _common_gold_metrics(c_det, c_dtok, c_thr, c_sen, c_stok, real9, aspects, cfg)
    control_metrics.update({"n_train": c_ntr, "n_calib": c_ncal})
    eng.log_event(f"[E2 seed={seed}] CONTROL common-gold micro_f1={control_metrics['detection_micro_f1']} "
                  f"sent_mse={control_metrics['sentiment_mse_common_gold']}")
    del c_det, c_sen

    result = {
        "experiment": "E2_covariate_matched_filtering",
        "seed": seed,
        "matching_seed": matching_seed,
        "aspects": aspects,
        "n_matched_pairs": int(len(matched_retained)),
        "covariate_balance": balance,
        "common_gold_eval": {
            "herath_overlap9_rows": int(len(real9)),
            "retained": retained_metrics,
            "control": control_metrics,
            "delta_sentiment_mse_retained_minus_control": round(
                retained_metrics["sentiment_mse_common_gold"] - control_metrics["sentiment_mse_common_gold"], 6),
            "delta_detection_f1_retained_minus_control": round(
                retained_metrics["detection_micro_f1"] - control_metrics["detection_micro_f1"], 4),
        },
        "win_definition": ("filtering gain = retained sentiment MSE LOWER than covariate-matched "
                           "control on the common gold set (negative delta_sentiment_mse)"),
    }

    if do_sanity:
        eng.log_event(f"[E2 seed={seed}] === SANITY: full-corpus -> Herath detection F1 ===")
        result["sanity_gate"] = sanity_full_corpus_f1(cov, real9, aspects, cfg, seed)
        eng.log_event(f"[E2 seed={seed}] SANITY full-corpus Herath micro_f1="
                      f"{result['sanity_gate']['full_corpus_herath_micro_f1']} "
                      f"ok={result['sanity_gate']['sanity_ok']}")

    (out_dir / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", required=True)
    p.add_argument("--corpus-path", default="/app/data/generated_reviews.jsonl")
    p.add_argument("--scores-csv", default="/app/data/per_row_scores.csv")
    p.add_argument("--herath-mapped-jsonl", default="/app/data/herath_mapped_real_reviews.jsonl")
    p.add_argument("--matching-seed", type=int, default=42)
    p.add_argument("--sanity", action="store_true")
    args = p.parse_args()

    eng.configure_console_encoding()
    eng.ensure_dirs()
    out_dir = Path(args.out)
    t0 = time.time()
    res = run(args.seed, out_dir, Path(args.corpus_path), Path(args.scores_csv),
              Path(args.herath_mapped_jsonl), args.matching_seed, args.sanity)
    res["elapsed_seconds"] = round(time.time() - t0, 1)
    (out_dir / "result.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print("RESULT_JSON_BEGIN")
    print(json.dumps(res, indent=2, ensure_ascii=False))
    print("RESULT_JSON_END")


if __name__ == "__main__":
    main()
