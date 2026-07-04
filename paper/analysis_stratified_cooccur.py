"""Stratified + prior-corrected benchmark analysis, and aspect co-occurrence
structure (real vs synthetic).

A) Stratify the in-domain BERT benchmark (synthetic test split) by opinion
   consistency {single, consistent-multi, mixed-multi}; report detection micro-F1
   and gold-present sentiment MSE per stratum; then post-stratify (importance
   weight) to the REAL stratum distribution and report the corrected estimate + ESS.
B) Aspect co-occurrence: do some aspects co-occur far more/less than independence
   predicts, and does synthetic (uniform aspect sampling) reproduce that structure?
C) Control demonstration: reweight the benchmark detection metric jointly to the
   real (aspect-count x consistency) distribution.
"""
from __future__ import annotations

import itertools
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BERT = ROOT / "paper/benchmark_outputs/runs/benchmark_full_20260403T173714Z/artifacts/bert-base-uncased_sample_predictions.jsonl"
REAL = {
    "herath": ROOT / "paper/real_transfer/herath_mapped_real_reviews.jsonl",
    "edurabsa": ROOT / "external_data/EduRABSA_mapped/edurabsa_all_mapped.jsonl",
    "oats": ROOT / "external_data/OATS_coursera/oats_mapped.jsonl",
}
OUT = ROOT / "paper/outputs/stratified_cooccur.json"
POL2NUM = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}


def stratum(aspmap: dict) -> str:
    pols = list(aspmap.values())
    if len(pols) <= 1:
        return "single"
    return "consistent" if len(set(pols)) == 1 else "mixed"


def load_real_maps(path):
    out = []
    for l in open(path, encoding="utf-8"):
        a = json.loads(l).get("aspects") or {}
        if a:
            out.append(a)
    return out


# ---------- real distributions ----------
real_maps = {k: load_real_maps(p) for k, p in REAL.items()}
real_strata = {k: Counter(stratum(m) for m in v) for k, v in real_maps.items()}
# pooled real stratum distribution (equal weight per corpus)
real_dist = defaultdict(float)
for k, cnt in real_strata.items():
    tot = sum(cnt.values())
    for s in ("single", "consistent", "mixed"):
        real_dist[s] += cnt[s] / tot / len(real_strata)
real_dist = {s: real_dist[s] for s in ("single", "consistent", "mixed")}


# ---------- Part A: stratified + prior-corrected benchmark ----------
recs = [json.loads(l) for l in open(BERT, encoding="utf-8")]
by = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "se": [], "n": 0})
for r in recs:
    gold = r.get("gold_aspects") or {}
    s = stratum(gold)
    dp, dt = r["detection_predictions"], r["detection_targets"]
    for a in dt:
        p, t = int(dp.get(a, 0)), int(dt[a])
        by[s]["tp"] += p & t
        by[s]["fp"] += p & (1 - t)
        by[s]["fn"] += (1 - p) & t
    sp, st = r.get("sentiment_prediction_values") or {}, r.get("sentiment_target_values") or {}
    for a, tv in st.items():
        if a in sp:
            by[s]["se"].append((float(sp[a]) - float(tv)) ** 2)
    by[s]["n"] += 1


def micro_f1(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


synth_n = {s: by[s]["n"] for s in ("single", "consistent", "mixed")}
synth_tot = sum(synth_n.values())
synth_dist = {s: synth_n[s] / synth_tot for s in synth_n}

partA = {"per_stratum": {}, "real_stratum_dist": real_dist, "synth_stratum_dist": synth_dist}
for s in ("single", "consistent", "mixed"):
    b = by[s]
    partA["per_stratum"][s] = {
        "n": b["n"], "det_micro_f1": round(micro_f1(b["tp"], b["fp"], b["fn"]), 4),
        "gold_sent_mse": round(float(np.mean(b["se"])) if b["se"] else float("nan"), 4),
    }
# raw pooled
raw_tp = sum(by[s]["tp"] for s in synth_n); raw_fp = sum(by[s]["fp"] for s in synth_n); raw_fn = sum(by[s]["fn"] for s in synth_n)
raw_se = [e for s in synth_n for e in by[s]["se"]]
# prior-corrected: weight each stratum's tp/fp/fn and per-review MSE by w=P_real/P_synth
w = {s: (real_dist[s] / synth_dist[s] if synth_dist[s] else 0.0) for s in synth_n}
rw_tp = sum(by[s]["tp"] * w[s] for s in synth_n)
rw_fp = sum(by[s]["fp"] * w[s] for s in synth_n)
rw_fn = sum(by[s]["fn"] * w[s] for s in synth_n)
rw_se = sum(np.sum(by[s]["se"]) * w[s] for s in synth_n) / sum(len(by[s]["se"]) * w[s] for s in synth_n)
# effective sample size of the review-level reweighting
wr = np.array([w[stratum(r.get("gold_aspects") or {})] for r in recs])
ess = float(wr.sum() ** 2 / (wr ** 2).sum())
partA["raw"] = {"det_micro_f1": round(micro_f1(raw_tp, raw_fp, raw_fn), 4), "gold_sent_mse": round(float(np.mean(raw_se)), 4)}
partA["prior_corrected_to_real"] = {"det_micro_f1": round(micro_f1(rw_tp, rw_fp, rw_fn), 4), "gold_sent_mse": round(float(rw_se), 4),
                                     "weights": {s: round(w[s], 3) for s in w}, "ess": round(ess, 0), "n": len(recs)}


# ---------- Part B: aspect co-occurrence structure ----------
def cooccur_stats(maps):
    n = len(maps)
    marg = Counter()
    pair = Counter()
    for m in maps:
        aset = list(m.keys())
        for a in aset:
            marg[a] += 1
        for a, b in itertools.combinations(sorted(aset), 2):
            pair[(a, b)] += 1
    lifts = {}
    for (a, b), c in pair.items():
        pa, pb, pab = marg[a] / n, marg[b] / n, c / n
        if pa > 0 and pb > 0 and c >= 8:  # support floor
            lifts[(a, b)] = pab / (pa * pb)
    logl = np.array([math.log(v) for v in lifts.values()]) if lifts else np.array([0.0])
    return {"n_reviews": n, "n_pairs_with_support": len(lifts),
            "log_lift_std": round(float(logl.std()), 3), "log_lift_mean": round(float(logl.mean()), 3),
            "frac_pairs_lift_gt_1p5": round(float(np.mean([v > 1.5 for v in lifts.values()])) if lifts else 0.0, 3),
            "frac_pairs_lift_lt_0p67": round(float(np.mean([v < 0.667 for v in lifts.values()])) if lifts else 0.0, 3),
            "top_over": sorted(lifts.items(), key=lambda x: -x[1])[:6],
            "top_under": sorted(lifts.items(), key=lambda x: x[1])[:6]}


# synthetic aspect maps from the benchmark gold (represents the generated corpus)
synth_maps = [r.get("gold_aspects") or {} for r in recs]
partB = {"synthetic": cooccur_stats(synth_maps)}
for k, v in real_maps.items():
    partB[k] = cooccur_stats(v)


def fmt_pairs(pairs):
    return [f"{a}+{b}={v:.2f}" for (a, b), v in pairs]


out = {"partA_stratified_prior_corrected": partA, "partB_cooccurrence": {
    k: {kk: (fmt_pairs(vv) if kk in ("top_over", "top_under") else vv) for kk, vv in v.items()} for k, v in partB.items()}}
OUT.write_text(json.dumps(out, indent=2))

# ---- console summary ----
print("=== PART A: stratified + prior-corrected (BERT in-domain benchmark) ===")
print(f"real stratum dist (pooled): { {s: round(real_dist[s],2) for s in real_dist} }")
print(f"synth stratum dist        : { {s: round(synth_dist[s],2) for s in synth_dist} }")
for s in ("single", "consistent", "mixed"):
    d = partA["per_stratum"][s]
    print(f"  {s:11s} n={d['n']:5d}  det_microF1={d['det_micro_f1']:.4f}  gold_sentMSE={d['gold_sent_mse']:.4f}")
print(f"  RAW pooled           det_microF1={partA['raw']['det_micro_f1']:.4f}  gold_sentMSE={partA['raw']['gold_sent_mse']:.4f}")
pc = partA["prior_corrected_to_real"]
print(f"  PRIOR-CORRECTED->real det_microF1={pc['det_micro_f1']:.4f}  gold_sentMSE={pc['gold_sent_mse']:.4f}  (ESS={pc['ess']:.0f}/{pc['n']}, w={pc['weights']})")
print("\n=== PART B: aspect co-occurrence structure (log-lift spread; higher = more structured) ===")
for k in ("synthetic", "herath", "edurabsa", "oats"):
    b = partB[k]
    print(f"  {k:9s} log_lift_std={b['log_lift_std']:.3f}  over(>1.5)={b['frac_pairs_lift_gt_1p5']:.2f} under(<0.67)={b['frac_pairs_lift_lt_0p67']:.2f}  (pairs={b['n_pairs_with_support']})")
print(f"  real OATS top co-occurring pairs: {fmt_pairs(partB['oats']['top_over'])[:4]}")
print(f"  synthetic top pairs             : {fmt_pairs(partB['synthetic']['top_over'])[:4]}")
print(f"\nwrote {OUT}")
