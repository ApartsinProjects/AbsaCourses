"""Aggregate the reviewer A/B Modal runs across seeds into means + 95% CIs.

Reads every paper/experiment_rounds/reviewer_AB_20260605/modal_summary_*.json
(each a list of {experiment, seed, headline}), de-dups by (experiment, seed)
keeping the latest, and reports:

  A: full-test micro-F1, faithful-only (row_score==1) micro-F1, and the
     (faithful - full) delta, mean +- std with a normal 95% CI over seeds.
  B: real-Herath-trained 9-aspect micro-F1, mean +- std + 95% CI, vs the
     synthetic-only reference 0.4593.

Writes a unique summary JSON + CSV; prints a paste-ready numbers block.
"""
import csv
import glob
import json
import math
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "experiment_rounds", "reviewer_AB_20260605")


def ci95(xs):
    if len(xs) < 2:
        return (xs[0], xs[0]) if xs else (float("nan"), float("nan"))
    m = st.mean(xs)
    se = st.stdev(xs) / math.sqrt(len(xs))
    # small-sample t for n=4 -> df=3, t.975=3.182; n=3 -> 4.303; n=2 -> 12.706
    t = {1: float("nan"), 2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365}.get(len(xs), 2.0)
    return (m - t * se, m + t * se)


def collect():
    byseed = {}  # (exp, seed) -> headline (latest wins by filename sort)
    for f in sorted(glob.glob(os.path.join(OUT, "modal_summary_*.json"))):
        try:
            data = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        for rec in data if isinstance(data, list) else [data]:
            if not isinstance(rec, dict) or not rec.get("ok", True):
                continue
            h = rec.get("headline") or {}
            exp = rec.get("experiment") or h.get("experiment", "")
            seed = rec.get("seed", h.get("seed"))
            if exp and seed is not None and h:
                byseed[(str(exp)[:1], int(seed))] = h
    return byseed


def main():
    bs = collect()
    A = {s: h for (e, s), h in bs.items() if e == "A"}
    B = {s: h for (e, s), h in bs.items() if e == "B"}
    out = {"experiment_A": {}, "experiment_B": {}}

    if A:
        full = [A[s]["full_test"]["micro_f1"] for s in sorted(A)]
        faith = [A[s]["faithful_only_row_score_eq_1"]["micro_f1"] for s in sorted(A)]
        delta = [A[s].get("delta_faithful_minus_full", faith[i] - full[i]) for i, s in enumerate(sorted(A))]
        nfaith = [A[s]["faithful_only_row_score_eq_1"]["n_rows"] for s in sorted(A)]
        out["experiment_A"] = {
            "seeds": sorted(A),
            "full_test_micro_f1_mean": round(st.mean(full), 4), "full_test_micro_f1_ci": [round(x, 4) for x in ci95(full)],
            "faithful_only_micro_f1_mean": round(st.mean(faith), 4), "faithful_only_micro_f1_ci": [round(x, 4) for x in ci95(faith)],
            "delta_mean": round(st.mean(delta), 4), "delta_ci": [round(x, 4) for x in ci95(delta)],
            "delta_wins": f"{sum(1 for d in delta if d > 0)}/{len(delta)}",
            "faithful_subset_n_rows_mean": round(st.mean(nfaith), 1),
            "per_seed_full": dict(zip(sorted(A), [round(x, 4) for x in full])),
            "per_seed_faithful": dict(zip(sorted(A), [round(x, 4) for x in faith])),
        }
    if B:
        rf = [B[s]["real_test_micro_f1"] for s in sorted(B)]
        out["experiment_B"] = {
            "seeds": sorted(B),
            "real_trained_micro_f1_mean": round(st.mean(rf), 4), "real_trained_micro_f1_ci": [round(x, 4) for x in ci95(rf)],
            "synthetic_only_reference": 0.4593,
            "per_seed": dict(zip(sorted(B), [round(x, 4) for x in rf])),
        }

    jp = os.path.join(OUT, "AB_aggregate_summary.json")
    json.dump(out, open(jp, "w", encoding="utf-8"), indent=2)
    print(json.dumps(out, indent=2))
    print("\nwrote", jp)


if __name__ == "__main__":
    main()
