"""Aggregate the validation-v5 experiment battery into summary JSON + CSV.

Scans paper/experiment_rounds/validation_v5_20260605/<RUN>/result.json for every run
pulled off the Modal volume, plus the local V2 floor, and emits:
  - aggregate_summary.json   (per-experiment means, 95% CIs, per-seed values)
  - aggregate_summary.csv     (flat one-row-per-metric table)

95% CIs use the small-sample t distribution (n-1 dof) on the per-seed micro-F1.
No external scipy dependency: t critical values are tabled for small n.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent.resolve()
ROUND = HERE / "experiment_rounds" / "validation_v5_20260605"

# two-sided 95% t critical values by dof (n-1)
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
       8: 2.306, 9: 2.262, 10: 2.228}


def mean_ci(vals):
    n = len(vals)
    m = sum(vals) / n
    if n < 2:
        return m, None, None, 0.0
    var = sum((v - m) ** 2 for v in vals) / (n - 1)
    sd = math.sqrt(var)
    se = sd / math.sqrt(n)
    tc = T95.get(n - 1, 1.96)
    return m, m - tc * se, m + tc * se, sd


def load_results():
    """run_name -> result.json dict, for every <RUN>/result.json under ROUND."""
    out = {}
    for d in sorted(ROUND.iterdir()):
        if not d.is_dir():
            continue
        rj = d / "result.json"
        if rj.exists():
            try:
                out[d.name] = json.loads(rj.read_text(encoding="utf-8"))
            except Exception as e:
                out[d.name] = {"_error": str(e)}
    return out


def main():
    runs = load_results()
    agg = {}
    csv_rows = []

    # group by experiment family
    fam = {}
    for name, r in runs.items():
        exp = r.get("experiment", "unknown")
        fam.setdefault(exp, []).append((name, r))

    # ---- V1 permutation control
    if "V1_permutation_control" in fam:
        vals = [r["permuted_micro_f1"] for _, r in fam["V1_permutation_control"]]
        seeds = [r["seed"] for _, r in fam["V1_permutation_control"]]
        m, lo, hi, sd = mean_ci(vals)
        ref = fam["V1_permutation_control"][0][1]["real_label_reference_micro_f1"]
        agg["V1_permutation_control"] = {
            "seeds": seeds, "permuted_micro_f1_per_seed": vals,
            "permuted_micro_f1_mean": m, "ci95": [lo, hi], "sd": sd,
            "real_label_reference": ref, "gap_real_minus_permuted_mean": round(ref - m, 4)}
        csv_rows.append({"experiment": "V1_permutation_control", "metric": "permuted_micro_f1",
                         "n_seeds": len(vals), "mean": m, "ci95_lo": lo, "ci95_hi": hi, "sd": sd,
                         "reference": ref, "per_seed": vals})

    # ---- V4 clean-label ceiling (two variants)
    if "V4_clean_label_ceiling" in fam:
        for variant in ("row_score_eq_1p0", "row_score_ge_0p5"):
            vals, seeds, ntests = [], [], []
            for _, r in fam["V4_clean_label_ceiling"]:
                v = r["variants"].get(variant)
                if v:
                    vals.append(v["micro_f1"]); seeds.append(r["seed"]); ntests.append(v["n_test"])
            if not vals:
                continue
            m, lo, hi, sd = mean_ci(vals)
            ref = fam["V4_clean_label_ceiling"][0][1]["full_data_reference_micro_f1"]
            agg[f"V4_{variant}"] = {
                "seeds": seeds, "micro_f1_per_seed": vals, "n_test_per_seed": ntests,
                "micro_f1_mean": m, "ci95": [lo, hi], "sd": sd,
                "full_data_reference": ref, "jump_over_full_mean": round(m - ref, 4)}
            csv_rows.append({"experiment": f"V4_{variant}", "metric": "clean_micro_f1",
                             "n_seeds": len(vals), "mean": m, "ci95_lo": lo, "ci95_hi": hi, "sd": sd,
                             "reference": ref, "per_seed": vals})

    # ---- V3 learning curve (per size, mean across seeds)
    if "V3_learning_curve" in fam:
        by_size = {}
        seeds = sorted({r["seed"] for _, r in fam["V3_learning_curve"]})
        for _, r in fam["V3_learning_curve"]:
            for pt in r["points"]:
                by_size.setdefault(pt["train_size"], []).append(pt["micro_f1"])
        curve = []
        for size in sorted(by_size):
            vals = by_size[size]
            m, lo, hi, sd = mean_ci(vals)
            curve.append({"train_size": size, "micro_f1_per_seed": vals,
                          "micro_f1_mean": m, "ci95": [lo, hi], "sd": sd})
            csv_rows.append({"experiment": "V3_learning_curve", "metric": f"micro_f1@{size}",
                             "n_seeds": len(vals), "mean": m, "ci95_lo": lo, "ci95_hi": hi, "sd": sd,
                             "reference": None, "per_seed": vals})
        agg["V3_learning_curve"] = {"seeds": seeds, "curve": curve}

    # ---- B5 synth->real fine-tune
    if "B5_synth_to_real_finetune" in fam:
        vals = [r["synth_to_real_micro_f1"] for _, r in fam["B5_synth_to_real_finetune"]]
        seeds = [r["seed"] for _, r in fam["B5_synth_to_real_finetune"]]
        m, lo, hi, sd = mean_ci(vals)
        r0 = fam["B5_synth_to_real_finetune"][0][1]
        agg["B5_synth_to_real_finetune"] = {
            "seeds": seeds, "micro_f1_per_seed": vals,
            "micro_f1_mean": m, "ci95": [lo, hi], "sd": sd,
            "ref_synthetic_only_transfer": r0["reference_synthetic_only_transfer_micro_f1"],
            "ref_real_only_training": r0["reference_real_only_training_micro_f1"],
            "delta_over_real_only_mean": round(m - r0["reference_real_only_training_micro_f1"], 4),
            "delta_over_synth_only_mean": round(m - r0["reference_synthetic_only_transfer_micro_f1"], 4)}
        csv_rows.append({"experiment": "B5_synth_to_real_finetune", "metric": "synth_to_real_micro_f1",
                         "n_seeds": len(vals), "mean": m, "ci95_lo": lo, "ci95_hi": hi, "sd": sd,
                         "reference": r0["reference_real_only_training_micro_f1"], "per_seed": vals})

    # ---- V2 trivial floor (local; pass through)
    if "V2_trivial_baseline_floor" in fam:
        r = fam["V2_trivial_baseline_floor"][0][1]
        agg["V2_trivial_baseline_floor"] = {
            "seed": r["seed"], "n_test_rows": r["n_test_rows"], "floors": r["floors"],
            "paper_bert_micro_f1": r["paper_bert_micro_f1"]}
        for name, f in r["floors"].items():
            mf1 = f.get("micro_f1", f.get("micro_f1_mean"))
            csv_rows.append({"experiment": "V2_trivial_floor", "metric": name, "n_seeds": 1,
                             "mean": mf1, "ci95_lo": None, "ci95_hi": None, "sd": None,
                             "reference": r["paper_bert_micro_f1"], "per_seed": [mf1]})

    (ROUND / "aggregate_summary.json").write_text(
        json.dumps(agg, indent=2, ensure_ascii=False), encoding="utf-8")
    if csv_rows:
        pd.DataFrame(csv_rows).to_csv(ROUND / "aggregate_summary.csv", index=False)
    print(json.dumps(agg, indent=2, ensure_ascii=False))
    print(f"\n[aggregate] {len(runs)} runs -> {ROUND/'aggregate_summary.json'}")


if __name__ == "__main__":
    main()
