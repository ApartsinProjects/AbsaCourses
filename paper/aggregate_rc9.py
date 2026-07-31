"""Aggregate the RC9 sample-efficiency runs into rc5_finetune_curve_summary.json.

Reads the newest modal_summary_*.json under experiment_rounds/rc9_sample_efficiency/,
computes mean / sample-std / 95% CI (t, n-1 dof) per (arm, N), and writes:
  - outputs/rc9_sample_efficiency_per_seed.csv   (tidy per-run table)
  - outputs/rc5_finetune_curve_summary.json      (updated in place, backward compatible)

Backward-compatible keys preserved:
  curve_by_real_train_n            -> the synth_pretrain_finetune arm (now 5 seeds)
  references.real_only_training    -> 0.7673 (mode B full-data)
New keys:
  real_only_curve_by_real_train_n  -> the real-only-from-scratch arm (5 seeds)
  arms.{arm}.{N}                   -> both arms, full stats
"""
import csv, glob, json, math
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parent
PHASE = ROOT / "experiment_rounds" / "rc9_sample_efficiency"
SUMMARY = ROOT / "outputs" / "rc5_finetune_curve_summary.json"
PERSEED_CSV = ROOT / "outputs" / "rc9_sample_efficiency_per_seed.csv"

T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447}  # two-sided 95%, dof


def _stats(vals):
    n = len(vals)
    m = mean(vals)
    sd = stdev(vals) if n > 1 else 0.0
    ci = (T95.get(n - 1, 1.96) * sd / math.sqrt(n)) if n > 1 else 0.0
    return {"n_seeds": n, "micro_f1_mean": round(m, 4), "micro_f1_std": round(sd, 4),
            "micro_f1_ci95": round(ci, 4)}


def main():
    summ_files = sorted(glob.glob(str(PHASE / "modal_summary_*.json")))
    if not summ_files:
        raise SystemExit(f"no modal_summary_*.json under {PHASE}")
    latest = summ_files[-1]
    print(f"aggregating {latest}")
    runs = json.load(open(latest, encoding="utf-8"))

    # tidy per-run rows + group by (arm, N)
    rows = []
    groups = {}  # (arm, n_key) -> [micro_f1,...]
    for r in runs:
        if not r.get("ok"):
            print(f"  WARN skipping failed run: {r.get('run_name')} rc={r.get('rc')}")
            continue
        h = r["headline"]
        arm = r.get("arm") or h.get("arm")
        n_real = int(h["n_real_train"])
        n_key = str(n_real)
        f1 = float(h["synth_to_real_micro_f1"])
        rows.append({"arm": arm, "requested": r["real_train_n"], "seed": r["seed"],
                     "n_real_train": n_real, "micro_f1": round(f1, 4)})
        groups.setdefault((arm, n_key), []).append(f1)

    rows.sort(key=lambda x: (x["arm"], x["n_real_train"], x["seed"]))
    with PERSEED_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["arm", "requested", "seed", "n_real_train", "micro_f1"])
        w.writeheader(); w.writerows(rows)
    print(f"wrote {PERSEED_CSV} ({len(rows)} rows)")

    arms = {}
    for (arm, n_key), vals in groups.items():
        arms.setdefault(arm, {})[n_key] = _stats(vals)

    synth = arms.get("synth_pretrain_finetune", {})
    real = arms.get("real_only", {})

    # load existing summary to preserve/extend references
    d = json.load(open(SUMMARY, encoding="utf-8"))
    d["curve_by_real_train_n"] = {k: synth[k] for k in sorted(synth, key=int)}
    d["real_only_curve_by_real_train_n"] = {k: real[k] for k in sorted(real, key=int)}
    d["arms"] = {a: {k: arms[a][k] for k in sorted(arms[a], key=int)} for a in arms}
    d["seeds"] = sorted({r["seed"] for r in rows})
    refs = d.setdefault("references", {})
    refs.setdefault("synthetic_only_transfer", 0.4593)
    refs["synthetic_only_transfer_5seed"] = 0.402
    refs.setdefault("real_only_training", 0.7673)
    refs.setdefault("synth_pretrain_full_finetune_paper", 0.784)
    d["provenance"] = {"source": Path(latest).name,
                       "grid": "N in {100,250,500,1000,full} x seed in {17,23,41,42,89} x arm in {real_only, synth_pretrain_finetune}",
                       "protocol": "9-aspect detection micro-F1 on fixed 566-row real Herath test; 283-row calib; bert-base-uncased"}
    SUMMARY.write_text(json.dumps(d, indent=2), encoding="utf-8")
    print(f"wrote {SUMMARY}")

    for arm in ("synth_pretrain_finetune", "real_only"):
        print(f"\n[{arm}]")
        for k in sorted(arms.get(arm, {}), key=int):
            s = arms[arm][k]
            print(f"  N={k:>4}  {s['micro_f1_mean']:.4f} +/- {s['micro_f1_std']:.4f}  (CI95 {s['micro_f1_ci95']:.4f}, n={s['n_seeds']})")


if __name__ == "__main__":
    main()
