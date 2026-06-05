"""Reviewer-response EVIDENCE ITEM 2: detection-set robustness of the
sentiment-MSE contrast.

Co-computes, in ONE pass over the SAME 8 seeds, the per-bucket 8-seed mean of:
  - sentiment_mse_detected  (the headline metric, MSE on detected aspects)
  - micro_recall            (the detection rate that determines the detected set)
  - micro_precision, micro_f1 (for context)
for every (architecture, target, bucket) cell.

Goal: show the headline contrasts (top50 vs random_5k; bot25 vs full) are not
explained by different detection sets -- i.e. compare detection rates side by
side with the MSE so the reader can see whether the MSE-on-detected differences
reflect sentiment quality rather than which aspects got scored.

Reconciliation: BERT-Herath per-bucket MSE means MUST match manuscript Table 8E
(top25=0.389, top50=0.356, full=0.351, bot25=0.710, random_5k=0.411).

Reads ONLY the on-disk per-run summary.csv files. No training, no API.
"""
from __future__ import annotations
import csv, json
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(r"E:\Claude\CourseABSA\hopeful-kowalevski-04ee10")
RUNS = REPO / "paper" / "experiment_rounds" / "phase_d2_filtering_20260604" / "runs"
OUT_CSV = REPO / "paper" / "outputs" / "tables" / "reviewer_response_item2_bucket_mse_recall.csv"
OUT_JSON = REPO / "paper" / "outputs" / "tables" / "reviewer_response_item2_bucket_mse_recall.json"

SEEDS = [17, 23, 41, 42, 53, 89, 101, 137]
BUCKETS = ["top25", "top50", "full", "bot25", "random_5k"]
# (label, dir-suffix). BERT-Herath uses '' suffix, and its seed 42 is the BARE
# bucket dir (no _seed42), all other seeds use _seed{N}. Other combos use an
# explicit _seed{N}{suffix} for every seed including 42.
COMBOS = [
    ("bert", "herath", ""),
    ("distilbert", "herath", "_distilbert"),
    ("bert", "edurabsa", "_edurabsa"),
    ("distilbert", "edurabsa", "_distilbert_edurabsa"),
]

METRICS = ["sentiment_mse_detected", "micro_recall", "micro_precision", "micro_f1"]


def run_dir_for(bucket, seed, suffix):
    """Resolve the on-disk run directory for a (bucket, seed, arch/target)."""
    if suffix == "":  # BERT-Herath
        if seed == 42:
            return RUNS / bucket          # bare dir holds seed 42
        return RUNS / f"{bucket}_seed{seed}"
    return RUNS / f"{bucket}_seed{seed}{suffix}"


def read_summary(d):
    s = pd.read_csv(d / "run" / "summary.csv")
    assert len(s) == 1, f"{d}: expected 1 row"
    return s.iloc[0]


def main():
    rows = []
    raw_per_seed = {}  # for audit trail
    for arch, target, suffix in COMBOS:
        for bucket in BUCKETS:
            vals = {m: [] for m in METRICS}
            seeds_found = []
            for seed in SEEDS:
                d = run_dir_for(bucket, seed, suffix)
                if not (d / "run" / "summary.csv").exists():
                    raise FileNotFoundError(f"missing summary for {arch}/{target}/{bucket}/seed{seed}: {d}")
                r = read_summary(d)
                # sanity: metadata seed should match
                md = json.loads((d / "run" / "metadata.json").read_text())
                assert md["seed"] == seed, f"{d}: metadata seed {md['seed']} != {seed}"
                for m in METRICS:
                    vals[m].append(float(r[m]))
                seeds_found.append(seed)
            assert sorted(seeds_found) == sorted(SEEDS), f"{arch}/{target}/{bucket}: seeds {seeds_found}"
            row = {"architecture": arch, "target": target, "bucket": bucket, "n_seeds": len(SEEDS)}
            for m in METRICS:
                arr = np.array(vals[m])
                row[f"{m}_mean"] = round(float(arr.mean()), 4)
                row[f"{m}_std"] = round(float(arr.std(ddof=1)), 4)
            rows.append(row)
            raw_per_seed[f"{arch}|{target}|{bucket}"] = {m: [round(x, 4) for x in vals[m]] for m in METRICS}

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    # Reconciliation vs Table 8E (BERT-Herath sentiment_mse_detected).
    table8e = {"top25": 0.389, "top50": 0.356, "full": 0.351, "bot25": 0.710, "random_5k": 0.411}
    recon = {}
    bh = df[(df.architecture == "bert") & (df.target == "herath")].set_index("bucket")
    recon_ok = True
    for b, expected in table8e.items():
        got = float(bh.loc[b, "sentiment_mse_detected_mean"])
        diff = round(abs(got - expected), 4)
        ok = bool(diff <= 0.001)
        recon_ok = recon_ok and ok
        recon[b] = {"paper_table8E": expected, "recomputed": got, "abs_diff": diff, "match": ok}

    out = {
        "evidence_item": "2 - detection-set robustness of sentiment-MSE contrast",
        "input_artifact_dir": str(RUNS),
        "seeds": SEEDS,
        "buckets": BUCKETS,
        "table8E_reconciliation_bert_herath": recon,
        "table8E_all_match": recon_ok,
        "per_cell": rows,
        "raw_per_seed": raw_per_seed,
        "contrast_notes": {},
    }

    # Contrast notes: detection-rate comparability for the two headline contrasts.
    for arch, target, _ in COMBOS:
        sub = df[(df.architecture == arch) & (df.target == target)].set_index("bucket")
        key = f"{arch}-{target}"
        f = lambda b_, c_: float(sub.loc[b_, c_])
        out["contrast_notes"][key] = {
            "top50_vs_random_5k": {
                "mse_top50": f("top50", "sentiment_mse_detected_mean"),
                "mse_random_5k": f("random_5k", "sentiment_mse_detected_mean"),
                "recall_top50": f("top50", "micro_recall_mean"),
                "recall_random_5k": f("random_5k", "micro_recall_mean"),
                "recall_abs_diff": round(abs(f("top50", "micro_recall_mean") - f("random_5k", "micro_recall_mean")), 4),
            },
            "bot25_vs_full": {
                "mse_bot25": f("bot25", "sentiment_mse_detected_mean"),
                "mse_full": f("full", "sentiment_mse_detected_mean"),
                "recall_bot25": f("bot25", "micro_recall_mean"),
                "recall_full": f("full", "micro_recall_mean"),
                "recall_abs_diff": round(abs(f("bot25", "micro_recall_mean") - f("full", "micro_recall_mean")), 4),
            },
        }

    OUT_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")

    # Console report
    print("=== EVIDENCE ITEM 2: per-bucket MSE + detection recall (8-seed mean) ===\n")
    pd.set_option("display.width", 200)
    for arch, target, _ in COMBOS:
        sub = df[(df.architecture == arch) & (df.target == target)]
        print(f"--- {arch}-{target} ---")
        print(sub[["bucket", "sentiment_mse_detected_mean", "micro_recall_mean",
                   "micro_precision_mean", "micro_f1_mean"]].to_string(index=False))
        print()
    print("=== Table 8E reconciliation (BERT-Herath sentiment_mse_detected) ===")
    for b, r in recon.items():
        print(f"  {b:10s} paper={r['paper_table8E']:.3f}  recomputed={r['recomputed']:.4f}  "
              f"diff={r['abs_diff']:.4f}  {'OK' if r['match'] else 'MISMATCH'}")
    print(f"\nAll Table 8E match within 0.001: {recon_ok}")
    print(f"\nwrote {OUT_CSV}")
    print(f"wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
