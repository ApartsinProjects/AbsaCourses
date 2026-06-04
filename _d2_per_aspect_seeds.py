"""Diagnostic: per-aspect threshold and F1 stability across (bucket, seed).

For each Herath-overlap aspect, show mean ± std across seeds for each bucket.
Helps diagnose whether the bucket-level micro-F1 differences are driven by
threshold-calibration noise on dominant aspects (lecturer_quality, etc.).
"""
from __future__ import annotations
import csv, statistics
from pathlib import Path
from datetime import datetime, timezone

HERE = Path(__file__).parent.resolve()
PHASE_OUT = HERE / "paper" / "experiment_rounds" / f"phase_d2_filtering_{datetime.now(timezone.utc).strftime('%Y%m%d')}"
RUNS_LOCAL = PHASE_OUT / "runs"
TABLES_DIR = HERE / "paper" / "outputs" / "tables"
BUCKET_NAMES = ["top25", "top50", "full", "bot25", "random_5k"]
SEEDS = [42, 17, 23, 41]


def find_pa(bucket: str, seed: int) -> Path | None:
    d = RUNS_LOCAL / (bucket if seed == 42 else f"{bucket}_seed{seed}")
    for cand in (d / "run" / "per_aspect.csv", d / "per_aspect.csv"):
        if cand.exists(): return cand
    cands = list(d.rglob("per_aspect.csv"))
    return cands[0] if cands else None


def main() -> int:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    data = {}
    for b in BUCKET_NAMES:
        for s in SEEDS:
            p = find_pa(b, s)
            if not p: continue
            with p.open(encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    if r.get("approach") != "bert-base-uncased": continue
                    asp = r["aspect"]
                    data.setdefault((b, asp), []).append({
                        "seed": s,
                        "f1": float(r["f1"]),
                        "threshold": float(r["threshold"]),
                        "precision": float(r["precision"]),
                        "recall": float(r["recall"]),
                    })

    # Aspects in support order from one of the runs
    aspects = sorted({asp for (_b, asp) in data.keys()})
    out_csv = TABLES_DIR / "phase_d2_filtering_per_aspect_seeds.csv"
    rows = []
    print(f"{'aspect':<25s} {'bucket':<11s} {'n':>3s} {'f1':>16s} {'threshold':>16s}")
    for asp in aspects:
        for b in BUCKET_NAMES:
            entries = data.get((b, asp), [])
            if not entries: continue
            f1s = [e["f1"] for e in entries]
            ts = [e["threshold"] for e in entries]
            f1_mean = sum(f1s)/len(f1s)
            f1_std = statistics.stdev(f1s) if len(f1s)>=2 else 0
            t_mean = sum(ts)/len(ts)
            t_std = statistics.stdev(ts) if len(ts)>=2 else 0
            rows.append({"aspect": asp, "bucket": b, "n_seeds": len(entries),
                         "f1_mean": round(f1_mean,3), "f1_std": round(f1_std,3),
                         "threshold_mean": round(t_mean,3), "threshold_std": round(t_std,3)})
            print(f"  {asp:<23s} {b:<10s} {len(entries):>3d}  {f1_mean:.3f}±{f1_std:.3f}   {t_mean:.3f}±{t_std:.3f}")
        print()

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["aspect","bucket","n_seeds","f1_mean","f1_std","threshold_mean","threshold_std"])
        w.writeheader()
        for r in rows: w.writerow(r)
    print(f"\nwrote: {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
