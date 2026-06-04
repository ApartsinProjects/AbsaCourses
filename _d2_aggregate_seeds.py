"""Phase D2 multi-seed aggregator.

Pulls all (bucket, seed) results from the `phase-d2-results` Modal volume,
combines them, and computes per-bucket mean ± std and bootstrap CI on the
headline metrics. Writes:
  - phase_d2_filtering_results_multiseed_long.csv  (one row per (bucket, seed))
  - phase_d2_filtering_results_multiseed_summary.csv  (one row per bucket)
  - phase_d2_filtering_micro_f1_multiseed.svg
"""
from __future__ import annotations
import csv, json, math, statistics, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).parent.resolve()
PHASE_OUT = HERE / "paper" / "experiment_rounds" / f"phase_d2_filtering_{datetime.now(timezone.utc).strftime('%Y%m%d')}"
RUNS_LOCAL = PHASE_OUT / "runs"
TABLES_DIR = HERE / "paper" / "outputs" / "tables"
FIGURES_DIR = HERE / "paper" / "outputs" / "figures"
BUCKET_NAMES = ["top25", "top50", "full", "bot25", "random_5k"]
SEEDS_TO_PULL = [42, 17, 23, 41]
MODAL_CLI = r"C:\Users\apart\AppData\Roaming\Python\Python314\Scripts\modal"
VOLUME = "phase-d2-results"


def modal_pull(remote: str, local: Path) -> bool:
    local.parent.mkdir(parents=True, exist_ok=True)
    cmd = [MODAL_CLI, "volume", "get", VOLUME, remote, str(local.parent), "--force"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode == 0


def find_summary(local_dir: Path) -> Path | None:
    for cand in (local_dir / "run" / "summary.csv", local_dir / "summary.csv"):
        if cand.exists(): return cand
    cands = list(local_dir.rglob("summary.csv"))
    return cands[0] if cands else None


def read_summary_row(p: Path) -> dict | None:
    with p.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        if "bert-base-uncased" in r.get("approach",""):
            return r
    return rows[0] if rows else None


def bootstrap_ci(values: list[float], n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple[float,float]:
    import random
    if len(values) < 2: return (values[0] if values else 0.0, values[0] if values else 0.0)
    rng = random.Random(seed)
    n = len(values)
    boots = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        boots.append(sum(sample)/n)
    boots.sort()
    lo = boots[int(alpha/2 * n_boot)]
    hi = boots[int((1-alpha/2) * n_boot)]
    return (lo, hi)


def main() -> int:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    long_csv = TABLES_DIR / "phase_d2_filtering_results_multiseed_long.csv"
    summ_csv = TABLES_DIR / "phase_d2_filtering_results_multiseed_summary.csv"

    long_rows = []
    for b in BUCKET_NAMES:
        for s in SEEDS_TO_PULL:
            # seed-42 dir is just `<bucket>/`; other seeds are `<bucket>_seed<S>/`
            if s == 42:
                remote = f"/{b}"
                local_dir = RUNS_LOCAL / b
            else:
                remote = f"/{b}_seed{s}"
                local_dir = RUNS_LOCAL / f"{b}_seed{s}"
            if not local_dir.exists():
                print(f"  pulling {remote}...")
                modal_pull(remote, local_dir)
            p = find_summary(local_dir)
            if not p:
                print(f"  no summary for {b} seed={s}")
                continue
            r = read_summary_row(p)
            if not r: continue
            long_rows.append({
                "bucket": b, "seed": s,
                "n_real_reviews": r.get("n_real_reviews",""),
                "n_overlap_aspects": r.get("n_overlap_aspects",""),
                "micro_precision": float(r["micro_precision"]),
                "micro_recall": float(r["micro_recall"]),
                "micro_f1": float(r["micro_f1"]),
                "macro_precision": float(r["macro_precision"]),
                "macro_recall": float(r["macro_recall"]),
                "macro_f1": float(r["macro_f1"]),
                "macro_balanced_accuracy": float(r["macro_balanced_accuracy"]),
                "sentiment_mse_detected": float(r["sentiment_mse_detected"]),
            })
    if not long_rows:
        print("no results found"); return 1

    with long_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(long_rows[0].keys()))
        w.writeheader()
        for r in long_rows: w.writerow(r)
    print(f"\nwrote: {long_csv} ({len(long_rows)} rows)")

    # Per-bucket aggregate
    metrics = ["micro_f1","macro_f1","macro_balanced_accuracy","sentiment_mse_detected","micro_precision","micro_recall","macro_recall"]
    agg_rows = []
    for b in BUCKET_NAMES:
        bs = [r for r in long_rows if r["bucket"]==b]
        if not bs: continue
        row = {"bucket": b, "n_seeds": len(bs), "seeds": ",".join(str(r["seed"]) for r in bs)}
        for m in metrics:
            vals = [r[m] for r in bs]
            row[f"{m}_mean"] = round(sum(vals)/len(vals), 4)
            row[f"{m}_std"] = round(statistics.stdev(vals) if len(vals)>=2 else 0.0, 4)
            lo, hi = bootstrap_ci(vals, seed=hash(b+m) & 0xffff)
            row[f"{m}_ci_lo"] = round(lo, 4)
            row[f"{m}_ci_hi"] = round(hi, 4)
        agg_rows.append(row)

    keys = ["bucket","n_seeds","seeds"]
    for m in metrics:
        keys += [f"{m}_mean", f"{m}_std", f"{m}_ci_lo", f"{m}_ci_hi"]
    with summ_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in agg_rows: w.writerow(r)
    print(f"wrote: {summ_csv}")
    print()
    print("Aggregate per-bucket Herath transfer (mean ± std across seeds):")
    print(f"{'bucket':<11s} {'n':>3s} {'micro_f1':>20s} {'macro_f1':>20s} {'macro_bal_acc':>20s} {'sent_mse':>20s}")
    for r in agg_rows:
        f1 = f"{r['micro_f1_mean']:.3f}±{r['micro_f1_std']:.3f}"
        mf1 = f"{r['macro_f1_mean']:.3f}±{r['macro_f1_std']:.3f}"
        ba = f"{r['macro_balanced_accuracy_mean']:.3f}±{r['macro_balanced_accuracy_std']:.3f}"
        sm = f"{r['sentiment_mse_detected_mean']:.3f}±{r['sentiment_mse_detected_std']:.3f}"
        print(f"  {r['bucket']:<10s} {r['n_seeds']:>3d} {f1:>20s} {mf1:>20s} {ba:>20s} {sm:>20s}")

    # Figure: bar with error bars
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7.5, 4.0))
        labels = [r["bucket"] for r in agg_rows]
        means = [r["micro_f1_mean"] for r in agg_rows]
        stds = [r["micro_f1_std"] for r in agg_rows]
        ci_lo = [r["micro_f1_ci_lo"] for r in agg_rows]
        ci_hi = [r["micro_f1_ci_hi"] for r in agg_rows]
        # asymmetric error bars
        err_lo = [m - lo for m, lo in zip(means, ci_lo)]
        err_hi = [hi - m for m, hi in zip(means, ci_hi)]
        colors = ["#2c7fb8" if l in ("top25","top50") else ("#7fcdbb" if l == "full" else ("#fdae61" if l == "random_5k" else "#d7191c")) for l in labels]
        ax.bar(labels, means, yerr=[err_lo, err_hi], color=colors, capsize=6)
        for x, (m, s_) in enumerate(zip(means, stds)):
            ax.text(x, m + 0.01, f"{m:.3f}", ha="center", va="bottom", fontsize=9)
        if "full" in labels:
            full_val = means[labels.index("full")]
            ax.axhline(full_val, color="#666", linestyle="--", linewidth=1, label=f"full = {full_val:.3f}")
            ax.legend(loc="best", fontsize=8)
        ax.set_ylabel("Herath micro-F1 (mean across seeds, 95% bootstrap CI)")
        ax.set_title("Phase D2 multi-seed: faithfulness-filtered BERT transfer to Herath")
        ax.set_ylim(0, max(means)*1.25)
        plt.tight_layout()
        fp_svg = FIGURES_DIR / "phase_d2_filtering_micro_f1_multiseed.svg"
        fp_png = FIGURES_DIR / "phase_d2_filtering_micro_f1_multiseed.png"
        fig.savefig(fp_svg); fig.savefig(fp_png, dpi=150)
        print(f"\nfigures: {fp_svg}, {fp_png}")
    except ImportError:
        print("matplotlib not available")
    return 0


if __name__ == "__main__":
    sys.exit(main())
