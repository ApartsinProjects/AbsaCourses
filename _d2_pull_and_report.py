"""Phase D2 Stages 5-6: pull Modal results, compile the per-bucket table, plot.

Pulls each bucket's `summary.csv` from the `phase-d2-results` Modal volume,
combines into one results CSV, and produces the headline bar chart.
"""
from __future__ import annotations
import csv, json, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).parent.resolve()
PHASE_OUT = HERE / "paper" / "experiment_rounds" / f"phase_d2_filtering_{datetime.now(timezone.utc).strftime('%Y%m%d')}"
RUNS_LOCAL = PHASE_OUT / "runs"
TABLES_DIR = HERE / "paper" / "outputs" / "tables"
FIGURES_DIR = HERE / "paper" / "outputs" / "figures"
BUCKET_NAMES = ["top25", "top50", "full", "bot25", "random_5k"]
MODAL_CLI = r"C:\Users\apart\AppData\Roaming\Python\Python314\Scripts\modal"
VOLUME = "phase-d2-results"


def modal_get(remote: str, local: Path) -> bool:
    local.parent.mkdir(parents=True, exist_ok=True)
    if local.exists():
        local.unlink()
    cmd = [MODAL_CLI, "volume", "get", VOLUME, remote, str(local)]
    print(f"  modal volume get {remote} -> {local}", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"    FAILED: {r.stderr.strip()[:300]}")
        return False
    return True


def pull_bucket(bucket: str) -> Path | None:
    """Pull the run dir for one bucket; return its local run path."""
    bdir = RUNS_LOCAL / bucket
    bdir.mkdir(parents=True, exist_ok=True)
    # Pull whole bucket dir (Modal CLI supports recursive get when remote is a dir)
    cmd = [MODAL_CLI, "volume", "get", VOLUME, f"/{bucket}", str(bdir.parent), "--force"]
    print(f"  modal volume get /{bucket} -> {bdir.parent}", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"    FAILED: {r.stderr.strip()[:300]}")
        return None
    # Expect summary.csv under run/ inside bdir
    run_root = bdir / "run"
    if not run_root.exists():
        print(f"    no run/ subdir found at {bdir}")
        return None
    return run_root


def read_summary(run_root: Path) -> dict | None:
    """Read summary.csv -> the first row (bert-base-uncased)."""
    p = run_root / "summary.csv"
    if not p.exists():
        # Fallback: look for any csv that has micro_f1
        candidates = list(run_root.rglob("*summary*.csv"))
        if not candidates: return None
        p = candidates[0]
    with p.open(encoding="utf-8") as f:
        rd = csv.DictReader(f)
        rows = list(rd)
    if not rows: return None
    # Prefer bert-base-uncased row
    for r in rows:
        if "bert-base-uncased" in r.get("approach","") or "bert-base-uncased" in r.get("name",""):
            return r
    return rows[0]


def main() -> int:
    RUNS_LOCAL.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    table_csv = TABLES_DIR / "phase_d2_filtering_results.csv"

    rows = []
    for b in BUCKET_NAMES:
        print(f"\n== {b} ==")
        run_root = pull_bucket(b)
        if run_root is None:
            rows.append({"bucket": b, "ok": False, "note": "pull failed"})
            continue
        s = read_summary(run_root)
        if s is None:
            rows.append({"bucket": b, "ok": False, "note": "no summary.csv"})
            continue
        rows.append({
            "bucket": b, "ok": True,
            "approach": s.get("approach", s.get("name", "")),
            "n_overlap_aspects": s.get("n_overlap_aspects",""),
            "n_real_reviews": s.get("n_real_reviews",""),
            "micro_precision": s.get("micro_precision",""),
            "micro_recall": s.get("micro_recall",""),
            "micro_f1": s.get("micro_f1",""),
            "macro_precision": s.get("macro_precision",""),
            "macro_recall": s.get("macro_recall",""),
            "macro_f1": s.get("macro_f1",""),
            "macro_balanced_accuracy": s.get("macro_balanced_accuracy",""),
            "sentiment_mse_detected": s.get("sentiment_mse_detected",""),
            "eval_split": s.get("eval_split",""),
        })

    keys = ["bucket","ok","approach","n_real_reviews","n_overlap_aspects",
            "micro_precision","micro_recall","micro_f1",
            "macro_precision","macro_recall","macro_f1",
            "macro_balanced_accuracy","sentiment_mse_detected",
            "eval_split","note"]
    with table_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nwrote: {table_csv}")
    for r in rows:
        print(f"  {r['bucket']:10s} ok={r.get('ok')} micro_f1={r.get('micro_f1','')} macro_f1={r.get('macro_f1','')} sent_mse={r.get('sentiment_mse_detected','')}")

    # Bar chart
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        labels, vals = [], []
        for r in rows:
            if r.get("ok") and r.get("micro_f1"):
                labels.append(r["bucket"])
                vals.append(float(r["micro_f1"]))
        if labels:
            fig, ax = plt.subplots(figsize=(6, 3.6))
            colors = ["#2c7fb8" if l in ("top25","top50") else ("#7fcdbb" if l == "full" else ("#fdae61" if l == "random_5k" else "#d7191c")) for l in labels]
            ax.bar(labels, vals, color=colors)
            for x, v in enumerate(vals):
                ax.text(x, v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
            if "full" in labels:
                full_val = vals[labels.index("full")]
                ax.axhline(full_val, color="#666", linestyle="--", linewidth=1, label=f"full = {full_val:.3f}")
                ax.legend(loc="best", fontsize=8)
            ax.set_ylabel("Herath micro-F1")
            ax.set_title("Phase D2 faithfulness-filtered training (BERT-base, seed 42)")
            ax.set_ylim(0, max(vals) * 1.18)
            plt.tight_layout()
            fig_path = FIGURES_DIR / "phase_d2_filtering_micro_f1.svg"
            fig.savefig(fig_path)
            png_path = FIGURES_DIR / "phase_d2_filtering_micro_f1.png"
            fig.savefig(png_path, dpi=150)
            print(f"\nfigures: {fig_path}, {png_path}")
    except ImportError:
        print("matplotlib not available; skipping figure")
    return 0


if __name__ == "__main__":
    sys.exit(main())
