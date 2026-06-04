"""Phase E1+E2+E3 aggregator.

Pulls all (bucket, seed, arch, target) results from Modal volume `phase-d2-
results`, computes per-(bucket,arch,target) mean and paired bootstrap CI
across seeds, and writes a long-format CSV plus per-table summary CSVs.
"""
from __future__ import annotations
import csv, json, random, statistics, subprocess
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).parent.resolve()
PHASE_OUT = HERE / "paper" / "experiment_rounds" / f"phase_d2_filtering_{datetime.now(timezone.utc).strftime('%Y%m%d')}"
RUNS_LOCAL = PHASE_OUT / "runs"
TABLES_DIR = HERE / "paper" / "outputs" / "tables"
BUCKETS = ["top25","top50","full","bot25","random_5k"]
ALL_SEEDS = [42, 17, 23, 41, 53, 89, 101, 137]
ARCHS = ["bert-base-uncased", "distilbert-base-uncased"]
TARGETS = ["herath", "edurabsa"]
MODAL_CLI = r"C:\Users\apart\AppData\Roaming\Python\Python314\Scripts\modal"
VOLUME = "phase-d2-results"


def remote_dir(bucket: str, seed: int, arch: str, target: str) -> str:
    """Mirror modal_phase_d2 train_bucket path-scheme."""
    arch_tag = arch.replace("-base-uncased","").replace("/","_")
    suffix = ""
    if arch != "bert-base-uncased": suffix += f"_{arch_tag}"
    if target != "herath": suffix += f"_{target}"
    if arch == "bert-base-uncased" and target == "herath":
        return f"/{bucket}" if seed == 42 else f"/{bucket}_seed{seed}"
    return f"/{bucket}_seed{seed}{suffix}"


def local_dir(remote: str) -> Path:
    return RUNS_LOCAL / remote.lstrip("/")


def modal_pull(remote: str, dest: Path) -> bool:
    dest.mkdir(parents=True, exist_ok=True)
    cmd = [MODAL_CLI, "volume", "get", VOLUME, remote, str(dest.parent), "--force"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode == 0


def find_summary(d: Path) -> Path | None:
    for c in (d / "run" / "summary.csv", d / "summary.csv"):
        if c.exists(): return c
    cands = list(d.rglob("summary.csv"))
    return cands[0] if cands else None


def read_summary_row(p: Path) -> dict | None:
    with p.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        if "bert" in r.get("approach","").lower() or "distilbert" in r.get("approach","").lower():
            return r
    return rows[0] if rows else None


def bootstrap_ci(values, n_boot=10000, alpha=0.05, seed=0):
    if len(values) < 2:
        v = values[0] if values else 0
        return v, v
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n))/n for _ in range(n_boot))
    return means[int(alpha/2*n_boot)], means[int((1-alpha/2)*n_boot)]


def paired_delta_ci(diffs, n_boot=10000, alpha=0.05, seed=0):
    rng = random.Random(seed)
    n = len(diffs)
    if n < 2:
        return (0, 0, 0)
    mean = sum(diffs)/n
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n))/n for _ in range(n_boot))
    return mean, means[int(alpha/2*n_boot)], means[int((1-alpha/2)*n_boot)]


def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    long_rows = []
    missing = []
    for b in BUCKETS:
        for s in ALL_SEEDS:
            for arch in ARCHS:
                for target in TARGETS:
                    rem = remote_dir(b, s, arch, target)
                    loc = local_dir(rem)
                    summary_path = find_summary(loc)
                    if not summary_path:
                        if not loc.exists():
                            if not modal_pull(rem, loc):
                                missing.append((b, s, arch, target, "pull-failed"))
                                continue
                        summary_path = find_summary(loc)
                        if not summary_path:
                            missing.append((b, s, arch, target, "no-summary"))
                            continue
                    r = read_summary_row(summary_path)
                    if not r:
                        missing.append((b, s, arch, target, "empty-summary"))
                        continue
                    long_rows.append({
                        "bucket": b, "seed": s, "arch": arch, "target": target,
                        "micro_f1": float(r["micro_f1"]),
                        "macro_f1": float(r["macro_f1"]),
                        "macro_balanced_accuracy": float(r["macro_balanced_accuracy"]),
                        "sentiment_mse_detected": float(r["sentiment_mse_detected"]),
                        "n_overlap_aspects": r.get("n_overlap_aspects",""),
                    })

    print(f"long rows: {len(long_rows)}")
    print(f"missing: {len(missing)}")
    for m in missing[:10]:
        print(f"  missing: {m}")

    long_csv = TABLES_DIR / "phase_e_long.csv"
    with long_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(long_rows[0].keys()) if long_rows else
                           ["bucket","seed","arch","target","micro_f1","macro_f1","macro_balanced_accuracy","sentiment_mse_detected","n_overlap_aspects"])
        w.writeheader()
        for r in long_rows: w.writerow(r)
    print(f"wrote {long_csv}")

    # Per-(arch, target) summary table per bucket
    metrics = ["sentiment_mse_detected","macro_balanced_accuracy","macro_f1","micro_f1"]
    summary_rows = []
    for arch in ARCHS:
        for target in TARGETS:
            for b in BUCKETS:
                vals = {m: [r[m] for r in long_rows if r["bucket"]==b and r["arch"]==arch and r["target"]==target] for m in metrics}
                if not vals[metrics[0]]: continue
                row = {"arch": arch, "target": target, "bucket": b, "n_seeds": len(vals[metrics[0]])}
                for m in metrics:
                    row[f"{m}_mean"] = round(sum(vals[m])/len(vals[m]), 4)
                    row[f"{m}_std"] = round(statistics.stdev(vals[m]) if len(vals[m])>=2 else 0, 4)
                    lo, hi = bootstrap_ci(vals[m], seed=hash(arch+target+b+m)&0xffff)
                    row[f"{m}_ci_lo"] = round(lo,4); row[f"{m}_ci_hi"] = round(hi,4)
                summary_rows.append(row)
    summ_csv = TABLES_DIR / "phase_e_summary.csv"
    keys = ["arch","target","bucket","n_seeds"]
    for m in metrics: keys += [f"{m}_mean",f"{m}_std",f"{m}_ci_lo",f"{m}_ci_hi"]
    with summ_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in summary_rows: w.writerow(r)
    print(f"wrote {summ_csv}")

    # Paired deltas for the key contrasts per (arch, target)
    contrasts = [("top50","random_5k"), ("top50","full"), ("bot25","full"), ("top25","full"), ("random_5k","full")]
    delta_rows = []
    for arch in ARCHS:
        for target in TARGETS:
            for a, b in contrasts:
                for m in metrics:
                    by_seed_a = {r["seed"]: r[m] for r in long_rows if r["bucket"]==a and r["arch"]==arch and r["target"]==target}
                    by_seed_b = {r["seed"]: r[m] for r in long_rows if r["bucket"]==b and r["arch"]==arch and r["target"]==target}
                    common_seeds = sorted(set(by_seed_a) & set(by_seed_b))
                    if len(common_seeds) < 2: continue
                    diffs = [by_seed_a[s] - by_seed_b[s] for s in common_seeds]
                    mean, lo, hi = paired_delta_ci(diffs, seed=hash(a+b+m+arch+target)&0xffff)
                    win = sum(1 for d in diffs if (d<0 if m=="sentiment_mse_detected" else d>0))
                    delta_rows.append({
                        "arch": arch, "target": target, "contrast": f"{a} - {b}",
                        "metric": m, "n_seeds": len(common_seeds),
                        "mean_delta": round(mean,4),
                        "ci_lo": round(lo,4), "ci_hi": round(hi,4),
                        "wins": f"{win}/{len(common_seeds)}",
                        "excludes_zero": "YES" if (lo>0 or hi<0) else "no",
                    })
    delta_csv = TABLES_DIR / "phase_e_paired_deltas.csv"
    with delta_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["arch","target","contrast","metric","n_seeds","mean_delta","ci_lo","ci_hi","wins","excludes_zero"])
        w.writeheader()
        for r in delta_rows: w.writerow(r)
    print(f"wrote {delta_csv}")
    print()
    # Print headline contrasts
    print("HEADLINE: top50 - random_5k on sentiment_mse_detected:")
    for r in delta_rows:
        if r["contrast"]=="top50 - random_5k" and r["metric"]=="sentiment_mse_detected":
            print(f"  arch={r['arch']:25s} target={r['target']:8s} n={r['n_seeds']} mean={r['mean_delta']:+.4f} CI=[{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}] wins={r['wins']} excl0={r['excludes_zero']}")
    print()
    print("HEADLINE: bot25 - full on sentiment_mse_detected:")
    for r in delta_rows:
        if r["contrast"]=="bot25 - full" and r["metric"]=="sentiment_mse_detected":
            print(f"  arch={r['arch']:25s} target={r['target']:8s} n={r['n_seeds']} mean={r['mean_delta']:+.4f} CI=[{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}] wins={r['wins']} excl0={r['excludes_zero']}")


if __name__ == "__main__":
    main()
