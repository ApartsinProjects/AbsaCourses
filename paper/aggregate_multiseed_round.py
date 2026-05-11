from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_OUT = ROOT / "paper" / "benchmark_outputs"
RUNS_DIR = BENCHMARK_OUT / "runs"
TABLE_DIR = ROOT / "paper" / "outputs" / "tables"


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_markdown(df: pd.DataFrame, path: Path) -> None:
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in df.astype(str).itertuples(index=False, name=None):
        lines.append("| " + " | ".join(row) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate a completed multiseed local benchmark round into stability summaries.")
    parser.add_argument("--status", required=True)
    args = parser.parse_args()

    status_path = Path(args.status).resolve()
    status = load_json(status_path)
    completed_batches = [batch for batch in status.get("batches", []) if batch.get("status") == "completed"]
    if not completed_batches:
        raise SystemExit("No completed batches found.")

    summary_rows: list[dict[str, Any]] = []
    per_aspect_rows: list[pd.DataFrame] = []
    source_run_dirs: list[str] = []

    for batch in completed_batches:
        resume_path = Path(batch["resume_path"]).resolve()
        resume_state = load_json(resume_path)
        run_dir = Path(str(resume_state.get("final_run_dir", ""))).resolve()
        if not run_dir.exists():
            raise SystemExit(f"Missing run dir for completed batch: {resume_path}")
        source_run_dirs.append(str(run_dir))
        metadata = load_json(run_dir / "metadata.json")
        summary = pd.read_csv(run_dir / "summary.csv")
        if summary.shape[0] != 1:
            raise SystemExit(f"Expected one-row summary in {run_dir}, found {summary.shape[0]}")
        row = summary.iloc[0].to_dict()
        row["seed"] = int(metadata.get("config", {}).get("seed", -1))
        summary_rows.append(row)

        per_aspect = pd.read_csv(run_dir / "per_aspect.csv").copy()
        per_aspect["seed"] = row["seed"]
        per_aspect_rows.append(per_aspect)

    summary_df = pd.DataFrame(summary_rows)
    metrics = [
        "micro_f1",
        "macro_f1",
        "micro_precision",
        "micro_recall",
        "macro_balanced_accuracy",
        "macro_specificity",
        "macro_mcc",
        "sentiment_mse_detected",
        "elapsed_seconds",
    ]
    agg = (
        summary_df.groupby("approach", as_index=False)[metrics]
        .agg(["mean", "std"])
        .reset_index()
    )
    agg.columns = [
        "approach"
        if col[0] == "approach"
        else f"{col[0]}_{col[1]}"
        for col in agg.columns.to_flat_index()
    ]
    if "approach_" in agg.columns:
        agg = agg.rename(columns={"approach_": "approach"})
    agg = agg.sort_values("micro_f1_mean", ascending=False).reset_index(drop=True)

    per_aspect_df = pd.concat(per_aspect_rows, ignore_index=True)
    per_aspect_agg = (
        per_aspect_df.groupby(["approach", "aspect"], as_index=False)[["f1", "precision", "recall", "mse"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    per_aspect_agg.columns = [
        col[0] if col[1] == "" else f"{col[0]}_{col[1]}"
        for col in per_aspect_agg.columns.to_flat_index()
    ]

    run_dir = RUNS_DIR / f"benchmark_multiseed_aggregate_{utc_stamp()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    agg.to_csv(run_dir / "summary.csv", index=False)
    per_aspect_agg.to_csv(run_dir / "per_aspect.csv", index=False)

    metadata = {
        "round_id": status.get("round_id"),
        "status_path": str(status_path),
        "source_run_dirs": source_run_dirs,
        "aggregated_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_completed": len(completed_batches),
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    pretty = agg.loc[:, [
        "approach",
        "micro_f1_mean",
        "micro_f1_std",
        "micro_recall_mean",
        "micro_recall_std",
        "sentiment_mse_detected_mean",
        "sentiment_mse_detected_std",
    ]].copy()
    for col in pretty.columns:
        if col != "approach":
            pretty[col] = pd.to_numeric(pretty[col], errors="coerce").map(lambda x: f"{x:.4f}" if pd.notna(x) else "")
    pretty.to_csv(TABLE_DIR / "multiseed_summary.csv", index=False)
    write_markdown(pretty, TABLE_DIR / "multiseed_summary.md")

    print(
        json.dumps(
            {
                "aggregate_run_dir": str(run_dir),
                "summary_csv": str(TABLE_DIR / "multiseed_summary.csv"),
                "rows": int(agg.shape[0]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
