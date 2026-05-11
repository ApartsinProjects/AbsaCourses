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


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate completed single-model benchmark runs into paper-facing latest outputs.")
    parser.add_argument("--status", required=True, help="Round status JSON from run_experiment_round.py.")
    parser.add_argument("--write-latest", action="store_true", help="Overwrite paper-facing benchmark outputs after aggregation.")
    args = parser.parse_args()

    status_path = Path(args.status).resolve()
    status = load_json(status_path)

    completed_batches = [batch for batch in status.get("batches", []) if batch.get("status") == "completed"]
    if not completed_batches:
        raise SystemExit("No completed batches found in the provided round status.")

    summary_frames: list[pd.DataFrame] = []
    per_aspect_frames: list[pd.DataFrame] = []
    metadata_entries: list[dict[str, Any]] = []
    source_run_dirs: list[str] = []
    dataset_path: str | None = None

    for batch in completed_batches:
        resume_path = Path(batch["resume_path"]).resolve()
        if not resume_path.exists():
            raise SystemExit(f"Missing resume checkpoint for completed batch: {resume_path}")
        resume_state = load_json(resume_path)
        run_dir_raw = resume_state.get("final_run_dir")
        if not run_dir_raw:
            raise SystemExit(f"Completed batch is missing final_run_dir in resume checkpoint: {resume_path}")
        run_dir = Path(run_dir_raw).resolve()
        source_run_dirs.append(str(run_dir))

        metadata = load_json(run_dir / "metadata.json")
        metadata_entries.append(metadata)
        dataset_path = dataset_path or metadata.get("dataset_path")
        if dataset_path != metadata.get("dataset_path"):
            raise SystemExit("Dataset mismatch across completed benchmark runs; refusing to aggregate mixed corpora.")

        summary_df = pd.read_csv(run_dir / "summary.csv")
        per_aspect_df = pd.read_csv(run_dir / "per_aspect.csv")
        summary_frames.append(summary_df)
        per_aspect_frames.append(per_aspect_df)

    summary = pd.concat(summary_frames, ignore_index=True)
    summary = summary.sort_values(["micro_f1", "macro_f1", "approach"], ascending=[False, False, True]).reset_index(drop=True)
    per_aspect = pd.concat(per_aspect_frames, ignore_index=True)

    aggregate_run_dir = RUNS_DIR / f"benchmark_phase_a_aggregate_{utc_stamp()}"
    aggregate_run_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(aggregate_run_dir / "summary.csv", index=False)
    per_aspect.to_csv(aggregate_run_dir / "per_aspect.csv", index=False)

    aggregate_metadata = {
        "round_id": status.get("round_id"),
        "status_path": str(status_path),
        "source_run_dirs": source_run_dirs,
        "dataset_path": dataset_path,
        "approaches": summary["approach"].tolist(),
        "aggregated_at_utc": datetime.now(timezone.utc).isoformat(),
        "metric_columns": list(summary.columns),
    }
    (aggregate_run_dir / "metadata.json").write_text(json.dumps(aggregate_metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.write_latest:
        summary.to_csv(BENCHMARK_OUT / "model_comparison_summary.csv", index=False)
        per_aspect.to_csv(BENCHMARK_OUT / "model_comparison_per_aspect.csv", index=False)
        latest_metadata = {
            "dataset_path": dataset_path,
            "source_round_id": status.get("round_id"),
            "source_status_path": str(status_path),
            "source_run_dirs": source_run_dirs,
            "approaches": summary["approach"].tolist(),
            "aggregated_run_dir": str(aggregate_run_dir),
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        (BENCHMARK_OUT / "model_comparison_metadata.json").write_text(
            json.dumps(latest_metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    print(
        json.dumps(
            {
                "aggregate_run_dir": str(aggregate_run_dir),
                "approach_count": int(summary.shape[0]),
                "write_latest": bool(args.write_latest),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
