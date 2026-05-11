# Experiment Configuration Audit

Date: 2026-04-03

This note records the corrected experiment configuration after the accidental
fallback to the legacy `edu/final_student_reviews.jsonl` dataset.

## Active benchmark targets

- Local benchmark default dataset:
  `/paper/generated_datasets/batch_69cc15c483488190941478aa4e3a976d_generated_reviews.jsonl`
- Mapped real-transfer synthetic source:
  `/paper/generated_datasets/batch_69cc15c483488190941478aa4e3a976d_generated_reviews.jsonl`
- Mapped real-transfer corpus:
  `/external_data/Student_feedback_analysis_dataset/Annotated Student Feedback Data`

## Corrected safeguards

- `absa_model_comparison.py` now defaults to the main 10K / 20-aspect corpus.
- `build_individual_experiment_round.py` passes explicit dataset paths for every
  local benchmark task via `--data-path`.
- `build_individual_experiment_round.py` passes explicit synthetic and Herath
  paths for every transfer and overlap task via `--synthetic-path` and
  `--herath-root`.
- Individual round tasks now pass `--no-write-latest` so single-model runs do
  not overwrite the paper-facing aggregate benchmark files.
- `run_experiment_round.py` now checks a content hash of the round plan before
  reusing an existing status file.
- `monitor_generation_job.py` now passes the submitted batch manifest explicitly
  into `consume_generation_batch.py`.
- `consume_generation_batch.py` now excludes non-completed rows by default,
  unless `--include-incomplete` is set deliberately.
- The stale grouped round and the misconfigured per-model round were stopped and
  archived.
- The top-level local benchmark summary files in
  `paper/benchmark_outputs/` were restored from the correct 10K / 20-aspect run
  bundle.

## Intended launch target

- Plan file:
  `/paper/experiment_rounds/next_round_individual_20260403_plan.json`
- Launch command:
  `python /paper/run_experiment_round.py --plan /paper/experiment_rounds/next_round_individual_20260403_plan.json --status /paper/experiment_rounds/next_round_individual_20260403/status.json`

## Notes

- GPU serialization, detailed logging, and resume checkpoints remain enabled.
- Only the plan file above should be treated as live for the next benchmark run.
