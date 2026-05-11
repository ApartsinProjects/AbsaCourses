# Experiment Artifact Preservation Audit

Date: 2026-04-03

## Goal

Verify that future experiment runs preserve low-level evidence needed for later reanalysis, debugging, and paper revision, including:

- run summaries
- per-aspect metrics
- per-example predictions
- detection probabilities / logits
- sentiment predictions
- raw and parsed LLM responses
- resume-safe checkpoint copies

## Findings Before Patch

The repository already preserved high-level outputs well, but sample-level artifacts were inconsistent:

- local benchmark runs saved summary and per-aspect CSVs, but not per-example prediction records
- transfer and overlap runs saved summary/per-aspect CSVs, but not sample-level prediction artifacts
- batch GPT evaluation preserved the raw batch JSONL externally, but did not archive a normalized per-example response file in the run bundle
- resume checkpoints did not mirror low-level artifacts, so an interrupted run could preserve summary rows while losing the detailed prediction evidence behind them

## Fixes Applied

### Local benchmark stack

Updated:

- `paper/absa_model_comparison.py`

Now each archived run bundle stores, per approach:

- `sample_predictions.jsonl`
- `thresholds.json`

Each `sample_predictions.jsonl` row contains:

- source row index
- review text
- gold aspect map
- predicted aspect map
- per-aspect detection probabilities
- per-aspect detection logits
- per-aspect binary predictions
- per-aspect binary targets
- sentiment prediction values
- sentiment target values
- thresholds used

The same artifacts are mirrored into the resume checkpoint directory after each completed approach.

### Real-transfer evaluation

Updated:

- `paper/evaluate_synthetic_to_real_transfer.py`

Each run now archives:

- summary
- per-aspect metrics
- per-approach artifact files under the run `artifacts/` directory

Resume checkpoints also store the low-level artifact files for each completed approach.

### Overlap comparison

Updated:

- `paper/evaluate_overlap_generalization.py`

Each run now archives, per approach:

- synthetic sample predictions
- real sample predictions
- synthetic thresholds
- real thresholds

These are preserved both in the final run bundle and the resume checkpoint directory.

### Batch GPT evaluation

Updated:

- `paper/openai_eval_batch_consume.py`

Each consumed batch run now archives:

- summary
- per-aspect metrics
- `artifacts/llm_responses.jsonl`

Each response row contains:

- custom id
- review text
- gold aspect map
- raw response text
- parsed response object
- normalized predicted aspect map
- response status

## Smoke Verification

Verified with:

- `paper/benchmark_outputs/runs/benchmark_smoke_20260403T161711Z`

Confirmed artifacts:

- `artifacts/tfidf_two_step_sample_predictions.jsonl`
- `artifacts/tfidf_two_step_thresholds.json`

Confirmed resume-safe mirror:

- `paper/benchmark_outputs/resume/preservation_smoke_v2.resume.tfidf_two_step.sample_predictions.jsonl`
- `paper/benchmark_outputs/resume/preservation_smoke_v2.resume.tfidf_two_step.thresholds.json`

## Remaining Boundary

The generation and realism-validation pipelines were already preserving their raw outputs reasonably well before this audit:

- OpenAI generation batch raw result JSONLs remain the source of truth for returned generations
- realism validation already saves raw judge outputs, summaries, and refinement artifacts

So the main preservation gap was the model-evaluation stack, which is now materially improved.

## Conclusion

Future experiment runs now preserve the low-level artifacts needed for reanalysis and reviewer follow-up, not only the headline tables.
