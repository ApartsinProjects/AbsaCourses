# Experiment Script Audit Round 2

Date: 2026-04-03

This pass focused on whether the experiment scripts:

- use the intended study datasets by default,
- preserve the paper-facing evidence boundary,
- avoid stale-plan and stale-manifest execution,
- expose a consistent metric schema across method families,
- and align with the reviewer feedback on trustworthiness and reproducibility.

## Resolved issues

1. `absa_model_comparison.py` now defaults to the main `10K / 20`-aspect
   corpus instead of the legacy `edu` dataset.

2. `build_individual_experiment_round.py` now passes explicit data arguments for
   every benchmark, transfer, and overlap task.

3. `run_experiment_round.py` now checks a content hash of the plan before
   reusing an existing status file. Changing a plan without changing the path no
   longer reuses stale commands.

4. Individual benchmark tasks now pass `--no-write-latest`, so a single-model
   rerun does not overwrite the paper-facing aggregate benchmark files.

5. `monitor_generation_job.py` now passes the submitted batch manifest into
   `consume_generation_batch.py`.

6. `consume_generation_batch.py` now rejects non-completed rows by default.
   Incomplete rows can still be included deliberately via `--include-incomplete`.

7. `evaluate_models()` in `absa_model_comparison.py` now returns the richer
   multilabel diagnostics already used by the TF-IDF, joint, and GPT paths:
   balanced accuracy, specificity, MCC, subset accuracy, samples-F1, and
   samples-Jaccard.

8. `evaluate_joint_model()` now emits per-aspect specificity, balanced
   accuracy, and MCC so its output schema matches the rest of the benchmark
   family better.

9. `realism_validation_experiment.py` now copies the sampled attribute dict
   before prompt construction, so prompt rendering does not mutate the logged
   attribute state.

10. `openai_eval_batch_prep.py` now states clearly that the batch path covers
    only batch-safe single-call GPT variants. Multi-stage prompted decompositions
    remain outside this batch-prep script.

11. `check_experiment_round_status.py` now fails gracefully when the requested
    status file does not exist yet.

## Remaining non-code follow-ups

1. The current top-level benchmark result files in `paper/benchmark_outputs/`
   still reflect the last restored aggregate run, not a fresh rerun under the
   newest script versions.

2. The corrected per-model round plan exists, but it has not yet been relaunched
   after this audit pass.

3. Paper tables and figures that depend on benchmark outputs should be refreshed
   only after the next clean rerun completes.

## Current recommended launch target

- Plan:
  `/paper/experiment_rounds/next_round_individual_20260403_plan.json`
- Status path:
  `/paper/experiment_rounds/next_round_individual_20260403/status.json`

## Bottom line

The main correctness risks in the experiment scripts were orchestration and
artifact-integrity issues rather than model math. Those script-level issues are
now addressed. The next required step is a clean rerun so the saved outputs, the
paper, and the refreshed script behavior all match.
