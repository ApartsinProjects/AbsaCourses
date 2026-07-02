# Additional Experiment Plan

This plan targets the remaining top-tier-reviewer risks in the current paper: trustworthiness of the main benchmark, breadth and stability of the reported comparisons, faithfulness of the synthetic labels, and narrowness of the external-validation story.

## Guiding rule

Each experiment is tied to one paper claim and one acceptance risk. The goal is not to enlarge the paper indiscriminately, but to close the smallest set of gaps that most improves reviewer confidence.

## Priority order

1. Repair the main benchmark trust boundary.
2. Add uncertainty estimates to the strongest local comparisons.
3. Close the joint-model and prompt-family execution gaps.
4. Turn label-faithfulness from a limitation into an experimentally grounded finding.
5. Strengthen external validation if a second aligned real dataset can be mapped cleanly.

## Pre-experiment protocol repairs

These are required before new result tables are treated as final.

| ID | Action | Why it matters | Output |
| --- | --- | --- | --- |
| P0 | Recompute local benchmark summary metrics from validation-calibrated thresholds rather than fixed `0.5` summaries | Aligns the code path with the manuscript's stated protocol | refreshed `benchmark_outputs/model_comparison_*.{csv,json}` and regenerated paper tables/figures |
| P1 | Archive sample-level predictions for all new local and GPT runs | Enables later computation of sample-level multilabel metrics and error analysis | run bundles under `benchmark_outputs/runs/<run_id>/predictions.*` |
| P2 | Save split indices for synthetic train/validation/test and overlap-only splits | Makes reruns and later audits reproducible | `paper/benchmark_outputs/splits/*.json` |
| P3 | Fix the manuscript sentence in Section 4.2 that still says joint models are not reported | Prevents a reviewer-visible contradiction | `course_absa_manuscript.html` |

## Experiment matrix

### E1. Calibrated benchmark rerun

- Purpose: make the main local benchmark fully trustworthy.
- Claim supported: the headline synthetic-benchmark table reflects the exact validation-calibrated evaluation protocol described in the paper.
- Models:
  - `tfidf_two_step`
  - `distilbert-base-uncased`
  - `bert-base-uncased`
  - `albert-base-v2`
  - `roberta-base`
  - `bert_joint`
  - `distilbert_joint`
- Locked protocol:
  - data: full `10K / 20`-aspect corpus
  - split: deterministic `8000 / 1000 / 1000`
  - seed: `42`
  - threshold policy: per-aspect validation calibration only
  - metrics:
    - micro-F1
    - macro-F1
    - micro-precision
    - micro-recall
    - sentiment MSE
    - macro balanced accuracy
    - macro specificity
    - macro MCC
    - samples-F1
    - subset accuracy
    - samples-Jaccard
    - Hamming loss
- Outputs:
  - `paper/benchmark_outputs/runs/benchmark_calibrated_<timestamp>/`
  - refreshed main `model_comparison_*` files

### E2. Multi-seed stability on the leading local models

- Purpose: reduce uncertainty around close model rankings.
- Claim supported: the strongest local comparisons are not artifacts of one random seed.
- Models:
  - `distilbert-base-uncased`
  - `bert-base-uncased`
  - `tfidf_two_step`
  - `bert_joint`
  - `distilbert_joint`
- Seeds: `3, 13, 23`
- Locked protocol:
  - same data and split logic as `E1`
  - same calibrated threshold procedure
- Reported outputs:
  - mean and standard deviation for all main detection metrics
  - seed-wise runtime
  - seed-wise per-aspect F1 for the top two models
- Outputs:
  - `paper/benchmark_outputs/runs/benchmark_multiseed_<timestamp>/`
  - publication table `outputs/tables/multiseed_summary.*`
  - figure `outputs/figures/seed_stability_*.svg`

### E3. Hyperparameter repair experiment for accuracy

- Purpose: test whether the benchmark can yield meaningfully stronger local performance without changing the dataset.
- Claim supported: the benchmark is not bottlenecked solely by a weak default training recipe.
- Models:
  - `bert-base-uncased`
  - `distilbert-base-uncased`
- Grid:
  - learning rate: `3e-5`, `2e-5`, `1e-5`
  - epochs: `3`, `4`, `5`
  - scheduler: `none`, `linear warmup 10%`
  - loss variant:
    - weighted BCE
    - focal BCE for detection
- Stopping rule:
  - choose one best configuration per model by validation micro-F1, then report once on test
- Success criterion:
  - improve the best held-out micro-F1 or reduce sentiment MSE without degrading balanced accuracy materially
- Outputs:
  - `paper/benchmark_outputs/runs/tuning_<timestamp>/`
  - compact tuning table and one validation-to-test transfer figure

### E4. Full prompt-family completion in batch mode

- Purpose: close the gap between the prompt family described in Section 4.2 and the actually executed evidence.
- Claim supported: the prompted ABSA family is evaluated more completely, not only through four direct prompting variants.
- Models:
  - `gpt-5.2` for full-test batch comparison
  - optional small diagnostic slice with `gpt-5.4` only if needed for prompt-accuracy analysis, not for the main leaderboard
- Variants to execute:
  - `zero-shot`
  - `few-shot`
  - `few-shot-diverse`
  - `retrieval-few-shot`
  - `two-pass`
  - `aspect-by-aspect`
- Locked protocol:
  - batch mode only
  - same `1000`-review test split
  - same sparse exact-key JSON contract
  - same evaluation metrics as `E1` where derivable
- Outputs:
  - `paper/batch_requests/openai_eval_batch_full_<timestamp>_*`
  - `paper/benchmark_outputs/runs/openai_batch_eval_full_<timestamp>/`
  - refreshed GPT tables and appendix configuration details

### E5. Faithfulness-aware data filtering ablation

- Purpose: turn the label-faithfulness limitation into a scientific experiment.
- Claim supported: higher-faithfulness synthetic supervision improves downstream ABSA quality, or at least changes the error profile in measurable ways.
- Design:
  - use the model-assisted faithfulness scorer to partition the synthetic corpus into:
    - full corpus
    - top `50%` by row-level support score
    - top `25%`
    - bottom `25%`
  - train the strongest two local models on each subset
- Models:
  - `bert-base-uncased`
  - `distilbert-base-uncased`
- Evaluation:
  - synthetic held-out test
  - mapped real-data overlap test
- Success criterion:
  - show whether cleaner synthetic labels buy either better internal F1, better real-transfer F1, or both
- Outputs:
  - `paper/faithfulness_filtering/`
  - one main-text table if gains are clear, otherwise appendix only

### E6. Group-wise aspect analysis

- Purpose: exploit the five pedagogical aspect groups already present in the paper.
- Claim supported: some pedagogical groups are systematically easier or harder, and conclusions are not driven only by aggregate F1.
- Groups:
  - instructional quality
  - assessment and course management
  - learning demand and readiness
  - learning environment
  - engagement and value
- Metrics:
  - group micro-F1
  - group macro-F1
  - group balanced accuracy
  - group sentiment MSE
- Comparisons:
  - best local two-step model
  - best joint model
  - best GPT variant
  - best mapped real-transfer model
- Outputs:
  - `outputs/tables/groupwise_summary.*`
  - `outputs/figures/groupwise_performance.svg`

### E7. Stronger external-validation check

- Purpose: improve the narrowness of the current real-data story.
- Claim supported: the synthetic supervision scheme is not specific to one mapped corpus.
- Preferred path:
  - add a second real educational dataset with aspect or opinion annotations that can be mapped conservatively
- If a second aligned dataset is unavailable:
  - run a stricter overlap stress test on the existing Herath mapping:
    - only aspects with sufficient support
    - only reviews with at least two mapped positive aspects
    - report calibration and balanced metrics
- Outputs:
  - `paper/real_transfer_external2/` if second dataset succeeds
  - otherwise `paper/real_transfer/robust_overlap_checks_*`

### E8. Calibration and ranking diagnostics

- Purpose: add metrics that are less sensitive to class count and more sensitive to ranking quality.
- Claim supported: the reported models are not only different in thresholded F1, but also in score calibration and ranking behavior.
- Metrics:
  - macro AUROC
  - macro average precision
  - Brier score for aspect detection
  - expected calibration error on the detection heads
- Scope:
  - strongest two local models
  - strongest GPT variant if probability-like scores can be derived; otherwise omit
- Outputs:
  - appendix table plus one calibration figure

## Batching and execution order

The next round should run in smaller background batches to keep manuscript editing independent from long training jobs.

**Execution rule:** every future training or evaluation run should be scheduled as an individual task per model or per prompt variant. Grouped batches such as “local_core” or “real_transfer” should not be used for new rounds. This keeps progress legible, makes resume behavior safer, and allows selective reruns without repeating already finished models.

### Round A: trust repair

1. `P0-P3`
2. `E1`
3. refresh main tables and figures

### Round B: uncertainty and stronger local evidence

1. `E2`
2. `E3`
3. refresh local-benchmark discussion and appendix robustness section

### Round C: prompt-family completion

1. `E4`
2. refresh GPT section, Table 7A, Table A5, and relevant figures

### Round D: dataset-quality science

1. `E5`
2. `E6`
3. refresh discussion and limitations

### Round E: external validation and calibration

1. `E7`
2. `E8`
3. refresh transfer section and appendix diagnostics

## Paper-facing deliverables

If all rounds complete, the paper should add or refresh:

- one corrected main benchmark table
- one multiseed stability table
- one tuning table
- one completed GPT-family table
- one faithfulness-filtering ablation table
- one group-wise pedagogical performance figure
- one expanded external-validation table
- one appendix calibration table and figure

## Suggested acceptance-safe claim after completion

If the experiments above succeed, the paper can more safely claim:

1. the synthetic benchmark is reproducible and internally learnable under calibrated evaluation;
2. the main rankings are stable across seeds and not artifacts of one training run;
3. prompted and trained ABSA methods can be compared under one shared output contract;
4. faithfulness quality is measurable and materially connected to downstream performance;
5. synthetic supervision shows at least conservative compatibility with real educational feedback under mapped overlap evaluation.

## Minimal acceptable completion set

If time is limited, the smallest set that most improves reviewer confidence is:

1. `P0-P3`
2. `E1`
3. `E2`
4. `E4`
5. `E5`

That set closes the largest trust and completeness gaps without requiring a second real dataset.
