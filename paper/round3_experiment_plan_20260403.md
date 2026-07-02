# Round 3 Experiment Plan

Date: 2026-04-03

## Objective

This round is designed to address the main reasons a top-tier AI journal reviewer would currently hesitate to recommend publication:

1. the synthetic labels are useful but noisy, especially for sentiment polarity;
2. the external validation story is still narrow;
3. the local benchmark needs stronger trust, uncertainty reporting, and diagnosis of weak baselines;
4. the prompted-LLM comparison is meaningful but still incomplete relative to the method family described in the paper;
5. the manuscript would benefit from stronger evidence that the benchmark is scientifically informative rather than only operationally usable.

The goal of Round 3 is not to enlarge the paper indiscriminately. The goal is to close the smallest set of empirical gaps that most improves reviewer confidence.

## Decision Rule

Each experiment in this round must support one paper claim and one reviewer concern. If an experiment does not materially reduce one of the risks above, it should not be added to the main submission package.

## Concern-to-Experiment Map

| Concern | Why it matters for acceptance | Round 3 response |
| --- | --- | --- |
| Label faithfulness is weak, especially for polarity | This directly limits the value of the benchmark as supervised training data | `R3-E4` faithfulness-aware filtering ablation; `R3-E5` expanded faithfulness audit |
| External validation is too narrow | One mapped corpus and a 9-aspect overlap are not enough for broad transfer claims | `R3-E6` stronger real-data validation or stricter overlap stress tests |
| Main benchmark needs stronger trust and uncertainty | Reviewers will worry that close rankings or pathologies reflect setup noise | `R3-E1` calibrated rerun; `R3-E2` multiseed stability |
| Some baselines look degenerate or under-optimized | Poorly calibrated weak models reduce confidence in the whole benchmark | `R3-E3` baseline repair and targeted tuning |
| GPT family is narrower than the described prompted family | Reviewers will ask whether the method table is broader than the actual evidence | `R3-E7` full prompt-family completion in Batch mode |
| Aggregate F1 hides where the benchmark is informative | Reviewers want to know whether the task is pedagogically meaningful rather than just sparse and noisy | `R3-E8` group-wise and per-aspect diagnostics |

## Locked Global Protocol

These settings apply unless a specific experiment overrides them.

- Synthetic corpus: [batch_69cc15c483488190941478aa4e3a976d_generated_reviews.jsonl](E:\Projects\CourseABSA\paper\generated_datasets\batch_69cc15c483488190941478aa4e3a976d_generated_reviews.jsonl)
- Main split: deterministic `8000 / 1000 / 1000`
- Main schema: `20` aspects grouped into five pedagogical blocks
- Overlap real-data evaluation:
  - mapped Herath benchmark at [synthetic_to_real_transfer_summary.csv](E:\Projects\CourseABSA\paper\real_transfer\synthetic_to_real_transfer_summary.csv)
  - overlap comparison at [overlap_internal_vs_external_summary.csv](E:\Projects\CourseABSA\paper\real_transfer\overlap_internal_vs_external_summary.csv)
- Validation policy:
  - model selection on validation only
  - threshold calibration on validation only
  - final reporting on held-out test only
- Save policy:
  - every run must preserve raw outputs, run metadata, per-example predictions, thresholds, and detailed logs
  - no run should overwrite paper-facing aggregate outputs unless it is an explicitly designated aggregate refresh step
- LLM policy:
  - large inference or audit runs use Batch API
  - live API calls are restricted to small smoke/debug tasks

## Core Metrics

The following metrics should be available for all local and GPT benchmark comparisons wherever applicable:

- micro-precision
- micro-recall
- micro-F1
- macro-precision
- macro-recall
- macro-F1
- sentiment MSE on detected aspects
- macro balanced accuracy
- macro specificity
- macro MCC
- samples-F1
- subset accuracy
- samples-Jaccard
- Hamming loss

Additional score-based metrics should be added for discriminative models when probabilities are available:

- macro AUROC
- macro average precision
- Brier score
- expected calibration error

## Experiment Matrix

### R3-E1. Calibrated Main Benchmark Rerun

Purpose:
- make the main benchmark fully trustworthy and aligned with the manuscript protocol.

Reviewer concern addressed:
- the headline benchmark can be challenged if it is not clearly validation-calibrated and fully reproducible.

Models:
- `tfidf_two_step`
- `distilbert-base-uncased`
- `bert-base-uncased`
- `albert-base-v2`
- `roberta-base`
- `bert_joint`
- `distilbert_joint`

Locked settings:
- same synthetic corpus
- same split
- seed `42`
- validation-calibrated thresholds only

Success criteria:
- refreshed benchmark files are fully consistent with the paper
- no metric is reported from a fixed-threshold shortcut path
- run bundles preserve sample-level predictions and thresholds

Paper impact:
- refresh Table 7, Figure 6, Appendix A.9, Appendix A.10, and corresponding prose

### R3-E2. Multi-Seed Stability for the Leading Local Models

Purpose:
- show whether close rankings are robust or mostly seed noise.

Reviewer concern addressed:
- top-tier reviewers will discount close margins without uncertainty.

Models:
- `tfidf_two_step`
- `distilbert-base-uncased`
- `bert-base-uncased`
- `bert_joint`
- `distilbert_joint`

Seeds:
- `3`, `13`, `23`

Success criteria:
- stable ordering or an honestly revised interpretation if rankings overlap materially
- mean ± standard deviation reported for main metrics

Paper impact:
- strengthen Appendix A.9
- if uncertainty is small, cite it in Section `6.3`

### R3-E3. Baseline Repair and Targeted Tuning

Purpose:
- determine whether weaker or degenerate baselines reflect the task or the recipe.

Reviewer concern addressed:
- `roberta-base` and `albert-base-v2` look pathological; stronger baselines must not fail for trivial reasons.

Tracks:

1. Calibration repair for weak models
- inspect threshold distributions
- inspect positive-rate behavior
- verify logits and score spread

2. Targeted tuning for the strongest practical encoders
- `bert-base-uncased`
- `distilbert-base-uncased`

Grid:
- learning rate: `3e-5`, `2e-5`, `1e-5`
- epochs: `3`, `4`, `5`
- scheduler: `none`, `linear warmup 10%`
- detection loss: weighted BCE vs focal BCE

Success criteria:
- either rescue the weak models into sensible operating ranges, or document clearly that they remain poor even after calibration repair
- improve the best held-out local result beyond the current tuned BERT reference or improve balanced metrics without harming sentiment MSE materially

Paper impact:
- strengthens Section `6.3`
- can justify narrowing the baseline narrative if weak encoders remain poor after repair

### R3-E4. Faithfulness-Aware Filtering Ablation

Purpose:
- convert the biggest limitation into a scientific experiment.

Reviewer concern addressed:
- if synthetic label fidelity is noisy, show whether filtering changes downstream utility.

Design:
- score the synthetic corpus with the model-assisted faithfulness pipeline
- partition into:
  - full corpus
  - top `50%` by row-level support/faithfulness score
  - top `25%`
  - bottom `25%`

Training models:
- `bert-base-uncased`
- `distilbert-base-uncased`

Evaluation:
- full synthetic test
- mapped real overlap test

Success criteria:
- show whether higher-faithfulness subsets improve internal results, external transfer, or both
- if gains are weak, that is still publishable as evidence that benchmark utility is robust to moderate label noise

Paper impact:
- highest-value new experiment for acceptance
- should move from appendix into the main body if the effect is clear

### R3-E5. Expanded Faithfulness Audit

Purpose:
- make the corpus-quality claim more statistically convincing than the current `80`-review audit.

Reviewer concern addressed:
- current audit is honest but still relatively small.

Design:
- run a larger Batch-based audit on a broader sample, for example `250` reviews
- stratify by:
  - review length band
  - aspect count
  - pedagogical group
  - faithfulness-filter bucket if `R3-E4` is ready

Outputs:
- overall support and sentiment-match rates
- stratified audit tables
- error taxonomy on the worst sentiment-failure cases

Success criteria:
- reduce uncertainty around the current full-corpus audit
- identify where the corpus is strongest and weakest

Paper impact:
- strengthens Section `6.7B` and Appendix A.8

### R3-E6. Stronger External Validation

Purpose:
- strengthen the real-data story beyond one mapped evaluation.

Reviewer concern addressed:
- one real dataset and one 9-aspect overlap are not enough for broad claims.

Preferred path:
- add a second educational review dataset if one can be mapped conservatively and defensibly

Fallback path if no second dataset is feasible:
- run stricter robustness checks on the Herath overlap:
  - aspects with stronger support only
  - reviews with at least two mapped aspects
  - grouped results by pedagogical block
  - calibration-aware diagnostics

Success criteria:
- either broaden the real-data evidence, or make the current real-data claim more defensible and better qualified

Paper impact:
- strengthens Section `6.6` and the discussion

### R3-E7. Complete the Prompted GPT Family in Batch Mode

Purpose:
- align executed evidence with the prompted family defined in the methods.

Reviewer concern addressed:
- the described prompted family is broader than the current reported runs.

Model:
- `gpt-5.2` for the main reported benchmark
- optional `gpt-5.4` diagnostic slice only if needed for methodological comparison, not as the main leaderboard basis unless fully executed fairly

Variants:
- `zero-shot`
- `few-shot`
- `few-shot-diverse`
- `retrieval-few-shot`
- `two-pass`
- `aspect-by-aspect`

Locked settings:
- same `1000`-review test split
- same exact-key sparse JSON contract
- Batch mode only

Success criteria:
- method table and result table are fully aligned
- prompted decompositions are either competitive enough to report centrally or clearly weaker but still fully executed

Paper impact:
- refresh Section `6.4`, Table `7A`, Table `A5`, and the method-family descriptions

### R3-E8. Group-Wise Pedagogical Analysis

Purpose:
- show where the benchmark is educationally informative rather than only sparse and noisy.

Reviewer concern addressed:
- aggregate F1 alone does not show whether the pedagogical structure matters.

Groups:
- instructional quality
- assessment and course management
- learning demand and readiness
- learning environment
- engagement and value

Compared systems:
- best local two-step model
- best joint model
- best GPT variant
- best mapped real-transfer model

Outputs:
- group micro-F1
- group macro-F1
- group balanced accuracy
- group sentiment MSE

Success criteria:
- identify stable pedagogical patterns that help interpret both benchmark difficulty and transfer behavior

Paper impact:
- good candidate for one main-text figure and one appendix table

### R3-E9. Calibration and Ranking Diagnostics

Purpose:
- complement thresholded F1 with score-quality evidence.

Reviewer concern addressed:
- class sparsity can make thresholded metrics incomplete.

Scope:
- strongest two local models
- strongest joint model
- weakest repaired local model if `R3-E3` reveals a calibration issue

Metrics:
- macro AUROC
- macro average precision
- Brier score
- ECE

Success criteria:
- show whether the better models are better because of ranking quality, calibration quality, or threshold effects

Paper impact:
- appendix-first, with selective use in main text if it clarifies close rankings

## Execution Order

### Phase 1: Trust and reproducibility
1. `R3-E1`
2. refresh aggregate local benchmark outputs
3. regenerate benchmark-linked figures/tables

### Phase 2: Uncertainty and baseline repair
1. `R3-E2`
2. `R3-E3`
3. refresh Section `6.3` and Appendix A.9-A.10

### Phase 3: Dataset-quality science
1. `R3-E4`
2. `R3-E5`
3. refresh Section `6.7B` and discussion of limitations

### Phase 4: Prompt-family completion
1. `R3-E7`
2. refresh Section `6.4`, Table `7A`, and Appendix A.5

### Phase 5: Stronger real-data interpretation
1. `R3-E6`
2. `R3-E8`
3. `R3-E9`
4. refresh Section `6.6`, Table `8A-8C`, related appendix material, and discussion

## Task Granularity and Compute Policy

To keep the experiment queue auditable and resumable:

- every model run is an individual task
- every prompt variant is an individual task
- every external-eval run is an individual task
- GPU training is serialized
- large LLM evaluation and faithfulness auditing run through Batch
- raw outputs and normalized summaries are both preserved

No new grouped mega-batches should be introduced for local training.

## Expected Best-Case Outcome

If Round 3 succeeds, the paper can more safely claim:

- the `10K / 20`-aspect benchmark is reproducible and methodologically trustworthy;
- the strongest local results are stable enough to interpret;
- the benchmark remains useful even after confronting label-noise concerns directly;
- prompted LLM baselines are fully represented rather than selectively reported;
- the real-data story is still conservative, but better evidenced and better qualified;
- the educational aspect grouping is not cosmetic, but analytically meaningful.

## Minimum Acceptance-Focused Subset

If compute or time is limited, the smallest high-value subset is:

1. `R3-E1`
2. `R3-E2`
3. `R3-E3`
4. `R3-E4`
5. `R3-E7`

That subset directly addresses the benchmark trust issue, uncertainty, baseline credibility, label-faithfulness concern, and prompt-family completeness.
