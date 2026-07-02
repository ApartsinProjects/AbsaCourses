# Integrated Paper Update Plan

Date: 2026-04-03

## Purpose

This document is the single source of truth for updating the paper. It integrates:

- the remaining experiments;
- the manuscript sections they support;
- the figures and tables they should refresh or create;
- the execution order;
- the reuse policy for already valid results.

The objective is to move from the current near-submission draft to a reviewer-defensible paper for a strong AI journal without letting experiments, manuscript edits, and artifact generation drift apart.

## Current Position

The paper is already strong in structure, clarity, and scope control. The main remaining risks are empirical rather than editorial:

1. benchmark trust and uncertainty;
2. synthetic-label faithfulness;
3. narrow external validation;
4. incomplete prompted-LLM family relative to the methods section;
5. need for stronger analysis beyond aggregate F1.

The paper should therefore now be treated as an evidence-refresh problem rather than a writing-from-scratch problem.

## Governing Principles

### P1. One paper, one main dataset

Use the current main synthetic corpus:

- [batch_69cc15c483488190941478aa4e3a976d_generated_reviews.jsonl](E:\Projects\CourseABSA\paper\generated_datasets\batch_69cc15c483488190941478aa4e3a976d_generated_reviews.jsonl)

Do not use the disabled legacy `edu` data path.

### P2. Reuse valid evidence

Do not rerun work that is still valid under the current paper protocol. Reuse already valid artifacts when:

- the dataset has not changed;
- the evaluation contract has not changed;
- the output bundle already preserves enough detail for the paper;
- the result is not affected by a fixed bug or protocol mismatch.

### P3. Rerun only what changed

Rerun experiments if any of the following changed:

- thresholding or calibration logic;
- metric definitions;
- artifact-preservation requirements;
- method-family completeness;
- dataset filtering policy;
- real-data mapping or external-validation protocol.

### P4. One experiment, one question

Every experiment in this plan must answer one reviewer concern and support one paper claim.

### P5. Preserve everything

All new runs must preserve:

- summary metrics;
- per-aspect metrics;
- sample-level predictions;
- detection probabilities and logits when available;
- thresholds;
- raw LLM responses and normalized parsed outputs for batch inference;
- detailed logs;
- resume checkpoints.

## What Can Be Reused Immediately

These artifacts are already valid unless replaced by fresher reruns under the same protocol:

### Reusable paper artifacts

- [course_absa_manuscript.html](E:\Projects\CourseABSA\paper\course_absa_manuscript.html)
- current publication figures and tables under:
  - [outputs/figures](E:\Projects\CourseABSA\paper\outputs\figures)
  - [outputs/tables](E:\Projects\CourseABSA\paper\outputs\tables)

### Reusable generator-side evidence

- realism-cycle artifacts under [validation](E:\Projects\CourseABSA\paper\validation)
- prompt package under [generation_protocol](E:\Projects\CourseABSA\paper\generation_protocol)
- current 10K generation outputs under [generated_datasets](E:\Projects\CourseABSA\paper\generated_datasets)

### Reusable real-data mapping artifacts

- [synthetic_to_real_transfer_summary.csv](E:\Projects\CourseABSA\paper\real_transfer\synthetic_to_real_transfer_summary.csv)
- [overlap_internal_vs_external_summary.csv](E:\Projects\CourseABSA\paper\real_transfer\overlap_internal_vs_external_summary.csv)
- overlap-support tables in [real_transfer](E:\Projects\CourseABSA\paper\real_transfer)

These should be refreshed only if reruns alter the underlying metrics or if stricter external-validation checks are added.

## What Must Be Refreshed

### M1. Main local benchmark

Reason:
- trust repair and alignment with the current evaluation stack.

Needs:
- calibrated rerun;
- aggregate refresh;
- multiseed uncertainty;
- baseline repair/tuning evidence.

### M2. GPT benchmark family

Reason:
- the paper now presents GPT-based inference as a tested method family, so the executed evidence should match the described family more completely.

Needs:
- full-test Batch evaluation for the complete planned prompted family.

### M3. Faithfulness evidence

Reason:
- this is the single biggest scientific concern in the current review.

Needs:
- larger audit;
- filtering ablation;
- clearer paper integration.

### M4. External validation

Reason:
- the current mapped real-data result is good but still narrow.

Needs:
- stronger overlap robustness checks or a second mapped educational dataset;
- better comparison framing if a second dataset is not feasible.

### M5. Extended diagnostics

Reason:
- the paper now benefits from metrics and analyses that are less brittle than F1 alone.

Needs:
- group-wise performance;
- calibration and ranking diagnostics;
- refreshed appendix and selective main-text use.

## Integrated Experiment Program

## Phase A. Benchmark Trust Repair

### A1. Calibrated Main Benchmark Rerun

Question:
- do the main local benchmark numbers remain under the exact validation-calibrated protocol described in the paper?

Run:
- `tfidf_two_step`
- `distilbert-base-uncased`
- `bert-base-uncased`
- `albert-base-v2`
- `roberta-base`
- `bert_joint`
- `distilbert_joint`

Outputs:
- refreshed `model_comparison_*`
- new run bundle under `benchmark_outputs/runs/`
- updated aggregate paper-facing files

Paper updates:
- Section `4.1`
- Section `4.2`
- Section `6.3`
- Table `7`
- Figure `6`
- Appendix `A.9`
- Appendix `A.10`

### A2. Aggregate Refresh Step

Purpose:
- after `A1`, update the paper-facing benchmark files and regenerate dependent visual artifacts.

Outputs:
- refreshed:
  - [model_comparison_summary.csv](E:\Projects\CourseABSA\paper\benchmark_outputs\model_comparison_summary.csv)
  - [model_comparison_per_aspect.csv](E:\Projects\CourseABSA\paper\benchmark_outputs\model_comparison_per_aspect.csv)
  - [model_comparison_metadata.json](E:\Projects\CourseABSA\paper\benchmark_outputs\model_comparison_metadata.json)

## Phase B. Uncertainty and Stronger Local Baselines

### B1. Multi-Seed Stability

Question:
- are the leading local model rankings robust across seeds?

Run:
- `tfidf_two_step`
- `distilbert-base-uncased`
- `bert-base-uncased`
- `bert_joint`
- `distilbert_joint`

Seeds:
- `3`, `13`, `23`

Paper updates:
- Section `6.3`
- Appendix `A.9`
- new stability figure/table if materially useful

### B2. Baseline Repair and Tuning

Question:
- are weaker baselines genuinely weak, or are some of them failing because of calibration or recipe issues?

Subtasks:
- inspect `roberta-base` and `albert-base-v2` for threshold pathology and prediction saturation
- tune `bert-base-uncased` and `distilbert-base-uncased`

Paper updates:
- Section `6.3`
- Table `7` if rankings change
- Appendix `A.9`
- tuning summary in appendix or main text if gain is meaningful

## Phase C. Prompted LLM Family Completion

### C1. Full Batch GPT Evaluation

Question:
- does the executed GPT family match the method family described in the paper?

Run in Batch mode:
- `zero-shot`
- `few-shot`
- `few-shot-diverse`
- `retrieval-few-shot`
- `two-pass`
- `aspect-by-aspect`

Primary model:
- `gpt-5.2`

Optional diagnostic:
- small `gpt-5.4` slice only if needed for prompt-method analysis, not as the headline leaderboard unless fully comparable

Paper updates:
- Section `4.2`
- Section `6.4`
- Table `7A`
- Table `A5`
- any GPT-related comparison figure if created

## Phase D. Corpus-Quality Science

### D1. Expanded Faithfulness Audit

Question:
- how reliable are declared aspect labels and polarities at corpus scale with lower uncertainty than the current `80`-review audit?

Run:
- Batch audit on a larger sample, preferably `250`
- stratify by:
  - aspect count
  - review length band
  - pedagogical group

Paper updates:
- Section `6.7B`
- Table `8D`
- Table `A7`
- discussion of corpus quality

### D2. Faithfulness-Aware Filtering Ablation

Question:
- does training on higher-faithfulness synthetic data improve downstream results?

Train on:
- full corpus
- top `50%`
- top `25%`
- bottom `25%`

Models:
- `bert-base-uncased`
- `distilbert-base-uncased`

Evaluate on:
- full synthetic test
- mapped real overlap test

Paper updates:
- likely main text if effect is clear
- otherwise appendix plus discussion/limitations

This is the single most valuable new experiment for acceptance.

## Phase E. Stronger External Validation

### E1. Second Real-Data Path or Robust Overlap Checks

Question:
- does the external story remain credible beyond the current one-corpus mapped-overlap result?

Preferred:
- identify and map a second educational review dataset

Fallback:
- stricter Herath overlap checks:
  - stronger-support aspects only
  - reviews with at least two mapped aspects
  - group-wise external results
  - balanced metrics and calibration-aware interpretation

Paper updates:
- Section `6.6`
- Table `8A`
- Table `8B`
- Table `8C`
- Figure `8`
- Figure `9`
- discussion and limitations

## Phase F. Extended Diagnostics

### F1. Group-Wise Pedagogical Analysis

Question:
- which pedagogical aspect groups are easiest or hardest, and is that pattern consistent across local, GPT, and real-transfer evaluation?

Groups:
- instructional quality
- assessment and course management
- learning demand and readiness
- learning environment
- engagement and value

Paper updates:
- Section `6.5` and/or `6.6`
- one main-text figure if clean
- appendix table for full metrics

### F2. Calibration and Ranking Diagnostics

Question:
- do score-based diagnostics explain model differences beyond thresholded F1?

Metrics:
- macro AUROC
- macro AP
- Brier
- ECE

Paper updates:
- appendix-first
- reference in Section `6.3` if it helps interpret close local rankings

## Manuscript Update Plan

After each experimental phase, update the paper immediately rather than waiting for every experiment to finish.

### After Phase A

Update:
- benchmark numbers
- trust language
- figure/table values tied to local benchmark

### After Phase B

Update:
- stability narrative
- tuned-versus-default interpretation
- appendix robustness section

### After Phase C

Update:
- GPT method-family description
- GPT results table
- any overstated or understated prompt-family prose

### After Phase D

Update:
- corpus-faithfulness interpretation
- limitations and contribution framing
- promote filtering ablation into the main body if results are strong

### After Phase E

Update:
- external validation framing
- transfer claim wording
- comparison to prior real educational sentiment work

### After Phase F

Update:
- richer diagnostic interpretation
- appendix metrics sections
- final consistency pass

## Figure and Table Refresh Map

### Main text

- Table `7`: refreshed after Phase A/B
- Figure `6`: refreshed after Phase A/B
- Table `7A`: refreshed after Phase C
- Table `8A-8C`: refreshed after Phase E
- Table `8D`: refreshed after Phase D
- Figure `8-9`: refreshed after Phase E
- Figure `7`: refresh only if per-aspect rankings change materially
- Figure `4-5`: refresh only if dataset filtering or corpus-profile reporting changes materially
- Figure `10`: refresh only if realism reruns are performed

### Appendix

- Table `A5`: refreshed after Phase C
- Table `A7`: refreshed after Phase D
- Table `A8-A10`: refreshed after Phase A/B/F
- add or refresh:
  - stability table
  - tuning table
  - group-wise diagnostics table
  - calibration table/figure

## Execution Policy

### Task structure

- one model per task
- one prompt variant per task
- one external-validation slice per task

### Compute policy

- serialize GPU training
- keep paper editing and Batch jobs in parallel
- preserve resume checkpoints at approach-level granularity

### OpenAI policy

- large GPT evaluation uses Batch
- large faithfulness audit uses Batch
- live API calls remain smoke/debug only

## Success Gates

The paper should be treated as substantially stronger only if these gates are met:

1. calibrated benchmark rerun completes cleanly;
2. uncertainty is reported for the leading local models;
3. prompted family is fully aligned with the paper’s method description;
4. faithfulness concern is investigated experimentally, not only acknowledged;
5. external validation is either broadened or made more rigorous;
6. aggregate claims stay aligned with executed evidence after refresh.

## Minimum High-Value Subset

If time or compute becomes limiting, prioritize:

1. Phase A
2. Phase B
3. Phase C
4. Phase D

That subset already addresses the core reviewer concerns about trust, uncertainty, method completeness, and label fidelity.

## Final Deliverable

When this integrated plan is complete, the paper should have:

- a refreshed and trustworthy main benchmark;
- uncertainty-aware local results;
- a fully executed prompted GPT comparison;
- experimentally grounded handling of faithfulness;
- a stronger and better qualified external-validation section;
- refreshed figures and tables tied directly to the final evidence layer.
