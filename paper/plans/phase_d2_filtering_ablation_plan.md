# Phase D2: Faithfulness-Aware Filtering Ablation

Date: 2026-06-04
Status: scaffolded, not yet run

## Purpose

Convert the §6.7B faithfulness audit from a *limitation* (42% sentiment-match
on 250 rows) into a *contribution* by showing that filtering training data
on a quality signal improves downstream transfer to the mapped Herath real
benchmark.

This is the Phase D2 item from the integrated plan
[plans/integrated_paper_update_plan_20260403.md](integrated_paper_update_plan_20260403.md),
flagged there as "the single most valuable new experiment for acceptance."

## Hypothesis

H1. Training BERT-base on the top-N% of the synthetic corpus by faithfulness
score yields higher mapped-Herath micro-F1 than training on the full corpus.

H2. The bottom-25% bucket performs noticeably worse than the full corpus on
the same metric, confirming that the quality signal is informative.

H3. The effect is stronger on aspect-sentiment match than on aspect detection
(since the audit specifically targets sentiment-polarity faithfulness).

## Design

### Stage 1: cheap-model calibration

Goal: select a low-cost audit model whose row-level scores rank-correlate
with the existing GPT-5.2 audit at Spearman rho >= 0.6.

Inputs:
- `paper/faithfulness_audit/faithfulness_audit_gpt-5_2_250_details.csv` (existing)
- `paper/faithfulness_audit/faithfulness_audit_gpt-5_2_250_llm_responses.jsonl` (existing)

Procedure:
1. Pick a candidate cheap model (start with `gpt-4o-mini`; fall back to
   `gpt-4.1-mini` then `gpt-5.2` if rho is too low).
2. Re-run the same audit prompt over the same 250 rows via OpenAI Batch.
3. Compute per-row quality score from the candidate's responses (same
   scoring as the existing audit: 1.0 if all declared aspects are supported
   AND all sentiments match, partial-credit otherwise).
4. Compute Spearman rho between candidate scores and GPT-5.2 scores on the
   joint 250-row sample.
5. Accept the candidate iff rho >= 0.6; otherwise escalate.

Expected cost: ~$0.20 to $1 per candidate, batch turnaround under 24 h.

Output: `paper/faithfulness_audit/calibration_<candidate>_vs_gpt-5_2.json`
containing the rho, the per-row deltas, and the recommendation
(accept / escalate).

### Stage 2: at-scale audit

Goal: produce per-row faithfulness scores for all 10K rows of the synthetic
corpus using the accepted cheap model.

Inputs:
- `paper/generated_datasets/batch_69cc15c483488190941478aa4e3a976d_generated_reviews.jsonl`
- the cheap model id chosen in Stage 1

Procedure:
1. Build the Batch input file from the corpus using the same audit prompt
   the existing `paper/label_faithfulness_audit.py` already produces.
2. Submit via OpenAI Batch (reuse `paper/submit_faithfulness_audit_batch.py`).
3. Consume the results (reuse `paper/consume_faithfulness_audit_batch.py`).
4. Compute per-row scores using the same scoring function as Stage 1.

Expected cost: ~$5 to $15 with `gpt-4o-mini`; ~$50 to $150 with `gpt-5.2`.
Batch turnaround under 24 h.

Output:
`paper/faithfulness_audit/at_scale_<model>_per_row_scores.csv` with columns
`row_id`, `score`, `n_aspects`, `n_supported`, `n_matched`.

### Stage 3: bucket assignment

Cut the 10K corpus into four training subsets by quality score:

| bucket | definition | expected size |
|---|---|---:|
| `top25`  | rows in the top quartile of score | 2,500 |
| `top50`  | rows in the top two quartiles | 5,000 |
| `full`   | the entire corpus (baseline) | 10,000 |
| `bot25`  | rows in the bottom quartile | 2,500 |

The `top50` bucket is the headline candidate; `top25` tests whether
"smaller but cleaner" beats "larger but noisier"; `full` is the baseline;
`bot25` is the negative control that should underperform if the signal is
real.

To control for training-size confounds, also report a `random_5k` bucket
(a random 5,000-row sample with the same size as `top50` but no quality
signal) so the comparison is "filtered 5k" vs "random 5k" vs "full 10k"
and not just "smaller" vs "larger".

Output: `paper/faithfulness_audit/buckets/<bucket>.row_ids.txt`, one row id
per line, deterministic from a fixed seed.

### Stage 4: training

Train `bert-base-uncased` on each bucket using the same recipe as the Phase
A calibrated benchmark (so results are comparable to §6.3):
- 3 epochs detection head, 3 epochs sentiment head
- threshold calibration on a held-out validation split *drawn from the same
  bucket*
- seed 42 (single seed for the headline run; add seeds 3, 13 only if H1 is
  ambiguous after seed 42)

Reuse `paper/absa_model_comparison.py` with `--approaches bert-base-uncased`
and per-bucket `--data-path` override.

Output: `paper/experiment_rounds/phase_d2_filtering_<date>/runs/<bucket>/` per
existing convention (model_comparison_summary.csv, model_comparison_per_aspect.csv).

Compute budget: ~5 runs (4 buckets + random_5k) × ~25 min on RTX 2060 =
~2.5 h GPU time. If multi-seed needed: ~7.5 h.

### Stage 5: evaluation

For each trained model, evaluate on two test sets:

A. Held-out synthetic test split (same as §6.3 internal benchmark).
B. Mapped Herath external benchmark (same as §6.6 transfer table).

Reuse `paper/evaluate_synthetic_to_real_transfer.py` for (B).

### Stage 6: report

Produce one results table and one figure:

| bucket | n train | internal micro-F1 | Herath micro-F1 | Herath sentiment MSE |
|---|---:|---:|---:|---:|
| top25 | 2,500 | ? | ? | ? |
| top50 | 5,000 | ? | ? | ? |
| random_5k | 5,000 | ? | ? | ? |
| full | 10,000 | ? | ? | ? |
| bot25 | 2,500 | ? | ? | ? |

Figure: bar chart of Herath micro-F1 by bucket, with the `full` baseline
drawn as a horizontal reference line.

Headline numbers to extract: the delta `top50 - full` and `top25 - full` on
Herath micro-F1, and the delta `random_5k - top50` (the value of the quality
signal at fixed training size).

Output: `paper/outputs/tables/phase_d2_filtering_results.csv` and
`paper/outputs/figures/phase_d2_filtering_micro_f1.svg`.

## Outcome interpretation

| outcome | manuscript impact |
|---|---|
| `top50` and `top25` both beat `full` AND `random_5k` | strongest: filtering matters and it is the score that helps, not just the smaller training size |
| `top50` beats `full` but ties `random_5k` | weaker positive: smaller training size is the explanation; report and explain |
| `top50` ties `full` AND `random_5k` | null result; report cleanly as "data quality is not the bottleneck at this scale" |
| `bot25` does NOT underperform `full` | scoring is unreliable; report and explain |

Even the null result strengthens the paper because it converts an open
question into a documented experiment.

## Manuscript integration

If H1 is supported, add a `6.7D Faithfulness-Aware Filtering` subsection
between 6.7C and 6.8 with the table and figure above, and rewrite the §7.2
constraint paragraph to point to a method ("we identify low-faithfulness
rows and demonstrate that filtering improves transfer F1 by X points")
instead of a limitation. Update the abstract to mention the result in one
sentence.

If H1 is not supported, add the same subsection as a documented experiment
("we evaluated whether the audit signal predicts utility and found no
material effect"), and leave §7.2 as currently written.

## Reproducibility

- All buckets are deterministic functions of (corpus, audit-scores, seed=42).
- The audit input file, response JSONL, and per-row scores are committed
  under `paper/faithfulness_audit/` so the bucketing and training are
  reproducible from those files alone.
- The orchestrator is `paper/run_phase_d2.py` with `--stage` flags.

## Cost summary

| stage | model | est. cost | est. wall-clock |
|---|---|---:|---|
| 1 calibration | gpt-4o-mini Batch | < $1 | ~24 h |
| 2 at-scale audit | gpt-4o-mini Batch | $5–15 | ~24 h |
| 3 bucket | n/a (local) | $0 | minutes |
| 4 training | local RTX 2060 | $0 | ~2.5 h (single seed) |
| 5 evaluation | local | $0 | ~30 min |
| 6 report | local | $0 | minutes |
| **total** | | **~$5–15** | **~2 days calendar** |

If `gpt-4o-mini` calibration fails (rho < 0.6), escalate to `gpt-4.1-mini`
(~same cost) or `gpt-5.2` (~$50–150 for the at-scale step).
