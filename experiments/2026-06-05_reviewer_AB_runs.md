# 2026-06-05 — Reviewer-response experiments A & B (Modal A10G)

**Status:** completed. Two reviewer-bound experiments, BERT-base-uncased on Modal A10G, 4 seeds each (42, 17, 23, 41).

## Experiment A — Filtered-test detection F1 (TMLR reviewer concern 1)

**Question:** is the moderate held-out detection F1 (~0.276) partly TEST-LABEL NOISE rather than task difficulty? Recompute micro-F1 on the high-faithfulness subset of the test split vs the full test split.

**Setup:** bert-base-uncased two-step (sigmoid detection head + per-aspect weighted BCE, pos-weights clipped [1,50]; tanh sentiment + masked MSE), 192 tokens, AdamW 3e-5, up to 3 epochs, patience 2, per-aspect thresholds calibrated on validation. Corpus `paper/reviewer_ab_data/generated_reviews_10k.jsonl` (the row_id-aligned 10K the at-scale faithfulness audit scored), 8000/1000/1000 split. Faithfulness per row from `at_scale_per_row_scores.csv` (gpt-4.1-mini at-scale audit).

**Reproduce-gate (mandatory):** full-test micro-F1 = **0.2774**, 95% CI [0.2702, 0.2847] — reproduces the paper's 0.276 (1-seed) / 0.2791 (3-seed). Gate PASSED, so the filtered number is trustworthy.

**Headline:**
| subset | micro-F1 (4-seed mean) | 95% CI | mean n rows |
|---|---|---|---|
| full test | 0.2774 | [0.2702, 0.2847] | 1000 |
| faithful only (row_score = 1.0) | 0.3072 | [0.2772, 0.3371] | 384.5 |
| paired delta (faithful − full) | **+0.0297** | **[0.0059, 0.0536]** | wins 4/4 seeds |

Per-seed full / faithful: 17 → 0.280/0.304, 23 → 0.282/0.332, 41 → 0.272/0.286, 42 → 0.276/0.307. The paired delta CI excludes zero: detection is measurably more learnable on faithfully-labeled test rows, so part of the moderate F1 ceiling is test-label noise, not model limitation.

## Experiment B — Real-Herath-trained baseline (TMLR reviewer concern 4)

**Question:** same-task upper reference for the synthetic-only transfer (0.4593 detection micro-F1 on the 9-aspect Herath overlap). Train on REAL Herath, test on REAL Herath, identical 9-aspect detection metric.

**Setup:** same recipe as A. Mapped Herath `paper/reviewer_ab_data/herath_mapped_real_reviews_2829.jsonl` (2,829 reviews, 9 overlap aspects: accessibility, assessment_design, exam_fairness, grading_transparency, lecturer_quality, materials, organization, overall_experience, workload). Per-seed split ~1980 train / 283 calib / 566 test (split indices saved per seed).

**Headline:** real-trained micro-F1 = **0.7673**, 95% CI [0.7342, 0.8005] (4 seeds; per-seed 0.743/0.790/0.778/0.759). Synthetic-only reference 0.4593 → synthetic-only supervision recovers ~60% of real-trained detection F1 with NO real training data, on the identical metric.

## Artifacts (all saved + committed)
- Per-seed summaries: `paper/experiment_rounds/reviewer_AB_20260605/modal_summary_*.json`
- Aggregate: `paper/experiment_rounds/reviewer_AB_20260605/AB_aggregate_summary.json` (via `paper/aggregate_reviewer_ab.py`)
- Full per-run outputs pulled from the Modal volume: `paper/experiment_rounds/reviewer_AB_20260605/modal_volume/{A,B}_seed{17,23,41,42}/` — `result.json`, `test_predictions.jsonl` (per-row gold vs predicted), B's `split_indices.json` + `per_aspect.csv`, `worker.log`.
- Harness: `modal_reviewer_ab.py`, `paper/reviewer_ab_worker.py`. Inputs: `paper/reviewer_ab_data/`.
- Cost: Modal A10G, 8 runs, ~3 min (B) / ~9 min (A) each, <$2 total.

## Conclusion
Both reviewer items resolved with real, multi-seed, CI-backed numbers. A confirms the test-label-noise interpretation (paired delta CI excludes zero, 4/4 seeds). B gives the same-task real-trained reference (0.767 vs synthetic-only 0.459). Both fold into §5.3 (A) and §5.6 (B).
