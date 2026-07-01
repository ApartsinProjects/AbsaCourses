# 2026-06-13 — RC2: the 841 output-token-capped rows (reviewer Cycle 1)

**Reviewer ask:** are the 841 `incomplete` rows (hit the output-token cap)
distributed differently across aspect/sentiment, and does excluding them change
benchmark results?

**Reconciliation (exact, no proxy):** the final corpus's per-row API status was
NOT kept in the cleaned corpus, but the generating batch output was preserved at
`Submitted/paper/batch_results/batch_69cc15c483488190941478aa4e3a976d_output.jsonl`.
Joining on `custom_id = gen_<sample_id>` matches 10000/10000 and gives
`{completed: 9159, incomplete: 841}`, all `incomplete_details.reason =
max_output_tokens` — exactly the paper's 841. Script: `paper/rc2_incomplete_analysis.py`.

**Finding (incomplete vs complete):** truncation yields SHORTER, FEWER-aspect rows
(opposite of the naive guess): words mean 100 vs 118.8; aspects-per-review
1/2/3 = 0.43/0.51/0.07 vs 0.29/0.38/0.33 (only 7% of incomplete rows keep 3
aspects vs 33%); sentiment mix similar with slightly less neutral; modest aspect
shifts (pacing/workload/practical_application over ~1.2-1.3x; support/materials/
overall_experience under ~0.7-0.8x). The cap truncated later aspect declarations.

**Artifacts:** `paper/outputs/rc2_incomplete_row_ids.json` (the exact 841 sample
ids = exclusion set), `paper/outputs/rc2_incomplete_analysis.json` (distributions).

**Part 2 (exclude-and-rerun, 3 seeds, BERT detection):** full (10,000) micro-F1
0.275 +/- 0.003, macro bal-acc 0.619; complete-only (9,159) micro-F1 0.264 +/- 0.007,
bal-acc 0.613; delta -0.011 / -0.006. The full baseline reproduces the paper (0.275
~ 0.276). Excluding the 841 does NOT materially change the benchmark; the small drop
tracks the ~8% smaller train set (and is inconsistent across seeds), so the reported
results are robust to the incomplete rows. Artifacts: outputs/rc2_exclude_rerun_per_seed.csv,
outputs/rc2_exclude_rerun_summary.json. Driver: rc2_exclude_rerun.py.

**Status:** DONE (parts 1 and 2).
