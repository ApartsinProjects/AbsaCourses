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

**Status:** part 1 (distribution) DONE. Part 2 (exclude-and-rerun benchmark on the
9159 complete rows, multi-seed, vs full-corpus) pending — uses the saved exclusion set.
