# 2026-06-05 — Reviewer-response evidence (TMLR revision)

## Question / hypothesis
A TMLR reviewer raised three concerns about CourseABSA. Produce real, audit-trailed
numbers from EXISTING on-disk artifacts (no retraining, no API) to address them:

1. **Test-split label faithfulness.** Is the low absolute held-out F1 partly a
   product of test-LABEL noise (unfaithful synthetic labels) rather than task
   difficulty? If the held-out split has the same ~42%-style mismatch as the full
   corpus, the achievable F1 ceiling is depressed by label noise.
2. **Detection-set robustness of the sentiment-MSE contrast.** The headline
   `sentiment_mse_detected` is conditioned on each bucket's OWN detected-aspect set.
   Are the headline contrasts (top50 vs random_5k; bot25 vs full) explained by the
   buckets scoring different detection sets, or by genuine sentiment quality?
3. **Multi-judge convergence.** The faithfulness audit is entirely LLM-based.
   How well do the cost-matched judges agree per-aspect with the gpt-5.2 reference?

## Setup (input artifacts — all already on disk)
- Item 1: `paper/faithfulness_audit/at_scale_gpt-4.1-mini_per_row_scores.csv`
  (10,000 per-row faithfulness scores; `row_id` = 0-based line index into the 10K
  synthetic training corpus).
- Item 2: `paper/experiment_rounds/phase_d2_filtering_20260604/runs/<bucket>[_seed{N}][_distilbert][_edurabsa]/run/summary.csv`
  (160 runs = 4 arch×target × 5 buckets × 8 seeds). BERT-Herath seed 42 is the BARE
  bucket dir (no `_seed42`); all other combos use `_seed42{suffix}`.
- Item 3: candidate raw responses
  `paper/faithfulness_audit/calibration_{gpt-4.1-mini,gpt-4o-mini,claude-3_5-haiku}_responses.jsonl`
  + gpt-5.2 reference
  `E:\Projects\CourseABSA\paper\faithfulness_audit\faithfulness_audit_gpt-5_2_250_llm_responses.jsonl`
  + the three `calibration_*_vs_gpt-5_2.json` files.
- Split logic source: `edu/absa_train_new.ipynb` (`three_way_split`, two nested
  `sklearn.train_test_split(random_state=42)`); manuscript Section 5.9 (line 597).

## Procedure
- **Item 1** (`paper/_reviewer_item1_faithfulness_split.py`): co-compute, in one
  pass, mean support_rate / match_rate / row_score and the quantized row_score
  distribution over (a) full 10K, (b) two seed-42 1000-row holdouts. The exact
  internal-benchmark TEST row_ids cannot be reconstructed with certainty (notebook
  splits via nested sklearn on `o4_mini_final_student_reviews_clean.jsonl`, which is
  NOT on disk, while the manuscript describes a numpy seeded permutation — different
  partitions). Both holdout reconstructions are reported as labelled fallbacks.
- **Item 2** (`paper/_reviewer_item2_bucket_mse_recall.py`): co-compute, in one pass
  over the SAME 8 seeds, the per-bucket 8-seed mean of `sentiment_mse_detected` AND
  `micro_recall` (and precision/F1) for all 4 (arch,target) cells. Reconcile
  BERT-Herath MSE means against manuscript Table 8E.
- **Item 3** (`paper/_reviewer_item3_judge_agreement.py`): per-aspect support and
  match agreement of each cost-matched judge vs gpt-5.2, computed over the
  (row_id, aspect) pairs both judges scored — the same metric the Haiku JSON stores.
  Methodology cross-check: recompute Haiku from raw responses and confirm it
  reproduces its stored values.

## Headline numbers

### Item 1 — test-split faithfulness (gpt-4.1-mini at-scale audit)
| subset | n | mean support_rate | mean match_rate (= row_score) | %0.00 / 0.33 / 0.50 / 0.67 / 1.00 |
|---|---|---|---|---|
| full 10K | 10000 | 0.8421 | 0.5766 | 22.16 / 11.13 / 18.17 / 11.01 / 37.53 |
| seed42 holdout (numpy perm, matches manuscript text) [FALLBACK] | 1000 | 0.8370 | 0.5763 | 22.30 / 10.40 / 18.20 / 12.10 / 37.00 |
| seed42 holdout (nested sklearn, matches notebook code) [FALLBACK] | 1000 | 0.8443 | 0.5750 | 23.20 / 9.30 / 19.00 / 10.80 / 37.70 |

- The held-out 1000-row split has the SAME faithfulness as the full corpus
  (match_rate 0.576 vs 0.576/0.575); both reconstructions agree to within 0.001, so
  the conclusion is robust to which split procedure is assumed.
- Pooled per-aspect match rate over the full 10K = 0.5597 (20,019 declared aspects).
  The paper's "0.42 match rate" is the 250-row **gpt-5.2** per-aspect audit (0.4232);
  the at-scale **gpt-4.1-mini** per-aspect match rate is higher (0.560), as expected
  for a different judge.
- **Filtered-test F1 NOT computable:** no per-row TEST predictions exist on disk
  (searched E:\Projects\CourseABSA and E:\Claude\CourseABSA for `test_results*.csv` /
  `*test_pred*`). The high-faithfulness-subset detection-F1 contrast is intentionally
  omitted, NOT fabricated.

### Item 2 — per-bucket 8-seed mean sentiment_mse_detected + micro_recall
**BERT-Herath** (reconciles with Table 8E within 0.001):
| bucket | MSE (detected) | micro_recall | micro_prec | micro_f1 |
|---|---|---|---|---|
| top25 | 0.3890 | 0.2929 | 0.3729 | 0.3165 |
| top50 | 0.3563 | 0.3709 | 0.3240 | 0.3407 |
| full | 0.3513 | 0.3802 | 0.3427 | 0.3542 |
| bot25 | 0.7097 | 0.7925 | 0.1906 | 0.3069 |
| random_5k | 0.4105 | 0.4378 | 0.3335 | 0.3727 |

Table 8E reconciliation: top25 0.389/0.389, top50 0.356/0.3563, full 0.351/0.3513,
bot25 0.710/0.7097, random_5k 0.411/0.4105 — all match within 0.001.

DistilBERT-Herath / BERT-EduRABSA / DistilBERT-EduRABSA per-bucket MSE+recall are in
the artifact JSON/CSV.

**Detection-set comparability of the two headline contrasts (BERT-Herath):**
- `top50 vs random_5k`: top50 has LOWER recall (0.371 vs 0.438) yet LOWER MSE
  (0.356 vs 0.411). The filtering win is not produced by an easier/larger detected
  set; top50 scores fewer aspects and still has lower polarity error. The contrast is
  conservative, not confounded.
- `bot25 vs full`: bot25 has MUCH HIGHER recall (0.793 vs 0.380) but MUCH WORSE MSE
  (0.710 vs 0.351). bot25 degenerates into a high-recall/low-precision detector and
  still has worse sentiment, so its MSE blow-up is not an artifact of scoring a
  smaller/harder detected set. (Same direction holds on all 4 arch×target cells.)

### Item 3 — per-judge per-aspect agreement vs gpt-5.2 (250-row sample, 501 ref aspects)
| judge | n_aspects_common | support agreement | match agreement | spearman_rho (stored) |
|---|---|---|---|---|
| gpt-4.1-mini (at-scale judge) | 501 | 0.8743 | 0.7904 | 0.5204 |
| gpt-4o-mini | 498 | 0.8454 | 0.7149 | 0.4593 |
| claude-3.5-haiku | 497 | 0.8169 | 0.6781 | — |

- Methodology cross-check: recomputing Haiku from raw responses reproduces its stored
  0.8169 / 0.6781 (n=497) exactly.
- **Manuscript label-swap flagged:** line 1018 cites gpt-4.1-mini at "support 0.845 /
  match 0.715". Those are gpt-4o-mini's values (0.8454 / 0.7149). The TRUE gpt-4.1-mini
  agreement is 0.8743 / 0.7904 (HIGHER, more favorable). The at-scale audit was run
  with gpt-4.1-mini (`_d2_atscale.py` MODEL='gpt-4.1-mini'), so the manuscript should
  cite 0.874 / 0.790.
- **Behavioral validity (strongest argument):** the bot25 negative control (lowest-
  audit rows) collapses downstream transfer on ALL (arch,target) conditions (Item 2),
  showing low-audit rows are genuinely worse training data independent of any human
  label.
- **Human study:** designed at `human/` (codebook.md, task_1, task_3) but the
  faithfulness/judge-agreement tasks have NO collected responses (empty .gitkeep). A
  separate real-vs-synthetic discrimination study (task_9) has one rater file but is a
  different study. Faithfulness/judge-agreement human validation is
  specified-but-not-executed; no human faithfulness results exist.

## Artifacts saved
- `paper/outputs/tables/reviewer_response_item1_testsplit_faithfulness.{json,csv}`
- `paper/outputs/tables/reviewer_response_item2_bucket_mse_recall.{json,csv}`
- `paper/outputs/tables/reviewer_response_item3_judge_agreement.{json,csv}`
- Scripts: `paper/_reviewer_item1_faithfulness_split.py`,
  `paper/_reviewer_item2_bucket_mse_recall.py`,
  `paper/_reviewer_item3_judge_agreement.py`

## Conclusion
1. The held-out split is as label-noisy as the full corpus (match_rate 0.576),
   supporting the rebuttal that the low F1 ceiling is partly test-label noise, not
   only task difficulty. Filtered-test-F1 not computable from saved artifacts.
2. The MSE contrasts are NOT confounded by detection set: in both headline contrasts
   the detection-rate difference runs opposite to (top50 win) or fails to explain
   (bot25 blow-up) the MSE difference, making the sentiment-quality interpretation
   conservative. BERT-Herath MSE reconciles exactly with Table 8E.
3. All three cost-matched judges converge with gpt-5.2 (support 0.82–0.87, match
   0.68–0.79); behavioral evidence (bot25 collapse) corroborates the audit
   independent of human labels. Flagged a manuscript label-swap (gpt-4.1-mini cited
   with gpt-4o-mini's agreement numbers; true gpt-4.1-mini numbers are higher).

## Status: completed
