# 2026-06-05 — Validation battery v5: signal, floor, ceiling, learning curve, synth->real transfer

**Status:** completed. Five experiments on Modal A10G (V1/V3/V4/B5) + one local-CPU floor (V2), all reusing the mode-A recipe (bert-base-uncased two-step, 192 tok, AdamW 3e-5, 3 epochs, patience 2, per-aspect val-calibrated thresholds). Harness: `paper/reviewer_ab_worker.py` (modes V1/V3/V4/B5), `paper/reviewer_v2_floor.py` (V2), `modal_reviewer_ab.py`, aggregate `paper/reviewer_v5_aggregate.py`.

## Question / hypothesis

The internal 20-aspect benchmark's detection micro-F1 sits at ~0.276. Is that (a) genuine learned signal above chance, (b) limited by label noise rather than the task, and (c) does synthetic supervision transfer to a real corpus? Falsifiers: if a label-permuted model also reached ~0.276, there would be no text->label signal; if clean-label filtering did not raise F1, the ceiling would be the task not the noise; if the learning curve were flat, more data would not help; if synth->real fine-tuning did not match real-only, transfer would not help the small real set.

## Setup (artifact paths)

- Internal corpus (20-aspect): `paper/reviewer_ab_data/generated_reviews_10k.jsonl` (10,000 row_id-aligned rows), three_way_split 8000/1000/1000, seeded; verified to match `eng.three_way_split` exactly.
- Per-row faithfulness: `paper/reviewer_ab_data/at_scale_per_row_scores.csv` (gpt-4.1-mini at-scale audit; 3,753 rows row_score==1.0, 6,671 rows >=0.5).
- Real corpus (9-aspect overlap, clean subset of the 20): `paper/reviewer_ab_data/herath_mapped_real_reviews_2829.jsonl` (2,829 mapped Herath reviews).
- Reference numbers: paper BERT full-test micro-F1 0.2760 (1-seed) / 0.2774 (4-seed, exp A); synthetic-only 9-aspect transfer 0.4593; real-only 9-aspect training 0.7673 (exp B).
- Seeds: V1/V4/B5 = 42,17,23,41 (4); V3 = 42,17 (2); V2 deterministic (seed 42).
- Compute: Modal A10G, ~14 containers fanned out via `.map()`, <$3 total.

## Procedure

1. **V1 permutation control.** Mode-A split; permute the train (text, label-vector) pairing with a distinct seeded RNG (label marginal preserved; ~0 fixed points on 8000 rows); train on the scrambled pairs; calibrate thresholds and evaluate on the unchanged real-label test split.
2. **V2 trivial floor (local CPU).** On the mode-A 20-aspect test split, compute detection micro-F1 for predict-all-negative, predict-all-positive, predict-by-train-frequency (present iff train prevalence > 0.5), and uniform-random per-aspect at train prevalence (200 draws).
3. **V3 learning curve.** Subsample the train split to [250,500,1000,2000,4000,8000] (nested seeded ordering), train each, evaluate the same full test split.
4. **V4 clean-label ceiling.** Filter train+calib+test to row_score==1.0 (and, separately, >=0.5), train+calibrate+evaluate on the filtered splits.
5. **B5 synth->real fine-tune.** Restrict to the 9 overlap aspects end-to-end. Pretrain a 9-aspect detection head on the synthetic train split; fine-tune the SAME head on the real-Herath train split (mode-B split + seed); evaluate on the real-Herath test split.

## Headline numbers

| Experiment | metric | value (mean) | 95% CI | reference | reading |
|---|---|---|---|---|---|
| V1 permutation | permuted-train test micro-F1 | **0.1822** | [0.1774, 0.1871] | real 0.276 | gap **0.094**; collapses to the all-positive floor -> genuine signal |
| V2 floor | predict-all-positive micro-F1 | 0.1829 | — | BERT 0.276 | highest trivial floor; BERT clears it by **+0.093** |
| V2 floor | all-negative / freq>0.5 / random@prev | 0.000 / 0.000 / 0.101 | — | — | other floors far below |
| V4 ceiling | row_score==1.0 micro-F1 | **0.3191** | [0.290, 0.348] | full 0.276 | **+0.043** clean jump |
| V4 ceiling | row_score>=0.5 micro-F1 | **0.3401** | [0.332, 0.348] | full 0.276 | **+0.064** clean jump -> ceiling is label noise |
| B5 transfer | synth->real real-test micro-F1 | **0.7839** | [0.758, 0.809] | real-only 0.7673 | **+0.017** over real-only, **+0.325** over synth-only 0.4593 |

**V3 learning curve (micro-F1, mean of seeds 42 & 17):**

| train size | 250 | 500 | 1000 | 2000 | 4000 | 8000 |
|---|---|---|---|---|---|---|
| micro-F1 | 0.183 | 0.190 | 0.207 | 0.227 | 0.251 | 0.285 |

Monotone increasing in BOTH seeds (no plateau); F1 rises with data from the chance floor (~0.183 at N=250) toward the full-data 0.276+.

## Per-config breakdown

- **V1 per-seed** (17/23/41/42): 0.1827 / 0.1785 / 0.1859 / 0.1819 — sd 0.0030, all at the all-positive floor.
- **V4 row_score==1.0 per-seed**: 0.3271 / 0.2989 / 0.3104 / 0.3401 (n_test 404/389/380/365). **>=0.5 per-seed**: 0.3353 / 0.3384 / 0.3472 / 0.3396 (n_test 683/668/673/665).
- **B5 per-seed** (synth->real, delta over real-only 0.7673): seed17 0.7819 (+0.015), seed23 0.8038 (+0.037), seed41 0.7854 (+0.018), seed42 0.7645 (−0.003). 3/4 seeds strictly above real-only. Fine-tune log confirms the head starts from synth-pretrained weights (val 0.60->0.74 over 3 fine-tune epochs).
- **V2 train prevalence**: every aspect's train prevalence is ~0.10 (max < 0.5), so predict-by-frequency degenerates to all-negative (0.0).

## Artifacts

- Per-run: `paper/experiment_rounds/validation_v5_20260605/{V1,V3,V4,B5}_seed{...}/` — `result.json`, `test_predictions*.jsonl` (per-row gold vs predicted), V1 `train_permutation.json`, B5 `split_indices.json`, `worker.log`.
- V2: `paper/experiment_rounds/validation_v5_20260605/V2_trivial_floor/{result.json,floors.csv}`.
- Aggregate: `paper/experiment_rounds/validation_v5_20260605/aggregate_summary.{json,csv}` (means + small-sample-t 95% CIs).
- Code: `paper/reviewer_ab_worker.py` (modes V1/V3/V4/B5), `paper/reviewer_v2_floor.py`, `paper/reviewer_v5_aggregate.py`, `modal_reviewer_ab.py`.

## Conclusion

The internal-benchmark 0.276 is (a) genuine signal: a label-permuted model collapses to the 0.182 all-positive floor (gap 0.094), and BERT clears the highest trivial floor by +0.093; (b) noise-capped, not task-capped: filtering to faithfully-labeled rows raises micro-F1 to 0.319 (==1.0) and 0.340 (>=0.5); (c) data-scalable: the learning curve rises monotonically 0.183 -> 0.285 across 250 -> 8000 train rows. Separately, B5 shows synthetic pretraining + real fine-tuning reaches 0.784 micro-F1 on the real 9-aspect set, matching-and-exceeding the real-only 0.767 (4-seed mean +0.017, 3/4 seeds above) and far above synthetic-only transfer 0.459. V1/V2/V3/V4 fold into the validation/diagnostics narrative; B5 is the transfer-improvement result.
