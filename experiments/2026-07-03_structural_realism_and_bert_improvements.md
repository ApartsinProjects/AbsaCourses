# 2026-07-03 - Structural realism gaps + BERT-ABSA transfer improvements

Two confirmed structural ways the synthetic corpus differs from real reviews, and
five experiments on improving BERT-ABSA transfer to long multi-aspect documents.

## Structural realism gaps (both confirmed on clean data)
1. **Polarity consistency (A.25).** Real multi-aspect reviews are opinion-consistent
   (halo effect): all-same-polarity 0.50 (Herath) / 0.62 (EduRABSA) / 0.72 (OATS);
   synthetic samples aspect polarities independently -> only 0.22 all-same, 0.33
   mix positive+negative. Script: within_review_sentiment.py. In paper (A.25/Table A23).
2. **Aspect co-occurrence (Part B).** The generator draws aspect sets uniformly
   (rng.sample), so NO pair co-occurs above chance: synthetic log-lift std 0.19,
   0% of pairs over/under-represented. Real reviews cluster semantically:
   Herath log-lift std 0.47 (17% pairs >1.5x, 14% <0.67x), OATS 0.30, EduRABSA 0.30.
   Real clusters: materials+organization 2.9x, clarity+organization 2.3x,
   assessment_design+grading_transparency 1.9x. Script: analysis_stratified_cooccur.py.

## Stratified + prior-corrected benchmark (Part A)
BERT in-domain benchmark stratified by consistency, then post-stratified to the real
stratum distribution (single 0.39 / consistent 0.39 / mixed 0.22 vs synthetic
0.30 / 0.15 / 0.55 -- synthetic over-represents hard mixed reviews 2.5x).
- Per stratum: det micro-F1 single 0.178 / consistent 0.294 / mixed 0.316;
  gold sentiment MSE single 0.473 / consistent 0.399 / mixed 0.596.
- RAW pooled det-F1 0.276, sent-MSE 0.546.
- PRIOR-CORRECTED to real: det-F1 0.259 (mildly deflated -- detection is easier on
  dense mixed reviews), sent-MSE 0.475 (improved -- synthetic test over-weights hard
  mixed reviews, so the raw number understates the sentiment head). ESS 614/1000.
Two-sided honest correction. Script: analysis_stratified_cooccur.py.

## Control for structure during transfer measurement (Part C)
Post-stratify the transfer metric to the real joint (aspect-count x consistency x
pairwise co-occurrence) distribution -- generalizes Part A. Practical: raking to match
real pairwise co-occurrence marginals; report per-aspect (marginal) F1 alongside micro;
state residual assumption (higher-order structure uncorrected).

## Exp1 - Windowed (per-sentence) inference + OR-pool (WIN)
Reuse the synthetic-trained model; split each real review into sentences, predict per
sentence, OR-pool detection to review level, take sentiment from the max-prob sentence.
- OATS (70w multi-sentence): whole-doc micro-F1 0.2754 -> windowed 0.4219 (+0.147,
  +53% rel), recall 0.253 -> 0.592. Windowed BERT now beats TF-IDF (0.343) and whole-doc.
- Herath (20w, ~1 sentence): 0.3268 -> 0.3299 (+0.003) -- CONTROL, no-op as expected.
- Sentiment MSE mildly worse under windowing (single-sentence polarity): OATS 0.446->0.512.
Mechanism: whole-doc BERT dilutes a localized aspect signal over ~100 words; per-sentence
inference concentrates it. Training-free. Scripts: exp1_windowed_transfer.py. -> paper.

## Exp2 - Sentence-level training (lexicon localization)   [RESULTS PENDING]
Localize each declared aspect to sentence(s) via a 20-aspect lexicon; train BERT on
10000 reviews -> 41593 sentences (0.54 aspects/sent, 62% empty as negatives); eval
Herath/M-ABSA/OATS. Script: exp2_sentence_train.py.

## Exp3 - Aspect-query (NLI) detection   [NULL - not for paper]
Represent each aspect as a phrase (multi subword tokens); classify (review [SEP] aspect
phrase) -> present/absent. Tests meaning-based aspect matching for cross-domain transfer.
Fixed: transformers.AdamW removed -> torch.optim.AdamW. Script: exp3_aspect_query.py.
RESULT (NULL/negative): aspect-query detection micro-F1 UNDERperforms the multi-label
baseline on both targets run: Herath 0.288 (vs baseline 0.327, sentence-train 0.369);
M-ABSA 0.188 (vs baseline 0.263, sentence-train 0.558). Train loss stayed flat
(0.559->0.571) -- a from-scratch pairwise classification head does not learn aspect
matching in 3 epochs and loses the multi-label head's shared representation. OATS target
skipped (both done targets clearly null). Stays in registry, not the paper.

## Exp4 - Correlated-sentiment regeneration   [WIN]
Regenerate 2000 reviews drawing a review-level disposition then conditioning per-aspect
polarity (all-same 0.51 vs 0.22 original, confirmed). Retrain sentiment on correlated vs
equal-N original; gold-present sentiment MSE on Herath/OATS. Scripts: exp4_generate.py,
exp4_train.py.
RESULT (WIN): correlated-sentiment training LOWERS gold-present sentiment MSE on both
real targets: Herath 0.473 vs 0.573 original (-0.100); OATS 0.453 vs 0.513 (-0.060).
Equal-N (~1100/1240). Fixing the A.25 polarity-independence gap causally improves the
sentiment head's transfer.

## Exp5 - Co-occurrence-aware + correlated generation   [RESULTS PENDING]
Bootstrap real aspect-sets (reproduces real co-occurrence log-lift std 0.386, in real
range) + correlated polarity (0.54). Fixes BOTH structural gaps. Retrain vs equal-N
original; eval OATS whole + windowed micro-F1 + sentiment MSE. Scripts: exp5_generate.py,
exp5_train.py.
Validated 3-way structural ladder (2000 gen each) [allsame / cooccur-loglift-std]:
  original 0.22 / 0.11  ->  exp4 polarity-only 0.51 / 0.22  ->  exp5 both 0.55 / 0.462.
  Real targets: OATS 0.72 / 0.30, Herath 0.50 / 0.47. exp5 matches real on both axes.
RESULT (WIN, OATS, equal-N ~1900 structured vs original):
  whole-doc detection F1 0.469 vs 0.399 (+0.070); windowed F1 0.459 vs 0.469 (~same);
  whole sentiment MSE 0.246 vs 0.740 (-0.494). Co-occurrence structure lifts whole-doc
  detection; correlated polarity slashes sentiment MSE. Windowed inference already
  concentrates signal, so structure adds little on top of windowing for detection.

## Overall narrative (today)
Two orthogonal levers improve synthetic->real ABSA transfer:
(1) GRANULARITY MATCHING: windowed inference for review targets (Exp1, OATS +0.147);
    sentence-level training for sentence targets (Exp2, M-ABSA +0.295, Herath +0.042).
(2) STRUCTURAL REALISM: correlated polarity lowers sentiment MSE (Exp4, -0.06..-0.10;
    Exp5 -0.49 on OATS); real aspect co-occurrence lifts whole-doc detection (Exp5 +0.070).
Aspect-query (Exp3) is a documented null (underfit; needs an NLI checkpoint).
Caveat to close: confirm the M-ABSA review-trained baseline (0.263) with the identical
harness/config before the +0.295 reaches the paper.

## Data integrity
All analyses use the regenerated clean Herath (2829 rows); the corrupted-cache bug
(herath_mapped overwritten by other transfer runs) is fixed and gitignored. exp1 confirms
Herath transfer path unaffected.
