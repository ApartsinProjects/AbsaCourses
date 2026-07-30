# Human-annotation study (validation of the label-faithfulness audit)

Three annotators independently labeled a stratified sample of 300 synthetic reviews
(610 declared review-aspect decisions), blind to the declared labels. For each declared
aspect they marked whether it is expressed in the review and, if so, its sentiment.

## Files

- `annotation_rater{1,2,3}_res.xlsx` : the three completed annotation workbooks. Columns:
  `item_id`, `review_text`, `aspect_to_judge`, `aspect_present? (Yes/No)`,
  `sentiment_if_present`. Item order is shuffled independently per rater.
- `annotation_key.csv` : the hidden scoring key (declared sentiment + audit row-score per
  item). Not shown to raters.
- `score_human_study.py` : computes human-vs-declared agreement, human-vs-audit agreement
  (joined against the per-row audit judgments), inter-rater Cohen's and Fleiss' kappa, and
  human confirmation stratified by audit-score quartile.
- `human_study_scores.json` : the scored results.

## Headline results

- Inter-annotator reliability: Fleiss kappa 0.70 on aspect presence; pairwise Cohen kappa
  0.60, 0.66, 0.87.
- Audit validated on synthetic text: human majority confirmation of a declared aspect rises
  monotonically with the audit score, 0.546 -> 0.711 -> 0.750 -> 0.786 across audit-score
  quartiles; human and audit presence judgments agree 76% of the time.
- Human annotation reproduces the audit's presence-faithful (about 70% confirmed) versus
  sentiment-noisier (about 0.40 agreement, matching the audit's strict 0.42) split.

## Reproduce

```bash
python score_human_study.py
```

The scorer expects the repository layout (it reads the corpus, key, and audit outputs by
relative path); adjust the paths at the top of the script for a standalone run.
