# Response to Reviewer nfat

*A Controlled Synthetic Benchmark for Educational Aspect-Based Sentiment Analysis (TMLR)*

We thank Reviewer nfat for a careful and constructive report. Both the validation of the audit and the internal consistency of the reporting are now stronger. We answer each point with the change and its location.

### 1. Human validation of the audit on the synthetic corpus

**Requested.** Human-annotate a representative sample of the actual synthetic corpus and report aspect and sentiment agreement.

Done (new Table 15). Three annotators labeled 300 synthetic reviews (610 declared aspect decisions), shown each candidate aspect but blind to its declared presence and sentiment. Human confirmation of the declared aspect rises monotonically with the audit score (0.55 to 0.79 across quartiles, with Wilson intervals and a review-level clustered bootstrap), and the human labels reproduce the audit's presence-faithful, sentiment-noisier split (human sentiment agreement 0.38, matching the audit's strict 0.42). Inter-annotator reliability is substantial for presence (Fleiss 0.70) and moderate for sentiment (Fleiss 0.49). The audit score is therefore a human-validated selection signal on the synthetic corpus.

*Location:* Section 5.7 (Table 15).

### 2. Match filtering subsets to isolate faithfulness

**Requested.** Match retained and control subsets by aspect, polarity, aspect count, length, and style.

Done (new Table 14). A covariate-matched comparison pairs the retained and control subsets one-to-one on aspect set, aspect count, polarity, length band, and formality (3,441 pairs, audit-score means 0.90 versus 0.26) and scores both on the same gold-present cells, removing the prediction-mask confound. The faithfulness-retained subset has lower transferred sentiment error in every seed (0.412 versus 0.519) and higher detection micro-F1 (0.400 versus 0.338), so the gain is attributable to faithfulness alone.

*Location:* Section 5.7 (Table 14).

### 3. Reconcile inconsistent numbers

**Requested.** Resolve the Tables 8 and 9 BERT scores, the aspect-count totals, and the incomplete abstract.

Done. The aspect-count totals sum to 10,000 (3,032 + 3,917 + 3,051) and the abstract is complete. The Table 8 versus Table 9 discrepancy came from two single-seed runs; the transfer is now a multi-seed table, so every score traces to one consistent set of runs.

*Location:* Abstract; Section 5.1; Section 5.4 (Tables 8 and 9).

### 4. Shorten repeated discussion

**Requested.** Shorten repeated discussion.

Done. The generation process is stated once, the shared-split contract is cross-referenced, and the Section 5.8 recap is collapsed into a short bridge, while every experiment and caveat is retained.

*Location:* Sections 3, 5.8, 6.

### 5. Frame the resource as noisy synthetic supervision

**Requested.** Describe the resource as noisy synthetic supervision rather than a gold benchmark.

Done. The abstract, Section 6.2, and conclusion describe a controlled synthetic-supervision resource whose label faithfulness is measured (strict 0.42 per-aspect, 0.58 per-row) and controlled by the audit-and-filter pipeline, not a gold-labeled corpus.

*Location:* Abstract; Section 6.2; Section 7.

### 6. Broaden the impact discussion

**Requested.** Address instructor evaluation, comment privacy, bias against non-native writing, and fictional negative reviews, and require human review, uncertainty reporting, data protection, and an appeals process.

Done. The ethics statement adds stylistic-bias monitoring for non-native and non-standard writing, a prohibition on attaching model-inferred negatives to identifiable courses or instructors, and per-aspect uncertainty with low-confidence routing to human review and an appeals process, on top of the existing no-identifiable-data, licensing, and re-consent provisions.

*Location:* Section 6.3.

### 7. Rule out generator-specific split effects

**Requested.** Random splits from the same generator and prompt may reward generator-specific patterns.

Clarified, with a new control. Regenerating and auditing with three independent model families reproduces the label fidelity, and a held-out-generator check shows a detector trained on one generator's data transfers to other generators' data without collapse.

*Location:* Appendix A.3.2, A.5.4, A.3.3.

---

Locations reference the revised manuscript; every requested change is in place at the cited location.
