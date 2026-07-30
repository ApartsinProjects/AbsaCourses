# Response to Reviewer nfat

*A Controlled Synthetic Benchmark for Educational Aspect-Based Sentiment Analysis (TMLR)*

We are grateful to Reviewer nfat for a careful and constructive report, and especially for pressing on the validation of the audit and the internal consistency of the reporting. Both have made the paper stronger. We answer each point below with the specific change and its location in the revised manuscript.

### 1. Human validation of the audit on the synthetic corpus

**Requested.** Human-annotate a representative sample of the actual synthetic corpus; the auditor is validated mainly on perturbed real labels rather than direct human annotation of synthetic reviews.

Done (new Table 14). Three annotators independently labeled a stratified sample of 300 synthetic reviews (610 declared review-aspect decisions), blind to the declared labels, marking per-aspect presence and sentiment. Inter-annotator reliability is substantial (Fleiss kappa 0.70 on aspect presence; pairwise Cohen kappa 0.60 to 0.87). The annotation validates the audit directly on synthetic text three ways: human confirmation of the declared aspect rises monotonically with the audit score (55% in the lowest audit-score quartile to 79% in the highest); human and audit presence judgments agree 76% of the time; and the human annotation independently reproduces the audit's central finding that aspect presence is the faithful signal (70% human-confirmed) while aspect sentiment is the noisier one (human-declared agreement about 0.40, closely matching the audit's strict 0.42). The audit score is therefore a valid, human-grounded quality signal on the synthetic corpus, and the corpus's measured noise level is accurate.

*Location:* Section 5.7 (Table 14); Section 6.1.

### 2. Match filtering subsets to isolate faithfulness

**Requested.** Sentiment MSE is measured only on aspects each model predicts, so filtering comparisons may be confounded; match subsets by aspect, polarity, aspect count, length, and style.

Done (new Table 13). We add a covariate-matched filtering comparison that matches the retained and control subsets one-to-one on aspect set, aspect count, polarity composition, length band, and formality band, so the two subsets are identical on every covariate except the audit score itself (matched-pair audit-score means 0.90 versus 0.26; 3,441 pairs). Both are then trained and scored on the same common gold-present aspect cells on the 9-aspect Herath overlap (4,289 shared cells), which removes the prediction-mask confound entirely. Under this strict design the faithfulness-retained subset has the lower transferred sentiment error in every seed (sentiment MSE 0.412 versus 0.519, a paired reduction of 0.108 across three seeds) and a higher detection micro-F1 (0.400 versus 0.338), so the filtering gain is attributable to label faithfulness alone rather than to differing composition or prediction masks. This complements the size-matched result already reported (retaining the top 50% cuts sentiment error at half the training cost, 7 of 8 seeds, replicated across architectures).

*Location:* Section 5.7 (Table 13).

### 3. Reconcile inconsistent numbers

**Requested.** Resolve inconsistent numbers, especially the BERT scores in Tables 8 and 9, aspect-count totals that do not sum to 10,000, and the incomplete abstract.

Done. The aspect-count totals now sum to 10,000 (3,032 plus 3,917 plus 3,051) and the abstract is complete. The Table 8 versus Table 9 BERT discrepancy arose because the two tables were built from two separate single-seed transfer runs; we now report the transfer with a multi-seed table so every transfer score and the derived generalization gap trace to one consistent set of runs.

*Location:* Abstract; Section 5.1; Section 5.4 (Tables 8 and 9).

### 4. Shorten repeated discussion

**Requested.** Shorten repeated discussion.

Done. We streamlined the discussion by removing verbatim restatements of the headline figures and caveats as they travel between sections, while retaining every experiment, control, and reviewer-requested caveat (each now appears once in its natural home).

*Location:* Section 6.

### 5. Frame the resource as noisy synthetic supervision

**Requested.** Clearly describe the resource as noisy synthetic supervision rather than a gold benchmark.

Done. The abstract, Section 6.2, and conclusion state that this is a controlled synthetic-supervision resource whose label faithfulness is explicitly measured (0.42 strict per-aspect lower bound, 0.58 per-row) and controlled by the audit-and-filter pipeline, with a documented benchmark setting, rather than a gold-labeled corpus. Quantifying and filtering label noise is a central contribution, not a caveat on it.

*Location:* Abstract; Section 6.2; Section 7.

### 6. Broaden the impact discussion

**Requested.** Address instructor evaluation, student-comment privacy, bias against non-native or unusual writing styles, and the risk of attaching fictional negative reviews to identifiable courses or instructors; require human review, uncertainty reporting, data protection, and an appeals process.

Done. The ethics statement now adds stylistic-bias monitoring for non-native and non-standard writing, a prohibition on attaching model-inferred negatives to identifiable courses or instructors, and per-aspect uncertainty reporting with low-confidence routing to human review and an appeals process, on top of the existing no-identifiable-data, licensing, re-consent, and high-stakes provisions.

*Location:* Section 6.3.

### 7. Rule out generator-specific split effects

**Requested.** Random splits from the same generator and prompt may reward generator-specific patterns.

Clarified, with a new control. The learnable signal is not a single generator's artifact: regenerating and auditing with three independent model families reproduces the label fidelity, and a held-out-generator transfer check (Appendix A.31) shows a detector trained on one generator's data transfers to other generators' data without collapse. An overlap-generalization analysis further separates aspect-composition effects from domain shift.

*Location:* Appendix A.19, A.24, A.31.

---

Locations reference the revised manuscript; every requested change is in place at the cited location.