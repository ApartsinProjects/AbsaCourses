# Summary of revisions

We thank the reviewers; the revision strengthens the evidence and reconciles the reporting.

- **Reporting reconciled.** All synthetic-to-real transfer numbers now derive from one multi-seed run (Tables 8, 9), the transfer figures were regenerated to match, aspect counts sum to 10,000, and the abstract is complete.
- **Direct human validation of the audit** (new Table 14): three annotators re-labeled a stratified synthetic sample (Fleiss kappa 0.70); human confirmation of declared aspects rises monotonically with the audit score, validating the audit on synthetic text (N1, H1).
- **Covariate-matched filtering** (new Table 13): matching on aspect, polarity, count, length, and style and scoring on a common gold set attributes the filtering gain to faithfulness alone (N2).
- **New robustness experiments:** independent open-weights auditors reproduce the audit (A.29), the token-capped rows are not biased (A.30), the learnable signal is generator-invariant (A.31), an audit-quartile dose-response is monotone (A.32), and a qualitative error analysis (A.33).
- **Framing sharpened:** measured-and-controlled noisy synthetic supervision (not a gold benchmark); broader-impact, generator-auditor circularity, transfer limits, and a practitioner roadmap made explicit; baselines span four providers; external checks span four real corpora.
- **Artifacts released:** corpus, mapped real data, human-annotation study, code, and best-per-target checkpoints deposited on Zenodo.

---

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

---

# Response to Reviewer h7LN

*A Controlled Synthetic Benchmark for Educational Aspect-Based Sentiment Analysis (TMLR)*

We thank Reviewer h7LN for a thorough and technically precise report, and for the concrete pointers on baselines, figures, and the truncated rows, which we acted on directly. Each point is answered with its evidence and location.

### 1. Statistical realism and full-schema re-annotation

**Requested.** LLM-as-judge realism does not statistically establish indistinguishability from real feedback; the full label set lacks comprehensive human re-annotation.

Reframed, with analysis and a new human study. The realism that matters for a labeled training benchmark is functional: a model trained on the synthetic corpus recovers real aspect and sentiment signal on independent external corpora. Judged sentence by sentence, synthetic sentences are near-indistinguishable from real ones (the judge is 60% accurate, near the 50% floor, versus 93% on whole reviews). On the label side, three annotators re-annotated a stratified sample (Table 15): inter-annotator reliability is substantial (Fleiss 0.70), human confirmation rises monotonically with the audit score, and the human labels reproduce the audit's presence-faithful, sentiment-noisier split, so label validation now rests on direct human annotation of the synthetic text.

*Location:* Section 5.7 (Table 15); Section 6.1; Appendix A.2.7.

### 2. Broaden baselines beyond one provider and prompt

**Requested.** Baselines are confined to a single provider's GPT family and one prompting method.

Done. The prompting axis is covered by four zero-, fixed-, diverse-, and retrieval-few-shot variants (Table 7). The provider axis is now covered on the full 1,000-review test split: the identical zero-shot contract run across four families spanning four providers (GPT-5.4, Gemini-2.5-Flash, GLM-4.6, Llama-3.3-70B) lands in a single narrow band (micro-F1 0.239 to 0.267), all below the trained encoders.

*Location:* Appendix A.3.2; Appendix A.6.2.

### 3. Figure and table formatting

**Requested.** The formatting of Figure 1, Table 5, Figure A2, and Figure A3 could benefit from refinement.

Done, and we thank the reviewer for the specific pointers. Figure 1 is tightened with even margins and a clean sans-serif axis, Table 5 was reset for consistent alignment, and the two appendix panels flagged (Figures A2 and A3 in the reviewed version, now Figures A3 and A6 after the appendix reorganization) were regenerated with matched fonts and spacing.

*Location:* Figure 1; Table 5; Figures A3 and A6.

### 4. The 841 token-capped rows

**Requested.** 841 of 10,000 samples hit the token cap; full-corpus length-band adherence is 0.6819.

Done. Regenerating those rows at a higher cap restores full-corpus length adherence. The truncated rows are not systematically biased: aspect distribution not skewed (chi-square p=0.23, V=0.034), polarity negligibly different (V=0.020), and audit faithfulness statistically identical (0.573 versus 0.577, p=0.76). They differ only mechanically, so excluding them does not change the benchmark.

*Location:* Appendix A.1.5, A.1.6.

### 5. Bottom-quartile rows and training value

**Requested.** The bottom 25% markedly inflate sentiment error and hold little training value.

Clarified. This is exactly the signal the filter exploits: the negative-control bottom quartile collapses on sentiment error across both architectures and both transfer targets (0 of 8 seeds recovering), which is why retaining the top 50% by audit score reduces error at half the training cost. A quartile dose-response confirms the effect is monotone across the whole distribution.

*Location:* Section 5.7.

### 6. Audit-human agreement (kappa 0.56)

**Requested.** Cohen's kappa 0.56 is moderate, so the audit cannot be a fully reliable proxy for human judgement.

Clarified. The 0.42 match is a strict per-aspect lower bound (per-row mean 0.58), and the audit's value is downstream: filtering by it reduces sentiment error at half the data, and it agrees with humans at kappa 0.56 and 0.62 across independent families. It is a validated selection instrument, not a ground-truth oracle; the synthetic-corpus annotation study measures its agreement on synthetic text directly.

*Location:* Section 5.7; Section 6.2.

### 7. Scope of the external validation

**Requested.** The OMSCS reviews number only 32 and Herath 2,829, so conclusions apply to a narrow scope.

Clarified. External validation now spans four real corpora (Herath, EduRABSA, M-ABSA, OATS) across institutional and MOOC settings. We state the current scope (English STEM and graduate) and identify cross-domain and cross-language extension as future work.

*Location:* Appendix A.5.4; Section 6.1.

---

Locations reference the revised manuscript; every requested change is in place at the cited location.

---

# Response to Reviewer dWED

*A Controlled Synthetic Benchmark for Educational Aspect-Based Sentiment Analysis (TMLR)*

We thank Reviewer dWED for a generous and constructive review. We are pleased the experimental design, filtering pipeline, and transfer evidence came through clearly, and we have strengthened each noted point below.

### 1. Generator-auditor circularity

**Requested.** The same provider family generates the data and performs the audit; a dedicated discussion of circularity is needed.

Done, with a new experiment. We re-ran the audit with two open-weights auditors from independent families (Llama-3.3-70B and GLM-4.6). All three families converge on the same per-aspect judgments (support-rate 0.77, 0.74, 0.73; cross-architecture Cohen's kappa 0.56 to 0.65), and the two open-weights auditors agree with each other at the same level, so the GPT auditor is not privileged. A same-family artifact would make out-of-family auditors diverge; instead they reproduce the audit, which shows it measures textual faithfulness.

*Location:* Section 6.1; Appendix A.3.2, A.4.4.

### 2. Bias analysis of the incomplete rows

**Requested.** Analyze whether the 841 incomplete rows are biased across aspects or polarities, and whether excluding them changes results.

Done. The truncated rows are not systematically biased across aspects (Cramer's V=0.034) or polarities (V=0.020), and their audit faithfulness matches the complete corpus (0.573 versus 0.577, p=0.76); the only differences are mechanical (fewer aspects, shorter text), so excluding them leaves the benchmark unchanged.

*Location:* Appendix A.1.5, A.1.6.

### 3. Broader related-work discussion

**Requested.** Discuss emerging methods, including multimodal sarcasm perception in vision-language models and set-matching for generalized category discovery.

Done, and we thank the reviewer for both pointers. The related-work section now cites probing how vision-language models perceive multimodal sarcasm [5] and exploiting relational structure among unlabelled samples for generalized category discovery [24], and states that our data-quality-control contribution is orthogonal to these modeling advances. Both are cited in the running text and the bibliography.

*Location:* Section 2; References [5] and [24].

### 4. Strengthen the transfer-limits statement

**Requested.** State prominently what practitioners should not conclude: the full 20-aspect schema lacks real validation and high-stakes decisions require human-in-the-loop review.

Done. Section 6.1 states that only 9 of 20 aspects are externally validated, that synthetic-only training recovers about half of a real-trained model (micro-F1 0.402 versus 0.767 across five seeds), that the full schema is not yet externally validated, and that high-stakes use requires human-in-the-loop review.

*Location:* Section 6.1.

### 5. Practitioner adoption roadmap

**Requested.** Provide concrete guidance: minimum fine-tuning data size, expected degradation, and monitoring.

Done. Section 6.2 gives a fine-tuning-size curve: roughly 250 to 500 local reviews capture most of the benefit, synthetic pretraining stays above from-scratch training at every budget and roughly doubles data efficiency in the low-data regime (Figure 6, five seeds), practitioners should expect the Figure 6 curve rather than internal-benchmark numbers, and deployments should be monitored against a held-out locally-adjudicated slice with re-checks on distribution shift.

*Location:* Section 6.2; Figure 6.

### 6. Moderate scores and qualitative error analysis

**Requested.** Add a qualitative error analysis of common failure modes.

Done. The moderate scores reflect the intrinsic difficulty of 20-aspect ABSA under conservative overlap. A new qualitative error analysis shows four systematic patterns: high-prevalence diffuse aspects are over-predicted while specific aspects are under-detected; missed specific aspects are substituted by generic ones; polarity compresses toward neutral; and a positive skew appears under real-review transfer. Practitioners can expect reliable detection and polarity on frequent, lexically distinctive aspects, and should treat fine-grained aspects and non-positive polarities on out-of-domain reviews as the weak regime.

*Location:* Section 6.1; Appendix A.7.1.

### 7. Broader impact

**Requested.** No broader-impact concerns were raised.

Thank you for the positive assessment. Prompted by the other reviews, we extended the ethics statement with stylistic-bias monitoring for non-native writing, a prohibition on attaching model-inferred negatives to identifiable courses or instructors, and per-aspect uncertainty with low-confidence routing to human review and an appeals process.

*Location:* Section 6.3.

---

Locations reference the revised manuscript; every requested change is in place at the cited location.

---
