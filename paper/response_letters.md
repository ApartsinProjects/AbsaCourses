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

We are grateful to Reviewer nfat for a careful and constructive report, and especially for pressing on the validation of the audit and the internal consistency of the reporting. Both have made the paper stronger. We answer each point below with the specific change and its location in the revised manuscript.

### 1. Human validation of the audit on the synthetic corpus

**Requested.** Human-annotate a representative sample of the actual synthetic corpus; the auditor is validated mainly on perturbed real labels rather than direct human annotation of synthetic reviews.

Done (new Table 15). Three annotators independently labeled a stratified sample of 300 synthetic reviews (610 declared review-aspect decisions), blind to the declared labels, marking per-aspect presence and sentiment. Inter-annotator reliability is substantial for both label types: Fleiss kappa 0.70 on aspect presence, with pairwise Cohen kappa 0.60 to 0.87 spanning presence and sentiment, and Table 15 now reports the sentiment inter-annotator agreement alongside the presence agreement. The annotation validates the audit directly on synthetic text three ways: human confirmation of the declared aspect rises monotonically with the audit score (55% in the lowest audit-score quartile to 79% in the highest); human and audit presence judgments agree 76% of the time; and the human annotation independently reproduces the audit's central finding that aspect presence is the faithful signal (70% human-confirmed) while aspect sentiment is the noisier one (human-declared agreement about 0.40, closely matching the audit's strict 0.42). The audit score is therefore a valid, human-grounded quality signal on the synthetic corpus, and the corpus's measured noise level is accurate.

*Location:* Section 5.7 (Table 15); Section 6.1.

### 2. Match filtering subsets to isolate faithfulness

**Requested.** Sentiment MSE is measured only on aspects each model predicts, so filtering comparisons may be confounded; match subsets by aspect, polarity, aspect count, length, and style.

Done (new Table 14). We add a covariate-matched filtering comparison that matches the retained and control subsets one-to-one on aspect set, aspect count, polarity composition, length band, and formality band, so the two subsets are identical on every covariate except the audit score itself (matched-pair audit-score means 0.90 versus 0.26; 3,441 pairs). Both are then trained and scored on the same common gold-present aspect cells on the 9-aspect Herath overlap (4,289 shared cells), which removes the prediction-mask confound entirely. Under this strict design the faithfulness-retained subset has the lower transferred sentiment error in every seed (sentiment MSE 0.412 versus 0.519, a paired reduction of 0.108 across three seeds) and a higher detection micro-F1 (0.400 versus 0.338), so the filtering gain is attributable to label faithfulness alone rather than to differing composition or prediction masks. This complements the size-matched result already reported (retaining the top 50% cuts sentiment error at half the training cost, 7 of 8 seeds, replicated across architectures).

*Location:* Section 5.7 (Table 14).

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

Done, and we thank the reviewer for this constructive list; every item now has a corresponding provision. On instructor evaluation, the ethics statement states that outputs are intended to inform formative course improvement and must not drive high-stakes instructor assessment without human oversight. On student-comment privacy and data protection, it retains the no-identifiable-data, licensing, and re-consent provisions and adds explicit data-protection and access-control language for any deployment on real comments. On bias, it adds stylistic-bias monitoring for non-native and non-standard writing. On the fictional-negative-review risk, it adds a prohibition on attaching model-inferred negatives to identifiable courses or instructors. And it adds per-aspect uncertainty reporting with low-confidence routing to human review and an appeals process.

*Location:* Section 6.3.

### 7. Rule out generator-specific split effects

**Requested.** Random splits from the same generator and prompt may reward generator-specific patterns.

Clarified, with a new control. The learnable signal is not a single generator's artifact: regenerating and auditing with three independent model families reproduces the label fidelity, and a held-out-generator transfer check (Appendix A.3.3) shows a detector trained on one generator's data transfers to other generators' data without collapse. An overlap-generalization analysis further separates aspect-composition effects from domain shift.

*Location:* Appendix A.3.2, A.5.4, A.3.3.

---

Locations reference the revised manuscript; every requested change is in place at the cited location.

---

# Response to Reviewer h7LN

*A Controlled Synthetic Benchmark for Educational Aspect-Based Sentiment Analysis (TMLR)*

We thank Reviewer h7LN for a thorough and technically precise report, and for the concrete pointers on baselines, figures, and the truncated rows, which we acted on directly. Each point is answered below with the specific evidence and its location.

### 1. Statistical realism and full-schema re-annotation

**Requested.** LLM-as-judge realism does not statistically establish indistinguishability from real feedback; the full dimensional label set lacks comprehensive human re-annotation.

Reframed, with analysis and a new human study. The realism that matters for a labeled training benchmark is functional: a model trained on the synthetic corpus recovers real aspect and sentiment signal on independent external corpora. Judged sentence by sentence, synthetic sentences are near-indistinguishable from real ones (the judge is only 60% accurate, near the 50% floor, versus 93% on whole reviews), and a sentence-level distributional check places the synthetic corpus close to real on the units that carry the supervision. On the label side, three annotators re-annotated a stratified sample of the synthetic corpus (Table 15, shared with point-N1): inter-annotator reliability is substantial (Fleiss kappa 0.70), human confirmation of declared aspects rises monotonically with the audit score, and the human labels reproduce the audit's presence-faithful, sentiment-noisier split, so the label validation now rests on direct human annotation of the synthetic text itself.

*Location:* Section 5.7 (Table 15); Section 6.1; Appendix A.2.7.

### 2. Broaden baselines beyond one provider and prompt

**Requested.** Baselines are confined to a single provider's GPT family and one structured prompting method.

Done. We broaden the baseline on both axes. The prompting axis is covered by the four zero-, fixed-, diverse-, and retrieval-few-shot variants in Table 7. The provider axis is now covered on the full 1,000-review test split: the identical zero-shot-glossary contract run across four families spanning four providers (GPT-5.4, Gemini-2.5-Flash, GLM-4.6, Llama-3.3-70B) lands in a single narrow band (detection micro-F1 0.239 to 0.267), all below the trained encoders, so the placement of zero-shot prompting holds regardless of provider.

*Location:* Appendix A.3.2; Appendix A.6.2.

### 3. Figure and table formatting

**Requested.** The formatting of Figure 1, Table 5, Figure A2, and Figure A3 could benefit from refinement.

Done, and we thank the reviewer for the specific pointers. Figure 1 is tightened to fill its frame with even margins, and its bar-chart axis and numeric labels now use a clean sans-serif (the y-axis label reads Micro-F1 correctly). Table 5 was reset for consistent column alignment and readability, and the two appendix panels the reviewer flagged (Figures A2 and A3 in the reviewed version, now Figures A3 and A6 after the appendix reorganization) were regenerated with matched fonts, axis labels, and spacing so they are legible at print size.

*Location:* Figure 1; Table 5; Figures A3 and A6.

### 4. The 841 token-capped rows

**Requested.** 841 of 10,000 samples hit the token cap; full-corpus length-band adherence is 0.6819, so a substantial share falls outside the length bounds.

Done. Regenerating exactly those rows at a higher cap eliminates the truncation and restores full-corpus length adherence. We further show the truncated rows are not systematically biased: their aspect distribution is not skewed (chi-square p=0.23, Cramer's V=0.034, Appendix A.1.6), their polarity distribution is negligibly different (V=0.020), and their audit faithfulness is statistically identical to the complete corpus (0.573 versus 0.577, p=0.76). They differ only in the mechanical ways (fewer aspects, shorter text), so excluding them does not change the benchmark.

*Location:* Appendix A.1.5, A.1.6.

### 5. Bottom-quartile rows and training value

**Requested.** The bottom 25% instances markedly inflate sentiment error (Table 12) and hold little training value.

Clarified. This is the signal the filtering recipe exploits: the negative-control bottom quartile collapses on sentiment error across both architectures and both transfer targets (0 of 8 seeds recovering), which is why retaining the top 50% by audit score reduces error at half the training cost. A quartile dose-response confirms the effect is monotone across the whole distribution (Appendix A.4.5), so the audit score discriminates informative from uninformative rows at both ends.

*Location:* Section 5.7.

### 6. Audit-human agreement (kappa 0.56)

**Requested.** Cohen's kappa 0.56 is moderate, so the audit cannot be a fully reliable proxy for human judgement.

Clarified. The 0.42 match is a strict per-aspect lower bound (per-row mean 0.58) and the audit's value is demonstrated downstream: filtering by it reduces sentiment error at half the data, and it agrees with humans at kappa 0.56 and 0.62 across independent families. It is a validated selection instrument rather than a ground-truth oracle; the synthetic-corpus annotation study (point-N1) measures its agreement on synthetic text directly.

*Location:* Section 5.7; Section 6.2.

### 7. Scope of the external validation

**Requested.** The OMSCS reviews number only 32 and the Herath corpus 2,829, so the conclusions apply to a narrow scope.

Clarified. External validation spans four independent real corpora (Herath, EduRABSA, M-ABSA, OATS) across institutional and MOOC settings. We state the current scope (English STEM and graduate) and identify cross-domain and cross-language extension as future work.

*Location:* Appendix A.5.4; Section 6.1.

---

Locations reference the revised manuscript; every requested change is in place at the cited location.

---

# Response to Reviewer dWED

*A Controlled Synthetic Benchmark for Educational Aspect-Based Sentiment Analysis (TMLR)*

We thank Reviewer dWED for a careful and generous review and for constructive, actionable suggestions. We are pleased the experimental design, filtering pipeline, and transfer evidence came through clearly, and we have strengthened each of the noted points as detailed below.

### 1. Generator-auditor circularity

**Requested.** The same provider family (OpenAI) generates the data and performs the audit; a dedicated discussion of potential circularity is needed.

Done, with a new experiment. Beyond the standalone discussion, we settle the concern empirically by re-running the audit with two open-weights auditors from independent families (Llama-3.3-70B from Meta and GLM-4.6 from Zhipu) on the same sample. All three families converge on the same per-aspect judgments (support-rate 0.77, 0.74, 0.73; cross-architecture Cohen's kappa 0.56 to 0.65, row-score Spearman 0.54 to 0.69, Appendix A.4.4), and the two open-weights auditors agree with each other at the same level, so the GPT auditor is not privileged. A same-family artifact would make out-of-family auditors diverge; instead they reproduce the audit, which shows it measures textual faithfulness.

*Location:* Section 6.1; Appendix A.3.2, A.4.4.

### 2. Bias analysis of the incomplete rows

**Requested.** Analyze whether the 841 incomplete rows are systematically biased across aspects or sentiment polarities, and whether excluding them changes benchmark results.

Done. The truncated rows are not systematically biased across aspects (Cramer's V=0.034) or polarities (V=0.020), and their audit faithfulness matches the complete corpus (0.573 versus 0.577, p=0.76); the only differences are mechanical (fewer aspects, shorter text). Their label and faithfulness profile is representative, so excluding them leaves the benchmark unchanged.

*Location:* Appendix A.1.5.

### 3. Broader related-work discussion

**Requested.** Discuss emerging methods, including multimodal sarcasm perception in vision-language models and set-matching for generalized category discovery.

Done, and we thank the reviewer for both pointers. The related-work section now places our contribution against these advances: it notes probing how vision-language models perceive nuanced signals such as multimodal sarcasm [5] and exploiting relational structure among unlabelled samples for generalized category discovery [24], and states that our data-quality-control contribution is orthogonal to these modeling advances. Both works are cited in the running text and listed in the bibliography.

*Location:* Section 2 (related work); References [5] and [24].

### 4. Strengthen the transfer-limits statement

**Requested.** State prominently what practitioners should not conclude: the full 20-aspect schema lacks real validation and high-stakes decisions require human-in-the-loop review.

Done. Section 6.1 states that only 9 of 20 aspects are externally validated, that synthetic-only training recovers about half of a real-trained model (micro-F1 0.402 versus 0.767 across five seeds), that the full schema is not yet externally validated, and that high-stakes use requires human-in-the-loop review.

*Location:* Section 6.1.

### 5. Practitioner adoption roadmap

**Requested.** Provide concrete guidance: minimum fine-tuning data size, expected performance degradation, and monitoring requirements.

Done. Section 6.2 provides a fine-tuning-size curve: roughly 250 to 500 local reviews capture most of the benefit, the synthetic pretrain reaches real-only quality with about half the real data, practitioners should expect the Figure 6 curve rather than internal-benchmark numbers, and deployments should be monitored against a held-out locally-adjudicated slice with re-checks on distribution shift.

*Location:* Section 6.2; Figure 6.

### 6. Moderate scores and qualitative error analysis

**Requested.** Moderate absolute performance leaves unclear what good-enough means; a qualitative error analysis of common failure modes would help practitioners diagnose systematic errors.

Done. Absolute scores reflect the intrinsic difficulty of 20-aspect ABSA under conservative overlap. We add a qualitative error analysis showing the failures are systematic in four recurring patterns: high-prevalence diffuse aspects are over-predicted while specific aspects are under-detected; missed specific aspects are substituted by generic evaluative ones; polarity compresses toward neutral on detected aspects; and a positive skew appears under real-review transfer. Practitioners can therefore expect reliable detection and polarity on frequent, lexically distinctive aspects, and should treat fine-grained aspects and non-positive polarities on out-of-domain reviews as the weak regime (Appendix A.7.1).

*Location:* Section 5; Section 6.1.

### 7. Broader impact

**Requested.** No broader-impact concerns were raised.

Thank you for the positive assessment of the broader-impact discussion. We have kept the ethics statement thorough and, prompted by the other reviews, extended it with stylistic-bias monitoring for non-native and non-standard writing, a prohibition on attaching model-inferred negatives to identifiable courses or instructors, and per-aspect uncertainty reporting with low-confidence routing to human review and an appeals process, on top of the existing no-identifiable-data, licensing, re-consent, and high-stakes-use provisions.

*Location:* Section 6.3.

---

Locations reference the revised manuscript; every requested change is in place at the cited location.

---
