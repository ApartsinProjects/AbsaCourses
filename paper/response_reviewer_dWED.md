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