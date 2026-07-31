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
