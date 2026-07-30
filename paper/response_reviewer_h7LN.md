# Response to Reviewer h7LN

*A Controlled Synthetic Benchmark for Educational Aspect-Based Sentiment Analysis (TMLR)*

We thank Reviewer h7LN for a thorough and technically precise report, and for the concrete pointers on baselines, figures, and the truncated rows, which we acted on directly. Each point is answered below with the specific evidence and its location.

### 1. Statistical realism and full-schema re-annotation

**Requested.** LLM-as-judge realism does not statistically establish indistinguishability from real feedback; the full dimensional label set lacks comprehensive human re-annotation.

Reframed, with analysis and a new human study. The realism that matters for a labeled training benchmark is functional: a model trained on the synthetic corpus recovers real aspect and sentiment signal on independent external corpora. Judged sentence by sentence, synthetic sentences are near-indistinguishable from real ones (the judge is only 60% accurate, near the 50% floor, versus 93% on whole reviews), and a sentence-level distributional check places the synthetic corpus close to real on the units that carry the supervision. On the label side, three annotators re-annotated a stratified sample of the synthetic corpus (Table 14, shared with point-N1): inter-annotator reliability is substantial (Fleiss kappa 0.70), human confirmation of declared aspects rises monotonically with the audit score, and the human labels reproduce the audit's presence-faithful, sentiment-noisier split, so the label validation now rests on direct human annotation of the synthetic text itself.

*Location:* Section 5.7 (Table 14); Section 6.1; Appendix A.23.

### 2. Broaden baselines beyond one provider and prompt

**Requested.** Baselines are confined to a single provider's GPT family and one structured prompting method.

Done. We broaden the baseline on both axes. The prompting axis is covered by the four zero-, fixed-, diverse-, and retrieval-few-shot variants in Table 7. The provider axis is now covered on the full 1,000-review test split: the identical zero-shot-glossary contract run across four families spanning four providers (GPT-5.4, Gemini-2.5-Flash, GLM-4.6, Llama-3.3-70B) lands in a single narrow band (detection micro-F1 0.239 to 0.267), all below the trained encoders, so the placement of zero-shot prompting holds regardless of provider.

*Location:* Appendix A.19; Appendix A.20.

### 3. Figure and table formatting

**Requested.** The formatting of Figure 1, Table 5, Figure A2, and Figure A3 could benefit from refinement.

Done. Figure 1 is tightened to fill its frame with even margins, and the bar-chart axis and numeric labels now use a clean sans-serif (the y-axis label reads Micro-F1 correctly). Table 5 was checked for readability.

*Location:* Figures 1, A2, A3.

### 4. The 841 token-capped rows

**Requested.** 841 of 10,000 samples hit the token cap; full-corpus length-band adherence is 0.6819, so a substantial share falls outside the length bounds.

Done. Regenerating exactly those rows at a higher cap eliminates the truncation and restores full-corpus length adherence. We further show the truncated rows are not systematically biased: their aspect distribution is not skewed (chi-square p=0.23, Cramer's V=0.034, Appendix A.30), their polarity distribution is negligibly different (V=0.020), and their audit faithfulness is statistically identical to the complete corpus (0.573 versus 0.577, p=0.76). They differ only in the mechanical ways (fewer aspects, shorter text), so excluding them does not change the benchmark.

*Location:* Appendix A.14, A.30.

### 5. Bottom-quartile rows and training value

**Requested.** The bottom 25% instances markedly inflate sentiment error (Table 12) and hold little training value.

Clarified. This is the signal the filtering recipe exploits: the negative-control bottom quartile collapses on sentiment error across both architectures and both transfer targets (0 of 8 seeds recovering), which is why retaining the top 50% by audit score reduces error at half the training cost. A quartile dose-response confirms the effect is monotone across the whole distribution (Appendix A.32), so the audit score discriminates informative from uninformative rows at both ends.

*Location:* Section 5.7.

### 6. Audit-human agreement (kappa 0.56)

**Requested.** Cohen's kappa 0.56 is moderate, so the audit cannot be a fully reliable proxy for human judgement.

Clarified. The 0.42 match is a strict per-aspect lower bound (per-row mean 0.58) and the audit's value is demonstrated downstream: filtering by it reduces sentiment error at half the data, and it agrees with humans at kappa 0.56 and 0.62 across independent families. It is a validated selection instrument rather than a ground-truth oracle; the synthetic-corpus annotation study (point-N1) measures its agreement on synthetic text directly.

*Location:* Section 5.7; Section 6.2.

### 7. Scope of the external validation

**Requested.** The OMSCS reviews number only 32 and the Herath corpus 2,829, so the conclusions apply to a narrow scope.

Clarified. External validation spans four independent real corpora (Herath, EduRABSA, M-ABSA, OATS) across institutional and MOOC settings. We state the current scope (English STEM and graduate) and identify cross-domain and cross-language extension as future work.

*Location:* Appendix A.24; Section 6.1.

---

Locations reference the revised manuscript; every requested change is in place at the cited location.