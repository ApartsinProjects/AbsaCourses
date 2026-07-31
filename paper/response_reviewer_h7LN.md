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
