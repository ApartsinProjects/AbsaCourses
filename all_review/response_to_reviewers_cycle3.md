# Response to reviewers (nfat, h7LN, dWED) — working draft

Tags: **[DONE]** in the revision · **[ADDING]** strengthening experiment in progress · **[CLARIFY]** already present, made prominent.
Locations reference the revised `course_absa_manuscript.html`.

We are grateful to all three reviewers for their careful and constructive reading, and we are glad that each of them sees the work as of interest to the TMLR audience. Their comments have helped us sharpen the framing, reconcile the reporting, and identify several experiments that further strengthen the contribution; we detail each below and are happy to expand on any point.

---

## Reviewer nfat

We thank Reviewer nfat for the detailed and constructive review, and in particular for pushing on the validation of the audit and the internal consistency of the reporting; both have made the paper stronger.


**N1. Validate the audit with direct human annotation of the synthetic corpus.**
The audit is already validated three independent ways: a behavioral perturbation control that the audit passes by construction, cross-provider judge agreement, and direct agreement with human labels on two annotated corpora (Cohen's kappa 0.56, and 0.62 with a fully independent model family). To extend this evidence to the synthetic text itself, we [ADDING] a human annotation of a stratified sample of the actual synthetic corpus, reporting aspect and sentiment agreement against both the declared labels and the audit, which measures the audit's agreement on synthetic text directly. *Location:* §5.7, §6.1 (revision in progress).

**N2. Isolate the filtering effect with covariate-matched subsets and a common aspect set.**
We [ADDED] a covariate-matched filtering comparison that holds aspect, polarity, aspect-count, length, and style fixed across the retained and control subsets and scores both on a common gold-present aspect set, so the filtering gain is attributable to faithfulness alone rather than to differing prediction masks. The size-matched filtering result already reported (top-50% cuts sentiment MSE at half the training cost, 7 of 8 seeds, replicated across architectures) motivates this stricter matched design. *Location:* §5.7 (revision in progress).

**N3. Reconcile inconsistent numbers (Tables 8/9, aspect-count totals, abstract).**
[DONE] Aspect-count totals sum to 10,000 (3,032 + 3,917 + 3,051) and the abstract is complete. Tables 8 and 9 now derive from a single consistent transfer run, so every transfer score and the generalization gap trace to the same models. *Location:* §5.1, §5.4, abstract.

**N4. Tighten repeated discussion.**
[DONE] The discussion is streamlined while retaining the control experiments the other reviewers found decisive. *Location:* §6.

**N5. Frame the resource as measured-noise synthetic supervision rather than a gold benchmark.**
[DONE] We make the framing precise: this is a controlled synthetic-supervision resource whose label faithfulness is explicitly *measured* (0.42 strict per-aspect lower bound, 0.58 per-row) and *controlled* by the audit-and-filter pipeline, with a documented benchmark setting. Quantifying and filtering label noise is a central contribution of the paper, not a caveat on it; the resource is a controlled instrument that institutions fine-tune and validate on their own data. *Location:* §6.2, §7, abstract.

**N6. Broaden the impact discussion.**
[DONE] The ethics statement adds stylistic-bias monitoring for non-native and non-standard writing, a prohibition on attaching model-inferred negatives to identifiable courses or instructors, and per-aspect uncertainty reporting with low-confidence routing to human review and an appeals process, on top of the existing no-identifiable-data, licensing, and high-stakes provisions. *Location:* §6.3.

**N7. Rule out that random splits reward generator-specific patterns.**
[CLARIFY] The learnable signal is not a single generator's artifact: regenerating and auditing with three independent model families (Gemini, GLM, Llama) reproduces the label fidelity, and an overlap-generalization analysis separates aspect-composition effects from domain shift. We surface this argument in the transfer discussion. *Location:* Appendix A.19, A.24.

---

## Reviewer h7LN

We thank Reviewer h7LN for the thorough and precise review, and for the concrete pointers on baselines, figures, and the incomplete rows, which we have acted on directly.

**H1. Establish realism beyond LLM self-judging; re-annotate the full label schema.**
Two results settle the realism question in the form that matters for a labeled training benchmark. First, functional realism: a model trained on the synthetic corpus recovers real aspect and sentiment signal on independent external corpora, which is the operative property for supervision. Second, sentence-level realism: judged sentence by sentence, synthetic sentences are near-indistinguishable from real ones (judge accuracy near chance), and a sentence-level distributional check places the synthetic corpus close to real on the units that carry the aspect-sentiment supervision. The human annotation added for N1 extends the label validation to the full synthetic sample. *Location:* §6.1, Appendix A.23.

**H2. Extend baselines beyond one provider and one prompt.**
[DONE] The multi-provider zero-shot baseline spans four generator families across four providers (GPT, Gemini, GLM, Llama). *Location:* Appendix A.19, A.20.

**H3. Refine Figure 1, Table 5, Figure A2, Figure A3.**
[DONE] Figure 1 is tightened to fill its frame, the bar-chart axis and number font is a clean sans-serif, and Table 5 is verified for readability. *Location:* Figures 1, A2, A3.

**H4. Address the 841 token-capped rows.**
[DONE] Regenerating exactly those rows at a higher cap eliminates the truncation and restores full-corpus length adherence. We further analyzed the truncated rows for bias: their aspect distribution is not skewed (chi-square p=0.23, Cramer's V=0.034) and their polarity distribution is negligibly different (Cramer's V=0.020); they differ only in the mechanically-expected ways (fewer labeled aspects, shorter text). Critically, their audit faithfulness is statistically indistinguishable from the complete corpus (mean audit score 0.573 vs 0.577, p=0.76), so excluding them does not change the benchmark's faithfulness or label profile. *Location:* Appendix A.14.

**H5. The bottom-25% low-fidelity rows inflate sentiment error.**
[CLARIFY] Precisely: that is the signal the filtering recipe exploits. The negative-control bottom quartile collapses on sentiment MSE across both architectures and both transfer targets (0 of 8 seeds recovering), which is why retaining the top 50% by audit score reduces error at half the training cost. The audit score discriminates informative from uninformative rows at both ends of the distribution. *Location:* §5.7.

**H6. 0.42 match and kappa 0.56 mean the audit is an imperfect proxy.**
[CLARIFY] The 0.42 is a strict per-aspect lower bound (per-row mean 0.58), and the audit's value is demonstrated downstream: filtering by it reduces sentiment error at half the data, and it agrees with humans at kappa 0.56 and 0.62 across independent families. It is a validated and useful selection instrument; the human annotation added for N1 measures its agreement on synthetic text directly. *Location:* §5.7, §6.2.

**H7. Scope beyond 32 OMSCS and 2,829 Herath reviews.**
[CLARIFY] External validation spans four independent real corpora (Herath, EduRABSA, M-ABSA, OATS) across institutional and MOOC settings. We state the current scope (English STEM/graduate) and identify cross-domain and cross-language extension as future work. *Location:* Appendix A.24, §6.1.

---

## Reviewer dWED

We thank Reviewer dWED for the careful and generous review and for the constructive, actionable suggestions; we are pleased the experimental design, filtering pipeline, and transfer evidence came through clearly.

**D1. Discuss generator-auditor circularity directly.**
[DONE] §6.1 gives a standalone treatment, and we settle the concern empirically: we re-ran the faithfulness audit with two open-weights auditors from independent families (Llama-3.3-70B, Meta; GLM-4.6, Zhipu) on the same 250-review sample. All three families converge on the same per-aspect judgments (support-rate GPT 0.77 / Llama 0.74 / GLM 0.73; cross-architecture Cohen's kappa 0.56 to 0.65, row-score Spearman 0.54 to 0.69), and the two open-weights auditors agree with each other at the same level, so the GPT auditor is not privileged. A same-family artifact would make out-of-family auditors diverge; instead they reproduce the audit, which shows it measures textual faithfulness rather than a generator-auditor family effect. This complements the earlier Gemini human-agreement check (kappa 0.62) and the behavioral perturbation control. *Location:* §6.1, Appendix A.19.

**D2. Analyze whether the 841 incomplete rows are biased and whether excluding them changes results.**
[DONE] The truncated rows are not systematically biased across aspects (Cramer's V=0.034) or polarities (V=0.020), and their audit faithfulness matches the complete corpus (0.573 vs 0.577, p=0.76); the only differences are the mechanical ones (fewer aspects, shorter text). Their label and faithfulness profile is representative, so excluding them leaves the benchmark unchanged (shared with H4). *Location:* Appendix A.14.

**D3. Cite emerging methods (multimodal-sarcasm VLMs; set-matching GCD).**
[DONE] Both are cited. *Location:* References.

**D4. Strengthen the transfer-limits statement.**
[DONE] §6.1 states that 9 of 20 aspects are externally validated, that synthetic-only training recovers about 60% of a real-trained model, that the full schema is not yet externally validated, and that high-stakes use requires human-in-the-loop review. *Location:* §6.1.

**D5. Give a practitioner roadmap.**
[DONE] §6.2 provides a fine-tuning-size curve: roughly 250-500 local reviews capture most of the benefit, the synthetic pretrain reaches real-only quality with about half the real data, practitioners should expect the Figure 6 curve rather than internal-benchmark numbers, and should monitor against a held-out locally-adjudicated slice with re-checks on distribution shift. *Location:* §6.2, Figure 6.

**D-additional (moderate scores / error analysis / generalizability).**
[DONE] Absolute scores reflect the intrinsic difficulty of 20-aspect ABSA under conservative overlap. We add a qualitative error analysis showing the failures are systematic, not random, in four recurring patterns: (1) high-prevalence diffuse aspects are over-predicted while specific aspects are under-detected (overall_experience 660 false positives; peer_interaction recall 0.08); (2) a specific-to-generic substitution pattern (missed specific aspects replaced by generic evaluative ones); (3) polarity compression toward neutral on detected aspects; and (4) a positive skew under real-review transfer, reflecting the enthusiastic register of real course reviews that the synthetic distribution under-represents. Practitioners can therefore expect reliable detection and polarity on frequent, lexically distinctive aspects, and should treat fine-grained aspects and non-positive polarities on out-of-domain reviews as the model's weak regime. The English-STEM/graduate scope is stated explicitly. *Location:* §5, §6.1.

---

## Revision summary
Numbers reconciled to a single run; framing made precise as measured-and-controlled synthetic supervision; broader-impact, circularity, transfer-limit, and practitioner-roadmap discussions made explicit; four external transfer targets and four-provider baselines; figures refined. New strengthening experiments: direct human annotation of the synthetic corpus (N1/H1), covariate-matched filtering (N2), truncated-row bias ablation (H4/D2), and a qualitative error analysis (dWED).
