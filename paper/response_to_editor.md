# Response to the Action Editor

*A Controlled Synthetic Benchmark for Educational Aspect-Based Sentiment Analysis (TMLR)*

Dear Action Editor,

Thank you, and please pass our thanks to the three reviewers, for a careful and constructive review. We have revised the paper thoroughly and believe every requested change is now in place. Detailed, point-by-point responses to each reviewer accompany this note; below we summarize the significant changes and confirm how the revision aligns with each reviewer's requests.

## Significant changes in this revision

- **Direct human validation of the audit on the synthetic corpus.** Three annotators labeled a stratified sample of the actual synthetic reviews; human confirmation rises monotonically with the audit score, and inter-annotator reliability, sentiment agreement, confidence intervals, and a review-level clustered bootstrap are reported (Table 15).
- **A second real benchmark, completed end to end.** EduRABSA is added as a full multi-seed synthetic-to-real transfer with a real-only reference and a synthetic-pretrain-plus-fine-tune result, reported with uncertainty (Table 10), so the transfer story now rests on two independent annotated corpora.
- **Audit independence and provider-agnostic pipeline.** The faithfulness audit is reproduced by two independent open-weights auditors, and both the generation pipeline and the prompted baseline are reproduced across four generator families spanning four providers.
- **A stronger sample-efficiency experiment.** Figure 6 now shows a real-only-from-scratch learning curve alongside the synthetic-pretrained curve at every data size across five seeds with confidence intervals, and the accompanying claim is stated conservatively.
- **Faithfulness-aware filtering isolated.** A covariate-matched comparison removes composition and prediction-mask confounds, and a quartile dose-response confirms the effect end to end.
- **Framing, transparency, and reproducibility.** The resource is described as quantified noisy synthetic supervision rather than a gold benchmark; a Data and Code Availability section points to an anonymized archive; the real-data mapping procedure and both mappings are documented; and the generation-cost claim is quantified.

We also corrected two factual issues we found while revising (a mislabeled figure panel and a transcription error in an appendix table), tightened the writing, and reconciled the internal numbers and citations.

## Alignment with each reviewer

**Reviewer nfat** asked for direct human annotation of the synthetic corpus, filtering subsets matched on covariates, resolution of the inconsistent numbers and the incomplete abstract, a clearer noisy-supervision framing, and a broader impact discussion. All are addressed: the human study validates the audit on synthetic text; a covariate-matched filtering comparison isolates faithfulness; the abstract is complete and the Table 8 versus Table 9 and aspect-count issues are reconciled through consistent multi-seed reporting; the corpus is framed as measured noisy supervision; and the ethics discussion now covers instructor evaluation, comment privacy, stylistic bias, and the safeguards for institutional use.

**Reviewer h7LN** asked about the token-capped rows, the low-value bottom quartile, the strict 0.42 match and the moderate audit-human agreement, statistical realism and full-schema re-annotation, broader baselines beyond one provider, figure and table formatting, and the narrow evaluation scope. All are addressed: the truncated rows are regenerated and shown to be unbiased; the bottom quartile is exactly the signal the filter exploits; the audit is positioned as a validated selection instrument, not an oracle, and is validated directly on synthetic text; the provider axis is covered across four families on the full test split; the named figures and table were reformatted; and the external evaluation now spans four real corpora with the scope stated explicitly.

**Reviewer dWED** (who assessed the evidence positively) asked for a dedicated discussion of generator-auditor circularity, a bias analysis of the incomplete rows, two additional citations, a stronger statement of transfer limits, and a concrete practitioner roadmap. All are addressed: circularity is settled empirically with independent open-weights and cross-family auditors; the incomplete rows are shown unbiased in aspect, polarity, and faithfulness; both suggested works are cited; the limits of the nine-aspect, partial-recovery transfer are stated prominently; and a fine-tuning-size roadmap with monitoring guidance is provided.

We are grateful for the feedback, which materially improved the paper, and we hope the revision now meets the bar for TMLR.

Sincerely,
The Authors
