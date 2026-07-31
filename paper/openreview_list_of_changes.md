## Summary of Changes Since the Original Submission

We thank the reviewers and the Action Editor for a careful and constructive review. The revision keeps the original contribution (a controlled synthetic ABSA corpus with an audit-filter-validate methodology) and strengthens it: the label-faithfulness audit is now validated against humans and against independent model families, the synthetic-to-real evaluation is completed on a second real benchmark, and every reported control is expanded to multiple seeds and a second architecture. No result or headline claim was withdrawn; the revision adds evidence, broadens the evaluation, and tightens the presentation.

**New empirical results**

1. Direct human validation of the audit on the synthetic corpus: a three-annotator study (Fleiss kappa 0.70 on aspect presence) in which human confirmation of a declared aspect rises monotonically with the audit score and reproduces the presence-faithful, sentiment-noisier pattern (Table 15, Section 5.7).

2. Audit independence: two independent open-weights auditors (Llama-3.3-70B and GLM-4.6) reproduce the audit at cross-family Cohen kappa 0.56 to 0.65 (Appendix A.4.4).

3. Provider-agnostic pipeline: the generation pipeline and the prompted baseline both reproduce across four generator families spanning four providers (Appendix A.3.2, A.6.2).

4. A second real benchmark, EduRABSA, evaluated end to end: a multi-seed synthetic-to-real transfer table, a real-only reference, and a synthetic-pretrain-plus-fine-tune result (Table 10).

5. Cross-architecture and multi-seed controls: the faithfulness-filtering contrasts are replicated on DistilBERT alongside BERT and extended to eight seeds, with a covariate-matched comparison isolating the filtering effect (Table 14, Table 13).

6. Signal-validity controls: label permutation, training-size scaling, and a clean-label ceiling establish learnable signal rather than label priors.

7. Truncated-row robustness: the 841 output-token-capped rows are regenerated at a higher cap and shown to leave the benchmark unchanged in aspect, polarity, and faithfulness (Appendix A.1.5, A.1.6).

**Presentation and structure**

8. The abstract is complete and tightened, Section 1.1 is de-duplicated, and the paper-structure paragraph now closes the Introduction.

9. All figures use white backgrounds with enlarged, non-overlapping labels; Figure 2 is redesigned with labeled panels, and Figures 4 and 5 are re-laid for legibility.

10. Table 1 formatting and inter-table spacing are standardized, and every table and figure is introduced in the immediately preceding paragraph.

11. The reference list is alphabetized by author surname (TMLR convention) with citations renumbered, and recent (2024 to 2026) citations are added.

12. The appendices are organized into seven thematic sections with subsections.

13. The corpus, datasets, human study, and model checkpoints are released as a versioned archive.

**Further strengthening in this round**

14. Figure 3's left panel now plots the Herath nine-aspect overlap, consistent with Table A3.

15. The EduRABSA transfer is reported with full uncertainty (Table 10 gives mean and standard deviation across eight seeds on all metrics, with the seeds and the train and test sizes), and the appendix screening table now reports the eight-seed EduRABSA value (0.448) consistent with Table 10.

16. Figure 6 now shows a real-only-from-scratch learning curve alongside the synthetic-pretrained curve at every sample size across five seeds with confidence intervals; the accompanying statement describes a consistent per-budget pre-training boost and roughly double data efficiency in the low-data regime, with the two curves converging near the full real set.

17. The human study now reports inter-annotator agreement for sentiment, Wilson intervals on the quartile confirmation rates, and a review-level clustered bootstrap, and states the annotation protocol (each candidate aspect shown, its declared presence and sentiment hidden) and the Table 15 column definition (declared aspect decisions).

18. A Data and Code Availability section (Section 6.4) gives the anonymized Zenodo DOI (10.5281/zenodo.21717252) and lists the archive contents (corpus, fixed splits, prompts, schema, both mappings, audit scores, scripts), and states that the released corpus ships the regenerated higher-cap rows together with the original rows.

19. Both real-data mappings are documented in a new appendix (A.8) with parallel Herath and EduRABSA source-label tables and the EduRABSA support and polarity distribution.

20. The generation cost is quantified (about US$3 for the full pipeline), roughly three orders of magnitude below fair-pay human annotation per label (Section 6.2).

21. The pre-training comparison is stated as matching or slightly exceeding real-only training, and the transfer metric is described as external real-data evaluation; the generation description reflects the inter-cycle prompt-revision loop; and the M-ABSA dataset is cited.

Counts: main tables 12 to 15, appendix tables 15 to 30, main figures 5 to 6, references 41 to 42. Point-by-point replies to each reviewer accompany the revision.
