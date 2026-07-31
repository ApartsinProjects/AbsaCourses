## Summary of Changes Since the Original Submission

We thank the reviewers and Action Editor. The revision retains the original contribution (a controlled synthetic ABSA corpus plus an audit-filter-validate methodology) and strengthens it on three fronts: the label-faithfulness audit is now validated against humans and independent model families, the synthetic-to-real story is completed on a second real benchmark, and every reported control is expanded to multiple seeds and a second architecture. No headline claim was retracted.

**New empirical results**

1. Direct human validation of the audit. A three-annotator study labels a stratified sample of the synthetic corpus for aspect presence and sentiment (Fleiss kappa 0.70 on presence). Human confirmation of a declared aspect rises monotonically with the audit score, and the human labels reproduce the audit's presence-faithful / sentiment-noisier split (human sentiment agreement about 0.40, matching the audit's strict 0.42). (New Table 15; Section 5.7.)

2. Audit independence. Two independent open-weights auditors (Llama-3.3-70B, GLM-4.6) reproduce the faithfulness audit (cross-family Cohen kappa 0.56 to 0.65), so the signal is not an artifact of a single judge family. (Appendix A.4.4.)

3. Provider-agnostic pipeline. Both the generation pipeline and the prompted baseline now reproduce across four generator families spanning four providers (GPT-5.4, Gemini-2.5-Flash, GLM-4.6, Llama-3.3-70B); the multi-provider zero-shot baseline lands in a single narrow band below the trained encoders. (Appendix A.3.2, A.6.2.)

4. Second real benchmark (EduRABSA) completed. Built out from a passing mention into a full multi-seed synthetic-to-real transfer table, a real-only trained reference, and a synthetic-pretrain plus real-fine-tune result. Synthetic-only training now recovers about half of a real-trained model on both datasets (52% Herath, 60% EduRABSA), and synthetic pre-training followed by real fine-tuning matches or exceeds real-only training on both. (New Table 10.)

5. Cross-architecture and multi-seed controls. The faithfulness-filtering contrasts are replicated on DistilBERT alongside BERT and extended to eight seeds, and a covariate-matched comparison isolates the filtering gain from composition and prediction-mask confounds. (New Table 14; Table 13.)

6. Learnable-signal controls. Label permutation collapses detection to the trivial floor (0.182 versus 0.276), accuracy scales monotonically with training size, and restricting to faithfully labeled rows raises the ceiling, establishing genuine learnable signal rather than label priors.

7. Truncated-row robustness. Regenerating the 841 output-token-capped rows at a raised cap restores full-corpus length adherence, and the capped rows are shown not to bias the benchmark in aspect, polarity, or faithfulness. (Appendix A.1.5, A.1.6.)

**Presentation and structure**

8. Abstract completed and tightened; the contribution statement de-duplicated against Section 1.1; the paper-structure paragraph moved to the end of the Introduction.

9. All figures redrawn on white backgrounds with enlarged, non-overlapping labels; Figure 2 redesigned with labeled panels and in-plot annotations; Figure 5's legend overlap removed; Figure 4 model names set horizontally.

10. Table 1 column widths and padding fixed and inter-table spacing standardized; every table and figure is now introduced in the immediately preceding paragraph with a note on what it shows.

11. Reference list alphabetized by author surname (TMLR requirement) with all in-text citations renumbered; seven recent (2024 to 2026) citations added for literature completeness.

12. Appendices reorganized from a long flat list into seven thematic sections (A.1 to A.7) with subsections, and a duplicated appendix table removed, for easier navigation.

13. Code, datasets, the human-ranking study, and model checkpoints released as a versioned archive.

Counts: main tables 12 to 15, appendix tables 15 to 28, main figures 5 to 6, appendix sections reorganized into 7 themes. Point-by-point replies appear in the per-reviewer author responses.
