# Response to the Action Editor

*A Controlled Synthetic Benchmark for Educational Aspect-Based Sentiment Analysis (TMLR)*

We thank the Action Editor for the exceptionally detailed and constructive list. Every point improved the paper; we address each below with the specific change and its location in the revised manuscript. Two of the items surfaced genuine errors (a mislabeled figure panel and a transcription error in an appendix table), which we were glad to catch and correct.

### 1. Figure 3 contained the wrong dataset

Fixed, and thank you for catching this. The left panel was reading a stale file that in fact held the EduRABSA seven-aspect overlap; it now plots the correct Herath nine-aspect overlap (lecturer_quality 2,190, overall_experience 557, workload 75, and so on), consistent with Table A3. The center and right panels were already correct Herath transfer scores.

*Location:* Figure 3.

### 2. External-validation section still had Herath-only wording

Done. The Section 5.4 heading and opening are now plural, Section 4.1 lists both external sets, Table 4 gains an EduRABSA row (2,152) and refers to three real-data pools, and the Section 5.8 summary now names both mapped evaluations (Herath nine-aspect and EduRABSA seven-aspect).

*Location:* Section 5.4; Section 4.1; Table 4; Section 5.8.

### 3. Table 10 not fully consistent with its caption

Done. The synthetic-only rows now report mean plus or minus standard deviation for macro-F1, micro-recall, and sentiment MSE (eight seeds), not micro-F1 alone. The caption lists the eight seeds and the 3,432 / 2,152 train / test sizes, and states explicitly that the real-only and synthetic-pretrained references are reported on detection micro-F1 only, mirroring the Herath reference in Tables 8 and 9.

*Location:* Table 10.

### 4. Appendix contradicted the new EduRABSA result; M-ABSA undefined

Done. The Appendix A.5.4 value of 0.275 was a transcription error, duplicated from the OATS screening cell; the actual eight-seed EduRABSA result is 0.448, and the appendix sentence, caption, and table cell are corrected to match Table 10 (this also strengthens the appendix thesis, since EduRABSA is register-aligned and transfers above the MOOC targets). M-ABSA is now defined and cited (Wu et al., 2025, EMNLP).

*Location:* Appendix A.5.4; Table A24; References.

### 5. Human-annotation description and Table 15

Done. We clarify that each annotator was shown the candidate declared aspect but was blind to its declared presence and sentiment. The Table 15 column is renamed to "Declared aspect decisions" (the 610 count). We add inter-annotator agreement for sentiment (Fleiss kappa 0.49; pairwise Cohen 0.38 to 0.57), state the majority-vote rule for deriving a single human label, add Wilson 95% intervals to the four quartile rates, and add a review-level clustered bootstrap (overall confirmation 0.70, 95% CI 0.66 to 0.74; highest-minus-lowest-quartile difference 0.24, 95% CI 0.12 to 0.36).

*Location:* Section 5.7; Table 15.

### 6. Synthetic test set not fully human-verified

Addressed as an explicit limitation. We state that the human study validates the precision of the declared labels, not the recall of aspects that are present but undeclared, and that the corpus does not yet have a fully adjudicated twenty-aspect gold test set; a full-schema re-annotation that would surface undeclared-but-present aspects is identified as the natural next step.

*Location:* Section 6.1.

### 7. Per-review refinement stage

Done. There is no per-review rewrite; the operator was inaccurate. We removed the R-phi operator and the "draft review / refinement step" language from Sections 3.1 and 3.5, so the text now describes only the inter-cycle prompt revision, consistent with Figure 1 and Appendix A.2.

*Location:* Section 3.1; Section 3.5.

### 8. Moderate the superiority claims

Done. The abstract now states that synthetic pre-training followed by real fine-tuning "matches or slightly exceeds" real-only training, and "the unbiased metric" is replaced with "external real-data evaluation."

*Location:* Abstract; Section 1.1.

### 9. Figure 6 did not support the sample-efficiency claim

Done, with a new experiment. We re-ran the sweep at five seeds and added a real-only-from-scratch learning curve at every sample size (100, 250, 500, 1,000, and the full real set), each with a 95% confidence band. The full-data point now reconciles (synthetic pre-train 0.780, 0.784 within the interval; real-only 0.770, matching the 0.767 reference). We also corrected the claim: the experiment supports a consistent pre-training boost at every budget and roughly double data efficiency in the low-data regime, with the two curves converging near the full real set. We no longer state that synthetic pre-training reaches real-only quality at half the data, which this experiment does not support.

*Location:* Figure 6; Section 6.2.

### 10. Data and Code Availability

Done. A new Section 6.4 points to an anonymized Zenodo archive (DOI 10.5281/zenodo.21717252) containing the synthetic corpus, the mapped Herath and EduRABSA evaluation sets, the fixed train / validation / test splits (seed 42, 8,000 / 1,000 / 1,000), the generation prompts and attribute schema, both real-data mapping files, the full-corpus audit scores, model configurations, and the training and evaluation scripts. We state that the released corpus ships the regenerated higher-cap replacements for the 841 token-capped rows and also includes the original truncated rows separately.

*Location:* Section 6.4.

### 11. Document and validate both real-data mappings

Done. A new Appendix A.8 describes the conservative mapping procedure (each source label assigned to the schema aspect with the clearest correspondence; ambiguous categories left unmapped) and provides parallel source-label-to-aspect tables for Herath and EduRABSA (Table A29) and the EduRABSA support and polarity distribution (Table A30). The tables make explicit that assessment_design and grading_transparency are Herath-only, which is why the overlaps are nine and seven aspects.

*Location:* Appendix A.8 (Tables A29, A30).

### 12. Quantify the cost claim

Done. Section 6.2 now reports concrete figures on the OpenAI Batch API: roughly US$3 total (about US$0.44 corpus generation with gpt-5-nano, about US$1.7 for the full-corpus faithfulness audit with gpt-4.1-mini over all 10,000 rows, and about US$0.5 for the three realism-validation cycles), which is about US$0.0003 per review and roughly three orders of magnitude cheaper per label than fair-pay human annotation.

*Location:* Section 6.2.

### 13. Reduce remaining repetition

Addressed by targeted tightening. The generation process is now stated once, with the duplicated per-draft narration removed from Section 3.1 while the formal account remains in Section 3.5, and the Section 5.8 recap is collapsed into a two-sentence bridge rather than restating results already given in Section 5 and the Discussion. These remove the two largest overlaps, and every experimental number is kept in place.

*Location:* Sections 3.1, 3.2, 3.5, 5.8, 6.1.

---

Locations reference the revised manuscript; every requested change is in place at the cited location.
