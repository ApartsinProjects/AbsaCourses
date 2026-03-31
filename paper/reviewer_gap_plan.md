# Reviewer Gap Plan

## Recommended paper type
Dataset-and-pipeline paper with conservative empirical claims.

## Strongest thesis
A synthetic educational review generation pipeline can produce a diverse, aspect-labeled ABSA dataset that supports internally consistent downstream modeling, calibration, and robustness analysis for higher-education review mining.

## Defensible now
- The repository contains a coherent synthetic data generation pipeline and a coherent ABSA analysis pipeline.
- The released JSONL dataset is diverse in style, short-form review length, and aspect composition.
- The recorded BERT notebook results show strong internal test performance on the project’s cleaned split.
- Additional baseline experiments confirm learnability, data-efficiency trends, and non-trivial style robustness on the released dataset.

## Not defensible yet
- Generalization from synthetic reviews to real student feedback.
- Claims of educational deployment readiness.
- Claims that the two-pass generator is necessary unless ablation experiments are added.
- Claims that the synthetic labels are faithful enough to replace human annotation without human evaluation.

## Highest-priority missing evidence
1. Synthetic-to-real transfer:
   Train on synthetic data and evaluate on a real student-feedback ABSA dataset.
2. Human validation:
   Ask domain raters to judge realism, aspect correctness, and sentiment faithfulness.
3. Generator ablations:
   Remove second-pass refinement, persona variation, and noise injection one at a time.
4. Hybrid training:
   Compare synthetic-only, real-only, and synthetic-plus-real fine-tuning.
5. Error analysis:
   Qualitatively inspect failures for confused-student style and weak aspects such as relevance and difficulty.

## Figures and tables that most help acceptance
- Figure: synthetic data generation pipeline
- Figure: ABSA analysis pipeline
- Figure: aspect-sentiment heatmap
- Figure: learning curve
- Figure: held-out-style robustness
- Table: dataset summary
- Table: representative examples
- Table: recorded notebook BERT results
- Table: multi-seed baseline summary

## Current artifact locations
- Manuscript draft: `E:\Projects\CourseABSA\paper\course_absa_manuscript.html`
- Analysis script: `E:\Projects\CourseABSA\paper\edu_absa_paper_analysis.py`
- Output directory: `E:\Projects\CourseABSA\paper\outputs`
