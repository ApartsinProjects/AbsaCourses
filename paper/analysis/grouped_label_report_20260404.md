# Grouped-Label Evaluation

This report recomputes multilabel detection metrics after collapsing the original 20 aspects into coarser groups.

## Grouping schemes

### Pedagogical groups from the paper
- `instructional_quality`: clarity, lecturer_quality, materials, feedback_quality
- `assessment_course_management`: assessment_design, exam_fairness, grading_transparency, organization, tooling_usability
- `learning_demand_readiness`: difficulty, workload, pacing, prerequisite_fit
- `learning_environment`: support, accessibility, peer_interaction
- `engagement_value`: relevance, interest, practical_application, overall_experience

### Confusion-informed coarse groups
These groups are regularized from observed confusion patterns rather than taken directly from a raw clustering output, because the fully unsupervised clustering produced unstable singleton groups for lightly confused labels.
- `teaching_content`: clarity, lecturer_quality, materials, relevance, practical_application
- `assessment_structure`: assessment_design, exam_fairness, grading_transparency, organization
- `demand_readiness`: difficulty, workload, pacing, prerequisite_fit
- `support_friction`: support, accessibility, peer_interaction, feedback_quality, tooling_usability
- `global_engagement`: interest, overall_experience

## Summary metrics

### none_20_aspects
- `bert-base-uncased`: micro-F1 0.2780, macro-F1 0.3410, samples-F1 0.3014, subset accuracy 0.0300
- `distilbert-base-uncased`: micro-F1 0.2690, macro-F1 0.3357, samples-F1 0.2960, subset accuracy 0.0270
- `distilbert_joint`: micro-F1 0.2524, macro-F1 0.3248, samples-F1 0.2668, subset accuracy 0.0050
- `openai-gpt-5.4-zero-shot`: micro-F1 0.2469, macro-F1 0.2364, samples-F1 0.2341, subset accuracy 0.0060
- `bert_joint`: micro-F1 0.2447, macro-F1 0.3208, samples-F1 0.2651, subset accuracy 0.0090
- `tfidf_two_step`: micro-F1 0.2441, macro-F1 0.2796, samples-F1 0.2372, subset accuracy 0.0020
- `albert-base-v2`: micro-F1 0.1829, macro-F1 0.1828, samples-F1 0.1805, subset accuracy 0.0000
- `roberta-base`: micro-F1 0.1829, macro-F1 0.1828, samples-F1 0.1805, subset accuracy 0.0000

### pedagogical
- `albert-base-v2`: micro-F1 0.5307, macro-F1 0.5283, samples-F1 0.5149, subset accuracy 0.0000
- `roberta-base`: micro-F1 0.5307, macro-F1 0.5283, samples-F1 0.5149, subset accuracy 0.0000
- `bert_joint`: micro-F1 0.5236, macro-F1 0.5244, samples-F1 0.5185, subset accuracy 0.0390
- `distilbert-base-uncased`: micro-F1 0.5188, macro-F1 0.5218, samples-F1 0.5177, subset accuracy 0.0550
- `distilbert_joint`: micro-F1 0.5124, macro-F1 0.5159, samples-F1 0.5046, subset accuracy 0.0320
- `bert-base-uncased`: micro-F1 0.5110, macro-F1 0.5148, samples-F1 0.5046, subset accuracy 0.0700
- `tfidf_two_step`: micro-F1 0.4991, macro-F1 0.5028, samples-F1 0.4800, subset accuracy 0.0280
- `openai-gpt-5.4-zero-shot`: micro-F1 0.4850, macro-F1 0.4574, samples-F1 0.4615, subset accuracy 0.0510

### confusion_regularized
- `albert-base-v2`: micro-F1 0.5281, macro-F1 0.5219, samples-F1 0.5127, subset accuracy 0.0000
- `roberta-base`: micro-F1 0.5281, macro-F1 0.5219, samples-F1 0.5127, subset accuracy 0.0000
- `bert_joint`: micro-F1 0.5267, macro-F1 0.5189, samples-F1 0.5197, subset accuracy 0.0420
- `distilbert_joint`: micro-F1 0.5248, macro-F1 0.5154, samples-F1 0.5170, subset accuracy 0.0380
- `openai-gpt-5.4-zero-shot`: micro-F1 0.5130, macro-F1 0.4439, samples-F1 0.4871, subset accuracy 0.0690
- `tfidf_two_step`: micro-F1 0.5112, macro-F1 0.4932, samples-F1 0.4944, subset accuracy 0.0420
- `distilbert-base-uncased`: micro-F1 0.5020, macro-F1 0.5010, samples-F1 0.4942, subset accuracy 0.0520
- `bert-base-uncased`: micro-F1 0.4982, macro-F1 0.4960, samples-F1 0.4885, subset accuracy 0.0730

