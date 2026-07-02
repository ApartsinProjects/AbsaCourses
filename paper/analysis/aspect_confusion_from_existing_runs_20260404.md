# Aspect Confusion From Existing Runs
This report uses saved per-example predictions, so it supports numerical confusion analysis without rerunning experiments. Because the task is multilabel, these are confusion proxies rather than a classical single-label confusion matrix.
## BERT-based confusion
Top missed->wrong-predicted pairs:
- peer_interaction -> overall_experience: 63
- peer_interaction -> interest: 61
- organization -> interest: 61
- organization -> overall_experience: 58
- materials -> interest: 56
- grading_transparency -> interest: 56
- materials -> overall_experience: 56
- practical_application -> overall_experience: 55
- clarity -> interest: 55
- grading_transparency -> overall_experience: 54
Most missed aspects:
- peer_interaction: 89 misses
- grading_transparency: 75 misses
- clarity: 74 misses
- practical_application: 71 misses
- difficulty: 71 misses
- relevance: 70 misses
- exam_fairness: 70 misses
- organization: 70 misses
- materials: 68 misses
- tooling_usability: 65 misses
Most common false-positive aspects:
- interest: 684 false positives
- overall_experience: 611 false positives
- support: 516 false positives
- accessibility: 280 false positives
- feedback_quality: 246 false positives
- prerequisite_fit: 163 false positives
- difficulty: 132 false positives
- clarity: 82 false positives
- lecturer_quality: 55 false positives
- relevance: 51 false positives
## GPT-5.4 zero-shot confusion
Top missed->wrong-predicted pairs:
- overall_experience -> tooling_usability: 44
- accessibility -> tooling_usability: 42
- overall_experience -> feedback_quality: 38
- difficulty -> tooling_usability: 38
- peer_interaction -> tooling_usability: 36
- relevance -> tooling_usability: 35
- materials -> tooling_usability: 35
- peer_interaction -> support: 33
- prerequisite_fit -> feedback_quality: 31
- overall_experience -> support: 30
Most missed aspects:
- overall_experience: 116 misses
- relevance: 90 misses
- interest: 86 misses
- accessibility: 85 misses
- difficulty: 85 misses
- peer_interaction: 83 misses
- lecturer_quality: 78 misses
- materials: 75 misses
- organization: 75 misses
- practical_application: 73 misses
Most common false-positive aspects:
- tooling_usability: 335 false positives
- feedback_quality: 333 false positives
- support: 273 false positives
- workload: 199 false positives
- pacing: 196 false positives
- clarity: 173 false positives
- grading_transparency: 151 false positives
- prerequisite_fit: 137 false positives
- peer_interaction: 101 false positives
- assessment_design: 89 false positives
## Interpretation
- Existing results are enough to establish numeric confusion proxies.
- A new experiment is only needed if we want semantic difficulty estimates that are less tied to one model's thresholding behavior, for example an LLM-judged aspect-pair confusability matrix.
