# GPT Diagnostics Report

This report summarizes detection-heavy diagnostics for the saved GPT ABSA batch runs. The main purpose is to separate missed-aspect behavior from extra-aspect behavior, quantify parse failures, and identify recurring confusion patterns.

## Headline summary

- `openai-gpt-5.2-few-shot`: micro behavior summary = exact 5 rows, miss-only 0, extra-only 133, both 862; parse success 1.000; mean gold aspects 2.013; mean predicted aspects 2.991.
- `openai-gpt-5.2-few-shot-diverse`: micro behavior summary = exact 5 rows, miss-only 0, extra-only 131, both 864; parse success 1.000; mean gold aspects 2.013; mean predicted aspects 2.982.
- `openai-gpt-5.2-retrieval-few-shot`: micro behavior summary = exact 0 rows, miss-only 0, extra-only 133, both 867; parse success 1.000; mean gold aspects 2.013; mean predicted aspects 2.977.
- `openai-gpt-5.2-zero-shot`: micro behavior summary = exact 4 rows, miss-only 0, extra-only 140, both 856; parse success 1.000; mean gold aspects 2.013; mean predicted aspects 2.996.
- `openai-gpt-5.4-zero-shot`: micro behavior summary = exact 6 rows, miss-only 28, extra-only 127, both 839; parse success 0.972; mean gold aspects 2.013; mean predicted aspects 2.888.
- `openai-gpt-5.4-zero-shot-glossary`: micro behavior summary = exact 6 rows, miss-only 42, extra-only 125, both 827; parse success 0.958; mean gold aspects 2.013; mean predicted aspects 2.821.

## Top confusion pairs

### openai-gpt-5.2-few-shot
- difficulty -> tooling_usability: 50
- overall_experience -> tooling_usability: 45
- relevance -> tooling_usability: 42
- overall_experience -> feedback_quality: 42
- accessibility -> tooling_usability: 39
- peer_interaction -> tooling_usability: 39
- practical_application -> tooling_usability: 39
- overall_experience -> pacing: 38
- difficulty -> feedback_quality: 37
- materials -> tooling_usability: 37

### openai-gpt-5.2-few-shot-diverse
- difficulty -> tooling_usability: 47
- overall_experience -> tooling_usability: 46
- relevance -> tooling_usability: 45
- peer_interaction -> tooling_usability: 43
- accessibility -> tooling_usability: 41
- overall_experience -> feedback_quality: 41
- lecturer_quality -> tooling_usability: 40
- relevance -> feedback_quality: 39
- materials -> tooling_usability: 39
- prerequisite_fit -> feedback_quality: 36

### openai-gpt-5.2-retrieval-few-shot
- difficulty -> tooling_usability: 47
- overall_experience -> tooling_usability: 46
- materials -> tooling_usability: 44
- overall_experience -> feedback_quality: 41
- accessibility -> tooling_usability: 40
- peer_interaction -> tooling_usability: 40
- relevance -> tooling_usability: 39
- relevance -> feedback_quality: 39
- practical_application -> tooling_usability: 39
- accessibility -> feedback_quality: 36

### openai-gpt-5.2-zero-shot
- overall_experience -> tooling_usability: 45
- overall_experience -> feedback_quality: 42
- difficulty -> tooling_usability: 41
- relevance -> tooling_usability: 41
- accessibility -> tooling_usability: 39
- relevance -> workload: 39
- materials -> tooling_usability: 39
- peer_interaction -> tooling_usability: 36
- relevance -> feedback_quality: 34
- support -> feedback_quality: 33

### openai-gpt-5.4-zero-shot
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

### openai-gpt-5.4-zero-shot-glossary
- overall_experience -> tooling_usability: 44
- accessibility -> tooling_usability: 40
- materials -> tooling_usability: 39
- overall_experience -> feedback_quality: 37
- relevance -> tooling_usability: 36
- peer_interaction -> tooling_usability: 35
- difficulty -> tooling_usability: 34
- lecturer_quality -> feedback_quality: 32
- overall_experience -> workload: 32
- relevance -> workload: 32

## Broad-label replacement patterns

### openai-gpt-5.2-few-shot
- difficulty -> tooling_usability: 50
- overall_experience -> tooling_usability: 45
- relevance -> tooling_usability: 42
- overall_experience -> feedback_quality: 42
- accessibility -> tooling_usability: 39
- peer_interaction -> tooling_usability: 39
- practical_application -> tooling_usability: 39
- difficulty -> feedback_quality: 37
- materials -> tooling_usability: 37
- support -> feedback_quality: 36

### openai-gpt-5.2-few-shot-diverse
- difficulty -> tooling_usability: 47
- overall_experience -> tooling_usability: 46
- relevance -> tooling_usability: 45
- peer_interaction -> tooling_usability: 43
- accessibility -> tooling_usability: 41
- overall_experience -> feedback_quality: 41
- lecturer_quality -> tooling_usability: 40
- relevance -> feedback_quality: 39
- materials -> tooling_usability: 39
- prerequisite_fit -> feedback_quality: 36

### openai-gpt-5.2-retrieval-few-shot
- difficulty -> tooling_usability: 47
- overall_experience -> tooling_usability: 46
- materials -> tooling_usability: 44
- overall_experience -> feedback_quality: 41
- accessibility -> tooling_usability: 40
- peer_interaction -> tooling_usability: 40
- relevance -> tooling_usability: 39
- relevance -> feedback_quality: 39
- practical_application -> tooling_usability: 39
- accessibility -> feedback_quality: 36

### openai-gpt-5.2-zero-shot
- overall_experience -> tooling_usability: 45
- overall_experience -> feedback_quality: 42
- difficulty -> tooling_usability: 41
- relevance -> tooling_usability: 41
- accessibility -> tooling_usability: 39
- materials -> tooling_usability: 39
- peer_interaction -> tooling_usability: 36
- relevance -> feedback_quality: 34
- support -> feedback_quality: 33
- prerequisite_fit -> feedback_quality: 33

### openai-gpt-5.4-zero-shot
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

### openai-gpt-5.4-zero-shot-glossary
- overall_experience -> tooling_usability: 44
- accessibility -> tooling_usability: 40
- materials -> tooling_usability: 39
- overall_experience -> feedback_quality: 37
- relevance -> tooling_usability: 36
- peer_interaction -> tooling_usability: 35
- difficulty -> tooling_usability: 34
- lecturer_quality -> feedback_quality: 32
- prerequisite_fit -> feedback_quality: 32
- practical_application -> tooling_usability: 32

