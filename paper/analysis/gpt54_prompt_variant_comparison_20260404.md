# GPT-5.4 Prompt Variant Comparison

| Approach | micro-P | micro-R | micro-F1 | macro-F1 | parse success | sentiment MSE |
|---|---:|---:|---:|---:|---:|---:|
| `openai-gpt-5.4-zero-shot` | 0.2095 | 0.3005 | 0.2469 | 0.2364 | 0.972 | 0.7322 |
| `openai-gpt-5.4-zero-shot-glossary` | 0.2081 | 0.2916 | 0.2429 | 0.2366 | 0.958 | 0.7087 |
| `bert-base-uncased` | 0.2012 | 0.4396 | 0.2760 | 0.3364 | n/a | 0.4959 |

The glossary prompt reduced recall slightly and did not improve overall micro-F1 relative to the sparse zero-shot baseline.
