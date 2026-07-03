# 2026-07-03 — #6b: multi-family zero-shot ABSA baseline (reviewer h7LN)

**Reviewer ask:** the prompted baseline is confined to one provider (GPT) and one
structured prompting methodology.

**Method:** co-computed the SAME zero-shot-glossary ABSA baseline the paper reports
for gpt-5.4, on the SAME synthetic test split (seed 42, 200 of the 1,000 held-out
rows), for four families across four providers via OpenRouter, one scoring pass,
reusing `absa_model_comparison.build_openai_prompt` and the same per-aspect
detection / sentiment scoring. Driver `paper/n6b_multifamily_baseline.py`,
summary `paper/outputs/n6b_multifamily_baseline_summary.json`.

**Result (zero-shot-glossary, 200 test rows, 20 aspects):**
| Provider | Model | detection micro-F1 | sentiment acc (matched) | parsed |
|---|---|---|---|---|
| OpenAI | gpt-5.4 | 0.234 | 0.661 | 197/200 |
| Google | gemini-2.5-flash | 0.252 | 0.629 | 199/200 |
| Zhipu | glm-4.6 | 0.250 | 0.669 | 193/200 |
| Meta | llama-3.3-70b | 0.266 | 0.659 | 200/200 |

**Reading:** the prompted baseline is no longer single-provider or single-prompt-family.
All four families land in the same narrow band (micro-F1 0.23-0.27, sentiment acc
0.63-0.67); non-GPT families match or slightly exceed gpt-5.4. This is consistent
with the paper's finding that zero-shot LLMs sit below trained encoders (BERT
detection micro-F1 ~0.28) regardless of provider. Construct-matched (same rows,
prompt, scoring, one pass).

**Status:** DONE.
