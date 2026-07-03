# 2026-07-03 — N3: generator-agnostic pipeline (reviewer h7LN, single provider)

**Reviewer ask:** generation is confined to a single provider's GPT family and a
single structured prompting methodology.

**Method:** regenerated the same 150 label-conditioned realism prompts with four
generator families across four providers via OpenRouter, then scored every
generation with the identical `gpt-5.2` label-fidelity audit (same prompt/schema
as Section 5.6). GLM-4.6 is a thinking model that returns empty content by default;
fixed with `reasoning:{enabled:false}`. Driver `paper/n3_generator_fidelity.py`,
summary `paper/outputs/n3_generator_fidelity_summary.json`.

**Result (per-aspect, same auditor, same prompts):**
| Provider | Model | aspects | support | sentiment-match |
|---|---|---|---|---|
| OpenAI | gpt-5-nano | 315 | 0.937 | 0.759 |
| Google | gemini-2.5-flash | 315 | 0.968 | 0.857 |
| Zhipu | glm-4.6 | 251 | 0.968 | 0.904 |
| Meta | llama-3.3-70b | 315 | 0.921 | 0.787 |

**Reading:** all four families produce auditable, label-faithful reviews (support
0.92-0.97, sentiment-match 0.76-0.90); the two non-GPT families (Google closed,
Zhipu open) match or exceed the same-family gpt-5-nano generator. The
generate-audit-filter pipeline is not specific to the GPT family, and the audit
behaves consistently across providers. Comparison is construct-matched (one auditor,
one prompt set); rates are read across families, not against the corpus-level 0.42.

**Status:** DONE.
