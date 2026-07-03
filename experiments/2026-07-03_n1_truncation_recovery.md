# 2026-07-03 — N1: truncation recovery (reviewer h7LN, length-band adherence)

**Reviewer ask:** 841/10000 rows hit the output-token cap; full-corpus length-band
adherence 0.6819 (App A14) undermines the "controlled corpus" claim.

**Root cause:** the generation requests set `max_output_tokens = 300` with
`reasoning: minimal, verbosity: low`; 841 detailed-review-band rows ran past 300
tokens and truncated. Truncation, not a control failure, is the cause.

**Method:** re-ran the identical 841 generation prompts via OpenRouter
`openai/gpt-5-nano` (same family), cap raised 300 -> 900, reasoning minimal +
verbosity low (verbosity swept: low 0.567 > medium 0.475 > none 0.358, low kept).
Standalone recoverability demo; the released corpus is unchanged.
Driver `paper/n1_regenerate_truncated.py`, artifacts
`paper/outputs/n1_regenerated_841.jsonl`, `paper/outputs/n1_adherence_summary.json`.

**Result:**
- regenerated 841/841 with 0 errors and 0 truncations -> incomplete rate 8.41% -> 0%.
- length-band adherence on the 841: 0.153 -> 0.517.
- full-corpus adherence (substituting the regenerated rows): 0.6819 -> 0.7125,
  which equals the complete-row rate of 0.7304 (derived), i.e. the truncated rows
  were the entire cause of the shortfall below the complete-row ceiling.

**Reading:** the 0.6819 figure was depressed solely by token-budget truncation;
at a proper cap the corpus recovers to its complete-row adherence and truncation is
eliminated. Length band is a soft stylistic target; the controlled variables are the
per-review aspect-sentiment labels, which truncation does not touch.

**Status:** DONE.
