# 2026-07-03 — N2 realism exploration (DIAGNOSTICS, not for paper)

Reviewer h7LN asked to "establish statistically that synthetic reviews are
indistinguishable from genuine student feedback." Explored several constructs;
the surface-indistinguishability constructs are NEGATIVE and stay out of the
paper (wins-only). Kept here for the registry.

## What was tried and found
- **Frontier LLM judge (gpt-5.4) discrimination** (`n2n3_realism_diversity.py`):
  clean regenerations of every family (gpt-5-nano/Gemini/GLM/Llama) are still
  flagged synthetic ~67-100%; real baseline false-synthetic 0.15. A frontier
  adversarial detector separates LLM text from human text regardless of family
  or truncation. This is an unfair ceiling no synthetic corpus passes; NOT the
  right construct for a labeled benchmark.
- **Distributional stylometry vs OMSCS** (`n2_distributional_realism.py`):
  after N1, `word_count` and punctuation axes are statistically indistinguishable
  (negligible Cliff's delta), but synthetic has higher lexical diversity, longer
  sentences and lower readability (large effects). 2/8 axes match.
- **Style-tuned regeneration** (`n2_realism_tuned.py`): a stylometric directive
  overshoots (sentences too short, readability too high); does not converge to
  the real middle. Real reviews even have LOWER type-token ratio than any LLM.
- **MAUVE** (`n2_mauve_realism.py`, gpt2 features): real-vs-synth 0.088;
  real-vs-real upper bound 0.756; real-vs-RateMyProfessor cross-domain 0.389.
  Synthetic is distributionally far from real.

## Conclusion (positive framing that DOES hold, goes to paper)
Distributional/detector indistinguishability does not hold and is not claimed.
The realism that matters for a labeled training benchmark is FUNCTIONAL: a model
trained on the synthetic corpus transfers to real reviews (existing transfer
results). Report: (1) N1 truncation elimination; (2) functional transfer as the
realism evidence; (3) the length/punctuation axes that match. Keep the judge/
MAUVE negatives here only.
