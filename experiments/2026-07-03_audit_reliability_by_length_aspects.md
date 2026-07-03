# 2026-07-03 — Audit reliability vs review length and aspect count

**Question (reviewer-driven):** why does audit-human agreement vary with real-review
length, and is the effect length or aspect-count?

**Method:** decompose the 2,482 v7 aspect decisions by review word count and by
number of declared aspects, for both auditors. Scripts `paper/v7_kappa_by_length.py`,
`paper/v7_perturbation_by_length.py`, `paper/v7_by_len_and_aspects.py`.
Artifact `paper/outputs/kappa_by_length.json`.

**Findings (gpt-4.1-mini; Gemini mirrors):**
- Raw kappa by length is an inverted-U: <=15w 0.51, 16-25 0.51, 26-40 0.63,
  41-70 0.64, 71+ 0.39.
- Mechanism (per-perturbation): recall (keep genuine faithful aspect) RISES with
  content 0.60 -> 0.77; specificity (reject injected absent aspect) FALLS with
  aspect density 0.90 (1 aspect) -> 0.74 (3-4+).
- length vs n_aspects Pearson 0.48 (confounded). Controlling n_aspects=1, kappa
  RISES with length (<=20w 0.50 -> 41+w 0.56): longer IS better when aspect count
  is fixed. The long-review penalty is an aspect-DENSITY effect (fabricated aspects
  blend into aspect-rich reviews), not length.

**Reading:** short reviews are hard because the audit cannot confirm real aspects
(low recall); aspect-dense reviews are hard because it cannot reject fabricated ones
(low specificity); agreement peaks at moderate length + aspect count. Explains the
Herath(short) < EduRABSA(longer) ordering of Table A19.

**In paper:** Section 5.7 pointer + Appendix A.22 / Table A20.
**Status:** DONE.
