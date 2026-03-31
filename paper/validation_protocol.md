# Real-vs-Synthetic Realism Validation Protocol

## Goal
Test whether a strong OpenAI judge model can reliably discriminate between real public course reviews and synthetic reviews generated with a richer attribute space.

## Real review source
- Public OMSCS Reviews course pages at `https://awaisrauf.com/omscs_reviews/`
- Proposed courses for the small-scale run:
  - `CS-6200`
  - `CS-6250`
  - `CS-6400`
  - `CS-7641`

## Why this is useful
- It gives the paper an external realism proxy instead of relying only on internal train/test performance.
- It creates a concrete prompt-improvement loop with a measurable stopping condition.
- It can expose the specific textual cues that make synthetic reviews still look synthetic.

## Richer attribute space for generation
- course code and title
- semester
- student background
- motivation for taking the course
- grade band
- workload intensity
- assessment profile
- instruction quality
- support-channel experience
- administrative friction
- emotional temperature
- linguistic style
- hedging and uncertainty
- specificity markers
- recommendation stance

## Small-scale prompt cycle design
1. Cycle 0:
   Rich attributes only.
2. Cycle 1:
   Remove checklist-like symmetry and reduce explicit contrast patterns.
3. Cycle 2:
   Add realistic messiness, partial contradiction, and more domain-specific detail.

## Judge setup
- Default generator model: `gpt-5.2`
- Default judge model: `gpt-5.4`
- Judge task:
  Label blinded reviews as REAL or SYNTHETIC and report confidence.

## Metrics
- Judge accuracy
- Judge confidence gap
- False-positive and false-negative cases
- Qualitative reasons given by the judge

## Important limitation
If the judge becomes uncertain, that is useful evidence of realism, but it is not a full substitute for human validation or synthetic-to-real transfer testing.
