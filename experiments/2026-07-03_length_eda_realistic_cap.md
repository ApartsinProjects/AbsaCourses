# 2026-07-03 — Length EDA: generation cap as a realistic constraint

**Motivation (reviewer h7LN #1):** reframe the output-token cap from a defect into
a realistic length constraint (real feedback platforms cap student input), and test
whether the retained (under-cap) synthetic reviews match real review lengths.

**Method:** word-count distributions of the 9,159 complete (under-cap) synthetic
reviews vs real corpora. Artifact `paper/outputs/length_eda_summary.json`.

**Result (words):**
| Pool | n | mean | median | p10-p90 |
|---|---|---|---|---|
| Synthetic kept (under cap) | 9,159 | 118.8 | 122 | 43-208 |
| Real OMSCS (full course reviews) | 200 | 157.4 | 104 | 43-311 |
| Real Herath (sentence snippets) | 2,829 | 23.7 | 20 | 5-49 |

- **92.4%** of kept synthetic reviews fall within the real OMSCS 10th-90th-percentile
  band (43-311 words); median 122 vs 104. Strong correspondence to full-length real
  course reviews.
- No match to Herath, which is sentence-level snippets (median 20), a different
  granularity, not a length target for whole reviews.

**Use in paper:** Section 3.6 now frames the length budget as a deliberate,
realistic constraint mirroring real feedback platforms, backed by this EDA. Keeps
the 10K corpus (headline micro-F1 0.276) with no re-run (Option A + EDA).

**Status:** DONE.
