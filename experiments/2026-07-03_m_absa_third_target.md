# 2026-07-03 — M-ABSA third transfer target (DIAGNOSTIC, kept out of paper)

**Goal:** test M-ABSA (Coursera, English) as a third external transfer target; keep
only if transfer holds up (reviewer h7LN scope / single-provider follow-up).

**Mapping:** `paper/m_absa_map.py` -> 1,732 sentences over 11 project aspects, but
56% is generic `overall_experience` and 77% positive-skewed; sentence-level
(median 14 words). Mapped file external_data/M-ABSA_coursera/m_absa_mapped.jsonl.

**Transfer (train on synthetic overlap, eval on M-ABSA):**
- tfidf_two_step: micro-F1 0.2345, sentiment MSE 0.9255.
- bert-base-uncased (seed 42): micro-F1 0.2628, sentiment MSE 0.4216.

**Reading:** above the permutation floor (0.182) so there is some signal, but well
below Herath (BERT 0.4811) and the weakest of all targets, driven by domain shift
(MOOC/Coursera), heavy positive skew, and generic-category dominance.

**Decision:** does NOT hold up as a headline transfer target. Kept OUT of the paper
(wins-only); recorded here in the registry. The two labeled corpora (Herath,
EduRABSA) remain the transfer evidence.

**Status:** DONE (diagnostic).
