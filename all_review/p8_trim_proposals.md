# P8 de-duplication proposals (to apply in the consolidated fold-in)

Conservative, ~150-185 words. Apply items 1-5 (clear wins); 6-7 optional. Preserve all
unique control content + every reviewer-requested caveat (each keeps >=1 occurrence).

1. "High-stakes decisions need human review" appears 4x (L1020, 1054, 1059, 1062).
   Keep 1054 (governance) + 1059 (ethics). Trim 1062's duplicated human-review+appeals tail
   to just the uncertainty-reporting point. (~30 words)
2. "0.42" faithfulness re-derived at 1054. Collapse to "(0.42 aspect-sentiment match,
   Section 5.7)"; full lower-bound/per-row explanation stays in 5.6/5.7. (~20 words)
3. §6.1 "Fourth" limitation restates the 5.7 filter mechanism (already at 986, 1004, 1054).
   Shorten to a one-clause pointer; keep it as a short numbered item (avoid renumbering). (~35 words)
4. Adjacent §6 opening paras (L1012, 1015) both say benchmark + internal learnability +
   two real annotation spaces. Merge into one, keeping the "strongest where generation and
   downstream pipeline connect" sentence. (~40 words)
5. "noisy resource / not calibrated ground truth" duplicated at 1054 and 1069; drop the
   duplicated "not as a calibrated ground truth" tail from one. (~12 words)
6. Audit review-shape reliability (A.22) at 1047 (detailed) + 1062 (cite) — minimal, only if
   merging with item 1. (~8 words)
7. "recovers ~60% of real-trained" at 844 (result) + 1020 (limitation) — defensible different
   framings, 170 lines apart; trim only for maximum tightening. (~10 words)

NOT to cut: MAUVE 0.23, sentence-vs-review granularity, OATS review-level check, polarity-
conditioning gains, cross-provider Gemini kappa 0.62, token-cap artifact, three-way audit
validation, any table/figure, and the numbers themselves (0.42/0.58/0.767/kappa/60%).
