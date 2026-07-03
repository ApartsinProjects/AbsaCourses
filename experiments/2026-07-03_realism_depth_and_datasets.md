# 2026-07-03 — Realism depth (judge errors, sentence-MAUVE) + dataset scouting

## Judge inability to separate (n2_judge_errors_and_sentence_mauve.py)
The frontier judge (gpt-5.4) is at ceiling on synthetic (97-100% detected regardless
of length/aspects/density), so its confusion is entirely FALSE POSITIVES on real
reviews (overall 15%), and it is LENGTH-driven:
- real reviews 25-64 words: FP 0.28; 64-104: 0.10; 104-202: 0.16; 202+: 0.06.
- spearman(length, false-synthetic) = -0.218, p=0.002. Short real reviews look
  synthetic; long messy personal reviews read human. Aspects/density irrelevant.
- Only 3/150 synthetic fooled the judge; all long (148-302w) and all non-GPT (2 GLM,
  1 Gemini). Consistent, too few for stats.

## Sentence-level MAUVE (apples-to-apples, N=1000/side)
- OMSCS vs OMSCS ceiling: 0.977
- OMSCS vs synthetic: 0.234
- OMSCS vs Herath (two REAL corpora): 0.027
- Herath vs synthetic: 0.009
Headline: synthetic is ~9x closer to OMSCS than a second real corpus (Herath) is.
Two real educational sources are distributionally far more different from each other
than synthetic is from its target. Added to paper limitation 2 + response #5a.

## Realism by density/aspects (n2_realism_by_density.py)
Not informative: judge at ceiling (96-100%) in every density/aspect bin;
spearman(density, detected)=-0.04 p=0.59. Kept out of paper.

## Dataset scouting for a third (review-level) transfer target
- OATS-ABSA Coursera tuples (RiTUAL-MBZUAI): REVIEW-LEVEL, 1,680 reviews, median 70
  words (closest granularity match to synthetic 122w), 11 project aspects after
  mapping. Ref [41] Chebolu et al. LREC-COLING 2024. TF-IDF transfer 0.343
  (near Herath 0.374). BERT running. Fixes the falsifiable "no full-review data".
- M-ABSA Coursera: sentence-level (14w), weak transfer (BERT 0.26); diagnostic only.
- Mendeley Teacher Performance Evaluation (>2M): overall sentiment ONLY, no per-aspect
  labels in the release; NOT usable as ABSA. Ruled out.
- No public full-review INSTITUTIONAL course-eval ABSA corpus exists (only MOOC/OATS).

## OATS transfer result + domain-vs-granularity
- OATS BERT transfer: micro-F1 0.2754 (TF-IDF 0.343). Weak, approximately equal to
  sentence-level M-ABSA (BERT 0.263), despite OATS being review-level (70w).
- Pattern across 4 real targets: institutional (Herath 0.48, EduRABSA ~0.4) transfer
  well; MOOC (M-ABSA 0.26, OATS 0.28) transfer weakly, INDEPENDENT of granularity.
  => the transfer bottleneck is DOMAIN (institutional vs MOOC), not review length.
  Synthetic is OMSCS-like (institutional). OATS NOT added as headline target.

## MAUVE entity-scrubbing fairness control (n2_mauve_scrubbed.py)
Scrubbing course codes / instructor names / platform terms from both sides:
synth-OMSCS 0.262 -> 0.245 (effect -0.017, negligible); real-real 0.030 -> 0.029;
ceiling ~0.96. The synth-real distributional gap is NOT entity/topic-driven; it is
genuine style. Added a robustness clause to limitation 2.

**Status:** realism-depth + scrubbing DONE and folded; OATS/domain framing pending user.
