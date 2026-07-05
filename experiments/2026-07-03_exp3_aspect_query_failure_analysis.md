# Exp3 aspect-query failure analysis (why it failed + mitigations)

Aspect-query detection micro-F1 UNDERperformed the multi-label baseline:
Herath 0.288 (base 0.327), M-ABSA 0.188 (base 0.263). Root cause is a training
failure, not a flaw in the reformulation idea.

## Smoking gun: the model collapsed to the class prior
Training cross-entropy stayed flat at 0.559->0.571 (Herath) / 0.572->0.569 (M-ABSA).
The cross-entropy of a constant predictor at the 25% positive base rate
(neg_per_pos=3 sampling) is -[0.25 ln0.25 + 0.75 ln0.75] = 0.5623. The observed loss
sits ON that floor: the network learned the prior, not the task. It never fit.

## Why it underfit (design contrast with the baseline)
| | Multi-label baseline (works) | Aspect-query Exp3 (failed) |
|---|---|---|
| Encoder passes | ONE pass/review, 20 aspects jointly | 20 SEPARATE passes/review (one per aspect) |
| Head | shared, learns aspect correlations | independent 2-class head, random init |
| Loss | BCEWithLogitsLoss + pos_weight (imbalance-corrected) | plain CrossEntropyLoss, no pos_weight |
| Checkpoint | bert-base (fine for multi-label) | bert-base -- NOT NLI-pretrained; must learn pairwise entailment from scratch |
| Budget | 3 epochs enough (easy shared task) | 3 epochs far too few for a harder pairwise task |

Four compounding reasons:
1. **No NLI pretraining.** The pair formulation (review [SEP] aspect phrase -> present?)
   is literally an entailment task. bert-base has no entailment prior, so the random
   [CLS] head has to learn cross-attention between review and phrase from scratch. In 3
   epochs at lr 3e-5 it doesn't move off the prior.
2. **No imbalance handling.** The baseline uses pos_weight; Exp3 relies only on 3:1 neg
   subsampling, so the model's easiest local optimum is "predict the 25% prior," exactly
   where the loss stalled.
3. **Train/eval prevalence mismatch.** Trained at 25% positive, evaluated at the real
   ~10-15% aspect prevalence; threshold calibration cannot rescue a model that only
   encodes the prior.
4. **20x the forward passes for the same budget.** Per review the pair setup does 20
   encodings vs the baseline's 1, so effective learning signal per aspect per epoch is
   much thinner.

## Mitigations (ranked by expected payoff)
1. **Start from an NLI/MNLI-pretrained checkpoint** (roberta-large-mnli, bart-large-mnli,
   deberta-v3-base-mnli, or a zero-shot NLI model). This is the standard recipe for
   query/hypothesis ABSA and directly addresses reason 1 -- the model already does
   pairwise entailment. Highest-leverage single change.
2. **Imbalance-aware loss**: weighted CrossEntropy / focal loss, or match training
   prevalence to the eval prevalence. Fixes reasons 2-3.
3. **More epochs + warmup + slightly higher lr** (5-10 epochs, lr warmup). The flat loss
   is underfitting; give it budget.
4. **Hypothesis templating**: "This course review discusses {aspect}: {description}."
   rather than a bare phrase -- gives the NLI head a natural-language hypothesis.
5. **Hybrid (keep the winning encoder)**: retain the shared multi-label head but CONDITION
   it on aspect-phrase embeddings (FiLM / cross-attention), so we get meaning-based
   matching WITHOUT discarding the shared representation that made the baseline strong.
6. **Dual-encoder / contrastive**: embed review and aspect separately, score by cosine.
   Cheap (no 20x passes), retrieval-style, and calibratable.

## Recommendation
Aspect-query is only worth revisiting with an NLI-pretrained checkpoint + imbalance loss
+ more epochs (mitigations 1-3). As a from-scratch bert-base multitask it is dominated by
the simpler multi-label head. Given Exp1 (windowing) and Exp2 (sentence-level training)
already deliver granularity-matched wins, aspect-query is LOW priority; documented here
as a null with a concrete revive path. A scoped retry ("Exp3b": roberta-large-mnli,
weighted loss, 6 epochs, Herath+M-ABSA only) would take ~30-40 min GPU if pursued.
