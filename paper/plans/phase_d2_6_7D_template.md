# Template: §6.7D Faithfulness-Aware Filtering (to be filled in after Modal results)

To be inserted into `paper/course_absa_manuscript.html` between §6.7C and §6.8 (currently between roughly line 1093 and line 1095) **only if H1 holds** (top50 or top25 micro-F1 > full micro-F1).

If H1 is supported (top-K > full):

> ### 6.7D Faithfulness-Aware Filtering of the Training Corpus
>
> The 250-review audit in §6.7B reports a corpus-level aspect-sentiment match rate of 0.42 and frames it as a noise floor. To convert that observation into a method-level finding, we extended the audit to the full 10,000-review corpus using a cost-matched judge (gpt-4.1-mini, calibrated against the gpt-5.2 audit on the original 250 rows at Spearman rho = 0.52; per-aspect support agreement was 0.845 and per-aspect sentiment-match agreement was 0.715). For each row we recorded the fraction of declared aspects whose polarity is supported by the text. We then partitioned the corpus into five training subsets: the highest-scoring 25 percent (`top25`, n = 2,500), the highest-scoring 50 percent (`top50`, n = 5,000), the full corpus (`full`, n = 10,000), the lowest-scoring 25 percent (`bot25`, n = 2,500), and a uniform random 5,000-row sample (`random_5k`) controlling for training size. The bottom and random subsets are negative controls that isolate the value of the quality signal.
>
> Table 8E reports BERT-base-uncased transfer micro-F1 on the 9-aspect mapped Herath benchmark for each subset trained under the §6.3 recipe at seed 42. [FILL: state headline outcome here.] The top-N filtered subsets [FILL: outperform / match] the full corpus, while the `bot25` subset is the weakest. The `random_5k` baseline isolates training size; the gap between `top50` and `random_5k` is [FILL: X micro-F1 points], indicating that [FILL: it is the faithfulness signal, not the smaller training size, that drives the gain / the gain mostly reflects sample selection rather than quality].
>
> [INSERT TABLE 8E HERE]
>
> [INSERT FIGURE: phase_d2_filtering_micro_f1.svg]
>
> This experiment converts the audit signal from a documented limitation into an actionable filter. Practical implication: when the synthetic corpus is used as a training resource, retaining the top-50 percent by faithfulness score yields a [FILL: 0.0X] absolute improvement in Herath transfer micro-F1 at half the training cost.

If H1 is NOT supported (top-K ties or underperforms full):

> ### 6.7D Faithfulness-Aware Filtering of the Training Corpus
>
> A natural follow-up to §6.7B asks whether the audit signal predicts downstream utility. We extended the audit to all 10,000 reviews using a cost-matched judge (gpt-4.1-mini, calibrated at Spearman rho = 0.52 against gpt-5.2), partitioned the corpus into five training subsets (`top25`, `top50`, `full`, `bot25`, `random_5k`), and retrained BERT-base on each under the §6.3 recipe at seed 42.
>
> Table 8E reports the transfer micro-F1 on the mapped Herath benchmark. [FILL: At this corpus scale and single seed, the faithfulness-filtered subsets perform within X micro-F1 points of the full corpus.] The result documents that label-quality-aware filtering does not by itself improve transfer at the present scale and is consistent with corpus quality not being the bottleneck for the 0.46 micro-F1 transfer ceiling. We retain the experiment as a documented null and recommend that future filtering work pair faithfulness scoring with regeneration of the discarded rows rather than simple subsetting.
>
> [INSERT TABLE 8E HERE]
>
> [INSERT FIGURE: phase_d2_filtering_micro_f1.svg]

## Abstract delta (only if H1 supported)

Insert into abstract after the existing transfer-result sentence (line 402):

> A faithfulness-aware filtering experiment shows that training BERT on the top 50 percent of the corpus by aspect-sentiment match yields a [FILL: 0.0X] absolute Herath micro-F1 improvement at half the training cost.

## §7.1 limitations rewrite (only if H1 supported)

The relevant sentence on line 1113 currently reads:

> Fourth, the label-faithfulness audit indicates that many declared aspect polarities are expressed only approximately in the generated text, so the corpus is best treated as a useful noisy benchmark rather than as a gold-standard annotation set.

Replace with:

> Fourth, the label-faithfulness audit confirms that polarity expression varies across the corpus and that the audit score is itself actionable: the §6.7D filtering experiment shows that training on the top-N by audit score improves transfer over training on the full corpus, so the audit functions as a built-in quality filter rather than as a fixed noise floor.

## §7.2 Educational Implications wording

The current text on line 1140 already mentions "the faithfulness-aware filtering extension flagged in Section 7.1 is the natural next step." If H1 holds, replace with:

> For the longitudinal-monitoring and reflective-practice workflows, deployments built on top of the corpus should retain only the top-50 percent of the corpus by faithfulness score (per §6.7D), and should still be checked against locally adjudicated examples.

## Table 8E format

Insert as a `<table class="paper-table">` in HTML, mirroring Table 8D format:

| Bucket | n train | n train (overlap) | Herath micro-F1 | Herath macro-F1 | Sentiment MSE |
|---|---:|---:|---:|---:|---:|
| top25 | 2,500 | [from summary] | [from summary] | [from summary] | [from summary] |
| top50 | 5,000 | [from summary] | [from summary] | [from summary] | [from summary] |
| random_5k | 5,000 | [from summary] | [from summary] | [from summary] | [from summary] |
| full | 10,000 | [from summary] | [from summary] | [from summary] | [from summary] |
| bot25 | 2,500 | [from summary] | [from summary] | [from summary] | [from summary] |

Caption: "Table 8E. Faithfulness-aware filtering of the 10K corpus. Each row reports BERT-base-uncased trained on the named subset and evaluated on the mapped 9-aspect Herath benchmark at seed 42."
