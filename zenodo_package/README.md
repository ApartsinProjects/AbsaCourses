# A Controlled Synthetic Benchmark for Educational Aspect-Based Sentiment Analysis

This deposit accompanies the paper *A Controlled Synthetic Benchmark for Educational
Aspect-Based Sentiment Analysis*. It contains the synthetic corpus, the mapped real
evaluation data, the human-annotation study that validates the label-faithfulness
audit, and the code needed to reproduce the benchmark and transfer results.

## Contents

```
data/
  synthetic_corpus_10k.jsonl          10,000 synthetic course reviews over a 20-aspect
                                       pedagogical schema; each row carries text, declared
                                       aspect-sentiment labels, and generation attributes.
  herath_mapped_real_reviews.jsonl     Herath et al. (2022) student-feedback corpus mapped
                                       to the 9-aspect overlap used for external transfer.
  herath_mapping.json                  The conservative schema mapping (Herath -> our aspects).
  faithfulness_audit_row_scores.csv    Per-row LLM faithfulness-audit scores for the 10K corpus
                                       (row_id, n_aspects, n_supported, n_matched, support_rate,
                                       match_rate, row_score).

code/
  absa_model_comparison.py             ABSA engine: detection + sentiment models, training,
                                       threshold calibration, evaluation.
  absa_data_io.py                      Dataset loading helpers.
  evaluate_synthetic_to_real_transfer.py  Real-data loading, overlap restriction, transfer eval.
  multiseed_transfer_worker.py         Multi-seed synthetic-to-real transfer runner.
  checkpoint_train_worker.py           Trains and persists the best-per-target checkpoints.
  build_annotation_xlsx.py             Builds the per-rater annotation workbooks.

human_study/
  annotation_rater{1,2,3}_res.xlsx     Three raters' completed annotations of a stratified
                                       300-review synthetic sample (610 declared review-aspect
                                       decisions), blind to the declared labels.
  annotation_key.csv                   Declared labels + audit row-score per item (scoring key).
  score_human_study.py                 Computes human-vs-declared, human-vs-audit, and
                                       inter-rater agreement.
  human_study_scores.json              Scored results.

checkpoints/                           Best-per-target BERT checkpoints (added in a later
                                       version of this record; see checkpoints/README.md).
```

## Key results reproduced by this deposit

- Synthetic-to-real transfer: a BERT detector trained only on the synthetic corpus recovers
  real aspect and polarity signal on the 9-aspect Herath overlap (five-seed mean micro-F1 0.402).
- Faithfulness-aware filtering: retaining the top 50% of the corpus by audit score reduces
  transferred sentiment error at half the training cost.
- Human validation of the audit: across a stratified 300-review sample, inter-annotator
  reliability is substantial (Fleiss kappa 0.70), and human confirmation of declared aspects
  rises monotonically with the audit score (0.55 to 0.79 across audit-score quartiles).

## Reproducing

The training and transfer code targets an NVIDIA A10G (or comparable) GPU.

```bash
pip install "transformers==4.46.0" "accelerate==1.1.1" "scikit-learn==1.5.2" \
            "pandas==2.2.3" "numpy==1.26.4" torch
# multi-seed synthetic-to-real transfer:
python code/multiseed_transfer_worker.py --approach bert-base-uncased --seed 42 --out ./out
# score the human study:
python human_study/score_human_study.py
```

## Provenance and licensing

- The synthetic corpus and the human-annotation study are released under CC-BY-4.0
  (`LICENSE-DATA`). The corpus is machine-generated and contains no identifiable student data.
- The code is released under the MIT License (`LICENSE-CODE`).
- The mapped Herath evaluation data derives from the publicly released Herath et al. (2022)
  student-feedback corpus under its MIT license; only a conservative schema mapping and
  evaluation use are included here. See `data/herath_mapping.json`.

## Citation

See `.zenodo.json` for structured metadata. Please cite the accompanying paper and this
deposit's DOI.
