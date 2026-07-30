# Model checkpoints (added in a later version of this record)

The best-per-target BERT checkpoints are being trained and will be added here:

- `synthetic_transfer/` : BERT detection + sentiment trained on the 9-aspect synthetic
  overlap (the zero-shot transfer model behind the Herath and EduRABSA results).
- `pretrain_finetune_herath/` : the synthetic model continue-trained (fine-tuned) on the
  real-Herath train split (best absolute Herath performance).
- `top50_filtered/` : BERT trained on the top-50%-by-audit-score synthetic overlap (the
  faithfulness-filtering recipe product).

Each checkpoint directory will contain `detection_state_dict.pt`, `sentiment_state_dict.pt`,
`tokenizer/`, `thresholds.json`, and `checkpoint_meta.json`. To reload, instantiate
`absa_model_comparison.DetectionModel(base_model, n_aspects)` and
`SentimentModel(base_model, n_aspects)` (from `code/`) and load the state dicts.
