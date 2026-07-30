# Model checkpoints (best per target)

Three BERT checkpoints, distributed in the archive `course_absa_checkpoints.zip`
(uploaded alongside the main package because of their size). Each was trained with
`code/checkpoint_train_worker.py` on an NVIDIA A10G.

| Checkpoint | What it is | Own evaluation (single seed 42) |
|------------|-----------|----------------------------------|
| `synthetic_transfer/` | BERT detection + sentiment trained on the 9-aspect synthetic overlap; the zero-shot transfer model behind the Herath and EduRABSA results. | Herath micro-F1 0.309, internal 9-aspect micro-F1 0.399 (five-seed mean transfer is 0.402, see the paper). |
| `pretrain_finetune_herath/` | The synthetic model continue-trained (fine-tuned) on a real-Herath train split; the best absolute Herath model. | Held-out real-Herath micro-F1 0.80. |
| `top50_filtered/` | BERT trained on the top-50%-by-audit-score synthetic overlap (the faithfulness-filtering recipe product). | Herath micro-F1 0.340, internal 9-aspect micro-F1 0.481. |

The single-seed numbers here are each checkpoint's own evaluation and are within the
run-to-run variance of this small-overlap setting; the paper reports five-seed means.

## Files per checkpoint

```
<checkpoint>/
  detection_state_dict.pt     torch state_dict for absa_model_comparison.DetectionModel
  sentiment_state_dict.pt     torch state_dict for absa_model_comparison.SentimentModel
  tokenizer/                  HuggingFace tokenizer (bert-base-uncased)
  thresholds.json             per-aspect detection thresholds calibrated on the calib split
  checkpoint_meta.json        base model, aspects, config, provenance, achieved metrics
```

## Reloading

```python
import torch, json
from absa_model_comparison import DetectionModel, SentimentModel   # from code/
meta = json.load(open("synthetic_transfer/checkpoint_meta.json"))
det = DetectionModel(meta["base_model"], meta["n_aspects"])
det.load_state_dict(torch.load("synthetic_transfer/detection_state_dict.pt", map_location="cpu"))
sent = SentimentModel(meta["base_model"], meta["n_aspects"])
sent.load_state_dict(torch.load("synthetic_transfer/sentiment_state_dict.pt", map_location="cpu"))
```
