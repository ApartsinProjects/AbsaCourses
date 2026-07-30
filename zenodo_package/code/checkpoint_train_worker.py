"""Train and PERSIST the best-per-target BERT checkpoints for the Zenodo deposit.

Three configs (each runs in its own container via the Modal driver):

  1. synthetic_transfer      : BERT det+sent on the 9-aspect synthetic overlap
                               (the zero-shot transfer model behind Herath/EduRABSA).
  2. pretrain_finetune_herath: config-1 model, then continue-trained (fine-tuned)
                               on the real-Herath train split (best absolute Herath).
  3. top50_filtered          : BERT det+sent on the top-50%-by-audit-score synthetic
                               overlap (the faithfulness-filtering recipe product).

Each config saves detection_state_dict.pt, sentiment_state_dict.pt, the tokenizer,
thresholds.json, and checkpoint_meta.json (model_name, aspects, config, provenance,
and the checkpoint's own evaluation) so it reloads with the shipped engine code.
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import absa_model_comparison as eng  # noqa: E402
from multiseed_transfer_worker import load_real_from_mapped_jsonl, restrict_to_overlap  # noqa: E402

MODEL = "bert-base-uncased"


def _save_model(model, path):
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, path)


def _finetune_detection(model, tokenizer, train_df, aspects, cfg, epochs=3):
    """Continue-train an already-trained detection model on new data (real Herath)."""
    from torch.optim import AdamW
    crit = torch.nn.BCEWithLogitsLoss(
        pos_weight=eng.compute_pos_weight(train_df, tokenizer, aspects, cfg.max_len, cfg.device))
    opt = AdamW(model.parameters(), lr=cfg.lr)
    loader = DataLoader(eng.DetectionDataset(train_df, tokenizer, aspects, cfg.max_len),
                        batch_size=cfg.batch_size, shuffle=True)
    model.train()
    for ep in range(epochs):
        for b in loader:
            opt.zero_grad()
            logits = model(b["input_ids"].to(cfg.device), b["attention_mask"].to(cfg.device))
            loss = crit(logits, b["labels"].to(cfg.device))
            loss.backward(); opt.step()
        eng.log_event(f"[finetune det] epoch {ep+1}/{epochs} done")
    return model


def _finetune_sentiment(model, tokenizer, train_df, aspects, cfg, epochs=3):
    """Sentiment head is a masked regression model (masked_mse_loss over targets/mask)."""
    from torch.optim import AdamW
    opt = AdamW(model.parameters(), lr=cfg.lr)
    loader = DataLoader(eng.SentimentDataset(train_df, tokenizer, aspects, cfg.max_len),
                        batch_size=cfg.batch_size, shuffle=True)
    model.train()
    for ep in range(epochs):
        for b in loader:
            opt.zero_grad()
            preds = model(b["input_ids"].to(cfg.device), b["attention_mask"].to(cfg.device))
            loss = eng.masked_mse_loss(preds, b["targets"].to(cfg.device), b["mask"].to(cfg.device))
            loss.backward(); opt.step()
        eng.log_event(f"[finetune sent] epoch {ep+1}/{epochs} done")
    return model


def _load_corpus_with_row_ids(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    for i, r in enumerate(rows):
        r["row_id"] = r.get("sample_id", i)
    return eng.load_jsonl(path).assign(row_id=[str(r["row_id"]) for r in rows])


def run(config: str, seed: int, synthetic_path: Path, herath_jsonl: Path,
        scores_csv: Path, out_dir: Path) -> dict:
    cfg = eng.Config(max_len=192, batch_size=8, lr=3e-5,
                     epochs_detection=3, epochs_sentiment=3, seed=seed)
    eng.set_seed(seed)
    synthetic_df = eng.load_jsonl(synthetic_path).reset_index(drop=True)
    real_df = load_real_from_mapped_jsonl(herath_jsonl)
    aspects = sorted({a for labels in real_df["aspects"] for a in labels.keys()})

    # optional top-50% faithfulness filter (config 3): keep rows whose audit row_score
    # is at or above the corpus median. row_id in the scores CSV is the 0-indexed corpus
    # position, and load_jsonl preserves file order, so we filter by position.
    if config == "top50_filtered":
        scores = pd.read_csv(scores_csv).set_index("row_id")["row_score"]
        med = float(scores.median())
        keep_ids = set(int(i) for i in scores[scores >= med].index)
        synthetic_df = synthetic_df[[i in keep_ids for i in range(len(synthetic_df))]].reset_index(drop=True)
        eng.log_event(f"[top50] kept {len(synthetic_df)} rows at row_score >= {med:.3f}")

    synthetic_overlap = restrict_to_overlap(synthetic_df, aspects)
    train_df, calib_df, internal_test_df = eng.three_way_split(
        synthetic_overlap, cfg.split_calib, cfg.split_test, seed)

    eng.set_seed(seed)
    det_model, det_tok = eng.train_detection(MODEL, train_df, calib_df, aspects, cfg)
    eng.set_seed(seed)
    sent_model, sent_tok = eng.train_sentiment(MODEL, train_df, calib_df, aspects, cfg)

    eval_target = real_df
    if config == "pretrain_finetune_herath":
        # split real Herath into finetune-train / held-out test, finetune on train
        real_train = real_df.sample(frac=0.7, random_state=seed)
        real_test = real_df.drop(real_train.index)
        eng.set_seed(seed)
        det_model = _finetune_detection(det_model, det_tok, real_train, aspects, cfg)
        sent_model = _finetune_sentiment(sent_model, sent_tok, real_train, aspects, cfg)
        calib_df = real_train
        eval_target = real_test

    thresholds = eng.calibrate_thresholds(det_model, calib_df, det_tok, aspects, cfg)
    _, eval_summary = eng.evaluate_models(MODEL, det_model, sent_model, eval_target,
                                          det_tok, sent_tok, aspects, thresholds, cfg)
    _, internal_summary = eng.evaluate_models(MODEL, det_model, sent_model, internal_test_df,
                                              det_tok, sent_tok, aspects, thresholds, cfg)

    # ---- persist checkpoint ----
    out_dir.mkdir(parents=True, exist_ok=True)
    _save_model(det_model, out_dir / "detection_state_dict.pt")
    _save_model(sent_model, out_dir / "sentiment_state_dict.pt")
    det_tok.save_pretrained(str(out_dir / "tokenizer"))
    (out_dir / "thresholds.json").write_text(json.dumps(
        {a: float(t) for a, t in thresholds.items()}, indent=2))
    meta = {
        "config": config, "base_model": MODEL, "seed": seed, "aspects": aspects,
        "n_aspects": len(aspects), "max_len": 192, "batch_size": 8, "lr": 3e-5,
        "epochs_detection": 3, "epochs_sentiment": 3,
        "reload": "instantiate absa_model_comparison.DetectionModel(base_model, n_aspects) "
                  "and SentimentModel(base_model, n_aspects), then load_state_dict from the .pt files",
        "eval_micro_f1": float(eval_summary.get("micro_f1", float("nan"))),
        "eval_sentiment_mse": float(eval_summary.get("sentiment_mse_detected", float("nan"))),
        "internal_micro_f1": float(internal_summary.get("micro_f1", float("nan"))),
        "n_train": int(len(train_df)),
    }
    (out_dir / "checkpoint_meta.json").write_text(json.dumps(meta, indent=2))
    eng.log_event(f"[{config}] saved. eval_micro_f1={meta['eval_micro_f1']:.4f} "
                  f"internal_micro_f1={meta['internal_micro_f1']:.4f}")
    return meta


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True,
                   choices=["synthetic_transfer", "pretrain_finetune_herath", "top50_filtered"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--synthetic", default="/app/data/generated_reviews.jsonl")
    p.add_argument("--herath", default="/app/data/herath_mapped_real_reviews.jsonl")
    p.add_argument("--scores", default="/app/data/at_scale_per_row_scores.csv")
    p.add_argument("--out", default="/results/ckpt")
    a = p.parse_args()
    eng.ensure_dirs()
    run(a.config, a.seed, Path(a.synthetic), Path(a.herath), Path(a.scores), Path(a.out))


if __name__ == "__main__":
    main()
