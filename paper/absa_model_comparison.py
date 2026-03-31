from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "edu" / "final_student_reviews.jsonl"
KEY_FILE = ROOT / ".opeai.key"
OUT_DIR = ROOT / "paper" / "benchmark_outputs"

SENT2VAL = {"negative": -1.0, "neutral": 0.0, "positive": 1.0}
DEFAULT_APPROACHES = [
    "bert-base-uncased",
    "distilbert-base-uncased",
    "roberta-base",
    "albert-base-v2",
]
OPENAI_APPROACH_NAME = "openai-gpt-5.2-prompt"


@dataclass
class Config:
    max_len: int = 192
    batch_size: int = 8
    lr: float = 3e-5
    epochs_detection: int = 3
    epochs_sentiment: int = 3
    patience: int = 2
    split_calib: float = 0.10
    split_test: float = 0.10
    seed: int = 42
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


CFG = Config()


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_jsonl(filepath: Path) -> pd.DataFrame:
    rows = []
    with filepath.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = (obj.get("review_text") or "").strip()
            aspects = obj.get("aspects") or {}
            if not text or not isinstance(aspects, dict) or not aspects:
                continue
            valid_aspects = {
                aspect: sentiment
                for aspect, sentiment in aspects.items()
                if sentiment in SENT2VAL
            }
            if not valid_aspects:
                continue
            rows.append(
                {
                    "text": text,
                    "aspects": valid_aspects,
                    "course_name": obj.get("course_name", ""),
                    "grade": obj.get("grade", ""),
                    "style": obj.get("style", ""),
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No valid rows loaded from dataset.")
    return df


def discover_aspects(df: pd.DataFrame) -> List[str]:
    seen = set()
    for item in df["aspects"]:
        seen.update(item.keys())
    return sorted(seen)


def three_way_split(df: pd.DataFrame, calib_size: float, test_size: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    hold = calib_size + test_size
    train_df, holdout_df = train_test_split(df, test_size=hold, random_state=seed)
    calib_df, test_df = train_test_split(holdout_df, test_size=test_size / hold, random_state=seed)
    return (
        train_df.reset_index(drop=True),
        calib_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


class DetectionDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, aspects: List[str], max_len: int):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.aspects = aspects
        self.a2i = {aspect: idx for idx, aspect in enumerate(aspects)}
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.loc[idx]
        labels = torch.zeros(len(self.aspects), dtype=torch.float)
        for aspect in row["aspects"]:
            labels[self.a2i[aspect]] = 1.0
        enc = self.tokenizer(
            row["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": labels,
        }


class SentimentDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, aspects: List[str], max_len: int):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.aspects = aspects
        self.a2i = {aspect: idx for idx, aspect in enumerate(aspects)}
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.loc[idx]
        targets = torch.zeros(len(self.aspects), dtype=torch.float)
        mask = torch.zeros(len(self.aspects), dtype=torch.float)
        for aspect, sentiment in row["aspects"].items():
            targets[self.a2i[aspect]] = SENT2VAL[sentiment]
            mask[self.a2i[aspect]] = 1.0
        enc = self.tokenizer(
            row["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "targets": targets,
            "mask": mask,
        }


class TransformerBackbone(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = getattr(self.encoder.config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(self.encoder.config, "dim")
        self.hidden_size = hidden_size

    def pooled(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).expand(last_hidden.shape).float()
        masked = last_hidden * mask
        summed = masked.sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts


class DetectionModel(nn.Module):
    def __init__(self, model_name: str, num_aspects: int):
        super().__init__()
        self.backbone = TransformerBackbone(model_name)
        self.classifier = nn.Linear(self.backbone.hidden_size, num_aspects)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.backbone.pooled(input_ids, attention_mask))


class SentimentModel(nn.Module):
    def __init__(self, model_name: str, num_aspects: int):
        super().__init__()
        self.backbone = TransformerBackbone(model_name)
        self.regressor = nn.Linear(self.backbone.hidden_size, num_aspects)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.regressor(self.backbone.pooled(input_ids, attention_mask)))


def masked_mse_loss(preds: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return ((preds - targets) ** 2 * mask).sum() / mask.sum().clamp(min=1.0)


def masked_mse_numpy(preds: np.ndarray, targets: np.ndarray, mask: np.ndarray) -> float:
    n = mask.sum()
    if n == 0:
        return float("nan")
    return float((((preds - targets) ** 2) * mask).sum() / n)


def compute_pos_weight(train_df: pd.DataFrame, tokenizer, aspects: List[str], max_len: int, device: torch.device) -> torch.Tensor:
    ds = DetectionDataset(train_df, tokenizer, aspects, max_len)
    loader = DataLoader(ds, batch_size=64, shuffle=False)
    pos = np.zeros(len(aspects))
    total = 0
    for batch in loader:
        y = batch["labels"].numpy()
        pos += y.sum(axis=0)
        total += y.shape[0]
    weight = np.clip((total - pos) / np.maximum(pos, 1.0), 1.0, 50.0)
    return torch.tensor(weight, dtype=torch.float, device=device)


@torch.no_grad()
def collect_detection(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs, true = [], []
    for batch in loader:
        logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        probs.append(torch.sigmoid(logits).cpu().numpy())
        true.append(batch["labels"].numpy())
    return np.vstack(probs), np.vstack(true)


@torch.no_grad()
def collect_sentiment(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    preds, tgt, mask = [], [], []
    for batch in loader:
        output = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        preds.append(output.cpu().numpy())
        tgt.append(batch["targets"].numpy())
        mask.append(batch["mask"].numpy())
    return np.vstack(preds), np.vstack(tgt), np.vstack(mask)


def detection_epoch_metrics(probs: np.ndarray, true: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    preds = (probs >= threshold).astype(int)
    return {
        "micro_precision": float(precision_score(true.ravel(), preds.ravel(), zero_division=0)),
        "micro_recall": float(recall_score(true.ravel(), preds.ravel(), zero_division=0)),
        "micro_f1": float(f1_score(true.ravel(), preds.ravel(), zero_division=0)),
        "macro_precision": float(precision_score(true, preds, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(true, preds, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(true, preds, average="macro", zero_division=0)),
    }


def train_detection(model_name: str, train_df: pd.DataFrame, val_df: pd.DataFrame, aspects: List[str], cfg: Config) -> tuple[nn.Module, object]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = DetectionModel(model_name, len(aspects)).to(cfg.device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=compute_pos_weight(train_df, tokenizer, aspects, cfg.max_len, cfg.device))
    optimizer = AdamW(model.parameters(), lr=cfg.lr)

    train_loader = DataLoader(DetectionDataset(train_df, tokenizer, aspects, cfg.max_len), batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(DetectionDataset(val_df, tokenizer, aspects, cfg.max_len), batch_size=cfg.batch_size, shuffle=False)
    best_score = -1.0
    best_state = None
    patience = 0

    for epoch in range(cfg.epochs_detection):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            logits = model(batch["input_ids"].to(cfg.device), batch["attention_mask"].to(cfg.device))
            loss = criterion(logits, batch["labels"].to(cfg.device))
            loss.backward()
            optimizer.step()

        val_probs, val_true = collect_detection(model, val_loader, cfg.device)
        score = detection_epoch_metrics(val_probs, val_true)["macro_f1"]
        if score > best_score:
            best_score = score
            best_state = {key: value.cpu() for key, value in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, tokenizer


def train_sentiment(model_name: str, train_df: pd.DataFrame, val_df: pd.DataFrame, aspects: List[str], cfg: Config) -> tuple[nn.Module, object]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = SentimentModel(model_name, len(aspects)).to(cfg.device)
    optimizer = AdamW(model.parameters(), lr=cfg.lr)

    train_loader = DataLoader(SentimentDataset(train_df, tokenizer, aspects, cfg.max_len), batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(SentimentDataset(val_df, tokenizer, aspects, cfg.max_len), batch_size=cfg.batch_size, shuffle=False)
    best_loss = float("inf")
    best_state = None
    patience = 0

    for epoch in range(cfg.epochs_sentiment):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            preds = model(batch["input_ids"].to(cfg.device), batch["attention_mask"].to(cfg.device))
            loss = masked_mse_loss(preds, batch["targets"].to(cfg.device), batch["mask"].to(cfg.device))
            loss.backward()
            optimizer.step()

        val_preds, val_tgt, val_mask = collect_sentiment(model, val_loader, cfg.device)
        val_loss = masked_mse_numpy(val_preds, val_tgt, val_mask)
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {key: value.cpu() for key, value in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, tokenizer


def calibrate_thresholds(det_model: nn.Module, calib_df: pd.DataFrame, tokenizer, aspects: List[str], cfg: Config) -> Dict[str, float]:
    loader = DataLoader(DetectionDataset(calib_df, tokenizer, aspects, cfg.max_len), batch_size=cfg.batch_size, shuffle=False)
    det_probs, det_true = collect_detection(det_model, loader, cfg.device)
    grid = np.linspace(0.05, 0.95, 19)
    thresholds = {}
    for aspect_idx, aspect in enumerate(aspects):
        best_f1 = -1.0
        best_threshold = 0.5
        for threshold in grid:
            preds = (det_probs[:, aspect_idx] >= threshold).astype(int)
            score = f1_score(det_true[:, aspect_idx], preds, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_threshold = float(threshold)
        thresholds[aspect] = best_threshold
    return thresholds


def evaluate_models(
    approach_name: str,
    det_model: nn.Module,
    sent_model: nn.Module,
    test_df: pd.DataFrame,
    det_tokenizer,
    sent_tokenizer,
    aspects: List[str],
    thresholds: Dict[str, float],
    cfg: Config,
) -> tuple[pd.DataFrame, Dict[str, float]]:
    det_loader = DataLoader(DetectionDataset(test_df, det_tokenizer, aspects, cfg.max_len), batch_size=cfg.batch_size, shuffle=False)
    sent_loader = DataLoader(SentimentDataset(test_df, sent_tokenizer, aspects, cfg.max_len), batch_size=cfg.batch_size, shuffle=False)
    det_probs, det_true = collect_detection(det_model, det_loader, cfg.device)
    sent_preds, sent_tgt, sent_mask = collect_sentiment(sent_model, sent_loader, cfg.device)
    thr_vec = np.array([thresholds[aspect] for aspect in aspects], dtype=np.float32)
    det_preds = (det_probs >= thr_vec).astype(int)

    rows = []
    for idx, aspect in enumerate(aspects):
        yt = det_true[:, idx]
        yp = det_preds[:, idx]
        tp = int((yp * yt).sum())
        tn = int(((1 - yp) * (1 - yt)).sum())
        fp = int((yp * (1 - yt)).sum())
        fn = int(((1 - yp) * yt).sum())
        eff_mask = yp * sent_mask[:, idx]
        rows.append(
            {
                "approach": approach_name,
                "aspect": aspect,
                "accuracy": (tp + tn) / len(test_df),
                "precision": precision_score(yt, yp, zero_division=0),
                "recall": recall_score(yt, yp, zero_division=0),
                "f1": f1_score(yt, yp, zero_division=0),
                "mse": masked_mse_numpy(sent_preds[:, idx], sent_tgt[:, idx], eff_mask),
                "threshold": thresholds[aspect],
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
            }
        )

    detection_metrics = detection_epoch_metrics(det_probs, det_true, threshold=0.5)
    detected_mask = det_preds * sent_mask
    summary = {
        "approach": approach_name,
        "micro_precision": detection_metrics["micro_precision"],
        "micro_recall": detection_metrics["micro_recall"],
        "micro_f1": detection_metrics["micro_f1"],
        "macro_precision": detection_metrics["macro_precision"],
        "macro_recall": detection_metrics["macro_recall"],
        "macro_f1": detection_metrics["macro_f1"],
        "sentiment_mse_detected": masked_mse_numpy(sent_preds, sent_tgt, detected_mask),
    }
    return pd.DataFrame(rows), summary


def load_openai_client() -> OpenAI:
    if OpenAI is None:
        raise RuntimeError("openai package is not installed.")
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key and KEY_FILE.exists():
        api_key = KEY_FILE.read_text(encoding="utf-8").strip()
    if not api_key:
        raise RuntimeError("No OpenAI API key found in OPENAI_API_KEY or .opeai.key")
    return OpenAI(api_key=api_key)


def openai_predict_review(client: OpenAI, model: str, text: str, aspects: List[str]) -> Dict[str, str]:
    aspect_list = ", ".join(aspects)
    prompt = (
        "You are performing aspect-based sentiment analysis on a student course review.\n"
        f"Allowed aspects: {aspect_list}.\n"
        "Return JSON only with a single key named aspects. The value must be an object mapping only the detected aspects "
        "to one of: positive, neutral, negative.\n"
        "Do not include any aspect that is not clearly present.\n\n"
        f"Review:\n{text}"
    )
    response = client.responses.create(model=model, input=prompt, max_output_tokens=250)
    output_text = response.output_text.strip()
    match = output_text[output_text.find("{") : output_text.rfind("}") + 1]
    parsed = json.loads(match)
    return parsed.get("aspects", {})


def evaluate_openai_prompt(
    test_df: pd.DataFrame,
    aspects: List[str],
    model: str,
    limit: int,
) -> tuple[pd.DataFrame, Dict[str, float]]:
    client = load_openai_client()
    if limit <= 0:
        limit = len(test_df)
    eval_df = test_df.head(limit).reset_index(drop=True)

    det_true = []
    det_pred = []
    sentiment_true = []
    sentiment_pred = []
    rows = []

    for _, row in eval_df.iterrows():
        pred_aspects = openai_predict_review(client, model, row["text"], aspects)
        true_map = row["aspects"]
        y_true = np.array([1 if aspect in true_map else 0 for aspect in aspects], dtype=int)
        y_pred = np.array([1 if aspect in pred_aspects else 0 for aspect in aspects], dtype=int)
        det_true.append(y_true)
        det_pred.append(y_pred)
        for aspect in aspects:
            if aspect in pred_aspects and aspect in true_map:
                sentiment_true.append(SENT2VAL[true_map[aspect]])
                sentiment_pred.append(SENT2VAL.get(pred_aspects[aspect], 0.0))

    det_true_arr = np.vstack(det_true)
    det_pred_arr = np.vstack(det_pred)
    for idx, aspect in enumerate(aspects):
        yt = det_true_arr[:, idx]
        yp = det_pred_arr[:, idx]
        rows.append(
            {
                "approach": OPENAI_APPROACH_NAME,
                "aspect": aspect,
                "accuracy": float((yt == yp).mean()),
                "precision": precision_score(yt, yp, zero_division=0),
                "recall": recall_score(yt, yp, zero_division=0),
                "f1": f1_score(yt, yp, zero_division=0),
                "mse": float("nan"),
                "threshold": float("nan"),
                "tp": int(((yp == 1) & (yt == 1)).sum()),
                "tn": int(((yp == 0) & (yt == 0)).sum()),
                "fp": int(((yp == 1) & (yt == 0)).sum()),
                "fn": int(((yp == 0) & (yt == 1)).sum()),
            }
        )

    summary = {
        "approach": OPENAI_APPROACH_NAME,
        "micro_precision": float(precision_score(det_true_arr.ravel(), det_pred_arr.ravel(), zero_division=0)),
        "micro_recall": float(recall_score(det_true_arr.ravel(), det_pred_arr.ravel(), zero_division=0)),
        "micro_f1": float(f1_score(det_true_arr.ravel(), det_pred_arr.ravel(), zero_division=0)),
        "macro_precision": float(precision_score(det_true_arr, det_pred_arr, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(det_true_arr, det_pred_arr, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(det_true_arr, det_pred_arr, average="macro", zero_division=0)),
        "sentiment_mse_detected": float(np.mean([(p - t) ** 2 for p, t in zip(sentiment_pred, sentiment_true)])) if sentiment_true else float("nan"),
    }
    return pd.DataFrame(rows), summary


def run_transformer_approach(approach_name: str, train_df: pd.DataFrame, calib_df: pd.DataFrame, test_df: pd.DataFrame, aspects: List[str], cfg: Config) -> tuple[pd.DataFrame, Dict[str, float]]:
    start = time.time()
    det_model, det_tokenizer = train_detection(approach_name, train_df, calib_df, aspects, cfg)
    sent_model, sent_tokenizer = train_sentiment(approach_name, train_df, calib_df, aspects, cfg)
    thresholds = calibrate_thresholds(det_model, calib_df, det_tokenizer, aspects, cfg)
    per_aspect_df, summary = evaluate_models(
        approach_name,
        det_model,
        sent_model,
        test_df,
        det_tokenizer,
        sent_tokenizer,
        aspects,
        thresholds,
        cfg,
    )
    summary["elapsed_seconds"] = round(time.time() - start, 1)
    return per_aspect_df, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark multiple ABSA approaches on the course review dataset.")
    parser.add_argument("--approaches", nargs="+", default=DEFAULT_APPROACHES, help="Hugging Face model names to fine-tune.")
    parser.add_argument("--include-openai", action="store_true", help="Also run an OpenAI prompt-based evaluation baseline.")
    parser.add_argument("--openai-model", default="gpt-5.2", help="OpenAI model for the prompt baseline.")
    parser.add_argument("--openai-test-limit", type=int, default=60, help="Number of test reviews for the OpenAI prompt baseline.")
    parser.add_argument("--epochs-detection", type=int, default=CFG.epochs_detection)
    parser.add_argument("--epochs-sentiment", type=int, default=CFG.epochs_sentiment)
    parser.add_argument("--batch-size", type=int, default=CFG.batch_size)
    parser.add_argument("--max-len", type=int, default=CFG.max_len)
    parser.add_argument("--lr", type=float, default=CFG.lr)
    parser.add_argument("--seed", type=int, default=CFG.seed)
    args = parser.parse_args()

    ensure_dirs()
    cfg = Config(
        max_len=args.max_len,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs_detection=args.epochs_detection,
        epochs_sentiment=args.epochs_sentiment,
        seed=args.seed,
    )
    set_seed(cfg.seed)

    df = load_jsonl(DATA_PATH)
    aspects = discover_aspects(df)
    train_df, calib_df, test_df = three_way_split(df, cfg.split_calib, cfg.split_test, cfg.seed)

    summary_rows = []
    all_per_aspect = []

    for approach_name in args.approaches:
        per_aspect_df, summary = run_transformer_approach(approach_name, train_df, calib_df, test_df, aspects, cfg)
        summary_rows.append(summary)
        all_per_aspect.append(per_aspect_df)
        safe_name = approach_name.replace("/", "__")
        per_aspect_df.to_csv(OUT_DIR / f"{safe_name}_per_aspect.csv", index=False)
        print(f"Completed {approach_name}: micro_f1={summary['micro_f1']:.4f}, sentiment_mse={summary['sentiment_mse_detected']:.4f}")

    if args.include_openai:
        per_aspect_df, summary = evaluate_openai_prompt(test_df, aspects, args.openai_model, args.openai_test_limit)
        summary_rows.append(summary)
        all_per_aspect.append(per_aspect_df)
        per_aspect_df.to_csv(OUT_DIR / f"{OPENAI_APPROACH_NAME}_per_aspect.csv", index=False)
        print(f"Completed {OPENAI_APPROACH_NAME}: micro_f1={summary['micro_f1']:.4f}, sentiment_mse={summary['sentiment_mse_detected']:.4f}")

    summary_df = pd.DataFrame(summary_rows).sort_values("micro_f1", ascending=False).reset_index(drop=True)
    per_aspect_df = pd.concat(all_per_aspect, ignore_index=True)
    summary_df.to_csv(OUT_DIR / "model_comparison_summary.csv", index=False)
    per_aspect_df.to_csv(OUT_DIR / "model_comparison_per_aspect.csv", index=False)

    metadata = {
        "dataset_path": str(DATA_PATH),
        "n_rows": int(len(df)),
        "split_sizes": {"train": int(len(train_df)), "calib": int(len(calib_df)), "test": int(len(test_df))},
        "aspects": aspects,
        "config": {
            "max_len": cfg.max_len,
            "batch_size": cfg.batch_size,
            "lr": cfg.lr,
            "epochs_detection": cfg.epochs_detection,
            "epochs_sentiment": cfg.epochs_sentiment,
            "seed": cfg.seed,
            "device": str(cfg.device),
        },
        "approaches": args.approaches + ([OPENAI_APPROACH_NAME] if args.include_openai else []),
    }
    (OUT_DIR / "model_comparison_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved benchmark outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
