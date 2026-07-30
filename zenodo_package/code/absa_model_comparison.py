from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, logging as hf_logging

from absa_data_io import dataset_summary, load_absa_dataset

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "paper" / "generated_datasets" / "batch_69cc15c483488190941478aa4e3a976d_generated_reviews.jsonl"
KEY_FILE = ROOT / ".opeai.key"
OUT_DIR = ROOT / "paper" / "benchmark_outputs"
RUNS_DIR = OUT_DIR / "runs"
REGISTRY_PATH = OUT_DIR / "experiment_registry.jsonl"
LOGS_DIR = OUT_DIR / "logs"
GPU_LOCK_PATH = OUT_DIR / "gpu_training.lock"
RESUME_DIR = OUT_DIR / "resume"

SENT2VAL = {"negative": -1.0, "neutral": 0.0, "positive": 1.0}
DEFAULT_APPROACHES = [
    "tfidf_two_step",
    "bert-base-uncased",
    "distilbert-base-uncased",
    "roberta-base",
    "albert-base-v2",
    "bert_joint",
    "distilbert_joint",
]
DEFAULT_OPENAI_VARIANTS = [
    "zero-shot",
    "zero-shot-glossary",
    "few-shot",
    "few-shot-diverse",
    "retrieval-few-shot",
    "two-pass",
    "aspect-by-aspect",
]
BATCH_SAFE_OPENAI_VARIANTS = [
    "zero-shot",
    "zero-shot-glossary",
    "few-shot",
    "few-shot-diverse",
    "retrieval-few-shot",
]
INTERACTIVE_OPENAI_MAX_EXAMPLES = 25
OPENAI_VARIANT_SHOTS = {
    "zero-shot": 0,
    "zero-shot-glossary": 0,
    "few-shot": 3,
    "few-shot-diverse": 5,
    "retrieval-few-shot": 5,
    "two-pass": 4,
    "aspect-by-aspect": 0,
}

ASPECT_GLOSSARY = {
    "accessibility": "Accessibility, inclusiveness, or ease of participation in materials, pace, or course processes.",
    "assessment_design": "How well assignments, projects, quizzes, and exams are structured and aligned with the course.",
    "clarity": "How understandable the teaching, explanations, instructions, or expectations are.",
    "difficulty": "How hard the intellectual content or tasks feel, independent of workload volume.",
    "exam_fairness": "Whether exams or tests feel fair, representative, and reasonable.",
    "feedback_quality": "Usefulness, specificity, clarity, or timeliness of feedback on student work.",
    "grading_transparency": "How clear, predictable, and understandable the grading criteria or rubric are.",
    "interest": "Whether the course feels engaging, motivating, or curiosity-inducing.",
    "lecturer_quality": "Perceived quality, effectiveness, or presence of the lecturer or lead instructor.",
    "materials": "Quality, usefulness, completeness, or helpfulness of readings, slides, videos, or notes.",
    "organization": "How well the course is organized, coordinated, and administratively coherent.",
    "overall_experience": "An overall judgment of the course experience that is not mainly about one narrower aspect.",
    "pacing": "Whether the course moves too fast, too slowly, or at a manageable pace.",
    "peer_interaction": "Collaboration, discussion, or interaction with classmates or group members.",
    "practical_application": "Connection of the course content to realistic, applied, or hands-on use.",
    "prerequisite_fit": "How well the course assumes the right prior preparation or background.",
    "relevance": "Whether the course feels useful or relevant to goals, specialization, or future work.",
    "support": "Helpfulness and availability of office hours, staff, forums, or other support channels.",
    "tooling_usability": "Friction or help created by LMS, submission systems, required software, or platforms.",
    "workload": "Amount of work required, including assignment load, deadlines, and time pressure.",
}

SMOKE_TEST_APPROACHES = ["tfidf_two_step"]


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
LOGGER_HANDLE = None


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    RESUME_DIR.mkdir(parents=True, exist_ok=True)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_log_path(prefix: str = "benchmark") -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    candidate = LOGS_DIR / f"{prefix}_{timestamp}.log"
    suffix = 1
    while candidate.exists():
        suffix += 1
        candidate = LOGS_DIR / f"{prefix}_{timestamp}_{suffix}.log"
    return candidate


def default_resume_path(prefix: str = "benchmark") -> Path:
    return RESUME_DIR / f"{prefix}.resume.json"


def configure_logger(log_path: Optional[str | Path]) -> Optional[Path]:
    global LOGGER_HANDLE
    target = log_path or os.environ.get("EXPERIMENT_LOG_FILE")
    if not target:
        return None
    path = Path(target)
    if not path.suffix:
        path = path.with_suffix(".log")
    path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER_HANDLE = path.open("a", encoding="utf-8")
    return path


def close_logger() -> None:
    global LOGGER_HANDLE
    if LOGGER_HANDLE is not None:
        LOGGER_HANDLE.flush()
        LOGGER_HANDLE.close()
        LOGGER_HANDLE = None


def log_event(message: str) -> None:
    line = f"[{utc_now()}] {message}"
    print(line, flush=True)
    if LOGGER_HANDLE is not None:
        LOGGER_HANDLE.write(line + "\n")
        LOGGER_HANDLE.flush()


def load_resume_state(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_resume_state(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def make_resume_signature(args: argparse.Namespace, resolved_data_path: Path) -> Dict[str, Any]:
    return {
        "script": "absa_model_comparison.py",
        "data_path": str(resolved_data_path),
        "approaches": list(args.approaches),
        "include_openai": bool(args.include_openai),
        "openai_model": args.openai_model,
        "openai_variants": list(args.openai_variants),
        "openai_test_limit": int(args.openai_test_limit),
        "epochs_detection": int(args.epochs_detection),
        "epochs_sentiment": int(args.epochs_sentiment),
        "batch_size": int(args.batch_size),
        "max_len": int(args.max_len),
        "lr": float(args.lr),
        "seed": int(args.seed),
    }


@contextmanager
def gpu_training_lock(device: torch.device, label: str, wait_seconds: int = 10, stale_seconds: int = 12 * 3600):
    if device.type != "cuda":
        yield
        return
    lock_path = GPU_LOCK_PATH
    wait_logged_at = 0.0
    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            payload = {
                "pid": os.getpid(),
                "label": label,
                "created_at_utc": utc_now(),
            }
            os.write(fd, json.dumps(payload, ensure_ascii=False).encode("utf-8"))
            os.close(fd)
            log_event(f"[{label}] acquired GPU training lock -> {lock_path}")
            break
        except FileExistsError:
            try:
                lock_payload = json.loads(lock_path.read_text(encoding="utf-8"))
            except Exception:
                lock_payload = {}
            owner_pid_raw = lock_payload.get("pid", -1)
            try:
                owner_pid = int(owner_pid_raw)
            except Exception:
                owner_pid = -1
            if owner_pid > 0 and not pid_is_alive(owner_pid):
                try:
                    lock_path.unlink()
                    log_event(f"[{label}] removed GPU lock held by dead pid {owner_pid} -> {lock_path}")
                    continue
                except FileNotFoundError:
                    continue
            try:
                age = time.time() - lock_path.stat().st_mtime
            except FileNotFoundError:
                age = 0
            if age > stale_seconds:
                try:
                    lock_path.unlink()
                    log_event(f"[{label}] removed stale GPU lock -> {lock_path}")
                    continue
                except FileNotFoundError:
                    continue
            if time.time() - wait_logged_at >= 30:
                wait_logged_at = time.time()
                log_event(f"[{label}] waiting for GPU training lock -> {lock_path}")
            time.sleep(max(wait_seconds, 2))
    try:
        yield
    finally:
        try:
            lock_payload = json.loads(lock_path.read_text(encoding="utf-8"))
        except Exception:
            lock_payload = {}
        try:
            if int(lock_payload.get("pid", -1)) == os.getpid():
                lock_path.unlink(missing_ok=True)
                log_event(f"[{label}] released GPU training lock")
        except Exception:
            pass


def make_run_dir(prefix: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    candidate = RUNS_DIR / f"{prefix}_{timestamp}"
    suffix = 1
    while candidate.exists():
        suffix += 1
        candidate = RUNS_DIR / f"{prefix}_{timestamp}_{suffix}"
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def append_registry_entry(entry: Dict[str, Any]) -> None:
    with REGISTRY_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def save_run_bundle(
    *,
    prefix: str,
    summary_df: Optional[pd.DataFrame] = None,
    per_aspect_df: Optional[pd.DataFrame] = None,
    metadata: Optional[Dict[str, Any]] = None,
    latest_summary_name: Optional[str] = None,
    latest_per_aspect_name: Optional[str] = None,
    latest_metadata_name: Optional[str] = None,
    per_approach_frames: Optional[Dict[str, pd.DataFrame]] = None,
    artifact_payloads: Optional[Dict[str, Dict[str, Any]]] = None,
    write_latest: bool = True,
) -> Path:
    run_dir = make_run_dir(prefix)
    files: Dict[str, str] = {}
    if summary_df is not None:
        run_summary_path = run_dir / "summary.csv"
        summary_df.to_csv(run_summary_path, index=False)
        files["summary"] = str(run_summary_path)
        if write_latest and latest_summary_name:
            summary_df.to_csv(OUT_DIR / latest_summary_name, index=False)
            files["latest_summary"] = str(OUT_DIR / latest_summary_name)
    if per_aspect_df is not None:
        run_per_aspect_path = run_dir / "per_aspect.csv"
        per_aspect_df.to_csv(run_per_aspect_path, index=False)
        files["per_aspect"] = str(run_per_aspect_path)
        if write_latest and latest_per_aspect_name:
            per_aspect_df.to_csv(OUT_DIR / latest_per_aspect_name, index=False)
            files["latest_per_aspect"] = str(OUT_DIR / latest_per_aspect_name)
    if metadata is not None:
        run_metadata_path = run_dir / "metadata.json"
        run_metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        files["metadata"] = str(run_metadata_path)
        if write_latest and latest_metadata_name:
            (OUT_DIR / latest_metadata_name).write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
            files["latest_metadata"] = str(OUT_DIR / latest_metadata_name)
    if per_approach_frames:
        per_approach_dir = run_dir / "per_approach"
        per_approach_dir.mkdir(parents=True, exist_ok=True)
        for approach_name, frame in per_approach_frames.items():
            safe_name = approach_name.replace("/", "__")
            frame.to_csv(per_approach_dir / f"{safe_name}_per_aspect.csv", index=False)
            if write_latest:
                frame.to_csv(OUT_DIR / f"{safe_name}_per_aspect.csv", index=False)
        files["per_approach_dir"] = str(per_approach_dir)
    if artifact_payloads:
        artifact_dir = run_dir / "artifacts"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        for approach_name, payload in artifact_payloads.items():
            safe_name = approach_name.replace("/", "__")
            for artifact_name, value in payload.items():
                target_base = artifact_dir / f"{safe_name}_{artifact_name}"
                if isinstance(value, pd.DataFrame):
                    target_path = Path(f"{target_base}.csv")
                    value.to_csv(target_path, index=False)
                elif isinstance(value, list):
                    target_path = Path(f"{target_base}.jsonl")
                    write_jsonl_records(target_path, value)
                else:
                    target_path = Path(f"{target_base}.json")
                    target_path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
            files["artifact_dir"] = str(artifact_dir)

    entry = {
        "run_dir": str(run_dir),
        "prefix": prefix,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": files,
        "approaches": list(summary_df["approach"]) if summary_df is not None and "approach" in summary_df.columns else [],
    }
    append_registry_entry(entry)
    log_event(f"Archived run bundle -> {run_dir}")
    return run_dir


def configure_console_encoding() -> None:
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")
    hf_logging.set_verbosity_error()
    disable_progress_bar = getattr(hf_logging, "disable_progress_bar", None)
    if callable(disable_progress_bar):
        disable_progress_bar()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_jsonl(filepath: Path) -> pd.DataFrame:
    return load_absa_dataset(filepath)


def resolve_data_path(data_path: str | Path) -> Path:
    path = Path(data_path)
    if path.is_absolute():
        return path
    repo_relative = (ROOT / path).resolve()
    if repo_relative.exists():
        return repo_relative
    return path.resolve()


def write_jsonl_records(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl_records(path: str | Path) -> List[Dict[str, Any]]:
    source = Path(path)
    rows: List[Dict[str, Any]] = []
    if not source.exists():
        return rows
    with source.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def probability_to_logit(prob: float) -> float:
    clipped = float(np.clip(prob, 1e-6, 1.0 - 1e-6))
    return float(np.log(clipped / (1.0 - clipped)))


def build_prediction_records(
    *,
    approach_name: str,
    eval_split: str,
    eval_df: pd.DataFrame,
    aspects: List[str],
    det_probs: np.ndarray,
    det_preds: np.ndarray,
    det_true: np.ndarray,
    sent_preds: np.ndarray,
    sent_tgt: np.ndarray,
    sent_mask: np.ndarray,
    thresholds: Dict[str, float],
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    eval_rows = eval_df.reset_index()
    for row_idx, row in eval_rows.iterrows():
        gold_map = row["aspects"] if isinstance(row["aspects"], dict) else {}
        pred_map = {
            aspect: {1.0: "positive", 0.0: "neutral", -1.0: "negative"}.get(float(np.clip(sent_preds[row_idx, aspect_idx], -1.0, 1.0)), "neutral")
            for aspect_idx, aspect in enumerate(aspects)
            if int(det_preds[row_idx, aspect_idx]) == 1
        }
        detection_probabilities = {aspect: float(det_probs[row_idx, aspect_idx]) for aspect_idx, aspect in enumerate(aspects)}
        detection_logits = {aspect: probability_to_logit(det_probs[row_idx, aspect_idx]) for aspect_idx, aspect in enumerate(aspects)}
        detection_predictions = {aspect: int(det_preds[row_idx, aspect_idx]) for aspect_idx, aspect in enumerate(aspects)}
        detection_targets = {aspect: int(det_true[row_idx, aspect_idx]) for aspect_idx, aspect in enumerate(aspects)}
        sentiment_prediction_values = {
            aspect: float(sent_preds[row_idx, aspect_idx])
            for aspect_idx, aspect in enumerate(aspects)
            if int(det_preds[row_idx, aspect_idx]) == 1 or float(sent_mask[row_idx, aspect_idx]) > 0.0
        }
        sentiment_target_values = {
            aspect: float(sent_tgt[row_idx, aspect_idx])
            for aspect_idx, aspect in enumerate(aspects)
            if float(sent_mask[row_idx, aspect_idx]) > 0.0
        }
        records.append(
            {
                "approach": approach_name,
                "eval_split": eval_split,
                "source_index": int(row["index"]),
                "text": str(row.get("text", "")),
                "gold_aspects": gold_map,
                "predicted_aspects": pred_map,
                "detection_probabilities": detection_probabilities,
                "detection_logits": detection_logits,
                "detection_predictions": detection_predictions,
                "detection_targets": detection_targets,
                "sentiment_prediction_values": sentiment_prediction_values,
                "sentiment_target_values": sentiment_target_values,
                "thresholds": thresholds,
            }
        )
    return records


def save_approach_artifact_checkpoint(
    base_dir: Path,
    stem: str,
    artifact_payload: Dict[str, Any],
) -> Dict[str, str]:
    base_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, str] = {}
    for artifact_name, value in artifact_payload.items():
        base_path = base_dir / f"{stem}.{artifact_name}"
        if isinstance(value, pd.DataFrame):
            target = Path(f"{base_path}.csv")
            value.to_csv(target, index=False)
        elif isinstance(value, list):
            target = Path(f"{base_path}.jsonl")
            write_jsonl_records(target, value)
        else:
            target = Path(f"{base_path}.json")
            target.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
        paths[artifact_name] = str(target)
    return paths


def load_approach_artifact_checkpoint(paths: Dict[str, str]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    for artifact_name, raw_path in paths.items():
        path = Path(raw_path)
        if not path.exists():
            continue
        if path.suffix == ".csv":
            payload[artifact_name] = pd.read_csv(path)
        elif path.suffix == ".jsonl":
            payload[artifact_name] = read_jsonl_records(path)
        else:
            payload[artifact_name] = json.loads(path.read_text(encoding="utf-8"))
    return payload


def discover_aspects(df: pd.DataFrame) -> List[str]:
    seen = set()
    for item in df["aspects"]:
        seen.update(item.keys())
    return sorted(seen)


def count_aspect_lengths(df: pd.DataFrame) -> Dict[int, int]:
    counts = {1: 0, 2: 0, 3: 0}
    for aspects in df["aspects"]:
        if isinstance(aspects, dict) and len(aspects) in counts:
            counts[len(aspects)] += 1
    return counts


def three_way_split(df: pd.DataFrame, calib_size: float, test_size: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    if n < 3:
        raise ValueError(f"Need at least 3 rows for train/validation/test split; got {n}.")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    counts = np.floor(np.array([1.0 - calib_size - test_size, calib_size, test_size]) * n).astype(int)
    counts = np.maximum(counts, 1)
    while counts.sum() > n:
        reducible = [idx for idx in np.argsort(-counts) if counts[idx] > 1]
        if not reducible:
            break
        counts[reducible[0]] -= 1
    if counts.sum() < n:
        counts[0] += n - counts.sum()
    train_n, calib_n, test_n = counts.tolist()
    train_idx = perm[:train_n]
    calib_idx = perm[train_n : train_n + calib_n]
    test_idx = perm[train_n + calib_n : train_n + calib_n + test_n]
    return (
        df.iloc[train_idx].reset_index(drop=True),
        df.iloc[calib_idx].reset_index(drop=True),
        df.iloc[test_idx].reset_index(drop=True),
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


class JointDataset(Dataset):
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
        targets = torch.zeros(len(self.aspects), dtype=torch.float)
        mask = torch.zeros(len(self.aspects), dtype=torch.float)
        for aspect, sentiment in row["aspects"].items():
            aspect_idx = self.a2i[aspect]
            labels[aspect_idx] = 1.0
            targets[aspect_idx] = SENT2VAL[sentiment]
            mask[aspect_idx] = 1.0
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


class JointABSAModel(nn.Module):
    def __init__(self, model_name: str, num_aspects: int):
        super().__init__()
        self.backbone = TransformerBackbone(model_name)
        self.detector = nn.Linear(self.backbone.hidden_size, num_aspects)
        self.sentiment = nn.Linear(self.backbone.hidden_size, num_aspects)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pooled = self.backbone.pooled(input_ids, attention_mask)
        return self.detector(pooled), torch.tanh(self.sentiment(pooled))


def masked_mse_loss(preds: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return ((preds - targets) ** 2 * mask).sum() / mask.sum().clamp(min=1.0)


def masked_mse_numpy(preds: np.ndarray, targets: np.ndarray, mask: np.ndarray) -> float:
    n = mask.sum()
    if n == 0:
        return float("nan")
    return float((((preds - targets) ** 2) * mask).sum() / n)


def safe_specificity(tp: int, tn: int, fp: int, fn: int) -> float:
    denom = tn + fp
    return float(tn / denom) if denom else 0.0


def safe_balanced_accuracy(tp: int, tn: int, fp: int, fn: int) -> float:
    recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
    specificity = safe_specificity(tp, tn, fp, fn)
    return float((recall + specificity) / 2.0)


def safe_mcc(tp: int, tn: int, fp: int, fn: int) -> float:
    numerator = (tp * tn) - (fp * fn)
    denom = float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    return float(numerator / denom) if denom else 0.0


def multilabel_detection_metrics(true: np.ndarray, preds: np.ndarray) -> Dict[str, float]:
    per_aspect_balanced = []
    per_aspect_specificity = []
    per_aspect_mcc = []
    for idx in range(true.shape[1]):
        yt = true[:, idx].astype(int)
        yp = preds[:, idx].astype(int)
        tp = int(((yp == 1) & (yt == 1)).sum())
        tn = int(((yp == 0) & (yt == 0)).sum())
        fp = int(((yp == 1) & (yt == 0)).sum())
        fn = int(((yp == 0) & (yt == 1)).sum())
        per_aspect_balanced.append(safe_balanced_accuracy(tp, tn, fp, fn))
        per_aspect_specificity.append(safe_specificity(tp, tn, fp, fn))
        per_aspect_mcc.append(safe_mcc(tp, tn, fp, fn))
    return {
        "label_accuracy": float((true == preds).mean()),
        "subset_accuracy": float(accuracy_score(true, preds)),
        "samples_f1": float(f1_score(true, preds, average="samples", zero_division=0)),
        "samples_jaccard": float(jaccard_score(true, preds, average="samples", zero_division=0)),
        "macro_balanced_accuracy": float(np.mean(per_aspect_balanced)),
        "macro_specificity": float(np.mean(per_aspect_specificity)),
        "macro_mcc": float(np.mean(per_aspect_mcc)),
    }


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


def safe_row_texts(df: pd.DataFrame) -> List[str]:
    return [str(text) for text in df["text"].tolist()]


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


def fit_constant_classifier(y: np.ndarray) -> Dict[str, float]:
    if len(y) == 0:
        return {"constant": 0.0}
    return {"constant": float(np.mean(y))}


def predict_constant_classifier(model: Dict[str, float], n_rows: int) -> np.ndarray:
    return np.full(n_rows, float(model.get("constant", 0.0)), dtype=np.float32)


def fit_tfidf_two_step(train_df: pd.DataFrame, aspects: List[str]) -> tuple[TfidfVectorizer, Dict[str, Any], Dict[str, Any]]:
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=5000)
    X_train = vectorizer.fit_transform(safe_row_texts(train_df))
    det_models: Dict[str, Any] = {}
    sent_models: Dict[str, Any] = {}
    for aspect in aspects:
        y_det = np.array([1 if aspect in row else 0 for row in train_df["aspects"]], dtype=int)
        if len(np.unique(y_det)) < 2:
            det_models[aspect] = fit_constant_classifier(y_det)
        else:
            clf = LogisticRegression(max_iter=1000, class_weight="balanced")
            clf.fit(X_train, y_det)
            det_models[aspect] = clf

        present_idx = [idx for idx, row in enumerate(train_df["aspects"]) if aspect in row]
        y_sent = np.array([SENT2VAL[str(train_df.iloc[idx]["aspects"][aspect])] for idx in present_idx], dtype=np.float32)
        if len(y_sent) < 2 or np.allclose(y_sent, y_sent[0] if len(y_sent) else 0.0):
            sent_models[aspect] = {"constant": float(y_sent.mean()) if len(y_sent) else 0.0}
        else:
            reg = Ridge(alpha=1.0)
            reg.fit(X_train[present_idx], y_sent)
            sent_models[aspect] = reg
    return vectorizer, det_models, sent_models


def tfidf_predict_detection_proba(model: Any, X) -> np.ndarray:
    if isinstance(model, dict) and "constant" in model:
        return np.full(X.shape[0], float(model["constant"]), dtype=np.float32)
    proba = model.predict_proba(X)
    if proba.ndim == 1:
        return proba.astype(np.float32)
    return proba[:, 1].astype(np.float32)


def tfidf_predict_sentiment(model: Any, X) -> np.ndarray:
    if isinstance(model, dict) and "constant" in model:
        return np.full(X.shape[0], float(model["constant"]), dtype=np.float32)
    return np.clip(model.predict(X).astype(np.float32), -1.0, 1.0)


def run_tfidf_two_step_approach(
    approach_name: str,
    train_df: pd.DataFrame,
    calib_df: pd.DataFrame,
    test_df: pd.DataFrame,
    aspects: List[str],
    return_artifacts: bool = False,
) -> tuple[pd.DataFrame, Dict[str, float]] | tuple[pd.DataFrame, Dict[str, float], Dict[str, Any]]:
    start = time.time()
    log_event(f"[{approach_name}] start classical two-step fit: train={len(train_df)} calib={len(calib_df)} test={len(test_df)} aspects={len(aspects)}")
    vectorizer, det_models, sent_models = fit_tfidf_two_step(train_df, aspects)
    log_event(f"[{approach_name}] vectorizer + models fit complete")
    X_calib = vectorizer.transform(safe_row_texts(calib_df))
    X_test = vectorizer.transform(safe_row_texts(test_df))

    calib_probs = np.vstack([tfidf_predict_detection_proba(det_models[aspect], X_calib) for aspect in aspects]).T
    thr_vec = []
    for aspect_idx, aspect in enumerate(aspects):
        y_true = np.array([1 if aspect in row else 0 for row in calib_df["aspects"]], dtype=int)
        y_prob = calib_probs[:, aspect_idx]
        grid = np.linspace(0.05, 0.95, 19)
        best_f1 = -1.0
        best_threshold = 0.5
        for threshold in grid:
            preds = (y_prob >= threshold).astype(int)
            score = f1_score(y_true, preds, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_threshold = float(threshold)
        thr_vec.append(best_threshold)
    thr_vec_arr = np.array(thr_vec, dtype=np.float32)
    log_event(f"[{approach_name}] calibrated {len(thr_vec_arr)} thresholds")

    det_probs = np.vstack([tfidf_predict_detection_proba(det_models[aspect], X_test) for aspect in aspects]).T
    det_true = np.vstack([[1 if aspect in row else 0 for aspect in aspects] for row in test_df["aspects"]]).astype(int)
    det_preds = (det_probs >= thr_vec_arr).astype(int)
    sent_preds = np.vstack([tfidf_predict_sentiment(sent_models[aspect], X_test) for aspect in aspects]).T
    sent_tgt = np.vstack([[SENT2VAL[row.get(aspect, "neutral")] if aspect in row else 0.0 for aspect in aspects] for row in test_df["aspects"]]).astype(np.float32)
    sent_mask = np.vstack([[1.0 if aspect in row else 0.0 for aspect in aspects] for row in test_df["aspects"]]).astype(np.float32)

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
                "accuracy": (tp + tn) / max(len(test_df), 1),
                "precision": precision_score(yt, yp, zero_division=0),
                "recall": recall_score(yt, yp, zero_division=0),
                "f1": f1_score(yt, yp, zero_division=0),
                "specificity": safe_specificity(tp, tn, fp, fn),
                "balanced_accuracy": safe_balanced_accuracy(tp, tn, fp, fn),
                "mcc": safe_mcc(tp, tn, fp, fn),
                "mse": masked_mse_numpy(sent_preds[:, idx], sent_tgt[:, idx], eff_mask),
                "threshold": thr_vec_arr[idx],
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
            }
        )

    detection_metrics = {
        "micro_precision": float(precision_score(det_true.ravel(), det_preds.ravel(), zero_division=0)),
        "micro_recall": float(recall_score(det_true.ravel(), det_preds.ravel(), zero_division=0)),
        "micro_f1": float(f1_score(det_true.ravel(), det_preds.ravel(), zero_division=0)),
        "macro_precision": float(precision_score(det_true, det_preds, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(det_true, det_preds, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(det_true, det_preds, average="macro", zero_division=0)),
    }
    detection_metrics.update(multilabel_detection_metrics(det_true, det_preds))
    detected_mask = det_preds * sent_mask
    summary = {
        "approach": approach_name,
        "micro_precision": detection_metrics["micro_precision"],
        "micro_recall": detection_metrics["micro_recall"],
        "micro_f1": detection_metrics["micro_f1"],
        "macro_precision": detection_metrics["macro_precision"],
        "macro_recall": detection_metrics["macro_recall"],
        "macro_f1": detection_metrics["macro_f1"],
        "macro_balanced_accuracy": detection_metrics["macro_balanced_accuracy"],
        "macro_specificity": detection_metrics["macro_specificity"],
        "macro_mcc": detection_metrics["macro_mcc"],
        "label_accuracy": detection_metrics["label_accuracy"],
        "subset_accuracy": detection_metrics["subset_accuracy"],
        "samples_f1": detection_metrics["samples_f1"],
        "samples_jaccard": detection_metrics["samples_jaccard"],
        "sentiment_mse_detected": masked_mse_numpy(sent_preds, sent_tgt, detected_mask),
        "elapsed_seconds": round(time.time() - start, 1),
    }
    log_event(
        f"[{approach_name}] finished classical two-step: "
        f"micro_f1={summary['micro_f1']:.4f} macro_f1={summary['macro_f1']:.4f} "
        f"sentiment_mse={summary['sentiment_mse_detected']:.4f} elapsed={summary['elapsed_seconds']:.1f}s"
    )
    if return_artifacts:
        artifact_payload = {
            "sample_predictions": build_prediction_records(
                approach_name=approach_name,
                eval_split="test",
                eval_df=test_df,
                aspects=aspects,
                det_probs=det_probs,
                det_preds=det_preds,
                det_true=det_true,
                sent_preds=sent_preds,
                sent_tgt=sent_tgt,
                sent_mask=sent_mask,
                thresholds={aspect: float(thr_vec_arr[idx]) for idx, aspect in enumerate(aspects)},
            ),
            "thresholds": {aspect: float(thr_vec_arr[idx]) for idx, aspect in enumerate(aspects)},
        }
        return pd.DataFrame(rows), summary, artifact_payload
    return pd.DataFrame(rows), summary


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
    with gpu_training_lock(cfg.device, f"{model_name}:detection"):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = DetectionModel(model_name, len(aspects)).to(cfg.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=compute_pos_weight(train_df, tokenizer, aspects, cfg.max_len, cfg.device))
        optimizer = AdamW(model.parameters(), lr=cfg.lr)

        train_loader = DataLoader(DetectionDataset(train_df, tokenizer, aspects, cfg.max_len), batch_size=cfg.batch_size, shuffle=True)
        val_loader = DataLoader(DetectionDataset(val_df, tokenizer, aspects, cfg.max_len), batch_size=cfg.batch_size, shuffle=False)
        best_score = -1.0
        best_state = None
        patience = 0
        log_event(
            f"[{model_name}] detection training start: train_rows={len(train_df)} val_rows={len(val_df)} "
            f"epochs={cfg.epochs_detection} batch_size={cfg.batch_size} device={cfg.device}"
        )

        for epoch in range(cfg.epochs_detection):
            model.train()
            epoch_start = time.time()
            batch_losses = []
            for batch in train_loader:
                optimizer.zero_grad()
                logits = model(batch["input_ids"].to(cfg.device), batch["attention_mask"].to(cfg.device))
                loss = criterion(logits, batch["labels"].to(cfg.device))
                loss.backward()
                optimizer.step()
                batch_losses.append(float(loss.detach().cpu().item()))

            val_probs, val_true = collect_detection(model, val_loader, cfg.device)
            epoch_metrics = detection_epoch_metrics(val_probs, val_true)
            score = epoch_metrics["macro_f1"]
            log_event(
                f"[{model_name}] detection epoch {epoch + 1}/{cfg.epochs_detection}: "
                f"train_loss={np.mean(batch_losses):.4f} val_micro_f1={epoch_metrics['micro_f1']:.4f} "
                f"val_macro_f1={epoch_metrics['macro_f1']:.4f} best_macro_f1={max(best_score, score):.4f} "
                f"patience={patience} epoch_seconds={time.time() - epoch_start:.1f}"
            )
            if score > best_score:
                best_score = score
                best_state = {key: value.cpu() for key, value in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= cfg.patience:
                    log_event(f"[{model_name}] detection early stop at epoch {epoch + 1}")
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        log_event(f"[{model_name}] detection training complete: best_macro_f1={best_score:.4f}")
        return model, tokenizer


def train_sentiment(model_name: str, train_df: pd.DataFrame, val_df: pd.DataFrame, aspects: List[str], cfg: Config) -> tuple[nn.Module, object]:
    with gpu_training_lock(cfg.device, f"{model_name}:sentiment"):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = SentimentModel(model_name, len(aspects)).to(cfg.device)
        optimizer = AdamW(model.parameters(), lr=cfg.lr)

        train_loader = DataLoader(SentimentDataset(train_df, tokenizer, aspects, cfg.max_len), batch_size=cfg.batch_size, shuffle=True)
        val_loader = DataLoader(SentimentDataset(val_df, tokenizer, aspects, cfg.max_len), batch_size=cfg.batch_size, shuffle=False)
        best_loss = float("inf")
        best_state = None
        patience = 0
        log_event(
            f"[{model_name}] sentiment training start: train_rows={len(train_df)} val_rows={len(val_df)} "
            f"epochs={cfg.epochs_sentiment} batch_size={cfg.batch_size} device={cfg.device}"
        )

        for epoch in range(cfg.epochs_sentiment):
            model.train()
            epoch_start = time.time()
            batch_losses = []
            for batch in train_loader:
                optimizer.zero_grad()
                preds = model(batch["input_ids"].to(cfg.device), batch["attention_mask"].to(cfg.device))
                loss = masked_mse_loss(preds, batch["targets"].to(cfg.device), batch["mask"].to(cfg.device))
                loss.backward()
                optimizer.step()
                batch_losses.append(float(loss.detach().cpu().item()))

            val_preds, val_tgt, val_mask = collect_sentiment(model, val_loader, cfg.device)
            val_loss = masked_mse_numpy(val_preds, val_tgt, val_mask)
            prospective_best = min(best_loss, val_loss)
            log_event(
                f"[{model_name}] sentiment epoch {epoch + 1}/{cfg.epochs_sentiment}: "
                f"train_loss={np.mean(batch_losses):.4f} val_mse={val_loss:.4f} "
                f"best_val_mse={prospective_best:.4f} patience={patience} "
                f"epoch_seconds={time.time() - epoch_start:.1f}"
            )
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {key: value.cpu() for key, value in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= cfg.patience:
                    log_event(f"[{model_name}] sentiment early stop at epoch {epoch + 1}")
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        log_event(f"[{model_name}] sentiment training complete: best_val_mse={best_loss:.4f}")
        return model, tokenizer


def base_model_name_for_approach(approach_name: str) -> str:
    joint_map = {
        "bert_joint": "bert-base-uncased",
        "distilbert_joint": "distilbert-base-uncased",
    }
    if approach_name in joint_map:
        return joint_map[approach_name]
    return approach_name


def train_joint_model(
    approach_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    aspects: List[str],
    cfg: Config,
) -> tuple[nn.Module, object]:
    with gpu_training_lock(cfg.device, f"{approach_name}:joint"):
        model_name = base_model_name_for_approach(approach_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = JointABSAModel(model_name, len(aspects)).to(cfg.device)
        pos_weight = compute_pos_weight(train_df, tokenizer, aspects, cfg.max_len, cfg.device)
        det_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = AdamW(model.parameters(), lr=cfg.lr)

        train_loader = DataLoader(JointDataset(train_df, tokenizer, aspects, cfg.max_len), batch_size=cfg.batch_size, shuffle=True)
        val_loader = DataLoader(JointDataset(val_df, tokenizer, aspects, cfg.max_len), batch_size=cfg.batch_size, shuffle=False)

        best_score = -1.0
        best_state = None
        patience = 0
        log_event(
            f"[{approach_name}] joint training start: train_rows={len(train_df)} val_rows={len(val_df)} "
            f"epochs={cfg.epochs_detection} batch_size={cfg.batch_size} device={cfg.device}"
        )

        for epoch in range(cfg.epochs_detection):
            model.train()
            epoch_start = time.time()
            det_losses = []
            sent_losses = []
            for batch in train_loader:
                optimizer.zero_grad()
                det_logits, sent_preds = model(batch["input_ids"].to(cfg.device), batch["attention_mask"].to(cfg.device))
                det_loss = det_criterion(det_logits, batch["labels"].to(cfg.device))
                sent_loss = masked_mse_loss(
                    sent_preds,
                    batch["targets"].to(cfg.device),
                    batch["mask"].to(cfg.device),
                )
                loss = det_loss + sent_loss
                loss.backward()
                optimizer.step()
                det_losses.append(float(det_loss.detach().cpu().item()))
                sent_losses.append(float(sent_loss.detach().cpu().item()))

            val_probs, val_true = collect_joint_detection(model, val_loader, cfg.device)
            val_sent_preds, val_sent_tgt, val_sent_mask = collect_joint_sentiment(model, val_loader, cfg.device)
            epoch_metrics = detection_epoch_metrics(val_probs, val_true)
            val_macro_f1 = epoch_metrics["macro_f1"]
            val_mse = masked_mse_numpy(val_sent_preds, val_sent_tgt, val_sent_mask)
            score = val_macro_f1 - val_mse
            prospective_best = max(best_score, score)
            log_event(
                f"[{approach_name}] joint epoch {epoch + 1}/{cfg.epochs_detection}: "
                f"train_det_loss={np.mean(det_losses):.4f} train_sent_loss={np.mean(sent_losses):.4f} "
                f"val_macro_f1={val_macro_f1:.4f} val_mse={val_mse:.4f} "
                f"selection_score={score:.4f} best_score={prospective_best:.4f} "
                f"patience={patience} epoch_seconds={time.time() - epoch_start:.1f}"
            )
            if score > best_score:
                best_score = score
                best_state = {key: value.cpu() for key, value in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= cfg.patience:
                    log_event(f"[{approach_name}] joint early stop at epoch {epoch + 1}")
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        log_event(f"[{approach_name}] joint training complete: best_score={best_score:.4f}")
        return model, tokenizer


@torch.no_grad()
def collect_joint_detection(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs, true = [], []
    for batch in loader:
        det_logits, _ = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        probs.append(torch.sigmoid(det_logits).cpu().numpy())
        true.append(batch["labels"].numpy())
    return np.vstack(probs), np.vstack(true)


@torch.no_grad()
def collect_joint_sentiment(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    preds, tgt, mask = [], [], []
    for batch in loader:
        _, sent_output = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        preds.append(sent_output.cpu().numpy())
        tgt.append(batch["targets"].numpy())
        mask.append(batch["mask"].numpy())
    return np.vstack(preds), np.vstack(tgt), np.vstack(mask)


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
    return_artifacts: bool = False,
) -> tuple[pd.DataFrame, Dict[str, float]] | tuple[pd.DataFrame, Dict[str, float], Dict[str, Any]]:
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
                "specificity": safe_specificity(tp, tn, fp, fn),
                "balanced_accuracy": safe_balanced_accuracy(tp, tn, fp, fn),
                "mcc": safe_mcc(tp, tn, fp, fn),
                "mse": masked_mse_numpy(sent_preds[:, idx], sent_tgt[:, idx], eff_mask),
                "threshold": thresholds[aspect],
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
            }
        )

    detection_metrics = {
        "micro_precision": float(precision_score(det_true.ravel(), det_preds.ravel(), zero_division=0)),
        "micro_recall": float(recall_score(det_true.ravel(), det_preds.ravel(), zero_division=0)),
        "micro_f1": float(f1_score(det_true.ravel(), det_preds.ravel(), zero_division=0)),
        "macro_precision": float(precision_score(det_true, det_preds, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(det_true, det_preds, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(det_true, det_preds, average="macro", zero_division=0)),
    }
    detection_metrics.update(multilabel_detection_metrics(det_true, det_preds))
    detected_mask = det_preds * sent_mask
    summary = {
        "approach": approach_name,
        "micro_precision": detection_metrics["micro_precision"],
        "micro_recall": detection_metrics["micro_recall"],
        "micro_f1": detection_metrics["micro_f1"],
        "macro_precision": detection_metrics["macro_precision"],
        "macro_recall": detection_metrics["macro_recall"],
        "macro_f1": detection_metrics["macro_f1"],
        "macro_balanced_accuracy": detection_metrics["macro_balanced_accuracy"],
        "macro_specificity": detection_metrics["macro_specificity"],
        "macro_mcc": detection_metrics["macro_mcc"],
        "label_accuracy": detection_metrics["label_accuracy"],
        "subset_accuracy": detection_metrics["subset_accuracy"],
        "samples_f1": detection_metrics["samples_f1"],
        "samples_jaccard": detection_metrics["samples_jaccard"],
        "sentiment_mse_detected": masked_mse_numpy(sent_preds, sent_tgt, detected_mask),
    }
    if return_artifacts:
        artifact_payload = {
            "sample_predictions": build_prediction_records(
                approach_name=approach_name,
                eval_split="test",
                eval_df=test_df,
                aspects=aspects,
                det_probs=det_probs,
                det_preds=det_preds,
                det_true=det_true,
                sent_preds=sent_preds,
                sent_tgt=sent_tgt,
                sent_mask=sent_mask,
                thresholds={aspect: float(thresholds[aspect]) for aspect in aspects},
            ),
            "thresholds": {aspect: float(thresholds[aspect]) for aspect in aspects},
        }
        return pd.DataFrame(rows), summary, artifact_payload
    return pd.DataFrame(rows), summary


def load_openai_client() -> OpenAI:
    if OpenAI is None:
        raise RuntimeError("openai package is not installed.")
    api_key = ""
    if KEY_FILE.exists():
        api_key = KEY_FILE.read_text(encoding="utf-8").strip()
    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("No OpenAI API key found in OPENAI_API_KEY or .opeai.key")
    return OpenAI(api_key=api_key)


def extract_json_block(text: str) -> Dict[str, object]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    if start == -1:
        raise ValueError(f"No JSON object found in response: {text[:400]}")
    depth = 0
    for idx in range(start, len(text)):
        if text[idx] == "{":
            depth += 1
        elif text[idx] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : idx + 1])
    raise ValueError(f"Incomplete JSON object in response: {text[:400]}")


def normalize_tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_]+", text.lower()))


def retrieve_similar_examples(
    train_df: pd.DataFrame,
    query_text: str,
    max_shots: int,
    seed: int,
) -> List[Dict[str, object]]:
    if max_shots <= 0 or train_df.empty:
        return []
    query_tokens = normalize_tokens(query_text)
    scored = []
    for idx, row in train_df.reset_index(drop=True).iterrows():
        row_tokens = normalize_tokens(row["text"])
        overlap = len(query_tokens & row_tokens)
        scored.append((overlap, idx, row))
    scored.sort(key=lambda item: (-item[0], item[1]))
    top_rows = [row for _, _, row in scored[: max_shots * 3]]
    rng = random.Random(seed)
    rng.shuffle(top_rows)
    selected = top_rows[:max_shots]
    return [{"text": row["text"], "aspects": row["aspects"]} for row in selected]


def build_openai_demonstrations(train_df: pd.DataFrame, variant: str, max_shots: int, seed: int) -> List[Dict[str, object]]:
    if max_shots <= 0:
        return []
    rng = random.Random(seed)
    pool = train_df.copy().reset_index(drop=True)
    pool["aspect_count"] = pool["aspects"].apply(lambda x: len(x) if isinstance(x, dict) else 0)
    records: List[Dict[str, object]] = []

    if variant in {"few-shot-diverse", "two-pass"}:
        for aspect_count in [1, 2, 3]:
            subset = pool[pool["aspect_count"] == aspect_count]
            if subset.empty:
                continue
            sample = subset.sample(n=1, random_state=rng.randint(0, 10_000)).iloc[0]
            records.append({"text": sample["text"], "aspects": sample["aspects"]})
        remaining = max_shots - len(records)
        if remaining > 0:
            extra = pool.sample(n=min(remaining, len(pool)), random_state=rng.randint(0, 10_000))
            for _, row in extra.iterrows():
                records.append({"text": row["text"], "aspects": row["aspects"]})
    else:
        sampled = pool.sample(n=min(max_shots, len(pool)), random_state=seed)
        for _, row in sampled.iterrows():
            records.append({"text": row["text"], "aspects": row["aspects"]})

    deduped: List[Dict[str, object]] = []
    seen = set()
    for row in records:
        key = row["text"]
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
        if len(deduped) >= max_shots:
            break
    return deduped


def aspects_dict_to_records(aspect_map: Dict[str, str]) -> List[Dict[str, str]]:
    return [{"aspect": aspect, "sentiment": sentiment} for aspect, sentiment in sorted(aspect_map.items())]


def build_aspect_glossary_text(aspects: List[str]) -> str:
    lines = []
    for aspect in aspects:
        gloss = ASPECT_GLOSSARY.get(aspect, "")
        lines.append(f"- {aspect}: {gloss}")
    return "\n".join(lines)


def build_openai_prompt(text: str, aspects: List[str], variant: str, demonstrations: List[Dict[str, object]]) -> str:
    aspect_list = ", ".join(aspects)
    prompt = (
        "You are performing aspect-based sentiment analysis on a student course review.\n"
        f"Allowed aspects: {aspect_list}.\n"
        "Return JSON only with a single key named aspects. The value must be an array of objects, where each object has "
        "exactly two keys: aspect and sentiment.\n"
        "The aspect value must be one of the allowed aspects above. The sentiment value must be one of: positive, neutral, negative.\n"
        "The output must usually contain 1 to 3 aspects; return fewer rather than guessing.\n"
        "Do not include any aspect that is not clearly present in the review text.\n"
        "Do not infer aspects from overall tone, generic course sentiment, or common course-review patterns.\n"
        "Use only the exact allowed aspect keys above. Do not invent synonyms, corrected spellings, or extra keys.\n"
        "If evidence is ambiguous, omit the aspect.\n"
        "If no allowed aspect is clearly present, return {\"aspects\": []}.\n\n"
    )
    if variant == "zero-shot":
        prompt += "Use zero-shot reasoning only; no demonstrations are provided.\n\n"
    elif variant == "zero-shot-glossary":
        prompt += (
            "Use zero-shot reasoning only; no demonstrations are provided.\n"
            "Use the aspect glossary below to separate closely related labels. Prefer the more specific aspect when the review clearly supports it, and use overall_experience only for broad course-level judgments that are not mainly about one narrower aspect.\n\n"
            "Aspect glossary:\n"
            f"{build_aspect_glossary_text(aspects)}\n\n"
        )
    elif variant == "few-shot":
        prompt += "Use the demonstrations as schema and label examples, but do not copy wording.\n\n"
    elif variant == "few-shot-diverse":
        prompt += "The demonstrations intentionally vary aspect count and style. Learn the output schema, not the wording.\n\n"
    elif variant == "retrieval-few-shot":
        prompt += "The demonstrations were retrieved because they are lexically similar to the review. Use them as label-pattern references only.\n\n"

    for idx, demo in enumerate(demonstrations, start=1):
        prompt += (
            f"Example {idx} review:\n{demo['text']}\n"
            f"Example {idx} output:\n{json.dumps({'aspects': aspects_dict_to_records(demo['aspects'])}, ensure_ascii=False)}\n\n"
        )

    prompt += f"Review:\n{text}"
    return prompt


def build_two_pass_detection_prompt(text: str, aspects: List[str], demonstrations: List[Dict[str, object]]) -> str:
    aspect_list = ", ".join(aspects)
    prompt = (
        "You are detecting which educational-review aspects are explicitly present in a student course review.\n"
        f"Allowed aspects: {aspect_list}.\n"
        "Return JSON only with a single key named aspects. The value must be an array of detected aspect names.\n"
        "The output must usually contain 1 to 3 aspects; return fewer rather than guessing.\n"
        "Only include aspects that are clearly present in the review.\n"
        "Do not infer aspects from overall tone, generic course sentiment, or common course-review patterns.\n"
        "Use only the exact allowed aspect keys above. Do not invent synonyms, corrected spellings, or extra keys.\n"
        "If evidence is ambiguous, omit the aspect.\n"
        "If no allowed aspect is clearly present, return {\"aspects\": []}.\n\n"
        "The demonstrations intentionally vary aspect count and style. Learn the output schema, not the wording.\n\n"
    )
    for idx, demo in enumerate(demonstrations, start=1):
        prompt += (
            f"Example {idx} review:\n{demo['text']}\n"
            f"Example {idx} output:\n{json.dumps({'aspects': sorted(demo['aspects'].keys())}, ensure_ascii=False)}\n\n"
        )
    prompt += f"Review:\n{text}"
    return prompt


def build_two_pass_sentiment_prompt(text: str, detected_aspects: List[str], demonstrations: List[Dict[str, object]]) -> str:
    prompt = (
        "You are assigning sentiment labels to already-detected aspects in a student course review.\n"
        "Return JSON only with a single key named aspects. The value must be an array of objects, where each object has "
        "exactly two keys: aspect and sentiment.\n"
        "Each aspect must be one of the provided known aspects. Each sentiment must be one of: positive, neutral, negative.\n"
        "Do not add extra aspects.\n\n"
    )
    for idx, demo in enumerate(demonstrations, start=1):
        filtered = {aspect: value for aspect, value in demo["aspects"].items()}
        prompt += (
            f"Example {idx} review:\n{demo['text']}\n"
            f"Example {idx} known aspects:\n{json.dumps(sorted(filtered.keys()), ensure_ascii=False)}\n"
            f"Example {idx} output:\n{json.dumps({'aspects': aspects_dict_to_records(filtered)}, ensure_ascii=False)}\n\n"
        )
    prompt += f"Known aspects:\n{json.dumps(detected_aspects, ensure_ascii=False)}\n\nReview:\n{text}"
    return prompt


def build_aspect_presence_prompt(text: str, aspect: str) -> str:
    return (
        "You are checking whether one aspect is clearly present in a student course review.\n"
        f"Aspect: {aspect}\n"
        "Return JSON only with keys present and confidence.\n"
        "Use true only if the aspect is clearly discussed in the review text.\n"
        "Do not infer the aspect from overall tone or nearby related aspects.\n\n"
        f"Review:\n{text}"
    )


def build_aspect_sentiment_prompt(text: str, aspect: str) -> str:
    return (
        "You are assigning sentiment for one already-detected aspect in a student course review.\n"
        f"Aspect: {aspect}\n"
        "Return JSON only with keys aspect and sentiment.\n"
        "Sentiment must be one of: positive, neutral, negative.\n\n"
        f"Review:\n{text}"
    )


def build_aspect_map_text_format(aspects: List[str]) -> Dict[str, object]:
    return {
        "verbosity": "low",
        "format": {
            "type": "json_schema",
            "name": "absa_aspect_map",
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "aspects": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "aspect": {"type": "string", "enum": aspects},
                                "sentiment": {"type": "string", "enum": ["positive", "neutral", "negative"]},
                            },
                            "required": ["aspect", "sentiment"],
                        },
                        "maxItems": 3,
                    }
                },
                "required": ["aspects"],
            },
            "strict": True,
        },
    }


def build_aspect_list_text_format(aspects: List[str]) -> Dict[str, object]:
    return {
        "verbosity": "low",
        "format": {
            "type": "json_schema",
            "name": "absa_detected_aspects",
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "aspects": {
                        "type": "array",
                        "items": {"type": "string", "enum": aspects},
                        "maxItems": 3,
                    }
                },
                "required": ["aspects"],
            },
            "strict": True,
        },
    }


def build_aspect_presence_text_format() -> Dict[str, object]:
    return {
        "verbosity": "low",
        "format": {
            "type": "json_schema",
            "name": "absa_aspect_presence",
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "present": {"type": "boolean"},
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                },
                "required": ["present", "confidence"],
            },
            "strict": True,
        },
    }


def build_aspect_sentiment_text_format() -> Dict[str, object]:
    return {
        "verbosity": "low",
        "format": {
            "type": "json_schema",
            "name": "absa_single_sentiment",
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "aspect": {"type": "string"},
                    "sentiment": {"type": "string", "enum": ["positive", "neutral", "negative"]},
                },
                "required": ["aspect", "sentiment"],
            },
            "strict": True,
        },
    }


def reasoning_config_for_model(model: str) -> Dict[str, str]:
    model_lower = model.lower()
    if model_lower.startswith("gpt-5.2"):
        return {"effort": "none"}
    if model_lower.startswith("gpt-5-mini") or model_lower.startswith("gpt-5-nano"):
        return {"effort": "minimal"}
    if model_lower.startswith("gpt-5"):
        return {"effort": "low"}
    return {"effort": "minimal"}


def call_openai_json(
    client: OpenAI,
    model: str,
    prompt: str,
    text_format: Dict[str, object],
    return_trace: bool = False,
) -> Dict[str, object] | tuple[Dict[str, object], Dict[str, Any]]:
    response = client.responses.create(
        model=model,
        input=prompt,
        max_output_tokens=260,
        reasoning=reasoning_config_for_model(model),
        text=text_format,
    )
    output_text = response.output_text.strip()
    parsed = extract_json_block(output_text)
    if return_trace:
        return parsed, {
            "model": model,
            "prompt": prompt,
            "text_format": text_format,
            "raw_response_text": output_text,
            "parsed_response": parsed,
            "response_id": getattr(response, "id", ""),
        }
    return parsed


def parse_aspect_map(parsed: Dict[str, object], aspects: List[str]) -> Dict[str, str]:
    raw_value = parsed.get("aspects", {}) if isinstance(parsed, dict) else {}
    if isinstance(raw_value, dict):
        return {
            aspect: str(sentiment).strip().lower()
            for aspect, sentiment in raw_value.items()
            if aspect in aspects and str(sentiment).strip().lower() in SENT2VAL
        }
    if not isinstance(raw_value, list):
        return {}
    out: Dict[str, str] = {}
    for item in raw_value:
        if not isinstance(item, dict):
            continue
        aspect = str(item.get("aspect", "")).strip()
        sentiment = str(item.get("sentiment", "")).strip().lower()
        if aspect in aspects and sentiment in SENT2VAL:
            out[aspect] = sentiment
    return out


def openai_predict_review(
    client: OpenAI,
    model: str,
    text: str,
    aspects: List[str],
    variant: str,
    train_df: pd.DataFrame,
    demonstrations: List[Dict[str, object]],
    seed: int,
    return_trace: bool = False,
) -> Dict[str, str] | tuple[Dict[str, str], List[Dict[str, Any]]]:
    call_traces: List[Dict[str, Any]] = []
    if variant == "retrieval-few-shot":
        demonstrations = retrieve_similar_examples(train_df, text, OPENAI_VARIANT_SHOTS[variant], seed)
    if variant == "two-pass":
        detection_prompt = build_two_pass_detection_prompt(text, aspects, demonstrations)
        detected_result = call_openai_json(client, model, detection_prompt, build_aspect_list_text_format(aspects), return_trace=return_trace)
        if return_trace:
            detected_raw, trace = detected_result
            call_traces.append(trace)
        else:
            detected_raw = detected_result
        detected = detected_raw.get("aspects", []) if isinstance(detected_raw, dict) else []
        if not isinstance(detected, list):
            return ({}, call_traces) if return_trace else {}
        detected_aspects = [aspect for aspect in detected if aspect in aspects]
        if not detected_aspects:
            return ({}, call_traces) if return_trace else {}
        sentiment_prompt = build_two_pass_sentiment_prompt(text, detected_aspects, demonstrations)
        sentiment_result = call_openai_json(
            client, model, sentiment_prompt, build_aspect_map_text_format(detected_aspects), return_trace=return_trace
        )
        if return_trace:
            sentiment_raw, trace = sentiment_result
            call_traces.append(trace)
        else:
            sentiment_raw = sentiment_result
        pred_map = parse_aspect_map(sentiment_raw, detected_aspects)
        return (pred_map, call_traces) if return_trace else pred_map
    if variant == "aspect-by-aspect":
        pred_map: Dict[str, str] = {}
        for aspect in aspects:
            presence_result = call_openai_json(
                client, model, build_aspect_presence_prompt(text, aspect), build_aspect_presence_text_format(), return_trace=return_trace
            )
            if return_trace:
                presence_raw, trace = presence_result
                trace["aspect"] = aspect
                call_traces.append(trace)
            else:
                presence_raw = presence_result
            present = bool(presence_raw.get("present", False)) if isinstance(presence_raw, dict) else False
            if not present:
                continue
            sentiment_result = call_openai_json(
                client, model, build_aspect_sentiment_prompt(text, aspect), build_aspect_sentiment_text_format(), return_trace=return_trace
            )
            if return_trace:
                sentiment_raw, trace = sentiment_result
                trace["aspect"] = aspect
                call_traces.append(trace)
            else:
                sentiment_raw = sentiment_result
            sentiment = str(sentiment_raw.get("sentiment", "")).strip().lower() if isinstance(sentiment_raw, dict) else ""
            if sentiment in SENT2VAL:
                pred_map[aspect] = sentiment
        return (pred_map, call_traces) if return_trace else pred_map

    prompt = build_openai_prompt(text, aspects, variant, demonstrations)
    response = call_openai_json(client, model, prompt, build_aspect_map_text_format(aspects), return_trace=return_trace)
    if return_trace:
        parsed, trace = response
        call_traces.append(trace)
    else:
        parsed = response
    pred_map = parse_aspect_map(parsed, aspects)
    return (pred_map, call_traces) if return_trace else pred_map


def evaluate_openai_prompt(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    aspects: List[str],
    model: str,
    limit: int,
    variant: str,
    seed: int,
    return_artifacts: bool = False,
) -> tuple[pd.DataFrame, Dict[str, float]] | tuple[pd.DataFrame, Dict[str, float], Dict[str, Any]]:
    client = load_openai_client()
    if limit <= 0:
        limit = len(test_df)
    eval_df = test_df.head(limit).reset_index(drop=True)
    demonstrations = build_openai_demonstrations(train_df, variant, OPENAI_VARIANT_SHOTS[variant], seed)
    approach_name = f"openai-{model}-{variant}"

    det_true = []
    det_pred = []
    sentiment_true = []
    sentiment_pred = []
    rows = []
    sample_predictions: List[Dict[str, Any]] = []
    llm_calls: List[Dict[str, Any]] = []

    eval_rows = eval_df.reset_index()
    for row_idx, row in eval_rows.iterrows():
        response = openai_predict_review(client, model, row["text"], aspects, variant, train_df, demonstrations, seed, return_trace=return_artifacts)
        if return_artifacts:
            pred_aspects, trace = response
        else:
            pred_aspects = response
            trace = []
        true_map = row["aspects"]
        y_true = np.array([1 if aspect in true_map else 0 for aspect in aspects], dtype=int)
        y_pred = np.array([1 if aspect in pred_aspects else 0 for aspect in aspects], dtype=int)
        det_true.append(y_true)
        det_pred.append(y_pred)
        for aspect in aspects:
            if aspect in pred_aspects and aspect in true_map:
                sentiment_true.append(SENT2VAL[true_map[aspect]])
                sentiment_pred.append(SENT2VAL.get(pred_aspects[aspect], 0.0))
        if return_artifacts:
            sample_predictions.append(
                {
                    "approach": approach_name,
                    "eval_split": "test",
                    "source_index": int(row["index"]),
                    "text": str(row["text"]),
                    "gold_aspects": true_map,
                    "predicted_aspects": pred_aspects,
                    "thresholds": {},
                }
            )
            llm_calls.append(
                {
                    "approach": approach_name,
                    "source_index": int(row["index"]),
                    "variant": variant,
                    "model": model,
                    "calls": trace,
                }
            )

    det_true_arr = np.vstack(det_true)
    det_pred_arr = np.vstack(det_pred)
    for idx, aspect in enumerate(aspects):
        yt = det_true_arr[:, idx]
        yp = det_pred_arr[:, idx]
        rows.append(
            {
                "approach": approach_name,
                "aspect": aspect,
                "accuracy": float((yt == yp).mean()),
                "precision": precision_score(yt, yp, zero_division=0),
                "recall": recall_score(yt, yp, zero_division=0),
                "f1": f1_score(yt, yp, zero_division=0),
                "specificity": safe_specificity(int(((yp == 1) & (yt == 1)).sum()), int(((yp == 0) & (yt == 0)).sum()), int(((yp == 1) & (yt == 0)).sum()), int(((yp == 0) & (yt == 1)).sum())),
                "balanced_accuracy": safe_balanced_accuracy(int(((yp == 1) & (yt == 1)).sum()), int(((yp == 0) & (yt == 0)).sum()), int(((yp == 1) & (yt == 0)).sum()), int(((yp == 0) & (yt == 1)).sum())),
                "mcc": safe_mcc(int(((yp == 1) & (yt == 1)).sum()), int(((yp == 0) & (yt == 0)).sum()), int(((yp == 1) & (yt == 0)).sum()), int(((yp == 0) & (yt == 1)).sum())),
                "mse": float("nan"),
                "threshold": float("nan"),
                "tp": int(((yp == 1) & (yt == 1)).sum()),
                "tn": int(((yp == 0) & (yt == 0)).sum()),
                "fp": int(((yp == 1) & (yt == 0)).sum()),
                "fn": int(((yp == 0) & (yt == 1)).sum()),
            }
        )

    detection_metrics = {
        "approach": approach_name,
        "micro_precision": float(precision_score(det_true_arr.ravel(), det_pred_arr.ravel(), zero_division=0)),
        "micro_recall": float(recall_score(det_true_arr.ravel(), det_pred_arr.ravel(), zero_division=0)),
        "micro_f1": float(f1_score(det_true_arr.ravel(), det_pred_arr.ravel(), zero_division=0)),
        "macro_precision": float(precision_score(det_true_arr, det_pred_arr, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(det_true_arr, det_pred_arr, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(det_true_arr, det_pred_arr, average="macro", zero_division=0)),
        "macro_balanced_accuracy": float("nan"),
        "macro_specificity": float("nan"),
        "macro_mcc": float("nan"),
        "label_accuracy": float("nan"),
        "subset_accuracy": float("nan"),
        "samples_f1": float("nan"),
        "samples_jaccard": float("nan"),
        "sentiment_mse_detected": float(np.mean([(p - t) ** 2 for p, t in zip(sentiment_pred, sentiment_true)])) if sentiment_true else float("nan"),
        "shots": len(demonstrations),
        "variant": variant,
    }
    detection_metrics.update(multilabel_detection_metrics(det_true_arr, det_pred_arr))
    if return_artifacts:
        artifact_payload = {
            "sample_predictions": sample_predictions,
            "llm_calls": llm_calls,
            "thresholds": {},
        }
        return pd.DataFrame(rows), detection_metrics, artifact_payload
    return pd.DataFrame(rows), detection_metrics


def run_transformer_approach(
    approach_name: str,
    train_df: pd.DataFrame,
    calib_df: pd.DataFrame,
    test_df: pd.DataFrame,
    aspects: List[str],
    cfg: Config,
    return_artifacts: bool = False,
) -> tuple[pd.DataFrame, Dict[str, float]] | tuple[pd.DataFrame, Dict[str, float], Dict[str, Any]]:
    start = time.time()
    log_event(f"[{approach_name}] transformer approach start")
    det_model, det_tokenizer = train_detection(approach_name, train_df, calib_df, aspects, cfg)
    sent_model, sent_tokenizer = train_sentiment(approach_name, train_df, calib_df, aspects, cfg)
    thresholds = calibrate_thresholds(det_model, calib_df, det_tokenizer, aspects, cfg)
    log_event(f"[{approach_name}] calibrated {len(thresholds)} thresholds")
    evaluation = evaluate_models(
        approach_name,
        det_model,
        sent_model,
        test_df,
        det_tokenizer,
        sent_tokenizer,
        aspects,
        thresholds,
        cfg,
        return_artifacts=return_artifacts,
    )
    if return_artifacts:
        per_aspect_df, summary, artifact_payload = evaluation
    else:
        per_aspect_df, summary = evaluation
    summary["elapsed_seconds"] = round(time.time() - start, 1)
    log_event(
        f"[{approach_name}] transformer approach complete: "
        f"micro_f1={summary['micro_f1']:.4f} macro_f1={summary['macro_f1']:.4f} "
        f"sentiment_mse={summary['sentiment_mse_detected']:.4f} elapsed={summary['elapsed_seconds']:.1f}s"
    )
    if return_artifacts:
        return per_aspect_df, summary, artifact_payload
    return per_aspect_df, summary


def run_joint_approach(
    approach_name: str,
    train_df: pd.DataFrame,
    calib_df: pd.DataFrame,
    test_df: pd.DataFrame,
    aspects: List[str],
    cfg: Config,
    return_artifacts: bool = False,
) -> tuple[pd.DataFrame, Dict[str, float]] | tuple[pd.DataFrame, Dict[str, float], Dict[str, Any]]:
    start = time.time()
    log_event(f"[{approach_name}] joint approach start")
    model, tokenizer = train_joint_model(approach_name, train_df, calib_df, aspects, cfg)
    thresholds = calibrate_joint_thresholds(model, calib_df, tokenizer, aspects, cfg)
    log_event(f"[{approach_name}] calibrated {len(thresholds)} joint thresholds")
    evaluation = evaluate_joint_model(approach_name, model, test_df, tokenizer, aspects, thresholds, cfg, return_artifacts=return_artifacts)
    if return_artifacts:
        per_aspect_df, summary, artifact_payload = evaluation
    else:
        per_aspect_df, summary = evaluation
    summary["elapsed_seconds"] = round(time.time() - start, 1)
    log_event(
        f"[{approach_name}] joint approach complete: "
        f"micro_f1={summary['micro_f1']:.4f} macro_f1={summary['macro_f1']:.4f} "
        f"sentiment_mse={summary['sentiment_mse_detected']:.4f} elapsed={summary['elapsed_seconds']:.1f}s"
    )
    if return_artifacts:
        return per_aspect_df, summary, artifact_payload
    return per_aspect_df, summary


def calibrate_joint_thresholds(model: nn.Module, calib_df: pd.DataFrame, tokenizer, aspects: List[str], cfg: Config) -> Dict[str, float]:
    loader = DataLoader(JointDataset(calib_df, tokenizer, aspects, cfg.max_len), batch_size=cfg.batch_size, shuffle=False)
    det_probs, det_true = collect_joint_detection(model, loader, cfg.device)
    grid = np.linspace(0.05, 0.95, 19)
    thresholds: Dict[str, float] = {}
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


def evaluate_joint_model(
    approach_name: str,
    model: nn.Module,
    test_df: pd.DataFrame,
    tokenizer,
    aspects: List[str],
    thresholds: Dict[str, float],
    cfg: Config,
    return_artifacts: bool = False,
) -> tuple[pd.DataFrame, Dict[str, float]] | tuple[pd.DataFrame, Dict[str, float], Dict[str, Any]]:
    loader = DataLoader(JointDataset(test_df, tokenizer, aspects, cfg.max_len), batch_size=cfg.batch_size, shuffle=False)
    det_probs, det_true = collect_joint_detection(model, loader, cfg.device)
    sent_preds, sent_tgt, sent_mask = collect_joint_sentiment(model, loader, cfg.device)
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
                "specificity": safe_specificity(tp, tn, fp, fn),
                "balanced_accuracy": safe_balanced_accuracy(tp, tn, fp, fn),
                "mcc": safe_mcc(tp, tn, fp, fn),
                "mse": masked_mse_numpy(sent_preds[:, idx], sent_tgt[:, idx], eff_mask),
                "threshold": thresholds[aspect],
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
            }
        )

    det_metrics = {
        "micro_precision": float(precision_score(det_true.ravel(), det_preds.ravel(), zero_division=0)),
        "micro_recall": float(recall_score(det_true.ravel(), det_preds.ravel(), zero_division=0)),
        "micro_f1": float(f1_score(det_true.ravel(), det_preds.ravel(), zero_division=0)),
        "macro_precision": float(precision_score(det_true, det_preds, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(det_true, det_preds, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(det_true, det_preds, average="macro", zero_division=0)),
    }
    det_metrics.update(multilabel_detection_metrics(det_true, det_preds))
    detected_mask = det_preds * sent_mask
    summary = {
        "approach": approach_name,
        "micro_precision": det_metrics["micro_precision"],
        "micro_recall": det_metrics["micro_recall"],
        "micro_f1": det_metrics["micro_f1"],
        "macro_precision": det_metrics["macro_precision"],
        "macro_recall": det_metrics["macro_recall"],
        "macro_f1": det_metrics["macro_f1"],
        "macro_balanced_accuracy": det_metrics["macro_balanced_accuracy"],
        "macro_specificity": det_metrics["macro_specificity"],
        "macro_mcc": det_metrics["macro_mcc"],
        "label_accuracy": det_metrics["label_accuracy"],
        "subset_accuracy": det_metrics["subset_accuracy"],
        "samples_f1": det_metrics["samples_f1"],
        "samples_jaccard": det_metrics["samples_jaccard"],
        "sentiment_mse_detected": masked_mse_numpy(sent_preds, sent_tgt, detected_mask),
    }
    if return_artifacts:
        artifact_payload = {
            "sample_predictions": build_prediction_records(
                approach_name=approach_name,
                eval_split="test",
                eval_df=test_df,
                aspects=aspects,
                det_probs=det_probs,
                det_preds=det_preds,
                det_true=det_true,
                sent_preds=sent_preds,
                sent_tgt=sent_tgt,
                sent_mask=sent_mask,
                thresholds={aspect: float(thresholds[aspect]) for aspect in aspects},
            ),
            "thresholds": {aspect: float(thresholds[aspect]) for aspect in aspects},
        }
        return pd.DataFrame(rows), summary, artifact_payload
    return pd.DataFrame(rows), summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark multiple ABSA approaches on the course review dataset.")
    parser.add_argument("--approaches", nargs="+", default=DEFAULT_APPROACHES, help="Hugging Face model names to fine-tune.")
    parser.add_argument("--include-openai", action="store_true", help="Also run a small live OpenAI prompt-based smoke baseline. Use the batch-eval scripts for larger or paper-facing GPT runs.")
    parser.add_argument("--openai-model", default="gpt-5.2", help="OpenAI model for the prompt baseline.")
    parser.add_argument("--openai-variants", nargs="+", default=DEFAULT_OPENAI_VARIANTS, choices=DEFAULT_OPENAI_VARIANTS, help="Prompting variants for the OpenAI prompt baseline.")
    parser.add_argument("--data-path", default=str(DATA_PATH), help="Path to the dataset in JSONL or CSV format.")
    parser.add_argument("--openai-test-limit", type=int, default=25, help="Number of test reviews for the live OpenAI smoke baseline. Larger batch-safe GPT runs should use openai_eval_batch_prep.py + submit_openai_eval_batch.py.")
    parser.add_argument("--dry-run", action="store_true", help="Load and inspect the dataset contract without training.")
    parser.add_argument("--dry-run-limit", type=int, default=5, help="Number of example rows to show in dry-run mode.")
    parser.add_argument("--allow-tiny-transformer", action="store_true", help="Allow transformer training on very small datasets instead of forcing the smoke path.")
    parser.add_argument("--epochs-detection", type=int, default=CFG.epochs_detection)
    parser.add_argument("--epochs-sentiment", type=int, default=CFG.epochs_sentiment)
    parser.add_argument("--batch-size", type=int, default=CFG.batch_size)
    parser.add_argument("--max-len", type=int, default=CFG.max_len)
    parser.add_argument("--lr", type=float, default=CFG.lr)
    parser.add_argument("--seed", type=int, default=CFG.seed)
    parser.add_argument("--smoke-test", action="store_true", help="Run a tiny low-cost smoke test path suitable for canary datasets.")
    parser.add_argument("--log-file", default="", help="Optional detailed experiment log path.")
    parser.add_argument("--resume-file", default="", help="Optional resume checkpoint path. Completed approaches are skipped on rerun.")
    parser.add_argument("--write-latest", dest="write_latest", action="store_true", help="Write paper-facing latest benchmark files in benchmark_outputs.")
    parser.add_argument("--no-write-latest", dest="write_latest", action="store_false", help="Archive the run bundle only without overwriting paper-facing latest benchmark files.")
    parser.set_defaults(write_latest=None)
    args = parser.parse_args()

    ensure_dirs()
    configure_console_encoding()
    log_path = configure_logger(args.log_file or default_log_path("benchmark"))
    cfg = Config(
        max_len=args.max_len,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs_detection=args.epochs_detection,
        epochs_sentiment=args.epochs_sentiment,
        seed=args.seed,
    )
    set_seed(cfg.seed)
    log_event(
        f"Benchmark start: data={args.data_path} device={cfg.device} seed={cfg.seed} "
        f"approaches={args.approaches} epochs_detection={cfg.epochs_detection} "
        f"epochs_sentiment={cfg.epochs_sentiment} batch_size={cfg.batch_size} max_len={cfg.max_len} lr={cfg.lr}"
    )
    if args.include_openai:
        batch_safe_requested = [variant for variant in args.openai_variants if variant in BATCH_SAFE_OPENAI_VARIANTS]
        if batch_safe_requested and args.openai_test_limit > INTERACTIVE_OPENAI_MAX_EXAMPLES:
            raise ValueError(
                "Interactive OpenAI evaluation is restricted to small smoke runs. "
                f"Requested {args.openai_test_limit} examples for batch-safe variants {batch_safe_requested}. "
                "Use openai_eval_batch_prep.py and submit_openai_eval_batch.py for efficient batch evaluation."
            )
    if log_path is not None:
        log_event(f"Detailed log file -> {log_path}")

    data_path = resolve_data_path(args.data_path)
    resume_path = Path(args.resume_file or os.environ.get("EXPERIMENT_RESUME_FILE") or default_resume_path("benchmark")).resolve()
    run_signature = make_resume_signature(args, data_path)
    resume_state = load_resume_state(resume_path)
    if resume_state and resume_state.get("signature") != run_signature:
        log_event(f"Resume signature mismatch at {resume_path}; starting a fresh checkpoint")
        resume_state = {}
    if not resume_state:
        resume_state = {
            "signature": run_signature,
            "created_at_utc": utc_now(),
            "updated_at_utc": utc_now(),
            "completed": {},
        }
        write_resume_state(resume_path, resume_state)
    log_event(f"Resume checkpoint -> {resume_path}")
    df = load_jsonl(data_path)
    aspects = discover_aspects(df)
    train_df, calib_df, test_df = three_way_split(df, cfg.split_calib, cfg.split_test, cfg.seed)
    log_event(
        f"Dataset loaded: rows={len(df)} train={len(train_df)} calib={len(calib_df)} test={len(test_df)} aspects={len(aspects)}"
    )

    if args.dry_run:
        summary = dataset_summary(df)
        summary.update(
            {
                "data_path": str(data_path),
                "aspect_count_distribution": count_aspect_lengths(df),
                "train_size": int(len(train_df)),
                "calib_size": int(len(calib_df)),
                "test_size": int(len(test_df)),
                "sample_rows": df.head(args.dry_run_limit).to_dict(orient="records"),
            }
        )
        save_run_bundle(
            prefix="dry_run",
            metadata=summary,
            latest_metadata_name="dry_run_summary.json",
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        close_logger()
        return

    if args.smoke_test or (len(df) < 50 and not args.allow_tiny_transformer):
        smoke_approaches = args.approaches if args.smoke_test and args.approaches != DEFAULT_APPROACHES else SMOKE_TEST_APPROACHES
        should_write_latest = args.write_latest if args.write_latest is not None else (len(smoke_approaches) > 1 or args.include_openai)
        summary_rows = []
        all_per_aspect = []
        per_approach_frames: Dict[str, pd.DataFrame] = {}
        artifact_payloads: Dict[str, Dict[str, Any]] = {}
        for approach_name in smoke_approaches:
            checkpoint_key = f"approach::{approach_name}"
            completed_entry = resume_state["completed"].get(checkpoint_key, {})
            if completed_entry:
                per_aspect_df = pd.read_csv(completed_entry["per_aspect_path"])
                summary = completed_entry["summary"]
                summary_rows.append(summary)
                all_per_aspect.append(per_aspect_df)
                per_approach_frames[approach_name] = per_aspect_df
                artifact_paths = completed_entry.get("artifact_paths", {})
                if artifact_paths:
                    artifact_payloads[approach_name] = load_approach_artifact_checkpoint(artifact_paths)
                log_event(f"[resume] restored {approach_name} from checkpoint")
                continue
            if approach_name == "tfidf_two_step":
                per_aspect_df, summary, artifact_payload = run_tfidf_two_step_approach(approach_name, train_df, calib_df, test_df, aspects, return_artifacts=True)
            elif approach_name.endswith("_joint"):
                per_aspect_df, summary, artifact_payload = run_joint_approach(approach_name, train_df, calib_df, test_df, aspects, cfg, return_artifacts=True)
            else:
                per_aspect_df, summary, artifact_payload = run_transformer_approach(approach_name, train_df, calib_df, test_df, aspects, cfg, return_artifacts=True)
            summary_rows.append(summary)
            all_per_aspect.append(per_aspect_df)
            per_approach_frames[approach_name] = per_aspect_df
            artifact_payloads[approach_name] = artifact_payload
            approach_checkpoint = resume_path.parent / f"{resume_path.stem}.{approach_name.replace('/', '__')}.per_aspect.csv"
            per_aspect_df.to_csv(approach_checkpoint, index=False)
            artifact_paths = save_approach_artifact_checkpoint(
                resume_path.parent,
                f"{resume_path.stem}.{approach_name.replace('/', '__')}",
                artifact_payload,
            )
            resume_state["completed"][checkpoint_key] = {
                "kind": "approach",
                "summary": summary,
                "per_aspect_path": str(approach_checkpoint),
                "artifact_paths": artifact_paths,
                "completed_at_utc": utc_now(),
            }
            resume_state["updated_at_utc"] = utc_now()
            write_resume_state(resume_path, resume_state)
            print(f"Completed {approach_name}: micro_f1={summary['micro_f1']:.4f}, sentiment_mse={summary['sentiment_mse_detected']:.4f}")
        summary_df = pd.DataFrame(summary_rows).sort_values("micro_f1", ascending=False).reset_index(drop=True)
        per_aspect_df = pd.concat(all_per_aspect, ignore_index=True)
        metadata = {
            "dataset_path": str(data_path),
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
                "data_path": str(data_path),
            },
            "approaches": smoke_approaches,
            "mode": "smoke-test",
            "artifact_policy": {
                "sample_predictions": True,
                "detection_probabilities": True,
                "detection_logits": True,
                "sentiment_prediction_values": True,
                "llm_calls": bool(args.include_openai),
                "checkpoint_artifact_mirror": True,
            },
            "openai_execution_policy": "interactive_smoke_only" if args.include_openai else "not_used",
            "command": sys.argv,
        }
        run_dir = save_run_bundle(
            prefix="benchmark_smoke",
            summary_df=summary_df,
            per_aspect_df=per_aspect_df,
            metadata=metadata,
            latest_summary_name="model_comparison_summary.csv",
            latest_per_aspect_name="model_comparison_per_aspect.csv",
            latest_metadata_name="model_comparison_metadata.json",
            per_approach_frames=per_approach_frames,
            artifact_payloads=artifact_payloads,
            write_latest=should_write_latest,
        )
        print(f"Saved smoke-test benchmark outputs to {OUT_DIR} and archived run to {run_dir}")
        close_logger()
        return

    summary_rows = []
    all_per_aspect = []
    per_approach_frames: Dict[str, pd.DataFrame] = {}
    artifact_payloads: Dict[str, Dict[str, Any]] = {}

    for approach_name in args.approaches:
        log_event(f"Dispatching approach -> {approach_name}")
        checkpoint_key = f"approach::{approach_name}"
        completed_entry = resume_state["completed"].get(checkpoint_key, {})
        if completed_entry:
            per_aspect_df = pd.read_csv(completed_entry["per_aspect_path"])
            summary = completed_entry["summary"]
            summary_rows.append(summary)
            all_per_aspect.append(per_aspect_df)
            per_approach_frames[approach_name] = per_aspect_df
            artifact_paths = completed_entry.get("artifact_paths", {})
            if artifact_paths:
                artifact_payloads[approach_name] = load_approach_artifact_checkpoint(artifact_paths)
            log_event(f"[resume] restored {approach_name} from checkpoint")
            continue
        if approach_name == "tfidf_two_step":
            per_aspect_df, summary, artifact_payload = run_tfidf_two_step_approach(approach_name, train_df, calib_df, test_df, aspects, return_artifacts=True)
        elif approach_name.endswith("_joint"):
            per_aspect_df, summary, artifact_payload = run_joint_approach(approach_name, train_df, calib_df, test_df, aspects, cfg, return_artifacts=True)
        else:
            per_aspect_df, summary, artifact_payload = run_transformer_approach(approach_name, train_df, calib_df, test_df, aspects, cfg, return_artifacts=True)
        summary_rows.append(summary)
        all_per_aspect.append(per_aspect_df)
        per_approach_frames[approach_name] = per_aspect_df
        artifact_payloads[approach_name] = artifact_payload
        approach_checkpoint = resume_path.parent / f"{resume_path.stem}.{approach_name.replace('/', '__')}.per_aspect.csv"
        per_aspect_df.to_csv(approach_checkpoint, index=False)
        artifact_paths = save_approach_artifact_checkpoint(
            resume_path.parent,
            f"{resume_path.stem}.{approach_name.replace('/', '__')}",
            artifact_payload,
        )
        resume_state["completed"][checkpoint_key] = {
            "kind": "approach",
            "summary": summary,
            "per_aspect_path": str(approach_checkpoint),
            "artifact_paths": artifact_paths,
            "completed_at_utc": utc_now(),
        }
        resume_state["updated_at_utc"] = utc_now()
        write_resume_state(resume_path, resume_state)
        print(f"Completed {approach_name}: micro_f1={summary['micro_f1']:.4f}, sentiment_mse={summary['sentiment_mse_detected']:.4f}")

    if args.include_openai:
        for variant in args.openai_variants:
            log_event(f"Dispatching OpenAI prompt variant -> {variant}")
            checkpoint_key = f"openai::{args.openai_model}::{variant}"
            completed_entry = resume_state["completed"].get(checkpoint_key, {})
            if completed_entry:
                per_aspect_df = pd.read_csv(completed_entry["per_aspect_path"])
                summary = completed_entry["summary"]
                summary_rows.append(summary)
                all_per_aspect.append(per_aspect_df)
                per_approach_frames[summary["approach"]] = per_aspect_df
                artifact_paths = completed_entry.get("artifact_paths", {})
                if artifact_paths:
                    artifact_payloads[summary["approach"]] = load_approach_artifact_checkpoint(artifact_paths)
                log_event(f"[resume] restored OpenAI variant {variant} from checkpoint")
                continue
            per_aspect_df, summary, artifact_payload = evaluate_openai_prompt(
                train_df,
                test_df,
                aspects,
                args.openai_model,
                args.openai_test_limit,
                variant,
                cfg.seed,
                return_artifacts=True,
            )
            summary_rows.append(summary)
            all_per_aspect.append(per_aspect_df)
            per_approach_frames[summary["approach"]] = per_aspect_df
            artifact_payloads[summary["approach"]] = artifact_payload
            approach_checkpoint = resume_path.parent / f"{resume_path.stem}.{summary['approach'].replace('/', '__')}.per_aspect.csv"
            per_aspect_df.to_csv(approach_checkpoint, index=False)
            artifact_paths = save_approach_artifact_checkpoint(
                resume_path.parent,
                f"{resume_path.stem}.{summary['approach'].replace('/', '__')}",
                artifact_payload,
            )
            resume_state["completed"][checkpoint_key] = {
                "kind": "openai_variant",
                "summary": summary,
                "per_aspect_path": str(approach_checkpoint),
                "artifact_paths": artifact_paths,
                "completed_at_utc": utc_now(),
            }
            resume_state["updated_at_utc"] = utc_now()
            write_resume_state(resume_path, resume_state)
            print(f"Completed {summary['approach']}: micro_f1={summary['micro_f1']:.4f}, sentiment_mse={summary['sentiment_mse_detected']:.4f}")

    summary_df = pd.DataFrame(summary_rows).sort_values("micro_f1", ascending=False).reset_index(drop=True)
    per_aspect_df = pd.concat(all_per_aspect, ignore_index=True)
    should_write_latest = args.write_latest if args.write_latest is not None else (len(summary_rows) > 1 or args.include_openai)

    metadata = {
        "dataset_path": str(data_path),
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
            "data_path": str(data_path),
        },
        "approaches": args.approaches + ([f"openai-{args.openai_model}-{variant}" for variant in args.openai_variants] if args.include_openai else []),
        "artifact_policy": {
            "sample_predictions": True,
            "detection_probabilities": True,
            "detection_logits": True,
            "sentiment_prediction_values": True,
            "llm_calls": bool(args.include_openai),
            "checkpoint_artifact_mirror": True,
        },
        "openai_execution_policy": "interactive_smoke_only" if args.include_openai else "not_used",
        "command": sys.argv,
    }
    run_dir = save_run_bundle(
        prefix="benchmark_full",
        summary_df=summary_df,
        per_aspect_df=per_aspect_df,
        metadata=metadata,
        latest_summary_name="model_comparison_summary.csv",
        latest_per_aspect_name="model_comparison_per_aspect.csv",
        latest_metadata_name="model_comparison_metadata.json",
        per_approach_frames=per_approach_frames,
        artifact_payloads=artifact_payloads,
        write_latest=should_write_latest,
    )
    print(f"Saved benchmark outputs to {OUT_DIR} and archived run to {run_dir}")
    resume_state["final_run_dir"] = str(run_dir)
    resume_state["finalized_at_utc"] = utc_now()
    write_resume_state(resume_path, resume_state)
    close_logger()


if __name__ == "__main__":
    main()
