from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

SENTIMENT_VALUES = {"negative", "neutral", "positive"}
TEXT_COLUMNS = ("text", "review_text", "review", "content", "sentence")
TARGET_COLUMNS = ("aspects", "target_attributes", "targets", "labels")
NUANCE_COLUMNS = ("nuance_attributes", "nuances", "attributes", "context_attributes")


def _parse_jsonish(value: Any) -> Any:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (dict, list, tuple)):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if (text.startswith("{") and text.endswith("}")) or (text.startswith("[") and text.endswith("]")):
            try:
                return json.loads(text)
            except Exception:
                pass
        if "|" in text and ":" in text:
            pairs: Dict[str, str] = {}
            for chunk in text.split("|"):
                chunk = chunk.strip()
                if ":" not in chunk:
                    continue
                key, raw_value = chunk.split(":", 1)
                key = key.strip()
                raw_value = raw_value.strip()
                if key and raw_value:
                    pairs[key] = raw_value
            if pairs:
                return pairs
        return text
    return value


def _first_existing(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    column_set = {str(c).lower(): c for c in columns}
    for candidate in candidates:
        if candidate.lower() in column_set:
            return column_set[candidate.lower()]
    return None


def normalize_text(row: pd.Series) -> str:
    column = _first_existing(row.index, TEXT_COLUMNS)
    if column is None:
        raise KeyError(f"Could not find a text column in {list(row.index)}")
    return str(row[column]).strip()


def normalize_target_attributes(row: pd.Series) -> Dict[str, str]:
    column = _first_existing(row.index, TARGET_COLUMNS)
    if column is None:
        return {}
    payload = _parse_jsonish(row[column])
    if isinstance(payload, dict):
        cleaned: Dict[str, str] = {}
        for key, value in payload.items():
            if value is None:
                continue
            cleaned[str(key).strip()] = str(value).strip().lower()
        return {k: v for k, v in cleaned.items() if k and v in SENTIMENT_VALUES}
    if isinstance(payload, list):
        cleaned = {}
        for item in payload:
            if isinstance(item, dict):
                aspect = item.get("aspect") or item.get("name") or item.get("label")
                sentiment = item.get("sentiment") or item.get("polarity") or item.get("value")
                if aspect and sentiment:
                    cleaned[str(aspect).strip()] = str(sentiment).strip().lower()
        return {k: v for k, v in cleaned.items() if v in SENTIMENT_VALUES}
    return {}


def normalize_nuance_attributes(row: pd.Series) -> Dict[str, str]:
    column = _first_existing(row.index, NUANCE_COLUMNS)
    if column is None:
        return {}
    payload = _parse_jsonish(row[column])
    if isinstance(payload, dict):
        cleaned: Dict[str, str] = {}
        for key, value in payload.items():
            if value is None:
                continue
            cleaned[str(key).strip()] = str(value).strip()
        return {k: v for k, v in cleaned.items() if k and v}
    if isinstance(payload, list):
        return {f"item_{idx}": str(item).strip() for idx, item in enumerate(payload) if str(item).strip()}
    if isinstance(payload, str) and payload:
        return {"value": payload}
    return {}


def infer_dataset_kind(df: pd.DataFrame) -> str:
    columns = {str(column).lower() for column in df.columns}
    if "review_text" in columns and "aspects" in columns:
        return "legacy_jsonl"
    if "text" in columns and ("target_attributes" in columns or "aspects" in columns):
        return "rich_tabular"
    return "unknown"


def load_absa_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() == ".csv":
        raw_df = pd.read_csv(path)
    else:
        records: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        raw_df = pd.DataFrame(records)

    rows: List[Dict[str, Any]] = []
    for _, row in raw_df.iterrows():
        text = normalize_text(row)
        target_attributes = normalize_target_attributes(row)
        if not target_attributes and isinstance(row.get("aspects"), dict):
            target_attributes = {
                str(key).strip(): str(value).strip().lower()
                for key, value in row.get("aspects", {}).items()
                if str(value).strip().lower() in SENTIMENT_VALUES
            }
        if not text or not target_attributes:
            continue
        rows.append(
            {
                "text": text,
                "aspects": target_attributes,
                "target_attributes": target_attributes,
                "nuance_attributes": normalize_nuance_attributes(row),
                "course_name": str(row.get("course_name", row.get("course_title", "")) or ""),
                "grade": str(row.get("grade", row.get("grade_band", "")) or ""),
                "style": str(row.get("style", row.get("linguistic_style", "")) or ""),
                "source_path": str(path),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No valid rows loaded from {path}")
    return df


def dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
    aspect_counts = df["aspects"].apply(lambda x: len(x) if isinstance(x, dict) else 0)
    return {
        "n_rows": int(len(df)),
        "n_text_chars_mean": round(float(df["text"].str.len().mean()), 2),
        "n_text_words_mean": round(float(df["text"].str.split().apply(len).mean()), 2),
        "n_aspects_mean": round(float(aspect_counts.mean()), 2),
        "n_aspects_min": int(aspect_counts.min()),
        "n_aspects_max": int(aspect_counts.max()),
        "columns": list(df.columns),
        "inferred_kind": infer_dataset_kind(df),
        "aspect_inventory": sorted({aspect for item in df["aspects"] if isinstance(item, dict) for aspect in item.keys()}),
    }
