from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from absa_data_io import load_absa_dataset

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "paper" / "generated_datasets" / "batch_69cc15c483488190941478aa4e3a976d_generated_reviews.jsonl"
KEY_FILE = ROOT / ".opeai.key"
OUT_DIR = ROOT / "paper" / "faithfulness_audit"
BATCH_DIR = ROOT / "paper" / "batch_requests"
INTERACTIVE_MAX_SAMPLE = 25


AUDIT_PROMPT = """You are auditing whether aspect-sentiment labels are faithful to a student course review.

Review:
{text}

Declared labels:
{labels}

Return JSON only with this schema:
{{
  "aspects": [
    {{
      "aspect": "aspect_name",
      "supported": true,
      "sentiment_match": true,
      "note": "short explanation"
    }}
  ],
  "overall_note": "one short sentence"
}}

Rules:
- "supported" means the review text clearly expresses that aspect.
- "sentiment_match" means the declared sentiment matches the review text for that aspect.
- If an aspect is not supported, set sentiment_match to false.
- Be strict and conservative.
"""


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_client() -> OpenAI:
    if OpenAI is None:
        raise RuntimeError("openai package is not available")
    api_key = None
    if KEY_FILE.exists():
        api_key = KEY_FILE.read_text(encoding="utf-8").strip()
    if not api_key:
        raise RuntimeError("No OpenAI API key found")
    return OpenAI(api_key=api_key)


def extract_json(text: str) -> Dict[str, object]:
    text = text.strip()
    start = text.find("{")
    if start == -1:
        raise ValueError(f"No JSON found in response: {text[:200]}")
    depth = 0
    for idx in range(start, len(text)):
        if text[idx] == "{":
            depth += 1
        elif text[idx] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : idx + 1])
    raise ValueError(f"Incomplete JSON in response: {text[:200]}")


def build_prompt(text: str, aspects: Dict[str, str]) -> str:
    labels = "\n".join(f"- {aspect}: {sentiment}" for aspect, sentiment in aspects.items())
    return AUDIT_PROMPT.format(text=text, labels=labels)


def build_text_format() -> Dict[str, object]:
    return {
        "verbosity": "low",
        "format": {
            "type": "json_schema",
            "name": "faithfulness_audit",
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
                                "aspect": {"type": "string"},
                                "supported": {"type": "boolean"},
                                "sentiment_match": {"type": "boolean"},
                                "note": {"type": "string"},
                            },
                            "required": ["aspect", "supported", "sentiment_match", "note"],
                        },
                    },
                    "overall_note": {"type": "string"},
                },
                "required": ["aspects", "overall_note"],
            },
            "strict": True,
        },
    }


def parse_aspects(value: object) -> Dict[str, str]:
    if isinstance(value, dict):
        return {str(k): str(v) for k, v in value.items()}
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        return json.loads(text)
    return {}


def sample_input_rows(df: pd.DataFrame, sample_size: int, seed: int, text_col: str, aspects_col: str) -> pd.DataFrame:
    sample_df = df.sample(n=min(sample_size, len(df)), random_state=seed).reset_index(drop=True).copy()
    sample_df["__text__"] = sample_df[text_col].astype(str)
    sample_df["__gold_aspects__"] = sample_df[aspects_col].apply(parse_aspects)
    return sample_df


def prepare_batch_requests(
    df: pd.DataFrame,
    model: str,
    sample_size: int,
    seed: int,
    text_col: str,
    aspects_col: str,
    prefix: str,
) -> tuple[Path, Path, Path]:
    ensure_dirs()
    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    sample_df = sample_input_rows(df, sample_size, seed, text_col, aspects_col)
    request_path = BATCH_DIR / f"{prefix}_requests.jsonl"
    manifest_path = BATCH_DIR / f"{prefix}_manifest.csv"
    metadata_path = BATCH_DIR / f"{prefix}_metadata.json"
    text_format = build_text_format()
    manifest_rows: List[Dict[str, object]] = []
    with request_path.open("w", encoding="utf-8") as handle:
        for row_id, row in sample_df.iterrows():
            custom_id = f"faithfulness_{row_id}"
            prompt = build_prompt(row["__text__"], row["__gold_aspects__"])
            request = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/responses",
                "body": {
                    "model": model,
                    "input": prompt,
                    "max_output_tokens": 800,
                    "text": text_format,
                },
            }
            handle.write(json.dumps(request, ensure_ascii=False) + "\n")
            manifest_rows.append(
                {
                    "custom_id": custom_id,
                    "row_id": int(row_id),
                    "text": row["__text__"],
                    "gold_aspects": json.dumps(row["__gold_aspects__"], ensure_ascii=False),
                    "model": model,
                }
            )
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    metadata = {
        "data_path": str(Path(DATA_PATH if not df.empty else DATA_PATH)),
        "prefix": prefix,
        "model": model,
        "sample_size_requested": int(sample_size),
        "sample_size_actual": int(len(sample_df)),
        "seed": int(seed),
        "text_col": text_col,
        "aspects_col": aspects_col,
        "batch_execution_policy": "batch_default_for_large_faithfulness_audits",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return request_path, manifest_path, metadata_path


def extract_output_text(result_row: Dict[str, object]) -> str:
    body = result_row.get("response", {}).get("body", {})
    for item in body.get("output", []):
        if item.get("type") == "message":
            for content in item.get("content", []):
                if content.get("type") == "output_text":
                    return str(content.get("text", "")).strip()
    return ""


def consume_batch_results(manifest_path: Path, results_path: Path, model: str) -> tuple[pd.DataFrame, Dict[str, object]]:
    manifest = pd.read_csv(manifest_path)
    by_custom: Dict[str, Dict[str, object]] = {}
    with results_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            by_custom[row["custom_id"]] = row

    rows: List[Dict[str, object]] = []
    response_archive: List[Dict[str, object]] = []
    for _, row in manifest.iterrows():
        gold = parse_aspects(row["gold_aspects"])
        result = by_custom.get(str(row["custom_id"]), {})
        output_text = extract_output_text(result)
        parsed: Dict[str, object] = {}
        if output_text:
            try:
                parsed = extract_json(output_text)
            except Exception:
                parsed = {}
        aspect_rows = parsed.get("aspects", []) if isinstance(parsed, dict) else []
        if not isinstance(aspect_rows, list):
            aspect_rows = []
        parsed_by_aspect = {}
        for item in aspect_rows:
            if isinstance(item, dict):
                parsed_by_aspect[str(item.get("aspect", "")).strip()] = item
        supported_all = True
        sentiment_all = True
        supported_count = 0
        sentiment_count = 0
        for aspect, sentiment in gold.items():
            item = parsed_by_aspect.get(aspect, {})
            supported = bool(item.get("supported", False))
            sentiment_match = bool(item.get("sentiment_match", False))
            note = str(item.get("note", "")).strip()
            supported_all = supported_all and supported
            sentiment_all = sentiment_all and sentiment_match
            supported_count += int(supported)
            sentiment_count += int(sentiment_match)
            rows.append(
                {
                    "row_id": int(row["row_id"]),
                    "text": row["text"],
                    "aspect": aspect,
                    "declared_sentiment": sentiment,
                    "supported": supported,
                    "sentiment_match": sentiment_match,
                    "note": note,
                    "overall_note": str(parsed.get("overall_note", "")).strip(),
                }
            )
        rows.append(
            {
                "row_id": int(row["row_id"]),
                "text": row["text"],
                "aspect": "__row_summary__",
                "declared_sentiment": "",
                "supported": supported_all,
                "sentiment_match": sentiment_all,
                "note": f"supported {supported_count}/{len(gold)}; sentiment {sentiment_count}/{len(gold)}",
                "overall_note": str(parsed.get("overall_note", "")).strip(),
            }
        )
        response_archive.append(
            {
                "custom_id": str(row["custom_id"]),
                "row_id": int(row["row_id"]),
                "text": row["text"],
                "gold_aspects": gold,
                "raw_response_text": output_text,
                "parsed_response": parsed,
                "response_status": result.get("response", {}).get("status", ""),
            }
        )
    audit_df = pd.DataFrame(rows)
    aspect_df = audit_df[audit_df["aspect"] != "__row_summary__"].copy()
    row_df = audit_df[audit_df["aspect"] == "__row_summary__"].copy()
    summary = {
        "model": model,
        "sample_size_reviews": int(len(row_df)),
        "sample_size_declared_aspects": int(len(aspect_df)),
        "aspect_supported_rate": round(float(aspect_df["supported"].mean()), 4) if not aspect_df.empty else None,
        "aspect_sentiment_match_rate": round(float(aspect_df["sentiment_match"].mean()), 4) if not aspect_df.empty else None,
        "row_full_support_rate": round(float(row_df["supported"].mean()), 4) if not row_df.empty else None,
        "row_full_sentiment_match_rate": round(float(row_df["sentiment_match"].mean()), 4) if not row_df.empty else None,
        "manifest_path": str(manifest_path.resolve()),
        "results_path": str(results_path.resolve()),
    }
    return audit_df, {"summary": summary, "response_archive": response_archive}


def run_audit(df: pd.DataFrame, model: str, sample_size: int, seed: int, text_col: str, aspects_col: str) -> tuple[pd.DataFrame, Dict[str, object]]:
    client = load_client()
    sample_df = df.sample(n=min(sample_size, len(df)), random_state=seed).reset_index(drop=True)
    rows: List[Dict[str, object]] = []

    for row_id, row in sample_df.iterrows():
        aspects = parse_aspects(row[aspects_col])
        prompt = build_prompt(row[text_col], aspects)
        response = client.responses.create(model=model, input=prompt, max_output_tokens=800)
        parsed = extract_json(response.output_text)
        aspect_rows = parsed.get("aspects", []) if isinstance(parsed, dict) else []
        if not isinstance(aspect_rows, list):
            aspect_rows = []

        declared = aspects
        supported_all = True
        sentiment_all = True
        supported_count = 0
        sentiment_count = 0

        parsed_by_aspect = {}
        for item in aspect_rows:
            if not isinstance(item, dict):
                continue
            aspect = str(item.get("aspect", "")).strip()
            parsed_by_aspect[aspect] = item

        for aspect, sentiment in declared.items():
            item = parsed_by_aspect.get(aspect, {})
            supported = bool(item.get("supported", False))
            sentiment_match = bool(item.get("sentiment_match", False))
            note = str(item.get("note", "")).strip()
            supported_all = supported_all and supported
            sentiment_all = sentiment_all and sentiment_match
            supported_count += int(supported)
            sentiment_count += int(sentiment_match)
            rows.append(
                {
                    "row_id": row_id,
                    "text": row[text_col],
                    "aspect": aspect,
                    "declared_sentiment": sentiment,
                    "supported": supported,
                    "sentiment_match": sentiment_match,
                    "note": note,
                    "overall_note": str(parsed.get("overall_note", "")).strip(),
                }
            )

        rows.append(
            {
                "row_id": row_id,
                "text": row[text_col],
                "aspect": "__row_summary__",
                "declared_sentiment": "",
                "supported": supported_all,
                "sentiment_match": sentiment_all,
                "note": f"supported {supported_count}/{len(declared)}; sentiment {sentiment_count}/{len(declared)}",
                "overall_note": str(parsed.get("overall_note", "")).strip(),
            }
        )

    audit_df = pd.DataFrame(rows)
    aspect_df = audit_df[audit_df["aspect"] != "__row_summary__"].copy()
    row_df = audit_df[audit_df["aspect"] == "__row_summary__"].copy()

    summary = {
        "model": model,
        "sample_size_reviews": int(len(row_df)),
        "sample_size_declared_aspects": int(len(aspect_df)),
        "aspect_supported_rate": round(float(aspect_df["supported"].mean()), 4) if not aspect_df.empty else None,
        "aspect_sentiment_match_rate": round(float(aspect_df["sentiment_match"].mean()), 4) if not aspect_df.empty else None,
        "row_full_support_rate": round(float(row_df["supported"].mean()), 4) if not row_df.empty else None,
        "row_full_sentiment_match_rate": round(float(row_df["sentiment_match"].mean()), 4) if not row_df.empty else None,
    }
    return audit_df, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a model-assisted label-faithfulness audit on released synthetic reviews.")
    parser.add_argument("--data-path", default=str(DATA_PATH))
    parser.add_argument("--model", default="gpt-5.2")
    parser.add_argument("--sample-size", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--aspects-col", default="aspects")
    parser.add_argument("--mode", choices=["interactive", "batch-prep", "batch-consume"], default="interactive")
    parser.add_argument("--batch-prefix", default="faithfulness_audit_batch")
    parser.add_argument("--manifest", default="")
    parser.add_argument("--results-path", default="")
    args = parser.parse_args()

    ensure_dirs()
    data_path = Path(args.data_path)
    if data_path.suffix.lower() == ".csv":
        df = pd.read_csv(data_path)
    else:
        df = load_absa_dataset(data_path)
    if args.mode == "interactive":
        if args.sample_size > INTERACTIVE_MAX_SAMPLE:
            raise ValueError(
                f"Interactive faithfulness audit is capped at {INTERACTIVE_MAX_SAMPLE} reviews. "
                "Use --mode batch-prep and Batch API submission for larger audits."
            )
        audit_df, summary = run_audit(df, args.model, args.sample_size, args.seed, args.text_col, args.aspects_col)
        stem = f"faithfulness_audit_{args.model.replace('.', '_')}_{args.sample_size}"
        audit_df.to_csv(OUT_DIR / f"{stem}_details.csv", index=False)
        (OUT_DIR / f"{stem}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return
    if args.mode == "batch-prep":
        request_path, manifest_path, metadata_path = prepare_batch_requests(
            df, args.model, args.sample_size, args.seed, args.text_col, args.aspects_col, args.batch_prefix
        )
        print(
            json.dumps(
                {
                    "request_file": str(request_path),
                    "manifest_file": str(manifest_path),
                    "metadata_file": str(metadata_path),
                    "model": args.model,
                    "sample_size": min(args.sample_size, len(df)),
                },
                indent=2,
            )
        )
        return
    manifest_path = Path(args.manifest) if args.manifest else BATCH_DIR / f"{args.batch_prefix}_manifest.csv"
    results_path = Path(args.results_path) if args.results_path else BATCH_DIR / f"{args.batch_prefix}_results.jsonl"
    audit_df, payload = consume_batch_results(manifest_path, results_path, args.model)
    summary = payload["summary"]
    stem = f"faithfulness_audit_{args.model.replace('.', '_')}_{summary['sample_size_reviews']}"
    audit_df.to_csv(OUT_DIR / f"{stem}_details.csv", index=False)
    (OUT_DIR / f"{stem}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (OUT_DIR / f"{stem}_llm_responses.jsonl").open("w", encoding="utf-8") as handle:
        for record in payload["response_archive"]:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
