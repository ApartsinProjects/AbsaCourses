from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI


ROOT = Path(__file__).resolve().parents[1]
BATCH_DIR = ROOT / "paper" / "batch_requests"
KEY_FILE = ROOT / ".opeai.key"


def infer_model(request_file: Path) -> str:
    first_line = request_file.read_text(encoding="utf-8").splitlines()[0]
    payload = json.loads(first_line)
    return str(payload["body"]["model"])


def load_client() -> OpenAI:
    api_key = KEY_FILE.read_text(encoding="utf-8").strip()
    return OpenAI(api_key=api_key)


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit a prepared OpenAI ABSA evaluation batch.")
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--completion-window", default="24h")
    args = parser.parse_args()

    request_file = BATCH_DIR / f"{args.prefix}_requests.jsonl"
    manifest_file = BATCH_DIR / f"{args.prefix}_manifest.csv"
    if not request_file.exists():
        raise FileNotFoundError(request_file)
    if not manifest_file.exists():
        raise FileNotFoundError(manifest_file)
    history_dir = BATCH_DIR / "submitted_batches"
    history_dir.mkdir(parents=True, exist_ok=True)

    client = load_client()
    with request_file.open("rb") as handle:
        uploaded = client.files.create(file=handle, purpose="batch")
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/responses",
        completion_window=args.completion_window,
    )
    submitted_at_utc = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload = {
        "submitted_at_utc": submitted_at_utc,
        "submitted_at_prefix": args.prefix,
        "batch_id": batch.id,
        "status": batch.status,
        "endpoint": "/v1/responses",
        "completion_window": args.completion_window,
        "input_file_id": uploaded.id,
        "output_file_id": getattr(batch, "output_file_id", None),
        "error_file_id": getattr(batch, "error_file_id", None),
        "request_file": str(request_file),
        "manifest_file": str(manifest_file),
        "model": infer_model(request_file),
        "request_count": sum(1 for _ in request_file.open("r", encoding="utf-8")),
    }
    record_path = BATCH_DIR / f"{args.prefix}_submitted_batch.json"
    history_path = history_dir / f"{args.prefix}_submitted_batch_{submitted_at_utc}.json"
    record_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    history_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
