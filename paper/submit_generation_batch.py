from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI


ROOT = Path(__file__).resolve().parents[1]
KEY_FILE = ROOT / ".opeai.key"
BATCH_DIR = ROOT / "paper" / "batch_requests"
METADATA_PATH = ROOT / "paper" / "generation_protocol" / "final_realism_prompt_metadata.json"


def load_client() -> OpenAI:
    api_key = KEY_FILE.read_text(encoding="utf-8").strip()
    return OpenAI(api_key=api_key)


def infer_model(request_file: Path) -> str:
    first_line = request_file.read_text(encoding="utf-8").splitlines()[0]
    payload = json.loads(first_line)
    return payload["body"]["model"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit a prepared generation request file to OpenAI Batch.")
    parser.add_argument("--generation-prefix", default="dataset_generation_10k")
    parser.add_argument("--completion-window", default="24h")
    parser.add_argument("--consume-when-complete", action="store_true")
    parser.add_argument("--run-benchmark-on-completion", action="store_true")
    args = parser.parse_args()

    request_file = BATCH_DIR / f"{args.generation_prefix}_requests.jsonl"
    manifest_file = BATCH_DIR / f"{args.generation_prefix}_manifest.csv"
    history_dir = BATCH_DIR / "submitted_batches"
    history_dir.mkdir(parents=True, exist_ok=True)
    prompt_metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8")) if METADATA_PATH.exists() else {}

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
        "dataset_size": sum(1 for _ in request_file.open("r", encoding="utf-8")),
        "prompt_run_id": prompt_metadata.get("selected_prompt_run_id"),
        "prompt_cycle_id": prompt_metadata.get("selected_prompt_cycle_id"),
        "selected_prompt_state_reference": prompt_metadata.get("selected_prompt_state_reference"),
        "consume_when_complete": bool(args.consume_when_complete),
        "run_benchmark_on_completion": bool(args.run_benchmark_on_completion),
    }
    latest_path = BATCH_DIR / "latest_submitted_batch.json"
    history_path = history_dir / f"batch_submission_{submitted_at_utc}.json"
    latest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    history_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
