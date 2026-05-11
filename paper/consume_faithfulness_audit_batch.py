from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from openai import OpenAI


ROOT = Path(__file__).resolve().parents[1]
KEY_FILE = ROOT / ".opeai.key"
BATCH_DIR = ROOT / "paper" / "batch_requests"
RESULTS_DIR = ROOT / "paper" / "batch_results"
AUDIT_DIR = ROOT / "paper" / "faithfulness_audit"


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)


def load_client() -> OpenAI:
    api_key = KEY_FILE.read_text(encoding="utf-8").strip()
    return OpenAI(api_key=api_key)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def download_file(client: OpenAI, file_id: str, path: Path) -> None:
    content = client.files.content(file_id)
    data = content.read()
    if isinstance(data, str):
        path.write_text(data, encoding="utf-8")
    else:
        path.write_bytes(data)


def run_python(args: list[str]) -> None:
    cmd = [sys.executable] + args
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and consume a completed faithfulness-audit batch.")
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--results-path", default="")
    parser.add_argument("--model", default="")
    args = parser.parse_args()

    ensure_dirs()
    record_path = BATCH_DIR / f"{args.prefix}_submitted_batch.json"
    manifest_path = BATCH_DIR / f"{args.prefix}_manifest.csv"
    if not record_path.exists():
        raise FileNotFoundError(record_path)
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)

    submitted = load_json(record_path)
    batch_id = str(submitted["batch_id"])
    client = load_client()
    batch = client.batches.retrieve(batch_id)
    if batch.status != "completed":
        raise RuntimeError(f"Batch {batch_id} is not completed yet: {batch.status}")
    output_file_id = getattr(batch, "output_file_id", None)
    if not output_file_id:
        raise RuntimeError(f"Batch {batch_id} has no output_file_id")

    result_path = Path(args.results_path) if args.results_path else (RESULTS_DIR / f"{batch_id}_output.jsonl")
    if not result_path.exists():
        download_file(client, output_file_id, result_path)

    run_python(
        [
            str(ROOT / "paper" / "label_faithfulness_audit.py"),
            "--mode",
            "batch-consume",
            "--batch-prefix",
            args.prefix,
            "--manifest",
            str(manifest_path),
            "--results-path",
            str(result_path),
            "--model",
            str(args.model or submitted.get("model", "")),
        ]
    )

    payload = {
        "prefix": args.prefix,
        "batch_id": batch_id,
        "status": batch.status,
        "results_path": str(result_path),
        "manifest_path": str(manifest_path),
    }
    consumed_path = AUDIT_DIR / f"{args.prefix}_consumed_batch.json"
    consumed_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
