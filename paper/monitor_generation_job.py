from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None


ROOT = Path(__file__).resolve().parents[1]
KEY_FILE = ROOT / ".opeai.key"
BATCH_DIR = ROOT / "paper" / "batch_requests"
RESULTS_DIR = ROOT / "paper" / "batch_results"
DATASET_DIR = ROOT / "paper" / "generated_datasets"
STATE_PATH = ROOT / "paper" / "batch_requests" / "automation_state.json"
LATEST_BATCH_PATH = BATCH_DIR / "latest_submitted_batch.json"
STATUS_LOG_PATH = BATCH_DIR / "latest_batch_status.json"


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)


def load_client() -> OpenAI:
    if OpenAI is None:
        raise RuntimeError("openai package is not available")
    api_key = KEY_FILE.read_text(encoding="utf-8").strip()
    return OpenAI(api_key=api_key)


def load_json(path: Path, default: dict | None = None) -> dict:
    if not path.exists():
        return default or {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def poll_batch(client: OpenAI, batch_id: str) -> dict:
    batch = client.batches.retrieve(batch_id)
    counts = getattr(batch, "request_counts", None)
    status = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "batch_id": batch.id,
        "status": batch.status,
        "input_file_id": getattr(batch, "input_file_id", None),
        "output_file_id": getattr(batch, "output_file_id", None),
        "error_file_id": getattr(batch, "error_file_id", None),
        "request_counts": None if counts is None else {
            "total": getattr(counts, "total", None),
            "completed": getattr(counts, "completed", None),
            "failed": getattr(counts, "failed", None),
        },
    }
    save_json(STATUS_LOG_PATH, status)
    return status


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
    parser = argparse.ArgumentParser(description="Poll the generation batch, download results, and trigger benchmarks once complete.")
    parser.add_argument("--run-openai-live", action="store_true", help="Run a small live OpenAI smoke baseline through absa_model_comparison.py. Not recommended for paper-facing GPT evaluation.")
    parser.add_argument("--prepare-openai-batch-eval", action="store_true", help="Prepare a batch GPT evaluation package once the generated dataset is assembled.")
    parser.add_argument("--submit-openai-batch-eval", action="store_true", help="Submit the prepared batch GPT evaluation package after preparation.")
    parser.add_argument("--openai-batch-prefix", default="openai_eval_post_generation")
    parser.add_argument("--openai-model", default="gpt-5.2")
    parser.add_argument("--openai-test-limit", type=int, default=25)
    args = parser.parse_args()

    ensure_dirs()
    state = load_json(STATE_PATH, default={})
    latest = load_json(LATEST_BATCH_PATH)
    batch_id = latest.get("batch_id")
    if not batch_id:
        raise RuntimeError(f"No batch id found in {LATEST_BATCH_PATH}")
    consume_when_complete = bool(latest.get("consume_when_complete", True))
    run_benchmark_on_completion = bool(latest.get("run_benchmark_on_completion", True))

    client = load_client()
    status = poll_batch(client, batch_id)
    state["last_polled_status"] = status

    if status["status"] != "completed":
        save_json(STATE_PATH, state)
        print(json.dumps({"message": "batch not ready", "status": status["status"]}, indent=2))
        return

    output_file_id = status.get("output_file_id")
    if not output_file_id:
        save_json(STATE_PATH, state)
        print(json.dumps({"message": "batch completed but no output_file_id yet"}, indent=2))
        return

    result_path = RESULTS_DIR / f"{batch_id}_output.jsonl"
    if not result_path.exists():
        download_file(client, output_file_id, result_path)
        state["downloaded_result_path"] = str(result_path)

    if not consume_when_complete:
        save_json(STATE_PATH, state)
        print(json.dumps({"message": "batch completed; auto-consume disabled", "result_path": str(result_path)}, indent=2))
        return

    manifest_path = latest.get("manifest_file")
    if not manifest_path:
        raise RuntimeError(f"No manifest_file found in {LATEST_BATCH_PATH}")

    dataset_path = DATASET_DIR / f"{batch_id}_generated_reviews.jsonl"
    if not dataset_path.exists():
        run_python(
            [
                str(ROOT / "paper" / "consume_generation_batch.py"),
                "--manifest",
                str(Path(manifest_path).resolve()),
                "--results-path",
                str(result_path),
                "--batch-id",
                batch_id,
            ]
        )
        state["assembled_dataset_path"] = str(dataset_path)
        state["manifest_path_used"] = str(Path(manifest_path).resolve())

    if run_benchmark_on_completion and not state.get("local_benchmark_completed"):
        benchmark_args = [
            str(ROOT / "paper" / "absa_model_comparison.py"),
            "--data-path",
            str(dataset_path),
        ]
        if args.run_openai_live:
            benchmark_args.extend(
                [
                    "--include-openai",
                    "--openai-model",
                    args.openai_model,
                    "--openai-test-limit",
                    str(args.openai_test_limit),
                ]
            )
        run_python(benchmark_args)
        state["local_benchmark_completed"] = True

    if args.prepare_openai_batch_eval and not state.get("openai_batch_eval_prepared"):
        prep_args = [
            str(ROOT / "paper" / "openai_eval_batch_prep.py"),
            "--data-path",
            str(dataset_path),
            "--prefix",
            args.openai_batch_prefix,
            "--models",
            args.openai_model,
            "--test-limit",
            str(args.openai_test_limit),
        ]
        run_python(prep_args)
        state["openai_batch_eval_prepared"] = True
        state["openai_batch_eval_prefix"] = args.openai_batch_prefix

    if args.submit_openai_batch_eval and state.get("openai_batch_eval_prepared") and not state.get("openai_batch_eval_submitted"):
        submit_args = [
            str(ROOT / "paper" / "submit_openai_eval_batch.py"),
            "--prefix",
            state.get("openai_batch_eval_prefix", args.openai_batch_prefix),
        ]
        run_python(submit_args)
        state["openai_batch_eval_submitted"] = True
        state["openai_batch_eval_prefix"] = state.get("openai_batch_eval_prefix", args.openai_batch_prefix)

    save_json(STATE_PATH, state)
    print(json.dumps({"message": "monitor pass completed", "state": state}, indent=2))


if __name__ == "__main__":
    main()
