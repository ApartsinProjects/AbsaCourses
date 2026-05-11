from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Print a compact status summary for a background experiment round.")
    parser.add_argument("--status", required=True)
    parser.add_argument("--show-batches", action="store_true", help="Include the status of every batch in the round.")
    args = parser.parse_args()

    status_path = Path(args.status)
    if not status_path.exists():
        print(
            json.dumps(
                {
                    "status_path": str(status_path),
                    "exists": False,
                    "message": "Status file not found. Launch the round first or provide the correct status path.",
                },
                indent=2,
            )
        )
        return

    payload = json.loads(status_path.read_text(encoding="utf-8"))
    counts = payload.get("counts", {})
    current = next((b for b in payload.get("batches", []) if b.get("status") == "running"), None)
    summary = {
        "round_id": payload.get("round_id"),
        "title": payload.get("title"),
        "progress_fraction": payload.get("progress_fraction"),
        "counts": counts,
        "current_batch": current.get("id") if current else None,
        "current_title": current.get("title") if current else None,
        "current_elapsed_seconds": current.get("elapsed_seconds") if current else None,
        "completed_at_utc": payload.get("completed_at_utc"),
        "interrupted": payload.get("interrupted", False),
        "interruption_reason": payload.get("interruption_reason", ""),
    }
    if args.show_batches:
        summary["batches"] = [
            {
                "id": batch.get("id"),
                "status": batch.get("status"),
                "elapsed_seconds": batch.get("elapsed_seconds"),
                "exit_code": batch.get("exit_code"),
                "log_path": batch.get("log_path"),
                "detail_log_path": batch.get("detail_log_path"),
                "resume_path": batch.get("resume_path"),
            }
            for batch in payload.get("batches", [])
        ]
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
