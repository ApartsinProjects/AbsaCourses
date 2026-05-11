from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_plan(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def plan_digest(plan: dict[str, Any]) -> str:
    payload = json.dumps(plan, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_existing_status(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def summarize_batches(batches: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"pending": 0, "running": 0, "completed": 0, "failed": 0, "skipped": 0}
    for batch in batches:
        status = str(batch.get("status", "pending"))
        counts[status] = counts.get(status, 0) + 1
    return counts


def write_status(status_path: Path, payload: dict[str, Any]) -> None:
    ensure_parent(status_path)
    payload["updated_at_utc"] = utc_now()
    counts = summarize_batches(payload["batches"])
    payload["counts"] = counts
    total = max(len(payload["batches"]), 1)
    payload["progress_fraction"] = round(counts.get("completed", 0) / total, 4)
    status_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a multi-batch experiment round with progress status.")
    parser.add_argument("--plan", required=True, help="Path to experiment round JSON plan.")
    parser.add_argument("--status", required=True, help="Path to write live round status JSON.")
    parser.add_argument("--poll-seconds", type=int, default=20, help="How often to refresh running batch status.")
    args = parser.parse_args()

    plan_path = Path(args.plan).resolve()
    status_path = Path(args.status).resolve()
    plan = load_plan(plan_path)
    digest = plan_digest(plan)
    existing = load_existing_status(status_path)

    if existing and existing.get("plan_path") == str(plan_path) and existing.get("plan_digest") == digest:
        round_state = existing
        for batch_state in round_state.get("batches", []):
            log_path = Path(batch_state["log_path"]).resolve()
            batch_state.setdefault("detail_log_path", str(log_path.with_name(log_path.stem + ".detail.log")))
            batch_state.setdefault("resume_path", str(log_path.with_suffix(".resume.json")))
            if batch_state.get("status") == "running":
                batch_state["status"] = "pending"
                batch_state["started_at_utc"] = None
                batch_state["elapsed_seconds"] = None
                batch_state["exit_code"] = None
    else:
        round_state = {
            "round_id": plan.get("round_id", plan_path.stem),
            "title": plan.get("title", ""),
            "purpose": plan.get("purpose", ""),
            "plan_path": str(plan_path),
            "plan_digest": digest,
            "started_at_utc": utc_now(),
            "completed_at_utc": None,
            "stop_on_failure": bool(plan.get("stop_on_failure", True)),
            "batches": [],
        }

        for batch in plan.get("batches", []):
            batch_id = batch["id"]
            log_path = Path(batch["log_path"]).resolve()
            resume_path = log_path.with_suffix(".resume.json")
            detail_log_path = log_path.with_name(log_path.stem + ".detail.log")
            round_state["batches"].append(
                {
                    "id": batch_id,
                    "title": batch.get("title", batch_id),
                    "purpose": batch.get("purpose", ""),
                    "status": "pending",
                    "command": batch["command"],
                    "workdir": str(Path(batch.get("workdir", str(ROOT))).resolve()),
                    "log_path": str(log_path),
                    "detail_log_path": str(detail_log_path),
                    "resume_path": str(resume_path),
                    "started_at_utc": None,
                    "completed_at_utc": None,
                    "elapsed_seconds": None,
                    "exit_code": None,
                }
            )

    write_status(status_path, round_state)

    for batch_state in round_state["batches"]:
        if batch_state.get("status") == "completed":
            continue
        log_path = Path(batch_state["log_path"])
        ensure_parent(log_path)
        workdir = Path(batch_state["workdir"])
        batch_state["status"] = "running"
        batch_state["started_at_utc"] = utc_now()
        write_status(status_path, round_state)

        with log_path.open("w", encoding="utf-8") as log_handle:
            log_handle.write(f"[{utc_now()}] START {' '.join(batch_state['command'])}\n")
            log_handle.flush()
            start = time.time()
            proc = subprocess.Popen(
                batch_state["command"],
                cwd=workdir,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                env={
                    **dict(os.environ),
                    "PYTHONUNBUFFERED": "1",
                    "EXPERIMENT_LOG_FILE": str(batch_state.get("detail_log_path", log_path.with_name(log_path.stem + ".detail.log"))),
                    "EXPERIMENT_RESUME_FILE": str(batch_state.get("resume_path", log_path.with_suffix(".resume.json"))),
                },
            )
            while True:
                code = proc.poll()
                elapsed = round(time.time() - start, 1)
                batch_state["elapsed_seconds"] = elapsed
                write_status(status_path, round_state)
                if code is not None:
                    batch_state["exit_code"] = int(code)
                    batch_state["completed_at_utc"] = utc_now()
                    batch_state["status"] = "completed" if code == 0 else "failed"
                    log_handle.write(f"\n[{utc_now()}] END exit_code={code} elapsed_seconds={elapsed}\n")
                    log_handle.flush()
                    write_status(status_path, round_state)
                    if code != 0 and round_state["stop_on_failure"]:
                        for later in round_state["batches"]:
                            if later["status"] == "pending":
                                later["status"] = "skipped"
                        round_state["completed_at_utc"] = utc_now()
                        write_status(status_path, round_state)
                        return
                    break
                time.sleep(max(args.poll_seconds, 5))

    round_state["completed_at_utc"] = utc_now()
    write_status(status_path, round_state)


if __name__ == "__main__":
    main()
