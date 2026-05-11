from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
VALIDATION_DIR = ROOT / "paper" / "validation"
LATEST_PATH = VALIDATION_DIR / "realism_poll_latest.json"
LOG_PATH = VALIDATION_DIR / "realism_poll_log.jsonl"


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def latest_cycle_progress() -> tuple[int | None, dict[str, Any] | None]:
    candidates = sorted(VALIDATION_DIR.glob("prompt_debug_cycle_*_progress.json"))
    best_cycle: int | None = None
    best_payload: dict[str, Any] | None = None
    for path in candidates:
        stem = path.stem
        try:
            cycle_id = int(stem.split("_")[3])
        except Exception:
            continue
        payload = load_json(path)
        if payload is None:
            continue
        if best_cycle is None or cycle_id > best_cycle:
            best_cycle = cycle_id
            best_payload = payload
    return best_cycle, best_payload


def latest_cycle_status(cycle_id: int | None) -> dict[str, Any] | None:
    if cycle_id is None:
        return None
    return load_json(VALIDATION_DIR / f"prompt_debug_cycle_{cycle_id}_status.json")


def completed_summaries() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(VALIDATION_DIR.glob("prompt_debug_cycle_*_summary.json")):
        payload = load_json(path)
        if payload is not None:
            rows.append(payload)
    return rows


def build_snapshot() -> dict[str, Any]:
    cycle_id, progress = latest_cycle_progress()
    status = latest_cycle_status(cycle_id)
    summaries = completed_summaries()
    active_stage = progress.get("stage") if progress else ""
    pair_index = int(progress.get("pair_index", 0)) if progress else 0
    total_pairs = int(progress.get("total_pairs", 0)) if progress else 0
    percent_complete = round((100.0 * pair_index / total_pairs), 2) if total_pairs else 0.0
    snapshot = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "active_cycle_id": cycle_id,
        "active_cycle_name": progress.get("cycle_name", "") if progress else "",
        "active_run_id": progress.get("run_id", "") if progress else "",
        "active_stage": active_stage,
        "active_course_code": progress.get("course_code", "") if progress else "",
        "pair_index": pair_index,
        "total_pairs": total_pairs,
        "pair_progress_pct": percent_complete,
        "status": status.get("status", "") if status else "",
        "completed_cycle_ids": [int(item.get("cycle_id", -1)) for item in summaries],
        "completed_cycle_count": len(summaries),
    }
    return snapshot


def write_snapshot(snapshot: dict[str, Any]) -> None:
    LATEST_PATH.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(snapshot, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Poll local realism-debug progress and persist snapshots.")
    parser.add_argument("--interval-seconds", type=int, default=300)
    parser.add_argument("--loop", action="store_true")
    args = parser.parse_args()

    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

    while True:
        snapshot = build_snapshot()
        write_snapshot(snapshot)
        print(json.dumps(snapshot, indent=2))
        if not args.loop:
            return
        time.sleep(max(1, int(args.interval_seconds)))


if __name__ == "__main__":
    main()
