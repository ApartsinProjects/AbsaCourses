"""Pull annotations from Argilla back into the study's responses/ folder.

Reads each task's `argilla_state.json` (written by push_to_argilla.py),
fetches submitted records with their per-user responses, and writes
`responses/task_<N>/rater_<L>_complete.csv` matching the original rater
file column schema.

User-to-rater mapping:

  The script reads `~/.argilla.json`'s optional `rater_emails` field, of
  the form `{"A": "alice@x.com", "B": "bob@x.com", "C": "carol@x.com"}`.
  Argilla records carry response.user_id; we look up the corresponding
  Argilla user's username/email and find the matching rater letter.

  If `rater_emails` is missing, the script uses the first letter of each
  unique annotator email as the rater letter (so two annotators with
  emails starting with the same letter would collide; pass an explicit
  mapping in that case).

Usage:

  python scripts/pull_from_argilla.py [--task 1] [--task 2]

Dependencies:

  pip install argilla
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import argilla as rg
except ImportError:
    rg = None  # type: ignore


ROOT = Path(__file__).resolve().parents[1]
SETTINGS_PATH = Path.home() / ".argilla.json"


def read_settings() -> Dict[str, Any]:
    if not SETTINGS_PATH.exists():
        print(f"Missing {SETTINGS_PATH}.")
        sys.exit(2)
    return json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))


def discover_tasks(only: Optional[List[int]]) -> List[int]:
    tasks_dir = ROOT / "tasks"
    nums: List[int] = []
    if not tasks_dir.exists():
        return nums
    for d in sorted(tasks_dir.iterdir()):
        if not d.is_dir() or not d.name.startswith("task_"):
            continue
        if not (d / "argilla_state.json").exists():
            continue
        try:
            n = int(d.name.split("_")[1])
        except (IndexError, ValueError):
            continue
        if only is None or n in only:
            nums.append(n)
    return sorted(set(nums))


def read_rater_columns(task_dir: Path, letter: str) -> List[str]:
    p = task_dir / f"rater_{letter}.csv"
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        return next(reader, [])


def build_user_to_letter(client: "rg.Argilla", explicit_map: Dict[str, str]) -> Dict[str, str]:
    """Return {argilla_user_id: rater_letter}.

    explicit_map is { "A": "alice@x.com", "B": "bob@x.com", ... } from the
    settings file. We resolve emails to Argilla user ids by listing users.
    """
    email_to_id: Dict[str, str] = {}
    try:
        for u in client.users:
            email = getattr(u, "username", None) or getattr(u, "email", None)
            uid = getattr(u, "id", None)
            if email and uid is not None:
                email_to_id[str(email).lower()] = str(uid)
    except Exception as exc:
        print(f"Warning: could not list Argilla users ({exc}); falling back to first-letter mapping")
    result: Dict[str, str] = {}
    if explicit_map:
        for letter, email in explicit_map.items():
            uid = email_to_id.get(str(email).lower())
            if uid:
                result[uid] = letter.upper()
    return result


def extract_response_value(question_name: str, response: Any) -> Optional[str]:
    """Pull the raw answer value for a question from a single response object."""
    # Argilla v2 returns responses as a dict keyed by question name with
    # objects that carry .value. Be defensive about the SDK shape.
    if response is None:
        return None
    val = None
    if isinstance(response, dict):
        val = response.get("value")
    else:
        val = getattr(response, "value", None)
    if val is None:
        return None
    if isinstance(val, list):
        return "|".join(str(v) for v in val)
    return str(val)


def fetch_records(client: "rg.Argilla", workspace: str, dataset_name: str) -> List[Any]:
    """Fetch all records for the dataset with their submitted responses."""
    dataset = client.datasets(name=dataset_name, workspace=workspace)
    if dataset is None:
        return []
    try:
        return list(dataset.records(with_responses=True))
    except TypeError:
        # Older SDK signatures: just iterate.
        return list(dataset.records)


def pull_one_task(client: "rg.Argilla", task_n: int, user_to_letter: Dict[str, str]) -> None:
    task_dir = ROOT / "tasks" / f"task_{task_n}"
    state = json.loads((task_dir / "argilla_state.json").read_text(encoding="utf-8"))
    dataset_name = state.get("dataset_name")
    workspace = state.get("workspace", "default")
    if not dataset_name:
        print(f"  task_{task_n}: missing dataset_name in state, skipping")
        return
    responses_dir = ROOT / "responses" / f"task_{task_n}"
    responses_dir.mkdir(parents=True, exist_ok=True)

    records = fetch_records(client, workspace, dataset_name)
    print(f"  task_{task_n}: fetched {len(records)} records from '{dataset_name}'")

    # Group answers by (letter, item_id).
    per_rater: Dict[str, Dict[str, Dict[str, str]]] = defaultdict(dict)
    for rec in records:
        # Argilla v2 uses rec.id; older versions used external_id.
        item_id = getattr(rec, "id", None) or getattr(rec, "external_id", None)
        if not item_id:
            md = getattr(rec, "metadata", None) or {}
            item_id = md.get("item_id") if isinstance(md, dict) else None
        if not item_id:
            continue
        responses = getattr(rec, "responses", None) or {}
        # responses can be a dict (question_name -> list of UserResponses)
        # or a flat list. Handle both.
        if isinstance(responses, dict):
            iter_pairs = responses.items()
        else:
            # Flat list of response objects; group by question name.
            grouped: Dict[str, list] = defaultdict(list)
            for r in responses:
                qname = getattr(r, "question_name", None) or getattr(r, "name", None)
                if qname:
                    grouped[qname].append(r)
            iter_pairs = grouped.items()  # type: ignore[assignment]
        for qname, user_responses in iter_pairs:
            for ur in (user_responses or []):
                user_id = (
                    getattr(ur, "user_id", None)
                    or (ur.get("user_id") if isinstance(ur, dict) else None)
                )
                value = extract_response_value(qname, ur)
                if user_id is None or value is None:
                    continue
                letter = user_to_letter.get(str(user_id), str(user_id)[:1].upper())
                per_rater[letter].setdefault(str(item_id), {})[qname] = value

    if not per_rater:
        print(f"  task_{task_n}: no submitted responses yet")
        return

    for letter, items in sorted(per_rater.items()):
        header = read_rater_columns(task_dir, letter)
        out_path = responses_dir / f"rater_{letter}_complete.csv"
        if not header:
            cols = ["item_id"] + sorted({k for v in items.values() for k in v})
            with out_path.open("w", encoding="utf-8", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow(cols)
                for iid, ans in items.items():
                    writer.writerow([iid] + [ans.get(c, "") for c in cols[1:]])
            print(f"  task_{task_n}: wrote {out_path.name} ({len(items)} rows, no original schema)")
            continue
        # Merge into original rater schema.
        rater_path = task_dir / f"rater_{letter}.csv"
        with rater_path.open("r", encoding="utf-8") as fin, \
             out_path.open("w", encoding="utf-8", newline="") as fout:
            reader = csv.DictReader(fin)
            writer = csv.DictWriter(fout, fieldnames=header)
            writer.writeheader()
            for row in reader:
                iid = row.get("item_id", "")
                ans = items.get(iid, {})
                merged = dict(row)
                for k, v in ans.items():
                    if k in header:
                        merged[k] = v
                writer.writerow(merged)
        print(f"  task_{task_n}: wrote {out_path.name} ({len(items)} answered items)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=int, action="append", default=None)
    args = parser.parse_args()

    if rg is None:
        print("argilla is not installed. Run: pip install argilla")
        sys.exit(2)

    cfg = read_settings()
    client = rg.Argilla(api_url=cfg.get("api_url", "").rstrip("/"), api_key=cfg.get("api_key", ""))
    explicit_map = cfg.get("rater_emails") or {}
    user_to_letter = build_user_to_letter(client, explicit_map)

    tasks = discover_tasks(args.task)
    if not tasks:
        print("No task folders with argilla_state.json found. Run push first.")
        sys.exit(2)
    print(f"Pulling tasks: {tasks}")
    for n in tasks:
        pull_one_task(client, n, user_to_letter)


if __name__ == "__main__":
    main()
