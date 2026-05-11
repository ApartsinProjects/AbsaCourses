"""Push human-labeling tasks into an Argilla server.

Default: a free Argilla instance running on Hugging Face Spaces, addressed
through `~/.argilla.json` of the form:

    { "api_url": "https://<user>-<space>.hf.space", "api_key": "...", "workspace": "default" }

Argilla v2 SDK is used (`pip install argilla`). The settings for each task
come from a Python module the user copies from this skill's
`assets/argilla_settings/pattern_<N>.py` into their study root at
`argilla_settings/task_<N>.py`, then customizes the inventory inside.

For each task `<N>` this script:

  1. Reads `manifest.csv` for the items.
  2. Imports `<study_root>/argilla_settings/task_<N>.py` and calls its
     `make_settings()` function to obtain the rg.Settings.
  3. Creates an Argilla dataset named `<study_id>__task_<N>` in the
     configured workspace (or skips it if one already exists, unless
     --recreate is passed).
  4. Logs one Record per manifest row, with `item_id` and any task-specific
     metadata.
  5. Writes `tasks/task_<N>/argilla_state.json` recording the dataset
     identifier so the pull script can join responses back.

Per-rater assignment is handled by Argilla's TaskDistribution
(`min_submitted=<n_raters>`), so each record collects one response from
each rater. Per-user random ordering is automatic in the Argilla UI.

Usage:

  python scripts/push_to_argilla.py --study-id myproject [--task 1] [--recreate]
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import stat
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import argilla as rg
except ImportError:
    rg = None  # type: ignore


ROOT = Path(__file__).resolve().parents[1]
SETTINGS_PATH = Path.home() / ".argilla.json"


def read_settings() -> Tuple[str, str, str]:
    if not SETTINGS_PATH.exists():
        print(f"Missing {SETTINGS_PATH}. Create it with:")
        print('  {"api_url": "https://<user>-<space>.hf.space", "api_key": "...", "workspace": "default"}')
        sys.exit(2)
    if os.name != "nt":
        mode = SETTINGS_PATH.stat().st_mode
        if mode & (stat.S_IRWXG | stat.S_IRWXO):
            print(f"{SETTINGS_PATH} is world or group-readable. chmod 600 first.")
            sys.exit(2)
    cfg = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    url = cfg.get("api_url", "").rstrip("/")
    key = cfg.get("api_key", "")
    ws = cfg.get("workspace", "default")
    if not url or not key:
        print(f"{SETTINGS_PATH} must have 'api_url' and 'api_key'.")
        sys.exit(2)
    return url, key, ws


def discover_tasks(only: Optional[List[int]]) -> List[int]:
    tasks_dir = ROOT / "tasks"
    nums: List[int] = []
    if not tasks_dir.exists():
        return nums
    for d in sorted(tasks_dir.iterdir()):
        if not d.is_dir() or not d.name.startswith("task_"):
            continue
        try:
            n = int(d.name.split("_")[1])
        except (IndexError, ValueError):
            continue
        if only is None or n in only:
            nums.append(n)
    return sorted(set(nums))


def read_manifest(task_dir: Path) -> List[Dict[str, str]]:
    p = task_dir / "manifest.csv"
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def load_settings_module(task_n: int) -> object:
    """Import argilla_settings/task_<N>.py from the study root."""
    module_path = ROOT / "argilla_settings" / f"task_{task_n}.py"
    if not module_path.exists():
        print(f"  task_{task_n}: missing settings module at {module_path}")
        print(f"  Copy one of the skill's pattern files into argilla_settings/ and customize it.")
        return None
    spec = importlib.util.spec_from_file_location(f"argilla_settings_task_{task_n}", module_path)
    if spec is None or spec.loader is None:
        print(f"  task_{task_n}: cannot load module spec")
        return None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[arg-type]
    if not hasattr(mod, "make_settings"):
        print(f"  task_{task_n}: module {module_path} has no make_settings()")
        return None
    return mod


def find_existing_dataset(client: "rg.Argilla", name: str, workspace: str) -> Optional["rg.Dataset"]:
    try:
        existing = client.datasets(name=name, workspace=workspace)
        if existing is not None:
            return existing
    except Exception:
        pass
    return None


def manifest_row_to_record(row: Dict[str, str], settings: "rg.Settings") -> "rg.Record":
    """Map a manifest row into an Argilla Record.

    Convention: `text` is the main field. Any other manifest columns become
    metadata. `item_id` becomes the external_id for joining back later.
    """
    field_names = {f.name for f in settings.fields}
    fields: Dict[str, str] = {}
    metadata: Dict[str, str] = {}
    for k, v in row.items():
        if k in field_names:
            fields[k] = v
        elif k != "item_id":
            metadata[k] = v
    # Always set the primary field. If the settings expect 'text' but the
    # manifest column is named 'review_text' or similar, fall back gracefully.
    if "text" in field_names and "text" not in fields:
        for alt in ("text", "review_text", "item_text", "content"):
            if alt in row:
                fields["text"] = row[alt]
                break
    return rg.Record(
        fields=fields,
        metadata=metadata,
        id=row.get("item_id", None),
    )


def push_one_task(
    client: "rg.Argilla",
    workspace: str,
    task_n: int,
    study_id: str,
    recreate: bool,
) -> Dict[str, object]:
    task_dir = ROOT / "tasks" / f"task_{task_n}"
    manifest_rows = read_manifest(task_dir)
    if not manifest_rows:
        print(f"  task_{task_n}: empty manifest, skipping")
        return {"skipped": True, "reason": "empty manifest"}

    settings_mod = load_settings_module(task_n)
    if settings_mod is None:
        return {"skipped": True, "reason": "missing settings module"}

    settings: "rg.Settings" = settings_mod.make_settings()  # type: ignore[attr-defined]

    dataset_name = f"{study_id}__task_{task_n}"
    existing = find_existing_dataset(client, dataset_name, workspace)
    if existing and not recreate:
        print(f"  task_{task_n}: dataset '{dataset_name}' already exists, reusing")
        dataset = existing
    elif existing and recreate:
        try:
            existing.delete()
            print(f"  task_{task_n}: deleted existing dataset")
        except Exception as exc:
            print(f"  task_{task_n}: failed to delete existing dataset ({exc})")
            return {"skipped": True, "reason": "delete failed"}
        dataset = rg.Dataset(name=dataset_name, workspace=workspace, settings=settings)
        dataset.create()
        print(f"  task_{task_n}: created dataset '{dataset_name}'")
    else:
        dataset = rg.Dataset(name=dataset_name, workspace=workspace, settings=settings)
        dataset.create()
        print(f"  task_{task_n}: created dataset '{dataset_name}'")

    records = [manifest_row_to_record(row, settings) for row in manifest_rows]
    try:
        dataset.records.log(records)
    except Exception as exc:
        print(f"  task_{task_n}: log failed ({exc})")
        return {"skipped": True, "reason": "log failed", "error": str(exc)}
    print(f"  task_{task_n}: logged {len(records)} records")

    info = {
        "dataset_name": dataset_name,
        "workspace": workspace,
        "n_records": len(records),
    }
    state_path = task_dir / "argilla_state.json"
    state_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-id", required=True,
                        help="Short identifier used to name datasets in Argilla.")
    parser.add_argument("--task", type=int, action="append", default=None,
                        help="Restrict to specific task numbers (repeatable). Default: all tasks.")
    parser.add_argument("--recreate", action="store_true",
                        help="Delete and recreate datasets with matching names.")
    args = parser.parse_args()

    if rg is None:
        print("argilla is not installed. Run: pip install argilla")
        sys.exit(2)

    url, key, workspace = read_settings()
    client = rg.Argilla(api_url=url, api_key=key)

    tasks = discover_tasks(args.task)
    if not tasks:
        print("No task folders found under tasks/.")
        sys.exit(2)
    print(f"Pushing tasks: {tasks}")

    summary: Dict[str, object] = {
        "study_id": args.study_id,
        "argilla_url": url,
        "workspace": workspace,
        "tasks": {},
    }
    for n in tasks:
        info = push_one_task(client, workspace, n, args.study_id, args.recreate)
        summary["tasks"][str(n)] = info  # type: ignore[index]

    summary_path = ROOT / "tasks" / "argilla_state.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nDone. Summary at {summary_path}")


if __name__ == "__main__":
    main()
