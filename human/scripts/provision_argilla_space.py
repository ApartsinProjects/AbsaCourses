"""Provision a free Argilla Space on Hugging Face for human-labeling tasks.

Duplicates the `argilla/argilla-template-space` into `<your-handle>/<name>`,
sets a strong random OWNER_API_KEY and OWNER_PASSWORD as Space secrets,
waits for the Space to finish building, and writes `~/.argilla.json` with
the API URL + key + workspace + a starter rater_emails mapping.

After this script returns, `push_to_argilla.py` can run unchanged against
the newly provisioned Space.

Authentication:
  Reads the HF token from one of (in order):
    - $HF_TOKEN
    - $HUGGING_FACE_HUB_TOKEN
    - the cached token from `huggingface-cli login`

  The token needs permission to create Spaces and to set Space secrets.
  A fine-grained token with "Manage your repositories: Create new repos"
  and "Write access to contents/settings" is sufficient.

Usage:
  python provision_argilla_space.py --name absa-labeling
  python provision_argilla_space.py --name my-study --public --workspace pilot
"""
from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import stat
import sys
import time
from pathlib import Path
from typing import Optional

try:
    from huggingface_hub import HfApi
    from huggingface_hub.utils import HfHubHTTPError
except ImportError:
    print("huggingface_hub is not installed. Run: pip install huggingface_hub")
    sys.exit(2)


TEMPLATE_REPO = "argilla/argilla-template-space"
DEFAULT_OWNER_USERNAME = "owner"
BUILD_TIMEOUT_SECONDS = 600  # 10 minutes
POLL_INTERVAL_SECONDS = 10


def slugify(name: str) -> str:
    """HF Space URL slug rules: lowercase, alphanumeric + hyphens."""
    return re.sub(r"[^a-z0-9-]+", "-", name.lower()).strip("-")


def resolve_token() -> Optional[str]:
    return (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )


def confirm_or_warn(prompt: str, default_yes: bool = False) -> bool:
    suffix = " [Y/n] " if default_yes else " [y/N] "
    try:
        ans = input(prompt + suffix).strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False
    if not ans:
        return default_yes
    return ans in ("y", "yes")


def write_argilla_config(config: dict, force: bool = False) -> Path:
    """Write ~/.argilla.json, backing up any prior file."""
    config_path = Path.home() / ".argilla.json"
    if config_path.exists() and not force:
        backup = config_path.with_suffix(".json.bak")
        backup.write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"  backed up existing config to {backup}")
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    # Restrict permissions on Unix (Windows ignores chmod).
    if os.name != "nt":
        try:
            os.chmod(config_path, stat.S_IRUSR | stat.S_IWUSR)
        except OSError:
            pass
    return config_path


def provision(
    space_name: str,
    private: bool,
    workspace: str,
    write_config: bool,
    rater_letter: str,
    hardware: Optional[str],
) -> int:
    token = resolve_token()
    api = HfApi(token=token)

    try:
        me = api.whoami()
    except Exception as exc:
        print(f"HF authentication failed: {exc}")
        print("Set HF_TOKEN in your environment, or run `huggingface-cli login` once.")
        return 2

    username = me.get("name") or me.get("user")
    if not username:
        print(f"whoami response did not include a username: {me}")
        return 2

    space_slug = slugify(space_name)
    repo_id = f"{username}/{space_slug}"
    api_url = f"https://{slugify(username)}-{space_slug}.hf.space"

    print(f"HF user:        {username}")
    print(f"target Space:   {repo_id}  (private={private})")
    print(f"API URL:        {api_url}")
    print(f"workspace:      {workspace}")
    print()

    # Generate strong, transient credentials.
    owner_api_key = secrets.token_urlsafe(32)
    owner_password = secrets.token_urlsafe(16)

    secrets_payload = [
        {"key": "OWNER_API_KEY", "value": owner_api_key},
        {"key": "OWNER_PASSWORD", "value": owner_password},
    ]

    duplicate_kwargs = dict(
        from_id=TEMPLATE_REPO,
        to_id=repo_id,
        private=private,
        secrets=secrets_payload,
        exist_ok=True,
    )
    if hardware:
        duplicate_kwargs["hardware"] = hardware

    print(f"Duplicating {TEMPLATE_REPO} -> {repo_id} ...")
    try:
        api.duplicate_space(**duplicate_kwargs)
        print("  duplicate_space request accepted")
    except HfHubHTTPError as exc:
        msg = str(exc)
        if "already exists" in msg.lower():
            print("  Space already exists, reusing")
            # Refresh secrets even when reusing so we know the credentials.
            try:
                api.add_space_secret(repo_id, "OWNER_API_KEY", owner_api_key)
                api.add_space_secret(repo_id, "OWNER_PASSWORD", owner_password)
                api.restart_space(repo_id)
                print("  refreshed owner secrets and triggered restart")
            except Exception as exc2:
                print(f"  WARNING: could not refresh secrets ({exc2}); using existing credentials means the printed values may be wrong")
        else:
            print(f"Failed to duplicate Space: {exc}")
            return 1

    # Poll until RUNNING (or fail fast on errors).
    deadline = time.time() + BUILD_TIMEOUT_SECONDS
    last_stage: Optional[str] = None
    print("Waiting for build...")
    while time.time() < deadline:
        try:
            info = api.get_space_runtime(repo_id)
            stage = str(info.stage)
        except Exception as exc:
            print(f"  poll failed ({exc}); retrying")
            time.sleep(POLL_INTERVAL_SECONDS)
            continue
        if stage != last_stage:
            print(f"  stage: {stage}")
            last_stage = stage
        if stage in ("RUNNING", "APP_STARTING"):
            # APP_STARTING usually transitions to RUNNING within seconds.
            if stage == "RUNNING":
                break
        if stage in ("CONFIG_ERROR", "BUILD_ERROR", "RUNTIME_ERROR", "NO_APP_FILE"):
            print(f"BUILD FAILED in stage {stage}")
            print(f"Investigate at https://huggingface.co/spaces/{repo_id}")
            return 1
        time.sleep(POLL_INTERVAL_SECONDS)
    else:
        print(f"Timed out after {BUILD_TIMEOUT_SECONDS}s waiting for RUNNING")
        return 1

    # Quick health check via the Argilla SDK if available.
    try:
        import argilla as rg
        time.sleep(5)  # give the argilla server a moment to settle
        client = rg.Argilla(api_url=api_url, api_key=owner_api_key)
        # Touch any endpoint that requires auth.
        _ = list(client.workspaces)
        print("  Argilla API responded to auth probe")
    except ImportError:
        print("  (skipping API health check; install argilla to enable)")
    except Exception as exc:
        print(f"  WARNING: Argilla API probe failed ({exc}). Space may still be warming up.")

    if write_config:
        config = {
            "api_url": api_url,
            "api_key": owner_api_key,
            "workspace": workspace,
            "rater_emails": {rater_letter: DEFAULT_OWNER_USERNAME},
        }
        path = write_argilla_config(config)
        print(f"  wrote {path}")

    print()
    print("=" * 64)
    print("Provisioning complete.")
    print(f"  Argilla UI:        {api_url}")
    print(f"  Owner username:    {DEFAULT_OWNER_USERNAME}")
    print(f"  Owner password:    {owner_password}")
    print(f"  Owner API key:     [in ~/.argilla.json]")
    print("=" * 64)
    print()
    print("Save the owner password if you want to log into the Argilla UI manually.")
    print("Next: push tasks with")
    print(f"  python human/scripts/push_to_argilla.py --study-id <id> --task <N>")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True,
                        help="Space name (used as <hf-user>/<name>).")
    parser.add_argument("--public", action="store_true",
                        help="Create as public; default is private.")
    parser.add_argument("--workspace", default="default",
                        help="Argilla workspace to record in ~/.argilla.json (default: default).")
    parser.add_argument("--no-config", action="store_true",
                        help="Skip writing ~/.argilla.json.")
    parser.add_argument("--rater-letter", default="A",
                        help="Letter for the owner in the rater_emails mapping (default: A).")
    parser.add_argument("--hardware", default=None,
                        help="HF Space hardware (default: free CPU). e.g. 'cpu-basic'.")
    args = parser.parse_args()

    return provision(
        space_name=args.name,
        private=not args.public,
        workspace=args.workspace,
        write_config=not args.no_config,
        rater_letter=args.rater_letter.upper()[:1] or "A",
        hardware=args.hardware,
    )


if __name__ == "__main__":
    sys.exit(main())
