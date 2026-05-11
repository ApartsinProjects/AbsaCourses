"""Provision a free Argilla Space on Hugging Face for human-labeling tasks.

Important behavior, learned from a live run against the v2 template
(commit 2026-05-11):

* The Argilla v2 HF Space template (FROM argilla/argilla-hf-spaces:v2.8.0)
  authenticates users via **HF OAuth**, not via OWNER_API_KEY env vars.
  Setting that secret on duplicate or via add_space_secret does not change
  the API key the running Argilla picks up: keys are minted per-user
  inside Argilla after each user's first OAuth sign-in.
* For the Argilla SDK to reach the Space at all, the **Space must be
  public**, because HF's edge routes non-OAuth API requests to a 404
  marketing page when a Space is private. We default to --public for that
  reason and require an explicit --private to override.
* The **default workspace** on the v2 template is `argilla`, not
  `default`. The script writes `workspace: "argilla"` into
  ~/.argilla.json accordingly.

What this script does:

  1. Duplicate `argilla/argilla-template-space` into `<your-handle>/<name>`
     on `cpu-basic` hardware (HF requires hardware to be specified).
  2. Wait until the Space stage is RUNNING.
  3. Write a partial ~/.argilla.json with api_url, workspace, and a
     rater_emails skeleton. The api_key field is left blank, intentionally,
     because only the user can mint it via OAuth login.
  4. Print clear next-step instructions for the user to sign in via the
     Argilla UI and paste the API key into ~/.argilla.json.

Authentication for THIS script (not for the Space):
  Reads the HF token from $HF_TOKEN, $HUGGING_FACE_HUB_TOKEN, or
  the cached token from `huggingface-cli login`. Token needs Space-write
  permission.

Usage:
  python provision_argilla_space.py --name absa-labeling
  python provision_argilla_space.py --name my-study --private  # only if you'll handle HF auth manually
"""
from __future__ import annotations

import argparse
import json
import os
import re
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
DEFAULT_WORKSPACE = "argilla"
DEFAULT_HARDWARE = "cpu-basic"
BUILD_TIMEOUT_SECONDS = 600
POLL_INTERVAL_SECONDS = 10


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9-]+", "-", name.lower()).strip("-")


def resolve_token() -> Optional[str]:
    return (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )


def write_argilla_config(config: dict, force: bool = False) -> Path:
    config_path = Path.home() / ".argilla.json"
    if config_path.exists() and not force:
        backup = config_path.with_suffix(".json.bak")
        backup.write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"  backed up existing config to {backup}")
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
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
    hardware: str,
) -> int:
    token = resolve_token()
    api = HfApi(token=token)

    try:
        me = api.whoami()
    except Exception as exc:
        print(f"HF authentication failed: {exc}")
        print("Set HF_TOKEN in your environment or run `huggingface-cli login` once.")
        return 2

    username = me.get("name") or me.get("user")
    if not username:
        print(f"whoami response did not include a username: {me}")
        return 2

    space_slug = slugify(space_name)
    repo_id = f"{username}/{space_slug}"
    api_url = f"https://{slugify(username)}-{space_slug}.hf.space"

    print(f"HF user:        {username}")
    print(f"target Space:   {repo_id}")
    print(f"visibility:     {'PRIVATE (API unreachable without HF auth header)' if private else 'public'}")
    print(f"API URL:        {api_url}")
    print(f"workspace:      {workspace}")
    print(f"hardware:       {hardware}")
    print()

    if private:
        print("WARNING: private Spaces reject Argilla SDK calls because HF's edge")
        print("         requires HF auth before forwarding to the container. For the")
        print("         standard workflow, prefer --public.")
        print()

    duplicate_kwargs = dict(
        from_id=TEMPLATE_REPO,
        to_id=repo_id,
        private=private,
        hardware=hardware,
        exist_ok=True,
    )

    print(f"Duplicating {TEMPLATE_REPO} -> {repo_id} ...")
    try:
        api.duplicate_space(**duplicate_kwargs)
        print("  duplicate_space request accepted")
    except HfHubHTTPError as exc:
        msg = str(exc)
        if "already exists" in msg.lower():
            print("  Space already exists, reusing")
        else:
            print(f"Failed to duplicate Space: {exc}")
            return 1

    # Poll until RUNNING.
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

    # Sanity-check that HTTP path resolves at all (the version endpoint is open).
    try:
        import httpx
        r = httpx.get(f"{api_url}/api/v1/version", timeout=15, follow_redirects=True)
        if r.status_code == 200 and "version" in r.text:
            print(f"  /api/v1/version responded: {r.json().get('version', '?')}")
        else:
            print(f"  WARNING: /api/v1/version returned {r.status_code}")
    except Exception as exc:
        print(f"  WARNING: HTTP probe failed ({exc})")

    if write_config:
        config = {
            "api_url": api_url,
            "api_key": "",  # filled in by the user after OAuth login
            "workspace": workspace,
            "rater_emails": {rater_letter: username},
        }
        path = write_argilla_config(config)
        print(f"  wrote skeleton {path} (api_key is blank pending OAuth login)")

    print()
    print("=" * 70)
    print("Space is provisioned. One manual step remains.")
    print("=" * 70)
    print()
    print(f"  1. Open in a browser:  {api_url}")
    print(f"  2. Click 'Sign in with Hugging Face' to authenticate via OAuth.")
    print(f"  3. Once you are in the Argilla UI, click your avatar -> My Settings")
    print(f"     -> API key -> Copy.")
    print(f"  4. Paste the API key into the 'api_key' field of:")
    print(f"       {Path.home() / '.argilla.json'}")
    print()
    print(f"After that, run:")
    print(f"  python human/scripts/push_to_argilla.py --study-id <id> --task <N>")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True,
                        help="Space name (used as <hf-user>/<name>).")
    parser.add_argument("--private", action="store_true",
                        help="Create as private (default: public). NOTE: private "
                             "Spaces are unreachable to the Argilla SDK because "
                             "HF's edge requires an HF auth header.")
    parser.add_argument("--workspace", default=DEFAULT_WORKSPACE,
                        help=f"Argilla workspace to record in ~/.argilla.json "
                             f"(default: {DEFAULT_WORKSPACE}, the v2 template's default).")
    parser.add_argument("--no-config", action="store_true",
                        help="Skip writing ~/.argilla.json.")
    parser.add_argument("--rater-letter", default="A",
                        help="Letter for the owner in the rater_emails mapping (default: A).")
    parser.add_argument("--hardware", default=DEFAULT_HARDWARE,
                        help=f"HF Space hardware (default: {DEFAULT_HARDWARE} = free CPU).")
    args = parser.parse_args()

    return provision(
        space_name=args.name,
        private=args.private,
        workspace=args.workspace,
        write_config=not args.no_config,
        rater_letter=args.rater_letter.upper()[:1] or "A",
        hardware=args.hardware,
    )


if __name__ == "__main__":
    sys.exit(main())
