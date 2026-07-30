"""Upload the CourseABSA Zenodo package as a DRAFT deposition (production zenodo.org).

Creates a new draft, zips zenodo_package/, uploads the archive + README, and sets
metadata from zenodo_package/.zenodo.json. Does NOT publish unless --publish is passed.
Re-runnable: pass --deposition-id to reuse an existing draft.

  python zenodo_upload.py                 # create draft + upload + set metadata
  python zenodo_upload.py --deposition-id 12345 --publish   # publish (mints DOI)

Token: read from ZENODO_TOKEN in Submitted/EnergeticDiffusion/.env (or --token).
"""
import argparse, json, os, sys, zipfile
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
PKG = ROOT / "zenodo_package"
ENV = ROOT.parent / "EnergeticDiffusion" / ".env"
API = "https://zenodo.org/api"


def load_token(explicit):
    if explicit:
        return explicit
    for line in ENV.read_text(encoding="utf-8").splitlines():
        if line.strip().upper().startswith("ZENODO_TOKEN"):
            return line.split("=", 1)[1].strip()
    sys.exit("ZENODO_TOKEN not found")


def zenodo_metadata():
    z = json.loads((PKG / ".zenodo.json").read_text(encoding="utf-8"))
    meta = {
        "title": z["title"],
        "upload_type": z.get("upload_type", "dataset"),
        "description": z["description"],
        "access_right": z.get("access_right", "open"),
        "license": z.get("license", "cc-by-4.0"),
        "keywords": z.get("keywords", []),
        "creators": z.get("creators", [{"name": "Unknown"}]),
    }
    if z.get("notes"):
        meta["notes"] = z["notes"]
    return {"metadata": meta}


def make_zip():
    out = ROOT / "course_absa_zenodo_v1.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for fp in sorted(PKG.rglob("*")):
            if fp.is_file():
                zf.write(fp, fp.relative_to(PKG.parent))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--token", default=None)
    ap.add_argument("--deposition-id", default=None)
    ap.add_argument("--publish", action="store_true")
    a = ap.parse_args()
    tok = load_token(a.token)
    p = {"access_token": tok}

    if a.deposition_id:
        dep_id = a.deposition_id
        r = requests.get(f"{API}/deposit/depositions/{dep_id}", params=p, timeout=60)
        r.raise_for_status()
        dep = r.json()
        print(f"[reuse] deposition {dep_id}")
    else:
        r = requests.post(f"{API}/deposit/depositions", params=p, json={}, timeout=60)
        r.raise_for_status()
        dep = r.json()
        dep_id = dep["id"]
        print(f"[create] deposition {dep_id}")

    # metadata
    r = requests.put(f"{API}/deposit/depositions/{dep_id}", params=p,
                     json=zenodo_metadata(), timeout=60)
    if r.status_code >= 400:
        print("[metadata] error", r.status_code, r.text[:500])
    else:
        print("[metadata] set")

    # files: upload the zip + a top-level README for preview
    bucket = dep["links"]["bucket"]
    zip_path = make_zip()
    for fp in [zip_path, PKG / "README.md"]:
        with open(fp, "rb") as fh:
            ru = requests.put(f"{bucket}/{fp.name}", data=fh, params=p, timeout=1200)
        print(f"[upload] {fp.name} -> {ru.status_code}")

    print(f"[draft] edit at: https://zenodo.org/deposit/{dep_id}")

    if a.publish:
        rp = requests.post(f"{API}/deposit/depositions/{dep_id}/actions/publish",
                           params=p, timeout=120)
        if rp.status_code >= 400:
            print("[publish] error", rp.status_code, rp.text[:800])
        else:
            j = rp.json()
            print("[publish] DONE. DOI:", j.get("doi"), "| record:", j.get("links", {}).get("record_html"))
    else:
        print("[publish] skipped (draft only). Re-run with --deposition-id", dep_id, "--publish")


if __name__ == "__main__":
    main()
