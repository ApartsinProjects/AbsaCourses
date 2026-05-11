from __future__ import annotations

import csv
import json
import re
import time
from pathlib import Path
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = ROOT / "paper"
MANUSCRIPT = PAPER_DIR / "course_absa_manuscript.html"
EXTRACTED_CSV = PAPER_DIR / "bibliography_extracted.csv"
REPORT_CSV = PAPER_DIR / "bibliography_verification_report.csv"
REPORT_JSON = PAPER_DIR / "bibliography_verification_summary.json"


def normalize_title(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip().lower()
    text = re.sub(r"[^a-z0-9 ]+", "", text)
    return text


def extract_references() -> list[dict[str, str]]:
    html = MANUSCRIPT.read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "html.parser")
    rows: list[dict[str, str]] = []
    for idx, li in enumerate(soup.select("ol.refs > li"), start=1):
        href = li.find("a")["href"] if li.find("a") else ""
        text = " ".join(li.get_text(" ", strip=True).split())
        rows.append({"index": str(idx), "href": href, "text": text})
    with EXTRACTED_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["index", "href", "text"])
        writer.writeheader()
        writer.writerows(rows)
    return rows


def infer_title(entry_text: str) -> str:
    year_match = re.search(r"\b(?:19|20)\d{2}\.\s+(.*?)\s+\.\s+", entry_text)
    if year_match:
        return year_match.group(1).strip()
    return entry_text


def classify_reference(row: dict[str, str], session: requests.Session) -> dict[str, str]:
    href = row["href"]
    title = infer_title(row["text"])
    doi = href.split("doi.org/", 1)[1] if "doi.org/" in href else ""
    landing_status = ""
    final_url = href
    official_title = ""
    crossref_ok = False
    openalex_ok = False
    notes: list[str] = []
    service = ""

    try:
        response = session.get(href, timeout=25, allow_redirects=True)
        landing_status = str(response.status_code)
        final_url = response.url
    except Exception as exc:  # pragma: no cover
        notes.append(f"landing_error:{type(exc).__name__}")

    try:
        if doi:
            response = session.get(f"https://api.crossref.org/works/{quote(doi)}", timeout=25)
            if response.ok:
                item = response.json()["message"]
                official_title = (item.get("title") or [""])[0]
                crossref_ok = True
                service = "crossref_doi"
        else:
            response = session.get(f"https://api.crossref.org/works?query.title={quote(title)}&rows=3", timeout=25)
            if response.ok:
                for item in response.json()["message"].get("items", []):
                    candidate = (item.get("title") or [""])[0]
                    if candidate and (
                        normalize_title(candidate) == normalize_title(title)
                        or normalize_title(title) in normalize_title(candidate)
                        or normalize_title(candidate) in normalize_title(title)
                    ):
                        official_title = candidate
                        crossref_ok = True
                        service = "crossref_title"
                        break
    except Exception as exc:  # pragma: no cover
        notes.append(f"crossref_error:{type(exc).__name__}")

    try:
        response = session.get(f"https://api.openalex.org/works?search={quote(title)}&per-page=3", timeout=25)
        if response.ok:
            for item in response.json().get("results", []):
                candidate = item.get("display_name", "")
                if candidate and (
                    normalize_title(candidate) == normalize_title(title)
                    or normalize_title(title) in normalize_title(candidate)
                    or normalize_title(candidate) in normalize_title(title)
                ):
                    if not official_title:
                        official_title = candidate
                    openalex_ok = True
                    if not service:
                        service = "openalex_title"
                    break
    except Exception as exc:  # pragma: no cover
        notes.append(f"openalex_error:{type(exc).__name__}")

    if href.lower().endswith(".pdf") and "aclanthology.org/" in href:
        notes.append("prefer_acl_landing_page")
    if "openreview.net/" in href:
        notes.append("verify_final_venue_on_openreview")
    if landing_status in {"403", "429"} and (crossref_ok or openalex_ok):
        notes.append("official_page_blocks_bot_access")
    if landing_status == "404":
        notes.append("broken_or_incorrect_link")

    status = "verified"
    if "broken_or_incorrect_link" in notes:
        status = "incorrect_or_broken"
    elif "prefer_acl_landing_page" in notes or "verify_final_venue_on_openreview" in notes:
        status = "verified_but_should_be_improved"
    elif not (crossref_ok or openalex_ok or "aclanthology.org/" in href):
        status = "needs_manual_review"

    return {
        "index": row["index"],
        "title": title,
        "href": href,
        "landing_status": landing_status,
        "final_url": final_url,
        "crossref_ok": str(crossref_ok),
        "openalex_ok": str(openalex_ok),
        "service_used": service,
        "official_title": official_title,
        "status": status,
        "notes": ";".join(notes),
    }


def main() -> None:
    session = requests.Session()
    session.headers.update({"User-Agent": "CourseABSA-BiblioVerifier/1.0"})

    rows = extract_references()
    verified: list[dict[str, str]] = []
    for row in rows:
        verified.append(classify_reference(row, session))
        time.sleep(0.2)

    with REPORT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(verified[0].keys()))
        writer.writeheader()
        writer.writerows(verified)

    summary: dict[str, int] = {}
    for row in verified:
        summary[row["status"]] = summary.get(row["status"], 0) + 1

    REPORT_JSON.write_text(
        json.dumps({"report": str(REPORT_CSV), "summary": summary}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(REPORT_JSON)


if __name__ == "__main__":
    main()
