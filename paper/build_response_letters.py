"""Build reviewer response artifacts from the per-reviewer Markdown files.

SINGLE SOURCE OF TRUTH: paper/response_reviewer_{nfat,h7LN,dWED}.md
Edit those .md files. This script PARSES them and regenerates every derived
artifact so nothing can drift:
  - response_reviewer_{id}.html   (house-style HTML letter)
  - response_letters.md           (summary + all three, concatenated)
  - response_letters_index.html   (index page)
  - changes_since_submission.html (the resp_{id} copy-paste boxes are re-synced)

The .md format each section follows (produced/consumed here):
  ### <title>
  **Requested.** <requested text>
  <response paragraph>
  *Location:* <location>
"""
from pathlib import Path
import html as _h
import re

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper"
REVIEWERS = ["nfat", "h7LN", "dWED"]  # display / concatenation order

CSS = """
  :root{ --ink:#111418; --soft:#2c3138; --muted:#5a626c; --accent:#14385c; --ok:#2f6b43; --wip:#8a5a1a; --rule:#d1d4d8; --paper:#fffdfa; }
  html{ background:#f3f0e8; }
  body{ margin:0; color:var(--ink); background:#f3f0e8;
    font-family:"Charter","Iowan Old Style","Source Serif Pro",Georgia,"Times New Roman",serif;
    line-height:1.55; font-size:11.3pt; }
  .sheet{ max-width:760px; margin:32px auto; background:var(--paper); padding:52px 58px 60px; box-shadow:0 8px 30px rgba(17,20,24,0.08); }
  h1{ font-size:17pt; font-weight:600; margin:0 0 2px; }
  .venue{ color:var(--muted); font-size:10pt; margin:0 0 22px; }
  p{ margin:0 0 10px; text-align:justify; hyphens:auto; }
  .lede{ margin-bottom:16px; }
  .rc{ margin:20px 0; padding:14px 16px; border:1px solid var(--rule); background:#fff; border-left:3px solid var(--accent); }
  .rc h2{ font-size:11.5pt; font-weight:700; color:var(--accent); margin:0 0 8px; }
  .req{ color:var(--soft); font-style:italic; margin:0 0 8px; }
  .resp b.tag{ color:var(--ok); font-style:normal; }
  .resp b.wip{ color:var(--wip); font-style:normal; }
  .loc{ display:inline-block; margin-top:4px; font-size:9.5pt; color:var(--muted); }
  code{ font-family:"Courier New",monospace; font-size:0.9em; }
  a.backlink{ position:fixed; top:16px; right:16px; z-index:1000; display:inline-flex; align-items:center; gap:7px; padding:8px 14px; background:#174b63; color:#fff; font-family:Cambria,Georgia,serif; font-size:0.92rem; text-decoration:none; border-radius:8px; box-shadow:0 6px 18px rgba(17,45,58,0.25); }
  a.backlink:hover{ background:#11607f; }
  .footer{ margin-top:24px; font-size:9pt; color:var(--muted); border-top:1px solid var(--rule); padding-top:12px; }
  @media print{ a.backlink{ display:none; } html,body{ background:#fff; } .sheet{ box-shadow:none; margin:0; } }
  @media (max-width:680px){ a.backlink{ position:static; margin:12px; } .sheet{ padding:32px 22px; } }
"""


def read_md(rid):
    return (OUT / f"response_reviewer_{rid}.md").read_text(encoding="utf-8")


def parse_letter(rid):
    """Parse response_reviewer_{rid}.md into {id, lede, points:[(title,tag,req,resp,loc)]}."""
    text = read_md(rid).strip()
    pre, sep, rest = text.partition("\n### ")
    # lede = last non-empty paragraph before the first '### ' (after title + subtitle)
    blocks = [b.strip() for b in pre.split("\n\n") if b.strip()]
    lede = blocks[-1] if blocks else ""
    points = []
    if sep:
        for chunk in ("### " + rest).split("\n### "):
            chunk = chunk.strip()
            if not chunk:
                continue
            if chunk.startswith("### "):
                chunk = chunk[4:]
            title, _, tail = chunk.partition("\n")
            tail = tail.split("\n---", 1)[0]  # drop the trailing footer on the last section
            mreq = re.search(r"\*\*Requested\.\*\*\s*(.+)", tail)
            mloc = re.search(r"\*Location:\*\s*(.+)", tail)
            req = mreq.group(1).strip() if mreq else ""
            loc = mloc.group(1).strip() if mloc else ""
            body = re.sub(r"\*\*Requested\.\*\*\s*.+", "", tail)
            body = re.sub(r"\*Location:\*\s*.+", "", body)
            resp = "\n".join(l for l in body.splitlines() if l.strip()).strip()
            # tag: 'wip' if the response opens with an in-progress marker, else 'ok'
            tag = "wip" if re.match(r"(Underway|In progress|Running)\b", resp) else "ok"
            points.append((title.strip(), tag, req, resp, loc))
    return {"id": rid, "lede": lede, "points": points}


def tagspan(tag):
    return '<b class="wip">' if tag == "wip" else '<b class="tag">'


def render(reviewer):
    rid = reviewer["id"]
    parts = [
        "<!DOCTYPE html>", '<html lang="en">', "<head>", '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>Response to Reviewer {rid}: A Controlled Synthetic Benchmark for Educational ABSA</title>",
        f"<style>{CSS}</style>", "</head>", "<body>",
        '<a class="backlink" href="course_absa_manuscript.html" title="Open the revised manuscript">&#8599; View the paper</a>',
        '<div class="sheet">',
        f"<h1>Response to Reviewer {rid}</h1>",
        '<p class="venue">A Controlled Synthetic Benchmark for Educational Aspect-Based Sentiment Analysis (TMLR)</p>',
        f'<p class="lede">{reviewer["lede"]}</p>',
    ]
    for title, tag, req, resp, loc in reviewer["points"]:
        opentag = tagspan(tag)
        lead, _, rest = resp.partition(". ")  # first sentence becomes the bold lead tag
        parts += [
            '<div class="rc">',
            f"<h2>{_h.escape(title)}</h2>",
            f'<p class="req">Requested: {_h.escape(req)}</p>',
            f'<p class="resp">{opentag}{_h.escape(lead)}.</b> {rest}</p>',
            f'<span class="loc">{loc}</span>',
            "</div>",
        ]
    parts += [
        '<p class="footer">Locations reference the revised <code>course_absa_manuscript.html</code>. Every requested change is in the revised manuscript at the cited location.</p>',
        "</div>", "</body>", "</html>",
    ]
    return "\n".join(parts)


def render_index(letters):
    cards = "\n".join(
        f'<div class="rc"><h2><a href="response_reviewer_{lv["id"]}.html">Response to Reviewer {lv["id"]}</a></h2>'
        f'<p class="req">{len(lv["points"])} points, all addressed. '
        f'Markdown: <a href="response_reviewer_{lv["id"]}.md">response_reviewer_{lv["id"]}.md</a></p></div>'
        for lv in letters)
    return "\n".join([
        "<!DOCTYPE html>", '<html lang="en">', "<head>", '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        "<title>Author responses: A Controlled Synthetic Benchmark for Educational ABSA</title>",
        f"<style>{CSS}</style>", "</head>", "<body>",
        '<a class="backlink" href="course_absa_manuscript.html" title="Open the revised manuscript">&#8599; View the paper</a>',
        '<div class="sheet">',
        "<h1>Author responses to reviewers</h1>",
        '<p class="venue">A Controlled Synthetic Benchmark for Educational Aspect-Based Sentiment Analysis (TMLR)</p>',
        '<p class="lede">Point-by-point responses to Reviewers nfat, h7LN, and dWED, plus a global summary of revisions. Every requested change is in the revised manuscript at the cited location.</p>',
        "<h2>Summary of revisions</h2>",
        '<p class="req">See <a href="response_letters.md">response_letters.md</a> for the combined set and the global summary of revisions.</p>',
        cards,
        '<p class="footer">Combined Markdown set: <code>response_letters.md</code>. Individual per-reviewer Markdown alongside each HTML letter.</p>',
        "</div>", "</body>", "</html>",
    ])


def sync_changes_boxes():
    """Re-sync the resp_{id} copy-paste textareas in changes_since_submission.html
    from the source .md, so the boxes never drift. No-op if the page is absent."""
    page_path = OUT / "changes_since_submission.html"
    if not page_path.exists():
        return
    page = page_path.read_text(encoding="utf-8")
    changed = 0
    for rid in REVIEWERS:
        esc = _h.escape(read_md(rid).strip())
        pat = re.compile(rf'(<textarea id="resp_{rid}"[^>]*>).*?(</textarea>)', re.S)
        if pat.search(page):
            page = pat.sub(lambda m: m.group(1) + esc + m.group(2), page)
            changed += 1
    page_path.write_text(page, encoding="utf-8")
    print(f"synced {changed} reviewer box(es) in changes_since_submission.html")


def main():
    letters = [parse_letter(rid) for rid in REVIEWERS]
    summary = (ROOT / "all_review" / "summary_of_revisions.md").read_text(encoding="utf-8")
    combined = [summary.strip(), "", "---", ""]
    for lv in letters:
        (OUT / f"response_reviewer_{lv['id']}.html").write_text(render(lv), encoding="utf-8")
        combined += [read_md(lv["id"]).strip(), "", "---", ""]
        print("wrote", f"response_reviewer_{lv['id']}.html", f"({len(lv['points'])} points)")
    (OUT / "response_letters.md").write_text("\n".join(combined), encoding="utf-8")
    (OUT / "response_letters_index.html").write_text(render_index(letters), encoding="utf-8")
    print("wrote response_letters.md + response_letters_index.html")
    sync_changes_boxes()


if __name__ == "__main__":
    main()
