# WHERE IS EVERYTHING — CourseABSA (read me first)

_Last updated: 2026-06-13_

## Canonical copy
**This folder — `E:\Projects\Submitted\CourseABSA` — is the main git repository.**
Its `main` branch equals `origin/main` (finished paper). Start any new Claude
session with THIS folder as the project root.

- GitHub remote: https://github.com/ApartsinProjects/AbsaCourses (branch `main`)
- GitHub Pages (live HTML): https://apartsinprojects.github.io/AbsaCourses/paper/course_absa_manuscript.html
- Overleaf (TMLR LaTeX): https://www.overleaf.com/project/6a21c5fdb69dd3a8a4f5ca4e

## Finish the folder cleanup (run once, in a normal terminal)
There is an orphaned linked worktree at `E:\Claude\CourseABSA\hopeful-kowalevski-04ee10`
(checked out on `main`; its `.git` points to the now-missing `E:\Projects\CourseABSA`).
Remove it so this repo can sit on `main`:

```bash
cd /e/Projects/Submitted/CourseABSA
git worktree remove --force /e/Claude/CourseABSA/hopeful-kowalevski-04ee10
git worktree prune
git checkout main
```
If `remove` errors on the broken link: `rm -rf /e/Claude/CourseABSA` then
`git worktree prune && git checkout main`. After this you may delete the whole
`E:\Claude\CourseABSA` folder.

## Your preserved local edits
The 40 uncommitted edits that were in this folder (a big
`realism_validation_experiment.py` change, modified paper HTML/scripts, and many
deletions) are saved on branch **`backup/pre-main-sync-20260613`** (commit
`e5951d4`). Inspect with `git show e5951d4` or `git checkout backup/pre-main-sync-20260613`.
Delete that branch only after salvaging anything you still want.

## Key locations (on `main`)
- `paper/course_absa_manuscript.html` — the manuscript (source of truth, 1641 lines)
- `paper/course_absa_manuscript_1col.docx` / `_2col.docx` — Word versions (see rebuild note)
- `paper/cover_letter.html` — TMLR cover letter
- `paper/outputs/figures/` — figure SVGs (Fig 1 = `synthetic_data_generation_pipeline.svg`)
- `paper/build_docx.py` — regenerates the DOCX from the HTML
- `paper/build_conceptual_svgs.py` — regenerates the conceptual figures
- `paper/_tmlr/` — TMLR LaTeX bundle (gitignored; regenerate via the html2tex skill)
- `experiments/` — experiment registry and diagnostics

## Open follow-ups
1. **Rotate the Overleaf Git token** (`olp_...`) — it appeared in a session transcript.
   New token -> `~/.overleaf-token`.
2. **Rebuild the DOCX** (`python paper/build_docx.py`) — the committed `.docx` are one
   revision behind the HTML (deferred on request).
3. **TMLR style-guide compliance audit** of the LaTeX bundle (queued).
4. Rebuild + push the Overleaf bundle so the new Fig 1 + restructure propagate.
