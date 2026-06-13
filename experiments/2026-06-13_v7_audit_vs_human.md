# 2026-06-13 — V7: faithfulness audit validated against human labels

**Hypothesis:** the gpt-4.1-mini label-faithfulness audit that drives the §5.9
filter agrees with HUMAN annotators, not only with other LLM judges. Falsified
if agreement is at-chance (kappa ~ 0); confirmed if the audit confirms human
gold and rejects controlled corruptions at high rates.

**Design (perturbation-controlled, no new human labeling):** from the two
human-annotated corpora (Herath, EduRABSA) build 1,200 label variants over ~400
sampled reviews: `faithful` (human gold), `flip` (polarity inverted), `inject`
(one absent aspect added). The §5.8 audit scores all three. Audit should confirm
faithful and reject flip/inject.

**Execution:** the original OpenAI Batch sat at 0/1200 for ~6 h (queue-bound,
`failed=0`). Per "race two horses, kill the loser", launched a real-time replay
of the identical 1,200 request bodies (`v7_realtime_run.py`, AsyncOpenAI,
concurrency 16). Completed 1200/1200 in 167 s, 0 errors; the batch was then
cancelled. Batch-First note: real-time was the justified exception (batch not
delivering, result gating the paper, ~$0.6 / 3 min).

**Result — per-aspect (filter-aligned; the filter audits each aspect):**
`v7_peraspect.py`, n = 2,482 aspect-label decisions.

| metric | ALL | Herath | EduRABSA |
|---|---|---|---|
| Cohen's kappa | **0.557** | 0.506 | 0.621 |
| precision | **0.879** | 0.869 | 0.890 |
| recall | 0.694 | 0.636 | 0.764 |
| F1 | 0.776 | 0.735 | 0.822 |
| accuracy | 0.776 | 0.749 | 0.810 |

Per-perturbation: faithful-keep 0.687, flip-reject **0.886**, inject-reject
**0.867**. (Set-level all-aspects conjunction is stricter: acc 0.80, kappa 0.52.)

**Conclusion:** moderate-to-substantial agreement with human annotators
(kappa 0.56, precision 0.88), with symmetric ~0.87-0.89 detection of polarity
and aspect-injection corruptions. The audit tracks human faithfulness judgment,
which rebuts the circularity objection (filter validated against humans, not only
LLMs). This is a clean positive; folded into §5.9 (new paragraph), abstract,
Limitations item 2, and the faithfulness-methodology contribution bullet.

**Artifacts:** `paper/outputs/v7_audit_vs_human.json`,
`paper/batch_requests/v7_audit_realtime_results.jsonl` (raw),
`paper/batch_requests/v7_audit_realtime_scored.csv` (set-level scored),
`paper/v7_realtime_run.py`, `paper/v7_peraspect.py`.

**Status:** completed.
