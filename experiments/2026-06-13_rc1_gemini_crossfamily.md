# 2026-06-13 — RC1: cross-FAMILY audit-vs-human with Gemini (reviewer Cycle 1)

**Reviewer ask:** generator (gpt-5-nano) and auditors (gpt-5.2/gpt-4.1-mini) are the
same provider family; is the audit detecting same-family latent patterns rather than
true textual faithfulness? Add a dedicated circularity discussion.

**Design:** replay the exact V7 human-validation perturbation set (1,200
faithful/flip/inject variants over Herath + EduRABSA) through a genuinely independent
family, **Google `gemini-2.5-flash` via OpenRouter**, and score Gemini-vs-HUMAN
per-aspect agreement identically to the GPT auditor (`v7_gemini_run.py`, mirrors
`v7_peraspect.py`). Key from `E:\Projects\.env.all` (OPENROUTER_API_KEY). Real-time,
1200/1200, 0 errors, ~2.5 min, cost ~$0.5.

**Result (per-aspect, n=2,482 decisions):**
| judge (family) | vs-human kappa | precision | flip-reject | inject-reject |
|---|---|---|---|---|
| gpt-4.1-mini (OpenAI) | 0.557 | 0.879 | 0.886 | 0.867 |
| gemini-2.5-flash (Google) | **0.623** | 0.892 | 0.905 | 0.848 |

Herath kappa 0.583, EduRABSA 0.674.

**Conclusion:** a fully independent model family agrees with the human labels at
kappa 0.62 — matching/exceeding the GPT auditor's 0.56 — so the audit's human
agreement is NOT an artifact of the generator's own family. This is the hard
sentence for the RC1 circularity paragraph (§6.1).

**Artifacts:** `paper/outputs/rc1_gemini_vs_human.json`,
`paper/batch_requests/v7_gemini_results.jsonl` (raw), `paper/v7_gemini_run.py`.

**Status:** completed.
