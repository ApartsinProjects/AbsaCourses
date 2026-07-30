# Paper 9739 — combined reviewer action plan (3 reviewers)

Source: `all_review/Absa Reviewes.md` (Reviewers nfat, h7LN, dWED; reviews dated Jul 2026).

**Verdicts** — claims supported? nfat **No**, h7LN **No**, dWED **Yes**. Audience interest? all three **Yes**.
**Dominant consensus ask:** direct **human annotation of the *actual synthetic corpus*** (E1).

Status legend: ✅ done (prior cycle) · 🔶 partial · 🆕 new/open. Statuses are best-estimate from prior
revision cycles and should be verified against the current manuscript before the response letter.

## a) Presentation / write-up fixes

| # | Fix | Raised by | Status |
|---|---|---|---|
| P1 | Resolve inconsistent numbers: BERT scores in Tables 8 & 9, aspect-count totals that don't sum to 10,000, and the incomplete abstract | nfat | 🔶 counts+abstract fixed prior; verify Tables 8/9 |
| P2 | Reframe the resource as "noisy synthetic supervision," not a gold benchmark, throughout | nfat, h7LN | 🆕 wording pass |
| P3 | Dedicated generator–auditor circularity limitation paragraph (same OpenAI family) | dWED, h7LN | 🔶 cross-provider check exists; make caveat explicit |
| P4 | Strengthen transfer-limits caveats: only 9/20 aspects validated, ~60% of real-trained perf; state what practitioners must NOT conclude | dWED, h7LN, nfat | 🔶 partly in §6.1; make prominent |
| P5 | Add a practitioner roadmap: min fine-tuning data size, expected degradation on own data, monitoring requirements | dWED | 🆕 new subsection |
| P6 | Honestly frame 0.42 sentiment-match / κ=0.56 as moderate agreement; contextualize modest absolute perf (F1 0.276, MSE ~0.50) and "good enough" thresholds | h7LN, dWED | 🔶 present; tighten framing |
| P7 | Expand broader-impact: instructor evaluation, student-comment privacy, bias vs non-native/unusual writing, risk of fictional negative reviews on identifiable courses; require human review, uncertainty reporting, data protection, appeal process | nfat | 🆕 expand ethics statement |
| P8 | Shorten repeated discussion | nfat | 🆕 trim |
| P9 | Refine formatting of Figure 1, Table 5, Figure A2, Figure A3 | h7LN | 🆕 figure/table polish |
| P10 | Cite emerging methods (multimodal-sarcasm VLM; set-matching GCD) | dWED | ✅ both already in bibliography |

## b) Experimentation / evidence fixes

| # | Fix | Raised by | Status |
|---|---|---|---|
| E1 | Human-annotate a representative sample of the ACTUAL synthetic corpus and report aspect + sentiment agreement (current audit validates mainly on perturbed-real labels) | nfat, h7LN | 🆕 TOP consensus gap |
| E2 | Match filtering subsets by aspect, polarity, aspect-count, length, style to isolate the faithfulness effect (fixes "sentiment MSE only on predicted aspects → different prediction sets" confound) | nfat | 🆕 new control |
| E3 | Analyze the 841 truncated rows: biased across aspects/polarities? does excluding them change benchmark results? | dWED, h7LN | 🔶 N1 regen done; add bias analysis |
| E4 | Broaden LLM baselines beyond a single GPT family + single prompt (mainstream closed and open-source) | h7LN | ✅ #6b multi-family (Gemini/GLM/Llama) done; can extend prompts |
| E5 | Broaden validation scope beyond 32 OMSCS + 2,829 Herath / English-STEM-grad (more corpora; K-12, non-STEM, other languages) | h7LN, dWED | 🔶 OATS/EduRABSA/M-ABSA added; note remaining |
| E6 | Establish statistical realism (indistinguishability from real), not just LLM-as-judge | h7LN | 🔶 sentence-level + MAUVE done; human discrimination study is the gap |
| E7 | Address split methodology: random splits from the same generator/prompt may reward generator-specific patterns | nfat | ✅ cross-generator N3 + overlap-generalization done; surface it |
| E8 | Add qualitative error analysis / common failure modes | dWED | 🆕 new analysis |
| E9 | Justify the utility of bottom-25% low-fidelity rows (Table 12 error inflation) | h7LN | 🔶 filtering result implies it; frame explicitly |

## Priority read
- **E1 dominates** — both "No" verdicts (nfat, h7LN) hinge on the audit being validated on perturbed-real rather than actual-synthetic labels. Highest-leverage single item (there is a `human-labeling` skill).
- **Cheapest high-value presentation wins:** P1 (Tables 8/9 + abstract), P2 (reframe — nfat + h7LN both want it), P5 + P7 (roadmap + broader impact).
- **Already addressed in prior cycles (make visible, don't re-run):** E4, E7, P10.

## Update 2026-07-05: presentation pass
- P2 ✅ conclusion reframed as noisy supervision (§6.2 already had it).
- P7 ✅ ethics ¶ added: stylistic bias, no fictional-negatives-on-identifiable-entities, uncertainty reporting + appeals.
- P3/P4/P5/P6/P10 ✅ verified ALREADY present in current manuscript (prior cycles).
- P1 🔄 Tables 8/9 numeric inconsistency -> fresh consistent overlap run on Modal A10G (local 6GB failed: paging os error 1455).
- P8 (trim) / P9 (figure render QA) remain, discretionary.
