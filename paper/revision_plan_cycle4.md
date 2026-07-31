# Revision Plan — Cycle 4 (AE / reviewer comments 1–13)

Investigation date: 2026-07-31. Scope: verify each of 13 comments against the actual
manuscript (`course_absa_manuscript.html`, 1962 lines), figure-build scripts, and data
under `outputs/`, `real_transfer/`, and `experiment_rounds/phase_d2_filtering_20260604/runs/`.
Convention: `_edurabsa`-suffixed run dirs are EduRABSA; unsuffixed are Herath.
Effort estimates are AGENT execution time + compute, not human-developer time.
No paper files were edited; this is investigation + plan only.

Line numbers below refer to `course_absa_manuscript.html` unless another file is named.

---

## 1. Figure 3 left panel shows EduRABSA overlap under a Herath label
**Status: CONFIRMED**

**Evidence.**
- `real_transfer/herath_overlap_summary.csv` contains 7 aspects with `overall_experience,3697`
  (accessibility 795, exam_fairness 1167, lecturer_quality 2153, materials 581,
  organization 819, overall_experience 3697, workload 1814). This is **byte-identical** to
  the EduRABSA run file `runs/full_seed42_edurabsa/herath_overlap_summary.csv`.
- The correct Herath overlap is `runs/full/herath_overlap_summary.csv`: **9 aspects** with
  `overall_experience,557` (adds `assessment_design` 235 and `grading_transparency` 146;
  lecturer_quality 2190, etc.), consistent with the paper's stated 9-aspect overlap (line 831).
- `build_real_transfer_artifacts.py:67` `load_overlap_summary()` reads
  `real_transfer/herath_overlap_summary.csv`; that `overlap` dataframe feeds:
  - `plot_real_transfer_overview()` (line 198) → `real_transfer_overview.svg` **left panel**
    (`axes[0]` barh of `review_count` by aspect, lines 112–116) = **Figure 3 left panel**.
  - `plot_real_transfer_polarity_balance()` (line 200) → `real_transfer_polarity_balance.svg`.
  - `build_overlap_table()` (line 191) → `outputs/tables/real_transfer_overlap_publication.csv/.md`.
- **Not affected:** `real_transfer_per_aspect_heatmap.svg` reads
  `synthetic_to_real_transfer_per_aspect.csv` (line 71/139), and
  `synthetic_to_real_transfer_summary.csv` is genuine Herath
  (`eval_split=real_herath_mapped`, `n_real_reviews=2829`, `n_overlap_aspects=9`,
  aspects include assessment_design/grading_transparency). So only the **overlap** file is wrong.

**Fix.** Replace `real_transfer/herath_overlap_summary.csv` with the 9-aspect
`runs/full/herath_overlap_summary.csv`, then re-run
`python build_real_transfer_artifacts.py` to re-render `real_transfer_overview.svg`,
`real_transfer_polarity_balance.svg`, and the `real_transfer_overlap_publication.*` tables.
(Alternative — relabel the panel as EduRABSA — is not recommended: the center/right panels
and Table 8 are Herath, so the panels must stay Herath-consistent.)

**Effort: QUICK.** ~1 file swap + one script run; ~2–3 min agent time, ~$0 compute (matplotlib only).

---

## 2. Section 5.4 heading and openers are singular; EduRABSA now a second external set
**Status: CONFIRMED**

**Evidence (all still singular / Herath-only):**
- Line 829 heading: `5.4 External Validation on a Mapped Real Student-Feedback Dataset` (singular).
- Line 830–831 opener: "The study also includes **one external validation** on the annotated
  student-feedback dataset of Herath et al. [17]."
- Section 4.1, line 605: "the mapped **Herath corpus** is used only after training as an
  external evaluation set" (EduRABSA omitted); line 599 "the real-review pool" (singular).
- **Table 4** (lines 687–707) lists only two real pools — "Real validation reviews 32" (OMSCS)
  and "Mapped external test reviews 2,829 … from Herath et al." — and its caption (line 688)
  says "the **two** real-data pools." **EduRABSA (2,152 test reviews) has no row**, so the table
  under-counts the real data actually used.
- **Section 5.8 is already plural/correct** (line 1061: "two independent annotated corpora
  (Herath and EduRABSA)"), and EduRABSA is properly introduced later in 5.4 (line 900, Table 10).
  So the defect is localized to the 5.4 heading + opener, Section 4.1, and Table 4.

**Fix.**
- Line 829: retitle to plural, e.g. "External Validation on Two Mapped Real Student-Feedback Datasets."
- Line 830–831: rephrase opener to "two external validations (Herath and EduRABSA)…".
- Line 605: add EduRABSA alongside Herath as an after-training external set.
- Table 4 (lines 696–706): add a row "Mapped external test reviews (EduRABSA) 2,152" and change
  caption/line-684 text from "two real-data pools" to "three real pools" (OMSCS + Herath + EduRABSA).

**Effort: QUICK.** Text/caption edits + one table row; ~3–4 min agent time, ~$0 compute.

---

## 3. Table 10 caption claims "±std across eight seeds" but only micro-F1 has std
**Status: CONFIRMED**

**Evidence (Table 10, lines 902–913).**
- Caption line 903: "reported as mean plus or minus standard deviation across eight seeds."
- Only **Detection micro-F1** carries ±: `0.448±0.051`, `0.435±0.039`, `0.751±0.005`, `0.753±0.006`.
- Macro-F1 / Micro-recall / Sentiment MSE are bare means (e.g. row 908: `0.303`, `0.443`, `0.483`).
- Real-only (910) and pretrain (911) rows are `n/a` for macro-F1/recall/MSE.
- EduRABSA **train/val sizes not reported** (only "2,152 held-out test reviews", line 902); the
  **eight seed values are not listed** (Herath Tables 8/9 list "seeds 42,17,23,41,89"; Table 10 just says "eight seeds").

**Data availability for the full-metrics fix:** per-seed EduRABSA metrics DO exist. Each
`runs/full_seed{17,23,41,42,53,89,101,137}_edurabsa/run/summary.csv` has
`micro_f1, macro_f1, micro_recall, sentiment_mse_detected` (verified: seed42 full =
micro_f1 0.4809, macro_f1 0.2697, micro_recall 0.4015, sentiment_mse 0.4222; note the file's
`eval_split` string is a stale "real_herath_mapped" but `n_overlap_aspects=7, n_real_reviews=5584`
confirm it is EduRABSA). So synthetic-only rows can be fully populated with ±std by aggregating
the 8 `_edurabsa` seed dirs. Real-only/pretrain EduRABSA rows would need their own run outputs
located (the eight-seed synthetic-only set exists; the 4-seed real-only/pretrain std for the
other metrics may need a small aggregation or is genuinely n/a).

**Fix (choose one).**
- (a) MEDIUM: aggregate the 8 `_edurabsa` seed `summary.csv` files, add ±std to macro-F1,
  micro-recall, sentiment-MSE for the synthetic-only rows; add EduRABSA train/val/test counts
  and the seed list to the caption. Populate real-only/pretrain other-metric cells if their
  run dirs carry them, else keep n/a and say so explicitly.
- (b) QUICK: narrow the caption+table to "detection micro-F1 (mean±std, eight seeds); other
  columns are single-aggregate means shown for context," and add the seed list + test size.

**Effort: (a) MEDIUM** ~10–15 min agent time to write an aggregation script over 8 CSVs + rebuild
table, ~$0 compute. **(b) QUICK** ~3 min caption edit.

---

## 4. Table 10 EduRABSA 0.448±0.051 vs Appendix Table A24 EduRABSA 0.275; M-ABSA undefined
**Status: CONFIRMED**

**Evidence.**
- Table 10, line 908: EduRABSA synthetic-only `bert-base-uncased` = **0.448 ± 0.051** (eight seeds).
- Appendix A.5.4, Table A24, line 1827: EduRABSA = **0.275**; caption line 1821 and text line 1818
  label the three non-Herath targets as "single-seed seed-42 **screening** runs." So the same
  EduRABSA synthetic-only quantity appears as two different numbers (0.275 screening vs 0.448
  eight-seed) with no cross-reference between the two tables. A reader lands on an apparent contradiction.
- **M-ABSA**: appears at lines 1818, 1828 (Table A24), 1850, 1858 (Table A26) with **no bracketed
  citation and no definition** — described only as "Coursera MOOC / sentence (14)." By contrast
  OATS-ABSA is cited `[4]` and the review corpus `[19]`. (Grep of the manuscript found no `[n]`
  attached to any M-ABSA mention.)

**Fix.**
- Reconcile the two EduRABSA numbers: in Table A24 replace the single-seed 0.275 with the
  eight-seed 0.448±0.051 (data exists, see item 3), OR add an explicit note in A.5.4 and the
  Table A24 caption that 0.275 is a seed-42 screening run superseded by the eight-seed Table 10
  value, cross-linking the two.
- Add an M-ABSA definition + citation at first mention (line 1818): the multilingual/Coursera
  ABSA dataset it refers to, with its reference in the bibliography.

**Effort: QUICK** for the citation + a reconciling note (~3–4 min). **MEDIUM** (~10 min) if
recomputing Table A24's EduRABSA cell to the eight-seed value and re-rendering. Verify the
M-ABSA reference with the `bibtest` skill before insertion.

---

## 5. Human-annotation prose + Table 15: "blind"/"declared" tension, mislabeled column, missing stats
**Status: PARTIAL** (several sub-claims CONFIRMED; some fixes need re-computation)

**Evidence.**
- Line 1035 + Table 15 caption (line 1038): annotators are "**blind to the declared labels,
  marking for each declared aspect** whether it is expressed … and with what sentiment." The
  annotators are given the aspect to check (so not blind to the *aspect*) but not its polarity —
  the single phrase "blind to the declared labels" reads as contradicting "for each declared
  aspect." CONFIRMED wording defect (clarify: blind to declared *sentiment/polarity*, shown the
  candidate aspect).
- Table 15 column "**Reviews**" (line 1043) totals 152+152+152+154 = **610**, but the sample is
  **300 reviews / 610 declared review-aspect decisions** (line 1035). The column counts decisions,
  not reviews. CONFIRMED — rename to "Declared review-aspect decisions."
- **Sentiment IAA not reported**: line 1035 gives IAA only for aspect **presence** (Fleiss κ 0.70;
  pairwise Cohen 0.60/0.66/0.87). No inter-annotator agreement figure for sentiment. CONFIRMED.
- **Majority-sentiment rule unstated**: line 1035 reports "human-declared aspect-sentiment
  agreement is about 0.40" but never states how a single human sentiment label was derived from
  three annotators. CONFIRMED.
- **No CIs on quartile rates**: Table 15 gives point confirmation rates (0.546, 0.711, 0.750, 0.786)
  with no interval. CONFIRMED.
- **No review-level clustering**: the 610 decisions are nested within 300 reviews; nothing in the
  prose or table accounts for within-review clustering. CONFIRMED.

**Fix.**
- Reword line 1035 + caption 1038: "shown each declared aspect but blind to its declared
  sentiment/polarity."
- Rename Table 15 column "Reviews" → "Declared review-aspect decisions" (header line 1043; the
  `build_annotation_xlsx.py` / annotation source if the table is generated).
- Add sentiment IAA and state the majority-vote rule; add Wilson/bootstrap CIs to the four
  quartile rates; add a clustered (review-level) robustness note or cluster-bootstrap CI.

**Effort:** column rename + wording = **QUICK** (~3 min). Adding sentiment IAA, CIs, and a
clustered analysis = **MEDIUM** if the raw per-annotator, per-decision responses are still on
disk (recompute from the annotation file; ~10–15 min agent time, ~$0 compute); if the raw
annotation records are not retained it becomes NEW-WORK (re-collect). Locate the annotation raw
file first (check `build_annotation_xlsx.py` inputs and `outputs/`).

---

## 6. Human study only checks declared aspects; no adjudicated gold / undeclared-present aspects
**Status: CONFIRMED (as a scope limitation) → NEW-WORK to fully resolve**

**Evidence.** Line 1035: annotators mark "**for each declared aspect** whether it is expressed."
The design conditions on the declared aspect set, so an aspect that is *present but not declared*
(false negatives of the generator's own labels) is never surfaced. There is no adjudicated
full-schema gold test set — Table 15 measures majority confirmation of declared aspects only, and
Limitation 1 (line 1074) already concedes "only 9 of the 20 aspects are externally checked."

**Fix.** Either (a) frame explicitly as a limitation: state that the human study validates
precision of declared labels, not recall of undeclared-but-present aspects, and that no
adjudicated 20-aspect gold set exists yet; or (b) run a new annotation pass where annotators
label all 20 aspects free-form on a sample (enables recall + an adjudicated gold set).

**Effort: NEW-WORK** for the real fix — a fresh human/LLM-adjudicated 20-aspect annotation round
(design + provision + adjudication). Using the `human-labeling` skill, ~1–2 hrs agent setup plus
external rater wall-clock (days) and rater cost; LLM-adjudicator variant is cheaper (~$5–20 API,
~30–60 min agent). The pure-framing option (a) is QUICK (~3 min text).

---

## 7. R_phi per-review refinement operator contradicts the inter-cycle-prompt-revision pipeline
**Status: CONFIRMED**

**Evidence.**
- Section 3.5 formalism, line 550: `x = R_{\phi}(x^{(0)}, A, N, I_c)` — a per-review operator that
  maps a draft `x^{(0)}` to a final review `x`. Line 553: "`R_φ` is the **refinement step** that
  attempts to preserve labels while removing cues." Line 544: "the model produces a draft review,
  a **refinement step removes obvious synthetic cues**" — reads as a per-review second pass.
- **Contradicted by Figure 1 / Section 3.1, line 508–510:** "The dashed feedback path is shown as
  an **inter-cycle revision rather than a per-row operation** because realism validation updates
  the stabilized instruction between complete prompt states." Figure 1 caption line 512.
- **Code confirms inter-cycle, not per-review:** `generation_protocol_upgrade.py` "refinement" is
  schema/prompt-level — `best_refinement_payload()` (line 649) loads
  `validation/prompt_debug_cycle2_*_refinement.json` and the `openai-refine-schema` mode (line 798)
  refines the *attribute schema*, feeding `cues_to_avoid` / `edit_actions` into the *prompt*
  metadata. `openai_batch_prep.py` / `consume_generation_batch.py` contain no per-draft second-pass
  rewrite call. The realism loop (Appendix A.2.4–A.2.5) is a between-cycle prompt-improvement loop.

**Fix (recommended: remove the per-review operator).** In line 550 drop the `R_φ(x^{(0)},…)`
term so the sampling reads `x ∼ p_θ(· | p_c)`; delete the `R_φ` definition in line 553 and the
"a refinement step removes obvious synthetic cues" clause in line 544, replacing with the accurate
inter-cycle description already used at lines 508–510. (Alternative — document an actual per-review
rewrite — is not supported by the code and should not be adopted.)

**Effort: QUICK.** Equation + two sentence edits; ~4 min agent time, ~$0 compute.

---

## 8. Abstract "exceeds" real-only with no paired test/CI; "the unbiased metric"
**Status: CONFIRMED**

**Evidence.**
- Abstract line 412: "Evaluated by transfer to real human-annotated feedback (**the unbiased
  metric**), … synthetic pre-training followed by real fine-tuning **exceeds** real-only training."
- Contributions line 439: "Pretraining … and fine-tuning on real labels **exceeds** real-only
  training (micro-F1 **0.784 versus 0.767** on Herath, four seeds) … matches or exceeds real-only
  training on both (Herath **0.784 versus 0.767**, EduRABSA **0.753 versus 0.751**)."
- The gaps are tiny (0.784 vs 0.767 = +0.017; 0.753 vs 0.751 = +0.002) and no paired per-seed test
  or CI on the *difference* is reported in the abstract (Section 5.4 gives overlapping per-model
  CIs at lines 843/846 but not a paired-difference test). The EduRABSA +0.002 is within noise.

**Fix.**
- Abstract line 412: change "exceeds real-only training" → "**matches or numerically slightly
  exceeds** real-only training." Contributions line 439 already hedges to "matches or exceeds" for
  the two-corpus claim; make the abstract consistent with it.
- Replace "**the unbiased metric**" (line 412) with "**external real-data evaluation**."
- Optionally add a one-clause caveat that the difference is not backed by a paired significance test.

**Effort: QUICK.** Two abstract-word edits; ~2 min agent time, ~$0 compute. (A genuine paired
bootstrap on the difference, if wanted, is MEDIUM: aggregate the four-seed paired runs, ~10 min.)

---

## 9. Figure 6 sample-efficiency: two seeds, no real-only curve, full point 0.77 vs 0.784
**Status: CONFIRMED**

**Evidence (`plot_finetune_curve.py`, Figure 6 = `rc5_finetune_curve.svg`, manuscript lines 1101–1103).**
- Docstring lines 3–7: curve points come from `outputs/rc5_finetune_curve_summary.json`,
  "**two-seed sweep, unchanged**."
- Only ONE curve is plotted with per-N points: `errorbar` of "synthetic pretrain + real fine-tune"
  (line 33). Real-only (0.767) and synthetic-only (0.402) are drawn as flat **`axhline` reference
  lines** (lines 35–38), **not curves** — there is no real-only curve at N = 100/250/500/1000.
- Manuscript line 1101: full real set reaches "**0.77**, matching a model trained on real data
  alone (0.767)"; Section 5.4 headline pretrain value is **0.784** (line 846). The Figure-6 full
  point (~0.77, two seeds) is below the 0.784 four-seed headline — an internal mismatch driven by
  the two-seed curve vs four-seed headline.

**Fix.** To fully address: (i) add a real-only-from-scratch curve at each N, and (ii) raise the
sweep to the same seed budget as the headline so the full point reconciles to ~0.784, then
re-render. Interim (quick): add a caption note that the curve is a two-seed sweep and its full
point (0.77) is the two-seed estimate of the value reported at four seeds as 0.784.

**Effort: NEW-WORK** for the real-only curve + more seeds — retrain the detector at
N∈{100,250,500,1000,full} for both "real-only" and "synth-pretrain+finetune" across ≥4 seeds.
This is GPU work: best offloaded (see `gpu2modal`); ~4×5×2 short fine-tunes, ~30–60 min wall-clock
on cloud GPU, ~$3–8 compute, plus ~15 min agent orchestration. The caption-note interim is QUICK.

---

## 10. No Data/Code Availability section with repo/DOI, despite abstract claiming release
**Status: CONFIRMED**

**Evidence.**
- Abstract line 412: "the corpus and code are released"; Contributions line 436: "a **released**
  10,000-review corpus." Grep of the manuscript for Availability / Zenodo / GitHub / DOI /
  repository returned **no availability section** — the only hits are the "released" adjectives in
  the abstract/contributions. There is no statement of where the corpus/code live or a DOI.
- Release material does exist on disk (`outputs/zenodo_checkpoints/…`), and the 841 token-capped
  rows are discussed (lines 734, 1074, Table A5 line 1321) but there is no statement of whether the
  **released** corpus ships the original 841 token-capped rows or their regenerated replacements.

**Fix.** Add a "Data and Code Availability" section (Zenodo DOI only — no GitHub, per author
preference; keep it anonymized-safe for the blind PDF, e.g. "a Zenodo record (DOI to be inserted on
acceptance)"). Explicitly state whether the released corpus contains the original 841 token-capped
rows or the regenerated higher-cap replacements (from Appendix A.1.5/A.1.6), and that fixed
train/val/test splits (seed 42, 8000/1000/1000) are included.

**Effort: QUICK.** One new short section; ~4–5 min agent time, ~$0 compute. (Author must supply /
mint the actual Zenodo DOI — external step outside agent control.)

---

## 11. Real-data mappings under-documented; no EduRABSA mapping artifact or parallel tables
**Status: CONFIRMED** (Herath partly documented in data; EduRABSA undocumented)

**Evidence.**
- Herath mapping exists as data: `real_transfer/herath_mapping.json` gives the source→schema dict
  (e.g. `Lecturer#X_* → lecturer_quality`, `CA#X_x → assessment_design`,
  `Course_Structure#X_5 → grading_transparency`, `Learning_Environment#X_x → accessibility`), a
  polarity map, and `n_reviews`. But it records **no annotator/provenance** (who mapped, independent
  check, disagreement resolution). Note its `overlap_aspects` list is **stale (7 aspects)** while the
  paper and `runs/full/herath_overlap_summary.csv` use **9** — worth reconciling.
- **No EduRABSA mapping artifact exists.** `find … -iname "*edurabsa*.json"` outside run/checkpoint
  dirs returns nothing; there is no `edurabsa_mapping.json` analogous to `herath_mapping.json`. The
  7 EduRABSA overlap aspects and their source labels, and EduRABSA support/polarity distributions,
  are not documented in the paper or repo (only run outputs under `_edurabsa/` dirs and
  `outputs/edurabsa/`).
- The paper gives the Herath 9-aspect list (line 831) but no EduRABSA aspect list, no mapping
  provenance, and no parallel Herath+EduRABSA mapping/support tables.

**Fix.** (i) Create an `edurabsa_mapping.json` documenting the 7 aspects + source labels + polarity
map (recoverable from `edurabsa_worker.py` and the `_edurabsa` run outputs). (ii) Add an appendix
subsection with **parallel mapping tables** for Herath (9) and EduRABSA (7): source label → schema
aspect, plus per-aspect support and polarity distribution (the EduRABSA support/polarity is
derivable from the corrected overlap summaries once item 1 is fixed). (iii) State who performed the
mapping, whether an independent check was done, and how disagreements were resolved (or state that
a single author mapped conservatively, if that is the truth). (iv) Reconcile the stale 7-aspect
`overlap_aspects` field in `herath_mapping.json` to 9.

**Effort: MEDIUM** for building the EduRABSA mapping JSON + parallel support tables from existing
run data (~15–20 min agent time, ~$0 compute). The **provenance prose** (who mapped / independent
check / disagreement resolution) is authored fact, not computable — QUICK to write but must reflect
what actually happened; if a second independent mapper is required to make the claim true, that part
is NEW-WORK.

---

## 12. Educational Implications claims synthetic generation far cheaper but gives no numbers
**Status: CONFIRMED**

**Evidence.**
- Section 6.2, line 1089: "even at modest academic-crowdsourcing rates, a multi-thousand-comment
  private corpus typically costs **several thousand dollars** … A 10,000-review synthetic corpus
  generated through an LLM API is a **one-time generation cost far below** that annotation budget."
- This is the only cost statement: one vague human-annotation assumption ("several thousand dollars")
  and a qualitative "far below." **No concrete numbers** for: API generation cost, realism-cycle
  cost, audit cost, or cost per usable review/label.
- No cost data in the repo: grep for cost/USD/per-token/generation-cost in md/json/py finds nothing
  relevant (`tuned_training_budget_summary.csv` is training F1/MSE, not dollars); no billing/token-
  count artifact was retained.

**Fix.** Add a small cost table/paragraph with: (a) generation token counts × `gpt-5-nano` batch
price → total + per-review generation cost; (b) audit token cost (per-row `gpt-5.2` fidelity audit)
× 10,000 rows; (c) realism-cycle cost (three cycles × judged items); (d) a stated human-annotation
assumption (rate × decisions) → cost per usable review/label; (e) the ratio. Estimable from the
batch request/result files (`batch_requests/`, `batch_results/`) if token counts were logged.

**Effort: NEW-WORK-ish (light).** Requires assembling token counts from `batch_requests/` /
`batch_results/` (if present) and current per-token prices (verify with the `claude-api` / web
lookup for the exact model list prices), then computing. ~20–30 min agent time, ~$0 compute; no new
experiments if the batch logs retain token usage. If token logs are gone, regenerate a small metered
sample to estimate per-review cost (small API spend, ~$1–3).

---

## 13. Reduce repetition ~10–15% (generation, splits, Tables 3/5, 5.8 vs Discussion, Limitations)
**Status: CONFIRMED (mostly), PARTIAL for Tables 3/5**

**Evidence.**
- **Generation described twice**: Section 3.1 (lines 500–513, "Synthetic Data Generation Pipeline")
  and Section 3.5 (lines 542–556, "Prompt Stabilization and Data Generation Process") both narrate
  the sample-targets-then-realize-in-context procedure. Overlap CONFIRMED.
- **Shared-split** stated in multiple places: Section 3.2 pipeline (line 515), Section 3.6 (line
  562, "Synthetic Training Data and Real Validation Data"), and Section 4.1 (lines 599–605) all
  restate that the real pool is never in the synthetic split. Overlap CONFIRMED (line 599 and line
  605 repeat "no real-data rows are merged").
- **Tables 3 and 5**: line 739 already argues "Table 5 therefore **complements rather than repeats**
  Table 3." PARTIAL — the authors pre-empt the overlap; still, Table 3 (benchmark matrix, line 684)
  and Table 5 (analysis-block plan) share method/family content and can be tightened.
- **Section 5.8 vs Discussion opening**: 5.8 "Overall Interpretation" (lines 1058–1061) and the
  Section 6 opening (lines 1068–1069) both give a whole-study recap in nearly the same terms.
  Overlap CONFIRMED.
- **Over-long Limitations**: Section 6.1 (line 1074) is a **single ~1,100-word paragraph** packed
  with new experimental *results* that belong in Results/Appendix pointers, not Limitations —
  M-ABSA/Herath sentence-training gains (+0.21/+0.04), MAUVE 0.23/0.63/0.98, polarity-conditioning
  MSE reductions (0.23/0.16), sentence-vs-review 60% vs 93%, truncation-recovery 0.6819→0.7125, etc.
  CONFIRMED — should be split into short limitation statements with pointers, moving the
  argumentation into the relevant Results/Appendix subsections that already contain those numbers.

**Fix.** (i) Merge 3.1/3.5: keep the formal generative model in 3.5 and cut the narrative
duplication in 3.1 to a one-line pointer. (ii) State the shared-split contract once (4.1) and
replace the 3.2/3.6 restatements with a cross-reference. (iii) Trim Table 5 to the interpretive
"what each block establishes" columns not already in Table 3. (iv) Collapse 5.8 into a 2–3 sentence
bridge or fold into the Discussion opening. (v) Break the 6.1 mega-paragraph into ~5 crisp
limitation sentences, each pointing to the Results/Appendix location that carries the supporting
numbers (do not delete the numbers, relocate the argument). Target ~10–15% length reduction.

**Effort: QUICK–MEDIUM.** Prose restructuring only, no recompute; ~20–30 min agent time across the
five sites, ~$0 compute. Use the `paper-reviewer` skill for the trim pass and a content-canary check
that no load-bearing number is lost.

---

# Triage summary

### A. Quick text / caption / number fixes (no recompute)
- **2** — pluralize 5.4 heading + opener, fix 4.1, add EduRABSA row to Table 4.
- **7** — remove the per-review `R_φ` operator; use the accurate inter-cycle description.
- **8** — soften abstract "exceeds"→"matches or slightly exceeds"; "unbiased metric"→"external real-data evaluation."
- **10** — add Data/Code Availability section (Zenodo DOI only, anonymized) + 841-row provenance sentence.
- **1** — one-file swap + one figure-build run (quick, borderline recompute).
- **3(b)** / **4** (M-ABSA cite + reconciling note) / **5** (column rename + wording) / **13** (prose trim) — text-side portions.

### B. Recompute + rebuild from existing data
- **1** — re-render `real_transfer_overview.svg`, `real_transfer_polarity_balance.svg`, and
  `real_transfer_overlap_publication.*` after swapping the overlap CSV.
- **3(a)** — aggregate the 8 `_edurabsa` seed `summary.csv` files → full ±std for Table 10's other metrics.
- **4** — recompute Table A24 EduRABSA cell to the eight-seed value (optional).
- **5** — recompute sentiment IAA, quartile CIs, clustered analysis IF raw annotation records are retained.
- **11** — build `edurabsa_mapping.json` + parallel Herath/EduRABSA mapping & support tables from run data.
- **12** — compute API generation/audit/realism cost from `batch_requests/`/`batch_results/` token logs.

### C. Needs new experiments / data
- **6** — full-schema (20-aspect) adjudicated human annotation to catch undeclared-but-present aspects
  and build a gold test set (or accept as a stated limitation).
- **9** — real-only sample-efficiency curve + higher seed budget for Figure 6 (GPU retraining, cloud offload).
- **5 / 12 (fallback)** — only NEW-WORK if the raw annotation responses (5) or batch token logs (12)
  were not retained; both appear likely recoverable.
- **11 (provenance)** — NEW-WORK only if an independent second mapper is required to substantiate the
  mapping-quality claim.

### Highest-priority three
1. **Comment 1** — a wrong data file puts EduRABSA numbers under a Herath figure/table (factual error in a headline figure). Quick to fix, high credibility impact.
2. **Comment 7** — the `R_φ` per-review operator misdescribes the actual pipeline and self-contradicts Figure 1; a reviewer can read this as an accuracy problem. Quick text fix.
3. **Comment 8** — abstract overclaims ("exceeds", "the unbiased metric") on within-noise margins with no paired test; softening protects the paper's central transfer claim. Quick.
