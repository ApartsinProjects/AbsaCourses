# 2026-06-13 — Threshold-free metrics (AUROC) + metric reframe

**Question:** the held-out micro-F1 (~0.276) reads as alarmingly low; is the detector actually weak, or is micro-F1 just a compressed operating-point metric on a 20-aspect sparse multilabel task with noisy labels?

**Setup:** local RTX 2060 (low-priority, GPU-free-gated). Re-ran the internal 20-aspect benchmark detection with `paper/compute_metrics_local.py` (bert-base-uncased, seed 42, 8000/1000/1000 split, same Config) and co-computed thresholded + threshold-free metrics on ONE model/split. Artifact: `paper/outputs/internal_auroc_local.json`.

**Result (reproduces the paper exactly, then adds the threshold-free view):**
| metric | value |
|---|---|
| micro-F1 (thresholded) | **0.2760** (= paper 0.2760) |
| macro balanced accuracy | **0.6229** (= paper) |
| micro recall / precision | 0.440 / 0.201 |
| **micro AUROC (threshold-free)** | **0.688** |
| **macro AUROC** | **0.681** |
| macro average precision | 0.344 |

Per-aspect AUROC 0.54–0.83: highest for lexically marked aspects (`pacing` 0.83, `workload` 0.81, `grading_transparency` 0.76, `lecturer_quality` 0.76, `tooling_usability` 0.76); lowest for implicit ones (`prerequisite_fit` 0.54, `support` 0.55, `overall_experience` 0.56) — matching the §5.5 hard-aspect pattern.

Transfer (computed earlier from saved C1 Herath predictions, n=2829): macro-AUROC 0.66, micro-AUROC 0.76.

**Conclusion:** the detector discriminates well above chance (AUROC 0.68, balanced accuracy 0.62); the low micro-F1 reflects the decision threshold and the measured label noise, not weak representations. AUROC is independent of the operating point and the noisy gold threshold, so it is the cleanest rebuttal to "F1 is low."

**Paper changes (committed):** (1) generalized Tier-0 reframe — abstract + contributions now lead with the general audit-filter-validate methodology (education as testbed), not the synthetic-test F1; (2) §5.3 elevates macro balanced accuracy beside micro-F1, frames micro-F1 as a strict operating-point metric (trivial floor 0.183), and reports the threshold-free AUROC.

**Status:** completed. (V7 audit-vs-human batch still pending; folds in separately.)
