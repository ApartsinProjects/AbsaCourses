# 2026-06-13 — Literature-completeness scout (TMLR prep)

Goal: find 2024-2026 work the paper should cite-and-contrast before submission.
All candidates validated with bibtest (.bib identifier check): 7/7 valid.

## Added (became refs [32]-[38])
- **[32] Xu, Zhang, Wang, Xu 2025, DS2-ABSA** (ACL 2025) — closest competitor: synthetic ABSA + in-generator label refinement for few-shot. Contrast: we separate audit-filter-validate as general data-QC with a behavioral negative control + human-kappa, in the educational domain. (§2 synthetic-data para)
- **[33] Hellwig, Fehle, Wolff 2025** (Expert Systems w/ Applications 261:125514) — LLM synthetic ABSA samples in low-resource. Contrast: fixed released benchmark + filter vs per-task few-shot aug. (§2)
- **[34] Long et al. 2024** (Findings of ACL) — synthetic-data generation/curation/evaluation survey. Cited as the framing pipeline; we instantiate the curation+evaluation stages for label faithfulness. (§2)
- **[35] Gilardi, Alizadeh, Kubli 2023** (PNAS 120(30)) — LLM annotators rival/exceed crowd workers. Anchors using an LLM as label auditor. (§5.9)
- **[36] Gu et al. 2025, A Survey on LLM-as-a-Judge** (arXiv:2411.15594 / The Innovation) — judge reliability + biases. Motivates cross-provider + human-label controls. (§5.9)
- **[37] Ye, Shah, Zhang, Chava 2025, SiDyP** (KDD 2025) — calibrate classifier against LLM-generated label noise. Contrast: we filter data-side before training vs their model-side calibration (complementary). (§2)
- **[38] Awadh, Sulaiman, Mahmoud 2025** (J. King Saud Univ. CIS) — systematic review of ABSA in MOOCs. Educational-ABSA currency + schema positioning. (§2 educational para)

## Not added (unverified author lists per scout; excluded pending confirmation)
- Computational Linguistics 2025 "Evaluating Synthetic Data Generation from User Generated Text" (MIT Press page 403'd; author list unconfirmed).
- IJAIED 2024 educational-survey-feedback LLM paper (first-author from snippet only).

## Key finding
The exact intersection {synthetic + educational + ABSA + faithfulness-filtering} has no
direct precedent, which supports the novelty framing. DS2-ABSA is the nearest neighbor
and is the one a TMLR reviewer would most likely fault us for missing; now cited.
