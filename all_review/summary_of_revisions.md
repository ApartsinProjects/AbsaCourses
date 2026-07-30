# Summary of revisions

We thank the reviewers; the revision strengthens the evidence and reconciles the reporting.

- **Reporting reconciled.** All synthetic-to-real transfer numbers now derive from one multi-seed run (Tables 8, 9), the transfer figures were regenerated to match, aspect counts sum to 10,000, and the abstract is complete.
- **Direct human validation of the audit** (new Table 14): three annotators re-labeled a stratified synthetic sample (Fleiss kappa 0.70); human confirmation of declared aspects rises monotonically with the audit score, validating the audit on synthetic text (N1, H1).
- **Covariate-matched filtering** (new Table 13): matching on aspect, polarity, count, length, and style and scoring on a common gold set attributes the filtering gain to faithfulness alone (N2).
- **New robustness experiments:** independent open-weights auditors reproduce the audit (A.29), the token-capped rows are not biased (A.30), the learnable signal is generator-invariant (A.31), an audit-quartile dose-response is monotone (A.32), and a qualitative error analysis (A.33).
- **Framing sharpened:** measured-and-controlled noisy synthetic supervision (not a gold benchmark); broader-impact, generator-auditor circularity, transfer limits, and a practitioner roadmap made explicit; baselines span four providers; external checks span four real corpora.
- **Artifacts released:** corpus, mapped real data, human-annotation study, code, and best-per-target checkpoints deposited on Zenodo.
