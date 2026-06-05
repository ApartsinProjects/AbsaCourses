# Experiment Registry — Index

| Date | Experiment | Status | One-line summary |
|---|---|---|---|
| 2026-06-05 | [reviewer_response_evidence](2026-06-05_reviewer_response_evidence.md) | completed | TMLR reviewer rebuttal: test-split faithfulness (held-out == full corpus, match_rate 0.576), detection-set robustness of MSE contrasts (not confounded; Table 8E reconciled), multi-judge convergence (support 0.82-0.87, match 0.68-0.79 vs gpt-5.2); flagged a manuscript judge label-swap. |

- 2026-06-05_reviewer_AB_runs.md — Modal A10G, 4 seeds: (A) filtered-test F1 +0.030 [0.006,0.054] faithful vs full, reproduce-gate 0.277; (B) real-Herath-trained 0.767 [0.734,0.800] vs synthetic-only 0.459. status: completed
- [2026-06-05_validation_v5.md](2026-06-05_validation_v5.md) — Modal A10G + local CPU: validation battery. V1 permutation control permuted 0.182 [0.177,0.187] vs real 0.276 (gap 0.094, == floor) -> genuine signal; V2 trivial floors (all-pos 0.183, random 0.101) -> BERT +0.093 above chance; V4 clean-label ceiling row_score==1.0 0.319 [0.290,0.348] / >=0.5 0.340 [0.332,0.348] (+0.043/+0.064) -> noise-capped; V3 learning curve 0.183->0.285 monotone over 250->8000 -> data-scalable; B5 synth->real fine-tune 0.784 [0.758,0.809], +0.017 over real-only 0.767 (3/4 seeds above), +0.325 over synth-only 0.459. status: completed
