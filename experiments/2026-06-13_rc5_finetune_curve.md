# 2026-06-13 — RC5: minimum fine-tuning-size curve (reviewer Cycle 1)

**Reviewer ask:** give practitioners concrete guidance — minimum fine-tuning data
size, expected performance degradation, monitoring.

**Design:** worker experiment B5 (synth-pretrain 9-aspect detection -> fine-tune on
real-Herath), with the real-train set subsampled to N in {100,250,500,1000,full}
(seeded), calib/test fixed. 2 seeds {42,17}. Modal A10G, 10 runs, ~6 min each,
all ok. Driver `modal_rc5.py`, worker `--real-train-n`.

**Curve (synth-pretrain -> real fine-tune, real-test micro-F1):**
| real fine-tune N | micro-F1 |
|---|---|
| 0 (synthetic-only) | 0.459 (ref) |
| 100 | 0.583 +/- 0.041 |
| 250 | 0.639 +/- 0.015 |
| 500 | 0.696 +/- 0.003 |
| 1000 | 0.741 +/- 0.005 |
| 1980 (full) | 0.771 +/- 0.006 (real-only ref 0.767) |

**Guidance:** ~100-250 real reviews already lift well above synthetic-only; ~500-1000
reach 90-97% of a fully real-trained model; synthetic pretraining reaches real-only
quality with roughly half the real data. Directly answers RC5.

**Artifacts:** `paper/outputs/rc5_finetune_curve_per_seed.csv`,
`paper/outputs/rc5_finetune_curve_summary.json`,
`paper/experiment_rounds/rc5_finetune_curve/modal_summary_*.json`.

**Status:** completed.
