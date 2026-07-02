# Visual Upgrade Plan

This note records the figure set that should remain active in the manuscript after the production 10K / 20-aspect benchmark update. It is informed by the current repository evidence and a Gemini-assisted figure audit.

## Main figures

1. `Figure 1`: realism-improvement curve
   Purpose: show cycle-level prompt-debug progress without overclaiming indistinguishability.

2. `Figure 2`: synthetic data generation pipeline
   File: `paper/outputs/figures/synthetic_data_generation_pipeline.svg`
   Purpose: explain the controllable generation workflow and evidence boundary.

3. `Figure 3`: ABSA benchmark overview
   File: `paper/outputs/figures/absa_analysis_pipeline.svg`
   Purpose: show the shared data contract, split policy, and executed local model matrix.

4. `Figure 4`: production aspect-sentiment heatmap
   File: `paper/outputs/figures/production_aspect_sentiment_heatmap.svg`
   Purpose: summarize label coverage across the active 20-aspect corpus.

5. `Figure 5`: production review-length distribution
   File: `paper/outputs/figures/production_review_length_distribution.svg`
   Purpose: show the actual output-length profile of the 10K corpus.

6. `Figure 6`: production method comparison
   File: `paper/outputs/figures/production_method_comparison.svg`
   Purpose: compare executed local baselines on the held-out test split.

7. `Figure 7`: per-aspect behavior of the best model
   File: `paper/outputs/figures/production_best_model_per_aspect.svg`
   Purpose: show which pedagogical aspects are easiest and hardest for the strongest local model.

8. `Figure 9`: realism-validation pipeline
   File: `paper/outputs/figures/realism_validation_pipeline.svg`
   Purpose: clarify that real reviews are used only for generator diagnosis, not benchmark training.

## Appendix / optional figures

1. `canary_smoke_method_comparison.svg`
   Keep as a workflow-validation artifact only if the paper still discusses the acceptance-gate canary.

2. `canary_smoke_aspect_count_distribution.svg`
   Move to appendix or archive if space is tight.

3. `grade_style_heatmap.png`
   Optional appendix EDA if the final paper needs more diversity evidence.

4. `learning_curve.png`
   Only keep if rerun on the 10K production dataset; otherwise archive because it reflects the older setup.

5. `style_holdout_micro_f1.png`
   Only keep if rerun on the 10K production dataset; otherwise archive because it reflects the older setup.

## Figures to retire or de-emphasize

- Any figure whose caption still refers to the old 10-aspect “released corpus” as the main benchmark.
- Any figure derived only from the earlier notebook run if a production-batch counterpart now exists.
- Decorative visuals that do not support a direct claim in the manuscript.

## Caption rules

- State exactly which corpus each figure uses.
- Separate generator validation from ABSA benchmark evidence.
- Avoid language like “human-like” or “indistinguishable” unless supported by the stated statistical test.
- Keep captions claim-matched: describe what the figure shows, not what the paper hopes it implies.
