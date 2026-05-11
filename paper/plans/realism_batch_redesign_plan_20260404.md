# Batch-First Realism Redesign Plan

## Goal

Replace the small interactive realism loop with a stronger two-stage protocol for the new binary-polarity setting. The redesigned protocol should separate:

1. a large-sample realism baseline on real reviews, and
2. iterative prompt improvement using batch-generated synthetic reviews and batch judging.

The core design principle is to use Batch API calls for all large-scale generation and judging steps, and to reserve only one small live model call per cycle for prompt rewriting.

The model roles should be asymmetric on purpose:

- **Judge**: use the strongest available model, currently `gpt-5.4`
- **Generator**: use the cheapest viable batch generation model, currently `gpt-5-nano`

This makes the realism audit stricter and more cost-efficient. The generator should not be upgraded just to make the realism loop easier; the point is to improve the prompt until low-cost generation withstands a stronger judge.

## High-Level Rationale

The previous cycle loop was useful for fast debugging, but it mixed prompt changes with synthetic-sample variation and relied on too few detected synthetic examples per cycle. That made it hard to tell whether realism was truly improving.

The redesigned workflow fixes that by:

- establishing a large real-review baseline first,
- generating a full synthetic batch for each cycle,
- judging the entire synthetic batch in batch mode,
- collecting all synthetic-detection justifications,
- rewriting the generation prompt once per cycle using the aggregated justifications,
- saving every intermediate artifact so no diagnostic evidence is lost,
- repeating on newly generated texts.

This keeps the cycle logic simple and makes the evidence more statistically useful.

## Stage 1: Large Real Baseline Audit

### Objective

Measure how often a judge flags genuinely real course reviews as synthetic, and identify the recurring reasons. This establishes the baseline false-suspicion rate of the judge on authentic data.

### Design

- Build a diverse real review pool with `n = 200`.
- Use multiple courses and preserve diversity in:
  - review length
  - sentiment mix
  - writing style
  - course difficulty
  - strongly positive / strongly negative / mixed overall tone
- Run a batch judge job over the `200` real reviews only.
- Judge model: `gpt-5.4`
- Judge task:
  - binary label: `real` or `synthetic`
  - short justification only when the label is `synthetic`
  - optional cue tags if we keep them

### Outputs

- real baseline manifest
- real baseline batch request file
- real baseline batch output file
- review-level judgment table
- aggregate summary with:
  - fraction labeled synthetic
  - mean confidence if confidence is retained
  - top repeated suspicion cues / justifications

### Key Interpretation

This stage tells us the judge's background suspicion level on real data. If many real reviews are called synthetic, that becomes the realism floor and should be accounted for when evaluating synthetic outputs.

## Stage 2: Iterative Batch Synthetic Realism Refinement

### Objective

Improve the generation prompt by repeatedly generating a fresh synthetic batch, judging it in batch mode, collecting justifications for the synthetic calls, and rewriting the generation prompt once per cycle.

### Per-Cycle Workflow

For each cycle:

1. Start from the current generation prompt.
2. Generate a synthetic batch of size `n` using the cheapest generation model in Batch API.
3. Judge the generated reviews with `gpt-5.4` in Batch API.
4. Ask only for:
   - binary label: `real` or `synthetic`
   - justification only when the label is `synthetic`
5. Collect all reviews judged as `synthetic`.
6. Extract the justifications for those synthetic calls only.
7. Run one additional `gpt-5.4` summarization call that condenses those synthetic-call justifications into:
   - major repeated failure reasons
   - recurring stylistic artifacts
   - recurring realism mismatches
   - prompt-level edits suggested by the evidence
8. Make one prompt-rewrite call that updates the generation prompt using that condensed justification summary.
9. Use the rewritten prompt in the next cycle.

### Important Design Choice

Each cycle uses newly generated synthetic texts. The goal is not matched-item comparison between prompt versions during prompt development. The goal is to iteratively remove systematic synthetic signatures found by the judge.

### Cycle Count

Run several cycles, for example:

- Cycle 0: initial prompt baseline
- Cycle 1: first rewrite from synthetic-call justifications
- Cycle 2: second rewrite
- Cycle 3: third rewrite
- Cycle 4: optional final polish cycle

Stop early if:

- the synthetic-call rate plateaus,
- justifications become repetitive,
- or later rewrites no longer change the dominant cues.

## Recommended Batch Sizes

### Stage 1

- Real baseline: `n = 200`

### Stage 2

Suggested starting point per synthetic cycle:

- synthetic generation: `n = 200`
- synthetic judging: `n = 200`
- generator model: `gpt-5-nano`
- judge model: `gpt-5.4`

This is large enough to expose repeated failure patterns without making prompt iteration too slow.

If cost is acceptable, use:

- `n = 300`

for later confirmation cycles.

## Metrics To Track

### Real Baseline Metrics

- `real_false_synthetic_rate`
  - fraction of real reviews judged synthetic
- `real_suspicion_rate_by_course`
- `real_suspicion_rate_by_length_band`
- `top_real_suspicion_justifications`
  - only from reviews labeled synthetic

### Synthetic Cycle Metrics

- `synthetic_labeled_synthetic_rate`
  - fraction of generated synthetic reviews judged synthetic
- `synthetic_labeled_real_rate`
- `mean_judge_confidence` if retained
- `top_synthetic_detection_justifications`
  - only from reviews labeled synthetic
- `top_repeated_cues`
- `justification diversity`
  - whether the same failure mode persists across many reviews

### Cross-Stage Comparative Metrics

- gap between:
  - real reviews judged synthetic
  - synthetic reviews judged synthetic
- this gap should shrink if realism improves

## Statistical Framing

After the prompt stabilizes, run one final fixed-manifest confirmation audit and compare:

- real baseline false-synthetic rate
- final synthetic false-synthetic rate

Possible statistical analyses:

- Wilson confidence intervals for suspicion rates
- difference-in-proportions confidence intervals
- bootstrap confidence intervals for gap estimates

The interactive prompt-improvement cycles are diagnostic. The final fixed-manifest audit is confirmatory.

## Artifact Preservation Rule

Every intermediate result should be saved. No cycle should overwrite the only copy of its evidence.

For each stage and cycle, preserve:

- request JSONL
- manifest CSV
- submitted batch metadata
- latest polled batch status
- raw batch output JSONL
- normalized judgment tables
- justification-only extracts for synthetic calls
- summarized failure-reason file
- rewrite-input file
- rewrite-output file
- cycle summary file

This is required so later analysis can audit:

- whether a rewrite was evidence-based,
- which synthetic reviews triggered the rewrite,
- how the judge rationale changed across cycles,
- and whether later conclusions remain reproducible.

## Recommended Output Artifacts

- `paper/validation/realism_real_baseline_200_manifest.csv`
- `paper/validation/realism_real_baseline_200_summary.json`
- `paper/validation/realism_real_baseline_200_judgments.csv`
- `paper/validation/realism_real_baseline_200_requests.jsonl`
- `paper/validation/realism_real_baseline_200_submitted_batch.json`
- `paper/validation/realism_real_baseline_200_latest_status.json`
- `paper/validation/realism_real_baseline_200_output.jsonl`
- `paper/validation/realism_synthetic_cycle_<k>_manifest.csv`
- `paper/validation/realism_synthetic_cycle_<k>_requests.jsonl`
- `paper/validation/realism_synthetic_cycle_<k>_submitted_batch.json`
- `paper/validation/realism_synthetic_cycle_<k>_latest_status.json`
- `paper/validation/realism_synthetic_cycle_<k>_generated_output.jsonl`
- `paper/validation/realism_synthetic_cycle_<k>_summary.json`
- `paper/validation/realism_synthetic_cycle_<k>_judgments.csv`
- `paper/validation/realism_synthetic_cycle_<k>_judge_requests.jsonl`
- `paper/validation/realism_synthetic_cycle_<k>_judge_submitted_batch.json`
- `paper/validation/realism_synthetic_cycle_<k>_judge_latest_status.json`
- `paper/validation/realism_synthetic_cycle_<k>_judge_output.jsonl`
- `paper/validation/realism_synthetic_cycle_<k>_synthetic_justifications.jsonl`
- `paper/validation/realism_synthetic_cycle_<k>_justification_summary.json`
- `paper/validation/realism_synthetic_cycle_<k>_rewrite_input.json`
- `paper/validation/realism_synthetic_cycle_<k>_rewrite_output.json`
- `paper/validation/realism_cycle_progress_table.csv`

## Final Recommended Protocol

1. Build a diverse `n=200` real review set.
2. Run a batch real-only baseline audit.
3. Start from the initial binary-polarity generation prompt.
4. For each cycle:
   - generate a synthetic batch,
   - judge the batch,
   - collect all synthetic-call justifications,
   - summarize those justifications,
   - rewrite the prompt once,
   - save all intermediate artifacts,
   - repeat.
5. After the final cycle, freeze the best prompt.
6. Run a final confirmation audit with:
   - the real baseline set
   - a large synthetic set from the frozen prompt
7. Compare suspicion rates and summarize the dominant remaining cues.

## Why This Design Is Better

- It uses a larger and more defensible baseline.
- It separates real-review judge bias from synthetic-review failure.
- It uses all synthetic-call justifications, not just one or two examples.
- It keeps the rewrite step global rather than overfitting to one review.
- It is compatible with Batch API and scales better than the previous live interactive loop.
- It tests whether a low-cost generator prompt can survive scrutiny from a stronger judge model.
