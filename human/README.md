# Human Labeling Studies

This folder contains everything needed to run the three human-rater studies that the
*Computers & Education* submission depends on for external validity:

| task                              | what it produces                                                            | reviewer concern it addresses                  |
|-----------------------------------|-----------------------------------------------------------------------------|-----------------------------------------------|
| **Task 1** realism + faithfulness | human judgments on synthetic-vs-real realism and on label faithfulness      | "does this look like real student writing?"   |
| **Task 2** Herath re-annotation   | independent 20-aspect annotations of a Herath slice                         | "is the schema mapping to Herath defensible?" |
| **Task 3** LLM-judge agreement    | human judgments on the same 250-item set the GPT-5.2 audit already covered  | "is the LLM-as-judge audit trustworthy?"      |

## Folder layout

```
human/
|-- README.md
|-- codebook.md                       <- canonical 20-aspect definitions
|-- instructions/
|   |-- task_1_realism_and_faithfulness.md
|   |-- task_2_herath_reannotation.md
|   `-- task_3_llm_judge_agreement.md
|-- scripts/
|   |-- sample_task_1.py              <- builds Task 1 rater files
|   |-- sample_task_2.py              <- builds Task 2 rater files
|   |-- sample_task_3.py              <- builds Task 3 rater files
|   |-- score_task_1.py               <- realism accuracy, faithfulness, kappa
|   |-- score_task_2.py               <- per-aspect kappa + mapping comparison
|   `-- score_task_3.py               <- human-vs-LLM kappa + disagreement analysis
|-- tasks/                            <- rater-facing files live here
|   |-- task_1_realism_and_faithfulness/
|   |-- task_2_herath_reannotation/
|   `-- task_3_llm_judge_agreement/
`-- responses/                        <- raters drop completed files here
    |-- task_1/
    |-- task_2/
    `-- task_3/
```

## Data sources

The samplers read from the project's existing files:

| variable           | path                                                                                                                 |
|--------------------|----------------------------------------------------------------------------------------------------------------------|
| synthetic corpus   | `paper/generated_datasets/batch_69cc15c483488190941478aa4e3a976d_generated_reviews.jsonl` (10K reviews, 20 aspects)  |
| real corpus        | `paper/real_transfer/herath_mapped_real_reviews.jsonl` (2,829 Herath reviews, mapped to 9 of our 20 aspects)         |
| LLM-judge audit    | `paper/faithfulness_audit/faithfulness_audit_gpt-5_2_250_details.csv` (250 reviews x 501 declared aspects)           |

The samplers use relative paths from the repo root, so run them from
`E:\Projects\CourseABSA\` (or any checkout of this branch where the data files are
in place).

## End-to-end flow

1. **Coordinator (you):**
   ```bash
   /c/Python314/python human/scripts/sample_task_1.py
   /c/Python314/python human/scripts/sample_task_2.py
   /c/Python314/python human/scripts/sample_task_3.py
   ```
   This populates `human/tasks/task_*/` with rater files, manifests, and hidden
   ground-truth files.

2. **Send rater files to raters.** Each rater gets the files matching their letter
   assignment (A, B, C). For Task 1, send Part 1 first; release Part 2 only after
   Part 1 is returned.

3. **Raters fill in CSVs** and place completed files in `human/responses/task_<N>/`,
   suffixed with `_complete.csv`.

4. **Coordinator scores:**
   ```bash
   /c/Python314/python human/scripts/score_task_1.py
   /c/Python314/python human/scripts/score_task_2.py
   /c/Python314/python human/scripts/score_task_3.py
   ```
   Each scoring script writes summary metrics and disagreement tables back into
   `human/tasks/task_<N>/scoring/` for inclusion in the paper.

## Rater assignments

Default is **3 raters** (A, B, C). Override per task with `--n_raters N`.

## Hidden files

These files live inside the task folders but must not be shown to raters:

- `task_1_realism_and_faithfulness/_truth.json` (source labels for Part 1)
- `task_3_llm_judge_agreement/_gpt_judgments.json` (LLM judge's calls)

Both have an underscore prefix to make them visually obvious. They are checked in
intentionally; the scoring scripts depend on them.

## Sample sizes

The default sample sizes are calibrated for what the paper needs at a publishable
power level given 3 raters:

| task   | items per rater | (review, aspect) pairs per rater | rationale                                              |
|--------|-----------------|----------------------------------|--------------------------------------------------------|
| 1      | 80              | ~120-160 in Part 2               | n >= 64 needed for Cohen kappa with adequate power     |
| 2      | 50              | n/a (whole-review, 20 aspects)   | small enough for full kappa per aspect; cheap to run   |
| 3      | 80              | 80                               | stratified across 4 LLM-judge buckets, 20 per bucket   |

You can override with CLI flags on each sampler. Smaller samples are fine for a
pilot pass.

## Citations and ethics

- The Herath corpus is used under its MIT license; the original release lives at
  `external_data/Student_feedback_analysis_dataset/`. Cite Herath et al. 2022 (LREC).
- All real reviews are public. We do not redact named instructors or TAs because the
  source is already public; raters should treat names as part of the review.
- Raters work under an honorary-author or paid-rater agreement set by the coordinator
  outside this repository.
