# Task 2: Herath Slice Re-annotation Under the 20-Aspect Schema

You will re-annotate a small slice of the Herath et al. 2022 student feedback corpus
using **this project's 20-aspect inventory** (see [codebook](../../codebook.md)).

The Herath corpus already has its own annotations under a different aspect schema, and
those annotations have been **mapped** to nine of our 20 aspects (see
`paper/real_transfer/herath_mapping.json`). Your independent annotations are used to
validate that mapping: if humans annotating from scratch under our schema agree with
the mapped labels, the mapping is defensible; if not, we learn where the mapping is
lossy.

You and the other raters annotate the same items independently. Disagreements are
resolved in a coordinator-led adjudication round.

## Workflow

1. Read the [codebook](../../codebook.md). For this task, all 20 aspects matter, not
   just the ones in the Herath mapping. Annotate every aspect that is actually
   discussed in the review, even if it falls outside the mapping's overlap set.
2. Open your file:
   `human/tasks/task_2_herath_reannotation/rater_<LETTER>.csv`
3. For each review, fill in `discussed_*` and `polarity_*` columns for the 20 aspects.
4. Save your completed file as `rater_<LETTER>_complete.csv` in
   `human/responses/task_2/`.
5. Wait for the coordinator to compile the adjudication round.

Expected time: ~3 to 5 minutes per review, depending on length. The full file should
take 2 to 4 hours.

## Important framing notes

- You are **not** being asked to reproduce the Herath team's annotations. You are
  annotating from scratch under our schema. Differences between your labels and
  Herath's mapped labels are evidence, not errors.
- Some Herath reviews are short fragments. If a review is too short or too generic to
  support any aspect-level annotation, mark all aspects as `discussed = no` and add
  a note.
- A few Herath reviews mention non-English text or are corrupted. If you cannot
  understand a review, mark all aspects as `unclear` and add a note.

## Column schema

| column                               | type                              | who fills it                          |
|--------------------------------------|-----------------------------------|---------------------------------------|
| `item_id`                            | stable identifier                 | already filled                        |
| `review_text`                        | the review                        | already filled                        |
| `discussed_<aspect>` (20 columns)    | `yes`, `no`, `unclear`            | **you**                               |
| `polarity_<aspect>` (20 columns)     | `positive`, `neutral`, `negative` | **you** if `discussed == yes`         |
| `notes`                              | free text                         | optional                              |

The full list of aspects is in the [codebook](../../codebook.md). The column names use
exactly those aspect identifiers.

If `discussed_<aspect>` is `no` or `unclear`, leave `polarity_<aspect>` blank.

## Adjudication

After all raters submit, the coordinator computes Cohen's kappa per aspect. Items where
raters disagree are re-presented in an `adjudication_round.csv` with all raters'
answers visible (but identities anonymized). You will either:

- agree with the majority answer, or
- argue for a different answer with a short justification.

The coordinator then sets the final label for each disagreement based on the
discussion.

## Important rules

- Do not consult other raters until the adjudication round.
- Do not edit the `review_text` column.
- Do not change row order; `item_id` is the join key.
- Annotate independently. Your value to this study is precisely that you do not see
  the other raters' decisions or the Herath team's existing labels.
- Do not look at `paper/real_transfer/herath_mapping.json` while annotating. Use only
  the [codebook](../../codebook.md).
