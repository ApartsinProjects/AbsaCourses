# Task 3: Human vs. LLM-Judge Agreement on Faithfulness

This is the most C&E-defensible study in the human-labeling track.

We previously ran an LLM-as-judge faithfulness audit: GPT-5.2 evaluated 250 synthetic
reviews and made two decisions for each declared aspect of each review:

1. **Aspect supported?** Did the review actually discuss the claimed aspect?
2. **Sentiment match?** Did the declared sentiment polarity match what the review
   actually said about that aspect?

The headline numbers from that LLM judge:
- aspect supported: **77.0%**
- aspect sentiment match: **42.3%**
- rows fully correct: **21.2%**

These numbers carry weight only if humans agree with the LLM judge on the same items.
Your job is to make exactly the same judgments on a stratified sample of those audit
items, and the coordinator will then compute human-vs-LLM agreement (Cohen's kappa)
plus correct human-vs-LLM disagreement examples for the appendix.

## Data

You will see (review, declared_aspect) pairs sampled from the audit. The sample is
stratified by the LLM judge's decisions so that you see a balanced mix of:

- supported = yes + sentiment_match = yes
- supported = yes + sentiment_match = no
- supported = no
- supported = unclear or ambiguous

You will **not** see what the LLM judge decided. Annotate blind.

## Workflow

1. Read the [codebook](../../codebook.md).
2. Open your file:
   `human/tasks/task_3_llm_judge_agreement/rater_<LETTER>.csv`
3. For each row, decide:
   - is the declared aspect actually discussed in the review?
   - is the declared sentiment polarity correct?
4. Save the completed file as `rater_<LETTER>_complete.csv` in
   `human/responses/task_3/`.

Expected time: ~1 to 2 minutes per row. The full file should take 1.5 to 3 hours.

## Column schema

| column                | meaning                                                                                                |
|-----------------------|--------------------------------------------------------------------------------------------------------|
| `item_id`             | stable identifier. Do not edit.                                                                        |
| `audit_row_id`        | the original row id in the 250-item GPT-5.2 audit. Do not edit. The coordinator uses it to join later. |
| `review_text`         | the synthetic review. Do not edit.                                                                     |
| `aspect`              | the aspect under judgment.                                                                             |
| `declared_polarity`   | the polarity the generator assigned for this aspect.                                                   |
| `aspect_supported`    | **you fill**: `yes`, `no`, or `unclear`.                                                               |
| `sentiment_match`     | **you fill**: `yes`, `no`, or `unclear`. Only meaningful when `aspect_supported == yes`.               |
| `notes`               | optional, one line of justification or rationale.                                                      |

## How to decide

- `aspect_supported = yes` when the review contains text that actually addresses this
  aspect under the codebook definition.
- `aspect_supported = no` when the review does not address this aspect at all.
- `aspect_supported = unclear` when the review hints at the aspect but the evidence is
  too oblique. Reserve this for genuinely ambiguous cases.
- `sentiment_match = yes` when the polarity in `declared_polarity` is consistent with
  what the review actually says about this aspect.
- `sentiment_match = no` when the review's tone on this aspect clearly differs from
  the declared polarity.
- `sentiment_match = unclear` when the aspect is discussed but the polarity in the text
  is genuinely ambiguous.
- If `aspect_supported != yes`, you may set `sentiment_match = unclear`.

Be conservative with `unclear`. The point of this task is to produce decisive human
calls so they can be compared head-to-head with the LLM judge's decisive calls.

## Why this matters

If human raters and the LLM judge agree at high kappa, the LLM-audit headline numbers
gain credibility and we can keep the LLM audit as the main faithfulness evidence in
the paper. If raters disagree systematically, we replace the LLM headline numbers with
the human numbers, and the LLM audit moves to the appendix as a calibration study.

Either outcome is publishable. Inconclusive results are not, so please make a call on
every row.

## Submitting

1. Save your completed file with the `_complete` suffix:
   `rater_<LETTER>_complete.csv`.
2. Drop it into `human/responses/task_3/`.
3. Email the coordinator.

Do not look at `_gpt_judgments.json` in the task folder. That file holds the LLM
judge's decisions and is for scoring only.
