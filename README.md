# A Controlled Synthetic Benchmark for Educational Aspect-Based Sentiment Analysis

A synthetic-data-centered study of aspect-based sentiment analysis (ABSA) for
higher-education course reviews. Two linked artifacts: (1) a generation pipeline
that produces 10,000 student-style reviews labeled across a **20-aspect pedagogical
inventory**, and (2) a reproducible benchmark that runs classical, transformer, joint,
and GPT-based ABSA approaches on those reviews and reports a conservative external
evaluation against the Herath et al. 2022 student-feedback corpus.

![Hero](assets/hero.png)

> **Manuscript (rendered):** [`paper/course_absa_manuscript.html`](paper/course_absa_manuscript.html)
> &nbsp;&middot;&nbsp; **Live page:** [https://apartsinprojects.github.io/AbsaCourses/](https://apartsinprojects.github.io/AbsaCourses/)
> &nbsp;&middot;&nbsp; **Authors:** Yehudit Aperstein, Alexander Apartsin

---

## Abstract

Educational ABSA can support course improvement, but public aspect-labeled student
feedback remains scarce because educational reviews are private, institution-specific,
and expensive to annotate. This study introduces a controlled synthetic benchmark for
educational ABSA built from **10,000** synthetic course reviews with explicit
train-validation-test splits and a **20-aspect** pedagogical schema spanning
instructional quality, assessment and course management, learning demand, learning
environment, and engagement.

The corpus is generated with sampled target labels, sampled nuance attributes, and a
realism-tuned prompt refined through a three-cycle judge-editor procedure. On the
resulting benchmark, local baselines show the task is nontrivial; the strongest untuned
local model, BERT-base, reaches **micro-F1 = 0.2760** on a held-out detection split,
and a tuned BERT schedule reaches **0.2930**. Full-test GPT-5.2 inference reaches
**0.2519** in zero-shot mode and **0.2501** with retrieval-based few-shot prompting.
A conservative external evaluation on **2,829 mapped Herath reviews** yields BERT
micro-F1 = **0.4593** on a 9-aspect overlap, indicating partial synthetic-to-real
transfer. Realism and faithfulness diagnostics are reported as generator-side analyses
that explain how the benchmark was stabilized and where label noise remains.

---

## What the evidence currently supports

| claim                                                                                | evidence                                                                                                |
|--------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------|
| 10K-review, 20-aspect synthetic corpus with multi-style coverage and prompt cycles.  | Manuscript Section 3 + [`paper/generated_datasets/`](paper/generated_datasets/) (data) + [`paper/plans/`](paper/plans/) (protocol). |
| Local benchmark is calibrated, multi-seed, and reproducible.                         | [`paper/benchmark_outputs/model_comparison_summary.csv`](paper/benchmark_outputs/model_comparison_summary.csv), Phase A and Phase B1 status files under [`paper/experiment_rounds/`](paper/experiment_rounds/). |
| GPT-5.2 inference family executed at full-test scale in batch mode.                  | [`paper/benchmark_outputs/openai_batch_eval_summary.csv`](paper/benchmark_outputs/openai_batch_eval_summary.csv) and per-prompt CSVs alongside. |
| Synthetic-to-real transfer measured on a public real corpus.                         | [`paper/real_transfer/synthetic_to_real_transfer_summary.csv`](paper/real_transfer/synthetic_to_real_transfer_summary.csv) + manuscript Section 6.6 + Figure A1. |
| LLM-judge faithfulness audit at corpus scale (n=250, 501 declared aspects).          | [`paper/faithfulness_audit/faithfulness_audit_gpt-5_2_250_summary.json`](paper/faithfulness_audit/faithfulness_audit_gpt-5_2_250_summary.json). |
| Three human-rater studies scaffolded and ready to run.                               | [`human/`](human/) — codebook, instructions, samplers, scoring scripts, pre-generated rater CSVs. **Not yet collected.** |

---

## Dataset

**Active corpus:** [`paper/generated_datasets/batch_69cc15c483488190941478aa4e3a976d_generated_reviews.jsonl`](paper/generated_datasets/) (10,000 records).

**Aspect inventory (20)**, grouped into five pedagogical blocks:

| Group                                  | Aspects |
|----------------------------------------|---------|
| Instructional quality                  | `clarity`, `lecturer_quality`, `materials`, `feedback_quality` |
| Assessment and course management       | `exam_fairness`, `assessment_design`, `grading_transparency`, `organization`, `tooling_usability` |
| Learning demand and readiness          | `difficulty`, `workload`, `pacing`, `prerequisite_fit` |
| Learning environment                   | `support`, `accessibility`, `peer_interaction` |
| Engagement and value                   | `relevance`, `interest`, `practical_application`, `overall_experience` |

Each record carries `target_attributes` (the sampled aspect-sentiment labels), a
`nuance_attributes` block (course, instructor, semester, persona, recommendation
stance, etc.), and the final review text. See manuscript Appendix A.1 for full
definitions.

**External evaluation corpus:** [`external_data/Student_feedback_analysis_dataset/`](external_data/Student_feedback_analysis_dataset/)
holds the Herath et al. 2022 LREC corpus (3,000 hand-annotated reviews, MIT
licensed). The mapping from Herath's aspect schema into 9 of our 20 aspects is at
[`paper/real_transfer/herath_mapping.json`](paper/real_transfer/herath_mapping.json).

---

## Headline results

**Local benchmark on the held-out detection split** (cells from [`paper/benchmark_outputs/model_comparison_summary.csv`](paper/benchmark_outputs/model_comparison_summary.csv)):

| approach              | micro-F1 | macro-F1 | sent-MSE |
|-----------------------|---------:|---------:|---------:|
| `bert-base-uncased`   |   0.2760 |   0.3364 |   0.4959 |
| `distilbert-base-uncased` | 0.2691 | 0.3376 |   0.5044 |
| `distilbert_joint`    |   0.2524 |   0.3248 |   0.5428 |
| `bert_joint`          |   0.2447 |   0.3208 |   0.5288 |
| `tfidf_two_step`      |   0.2326 |   0.2867 |   0.6830 |
| `albert-base-v2`      |   0.1829 |   0.1828 |   0.5773 |
| `roberta-base`        |   0.1829 |   0.1828 |   0.6838 |

**GPT-5.2 inference (full 1,000-review test split, batch mode):** see
[`paper/benchmark_outputs/openai_batch_eval_summary.csv`](paper/benchmark_outputs/openai_batch_eval_summary.csv).
Headline: zero-shot 0.2519, retrieval-based few-shot 0.2501. Aspect-by-aspect and
two-pass variants still pending.

**Synthetic-to-real transfer on Herath (n=2,829, 9-aspect overlap):**

| approach              | micro-F1 | macro-F1 | sent-MSE |
|-----------------------|---------:|---------:|---------:|
| `bert-base-uncased`   |   0.4593 |   0.3059 |   0.3990 |
| `distilbert-base-uncased` | 0.4156 | 0.3515 |   0.3888 |
| `tfidf_two_step`      |   0.3740 |   0.2303 |   0.7019 |

Source: [`paper/real_transfer/synthetic_to_real_transfer_summary.csv`](paper/real_transfer/synthetic_to_real_transfer_summary.csv).

**LLM-judge faithfulness audit on synthetic labels (n=250 reviews, 501 declared
aspects):**

| metric                                  | rate    |
|-----------------------------------------|--------:|
| aspect supported                        |  0.7705 |
| aspect-sentiment match                  |  0.4232 |
| rows fully supported                    |  0.5920 |
| rows fully sentiment-match              |  0.2120 |

Source: [`paper/faithfulness_audit/faithfulness_audit_gpt-5_2_250_summary.json`](paper/faithfulness_audit/faithfulness_audit_gpt-5_2_250_summary.json).
A human-rater replay of the same audit is scaffolded in `human/tasks/task_3_llm_judge_agreement/`.

---

## Human-rater studies

Three studies are pre-built under [`human/`](human/) and ready to send to raters.
None of them is collected yet.

| task | purpose | sample |
|---|---|---|
| **Task 1** realism + label faithfulness | can humans distinguish synthetic from real reviews? are the synthetic labels faithful? | 40 synthetic + 40 real, 3 raters |
| **Task 2** Herath re-annotation under the 20-aspect schema | does our schema-mapping for Herath agree with independent humans? | 50 Herath reviews, 3 raters |
| **Task 3** human vs. LLM-judge agreement | does GPT-5.2's faithfulness audit agree with human raters? | 80 stratified audit pairs, 3 raters |

Each task has rater CSVs, a hidden ground-truth file, a codebook, instructions, and
a scoring script that produces paper-ready tables and Cohen's kappa. See
[`human/README.md`](human/README.md) for the full workflow.

---

## Repository layout

```
.
|-- README.md
|-- index.html                         <- Pages entry: redirects to the manuscript
|-- .nojekyll                          <- disables Jekyll on Pages
|-- assets/hero.png                    <- README banner
|-- human/                             <- three human-rater studies (codebook, instructions, samplers, scoring, generated CSVs)
|-- external_data/                     <- Herath et al. 2022 LREC corpus (MIT)
|-- edu/                               <- legacy 6K/10-aspect notebooks (superseded by the 10K corpus under paper/generated_datasets/)
`-- paper/
    |-- course_absa_manuscript.html    <- 1,490-line draft (Sections 1-8 + Appendix)
    |-- plans/                         <- experiment plans and audits (live_todo, integrated_plan, round plans, etc.)
    |-- *.py, *.ps1                    <- production scripts: build_, aggregate_, evaluate_, submit_, consume_, analyze_, ...
    |-- benchmark_outputs/             <- local benchmark CSVs (model_comparison_summary, openai_batch_eval_summary)
    |-- real_transfer/                 <- Herath transfer outputs and aspect-mapping
    |-- faithfulness_audit/            <- LLM-judge audit details at multiple sample sizes
    |-- experiment_rounds/             <- phase A and B1 run statuses + logs
    |-- generated_datasets/            <- the 10K corpus and intermediate generation outputs
    |-- generation_protocol/           <- prompt package and realism-cycle artifacts
    |-- validation/                    <- realism-cycle batches and OMSCS samples
    |-- analysis/                      <- diagnostic CSVs and reports
    `-- outputs/                       <- publication-ready figures (PNG + SVG) and tables
```

---

## Reproducing the analysis

Each script is independent and can be run from the repo root. They expect the data
files in `paper/generated_datasets/`, `paper/real_transfer/`, and `external_data/` to
be present; large data folders are not all checked in. See `paper/plans/` for the
governing protocols.

```bash
# Local benchmark (transformer + classical + joint + GPT-batch family)
python paper/absa_model_comparison.py

# Synthetic-to-real transfer on Herath
python paper/evaluate_synthetic_to_real_transfer.py

# LLM-judge faithfulness audit
python paper/label_faithfulness_audit.py

# Build the three human studies (already executed; rerun to regenerate)
python human/scripts/sample_task_1.py
python human/scripts/sample_task_2.py
python human/scripts/sample_task_3.py
```

---

## Scope and limitations

**Supported by current evidence:**

- A multi-aspect, multi-style synthetic corpus that is learnable across classical,
  transformer, joint, and GPT-based inference modes.
- Conservative external validation on a public, hand-annotated educational corpus.
- Corpus-scale label-quality diagnostics from an LLM-judge audit.

**Not yet defensible, by design of this draft:**

- **Human-rater confirmation of label faithfulness and realism.** The studies are
  scaffolded in [`human/`](human/) but data collection has not begun.
- **A second real-data corpus.** Transfer is currently evaluated on Herath only.
- **A faithfulness-aware filtering ablation.** Internal plans flag this as the
  single most valuable remaining experiment for journal acceptance.
- **Hybrid synthetic-plus-real fine-tuning** against a small real training slice.

See [`paper/plans/integrated_paper_update_plan_20260403.md`](paper/plans/integrated_paper_update_plan_20260403.md)
and [`paper/plans/live_todo_20260403.md`](paper/plans/live_todo_20260403.md) for the
full status of remaining work.

---

## Citation

```bibtex
@misc{aperstein2026courseabsa,
  title  = {A Controlled Synthetic Benchmark for Educational Aspect-Based Sentiment Analysis},
  author = {Aperstein, Yehudit and Apartsin, Alexander},
  year   = {2026},
  url    = {https://github.com/ApartsinProjects/AbsaCourses}
}
```

---

## Acknowledgements

External validation uses the publicly released **Herath et al. 2022** student
feedback corpus (LREC), distributed under MIT. Cite the original release:

> Herath, M., Chamindu, K., Maduwantha, H., & Ranathunga, S. (2022). Dataset and
> Baseline for Automatic Student Feedback Analysis. *Proceedings of the Thirteenth
> Language Resources and Evaluation Conference*, 2042-2049.

Realism-validation sample selection drew from public OMSCS course pages for the
preliminary cycle-0 exploration.
