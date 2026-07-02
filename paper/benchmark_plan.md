## ABSA Benchmark Plan

This note defines the current paper-facing benchmark matrix for the synthetic educational ABSA study. It is aligned to the executed `10,000`-review corpus and the active `20`-aspect pedagogical schema.

### Benchmark scope

- Primary benchmark data: production synthetic review corpus with `10,000` assembled reviews over `20` aspects.
- Split policy: fixed three-way `8,000 / 1,000 / 1,000` train/validation/test split.
- Seeds: the canonical result table uses seed `42`, and three-seed robustness summaries are now available for TF-IDF, DistilBERT, and BERT.
- Detection task: multi-label aspect detection.
- Sentiment task: sentiment prediction for present or detected aspects, reported as detected-aspect MSE in the current production table.
- Validation use: the held-out validation split is used for threshold selection and model-choice decisions.
- Test reporting: final benchmark numbers are reported on the held-out test split only.
- Real review role: real OMSCS reviews are used only for realism validation and are not mixed into the benchmark split.

### Matrix interpretation

- `executed`: full result exists on the production `10K / 20`-aspect benchmark or the full held-out test split.
- `executed supporting`: result exists as a supporting comparison or robustness analysis tied to the main benchmark.
- `implemented, not reported`: the method is implemented in the benchmark code, but the corresponding full reported result is not yet included in the manuscript tables.

### Design choices

- Two-step discriminative modeling remains the main benchmark because it matches the repository's implemented training logic and makes omission versus polarity errors easier to analyze.
- Transformer encoders are compared under the same train/validation/test split to measure internal learnability of the synthetic corpus.
- Prompt-based baselines are included as deployment-relevant comparators and are now executed on the full held-out test split with schema-constrained batch inference.
- The prompt-based branch has a batch-evaluable path for zero-shot, fixed few-shot, diverse few-shot, and retrieval-based few-shot variants using exact-key structured output.
- The realism-validation judge loop is separate from this benchmark and is not itself an ABSA baseline.

### Formal task view

Let \(x\) denote a review, let \(\mathcal{A}=\{a_1,\dots,a_K\}\) denote the aspect inventory, let \(z \in \{0,1\}^K\) denote aspect presence, and let \(s \in \{-1,0,1\}^K\) denote aspect sentiment for present aspects.

The benchmark goal is to produce

\[
x \mapsto (\hat{z}, \hat{s}).
\]

The main benchmark families are:

- Two-step discriminative:
  \[
  \hat{z} = f_\theta(x), \qquad \hat{s} = g_\phi(x,\hat{z})
  \]
  where aspect detection and sentiment prediction are learned separately.

- Single-stage joint discriminative:
  \[
  (\hat{z}, \hat{s}) = h_\psi(x)
  \]
  where one model predicts both outputs together.

- Prompt-based structured generation:
  \[
  (\hat{z}, \hat{s}) = G_{\text{LLM}}(x;\pi,D)
  \]
  where \(\pi\) is an instruction template and \(D\) is either empty, fixed labeled demonstrations, diverse demonstrations, or retrieved demonstrations.

- Two-pass prompted ABSA:
  \[
  \hat{z} = G_{\text{det}}(x), \qquad \hat{s} = G_{\text{sent}}(x,\hat{z})
  \]
  which mirrors the supervised two-step pipeline.

- Aspect-by-aspect prompted ABSA:
  \[
  \hat{z}_k = G^{(k)}_{\text{pres}}(x), \qquad \hat{s}_k = G^{(k)}_{\text{pol}}(x) \text{ if } \hat{z}_k = 1
  \]
  which asks each aspect independently and then predicts sentiment only where the aspect is present.

### Current evidence boundary

- Executed production benchmark:
  - `tfidf_two_step`
  - `distilbert-base-uncased`
  - `bert-base-uncased`
  - `albert-base-v2`
  - `roberta-base`
  - `bert_joint`
  - `distilbert_joint`
- Executed prompt baselines on the full held-out test split:
  - schema-constrained batch `zero-shot` with `gpt-5.2`
  - schema-constrained batch `few-shot` with `gpt-5.2`
  - schema-constrained batch `few-shot-diverse` with `gpt-5.2`
  - schema-constrained batch `retrieval-few-shot` with `gpt-5.2`
- Executed robustness analyses:
  - three-seed stability for `tfidf_two_step`, `distilbert-base-uncased`, and `bert-base-uncased`
  - targeted lower-rate longer-budget tuning for `distilbert-base-uncased` and `bert-base-uncased`
- Implemented but not fully executed:
  - `two-pass`
  - `aspect-by-aspect`

### Artifacts

- Machine-readable matrix: `paper/outputs/tables/benchmark_matrix.csv`
- Paper-ready matrix: `paper/outputs/tables/benchmark_matrix.md`
- Executed local benchmark results: `paper/outputs/tables/combined_local_benchmark_with_joint.csv`
- Full-test GPT results: `paper/benchmark_outputs/openai_batch_eval_summary.csv`
- Robustness tables: `paper/outputs/tables/seed_stability_summary.csv`, `paper/outputs/tables/joint_vs_two_step_summary.csv`, `paper/outputs/tables/tuned_training_budget_summary.csv`
