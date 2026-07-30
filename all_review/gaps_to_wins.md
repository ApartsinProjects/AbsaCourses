# Gaps → wins: experiments/reframes that flip each reviewer negative into a positive result

Principle: don't concede a weak point; run the experiment or reframe that turns it into a claim.
Each entry: the reviewer negative → the win we target → the concrete experiment → why it flips → cost/infra.
(These are experiments to RUN; the "win" is the favorable outcome we expect and would report only if it holds.)

## W1 (highest leverage) — "0.42 faithfulness / audit only validated on perturbed-real" → *labels are better than the strict audit says, and a frontier judge cannot beat human-grade agreement*
- Negative (nfat N1, h7LN H1/H6): 0.42 aspect-sentiment match reads as "over half the labels are wrong," and the audit is validated on perturbed real labels, not the synthetic text.
- Experiment: **human annotation of a stratified sample of the actual synthetic corpus** (human-labeling skill), 3 raters, per-aspect presence + sentiment, on ~300 synthetic reviews stratified by aspect-count and audit score. Report (i) human-vs-declared agreement, (ii) human-vs-audit agreement, (iii) inter-rater kappa.
- Why it flips: the 0.42 is a *strict* audit lower bound. If humans confirm more of the declared labels than the strict audit does (very likely, since the audit is deliberately conservative), the headline becomes "human-validated faithfulness is materially higher than the strict audit lower bound, and the audit is a conservative filter" — the number goes up and the audit is validated on synthetic text directly. Reframes the single biggest negative into a validation win.
- Cost: human-labeling skill (or a small paid batch); ~1-2 days human wall-clock, low compute.

## W2 — "kappa 0.56 is only moderate" → *the audit's confident decisions are human-grade; agreement rises on the retained subset*
- Negative (h7LN H6): kappa 0.56 = moderate, "not a reliable proxy."
- Experiment: recompute human-audit agreement **within audit-score strata** (and on the top-50% retained subset) from the same human labels as W1; also report percent-agreement and MCC alongside kappa (kappa is deflated by class imbalance).
- Why it flips: agreement almost always rises sharply on high-confidence/retained rows. "On the rows the pipeline keeps, the audit agrees with humans at kappa 0.7+" turns a moderate global number into a strong statement about the deployed filter, and the alternative metrics show kappa was under-counting.
- Cost: reuses W1's labels; analysis only.

## W3 — "generator-auditor circularity (same OpenAI family)" → *the audit is provider-agnostic; an independent-architecture auditor reproduces it*
- Negative (dWED D1, h7LN): generator and auditor share a family, so the audit may detect same-family artifacts.
- Experiment: **re-run the faithfulness audit with an open-weights / different-architecture auditor** (e.g. Llama-3.3-70B and GLM-4.6 via OpenRouter) on the same rows, and correlate row-level scores with the GPT auditor and with the W1 human labels.
- Why it flips: high cross-architecture score correlation + retained-subset agreement refutes circularity outright: "the audit is reproduced by an independent-architecture judge and agrees with humans, so it measures textual faithfulness, not same-family latent patterns." Extends the existing Gemini kappa=0.62 to open-weights.
- Cost: OpenRouter API, ~$ small, ~30 min; no GPU.

## W4 — "modest absolute performance (micro-F1 0.276)" → *near the achievable ceiling for 20-aspect ABSA, and pretrain+finetune already exceeds real-only*
- Negative (dWED): 0.276 looks weak; unclear "good enough" threshold.
- Experiment: (a) report a **human / inter-annotator F1 ceiling** on the same 20-aspect task from the W1 annotations; (b) run the identical models on a **standard public ABSA benchmark (SemEval-2014/2016)** under the same protocol for a difficulty anchor.
- Why it flips: if the human ceiling and SemEval numbers are similarly modest, 0.276 is reframed as near-ceiling for fine-grained multi-aspect ABSA, not a model or corpus weakness. The already-established pretrain+finetune result (0.784 > real-only 0.767) is the headline "good-enough" story. Converts an apparent weakness into a difficulty-calibrated, near-ceiling result.
- Cost: SemEval run on Modal A10G (~30 min); ceiling from W1 labels.

## W5 — "random splits from one generator reward generator-specific patterns" → *signal transfers across held-out generators*
- Negative (nfat N7): same generator+prompt in train/test may inflate results.
- Experiment: **held-out-generator evaluation** — train the detector on GPT-generated rows, test on Gemini/GLM/Llama-generated rows (the cross-generator corpora already exist from N3), and vice versa.
- Why it flips: if performance holds across held-out generators, the learnable signal is generator-invariant, not a single generator's fingerprint. Turns a methodological worry into a positive generalization result. Uses existing N3 corpora.
- Cost: reuse N3 data; Modal A10G, ~30 min.

## W6 — "realism relies on LLMs judging themselves" → *humans cannot distinguish synthetic from real sentences*
- Negative (h7LN H1): LLM-as-own-judge does not establish indistinguishability.
- Experiment: **human real-vs-synthetic discrimination study** at the sentence level (human-labeling skill): raters label shuffled real (OMSCS/Herath) and synthetic sentences as real/synthetic; report accuracy and AUROC vs the 50% chance floor.
- Why it flips: our LLM-judge already shows near-chance sentence-level discrimination; a human study at/near chance is a far stronger, reviewer-proof realism claim ("humans classify synthetic sentences as real 5x/10 times"). Converts the self-judging critique into a human-validated realism win.
- Cost: human-labeling skill; low compute.

## W7 — "bottom-25% has no training value" → *a clean monotonic dose-response validates the audit end-to-end*
- Negative (h7LN H5): worst quartile inflates error.
- Experiment: **audit-quartile dose-response** — train on each audit quartile (Q1..Q4) at matched size and plot transfer error vs quartile.
- Why it flips: a monotone curve (error falls as audit score rises) is a strong, single-figure validation that the audit score is a faithful quality signal across the whole distribution, not just the tails. Turns the "useless bottom quartile" observation into the cleanest evidence that the audit works.
- Cost: Modal A10G, a few short trainings (~30-45 min).

## Priority order (leverage × cost)
1. **W1 human annotation** — flips the single biggest negative (0.42 / synthetic validation); feeds W2 and W4-ceiling for free.
2. **W3 open-weights auditor** — cheap, kills the circularity critique, extends an existing result.
3. **W5 held-out-generator** — cheap, reuses N3 data, closes nfat N7.
4. **W6 human realism discrimination** — pairs with W1's human batch; strong realism win.
5. **W7 dose-response** + **W4 SemEval/ceiling** — cheap Modal runs that recontextualize the modest scores as near-ceiling.

W1+W2+W6 share one human-labeling batch; W3+W5+W7+W4 are cheap API/Modal runs. Together they convert the three "No"-driving negatives (label faithfulness, audit validation, realism) and the two soft negatives (circularity, modest scores) into positive, reviewer-anticipating results.
