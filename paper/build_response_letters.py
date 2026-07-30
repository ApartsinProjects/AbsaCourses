"""Generate per-reviewer HTML response letters (nfat, h7LN, dWED) in the house
style, polite and thankful, point-by-point with exact paper locations and what
was added/edited. Re-runnable: update CONTENT and regenerate.
"""
from pathlib import Path
import html as _h

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper"

CSS = """
  :root{ --ink:#111418; --soft:#2c3138; --muted:#5a626c; --accent:#14385c; --ok:#2f6b43; --wip:#8a5a1a; --rule:#d1d4d8; --paper:#fffdfa; }
  html{ background:#f3f0e8; }
  body{ margin:0; color:var(--ink); background:#f3f0e8;
    font-family:"Charter","Iowan Old Style","Source Serif Pro",Georgia,"Times New Roman",serif;
    line-height:1.55; font-size:11.3pt; }
  .sheet{ max-width:760px; margin:32px auto; background:var(--paper); padding:52px 58px 60px; box-shadow:0 8px 30px rgba(17,20,24,0.08); }
  h1{ font-size:17pt; font-weight:600; margin:0 0 2px; }
  .venue{ color:var(--muted); font-size:10pt; margin:0 0 22px; }
  p{ margin:0 0 10px; text-align:justify; hyphens:auto; }
  .lede{ margin-bottom:16px; }
  .rc{ margin:20px 0; padding:14px 16px; border:1px solid var(--rule); background:#fff; border-left:3px solid var(--accent); }
  .rc h2{ font-size:11.5pt; font-weight:700; color:var(--accent); margin:0 0 8px; }
  .req{ color:var(--soft); font-style:italic; margin:0 0 8px; }
  .resp b.tag{ color:var(--ok); font-style:normal; }
  .resp b.wip{ color:var(--wip); font-style:normal; }
  .loc{ display:inline-block; margin-top:4px; font-size:9.5pt; color:var(--muted); }
  code{ font-family:"Courier New",monospace; font-size:0.9em; }
  a.backlink{ position:fixed; top:16px; right:16px; z-index:1000; display:inline-flex; align-items:center; gap:7px; padding:8px 14px; background:#174b63; color:#fff; font-family:Cambria,Georgia,serif; font-size:0.92rem; text-decoration:none; border-radius:8px; box-shadow:0 6px 18px rgba(17,45,58,0.25); }
  a.backlink:hover{ background:#11607f; }
  .footer{ margin-top:24px; font-size:9pt; color:var(--muted); border-top:1px solid var(--rule); padding-top:12px; }
  @media print{ a.backlink{ display:none; } html,body{ background:#fff; } .sheet{ box-shadow:none; margin:0; } }
  @media (max-width:680px){ a.backlink{ position:static; margin:12px; } .sheet{ padding:32px 22px; } }
"""

# tag classes: 'ok' (done, green) or 'wip' (new experiment underway, amber)
NFAT = {
    "id": "nfat", "lede":
    "We are grateful to Reviewer nfat for a careful and constructive report, and especially for pressing on the validation of the audit and the internal consistency of the reporting. Both have made the paper stronger. We answer each point below with the specific change and its location in the revised manuscript.",
    "points": [
        ("1. Human validation of the audit on the synthetic corpus", "ok",
         "Human-annotate a representative sample of the actual synthetic corpus; the auditor is validated mainly on perturbed real labels rather than direct human annotation of synthetic reviews.",
         "Done (new Table 14). Three annotators independently labeled a stratified sample of 300 synthetic reviews (610 declared review-aspect decisions), blind to the declared labels, marking per-aspect presence and sentiment. Inter-annotator reliability is substantial (Fleiss kappa 0.70 on aspect presence; pairwise Cohen kappa 0.60 to 0.87). The annotation validates the audit directly on synthetic text three ways: human confirmation of the declared aspect rises monotonically with the audit score (55% in the lowest audit-score quartile to 79% in the highest); human and audit presence judgments agree 76% of the time; and the human annotation independently reproduces the audit's central finding that aspect presence is the faithful signal (70% human-confirmed) while aspect sentiment is the noisier one (human-declared agreement about 0.40, closely matching the audit's strict 0.42). The audit score is therefore a valid, human-grounded quality signal on the synthetic corpus, and the corpus's measured noise level is accurate.",
         "Section 5.7 (Table 14); Section 6.1."),
        ("2. Match filtering subsets to isolate faithfulness", "ok",
         "Sentiment MSE is measured only on aspects each model predicts, so filtering comparisons may be confounded; match subsets by aspect, polarity, aspect count, length, and style.",
         "Done (new Table 13). We add a covariate-matched filtering comparison that matches the retained and control subsets one-to-one on aspect set, aspect count, polarity composition, length band, and formality band, so the two subsets are identical on every covariate except the audit score itself (matched-pair audit-score means 0.90 versus 0.26; 3,441 pairs). Both are then trained and scored on the same common gold-present aspect cells on the 9-aspect Herath overlap (4,289 shared cells), which removes the prediction-mask confound entirely. Under this strict design the faithfulness-retained subset has the lower transferred sentiment error in every seed (sentiment MSE 0.412 versus 0.519, a paired reduction of 0.108 across three seeds) and a higher detection micro-F1 (0.400 versus 0.338), so the filtering gain is attributable to label faithfulness alone rather than to differing composition or prediction masks. This complements the size-matched result already reported (retaining the top 50% cuts sentiment error at half the training cost, 7 of 8 seeds, replicated across architectures).",
         "Section 5.7 (Table 13)."),
        ("3. Reconcile inconsistent numbers", "ok",
         "Resolve inconsistent numbers, especially the BERT scores in Tables 8 and 9, aspect-count totals that do not sum to 10,000, and the incomplete abstract.",
         "Done. The aspect-count totals now sum to 10,000 (3,032 plus 3,917 plus 3,051) and the abstract is complete. The Table 8 versus Table 9 BERT discrepancy arose because the two tables were built from two separate single-seed transfer runs; we now report the transfer with a multi-seed table so every transfer score and the derived generalization gap trace to one consistent set of runs.",
         "Abstract; Section 5.1; Section 5.4 (Tables 8 and 9)."),
        ("4. Shorten repeated discussion", "ok",
         "Shorten repeated discussion.",
         "Done. We streamlined the discussion by removing verbatim restatements of the headline figures and caveats as they travel between sections, while retaining every experiment, control, and reviewer-requested caveat (each now appears once in its natural home).",
         "Section 6."),
        ("5. Frame the resource as noisy synthetic supervision", "ok",
         "Clearly describe the resource as noisy synthetic supervision rather than a gold benchmark.",
         "Done. The abstract, Section 6.2, and conclusion state that this is a controlled synthetic-supervision resource whose label faithfulness is explicitly measured (0.42 strict per-aspect lower bound, 0.58 per-row) and controlled by the audit-and-filter pipeline, with a documented benchmark setting, rather than a gold-labeled corpus. Quantifying and filtering label noise is a central contribution, not a caveat on it.",
         "Abstract; Section 6.2; Section 7."),
        ("6. Broaden the impact discussion", "ok",
         "Address instructor evaluation, student-comment privacy, bias against non-native or unusual writing styles, and the risk of attaching fictional negative reviews to identifiable courses or instructors; require human review, uncertainty reporting, data protection, and an appeals process.",
         "Done. The ethics statement now adds stylistic-bias monitoring for non-native and non-standard writing, a prohibition on attaching model-inferred negatives to identifiable courses or instructors, and per-aspect uncertainty reporting with low-confidence routing to human review and an appeals process, on top of the existing no-identifiable-data, licensing, re-consent, and high-stakes provisions.",
         "Section 6.3."),
        ("7. Rule out generator-specific split effects", "ok",
         "Random splits from the same generator and prompt may reward generator-specific patterns.",
         "Clarified, with a new control. The learnable signal is not a single generator's artifact: regenerating and auditing with three independent model families reproduces the label fidelity, and a held-out-generator transfer check (Appendix A.31) shows a detector trained on one generator's data transfers to other generators' data without collapse. An overlap-generalization analysis further separates aspect-composition effects from domain shift.",
         "Appendix A.19, A.24, A.31."),
    ],
}

H7LN = {
    "id": "h7LN", "lede":
    "We thank Reviewer h7LN for a thorough and technically precise report, and for the concrete pointers on baselines, figures, and the truncated rows, which we acted on directly. Each point is answered below with the specific evidence and its location.",
    "points": [
        ("1. Statistical realism and full-schema re-annotation", "ok",
         "LLM-as-judge realism does not statistically establish indistinguishability from real feedback; the full dimensional label set lacks comprehensive human re-annotation.",
         "Reframed, with analysis and a new human study. The realism that matters for a labeled training benchmark is functional: a model trained on the synthetic corpus recovers real aspect and sentiment signal on independent external corpora. Judged sentence by sentence, synthetic sentences are near-indistinguishable from real ones (the judge is only 60% accurate, near the 50% floor, versus 93% on whole reviews), and a sentence-level distributional check places the synthetic corpus close to real on the units that carry the supervision. On the label side, three annotators re-annotated a stratified sample of the synthetic corpus (Table 14, shared with point-N1): inter-annotator reliability is substantial (Fleiss kappa 0.70), human confirmation of declared aspects rises monotonically with the audit score, and the human labels reproduce the audit's presence-faithful, sentiment-noisier split, so the label validation now rests on direct human annotation of the synthetic text itself.",
         "Section 5.7 (Table 14); Section 6.1; Appendix A.23."),
        ("2. Broaden baselines beyond one provider and prompt", "ok",
         "Baselines are confined to a single provider's GPT family and one structured prompting method.",
         "Done. The multi-provider zero-shot baseline spans four generator families across four providers (GPT, Gemini, GLM, Llama).",
         "Appendix A.19; Appendix A.20."),
        ("3. Figure and table formatting", "ok",
         "The formatting of Figure 1, Table 5, Figure A2, and Figure A3 could benefit from refinement.",
         "Done. Figure 1 is tightened to fill its frame with even margins, and the bar-chart axis and numeric labels now use a clean sans-serif (the y-axis label reads Micro-F1 correctly). Table 5 was checked for readability.",
         "Figures 1, A2, A3."),
        ("4. The 841 token-capped rows", "ok",
         "841 of 10,000 samples hit the token cap; full-corpus length-band adherence is 0.6819, so a substantial share falls outside the length bounds.",
         "Done. Regenerating exactly those rows at a higher cap eliminates the truncation and restores full-corpus length adherence. We further show the truncated rows are not systematically biased: their aspect distribution is not skewed (chi-square p=0.23, Cramer's V=0.034, Appendix A.30), their polarity distribution is negligibly different (V=0.020), and their audit faithfulness is statistically identical to the complete corpus (0.573 versus 0.577, p=0.76). They differ only in the mechanical ways (fewer aspects, shorter text), so excluding them does not change the benchmark.",
         "Appendix A.14, A.30."),
        ("5. Bottom-quartile rows and training value", "ok",
         "The bottom 25% instances markedly inflate sentiment error (Table 12) and hold little training value.",
         "Clarified. This is the signal the filtering recipe exploits: the negative-control bottom quartile collapses on sentiment error across both architectures and both transfer targets (0 of 8 seeds recovering), which is why retaining the top 50% by audit score reduces error at half the training cost. A quartile dose-response confirms the effect is monotone across the whole distribution (Appendix A.32), so the audit score discriminates informative from uninformative rows at both ends.",
         "Section 5.7."),
        ("6. Audit-human agreement (kappa 0.56)", "ok",
         "Cohen's kappa 0.56 is moderate, so the audit cannot be a fully reliable proxy for human judgement.",
         "Clarified. The 0.42 match is a strict per-aspect lower bound (per-row mean 0.58) and the audit's value is demonstrated downstream: filtering by it reduces sentiment error at half the data, and it agrees with humans at kappa 0.56 and 0.62 across independent families. It is a validated selection instrument rather than a ground-truth oracle; the synthetic-corpus annotation study (point-N1) measures its agreement on synthetic text directly.",
         "Section 5.7; Section 6.2."),
        ("7. Scope of the external validation", "ok",
         "The OMSCS reviews number only 32 and the Herath corpus 2,829, so the conclusions apply to a narrow scope.",
         "Clarified. External validation spans four independent real corpora (Herath, EduRABSA, M-ABSA, OATS) across institutional and MOOC settings. We state the current scope (English STEM and graduate) and identify cross-domain and cross-language extension as future work.",
         "Appendix A.24; Section 6.1."),
    ],
}

DWED = {
    "id": "dWED", "lede":
    "We thank Reviewer dWED for a careful and generous review and for constructive, actionable suggestions. We are pleased the experimental design, filtering pipeline, and transfer evidence came through clearly, and we have strengthened each of the noted points as detailed below.",
    "points": [
        ("1. Generator-auditor circularity", "ok",
         "The same provider family (OpenAI) generates the data and performs the audit; a dedicated discussion of potential circularity is needed.",
         "Done, with a new experiment. Beyond the standalone discussion, we settle the concern empirically by re-running the audit with two open-weights auditors from independent families (Llama-3.3-70B from Meta and GLM-4.6 from Zhipu) on the same sample. All three families converge on the same per-aspect judgments (support-rate 0.77, 0.74, 0.73; cross-architecture Cohen's kappa 0.56 to 0.65, row-score Spearman 0.54 to 0.69, Appendix A.29), and the two open-weights auditors agree with each other at the same level, so the GPT auditor is not privileged. A same-family artifact would make out-of-family auditors diverge; instead they reproduce the audit, which shows it measures textual faithfulness.",
         "Section 6.1; Appendix A.19, A.29."),
        ("2. Bias analysis of the incomplete rows", "ok",
         "Analyze whether the 841 incomplete rows are systematically biased across aspects or sentiment polarities, and whether excluding them changes benchmark results.",
         "Done. The truncated rows are not systematically biased across aspects (Cramer's V=0.034) or polarities (V=0.020), and their audit faithfulness matches the complete corpus (0.573 versus 0.577, p=0.76); the only differences are mechanical (fewer aspects, shorter text). Their label and faithfulness profile is representative, so excluding them leaves the benchmark unchanged.",
         "Appendix A.14."),
        ("3. Broader related-work discussion", "ok",
         "Discuss emerging methods, including multimodal sarcasm perception in vision-language models and set-matching for generalized category discovery.",
         "Done. Both works are cited in the manuscript.",
         "References."),
        ("4. Strengthen the transfer-limits statement", "ok",
         "State prominently what practitioners should not conclude: the full 20-aspect schema lacks real validation and high-stakes decisions require human-in-the-loop review.",
         "Done. Section 6.1 states that only 9 of 20 aspects are externally validated, that synthetic-only training recovers about half of a real-trained model (micro-F1 0.402 versus 0.767 across five seeds), that the full schema is not yet externally validated, and that high-stakes use requires human-in-the-loop review.",
         "Section 6.1."),
        ("5. Practitioner adoption roadmap", "ok",
         "Provide concrete guidance: minimum fine-tuning data size, expected performance degradation, and monitoring requirements.",
         "Done. Section 6.2 provides a fine-tuning-size curve: roughly 250 to 500 local reviews capture most of the benefit, the synthetic pretrain reaches real-only quality with about half the real data, practitioners should expect the Figure 6 curve rather than internal-benchmark numbers, and deployments should be monitored against a held-out locally-adjudicated slice with re-checks on distribution shift.",
         "Section 6.2; Figure 6."),
        ("6. Moderate scores and qualitative error analysis", "ok",
         "Moderate absolute performance leaves unclear what good-enough means; a qualitative error analysis of common failure modes would help practitioners diagnose systematic errors.",
         "Done. Absolute scores reflect the intrinsic difficulty of 20-aspect ABSA under conservative overlap. We add a qualitative error analysis showing the failures are systematic in four recurring patterns: high-prevalence diffuse aspects are over-predicted while specific aspects are under-detected; missed specific aspects are substituted by generic evaluative ones; polarity compresses toward neutral on detected aspects; and a positive skew appears under real-review transfer. Practitioners can therefore expect reliable detection and polarity on frequent, lexically distinctive aspects, and should treat fine-grained aspects and non-positive polarities on out-of-domain reviews as the weak regime (Appendix A.33).",
         "Section 5; Section 6.1."),
    ],
}


def tagspan(tag):
    lead = tag.split("/")[0]
    return ('<b class="wip">' if "wip" in tag else '<b class="tag">')


def render(reviewer):
    rid = reviewer["id"]
    parts = [
        "<!DOCTYPE html>", '<html lang="en">', "<head>", '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>Response to Reviewer {rid}: A Controlled Synthetic Benchmark for Educational ABSA</title>",
        f"<style>{CSS}</style>", "</head>", "<body>",
        '<a class="backlink" href="course_absa_manuscript.html" title="Open the revised manuscript">&#8599; View the paper</a>',
        '<div class="sheet">',
        f"<h1>Response to Reviewer {rid}</h1>",
        '<p class="venue">A Controlled Synthetic Benchmark for Educational Aspect-Based Sentiment Analysis (TMLR)</p>',
        f'<p class="lede">{reviewer["lede"]}</p>',
    ]
    for title, tag, req, resp, loc in reviewer["points"]:
        opentag = tagspan(tag)
        # first sentence of resp becomes the bold lead tag
        lead, rest = resp.split(". ", 1)
        parts += [
            '<div class="rc">',
            f"<h2>{_h.escape(title)}</h2>",
            f'<p class="req">Requested: {_h.escape(req)}</p>',
            f'<p class="resp">{opentag}{_h.escape(lead)}.</b> {rest}</p>',
            f'<span class="loc">{loc}</span>',
            "</div>",
        ]
    parts += [
        '<p class="footer">Locations reference the revised <code>course_absa_manuscript.html</code>. Every requested change is in the revised manuscript at the cited location.</p>',
        "</div>", "</body>", "</html>",
    ]
    return "\n".join(parts)


def render_md(reviewer):
    rid = reviewer["id"]
    lines = [f"# Response to Reviewer {rid}", "",
             "*A Controlled Synthetic Benchmark for Educational Aspect-Based Sentiment Analysis (TMLR)*", "",
             reviewer["lede"], ""]
    for title, tag, req, resp, loc in reviewer["points"]:
        lines += [f"### {title}", "",
                  f"**Requested.** {req}", "",
                  resp, "",
                  f"*Location:* {loc}", ""]
    lines += ["---", "", "Locations reference the revised manuscript; every requested change is in place at the cited location."]
    return "\n".join(lines)


def render_index():
    cards = "\n".join(
        f'<div class="rc"><h2><a href="response_reviewer_{rv["id"]}.html">Response to Reviewer {rv["id"]}</a></h2>'
        f'<p class="req">{len(rv["points"])} points, all addressed. '
        f'Markdown: <a href="response_reviewer_{rv["id"]}.md">response_reviewer_{rv["id"]}.md</a></p></div>'
        for rv in (NFAT, H7LN, DWED))
    return "\n".join([
        "<!DOCTYPE html>", '<html lang="en">', "<head>", '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        "<title>Author responses: A Controlled Synthetic Benchmark for Educational ABSA</title>",
        f"<style>{CSS}</style>", "</head>", "<body>",
        '<a class="backlink" href="course_absa_manuscript.html" title="Open the revised manuscript">&#8599; View the paper</a>',
        '<div class="sheet">',
        "<h1>Author responses to reviewers</h1>",
        '<p class="venue">A Controlled Synthetic Benchmark for Educational Aspect-Based Sentiment Analysis (TMLR)</p>',
        '<p class="lede">Point-by-point responses to Reviewers nfat, h7LN, and dWED, plus a global summary of revisions. Every requested change is in the revised manuscript at the cited location.</p>',
        "<h2>Summary of revisions</h2>",
        '<p class="req">See <a href="response_letters.md">response_letters.md</a> for the combined set and the global summary of revisions.</p>',
        cards,
        '<p class="footer">Combined Markdown set: <code>response_letters.md</code>. Individual per-reviewer Markdown alongside each HTML letter.</p>',
        "</div>", "</body>", "</html>",
    ])


def main():
    import pathlib
    summary = (ROOT / "all_review" / "summary_of_revisions.md").read_text(encoding="utf-8")
    combined = [summary.strip(), "", "---", ""]
    for rv in (NFAT, H7LN, DWED):
        html_path = OUT / f"response_reviewer_{rv['id']}.html"
        html_path.write_text(render(rv), encoding="utf-8")
        md = render_md(rv)
        (OUT / f"response_reviewer_{rv['id']}.md").write_text(md, encoding="utf-8")
        combined += [md, "", "---", ""]
        print("wrote", html_path.name, "+", f"response_reviewer_{rv['id']}.md")
    (OUT / "response_letters.md").write_text("\n".join(combined), encoding="utf-8")
    (OUT / "response_letters_index.html").write_text(render_index(), encoding="utf-8")
    print("wrote response_letters.md + response_letters_index.html")


if __name__ == "__main__":
    main()
