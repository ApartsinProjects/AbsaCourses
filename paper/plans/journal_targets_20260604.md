# Journal targets for the AbsaCourses paper

Date: 2026-06-04
Status: shortlist, decision-ready
Scope: Q1/Q2 venues with high acceptance probability for the
current draft (synthetic ABSA benchmark + 9-aspect Herath transfer
+ §6.7D faithfulness-aware filtering result).

Verified against journal author-fee pages and Scimago quartile data
as of June 2026; SJR values may shift by ±0.05 on the next refresh.

## Recommended submission ladder

1. **TMLR** (Transactions on Machine Learning Research)
   - APC: **$0** (Diamond OA via JMLR)
   - Quartile: DOAJ-listed, JMLR-affiliated, treated as Q1-equivalent
     in ML; not yet JCR-indexed.
   - Why first: TMLR's stated acceptance criterion is "claims are
     well-supported by accurate, convincing, and clear evidence",
     which matches the §6.7D paired bootstrap CI excluding zero across
     four seeds plus the multi-seed-discipline framing. OpenReview-
     style transparent review; no "novelty bar"; 6-8 week decision
     window typical.
   - Pre-submission: Action-Editor request, OpenReview profile,
     TMLR LaTeX template (differs from generic camera-ready).
   - Site: https://jmlr.org/tmlr/

2. **Computers and Education Open** (Elsevier, born-OA)
   - APC: ~$1,670
   - Quartile: Q1 (SJR 1.678, 2024)
   - Why second: direct topical fit (AI in education) with explicit
     scope for benchmark contributions. Faster review than the
     flagship Computers & Education. Reasonable APC.
   - Site: https://www.elsevier.com/journals/computers-and-education-open/

3. **PeerJ Computer Science**
   - APC: ~$1,395 (or via membership)
   - Quartile: Q1 (SJR 0.719)
   - Why: accepts CS benchmark + dataset contributions; open peer
     review; lower bar than the top NLP journals.
   - Site: https://peerj.com/computer-science/

4. **Computers & Education** (Elsevier, subscription track)
   - APC: **$0** if the subscription publication track is taken
   - Quartile: Q1 (SJR 3.343, 2024)
   - Why fallback only: top ed-tech venue, but reviewers expect
     pedagogical / learning-outcome contribution beyond the corpus.
     §7.2 pedagogical use cases helps but is not on its own a
     learning-outcome study; reject probability is non-trivial.
   - Site: https://www.sciencedirect.com/journal/computers-and-education

5. **Education and Information Technologies** (Springer, subscription track)
   - APC: **$0** if the subscription track is taken
   - Quartile: Q1 (SJR 1.654)
   - Why fallback: hybrid still has a no-fee subscription option;
     reviewers expect either learning-outcome data or an applied
     case study.

## High-fit, higher-APC venues

| Journal | Quartile | APC | Note |
|---|---|---:|---|
| Computers & Education: AI | Q1 (SJR 5.217) | ~$1,800 | Tightest topical match; rising IF |
| Natural Language Processing (Cambridge, ex-NLE) | Q1 Ling., Q2 AI | $3,655 | Strong applied-NLP + corpora fit |
| IJAIED (Springer until Jan 2026, then Elsevier) | Q1 (IF 8.5) | ~$3,190 | Premier AIED venue; editorial transition |
| Expert Systems with Applications | Q1 (SJR 1.854) | $3,490 | Applications-oriented |
| Knowledge-Based Systems | Q1 (SJR 1.934) | $3,350 | Strong CS-AI tier |

## Avoid

- **Education Sciences (MDPI)** — Q1 nominal, but special-issue
  volume model and reputation risk outweigh the prestige signal.
- **IEEE Access** — Q1/Q2 multidisciplinary, $2,160 APC, declining
  reputation; generic acceptance signal.
- **Computational Linguistics (MIT Press / ACL)** — Diamond OA and
  $0 APC, but review cycle of 9-12 months and a stronger linguistic-
  analysis bar than this paper carries.

## One-line strategy

Submit to **TMLR** first (free, fast, designed for this exact paper
profile). If rejected on scope, fall to **Computers and Education
Open** ($1,670, Q1, born-OA, direct ed-tech fit).
