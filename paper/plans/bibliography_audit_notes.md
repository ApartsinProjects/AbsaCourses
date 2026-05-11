# Bibliography Audit Notes

Date: 2026-05-11

This document records corrections applied to the manuscript's bibliography after
a second-pass audit. The earlier verification artifact at
[bibliography_verification_report.csv](/paper/bibliography_verification_report.csv)
checked URLs and titles against Crossref/OpenAlex but did **not** verify author
names, author order, or publication year/volume. This audit closed that gap by
spot-checking each entry's author list against Crossref and the published landing
page, and corrected six entries.

## Corrections applied to `paper/course_absa_manuscript.html`

| ref # | field         | before                                                                                          | after                                                                                                       | source                            |
|------:|---------------|-------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------|-----------------------------------|
|     1 | author 1      | `Denise M. Gikandi`                                                                             | `J.W. Gikandi`                                                                                              | Crossref `10.1016/j.compedu.2011.06.004` |
|    11 | authors       | `Kushan Nilanga, Missaka Herath, Hashan Maduwantha, and Surangika Ranathunga`                   | `Missaka Herath, Kushan Chamindu, Hashan Maduwantha, and Surangika Ranathunga`                              | Crossref title search; matches the Herath et al. 2022 release in `external_data/Student_feedback_analysis_dataset/README.md` |
|    16 | author 1      | `Maria Misuraca`                                                                                | `Michelangelo Misuraca`                                                                                     | Crossref `10.1016/j.stueduc.2021.100979` |
|    16 | author 2      | `Matteo Giampà`                                                                                 | `Germana Scepi`                                                                                             | Crossref `10.1016/j.stueduc.2021.100979` |
|    16 | author 3      | `Daniela Cacciaguerra`                                                                          | `Maria Spano`                                                                                               | Crossref `10.1016/j.stueduc.2021.100979` |
|    16 | year          | `2022`                                                                                          | `2021`                                                                                                      | Crossref                          |
|    16 | DOI in link   | `10.1016/j.stueduc.2022.101177`                                                                 | `10.1016/j.stueduc.2021.100979`                                                                             | the original DOI pointed at "Adapting early childhood education interventions to contexts" (Trevino and Godoy, 2022). |
|    16 | volume/page   | `75, 101177`                                                                                    | `68, 100979`                                                                                                | Crossref                          |
|    17 | authors       | `Thanveer Shaik, Xiaohui Tao, Christopher Dann, Petrea Redmond, and Linda Galligan`             | `Thanveer Shaik, Xiaohui Tao, Christopher Dann, Haoran Xie, Yan Li, and Linda Galligan`                     | Crossref `10.1016/j.nlp.2022.100003` |
|    17 | URL           | `https://www.sciencedirect.com/science/article/pii/S2949719123000122`                           | `https://doi.org/10.1016/j.nlp.2022.100003`                                                                 | the prior pii redirected away; the canonical DOI is more stable. |
|    17 | volume/page   | `3, 100012`                                                                                     | `2, 100003`                                                                                                 | Crossref                          |
|    19 | author 1      | `Clifton M. Edwards`                                                                            | `Aleksandra Edwards`                                                                                        | Crossref title search; ACL Anthology landing page |
|    22 | authors       | `Michael Henderson, Rola Ajjawi, David Boud, Elizabeth Molloy, and Hannah Sutton`               | `Michael Henderson, Michael Phillips, Tracii Ryan, David Boud, Phillip Dawson, Elizabeth Molloy, and Paige Mahoney` | Crossref `10.1080/07294360.2019.1657807` |

## Entries spot-checked and confirmed correct

References 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 18, 20, 21, 23, 24, 25, 26,
27, 28, 29, 30. For each, the first author and a representative middle author were
matched against Crossref or the canonical ACL Anthology landing page.

## What the original verifier missed

The verification script verified title equality against Crossref and OpenAlex but
treated the `verified` flag as satisfied as long as the URL resolved and the
title matched. Reference 16 in particular reached `verified` even though
Crossref returned a completely different paper at the listed DOI; the
`official_title` column did record the mismatch but the `status` field did not
flag it. A follow-up improvement to `paper/verify_bibliography.py` should:

- compare the first author family name to the Crossref `author[0].family` field;
- treat any pairwise title difference greater than a small edit distance as a
  warning rather than as verified;
- flag DOI->title mismatches explicitly.

These changes are out of scope for this audit and tracked here as future work.

## Verification source

All Crossref lookups in this audit went through
`https://api.crossref.org/works/<DOI>` and
`https://api.crossref.org/works?query.title=...`.
ACL Anthology lookups went through `https://aclanthology.org/<id>/` directly.
