# 20-aspect pedagogical schema

The synthetic corpus is labeled over 20 aspects grouped into five pedagogical blocks
(Appendix A.1 of the paper). Each declared aspect in `synthetic_corpus_10k.jsonl`
carries a sentiment in {positive, neutral, negative}.

| Block | Aspect | Description |
|-------|--------|-------------|
| Instructional quality | clarity | How understandable the teaching and explanations feel. |
| Instructional quality | lecturer_quality | Perceived quality of the lecturer or lead instructor. |
| Instructional quality | materials | Usefulness of slides, notes, readings, and resources. |
| Instructional quality | feedback_quality | Usefulness and timeliness of feedback on student work. |
| Assessment and course management | exam_fairness | Whether exams feel aligned and fair. |
| Assessment and course management | assessment_design | Alignment and structure of assignments, projects, and exams. |
| Assessment and course management | grading_transparency | How clearly grading criteria, rubrics, and score interpretation are communicated. |
| Assessment and course management | organization | Administrative clarity, course structure, and coordination. |
| Assessment and course management | tooling_usability | Friction or support created by LMS, submission systems, and required software. |
| Learning demand and readiness | difficulty | Conceptual or technical challenge of the course. |
| Learning demand and readiness | workload | Amount of sustained effort required across the term. |
| Learning demand and readiness | pacing | Whether the course tempo and weekly rhythm are manageable. |
| Learning demand and readiness | prerequisite_fit | How well the course matches the advertised prerequisite level and student preparation. |
| Learning environment | support | Quality of help from instructor, TAs, or forums. |
| Learning environment | accessibility | Perceived accessibility and inclusiveness of materials, pace, and course participation. |
| Learning environment | peer_interaction | Whether peer discussion, teamwork, and class community help or hinder learning. |
| Engagement and value | relevance | Perceived usefulness to the program or future goals. |
| Engagement and value | interest | Level of engagement or curiosity the course creates. |
| Engagement and value | practical_application | Connection to real-world practice or authentic tasks. |
| Engagement and value | overall_experience | Global student impression after tradeoffs. |

The 9-aspect Herath overlap used for external transfer is: accessibility,
assessment_design, exam_fairness, grading_transparency, lecturer_quality, materials,
organization, overall_experience, workload (see `herath_mapping.json`).
