# Codebook: 20-aspect inventory, sentiment values, and edge cases

This codebook is the single source of truth for every human labeling task in this
project. Raters in **Task 1 (realism + faithfulness)**, **Task 2 (Herath
re-annotation)**, and **Task 3 (human vs. LLM-judge agreement)** must all follow it.

If you encounter a case not covered here, default to `unclear` rather than guessing,
and add a one-line note in the row's `notes` column.

## Aspect inventory (20)

The inventory is organized into five pedagogical groups. For each aspect we give a
short definition, what counts as evidence the aspect is **discussed**, and one example
each for positive, neutral, and negative.

### Group 1: Instructional quality

#### 1. `clarity`
How understandable the teaching and explanations feel.
- positive: "her examples made even hard stuff click."
- neutral: "lectures were standard, nothing memorable."
- negative: "i never understood what he was actually trying to teach."

#### 2. `lecturer_quality`
Perceived quality of the lecturer or lead instructor. Distinct from `clarity`: a clear
lecturer may still be unenthusiastic; a likable one may still be unclear.
- positive: "prof was responsive on piazza and clearly cared."
- neutral: "instructor was fine, did the job."
- negative: "he seemed annoyed every time someone asked a question."

#### 3. `materials`
Usefulness of slides, notes, readings, code repos, datasets, software, or recorded
video.
- positive: "the slides alone are basically a textbook."
- neutral: "readings were typical academic papers."
- negative: "half the linked resources were dead links."

#### 4. `feedback_quality`
Usefulness and timeliness of feedback on student work. This is feedback **content**
(graded comments, rubric explanations, tutor remarks), not the existence of office
hours or forums.
- positive: "the rubric comments told me exactly what to fix."
- neutral: "feedback came back, mostly just a score."
- negative: "feedback was vague and arrived after the next deadline."

### Group 2: Assessment and course management

#### 5. `exam_fairness`
Whether exams feel aligned with what was taught and time given, and fair in their
expectations.
- positive: "exams matched what we covered, very fair."
- neutral: "tests were what you'd expect."
- negative: "the midterm had stuff we never saw in class."

#### 6. `assessment_design`
Alignment and structure of assignments, projects, and exams as a system. Distinct
from `exam_fairness`: this is about whether the overall assessment scheme makes
pedagogical sense.
- positive: "the project built nicely on each lecture."
- neutral: "standard mix of homeworks and a final."
- negative: "the project was disconnected from the lecture content."

#### 7. `grading_transparency`
How clearly grading criteria, rubrics, and score interpretation are communicated.
Distinct from `feedback_quality`: this is the published criteria, not the remarks on a
particular submission.
- positive: "the rubric was public and unambiguous."
- neutral: "the grading scheme was on the syllabus."
- negative: "no idea why i lost 15 points; rubric was vague."

#### 8. `organization`
Administrative clarity, course structure, and coordination. Includes syllabus
consistency, deadline coherence, and changes mid-term.
- positive: "everything was on the syllabus, no surprises."
- neutral: "course ran on a normal weekly cadence."
- negative: "deadlines shifted three times with no warning."

#### 9. `tooling_usability`
Friction or support created by LMS, submission systems, gradescope, and required
software. Includes platform reliability and the submission workflow.
- positive: "the submission portal just worked every time."
- neutral: "canvas was canvas, nothing special."
- negative: "the autograder rejected valid submissions repeatedly."

### Group 3: Learning demand and readiness

#### 10. `difficulty`
Conceptual or technical challenge of the course content (not time spent).
- positive: "challenging in a good way, pushed me to think."
- neutral: "about as hard as i expected."
- negative: "way over my head from week 2."

#### 11. `workload`
Amount of sustained effort required across the term (time, not difficulty).
- positive: "manageable, well-paced workload."
- neutral: "standard workload for the program."
- negative: "took over my life for the whole semester."

#### 12. `pacing`
Whether the course tempo and weekly rhythm are manageable. Distinct from `workload`:
pacing is the distribution of effort, workload is the total amount.
- positive: "weekly modules made the pace feel steady."
- neutral: "pace was about what i expected."
- negative: "everything dumped in the last three weeks."

#### 13. `prerequisite_fit`
How well the course matches the advertised prerequisite level and student preparation.
- positive: "the prereqs were exactly what you needed."
- neutral: "the listed prereqs covered the basics."
- negative: "the prereqs undersold what you actually need to know."

### Group 4: Learning environment

#### 14. `support`
Quality of help from instructor, TAs, or forums (Piazza, Slack, office hours). Distinct
from `lecturer_quality`: this is responsiveness and helpfulness when stuck, not
teaching ability.
- positive: "TAs were on piazza within an hour every time."
- neutral: "office hours existed if you needed them."
- negative: "asked three times on piazza, never got an answer."

#### 15. `accessibility`
Perceived accessibility and inclusiveness of materials, pace, and course participation,
including for non-traditional students, working professionals, and students with
disabilities.
- positive: "recordings made it manageable while working full-time."
- neutral: "course was accessible enough on a normal schedule."
- negative: "no captioning on videos and no flexibility on deadlines."

#### 16. `peer_interaction`
Whether peer discussion, teamwork, and class community help or hinder learning.
- positive: "the study group made the hard topics click."
- neutral: "minimal group work, mostly individual."
- negative: "the assigned team disengaged after week 3."

### Group 5: Engagement and value

#### 17. `relevance`
Perceived usefulness to the student's program of study or future goals.
- positive: "fits exactly into my specialization."
- neutral: "useful in some areas, not in others."
- negative: "couldn't see how any of this connects to my track."

#### 18. `interest`
Level of engagement or curiosity the course creates.
- positive: "i actually looked forward to lectures."
- neutral: "some topics interesting, others dry."
- negative: "lost interest by week 4."

#### 19. `practical_application`
Connection to real-world practice or authentic tasks. Distinct from `relevance`:
relevance is about career/program fit; practical_application is about whether the work
itself mimics real practice.
- positive: "i used these techniques at work the same week."
- neutral: "labs were textbook style."
- negative: "everything was toy problems, no real-world flavor."

#### 20. `overall_experience`
Global student impression after tradeoffs, including recommendation stance.
- positive: "best class i've taken so far in the program."
- neutral: "decent overall, no regrets, no rave."
- negative: "would not recommend, take something else."

## Sentiment polarity

| label    | meaning                                                                                   |
|----------|-------------------------------------------------------------------------------------------|
| positive | clear praise or favorable description of this aspect                                      |
| neutral  | the aspect is discussed but without strong polarity in either direction, or genuinely mixed within the same statement |
| negative | clear complaint or unfavorable description of this aspect                                 |

## Discussed vs. not discussed

For Task 2 (annotating Herath reviews) and Task 3 (judging audit items), you must
first decide whether each aspect is **discussed** in the review.

- **discussed = yes:** the review contains at least one sentence, clause, or phrase
  that addresses this aspect, even briefly.
- **discussed = no:** the aspect is not mentioned. This is the most common case for any
  single review.
- **discussed = unclear:** the review hints at the aspect but the evidence is too
  oblique to call. Use this sparingly.

If the aspect is **not discussed**, leave the polarity column blank.

## Edge cases

- **Sarcasm.** Rate sarcasm at its intended polarity, not its literal one. "great, more
  readings to skim before the quiz" is **negative** on `materials`.
- **Hedged statements.** "it was okay, i guess" lean toward **neutral** unless context
  strengthens one side.
- **Aspect overlap.** "exams were unfair because we never covered the material" is
  evidence on both `exam_fairness` (negative) and `clarity` or `materials`. Label every
  aspect that is genuinely supported. Do not duplicate evidence if one aspect is just a
  synonym for another in context.
- **Aggregations.** "the course was hard but i learned a lot" splits into
  `difficulty=neutral/negative` and `overall_experience=positive`. Do not collapse
  multi-aspect sentences into one label.
- **Grade complaints alone.** "got a C, hated it" without a stated reason counts only
  toward `overall_experience=negative`. Do not infer fairness or workload from a bare
  grade.
- **Specific TAs, professors, or assignments.** Use these mentions as evidence for the
  relevant aspect (`lecturer_quality`, `support`, `materials`, etc.). Do not treat names
  as identifying information that requires redaction; they are part of the public
  review.
- **Tooling vs. support.** A complaint about the LMS or autograder is `tooling_usability`,
  not `support`. A complaint that "support never answered me" is `support`.
- **Feedback vs. assessment_design.** A complaint about how an assignment was *graded*
  is `feedback_quality` or `grading_transparency`. A complaint about how an assignment
  was *structured* is `assessment_design`.

## Polarity is per aspect, not per review

Each aspect's polarity is independent of others. The same review can carry
`workload=negative`, `interest=positive`, and `lecturer_quality=neutral`. Do not let
one aspect's polarity bleed into another's.

## `unclear` is allowed but should be rare

Use `unclear` only after re-reading the relevant sentence(s). If you find yourself
using it more than twice per review, you are probably being too strict; lean toward a
concrete label.
