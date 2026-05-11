from __future__ import annotations

import argparse
import json
import os
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None


ROOT = Path(__file__).resolve().parents[1]
KEY_FILE = ROOT / ".opeai.key"
OUT_DIR = ROOT / "paper" / "generation_protocol"
VALIDATION_DIR = ROOT / "paper" / "validation"
FINAL_TEMPLATE_PATH = OUT_DIR / "final_realism_prompt_template.txt"
FINAL_METADATA_PATH = OUT_DIR / "final_realism_prompt_metadata.json"
FINAL_SCRIPT_PATH = OUT_DIR / "final_prompt_generation_script.py"
FINAL_SAMPLE_PATH = OUT_DIR / "final_prompt_samples.csv"
STABLE_PROMPT_STATE_PATH = VALIDATION_DIR / "stable_realism_prompt_instruction.json"
POLARITY_MODE = "binary_positive_negative"
DEFAULT_ASPECT_COUNT_DISTRIBUTION = {"1": 0.30, "2": 0.40, "3": 0.30}

CURRENT_RELEASE_ASPECTS = [
    "difficulty",
    "clarity",
    "workload",
    "lecturer_quality",
    "exam_fairness",
    "relevance",
    "interest",
    "support",
    "materials",
    "overall_experience",
]
SENTIMENTS = ["positive", "negative"]

PROPOSED_EXTENDED_ASPECTS = {
    "difficulty": "Conceptual or technical challenge of the course.",
    "clarity": "How understandable the teaching and explanations feel.",
    "workload": "Amount of sustained effort required across the term.",
    "lecturer_quality": "Perceived quality of the lecturer or lead instructor.",
    "exam_fairness": "Whether exams feel aligned and fair.",
    "relevance": "Perceived usefulness to the program or future goals.",
    "interest": "Level of engagement or curiosity the course creates.",
    "support": "Quality of help from instructor, TAs, or forums.",
    "materials": "Usefulness of slides, notes, readings, and resources.",
    "overall_experience": "Global student impression after tradeoffs.",
    "feedback_quality": "Usefulness and timeliness of feedback on student work.",
    "assessment_design": "Alignment and structure of assignments, projects, and exams.",
    "pacing": "Whether the course tempo and weekly rhythm are manageable.",
    "organization": "Administrative clarity, course structure, and coordination.",
    "practical_application": "Connection to real-world practice or authentic tasks.",
    "tooling_usability": "Friction or support created by LMS, submission systems, and required software.",
    "accessibility": "Perceived accessibility and inclusiveness of materials, pace, and course participation.",
    "grading_transparency": "How clearly grading criteria, rubrics, and score interpretation are communicated.",
    "peer_interaction": "Whether peer discussion, teamwork, and class community help or hinder learning.",
    "prerequisite_fit": "How well the course matches the advertised prerequisite level and student preparation.",
}
ASPECTS = list(PROPOSED_EXTENDED_ASPECTS)
FINAL_INSTRUCTION_FALLBACK = (
    "Write a realistic first-person course review in a mildly informal voice. "
    "Make it feel like recalled experience, not a balanced evaluation: focus on 1-2 things the student would actually remember, "
    "with uneven detail and some partialness. Include at most 1-2 concrete, course-plausible specifics "
    "(such as a project, tool, grading quirk, deadline pattern, exam format, or small incident), and make at least one of them individualized "
    "rather than just subject jargon. Do not try to cover every aspect or balance praise and criticism; avoid tidy contrast patterns, "
    "stacked common review motifs, generic domain-term lists, and polished summary phrasing. Keep the sentiment and details consistent with "
    "the sampled student persona and the real course/instructor context, and end with a natural complete final sentence."
)

ATTRIBUTE_SAMPLING_POLICY = {
    "core_context": {"required": ["course_name"], "sample_size": 5},
    "assessment_and_teaching": {"required": [], "sample_size": 4},
    "linguistic_diversity": {"required": [], "sample_size": 3},
    "realism_controls": {"required": [], "sample_size": 3},
}


SEED_SCHEMA = {
    "core_context": {
        "course_name": [
            "Linear Algebra",
            "Computer Networks",
            "Data Structures",
            "Operating Systems",
            "Machine Learning",
            "Cyber Security Basics",
            "Database Systems",
            "Calculus II",
        ],
        "course_level": [
            "introductory service course",
            "mid-program core course",
            "advanced elective",
            "required course for the track",
        ],
        "semester_stage": [
            "first semester in the program",
            "middle of a heavy semester",
            "last semester before graduation",
            "summer term with compressed pacing",
        ],
        "grade_band": [
            "A after a curve",
            "solid B and felt it matched the effort",
            "C and unsure whether it reflected understanding",
            "barely passed after struggling most of the term",
            "failed and still has mixed feelings about the course",
        ],
        "student_background": [
            "strong coding background but weak theory",
            "comfortable with proofs but not systems work",
            "full-time job and limited weekly study time",
            "returning student who feels rusty",
            "new to the discipline and learning the basics",
        ],
        "lecturer_name": [
            "Dr. Chen",
            "Prof. Alvarez",
            "Dr. Singh",
            "Prof. Cohen",
            "Dr. Park",
            "Prof. Haddad",
            "Dr. Miller",
            "Prof. Rios",
        ],
        "motivation_for_taking_course": [
            "required for the degree",
            "needed for a specialization",
            "heard it was useful for industry work",
            "fit the schedule more than the interests",
            "genuine curiosity about the topic",
        ],
        "attendance_pattern": [
            "kept up with nearly every lecture",
            "mostly relied on recordings and notes",
            "fell behind and caught up around deadlines",
            "used the materials selectively when needed",
            "started engaged but attendance faded mid-semester",
        ],
        "study_context": [
            "took it alongside a full-time job",
            "balanced it with multiple demanding courses",
            "made it the main focus course that term",
            "fit study around short evening sessions",
            "had enough time but inconsistent motivation",
        ],
    },
    "assessment_and_teaching": {
        "assessment_profile": [
            "project-heavy with high-stakes exams",
            "assignment-heavy with frequent deadlines",
            "few assessments but each one matters a lot",
            "report-heavy and time-consuming",
            "assessment mix feels mismatched with lectures",
        ],
        "instruction_delivery": [
            "clear lectures with decent pacing",
            "recorded lectures are useful but dry",
            "conceptual explanations are strong but examples are thin",
            "slides are serviceable but not memorable",
            "teaching feels uneven week to week",
        ],
        "support_channel_experience": [
            "TAs are responsive and practical",
            "forums are helpful but sometimes dismissive",
            "email support is slow when deadlines hit",
            "office hours rescue the course",
            "students help each other more than staff do",
        ],
        "administrative_friction": [
            "grading rubric feels vague",
            "instructions leave too much interpretation",
            "tooling setup is frustrating",
            "feedback arrives too late to help",
            "policies are fine but communication is sloppy",
        ],
        "feedback_timing": [
            "feedback comes quickly enough to adjust",
            "feedback is sparse until it is too late",
            "grading comments are detailed but delayed",
            "scores arrive fast but explanations are thin",
            "feedback quality varies a lot by assignment",
        ],
        "prerequisite_fit": [
            "assumes more prior knowledge than advertised",
            "builds well on the stated prerequisites",
            "starts accessible but ramps up sharply",
            "works best if you already know adjacent material",
            "fills prerequisite gaps better than expected",
        ],
        "collaboration_structure": [
            "mostly individual work with little peer dependency",
            "group work adds value when the team functions well",
            "peer discussion helps more than the official materials",
            "collaboration expectations feel unclear",
            "the course works better if you have a study group",
        ],
        "platform_and_tooling": [
            "the LMS and submission flow are smooth",
            "platform quirks create avoidable stress",
            "the software stack is rough at first but manageable later",
            "tool setup takes too much effort for the learning payoff",
            "tooling problems make deadline weeks worse",
        ],
    },
    "linguistic_diversity": {
        "writing_style": [
            "casual and compressed",
            "plainspoken and direct",
            "analytic but simple",
            "rant-like and emotional",
            "hesitant and second-guessing",
            "short fragmented remarks",
        ],
        "emotional_temperature": [
            "mostly calm",
            "frustrated but fair",
            "tired and resigned",
            "genuinely enthusiastic",
            "mixed and conflicted",
        ],
        "hedging_level": [
            "confident and decisive",
            "mildly hedged",
            "frequently unsure",
            "changes opinion mid-review",
        ],
        "specificity_level": [
            "mentions concrete assignments or exams",
            "references forums, office hours, or TAs",
            "includes one detailed pain point",
            "stays general with only one anchor detail",
        ],
        "recommendation_stance": [
            "would recommend broadly",
            "would recommend only with preparation",
            "cannot recommend despite some value",
            "not sure whether to recommend",
        ],
        "review_length_band": [
            "very short comment",
            "compact but informative review",
            "mid-length reflective review",
            "detailed review with one dominant complaint",
        ],
        "formality_level": [
            "informal and conversational",
            "neutral everyday prose",
            "fairly polished and careful",
            "mixed academic and casual phrasing",
        ],
    },
    "realism_controls": {
        "review_shape": [
            "starts with a general verdict then gets specific",
            "starts with one complaint and later softens it",
            "lists tradeoffs rather than a single conclusion",
            "wanders slightly before landing the point",
        ],
        "natural_noise": [
            "minor lowercase or punctuation looseness",
            "slight repetition from frustration",
            "one colloquial phrase but not many",
            "clean prose with small informal touches",
        ],
        "contradiction_pattern": [
            "learned a lot but disliked the experience",
            "liked the instructor but disliked the grading",
            "course felt useful but not enjoyable",
            "assessment was fair but still exhausting",
            "would not take it again even though it helped",
        ],
        "time_pressure_context": [
            "took it with another hard class",
            "managed it alongside full-time work",
            "had a rough mid-semester crunch",
            "the deadlines stacked up unexpectedly",
            "time pressure was manageable most weeks",
        ],
        "comparison_frame": [
            "compares it to an easier prerequisite",
            "compares it to another course from the same term",
            "judges it against expectations rather than another course",
            "avoids explicit comparison and stays course-specific",
        ],
        "memory_anchor": [
            "mentions one memorable exam or assignment week",
            "mentions a concrete office-hours or forum interaction",
            "anchors the review in a late-semester turning point",
            "recalls an early confusion that shaped the rest of the term",
        ],
    },
}


ATTRIBUTE_SCHEMA_DESIGN_PROMPT = """
You are designing an attribute schema for generating realistic and diverse student course reviews for aspect-based sentiment analysis.

Goals:
1. Increase diversity beyond only grade, course, and writing style.
2. Increase realism so reviews do not sound like synthetic checklist outputs.
3. Keep the schema compact enough for controlled prompting.

Constraints:
- The domain is university course reviews.
- Downstream labels are aspect sentiments over these aspects:
  difficulty, clarity, workload, lecturer_quality, exam_fairness, relevance, interest, support, materials, overall_experience, feedback_quality, assessment_design, pacing, organization, practical_application, tooling_usability, accessibility, grading_transparency, peer_interaction, prerequisite_fit
- The schema should include both diversity attributes and realism-control attributes.
- Each attribute should have 4 to 8 possible values.
- Values should be plausible, non-overlapping when possible, and easy to sample randomly.
- Avoid attributes that directly leak sentiment labels.

Return strict JSON with top-level keys:
- schema: object mapping group names to objects of attribute -> list of values
- protocol_notes: list of short notes about how to use the schema
- anti_patterns: list of cues that make synthetic reviews feel fake
"""


ATTRIBUTE_SCHEMA_REVIEW_PROMPT = """
You are reviewing a synthetic course-review generation protocol for two criteria:
1. diversity
2. realism

You will receive:
- an attribute schema
- a generation prompt template

Your job:
- identify weak spots that still reduce diversity
- identify cues that still make outputs sound synthetic
- propose concrete prompt revisions
- propose better sampling rules where needed

Return strict JSON with keys:
- diversity_gaps
- realism_gaps
- prompt_revisions
- sampling_revisions
- final_verdict
"""


def resolved_final_instruction() -> Dict[str, str]:
    if STABLE_PROMPT_STATE_PATH.exists():
        payload = json.loads(STABLE_PROMPT_STATE_PATH.read_text(encoding="utf-8"))
        instruction_text = str(payload.get("instruction_text", "")).strip()
        if instruction_text:
            return {
                "instruction_text": instruction_text,
                "source": str(STABLE_PROMPT_STATE_PATH),
                "run_id": str(payload.get("run_id", "")),
                "cycle_id": str(payload.get("cycle_id", "")),
            }
    return {
        "instruction_text": FINAL_INSTRUCTION_FALLBACK,
        "source": "fallback_constant",
        "run_id": "",
        "cycle_id": "",
    }


def build_generation_prompt_template(final_instruction: str) -> str:
    return (
        "You are writing one realistic student course review for research validation.\n"
        "The review must feel like a naturally written student comment rather than a labeled synthetic sample.\n\n"
        "Target aspect sentiments:\n"
        "{aspect_block}\n\n"
        "Target attributes:\n"
        "{attribute_block}\n\n"
        "Requirements:\n"
        "- Keep the review first-person and specific.\n"
        "- Do not mention aspect labels or sentiment labels explicitly.\n"
        "- Do not force a tidy conclusion.\n"
        "- Do not cover every aspect with the same level of detail.\n"
        "- Let at least one point feel incidental rather than checklist-driven.\n"
        "- Preserve mixed feelings when the attributes imply them.\n"
        f"- Additional stable realism instruction: {final_instruction}\n\n"
        "Return only the review text.\n"
    )


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_api_key() -> str:
    if KEY_FILE.exists():
        return KEY_FILE.read_text(encoding="utf-8").strip()
    env_value = os.environ.get("OPENAI_API_KEY", "").strip()
    if env_value:
        return env_value
    raise RuntimeError("No OpenAI API key found.")


def load_client() -> OpenAI:
    if OpenAI is None:
        raise RuntimeError("openai package is not installed.")
    return OpenAI(api_key=load_api_key())


def response_text(client: OpenAI, model: str, prompt: str, max_output_tokens: int = 900) -> str:
    response = client.responses.create(
        model=model,
        input=prompt,
        max_output_tokens=max_output_tokens,
    )
    return (response.output_text or "").strip()


def strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def extract_json_block(text: str) -> str:
    text = strip_code_fences(text)
    match = re.search(r"\{", text)
    if not match:
        raise ValueError(f"No JSON object found in response: {text[:300]}")
    start = match.start()
    depth = 0
    for idx in range(start, len(text)):
        char = text[idx]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    raise ValueError(f"Incomplete JSON object in response: {text[:300]}")


def parse_json(text: str) -> Dict[str, object]:
    return json.loads(extract_json_block(text))


def schema_parameter_summary() -> Dict[str, Dict[str, int]]:
    return {
        group_name: {attribute: len(values) for attribute, values in group.items()}
        for group_name, group in SEED_SCHEMA.items()
    }


def aspect_count_distribution() -> Dict[str, float]:
    # Binary-polarity reset: do not depend on any old dataset artifact to define the
    # new prompt package. Keep a fixed practical distribution until the new corpus is built.
    return dict(DEFAULT_ASPECT_COUNT_DISTRIBUTION)


def sample_aspect_labels(rng: random.Random, n_aspects: int) -> Dict[str, str]:
    selected = rng.sample(ASPECTS, n_aspects)
    labels = {}
    for aspect in selected:
        labels[aspect] = rng.choice(SENTIMENTS)
    return labels


def sample_attributes(schema: Dict[str, Dict[str, List[str]]], rng: random.Random) -> Dict[str, str]:
    sampled = {}
    for group_name, group in schema.items():
        policy = ATTRIBUTE_SAMPLING_POLICY.get(group_name, {"required": [], "sample_size": len(group)})
        required = [name for name in policy["required"] if name in group]
        optional = [name for name in group.keys() if name not in required]
        target_count = min(len(group), int(policy["sample_size"]))
        additional = max(0, target_count - len(required))
        chosen = required + rng.sample(optional, k=min(additional, len(optional)))
        for attribute in chosen:
            sampled[attribute] = rng.choice(group[attribute])
    return sampled


def render_prompt(aspect_labels: Dict[str, str], attributes: Dict[str, str]) -> str:
    template = build_generation_prompt_template(resolved_final_instruction()["instruction_text"])
    aspect_block = "\n".join(f"- {aspect}: {sentiment}" for aspect, sentiment in aspect_labels.items())
    attribute_block = "\n".join(f"- {key}: {value}" for key, value in attributes.items())
    return template.format(aspect_block=aspect_block, attribute_block=attribute_block)


def export_seed_schema() -> None:
    current_distribution = aspect_count_distribution()
    payload = {
        "polarity_mode": POLARITY_MODE,
        "supported_sentiments": SENTIMENTS,
        "schema": SEED_SCHEMA,
        "current_release_aspects": CURRENT_RELEASE_ASPECTS,
        "proposed_extended_aspects": PROPOSED_EXTENDED_ASPECTS,
        "attribute_parameter_summary": schema_parameter_summary(),
        "empirical_aspect_count_distribution": current_distribution,
        "recommended_sampling_rule": {
            "aspect_count": "Sample 1, 2, or 3 aspects using the fixed binary-polarity distribution for the new generation cycle.",
            "current_distribution": current_distribution,
            "practical_override": current_distribution,
        },
        "nuance_sampling_policy": ATTRIBUTE_SAMPLING_POLICY,
        "protocol_review": {
            "diversity_strengths": [
                "The schema now varies student background, motivation, assessment structure, support experience, hedging, contradiction patterns, and recommendation stance.",
                "The sampled attributes create broader lexical and situational diversity than style plus grade alone.",
                "Only a subset of nuance attributes is surfaced per prompt so reviews do not read like exhaustive templates.",
            ],
            "realism_strengths": [
                "The schema includes grounded friction, time pressure, and contradiction patterns that mimic real student reviews.",
                "The prompt explicitly discourages checklist structure and over-balanced summaries.",
            ],
            "aspect_extension_rationale": [
                "The proposed next-generation inventory separates feedback quality, pacing, organization, assessment design, tooling usability, accessibility, grading transparency, peer interaction, and prerequisite fit from generic overall experience.",
                "Those extra aspects are pedagogically actionable and reduce the collapse of distinct student concerns into a single broad sentiment label.",
            ],
            "remaining_risks": [
                "Without live OpenAI prompt-debug cycles, some attribute combinations may still generate overly tidy reviews.",
                "Human validation and synthetic-vs-real discrimination remain necessary for external realism claims.",
            ],
        },
    }
    (OUT_DIR / "seed_attribute_schema.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_final_script_text(template_text: str) -> str:
    template_literal = json.dumps(template_text)
    return f'''from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "paper" / "generation_protocol"
SCHEMA_PATH = OUT_DIR / "seed_attribute_schema.json"
TEMPLATE_PATH = OUT_DIR / "final_realism_prompt_template.txt"
DEFAULT_OUTPUT_PATH = OUT_DIR / "final_prompt_samples.csv"
SENTIMENTS = ["positive", "negative"]
ASPECTS = [
    "difficulty",
    "clarity",
    "workload",
    "lecturer_quality",
    "exam_fairness",
    "relevance",
    "interest",
    "support",
    "materials",
    "overall_experience",
    "feedback_quality",
    "assessment_design",
    "pacing",
    "organization",
    "practical_application",
    "tooling_usability",
    "accessibility",
    "grading_transparency",
    "peer_interaction",
    "prerequisite_fit",
]


def load_schema() -> Dict[str, object]:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def load_template() -> str:
    return TEMPLATE_PATH.read_text(encoding="utf-8")


def sample_aspect_labels(rng: random.Random, n_aspects: int) -> Dict[str, str]:
    selected = rng.sample(ASPECTS, n_aspects)
    return {{aspect: rng.choice(SENTIMENTS) for aspect in selected}}


def sample_attributes(schema: Dict[str, object], rng: random.Random) -> Dict[str, str]:
    sampled: Dict[str, str] = {{}}
    schema_obj = schema["schema"]
    policy_obj = schema["nuance_sampling_policy"]
    for group_name, group in schema_obj.items():
        policy = policy_obj.get(group_name, {{"required": [], "sample_size": len(group)}})
        required = [name for name in policy.get("required", []) if name in group]
        optional = [name for name in group if name not in required]
        target_count = min(len(group), int(policy.get("sample_size", len(group))))
        chosen = required + rng.sample(optional, k=min(max(0, target_count - len(required)), len(optional)))
        for attribute_name in chosen:
            sampled[attribute_name] = rng.choice(group[attribute_name])
    return sampled


def render_prompt(template: str, aspect_labels: Dict[str, str], attributes: Dict[str, str]) -> str:
    aspect_block = "\\n".join(f"- {{aspect}}: {{sentiment}}" for aspect, sentiment in aspect_labels.items())
    attribute_block = "\\n".join(f"- {{key}}: {{value}}" for key, value in attributes.items())
    return template.format(aspect_block=aspect_block, attribute_block=attribute_block)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate sampled prompts from the saved final realism prompt template.")
    parser.add_argument("--sample-count", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH))
    args = parser.parse_args()

    schema = load_schema()
    template = load_template()
    rng = random.Random(args.seed)
    aspect_distribution = schema["recommended_sampling_rule"]["practical_override"]
    options = [1, 2, 3]
    weights = [aspect_distribution[str(option)] for option in options]

    rows: List[Dict[str, str]] = []
    for idx in range(args.sample_count):
        n_aspects = rng.choices(options, weights=weights, k=1)[0]
        aspect_labels = sample_aspect_labels(rng, n_aspects)
        attributes = sample_attributes(schema, rng)
        rows.append(
            {{
                "sample_id": str(idx),
                "n_aspects": str(n_aspects),
                "aspect_labels": json.dumps(aspect_labels, ensure_ascii=False),
                "attributes": json.dumps(attributes, ensure_ascii=False),
                "prompt": render_prompt(template, aspect_labels, attributes),
            }}
        )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "n_aspects", "aspect_labels", "attributes", "prompt"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {{len(rows)}} prompts to {{output_path}}")


if __name__ == "__main__":
    main()
'''


def best_refinement_payload() -> Dict[str, object] | None:
    preferred = sorted((ROOT / "paper" / "validation").glob("prompt_debug_cycle2_n30_*_refinement.json"))
    if preferred:
        top_path = preferred[-1]
        top_payload = json.loads(top_path.read_text(encoding="utf-8"))
        top_payload["_selected_from"] = str(top_path)
        return top_payload
    candidates = sorted((ROOT / "paper" / "validation").glob("prompt_debug*_refinement.json"))
    scored = []
    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        cues = payload.get("cues_to_avoid") or []
        edit_actions = payload.get("edit_actions") or []
        editor_input_count = int(payload.get("editor_input_count", 0) or 0)
        score = (int(bool(cues)), int(bool(edit_actions)), editor_input_count, path.stat().st_mtime)
        scored.append((score, path, payload))
    if not scored:
        return None
    scored.sort(reverse=True)
    top_path, top_payload = scored[0][1], scored[0][2]
    top_payload["_selected_from"] = str(top_path)
    return top_payload


def export_final_package(sample_count: int, seed: int) -> None:
    final_instruction_payload = resolved_final_instruction()
    template_text = build_generation_prompt_template(final_instruction_payload["instruction_text"])
    metadata = {
        "package_name": "final_realism_prompt_package_binary_polarity",
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "template_path": str(FINAL_TEMPLATE_PATH),
        "metadata_path": str(FINAL_METADATA_PATH),
        "script_path": str(FINAL_SCRIPT_PATH),
        "sample_output_path": str(FINAL_SAMPLE_PATH),
        "polarity_mode": POLARITY_MODE,
        "supported_sentiments": SENTIMENTS,
        "interactive_debug_model": "gpt-5.4",
        "planned_batch_generation_model": "gpt-5-nano",
        "purpose": "Final saved prompt template and prompt-generation helper after the binary-polarity realism-debug cycle.",
        "aspect_inventory": PROPOSED_EXTENDED_ASPECTS,
        "nuance_sampling_policy": ATTRIBUTE_SAMPLING_POLICY,
        "recommended_aspect_count_distribution": DEFAULT_ASPECT_COUNT_DISTRIBUTION,
        "selected_prompt_state_reference": final_instruction_payload["source"],
        "selected_prompt_run_id": final_instruction_payload["run_id"],
        "selected_prompt_cycle_id": final_instruction_payload["cycle_id"],
        "selected_prompt_instruction": final_instruction_payload["instruction_text"],
        "prompt_lineage_policy": "Use the stable prompt state saved at the end of the completed five-cycle binary-polarity realism sequence.",
    }
    refinement = best_refinement_payload()
    if refinement is not None:
        metadata["selected_refinement_reference"] = refinement.get("_selected_from")
        metadata["selected_refinement_status"] = refinement.get("status")
        metadata["selected_cues_to_avoid"] = refinement.get("cues_to_avoid", [])
        metadata["selected_edit_actions"] = refinement.get("edit_actions", [])

    FINAL_TEMPLATE_PATH.write_text(template_text, encoding="utf-8")
    FINAL_METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    FINAL_SCRIPT_PATH.write_text(build_final_script_text(template_text), encoding="utf-8")
    sample_prompt_configs(sample_count, seed)
    sample_source_path = OUT_DIR / "sampled_prompt_configs.csv"
    if sample_source_path.exists():
        FINAL_SAMPLE_PATH.write_text(sample_source_path.read_text(encoding="utf-8"), encoding="utf-8")


def sample_prompt_configs(sample_count: int, seed: int) -> None:
    rng = random.Random(seed)
    empirical = aspect_count_distribution()
    options = [1, 2, 3]
    weights = [empirical[str(k)] for k in options]
    rows = []
    for idx in range(sample_count):
        n_aspects = rng.choices(options, weights=weights, k=1)[0]
        aspect_labels = sample_aspect_labels(rng, n_aspects)
        attributes = sample_attributes(SEED_SCHEMA, rng)
        rows.append(
            {
                "sample_id": idx,
                "n_aspects": n_aspects,
                "aspect_labels": json.dumps(aspect_labels, ensure_ascii=False),
                "attributes": json.dumps(attributes, ensure_ascii=False),
                "prompt": render_prompt(aspect_labels, attributes),
            }
        )
    pd = __import__("pandas")
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "sampled_prompt_configs.csv", index=False)


def openai_refine_schema(model: str) -> None:
    client = load_client()
    seed_payload = {
        "schema": SEED_SCHEMA,
        "empirical_aspect_count_distribution": aspect_count_distribution(),
    }
    prompt = (
        ATTRIBUTE_SCHEMA_DESIGN_PROMPT.strip()
        + "\n\nCurrent seed schema:\n"
        + json.dumps(seed_payload, indent=2)
    )
    result = parse_json(response_text(client, model, prompt, max_output_tokens=3200))
    (OUT_DIR / "openai_refined_attribute_schema.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


def openai_review_protocol(model: str) -> None:
    client = load_client()
    sample_aspects = {"clarity": "negative", "support": "positive"}
    sample_attributes = {
        "course_name": "Computer Networks",
        "grade_band": "solid B and felt it matched the effort",
        "student_background": "full-time job and limited weekly study time",
        "assessment_profile": "project-heavy with high-stakes exams",
        "support_channel_experience": "forums are helpful but sometimes dismissive",
        "writing_style": "hesitant and second-guessing",
        "contradiction_pattern": "liked the instructor but disliked the grading",
    }
    prompt = (
        ATTRIBUTE_SCHEMA_REVIEW_PROMPT.strip()
        + "\n\nAttribute schema:\n"
        + json.dumps(SEED_SCHEMA, indent=2)
        + "\n\nGeneration prompt template example:\n"
        + render_prompt(sample_aspects, sample_attributes)
    )
    result = parse_json(response_text(client, model, prompt, max_output_tokens=3600))
    (OUT_DIR / "openai_protocol_review.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Upgrade and review the course-review generation protocol for realism and diversity.")
    parser.add_argument("mode", choices=["export-seed-schema", "sample-prompts", "openai-refine-schema", "openai-review-protocol", "export-final-package"])
    parser.add_argument("--sample-count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="gpt-5.4")
    args = parser.parse_args()

    ensure_dirs()

    if args.mode == "export-seed-schema":
        export_seed_schema()
        print(f"Saved seed schema to {OUT_DIR}")
        return

    if args.mode == "sample-prompts":
        sample_prompt_configs(args.sample_count, args.seed)
        print(f"Saved sampled prompt configs to {OUT_DIR}")
        return

    if args.mode == "openai-refine-schema":
        openai_refine_schema(args.model)
        print(f"Saved OpenAI-refined schema to {OUT_DIR}")
        return

    if args.mode == "openai-review-protocol":
        openai_review_protocol(args.model)
        print(f"Saved OpenAI protocol review to {OUT_DIR}")
        return

    if args.mode == "export-final-package":
        export_seed_schema()
        export_final_package(args.sample_count, args.seed)
        print(f"Saved final realism prompt package to {OUT_DIR}")


if __name__ == "__main__":
    main()
