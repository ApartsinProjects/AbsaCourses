from __future__ import annotations

import argparse
import json
import os
import random
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import requests
from bs4 import BeautifulSoup

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None


ROOT = Path(__file__).resolve().parents[1]
VALIDATION_DIR = ROOT / "paper" / "validation"
KEY_FILE = ROOT / ".opeai.key"

COURSE_SOURCES = {
    "CS-6200": "https://awaisrauf.com/omscs_reviews/CS-6200",
    "CS-6250": "https://awaisrauf.com/omscs_reviews/CS-6250",
    "CS-6400": "https://awaisrauf.com/omscs_reviews/CS-6400",
    "CS-7641": "https://awaisrauf.com/omscs_reviews/CS-7641",
}

GENERATOR_MODEL = "gpt-5.2"
JUDGE_MODEL = "gpt-5.4"
RANDOM_SEED = 42
COURSE_TITLES = {
    "CS-6200": "Introduction to Operating Systems",
    "CS-6250": "Computer Networks",
    "CS-6400": "Database Systems Concepts & Design",
    "CS-7641": "Machine Learning",
}
COURSE_SPECIFIC_MARKERS = {
    "CS-6200": [
        "projects in C",
        "heavy exams relative to projects",
        "Piazza or Slack interactions with TAs",
        "papers and lecture quizzes",
    ],
    "CS-6250": [
        "Wireshark-heavy assignments",
        "network protocol diagrams",
        "labs and packet-level debugging",
        "time pressure on projects",
    ],
    "CS-6400": [
        "schema design and SQL work",
        "project milestones",
        "database normalization tradeoffs",
        "team coordination friction",
    ],
    "CS-7641": [
        "long reports for experiments",
        "vague project instructions",
        "graphs and hyperparameter sweeps",
        "frustration with grading clarity",
    ],
}
BACKGROUND_OPTIONS = [
    "strong programming background but limited formal theory",
    "coming from industry and rusty on academic coursework",
    "solid math comfort but uneven systems experience",
    "little prior background and taking the course as a stretch",
]
MOTIVATION_OPTIONS = [
    "required for a specialization",
    "wanted a practical foundation for work",
    "curiosity about the topic after hearing it was important",
    "fit well with the rest of the semester schedule",
]
GRADE_BANDS = [
    "earned an A after a curve",
    "finished around a B and felt that was fair",
    "barely held onto a passing grade",
    "not fully sure of the final grade until the end",
]
WORKLOAD_OPTIONS = [
    "steady but manageable most weeks, then spiky around deadlines",
    "consistently heavy and mentally draining",
    "moderate overall but very bursty near projects",
    "surprisingly time consuming given the course description",
]
ASSESSMENT_OPTIONS = [
    "projects dominated the effort while exams still carried a lot of the grade",
    "assignments were doable but the grading rubric felt opaque",
    "exams and projects felt somewhat disconnected",
    "the report-writing burden mattered almost as much as the technical work",
]
INSTRUCTION_OPTIONS = [
    "lectures were clear overall but occasionally long-winded",
    "lectures had useful intuition even when specific tasks felt disconnected",
    "recorded material was strong but not always aligned with assessments",
    "core explanations were helpful, but some topics felt under-motivated",
]
SUPPORT_OPTIONS = [
    "TAs were responsive and the forums were useful",
    "peer discussion was more helpful than official responses",
    "support existed but timing was inconsistent when stress peaked",
    "some answers were helpful while others felt dismissive",
]
FRICTION_OPTIONS = [
    "grading standards felt vague",
    "instructions left room for too much interpretation",
    "tooling and setup overhead added avoidable frustration",
    "administrative communication was the weakest part of the course",
]
TEMPERATURE_OPTIONS = [
    "mostly measured with a few frustrated edges",
    "openly annoyed but still fair",
    "tired and slightly resigned",
    "mixed and reflective rather than fully positive or negative",
]
STYLE_OPTIONS = [
    "moderately polished with occasional casual phrasing",
    "plainspoken and a little compressed",
    "slightly rant-like in places but still specific",
    "reflective with hedging and self-correction",
]
HEDGING_OPTIONS = [
    "some uncertainty and mixed feelings should remain visible",
    "the review should not sound fully resolved about whether the course was good",
    "allow tension between learning value and course frustration",
    "include at least one partial reversal such as praise followed by a caveat",
]
RECOMMENDATION_OPTIONS = [
    "recommend only to students with the right preparation",
    "recommend with reservations",
    "not a clear recommendation despite some value",
    "depends strongly on background and semester load",
]

RICH_ATTRIBUTE_SPACE = {
    "course_code": "Exact course identifier used in the target review domain.",
    "course_title": "Human-readable course title.",
    "semester": "Term marker such as Fall 2022 or Spring 2023.",
    "student_background": "Prior familiarity with the topic, math maturity, coding fluency, or security/network background.",
    "motivation": "Why the student took the course, such as requirement, curiosity, or specialization fit.",
    "grade_band": "Observed or expected performance level, including ambiguity around final grade when natural.",
    "workload_intensity": "Perceived weekly effort and project crunch level.",
    "assessment_profile": "Relative emphasis on projects, exams, reports, quizzes, or participation.",
    "instruction_quality": "Perception of lectures, notes, examples, and conceptual clarity.",
    "support_channel_experience": "Office hours, forums, TAs, Piazza, email responsiveness, or peer support.",
    "administrative_friction": "Confusing rubrics, grading opacity, deadlines, tooling issues, or policy frustration.",
    "emotional_temperature": "Measured, mixed, enthusiastic, bitter, exhausted, or resigned tone.",
    "linguistic_style": "Analytic, terse, rant-like, uncertain, casually compressed, or moderately polished prose.",
    "hedging_and_uncertainty": "Degree of confidence, second-guessing, or partial disagreement inside the same review.",
    "specificity_markers": "Realistic concrete references to assignments, lectures, datasets, readings, forums, or deadlines.",
    "recommendation_stance": "Would recommend, only for prepared students, or actively discourage.",
}

PROMPT_CYCLES = [
    {
        "cycle_id": 0,
        "name": "rich_attributes_baseline",
        "goal": "Generate domain-matched reviews using a larger attribute space while preserving aspect-level labels.",
        "additional_instruction": (
            "Write a realistic first-person course review with uneven detail. Avoid textbook sentiment wording, "
            "avoid obvious label leakage, and keep the tone consistent with the sampled student persona."
        ),
    },
    {
        "cycle_id": 1,
        "name": "reduce_synthetic_signatures",
        "goal": "Suppress robotic balance and over-explicit contrasts that make synthetic text easy to detect.",
        "additional_instruction": (
            "Use partial contradictions, hedging, and one or two concrete course-specific details. Do not summarize "
            "every aspect neatly. Let some judgments feel incidental rather than checklist-like."
        ),
    },
    {
        "cycle_id": 2,
        "name": "messier_realism",
        "goal": "Increase natural unevenness and domain realism without collapsing label faithfulness.",
        "additional_instruction": (
            "Allow sentence fragments, imperfect transitions, asymmetric emphasis, grade uncertainty, and references "
            "to forums, projects, grading confusion, or time pressure when appropriate. Keep the text plausible for a real student review."
        ),
    },
]


@dataclass
class ReviewRecord:
    course_code: str
    source_url: str
    semester_raw: str
    review_text: str
    word_count: int


def ensure_dirs() -> None:
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def collapse_repeated_text(text: str) -> str:
    words = text.split()
    for span in range(len(words) // 2, 24, -1):
        first = words[:span]
        second = words[span : 2 * span]
        if len(second) < int(span * 0.9):
            continue
        if first == second:
            return " ".join(first)
    if len(words) >= 80:
        midpoint = len(words) // 2
        left = " ".join(words[:midpoint]).strip()
        right = " ".join(words[midpoint:]).strip()
        if left and right and left[:200] == right[:200]:
            return left
    return text


def parse_real_reviews() -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for course_code, url in COURSE_SOURCES.items():
        html = requests.get(url, timeout=30).text
        soup = BeautifulSoup(html, "html.parser")
        cards = soup.find_all("div", class_="card")
        for card in cards:
            header = card.find("div", class_="card-header")
            body = card.find("div", class_="card-body")
            if header is None or body is None:
                continue
            header_html = str(header)
            semester_match = re.search(r"get_semester_name\('([^']+)'\)", header_html)
            semester_raw = semester_match.group(1) if semester_match else normalize_space(header.get_text(" ", strip=True))

            text_container = body.find("p", class_="card-text")
            if text_container is None:
                continue
            paragraphs = [normalize_space(p.get_text(" ", strip=True)) for p in text_container.find_all("p")]
            if not paragraphs:
                paragraphs = [normalize_space(text_container.get_text(" ", strip=True))]
            paragraphs = [p for p in paragraphs if p]
            if not paragraphs:
                continue
            review_text = collapse_repeated_text(normalize_space(" ".join(paragraphs)))
            word_count = len(review_text.split())
            if word_count < 25:
                continue
            rows.append(
                ReviewRecord(
                    course_code=course_code,
                    source_url=url,
                    semester_raw=semester_raw,
                    review_text=review_text,
                    word_count=word_count,
                ).__dict__
            )
    df = pd.DataFrame(rows).drop_duplicates(subset=["review_text"]).reset_index(drop=True)
    return df


def export_real_review_sample(df: pd.DataFrame, per_course: int = 8) -> pd.DataFrame:
    rng = random.Random(RANDOM_SEED)
    sampled_frames = []
    for course_code, group in df.groupby("course_code"):
        indices = list(group.index)
        rng.shuffle(indices)
        take = min(per_course, len(indices))
        sampled_frames.append(group.loc[indices[:take]])
    sample_df = pd.concat(sampled_frames).sort_values(["course_code", "word_count"], ascending=[True, False]).reset_index(drop=True)
    sample_df.to_csv(VALIDATION_DIR / "real_reviews_omscs_sample.csv", index=False)
    summary = sample_df.groupby("course_code").agg(n_reviews=("review_text", "count"), mean_words=("word_count", "mean"))
    summary.round(2).to_csv(VALIDATION_DIR / "real_reviews_omscs_summary.csv")
    return sample_df


def write_protocol(sample_df: pd.DataFrame) -> None:
    protocol = {
        "real_review_source": "OMSCS Reviews public course pages",
        "source_urls": COURSE_SOURCES,
        "sample_size": int(len(sample_df)),
        "generator_model_default": GENERATOR_MODEL,
        "judge_model_default": JUDGE_MODEL,
        "random_seed": RANDOM_SEED,
        "attribute_space": RICH_ATTRIBUTE_SPACE,
        "prompt_cycles": PROMPT_CYCLES,
        "recommended_protocol": {
            "step_1": "Sample a matched set of real reviews from the public OMSCS source.",
            "step_2": "Generate synthetic reviews with cycle 0 using the richer attribute space.",
            "step_3": "Ask the judge model to label each blind item as real or synthetic and provide confidence.",
            "step_4": "Inspect false positives and false negatives for synthetic signatures.",
            "step_5": "Refine the prompt and rerun cycles 1 and 2.",
            "step_6": "Stop once the judge is near chance or once prompt changes start harming label control.",
        },
        "evaluation_targets": {
            "primary": "Judge accuracy on real-vs-synthetic discrimination",
            "secondary": "Judge confidence gap between real and synthetic items",
            "tertiary": "Qualitative reasons the judge gives for calling a review synthetic",
        },
    }
    (VALIDATION_DIR / "realism_experiment_protocol.json").write_text(
        json.dumps(protocol, indent=2),
        encoding="utf-8",
    )


def load_api_key() -> str:
    env_value = os.environ.get("OPENAI_API_KEY", "").strip()
    if env_value:
        return env_value
    if KEY_FILE.exists():
        return KEY_FILE.read_text(encoding="utf-8").strip()
    raise RuntimeError(f"No OpenAI API key found in OPENAI_API_KEY or {KEY_FILE}")


def build_generator_prompt(attributes: Dict[str, str], cycle_instruction: str) -> str:
    return (
        "You are writing one realistic student course review for research validation.\n"
        "The review must feel like a naturally written student comment rather than a labeled synthetic sample.\n\n"
        "Target attributes:\n"
        + "\n".join(f"- {key}: {value}" for key, value in attributes.items())
        + "\n\n"
        "Requirements:\n"
        "- Keep the review first-person and specific.\n"
        "- Do not mention aspect labels or sentiment labels explicitly.\n"
        "- Do not force a tidy conclusion.\n"
        "- Preserve mixed feelings when the attributes imply them.\n"
        f"- Additional cycle instruction: {cycle_instruction}\n\n"
        "Return only the review text."
    )


def build_judge_prompt(real_text: str, synthetic_text: str) -> str:
    return (
        "You are evaluating whether course reviews are real or synthetic.\n"
        "You will receive two reviews in randomized order. For each review, decide whether it is more likely REAL or SYNTHETIC, "
        "assign a confidence from 0 to 100, and explain the cues briefly.\n"
        "Focus on linguistic naturalness, unevenness, specificity, hesitation, and review-like texture rather than topic match alone.\n\n"
        f"Review A:\n{real_text}\n\n"
        f"Review B:\n{synthetic_text}\n\n"
        "Return strict JSON with keys review_a_label, review_a_confidence, review_a_reason, review_b_label, review_b_confidence, review_b_reason."
    )


def require_openai_client() -> OpenAI:
    if OpenAI is None:
        raise RuntimeError("The openai package is not installed.")
    api_key = load_api_key()
    return OpenAI(api_key=api_key)


def response_text(client: OpenAI, model: str, prompt: str, max_output_tokens: int) -> str:
    response = client.responses.create(
        model=model,
        input=prompt,
        max_output_tokens=max_output_tokens,
    )
    return (response.output_text or "").strip()


def parse_json_object(text: str) -> Dict[str, object]:
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError(f"No JSON object found in response: {text[:400]}")
    return json.loads(match.group(0))


def sample_attribute_bundle(course_code: str, cycle_id: int, rng: random.Random) -> Dict[str, str]:
    return {
        "course_code": course_code,
        "course_title": COURSE_TITLES[course_code],
        "semester": rng.choice(["Fall 2021", "Spring 2022", "Fall 2022", "Spring 2023"]),
        "student_background": rng.choice(BACKGROUND_OPTIONS),
        "motivation": rng.choice(MOTIVATION_OPTIONS),
        "grade_band": rng.choice(GRADE_BANDS),
        "workload_intensity": rng.choice(WORKLOAD_OPTIONS),
        "assessment_profile": rng.choice(ASSESSMENT_OPTIONS),
        "instruction_quality": rng.choice(INSTRUCTION_OPTIONS),
        "support_channel_experience": rng.choice(SUPPORT_OPTIONS),
        "administrative_friction": rng.choice(FRICTION_OPTIONS),
        "emotional_temperature": rng.choice(TEMPERATURE_OPTIONS),
        "linguistic_style": rng.choice(STYLE_OPTIONS),
        "hedging_and_uncertainty": rng.choice(HEDGING_OPTIONS),
        "specificity_markers": rng.choice(COURSE_SPECIFIC_MARKERS[course_code]),
        "recommendation_stance": rng.choice(RECOMMENDATION_OPTIONS),
        "prompt_cycle": str(cycle_id),
    }


def shorten_review(text: str, max_words: int = 220) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]).strip()


def build_refinement_prompt(current_instruction: str, synthetic_flags: List[Dict[str, object]]) -> str:
    reasons = "\n".join(
        f"- confidence={item['confidence']}: {item['reason']}"
        for item in synthetic_flags
    )
    return (
        "You are improving a prompt for generating realistic student course reviews.\n"
        "A judge model reviewed synthetic outputs and explained why some of them looked synthetic.\n"
        "Revise the generation instruction so the next cycle removes those cues while preserving realism and label control.\n\n"
        f"Current instruction:\n{current_instruction}\n\n"
        f"Judge critiques:\n{reasons}\n\n"
        "Return strict JSON with keys revised_instruction, cues_to_avoid, and rationale."
    )


def resolve_cycle_instruction(client: OpenAI, cycle_id: int) -> str:
    base_instruction = next(item["additional_instruction"] for item in PROMPT_CYCLES if item["cycle_id"] == cycle_id)
    if cycle_id == 0:
        return base_instruction
    prior_refinement = VALIDATION_DIR / f"prompt_debug_cycle_{cycle_id - 1}_refinement.json"
    if not prior_refinement.exists():
        return base_instruction
    payload = json.loads(prior_refinement.read_text(encoding="utf-8"))
    revised = str(payload.get("revised_instruction", "")).strip()
    if not revised:
        return base_instruction
    return revised


def refine_generation_prompt(client: OpenAI, cycle_id: int, current_instruction: str, synthetic_flags: List[Dict[str, object]]) -> None:
    if not synthetic_flags:
        payload = {
            "cycle_id": cycle_id,
            "status": "no_refinement_needed",
            "revised_instruction": current_instruction,
            "cues_to_avoid": [],
            "rationale": "No synthetic reviews were explicitly flagged by the judge in this cycle.",
        }
    else:
        prompt = build_refinement_prompt(current_instruction, synthetic_flags)
        refinement_text = response_text(client, JUDGE_MODEL, prompt, max_output_tokens=350)
        payload = parse_json_object(refinement_text)
        payload["cycle_id"] = cycle_id
        payload["status"] = "completed"
    (VALIDATION_DIR / f"prompt_debug_cycle_{cycle_id}_refinement.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def run_debug_prompt(cycle_id: int, sample_size: int) -> None:
    sample_path = VALIDATION_DIR / "real_reviews_omscs_sample.csv"
    if not sample_path.exists():
        raise FileNotFoundError("Run the prepare step first.")

    protocol = {item["cycle_id"]: item for item in PROMPT_CYCLES}
    if cycle_id not in protocol:
        raise ValueError(f"Unknown cycle_id {cycle_id}")

    client = require_openai_client()
    cycle = protocol[cycle_id]
    rng = random.Random(RANDOM_SEED + cycle_id)
    real_df = pd.read_csv(sample_path)
    cycle_instruction = resolve_cycle_instruction(client, cycle_id)

    selected = (
        real_df.groupby("course_code", group_keys=False)
        .head(max(1, sample_size // max(1, real_df["course_code"].nunique())))
        .head(sample_size)
        .reset_index(drop=True)
    )
    while len(selected) < sample_size and len(selected) < len(real_df):
        remainder = real_df.loc[~real_df.index.isin(selected.index)].head(sample_size - len(selected))
        selected = pd.concat([selected, remainder], ignore_index=True)

    generations: List[Dict[str, object]] = []
    judgments: List[Dict[str, object]] = []

    for idx, row in selected.iterrows():
        attributes = sample_attribute_bundle(str(row["course_code"]), cycle_id, rng)
        gen_prompt = build_generator_prompt(attributes, cycle_instruction)
        synthetic_text = response_text(client, GENERATOR_MODEL, gen_prompt, max_output_tokens=500)
        synthetic_text = normalize_space(synthetic_text)
        real_text = shorten_review(normalize_space(str(row["review_text"])))

        pair = [("real", real_text), ("synthetic", synthetic_text)]
        rng.shuffle(pair)
        review_a_truth, review_a_text = pair[0]
        review_b_truth, review_b_text = pair[1]

        judge_prompt = build_judge_prompt(review_a_text, review_b_text)
        judge_raw = response_text(client, JUDGE_MODEL, judge_prompt, max_output_tokens=400)
        judge_json = parse_json_object(judge_raw)

        a_label = str(judge_json["review_a_label"]).strip().lower()
        b_label = str(judge_json["review_b_label"]).strip().lower()
        a_correct = int(a_label == review_a_truth)
        b_correct = int(b_label == review_b_truth)

        generations.append(
            {
                "pair_id": idx,
                "cycle_id": cycle_id,
                "cycle_name": cycle["name"],
                "cycle_instruction": cycle_instruction,
                "course_code": row["course_code"],
                "attributes": json.dumps(attributes, ensure_ascii=False),
                "real_review_text": real_text,
                "synthetic_review_text": synthetic_text,
            }
        )
        judgments.append(
            {
                "pair_id": idx,
                "cycle_id": cycle_id,
                "course_code": row["course_code"],
                "review_a_truth": review_a_truth,
                "review_a_label": a_label,
                "review_a_confidence": judge_json.get("review_a_confidence"),
                "review_a_reason": judge_json.get("review_a_reason"),
                "review_a_correct": a_correct,
                "review_b_truth": review_b_truth,
                "review_b_label": b_label,
                "review_b_confidence": judge_json.get("review_b_confidence"),
                "review_b_reason": judge_json.get("review_b_reason"),
                "review_b_correct": b_correct,
            }
        )

    generations_df = pd.DataFrame(generations)
    judgments_df = pd.DataFrame(judgments)
    generations_df.to_csv(VALIDATION_DIR / f"prompt_debug_cycle_{cycle_id}_generations.csv", index=False)
    judgments_df.to_csv(VALIDATION_DIR / f"prompt_debug_cycle_{cycle_id}_judgments.csv", index=False)

    item_accuracy = float(
        pd.concat(
            [
                judgments_df["review_a_correct"],
                judgments_df["review_b_correct"],
            ],
            ignore_index=True,
        ).mean()
    )
    confidence_gap = float(
        pd.concat(
            [
                judgments_df["review_a_confidence"].astype(float),
                judgments_df["review_b_confidence"].astype(float),
            ],
            ignore_index=True,
        ).mean()
    )
    synthetic_flags = []
    for _, row in judgments_df.iterrows():
        if row["review_a_truth"] == "synthetic":
            synthetic_flags.append(
                {
                    "predicted_label": row["review_a_label"],
                    "confidence": row["review_a_confidence"],
                    "reason": row["review_a_reason"],
                }
            )
        if row["review_b_truth"] == "synthetic":
            synthetic_flags.append(
                {
                    "predicted_label": row["review_b_label"],
                    "confidence": row["review_b_confidence"],
                    "reason": row["review_b_reason"],
                }
            )

    summary = {
        "cycle_id": cycle_id,
        "cycle_name": cycle["name"],
        "cycle_instruction": cycle_instruction,
        "generator_model": GENERATOR_MODEL,
        "judge_model": JUDGE_MODEL,
        "n_pairs": int(len(judgments_df)),
        "judge_item_accuracy": round(item_accuracy, 4),
        "mean_judge_confidence": round(confidence_gap, 2),
        "synthetic_predictions": synthetic_flags,
    }
    (VALIDATION_DIR / f"prompt_debug_cycle_{cycle_id}_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    refine_generation_prompt(client, cycle_id, cycle_instruction, synthetic_flags)
    print(textwrap.dedent(
        f"""
        Prompt debug completed.
        Cycle: {cycle['name']}
        Pairs: {len(judgments_df)}
        Judge item accuracy: {item_accuracy:.4f}
        Mean judge confidence: {confidence_gap:.2f}
        Outputs written to {VALIDATION_DIR}
        """
    ).strip())


def log_cycle_status(cycle_id: int, sample_size: int, status: str, error_message: str = "") -> None:
    cycle_name = next((item["name"] for item in PROMPT_CYCLES if item["cycle_id"] == cycle_id), f"cycle_{cycle_id}")
    payload = {
        "cycle_id": cycle_id,
        "cycle_name": cycle_name,
        "sample_size": sample_size,
        "status": status,
        "generator_model": GENERATOR_MODEL,
        "judge_model": JUDGE_MODEL,
    }
    if error_message:
        payload["error"] = error_message
    (VALIDATION_DIR / f"prompt_debug_cycle_{cycle_id}_status.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def dry_run_summary() -> None:
    sample_path = VALIDATION_DIR / "real_reviews_omscs_sample.csv"
    protocol_path = VALIDATION_DIR / "realism_experiment_protocol.json"
    if not sample_path.exists() or not protocol_path.exists():
        raise FileNotFoundError("Run the prepare step first.")
    print(f"Prepared real review sample: {sample_path}")
    print(f"Prepared experiment protocol: {protocol_path}")
    print(f"Default generator model: {GENERATOR_MODEL}")
    print(f"Default judge model: {JUDGE_MODEL}")
    print("No OpenAI calls have been made in this mode.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a realism-validation experiment for real vs synthetic course reviews.")
    parser.add_argument(
        "mode",
        choices=["prepare", "dry-run-openai", "debug-prompt", "run-cycle-sequence"],
        help="prepare exports real reviews; dry-run-openai checks readiness; debug-prompt runs one cycle; run-cycle-sequence runs iterative prompt refinement.",
    )
    parser.add_argument("--cycle-id", type=int, default=0)
    parser.add_argument("--sample-size", type=int, default=4)
    args = parser.parse_args()

    ensure_dirs()

    if args.mode == "prepare":
        real_df = parse_real_reviews()
        sample_df = export_real_review_sample(real_df)
        write_protocol(sample_df)
        print(f"Saved {len(sample_df)} real reviews to {VALIDATION_DIR}")
        return

    if args.mode == "dry-run-openai":
        dry_run_summary()
        return

    if args.mode == "debug-prompt":
        try:
            log_cycle_status(args.cycle_id, args.sample_size, "started")
            run_debug_prompt(args.cycle_id, args.sample_size)
            log_cycle_status(args.cycle_id, args.sample_size, "completed")
        except Exception as exc:
            log_cycle_status(args.cycle_id, args.sample_size, "failed", str(exc))
            raise
        return

    if args.mode == "run-cycle-sequence":
        for cycle in PROMPT_CYCLES:
            cycle_id = cycle["cycle_id"]
            try:
                log_cycle_status(cycle_id, args.sample_size, "started")
                run_debug_prompt(cycle_id, args.sample_size)
                log_cycle_status(cycle_id, args.sample_size, "completed")
            except Exception as exc:
                log_cycle_status(cycle_id, args.sample_size, "failed", str(exc))
                raise


if __name__ == "__main__":
    main()
