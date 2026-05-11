from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import textwrap
import time
from datetime import datetime, timezone
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
GEN_PROTOCOL_DIR = ROOT / "paper" / "generation_protocol"
SEED_SCHEMA_PATH = GEN_PROTOCOL_DIR / "seed_attribute_schema.json"
ATTEMPT_LOG_PATH = VALIDATION_DIR / "prompt_debug_attempts.jsonl"
STABLE_PROMPT_STATE_PATH = VALIDATION_DIR / "stable_realism_prompt_instruction.json"

COURSE_SOURCES = {
    "CS-6200": "https://awaisrauf.com/omscs_reviews/CS-6200",
    "CS-6250": "https://awaisrauf.com/omscs_reviews/CS-6250",
    "CS-6400": "https://awaisrauf.com/omscs_reviews/CS-6400",
    "CS-7641": "https://awaisrauf.com/omscs_reviews/CS-7641",
}

COURSE_CONTEXT = {
    "CS-6200": {
        "course_title": "Graduate Introduction to Operating Systems",
        "course_name": "Graduate Introduction to Operating Systems",
        "specificity_markers": [
            "mentions projects, kernel concepts, or implementation-heavy weeks",
            "references debugging, systems assignments, or staying up late before deadlines",
            "mentions lectures, project specs, or office hours in a systems-course context",
        ],
    },
    "CS-6250": {
        "course_title": "Computer Networks",
        "course_name": "Computer Networks",
        "specificity_markers": [
            "mentions packet-level thinking, labs, or network-project work",
            "references traces, protocols, or networking assignments in a course-safe way",
            "mentions forum clarifications, lectures, or exam prep in a networking-course context",
        ],
    },
    "CS-6400": {
        "course_title": "Database Systems Concepts and Design",
        "course_name": "Database Systems Concepts and Design",
        "specificity_markers": [
            "mentions schema work, projects, or database-design assignments",
            "references SQL, project coordination, or database tooling in a course-safe way",
            "mentions grading rubrics, project checkpoints, or design writeups in a database-course context",
        ],
    },
    "CS-7641": {
        "course_title": "Machine Learning",
        "course_name": "Machine Learning",
        "specificity_markers": [
            "mentions experiments, reports, or model-tuning work in a course-safe way",
            "references lectures, assignments, or project pressure in a machine-learning-course context",
            "mentions confusion around expectations, writeups, or grading in an ML-course context",
        ],
    },
}

INTERACTIVE_DEBUG_MODEL = "gpt-5.4"
JUDGE_MODEL = "gpt-5.4"
RANDOM_SEED = 42
EQUIVALENCE_MARGIN = 0.10
OPENAI_REQUEST_TIMEOUT_SECONDS = 90
OPENAI_REQUEST_MAX_ATTEMPTS = 4
OPENAI_RETRY_BACKOFF_SECONDS = 5
POLARITY_MODE = "binary_positive_negative"
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
SENTIMENTS = ["positive", "negative"]

RICH_ATTRIBUTE_SPACE = {
    "course_code": "Exact course identifier used in the target review domain.",
    "course_title": "Human-readable course title.",
    "semester": "Term marker such as Fall 2022 or Spring 2023.",
    "student_background": "Prior familiarity with the topic, math maturity, coding fluency, or security/network background.",
    "motivation": "Why the student took the course, such as requirement, curiosity, or specialization fit.",
    "attendance_pattern": "Whether the student kept up regularly, relied on recordings, or caught up around deadlines.",
    "study_context": "Workload context outside the course, such as full-time work or a heavy semester.",
    "grade_band": "Observed or expected performance level, including ambiguity around final grade when natural.",
    "workload_intensity": "Perceived weekly effort and project crunch level.",
    "assessment_profile": "Relative emphasis on projects, exams, reports, quizzes, or participation.",
    "instruction_quality": "Perception of lectures, notes, examples, and conceptual clarity.",
    "support_channel_experience": "Office hours, forums, TAs, Piazza, email responsiveness, or peer support.",
    "administrative_friction": "Confusing rubrics, grading opacity, deadlines, tooling issues, or policy frustration.",
    "feedback_timing": "How quickly and usefully comments or grades arrive.",
    "prerequisite_fit": "Whether the course matches or exceeds the advertised prerequisite level.",
    "collaboration_structure": "Solo work, teamwork, peer learning, and how collaboration is experienced.",
    "platform_and_tooling": "LMS, submission systems, software stacks, or setup friction.",
    "emotional_temperature": "Measured, mixed, enthusiastic, bitter, exhausted, or resigned tone.",
    "linguistic_style": "Analytic, terse, rant-like, uncertain, casually compressed, or moderately polished prose.",
    "hedging_and_uncertainty": "Degree of confidence, second-guessing, or partial disagreement inside the same review.",
    "specificity_markers": "Realistic concrete references to assignments, lectures, datasets, readings, forums, or deadlines.",
    "review_length_band": "Very short, compact, mid-length, or complaint-dominant review length preference.",
    "formality_level": "Informal, neutral, polished, or mixed academic-casual wording.",
    "recommendation_stance": "Would recommend, only for prepared students, or actively discourage.",
    "comparison_frame": "Whether the student compares the course to another course or to prior expectations.",
    "memory_anchor": "One memorable week, assignment, interaction, or turning point that grounds the review.",
}

PROMPT_CYCLES = [
    {
        "cycle_id": 0,
        "name": "binary_rich_attributes_baseline",
        "goal": "Generate domain-matched binary-polarity reviews with recoverable aspect evidence and richer contextual variation.",
        "additional_instruction": (
            "Write a realistic first-person course review with uneven detail. Avoid textbook sentiment wording, "
            "avoid obvious label leakage, and keep the tone consistent with the sampled student persona. "
            "Each target aspect should be expressed clearly enough that a reader can recover a positive or negative stance from the text."
        ),
    },
    {
        "cycle_id": 1,
        "name": "reduce_synthetic_signatures",
        "goal": "Suppress robotic balance and overt checklist structure without weakening binary label recoverability.",
        "additional_instruction": (
            "Use partial contradictions, hedging, and one or two concrete course-specific details. Do not summarize "
            "every aspect neatly. Let some judgments feel incidental rather than checklist-like, but keep target positive and negative stances textually recoverable."
        ),
    },
    {
        "cycle_id": 2,
        "name": "messier_realism",
        "goal": "Increase natural unevenness and course-specific realism without collapsing binary aspect faithfulness.",
        "additional_instruction": (
            "Allow sentence fragments, imperfect transitions, asymmetric emphasis, grade uncertainty, and references "
            "to forums, projects, grading confusion, or time pressure when appropriate. Keep the text plausible for a real student review."
        ),
    },
    {
        "cycle_id": 3,
        "name": "latent_aspect_recoverability",
        "goal": "Make subtle aspects recoverable without resorting to explicit label wording.",
        "additional_instruction": (
            "When a target aspect is subtle, anchor it in one concrete memory, comparison, or consequence so the positive or negative stance is inferable without naming the aspect directly."
        ),
    },
    {
        "cycle_id": 4,
        "name": "anti_overclaim_and_final_polish",
        "goal": "Reduce synthetic overstatement, over-explanation, and tidy endings while preserving exact binary supervision.",
        "additional_instruction": (
            "Prefer specific lived details over generic judgments, avoid wrapping up every point cleanly, and do not introduce extra evaluative themes beyond the sampled target aspects and sampled context."
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
    GEN_PROTOCOL_DIR.mkdir(parents=True, exist_ok=True)


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
        "polarity_mode": POLARITY_MODE,
        "supported_sentiments": SENTIMENTS,
        "interactive_debug_model_default": INTERACTIVE_DEBUG_MODEL,
        "judge_model_default": JUDGE_MODEL,
        "random_seed": RANDOM_SEED,
        "attribute_space": RICH_ATTRIBUTE_SPACE,
        "prompt_cycles": PROMPT_CYCLES,
        "recommended_protocol": {
            "step_1": "Sample a matched set of real reviews from the public OMSCS source.",
            "step_2": "Generate synthetic reviews with cycle 0 using the richer attribute space and binary positive-negative aspect labels.",
            "step_3": "Ask the judge model to label each blind item as real or synthetic and provide confidence.",
            "step_4": "Inspect false positives and false negatives for synthetic signatures.",
            "step_5": "Refine the prompt only between full cycles and rerun the longer 5-cycle sequence.",
            "step_6": "Stop once the judge is near chance or once prompt changes start harming binary label recoverability.",
        },
        "evaluation_targets": {
            "primary": "Judge accuracy on real-vs-synthetic discrimination",
            "secondary": "Judge confidence gap between real and synthetic items",
            "tertiary": "Qualitative reasons the judge gives for calling a review synthetic",
            "chance_confusion_pct": "Normalized confusion metric where 100 means exactly chance-level discrimination",
        },
    }
    (VALIDATION_DIR / "realism_experiment_protocol.json").write_text(
        json.dumps(protocol, indent=2),
        encoding="utf-8",
    )


def append_attempt_log(payload: Dict[str, object]) -> None:
    payload = dict(payload)
    payload["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    with ATTEMPT_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def make_run_id(cycle_id: int, sample_size: int) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"binpol_cycle{cycle_id}_n{sample_size}_{stamp}"


def latest_cycle_path(cycle_id: int, artifact_name: str, suffix: str) -> Path:
    return VALIDATION_DIR / f"prompt_debug_cycle_{cycle_id}_{artifact_name}.{suffix}"


def archived_cycle_path(run_id: str, artifact_name: str, suffix: str) -> Path:
    return VALIDATION_DIR / f"prompt_debug_{run_id}_{artifact_name}.{suffix}"


def write_cycle_progress(
    cycle_id: int,
    run_id: str,
    sample_size: int,
    pair_idx: int,
    total_pairs: int,
    stage: str,
    course_code: str = "",
    note: str = "",
) -> None:
    cycle_name = next((item["name"] for item in PROMPT_CYCLES if item["cycle_id"] == cycle_id), f"cycle_{cycle_id}")
    payload = {
        "cycle_id": cycle_id,
        "cycle_name": cycle_name,
        "run_id": run_id,
        "sample_size": sample_size,
        "pair_index": pair_idx,
        "total_pairs": total_pairs,
        "stage": stage,
        "course_code": course_code,
        "note": note,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    serialized = json.dumps(payload, indent=2)
    latest_cycle_path(cycle_id, "progress", "json").write_text(serialized, encoding="utf-8")
    archived_cycle_path(run_id, "progress", "json").write_text(serialized, encoding="utf-8")


def load_generation_schema() -> Dict[str, object]:
    if not SEED_SCHEMA_PATH.exists():
        raise FileNotFoundError(
            f"Seed schema not found at {SEED_SCHEMA_PATH}. Run generation_protocol_upgrade.py export-seed-schema first."
        )
    payload = json.loads(SEED_SCHEMA_PATH.read_text(encoding="utf-8"))
    return payload


def base_cycle_instruction(cycle_id: int) -> str:
    return next(item["additional_instruction"] for item in PROMPT_CYCLES if item["cycle_id"] == cycle_id)


def load_stable_prompt_state() -> Dict[str, object] | None:
    if not STABLE_PROMPT_STATE_PATH.exists():
        return None
    try:
        return json.loads(STABLE_PROMPT_STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_stable_prompt_state(cycle_id: int, run_id: str, instruction_text: str, source: str) -> None:
    payload = {
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "cycle_id": cycle_id,
        "run_id": run_id,
        "source": source,
        "instruction_text": instruction_text,
    }
    STABLE_PROMPT_STATE_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_api_key() -> str:
    if KEY_FILE.exists():
        return KEY_FILE.read_text(encoding="utf-8").strip()
    env_value = os.environ.get("OPENAI_API_KEY", "").strip()
    if env_value:
        return env_value
    raise RuntimeError(f"No OpenAI API key found in OPENAI_API_KEY or {KEY_FILE}")


def build_generator_prompt(attributes: Dict[str, str], cycle_instruction: str) -> str:
    attributes = dict(attributes)
    aspect_lines = attributes.pop("__aspect_lines__")
    return (
        "You are writing one realistic student course review for research validation.\n"
        "The review must feel like a naturally written student comment rather than a labeled synthetic sample.\n\n"
        f"Target aspect polarities ({POLARITY_MODE}):\n"
        + aspect_lines
        + "\n\n"
        "Target attributes:\n"
        + "\n".join(f"- {key}: {value}" for key, value in attributes.items())
        + "\n\n"
        "Requirements:\n"
        "- Keep the review first-person and specific.\n"
        "- Do not mention aspect labels or polarity labels explicitly.\n"
        "- Do not force a tidy conclusion.\n"
        "- Do not cover every aspect with the same level of detail.\n"
        "- Let at least one point feel incidental rather than checklist-driven.\n"
        "- Do not neutralize a target aspect: each listed aspect should read as clearly positive or clearly negative in context.\n"
        "- It is acceptable for the overall review tone to be mixed, but each target aspect polarity must stay recoverable from the text.\n"
        f"- Additional cycle instruction: {cycle_instruction}\n\n"
        "Return only the review text."
    )


def build_judge_prompt(real_text: str, synthetic_text: str) -> str:
    return (
        "You are evaluating whether course reviews are real or synthetic.\n"
        "You will receive two reviews in randomized order. For each review, decide whether it is more likely REAL or SYNTHETIC, "
        "assign a confidence from 0 to 100, and explain the cues briefly.\n"
        "Focus on linguistic naturalness, unevenness, specificity, hesitation, and review-like texture rather than topic match alone.\n"
        "When you think a review is synthetic, explicitly name the giveaway cues.\n"
        "Use short cue tags chosen from: course_mismatch, generic_specificity, overbalanced_structure, overpolished_prose, stacked_motifs, inconsistent_detail, label_leakage_tone, implausible_memory, weak_persona_fit, other.\n\n"
        f"Review A:\n{real_text}\n\n"
        f"Review B:\n{synthetic_text}\n\n"
        "Return strict JSON with keys "
        "review_a_label, review_a_confidence, review_a_reason, review_a_cue_tags, review_a_justification, "
        "review_b_label, review_b_confidence, review_b_reason, review_b_cue_tags, review_b_justification. "
        "The cue tag fields must be arrays. The justification fields should be 1-3 sentences."
    )


def build_single_review_judge_prompt(review_text: str, review_name: str = "Review") -> str:
    return (
        "You are evaluating whether a course review is real or synthetic.\n"
        "Decide whether the review is more likely REAL or SYNTHETIC, assign a confidence from 0 to 100, "
        "and explain the cues briefly.\n"
        "When you think a review is synthetic, explicitly name the giveaway cues.\n"
        "Use short cue tags chosen from: course_mismatch, generic_specificity, overbalanced_structure, "
        "overpolished_prose, stacked_motifs, inconsistent_detail, label_leakage_tone, implausible_memory, weak_persona_fit, other.\n\n"
        f"{review_name}:\n{review_text}\n\n"
        "Return strict JSON with keys label, confidence, reason, cue_tags, and justification."
    )


def require_openai_client() -> OpenAI:
    if OpenAI is None:
        raise RuntimeError("The openai package is not installed.")
    api_key = load_api_key()
    return OpenAI(api_key=api_key)


def response_text(client: OpenAI, model: str, prompt: str, max_output_tokens: int) -> str:
    last_error: Exception | None = None
    for attempt in range(1, OPENAI_REQUEST_MAX_ATTEMPTS + 1):
        try:
            response = client.responses.create(
                model=model,
                input=prompt,
                max_output_tokens=max_output_tokens,
                timeout=OPENAI_REQUEST_TIMEOUT_SECONDS,
            )
            text = (response.output_text or "").strip()
            if text:
                return text
            raise ValueError(f"Empty response text from model {model} on attempt {attempt}.")
        except Exception as exc:
            last_error = exc
            if attempt >= OPENAI_REQUEST_MAX_ATTEMPTS:
                break
            sleep_seconds = OPENAI_RETRY_BACKOFF_SECONDS * attempt
            print(
                f"[realism] request failure on model={model} attempt={attempt}/{OPENAI_REQUEST_MAX_ATTEMPTS}: "
                f"{exc}. Retrying in {sleep_seconds}s...",
                flush=True,
            )
            time.sleep(sleep_seconds)
    raise RuntimeError(
        f"OpenAI request failed after {OPENAI_REQUEST_MAX_ATTEMPTS} attempts for model {model}: {last_error}"
    ) from last_error


def parse_json_object(text: str) -> Dict[str, object]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    match = re.search(r"\{", text)
    if not match:
        raise ValueError(f"No JSON object found in response: {text[:400]}")
    start = match.start()
    depth = 0
    for idx in range(start, len(text)):
        char = text[idx]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : idx + 1])
    raise ValueError(f"Incomplete JSON object found in response: {text[:400]}")


def coerce_judge_payload(payload: Dict[str, object]) -> Dict[str, object]:
    normalized: Dict[str, object] = {}
    for prefix in ("review_a", "review_b"):
        nested = payload.get(prefix, {})
        if not isinstance(nested, dict):
            nested = {}
        normalized[f"{prefix}_label"] = payload.get(f"{prefix}_label", nested.get("label", nested.get("predicted_label", "")))
        normalized[f"{prefix}_confidence"] = payload.get(f"{prefix}_confidence", nested.get("confidence", 50))
        normalized[f"{prefix}_reason"] = payload.get(f"{prefix}_reason", nested.get("reason", nested.get("justification", "")))
        normalized[f"{prefix}_cue_tags"] = payload.get(f"{prefix}_cue_tags", nested.get("cue_tags", nested.get("cues", [])))
        normalized[f"{prefix}_justification"] = payload.get(
            f"{prefix}_justification",
            nested.get("justification", normalized[f"{prefix}_reason"]),
        )
    return normalized


def judge_labels_present(payload: Dict[str, object]) -> bool:
    a_label = normalize_space(str(payload.get("review_a_label", ""))).lower()
    b_label = normalize_space(str(payload.get("review_b_label", ""))).lower()
    return bool(a_label) and bool(b_label)


def judge_missing_side(client: OpenAI, review_name: str, review_text: str) -> Dict[str, object]:
    prompt = build_single_review_judge_prompt(review_text, review_name=review_name)
    raw = response_text(client, JUDGE_MODEL, prompt, max_output_tokens=300)
    payload = parse_json_object(raw)
    return {
        "label": str(payload.get("label", payload.get("predicted_label", ""))).strip().lower(),
        "confidence": payload.get("confidence", 50),
        "reason": payload.get("reason", payload.get("justification", "")),
        "cue_tags": normalize_cue_tags(payload.get("cue_tags", payload.get("cues", []))),
        "justification": payload.get("justification", payload.get("reason", "")),
    }


def judge_with_retry(
    client: OpenAI,
    prompt: str,
    cycle_id: int,
    pair_idx: int,
    review_a_text: str,
    review_b_text: str,
    max_attempts: int = 3,
) -> Tuple[Dict[str, object], str]:
    last_raw = ""
    repair_prompt = ""
    for attempt in range(1, max_attempts + 1):
        effective_prompt = prompt if attempt == 1 else (
            prompt
            + "\n\nYour previous reply omitted one of the required review outputs. "
            + "Return both Review A and Review B in the exact required JSON schema with no extra text."
        )
        raw = response_text(client, JUDGE_MODEL, effective_prompt, max_output_tokens=500)
        last_raw = raw
        try:
            parsed = coerce_judge_payload(parse_json_object(raw))
        except Exception:
            repair_prompt = raw
            continue
        if judge_labels_present(parsed):
            return parsed, raw
        if normalize_space(str(parsed.get("review_a_label", ""))) and not normalize_space(str(parsed.get("review_b_label", ""))):
            missing = judge_missing_side(client, "Review B", review_b_text)
            parsed["review_b_label"] = missing["label"]
            parsed["review_b_confidence"] = missing["confidence"]
            parsed["review_b_reason"] = missing["reason"]
            parsed["review_b_cue_tags"] = missing["cue_tags"]
            parsed["review_b_justification"] = missing["justification"]
            if judge_labels_present(parsed):
                return parsed, raw
        if normalize_space(str(parsed.get("review_b_label", ""))) and not normalize_space(str(parsed.get("review_a_label", ""))):
            missing = judge_missing_side(client, "Review A", review_a_text)
            parsed["review_a_label"] = missing["label"]
            parsed["review_a_confidence"] = missing["confidence"]
            parsed["review_a_reason"] = missing["reason"]
            parsed["review_a_cue_tags"] = missing["cue_tags"]
            parsed["review_a_justification"] = missing["justification"]
            if judge_labels_present(parsed):
                return parsed, raw
        repair_prompt = raw

    raw_path = VALIDATION_DIR / f"prompt_debug_cycle_{cycle_id}_pair_{pair_idx}_judge_raw.txt"
    raw_path.write_text(last_raw or repair_prompt, encoding="utf-8")
    raise ValueError(f"Judge response missing labels for pair {pair_idx}. Raw response saved to {raw_path}.")


def single_review_judge(client: OpenAI, review_text: str) -> Dict[str, object]:
    raw = response_text(client, JUDGE_MODEL, build_single_review_judge_prompt(review_text), max_output_tokens=300)
    payload = parse_json_object(raw)
    return {
        "label": str(payload.get("label", payload.get("predicted_label", ""))).strip().lower(),
        "confidence": payload.get("confidence", 50),
        "reason": payload.get("reason", payload.get("justification", "")),
        "cue_tags": normalize_cue_tags(payload.get("cue_tags", payload.get("cues", []))),
        "justification": payload.get("justification", payload.get("reason", "")),
        "raw": raw,
    }


def compute_item_confusion(predicted_label: str, true_label: str, confidence_value: object) -> float:
    try:
        confidence = float(confidence_value)
    except Exception:
        confidence = 50.0
    confidence = max(0.0, min(100.0, confidence))
    if predicted_label == true_label:
        return round(1.0 - (confidence / 100.0), 4)
    return 1.0


def compute_chance_confusion_pct(item_accuracy: float) -> float:
    item_accuracy = max(0.0, min(1.0, float(item_accuracy)))
    return round(100.0 * max(0.0, 1.0 - abs(item_accuracy - 0.5) / 0.5), 2)


def binary_entropy_from_probability(probability: float) -> float:
    p = max(0.0, min(1.0, float(probability)))
    q = 1.0 - p
    if p in (0.0, 1.0) or q in (0.0, 1.0):
        return 0.0
    return round(-(p * math.log2(p) + q * math.log2(q)), 4)


def binary_entropy_from_confidence(confidence_value: object) -> float:
    try:
        confidence = float(confidence_value)
    except Exception:
        confidence = 50.0
    p = max(0.5, min(1.0, confidence / 100.0))
    return binary_entropy_from_probability(p)


def binomial_pmf(k: int, n: int, p: float) -> float:
    return math.comb(n, k) * (p ** k) * ((1.0 - p) ** (n - k))


def exact_binomial_test_two_sided(k: int, n: int, p0: float = 0.5) -> float:
    if n <= 0:
        return 1.0
    observed = binomial_pmf(k, n, p0)
    total = 0.0
    for idx in range(n + 1):
        current = binomial_pmf(idx, n, p0)
        if current <= observed + 1e-15:
            total += current
    return min(1.0, total)


def wilson_interval(k: int, n: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    if n <= 0:
        return (0.0, 1.0)
    phat = k / n
    z2 = z * z
    denom = 1.0 + (z2 / n)
    center = (phat + (z2 / (2.0 * n))) / denom
    margin = (z / denom) * math.sqrt((phat * (1.0 - phat) / n) + (z2 / (4.0 * n * n)))
    return (max(0.0, center - margin), min(1.0, center + margin))


def equivalence_to_chance(item_accuracy: float, n_items: int, margin: float = EQUIVALENCE_MARGIN) -> Dict[str, object]:
    correct = int(round(item_accuracy * n_items))
    ci_low, ci_high = wilson_interval(correct, n_items)
    p_value = exact_binomial_test_two_sided(correct, n_items, p0=0.5)
    return {
        "n_items": n_items,
        "correct_items": correct,
        "accuracy": round(item_accuracy, 4),
        "binomial_p_value_vs_chance": round(p_value, 6),
        "wilson_95ci": [round(ci_low, 4), round(ci_high, 4)],
        "equivalence_margin": margin,
        "equivalent_to_chance_within_margin": bool(ci_low >= 0.5 - margin and ci_high <= 0.5 + margin),
        "distance_from_chance": round(abs(item_accuracy - 0.5), 4),
    }


def recommended_n_for_margin(margin: float, z: float = 1.959963984540054) -> int:
    return int(math.ceil((0.25 * (z ** 2)) / (margin ** 2)))


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


def normalize_cue_tags(value: object) -> List[str]:
    if isinstance(value, list):
        tags = [normalize_space(str(item)).lower().replace(" ", "_") for item in value if str(item).strip()]
        return sorted(dict.fromkeys(tags))
    if isinstance(value, str):
        parts = re.split(r"[,;/|]\s*", value)
        tags = [normalize_space(item).lower().replace(" ", "_") for item in parts if normalize_space(item)]
        return sorted(dict.fromkeys(tags))
    return []


def build_refinement_prompt(current_instruction: str, correctly_detected_synthetic: List[Dict[str, object]]) -> str:
    critique_blob = json.dumps(correctly_detected_synthetic, indent=2, ensure_ascii=False)
    return (
        "You are an editor LLM improving a prompt for generating realistic student course reviews.\n"
        "A judge model correctly identified some generated reviews as SYNTHETIC and explained the giveaway cues.\n"
        "Your job is to reformulate the prompt so the next cycle avoids those cues while preserving binary aspect-polarity faithfulness and realistic diversity.\n"
        "Do not make the instruction longer unless needed. Prefer direct, operational constraints that remove detectable artifacts.\n\n"
        f"Current instruction:\n{current_instruction}\n\n"
        f"Correctly detected synthetic cases:\n{critique_blob}\n\n"
        "Return strict JSON with keys revised_instruction, cues_to_avoid, edit_actions, and rationale. "
        "cues_to_avoid must be a concise deduplicated array. edit_actions must describe what changed in the instruction."
    )


def sample_aspect_count(rng: random.Random, distribution: Dict[str, float]) -> int:
    options = [1, 2, 3]
    weights = [distribution.get(str(opt), 1 / 3) for opt in options]
    return rng.choices(options, weights=weights, k=1)[0]


def sample_aspect_labels(rng: random.Random, n_aspects: int) -> Dict[str, str]:
    chosen = rng.sample(ASPECTS, n_aspects)
    return {aspect: rng.choice(SENTIMENTS) for aspect in chosen}


def sample_rich_attributes(rng: random.Random, course_code: str, schema: Dict[str, object]) -> Dict[str, str]:
    schema_obj = schema["schema"]
    policy_obj = schema.get("nuance_sampling_policy", {})
    attributes = {}
    for group_name, group in schema_obj.items():
        policy = policy_obj.get(group_name, {"required": [], "sample_size": len(group)})
        required = [name for name in policy.get("required", []) if name in group]
        optional = [name for name in group.keys() if name not in required]
        target_count = min(len(group), int(policy.get("sample_size", len(group))))
        chosen = required + rng.sample(optional, k=min(max(0, target_count - len(required)), len(optional)))
        for attribute_name in chosen:
            values = group[attribute_name]
            value = rng.choice(values)
            attributes[attribute_name] = value
    attributes["course_code"] = course_code
    course_context = COURSE_CONTEXT.get(course_code, {})
    for key in ("course_title", "course_name"):
        if key in course_context:
            attributes[key] = str(course_context[key])
    if "specificity_markers" in course_context:
        attributes["specificity_markers"] = rng.choice(course_context["specificity_markers"])
    return attributes


def resolve_cycle_instruction(client: OpenAI, cycle_id: int) -> str:
    base_instruction = base_cycle_instruction(cycle_id)
    if cycle_id == 0:
        return base_instruction
    stable_state = load_stable_prompt_state()
    if stable_state:
        instruction_text = str(stable_state.get("instruction_text", "")).strip()
        if instruction_text:
            return instruction_text
    prior_refinement = VALIDATION_DIR / f"prompt_debug_cycle_{cycle_id - 1}_refinement.json"
    if not prior_refinement.exists():
        return base_instruction
    payload = json.loads(prior_refinement.read_text(encoding="utf-8"))
    revised = str(payload.get("revised_instruction", "")).strip()
    if not revised:
        return base_instruction
    return revised


def refine_generation_prompt(
    client: OpenAI,
    cycle_id: int,
    run_id: str,
    current_instruction: str,
    correctly_detected_synthetic: List[Dict[str, object]],
    synthetic_predictions: List[Dict[str, object]],
) -> Dict[str, object]:
    if not correctly_detected_synthetic:
        payload = {
            "cycle_id": cycle_id,
            "status": "no_refinement_needed",
            "revised_instruction": current_instruction,
            "cues_to_avoid": [],
            "edit_actions": [],
            "rationale": "No synthetic reviews were correctly identified as synthetic by the judge in this cycle.",
            "editor_input_count": 0,
        }
    else:
        prompt = build_refinement_prompt(current_instruction, correctly_detected_synthetic)
        refinement_text = response_text(client, JUDGE_MODEL, prompt, max_output_tokens=1200)
        payload = parse_json_object(refinement_text)
        payload["cycle_id"] = cycle_id
        payload["status"] = "completed"
        payload["editor_input_count"] = len(correctly_detected_synthetic)
    payload["run_id"] = run_id
    payload["correctly_detected_synthetic"] = correctly_detected_synthetic
    payload["all_synthetic_predictions"] = synthetic_predictions
    serialized = json.dumps(payload, indent=2)
    latest_cycle_path(cycle_id, "refinement", "json").write_text(serialized, encoding="utf-8")
    archived_cycle_path(run_id, "refinement", "json").write_text(serialized, encoding="utf-8")
    revised_instruction = str(payload.get("revised_instruction", "")).strip()
    if revised_instruction:
        save_stable_prompt_state(
            cycle_id=cycle_id,
            run_id=run_id,
            instruction_text=revised_instruction,
            source="editor_refinement" if payload.get("status") == "completed" else "carry_forward",
        )
    return payload


def run_debug_prompt(cycle_id: int, sample_size: int, run_id: str) -> None:
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
    save_stable_prompt_state(cycle_id=cycle_id, run_id=run_id, instruction_text=cycle_instruction, source="cycle_start")
    schema_payload = load_generation_schema()
    aspect_distribution = schema_payload["recommended_sampling_rule"]["practical_override"]

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
    total_pairs = int(len(selected))
    write_cycle_progress(
        cycle_id=cycle_id,
        run_id=run_id,
        sample_size=sample_size,
        pair_idx=0,
        total_pairs=total_pairs,
        stage="cycle_started",
        note="Beginning binary-polarity realism cycle.",
    )

    for idx, row in selected.iterrows():
        pair_number = int(idx) + 1
        n_aspects = sample_aspect_count(rng, aspect_distribution)
        aspect_labels = sample_aspect_labels(rng, n_aspects)
        attributes = sample_rich_attributes(rng, str(row["course_code"]), schema_payload)
        attributes["__aspect_lines__"] = "\n".join(f"- {k}: {v}" for k, v in aspect_labels.items())
        gen_prompt = build_generator_prompt(attributes, cycle_instruction)
        write_cycle_progress(
            cycle_id=cycle_id,
            run_id=run_id,
            sample_size=sample_size,
            pair_idx=pair_number,
            total_pairs=total_pairs,
            stage="generating_synthetic_review",
            course_code=str(row["course_code"]),
            note=f"Generating pair {pair_number} of {total_pairs}.",
        )
        synthetic_text = response_text(client, INTERACTIVE_DEBUG_MODEL, gen_prompt, max_output_tokens=500)
        synthetic_text = normalize_space(synthetic_text)
        real_text = shorten_review(normalize_space(str(row["review_text"])))

        review_a_truth = "real"
        review_b_truth = "synthetic"
        review_a_text = real_text
        review_b_text = synthetic_text
        judge_mode = "independent_binary"
        write_cycle_progress(
            cycle_id=cycle_id,
            run_id=run_id,
            sample_size=sample_size,
            pair_idx=pair_number,
            total_pairs=total_pairs,
            stage="judging_real_review",
            course_code=str(row["course_code"]),
            note=f"Judging real review for pair {pair_number} of {total_pairs}.",
        )
        review_a_judgment = single_review_judge(client, review_a_text)
        write_cycle_progress(
            cycle_id=cycle_id,
            run_id=run_id,
            sample_size=sample_size,
            pair_idx=pair_number,
            total_pairs=total_pairs,
            stage="judging_synthetic_review",
            course_code=str(row["course_code"]),
            note=f"Judging synthetic review for pair {pair_number} of {total_pairs}.",
        )
        review_b_judgment = single_review_judge(client, review_b_text)
        judge_json = {
            "review_a_label": review_a_judgment["label"],
            "review_a_confidence": review_a_judgment["confidence"],
            "review_a_reason": review_a_judgment["reason"],
            "review_a_cue_tags": review_a_judgment["cue_tags"],
            "review_a_justification": review_a_judgment["justification"],
            "review_b_label": review_b_judgment["label"],
            "review_b_confidence": review_b_judgment["confidence"],
            "review_b_reason": review_b_judgment["reason"],
            "review_b_cue_tags": review_b_judgment["cue_tags"],
            "review_b_justification": review_b_judgment["justification"],
        }
        judge_raw = json.dumps(
            {
                "judge_mode": judge_mode,
                "review_a_raw": review_a_judgment["raw"],
                "review_b_raw": review_b_judgment["raw"],
            },
            ensure_ascii=False,
        )

        a_label = str(judge_json["review_a_label"]).strip().lower()
        b_label = str(judge_json["review_b_label"]).strip().lower()
        a_cue_tags = normalize_cue_tags(judge_json.get("review_a_cue_tags", []))
        b_cue_tags = normalize_cue_tags(judge_json.get("review_b_cue_tags", []))
        a_correct = int(a_label == review_a_truth)
        b_correct = int(b_label == review_b_truth)
        a_confusion = compute_item_confusion(a_label, review_a_truth, judge_json.get("review_a_confidence"))
        b_confusion = compute_item_confusion(b_label, review_b_truth, judge_json.get("review_b_confidence"))
        a_entropy = binary_entropy_from_confidence(judge_json.get("review_a_confidence"))
        b_entropy = binary_entropy_from_confidence(judge_json.get("review_b_confidence"))

        generations.append(
            {
                "pair_id": idx,
                "cycle_id": cycle_id,
                "cycle_name": cycle["name"],
                "cycle_instruction": cycle_instruction,
                "course_code": row["course_code"],
                "n_aspects": n_aspects,
                "aspect_labels": json.dumps(aspect_labels, ensure_ascii=False),
                "attributes": json.dumps(attributes, ensure_ascii=False),
                "generation_prompt": gen_prompt,
                "judge_protocol": judge_mode,
                "real_review_text": real_text,
                "synthetic_review_text": synthetic_text,
                "judge_raw": judge_raw,
            }
        )
        judgments.append(
            {
                "pair_id": idx,
                "cycle_id": cycle_id,
                "course_code": row["course_code"],
                "judge_mode": judge_mode,
                "review_a_truth": review_a_truth,
                "review_a_label": a_label,
                "review_a_confidence": judge_json.get("review_a_confidence"),
                "review_a_reason": judge_json.get("review_a_reason"),
                "review_a_cue_tags": json.dumps(a_cue_tags, ensure_ascii=False),
                "review_a_justification": judge_json.get("review_a_justification", judge_json.get("review_a_reason")),
                "review_a_correct": a_correct,
                "review_a_confusion": a_confusion,
                "review_a_entropy": a_entropy,
                "review_b_truth": review_b_truth,
                "review_b_label": b_label,
                "review_b_confidence": judge_json.get("review_b_confidence"),
                "review_b_reason": judge_json.get("review_b_reason"),
                "review_b_cue_tags": json.dumps(b_cue_tags, ensure_ascii=False),
                "review_b_justification": judge_json.get("review_b_justification", judge_json.get("review_b_reason")),
                "review_b_correct": b_correct,
                "review_b_confusion": b_confusion,
                "review_b_entropy": b_entropy,
            }
        )
        write_cycle_progress(
            cycle_id=cycle_id,
            run_id=run_id,
            sample_size=sample_size,
            pair_idx=pair_number,
            total_pairs=total_pairs,
            stage="pair_completed",
            course_code=str(row["course_code"]),
            note=f"Completed pair {pair_number} of {total_pairs}.",
        )
        print(
            f"[realism] cycle={cycle_id} pair={pair_number}/{total_pairs} course={row['course_code']} completed",
            flush=True,
        )

    generations_df = pd.DataFrame(generations)
    judgments_df = pd.DataFrame(judgments)
    generations_df.to_csv(latest_cycle_path(cycle_id, "generations", "csv"), index=False)
    judgments_df.to_csv(latest_cycle_path(cycle_id, "judgments", "csv"), index=False)
    generations_df.to_csv(archived_cycle_path(run_id, "generations", "csv"), index=False)
    judgments_df.to_csv(archived_cycle_path(run_id, "judgments", "csv"), index=False)

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
    mean_confusion = float(
        pd.concat(
            [
                judgments_df["review_a_confusion"].astype(float),
                judgments_df["review_b_confusion"].astype(float),
            ],
            ignore_index=True,
        ).mean()
    )
    mean_entropy = float(
        pd.concat(
            [
                judgments_df["review_a_entropy"].astype(float),
                judgments_df["review_b_entropy"].astype(float),
            ],
            ignore_index=True,
        ).mean()
    )
    predicted_labels = pd.concat(
        [
            judgments_df["review_a_label"].astype(str).str.strip().str.lower(),
            judgments_df["review_b_label"].astype(str).str.strip().str.lower(),
        ],
        ignore_index=True,
    )
    predicted_real_rate = float((predicted_labels == "real").mean())
    predicted_synthetic_rate = float((predicted_labels == "synthetic").mean())
    empirical_accuracy_entropy_bits = binary_entropy_from_probability(item_accuracy)
    prediction_label_entropy_bits = binary_entropy_from_probability(predicted_real_rate)
    synthetic_predictions = []
    correctly_detected_synthetic = []
    for _, row in judgments_df.iterrows():
        if row["review_a_truth"] == "synthetic":
            record = {
                "pair_id": int(row["pair_id"]),
                "position": "review_a",
                "predicted_label": row["review_a_label"],
                "confidence": row["review_a_confidence"],
                "reason": row["review_a_reason"],
                "justification": row["review_a_justification"],
                "cue_tags": json.loads(row["review_a_cue_tags"]),
                "is_correct": bool(row["review_a_correct"]),
            }
            synthetic_predictions.append(record)
            if row["review_a_correct"] and row["review_a_label"] == "synthetic":
                correctly_detected_synthetic.append(record)
        if row["review_b_truth"] == "synthetic":
            record = {
                "pair_id": int(row["pair_id"]),
                "position": "review_b",
                "predicted_label": row["review_b_label"],
                "confidence": row["review_b_confidence"],
                "reason": row["review_b_reason"],
                "justification": row["review_b_justification"],
                "cue_tags": json.loads(row["review_b_cue_tags"]),
                "is_correct": bool(row["review_b_correct"]),
            }
            synthetic_predictions.append(record)
            if row["review_b_correct"] and row["review_b_label"] == "synthetic":
                correctly_detected_synthetic.append(record)

    cue_histogram: Dict[str, int] = {}
    for item in correctly_detected_synthetic:
        for cue_tag in item["cue_tags"]:
            cue_histogram[cue_tag] = cue_histogram.get(cue_tag, 0) + 1

    summary = {
        "cycle_id": cycle_id,
        "run_id": run_id,
        "cycle_name": cycle["name"],
        "cycle_instruction": cycle_instruction,
        "interactive_debug_model": INTERACTIVE_DEBUG_MODEL,
        "judge_model": JUDGE_MODEL,
        "judge_protocol": "independent_binary_with_justification",
        "n_pairs": int(len(judgments_df)),
        "n_judge_questions": int(len(judgments_df) * 2),
        "judge_item_accuracy": round(item_accuracy, 4),
        "mean_judge_confidence": round(confidence_gap, 2),
        "mean_confusion": round(mean_confusion, 4),
        "mean_entropy_bits": round(mean_entropy, 4),
        "empirical_accuracy_entropy_bits": empirical_accuracy_entropy_bits,
        "prediction_label_entropy_bits": prediction_label_entropy_bits,
        "predicted_real_rate": round(predicted_real_rate, 4),
        "predicted_synthetic_rate": round(predicted_synthetic_rate, 4),
        "chance_confusion_pct": compute_chance_confusion_pct(item_accuracy),
        "synthetic_predictions": synthetic_predictions,
        "correctly_detected_synthetic_count": len(correctly_detected_synthetic),
        "editor_triggered": bool(correctly_detected_synthetic),
        "cue_histogram": cue_histogram,
        "statistical_indistinguishability": equivalence_to_chance(
            item_accuracy=item_accuracy,
            n_items=int(len(judgments_df) * 2),
            margin=EQUIVALENCE_MARGIN,
        ),
        "recommended_items_for_margin_0_10": recommended_n_for_margin(0.10),
        "recommended_items_for_margin_0_05": recommended_n_for_margin(0.05),
    }
    summary_text = json.dumps(summary, indent=2)
    latest_cycle_path(cycle_id, "summary", "json").write_text(summary_text, encoding="utf-8")
    archived_cycle_path(run_id, "summary", "json").write_text(summary_text, encoding="utf-8")
    refinement_payload = refine_generation_prompt(
        client,
        cycle_id,
        run_id,
        cycle_instruction,
        correctly_detected_synthetic,
        synthetic_predictions,
    )
    append_attempt_log(
        {
            "cycle_id": cycle_id,
            "cycle_name": cycle["name"],
            "status": "completed",
            "sample_size": sample_size,
            "run_id": run_id,
            "judge_protocol": "independent_binary_with_justification",
            "judge_item_accuracy": round(item_accuracy, 4),
            "mean_judge_confidence": round(confidence_gap, 2),
            "mean_confusion": round(mean_confusion, 4),
            "mean_entropy_bits": round(mean_entropy, 4),
            "empirical_accuracy_entropy_bits": empirical_accuracy_entropy_bits,
            "prediction_label_entropy_bits": prediction_label_entropy_bits,
            "predicted_real_rate": round(predicted_real_rate, 4),
            "predicted_synthetic_rate": round(predicted_synthetic_rate, 4),
            "chance_confusion_pct": compute_chance_confusion_pct(item_accuracy),
            "cycle_instruction": cycle_instruction,
            "correctly_detected_synthetic_count": len(correctly_detected_synthetic),
            "cue_histogram": cue_histogram,
            "refinement_status": refinement_payload.get("status"),
        }
    )
    write_cycle_progress(
        cycle_id=cycle_id,
        run_id=run_id,
        sample_size=sample_size,
        pair_idx=total_pairs,
        total_pairs=total_pairs,
        stage="cycle_completed",
        note="Cycle outputs written and refinement completed.",
    )
    print(textwrap.dedent(
        f"""
        Prompt debug completed.
        Run ID: {run_id}
        Cycle: {cycle['name']}
        Pairs: {len(judgments_df)}
        Judge item accuracy: {item_accuracy:.4f}
        Mean judge confidence: {confidence_gap:.2f}
        Empirical accuracy entropy (bits): {empirical_accuracy_entropy_bits:.4f}
        Prediction label entropy (bits): {prediction_label_entropy_bits:.4f}
        Chance confusion (%): {compute_chance_confusion_pct(item_accuracy):.2f}
        Correctly detected synthetic items: {len(correctly_detected_synthetic)}
        Outputs written to {VALIDATION_DIR}
        """
    ).strip())


def log_cycle_status(cycle_id: int, sample_size: int, run_id: str, status: str, error_message: str = "") -> None:
    cycle_name = next((item["name"] for item in PROMPT_CYCLES if item["cycle_id"] == cycle_id), f"cycle_{cycle_id}")
    payload = {
        "cycle_id": cycle_id,
        "cycle_name": cycle_name,
        "sample_size": sample_size,
        "run_id": run_id,
        "status": status,
        "interactive_debug_model": INTERACTIVE_DEBUG_MODEL,
        "judge_model": JUDGE_MODEL,
    }
    if error_message:
        payload["error"] = error_message
    payload_text = json.dumps(payload, indent=2)
    latest_cycle_path(cycle_id, "status", "json").write_text(payload_text, encoding="utf-8")
    archived_cycle_path(run_id, "status", "json").write_text(payload_text, encoding="utf-8")
    append_attempt_log(payload)


def dry_run_summary() -> None:
    sample_path = VALIDATION_DIR / "real_reviews_omscs_sample.csv"
    protocol_path = VALIDATION_DIR / "realism_experiment_protocol.json"
    if not sample_path.exists() or not protocol_path.exists():
        raise FileNotFoundError("Run the prepare step first.")
    print(f"Prepared real review sample: {sample_path}")
    print(f"Prepared experiment protocol: {protocol_path}")
    print(f"Default interactive debug model: {INTERACTIVE_DEBUG_MODEL}")
    print(f"Default judge model: {JUDGE_MODEL}")
    print("No OpenAI calls have been made in this mode.")


def summarize_cycle_improvements() -> None:
    rows: List[Dict[str, object]] = []
    previous: Dict[str, object] | None = None
    for cycle in PROMPT_CYCLES:
        summary_path = latest_cycle_path(cycle["cycle_id"], "summary", "json")
        if not summary_path.exists():
            continue
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        current = {
            "cycle_id": int(payload["cycle_id"]),
            "cycle_name": str(payload["cycle_name"]),
            "run_id": str(payload.get("run_id", "")),
            "n_pairs": int(payload.get("n_pairs", 0)),
            "n_judge_questions": int(payload.get("n_judge_questions", 0)),
            "judge_item_accuracy": float(payload.get("judge_item_accuracy", 0.0)),
            "judge_item_accuracy_distance_to_chance": abs(float(payload.get("judge_item_accuracy", 0.0)) - 0.5),
            "mean_judge_confidence": float(payload.get("mean_judge_confidence", 0.0)),
            "mean_judge_confidence_distance_to_50": abs(float(payload.get("mean_judge_confidence", 0.0)) - 50.0),
            "mean_confusion": float(payload.get("mean_confusion", 0.0)),
            "mean_entropy_bits": float(payload.get("mean_entropy_bits", 0.0)),
            "empirical_accuracy_entropy_bits": float(payload.get("empirical_accuracy_entropy_bits", 0.0)),
            "prediction_label_entropy_bits": float(payload.get("prediction_label_entropy_bits", 0.0)),
            "predicted_real_rate": float(payload.get("predicted_real_rate", 0.0)),
            "predicted_synthetic_rate": float(payload.get("predicted_synthetic_rate", 0.0)),
            "chance_confusion_pct": float(payload.get("chance_confusion_pct", 0.0)),
            "chance_confusion_gap_to_100": abs(float(payload.get("chance_confusion_pct", 0.0)) - 100.0),
            "correctly_detected_synthetic_count": int(payload.get("correctly_detected_synthetic_count", 0)),
            "editor_triggered": bool(payload.get("editor_triggered", False)),
            "statistical_indistinguishability_pass": bool(
                payload.get("statistical_indistinguishability", {}).get("equivalent_to_chance", False)
            ),
        }
        if previous is None:
            enriched = {
                **current,
                "delta_judge_item_accuracy": "",
                "delta_accuracy_distance_to_chance": "",
                "delta_mean_judge_confidence": "",
                "delta_confidence_distance_to_50": "",
                "delta_mean_confusion": "",
                "delta_mean_entropy_bits": "",
                "delta_empirical_accuracy_entropy_bits": "",
                "delta_prediction_label_entropy_bits": "",
                "delta_predicted_real_rate": "",
                "delta_predicted_synthetic_rate": "",
                "delta_chance_confusion_pct": "",
                "delta_chance_confusion_gap_to_100": "",
                "delta_correctly_detected_synthetic_count": "",
            }
        else:
            enriched = {
                **current,
                "delta_judge_item_accuracy": round(current["judge_item_accuracy"] - previous["judge_item_accuracy"], 4),
                "delta_accuracy_distance_to_chance": round(
                    current["judge_item_accuracy_distance_to_chance"] - previous["judge_item_accuracy_distance_to_chance"], 4
                ),
                "delta_mean_judge_confidence": round(
                    current["mean_judge_confidence"] - previous["mean_judge_confidence"], 2
                ),
                "delta_confidence_distance_to_50": round(
                    current["mean_judge_confidence_distance_to_50"] - previous["mean_judge_confidence_distance_to_50"], 2
                ),
                "delta_mean_confusion": round(current["mean_confusion"] - previous["mean_confusion"], 4),
                "delta_mean_entropy_bits": round(current["mean_entropy_bits"] - previous["mean_entropy_bits"], 4),
                "delta_empirical_accuracy_entropy_bits": round(
                    current["empirical_accuracy_entropy_bits"] - previous["empirical_accuracy_entropy_bits"], 4
                ),
                "delta_prediction_label_entropy_bits": round(
                    current["prediction_label_entropy_bits"] - previous["prediction_label_entropy_bits"], 4
                ),
                "delta_predicted_real_rate": round(current["predicted_real_rate"] - previous["predicted_real_rate"], 4),
                "delta_predicted_synthetic_rate": round(
                    current["predicted_synthetic_rate"] - previous["predicted_synthetic_rate"], 4
                ),
                "delta_chance_confusion_pct": round(
                    current["chance_confusion_pct"] - previous["chance_confusion_pct"], 2
                ),
                "delta_chance_confusion_gap_to_100": round(
                    current["chance_confusion_gap_to_100"] - previous["chance_confusion_gap_to_100"], 2
                ),
                "delta_correctly_detected_synthetic_count": (
                    current["correctly_detected_synthetic_count"] - previous["correctly_detected_synthetic_count"]
                ),
            }
        rows.append(enriched)
        previous = current

    if not rows:
        raise FileNotFoundError("No cycle summary files found yet in the validation directory.")

    df = pd.DataFrame(rows)
    csv_path = VALIDATION_DIR / "realism_cycle_improvement_summary.csv"
    json_path = VALIDATION_DIR / "realism_cycle_improvement_summary.json"
    md_path = VALIDATION_DIR / "realism_cycle_improvement_summary.md"
    df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    markdown_lines = [
        "# Realism Cycle Improvement Summary",
        "",
        "This table compares each completed realism cycle against the previous completed cycle.",
        "",
        "| Cycle | Accuracy | Δ Accuracy | |acc-0.5| | Δ |acc-0.5| | Mean conf | Δ conf | |conf-50| | Δ |conf-50| | Mean confusion | Δ confusion | Confidence entropy | Δ conf entropy | Accuracy entropy | Δ acc entropy | Label entropy | Δ label entropy | Pred real | Δ pred real | Pred synth | Δ pred synth | Chance confusion % | Δ chance confusion % | Detected synthetic | Δ detected synthetic | Eq. to chance |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in rows:
        markdown_lines.append(
            "| "
            + f"{row['cycle_id']} {row['cycle_name']} | "
            + f"{row['judge_item_accuracy']:.4f} | {row['delta_judge_item_accuracy']} | "
            + f"{row['judge_item_accuracy_distance_to_chance']:.4f} | {row['delta_accuracy_distance_to_chance']} | "
            + f"{row['mean_judge_confidence']:.2f} | {row['delta_mean_judge_confidence']} | "
            + f"{row['mean_judge_confidence_distance_to_50']:.2f} | {row['delta_confidence_distance_to_50']} | "
            + f"{row['mean_confusion']:.4f} | {row['delta_mean_confusion']} | "
            + f"{row['mean_entropy_bits']:.4f} | {row['delta_mean_entropy_bits']} | "
            + f"{row['empirical_accuracy_entropy_bits']:.4f} | {row['delta_empirical_accuracy_entropy_bits']} | "
            + f"{row['prediction_label_entropy_bits']:.4f} | {row['delta_prediction_label_entropy_bits']} | "
            + f"{row['predicted_real_rate']:.4f} | {row['delta_predicted_real_rate']} | "
            + f"{row['predicted_synthetic_rate']:.4f} | {row['delta_predicted_synthetic_rate']} | "
            + f"{row['chance_confusion_pct']:.2f} | {row['delta_chance_confusion_pct']} | "
            + f"{row['correctly_detected_synthetic_count']} | {row['delta_correctly_detected_synthetic_count']} | "
            + ("yes" if row["statistical_indistinguishability_pass"] else "no")
            + " |"
        )
    md_path.write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")
    print(f"Wrote realism improvement summaries to {csv_path}, {json_path}, and {md_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a realism-validation experiment for real vs synthetic course reviews.")
    parser.add_argument(
        "mode",
        choices=["prepare", "dry-run-openai", "debug-prompt", "run-cycle-sequence", "summarize-sequence"],
        help="prepare exports real reviews; dry-run-openai checks readiness; debug-prompt runs one cycle; run-cycle-sequence runs iterative prompt refinement; summarize-sequence compares completed cycles.",
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
        run_id = make_run_id(args.cycle_id, args.sample_size)
        try:
            log_cycle_status(args.cycle_id, args.sample_size, run_id, "started")
            run_debug_prompt(args.cycle_id, args.sample_size, run_id)
            log_cycle_status(args.cycle_id, args.sample_size, run_id, "completed")
        except Exception as exc:
            log_cycle_status(args.cycle_id, args.sample_size, run_id, "failed", str(exc))
            raise
        return

    if args.mode == "summarize-sequence":
        summarize_cycle_improvements()
        return

    if args.mode == "run-cycle-sequence":
        bootstrap_run_id = make_run_id(0, args.sample_size)
        save_stable_prompt_state(
            cycle_id=0,
            run_id=bootstrap_run_id,
            instruction_text=base_cycle_instruction(0),
            source="sequence_bootstrap",
        )
        for cycle in PROMPT_CYCLES:
            cycle_id = cycle["cycle_id"]
            run_id = make_run_id(cycle_id, args.sample_size)
            try:
                log_cycle_status(cycle_id, args.sample_size, run_id, "started")
                run_debug_prompt(cycle_id, args.sample_size, run_id)
                log_cycle_status(cycle_id, args.sample_size, run_id, "completed")
            except Exception as exc:
                log_cycle_status(cycle_id, args.sample_size, run_id, "failed", str(exc))
                raise


if __name__ == "__main__":
    main()
