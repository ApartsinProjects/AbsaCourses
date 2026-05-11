"""Shared constants and helpers for the human-labeling samplers and scorers."""
from __future__ import annotations

import json
import string
from pathlib import Path
from typing import Dict, List, Sequence

ROOT = Path(__file__).resolve().parents[2]
HUMAN_ROOT = ROOT / "human"

# Data sources
SYNTHETIC_PATH = ROOT / "paper" / "generated_datasets" / "batch_69cc15c483488190941478aa4e3a976d_generated_reviews.jsonl"
HERATH_PATH = ROOT / "paper" / "real_transfer" / "herath_mapped_real_reviews.jsonl"
AUDIT_DETAILS_PATH = ROOT / "paper" / "faithfulness_audit" / "faithfulness_audit_gpt-5_2_250_details.csv"
HERATH_MAPPING_PATH = ROOT / "paper" / "real_transfer" / "herath_mapping.json"

# The 20-aspect inventory, ordered by pedagogical group then by name.
ASPECTS: List[str] = [
    # Instructional quality
    "clarity",
    "lecturer_quality",
    "materials",
    "feedback_quality",
    # Assessment and course management
    "exam_fairness",
    "assessment_design",
    "grading_transparency",
    "organization",
    "tooling_usability",
    # Learning demand and readiness
    "difficulty",
    "workload",
    "pacing",
    "prerequisite_fit",
    # Learning environment
    "support",
    "accessibility",
    "peer_interaction",
    # Engagement and value
    "relevance",
    "interest",
    "practical_application",
    "overall_experience",
]

POLARITIES = ("positive", "neutral", "negative")
DISCUSSED_VALUES = ("yes", "no", "unclear")


def rater_letters(n: int) -> List[str]:
    """A, B, C, ... up to n raters. Fails for n > 26 because we never need more."""
    if n < 1 or n > 26:
        raise ValueError(f"n_raters must be in [1, 26], got {n}")
    return list(string.ascii_uppercase[:n])


def load_jsonl(path: Path) -> List[Dict[str, object]]:
    """Read a JSONL file into a list of dicts. Skips blank lines."""
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def word_count(text: str) -> int:
    return len(text.split())


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
