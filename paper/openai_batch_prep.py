from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parents[1]
GEN_PROTOCOL_DIR = ROOT / "paper" / "generation_protocol"
VALIDATION_DIR = ROOT / "paper" / "validation"
BATCH_DIR = ROOT / "paper" / "batch_requests"

SCHEMA_PATH = GEN_PROTOCOL_DIR / "seed_attribute_schema.json"
TEMPLATE_PATH = GEN_PROTOCOL_DIR / "final_realism_prompt_template.txt"
METADATA_PATH = GEN_PROTOCOL_DIR / "final_realism_prompt_metadata.json"
REAL_REVIEWS_PATH = VALIDATION_DIR / "real_reviews_omscs_sample.csv"

GENERATOR_MODEL = "gpt-5-nano"
JUDGE_MODEL = "gpt-5.4"
SENTIMENTS = ["positive", "negative", "neutral"]
GENERATOR_MAX_OUTPUT_TOKENS = 420
GENERATOR_REASONING_EFFORT = "minimal"
GENERATOR_TEXT_VERBOSITY = "low"
ALWAYS_INCLUDE_ATTRIBUTES = {"linguistic_diversity": ["review_length_band"]}


def ensure_dirs() -> None:
    BATCH_DIR.mkdir(parents=True, exist_ok=True)


def load_schema() -> Dict[str, object]:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def load_template() -> str:
    return TEMPLATE_PATH.read_text(encoding="utf-8")


def load_metadata() -> Dict[str, object]:
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


def load_real_reviews(limit: int | None = None) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with REAL_REVIEWS_PATH.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def sample_aspect_labels(rng: random.Random, aspects: List[str], distribution: Dict[str, float]) -> Dict[str, str]:
    options = [1, 2, 3]
    weights = [distribution[str(option)] for option in options]
    n_aspects = rng.choices(options, weights=weights, k=1)[0]
    selected = rng.sample(aspects, n_aspects)
    return {aspect: rng.choice(SENTIMENTS) for aspect in selected}


def sample_attributes(schema: Dict[str, object], rng: random.Random) -> Dict[str, str]:
    sampled: Dict[str, str] = {}
    schema_obj = schema["schema"]
    policy_obj = schema["nuance_sampling_policy"]
    for group_name, group in schema_obj.items():
        policy = policy_obj.get(group_name, {"required": [], "sample_size": len(group)})
        required = [name for name in policy.get("required", []) if name in group]
        required.extend([name for name in ALWAYS_INCLUDE_ATTRIBUTES.get(group_name, []) if name in group and name not in required])
        optional = [name for name in group if name not in required]
        target_count = min(len(group), int(policy.get("sample_size", len(group))))
        take_optional = min(max(0, target_count - len(required)), len(optional))
        chosen = required + rng.sample(optional, k=take_optional)
        for attribute_name in chosen:
            sampled[attribute_name] = rng.choice(group[attribute_name])
    return sampled


def render_prompt(template: str, aspect_labels: Dict[str, str], attributes: Dict[str, str]) -> str:
    aspect_block = "\n".join(f"- {aspect}: {sentiment}" for aspect, sentiment in aspect_labels.items())
    attribute_block = "\n".join(f"- {key}: {value}" for key, value in attributes.items())
    prompt = template.format(aspect_block=aspect_block, attribute_block=attribute_block)
    length_guidance = review_length_instruction(attributes.get("review_length_band", ""))
    if length_guidance:
        prompt = prompt.replace(
            "\n\nReturn only the review text.\n",
            f"\n\nLength guidance:\n- {length_guidance}\n- Treat the length guidance as a hard constraint that overrides any generic tendency to elaborate.\n\nReturn only the review text.\n",
        )
    return prompt


def batch_line(custom_id: str, model: str, prompt: str, max_output_tokens: int) -> Dict[str, object]:
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/responses",
        "body": {
            "model": model,
            "input": prompt,
            "max_output_tokens": max_output_tokens,
            "reasoning": {"effort": GENERATOR_REASONING_EFFORT},
            "text": {"verbosity": GENERATOR_TEXT_VERBOSITY},
        },
    }


def review_length_instruction(review_length_band: str) -> str:
    normalized = review_length_band.strip().lower()
    if normalized == "very short comment":
        return "Write exactly one short paragraph of 20-40 words, ideally 1-2 sentences, and do not exceed 45 words. If needed, cut specifics rather than exceed the limit."
    if normalized == "compact but informative review":
        return "Write one compact paragraph of 45-75 words, ideally 2-3 sentences, and do not exceed 85 words. If needed, cut secondary detail rather than exceed the limit."
    if normalized == "mid-length reflective review":
        return "Write one mid-length paragraph of 85-125 words, ideally 3-5 sentences, and do not exceed 140 words. If needed, cut secondary detail rather than exceed the limit."
    if normalized == "detailed review with one dominant complaint":
        return "Write one detailed paragraph of 140-190 words, ideally 5-7 sentences, and do not exceed 210 words. If needed, cut side comments rather than exceed the limit."
    return "Write one concise paragraph of 55-95 words, ideally 2-4 sentences, and do not exceed 105 words. If needed, cut secondary detail rather than exceed the limit."


def generation_max_output_tokens(review_length_band: str, n_aspects: int) -> int:
    normalized = review_length_band.strip().lower()
    if normalized == "very short comment":
        return 90 if n_aspects <= 1 else 110
    if normalized == "compact but informative review":
        return 120 if n_aspects <= 2 else 140
    if normalized == "mid-length reflective review":
        return 210 if n_aspects <= 2 else 230
    if normalized == "detailed review with one dominant complaint":
        return 300 if n_aspects <= 2 else 330
    return 160 if n_aspects <= 2 else 180


def write_jsonl(path: Path, items: List[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def prepare_generation_requests(limit: int, prefix: str, seed: int) -> None:
    schema = load_schema()
    template = load_template()
    metadata = load_metadata()
    aspects = list(metadata["aspect_inventory"].keys())
    distribution = metadata["recommended_aspect_count_distribution"]
    rng = random.Random(seed)

    requests: List[Dict[str, object]] = []
    manifest_rows: List[Dict[str, str]] = []
    for idx in range(limit):
        aspect_labels = sample_aspect_labels(rng, aspects, distribution)
        attributes = sample_attributes(schema, rng)
        prompt = render_prompt(template, aspect_labels, attributes)
        custom_id = f"gen_{idx}"
        max_output_tokens = generation_max_output_tokens(attributes.get("review_length_band", ""), len(aspect_labels))
        requests.append(batch_line(custom_id, GENERATOR_MODEL, prompt, max_output_tokens))
        manifest_rows.append(
            {
                "custom_id": custom_id,
                "sample_id": str(idx),
                "n_aspects": str(len(aspect_labels)),
                "target_attributes": json.dumps(aspect_labels, ensure_ascii=False),
                "nuance_attributes": json.dumps(attributes, ensure_ascii=False),
                "model": GENERATOR_MODEL,
                "prompt_run_id": str(metadata.get("selected_prompt_run_id", "")),
                "max_output_tokens": str(max_output_tokens),
                "reasoning_effort": GENERATOR_REASONING_EFFORT,
                "text_verbosity": GENERATOR_TEXT_VERBOSITY,
                "review_length_guidance": review_length_instruction(attributes.get("review_length_band", "")),
            }
        )

    write_jsonl(BATCH_DIR / f"{prefix}_requests.jsonl", requests)
    with (BATCH_DIR / f"{prefix}_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)


def prepare_schema_review_requests() -> None:
    schema_payload = load_schema()
    template = load_template()
    prompt_1 = (
        "Review this educational ABSA generation schema for diversity, realism, and label leakage risk. "
        "Return concise JSON with keys diversity_gaps, realism_gaps, prompt_revisions, sampling_revisions, and final_verdict.\n\n"
        f"Schema:\n{json.dumps(schema_payload['schema'], indent=2)}\n\n"
        f"Prompt template:\n{template}"
    )
    prompt_2 = (
        "Review this educational ABSA generation protocol from the perspective of a skeptical paper reviewer. "
        "Return concise JSON with keys strengths, weaknesses, threats_to_validity, and batch_readiness.\n\n"
        f"Schema metadata:\n{json.dumps(schema_payload, indent=2)}"
    )
    requests = [
        batch_line("schema_review_diversity", GENERATOR_MODEL, prompt_1, 1200),
        batch_line("schema_review_reviewer", JUDGE_MODEL, prompt_2, 1200),
    ]
    write_jsonl(BATCH_DIR / "schema_review_requests.jsonl", requests)


def prepare_judge_template(limit: int) -> None:
    real_reviews = load_real_reviews(limit=limit)
    rows = []
    for idx, row in enumerate(real_reviews):
        rows.append(
            {
                "judge_pair_id": idx,
                "real_review_source": row["source_url"],
                "real_review_text": row["review_text"],
                "synthetic_custom_id_placeholder": f"gen_{idx}",
                "notes": "Fill synthetic_review_text after generation batch completes, then build judge batch from this manifest.",
            }
        )
    with (BATCH_DIR / "judge_pair_template.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["judge_pair_id"])
        writer.writeheader()
        writer.writerows(rows)


def write_batch_readme(prefix: str, generation_limit: int) -> None:
    text = f"""# OpenAI Batch Preparation

This folder contains batch-ready request files only. Nothing here executes API jobs.

Files:
- `schema_review_requests.jsonl`
  Small batch for protocol review.
- `{prefix}_requests.jsonl`
  Generation requests using the current stable realism prompt package.
- `{prefix}_manifest.csv`
  Manifest containing target aspects, nuance attributes, model, and prompt lineage.
- `judge_pair_template.csv`
  Template that pairs real reviews with placeholders for generated synthetic outputs.

Batch boundary:
- generation model: `{GENERATOR_MODEL}`
- judge/protocol-review model: `{JUDGE_MODEL}`
- prepared generation size: `{generation_limit}`
- generation max output tokens: band-dependent, default upper bound `{GENERATOR_MAX_OUTPUT_TOKENS}`
- generation reasoning effort: `{GENERATOR_REASONING_EFFORT}`
- generation text verbosity: `{GENERATOR_TEXT_VERBOSITY}`
- prompt source: `paper/generation_protocol/final_realism_prompt_template.txt`
- aspect inventory: `20` forward-protocol aspects
"""
    (BATCH_DIR / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare batch request files for OpenAI-based educational ABSA generation.")
    parser.add_argument("--generation-limit", type=int, default=10000)
    parser.add_argument("--judge-limit", type=int, default=30)
    parser.add_argument("--generation-prefix", default="dataset_generation_10k")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ensure_dirs()
    prepare_schema_review_requests()
    prepare_generation_requests(args.generation_limit, args.generation_prefix, args.seed)
    prepare_judge_template(args.judge_limit)
    write_batch_readme(args.generation_prefix, args.generation_limit)
    print(f"Batch request files written to {BATCH_DIR}")


if __name__ == "__main__":
    main()
