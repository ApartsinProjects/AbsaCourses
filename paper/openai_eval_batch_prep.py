from __future__ import annotations

import argparse
import json
from pathlib import Path

from absa_model_comparison import (
    DEFAULT_OPENAI_VARIANTS,
    OPENAI_VARIANT_SHOTS,
    build_aspect_map_text_format,
    build_openai_demonstrations,
    build_openai_prompt,
    discover_aspects,
    load_jsonl,
    resolve_data_path,
    retrieve_similar_examples,
    three_way_split,
)


ROOT = Path(__file__).resolve().parents[1]
BATCH_DIR = ROOT / "paper" / "batch_requests"
DEFAULT_DATA_PATH = ROOT / "paper" / "generated_datasets" / "batch_69cc15c483488190941478aa4e3a976d_generated_reviews.jsonl"
SUPPORTED_BATCH_VARIANTS = ["zero-shot", "zero-shot-glossary", "few-shot", "few-shot-diverse", "retrieval-few-shot"]
DEFAULT_BATCH_MODELS = ["gpt-5.2", "gpt-5-mini"]


def reasoning_config_for_model(model: str) -> dict[str, str]:
    model_lower = model.lower()
    if model_lower.startswith("gpt-5.2"):
        return {"effort": "none"}
    if model_lower.startswith("gpt-5-mini") or model_lower.startswith("gpt-5-nano"):
        return {"effort": "minimal"}
    if model_lower.startswith("gpt-5"):
        return {"effort": "low"}
    return {"effort": "minimal"}


def prepare_eval_requests(
    data_path: Path,
    prefix: str,
    models: list[str],
    variants: list[str],
    test_limit: int,
    seed: int,
) -> tuple[Path, Path, Path]:
    df = load_jsonl(resolve_data_path(data_path))
    aspects = discover_aspects(df)
    train_df, _, test_df = three_way_split(df, 0.10, 0.10, seed)
    if test_limit > 0:
        test_df = test_df.head(test_limit).reset_index(drop=True)

    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    request_path = BATCH_DIR / f"{prefix}_requests.jsonl"
    manifest_path = BATCH_DIR / f"{prefix}_manifest.csv"
    metadata_path = BATCH_DIR / f"{prefix}_metadata.json"

    manifest_rows = []
    base_demos_by_variant = {}
    for variant in variants:
        if variant not in SUPPORTED_BATCH_VARIANTS:
            raise ValueError(f"Variant {variant} is not supported in the batch prep path.")
        base_demos_by_variant[variant] = build_openai_demonstrations(train_df, variant, OPENAI_VARIANT_SHOTS[variant], seed)

    with request_path.open("w", encoding="utf-8") as out:
        for model in models:
            for variant in variants:
                base_demos = base_demos_by_variant[variant]
                for row_idx, row in test_df.iterrows():
                    demos = base_demos
                    if variant == "retrieval-few-shot":
                        demos = retrieve_similar_examples(
                            train_df, row["text"], OPENAI_VARIANT_SHOTS[variant], seed + row_idx
                        )
                    prompt = build_openai_prompt(str(row["text"]), aspects, variant, demos)
                    safe_model = model.replace(".", "_")
                    custom_id = f"{safe_model}__{variant}__test_{row_idx}"
                    request = {
                        "custom_id": custom_id,
                        "method": "POST",
                        "url": "/v1/responses",
                        "body": {
                            "model": model,
                            "input": prompt,
                            "max_output_tokens": 260,
                            "reasoning": reasoning_config_for_model(model),
                            "text": build_aspect_map_text_format(aspects),
                        },
                    }
                    out.write(json.dumps(request, ensure_ascii=False) + "\n")
                    manifest_rows.append(
                        {
                            "custom_id": custom_id,
                            "variant": variant,
                            "test_index": row_idx,
                            "gold_aspects": json.dumps(row["aspects"], ensure_ascii=False),
                            "text": str(row["text"]),
                            "model": model,
                            "approach": f"openai-{model}-{variant}",
                            "shots": len(demos),
                        }
                    )

    import pandas as pd

    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    metadata = {
        "data_path": str(resolve_data_path(data_path)),
        "prefix": prefix,
        "models": models,
        "variants": variants,
        "test_limit": int(test_limit),
        "seed": int(seed),
        "n_rows": int(len(df)),
        "n_test_rows_pre_limit": int(len(three_way_split(df, 0.10, 0.10, seed)[2])),
        "n_manifest_rows": int(len(manifest_rows)),
        "aspects": aspects,
        "batch_execution_policy": "batch_default_for_paper_facing_gpt_eval",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return request_path, manifest_path, metadata_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare batch-evaluable OpenAI ABSA prompt-baseline requests.")
    parser.add_argument("--data-path", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--prefix", default="openai_eval_batch_v1")
    parser.add_argument("--models", nargs="+", default=DEFAULT_BATCH_MODELS)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=SUPPORTED_BATCH_VARIANTS,
        choices=SUPPORTED_BATCH_VARIANTS,
        help="Batch-safe single-call prompt variants. Multi-stage variants are evaluated through the interactive pipeline, not this batch prep path.",
    )
    parser.add_argument("--test-limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    request_path, manifest_path, metadata_path = prepare_eval_requests(
        Path(args.data_path), args.prefix, args.models, args.variants, args.test_limit, args.seed
    )
    print(
        json.dumps(
            {
                "request_file": str(request_path),
                "manifest_file": str(manifest_path),
                "metadata_file": str(metadata_path),
                "models": args.models,
                "variants": args.variants,
                "test_limit": args.test_limit,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
